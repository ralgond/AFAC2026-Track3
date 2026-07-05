# -*- coding: utf-8 -*-
"""
DIN + DeepFM + SASRec (fused) for next-item / sequential recommendation.

This script imitates the structure of `din_model_2.py` (same data schema, same
encoders, same full-softmax-over-the-catalog training objective, same NDCG@10
eval loop and submission format) but replaces the single DIN branch with a
THREE-branch user encoder:

  1. DIN branch      -- the classic candidate-conditioned local activation-unit
                         attention over the (deduplicated) history. Re-weights
                         history items by "how relevant is this item to the
                         candidate", but has no notion of the ORDER the items
                         happened in, and needs a candidate to run.
  2. SASRec branch   -- a stack of causal self-attention blocks (Transformer
                         encoder, SASRec-style) over the history in the order
                         it occurred, with learned positional embeddings. The
                         representation at the most-recent position summarizes
                         the user's evolving interest and is candidate-free, so
                         it is complementary to the (order-blind) DIN branch.
  3. DeepFM branch   -- two small DeepFM towers (1st-order linear + 2nd-order
                         FM pairwise interactions + a deep MLP) that model
                         feature interactions among a user's own profile
                         fields (u_cat_01..08) and, separately, among an item's
                         own profile fields (i_cat_01..03, i_bucket_01). Both
                         towers are candidate-free so they can be evaluated
                         once per user / once per item.

All three branches are fused (together with the raw user embedding and the
candidate anchor embedding) by an MLP into a single `user_vec`, exactly the
same "candidate baked into one MLP forward, then dot-product against static
item vectors" trick `din_model_2.py` uses -- this is what makes scoring the
FULL catalog in one shot possible even though DIN/attention are normally
candidate-specific. The item side mirrors this: the DeepFM item tower is
folded into a candidate-free `item_static_vec` used both for the full-softmax
training loss and for full-catalog ranking at eval / prediction time.

Data schema
-----------
user.csv : uid,u_cat_01..u_cat_08                 (8 user categorical features, 0 is a VALID value)
item.csv : iid,i_cat_01,i_cat_02,i_cat_03,i_bucket_01   (item categorical features, 0 is a VALID value)
train.csv: uid,target_iid,item_seq_raw,item_seq_dedup,item_seq_counts
test.csv : uid,item_seq_raw,item_seq_dedup,item_seq_counts

item_seq_raw    : "i000001,i000002,..."            (full click history, may contain repeats)
item_seq_dedup  : same format, deduplicated
item_seq_counts : "i000001:18,i000002:17,..."       (item -> count in history)

Task
----
Given a user's historical item sequence (+ user/item side features), predict
the target item the user will interact with next. This is cast as a
candidate-ranking problem: the fused model scores (user, history, candidate)
triples. Training uses a FULL softmax loss over the entire item catalog
(cross-entropy against every item, no negative sampling) -- this matches the
candidate space used at evaluation time (ranking the full catalog for
NDCG@10) and avoids a train/eval distribution mismatch that sampled softmax
with a small random-negative pool would introduce.

Output
------
- Training prints valid NDCG@10 each epoch.
- run_predict produces submission.csv with columns:
    uid,prediction
  where prediction is a comma-quoted string of top-10 item ids, e.g.:
    u000009,"i001952,i001038,i001710,i001046,i000401,i001445,i001069,i001002,i001673,i000661"

item_seq_counts is used as a per-history-item frequency weight, exactly as in
`din_model_2.py`: we feed log1p(count) into the DIN activation unit as an
extra interaction signal, and also use it to scale the attention-pooled
interest vector, so items a user has repeatedly interacted with carry more
weight than items seen only once. We switch to the deduplicated sequence
(item_seq_dedup) as the set of distinct history items, paired with their
counts, and keep it in first-seen order so the SASRec branch has a real
(if coarse) notion of "which distinct item came before which").

Usage
-----
python din_deepfm_sasrec_model.py --data_dir /path/to/data --out_dir /path/to/output
All configuration lives in the `Config` class below.
"""

import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

PAD_IDX = 0  # index 0 in EMBEDDING tables is reserved for padding / unknown


# ========================================================================================
# 0. Config
# ========================================================================================

@dataclass
class Config:
    """All hyperparameters / paths / switches for this script live here.
    Construct with defaults via Config()."""

    # paths
    data_dir = "../data/A2-Rec"
    out_dir = "./"

    # data / sequence handling
    max_seq_len = 50          # truncate/pad history to this many distinct items (most recent kept)
    val_frac = 0.1          # fraction of train.csv randomly held out as the valid set
    use_synthetic = False    # generate synthetic data into data_dir if real files are missing

    # shared embedding sizes
    emb_dim = 32
    side_emb_ratio = 0.5    # side-feature embedding dim = emb_dim * side_emb_ratio

    # DIN branch
    attn_hidden = (80, 40)  # activation-unit MLP hidden sizes

    # SASRec branch
    sasrec_blocks = 2       # number of self-attention blocks
    sasrec_heads = 2        # attention heads per block (item_full_dim must be divisible by this)
    sasrec_dropout = 0.2

    # DeepFM branch (used once for the user-field tower, once for the item-field tower)
    deepfm_hidden = (64, 32)

    # fusion MLP (combines DIN + SASRec + DeepFM + raw embeddings -> final user vector)
    mlp_hidden = (200, 80)
    dropout = 0.2

    # training
    epochs = 20
    batch_size = 512
    lr = 1e-3
    grad_clip = 5.0
    seed = 42

    # prediction
    topk = 10


# ========================================================================================
# 1. Vocab / encoding utilities  (identical to din_model_2.py)
# ========================================================================================

class CategoryEncoder:
    """Maps raw categorical values (which may legitimately be 0) to dense indices
    starting at 1, reserving 0 for PAD/UNK. This is essential because the raw '0'
    value is a valid category and must NOT collide with the padding index."""

    def __init__(self):
        self.value2idx = {}
        self.n = 1  # 0 reserved for pad/unk

    def fit(self, values):
        for v in values:
            v = str(v)
            if v not in self.value2idx:
                self.value2idx[v] = self.n
                self.n += 1
        return self

    def transform(self, values):
        return np.array([self.value2idx.get(str(v), 0) for v in values], dtype=np.int64)

    def __len__(self):
        return self.n  # vocab size including pad/unk at 0


class IdEncoder:
    """Encoder specifically for uid / iid strings -> contiguous int ids (1..N), 0 = pad/unk."""

    def __init__(self):
        self.value2idx = {}
        self.idx2value = {0: "<PAD>"}
        self.n = 1

    def fit(self, values):
        for v in values:
            if v not in self.value2idx:
                self.value2idx[v] = self.n
                self.idx2value[self.n] = v
                self.n += 1
        return self

    def transform_one(self, v):
        return self.value2idx.get(v, 0)

    def transform(self, values):
        return np.array([self.transform_one(v) for v in values], dtype=np.int64)

    def __len__(self):
        return self.n


def parse_seq_raw(s):
    if not isinstance(s, str) or s == "":
        return []
    return s.split(",")


def parse_seq_counts(s):
    """Parses 'i000001:18,i000002:17,...' -> dict[str, int]."""
    if not isinstance(s, str) or s == "":
        return {}
    out = {}
    for tok in s.split(","):
        if ":" not in tok:
            continue
        iid, cnt = tok.rsplit(":", 1)
        try:
            out[iid] = int(cnt)
        except ValueError:
            continue
    return out


# ========================================================================================
# 2. Data loading  (identical to din_model_2.py)
# ========================================================================================

class DataBundle:
    """Holds encoders, feature tables, and raw frames needed by the dataset / model."""
    pass


def load_data(cfg: "Config"):
    data_dir = cfg.data_dir
    user_df = pd.read_csv(os.path.join(data_dir, "user.csv"), dtype=str)
    item_df = pd.read_csv(os.path.join(data_dir, "item.csv"), dtype=str)
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), dtype=str)
    test_path = os.path.join(data_dir, "test.csv")
    test_df = pd.read_csv(test_path, dtype=str) if os.path.exists(test_path) else None

    user_df = user_df.fillna("0")
    item_df = item_df.fillna("0")
    train_df = train_df.fillna("")
    if test_df is not None:
        test_df = test_df.fillna("")

    bundle = DataBundle()

    # ---- item id encoder (drives the candidate / softmax space) ----
    item_ids = item_df["iid"].tolist()
    iid_enc = IdEncoder()
    iid_enc.fit(item_ids)
    bundle.iid_enc = iid_enc
    bundle.n_items = len(iid_enc)  # includes pad/unk at 0

    # ---- user id encoder ----
    uid_enc = IdEncoder()
    uid_enc.fit(user_df["uid"].tolist())
    bundle.uid_enc = uid_enc
    bundle.n_users = len(uid_enc)

    # ---- user categorical features ----
    u_cat_cols = [c for c in user_df.columns if c.startswith("u_cat_")]
    bundle.u_cat_cols = u_cat_cols
    u_encoders = {}
    for c in u_cat_cols:
        enc = CategoryEncoder().fit(user_df[c].tolist())
        u_encoders[c] = enc
    bundle.u_encoders = u_encoders

    # build user feature matrix indexed by encoded uid (0 = pad row, all zeros)
    n_u_cat = len(u_cat_cols)
    user_feat = np.zeros((bundle.n_users, n_u_cat), dtype=np.int64)
    uidx_all = uid_enc.transform(user_df["uid"].tolist())
    for j, c in enumerate(u_cat_cols):
        user_feat[uidx_all, j] = u_encoders[c].transform(user_df[c].tolist())
    bundle.user_feat = user_feat  # [n_users, n_u_cat]

    # ---- item categorical features (includes the bucket feature) ----
    i_cat_cols = [c for c in item_df.columns if c.startswith("i_cat_") or c.startswith("i_bucket_")]
    bundle.i_cat_cols = i_cat_cols
    i_encoders = {}
    for c in i_cat_cols:
        enc = CategoryEncoder().fit(item_df[c].tolist())
        i_encoders[c] = enc
    bundle.i_encoders = i_encoders

    n_i_cat = len(i_cat_cols)
    item_feat = np.zeros((bundle.n_items, n_i_cat), dtype=np.int64)
    iidx_all = iid_enc.transform(item_df["iid"].tolist())
    for j, c in enumerate(i_cat_cols):
        item_feat[iidx_all, j] = i_encoders[c].transform(item_df[c].tolist())
    bundle.item_feat = item_feat  # [n_items, n_i_cat]

    bundle.train_df = train_df
    bundle.test_df = test_df
    bundle.user_df = user_df
    bundle.item_df = item_df

    return bundle


# ========================================================================================
# 3. Dataset  (identical to din_model_2.py)
# ========================================================================================

class SeqDataset(Dataset):
    """
    Each sample: (uid_idx, hist_item_idx[seq_len], hist_count[seq_len], hist_len, target_item_idx)

    History is built from `item_seq_dedup` (distinct items, in first-seen order)
    paired with their frequency from `item_seq_counts`. The count signal feeds
    the DIN activation unit (as a log-count feature) and reweights the
    attention-pooled interest; the first-seen ORDER of `item_seq_dedup` is what
    the SASRec branch's positional embeddings key off of.

    Side features (user_feat / item_feat) are looked up inside the model via
    embedding tables indexed by uid_idx / item_idx, so the Dataset only needs ids.
    """

    def __init__(self, df, uid_enc, iid_enc, max_len, has_target=True):
        self.uids = df["uid"].tolist()
        self.dedup_seqs = df["item_seq_dedup"].tolist()
        self.count_strs = df["item_seq_counts"].tolist()
        self.has_target = has_target
        if has_target:
            self.targets = df["target_iid"].tolist()
        self.uid_enc = uid_enc
        self.iid_enc = iid_enc
        self.max_len = max_len

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid_idx = self.uid_enc.transform_one(self.uids[idx])
        items = parse_seq_raw(self.dedup_seqs[idx])
        counts_map = parse_seq_counts(self.count_strs[idx])

        # item_seq_dedup is in first-seen order; keep the most RECENT max_len distinct
        # items (i.e. the tail of the dedup list).
        items = items[-self.max_len:]
        counts = [counts_map.get(i, 1) for i in items]  # default count=1 if missing (defensive)

        hist_idx = [self.iid_enc.transform_one(i) for i in items]
        hist_len = len(hist_idx)

        pad_n = self.max_len - hist_len
        hist_idx = [PAD_IDX] * pad_n + hist_idx
        hist_cnt = [0] * pad_n + counts  # 0 count on padding positions -> contributes no weight

        sample = {
            "uid_idx": uid_idx,
            "hist": np.array(hist_idx, dtype=np.int64),
            "hist_cnt": np.array(hist_cnt, dtype=np.float32),
            "hist_len": hist_len,
        }
        if self.has_target:
            sample["target"] = self.iid_enc.transform_one(self.targets[idx])
        return sample


def collate_fn(batch):
    uid_idx = torch.tensor([b["uid_idx"] for b in batch], dtype=torch.long)
    hist = torch.tensor(np.stack([b["hist"] for b in batch]), dtype=torch.long)
    hist_cnt = torch.tensor(np.stack([b["hist_cnt"] for b in batch]), dtype=torch.float32)
    hist_len = torch.tensor([b["hist_len"] for b in batch], dtype=torch.long)
    out = {"uid_idx": uid_idx, "hist": hist, "hist_cnt": hist_cnt, "hist_len": hist_len}
    if "target" in batch[0]:
        out["target"] = torch.tensor([b["target"] for b in batch], dtype=torch.long)
    return out


# ========================================================================================
# 4. Branch 1 -- DIN activation unit  (identical to din_model_2.py)
# ========================================================================================

class ActivationUnit(nn.Module):
    """DIN's local activation unit: computes attention weight between a candidate
    item embedding and each historical item embedding, using the classic
    [hist, cand, hist-cand, hist*cand] interaction features fed through an MLP,
    PLUS an extra scalar feature: log1p(item_seq_counts) for that history item."""

    def __init__(self, emb_dim, hidden=(80, 40)):
        super().__init__()
        layers = []
        in_dim = emb_dim * 4 + 1  # +1 for the log-count feature
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.PReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hist_emb, cand_emb, mask, hist_cnt):
        # hist_emb: [B, L, D], cand_emb: [B, D], mask: [B, L], hist_cnt: [B, L] (raw counts, 0 on pad)
        L = hist_emb.size(1)
        cand_exp = cand_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        log_cnt = torch.log1p(hist_cnt).unsqueeze(-1)  # [B, L, 1]
        feat = torch.cat([hist_emb, cand_exp, hist_emb - cand_exp, hist_emb * cand_exp, log_cnt], dim=-1)
        scores = self.mlp(feat).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)  # [B, L]
        weights = weights.masked_fill(mask == 0, 0.0)
        return weights


# ========================================================================================
# 5. Branch 2 -- SASRec self-attentive sequence encoder
# ========================================================================================

class SASRecBlock(nn.Module):
    """One pre-norm Transformer block: causal self-attention + position-wise FFN,
    both with residual connections, following the standard SASRec block design."""

    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, key_padding_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask,
                                 key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.ln2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class SASRecEncoder(nn.Module):
    """Self-attentive sequential encoder (SASRec). Encodes the user's history
    sequence with learned positional embeddings and stacked CAUSAL self-attention
    blocks, so the representation at the last (= most recent) valid position
    summarizes the sequence in the order it happened -- complementary to DIN's
    candidate-conditioned attention, which re-weights history items but is
    blind to their relative order, and needs a candidate to even run.

    History in this pipeline is LEFT-padded (pad tokens first, real items last)
    so that `hist[:, -1]` is always the most recent real item -- this is the
    convention the DIN branch and the training loop's attention anchor rely on.
    A causal mask combined directly with left-padding is a correctness trap:
    every padding query position can (by causality) only attend to earlier
    positions, which are ALSO all padding, so its entire attention row is
    masked -> softmax over an all -inf row -> NaN. PyTorch's fast attention
    path (used whenever this runs under `torch.no_grad()`, i.e. every eval /
    predict call) does NOT guard against this and returns NaN outright, which
    then contaminates every other position in the batch once those NaN values
    are used as keys/values elsewhere (a masked attention weight of 0 times a
    NaN value is still NaN). This silently produces garbage user vectors at
    eval time while training (which happens to hit a different, safe kernel
    because gradients are required) looks completely normal -- exactly the
    "train loss goes down, NDCG is flat/garbage" symptom this caused.

    Fix: internally re-express the sequence as RIGHT-padded (real items moved
    to the front, in the same relative order) via a per-row cyclic gather
    before running causal attention, so no query position's causal receptive
    field is ever 100% padding. We then gather the output at each row's own
    last REAL position (hist_len - 1) rather than assuming a fixed index."""

    def __init__(self, item_full_dim, max_len, n_blocks=2, n_heads=2, dropout=0.2):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, item_full_dim)
        self.in_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([SASRecBlock(item_full_dim, n_heads, dropout) for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(item_full_dim)
        self.max_len = max_len

    def forward(self, hist_full, mask):
        # hist_full: [B, L, D], mask: [B, L] (1 = valid item, 0 = pad, left-padded)
        B, L, D = hist_full.shape
        hist_len = mask.sum(dim=1)               # [B] number of real items
        pad_n = L - hist_len                      # [B] amount of left padding

        # cyclic left-shift by pad_n turns "pad_n pads then hist_len real items"
        # into "hist_len real items then pad_n pads", preserving the real items'
        # relative (chronological) order -- exactly a left-pad -> right-pad
        # conversion, done with a single gather (no python-level loop).
        shift_idx = (torch.arange(L, device=hist_full.device).unsqueeze(0) + pad_n.unsqueeze(1)) % L  # [B, L]
        gather_idx = shift_idx.unsqueeze(-1).expand(-1, -1, D)
        x_right = torch.gather(hist_full, 1, gather_idx)      # [B, L, D], real items now at the FRONT
        mask_right = torch.gather(mask, 1, shift_idx)          # [B, L]

        pos_ids = torch.arange(L, device=hist_full.device).unsqueeze(0).expand(B, -1)
        x = x_right + self.pos_emb(pos_ids)
        x = self.in_dropout(x)
        mask_f = mask_right.unsqueeze(-1).float()
        x = x * mask_f  # zero out padding positions before they enter attention

        # causal mask: query position i may only attend to keys at positions <= i.
        # With right-padding, a real query at the front can always see at least
        # itself (a real key), so no real position's row is ever fully masked.
        causal = torch.triu(torch.ones(L, L, device=hist_full.device, dtype=torch.bool), diagonal=1)
        key_pad = (mask_right == 0)  # [B, L] True where padding -> never attend to these keys

        # the only remaining fully-masked-row case is a genuinely EMPTY history
        # (hist_len == 0): unmask such rows (attention over pure padding is
        # harmless busywork; the output is discarded via mask_f / the gather below).
        fully_pad = key_pad.all(dim=1)
        if fully_pad.any():
            key_pad = key_pad.clone()
            key_pad[fully_pad, :] = False

        for blk in self.blocks:
            x = blk(x, causal_mask=causal, key_padding_mask=key_pad)
            x = x * mask_f

        x = self.ln_out(x)  # [B, L, D], real items occupy positions [0, hist_len)

        # pull out each row's own last REAL position instead of assuming index -1
        # (which, after the right-pad conversion, is generally a padding slot).
        last_pos = (hist_len - 1).clamp(min=0)  # [B]
        seq_vec = x[torch.arange(B, device=x.device), last_pos, :]  # [B, D]
        return seq_vec


# ========================================================================================
# 6. Branch 3 -- DeepFM tower (1st-order + 2nd-order FM + deep MLP over a field set)
# ========================================================================================

class FMDeepTower(nn.Module):
    """Generic DeepFM-style tower over a fixed set of categorical fields: one "id"
    field (uid or iid) plus K "side" fields (u_cat_* or i_cat_*/i_bucket_*).
    Reuses the raw embeddings the caller already computed elsewhere (no duplicate
    embedding tables for the deep/2nd-order path) and produces:
      - `fm_score`  : scalar per row = 1st-order linear term + 2nd-order pairwise
                      FM interaction term (the classic factorization-machine sum).
      - `deep_vec`  : a `deep_dim`-sized vector from an MLP over the concatenated
                      raw field embeddings (the "deep" half of DeepFM).
    Both are candidate-free (they only look at one entity's own fields), so this
    tower can be evaluated once per user / once per item and reused across the
    whole catalog, which is what keeps full-catalog scoring cheap."""

    def __init__(self, id_vocab, side_vocabs, emb_dim, side_dim, fm_dim=None, deep_hidden=(64, 32), dropout=0.1):
        super().__init__()
        fm_dim = fm_dim or emb_dim

        # 1st-order (linear) weights: one learned scalar per field value.
        self.fo_id = nn.Embedding(id_vocab, 1, padding_idx=PAD_IDX)
        self.fo_side = nn.ModuleList([nn.Embedding(v, 1, padding_idx=PAD_IDX) for v in side_vocabs])

        # project each field's raw embedding into a shared fm_dim so pairwise
        # dot products (2nd-order FM interactions) are well-defined across fields
        # of different native widths (id field is emb_dim, side fields are side_dim).
        self.fm_proj_id = nn.Linear(emb_dim, fm_dim, bias=False)
        self.fm_proj_side = nn.ModuleList([nn.Linear(side_dim, fm_dim, bias=False) for _ in side_vocabs])

        deep_in = emb_dim + side_dim * len(side_vocabs)
        layers, in_dim = [], deep_in
        for h in deep_hidden:
            layers += [nn.Linear(in_dim, h), nn.PReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.deep_mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.deep_dim = in_dim

    def forward(self, id_idx, side_idx_list, id_emb, side_embs):
        """
        id_idx        : [...]            long ids for the id field (first-order lookup)
        side_idx_list : list of [...]    long ids, one per side field (first-order lookup)
        id_emb        : [..., emb_dim]   already-computed raw id embedding (reused, not recomputed)
        side_embs     : list of [..., side_dim] raw side embeddings (reused, not recomputed)
        returns: fm_score [...], deep_vec [..., deep_dim]
        """
        # ---- 1st order ----
        fo = self.fo_id(id_idx).squeeze(-1)
        for emb, idx in zip(self.fo_side, side_idx_list):
            fo = fo + emb(idx).squeeze(-1)

        # ---- 2nd order pairwise FM: 0.5 * (sum(v)^2 - sum(v^2)) summed over fm_dim ----
        fields = [self.fm_proj_id(id_emb)] + [proj(s) for proj, s in zip(self.fm_proj_side, side_embs)]
        stacked = torch.stack(fields, dim=-2)          # [..., n_fields, fm_dim]
        sum_then_sq = stacked.sum(dim=-2).pow(2)         # [..., fm_dim]
        sq_then_sum = stacked.pow(2).sum(dim=-2)         # [..., fm_dim]
        second_order = 0.5 * (sum_then_sq - sq_then_sum).sum(dim=-1)  # [...]
        fm_score = fo + second_order

        # ---- deep ----
        deep_in = torch.cat([id_emb] + list(side_embs), dim=-1)
        deep_vec = self.deep_mlp(deep_in)
        return fm_score, deep_vec


# ========================================================================================
# 7. Fused model: DIN + SASRec + DeepFM
# ========================================================================================

class DINDeepFMSASRec(nn.Module):
    """Encodes (user, history, candidate) into a single user_vec by fusing three
    complementary signals plus the raw embeddings, and exposes a candidate-free
    `item_static_vec` (also DeepFM-enriched) for full-catalog scoring."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        emb_dim = cfg.emb_dim
        self.emb_dim = emb_dim
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))

        # ---- core id embeddings (shared raw material for all three branches) ----
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=PAD_IDX)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=PAD_IDX)  # used as history, candidate & catalog emb
        self.user_side_embs = nn.ModuleList([nn.Embedding(v, side_dim, padding_idx=PAD_IDX) for v in user_feat_vocabs])
        self.item_side_embs = nn.ModuleList([nn.Embedding(v, side_dim, padding_idx=PAD_IDX) for v in item_feat_vocabs])

        item_full_dim = emb_dim + side_dim * len(item_feat_vocabs)
        user_full_dim = emb_dim + side_dim * len(user_feat_vocabs)
        self.item_full_dim = item_full_dim
        self.user_full_dim = user_full_dim

        # item_full_dim must divide evenly by sasrec_heads for nn.MultiheadAttention.
        n_heads = cfg.sasrec_heads
        while item_full_dim % n_heads != 0 and n_heads > 1:
            n_heads -= 1

        # ---- branch 1: DIN activation-unit attention, candidate-conditioned, order-blind ----
        self.activation_unit = ActivationUnit(item_full_dim, hidden=cfg.attn_hidden)

        # ---- branch 2: SASRec causal self-attention over the ordered history, candidate-free ----
        self.sasrec = SASRecEncoder(item_full_dim, max_len=cfg.max_seq_len, n_blocks=cfg.sasrec_blocks,
                                     n_heads=n_heads, dropout=cfg.sasrec_dropout)

        # ---- branch 3: DeepFM towers over each entity's own fields, candidate-free ----
        self.user_fm_deep = FMDeepTower(n_users, user_feat_vocabs, emb_dim, side_dim,
                                         deep_hidden=cfg.deepfm_hidden, dropout=cfg.dropout)
        self.item_fm_deep = FMDeepTower(n_items, item_feat_vocabs, emb_dim, side_dim,
                                         deep_hidden=cfg.deepfm_hidden, dropout=cfg.dropout)

        # ---- fusion MLP: [raw user, user-DeepFM, DIN interest, SASRec state, candidate] -> user_vec ----
        fuse_in = (user_full_dim
                   + self.user_fm_deep.deep_dim + 1   # DeepFM user-field deep vec + fm scalar
                   + item_full_dim                     # DIN attention-pooled interest
                   + item_full_dim                     # SASRec last-position sequence state
                   + item_full_dim)                    # candidate anchor embedding
        h1, h2 = cfg.mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(fuse_in, h1), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h1, h2), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h2, emb_dim),  # project to emb_dim so we can score vs. full item catalog via dot product
        )

        # ---- candidate-free item tower for full-catalog scoring (DeepFM-enriched) ----
        item_tower_in = item_full_dim + self.item_fm_deep.deep_dim + 1
        self.item_score_head = nn.Sequential(
            nn.Linear(item_tower_in, emb_dim), nn.PReLU(), nn.Linear(emb_dim, emb_dim),
        )

        # registered buffers filled in by set_feature_tables()
        self.register_buffer("user_feat_table", torch.zeros(n_users, max(1, len(user_feat_vocabs)), dtype=torch.long),
                              persistent=False)
        self.register_buffer("item_feat_table", torch.zeros(n_items, max(1, len(item_feat_vocabs)), dtype=torch.long),
                              persistent=False)

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.user_feat_table = user_feat_table
        self.item_feat_table = item_feat_table

    # ---- side-feature helpers: return both raw indices (for FM 1st-order) and embeddings ----
    def _item_side(self, item_idx):
        feats = self.item_feat_table[item_idx]  # [..., n_i_cat]
        idx_list = [feats[..., j] for j in range(feats.shape[-1])]
        emb_list = [emb(idx_list[j]) for j, emb in enumerate(self.item_side_embs)]
        return idx_list, emb_list

    def _user_side(self, uid_idx):
        feats = self.user_feat_table[uid_idx]  # [..., n_u_cat]
        idx_list = [feats[..., j] for j in range(feats.shape[-1])]
        emb_list = [emb(idx_list[j]) for j, emb in enumerate(self.user_side_embs)]
        return idx_list, emb_list

    def _item_full_emb(self, item_idx):
        base = self.item_emb(item_idx)
        _, side_parts = self._item_side(item_idx)
        return torch.cat([base] + side_parts, dim=-1)

    def item_static_vec(self, item_idx):
        """Candidate-free item representation used for full-catalog scoring: raw
        item embedding fused with the item-field DeepFM tower's fm score + deep vec."""
        base = self.item_emb(item_idx)
        idx_list, emb_list = self._item_side(item_idx)
        item_full = torch.cat([base] + emb_list, dim=-1)
        fm_score, deep_vec = self.item_fm_deep(item_idx, idx_list, base, emb_list)
        tower_in = torch.cat([item_full, deep_vec, fm_score.unsqueeze(-1)], dim=-1)
        return self.item_score_head(tower_in)

    def encode_user(self, uid_idx, hist, hist_cnt, hist_len, cand_item_idx):
        """Produces the fused user representation vector for scoring against item
        embeddings. cand_item_idx is used to drive the DIN attention AND is
        concatenated directly into the fusion MLP (same "candidate anchor" trick
        `din_model_2.py` uses), while the SASRec / DeepFM(user) branches are
        candidate-free."""
        mask = (hist != PAD_IDX).long()  # [B, L]
        hist_full = self._item_full_emb(hist)          # [B, L, item_full_dim]
        cand_full = self._item_full_emb(cand_item_idx)  # [B, item_full_dim]

        # ---- branch 1: DIN candidate-conditioned attention pooling ----
        attn_w = self.activation_unit(hist_full, cand_full, mask, hist_cnt)  # [B, L]
        freq_w = torch.log1p(hist_cnt) * mask.float()
        combined_w = attn_w * (1.0 + freq_w)
        combined_w = combined_w / (combined_w.sum(dim=-1, keepdim=True) + 1e-9)
        din_interest = torch.bmm(combined_w.unsqueeze(1), hist_full).squeeze(1)  # [B, item_full_dim]

        # ---- branch 2: SASRec order-aware self-attention over history ----
        seq_vec = self.sasrec(hist_full, mask)   # [B, item_full_dim], state at the last REAL position

        # ---- branch 3: DeepFM over the user's own profile fields ----
        user_base = self.user_emb(uid_idx)
        u_idx_list, u_emb_list = self._user_side(uid_idx)
        user_full = torch.cat([user_base] + u_emb_list, dim=-1)
        u_fm_score, u_deep_vec = self.user_fm_deep(uid_idx, u_idx_list, user_base, u_emb_list)

        # ---- fuse everything + the candidate anchor ----
        x = torch.cat([user_full, u_deep_vec, u_fm_score.unsqueeze(-1), din_interest, seq_vec, cand_full], dim=-1)
        user_vec = self.mlp(x)
        return user_vec


class FusionRanker(nn.Module):
    """Thin wrapper mirroring din_model_2.py's DINRanker: exposes forward() ->
    user_vec, item_static_vec() for full-catalog vectors, and dot-product scoring."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        self.core = DINDeepFMSASRec(n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg)
        self.n_items = n_items

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.core.set_feature_tables(user_feat_table, item_feat_table)

    def item_static_vec(self, item_idx):
        return self.core.item_static_vec(item_idx)

    def forward(self, uid_idx, hist, hist_cnt, hist_len, target_idx):
        return self.core.encode_user(uid_idx, hist, hist_cnt, hist_len, target_idx)

    def score_against_catalog(self, user_vec, item_vecs):
        # user_vec: [B, D], item_vecs: [N, D] -> [B, N]
        return user_vec @ item_vecs.t()


# ========================================================================================
# 8. Training: full softmax over the entire item catalog  (identical objective to din_model_2.py)
# ========================================================================================

def full_softmax_loss(user_vec, target_idx, all_item_vecs):
    """
    user_vec: [B, D]
    target_idx: [B]  (true next-item index for each sample)
    all_item_vecs: [n_items, D]  static item vectors for the ENTIRE catalog (index 0 = PAD)

    Computes logits against every item in the catalog (no negative sampling),
    masks out the PAD index, and applies standard cross-entropy with the true
    target as the label -- matching the full candidate space used at eval time.
    """
    logits = user_vec @ all_item_vecs.t()  # [B, n_items]
    logits = logits.clone()
    logits[:, PAD_IDX] = -1e9  # never let the model assign probability mass to PAD
    loss = F.cross_entropy(logits, target_idx)
    return loss


# ========================================================================================
# 9. NDCG@10 metric  (identical to din_model_2.py)
# ========================================================================================

def ndcg_at_k(ranked_item_ids, true_item_id, k=10):
    """Single-relevant-item NDCG@k: 1/log2(rank+1) if hit within top-k else 0."""
    try:
        pos = ranked_item_ids[:k].index(true_item_id)
        return 1.0 / math.log2(pos + 2)
    except ValueError:
        return 0.0


@torch.no_grad()
def evaluate_ndcg(model, loader, item_vecs_all, k=10, device="cpu"):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        uid_idx = batch["uid_idx"].to(device)
        hist = batch["hist"].to(device)
        hist_cnt = batch["hist_cnt"].to(device)
        hist_len = batch["hist_len"].to(device)
        target = batch["target"].to(device)

        # at eval time we don't know the true candidate in advance for the DIN
        # attention anchor (using it would leak the label), so we drive it with
        # the most recent history item -- the same convention used at train time
        # for the anchor position (see train_model below).
        batch_idx = torch.arange(hist.size(0), device=device)
        last_valid_pos = hist.size(1) - 1
        last_item = hist[batch_idx, last_valid_pos]

        user_vec = model(uid_idx, hist, hist_cnt, hist_len, last_item)  # [B, D]
        scores = model.score_against_catalog(user_vec, item_vecs_all)  # [B, n_items]
        scores[:, PAD_IDX] = -1e9  # never recommend pad

        topk = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
        target_np = target.cpu().numpy()
        for row, t in zip(topk, target_np):
            total += ndcg_at_k(row.tolist(), int(t), k=k)
            n += 1
    model.train()
    return total / max(n, 1)


# ========================================================================================
# 10. Training loop
# ========================================================================================

def train_model(bundle, cfg: "Config", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    train_full = bundle.train_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    n_val = int(len(train_full) * cfg.val_frac)
    val_df = train_full.iloc[:n_val].reset_index(drop=True)
    tr_df = train_full.iloc[n_val:].reset_index(drop=True)
    print(f"[INFO] train={len(tr_df)}  valid={len(val_df)}")

    train_ds = SeqDataset(tr_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=True)
    val_ds = SeqDataset(val_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    user_feat_vocabs = [len(bundle.u_encoders[c]) for c in bundle.u_cat_cols]
    item_feat_vocabs = [len(bundle.i_encoders[c]) for c in bundle.i_cat_cols]

    model = FusionRanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
    model.set_feature_tables(
        torch.tensor(bundle.user_feat, dtype=torch.long),
        torch.tensor(bundle.item_feat, dtype=torch.long),
    )
    model.to(device)
    model.core.user_feat_table = model.core.user_feat_table.to(device)
    model.core.item_feat_table = model.core.item_feat_table.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    all_item_ids = torch.arange(bundle.n_items, device=device)

    best_ndcg = -1.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "din_deepfm_sasrec_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            uid_idx = batch["uid_idx"].to(device)
            hist = batch["hist"].to(device)
            hist_cnt = batch["hist_cnt"].to(device)
            hist_len = batch["hist_len"].to(device)
            target = batch["target"].to(device)

            # IMPORTANT: drive the DIN attention unit with the most-recent HISTORY
            # item, not the true target -- using the target would leak label
            # information into attention during training while evaluation (which
            # cannot know the label) has to fall back to the last history item.
            # Using the same "last history item as attention anchor" convention at
            # both train and eval time means the model learns the attention
            # pattern it will actually be evaluated with.
            batch_idx_ = torch.arange(hist.size(0), device=device)
            last_valid_pos_ = hist.size(1) - 1
            attn_cand = hist[batch_idx_, last_valid_pos_]

            user_vec = model(uid_idx, hist, hist_cnt, hist_len, attn_cand)

            # recompute the full-catalog item vectors every step (item_score_head's
            # weights change each update, so these can't be cached across steps).
            all_item_vecs = model.item_static_vec(all_item_ids)  # [n_items, D]
            loss = full_softmax_loss(user_vec, target, all_item_vecs)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # full catalog item vectors for eval-time scoring
        with torch.no_grad():
            item_vecs_all = model.item_static_vec(all_item_ids)  # [n_items, D]
        val_ndcg = evaluate_ndcg(model, val_loader, item_vecs_all, k=cfg.topk, device=device)

        print(f"[Epoch {epoch}/{cfg.epochs}] train_loss={avg_loss:.4f}  valid_ndcg@{cfg.topk}={val_ndcg:.4f}")

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best model saved (ndcg@{cfg.topk}={best_ndcg:.4f})")

    print(f"[INFO] training done. best valid ndcg@{cfg.topk} = {best_ndcg:.4f}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, device


# ========================================================================================
# 11. Prediction
# ========================================================================================

@torch.no_grad()
def run_predict(model, bundle, cfg: "Config", device):
    test_df = bundle.test_df
    if test_df is None:
        print("[WARN] no test.csv found, skipping prediction.")
        return

    model.eval()
    ds = SeqDataset(test_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    all_item_ids = torch.arange(bundle.n_items, device=device)
    item_vecs_all = model.item_static_vec(all_item_ids)

    rows = []
    uids = test_df["uid"].tolist()
    for batch in loader:
        uid_idx = batch["uid_idx"].to(device)
        hist = batch["hist"].to(device)
        hist_cnt = batch["hist_cnt"].to(device)
        hist_len = batch["hist_len"].to(device)

        last_valid_pos = hist.size(1) - 1
        batch_idx = torch.arange(hist.size(0), device=device)
        last_item = hist[batch_idx, last_valid_pos]

        user_vec = model(uid_idx, hist, hist_cnt, hist_len, last_item)
        scores = model.score_against_catalog(user_vec, item_vecs_all)
        scores[:, PAD_IDX] = -1e9

        # mask out PAD only; (optionally could exclude already-seen items, kept simple/general here)
        topk = torch.topk(scores, k=cfg.topk, dim=1).indices.cpu().numpy()

        for row in topk:
            item_strs = [bundle.iid_enc.idx2value.get(int(i), "i000000") for i in row]
            rows.append(",".join(item_strs))

    out_df = pd.DataFrame({"uid": uids, "prediction": rows})
    out_path = os.path.join(cfg.out_dir, "submission.csv")
    # QUOTE_MINIMAL: pandas/csv automatically quotes only fields containing commas
    # (i.e. 'prediction'), leaving 'uid' unquoted -- matches the spec's example.
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] predictions written to {out_path}")
    return out_path


# ========================================================================================
# 12. Synthetic data generator (for local smoke-testing when no real data is uploaded)
# ========================================================================================

def make_synthetic_data(data_dir, n_users=500, n_items=300, n_train=3000, n_test=200):
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    uids = [f"u{str(i).zfill(6)}" for i in range(1, n_users + 1)]
    iids = [f"i{str(i).zfill(6)}" for i in range(1, n_items + 1)]

    user_df = pd.DataFrame({"uid": uids})
    for c in [f"u_cat_{str(i).zfill(2)}" for i in range(1, 9)]:
        user_df[c] = rng.integers(0, 20, size=n_users)  # includes 0 as valid value
    user_df.to_csv(os.path.join(data_dir, "user.csv"), index=False)

    item_df = pd.DataFrame({"iid": iids})
    item_df["i_cat_01"] = rng.integers(0, 50, size=n_items)
    item_df["i_cat_02"] = rng.integers(0, 30, size=n_items)
    item_df["i_cat_03"] = rng.integers(0, 10, size=n_items)
    item_df["i_bucket_01"] = rng.integers(0, 5, size=n_items)
    item_df.to_csv(os.path.join(data_dir, "item.csv"), index=False)

    def gen_seq_row(uid):
        seq_len = rng.integers(5, 30)
        seq = rng.choice(iids, size=seq_len, replace=True).tolist()
        raw = ",".join(seq)
        dedup_items = list(dict.fromkeys(seq))
        dedup = ",".join(dedup_items)
        counts = defaultdict(int)
        for it in seq:
            counts[it] += 1
        counts_str = ",".join(f"{k}:{v}" for k, v in counts.items())
        target = rng.choice(iids)
        return raw, dedup, counts_str, target

    train_rows = []
    for _ in range(n_train):
        uid = rng.choice(uids)
        raw, dedup, counts_str, target = gen_seq_row(uid)
        train_rows.append([uid, target, raw, dedup, counts_str])
    train_df = pd.DataFrame(train_rows, columns=["uid", "target_iid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"])
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)

    test_rows = []
    for _ in range(n_test):
        uid = rng.choice(uids)
        raw, dedup, counts_str, _ = gen_seq_row(uid)
        test_rows.append([uid, raw, dedup, counts_str])
    test_df = pd.DataFrame(test_rows, columns=["uid", "item_seq_raw", "item_seq_dedup", "item_seq_counts"])
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

    print(f"[INFO] synthetic data written to {data_dir}")


# ========================================================================================
# 13. Main
# ========================================================================================

def main():
    cfg = Config()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    required = ["user.csv", "item.csv", "train.csv"]
    missing = [f for f in required if not os.path.exists(os.path.join(cfg.data_dir, f))]
    if missing:
        if cfg.use_synthetic:
            print(f"[WARN] missing {missing}; generating synthetic data instead.")
            make_synthetic_data(cfg.data_dir)
        else:
            raise FileNotFoundError(
                f"Missing required files {missing} in {cfg.data_dir}. "
                f"Set cfg.use_synthetic = True to test the pipeline on fake data."
            )

    bundle = load_data(cfg)
    print(f"[INFO] n_users={bundle.n_users}  n_items={bundle.n_items}  "
          f"u_cat_cols={bundle.u_cat_cols}  i_cat_cols={bundle.i_cat_cols}")

    model, device = train_model(bundle, cfg)

    run_predict(model, bundle, cfg, device)


if __name__ == "__main__":
    main()
