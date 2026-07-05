# -*- coding: utf-8 -*-
"""
SASRecF (Self-Attentive Sequential Recommendation with Features) for
next-item / sequential recommendation.

This is a drop-in architectural sibling of bst_model.py / din_model.py:
identical data schema, identical encoders / Dataset / training & evaluation
harness / prediction / synthetic-data generator, but the user-interest
extractor is swapped from BST's "append-candidate-as-target-token +
bidirectional self-attention" scheme to SASRec's original design (Kang &
McAuley, "Self-Attentive Sequential Recommendation") extended with side
(F = "Feature") information fused into every token embedding, following the
common "SASRecF" variant used e.g. in RecBole.

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
Given a user's historical item sequence (+ user/item side features), predict the
target item the user will interact with next. As in bst_model.py / din_model.py
this is cast as a candidate-ranking problem trained with a FULL softmax loss
over the entire item catalog (no negative sampling), matching the candidate
space used at evaluation time (ranking the full catalog for NDCG@10).

Output
------
- Training prints valid NDCG@10 each epoch.
- predict.py-equivalent (run_predict) produces submission.csv with columns:
    uid,prediction
  where prediction is a comma-quoted string of top-10 item ids, e.g.:
    u000009,"i001952,i001038,i001710,i001046,i000401,i001445,i001069,i001002,i001673,i000661"

How SASRecF differs from BST here
----------------------------------
BST appends the CANDIDATE item as an extra token at the end of the sequence
and reads off the Transformer's output AT THAT TARGET-TOKEN POSITION as the
"attended interest" -- so BST needs a candidate at inference time and (to
avoid label leakage) has to stand in the last history item for it. SASRecF
instead follows the original SASRec recipe:
  1. Embeds ONLY the historical sequence (no candidate token is ever
     appended). Item token = item-id embedding CONCATENATED with embeddings
     of its side/categorical features (this feature fusion is the "F" in
     SASRecF) + a token-level frequency signal (log1p(item_seq_counts),
     projected and added), exactly the same feature set BST uses.
  2. Adds learned positional embeddings (position = index within the
     left-padded window, oldest -> most recent).
  3. Runs a stack of CAUSAL (unidirectional) multi-head self-attention
     Transformer encoder blocks (Pre-LN) over the history, with a combined
     causal mask (a position can only attend to itself and earlier
     positions) and a padding mask (padded positions are never attended to
     as keys). This is the key architectural difference from BST, which
     uses full bidirectional attention over [history ; candidate].
  4. Takes the Transformer's output AT THE LAST (most-recent, i.e.
     right-most non-padded) POSITION as the user's sequential interest
     representation -- standard SASRec inference -- concatenates it with the
     user's side-feature embedding, and projects through an MLP to the
     shared scoring space. Because no candidate is involved in building this
     representation, there is no train/eval leakage issue and no need for
     BST's "stand-in candidate = last history item" trick: the SAME
     forward pass is used unchanged at both training and inference time.

Usage
-----
python sasrecf_model.py --data_dir /path/to/data --out_dir /path/to/output --epochs 5
All configuration lives in the `Config` class below; CLI flags simply override
its defaults (see `Config.from_args`).
"""

import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from dataclasses import dataclass, fields

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
    Construct with defaults via Config(), or parse CLI overrides via Config.from_args()."""

    # paths
    data_dir = "../data/A2-Rec"
    out_dir = "./"

    # data / sequence handling
    max_seq_len = 50          # truncate/pad history to this many distinct items (most recent kept)
    val_frac = 0.1          # fraction of train.csv randomly held out as the valid set
    use_synthetic = False    # generate synthetic data into data_dir if real files are missing

    # model
    emb_dim = 32
    side_emb_ratio = 0.5    # side-feature embedding dim = emb_dim * side_emb_ratio
    n_heads = 4              # Transformer multi-head attention heads
    n_layers = 2              # number of causal Transformer encoder blocks
    ffn_mult = 4              # Transformer feed-forward hidden = item_full_dim * ffn_mult
    mlp_hidden = (200, 80)  # final user-representation MLP hidden sizes
    dropout = 0.2

    # training
    epochs = 20
    batch_size = 512
    lr = 5e-4
    grad_clip = 5.0
    seed = 42

    # prediction
    topk = 10

    @classmethod
    def from_args(cls):
        cfg = cls()
        parser = argparse.ArgumentParser()
        for f in fields(cls):
            pass  # placeholder kept for symmetry with bst_model.py's CLI hook
        defaults = {f.name: getattr(cfg, f.name, None) for f in fields(cls)}
        for name, val in vars(cfg).items():
            if name not in defaults:
                defaults[name] = val
        for name, val in defaults.items():
            if val is None:
                continue
            arg_type = type(val) if not isinstance(val, bool) else str
            parser.add_argument(f"--{name}", type=arg_type, default=None)
        args, _ = parser.parse_known_args()
        for name, val in vars(args).items():
            if val is not None:
                setattr(cfg, name, val)
        return cfg


# ========================================================================================
# 1. Vocab / encoding utilities  (identical to bst_model.py / din_model.py)
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
# 2. Data loading  (identical to bst_model.py / din_model.py)
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
# 3. Dataset  (identical to bst_model.py -- SASRecF consumes the exact same sequence
#    of (item, count) pairs; the only difference is how the MODEL uses them)
# ========================================================================================

class SasRecDataset(Dataset):
    """
    Each sample: (uid_idx, hist_item_idx[seq_len], hist_count[seq_len], hist_len, target_item_idx)

    History is built from `item_seq_dedup` (distinct items, in first-seen order)
    paired with their frequency from `item_seq_counts`, exactly as in
    bst_model.py / din_model.py. The count signal is fed into the Transformer
    as a token-level frequency feature (log1p(count) projected and added to
    the token embedding), same as BST.

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
        # items (i.e. the tail of the dedup list) -- this ordering matters for SASRecF
        # since positional embeddings AND the causal mask both rely on recency order
        # within the kept window.
        items = items[-self.max_len:]
        counts = [counts_map.get(i, 1) for i in items]  # default count=1 if missing (defensive)

        hist_idx = [self.iid_enc.transform_one(i) for i in items]
        hist_len = len(hist_idx)

        # LEFT-pad so the most recent item always sits at the last position
        # (max_len - 1) -- this keeps positional embeddings (0..max_len-1,
        # oldest-to-most-recent) and the causal mask consistent across
        # sequences of different lengths, and guarantees the "most recent
        # item" output (used as the user representation) is always read
        # from a fixed index.
        pad_n = self.max_len - hist_len
        hist_idx = [PAD_IDX] * pad_n + hist_idx
        hist_cnt = [0] * pad_n + counts  # 0 count on padding positions -> no frequency signal

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
# 4. SASRecF model
# ========================================================================================

class CausalTransformerBlock(nn.Module):
    """Pre-LN Transformer encoder block with CAUSAL (unidirectional) multi-head
    self-attention + position-wise feed-forward, each wrapped in a residual
    connection. Pre-LN trains more stably than the original Post-LN
    formulation. Unlike BST's TransformerBlock (bidirectional, only a padding
    mask), this block additionally takes a causal `attn_mask` so position i
    can only attend to positions <= i -- this is what turns the encoder into
    an autoregressive next-item model rather than a candidate-conditioned
    bidirectional encoder."""

    def __init__(self, dim, n_heads, ffn_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask, key_padding_mask):
        # x: [B, T, D]
        # attn_mask: [T, T] bool, True at (query, key) pairs that must be IGNORED (future positions)
        # key_padding_mask: [B, T] bool, True at positions to IGNORE (padding)
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class SASRecF(nn.Module):
    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        """
        user_feat_vocabs: list[int] vocab sizes for each u_cat_* column
        item_feat_vocabs: list[int] vocab sizes for each i_cat_*/i_bucket_* column
        """
        super().__init__()
        emb_dim = cfg.emb_dim
        self.emb_dim = emb_dim
        self.max_seq_len = cfg.max_seq_len

        # core id embeddings
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=PAD_IDX)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=PAD_IDX)  # used both as history & scoring emb

        # side feature embeddings (smaller dim each, concatenated) -- this
        # concatenation of id-embedding + categorical side-feature embeddings
        # into a single token vector is the "F" (feature fusion) in SASRecF.
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))
        self.user_side_embs = nn.ModuleList([
            nn.Embedding(v, side_dim, padding_idx=PAD_IDX) for v in user_feat_vocabs
        ])
        self.item_side_embs = nn.ModuleList([
            nn.Embedding(v, side_dim, padding_idx=PAD_IDX) for v in item_feat_vocabs
        ])

        item_full_dim = emb_dim + side_dim * len(item_feat_vocabs)
        user_full_dim = emb_dim + side_dim * len(user_feat_vocabs)
        self.item_full_dim = item_full_dim

        # NOTE: unlike BST, there is no appended candidate/target token, so the
        # positional table only needs to cover the history window itself
        # (0..max_seq_len-1), not max_seq_len + 1.
        self.pos_emb = nn.Embedding(cfg.max_seq_len, item_full_dim)
        # projects log1p(count) -> item_full_dim, added to the token embedding as a
        # frequency signal (0 vector on padding positions, which carry no count)
        self.count_proj = nn.Linear(1, item_full_dim)

        ffn_dim = item_full_dim * cfg.ffn_mult
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(item_full_dim, cfg.n_heads, ffn_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.final_ln = nn.LayerNorm(item_full_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # final MLP: user side feat + Transformer's last-position output
        # (no raw candidate embedding here -- SASRecF's interest vector is
        # built without ever seeing a candidate, unlike BST/DIN's
        # candidate-conditioned mlp_in).
        mlp_in = user_full_dim + item_full_dim
        h1, h2 = cfg.mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, h1), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h1, h2), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h2, emb_dim),  # project to emb_dim so we can score vs. full item catalog via dot product
        )

        # registered buffers filled in by set_feature_tables()
        self.register_buffer("user_feat_table", torch.zeros(n_users, 1, dtype=torch.long), persistent=False)
        self.register_buffer("item_feat_table", torch.zeros(n_items, 1, dtype=torch.long), persistent=False)
        # cached causal mask, (re)built lazily per sequence length in `_causal_mask`
        self._causal_mask_cache = {}

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.user_feat_table = user_feat_table
        self.item_feat_table = item_feat_table

    def _item_full_emb(self, item_idx):
        # item_idx: [...,] long tensor of item ids
        base = self.item_emb(item_idx)  # [..., emb_dim]
        feats = self.item_feat_table[item_idx]  # [..., n_i_cat]
        side_parts = [emb(feats[..., j]) for j, emb in enumerate(self.item_side_embs)]
        return torch.cat([base] + side_parts, dim=-1)

    def _user_full_emb(self, uid_idx):
        base = self.user_emb(uid_idx)  # [B, emb_dim]
        feats = self.user_feat_table[uid_idx]  # [B, n_u_cat]
        side_parts = [emb(feats[..., j]) for j, emb in enumerate(self.user_side_embs)]
        return torch.cat([base] + side_parts, dim=-1)

    def _causal_mask(self, L, device):
        # True at (query, key) pairs where key is a FUTURE position relative
        # to query -- these must never be attended to. Cached per (L, device)
        # since it only depends on the sequence length, not on the batch.
        key = (L, device)
        mask = self._causal_mask_cache.get(key)
        if mask is None:
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            self._causal_mask_cache[key] = mask
        return mask

    def encode_user(self, uid_idx, hist, hist_cnt, hist_len):
        """Produces the user representation vector for scoring against item embeddings.

        Embeds the history sequence (token = item-id emb + side-feature embs +
        frequency signal + positional emb), runs it through a CAUSAL
        Transformer encoder (each position only attends to itself and earlier
        positions, plus a padding mask so left-padding is never attended to
        as keys), and reads off the output at the LAST position -- which,
        thanks to left-padding, always corresponds to the most recent real
        history item -- as the sequential interest representation. This is
        exactly SASRec's inference-time recipe (use h_t, the representation
        after consuming the whole history, to score the next item), with no
        candidate ever entering the encoder.
        """
        B, L = hist.shape
        mask = (hist != PAD_IDX)  # [B, L] True where a real history item is present

        hist_full = self._item_full_emb(hist)  # [B, L, D]

        # frequency signal: log1p(count), 0 on padding positions.
        hist_freq = self.count_proj(torch.log1p(hist_cnt).unsqueeze(-1))  # [B, L, D]
        hist_freq = hist_freq * mask.unsqueeze(-1).float()

        seq = hist_full + hist_freq  # [B, L, D]

        pos_ids = torch.arange(L, device=hist.device).unsqueeze(0).expand(B, -1)  # [B, L]
        seq = seq + self.pos_emb(pos_ids)
        seq = self.emb_dropout(seq)

        causal_mask = self._causal_mask(L, hist.device)          # [L, L]
        key_padding_mask = ~mask                                  # [B, L], True = ignore (padding)

        x = seq
        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)
            # IMPORTANT: a padded position that sits before ALL real history
            # (very common with left-padding + a causal mask: such a query's
            # allowed key set -- causal AND non-padded -- is empty) produces
            # an attention output of NaN for that row whenever PyTorch takes
            # its fused/fast attention kernel path (this path is used e.g. in
            # eval() mode with dropout=0, but not necessarily in train()
            # mode -- so this bug can silently pass unit/training checks and
            # only surface at validation/inference time). If left alone,
            # that NaN is carried forward as a KEY/VALUE in the *next*
            # layer's attention and contaminates real, unmasked positions
            # too (attention weight ~0 times a NaN value is still NaN, not
            # 0). We defend against this exactly like the original SASRec
            # implementation does: explicitly zero out padded positions
            # after every block. `masked_fill` (not multiplication by the
            # mask) is required here since 0 * NaN is still NaN -- only an
            # outright overwrite actually clears it.
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = self.final_ln(x)

        # last position == most-recent real history item, thanks to left-padding
        seq_repr = x[:, -1, :]  # [B, D]

        user_full = self._user_full_emb(uid_idx)  # [B, user_full_dim]

        x_cat = torch.cat([user_full, seq_repr], dim=-1)
        user_vec = self.mlp(x_cat)  # [B, emb_dim]
        return user_vec


class SASRecFRanker(nn.Module):
    """Wraps SASRecF: trains with a FULL softmax over the entire item catalog
    using the user vector produced by SASRecF's causal Transformer, and a
    separate static item-scoring embedding (item_score_head) used to score
    ALL items at both train and inference time -- identical strategy to
    BSTRanker / DINRanker, so results stay directly comparable across the
    three model variants."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        self.sasrecf = SASRecF(n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg)
        emb_dim = cfg.emb_dim
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))
        item_full_dim = emb_dim + side_dim * len(item_feat_vocabs)
        self.item_score_head = nn.Linear(item_full_dim, emb_dim)  # static item vector for full-catalog scoring
        self.n_items = n_items

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.sasrecf.set_feature_tables(user_feat_table, item_feat_table)

    def item_static_vec(self, item_idx):
        full = self.sasrecf._item_full_emb(item_idx)
        return self.item_score_head(full)

    def forward(self, uid_idx, hist, hist_cnt, hist_len):
        user_vec = self.sasrecf.encode_user(uid_idx, hist, hist_cnt, hist_len)  # [B, D]
        return user_vec

    def score_against_catalog(self, user_vec, item_vecs):
        # user_vec: [B, D], item_vecs: [N, D] -> [B, N]
        return user_vec @ item_vecs.t()


# ========================================================================================
# 5. Training: full softmax over the entire item catalog  (identical to bst_model.py)
# ========================================================================================

def full_softmax_loss(user_vec, target_idx, all_item_vecs):
    """
    user_vec: [B, D]
    target_idx: [B]  (true next-item index for each sample)
    all_item_vecs: [n_items, D]  static item vectors for the ENTIRE catalog (index 0 = PAD)

    Computes logits against every item in the catalog (no negative sampling),
    masks out the PAD index so the model never learns to score it, and applies
    standard cross-entropy with the true target as the label. This removes the
    train/eval distribution mismatch that sampled softmax introduces.
    """
    logits = user_vec @ all_item_vecs.t()  # [B, n_items]
    logits = logits.clone()
    logits[:, PAD_IDX] = -1e9  # never let the model assign probability mass to PAD
    loss = F.cross_entropy(logits, target_idx)
    return loss


# ========================================================================================
# 6. NDCG@10 metric  (identical to bst_model.py / din_model.py)
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

        # unlike BST, SASRecF never needs a stand-in candidate: the encoder
        # only ever sees the history, so the exact same forward pass is used
        # here as at training time -- no train/eval mismatch to work around.
        user_vec = model(uid_idx, hist, hist_cnt, hist_len)  # [B, D]
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
# 7. Training loop
# ========================================================================================

def train_model(bundle, cfg: "Config", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    train_full = bundle.train_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    n_val = int(len(train_full) * cfg.val_frac)
    val_df = train_full.iloc[:n_val].reset_index(drop=True)
    tr_df = train_full.iloc[n_val:].reset_index(drop=True)
    print(f"[INFO] train={len(tr_df)}  valid={len(val_df)}")

    train_ds = SasRecDataset(tr_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=True)
    val_ds = SasRecDataset(val_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    user_feat_vocabs = [len(bundle.u_encoders[c]) for c in bundle.u_cat_cols]
    item_feat_vocabs = [len(bundle.i_encoders[c]) for c in bundle.i_cat_cols]

    model = SASRecFRanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
    model.set_feature_tables(
        torch.tensor(bundle.user_feat, dtype=torch.long),
        torch.tensor(bundle.item_feat, dtype=torch.long),
    )
    model.to(device)
    model.sasrecf.user_feat_table = model.sasrecf.user_feat_table.to(device)
    model.sasrecf.item_feat_table = model.sasrecf.item_feat_table.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    all_item_ids = torch.arange(bundle.n_items, device=device)

    best_ndcg = -1.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "sasrecf_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            uid_idx = batch["uid_idx"].to(device)
            hist = batch["hist"].to(device)
            hist_cnt = batch["hist_cnt"].to(device)
            hist_len = batch["hist_len"].to(device)
            target = batch["target"].to(device)

            user_vec = model(uid_idx, hist, hist_cnt, hist_len)

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
# 8. Prediction
# ========================================================================================

@torch.no_grad()
def run_predict(model, bundle, cfg: "Config", device):
    test_df = bundle.test_df
    if test_df is None:
        print("[WARN] no test.csv found, skipping prediction.")
        return

    model.eval()
    ds = SasRecDataset(test_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=False)
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

        user_vec = model(uid_idx, hist, hist_cnt, hist_len)
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
# 9. Synthetic data generator (for local smoke-testing when no real data is uploaded)
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
# 10. Main
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
                f"Re-run with --use_synthetic to test the pipeline on fake data."
            )

    bundle = load_data(cfg)
    print(f"[INFO] n_users={bundle.n_users}  n_items={bundle.n_items}  "
          f"u_cat_cols={bundle.u_cat_cols}  i_cat_cols={bundle.i_cat_cols}")

    model, device = train_model(bundle, cfg)

    run_predict(model, bundle, cfg, device)


if __name__ == "__main__":
    main()
