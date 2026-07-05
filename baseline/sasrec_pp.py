# -*- coding: utf-8 -*-
"""
SASRec++ : Self-Attentive Sequential Recommendation (enhanced) for
next-item / sequential recommendation.

This file mirrors the structure of din_model_2.py (same Config style, same
data schema, same encoders, same full-softmax training / NDCG@10 evaluation /
submission format) but replaces DIN's target-dependent local activation-unit
attention with SASRec's causal self-attention Transformer over the user's
history. On top of vanilla SASRec we add a few "++" enhancements that reuse
signal already present in the DIN pipeline:

  1. Side-feature fusion   : user/item categorical side features are embedded
                              and concatenated into the item/user representations
                              (DIN does this too; vanilla SASRec normally only
                              has item ids).
  2. Frequency-aware gate  : log1p(item_seq_counts) for each history item is
                              passed through a small gate that rescales that
                              item's embedding before it enters self-attention,
                              so items the user repeatedly interacted with get
                              a stronger signal (DIN uses log-count as an extra
                              activation-unit feature; here it modulates the
                              input embedding instead of an attention weight).
  3. Auxiliary dense supervision : besides the standard "encode whole history
                              -> predict target_iid" full-softmax loss, we
                              optionally add SASRec's original per-position
                              next-item loss (predict item[t+1] from item[t]
                              for every valid position in the history), which
                              gives far more training signal per sequence.
                              Off by default (aux_loss_weight = 0.0) to keep
                              default training cost comparable to DIN; set
                              cfg.aux_loss_weight > 0 to enable it.

Data schema
-----------
user.csv : uid,u_cat_01..u_cat_08                 (8 user categorical features, 0 is a VALID value)
item.csv : iid,i_cat_01,i_cat_02,i_cat_03,i_bucket_01   (item categorical features, 0 is a VALID value)
train.csv: uid,target_iid,item_seq_raw,item_seq_dedup,item_seq_counts
test.csv : uid,item_seq_raw,item_seq_dedup,item_seq_counts

item_seq_raw    : "i000001,i000002,..."            (full click history, may contain repeats, in order)
item_seq_dedup  : same format, deduplicated, first-seen order
item_seq_counts : "i000001:18,i000002:17,..."       (item -> count in history)

Task
----
Given a user's historical item sequence (+ user/item side features), predict
the target item the user will interact with next. Like DIN, this is cast as a
candidate-ranking problem trained with a FULL softmax loss over the entire
item catalog (no negative sampling), matching the full-catalog ranking used
at evaluation time (NDCG@10).

Unlike DIN, SASRec's self-attention is over the history sequence ONLY (it is
NOT conditioned on a candidate item), so there is no "leak the label into
attention" hazard and no need for the last-history-item-as-candidate trick
that DIN's training loop requires -- the same forward pass is used verbatim
at train, validation and prediction time.

Output
------
- Training prints valid NDCG@10 each epoch.
- run_predict produces submission.csv with columns:
    uid,prediction
  where prediction is a comma-quoted string of top-10 item ids, e.g.:
    u000009,"i001952,i001038,i001710,i001046,i000401,i001445,i001069,i001002,i001673,i000661"

Usage
-----
python sasrec_pp.py --data_dir /path/to/data --out_dir /path/to/output --epochs 5
All configuration lives in the `Config` class below.
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
    max_seq_len = 50          # truncate/pad history to this many items (most recent kept)
    use_dedup_sequence = True  # True: build history from item_seq_dedup+counts (matches DIN);
                               # False: use item_seq_raw (keeps repeats, weight-by-count disabled)
    val_frac = 0.1           # fraction of train.csv randomly held out as the valid set
    use_synthetic = False     # generate synthetic data into data_dir if real files are missing

    # model (SASRec++ self-attention encoder)
    emb_dim = 32              # == transformer d_model, so id-embedding dot-product scoring works
    side_emb_ratio = 0.5      # side-feature embedding dim = emb_dim * side_emb_ratio
    n_blocks = 2              # number of self-attention blocks
    n_heads = 2               # attention heads per block (must divide emb_dim)
    ffn_hidden = 64           # point-wise feed-forward hidden size inside each block
    dropout = 0.2

    # training
    epochs = 20
    batch_size = 512
    lr = 1e-3
    grad_clip = 5.0
    seed = 42
    aux_loss_weight = 0.0     # >0 enables SASRec's classic per-position next-item loss

    # prediction
    topk = 10


# ========================================================================================
# 1. Vocab / encoding utilities
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
# 2. Data loading (identical schema / logic to the DIN pipeline)
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
# 3. Dataset
# ========================================================================================

class SASRecDataset(Dataset):
    """
    Each sample: (uid_idx, hist_item_idx[seq_len], hist_count[seq_len], hist_len, target_item_idx)

    Unlike DIN (where history order only mattered for truncation), SASRec's
    self-attention is position-aware, so the ORDER of `hist` matters for the
    positional embeddings and the causal mask. History items are left-padded
    so the most-recent item always sits at position `max_len - 1`.

    If cfg.use_dedup_sequence is True we build history from item_seq_dedup
    (distinct items, first-seen order) paired with item_seq_counts, exactly
    like DIN -- this keeps the frequency-aware gate meaningful and avoids
    burning attention positions on immediate repeats. If False, item_seq_raw
    is used instead (true chronological order incl. repeats; counts default
    to 1 since a raw-sequence position is a single occurrence).
    """

    def __init__(self, df, uid_enc, iid_enc, max_len, use_dedup=True, has_target=True):
        self.uids = df["uid"].tolist()
        self.seq_col = df["item_seq_dedup" if use_dedup else "item_seq_raw"].tolist()
        self.count_strs = df["item_seq_counts"].tolist()
        self.use_dedup = use_dedup
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
        items = parse_seq_raw(self.seq_col[idx])
        counts_map = parse_seq_counts(self.count_strs[idx]) if self.use_dedup else {}

        # keep the most RECENT max_len items (tail of the sequence)
        items = items[-self.max_len:]
        if self.use_dedup:
            counts = [counts_map.get(i, 1) for i in items]
        else:
            counts = [1 for _ in items]  # raw sequence: each position is one occurrence

        hist_idx = [self.iid_enc.transform_one(i) for i in items]
        hist_len = len(hist_idx)

        pad_n = self.max_len - hist_len
        hist_idx = [PAD_IDX] * pad_n + hist_idx
        hist_cnt = [0] * pad_n + counts  # 0 count on padding positions -> no gate contribution

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
# 4. SASRec++ model
# ========================================================================================

class PositionwiseFeedForward(nn.Module):
    """Point-wise feed-forward sub-layer with its own residual + post-LayerNorm,
    exactly as in the original SASRec block (Conv1d-equivalent, done here as Linear
    since we operate on [B, L, D] directly)."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))


class SASRecBlock(nn.Module):
    """One causal self-attention block: pre-LN multi-head self-attention with a
    residual connection, followed by the point-wise feed-forward sub-layer."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, causal_mask, key_padding_mask):
        # x: [B, L, D]; causal_mask: [L, L] additive; key_padding_mask: [B, L] True=ignore
        h = self.attn_norm(x)
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.attn_dropout(attn_out)
        x = self.ffn(x)
        return x


class SASRecEncoder(nn.Module):
    """Encodes a history sequence into contextualized item representations using
    causal self-attention. Item id embeddings are shared between the input
    sequence and the full-catalog scoring vectors (weight tying, as in the
    original SASRec paper), fused with side-feature embeddings ("++" #1) and
    rescaled by a frequency-aware gate built from log1p(item_seq_counts)
    ("++" #2) before entering the transformer."""

    def __init__(self, n_items, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        emb_dim = cfg.emb_dim
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))

        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=PAD_IDX)
        self.item_side_embs = nn.ModuleList([
            nn.Embedding(v, side_dim, padding_idx=PAD_IDX) for v in item_feat_vocabs
        ])
        item_full_dim = emb_dim + side_dim * len(item_feat_vocabs)

        self.input_proj = nn.Linear(item_full_dim, emb_dim)  # -> d_model; reused for catalog scoring
        self.pos_emb = nn.Embedding(cfg.max_seq_len, emb_dim)
        self.freq_gate = nn.Linear(1, emb_dim)  # log1p(count) -> per-dim gate, "++" enhancement
        self.input_dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            SASRecBlock(emb_dim, cfg.n_heads, cfg.ffn_hidden, cfg.dropout)
            for _ in range(cfg.n_blocks)
        ])
        self.out_norm = nn.LayerNorm(emb_dim)

        self.register_buffer("item_feat_table", torch.zeros(n_items, 1, dtype=torch.long), persistent=False)

    def set_feature_table(self, item_feat_table):
        self.item_feat_table = item_feat_table

    def item_full_emb(self, item_idx):
        base = self.item_emb(item_idx)
        feats = self.item_feat_table[item_idx]
        side_parts = [emb(feats[..., j]) for j, emb in enumerate(self.item_side_embs)]
        return torch.cat([base] + side_parts, dim=-1)

    def item_static_vec(self, item_idx):
        """Item vector used for full-catalog scoring. Reuses input_proj so the
        input embedding and the output (scoring) embedding are weight-tied,
        matching SASRec's original design instead of a separate scoring head."""
        return self.input_proj(self.item_full_emb(item_idx))

    def forward(self, hist, hist_cnt):
        """hist: [B, L] item ids (0 = pad, left-padded); hist_cnt: [B, L] raw counts.
        Returns contextualized sequence representations [B, L, D]."""
        B, L = hist.shape
        device = hist.device
        mask = (hist != PAD_IDX)  # [B, L] bool, True = valid position

        x = self.input_proj(self.item_full_emb(hist))  # [B, L, D]

        log_cnt = torch.log1p(hist_cnt).unsqueeze(-1)  # [B, L, 1]
        gate = torch.sigmoid(self.freq_gate(log_cnt))  # [B, L, D]
        x = x * gate

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(positions)
        x = self.input_dropout(x)
        x = x * mask.unsqueeze(-1).float()  # zero out padding before attention

        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1
        )  # [L, L], True = position i may NOT attend to j (j > i)
        key_padding_mask = ~mask  # [B, L], True = ignore (padding)
        # both masks are boolean so PyTorch combines them without a dtype-mismatch warning

        for blk in self.blocks:
            x = blk(x, causal_mask, key_padding_mask)
            x = x * mask.unsqueeze(-1).float()  # re-zero padding after each block

        return self.out_norm(x)  # [B, L, D]


class SASRecRanker(nn.Module):
    """Wraps the SASRec++ encoder: the representation at the last (most recent,
    right-most thanks to left-padding) valid position is fused with the user's
    side features to form the user vector, which is scored against the full
    item catalog via full softmax -- same training/eval contract as DIN, but
    with NO candidate-dependent attention, so the exact same forward pass runs
    at train, validation and inference time."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        emb_dim = cfg.emb_dim
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))

        self.encoder = SASRecEncoder(n_items, item_feat_vocabs, cfg)

        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=PAD_IDX)
        self.user_side_embs = nn.ModuleList([
            nn.Embedding(v, side_dim, padding_idx=PAD_IDX) for v in user_feat_vocabs
        ])
        user_full_dim = emb_dim + side_dim * len(user_feat_vocabs)

        self.user_proj = nn.Sequential(
            nn.Linear(emb_dim + user_full_dim, emb_dim), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(emb_dim, emb_dim),
        )

        self.register_buffer("user_feat_table", torch.zeros(n_users, 1, dtype=torch.long), persistent=False)
        self.n_items = n_items

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.user_feat_table = user_feat_table
        self.encoder.set_feature_table(item_feat_table)

    def user_full_emb(self, uid_idx):
        base = self.user_emb(uid_idx)
        feats = self.user_feat_table[uid_idx]
        side_parts = [emb(feats[..., j]) for j, emb in enumerate(self.user_side_embs)]
        return torch.cat([base] + side_parts, dim=-1)

    def encode_sequence(self, hist, hist_cnt):
        """Returns the full contextualized sequence [B, L, D] -- exposed separately
        so the training loop can compute the optional per-position auxiliary loss
        without re-running the encoder."""
        return self.encoder(hist, hist_cnt)

    def forward(self, uid_idx, hist, hist_cnt, hist_len):
        seq_out = self.encode_sequence(hist, hist_cnt)  # [B, L, D]
        last_hidden = seq_out[:, -1, :]  # left-padded -> last position is always the most recent valid item
        user_full = self.user_full_emb(uid_idx)
        user_vec = self.user_proj(torch.cat([last_hidden, user_full], dim=-1))
        return user_vec

    def item_static_vec(self, item_idx):
        return self.encoder.item_static_vec(item_idx)

    def score_against_catalog(self, user_vec, item_vecs):
        # user_vec: [B, D], item_vecs: [N, D] -> [B, N]
        return user_vec @ item_vecs.t()


# ========================================================================================
# 5. Training: full softmax over the entire item catalog (+ optional auxiliary loss)
# ========================================================================================

def full_softmax_loss(user_vec, target_idx, all_item_vecs):
    """
    user_vec: [B, D]
    target_idx: [B]  (true next-item index for each sample)
    all_item_vecs: [n_items, D]  static item vectors for the ENTIRE catalog (index 0 = PAD)

    Computes logits against every item in the catalog (no negative sampling),
    masks out the PAD index, and applies standard cross-entropy against the
    true target. Same objective as DIN's full-softmax loss.
    """
    logits = user_vec @ all_item_vecs.t()  # [B, n_items]
    logits = logits.clone()
    logits[:, PAD_IDX] = -1e9
    loss = F.cross_entropy(logits, target_idx)
    return loss


def aux_next_item_loss(seq_out, hist, target_idx, all_item_vecs, proj):
    """SASRec's classic per-position dense supervision: at every valid position t
    in the history, predict the item that comes right after it (hist[t+1], or
    `target_idx` for the very last position). This gives many more training
    signals per sequence than the single final-position loss above.

    seq_out : [B, L, D] contextualized representations from the encoder
    hist    : [B, L]    input item ids (0 = pad)
    proj    : callable mapping raw item ids -> catalog scoring vectors
              (SASRecRanker.item_static_vec), used to build the shifted labels'
              logits via the same weight-tied embedding table.
    """
    B, L = hist.shape
    mask = (hist != PAD_IDX)  # [B, L] positions that HAVE a valid "current" item

    # label at position t is hist[t+1] for t < L-1, and target_idx for t == L-1
    shifted_labels = torch.cat([hist[:, 1:], target_idx.unsqueeze(1)], dim=1)  # [B, L]

    flat_logits = seq_out.reshape(B * L, -1) @ all_item_vecs.t()  # [B*L, n_items]
    flat_logits = flat_logits.clone()
    flat_logits[:, PAD_IDX] = -1e9
    flat_labels = shifted_labels.reshape(-1)
    flat_mask = mask.reshape(-1)

    if flat_mask.sum() == 0:
        return seq_out.new_zeros(())

    loss = F.cross_entropy(flat_logits[flat_mask], flat_labels[flat_mask])
    return loss


# ========================================================================================
# 6. NDCG@10 metric
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

        # No candidate-dependent attention in SASRec, so this is exactly the
        # same forward pass used during training -- no train/eval mismatch.
        user_vec = model(uid_idx, hist, hist_cnt, hist_len)  # [B, D]
        scores = model.score_against_catalog(user_vec, item_vecs_all)  # [B, n_items]
        scores[:, PAD_IDX] = -1e9

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

    train_ds = SASRecDataset(tr_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len,
                              use_dedup=cfg.use_dedup_sequence, has_target=True)
    val_ds = SASRecDataset(val_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len,
                            use_dedup=cfg.use_dedup_sequence, has_target=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    user_feat_vocabs = [len(bundle.u_encoders[c]) for c in bundle.u_cat_cols]
    item_feat_vocabs = [len(bundle.i_encoders[c]) for c in bundle.i_cat_cols]

    assert cfg.emb_dim % cfg.n_heads == 0, "emb_dim must be divisible by n_heads for multi-head attention"

    model = SASRecRanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
    model.set_feature_tables(
        torch.tensor(bundle.user_feat, dtype=torch.long),
        torch.tensor(bundle.item_feat, dtype=torch.long),
    )
    model.to(device)
    model.user_feat_table = model.user_feat_table.to(device)
    model.encoder.item_feat_table = model.encoder.item_feat_table.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    all_item_ids = torch.arange(bundle.n_items, device=device)

    best_ndcg = -1.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "sasrec_pp_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            uid_idx = batch["uid_idx"].to(device)
            hist = batch["hist"].to(device)
            hist_cnt = batch["hist_cnt"].to(device)
            hist_len = batch["hist_len"].to(device)
            target = batch["target"].to(device)

            seq_out = model.encode_sequence(hist, hist_cnt)  # [B, L, D]
            last_hidden = seq_out[:, -1, :]
            user_full = model.user_full_emb(uid_idx)
            user_vec = model.user_proj(torch.cat([last_hidden, user_full], dim=-1))

            # recompute the full-catalog item vectors every step (weight-tied
            # embeddings change each update, so these can't be cached).
            all_item_vecs = model.item_static_vec(all_item_ids)  # [n_items, D]
            loss = full_softmax_loss(user_vec, target, all_item_vecs)

            if cfg.aux_loss_weight > 0:
                loss = loss + cfg.aux_loss_weight * aux_next_item_loss(
                    seq_out, hist, target, all_item_vecs, model.item_static_vec
                )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

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
    ds = SASRecDataset(test_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len,
                        use_dedup=cfg.use_dedup_sequence, has_target=False)
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

        topk = torch.topk(scores, k=cfg.topk, dim=1).indices.cpu().numpy()

        for row in topk:
            item_strs = [bundle.iid_enc.idx2value.get(int(i), "i000000") for i in row]
            rows.append(",".join(item_strs))

    out_df = pd.DataFrame({"uid": uids, "prediction": rows})
    out_path = os.path.join(cfg.out_dir, "submission.csv")
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
