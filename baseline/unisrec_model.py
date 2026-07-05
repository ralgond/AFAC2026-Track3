# -*- coding: utf-8 -*-
"""
UniSRec (content-based, ID-free universal sequence recommender) for
next-item / sequential recommendation.

This script imitates the structure of `din_model_2.py` (same data schema,
same encoders/Dataset, same full-softmax-over-the-catalog training objective,
same NDCG@10 eval loop and submission format) but replaces DIN's ID-embedding
+ attention design with UniSRec's (Hou et al., KDD 2022) core idea: represent
every item purely from its CONTENT, never from a learned item-ID embedding
table, so the sequence encoder could in principle transfer to items/domains
it has never seen an ID for.

A note on what had to be adapted from the original paper
----------------------------------------------------------
The original UniSRec builds each item's content vector by running its text
(title / description) through a FROZEN pretrained language model, then feeds
that fixed text embedding through a learned "MoE-enhanced adaptor" into the
recommendation embedding space. This dataset has no item text -- only the
categorical side features `i_cat_01, i_cat_02, i_cat_03, i_bucket_01`. We
substitute those as the "content" signal: conceptually they play the same
role a PLM text embedding would (a description of the item that exists
independently of its arbitrary integer id), just categorical instead of
textual. Everything downstream of that -- the MoE adaptor, the SASRec-style
sequence encoder, the contrastive objectives -- is implemented as in the
paper. This is the same kind of substitution `din_dcnv2_model.py` etc. make
when a listed branch's usual raw input isn't present in this schema; it is
flagged here because it is the single most load-bearing design choice in
this file (an item-ID embedding table would silently defeat the entire
point of the architecture).

Concretely, UniSRec contributes two ideas on top of a plain SASRec encoder:

  1. MoE-enhanced adaptor -- an item's content vector is transformed into the
     embedding space used for retrieval by K learned "expert" projections,
     combined with weights from a gate that is itself conditioned on the SAME
     content vector (content-aware routing), rather than by a single fixed
     linear map. This lets structurally different kinds of item content route
     through different experts. Crucially, the SAME adaptor is used for BOTH
     a history item's embedding inside the sequence encoder AND a catalog
     item's embedding at scoring time -- there is only one universal item
     representation function in this model, no separate "history" vs
     "catalog" item embedding tables the way din_model_2.py has.
  2. Sequence-sequence contrastive learning -- besides the main next-item
     (full-softmax) objective, two randomly-augmented views of the same
     history (independent random item dropout) are encoded through the same
     sequence encoder, and an InfoNCE loss pulls the two views' final states
     together (positive pair) while pushing apart every other sequence in the
     batch (negatives). This is the paper's mechanism for learning sequence
     representations that are robust to which exact items happened to be
     observed, which is what lets the learned encoder generalize.

Data schema
-----------
user.csv : uid,u_cat_01..u_cat_08                 (8 user categorical features, 0 is a VALID value)
item.csv : iid,i_cat_01,i_cat_02,i_cat_03,i_bucket_01   (item categorical features, 0 is a VALID value)
train.csv: uid,target_iid,item_seq_raw,item_seq_dedup,item_seq_counts
test.csv : uid,item_seq_raw,item_seq_dedup,item_seq_counts

Task
----
Given a user's historical item sequence (+ user/item side features), predict
the target item the user will interact with next, cast as full-catalog
ranking. Training uses a FULL softmax loss over the entire item catalog
(cross-entropy against every item, no negative sampling) -- this matches the
candidate space used at evaluation time (ranking the full catalog for
NDCG@10) and avoids a train/eval distribution mismatch that sampled softmax
with a small random-negative pool would introduce. This full-softmax loss
plays the role of UniSRec's sequence-item contrastive task.

item_seq_counts is used in the spirit of `din_model_2.py`: each history
step's adapted item embedding is scaled by (1 + log1p(count)) before it
enters the sequence encoder, so repeatedly-clicked items leave a stronger
mark on the final hidden state. We use the deduplicated sequence
(item_seq_dedup), kept in first-seen order, paired with its counts.

History is LEFT-padded (pad tokens first, real items last) to match this
codebase's convention. The sequence encoder internally converts this to
right-padding via a per-row cyclic gather before running causal self-
attention, exactly as in `din_deepfm_sasrec_model.py` -- see that script's
docstring for why a causal mask combined directly with left-padding is a
correctness trap (fully-masked query rows -> NaN) that this avoids.

Usage
-----
python unisrec_model.py --data_dir /path/to/data --out_dir /path/to/output
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
    emb_dim = 32             # the "universal" item/user retrieval embedding dimension
    side_emb_ratio = 0.5    # side-feature embedding dim = emb_dim * side_emb_ratio (user side only;
                             # item side feeds the content encoder instead, see below)

    # content encoder (stand-in for UniSRec's frozen-PLM text embedding, see module docstring)
    content_dim = 32        # width of the item "content" vector before the MoE adaptor
    content_hidden = (64,)   # hidden sizes of the MLP that builds the content vector

    # MoE-enhanced adaptor
    n_experts = 4
    adaptor_dropout = 0.1

    # SASRec-style sequence encoder
    seq_blocks = 2
    seq_heads = 2
    seq_dropout = 0.2

    # sequence-sequence contrastive learning (UniSRec's auxiliary task)
    cl_weight = 0.1          # weight of the contrastive loss added to the main full-softmax loss
    cl_temperature = 0.2
    cl_dropout_rate = 0.3    # probability of masking out each REAL history item per augmented view

    # fusion MLP (combines raw user embedding + sequence state -> final user vector)
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

    # ---- item id encoder (drives the candidate / softmax space -- NOT fed into the model
    #      as a learned embedding, only used to index into the item_feat/content table) ----
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

    # ---- item categorical features (this IS the item's "content" in this adaptation) ----
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
    paired with their frequency from `item_seq_counts`. The count signal scales
    each step's adapted item embedding before it enters the sequence encoder
    (see module docstring); the first-seen ORDER of `item_seq_dedup` is what
    the encoder's causal self-attention is keyed off of.

    Item ids here are only used to look up each item's SIDE/content features
    inside the model (via `item_feat_table`) -- unlike din_model_2.py, there
    is no item-id embedding table anywhere in this model.
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
# 4. Item content representation + MoE-enhanced adaptor
# ========================================================================================

class ContentEncoder(nn.Module):
    """Builds an item's CONTENT vector purely from its side categorical features
    (i_cat_01..i_bucket_01) -- no item-id embedding table anywhere. See the
    module docstring for why this substitutes for UniSRec's frozen-PLM text
    embedding in the original paper."""

    def __init__(self, side_dim, n_side_fields, content_dim, hidden=(64,), dropout=0.1):
        super().__init__()
        in_dim = side_dim * n_side_fields
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.PReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, content_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, side_embs):
        x = torch.cat(list(side_embs), dim=-1)
        return self.mlp(x)  # [..., content_dim]


class MoEAdaptor(nn.Module):
    """UniSRec's MoE-enhanced adaptor: transforms a content vector into the
    recommendation embedding space via K learned "expert" linear projections,
    combined with weights from a gate that is itself conditioned on the SAME
    content vector (content-aware routing) rather than a single fixed linear
    map. The SAME adaptor instance is reused for both history items (inside
    the sequence encoder) and catalog items (at full-catalog scoring time) --
    there is only one universal item representation function in this model."""

    def __init__(self, content_dim, emb_dim, n_experts=4, dropout=0.1):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(content_dim, emb_dim) for _ in range(n_experts)])
        self.gate = nn.Linear(content_dim, n_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, content_vec):
        # content_vec: [..., content_dim]
        gate_w = torch.softmax(self.gate(content_vec), dim=-1)               # [..., n_experts]
        expert_outs = torch.stack([e(content_vec) for e in self.experts], dim=-2)  # [..., n_experts, emb_dim]
        out = (gate_w.unsqueeze(-1) * expert_outs).sum(dim=-2)               # [..., emb_dim]
        return self.dropout(out)


# ========================================================================================
# 5. SASRec-style causal self-attention sequence encoder
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
    """Self-attentive sequential encoder (SASRec). Encodes a sequence of
    (already-adapted) item embeddings with learned positional embeddings and
    stacked CAUSAL self-attention blocks; the representation at the last
    (= most recent) valid position summarizes the sequence in the order it
    happened.

    History in this pipeline is LEFT-padded (pad tokens first, real items
    last). A causal mask combined directly with left-padding is a
    correctness trap: every padding query position can (by causality) only
    attend to earlier positions, which are ALSO all padding, so its entire
    attention row is masked -> softmax over an all -inf row -> NaN, which
    PyTorch's fast attention path (used under `torch.no_grad()`, i.e. every
    eval / predict call) does not guard against. We avoid this by internally
    re-expressing the sequence as RIGHT-padded (real items moved to the
    front, same relative order, via a per-row cyclic gather) before running
    causal attention, then gathering the output at each row's own last REAL
    position -- see `din_deepfm_sasrec_model.py` for the full story of why
    this matters."""

    def __init__(self, dim, max_len, n_blocks=2, n_heads=2, dropout=0.2):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, dim)
        self.in_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([SASRecBlock(dim, n_heads, dropout) for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(dim)
        self.max_len = max_len

    def forward(self, seq_emb, mask):
        # seq_emb: [B, L, D] (already-adapted item embeddings), mask: [B, L] (1=valid,0=pad, LEFT-padded)
        B, L, D = seq_emb.shape
        hist_len = mask.sum(dim=1)               # [B]
        pad_n = L - hist_len                      # [B]

        shift_idx = (torch.arange(L, device=seq_emb.device).unsqueeze(0) + pad_n.unsqueeze(1)) % L
        gather_idx = shift_idx.unsqueeze(-1).expand(-1, -1, D)
        x_right = torch.gather(seq_emb, 1, gather_idx)      # [B, L, D], real items now at the FRONT
        mask_right = torch.gather(mask, 1, shift_idx)        # [B, L]

        pos_ids = torch.arange(L, device=seq_emb.device).unsqueeze(0).expand(B, -1)
        x = x_right + self.pos_emb(pos_ids)
        x = self.in_dropout(x)
        mask_f = mask_right.unsqueeze(-1).float()
        x = x * mask_f

        causal = torch.triu(torch.ones(L, L, device=seq_emb.device, dtype=torch.bool), diagonal=1)
        key_pad = (mask_right == 0)

        fully_pad = key_pad.all(dim=1)
        if fully_pad.any():
            key_pad = key_pad.clone()
            key_pad[fully_pad, :] = False

        for blk in self.blocks:
            x = blk(x, causal_mask=causal, key_padding_mask=key_pad)
            x = x * mask_f

        x = self.ln_out(x)

        last_pos = (hist_len - 1).clamp(min=0)
        seq_vec = x[torch.arange(B, device=x.device), last_pos, :]  # [B, D]
        return seq_vec


# ========================================================================================
# 6. Fused model: UniSRec (content + MoE adaptor + SASRec, ID-free item representations)
# ========================================================================================

class UniSRec(nn.Module):
    """Encodes (user, history) into a candidate-FREE user_vec. Every item
    representation -- whether a history step inside the sequence encoder or a
    catalog item at scoring time -- goes through the exact same
    content -> MoE-adaptor pipeline (`item_repr`), never through a learned
    per-item-id embedding table."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        emb_dim = cfg.emb_dim
        self.emb_dim = emb_dim
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))

        # ---- user side: normal id + side embeddings (only the ITEM side is made ID-free) ----
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=PAD_IDX)
        self.user_side_embs = nn.ModuleList([nn.Embedding(v, side_dim, padding_idx=PAD_IDX) for v in user_feat_vocabs])
        user_full_dim = emb_dim + side_dim * len(user_feat_vocabs)
        self.user_full_dim = user_full_dim

        # ---- item side: content-only, no item_emb id table ----
        self.item_side_embs = nn.ModuleList([nn.Embedding(v, side_dim, padding_idx=PAD_IDX) for v in item_feat_vocabs])
        self.content_encoder = ContentEncoder(side_dim, len(item_feat_vocabs), cfg.content_dim,
                                               hidden=cfg.content_hidden, dropout=cfg.dropout)
        self.moe_adaptor = MoEAdaptor(cfg.content_dim, emb_dim, n_experts=cfg.n_experts,
                                       dropout=cfg.adaptor_dropout)

        # ---- SASRec sequence encoder over adapted item embeddings ----
        self.sasrec = SASRecEncoder(emb_dim, max_len=cfg.max_seq_len, n_blocks=cfg.seq_blocks,
                                     n_heads=cfg.seq_heads, dropout=cfg.seq_dropout)

        # ---- fusion MLP: [raw user, sequence state] -> user_vec ----
        fuse_in = user_full_dim + emb_dim
        h1, h2 = cfg.mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(fuse_in, h1), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h1, h2), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h2, emb_dim),
        )

        self.cl_dropout_rate = cfg.cl_dropout_rate

        self.register_buffer("user_feat_table", torch.zeros(n_users, max(1, len(user_feat_vocabs)), dtype=torch.long),
                              persistent=False)
        self.register_buffer("item_feat_table", torch.zeros(n_items, max(1, len(item_feat_vocabs)), dtype=torch.long),
                              persistent=False)

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.user_feat_table = user_feat_table
        self.item_feat_table = item_feat_table

    def _user_side(self, uid_idx):
        feats = self.user_feat_table[uid_idx]
        return [emb(feats[..., j]) for j, emb in enumerate(self.user_side_embs)]

    def item_repr(self, item_idx):
        """THE universal item representation function: content features -> MoE
        adaptor -> emb_dim vector. Used identically for history items and for
        the full catalog -- no item-id embedding table exists in this model."""
        feats = self.item_feat_table[item_idx]  # [..., n_i_cat]
        side_parts = [emb(feats[..., j]) for j, emb in enumerate(self.item_side_embs)]
        content = self.content_encoder(side_parts)
        return self.moe_adaptor(content)  # [..., emb_dim]

    def item_static_vec(self, item_idx):
        """Candidate-free item representation for full-catalog scoring -- for
        UniSRec this is simply `item_repr`, since there was never a separate
        'history embedding' vs 'catalog embedding' to begin with."""
        return self.item_repr(item_idx)

    def _encode_sequence(self, hist, hist_cnt, mask):
        item_seq_emb = self.item_repr(hist)                    # [B, L, emb_dim]
        freq_scale = (1.0 + torch.log1p(hist_cnt)).unsqueeze(-1)  # [B, L, 1]
        item_seq_emb = item_seq_emb * freq_scale
        return self.sasrec(item_seq_emb, mask)                  # [B, emb_dim]

    def encode_user(self, uid_idx, hist, hist_cnt, hist_len):
        """Produces the candidate-FREE fused user representation vector."""
        mask = (hist != PAD_IDX).long()
        seq_vec = self._encode_sequence(hist, hist_cnt, mask)   # [B, emb_dim]

        user_base = self.user_emb(uid_idx)
        u_side_parts = self._user_side(uid_idx)
        user_full = torch.cat([user_base] + u_side_parts, dim=-1)

        x = torch.cat([user_full, seq_vec], dim=-1)
        return self.mlp(x)

    def encode_sequence_view(self, hist, hist_cnt, mask):
        """Encodes ONE (possibly augmented) view of a history sequence into a
        pooled sequence vector, without touching the user-id side at all --
        used by the seq-seq contrastive loss, which only cares about whether
        two augmented views of the SAME underlying history end up close in
        embedding space."""
        return self._encode_sequence(hist, hist_cnt, mask)

    def augment(self, hist, hist_cnt, hist_len):
        """Builds one randomly-augmented view of a batch of histories via
        independent per-item dropout on the REAL (non-pad) positions, always
        keeping at least one real item so the sequence never becomes
        fully empty for every single sample simultaneously... a genuinely
        empty result for a given row is fine (SASRecEncoder handles hist_len
        == 0 safely), this floor is only to keep augmentation meaningfully
        different from just "drop everything"."""
        mask = (hist != PAD_IDX)
        keep_prob = 1.0 - self.cl_dropout_rate
        keep = (torch.rand_like(hist, dtype=torch.float32) < keep_prob) & mask
        # make sure at least the most recent real item survives in each row that has any history,
        # so the augmented view isn't trivially "empty" for every sample at once
        B, L = hist.shape
        batch_idx = torch.arange(B, device=hist.device)
        last_valid_pos = L - 1
        has_hist = mask[:, last_valid_pos]  # crude but sufficient: most rows' last col is their most recent item
        keep[batch_idx, last_valid_pos] = keep[batch_idx, last_valid_pos] | has_hist

        aug_hist = torch.where(keep, hist, torch.zeros_like(hist))
        aug_cnt = torch.where(keep, hist_cnt, torch.zeros_like(hist_cnt))
        aug_mask = keep.long()
        return aug_hist, aug_cnt, aug_mask


class UniSRecRanker(nn.Module):
    """Thin wrapper mirroring din_model_2.py's DINRanker: exposes forward() ->
    user_vec, item_static_vec() for full-catalog vectors, and dot-product
    scoring. Also exposes the pieces the training loop needs for the
    sequence-sequence contrastive loss."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        self.core = UniSRec(n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg)
        self.n_items = n_items

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.core.set_feature_tables(user_feat_table, item_feat_table)

    def item_static_vec(self, item_idx):
        return self.core.item_static_vec(item_idx)

    def forward(self, uid_idx, hist, hist_cnt, hist_len):
        return self.core.encode_user(uid_idx, hist, hist_cnt, hist_len)

    def score_against_catalog(self, user_vec, item_vecs):
        return user_vec @ item_vecs.t()

    def contrastive_loss(self, hist, hist_cnt, hist_len, temperature=0.2):
        """UniSRec's sequence-sequence contrastive task: two independently
        augmented views of the same batch of histories should encode to
        similar vectors (in-batch InfoNCE, symmetric both directions)."""
        hist_a, cnt_a, mask_a = self.core.augment(hist, hist_cnt, hist_len)
        hist_b, cnt_b, mask_b = self.core.augment(hist, hist_cnt, hist_len)
        z_a = self.core.encode_sequence_view(hist_a, cnt_a, mask_a)  # [B, D]
        z_b = self.core.encode_sequence_view(hist_b, cnt_b, mask_b)  # [B, D]

        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)
        logits = z_a @ z_b.t() / temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.t(), labels)
        return (loss_ab + loss_ba) / 2.0


# ========================================================================================
# 7. Training: full softmax over the entire item catalog  (identical objective to din_model_2.py)
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
# 8. NDCG@10 metric  (identical to din_model_2.py)
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

        # UniSRec is candidate-free (no DIN-style attention anchor needed), so
        # train-time and eval-time forward passes are identical.
        user_vec = model(uid_idx, hist, hist_cnt, hist_len)
        scores = model.score_against_catalog(user_vec, item_vecs_all)
        scores[:, PAD_IDX] = -1e9

        topk = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
        target_np = target.cpu().numpy()
        for row, t in zip(topk, target_np):
            total += ndcg_at_k(row.tolist(), int(t), k=k)
            n += 1
    model.train()
    return total / max(n, 1)


# ========================================================================================
# 9. Training loop
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

    model = UniSRecRanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
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
    best_path = os.path.join(cfg.out_dir, "unisrec_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, total_main, total_cl, n_batches = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            uid_idx = batch["uid_idx"].to(device)
            hist = batch["hist"].to(device)
            hist_cnt = batch["hist_cnt"].to(device)
            hist_len = batch["hist_len"].to(device)
            target = batch["target"].to(device)

            user_vec = model(uid_idx, hist, hist_cnt, hist_len)

            # recompute the full-catalog item vectors every step (the MoE adaptor's
            # weights change each update, so these can't be cached across steps).
            all_item_vecs = model.item_static_vec(all_item_ids)  # [n_items, D]
            main_loss = full_softmax_loss(user_vec, target, all_item_vecs)

            cl_loss = model.contrastive_loss(hist, hist_cnt, hist_len, temperature=cfg.cl_temperature)
            loss = main_loss + cfg.cl_weight * cl_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total_loss += loss.item()
            total_main += main_loss.item()
            total_cl += cl_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_main = total_main / max(n_batches, 1)
        avg_cl = total_cl / max(n_batches, 1)

        with torch.no_grad():
            item_vecs_all = model.item_static_vec(all_item_ids)
        val_ndcg = evaluate_ndcg(model, val_loader, item_vecs_all, k=cfg.topk, device=device)

        print(f"[Epoch {epoch}/{cfg.epochs}] loss={avg_loss:.4f} (main={avg_main:.4f}, cl={avg_cl:.4f})  "
              f"valid_ndcg@{cfg.topk}={val_ndcg:.4f}")

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best model saved (ndcg@{cfg.topk}={best_ndcg:.4f})")

    print(f"[INFO] training done. best valid ndcg@{cfg.topk} = {best_ndcg:.4f}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, device


# ========================================================================================
# 10. Prediction
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
# 11. Synthetic data generator (for local smoke-testing when no real data is uploaded)
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
# 12. Main
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
