# -*- coding: utf-8 -*-
"""
SASRecF + CL4SRec-style contrastive learning for next-item / sequential
recommendation.

This is a drop-in architectural sibling of sasrecf_model.py: identical data
schema, identical encoders / SASRecF backbone / training & evaluation
harness / prediction / synthetic-data generator, but the training objective
is augmented with a CL4SRec-style self-supervised contrastive loss (Xie et
al., "Contrastive Learning for Sequential Recommendation", ICDE 2022), on
top of the existing full-softmax next-item loss.

Data schema
-----------
user.csv : uid,u_cat_01..u_cat_08                 (8 user categorical features, 0 is a VALID value)
item.csv : iid,i_cat_01,i_cat_02,i_cat_03,i_bucket_01   (item categorical features, 0 is a VALID value)
train.csv: uid,target_iid,item_seq_raw,item_seq_dedup,item_seq_counts
test.csv : uid,item_seq_raw,item_seq_dedup,item_seq_counts

Task
----
Same as sasrecf_model.py: predict the next item a user will interact with,
via a full-softmax ranking loss over the entire catalog. See that file for
the base architecture (causal self-attention over the history, feature
fusion of user/item side info, static item-scoring head).

Why add contrastive learning at all
--------------------------------------
CL4SRec's whole motivation is that supervised next-item prediction gives
only ONE training signal per sequence (predict target_iid from history),
which starves data-hungry Transformer encoders on small/sparse datasets.
Contrastive learning manufactures EXTRA, label-free training signal from the
exact same data: it builds two randomly-AUGMENTED views of each user's
history, pushes their representations (produced by the SAME encoder used
for the main task) together, and pushes every other sequence's
representations in the batch apart (InfoNCE / NT-Xent loss). This acts as a
regularizer that shapes the encoder's representation space using far more
comparisons per epoch than the single-target supervised loss alone provides
-- exactly the small-dataset pain point this whole model family was asked to
address.

Total training loss = full_softmax_loss (unchanged from sasrecf_model.py)
                       + cl_weight * info_nce_loss(view1, view2)

The two augmented views are encoded by SASRecF's OWN sequence encoder (the
same causal Transformer used for the main task), taking the pre-user,
pre-MLP sequence representation (`encode_sequence`) as the contrastive
representation -- exactly what CL4SRec's paper does (no extra projection
head), so the contrastive loss directly shapes the same encoder weights used
for ranking, rather than training a separate side-network.

Two INDEPENDENT uses of item_seq_counts in this file
--------------------------------------------------------
1. Inside the SASRecF backbone itself: identical to sasrecf_model.py --
   log1p(count) is projected and added to every history token's embedding
   (see SASRecF.encode_sequence).
2. NEW, CoSeRec-inspired use, inside the CONTRASTIVE AUGMENTATION pipeline:
   the "mask" (item-dropout) augmentation preferentially drops LOW-frequency
   items and preferentially KEEPS high-frequency ones (see
   `augment_mask_freq_guided`). The intuition (from CoSeRec, Liu et al.
   2021): randomly deleting a user's most it's revisited "core interest"
   items destroys exactly the signal the sequence encoder most needs to
   preserve across the two augmented views, making the contrastive task
   either trivially easy (both crippled views look equally bad) or actively
   harmful (a genuinely useful item is thrown away). Standard CL4SRec has no
   such frequency field to consult and drops items uniformly at random --
   this dataset's explicit item_seq_counts lets the augmentation be more
   surgical about which items are "safe" to remove.

Augmentations (following CL4SRec's three operators, adapted to a causal,
left-padded, DEDUPLICATED history representation)
-----------------------------------------------------------------------------
- **crop**: keep one random contiguous sub-window of the (deduplicated,
  chronologically-ordered) history.
- **mask**: randomly DROP a subset of items from the history (shortening it,
  then re-left-padding as usual). We drop rather than replace-in-place with
  a dedicated [MASK] id, because SASRecF's causal encoder treats item id 0
  strictly as an attention-excluded PAD token; replacing an item in the
  MIDDLE of the real history with id 0 would carve an attention "hole" and
  violate the "all real items are a contiguous suffix" invariant the
  positional embeddings and causal mask rely on. Dropping-and-re-padding
  keeps every architectural invariant intact.
- **reorder**: randomly shuffle the items within one contiguous sub-window,
  leaving items outside that window untouched.

Each augmented view independently samples ONE of these three operators
uniformly at random per CL4SRec's protocol (not all three composed
together).

Usage
-----
python sasrecf_cl4srec_model.py --data_dir /path/to/data --out_dir /path/to/output --epochs 5
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
    max_seq_len = 80          # truncate/pad history to this many distinct items (most recent kept)
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

    # CL4SRec contrastive learning
    cl_weight = 0.2          # lambda: weight of the contrastive loss relative to full_softmax_loss
    cl_temperature = 0.05     # tau: InfoNCE / NT-Xent temperature
    aug_crop_ratio = 0.6      # eta: fraction of the sequence kept by the "crop" augmentation
    aug_mask_ratio = 0.3      # gamma: fraction of items dropped by the "mask" augmentation
    aug_reorder_ratio = 0.2   # beta: fraction of the sequence shuffled by the "reorder" augmentation
    use_freq_guided_aug = True  # if True, "mask" preferentially drops LOW item_seq_counts items (CoSeRec-style)
    min_real_len_for_aug = 2   # sequences shorter than this are returned unaugmented (avoid degenerate crops/reorders)

    # training
    epochs = 20
    batch_size = 512
    lr = 1e-3
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
# 1. Vocab / encoding utilities  (identical to sasrecf_model.py)
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
# 2. Data loading  (identical to sasrecf_model.py)
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
# 3. CL4SRec-style data augmentation (operates on python lists of item ids /
#    counts BEFORE padding, exactly the (items, counts) representation
#    sasrecf_model.py's Dataset already builds from item_seq_dedup +
#    item_seq_counts)
# ========================================================================================

def augment_crop(items, counts, eta):
    """Keep one random contiguous sub-window covering ~eta of the sequence."""
    n = len(items)
    crop_len = max(1, int(round(n * eta)))
    if crop_len >= n:
        return list(items), list(counts)
    start = random.randint(0, n - crop_len)
    return items[start:start + crop_len], counts[start:start + crop_len]


def augment_mask_uniform(items, counts, gamma):
    """Randomly DROP a `gamma` fraction of items uniformly at random (see
    module docstring for why "drop" rather than "replace with a [MASK] id")."""
    n = len(items)
    n_drop = int(round(n * gamma))
    n_drop = min(n_drop, n - 1)  # never drop everything
    if n_drop <= 0:
        return list(items), list(counts)
    drop_idx = set(random.sample(range(n), n_drop))
    kept_items = [it for i, it in enumerate(items) if i not in drop_idx]
    kept_counts = [c for i, c in enumerate(counts) if i not in drop_idx]
    return kept_items, kept_counts


def augment_mask_freq_guided(items, counts, gamma):
    """CoSeRec-inspired variant of `augment_mask_uniform`: instead of
    dropping items uniformly at random, sample drop candidates with
    probability INVERSELY proportional to their item_seq_counts frequency,
    so frequently-revisited "core interest" items are preferentially kept
    and rarely-seen items are preferentially dropped -- this is the
    dedicated use of item_seq_counts inside the augmentation pipeline (see
    module docstring, use #2)."""
    n = len(items)
    n_drop = int(round(n * gamma))
    n_drop = min(n_drop, n - 1)
    if n_drop <= 0:
        return list(items), list(counts)
    counts_arr = np.asarray(counts, dtype=np.float64)
    weights = 1.0 / (counts_arr + 1.0)  # low count -> high drop weight
    weights = weights / weights.sum()
    drop_idx = set(np.random.choice(n, size=n_drop, replace=False, p=weights).tolist())
    kept_items = [it for i, it in enumerate(items) if i not in drop_idx]
    kept_counts = [c for i, c in enumerate(counts) if i not in drop_idx]
    return kept_items, kept_counts


def augment_reorder(items, counts, beta):
    """Shuffle the items within one random contiguous sub-window of length
    ~beta * n, leaving everything outside that window untouched. Counts are
    permuted together with their items since count is a per-item attribute."""
    n = len(items)
    win_len = max(2, int(round(n * beta)))
    if win_len >= n:
        win_len = n
    start = random.randint(0, n - win_len)
    items = list(items)
    counts = list(counts)
    idx = list(range(start, start + win_len))
    perm = idx[:]
    random.shuffle(perm)
    new_items, new_counts = list(items), list(counts)
    for dst, src in zip(idx, perm):
        new_items[dst] = items[src]
        new_counts[dst] = counts[src]
    return new_items, new_counts


def random_augment_view(items, counts, cfg: "Config"):
    """Samples ONE of {crop, mask, reorder} uniformly at random and applies
    it, following CL4SRec's protocol of using a single random operator per
    augmented view (not composing all three). Sequences shorter than
    `cfg.min_real_len_for_aug` are returned unaugmented since crop/mask/
    reorder are degenerate (or destructive) on trivially short histories."""
    if len(items) < cfg.min_real_len_for_aug:
        return list(items), list(counts)
    op = random.choice(("crop", "mask", "reorder"))
    if op == "crop":
        return augment_crop(items, counts, cfg.aug_crop_ratio)
    elif op == "mask":
        if cfg.use_freq_guided_aug:
            return augment_mask_freq_guided(items, counts, cfg.aug_mask_ratio)
        return augment_mask_uniform(items, counts, cfg.aug_mask_ratio)
    else:
        return augment_reorder(items, counts, cfg.aug_reorder_ratio)


# ========================================================================================
# 4. Dataset  (extends sasrecf_model.py's SasRecDataset with two augmented
#    views per training sample, used only by the contrastive loss)
# ========================================================================================

def _encode_fixed_len(items, counts, max_len, iid_enc):
    """Shared left-padding logic used for the main history AND for both
    augmented views, identical in spirit to sasrecf_model.py's
    SasRecDataset.__getitem__."""
    items = items[-max_len:]
    counts = counts[-max_len:]
    hist_idx = [iid_enc.transform_one(i) for i in items]
    hist_len = len(hist_idx)
    pad_n = max_len - hist_len
    hist_idx = [PAD_IDX] * pad_n + hist_idx
    hist_cnt = [0] * pad_n + counts
    return (
        np.array(hist_idx, dtype=np.int64),
        np.array(hist_cnt, dtype=np.float32),
        hist_len,
    )


class SasRecCLDataset(Dataset):
    """Like sasrecf_model.py's SasRecDataset, but when `training_mode=True`
    ALSO returns two independently-augmented views (aug1_*/aug2_*) of the
    same underlying (items, counts) sequence for the contrastive loss.
    Augmentations are re-sampled fresh every time __getitem__ is called, so
    each epoch sees new random augmentations (standard CL4SRec practice --
    the augmentation is "online", not precomputed once).

    `training_mode` is deliberately separate from `has_target`: validation
    and test sets still need `has_target` to control whether a label column
    exists, but they never need augmented views (contrastive learning is a
    training-only auxiliary objective), so callers should pass
    `training_mode=False` for val/test loaders even when has_target=True
    (as evaluate_ndcg needs targets but not augmentations).
    """

    def __init__(self, df, uid_enc, iid_enc, max_len, has_target=True,
                 training_mode=False, cfg: "Config" = None):
        self.uids = df["uid"].tolist()
        self.dedup_seqs = df["item_seq_dedup"].tolist()
        self.count_strs = df["item_seq_counts"].tolist()
        self.has_target = has_target
        if has_target:
            self.targets = df["target_iid"].tolist()
        self.uid_enc = uid_enc
        self.iid_enc = iid_enc
        self.max_len = max_len
        self.training_mode = training_mode
        self.cfg = cfg

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid_idx = self.uid_enc.transform_one(self.uids[idx])
        items = parse_seq_raw(self.dedup_seqs[idx])
        counts_map = parse_seq_counts(self.count_strs[idx])
        counts = [counts_map.get(i, 1) for i in items]

        hist_idx, hist_cnt, hist_len = _encode_fixed_len(items, counts, self.max_len, self.iid_enc)
        sample = {
            "uid_idx": uid_idx,
            "hist": hist_idx,
            "hist_cnt": hist_cnt,
            "hist_len": hist_len,
        }
        if self.has_target:
            sample["target"] = self.iid_enc.transform_one(self.targets[idx])

        if self.training_mode:
            items1, counts1 = random_augment_view(items, counts, self.cfg)
            items2, counts2 = random_augment_view(items, counts, self.cfg)
            a1_idx, a1_cnt, a1_len = _encode_fixed_len(items1, counts1, self.max_len, self.iid_enc)
            a2_idx, a2_cnt, a2_len = _encode_fixed_len(items2, counts2, self.max_len, self.iid_enc)
            sample["aug1_hist"], sample["aug1_cnt"], sample["aug1_len"] = a1_idx, a1_cnt, a1_len
            sample["aug2_hist"], sample["aug2_cnt"], sample["aug2_len"] = a2_idx, a2_cnt, a2_len
        return sample


def collate_fn(batch):
    """Generic collate: works for both plain samples (hist/hist_cnt/hist_len
    [+target]) and CL-augmented samples (also aug1_*/aug2_*), based on key
    naming convention -- so the exact same function serves train (with
    augmentations), valid, and test (without) loaders."""
    out = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        if key in ("uid_idx", "target") or key.endswith("_len"):
            out[key] = torch.tensor(vals, dtype=torch.long)
        elif key.endswith("_cnt"):
            out[key] = torch.tensor(np.stack(vals), dtype=torch.float32)
        else:  # 'hist', 'aug1_hist', 'aug2_hist'
            out[key] = torch.tensor(np.stack(vals), dtype=torch.long)
    return out


# ========================================================================================
# 5. SASRecF model  (identical backbone to sasrecf_model.py, with
#    `encode_sequence` factored out so the contrastive loss can reuse it
#    without going through the user-feature/MLP scoring head)
# ========================================================================================

class CausalTransformerBlock(nn.Module):
    """Pre-LN Transformer encoder block with CAUSAL (unidirectional) multi-head
    self-attention + position-wise feed-forward, each wrapped in a residual
    connection."""

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
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x


class SASRecF(nn.Module):
    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        emb_dim = cfg.emb_dim
        self.emb_dim = emb_dim
        self.max_seq_len = cfg.max_seq_len

        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=PAD_IDX)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=PAD_IDX)

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

        self.pos_emb = nn.Embedding(cfg.max_seq_len, item_full_dim)
        self.count_proj = nn.Linear(1, item_full_dim)

        ffn_dim = item_full_dim * cfg.ffn_mult
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(item_full_dim, cfg.n_heads, ffn_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.final_ln = nn.LayerNorm(item_full_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        mlp_in = user_full_dim + item_full_dim
        h1, h2 = cfg.mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, h1), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h1, h2), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h2, emb_dim),
        )

        self.register_buffer("user_feat_table", torch.zeros(n_users, 1, dtype=torch.long), persistent=False)
        self.register_buffer("item_feat_table", torch.zeros(n_items, 1, dtype=torch.long), persistent=False)
        self._causal_mask_cache = {}

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.user_feat_table = user_feat_table
        self.item_feat_table = item_feat_table

    def _item_full_emb(self, item_idx):
        base = self.item_emb(item_idx)
        feats = self.item_feat_table[item_idx]
        side_parts = [emb(feats[..., j]) for j, emb in enumerate(self.item_side_embs)]
        return torch.cat([base] + side_parts, dim=-1)

    def _user_full_emb(self, uid_idx):
        base = self.user_emb(uid_idx)
        feats = self.user_feat_table[uid_idx]
        side_parts = [emb(feats[..., j]) for j, emb in enumerate(self.user_side_embs)]
        return torch.cat([base] + side_parts, dim=-1)

    def _causal_mask(self, L, device):
        key = (L, device)
        mask = self._causal_mask_cache.get(key)
        if mask is None:
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)
            self._causal_mask_cache[key] = mask
        return mask

    def encode_sequence(self, hist, hist_cnt, hist_len):
        """Runs the causal Transformer over a history tensor and returns the
        output at the last (most-recent) position -- the PURE sequence
        representation, with NO user-feature or task-specific MLP mixed in
        yet. Used both by `encode_user` (for the main ranking task) and
        directly by the training loop (for the contrastive loss on
        augmented views), so both branches share literally the same
        encoder weights."""
        B, L = hist.shape
        mask = (hist != PAD_IDX)

        hist_full = self._item_full_emb(hist)
        hist_freq = self.count_proj(torch.log1p(hist_cnt).unsqueeze(-1))
        hist_freq = hist_freq * mask.unsqueeze(-1).float()

        seq = hist_full + hist_freq
        pos_ids = torch.arange(L, device=hist.device).unsqueeze(0).expand(B, -1)
        seq = seq + self.pos_emb(pos_ids)
        seq = self.emb_dropout(seq)

        causal_mask = self._causal_mask(L, hist.device)
        key_padding_mask = ~mask

        x = seq
        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)
            # see sasrecf_model.py's fix note: sanitize padded positions with
            # masked_fill (not multiplication) after every block, since a
            # fully-masked query row can produce NaN via PyTorch's fused
            # attention kernel in eval() mode, and 0 * NaN is still NaN.
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = self.final_ln(x)

        return x[:, -1, :]  # [B, item_full_dim]

    def encode_user(self, uid_idx, hist, hist_cnt, hist_len):
        seq_repr = self.encode_sequence(hist, hist_cnt, hist_len)
        user_full = self._user_full_emb(uid_idx)
        x_cat = torch.cat([user_full, seq_repr], dim=-1)
        user_vec = self.mlp(x_cat)
        return user_vec


class SASRecFRanker(nn.Module):
    """Wraps SASRecF: trains with a FULL softmax over the entire item catalog
    (identical to sasrecf_model.py), and additionally exposes
    `encode_sequence_only` for the contrastive loss branch."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        self.sasrecf = SASRecF(n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg)
        emb_dim = cfg.emb_dim
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))
        item_full_dim = emb_dim + side_dim * len(item_feat_vocabs)
        self.item_score_head = nn.Linear(item_full_dim, emb_dim)
        self.n_items = n_items

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.sasrecf.set_feature_tables(user_feat_table, item_feat_table)

    def item_static_vec(self, item_idx):
        full = self.sasrecf._item_full_emb(item_idx)
        return self.item_score_head(full)

    def forward(self, uid_idx, hist, hist_cnt, hist_len):
        return self.sasrecf.encode_user(uid_idx, hist, hist_cnt, hist_len)

    def encode_sequence_only(self, hist, hist_cnt, hist_len):
        """Used only for the contrastive loss: returns the pure sequence
        representation (item_full_dim), skipping the user-feature
        concatenation and final task MLP."""
        return self.sasrecf.encode_sequence(hist, hist_cnt, hist_len)

    def score_against_catalog(self, user_vec, item_vecs):
        return user_vec @ item_vecs.t()


# ========================================================================================
# 6. Losses: full softmax (main task) + InfoNCE (contrastive)
# ========================================================================================

def full_softmax_loss(user_vec, target_idx, all_item_vecs):
    """Identical to sasrecf_model.py: cross-entropy against the FULL catalog,
    with PAD masked out."""
    logits = user_vec @ all_item_vecs.t()
    logits = logits.clone()
    logits[:, PAD_IDX] = -1e9
    loss = F.cross_entropy(logits, target_idx)
    return loss


def info_nce_loss(z1, z2, temperature):
    """Standard NT-Xent / InfoNCE loss (SimCLR / CL4SRec style), vectorized.

    z1, z2: [B, D] representations of the two augmented views of the SAME B
    underlying sequences. For each of the 2B rows (both views of every
    sequence stacked together), the positive is its OTHER view; every other
    row (both views of every OTHER sequence) is a negative. Cosine
    similarity / temperature, cross-entropy against the known positive
    index -- this is exactly CL4SRec's contrastive objective.
    """
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    sim = z @ z.t() / temperature  # [2B, 2B]
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)  # never let a row match itself

    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = torch.where(pos_idx < B, pos_idx + B, pos_idx - B)  # view i's positive is its twin view

    return F.cross_entropy(sim, pos_idx)


# ========================================================================================
# 7. NDCG@10 metric  (identical to sasrecf_model.py)
# ========================================================================================

def ndcg_at_k(ranked_item_ids, true_item_id, k=10):
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
# 8. Training loop
# ========================================================================================

def train_model(bundle, cfg: "Config", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    train_full = bundle.train_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    n_val = int(len(train_full) * cfg.val_frac)
    val_df = train_full.iloc[:n_val].reset_index(drop=True)
    tr_df = train_full.iloc[n_val:].reset_index(drop=True)
    print(f"[INFO] train={len(tr_df)}  valid={len(val_df)}")

    # training_mode=True -> also produce the two augmented views needed by
    # the contrastive loss. Validation never augments.
    train_ds = SasRecCLDataset(tr_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len,
                                has_target=True, training_mode=True, cfg=cfg)
    val_ds = SasRecCLDataset(val_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len,
                              has_target=True, training_mode=False, cfg=cfg)

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
    best_path = os.path.join(cfg.out_dir, "sasrecf_cl4srec_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_main, total_cl, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            uid_idx = batch["uid_idx"].to(device)
            hist = batch["hist"].to(device)
            hist_cnt = batch["hist_cnt"].to(device)
            hist_len = batch["hist_len"].to(device)
            target = batch["target"].to(device)
            aug1_hist = batch["aug1_hist"].to(device)
            aug1_cnt = batch["aug1_cnt"].to(device)
            aug1_len = batch["aug1_len"].to(device)
            aug2_hist = batch["aug2_hist"].to(device)
            aug2_cnt = batch["aug2_cnt"].to(device)
            aug2_len = batch["aug2_len"].to(device)

            user_vec = model(uid_idx, hist, hist_cnt, hist_len)
            all_item_vecs = model.item_static_vec(all_item_ids)
            main_loss = full_softmax_loss(user_vec, target, all_item_vecs)

            z1 = model.encode_sequence_only(aug1_hist, aug1_cnt, aug1_len)
            z2 = model.encode_sequence_only(aug2_hist, aug2_cnt, aug2_len)
            cl_loss = info_nce_loss(z1, z2, cfg.cl_temperature)

            loss = main_loss + cfg.cl_weight * cl_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total_main += main_loss.item()
            total_cl += cl_loss.item()
            n_batches += 1

        avg_main = total_main / max(n_batches, 1)
        avg_cl = total_cl / max(n_batches, 1)

        with torch.no_grad():
            item_vecs_all = model.item_static_vec(all_item_ids)
        val_ndcg = evaluate_ndcg(model, val_loader, item_vecs_all, k=cfg.topk, device=device)

        print(f"[Epoch {epoch}/{cfg.epochs}] main_loss={avg_main:.4f}  cl_loss={avg_cl:.4f}  "
              f"valid_ndcg@{cfg.topk}={val_ndcg:.4f}")

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best model saved (ndcg@{cfg.topk}={best_ndcg:.4f})")

    print(f"[INFO] training done. best valid ndcg@{cfg.topk} = {best_ndcg:.4f}")
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, device


# ========================================================================================
# 9. Prediction  (no augmentation at inference time -- identical to sasrecf_model.py)
# ========================================================================================

@torch.no_grad()
def run_predict(model, bundle, cfg: "Config", device):
    test_df = bundle.test_df
    if test_df is None:
        print("[WARN] no test.csv found, skipping prediction.")
        return

    model.eval()
    ds = SasRecCLDataset(test_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len,
                          has_target=False, training_mode=False, cfg=cfg)
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
# 10. Synthetic data generator (identical to sasrecf_model.py)
# ========================================================================================

def make_synthetic_data(data_dir, n_users=500, n_items=300, n_train=3000, n_test=200):
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    uids = [f"u{str(i).zfill(6)}" for i in range(1, n_users + 1)]
    iids = [f"i{str(i).zfill(6)}" for i in range(1, n_items + 1)]

    user_df = pd.DataFrame({"uid": uids})
    for c in [f"u_cat_{str(i).zfill(2)}" for i in range(1, 9)]:
        user_df[c] = rng.integers(0, 20, size=n_users)
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
# 11. Main
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
