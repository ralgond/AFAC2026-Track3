# -*- coding: utf-8 -*-
"""
MIND (Multi-Interest Network with Dynamic Routing) for next-item / sequential
recommendation.

This script mirrors the data pipeline, training loop, NDCG@10 evaluation and
prediction/submission format of the accompanying DIN script, but replaces
DIN's candidate-aware local activation unit with MIND's two core ideas:

  1. Multi-Interest Extractor Layer (B2I dynamic routing, "capsule network"):
     the user's item-history embeddings (a "behavior" layer of capsules) are
     routed, via an EM-like iterative procedure, into K "interest" capsules.
     Each interest capsule is a vector summarizing one facet of the user's
     interests (e.g. one capsule might capture "electronics", another
     "books"), instead of DIN's single attention-pooled vector.

  2. Label-aware attention: at TRAINING time, the K interest capsules are
     combined into a single user vector via a softmax attention over the
     capsules keyed on the (known) target item embedding -- the capsule most
     aligned with the true target is up-weighted. At EVAL/INFERENCE time the
     target is unknown, so each of the K capsules is scored independently
     against the full catalog and we keep, for every item, the BEST
     (max-over-interests) score across the K capsules -- this is the standard
     multi-interest retrieval/ranking inference scheme used by MIND.

Data schema (identical to the DIN script)
------------------------------------------
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
the target item the user will interact with next. Training uses a FULL
softmax loss over the entire item catalog (cross-entropy against every item,
no negative sampling), matching the candidate space used at evaluation time
(ranking the full catalog for NDCG@10).

Output
------
- Training prints valid NDCG@10 each epoch.
- run_predict produces submission.csv with columns:
    uid,prediction
  where prediction is a comma-quoted string of top-10 item ids, e.g.:
    u000009,"i001952,i001038,i001710,i001046,i000401,i001445,i001069,i001002,i001673,i000661"

item_seq_counts is used as a per-history-item frequency weight: log1p(count)
scales each history item's embedding before it enters the dynamic-routing
capsule layer, so items the user has repeatedly interacted with carry more
"mass" when routed into interest capsules than items seen only once -- we
switch to the deduplicated sequence (item_seq_dedup) as the set of distinct
history items, paired with their counts, exactly as in the DIN script.

Usage
-----
python mind_model.py --data_dir /path/to/data --out_dir /path/to/output --epochs 5
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
    num_interests = 2       # K: number of interest capsules extracted per user (MIND's core hyperparameter)
    routing_iters = 3       # number of dynamic-routing iterations (B2I routing)
    mlp_hidden = (200, 80)  # final interest-projection MLP hidden sizes
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
# 1. Vocab / encoding utilities  (identical to DIN script)
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
# 2. Data loading  (identical to DIN script)
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
# 3. Dataset  (identical to DIN script -- MIND consumes the exact same fields)
# ========================================================================================

class MINDDataset(Dataset):
    """
    Each sample: (uid_idx, hist_item_idx[seq_len], hist_count[seq_len], hist_len, target_item_idx)

    History is built from `item_seq_dedup` (distinct items, in first-seen order)
    paired with their frequency from `item_seq_counts`. The count signal is fed
    into the multi-interest extractor as a multiplicative weight on each history
    item's embedding BEFORE dynamic routing, so items the user interacted with
    many times contribute proportionally more "vote mass" toward whichever
    interest capsule they get routed to than one-off interactions.

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
        # items (i.e. the tail of the dedup list), matching the truncation policy
        # used for the raw sequence in the original design.
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
# 4. MIND model
# ========================================================================================

class MultiInterestExtractor(nn.Module):
    """B2I (Behavior-to-Interest) dynamic routing layer, the heart of MIND.

    Treats each historical item embedding as a "low-level capsule" (a vote)
    and iteratively routes these votes into K "high-level capsules" (interest
    vectors), following the same iterative procedure as CapsNet's dynamic
    routing, adapted by MIND for sequential recommendation:

      1. Each behavior capsule i casts a vote for each interest capsule k via
         a SHARED, randomly-initialized (but learnable) bilinear mapping
         matrix S: vote_{i->k} = S @ hist_emb_i  (MIND ties S across i and k
         to keep parameter count low and to be length-invariant).
      2. Coupling coefficients c_{i,k} = softmax_k(routing_logits_{i,k}) decide
         how much of behavior-capsule i's vote goes to interest-capsule k.
      3. Interest capsule k = squash(sum_i c_{i,k} * vote_{i->k}), where squash
         is CapsNet's non-linear "squashing" function that keeps capsule
         vectors at norm < 1 while preserving direction.
      4. routing_logits_{i,k} are then updated by agreement: routing_logits_{i,k}
         += vote_{i->k} . interest_k, and the loop repeats for `routing_iters`
         iterations.

    Padding positions are masked out of routing entirely (their logits are
    fixed at -inf so they receive 0 coupling weight, but unlike DIN they are
    not given a count-based attention adjustment after the fact -- instead the
    raw history embeddings are pre-scaled by log1p(count) before routing, so
    repeated items cast a proportionally stronger "vote").
    """

    def __init__(self, dim, num_interests, routing_iters=3):
        super().__init__()
        self.dim = dim
        self.K = num_interests
        self.iters = routing_iters
        # shared bilinear routing matrix S (dim x dim), as in the MIND paper
        self.S = nn.Parameter(torch.randn(dim, dim) * (1.0 / math.sqrt(dim)))

    @staticmethod
    def squash(x, dim=-1, eps=1e-9):
        # CapsNet squashing non-linearity: ||x|| -> [0, 1), direction preserved
        sq_norm = (x * x).sum(dim=dim, keepdim=True)
        scale = sq_norm / (1.0 + sq_norm)
        return scale * x / torch.sqrt(sq_norm + eps)

    def forward(self, hist_emb, mask):
        """
        hist_emb: [B, L, D]  (already count-weighted item embeddings)
        mask:     [B, L]     (1 = real item, 0 = pad)
        returns:  interests [B, K, D], interest_mask [B, K] (which capsules are "active")
        """
        B, L, D = hist_emb.shape
        K = self.K

        # votes_{i->k} = hist_emb_i @ S  (same projected vote used for every k,
        # as in MIND's shared-S formulation): [B, L, D]
        votes = hist_emb @ self.S  # [B, L, D]

        # routing logits b_{i,k}, initialized to 0 (uniform prior), masked positions -> -inf
        logits = hist_emb.new_zeros(B, L, K)
        neg_inf_mask = (mask == 0).unsqueeze(-1).expand(B, L, K)
        logits = logits.masked_fill(neg_inf_mask, -1e9)

        interests = hist_emb.new_zeros(B, K, D)
        for it in range(self.iters):
            c = torch.softmax(logits, dim=2)  # coupling coeffs over K, per behavior capsule: [B, L, K]
            c = c * mask.unsqueeze(-1)        # zero out padded behaviors entirely
            # weighted sum of votes per interest capsule: [B, K, D]
            s = torch.einsum("blk,bld->bkd", c, votes)
            interests = self.squash(s, dim=-1)  # [B, K, D]
            if it < self.iters - 1:
                # agreement update: how well does each vote align with the
                # resulting interest capsule -> reinforce that routing
                agreement = torch.einsum("bld,bkd->blk", votes, interests)  # [B, L, K]
                logits = logits + agreement
                logits = logits.masked_fill(neg_inf_mask, -1e9)

        # an interest capsule is "active" for a user only if they have at least
        # one real history item to route into it (matters for very short
        # histories where K may exceed the number of distinct interests).
        n_valid = mask.sum(dim=1)  # [B]
        interest_mask = (n_valid > 0).unsqueeze(-1).expand(B, K).float()
        return interests, interest_mask


class MIND(nn.Module):
    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        """
        user_feat_vocabs: list[int] vocab sizes for each u_cat_* column
        item_feat_vocabs: list[int] vocab sizes for each i_cat_*/i_bucket_* column
        """
        super().__init__()
        emb_dim = cfg.emb_dim
        self.emb_dim = emb_dim
        self.K = cfg.num_interests

        # core id embeddings
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=PAD_IDX)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=PAD_IDX)  # used both as history & candidate emb

        # side feature embeddings (smaller dim each, concatenated)
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

        # multi-interest extractor operates purely on item-side ("behavior") embeddings
        self.extractor = MultiInterestExtractor(item_full_dim, cfg.num_interests, cfg.routing_iters)

        # project [user_full ; interest_capsule] -> emb_dim so we can do a plain
        # dot product against the (also emb_dim) full-catalog item vectors.
        mlp_in = user_full_dim + item_full_dim
        h1, h2 = cfg.mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, h1), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h1, h2), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h2, emb_dim),
        )

        # registered buffers filled in by set_feature_tables()
        self.register_buffer("user_feat_table", torch.zeros(n_users, 1, dtype=torch.long), persistent=False)
        self.register_buffer("item_feat_table", torch.zeros(n_items, 1, dtype=torch.long), persistent=False)

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

    def extract_interests(self, hist, hist_cnt):
        """Runs dynamic routing and returns the K raw interest capsules
        (in item_full_dim space) plus their validity mask.
        hist_cnt: [B, L] raw item_seq_counts values (0 on padding) used to
        pre-scale each history item's embedding by log1p(count) before it
        is routed -- frequently-interacted items cast stronger votes."""
        mask = (hist != PAD_IDX).float()  # [B, L]
        hist_full = self._item_full_emb(hist)  # [B, L, item_full_dim]
        weight = torch.log1p(hist_cnt).unsqueeze(-1) * mask.unsqueeze(-1)  # [B, L, 1], 0 on pad
        # guard against an all-zero weight row collapsing routing for a real
        # item (log1p(count) could be 0 if count was 0 for a non-pad slot);
        # ensure every real position contributes at least weight 1.
        weight = torch.where(mask.unsqueeze(-1) > 0, weight.clamp(min=1.0), weight)
        hist_weighted = hist_full * weight  # [B, L, item_full_dim]

        interests, interest_mask = self.extractor(hist_weighted, mask)  # [B, K, item_full_dim], [B, K]
        return interests, interest_mask

    def user_vectors_all_interests(self, uid_idx, hist, hist_cnt):
        """Projects every interest capsule (concatenated with user side-features)
        into the shared emb_dim scoring space. Used both for label-aware
        attention at train time and for max-over-interests scoring at eval time.
        Returns: user_vecs [B, K, emb_dim], interest_mask [B, K]
        """
        interests, interest_mask = self.extract_interests(hist, hist_cnt)  # [B, K, item_full_dim]
        user_full = self._user_full_emb(uid_idx)  # [B, user_full_dim]
        B, K, _ = interests.shape
        user_full_exp = user_full.unsqueeze(1).expand(-1, K, -1)  # [B, K, user_full_dim]
        x = torch.cat([user_full_exp, interests], dim=-1)  # [B, K, mlp_in]
        user_vecs = self.mlp(x)  # [B, K, emb_dim]
        return user_vecs, interest_mask

    def encode_user_label_aware(self, uid_idx, hist, hist_cnt, target_item_idx):
        """TRAINING path: label-aware attention over the K interest vectors,
        keyed on the (known) target item's embedding -- MIND's mechanism for
        collapsing K capsules into a single vector to compute a loss against.
        The capsule most aligned with the true target gets (softly) selected,
        which is what teaches different capsules to specialize on different
        facets of user interest in the first place."""
        user_vecs, interest_mask = self.user_vectors_all_interests(uid_idx, hist, hist_cnt)  # [B, K, D]
        target_vec = self.item_score_vec(target_item_idx)  # [B, D]  (static scoring-space item vector)

        # attention logits = scaled dot product between each interest vector and target
        D = user_vecs.size(-1)
        logits = torch.einsum("bkd,bd->bk", user_vecs, target_vec) / math.sqrt(D)  # [B, K]
        logits = logits.masked_fill(interest_mask == 0, -1e9)
        attn = torch.softmax(logits, dim=-1)  # [B, K]
        user_vec = torch.einsum("bk,bkd->bd", attn, user_vecs)  # [B, D]
        return user_vec

    def item_score_vec(self, item_idx):
        raise NotImplementedError  # implemented on the wrapping Ranker (needs item_score_head)


class MINDRanker(nn.Module):
    """Wraps MIND: trains with a FULL softmax over the entire item catalog.

    Training uses label-aware attention (collapsing the K interest capsules
    into one vector using the known target, see encode_user_label_aware)
    against a static per-item scoring embedding (item_score_head), exactly
    mirroring the DIN script's "static item vector trained jointly with the
    attended user vector" design -- but here the *user* side is multi-vector
    (K interests) rather than single-vector.

    At INFERENCE/EVAL time the target is unknown, so each of the K interest
    vectors is dot-producted against the full catalog independently and we
    take, for every candidate item, the MAX score across the K interests
    (the standard "best matching interest wins" retrieval rule used by MIND).
    """

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        self.mind = MIND(n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg)
        emb_dim = cfg.emb_dim
        item_full_dim = self.mind.item_full_dim
        self.item_score_head = nn.Linear(item_full_dim, emb_dim)  # static item vector for full-catalog scoring
        self.n_items = n_items
        self.K = cfg.num_interests

        # bind item_score_vec onto the inner MIND module so encode_user_label_aware can call it
        self.mind.item_score_vec = self.item_static_vec

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.mind.set_feature_tables(user_feat_table, item_feat_table)

    def item_static_vec(self, item_idx):
        full = self.mind._item_full_emb(item_idx)
        return self.item_score_head(full)

    def forward(self, uid_idx, hist, hist_cnt, hist_len, target_idx):
        """Training-time forward: returns a single [B, D] user vector obtained
        via label-aware attention over the K interest capsules."""
        user_vec = self.mind.encode_user_label_aware(uid_idx, hist, hist_cnt, target_idx)
        return user_vec

    def user_vectors_all_interests(self, uid_idx, hist, hist_cnt):
        """Inference-time path: returns all K per-user interest vectors
        (no label-aware collapsing, since the target is unknown)."""
        return self.mind.user_vectors_all_interests(uid_idx, hist, hist_cnt)

    def score_against_catalog(self, user_vec, item_vecs):
        # user_vec: [B, D], item_vecs: [N, D] -> [B, N]
        return user_vec @ item_vecs.t()

    def score_against_catalog_multi(self, user_vecs, interest_mask, item_vecs):
        """user_vecs: [B, K, D], interest_mask: [B, K], item_vecs: [N, D]
        -> [B, N] scores, taking the max over the K interests for every item
        (inactive/padded interest capsules are excluded via -inf masking)."""
        # [B, K, N]
        scores = torch.einsum("bkd,nd->bkn", user_vecs, item_vecs)
        mask = (interest_mask == 0).unsqueeze(-1)  # [B, K, 1]
        scores = scores.masked_fill(mask, -1e9)
        return scores.max(dim=1).values  # [B, N]


# ========================================================================================
# 5. Training: full softmax over the entire item catalog
# ========================================================================================

def full_softmax_loss(user_vec, target_idx, all_item_vecs):
    """
    user_vec: [B, D]
    target_idx: [B]  (true next-item index for each sample)
    all_item_vecs: [n_items, D]  static item vectors for the ENTIRE catalog (index 0 = PAD)

    Computes logits against every item in the catalog (no negative sampling),
    masks out the PAD index so the model never learns to score it, and applies
    standard cross-entropy with the true target as the label. This removes the
    train/eval distribution mismatch that sampled softmax introduces (training
    only ever saw a handful of random negatives, while evaluation ranks
    against the full catalog).
    """
    logits = user_vec @ all_item_vecs.t()  # [B, n_items]
    logits = logits.clone()
    logits[:, PAD_IDX] = -1e9  # never let the model assign probability mass to PAD
    loss = F.cross_entropy(logits, target_idx)
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
        target = batch["target"].to(device)

        # at eval time the true target is unknown, so we cannot use label-aware
        # attention to collapse the K interest capsules into one vector. We
        # instead score every interest capsule against the full catalog
        # independently and keep, per item, the best (max) score across
        # interests -- MIND's standard multi-interest retrieval/ranking rule.
        user_vecs, interest_mask = model.user_vectors_all_interests(uid_idx, hist, hist_cnt)  # [B, K, D]
        scores = model.score_against_catalog_multi(user_vecs, interest_mask, item_vecs_all)  # [B, n_items]
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

    train_ds = MINDDataset(tr_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=True)
    val_ds = MINDDataset(val_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    user_feat_vocabs = [len(bundle.u_encoders[c]) for c in bundle.u_cat_cols]
    item_feat_vocabs = [len(bundle.i_encoders[c]) for c in bundle.i_cat_cols]

    model = MINDRanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
    model.set_feature_tables(
        torch.tensor(bundle.user_feat, dtype=torch.long),
        torch.tensor(bundle.item_feat, dtype=torch.long),
    )
    model.to(device)
    model.mind.user_feat_table = model.mind.user_feat_table.to(device)
    model.mind.item_feat_table = model.mind.item_feat_table.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    all_item_ids = torch.arange(bundle.n_items, device=device)

    best_ndcg = -1.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "mind_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            uid_idx = batch["uid_idx"].to(device)
            hist = batch["hist"].to(device)
            hist_cnt = batch["hist_cnt"].to(device)
            hist_len = batch["hist_len"].to(device)
            target = batch["target"].to(device)

            # TRAINING uses the known target to drive label-aware attention
            # over the K interest capsules (standard MIND training trick --
            # exactly analogous to DIN feeding the target into its candidate-
            # aware activation unit during training).
            user_vec = model(uid_idx, hist, hist_cnt, hist_len, target)

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
    ds = MINDDataset(test_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    all_item_ids = torch.arange(bundle.n_items, device=device)
    item_vecs_all = model.item_static_vec(all_item_ids)

    rows = []
    uids = test_df["uid"].tolist()
    for batch in loader:
        uid_idx = batch["uid_idx"].to(device)
        hist = batch["hist"].to(device)
        hist_cnt = batch["hist_cnt"].to(device)

        user_vecs, interest_mask = model.user_vectors_all_interests(uid_idx, hist, hist_cnt)  # [B, K, D]
        scores = model.score_against_catalog_multi(user_vecs, interest_mask, item_vecs_all)  # [B, n_items]
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
