# -*- coding: utf-8 -*-
"""
DIN (Deep Interest Network) for next-item / sequential recommendation.

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
target item the user will interact with next. This is cast as a candidate-ranking
problem: DIN scores (user, history, candidate_item) triples. Training uses a
PAIRWISE ranking loss (BPR: Bayesian Personalized Ranking) -- for each positive
(user, target_item) pair we sample `num_neg` random negative items from the
catalog and push the positive item's score above each negative's score via
loss = -log(sigmoid(pos_score - neg_score)). This directly optimizes for the
relative ordering of items (what NDCG@10 actually measures) rather than for a
full probability distribution over the catalog, and is far cheaper per step
than scoring/normalizing over every item in the catalog.

Output
------
- Training prints valid NDCG@10 each epoch.
- predict.py-equivalent (run_predict) produces submission.csv with columns:
    uid,prediction
  where prediction is a comma-quoted string of top-10 item ids, e.g.:
    u000009,"i001952,i001038,i001710,i001046,i000401,i001445,i001069,i001002,i001673,i000661"

item_seq_counts is used as a per-history-item frequency weight: we feed
log1p(count) into the DIN activation unit as an extra interaction signal, and
also use it to scale the attention-pooled interest vector, so items a user has
repeatedly interacted with carry more weight in the user representation than
items seen only once -- we switch to the deduplicated sequence (item_seq_dedup)
as the set of distinct history items, paired with their counts.

Usage
-----
python din_model.py --data_dir /path/to/data --out_dir /path/to/output --epochs 5
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
    attn_hidden = (80, 40)  # activation-unit MLP hidden sizes
    mlp_hidden = (200, 80)  # final user-representation MLP hidden sizes
    dropout = 0.2

    # training
    epochs = 30
    batch_size = 512
    lr = 1e-3
    grad_clip = 5.0
    seed = 42
    num_neg = 5             # number of sampled negatives per positive for the pairwise (BPR) loss

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
# 2. Data loading
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
# 3. Dataset
# ========================================================================================

class DINDataset(Dataset):
    """
    Each sample: (uid_idx, hist_item_idx[seq_len], hist_count[seq_len], hist_len, target_item_idx)

    History is built from `item_seq_dedup` (distinct items, in first-seen order)
    paired with their frequency from `item_seq_counts`. The count signal is fed
    into the model both as an extra interaction feature in the activation unit
    and as an explicit multiplicative weight on the attention-pooled interest,
    so items the user interacted with many times influence the user
    representation more than one-off interactions.

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
# 4. DIN model
# ========================================================================================

class ActivationUnit(nn.Module):
    """DIN's local activation unit: computes attention weight between a candidate
    item embedding and each historical item embedding, using the classic
    [hist, cand, hist-cand, hist*cand] interaction features fed through an MLP,
    PLUS an extra scalar feature: log1p(item_seq_counts) for that history item,
    giving the network direct access to "how many times has the user interacted
    with this item" when deciding how much attention to pay to it."""

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


class DIN(nn.Module):
    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        """
        user_feat_vocabs: list[int] vocab sizes for each u_cat_* column
        item_feat_vocabs: list[int] vocab sizes for each i_cat_*/i_bucket_* column
        """
        super().__init__()
        emb_dim = cfg.emb_dim
        self.emb_dim = emb_dim

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

        # activation unit operates in item_full_dim space (history items & candidate item)
        self.activation_unit = ActivationUnit(item_full_dim, hidden=cfg.attn_hidden)

        mlp_in = user_full_dim + item_full_dim + item_full_dim  # user feat + sum-pooled interest + candidate
        h1, h2 = cfg.mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, h1), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h1, h2), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h2, emb_dim),  # project to emb_dim so we can score vs. full item catalog via dot product
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

    def encode_user(self, uid_idx, hist, hist_cnt, hist_len, cand_item_idx):
        """Produces the user representation vector for scoring against item embeddings.
        cand_item_idx is used only to drive the DIN attention (one candidate per row,
        typically the positive target during training).
        hist_cnt: [B, L] raw item_seq_counts values (0 on padding) -- used both inside
        the activation unit (as a log-count feature) and as an explicit multiplicative
        weight on the softmax attention scores, so high-frequency history items
        contribute proportionally more to the pooled interest vector."""
        mask = (hist != PAD_IDX).long()  # [B, L]
        hist_full = self._item_full_emb(hist)  # [B, L, item_full_dim]
        cand_full = self._item_full_emb(cand_item_idx)  # [B, item_full_dim]

        attn_w = self.activation_unit(hist_full, cand_full, mask, hist_cnt)  # [B, L], softmax over valid positions

        # re-weight by interaction frequency (log1p to dampen extreme outlier counts),
        # then renormalize so the pooling remains a convex combination.
        freq_w = torch.log1p(hist_cnt) * mask.float()  # [B, L], 0 on pad
        combined_w = attn_w * (1.0 + freq_w)
        combined_w = combined_w / (combined_w.sum(dim=-1, keepdim=True) + 1e-9)

        interest = torch.bmm(combined_w.unsqueeze(1), hist_full).squeeze(1)  # [B, item_full_dim]

        user_full = self._user_full_emb(uid_idx)  # [B, user_full_dim]

        x = torch.cat([user_full, interest, cand_full], dim=-1)
        user_vec = self.mlp(x)  # [B, emb_dim]
        return user_vec

    def all_item_embeddings(self):
        """Returns full-catalog item embeddings for scoring at inference time.
        NOTE: candidate-dependent attention means there's no single fixed item
        embedding usable for ranking against ALL items without recomputation.
        We instead project item_full_emb -> emb_dim via a shared linear head so a
        plain dot-product over the whole catalog is possible at inference; this
        approximates DIN's attention with the item's own static representation."""
        raise NotImplementedError  # not used; see ScoreHead below


class DINRanker(nn.Module):
    """Wraps DIN: trains with a PAIRWISE (BPR) ranking loss against sampled negative
    items using the user vector from DIN's attention pooling (with candidate = the
    positive target during training, which is the standard trick used at training
    time), and a separate static item-scoring embedding (i_static) used to score
    the positive/negative items during training and ALL items at inference time,
    since true DIN attention is candidate-specific and we approximate full-catalog
    scoring via a learned static projection of item_full_emb that is trained
    jointly (via the pairwise loss) to be consistent with the attended user vector."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        self.din = DIN(n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg)
        emb_dim = cfg.emb_dim
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))
        item_full_dim = emb_dim + side_dim * len(item_feat_vocabs)
        self.item_score_head = nn.Linear(item_full_dim, emb_dim)  # static item vector for full-catalog scoring
        self.n_items = n_items

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.din.set_feature_tables(user_feat_table, item_feat_table)

    def item_static_vec(self, item_idx):
        full = self.din._item_full_emb(item_idx)
        return self.item_score_head(full)

    def forward(self, uid_idx, hist, hist_cnt, hist_len, target_idx):
        user_vec = self.din.encode_user(uid_idx, hist, hist_cnt, hist_len, target_idx)  # [B, D]
        return user_vec

    def score_against_catalog(self, user_vec, item_vecs):
        # user_vec: [B, D], item_vecs: [N, D] -> [B, N]
        return user_vec @ item_vecs.t()


# ========================================================================================
# 5. Training: pairwise (BPR) ranking loss
# ========================================================================================

def pairwise_bpr_loss(user_vec, target_idx, all_item_vecs, num_neg=5):
    """
    Bayesian Personalized Ranking (BPR) pairwise loss.

    user_vec: [B, D]
    target_idx: [B]  (true next-item index for each sample -- the positive item)
    all_item_vecs: [n_items, D]  static item vectors for the ENTIRE catalog (index 0 = PAD)
    num_neg: number of negative items sampled per positive example

    For each (user, positive_item) pair, `num_neg` negative items are sampled
    uniformly at random from the catalog (excluding PAD and, via resampling,
    excluding accidental collisions with the positive item itself). The loss
    pushes the positive item's score above each sampled negative's score:

        loss = -mean( log(sigmoid(score_pos - score_neg)) )

    This is the classic BPR objective: it only cares about the relative order
    of a positive vs. its negatives (matching the ranking metric NDCG@10 used
    for evaluation), and -- unlike full softmax -- its cost per step doesn't
    scale with the full catalog size, only with `num_neg`.
    """
    B = user_vec.size(0)
    n_items = all_item_vecs.size(0)
    device = user_vec.device

    pos_vec = all_item_vecs[target_idx]  # [B, D]
    pos_score = (user_vec * pos_vec).sum(dim=-1, keepdim=True)  # [B, 1]

    # sample negatives uniformly from the real catalog (indices 1..n_items-1, 0 is PAD),
    # resampling any that accidentally collide with the true positive item.
    neg_idx = torch.randint(1, n_items, (B, num_neg), device=device)
    collide = neg_idx == target_idx.unsqueeze(1)
    while collide.any():
        n_collide = int(collide.sum().item())
        neg_idx[collide] = torch.randint(1, n_items, (n_collide,), device=device)
        collide = neg_idx == target_idx.unsqueeze(1)

    neg_vec = all_item_vecs[neg_idx]  # [B, num_neg, D]
    neg_score = torch.bmm(neg_vec, user_vec.unsqueeze(-1)).squeeze(-1)  # [B, num_neg]

    diff = pos_score - neg_score  # [B, num_neg], broadcasts pos_score across negatives
    loss = -F.logsigmoid(diff).mean()
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

        # at eval time we don't know the true candidate in advance for attention
        # (using it would leak the label), so we drive attention with the most
        # recent history item as a stand-in candidate -- a common DIN inference
        # approximation. Short/empty sequences fall back to PAD_IDX gracefully.
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

    train_ds = DINDataset(tr_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=True)
    val_ds = DINDataset(val_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    user_feat_vocabs = [len(bundle.u_encoders[c]) for c in bundle.u_cat_cols]
    item_feat_vocabs = [len(bundle.i_encoders[c]) for c in bundle.i_cat_cols]

    model = DINRanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
    model.set_feature_tables(
        torch.tensor(bundle.user_feat, dtype=torch.long),
        torch.tensor(bundle.item_feat, dtype=torch.long),
    )
    model.to(device)
    model.din.user_feat_table = model.din.user_feat_table.to(device)
    model.din.item_feat_table = model.din.item_feat_table.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    all_item_ids = torch.arange(bundle.n_items, device=device)

    best_ndcg = -1.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "din_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            uid_idx = batch["uid_idx"].to(device)
            hist = batch["hist"].to(device)
            hist_cnt = batch["hist_cnt"].to(device)
            hist_len = batch["hist_len"].to(device)
            target = batch["target"].to(device)

            # IMPORTANT: drive the attention unit with the most-recent HISTORY item,
            # not the true target. Using the target here leaks label information into
            # attention during training while evaluation (which cannot know the label)
            # has to fall back to the last history item -- a train/inference mismatch
            # that was silently capping NDCG. Using the same "last history item as
            # attention candidate" convention at both train and eval time means the
            # model actually learns the attention pattern it will be evaluated with.
            batch_idx_ = torch.arange(hist.size(0), device=device)
            last_valid_pos_ = hist.size(1) - 1
            attn_cand = hist[batch_idx_, last_valid_pos_]

            user_vec = model(uid_idx, hist, hist_cnt, hist_len, attn_cand)

            # recompute the full-catalog item vectors every step (item_score_head's
            # weights change each update, so these can't be cached across steps).
            all_item_vecs = model.item_static_vec(all_item_ids)  # [n_items, D]
            loss = pairwise_bpr_loss(user_vec, target, all_item_vecs, num_neg=cfg.num_neg)

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
    ds = DINDataset(test_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=False)
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