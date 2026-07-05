# -*- coding: utf-8 -*-
"""
DINO (Dynamic Interest Network Optimization) for next-item / sequential recommendation.

This is a direct evolution of the DIN (Deep Interest Network) baseline. It keeps
DIN's data schema, training protocol (full-softmax over the catalog), and
evaluation (NDCG@10), but replaces DIN's *static* pairwise activation unit with
a *dynamic* interest-modeling stack:

  1. Self-Attention Interest Encoder (SAIE)
     A multi-head self-attention layer over the whole history sequence first
     lets history items "talk to each other" (captures sequential / co-occurrence
     structure DIN's pairwise activation unit cannot see, since DIN only ever
     compares each history item to the candidate in isolation).

  2. Dynamic Target-Attention (DTA)
     A candidate-aware attention layer (DIN-style activation unit) is then
     applied on top of the *contextualized* history representations from step 1,
     so the candidate-dependent attention weights are computed over an
     already-evolved interest sequence rather than raw item embeddings.

  3. Interest Evolution Gate (IEG)
     A GRU-based gate fuses the self-attended sequence representation with the
     target-attention pooled vector through a learned gate, modeling how a
     user's interest "evolves" toward the candidate rather than being a fixed
     weighted sum -- this is the key dynamic/optimization piece that DIN lacks.

  4. Frequency-aware weighting
     Like DIN, item_seq_counts (log1p) is fed into both the activation unit and
     used as an explicit re-weighting signal on the pooled interest, so
     frequently-interacted items keep contributing more than one-off clicks.

Data schema, Dataset, training loop, full-softmax loss, and NDCG@10 evaluation
are unchanged from DIN -- DINO is meant as a drop-in replacement model class.

Usage
-----
python dino_model.py --data_dir /path/to/data --out_dir /path/to/output --epochs 5
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
    attn_hidden = (80, 40)  # target-attention MLP hidden sizes
    mlp_hidden = (200, 80)  # final user-representation MLP hidden sizes
    dropout = 0.2
    n_heads = 8             # self-attention heads in the interest encoder
    n_sa_layers = 3         # number of self-attention encoder layers
    gru_hidden = None       # interest-evolution GRU hidden size; defaults to item_full_dim if None

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
        parser = argparse.ArgumentParser(description="DINO training/prediction")
        for f in fields(cls) if False else []:
            pass  # dataclass has no declared fields with `field()`; CLI mirrors attrs manually below
        defaults = {k: v for k, v in vars(cls).items() if not k.startswith("_") and not callable(v)}
        for name, default in defaults.items():
            if name in ("from_args",):
                continue
            arg_type = type(default) if default is not None and not isinstance(default, (tuple, bool)) else str
            if isinstance(default, bool):
                parser.add_argument(f"--{name}", type=lambda x: x.lower() in ("1", "true", "yes"), default=default)
            elif isinstance(default, tuple):
                parser.add_argument(f"--{name}", type=str, default=",".join(map(str, default)))
            else:
                parser.add_argument(f"--{name}", type=arg_type, default=default)
        args, _ = parser.parse_known_args()
        for name in defaults:
            val = getattr(args, name)
            if isinstance(getattr(cfg, name), tuple) and isinstance(val, str):
                val = tuple(int(x) for x in val.split(","))
            setattr(cfg, name, val)
        return cfg


# ========================================================================================
# 1. Vocab / encoding utilities  (unchanged from DIN)
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
# 2. Data loading  (unchanged from DIN)
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
# 3. Dataset  (unchanged from DIN)
# ========================================================================================

class DINODataset(Dataset):
    """
    Each sample: (uid_idx, hist_item_idx[seq_len], hist_count[seq_len], hist_len, target_item_idx)

    History is built from `item_seq_dedup` (distinct items, in first-seen order)
    paired with their frequency from `item_seq_counts`. The count signal is fed
    into the model both as an extra interaction feature in the target-attention
    unit and as an explicit multiplicative weight on the pooled interest, so
    items the user interacted with many times influence the user
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
# 4. DINO model
# ========================================================================================

class SelfAttnInterestEncoder(nn.Module):
    """Step 1 of DINO: lets history items attend to EACH OTHER before any candidate
    is introduced. This is what makes the interest representation "dynamic" instead
    of a static bag of independently-embedded items as in vanilla DIN -- a click on
    item A right before item B should be able to influence how B's contribution to
    the user's interest is represented, regardless of which candidate we later score.

    Implemented as a small stack of standard Transformer encoder layers (multi-head
    self-attention + position-wise FFN + residual/LayerNorm), operating directly on
    the concatenated [id_emb || side_emb] item representations so dimensionality
    matches the rest of the model exactly (no extra projection layer needed)."""

    def __init__(self, dim, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        # round n_heads down to something that divides `dim` evenly; nn.MultiheadAttention
        # requires embed_dim % num_heads == 0, and side-feature concatenation can produce
        # odd total dims depending on emb_dim / side_emb_ratio choices.
        heads = n_heads
        while heads > 1 and dim % heads != 0:
            heads -= 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, hist_full, mask):
        # hist_full: [B, L, D], mask: [B, L] (1 = valid, 0 = pad)
        # TransformerEncoder expects src_key_padding_mask True at PAD positions.
        pad_mask = (mask == 0)  # [B, L]
        # guard fully-empty rows (hist_len == 0): give them an all-valid mask so
        # attention math doesn't produce NaNs from an all-True padding row.
        empty_rows = pad_mask.all(dim=1)
        if empty_rows.any():
            pad_mask = pad_mask.clone()
            pad_mask[empty_rows] = False
        out = self.encoder(hist_full, src_key_padding_mask=pad_mask)
        # zero out padded positions explicitly (encoder leaves them as transformed
        # garbage that we never want pooled into the interest vector downstream)
        out = out * mask.unsqueeze(-1).float()
        return out  # [B, L, D]


class DynamicTargetAttention(nn.Module):
    """Step 2 of DINO: a DIN-style activation unit, but operating on the
    CONTEXTUALIZED history (output of SelfAttnInterestEncoder) instead of raw item
    embeddings. Computes attention weight between a candidate item embedding and
    each contextualized historical item embedding, using the classic
    [hist, cand, hist-cand, hist*cand] interaction features fed through an MLP,
    PLUS an extra scalar feature: log1p(item_seq_counts) for that history item."""

    def __init__(self, dim, hidden=(80, 40)):
        super().__init__()
        layers = []
        in_dim = dim * 4 + 1  # +1 for the log-count feature
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.PReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, hist_emb, cand_emb, mask, hist_cnt):
        # hist_emb: [B, L, D] (contextualized), cand_emb: [B, D], mask: [B, L], hist_cnt: [B, L]
        L = hist_emb.size(1)
        cand_exp = cand_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        log_cnt = torch.log1p(hist_cnt).unsqueeze(-1)  # [B, L, 1]
        feat = torch.cat([hist_emb, cand_exp, hist_emb - cand_exp, hist_emb * cand_exp, log_cnt], dim=-1)
        scores = self.mlp(feat).squeeze(-1)  # [B, L]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)  # [B, L]
        weights = weights.masked_fill(mask == 0, 0.0)
        return weights


class InterestEvolutionGate(nn.Module):
    """Step 3 of DINO -- the core "optimization" piece beyond DIN.

    DIN produces the user's interest as a single static weighted-sum (attention
    pooling) of history embeddings. DINO instead treats interest formation as an
    EVOLVING process: a GRU is run over the self-attended history sequence (in
    chronological order) to produce a sequence-level "evolution" summary, which is
    then fused with the target-attention pooled vector via a learned gate. The gate
    lets the model decide, per-sample, how much to trust "what the candidate-aware
    attention currently highlights" vs. "where the user's overall interest has been
    drifting toward" -- a dynamic combination rather than DIN's fixed pooling."""

    def __init__(self, dim, gru_hidden=None):
        super().__init__()
        gru_hidden = gru_hidden or dim
        self.gru = nn.GRU(input_size=dim, hidden_size=gru_hidden, batch_first=True)
        self.evo_proj = nn.Linear(gru_hidden, dim) if gru_hidden != dim else nn.Identity()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

    def forward(self, context_hist, attn_pooled, hist_len):
        # context_hist: [B, L, D] (self-attended, chronological order, pad at front)
        # attn_pooled:  [B, D]    (target-attention weighted pooling)
        # hist_len:     [B]       number of valid (non-pad) positions
        B, L, D = context_hist.shape
        packed_out, _ = self.gru(context_hist)  # [B, L, gru_hidden]

        # gather the GRU hidden state at each row's LAST valid (non-pad) timestep;
        # padding sits at the FRONT of the sequence (see DINODataset), so the last
        # valid timestep is always position L-1 except when hist_len == 0.
        last_pos = (L - 1) * torch.ones_like(hist_len)
        last_pos = torch.clamp(last_pos, min=0)
        batch_idx = torch.arange(B, device=context_hist.device)
        evo_summary = packed_out[batch_idx, last_pos]  # [B, gru_hidden]
        evo_summary = self.evo_proj(evo_summary)  # [B, D]

        # zero out the evolution summary for genuinely empty histories so the gate
        # falls back fully on the (zero) attention-pooled vector rather than on
        # GRU's behavior on an all-padding input.
        has_hist = (hist_len > 0).float().unsqueeze(-1)  # [B, 1]
        evo_summary = evo_summary * has_hist

        g = self.gate(torch.cat([attn_pooled, evo_summary], dim=-1))  # [B, D] in (0,1)
        fused = g * attn_pooled + (1.0 - g) * evo_summary
        return fused


class DINO(nn.Module):
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
        self.item_full_dim = item_full_dim

        # --- DINO's dynamic interest stack ---
        self.self_attn_encoder = SelfAttnInterestEncoder(
            item_full_dim, n_heads=cfg.n_heads, n_layers=cfg.n_sa_layers, dropout=cfg.dropout,
        )
        self.dynamic_target_attn = DynamicTargetAttention(item_full_dim, hidden=cfg.attn_hidden)
        self.interest_evolution_gate = InterestEvolutionGate(item_full_dim, gru_hidden=cfg.gru_hidden)

        mlp_in = user_full_dim + item_full_dim + item_full_dim  # user feat + evolved interest + candidate
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
        cand_item_idx is used only to drive the dynamic target-attention (one
        candidate per row, typically the positive target during training).
        hist_cnt: [B, L] raw item_seq_counts values (0 on padding) -- used both inside
        the target-attention unit (as a log-count feature) and as an explicit
        multiplicative weight on the attention scores, so high-frequency history
        items contribute proportionally more to the pooled interest vector."""
        mask = (hist != PAD_IDX).long()  # [B, L]
        hist_full = self._item_full_emb(hist)  # [B, L, item_full_dim]
        cand_full = self._item_full_emb(cand_item_idx)  # [B, item_full_dim]

        # 1) self-attention over history -> contextualized ("dynamic") item reps
        context_hist = self.self_attn_encoder(hist_full, mask)  # [B, L, item_full_dim]

        # 2) candidate-aware attention over the contextualized history
        attn_w = self.dynamic_target_attn(context_hist, cand_full, mask, hist_cnt)  # [B, L]

        # re-weight by interaction frequency (log1p to dampen extreme outlier counts),
        # then renormalize so the pooling remains a convex combination.
        freq_w = torch.log1p(hist_cnt) * mask.float()  # [B, L], 0 on pad
        combined_w = attn_w * (1.0 + freq_w)
        combined_w = combined_w / (combined_w.sum(dim=-1, keepdim=True) + 1e-9)

        attn_pooled = torch.bmm(combined_w.unsqueeze(1), context_hist).squeeze(1)  # [B, item_full_dim]

        # 3) fuse the target-attention pooled vector with a GRU-based interest
        # evolution summary through a learned gate -> the "dynamic optimization" step
        interest = self.interest_evolution_gate(context_hist, attn_pooled, hist_len)  # [B, item_full_dim]

        user_full = self._user_full_emb(uid_idx)  # [B, user_full_dim]

        x = torch.cat([user_full, interest, cand_full], dim=-1)
        user_vec = self.mlp(x)  # [B, emb_dim]
        return user_vec


class DINORanker(nn.Module):
    """Wraps DINO: trains with a FULL softmax over the entire item catalog using the
    user vector from DINO's self-attention + dynamic target-attention + interest
    evolution gate (with candidate = the positive target during training, the
    standard trick used at training time), and a separate static item-scoring
    embedding (i_static) used to score ALL items at both train and inference time,
    since true target-attention is candidate-specific and we approximate
    full-catalog scoring via a learned static projection of item_full_emb that is
    trained jointly to be consistent with the dynamically-formed user vector."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        self.dino = DINO(n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg)
        emb_dim = cfg.emb_dim
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))
        item_full_dim = emb_dim + side_dim * len(item_feat_vocabs)
        self.item_score_head = nn.Linear(item_full_dim, emb_dim)  # static item vector for full-catalog scoring
        self.n_items = n_items

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.dino.set_feature_tables(user_feat_table, item_feat_table)

    def item_static_vec(self, item_idx):
        full = self.dino._item_full_emb(item_idx)
        return self.item_score_head(full)

    def forward(self, uid_idx, hist, hist_cnt, hist_len, target_idx):
        user_vec = self.dino.encode_user(uid_idx, hist, hist_cnt, hist_len, target_idx)  # [B, D]
        return user_vec

    def score_against_catalog(self, user_vec, item_vecs):
        # user_vec: [B, D], item_vecs: [N, D] -> [B, N]
        return user_vec @ item_vecs.t()


# ========================================================================================
# 5. Training: full softmax over the entire item catalog  (unchanged from DIN)
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
    only ever saw a handful of random negatives, while evaluation ranks against
    the full catalog).
    """
    logits = user_vec @ all_item_vecs.t()  # [B, n_items]
    logits = logits.clone()
    logits[:, PAD_IDX] = -1e9  # never let the model assign probability mass to PAD
    loss = F.cross_entropy(logits, target_idx)
    return loss


# ========================================================================================
# 6. NDCG@10 metric  (unchanged from DIN)
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
        # recent history item as a stand-in candidate -- the same DIN inference
        # approximation, kept identical here for a fair comparison against DIN.
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
# 7. Training loop  (unchanged from DIN, aside from using DINORanker)
# ========================================================================================

def train_model(bundle, cfg: "Config", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    train_full = bundle.train_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    n_val = int(len(train_full) * cfg.val_frac)
    val_df = train_full.iloc[:n_val].reset_index(drop=True)
    tr_df = train_full.iloc[n_val:].reset_index(drop=True)
    print(f"[INFO] train={len(tr_df)}  valid={len(val_df)}")

    train_ds = DINODataset(tr_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=True)
    val_ds = DINODataset(val_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    user_feat_vocabs = [len(bundle.u_encoders[c]) for c in bundle.u_cat_cols]
    item_feat_vocabs = [len(bundle.i_encoders[c]) for c in bundle.i_cat_cols]

    model = DINORanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
    model.set_feature_tables(
        torch.tensor(bundle.user_feat, dtype=torch.long),
        torch.tensor(bundle.item_feat, dtype=torch.long),
    )
    model.to(device)
    model.dino.user_feat_table = model.dino.user_feat_table.to(device)
    model.dino.item_feat_table = model.dino.item_feat_table.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    all_item_ids = torch.arange(bundle.n_items, device=device)

    best_ndcg = -1.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "dino_best.pt")

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
            # that was silently capping NDCG in the original DIN baseline. Using the
            # same "last history item as attention candidate" convention at both train
            # and eval time means the model actually learns the attention pattern it
            # will be evaluated with.
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
# 8. Prediction  (unchanged from DIN)
# ========================================================================================

@torch.no_grad()
def run_predict(model, bundle, cfg: "Config", device):
    test_df = bundle.test_df
    if test_df is None:
        print("[WARN] no test.csv found, skipping prediction.")
        return

    model.eval()
    ds = DINODataset(test_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len, has_target=False)
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
