# -*- coding: utf-8 -*-
"""
DSIN (Deep Session Interest Network) for next-item / sequential recommendation.

This script reuses, almost verbatim, every data-handling and training-loop
component from `din_model.py` (the DIN baseline it sits next to):

    - CategoryEncoder / IdEncoder                  (vocab utilities)
    - parse_seq_raw / parse_seq_counts             (sequence string parsing)
    - load_data / DataBundle                       (csv -> tensors)
    - ActivationUnit                                (DIN-style local activation
                                                      unit, reused unchanged for
                                                      the final interest-activation
                                                      step over session vectors)
    - full_softmax_loss                             (full-catalog softmax loss)
    - ndcg_at_k / evaluate_ndcg                     (metric)
    - collate_fn, train_model skeleton, run_predict skeleton
    - Config dataclass (extended with DSIN-only knobs)

What's NEW for DSIN (the actual modeling contribution of the paper
"Deep Session Interest Network for Click-Through Rate Prediction", Feng et al.
2019), replacing DIN's single flat-attention pooling step:

    1. Session division layer
       The behaviour sequence is split into sessions: a new session starts
       whenever the gap between two consecutive interactions exceeds a time
       threshold. Since this dataset has no explicit per-item timestamps, we
       approximate session boundaries from the *order* of item_seq_raw using
       a fixed-size sliding window (configurable `session_len`), which is the
       standard practical fallback used in DSIN re-implementations when
       timestamps aren't available, while leaving a clear extension point to
       plug in true timestamp-based session splitting if such a column exists.

    2. Session interest extractor layer
       Inside each session, a multi-head self-attention (Transformer encoder)
       layer with bias-encoding (a learnable per-position/per-session bias
       added before self-attention, as in the paper) models the dependency
       between behaviors in the same session, and the session is then
       sum/average-pooled into a single session-interest vector.

    3. Session interest interacting layer
       A Bi-LSTM runs over the sequence of session-interest vectors to model
       how a user's interest evolves *across* sessions.

    4. Session interest activating layer
       DIN's `ActivationUnit` (imported unchanged from din_model.py) is reused
       twice here: once to attention-pool the self-attention session vectors
       w.r.t. the candidate item, and once to attention-pool the Bi-LSTM
       hidden states w.r.t. the candidate item. Both attended vectors plus the
       user profile and candidate embedding are concatenated and fed through
       an MLP -> user vector, exactly mirroring DIN's final stage so the rest
       of the training/eval/predict pipeline needs no changes.

Data schema, task framing, full-softmax training objective, and submission
format are all identical to din_model.py -- see its docstring for details.

Usage
-----
python dsin_model.py --data_dir /path/to/data --out_dir /path/to/output --epochs 5
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
# Reuse everything generic from din_model.py instead of re-implementing it.
# --------------------------------------------------------------------------------------
from din_model_2 import (
    SEED, PAD_IDX,
    CategoryEncoder, IdEncoder,
    parse_seq_raw, parse_seq_counts,
    DataBundle, load_data,
    collate_fn,
    ActivationUnit,            # reused unchanged for the activation layer
    full_softmax_loss,         # reused unchanged training objective
    ndcg_at_k,                 # reused unchanged metric
    make_synthetic_data,       # reused unchanged synthetic data generator
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ========================================================================================
# 0. Config
# ========================================================================================

@dataclass
class Config:
    """Same fields as DIN's Config (paths / data handling / training / prediction),
    plus DSIN-specific knobs for session division, self-attention, and Bi-LSTM."""

    # paths
    data_dir = "../data/A2-Rec"
    out_dir = "./"

    # data / sequence handling
    max_seq_len = 50          # truncate/pad full history to this many distinct items
    val_frac = 0.1
    use_synthetic = False

    # model: shared with DIN
    emb_dim = 32
    side_emb_ratio = 0.5
    attn_hidden = (80, 40)    # activation-unit MLP hidden sizes (reused ActivationUnit)
    mlp_hidden = (200, 80)    # final user-representation MLP hidden sizes
    dropout = 0.2

    # model: DSIN-specific
    session_len = 10          # max items per session (sliding-window session split)
    max_sessions = 5          # max number of sessions kept per user (most recent kept)
    n_attn_heads = 4          # self-attention heads in the session interest extractor
    lstm_hidden = 32          # Bi-LSTM hidden size per direction for session interaction layer

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
        p = argparse.ArgumentParser()
        for f in fields(cls):
            default = getattr(cfg, f.name, None)
            t = type(default) if default is not None else str
            if t is bool:
                p.add_argument(f"--{f.name}", type=lambda x: str(x).lower() == "true", default=default)
            elif f.name in ("attn_hidden", "mlp_hidden"):
                p.add_argument(f"--{f.name}", type=str, default=None,
                                help="comma-separated ints, e.g. 80,40")
            else:
                p.add_argument(f"--{f.name}", type=t, default=default)
        args = p.parse_args()
        for f in fields(cls):
            val = getattr(args, f.name)
            if val is None:
                continue
            if f.name in ("attn_hidden", "mlp_hidden") and isinstance(val, str):
                val = tuple(int(x) for x in val.split(","))
            setattr(cfg, f.name, val)
        return cfg


# ========================================================================================
# 1. Session-aware Dataset
# ========================================================================================

def split_into_sessions(item_ids, counts, session_len, max_sessions):
    """Splits a chronological list of distinct history items (+ their counts)
    into fixed-size sliding-window 'sessions' of length `session_len`, keeping
    only the most recent `max_sessions` sessions.

    In the original DSIN paper, sessions are split by a 30-minute inactivity
    gap using real timestamps. This dataset only exposes order (no
    timestamps), so we approximate with contiguous windows over the ordered
    sequence -- a documented, standard fallback. If a timestamp column is
    later made available, only this function needs to change (replace the
    fixed-size chunking with a gap-based split); the rest of the model is
    agnostic to how sessions were formed.
    """
    sessions, session_counts = [], []
    for i in range(0, len(item_ids), session_len):
        sessions.append(item_ids[i:i + session_len])
        session_counts.append(counts[i:i + session_len])
    # keep most recent max_sessions sessions
    sessions = sessions[-max_sessions:]
    session_counts = session_counts[-max_sessions:]
    return sessions, session_counts


class DSINDataset(Dataset):
    """
    Each sample provides:
        uid_idx            : scalar
        sessions           : [n_sessions, session_len]   item ids, 0 = pad
        session_counts     : [n_sessions, session_len]   item_seq_counts values, 0 = pad
        session_item_mask  : [n_sessions, session_len]   1 = real item, 0 = pad
        session_mask       : [n_sessions]                1 = real session, 0 = pad session
        target             : scalar (only if has_target)

    Built from item_seq_dedup (distinct items, first-seen/chronological order)
    + item_seq_counts, identical source fields to DIN's dataset -- only the
    bucketing into sessions is new.
    """

    def __init__(self, df, uid_enc, iid_enc, session_len, max_sessions, has_target=True):
        self.uids = df["uid"].tolist()
        self.dedup_seqs = df["item_seq_dedup"].tolist()
        self.count_strs = df["item_seq_counts"].tolist()
        self.has_target = has_target
        if has_target:
            self.targets = df["target_iid"].tolist()
        self.uid_enc = uid_enc
        self.iid_enc = iid_enc
        self.session_len = session_len
        self.max_sessions = max_sessions

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid_idx = self.uid_enc.transform_one(self.uids[idx])
        items = parse_seq_raw(self.dedup_seqs[idx])
        counts_map = parse_seq_counts(self.count_strs[idx])

        # cap total history considered, same truncation convention as DIN
        # (keep the most recent max_sessions * session_len distinct items)
        cap = self.max_sessions * self.session_len
        items = items[-cap:]
        counts = [counts_map.get(i, 1) for i in items]
        item_idx_list = [self.iid_enc.transform_one(i) for i in items]

        sessions, session_counts = split_into_sessions(
            item_idx_list, counts, self.session_len, self.max_sessions
        )

        n_sessions = len(sessions)
        sess_arr = np.zeros((self.max_sessions, self.session_len), dtype=np.int64)
        cnt_arr = np.zeros((self.max_sessions, self.session_len), dtype=np.float32)
        item_mask = np.zeros((self.max_sessions, self.session_len), dtype=np.int64)
        sess_mask = np.zeros((self.max_sessions,), dtype=np.int64)

        # left-pad sessions within the max_sessions axis so the most recent
        # sessions sit at the END, mirroring DIN's "pad at front" convention.
        pad_sessions = self.max_sessions - n_sessions
        for s_i, (sess, cnts) in enumerate(zip(sessions, session_counts)):
            row = pad_sessions + s_i
            L = len(sess)
            # left-pad within-session too (most recent items at the end)
            pad_items = self.session_len - L
            sess_arr[row, pad_items:] = sess
            cnt_arr[row, pad_items:] = cnts
            item_mask[row, pad_items:] = 1
            sess_mask[row] = 1

        sample = {
            "uid_idx": uid_idx,
            "sessions": sess_arr,
            "session_counts": cnt_arr,
            "session_item_mask": item_mask,
            "session_mask": sess_mask,
        }
        if self.has_target:
            sample["target"] = self.iid_enc.transform_one(self.targets[idx])
        return sample


def dsin_collate_fn(batch):
    uid_idx = torch.tensor([b["uid_idx"] for b in batch], dtype=torch.long)
    sessions = torch.tensor(np.stack([b["sessions"] for b in batch]), dtype=torch.long)
    session_counts = torch.tensor(np.stack([b["session_counts"] for b in batch]), dtype=torch.float32)
    session_item_mask = torch.tensor(np.stack([b["session_item_mask"] for b in batch]), dtype=torch.long)
    session_mask = torch.tensor(np.stack([b["session_mask"] for b in batch]), dtype=torch.long)
    out = {
        "uid_idx": uid_idx,
        "sessions": sessions,
        "session_counts": session_counts,
        "session_item_mask": session_item_mask,
        "session_mask": session_mask,
    }
    if "target" in batch[0]:
        out["target"] = torch.tensor([b["target"] for b in batch], dtype=torch.long)
    return out


# ========================================================================================
# 2. DSIN model
# ========================================================================================

class BiasEncoding(nn.Module):
    """Learnable bias added to item embeddings before self-attention, as in the
    DSIN paper's 'bias encoding' layer: separate bias terms for (session
    position, in-session position, embedding dimension), broadcast-summed.
    This lets the self-attention layer distinguish *which* session and *where
    within the session* a behavior occurred -- standard positional info that
    plain self-attention has no other way to see."""

    def __init__(self, max_sessions, session_len, emb_dim):
        super().__init__()
        self.session_bias = nn.Parameter(torch.zeros(max_sessions, 1, emb_dim))
        self.position_bias = nn.Parameter(torch.zeros(1, session_len, emb_dim))
        self.dim_bias = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.normal_(self.session_bias, std=0.02)
        nn.init.normal_(self.position_bias, std=0.02)
        nn.init.normal_(self.dim_bias, std=0.02)

    def forward(self, x, session_idx):
        # x: [B, session_len, D] (one session at a time)
        return x + self.session_bias[session_idx] + self.position_bias + self.dim_bias


class SessionInterestExtractor(nn.Module):
    """Self-attention (Transformer-encoder-style, multi-head) layer applied
    independently within each session, followed by masked mean-pooling to
    produce a single interest vector per session. This is DSIN's 'session
    interest extractor layer'."""

    def __init__(self, emb_dim, n_heads, max_sessions, session_len, dropout):
        super().__init__()
        self.bias_encoding = BiasEncoding(max_sessions, session_len, emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), nn.ReLU(), nn.Linear(emb_dim * 2, emb_dim)
        )
        self.ln2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, session_item_emb, item_mask, session_mask):
        """
        session_item_emb : [B, S, L, D]
        item_mask        : [B, S, L]   1 = real item
        session_mask     : [B, S]      1 = real (non-padding) session
        returns          : [B, S, D]   one interest vector per session
        """
        B, S, L, D = session_item_emb.shape
        x = session_item_emb.view(B * S, L, D)
        mask = item_mask.view(B * S, L)

        session_idx = torch.arange(S, device=x.device).repeat_interleave(B)
        # bias encoding expects [N, L, D] with matching per-row session index;
        # build per-row session ids in (B*S) order matching x's flattening (S-major within each B)
        session_idx_full = torch.arange(S, device=x.device).unsqueeze(0).expand(B, S).reshape(-1)
        x = self.bias_encoding(x, session_idx_full)

        key_padding_mask = (mask == 0)  # True = ignore
        # guard against fully-padded sessions (all-True key_padding_mask -> NaN in attention);
        # for those rows attention output is irrelevant since session_mask will zero them out later.
        all_pad_rows = key_padding_mask.all(dim=-1)
        safe_kpm = key_padding_mask.clone()
        safe_kpm[all_pad_rows] = False

        attn_out, _ = self.attn(x, x, x, key_padding_mask=safe_kpm, need_weights=False)
        x = self.ln1(x + self.dropout(attn_out))
        x = self.ln2(x + self.dropout(self.ffn(x)))  # [B*S, L, D]

        # masked mean-pool over the (real) items in the session
        mask_f = mask.unsqueeze(-1).float()  # [B*S, L, 1]
        summed = (x * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        session_vec = summed / denom  # [B*S, D]

        session_vec = session_vec.view(B, S, D)
        session_vec = session_vec * session_mask.unsqueeze(-1).float()  # zero out pad sessions
        return session_vec


class SessionInterestInteracting(nn.Module):
    """Bi-LSTM over the sequence of session-interest vectors, modeling how
    interest evolves across sessions. This is DSIN's 'session interest
    interacting layer'."""

    def __init__(self, emb_dim, lstm_hidden):
        super().__init__()
        self.lstm = nn.LSTM(emb_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.out_dim = lstm_hidden * 2

    def forward(self, session_vec, session_mask):
        # session_vec: [B, S, D], session_mask: [B, S]
        lengths = session_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            session_vec, lengths, batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=session_vec.size(1)
        )
        out = out * session_mask.unsqueeze(-1).float()
        return out  # [B, S, 2*lstm_hidden]


class DSIN(nn.Module):
    """Mirrors DIN's structure (id + side-feature embeddings, final MLP) but
    replaces DIN's single flat-history activation pooling with DSIN's
    session-based pipeline: split -> self-attention extractor -> Bi-LSTM
    interacting -> two activation-unit poolings (reusing DIN's
    ActivationUnit) w.r.t. the candidate item."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        emb_dim = cfg.emb_dim
        self.emb_dim = emb_dim
        self.max_sessions = cfg.max_sessions
        self.session_len = cfg.session_len

        # core id embeddings -- identical role to DIN
        self.user_emb = nn.Embedding(n_users, emb_dim, padding_idx=PAD_IDX)
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=PAD_IDX)

        # side feature embeddings -- identical to DIN
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

        # ---- DSIN-specific layers ----
        # self-attention operates on the *base* item id embedding (emb_dim);
        # side features are folded in afterwards when forming the candidate
        # comparison space for the activation units, keeping the Transformer
        # block's width manageable while still letting side info influence
        # the final activation/MLP stage.
        self.session_extractor = SessionInterestExtractor(
            emb_dim, cfg.n_attn_heads, cfg.max_sessions, cfg.session_len, cfg.dropout
        )
        self.session_interacting = SessionInterestInteracting(emb_dim, cfg.lstm_hidden)
        lstm_out_dim = self.session_interacting.out_dim  # 2 * lstm_hidden

        # project session vectors (self-attn output: emb_dim) and Bi-LSTM
        # output (lstm_out_dim) into item_full_dim so DIN's ActivationUnit
        # (which expects matching hist/cand dims) can be reused unchanged
        # against the full (id + side-feature) candidate representation.
        self.sess_to_full = nn.Linear(emb_dim, item_full_dim)
        self.lstm_to_full = nn.Linear(lstm_out_dim, item_full_dim)

        # reuse DIN's ActivationUnit verbatim, once per DSIN interest branch
        self.activation_unit_sess = ActivationUnit(item_full_dim, hidden=cfg.attn_hidden)
        self.activation_unit_lstm = ActivationUnit(item_full_dim, hidden=cfg.attn_hidden)

        # final MLP: user profile + session-interest (activated) + bilstm-interest (activated) + candidate
        mlp_in = user_full_dim + item_full_dim + item_full_dim + item_full_dim
        h1, h2 = cfg.mlp_hidden
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, h1), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h1, h2), nn.PReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(h2, emb_dim),
        )

        self.register_buffer("user_feat_table", torch.zeros(n_users, 1, dtype=torch.long), persistent=False)
        self.register_buffer("item_feat_table", torch.zeros(n_items, 1, dtype=torch.long), persistent=False)

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

    def encode_user(self, uid_idx, sessions, session_counts, session_item_mask, session_mask, cand_item_idx):
        """
        sessions           : [B, S, L]
        session_counts     : [B, S, L]
        session_item_mask  : [B, S, L]
        session_mask       : [B, S]
        cand_item_idx      : [B]
        """
        B, S, L = sessions.shape

        # 1) session interest extractor (self-attention + pooling) over base id embeddings
        session_item_emb = self.item_emb(sessions)  # [B, S, L, emb_dim]
        session_vecs = self.session_extractor(session_item_emb, session_item_mask, session_mask)  # [B, S, emb_dim]

        # 2) session interest interacting (Bi-LSTM across sessions)
        lstm_out = self.session_interacting(session_vecs, session_mask)  # [B, S, 2*lstm_hidden]

        # candidate full representation (id + side feats), used for both activation units
        cand_full = self._item_full_emb(cand_item_idx)  # [B, item_full_dim]

        # 3) session interest activating layer -- reuse DIN's ActivationUnit twice
        sess_full = self.sess_to_full(session_vecs)              # [B, S, item_full_dim]
        lstm_full = self.lstm_to_full(lstm_out)                  # [B, S, item_full_dim]

        # ActivationUnit expects a per-position raw "count" feature (log1p(count));
        # we use the per-session aggregate count (mean of in-session item counts,
        # masked) as the session-level frequency signal it was designed to consume.
        cnt_mask = session_item_mask.float()
        sess_cnt = (session_counts * cnt_mask).sum(dim=-1) / cnt_mask.sum(dim=-1).clamp(min=1.0)  # [B, S]
        sess_cnt = sess_cnt * session_mask.float()

        attn_w_sess = self.activation_unit_sess(sess_full, cand_full, session_mask, sess_cnt)  # [B, S]
        attn_w_lstm = self.activation_unit_lstm(lstm_full, cand_full, session_mask, sess_cnt)  # [B, S]

        sess_interest = torch.bmm(attn_w_sess.unsqueeze(1), sess_full).squeeze(1)  # [B, item_full_dim]
        lstm_interest = torch.bmm(attn_w_lstm.unsqueeze(1), lstm_full).squeeze(1)  # [B, item_full_dim]

        user_full = self._user_full_emb(uid_idx)  # [B, user_full_dim]

        x = torch.cat([user_full, sess_interest, lstm_interest, cand_full], dim=-1)
        user_vec = self.mlp(x)  # [B, emb_dim]
        return user_vec


class DSINRanker(nn.Module):
    """Wraps DSIN exactly the way DINRanker wraps DIN: a static per-item
    scoring head (learned projection of item_full_emb) is used for
    full-catalog dot-product scoring at both train and inference time, while
    DSIN's session-based attention pipeline produces the candidate-specific
    user vector used as the softmax query during training/eval."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        self.dsin = DSIN(n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg)
        emb_dim = cfg.emb_dim
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))
        item_full_dim = emb_dim + side_dim * len(item_feat_vocabs)
        self.item_score_head = nn.Linear(item_full_dim, emb_dim)
        self.n_items = n_items

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.dsin.set_feature_tables(user_feat_table, item_feat_table)

    def item_static_vec(self, item_idx):
        full = self.dsin._item_full_emb(item_idx)
        return self.item_score_head(full)

    def forward(self, uid_idx, sessions, session_counts, session_item_mask, session_mask, target_idx):
        user_vec = self.dsin.encode_user(uid_idx, sessions, session_counts, session_item_mask, session_mask, target_idx)
        return user_vec

    def score_against_catalog(self, user_vec, item_vecs):
        return user_vec @ item_vecs.t()


# ========================================================================================
# 3. NDCG evaluation (DSIN-specific batch unpacking; metric logic reused from din_model)
# ========================================================================================

def _last_real_item_per_row(sessions, session_item_mask):
    """Finds, for each batch row, the most recent real (non-pad) item across
    all sessions -- used as the attention-driving candidate stand-in at
    inference time (DIN/DSIN convention: never peek at the true label)."""
    B, S, L = sessions.shape
    flat_items = sessions.view(B, S * L)
    flat_mask = session_item_mask.view(B, S * L)
    # positions are already in chronological (oldest..newest) order along S*L
    # because both session order and within-session order were built that way.
    idx = torch.arange(S * L, device=sessions.device).unsqueeze(0).expand(B, -1)
    masked_idx = torch.where(flat_mask.bool(), idx, torch.full_like(idx, -1))
    last_pos = masked_idx.max(dim=1).values.clamp(min=0)  # [B]
    last_item = flat_items.gather(1, last_pos.unsqueeze(1)).squeeze(1)
    # rows with no real items at all -> PAD_IDX
    has_any = flat_mask.any(dim=1)
    last_item = torch.where(has_any, last_item, torch.full_like(last_item, PAD_IDX))
    return last_item


@torch.no_grad()
def evaluate_ndcg(model, loader, item_vecs_all, k=10, device="cpu"):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        uid_idx = batch["uid_idx"].to(device)
        sessions = batch["sessions"].to(device)
        session_counts = batch["session_counts"].to(device)
        session_item_mask = batch["session_item_mask"].to(device)
        session_mask = batch["session_mask"].to(device)
        target = batch["target"].to(device)

        attn_cand = _last_real_item_per_row(sessions, session_item_mask)
        user_vec = model(uid_idx, sessions, session_counts, session_item_mask, session_mask, attn_cand)
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
# 4. Training loop (mirrors din_model.train_model; same hyperparameter handling,
#    only the dataset/model classes and batch field names differ)
# ========================================================================================

def train_model(bundle, cfg: "Config", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    train_full = bundle.train_df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    n_val = int(len(train_full) * cfg.val_frac)
    val_df = train_full.iloc[:n_val].reset_index(drop=True)
    tr_df = train_full.iloc[n_val:].reset_index(drop=True)
    print(f"[INFO] train={len(tr_df)}  valid={len(val_df)}")

    train_ds = DSINDataset(tr_df, bundle.uid_enc, bundle.iid_enc,
                            session_len=cfg.session_len, max_sessions=cfg.max_sessions, has_target=True)
    val_ds = DSINDataset(val_df, bundle.uid_enc, bundle.iid_enc,
                          session_len=cfg.session_len, max_sessions=cfg.max_sessions, has_target=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               collate_fn=dsin_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=dsin_collate_fn, num_workers=0)

    user_feat_vocabs = [len(bundle.u_encoders[c]) for c in bundle.u_cat_cols]
    item_feat_vocabs = [len(bundle.i_encoders[c]) for c in bundle.i_cat_cols]

    model = DSINRanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
    model.set_feature_tables(
        torch.tensor(bundle.user_feat, dtype=torch.long),
        torch.tensor(bundle.item_feat, dtype=torch.long),
    )
    model.to(device)
    model.dsin.user_feat_table = model.dsin.user_feat_table.to(device)
    model.dsin.item_feat_table = model.dsin.item_feat_table.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    all_item_ids = torch.arange(bundle.n_items, device=device)

    best_ndcg = -1.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "dsin_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            uid_idx = batch["uid_idx"].to(device)
            sessions = batch["sessions"].to(device)
            session_counts = batch["session_counts"].to(device)
            session_item_mask = batch["session_item_mask"].to(device)
            session_mask = batch["session_mask"].to(device)
            target = batch["target"].to(device)

            # same anti-leakage convention as DIN: drive attention with the
            # most-recent real history item, never the true target.
            attn_cand = _last_real_item_per_row(sessions, session_item_mask)

            user_vec = model(uid_idx, sessions, session_counts, session_item_mask, session_mask, attn_cand)

            all_item_vecs = model.item_static_vec(all_item_ids)  # recomputed every step
            loss = full_softmax_loss(user_vec, target, all_item_vecs)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        with torch.no_grad():
            item_vecs_all = model.item_static_vec(all_item_ids)
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
# 5. Prediction (mirrors din_model.run_predict)
# ========================================================================================

@torch.no_grad()
def run_predict(model, bundle, cfg: "Config", device):
    test_df = bundle.test_df
    if test_df is None:
        print("[WARN] no test.csv found, skipping prediction.")
        return

    model.eval()
    ds = DSINDataset(test_df, bundle.uid_enc, bundle.iid_enc,
                      session_len=cfg.session_len, max_sessions=cfg.max_sessions, has_target=False)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=dsin_collate_fn)

    all_item_ids = torch.arange(bundle.n_items, device=device)
    item_vecs_all = model.item_static_vec(all_item_ids)

    rows = []
    uids = test_df["uid"].tolist()
    for batch in loader:
        uid_idx = batch["uid_idx"].to(device)
        sessions = batch["sessions"].to(device)
        session_counts = batch["session_counts"].to(device)
        session_item_mask = batch["session_item_mask"].to(device)
        session_mask = batch["session_mask"].to(device)

        attn_cand = _last_real_item_per_row(sessions, session_item_mask)
        user_vec = model(uid_idx, sessions, session_counts, session_item_mask, session_mask, attn_cand)
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
# 6. Main (mirrors din_model.main)
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
