# -*- coding: utf-8 -*-
"""
LESSR (Lossless Edge-order preserving aggregation and Shortcut graph attention
for Session-based Recommendation) for next-item / sequential recommendation,
implemented on top of PyTorch Geometric (PyG).

This is a drop-in architectural sibling of bst_model.py / sasrecf_model.py:
identical data schema, identical encoders / training & evaluation harness /
prediction / synthetic-data generator, but the user-interest extractor is
swapped from a sequence model (Transformer, in BST/SASRecF) to a GRAPH neural
network operating over TWO graphs built from each user's session, following
Chen & Wong, "Handling Information Loss of Graph Neural Networks for
Session-based Recommendation" (KDD 2020).

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
target item the user will interact with next. As in bst_model.py / sasrecf_model.py
this is cast as a candidate-ranking problem trained with a FULL softmax loss over
the entire item catalog (no negative sampling), matching the candidate space used
at evaluation time (ranking the full catalog for NDCG@10).

Output
------
- Training prints valid NDCG@10 each epoch.
- predict.py-equivalent (run_predict) produces submission.csv with columns:
    uid,prediction
  where prediction is a comma-quoted string of top-10 item ids, e.g.:
    u000009,"i001952,i001038,i001710,i001046,i000401,i001445,i001069,i001002,i001673,i000661"

How LESSR differs from BST / SASRecF here
--------------------------------------------
BST and SASRecF both treat a user's history as a 1-D SEQUENCE and use
attention (bidirectional-with-candidate-token / causal, respectively) over
it. LESSR instead treats each session as a small GRAPH with two views:

  1. EOP graph (Edge-Order-Preserving multigraph, `edge_index_seq`): one node
     per DISTINCT item the user interacted with (deduplicated, same "kept
     window" semantics as item_seq_dedup truncated to the most recent
     `max_seq_len` distinct items). For every consecutive pair of items in
     the RAW (repeats-included) interaction order, a directed edge is added
     from the earlier item's node to the later item's node -- and, crucially,
     REPEATED transitions are kept as separate, time-tagged multi-edges
     rather than merged into one weighted edge the way classic session-graph
     methods (e.g. SR-GNN) do. Merging loses the temporal order of repeated
     visits; LESSR's whole premise is that this is exactly the information a
     plain session graph throws away.
  2. Shortcut graph (`edge_index_sc`): a directed edge from EVERY earlier
     item to EVERY later item in the (deduplicated) session, i.e. all
     "shortcuts", not just consecutive ones. This lets information travel
     between distant items in one hop instead of being diluted over many
     GNN layers of only-local (EOP) propagation.

  Each LESSR layer alternates:
    - EOPA (Edge-Order Preserving Aggregation): for every node, its incoming
      EOP-graph neighbors are fed, IN TEMPORAL ORDER, into a GRU; the GRU's
      final hidden state is the node's aggregated "local" message. This is
      literally an RNN over each node's ordered visit history, which is how
      LESSR avoids losing edge-order information from the multigraph.
    - SGAT (Shortcut Graph Attention): a multi-head dot-product graph
      attention layer (built on PyG's `MessagePassing`) over the shortcut
      graph, giving every node a "global" message mixed from every other
      relevant item in the session in one hop.
    After each of EOPA/SGAT, the aggregated message is fused into the node's
    running representation via a GRUCell update (the same gated-update idea
    GGNN/LESSR use), rather than a plain residual add.

  Finally, a soft-attention READOUT (query = the embedding of the session's
  most-recently-interacted node, exactly SR-GNN/LESSR's readout mechanism)
  produces a session-level "global interest" vector, which is combined with
  the last item's own node embedding ("local interest") into the final
  session representation, concatenated with the user's side-feature
  embedding, and projected through an MLP to the shared scoring space --
  mirroring BST/SASRecF's mlp_in = user + interest concatenation.

How item_seq_counts is used (the "确保充分利用" requirement)
--------------------------------------------------------------
item_seq_counts is used in TWO complementary, deliberate ways so the signal
isn't just implicitly hinted at by graph structure:

  (a) EXPLICIT node feature: exactly like BST/SASRecF's frequency signal,
      log1p(count) is projected through a small linear layer and ADDED to
      every node's initial embedding before any graph layer runs, so every
      node "knows" its own overall visit frequency regardless of how many
      or how few edges happen to touch it in this particular session graph.
  (b) IMPLICIT structural reinforcement via the EOP multigraph: because
      repeated transitions are preserved as distinct multi-edges (rather
      than collapsed into one edge, which is what would happen if counts
      were only used as an edge WEIGHT the way plain session-graph methods
      do), a frequently-revisited item naturally receives more, and more
      temporally-informative, incoming messages through EOPA's GRU
      aggregation. The multigraph structure and the explicit count feature
      are therefore two independent, additive channels through which
      "how many times did the user interact with this item" reaches the
      model, instead of a single easily-diluted signal.

A note on the shortcut graph's cost
--------------------------------------
The shortcut graph has O(N^2) edges for a session with N distinct items, so
this script defaults `max_seq_len` (the number of most-recent distinct items
kept per session) considerably lower than BST/SASRecF's default of 100 --
tune `--max_seq_len` down further for very long, dense histories, or up if
your sessions are short and you want more shortcut connectivity.

Usage
-----
python lessr_model.py --data_dir /path/to/data --out_dir /path/to/output --epochs 5
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
from collections import defaultdict
from dataclasses import dataclass, fields

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.utils import scatter as pyg_scatter

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
    # NOTE: much smaller default than bst_model.py / sasrecf_model.py's 100 --
    # the shortcut graph is O(max_seq_len^2) edges per session, so keep this
    # modest unless you've confirmed your sessions/hardware can afford more.
    max_seq_len = 20           # truncate/pad history to this many distinct items (most recent kept)
    max_raw_mult = 5           # cap the raw (repeats-included) sequence used to build EOP edges to max_seq_len * max_raw_mult tokens
    val_frac = 0.1          # fraction of train.csv randomly held out as the valid set
    use_synthetic = False    # generate synthetic data into data_dir if real files are missing

    # model
    emb_dim = 32
    side_emb_ratio = 0.5    # side-feature embedding dim = emb_dim * side_emb_ratio
    n_layers = 2              # number of stacked (EOPA -> SGAT) layer pairs
    n_heads = 4              # SGAT multi-head attention heads
    mlp_hidden = (200, 80)  # final user-representation MLP hidden sizes
    dropout = 0.2

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
# 1. Vocab / encoding utilities  (identical to bst_model.py / sasrecf_model.py)
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
# 2. Data loading  (identical to bst_model.py / sasrecf_model.py)
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
# 3. Session-graph construction + PyG Dataset
# ========================================================================================

def build_session_graph(dedup_items, raw_items, counts_map, iid_enc, max_len, max_raw_len):
    """Builds the two graphs LESSR needs for one session, using LOCAL node
    indices (0..N-1). PyG's default `Data.__inc__` auto-offsets any
    attribute whose name contains the substring "index" when multiple
    graphs are batched together -- which is why every node-index-valued
    field below is deliberately named *_index (edge_index_seq,
    edge_index_sc, last_node_index), so batching "just works" without a
    custom Data subclass.

    Returns a dict of numpy/python values ready to be wrapped into a
    torch_geometric.data.Data object.
    """
    dedup_items = dedup_items[-max_len:]
    if len(dedup_items) == 0:
        # defensive fallback for a user with literally no history: a single
        # PAD node with no edges. transform_one() on an unseen sentinel
        # string returns PAD_IDX (0) automatically.
        dedup_items = ["<EMPTY_SESSION>"]

    kept_set = set(dedup_items)
    node_id_map = {it: i for i, it in enumerate(dedup_items)}  # first-seen order -> local node id
    N = len(dedup_items)

    # keep only occurrences of items that survived truncation, preserving
    # their original relative order -- this reconstructs "the raw order
    # restricted to the kept window" without needing to find an exact
    # cut point in item_seq_raw.
    raw_filtered = [it for it in raw_items if it in kept_set]
    raw_filtered = raw_filtered[-max_raw_len:]

    # ---- EOP (edge-order-preserving) multigraph: consecutive pairs in the
    # raw (repeats-included) order, as separate time-tagged edges. ----
    seq_src, seq_dst, seq_time = [], [], []
    for t in range(len(raw_filtered) - 1):
        u, v = raw_filtered[t], raw_filtered[t + 1]
        seq_src.append(node_id_map[u])
        seq_dst.append(node_id_map[v])
        seq_time.append(t)

    # ---- shortcut graph: every earlier distinct item -> every later
    # distinct item (first-seen order used as the ordering), O(N^2). ----
    sc_src, sc_dst = [], []
    for i in range(N):
        for j in range(i + 1, N):
            sc_src.append(i)
            sc_dst.append(j)

    node_item_idx = np.array([iid_enc.transform_one(it) for it in dedup_items], dtype=np.int64)
    node_count = np.array([counts_map.get(it, 1) for it in dedup_items], dtype=np.float32)

    if raw_filtered:
        last_node_index = node_id_map[raw_filtered[-1]]
    else:
        last_node_index = N - 1  # fall back to the most-recently-first-seen item

    return dict(
        node_item_idx=node_item_idx,
        node_count=node_count,
        seq_src=seq_src, seq_dst=seq_dst, seq_time=seq_time,
        sc_src=sc_src, sc_dst=sc_dst,
        last_node_index=last_node_index,
        n_nodes=N,
    )


class LessrDataset(torch.utils.data.Dataset):
    """Each sample is a torch_geometric.data.Data session-graph object (see
    build_session_graph). Unlike BSTDataset/SasRecDataset (which return
    plain padded tensors and need a custom collate_fn), PyG's own
    `torch_geometric.loader.DataLoader` already knows how to batch a list of
    Data objects into one big disjoint-union graph (with a `batch` vector
    marking which nodes belong to which graph) -- no collate_fn needed."""

    def __init__(self, df, uid_enc, iid_enc, max_len, has_target=True, max_raw_mult=5):
        self.uids = df["uid"].tolist()
        self.raw_seqs = df["item_seq_raw"].tolist()
        self.dedup_seqs = df["item_seq_dedup"].tolist()
        self.count_strs = df["item_seq_counts"].tolist()
        self.has_target = has_target
        if has_target:
            self.targets = df["target_iid"].tolist()
        self.uid_enc = uid_enc
        self.iid_enc = iid_enc
        self.max_len = max_len
        self.max_raw_len = max_len * max_raw_mult

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid_idx = self.uid_enc.transform_one(self.uids[idx])
        dedup_items = parse_seq_raw(self.dedup_seqs[idx])
        raw_items = parse_seq_raw(self.raw_seqs[idx])
        counts_map = parse_seq_counts(self.count_strs[idx])

        g = build_session_graph(dedup_items, raw_items, counts_map, self.iid_enc,
                                 self.max_len, self.max_raw_len)

        data = Data()
        data.x = torch.as_tensor(g["node_item_idx"], dtype=torch.long)
        data.node_count = torch.as_tensor(g["node_count"], dtype=torch.float32)
        if g["seq_src"]:
            data.edge_index_seq = torch.tensor([g["seq_src"], g["seq_dst"]], dtype=torch.long)
            data.edge_time_seq = torch.tensor(g["seq_time"], dtype=torch.long)
        else:
            data.edge_index_seq = torch.empty((2, 0), dtype=torch.long)
            data.edge_time_seq = torch.empty((0,), dtype=torch.long)
        if g["sc_src"]:
            data.edge_index_sc = torch.tensor([g["sc_src"], g["sc_dst"]], dtype=torch.long)
        else:
            data.edge_index_sc = torch.empty((2, 0), dtype=torch.long)
        data.last_node_index = torch.tensor([g["last_node_index"]], dtype=torch.long)
        data.uid_idx = torch.tensor([uid_idx], dtype=torch.long)
        data.num_nodes = g["n_nodes"]
        if self.has_target:
            data.target = torch.tensor([self.iid_enc.transform_one(self.targets[idx])], dtype=torch.long)
        return data


# ========================================================================================
# 4. LESSR model
# ========================================================================================

class EOPALayer(nn.Module):
    """Edge-Order Preserving Aggregation. For every destination node, its
    incoming EOP-graph neighbor embeddings are fed, IN TEMPORAL ORDER, into a
    GRU; the GRU's final hidden state is that node's aggregated "local"
    message. Nodes with no incoming edges get a zero message. The message is
    then fused into the node's running representation with a GRUCell -- a
    gated update rather than a plain residual add, following LESSR/GGNN."""

    def __init__(self, dim):
        super().__init__()
        self.msg_gru = nn.GRU(dim, dim, batch_first=True)
        self.update_cell = nn.GRUCell(dim, dim)

    def forward(self, h, edge_index_seq, edge_time_seq, num_nodes):
        device = h.device
        dim = h.size(-1)
        m = torch.zeros(num_nodes, dim, device=device)

        src, dst = edge_index_seq
        if src.numel() > 0:
            # stable-sort by time first, then by dst: the second (also
            # stable) sort preserves the relative time order WITHIN each
            # destination's group of incoming edges.
            time_order = torch.argsort(edge_time_seq, stable=True)
            src_t, dst_t = src[time_order], dst[time_order]
            dst_order = torch.argsort(dst_t, stable=True)
            src_s, dst_s = src_t[dst_order], dst_t[dst_order]

            uniq_dst, counts = torch.unique_consecutive(dst_s, return_counts=True)
            counts_list = counts.tolist()
            max_deg = max(counts_list)
            n_groups = uniq_dst.size(0)

            group_idx = torch.repeat_interleave(torch.arange(n_groups, device=device), counts)
            group_pos = torch.cat([torch.arange(c, device=device) for c in counts_list])

            padded = torch.zeros(n_groups, max_deg, dim, device=device)
            padded[group_idx, group_pos] = h[src_s]

            packed = nn.utils.rnn.pack_padded_sequence(
                padded, counts.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.msg_gru(packed)  # h_n: [1, n_groups, dim] -- already the correct
            # last-valid-step hidden state per group, regardless of padding, because
            # pack_padded_sequence tells the GRU each group's true length.
            m[uniq_dst] = h_n.squeeze(0)

        return self.update_cell(m, h)


class SGATLayer(MessagePassing):
    """Shortcut Graph Attention. Standard multi-head dot-product graph
    attention over the shortcut graph, implemented via PyG's `MessagePassing`
    (message() computes per-edge, per-head attention logits and applies
    `torch_geometric.utils.softmax` grouped by destination node; `aggr='add'`
    then sums the attention-weighted values per destination). The aggregated
    message is fused into the node's running representation with a GRUCell,
    matching EOPALayer's update rule."""

    def __init__(self, dim, n_heads):
        super().__init__(aggr="add", node_dim=0)
        assert dim % n_heads == 0, "item_full_dim must be divisible by n_heads for SGAT"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.update_cell = nn.GRUCell(dim, dim)

    def forward(self, h, edge_index_sc, num_nodes):
        if edge_index_sc.numel() == 0:
            m = torch.zeros_like(h)
        else:
            q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
            m = self.propagate(edge_index_sc, q=q, k=k, v=v, size=(num_nodes, num_nodes))
            m = self.out_proj(m)
        return self.update_cell(m, h)

    def message(self, q_i, k_j, v_j, index, size_i):
        H, Dh = self.n_heads, self.head_dim
        q_i = q_i.view(-1, H, Dh)
        k_j = k_j.view(-1, H, Dh)
        v_j = v_j.view(-1, H, Dh)
        score = (q_i * k_j).sum(-1) / math.sqrt(Dh)          # [E, H]
        alpha = pyg_softmax(score, index, num_nodes=size_i)   # softmax over incoming edges per dst, per head
        out = alpha.unsqueeze(-1) * v_j                        # [E, H, Dh]
        return out.reshape(-1, H * Dh)


class LESSR(nn.Module):
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
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=PAD_IDX)  # used both as node & scoring emb

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

        # explicit frequency signal from item_seq_counts -- see module
        # docstring, channel (a). 0 vector would be added for a count of 0,
        # but every real node has count >= 1 by construction.
        self.count_proj = nn.Linear(1, item_full_dim)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "eopa": EOPALayer(item_full_dim),
                "sgat": SGATLayer(item_full_dim, cfg.n_heads),
            })
            for _ in range(cfg.n_layers)
        ])
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # soft-attention readout (SR-GNN/LESSR style): query = last-item
        # node embedding, key = every node's embedding.
        self.readout_q = nn.Linear(item_full_dim, item_full_dim)
        self.readout_k = nn.Linear(item_full_dim, item_full_dim)
        self.readout_score = nn.Linear(item_full_dim, 1)
        # combine global (attention-pooled) + local (last-item) session views
        self.combine = nn.Linear(item_full_dim * 2, item_full_dim)

        # final MLP: user side feat + session representation
        # (mirrors BST/SASRecF's mlp_in = user_full + interest concatenation)
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

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.user_feat_table = user_feat_table
        self.item_feat_table = item_feat_table

    def _item_full_emb(self, item_idx):
        base = self.item_emb(item_idx)  # [..., emb_dim]
        feats = self.item_feat_table[item_idx]  # [..., n_i_cat]
        side_parts = [emb(feats[..., j]) for j, emb in enumerate(self.item_side_embs)]
        return torch.cat([base] + side_parts, dim=-1)

    def _user_full_emb(self, uid_idx):
        base = self.user_emb(uid_idx)  # [B, emb_dim]
        feats = self.user_feat_table[uid_idx]  # [B, n_u_cat]
        side_parts = [emb(feats[..., j]) for j, emb in enumerate(self.user_side_embs)]
        return torch.cat([base] + side_parts, dim=-1)

    def encode_user(self, batch):
        """Produces the user representation vector for scoring against item
        embeddings, from a batched torch_geometric.data.Batch of session
        graphs (see LessrDataset). Unlike BST/SASRecF's forward(uid, hist,
        hist_cnt, hist_len[, cand]) signature, LESSR takes the WHOLE graph
        batch as a single argument, since node/edge counts vary per sample
        and PyG's Batch object is what carries that variable structure.
        """
        node_item_idx = batch.x
        h0 = self._item_full_emb(node_item_idx)  # [N_total, item_full_dim]

        # channel (a): explicit item_seq_counts frequency signal, added to
        # every node's initial embedding (see module docstring).
        freq = self.count_proj(torch.log1p(batch.node_count).unsqueeze(-1))
        h = h0 + freq
        h = self.emb_dropout(h)

        num_nodes = h.size(0)
        for layer in self.layers:
            # channel (b): the EOP multigraph passed to EOPA below preserves
            # repeated transitions as distinct, time-ordered edges (never
            # merged), so frequently-revisited items keep receiving
            # temporally-informative messages proportional to how often
            # they were actually revisited.
            h = layer["eopa"](h, batch.edge_index_seq, batch.edge_time_seq, num_nodes)
            h = layer["sgat"](h, batch.edge_index_sc, num_nodes)

        # ---- soft-attention readout ----
        last_h = h[batch.last_node_index]           # [B, D] one row per graph
        last_h_per_node = last_h[batch.batch]        # [N_total, D] broadcast to every node in its graph
        score = self.readout_score(torch.sigmoid(self.readout_q(last_h_per_node) + self.readout_k(h)))  # [N_total, 1]
        alpha = pyg_softmax(score.squeeze(-1), batch.batch, num_nodes=batch.num_graphs)  # normalized per graph
        s_g = pyg_scatter(alpha.unsqueeze(-1) * h, batch.batch, dim=0,
                           dim_size=batch.num_graphs, reduce="sum")  # [B, D] global session view
        s_h = self.combine(torch.cat([s_g, last_h], dim=-1))  # [B, D] final session representation

        user_full = self._user_full_emb(batch.uid_idx)  # [B, user_full_dim]
        x_cat = torch.cat([user_full, s_h], dim=-1)
        user_vec = self.mlp(x_cat)  # [B, emb_dim]
        return user_vec


class LESSRRanker(nn.Module):
    """Wraps LESSR: trains with a FULL softmax over the entire item catalog
    using the user vector produced by LESSR's graph encoder, and a separate
    static item-scoring embedding (item_score_head) used to score ALL items
    at both train and inference time -- identical strategy to
    BSTRanker/SASRecFRanker, so results stay directly comparable across
    model variants."""

    def __init__(self, n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg: "Config"):
        super().__init__()
        self.lessr = LESSR(n_users, n_items, user_feat_vocabs, item_feat_vocabs, cfg)
        emb_dim = cfg.emb_dim
        side_dim = max(1, int(emb_dim * cfg.side_emb_ratio))
        item_full_dim = emb_dim + side_dim * len(item_feat_vocabs)
        self.item_score_head = nn.Linear(item_full_dim, emb_dim)  # static item vector for full-catalog scoring
        self.n_items = n_items

    def set_feature_tables(self, user_feat_table, item_feat_table):
        self.lessr.set_feature_tables(user_feat_table, item_feat_table)

    def item_static_vec(self, item_idx):
        full = self.lessr._item_full_emb(item_idx)
        return self.item_score_head(full)

    def forward(self, batch):
        return self.lessr.encode_user(batch)  # [B, D]

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
# 6. NDCG@10 metric  (identical to bst_model.py / sasrecf_model.py)
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
        batch = batch.to(device)
        user_vec = model(batch)  # [B, D]
        scores = model.score_against_catalog(user_vec, item_vecs_all)  # [B, n_items]
        scores[:, PAD_IDX] = -1e9  # never recommend pad

        topk = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
        target_np = batch.target.cpu().numpy()
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

    train_ds = LessrDataset(tr_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len,
                             has_target=True, max_raw_mult=cfg.max_raw_mult)
    val_ds = LessrDataset(val_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len,
                           has_target=True, max_raw_mult=cfg.max_raw_mult)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    user_feat_vocabs = [len(bundle.u_encoders[c]) for c in bundle.u_cat_cols]
    item_feat_vocabs = [len(bundle.i_encoders[c]) for c in bundle.i_cat_cols]

    model = LESSRRanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
    model.set_feature_tables(
        torch.tensor(bundle.user_feat, dtype=torch.long),
        torch.tensor(bundle.item_feat, dtype=torch.long),
    )
    model.to(device)
    model.lessr.user_feat_table = model.lessr.user_feat_table.to(device)
    model.lessr.item_feat_table = model.lessr.item_feat_table.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    all_item_ids = torch.arange(bundle.n_items, device=device)

    best_ndcg = -1.0
    os.makedirs(cfg.out_dir, exist_ok=True)
    best_path = os.path.join(cfg.out_dir, "lessr_best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            target = batch.target

            user_vec = model(batch)

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
    ds = LessrDataset(test_df, bundle.uid_enc, bundle.iid_enc, max_len=cfg.max_seq_len,
                       has_target=False, max_raw_mult=cfg.max_raw_mult)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    all_item_ids = torch.arange(bundle.n_items, device=device)
    item_vecs_all = model.item_static_vec(all_item_ids)

    rows = []
    uids = test_df["uid"].tolist()
    for batch in loader:
        batch = batch.to(device)
        user_vec = model(batch)
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
