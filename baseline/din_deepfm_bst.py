"""
DIN + DeepFM + BST  端到端推荐系统
====================================

架构设计
--------
BST (Behavior Sequence Transformer) 是本模型的核心序列编码器。
与 SASRec 的因果自注意力不同，BST 使用**双向全局自注意力**，
将候选 item 拼入序列一起编码，让每个历史行为都能感知候选 item，
从而实现序列行为与候选的深度交叉。

但 BST 原始形态依赖候选 item，无法对全量 item 直接打分。
本实现采用以下策略兼顾效果与效率：

  训练时（BST 模式）
    将 target item 拼入序列末尾，用双向 Transformer 编码，
    取 target 位置的输出作为"序列-候选交叉表示"
    → 全量 Softmax CE 训练

  推理时（高效近似）
    不逐个拼入候选，而是：
    ① 用双向 Transformer 对序列单独编码，取 [CLS] 聚合表示
    ② 同时取序列最后位置表示
    ③ 两者拼接 → MLP → user_repr
    → user_repr 与全量 item_repr 矩阵乘法，一次完成打分

整体流程
--------

  ┌──────────────────────────────────────────────┐
  │               BST Encoder                    │
  │  [CLS] + 历史序列 + [target]（训练时）        │
  │  位置编码 = 绝对位置 + item特征融合            │
  │  双向 Transformer（无因果 mask）               │
  │  → 取 [CLS] 位 + 取 target 位（训练）         │
  └──────────────────────────────────────────────┘
          ↓                         ↓
    seq_cls_repr (D)         target_repr (D)   ← 训练专用
          ↓
  用户特征8域 → FM二阶 → user_fm (E)
          ↓
  DIN：user_feat_vec 为 query，序列各位置为 key/value
       → din_interest (D)
          ↓
  concat[seq_cls_repr; din_interest; user_fm_proj]
  → MLP → user_repr (D)

  Item Tower（与 v2 一致）
  item_id + 4域特征 → FM二阶 + MLP → item_repr (D)

  打分（训练）
    main_score  = cosine(user_repr, item_repr[target]) * τ
    bst_score   = W · target_repr                        ← BST 交叉分
    final_score = main_score + bst_score
    loss = 全量 Softmax CE（label smoothing）
         + 辅助 Masked Item Prediction × aux_weight

  打分（推理）
    score = cosine(user_repr, all_item_repr) * τ

数值稳定保护
------------
  - F.normalize eps=1e-8
  - log_temp clamp(-4.6, 4.6)
  - logits clamp(-50, 50)
  - train_epoch 跳过 NaN/Inf batch
  - SASRec block nan_to_num
"""

import math, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# ══════════════════════════════════════════════════════════
# 0. 配置
# ══════════════════════════════════════════════════════════
class Config:
    train_path   = "../data/A2-Rec/train.csv"
    test_path    = "../data/A2-Rec/test.csv"
    user_path    = "../data/A2-Rec/user.csv"
    item_path    = "../data/A2-Rec/item.csv"
    output_path  = "submission_dd_bst.csv"

    max_seq_len  = 50       # 历史序列最大长度（不含 CLS/target token）
    emb_dim      = 32       # 每个特征域 embedding 维度
    repr_dim     = 128      # user / item tower 输出维度

    # BST Transformer
    n_heads      = 4
    n_layers     = 2        # BST 层数（双向，比因果模型需要的层数少）

    # DIN
    din_hidden   = 64

    # MLP
    user_mlp_dims = [256, 128]
    item_mlp_dims = [256, 128]
    dropout       = 0.2

    # 训练
    epochs        = 50
    batch_size    = 256
    lr            = 1e-3
    weight_decay  = 1e-5
    label_smooth  = 0.0     # 数据少时关闭，数据多时建议 0.05~0.1
    aux_weight    = 0.05    # Masked Item Prediction 辅助损失权重
    mask_prob     = 0.15    # 序列 mask 概率
    bst_weight    = 0.3     # BST 交叉分在训练 loss 中的权重
    mixup_alpha   = 0.1     # Mixup 强度，0=关闭

    seed          = 42
    topk          = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)


# ══════════════════════════════════════════════════════════
# 1. 数据处理
# ══════════════════════════════════════════════════════════
class DataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.item2id  = {}
        self.id2item  = {}
        self.iid2feat = {}
        self.uid2feat = {}
        self.item_feat_cols    = ["i_cat_01","i_cat_02","i_cat_03","i_bucket_01"]
        self.user_feat_cols    = [f"u_cat_0{i}" for i in range(1, 9)]
        self.item_feat_dims    = []
        self.item_feat_padidxs = []
        self.user_feat_dims    = []
        self.user_feat_padidxs = []
        self.n_items = 0

    def _parse_seq(self, s):
        if pd.isna(s) or not str(s).strip(): return []
        return [x.strip() for x in str(s).split(",") if x.strip()]

    def _parse_counts(self, s):
        if pd.isna(s) or not str(s).strip(): return {}
        d = {}
        for part in str(s).split(","):
            if ":" in part:
                k, v = part.strip().rsplit(":", 1)
                d[k.strip()] = int(v)
        return d

    def _weight_seq(self, seq_strs, counts):
        base  = [self.item2id[i] for i in seq_strs if i in self.item2id]
        extra = [self.item2id[i] for i, c in counts.items()
                 if c >= 2 and i in self.item2id]
        return base + extra

    def load_and_build(self):
        print("Loading data...")
        train_df = pd.read_csv(self.cfg.train_path)
        test_df  = pd.read_csv(self.cfg.test_path)
        user_df  = pd.read_csv(self.cfg.user_path)
        item_df  = pd.read_csv(self.cfg.item_path)
        for df in [train_df, test_df, user_df, item_df]:
            df.columns = df.columns.str.strip()

        # ── item 词表 ──
        all_items = set(item_df["iid"].str.strip())
        for col in ["item_seq_raw","item_seq_dedup"]:
            for df in [train_df, test_df]:
                if col in df.columns:
                    for s in df[col].dropna():
                        all_items.update(self._parse_seq(s))
        all_items.update(train_df["target_iid"].str.strip().dropna())
        all_items = sorted(all_items)
        # 0=PAD, 1..N=item, N+1=CLS, N+2=MASK
        self.item2id = {iid: idx + 1 for idx, iid in enumerate(all_items)}
        self.id2item = {v: k for k, v in self.item2id.items()}
        self.n_items = len(self.item2id)
        self.CLS_ID  = self.n_items + 1
        self.MASK_ID = self.n_items + 2
        print(f"  Total items : {self.n_items}  CLS={self.CLS_ID}  MASK={self.MASK_ID}")

        # ── item 特征（pad_idx = max+1，emb_size = max+2）──
        item_df["iid"] = item_df["iid"].str.strip()
        for col in self.item_feat_cols:
            item_df[col] = item_df[col].fillna(-1).astype(int)
        for col in self.item_feat_cols:
            rmax = int(item_df[col].max())
            self.item_feat_padidxs.append(rmax + 1)
            self.item_feat_dims.append(rmax + 2)
        for _, row in item_df.iterrows():
            iid = self.item2id.get(row["iid"])
            if iid:
                self.iid2feat[iid] = [int(row[c]) for c in self.item_feat_cols]
        pad_ifeat = list(self.item_feat_padidxs)
        for iid in self.id2item:
            if iid not in self.iid2feat:
                self.iid2feat[iid] = pad_ifeat

        # ── user 特征 ──
        user_df["uid"] = user_df["uid"].str.strip()
        for col in self.user_feat_cols:
            user_df[col] = user_df[col].fillna(-1).astype(int)
        for col in self.user_feat_cols:
            rmax = int(user_df[col].max())
            self.user_feat_padidxs.append(rmax + 1)
            self.user_feat_dims.append(rmax + 2)
        uid_feat_map = {row["uid"]: [int(row[c]) for c in self.user_feat_cols]
                        for _, row in user_df.iterrows()}
        pad_ufeat = list(self.user_feat_padidxs)

        # ── 样本 ──
        train_df["uid"]        = train_df["uid"].str.strip()
        train_df["target_iid"] = train_df["target_iid"].str.strip()
        test_df["uid"]         = test_df["uid"].str.strip()

        train_samples, test_samples = [], []
        for _, row in train_df.iterrows():
            uid    = row["uid"]
            target = self.item2id.get(row["target_iid"])
            if not target: continue
            seq  = self._parse_seq(row["item_seq_dedup"])
            cnts = self._parse_counts(row.get("item_seq_counts", ""))
            train_samples.append({
                "uid":       uid,
                "seq":       self._weight_seq(seq, cnts),
                "target":    target,
                "user_feat": uid_feat_map.get(uid, pad_ufeat),
            })
        for _, row in test_df.iterrows():
            uid  = row["uid"]
            seq  = self._parse_seq(row["item_seq_dedup"])
            cnts = self._parse_counts(row.get("item_seq_counts", ""))
            test_samples.append({
                "uid":       uid,
                "seq":       self._weight_seq(seq, cnts),
                "user_feat": uid_feat_map.get(uid, pad_ufeat),
            })
        print(f"  Train: {len(train_samples)} | Test: {len(test_samples)}")

        # 全量 item 特征张量 (N+3, 4)  index 0=PAD, N+1=CLS, N+2=MASK 用 pad 值填充
        vocab_size = self.n_items + 3
        ift = torch.zeros(vocab_size, 4, dtype=torch.long)
        for iid, feats in self.iid2feat.items():
            ift[iid] = torch.tensor(feats)
        # CLS/MASK 的特征用 pad 值（不影响 embedding，padding_idx 会归零）
        ift[self.CLS_ID]  = torch.tensor(pad_ifeat)
        ift[self.MASK_ID] = torch.tensor(pad_ifeat)
        self.item_feat_tensor = ift

        return train_samples, test_samples


# ══════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════
def pad_seq(seq, max_len):
    """左 padding，返回固定长度列表"""
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


class RecDataset(Dataset):
    """
    训练：返回 (seq, masked_seq, user_feat, target, mask_targets)
    测试：返回 (seq, user_feat, uid)

    BST 训练时额外需要 target 用于拼接序列，由模型内部处理。
    """
    def __init__(self, samples, max_len, n_items, mask_id,
                 mask_prob=0.0, mode="train"):
        self.samples   = samples
        self.max_len   = max_len
        self.n_items   = n_items
        self.mask_id   = mask_id
        self.mask_prob = mask_prob
        self.mode      = mode

    def __len__(self): return len(self.samples)

    def _mask(self, seq_ids):
        masked  = seq_ids.copy()
        targets = [0] * len(seq_ids)
        for i, sid in enumerate(seq_ids):
            if sid != 0 and random.random() < self.mask_prob:
                targets[i] = sid
                masked[i]  = self.mask_id
        return masked, targets

    def __getitem__(self, idx):
        s   = self.samples[idx]
        seq = pad_seq(s["seq"], self.max_len)
        uf  = torch.tensor(s["user_feat"], dtype=torch.long)
        if self.mode == "train":
            masked, mtgt = self._mask(seq)
            return (
                torch.tensor(seq,    dtype=torch.long),
                torch.tensor(masked, dtype=torch.long),
                uf,
                torch.tensor(s["target"], dtype=torch.long),
                torch.tensor(mtgt,        dtype=torch.long),
            )
        return torch.tensor(seq, dtype=torch.long), uf, s["uid"]


# ══════════════════════════════════════════════════════════
# 3. 模型组件
# ══════════════════════════════════════════════════════════

def build_mlp(in_dim, hidden_dims, out_dim, dropout):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def fm_second_order(emb_list):
    """list of (..., E) → (..., E)  FM 二阶交叉"""
    stacked = torch.stack(emb_list, dim=-2)
    return 0.5 * (stacked.sum(-2)**2 - (stacked**2).sum(-2))


# ── BST Block（双向 Transformer，无因果 mask）──
class BSTBlock(nn.Module):
    """
    标准 Transformer Encoder 块，双向全局自注意力。
    与 SASRec Block 的区别：不传入 causal_mask，历史 item 与目标 item 互相可见。
    """
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        """
        x        : (B, L, d)
        pad_mask : (B, L) bool  True = padding 位置，不参与 attention
        """
        r = x; x = self.n1(x)
        x, _ = self.attn(x, x, x, key_padding_mask=pad_mask, need_weights=False)
        x = torch.nan_to_num(self.drop(x), nan=0.0) + r
        r = x; x = self.n2(x)
        return torch.nan_to_num(self.drop(self.ff(x)), nan=0.0) + r


# ── BST Encoder ──
class BSTEncoder(nn.Module):
    """
    输入：
      seq_ids     (B, L)       历史序列（含 PAD=0，不含 CLS/target）
      target_ids  (B,)         候选 item id（训练时传入，推理时为 None）
      item_feats  (vocab, 4)   全量 item 特征查找表

    序列构造：
      训练：[CLS] + seq + [target]  长度 = 1 + L + 1
      推理：[CLS] + seq             长度 = 1 + L

    位置编码：
      item_id_emb(E) + 4个特征域 emb 均值 → 融合投影 → D 维
      + 可学习绝对位置编码(D)

    输出：
      cls_repr    (B, D)       [CLS] 位置的 hidden state，作为序列全局表示
      target_repr (B, D)       [target] 位置的 hidden state（训练时），推理时为 None
      all_hidden  (B, L, D)    序列（不含 CLS 和 target）的 hidden states，供 DIN 使用
    """
    def __init__(self, cfg, n_items, item_feat_dims, item_feat_padidxs):
        super().__init__()
        E = cfg.emb_dim
        D = cfg.repr_dim
        max_total = cfg.max_seq_len + 2  # CLS + seq + target

        # item token embedding（id + 4个特征域均值融合）
        vocab_size = n_items + 3         # 0=PAD, 1..N=item, N+1=CLS, N+2=MASK
        self.item_id_emb = nn.Embedding(vocab_size, E, padding_idx=0)
        self.feat_embs   = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)
        ])
        # 融合投影：item_id(E) + feat_mean(E) → D
        self.token_proj  = nn.Linear(E * 2, D)

        # 可学习绝对位置编码
        self.pos_emb = nn.Embedding(max_total + 1, D)

        # Transformer 层
        self.layers  = nn.ModuleList([
            BSTBlock(D, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm    = nn.LayerNorm(D)
        self.drop    = nn.Dropout(cfg.dropout)

        self.max_seq = cfg.max_seq_len
        self.CLS_ID  = n_items + 1

    def _token_emb(self, token_ids, ift):
        """
        token_ids : (...,) long
        ift       : (vocab, 4) long
        return    : (..., D)
        将 item_id embedding 与 4个特征域 embedding 均值融合后投影到 D 维
        """
        id_e  = self.item_id_emb(token_ids)                    # (..., E)
        feats = ift[token_ids]                                  # (..., 4)
        f_e   = torch.stack(
            [e(feats[..., j]) for j, e in enumerate(self.feat_embs)],
            dim=-2).mean(-2)                                    # (..., E)
        x = torch.cat([id_e, f_e], dim=-1)                     # (..., 2E)
        return F.gelu(self.token_proj(x))                       # (..., D)

    def forward(self, seq_ids, ift, target_ids=None):
        B, L   = seq_ids.shape
        device = seq_ids.device
        ift    = ift.to(device)

        # ── 构造序列 ──
        cls_tok = torch.full((B, 1), self.CLS_ID,
                             dtype=torch.long, device=device)   # (B, 1)
        if target_ids is not None:
            # 训练：[CLS, seq..., target]
            tgt_tok   = target_ids.unsqueeze(1)                  # (B, 1)
            token_ids = torch.cat([cls_tok, seq_ids, tgt_tok], dim=1)  # (B, L+2)
            seq_len   = L + 2
        else:
            # 推理：[CLS, seq...]
            token_ids = torch.cat([cls_tok, seq_ids], dim=1)    # (B, L+1)
            seq_len   = L + 1

        # ── token embedding ──
        x = self._token_emb(token_ids, ift)                     # (B, seq_len, D)

        # ── 位置编码（1-indexed，0 留给 PAD）──
        pos = torch.arange(1, seq_len + 1, device=device).unsqueeze(0)
        x   = self.drop(x + self.pos_emb(pos))                 # (B, seq_len, D)

        # ── padding mask：原始 seq_ids==0 的位置 → True；CLS 和 target 位永不 mask ──
        seq_pad  = (seq_ids == 0)                               # (B, L)
        cls_pad  = torch.zeros(B, 1, dtype=torch.bool, device=device)
        if target_ids is not None:
            tgt_pad  = torch.zeros(B, 1, dtype=torch.bool, device=device)
            pad_mask = torch.cat([cls_pad, seq_pad, tgt_pad], dim=1)  # (B, L+2)
        else:
            pad_mask = torch.cat([cls_pad, seq_pad], dim=1)    # (B, L+1)

        # ── Transformer ──
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask)
        x = self.norm(x)                                        # (B, seq_len, D)

        # ── 提取各部分输出 ──
        cls_repr    = x[:, 0, :]                                # (B, D)  [CLS]
        seq_hidden  = x[:, 1:1+L, :]                           # (B, L, D) 历史位置
        target_repr = x[:, -1, :] if target_ids is not None else None  # (B, D)

        return cls_repr, seq_hidden, target_repr


# ── DIN Attention（用户特征 query 池化，无候选依赖）──
class DINPooling(nn.Module):
    """
    以 user_feat_vec 为 query，对序列 hidden states 做 attention 池化。
    不依赖候选 item，与全量打分兼容。
    activation unit 输入：[query; key; query-key; query*key] → MLP → 标量
    """
    def __init__(self, d, hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d * 4, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, query, seq_hidden, pad_mask):
        """
        query      : (B, D)
        seq_hidden : (B, L, D)
        pad_mask   : (B, L) True=padding
        return     : (B, D)
        """
        B, L, D = seq_hidden.shape
        q_exp = query.unsqueeze(1).expand(-1, L, -1)            # (B, L, D)
        inp   = torch.cat([q_exp, seq_hidden,
                           q_exp - seq_hidden,
                           q_exp * seq_hidden], dim=-1)          # (B, L, 4D)
        score  = self.mlp(inp).squeeze(-1)                       # (B, L)
        score  = score.masked_fill(pad_mask, -1e9)
        weight = torch.softmax(score, dim=-1)                    # (B, L)
        return (weight.unsqueeze(-1) * seq_hidden).sum(1)        # (B, D)


# ── DeepFM 特征交叉：FM 二阶 + Deep MLP ──
class DeepFMCross(nn.Module):
    """
    输入：多个特征域的 embedding list（每个 (B, E)）
    输出：(B, out_dim)  FM二阶交叉 + Deep MLP 的融合表示
    """
    def __init__(self, n_fields, emb_dim, mlp_dims, out_dim, dropout):
        super().__init__()
        self.fm_out   = emb_dim                        # FM 二阶输出维度 = E
        deep_in       = n_fields * emb_dim
        self.deep_mlp = build_mlp(deep_in, mlp_dims, out_dim, dropout)
        self.fm_proj  = nn.Linear(emb_dim, out_dim)    # FM 输出投影

    def forward(self, emb_list):
        """emb_list: list of (B, E)"""
        # FM 二阶
        fm2 = fm_second_order(emb_list)               # (B, E)
        fm2 = self.fm_proj(fm2)                       # (B, out_dim)
        # Deep MLP
        flat = torch.cat(emb_list, dim=-1)            # (B, n_fields*E)
        deep = self.deep_mlp(flat)                    # (B, out_dim)
        return fm2 + deep                             # (B, out_dim)


# ── Item Tower ──
class ItemTower(nn.Module):
    """
    item_id + 4个特征域 → FM二阶 + DeepFM MLP → item_repr (D)
    独立的 item_id embedding（不与 BST 共享，避免梯度冲突）
    """
    def __init__(self, cfg, n_items, item_feat_dims, item_feat_padidxs):
        super().__init__()
        E   = cfg.emb_dim
        D   = cfg.repr_dim
        n_i = len(item_feat_dims)   # 4

        # item_id emb（独立）
        self.item_id_emb = nn.Embedding(n_items + 1, E, padding_idx=0)
        # 4个特征域 emb
        self.feat_embs   = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)
        ])
        # n_fields = item_id(1) + 4个特征域 = 5
        self.cross = DeepFMCross(
            n_fields=1 + n_i, emb_dim=E,
            mlp_dims=cfg.item_mlp_dims, out_dim=D, dropout=cfg.dropout)

    def forward(self, item_ids, item_feats):
        """
        item_ids   : (...,) long
        item_feats : (..., 4) long
        return     : (..., D)
        """
        id_e    = self.item_id_emb(item_ids)
        fi_list = [e(item_feats[..., j]) for j, e in enumerate(self.feat_embs)]
        return self.cross([id_e] + fi_list)


# ── User Tower ──
class UserTower(nn.Module):
    """
    BST Encoder 输出 + 用户特征 → user_repr (D)

    组成：
      1. 用户8个特征域 → FM二阶 + DeepFM → user_cross (D)
      2. DIN 池化（user_feat_vec 为 query） → din_interest (D)
      3. concat[cls_repr; din_interest; user_cross] → MLP → user_repr (D)
    """
    def __init__(self, cfg, user_feat_dims, user_feat_padidxs):
        super().__init__()
        E   = cfg.emb_dim
        D   = cfg.repr_dim
        n_u = len(user_feat_dims)   # 8

        # 用户特征域 embedding
        self.u_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(user_feat_dims, user_feat_padidxs)
        ])
        # 用户特征 FM二阶 + Deep
        self.user_cross = DeepFMCross(
            n_fields=n_u, emb_dim=E,
            mlp_dims=[D], out_dim=D, dropout=cfg.dropout)

        # DIN 兴趣池化
        self.din = DINPooling(D, cfg.din_hidden)

        # 融合：cls_repr(D) + din_interest(D) + user_cross(D) → MLP → D
        self.out_mlp = build_mlp(D * 3, cfg.user_mlp_dims, D, cfg.dropout)

    def forward(self, cls_repr, seq_hidden, seq_ids, user_feats):
        """
        cls_repr   : (B, D)     BST [CLS] 输出
        seq_hidden : (B, L, D)  BST 序列位置输出（供 DIN 使用）
        seq_ids    : (B, L)     原始序列 id（用于构造 pad_mask）
        user_feats : (B, 8)     用户特征
        return     : (B, D)
        """
        # 用户特征域 embedding 列表
        u_emb_list = [e(user_feats[:, j]) for j, e in enumerate(self.u_embs)]
        # 用户特征 FM + Deep 交叉
        user_cross = self.user_cross(u_emb_list)               # (B, D)

        # DIN：以 user_cross 为 query，对序列做 attention 池化
        pad_mask    = (seq_ids == 0)                            # (B, L)
        din_interest = self.din(user_cross, seq_hidden, pad_mask)  # (B, D)

        # 融合
        concat = torch.cat([cls_repr, din_interest, user_cross], dim=-1)
        out    = self.out_mlp(concat)
        return torch.nan_to_num(out, nan=0.0)                   # (B, D)


# ══════════════════════════════════════════════════════════
# 4. 主模型
# ══════════════════════════════════════════════════════════
class DINDeepFMBST(nn.Module):
    """
    DIN + DeepFM + BST 端到端推荐模型

    训练策略
    --------
    main loss  : 全量 Softmax CE
                 score = cosine(user_repr, item_repr) × τ
    bst  loss  : 用 BST target 位输出直接预测 item（辅助监督）
                 score_bst = W · target_repr → Softmax CE
    aux  loss  : Masked Item Prediction（序列随机 mask 后预测）

    total = main_loss + bst_weight × bst_loss + aux_weight × aux_loss

    推理策略
    --------
    仅使用 main score 路径（不需要候选 item），
    预计算全量 item_repr，一次矩阵乘法完成打分。
    """
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 user_feat_dims, user_feat_padidxs):
        super().__init__()
        D = cfg.repr_dim
        E = cfg.emb_dim
        self.n_items    = n_items
        self.label_smooth = cfg.label_smooth
        self.aux_weight   = cfg.aux_weight
        self.bst_weight   = cfg.bst_weight
        self.MASK_ID      = n_items + 2
        vocab_size        = n_items + 3

        # BST Encoder（含自己的 item embedding）
        self.bst = BSTEncoder(cfg, n_items, item_feat_dims, item_feat_padidxs)

        # User Tower（DeepFM 用户交叉 + DIN 兴趣）
        self.user_tower = UserTower(cfg, user_feat_dims, user_feat_padidxs)

        # Item Tower（独立 embedding，不与 BST 共享）
        self.item_tower = ItemTower(cfg, n_items, item_feat_dims, item_feat_padidxs)

        # BST 辅助打分头：target_repr (D) → logit 分数
        self.bst_head = nn.Linear(D, n_items + 1)   # index 0 = PAD，不使用

        # Masked Item Prediction 辅助头
        self.aux_head = nn.Linear(D, vocab_size)

        # 可学习温度系数（cosine 打分缩放）
        self.log_temp = nn.Parameter(torch.tensor(3.0))  # exp(3) ≈ 20

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ── 打分 ──
    def _score(self, user_repr, item_repr):
        """
        余弦相似度 + 可学习温度，数值稳定版本。
        user_repr : (B, D)
        item_repr : (N, D)
        return    : (B, N)
        """
        u = F.normalize(user_repr, p=2, dim=-1, eps=1e-8)
        v = F.normalize(item_repr, p=2, dim=-1, eps=1e-8)
        t = torch.clamp(self.log_temp, -4.6, 4.6).exp()
        return (u @ v.T) * t

    # ── Label Smoothing CE ──
    def _ce(self, logits, targets):
        logits = torch.clamp(logits, -50.0, 50.0)
        if self.label_smooth == 0.0:
            return F.cross_entropy(logits, targets)
        n      = logits.size(-1)
        lp     = F.log_softmax(logits, dim=-1)
        smooth = self.label_smooth / n
        oh     = torch.zeros_like(logits).scatter_(-1, targets.unsqueeze(-1), 1.0)
        return -(oh * (1 - self.label_smooth) * lp + smooth * lp).sum(-1).mean()

    # ── 辅助：Masked Item Prediction ──
    def _aux_loss(self, bst_seq_hidden, mask_targets):
        """
        bst_seq_hidden : (B, L, D)  BST 序列位置输出
        mask_targets   : (B, L) long  被 mask 位置的原始 item id，其余为 0
        """
        mask_pos = (mask_targets != 0)
        if not mask_pos.any():
            return torch.tensor(0.0, device=bst_seq_hidden.device)
        hidden = bst_seq_hidden[mask_pos]                       # (M, D)
        tgt    = mask_targets[mask_pos]                         # (M,)
        logits = torch.clamp(self.aux_head(hidden), -50, 50)
        return F.cross_entropy(logits, tgt)

    def forward(self, seq_ids, masked_seq_ids, ift, user_feats,
                target_ids=None, mask_targets=None, mixup_alpha=0.0):
        """
        seq_ids       : (B, L)   原始序列（供 DIN pad_mask 使用）
        masked_seq_ids: (B, L)   mask 后的序列（供 BST + 辅助任务使用）
        ift           : (vocab, 4)
        user_feats    : (B, 8)
        target_ids    : (B,)     训练时传入
        mask_targets  : (B, L)   辅助任务标签
        mixup_alpha   : float    Mixup 强度

        训练时返回 loss；推理时（target_ids=None）返回 (B, N+1) logits
        """
        device = seq_ids.device
        ift    = ift.to(device)

        # ── BST 编码 ──
        # 训练：用 masked_seq 输入（防止 target 信息泄漏到序列），target 拼末尾
        # 推理：用原始 seq，不拼 target
        bst_input = masked_seq_ids if target_ids is not None else seq_ids
        cls_repr, seq_hidden, target_repr = self.bst(
            bst_input, ift,
            target_ids=target_ids if target_ids is not None else None)

        # ── User Tower ──
        user_repr = self.user_tower(cls_repr, seq_hidden, seq_ids, user_feats)

        # ── Item Tower：全量 ──
        all_ids   = torch.arange(self.n_items + 1, device=device)
        all_feats = ift[:self.n_items + 1]
        item_repr = self.item_tower(all_ids, all_feats)         # (N+1, D)

        # ── 推理 ──
        if target_ids is None:
            return self._score(user_repr, item_repr)            # (B, N+1)

        # ── 训练：主 loss ──
        # Mixup 数据增强
        if mixup_alpha > 0:
            lam     = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx     = torch.randperm(user_repr.size(0), device=device)
            user_repr_m  = lam * user_repr + (1 - lam) * user_repr[idx]
            target_ids_b = target_ids[idx]
        else:
            user_repr_m  = user_repr
            lam          = 1.0
            target_ids_b = target_ids

        logits    = self._score(user_repr_m, item_repr)         # (B, N+1)
        main_loss = self._ce(logits[:, 1:], target_ids - 1)
        if mixup_alpha > 0:
            main_loss = (lam * main_loss +
                (1 - lam) * self._ce(logits[:, 1:], target_ids_b - 1))

        # ── BST 辅助 loss：target_repr 直接预测 target item ──
        bst_loss = torch.tensor(0.0, device=device)
        if self.bst_weight > 0 and target_repr is not None:
            bst_logits = torch.clamp(self.bst_head(target_repr), -50, 50)
            bst_loss   = self._ce(bst_logits[:, 1:], target_ids - 1)

        # ── Masked Item Prediction 辅助 loss ──
        aux_loss = torch.tensor(0.0, device=device)
        if self.aux_weight > 0 and mask_targets is not None:
            aux_loss = self._aux_loss(seq_hidden, mask_targets)

        total = main_loss + self.bst_weight * bst_loss + self.aux_weight * aux_loss
        return total


# ══════════════════════════════════════════════════════════
# 5. 训练 & 评估
# ══════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, ift, device, mixup_alpha):
    model.train()
    total, n_step = 0.0, 0
    for seq, mseq, uf, tgt, mtgt in loader:
        seq  = seq.to(device);  mseq = mseq.to(device)
        uf   = uf.to(device);   tgt  = tgt.to(device)
        mtgt = mtgt.to(device)

        optimizer.zero_grad()
        loss = model(seq, mseq, ift, uf, tgt, mtgt, mixup_alpha)

        if torch.isnan(loss) or torch.isinf(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        has_bad = any(
            p.grad is not None and
            (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in model.parameters())
        if has_bad:
            optimizer.zero_grad()
            continue

        optimizer.step()
        total  += loss.item()
        n_step += 1

    return total / max(n_step, 1)


@torch.no_grad()
def evaluate(model, loader, ift, device, topk):
    model.eval()
    hits, ndcgs = [], []
    for seq, mseq, uf, tgt, _ in loader:
        seq, uf, tgt = seq.to(device), uf.to(device), tgt.to(device)
        logits       = model(seq, seq, ift, uf)     # 推理时 masked_seq=seq（不 mask）
        logits[:, 0] = -1e9
        tk = logits.topk(topk, dim=-1).indices
        for i in range(len(tgt)):
            t   = tgt[i].item()
            lst = tk[i].tolist()
            if t in lst:
                rank = lst.index(t) + 1
                hits.append(1); ndcgs.append(1.0 / math.log2(rank + 1))
            else:
                hits.append(0); ndcgs.append(0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs))


@torch.no_grad()
def predict(model, loader, ift, device, topk, id2item):
    model.eval()
    rows = []
    for seq, uf, uids in loader:
        seq, uf = seq.to(device), uf.to(device)
        logits  = model(seq, seq, ift, uf)
        logits[:, 0] = -1e9
        tk = logits.topk(topk, dim=-1).indices.cpu().tolist()
        for i, uid in enumerate(uids):
            items = [id2item[iid] for iid in tk[i] if iid in id2item]
            rows.append({"uid": uid, "predicted_items": ",".join(items)})
    return rows


# ══════════════════════════════════════════════════════════
# 6. 主函数
# ══════════════════════════════════════════════════════════
def main():
    print(f"Device: {cfg.device}\n")

    proc = DataProcessor(cfg)
    train_samples, test_samples = proc.load_and_build()

    # 用户级验证集划分（后10% uid）
    all_uids  = sorted(set(s["uid"] for s in train_samples))
    n_val_u   = max(1, int(len(all_uids) * 0.1))
    val_uids  = set(all_uids[-n_val_u:])
    val_samps = [s for s in train_samples if     s["uid"] in val_uids]
    tr_samps  = [s for s in train_samples if not s["uid"] in val_uids]
    print(f"  Val users={len(val_uids)} | Val={len(val_samps)} | Train={len(tr_samps)}\n")

    tr_ds = RecDataset(tr_samps,  cfg.max_seq_len, proc.n_items,
                       proc.MASK_ID, mask_prob=cfg.mask_prob, mode="train")
    va_ds = RecDataset(val_samps, cfg.max_seq_len, proc.n_items,
                       proc.MASK_ID, mask_prob=0.0,           mode="train")
    te_ds = RecDataset(test_samples, cfg.max_seq_len, proc.n_items,
                       proc.MASK_ID, mask_prob=0.0,           mode="test")

    tr_ld = DataLoader(tr_ds, cfg.batch_size, shuffle=True,  num_workers=0)
    va_ld = DataLoader(va_ds, cfg.batch_size, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, cfg.batch_size, shuffle=False, num_workers=0)

    model = DINDeepFMBST(
        cfg, proc.n_items,
        proc.item_feat_dims, proc.item_feat_padidxs,
        proc.user_feat_dims, proc.user_feat_padidxs,
    ).to(cfg.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    ift       = proc.item_feat_tensor
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.lr * 0.01)

    best_ndcg, best_state = 0.0, None

    for ep in range(1, cfg.epochs + 1):
        loss = train_epoch(model, tr_ld, optimizer, ift,
                           cfg.device, cfg.mixup_alpha)
        scheduler.step()

        if ep % 5 == 0 or ep == cfg.epochs:
            hr, ndcg = evaluate(model, va_ld, ift, cfg.device, cfg.topk)
            print(f"Ep {ep:3d} | Loss {loss:.4f} | "
                  f"HR@{cfg.topk}: {hr:.4f} | NDCG@{cfg.topk}: {ndcg:.4f} | "
                  f"τ={model.log_temp.clamp(-4.6,4.6).exp().item():.2f}")
            if ndcg > best_ndcg:
                best_ndcg  = ndcg
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                print(f"  ✓ Best NDCG@{cfg.topk}: {best_ndcg:.4f} — saved")
        else:
            print(f"Ep {ep:3d} | Loss {loss:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(cfg.device)
                               for k, v in best_state.items()})
    print(f"\nFinal best NDCG@{cfg.topk}: {best_ndcg:.4f}")

    rows = predict(model, te_ld, ift, cfg.device, cfg.topk, proc.id2item)
    pd.DataFrame(rows).to_csv(cfg.output_path, index=False)
    print(f"Saved → {cfg.output_path}")
    print(pd.DataFrame(rows).head(3).to_string())


if __name__ == "__main__":
    main()
