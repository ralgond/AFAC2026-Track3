"""
HyFormer：端到端 Next-Item 推荐系统
=====================================
论文：HyFormer: Revisiting the Roles of Sequence Modeling and
      Feature Interaction in CTR Prediction (ByteDance, 2025)
arxiv: https://arxiv.org/abs/2601.12681

核心思想
--------
传统架构"先压缩序列，再做特征交叉"是两阶段解耦流水线，
序列侧和特征侧的信息交互只发生在最末层，表达能力受限。

HyFormer 引入一组 Global Token（GT），作为序列与异构特征的
共享语义接口，在每一层同时执行两个互补操作：

  Query Decoding（QD）
    GT 作为 query，对行为序列的逐层 KV 表示做 cross-attention，
    使全局上下文在每层都能直接感知序列信息。
    序列侧同时用 GT 做一次 self-attention 内部编码（LONGER 风格）。

  Query Boosting（QB）
    将所有 GT 与用户/item 异构特征 token 合并，
    做一次轻量 Transformer self-attention，
    实现跨 query 和跨特征域的深度交叉。

两个操作交替叠加 n_layers 次，每层输出的 GT 携带越来越丰富
的序列-特征联合语义。

适配说明（原论文 → 本实现）
----------------------------
原论文针对 CTR（点击率预测，二分类）设计，使用候选 item 作为 query 之一。
本实现针对 Next-Item Retrieval（全量 Softmax，无候选 item）：

  1. Global Token 由用户8个特征域 embedding 初始化（每域1个GT，共8个）
  2. 序列 token = item_id_emb + 4个item特征均值 + 位置编码
  3. QB 阶段的异构特征 token = GT（不再加候选 item token）
  4. 最终 user_repr = GT 全局平均池化 → MLP
  5. Item Tower 独立，user_repr @ item_repr.T 全量打分
  6. 训练目标：全量 Softmax CE + Masked Item Prediction 辅助任务
  7. 数值稳定：logit clamp / nan_to_num / log_temp clamp / NaN batch 跳过

整体流程
--------

  输入特征
  ├─ 行为序列  seq_ids (B, L)
  │    → seq_token_emb : item_id(E) + 4域特征均值(E) → proj → D
  │    → + 位置编码(D)
  │    → SeqTokens  (B, L, D)
  │
  └─ 用户特征  user_feats (B, 8)
       → 每域 embedding(E) → Linear → D
       → GlobalTokens  (B, n_gt, D)   n_gt = 8

  HyFormer Layer × n_layers
  ┌─────────────────────────────────────────────────┐
  │  Query Decoding                                  │
  │   SeqTokens  → SASRec-style causal self-attn    │
  │   GT cross-attn(Q=GT, KV=SeqTokens) → GT'       │
  │                                                  │
  │  Query Boosting                                  │
  │   concat[GT'; SeqTokens_mean_token]             │
  │   → Transformer self-attn(full)                  │
  │   → 取 GT 对应位置输出 → GT''                   │
  └─────────────────────────────────────────────────┘

  GT_final (B, n_gt, D)
  → mean pooling → (B, D)
  → MLP → user_repr (B, D)

  Item Tower
  item_id + 4域特征 → FM二阶 + MLP → item_repr (N+1, D)

  打分
  cosine(user_repr, item_repr) × exp(log_temp)
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
    output_path  = "submission_hyformer.csv"

    max_seq_len  = 50       # 行为序列最大长度
    emb_dim      = 32       # 每个特征域 embedding 维度
    repr_dim     = 128      # 主干隐层维度 D（GT / SeqToken / user_repr）

    # HyFormer 核心
    n_layers     = 2        # HyFormer 层数（QD+QB 交替次数）
    n_heads      = 4        # attention 头数
    n_gt         = 8        # Global Token 数量（= 用户特征域数）
    qb_n_heads   = 4        # Query Boosting self-attn 头数

    # Item Tower
    item_mlp_dims = [256, 128]
    dropout       = 0.2

    # 训练
    epochs        = 50
    batch_size    = 256
    lr            = 1e-3
    weight_decay  = 1e-5
    label_smooth  = 0.0
    aux_weight    = 0.05    # Masked Item Prediction 辅助损失权重
    mask_prob     = 0.15    # 序列 mask 概率
    mixup_alpha   = 0.1

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
        self.item2id  = {}; self.id2item  = {}
        self.iid2feat = {}; self.uid2feat = {}
        self.item_feat_cols    = ["i_cat_01","i_cat_02","i_cat_03","i_bucket_01"]
        self.user_feat_cols    = [f"u_cat_0{i}" for i in range(1, 9)]
        self.item_feat_dims    = []; self.item_feat_padidxs = []
        self.user_feat_dims    = []; self.user_feat_padidxs = []
        self.n_items = 0

    def _parse_seq(self, s):
        if pd.isna(s) or not str(s).strip(): return []
        return [x.strip() for x in str(s).split(",") if x.strip()]

    def _parse_counts(self, s):
        if pd.isna(s) or not str(s).strip(): return {}
        d = {}
        for p in str(s).split(","):
            if ":" in p:
                k, v = p.strip().rsplit(":", 1); d[k.strip()] = int(v)
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

        # item 词表  0=PAD, 1..N=item, N+1=MASK
        all_items = set(item_df["iid"].str.strip())
        for col in ["item_seq_raw","item_seq_dedup"]:
            for df in [train_df, test_df]:
                if col in df.columns:
                    for s in df[col].dropna():
                        all_items.update(self._parse_seq(s))
        all_items.update(train_df["target_iid"].str.strip().dropna())
        all_items = sorted(all_items)
        self.item2id = {iid: idx+1 for idx, iid in enumerate(all_items)}
        self.id2item = {v: k for k, v in self.item2id.items()}
        self.n_items = len(self.item2id)
        self.MASK_ID = self.n_items + 1
        print(f"  Total items : {self.n_items}  MASK={self.MASK_ID}")

        # item 特征  pad_idx = max+1
        item_df["iid"] = item_df["iid"].str.strip()
        for col in self.item_feat_cols:
            item_df[col] = item_df[col].fillna(-1).astype(int)
        for col in self.item_feat_cols:
            rmax = int(item_df[col].max())
            self.item_feat_padidxs.append(rmax+1); self.item_feat_dims.append(rmax+2)
        for _, row in item_df.iterrows():
            iid = self.item2id.get(row["iid"])
            if iid: self.iid2feat[iid] = [int(row[c]) for c in self.item_feat_cols]
        pad_ifeat = list(self.item_feat_padidxs)
        for iid in self.id2item:
            if iid not in self.iid2feat: self.iid2feat[iid] = pad_ifeat

        # user 特征
        user_df["uid"] = user_df["uid"].str.strip()
        for col in self.user_feat_cols:
            user_df[col] = user_df[col].fillna(-1).astype(int)
        for col in self.user_feat_cols:
            rmax = int(user_df[col].max())
            self.user_feat_padidxs.append(rmax+1); self.user_feat_dims.append(rmax+2)
        uid_feat_map = {row["uid"]: [int(row[c]) for c in self.user_feat_cols]
                        for _, row in user_df.iterrows()}
        pad_ufeat = list(self.user_feat_padidxs)

        train_df["uid"]        = train_df["uid"].str.strip()
        train_df["target_iid"] = train_df["target_iid"].str.strip()
        test_df["uid"]         = test_df["uid"].str.strip()

        train_samples, test_samples = [], []
        for _, row in train_df.iterrows():
            uid = row["uid"]; target = self.item2id.get(row["target_iid"])
            if not target: continue
            seq  = self._parse_seq(row["item_seq_dedup"])
            cnts = self._parse_counts(row.get("item_seq_counts",""))
            train_samples.append({"uid": uid, "seq": self._weight_seq(seq, cnts),
                                  "target": target, "user_feat": uid_feat_map.get(uid, pad_ufeat)})
        for _, row in test_df.iterrows():
            uid = row["uid"]
            seq  = self._parse_seq(row["item_seq_dedup"])
            cnts = self._parse_counts(row.get("item_seq_counts",""))
            test_samples.append({"uid": uid, "seq": self._weight_seq(seq, cnts),
                                 "user_feat": uid_feat_map.get(uid, pad_ufeat)})
        print(f"  Train: {len(train_samples)} | Test: {len(test_samples)}")

        # 全量 item 特征张量 (N+2, 4)
        ift = torch.zeros(self.n_items+2, 4, dtype=torch.long)
        for iid, feats in self.iid2feat.items():
            ift[iid] = torch.tensor(feats)
        ift[self.MASK_ID] = torch.tensor(pad_ifeat)
        self.item_feat_tensor = ift
        return train_samples, test_samples


# ══════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════
def pad_seq(seq, max_len):
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq

class RecDataset(Dataset):
    def __init__(self, samples, max_len, n_items, mask_id, mask_prob=0.0, mode="train"):
        self.samples  = samples; self.max_len = max_len
        self.n_items  = n_items; self.mask_id = mask_id
        self.mask_prob = mask_prob; self.mode = mode

    def __len__(self): return len(self.samples)

    def _mask(self, seq):
        masked = seq.copy(); tgt = [0]*len(seq)
        for i, s in enumerate(seq):
            if s != 0 and random.random() < self.mask_prob:
                tgt[i] = s; masked[i] = self.mask_id
        return masked, tgt

    def __getitem__(self, idx):
        s   = self.samples[idx]
        seq = pad_seq(s["seq"], self.max_len)
        uf  = torch.tensor(s["user_feat"], dtype=torch.long)
        if self.mode == "train":
            masked, mtgt = self._mask(seq)
            return (torch.tensor(seq,    dtype=torch.long),
                    torch.tensor(masked, dtype=torch.long),
                    uf,
                    torch.tensor(s["target"], dtype=torch.long),
                    torch.tensor(mtgt,        dtype=torch.long))
        return torch.tensor(seq, dtype=torch.long), uf, s["uid"]


# ══════════════════════════════════════════════════════════
# 3. 模型组件
# ══════════════════════════════════════════════════════════

def build_mlp(in_dim, hidden_dims, out_dim, dropout):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]; d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)

def fm_second_order(emb_list):
    stacked = torch.stack(emb_list, dim=-2)
    return 0.5 * (stacked.sum(-2)**2 - (stacked**2).sum(-2))


# ── Query Decoding（QD）──────────────────────────────────
class QueryDecoding(nn.Module):
    """
    两步：
    1. 序列侧因果自注意力（捕捉序列内部时序依赖，同 SASRec）
    2. GT cross-attention（Q=GT, KV=seq_hidden）→ GT 感知序列

    论文 Eq.(3)(4)(5)
    """
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        # 序列内部因果自注意力
        self.seq_self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.seq_n1        = nn.LayerNorm(d)
        self.seq_ff        = nn.Sequential(nn.Linear(d,d*4), nn.GELU(),
                                           nn.Dropout(dropout), nn.Linear(d*4,d))
        self.seq_n2        = nn.LayerNorm(d)

        # GT cross-attention（Q=GT, KV=seq）
        self.gt_cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.gt_n1         = nn.LayerNorm(d)
        self.gt_ff         = nn.Sequential(nn.Linear(d,d*4), nn.GELU(),
                                           nn.Dropout(dropout), nn.Linear(d*4,d))
        self.gt_n2         = nn.LayerNorm(d)
        self.drop          = nn.Dropout(dropout)

    def _causal(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

    def forward(self, gt, seq_hidden, seq_pad_mask):
        """
        gt          : (B, n_gt, D)
        seq_hidden  : (B, L, D)
        seq_pad_mask: (B, L) True=pad
        returns     : gt' (B, n_gt, D),  seq_hidden' (B, L, D)
        """
        B, L, D = seq_hidden.shape

        # ── 1. 序列因果自注意力 ──
        causal = self._causal(L, seq_hidden.device)
        r = seq_hidden; x = self.seq_n1(seq_hidden)
        x, _ = self.seq_self_attn(x, x, x, attn_mask=causal,
                                  key_padding_mask=seq_pad_mask, need_weights=False)
        x = torch.nan_to_num(self.drop(x), nan=0.0) + r
        r = x; x = self.seq_n2(x)
        seq_hidden = torch.nan_to_num(self.drop(self.seq_ff(x)), nan=0.0) + r  # (B,L,D)

        # ── 2. GT cross-attention（Q=GT, KV=seq_hidden）──
        r = gt; q = self.gt_n1(gt)
        x, _ = self.gt_cross_attn(q, seq_hidden, seq_hidden,
                                   key_padding_mask=seq_pad_mask, need_weights=False)
        x = torch.nan_to_num(self.drop(x), nan=0.0) + r
        r = x; x = self.gt_n2(x)
        gt = torch.nan_to_num(self.drop(self.gt_ff(x)), nan=0.0) + r            # (B,n_gt,D)

        return gt, seq_hidden


# ── Query Boosting（QB）─────────────────────────────────
class QueryBoosting(nn.Module):
    """
    将 GT 与一个序列摘要 token（序列均值）拼接，做全局双向 self-attention，
    实现跨 GT 和序列摘要的异构特征深度交叉。
    取前 n_gt 个 token 输出作为更新后的 GT。

    论文 Eq.(6)(7)
    """
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.ff   = nn.Sequential(nn.Linear(d,d*4), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d*4,d))
        self.drop = nn.Dropout(dropout)

    def forward(self, gt, seq_hidden, seq_pad_mask):
        """
        gt          : (B, n_gt, D)
        seq_hidden  : (B, L, D)
        seq_pad_mask: (B, L) True=pad
        returns     : gt' (B, n_gt, D)
        """
        # 序列摘要：masked mean pooling
        valid    = (~seq_pad_mask).float().unsqueeze(-1)            # (B,L,1)
        seq_mean = (seq_hidden * valid).sum(1) / valid.sum(1).clamp(min=1)  # (B,D)
        seq_tok  = seq_mean.unsqueeze(1)                            # (B,1,D)

        # 拼接 [GT | seq_mean_token]  长度 = n_gt + 1
        tokens = torch.cat([gt, seq_tok], dim=1)                    # (B, n_gt+1, D)

        # 双向全局 self-attention（无 mask）
        r = tokens; x = self.n1(tokens)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = torch.nan_to_num(self.drop(x), nan=0.0) + r
        r = x; x = self.n2(x)
        x = torch.nan_to_num(self.drop(self.ff(x)), nan=0.0) + r   # (B, n_gt+1, D)

        # 只取前 n_gt 个（GT 对应位置），丢弃 seq_mean token
        return x[:, :gt.shape[1], :]                                # (B, n_gt, D)


# ── HyFormer Layer：QD + QB 交替 ────────────────────────
class HyFormerLayer(nn.Module):
    """单个 HyFormer 层 = QueryDecoding + QueryBoosting"""
    def __init__(self, d, n_heads, qb_n_heads, dropout):
        super().__init__()
        self.qd = QueryDecoding(d, n_heads, dropout)
        self.qb = QueryBoosting(d, qb_n_heads, dropout)

    def forward(self, gt, seq_hidden, seq_pad_mask):
        gt, seq_hidden = self.qd(gt, seq_hidden, seq_pad_mask)
        gt             = self.qb(gt, seq_hidden, seq_pad_mask)
        return gt, seq_hidden


# ── 序列 Token 编码 ──────────────────────────────────────
class SeqTokenizer(nn.Module):
    """
    seq_ids (B, L) → SeqTokens (B, L, D)
    item_id_emb(E) + 4域特征均值(E) → concat → proj(D) + 位置编码(D)
    """
    def __init__(self, cfg, n_items, item_feat_dims, item_feat_padidxs):
        super().__init__()
        E          = cfg.emb_dim
        D          = cfg.repr_dim
        vocab_size = n_items + 2       # 0=PAD, N+1=MASK

        self.item_id_emb = nn.Embedding(vocab_size, E, padding_idx=0)
        self.feat_embs   = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)])
        self.proj    = nn.Linear(E * 2, D)      # id(E) + feat_mean(E) → D
        self.pos_emb = nn.Embedding(cfg.max_seq_len + 1, D)
        self.norm    = nn.LayerNorm(D)
        self.drop    = nn.Dropout(cfg.dropout)

    def forward(self, seq_ids, ift):
        """
        seq_ids : (B, L)
        ift     : (vocab, 4)
        return  : (B, L, D)
        """
        B, L   = seq_ids.shape
        device = seq_ids.device
        ift    = ift.to(device)

        id_e   = self.item_id_emb(seq_ids)                          # (B, L, E)
        feats  = ift[seq_ids]                                        # (B, L, 4)
        f_list = [self.feat_embs[j](feats[:,:,j]) for j in range(4)]
        f_mean = torch.stack(f_list, dim=-2).mean(-2)               # (B, L, E)

        x = F.gelu(self.proj(torch.cat([id_e, f_mean], dim=-1)))    # (B, L, D)
        pos = torch.arange(1, L+1, device=device).unsqueeze(0)
        x = self.norm(self.drop(x + self.pos_emb(pos)))             # (B, L, D)
        return x


# ── Global Token 初始化 ──────────────────────────────────
class GlobalTokenInit(nn.Module):
    """
    用户8个特征域各自 embedding → Linear → D，
    每个域对应一个 Global Token（共 n_gt=8 个）。
    """
    def __init__(self, cfg, user_feat_dims, user_feat_padidxs):
        super().__init__()
        E    = cfg.emb_dim
        D    = cfg.repr_dim
        n_gt = cfg.n_gt   # 8
        assert len(user_feat_dims) == n_gt

        self.u_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(user_feat_dims, user_feat_padidxs)])
        self.projs  = nn.ModuleList([nn.Linear(E, D) for _ in range(n_gt)])
        self.norm   = nn.LayerNorm(D)

    def forward(self, user_feats):
        """user_feats: (B, 8) → (B, 8, D)"""
        gts = []
        for j, (emb, proj) in enumerate(zip(self.u_embs, self.projs)):
            gts.append(proj(emb(user_feats[:, j])))    # (B, D)
        gt = torch.stack(gts, dim=1)                    # (B, 8, D)
        return self.norm(gt)


# ── Item Tower（独立，不与序列侧共享 embedding）──────────
class ItemTower(nn.Module):
    def __init__(self, cfg, n_items, item_feat_dims, item_feat_padidxs):
        super().__init__()
        E   = cfg.emb_dim; D = cfg.repr_dim; n_i = 4
        self.item_id_emb = nn.Embedding(n_items+1, E, padding_idx=0)
        self.feat_embs   = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)])
        # FM二阶(E) + item_id(E) + feat_concat(4E) → MLP → D
        self.mlp = build_mlp(E + E*n_i + E, cfg.item_mlp_dims, D, cfg.dropout)

    def forward(self, item_ids, item_feats):
        id_e    = self.item_id_emb(item_ids)
        fi_list = [e(item_feats[..., j]) for j, e in enumerate(self.feat_embs)]
        fm2     = fm_second_order(fi_list)
        fi_cat  = torch.cat(fi_list, dim=-1)
        return self.mlp(torch.cat([id_e, fi_cat, fm2], dim=-1))


# ══════════════════════════════════════════════════════════
# 4. HyFormer 主模型
# ══════════════════════════════════════════════════════════
class HyFormerModel(nn.Module):
    """
    完整 HyFormer 端到端推荐模型。

    训练：全量 Softmax CE (主) + Masked Item Prediction (辅)
    推理：cosine(user_repr, all_item_repr) × τ，一次矩阵乘法
    """
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 user_feat_dims, user_feat_padidxs):
        super().__init__()
        D = cfg.repr_dim
        self.n_items     = n_items
        self.label_smooth = cfg.label_smooth
        self.aux_weight   = cfg.aux_weight
        self.MASK_ID      = n_items + 1

        # ── 编码模块 ──
        self.seq_tokenizer = SeqTokenizer(cfg, n_items, item_feat_dims, item_feat_padidxs)
        self.gt_init       = GlobalTokenInit(cfg, user_feat_dims, user_feat_padidxs)

        # ── HyFormer 层堆叠 ──
        self.layers = nn.ModuleList([
            HyFormerLayer(D, cfg.n_heads, cfg.qb_n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)])

        # ── 最终聚合 → user_repr ──
        # GT mean pooling(D) → MLP → D
        self.user_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D, D))

        # ── Item Tower ──
        self.item_tower = ItemTower(cfg, n_items, item_feat_dims, item_feat_padidxs)

        # ── 辅助任务：Masked Item Prediction ──
        vocab_size    = n_items + 2
        self.aux_head = nn.Linear(D, vocab_size)

        # ── 可学习温度 ──
        self.log_temp = nn.Parameter(torch.tensor(3.0))  # exp(3)≈20

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _score(self, user_repr, item_repr):
        u = F.normalize(user_repr, p=2, dim=-1, eps=1e-8)
        v = F.normalize(item_repr, p=2, dim=-1, eps=1e-8)
        t = torch.clamp(self.log_temp, -4.6, 4.6).exp()
        return (u @ v.T) * t

    def _ce(self, logits, targets):
        logits = torch.clamp(logits, -50.0, 50.0)
        if self.label_smooth == 0.0:
            return F.cross_entropy(logits, targets)
        n  = logits.size(-1)
        lp = F.log_softmax(logits, dim=-1)
        s  = self.label_smooth / n
        oh = torch.zeros_like(logits).scatter_(-1, targets.unsqueeze(-1), 1.0)
        return -(oh * (1-self.label_smooth) * lp + s * lp).sum(-1).mean()

    def _aux_loss(self, seq_hidden_final, mask_targets):
        mask_pos = (mask_targets != 0)
        if not mask_pos.any():
            return torch.tensor(0.0, device=seq_hidden_final.device)
        hidden = seq_hidden_final[mask_pos]
        tgt    = mask_targets[mask_pos]
        logits = torch.clamp(self.aux_head(hidden), -50, 50)
        return F.cross_entropy(logits, tgt)

    def encode_user(self, seq_ids, masked_seq_ids, ift, user_feats):
        """
        seq_ids        : (B, L)  原始序列（用于 pad_mask）
        masked_seq_ids : (B, L)  mask 后序列（作为 SeqTokenizer 输入）
        ift            : (vocab, 4)
        user_feats     : (B, 8)
        return         : user_repr (B, D), seq_hidden_final (B, L, D)
        """
        device = seq_ids.device
        ift    = ift.to(device)

        # 序列 token
        seq_hidden   = self.seq_tokenizer(masked_seq_ids, ift)  # (B, L, D)
        seq_pad_mask = (seq_ids == 0)                           # (B, L) True=pad

        # Global Token 初始化（由用户特征域 embedding 生成）
        gt = self.gt_init(user_feats)                           # (B, n_gt, D)

        # HyFormer 层叠加
        for layer in self.layers:
            gt, seq_hidden = layer(gt, seq_hidden, seq_pad_mask)

        # GT mean pooling → user_repr
        gt_mean    = gt.mean(dim=1)                             # (B, D)
        user_repr  = self.user_head(gt_mean)
        user_repr  = torch.nan_to_num(user_repr, nan=0.0)
        return user_repr, seq_hidden

    def forward(self, seq_ids, masked_seq_ids, ift, user_feats,
                target_ids=None, mask_targets=None, mixup_alpha=0.0):
        device = seq_ids.device

        user_repr, seq_hidden = self.encode_user(
            seq_ids, masked_seq_ids, ift, user_feats)

        # Item Tower：全量
        all_ids   = torch.arange(self.n_items+1, device=device)
        all_feats = ift.to(device)[:self.n_items+1]
        item_repr = self.item_tower(all_ids, all_feats)         # (N+1, D)

        # 推理
        if target_ids is None:
            return self._score(user_repr, item_repr)

        # Mixup
        if mixup_alpha > 0:
            lam  = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx  = torch.randperm(user_repr.size(0), device=device)
            user_repr_m  = lam * user_repr + (1-lam) * user_repr[idx]
            target_ids_b = target_ids[idx]
        else:
            user_repr_m  = user_repr; lam = 1.0; target_ids_b = target_ids

        # 主任务 loss
        logits    = self._score(user_repr_m, item_repr)         # (B, N+1)
        main_loss = self._ce(logits[:, 1:], target_ids - 1)
        if mixup_alpha > 0:
            main_loss = (lam * main_loss +
                (1-lam) * self._ce(logits[:, 1:], target_ids_b - 1))

        # 辅助任务 loss
        aux_loss = torch.tensor(0.0, device=device)
        if self.aux_weight > 0 and mask_targets is not None:
            aux_loss = self._aux_loss(seq_hidden, mask_targets)

        return main_loss + self.aux_weight * aux_loss


# ══════════════════════════════════════════════════════════
# 5. 训练 & 评估
# ══════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, ift, device, mixup_alpha):
    model.train(); total, n = 0.0, 0
    for seq, mseq, uf, tgt, mtgt in loader:
        seq  = seq.to(device);  mseq = mseq.to(device)
        uf   = uf.to(device);   tgt  = tgt.to(device); mtgt = mtgt.to(device)
        optimizer.zero_grad()
        loss = model(seq, mseq, ift, uf, tgt, mtgt, mixup_alpha)
        if torch.isnan(loss) or torch.isinf(loss): continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        bad = any(p.grad is not None and
                  (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                  for p in model.parameters())
        if bad: optimizer.zero_grad(); continue
        optimizer.step(); total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, ift, device, topk):
    model.eval(); hits, ndcgs = [], []
    for seq, mseq, uf, tgt, _ in loader:
        seq, uf, tgt = seq.to(device), uf.to(device), tgt.to(device)
        logits       = model(seq, seq, ift, uf)
        logits[:, 0] = -1e9
        tk = logits.topk(topk, dim=-1).indices
        for i in range(len(tgt)):
            t = tgt[i].item(); lst = tk[i].tolist()
            if t in lst:
                rank = lst.index(t)+1
                hits.append(1); ndcgs.append(1.0/math.log2(rank+1))
            else:
                hits.append(0); ndcgs.append(0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs))


@torch.no_grad()
def predict(model, loader, ift, device, topk, id2item):
    model.eval(); rows = []
    for seq, uf, uids in loader:
        seq, uf = seq.to(device), uf.to(device)
        logits  = model(seq, seq, ift, uf)
        logits[:, 0] = -1e9
        tk = logits.topk(topk, dim=-1).indices.cpu().tolist()
        for i, uid in enumerate(uids):
            items = [id2item[iid] for iid in tk[i] if iid in id2item]
            rows.append({"uid": uid, "prediction": ",".join(items)})
    return rows


# ══════════════════════════════════════════════════════════
# 6. 主函数
# ══════════════════════════════════════════════════════════
def main():
    print(f"Device: {cfg.device}\n")
    proc = DataProcessor(cfg)
    train_samples, test_samples = proc.load_and_build()

    all_uids  = sorted(set(s["uid"] for s in train_samples))
    n_val_u   = max(1, int(len(all_uids) * 0.1))
    val_uids  = set(all_uids[-n_val_u:])
    val_samps = [s for s in train_samples if     s["uid"] in val_uids]
    tr_samps  = [s for s in train_samples if not s["uid"] in val_uids]
    print(f"  Val users={len(val_uids)} | Val={len(val_samps)} | Train={len(tr_samps)}\n")

    tr_ds = RecDataset(tr_samps,     cfg.max_seq_len, proc.n_items,
                       proc.MASK_ID, mask_prob=cfg.mask_prob, mode="train")
    va_ds = RecDataset(val_samps,    cfg.max_seq_len, proc.n_items,
                       proc.MASK_ID, mask_prob=0.0,           mode="train")
    te_ds = RecDataset(test_samples, cfg.max_seq_len, proc.n_items,
                       proc.MASK_ID, mask_prob=0.0,           mode="test")
    tr_ld = DataLoader(tr_ds, cfg.batch_size, shuffle=True,  num_workers=0)
    va_ld = DataLoader(va_ds, cfg.batch_size, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, cfg.batch_size, shuffle=False, num_workers=0)

    model = HyFormerModel(
        cfg, proc.n_items,
        proc.item_feat_dims, proc.item_feat_padidxs,
        proc.user_feat_dims, proc.user_feat_padidxs,
    ).to(cfg.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    ift       = proc.item_feat_tensor
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.lr*0.01)

    best_ndcg, best_state = 0.0, None
    for ep in range(1, cfg.epochs+1):
        loss = train_epoch(model, tr_ld, optimizer, ift, cfg.device, cfg.mixup_alpha)
        scheduler.step()
        if ep % 5 == 0 or ep == cfg.epochs:
            hr, ndcg = evaluate(model, va_ld, ift, cfg.device, cfg.topk)
            tau = model.log_temp.clamp(-4.6,4.6).exp().item()
            print(f"Ep {ep:3d} | Loss {loss:.4f} | "
                  f"HR@{cfg.topk}: {hr:.4f} | NDCG@{cfg.topk}: {ndcg:.4f} | τ={tau:.2f}")
            if ndcg > best_ndcg:
                best_ndcg  = ndcg
                best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
                print(f"  ✓ Best NDCG@{cfg.topk}: {best_ndcg:.4f} — saved")
        else:
            print(f"Ep {ep:3d} | Loss {loss:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(cfg.device) for k,v in best_state.items()})
    print(f"\nFinal best NDCG@{cfg.topk}: {best_ndcg:.4f}")

    rows = predict(model, te_ld, ift, cfg.device, cfg.topk, proc.id2item)
    pd.DataFrame(rows).to_csv(cfg.output_path, index=False)
    print(f"Saved → {cfg.output_path}")
    print(pd.DataFrame(rows).head(3).to_string())

if __name__ == "__main__":
    main()
