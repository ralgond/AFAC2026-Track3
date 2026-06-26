"""
Transformer + DCNv2 + Contrastive Learning  推荐模型
=====================================================

架构设计
--------

三个组件各司其职，相互增强：

┌─────────────────────────────────────────────────────────────┐
│  Transformer 序列编码器（SASRec 风格）                        │
│  行为序列 → 多层因果自注意力 → seq_repr (D)                   │
│  捕捉用户行为的时序依赖与位置信息                              │
└─────────────────────────────────────────────────────────────┘
          ↓ seq_repr
┌─────────────────────────────────────────────────────────────┐
│  DCNv2 特征交叉层                                             │
│  输入：concat[用户8域emb; item4域emb; seq_repr]               │
│  Cross Network v2：逐层显式高阶特征交叉（矩阵权重版）          │
│  Deep Network   ：并行 MLP 捕捉隐式非线性交叉                 │
│  输出：concat[cross_out; deep_out] → Linear → user_repr (D)  │
└─────────────────────────────────────────────────────────────┘
          ↓ user_repr
┌─────────────────────────────────────────────────────────────┐
│  对比学习（CL）辅助任务                                       │
│  View1：原始序列 → Transformer → DCNv2 → z1                  │
│  View2：增强序列 → Transformer → DCNv2 → z2                  │
│                                                               │
│  增强策略（同一 batch 内随机选一种）：                         │
│    ① Item Crop    ：随机截取序列后半段                        │
│    ② Item Mask    ：随机将 20% 位置替换为 MASK token          │
│    ③ Item Reorder ：随机打乱序列中 20% 的片段顺序             │
│                                                               │
│  损失：InfoNCE（同一用户的两个视图为正对，batch 内其余为负例）  │
└─────────────────────────────────────────────────────────────┘

损失函数
--------
  L = L_main + λ_cl × L_cl + λ_aux × L_aux

  L_main  : 全量 Softmax CE（主任务，next item prediction）
  L_cl    : InfoNCE 对比损失（序列增强视图对齐）
  L_aux   : Masked Item Prediction（序列随机 mask，预测原 item）

DCNv2 Cross Network 原理
--------------------------
  原论文：DCN V2: Improved Deep & Cross Network (Wang et al., 2021)
  每层：x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
  其中 W_l ∈ R^{d×d}（矩阵权重，比 DCN v1 的向量权重表达力更强）
  叠加 n_cross 层可捕捉到 n_cross+1 阶显式特征交叉

打分
----
  cosine(user_repr, item_repr) × exp(log_temp)
  item_repr 由独立 Item Tower（item_id + 4域特征 FM+MLP）生成
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
    output_path  = "submission_tdcl.csv"

    # 序列
    max_seq_len  = 50
    emb_dim      = 32       # 每个特征域 embedding 维度 E
    repr_dim     = 128      # 主干隐层维度 D

    # Transformer 序列编码器
    n_heads      = 4
    n_layers     = 2

    # DCNv2
    n_cross      = 3        # Cross Network 层数（显式阶数 = n_cross+1）
    deep_dims    = [256, 128]   # Deep Network 各层宽度

    # Item Tower
    item_mlp_dims = [256, 128]

    dropout      = 0.2

    # 对比学习
    cl_temp      = 0.2      # InfoNCE 温度（较小值让负例判别更锐利）
    cl_weight    = 0.1      # CL 损失权重
    aug_prob     = [0.4, 0.3, 0.3]  # [crop, mask, reorder] 三种增强的概率
    crop_ratio   = 0.6      # Crop：保留后 crop_ratio 比例
    mask_ratio   = 0.2      # Mask：mask 比例
    reorder_ratio = 0.2     # Reorder：乱序片段比例

    # 辅助任务
    aux_weight   = 0.05
    mask_prob    = 0.15     # Masked Item Prediction mask 比例

    # 训练
    epochs       = 50
    batch_size   = 256
    lr           = 1e-3
    weight_decay = 1e-5
    label_smooth = 0.0
    mixup_alpha  = 0.1

    seed         = 42
    topk         = 10
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
            self.item_feat_padidxs.append(rmax+1)
            self.item_feat_dims.append(rmax+2)
        for _, row in item_df.iterrows():
            iid = self.item2id.get(row["iid"])
            if iid: self.iid2feat[iid] = [int(row[c]) for c in self.item_feat_cols]
        pad_ifeat = list(self.item_feat_padidxs)
        for iid in self.id2item:
            if iid not in self.iid2feat: self.iid2feat[iid] = pad_ifeat

        # user 特征  pad_idx = max+1
        user_df["uid"] = user_df["uid"].str.strip()
        for col in self.user_feat_cols:
            user_df[col] = user_df[col].fillna(-1).astype(int)
        for col in self.user_feat_cols:
            rmax = int(user_df[col].max())
            self.user_feat_padidxs.append(rmax+1)
            self.user_feat_dims.append(rmax+2)
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
            train_samples.append({
                "uid": uid, "seq": self._weight_seq(seq, cnts),
                "target": target, "user_feat": uid_feat_map.get(uid, pad_ufeat)})
        for _, row in test_df.iterrows():
            uid = row["uid"]
            seq  = self._parse_seq(row["item_seq_dedup"])
            cnts = self._parse_counts(row.get("item_seq_counts",""))
            test_samples.append({
                "uid": uid, "seq": self._weight_seq(seq, cnts),
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
# 2. 序列增强（对比学习用）
# ══════════════════════════════════════════════════════════
def augment_seq(seq_ids: torch.Tensor, mask_id: int,
                aug_type: str, cfg) -> torch.Tensor:
    """
    对一个 batch 的序列做数据增强，返回增强后的序列。
    seq_ids : (B, L) long，左 padding（0=PAD）
    aug_type: "crop" | "mask" | "reorder"
    保持输出 shape = (B, L)，左 padding 不变。
    """
    B, L   = seq_ids.shape
    device = seq_ids.device
    result = seq_ids.clone()

    for b in range(B):
        row    = seq_ids[b].tolist()
        # 找到有效（非PAD）的 item 位置
        valid_pos = [i for i, x in enumerate(row) if x != 0]
        n_valid   = len(valid_pos)
        if n_valid == 0:
            continue

        if aug_type == "crop":
            # 保留后 crop_ratio 比例的有效 item
            keep = max(1, int(n_valid * cfg.crop_ratio))
            keep_pos = valid_pos[-keep:]          # 取最近的 keep 个
            new_row  = [0] * L
            # 右对齐放入（左 padding 风格）
            for j, p in enumerate(keep_pos):
                new_row[L - keep + j] = row[p]
            result[b] = torch.tensor(new_row, dtype=torch.long, device=device)

        elif aug_type == "mask":
            # 随机将 mask_ratio 比例的有效位置替换为 MASK token
            n_mask = max(1, int(n_valid * cfg.mask_ratio))
            to_mask = random.sample(valid_pos, n_mask)
            for p in to_mask:
                result[b, p] = mask_id

        elif aug_type == "reorder":
            # 随机选一段连续片段打乱顺序
            n_reorder = max(2, int(n_valid * cfg.reorder_ratio))
            if n_valid < 2:
                continue
            start_idx = random.randint(0, n_valid - n_reorder)
            seg_pos   = valid_pos[start_idx: start_idx + n_reorder]
            seg_vals  = [row[p] for p in seg_pos]
            random.shuffle(seg_vals)
            for p, v in zip(seg_pos, seg_vals):
                result[b, p] = v

    return result


def random_augment(seq_ids: torch.Tensor, mask_id: int, cfg) -> torch.Tensor:
    """按 aug_prob 权重随机选一种增强策略"""
    aug_types = ["crop", "mask", "reorder"]
    probs     = cfg.aug_prob
    aug_type  = random.choices(aug_types, weights=probs, k=1)[0]
    return augment_seq(seq_ids, mask_id, aug_type, cfg)


# ══════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════
def pad_seq(seq, max_len):
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq

class RecDataset(Dataset):
    """
    训练：(seq, masked_seq, user_feat, target, mask_targets)
    测试：(seq, user_feat, uid)
    对比学习的序列增强在训练循环中动态生成（保证每 epoch 不同）
    """
    def __init__(self, samples, max_len, n_items, mask_id,
                 mask_prob=0.0, mode="train"):
        self.samples   = samples; self.max_len = max_len
        self.n_items   = n_items; self.mask_id = mask_id
        self.mask_prob = mask_prob; self.mode   = mode

    def __len__(self): return len(self.samples)

    def _mask(self, seq):
        masked = seq.copy(); tgt = [0] * len(seq)
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
# 4. 模型组件
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


# ── 4a. Transformer 序列编码器 ─────────────────────────────
class TransformerBlock(nn.Module):
    """SASRec 风格的因果 Transformer 块"""
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d, d*4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d*4, d))
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, causal_mask, pad_mask):
        r = x; x = self.n1(x)
        x, _ = self.attn(x, x, x, attn_mask=causal_mask,
                         key_padding_mask=pad_mask, need_weights=False)
        x = torch.nan_to_num(self.drop(x), nan=0.0) + r
        r = x; x = self.n2(x)
        return torch.nan_to_num(self.drop(self.ff(x)), nan=0.0) + r


class SeqEncoder(nn.Module):
    """
    行为序列 → seq_repr (B, D)
    token = item_id_emb(E) + item_feat_mean(E) → proj(D) + pos_emb(D)
    → n_layers 因果 Transformer
    → 取最后有效位置（左padding 即 index L-1）输出
    """
    def __init__(self, cfg, n_items, item_feat_dims, item_feat_padidxs):
        super().__init__()
        E          = cfg.emb_dim
        D          = cfg.repr_dim
        vocab_size = n_items + 2    # 0=PAD, N+1=MASK

        self.item_id_emb = nn.Embedding(vocab_size, E, padding_idx=0)
        self.feat_embs   = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)])
        self.token_proj  = nn.Linear(E * 2, D)
        self.pos_emb     = nn.Embedding(cfg.max_seq_len + 1, D)
        self.layers      = nn.ModuleList([
            TransformerBlock(D, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)])
        self.norm  = nn.LayerNorm(D)
        self.drop  = nn.Dropout(cfg.dropout)

    def _causal(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

    def forward(self, seq_ids, ift):
        """
        seq_ids : (B, L)
        ift     : (vocab, 4)
        return  : seq_repr (B, D),  seq_hidden (B, L, D)
        """
        B, L   = seq_ids.shape
        device = seq_ids.device
        ift    = ift.to(device)

        id_e    = self.item_id_emb(seq_ids)                         # (B, L, E)
        feats   = ift[seq_ids]                                       # (B, L, 4)
        f_mean  = torch.stack(
            [self.feat_embs[j](feats[:,:,j]) for j in range(4)],
            dim=-2).mean(-2)                                         # (B, L, E)

        x = F.gelu(self.token_proj(torch.cat([id_e, f_mean], -1)))  # (B, L, D)
        pos = torch.arange(1, L+1, device=device).unsqueeze(0)
        x = self.drop(x + self.pos_emb(pos))                        # (B, L, D)

        pad_mask = (seq_ids == 0)
        causal   = self._causal(L, device)
        for layer in self.layers:
            x = layer(x, causal, pad_mask)
        x = self.norm(x)                                             # (B, L, D)

        # 取最后位置（左 padding → 最后一列是最新 item）
        seq_repr = torch.nan_to_num(x[:, -1, :], nan=0.0)           # (B, D)
        return seq_repr, x                                           # seq_hidden 供辅助任务


# ── 4b. DCNv2 Cross Network ────────────────────────────────
class CrossNetV2(nn.Module):
    """
    DCN v2 Cross Network：矩阵权重显式高阶特征交叉
    每层：x_{l+1} = x_0 ⊙ (W_l x_l + b_l) + x_l
    W_l ∈ R^{d×d}（相比 v1 的向量 w_l ∈ R^d，表达力更强）
    叠加 n_cross 层 → 可捕捉 n_cross+1 阶显式交叉

    低秩分解（可选）：W_l ≈ U_l V_l^T，U,V ∈ R^{d×r}
    当 d 较大时用低秩近似降低参数量，r = d // 4
    """
    def __init__(self, d, n_cross, low_rank=True):
        super().__init__()
        r = max(1, d // 4) if low_rank else d
        self.W_us = nn.ParameterList(
            [nn.Parameter(torch.empty(d, r)) for _ in range(n_cross)])
        self.W_vs = nn.ParameterList(
            [nn.Parameter(torch.empty(d, r)) for _ in range(n_cross)])
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(d)) for _ in range(n_cross)])
        for i in range(n_cross):
            nn.init.xavier_normal_(self.W_us[i])
            nn.init.xavier_normal_(self.W_vs[i])

    def forward(self, x):
        """x: (B, d) → (B, d)"""
        x0 = x
        for Wu, Wv, b in zip(self.W_us, self.W_vs, self.biases):
            # 低秩矩阵乘：x @ Wv @ Wu.T + b
            # = x0 ⊙ ((x Wv) Wu^T + b) + x
            tmp = x @ Wv           # (B, r)
            tmp = tmp @ Wu.T       # (B, d)
            x   = x0 * (tmp + b) + x
        return x


class DCNv2(nn.Module):
    """
    DCN v2 = 并行 Cross Network + Deep Network → 拼接 → Linear

    输入：拼接后的特征向量 (B, in_dim)
    输出：(B, out_dim)

    parallel 模式（本实现）：
      cross_out = CrossNet(x)              (B, in_dim)
      deep_out  = MLP(x)                   (B, deep_dims[-1])
      out       = Linear(concat[cross; deep])
    """
    def __init__(self, in_dim, n_cross, deep_dims, out_dim, dropout, low_rank=True):
        super().__init__()
        self.cross = CrossNetV2(in_dim, n_cross, low_rank)
        self.deep  = build_mlp(in_dim, deep_dims[:-1], deep_dims[-1], dropout)
        self.out   = nn.Linear(in_dim + deep_dims[-1], out_dim)
        self.norm  = nn.LayerNorm(out_dim)

    def forward(self, x):
        cross_out = self.cross(x)                          # (B, in_dim)
        deep_out  = self.deep(x)                           # (B, deep[-1])
        combined  = torch.cat([cross_out, deep_out], -1)   # (B, in_dim+deep[-1])
        return self.norm(F.gelu(self.out(combined)))        # (B, out_dim)


# ── 4c. Item Tower ─────────────────────────────────────────
class ItemTower(nn.Module):
    """item_id + 4域特征 → FM二阶 + MLP → item_repr (D)（独立 embedding）"""
    def __init__(self, cfg, n_items, item_feat_dims, item_feat_padidxs):
        super().__init__()
        E = cfg.emb_dim; D = cfg.repr_dim; n_i = 4
        self.item_id_emb = nn.Embedding(n_items+1, E, padding_idx=0)
        self.feat_embs   = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)])
        self.mlp = build_mlp(E + E*n_i + E, cfg.item_mlp_dims, D, cfg.dropout)

    def forward(self, item_ids, item_feats):
        id_e    = self.item_id_emb(item_ids)
        fi_list = [e(item_feats[...,j]) for j,e in enumerate(self.feat_embs)]
        fm2     = fm_second_order(fi_list)
        fi_cat  = torch.cat(fi_list, -1)
        return self.mlp(torch.cat([id_e, fi_cat, fm2], -1))


# ── 4d. 对比学习：投影头 + InfoNCE ─────────────────────────
class ProjectionHead(nn.Module):
    """
    将 user_repr 投影到对比学习空间。
    SimCLR 原论文建议使用 2 层非线性 MLP，然后在投影后的空间计算对比损失，
    保留原 repr 空间用于主任务（不受对比 loss 直接污染）。
    """
    def __init__(self, in_dim, proj_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1, eps=1e-8)


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    InfoNCE（NT-Xent）双向对比损失。
    z1, z2 : (B, proj_dim)，已 L2 归一化
    正对   : (z1[i], z2[i])  同一用户的两个增强视图
    负例   : batch 内其余 2B-2 个样本（无需额外采样）

    Loss = -1/(2B) * Σ [ log sim(zi,zj) / Σ_{k≠i} sim(zi,zk) ]
         = CrossEntropy(sim_matrix / τ, labels)
    """
    B      = z1.size(0)
    device = z1.device
    # 拼成 2B
    z      = torch.cat([z1, z2], dim=0)             # (2B, D)
    # 相似度矩阵
    sim    = z @ z.T / temperature                   # (2B, 2B)
    # 对角线（自身）mask 掉
    mask   = torch.eye(2*B, dtype=torch.bool, device=device)
    sim    = sim.masked_fill(mask, -1e9)
    # 正样本标签：z1[i] 的正对是 z2[i]（index B+i），反之亦然
    labels = torch.cat([
        torch.arange(B, 2*B, device=device),
        torch.arange(0, B,   device=device)], dim=0)  # (2B,)
    return F.cross_entropy(sim, labels)


# ══════════════════════════════════════════════════════════
# 5. 主模型
# ══════════════════════════════════════════════════════════
class TransDCNv2CL(nn.Module):
    """
    Transformer + DCNv2 + Contrastive Learning 推荐模型

    User Tower 流程：
      seq_ids, user_feats
        ↓
      SeqEncoder → seq_repr (D)
        ↓
      concat[seq_repr; user_feat_embs_concat; item_feat_embs_of_last_item]
        → DCNv2 → user_repr (D)
        ↓
      ProjectionHead → z (proj_dim)  ← 对比学习专用

    Item Tower：独立，item_repr (N+1, D)

    打分：cosine(user_repr, item_repr) × τ
    """
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 user_feat_dims, user_feat_padidxs):
        super().__init__()
        E = cfg.emb_dim
        D = cfg.repr_dim
        self.n_items     = n_items
        self.MASK_ID     = n_items + 1
        self.label_smooth = cfg.label_smooth
        self.aux_weight   = cfg.aux_weight
        self.cl_weight    = cfg.cl_weight
        self.cl_temp      = cfg.cl_temp

        # ── 序列编码器 ──
        self.seq_enc = SeqEncoder(cfg, n_items, item_feat_dims, item_feat_padidxs)

        # ── 用户特征 embedding（8域）──
        self.u_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(user_feat_dims, user_feat_padidxs)])
        n_u = len(user_feat_dims)   # 8

        # ── DCNv2 输入维度 = seq_repr(D) + 用户特征concat(8E) ──
        dcn_in = D + E * n_u
        self.dcn = DCNv2(
            in_dim   = dcn_in,
            n_cross  = cfg.n_cross,
            deep_dims= cfg.deep_dims,
            out_dim  = D,
            dropout  = cfg.dropout)

        # ── Item Tower ──
        self.item_tower = ItemTower(cfg, n_items, item_feat_dims, item_feat_padidxs)

        # ── 对比学习投影头 ──
        self.proj_head = ProjectionHead(D, D // 2, cfg.dropout)

        # ── 辅助任务：Masked Item Prediction ──
        self.aux_head = nn.Linear(D, n_items + 2)

        # ── 可学习温度（主任务打分）──
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
        return -(oh*(1-self.label_smooth)*lp + s*lp).sum(-1).mean()

    def _aux_loss(self, seq_hidden, mask_targets):
        mask_pos = (mask_targets != 0)
        if not mask_pos.any():
            return torch.tensor(0.0, device=seq_hidden.device)
        hidden = seq_hidden[mask_pos]
        tgt    = mask_targets[mask_pos]
        logits = torch.clamp(self.aux_head(hidden), -50, 50)
        return F.cross_entropy(logits, tgt)

    def encode_user(self, seq_ids, ift, user_feats):
        """
        seq_ids    : (B, L)
        ift        : (vocab, 4)
        user_feats : (B, 8)
        return     : user_repr (B, D),  seq_hidden (B, L, D)
        """
        device = seq_ids.device

        # 序列编码
        seq_repr, seq_hidden = self.seq_enc(seq_ids, ift)   # (B,D), (B,L,D)

        # 用户特征 embedding 拼接
        u_emb_list = [e(user_feats[:,j]) for j,e in enumerate(self.u_embs)]
        u_cat      = torch.cat(u_emb_list, dim=-1)          # (B, 8E)

        # DCNv2 输入
        dcn_input  = torch.cat([seq_repr, u_cat], dim=-1)   # (B, D+8E)
        user_repr  = self.dcn(dcn_input)                    # (B, D)
        user_repr  = torch.nan_to_num(user_repr, nan=0.0)

        return user_repr, seq_hidden

    def forward(self, seq_ids, masked_seq_ids, ift, user_feats,
                target_ids=None, mask_targets=None, mixup_alpha=0.0):
        device = seq_ids.device

        # ── User Tower（用 masked_seq 训练，原始 seq 推理）──
        enc_input = masked_seq_ids if target_ids is not None else seq_ids
        user_repr, seq_hidden = self.encode_user(enc_input, ift, user_feats)

        # ── Item Tower：全量 ──
        all_ids   = torch.arange(self.n_items+1, device=device)
        all_feats = ift.to(device)[:self.n_items+1]
        item_repr = self.item_tower(all_ids, all_feats)      # (N+1, D)

        # ── 推理 ──
        if target_ids is None:
            return self._score(user_repr, item_repr)

        # ── 训练：Mixup ──
        if mixup_alpha > 0:
            lam  = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx  = torch.randperm(user_repr.size(0), device=device)
            user_repr_m  = lam * user_repr + (1-lam) * user_repr[idx]
            target_ids_b = target_ids[idx]
        else:
            user_repr_m  = user_repr; lam = 1.0; target_ids_b = target_ids

        # ── 主任务 loss ──
        logits    = self._score(user_repr_m, item_repr)
        main_loss = self._ce(logits[:,1:], target_ids - 1)
        if mixup_alpha > 0:
            main_loss = (lam*main_loss +
                (1-lam)*self._ce(logits[:,1:], target_ids_b - 1))

        # ── 对比学习 loss ──
        # 对原始序列做两次独立增强，各自过 encode_user → 投影头
        cl_loss = torch.tensor(0.0, device=device)
        if self.cl_weight > 0:
            seq_aug1 = random_augment(seq_ids, self.MASK_ID, cfg)
            seq_aug2 = random_augment(seq_ids, self.MASK_ID, cfg)
            repr1, _ = self.encode_user(seq_aug1, ift, user_feats)
            repr2, _ = self.encode_user(seq_aug2, ift, user_feats)
            z1 = self.proj_head(repr1)    # (B, proj_dim) 已 L2 归一化
            z2 = self.proj_head(repr2)
            cl_loss  = info_nce_loss(z1, z2, self.cl_temp)

        # ── 辅助任务 loss（Masked Item Prediction）──
        aux_loss = torch.tensor(0.0, device=device)
        if self.aux_weight > 0 and mask_targets is not None:
            aux_loss = self._aux_loss(seq_hidden, mask_targets)

        return main_loss + self.cl_weight * cl_loss + self.aux_weight * aux_loss


# ══════════════════════════════════════════════════════════
# 6. 训练 & 评估
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
        logits[:,0]  = -1e9
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
        logits[:,0] = -1e9
        tk = logits.topk(topk, dim=-1).indices.cpu().tolist()
        for i, uid in enumerate(uids):
            items = [id2item[iid] for iid in tk[i] if iid in id2item]
            rows.append({"uid": uid, "predicted_items": ",".join(items)})
    return rows


# ══════════════════════════════════════════════════════════
# 7. 主函数
# ══════════════════════════════════════════════════════════
def main():
    print(f"Device: {cfg.device}\n")
    proc = DataProcessor(cfg)
    train_samples, test_samples = proc.load_and_build()

    all_uids  = sorted(set(s["uid"] for s in train_samples))
    n_val_u   = max(1, int(len(all_uids)*0.1))
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

    model = TransDCNv2CL(
        cfg, proc.n_items,
        proc.item_feat_dims, proc.item_feat_padidxs,
        proc.user_feat_dims, proc.user_feat_padidxs,
    ).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

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
