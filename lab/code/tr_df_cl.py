"""
ContrastFormer: Transformer + DeepFM + 对比学习 推荐系统
=========================================================

设计思路
--------
在 HyFormer 基础上，引入三点核心改进：

1. DeepFM 增强的 Item Tower
   原 HyFormer 的 Item Tower 只做 FM 二阶 + MLP（浅层交叉）。
   本模型将 Item Tower 替换为完整 DeepFM 结构：
     FM 一阶（线性项）
     FM 二阶（embedding 内积，低阶交叉）
     Deep 分支（embedding concat → MLP，高阶交叉）
   三路输出拼接 → Linear → item_repr

2. DeepFM 增强的 User Side
   User Tower 同样引入 DeepFM 分支：
   在 GT mean pooling 基础上，再补充一路 DeepFM Cross（用户8个特征域的
   FM 二阶 + Deep），与 GT pooling 拼接后过 MLP 得到 user_repr。
   低阶（FM）和高阶（Transformer GT）特征互补。

3. 对比学习（InfoNCE / NT-Xent）
   使用两路增强视图生成用户表示，训练时对比：
     View-A：原始序列 + 随机 Dropout（mask_prob=0.15）
     View-B：序列时序打乱（shuffle augmentation）
   对同一用户的两路表示做 NT-Xent 对比损失（batch 内其他用户作负样本），
   约束用户表示在语义上保持一致，抑制噪声序列带来的表示抖动。

   总损失 = main_CE + aux_weight * MIP + cl_weight * NT-Xent

整体流程
--------

  输入特征
  ├─ 行为序列   seq_ids (B, L)
  │    → SeqTokenizer → SeqTokens (B, L, D)
  │    → View-A (mask)  / View-B (shuffle)
  │
  └─ 用户特征   user_feats (B, n_u)
       → GlobalTokenInit → GT (B, n_gt, D)
       → UserDeepFM      → fm_repr (B, D)

  HyFormer Layer × n_layers
  ┌───────────────────────────────────┐
  │  QueryDecoding (QD)               │
  │    seq causal self-attn           │
  │    GT cross-attn(Q=GT, KV=seq)    │
  │  QueryBoosting (QB)               │
  │    [GT | seq_mean] self-attn      │
  └───────────────────────────────────┘

  GT_final mean-pool (B, D) → concat → fm_repr (B, D)
  → UserMLP → user_repr (B, D)

  Item Tower (DeepFM)
  item_id + n_i 特征 → FM1 + FM2 + Deep → item_repr (N+1, D)

  打分
  cosine(user_repr, item_repr) × exp(log_temp)

  损失
  main_CE (全量 Softmax) +
  aux_weight * MIP (Masked Item Prediction) +
  cl_weight * NT-Xent (对比学习，两视图)
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
# class Config:
#     train_path  = "../data/A2-Rec/train.csv"
#     test_path   = "../data/A2-Rec/test.csv"
#     user_path   = "../data/A2-Rec/user.csv"
#     item_path   = "../data/A2-Rec/item.csv"
#     output_path = "submission_contrastformer.csv"

#     max_seq_len = 50
#     emb_dim     = 32        # 每个特征域 embedding 维度 E
#     repr_dim    = 128       # 主干隐层维度 D

#     # HyFormer backbone
#     n_layers    = 2
#     n_heads     = 4
#     n_gt        = 8         # Global Token 数量 = 用户特征域数
#     qb_n_heads  = 4

#     # Item/User DeepFM
#     deep_dims   = [256, 128]   # Deep 分支 hidden dims
#     dropout     = 0.2

#     # 对比学习
#     cl_weight   = 0.1          # NT-Xent 损失权重
#     cl_temp     = 0.07         # 对比温度
#     shuffle_prob = 0.3         # View-B 序列打乱比例

#     # 辅助任务 Masked Item Prediction
#     aux_weight  = 0.05
#     mask_prob   = 0.15

#     # 训练
#     epochs      = 50
#     batch_size  = 256
#     lr          = 1e-3
#     weight_decay = 1e-5
#     label_smooth = 0.0
#     mixup_alpha  = 0.1

#     seed = 42
#     topk = 10

#     device = "cuda" if torch.cuda.is_available() else "cpu"

from config import Config

cfg = Config()
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)


# ══════════════════════════════════════════════════════════
# 1. 数据处理（沿用 HyFormer 结构，支持 item_seq_counts）
# ══════════════════════════════════════════════════════════
class DataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.item2id = {}; self.id2item = {}
        self.iid2feat = {}; self.uid2feat = {}
        # 4个 item cat 特征列（根据实际数据修改）
        self.item_feat_cols = ["i_cat_01", "i_cat_02", "i_cat_03", "i_bucket_01"]
        # 8个 user cat 特征列
        self.user_feat_cols = [f"u_cat_0{i}" for i in range(1, 9)]
        self.item_feat_dims = []; self.item_feat_padidxs = []
        self.user_feat_dims = []; self.user_feat_padidxs = []
        self.n_items = 0

    def _parse_seq(self, s):
        if pd.isna(s) or not str(s).strip(): return []
        return [x.strip() for x in str(s).split(",") if x.strip()]

    def _parse_counts(self, s):
        """解析 item_seq_counts: 'item1:3,item2:1,...' → dict"""
        if pd.isna(s) or not str(s).strip(): return {}
        d = {}
        for p in str(s).split(","):
            if ":" in p:
                k, v = p.strip().rsplit(":", 1)
                d[k.strip()] = int(v)
        return d

    def _weight_seq(self, seq_strs, counts):
        """
        将点击次数 >= 2 的 item 追加到序列末尾（频次加权）。
        与 HyFormer 保持一致。
        """
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

        # ── item 词表：0=PAD, 1..N=item, N+1=MASK ──
        all_items = set(item_df["iid"].str.strip())
        for col in ["item_seq_raw", "item_seq_dedup"]:
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
        print(f"  Items: {self.n_items}  MASK_ID: {self.MASK_ID}")

        # ── item 特征 ──
        item_df["iid"] = item_df["iid"].str.strip()
        for col in self.item_feat_cols:
            item_df[col] = item_df[col].fillna(-1).astype(int)
        for col in self.item_feat_cols:
            rmax = int(item_df[col].max())
            self.item_feat_padidxs.append(rmax + 1)
            self.item_feat_dims.append(rmax + 2)
        pad_ifeat = list(self.item_feat_padidxs)
        for _, row in item_df.iterrows():
            iid = self.item2id.get(row["iid"])
            if iid:
                self.iid2feat[iid] = [int(row[c]) for c in self.item_feat_cols]
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
        pad_ufeat = list(self.user_feat_padidxs)
        uid_feat_map = {row["uid"]: [int(row[c]) for c in self.user_feat_cols]
                        for _, row in user_df.iterrows()}

        # ── 构建样本 ──
        train_df["uid"]        = train_df["uid"].str.strip()
        train_df["target_iid"] = train_df["target_iid"].str.strip()
        test_df["uid"]         = test_df["uid"].str.strip()

        train_samples, test_samples = [], []
        for _, row in train_df.iterrows():
            uid    = row["uid"]
            target = self.item2id.get(row["target_iid"])
            if not target: continue
            seq  = self._parse_seq(row.get("item_seq_dedup", ""))
            cnts = self._parse_counts(row.get("item_seq_counts", ""))
            train_samples.append({
                "uid": uid,
                "seq": self._weight_seq(seq, cnts),
                "target": target,
                "user_feat": uid_feat_map.get(uid, pad_ufeat),
            })
        for _, row in test_df.iterrows():
            uid  = row["uid"]
            seq  = self._parse_seq(row.get("item_seq_dedup", ""))
            cnts = self._parse_counts(row.get("item_seq_counts", ""))
            test_samples.append({
                "uid": uid,
                "seq": self._weight_seq(seq, cnts),
                "user_feat": uid_feat_map.get(uid, pad_ufeat),
            })
        print(f"  Train: {len(train_samples)} | Test: {len(test_samples)}")

        # 全量 item 特征张量 (N+2, n_i)
        n_i = len(self.item_feat_cols)
        ift = torch.zeros(self.n_items + 2, n_i, dtype=torch.long)
        for iid, feats in self.iid2feat.items():
            ift[iid] = torch.tensor(feats)
        ift[self.MASK_ID] = torch.tensor(pad_ifeat)
        self.item_feat_tensor = ift
        return train_samples, test_samples


# ══════════════════════════════════════════════════════════
# 2. Dataset（返回两路对比视图 + 原序列 + mask序列）
# ══════════════════════════════════════════════════════════
def pad_seq(seq, max_len):
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


class RecDataset(Dataset):
    """
    训练时返回：
      seq_orig    原始序列（用于 pad_mask）
      seq_mask    View-A：随机 mask 序列（MIP辅助任务输入）
      seq_shuffle View-B：随机 shuffle 部分序列（对比第二视图）
      user_feat   用户特征
      target      目标 item id
      mask_tgt    MIP 标签

    推理时返回：seq, user_feat, uid
    """
    def __init__(self, samples, max_len, n_items, mask_id,
                 mask_prob=0.0, shuffle_prob=0.0, mode="train"):
        self.samples     = samples
        self.max_len     = max_len
        self.n_items     = n_items
        self.mask_id     = mask_id
        self.mask_prob   = mask_prob
        self.shuffle_prob = shuffle_prob
        self.mode        = mode

    def __len__(self): return len(self.samples)

    def _mask(self, seq):
        """随机 mask → View-A / MIP 标签"""
        masked = seq.copy(); tgt = [0] * len(seq)
        for i, s in enumerate(seq):
            if s != 0 and random.random() < self.mask_prob:
                tgt[i] = s; masked[i] = self.mask_id
        return masked, tgt

    def _shuffle(self, seq):
        """
        View-B：在非 PAD 段中随机交换若干对位置，保持长度不变。
        shuffle_prob 控制被交换的 token 比例。
        """
        arr = seq.copy()
        valid_idx = [i for i, s in enumerate(arr) if s != 0]
        n_swap = max(1, int(len(valid_idx) * self.shuffle_prob))
        for _ in range(n_swap):
            if len(valid_idx) < 2: break
            i, j = random.sample(valid_idx, 2)
            arr[i], arr[j] = arr[j], arr[i]
        return arr

    def __getitem__(self, idx):
        s   = self.samples[idx]
        seq = pad_seq(s["seq"], self.max_len)
        uf  = torch.tensor(s["user_feat"], dtype=torch.long)

        if self.mode == "train":
            seq_mask, mtgt = self._mask(seq)
            seq_shuf       = self._shuffle(seq)
            return (
                torch.tensor(seq,      dtype=torch.long),   # 原始序列
                torch.tensor(seq_mask, dtype=torch.long),   # View-A
                torch.tensor(seq_shuf, dtype=torch.long),   # View-B
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
    """FM 二阶交叉：∑i∑j<vi,vj> = 0.5*(∑vi)^2 - ∑vi^2"""
    stacked = torch.stack(emb_list, dim=-2)          # (..., n_fields, E)
    return 0.5 * (stacked.sum(-2)**2 - (stacked**2).sum(-2))


# ── Query Decoding（与 HyFormer 完全一致）────────────────
class QueryDecoding(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.seq_self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.seq_n1 = nn.LayerNorm(d)
        self.seq_ff = nn.Sequential(nn.Linear(d, d*4), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(d*4, d))
        self.seq_n2 = nn.LayerNorm(d)
        self.gt_cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.gt_n1  = nn.LayerNorm(d)
        self.gt_ff  = nn.Sequential(nn.Linear(d, d*4), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(d*4, d))
        self.gt_n2  = nn.LayerNorm(d)
        self.drop   = nn.Dropout(dropout)

    def _causal(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

    def forward(self, gt, seq_hidden, seq_pad_mask):
        B, L, D = seq_hidden.shape
        causal = self._causal(L, seq_hidden.device)

        # 序列因果自注意力
        r = seq_hidden; x = self.seq_n1(seq_hidden)
        x, _ = self.seq_self_attn(x, x, x, attn_mask=causal,
                                  key_padding_mask=seq_pad_mask, need_weights=False)
        x = torch.nan_to_num(self.drop(x), nan=0.0) + r
        r = x; x = self.seq_n2(x)
        seq_hidden = torch.nan_to_num(self.drop(self.seq_ff(x)), nan=0.0) + r

        # GT cross-attention
        r = gt; q = self.gt_n1(gt)
        x, _ = self.gt_cross_attn(q, seq_hidden, seq_hidden,
                                   key_padding_mask=seq_pad_mask, need_weights=False)
        x = torch.nan_to_num(self.drop(x), nan=0.0) + r
        r = x; x = self.gt_n2(x)
        gt = torch.nan_to_num(self.drop(self.gt_ff(x)), nan=0.0) + r
        return gt, seq_hidden


# ── Query Boosting（与 HyFormer 完全一致）───────────────
class QueryBoosting(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.n1   = nn.LayerNorm(d)
        self.n2   = nn.LayerNorm(d)
        self.ff   = nn.Sequential(nn.Linear(d, d*4), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d*4, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, gt, seq_hidden, seq_pad_mask):
        valid    = (~seq_pad_mask).float().unsqueeze(-1)
        seq_mean = (seq_hidden * valid).sum(1) / valid.sum(1).clamp(min=1)
        seq_tok  = seq_mean.unsqueeze(1)
        tokens   = torch.cat([gt, seq_tok], dim=1)

        r = tokens; x = self.n1(tokens)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = torch.nan_to_num(self.drop(x), nan=0.0) + r
        r = x; x = self.n2(x)
        x = torch.nan_to_num(self.drop(self.ff(x)), nan=0.0) + r
        return x[:, :gt.shape[1], :]


class HyFormerLayer(nn.Module):
    def __init__(self, d, n_heads, qb_n_heads, dropout):
        super().__init__()
        self.qd = QueryDecoding(d, n_heads, dropout)
        self.qb = QueryBoosting(d, qb_n_heads, dropout)

    def forward(self, gt, seq_hidden, seq_pad_mask):
        gt, seq_hidden = self.qd(gt, seq_hidden, seq_pad_mask)
        gt             = self.qb(gt, seq_hidden, seq_pad_mask)
        return gt, seq_hidden


# ── SeqTokenizer（与 HyFormer 一致）─────────────────────
class SeqTokenizer(nn.Module):
    def __init__(self, cfg, n_items, item_feat_dims, item_feat_padidxs):
        super().__init__()
        E = cfg.emb_dim; D = cfg.repr_dim
        self.item_id_emb = nn.Embedding(n_items + 2, E, padding_idx=0)
        self.feat_embs   = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)])
        self.n_feat = len(item_feat_dims)
        self.proj   = nn.Linear(E * 2, D)
        self.pos_emb = nn.Embedding(cfg.max_seq_len + 1, D)
        self.norm   = nn.LayerNorm(D)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, seq_ids, ift):
        B, L   = seq_ids.shape
        device = seq_ids.device
        ift    = ift.to(device)
        id_e   = self.item_id_emb(seq_ids)
        feats  = ift[seq_ids]
        f_list = [self.feat_embs[j](feats[:, :, j]) for j in range(self.n_feat)]
        f_mean = torch.stack(f_list, dim=-2).mean(-2)
        x = F.gelu(self.proj(torch.cat([id_e, f_mean], dim=-1)))
        pos = torch.arange(1, L + 1, device=device).unsqueeze(0)
        x = self.norm(self.drop(x + self.pos_emb(pos)))
        return x


# ── GlobalTokenInit（与 HyFormer 一致）──────────────────
class GlobalTokenInit(nn.Module):
    def __init__(self, cfg, user_feat_dims, user_feat_padidxs):
        super().__init__()
        E = cfg.emb_dim; D = cfg.repr_dim
        self.u_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(user_feat_dims, user_feat_padidxs)])
        self.projs  = nn.ModuleList([nn.Linear(E, D) for _ in user_feat_dims])
        self.norm   = nn.LayerNorm(D)

    def forward(self, user_feats):
        gts = [proj(emb(user_feats[:, j]))
               for j, (emb, proj) in enumerate(zip(self.u_embs, self.projs))]
        gt = torch.stack(gts, dim=1)
        return self.norm(gt)


# ──【新增】UserDeepFM：用户侧低阶特征交叉 ───────────────
class UserDeepFM(nn.Module):
    """
    用户 n_u 个 cat 特征 → FM 二阶 + Deep MLP → repr_dim
    与 Transformer GT 路径互补，捕捉用户特征间低/高阶交叉。

    输出维度 = repr_dim，与 GT pooling 拼接后送入 user_head。
    """
    def __init__(self, cfg, user_feat_dims, user_feat_padidxs):
        super().__init__()
        E  = cfg.emb_dim
        D  = cfg.repr_dim
        nu = len(user_feat_dims)

        self.u_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(user_feat_dims, user_feat_padidxs)])

        # FM 一阶线性项（每域 embedding → 标量求和）
        self.fm1_projs = nn.ModuleList([nn.Linear(E, 1, bias=False) for _ in range(nu)])

        # Deep 分支：concat all embeddings → MLP
        self.deep = build_mlp(E * nu, cfg.deep_dims, D, cfg.dropout)

        # FM 二阶 (E) + Deep (D) → D
        self.out_proj = nn.Linear(E + D, D)
        self.norm     = nn.LayerNorm(D)

    def forward(self, user_feats):
        """user_feats: (B, n_u) → (B, D)"""
        embs = [self.u_embs[j](user_feats[:, j]) for j in range(len(self.u_embs))]

        # FM 一阶（可选，加入偏置项）
        # fm1 = sum(proj(e) for proj, e in zip(self.fm1_projs, embs))  # (B,1)

        # FM 二阶
        fm2 = fm_second_order(embs)         # (B, E)

        # Deep
        cat_emb = torch.cat(embs, dim=-1)   # (B, n_u*E)
        deep    = self.deep(cat_emb)        # (B, D)

        out = self.out_proj(torch.cat([fm2, deep], dim=-1))  # (B, D)
        return self.norm(out)


# ──【新增】DeepFM Item Tower：FM1 + FM2 + Deep ──────────
class DeepFMItemTower(nn.Module):
    """
    完整 DeepFM Item Tower：
      FM 一阶  : item_id + n_i 特征各自线性投影求和
      FM 二阶  : embedding 内积（fm_second_order）
      Deep 分支: [item_id_emb; feat1_emb; ...; featN_emb] → MLP
    三路拼接 → Linear → repr_dim
    """
    def __init__(self, cfg, n_items, item_feat_dims, item_feat_padidxs):
        super().__init__()
        E  = cfg.emb_dim
        D  = cfg.repr_dim
        ni = len(item_feat_dims)

        # embeddings
        self.item_id_emb = nn.Embedding(n_items + 1, E, padding_idx=0)
        self.feat_embs   = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)])

        # FM 一阶：每个域一个标量 bias
        self.fm1_bias = nn.Parameter(torch.zeros(1))
        self.fm1_projs = nn.ModuleList(
            [nn.Linear(E, 1, bias=False) for _ in range(ni + 1)])  # +1 for item_id

        # Deep 分支：(ni+1)*E → hidden → D
        self.deep = build_mlp(E * (ni + 1), cfg.deep_dims, D, cfg.dropout)

        # FM二阶(E) + Deep(D) → D
        self.out_proj = nn.Linear(E + D, D)
        self.norm     = nn.LayerNorm(D)

    def forward(self, item_ids, item_feats):
        """
        item_ids   : (...,)
        item_feats : (..., ni)
        returns    : (..., D)
        """
        id_e    = self.item_id_emb(item_ids)                           # (..., E)
        fi_list = [e(item_feats[..., j]) for j, e in enumerate(self.feat_embs)]

        # FM 二阶（item_id + 各特征域一起参与）
        all_embs = [id_e] + fi_list
        fm2 = fm_second_order(all_embs)                                # (..., E)

        # Deep
        cat_emb = torch.cat(all_embs, dim=-1)                         # (..., (ni+1)*E)
        deep    = self.deep(cat_emb)                                   # (..., D)

        out = self.out_proj(torch.cat([fm2, deep], dim=-1))            # (..., D)
        return self.norm(out)


# ══════════════════════════════════════════════════════════
# 4. ContrastFormer 主模型
# ══════════════════════════════════════════════════════════
class ContrastFormerModel(nn.Module):
    """
    Transformer + DeepFM + 对比学习 端到端推荐模型。

    User side：
      HyFormer backbone（GT via QD+QB）
      UserDeepFM（用户特征低阶交叉）
      concat → user_head → user_repr

    Item side：
      DeepFM Item Tower（FM1+FM2+Deep）

    训练损失：
      main  = 全量 Softmax CE
      aux   = Masked Item Prediction（MIP）
      cl    = NT-Xent 对比（View-A mask vs View-B shuffle）
      total = main + aux_weight*aux + cl_weight*cl
    """
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 user_feat_dims, user_feat_padidxs):
        super().__init__()
        D = cfg.repr_dim
        self.n_items      = n_items
        self.label_smooth = cfg.label_smooth
        self.aux_weight   = cfg.aux_weight
        self.cl_weight    = cfg.cl_weight
        self.cl_temp      = cfg.cl_temp
        self.MASK_ID      = n_items + 1

        # ── User Encoder ──
        self.seq_tokenizer = SeqTokenizer(cfg, n_items, item_feat_dims, item_feat_padidxs)
        self.gt_init       = GlobalTokenInit(cfg, user_feat_dims, user_feat_padidxs)
        self.layers        = nn.ModuleList([
            HyFormerLayer(D, cfg.n_heads, cfg.qb_n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)])

        # ── UserDeepFM（新增：低阶用户特征交叉）──
        self.user_deepfm = UserDeepFM(cfg, user_feat_dims, user_feat_padidxs)

        # ── User Head：GT pooling (D) + DeepFM (D) → D ──
        self.user_head = nn.Sequential(
            nn.LayerNorm(D * 2),
            nn.Linear(D * 2, D),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(D, D))

        # ── Item Tower（新增：完整 DeepFM）──
        self.item_tower = DeepFMItemTower(cfg, n_items, item_feat_dims, item_feat_padidxs)

        # ── 辅助任务：MIP ──
        self.aux_head = nn.Linear(D, n_items + 2)

        # ── 对比学习投影头（NT-Xent 用）──
        self.cl_proj = nn.Sequential(
            nn.Linear(D, D),
            nn.GELU(),
            nn.Linear(D, D))

        # ── 可学习温度 ──
        self.log_temp = nn.Parameter(torch.tensor(3.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ──────────────────────────────────────────────────────
    # 辅助：打分 / CE / MIP / 对比损失
    # ──────────────────────────────────────────────────────
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
        return -(oh * (1 - self.label_smooth) * lp + s * lp).sum(-1).mean()

    def _aux_loss(self, seq_hidden, mask_targets):
        mask_pos = (mask_targets != 0)
        if not mask_pos.any():
            return torch.tensor(0.0, device=seq_hidden.device)
        hidden = seq_hidden[mask_pos]
        tgt    = mask_targets[mask_pos]
        logits = torch.clamp(self.aux_head(hidden), -50, 50)
        return F.cross_entropy(logits, tgt)

    def _nt_xent(self, z1, z2):
        """
        NT-Xent 对比损失（InfoNCE / SimCLR 风格）。
        z1, z2: (B, D) 已 L2 归一化
        同一位置的 (z1[i], z2[i]) 为正样本对，
        batch 内其他位置为负样本。
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                     # (2B, D)
        z = F.normalize(z, p=2, dim=-1, eps=1e-8)
        sim = (z @ z.T) / self.cl_temp                     # (2B, 2B)

        # 去掉对角线自相似
        mask_diag = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask_diag, -1e9)

        # 正样本标签：z1[i] 的正样本是 z2[i]（位置 B+i），反之亦然
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B,     device=z.device)], dim=0)

        return F.cross_entropy(sim, labels)

    # ──────────────────────────────────────────────────────
    # 核心编码：接受一路 seq_ids（已 mask 或 shuffle）
    # ──────────────────────────────────────────────────────
    def _encode_user(self, seq_ids_raw, seq_ids_view, ift, user_feats):
        """
        seq_ids_raw  : (B, L) 原始序列（用于 pad_mask）
        seq_ids_view : (B, L) 增强视图序列（View-A mask / View-B shuffle）
        ift          : (vocab, n_i) item 特征张量
        user_feats   : (B, n_u)
        returns      : user_repr (B, D), seq_hidden (B, L, D)
        """
        device       = seq_ids_raw.device
        ift          = ift.to(device)
        seq_pad_mask = (seq_ids_raw == 0)                    # (B, L) True=pad

        # 序列 token（用增强视图）
        seq_hidden = self.seq_tokenizer(seq_ids_view, ift)   # (B, L, D)

        # Global Token 初始化
        gt = self.gt_init(user_feats)                        # (B, n_gt, D)

        # HyFormer 层
        for layer in self.layers:
            gt, seq_hidden = layer(gt, seq_hidden, seq_pad_mask)

        # GT mean pooling
        gt_pool = gt.mean(dim=1)                             # (B, D)

        # UserDeepFM（低阶用户特征交叉）
        fm_repr = self.user_deepfm(user_feats)               # (B, D)

        # 拼接 → user_repr
        user_repr = self.user_head(torch.cat([gt_pool, fm_repr], dim=-1))  # (B, D)
        user_repr = torch.nan_to_num(user_repr, nan=0.0)
        return user_repr, seq_hidden

    # ──────────────────────────────────────────────────────
    # forward
    # ──────────────────────────────────────────────────────
    def forward(self, seq_orig, seq_mask, seq_shuf,
                ift, user_feats,
                target_ids=None, mask_targets=None,
                mixup_alpha=0.0):
        """
        seq_orig    : (B, L) 原始序列（pad_mask 基准）
        seq_mask    : (B, L) View-A（随机 mask 序列，MIP 输入）
        seq_shuf    : (B, L) View-B（随机 shuffle 序列，对比第二视图）
        ift         : (vocab, n_i)
        user_feats  : (B, n_u)
        target_ids  : (B,) 训练用，None 时走推理
        mask_targets: (B, L) MIP 标签
        mixup_alpha : Mixup 系数（0 关闭）
        """
        device = seq_orig.device

        # ── 编码 View-A（mask序列，也作为主任务 user_repr）──
        user_repr_a, seq_hidden_a = self._encode_user(
            seq_orig, seq_mask, ift, user_feats)

        # 推理路径
        if target_ids is None:
            all_ids   = torch.arange(self.n_items + 1, device=device)
            all_feats = ift.to(device)[:self.n_items + 1]
            item_repr = self.item_tower(all_ids, all_feats)
            return self._score(user_repr_a, item_repr)

        # ── 编码 View-B（shuffle序列，用于对比学习）──
        user_repr_b, _ = self._encode_user(
            seq_orig, seq_shuf, ift, user_feats)

        # ── Item Tower（全量 DeepFM）──
        all_ids   = torch.arange(self.n_items + 1, device=device)
        all_feats = ift.to(device)[:self.n_items + 1]
        item_repr = self.item_tower(all_ids, all_feats)              # (N+1, D)

        # ── Mixup（只在 View-A 的 user_repr 上做）──
        if mixup_alpha > 0:
            lam          = float(np.random.beta(mixup_alpha, mixup_alpha))
            idx          = torch.randperm(user_repr_a.size(0), device=device)
            user_repr_m  = lam * user_repr_a + (1 - lam) * user_repr_a[idx]
            target_ids_b = target_ids[idx]
        else:
            user_repr_m  = user_repr_a; lam = 1.0; target_ids_b = target_ids

        # ── 主任务：全量 Softmax CE ──
        logits    = self._score(user_repr_m, item_repr)              # (B, N+1)
        main_loss = self._ce(logits[:, 1:], target_ids - 1)
        if mixup_alpha > 0:
            main_loss = (lam * main_loss +
                (1 - lam) * self._ce(logits[:, 1:], target_ids_b - 1))

        # ── 辅助任务：MIP ──
        aux_loss = torch.tensor(0.0, device=device)
        if self.aux_weight > 0 and mask_targets is not None:
            aux_loss = self._aux_loss(seq_hidden_a, mask_targets)

        # ── 对比学习：NT-Xent（View-A vs View-B）──
        cl_loss = torch.tensor(0.0, device=device)
        if self.cl_weight > 0:
            z_a = self.cl_proj(user_repr_a)   # (B, D)
            z_b = self.cl_proj(user_repr_b)   # (B, D)
            cl_loss = self._nt_xent(z_a, z_b)

        total = main_loss + self.aux_weight * aux_loss + self.cl_weight * cl_loss
        return total


# ══════════════════════════════════════════════════════════
# 5. 训练 & 评估
# ══════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, ift, device, mixup_alpha):
    model.train()
    total, n = 0.0, 0
    for seq, seq_mask, seq_shuf, uf, tgt, mtgt in loader:
        seq      = seq.to(device);      seq_mask = seq_mask.to(device)
        seq_shuf = seq_shuf.to(device); uf       = uf.to(device)
        tgt      = tgt.to(device);      mtgt     = mtgt.to(device)

        optimizer.zero_grad()
        loss = model(seq, seq_mask, seq_shuf, ift, uf, tgt, mtgt, mixup_alpha)
        if torch.isnan(loss) or torch.isinf(loss): continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        bad = any(p.grad is not None and
                  (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                  for p in model.parameters())
        if bad:
            optimizer.zero_grad(); continue

        optimizer.step()
        total += loss.item(); n += 1

    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, ift, device, topk):
    model.eval()
    hits, ndcgs = [], []
    for seq, seq_mask, seq_shuf, uf, tgt, _ in loader:
        seq, uf, tgt = seq.to(device), uf.to(device), tgt.to(device)
        # 评估用原始序列，View-A=View-B=原始序列（无增强）
        logits = model(seq, seq, seq, ift, uf)
        logits[:, 0] = -1e9
        tk = logits.topk(topk, dim=-1).indices
        for i in range(len(tgt)):
            t   = tgt[i].item(); lst = tk[i].tolist()
            if t in lst:
                rank = lst.index(t) + 1
                hits.append(1); ndcgs.append(1.0 / math.log2(rank + 1))
            else:
                hits.append(0); ndcgs.append(0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs))


@torch.no_grad()
def predict(model, loader, ift, device, topk, id2item):
    model.eval(); rows = []
    for seq, uf, uids in loader:
        seq, uf = seq.to(device), uf.to(device)
        logits  = model(seq, seq, seq, ift, uf)
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
                       proc.MASK_ID, mask_prob=cfg.mask_prob,
                       shuffle_prob=cfg.shuffle_prob, mode="train")
    va_ds = RecDataset(val_samps,    cfg.max_seq_len, proc.n_items,
                       proc.MASK_ID, mask_prob=0.0,
                       shuffle_prob=0.0, mode="train")
    te_ds = RecDataset(test_samples, cfg.max_seq_len, proc.n_items,
                       proc.MASK_ID, mask_prob=0.0,
                       shuffle_prob=0.0, mode="test")

    tr_ld = DataLoader(tr_ds, cfg.batch_size, shuffle=True,  num_workers=0)
    va_ld = DataLoader(va_ds, cfg.batch_size, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, cfg.batch_size, shuffle=False, num_workers=0)

    model = ContrastFormerModel(
        cfg, proc.n_items,
        proc.item_feat_dims, proc.item_feat_padidxs,
        proc.user_feat_dims, proc.user_feat_padidxs,
    ).to(cfg.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    ift       = proc.item_feat_tensor
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.lr * 0.01)

    best_ndcg, best_state = 0.0, None
    for ep in range(1, cfg.epochs + 1):
        loss = train_epoch(model, tr_ld, optimizer, ift, cfg.device, cfg.mixup_alpha)
        scheduler.step()
        if ep % 5 == 0 or ep == cfg.epochs:
            hr, ndcg = evaluate(model, va_ld, ift, cfg.device, cfg.topk)
            tau = model.log_temp.clamp(-4.6, 4.6).exp().item()
            print(f"Ep {ep:3d} | Loss {loss:.4f} | "
                  f"HR@{cfg.topk}: {hr:.4f} | NDCG@{cfg.topk}: {ndcg:.4f} | τ={tau:.2f}")
            if ndcg > best_ndcg:
                best_ndcg  = ndcg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  ✓ Best NDCG@{cfg.topk}: {best_ndcg:.4f} — saved")
        else:
            print(f"Ep {ep:3d} | Loss {loss:.4f}")

    if best_state:
        model.load_state_dict({k: v.to(cfg.device) for k, v in best_state.items()})
    print(f"\nFinal best NDCG@{cfg.topk}: {best_ndcg:.4f}")

    rows = predict(model, te_ld, ift, cfg.device, cfg.topk, proc.id2item)
    pd.DataFrame(rows).to_csv(cfg.output_path, index=False)
    print(f"Saved → {cfg.output_path}")
    print(pd.DataFrame(rows).head(3).to_string())


if __name__ == "__main__":
    main()
