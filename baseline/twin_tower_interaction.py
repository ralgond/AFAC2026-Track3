"""
Twin Tower with Interaction (TTI)
==================================

架构概览
--------
在原 DIN+DeepFM 双塔基础上，增加 Cross-Tower Interaction 层：

    User Tower  ──→  user_repr  (B, D)  ─┐
                                           ├──→  Interaction Layer ──→ final scores (B, N)
    Item Tower  ──→  item_repr  (N, D)  ─┘

Interaction Layer 提供三种模式（cfg.interaction_mode 控制）：
  1. "dot"       : 原始点积，无交互参数（基线）
  2. "mlp"       : 拼接后过 MLP，适合 top-k 推理（训练时对候选集采样）
  3. "attention" : User repr 作为 query，Item repr 作为 key/value，
                   做 cross-attention → 增强 user 向量后再点积
                   推理时仍兼容全量 item（批量 attention）

训练策略
--------
- dot / attention 模式：全量 Softmax CE（同原始代码）
- mlp 模式：负采样 BPR Loss（全量 MLP 开销过大，改用正样本 + K 个随机负样本）

文件结构
--------
DataProcessor / RecDataset / build_mlp / fm_second_order   ← 与原代码完全一致
UserInterestPooling / UserTower / ItemTower                 ← 与原代码完全一致
CrossTowerInteraction                                        ← 新增
TwinTowerWithInteraction                                     ← 新增（替换 DINDeepFME2E）
train_epoch / evaluate / predict / main                     ← 微调以支持新模型

验证指标：NDCG@10
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
    train_path  = "../data/A2-Rec/train.csv"
    test_path   = "../data/A2-Rec/test.csv"
    user_path   = "../data/A2-Rec/user.csv"
    item_path   = "../data/A2-Rec/item.csv"
    output_path = "submission_tti.csv"

    max_seq_len = 50
    emb_dim     = 32
    repr_dim    = 128          # user/item tower 输出维度
    n_heads     = 4
    mlp_dims    = [256, 128]
    dropout     = 0.2

    # ── Interaction 相关配置 ──
    # 可选: "dot" | "mlp" | "attention"
    interaction_mode   = "attention"

    # mlp 模式：BPR 负采样数
    n_neg_samples      = 50

    # attention 模式：cross-attention 头数与层数
    cross_attn_heads   = 4
    cross_attn_layers  = 2

    # mlp 模式：interaction MLP 结构（输入 = 2*repr_dim）
    interaction_mlp_dims = [256, 128]

    epochs      = 30
    batch_size  = 256
    lr          = 1e-3
    weight_decay= 1e-5
    seed        = 42
    topk        = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)


# ══════════════════════════════════════════════════════════
# 1. 数据处理（与原代码一致）
# ══════════════════════════════════════════════════════════
class DataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.item2id  = {}
        self.id2item  = {}
        self.iid2feat = {}
        self.uid2feat = {}
        self.item_feat_cols = ["i_cat_01","i_cat_02","i_cat_03","i_bucket_01"]
        self.user_feat_cols = [f"u_cat_0{i}" for i in range(1, 9)]
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
        print(f"  Total items : {self.n_items}")

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
                "uid": uid,
                "seq": self._weight_seq(seq, cnts),
                "target": target,
                "user_feat": uid_feat_map.get(uid, pad_ufeat),
            })
        for _, row in test_df.iterrows():
            uid  = row["uid"]
            seq  = self._parse_seq(row["item_seq_dedup"])
            cnts = self._parse_counts(row.get("item_seq_counts", ""))
            test_samples.append({
                "uid": uid,
                "seq": self._weight_seq(seq, cnts),
                "user_feat": uid_feat_map.get(uid, pad_ufeat),
            })
        print(f"  Train: {len(train_samples)} | Test: {len(test_samples)}")

        ift = torch.zeros(self.n_items + 1, 4, dtype=torch.long)
        for iid, feats in self.iid2feat.items():
            ift[iid] = torch.tensor(feats)
        self.item_feat_tensor = ift

        return train_samples, test_samples


# ══════════════════════════════════════════════════════════
# 2. Dataset（与原代码一致）
# ══════════════════════════════════════════════════════════
def pad_seq(seq, max_len):
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq

class RecDataset(Dataset):
    def __init__(self, samples, max_len, mode="train"):
        self.samples = samples
        self.max_len = max_len
        self.mode    = mode

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        seq = torch.tensor(pad_seq(s["seq"], self.max_len), dtype=torch.long)
        uf  = torch.tensor(s["user_feat"], dtype=torch.long)
        if self.mode == "train":
            return seq, uf, torch.tensor(s["target"], dtype=torch.long)
        return seq, uf, s["uid"]


# ══════════════════════════════════════════════════════════
# 3. 模型组件（原有部分与原代码一致）
# ══════════════════════════════════════════════════════════

# ── 3a. 共用 MLP ──
def build_mlp(in_dim, hidden_dims, out_dim, dropout):
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ── 3b. FM 二阶交叉 ──
def fm_second_order(emb_list):
    stacked = torch.stack(emb_list, dim=1)
    sum_sq  = stacked.sum(dim=1) ** 2
    sq_sum  = (stacked ** 2).sum(dim=1)
    return 0.5 * (sum_sq - sq_sum)


# ── 3c. 用户侧兴趣池化（DIN 风格）──
class UserInterestPooling(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 2, d)
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, user_vec, seq_emb, pad_mask):
        q   = self.norm1(user_vec).unsqueeze(1)
        out, _ = self.attn(q, seq_emb, seq_emb,
                           key_padding_mask=pad_mask,
                           need_weights=False)
        q   = self.drop(out) + user_vec.unsqueeze(1)
        q   = self.drop(self.ff(self.norm2(q))) + q
        return q.squeeze(1)


# ── 3d. User Tower ──
class UserTower(nn.Module):
    def __init__(self, cfg, n_items,
                 user_feat_dims, user_feat_padidxs,
                 item_id_emb_shared):
        super().__init__()
        E = cfg.emb_dim
        D = cfg.repr_dim

        self.u_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(user_feat_dims, user_feat_padidxs)
        ])
        n_u = len(user_feat_dims)

        self.u_feat_mlp = build_mlp(E * n_u, [D], D, cfg.dropout)
        self.item_id_emb = item_id_emb_shared
        self.pos_emb = nn.Embedding(cfg.max_seq_len + 1, E)
        self.seq_proj = nn.Linear(E, D)
        self.din = UserInterestPooling(D, cfg.n_heads, cfg.dropout)
        self.out_mlp = build_mlp(D * 3 + E, cfg.mlp_dims, D, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, seq_ids, user_feats):
        B, L   = seq_ids.shape
        device = seq_ids.device

        u_emb_list = [e(user_feats[:, i]) for i, e in enumerate(self.u_embs)]
        user_fm = fm_second_order(u_emb_list)
        u_cat = torch.cat(u_emb_list, dim=-1)
        user_feat_vec = self.u_feat_mlp(u_cat)

        pos   = torch.arange(1, L+1, device=device).unsqueeze(0)
        seq_e = self.drop(self.item_id_emb(seq_ids) + self.pos_emb(pos))
        seq_d = self.drop(F.gelu(self.seq_proj(seq_e)))

        pad_mask = (seq_ids == 0)
        valid    = (~pad_mask).float().unsqueeze(-1)
        hist_vec = (seq_d * valid).sum(1) / valid.sum(1).clamp(min=1)

        din_vec = self.din(user_feat_vec, seq_d, pad_mask)

        concat = torch.cat([user_feat_vec, hist_vec, din_vec, user_fm], dim=-1)
        return self.out_mlp(concat)


# ── 3e. Item Tower ──
class ItemTower(nn.Module):
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 item_id_emb_shared):
        super().__init__()
        E = cfg.emb_dim
        D = cfg.repr_dim

        self.item_id_emb = item_id_emb_shared
        self.i_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)
        ])
        n_i = len(item_feat_dims)
        self.out_mlp = build_mlp(E + E * n_i + E, cfg.mlp_dims, D, cfg.dropout)

    def forward(self, item_ids, item_feats):
        id_vec  = self.item_id_emb(item_ids)
        fi_list = [e(item_feats[..., j]) for j, e in enumerate(self.i_embs)]
        item_fm = fm_second_order(fi_list)
        fi_cat  = torch.cat(fi_list, dim=-1)
        concat  = torch.cat([id_vec, fi_cat, item_fm], dim=-1)
        return self.out_mlp(concat)


# ══════════════════════════════════════════════════════════
# 3f. Cross-Tower Interaction Layer  ← 核心新增
# ══════════════════════════════════════════════════════════
class CrossTowerInteraction(nn.Module):
    """
    三种 interaction 模式：

    "dot" (baseline)
    ────────────────
      score = user_repr · item_repr^T
      参数量：0（纯点积，保留作对比）

    "attention" (推荐模式)
    ──────────────────────
      思路：让 user_repr 作为 Query，item_repr 作为 Key/Value，
            做多层 cross-attention，增强后的 user_repr 再与 item_repr 点积。

      训练：对全量 item 做矩阵 attention（批量高效）
      推理：item_repr 预先缓存，attention 仍为 O(B·N·D)

      公式（单层）：
        enhanced_user = LayerNorm(user + Attention(user, items, items))
        enhanced_user = LayerNorm(enhanced_user + FFN(enhanced_user))
        score = enhanced_user · item_repr^T

      多头 cross-attention（n_layers 层堆叠，残差 + LayerNorm）

    "mlp" (精排模式)
    ─────────────────
      思路：拼接 [user_repr; item_repr] → MLP → scalar score
            因为每对 (u, i) 需要单独过 MLP，开销为 O(B·N)，
            训练时用负采样（正样本 + K 个随机负样本）。

      推理：仍用全量打分，但 MLP 逐对计算，适合候选集规模适中（< 1k）的场景。
    """
    def __init__(self, cfg):
        super().__init__()
        D = cfg.repr_dim
        self.mode = cfg.interaction_mode

        if self.mode == "attention":
            # 多层 cross-attention：user 作为 query，item 作为 key/value
            self.cross_layers = nn.ModuleList([
                _CrossAttentionBlock(D, cfg.cross_attn_heads, cfg.dropout)
                for _ in range(cfg.cross_attn_layers)
            ])

        elif self.mode == "mlp":
            # 拼接后 MLP → scalar（最后输出 1 维）
            self.interaction_mlp = build_mlp(
                2 * D, cfg.interaction_mlp_dims, 1, cfg.dropout
            )

        # "dot" 模式无额外参数

    # ── forward (训练/全量推理) ──
    def score_full(self, user_repr, item_repr):
        """
        user_repr : (B, D)
        item_repr : (N, D)
        return    : (B, N)
        """
        if self.mode == "dot":
            return user_repr @ item_repr.T                     # (B, N)

        elif self.mode == "attention":
            # item_repr 扩展为 (B, N, D)，让每个用户独立做 cross-attention
            B, D = user_repr.shape
            N    = item_repr.shape[0]
            items_exp = item_repr.unsqueeze(0).expand(B, N, D) # (B, N, D)

            # user_repr 作为单 token query：(B, 1, D)
            u = user_repr.unsqueeze(1)                         # (B, 1, D)
            for layer in self.cross_layers:
                u = layer(u, items_exp)                        # (B, 1, D)
            enhanced = u.squeeze(1)                            # (B, D)
            return enhanced @ item_repr.T                      # (B, N)

        elif self.mode == "mlp":
            # 全量模式：(B, N, 2D) → MLP → (B, N)
            B, D = user_repr.shape
            N    = item_repr.shape[0]
            u_exp = user_repr.unsqueeze(1).expand(B, N, D)    # (B, N, D)
            i_exp = item_repr.unsqueeze(0).expand(B, N, D)    # (B, N, D)
            pair  = torch.cat([u_exp, i_exp], dim=-1)         # (B, N, 2D)
            return self.interaction_mlp(pair).squeeze(-1)     # (B, N)

    # ── forward (训练时 mlp 模式负采样) ──
    def score_sampled(self, user_repr, pos_item_repr, neg_item_repr):
        """
        仅 mlp 模式使用，避免全量 O(B*N) MLP 展开。
        user_repr      : (B, D)
        pos_item_repr  : (B, D)
        neg_item_repr  : (B, K, D)
        return         : pos_scores (B,), neg_scores (B, K)
        """
        B, D = user_repr.shape
        K    = neg_item_repr.shape[1]

        # 正样本
        pos_pair = torch.cat([user_repr, pos_item_repr], dim=-1)   # (B, 2D)
        pos_scores = self.interaction_mlp(pos_pair).squeeze(-1)    # (B,)

        # 负样本
        u_exp  = user_repr.unsqueeze(1).expand(B, K, D)           # (B, K, D)
        n_pair = torch.cat([u_exp, neg_item_repr], dim=-1)        # (B, K, 2D)
        neg_scores = self.interaction_mlp(n_pair).squeeze(-1)     # (B, K)

        return pos_scores, neg_scores


class _CrossAttentionBlock(nn.Module):
    """
    单层 cross-attention：
      query = user token(s)   (B, Lq, D)
      key   = item tokens     (B, N,  D)
    输出：增强后的 query (B, Lq, D)
    """
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads,
                                           dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, query, kv):
        """
        query : (B, Lq, D)
        kv    : (B, N,  D)
        """
        # Cross-attention（Pre-LN）
        q_norm = self.norm1(query)
        attn_out, _ = self.attn(q_norm, kv, kv, need_weights=False)
        query = query + self.drop(attn_out)                    # 残差

        # FFN（Pre-LN）
        query = query + self.drop(self.ff(self.norm2(query)))  # 残差
        return query


# ══════════════════════════════════════════════════════════
# 3g. 整体模型：Twin Tower with Interaction
# ══════════════════════════════════════════════════════════
class TwinTowerWithInteraction(nn.Module):
    """
    User Tower + Item Tower + Cross-Tower Interaction Layer

    训练时：
      - dot / attention 模式：全量 Softmax CE
      - mlp 模式：BPR Loss（负采样）

    推理时：
      - 所有模式都支持全量打分（B × N 矩阵）
      - attention 模式：item_repr 可离线缓存，但每次推理仍需 cross-attention
      - dot / mlp 模式：item_repr 完全离线缓存，推理效率最高
    """
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 user_feat_dims, user_feat_padidxs):
        super().__init__()
        self.cfg     = cfg
        self.n_items = n_items

        # 共享 item id embedding（两塔对齐语义空间）
        self.item_id_emb = nn.Embedding(n_items + 1, cfg.emb_dim, padding_idx=0)

        self.user_tower = UserTower(
            cfg, n_items, user_feat_dims, user_feat_padidxs,
            self.item_id_emb)

        self.item_tower = ItemTower(
            cfg, n_items, item_feat_dims, item_feat_padidxs,
            self.item_id_emb)

        self.interaction = CrossTowerInteraction(cfg)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _get_all_item_repr(self, ift, device):
        """计算/返回全量 item 表示 (N+1, D)"""
        all_ids   = torch.arange(self.n_items + 1, device=device)
        all_feats = ift.to(device)
        return self.item_tower(all_ids, all_feats)              # (N+1, D)

    def forward(self, seq_ids, ift, user_feats, target_ids=None):
        """
        seq_ids    : (B, L)
        ift        : (N+1, 4)  全量 item 特征查找表
        user_feats : (B, 8)
        target_ids : (B,) long  训练时传入，推理为 None
        """
        device    = seq_ids.device
        user_repr = self.user_tower(seq_ids, user_feats)        # (B, D)
        item_repr = self._get_all_item_repr(ift, device)        # (N+1, D)

        if target_ids is not None:
            # ── 训练 ──
            if self.cfg.interaction_mode == "mlp":
                return self._train_mlp(user_repr, item_repr, target_ids)
            else:
                # dot / attention：全量 softmax CE
                logits = self.interaction.score_full(user_repr, item_repr)  # (B, N+1)
                return F.cross_entropy(logits[:, 1:], target_ids - 1)

        else:
            # ── 推理 ──
            logits = self.interaction.score_full(user_repr, item_repr)      # (B, N+1)
            return logits

    def _train_mlp(self, user_repr, item_repr, target_ids):
        """
        mlp 模式训练：BPR Loss + 随机负采样
          正样本：target item
          负样本：随机采 n_neg_samples 个不等于 target 的 item
        """
        B    = user_repr.shape[0]
        K    = self.cfg.n_neg_samples
        device = user_repr.device

        # 正样本 repr (B, D)
        pos_repr = item_repr[target_ids]                       # (B, D)

        # 负采样：对每个样本随机采 K 个 item_id（1 ~ n_items，不含 target）
        # 简化实现：直接随机采样，不严格排除 target（概率极低，可接受）
        neg_ids  = torch.randint(1, self.n_items + 1,
                                 (B, K), device=device)        # (B, K)
        neg_repr = item_repr[neg_ids]                          # (B, K, D)

        pos_scores, neg_scores = self.interaction.score_sampled(
            user_repr, pos_repr, neg_repr)                     # (B,), (B, K)

        # BPR Loss: -log(σ(pos - neg))，对 K 个负样本取均值
        diff = pos_scores.unsqueeze(1) - neg_scores            # (B, K)
        loss = -F.logsigmoid(diff).mean()
        return loss


# ══════════════════════════════════════════════════════════
# 4. 训练 & 评估
# ══════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, ift, device):
    model.train()
    total = 0
    for seq, uf, target in loader:
        seq, uf, target = seq.to(device), uf.to(device), target.to(device)
        optimizer.zero_grad()
        loss = model(seq, ift, uf, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, ift, device, topk):
    model.eval()
    hits, ndcgs = [], []
    for seq, uf, target in loader:
        seq, uf, target = seq.to(device), uf.to(device), target.to(device)
        logits = model(seq, ift, uf)
        logits[:, 0] = -1e9                                    # 屏蔽 PAD
        tk = logits.topk(topk, dim=-1).indices
        for i in range(len(target)):
            t   = target[i].item()
            lst = tk[i].tolist()
            if t in lst:
                rank = lst.index(t) + 1
                hits.append(1); ndcgs.append(1 / math.log2(rank + 1))
            else:
                hits.append(0); ndcgs.append(0)
    return float(np.mean(hits)), float(np.mean(ndcgs))


@torch.no_grad()
def predict(model, loader, ift, device, topk, id2item):
    model.eval()
    rows = []
    for seq, uf, uids in loader:
        seq, uf = seq.to(device), uf.to(device)
        logits  = model(seq, ift, uf)
        logits[:, 0] = -1e9
        tk = logits.topk(topk, dim=-1).indices.cpu().tolist()
        for i, uid in enumerate(uids):
            items = [id2item[iid] for iid in tk[i] if iid in id2item]
            rows.append({"uid": uid, "predicted_items": ",".join(items)})
    return rows


# ══════════════════════════════════════════════════════════
# 5. 主函数
# ══════════════════════════════════════════════════════════
def main():
    print(f"Device: {cfg.device}")
    print(f"Interaction mode: {cfg.interaction_mode}\n")

    proc = DataProcessor(cfg)
    train_samples, test_samples = proc.load_and_build()

    all_uids  = sorted(set(s["uid"] for s in train_samples))
    n_val_u   = max(1, int(len(all_uids) * 0.1))
    val_uids  = set(all_uids[-n_val_u:])
    val_samps = [s for s in train_samples if     s["uid"] in val_uids]
    tr_samps  = [s for s in train_samples if not s["uid"] in val_uids]
    print(f"  Val users={len(val_uids)} | Val={len(val_samps)} | Train={len(tr_samps)}\n")

    tr_ds  = RecDataset(tr_samps,     cfg.max_seq_len, mode="train")
    val_ds = RecDataset(val_samps,    cfg.max_seq_len, mode="train")
    te_ds  = RecDataset(test_samples, cfg.max_seq_len, mode="test")
    tr_ld  = DataLoader(tr_ds,  cfg.batch_size, shuffle=True,  num_workers=0)
    val_ld = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=0)
    te_ld  = DataLoader(te_ds,  cfg.batch_size, shuffle=False, num_workers=0)

    model = TwinTowerWithInteraction(
        cfg, proc.n_items,
        proc.item_feat_dims, proc.item_feat_padidxs,
        proc.user_feat_dims, proc.user_feat_padidxs,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"  User Tower params : "
          f"{sum(p.numel() for p in model.user_tower.parameters()):,}")
    print(f"  Item Tower params : "
          f"{sum(p.numel() for p in model.item_tower.parameters()):,}")
    print(f"  Interaction params: "
          f"{sum(p.numel() for p in model.interaction.parameters()):,}\n")

    ift       = proc.item_feat_tensor
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.lr * 0.01)

    best_ndcg, best_state = 0.0, None

    for ep in range(1, cfg.epochs + 1):
        loss = train_epoch(model, tr_ld, optimizer, ift, cfg.device)
        scheduler.step()
        if ep % 5 == 0 or ep == cfg.epochs:
            hr, ndcg = evaluate(model, val_ld, ift, cfg.device, cfg.topk)
            print(f"Ep {ep:3d} | Loss {loss:.4f} | "
                  f"HR@{cfg.topk}: {hr:.4f} | NDCG@{cfg.topk}: {ndcg:.4f}")
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
