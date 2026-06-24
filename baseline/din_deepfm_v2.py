"""
DIN+DeepFM v2  ——  端到端，无召回阶段
======================================================

相比 v1 的改进点
----------------
1. 序列编码升级
   mean pooling + 单层 cross-attention
   → 多层因果 Transformer（SASRec 风格）
   捕捉序列时序依赖，每层叠加位置感知的上下文表示

2. 打分层浅交叉（SENet 风格 feature-wise gating）
   两塔点积之前，用 user_repr 对 item_repr 做 element-wise 门控，
   引入 user × item 的细粒度交叉而不破坏两塔可缓存的特性

3. 多任务辅助损失
   主任务：全量 Softmax CE（next item prediction）
   辅助任务：序列中间位置的 masked item prediction（类 BERT4Rec）
   增加序列内部的监督信号，缓解数据稀疏

4. Item Tower 加深
   FM 二阶 + 更深的 MLP（含残差连接）

5. 训练技巧
   - Label Smoothing（防止过拟合）
   - Mixup on embeddings（序列表示层面的数据增强）
   - 温度系数缩放打分

架构总览
--------
User Tower
  ├─ 用户特征8域 → FM二阶 + MLP → user_feat_vec (D)
  ├─ 序列 → SASRec Transformer（多层因果自注意力）→ seq_repr (D)
  └─ concat[user_feat_vec; seq_repr] → MLP → user_repr (D)

Item Tower
  ├─ item_id emb (E)
  ├─ 4个item特征域 → FM二阶 + 残差MLP → item_feat_vec (D)
  └─ concat → MLP → item_repr (D)

打分
  gated_item = item_repr ⊙ sigmoid(W · user_repr)   ← 浅层交叉
  score = user_repr · gated_item^T / τ               ← 温度缩放
  训练：全量 Softmax CE（label smoothing）+ 辅助 Masked Item 损失
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
    output_path = "submission_v2.csv"

    max_seq_len  = 50
    emb_dim      = 32          # 每个特征域 embedding 维度
    repr_dim     = 128         # user/item tower 输出维度
    # SASRec 序列编码
    n_heads      = 4
    n_layers     = 3           # Transformer 层数（v1=1，v2=3）
    # MLP
    user_mlp_dims = [256, 128]
    item_mlp_dims = [256, 128]
    dropout      = 0.2

    # 训练
    epochs       = 50          # 更多 epoch（cosine 衰减兜底）
    batch_size   = 256
    lr           = 1e-3
    weight_decay = 1e-5
    label_smooth = 0.0         # label smoothing（小数据集关闭）
    aux_weight   = 0.05        # 辅助损失权重（小数据适当降低）
    mask_prob    = 0.2         # 序列 mask 概率（辅助任务）
    temperature  = 1.0         # 打分温度系数（0.07 过小会梯度爆炸）
    mixup_alpha  = 0.1         # Mixup 强度（适当降低）

    seed         = 42
    topk         = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)


# ══════════════════════════════════════════════════════════
# 1. 数据处理（与原系统保持一致）
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

        # item 词表
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

        # item 特征
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

        # user 特征
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
        all_users = sorted(set(train_df["uid"].str.strip()) |
                           set(test_df["uid"].str.strip()))

        # 样本构建
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
# 2. Dataset
# ══════════════════════════════════════════════════════════
def pad_seq(seq, max_len):
    seq = seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


class RecDataset(Dataset):
    """
    训练模式：同时生成主任务 target 和辅助任务 masked_seq / mask_targets。
    masked_seq  : 随机将序列中 mask_prob 比例的位置替换为 MASK_ID（= n_items+1）
    mask_targets: 被替换位置的原始 item id（其余位置为 0，loss 计算时忽略）
    """
    def __init__(self, samples, max_len, n_items, mask_prob=0.0, mode="train"):
        self.samples   = samples
        self.max_len   = max_len
        self.n_items   = n_items
        self.mask_prob = mask_prob
        self.mode      = mode
        self.MASK_ID   = n_items + 1   # 专用 MASK token

    def __len__(self): return len(self.samples)

    def _mask_seq(self, seq_ids):
        """对非 PAD 位置按 mask_prob 随机替换为 MASK_ID，返回 masked_seq 和 targets"""
        masked = seq_ids.copy()
        targets = [0] * len(seq_ids)
        for i, sid in enumerate(seq_ids):
            if sid != 0 and random.random() < self.mask_prob:
                targets[i] = sid
                masked[i]  = self.MASK_ID
        return masked, targets

    def __getitem__(self, idx):
        s   = self.samples[idx]
        seq = pad_seq(s["seq"], self.max_len)
        uf  = torch.tensor(s["user_feat"], dtype=torch.long)
        if self.mode == "train":
            masked_seq, mask_tgt = self._mask_seq(seq)
            return (
                torch.tensor(seq,        dtype=torch.long),
                torch.tensor(masked_seq, dtype=torch.long),
                uf,
                torch.tensor(s["target"],  dtype=torch.long),
                torch.tensor(mask_tgt,     dtype=torch.long),
            )
        return torch.tensor(seq, dtype=torch.long), uf, s["uid"]


# ══════════════════════════════════════════════════════════
# 3. 模型组件
# ══════════════════════════════════════════════════════════

def build_mlp(in_dim, hidden_dims, out_dim, dropout, residual=False):
    """MLP，不使用 LayerNorm（避免在全量 item 前向时抹平 item 间差异）"""
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    mlp = nn.Sequential(*layers)
    if residual and in_dim == out_dim:
        class ResidualMLP(nn.Module):
            def __init__(self, mlp): super().__init__(); self.mlp = mlp
            def forward(self, x): return self.mlp(x) + x
        return ResidualMLP(mlp)
    return mlp


def fm_second_order(emb_list):
    """emb_list: list of (..., E) → (..., E)"""
    stacked = torch.stack(emb_list, dim=-2)      # (..., F, E)
    sum_sq  = stacked.sum(dim=-2) ** 2
    sq_sum  = (stacked ** 2).sum(dim=-2)
    return 0.5 * (sum_sq - sq_sum)


class SASRecBlock(nn.Module):
    def __init__(self, d, n_heads, dropout):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(d * 4, d))
        self.n1    = nn.LayerNorm(d)
        self.n2    = nn.LayerNorm(d)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None, pad_mask=None):
        r = x; x = self.n1(x)
        x, _ = self.attn(x, x, x, attn_mask=causal_mask,
                         key_padding_mask=pad_mask, need_weights=False)
        x = torch.nan_to_num(x, nan=0.0)   # 全PAD行 softmax→NaN 保护
        x = self.drop(x) + r
        r = x; x = self.n2(x)
        return self.drop(self.ff(x)) + r


class UserTower(nn.Module):
    """
    改进：
    - SASRec 多层因果 Transformer 替代 mean pooling
    - 最后一个有效位置的输出作为序列表示（而非 mean pooling）
    - user_feat_vec 与 seq_repr concat 后接 MLP
    """
    def __init__(self, cfg, n_items,
                 user_feat_dims, user_feat_padidxs,
                 item_id_emb_shared):
        super().__init__()
        E = cfg.emb_dim
        D = cfg.repr_dim
        n_u = len(user_feat_dims)

        # 用户特征域 embedding
        self.u_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(user_feat_dims, user_feat_padidxs)
        ])
        self.u_feat_mlp = build_mlp(E * n_u, [D], D, cfg.dropout)

        # 序列编码
        self.item_id_emb = item_id_emb_shared
        self.pos_emb     = nn.Embedding(cfg.max_seq_len + 1, D)
        # item_id(E) → D，再输入 Transformer
        self.seq_in_proj = nn.Linear(E, D)
        self.layers = nn.ModuleList([
            SASRecBlock(D, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.seq_norm = nn.LayerNorm(D)

        # 最终融合
        # concat[user_feat_vec(D); seq_repr(D); user_fm(E)] → MLP → D
        self.out_mlp = build_mlp(D * 2 + E, cfg.user_mlp_dims, D, cfg.dropout)
        self.drop    = nn.Dropout(cfg.dropout)

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()

    def encode_seq(self, seq_ids):
        """seq_ids: (B, L) → (B, D) 取最后非 PAD 位置的 Transformer 输出"""
        B, L   = seq_ids.shape
        device = seq_ids.device
        pos    = torch.arange(1, L + 1, device=device).unsqueeze(0)
        seq_proj = F.gelu(self.seq_in_proj(self.item_id_emb(seq_ids)))
        seq_proj = torch.clamp(seq_proj, -10.0, 10.0)   # 防止投影层输出爆炸
        x = self.drop(seq_proj + self.pos_emb(pos))           # (B, L, D)
        pad_mask   = (seq_ids == 0)
        causal     = self._causal_mask(L, device)
        for layer in self.layers:
            x = layer(x, causal_mask=causal, pad_mask=pad_mask)
        x = self.seq_norm(x)                                  # (B, L, D)
        # 取最后一个非 PAD 位置（左 padding，最后一列是最新位置）
        # 若整条序列全 PAD 则输出 zero vec（nan_to_num 兜底）
        out = x[:, -1, :]
        return torch.nan_to_num(out, nan=0.0)                 # (B, D)

    def forward(self, seq_ids, user_feats):
        # 用户特征
        u_emb_list    = [e(user_feats[:, i]) for i, e in enumerate(self.u_embs)]
        user_fm       = fm_second_order(u_emb_list)           # (B, E)
        user_feat_vec = self.u_feat_mlp(torch.cat(u_emb_list, dim=-1))  # (B, D)
        # 序列表示
        seq_repr = self.encode_seq(seq_ids)                   # (B, D)
        # 融合
        concat = torch.cat([user_feat_vec, seq_repr, user_fm], dim=-1)
        out = self.out_mlp(concat)
        return torch.nan_to_num(out, nan=0.0)                 # (B, D)


class ItemTower(nn.Module):
    """
    改进：FM 二阶 + 残差 MLP，输出归一化
    """
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 item_id_emb_shared):
        super().__init__()
        E   = cfg.emb_dim
        D   = cfg.repr_dim
        n_i = len(item_feat_dims)

        self.item_id_emb = item_id_emb_shared
        self.i_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)
        ])
        # item_id(E) + item_feat concat(n_i*E) + item_fm(E) → MLP → D
        self.out_mlp = build_mlp(E + E * n_i + E, cfg.item_mlp_dims, D, cfg.dropout)

    def forward(self, item_ids, item_feats):
        id_vec  = self.item_id_emb(item_ids)
        fi_list = [e(item_feats[..., j]) for j, e in enumerate(self.i_embs)]
        item_fm = fm_second_order(fi_list)
        fi_cat  = torch.cat(fi_list, dim=-1)
        out = self.out_mlp(torch.cat([id_vec, fi_cat, item_fm], dim=-1))
        return out


class DINDeepFMv2(nn.Module):
    """
    改进点集成：
    1. SASRec 多层序列编码
    2. 打分层浅交叉（user 引导的 item gating）
    3. 温度系数缩放
    4. 辅助 Masked Item Prediction 头
    """
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 user_feat_dims, user_feat_padidxs):
        super().__init__()
        D = cfg.repr_dim
        E = cfg.emb_dim
        self.n_items     = n_items
        self.label_smooth= cfg.label_smooth
        self.aux_weight  = cfg.aux_weight
        self.MASK_ID     = n_items + 1
        # 可学习温度系数，初始化为 log(20)≈3.0（τ≈0.05，余弦×20让分布更尖锐）
        self.log_temp    = nn.Parameter(torch.tensor(3.0))

        # User Tower 和 Item Tower 各用独立的 item id embedding
        # 共享会导致两侧梯度方向冲突，item_repr 退化到接近 0
        self.user_item_emb = nn.Embedding(n_items + 2, E, padding_idx=0)
        self.item_item_emb = nn.Embedding(n_items + 2, E, padding_idx=0)

        self.user_tower = UserTower(
            cfg, n_items, user_feat_dims, user_feat_padidxs,
            self.user_item_emb)
        self.item_tower = ItemTower(
            cfg, n_items, item_feat_dims, item_feat_padidxs,
            self.item_item_emb)

        # 浅层交叉：user_repr → 生成 item_repr 的逐元素门控权重
        self.cross_gate = nn.Sequential(
            nn.Linear(D, D), nn.Sigmoid())

        # 辅助任务：masked item prediction 头
        # 输入：被 mask 位置的 Transformer 输出 (D) → 预测 item id
        self.aux_head = nn.Linear(D, n_items + 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _smooth_ce(self, logits, targets, n_classes):
        """Label Smoothing Cross Entropy，含数值稳定保护"""
        logits   = torch.clamp(logits, -50.0, 50.0)   # 防止 softmax 上溢
        log_prob = F.log_softmax(logits, dim=-1)
        if self.label_smooth == 0.0:
            return F.nll_loss(log_prob, targets)
        smooth        = self.label_smooth / n_classes
        one_hot       = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
        smooth_target = one_hot * (1 - self.label_smooth) + smooth
        return -(smooth_target * log_prob).sum(dim=-1).mean()

    def _aux_loss(self, masked_seq_ids, mask_targets, ift):
        """
        对 masked_seq 中被 mask 的位置，用 User Tower 序列编码器输出的
        对应位置 hidden state 预测原始 item id。
        """
        device = masked_seq_ids.device
        B, L   = masked_seq_ids.shape

        # 复用 User Tower 的序列编码器（不重新计算）
        tower  = self.user_tower
        pos    = torch.arange(1, L + 1, device=device).unsqueeze(0)
        x = tower.drop(
            F.gelu(tower.seq_in_proj(tower.item_id_emb(masked_seq_ids)))
            + tower.pos_emb(pos))
        pad_mask = (masked_seq_ids == 0)
        causal   = tower._causal_mask(L, device)
        for layer in tower.layers:
            x = layer(x, causal_mask=causal, pad_mask=pad_mask)
        x = tower.seq_norm(x)   # (B, L, D)

        # 只对被 mask 的位置计算 loss
        mask_pos = (mask_targets != 0)    # (B, L) True=被mask
        if not mask_pos.any():
            return torch.tensor(0.0, device=device)

        hidden   = x[mask_pos]            # (M, D)
        tgt      = mask_targets[mask_pos] # (M,)
        logits   = self.aux_head(hidden)  # (M, N+2)
        logits   = torch.clamp(logits, -50.0, 50.0)
        return F.cross_entropy(logits, tgt)

    def get_all_item_repr(self, ift, device):
        """预计算全量 item repr，推理时可缓存"""
        all_ids   = torch.arange(self.n_items + 1, device=device)
        all_feats = ift.to(device)
        return self.item_tower(all_ids, all_feats)   # (N+1, D)

    def score(self, user_repr, item_repr):
        """
        L2 归一化余弦相似度 + 可学习温度系数打分。
        两塔输出 scale 差异大时点积会爆炸，归一化是标准解法。
        gate 先对 user_repr 做逐元素加权，再归一化，
        既保留浅层交叉语义，又保证数值稳定。

        user_repr : (B, D)
        item_repr : (N, D)
        return    : (B, N)
        """
        gate       = self.cross_gate(user_repr)                # (B, D)
        gated_user = user_repr * gate                          # (B, D)
        # L2 归一化，让两侧在单位球面上打分
        u_norm = F.normalize(gated_user, p=2, dim=-1, eps=1e-8) # (B, D)
        i_norm = F.normalize(item_repr,  p=2, dim=-1, eps=1e-8) # (N, D)
        scores = u_norm @ i_norm.T                             # (B, N)
        # 可学习温度（初始化为 log(1/0.07)≈2.66，即 τ≈0.07）
        # log_temp 钳位：τ 范围约 [0.01, 100]，即 log_temp ∈ [-4.6, 4.6]
        log_temp = torch.clamp(self.log_temp, -4.6, 4.6)
        return scores * log_temp.exp()

    def forward(self, seq_ids, masked_seq_ids, ift, user_feats,
                target_ids=None, mask_targets=None, mixup_alpha=0.0):
        device = seq_ids.device

        # User Tower（用原始序列，不用 masked）
        user_repr = self.user_tower(seq_ids, user_feats)      # (B, D)

        # Mixup：对 user_repr 做插值增强（仅训练时）
        if mixup_alpha > 0 and self.training:
            lam   = np.random.beta(mixup_alpha, mixup_alpha)
            idx   = torch.randperm(user_repr.size(0), device=device)
            user_repr = lam * user_repr + (1 - lam) * user_repr[idx]
            if target_ids is not None:
                target_ids_b = target_ids[idx]

        # Item Tower（全量）
        item_repr = self.get_all_item_repr(ift, device)       # (N+1, D)

        # 全量打分
        logits = self.score(user_repr, item_repr)             # (B, N+1)

        if target_ids is None:
            return logits

        # 主任务 loss（label smoothing CE，跳过 PAD index 0）
        main_loss = self._smooth_ce(
            logits[:, 1:], target_ids - 1, self.n_items)

        # Mixup 第二项
        if mixup_alpha > 0 and self.training:
            main_loss = (main_loss +
                self._smooth_ce(logits[:, 1:], target_ids_b - 1, self.n_items)
                         * (1 - lam) / lam) * lam

        # 辅助任务 loss
        if mask_targets is not None and self.aux_weight > 0:
            aux_loss  = self._aux_loss(masked_seq_ids, mask_targets, ift)
            total_loss = main_loss + self.aux_weight * aux_loss
        else:
            total_loss = main_loss

        return total_loss


# ══════════════════════════════════════════════════════════
# 4. 训练 & 评估
# ══════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, ift, device, mixup_alpha):
    model.train()
    total = 0
    for seq, masked_seq, uf, target, mask_tgt in loader:
        seq        = seq.to(device)
        masked_seq = masked_seq.to(device)
        uf         = uf.to(device)
        target     = target.to(device)
        mask_tgt   = mask_tgt.to(device)

        optimizer.zero_grad()
        loss = model(seq, masked_seq, ift, uf, target, mask_tgt, mixup_alpha)
        if torch.isnan(loss) or torch.isinf(loss):
            continue                          # 跳过坏 batch，不更新参数
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # 梯度中若仍有 NaN（极端情况），跳过本步
        has_nan_grad = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in model.parameters())
        if has_nan_grad:
            optimizer.zero_grad()
            continue
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, ift, device, topk):
    model.eval()
    hits, ndcgs = [], []
    for seq, masked_seq, uf, target, _ in loader:
        seq, uf, target = seq.to(device), uf.to(device), target.to(device)
        logits = model(seq, seq, ift, uf)   # eval 时不用 masked_seq
        logits[:, 0] = -1e9
        tk = logits.topk(topk, dim=-1).indices
        for i in range(len(target)):
            t = target[i].item()
            lst = tk[i].tolist()
            if t in lst:
                rank = lst.index(t) + 1
                hits.append(1); ndcgs.append(1 / math.log2(rank + 1))
            else:
                hits.append(0); ndcgs.append(0.0)
    return float(np.mean(hits)), float(np.mean(ndcgs))


@torch.no_grad()
def predict(model, loader, ift, device, topk, id2item):
    model.eval()
    rows = []
    for seq, uf, uids in loader:
        seq, uf = seq.to(device), uf.to(device)
        # eval 模式 forward：seq 传两次，masked_seq 无效
        logits  = model(seq, seq, ift, uf)
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

    tr_ds  = RecDataset(tr_samps,    cfg.max_seq_len, proc.n_items,
                        mask_prob=cfg.mask_prob, mode="train")
    val_ds = RecDataset(val_samps,   cfg.max_seq_len, proc.n_items,
                        mask_prob=cfg.mask_prob, mode="train")
    te_ds  = RecDataset(test_samples,cfg.max_seq_len, proc.n_items,
                        mask_prob=0.0, mode="test")
    tr_ld  = DataLoader(tr_ds,  cfg.batch_size, shuffle=True,  num_workers=0)
    val_ld = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=0)
    te_ld  = DataLoader(te_ds,  cfg.batch_size, shuffle=False, num_workers=0)

    model = DINDeepFMv2(
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
            hr, ndcg = evaluate(model, val_ld, ift, cfg.device, cfg.topk)
            print(f"Ep {ep:3d} | Loss {loss:.4f} | "
                  f"HR@{cfg.topk}: {hr:.4f} | NDCG@{cfg.topk}: {ndcg:.4f}")
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
