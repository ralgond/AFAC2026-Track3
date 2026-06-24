"""
End-to-End BST (Behavior Sequence Transformer)
===============================================

论文：Behavior Sequence Transformer for E-commerce Recommendation in Alibaba
       https://arxiv.org/abs/1905.06874

与原 DIN+DeepFM 的关键区别
--------------------------
原版 DIN+DeepFM：
  - 历史序列用 mean-pooling + cross-attention 兴趣池化
  - FM 二阶特征交叉
  - user/item tower 解耦，全量打分

BST：
  - 历史序列用 Transformer Encoder 建模（多层 Self-Attention）
  - 目标 item 拼入序列末尾，让序列 token 与目标 item 充分交互
  - Transformer 输出拼接用户画像特征 → MLP → 点击率预测
  - 保留 user/item tower 解耦设计，支持全量打分（无需召回）

架构
----
User Tower:
  ① 用户8个特征域 embedding → concat → MLP → u_feat_vec   (B, D)
  ② 历史序列 item_id + 位置编码 → Transformer Encoder
     → 取 [CLS] token 或 mean-pool → seq_repr              (B, D)
  ③ concat [u_feat_vec; seq_repr] → MLP → user_repr        (B, D)

Item Tower:
  ① item_id embedding                                      (N, E)
  ② 4个item特征域 embedding → FM二阶交叉 → item_fm_vec      (N, E)
  ③ concat → MLP → item_repr                               (N, D)

打分:
  score(u, i) = user_repr(u) · item_repr(i)
  训练: 全量 Softmax Cross-Entropy

验证指标：NDCG@10（用户级划分，后10% uid）
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
    output_path = "submission_bst.csv"

    max_seq_len  = 50          # 历史序列截断长度（不含 target token）
    emb_dim      = 32          # 每个特征域的 embedding 维度
    repr_dim     = 128         # user/item tower 输出维度

    # ── BST Transformer 参数 ──
    n_heads      = 4           # Multi-Head Attention 头数
    n_layers     = 2           # Transformer Encoder 层数
    ffn_dim      = 256         # Transformer FFN 中间维度

    mlp_dims     = [256, 128]  # User Tower 最终 MLP 各层宽度
    dropout      = 0.2

    epochs       = 50
    batch_size   = 256
    lr           = 1e-3
    weight_decay = 1e-5
    seed         = 42
    topk         = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)


# ══════════════════════════════════════════════════════════
# 1. 数据处理（与 DIN+DeepFM 版本完全一致）
# ══════════════════════════════════════════════════════════
class DataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.item2id  = {}
        self.id2item  = {}
        self.iid2feat = {}
        self.uid2feat = {}
        self.item_feat_cols = ["i_cat_01", "i_cat_02", "i_cat_03", "i_bucket_01"]
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
        for col in ["item_seq_raw", "item_seq_dedup"]:
            for df in [train_df, test_df]:
                if col in df.columns:
                    for s in df[col].dropna():
                        all_items.update(self._parse_seq(s))
        all_items.update(train_df["target_iid"].str.strip().dropna())
        all_items = sorted(all_items)
        self.item2id = {iid: idx + 1 for idx, iid in enumerate(all_items)}
        self.id2item = {v: k for k, v in self.item2id.items()}
        self.n_items = len(self.item2id)
        print(f"  Total items : {self.n_items}")

        # item 特征（pad_idx = max+1，emb_size = max+2）
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

        # 全量 item 特征张量  (N+1, 4)
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
# 3. 模型组件
# ══════════════════════════════════════════════════════════

# ── 3a. 共用工具函数 ──
def build_mlp(in_dim, hidden_dims, out_dim, dropout):
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def fm_second_order(emb_list):
    """
    emb_list : list of (..., E) tensors
    return   : (..., E)  FM 二阶交叉项
    """
    stacked = torch.stack(emb_list, dim=-2)     # (..., F, E)
    sum_sq  = stacked.sum(dim=-2) ** 2          # (..., E)
    sq_sum  = (stacked ** 2).sum(dim=-2)         # (..., E)
    return 0.5 * (sum_sq - sq_sum)


# ── 3b. BST Transformer Encoder Block ──
class BSTTransformerLayer(nn.Module):
    """
    标准 Pre-LN Transformer Encoder Layer。
    BST 原文使用 Leaky-ReLU FFN，此处使用 GELU 效果相当。
    """
    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        """
        x               : (B, L, D)
        key_padding_mask: (B, L) True=padding
        """
        # Pre-LN Self-Attention + 残差
        h = self.norm1(x)
        h, _ = self.attn(h, h, h,
                         key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + h
        # Pre-LN FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x


# ── 3c. BST Sequence Encoder ──
class BSTSequenceEncoder(nn.Module):
    """
    BST 核心模块：
      - 历史序列 + [可选 target token] 拼成一条序列
      - 加入可学习位置编码
      - 多层 Transformer Encoder
      - 取序列最后有效位置的输出作为序列表示

    全量打分模式下 target_ids=None，仅对历史序列建模；
    输出取 mean-pool（忽略 padding）作为序列表示。
    """
    def __init__(self, d_model, n_heads, n_layers, ffn_dim,
                 max_seq_len, dropout):
        super().__init__()
        # +1 为 target token 留位置；+1 为 padding=0
        self.pos_emb = nn.Embedding(max_seq_len + 2, d_model)
        self.layers  = nn.ModuleList([
            BSTTransformerLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, seq_emb, pad_mask):
        """
        seq_emb  : (B, L, D)  已经是 item embedding，但还未加位置编码
        pad_mask : (B, L)     True = padding 位置
        return   : (B, D)     序列表示
        """
        B, L, D = seq_emb.shape
        device   = seq_emb.device

        # 位置编码：1-indexed，padding 位置 pos=0（会被 mask）
        positions = torch.arange(1, L + 1, device=device).unsqueeze(0)  # (1, L)
        positions = positions.masked_fill(pad_mask, 0)                   # pad 位置 pos→0
        x = self.drop(seq_emb + self.pos_emb(positions))                 # (B, L, D)

        # 多层 Transformer
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask)
        x = self.norm(x)

        # Mean-pool（忽略 padding）
        valid   = (~pad_mask).float().unsqueeze(-1)            # (B, L, 1)
        seq_out = (x * valid).sum(1) / valid.sum(1).clamp(min=1)  # (B, D)
        return seq_out


# ── 3d. User Tower（BST 版）──
class UserTower(nn.Module):
    """
    BST User Tower：
      ① 8个用户特征域 embedding → concat → MLP → u_feat_vec  (B, D)
      ② item_id embedding + 位置编码 → BST Transformer → seq_repr  (B, D)
      ③ concat [u_feat_vec; seq_repr] → MLP → user_repr  (B, D)

    注意：BST 原文将 target item 拼在序列末位，与序列做交互后用其输出作为最终表示。
         全量打分模式下目标 item 未知，改为 mean-pool 序列输出。
         如需精排（候选已知）可在 forward 中传入 target_emb 拼入序列末位。
    """
    def __init__(self, cfg, n_items,
                 user_feat_dims, user_feat_padidxs,
                 item_id_emb_shared):
        super().__init__()
        E = cfg.emb_dim
        D = cfg.repr_dim

        # 用户特征 embedding（8个域）
        self.u_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(user_feat_dims, user_feat_padidxs)
        ])
        n_u = len(user_feat_dims)  # 8

        # 用户特征 MLP：8E → D
        self.u_feat_mlp = build_mlp(E * n_u, [D], D, cfg.dropout)

        # 共享 item id embedding
        self.item_id_emb = item_id_emb_shared

        # 序列 embedding 投影：E → D
        self.seq_proj = nn.Linear(E, D)

        # BST Transformer Encoder
        self.bst = BSTSequenceEncoder(
            d_model    = D,
            n_heads    = cfg.n_heads,
            n_layers   = cfg.n_layers,
            ffn_dim    = cfg.ffn_dim,
            max_seq_len= cfg.max_seq_len,
            dropout    = cfg.dropout,
        )

        # 最终输出 MLP：u_feat_vec(D) + seq_repr(D) → D
        self.out_mlp = build_mlp(D * 2, cfg.mlp_dims, D, cfg.dropout)

        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, seq_ids, user_feats):
        """
        seq_ids    : (B, L)   历史序列 item id（0=padding）
        user_feats : (B, 8)   用户离散特征
        return     : (B, D)   用户表示
        """
        # ① 用户特征编码
        u_emb_list  = [e(user_feats[:, i]) for i, e in enumerate(self.u_embs)]
        u_cat        = torch.cat(u_emb_list, dim=-1)      # (B, 8E)
        u_feat_vec   = self.u_feat_mlp(u_cat)             # (B, D)

        # ② 历史序列 → BST
        pad_mask = (seq_ids == 0)                          # (B, L) True=padding
        seq_e    = self.drop(
            F.gelu(self.seq_proj(self.item_id_emb(seq_ids)))
        )                                                  # (B, L, D)
        seq_repr = self.bst(seq_e, pad_mask)               # (B, D)

        # ③ 拼接 → 最终 MLP
        concat = torch.cat([u_feat_vec, seq_repr], dim=-1) # (B, 2D)
        return self.out_mlp(concat)                        # (B, D)


# ── 3e. Item Tower（与原 DIN+DeepFM 相同）──
class ItemTower(nn.Module):
    """
    Item Tower（结构与原版一致，保证 user/item 侧可解耦缓存）：
      ① item_id embedding                 (..., E)
      ② 4个item特征域 → FM 二阶交叉        (..., E)
      ③ 4个item特征域 concat              (..., 4E)
      ④ concat → MLP → item_repr         (..., D)
    """
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
        n_i = len(item_feat_dims)  # 4

        # item_id(E) + item_feat concat(4E) + item_fm(E) → MLP → D
        self.out_mlp = build_mlp(E + E * n_i + E, cfg.mlp_dims, D, cfg.dropout)

    def forward(self, item_ids, item_feats):
        """
        item_ids   : (...,)    long
        item_feats : (..., 4)  long
        return     : (..., D)
        """
        id_vec  = self.item_id_emb(item_ids)                   # (..., E)
        fi_list = [e(item_feats[..., j]) for j, e in enumerate(self.i_embs)]
        item_fm = fm_second_order(fi_list)                     # (..., E)
        fi_cat  = torch.cat(fi_list, dim=-1)                   # (..., 4E)
        concat  = torch.cat([id_vec, fi_cat, item_fm], dim=-1)
        return self.out_mlp(concat)                            # (..., D)


# ── 3f. 整体模型（BST E2E）──
class BSTE2E(nn.Module):
    """
    端到端 BST 推荐模型。
    训练：全量 Softmax Cross-Entropy（同 SASRec / DIN+DeepFM 版本）
    推理：item_repr 可离线缓存，推理仅需一次矩阵乘法
    """
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 user_feat_dims, user_feat_padidxs):
        super().__init__()
        E = cfg.emb_dim
        self.n_items = n_items

        # User/Item Tower 共享 item id embedding，使两侧在同一语义空间
        self.item_id_emb = nn.Embedding(n_items + 1, E, padding_idx=0)

        self.user_tower = UserTower(
            cfg, n_items, user_feat_dims, user_feat_padidxs,
            self.item_id_emb)

        self.item_tower = ItemTower(
            cfg, n_items, item_feat_dims, item_feat_padidxs,
            self.item_id_emb)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, seq_ids, ift, user_feats, target_ids=None):
        """
        seq_ids    : (B, L)
        ift        : (N+1, 4)  全量 item 特征查找表
        user_feats : (B, 8)
        target_ids : (B,) long  训练时传入，推理时为 None
        """
        device = seq_ids.device
        ift    = ift.to(device)

        # 用户侧：BST 序列建模
        user_repr = self.user_tower(seq_ids, user_feats)          # (B, D)

        # item 侧：全量批量计算
        all_ids   = torch.arange(self.n_items + 1, device=device)  # (N+1,)
        all_feats = ift                                              # (N+1, 4)
        item_repr = self.item_tower(all_ids, all_feats)             # (N+1, D)

        # 全量打分：(B, N+1)
        logits = user_repr @ item_repr.T

        if target_ids is not None:
            # 全量 Softmax CE（跳过 PAD index=0）
            loss = F.cross_entropy(logits[:, 1:], target_ids - 1)
            return loss
        return logits


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
        logits[:, 0] = -1e9          # 屏蔽 PAD item
        tk = logits.topk(topk, dim=-1).indices
        for i in range(len(target)):
            t   = target[i].item()
            lst = tk[i].tolist()
            if t in lst:
                rank = lst.index(t) + 1
                hits.append(1)
                ndcgs.append(1 / math.log2(rank + 1))
            else:
                hits.append(0)
                ndcgs.append(0)
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

    tr_ds  = RecDataset(tr_samps,     cfg.max_seq_len, mode="train")
    val_ds = RecDataset(val_samps,    cfg.max_seq_len, mode="train")
    te_ds  = RecDataset(test_samples, cfg.max_seq_len, mode="test")
    tr_ld  = DataLoader(tr_ds,  cfg.batch_size, shuffle=True,  num_workers=0)
    val_ld = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=0)
    te_ld  = DataLoader(te_ds,  cfg.batch_size, shuffle=False, num_workers=0)

    model = BSTE2E(
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
