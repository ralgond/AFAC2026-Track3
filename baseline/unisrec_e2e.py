"""
End-to-End UniSRec (Universal Sequence Representation for Recommendation)
=========================================================================

论文：Towards Universal Sequence Representations for Recommendation
      Yupeng Hou et al., KDD 2022  https://arxiv.org/abs/2206.05941

核心创新（相对于 SASRec / BST）
--------------------------------
1. **Item 文本语义表示**
   原论文用预训练语言模型（BERT）对 item 描述文本编码，得到语义向量 e_text。
   本实现用 item 的多个离散特征（类目、桶值等）通过 MoE Adaptor 融合，
   模拟跨域迁移能力，同时保持与本数据集的完全兼容。

2. **MoE Adaptor（Mixture-of-Experts Adaptor）**
   将 item 语义向量（或特征融合向量）映射到推荐空间：
     e_adapted = Σ_k  g_k(x) · FFN_k(x)
   门控 g_k 由 softmax 决定，K 个专家分别捕捉不同的特征模式。

3. **参数化白化（Parametric Whitening）**
   对 item 表示做白化（零均值、单位协方差），缓解各向异性问题，
   使向量空间更均匀，改善 dot-product 召回的 NDCG。
   实现为可学习的 LayerNorm（等价于近似白化）。

4. **对比学习辅助任务（Sequence-Item Contrastive）**
   序列表示 z_u 与目标 item 表示 z_i 在 batch 内做 InfoNCE 对比学习，
   作为主损失（全量 Softmax CE）的辅助项：
     L = L_main + λ * L_cl

架构
----
Item Encoder（MoE Adaptor）:
  ① 4个离散特征域 embedding → concat → item_raw  (N, 4E)
  ② MoE Adaptor: K 个 FFN expert + 门控网络 → item_repr  (N, D)
  ③ 参数化白化（Parametric Whitening）→ item_repr_w  (N, D)

User Encoder（SASRec 骨干）:
  ① 历史序列 → item_repr(白化后) 查表 → seq_emb  (B, L, D)
  ② 因果 Transformer（单向 Attention）→ 取最后有效 token → seq_repr  (B, D)
  ③ 8个用户特征 embedding → MoE Adaptor → u_feat_repr  (B, D)
  ④ seq_repr + u_feat_repr → gate fusion → user_repr  (B, D)

损失:
  L_main = FullSoftmax_CE(user_repr · item_repr_w^T, target)
  L_cl   = InfoNCE(user_repr, target_item_repr_w)
  L      = L_main + λ * L_cl

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
    output_path = "submission_unisrec.csv"

    max_seq_len  = 50           # 历史序列最大长度
    emb_dim      = 32           # 离散特征域 embedding 维度
    repr_dim     = 128          # 统一隐向量维度 D

    # ── SASRec Transformer ──
    n_heads      = 4
    n_layers     = 2
    ffn_dim      = 256
    dropout      = 0.2

    # ── MoE Adaptor ──
    n_experts    = 4            # 专家数量 K
    expert_dim   = 256          # 每个 expert 的 FFN 中间维度

    # ── 对比学习 ──
    cl_lambda    = 0.1          # 对比学习损失权重 λ
    cl_temp      = 0.07         # InfoNCE 温度 τ

    # ── 训练 ──
    epochs       = 30
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
# 1. 数据处理（与 BST/DIN 版本一致）
# ══════════════════════════════════════════════════════════
class DataProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.item2id  = {}
        self.id2item  = {}
        self.iid2feat = {}
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

        # ── item 词表 ──
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

        # ── item 特征 ──
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

        # ── 样本构建 ──
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

        # ── 全量 item 特征张量 (N+1, 4) ──
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

# ── 3a. 工具 ──
def build_mlp(in_dim, hidden_dims, out_dim, dropout):
    layers, d = [], in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# ── 3b. MoE Adaptor ──
class MoEAdaptor(nn.Module):
    """
    Mixture-of-Experts Adaptor（UniSRec 核心模块）。

    将输入向量 x ∈ R^{in_dim} 通过 K 个专家 FFN 映射到 R^{out_dim}：
        g = softmax(W_g · x)              门控权重  (B, K)
        h_k = GELU(W_{k,1} · x) W_{k,2}   第 k 个专家输出  (B, out_dim)
        out = Σ_k  g_k * h_k

    语义：每个专家捕捉不同的特征交互模式，门控网络自适应地组合它们。
    这是 UniSRec 实现跨域迁移的关键——不同域的 item 可以激活不同的专家路径。
    """
    def __init__(self, in_dim, out_dim, n_experts, expert_dim, dropout=0.0):
        super().__init__()
        self.n_experts = n_experts

        # K 个专家 FFN：in_dim → expert_dim → out_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, expert_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, out_dim),
            )
            for _ in range(n_experts)
        ])

        # 门控网络
        self.gate   = nn.Linear(in_dim, n_experts)
        self.norm   = nn.LayerNorm(out_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        """
        x   : (..., in_dim)
        out : (..., out_dim)
        """
        g = F.softmax(self.gate(x), dim=-1)              # (..., K)
        # 各专家输出堆叠：(..., K, out_dim)
        expert_out = torch.stack([e(x) for e in self.experts], dim=-2)
        # 加权求和：g (..., K, 1) * expert_out (..., K, D) → (..., D)
        out = (g.unsqueeze(-1) * expert_out).sum(dim=-2)
        return self.norm(self.drop(out))


# ── 3c. 参数化白化（Parametric Whitening）──
class ParametricWhitening(nn.Module):
    """
    UniSRec 中的参数化白化层。

    原论文对 item 表示进行白化预处理以缓解向量各向异性（anisotropy），
    使高维空间更均匀地被利用，改善基于 dot-product 的检索质量。

    实现为可学习 LayerNorm（等价于近似白化 + 仿射变换）：
        out = γ ⊙ (x - μ) / σ + β
    参数 γ, β 全部可学习，初始化为 γ=1, β=0（恒等变换）后自适应调整。
    """
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x):
        return self.ln(x)


# ── 3d. 单向（因果）Transformer Layer（SASRec 风格）──
class CausalTransformerLayer(nn.Module):
    """
    Pre-LN 单向（Causal）Transformer。
    通过 attn_mask 屏蔽未来位置，使序列建模具有自回归性质。
    """
    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None, attn_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h,
                         key_padding_mask=key_padding_mask,
                         attn_mask=attn_mask,
                         need_weights=False)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


# ── 3e. Item Encoder（MoE Adaptor + 参数化白化）──
class ItemEncoder(nn.Module):
    """
    UniSRec Item Encoder：
      ① 4个离散特征域 embedding → concat → item_raw  (..., 4E)
      ② MoE Adaptor → item_repr  (..., D)
      ③ 参数化白化 → item_repr_w  (..., D)

    白化后的 item_repr_w 同时用于：
      - 序列建模时的 token embedding 查表
      - 最终打分 user_repr · item_repr_w^T
    因此需要在每个 forward 中重新计算全量（或批量）item 表示。
    推理时 item_repr_w 可离线缓存。
    """
    def __init__(self, cfg, item_feat_dims, item_feat_padidxs):
        super().__init__()
        E = cfg.emb_dim
        D = cfg.repr_dim

        self.i_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(item_feat_dims, item_feat_padidxs)
        ])
        n_i = len(item_feat_dims)  # 4

        # MoE Adaptor：4E → D
        self.moe = MoEAdaptor(
            in_dim    = E * n_i,
            out_dim   = D,
            n_experts = cfg.n_experts,
            expert_dim= cfg.expert_dim,
            dropout   = cfg.dropout,
        )

        # 参数化白化
        self.whitening = ParametricWhitening(D)

    def forward(self, item_ids_ignored, item_feats):
        """
        item_feats : (..., 4)  long
        return     : (..., D)  白化后的 item 表示
        """
        fi_list = [e(item_feats[..., j]) for j, e in enumerate(self.i_embs)]
        fi_cat  = torch.cat(fi_list, dim=-1)       # (..., 4E)
        item_r  = self.moe(fi_cat)                 # (..., D)
        return self.whitening(item_r)              # (..., D)


# ── 3f. User Encoder（SASRec + MoE 用户特征融合）──
class UserEncoder(nn.Module):
    """
    UniSRec User Encoder：
      ① 历史序列 → item_repr(白化后) 查表 + 位置编码 → seq_emb  (B, L, D)
      ② 因果 Transformer → 取最后有效 token 输出 → seq_repr  (B, D)
      ③ 8个用户特征 embedding → concat → MoE Adaptor → u_feat_repr  (B, D)
      ④ Gate Fusion：seq_repr 和 u_feat_repr 加权融合 → user_repr  (B, D)
    """
    def __init__(self, cfg, user_feat_dims, user_feat_padidxs):
        super().__init__()
        E = cfg.emb_dim
        D = cfg.repr_dim

        # 位置编码
        self.pos_emb = nn.Embedding(cfg.max_seq_len + 1, D)

        # 因果 Transformer
        self.layers = nn.ModuleList([
            CausalTransformerLayer(D, cfg.n_heads, cfg.ffn_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.seq_norm = nn.LayerNorm(D)

        # 用户特征 MoE Adaptor
        n_u = len(user_feat_dims)  # 8
        self.u_embs = nn.ModuleList([
            nn.Embedding(dim, E, padding_idx=pidx)
            for dim, pidx in zip(user_feat_dims, user_feat_padidxs)
        ])
        self.u_moe = MoEAdaptor(
            in_dim    = E * n_u,
            out_dim   = D,
            n_experts = cfg.n_experts,
            expert_dim= cfg.expert_dim,
            dropout   = cfg.dropout,
        )

        # Gate Fusion：seq_repr + u_feat_repr → user_repr
        # gate ∈ [0,1]，自适应决定两路贡献比例
        self.gate_fc = nn.Linear(D * 2, 1)
        self.out_norm = nn.LayerNorm(D)
        self.drop = nn.Dropout(cfg.dropout)

    @staticmethod
    def _causal_mask(L, device):
        """上三角为 True（屏蔽未来），对角线允许（自身可见）"""
        mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        return mask  # (L, L)

    def forward(self, seq_ids, user_feats, item_repr_all):
        """
        seq_ids       : (B, L)      历史 item id（0=padding）
        user_feats    : (B, 8)      用户离散特征
        item_repr_all : (N+1, D)    全量 item 白化表示（含 pad index=0）
        return        : (B, D)
        """
        B, L   = seq_ids.shape
        device = seq_ids.device

        # ① 序列 token embedding（用白化后的 item 表示）
        seq_e = item_repr_all[seq_ids]                     # (B, L, D)

        # 位置编码（1-indexed，padding 位置 pos=0）
        pad_mask = (seq_ids == 0)                          # (B, L)
        positions = torch.arange(1, L + 1, device=device).unsqueeze(0)
        positions = positions.masked_fill(pad_mask, 0)
        seq_e = self.drop(seq_e + self.pos_emb(positions)) # (B, L, D)

        # ② 因果 Transformer
        causal = self._causal_mask(L, device)              # (L, L)
        x = seq_e
        for layer in self.layers:
            x = layer(x, key_padding_mask=pad_mask, attn_mask=causal)
        x = self.seq_norm(x)                               # (B, L, D)

        # 取最后一个非 padding token 的输出
        lengths = (~pad_mask).sum(dim=1).clamp(min=1)      # (B,)
        last_idx = (lengths - 1).clamp(min=0)              # (B,)
        seq_repr = x[torch.arange(B, device=device), last_idx]  # (B, D)

        # ③ 用户特征 MoE
        u_emb_list = [e(user_feats[:, i]) for i, e in enumerate(self.u_embs)]
        u_cat       = torch.cat(u_emb_list, dim=-1)        # (B, 8E)
        u_feat_repr = self.u_moe(u_cat)                    # (B, D)

        # ④ Gate Fusion
        gate = torch.sigmoid(
            self.gate_fc(torch.cat([seq_repr, u_feat_repr], dim=-1))
        )                                                  # (B, 1)
        user_repr = gate * seq_repr + (1 - gate) * u_feat_repr
        return self.out_norm(user_repr)                    # (B, D)


# ── 3g. 对比学习损失（InfoNCE）──
def info_nce_loss(user_repr, item_repr_pos, temperature=0.07):
    """
    Sequence-Item Contrastive Learning（UniSRec 辅助任务）。

    正例：(user_repr[i], item_repr_pos[i])  ← 同一条样本的序列和目标 item
    负例：batch 内其他样本的目标 item（in-batch negatives）

    L_cl = -log( exp(sim(u,i+)/τ) / Σ_j exp(sim(u,ij)/τ) )

    user_repr    : (B, D)  L2-归一化
    item_repr_pos: (B, D)  L2-归一化
    """
    u = F.normalize(user_repr, dim=-1)       # (B, D)
    v = F.normalize(item_repr_pos, dim=-1)   # (B, D)
    logits = u @ v.T / temperature           # (B, B)
    labels = torch.arange(len(u), device=u.device)
    return F.cross_entropy(logits, labels)


# ── 3h. 整体模型（UniSRec E2E）──
class UniSRec(nn.Module):
    """
    端到端 UniSRec 推荐模型。

    设计原则：
      - Item Encoder（MoE + 白化）与 User Encoder 解耦
      - 推理时 item_repr_w 可离线缓存，打分为一次矩阵乘法
      - 训练时加入 InfoNCE 对比学习辅助 NDCG 优化
    """
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 user_feat_dims, user_feat_padidxs):
        super().__init__()
        self.n_items   = n_items
        self.cl_lambda = cfg.cl_lambda
        self.cl_temp   = cfg.cl_temp

        self.item_encoder = ItemEncoder(cfg, item_feat_dims, item_feat_padidxs)
        self.user_encoder = UserEncoder(cfg, user_feat_dims, user_feat_padidxs)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def get_all_item_repr(self, ift, device):
        """
        计算全量 item 白化表示 (N+1, D)。
        推理时可离线缓存，训练时每个 mini-batch 重新计算（保证梯度流动）。

        ift : (N+1, 4) 全量 item 特征查找表
        """
        all_ids   = torch.arange(self.n_items + 1, device=device)
        all_feats = ift.to(device)
        return self.item_encoder(all_ids, all_feats)   # (N+1, D)

    def forward(self, seq_ids, ift, user_feats, target_ids=None):
        """
        seq_ids    : (B, L)
        ift        : (N+1, 4)
        user_feats : (B, 8)
        target_ids : (B,) long  训练时传入，None 时返回全量打分
        """
        device = seq_ids.device

        # 全量 item 表示（每 forward 重新算，保证白化层梯度）
        item_repr_all = self.get_all_item_repr(ift, device)  # (N+1, D)

        # 用户表示（依赖白化后的 item repr 做 lookup）
        user_repr = self.user_encoder(seq_ids, user_feats, item_repr_all)  # (B, D)

        # 全量打分 (B, N+1)
        logits = user_repr @ item_repr_all.T

        if target_ids is not None:
            # ── 主损失：全量 Softmax CE ──
            loss_main = F.cross_entropy(logits[:, 1:], target_ids - 1)

            # ── 辅助损失：InfoNCE 对比学习 ──
            # target item 的白化表示作为正例
            target_repr = item_repr_all[target_ids]          # (B, D)
            loss_cl = info_nce_loss(user_repr, target_repr, self.cl_temp)

            loss = loss_main + self.cl_lambda * loss_cl
            return loss, loss_main.item(), loss_cl.item()

        return logits


# ══════════════════════════════════════════════════════════
# 4. 训练 & 评估
# ══════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, ift, device):
    model.train()
    total_loss, total_main, total_cl = 0.0, 0.0, 0.0
    for seq, uf, target in loader:
        seq, uf, target = seq.to(device), uf.to(device), target.to(device)
        optimizer.zero_grad()
        loss, l_main, l_cl = model(seq, ift, uf, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        total_main += l_main
        total_cl   += l_cl
    n = len(loader)
    return total_loss / n, total_main / n, total_cl / n


@torch.no_grad()
def evaluate(model, loader, ift, device, topk):
    model.eval()
    hits, ndcgs = [], []
    for seq, uf, target in loader:
        seq, uf, target = seq.to(device), uf.to(device), target.to(device)
        logits = model(seq, ift, uf)
        logits[:, 0] = -1e9
        tk = logits.topk(topk, dim=-1).indices
        for i in range(len(target)):
            t = target[i].item()
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
    all_uids = sorted(set(s["uid"] for s in train_samples))
    n_val_u  = max(1, int(len(all_uids) * 0.1))
    val_uids = set(all_uids[-n_val_u:])
    val_samps= [s for s in train_samples if     s["uid"] in val_uids]
    tr_samps = [s for s in train_samples if not s["uid"] in val_uids]
    print(f"  Val users={len(val_uids)} | Val={len(val_samps)} | Train={len(tr_samps)}\n")

    tr_ds  = RecDataset(tr_samps,     cfg.max_seq_len, mode="train")
    val_ds = RecDataset(val_samps,    cfg.max_seq_len, mode="train")
    te_ds  = RecDataset(test_samples, cfg.max_seq_len, mode="test")
    tr_ld  = DataLoader(tr_ds,  cfg.batch_size, shuffle=True,  num_workers=0)
    val_ld = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=0)
    te_ld  = DataLoader(te_ds,  cfg.batch_size, shuffle=False, num_workers=0)

    model = UniSRec(
        cfg, proc.n_items,
        proc.item_feat_dims, proc.item_feat_padidxs,
        proc.user_feat_dims, proc.user_feat_padidxs,
    ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"  cl_lambda={cfg.cl_lambda}  cl_temp={cfg.cl_temp}"
          f"  n_experts={cfg.n_experts}\n")

    ift       = proc.item_feat_tensor
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, cfg.epochs, eta_min=cfg.lr * 0.01)

    best_ndcg, best_state = 0.0, None

    for ep in range(1, cfg.epochs + 1):
        loss, l_main, l_cl = train_epoch(model, tr_ld, optimizer, ift, cfg.device)
        scheduler.step()

        if ep % 5 == 0 or ep == cfg.epochs:
            hr, ndcg = evaluate(model, val_ld, ift, cfg.device, cfg.topk)
            print(
                f"Ep {ep:3d} | Loss {loss:.4f} "
                f"(main {l_main:.4f} + cl {l_cl:.4f}) | "
                f"HR@{cfg.topk}: {hr:.4f} | NDCG@{cfg.topk}: {ndcg:.4f}"
            )
            if ndcg > best_ndcg:
                best_ndcg  = ndcg
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  ✓ Best NDCG@{cfg.topk}: {best_ndcg:.4f} — saved")
        else:
            print(f"Ep {ep:3d} | Loss {loss:.4f} (main {l_main:.4f} + cl {l_cl:.4f})")

    if best_state:
        model.load_state_dict({k: v.to(cfg.device) for k, v in best_state.items()})
    print(f"\nFinal best NDCG@{cfg.topk}: {best_ndcg:.4f}")

    rows = predict(model, te_ld, ift, cfg.device, cfg.topk, proc.id2item)
    pd.DataFrame(rows).to_csv(cfg.output_path, index=False)
    print(f"Saved → {cfg.output_path}")
    print(pd.DataFrame(rows).head(3).to_string())


if __name__ == "__main__":
    main()
