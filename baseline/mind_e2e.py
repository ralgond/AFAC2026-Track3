"""
End-to-End MIND（Multi-Interest Network with Dynamic Routing）
=============================================================

论文：Multi-Interest Network with Dynamic Routing for Recommendation at Tmall
       https://arxiv.org/abs/1904.08030

与 BST 的关键区别
-----------------
BST：
  - 历史序列 → Transformer Encoder → 单一序列向量
  - user_repr 是一个固定向量，全量点积打分

MIND：
  - 历史序列 → 胶囊网络（Dynamic Routing）→ K 个兴趣胶囊
  - 推理时用目标 item 向量与 K 个兴趣胶囊做 soft-attention，
    选出最相关兴趣作为 user_repr
  - 训练时引入 Label-aware Attention（用 target item 引导路由）
  - 支持全量 Softmax 训练 + NDCG@K 评估（与 BST pipeline 完全兼容）

架构
----
User Tower（MIND）：
  ① 8个用户特征域 embedding → concat → MLP → u_feat_vec   (B, D)
  ② 历史序列 item_id embedding → 胶囊路由 → K 个兴趣胶囊   (B, K, D)
     训练：Label-aware Attention 用 target item 加权路由
     推理：用全量 item 向量 soft-attention 选出最优兴趣
  ③ concat [u_feat_vec; 选中兴趣向量] → MLP → user_repr   (B, D)

Item Tower（与 BST 完全一致）：
  ① item_id embedding                                     (N, E)
  ② 4个item特征域 → FM二阶交叉 → item_fm_vec              (N, E)
  ③ concat → MLP → item_repr                              (N, D)

打分：
  score(u, i) = user_repr(u) · item_repr(i)
  训练：全量 Softmax Cross-Entropy

验证指标：NDCG@10（用户级划分，后 10% uid）
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
    output_path = "submission_mind.csv"

    max_seq_len  = 50           # 历史序列截断长度
    emb_dim      = 32           # 每个特征域的 embedding 维度
    repr_dim     = 128          # user/item tower 输出维度

    # ── MIND 胶囊路由参数 ──
    num_interests = 4           # 兴趣胶囊数量 K（论文建议 2~7）
    routing_iters = 3           # 动态路由迭代次数（论文默认 3）
    pow_p         = 2           # Label-aware Attention 幂次（论文建议 2~5）

    mlp_dims     = [256, 128]   # User Tower 最终 MLP 各层宽度
    dropout      = 0.2

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
# 1. 数据处理（与 BST 版本完全一致）
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
# 2. Dataset（与 BST 完全一致）
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

# ── 3a. 共用工具函数（与 BST 一致）──
def build_mlp(in_dim, hidden_dims, out_dim, dropout):
    layers = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def fm_second_order(emb_list):
    """FM 二阶交叉项（与 BST 完全一致）"""
    stacked = torch.stack(emb_list, dim=-2)   # (..., F, E)
    sum_sq  = stacked.sum(dim=-2) ** 2        # (..., E)
    sq_sum  = (stacked ** 2).sum(dim=-2)      # (..., E)
    return 0.5 * (sum_sq - sq_sum)


# ── 3b. 动态路由胶囊网络（MIND 核心）──
class CapsuleRouting(nn.Module):
    """
    MIND 动态路由模块。

    原理（论文 Sec 3.2）：
      - 输入 item embedding 集合 H = {h_1, ..., h_n}  (B, L, D)
      - 每个 item 通过可学习变换矩阵 S ∈ R^{K×D×D} 映射到 K 个胶囊空间：
          u_hat_{ji} = S_j · h_i       (B, L, D)  for each capsule j
      - 通过 softmax 路由权重 c 迭代更新 K 个兴趣胶囊 v：
          b_{ij} += v_j · u_hat_{ji}
          c_{ij}  = softmax_j(b_{ij})
          v_j     = squash( Σ_i c_{ij} · u_hat_{ji} )
      - squash 激活保证向量长度 < 1，用长度表示概率

    注意事项：
      1. padding 位置（item_id=0）的 h 全零，贡献为 0，无需额外 mask
      2. 推理时返回全部 K 个胶囊；训练时配合 Label-aware Attention
      3. 路由中 b 不参与梯度（detach），仅胶囊向量参与反传
    """

    def __init__(self, in_dim: int, out_dim: int,
                 num_interests: int, routing_iters: int):
        """
        in_dim        : item embedding 维度 D_in
        out_dim       : 兴趣胶囊维度 D_out（通常 = repr_dim）
        num_interests : 兴趣胶囊数量 K
        routing_iters : 动态路由迭代次数（默认 3）
        """
        super().__init__()
        self.K    = num_interests
        self.iter = routing_iters

        # 变换矩阵 S：将 item embedding 映射到每个胶囊空间
        # 共享权重方案：用 Linear(D_in, K*D_out)，相当于 K 个变换共享输入投影
        # 等价于论文中 B ∈ R^{K×D_in×D_out}，此处用矩阵乘法高效实现
        self.W = nn.Linear(in_dim, num_interests * out_dim, bias=False)
        self.out_dim = out_dim

    @staticmethod
    def squash(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Squash 激活（论文公式 3）：
          squash(v) = ||v||² / (1 + ||v||²) * v / ||v||
        保证输出向量长度 ∈ [0, 1)
        """
        norm_sq = (x ** 2).sum(dim=dim, keepdim=True)   # (..., 1)
        norm    = norm_sq.sqrt()
        scale   = norm_sq / (1.0 + norm_sq) / (norm + 1e-8)
        return scale * x

    def forward(self, seq_emb: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        """
        seq_emb  : (B, L, D_in)   item embedding 序列
        pad_mask : (B, L)         True = padding 位置
        return   : (B, K, D_out) K 个兴趣胶囊向量
        """
        B, L, _ = seq_emb.shape
        K = self.K
        D = self.out_dim

        # 将 padding 位置的 embedding 置零，确保不污染路由
        valid_mask = (~pad_mask).float().unsqueeze(-1)  # (B, L, 1)
        seq_emb    = seq_emb * valid_mask               # (B, L, D_in)

        # u_hat : item → K 个胶囊的候选向量
        # (B, L, D_in) @ W^T → (B, L, K*D) → (B, L, K, D)
        u_hat = self.W(seq_emb).view(B, L, K, D)       # (B, L, K, D)

        # 初始化路由 logit b（不参与梯度）
        b = torch.zeros(B, L, K, device=seq_emb.device)  # (B, L, K)

        # 将 padding 位置的 logit 设为极小值，使其路由权重 ≈ 0
        inf_mask = pad_mask.float() * (-1e9)              # (B, L)

        v = None
        for _ in range(self.iter):
            # 路由权重（对胶囊维 K softmax）
            c = F.softmax(b + inf_mask.unsqueeze(-1), dim=2)  # (B, L, K)

            # 汇聚：v_j = Σ_i c_{ij} * u_hat_{ij}
            # (B, L, K, 1) * (B, L, K, D) → sum over L → (B, K, D)
            v = (c.unsqueeze(-1) * u_hat).sum(dim=1)          # (B, K, D)

            # Squash
            v = self.squash(v, dim=-1)                        # (B, K, D)

            if _ < self.iter - 1:
                # 更新路由 logit（detach v 避免二阶梯度问题）
                # b += v · u_hat（点积相似度）
                # (B, K, D) 与 (B, L, K, D) 点积 → (B, L, K)
                b = b + (v.detach().unsqueeze(1) * u_hat.detach()).sum(dim=-1)

        return v  # (B, K, D)


# ── 3c. Label-aware Attention（训练时用 target item 引导路由）──
def label_aware_attention(interest_caps: torch.Tensor,
                           target_emb:   torch.Tensor,
                           pow_p:        int = 2) -> torch.Tensor:
    """
    论文公式 (6)：用 target item embedding 对 K 个兴趣胶囊做幂次 soft-attention，
    得到训练时的用户兴趣向量。

    Args:
        interest_caps : (B, K, D)  K 个兴趣胶囊
        target_emb    : (B, D)     目标 item embedding
        pow_p         : int        幂次（越大越尖锐，论文推荐 2~5）
    Return:
        user_interest : (B, D)     加权聚合后的兴趣向量
    """
    # 相似度 (B, K)
    sim = (interest_caps * target_emb.unsqueeze(1)).sum(-1)  # (B, K)
    # 幂次放大，再 softmax
    w   = F.softmax(sim.pow(pow_p), dim=-1)                  # (B, K)
    # 加权求和
    out = (w.unsqueeze(-1) * interest_caps).sum(1)           # (B, D)
    return out


# ── 3d. 推理时的兴趣聚合（取最相关兴趣）──
def inference_interest_agg(interest_caps: torch.Tensor,
                            item_repr:     torch.Tensor) -> torch.Tensor:
    """
    推理时对每个用户选出与候选 item 最匹配的兴趣向量。
    用于全量打分场景，实现为矩阵运算：

      score_{u,i} = max_k ( interest_caps[u,k] · item_repr[i] )

    Args:
        interest_caps : (B, K, D)   K 个兴趣胶囊
        item_repr     : (N, D)      全量 item 表示
    Return:
        logits : (B, N)             全量打分
    """
    # (B, K, D) @ (D, N) → (B, K, N)
    scores_per_k = interest_caps @ item_repr.T   # (B, K, N)
    # 对 K 维取 max → (B, N)
    logits, _    = scores_per_k.max(dim=1)
    return logits


# ── 3e. User Tower（MIND 版）──
class UserTower(nn.Module):
    """
    MIND User Tower：

    训练模式（target_emb 已知）：
      ① 8个用户特征域 → concat → MLP → u_feat_vec       (B, D)
      ② 历史序列 → 胶囊路由 → K 个兴趣胶囊              (B, K, D)
         → Label-aware Attention（target_emb）→ interest  (B, D)
      ③ concat [u_feat_vec; interest] → MLP → user_repr  (B, D)

    推理模式（target_emb=None，返回兴趣胶囊供外部全量打分）：
      ① 同上
      ② 历史序列 → 胶囊路由 → K 个兴趣胶囊              (B, K, D)
      返回 (u_feat_vec, interest_caps) 供 MINDE2E.forward 做全量打分
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
        n_u = len(user_feat_dims)

        # 用户特征 MLP：n_u * E → D
        self.u_feat_mlp = build_mlp(E * n_u, [D], D, cfg.dropout)

        # 共享 item id embedding
        self.item_id_emb = item_id_emb_shared

        # item embedding 投影：E → D（保持与胶囊维度一致）
        self.seq_proj = nn.Linear(E, D)

        # MIND 胶囊路由
        self.capsule = CapsuleRouting(
            in_dim        = D,
            out_dim       = D,
            num_interests = cfg.num_interests,
            routing_iters = cfg.routing_iters,
        )

        self.pow_p = cfg.pow_p

        # 最终输出 MLP：u_feat_vec(D) + interest(D) → D
        self.out_mlp = build_mlp(D * 2, cfg.mlp_dims, D, cfg.dropout)

        self.drop = nn.Dropout(cfg.dropout)

    def encode_seq(self, seq_ids: torch.Tensor):
        """
        将序列 item_id → embedding，返回 (seq_emb, pad_mask)。
        seq_emb  : (B, L, D)
        pad_mask : (B, L)
        """
        pad_mask = (seq_ids == 0)
        seq_e    = F.gelu(self.seq_proj(self.item_id_emb(seq_ids)))  # (B, L, D)
        seq_e    = self.drop(seq_e)
        return seq_e, pad_mask

    def encode_user_feat(self, user_feats: torch.Tensor):
        """用户特征编码，返回 (B, D)"""
        u_emb_list = [e(user_feats[:, i]) for i, e in enumerate(self.u_embs)]
        u_cat      = torch.cat(u_emb_list, dim=-1)   # (B, n_u*E)
        return self.u_feat_mlp(u_cat)                # (B, D)

    def forward(self, seq_ids: torch.Tensor,
                user_feats: torch.Tensor,
                target_emb: torch.Tensor | None = None):
        """
        seq_ids    : (B, L)   历史序列 item id（0=padding）
        user_feats : (B, 8)   用户离散特征
        target_emb : (B, D) or None
            训练时传入，使用 Label-aware Attention 聚合兴趣；
            推理时为 None，返回 (u_feat_vec, interest_caps)。

        Returns:
            训练模式 → user_repr : (B, D)
            推理模式 → (u_feat_vec : (B, D), interest_caps : (B, K, D))
        """
        # ① 用户特征编码
        u_feat_vec = self.encode_user_feat(user_feats)     # (B, D)

        # ② 历史序列 → 胶囊路由
        seq_emb, pad_mask = self.encode_seq(seq_ids)        # (B, L, D), (B, L)
        interest_caps = self.capsule(seq_emb, pad_mask)     # (B, K, D)

        if target_emb is not None:
            # 训练：Label-aware Attention
            interest = label_aware_attention(
                interest_caps, target_emb, self.pow_p)      # (B, D)
            concat = torch.cat([u_feat_vec, interest], dim=-1)  # (B, 2D)
            return self.out_mlp(concat)                     # (B, D) = user_repr

        # 推理：返回原始兴趣胶囊，由 MINDE2E.forward 做全量 max 打分
        return u_feat_vec, interest_caps


# ── 3f. Item Tower（与 BST 完全一致）──
class ItemTower(nn.Module):
    """
    Item Tower（结构与 BST 版本完全一致）：
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


# ── 3g. 整体模型（MIND E2E）──
class MINDE2E(nn.Module):
    """
    端到端 MIND 推荐模型。

    训练流程：
      1. Item Tower 计算全量 item_repr              (N+1, D)
      2. 取出 target item 对应的 item_repr          (B, D)
      3. User Tower（Label-aware Attention 模式）   (B, D)
      4. 全量 Softmax CE

    推理流程（全量打分）：
      1. Item Tower 计算全量 item_repr              (N+1, D)
      2. User Tower 返回 (u_feat_vec, interest_caps)
      3. 拼接 u_feat_vec 到每个兴趣胶囊后过 MLP 得到 K 套 user_repr
         * 注：为兼容矩阵乘法全量打分，推理时对 K 个兴趣胶囊分别生成 user_repr，
                然后对每个 item 取 max
    """
    def __init__(self, cfg, n_items,
                 item_feat_dims, item_feat_padidxs,
                 user_feat_dims, user_feat_padidxs):
        super().__init__()
        E = cfg.emb_dim
        D = cfg.repr_dim
        self.n_items  = n_items
        self.repr_dim = D

        # User/Item Tower 共享 item id embedding
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

    def _all_item_repr(self, ift: torch.Tensor, device):
        """计算全量 item 表示，返回 (N+1, D)"""
        all_ids   = torch.arange(self.n_items + 1, device=device)
        all_feats = ift.to(device)
        return self.item_tower(all_ids, all_feats)   # (N+1, D)

    def forward(self, seq_ids, ift, user_feats, target_ids=None):
        """
        seq_ids    : (B, L)
        ift        : (N+1, 4)  全量 item 特征查找表
        user_feats : (B, 8)
        target_ids : (B,) long  训练时传入，推理时为 None
        """
        device    = seq_ids.device
        item_repr = self._all_item_repr(ift, device)       # (N+1, D)

        if target_ids is not None:
            # ─── 训练模式 ───
            # 取出 target item embedding 用于 Label-aware Attention
            target_emb = item_repr[target_ids]             # (B, D)

            # User Tower：Label-aware Attention → user_repr (B, D)
            user_repr = self.user_tower(
                seq_ids, user_feats, target_emb=target_emb)  # (B, D)

            # 全量打分
            logits = user_repr @ item_repr.T               # (B, N+1)
            loss   = F.cross_entropy(logits[:, 1:], target_ids - 1)
            return loss

        else:
            # ─── 推理模式 ───
            # User Tower：返回 K 个兴趣胶囊
            u_feat_vec, interest_caps = self.user_tower(
                seq_ids, user_feats, target_emb=None)      # (B, D), (B, K, D)

            # 对 K 个兴趣胶囊分别过 out_mlp，得到 K 套 user_repr
            # 复用 user_tower.out_mlp：concat [u_feat_vec; interest_k] → D
            K = interest_caps.shape[1]
            B = seq_ids.shape[0]
            D = self.repr_dim

            u_expanded = u_feat_vec.unsqueeze(1).expand(B, K, D)   # (B, K, D)
            concat_k   = torch.cat([u_expanded, interest_caps], dim=-1)  # (B, K, 2D)
            # (B*K, 2D) → out_mlp → (B*K, D) → (B, K, D)
            repr_k = self.user_tower.out_mlp(
                concat_k.view(B * K, -1)).view(B, K, D)             # (B, K, D)

            # 全量打分：对 K 维取 max
            # (B, K, D) @ (D, N+1) → (B, K, N+1) → max over K → (B, N+1)
            logits_k = repr_k @ item_repr.T                         # (B, K, N+1)
            logits, _ = logits_k.max(dim=1)                         # (B, N+1)
            return logits


# ══════════════════════════════════════════════════════════
# 4. 训练 & 评估（与 BST 完全一致）
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

    model = MINDE2E(
        cfg, proc.n_items,
        proc.item_feat_dims, proc.item_feat_padidxs,
        proc.user_feat_dims, proc.user_feat_padidxs,
    ).to(cfg.device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"MIND config  : K={cfg.num_interests} interests | "
          f"routing_iters={cfg.routing_iters} | pow_p={cfg.pow_p}\n")

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
