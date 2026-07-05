"""
ensemble_train.py
==================

统一驱动 5 个图神经网络模型（GCN / GIN / GraphSAGE / GATv2 / Residual GCN）
在同一份数据、同一种 train/valid 划分下分别训练，训练完成后的模型全部保留在
内存中，对 valid_idx 和 test_idx 做预测，并按"出现次数最多的类别"做多数投票
融合（题目里说的 max pool，其实就是逐节点的众数投票 / majority vote）。

用法：
    python ensemble_train.py

依赖：
    torch, torch_geometric, scipy, numpy, pandas, scikit-learn
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATv2Conv
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 0. 全局配置
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = 'A1.npz'
VALID_RATIO = 0.2      # 从 train_idx 中划出多少比例作为验证集
SPLIT_SEED  = 42        # 划分 train/valid 的随机种子，保证所有模型用同一份划分
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────────────────────────────────────
# 1. 数据加载（只做一次，所有模型共用）
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: str):
    npz = np.load(path, allow_pickle=True)

    adj = csr_matrix(
        (npz['adj_data'], npz['adj_indices'], npz['adj_indptr']),
        shape=tuple(npz['adj_shape'])
    )
    attr = csr_matrix(
        (npz['attr_data'], npz['attr_indices'], npz['attr_indptr']),
        shape=tuple(npz['attr_shape'])
    )

    labels    = torch.tensor(npz['labels'],    dtype=torch.long)
    train_idx = npz['train_idx']
    test_idx  = npz['test_idx']

    num_nodes   = int(npz['adj_shape'][0])
    num_feats   = int(npz['attr_shape'][1])
    num_classes = int(labels[labels >= 0].max().item()) + 1

    print(f"Nodes: {num_nodes} | Features: {num_feats} | Classes: {num_classes}")
    print(f"Train(all): {len(train_idx)} | Test: {len(test_idx)}")

    x = torch.tensor(attr.toarray(), dtype=torch.float32)

    cx = adj.tocoo()
    edge_index = torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    print(f"Edges (undirected, deduped): {edge_index.shape[1]}")

    data = Data(x=x, edge_index=edge_index, y=labels, num_nodes=num_nodes)
    return data, npz, num_nodes, num_feats, num_classes, train_idx, test_idx


def split_train_valid(train_idx: np.ndarray, labels: torch.Tensor,
                       valid_ratio: float, seed: int):
    """
    把 train_idx 中"确实有标签"的节点，按固定随机种子切分成
    sub_train_idx / valid_idx，所有模型共用这一份划分。
    """
    train_idx = np.asarray(train_idx)
    y_train_idx = labels[train_idx].numpy()

    labeled_mask = y_train_idx >= 0
    labeled_nodes = train_idx[labeled_mask]
    labeled_y     = y_train_idx[labeled_mask]

    # 尝试按类别分层抽样；若某些类别样本太少无法分层，退回普通随机切分
    try:
        sub_train_idx, valid_idx = train_test_split(
            labeled_nodes, test_size=valid_ratio,
            random_state=seed, stratify=labeled_y
        )
    except ValueError:
        sub_train_idx, valid_idx = train_test_split(
            labeled_nodes, test_size=valid_ratio, random_state=seed
        )

    print(f"Split -> sub_train: {len(sub_train_idx)} | valid: {len(valid_idx)}")
    return (torch.tensor(sub_train_idx, dtype=torch.long),
            torch.tensor(valid_idx,     dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# 2. 四个模型定义（结构与各自原始脚本保持一致）
# ─────────────────────────────────────────────────────────────────────────────
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers=3, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def build_gin_mlp(in_channels, out_channels, mlp_layers):
    layers = []
    for i in range(mlp_layers):
        in_c  = in_channels if i == 0 else out_channels
        layers.append(nn.Linear(in_c, out_channels))
        if i < mlp_layers - 1:
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GIN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128, num_layers=3, mlp_layers=2, dropout=0.5, train_eps=True):
        super().__init__()
        self.dropout = dropout
        hid = hidden_channels
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GINConv(build_gin_mlp(in_channels, hid, mlp_layers),
                                   train_eps=train_eps))
        self.bns.append(nn.BatchNorm1d(hid))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(build_gin_mlp(hid, hid, mlp_layers),
                                       train_eps=train_eps))
            self.bns.append(nn.BatchNorm1d(hid))
        self.classifier = nn.Linear(hid, out_channels)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=256, num_layers=3, dropout=0.5, aggr='mean'):
        super().__init__()
        self.dropout = dropout
        hid = hidden_channels
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hid, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hid))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hid, hid, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hid))
        self.convs.append(SAGEConv(hid, out_channels, aggr=aggr))

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class ResidualGCNBlock(nn.Module):
    """
    单个残差 GCN block：
        h' = Dropout(ReLU(BN(GCNConv(h))))
        h' = h' + shortcut(h)     # 残差连接
    输入输出维度不同时，shortcut 用 Linear 做投影，否则用 Identity。
    """
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv    = GCNConv(in_channels, out_channels)
        self.bn      = nn.BatchNorm1d(out_channels)
        self.dropout = dropout
        self.shortcut = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x, edge_index):
        h = self.conv(x, edge_index)
        h = self.bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h + self.shortcut(x)


class ResidualGCN(nn.Module):
    """
    堆叠残差 GCN block + 线性分类头。
    Architecture:
        ResidualGCNBlock(in  -> hid)
        ResidualGCNBlock(hid -> hid)  x (num_layers - 2)
        GCNConv(hid -> out)           # 输出层，无残差/激活
    """
    def __init__(self, in_channels, out_channels, hidden_channels=128, num_layers=4, dropout=0.5):
        super().__init__()
        hid = hidden_channels
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualGCNBlock(in_channels, hid, dropout))
        for _ in range(num_layers - 2):
            self.blocks.append(ResidualGCNBlock(hid, hid, dropout))
        self.out_conv = GCNConv(hid, out_channels)

    def forward(self, x, edge_index):
        for block in self.blocks:
            x = block(x, edge_index)
        return self.out_conv(x, edge_index)


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=3, dropout=0.5, heads=8, attn_dropout=0.3):
        super().__init__()
        self.dropout = dropout
        hid, H = hidden_channels, heads
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hid, heads=H,
                                     dropout=attn_dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(hid * H))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hid * H, hid, heads=H,
                                         dropout=attn_dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hid * H))
        self.convs.append(GATv2Conv(hid * H, out_channels, heads=1,
                                     dropout=attn_dropout, concat=False))

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 通用训练函数（沿用各原始脚本里"记录 best_loss 权重 + early stopping"的逻辑）
# ─────────────────────────────────────────────────────────────────────────────
def train_model(name, model, data, valid_train_mask, epochs, lr, weight_decay,
                 patience, min_epochs, grad_clip_norm=1.0, log_every=50, seed=42):
    torch.manual_seed(seed)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss, best_state = float('inf'), None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[valid_train_mask], data.y[valid_train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss - 1e-6:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                acc = (out[valid_train_mask].argmax(1) ==
                       data.y[valid_train_mask]).float().mean().item()
            print(f"  [{name}] Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {acc:.4f} | Best loss: {best_loss:.4f}")

        if no_improve >= patience and epoch > min_epochs:
            print(f"  [{name}] Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_all_nodes(model, data):
    return model(data.x, data.edge_index).argmax(dim=1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 4. 多数投票（逐节点，出现次数最多的类别 = 最终预测）
# ─────────────────────────────────────────────────────────────────────────────
def majority_vote(pred_matrix: np.ndarray) -> np.ndarray:
    """
    pred_matrix: shape [num_models, num_query_nodes]
    返回: shape [num_query_nodes] 的融合预测标签
    """
    final = np.zeros(pred_matrix.shape[1], dtype=pred_matrix.dtype)
    for j in range(pred_matrix.shape[1]):
        counts = Counter(pred_matrix[:, j].tolist())
        most_common = counts.most_common()
        max_count = most_common[0][1]
        candidates = sorted(l for l, c in most_common if c == max_count)
        final[j] = candidates[0]   # 平票时取标签值最小的
    return final


# ─────────────────────────────────────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # 5.1 加载数据（所有模型共用）
    data, npz, num_nodes, num_feats, num_classes, train_idx, test_idx = load_data(DATA_PATH)
    data = data.to(DEVICE)

    # 5.2 划分 train/valid（所有模型共用同一份划分）
    sub_train_idx, valid_idx = split_train_valid(
        train_idx, data.y.cpu(), VALID_RATIO, SPLIT_SEED
    )
    sub_train_idx = sub_train_idx.to(DEVICE)
    valid_idx     = valid_idx.to(DEVICE)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
    train_mask[sub_train_idx] = True
    valid_train_mask = train_mask & (data.y >= 0)   # 用于训练的 mask

    # 5.3 各模型的构造参数 + 训练超参数（沿用各自原始脚本的设置）
    model_specs = {
        'GCN': dict(
            model=GCN(num_feats, num_classes, hidden_channels=128, num_layers=3, dropout=0.5),
            epochs=500, lr=0.01, weight_decay=5e-4, patience=30, min_epochs=100,
        ),
        'GIN': dict(
            model=GIN(num_feats, num_classes, hidden_channels=128,
                      num_layers=3, mlp_layers=2, dropout=0.5, train_eps=True),
            epochs=500, lr=0.01, weight_decay=5e-4, patience=30, min_epochs=100,
        ),
        'SAGE': dict(
            model=GraphSAGE(num_feats, num_classes, hidden_channels=256,
                             num_layers=3, dropout=0.5, aggr='mean'),
            epochs=300, lr=0.01, weight_decay=5e-4, patience=30, min_epochs=100,
        ),
        'GAT': dict(
            model=GAT(num_feats, num_classes, hidden_channels=64,
                      num_layers=3, heads=8, dropout=0.5, attn_dropout=0.3),
            epochs=1000, lr=0.005, weight_decay=5e-4, patience=100, min_epochs=100,
        ),
        'ResidualGCN': dict(
            model=ResidualGCN(num_feats, num_classes, hidden_channels=128,
                               num_layers=4, dropout=0.5),
            epochs=500, lr=0.01, weight_decay=5e-4, patience=30, min_epochs=100,
        ),
    }

    trained_models = {}
    valid_preds = {}   # name -> np.array, 每个 valid 节点的预测标签
    test_preds  = {}   # name -> np.array, 每个 test 节点的预测标签

    # 5.4 依次训练 4 个模型，全部保留在内存中
    for name, spec in model_specs.items():
        print(f"\n===== Training {name} =====")
        model = train_model(
            name, spec['model'], data, valid_train_mask,
            epochs=spec['epochs'], lr=spec['lr'], weight_decay=spec['weight_decay'],
            patience=spec['patience'], min_epochs=spec['min_epochs'],
        )
        trained_models[name] = model   # 保存在内存中，供后续复用

        all_preds = predict_all_nodes(model, data)
        valid_preds[name] = all_preds[valid_idx.cpu().numpy()]
        test_preds[name]  = all_preds[test_idx]

    # 5.5 对 valid_idx 做多数投票融合，并计算融合后的验证集准确率
    valid_pred_matrix = np.stack([valid_preds[n] for n in model_specs.keys()], axis=0)
    valid_ensemble = majority_vote(valid_pred_matrix)

    y_valid_true = data.y[valid_idx].cpu().numpy()
    ensemble_acc = (valid_ensemble == y_valid_true).mean()

    print("\n===== Validation set: per-model vs. ensemble accuracy =====")
    for name in model_specs.keys():
        acc = (valid_preds[name] == y_valid_true).mean()
        print(f"  {name:5s} valid acc: {acc:.4f}")
    print(f"  {'ENSEMBLE':5s} valid acc: {ensemble_acc:.4f}")

    # 5.6 对 test_idx 做多数投票融合，作为最终提交结果
    test_pred_matrix = np.stack([test_preds[n] for n in model_specs.keys()], axis=0)
    test_ensemble = majority_vote(test_pred_matrix)

    out_df = pd.DataFrame({'test_idx': test_idx, 'label': test_ensemble})
    out_df.to_csv('predictions_ensemble.csv', index=False)

    print(f"\n✓ Saved {len(out_df)} ensemble predictions to predictions_ensemble.csv")
    print("\nFirst 10 rows:")
    print(out_df.head(10).to_string(index=False))
    print("\nPredicted class distribution:")
    print(out_df['label'].value_counts().sort_index().to_string())

    return trained_models, valid_ensemble, test_ensemble


if __name__ == '__main__':
    main()
