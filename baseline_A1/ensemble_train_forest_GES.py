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
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATv2Conv, GPSConv, GENConv, DeepGCNLayer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
from dataclasses import dataclass, asdict, fields
import pandas as pd
import json
import os
from openai import OpenAI
import copy
import time
import random
import numpy as np
from typing import List, Dict, Any

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://llm-sctg3o0ri7j4gobl.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
)

# ─────────────────────────────────────────────────────────────────────────────
# 0. 全局配置
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH   = 'A1.npz'
VALID_RATIO = 0.2      # 从 train_idx 中划出多少比例作为验证集
SPLIT_SEED  = 42        # 划分 train/valid 的随机种子，保证所有模型用同一份划分
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

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
# 2. 各模型的配置类（dataclass）
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphGPSConfig:
    in_channels: int
    out_channels: int
    hidden_channels: int = 64      # 节点数较多（1万级别），hidden 保守一些控制显存
    num_layers: int = 3            # 总层数
    num_global_layers: int = 1     # 其中启用全局注意力(GPSConv)的层数，从最后往前数
    heads: int = 4                 # 注意力头数（保守设置）
    dropout: float = 0.5
    attn_dropout: float = 0.5
    
@dataclass
class GCNConfig:
    in_channels: int
    out_channels: int
    hidden_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.5


@dataclass
class GINConfig:
    in_channels: int
    out_channels: int
    hidden_channels: int = 128
    num_layers: int = 3
    mlp_layers: int = 2
    dropout: float = 0.5
    train_eps: bool = True


@dataclass
class GraphSAGEConfig:
    in_channels: int
    out_channels: int
    hidden_channels: int = 256
    num_layers: int = 3
    dropout: float = 0.5
    aggr: str = 'mean'


@dataclass
class ResidualGCNConfig:
    in_channels: int
    out_channels: int
    hidden_channels: int = 128
    num_layers: int = 4
    dropout: float = 0.5


@dataclass
class GATConfig:
    in_channels: int
    out_channels: int
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.5
    heads: int = 8
    attn_dropout: float = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# 3. 五个模型定义（结构与各自原始脚本保持一致，初始化统一接收 cfg）
# ─────────────────────────────────────────────────────────────────────────────        
class GraphGPS(nn.Module):
    """
    GraphGPS（General, Powerful, Scalable Graph Transformer）精简版。
    每一层 = 本地消息传递（GINConv） +（可选）全局自注意力（PyG 的 GPSConv）。
    整图一次性前向、不做 mini-batch 采样，适用于中等规模图（本任务约 1万节点）。
 
    为控制显存（全局注意力是 O(N^2)），只在最后 num_global_layers 层启用
    GPSConv（局部+全局混合），前面的层用纯 GINConv 做局部消息传递。
    """
    def __init__(self, cfg: GraphGPSConfig):
        super().__init__()
        self.cfg = cfg
        self.dropout = cfg.dropout
        hid = cfg.hidden_channels
 
        self.input_proj = nn.Linear(cfg.in_channels, hid)
 
        n_global = max(1, min(cfg.num_global_layers, cfg.num_layers))
        n_local  = cfg.num_layers - n_global
 
        self.layers = nn.ModuleList()
        self.layer_types = []   # 与 self.layers 一一对应，'local' 或 'global'
        self.bns = nn.ModuleList()   # 只给纯 local 层配 BN（GPSConv 内部自带 norm）
 
        for i in range(cfg.num_layers):
            mlp = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid))
            local_conv = GINConv(mlp, train_eps=True)
            if i >= n_local:
                self.layers.append(GPSConv(
                    hid, conv=local_conv, heads=cfg.heads,
                    dropout=cfg.attn_dropout, act='relu', norm='batch_norm',
                ))
                self.layer_types.append('global')
            else:
                self.layers.append(local_conv)
                self.bns.append(nn.BatchNorm1d(hid))
                self.layer_types.append('local')
 
        self.out_lin = nn.Linear(hid, cfg.out_channels)
 
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        bn_idx = 0
        for layer, ltype in zip(self.layers, self.layer_types):
            if ltype == 'local':
                x = layer(x, edge_index)
                x = self.bns[bn_idx](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                bn_idx += 1
            else:
                # GPSConv 内部已经包含了 局部conv + 全局注意力 + 残差 + norm + FFN
                x = layer(x, edge_index)
        return self.out_lin(x)
        
class GCN(nn.Module):
    def __init__(self, cfg: GCNConfig):
        super().__init__()
        self.cfg = cfg
        self.dropout = cfg.dropout
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GCNConv(cfg.in_channels, cfg.hidden_channels))
        self.bns.append(nn.BatchNorm1d(cfg.hidden_channels))
        for _ in range(cfg.num_layers - 2):
            self.convs.append(GCNConv(cfg.hidden_channels, cfg.hidden_channels))
            self.bns.append(nn.BatchNorm1d(cfg.hidden_channels))
        self.convs.append(GCNConv(cfg.hidden_channels, cfg.out_channels))

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
    def __init__(self, cfg: GINConfig):
        super().__init__()
        self.cfg = cfg
        self.dropout = cfg.dropout
        hid = cfg.hidden_channels
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GINConv(build_gin_mlp(cfg.in_channels, hid, cfg.mlp_layers),
                                   train_eps=cfg.train_eps))
        self.bns.append(nn.BatchNorm1d(hid))
        for _ in range(cfg.num_layers - 1):
            self.convs.append(GINConv(build_gin_mlp(hid, hid, cfg.mlp_layers),
                                       train_eps=cfg.train_eps))
            self.bns.append(nn.BatchNorm1d(hid))
        self.classifier = nn.Linear(hid, cfg.out_channels)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class GraphSAGE(nn.Module):
    def __init__(self, cfg: GraphSAGEConfig):
        super().__init__()
        self.cfg = cfg
        self.dropout = cfg.dropout
        hid = cfg.hidden_channels
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(SAGEConv(cfg.in_channels, hid, aggr=cfg.aggr))
        self.bns.append(nn.BatchNorm1d(hid))
        for _ in range(cfg.num_layers - 2):
            self.convs.append(SAGEConv(hid, hid, aggr=cfg.aggr))
            self.bns.append(nn.BatchNorm1d(hid))
        self.convs.append(SAGEConv(hid, cfg.out_channels, aggr=cfg.aggr))

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
    def __init__(self, cfg: ResidualGCNConfig):
        super().__init__()
        self.cfg = cfg
        hid = cfg.hidden_channels
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualGCNBlock(cfg.in_channels, hid, cfg.dropout))
        for _ in range(cfg.num_layers - 2):
            self.blocks.append(ResidualGCNBlock(hid, hid, cfg.dropout))
        self.out_conv = GCNConv(hid, cfg.out_channels)

    def forward(self, x, edge_index):
        for block in self.blocks:
            x = block(x, edge_index)
        return self.out_conv(x, edge_index)


class GAT(nn.Module):
    def __init__(self, cfg: GATConfig):
        super().__init__()
        self.cfg = cfg
        self.dropout = cfg.dropout
        hid, H = cfg.hidden_channels, cfg.heads
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        self.convs.append(GATv2Conv(cfg.in_channels, hid, heads=H,
                                     dropout=cfg.attn_dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(hid * H))
        for _ in range(cfg.num_layers - 2):
            self.convs.append(GATv2Conv(hid * H, hid, heads=H,
                                         dropout=cfg.attn_dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hid * H))
        self.convs.append(GATv2Conv(hid * H, cfg.out_channels, heads=1,
                                     dropout=cfg.attn_dropout, concat=False))

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

# ─────────────────────────────────────────────────────────────────────────────
# 4. 通用训练函数（沿用各原始脚本里"记录 best_loss 权重 + early stopping"的逻辑）
# ─────────────────────────────────────────────────────────────────────────────
def train_model(name, model, data, valid_train_mask, valid_mask, epochs, lr, weight_decay,
                 patience, min_epochs, grad_clip_norm=1.0, log_every=50, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc, best_state = -1.0, None
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

        # 每个 epoch 都在验证集上评估一次，作为 early stopping 的依据
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_acc = (val_out[valid_mask].argmax(1) ==
                       data.y[valid_mask]).float().mean().item()

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if epoch % log_every == 0:
            with torch.no_grad():
                train_acc = (out[valid_train_mask].argmax(1) ==
                             data.y[valid_train_mask]).float().mean().item()
            print(f"  [{name}] Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                  f"Best Val Acc: {best_val_acc:.4f}")

        if no_improve >= patience and epoch > min_epochs:
            print(f"  [{name}] Early stopping at epoch {epoch} (Best Val Acc: {best_val_acc:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_all_nodes(model, data):
    return model(data.x, data.edge_index).argmax(dim=1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# 5. 多数投票（逐节点，出现次数最多的类别 = 最终预测）
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
# 6. 数据类，给后面的Agent使用
# ─────────────────────────────────────────────────────────────────────────────
class GlobalData:
    def __init__(self):
        # 6.1 加载数据（所有模型共用）
        self.data, self.npz, self.num_nodes, self.num_feats, self.num_classes, self.train_idx, self.test_idx = load_data(DATA_PATH)
        data = self.data.to(DEVICE)
    
        self.origin_config_specs = {
            # 'GraphGPS': GraphGPSConfig(self.num_feats, self.num_classes, hidden_channels=64, num_layers=3, num_global_layers=1, heads=4, dropout=0.5, attn_dropout=0.5),
            'GCN': GCNConfig(self.num_feats, self.num_classes, hidden_channels=128, num_layers=3, dropout=0.5),
            'GIN': GINConfig(self.num_feats, self.num_classes, hidden_channels=128, num_layers=3, mlp_layers=2, dropout=0.5, train_eps=True),
            'GraphSAGE': GraphSAGEConfig(self.num_feats, self.num_classes, hidden_channels=256, num_layers=3, dropout=0.5, aggr='mean'),
            'GAT': GATConfig(self.num_feats, self.num_classes, hidden_channels=64, num_layers=3, heads=8, dropout=0.5, attn_dropout=0.3),
            'ResidualGCN': ResidualGCNConfig(self.num_feats, self.num_classes, hidden_channels=16, num_layers=16, dropout=0.5),
        }

        # 6.2 划分 train/valid（所有模型共用同一份划分）
        self.sub_train_idx, self.valid_idx = split_train_valid(
            self.train_idx, self.data.y.cpu(), VALID_RATIO, SPLIT_SEED
        )
        self.sub_train_idx = self.sub_train_idx.to(DEVICE)
        self.valid_idx     = self.valid_idx.to(DEVICE)
    
        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=DEVICE)
        self.train_mask[self.sub_train_idx] = True
        self.valid_train_mask = self.train_mask & (data.y >= 0)   # 用于训练的 mask

        # 真正的验证集 mask（与 valid_idx 对应），用于 early stopping
        self.valid_mask = torch.zeros(self.num_nodes, dtype=torch.bool, device=DEVICE)
        self.valid_mask[self.valid_idx] = True
        self.valid_mask = self.valid_mask & (data.y >= 0)

        # 挑选能增加best_snapshot_valid_acc的valid_pred
        self.best_snapshot_valid_acc = -1.
        self.best_snapshot_l = []

    def y_valid_true(self):
        return self.data.y[self.valid_idx].cpu().numpy()

    def get_valid_test_acc(self, model):
        all_preds = predict_all_nodes(model, self.data)
        valid_pred = all_preds[self.valid_idx.cpu().numpy()]
        test_pred = all_preds[self.test_idx]
        acc = (valid_pred == self.y_valid_true()).mean()
        return all_preds, valid_pred, test_pred, acc

    def check_incr_best_snapshot_valid_acc(self, snapshot):
        l = self.best_snapshot_l.copy() + [snapshot]

        _l = []
        for result_snapshot in l:
            valid_pred = result_snapshot['valid_pred']
            _l.append(valid_pred)
        
        valid_pred_matrix = np.stack(_l, axis=0)
        valid_ensemble = majority_vote(valid_pred_matrix)
        ensemble_acc = (valid_ensemble == gd.y_valid_true()).mean()

        if ensemble_acc > self.best_snapshot_valid_acc:
            self.best_snapshot_l.append(snapshot)
            self.best_snapshot_valid_acc = ensemble_acc
            print("\n\n"+"="*80)
            print(f'[ENSEMBLE ACCURARY] {self.best_snapshot_valid_acc}')
            print("="*80)

gd = GlobalData()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Agent, 每一个Agent都为一种模型服务，每次执行5分钟
# ─────────────────────────────────────────────────────────────────────────────
PROMPT='''
你是一个图模型调参专家

## 现在正在调试的模型
{tuning_graph_model}

## 图的配置如下
{config_spec}

## 历史修改轨迹(history trajectory)
历史修改轨迹由多个节点组成，整体是一棵树，节点的有pid属性，id属性，l属性。pid为当前节点的父节点的id，l为true时表示该节点是一个叶子，也即是一条trajectory的tail。给定一个叶子节点id，通过追溯它的p直到pid为-1，也即是根节点，可得到一个trajectory。
{history_trajectory}

## 模型的valid accuracy如下
{valid_accuracy}

## 目标
使模型计算出来的best_score(best valid accuracy)尽可能地大，当前的最大best_score为{best_score}。

## 调参细节
- valid accuracy低于0.7的模型是重点修改对象。
- 如果参数增大，best_score增大，则继续尝试增大参数；如果参数减小，best_score增大，则继续尝试减小参数。
- 当模型为GraphSAGE时，不要调试aggr参数。
- 参数一定要大于0。
- ResidualGCN可以堆叠很多层比如16、32、64。
- num_layers比hidden_channels更容易影响模型的输出精度。
- hidden_channels不能超过512，且它应该是16的倍数。
- dropout不得高于0.5。

## 输出
- 输出为调整后的参数，每次只修改一个参数，格式一定是合法的json格式，不能是Markdown格式(不能以```json开头)，不要输出思考过程，例子如{{"id":0, "pid":-1, p:"hidden_channels", v:"128->125"}}
- 注意，输出的json中，id是历史修改轨迹的最大id+1。
- 注意，要以拥有最大best_score的节点为父节点。
'''

@dataclass
class EditNode:
    id: int
    pid: int
    p: str
    v: str
    best_score: float
    l: bool

class EditTree:
    def __init__(self):
        self.node_d = dict()

    def add_node(self, id, pid, p, v, best_score):
        if pid != -1 and pid not in self.node_d:
            raise ValueError(f"pid={pid} not exists.")
        if id in self.node_d:
            raise ValueError(f"id={id} exists.")
        self.node_d[id] = EditNode(id, pid, p, v, best_score, True)
        if pid >= 0:
            self.node_d[pid].l = False
        return self.node_d[id]

    def get_best_trajectory(self):
        l = [node for _, node in self.node_d.items()]
        l.sort(key=lambda x: x.id)
            
        best_score = -1
        best_id = -1
        for node in l:
            if node.best_score > best_score:
                best_score = node.best_score
                best_id = node.id

        if best_id == -1:
            return []
    
        l = [self.node_d[best_id]]
        while l[-1].pid > -1:
            l.append(self.node_d[l[-1].pid])
    
        l.reverse()
    
        return l

    def get_best_score(self):
        best_score = -1
        best_id = -1
        for id, node in self.node_d.items():
            if node.best_score > best_score:
                best_score = node.best_score
                best_id = node.id
        return best_score

    def edit_config_spec(self, config_spec):
        ret = copy.deepcopy(config_spec)
        best_trajectory = self.get_best_trajectory()
        print(best_trajectory)
        for en in best_trajectory:
            if en.pid == -1:
                continue
            for f in fields(ret):
                # print(f"====> {f.name}")
                if en.p == f.name:
                    old_value, new_value = en.v.split('->')
                    print(f"====>setattr: {f.name}, {f.type(new_value)}")
                    setattr(ret, f.name, f.type(new_value))
        return ret

    def size(self):
        return len(self.node_d)

    def to_string(self):
        l = sorted([asdict(n) for id, n in self.node_d.items()], key=lambda x: x['id'])
        return json.dumps(l)


def dump_config(config_spec):
    ret = asdict(config_spec)
    del ret['in_channels']
    del ret['out_channels']
    return json.dumps(ret)


    
class Agent:
    def __init__(self, model_name, d: GlobalData):
        self.origin_config_spec = d.origin_config_specs[model_name]
        self.edit_tree = EditTree()
        self.model_name = model_name
        self.d = d
        self.next_action = None
        self.best_snapshot = None
        self.best_score = -1.
    #     self.result_snapshot = []
    #     self.result_snapshot_read_idx = -1

    # def read_result_snapshot(self):
    #     if self.result_snapshot_read_idx == -1:
    #         self.result_snapshot_read_idx = len(self.result_snapshot)
    #         ret = self.result_snapshot.copy()
    #         return ret
    #     else:
    #         ret = self.result_snapshot[self.result_snapshot_read_idx:].copy()
    #         self.result_snapshot_read_idx = len(self.result_snapshot)
    #         return ret

    def _train_model(self, config_spec):
        print(f"\n===== Training {self.model_name} =====")
        _model = globals()[self.model_name](config_spec)
        model = train_model(
            self.model_name, _model, self.d.data, self.d.valid_train_mask, self.d.valid_mask,
            epochs=1000, lr=0.01, weight_decay=5e-4, patience=30, min_epochs=100,
        )
        return model

    def gen_next_action(self, config_spec, valid_accurary, best_score):
        history_trajectory = self.edit_tree.to_string()

        prompt = PROMPT.format(
            tuning_graph_model=self.model_name,
            config_spec=dump_config(config_spec), 
            valid_accuracy=f"{valid_accurary}", 
            history_trajectory=history_trajectory,
            best_score=best_score
        )
        print(prompt)
        completion = client.chat.completions.create(
            model="qwen3.7-plus",
            messages=[{'role': 'user', 'content': prompt}]
        )
        json_str = completion.choices[0].message.content
        print(json_str)
    
        d = json.loads(json_str)
        return d

    def run(self):
        start_time = time.time()

        # while time.time() - start_time < 300:
        if self.edit_tree.size() == 0:
            model = self._train_model(self.origin_config_spec)

            all_preds, valid_pred, test_pred, acc = self.d.get_valid_test_acc(model)

            snapshot = {'all_preds':all_preds, 'valid_pred':valid_pred, 'test_pred':test_pred, 'acc':acc}
            # self.d.check_incr_best_snapshot_valid_acc(snapshot)
            if acc > self.best_score:
                self.best_score = acc
                self.best_snapshot = snapshot

            self.edit_tree.add_node(0, -1, '', '', acc)

            self.next_action = self.gen_next_action(self.origin_config_spec, acc, acc)

            # print("[0] next_action:", next_action)
        else:
            print('\n\n'+"="*80+'\n')
            new_node = self.edit_tree.add_node(self.next_action['id'], self.next_action['pid'], self.next_action['p'], self.next_action['v'], 100)
            config_spec = self.edit_tree.edit_config_spec(self.origin_config_spec)
            print(config_spec)
            
            model = self._train_model(config_spec)

            all_preds, valid_pred, test_pred, acc = self.d.get_valid_test_acc(model)

            snapshot = {'all_preds':all_preds, 'valid_pred':valid_pred, 'test_pred':test_pred, 'acc':acc}
            # self.d.check_incr_best_snapshot_valid_acc(snapshot)
            if acc > self.best_score:
                self.best_score = acc
                self.best_snapshot = snapshot

            new_node.best_score = acc

            self.next_action = self.gen_next_action(config_spec, acc, self.edit_tree.get_best_score())

            # print(f"[{id}] next_action:", next_action)



# def greedy_ensemble_selection(
#     agent_l: List["Agent"],
#     y_valid_true: np.ndarray,
#     max_size: int = 100,
#     patience: int = 10,
#     min_delta: float = 1e-6,
# ) -> Dict[str, Any]:
#     """
#     Caruana-style greedy ensemble selection（有放回选择 + 探索式早停）。

#     核心改动：每一轮都强制加入当轮最优候选（即使没有提升），
#     从而允许算法"越过"局部平票/局部下降的陷阱去探索更大的组合；
#     同时单独记录历史最优点，早停后回退（truncate）到历史最优状态，
#     保证最终结果绝不差于任何中间步骤。
#     """
#     n_valid = len(y_valid_true)
#     y_valid_true = np.asarray(y_valid_true)

#     all_labels = [y_valid_true]
#     for agent in agent_l:
#         all_labels.append(np.asarray(agent.best_snapshot["valid_pred"]))
#         all_labels.append(np.asarray(agent.best_snapshot["test_pred"]))
#     classes = np.unique(np.concatenate(all_labels))
#     class_to_idx = {c: i for i, c in enumerate(classes)}
#     n_classes = len(classes)

#     def encode(arr):
#         return np.array([class_to_idx[v] for v in arr], dtype=np.int64)

#     y_valid_true_enc = encode(y_valid_true)
#     valid_preds_enc = [encode(a.best_snapshot["valid_pred"]) for a in agent_l]
#     test_preds_enc = [encode(a.best_snapshot["test_pred"]) for a in agent_l]
#     n_test = len(test_preds_enc[0])

#     vote_counts_valid = np.zeros((n_valid, n_classes), dtype=np.int64)

#     def vote_predict(vote_counts):
#         return np.argmax(vote_counts, axis=1)

#     def accuracy_from_counts(vote_counts, y_true_enc):
#         return (vote_predict(vote_counts) == y_true_enc).mean()

#     selected_indices: List[int] = []
#     history: List[float] = []

#     best_overall_acc = -np.inf
#     best_overall_len = 0          # 历史最优时刻，ensemble 的长度
#     best_overall_counts = vote_counts_valid.copy()
#     no_improve_rounds = 0

#     for step in range(max_size):
#         best_step_acc = -np.inf
#         best_step_idx = -1
#         best_step_counts = None

#         # 遍历所有候选，找本轮加入后能带来最高 accuracy 的那个（即使不如历史最优，也选本轮最好的）
#         for idx, pred_enc in enumerate(valid_preds_enc):
#             trial_counts = vote_counts_valid.copy()
#             trial_counts[np.arange(n_valid), pred_enc] += 1
#             acc = accuracy_from_counts(trial_counts, y_valid_true_enc)
#             if acc > best_step_acc:
#                 best_step_acc = acc
#                 best_step_idx = idx
#                 best_step_counts = trial_counts

#         # 强制推进：无论是否提升，都接受本轮最优候选
#         vote_counts_valid = best_step_counts
#         selected_indices.append(best_step_idx)
#         history.append(best_step_acc)

#         # 更新历史最优记录
#         if best_step_acc > best_overall_acc + min_delta:
#             best_overall_acc = best_step_acc
#             best_overall_len = len(selected_indices)
#             best_overall_counts = vote_counts_valid.copy()
#             no_improve_rounds = 0
#         else:
#             no_improve_rounds += 1
#             if no_improve_rounds >= patience:
#                 break

#     # ---- 回退截断到历史最优点 ----
#     selected_indices = selected_indices[:best_overall_len]
#     vote_counts_valid = best_overall_counts

#     weights: Dict[int, int] = {}
#     for idx in selected_indices:
#         weights[idx] = weights.get(idx, 0) + 1

#     vote_counts_test = np.zeros((n_test, n_classes), dtype=np.int64)
#     for idx, w in weights.items():
#         vote_counts_test[np.arange(n_test), test_preds_enc[idx]] += w

#     idx_to_class = {i: c for c, i in class_to_idx.items()}
#     ensemble_valid_pred = np.array([idx_to_class[i] for i in vote_predict(vote_counts_valid)])
#     ensemble_test_pred = np.array([idx_to_class[i] for i in vote_predict(vote_counts_test)])

#     return {
#         "selected_indices": selected_indices,
#         "weights": weights,
#         "best_valid_acc": best_overall_acc if selected_indices else 0.0,
#         "ensemble_valid_pred": ensemble_valid_pred,
#         "ensemble_test_pred": ensemble_test_pred,
#         "history": history,   # 注意：这里是"探索过程"的 acc，不是回退后的最终 acc
#     }



def _single_greedy_run(
    valid_preds_enc: List[np.ndarray],
    y_valid_true_enc: np.ndarray,
    n_classes: int,
    sample_idx: np.ndarray,
    max_size: int,
    patience: int,
    min_delta: float,
) -> Dict[int, int]:
    """
    在给定的样本子集(sample_idx)上跑一次探索式贪心选择，返回 {agent_idx: 被选中次数}。
    这是内部工具函数，不直接暴露给用户。
    """
    n_sub = len(sample_idx)
    y_sub = y_valid_true_enc[sample_idx]
    preds_sub = [p[sample_idx] for p in valid_preds_enc]

    vote_counts = np.zeros((n_sub, n_classes), dtype=np.int64)

    def vote_predict(vc):
        return np.argmax(vc, axis=1)

    def acc_from_counts(vc):
        return (vote_predict(vc) == y_sub).mean()

    selected_indices: List[int] = []
    best_overall_acc = -np.inf
    best_overall_len = 0
    best_overall_counts = vote_counts.copy()
    no_improve_rounds = 0

    for _ in range(max_size):
        best_step_acc = -np.inf
        best_step_idx = -1
        best_step_counts = None

        for idx, pred_enc in enumerate(preds_sub):
            trial = vote_counts.copy()
            trial[np.arange(n_sub), pred_enc] += 1
            acc = acc_from_counts(trial)
            if acc > best_step_acc:
                best_step_acc = acc
                best_step_idx = idx
                best_step_counts = trial

        vote_counts = best_step_counts
        selected_indices.append(best_step_idx)

        if best_step_acc > best_overall_acc + min_delta:
            best_overall_acc = best_step_acc
            best_overall_len = len(selected_indices)
            best_overall_counts = vote_counts.copy()
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds >= patience:
                break

    selected_indices = selected_indices[:best_overall_len]

    weights: Dict[int, int] = {}
    for idx in selected_indices:
        weights[idx] = weights.get(idx, 0) + 1
    return weights


def greedy_ensemble_selection(
    agent_l: List["Agent"],
    y_valid_true,
    max_size: int = 50,
    patience: int = 5,
    min_delta: float = 1e-6,
    n_bags: int = 20,
    bag_fraction: float = 1.0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Bagged Caruana-style greedy ensemble selection（接口与原版兼容，新增参数均有默认值）。

    额外参数（可选，不传则使用默认值，等价于合理的 Bagging 配置）：
        n_bags: bootstrap 重采样轮数，每轮独立跑一次贪心选择
        bag_fraction: 每个 bag 的采样比例（相对完整 valid 集大小），默认 1.0 即与 Caruana 论文一致（有放回，等大小重采样）
        random_state: 随机种子，保证可复现

    Returns: 结构与原函数完全一致：
        {
            'selected_indices': ...,   # 展开后的模型下标列表（按权重展开，兼容旧字段含义）
            'weights': Dict[int, int], # 所有 bag 累加后的模型权重
            'best_valid_acc': float,   # 最终 ensemble 在完整 valid 集上的 accuracy
            'ensemble_valid_pred': np.ndarray,
            'ensemble_test_pred': np.ndarray,
            'history': List[float],    # 每个 bag 跑完之后，累计 ensemble 在完整 valid 集上的 accuracy变化
        }
    """
    rng = np.random.RandomState(random_state)

    n_valid = len(y_valid_true)
    y_valid_true = np.asarray(y_valid_true)

    # ---- 统一类别编码 ----
    all_labels = [y_valid_true]
    for agent in agent_l:
        all_labels.append(np.asarray(agent.best_snapshot["valid_pred"]))
        all_labels.append(np.asarray(agent.best_snapshot["test_pred"]))
    classes = np.unique(np.concatenate(all_labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    def encode(arr):
        return np.array([class_to_idx[v] for v in arr], dtype=np.int64)

    y_valid_true_enc = encode(y_valid_true)
    valid_preds_enc = [encode(a.best_snapshot["valid_pred"]) for a in agent_l]
    test_preds_enc = [encode(a.best_snapshot["test_pred"]) for a in agent_l]
    n_test = len(test_preds_enc[0])

    def vote_predict(vc):
        return np.argmax(vc, axis=1)

    def acc_on_full_valid(agg_weights: Dict[int, int]):
        vc = np.zeros((n_valid, n_classes), dtype=np.int64)
        for idx, w in agg_weights.items():
            vc[np.arange(n_valid), valid_preds_enc[idx]] += w
        return (vote_predict(vc) == y_valid_true_enc).mean(), vc

    # ---- Bagging 主循环 ----
    agg_weights: Dict[int, int] = {}
    history: List[float] = []
    n_sample = max(1, int(round(n_valid * bag_fraction)))

    for bag_i in range(n_bags):
        sample_idx = rng.choice(n_valid, size=n_sample, replace=True)

        bag_weights = _single_greedy_run(
            valid_preds_enc=valid_preds_enc,
            y_valid_true_enc=y_valid_true_enc,
            n_classes=n_classes,
            sample_idx=sample_idx,
            max_size=max_size,
            patience=patience,
            min_delta=min_delta,
        )

        for idx, w in bag_weights.items():
            agg_weights[idx] = agg_weights.get(idx, 0) + w

        # 记录累计到当前 bag 为止，在完整 valid 集上的 accuracy 变化趋势
        cur_acc, _ = acc_on_full_valid(agg_weights) if agg_weights else (0.0, None)
        history.append(cur_acc)

    # ---- 用累加权重在完整 valid 集上算最终结果 ----
    best_valid_acc, vote_counts_valid = acc_on_full_valid(agg_weights)

    vote_counts_test = np.zeros((n_test, n_classes), dtype=np.int64)
    for idx, w in agg_weights.items():
        vote_counts_test[np.arange(n_test), test_preds_enc[idx]] += w

    idx_to_class = {i: c for c, i in class_to_idx.items()}
    ensemble_valid_pred = np.array([idx_to_class[i] for i in vote_predict(vote_counts_valid)])
    ensemble_test_pred = np.array([idx_to_class[i] for i in vote_predict(vote_counts_test)])

    # 兼容旧字段：把权重展开成下标列表
    selected_indices: List[int] = []
    for idx, w in agg_weights.items():
        selected_indices.extend([idx] * w)

    return {
        "selected_indices": selected_indices,
        "weights": agg_weights,
        "best_valid_acc": best_valid_acc,
        "ensemble_valid_pred": ensemble_valid_pred,
        "ensemble_test_pred": ensemble_test_pred,
        "history": history,
    }

best_GES = None

start_time = time.time()

agent_l = []
for model_name in gd.origin_config_specs.keys():
    agent = Agent(model_name, gd)
    agent_l.append(agent)

try:
    while time.time() - start_time < 7200 - 300:
        for agent in agent_l:
            agent.run()
            
        ret = greedy_ensemble_selection(agent_l, gd.y_valid_true())
        print("="*80)
        print("greedy_ensemble_selection.ret:", ret['best_valid_acc'])
        if best_GES is None:
            best_GES = ret
            print("best_GES.best_valid_acc:", best_GES['best_valid_acc'])
        elif best_GES['best_valid_acc'] < ret['best_valid_acc']:
            best_GES = ret
            print("best_GES.best_valid_acc:", best_GES['best_valid_acc'])
        print("="*80)
except Exception as e:
    print(e)
finally: 
    print("=====>GES:", best_GES['best_valid_acc'])
    # print("=====>GES.history", best_GES['history'])
    # print("=====>VOTE:", gd.best_snapshot_valid_acc)
    
    # 对 test_idx 做多数投票融合，作为最终提交结果
    # test_pred_matrix = np.stack([result_snapshot['test_pred'] for result_snapshot in gd.best_snapshot_l], axis=0)
    # test_ensemble = majority_vote(test_pred_matrix)
    
    test_ensemble = best_GES['ensemble_test_pred']
    
    out_df = pd.DataFrame({'test_idx': gd.test_idx, 'label': test_ensemble})
    out_df.to_csv('predictions_ensemble_forest.csv', index=False)
    
    print(f"\n✓ Saved {len(out_df)} ensemble predictions to predictions_ensemble.csv")
    print("\nFirst 10 rows:")
    print(out_df.head(10).to_string(index=False))
    print("\nPredicted class distribution:")
    print(out_df['label'].value_counts().sort_index().to_string())
                
                
            
