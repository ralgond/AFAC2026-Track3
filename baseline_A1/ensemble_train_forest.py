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
from dataclasses import dataclass, asdict, fields
import pandas as pd
import json
import os
from openai import OpenAI
import copy
import time
import random

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

        while time.time() - start_time < 300:
            if self.edit_tree.size() == 0:
                model = self._train_model(self.origin_config_spec)

                all_preds, valid_pred, test_pred, acc = self.d.get_valid_test_acc(model)

                self.d.check_incr_best_snapshot_valid_acc({'all_preds':all_preds, 'valid_pred':valid_pred, 'test_pred':test_pred, 'acc':acc})

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

                self.d.check_incr_best_snapshot_valid_acc({'all_preds':all_preds, 'valid_pred':valid_pred, 'test_pred':test_pred, 'acc':acc})

                new_node.best_score = acc

                self.next_action = self.gen_next_action(config_spec, acc, self.edit_tree.get_best_score())

                # print(f"[{id}] next_action:", next_action)

all_result_snapshot = []
start_time = time.time()

agent_l = []
for model_name in gd.origin_config_specs.keys():
    agent = Agent(model_name, gd)
    agent_l.append(agent)
        
while time.time() - start_time < 3600 - 500:
    for agent in agent_l:
        if time.time() - start_time < 3600 - 500:
            agent.run()
        else:
            break
    
# 对 test_idx 做多数投票融合，作为最终提交结果
test_pred_matrix = np.stack([result_snapshot['test_pred'] for result_snapshot in gd.best_snapshot_l], axis=0)
test_ensemble = majority_vote(test_pred_matrix)

out_df = pd.DataFrame({'test_idx': gd.test_idx, 'label': test_ensemble})
out_df.to_csv('predictions_ensemble_forest.csv', index=False)

print(f"\n✓ Saved {len(out_df)} ensemble predictions to predictions_ensemble.csv")
print("\nFirst 10 rows:")
print(out_df.head(10).to_string(index=False))
print("\nPredicted class distribution:")
print(out_df['label'].value_counts().sort_index().to_string())
                
                
            
