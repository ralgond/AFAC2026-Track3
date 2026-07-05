"""
GAT (Graph Attention Network) node classifier using PyTorch Geometric.
Uses GATv2Conv (Brody et al. 2022) for node classification.

GATv2 improves over GATv1 by computing dynamic attention:
    e(h_i, h_j) = a^T · LeakyReLU( W · [h_i || h_j] )

This fixes GATv1's "static attention" issue where the ranking of neighbors
is independent of the query node — GATv2 is strictly more expressive.

Dataset format (A1.npz):
  adj_data / adj_indices / adj_indptr / adj_shape   -> CSR adjacency matrix
  attr_data / attr_indices / attr_indptr / attr_shape -> CSR feature matrix
  labels     : node labels (-1 = unlabeled)
  train_idx  : labeled training node indices
  test_idx   : test node indices to predict
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from scipy.sparse import csr_matrix
import pandas as pd


# ── Config ─────────────────────────────────────────────────────────────────────
class Config:
    # Paths
    data_path   = 'A1.npz'
    output_path = 'predictions_gat.csv'

    # Model architecture
    # NOTE: GAT is significantly more memory/compute intensive than GCN or SAGE.
    #   Each attention head computes pairwise scores over all edges.
    #   Recommended settings by device:
    #     CPU  : hidden_channels=32,  heads=4, num_layers=2  (~180s / 300 epochs)
    #     GPU  : hidden_channels=64,  heads=8, num_layers=3  (~30s  / 300 epochs)
    hidden_channels = 64    # channels per attention head
    num_layers      = 3     # intermediate dim = hidden_channels * heads
    heads           = 8     # number of attention heads
    dropout         = 0.5
    attn_dropout    = 0.3   # dropout applied to attention coefficients

    # Training
    lr             = 0.005
    weight_decay   = 5e-4
    epochs         = 1000
    patience       = 100     # early-stop patience
    min_epochs     = 100    # don't early-stop before this
    grad_clip_norm = 1.0
    log_every      = 50     # print progress every N epochs

    # Misc
    seed   = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── 1. Load data ───────────────────────────────────────────────────────────────
def load_data(cfg: Config):
    npz = np.load(cfg.data_path, allow_pickle=True)

    adj = csr_matrix(
        (npz['adj_data'], npz['adj_indices'], npz['adj_indptr']),
        shape=tuple(npz['adj_shape'])
    )
    attr = csr_matrix(
        (npz['attr_data'], npz['attr_indices'], npz['attr_indptr']),
        shape=tuple(npz['attr_shape'])
    )

    labels    = torch.tensor(npz['labels'],    dtype=torch.long)
    train_idx = torch.tensor(npz['train_idx'], dtype=torch.long)

    num_nodes   = int(npz['adj_shape'][0])
    num_feats   = int(npz['attr_shape'][1])
    num_classes = int(labels[labels >= 0].max().item()) + 1

    print(f"Nodes: {num_nodes} | Features: {num_feats} | Classes: {num_classes}")
    print(f"Train: {len(train_idx)} | Test: {len(npz['test_idx'])}")

    # Dense node feature matrix [num_nodes, num_feats]
    x = torch.tensor(attr.toarray(), dtype=torch.float32)

    # Convert CSR adjacency -> COO edge_index, make undirected, dedup
    cx = adj.tocoo()
    edge_index = torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    print(f"Edges (undirected, deduped): {edge_index.shape[1]}")

    data = Data(x=x, edge_index=edge_index, y=labels, num_nodes=num_nodes)
    return data, npz, num_nodes, num_feats, num_classes, train_idx


# ── 2. GAT model ───────────────────────────────────────────────────────────────
class GAT(nn.Module):
    """
    Multi-layer GATv2 for node classification.

    Architecture:
        GATv2Conv(in,    hid, heads=H, concat=True)  -> BN -> ELU -> Dropout
        GATv2Conv(hid*H, hid, heads=H, concat=True)  -> BN -> ELU -> Dropout  (num_layers-2 times)
        GATv2Conv(hid*H, out, heads=1, concat=False)  # mean over heads, no activation

    Intermediate layers use concat=True so each head learns an independent
    feature subspace (output dim = hid * H).
    The output layer uses a single head with concat=False to produce a fixed
    [num_nodes, out_channels] tensor regardless of how many heads are used.
    """
    def __init__(self, in_channels: int, out_channels: int, cfg: Config):
        super().__init__()
        self.dropout = cfg.dropout
        hid, H       = cfg.hidden_channels, cfg.heads

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Input layer: in -> hid*H
        self.convs.append(
            GATv2Conv(in_channels, hid, heads=H,
                      dropout=cfg.attn_dropout, concat=True)
        )
        self.bns.append(nn.BatchNorm1d(hid * H))

        # Hidden layers: hid*H -> hid*H
        for _ in range(cfg.num_layers - 2):
            self.convs.append(
                GATv2Conv(hid * H, hid, heads=H,
                          dropout=cfg.attn_dropout, concat=True)
            )
            self.bns.append(nn.BatchNorm1d(hid * H))

        # Output layer: hid*H -> out  (1 head, mean aggregation)
        self.convs.append(
            GATv2Conv(hid * H, out_channels, heads=1,
                      dropout=cfg.attn_dropout, concat=False)
        )

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)                    # ELU is standard activation for GAT
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ── 3. Training ─────────────────────────────────────────────────────────────────
def train(cfg: Config, data, model, train_idx, num_nodes):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Mask: training nodes that actually have a label (label >= 0)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=cfg.device)
    train_mask[train_idx] = True
    valid_train = train_mask & (data.y >= 0)

    best_loss, best_state = float('inf'), None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[valid_train], data.y[valid_train])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss - 1e-6:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % cfg.log_every == 0:
            model.eval()
            with torch.no_grad():
                acc = (out[valid_train].argmax(1) == data.y[valid_train]).float().mean().item()
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {acc:.4f} | Best loss: {best_loss:.4f}")

        if no_improve >= cfg.patience and epoch > cfg.min_epochs:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


# ── 4. Inference & save ────────────────────────────────────────────────────────
def predict_and_save(cfg: Config, model, data, npz):
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).argmax(dim=1).cpu().numpy()

    out_df = pd.DataFrame({
        'test_idx': npz['test_idx'],
        'label':    preds[npz['test_idx']],
    })
    out_df.to_csv(cfg.output_path, index=False)

    print(f"\n✓ Saved {len(out_df)} predictions to {cfg.output_path}")
    print("\nFirst 10 rows:")
    print(out_df.head(10).to_string(index=False))
    print("\nPredicted class distribution:")
    print(out_df['label'].value_counts().sort_index().to_string())


# ── 5. Main ─────────────────────────────────────────────────────────────────────
def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    print(f"Device: {cfg.device}")
    data, npz, num_nodes, num_feats, num_classes, train_idx = load_data(cfg)
    data = data.to(cfg.device)

    model = GAT(in_channels=num_feats, out_channels=num_classes, cfg=cfg).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.num_layers}-layer GATv2 | "
          f"hidden={cfg.hidden_channels} × heads={cfg.heads} "
          f"(dim={cfg.hidden_channels * cfg.heads}) | "
          f"params={total_params:,}")

    model = train(cfg, data, model, train_idx, num_nodes)
    predict_and_save(cfg, model, data, npz)


if __name__ == '__main__':
    main()
