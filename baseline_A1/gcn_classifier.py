"""
GCN-based node classifier using PyTorch Geometric.
Uses GCNConv (Kipf & Welling 2017) for node classification.

Dataset format (A1.npz):
  adj_data / adj_indices / adj_indptr / adj_shape  -> CSR adjacency matrix
  attr_data / attr_indices / attr_indptr / attr_shape -> CSR feature matrix
  labels     : node labels (-1 = unlabeled)
  train_idx  : labeled training node indices
  test_idx   : test node indices to predict
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from scipy.sparse import csr_matrix
import pandas as pd


# ── 1. Load data ───────────────────────────────────────────────────────────────
npz = np.load('A1.npz', allow_pickle=True)

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
test_idx  = torch.tensor(npz['test_idx'],  dtype=torch.long)

num_nodes   = int(npz['adj_shape'][0])
num_feats   = int(npz['attr_shape'][1])
num_classes = int(labels[labels >= 0].max().item()) + 1

print(f"Nodes: {num_nodes} | Features: {num_feats} | Classes: {num_classes}")
print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")


# ── 2. Build PyG Data object ───────────────────────────────────────────────────
# Dense node feature matrix  [num_nodes, num_feats]
x = torch.tensor(attr.toarray(), dtype=torch.float32)

# Convert CSR adjacency to COO edge_index, then make undirected
cx = adj.tocoo()
edge_index = torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)   # add reverse
edge_index = torch.unique(edge_index, dim=1)                       # dedup

data = Data(x=x, edge_index=edge_index, y=labels, num_nodes=num_nodes)
print(f"Edges (undirected, deduped): {edge_index.shape[1]}")


# ── 3. GCN model ───────────────────────────────────────────────────────────────
class GCN(nn.Module):
    """
    Multi-layer GCN for node classification.

    Architecture:
        GCNConv(in  -> hid) -> BN -> ReLU -> Dropout
        GCNConv(hid -> hid) -> BN -> ReLU -> Dropout   (num_layers - 2 times)
        GCNConv(hid -> out)
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Output layer (no BN / activation)
        self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


# ── 4. Training setup ──────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

data  = data.to(device)
model = GCN(
    in_channels=num_feats,
    hidden_channels=128,
    out_channels=num_classes,
    num_layers=3,
    dropout=0.5,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

# Mask: training nodes that actually have a label (label >= 0)
train_mask  = torch.zeros(num_nodes, dtype=torch.bool, device=device)
train_mask[train_idx] = True
valid_train = train_mask & (data.y >= 0)

best_loss, best_state = float('inf'), None
patience, no_improve  = 30, 0


# ── 5. Training loop ───────────────────────────────────────────────────────────
EPOCHS = 500
for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[valid_train], data.y[valid_train])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    # Track best checkpoint
    if loss.item() < best_loss - 1e-6:
        best_loss  = loss.item()
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            acc = (out[valid_train].argmax(1) == data.y[valid_train]).float().mean().item()
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
              f"Train Acc: {acc:.4f} | Best loss: {best_loss:.4f}")

    if no_improve >= patience and epoch > 100:
        print(f"Early stopping at epoch {epoch}")
        break


# ── 6. Inference ───────────────────────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    preds = model(data.x, data.edge_index).argmax(dim=1).cpu().numpy()


# ── 7. Save predictions ────────────────────────────────────────────────────────
out_df = pd.DataFrame({
    'test_idx': npz['test_idx'],
    'label':    preds[npz['test_idx']],
})
out_df.to_csv('predictions.csv', index=False)

print(f"\n✓ Saved {len(out_df)} predictions to predictions.csv")
print("\nFirst 10 rows:")
print(out_df.head(10).to_string(index=False))
print("\nPredicted class distribution:")
print(out_df['label'].value_counts().sort_index().to_string())
