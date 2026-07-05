"""
GIN (Graph Isomorphism Network) node classifier using PyTorch Geometric.
Uses GINConv (Xu et al. 2019, "How Powerful are Graph Neural Networks?").

GIN aggregation:
    h_v^{(k)} = MLP^{(k)}( (1 + ε^{(k)}) · h_v^{(k-1)} + Σ_{u∈N(v)} h_u^{(k-1)} )

Key properties:
  - Theoretically as powerful as the Weisfeiler-Lehman graph isomorphism test
  - Uses an MLP (instead of a single linear layer) after aggregation,
    giving it strictly higher expressive power than GCN / SAGE / GAT
  - ε is a learnable parameter that controls the weight of the center node
    relative to its neighbors

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
from torch_geometric.nn import GINConv
from scipy.sparse import csr_matrix
import pandas as pd


# ── Config ─────────────────────────────────────────────────────────────────────
class Config:
    # Paths
    data_path   = 'A1.npz'
    output_path = 'predictions_gin.csv'

    # Model architecture
    hidden_channels = 128   # width of each GIN layer and inner MLP
    num_layers      = 3     # number of GINConv layers
    mlp_layers      = 2     # depth of the MLP inside each GINConv (>=2 required by theory)
    dropout         = 0.5
    train_eps       = True  # whether ε is learnable (True) or fixed at 0 (False)

    # Training
    lr             = 0.01
    weight_decay   = 5e-4
    epochs         = 500
    patience       = 30     # early-stop patience
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


# ── 2. MLP builder (used inside each GINConv) ──────────────────────────────────
def build_mlp(in_channels: int, out_channels: int, cfg: Config) -> nn.Sequential:
    """
    Build the MLP used inside a single GINConv layer.

    Structure (mlp_layers=2):
        Linear(in -> out) -> BN -> ReLU -> Linear(out -> out)

    Structure (mlp_layers=3):
        Linear(in -> out) -> BN -> ReLU
        Linear(out -> out) -> BN -> ReLU
        Linear(out -> out)

    Using >=2 layers is required for GIN to achieve its theoretical expressive
    power — a single linear layer degenerates to a GCN-like model.
    """
    layers = []
    for i in range(cfg.mlp_layers):
        in_c  = in_channels if i == 0 else out_channels
        out_c = out_channels
        layers.append(nn.Linear(in_c, out_c))
        if i < cfg.mlp_layers - 1:          # no BN/ReLU after the last linear
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


# ── 3. GIN model ───────────────────────────────────────────────────────────────
class GIN(nn.Module):
    """
    Multi-layer GIN for node classification.

    Architecture:
        GINConv( MLP(in  -> hid) ) -> BN -> ReLU -> Dropout
        GINConv( MLP(hid -> hid) ) -> BN -> ReLU -> Dropout  (num_layers-2 times)
        GINConv( MLP(hid -> hid) ) -> linear head -> out      (output layer)

    The classifier head is a separate Linear applied after the last GINConv,
    keeping the graph convolution and the prediction head decoupled.
    """
    def __init__(self, in_channels: int, out_channels: int, cfg: Config):
        super().__init__()
        self.dropout = cfg.dropout
        hid          = cfg.hidden_channels

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Input layer: in -> hid
        self.convs.append(
            GINConv(build_mlp(in_channels, hid, cfg), train_eps=cfg.train_eps)
        )
        self.bns.append(nn.BatchNorm1d(hid))

        # Hidden layers: hid -> hid
        for _ in range(cfg.num_layers - 1):
            self.convs.append(
                GINConv(build_mlp(hid, hid, cfg), train_eps=cfg.train_eps)
            )
            self.bns.append(nn.BatchNorm1d(hid))

        # Classifier head: hid -> out (plain linear, no graph conv)
        self.classifier = nn.Linear(hid, out_channels)

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


# ── 4. Training ─────────────────────────────────────────────────────────────────
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


# ── 5. Inference & save ────────────────────────────────────────────────────────
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


# ── 6. Main ─────────────────────────────────────────────────────────────────────
def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    print(f"Device: {cfg.device}")
    data, npz, num_nodes, num_feats, num_classes, train_idx = load_data(cfg)
    data = data.to(cfg.device)

    model = GIN(in_channels=num_feats, out_channels=num_classes, cfg=cfg).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.num_layers}-layer GIN | "
          f"hidden={cfg.hidden_channels} | "
          f"mlp_layers={cfg.mlp_layers} | "
          f"train_eps={cfg.train_eps} | "
          f"params={total_params:,}")

    model = train(cfg, data, model, train_idx, num_nodes)
    predict_and_save(cfg, model, data, npz)


if __name__ == '__main__':
    main()
