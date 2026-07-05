"""
GCNII node classifier using PyTorch Geometric.
Chen et al. 2020, "Simple and Deep Graph Convolutional Networks".

GCNII extends GCN with two techniques that together eliminate over-smoothing,
allowing the network to scale to 64+ layers:

  1. Initial residual connection:
        H^{(l+1)} = σ( ((1-α)·Â·H^{(l)} + α·H^{(0)}) · W^{(l)} )
     Every layer receives a fraction α of the very first embedding H^{(0)},
     preventing all node representations from collapsing to the same vector.

  2. Identity mapping (weight regularisation):
        W^{(l)} = (1-β)·I + β·Ω^{(l)}
     The weight matrix is initialised close to the identity matrix and
     regularised toward it throughout training, keeping each layer's
     transformation mild and stable.

PyG encodes both in GCN2Conv(channels, alpha, theta, layer):
  - alpha  : initial-residual mixing coefficient  (typical 0.1–0.2)
  - theta  : identity-mapping strength β = log(θ/l + 1)  (typical 0.5–1.5)
  - layer  : the layer index l, used to compute β per-layer

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
from torch_geometric.nn import GCN2Conv
from scipy.sparse import csr_matrix
import pandas as pd


# ── Config ─────────────────────────────────────────────────────────────────────
class Config:
    # Paths
    data_path   = 'A1.npz'
    output_path = 'predictions_gcnii.csv'

    # Model architecture
    hidden_channels = 128   # fixed width across ALL layers (GCN2Conv requires this)
    num_layers      = 16     # CPU default; GPU recommended: 16–64
                            # GCNII's key advantage: can go deep without over-smoothing

    # GCNII-specific hyperparameters
    alpha           = 0.1   # initial residual strength: fraction of H^{(0)} mixed in
                            # typical range 0.1–0.2; larger → stronger pull to input
    theta           = 0.5   # identity mapping strength: controls β = log(θ/l + 1)
                            # typical range 0.5–1.5; larger → weights stay closer to I

    # Training
    dropout        = 0.5
    lr             = 0.01
    weight_decay   = 5e-4
    epochs         = 1000
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


# ── 2. GCNII model ─────────────────────────────────────────────────────────────
class GCNII(nn.Module):
    """
    GCNII for node classification.

    Architecture:
        Linear(in → hid)                     # input projection → H^{(0)}
        GCN2Conv(hid, alpha, theta, l=1)  → BN → ReLU → Dropout
        GCN2Conv(hid, alpha, theta, l=2)  → BN → ReLU → Dropout
        ...
        GCN2Conv(hid, alpha, theta, l=L)  → BN → ReLU → Dropout
        Linear(hid → out)                    # classifier head

    Every GCN2Conv layer takes BOTH the current H^{(l)} and H^{(0)} as input.
    The layer index l is passed so each layer computes its own β = log(θ/l + 1),
    making early layers transform more aggressively and deep layers stay close
    to the identity — a natural depth-aware regularisation schedule.
    """
    def __init__(self, in_channels: int, out_channels: int, cfg: Config):
        super().__init__()
        self.dropout = cfg.dropout
        hid          = cfg.hidden_channels

        # Input projection: maps raw features into the fixed hidden dimension
        # H^{(0)} is produced here and reused by every GCN2Conv layer
        self.input_proj = nn.Linear(in_channels, hid)

        # Deep GCN2Conv stack — all layers share the same width (hid)
        # Each layer receives its 1-based index for per-layer β computation
        self.convs = nn.ModuleList([
            GCN2Conv(
                channels=hid,
                alpha=cfg.alpha,
                theta=cfg.theta,
                layer=l + 1,          # 1-indexed: layer 1 … num_layers
                shared_weights=True,  # share Ω across the two weight copies
                cached=False,
            )
            for l in range(cfg.num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hid) for _ in range(cfg.num_layers)
        ])

        # Classifier head
        self.classifier = nn.Linear(hid, out_channels)

    def forward(self, x, edge_index):
        # Input projection → H^{(0)}
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.input_proj(x))
        x_0 = x                       # H^{(0)}: held fixed throughout propagation

        # Deep propagation with initial residual + identity mapping
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, x_0, edge_index)   # GCN2Conv needs both H^{(l)} and H^{(0)}
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)


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

    model = GCNII(in_channels=num_feats, out_channels=num_classes, cfg=cfg).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.num_layers}-layer GCNII | "
          f"hidden={cfg.hidden_channels} | "
          f"alpha={cfg.alpha} | theta={cfg.theta} | "
          f"params={total_params:,}")

    model = train(cfg, data, model, train_idx, num_nodes)
    predict_and_save(cfg, model, data, npz)


if __name__ == '__main__':
    main()
