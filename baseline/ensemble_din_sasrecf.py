# -*- coding: utf-8 -*-
"""
ensemble_din_sasrecf.py -- Score-level ensemble of a trained DIN ranker and a
trained SASRecF ranker for the same next-item recommendation task.

Both din_model.py and sasrecf_model.py train independently and each save a
checkpoint (din_best.pt / sasrecf_best.pt) containing the FULL state_dict of
their respective Ranker wrapper (DINRanker / SASRecFRanker). This script:

  1. Dynamically imports both training scripts as modules (by file path, so
     it doesn't care what you've named them -- pass --din_script /
     --sasrecf_script if they're not named din_model.py / sasrecf_model.py).
  2. Rebuilds the SAME data bundle (encoders, user/item feature tables) that
     both models were trained against, using load_data() from either module
     (the two scripts' load_data() implementations are functionally
     identical, so either one fitting the same CSVs reproduces the exact
     same uid/iid -> index mappings the checkpoints expect).
  3. Reconstructs DINRanker / SASRecFRanker with each script's own default
     Config() (so architecture dims match what was actually trained: note
     DIN and SASRecF may use a different max_seq_len, which is fine -- it
     only affects how each model's own Dataset pads/truncates history, not
     the shared item/user vocabularies), loads the checkpoint weights, sets
     eval mode.
  4. Reconstructs the EXACT SAME validation split each training script used
     internally (same `frac=1.0` shuffle with the same seed, same val_frac),
     so DIN and SASRecF are evaluated on the identical set of (uid, target)
     rows -- this is what makes a row-by-row score fusion valid.
  5. Computes full-catalog scores from each model separately, evaluates each
     model's own NDCG@10 as a sanity-check baseline, then fuses the two
     score matrices in two ways and reports NDCG@10 for each:
       a) weighted average of PER-ROW Z-SCORE NORMALIZED scores, swept over
          alpha in [0, 1] (alpha=1 -> pure DIN, alpha=0 -> pure SASRecF) so
          you can see the full curve and pick the best blend.
       b) reciprocal-rank fusion (RRF) -- score-scale-free, combines based on
          each model's rank position rather than raw score magnitude.
  6. Optionally (--predict) re-runs the same fusion (with a fixed --alpha or
     --fusion rrf) over test.csv and writes submission.csv in the same
     format the two training scripts use.

Why z-score normalization before averaging?
---------------------------------------------
DIN's and SASRecF's logits come from two independently-trained dot-product
heads with no shared scale -- one might routinely produce logits in [-5, 5]
and the other in [-50, 50]. Averaging raw scores would let whichever model
happens to have larger magnitude dominate the sum regardless of which model
is actually more confident/correct. Standardizing each model's score row to
zero mean / unit variance before combining puts both models on a comparable
footing, the same trick used in most production-style score-level
ensembling.

A note on how DIN and SASRecF are scored differently here
-----------------------------------------------------------
DIN's forward pass is candidate-conditioned: it needs an item id to attend
against, so -- exactly as inside din_model.py's own eval loop -- we stand in
the most-recent history item as that candidate (never the true target, which
would leak the label). SASRecF's forward pass takes NO candidate at all: its
causal self-attention encoder summarizes the history into a single
unconditional interest vector, and the SAME forward pass is used at both
training and inference time (see sasrecf_model.py for why this removes the
BST-style "stand-in candidate" requirement entirely). This script accounts
for that by dispatching each model through its own small forward-wrapper
(`_din_forward` / `_sasrecf_forward`) rather than assuming a single shared
calling convention.

Usage
-----
python ensemble_din_sasrecf.py \
    --din_script din_model_2.py --din_ckpt ./out_din/din_best.pt \
    --sasrecf_script sasrecf_model.py --sasrecf_ckpt ./out_sasrecf/sasrecf_best.pt \
    --data_dir /path/to/data --out_dir ./ensemble_out

Add --predict to also write submission.csv from test.csv using the best
alpha found on the validation sweep (or --fusion rrf / --alpha <fixed value>
to skip the sweep and force a specific fusion).
"""

import os
import sys
import argparse
import importlib.util

import numpy as np
import torch
from torch.utils.data import DataLoader

PAD_IDX = 0


# ========================================================================================
# 0. Dynamic import of the two training scripts
# ========================================================================================

def import_module_from_path(module_name, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find script: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # so the module's own internal imports resolve
    spec.loader.exec_module(module)
    return module


# ========================================================================================
# 1. Rebuilding the shared data bundle + the EXACT validation split each
#    training script used internally
# ========================================================================================

def build_bundle(din_mod, data_dir):
    """Either module's load_data() reproduces the same uid/iid -> index
    mappings as long as it's pointed at the same CSV files (fitting order is
    purely a function of row order in the CSVs), so we just use DIN's."""
    cfg = din_mod.Config()
    cfg.data_dir = data_dir
    return din_mod.load_data(cfg)


def split_val_df(bundle, seed, val_frac):
    """Mirrors the exact split logic inside both din_model.py's and
    sasrecf_model.py's train_model(): same shuffle seed + same val_frac on
    the same train_df produces the same set of (uid, target_iid, ...) rows
    in the same order, which is what lets us fuse DIN's and SASRecF's
    per-row scores."""
    train_full = bundle.train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(train_full) * val_frac)
    val_df = train_full.iloc[:n_val].reset_index(drop=True)
    return val_df


# ========================================================================================
# 2. Loading each model from its checkpoint
# ========================================================================================

def load_din_model(din_mod, bundle, ckpt_path, device):
    cfg = din_mod.Config()  # architecture must match training-time defaults
    user_feat_vocabs = [len(bundle.u_encoders[c]) for c in bundle.u_cat_cols]
    item_feat_vocabs = [len(bundle.i_encoders[c]) for c in bundle.i_cat_cols]
    model = din_mod.DINRanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
    model.set_feature_tables(
        torch.tensor(bundle.user_feat, dtype=torch.long),
        torch.tensor(bundle.item_feat, dtype=torch.long),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.din.user_feat_table = model.din.user_feat_table.to(device)
    model.din.item_feat_table = model.din.item_feat_table.to(device)
    model.eval()
    return model, cfg


def load_sasrecf_model(sasrecf_mod, bundle, ckpt_path, device):
    cfg = sasrecf_mod.Config()  # architecture must match training-time defaults
    user_feat_vocabs = [len(bundle.u_encoders[c]) for c in bundle.u_cat_cols]
    item_feat_vocabs = [len(bundle.i_encoders[c]) for c in bundle.i_cat_cols]
    model = sasrecf_mod.SASRecFRanker(bundle.n_users, bundle.n_items, user_feat_vocabs, item_feat_vocabs, cfg)
    model.set_feature_tables(
        torch.tensor(bundle.user_feat, dtype=torch.long),
        torch.tensor(bundle.item_feat, dtype=torch.long),
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.sasrecf.user_feat_table = model.sasrecf.user_feat_table.to(device)
    model.sasrecf.item_feat_table = model.sasrecf.item_feat_table.to(device)
    model.eval()
    return model, cfg


# ========================================================================================
# 3. Per-model full-catalog scoring
#
# DIN and SASRecF have DIFFERENT forward() calling conventions:
#   - DIN.forward(uid, hist, hist_cnt, hist_len, cand)   -- candidate-conditioned,
#     mirrors din_model.py's own eval loop by standing in the most-recent
#     history item as the candidate (never the unknown true target).
#   - SASRecF.forward(uid, hist, hist_cnt, hist_len)     -- no candidate at all;
#     its causal self-attention encoder already summarizes the whole history
#     into a single interest vector, so the SAME call is used at train and
#     eval time (see sasrecf_model.py).
# `score_batches` stays generic by taking a small `forward_fn` that knows how
# to call a specific model; each model gets its own wrapper below.
# ========================================================================================

def _din_forward(model, uid_idx, hist, hist_cnt, hist_len, device):
    batch_idx = torch.arange(hist.size(0), device=device)
    last_valid_pos = hist.size(1) - 1
    last_item = hist[batch_idx, last_valid_pos]
    return model(uid_idx, hist, hist_cnt, hist_len, last_item)


def _sasrecf_forward(model, uid_idx, hist, hist_cnt, hist_len, device):
    return model(uid_idx, hist, hist_cnt, hist_len)


@torch.no_grad()
def score_batches(model, loader, item_vecs_all, device, forward_fn):
    """Yields (uid_idx_np, target_np, scores_np[B, n_items]) per batch, with
    PAD masked out. `forward_fn(model, uid_idx, hist, hist_cnt, hist_len,
    device) -> user_vec` encapsulates whatever calling convention the given
    model needs (see _din_forward / _sasrecf_forward above)."""
    for batch in loader:
        uid_idx = batch["uid_idx"].to(device)
        hist = batch["hist"].to(device)
        hist_cnt = batch["hist_cnt"].to(device)
        hist_len = batch["hist_len"].to(device)
        target = batch["target"].to(device) if "target" in batch else None

        user_vec = forward_fn(model, uid_idx, hist, hist_cnt, hist_len, device)
        scores = model.score_against_catalog(user_vec, item_vecs_all)
        scores[:, PAD_IDX] = -1e9

        yield (
            uid_idx.cpu().numpy(),
            target.cpu().numpy() if target is not None else None,
            scores.cpu().numpy(),
        )


def collect_all_scores(model, df, uid_enc, iid_enc, max_seq_len, dataset_cls, collate_fn,
                        item_vecs_all, device, batch_size, forward_fn, has_target=True):
    """Runs the full dataframe through one model and returns (uids[N], targets[N] or
    None, scores[N, n_items]) with row order preserved (shuffle=False)."""
    ds = dataset_cls(df, uid_enc, iid_enc, max_len=max_seq_len, has_target=has_target)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_uids, all_targets, all_scores = [], [], []
    for uid_np, target_np, scores_np in score_batches(model, loader, item_vecs_all, device, forward_fn):
        all_uids.append(uid_np)
        all_scores.append(scores_np)
        if target_np is not None:
            all_targets.append(target_np)

    uids = np.concatenate(all_uids)
    scores = np.concatenate(all_scores, axis=0)
    targets = np.concatenate(all_targets) if all_targets else None
    return uids, targets, scores


# ========================================================================================
# 4. Score fusion
# ========================================================================================

def zscore_rows(scores):
    """Per-row standardization: zero mean / unit std along the item axis,
    so two models with different score scales become comparable before
    being averaged."""
    mu = scores.mean(axis=1, keepdims=True)
    sd = scores.std(axis=1, keepdims=True) + 1e-9
    return (scores - mu) / sd


def weighted_fuse(scores_a, scores_b, alpha):
    """alpha=1.0 -> pure model A, alpha=0.0 -> pure model B. Both inputs
    should already be row-normalized (see zscore_rows) for this to be a fair
    blend rather than one model dominating by raw magnitude."""
    return alpha * scores_a + (1.0 - alpha) * scores_b


def rrf_fuse(scores_a, scores_b, k=60):
    """Reciprocal Rank Fusion: convert each model's per-row scores to ranks
    (1 = best), combine via sum of 1/(k + rank). Scale-free -- doesn't care
    whether one model's logits are bigger than the other's, only relative
    ordering within each model matters. k=60 is the standard RRF default."""
    rank_a = (-scores_a).argsort(axis=1).argsort(axis=1) + 1  # 1-indexed rank, best=1
    rank_b = (-scores_b).argsort(axis=1).argsort(axis=1) + 1
    return 1.0 / (k + rank_a) + 1.0 / (k + rank_b)


# ========================================================================================
# 5. NDCG@10 (same single-relevant-item formula used by both training scripts)
# ========================================================================================

def ndcg_at_k_matrix(scores, targets, k=10):
    """Vectorized NDCG@k for a [N, n_items] score matrix against a [N] true-item
    index array. Returns the mean NDCG@k over all N rows."""
    topk_idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    # re-sort just the top-k slice by actual score so rank order within top-k is correct
    row_scores = np.take_along_axis(scores, topk_idx, axis=1)
    order = np.argsort(-row_scores, axis=1)
    topk_idx = np.take_along_axis(topk_idx, order, axis=1)

    hit_pos = np.full(scores.shape[0], -1, dtype=np.int64)
    matches = (topk_idx == targets[:, None])
    hit_rows, hit_cols = np.where(matches)
    hit_pos[hit_rows] = hit_cols  # 0-indexed rank within top-k

    ndcg = np.zeros(scores.shape[0], dtype=np.float64)
    found = hit_pos >= 0
    ndcg[found] = 1.0 / np.log2(hit_pos[found] + 2)
    return float(ndcg.mean())


# ========================================================================================
# 6. Main
# ========================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--din_script", default="din_model.py")
    parser.add_argument("--sasrecf_script", default="sasrecf_model.py")
    parser.add_argument("--din_ckpt", default="./din_best.pt")
    parser.add_argument("--sasrecf_ckpt", default="./sasrecf_best.pt")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", default="./ensemble_out")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42, help="must match the seed used by both training scripts")
    parser.add_argument("--val_frac", type=float, default=0.1, help="must match val_frac used by both training scripts")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rrf_k", type=int, default=60)
    parser.add_argument("--fusion", choices=["sweep", "weighted", "rrf"], default="sweep",
                         help="'sweep': scan alpha in [0,1] for weighted fusion and report the best "
                              "(plus RRF for comparison). 'weighted'/'rrf': skip the sweep and use --alpha "
                              "or pure RRF directly (useful with --predict to lock in a chosen fusion).")
    parser.add_argument("--alpha", type=float, default=0.5, help="DIN weight when --fusion weighted")
    parser.add_argument("--predict", action="store_true", help="also score test.csv and write submission.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    din_mod = import_module_from_path("din_model_user", args.din_script)
    sasrecf_mod = import_module_from_path("sasrecf_model_user", args.sasrecf_script)

    bundle = build_bundle(din_mod, args.data_dir)
    print(f"[INFO] n_users={bundle.n_users}  n_items={bundle.n_items}")

    din_model, din_cfg = load_din_model(din_mod, bundle, args.din_ckpt, device)
    sasrecf_model, sasrecf_cfg = load_sasrecf_model(sasrecf_mod, bundle, args.sasrecf_ckpt, device)
    print(f"[INFO] DIN max_seq_len={din_cfg.max_seq_len}  SASRecF max_seq_len={sasrecf_cfg.max_seq_len}")

    val_df = split_val_df(bundle, args.seed, args.val_frac)
    print(f"[INFO] valid set size = {len(val_df)} (must match each script's own training-time split)")

    all_item_ids = torch.arange(bundle.n_items, device=device)
    with torch.no_grad():
        din_item_vecs = din_model.item_static_vec(all_item_ids)
        sasrecf_item_vecs = sasrecf_model.item_static_vec(all_item_ids)

    uids_din, targets_din, scores_din = collect_all_scores(
        din_model, val_df, bundle.uid_enc, bundle.iid_enc, din_cfg.max_seq_len,
        din_mod.DINDataset, din_mod.collate_fn, din_item_vecs, device, args.batch_size,
        forward_fn=_din_forward,
    )
    uids_sasrecf, targets_sasrecf, scores_sasrecf = collect_all_scores(
        sasrecf_model, val_df, bundle.uid_enc, bundle.iid_enc, sasrecf_cfg.max_seq_len,
        sasrecf_mod.SasRecDataset, sasrecf_mod.collate_fn, sasrecf_item_vecs, device, args.batch_size,
        forward_fn=_sasrecf_forward,
    )

    # sanity check: both loaders walked val_df with shuffle=False, so row i in
    # both score matrices must refer to the same (uid, target) -- if this ever
    # trips, val_df / encoders are out of sync between the two models.
    assert np.array_equal(uids_din, uids_sasrecf), \
        "uid order mismatch between DIN and SASRecF scoring passes -- check that both " \
        "models were trained on the same data_dir/seed/val_frac."
    assert np.array_equal(targets_din, targets_sasrecf), "target mismatch between DIN and SASRecF scoring passes."
    targets = targets_din

    ndcg_din = ndcg_at_k_matrix(scores_din, targets, k=args.topk)
    ndcg_sasrecf = ndcg_at_k_matrix(scores_sasrecf, targets, k=args.topk)
    print(f"[RESULT] DIN-only      valid_ndcg@{args.topk} = {ndcg_din:.4f}")
    print(f"[RESULT] SASRecF-only  valid_ndcg@{args.topk} = {ndcg_sasrecf:.4f}")

    z_din = zscore_rows(scores_din)
    z_sasrecf = zscore_rows(scores_sasrecf)

    best_alpha, best_ndcg = None, -1.0
    if args.fusion == "sweep":
        print(f"[INFO] sweeping weighted-fusion alpha (1.0=pure DIN, 0.0=pure SASRecF)...")
        for alpha in np.round(np.arange(0.0, 1.01, 0.1), 2):
            fused = weighted_fuse(z_din, z_sasrecf, alpha)
            ndcg = ndcg_at_k_matrix(fused, targets, k=args.topk)
            marker = ""
            if ndcg > best_ndcg:
                best_ndcg, best_alpha = ndcg, alpha
                marker = "  <- best so far"
            print(f"    alpha={alpha:.1f}  valid_ndcg@{args.topk}={ndcg:.4f}{marker}")

        rrf_scores = rrf_fuse(scores_din, scores_sasrecf, k=args.rrf_k)
        ndcg_rrf = ndcg_at_k_matrix(rrf_scores, targets, k=args.topk)
        print(f"[RESULT] best weighted fusion: alpha={best_alpha:.1f}  valid_ndcg@{args.topk}={best_ndcg:.4f}")
        print(f"[RESULT] RRF fusion (k={args.rrf_k})  valid_ndcg@{args.topk}={ndcg_rrf:.4f}")

        if ndcg_rrf > best_ndcg:
            print(f"[INFO] RRF beat the best weighted alpha -- consider --fusion rrf for --predict.")
    elif args.fusion == "weighted":
        fused = weighted_fuse(z_din, z_sasrecf, args.alpha)
        best_ndcg = ndcg_at_k_matrix(fused, targets, k=args.topk)
        best_alpha = args.alpha
        print(f"[RESULT] weighted fusion alpha={args.alpha:.2f}  valid_ndcg@{args.topk}={best_ndcg:.4f}")
    else:  # rrf
        rrf_scores = rrf_fuse(scores_din, scores_sasrecf, k=args.rrf_k)
        best_ndcg = ndcg_at_k_matrix(rrf_scores, targets, k=args.topk)
        print(f"[RESULT] RRF fusion (k={args.rrf_k})  valid_ndcg@{args.topk}={best_ndcg:.4f}")

    print(f"\n[SUMMARY] DIN-only={ndcg_din:.4f}  SASRecF-only={ndcg_sasrecf:.4f}  "
          f"Ensemble({args.fusion})={best_ndcg:.4f}  "
          f"(improvement over best single model: {best_ndcg - max(ndcg_din, ndcg_sasrecf):+.4f})")

    if not args.predict:
        return

    # -------- inference on test.csv with the chosen fusion --------
    test_df = bundle.test_df
    if test_df is None:
        print("[WARN] no test.csv found, skipping prediction.")
        return

    uids_din_t, _, scores_din_t = collect_all_scores(
        din_model, test_df, bundle.uid_enc, bundle.iid_enc, din_cfg.max_seq_len,
        din_mod.DINDataset, din_mod.collate_fn, din_item_vecs, device, args.batch_size,
        forward_fn=_din_forward, has_target=False,
    )
    uids_sasrecf_t, _, scores_sasrecf_t = collect_all_scores(
        sasrecf_model, test_df, bundle.uid_enc, bundle.iid_enc, sasrecf_cfg.max_seq_len,
        sasrecf_mod.SasRecDataset, sasrecf_mod.collate_fn, sasrecf_item_vecs, device, args.batch_size,
        forward_fn=_sasrecf_forward, has_target=False,
    )
    assert np.array_equal(uids_din_t, uids_sasrecf_t), "uid order mismatch on test.csv between DIN and SASRecF."

    if args.fusion == "rrf":
        fused_t = rrf_fuse(scores_din_t, scores_sasrecf_t, k=args.rrf_k)
    else:
        alpha = best_alpha if best_alpha is not None else args.alpha
        fused_t = weighted_fuse(zscore_rows(scores_din_t), zscore_rows(scores_sasrecf_t), alpha)

    fused_t[:, PAD_IDX] = -1e9
    topk_idx = np.argsort(-fused_t, axis=1)[:, :args.topk]

    rows = []
    for row in topk_idx:
        item_strs = [bundle.iid_enc.idx2value.get(int(i), "i000000") for i in row]
        rows.append(",".join(item_strs))

    out_df = __import__("pandas").DataFrame({"uid": test_df["uid"].tolist(), "prediction": rows})
    out_path = os.path.join(args.out_dir, "submission.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] ensemble predictions written to {out_path}")


if __name__ == "__main__":
    main()
