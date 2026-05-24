#!/usr/bin/env python3
"""
Experiment 2: Popularity-Stratified Cross-Modal Neighbor Agreement
===================================================================

Proves the "reliability noise" hypothesis WITHOUT training any model:

  For each item, find its top-K neighbors in text space vs collab space,
  then measure how much the two neighbor sets overlap (Jaccard).

  - Popular items: high overlap → modalities agree, collab is reliable
  - Cold items:     low overlap  → collab diverges from text (unreliable)

This directly motivates Stage 1's MCD (Mutual Cross-modal Denoising):
  we need cross-modal consistency checking because collab features
  are unreliable for long-tail items.

Output: exp2_neighbor_agreement.json
"""

import os, sys, json, time
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
OUTPUT_DIR = "scripts/prism/noise_analysis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "exp2_neighbor_agreement.json")

K = 20                        # top-K neighbors
NUM_BUCKETS = 8               # log-spaced popularity buckets
MIN_POP_FOR_ITEM = 1         # skip zero-interaction items

print("=" * 70)
print("Experiment 2: Cross-Modal Neighbor Agreement")
print("=" * 70)

# ── Load embeddings ─────────────────────────────────────────────────────
print("\nLoading embeddings...")
emb_df = pd.read_parquet(os.path.join(DATA_DIR, "item_emb.parquet"))
item_ids = emb_df["ItemID"].values
id_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}
n_items = len(item_ids)

text_embs = np.stack([np.array(e, dtype=np.float32) for e in emb_df["embedding"]])
t_norm = np.linalg.norm(text_embs, axis=1, keepdims=True)
text_normed = text_embs / (t_norm + 1e-8)

collab_path = os.path.join(DATA_DIR, "lightgcn/item_embeddings_collab.npy")
collab_all = np.load(collab_path).astype(np.float32)
collab_embs = np.stack([collab_all[iid] for iid in item_ids])
c_norm = np.linalg.norm(collab_embs, axis=1, keepdims=True)
collab_normed = collab_embs / (c_norm + 1e-8)

# ── True popularity from train ──────────────────────────────────────────
print("Computing item popularity...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
pop = np.zeros(n_items, dtype=np.int32)
for _, row in train_df.iterrows():
    for h in row["history"]:
        idx = id_to_idx.get(int(h), -1)
        if idx >= 0: pop[idx] += 1
    idx = id_to_idx.get(int(row["target"]), -1)
    if idx >= 0: pop[idx] += 1

# ── Build log-spaced popularity buckets (by ITEM, not sample) ───────────
valid_mask = pop >= MIN_POP_FOR_ITEM
valid_pop = pop[valid_mask]
log_pop = np.log10(valid_pop + 1)
edges = np.percentile(log_pop, np.linspace(0, 100, NUM_BUCKETS + 1))
edges[0] -= 0.01; edges[-1] += 0.01

bucket_of_item = np.full(n_items, -1, dtype=int)
bucket_labels = []
for b in range(NUM_BUCKETS):
    in_bucket = (log_pop >= edges[b]) & (log_pop < edges[b + 1])
    full = np.zeros(n_items, dtype=bool)
    full[valid_mask] = in_bucket
    bucket_of_item[full] = b
    lo = int(10 ** edges[b]); hi = int(10 ** edges[b + 1])
    bucket_labels.append(f"B{b} [{lo}-{hi}]")
    print(f"  Bucket {b} [{lo}-{hi}]: {full.sum()} items")

# ── Compute neighbor overlap per item ───────────────────────────────────
print(f"\nComputing top-{K} neighbors for all {n_items} items...")
t0 = time.time()

# Batch computation: text similarity matrix (n_items x n_items is ~600MB, too large)
# Instead: for each item, compute its top-K neighbors via argpartition
def compute_neighbor_set(emb_normed, k):
    """Return (n_items, k) array of neighbor indices for each item (excl. self)."""
    n = emb_normed.shape[0]
    # Process in chunks to avoid OOM
    chunk = 2000
    neighbors = np.zeros((n, k), dtype=np.int32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        chunk_emb = emb_normed[start:end]
        sim = np.dot(chunk_emb, emb_normed.T)  # (chunk, n)
        sim[:, np.arange(start, end)] = -2.0   # exclude self
        topk_idx = np.argpartition(-sim, k, axis=1)[:, :k]
        neighbors[start:end] = topk_idx
    return neighbors

text_neighbors = compute_neighbor_set(text_normed, K)
collab_neighbors = compute_neighbor_set(collab_normed, K)

# ── Compute Jaccard overlap per item ────────────────────────────────────
jaccard = np.zeros(n_items, dtype=np.float32)
for i in range(n_items):
    t_set = set(text_neighbors[i].tolist())
    c_set = set(collab_neighbors[i].tolist())
    inter = len(t_set & c_set)
    union = len(t_set | c_set)
    jaccard[i] = inter / union if union > 0 else 0.0

# ── Aggregate by popularity bucket ──────────────────────────────────────
bucket_jaccard = []
bucket_sizes = []
for b in range(NUM_BUCKETS):
    mask = bucket_of_item == b
    bucket_sizes.append(int(mask.sum()))
    if mask.sum() > 0:
        bucket_jaccard.append(float(jaccard[mask].mean()))
    else:
        bucket_jaccard.append(0.0)

# Also compute recall@K style: for each item, what fraction of text neighbors
# are also in collab neighbors (precision), and vice versa
text_precision = np.zeros(n_items, dtype=np.float32)  # text neighbors also in collab
collab_precision = np.zeros(n_items, dtype=np.float32)  # collab neighbors also in text
for i in range(n_items):
    t_set = set(text_neighbors[i].tolist())
    c_set = set(collab_neighbors[i].tolist())
    text_precision[i] = len(t_set & c_set) / K
    collab_precision[i] = len(c_set & t_set) / K

bucket_tp = []
bucket_cp = []
for b in range(NUM_BUCKETS):
    mask = bucket_of_item == b
    bucket_tp.append(float(text_precision[mask].mean()) if mask.sum() > 0 else 0.0)
    bucket_cp.append(float(collab_precision[mask].mean()) if mask.sum() > 0 else 0.0)

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")

# ── Results ─────────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print(f"{'Bucket':<18} {'Items':>7} {'Jaccard':>9} {'T→C Prec':>10} {'C→T Prec':>10}")
print("-" * 70)
for b in range(NUM_BUCKETS):
    print(f"{bucket_labels[b]:<18} {bucket_sizes[b]:>7} {bucket_jaccard[b]:>9.4f} "
          f"{bucket_tp[b]:>10.4f} {bucket_cp[b]:>10.4f}")

cold_avg = np.mean(bucket_jaccard[:3]) if NUM_BUCKETS >= 3 else bucket_jaccard[0]
hot_avg = np.mean(bucket_jaccard[-3:]) if NUM_BUCKETS >= 3 else bucket_jaccard[-1]
print("-" * 70)
print(f"Cold (B0-B2) mean Jaccard: {cold_avg:.4f}")
print(f"Hot (B5-B7) mean Jaccard:  {hot_avg:.4f}")
print(f"Hot/Cold ratio:            {hot_avg/cold_avg:.2f}x")

# ── Save ────────────────────────────────────────────────────────────────
results = {
    "description": "Cross-modal neighbor agreement by popularity — no training",
    "config": {"K": K, "NUM_BUCKETS": NUM_BUCKETS, "bucketing": "log-spaced item popularity"},
    "buckets": {str(b): {
        "label": bucket_labels[b], "n_items": bucket_sizes[b],
    } for b in range(NUM_BUCKETS)},
    "jaccard": [round(v, 6) for v in bucket_jaccard],
    "text_precision": [round(v, 6) for v in bucket_tp],
    "collab_precision": [round(v, 6) for v in bucket_cp],
    "cold_mean_jaccard": round(cold_avg, 6),
    "hot_mean_jaccard": round(hot_avg, 6),
    "hot_cold_ratio": round(hot_avg / max(cold_avg, 1e-8), 4),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Done.")
