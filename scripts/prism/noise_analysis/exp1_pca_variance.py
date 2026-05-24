#!/usr/bin/env python3
"""
Experiment 1 (Revised): PCA Cumulative Explained Variance
=========================================================

Proves the "information density" hypothesis WITHOUT training any model:
  - Text (768D): massive redundancy — few PCs explain most variance
  - Collab (64D): dense  — every dimension carries non-trivial variance

This directly motivates Stage 1's IDE (Information Density Equalization):
  why we must compress text from 768D → 128D while preserving collab structure.

Output: exp1_pca_results.json
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
OUTPUT_DIR = "scripts/prism/noise_analysis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "exp1_pca_results.json")

PCA_MAX_COMPONENTS = 256   # cap for efficiency (enough to show convergence)

print("=" * 70)
print("Experiment 1: PCA Cumulative Explained Variance")
print("=" * 70)

# ── Load data ───────────────────────────────────────────────────────────
print("\nLoading embeddings...")
emb_df = pd.read_parquet(os.path.join(DATA_DIR, "item_emb.parquet"))
text_embs = np.stack([np.array(e, dtype=np.float32) for e in emb_df["embedding"]])
print(f"  Text: {text_embs.shape}")

collab_path = os.path.join(DATA_DIR, "lightgcn/item_embeddings_collab.npy")
collab_all = np.load(collab_path).astype(np.float32)
item_ids = emb_df["ItemID"].values
collab_embs = np.stack([collab_all[iid] for iid in item_ids])
print(f"  Collab: {collab_embs.shape}")

# ── PCA on Text ─────────────────────────────────────────────────────────
print("\nRunning PCA on Text (768D)...")
t0 = time.time()
pca_text = PCA(n_components=min(PCA_MAX_COMPONENTS, text_embs.shape[1]))
pca_text.fit(text_embs)
text_cumvar = np.cumsum(pca_text.explained_variance_ratio_)
text_time = time.time() - t0
print(f"  Done in {text_time:.1f}s")

# ── PCA on Collab ───────────────────────────────────────────────────────
print("Running PCA on Collab (64D)...")
t0 = time.time()
pca_collab = PCA(n_components=min(PCA_MAX_COMPONENTS, collab_embs.shape[1]))
pca_collab.fit(collab_embs)
collab_cumvar = np.cumsum(pca_collab.explained_variance_ratio_)
collab_time = time.time() - t0
print(f"  Done in {collab_time:.1f}s")

# ── Key metrics ─────────────────────────────────────────────────────────
def components_for_variance(cumvar, threshold):
    idx = np.searchsorted(cumvar, threshold)
    return int(idx) + 1

text_90 = components_for_variance(text_cumvar, 0.90)
text_95 = components_for_variance(text_cumvar, 0.95)
text_99 = components_for_variance(text_cumvar, 0.99)
collab_90 = components_for_variance(collab_cumvar, 0.90)
collab_95 = components_for_variance(collab_cumvar, 0.95)
collab_99 = components_for_variance(collab_cumvar, 0.99)

print("\n" + "-" * 60)
print(f"{'Threshold':<12} {'Text (768D)':>18} {'Collab (64D)':>18}")
print("-" * 60)
print(f"{'90%':<12} {f'{text_90} dims ({text_90/768*100:.1f}%)':>18} {f'{collab_90} dims ({collab_90/64*100:.1f}%)':>18}")
print(f"{'95%':<12} {f'{text_95} dims ({text_95/768*100:.1f}%)':>18} {f'{collab_95} dims ({collab_95/64*100:.1f}%)':>18}")
print(f"{'99%':<12} {f'{text_99} dims ({text_99/768*100:.1f}%)':>18} {f'{collab_99} dims ({collab_99/64*100:.1f}%)':>18}")
print("-" * 60)

# ── Save ────────────────────────────────────────────────────────────────
results = {
    "description": "PCA cumulative explained variance — no model training",
    "text": {
        "original_dim": int(text_embs.shape[1]),
        "cumulative_variance": text_cumvar.tolist(),
        "components_for_90pct": text_90,
        "components_for_95pct": text_95,
        "components_for_99pct": text_99,
        "pca_time_s": round(text_time, 1),
    },
    "collab": {
        "original_dim": int(collab_embs.shape[1]),
        "cumulative_variance": collab_cumvar.tolist(),
        "components_for_90pct": collab_90,
        "components_for_95pct": collab_95,
        "components_for_99pct": collab_99,
        "pca_time_s": round(collab_time, 1),
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {OUTPUT_FILE}")
print("Done.")
