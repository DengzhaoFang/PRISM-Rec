#!/usr/bin/env python3
"""
Experiment 2b (Supplementary): Zero-Shot Mean-Pooling Retrieval
================================================================

Validates retrieval quality without training.
Key fixes vs. previous version:
  1. pop[id_to_idx[iid]] instead of pop[iid]  (index vs ItemID bug)
  2. History items excluded from candidate set (standard IR evaluation)
  3. Unified cosine similarity (fair Text vs Collab comparison)

Output: exp2_retrieval_results.json
"""

import os, sys, json, time
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
OUTPUT_DIR = "scripts/prism/noise_analysis"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "exp2_retrieval_results.json")

K = 10
MAX_HIST_LEN = 20

print("=" * 70)
print("Experiment 2b: Zero-Shot Retrieval (bug-fixed)")
print("=" * 70)

# ── Load ────────────────────────────────────────────────────────────────
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

# Popularity (FIXED: index by id_to_idx, not raw ItemID)
print("Computing popularity...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
pop = np.zeros(n_items, dtype=np.int32)
for _, row in train_df.iterrows():
    for h in row["history"]:
        idx = id_to_idx.get(int(h), -1)
        if idx >= 0: pop[idx] += 1
    idx = id_to_idx.get(int(row["target"]), -1)
    if idx >= 0: pop[idx] += 1

# ── Test set ─────────────────────────────────────────────────────────────
print("Loading test set...")
test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
test_histories = []
test_targets = []
test_target_pop = []
for _, row in test_df.iterrows():
    hist = list(row["history"])[-MAX_HIST_LEN:]
    tgt = row["target"]
    tgt_idx = id_to_idx.get(int(tgt), -1)
    if tgt_idx < 0: continue
    hist_idx = [id_to_idx.get(int(h), -1) for h in hist]
    hist_idx = [h for h in hist_idx if h >= 0]
    if len(hist_idx) == 0: continue
    test_histories.append(hist_idx)
    test_targets.append(tgt_idx)
    test_target_pop.append(pop[tgt_idx])  # FIXED: pop indexed by tgt_idx (0..n_items-1)

n_test = len(test_targets)
test_target_pop = np.array(test_target_pop)
print(f"  Test samples: {n_test}")

# ── Log-spaced buckets by ITEM popularity (not test-sample-count) ────────
NUM_BUCKETS = 8
valid_pop = pop[pop >= 1]
log_pop = np.log10(valid_pop + 1)
edges = np.percentile(log_pop, np.linspace(0, 100, NUM_BUCKETS + 1))
edges[0] -= 0.01; edges[-1] += 0.01

log_test_pop = np.log10(test_target_pop + 1)
bucket_of_sample = np.full(n_test, -1, dtype=int)
bucket_labels = []
for b in range(NUM_BUCKETS):
    mask = (log_test_pop >= edges[b]) & (log_test_pop < edges[b + 1])
    bucket_of_sample[mask] = b
    lo = int(10 ** edges[b]); hi = int(10 ** edges[b + 1])
    bucket_labels.append(f"B{b} [{lo}-{hi}]")
    print(f"  Bucket {b} [{lo}-{hi}]: {mask.sum()} test samples")

# ── Retrieval (FIXED: history exclusion, unified cosine) ─────────────────
def retrieval(emb_normed):
    """Mean-pool history → cosine vs all candidates (excluding history items)."""
    hits = np.zeros(n_test, dtype=bool)
    for i in range(n_test):
        hist_idx = test_histories[i]
        user_vec = emb_normed[hist_idx].mean(axis=0)
        user_vec = user_vec / (np.linalg.norm(user_vec) + 1e-8)
        scores = np.dot(user_vec, emb_normed.T)
        scores[hist_idx] = -1e9      # FIXED: exclude history items
        topk = np.argpartition(-scores, K)[:K]
        hits[i] = test_targets[i] in topk
    return hits

print("\nRunning Text retrieval (cosine)...")
t0 = time.time()
text_hits = retrieval(text_normed)
text_time = time.time() - t0
text_recall = float(text_hits.mean())

print("Running Collab retrieval (cosine)...")
t0 = time.time()
collab_hits = retrieval(collab_normed)
collab_time = time.time() - t0
collab_recall = float(collab_hits.mean())

# ── Per-bucket recall ────────────────────────────────────────────────────
text_bucket = {}
collab_bucket = {}
for b in range(NUM_BUCKETS):
    m = bucket_of_sample == b
    text_bucket[b] = float(text_hits[m].mean()) if m.sum() > 0 else 0.0
    collab_bucket[b] = float(collab_hits[m].mean()) if m.sum() > 0 else 0.0

# ── Report ───────────────────────────────────────────────────────────────
print(f"\n  Text   Recall@{K}: {text_recall:.4f}  ({text_time:.1f}s)")
print(f"  Collab Recall@{K}: {collab_recall:.4f}  ({collab_time:.1f}s)")
print(f"\n{'Bucket':<18} {'Text':>8} {'Collab':>8}")
print("-" * 36)
for b in range(NUM_BUCKETS):
    print(f"{bucket_labels[b]:<18} {text_bucket[b]:>8.4f} {collab_bucket[b]:>8.4f}")

cold = [0, 1, 3]; hot = [5, 6, 7]
tc = np.mean([text_bucket[b] for b in cold]); th = np.mean([text_bucket[b] for b in hot])
cc = np.mean([collab_bucket[b] for b in cold]); ch = np.mean([collab_bucket[b] for b in hot])
print("-" * 36)
print(f"{'Cold/Hot ratio':<18} {tc/th:>8.3f} {cc/ch:>8.3f}")

# ── Save ────────────────────────────────────────────────────────────────
results = {
    "description": "Zero-shot retrieval with history exclusion + unified cosine",
    "config": {"K": K, "MAX_HIST_LEN": MAX_HIST_LEN, "similarity": "cosine"},
    "buckets": {str(b): {"label": bucket_labels[b]} for b in range(NUM_BUCKETS)},
    "text": {
        "recall@10": round(text_recall, 6),
        "bucket_recall": {str(b): round(v, 6) for b, v in text_bucket.items()},
        "cold_mean": round(tc, 6), "hot_mean": round(th, 6),
        "cold_hot_ratio": round(tc / max(th, 1e-8), 4),
    },
    "collab": {
        "recall@10": round(collab_recall, 6),
        "bucket_recall": {str(b): round(v, 6) for b, v in collab_bucket.items()},
        "cold_mean": round(cc, 6), "hot_mean": round(ch, 6),
        "cold_hot_ratio": round(cc / max(ch, 1e-8), 4),
    },
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUTPUT_FILE}")
print("Done.")
