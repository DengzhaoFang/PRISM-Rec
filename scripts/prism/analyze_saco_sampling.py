#!/usr/bin/env python3
"""
Analyze SACO (Sequence-Aware Contrastive Objective) sampling strategy.

Checks:
1. Co-occurrence graph statistics (degree distribution, density)
2. False negative rate in in-batch InfoNCE
3. Overlap between different anchors' co-occurrence sets
4. Simulates batch sampling and measures expected false negative rate
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/sid_tokenizer/prism'))

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
import random

random.seed(42)
np.random.seed(42)

# ============================================================
# Config: choose dataset
# ============================================================
# Use existing dataset for graph analysis (same train.parquet, same item IDs)
DATA_DIR = Path('/home/fangdengzhao/SID-GR/dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty')
BATCH_SIZE = 512
NUM_SIMULATED_BATCHES = 200

# ============================================================
# 1. Build co-occurrence graph (same logic as PRISMDataset)
# ============================================================
print("=" * 70)
print("1. Building co-occurrence graph...")
print("=" * 70)

train_seq_path = DATA_DIR / 'train.parquet'
df = pd.read_parquet(train_seq_path)
print(f"   Sequences: {len(df)}")

item_id_to_idx = {}
# Also load item IDs from embeddings
emb_path = DATA_DIR / 'item_emb.parquet'
emb_df = pd.read_parquet(emb_path)
for idx, item_id in enumerate(emb_df['ItemID'].values):
    item_id_to_idx[int(item_id)] = idx

n_items = len(item_id_to_idx)
print(f"   Unique items with embeddings: {n_items}")

cooc_graph = defaultdict(list)
window = 4

for _, row in df.iterrows():
    seq = list(row['history']) + [row['target']]
    seq = [item_id for item_id in seq if item_id in item_id_to_idx]
    for i in range(len(seq)):
        for j in range(i + 1, min(i + window + 1, len(seq))):
            a, b = seq[i], seq[j]
            if a != b:
                cooc_graph[a].append(b)
                cooc_graph[b].append(a)

# Remove items with no co-occurrences
cooc_graph = {k: v for k, v in cooc_graph.items() if len(v) > 0}
n_with_cooc = len(cooc_graph)
print(f"   Items with co-occurrences: {n_with_cooc} / {n_items} "
      f"({n_with_cooc/n_items*100:.1f}%)")
print(f"   Total edges: {sum(len(v) for v in cooc_graph.values())}")

# ============================================================
# 2. Degree distribution analysis
# ============================================================
print("\n" + "=" * 70)
print("2. Co-occurrence degree distribution")
print("=" * 70)

degrees = np.array([len(v) for v in cooc_graph.values()])
degrees_unique = np.array([len(set(v)) for v in cooc_graph.values()])

print(f"   Mean degree (with repeats):    {degrees.mean():.1f}")
print(f"   Median degree (with repeats):  {np.median(degrees):.1f}")
print(f"   Mean unique neighbors:          {degrees_unique.mean():.1f}")
print(f"   Min/Max unique neighbors:       {degrees_unique.min()}/{degrees_unique.max()}")
print(f"   Degree percentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"     P{p:2d}: {np.percentile(degrees_unique, p):6.0f}")

# ============================================================
# 3. False negative analysis: for each item, what % of its
#    co-occurring partners also co-occur with each other?
# ============================================================
print("\n" + "=" * 70)
print("3. Co-occurrence set overlap analysis (false negative risk)")
print("=" * 70)

# Sample items to estimate overlap
sample_items = random.sample(list(cooc_graph.keys()), min(2000, len(cooc_graph)))

# For each sampled item i, compute: for each neighbor j of i,
# what fraction of j's neighbors are also neighbors of i?
# High overlap -> high false negative risk in InfoNCE
overlap_ratios = []
for item in sample_items:
    my_neighbors = set(cooc_graph[item])
    if len(my_neighbors) < 2:
        continue
    for neighbor in list(my_neighbors)[:20]:  # check up to 20 neighbors
        their_neighbors = set(cooc_graph.get(neighbor, []))
        if len(their_neighbors) == 0:
            continue
        overlap = len(my_neighbors & their_neighbors)
        overlap_ratios.append(overlap / len(their_neighbors))

overlap_ratios = np.array(overlap_ratios)
print(f"   Sampled {len(sample_items)} items, {len(overlap_ratios)} neighbor pairs")
print(f"   Mean overlap ratio (Jaccard-like): {overlap_ratios.mean():.4f}")
print(f"   Median overlap ratio:              {np.median(overlap_ratios):.4f}")
print(f"   % pairs with >10% overlap:         {(overlap_ratios > 0.1).mean()*100:.1f}%")
print(f"   % pairs with >30% overlap:         {(overlap_ratios > 0.3).mean()*100:.1f}%")
print(f"   % pairs with >50% overlap:         {(overlap_ratios > 0.5).mean()*100:.1f}%")

# ============================================================
# 4. Simulate batch sampling and measure false negative rate
# ============================================================
print("\n" + "=" * 70)
print(f"4. Simulating {NUM_SIMULATED_BATCHES} batches (batch_size={BATCH_SIZE})")
print("=" * 70)

cooc_sets = {k: set(v) for k, v in cooc_graph.items()}
all_items_with_cooc = list(cooc_graph.keys())

total_pairs = 0
false_negative_pairs = 0
batch_fnr_list = []

for batch_idx in range(NUM_SIMULATED_BATCHES):
    # Sample batch (with replacement, simulating DataLoader shuffle)
    batch_items = random.choices(all_items_with_cooc, k=BATCH_SIZE)
    batch_set = set(batch_items)

    # For each anchor, check its positive (co-occurring sample)
    # against all other positives in the batch
    batch_fn = 0
    batch_total = 0

    for i, anchor in enumerate(batch_items):
        # Sample a positive (same logic as PRISMDataset._sample_positive)
        cooc_list = cooc_graph.get(anchor, [])
        if not cooc_list:
            continue
        pos = int(random.choice(cooc_list))

        # The positive is in the batch (it is z_pos[i])
        # For all OTHER anchors j≠i, check if their positives
        # also co-occur with anchor i
        anchor_cooc_set = cooc_sets.get(anchor, set())
        for j in range(len(batch_items)):
            if j == i:
                continue
            # The negative for anchor i is z_pos[j]
            # Is z_pos[j] actually a co-occurring partner of anchor i?
            pos_j = None
            cooc_j = cooc_graph.get(batch_items[j], [])
            if cooc_j:
                pos_j = int(random.choice(cooc_j))

            if pos_j is not None and pos_j in anchor_cooc_set:
                batch_fn += 1
            batch_total += 1

    if batch_total > 0:
        batch_fnr_list.append(batch_fn / batch_total)
    total_pairs += batch_total
    false_negative_pairs += batch_fn

mean_fnr = false_negative_pairs / total_pairs if total_pairs > 0 else 0
print(f"   Total anchor-negative pairs checked: {total_pairs}")
print(f"   False negatives (negatives that actually co-occur): {false_negative_pairs}")
print(f"   Expected false negative rate: {mean_fnr*100:.2f}%")
print(f"   Batch FNR mean ± std: {np.mean(batch_fnr_list)*100:.2f}% ± {np.std(batch_fnr_list)*100:.2f}%")
print(f"   Batch FNR range: [{np.min(batch_fnr_list)*100:.2f}%, {np.max(batch_fnr_list)*100:.2f}%]")

# ============================================================
# 5. Impact analysis: what does the false negative rate mean?
# ============================================================
print("\n" + "=" * 70)
print("5. Impact assessment")
print("=" * 70)

# In InfoNCE, false negatives weaken the contrastive signal:
# items that should be close get pushed apart.
# But with small τ=0.07, the loss is dominated by the hardest negatives.

# Key metric: what fraction of the "hardest negatives" are false?
# Hard negatives = negatives with high similarity to anchor.
# We can't compute exact similarities without embeddings, but we can
# bound it: items that share many co-occurring partners are more
# likely to be semantically similar (and thus hard negatives).

# Measure: for each item, how many items share >50% of co-occurring partners?
high_overlap_pairs = 0
total_possible_pairs = 0
for i, item1 in enumerate(sample_items[:500]):
    neighbors1 = cooc_sets.get(item1, set())
    if len(neighbors1) < 5:
        continue
    for item2 in sample_items[i+1:500]:
        neighbors2 = cooc_sets.get(item2, set())
        if len(neighbors2) < 5:
            continue
        total_possible_pairs += 1
        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        if union > 0 and intersection / union > 0.5:
            high_overlap_pairs += 1

print(f"   Item pairs with >50% co-occurrence overlap (Jaccard):")
print(f"     {high_overlap_pairs} / {total_possible_pairs} "
      f"({high_overlap_pairs/total_possible_pairs*100:.3f}%)")
print(f"   These are 'hard false negatives' — items that are semantically")
print(f"   very close but would be pushed apart by InfoNCE.")

# ============================================================
# 6. Recommendations
# ============================================================
print("\n" + "=" * 70)
print("6. Robustness assessment & recommendations")
print("=" * 70)

if mean_fnr < 0.01:
    print("   ✓ False negative rate is negligible (<1%)")
    print("   → In-batch negatives are mostly genuine negatives")
elif mean_fnr < 0.05:
    print("   ⚠ False negative rate is moderate (1-5%)")
    print("   → Some noise in the contrastive signal, but likely tolerable")
    print("   → Consider: larger batch size to dilute false negatives")
else:
    print("   ✗ False negative rate is high (>5%)")
    print("   → Significant noise — many negatives are actually positives")
    print("   → Consider: explicit negative sampling, or use debiased InfoNCE")

# Additional recommendations
if n_with_cooc < n_items * 0.5:
    print(f"   ⚠ Only {n_with_cooc/n_items*100:.1f}% items have co-occurrences")
    print("   → SACO is only effective for popular/active items")
    print("   → Cold-start items get no SACO signal (they fall back to")
    print("     self-as-positive, which gives zero InfoNCE loss)")
    print("   → This creates a popularity bias in the latent space")

# Check coverage: what % of randomly sampled batches have
# at least one item without co-occurrences?
no_cooc_count = 0
for _ in range(1000):
    batch = random.choices(list(item_id_to_idx.keys()), k=BATCH_SIZE)
    has_cooc = sum(1 for item in batch if item in cooc_graph)
    if has_cooc < BATCH_SIZE:
        no_cooc_count += 1
print(f"\n   Batches with cold-start items (no co-occurrences): {no_cooc_count/10:.1f}%")

print("\n" + "=" * 70)
print("Done.")
print("=" * 70)
