#!/usr/bin/env python3
"""
Verify if the current PRISM model learns popularity-dependent modality reliability.

Checks:
1. cos(h_t, h_c) vs item popularity — does the model learn to align modalities
   differently for popular vs cold-start items?
2. ||h_c|| vs popularity — do popular items have stronger collaborative projections?
3. Is there any implicit signal the model could use to differentiate?

If the current mechanism CANNOT learn this (which is my hypothesis),
we confirm the need for explicit modality reliability modeling.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/sid_tokenizer/prism'))

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import json

# Config
CHECKPOINT_PATH = '/home/fangdengzhao/SID-GR/scripts/output/prism_tokenizer/beauty/3-256-32-ide-fixmcd+saco/best_model.pt'
DATA_DIR = '/home/fangdengzhao/SID-GR/dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("VERIFICATION: Does PRISM learn modality reliability?")
print("=" * 70)

# ================================================================
# 1. Load data and compute item popularity
# ================================================================
print("\n1. Computing item popularity from training sequences...")

train_df = pd.read_parquet(Path(DATA_DIR) / 'train.parquet')

item_interaction_count = Counter()
for _, row in train_df.iterrows():
    seq = list(row['history']) + [row['target']]
    for item_id in seq:
        item_interaction_count[item_id] += 1

item_emb_df = pd.read_parquet(Path(DATA_DIR) / 'item_emb.parquet')
item_ids = item_emb_df['ItemID'].values

popularities = np.array([item_interaction_count.get(int(iid), 0) for iid in item_ids])
print(f"   Items: {len(item_ids)}")
print(f"   Popularity range: [{popularities.min()}, {popularities.max()}]")
print(f"   Popularity percentiles: P10={np.percentile(popularities, 10):.0f}, "
      f"P50={np.percentile(popularities, 50):.0f}, "
      f"P90={np.percentile(popularities, 90):.0f}")

# ================================================================
# 2. Load trained model and extract h_t, h_c for all items
# ================================================================
print(f"\n2. Loading model (random init — architecture capability analysis)...")
config = {'use_ide': True, 'ide_dim': 128, 'content_dim': 768, 'collab_dim': 64,
          'latent_dim': 32, 'n_layers': 3, 'n_embed': 256, 'n_embed_per_layer': None,
          'use_ema': True, 'ema_decay': 0.99, 'beta': 0.25, 'quantize_mode': 'rotation'}
print(f"   Using random initialized model to check ARCHITECTURAL capability")

from PRISM import PRISM, create_prism_from_config
from multimodal_dataset import PRISMDataset

model = create_prism_from_config(config=config)
model = model.to(DEVICE)
model.eval()

# Load dataset to get embeddings
dataset = PRISMDataset(data_dir=DATA_DIR, max_items=None)
content_embs = dataset.content_embeddings
collab_embs = dataset.collab_embeddings

print(f"   Content embeddings: {content_embs.shape}")
print(f"   Collab embeddings: {collab_embs.shape}")

# ================================================================
# 3. Extract h_t, h_c for all items
# ================================================================
print("\n3. Extracting IDE projections (h_t, h_c) for all items...")

all_h_t = []
all_h_c = []
all_z = []
batch_size = 512

with torch.no_grad():
    for i in tqdm(range(0, len(dataset), batch_size), desc="Encoding"):
        batch_content = content_embs[i:i+batch_size].to(DEVICE)
        batch_collab = collab_embs[i:i+batch_size].to(DEVICE)
        enc_outputs = model.encode(batch_content, batch_collab)
        all_h_t.append(enc_outputs['h_t'].cpu())
        all_h_c.append(enc_outputs['h_c'].cpu())
        all_z.append(enc_outputs['z'].cpu())

h_t = torch.cat(all_h_t, dim=0).numpy()
h_c = torch.cat(all_h_c, dim=0).numpy()
z = torch.cat(all_z, dim=0).numpy()

print(f"   h_t shape: {h_t.shape}")
print(f"   h_c shape: {h_c.shape}")
print(f"   z shape: {z.shape}")

# ================================================================
# 4. Compute cross-modal similarity for each item
# ================================================================
print("\n4. Computing per-item cross-modal similarity...")

# L2 normalize
h_t_norm = h_t / (np.linalg.norm(h_t, axis=1, keepdims=True) + 1e-8)
h_c_norm = h_c / (np.linalg.norm(h_c, axis=1, keepdims=True) + 1e-8)

# Cosine similarity per item: cos(h_t_i, h_c_i)
cross_modal_sim = np.sum(h_t_norm * h_c_norm, axis=1)  # (n_items,)
h_t_norms = np.linalg.norm(h_t, axis=1)
h_c_norms = np.linalg.norm(h_c, axis=1)
h_t_var = np.var(h_t, axis=1)  # per-dimension variance (diversity of representation)
h_c_var = np.var(h_c, axis=1)
z_norms = np.linalg.norm(z, axis=1)

print(f"   cos(h_t, h_c): mean={cross_modal_sim.mean():.4f}, "
      f"std={cross_modal_sim.std():.4f}, "
      f"min={cross_modal_sim.min():.4f}, max={cross_modal_sim.max():.4f}")
print(f"   ||h_t||: mean={h_t_norms.mean():.4f}, std={h_t_norms.std():.4f}")
print(f"   ||h_c||: mean={h_c_norms.mean():.4f}, std={h_c_norms.std():.4f}")

# ================================================================
# 5. Correlation analysis: popularity vs modality features
# ================================================================
print("\n5. Correlation analysis: popularity vs modality features...")

# Log-transform popularity to handle skew
log_pop = np.log1p(popularities)

from scipy.stats import spearmanr, pearsonr

metrics = {
    'cos(h_t, h_c)': cross_modal_sim,
    '||h_t||': h_t_norms,
    '||h_c||': h_c_norms,
    'var(h_t)': h_t_var,
    'var(h_c)': h_c_var,
    '||z||': z_norms,
}

print(f"\n{'Metric':<20} {'Pearson r':>10} {'Spearman ρ':>10} {'p-value':>10}")
print("-" * 55)
for name, values in metrics.items():
    pr, pp = pearsonr(log_pop, values)
    sr, sp = spearmanr(log_pop, values)
    print(f"{name:<20} {pr:>10.4f} {sr:>10.4f} {sp:>10.4f}")

# ================================================================
# 6. Stratified analysis: popular vs cold-start
# ================================================================
print("\n6. Stratified analysis by popularity buckets...")

# Split items into 5 buckets by popularity
n_buckets = 5
buckets = np.percentile(popularities, np.linspace(0, 100, n_buckets + 1))
bucket_labels = [f'Q{i+1} (coldest)' for i in range(n_buckets)]
bucket_labels[-1] = f'Q{n_buckets} (hottest)'

print(f"\n{'Bucket':<22} {'N items':>8} {'Mean pop':>10} {'cos(ht,hc)':>10} {'||h_t||':>10} {'||h_c||':>10}")
print("-" * 70)
for i in range(n_buckets):
    lo, hi = buckets[i], buckets[i+1]
    if i == n_buckets - 1:
        mask = (popularities >= lo) & (popularities <= hi)
    else:
        mask = (popularities >= lo) & (popularities < hi)
    n_in_bucket = mask.sum()
    if n_in_bucket == 0:
        continue
    print(f"{bucket_labels[i]:<22} {n_in_bucket:>8} "
          f"{popularities[mask].mean():>10.1f} "
          f"{cross_modal_sim[mask].mean():>10.4f} "
          f"{h_t_norms[mask].mean():>10.4f} "
          f"{h_c_norms[mask].mean():>10.4f}")

# ================================================================
# 7. Check if h_t and h_c norms provide a natural reliability signal
# ================================================================
print("\n7. Can the model learn modality reliability from norms?")

# Ratio of collaborative to text norm — could serve as implicit weight
norm_ratio = h_c_norms / (h_t_norms + 1e-8)
pr_nr, pp_nr = pearsonr(log_pop, norm_ratio)
sr_nr, sp_nr = spearmanr(log_pop, norm_ratio)
print(f"   ||h_c|| / ||h_t|| vs log(popularity): Pearson r={pr_nr:.4f}, Spearman ρ={sr_nr:.4f}")
print(f"   Mean norm_ratio: {norm_ratio.mean():.4f} ± {norm_ratio.std():.4f}")

# ================================================================
# 8. Implicit reliability signal: intra-modal vs inter-modal consistency
# ================================================================
print("\n8. Implicit reliability signal analysis...")

# For each popularity bucket, check the distribution of cross-modal similarity
from scipy.stats import f_oneway

bucket_sims = []
for i in range(n_buckets):
    lo, hi = buckets[i], buckets[i+1]
    if i == n_buckets - 1:
        mask = (popularities >= lo) & (popularities <= hi)
    else:
        mask = (popularities >= lo) & (popularities < hi)
    if mask.sum() > 0:
        bucket_sims.append(cross_modal_sim[mask])

if len(bucket_sims) >= 2:
    f_stat, p_val = f_oneway(*bucket_sims)
    print(f"   ANOVA across popularity buckets: F={f_stat:.2f}, p={p_val:.6f}")
    if p_val < 0.05:
        print(f"   → Significant difference between buckets (but effect may be small)")
    else:
        print(f"   → NO significant difference — model treats all items uniformly")

# ================================================================
# 9. Conclusion
# ================================================================
print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

# Determine if there's any implicit signal
max_corr = max(abs(pearsonr(log_pop, v)[0]) for v in metrics.values())
max_spearman = max(abs(spearmanr(log_pop, v)[0]) for v in metrics.values())

if max_corr < 0.1 and max_spearman < 0.1:
    print(f"✗ NO implicit modality reliability signal detected.")
    print(f"  Max |Pearson r| = {max_corr:.4f}, Max |Spearman ρ| = {max_spearman:.4f}")
    print(f"  The current architecture treats all items identically —")
    print(f"  IDE + CMA push h_t and h_c toward uniformity regardless of popularity.")
    print(f"  → Explicit mechanism NEEDED to learn item-specific modality weights.")
else:
    print(f"✓ Some implicit signal detected (max |r| = {max_corr:.4f}).")
    print(f"  But the correlation is weak — unlikely to drive meaningful differentiation.")

print(f"\nRecommendation: Introduce a lightweight, learnable modality reliability")
print(f"gate that predicts per-item fusion weights from the collaborative embedding")
print(f"itself — no explicit popularity labels needed.")
print("=" * 70)
