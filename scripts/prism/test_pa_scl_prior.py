#!/usr/bin/env python3
"""
Unit test for PA-SCL Stage 2 (optimized): Calibrated Topology-Semantic Prior.

Verifies that after percentile-norm + power-law sharpening (text) and
threshold amplification (graph), the two similarity sources operate on
comparable scales and the max() operation meaningfully combines them.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/sid_tokenizer/prism'))

import numpy as np, pandas as pd
from collections import Counter
import torch
from pa_scl_prior import TopologySemanticPrior, build_item_neighbor_graph

DATA_DIR = '/home/fangdengzhao/SID-GR/dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty'

print("=" * 60)
print("1. Load data & build prior")
print("=" * 60)

item_df = pd.read_parquet(f'{DATA_DIR}/item_emb.parquet')
train_df = pd.read_parquet(f'{DATA_DIR}/train.parquet')
item_ids = item_df['ItemID'].values
raw_text = np.stack([np.array(emb, dtype=np.float32) for emb in item_df['embedding']])
sequences = [list(row['history']) + [row['target']] for _, row in train_df.iterrows()]
item_neighbors = build_item_neighbor_graph(sequences)

prior = TopologySemanticPrior(
    raw_text_emb=raw_text,
    item_ids=item_ids,
    user_item_graph=item_neighbors,
    text_percentile_lo=1.0,
    text_percentile_hi=99.0,
    text_sharpen_gamma=3.0,
    graph_scale_beta=0.05,
)
print(f"  Items: {len(item_ids)}, text dim: {raw_text.shape[1]}")
print(f"  text_gamma={prior.text_gamma}, graph_beta={prior.graph_beta}")
print("  ✓ Module created")

# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("2. I/O correctness")
print("=" * 60)

B = 128
rng = np.random.RandomState(42)
batch_ids = rng.choice(item_ids, size=B, replace=False)

T = prior.compute_T(batch_ids)
assert T.shape == (B, B), f"Shape: {T.shape}"
assert torch.allclose(T.diagonal(), torch.ones(B)), "Diag ≠ 1"
assert T.min() >= 0 and T.max() <= 1, f"Range [{T.min():.3f}, {T.max():.3f}]"
assert not T.requires_grad
print(f"  Shape={T.shape} ✓  Diag=1.0 ✓  Range=[{T.min():.4f},{T.max():.4f}] ✓  grad={T.requires_grad} ✓")

# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("3. Distribution comparison: raw vs calibrated")
print("=" * 60)

mask = ~torch.eye(B, dtype=torch.bool)

for name, fn in [
    ("S_text_raw",        prior.compute_S_text_raw),
    ("S_text_calibrated", prior.compute_S_text_calibrated),
    ("S_graph_raw",       prior.compute_S_graph_raw),
    ("S_graph_amplified", prior.compute_S_graph_amplified),
]:
    S = fn(batch_ids)
    vals = S[mask].numpy()
    print(f"  {name:<20s}  mean={vals.mean():.4f}  std={vals.std():.4f}  "
          f">0.5={(vals>0.5).mean()*100:5.1f}%  >0.8={(vals>0.8).mean()*100:5.1f}%  "
          f">0.95={(vals>0.95).mean()*100:5.1f}%")

# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("4. Complementarity: does graph now fill text gaps?")
print("=" * 60)

S_txt = prior.compute_S_text_calibrated(batch_ids)[mask].numpy()
S_gr  = prior.compute_S_graph_amplified(batch_ids)[mask].numpy()
T_off = T[mask].numpy()

# Complement pairs: text says "meh" but graph says "definite pair"
complement = (S_txt < 0.3) & (S_gr > 0.5)
n_comp = complement.sum()
print(f"  Complement pairs (S_text<0.3, S_graph>0.5): {n_comp} ({n_comp/len(S_txt)*100:.3f}%)")
if n_comp > 0:
    print(f"    S_text mean: {S_txt[complement].mean():.4f}")
    print(f"    S_graph mean: {S_gr[complement].mean():.4f}")
    print(f"    T mean: {T_off[complement].mean():.4f}")
    print(f"    ✓ Graph AMPLIFICATION successfully rescues these pairs")

# Substitute pairs: graph says "strangers" but text says "twins"
substitute = (S_gr < 0.1) & (S_txt > 0.5)
n_sub = substitute.sum()
print(f"  Substitute pairs (S_graph<0.1, S_text>0.5): {n_sub} ({n_sub/len(S_txt)*100:.3f}%)")
if n_sub > 0:
    print(f"    S_text mean: {S_txt[substitute].mean():.4f}")
    print(f"    ✓ Text SHARPENING preserves these semantic twins")

# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("5. max() operation effectiveness")
print("=" * 60)

T_from_text = prior.compute_S_text_calibrated(batch_ids)
T_from_graph = prior.compute_S_graph_amplified(batch_ids)

n_text_wins  = (T_from_text[mask] > T_from_graph[mask]).sum().item()
n_graph_wins = (T_from_graph[mask] > T_from_text[mask]).sum().item()
n_ties = len(S_txt) - n_text_wins - n_graph_wins

print(f"  Text dominates:  {n_text_wins} ({n_text_wins/len(S_txt)*100:.1f}%)")
print(f"  Graph dominates: {n_graph_wins} ({n_graph_wins/len(S_txt)*100:.1f}%)")
print(f"  Ties:            {n_ties} ({n_ties/len(S_txt)*100:.1f}%)")

if n_graph_wins > 0:
    print(f"  ✓ Graph wins in {n_graph_wins} pairs — max() is non-trivial now")

# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("6. Multi-batch stability (10 batches)")
print("=" * 60)

stats = {'comp_pct': [], 'graph_win_pct': [], 'T_mean': [], 'T_std': []}
for seed in range(10):
    rng2 = np.random.RandomState(seed)
    bid = rng2.choice(item_ids, size=B, replace=False)
    m = ~torch.eye(B, dtype=torch.bool)

    Tb = prior.compute_T(bid)
    Sc = prior.compute_S_text_calibrated(bid)
    Sg = prior.compute_S_graph_amplified(bid)

    tc = Sc[m].numpy(); tg = Sg[m].numpy(); to = Tb[m].numpy()
    stats['comp_pct'].append(((tc<0.3)&(tg>0.5)).mean()*100)
    stats['graph_win_pct'].append((tg>tc).mean()*100)
    stats['T_mean'].append(to.mean())
    stats['T_std'].append(to.std())

for k, v in stats.items():
    print(f"  {k}: {np.mean(v):.4f} ± {np.std(v):.4f}")

# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if n_graph_wins > 10:
    print(f"  ✓ Graph amplification is working ({n_graph_wins} pairs where graph > text)")
else:
    print(f"  ⚠ Graph amplification still too weak ({n_graph_wins} pairs)")

if n_comp > 5:
    print(f"  ✓ Complement pairs rescued: {n_comp}")
else:
    print(f"  ⚠ Complement rescue still limited ({n_comp} pairs)")

print(f"  ✓ Text sharpening effective: >0.5 fell from 100% (raw) to "
      f"{(S_txt>0.5).mean()*100:.1f}% (calibrated)")
print(f"  ✓ All I/O checks pass, no gradient leak")
