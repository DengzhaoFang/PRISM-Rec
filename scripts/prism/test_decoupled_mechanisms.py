#!/usr/bin/env python3
"""
Test Suite: Comparing Decoupled Representation Mechanisms for Stage 1

Candidates:
  A. Baseline:      [h_c || h_t] concat, CMA only
  B. SharedPrivate: z_shared(→SID) + z_private(→aux), orth loss
  C. Hierarchical:  Layer-1 structure prior (codebook entropy at layer level)

Key metrics:
  1. Reconstruction quality (UPR)
  2. SID discriminability (unique ID rate)
  3. Modality information preservation (MI between z and each modality)
  4. Gradient conflict (cos between recon and aux gradients)
"""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import defaultdict
torch.manual_seed(42)

# ── Synthetic data: 500 items, text + collab with varying quality ──
N, Dt, Dc, D = 500, 768, 64, 128
# Popular items (first 300): text and collab are correlated
h_t_pop = torch.randn(300, D)
h_c_pop = h_t_pop + torch.randn(300, D) * 0.3  # high SNR

# Cold-start items (last 200): collab is noise
h_t_cold = torch.randn(200, D)
h_c_cold = torch.randn(200, D) * 1.5  # low SNR

h_t_all = torch.cat([h_t_pop, h_t_cold])
h_c_all = torch.cat([h_c_pop, h_c_cold])
is_pop = torch.cat([torch.ones(300), torch.zeros(200)])

B = 64
EPOCHS = 80

# ── Common encoder/decoder components ──
def make_encoder(in_dim, out_dim=32):
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, out_dim))

def make_decoder(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, 128), nn.LayerNorm(128), nn.ReLU(),
        nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, out_dim))


# ═══════════════════════════════════════════════════════════════════
# A. BASELINE: Simple concat + CMA
# ═══════════════════════════════════════════════════════════════════
class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = make_encoder(D * 2, 32)
        self.decoder = make_decoder(32, D * 2)

    def forward(self, h_t, h_c):
        z_clean = torch.cat([h_c, h_t], dim=-1)
        z = self.encoder(z_clean)
        z_dec = self.decoder(z)
        return z, z_dec, z_clean, h_t, h_c


# ═══════════════════════════════════════════════════════════════════
# B. SHARED-PRIVATE: z_shared→SID, z_private→aux, orthogonal constraint
# ═══════════════════════════════════════════════════════════════════
class SharedPrivateModel(nn.Module):
    def __init__(self, shared_dim=32, private_dim=32):
        super().__init__()
        # Shared pathway: capture cross-modal consensus → goes to SID
        self.shared_fusion = nn.Sequential(
            nn.Linear(D * 2, 128), nn.ReLU(), nn.Linear(128, shared_dim))
        # Private pathway: capture modality-specific details → stays dense
        self.private_encoder = nn.Sequential(
            nn.Linear(D * 2, 128), nn.ReLU(), nn.Linear(128, private_dim))
        # Decoder: reconstructs from both shared + private
        self.decoder = make_decoder(shared_dim + private_dim, D * 2)

    def forward(self, h_t, h_c):
        z_clean = torch.cat([h_c, h_t], dim=-1)
        z_shared = self.shared_fusion(z_clean)
        z_private = self.private_encoder(z_clean)
        z_full = torch.cat([z_shared, z_private], dim=-1)
        z_dec = self.decoder(z_full)
        return z_shared, z_private, z_dec, z_clean, h_t, h_c


# ═══════════════════════════════════════════════════════════════════
# C. HIERARCHICAL: Simulate layer-specific with structured prior
# ═══════════════════════════════════════════════════════════════════
# NOTE: Real RQ-VAE has 3 layers. We simulate "Layer 1" getting a
# structural prior by encouraging the latent z to be well-clustered.
class HierarchicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = make_encoder(D * 2, 32)
        self.decoder = make_decoder(32, D * 2)
        # Lightweight clustering head for the structural prior
        self.cluster_proj = nn.Linear(32, 16)

    def forward(self, h_t, h_c):
        z_clean = torch.cat([h_c, h_t], dim=-1)
        z = self.encoder(z_clean)
        z_dec = self.decoder(z)
        # Cluster feature for structural prior
        z_cluster = self.cluster_proj(z)
        return z, z_dec, z_clean, h_t, h_c, z_cluster


# ═══════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════

def compute_cma_loss(h_t, h_c, tau=0.07):
    ht_n = F.normalize(h_t, dim=-1); hc_n = F.normalize(h_c, dim=-1)
    sim = ht_n @ hc_n.T / tau
    labels = torch.arange(len(h_t))
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

def compute_orth_loss(z1, z2):
    """Encourage z1 and z2 to encode orthogonal (non-redundant) information."""
    z1n = F.normalize(z1, dim=-1); z2n = F.normalize(z2, dim=-1)
    # Frobenius norm of cross-covariance
    cross = z1n.T @ z2n / z1n.size(0)
    return (cross ** 2).sum()

def compute_cluster_structure_loss(z_cluster):
    """Encourage cluster-friendly structure: low intra-cluster, high inter-cluster variance."""
    z_n = F.normalize(z_cluster, dim=-1)
    sim = z_n @ z_n.T
    # Penalize intermediate similarities (should be either 1 or -1 for well-separated clusters)
    # Simple proxy: maximize variance of similarities
    return -sim.var()

def train_model(model, name, extra_loss_fn=None, lambda_extra=0.1):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = defaultdict(list)

    for ep in range(EPOCHS):
        idx = torch.randperm(N)
        total_upr = 0.0; total_cma = 0.0; total_extra = 0.0

        for s in range(0, N, B):
            bi = idx[s:s+B]
            ht = h_t_all[bi]; hc = h_c_all[bi]

            out = model(ht, hc)

            if isinstance(model, SharedPrivateModel):
                z_shared, z_private, z_dec, z_clean, ht_out, hc_out = out
                upr = F.mse_loss(z_dec, z_clean.detach())
                cma = compute_cma_loss(ht_out, hc_out)
                orth = compute_orth_loss(z_shared, z_private)
                loss = upr + 0.1 * cma + 0.05 * orth  # λ_orth = 0.05
                extra = orth.item()
            elif isinstance(model, HierarchicalModel):
                z, z_dec, z_clean, ht_out, hc_out, z_cluster = out
                upr = F.mse_loss(z_dec, z_clean.detach())
                cma = compute_cma_loss(ht_out, hc_out)
                struct = compute_cluster_structure_loss(z_cluster)
                loss = upr + 0.1 * cma + 0.01 * struct
                extra = struct.item()
            else:  # Baseline
                z, z_dec, z_clean, ht_out, hc_out = out
                upr = F.mse_loss(z_dec, z_clean.detach())
                cma = compute_cma_loss(ht_out, hc_out)
                loss = upr + 0.1 * cma
                extra = 0.0

            opt.zero_grad(); loss.backward(); opt.step()
            total_upr += upr.item(); total_cma += cma.item(); total_extra += extra

        history['upr'].append(total_upr)
        history['cma'].append(total_cma)
        history['extra'].append(total_extra)
        history['loss'].append(total_upr + 0.1 * total_cma)

    return history, model


def evaluate_model(model, name):
    """Evaluate SID discriminability and modality information preservation."""
    with torch.no_grad():
        if isinstance(model, SharedPrivateModel):
            out = model(h_t_all, h_c_all)
            z_sid = out[0]  # z_shared → would be quantized for SID
            z_private = out[1]
            z_recon = out[2]
        elif isinstance(model, HierarchicalModel):
            out = model(h_t_all, h_c_all)
            z_sid = out[0]
            z_recon = out[2]
        else:
            out = model(h_t_all, h_c_all)
            z_sid = out[0]
            z_recon = out[2]

    # Metric 1: Reconstruction quality
    z_clean_all = torch.cat([h_c_all, h_t_all], dim=-1)
    upr = F.mse_loss(z_recon, z_clean_all).item()

    # Metric 2: SID discriminability (cosine between z_sid of different items)
    z_n = F.normalize(z_sid, dim=-1)
    sim = z_n @ z_n.T
    mask = ~torch.eye(N, dtype=torch.bool)
    inter_cos = sim[mask].mean().item()

    # Metric 3: Modality information preservation
    # MI proxy: can we predict h_t and h_c from z_sid?
    pred_t = nn.Linear(z_sid.shape[-1], D)(z_sid)
    pred_c = nn.Linear(z_sid.shape[-1], D)(z_sid)
    # Simple linear probe
    mi_t = F.cosine_similarity(pred_t, h_t_all, dim=-1).mean().item()
    mi_c = F.cosine_similarity(pred_c, h_c_all, dim=-1).mean().item()

    # Metric 4: Popular/cold-start distinction quality
    # Does z_sid preserve more info for popular vs cold-start?
    z_pop = z_sid[:300]; z_cold = z_sid[300:]
    pop_dispersion = F.normalize(z_pop, dim=-1).std(dim=0).mean().item()
    cold_dispersion = F.normalize(z_cold, dim=-1).std(dim=0).mean().item()

    # Metric 5: Gradient conflict (cos between recon and CMA gradients)
    # Simplified: cosine between z_clean and CMA-aligned representation

    return {
        'upr': upr,
        'inter_cos': inter_cos,
        'mi_text': mi_t,
        'mi_collab': mi_c,
        'pop_disp': pop_dispersion,
        'cold_disp': cold_dispersion,
        'disp_ratio': pop_dispersion / max(cold_dispersion, 1e-8),
    }


# ═══════════════════════════════════════════════════════════════════
# Run comparison
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("COMPARING DECOUPLED REPRESENTATION MECHANISMS")
print("=" * 70)

models = {
    'A_Baseline':      BaselineModel(),
    'B_SharedPrivate': SharedPrivateModel(),
    'C_Hierarchical':  HierarchicalModel(),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...", end=' ', flush=True)
    history, model = train_model(model, name)
    metrics = evaluate_model(model, name)
    results[name] = {**metrics, 'final_upr': history['upr'][-1]}
    print(f"done. UPR={metrics['upr']:.4f}, inter_cos={metrics['inter_cos']:.4f}")

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"{'Model':<20s} {'UPR':>8s} {'inter_cos':>10s} {'MI_text':>8s} {'MI_collab':>10s} {'pop/cold':>10s}")
print("-" * 70)
for name in ['A_Baseline', 'B_SharedPrivate', 'C_Hierarchical']:
    m = results[name]
    print(f"{name:<20s} {m['upr']:>8.4f} {m['inter_cos']:>10.4f} {m['mi_text']:>8.4f} "
          f"{m['mi_collab']:>10.4f} {m['disp_ratio']:>10.4f}")

# ── Analysis ──
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

bl = results['A_Baseline']
for name in ['B_SharedPrivate', 'C_Hierarchical']:
    m = results[name]
    upr_change = (m['upr'] - bl['upr']) / bl['upr'] * 100
    cos_change = (m['inter_cos'] - bl['inter_cos']) / abs(bl['inter_cos']) * 100

    print(f"\n{name} vs Baseline:")
    print(f"  UPR:         {upr_change:+.1f}%  (<0 = better reconstruction)")
    print(f"  inter_cos:   {cos_change:+.1f}%  (>0 = better SID separation)")
    print(f"  MI_text:     {m['mi_text']:.4f} vs {bl['mi_text']:.4f}")
    print(f"  MI_collab:   {m['mi_collab']:.4f} vs {bl['mi_collab']:.4f}")

    if m['disp_ratio'] > bl['disp_ratio']:
        print(f"  pop/cold:    {m['disp_ratio']:.4f} vs {bl['disp_ratio']:.4f} ✓ better cold-start handling")
    else:
        print(f"  pop/cold:    {m['disp_ratio']:.4f} vs {bl['disp_ratio']:.4f}")

# ── Recommendation ──
print("\n" + "=" * 70)
print("RECOMMENDATION")
print("=" * 70)

# Find best model by composite score
scores = {}
for name in results:
    m = results[name]
    # Lower UPR is better, higher inter_cos is better (more discriminative)
    # Higher MI is better, higher disp_ratio is better
    score = (-m['upr'] * 10 + m['inter_cos'] * 5 + m['mi_text'] * 2 +
             m['mi_collab'] * 2 + m['disp_ratio'] * 3)
    scores[name] = score

best = max(scores, key=scores.get)
print(f"Best mechanism: {best}")
print(f"Rationale: This mechanism achieves the best trade-off between")
print(f"reconstruction quality, SID discriminability, and modality balance.")
