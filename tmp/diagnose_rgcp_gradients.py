#!/usr/bin/env python3
"""
Diagnostic: RGCP gradient flow and multi-objective interaction analysis.
Tests whether RGCP gradients collaborate or conflict with UPR + CMA.

Usage: python tmp/diagnose_rgcp_gradients.py
"""

import sys
sys.path.insert(0, 'src/sid_tokenizer/prism')

import torch
import torch.nn as nn
import numpy as np

from PRISM import PRISM, create_prism_from_config
from prism_losses import PRISMTotalLoss, RGCPLoss


def build_model(use_rgcp=True):
    return PRISM(
        content_dim=768, collab_dim=64,
        latent_dim=32, n_layers=3,
        n_embed_per_layer=[256, 256, 256],
        encoder_hidden_dims=[512, 256, 128],
        decoder_hidden_dims=[128, 256, 512],
        use_ide=True, ide_dim=128,
        use_rgcp=use_rgcp, rgcp_hidden_dim=128,
        use_ema=True, ema_decay=0.99, beta=0.25,
        quantize_mode='rotation',
    )


def make_batch(bs=64):
    return (
        torch.randn(bs, 768),  # content_emb
        torch.randn(bs, 64),   # collab_emb
    )


def analyze_rgcp_gradient_conflict():
    """Check if RGCP's anchor loss conflicts with main UPR gradient."""
    print("=" * 70)
    print("TEST 1: RGCP Anchor Loss vs UPR gradient conflict")
    print("=" * 70)

    model = build_model(use_rgcp=True)
    model.train()

    loss_fn = PRISMTotalLoss(commitment_weight=0.25, lambda_cma=0.05)
    rgcp_loss_fn = RGCPLoss(
        reliability_balance_weight=0.01,
        reliability_variance_weight=0.01,
        anchor_weight=0.01,
        recommendation_weight=0.05,
    )

    content_emb, collab_emb = make_batch(128)
    out = model(content_emb, collab_emb, return_codes=True)

    # 1. Compute each loss separately
    total_loss, _ = loss_fn(
        z_dec=out['z_dec'], z_clean=out['z_clean'],
        commitment_loss=out['codebook_loss'],
        cma_t=out.get('cma_t'), cma_c=out.get('cma_c'),
    )

    anchor_loss = rgcp_loss_fn.anchor_loss(
        out['h_t'], out['h_c'],
        out['h_t_refined'], out['h_c_refined'],
        out['delta_t'], out['delta_c'],
    )

    # 2. Compute individual gradients for h_t_refined
    h_t_ref = out['h_t_refined'].clone().detach().requires_grad_(True)
    h_c_ref = out['h_c_refined'].clone().detach().requires_grad_(True)

    # Anchor: shift_loss = MSE(h_t_refined, h_t.detach()) / 2
    shift_grad_t = (h_t_ref - out['h_t'].detach()) / h_t_ref.size(0)

    # Anchor: anchor_pull = MSE(h_t_refined, (h_t+h_c)/2) / 2 * 0.5
    anchor_target = (out['h_t'].detach() + out['h_c'].detach()) * 0.5
    anchor_pull_grad_t = (h_t_ref - anchor_target) / h_t_ref.size(0) * 0.5

    # Combined anchor gradient on h_t_refined
    anchor_grad_t = 0.01 * (shift_grad_t + anchor_pull_grad_t)

    # 3. Compute UPR gradient on h_t_refined via encoder path
    z_clean = torch.cat([h_t_ref, h_c_ref], dim=-1)
    z_clean.retain_grad()

    # Simulate encoder forward
    encoder = model.encoder.encoder
    z = encoder(z_clean)

    # Simulate quantize
    z_q = z  # simplified: no quantization for gradient check
    decoder = model.decoder
    z_dec = decoder(z_q)

    upr = nn.functional.mse_loss(z_dec, z_clean.detach())
    upr.backward(retain_graph=True)
    upr_grad_t = h_t_ref.grad.clone()

    # 4. Check cosine similarity between UPR grad and Anchor grad
    upr_flat = upr_grad_t.flatten()
    anchor_flat = anchor_grad_t.flatten()

    cos_sim = torch.nn.functional.cosine_similarity(
        upr_flat.unsqueeze(0), anchor_flat.unsqueeze(0)
    ).item()

    upr_norm = upr_flat.norm().item()
    anchor_norm = anchor_flat.norm().item()
    ratio = anchor_norm / (upr_norm + 1e-8)

    print(f"  UPR grad norm on h_t_refined:     {upr_norm:.6f}")
    print(f"  Anchor grad norm on h_t_refined:  {anchor_norm:.8f}")
    print(f"  Anchor/UPR ratio:                 {ratio:.2e}")
    print(f"  Cosine similarity (UPR vs Anchor): {cos_sim:.4f}")

    if cos_sim < -0.3:
        print("  ⚠️  CONFLICT: UPR and Anchor gradients are anti-aligned!")
    elif cos_sim < 0.3:
        print("  ⚠️  WEAK: UPR and Anchor gradients are near-orthogonal.")
    else:
        print("  ✓  UPR and Anchor gradients are aligned.")

    if ratio > 0.1:
        print("  ⚠️  Anchor grad is >10% of UPR grad — may dominate refinement.")
    elif ratio < 0.001:
        print("  ⚠️  Anchor grad is <0.1% of UPR grad — too weak to matter.")
    else:
        print("  ✓  Anchor grad ratio is in reasonable range.")

    model.zero_grad(set_to_none=True)
    return cos_sim, ratio


def analyze_rgcp_gate_gradients():
    """Check how much gradient the reliability network receives."""
    print("\n" + "=" * 70)
    print("TEST 2: Reliability Network Gradient Magnitudes")
    print("=" * 70)

    model = build_model(use_rgcp=True)
    model.train()

    loss_fn = PRISMTotalLoss(commitment_weight=0.25, lambda_cma=0.05)
    rgcp_loss_fn = RGCPLoss(
        reliability_balance_weight=0.01,
        reliability_variance_weight=0.01,
        anchor_weight=0.01,
        recommendation_weight=0.05,
    )

    content_emb, collab_emb = make_batch(128)
    out = model(content_emb, collab_emb, return_codes=True)

    total_loss, _ = loss_fn(
        z_dec=out['z_dec'], z_clean=out['z_clean'],
        commitment_loss=out['codebook_loss'],
        cma_t=out.get('cma_t'), cma_c=out.get('cma_c'),
    )

    rel_loss = rgcp_loss_fn.reliability_loss(out['reliability_t'], out['reliability_c'])
    anchor_loss = rgcp_loss_fn.anchor_loss(
        out['h_t'], out['h_c'],
        out['h_t_refined'], out['h_c_refined'],
        out['delta_t'], out['delta_c'],
    )

    # Backward all losses together
    (total_loss + rel_loss + anchor_loss).backward()

    rgcp = model.encoder.rgcp
    grads = {}
    for name, param in rgcp.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.norm().item()

    # Also check other modules
    ide_grads = {}
    for name, param in model.encoder.ide.named_parameters():
        if param.grad is not None:
            ide_grads[name] = param.grad.norm().item()

    encoder_grads = {}
    for name, param in model.encoder.encoder.named_parameters():
        if param.grad is not None:
            encoder_grads[name] = param.grad.norm().item()

    print(f"  RGCP parameter gradients:")
    for k, v in sorted(grads.items()):
        print(f"    {k}: {v:.2e}")

    rgcp_avg = np.mean(list(grads.values())) if grads else 0
    ide_avg = np.mean(list(ide_grads.values())) if ide_grads else 0
    enc_avg = np.mean(list(encoder_grads.values())) if encoder_grads else 0

    print(f"\n  Mean grad norm - RGCP: {rgcp_avg:.2e}, IDE: {ide_avg:.2e}, Encoder: {enc_avg:.2e}")
    print(f"  RGCP/Encoder ratio: {rgcp_avg/(enc_avg+1e-8):.2e}")
    print(f"  RGCP/IDE ratio:     {rgcp_avg/(ide_avg+1e-8):.2e}")

    if rgcp_avg < enc_avg * 1e-3:
        print("  ⚠️  RGCP gradients are very small relative to encoder — training signal is dominated.")
    else:
        print("  ✓  RGCP gradients are in a reasonable range.")

    model.zero_grad(set_to_none=True)
    return rgcp_avg, enc_avg


def analyze_reliability_network_capacity():
    """Check if the reliability network can produce diverse outputs."""
    print("\n" + "=" * 70)
    print("TEST 3: Reliability Network Output Diversity")
    print("=" * 70)

    model = build_model(use_rgcp=True)
    model.eval()

    rgcp = model.encoder.rgcp

    # Test with different types of inputs
    bs = 256
    # Case 1: Random inputs
    h_t1 = torch.randn(bs, 128)
    h_c1 = torch.randn(bs, 128)
    out1 = rgcp(h_t1, h_c1)

    # Case 2: Highly correlated inputs
    h_t2 = torch.randn(bs, 128)
    h_c2 = h_t2 + 0.1 * torch.randn(bs, 128)
    out2 = rgcp(h_t2, h_c2)

    # Case 3: Anti-correlated inputs
    h_t3 = torch.randn(bs, 128)
    h_c3 = -h_t3 + 0.1 * torch.randn(bs, 128)
    out3 = rgcp(h_t3, h_c3)

    print(f"  Random inputs:")
    print(f"    r_t: mean={out1['reliability_t'].mean():.4f}, std={out1['reliability_t'].std():.4f}")
    print(f"    r_c: mean={out1['reliability_c'].mean():.4f}, std={out1['reliability_c'].std():.4f}")
    print(f"    r_t - r_c: mean={out1['reliability_t'].mean()-out1['reliability_c'].mean():.4f}")
    print(f"    delta_t norm: {out1['delta_t'].norm(dim=-1).mean():.4f}")
    print(f"    delta_c norm: {out1['delta_c'].norm(dim=-1).mean():.4f}")

    print(f"  Correlated inputs:")
    print(f"    r_t: mean={out2['reliability_t'].mean():.4f}, std={out2['reliability_t'].std():.4f}")
    print(f"    r_c: mean={out2['reliability_c'].mean():.4f}, std={out2['reliability_c'].std():.4f}")

    print(f"  Anti-correlated inputs:")
    print(f"    r_t: mean={out3['reliability_t'].mean():.4f}, std={out3['reliability_t'].std():.4f}")
    print(f"    r_c: mean={out3['reliability_c'].mean():.4f}, std={out3['reliability_c'].std():.4f}")

    # Compute refinement magnitude
    ref_t_1 = (out1['h_t_refined'] - h_t1).norm(dim=-1).mean()
    ref_c_1 = (out1['h_c_refined'] - h_c1).norm(dim=-1).mean()
    print(f"\n  Refinement magnitude (random inputs):")
    print(f"    ||h_t' - h_t||: {ref_t_1:.6f}")
    print(f"    ||h_c' - h_c||: {ref_c_1:.6f}")

    if ref_t_1 < 0.01:
        print("  ⚠️  RGCP refinement is negligible (< 0.01) — it's barely changing representations!")
    else:
        print("  ✓  RGCP is making noticeable changes to representations.")

    return ref_t_1, ref_c_1


def analyze_alpha_evolution():
    """Check the current alpha values and their effective range."""
    print("\n" + "=" * 70)
    print("TEST 4: Alpha Parameter Analysis")
    print("=" * 70)

    model = build_model(use_rgcp=True)
    rgcp = model.encoder.rgcp

    # Simulate what different alpha values mean
    for init_val in [0.05, 0.2, 0.5, 1.0, 2.0]:
        alpha_t = torch.sigmoid(torch.tensor(init_val)) * 0.1
        alpha_c = torch.sigmoid(torch.tensor(init_val)) * 0.1
        print(f"  raw_alpha={init_val:.2f} → effective_alpha={alpha_t.item():.4f} (max possible=0.1)")

    print(f"\n  Current alpha_t = sigmoid({rgcp.alpha_t.item():.4f}) * 0.1 = {torch.sigmoid(rgcp.alpha_t).item() * 0.1:.4f}")
    print(f"  Current alpha_c = sigmoid({rgcp.alpha_c.item():.4f}) * 0.1 = {torch.sigmoid(rgcp.alpha_c).item() * 0.1:.4f}")
    print(f"  The * 0.1 factor caps alpha at 0.1 — refinement at most 10% of delta")


def analyze_gradient_paths():
    """Trace which parameters get gradients from which losses."""
    print("\n" + "=" * 70)
    print("TEST 5: Gradient Path Tracing")
    print("=" * 70)

    model = build_model(use_rgcp=True)
    model.train()

    loss_fn = PRISMTotalLoss(commitment_weight=0.25, lambda_cma=0.05)
    rgcp_loss_fn = RGCPLoss(
        reliability_balance_weight=0.01,
        reliability_variance_weight=0.01,
        anchor_weight=0.01,
        recommendation_weight=0.05,
    )

    content_emb, collab_emb = make_batch(128)
    out = model(content_emb, collab_emb, return_codes=True)

    # Test each loss independently
    components = {}

    # UPR only
    model.zero_grad(set_to_none=True)
    upr_loss = loss_fn.upr_loss(out['z_dec'], out['z_clean'])
    upr_loss.backward(retain_graph=True)
    components['UPR'] = {
        'rgcp': sum(p.grad.norm().item() for p in model.encoder.rgcp.parameters() if p.grad is not None),
        'ide': sum(p.grad.norm().item() for p in model.encoder.ide.parameters() if p.grad is not None),
        'encoder': sum(p.grad.norm().item() for p in model.encoder.encoder.parameters() if p.grad is not None),
        'decoder': sum(p.grad.norm().item() for p in model.decoder.parameters() if p.grad is not None),
    }

    # CMA only
    model.zero_grad(set_to_none=True)
    cma_loss = loss_fn.cma_loss(out['cma_t'], out['cma_c'])
    cma_loss.backward(retain_graph=True)
    components['CMA'] = {
        'rgcp': sum(p.grad.norm().item() for p in model.encoder.rgcp.parameters() if p.grad is not None),
        'ide': sum(p.grad.norm().item() for p in model.encoder.ide.parameters() if p.grad is not None),
        'encoder': sum(p.grad.norm().item() for p in model.encoder.encoder.parameters() if p.grad is not None),
    }

    # Anchor only
    model.zero_grad(set_to_none=True)
    anchor_loss = rgcp_loss_fn.anchor_loss(
        out['h_t'], out['h_c'],
        out['h_t_refined'], out['h_c_refined'],
        out['delta_t'], out['delta_c'],
    )
    anchor_loss.backward(retain_graph=True)
    components['ANCHOR'] = {
        'rgcp': sum(p.grad.norm().item() for p in model.encoder.rgcp.parameters() if p.grad is not None),
        'ide': sum(p.grad.norm().item() for p in model.encoder.ide.parameters() if p.grad is not None),
        'encoder': sum(p.grad.norm().item() for p in model.encoder.encoder.parameters() if p.grad is not None),
    }

    # Reliability only
    model.zero_grad(set_to_none=True)
    rel_loss = rgcp_loss_fn.reliability_loss(out['reliability_t'], out['reliability_c'])
    rel_loss.backward()
    components['RELIABILITY'] = {
        'rgcp': sum(p.grad.norm().item() for p in model.encoder.rgcp.parameters() if p.grad is not None),
        'ide': sum(p.grad.norm().item() for p in model.encoder.ide.parameters() if p.grad is not None),
        'encoder': sum(p.grad.norm().item() for p in model.encoder.encoder.parameters() if p.grad is not None),
    }

    print(f"  {'Loss':<14} {'RGCP':>10} {'IDE':>10} {'Encoder':>10} {'Decoder':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for loss_name, grads in components.items():
        rgcp_s = f"{grads.get('rgcp', 0):.2e}" if grads.get('rgcp', 0) > 0 else "0.0"
        ide_s = f"{grads.get('ide', 0):.2e}" if grads.get('ide', 0) > 0 else "0.0"
        enc_s = f"{grads.get('encoder', 0):.2e}" if grads.get('encoder', 0) > 0 else "0.0"
        dec_s = f"{grads.get('decoder', 0):.2e}" if grads.get('decoder', 0) > 0 else "0.0"
        print(f"  {loss_name:<14} {rgcp_s:>10} {ide_s:>10} {enc_s:>10} {dec_s:>10}")

    print(f"\n  Key observations:")
    upr_grad = components['UPR']
    cma_grad = components['CMA']
    if upr_grad['rgcp'] > 0:
        print(f"  ✓ UPR provides gradient to RGCP (through z_clean → encoder → h_t/c_refined)")
    else:
        print(f"  ⚠️  UPR does NOT provide gradient to RGCP!")
    if cma_grad['rgcp'] > 0:
        print(f"  ⚠️  CMA provides gradient to RGCP (cma_t/c should only depend on pre-RGCP h_t/c)")
    else:
        print(f"  ✓ CMA correctly only flows to pre-RGCP representations (IDE)")
    print(f"  Note: RGCP is trained by: RELIABILITY + ANCHOR + RECOMMENDATION (small weights)")
    print(f"        Plus indirect UPR gradient through encoder backprop to h_t/c_refined")


if __name__ == "__main__":
    print("RGCP Gradient Flow & Architecture Diagnostics")
    print("=" * 70)
    print()

    cos_sim, ratio = analyze_rgcp_gradient_conflict()
    rgcp_grad, enc_grad = analyze_rgcp_gate_gradients()
    ref_t, ref_c = analyze_reliability_network_capacity()
    analyze_alpha_evolution()
    analyze_gradient_paths()

    print("\n" + "=" * 70)
    print("SUMMARY OF FINDINGS")
    print("=" * 70)
    issues = []
    if cos_sim is not None and cos_sim < -0.2:
        issues.append("UPR and Anchor gradients are in CONFLICT")
    if ratio is not None and ratio < 0.001:
        issues.append("Anchor gradient is too WEAK to influence RGCP")
    if ratio is not None and ratio > 0.1:
        issues.append("Anchor gradient may DOMINATE UPR on refined representations")
    if rgcp_grad is not None and enc_grad is not None and rgcp_grad < enc_grad * 1e-3:
        issues.append("RGCP gets very SMALL gradients relative to encoder")
    if ref_t is not None and ref_t < 0.01:
        issues.append("RGCP refinement magnitude is NEGLIGIBLE")

    if issues:
        for i, issue in enumerate(issues):
            print(f"  {i+1}. {issue}")
    else:
        print("  No critical issues detected in gradient flow.")
