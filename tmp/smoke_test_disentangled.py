#!/usr/bin/env python3
"""Smoke test for DisentangledIDE: structure, forward pass, gradient flow, checkpoint compat."""

import sys
sys.path.insert(0, 'src/sid_tokenizer/prism')

import torch
import numpy as np

# ── 1. Module import & structure ──────────────────────────────────────────
print("=" * 60)
print("TEST 1: Import & Structure")
print("=" * 60)

from ide import DisentangledIDE, IDEEqualizer, CM_IDE
assert IDEEqualizer is DisentangledIDE
assert CM_IDE is DisentangledIDE
print("  ✓ All backward compat aliases work")

ide = DisentangledIDE(content_dim=768, collab_dim=64, d=128, shared_dim=64)
total_params = sum(p.numel() for p in ide.parameters())
print(f"  ✓ DisentangledIDE created: {total_params:,} params")
print(f"    Shared dim: {ide.shared_dim}, Specific dim: {ide.specific_dim}")
print(f"    Projections: W_t_shared {tuple(ide.W_t_shared.weight.shape)}")
print(f"                 W_t_specific {tuple(ide.W_t_specific.weight.shape)}")
print(f"                 W_c_shared {tuple(ide.W_c_shared.weight.shape)}")
print(f"                 W_c_specific {tuple(ide.W_c_specific.weight.shape)}")

# ── 2. Forward pass correctness ───────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: Forward Pass")
print("=" * 60)

bs = 64
e_t = torch.randn(bs, 768)
e_c = torch.randn(bs, 64)
out = ide(e_t, e_c)

for key, expected_shape in [
    ('h_t', (bs, 128)), ('h_c', (bs, 128)),
    ('s_t', (bs, 64)), ('s_c', (bs, 64)),
    ('p_t', (bs, 64)), ('p_c', (bs, 64)),
]:
    assert out[key].shape == expected_shape, f"{key}: expected {expected_shape}, got {out[key].shape}"
    print(f"  ✓ {key}: {out[key].shape}")

# ── 3. Full PRISM model forward ───────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: Full PRISM Forward Pass")
print("=" * 60)

from PRISM import PRISM, create_prism_from_config, load_prism_state_dict

config = {
    'content_dim': 768, 'collab_dim': 64,
    'latent_dim': 32, 'n_layers': 3,
    'n_embed_per_layer': [256, 256, 256],
    'use_ide': True, 'ide_dim': 128, 'shared_dim': 64,
    'use_ema': True, 'ema_decay': 0.99, 'beta': 0.25,
    'quantize_mode': 'rotation',
}
model = create_prism_from_config(config)
model.eval()
print(f"  ✓ PRISM created: {sum(p.numel() for p in model.parameters()):,} params")

with torch.no_grad():
    out = model(e_t, e_c, return_codes=True)

for key in ['z_dec', 'z_clean', 'z', 'h_t', 'h_c', 's_t', 's_c', 'p_t', 'p_c',
            'codebook_loss', 'n_used_codes', 'z_q', 'encoding_indices']:
    assert key in out, f"Missing key: {key}"
print(f"  ✓ All expected output keys present")
print(f"    z_dec: {out['z_dec'].shape}, z_clean: {out['z_clean'].shape}")
print(f"    z: {out['z'].shape}, s_t: {out['s_t'].shape}")
print(f"    encoding_indices: {torch.stack(out['encoding_indices'], dim=1).shape}")

# ── 4. Loss computation ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: Loss Computation")
print("=" * 60)

from prism_losses import PRISMTotalLoss, SharedAlignmentLoss, OrthogonalityLoss, UPRLoss

loss_fn = PRISMTotalLoss(
    commitment_weight=0.25, lambda_align=0.1,
    align_temperature=0.07, lambda_ortho=0.01,
)

total_loss, ld = loss_fn(
    z_dec=out['z_dec'], z_clean=out['z_clean'],
    commitment_loss=out['codebook_loss'],
    s_t=out['s_t'], s_c=out['s_c'],
    p_t=out['p_t'], p_c=out['p_c'],
)
print(f"  ✓ Loss computed successfully")
for k, v in ld.items():
    print(f"    {k}: {v:.4f}")

# Test with both losses disabled
total_loss2, ld2 = loss_fn(
    z_dec=out['z_dec'], z_clean=out['z_clean'],
    commitment_loss=out['codebook_loss'],
    s_t=out['s_t'], s_c=out['s_c'],
    p_t=out['p_t'], p_c=out['p_c'],
    align_weight=0.0, ortho_weight=0.0,
)
assert ld2.get('align', 0.0) == 0.0
assert ld2.get('ortho', 0.0) == 0.0
print(f"  ✓ Weight overrides work (align=0, ortho=0)")

# ── 5. Gradient flow ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 5: Gradient Flow")
print("=" * 60)

model.train()
out = model(e_t, e_c, return_codes=True)
total_loss, ld = loss_fn(
    z_dec=out['z_dec'], z_clean=out['z_clean'],
    commitment_loss=out['codebook_loss'],
    s_t=out['s_t'], s_c=out['s_c'],
    p_t=out['p_t'], p_c=out['p_c'],
)
total_loss.backward()

# Check all major modules get gradients
modules_ok = []
for name, module in [
    ('ide.W_t_shared', model.encoder.ide.W_t_shared),
    ('ide.W_t_specific', model.encoder.ide.W_t_specific),
    ('ide.W_c_shared', model.encoder.ide.W_c_shared),
    ('ide.W_c_specific', model.encoder.ide.W_c_specific),
    ('ide.fuse_t', model.encoder.ide.fuse_t),
    ('ide.fuse_c', model.encoder.ide.fuse_c),
    ('encoder', model.encoder.encoder),
    ('decoder', model.decoder),
]:
    has_grad = all(p.grad is not None for p in module.parameters())
    status = "✓" if has_grad else "✗"
    modules_ok.append(has_grad)
    grad_norm = sum(p.grad.norm().item() for p in module.parameters() if p.grad is not None)
    print(f"  {status} {name}: grad_norm={grad_norm:.2e}")

assert all(modules_ok), "Some modules have no gradients!"
print(f"  ✓ All modules receive gradients")

# Check no NaN
has_nan = False
for name, p in model.named_parameters():
    if p.grad is not None and torch.isnan(p.grad).any():
        print(f"  ✗ NaN in {name}.grad!")
        has_nan = True
if not has_nan:
    print(f"  ✓ No NaN gradients")

# ── 6. Gradient conflict check ────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 6: Gradient Conflict (Align vs UPR)")
print("=" * 60)

model.zero_grad(set_to_none=True)
out = model(e_t, e_c, return_codes=True)

# UPR gradient on s_t
upr_grad = torch.zeros_like(out['s_t'])
out['s_t'].retain_grad()
upr_loss = UPRLoss()(out['z_dec'], out['z_clean'])
upr_loss.backward(retain_graph=True)
if out['s_t'].grad is not None:
    upr_grad = out['s_t'].grad.clone()
model.zero_grad(set_to_none=True)

# Align gradient on s_t
out = model(e_t, e_c)
out['s_t'].retain_grad()
align_loss = SharedAlignmentLoss()(out['s_t'], out['s_c'])
align_loss.backward(retain_graph=True)
align_grad = out['s_t'].grad.clone() if out['s_t'].grad is not None else torch.zeros_like(out['s_t'])

cos_sim = torch.nn.functional.cosine_similarity(
    upr_grad.flatten().unsqueeze(0), align_grad.flatten().unsqueeze(0)
).item()
upr_norm = upr_grad.norm().item()
align_norm = align_grad.norm().item()

print(f"  UPR grad norm on s_t:    {upr_norm:.6f}")
print(f"  Align grad norm on s_t:  {align_norm:.6f}")
print(f"  Cosine similarity:       {cos_sim:.4f}")
print(f"  Align/UPR ratio:         {align_norm/(upr_norm+1e-8):.2e}")

if cos_sim < -0.3:
    print(f"  ⚠️  NEGATIVE correlation — Align and UPR may conflict")
elif cos_sim > 0.3:
    print(f"  ✓ POSITIVE correlation — Align supports UPR objective")
else:
    print(f"  ~ Near-orthogonal — Align provides independent signal")

# Compare with old RGCP conflict (Anchor vs UPR was 0.01 cosine)
print(f"\n  Comparison: Old RGCP anchor loss had cosine_sim ≈ 0.01 with UPR")
print(f"  New Align loss has cosine_sim = {cos_sim:.4f} with UPR")

# ── 7. Checkpoint compat ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 7: Checkpoint Save/Load")
print("=" * 60)

import tempfile, os
tmpdir = tempfile.mkdtemp()
cp_path = os.path.join(tmpdir, 'test.pt')

torch.save({
    'epoch': 1, 'global_step': 100,
    'model_state_dict': model.state_dict(),
    'config': config,
}, cp_path)

model2 = create_prism_from_config(config)
cp = torch.load(cp_path, map_location='cpu')
ignored = load_prism_state_dict(model2, cp['model_state_dict'])
print(f"  ✓ Checkpoint loaded: {len(ignored)} keys ignored")
model2.eval()
with torch.no_grad():
    out2 = model2(e_t, e_c)
    assert out2['z_dec'].shape == out['z_dec'].shape
print(f"  ✓ Loaded model produces correct output shapes")

import shutil
shutil.rmtree(tmpdir)

# ── 8. Export interface compat ────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 8: Export Interface (Stage 2 compat)")
print("=" * 60)
# The export writes: item_purified_content.npy (h_t), item_purified_collab.npy (h_c)
# These are 128D each — exact same interface as before
assert out['h_t'].shape[-1] == 128, f"h_t dim={out['h_t'].shape[-1]}, expected 128"
assert out['h_c'].shape[-1] == 128, f"h_c dim={out['h_c'].shape[-1]}, expected 128"
assert out['z_clean'].shape[-1] == 256, f"z_clean dim={out['z_clean'].shape[-1]}, expected 256"
assert out['z'].shape[-1] == 32, f"z dim={out['z'].shape[-1]}, expected 32"
print(f"  ✓ h_t={out['h_t'].shape[-1]}D, h_c={out['h_c'].shape[-1]}D")
print(f"  ✓ z_clean={out['z_clean'].shape[-1]}D, z={out['z'].shape[-1]}D")
print(f"  ✓ Stage 2 interface unchanged")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
