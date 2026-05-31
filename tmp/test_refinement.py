"""Test script for RGCMR (Recommendation-Guided Cross-Modal Refinement) module.

Verifies:
1. Module imports and instantiation
2. Forward pass shapes
3. Gradient flow through all new modules
4. Integration with PRISM
5. Loss computation
6. Teacher prototype computation (mock data)
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add the prism directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'src/sid_tokenizer/prism'))

from cross_modal_refinement import CrossModalRefinement, TeacherGuidedCrossGate
from refinement_losses import ContextAlignmentLoss, DriftRegularizer, RefinementLoss
from recommendation_teacher import RecommendationTeacher, build_and_save_teacher
from multimodal_dataset import PRISMDataset
from PRISM import PRISM, MultiModalEncoder, create_prism_from_config
from prism_losses import PRISMTotalLoss


def test_teacher_computation():
    """Test RecommendationTeacher with mock data."""
    print("\n=== Test 1: RecommendationTeacher ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock content/collab embeddings
        num_items = 100
        content_emb = np.random.randn(num_items, 768).astype(np.float32)
        collab_emb = np.random.randn(num_items, 64).astype(np.float32)
        item_id_to_idx = {i: i for i in range(num_items)}

        # Create mock train.parquet
        histories = []
        for _ in range(500):
            hist_len = np.random.randint(1, 6)
            history = list(np.random.choice(num_items, hist_len, replace=False))
            target = np.random.choice(num_items)
            histories.append({'history': history, 'target': target})

        df = pd.DataFrame(histories)
        df.to_parquet(os.path.join(tmpdir, 'train.parquet'))

        # Build teacher
        teacher_obj = RecommendationTeacher(
            data_dir=tmpdir,
            content_embeddings=content_emb,
            collab_embeddings=collab_emb,
            item_id_to_idx=item_id_to_idx,
            recency_gamma=0.5,
            backoff_tau=5.0,
            teacher_dim=256,
        )
        teacher_matrix = teacher_obj.build()

        assert teacher_matrix.shape == (num_items, 256), f"Expected ({num_items}, 256), got {teacher_matrix.shape}"
        assert not np.isnan(teacher_matrix).any(), "Teacher matrix contains NaN"

        # Verify backoff alpha
        assert teacher_obj.alpha.shape == (num_items,)
        assert (teacher_obj.alpha >= 0).all() and (teacher_obj.alpha <= 1).all()

        # Save and reload
        teacher_obj.save(os.path.join(tmpdir, 'teachers'))
        loaded = RecommendationTeacher.load(os.path.join(tmpdir, 'teachers'), num_items)
        assert np.allclose(teacher_matrix, loaded.teacher, atol=1e-5), "Save/load mismatch"

        print("  PASS: Teacher computation, save/load, backoff weights all correct")


def test_cross_modal_refinement():
    """Test CrossModalRefinement forward pass and gradient flow."""
    print("\n=== Test 2: CrossModalRefinement ===")

    batch_size = 32
    ide_dim = 128
    teacher_dim = 256

    refinement = CrossModalRefinement(
        ide_dim=ide_dim,
        teacher_dim=teacher_dim,
        num_layers=2,
    )

    h_t = torch.randn(batch_size, ide_dim, requires_grad=True)
    h_c = torch.randn(batch_size, ide_dim, requires_grad=True)
    teacher = torch.randn(batch_size, teacher_dim)

    # Test forward without anchors
    h_t_r, h_c_r, h_fused = refinement(h_t, h_c, teacher)
    assert h_t_r.shape == (batch_size, ide_dim)
    assert h_c_r.shape == (batch_size, ide_dim)
    assert h_fused.shape == (batch_size, ide_dim * 2)
    print(f"  Forward shapes: h_t={list(h_t_r.shape)}, h_c={list(h_c_r.shape)}, h_fused={list(h_fused.shape)}")

    # Test forward with anchors
    h_t_r2, h_c_r2, h_fused2, anchor_t, anchor_c = refinement(h_t, h_c, teacher, return_anchors=True)
    assert torch.allclose(anchor_t, h_t)
    assert torch.allclose(anchor_c, h_c)

    # Test drift computation
    drift = refinement.compute_drift(h_t_r2, h_c_r2, anchor_t, anchor_c)
    assert drift.item() >= 0
    print(f"  Drift: {drift.item():.6f}")

    # Test gradient flow
    loss = h_fused.sum() + h_t_r.sum() + h_c_r.sum()
    loss.backward()

    # Check that gradients flow back through all layers
    params_with_grad = 0
    params_total = 0
    for name, param in refinement.named_parameters():
        params_total += 1
        if param.grad is not None and param.grad.norm() > 0:
            params_with_grad += 1

    assert params_with_grad == params_total, \
        f"Only {params_with_grad}/{params_total} params received gradients"
    print(f"  Gradient flow: {params_with_grad}/{params_total} params have non-zero gradients")

    # Test residual_scale starts small
    for layer in refinement.layers:
        assert 0 < layer.residual_scale.item() < 0.5, \
            f"residual_scale should start small, got {layer.residual_scale.item()}"
    print("  Residual scale check: conservative initialization confirmed")

    print("  PASS: Forward shapes, gradient flow, residual init all correct")


def test_refinement_losses():
    """Test refinement loss functions."""
    print("\n=== Test 3: RefinementLosses ===")

    batch_size = 32
    dim = 256

    # ContextAlignmentLoss
    ctx_loss_fn = ContextAlignmentLoss(contrastive_weight=0.1)
    h_fused = torch.randn(batch_size, dim)
    teacher = torch.randn(batch_size, dim)

    loss, info = ctx_loss_fn(h_fused, teacher)
    assert loss.item() > 0
    assert 'cos_align' in info
    assert 'contrastive' in info
    print(f"  ContextAlignment: loss={loss.item():.4f}, cos_align={info['cos_align']:.3f}")

    # Perfect alignment case
    h_perfect = teacher.clone()
    loss_perfect, info_perfect = ctx_loss_fn(h_perfect, teacher)
    assert loss_perfect.item() < 0.1  # Should be near zero for perfect match
    print(f"  Perfect alignment: loss={loss_perfect.item():.6f}, cos_align={info_perfect['cos_align']:.3f}")

    # DriftRegularizer
    drift_reg = DriftRegularizer()
    h_t = torch.randn(batch_size, 128)
    h_c = torch.randn(batch_size, 128)
    drift = drift_reg(h_t, h_c, h_t, h_c)  # No drift
    assert drift.item() < 1e-5
    drift2 = drift_reg(h_t, h_c, h_t + 1.0, h_c + 1.0)  # Large drift
    assert drift2.item() > 0.5
    print(f"  Drift: no-drift={drift.item():.6f}, large-drift={drift2.item():.4f}")

    # Combined RefinementLoss
    refine_loss = RefinementLoss(lambda_ctx=0.1, lambda_drift=0.05)
    total, ld = refine_loss(h_fused, teacher, h_t, h_c, h_t, h_c)
    assert total.item() > 0
    assert 'refine_ctx' in ld
    assert 'refine_drift' in ld
    print(f"  Combined: total={total.item():.4f}, ctx={ld['refine_ctx']:.4f}, drift={ld['refine_drift']:.4f}")

    print("  PASS: All loss functions work correctly")


def test_prism_integration():
    """Test full PRISM model with refinement module."""
    print("\n=== Test 4: PRISM + RGCMR Integration ===")

    batch_size = 16
    content_dim = 768
    collab_dim = 64
    ide_dim = 128
    latent_dim = 32
    teacher_dim = 256

    model = PRISM(
        content_dim=content_dim, collab_dim=collab_dim,
        latent_dim=latent_dim, n_layers=3,
        n_embed=256, n_embed_per_layer=[256, 256, 256],
        use_ide=True, ide_dim=ide_dim,
        use_refinement=True, refinement_layers=2,
        teacher_dim=teacher_dim,
        use_ema=True,
    )

    content = torch.randn(batch_size, content_dim)
    collab = torch.randn(batch_size, collab_dim)
    teacher = torch.randn(batch_size, teacher_dim)

    # Forward pass without refinement (teacher=None)
    out_no_refine = model(content, collab, return_codes=True, teacher=None)
    # Should still work (refinement is bypassed when teacher is None)
    assert 'z_dec' in out_no_refine
    print(f"  Without teacher: z_dec={list(out_no_refine['z_dec'].shape)}, z={list(out_no_refine['z'].shape)}")

    # Forward pass with refinement
    out = model(content, collab, return_codes=True, teacher=teacher)
    assert out['z_dec'].shape == (batch_size, model.output_dim)
    assert out['z'].shape == (batch_size, latent_dim)
    assert out['h_t'].shape == (batch_size, ide_dim)
    assert out['h_c'].shape == (batch_size, ide_dim)
    assert out['h_t_raw'] is not None
    assert out['anchor_t'] is not None
    assert out['z_clean'].shape == (batch_size, ide_dim * 2)

    print(f"  With teacher: z_dec={list(out['z_dec'].shape)}, h_t={list(out['h_t'].shape)}")

    # Test loss computation
    loss_fn = PRISMTotalLoss(
        commitment_weight=0.25,
        lambda_refine=0.1, lambda_drift=0.05,
    )

    total, ld = loss_fn(
        z_dec=out['z_dec'], z_clean=out['z_clean'].detach(),
        commitment_loss=out['codebook_loss'],
        h_fused=out['z_clean'],
        teacher=teacher,
        h_t=out['h_t'], h_c=out['h_c'],
        anchor_t=out['anchor_t'], anchor_c=out['anchor_c'],
    )

    assert total.requires_grad, "Loss should require gradients"
    assert 'refine_ctx' in ld
    assert 'refine_drift' in ld
    print(f"  Total loss: {total.item():.4f}, UPR={ld['upr']:.4f}, r_ctx={ld['refine_ctx']:.4f}, r_drift={ld['refine_drift']:.4f}")

    # Test gradient flow end-to-end
    total.backward()

    # Check gradients reach refinement module
    refinement = model.encoder.refinement
    refine_grad_norms = []
    for name, param in refinement.named_parameters():
        if param.grad is not None:
            refine_grad_norms.append(param.grad.norm().item())
    assert len(refine_grad_norms) > 0, "No gradients in refinement module!"
    print(f"  Refinement grad norms: min={min(refine_grad_norms):.6f}, max={max(refine_grad_norms):.6f}")

    # Check gradients reach IDE
    ide = model.encoder.ide
    ide_grad_norms = []
    for name, param in ide.named_parameters():
        if param.grad is not None:
            ide_grad_norms.append(param.grad.norm().item())
    assert len(ide_grad_norms) > 0, "No gradients in IDE module!"
    print(f"  IDE grad norms: min={min(ide_grad_norms):.6f}, max={max(ide_grad_norms):.6f}")

    # Verify no NaN in any parameter
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"
    print("  NaN check: OK (no NaN in any gradients)")

    print("  PASS: Full integration works, gradients flow correctly")


def test_config_creation():
    """Test create_prism_from_config with refinement args."""
    print("\n=== Test 5: create_prism_from_config ===")

    config = {
        'use_refinement': True,
        'refinement_layers': 2,
        'teacher_dim': 256,
        'lambda_refine': 0.1,
        'lambda_drift': 0.05,
    }
    model = create_prism_from_config(config)
    assert model.use_refinement == True
    assert model.encoder.refinement is not None
    assert len(model.encoder.refinement.layers) == 2
    print("  PASS: Config-based creation works correctly")


def test_without_teacher():
    """Test that refinement gracefully handles missing teacher."""
    print("\n=== Test 6: Graceful degradation without teacher ===")

    model = PRISM(
        use_ide=True, ide_dim=128,
        use_refinement=True, refinement_layers=2,
        teacher_dim=256, use_ema=True,
    )

    content = torch.randn(8, 768)
    collab = torch.randn(8, 64)

    # Without teacher - should fall back to standard path
    out = model(content, collab, teacher=None)
    assert out['z_dec'].shape == (8, 256)
    assert out['h_t_raw'] is None  # No raw because refinement was skipped
    assert out['anchor_t'] is None
    print("  No teacher: bypasses refinement correctly")

    # With teacher - full refinement path
    teacher = torch.randn(8, 256)
    out2 = model(content, collab, teacher=teacher)
    assert out2['h_t_raw'] is not None
    assert out2['anchor_t'] is not None
    print("  With teacher: full refinement path active")

    print("  PASS: Graceful degradation works correctly")


def test_save_load_forward():
    """Test model save/load roundtrip with refinement module."""
    print("\n=== Test 7: Save/Load roundtrip ===")

    import tempfile

    model = PRISM(
        use_ide=True, ide_dim=128,
        use_refinement=True, refinement_layers=2,
        teacher_dim=256, use_ema=True, n_layers=2,
        n_embed=64, n_embed_per_layer=[64, 64],
    )

    content = torch.randn(4, 768)
    collab = torch.randn(4, 64)
    teacher = torch.randn(4, 256)

    # Get output before save
    model.eval()
    with torch.no_grad():
        out_before = model(content, collab, teacher=teacher, return_codes=True)

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'model.pt')
        torch.save({'model_state_dict': model.state_dict()}, path)

        # Create fresh model and load
        model2 = PRISM(
            use_ide=True, ide_dim=128,
            use_refinement=True, refinement_layers=2,
            teacher_dim=256, use_ema=True, n_layers=2,
            n_embed=64, n_embed_per_layer=[64, 64],
        )

        cp = torch.load(path)
        from PRISM import load_prism_state_dict
        load_prism_state_dict(model2, cp['model_state_dict'])
        model2.eval()

        with torch.no_grad():
            out_after = model2(content, collab, teacher=teacher, return_codes=True)

        # Compare outputs (non-EMA codebook creates randomness, compare encoded features)
        assert torch.allclose(out_before['z_clean'], out_after['z_clean'], atol=1e-5), \
            "z_clean mismatch after load"
        assert torch.allclose(out_before['h_t'], out_after['h_t'], atol=1e-5), \
            "h_t mismatch after load"

    print("  PASS: Save/load roundtrip preserves outputs")


if __name__ == '__main__':
    print("=" * 70)
    print("RGCMR Module — Comprehensive Test Suite")
    print("=" * 70)

    test_teacher_computation()
    test_cross_modal_refinement()
    test_refinement_losses()
    test_prism_integration()
    test_config_creation()
    test_without_teacher()
    test_save_load_forward()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
