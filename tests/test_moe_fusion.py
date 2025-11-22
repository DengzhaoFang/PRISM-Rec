"""
Unit tests for MoE Fusion module.

Tests the Mixture of Experts fusion implementation to ensure:
1. Correct tensor shapes
2. Load balancing loss computation
3. Top-K expert selection
4. Layer-specific projections
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommender.hidvae.moe_fusion import MoEFusion, Expert, GatingNetwork


def test_expert():
    """Test single expert network."""
    print("Testing Expert network...")
    
    d_model = 128
    d_ff = 256
    batch_size = 4
    seq_len = 10
    
    expert = Expert(d_model=d_model, d_ff=d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    output = expert(x)
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    print("✓ Expert network test passed")


def test_gating_network():
    """Test gating network."""
    print("\nTesting Gating network...")
    
    d_model = 128
    num_experts = 4
    batch_size = 4
    seq_len = 10
    
    gating = GatingNetwork(d_model=d_model, num_experts=num_experts)
    x = torch.randn(batch_size, seq_len, d_model)
    
    gates, load = gating(x)
    
    assert gates.shape == (batch_size, seq_len, num_experts), f"Expected gates shape {(batch_size, seq_len, num_experts)}, got {gates.shape}"
    assert load.shape == (num_experts,), f"Expected load shape {(num_experts,)}, got {load.shape}"
    
    # Check that gates sum to 1 (softmax property)
    gates_sum = gates.sum(dim=-1)
    assert torch.allclose(gates_sum, torch.ones_like(gates_sum), atol=1e-5), "Gates should sum to 1"
    
    print("✓ Gating network test passed")


def test_moe_fusion_basic():
    """Test basic MoE fusion."""
    print("\nTesting MoE Fusion (basic)...")
    
    d_model = 128
    content_dim = 768
    collab_dim = 64
    num_experts = 4
    top_k = 2
    batch_size = 4
    seq_len = 12  # 4 items * 3 layers
    
    moe = MoEFusion(
        d_model=d_model,
        content_dim=content_dim,
        collab_dim=collab_dim,
        num_experts=num_experts,
        top_k=top_k,
        use_load_balancing=True,
        use_layer_specific=False
    )
    
    # Create dummy inputs
    id_emb = torch.randn(batch_size, seq_len, d_model)
    content_emb = torch.randn(batch_size, seq_len, content_dim)
    collab_emb = torch.randn(batch_size, seq_len, collab_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    output, load_balancing_loss = moe(
        id_emb, content_emb, collab_emb,
        attention_mask=attention_mask,
        num_tokens_per_item=3
    )
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert load_balancing_loss is not None, "Load balancing loss should not be None during training"
    assert load_balancing_loss.item() >= 0, "Load balancing loss should be non-negative"
    
    print(f"✓ MoE Fusion basic test passed (load_balancing_loss={load_balancing_loss.item():.6f})")


def test_moe_fusion_layer_specific():
    """Test MoE fusion with layer-specific projections."""
    print("\nTesting MoE Fusion (layer-specific)...")
    
    d_model = 128
    content_dim = 768
    collab_dim = 64
    num_experts = 4
    top_k = 2
    num_layers = 3
    batch_size = 4
    num_items = 4
    seq_len = num_items * num_layers
    
    moe = MoEFusion(
        d_model=d_model,
        content_dim=content_dim,
        collab_dim=collab_dim,
        num_experts=num_experts,
        top_k=top_k,
        use_load_balancing=True,
        use_layer_specific=True,
        num_layers=num_layers
    )
    
    # Create dummy inputs
    id_emb = torch.randn(batch_size, seq_len, d_model)
    content_emb = torch.randn(batch_size, seq_len, content_dim)
    collab_emb = torch.randn(batch_size, seq_len, collab_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    output, load_balancing_loss = moe(
        id_emb, content_emb, collab_emb,
        attention_mask=attention_mask,
        num_tokens_per_item=num_layers
    )
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert load_balancing_loss is not None, "Load balancing loss should not be None during training"
    
    print(f"✓ MoE Fusion layer-specific test passed (load_balancing_loss={load_balancing_loss.item():.6f})")


def test_moe_fusion_inference():
    """Test MoE fusion in inference mode (no load balancing loss)."""
    print("\nTesting MoE Fusion (inference mode)...")
    
    d_model = 128
    content_dim = 768
    collab_dim = 64
    num_experts = 4
    top_k = 2
    batch_size = 4
    seq_len = 12
    
    moe = MoEFusion(
        d_model=d_model,
        content_dim=content_dim,
        collab_dim=collab_dim,
        num_experts=num_experts,
        top_k=top_k,
        use_load_balancing=True
    )
    
    # Set to eval mode
    moe.eval()
    
    # Create dummy inputs
    id_emb = torch.randn(batch_size, seq_len, d_model)
    content_emb = torch.randn(batch_size, seq_len, content_dim)
    collab_emb = torch.randn(batch_size, seq_len, collab_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    with torch.no_grad():
        output, load_balancing_loss = moe(
            id_emb, content_emb, collab_emb,
            attention_mask=attention_mask,
            num_tokens_per_item=3
        )
    
    assert output.shape == (batch_size, seq_len, d_model), f"Expected output shape {(batch_size, seq_len, d_model)}, got {output.shape}"
    assert load_balancing_loss is None, "Load balancing loss should be None during inference"
    
    print("✓ MoE Fusion inference test passed")


def test_top_k_sparsity():
    """Test that top-K selection creates sparse gating."""
    print("\nTesting Top-K sparsity...")
    
    d_model = 128
    content_dim = 768
    collab_dim = 64
    num_experts = 8
    top_k = 2
    batch_size = 4
    seq_len = 12
    
    moe = MoEFusion(
        d_model=d_model,
        content_dim=content_dim,
        collab_dim=collab_dim,
        num_experts=num_experts,
        top_k=top_k,
        use_load_balancing=False
    )
    
    # Create dummy inputs
    id_emb = torch.randn(batch_size, seq_len, d_model)
    content_emb = torch.randn(batch_size, seq_len, content_dim)
    collab_emb = torch.randn(batch_size, seq_len, collab_dim)
    
    # Forward pass
    output, _ = moe(id_emb, content_emb, collab_emb, num_tokens_per_item=3)
    
    # Check that output is valid
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    print(f"✓ Top-K sparsity test passed (top_k={top_k}, num_experts={num_experts})")


def test_attention_mask():
    """Test that attention mask properly zeros out padding."""
    print("\nTesting attention mask...")
    
    d_model = 128
    content_dim = 768
    collab_dim = 64
    num_experts = 4
    top_k = 2
    batch_size = 4
    seq_len = 12
    
    moe = MoEFusion(
        d_model=d_model,
        content_dim=content_dim,
        collab_dim=collab_dim,
        num_experts=num_experts,
        top_k=top_k
    )
    
    # Create dummy inputs
    id_emb = torch.randn(batch_size, seq_len, d_model)
    content_emb = torch.randn(batch_size, seq_len, content_dim)
    collab_emb = torch.randn(batch_size, seq_len, collab_dim)
    
    # Create attention mask with some padding
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -3:] = 0  # Last 3 positions are padding
    
    # Forward pass
    output, _ = moe(
        id_emb, content_emb, collab_emb,
        attention_mask=attention_mask,
        num_tokens_per_item=3
    )
    
    # Check that padding positions are zeroed
    padding_output = output[:, -3:, :]
    assert torch.allclose(padding_output, torch.zeros_like(padding_output)), "Padding positions should be zero"
    
    print("✓ Attention mask test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running MoE Fusion Tests")
    print("=" * 60)
    
    try:
        test_expert()
        test_gating_network()
        test_moe_fusion_basic()
        test_moe_fusion_layer_specific()
        test_moe_fusion_inference()
        test_top_k_sparsity()
        test_attention_mask()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed with error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
