#!/usr/bin/env python3
"""
Test HID-VAE Components

Quick test to verify all components work correctly without full training.
"""

import torch
import numpy as np
from pathlib import Path

print("Testing HID-VAE components...")
print("=" * 80)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from HID_VAE import HIDVAE, create_hidvae_from_config
    from multimodal_dataset import HIDVAEDataset, create_dataloaders
    from hid_losses import HIDVAETotalLoss
    from hierarchical_classifiers import HierarchicalClassifiers
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: Create dummy model
print("\n2. Testing model creation...")
try:
    config = {
        'content_dim': 768,
        'collab_dim': 64,
        'latent_dim': 32,
        'n_layers': 3,
        'n_embed': 256,
        'use_ema': True,
        'ema_decay': 0.99,
        'beta': 0.25,
        'quantize_mode': 'rotation'
    }
    
    num_classes_per_layer = [7, 38, 149]  # L2, L3, L4 (with PAD)
    
    model = create_hidvae_from_config(config, num_classes_per_layer)
    print(f"   ✓ Model created")
    print(f"     - Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    exit(1)

# Test 3: Test forward pass with dummy data
print("\n3. Testing forward pass...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    batch_size = 4
    content_emb = torch.randn(batch_size, 768).to(device)
    collab_emb = torch.randn(batch_size, 64).to(device)
    
    with torch.no_grad():
        outputs = model(content_emb, collab_emb, return_codes=True)
    
    print(f"   ✓ Forward pass successful")
    print(f"     - Content recon shape: {outputs['content_recon'].shape}")
    print(f"     - Collab recon shape: {outputs['collab_recon'].shape}")
    print(f"     - Semantic IDs shape: ({len(outputs['encoding_indices'])}, {batch_size})")
    
    if 'predictions' in outputs:
        print(f"     - Predictions: {len(outputs['predictions'])} layers")
except Exception as e:
    print(f"   ✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test loss computation
print("\n4. Testing loss computation...")
try:
    loss_fn = HIDVAETotalLoss(
        lambda_content=1.0,
        lambda_collab=1.0,
        tag_embed_dim=768,
        codebook_dim=32,
        beta_weights=None,
        gamma_weights=None,
        delta_weight=1.0,
        commitment_weight=0.25,
        n_layers=3,
        ignore_index=0
    ).to(device)
    
    # Dummy tag embeddings (per layer)
    tag_embeddings_per_layer = [
        torch.randn(6, 768).to(device),   # L2: 6 tags
        torch.randn(37, 768).to(device),  # L3: 37 tags
        torch.randn(148, 768).to(device)  # L4: 148 tags
    ]
    
    # Get codebooks
    codebooks = model.get_codebooks()
    
    # Dummy targets
    tag_ids = torch.randint(0, 7, (batch_size, 3)).to(device)
    tag_mask = torch.ones(batch_size, 3).to(device)
    
    targets_per_layer = [tag_ids[:, i] for i in range(3)]
    masks_per_layer = [tag_mask[:, i] for i in range(3)]
    
    # Compute loss
    total_loss, loss_dict = loss_fn(
        pred_content=outputs['content_recon'],
        target_content=content_emb,
        pred_collab=outputs['collab_recon'],
        target_collab=collab_emb,
        tag_embeddings_per_layer=tag_embeddings_per_layer,
        codebooks=codebooks,
        encoding_indices_per_layer=outputs['encoding_indices'],
        n_embed_per_layer=[256, 256, 256],
        predictions_per_layer=outputs.get('predictions', []),
        targets_per_layer=targets_per_layer,
        masks_per_layer=masks_per_layer,
        commitment_loss=outputs['codebook_loss']
    )
    
    print(f"   ✓ Loss computation successful")
    print(f"     - Total loss: {total_loss.item():.4f}")
    print(f"     - Recon loss: {loss_dict['recon_total']:.4f}")
    print(f"     - Anchor loss: {loss_dict['anchor_total']:.4f}")
    print(f"     - Balance loss: {loss_dict['balance_total']:.4f}")
    if 'class_total' in loss_dict:
        print(f"     - Class loss: {loss_dict['class_total']:.4f}")
except Exception as e:
    print(f"   ✗ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test semantic ID generation
print("\n5. Testing semantic ID generation...")
try:
    with torch.no_grad():
        semantic_ids = model.generate_semantic_ids(content_emb, collab_emb)
    
    print(f"   ✓ Semantic ID generation successful")
    print(f"     - Shape: {semantic_ids.shape}")
    print(f"     - Sample IDs:")
    for i in range(min(3, batch_size)):
        print(f"       Item {i}: {semantic_ids[i].cpu().numpy()}")
except Exception as e:
    print(f"   ✗ Semantic ID generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Test backward pass
print("\n6. Testing backward pass...")
try:
    model.train()
    
    # Use larger batch size for k-means initialization
    batch_size_large = 512
    content_emb_large = torch.randn(batch_size_large, 768).to(device)
    collab_emb_large = torch.randn(batch_size_large, 64).to(device)
    tag_ids_large = torch.randint(0, 7, (batch_size_large, 3)).to(device)
    tag_mask_large = torch.ones(batch_size_large, 3).to(device)
    targets_per_layer_large = [tag_ids_large[:, i] for i in range(3)]
    masks_per_layer_large = [tag_mask_large[:, i] for i in range(3)]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Forward with large batch
    outputs = model(content_emb_large, collab_emb_large, return_codes=True)
    
    # Compute loss
    total_loss, _ = loss_fn(
        pred_content=outputs['content_recon'],
        target_content=content_emb_large,
        pred_collab=outputs['collab_recon'],
        target_collab=collab_emb_large,
        tag_embeddings_per_layer=tag_embeddings_per_layer,
        codebooks=model.get_codebooks(),
        encoding_indices_per_layer=outputs['encoding_indices'],
        n_embed_per_layer=[256, 256, 256],
        predictions_per_layer=outputs.get('predictions', []),
        targets_per_layer=targets_per_layer_large,
        masks_per_layer=masks_per_layer_large,
        commitment_loss=outputs['codebook_loss']
    )
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"   ✓ Backward pass successful")
    print(f"     - Gradients computed and optimizer step completed")
except Exception as e:
    print(f"   ✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✓ All tests passed!")
print("=" * 80)
print("\nHID-VAE is ready to use. You can now:")
print("1. Train the model using train_hidvae.py")
print("2. Generate semantic IDs using generate_semantic_ids.py")
print("3. Resolve collisions using sinkhorn_reassignment.py")

