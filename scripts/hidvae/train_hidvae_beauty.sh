#!/bin/bash
# HID-VAE Training Script for Amazon Beauty Dataset
# 
# Features:
# - Multi-modal inputs (content + collaborative embeddings)
# - Tag-guided hierarchical learning
# - 3-layer RQ-VAE with codebook sizes [256, 256, 256]
# - EMA codebook updates
# - Hierarchical classification loss
# - Tag anchoring loss
# - Codebook balance loss

cd ../../src/sid_tokenizer/hidvae

python train_hidvae.py \
    --data_path ../../../dataset/Amazon-Beauty/processed/beauty-hidvae-sentenceT5base/Beauty \
    --output_dir ../../../scripts/output/hidvae_tokenizer/beauty/3-256-32-hidvae-multimodal-1000epoch \
    \
    --n_layers 3 \
    --n_embed 256 \
    --latent_dim 32 \
    --content_dim 768 \
    --collab_dim 64 \
    \
    --epochs 1000 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --grad_clip 1.0 \
    \
    --lambda_content 1.0 \
    --lambda_collab 1.0 \
    --delta_weight 0.3 \
    --beta 0.25 \
    --gamma_weight 0.01 \
    --beta_anchor_weight 0.5 \
    \
    --use_ema \
    --ema_decay 0.99 \
    --quantize_mode rotation \
    \
    --use_scheduler \
    --scheduler_type warmup_cosine \
    --warmup_ratio 0.1 \
    \
    --early_stop_patience 100 \
    --early_stop_min_delta 1e-4 \
    --save_every 100 \
    \
    --device cuda \
    --num_workers 4 \
    --log_level INFO

echo ""
echo "======================================"
echo "✓ HID-VAE Training completed!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Check training logs and metrics"
echo "2. Generate semantic IDs for all items"
echo "3. Analyze hierarchical overlap rates"
echo "4. Apply Sinkhorn for ID uniqueness"
echo ""

