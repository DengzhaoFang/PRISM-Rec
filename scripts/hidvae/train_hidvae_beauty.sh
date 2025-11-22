#!/bin/bash

# HID-VAE Training Script with Similarity-First Curriculum Strategy
# 
# Strategy: Learn similarity first, then distinctiveness
# Phase 1 (Epochs 1-85): Recon + Class + Anchor (capture similarity)
# Phase 2 (Epochs 86+): Add Balance, reduce Anchor (increase distinctiveness)
cd ../../src/sid_tokenizer/hidvae

python train_hidvae.py \
    --data_path ../../../dataset/Amazon-Beauty/processed/beauty-hidvae-sentenceT5base/Beauty \
    --output_dir ../../../scripts/output/hidvae_tokenizer/beauty/3-256-anti \
    \
    --n_layers 3 \
    --n_embed_per_layer "256,256,256" \
    --latent_dim 32 \
    --content_dim 768 \
    --collab_dim 64 \
    \
    --epochs 500 \
    --batch_size 512 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    \
    --lambda_content 1.0 \
    --lambda_collab 1.0 \
    --delta_weight 0.25 \
    --beta 0.25 \
    --gamma_weight 0.1 \
    --beta_anchor_weight 0.3 \
    \
    --use_ema \
    --ema_decay 0.99 \
    --quantize_mode rotation \
    \
    --use_scheduler \
    --scheduler_type warmup_cosine \
    --warmup_ratio 0.1 \
    \
    --early_stop_patience 30 \
    --early_stop_min_delta 1e-5 \
    --save_every 50 \
    \
    --device cuda \
    --num_workers 4 \
    --log_level INFO \
    \
    --curriculum_strategy similarity_first \
    --curriculum_warmup_epochs 5 \
    --curriculum_phase1_duration 100 \
    --curriculum_class_ramp 20 \
    --curriculum_anchor_delay 5 \
    --curriculum_anchor_ramp 50 \
    --curriculum_anchor_decay 0.1 \
    --curriculum_balance_ramp 60 \
    --curriculum_class_target 1.0 \
    --curriculum_anchor_target 1.0 \
    --curriculum_balance_target 1.0 \
    \
    --use_gate_supervision \
    --gate_supervision_weight 0.8 \
    --gate_diversity_weight 2.0 \
    --gate_target_std 0.2


# default: 原始策略（Class → Balance → Anchor）
# similarity_first: 新策略（Anchor → Balance，先学相似性再学区分性）
# delay = 起跑线（什么时候开始跑）
# ramp = 加速度（多快跑到终点）
# target = 终点线（最终要跑到哪里）
# decay = 开始减速，减速多少

# Expected Results:
# - Uniqueness: 96-98%
# - L1 Purity: 30-45% (vs 15-25% with default)
# - L2 Purity: 20-35% (vs 8-15% with default)
# - L3 Purity: 25-40% (vs 10-18% with default)
# - Codebook Usage: 95-100%
#
# Key Improvements:
# - Better tag alignment (2-3x improvement)
# - Semantic IDs capture both similarity and distinctiveness
# - More stable training (anchor and balance don't conflict)

