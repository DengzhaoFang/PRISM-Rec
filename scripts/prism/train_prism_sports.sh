#!/bin/bash

# PRISM Training Script with Similarity-First Curriculum Strategy
# 
# Strategy: Learn similarity first, then distinctiveness
# Phase 1 (Epochs 1-85): Recon + Class + Anchor (capture similarity)
# Phase 2 (Epochs 86+): Add Balance, reduce Anchor (increase distinctiveness)
cd ../../src/sid_tokenizer/prism

python train_prism.py \
    --data_path ../../../dataset/Amazon-Sports/processed/sports-prism-sentenceT5base/Sports \
    --output_dir ../../../scripts/output/prism_tokenizer/sports/3-256-32-ema-only-5-core-items  \
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



