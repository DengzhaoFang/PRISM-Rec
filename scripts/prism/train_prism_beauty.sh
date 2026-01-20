#!/bin/bash

# PRISM Training Script with Similarity-First Curriculum Strategy
# 
# Strategy: Learn similarity first, then distinctiveness
# Phase 1 (Epochs 1-85): Recon + Class + Anchor (capture similarity)
# Phase 2 (Epochs 86+): Add Balance, reduce Anchor (increase distinctiveness)


# --beta_anchor_weight 0.3 \
#  --delta_weight 0.25 \

# Decoder Mode Control:
# Default (dual decoder): Separate decoder heads for content (768D) and collab (64D)
# Single decoder: Add --no_dual_decoder to use single decoder for concatenated embedding (832D)

cd ../../src/sid_tokenizer/prism

python train_prism.py \
    --data_path ../../../dataset/Amazon-Beauty/processed/beauty-prism-sentenceT5base/Beauty \
    --output_dir ../../../scripts/output/prism_tokenizer/beauty/3-256-32-wo-hsa-4 \
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
    --lambda_collab 2.0 \
    --beta 0.25 \
    --delta_weight 0 \
    --beta_anchor_weight 0 \
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
    --gate_diversity_weight 3.5 \
    --gate_target_std 0.3 \




    # --delta_weight 0.0 \ # wo HSA
    # --beta_anchor_weight 0.0 \

    # --no_gated_fusion # wo ACD
    # --no_dual_decoder # wo DHR


