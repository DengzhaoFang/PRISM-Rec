#!/bin/bash

# PRISM Training Script (without HSA modules)
#
# Removed: Tag Anchoring Loss, Codebook Balance Loss, Hierarchical Classification Loss
# Kept: Reconstruction Loss + Commitment Loss + Gate Supervision


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
    --use_gate_supervision \
    --gate_supervision_weight 0.8 \
    --gate_diversity_weight 3.5 \
    --gate_target_std 0.3
