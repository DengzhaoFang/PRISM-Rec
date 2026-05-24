#!/bin/bash

# PRISM Training Script with IDE + MCD + UPR + CMA + SACO
#
# Pipeline: IDE -> MCD -> z_clean(256D) -> Encoder -> RQ-VAE -> UnifiedDecoder -> z_dec(256D)
# Loss: L_UPR + beta * L_commit + lambda_cma * L_CMA + lambda_sac * L_SACO
#   UPR = MSE(z_dec, z_clean.detach())   -- local fidelity
#   CMA = InfoNCE(h_t, h_c)              -- cross-modal alignment (fixes cos≈0)
#   SACO = InfoNCE(z_anchor, z_pos)      -- global co-occurrence structure (paired)
#
# Ablation guide:
#   --ide off          disable IDE + CMA, revert to raw 768D+64D input
#   --mcd off          disable MCD, skip cross-modal denoising
#   remove --use_saco  disable SACO contrastive loss
#
# Key hyperparameters:
#   --ide_dim 128            IDE projection dimension d (z_clean = 2*d = 256D)
#   --lambda_cma 0.1         CMA cross-modal alignment weight
#   --lambda_sac 0.1         SACO loss weight
#   --saco_temperature 0.07  SACO softmax temperature


cd ../../src/sid_tokenizer/prism

python train_prism.py \
    --data_path ../../../dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty \
    --output_dir ../../../scripts/output/prism_tokenizer/beauty/3-256-32-ide+mcd+saco-cleancollab \
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
    --ide on \
    --ide_dim 128 \
    --mcd on \
    --lambda_cma 0.1 \
    --use_saco \
    --lambda_sac 0.1 \
    --saco_temperature 0.07
