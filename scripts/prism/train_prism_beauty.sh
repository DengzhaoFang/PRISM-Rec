#!/bin/bash
#
# PRISM Stage 1 — Single Experiment Script (Beauty)
#
# Usage:
#   bash scripts/prism/train_prism_beauty.sh
#
# Edit the flags below to switch between alignment paradigms.

cd "$(dirname "$0")/../.."  # project root
source .venv/bin/activate

# ── Data paths ────────────────────────────────────────────────────────────
DATA_PATH="dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
OUTPUT_DIR="scripts/output/prism_tokenizer/beauty/single_exp_cma-only"

# ── Model architecture (fixed) ────────────────────────────────────────────
MODEL_ARGS=(
    --n_layers 3 --n_embed_per_layer "256,256,256" --latent_dim 32
    --content_dim 768 --collab_dim 64
    --ide on --ide_dim 128
)

# ── Training hyperparams ──────────────────────────────────────────────────
TRAIN_ARGS=(
    --epochs 500 --batch_size 512 --learning_rate 1e-4
    --weight_decay 1e-4 --grad_clip 1.0
    --commit_weight 0.25
    --use_ema --ema_decay 0.99 --quantize_mode rotation
    --use_scheduler --scheduler_type warmup_cosine --warmup_ratio 0.1
    --early_stop_patience 50 --early_stop_min_delta 1e-5
    --early_stop_cooldown 3 --early_stop_warmup_epochs 5
    --perplexity_collapse_ratio 0.35 --perplexity_collapse_patience 3
    --kmeans_init_samples 8192
    --save_every 50 --num_workers 4 --log_level INFO
    --device cuda
)

# ═══════════════════════════════════════════════════════════════════════════
# Alignment mode — choose ONE:
# ═══════════════════════════════════════════════════════════════════════════

# **[CMA only] — hard contrastive learning alignment**
ALIGN_ARGS=(
    --lambda_cma 0.1
    --cma_temperature 0.07
)

# **[PA-SCL] — popular-aware soft contrastive alignment (uncomment to use)**
# ALIGN_ARGS=(
#     --use_pa_scl
#     --lambda_pa_scl 0.1
#     --pa_scl_temperature 0.2
#     --pa_scl_topk 5
#     --text_sharpen_gamma 1.0
#     --graph_scale_beta 0.05
# )

# ═══════════════════════════════════════════════════════════════════════════
# Optional: Dual-Head Decoder (uncomment to enable)
# ═══════════════════════════════════════════════════════════════════════════
# DUAL_HEAD_ARGS=(--use_dual_head --dual_head_pop_weight true)
DUAL_HEAD_ARGS=()

# ── Run ───────────────────────────────────────────────────────────────────
python src/sid_tokenizer/prism/train_prism.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    "${MODEL_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${ALIGN_ARGS[@]}" \
    "${DUAL_HEAD_ARGS[@]}"
