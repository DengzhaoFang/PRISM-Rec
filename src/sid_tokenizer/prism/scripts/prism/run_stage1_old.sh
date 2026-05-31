#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
OUTPUT_DIR="${1:-$PROJECT_ROOT/scripts/output/prism_tokenizer/beauty/old_restored}"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT/src/sid_tokenizer/prism"
PYTHONPATH="$PROJECT_ROOT/src/sid_tokenizer/prism:${PYTHONPATH:-}" \
python train_prism.py \
    --data_path "$DATA_DIR" --output_dir "$OUTPUT_DIR" \
    --n_layers 3 --n_embed_per_layer 256,256,256 --latent_dim 32 \
    --content_dim 768 --collab_dim 64 \
    --ide on --mcd on --ide_dim 128 \
    --use_saco --lambda_sac 0.1 --saco_temperature 0.07 \
    --lambda_cma 0.1 --cma_temperature 0.07 \
    --epochs 500 --batch_size 512 --learning_rate 1e-4 \
    --weight_decay 1e-4 --grad_clip 1.0 \
    --beta 0.25 --use_ema --ema_decay 0.99 --quantize_mode rotation \
    --use_scheduler --scheduler_type warmup_cosine --warmup_ratio 0.1 \
    --early_stop_patience 30 --early_stop_min_delta 1e-5 \
    --save_every 50 --num_workers 4 --log_level INFO \
    --no_hierarchical_kmeans_init --kmeans_init_samples 8192
