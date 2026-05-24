#!/bin/bash
#
# PRISM Stage 1 — Hyperparameter Sensitivity Analysis
#
# Experiments for the three core innovations:
#   1. λ_cma  — Cross-Modal Alignment weight    (IDE + MCD mechanism)
#   2. λ_sac  — SACO sequence contrastive weight (global co-occurrence)
#   3. ide_dim — Shared information bottleneck    (density equalization)
#
# Plus codebook structure ablation:
#   4. (n_layers, n_embed) — capacity-preserving width-vs-depth trade-off
#
# Usage:
#   bash scripts/prism/hparam_sensitivity.sh [DATASET]
#
#   DATASET: beauty (default) | sports | toys | cds
#
# Examples:
#   bash scripts/prism/hparam_sensitivity.sh
#   bash scripts/prism/hparam_sensitivity.sh sports
#

set -euo pipefail

# ── Dataset selection ─────────────────────────────────────────────────────
DATASET="${1:-beauty}"

case "$DATASET" in
    beauty)
        DATA_DIR="Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
        ;;
    sports)
        DATA_DIR="Amazon-Sports/processed/sports-tiger-sentenceT5base/Sports"
        ;;
    toys)
        DATA_DIR="Amazon-Toys/processed/toys-tiger-sentenceT5base/Toys"
        ;;
    cds)
        DATA_DIR="Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs"
        ;;
    *)
        echo "ERROR: Unknown dataset '$DATASET'"
        echo "  Supported: beauty | sports | toys | cds"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_SCRIPT="$PROJECT_ROOT/src/sid_tokenizer/prism/train_prism.py"
DATA_PATH="$PROJECT_ROOT/dataset/$DATA_DIR"
OUTPUT_BASE="$PROJECT_ROOT/scripts/output/prism_tokenizer/$DATASET/hparam_stage1"

echo "================================================================================"
echo "  PRISM Stage 1 — Hyperparameter Sensitivity Analysis"
echo "  Dataset: $DATASET"
echo "  Data:    $DATA_PATH"
echo "  Output:  $OUTPUT_BASE"
echo "================================================================================"

# ── Shared baseline arguments ──────────────────────────────────────────────
BASE_ARGS=(
    --data_path "$DATA_PATH"
    --latent_dim 32
    --content_dim 768
    --collab_dim 64
    --epochs 500
    --batch_size 512
    --learning_rate 1e-4
    --weight_decay 1e-4
    --grad_clip 1.0
    --beta 0.25
    --use_ema --ema_decay 0.99
    --quantize_mode rotation
    --use_scheduler --scheduler_type warmup_cosine --warmup_ratio 0.1
    --early_stop_patience 30 --early_stop_min_delta 1e-5
    --save_every 50
    --device cuda --num_workers 4 --log_level INFO
    --ide on --mcd on
    --use_saco --saco_temperature 0.07
)

RUN_COUNT=0
FAIL_COUNT=0

run_experiment() {
    local name="$1"; shift
    local output_dir="$OUTPUT_BASE/$name"

    RUN_COUNT=$((RUN_COUNT + 1))

    echo ""
    echo "================================================================================"
    echo "  Experiment $RUN_COUNT: $name"
    echo "  Output: $output_dir"
    echo "================================================================================"

    if [ -f "$output_dir/best_model.pt" ]; then
        echo "  [SKIP] best_model.pt already exists — run completed previously"
        return 0
    fi

    mkdir -p "$output_dir"

    cd "$PROJECT_ROOT/src/sid_tokenizer/prism"
    if python "$TRAIN_SCRIPT" \
        --output_dir "$output_dir" \
        "$@" \
        2>&1 | tee "$output_dir/training.log"; then
        echo "  [OK] Completed successfully"
    else
        echo "  [FAIL] Exited with error code $?"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    cd "$PROJECT_ROOT"
}

# ═══════════════════════════════════════════════════════════════════════════
#  Experiment Group 1: λ_cma — Cross-Modal Alignment weight
#  Story: IDE + MCD denoising mechanism. Controls cos(h_t, h_c) alignment.
#  Trade-off: too small → modalities stay orthogonal (s ≈ 0.5)
#             too large  → modality collapse, loses complementarity
#  Values: [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "##############################################################################"
echo "  GROUP 1: λ_cma sensitivity  (λ_sac=0.1, ide_dim=128, L3K256 baseline)"
echo "##############################################################################"

for cma in 0.01 0.05 0.1 0.2 0.5 1.0; do
    run_experiment "stage1_cma_${cma}" \
        "${BASE_ARGS[@]}" \
        --n_layers 3 --n_embed_per_layer "256,256,256" \
        --ide_dim 128 \
        --lambda_cma "$cma" \
        --lambda_sac 0.1
done

# ═══════════════════════════════════════════════════════════════════════════
#  Experiment Group 2: λ_sac — SACO sequence contrastive weight
#  Story: Global co-occurrence structure injection.
#  Trade-off: too small → generated IDs lack sequential structure
#             too large  → contrastive loss dominates, representation distortion
#  Values: [0.01, 0.05, 0.1, 0.2, 0.5]
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "##############################################################################"
echo "  GROUP 2: λ_sac sensitivity  (λ_cma=0.1, ide_dim=128, L3K256 baseline)"
echo "##############################################################################"

for sac in 0.01 0.05 0.1 0.2 0.5; do
    run_experiment "stage1_sac_${sac}" \
        "${BASE_ARGS[@]}" \
        --n_layers 3 --n_embed_per_layer "256,256,256" \
        --ide_dim 128 \
        --lambda_cma 0.1 \
        --lambda_sac "$sac"
done

# ═══════════════════════════════════════════════════════════════════════════
#  Experiment Group 3: ide_dim — Shared information bottleneck
#  Story: Density equalization dimension for 768D text + 64D collab.
#  Trade-off: too small  → text semantics over-compressed, information loss
#             too large  → collab features under-regularized, gradient imbalance returns
#  Values: [32, 64, 128, 256]
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "##############################################################################"
echo "  GROUP 3: ide_dim sensitivity  (λ_cma=0.1, λ_sac=0.1, L3K256 baseline)"
echo "##############################################################################"

for d in 32 64 128 256; do
    run_experiment "stage1_ide_${d}" \
        "${BASE_ARGS[@]}" \
        --n_layers 3 --n_embed_per_layer "256,256,256" \
        --ide_dim "$d" \
        --lambda_cma 0.1 \
        --lambda_sac 0.1
done

# ═══════════════════════════════════════════════════════════════════════════
#  Experiment Group 4: Codebook structure — width vs depth
#  Story: Total capacity ≈ 16M (256³) kept constant via controlled variable.
#    A. L=2, K=1024 → capacity = 1024² ≈ 1.0M  (short seq, large vocab)
#    B. L=3, K=256  → capacity = 256³  = 16.8M  (baseline)
#    C. L=4, K=64   → capacity = 64⁴   ≈ 16.8M  (long seq, small vocab)
#  Trade-off: more layers → longer generation sequence → harder autoregressive
#             larger K  → more collisions at each layer → less precise IDs
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "##############################################################################"
echo "  GROUP 4: Codebook structure  (capacity-preserving L vs K trade-off)"
echo "##############################################################################"

# A: Short sequence, large codebook
run_experiment "stage1_struct_L2K1024" \
    "${BASE_ARGS[@]}" \
    --n_layers 2 --n_embed_per_layer "1024,1024" \
    --ide_dim 128 \
    --lambda_cma 0.1 \
    --lambda_sac 0.1

# B: Baseline (medium)
run_experiment "stage1_struct_L3K256" \
    "${BASE_ARGS[@]}" \
    --n_layers 3 --n_embed_per_layer "256,256,256" \
    --ide_dim 128 \
    --lambda_cma 0.1 \
    --lambda_sac 0.1

# C: Long sequence, small codebook
run_experiment "stage1_struct_L4K64" \
    "${BASE_ARGS[@]}" \
    --n_layers 4 --n_embed_per_layer "64,64,64,64" \
    --ide_dim 128 \
    --lambda_cma 0.1 \
    --lambda_sac 0.1

# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Dataset: $DATASET"
echo "  Total: $RUN_COUNT  |  Failed: $FAIL_COUNT  |  Output base: $OUTPUT_BASE"
echo "================================================================================"
