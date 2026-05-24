#!/bin/bash
#
# PRISM Stage 2 — Hyperparameter Sensitivity Runner
#
# Iterates over all Stage 1 hparam experiment outputs, runs Stage 2
# recommender training for each using default hyperparameters.
#
# Features:
#   - Dynamic GPU VRAM sniffing (threshold 4000 MB)
#   - Auto GPU assignment, max concurrent jobs
#   - Staggered startup to avoid I/O thrashing
#   - Resume capability (auto-skip completed experiments)
#   - Clean progress dashboard
#
# Usage:
#   bash scripts/prism/hparam_sensitivity_stage1Rec.sh [DATASET]
#
#   DATASET: beauty (default) | sports | toys | cds
#

set -euo pipefail

# ── Dataset selection ─────────────────────────────────────────────────────
DATASET="${1:-beauty}"


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

STAGE1_BASE="$PROJECT_ROOT/scripts/output/prism_tokenizer/$DATASET/hparam_stage1"
STAGE1_Rec="$PROJECT_ROOT/scripts/output/recommender/prism/$DATASET/hparam_stage1Rec"

# ── Scheduler configuration ───────────────────────────────────────────────
VRAM_THRESHOLD=8500       # Minimum free VRAM (MB) — ~2 jobs per 24GB GPU with beam=30 eval spikes
MAX_CONCURRENT=6          # Global max concurrent jobs
STAGGER_DELAY=15          # Seconds between job launches (I/O smoothing)

# ── Shared Stage 2 arguments ──────────────────────────────────────────────
CONFIG="$DATASET"
MODEL_TYPE="t5-tiny-2"
DEVICE="cuda:0"            # Always cuda:0 inside CUDA_VISIBLE_DEVICES sandbox
NUM_WORKERS=4

mkdir -p "$STAGE1_Rec"

# ── Counters ──────────────────────────────────────────────────────────────
TOTAL=0
COMPLETED=0
RUNNING=0
FAILED=0
SKIPPED=0

# job tracker: each background job writes its status to a file on exit
STATUS_DIR="$STAGE1_Rec/.status"
mkdir -p "$STATUS_DIR"

# ── Dashboard ─────────────────────────────────────────────────────────────
show_dashboard() {
    # Count statuses from status files (these are written by background jobs)
    local c=$(find "$STATUS_DIR" -name "ok_*" 2>/dev/null | wc -l)
    local f=$(find "$STATUS_DIR" -name "fail_*" 2>/dev/null | wc -l)
    local r=$(jobs -r | wc -l)

    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Stage 2 HParam Sweep — $DATASET                                   ║"
    printf "║  Done: %-3d  Running: %-3d  Failed: %-3d  Queued: %-3d        ║\n" \
        "$c" "$r" "$f" "$((TOTAL - c - f - r))"
    echo "╚══════════════════════════════════════════════════════════════╝"
}

# ── GPU detector ──────────────────────────────────────────────────────────
get_free_gpu() {
    while true; do
        local free_mem
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo "")
        if [ -z "$free_mem" ]; then
            sleep 30
            continue
        fi

        # Pick the GPU with the MOST free VRAM (natural load balancing)
        local best_gpu=-1
        local best_mem=0
        local gpu_id=0
        for mem in $free_mem; do
            if [ "$mem" -ge "$VRAM_THRESHOLD" ] && [ "$mem" -gt "$best_mem" ]; then
                best_gpu=$gpu_id
                best_mem=$mem
            fi
            gpu_id=$((gpu_id + 1))
        done

        if [ "$best_gpu" -ge 0 ]; then
            echo "$best_gpu"
            return 0
        fi
        sleep 30
    done
}

# ── Experiment runner ─────────────────────────────────────────────────────
run_stage1rec() {
    local exp_name="$1"
    local gpu_id="$2"
    local status_file="$STATUS_DIR/.running_${exp_name}"

    touch "$status_file"

    local semantic_map="$STAGE1_BASE/$exp_name/semantic_id_mappings.json"
    local purified_content="$STAGE1_BASE/$exp_name/item_purified_content.npy"
    local purified_collab="$STAGE1_BASE/$exp_name/item_purified_collab.npy"
    local output_dir="$STAGE1_Rec/$exp_name"
    local log_file="$output_dir/stage1rec_training.log"

    # Dynamically detect purified_dim from experiment name
    #   stage1_ide_32   → 32    stage1_ide_256 → 256
    #   stage1_cma_0.1  → 128   (default for all non-ide experiments)
    local p_dim=128
    if [[ "$exp_name" == stage1_ide_* ]]; then
        p_dim="${exp_name#stage1_ide_}"
    fi

    mkdir -p "$output_dir"

    export CUDA_VISIBLE_DEVICES="$gpu_id"

    cd "$PROJECT_ROOT"
    if python -m src.recommender.prism.train \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --num_workers "$NUM_WORKERS" \
        --model_type "$MODEL_TYPE" \
        --output_dir "$output_dir" \
        --semantic_mapping_path "$semantic_map" \
        --purified_content_path "$purified_content" \
        --purified_collab_path "$purified_collab" \
        --purified_dim "$p_dim" \
        --use_multimodal_fusion \
        --fusion_gate_type dense \
        --use_purified_predictor \
        --purified_predictor_weight 0.1 \
        --use_item_layer_emb --use_temporal_decay \
        --use_trie_constraints \
        --use_adaptive_temperature \
        --tau_alpha 0.5 --tau_min 0.7 --tau_max 0.8 --tau_start_layer 1 \
        --lr_scheduler warmup_cosine \
        --eval_every_n_epochs 3 \
        > "$log_file" 2>&1; then

        rm -f "$status_file"
        touch "$STATUS_DIR/ok_${exp_name}"
    else
        rm -f "$status_file"
        touch "$STATUS_DIR/fail_${exp_name}"
        echo "[FAIL] $exp_name on GPU $gpu_id — see $log_file"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

echo ""
echo "========================================================================"
echo "  PRISM Stage 2 — HParam Sensitivity Runner"
echo "  Dataset:     $DATASET"
echo "  Stage 1:     $STAGE1_BASE"
echo "  Stage 2:     $STAGE1_Rec"
echo "  Max Jobs:    $MAX_CONCURRENT"
echo "  VRAM Thresh: $VRAM_THRESHOLD MB"
echo "========================================================================"

# Collect all valid experiments
EXPERIMENTS=()
for dir in "$STAGE1_BASE"/*/; do
    [ -d "$dir" ] || continue
    exp_name=$(basename "$dir")

    # Must have Stage 1 outputs
    if [ ! -f "$dir/semantic_id_mappings.json" ]; then
        continue
    fi

    # Skip if already completed
    if [ -f "$STATUS_DIR/ok_${exp_name}" ]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Skip if failed previously — user can delete the fail marker to retry
    if [ -f "$STATUS_DIR/fail_${exp_name}" ]; then
        continue
    fi

    EXPERIMENTS+=("$exp_name")
done

TOTAL=${#EXPERIMENTS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo ""
    echo "No pending experiments. All done!"
    exit 0
fi

echo "Pending experiments: $TOTAL"
echo ""

# ── Main dispatch loop ────────────────────────────────────────────────────
QUEUED=$TOTAL
for exp_name in "${EXPERIMENTS[@]}"; do
    # Wait if at max concurrency
    while [ "$(jobs -r | wc -l)" -ge "$MAX_CONCURRENT" ]; do
        sleep 10
        show_dashboard
    done

    # Find a free GPU with enough VRAM
    gpu=$(get_free_gpu)

    run_stage1rec "$exp_name" "$gpu" &
    QUEUED=$((QUEUED - 1))

    show_dashboard

    # Staggered startup
    if [ "$QUEUED" -gt 0 ]; then
        sleep "$STAGGER_DELAY"
    fi
done

# ── Wait for all jobs to finish ───────────────────────────────────────────
echo ""
echo "All jobs dispatched. Waiting for remaining tasks..."

while [ "$(jobs -r | wc -l)" -gt 0 ]; do
    sleep 15
    show_dashboard
done

# ── Final report ──────────────────────────────────────────────────────────
OK_COUNT=$(find "$STATUS_DIR" -name "ok_*" 2>/dev/null | wc -l)
FAIL_COUNT=$(find "$STATUS_DIR" -name "fail_*" 2>/dev/null | wc -l)

echo ""
echo "========================================================================"
echo "  STAGE 2 SWEEP COMPLETE"
echo "  Dataset: $DATASET"
echo "  Successful: $OK_COUNT  |  Failed: $FAIL_COUNT"
echo "  Output: $STAGE1_Rec"
echo "========================================================================"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    find "$STATUS_DIR" -name "fail_*" | while read f; do
        echo "  $(basename "$f" | sed 's/fail_//')"
    done
fi
