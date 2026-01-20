#!/bin/bash

cd ../..

# Long-tail Comparison: TIGER vs PRISM vs ActionPiece
# 
# Compares models on Beauty and CDs datasets.
# Outputs a single publication-quality figure with side-by-side comparison.
#
# Usage:
#   bash scripts/prism/longtail_eval.sh

echo "=================================================="
echo "Long-tail Comparison: TIGER vs PRISM vs ActionPiece"
echo "=================================================="
echo ""

# ============================================================
# Configuration
# ============================================================

DEVICE="cuda:2"

# Beauty checkpoints
TIGER_CHECKPOINT_BEAUTY="scripts/output/recommender/tiger/beauty/2026-01-06-22-02-28_3layer-tiger/best_model.pt"
PRISM_CHECKPOINT_BEAUTY="scripts/output/recommender/prism/beauty/2026-01-06-21-58-26_3layer-prism/best_model.pt"
ACTIONPIECE_CHECKPOINT_BEAUTY="scripts/output/recommender/actionpiece/beauty/2026-01-10-23-31-13_actionpiece/best_model.pt"

# CDs checkpoints (set to empty string to skip CDs evaluation)
TIGER_CHECKPOINT_CDS="scripts/output/recommender/tiger/cds/2025-12-13-00-29-43_3layer-tiger/best_model.pt"
PRISM_CHECKPOINT_CDS="scripts/output/recommender/prism/cds/2025-12-29-01-59-37_3layer-prism-tile/best_model.pt"
ACTIONPIECE_CHECKPOINT_CDS="scripts/output/recommender/actionpiece/cds/2026-01-09-20-26-12_actionpiece-large/best_model.pt"

# Output directory
OUTPUT_DIR="scripts/output/longtail_comparison_new"

# Number of popularity groups (3 recommended: Popular, Medium, Long-tail)
NUM_GROUPS=3

# Beam size for generation
BEAM_SIZE=30

# Metrics to display
METRICS="Recall@10 NDCG@10"

# Quick test mode: limit samples per group (set to empty for full evaluation)
MAX_SAMPLES_PER_GROUP=

# Plot-only mode: skip evaluation and only generate plots from existing JSON files
# Set to "true" to enable, empty to disable
PLOT_ONLY="true"

# ActionPiece-only evaluation mode: only evaluate ActionPiece, load TIGER/PRISM from existing JSON
# Set to "true" to enable, empty to disable
# This is useful when you already have TIGER/PRISM results and only need to add ActionPiece
EVAL_ACTIONPIECE_ONLY=""

# ============================================================
# Validation
# ============================================================

if [ "$PLOT_ONLY" = "true" ]; then
    echo "📊 Plot-only mode: Will generate plots from existing JSON files"
    echo "   Output directory: ${OUTPUT_DIR}"
elif [ "$EVAL_ACTIONPIECE_ONLY" = "true" ]; then
    echo "🔧 ActionPiece-only evaluation mode: Will evaluate ActionPiece and load TIGER/PRISM from JSON"
    echo "   Output directory: ${OUTPUT_DIR}"
    
    if [ ! -f "$ACTIONPIECE_CHECKPOINT_BEAUTY" ]; then
        echo "ERROR: ActionPiece Beauty checkpoint not found: ${ACTIONPIECE_CHECKPOINT_BEAUTY}"
        exit 1
    fi
    
    echo "✅ ActionPiece Beauty checkpoint:"
    echo "   ${ACTIONPIECE_CHECKPOINT_BEAUTY}"
    
    if [ -n "$ACTIONPIECE_CHECKPOINT_CDS" ] && [ -f "$ACTIONPIECE_CHECKPOINT_CDS" ]; then
        echo "✅ ActionPiece CDs checkpoint:"
        echo "   ${ACTIONPIECE_CHECKPOINT_CDS}"
    fi
else
    if [ ! -f "$TIGER_CHECKPOINT_BEAUTY" ]; then
        echo "ERROR: TIGER Beauty checkpoint not found: ${TIGER_CHECKPOINT_BEAUTY}"
        exit 1
    fi

    if [ ! -f "$PRISM_CHECKPOINT_BEAUTY" ]; then
        echo "ERROR: PRISM Beauty checkpoint not found: ${PRISM_CHECKPOINT_BEAUTY}"
        exit 1
    fi

    if [ ! -f "$ACTIONPIECE_CHECKPOINT_BEAUTY" ]; then
        echo "ERROR: ActionPiece Beauty checkpoint not found: ${ACTIONPIECE_CHECKPOINT_BEAUTY}"
        exit 1
    fi

    echo "✅ Beauty checkpoints:"
    echo "   TIGER: ${TIGER_CHECKPOINT_BEAUTY}"
    echo "   PRISM: ${PRISM_CHECKPOINT_BEAUTY}"
    echo "   ActionPiece: ${ACTIONPIECE_CHECKPOINT_BEAUTY}"
fi

# Build command
if [ "$PLOT_ONLY" = "true" ]; then
    CMD="PYTHONPATH=\"${PWD}:\${PYTHONPATH}\" python scripts/prism/compare_longtail.py \
        --plot_only \
        --output_dir \"${OUTPUT_DIR}\" \
        --metrics ${METRICS}"
elif [ "$EVAL_ACTIONPIECE_ONLY" = "true" ]; then
    CMD="PYTHONPATH=\"${PWD}:\${PYTHONPATH}\" python scripts/prism/compare_longtail.py \
        --eval_actionpiece_only \
        --actionpiece_checkpoint_beauty \"${ACTIONPIECE_CHECKPOINT_BEAUTY}\" \
        --output_dir \"${OUTPUT_DIR}\" \
        --device \"${DEVICE}\" \
        --num_groups ${NUM_GROUPS} \
        --beam_size ${BEAM_SIZE} \
        --metrics ${METRICS}"
    
    # Add max samples per group if set
    if [ -n "$MAX_SAMPLES_PER_GROUP" ]; then
        CMD="$CMD --max_samples_per_group ${MAX_SAMPLES_PER_GROUP}"
        echo "   Quick test mode: ${MAX_SAMPLES_PER_GROUP} samples per group"
    fi
    
    # Add CDs ActionPiece checkpoint if exists
    if [ -n "$ACTIONPIECE_CHECKPOINT_CDS" ] && [ -f "$ACTIONPIECE_CHECKPOINT_CDS" ]; then
        CMD="$CMD --actionpiece_checkpoint_cds \"${ACTIONPIECE_CHECKPOINT_CDS}\""
    fi
else
    CMD="PYTHONPATH=\"${PWD}:\${PYTHONPATH}\" python scripts/prism/compare_longtail.py \
        --tiger_checkpoint_beauty \"${TIGER_CHECKPOINT_BEAUTY}\" \
        --prism_checkpoint_beauty \"${PRISM_CHECKPOINT_BEAUTY}\" \
        --actionpiece_checkpoint_beauty \"${ACTIONPIECE_CHECKPOINT_BEAUTY}\" \
        --output_dir \"${OUTPUT_DIR}\" \
        --device \"${DEVICE}\" \
        --num_groups ${NUM_GROUPS} \
        --beam_size ${BEAM_SIZE} \
        --metrics ${METRICS}"
    
    # Add max samples per group if set
    if [ -n "$MAX_SAMPLES_PER_GROUP" ]; then
        CMD="$CMD --max_samples_per_group ${MAX_SAMPLES_PER_GROUP}"
        echo "   Quick test mode: ${MAX_SAMPLES_PER_GROUP} samples per group"
    fi
    
    # Add CDs if checkpoints exist
    if [ -n "$TIGER_CHECKPOINT_CDS" ] && [ -n "$PRISM_CHECKPOINT_CDS" ] && [ -n "$ACTIONPIECE_CHECKPOINT_CDS" ]; then
        if [ -f "$TIGER_CHECKPOINT_CDS" ] && [ -f "$PRISM_CHECKPOINT_CDS" ] && [ -f "$ACTIONPIECE_CHECKPOINT_CDS" ]; then
            echo ""
            echo "✅ CDs checkpoints:"
            echo "   TIGER: ${TIGER_CHECKPOINT_CDS}"
            echo "   PRISM: ${PRISM_CHECKPOINT_CDS}"
            echo "   ActionPiece: ${ACTIONPIECE_CHECKPOINT_CDS}"
            CMD="$CMD --tiger_checkpoint_cds \"${TIGER_CHECKPOINT_CDS}\" --prism_checkpoint_cds \"${PRISM_CHECKPOINT_CDS}\" --actionpiece_checkpoint_cds \"${ACTIONPIECE_CHECKPOINT_CDS}\""
        else
            echo ""
            echo "⚠️  CDs checkpoints not found, skipping CDs evaluation"
        fi
    fi
fi

echo ""
if [ "$PLOT_ONLY" != "true" ]; then
    echo "   Device: ${DEVICE}"
    echo "   Num groups: ${NUM_GROUPS}"
    echo "   Beam size: ${BEAM_SIZE}"
fi
echo "   Output: ${OUTPUT_DIR}"
echo ""
echo "=================================================="

# Run
eval $CMD

echo ""
echo "=================================================="
echo "✓ Long-tail comparison completed!"
echo "=================================================="
echo ""
echo "Output files:"
echo "  - ${OUTPUT_DIR}/longtail_comparison.pdf (for LaTeX)"
echo "  - ${OUTPUT_DIR}/longtail_comparison.png (for preview)"
echo "  - ${OUTPUT_DIR}/*_results_*.json (raw metrics)"
