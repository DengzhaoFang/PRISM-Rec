#!/bin/bash

cd ../..

# Train TIGER Recommender with configurable options
# 
# This script provides a more structured way to configure training parameters
# compared to the simple train_recommender.sh script.

echo "=================================================="
echo "Training TIGER Recommender"
echo "=================================================="
echo ""

# ============================================================
# Configuration
# ============================================================
CONFIG="beauty"
DEVICE="cuda:1"
NUM_WORKERS=4
MODEL_TYPE="t5-small"

# Output directory keywords (optional, for custom naming)
# Example: "baseline-experiment" will create dir like "2025-11-06-17-26-56_baseline-experiment"
OUTPUT_KEYWORDS="3layer-tiger-beamsize-small"

# ============================================================
# Learning Rate Scheduler
# ============================================================
# Options: 'none', 'warmup_cosine', 'reduce_on_plateau', 'exponential', 'step'
#   - 'none': Disable dynamic learning rate (use fixed LR)
#   - 'warmup_cosine': Warmup + cosine annealing (recommended)
#   - 'reduce_on_plateau': Reduce LR when validation metric plateaus
#   - 'exponential': Exponential decay
#   - 'step': Step decay (reduce every N epochs)
# Set to empty string ("") to use default from config.py
LR_SCHEDULER="warmup_cosine"

# ============================================================
# Verbose Logging
# ============================================================
# Enable verbose sample printing during validation and testing
# When enabled, randomly samples 10 examples per eval and prints:
#   - Input item IDs
#   - Predicted semantic IDs
#   - Ground truth semantic IDs
VERBOSE=false

# ============================================================
# Training Script
# ============================================================

echo "✅ Configuration: ${CONFIG}"
echo "   Device: ${DEVICE}"
echo "   Model type: ${MODEL_TYPE}"
echo "   Num workers: ${NUM_WORKERS}"
echo ""

echo "✅ Verbose Logging: ${VERBOSE}"
echo "✅ LR Scheduler: ${LR_SCHEDULER:-default from config}"
echo ""

echo "=================================================="

# Build command
CMD="python -m src.recommender.tiger.train \
    --config ${CONFIG} \
    --device ${DEVICE} \
    --num_workers ${NUM_WORKERS} \
    --model_type ${MODEL_TYPE} \
    --output_keywords ${OUTPUT_KEYWORDS}"

# Add learning rate scheduler if specified
if [ -n "$LR_SCHEDULER" ]; then
    CMD="$CMD --lr_scheduler ${LR_SCHEDULER}"
fi

# Add verbose flag if enabled
if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

eval $CMD

# ============================================================
# Additional Options
# ============================================================
# To resume from checkpoint, add:
#   --resume "path/to/checkpoint.pt"
#
# To use a custom output directory:
#   --output_dir "path/to/output"

echo ""
echo "=================================================="
echo "✓ Training completed!"
echo "=================================================="


