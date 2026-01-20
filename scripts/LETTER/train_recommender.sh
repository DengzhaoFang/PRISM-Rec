#!/bin/bash

cd ../..


echo "=================================================="
echo "Training LETTER Recommender"
echo "=================================================="
echo ""

# ============================================================
# Configuration
# ============================================================
CONFIG="sports"
DEVICE="cuda:1"
NUM_WORKERS=4
MODEL_TYPE="t5-tiny-2"

# Output directory keywords (optional, for custom naming)
OUTPUT_KEYWORDS="3layer-letter"

# ============================================================
# Learning Rate Scheduler
# ============================================================
# Options: 'none', 'warmup_cosine', 'reduce_on_plateau', 'exponential', 'step'

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

# Ranking-guided Generation Loss Temperature
# Recommended: 0.8 - 1.0
TEMPERATURE=0.8

# ============================================================
# Training Script
# ============================================================

echo "✅ Configuration: ${CONFIG}"
echo "   Device: ${DEVICE}"
echo "   Model type: ${MODEL_TYPE}"
echo "   Num workers: ${NUM_WORKERS}"
echo "   Temperature: ${TEMPERATURE}"
echo ""

echo "✅ Verbose Logging: ${VERBOSE}"
echo "✅ LR Scheduler: ${LR_SCHEDULER:-default from config}"
echo ""

echo "=================================================="

# Build command
CMD="python -m src.recommender.LETTER.train \
    --config ${CONFIG} \
    --device ${DEVICE} \
    --num_workers ${NUM_WORKERS} \
    --model_type ${MODEL_TYPE} \
    --output_keywords ${OUTPUT_KEYWORDS} \
    --temperature ${TEMPERATURE}"

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


