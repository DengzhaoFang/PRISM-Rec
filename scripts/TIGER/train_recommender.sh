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
DEVICE="cuda:0"
NUM_WORKERS=4
MODEL_TYPE="t5-tiny-2"

# Output directory keywords (optional, for custom naming)
OUTPUT_KEYWORDS="3layer-tiger"

# ============================================================
# Learning Rate Scheduler
# ============================================================
# Options: 'none', 'warmup_cosine', 'reduce_on_plateau', 'exponential', 'step'
LR_SCHEDULER="warmup_cosine"

VERBOSE=false




# Build command
CMD="python -m src.recommender.TIGER.train \
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


echo ""
echo "=================================================="
echo "✓ Training completed!"
echo "=================================================="
