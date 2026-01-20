#!/bin/bash
# EAGER Two-Stream Generative Recommender Training Script
# Implements dual-stream architecture with behavior-semantic collaboration

cd ../..

echo "=================================================="
echo "Training EAGER Two-Stream Recommender"
echo "=================================================="
echo ""

# ============================================================
# Configuration
# ============================================================
CONFIG="toys"
DEVICE="cuda:1"
NUM_WORKERS=4
MODEL_TYPE="t5-tiny-2"

# Output directory keywords (optional, for custom naming)
# Example: "eager-baseline" will create dir like "2025-11-25-19-00-00_eager-baseline"
OUTPUT_KEYWORDS="eager-dual-stream"

# ============================================================
# EAGER Loss Weights (from paper Section 4.1)
# ============================================================
# Total Loss: L_total = L_gen + λ1*L_con + λ2*(L_recon + L_recog)
LAMBDA_1=1.0  # Weight for Global Contrastive Task (GCT)
LAMBDA_2=1.0  # Weight for Semantic-Guided Transfer Task (STT)

# ============================================================
# STT Masking Ratios (from paper Section 3.3)
# ============================================================
# For Reconstruction task: randomly mask 50% of behavior tokens
MASK_RATIO_RECON=0.5

# For Recognition task: randomly replace 50% of behavior tokens as negatives
MASK_RATIO_RECOG=0.5

# ============================================================
# Learning Rate Scheduler
# ============================================================
# Options: 'none', 'warmup_cosine', 'reduce_on_plateau', 'exponential', 'step'
LR_SCHEDULER="warmup_cosine"

# ============================================================
# Verbose Logging
# ============================================================
# Enable verbose sample printing during validation and testing
VERBOSE=false

# ============================================================
# Training Script
# ============================================================

echo "✅ Configuration: ${CONFIG}"
echo "   Device: ${DEVICE}"
echo "   Model type: ${MODEL_TYPE}"
echo "   Num workers: ${NUM_WORKERS}"
echo ""

echo "✅ EAGER Dual-Path Semantic IDs:"
echo "   Behavior mapping: ${BEHAVIOR_MAPPING_PATH}"
echo "   Semantic mapping: ${SEMANTIC_MAPPING_PATH}"
echo ""

echo "✅ EAGER Loss Weights:"
echo "   Lambda 1 (GCT): ${LAMBDA_1}"
echo "   Lambda 2 (STT): ${LAMBDA_2}"
echo ""

echo "✅ STT Masking Ratios:"
echo "   Reconstruction: ${MASK_RATIO_RECON}"
echo "   Recognition: ${MASK_RATIO_RECOG}"
echo ""

echo "✅ Verbose Logging: ${VERBOSE}"
echo "✅ LR Scheduler: ${LR_SCHEDULER}"
echo ""

echo "=================================================="

# Build command
CMD="python -m src.recommender.EAGER.train \
    --config ${CONFIG} \
    --device ${DEVICE} \
    --num_workers ${NUM_WORKERS} \
    --model_type ${MODEL_TYPE} \
    --output_keywords ${OUTPUT_KEYWORDS} \
    --lambda_1 ${LAMBDA_1} \
    --lambda_2 ${LAMBDA_2} \
    --mask_ratio_recon ${MASK_RATIO_RECON} \
    --mask_ratio_recog ${MASK_RATIO_RECOG}"

# Add learning rate scheduler
CMD="$CMD --lr_scheduler ${LR_SCHEDULER}"

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
echo "✓ EAGER Training completed!"
echo "=================================================="


