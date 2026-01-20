#!/bin/bash

# ActionPiece Recommender Training Script
# 
# - Dynamic SPR (Set Permutation Regularization) augmentation during training
# - Inference-time ensemble with multiple SPR augmentations
# - nDCG-based score aggregation
#
# Paper settings:
# - T5 architecture: 4 layers, 6 heads, d_model=128, d_ff=1024
# - Batch size: 256
# - Learning rate: 0.001 (Beauty), 0.005 (Sports)
# - Warmup steps: 10000
# - Beam size: 30 (aligned with our framework, paper uses 50)
# - Inference ensemble: q=5
#
# Resume training:
# - Set RESUME_CHECKPOINT to the path of a checkpoint file
# - All hyperparameters will be automatically loaded from the checkpoint
# - Only device and num_workers can be overridden

cd ../..

echo "=================================================="
echo "Training ActionPiece Recommender"
echo "=================================================="
echo ""

# ============================================================
# Configuration
# ============================================================
DATASET="beauty"  # Options: beauty, sports, toys, cds
DEVICE="cuda:1"
NUM_WORKERS=4
MODEL_TYPE="t5-tiny-2" # actionpiece-paper

# ActionPiece specific settings
N_ENSEMBLE=5          # q=5 in paper
TRAIN_SHUFFLE="feature"  # SPR augmentation
BEAM_SIZE=30          # Aligned with our framework (paper uses 50)

# Output directory keywords
OUTPUT_KEYWORDS="actionpiece"

# ============================================================
# Resume Training Configuration (optional)
# ============================================================
# Set this to resume from a checkpoint, leave empty for fresh training
# When resuming, all hyperparameters will be loaded from the checkpoint
# Only device and num_workers can be overridden
RESUME_CHECKPOINT=""

# ============================================================
# Path Configuration
# ============================================================
case $DATASET in
    "beauty")
        SEQ_DATA_PATH="dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
        TOKENIZER_PATH="scripts/output/actionpiece_tokenizer/beauty/actionpiece.json"
        ITEM2FEAT_PATH="scripts/output/actionpiece_tokenizer/beauty/item2feat.json"
        LEARNING_RATE=1e-3
        ;;
    "sports")
        SEQ_DATA_PATH="dataset/Amazon-Sports/processed/sports-tiger-sentenceT5base/Sports"
        TOKENIZER_PATH="scripts/output/actionpiece_tokenizer/sports/actionpiece.json"
        ITEM2FEAT_PATH="scripts/output/actionpiece_tokenizer/sports/item2feat.json"
        LEARNING_RATE=5e-3
        ;;
    "toys")
        SEQ_DATA_PATH="dataset/Amazon-Toys/processed/toys-tiger-sentenceT5base/Toys"
        TOKENIZER_PATH="scripts/output/actionpiece_tokenizer/toys/actionpiece.json"
        ITEM2FEAT_PATH="scripts/output/actionpiece_tokenizer/toys/item2feat.json"
        LEARNING_RATE=1e-3
        ;;
    "cds")
        SEQ_DATA_PATH="dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs"
        TOKENIZER_PATH="scripts/output/actionpiece_tokenizer/cds/actionpiece.json"
        ITEM2FEAT_PATH="scripts/output/actionpiece_tokenizer/cds/item2feat.json"
        LEARNING_RATE=1e-3
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

# ============================================================
# Training
# ============================================================

echo "✅ Dataset: ${DATASET}"
echo "   Sequence data: ${SEQ_DATA_PATH}"
echo "   Tokenizer: ${TOKENIZER_PATH}"
echo "   Item2feat: ${ITEM2FEAT_PATH}"
echo ""
echo "✅ Model: ${MODEL_TYPE}"
echo "   Device: ${DEVICE}"
echo ""
echo "✅ ActionPiece Settings:"
echo "   Train shuffle: ${TRAIN_SHUFFLE} (SPR augmentation)"
echo "   Inference ensemble: ${N_ENSEMBLE}"
echo "   Beam size: ${BEAM_SIZE}"
echo "   Learning rate: ${LEARNING_RATE}"
echo ""

# Check if resuming from checkpoint
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    echo "🔄 RESUMING FROM CHECKPOINT"
    echo "   Checkpoint: ${RESUME_CHECKPOINT}"
    echo "   All hyperparameters will be loaded from checkpoint"
    echo "   Only device and num_workers can be overridden"
    echo ""
    echo "=================================================="
    
    # When resuming, only pass essential runtime parameters
    python -m src.recommender.ActionPiece.actionpiece_train \
        --resume ${RESUME_CHECKPOINT} \
        --device ${DEVICE} \
        --num_workers ${NUM_WORKERS}
else
    echo "✅ Fresh Training (no checkpoint)"
    echo ""
    echo "=================================================="
    
    # Fresh training with all parameters
    python -m src.recommender.ActionPiece.actionpiece_train \
        --config ${DATASET} \
        --device ${DEVICE} \
        --num_workers ${NUM_WORKERS} \
        --model_type ${MODEL_TYPE} \
        --n_ensemble ${N_ENSEMBLE} \
        --train_shuffle ${TRAIN_SHUFFLE} \
        --beam_size ${BEAM_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --sequence_data_path ${SEQ_DATA_PATH} \
        --tokenizer_path ${TOKENIZER_PATH} \
        --item2feat_path ${ITEM2FEAT_PATH} \
        --output_keywords ${OUTPUT_KEYWORDS}
fi

echo ""
echo "=================================================="
echo "✓ ActionPiece recommender training completed!"
echo "=================================================="
