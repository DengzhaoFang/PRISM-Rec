#!/bin/bash
# Train HiD-VAE with FIXED DEPTH and ADAPTIVE LOSS WEIGHTING
# This script demonstrates the new fixed-depth approach with adaptive weighting
# that automatically balances loss scales without manual tuning.

cd ../../src/sid_tokenizer/hidvae

# Dataset configuration
DATASET="Beauty"
DATA_DIR="../../../dataset/Amazon-Beauty/processed/beauty-hidvae-sentenceT5base/${DATASET}"
OUTPUT_DIR="../../../scripts/output/hidvae/${DATASET}/3-256-32-hidvae-2000epoch-semantic-space-alignment"

# Embedding file paths
TEXT_EMB_FILE="item_emb.parquet"
COLLAB_EMB_FILE="lightgcn/item_embeddings_collab.npy"

# ============================================================
# COLLABORATIVE KNOWLEDGE FUSION MODE (NEW FEATURE)
# ============================================================
# - alignment: Use collaborative embeddings for alignment loss (original method)
# - concat: PCA + concatenate collaborative embeddings with semantic embeddings as input
COLLAB_MODE="alignment"  # Options: alignment, concat
PCA_DIM="64"  # PCA dimension for concat mode (auto-set to collab_dim if empty)

# Model hyperparameters
LATENT_DIM=32 # the dimension of the latent space
N_EMBED=256 # the size of the codebook
NUM_LAYERS=3 # number of quantization layers

# ============================================================
# ADAPTIVE LOSS WEIGHTING (NEW FEATURE)
# ============================================================
# When enabled, loss weights are automatically adjusted based on
# their running scales, so you don't need to manually tune them!
USE_ADAPTIVE_WEIGHTS=true
ADAPTIVE_ALPHA=0.1  # EMA coefficient (0.05-0.2 recommended)

# Loss weights (now representing RELATIVE IMPORTANCE, not absolute scale)
# With adaptive weighting, these represent how important each loss is
# relative to others, not their actual magnitudes
RECON_WEIGHT=1.0           # Reconstruction is critical
COMMIT_WEIGHT=1.0          # Commitment is critical  
COLLAB_WEIGHT=1.0          # Collaborative alignment equally important
LABEL_PRED_WEIGHT=0.5      # Label prediction moderately important (with padding-aware loss)
LABEL_ALIGN_WEIGHT=1.0     # CRITICAL: Now uses decoder for projection - needs strong weight!
DISENTANGLE_WEIGHT=0.5     # Disentanglement least important

# Training hyperparameters
EPOCHS=2000
BATCH_SIZE=256
LR=1e-3
WEIGHT_DECAY=0.01
GRAD_CLIP=1.0

# Quantization settings
USE_EMA=true
EMA_DECAY=0.999  # CRITICAL: Very slow decay for maximum codebook diversity
BETA=0.25
NORMALIZE_RESIDUAL=""  # CRITICAL FIX: Disable residual normalization to preserve codebook diversity (set to "true" to enable)

# Learning rate scheduler
USE_SCHEDULER=true
SCHEDULER_TYPE="warmup_cosine"
WARMUP_RATIO=0.1

# Early stopping (triggers global uniqueness checking)
EARLY_STOP_PATIENCE=50  
EARLY_STOP_MIN_DELTA=1e-5  

# Evaluation and saving
EVAL_EVERY=5
SAVE_EVERY=50

# System
DEVICE="cuda:0"
NUM_WORKERS=4
SEED=42
LOG_LEVEL="INFO"

# Optional: For debugging
MAX_ITEMS=""

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=================================================="
echo "Training HiD-VAE with TRAINING-TIME UNIQUENESS ENFORCEMENT"
echo "=================================================="
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Model Configuration:"
echo "  Num Layers: ${NUM_LAYERS} (fixed depth)"
echo "  Codebook Size: ${N_EMBED}"
echo "  Latent Dim: ${LATENT_DIM}"
echo ""
echo "🆕 UNIQUENESS STRATEGY (3-PHASE APPROACH):"
echo "  Phase 1: Normal Training"
echo "    • Batch-level collision detection (fast, immediate feedback)"
echo "    • No global checks (zero overhead)"
echo "    • Continues until early-stop condition"
echo ""
echo "  Phase 2: Global Uniqueness Check (triggered by early-stop)"
echo "    • Check full dataset uniqueness"
echo "    • If 100% unique → STOP immediately"
echo "    • If <100% unique → Enter Phase 3"
echo ""
echo "  Phase 3: Global Enforcement (only if needed)"
echo "    • Apply EXTREME penalties to collision pairs"
echo "    • Continue training until 100% uniqueness"
echo "    • Auto-stop when achieved"
echo ""
echo "  Early Stop Patience: ${EARLY_STOP_PATIENCE} epochs"
echo ""
echo "Collaborative Knowledge Fusion:"
echo "  Mode: ${COLLAB_MODE}"
if [ "${COLLAB_MODE}" = "concat" ]; then
    if [ -n "${PCA_DIM}" ]; then
        echo "  PCA Dimension: ${PCA_DIM}"
    else
        echo "  PCA Dimension: auto (match collab_dim)"
    fi
fi
echo ""
echo "Adaptive Weighting:"
echo "  Enabled: ${USE_ADAPTIVE_WEIGHTS}"
echo "  Alpha: ${ADAPTIVE_ALPHA}"
echo ""
echo "Relative Loss Importances:"
echo "  Reconstruction:    ${RECON_WEIGHT}"
if [ "${COLLAB_MODE}" = "alignment" ]; then
    echo "  Collaborative:     ${COLLAB_WEIGHT} (alignment loss)"
else
    echo "  Collaborative:     ${COLLAB_WEIGHT} (auto-disabled in concat mode)"
fi
echo "  Commitment:        ${COMMIT_WEIGHT}"
echo "  Label Prediction:  ${LABEL_PRED_WEIGHT} (padding-aware)"
echo "  Label Alignment:   ${LABEL_ALIGN_WEIGHT}"
echo "  Disentanglement:   ${DISENTANGLE_WEIGHT} (includes online collision detection)"
echo "=================================================="

# Run training
python train_hidvae.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --stage 0 \
    --embedding_file ${TEXT_EMB_FILE} \
    --collab_emb_file ${COLLAB_EMB_FILE} \
    --collab_mode ${COLLAB_MODE} \
    ${PCA_DIM:+--pca_dim ${PCA_DIM}} \
    --latent_dim ${LATENT_DIM} \
    --n_embed ${N_EMBED} \
    --num_layers ${NUM_LAYERS} \
    --recon_weight ${RECON_WEIGHT} \
    --commit_weight ${COMMIT_WEIGHT} \
    --collab_weight ${COLLAB_WEIGHT} \
    --label_pred_weight ${LABEL_PRED_WEIGHT} \
    --label_align_weight ${LABEL_ALIGN_WEIGHT} \
    --disentangle_weight ${DISENTANGLE_WEIGHT} \
    --use_adaptive_weights \
    --adaptive_alpha ${ADAPTIVE_ALPHA} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --grad_clip ${GRAD_CLIP} \
    --use_ema \
    --ema_decay ${EMA_DECAY} \
    --beta ${BETA} \
    ${NORMALIZE_RESIDUAL:+--normalize_residual} \
    --use_scheduler \
    --scheduler_type ${SCHEDULER_TYPE} \
    --warmup_ratio ${WARMUP_RATIO} \
    --early_stop_patience ${EARLY_STOP_PATIENCE} \
    --early_stop_min_delta ${EARLY_STOP_MIN_DELTA} \
    --eval_every ${EVAL_EVERY} \
    --save_every ${SAVE_EVERY} \
    --device ${DEVICE} \
    --num_workers ${NUM_WORKERS} \
    --seed ${SEED} \
    --log_level ${LOG_LEVEL} \
    ${MAX_ITEMS:+--max_items ${MAX_ITEMS}}

echo ""
echo "=================================================="
echo "✓ Training completed!"
echo "=================================================="
echo "Check results in: ${OUTPUT_DIR}"
echo ""




