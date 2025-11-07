#!/bin/bash

cd ../..

# Train HiD-VAE Recommender with optional pretrained embedding initialization
# 
# This script supports two modes:
# 1. Without pretrained embeddings (random initialization)
# 2. With HiD-VAE pretrained embeddings (recommended)
#
# See EMBEDDING_INIT_GUIDE.md for detailed usage instructions

echo "=================================================="
echo "Training HiD-VAE Recommender"
echo "=================================================="
echo ""

# ============================================================
# Configuration
# ============================================================
CONFIG="beauty"
DEVICE="cuda:2"
NUM_WORKERS=4
MODEL_TYPE="t5-small"

# Output directory keywords (optional, for custom naming)
# Example: "pretrained-codebook" will create dir like "2025-11-04-17-26-56_pretrained-codebook"
OUTPUT_KEYWORDS="3layer-t5-small-semantic-space-alignment"

# ============================================================
# Embedding Initialization (Optional)
# ============================================================
# Set to true to enable HiD-VAE pretrained embedding initialization
# Note: embedding path is configured in config.py for each dataset
USE_PRETRAINED_EMBEDDINGS=false

# Embedding type: "codebook" (z_q_total, latent_dim=32) or "collaborative" (z_q_proj, collab_dim=64)
# - codebook: Better semantic granularity, smaller dimension (recommended)
# - collaborative: Better collaborative filtering signal, larger dimension
EMBEDDING_INIT_TYPE="codebook"   # "codebook" or "collaborative"

# Whether to freeze embeddings after initialization
# - false: Embeddings can be fine-tuned (recommended)
# - true: Embeddings are frozen (preserve HiD-VAE semantics)
FREEZE_EMBEDDINGS=false

# ============================================================
# Multi-Task Learning (Collaborative Filtering + Tag Prediction)
# ============================================================
# Enable multi-task learning to jointly optimize:
#   1. Semantic ID generation (main task)
#   2. Collaborative embedding prediction (auxiliary task, preserves CF signal)
#   3. Hierarchical tag prediction (auxiliary task, improves semantic granularity)
#
# IMPORTANT: Multi-task learning requires item_embeddings.json to load auxiliary data
#            (collaborative embeddings and tag labels). You must set EMBEDDING_INIT_PATH
#            below even if you don't want to use pretrained embeddings for initialization.

# Enable collaborative filtering loss (recommended for better CF signal retention)
ENABLE_COLLAB_LOSS=false
COLLAB_LOSS_WEIGHT=0.1  # Recommended: 0.05-0.2

# Enable tag prediction loss (recommended for better semantic alignment)
ENABLE_TAG_LOSS=false
TAG_LOSS_WEIGHT=0.05  # Recommended: 0.01-0.1

# Path to item_embeddings.json (required for multi-task learning)
# This is used to load auxiliary data even if USE_PRETRAINED_EMBEDDINGS=false
EMBEDDING_INIT_PATH="scripts/output/hidvae/Beauty/3-256-32-hidvae-2000epoch-alignment/item_embeddings.json"

# ============================================================
# Verbose Logging
# ============================================================
# Enable verbose sample printing during validation and testing
# When enabled, randomly samples 10 examples per eval and prints:
#   - Input item IDs
#   - Predicted semantic IDs
#   - Ground truth semantic IDs
#   - Tag predictions (if multi-task learning is enabled)
VERBOSE=false

# ============================================================
# Learning Rate Scheduler (Adaptive Learning)
# ============================================================
# Scheduler type for adaptive learning rate (improved semantic ID fitting)
# Options:
#   - "warmup_cosine": Warmup + Cosine Annealing (RECOMMENDED, smooth decay)
#   - "warmup_linear": Warmup + Linear Decay (more aggressive)
#   - "plateau": Adaptive reduction based on validation metrics
#   - "onecycle": OneCycleLR (cyclic learning rate)
#   - "cosine_restarts": Cosine with periodic restarts
#   - "warmup_constant": Warmup only (constant after warmup)
#   - "none": No scheduler (constant learning rate)
LR_SCHEDULER_TYPE="warmup_cosine"

# Warmup steps (number of training steps for warmup phase)
# - Recommended: 1000-2000 for better training stability
WARMUP_STEPS=1000

# Minimum learning rate ratio (for warmup_cosine/warmup_linear)
# - Final LR will be base_lr * min_lr_ratio
# - Recommended: 0.1 (i.e., decay to 10% of initial LR)
MIN_LR_RATIO=0.1

# Plateau scheduler parameters (only used if LR_SCHEDULER_TYPE="plateau")
# - patience: Number of epochs to wait before reducing LR
# - factor: LR reduction factor (new_lr = old_lr * factor)
PLATEAU_PATIENCE=5
PLATEAU_FACTOR=0.5

# ============================================================
# Training Script
# ============================================================

echo "✅ Verbose Logging: ${VERBOSE}"
echo ""

echo "✅ Learning Rate Scheduler: ${LR_SCHEDULER_TYPE}"
echo "   Warmup steps: ${WARMUP_STEPS}"
echo "   Min LR ratio: ${MIN_LR_RATIO}"
echo ""

echo "✅ Multi-Task Learning:"
if [ "$ENABLE_COLLAB_LOSS" = true ]; then
    echo "   Collaborative loss: ENABLED (weight=${COLLAB_LOSS_WEIGHT})"
else
    echo "   Collaborative loss: DISABLED"
fi
if [ "$ENABLE_TAG_LOSS" = true ]; then
    echo "   Tag prediction loss: ENABLED (weight=${TAG_LOSS_WEIGHT})"
else
    echo "   Tag prediction loss: DISABLED"
fi
echo ""

if [ "$USE_PRETRAINED_EMBEDDINGS" = true ]; then
    echo "✅ Embedding initialization: ENABLED"
    echo "   Type: ${EMBEDDING_INIT_TYPE}"
    echo "   Path: (using default from config.py)"
    echo "   Freeze: ${FREEZE_EMBEDDINGS}"
    echo ""
    echo "=================================================="
    
    # Build command with conditional flags
    CMD="python -m src.recommender.hidvae.train \
        --config ${CONFIG} \
        --device ${DEVICE} \
        --num_workers ${NUM_WORKERS} \
        --model_type ${MODEL_TYPE} \
        --output_keywords ${OUTPUT_KEYWORDS} \
        --use_pretrained_embeddings \
        --embedding_init_type ${EMBEDDING_INIT_TYPE} \
        --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
        --warmup_steps ${WARMUP_STEPS} \
        --min_lr_ratio ${MIN_LR_RATIO} \
        --plateau_patience ${PLATEAU_PATIENCE} \
        --plateau_factor ${PLATEAU_FACTOR}"
    
    # Add freeze flag if explicitly set to true
    if [ "$FREEZE_EMBEDDINGS" = true ]; then
        CMD="$CMD --freeze_embeddings"
    fi
    
    # Add multi-task learning flags
    if [ "$ENABLE_COLLAB_LOSS" = true ]; then
        CMD="$CMD --enable_collab_loss --collab_loss_weight ${COLLAB_LOSS_WEIGHT}"
    fi
    if [ "$ENABLE_TAG_LOSS" = true ]; then
        CMD="$CMD --enable_tag_loss --tag_loss_weight ${TAG_LOSS_WEIGHT}"
    fi
    
    # Add verbose flag if enabled
    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi
    
    eval $CMD
else
    echo "✅ Embedding initialization: DISABLED (random initialization)"
    echo ""
    echo "=================================================="
    
    # Build command for non-pretrained case
    CMD="python -m src.recommender.hidvae.train \
        --config ${CONFIG} \
        --device ${DEVICE} \
        --num_workers ${NUM_WORKERS} \
        --model_type ${MODEL_TYPE} \
        --output_keywords ${OUTPUT_KEYWORDS} \
        --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
        --warmup_steps ${WARMUP_STEPS} \
        --min_lr_ratio ${MIN_LR_RATIO} \
        --plateau_patience ${PLATEAU_PATIENCE} \
        --plateau_factor ${PLATEAU_FACTOR}"
    
    # Add multi-task learning flags
    # Note: Multi-task learning requires embedding_init_path for auxiliary data
    if [ "$ENABLE_COLLAB_LOSS" = true ] || [ "$ENABLE_TAG_LOSS" = true ]; then
        if [ -n "$EMBEDDING_INIT_PATH" ]; then
            CMD="$CMD --embedding_init_path ${EMBEDDING_INIT_PATH}"
        fi
    fi
    
    if [ "$ENABLE_COLLAB_LOSS" = true ]; then
        CMD="$CMD --enable_collab_loss --collab_loss_weight ${COLLAB_LOSS_WEIGHT}"
    fi
    if [ "$ENABLE_TAG_LOSS" = true ]; then
        CMD="$CMD --enable_tag_loss --tag_loss_weight ${TAG_LOSS_WEIGHT}"
    fi
    
    # Add verbose flag if enabled
    if [ "$VERBOSE" = true ]; then
        CMD="$CMD --verbose"
    fi
    
    eval $CMD
fi

# ============================================================
# Additional Options
# ============================================================
# To resume from checkpoint, add:
#   --resume "path/to/checkpoint.pt"
#
# To use a custom embedding path (override config default):
#   --embedding_init_path "path/to/custom/item_embeddings.json"
#
# Embedding paths are configured in config.py:
#   - Beauty: scripts/output/hidvae/Beauty/3-256-32-hidvae-2000epoch-modify-disentanglement/item_embeddings.json
#   - Sports: (add your path in config.py)
#   - Toys: (add your path in config.py)

echo ""
echo "=================================================="
echo "✓ Training completed!"
echo "=================================================="

