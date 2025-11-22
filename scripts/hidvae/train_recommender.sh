#!/bin/bash

cd ../..


echo "=================================================="
echo "Training HIDVAE Recommender"
echo "=================================================="
echo ""

# ============================================================
# Configuration
# ============================================================
CONFIG="beauty"
DEVICE="cuda:2"
NUM_WORKERS=4
MODEL_TYPE="t5-tiny-2"

# Output directory keywords (optional, for custom naming)
# Example: "baseline-experiment" will create dir like "2025-11-06-17-26-56_baseline-experiment"
OUTPUT_KEYWORDS="fixall-vs-v9-trie"

# ============================================================
# Feature: Trie-Constrained Decoding
# ============================================================
# Ensures every decoding step points to a path that can lead to a real item
# This eliminates invalid predictions and improves accuracy
USE_TRIE_CONSTRAINTS=true

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

# ============================================================
# NEW FEATURES: Multi-source Information Fusion
# ============================================================

# Feature 1: Codebook Vector Warm-start
USE_CODEBOOK_WARMSTART=true
CODEBOOK_WARMSTART_FREEZE=false  # Whether to freeze warmstarted embeddings

# Feature 2: Codebook Vector Prediction
USE_CODEBOOK_PREDICTION=true  # Start with false, test incrementally
CODEBOOK_PREDICTION_WEIGHT=0.0005  # 0.001

# Feature 3: Tag ID Prediction
USE_TAG_PREDICTION=true  
TAG_PREDICTION_WEIGHT=0.0005  # 0.001
PREDICT_TAGS_FIRST=true

# Feature 4: Multi-source Embedding Fusion
USE_MULTIMODAL_FUSION=true
FUSION_GATE_TYPE="learned"  # Options: learned, fixed, attention
USE_LAYER_SPECIFIC_FUSION=true  # Use layer-specific projections (recommended for better performance)
# IMPORTANT: For fixed fusion, use conservative weights (ID should dominate)
CONTENT_EMB_WEIGHT=0.3      # For fixed fusion 
COLLAB_EMB_WEIGHT=0.3       # For fixed fusion 
ID_EMB_WEIGHT=0.4           # For fixed fusion


# Collaborative embedding path (optional override)
COLLAB_EMBEDDING_PATH=""

# ============================================================
# NEW FEATURES: Structural Improvements
# ============================================================
# 注意：只开6，不开5 7
# Feature 5: Dynamic Batching (reduces 70% padding waste)
USE_DYNAMIC_BATCHING=false  # Start with false, test incrementally

# Feature 6: Item/Layer Position Embeddings
# Helps model recognize item boundaries and layer hierarchy
USE_ITEM_LAYER_EMB=true
USE_TEMPORAL_DECAY=true  # Add recency information

# Feature 7: Hierarchical Attention
# Item-level: which items are important
# Layer-level: which layers (coarse/fine) are important
USE_HIERARCHICAL_ATTN=false
USE_ITEM_ATTENTION=false   # Inter-item and intra-item attention
USE_LAYER_ATTENTION=false  # Layer-level attention

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

# Display feature status
echo "🚀 Enhanced Features:"
echo "   Codebook Warmstart: ${USE_CODEBOOK_WARMSTART}"
echo "   Codebook Prediction: ${USE_CODEBOOK_PREDICTION}"
echo "   Tag Prediction: ${USE_TAG_PREDICTION}"
echo "   Multimodal Fusion: ${USE_MULTIMODAL_FUSION}"

echo "   Trie-Constrained Decoding: ${USE_TRIE_CONSTRAINTS}"
echo ""
echo "🔧 Structural Improvements:"
echo "   Dynamic Batching: ${USE_DYNAMIC_BATCHING}"
echo "   Item/Layer Embeddings: ${USE_ITEM_LAYER_EMB}"
if [ "$USE_ITEM_LAYER_EMB" = true ]; then
    echo "     Temporal Decay: ${USE_TEMPORAL_DECAY}"
fi
echo "   Hierarchical Attention: ${USE_HIERARCHICAL_ATTN}"
if [ "$USE_HIERARCHICAL_ATTN" = true ]; then
    echo "     Item Attention: ${USE_ITEM_ATTENTION}"
    echo "     Layer Attention: ${USE_LAYER_ATTENTION}"
fi


echo "=================================================="

# Build command
CMD="python -m src.recommender.hidvae.train \
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



     


# Feature 1: Codebook Warmstart
if [ "$USE_CODEBOOK_WARMSTART" = true ]; then
    CMD="$CMD --use_codebook_warmstart"
    if [ "$CODEBOOK_WARMSTART_FREEZE" = true ]; then
        CMD="$CMD --codebook_warmstart_freeze"
    fi
fi

# Feature 2: Codebook Prediction
if [ "$USE_CODEBOOK_PREDICTION" = true ]; then
    CMD="$CMD --use_codebook_prediction --codebook_prediction_weight ${CODEBOOK_PREDICTION_WEIGHT}"
fi

# Feature 3: Tag Prediction
if [ "$USE_TAG_PREDICTION" = true ]; then
    CMD="$CMD --use_tag_prediction --tag_prediction_weight ${TAG_PREDICTION_WEIGHT}"
    if [ "$PREDICT_TAGS_FIRST" = true ]; then
        CMD="$CMD --predict_tags_first"
    fi
fi

# Feature 4: Multimodal Fusion
if [ "$USE_MULTIMODAL_FUSION" = true ]; then
    CMD="$CMD --use_multimodal_fusion --fusion_gate_type ${FUSION_GATE_TYPE}"
    if [ "$USE_LAYER_SPECIFIC_FUSION" = true ]; then
        CMD="$CMD --use_layer_specific_fusion"
    fi
    if [ "$FUSION_GATE_TYPE" = "fixed" ]; then
        CMD="$CMD --content_emb_weight ${CONTENT_EMB_WEIGHT}"
        CMD="$CMD --collab_emb_weight ${COLLAB_EMB_WEIGHT}"
        CMD="$CMD --id_emb_weight ${ID_EMB_WEIGHT}"
    fi
fi

# Collaborative embedding path
if [ -n "$COLLAB_EMBEDDING_PATH" ]; then
    CMD="$CMD --collab_embedding_path ${COLLAB_EMBEDDING_PATH}"
fi

# Feature 5: Dynamic Batching
if [ "$USE_DYNAMIC_BATCHING" = true ]; then
    CMD="$CMD --use_dynamic_batching"
fi

# Feature 6: Item/Layer Position Embeddings
if [ "$USE_ITEM_LAYER_EMB" = true ]; then
    CMD="$CMD --use_item_layer_emb"
    if [ "$USE_TEMPORAL_DECAY" = true ]; then
        CMD="$CMD --use_temporal_decay"
    fi
fi

# Feature 7: Hierarchical Attention
if [ "$USE_HIERARCHICAL_ATTN" = true ]; then
    CMD="$CMD --use_hierarchical_attn"
    if [ "$USE_ITEM_ATTENTION" = true ]; then
        CMD="$CMD --use_item_attention"
    fi
    if [ "$USE_LAYER_ATTENTION" = true ]; then
        CMD="$CMD --use_layer_attention"
    fi
fi

# Feature 8: Trie-Constrained Decoding
if [ "$USE_TRIE_CONSTRAINTS" = true ]; then
    CMD="$CMD --use_trie_constraints"
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


