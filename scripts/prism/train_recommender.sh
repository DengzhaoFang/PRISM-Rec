#!/bin/bash

cd ../..


echo "=================================================="
echo "Training PRISM Recommender"
echo "=================================================="
echo ""

# ============================================================
# Resume from Checkpoint
# ============================================================
# Set this to resume training from a saved checkpoint
# When set, all hyperparameters will be loaded from the checkpoint
# Leave empty ("") to start fresh training
RESUME_CHECKPOINT=""

# ============================================================
# Configuration
# ============================================================
CONFIG="beauty"
DEVICE="cuda:0"
NUM_WORKERS=4
MODEL_TYPE="t5-tiny-2"

# Output directory keywords (optional, for custom naming)
# Example: "baseline-experiment" will create dir like "2025-11-06-17-26-56_baseline-experiment"
OUTPUT_KEYWORDS="3layer-prism-300epoch"


# Feature: Trie-Constrained Decoding
# Ensures every decoding step points to a path that can lead to a real item
# This eliminates invalid predictions and improves accuracy
USE_TRIE_CONSTRAINTS=true

# Feature: Adaptive Temperature Scaling
# Dynamically adjusts temperature based on semantic ID branch density
# Smaller temperature for dense branches (hard negatives) → stronger penalty
# Larger temperature for sparse branches (easy cases) → gentler penalty
USE_ADAPTIVE_TEMPERATURE=true  # Set to true to enable
TAU_ALPHA=0.5                   
TAU_MIN=0.7                     
TAU_MAX=0.8                     
TAU_START_LAYER=1               

# ============================================================
# Learning Rate Scheduler
# ============================================================
# Options: 'none', 'warmup_cosine', 'reduce_on_plateau', 'exponential', 'step'

LR_SCHEDULER="warmup_cosine"

# Enable verbose sample printing during validation and testing
VERBOSE=false


# Codebook Vector Prediction 
USE_CODEBOOK_PREDICTION=true
CODEBOOK_PREDICTION_WEIGHT=0.0005

# Tag ID Prediction 
USE_TAG_PREDICTION=true  
TAG_PREDICTION_WEIGHT=0.0005
PREDICT_TAGS_FIRST=true

# Multi-source Embedding Fusion
USE_MULTIMODAL_FUSION=true
FUSION_GATE_TYPE="moe"  # Options: learned, fixed, attention, moe
USE_LAYER_SPECIFIC_FUSION=true  # Use layer-specific projections (recommended for better performance)

# For fixed fusion: use conservative weights (ID should dominate)
CONTENT_EMB_WEIGHT=0.3      # For fixed fusion 
COLLAB_EMB_WEIGHT=0.3       # For fixed fusion 
ID_EMB_WEIGHT=0.4           # For fixed fusion

# For MOE fusion: advanced non-linear fusion with expert networks
MOE_NUM_EXPERTS=3           
MOE_EXPERT_HIDDEN_DIM=256   
MOE_TOP_K=2                
MOE_USE_LOAD_BALANCING=false 
MOE_LOAD_BALANCE_WEIGHT=0.001  
                            

# Improved Projection Mechanism for MOE Fusion
# When enabled, uses different projection dimensions for each source
MOE_USE_IMPROVED_PROJECTION=true  
MOE_CODEBOOK_DIM=32               # Codebook embedding dimension




# Item/Layer Position Embeddings
# Helps model recognize item boundaries and layer hierarchy
USE_ITEM_LAYER_EMB=true
USE_TEMPORAL_DECAY=true  # Add recency information



# Check if resuming from checkpoint
if [ -n "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ]; then
    echo "🔄 RESUMING FROM CHECKPOINT"
    echo "   Checkpoint: ${RESUME_CHECKPOINT}"
    echo "   All hyperparameters will be loaded from checkpoint"
    echo "   Only device and num_workers can be overridden"
    echo ""
else
    # Build command
    CMD="python -m src.recommender.prism.train \
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



    if [ "$USE_CODEBOOK_PREDICTION" = true ]; then
        CMD="$CMD --use_codebook_prediction --codebook_prediction_weight ${CODEBOOK_PREDICTION_WEIGHT}"
    fi

    if [ "$USE_TAG_PREDICTION" = true ]; then
        CMD="$CMD --use_tag_prediction --tag_prediction_weight ${TAG_PREDICTION_WEIGHT}"
        if [ "$PREDICT_TAGS_FIRST" = true ]; then
            CMD="$CMD --predict_tags_first"
        fi
    fi

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
        if [ "$FUSION_GATE_TYPE" = "moe" ]; then
            CMD="$CMD --moe_num_experts ${MOE_NUM_EXPERTS}"
            CMD="$CMD --moe_expert_hidden_dim ${MOE_EXPERT_HIDDEN_DIM}"
            CMD="$CMD --moe_top_k ${MOE_TOP_K}"
            if [ "$MOE_USE_LOAD_BALANCING" = true ]; then
                CMD="$CMD --moe_use_load_balancing"
            fi
            CMD="$CMD --moe_load_balance_weight ${MOE_LOAD_BALANCE_WEIGHT}"
            if [ "$MOE_USE_IMPROVED_PROJECTION" = true ]; then
                CMD="$CMD --moe_use_improved_projection"
            fi
            CMD="$CMD --moe_codebook_dim ${MOE_CODEBOOK_DIM}"
        fi
    fi

    # Collaborative embedding path
    if [ -n "$COLLAB_EMBEDDING_PATH" ]; then
        CMD="$CMD --collab_embedding_path ${COLLAB_EMBEDDING_PATH}"
    fi

    if [ "$USE_DYNAMIC_BATCHING" = true ]; then
        CMD="$CMD --use_dynamic_batching"
    fi

    if [ "$USE_ITEM_LAYER_EMB" = true ]; then
        CMD="$CMD --use_item_layer_emb"
        if [ "$USE_TEMPORAL_DECAY" = true ]; then
            CMD="$CMD --use_temporal_decay"
        fi
    fi

    if [ "$USE_TRIE_CONSTRAINTS" = true ]; then
        CMD="$CMD --use_trie_constraints"
    fi

    if [ "$USE_ADAPTIVE_TEMPERATURE" = true ]; then
        CMD="$CMD --use_adaptive_temperature"
        if [ -n "$TAU_ALPHA" ]; then
            CMD="$CMD --tau_alpha ${TAU_ALPHA}"
        fi
        if [ -n "$TAU_MIN" ]; then
            CMD="$CMD --tau_min ${TAU_MIN}"
        fi
        if [ -n "$TAU_MAX" ]; then
            CMD="$CMD --tau_max ${TAU_MAX}"
        fi
        if [ -n "$TAU_START_LAYER" ]; then
            CMD="$CMD --tau_start_layer ${TAU_START_LAYER}"
        fi
    fi
fi

# If resuming from checkpoint, override the command
if [ -n "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ]; then
    CMD="python -m src.recommender.prism.train \
        --resume ${RESUME_CHECKPOINT} \
        --device ${DEVICE} \
        --num_workers ${NUM_WORKERS}"
fi

eval $CMD


echo ""
echo "=================================================="
echo "✓ Training completed!"
echo "=================================================="


