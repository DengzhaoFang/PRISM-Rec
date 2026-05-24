#!/bin/bash

cd ../..

echo "=================================================="
echo "Training PRISM Recommender (DSI with Purified Features)"
echo "=================================================="
echo ""

# ============================================================
# Resume from Checkpoint
# ============================================================
RESUME_CHECKPOINT=""

# ============================================================
# Configuration
# ============================================================
CONFIG="beauty"
DEVICE="cuda:2"
NUM_WORKERS=4
MODEL_TYPE="t5-tiny-2"
OUTPUT_KEYWORDS="clean-collab-teach-text2"

# ============================================================
# DSI: Dynamic Semantic Integration (Dense Modality Routing)
# ============================================================
# gate_type="dense": softmax router, all 3 experts always active
#   No top-k truncation, no load balancing — continuous modality mixing
USE_MULTIMODAL_FUSION=true
FUSION_GATE_TYPE="dense"      # dense | moe | learned | attention | fixed
MOE_NUM_EXPERTS=3
MOE_EXPERT_HIDDEN_DIM=256
MOE_TOP_K=2                   # ignored in dense mode
MOE_USE_LOAD_BALANCING=false  # ignored in dense mode
MOE_LOAD_BALANCE_WEIGHT=0.01  # ignored in dense mode

# ============================================================
# Purified Semantic Predictor (auxiliary MSE on target z_clean)
# ============================================================
USE_PURIFIED_PREDICTOR=true
PURIFIED_PREDICTOR_WEIGHT=0.1

# ============================================================
# Structural features
# ============================================================
USE_ITEM_LAYER_EMB=true
USE_TEMPORAL_DECAY=true

# ============================================================
# Trie-Constrained Decoding
# ============================================================
USE_TRIE_CONSTRAINTS=true

# ============================================================
# Adaptive Temperature Scaling
# ============================================================
USE_ADAPTIVE_TEMPERATURE=true
TAU_ALPHA=0.5
TAU_MIN=0.7  
TAU_MAX=0.8
TAU_START_LAYER=1

# ============================================================
# Learning Rate Scheduler
# ============================================================
LR_SCHEDULER="warmup_cosine"
EVAL_EVERY_N=20  # evaluate every N epochs (1 = every epoch, 3 = every 3rd)
VERBOSE=false

# Check if resuming
if [ -n "$RESUME_CHECKPOINT" ] && [ -f "$RESUME_CHECKPOINT" ]; then
    echo "Resuming from checkpoint: ${RESUME_CHECKPOINT}"
    CMD="python -m src.recommender.prism.train \
        --resume ${RESUME_CHECKPOINT} \
        --device ${DEVICE} \
        --num_workers ${NUM_WORKERS}"
else
    CMD="python -m src.recommender.prism.train \
        --config ${CONFIG} \
        --device ${DEVICE} \
        --num_workers ${NUM_WORKERS} \
        --model_type ${MODEL_TYPE} \
        --output_keywords ${OUTPUT_KEYWORDS}"

    [ -n "$LR_SCHEDULER" ] && CMD="$CMD --lr_scheduler ${LR_SCHEDULER}"
    [ -n "$EVAL_EVERY_N" ] && CMD="$CMD --eval_every_n_epochs ${EVAL_EVERY_N}"
    [ "$VERBOSE" = true ] && CMD="$CMD --verbose"

    if [ "$USE_MULTIMODAL_FUSION" = true ]; then
        CMD="$CMD --use_multimodal_fusion --fusion_gate_type ${FUSION_GATE_TYPE}"
        if [ "$FUSION_GATE_TYPE" = "moe" ]; then
            CMD="$CMD --moe_num_experts ${MOE_NUM_EXPERTS}"
            CMD="$CMD --moe_expert_hidden_dim ${MOE_EXPERT_HIDDEN_DIM}"
            CMD="$CMD --moe_top_k ${MOE_TOP_K}"
            [ "$MOE_USE_LOAD_BALANCING" = true ] && CMD="$CMD --moe_use_load_balancing"
            CMD="$CMD --moe_load_balance_weight ${MOE_LOAD_BALANCE_WEIGHT}"
        fi
    fi

    if [ "$USE_PURIFIED_PREDICTOR" = true ]; then
        CMD="$CMD --use_purified_predictor"
        [ -n "$PURIFIED_PREDICTOR_WEIGHT" ] && CMD="$CMD --purified_predictor_weight ${PURIFIED_PREDICTOR_WEIGHT}"
    fi

    if [ "$USE_ITEM_LAYER_EMB" = true ]; then
        CMD="$CMD --use_item_layer_emb"
        [ "$USE_TEMPORAL_DECAY" = true ] && CMD="$CMD --use_temporal_decay"
    fi

    [ "$USE_TRIE_CONSTRAINTS" = true ] && CMD="$CMD --use_trie_constraints"

    if [ "$USE_ADAPTIVE_TEMPERATURE" = true ]; then
        CMD="$CMD --use_adaptive_temperature"
        [ -n "$TAU_ALPHA" ] && CMD="$CMD --tau_alpha ${TAU_ALPHA}"
        [ -n "$TAU_MIN" ] && CMD="$CMD --tau_min ${TAU_MIN}"
        [ -n "$TAU_MAX" ] && CMD="$CMD --tau_max ${TAU_MAX}"
        [ -n "$TAU_START_LAYER" ] && CMD="$CMD --tau_start_layer ${TAU_START_LAYER}"
    fi
fi

eval $CMD

echo ""
echo "=================================================="
echo "Training completed!"
echo "=================================================="
