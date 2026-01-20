#!/bin/bash
# ============================================================================
# RQ-VAE Ablation Study: Comparing Different Input Modes
# ============================================================================
#
# This script runs five RQ-VAE training modes to compare:
# Mode 1 (semantic_only): RQ-VAE input is only semantic embedding
# Mode 2 (collab_only): RQ-VAE input is only collaborative embedding
# Mode 3 (concat): RQ-VAE input is concatenation of semantic and collab embeddings
# Mode 4 (contrastive): RQ-VAE input is semantic, with contrastive loss to collab
# Mode 5 (gated_dual): RQ-VAE input is semantic + gated(denoised) collab,
#                      with dual reconstruction heads
#
# Usage:
#   ./run_rqvae_ablation.sh [dataset] [gpu_id] [mode]
#
# Examples:
#   ./run_rqvae_ablation.sh beauty 0           # Run all modes on beauty dataset
#   ./run_rqvae_ablation.sh beauty 0 semantic_only  # Run only semantic_only mode
#   ./run_rqvae_ablation.sh sports 1 contrastive    # Run contrastive mode on sports
#   ./run_rqvae_ablation.sh beauty 0 gated_dual     # Run gated_dual mode
#
# ============================================================================

set -e

# Parse arguments
DATASET=${1:-beauty}
GPU_ID=${2:-0}
MODE=${3:-all}  # all, semantic_only, collab_only, concat, contrastive

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Navigate to project root
cd "$(dirname "$0")/../.."

# ============================================================================
# Dataset paths configuration
# ============================================================================
case $DATASET in
    beauty)
        DATA_DIR="dataset/Amazon-Beauty/processed/beauty-prism-sentenceT5base/Beauty"
        SEMANTIC_EMB="${DATA_DIR}/item_emb.parquet"
        COLLAB_EMB="${DATA_DIR}/lightgcn/item_embeddings_collab.npy"
        ;;
    sports)
        DATA_DIR="dataset/Amazon-Sports/processed/sports-prism-sentenceT5base/Sports"
        SEMANTIC_EMB="${DATA_DIR}/item_emb.parquet"
        COLLAB_EMB="${DATA_DIR}/lightgcn/item_embeddings_collab.npy"
        ;;
    toys)
        DATA_DIR="dataset/Amazon-Toys/processed/toys-prism-sentenceT5base/Toys"
        SEMANTIC_EMB="${DATA_DIR}/item_emb.parquet"
        COLLAB_EMB="${DATA_DIR}/lightgcn/item_embeddings_collab.npy"
        ;;
    cds)
        DATA_DIR="dataset/Amazon-CDs/processed/cds-prism-sentenceT5base/CDs"
        SEMANTIC_EMB="${DATA_DIR}/item_emb.parquet"
        COLLAB_EMB="${DATA_DIR}/lightgcn/item_embeddings_collab.npy"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Supported: beauty, sports, toys, cds"
        exit 1
        ;;
esac

# Output base directory
OUTPUT_BASE="scripts/output/rqvae_ablation/${DATASET}"

# ============================================================================
# Common hyperparameters
# ============================================================================
EPOCHS=500
BATCH_SIZE=256
LEARNING_RATE=1e-4
LATENT_DIM=64
N_EMBED=256
N_LAYERS=3
BETA=0.25
EARLY_STOP_PATIENCE=50

# Contrastive learning specific
CONTRASTIVE_WEIGHT=1.0
CONTRASTIVE_TEMPERATURE=0.07

# Gated dual mode specific
SEMANTIC_RECON_WEIGHT=1.0
COLLAB_RECON_WEIGHT=2.0  # Higher weight to encourage collab reconstruction

# Gate supervision parameters (from PRISM)
USE_GATE_SUPERVISION="--use_gate_supervision"
GATE_SUPERVISION_WEIGHT=0.8   # Much higher than default 0.1
GATE_DIVERSITY_WEIGHT=3.5     # Much higher than default 0.5
GATE_TARGET_STD=0.3           # Higher target std

# ============================================================================
# Training functions
# ============================================================================
run_mode() {
    local mode=$1
    local output_dir="${OUTPUT_BASE}/${mode}"
    
    echo "============================================================================"
    echo "Running Mode: ${mode}"
    echo "Output: ${output_dir}"
    echo "============================================================================"
    
    mkdir -p "$output_dir"
    
    # Build command
    CMD="python scripts/prism/rqvae_ablation_study.py \
        --semantic_emb_path ${SEMANTIC_EMB} \
        --output_dir ${output_dir} \
        --mode ${mode} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --latent_dim ${LATENT_DIM} \
        --n_embed ${N_EMBED} \
        --n_layers ${N_LAYERS} \
        --beta ${BETA} \
        --early_stop_patience ${EARLY_STOP_PATIENCE} \
        --use_ema"
    
    # Add collab embedding path for modes that need it
    if [[ "$mode" != "semantic_only" ]]; then
        CMD="${CMD} --collab_emb_path ${COLLAB_EMB}"
    fi
    
    # Add contrastive-specific params
    if [[ "$mode" == "contrastive" ]]; then
        CMD="${CMD} \
            --contrastive_weight ${CONTRASTIVE_WEIGHT} \
            --contrastive_temperature ${CONTRASTIVE_TEMPERATURE}"
    fi
    
    # Add gated_dual-specific params
    if [[ "$mode" == "gated_dual" ]]; then
        CMD="${CMD} \
            --semantic_recon_weight ${SEMANTIC_RECON_WEIGHT} \
            --collab_recon_weight ${COLLAB_RECON_WEIGHT} \
            ${USE_GATE_SUPERVISION} \
            --gate_supervision_weight ${GATE_SUPERVISION_WEIGHT} \
            --gate_diversity_weight ${GATE_DIVERSITY_WEIGHT} \
            --gate_target_std ${GATE_TARGET_STD}"
        # Note: popularity_score is now loaded from item_emb.parquet directly
        # No need for separate --popularity_path argument
    fi
    
    echo "Command: $CMD"
    echo ""
    
    # Run training
    eval $CMD
    
    echo ""
    echo "✓ Mode ${mode} completed!"
    echo ""
}

# ============================================================================
# Main execution
# ============================================================================
echo "============================================================================"
echo "RQ-VAE Ablation Study"
echo "============================================================================"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU_ID}"
echo "Mode: ${MODE}"
echo "Semantic embeddings: ${SEMANTIC_EMB}"
echo "Collaborative embeddings: ${COLLAB_EMB}"
echo "============================================================================"
echo ""

# Check if files exist
if [[ ! -f "$SEMANTIC_EMB" ]]; then
    echo "Error: Semantic embedding file not found: $SEMANTIC_EMB"
    exit 1
fi

if [[ "$MODE" != "semantic_only" && ! -f "$COLLAB_EMB" ]]; then
    echo "Error: Collaborative embedding file not found: $COLLAB_EMB"
    echo "Please train LightGCN first to generate collaborative embeddings."
    exit 1
fi

# Run specified mode(s)
if [[ "$MODE" == "all" ]]; then
    echo "Running all five modes..."
    echo ""
    
    run_mode "semantic_only"
    run_mode "collab_only"
    run_mode "concat"
    run_mode "contrastive"
    run_mode "gated_dual"
    
    echo "============================================================================"
    echo "All modes completed!"
    echo "Results saved to: ${OUTPUT_BASE}"
    echo "============================================================================"
else
    run_mode "$MODE"
fi

echo ""
echo "Done!"
