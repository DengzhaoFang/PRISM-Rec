#!/bin/bash

# TIGER vs PRISM Embedding Comparison Visualization Script
#
# This script generates a publication-quality figure comparing TIGER and PRISM
# recommendation embeddings side by side with a shared legend (2 rows for 10 categories).

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=================================================="
echo "Embedding Quality Analysis: PRISM vs TIGER"
echo "=================================================="
echo ""

# ============================================================
# Configuration
# ============================================================

# Dataset
DATASET="beauty"

# Model checkpoints
PRISM_CHECKPOINT="scripts/output/recommender/prism/beauty/2026-01-06-21-58-26_3layer-prism/best_model.pt"
TIGER_CHECKPOINT="scripts/output/recommender/tiger/beauty/2026-01-06-22-02-28_3layer-tiger/best_model.pt"

# Output directory
OUTPUT_DIR="scripts/prism/embedding_analysis_${DATASET}"

# Device
DEVICE="cuda:3"

# Number of items to sample for analysis
NUM_SAMPLES=50000

# Visualization method (tsne or umap)
VIS_METHOD="tsne"

# ============================================================
# Run Analysis
# ============================================================

echo "Configuration:"
echo "  Dataset: ${DATASET}"
echo "  PRISM checkpoint: ${PRISM_CHECKPOINT}"
echo "  TIGER checkpoint: ${TIGER_CHECKPOINT}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Device: ${DEVICE}"
echo "  Number of samples: ${NUM_SAMPLES}"
echo "  Visualization method: ${VIS_METHOD}"
echo ""
echo "=================================================="
echo ""

python scripts/prism/analyze_embedding_geometry.py \
    --prism_checkpoint "${PRISM_CHECKPOINT}" \
    --tiger_checkpoint "${TIGER_CHECKPOINT}" \
    --dataset "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}" \
    --num_samples ${NUM_SAMPLES} \
    --vis_method "${VIS_METHOD}"

echo ""
echo "=================================================="
echo "✓ Analysis completed!"
echo ""
echo "Output files:"
echo "  - ${OUTPUT_DIR}/embedding_${VIS_METHOD}_comparison.png (raster, for preview)"
echo "  - ${OUTPUT_DIR}/embedding_${VIS_METHOD}_comparison.pdf (vector, for LaTeX)"
echo "  - ${OUTPUT_DIR}/embedding_${VIS_METHOD}_comparison.svg (vector, for editing)"
echo "  - ${OUTPUT_DIR}/metrics_comparison.png (metrics bar chart)"
echo "  - ${OUTPUT_DIR}/metrics.json (quantitative metrics)"
echo ""
echo "Layout:"
echo "  ✓ Legend on top (2 rows, 5 categories each)"
echo "  ✓ TIGER Recommendation (left) | PRISM Recommendation (right)"
echo "=================================================="
