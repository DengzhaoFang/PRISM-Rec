#!/bin/bash

cd ../..

# Efficiency Benchmark for Generative vs Discriminative Recommender Models
#
# Measures inference speed and activated parameters for:
# - Generative: TIGER, LETTER, ActionPiece, EAGER, Prism
# - Discriminative: SASRec (simulated)
#
# Supports multiple datasets to demonstrate:
# - Generative models: O(1) w.r.t. catalog size
# - Discriminative models: O(N) or O(log N) w.r.t. catalog size
#
# Usage:
#   bash efficiency_benchmark.sh                    # Run full benchmark
#   bash efficiency_benchmark.sh --plot             # Only regenerate plots from existing results
#   bash efficiency_benchmark.sh --only_AP_CDs      # Only test ActionPiece on CDs dataset

echo "=================================================="
echo "Efficiency Benchmark: Generative vs Discriminative"
echo "=================================================="
echo ""

# ============================================================
# Configuration
# ============================================================

DEVICE="cuda:2"

# Number of test samples (same samples used for all models)
NUM_SAMPLES=500

# Beam size for generation
BEAM_SIZE=20

# Output directory
OUTPUT_DIR="scripts/output/efficiency_benchmark"

# Models to benchmark (space-separated)
# Available: TIGER LETTER ActionPiece EAGER Prism
MODELS="TIGER LETTER ActionPiece EAGER Prism"

# Datasets to benchmark (space-separated)
# Available: beauty cds
# beauty: ~12K items (smaller catalog)
# cds: ~64K items (larger catalog, shows SASRec scaling)
DATASETS="beauty cds"

PLOT_ONLY=""

ONLY_AP_CDS=""
if [ "$1" == "--only_AP_CDs" ]; then
    ONLY_AP_CDS="--only_AP_CDs"
    MODELS="ActionPiece"
    DATASETS="cds"
    echo "🎯 Only testing ActionPiece on CDs dataset"
    echo ""
fi

# ============================================================
# Run Benchmark
# ============================================================

echo "✅ Device: ${DEVICE}"
echo "   Num samples: ${NUM_SAMPLES}"
echo "   Beam size: ${BEAM_SIZE}"
echo "   Models: ${MODELS}"
echo "   Datasets: ${DATASETS}"
echo "   Output: ${OUTPUT_DIR}"
echo ""
echo "=================================================="

python -m src.recommender.prism.efficiency_benchmark \
    --device ${DEVICE} \
    --num_samples ${NUM_SAMPLES} \
    --beam_size ${BEAM_SIZE} \
    --output_dir "${OUTPUT_DIR}" \
    --models ${MODELS} \
    --datasets ${DATASETS} \
    ${PLOT_ONLY} \
    ${ONLY_AP_CDS}

echo ""
echo "=================================================="
echo "✓ Benchmark completed!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - ${OUTPUT_DIR}/efficiency_results.json"
echo "  - ${OUTPUT_DIR}/efficiency_comparison.pdf"
echo "  - ${OUTPUT_DIR}/efficiency_comparison.png"
echo ""


