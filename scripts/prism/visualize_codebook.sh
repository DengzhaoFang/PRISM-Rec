#!/bin/bash

# TIGER vs PRISM Codebook Comparison Visualization Script
# 
# This script generates a publication-quality figure comparing TIGER and PRISM
# codebook embeddings side by side with a shared legend.
# 
# Output: PNG, PDF, SVG formats for direct LaTeX insertion

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Model paths (relative to project root)
TIGER_MODEL="$PROJECT_ROOT/scripts/output/tiger_tokenizer/beauty/3-256-32-ema-only-5-core-items/final_model.pt"
PRISM_MODEL="$PROJECT_ROOT/scripts/output/prism_tokenizer/beauty/3-256-32-ema-only-5-core-items/best_model.pt"

# Output path (without extension)
OUTPUT_DIR="$PROJECT_ROOT/scripts/output/visualizations"
OUTPUT_PATH="$OUTPUT_DIR/codebook_comparison"

# Create output directory
mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_DIR/visualize_codebook_comparison.py" \
    --tiger_model "$TIGER_MODEL" \
    --prism_model "$PRISM_MODEL" \
    --output_path "$OUTPUT_PATH" \
    --perplexity 60 \
    --n_iter 3000 \
    --init pca \
    --dpi 300

echo ""
echo "✓ Codebook comparison visualization completed!"
echo ""
echo "Output files:"
echo "  - ${OUTPUT_PATH}.png (raster, for preview)"
echo "  - ${OUTPUT_PATH}.pdf (vector, for LaTeX)"
echo "  - ${OUTPUT_PATH}.svg (vector, for editing)"
