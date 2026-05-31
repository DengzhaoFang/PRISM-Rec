#!/bin/bash

# Quick script to generate publication-quality efficiency comparison plots
# 
# Usage:
#   bash plot_efficiency.sh              # Generate all plots
#   bash plot_efficiency.sh scatter      # Generate only scatter plot
#   bash plot_efficiency.sh bars         # Generate only bar chart
#   bash plot_efficiency.sh radar        # Generate only radar chart

cd ../..

PLOT_TYPE=${1:-all}

echo "=================================================="
echo "Generating Efficiency Comparison Plots"
echo "=================================================="
echo ""
echo "Plot type: ${PLOT_TYPE}"
echo ""

python scripts/prism/plot_efficiency_comparison.py \
    --plot_type ${PLOT_TYPE} \
    --results_path scripts/output/efficiency_benchmark/efficiency_results.json \
    --output_dir scripts/output/efficiency_benchmark

echo ""
echo "=================================================="
echo "✓ Plots generated!"
echo "=================================================="
echo ""
echo "Output directory: scripts/output/efficiency_benchmark/"
echo ""
echo "Generated plots:"
if [ "$PLOT_TYPE" = "all" ] || [ "$PLOT_TYPE" = "scatter" ]; then
    echo "  - efficiency_scatter.pdf (Pareto frontier visualization)"
fi
if [ "$PLOT_TYPE" = "all" ] || [ "$PLOT_TYPE" = "bars" ]; then
    echo "  - efficiency_bars_clean.pdf (Clean grouped comparison)"
fi
if [ "$PLOT_TYPE" = "all" ] || [ "$PLOT_TYPE" = "radar" ]; then
    echo "  - efficiency_radar.pdf (Multi-dimensional trade-off)"
fi
echo ""

