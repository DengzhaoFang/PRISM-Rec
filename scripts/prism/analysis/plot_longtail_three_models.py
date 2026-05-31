"""
Plotting functions for three-model longtail comparison.
Optimized for TIGER vs PRISM vs ActionPiece.
"""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import numpy as np
import os
from typing import Dict, List, Tuple
from pathlib import Path
import logging

matplotlib.use('Agg')
logger = logging.getLogger(__name__)

# Configure Linux Libertine font
libertine_font_dir = '/home/fangdengzhao/Fonts/libertine/opentype'
libertine_fonts = [
    f'{libertine_font_dir}/LinLibertine_R.otf',      # Regular
    f'{libertine_font_dir}/LinLibertine_RI.otf',     # Italic
    f'{libertine_font_dir}/LinLibertine_RB.otf',     # Bold
    f'{libertine_font_dir}/LinLibertine_RBI.otf',    # Bold Italic
]

for font_file in libertine_fonts:
    if os.path.exists(font_file):
        fm.fontManager.addfont(font_file)


def plot_multi_dataset_comparison_three_models(
    all_results: Dict[str, Tuple[Dict, Dict, Dict]],
    output_path: str,
    metrics: List[str] = None,
    dataset_configs: Dict = None
):
    """
    Generate publication-quality figure with three models (TIGER, PRISM, ActionPiece).
    
    Layout per group: TIGER_R10, ActionPiece_R10, PRISM_R10, TIGER_N10, ActionPiece_N10, PRISM_N10
    Models distinguished by hatch patterns, metrics by colors.
    """
    metrics = metrics or ['Recall@10', 'NDCG@10']
    
    # Publication-quality settings - same as efficiency plot
    plt.rcParams.update({
        'font.family': 'Linux Libertine O',
        'font.size': 31,
        'axes.labelsize': 27,
        'axes.titlesize': 27,
        'xtick.labelsize': 27,
        'ytick.labelsize': 27,
        'legend.fontsize': 29,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'text.usetex': False,
    })
    
    dataset_names = list(all_results.keys())
    num_datasets = len(dataset_names)
    
    # Figure size - enlarged to accommodate larger font
    fig, axes = plt.subplots(1, num_datasets, figsize=(17, 8))
    if num_datasets == 1:
        axes = [axes]
    
    # Color scheme - models distinguished by color (light pastel colors)
    model_colors = {
        'TIGER': '#F5B7B1',       # Light red/coral for TIGER
        'ActionPiece': '#A8C5E3', # Light blue for ActionPiece
        'PRISM': '#C5E3A8',       # Light green for PRISM
    }
    
    # Hatch patterns - metrics distinguished by pattern
    metric_hatches = {
        'Recall': '//',       # Forward slash pattern for Recall
        'NDCG': 'oo',         # Circle pattern for NDCG
    }
    
    # Model order for each metric group
    model_order = ['TIGER', 'ActionPiece', 'PRISM']
    
    for ax_idx, dataset_name in enumerate(dataset_names):
        ax = axes[ax_idx]
        tiger_results, prism_results, ap_results = all_results[dataset_name]
        
        results_map = {
            'TIGER': tiger_results,
            'ActionPiece': ap_results,
            'PRISM': prism_results
        }
        
        group_names = list(tiger_results['per_group'].keys())
        num_groups = len(group_names)
        
        # Bar layout: 6 bars per group (3 models x 2 metrics)
        # Order: TIGER_R10, AP_R10, PRISM_R10, TIGER_N10, AP_N10, PRISM_N10
        bar_width = 0.12
        gap_within_metric = 0.02  # Small gap between bars of same metric
        gap_between_metrics = 0.08  # Larger gap between R10 and N10 groups
        
        # Calculate positions
        group_width = 3 * bar_width + 2 * gap_within_metric + gap_between_metrics + 3 * bar_width + 2 * gap_within_metric
        x = np.arange(num_groups) * (group_width + 0.3)  # Space between groups
        
        # Collect all values for y-axis auto-scaling
        all_values = []
        for g in group_names:
            for model_name in model_order:
                for metric in metrics:
                    val = results_map[model_name]['per_group'][g].get(metric, 0)
                    all_values.append(val)
        
        # Plot bars
        for metric_idx, metric in enumerate(metrics):
            metric_type = 'Recall' if 'Recall' in metric else 'NDCG'
            hatch = metric_hatches[metric_type]
            
            for model_idx, model_name in enumerate(model_order):
                values = [results_map[model_name]['per_group'][g].get(metric, 0) for g in group_names]
                color = model_colors[model_name]
                
                # Calculate bar position
                # First metric group: positions 0, 1, 2
                # Second metric group: positions 3, 4, 5 (with gap)
                if metric_idx == 0:
                    offset = model_idx * (bar_width + gap_within_metric) - group_width / 2 + bar_width / 2
                else:
                    offset = (3 * (bar_width + gap_within_metric) + gap_between_metrics + 
                             model_idx * (bar_width + gap_within_metric) - group_width / 2 + bar_width / 2)
                
                ax.bar(
                    x + offset, values, bar_width,
                    color=color,
                    alpha=0.9,
                    edgecolor='#333333',
                    linewidth=0.8,
                    hatch=hatch,
                    zorder=3
                )
        
        # Auto-scale y-axis based on actual values
        y_max = max(all_values) * 1.15 if all_values else 0.15
        ax.set_ylim(0, y_max)
        
        # Subplot styling - same as efficiency plot
        display_name = dataset_configs[dataset_name]['display_name'] if dataset_configs else dataset_name.title()
        ax.set_title(f'{display_name}', fontweight='normal', pad=12, fontsize=33)
        ax.set_xticks(x)
        
        # Create x-axis labels with sample counts below group names
        group_counts = tiger_results.get('group_counts', [0] * num_groups)
        xlabels = [f'{name}\n(n={group_counts[i]:,})' for i, name in enumerate(group_names)]
        ax.set_xticklabels(xlabels, rotation=0, fontsize=33, linespacing=1.3)
        
        ax.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='#CCCCCC', zorder=0)
        ax.set_axisbelow(True)
        ax.set_facecolor('#FFFFFF')
    
    # Create legend elements
    from matplotlib.patches import Patch
    
    # Model legend (colors without pattern)
    model_legend_elements = [
        Patch(facecolor=model_colors[model], edgecolor='#333333', linewidth=0.8,
              label=model)
        for model in model_order
    ]
    
    # Metric legend (patterns with white background)
    metric_legend_elements = [
        Patch(facecolor='#FFFFFF', edgecolor='#333333', linewidth=0.8,
              hatch=metric_hatches['Recall'], label='Recall@10'),
        Patch(facecolor='#FFFFFF', edgecolor='#333333', linewidth=0.8,
              hatch=metric_hatches['NDCG'], label='NDCG@10'),
    ]
    
    # Combine legends
    all_legend_elements = model_legend_elements + metric_legend_elements
    
    # Add legend at the top of the figure, horizontal layout
    fig.legend(handles=all_legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 1.02), ncol=5,
               frameon=True, framealpha=0.98, edgecolor='#CCCCCC', fancybox=False,
               borderpad=0.7, columnspacing=1.5)
    
    plt.tight_layout(pad=1.5, rect=[0, 0, 1, 0.92])  # Leave space at top for legend
    
    # Save in multiple formats
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    
    # PDF
    pdf_path = f"{base_path}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.08)
    logger.info(f"Saved PDF: {pdf_path}")
    
    # PNG
    png_path = f"{base_path}.png"
    plt.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.08, dpi=300)
    logger.info(f"Saved PNG: {png_path}")
    
    plt.close()


def print_results_table_three_models(all_results: Dict[str, Tuple[Dict, Dict, Dict]], metrics: List[str]):
    """Print comparison table for all three models."""
    for dataset_name, (tiger_results, prism_results, ap_results) in all_results.items():
        group_names = list(tiger_results['per_group'].keys())
        
        print("\n" + "=" * 90)
        print(f"LONG-TAIL COMPARISON: {dataset_name.upper()} (TIGER vs PRISM vs ActionPiece)")
        print("=" * 90)
        
        for group in group_names:
            count = tiger_results['group_counts'][group_names.index(group)]
            print(f"\n{group} (n={count}):")
            print("-" * 80)
            
            for metric in metrics:
                t_val = tiger_results['per_group'][group].get(metric, 0)
                p_val = prism_results['per_group'][group].get(metric, 0)
                a_val = ap_results['per_group'][group].get(metric, 0)
                
                # Calculate improvements
                if t_val > 0:
                    p_imp = (p_val - t_val) / t_val * 100
                    a_imp = (a_val - t_val) / t_val * 100
                    p_imp_str = f"{p_imp:+.1f}%"
                    a_imp_str = f"{a_imp:+.1f}%"
                else:
                    p_imp_str = "N/A"
                    a_imp_str = "N/A"
                
                print(f"  {metric:12s}: TIGER={t_val:.4f}, PRISM={p_val:.4f} ({p_imp_str}), ActionPiece={a_val:.4f} ({a_imp_str})")
        
        print("\n" + "-" * 80)
        print("Overall:")
        for metric in metrics:
            t_val = tiger_results['overall'].get(metric, 0)
            p_val = prism_results['overall'].get(metric, 0)
            a_val = ap_results['overall'].get(metric, 0)
            
            if t_val > 0:
                p_imp = (p_val - t_val) / t_val * 100
                a_imp = (a_val - t_val) / t_val * 100
                p_imp_str = f"{p_imp:+.1f}%"
                a_imp_str = f"{a_imp:+.1f}%"
            else:
                p_imp_str = "N/A"
                a_imp_str = "N/A"
            
            print(f"  {metric:12s}: TIGER={t_val:.4f}, PRISM={p_val:.4f} ({p_imp_str}), ActionPiece={a_val:.4f} ({a_imp_str})")
        
        print("=" * 90)
