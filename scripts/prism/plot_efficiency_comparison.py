"""
Publication-quality efficiency comparison plot for generative recommender models.

This script creates a sophisticated visualization comparing models across three dimensions:
- Model size (activated parameters)
- Inference latency
- Recommendation quality (R@10)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import argparse

# Try to import scienceplots for publication-quality styling
try:
    import scienceplots
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False
    print("Warning: scienceplots not installed. Using custom styling.")
    print("Install with: pip install scienceplots")


# R@10 performance data (from experimental results)
RECALL_AT_10 = {
    'beauty': {
        'TIGER': 0.0588,
        'LETTER': 0.0616,
        'EAGER': 0.0600,
        'ActionPiece': 0.0680,
        'Prism': 0.0713,
    },
    'cds': {
        'TIGER': 0.0580,
        'LETTER': 0.0515,
        'EAGER': 0.0510,
        'ActionPiece': 0.0348,
        'Prism': 0.0777,
    }
}


def setup_publication_style():
    """Setup publication-quality matplotlib style."""
    
    # Configure Linux Libertine font
    libertine_font_dir = '/home/fangdengzhao/Fonts/libertine/opentype'
    libertine_fonts = [
        f'{libertine_font_dir}/LinLibertine_R.otf',
        f'{libertine_font_dir}/LinLibertine_RI.otf',
        f'{libertine_font_dir}/LinLibertine_RB.otf',
        f'{libertine_font_dir}/LinLibertine_RBI.otf',
    ]
    
    for font_file in libertine_fonts:
        if Path(font_file).exists():
            fm.fontManager.addfont(font_file)
    
    if HAS_SCIENCEPLOTS:
        try:
            # Use IEEE style as base (clean, professional) - no-latex version
            plt.style.use(['science', 'no-latex'])
        except:
            # If that fails, just use science
            try:
                plt.style.use('science')
            except:
                pass
    
    # Custom refinements for top-tier conference papers
    plt.rcParams.update({
        'font.family': 'Linux Libertine O',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'lines.linewidth': 1.5,
        'lines.markersize': 8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'text.usetex': False,  # Disable LaTeX
    })


def get_model_colors():
    """Get sophisticated color palette for models.
    
    Uses muted, professional colors inspired by Nature/Science journals.
    Avoids high saturation and ensures good contrast.
    """
    return {
        'TIGER': '#2E5F8A',      # Deep blue
        'LETTER': '#C85A54',     # Terracotta
        'EAGER': '#8B7BA8',      # Muted purple
        'ActionPiece': '#6B9D59', # Olive green
        'PRISM': '#D4A03A',      # Golden (our method - stands out but not garish)
    }


def get_display_names():
    """Get display names for models."""
    return {
        'TIGER': 'TIGER',
        'LETTER': 'LETTER',
        'EAGER': 'EAGER',
        'ActionPiece': 'ActionPiece',
        'Prism': 'PRISM',
    }


def load_efficiency_data(results_path):
    """Load efficiency benchmark results."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_scatter_comparison(all_results, output_path):
    """Create scatter plot comparing efficiency vs performance.
    
    This visualization shows the trade-off between:
    - X-axis: Inference latency (ms)
    - Y-axis: Recommendation quality (R@10)
    - Marker size: Model size (activated parameters)
    
    This is a common visualization in ML papers to show Pareto frontiers.
    """
    setup_publication_style()
    
    model_colors = get_model_colors()
    display_names = get_display_names()
    
    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    
    # Create figure with subplots for each dataset
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5))
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        results = all_results[dataset]
        recall_data = RECALL_AT_10[dataset]
        
        # Collect data for each model
        latencies = []
        recalls = []
        params = []
        colors = []
        labels = []
        
        for model_name in ['TIGER', 'LETTER', 'EAGER', 'ActionPiece', 'Prism']:
            if model_name not in results:
                continue
            
            latency = results[model_name]['timing']['mean_ms']
            recall = recall_data[model_name]
            param = results[model_name]['activated_params'] / 1e6  # Convert to millions
            
            latencies.append(latency)
            recalls.append(recall)
            params.append(param)
            colors.append(model_colors[display_names.get(model_name, model_name)])
            labels.append(display_names.get(model_name, model_name))
        
        # Normalize marker sizes (scale by parameter count)
        # Use square root scaling for better visual perception
        min_size = 100
        max_size = 500
        param_array = np.array(params)
        sizes = min_size + (max_size - min_size) * (param_array - param_array.min()) / (param_array.max() - param_array.min())
        
        # Plot scatter points
        for i, label in enumerate(labels):
            ax.scatter(latencies[i], recalls[i], s=sizes[i], c=[colors[i]], 
                      alpha=0.7, edgecolors='white', linewidth=1.5, 
                      label=f'{label} ({params[i]:.1f}M)', zorder=3)
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        # Labels and title
        ax.set_xlabel('Inference Latency (ms)')
        ax.set_ylabel('Recall@10')
        
        dataset_display = 'Beauty' if dataset == 'beauty' else 'CDs'
        ax.set_title(f'{dataset_display} Dataset', fontweight='normal')
        
        # Legend - place outside plot area to avoid occlusion
        ax.legend(loc='upper left', frameon=True, framealpha=0.95, 
                 edgecolor='#cccccc', fancybox=False, fontsize=9)
        
        # Add annotation for Pareto frontier concept
        # Find the best model (highest recall)
        best_idx = np.argmax(recalls)
        ax.annotate('', xy=(latencies[best_idx], recalls[best_idx]),
                   xytext=(latencies[best_idx] + 5, recalls[best_idx] - 0.003),
                   arrowprops=dict(arrowstyle='->', lw=1, color='#666666'))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', 
               bbox_inches='tight', pad_inches=0.05, dpi=300)
    print(f"Scatter plot saved to {output_path}")
    plt.close()


def plot_grouped_bars_clean(all_results, output_path):
    """Create clean grouped bar chart without hatching.
    
    Publication-quality visualization for top-tier conferences (KDD, NeurIPS, etc.).
    Only shows parameters and latency (2 subplots) with large, readable fonts.
    """
    # Custom style for maximum readability in papers
    libertine_font_dir = '/home/fangdengzhao/Fonts/libertine/opentype'
    libertine_fonts = [
        f'{libertine_font_dir}/LinLibertine_R.otf',
        f'{libertine_font_dir}/LinLibertine_RI.otf',
        f'{libertine_font_dir}/LinLibertine_RB.otf',
        f'{libertine_font_dir}/LinLibertine_RBI.otf',
    ]
    
    for font_file in libertine_fonts:
        if Path(font_file).exists():
            fm.fontManager.addfont(font_file)
    
    # Large fonts for paper readability - unified font size
    plt.rcParams.update({
        'font.family': 'Linux Libertine O',
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 20,  # Same as other text
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
    
    # Color palette inspired by the reference figure
    # Soft, muted colors similar to the uploaded example
    model_colors = {
        'TIGER': '#8FA8C8',      # Muted blue-gray
        'LETTER': '#C89090',     # Dusty rose
        'ActionPiece': '#90C8B8', # Soft teal
        'EAGER': '#C8A890',      # Warm tan
        'PRISM': '#C89090',      # Coral pink (similar to reference)
    }
    
    display_names = {
        'TIGER': 'TIGER',
        'LETTER': 'LETTER',
        'ActionPiece': 'ActionPiece',
        'EAGER': 'EAGER',
        'Prism': 'PRISM',
    }
    
    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    
    # Model order
    model_order = ['TIGER', 'LETTER', 'EAGER', 'ActionPiece', 'Prism']
    models = [m for m in model_order if any(m in all_results[d] for d in datasets)]
    n_models = len(models)
    
    # Create figure with 2 subplots (params and latency only)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    x = np.arange(n_models)
    bar_width = 0.25 if n_datasets == 2 else 0.20  # Thinner bars like reference
    
    # Dataset styles - use distinct colors AND hatch patterns
    # Each dataset has its own color + pattern combination
    # Beauty: light blue with xx pattern
    # CDs: light coral with oo pattern
    dataset_styles = [
        {'color': '#A8C5E3', 'alpha': 0.9, 'edgecolor': '#333333', 'linewidth': 0.8, 'hatch': 'xx'},
        {'color': '#F5B7B1', 'alpha': 0.9, 'edgecolor': '#333333', 'linewidth': 0.8, 'hatch': 'oo'},
    ] if n_datasets == 2 else [
        {'color': '#A8C5E3', 'alpha': 0.9, 'edgecolor': '#333333', 'linewidth': 0.8, 'hatch': 'xx'},
        {'color': '#F5B7B1', 'alpha': 0.9, 'edgecolor': '#333333', 'linewidth': 0.8, 'hatch': 'oo'},
        {'color': '#A9DFBF', 'alpha': 0.9, 'edgecolor': '#333333', 'linewidth': 0.8, 'hatch': '++'},
    ]
    dataset_labels = {'beauty': 'Beauty', 'cds': 'CDs'}
    
    # Subplot 1: Activated Parameters
    ax1 = axes[0]
    for i, dataset in enumerate(datasets):
        results = {k: v for k, v in all_results[dataset].items() if k != 'SASRec'}
        params = [results[m]['activated_params'] / 1e6 if m in results else 0 for m in models]
        
        offset = (i - (n_datasets - 1) / 2) * bar_width
        # Use dataset color for all bars in this dataset
        colors = [dataset_styles[i]['color']] * len(models)
        
        # Create bars with dataset-specific color and pattern
        bars = ax1.bar(x + offset, params, bar_width, 
                      color=colors, alpha=dataset_styles[i]['alpha'],
                      edgecolor=dataset_styles[i]['edgecolor'], 
                      linewidth=dataset_styles[i]['linewidth'],
                      hatch=dataset_styles[i]['hatch'],
                      zorder=3)
    
    # Move y-axis label to top like reference figure
    ax1.text(0.5, 1.08, 'Activated Parameters (M)', 
            transform=ax1.transAxes, ha='center', va='bottom', fontsize=24)
    ax1.set_xticks(x)
    ax1.set_xticklabels([display_names.get(m, m) for m in models], rotation=25, ha='right')
    ax1.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='#CCCCCC', zorder=0)
    ax1.set_axisbelow(True)
    
    # Add clean white background for professional appearance
    ax1.set_facecolor('#FFFFFF')
    
    # Add legend to the FIRST subplot with custom patches showing dataset distinction
    # Use dataset-specific colors with their patterns
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=dataset_styles[i]['color'], alpha=dataset_styles[i]['alpha'],
              edgecolor=dataset_styles[i]['edgecolor'], 
              linewidth=dataset_styles[i]['linewidth'],
              hatch=dataset_styles[i]['hatch'],
              label=dataset_labels[datasets[i]])
        for i in range(n_datasets)
    ]
    ax1.legend(handles=legend_elements, loc='upper left', frameon=True, 
              framealpha=0.98, edgecolor='#CCCCCC', fancybox=False,
              borderpad=0.7, labelspacing=0.6)
    
    # Subplot 2: Inference Latency
    ax2 = axes[1]
    for i, dataset in enumerate(datasets):
        results = {k: v for k, v in all_results[dataset].items() if k != 'SASRec'}
        latencies = [results[m]['timing']['mean_ms'] if m in results else 0 for m in models]
        
        offset = (i - (n_datasets - 1) / 2) * bar_width
        # Use dataset color for all bars in this dataset
        colors = [dataset_styles[i]['color']] * len(models)
        
        # Create bars with dataset-specific color and pattern
        bars = ax2.bar(x + offset, latencies, bar_width,
                      color=colors, alpha=dataset_styles[i]['alpha'],
                      edgecolor=dataset_styles[i]['edgecolor'],
                      linewidth=dataset_styles[i]['linewidth'],
                      hatch=dataset_styles[i]['hatch'],
                      zorder=3)
    
    # Move y-axis label to top like reference figure
    ax2.text(0.5, 1.08, 'Inference Latency (ms)', 
            transform=ax2.transAxes, ha='center', va='bottom', fontsize=24)
    ax2.set_xticks(x)
    ax2.set_xticklabels([display_names.get(m, m) for m in models], rotation=25, ha='right')
    ax2.grid(True, axis='y', linestyle='-', alpha=0.2, linewidth=0.5, color='#CCCCCC', zorder=0)
    ax2.set_axisbelow(True)
    
    # Add clean white background for professional appearance
    ax2.set_facecolor('#FFFFFF')
    
    plt.tight_layout(pad=1.5)
    
    # Save with high quality
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.08)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png',
               bbox_inches='tight', pad_inches=0.08, dpi=300)
    print(f"Grouped bar chart saved to {output_path}")
    plt.close()


def plot_radar_chart(all_results, output_path):
    """Create radar chart for multi-dimensional comparison.
    
    Shows normalized metrics across all dimensions for easy comparison.
    Popular in ML papers for showing model trade-offs.
    """
    setup_publication_style()
    
    model_colors = get_model_colors()
    display_names = get_display_names()
    
    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6),
                            subplot_kw=dict(projection='polar'))
    if n_datasets == 1:
        axes = [axes]
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        results = all_results[dataset]
        recall_data = RECALL_AT_10[dataset]
        
        # Metrics: Recall@10 (higher better), 1/Latency (higher better), 1/Params (higher better)
        categories = ['Recall@10\n(↑)', 'Speed\n(↑)', 'Efficiency\n(↑)']
        n_cats = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        
        # Plot each model
        for model_name in ['TIGER', 'LETTER', 'EAGER', 'ActionPiece', 'Prism']:
            if model_name not in results:
                continue
            
            recall = recall_data[model_name]
            latency = results[model_name]['timing']['mean_ms']
            params = results[model_name]['activated_params'] / 1e6
            
            # Normalize metrics to [0, 1] range
            # For recall: use as-is (already small values)
            # For speed: inverse of latency, normalized
            # For efficiency: inverse of params, normalized
            
            # Collect all values for normalization
            all_recalls = [recall_data[m] for m in ['TIGER', 'LETTER', 'EAGER', 'ActionPiece', 'Prism'] if m in results]
            all_latencies = [results[m]['timing']['mean_ms'] for m in ['TIGER', 'LETTER', 'EAGER', 'ActionPiece', 'Prism'] if m in results]
            all_params = [results[m]['activated_params'] / 1e6 for m in ['TIGER', 'LETTER', 'EAGER', 'ActionPiece', 'Prism'] if m in results]
            
            # Normalize
            recall_norm = (recall - min(all_recalls)) / (max(all_recalls) - min(all_recalls))
            speed_norm = (max(all_latencies) - latency) / (max(all_latencies) - min(all_latencies))
            efficiency_norm = (max(all_params) - params) / (max(all_params) - min(all_params))
            
            values = [recall_norm, speed_norm, efficiency_norm]
            values += values[:1]  # Complete the circle
            
            color = model_colors[display_names.get(model_name, model_name)]
            ax.plot(angles, values, 'o-', linewidth=2, color=color,
                   label=display_names.get(model_name, model_name))
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        dataset_display = 'Beauty' if dataset == 'beauty' else 'CDs'
        ax.set_title(f'{dataset_display} Dataset', fontweight='normal', pad=20)
        
        # Legend below the plot
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0),
                 frameon=True, framealpha=0.95, edgecolor='#cccccc', fancybox=False)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png',
               bbox_inches='tight', pad_inches=0.1, dpi=300)
    print(f"Radar chart saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create publication-quality efficiency comparison plots"
    )
    parser.add_argument('--results_path', type=str,
                       default='scripts/output/efficiency_benchmark/efficiency_results.json',
                       help='Path to efficiency results JSON')
    parser.add_argument('--output_dir', type=str,
                       default='scripts/output/efficiency_benchmark',
                       help='Output directory for plots')
    parser.add_argument('--plot_type', type=str, default='all',
                       choices=['all', 'scatter', 'bars', 'radar'],
                       help='Type of plot to generate')
    
    args = parser.parse_args()
    
    # Load data
    results_path = Path(args.results_path)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        print("Run efficiency benchmark first to generate results.")
        return
    
    print(f"Loading results from {results_path}")
    all_results = load_efficiency_data(results_path)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if args.plot_type in ['all', 'scatter']:
        print("\nGenerating scatter plot (efficiency vs performance)...")
        scatter_path = output_dir / 'efficiency_scatter.pdf'
        plot_scatter_comparison(all_results, str(scatter_path))
    
    if args.plot_type in ['all', 'bars']:
        print("\nGenerating grouped bar chart...")
        bars_path = output_dir / 'efficiency_bars_clean.pdf'
        plot_grouped_bars_clean(all_results, str(bars_path))
    
    if args.plot_type in ['all', 'radar']:
        print("\nGenerating radar chart...")
        radar_path = output_dir / 'efficiency_radar.pdf'
        plot_radar_chart(all_results, str(radar_path))
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nRecommended for paper:")
    print("  - Scatter plot: Shows Pareto frontier (efficiency vs quality)")
    print("  - Grouped bars: Clean comparison across all metrics")
    print("  - Radar chart: Multi-dimensional trade-off visualization")


if __name__ == "__main__":
    main()
