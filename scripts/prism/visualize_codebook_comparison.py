#!/usr/bin/env python3
"""
TIGER vs PRISM Codebook t-SNE Comparison Visualization

This script generates a publication-quality figure comparing TIGER and PRISM
codebook embeddings side by side with a shared legend.

Output format: PNG, PDF, SVG (for direct LaTeX insertion)
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Publication-quality settings (NeurIPS/KDD standard)
# Configure Linux Libertine font
import matplotlib.font_manager as fm

# Add Linux Libertine fonts from the opentype directory
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

plt.rcParams.update({
    'font.family': 'Linux Libertine O',
    'font.weight': 'normal',  # Ensure no bold
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Linux Libertine O',
    'mathtext.it': 'Linux Libertine O:italic',
    'mathtext.bf': 'Linux Libertine O:bold',
    'font.size': 12,           # Base font size for publication
    'axes.labelsize': 12,      # Axis labels
    'axes.titlesize': 14,      # Subplot titles
    'xtick.labelsize': 11,     # X-axis tick labels
    'ytick.labelsize': 11,     # Y-axis tick labels
    'legend.fontsize': 7,     # Legend text
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Okabe-Ito colorblind-friendly palette (KDD/NeurIPS standard)
LAYER_COLORS = ['#009E73', '#D55E00', '#0072B2']  # Teal, Vermillion, Blue


def load_codebooks_from_checkpoint(checkpoint_path: str) -> List[np.ndarray]:
    """Load codebook embeddings from model checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        config = {'n_layers': checkpoint.get('n_layers', 3)}
    else:
        raise ValueError("Unknown checkpoint format")
    
    codebooks = []
    n_layers = config.get('n_layers', 3)
    
    for layer_idx in range(n_layers):
        key = f'quantizers.{layer_idx}.embedding'
        if key in state_dict:
            codebook = state_dict[key].numpy()
        elif f'quantizers.{layer_idx}.embedding.weight' in state_dict:
            codebook = state_dict[f'quantizers.{layer_idx}.embedding.weight'].numpy()
        else:
            raise ValueError(f"Cannot find codebook for layer {layer_idx}")
        
        codebooks.append(codebook)
        print(f"  Layer {layer_idx}: shape = {codebook.shape}")
    
    return codebooks


def compute_tsne(embeddings: np.ndarray, perplexity: int = 60, n_iter: int = 3000,
                 random_state: int = 42, init: str = 'pca') -> np.ndarray:
    """Compute t-SNE projection to 2D."""
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=random_state,
        init=init,
        learning_rate='auto'
    )
    return tsne.fit_transform(embeddings)


def prepare_tsne_data(codebooks: List[np.ndarray], perplexity: int = 60, 
                      n_iter: int = 3000, init: str = 'pca') -> Tuple[np.ndarray, np.ndarray]:
    """Prepare t-SNE embeddings and layer labels."""
    all_embeddings = np.concatenate(codebooks, axis=0)
    layer_labels = np.concatenate([
        np.full(cb.shape[0], i) for i, cb in enumerate(codebooks)
    ])
    
    embeddings_2d = compute_tsne(all_embeddings, perplexity=perplexity, 
                                  n_iter=n_iter, init=init)
    
    # Remove outliers (keep 95%)
    x_min, x_max = np.percentile(embeddings_2d[:, 0], [2.5, 97.5])
    y_min, y_max = np.percentile(embeddings_2d[:, 1], [2.5, 97.5])
    
    mask = (
        (embeddings_2d[:, 0] >= x_min) & (embeddings_2d[:, 0] <= x_max) &
        (embeddings_2d[:, 1] >= y_min) & (embeddings_2d[:, 1] <= y_max)
    )
    
    embeddings_2d = embeddings_2d[mask]
    layer_labels = layer_labels[mask]
    
    # Shuffle to prevent z-order occlusion
    indices = np.arange(len(embeddings_2d))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    return embeddings_2d[indices], layer_labels[indices]


def plot_comparison(
    tiger_codebooks: List[np.ndarray],
    prism_codebooks: List[np.ndarray],
    output_path: str,
    perplexity: int = 60,
    n_iter: int = 3000,
    init: str = 'pca',
    dpi: int = 300
):
    """
    Plot TIGER and PRISM codebook comparison in publication-quality format.
    
    Layout: Legend on top, TIGER (left) and PRISM (right) below.
    Style: Clean, borderless scatter plots (NeurIPS/KDD standard)
    """
    print("\nComputing t-SNE for TIGER...")
    tiger_2d, tiger_labels = prepare_tsne_data(tiger_codebooks, perplexity, n_iter, init)
    
    print("Computing t-SNE for PRISM...")
    prism_2d, prism_labels = prepare_tsne_data(prism_codebooks, perplexity, n_iter, init)
    
    n_layers = len(tiger_codebooks)
    
    # Create figure with GridSpec for precise layout control
    # Reduced figure size for better paper integration while maintaining clarity
    fig = plt.figure(figsize=(4.5, 2.2), dpi=dpi)
    fig.patch.set_facecolor('white')
    
    # GridSpec: 2 rows (legend + plots), 2 columns (TIGER + PRISM)
    # Adjusted height_ratios and hspace for more compact layout
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.08, 1], 
                  hspace=0.04, wspace=0.12)
    
    # Top row: shared legend spanning both columns
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')
    
    # Create legend elements with circles (matching scatter points)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=LAYER_COLORS[i],
               markersize=8, label=f'Layer {i}', markeredgecolor='white', markeredgewidth=0.5)
        for i in range(n_layers)
    ]
    
    legend = ax_legend.legend(
        handles=legend_elements,
        loc='center',
        ncol=n_layers,
        frameon=False,
        fontsize=7,
        handletextpad=0.3,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.05)  # Move legend down slightly
    )
    
    # Bottom row: TIGER (left) and PRISM (right)
    ax_tiger = fig.add_subplot(gs[1, 0])
    ax_prism = fig.add_subplot(gs[1, 1])
    
    # Plot function for each subplot
    def plot_scatter(ax, embeddings_2d, layer_labels, title):
        ax.set_facecolor('white')
        
        colors = [LAYER_COLORS[int(label)] for label in layer_labels]
        
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=colors,
            alpha=0.7,
            s=12,
            edgecolors='white',
            linewidth=0.3,
            rasterized=False
        )
        
        # Force square aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Set equal axis limits with padding
        x_range = embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min()
        y_range = embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min()
        max_range = max(x_range, y_range)
        
        x_center = (embeddings_2d[:, 0].max() + embeddings_2d[:, 0].min()) / 2
        y_center = (embeddings_2d[:, 1].max() + embeddings_2d[:, 1].min()) / 2
        margin = max_range * 0.08
        
        ax.set_xlim(x_center - max_range/2 - margin, x_center + max_range/2 + margin)
        ax.set_ylim(y_center - max_range/2 - margin, y_center + max_range/2 + margin)
        
        # Clean style: no border, no ticks (common in top venues)
        ax.axis('off')
        
        # Add title below the plot - moved up for more compact layout
        ax.set_title(title, fontsize=7, fontweight='normal', pad=2, y=-0.06)
    
    # Plot both
    plot_scatter(ax_tiger, tiger_2d, tiger_labels, 'TIGER Codebook')
    plot_scatter(ax_prism, prism_2d, prism_labels, 'PRISM Codebook')
    
    plt.tight_layout()
    
    # Save in multiple formats
    base_path = output_path.rsplit('.', 1)[0]
    
    # PNG
    png_path = f"{base_path}.png"
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    print(f"Saved PNG: {png_path}")
    
    # PDF (vector format for LaTeX)
    pdf_path = f"{base_path}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.02)
    print(f"Saved PDF: {pdf_path}")
    
    # SVG (vector format for editing)
    svg_path = f"{base_path}.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.02)
    print(f"Saved SVG: {svg_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate TIGER vs PRISM codebook comparison visualization'
    )
    parser.add_argument('--tiger_model', type=str, required=True,
                        help='Path to TIGER model checkpoint')
    parser.add_argument('--prism_model', type=str, required=True,
                        help='Path to PRISM model checkpoint')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Output path (without extension, will save .png/.pdf/.svg)')
    parser.add_argument('--perplexity', type=int, default=60,
                        help='t-SNE perplexity (default: 60)')
    parser.add_argument('--n_iter', type=int, default=3000,
                        help='t-SNE iterations (default: 3000)')
    parser.add_argument('--init', type=str, default='pca', choices=['pca', 'random'],
                        help='t-SNE initialization (default: pca)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output DPI (default: 300)')
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("TIGER vs PRISM Codebook Comparison")
    print("=" * 60)
    
    # Load codebooks
    print("\nLoading TIGER codebooks...")
    tiger_codebooks = load_codebooks_from_checkpoint(args.tiger_model)
    
    print("\nLoading PRISM codebooks...")
    prism_codebooks = load_codebooks_from_checkpoint(args.prism_model)
    
    # Generate comparison plot
    print("\nGenerating comparison visualization...")
    plot_comparison(
        tiger_codebooks=tiger_codebooks,
        prism_codebooks=prism_codebooks,
        output_path=args.output_path,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        init=args.init,
        dpi=args.dpi
    )
    
    print("\n" + "=" * 60)
    print("✓ Visualization completed!")
    print("=" * 60)
    print(f"\nOutput files:")
    base = args.output_path.rsplit('.', 1)[0]
    print(f"  - {base}.png (raster, for preview)")
    print(f"  - {base}.pdf (vector, for LaTeX)")
    print(f"  - {base}.svg (vector, for editing)")


if __name__ == '__main__':
    main()
