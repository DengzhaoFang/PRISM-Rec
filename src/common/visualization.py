#!/usr/bin/env python3
"""
Visualization tools for semantic ID analysis

Provides tools to visualize and analyze semantic ID spaces,
including t-SNE/UMAP embeddings, codebook usage, and hierarchical structure.
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict, Counter


def visualize_semantic_id_space(
    model,
    semantic_id_mappings: Union[Dict, str],
    output_path: str = 'semantic_id_tsne.png',
    method: str = 'tsne',
    color_by: str = 'layer_0',
    max_items: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 8),
    device: str = 'cpu'
):
    """
    Visualize semantic ID space using dimensionality reduction.
    
    Args:
        model: Trained RQ-VAE model with decode_from_codes method
        semantic_id_mappings: Dict or path to semantic_id_mappings.json
        output_path: Output path for visualization
        method: Dimensionality reduction method ('tsne', 'umap', 'pca')
        color_by: How to color points ('layer_0', 'layer_1', 'layer_2', 'random')
        max_items: Maximum number of items to visualize (for performance)
        figsize: Figure size
        device: Device for model inference
    """
    print("="*60)
    print("SEMANTIC ID SPACE VISUALIZATION")
    print("="*60)
    
    # Load mappings
    if isinstance(semantic_id_mappings, str):
        with open(semantic_id_mappings, 'r') as f:
            mappings = json.load(f)
    else:
        mappings = semantic_id_mappings
    
    # Sample items if needed
    if max_items and len(mappings) > max_items:
        import random
        items = list(mappings.items())
        random.shuffle(items)
        mappings = dict(items[:max_items])
        print(f"Sampled {max_items} items from {len(items)} total")
    
    print(f"Visualizing {len(mappings):,} semantic IDs...")
    
    # Collect embeddings
    model.eval()
    model.to(device)
    
    all_embeddings = []
    all_labels = []
    all_codes = []
    
    with torch.no_grad():
        for item_id, codes in mappings.items():
            # Use only first 3 layers (excluding dedup layer if present)
            codes_3layer = codes[:3] if len(codes) >= 3 else codes
            codes_tensor = torch.tensor([codes_3layer], device=device)
            
            # Get embedding by decoding from codes
            if hasattr(model, 'decode_from_codes'):
                emb = model.decode_from_codes(codes_tensor)
            else:
                # Fallback: reconstruct through full pipeline
                print("Warning: model doesn't have decode_from_codes, using fallback")
                emb = model.decoder(model.quantizers[0].embedding(codes_tensor[:, 0]))
            
            all_embeddings.append(emb.cpu().numpy().flatten())
            all_labels.append(str(item_id))
            all_codes.append(codes_3layer)
    
    embeddings_matrix = np.array(all_embeddings)
    codes_matrix = np.array(all_codes)
    
    print(f"Embedding matrix shape: {embeddings_matrix.shape}")
    
    # Apply dimensionality reduction
    print(f"Applying {method.upper()} dimensionality reduction...")
    
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings_2d = reducer.fit_transform(embeddings_matrix)
    
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            embeddings_2d = reducer.fit_transform(embeddings_matrix)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings_matrix)
    
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings_matrix)
        print(f"PCA explained variance: {reducer.explained_variance_ratio_.sum():.3f}")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Determine colors
    if color_by.startswith('layer_'):
        layer_idx = int(color_by.split('_')[1])
        colors = codes_matrix[:, layer_idx]
        color_label = f"Layer {layer_idx} Code"
    elif color_by == 'random':
        colors = np.random.rand(len(embeddings_2d))
        color_label = "Random"
    else:
        colors = np.arange(len(embeddings_2d))
        color_label = "Item Index"
    
    # Create visualization
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=colors,
        cmap='tab20',
        alpha=0.6,
        s=20,
        edgecolors='none'
    )
    
    plt.colorbar(scatter, label=color_label)
    plt.title(f'Semantic ID Space Visualization ({method.upper()})\n{len(mappings):,} items')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()
    
    return embeddings_2d, codes_matrix


def visualize_hierarchical_distribution(
    semantic_id_mappings: Union[Dict, str],
    output_dir: str = '.',
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Visualize the distribution of semantic IDs at each hierarchy level.
    
    Args:
        semantic_id_mappings: Dict or path to semantic_id_mappings.json
        output_dir: Output directory for visualizations
        figsize: Figure size
    """
    print("\nVisualizing hierarchical distributions...")
    
    # Load mappings
    if isinstance(semantic_id_mappings, str):
        with open(semantic_id_mappings, 'r') as f:
            mappings = json.load(f)
    else:
        mappings = semantic_id_mappings
    
    # Extract codes by layer
    codes_by_layer = defaultdict(list)
    n_layers = len(next(iter(mappings.values())))
    
    for item_id, codes in mappings.items():
        for layer_idx in range(min(len(codes), n_layers)):
            codes_by_layer[layer_idx].append(codes[layer_idx])
    
    # Create subplots
    fig, axes = plt.subplots(1, min(n_layers, 4), figsize=figsize)
    if n_layers == 1:
        axes = [axes]
    
    for layer_idx in range(min(n_layers, 4)):
        codes = codes_by_layer[layer_idx]
        counter = Counter(codes)
        
        # Get statistics
        unique_codes = len(counter)
        total_items = len(codes)
        usage_rate = unique_codes / max(codes) if codes else 0
        
        # Plot histogram
        ax = axes[layer_idx]
        values, counts = zip(*sorted(counter.items()))
        
        ax.bar(values, counts, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel(f'Code Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Layer {layer_idx}\n{unique_codes:,} unique / {total_items:,} items')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'hierarchical_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Hierarchical distribution saved to: {output_path}")
    plt.close()


def visualize_codebook_usage(
    model,
    output_path: str = 'codebook_usage.png',
    figsize: Tuple[int, int] = (12, 6),
    device: str = 'cpu'
):
    """
    Visualize codebook usage statistics.
    
    Args:
        model: Trained RQ-VAE model
        output_path: Output path for visualization
        figsize: Figure size
        device: Device for model
    """
    print("\nVisualizing codebook usage...")
    
    model.eval()
    model.to(device)
    
    n_layers = model.n_layers
    n_embed = model.n_embed
    
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    if n_layers == 1:
        axes = [axes]
    
    for layer_idx, quantizer in enumerate(model.quantizers):
        ax = axes[layer_idx]
        
        # Get codebook weights
        if hasattr(quantizer, 'embedding'):
            if isinstance(quantizer.embedding, torch.nn.Embedding):
                codebook_weights = quantizer.embedding.weight.data.cpu().numpy()
            else:
                codebook_weights = quantizer.embedding.cpu().numpy()
        else:
            print(f"Warning: Layer {layer_idx} doesn't have expected embedding structure")
            continue
        
        # Compute norms
        norms = np.linalg.norm(codebook_weights, axis=1)
        
        # Plot
        ax.bar(range(len(norms)), norms, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Codebook Index')
        ax.set_ylabel('L2 Norm')
        ax.set_title(f'Layer {layer_idx} Codebook Usage\n(higher norm = more used)')
        ax.grid(alpha=0.3)
        
        # Add statistics
        mean_norm = norms.mean()
        std_norm = norms.std()
        ax.axhline(mean_norm, color='r', linestyle='--', label=f'Mean: {mean_norm:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Codebook usage saved to: {output_path}")
    plt.close()


def analyze_semantic_id_statistics(
    semantic_id_mappings: Union[Dict, str],
    output_path: str = 'semantic_id_stats.txt'
):
    """
    Generate detailed statistics about semantic ID distribution.
    
    Args:
        semantic_id_mappings: Dict or path to semantic_id_mappings.json
        output_path: Output path for statistics report
    """
    print("\nAnalyzing semantic ID statistics...")
    
    # Load mappings
    if isinstance(semantic_id_mappings, str):
        with open(semantic_id_mappings, 'r') as f:
            mappings = json.load(f)
    else:
        mappings = semantic_id_mappings
    
    n_items = len(mappings)
    n_layers = len(next(iter(mappings.values())))
    
    # Analyze each layer
    stats_lines = []
    stats_lines.append("="*60)
    stats_lines.append("SEMANTIC ID STATISTICS")
    stats_lines.append("="*60)
    stats_lines.append(f"Total items: {n_items:,}")
    stats_lines.append(f"Hierarchy levels: {n_layers}")
    stats_lines.append("")
    
    for level in range(n_layers):
        stats_lines.append(f"Layer {level}:")
        stats_lines.append("-" * 40)
        
        # Extract prefixes
        prefixes = defaultdict(list)
        for item_id, codes in mappings.items():
            prefix = tuple(codes[:level+1])
            prefixes[prefix].append(item_id)
        
        unique_prefixes = len(prefixes)
        duplicate_rate = 1.0 - (unique_prefixes / n_items)
        
        # Group sizes
        group_sizes = [len(items) for items in prefixes.values()]
        max_group = max(group_sizes)
        avg_group = np.mean(group_sizes)
        
        stats_lines.append(f"  Unique prefixes: {unique_prefixes:,} / {n_items:,}")
        stats_lines.append(f"  Duplicate rate: {duplicate_rate:.4f}")
        stats_lines.append(f"  Max items per prefix: {max_group:,}")
        stats_lines.append(f"  Avg items per prefix: {avg_group:.2f}")
        
        # Code distribution
        if level < n_layers:
            codes_at_layer = [codes[level] for codes in mappings.values()]
            unique_codes = len(set(codes_at_layer))
            max_code = max(codes_at_layer)
            min_code = min(codes_at_layer)
            
            stats_lines.append(f"  Unique codes: {unique_codes}")
            stats_lines.append(f"  Code range: [{min_code}, {max_code}]")
        
        stats_lines.append("")
    
    # Write to file
    stats_text = '\n'.join(stats_lines)
    with open(output_path, 'w') as f:
        f.write(stats_text)
    
    print(f"✓ Statistics saved to: {output_path}")
    print("\n" + stats_text)


# Example usage
if __name__ == '__main__':
    # This would be called after training
    print("Semantic ID Visualization Tools")
    print("Import this module and use the visualization functions after training.")
    
    # Example:
    # from src.common.visualization import visualize_semantic_id_space
    # visualize_semantic_id_space(
    #     model=trained_model,
    #     semantic_id_mappings='output/semantic_id_mappings.json',
    #     output_path='semantic_id_tsne.png'
    # )

