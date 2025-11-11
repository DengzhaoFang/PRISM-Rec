#!/usr/bin/env python3
import os
import json
import argparse
import logging
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def setup_logger():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger("visualize")


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_results(output_dir: str) -> Dict[str, Any]:
    results_path = os.path.join(output_dir, 'training_results.json')
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_training_metrics(output_dir: str, plot_dir: str, results: Dict[str, Any]):
    logger = setup_logger()
    stats: List[Dict[str, Any]] = results.get('training_stats', [])
    if not stats:
        logger.warning("No training_stats found in results; skip training metric plots")
        return

    epochs = [s.get('epoch', i) + 1 for i, s in enumerate(stats)]

    def arr(key: str):
        return [s.get(key, np.nan) for s in stats]

    # Loss curves (train + optional eval)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, arr('total_loss'), label='train_total')
    plt.plot(epochs, arr('recon_loss'), label='train_recon')
    plt.plot(epochs, arr('vq_loss'), label='train_vq')
    plt.plot(epochs, arr('codebook_loss'), label='train_codebook')
    plt.plot(epochs, arr('commitment_loss'), label='train_commitment')
    if any('eval_loss' in s for s in stats):
        plt.plot(epochs, arr('eval_loss'), '--', label='eval_total')
        plt.plot(epochs, arr('eval_recon_loss'), '--', label='eval_recon')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'loss_curves.png'))
    plt.close()

    # Temperature / LR
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, arr('temperature'))
    plt.title('Temperature'); plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, arr('lr'))
    plt.title('Learning Rate'); plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'temp_lr.png'))
    plt.close()

    # Duplicate rate pre
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, arr('duplicate_rate_pre'))
    plt.title('Duplicate Rate (Pre)'); plt.xlabel('Epoch'); plt.ylabel('Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'dup_rate_pre.png'))
    plt.close()

    # Per-layer used codes / perplexity (infer layer indices by keys)
    layer_idxs = sorted({int(k.split('_')[1])
                         for s in stats for k in s.keys()
                         if k.startswith('layer_') and k.endswith('_used_codes')})
    for i in layer_idxs:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, arr(f'layer_{i}_used_codes'))
        plt.title(f'Layer {i} Used Codes'); plt.xlabel('Epoch'); plt.ylabel('Count')
        plt.subplot(1, 2, 2)
        plt.plot(epochs, arr(f'layer_{i}_perplexity'))
        plt.title(f'Layer {i} Perplexity'); plt.xlabel('Epoch'); plt.ylabel('Perplexity')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'layer_{i}_usage_ppl.png'))
        plt.close()

    logger.info(f"Training metric plots saved to {plot_dir}")


def plot_tsne(output_dir: str, data_path: str, embedding_file: str, plot_dir: str, tsne_sample: int = 2000):
    logger = setup_logger()
    # Load embeddings and align with semantic id mappings for color
    emb_path = os.path.join(data_path, embedding_file)
    df = pd.read_parquet(emb_path)
    if 'embedding' not in df.columns:
        logger.warning("embedding column not found; skip t-SNE")
        return
    if 'ItemID' not in df.columns:
        logger.warning("ItemID column not found; cannot color by semantic IDs; skip t-SNE")
        return
    mappings_path = os.path.join(output_dir, 'semantic_id_mappings.json')
    if not os.path.exists(mappings_path):
        logger.warning(f"semantic_id_mappings.json not found at {mappings_path}; skip t-SNE")
        return
    with open(mappings_path, 'r') as f:
        item_to_codes = json.load(f)
    item_to_codes = {int(k): v for k, v in item_to_codes.items()}

    # Filter rows that appear in mappings and sample
    df = df[df['ItemID'].astype(int).isin(item_to_codes.keys())].head(tsne_sample)
    if df.empty:
        logger.warning("No overlapping items between embeddings and mappings; skip t-SNE")
        return
    X = np.stack(df['embedding'].to_numpy()).astype(np.float32)
    c0 = np.array([item_to_codes[int(i)][0] if len(item_to_codes[int(i)]) > 0 else 0 for i in df['ItemID'].tolist()])

    tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=30)
    X2 = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=c0, cmap='tab20', s=6, alpha=0.85)
    plt.title('t-SNE of Item Embeddings (color: Layer-0 code)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'tsne_layer0.png'), dpi=200)
    plt.close()

    logger.info(f"t-SNE saved to {plot_dir}")


def plot_prefix_distribution(output_dir: str, plot_dir: str):
    logger = setup_logger()
    mappings_path = os.path.join(output_dir, 'semantic_id_mappings.json')
    if not os.path.exists(mappings_path):
        logger.warning("semantic_id_mappings.json not found; skip prefix distribution")
        return

    with open(mappings_path, 'r') as f:
        item_to_codes = json.load(f)
    item_to_codes = {int(k): v for k, v in item_to_codes.items()}
    if not item_to_codes:
        logger.warning("Empty semantic_id_mappings; skip prefix distribution")
        return

    n_hier = len(next(iter(item_to_codes.values())))
    for level in range(1, n_hier + 1):
        prefix_groups = {}
        for _, codes in item_to_codes.items():
            prefix = tuple(codes[:level])
            prefix_groups[prefix] = prefix_groups.get(prefix, 0) + 1
        group_sizes = list(prefix_groups.values())
        if not group_sizes:
            continue
        plt.figure(figsize=(8, 4))
        plt.hist(group_sizes, bins=50, log=True)
        plt.title(f'Prefix Co-occurrence Distribution (first {level} codes)')
        plt.xlabel('Group size (#items sharing prefix)'); plt.ylabel('Count (log)')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'prefix_dist_L{level}.png'), dpi=200)
        plt.close()

    logger.info(f"Prefix distribution plots saved to {plot_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualization for tokenizer training')
    parser.add_argument('--output_dir', type=str, required=True, help='Training output dir containing results & mappings')
    parser.add_argument('--data_path', type=str, required=True, help='Data path containing embeddings parquet')
    parser.add_argument('--embedding_file', type=str, default='item_emb_all.parquet', help='Embedding parquet filename')
    parser.add_argument('--plot_output_dir', type=str, default=None, help='Directory to save plots (default: <output_dir>/plots)')
    parser.add_argument('--tsne_sample', type=int, default=2000, help='Max items for t-SNE')

    args = parser.parse_args()
    logger = setup_logger()

    plot_dir = args.plot_output_dir or os.path.join(args.output_dir, 'plots')
    ensure_dir(plot_dir)

    results = load_results(args.output_dir)

    plot_training_metrics(args.output_dir, plot_dir, results)
    plot_tsne(args.output_dir, args.data_path, args.embedding_file, plot_dir, tsne_sample=args.tsne_sample)
    plot_prefix_distribution(args.output_dir, plot_dir)

    logger.info(f"All plots saved to {plot_dir}")


if __name__ == '__main__':
    main()
