"""
Long-tail evaluation for generative recommender models (TIGER & Prism).
"""

import argparse
import logging
import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

# Beauty dataset paths (hardcoded for simplicity)
# IMPORTANT: Use unified item_emb_path for fair popularity grouping comparison
UNIFIED_ITEM_EMB_PATH = "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty/item_emb.parquet"

BEAUTY_PATHS = {
    'tiger': {
        'sequence_data_path': "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty",
        'semantic_mapping_path': "scripts/output/tiger_tokenizer/beauty/3-256-32-ema-only-5-core-items/semantic_id_mappings.json",
        'item_emb_path': UNIFIED_ITEM_EMB_PATH,  # Use unified path
    },
    'prism': {
        'sequence_data_path': "dataset/Amazon-Beauty/processed/beauty-prism-sentenceT5base/Beauty",
        'semantic_mapping_path': "scripts/output/prism_tokenizer/beauty/3-256-32-ema-only-5-core-items/semantic_id_mappings.json",
        'item_emb_path': UNIFIED_ITEM_EMB_PATH,  # Use unified path for fair comparison
    }
}


class PopularityGrouper:
    """Groups items by popularity score into buckets using rank-based assignment."""
    
    def __init__(
        self, 
        item_emb_path: str, 
        num_groups: int = 5,
        group_method: str = 'rank'
    ):
        """Initialize popularity grouper.
        
        Args:
            item_emb_path: Path to item_emb.parquet with popularity_score column
            num_groups: Number of popularity groups (default: 5)
            group_method: 'rank' for rank-based equal-sized groups (recommended),
                         'quantile' for score-based quantiles
        """
        self.num_groups = num_groups
        self.group_method = group_method
        
        logger.info(f"Loading item popularity from {item_emb_path}")
        df = pd.read_parquet(item_emb_path)
        
        if 'popularity_score' not in df.columns:
            raise ValueError(f"popularity_score column not found in {item_emb_path}")
        
        self.item_popularity = dict(zip(df['ItemID'], df['popularity_score']))
        
        # Use rank-based grouping for better distribution
        self._compute_groups_by_rank(df)
        
        self._log_group_stats()
    
    def _compute_groups_by_rank(self, df: pd.DataFrame):
        """Compute groups based on popularity rank (ensures equal-sized groups)."""
        # Sort by popularity score descending (higher score = more popular = lower rank)
        df_sorted = df.sort_values('popularity_score', ascending=False).reset_index(drop=True)
        
        n_items = len(df_sorted)
        items_per_group = n_items // self.num_groups
        
        self.item_to_group = {}
        
        for idx, row in df_sorted.iterrows():
            item_id = row['ItemID']
            # Group 0 = most popular (top ranks), Group N-1 = least popular (bottom ranks)
            group = min(idx // items_per_group, self.num_groups - 1)
            self.item_to_group[item_id] = group
        
        # Store thresholds for logging (approximate score boundaries)
        self.thresholds = []
        for g in range(self.num_groups):
            group_items = df_sorted.iloc[g * items_per_group : (g + 1) * items_per_group]
            if len(group_items) > 0:
                self.thresholds.append((group_items['popularity_score'].min(), 
                                       group_items['popularity_score'].max()))
        
        logger.info(f"Rank-based grouping: ~{items_per_group} items per group")
    
    def _log_group_stats(self):
        """Log statistics for each group."""
        group_counts = [0] * self.num_groups
        group_scores = [[] for _ in range(self.num_groups)]
        
        for item_id, group in self.item_to_group.items():
            group_counts[group] += 1
            group_scores[group].append(self.item_popularity[item_id])
        
        logger.info("\nPopularity Group Statistics:")
        group_names = self.get_group_names()
        for i in range(self.num_groups):
            scores = group_scores[i]
            if scores:
                logger.info(
                    f"  {group_names[i]}: {group_counts[i]} items, "
                    f"score range [{min(scores):.3f}, {max(scores):.3f}]"
                )
            else:
                logger.info(f"  {group_names[i]}: 0 items")
    
    def get_group(self, item_id: int) -> int:
        """Get group index for an item."""
        return self.item_to_group.get(item_id, self.num_groups - 1)
    
    def get_group_names(self) -> List[str]:
        """Get human-readable group names."""
        if self.num_groups == 5:
            return ['Popular', 'Mid-High', 'Medium', 'Mid-Low', 'Long-tail']
        elif self.num_groups == 3:
            return ['Popular', 'Medium', 'Long-tail']
        else:
            return [f'Group {i+1}' for i in range(self.num_groups)]


def pad_or_truncate(sequence: List[int], max_len: int, pad_token_id: int = 0) -> List[int]:
    """Pad or truncate a sequence to a specified maximum length."""
    if len(sequence) > max_len:
        return sequence[-max_len:]
    else:
        return [pad_token_id] * (max_len - len(sequence)) + sequence


class LongTailEvaluator:
    """Evaluates model performance across popularity groups."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        semantic_mapper,
        popularity_grouper: PopularityGrouper,
        device: str = 'cuda',
        beam_size: int = 30,
        topk_list: List[int] = None,
        # Multimodal support for Prism
        content_embeddings: Optional[Dict[int, np.ndarray]] = None,
        collab_embeddings: Optional[Dict[int, np.ndarray]] = None,
        codebook_vectors: Optional[Dict[int, np.ndarray]] = None,
        use_multimodal: bool = False,
        num_code_layers: int = 3,
        # Trie-constrained decoding
        use_trie_constraints: bool = False,
        pad_token_id: int = 0,
        eos_token_id: int = 1
    ):
        self.model = model
        self.semantic_mapper = semantic_mapper
        self.popularity_grouper = popularity_grouper
        self.device = torch.device(device)
        self.beam_size = beam_size
        self.topk_list = topk_list or [5, 10, 20]
        
        # Multimodal embeddings
        self.content_embeddings = content_embeddings or {}
        self.collab_embeddings = collab_embeddings or {}
        self.codebook_vectors = codebook_vectors or {}
        self.use_multimodal = use_multimodal
        self.num_code_layers = num_code_layers
        
        # Trie-constrained decoding
        self.use_trie_constraints = use_trie_constraints
        self.trie_logits_processor = None
        
        if use_trie_constraints:
            from src.recommender.prism.trie_constrained_decoder import (
                SemanticIDTrie, TrieConstrainedLogitsProcessor
            )
            logger.info("Building Trie for constrained decoding...")
            self.trie = SemanticIDTrie.from_semantic_mapper(semantic_mapper)
            self.trie_logits_processor = TrieConstrainedLogitsProcessor(
                trie=self.trie,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                num_beams=beam_size
            )
            logger.info("Trie-constrained decoding enabled")
        
        # Determine embedding dimensions
        if self.content_embeddings:
            self.content_dim = next(iter(self.content_embeddings.values())).shape[0]
        else:
            self.content_dim = 768
        
        if self.collab_embeddings:
            self.collab_dim = next(iter(self.collab_embeddings.values())).shape[0]
        else:
            self.collab_dim = 64
        
        self.model.to(self.device)
        self.model.eval()
    
    def _get_multimodal_inputs(self, history_padded: List[int], max_len: int):
        """Get multimodal embeddings for history items."""
        content_embs = []
        collab_embs = []
        codebook_vecs = []
        
        for item_id in history_padded:
            # Content embedding
            if item_id in self.content_embeddings:
                content_embs.append(self.content_embeddings[item_id])
            else:
                content_embs.append(np.zeros(self.content_dim, dtype=np.float32))
            
            # Collab embedding
            if item_id in self.collab_embeddings:
                collab_embs.append(self.collab_embeddings[item_id])
            else:
                collab_embs.append(np.zeros(self.collab_dim, dtype=np.float32))
            
            # Codebook vectors
            if item_id in self.codebook_vectors:
                codebook_vecs.append(self.codebook_vectors[item_id])
            else:
                codebook_vecs.append(np.zeros((self.num_code_layers, 32), dtype=np.float32))
        
        content_tensor = torch.tensor(np.array(content_embs), dtype=torch.float32, device=self.device).unsqueeze(0)
        collab_tensor = torch.tensor(np.array(collab_embs), dtype=torch.float32, device=self.device).unsqueeze(0)
        codebook_tensor = torch.tensor(np.array(codebook_vecs), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        return content_tensor, collab_tensor, codebook_tensor
    
    def evaluate(self, test_data_path: str, max_len: int = 20) -> Dict:
        """Evaluate model on test data with popularity grouping."""
        df = pd.read_parquet(test_data_path)
        logger.info(f"Loaded {len(df)} test samples")
        
        num_groups = self.popularity_grouper.num_groups
        
        group_metrics = {
            g: {f'Recall@{k}': [] for k in self.topk_list} | {f'NDCG@{k}': [] for k in self.topk_list}
            for g in range(num_groups)
        }
        group_counts = [0] * num_groups
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            history = list(row['history'])
            target_item = row['target']
            
            group = self.popularity_grouper.get_group(target_item)
            group_counts[group] += 1
            
            history_padded = pad_or_truncate(history, max_len, 0)
            history_codes = []
            for item_id in history_padded:
                codes = self.semantic_mapper.get_codes(item_id)
                history_codes.extend(codes)
            
            target_codes = self.semantic_mapper.get_codes(target_item)
            
            input_ids = torch.tensor([history_codes], dtype=torch.long, device=self.device)
            attention_mask = (input_ids != 0).long()
            
            with torch.no_grad():
                max_gen_length = self.semantic_mapper.num_layers + 1
                
                if self.use_multimodal:
                    # Get multimodal embeddings
                    content_embs, collab_embs, codebook_vecs = self._get_multimodal_inputs(
                        history_padded, max_len
                    )
                    preds = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_beams=self.beam_size,
                        max_length=max_gen_length,
                        content_embs=content_embs,
                        collab_embs=collab_embs,
                        history_codebook_vecs=codebook_vecs,
                        logits_processor=self.trie_logits_processor
                    )
                else:
                    preds = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_beams=self.beam_size,
                        max_length=max_gen_length,
                        logits_processor=self.trie_logits_processor
                    )
            
            preds = preds[:, 1:].cpu()
            target_tensor = torch.tensor([target_codes])
            
            pos_index = self._calculate_pos_index(preds, target_tensor)
            
            for k in self.topk_list:
                recall = self._recall_at_k(pos_index, k)
                ndcg = self._ndcg_at_k(pos_index, k)
                group_metrics[group][f'Recall@{k}'].append(recall)
                group_metrics[group][f'NDCG@{k}'].append(ndcg)
        
        results = {'per_group': {}, 'overall': {}, 'group_counts': group_counts}
        group_names = self.popularity_grouper.get_group_names()
        
        overall_metrics = {f'Recall@{k}': [] for k in self.topk_list} | {f'NDCG@{k}': [] for k in self.topk_list}
        
        for g in range(num_groups):
            results['per_group'][group_names[g]] = {}
            for metric_name, values in group_metrics[g].items():
                if values:
                    avg_value = np.mean(values)
                    results['per_group'][group_names[g]][metric_name] = avg_value
                    overall_metrics[metric_name].extend(values)
                else:
                    results['per_group'][group_names[g]][metric_name] = 0.0
        
        for metric_name, values in overall_metrics.items():
            results['overall'][metric_name] = np.mean(values) if values else 0.0
        
        return results
    
    def _calculate_pos_index(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = labels.shape[0]
        maxk = preds.shape[0]
        pos_index = torch.zeros((batch_size, maxk), dtype=torch.bool)
        
        for i in range(batch_size):
            cur_label = labels[i].tolist()
            for j in range(maxk):
                cur_pred = preds[j].tolist()
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        
        return pos_index
    
    def _recall_at_k(self, pos_index: torch.Tensor, k: int) -> float:
        return pos_index[:, :k].sum().float().item()
    
    def _ndcg_at_k(self, pos_index: torch.Tensor, k: int) -> float:
        ranks = torch.arange(1, pos_index.shape[-1] + 1, dtype=torch.float32)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, torch.tensor(0.0))
        return dcg[:, :k].sum().item()



def plot_longtail_results(
    results: Dict,
    output_path: str,
    model_name: str = 'Model',
    metrics_to_plot: List[str] = None
):
    """Plot long-tail evaluation results - single figure with Recall and NDCG combined."""
    metrics_to_plot = metrics_to_plot or ['Recall@10', 'NDCG@10']
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })
    
    group_names = list(results['per_group'].keys())
    num_groups = len(group_names)
    num_metrics = len(metrics_to_plot)
    
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))
    
    x = np.arange(num_groups)
    bar_width = 0.35
    
    # Colors for different metrics
    metric_colors = ['#4E79A7', '#F28E2B']
    
    for m_idx, metric in enumerate(metrics_to_plot):
        values = [results['per_group'][g].get(metric, 0) for g in group_names]
        offset = (m_idx - 0.5) * bar_width
        
        bars = ax.bar(x + offset, values, bar_width, label=metric,
                     color=metric_colors[m_idx % len(metric_colors)],
                     edgecolor='#333333', linewidth=0.5, zorder=3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 1),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=6.5,
                       color='#333333')
    
    ax.set_xlabel('Popularity Group')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    
    short_names = ['Pop.', 'Mid-H', 'Med.', 'Mid-L', 'Tail']
    if num_groups == 5:
        ax.set_xticklabels(short_names, rotation=0)
    else:
        ax.set_xticklabels(group_names, rotation=20, ha='right')
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.2, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False,
             edgecolor='#cccccc', framealpha=0.95)
    
    ax.set_title(f'{model_name} Performance by Popularity Group', fontsize=11)
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', 
                pad_inches=0.02, dpi=300)
    logger.info(f"Figure saved to {output_path}")
    plt.close()


def plot_comparison_results(
    all_results: Dict[str, Dict],
    output_path: str,
    metrics_to_plot: List[str] = None
):
    """Plot comparison of multiple models - single figure with grouped bars for Recall and NDCG."""
    metrics_to_plot = metrics_to_plot or ['Recall@10', 'NDCG@10']
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })
    
    model_names = list(all_results.keys())
    first_result = list(all_results.values())[0]
    group_names = list(first_result['per_group'].keys())
    num_groups = len(group_names)
    num_models = len(model_names)
    num_metrics = len(metrics_to_plot)
    
    # Single figure with all metrics combined
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2))
    
    # Create grouped bars: for each group, show model1_metric1, model1_metric2, model2_metric1, model2_metric2
    total_bars_per_group = num_models * num_metrics
    bar_width = 0.8 / total_bars_per_group
    x = np.arange(num_groups)
    
    # Color scheme: different colors for models, different shades for metrics
    model_base_colors = {
        'TIGER': ['#4E79A7', '#7BA3C9'],  # Blue shades for Recall, NDCG
        'Prism': ['#F28E2B', '#F5B366'],  # Orange shades
    }
    
    hatches = {'Recall': '', 'NDCG': '//'}
    
    bar_idx = 0
    for model_name in model_names:
        results = all_results[model_name]
        colors = model_base_colors.get(model_name, ['#888888', '#AAAAAA'])
        
        for m_idx, metric in enumerate(metrics_to_plot):
            values = [results['per_group'][g].get(metric, 0) for g in group_names]
            offset = (bar_idx - total_bars_per_group / 2 + 0.5) * bar_width
            
            metric_type = 'Recall' if 'Recall' in metric else 'NDCG'
            label = f'{model_name} {metric}'
            
            ax.bar(x + offset, values, bar_width, label=label,
                  color=colors[m_idx % len(colors)],
                  edgecolor='#333333', linewidth=0.4,
                  hatch=hatches.get(metric_type, ''), zorder=3)
            bar_idx += 1
    
    ax.set_xlabel('Popularity Group')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    
    short_names = ['Pop.', 'Mid-H', 'Med.', 'Mid-L', 'Tail']
    if num_groups == 5:
        ax.set_xticklabels(short_names, rotation=0)
    else:
        ax.set_xticklabels(group_names, rotation=20, ha='right')
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.2, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper right', frameon=True, fancybox=False,
             edgecolor='#cccccc', framealpha=0.95, ncol=2, fontsize=7)
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', 
                pad_inches=0.02, dpi=300)
    logger.info(f"Comparison figure saved to {output_path}")
    plt.close()


def plot_relative_improvement(
    baseline_results: Dict,
    improved_results: Dict,
    output_path: str,
    baseline_name: str = 'TIGER',
    improved_name: str = 'Prism',
    metrics_to_plot: List[str] = None
):
    """Plot relative improvement of improved model over baseline - single combined figure.
    
    This visualization highlights where the improved model excels, especially for long-tail items.
    """
    metrics_to_plot = metrics_to_plot or ['Recall@10', 'NDCG@10']
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    group_names = list(baseline_results['per_group'].keys())
    num_groups = len(group_names)
    num_metrics = len(metrics_to_plot)
    
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.0))
    
    x = np.arange(num_groups)
    bar_width = 0.35
    
    metric_colors = ['#59A14F', '#76B7B2']  # Green shades for improvement
    
    for m_idx, metric in enumerate(metrics_to_plot):
        improvements = []
        for g in group_names:
            baseline_val = baseline_results['per_group'][g].get(metric, 0)
            improved_val = improved_results['per_group'][g].get(metric, 0)
            if baseline_val > 0:
                rel_imp = (improved_val - baseline_val) / baseline_val * 100
            else:
                rel_imp = 0
            improvements.append(rel_imp)
        
        offset = (m_idx - 0.5) * bar_width
        bars = ax.bar(x + offset, improvements, bar_width, label=metric,
                     color=metric_colors[m_idx % len(metric_colors)],
                     edgecolor='#333333', linewidth=0.5, zorder=3)
        
        # Add value labels
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset_y = 1 if height >= 0 else -1
            ax.annotate(f'{val:+.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset_y),
                       textcoords="offset points",
                       ha='center', va=va, fontsize=6.5,
                       color='#333333', fontweight='bold')
    
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.8, zorder=2)
    
    ax.set_xlabel('Popularity Group')
    ax.set_ylabel(f'Relative Improvement (%)\n({improved_name} vs {baseline_name})')
    ax.set_xticks(x)
    
    short_names = ['Pop.', 'Mid-H', 'Med.', 'Mid-L', 'Tail']
    if num_groups == 5:
        ax.set_xticklabels(short_names, rotation=0)
    else:
        ax.set_xticklabels(group_names, rotation=20, ha='right')
    
    ax.yaxis.grid(True, linestyle='-', alpha=0.2, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper left', frameon=True, fancybox=False,
             edgecolor='#cccccc', framealpha=0.95)
    
    plt.tight_layout(pad=0.5)
    
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.02)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', 
                pad_inches=0.02, dpi=300)
    logger.info(f"Relative improvement figure saved to {output_path}")
    plt.close()


def setup_logging(output_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "longtail_eval.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def load_model_and_mapper(model_type: str, checkpoint_path: str, device: str):
    """Load model and semantic mapper based on model type.
    
    Args:
        model_type: 'tiger' or 'prism'
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, semantic_mapper, paths_config)
    """
    paths = BEAUTY_PATHS[model_type]
    
    # Load checkpoint first to get saved config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if model_type == 'tiger':
        from src.recommender.TIGER.dataset import SemanticIDMapper
        from src.recommender.TIGER.model import create_model
        
    elif model_type == 'prism':
        from src.recommender.prism.dataset import SemanticIDMapper
        from src.recommender.prism.model import create_model
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'tiger' or 'prism'")
    
    # Use config from checkpoint (preserves model architecture including vocab_size)
    config = checkpoint['config']
    logger.info(f"Using config from checkpoint: d_model={config['model'].d_model}, "
                f"num_layers={config['model'].num_layers}, vocab_size={config['model'].vocab_size}")
    
    # Load semantic mapper using config from checkpoint
    semantic_mapper = SemanticIDMapper(
        config['data'].semantic_mapping_path,
        codebook_size=config['model'].codebook_size,
        num_layers=config['model'].num_code_layers
    )
    
    # IMPORTANT: Do NOT override vocab_size from checkpoint!
    # The checkpoint's vocab_size is the correct one used during training.
    # Just update num_code_layers if it was auto-detected differently
    if semantic_mapper.num_layers != config['model'].num_code_layers:
        logger.warning(f"num_code_layers mismatch: mapper={semantic_mapper.num_layers}, "
                      f"config={config['model'].num_code_layers}, using config value")
    
    # Create model with training_config for Prism (needed for fusion modules)
    if model_type == 'prism':
        model = create_model(config['model'], config.get('training', None))
    else:
        model = create_model(config['model'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded {model_type.upper()} checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return model, semantic_mapper, paths, config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Long-tail evaluation for TIGER/Prism on Beauty dataset"
    )
    
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['tiger', 'prism'],
                       help='Model type: tiger or prism')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--num_groups', type=int, default=5,
                       help='Number of popularity groups')
    parser.add_argument('--beam_size', type=int, default=30,
                       help='Beam size for generation')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        checkpoint_dir = Path(args.checkpoint).parent
        output_dir = checkpoint_dir / 'longtail_eval'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir)
    logger.info(f"Long-tail evaluation for {args.model_type.upper()} on Beauty dataset")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # Load model and mapper
    model, semantic_mapper, paths, config = load_model_and_mapper(
        args.model_type, args.checkpoint, args.device
    )
    
    # Create popularity grouper
    popularity_grouper = PopularityGrouper(
        paths['item_emb_path'],
        num_groups=args.num_groups
    )
    
    # Load multimodal embeddings for Prism
    content_embeddings = None
    collab_embeddings = None
    codebook_vectors = None
    use_multimodal = False
    use_trie_constraints = False
    
    # Get training config from checkpoint
    training_config = config.get('training', None)
    
    # Check for Trie-constrained decoding (applies to both TIGER and Prism)
    if training_config and hasattr(training_config, 'use_trie_constraints'):
        use_trie_constraints = training_config.use_trie_constraints
    
    if use_trie_constraints:
        logger.info("Trie-constrained decoding enabled (matching training config)")
    
    if args.model_type == 'prism':
        if training_config and hasattr(training_config, 'use_multimodal_fusion') and training_config.use_multimodal_fusion:
            use_multimodal = True
            logger.info("Loading multimodal embeddings for Prism...")
            
            # Import loading functions from dataset.py
            from src.recommender.prism.dataset import (
                load_content_embeddings,
                load_collab_embeddings,
                load_codebook_mappings
            )
            
            # Load content embeddings from item_emb.parquet
            data_dir = paths['sequence_data_path']
            content_embeddings = load_content_embeddings(data_dir)
            logger.info(f"Loaded content embeddings for {len(content_embeddings)} items")
            
            # Load collab embeddings from lightgcn directory
            collab_path = Path(data_dir) / 'lightgcn' / 'item_embeddings_collab.npy'
            if collab_path.exists():
                collab_embeddings = load_collab_embeddings(str(collab_path))
                logger.info(f"Loaded collab embeddings for {len(collab_embeddings)} items")
            else:
                logger.warning(f"Collab embeddings not found at {collab_path}")
            
            # Load codebook vectors from tokenizer output
            tokenizer_dir = Path(config['data'].semantic_mapping_path).parent
            codebook_vectors, _ = load_codebook_mappings(str(tokenizer_dir))
            logger.info(f"Loaded codebook vectors for {len(codebook_vectors)} items")
    
    # Create evaluator with multimodal and Trie support
    evaluator = LongTailEvaluator(
        model=model,
        semantic_mapper=semantic_mapper,
        popularity_grouper=popularity_grouper,
        device=args.device,
        beam_size=args.beam_size,
        content_embeddings=content_embeddings,
        collab_embeddings=collab_embeddings,
        codebook_vectors=codebook_vectors,
        use_multimodal=use_multimodal,
        num_code_layers=config['model'].num_code_layers,
        use_trie_constraints=use_trie_constraints,
        pad_token_id=config['model'].pad_token_id,
        eos_token_id=config['model'].eos_token_id
    )
    
    # Run evaluation
    test_path = Path(paths['sequence_data_path']) / 'test.parquet'
    results = evaluator.evaluate(str(test_path), max_len=config['data'].max_seq_length)
    
    # Save results
    results_path = output_dir / f'longtail_results_{args.model_type}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info(f"LONG-TAIL EVALUATION RESULTS ({args.model_type.upper()})")
    logger.info("=" * 60)
    
    group_names = popularity_grouper.get_group_names()
    for g_name in group_names:
        logger.info(f"\n{g_name} (n={results['group_counts'][group_names.index(g_name)]}):")
        for metric, value in results['per_group'][g_name].items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"\nOverall:")
    for metric, value in results['overall'].items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Plot results
    plot_path = str(output_dir / f'longtail_performance_{args.model_type}.pdf')
    plot_longtail_results(results, plot_path, model_name=args.model_type.upper())
    
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
