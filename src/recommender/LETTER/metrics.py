"""
Evaluation metrics for generative recommendation.

Implements Recall@K and NDCG@K metrics.
"""

import torch
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def calculate_pos_index(
    preds: torch.Tensor,
    labels: torch.Tensor,
    maxk: int = 20
) -> torch.Tensor:
    """Calculate position indices of ground truth items in predictions.
    
    This function checks if the predicted sequences match the ground truth
    labels and returns a boolean tensor indicating matches at each beam position.
    
    Args:
        preds: Predicted token sequences, shape (batch_size, maxk, seq_len)
               where maxk is the beam size
        labels: Ground truth token sequences, shape (batch_size, seq_len)
        maxk: Maximum k value (beam size)
    
    Returns:
        Boolean tensor of shape (batch_size, maxk) indicating whether the
        prediction at each position matches the ground truth
    """
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    
    assert preds.shape[1] == maxk, f"preds.shape[1] = {preds.shape[1]} != {maxk}"
    
    batch_size = preds.shape[0]
    pos_index = torch.zeros((batch_size, maxk), dtype=torch.bool)
    
    for i in range(batch_size):
        cur_label = labels[i].tolist()
        for j in range(maxk):
            cur_pred = preds[i, j].tolist()
            if cur_pred == cur_label:
                pos_index[i, j] = True
                break  # Only mark the first match
    
    return pos_index


def recall_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    """Calculate Recall@K.
    
    Recall@K measures the proportion of test cases where the ground truth
    item appears in the top-k predictions.
    
    Args:
        pos_index: Boolean tensor of shape (batch_size, maxk) indicating matches
        k: K value for Recall@K
    
    Returns:
        Tensor of shape (batch_size,) containing recall values for each sample
    """
    return pos_index[:, :k].sum(dim=1).float()


def ndcg_at_k(pos_index: torch.Tensor, k: int) -> torch.Tensor:
    """Calculate NDCG@K.
    
    Normalized Discounted Cumulative Gain (NDCG) measures the quality of ranking.
    It gives higher weights to items appearing earlier in the ranking.
    
    For single ground truth item (our case):
    NDCG@K = DCG@K / IDCG@K = DCG@K (since IDCG=1 for single item)
    DCG@K = sum(1 / log2(rank + 1)) for matching items in top-k
    
    Args:
        pos_index: Boolean tensor of shape (batch_size, maxk) indicating matches
        k: K value for NDCG@K
    
    Returns:
        Tensor of shape (batch_size,) containing NDCG values for each sample
    """
    # Create ranking positions: [1, 2, 3, ..., maxk]
    ranks = torch.arange(1, pos_index.shape[-1] + 1, dtype=torch.float32)
    
    # Calculate discount: 1 / log2(rank + 1)
    dcg = 1.0 / torch.log2(ranks + 1)
    
    # Apply mask: only count matching positions
    dcg = torch.where(
        pos_index,
        dcg,
        torch.tensor(0.0, dtype=torch.float32)
    )
    
    # Sum DCG values for top-k positions
    return dcg[:, :k].sum(dim=1)


class MetricsCalculator:
    """Helper class to calculate and aggregate metrics."""
    
    def __init__(self, topk_list: List[int] = None, num_layers: int = 4):
        """Initialize metrics calculator.
        
        Args:
            topk_list: List of K values for metrics (default: [5, 10, 20])
            num_layers: Number of semantic ID layers (for layer-wise accuracy)
        """
        self.topk_list = topk_list or [5, 10, 20]
        self.num_layers = num_layers
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.recalls = {f"Recall@{k}": [] for k in self.topk_list}
        self.ndcgs = {f"NDCG@{k}": [] for k in self.topk_list}
        
        # Detailed semantic ID prediction statistics
        self.layer_accuracies = []  # Per-layer accuracy for top-1 prediction
        self.exact_match_count = 0  # Samples with exact match in top-1
        self.total_samples = 0
        self.partial_match_counts = []  # Number of layers matched for top-1
    
    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        beam_size: int
    ):
        """Update metrics with a batch of predictions.
        
        Args:
            preds: Predicted sequences, shape (batch_size * beam_size, seq_len)
            labels: Ground truth sequences, shape (batch_size, target_len)
            beam_size: Number of beams used in generation
        """
        # Reshape predictions to (batch_size, beam_size, seq_len)
        batch_size = labels.shape[0]
        preds = preds.reshape(batch_size, beam_size, -1)
        
        # Calculate position indices
        pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
        
        # Calculate metrics for each K
        for k in self.topk_list:
            recall = recall_at_k(pos_index, k).mean().item()
            ndcg = ndcg_at_k(pos_index, k).mean().item()
            
            self.recalls[f"Recall@{k}"].append(recall)
            self.ndcgs[f"NDCG@{k}"].append(ndcg)
        
        # Calculate detailed semantic ID statistics (using top-1 prediction)
        self._update_semantic_id_stats(preds, labels)
    
    def _update_semantic_id_stats(self, preds: torch.Tensor, labels: torch.Tensor):
        """Update detailed semantic ID prediction statistics.
        
        Args:
            preds: Predicted sequences, shape (batch_size, beam_size, seq_len)
            labels: Ground truth sequences, shape (batch_size, seq_len)
        """
        batch_size = preds.shape[0]
        seq_len = labels.shape[1]
        
        # Use top-1 predictions for layer-wise analysis
        top1_preds = preds[:, 0, :].detach().cpu()  # (batch_size, seq_len)
        labels_cpu = labels.detach().cpu()
        
        # Calculate per-layer accuracy
        layer_correct = torch.zeros(seq_len)
        exact_matches = 0
        
        for i in range(batch_size):
            pred_seq = top1_preds[i]
            label_seq = labels_cpu[i]
            
            # Check each layer
            matches = (pred_seq == label_seq).float()
            layer_correct += matches
            
            # Check exact match
            if torch.all(pred_seq == label_seq):
                exact_matches += 1
            
            # Count partial matches
            num_matched_layers = matches.sum().item()
            self.partial_match_counts.append(num_matched_layers)
        
        # Store layer accuracies for this batch
        self.layer_accuracies.append(layer_correct / batch_size)
        
        # Update counters
        self.exact_match_count += exact_matches
        self.total_samples += batch_size
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics.
        
        Returns:
            Dictionary containing average metrics
        """
        metrics = {}
        
        # Average recalls
        for metric_name, values in self.recalls.items():
            if len(values) > 0:
                metrics[metric_name] = sum(values) / len(values)
            else:
                metrics[metric_name] = 0.0
        
        # Average NDCGs
        for metric_name, values in self.ndcgs.items():
            if len(values) > 0:
                metrics[metric_name] = sum(values) / len(values)
            else:
                metrics[metric_name] = 0.0
        
        # Detailed semantic ID statistics
        if len(self.layer_accuracies) > 0:
            # Per-layer accuracy
            all_layer_accs = torch.stack(self.layer_accuracies)  # (num_batches, seq_len)
            avg_layer_accs = all_layer_accs.mean(dim=0)  # Average over batches
            
            for layer_idx, acc in enumerate(avg_layer_accs.tolist()):
                metrics[f"LayerAcc_L{layer_idx}"] = acc
            
            # Exact match accuracy (same as Recall@1 but clearer naming)
            if self.total_samples > 0:
                metrics["ExactMatch"] = self.exact_match_count / self.total_samples
            
            # Partial match statistics
            if len(self.partial_match_counts) > 0:
                import numpy as np
                partial_matches = np.array(self.partial_match_counts)
                
                # Average number of matched layers
                metrics["AvgMatchedLayers"] = partial_matches.mean()
                
                # Percentage of predictions matching at least N layers
                seq_len = avg_layer_accs.shape[0]
                for threshold in [1, 2, 3]:
                    if threshold <= seq_len:
                        pct = (partial_matches >= threshold).mean()
                        metrics[f"Match>={threshold}Layers"] = pct
        
        return metrics
    
    def compute_and_reset(self) -> Dict[str, float]:
        """Compute average metrics and reset.
        
        Returns:
            Dictionary containing average metrics
        """
        metrics = self.compute()
        self.reset()
        return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dictionary as a readable string.
    
    Args:
        metrics: Dictionary of metric names to values
    
    Returns:
        Formatted string
    """
    lines = []
    
    # Group metrics by category for better readability
    recall_metrics = {}
    ndcg_metrics = {}
    layer_acc_metrics = {}
    semantic_stats = {}
    other_metrics = {}
    
    for key, value in metrics.items():
        if key.startswith("Recall@"):
            recall_metrics[key] = value
        elif key.startswith("NDCG@"):
            ndcg_metrics[key] = value
        elif key.startswith("LayerAcc_"):
            layer_acc_metrics[key] = value
        elif key in ["ExactMatch", "AvgMatchedLayers"] or key.startswith("Match>="):
            semantic_stats[key] = value
        else:
            other_metrics[key] = value
    
    # Format recall metrics
    if recall_metrics:
        lines.append("Recall Metrics:")
        for key in sorted(recall_metrics.keys()):
            lines.append(f"  {key}: {recall_metrics[key]:.4f}")
    
    # Format NDCG metrics
    if ndcg_metrics:
        lines.append("NDCG Metrics:")
        for key in sorted(ndcg_metrics.keys()):
            lines.append(f"  {key}: {ndcg_metrics[key]:.4f}")
    
    # Format semantic ID statistics
    if semantic_stats or layer_acc_metrics:
        lines.append("Semantic ID Prediction Statistics:")
        
        # Exact match first
        if "ExactMatch" in semantic_stats:
            lines.append(f"  Exact Match (Top-1): {semantic_stats['ExactMatch']:.4f}")
        
        # Layer-wise accuracy
        if layer_acc_metrics:
            lines.append("  Per-Layer Accuracy (Top-1):")
            for key in sorted(layer_acc_metrics.keys()):
                layer_num = key.split("_L")[1]
                lines.append(f"    Layer {layer_num}: {layer_acc_metrics[key]:.4f}")
        
        # Average matched layers
        if "AvgMatchedLayers" in semantic_stats:
            lines.append(f"  Avg Matched Layers: {semantic_stats['AvgMatchedLayers']:.4f}")
        
        # Partial match percentages
        partial_match_keys = [k for k in semantic_stats.keys() if k.startswith("Match>=")]
        if partial_match_keys:
            lines.append("  Partial Match Percentages:")
            for key in sorted(partial_match_keys):
                lines.append(f"    {key}: {semantic_stats[key]:.4f}")
    
    # Format other metrics
    if other_metrics:
        lines.append("Other Metrics:")
        for key in sorted(other_metrics.keys()):
            lines.append(f"  {key}: {other_metrics[key]:.4f}")
    
    return "\n".join(lines)

