"""
Evaluation metrics for recommendation systems
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def recall_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Calculate Recall@K
    
    Args:
        predictions: Predicted item indices [batch_size, k]
        ground_truth: Ground truth item indices [batch_size]
        k: Top-K value
    
    Returns:
        recall: Recall@K score
    """
    hits = 0
    total = len(ground_truth)
    
    for i in range(total):
        if ground_truth[i] in predictions[i, :k]:
            hits += 1
    
    return hits / total


def ndcg_at_k(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain)
    
    Args:
        predictions: Predicted item indices [batch_size, k]
        ground_truth: Ground truth item indices [batch_size]
        k: Top-K value
    
    Returns:
        ndcg: NDCG@K score
    """
    ndcg_sum = 0.0
    total = len(ground_truth)
    
    for i in range(total):
        # Find position of ground truth in predictions
        try:
            pos = np.where(predictions[i, :k] == ground_truth[i])[0]
            if len(pos) > 0:
                # DCG: 1 / log2(position + 2)
                ndcg_sum += 1.0 / np.log2(pos[0] + 2)
        except:
            pass
    
    return ndcg_sum / total


def evaluate_model(
    model,
    dataset,
    test_data,
    k_list: List[int] = [5, 10, 20],
    batch_size: int = 256,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate model on test data
    
    Args:
        model: LightGCN model
        dataset: Training dataset (for filtering seen items)
        test_data: Test data (pandas DataFrame with 'user', 'history', 'target')
        k_list: List of K values to evaluate
        batch_size: Batch size for evaluation
        device: Device to use
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    device = torch.device(device)
    
    # Get all item embeddings
    with torch.no_grad():
        all_users_emb, all_items_emb = model.compute_embeddings()
    
    metrics = {f'Recall@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    
    max_k = max(k_list)
    
    # Evaluate in batches
    n_samples = len(test_data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    logger.info(f"Evaluating on {n_samples} samples...")
    
    for batch_idx in tqdm(range(n_batches), desc="Evaluating", leave=False):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_data = test_data.iloc[start_idx:end_idx]
        
        batch_users = batch_data['user'].values
        batch_targets = batch_data['target'].values
        
        # Get user embeddings
        user_indices = torch.LongTensor(batch_users).to(device)
        user_emb = all_users_emb[user_indices]  # [batch_size, emb_dim]
        
        # Compute scores for all items
        scores = torch.matmul(user_emb, all_items_emb.t())  # [batch_size, n_items]
        
        # Filter out training items (items in history)
        for i, (user_id, history) in enumerate(zip(batch_users, batch_data['history'].values)):
            # Get all positive items for this user from training set
            train_pos_items = dataset.user_pos_items.get(user_id, np.array([]))
            
            # Set scores of seen items to -inf
            if len(train_pos_items) > 0:
                scores[i, train_pos_items] = float('-inf')
        
        # Get top-K predictions
        _, top_indices = torch.topk(scores, k=max_k, dim=1)
        top_indices = top_indices.cpu().numpy()  # [batch_size, max_k]
        
        # Calculate metrics for each K
        for k in k_list:
            recall = recall_at_k(top_indices, batch_targets, k)
            ndcg = ndcg_at_k(top_indices, batch_targets, k)
            
            metrics[f'Recall@{k}'].append(recall)
            metrics[f'NDCG@{k}'].append(ndcg)
    
    # Average metrics
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return metrics


def evaluate_full(
    model,
    train_dataset,
    valid_data,
    test_data,
    k_list: List[int] = [5, 10, 20],
    batch_size: int = 256,
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on both validation and test sets
    
    Args:
        model: LightGCN model
        train_dataset: Training dataset
        valid_data: Validation data
        test_data: Test data
        k_list: List of K values
        batch_size: Batch size for evaluation
        device: Device to use
    
    Returns:
        results: Dictionary with 'valid' and 'test' metrics
    """
    results = {}
    
    # Evaluate on validation set
    logger.info("Evaluating on validation set...")
    valid_metrics = evaluate_model(
        model, train_dataset, valid_data, k_list, batch_size, device
    )
    results['valid'] = valid_metrics
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(
        model, train_dataset, test_data, k_list, batch_size, device
    )
    results['test'] = test_metrics
    
    return results


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics"""
    if prefix:
        logger.info(f"{prefix}:")
    
    # Group by metric type
    recall_metrics = {k: v for k, v in metrics.items() if 'Recall' in k}
    ndcg_metrics = {k: v for k, v in metrics.items() if 'NDCG' in k}
    
    # Print Recall
    recall_str = " | ".join([f"{k}: {v:.4f}" for k, v in sorted(recall_metrics.items())])
    logger.info(f"  {recall_str}")
    
    # Print NDCG
    ndcg_str = " | ".join([f"{k}: {v:.4f}" for k, v in sorted(ndcg_metrics.items())])
    logger.info(f"  {ndcg_str}")

