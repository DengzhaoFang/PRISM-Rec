"""
DataLoader implementation for generative recommendation.

Handles batching and collating of sequences with semantic codes.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def collate_fn(batch: List[Dict], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Collate function for batching samples.
    
    This function:
    1. Stacks history and target sequences into tensors
    2. Creates attention masks for padded positions
    
    Args:
        batch: List of samples from the dataset
        pad_token_id: Padding token ID
    
    Returns:
        Dictionary containing:
            - 'history': Tensor of shape (batch_size, seq_len)
            - 'target': Tensor of shape (batch_size, num_layers)
            - 'attention_mask': Tensor of shape (batch_size, seq_len)
    """
    histories = [item['history'] for item in batch]
    targets = [item['target'] for item in batch]
    
    # Convert to tensors
    history_tensor = torch.tensor(histories, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (history_tensor != pad_token_id).long()
    
    return {
        'history': history_tensor,
        'target': target_tensor,
        'attention_mask': attention_mask
    }


class GenRecDataLoader(DataLoader):
    """DataLoader for generative recommendation.
    
    Wraps PyTorch DataLoader with custom collate function.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pad_token_id: int = 0,
        **kwargs
    ):
        """Initialize the DataLoader.
        
        Args:
            dataset: GenRecDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            pad_token_id: Padding token ID
            **kwargs: Additional arguments passed to DataLoader
        """
        # Create collate function with pad_token_id
        def _collate_fn(batch):
            return collate_fn(batch, pad_token_id=pad_token_id)
        
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            **kwargs
        )


def create_dataloaders(
    train_dataset,
    valid_dataset,
    test_dataset,
    batch_size: int = 128,
    eval_batch_size: int = 96,
    num_workers: int = 4,
    pad_token_id: int = 0
):
    """Create DataLoaders for train, validation, and test datasets.
    
    Args:
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_workers: Number of data loading workers
        pad_token_id: Padding token ID
    
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    train_loader = GenRecDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pad_token_id=pad_token_id
    )
    
    valid_loader = GenRecDataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pad_token_id=pad_token_id
    )
    
    test_loader = GenRecDataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pad_token_id=pad_token_id
    )
    
    logger.info(f"Created DataLoaders:")
    logger.info(f"  Train: {len(train_loader)} batches (batch_size={batch_size})")
    logger.info(f"  Valid: {len(valid_loader)} batches (batch_size={eval_batch_size})")
    logger.info(f"  Test: {len(test_loader)} batches (batch_size={eval_batch_size})")
    
    return train_loader, valid_loader, test_loader

