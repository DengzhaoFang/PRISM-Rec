"""
DataLoader implementation for generative recommendation.

Handles batching and collating of sequences with semantic codes.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def collate_fn(batch: List[Dict], pad_token_id: int = 0, use_dynamic_batching: bool = False) -> Dict[str, torch.Tensor]:
    """Collate function for batching samples.
    """
    histories = [item['history'] for item in batch]
    targets = [item['target'] for item in batch]
    
    # Dynamic batching: pad to max length in this batch
    if use_dynamic_batching:
        # Find max length in this batch
        max_len_in_batch = max(len(h) for h in histories)
        
        # Pad histories to max_len_in_batch
        padded_histories = []
        for h in histories:
            if len(h) < max_len_in_batch:
                # Pad on the left
                padded_h = [pad_token_id] * (max_len_in_batch - len(h)) + h
            else:
                padded_h = h
            padded_histories.append(padded_h)
        
        history_tensor = torch.tensor(padded_histories, dtype=torch.long)
    else:
        # Standard batching: all sequences already padded to same length
        history_tensor = torch.tensor(histories, dtype=torch.long)
    
    target_tensor = torch.tensor(targets, dtype=torch.long)
    
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (history_tensor != pad_token_id).long()
    
    result = {
        'history': history_tensor,
        'target': target_tensor,
        'attention_mask': attention_mask
    }
    
    # Add purified DSI features if present
    if 'history_purified_content' in batch[0]:
        result['history_purified_content'] = torch.stack([
            torch.from_numpy(item['history_purified_content']) for item in batch
        ])

    if 'history_purified_collab' in batch[0]:
        result['history_purified_collab'] = torch.stack([
            torch.from_numpy(item['history_purified_collab']) for item in batch
        ])

    if 'history_codebook_zq' in batch[0]:
        result['history_codebook_zq'] = torch.stack([
            torch.from_numpy(item['history_codebook_zq']) for item in batch
        ])

    if 'target_z_clean' in batch[0]:
        result['target_z_clean'] = torch.stack([
            torch.from_numpy(item['target_z_clean']) for item in batch
        ])
    
    if 'history_item_ids' in batch[0]:
        result['history_item_ids'] = [item['history_item_ids'] for item in batch]
    
    if 'target_item_id' in batch[0]:
        result['target_item_id'] = [item['target_item_id'] for item in batch]

    # Teacher prototypes for TCAF
    if 'target_teacher' in batch[0]:
        result['target_teacher'] = torch.stack([
            torch.from_numpy(item['target_teacher']) for item in batch
        ])

    return result


class GenRecDataLoader(DataLoader):
    """DataLoader for generative recommendation.
    
    Wraps PyTorch DataLoader with custom collate function.
    Supports dynamic batching to reduce padding waste.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pad_token_id: int = 0,
        use_dynamic_batching: bool = False,
        **kwargs
    ):
        """Initialize the DataLoader.
        
        Args:
            dataset: GenRecDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            num_workers: Number of data loading workers
            pad_token_id: Padding token ID
            use_dynamic_batching: If True, pad to max length in batch instead of global max
            **kwargs: Additional arguments passed to DataLoader
        """
        # Create collate function with pad_token_id and dynamic batching
        def _collate_fn(batch):
            return collate_fn(batch, pad_token_id=pad_token_id, use_dynamic_batching=use_dynamic_batching)

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            **kwargs
        )


def create_dataloaders(
    train_dataset,
    valid_dataset,
    test_dataset,
    batch_size: int = 128,
    eval_batch_size: int = 96,
    num_workers: int = 4,
    pad_token_id: int = 0,
    use_dynamic_batching: bool = False
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
        use_dynamic_batching: If True, use dynamic batching to reduce padding
    
    Returns:
        Tuple of (train_loader, valid_loader, test_loader)
    """
    train_loader = GenRecDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pad_token_id=pad_token_id,
        use_dynamic_batching=use_dynamic_batching
    )
    
    valid_loader = GenRecDataLoader(
        valid_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pad_token_id=pad_token_id,
        use_dynamic_batching=use_dynamic_batching
    )
    
    test_loader = GenRecDataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pad_token_id=pad_token_id,
        use_dynamic_batching=use_dynamic_batching
    )
    
    logger.info(f"Created DataLoaders:")
    logger.info(f"  Train: {len(train_loader)} batches (batch_size={batch_size})")
    logger.info(f"  Valid: {len(valid_loader)} batches (batch_size={eval_batch_size})")
    logger.info(f"  Test: {len(test_loader)} batches (batch_size={eval_batch_size})")
    if use_dynamic_batching:
        logger.info(f"  Dynamic batching enabled (reduces padding waste)")
    
    return train_loader, valid_loader, test_loader

