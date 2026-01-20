"""
ActionPiece Dataset implementation for generative recommendation.

Handles loading sequence data with ActionPiece tokenization,
including dynamic SPR (Set Permutation Regularization) augmentation.
"""

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Import local ActionPieceCore from sid_tokenizer
from src.sid_tokenizer.ActionPiece.actionpiece_core import ActionPieceCore

logger = logging.getLogger(__name__)


class ActionPieceMapper:
    """Manages ActionPiece tokenization for items.
    
    This class handles:
    1. Loading ActionPiece tokenizer and item features
    2. Converting item sequences to token sequences with SPR augmentation
    3. Managing vocabulary and special tokens
    """
    
    def __init__(
        self, 
        tokenizer_path: str,
        item2feat_path: str,
        vocab_size: Optional[int] = None,
        pad_token_id: int = 0
    ):
        """Initialize the ActionPiece mapper.
        
        Args:
            tokenizer_path: Path to actionpiece.json file
            item2feat_path: Path to item2feat.json file
            vocab_size: Optional vocab size limit (uses full vocab if None)
            pad_token_id: ID for padding token (default: 0)
        """
        self.pad_token_id = pad_token_id
        
        # Load ActionPiece tokenizer
        logger.info(f"Loading ActionPiece tokenizer from {tokenizer_path}")
        self.actionpiece = ActionPieceCore.from_pretrained(
            tokenizer_path, 
            vocab_size=vocab_size
        )
        
        # Load item to feature mapping
        logger.info(f"Loading item features from {item2feat_path}")
        with open(item2feat_path, 'r') as f:
            raw_item2feat = json.load(f)
        
        # Convert string keys to integers
        self.item2feat = {int(k): tuple(v) for k, v in raw_item2feat.items()}
        
        # Special tokens
        self.bos_token = self.actionpiece.vocab_size
        self.eos_token = self.actionpiece.vocab_size + 1
        
        # Number of feature categories (m in paper, typically 4 or 5 with hash)
        self.n_categories = self.actionpiece.n_categories
        
        logger.info(f"ActionPiece vocab size: {self.actionpiece.vocab_size}")
        logger.info(f"Number of categories: {self.n_categories}")
        logger.info(f"Number of items: {len(self.item2feat)}")
        logger.info(f"BOS token: {self.bos_token}, EOS token: {self.eos_token}")
    
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self.eos_token + 1
    
    def item_to_state(self, item_id: int) -> Optional[np.ndarray]:
        """Convert item ID to state (feature indices).
        
        Args:
            item_id: Item ID
            
        Returns:
            numpy array of feature indices, or None if item not found
        """
        if item_id not in self.item2feat:
            return None
        
        feat = self.item2feat[item_id]
        # Convert features to token indices using actionpiece rank
        state = []
        for i, f in enumerate(feat):
            token_idx = self.actionpiece.rank.get((i, f))
            if token_idx is not None:
                state.append(token_idx)
        
        return np.array(state) if state else None
    
    def encode_sequence(
        self, 
        item_ids: List[int], 
        shuffle: str = 'feature'
    ) -> List[int]:
        """Encode a sequence of items to tokens using ActionPiece.
        
        Args:
            item_ids: List of item IDs
            shuffle: Shuffle strategy ('feature' for SPR, 'none' for deterministic)
            
        Returns:
            List of token IDs
        """
        # Convert items to states
        states = []
        for item_id in item_ids:
            state = self.item_to_state(item_id)
            if state is not None:
                states.append(state)
        
        if not states:
            return []
        
        state_seq = np.array(states)
        
        # Encode using ActionPiece with specified shuffle strategy
        tokens = self.actionpiece.encode(state_seq, shuffle=shuffle)
        
        return tokens
    
    def encode_label_for_train(self, item_id: int) -> List[int]:
        """Encode a single item as label for training (with merging, no shuffle).
        
        During training, labels are encoded using ActionPiece with shuffle='none',
        which applies merge rules but maintains deterministic order.
        
        Args:
            item_id: Item ID
            
        Returns:
            List of token IDs (merged tokens)
        """
        state = self.item_to_state(item_id)
        if state is None:
            return [self.pad_token_id] * self.n_categories
        
        # Encode using ActionPiece with shuffle='none' (merge but no random permutation)
        state_seq = np.array([state])
        encoded = self.actionpiece.encode(state_seq, shuffle='none')
        return encoded
    
    def get_raw_state(self, item_id: int) -> Optional[np.ndarray]:
        """Get raw state (original feature tokens) for evaluation.
        
        During evaluation, we compare model outputs with raw states (original features).
        
        Args:
            item_id: Item ID
            
        Returns:
            numpy array of original feature token indices
        """
        return self.item_to_state(item_id)
    
    def decode_tokens(self, tokens: List[int]) -> Optional[List[Tuple[int, int]]]:
        """Decode tokens back to features.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            List of (category, feature) tuples, or None if invalid
        """
        return self.actionpiece.decode_single_state(tokens)


class ActionPieceDataset(Dataset):
    """Dataset for ActionPiece-based generative recommendation.
    
    This dataset:
    1. Loads sequence data from parquet files
    2. Uses ActionPiece tokenization with dynamic SPR augmentation
    3. Handles padding and truncation
    """
    
    def __init__(
        self,
        sequence_file: str,
        actionpiece_mapper: ActionPieceMapper,
        mode: str = 'train',
        max_len: int = 20,
        shuffle: str = 'feature',  # 'feature' for SPR, 'none' for deterministic
        pad_token_id: int = 0
    ):
        """Initialize the dataset.
        
        Args:
            sequence_file: Path to the parquet sequence file
            actionpiece_mapper: ActionPieceMapper instance
            mode: Processing mode ('train' or 'evaluation')
            max_len: Maximum sequence length (number of items)
            shuffle: Shuffle strategy for encoding
            pad_token_id: Padding token ID
        """
        self.sequence_file = sequence_file
        self.mapper = actionpiece_mapper
        self.mode = mode
        self.max_len = max_len
        self.shuffle = shuffle
        self.pad_token_id = pad_token_id
        
        # Load and process sequence data
        self.data = self._load_data()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples (mode={mode})")
    
    def _load_data(self) -> List[Dict]:
        """Load and process sequence data."""
        logger.info(f"Loading sequence data from {self.sequence_file}")
        
        df = pd.read_parquet(self.sequence_file)
        processed_data = []
        
        if self.mode == 'train':
            # Sliding window: generate multiple training samples
            for _, row in df.iterrows():
                history = list(row['history'])
                target = row['target']
                sequence = history + [target]
                
                # Create training samples using sliding window
                for i in range(1, len(sequence)):
                    # Take at most max_len items before target
                    start_idx = max(0, i - self.max_len)
                    processed_data.append({
                        'history': sequence[start_idx:i],
                        'target': sequence[i]
                    })
        else:
            # Evaluation: use last item as target
            for _, row in df.iterrows():
                history = list(row['history'])
                target = row['target']
                
                # Take at most max_len items
                history = history[-self.max_len:]
                
                processed_data.append({
                    'history': history,
                    'target': target
                })
        
        logger.info(f"Generated {len(processed_data)} samples")
        return processed_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with dynamic tokenization.
        
        For training:
        - Input: SPR augmentation (random shuffle)
        - Label: ActionPiece encoded with shuffle='none' (merged but deterministic)
        
        For evaluation:
        - Input: deterministic encoding (shuffle='none')
        - Label: raw state (original feature tokens) for metric computation
        """
        item = self.data[idx]
        
        # Determine shuffle strategy for input
        if self.mode == 'train':
            shuffle = self.shuffle  # Usually 'feature' for SPR
        else:
            shuffle = 'none'  # Deterministic for evaluation
        
        # Encode history sequence
        history_tokens = self.mapper.encode_sequence(item['history'], shuffle=shuffle)
        
        # Add BOS and EOS tokens
        input_tokens = [self.mapper.bos_token] + history_tokens + [self.mapper.eos_token]
        
        # Get raw state for evaluation metrics
        raw_state = self.mapper.get_raw_state(item['target'])
        
        if self.mode == 'train':
            # Training: label is ActionPiece encoded (merged, no shuffle)
            target_tokens = self.mapper.encode_label_for_train(item['target'])
            target_tokens = target_tokens + [self.mapper.eos_token]
        else:
            # Evaluation: label is raw state (original features) for metric computation
            # We still need encoded version for loss computation during validation
            target_tokens = raw_state.tolist() if raw_state is not None else [self.pad_token_id] * self.mapper.n_categories
            target_tokens = target_tokens + [self.mapper.eos_token]
        
        return {
            'input_ids': input_tokens,
            'target': target_tokens,
            'target_state': raw_state,  # Raw state for evaluation metrics
            'item_id': item['target'],
            'history': item['history']  # For ensemble evaluation
        }
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            'num_samples': len(self),
            'mode': self.mode,
            'max_len': self.max_len,
            'n_categories': self.mapper.n_categories,
            'vocab_size': self.mapper.vocab_size,
            'shuffle': self.shuffle
        }


def collate_fn_actionpiece(
    batch: List[Dict], 
    pad_token_id: int = 0,
    ignored_label: int = -100,
    n_categories: int = 5,
    max_label_len: int = None
) -> Dict[str, torch.Tensor]:
    """Collate function for ActionPiece batches.
    
    Args:
        batch: List of samples from dataset
        pad_token_id: Padding token ID
        ignored_label: Label to ignore in loss computation
        n_categories: Number of feature categories
        max_label_len: Maximum label length (for training with merged tokens)
    
    Returns:
        Dictionary with batched tensors
    """
    input_ids = [item['input_ids'] for item in batch]
    targets = [item['target'] for item in batch]
    target_states = [item['target_state'] for item in batch]
    item_ids = [item['item_id'] for item in batch]
    histories = [item['history'] for item in batch]
    
    # Pad input sequences
    max_input_len = max(len(ids) for ids in input_ids)
    padded_inputs = []
    attention_masks = []
    
    for ids in input_ids:
        padding_len = max_input_len - len(ids)
        padded_inputs.append(ids + [pad_token_id] * padding_len)
        attention_masks.append([1] * len(ids) + [0] * padding_len)
    
    # Determine target length
    # For training: targets may have variable length due to merging
    # For evaluation: targets are n_categories + 1 (raw features + EOS)
    if max_label_len is None:
        max_label_len = n_categories + 1
    
    # Pad targets
    max_target_len = max(len(t) for t in targets)
    target_len = max(max_target_len, max_label_len)
    
    padded_targets = []
    for target in targets:
        if len(target) < target_len:
            padded_targets.append(target + [ignored_label] * (target_len - len(target)))
        else:
            padded_targets.append(target[:target_len])
    
    # Convert target states to tensor (for evaluation metrics)
    # These are always raw states (original features)
    padded_states = []
    for state in target_states:
        if state is not None:
            padded_states.append(state.tolist())
        else:
            padded_states.append([pad_token_id] * n_categories)
    
    return {
        'input_ids': torch.tensor(padded_inputs, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'labels': torch.tensor(padded_targets, dtype=torch.long),
        'target_states': torch.tensor(padded_states, dtype=torch.long),
        'item_ids': torch.tensor(item_ids, dtype=torch.long),
        'histories': histories  # Keep as list for dynamic ensemble encoding
    }


def collate_fn_actionpiece_test(
    batch: List[Dict],
    mapper: 'ActionPieceMapper',
    n_ensemble: int = 5,
    pad_token_id: int = 0,
    n_categories: int = 5
) -> Dict[str, torch.Tensor]:
    """Collate function for ActionPiece test batches with ensemble.
    
    Following the original ActionPiece implementation:
    - Generate n_ensemble SPR augmentations for each sample
    - input_ids shape: (batch_size * n_ensemble, seq_len)
    - labels/target_states shape: (batch_size, n_categories)
    
    Args:
        batch: List of samples from dataset
        mapper: ActionPieceMapper for encoding
        n_ensemble: Number of ensemble augmentations per sample
        pad_token_id: Padding token ID
        n_categories: Number of feature categories
    
    Returns:
        Dictionary with batched tensors
    """
    input_ids = []
    target_states = []
    
    for item in batch:
        history = item['history']
        target_state = item['target_state']
        
        # Generate n_ensemble SPR augmentations for this sample
        for _ in range(n_ensemble):
            # Encode with SPR augmentation (shuffle='feature')
            tokens = mapper.encode_sequence(history, shuffle='feature')
            input_tokens = [mapper.bos_token] + tokens + [mapper.eos_token]
            input_ids.append(input_tokens)
        
        # Label is added only once per original sample
        if target_state is not None:
            target_states.append(target_state.tolist())
        else:
            target_states.append([pad_token_id] * n_categories)
    
    # Pad input sequences
    max_input_len = max(len(ids) for ids in input_ids)
    padded_inputs = []
    attention_masks = []
    
    for ids in input_ids:
        padding_len = max_input_len - len(ids)
        padded_inputs.append(ids + [pad_token_id] * padding_len)
        attention_masks.append([1] * len(ids) + [0] * padding_len)
    
    return {
        'input_ids': torch.tensor(padded_inputs, dtype=torch.long),
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'target_states': torch.tensor(target_states, dtype=torch.long),
        'n_ensemble': n_ensemble,
        'batch_size': len(batch)
    }


class ActionPieceEnsembleDataset(Dataset):
    """Dataset for ActionPiece with inference-time ensemble.
    
    This dataset generates multiple SPR-augmented versions of each sample
    for inference-time ensemble, following the original ActionPiece paper.
    """
    
    def __init__(
        self,
        sequence_file: str,
        actionpiece_mapper: ActionPieceMapper,
        max_len: int = 20,
        n_ensemble: int = 5,
        pad_token_id: int = 0
    ):
        """Initialize the ensemble dataset.
        
        Args:
            sequence_file: Path to the parquet sequence file
            actionpiece_mapper: ActionPieceMapper instance
            max_len: Maximum sequence length (number of items)
            n_ensemble: Number of SPR augmentations per sample
            pad_token_id: Padding token ID
        """
        self.sequence_file = sequence_file
        self.mapper = actionpiece_mapper
        self.max_len = max_len
        self.n_ensemble = n_ensemble
        self.pad_token_id = pad_token_id
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Ensemble dataset initialized with {len(self.data)} samples, n_ensemble={n_ensemble}")
    
    def _load_data(self) -> List[Dict]:
        """Load sequence data."""
        logger.info(f"Loading sequence data from {self.sequence_file}")
        
        df = pd.read_parquet(self.sequence_file)
        processed_data = []
        
        for _, row in df.iterrows():
            history = list(row['history'])
            target = row['target']
            history = history[-self.max_len:]
            
            processed_data.append({
                'history': history,
                'target': target
            })
        
        return processed_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample with multiple SPR-augmented inputs.
        
        Returns:
            Dictionary with:
            - input_ids_list: List of n_ensemble different SPR encodings
            - target_state: Raw state for evaluation
            - item_id: Target item ID
        """
        item = self.data[idx]
        
        # Generate n_ensemble different SPR encodings
        input_ids_list = []
        for _ in range(self.n_ensemble):
            # Each call to encode_sequence with shuffle='feature' gives different result
            history_tokens = self.mapper.encode_sequence(item['history'], shuffle='feature')
            input_tokens = [self.mapper.bos_token] + history_tokens + [self.mapper.eos_token]
            input_ids_list.append(input_tokens)
        
        # Get raw state for evaluation
        raw_state = self.mapper.get_raw_state(item['target'])
        
        return {
            'input_ids_list': input_ids_list,
            'target_state': raw_state,
            'item_id': item['target']
        }


def collate_fn_ensemble(
    batch: List[Dict],
    pad_token_id: int = 0,
    n_categories: int = 5,
    n_ensemble: int = 5
) -> Dict[str, torch.Tensor]:
    """Collate function for ensemble batches.
    
    Args:
        batch: List of samples from ensemble dataset
        pad_token_id: Padding token ID
        n_categories: Number of feature categories
        n_ensemble: Number of ensemble runs
    
    Returns:
        Dictionary with batched tensors
    """
    batch_size = len(batch)
    
    # Flatten all input_ids from all samples and all ensemble runs
    all_input_ids = []
    for item in batch:
        all_input_ids.extend(item['input_ids_list'])
    
    # Pad input sequences
    max_input_len = max(len(ids) for ids in all_input_ids)
    padded_inputs = []
    attention_masks = []
    
    for ids in all_input_ids:
        padding_len = max_input_len - len(ids)
        padded_inputs.append(ids + [pad_token_id] * padding_len)
        attention_masks.append([1] * len(ids) + [0] * padding_len)
    
    # Target states (one per sample, not per ensemble)
    target_states = []
    item_ids = []
    for item in batch:
        state = item['target_state']
        if state is not None:
            target_states.append(state.tolist())
        else:
            target_states.append([pad_token_id] * n_categories)
        item_ids.append(item['item_id'])
    
    return {
        'input_ids': torch.tensor(padded_inputs, dtype=torch.long),  # (batch_size * n_ensemble, seq_len)
        'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        'target_states': torch.tensor(target_states, dtype=torch.long),  # (batch_size, n_categories)
        'item_ids': torch.tensor(item_ids, dtype=torch.long),
        'batch_size': batch_size,
        'n_ensemble': n_ensemble
    }


def create_actionpiece_datasets(
    sequence_data_dir: str,
    tokenizer_path: str,
    item2feat_path: str,
    max_len: int = 20,
    vocab_size: Optional[int] = None,
    train_shuffle: str = 'feature',
    pad_token_id: int = 0,
    n_ensemble: int = 5
) -> Tuple['ActionPieceDataset', 'ActionPieceDataset', 'ActionPieceDataset', 'ActionPieceMapper']:
    """Create train, validation, and test datasets.
    
    Args:
        sequence_data_dir: Directory containing train.parquet, valid.parquet, test.parquet
        tokenizer_path: Path to actionpiece.json
        item2feat_path: Path to item2feat.json
        max_len: Maximum sequence length
        vocab_size: Optional vocabulary size limit
        train_shuffle: Shuffle strategy for training ('feature' for SPR)
        pad_token_id: Padding token ID
        n_ensemble: Number of ensemble runs for test set
    
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset, mapper)
    """
    # Create shared mapper
    mapper = ActionPieceMapper(
        tokenizer_path=tokenizer_path,
        item2feat_path=item2feat_path,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id
    )
    
    data_dir = Path(sequence_data_dir)
    
    # Create datasets
    train_dataset = ActionPieceDataset(
        sequence_file=str(data_dir / "train.parquet"),
        actionpiece_mapper=mapper,
        mode='train',
        max_len=max_len,
        shuffle=train_shuffle,
        pad_token_id=pad_token_id
    )
    
    valid_dataset = ActionPieceDataset(
        sequence_file=str(data_dir / "valid.parquet"),
        actionpiece_mapper=mapper,
        mode='evaluation',
        max_len=max_len,
        shuffle='none',
        pad_token_id=pad_token_id
    )
    
    test_dataset = ActionPieceDataset(
        sequence_file=str(data_dir / "test.parquet"),
        actionpiece_mapper=mapper,
        mode='evaluation',
        max_len=max_len,
        shuffle='none',
        pad_token_id=pad_token_id
    )
    
    return train_dataset, valid_dataset, test_dataset, mapper
