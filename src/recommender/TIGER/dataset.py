"""
Dataset implementation for generative recommendation.

Handles loading sequence data and semantic ID mappings.
"""

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SemanticIDMapper:
    """Manages the mapping from item IDs to semantic codes.
    
    This class handles:
    1. Loading semantic ID mappings from JSON
    2. Converting item IDs to semantic codes
    3. Applying offset transformation for multi-layer codes
    4. Computing actual vocabulary size based on data
    """
    
    def __init__(self, mapping_path: str, codebook_size: int = 256, num_layers: int = 4, pad_token_id: int = 0):
        """Initialize the semantic ID mapper.
        
        Args:
            mapping_path: Path to the semantic_id_mappings.json file
            codebook_size: Size of each codebook (default: 256)
            num_layers: Number of RQ-VAE layers (default: 4)
            pad_token_id: ID for padding token (default: 0)
        """
        self.mapping_path = mapping_path
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        
        # Load the mapping
        logger.info(f"Loading semantic ID mapping from {mapping_path}")
        with open(mapping_path, 'r') as f:
            raw_mapping = json.load(f)
        
        # Handle two formats:
        # 1. Direct mapping: {"123": [1,2,3], ...}
        # 2. Prism format: {"item_to_codes": {"123": [1,2,3]}, ...}
        if 'item_to_codes' in raw_mapping:
            logger.info("Detected Prism format with metadata")
            raw_mapping = raw_mapping['item_to_codes']
        
        # CRITICAL: Auto-detect actual max layers for variable-length IDs
        actual_max_layers = max(len(codes) for codes in raw_mapping.values())
        if actual_max_layers != num_layers:
            logger.warning(f"Detected variable-length IDs: max_layers={actual_max_layers}, config={num_layers}")
            logger.warning(f"Auto-adjusting to max_layers={actual_max_layers}")
            num_layers = actual_max_layers
            self.num_layers = num_layers
        
        # Convert string keys to integers and apply offset transformation
        # Also track the actual max values per layer for vocab size calculation
        self.item_to_codes = {}
        layer_max_values = [0] * num_layers
        
        for item_id_str, codes in raw_mapping.items():
            item_id = int(item_id_str)
            
            # Handle variable-length: pad shorter codes to max_layers
            if len(codes) < num_layers:
                # Pad with pad_token_id (will be handled in offset)
                codes = codes + [pad_token_id] * (num_layers - len(codes))
            
            offset_codes = self._apply_offset(codes)
            self.item_to_codes[item_id] = offset_codes
            
            # Track max values per layer (before offset, excluding padding)
            for i, code in enumerate(codes):
                if code != pad_token_id:  # Don't count padding in max values
                    layer_max_values[i] = max(layer_max_values[i], code)
        
        # Calculate actual vocabulary size based on data
        # Formula: max(offset_code) + 1 for each layer, plus 1 for padding
        self._actual_vocab_size = self._compute_vocab_size(layer_max_values)
        self._layer_max_values = layer_max_values
        
        logger.info(f"Loaded {len(self.item_to_codes)} item-to-code mappings")
        logger.info(f"Layer max values: {layer_max_values}")
        logger.info(f"Computed vocab size: {self._actual_vocab_size}")
        
        # Create padding code
        self.pad_codes = [pad_token_id] * num_layers
    
    def _apply_offset(self, codes: List[int]) -> List[int]:
        """Apply offset transformation to semantic codes.
        
        Formula: offset_code[i] = original_code[i] + i * codebook_size + 1
        
        This ensures that codes from different layers occupy different ID ranges
        in the vocabulary, preventing ambiguity.
        
        Args:
            codes: Original semantic codes (length = num_layers)
        
        Returns:
            Offset-transformed codes
        """
        return [code + i * self.codebook_size + 1 for i, code in enumerate(codes)]
    
    def get_codes(self, item_id: int) -> List[int]:
        """Get semantic codes for an item ID.
        
        Args:
            item_id: The item ID
        
        Returns:
            List of semantic codes (with offset applied)
            If item_id not found, returns padding codes
        """
        return self.item_to_codes.get(item_id, self.pad_codes)
    
    def get_codes_batch(self, item_ids: List[int]) -> List[List[int]]:
        """Get semantic codes for a batch of item IDs.
        
        Args:
            item_ids: List of item IDs
        
        Returns:
            List of semantic code lists
        """
        return [self.get_codes(item_id) for item_id in item_ids]
    
    def __len__(self) -> int:
        """Return the number of items in the mapping."""
        return len(self.item_to_codes)
    
    def _compute_vocab_size(self, layer_max_values: List[int]) -> int:
        """Compute actual vocabulary size based on layer max values.
        
        The vocab size is determined by the maximum token ID that can appear
        after offset transformation, plus 1 for padding token (ID=0).
        
        Formula: max(layer_i_max + i * codebook_size + 1) + 1
        
        Args:
            layer_max_values: List of maximum values for each layer
        
        Returns:
            Actual vocabulary size needed
        """
        max_token_id = 0
        for i, layer_max in enumerate(layer_max_values):
            # Calculate the offset token ID for this layer's max value
            offset_token_id = layer_max + i * self.codebook_size + 1
            max_token_id = max(max_token_id, offset_token_id)
        
        # Add 1 because vocab size is max_id + 1
        return max_token_id + 1
    
    def get_vocab_size(self, use_actual: bool = True) -> int:
        """Get vocabulary size.
        
        Args:
            use_actual: If True, return actual vocab size based on data.
                       If False, return theoretical max (num_layers * codebook_size + 1)
        
        Returns:
            Vocabulary size
        """
        if use_actual:
            return self._actual_vocab_size
        else:
            # Theoretical maximum (may waste some embedding space)
            return self.num_layers * self.codebook_size + 1
    
    def get_layer_stats(self) -> Dict:
        """Get statistics about each layer.
        
        Returns:
            Dictionary with layer statistics
        """
        return {
            'num_layers': self.num_layers,
            'codebook_size': self.codebook_size,
            'layer_max_values': self._layer_max_values,
            'actual_vocab_size': self._actual_vocab_size,
            'theoretical_vocab_size': self.num_layers * self.codebook_size + 1,
            'savings': (self.num_layers * self.codebook_size + 1) - self._actual_vocab_size
        }


def process_sequence_data(
    file_path: str,
    mode: str,
    max_len: int,
    pad_token_id: int = 0
) -> List[Dict]:
    """Process parquet sequence data.
    
    Args:
        file_path: Path to the parquet file
        mode: Processing mode ('train' or 'evaluation')
        max_len: Maximum sequence length
        pad_token_id: Padding token ID
    
    Returns:
        List of processed data items
    """
    logger.info(f"Processing sequence data from {file_path} in {mode} mode")
    
    # Load parquet data
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} sequences")
    
    processed_data = []
    
    if mode == 'train':
        # Sliding window: generate multiple training samples from each sequence
        for idx, row in df.iterrows():
            history = list(row['history'])
            target = row['target']
            sequence = history + [target]
            
            # Create training samples using sliding window
            for i in range(1, len(sequence)):
                processed_data.append({
                    'history': sequence[:i],
                    'target': sequence[i]
                })
    
    elif mode == 'evaluation':
        # Use the last item as target and the rest as history
        for idx, row in df.iterrows():
            history = list(row['history'])
            target = row['target']
            
            processed_data.append({
                'history': history,
                'target': target
            })
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'evaluation'.")
    
    logger.info(f"Generated {len(processed_data)} samples in {mode} mode")
    
    return processed_data


def pad_or_truncate(sequence: List[int], max_len: int, pad_token_id: int = 0) -> List[int]:
    """Pad or truncate a sequence to a specified maximum length.
    
    Args:
        sequence: Input sequence of item IDs
        max_len: Maximum length
        pad_token_id: Padding token ID
    
    Returns:
        Padded or truncated sequence
    """
    if len(sequence) > max_len:
        # Truncate from the left (keep recent items)
        return sequence[-max_len:]
    else:
        # Left pad with pad_token_id
        return [pad_token_id] * (max_len - len(sequence)) + sequence


class GenRecDataset(Dataset):
    """Dataset for generative recommendation.
    
    This dataset:
    1. Loads sequence data from parquet files
    2. Loads semantic ID mappings from JSON
    3. Converts item IDs to semantic codes
    4. Handles padding and truncation
    """
    
    def __init__(
        self,
        sequence_file: str,
        semantic_mapper: SemanticIDMapper,
        mode: str = 'train',
        max_len: int = 20,
        pad_token_id: int = 0
    ):
        """Initialize the dataset.
        
        Args:
            sequence_file: Path to the parquet sequence file
            semantic_mapper: SemanticIDMapper instance
            mode: Processing mode ('train' or 'evaluation')
            max_len: Maximum sequence length
            pad_token_id: Padding token ID
        """
        self.sequence_file = sequence_file
        self.semantic_mapper = semantic_mapper
        self.mode = mode
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        
        # Process sequence data
        self.data = process_sequence_data(
            sequence_file, mode, max_len, pad_token_id
        )
        
        # Convert to semantic codes
        self._convert_to_codes()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
    
    def _convert_to_codes(self):
        """Convert item IDs to semantic codes."""
        logger.info("Converting item IDs to semantic codes...")
        
        missing_items = set()
        
        for item in self.data:
            # Convert history
            history_ids = pad_or_truncate(
                item['history'], self.max_len, self.pad_token_id
            )
            history_codes = []
            for item_id in history_ids:
                codes = self.semantic_mapper.get_codes(item_id)
                if item_id != self.pad_token_id and codes == self.semantic_mapper.pad_codes:
                    missing_items.add(item_id)
                history_codes.extend(codes)
            
            # Convert target
            target_id = item['target']
            target_codes = self.semantic_mapper.get_codes(target_id)
            if target_codes == self.semantic_mapper.pad_codes:
                missing_items.add(target_id)
            
            # Store as codes
            item['history_codes'] = history_codes
            item['target_codes'] = target_codes
        
        if missing_items:
            logger.warning(
                f"Found {len(missing_items)} items without semantic mappings. "
                f"Using padding codes for these items."
            )
            logger.debug(f"Missing items (first 10): {list(missing_items)[:10]}")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with 'history' and 'target' as semantic code sequences
        """
        item = self.data[idx]
        return {
            'history': item['history_codes'],
            'target': item['target_codes']
        }
    
    def get_stats(self) -> Dict:
        """Get dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return {
            'num_samples': len(self),
            'mode': self.mode,
            'max_len': self.max_len,
            'num_layers': self.semantic_mapper.num_layers,
            'sequence_length': self.max_len * self.semantic_mapper.num_layers,
            'vocab_size': self.semantic_mapper.get_vocab_size()
        }


def create_datasets(
    sequence_data_dir: str,
    semantic_mapping_path: str,
    max_len: int = 20,
    codebook_size: int = 256,
    num_layers: int = 4,
    pad_token_id: int = 0,
    model_config: Optional[any] = None
) -> Tuple[GenRecDataset, GenRecDataset, GenRecDataset, SemanticIDMapper]:
    """Create train, validation, and test datasets.
    
    Args:
        sequence_data_dir: Directory containing train.parquet, valid.parquet, test.parquet
        semantic_mapping_path: Path to semantic_id_mappings.json
        max_len: Maximum sequence length
        codebook_size: Codebook size
        num_layers: Number of RQ-VAE layers
        pad_token_id: Padding token ID
        model_config: Optional ModelConfig to update with actual vocab size
    
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset, semantic_mapper)
    """
    # Create semantic mapper (shared across all datasets)
    semantic_mapper = SemanticIDMapper(
        semantic_mapping_path,
        codebook_size=codebook_size,
        num_layers=num_layers,
        pad_token_id=pad_token_id
    )
    
    # Check if num_layers was auto-adjusted (for variable-length IDs)
    if semantic_mapper.num_layers != num_layers:
        logger.warning(f"⚠ num_layers auto-adjusted: {num_layers} → {semantic_mapper.num_layers}")
        logger.warning(f"⚠ This is expected for Prism variable-length IDs")
        num_layers = semantic_mapper.num_layers
    
    # Update model config with actual vocab size if provided
    if model_config is not None:
        actual_vocab_size = semantic_mapper.get_vocab_size(use_actual=True)
        model_config.set_vocab_size(actual_vocab_size)
        logger.info(f"Updated model config vocab_size to {actual_vocab_size}")
        
        # CRITICAL FIX: Update num_code_layers to match actual detected layers
        model_config.num_code_layers = num_layers
        logger.info(f"Updated model config num_code_layers to {num_layers}")
        
        # Log savings
        stats = semantic_mapper.get_layer_stats()
        logger.info(f"Vocab size optimization: saved {stats['savings']} token embeddings")
        
        # Log num_layers info
        logger.info(f"Semantic ID layers: {num_layers} (after auto-detection)")
    
    # Create datasets
    data_dir = Path(sequence_data_dir)
    
    train_dataset = GenRecDataset(
        sequence_file=str(data_dir / "train.parquet"),
        semantic_mapper=semantic_mapper,
        mode='train',
        max_len=max_len,
        pad_token_id=pad_token_id
    )
    
    valid_dataset = GenRecDataset(
        sequence_file=str(data_dir / "valid.parquet"),
        semantic_mapper=semantic_mapper,
        mode='evaluation',
        max_len=max_len,
        pad_token_id=pad_token_id
    )
    
    test_dataset = GenRecDataset(
        sequence_file=str(data_dir / "test.parquet"),
        semantic_mapper=semantic_mapper,
        mode='evaluation',
        max_len=max_len,
        pad_token_id=pad_token_id
    )
    
    return train_dataset, valid_dataset, test_dataset, semantic_mapper

