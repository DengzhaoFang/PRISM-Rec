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
        
        # Build reverse mapping: codes -> item_id
        # Use tuple of codes as key for hashability
        self.codes_to_item = {}
        for item_id, codes in self.item_to_codes.items():
            codes_tuple = tuple(codes)
            # Handle potential collisions (multiple items with same codes)
            if codes_tuple in self.codes_to_item:
                logger.warning(f"Code collision detected: {codes_tuple} maps to both {self.codes_to_item[codes_tuple]} and {item_id}")
            else:
                self.codes_to_item[codes_tuple] = item_id
        
        logger.info(f"Built reverse mapping with {len(self.codes_to_item)} unique code sequences")
        
        # Create padding code
        self.pad_codes = [pad_token_id] * num_layers
    
    def _apply_offset(self, codes: List[int]) -> List[int]:
        """Apply offset transformation to semantic codes.
        
        Formula: offset_code[i] = original_code[i] + i * codebook_size + 3
        
        This ensures:
        1. Special tokens (0=PAD, 1=EOS, 2=MASK) are reserved
        2. Codes from different layers occupy different ID ranges
        
        Args:
            codes: Original semantic codes (length = num_layers)
        
        Returns:
            Offset-transformed codes
        """
        # Reserve first 3 token IDs: 0=PAD, 1=EOS, 2=MASK
        return [code + i * self.codebook_size + 3 for i, code in enumerate(codes)]
    
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
    
    def decode_codes(self, codes: List[int]) -> Optional[int]:
        """Decode semantic codes back to item ID.
        
        Args:
            codes: List of semantic codes (with offset applied)
        
        Returns:
            Item ID if found, None otherwise
        """
        codes_tuple = tuple(codes)
        return self.codes_to_item.get(codes_tuple, None)
    
    def decode_codes_batch(self, codes_batch: List[List[int]]) -> List[Optional[int]]:
        """Decode a batch of semantic codes back to item IDs.
        
        Args:
            codes_batch: List of semantic code lists
        
        Returns:
            List of item IDs (None for codes not found)
        """
        return [self.decode_codes(codes) for codes in codes_batch]
    
    def __len__(self) -> int:
        """Return the number of items in the mapping."""
        return len(self.item_to_codes)
    
    def _compute_vocab_size(self, layer_max_values: List[int]) -> int:
        """Compute actual vocabulary size based on layer max values.
        
        The vocab size is determined by the maximum token ID that can appear
        after offset transformation, plus 1 for padding token (ID=0).
        
        Formula: max(layer_i_max + i * codebook_size + 3) + 1
        
        Args:
            layer_max_values: List of maximum values for each layer
        
        Returns:
            Actual vocabulary size needed
        """
        max_token_id = 0
        for i, layer_max in enumerate(layer_max_values):
            # Calculate the offset token ID for this layer's max value
            # Must match the offset formula in _apply_offset: +3 for special tokens
            offset_token_id = layer_max + i * self.codebook_size + 3
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


class DualSemanticIDMapper:
    """Manages dual mappings for EAGER: behavior codes and semantic codes.
    
    This class handles:
    1. Loading both behavior and semantic ID mappings
    2. Converting item IDs to both types of codes
    3. Managing two separate vocabularies
    """
    
    def __init__(
        self,
        behavior_mapping_path: str,
        semantic_mapping_path: str,
        codebook_size: int = 256,
        num_layers: int = 4,
        pad_token_id: int = 0
    ):
        """Initialize the dual semantic ID mapper.
        
        Args:
            behavior_mapping_path: Path to behavior semantic IDs (from collaborative embeddings)
            semantic_mapping_path: Path to semantic IDs (from content embeddings)
            codebook_size: Size of each codebook
            num_layers: Number of layers (HKM depth)
            pad_token_id: ID for padding token
        """
        logger.info("Initializing DualSemanticIDMapper for EAGER")
        
        # Create two separate mappers
        self.behavior_mapper = SemanticIDMapper(
            behavior_mapping_path,
            codebook_size=codebook_size,
            num_layers=num_layers,
            pad_token_id=pad_token_id
        )
        logger.info(f"Behavior mapper: {len(self.behavior_mapper)} items, "
                   f"vocab_size={self.behavior_mapper.get_vocab_size()}")
        
        self.semantic_mapper = SemanticIDMapper(
            semantic_mapping_path,
            codebook_size=codebook_size,
            num_layers=num_layers,
            pad_token_id=pad_token_id
        )
        logger.info(f"Semantic mapper: {len(self.semantic_mapper)} items, "
                   f"vocab_size={self.semantic_mapper.get_vocab_size()}")
        
        # Use the larger vocab size to accommodate both
        self.codebook_size = codebook_size
        self.num_layers = max(self.behavior_mapper.num_layers, self.semantic_mapper.num_layers)
        self.pad_token_id = pad_token_id
        
        logger.info(f"DualSemanticIDMapper initialized with {self.num_layers} layers")
    
    def get_behavior_codes(self, item_id: int) -> List[int]:
        """Get behavior codes for an item ID."""
        return self.behavior_mapper.get_codes(item_id)
    
    def get_semantic_codes(self, item_id: int) -> List[int]:
        """Get semantic codes for an item ID."""
        return self.semantic_mapper.get_codes(item_id)
    
    def get_behavior_codes_batch(self, item_ids: List[int]) -> List[List[int]]:
        """Get behavior codes for a batch of item IDs."""
        return self.behavior_mapper.get_codes_batch(item_ids)
    
    def get_semantic_codes_batch(self, item_ids: List[int]) -> List[List[int]]:
        """Get semantic codes for a batch of item IDs."""
        return self.semantic_mapper.get_codes_batch(item_ids)
    
    def decode_behavior_codes(self, codes: List[int]) -> Optional[int]:
        """Decode behavior codes back to item ID."""
        return self.behavior_mapper.decode_codes(codes)
    
    def decode_semantic_codes(self, codes: List[int]) -> Optional[int]:
        """Decode semantic codes back to item ID."""
        return self.semantic_mapper.decode_codes(codes)
    
    def decode_behavior_codes_batch(self, codes_batch: List[List[int]]) -> List[Optional[int]]:
        """Decode a batch of behavior codes back to item IDs."""
        return self.behavior_mapper.decode_codes_batch(codes_batch)
    
    def decode_semantic_codes_batch(self, codes_batch: List[List[int]]) -> List[Optional[int]]:
        """Decode a batch of semantic codes back to item IDs."""
        return self.semantic_mapper.decode_codes_batch(codes_batch)
    
    def get_vocab_size(self) -> int:
        """Get the maximum vocabulary size needed for both mappers."""
        return max(
            self.behavior_mapper.get_vocab_size(),
            self.semantic_mapper.get_vocab_size()
        )
    
    def get_layer_stats(self) -> Dict:
        """Get statistics about both mappers."""
        return {
            'num_layers': self.num_layers,
            'codebook_size': self.codebook_size,
            'behavior_stats': self.behavior_mapper.get_layer_stats(),
            'semantic_stats': self.semantic_mapper.get_layer_stats(),
            'combined_vocab_size': self.get_vocab_size()
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
        semantic_mapper,  # Can be SemanticIDMapper or DualSemanticIDMapper
        mode: str = 'train',
        max_len: int = 20,
        pad_token_id: int = 0
    ):
        """Initialize the dataset.
        
        Args:
            sequence_file: Path to the parquet sequence file
            semantic_mapper: SemanticIDMapper or DualSemanticIDMapper instance
            mode: Processing mode ('train' or 'evaluation')
            max_len: Maximum sequence length
            pad_token_id: Padding token ID
        """
        self.sequence_file = sequence_file
        self.semantic_mapper = semantic_mapper
        self.mode = mode
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        
        # Detect if using dual mapper (EAGER)
        self.is_dual = isinstance(semantic_mapper, DualSemanticIDMapper)
        
        # Process sequence data
        self.data = process_sequence_data(
            sequence_file, mode, max_len, pad_token_id
        )
        
        # Convert to semantic codes
        self._convert_to_codes()
        
        mode_str = "dual-stream (EAGER)" if self.is_dual else "single-stream"
        logger.info(f"Dataset initialized with {len(self.data)} samples ({mode_str})")
    
    def _convert_to_codes(self):
        """Convert item IDs to semantic codes (or dual codes for EAGER)."""
        if self.is_dual:
            logger.info("Converting item IDs to dual codes (behavior + semantic)...")
        else:
            logger.info("Converting item IDs to semantic codes...")
        
        missing_items = set()
        
        for item_data in self.data:
            # Get raw history IDs
            history_ids = pad_or_truncate(
                item_data['history'], self.max_len, self.pad_token_id
            )
            
            if self.is_dual:
                # EAGER: Convert to both behavior and semantic codes
                history_behavior_codes = []
                history_semantic_codes = []
                
                for item_id in history_ids:
                    behavior_codes = self.semantic_mapper.get_behavior_codes(item_id)
                    semantic_codes = self.semantic_mapper.get_semantic_codes(item_id)
                    
                    if item_id != self.pad_token_id:
                        if behavior_codes == self.semantic_mapper.behavior_mapper.pad_codes:
                            missing_items.add(item_id)
                    
                    history_behavior_codes.extend(behavior_codes)
                    history_semantic_codes.extend(semantic_codes)
                
                # Convert target
                target_id = item_data['target']
                target_behavior_codes = self.semantic_mapper.get_behavior_codes(target_id)
                target_semantic_codes = self.semantic_mapper.get_semantic_codes(target_id)
                
                if target_behavior_codes == self.semantic_mapper.behavior_mapper.pad_codes:
                    missing_items.add(target_id)
                
                # CRITICAL: Add EOS token at the end for GCT summary
                # Paper: "insert a learnable token y_[EOS] at the end of the sequence"
                # This enables the final token to make a summary of the entire sequence
                # EOS token ID is 1 (reserved special token)
                eos_token_id = 1  # Must match config.eos_token_id
                target_behavior_codes_with_eos = target_behavior_codes + [eos_token_id]
                target_semantic_codes_with_eos = target_semantic_codes + [eos_token_id]
                
                # Store dual codes
                item_data['history_item_ids'] = history_ids  # Raw IDs for encoder
                item_data['history_behavior_codes'] = history_behavior_codes
                item_data['history_semantic_codes'] = history_semantic_codes
                item_data['target_behavior_codes'] = target_behavior_codes_with_eos  # With EOS
                item_data['target_semantic_codes'] = target_semantic_codes_with_eos  # With EOS
                item_data['target_item_id'] = target_id  # For GCT task
                
            else:
                # Single semantic mapper (TIGER/LETTER)
                history_codes = []
                for item_id in history_ids:
                    codes = self.semantic_mapper.get_codes(item_id)
                    if item_id != self.pad_token_id and codes == self.semantic_mapper.pad_codes:
                        missing_items.add(item_id)
                    history_codes.extend(codes)
                
                # Convert target
                target_id = item_data['target']
                target_codes = self.semantic_mapper.get_codes(target_id)
                if target_codes == self.semantic_mapper.pad_codes:
                    missing_items.add(target_id)
                
                # Store as codes
                item_data['history_codes'] = history_codes
                item_data['target_codes'] = target_codes
        
        if missing_items:
            logger.warning(
                f"Found {len(missing_items)} items without semantic mappings. "
                f"Using padding codes for these items."
            )
            logger.debug(f"Missing items (first 10): {list(missing_items)[:10]}")
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            For dual mapper (EAGER): Dictionary with dual codes
            For single mapper: Dictionary with 'history' and 'target'
        """
        item = self.data[idx]
        
        if self.is_dual:
            # EAGER: Return dual codes
            return {
                'history_item_ids': item['history_item_ids'],
                'history_behavior_codes': item['history_behavior_codes'],
                'history_semantic_codes': item['history_semantic_codes'],
                'target_behavior_codes': item['target_behavior_codes'],
                'target_semantic_codes': item['target_semantic_codes'],
                'target_item_id': item['target_item_id']
            }
        else:
            # TIGER/LETTER: Return single codes
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
    behavior_mapping_path: Optional[str] = None,  # For EAGER dual-stream
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
        behavior_mapping_path: Optional path to behavior semantic IDs (for EAGER)
        max_len: Maximum sequence length
        codebook_size: Codebook size
        num_layers: Number of layers (HKM depth)
        pad_token_id: Padding token ID
        model_config: Optional ModelConfig to update with actual vocab size
    
    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset, semantic_mapper)
    """
    # Create semantic mapper (shared across all datasets)
    if behavior_mapping_path is not None:
        # EAGER: Use dual semantic ID mapper
        logger.info("Creating DualSemanticIDMapper for EAGER dual-stream architecture")
        semantic_mapper = DualSemanticIDMapper(
            behavior_mapping_path=behavior_mapping_path,
            semantic_mapping_path=semantic_mapping_path,
            codebook_size=codebook_size,
            num_layers=num_layers,
            pad_token_id=pad_token_id
        )
    else:
        # TIGER/LETTER: Use single semantic ID mapper
        logger.info("Creating SemanticIDMapper for single-stream architecture")
        semantic_mapper = SemanticIDMapper(
            semantic_mapping_path,
            codebook_size=codebook_size,
            num_layers=num_layers,
            pad_token_id=pad_token_id
        )
    
    # Get num_layers (may have been auto-adjusted)
    num_layers = semantic_mapper.num_layers
    
    # Update model config with actual vocab size if provided
    if model_config is not None:
        actual_vocab_size = semantic_mapper.get_vocab_size()
        num_items = len(semantic_mapper.item_to_codes) if not isinstance(semantic_mapper, DualSemanticIDMapper) else len(semantic_mapper.behavior_mapper.item_to_codes)
        
        model_config.set_vocab_size(actual_vocab_size, num_items=num_items)
        logger.info(f"Updated model config vocab_size to {actual_vocab_size}")
        logger.info(f"Updated model config num_items to {num_items}")
        
        # Update num_code_layers to match actual detected layers
        model_config.num_code_layers = num_layers
        logger.info(f"Updated model config num_code_layers to {num_layers}")
        
        # Log stats
        stats = semantic_mapper.get_layer_stats()
        if isinstance(semantic_mapper, DualSemanticIDMapper):
            logger.info(f"Dual mapper statistics:")
            logger.info(f"  Behavior vocab: {stats['behavior_stats']['actual_vocab_size']}")
            logger.info(f"  Semantic vocab: {stats['semantic_stats']['actual_vocab_size']}")
            logger.info(f"  Combined vocab: {stats['combined_vocab_size']}")
        else:
            logger.info(f"Vocab size optimization: saved {stats['savings']} token embeddings")
        
        logger.info(f"Semantic ID layers: {num_layers}")
    
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
