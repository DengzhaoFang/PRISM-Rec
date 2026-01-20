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
import os

logger = logging.getLogger(__name__)


def load_codebook_mappings(mapping_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, List[int]]]:
    """Load codebook vectors and tag IDs from item_codebook_mappings.npz.
    """
    # Handle both file path and directory path
    if os.path.isdir(mapping_path):
        npz_path = os.path.join(mapping_path, 'item_codebook_mappings.npz')
    else:
        npz_path = mapping_path
    
    if not os.path.exists(npz_path):
        logger.warning(f"Codebook mappings not found at {npz_path}, returning empty dicts")
        return {}, {}
    
    logger.info(f"Loading codebook mappings from {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    
    item_ids = data['item_ids']
    codebook_vectors = data['codebook_vectors']  # Shape: (n_items, n_layers, latent_dim)
    predicted_tags = data.get('predicted_tags', np.array([]))  # Shape: (n_items, n_layers)
    
    # Create dictionaries
    codebook_dict = {int(iid): codebook_vectors[i] for i, iid in enumerate(item_ids)}
    
    if predicted_tags.size > 0:
        tag_dict = {int(iid): predicted_tags[i].tolist() for i, iid in enumerate(item_ids)}
    else:
        tag_dict = {}
    
    logger.info(f"Loaded codebook vectors for {len(codebook_dict)} items")
    if tag_dict:
        logger.info(f"Loaded tag IDs for {len(tag_dict)} items")
    
    return codebook_dict, tag_dict


def load_content_embeddings(data_dir: str) -> Dict[int, np.ndarray]:
    """Load content embeddings from parquet, NPZ or NPY file.
    """
    # Try parquet first (PRISM format)
    parquet_path = os.path.join(data_dir, 'item_emb.parquet')
    npz_path = os.path.join(data_dir, 'item_content_embeddings.npz')
    npy_path = os.path.join(data_dir, 'item_content_embeddings.npy')
    
    if os.path.exists(parquet_path):
        logger.info(f"Loading content embeddings from {parquet_path}")
        import pandas as pd
        
        item_df = pd.read_parquet(parquet_path)
        
        # Extract ItemID and embedding (support both 'attribute_embedding' and 'embedding' column names)
        item_ids = item_df['ItemID'].values
        if 'attribute_embedding' in item_df.columns:
            emb_col = 'attribute_embedding'
        elif 'embedding' in item_df.columns:
            emb_col = 'embedding'
        else:
            raise KeyError(f"Expected 'attribute_embedding' or 'embedding' column, found: {item_df.columns.tolist()}")
        
        embeddings = np.stack([np.array(emb) for emb in item_df[emb_col]])
        
        content_dict = {int(iid): embeddings[i] for i, iid in enumerate(item_ids)}
        
        logger.info(f"Loaded parquet format with {emb_col} column")
        
    elif os.path.exists(npz_path):
        logger.info(f"Loading content embeddings from {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        
        item_ids = data['item_ids']
        embeddings = data['embeddings']  # Shape: (n_items, 768)
        
        content_dict = {int(iid): embeddings[i] for i, iid in enumerate(item_ids)}
        
    elif os.path.exists(npy_path):
        logger.info(f"Loading content embeddings from {npy_path}")
        embeddings = np.load(npy_path, allow_pickle=True)  # Shape: (n_items, 768)
        
        # Create mapping: item_id (0-indexed) -> embedding
        content_dict = {i: embeddings[i] for i in range(len(embeddings))}
        
        logger.info(f"Loaded NPY format, assuming item_id = array index")
    else:
        logger.warning(f"Content embeddings not found at {parquet_path}, {npz_path} or {npy_path}, returning empty dict")
        return {}
    
    logger.info(f"Loaded content embeddings for {len(content_dict)} items")
    logger.info(f"  Embedding shape: {next(iter(content_dict.values())).shape}")
    
    return content_dict


def load_collab_embeddings(file_path: str) -> Dict[int, np.ndarray]:
    """Load collaborative embeddings from NPZ or NPY file.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Collaborative embeddings not found at {file_path}, returning empty dict")
        return {}
    
    logger.info(f"Loading collaborative embeddings from {file_path}")
    
    # Check file extension
    if file_path.endswith('.npz'):
        # NPZ format: expects 'item_ids' and 'embeddings' keys
        data = np.load(file_path, allow_pickle=True)
        item_ids = data['item_ids']
        embeddings = data['embeddings']  # Shape: (n_items, emb_dim)
        
        collab_dict = {int(iid): embeddings[i] for i, iid in enumerate(item_ids)}
        
    elif file_path.endswith('.npy'):
        # NPY format: direct array, assume item_id = index
        embeddings = np.load(file_path, allow_pickle=True)  # Shape: (n_items, emb_dim)
        
        # Create mapping: item_id (0-indexed) -> embedding
        collab_dict = {i: embeddings[i] for i in range(len(embeddings))}
        
        logger.info(f"Loaded NPY format, assuming item_id = array index")
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Expected .npz or .npy")
    
    logger.info(f"Loaded collaborative embeddings for {len(collab_dict)} items")
    logger.info(f"  Embedding shape: {next(iter(collab_dict.values())).shape}")
    
    return collab_dict


class SemanticIDMapper:
    """Manages the mapping from item IDs to semantic codes.
    """
    
    def __init__(self, mapping_path: str, codebook_size: int = 256, num_layers: int = 4, pad_token_id: int = 0, codebook_sizes: Optional[List[int]] = None):
        """Initialize the semantic ID mapper.
        """
        self.mapping_path = mapping_path
        self.codebook_size = codebook_size  # Keep for backward compatibility
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.codebook_sizes = codebook_sizes  # Variable sizes per layer
        
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
        
        # Auto-detect variable codebook sizes if not provided
        if self.codebook_sizes is None:
            # Try to detect from data
            detected_sizes = self._detect_codebook_sizes(raw_mapping, num_layers)
            if detected_sizes is not None:
                self.codebook_sizes = detected_sizes
                logger.info(f"Auto-detected variable codebook sizes: {detected_sizes}")
            else:
                # Use uniform size
                self.codebook_sizes = [codebook_size] * num_layers
                logger.info(f"Using uniform codebook size: {codebook_size} for all {num_layers} layers")
        else:
            # Validate provided codebook_sizes
            if len(self.codebook_sizes) != num_layers:
                logger.warning(
                    f"codebook_sizes length ({len(self.codebook_sizes)}) != num_layers ({num_layers}), "
                    f"adjusting..."
                )
                if len(self.codebook_sizes) < num_layers:
                    # Pad with last value
                    self.codebook_sizes = self.codebook_sizes + [self.codebook_sizes[-1]] * (num_layers - len(self.codebook_sizes))
                else:
                    # Truncate
                    self.codebook_sizes = self.codebook_sizes[:num_layers]
            logger.info(f"Using provided codebook sizes: {self.codebook_sizes}")
        
        # STEP 1: First pass - compute layer_max_values
        # We need this before applying offsets for optimization
        layer_max_values = [0] * num_layers
        
        for item_id_str, codes in raw_mapping.items():
            # Handle variable-length: pad shorter codes to max_layers
            if len(codes) < num_layers:
                codes = codes + [pad_token_id] * (num_layers - len(codes))
            
            # Track max values per layer (before offset, excluding padding)
            for i, code in enumerate(codes):
                if code != pad_token_id:  # Don't count padding in max values
                    layer_max_values[i] = max(layer_max_values[i], code)
        
        # Store layer_max_values for use in _apply_offset
        self._layer_max_values = layer_max_values
        
        # STEP 2: Second pass - apply offset transformation
        self.item_to_codes = {}
        
        for item_id_str, codes in raw_mapping.items():
            item_id = int(item_id_str)
            
            # Handle variable-length: pad shorter codes to max_layers
            if len(codes) < num_layers:
                codes = codes + [pad_token_id] * (num_layers - len(codes))
            
            offset_codes = self._apply_offset(codes)
            self.item_to_codes[item_id] = offset_codes
        
        # Calculate actual vocabulary size based on data
        # Formula: max(offset_code) + 1 for each layer, plus 1 for padding
        self._actual_vocab_size = self._compute_vocab_size(layer_max_values)
        
        logger.info(f"Loaded {len(self.item_to_codes)} item-to-code mappings")
        logger.info(f"Layer max values: {layer_max_values}")
        logger.info(f"Computed vocab size: {self._actual_vocab_size}")
        
        # Create padding code
        self.pad_codes = [pad_token_id] * num_layers
    
    def _detect_codebook_sizes(self, raw_mapping: dict, num_layers: int) -> Optional[List[int]]:
        """Auto-detect variable codebook sizes from data.
        """
        layer_max_values = [0] * num_layers
        
        for codes in raw_mapping.values():
            for i, code in enumerate(codes[:num_layers]):
                if code != self.pad_token_id:
                    layer_max_values[i] = max(layer_max_values[i], code)
        
        # Check if sizes are clearly variable (heuristic: >20% difference between layers)
        if len(set(layer_max_values)) > 1:
            max_val = max(layer_max_values)
            min_val = min([v for v in layer_max_values if v > 0])
            if max_val > min_val * 1.2:  # 20% threshold
                # Round up to nearest power of 2 or common size
                detected_sizes = []
                for max_val in layer_max_values:
                    # Find next power of 2 or common size (64, 128, 256, 512, 1024)
                    common_sizes = [64, 128, 256, 512, 1024, 2048]
                    size = next((s for s in common_sizes if s > max_val), max_val + 1)
                    detected_sizes.append(size)
                return detected_sizes
        
        return None
    
    def _apply_offset(self, codes: List[int]) -> List[int]:
        """Apply offset transformation to semantic codes.
        """
        offset_codes = []
        cumulative_offset = 1  # Start from 1 (0 is reserved for PAD)
        
        for i, code in enumerate(codes):
            offset_codes.append(code + cumulative_offset)
            # OPTIMIZATION: Use actual max value + 1 instead of codebook_size
            # This requires layer_max_values to be computed first
            cumulative_offset += (self._layer_max_values[i] + 1)
        
        return offset_codes
    
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
        """
        max_token_id = 0
        cumulative_offset = 1  # Start from 1 (0 is PAD)
        
        for i, layer_max in enumerate(layer_max_values):
            # Calculate the offset token ID for this layer's max value
            offset_token_id = layer_max + cumulative_offset
            max_token_id = max(max_token_id, offset_token_id)
            # OPTIMIZATION: Use actual max value + 1 instead of codebook_size
            # This ensures we only allocate space for tokens that actually exist
            cumulative_offset += (layer_max + 1)
        
        # Add 1 because vocab size is max_id + 1
        return max_token_id + 1
    
    def get_vocab_size(self, use_actual: bool = True) -> int:
        """Get vocabulary size.
        """
        if use_actual:
            return self._actual_vocab_size
        else:
            # Theoretical maximum (may waste some embedding space)
            # For variable codebook sizes: 1 + sum(codebook_sizes)
            return 1 + sum(self.codebook_sizes)
    
    def get_layer_stats(self) -> Dict:
        """Get statistics about each layer.
        """
        theoretical_vocab_size = 1 + sum(self.codebook_sizes)
        
        return {
            'num_layers': self.num_layers,
            'codebook_size': self.codebook_size,  # Keep for backward compatibility
            'codebook_sizes': self.codebook_sizes,  # Variable sizes per layer
            'layer_max_values': self._layer_max_values,
            'actual_vocab_size': self._actual_vocab_size,
            'theoretical_vocab_size': theoretical_vocab_size,
            'savings': theoretical_vocab_size - self._actual_vocab_size
        }


def process_sequence_data(
    file_path: str,
    mode: str,
    max_len: int,
    pad_token_id: int = 0
) -> List[Dict]:
    """Process parquet sequence data.
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
    """
    if len(sequence) > max_len:
        # Truncate from the left (keep recent items)
        return sequence[-max_len:]
    else:
        # Left pad with pad_token_id
        return [pad_token_id] * (max_len - len(sequence)) + sequence


class GenRecDataset(Dataset):
    """Dataset for generative recommendation.
    """
    
    def __init__(
        self,
        sequence_file: str,
        semantic_mapper: SemanticIDMapper,
        mode: str = 'train',
        max_len: int = 20,
        pad_token_id: int = 0,
        # NEW: Multi-source information
        codebook_vectors: Optional[Dict[int, np.ndarray]] = None,
        content_embeddings: Optional[Dict[int, np.ndarray]] = None,
        collab_embeddings: Optional[Dict[int, np.ndarray]] = None,
        tag_ids: Optional[Dict[int, List[int]]] = None,
        use_multimodal: bool = False
    ):
        """Initialize the dataset.
        """
        self.sequence_file = sequence_file
        self.semantic_mapper = semantic_mapper
        self.mode = mode
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        
        # NEW: Store multi-source information
        self.codebook_vectors = codebook_vectors or {}
        self.content_embeddings = content_embeddings or {}
        self.collab_embeddings = collab_embeddings or {}
        self.tag_ids = tag_ids or {}
        self.use_multimodal = use_multimodal
        
        # Determine dimensions from data
        if self.codebook_vectors:
            sample_item = next(iter(self.codebook_vectors.values()))
            self.n_layers = sample_item.shape[0]
            self.latent_dim = sample_item.shape[1]
        else:
            self.n_layers = semantic_mapper.num_layers
            self.latent_dim = 32  # Default
        
        if self.content_embeddings:
            sample_content = next(iter(self.content_embeddings.values()))
            self.content_dim = sample_content.shape[0]
        else:
            self.content_dim = 768  # Default
        
        if self.collab_embeddings:
            sample_collab = next(iter(self.collab_embeddings.values()))
            self.collab_dim = sample_collab.shape[0]
        else:
            self.collab_dim = 64  # Default
        
        # Process sequence data
        self.data = process_sequence_data(
            sequence_file, mode, max_len, pad_token_id
        )
        
        # Convert to semantic codes
        self._convert_to_codes()
        
        logger.info(f"Dataset initialized with {len(self.data)} samples")
        if use_multimodal:
            logger.info(f"  Multimodal features enabled:")
            logger.info(f"    Codebook vectors: {len(self.codebook_vectors)} items")
            logger.info(f"    Content embeddings: {len(self.content_embeddings)} items")
            logger.info(f"    Collab embeddings: {len(self.collab_embeddings)} items")
            logger.info(f"    Tag IDs: {len(self.tag_ids)} items")
    
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
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with semantic codes and optionally multi-source information
        """
        item = self.data[idx]
        
        # Always include item IDs (needed for adaptive temperature and debugging)
        history_item_ids = item['history']  # Original item IDs (before padding)
        target_item_id = item['target']
        
        # Pad history_item_ids to max_len
        history_item_ids_padded = pad_or_truncate(
            history_item_ids, self.max_len, self.pad_token_id
        )
        
        result = {
            'history': item['history_codes'],
            'target': item['target_codes'],
            'history_item_ids': history_item_ids_padded,
            'target_item_id': target_item_id
        }
        
        # Add multi-source information if enabled
        if self.use_multimodal:
            # Get codebook vectors for each item in history
            history_codebook_vecs = []
            for iid in history_item_ids_padded:
                if iid in self.codebook_vectors:
                    history_codebook_vecs.append(self.codebook_vectors[iid])
                else:
                    # Use zero vector for padding or missing items
                    history_codebook_vecs.append(
                        np.zeros((self.n_layers, self.latent_dim), dtype=np.float32)
                    )
            
            # Get target codebook vectors
            if target_item_id in self.codebook_vectors:
                target_codebook_vecs = self.codebook_vectors[target_item_id]
            else:
                target_codebook_vecs = np.zeros((self.n_layers, self.latent_dim), dtype=np.float32)
            
            # Get content embeddings
            history_content_embs = []
            for iid in history_item_ids_padded:
                if iid in self.content_embeddings:
                    history_content_embs.append(self.content_embeddings[iid])
                else:
                    history_content_embs.append(np.zeros(self.content_dim, dtype=np.float32))
            
            if target_item_id in self.content_embeddings:
                target_content_emb = self.content_embeddings[target_item_id]
            else:
                target_content_emb = np.zeros(self.content_dim, dtype=np.float32)
            
            # Get collaborative embeddings
            history_collab_embs = []
            for iid in history_item_ids_padded:
                if iid in self.collab_embeddings:
                    history_collab_embs.append(self.collab_embeddings[iid])
                else:
                    history_collab_embs.append(np.zeros(self.collab_dim, dtype=np.float32))
            
            if target_item_id in self.collab_embeddings:
                target_collab_emb = self.collab_embeddings[target_item_id]
            else:
                target_collab_emb = np.zeros(self.collab_dim, dtype=np.float32)
            
            # Get tag IDs
            history_tag_ids = []
            for iid in history_item_ids_padded:
                if iid in self.tag_ids:
                    history_tag_ids.append(self.tag_ids[iid])
                else:
                    history_tag_ids.append([0] * self.n_layers)  # Padding tags
            
            if target_item_id in self.tag_ids:
                target_tag_ids = self.tag_ids[target_item_id]
            else:
                target_tag_ids = [0] * self.n_layers
            
            # Add multimodal data to result
            result.update({
                'history_codebook_vecs': np.array(history_codebook_vecs, dtype=np.float32),  # (max_len, n_layers, latent_dim)
                'target_codebook_vecs': target_codebook_vecs.astype(np.float32),  # (n_layers, latent_dim)
                'history_content_embs': np.array(history_content_embs, dtype=np.float32),  # (max_len, content_dim)
                'target_content_emb': target_content_emb.astype(np.float32),  # (content_dim,)
                'history_collab_embs': np.array(history_collab_embs, dtype=np.float32),  # (max_len, collab_dim)
                'target_collab_emb': target_collab_emb.astype(np.float32),  # (collab_dim,)
                'history_tag_ids': history_tag_ids,  # List of lists
                'target_tag_ids': target_tag_ids  # List
            })
        else:
            # Even if multimodal fusion is disabled, we still need to load
            # codebook vectors and tag IDs for auxiliary prediction tasks
            # CRITICAL FIX: Load target data for auxiliary tasks even without fusion
            
            # Get target codebook vectors (for codebook prediction task)
            if target_item_id in self.codebook_vectors:
                target_codebook_vecs = self.codebook_vectors[target_item_id]
            else:
                target_codebook_vecs = np.zeros((self.n_layers, self.latent_dim), dtype=np.float32)
            
            # Get target tag IDs (for tag prediction task)
            if target_item_id in self.tag_ids:
                target_tag_ids = self.tag_ids[target_item_id]
            else:
                target_tag_ids = [0] * self.n_layers
            
            # Add auxiliary task data to result
            result.update({
                'target_codebook_vecs': target_codebook_vecs.astype(np.float32),  # (n_layers, latent_dim)
                'target_tag_ids': target_tag_ids  # List
            })
        
        return result
    
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
    model_config: Optional[any] = None,
    codebook_sizes: Optional[List[int]] = None,
    # NEW: Multi-source information paths
    collab_embedding_path: Optional[str] = None,
    use_multimodal: bool = False
) -> Tuple[GenRecDataset, GenRecDataset, GenRecDataset, SemanticIDMapper]:
    """Create train, validation, and test datasets.
    """
    # Create semantic mapper (shared across all datasets)
    semantic_mapper = SemanticIDMapper(
        semantic_mapping_path,
        codebook_size=codebook_size,
        num_layers=num_layers,
        pad_token_id=pad_token_id,
        codebook_sizes=codebook_sizes
    )
    
    # NEW: Load multi-source information if enabled
    codebook_vectors_dict = {}
    content_embeddings_dict = {}
    collab_embeddings_dict = {}
    tag_ids_dict = {}
    
    # Load auxiliary task data (codebook vectors and tags)
    # These should be loaded regardless of use_multimodal setting
    # because they're needed for auxiliary prediction tasks
    semantic_mapping_dir = os.path.dirname(semantic_mapping_path)
    codebook_vectors_dict, tag_ids_dict = load_codebook_mappings(semantic_mapping_dir)
    logger.info(f"Loaded auxiliary task data: {len(codebook_vectors_dict)} codebook vectors, {len(tag_ids_dict)} tag mappings")
    
    if use_multimodal:
        logger.info("Loading multi-source information for fusion...")
        
        # Load content embeddings from sequence_data_dir
        content_embeddings_dict = load_content_embeddings(sequence_data_dir)
        
        # Load collaborative embeddings if path provided
        if collab_embedding_path:
            collab_embeddings_dict = load_collab_embeddings(collab_embedding_path)
        
        logger.info("Multi-source information loaded successfully")
    
    # Check if num_layers was auto-adjusted (for variable-length IDs)
    if semantic_mapper.num_layers != num_layers:
        logger.warning(f"⚠ num_layers auto-adjusted: {num_layers} → {semantic_mapper.num_layers}")
        logger.warning(f"⚠ This is expected for Prism variable-length IDs")
        num_layers = semantic_mapper.num_layers
    
    # Update model config with actual vocab size if provided
    if model_config is not None:
        actual_vocab_size = semantic_mapper.get_vocab_size(use_actual=True)
        
        # If tag data is available, compute tag statistics and update config
        # This is needed for tag prediction task, independent of multimodal fusion
        num_tag_tokens = 0
        if tag_ids_dict:
            # Get max tag ID for each layer
            max_tag_ids = [0] * num_layers
            for tag_list in tag_ids_dict.values():
                for layer_idx, tag_id in enumerate(tag_list[:num_layers]):
                    if tag_id > 0:  # Ignore padding
                        max_tag_ids[layer_idx] = max(max_tag_ids[layer_idx], tag_id)
            
            # Total tag tokens needed (sum of max_tag_id + 1 for each layer)
            num_tag_tokens = sum(max_id + 1 for max_id in max_tag_ids)
            
            logger.info(f"Tag token statistics:")
            for layer_idx, max_id in enumerate(max_tag_ids):
                logger.info(f"  Layer {layer_idx + 1}: max_tag_id={max_id}, tokens_needed={max_id + 1}")
            logger.info(f"  Total tag tokens: {num_tag_tokens}")
            
            # Store tag token offset (where tag tokens start in vocab)
            model_config.tag_token_offset = actual_vocab_size
            model_config.num_tag_tokens = num_tag_tokens
            model_config.max_tag_ids_per_layer = max_tag_ids
            
            # Extend vocab size
            extended_vocab_size = actual_vocab_size + num_tag_tokens
            model_config.set_vocab_size(extended_vocab_size)
            logger.info(f"Extended vocab_size: {actual_vocab_size} (semantic) + {num_tag_tokens} (tags) = {extended_vocab_size}")
        else:
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
        pad_token_id=pad_token_id,
        codebook_vectors=codebook_vectors_dict,
        content_embeddings=content_embeddings_dict,
        collab_embeddings=collab_embeddings_dict,
        tag_ids=tag_ids_dict,
        use_multimodal=use_multimodal
    )
    
    valid_dataset = GenRecDataset(
        sequence_file=str(data_dir / "valid.parquet"),
        semantic_mapper=semantic_mapper,
        mode='evaluation',
        max_len=max_len,
        pad_token_id=pad_token_id,
        codebook_vectors=codebook_vectors_dict,
        content_embeddings=content_embeddings_dict,
        collab_embeddings=collab_embeddings_dict,
        tag_ids=tag_ids_dict,
        use_multimodal=use_multimodal
    )
    
    test_dataset = GenRecDataset(
        sequence_file=str(data_dir / "test.parquet"),
        semantic_mapper=semantic_mapper,
        mode='evaluation',
        max_len=max_len,
        pad_token_id=pad_token_id,
        codebook_vectors=codebook_vectors_dict,
        content_embeddings=content_embeddings_dict,
        collab_embeddings=collab_embeddings_dict,
        tag_ids=tag_ids_dict,
        use_multimodal=use_multimodal
    )
    
    return train_dataset, valid_dataset, test_dataset, semantic_mapper

