"""
Trie-Constrained Decoder for Prism Recommender.

Implements a Trie-based constrained decoding strategy that ensures
every decoding step points to a path that can lead to a real item.

Key Features:
- Efficient Trie structure for O(1) path validation
- Seamless integration with HuggingFace's beam search
- Minimal overhead during generation
- Supports variable-length semantic IDs
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrieNode:
    """Node in the Trie structure for semantic ID paths."""
    
    __slots__ = ['children', 'is_end', 'item_id']
    
    def __init__(self):
        self.children: Dict[int, 'TrieNode'] = {}  # token_id -> TrieNode
        self.is_end: bool = False  # True if this node represents a complete item
        self.item_id: Optional[int] = None  # Item ID if is_end is True
    
    def add_child(self, token_id: int) -> 'TrieNode':
        """Add a child node for the given token ID."""
        if token_id not in self.children:
            self.children[token_id] = TrieNode()
        return self.children[token_id]
    
    def get_child(self, token_id: int) -> Optional['TrieNode']:
        """Get child node for the given token ID."""
        return self.children.get(token_id)
    
    def get_valid_tokens(self) -> Set[int]:
        """Get all valid token IDs that can follow this node."""
        return set(self.children.keys())


class SemanticIDTrie:
    """Trie structure for efficient semantic ID path validation.
    
    This Trie stores all valid semantic ID sequences (paths to real items).
    During decoding, it ensures that each generated token leads to a valid path.
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.num_items = 0
        self.max_depth = 0
    
    def insert(self, token_sequence: List[int], item_id: int):
        """Insert a semantic ID sequence into the Trie.
        
        Args:
            token_sequence: List of token IDs representing a semantic ID
            item_id: The item ID this sequence represents
        """
        node = self.root
        for token_id in token_sequence:
            node = node.add_child(token_id)
        
        node.is_end = True
        node.item_id = item_id
        self.num_items += 1
        self.max_depth = max(self.max_depth, len(token_sequence))
    
    def get_valid_tokens_at_depth(self, prefix: List[int]) -> Set[int]:
        """Get all valid tokens that can follow the given prefix.
        
        Args:
            prefix: List of token IDs generated so far
        
        Returns:
            Set of valid token IDs that can extend this prefix
        """
        node = self.root
        
        # Navigate to the node corresponding to the prefix
        for token_id in prefix:
            node = node.get_child(token_id)
            if node is None:
                # Invalid prefix - return empty set
                return set()
        
        # Return all valid next tokens
        return node.get_valid_tokens()
    
    def is_valid_sequence(self, token_sequence: List[int]) -> bool:
        """Check if a token sequence represents a valid item.
        
        Args:
            token_sequence: List of token IDs
        
        Returns:
            True if this sequence leads to a real item
        """
        node = self.root
        for token_id in token_sequence:
            node = node.get_child(token_id)
            if node is None:
                return False
        return node.is_end
    
    def get_item_id(self, token_sequence: List[int]) -> Optional[int]:
        """Get the item ID for a complete token sequence.
        
        Args:
            token_sequence: List of token IDs
        
        Returns:
            Item ID if sequence is valid and complete, None otherwise
        """
        node = self.root
        for token_id in token_sequence:
            node = node.get_child(token_id)
            if node is None:
                return None
        return node.item_id if node.is_end else None
    
    @classmethod
    def from_semantic_mapper(cls, semantic_mapper) -> 'SemanticIDTrie':
        """Build a Trie from a SemanticIDMapper.
        
        Args:
            semantic_mapper: SemanticIDMapper instance with item_to_codes mapping
        
        Returns:
            Constructed SemanticIDTrie
        """
        trie = cls()
        
        logger.info("Building Trie from semantic ID mappings...")
        for item_id, token_sequence in semantic_mapper.item_to_codes.items():
            # Filter out padding tokens (assuming 0 is padding)
            valid_tokens = [t for t in token_sequence if t != 0]
            if valid_tokens:
                trie.insert(valid_tokens, item_id)
        
        logger.info(f"Trie built: {trie.num_items} items, max depth: {trie.max_depth}")
        return trie


class TrieConstrainedLogitsProcessor:
    """Logits processor that enforces Trie-based constraints during generation.
    
    This processor modifies the logits at each decoding step to ensure
    only valid tokens (those leading to real items) can be selected.
    
    Compatible with HuggingFace's generation API.
    
    OPTIMIZED with caching for faster inference.
    """
    
    def __init__(
        self,
        trie: SemanticIDTrie,
        pad_token_id: int = 0,
        eos_token_id: int = 0,
        num_beams: int = 1,
        batch_size: int = 1
    ):
        """Initialize the logits processor.
        
        Args:
            trie: SemanticIDTrie instance
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            num_beams: Number of beams for beam search
            batch_size: Batch size
        """
        self.trie = trie
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.num_beams = num_beams
        self.batch_size = batch_size
        
        # Track the prefix for each beam
        # Key: (batch_idx, beam_idx), Value: List[int] (prefix tokens)
        self.beam_prefixes: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        
        # OPTIMIZATION: Cache valid tokens for each prefix
        # Key: tuple(prefix), Value: Set[int] (valid tokens)
        self._valid_tokens_cache: Dict[Tuple[int, ...], Set[int]] = {}
        
        # OPTIMIZATION: Cache mask tensors for each prefix
        # Key: tuple(prefix), Value: torch.BoolTensor (mask)
        self._mask_cache: Dict[Tuple[int, ...], torch.BoolTensor] = {}
        
        # OPTIMIZATION: Pre-compute valid tokens for root (first token)
        self._root_valid_tokens = self.trie.root.get_valid_tokens()
        
        # Store vocab_size for mask creation
        self._vocab_size = None
        self._device = None
        
        logger.info(f"TrieConstrainedLogitsProcessor initialized with caching (root has {len(self._root_valid_tokens)} valid tokens)")
    
    def _get_valid_tokens_cached(self, prefix: List[int]) -> Set[int]:
        """Get valid tokens with caching for performance.
        
        Args:
            prefix: List of token IDs
        
        Returns:
            Set of valid token IDs
        """
        # Fast path: empty prefix (first token)
        if not prefix:
            return self._root_valid_tokens
        
        # Convert to tuple for hashing
        prefix_tuple = tuple(prefix)
        
        # Check cache
        if prefix_tuple in self._valid_tokens_cache:
            return self._valid_tokens_cache[prefix_tuple]
        
        # Compute and cache
        valid_tokens = self.trie.get_valid_tokens_at_depth(prefix)
        self._valid_tokens_cache[prefix_tuple] = valid_tokens
        
        return valid_tokens
    
    def _get_mask_for_prefix(self, prefix_tuple: Tuple[int, ...], vocab_size: int, device: torch.device) -> torch.BoolTensor:
        """Get or create mask tensor for a prefix.
        
        Args:
            prefix_tuple: Tuple of token IDs
            vocab_size: Vocabulary size
            device: Device for tensor
        
        Returns:
            Boolean mask tensor (True = invalid, False = valid)
        """
        # Check cache
        if prefix_tuple in self._mask_cache:
            cached_mask = self._mask_cache[prefix_tuple]
            # Ensure device matches
            if cached_mask.device == device:
                return cached_mask
        
        # Create new mask
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        
        # Get valid tokens
        if not prefix_tuple:
            valid_tokens = self._root_valid_tokens
        else:
            valid_tokens = self._get_valid_tokens_cached(list(prefix_tuple))
        
        if not valid_tokens:
            # No valid continuation - only EOS is valid
            mask[self.eos_token_id] = False
        else:
            # Mark valid tokens
            valid_token_list = list(valid_tokens)
            mask[valid_token_list] = False
        
        # Cache the mask
        self._mask_cache[prefix_tuple] = mask
        
        return mask
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Process logits to enforce Trie constraints.
        
        HIGHLY OPTIMIZED VERSION: Uses pre-computed mask tensors and caching.
        
        Args:
            input_ids: Generated token IDs so far, shape (batch_size * num_beams, seq_len)
            scores: Logits for next token, shape (batch_size * num_beams, vocab_size)
        
        Returns:
            Modified logits with invalid tokens masked out
        """
        batch_beam_size, vocab_size = scores.shape
        device = scores.device
        
        # Initialize vocab_size and device on first call
        if self._vocab_size is None:
            self._vocab_size = vocab_size
            self._device = device
        
        # Collect masks for all beams
        masks = []
        for beam_idx in range(batch_beam_size):
            # Extract the prefix (skip decoder start token)
            prefix = input_ids[beam_idx, 1:].tolist()
            prefix_tuple = tuple(prefix)
            
            # Get cached mask
            mask = self._get_mask_for_prefix(prefix_tuple, vocab_size, device)
            masks.append(mask)
        
        # Stack masks and apply in one operation
        full_mask = torch.stack(masks, dim=0)  # (batch_beam_size, vocab_size)
        scores = scores.masked_fill(full_mask, float('-inf'))
        
        return scores
    
    def reset(self):
        """Reset the processor state (for new generation)."""
        self.beam_prefixes.clear()
        # Keep the cache across batches for better performance
        # Only clear if memory becomes an issue
        # self._valid_tokens_cache.clear()


def create_trie_constrained_generation(
    model: nn.Module,
    semantic_mapper,
    num_beams: int = 20,
    max_length: int = 5
) -> Tuple[nn.Module, TrieConstrainedLogitsProcessor]:
    """Create a model with Trie-constrained generation.
    
    Args:
        model: TIGER model instance
        semantic_mapper: SemanticIDMapper instance
        num_beams: Number of beams for beam search
        max_length: Maximum generation length
    
    Returns:
        Tuple of (model, logits_processor)
    """
    # Build Trie from semantic mapper
    trie = SemanticIDTrie.from_semantic_mapper(semantic_mapper)
    
    # Create logits processor
    logits_processor = TrieConstrainedLogitsProcessor(
        trie=trie,
        pad_token_id=model.config.pad_token_id,
        eos_token_id=model.config.eos_token_id,
        num_beams=num_beams
    )
    
    logger.info(f"Trie-constrained generation enabled with {num_beams} beams")
    
    return model, logits_processor


class TrieConstrainedBeamSearchScorer:
    """Custom beam search scorer with Trie constraints.
    
    This is an alternative approach that integrates Trie constraints
    directly into the beam search scoring mechanism.
    """
    
    def __init__(
        self,
        trie: SemanticIDTrie,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: float = 1.0,
        do_early_stopping: bool = False
    ):
        """Initialize the beam search scorer.
        
        Args:
            trie: SemanticIDTrie instance
            batch_size: Batch size
            num_beams: Number of beams
            device: Device for tensors
            length_penalty: Length penalty for beam search
            do_early_stopping: Whether to stop early when all beams are complete
        """
        self.trie = trie
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        
        # Track beam states
        self.beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device
        )
        self.beam_tokens = [[] for _ in range(batch_size * num_beams)]
        self.is_done = torch.zeros(
            (batch_size, num_beams), dtype=torch.bool, device=device
        )
    
    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        """Process beam search step with Trie constraints.
        
        Args:
            input_ids: Current token IDs
            next_scores: Scores for next tokens
            next_tokens: Next token candidates
            next_indices: Beam indices for next tokens
        
        Returns:
            Tuple of (filtered_scores, filtered_tokens, filtered_indices)
        """
        # Apply Trie constraints to filter invalid beams
        # This is called during beam search to prune invalid paths
        
        batch_beam_size = input_ids.shape[0]
        vocab_size = next_scores.shape[-1]
        
        # For each beam, check if next tokens are valid
        valid_mask = torch.zeros_like(next_scores, dtype=torch.bool)
        
        for beam_idx in range(batch_beam_size):
            prefix = input_ids[beam_idx, 1:].tolist()  # Skip decoder start token
            valid_tokens = self.trie.get_valid_tokens_at_depth(prefix)
            
            for token_id in valid_tokens:
                valid_mask[beam_idx, token_id] = True
        
        # Mask out invalid tokens
        next_scores = next_scores.masked_fill(~valid_mask, float('-inf'))
        
        return next_scores, next_tokens, next_indices
    
    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        max_length: int
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """Finalize beam search and return best sequences.
        
        Args:
            input_ids: Generated token IDs
            final_beam_scores: Final scores for each beam
            final_beam_tokens: Final tokens for each beam
            final_beam_indices: Final beam indices
            max_length: Maximum sequence length
        
        Returns:
            Tuple of (best_sequences, best_scores)
        """
        # Select best beams based on scores
        # Ensure selected sequences are valid in Trie
        
        batch_size = self.batch_size
        num_beams = self.num_beams
        
        best_sequences = []
        best_scores = []
        
        for batch_idx in range(batch_size):
            batch_beams = []
            batch_scores = []
            
            for beam_idx in range(num_beams):
                global_idx = batch_idx * num_beams + beam_idx
                sequence = input_ids[global_idx, 1:].tolist()  # Skip decoder start
                
                # Verify sequence is valid
                if self.trie.is_valid_sequence(sequence):
                    batch_beams.append(sequence)
                    batch_scores.append(final_beam_scores[global_idx].item())
            
            if batch_beams:
                # Sort by score and take best
                sorted_indices = sorted(
                    range(len(batch_scores)),
                    key=lambda i: batch_scores[i],
                    reverse=True
                )
                best_sequences.append(batch_beams[sorted_indices[0]])
                best_scores.append(batch_scores[sorted_indices[0]])
            else:
                # Fallback: return empty or first beam
                logger.warning(f"Batch {batch_idx}: No valid sequences found")
                best_sequences.append([])
                best_scores.append(float('-inf'))
        
        return torch.tensor(best_sequences, device=self.device), torch.tensor(best_scores, device=self.device)


def apply_trie_constraints_to_logits(
    logits: torch.FloatTensor,
    input_ids: torch.LongTensor,
    trie: SemanticIDTrie,
    pad_token_id: int = 0
) -> torch.FloatTensor:
    """Apply Trie constraints to logits (functional API).
    
    This is a functional version of the logits processor for easier integration.
    
    Args:
        logits: Logits tensor, shape (batch_size, vocab_size)
        input_ids: Generated token IDs so far, shape (batch_size, seq_len)
        trie: SemanticIDTrie instance
        pad_token_id: Padding token ID
    
    Returns:
        Constrained logits with invalid tokens masked
    """
    batch_size, vocab_size = logits.shape
    constrained_logits = logits.clone()
    
    for batch_idx in range(batch_size):
        # Extract prefix (skip decoder start token)
        prefix = input_ids[batch_idx, 1:].tolist()
        
        # Get valid tokens
        valid_tokens = trie.get_valid_tokens_at_depth(prefix)
        
        if not valid_tokens:
            # No valid continuation - mask all except EOS
            constrained_logits[batch_idx, :] = float('-inf')
            constrained_logits[batch_idx, pad_token_id] = 0.0
        else:
            # Mask invalid tokens
            mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
            for token_id in valid_tokens:
                mask[token_id] = False
            constrained_logits[batch_idx, mask] = float('-inf')
    
    return constrained_logits


