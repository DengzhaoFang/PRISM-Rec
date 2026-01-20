"""
Trie implementation for constrained generation.

Ensures that generated semantic IDs are valid according to the codebook.
"""
import torch
from typing import List, Dict, Optional, Iterable

class Trie:
    """Prefix tree for valid semantic ID sequences."""
    
    def __init__(self, sequences: Iterable[List[int]] = None):
        """Initialize the Trie.
        
        Args:
            sequences: Optional list of sequences to insert
        """
        self.trie_dict = {}
        self.all_first_tokens = set()  # Cache first tokens for efficiency
        if sequences:
            for seq in sequences:
                self.insert(seq)
                
    def insert(self, sequence: List[int]):
        """Insert a sequence into the Trie.
        
        Args:
            sequence: List of token IDs
        """
        if len(sequence) == 0:
            return
            
        # Track first tokens
        self.all_first_tokens.add(sequence[0])
        
        node = self.trie_dict
        for token in sequence:
            if token not in node:
                node[token] = {}
            node = node[token]
        # Mark end of sequence
        node['__end__'] = True
        
    def get_next_tokens(self, prefix: List[int]) -> List[int]:
        """Get valid next tokens for a given prefix.
        
        Args:
            prefix: List of token IDs
            
        Returns:
            List of valid next token IDs
        """
        node = self.trie_dict
        for token in prefix:
            if token not in node:
                return []
            node = node[token]
            
        # Return all keys except internal markers
        return [k for k in node.keys() if k != '__end__']
    
    def is_valid_sequence(self, sequence: List[int]) -> bool:
        """Check if a sequence is a valid complete sequence in the Trie.
        
        Args:
            sequence: List of token IDs
            
        Returns:
            True if sequence exists and is marked as complete
        """
        node = self.trie_dict
        for token in sequence:
            if token not in node:
                return False
            node = node[token]
        return '__end__' in node
    
    def get_all_first_tokens(self) -> List[int]:
        """Get all valid first tokens.
        
        Returns:
            List of valid first token IDs
        """
        return list(self.all_first_tokens)


def get_prefix_allowed_tokens_fn(trie: Trie, eos_token_id: int = 0):
    """Create a function for HuggingFace generate's prefix_allowed_tokens_fn.
    
    This function constrains beam search to only generate valid semantic ID sequences
    that exist in the Trie (i.e., correspond to actual items).
    
    Args:
        trie: Populated Trie instance with all valid semantic ID sequences
        eos_token_id: End of sequence token ID (default 0, same as pad_token_id)
        
    Returns:
        Function that takes (batch_id, input_ids) and returns allowed tokens
    """
    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
        """
        Determine allowed next tokens based on the current generation prefix.
        
        Args:
            batch_id: Index of the current batch item (used by HuggingFace internally)
            input_ids: Current generated sequence (seq_len,) - includes decoder_start_token
            
        Returns:
            List of allowed next token IDs
        """
        # Convert tensor to list
        if input_ids.dim() > 1:
            prefix = input_ids[0].tolist()  # Take first if batched
        else:
            prefix = input_ids.tolist()
        
        # T5 decoder starts with decoder_start_token_id (usually pad_token_id = 0)
        # The first token in input_ids is this start token, not part of our semantic IDs
        # We need to skip it for Trie lookup
        
        # Remove the decoder start token (first token, which is 0)
        # Our semantic codes are the tokens after the start token
        if len(prefix) > 0 and prefix[0] == eos_token_id:
            semantic_prefix = prefix[1:]  # Skip decoder start token
        else:
            semantic_prefix = prefix
        
        # Also filter out any padding tokens that might appear
        semantic_prefix = [t for t in semantic_prefix if t != eos_token_id]
        
        # Get allowed next tokens from Trie
        allowed = trie.get_next_tokens(semantic_prefix)
        
        # If we're at the start (no semantic tokens yet), return all first tokens
        if len(semantic_prefix) == 0:
            allowed = trie.get_all_first_tokens()
        
        # Check if current prefix is a complete valid sequence
        # If so, also allow EOS token to end generation
        if len(semantic_prefix) > 0:
            node = trie.trie_dict
            valid_path = True
            for token in semantic_prefix:
                if token in node:
                    node = node[token]
                else:
                    valid_path = False
                    break
            
            if valid_path and '__end__' in node:
                # We've reached a valid complete sequence, allow EOS
                if eos_token_id not in allowed:
                    allowed.append(eos_token_id)
        
        # Safety: if no tokens are allowed (shouldn't happen with proper Trie),
        # allow EOS to prevent generation from hanging
        if len(allowed) == 0:
            allowed = [eos_token_id]
        
        return allowed

    return prefix_allowed_tokens_fn
