"""
ActionPiece Tokenizer module.

This module implements the ActionPiece tokenization approach:
- OPQ (Optimized Product Quantization) for feature extraction
- BPE-like vocabulary construction with weighted co-occurrence
- SPR (Set Permutation Regularization) for data augmentation

Based on the paper: "ActionPiece: Contextually Tokenizing Action Sequences 
for Generative Recommendation"
"""

__version__ = "0.1.0"
