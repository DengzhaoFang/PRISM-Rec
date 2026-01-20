#!/usr/bin/env python3
"""
ActionPiece Tokenizer Training Script

This script trains the ActionPiece tokenizer using:
1. Faiss OPQ (Optimized Product Quantization) for feature extraction
2. ActionPiece BPE-like algorithm for vocabulary construction

Based on the paper: "ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation"
"""

import os
import sys
import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm

# Import local ActionPieceCore
from .actionpiece_core import ActionPieceCore


def setup_logging(output_dir: str, log_level: str = "INFO"):
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(output_dir, f"{timestamp}_actionpiece_tokenizer.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode='w', encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_path}")
    return logger


class ActionPieceTokenizerTrainer:
    """Trainer for ActionPiece tokenizer."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # OPQ parameters (from paper)
        self.pq_n_codebooks = config.get('pq_n_codebooks', 4)  # m=4 in paper
        self.pq_codebook_size = config.get('pq_codebook_size', 256)  # 256 codes per codebook
        self.n_hash_buckets = config.get('n_hash_buckets', 128)  # For collision handling
        
        # ActionPiece vocabulary size
        self.vocab_size = config.get('vocab_size', 40000)
        
        # Data
        self.item2feat = {}
        self.item2sem_ids = {}
        
    def load_embeddings(self) -> Tuple[np.ndarray, List[int]]:
        """Load item embeddings from parquet file."""
        embedding_file = os.path.join(
            self.config['data_path'], 
            self.config.get('embedding_file', 'item_emb.parquet')
        )
        self.logger.info(f"Loading embeddings from: {embedding_file}")
        
        df = pd.read_parquet(embedding_file)
        
        # Convert embeddings to numpy array
        embeddings = np.stack(df['embedding'].values).astype(np.float32)
        item_ids = df['ItemID'].values.tolist()
        
        self.logger.info(f"Loaded {len(item_ids)} items with embedding dim {embeddings.shape[1]}")
        return embeddings, item_ids
    
    def load_sequences(self) -> List[List[int]]:
        """Load training sequences from parquet file."""
        train_file = os.path.join(self.config['data_path'], 'train.parquet')
        self.logger.info(f"Loading sequences from: {train_file}")
        
        df = pd.read_parquet(train_file)
        sequences = []
        
        for _, row in df.iterrows():
            history = list(row['history'])
            target = row['target']
            sequence = history + [target]
            sequences.append(sequence)
        
        self.logger.info(f"Loaded {len(sequences)} training sequences")
        return sequences
    
    def train_pq(self, embeddings: np.ndarray, item_ids: List[int]) -> Dict[int, Tuple[int, ...]]:
        """Train OPQ (Optimized Product Quantization) and generate semantic IDs for items.
        
        Uses Faiss OPQ to quantize embeddings into m codes, each from a codebook of size 256.
        This follows the original ActionPiece paper implementation exactly.
        
        Reference: src/action_piece/genrec/models/ActionPiece/tokenizer.py
        
        Args:
            embeddings: Item embeddings, shape (n_items, embedding_dim)
            item_ids: List of item IDs corresponding to embeddings
            
        Returns:
            Dictionary mapping item_id to tuple of PQ codes
        """
        self.logger.info("Training OPQ (Optimized Product Quantization) with Faiss...")
        self.logger.info(f"  Codebooks: {self.pq_n_codebooks}")
        self.logger.info(f"  Codebook size: {self.pq_codebook_size}")
        
        n_samples, dim = embeddings.shape
        self.logger.info(f"  Embedding dim: {dim}")
        self.logger.info(f"  Number of items: {n_samples}")
        
        # Set number of threads for faiss (following original implementation)
        n_threads = self.config.get('n_threads', 4)
        faiss.omp_set_num_threads(n_threads)
        self.logger.info(f"  Using {n_threads} threads")
        
        # Build OPQ index (following original implementation exactly)
        # Format: OPQ{m},IVF1,PQ{m}x{bits}
        # - OPQ{m}: Optimized rotation for m subspaces
        # - IVF1: Single inverted list (no clustering, direct quantization)
        # - PQ{m}x{bits}: Product quantization with m codebooks, each with 2^bits codes
        bits_per_code = int(np.log2(self.pq_codebook_size))
        index_string = f"OPQ{self.pq_n_codebooks},IVF1,PQ{self.pq_n_codebooks}x{bits_per_code}"
        
        self.logger.info(f"  Index string: {index_string}")
        
        # Create index with inner product metric (following original)
        index = faiss.index_factory(
            dim,
            index_string,
            faiss.METRIC_INNER_PRODUCT
        )
        
        # Train the index on all embeddings
        # Note: Original code trains on training items only, but we use all items here
        # since we're training a standalone tokenizer
        self.logger.info("  Training OPQ index...")
        index.train(embeddings)
        
        # Add all embeddings to the index
        self.logger.info("  Adding embeddings to index...")
        index.add(embeddings)
        
        # Extract PQ codes (following original implementation exactly)
        self.logger.info("  Extracting PQ codes...")
        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        codes = codes.reshape(-1, invlists.code_size)
        
        # Create item to semantic ID mapping
        item2sem_ids = {}
        for i, item_id in enumerate(item_ids):
            item2sem_ids[item_id] = tuple(codes[i].tolist())
        
        self.logger.info(f"Generated semantic IDs for {len(item2sem_ids)} items")
        
        # Log statistics
        unique_codes = set(item2sem_ids.values())
        self.logger.info(f"  Unique semantic ID combinations: {len(unique_codes)}")
        self.logger.info(f"  Collision rate: {1 - len(unique_codes) / len(item2sem_ids):.4f}")
        
        return item2sem_ids
    
    def add_hash_buckets(self, item2sem_ids: Dict[int, Tuple[int, ...]]) -> Dict[int, Tuple[int, ...]]:
        """Add hash bucket to handle collisions in semantic IDs.
        
        Items with the same semantic ID get different hash bucket values.
        """
        self.logger.info("Adding hash buckets for collision handling...")
        
        from collections import defaultdict
        
        # Group items by their semantic ID
        sem_id_to_items = defaultdict(list)
        for item_id, sem_id in item2sem_ids.items():
            sem_id_to_items[sem_id].append(item_id)
        
        # Count collisions
        collision_count = sum(1 for items in sem_id_to_items.values() if len(items) > 1)
        max_collision = max(len(items) for items in sem_id_to_items.values())
        self.logger.info(f"  Semantic IDs with collisions: {collision_count}")
        self.logger.info(f"  Maximum collision size: {max_collision}")
        
        if max_collision > self.n_hash_buckets:
            self.logger.warning(
                f"  Maximum collision ({max_collision}) exceeds hash buckets ({self.n_hash_buckets})!"
            )
        
        # Assign hash buckets
        item2hashed_feat = {}
        for sem_id, items in sem_id_to_items.items():
            # Random permutation for hash bucket assignment
            hash_ids = np.random.permutation(self.n_hash_buckets)
            for idx, item_id in enumerate(sorted(items)):
                if idx >= self.n_hash_buckets:
                    self.logger.warning(f"  Item {item_id} exceeds hash bucket limit!")
                    idx = idx % self.n_hash_buckets
                item2hashed_feat[item_id] = sem_id + (hash_ids[idx].item(),)
        
        # Verify no collisions after hashing
        final_feats = set(item2hashed_feat.values())
        self.logger.info(f"  Final unique feature combinations: {len(final_feats)}")
        
        return item2hashed_feat
    
    def train_actionpiece(
        self, 
        item2feat: Dict[int, Tuple[int, ...]], 
        sequences: List[List[int]]
    ) -> ActionPieceCore:
        """Train ActionPiece tokenizer using BPE-like algorithm.
        
        Args:
            item2feat: Mapping from item ID to feature tuple
            sequences: List of item sequences for training
        """
        self.logger.info("Training ActionPiece tokenizer...")
        self.logger.info(f"  Target vocabulary size: {self.vocab_size}")
        
        # Convert item2feat to state2feat format (item_id -> feature tuple)
        state2feat = {str(item_id): list(feat) for item_id, feat in item2feat.items()}
        
        # Initialize ActionPiece
        actionpiece = ActionPieceCore(state2feat=state2feat)
        self.logger.info(f"  Initial vocabulary size: {actionpiece.vocab_size}")
        self.logger.info(f"  Number of categories: {actionpiece.n_categories}")
        
        # Convert sequences to state sequences (list of item_id strings)
        state_corpus = []
        for seq in sequences:
            state_seq = [str(item_id) for item_id in seq if str(item_id) in state2feat]
            if len(state_seq) >= 2:  # Need at least 2 items
                state_corpus.append(state_seq)
        
        self.logger.info(f"  Training corpus size: {len(state_corpus)} sequences")
        
        # Train ActionPiece
        actionpiece.train(
            state_corpus=state_corpus,
            target_vocab_size=self.vocab_size
        )
        
        self.logger.info(f"  Final vocabulary size: {actionpiece.vocab_size}")
        
        return actionpiece
    
    def save_results(
        self, 
        item2feat: Dict[int, Tuple[int, ...]], 
        item2sem_ids: Dict[int, Tuple[int, ...]],
        actionpiece: ActionPieceCore
    ):
        """Save all results to output directory.
        
        Saves in format compatible with existing framework:
        - semantic_id_mappings.json: item_id -> feature codes (with hash bucket for uniqueness)
        """
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Verify uniqueness of item2feat (must be 1-to-1 mapping)
        feat_set = set(item2feat.values())
        if len(feat_set) != len(item2feat):
            self.logger.error(f"CRITICAL: item2feat is not unique! {len(item2feat)} items -> {len(feat_set)} unique features")
            raise ValueError("item2feat must be unique (1-to-1 mapping)")
        self.logger.info(f"✓ Verified: {len(item2feat)} items have unique semantic IDs")
        
        # Save semantic_id_mappings.json (main output, compatible with framework)
        # Format: {"item_id": [code1, code2, code3, code4, hash_bucket], ...}
        semantic_id_mappings_path = os.path.join(output_dir, 'semantic_id_mappings.json')
        with open(semantic_id_mappings_path, 'w') as f:
            json.dump({str(k): list(v) for k, v in item2feat.items()}, f, indent=2)
        self.logger.info(f"Saved semantic_id_mappings.json to: {semantic_id_mappings_path}")
        
        # Save item2feat.json (alias for compatibility with ActionPiece recommender)
        item2feat_path = os.path.join(output_dir, 'item2feat.json')
        with open(item2feat_path, 'w') as f:
            json.dump({str(k): list(v) for k, v in item2feat.items()}, f, indent=2)
        self.logger.info(f"Saved item2feat.json to: {item2feat_path}")
        
        # Save item to semantic ID mapping (without hash bucket, for reference)
        item2sem_ids_path = os.path.join(output_dir, 'item2sem_ids.json')
        with open(item2sem_ids_path, 'w') as f:
            json.dump({str(k): list(v) for k, v in item2sem_ids.items()}, f, indent=2)
        self.logger.info(f"Saved item2sem_ids.json to: {item2sem_ids_path}")
        
        # Save ActionPiece tokenizer
        actionpiece_path = os.path.join(output_dir, 'actionpiece.json')
        actionpiece.save(actionpiece_path)
        self.logger.info(f"Saved ActionPiece tokenizer to: {actionpiece_path}")
        
        # Save config
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        self.logger.info(f"Saved config to: {config_path}")
        
        # Save statistics
        stats = {
            'n_items': len(item2feat),
            'n_unique_semantic_ids': len(feat_set),
            'n_categories': actionpiece.n_categories,
            'vocab_size': actionpiece.vocab_size,
            'n_init_feats': actionpiece.n_init_feats,
            'pq_n_codebooks': self.pq_n_codebooks,
            'pq_codebook_size': self.pq_codebook_size,
            'n_hash_buckets': self.n_hash_buckets
        }
        stats_path = os.path.join(output_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"Saved statistics to: {stats_path}")
    
    def train(self):
        """Main training pipeline."""
        self.logger.info("=" * 60)
        self.logger.info("ActionPiece Tokenizer Training")
        self.logger.info("=" * 60)
        
        # Step 1: Load embeddings
        embeddings, item_ids = self.load_embeddings()
        
        # Step 2: Train PQ and get semantic IDs
        item2sem_ids = self.train_pq(embeddings, item_ids)
        self.item2sem_ids = item2sem_ids
        
        # Step 3: Add hash buckets for collision handling
        item2feat = self.add_hash_buckets(item2sem_ids)
        self.item2feat = item2feat
        
        # Step 4: Load training sequences
        sequences = self.load_sequences()
        
        # Step 5: Train ActionPiece tokenizer
        actionpiece = self.train_actionpiece(item2feat, sequences)
        
        # Step 6: Save results
        self.save_results(item2feat, item2sem_ids, actionpiece)
        
        self.logger.info("=" * 60)
        self.logger.info("Training completed!")
        self.logger.info("=" * 60)
        
        return actionpiece


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ActionPiece Tokenizer")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed dataset directory')
    parser.add_argument('--embedding_file', type=str, default='item_emb.parquet',
                       help='Embedding file name (default: item_emb.parquet)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for trained tokenizer')
    
    # OPQ parameters (paper defaults)
    parser.add_argument('--pq_n_codebooks', type=int, default=4,
                       help='Number of PQ codebooks (default: 4, paper setting)')
    parser.add_argument('--pq_codebook_size', type=int, default=256,
                       help='Size of each codebook (default: 256)')
    parser.add_argument('--n_hash_buckets', type=int, default=128,
                       help='Number of hash buckets for collision handling (default: 128)')
    
    # ActionPiece parameters
    parser.add_argument('--vocab_size', type=int, default=40000,
                       help='Target vocabulary size (default: 40000, paper setting)')
    
    # Other arguments
    parser.add_argument('--n_threads', type=int, default=4,
                       help='Number of threads for Faiss (default: 4)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Setup logging
    logger = setup_logging(args.output_dir, args.log_level)
    
    # Create config
    config = {
        'data_path': args.data_path,
        'embedding_file': args.embedding_file,
        'output_dir': args.output_dir,
        'pq_n_codebooks': args.pq_n_codebooks,
        'pq_codebook_size': args.pq_codebook_size,
        'n_hash_buckets': args.n_hash_buckets,
        'vocab_size': args.vocab_size,
        'n_threads': args.n_threads,
        'seed': args.seed
    }
    
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Train tokenizer
    trainer = ActionPieceTokenizerTrainer(config)
    trainer.train()
    
    print(f"\nTraining completed! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
