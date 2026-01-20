#!/usr/bin/env python3
"""
EAGER Dual-Path Semantic ID Tokenizer Training Script

Implements hierarchical K-means clustering for both behavior and semantic paths.
Generates independent semantic IDs for each path and computes collision statistics.

Reference: EAGER paper - Two-Stream Generative Recommender
"""

import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Optional, List
from Hierarchical_KMeans import HierarchicalKMeans


class DualEmbeddingDataset(Dataset):
    """Dataset for loading both collaborative and semantic embeddings"""
    
    def __init__(
        self,
        semantic_embedding_file: str,
        cf_embedding_file: Optional[str] = None,
        max_items: Optional[int] = None
    ):
        """
        Initialize dual embedding dataset.
        
        Args:
            semantic_embedding_file: Path to semantic embeddings (e.g., item_emb.parquet)
            cf_embedding_file: Path to collaborative embeddings (e.g., lightgcn embeddings)
            max_items: Maximum number of items to load
        """
        # Load semantic embeddings
        self.semantic_df = pd.read_parquet(semantic_embedding_file)
        
        # Load collaborative embeddings if provided
        self.has_cf = cf_embedding_file is not None
        if self.has_cf:
            # Load CF embeddings (assumed to be .npy file with shape [n_items, emb_dim])
            cf_emb_array = np.load(cf_embedding_file)
            
            # Check if we need to align the items
            n_semantic = len(self.semantic_df)
            n_cf = len(cf_emb_array)
            
            if n_cf != n_semantic:
                logger = logging.getLogger(__name__)
                logger.warning(f"Mismatch in embedding counts: CF={n_cf}, Semantic={n_semantic}")
                
                # Assume CF embeddings are indexed by ItemID (0 to max_item_id)
                # and semantic_df has ItemID column
                # We'll align by taking only items present in semantic_df
                item_ids = self.semantic_df['ItemID'].values
                
                # Filter CF embeddings to match semantic embeddings
                # Assuming CF embeddings are indexed by ItemID
                if n_cf >= max(item_ids) + 1:
                    # CF has enough items, select by index
                    cf_emb_array = cf_emb_array[item_ids]
                    logger.info(f"Aligned CF embeddings: selected {len(item_ids)} items")
                else:
                    raise ValueError(
                        f"CF embeddings ({n_cf}) has fewer items than max ItemID ({max(item_ids)}). "
                        "Cannot align embeddings."
                    )
            
            self.cf_embeddings = torch.tensor(cf_emb_array, dtype=torch.float32)
        else:
            self.cf_embeddings = None
        
        # Apply max_items limit after alignment
        if max_items is not None:
            self.semantic_df = self.semantic_df.head(max_items)
            if self.has_cf:
                self.cf_embeddings = self.cf_embeddings[:max_items]
        
        # Convert semantic embeddings to tensor
        self.semantic_embeddings = torch.stack([
            torch.tensor(emb, dtype=torch.float32)
            for emb in self.semantic_df['attribute_embedding']
        ])
        
        self.item_ids = self.semantic_df['ItemID'].values
    
    def __len__(self):
        return len(self.semantic_embeddings)
    
    def __getitem__(self, idx):
        item = {
            'item_id': self.item_ids[idx],
            'semantic_embedding': self.semantic_embeddings[idx]
        }
        
        if self.has_cf:
            item['cf_embedding'] = self.cf_embeddings[idx]
        
        return item


class EAGERTokenizerTrainer:
    """Trainer for EAGER dual-path HKM tokenizer"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        logs_dir = self.config['output_dir']
        os.makedirs(logs_dir, exist_ok=True)
        
        data_path = self.config['data_path']
        dataset_name = os.path.basename(data_path.rstrip('/'))
        if not dataset_name:
            dataset_name = os.path.basename(os.path.dirname(data_path))
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        k = self.config.get('hkm_k', 8)
        depth = self.config.get('hkm_depth', 3)
        
        log_filename = f"{timestamp}_{dataset_name}_eager_k{k}_d{depth}.log"
        log_path = os.path.join(logs_dir, log_filename)
        
        logging.getLogger().handlers.clear()
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, mode='w', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Logging to file: {log_path}")
        self.logger.info(f"Dataset: {dataset_name}, Mode: EAGER HKM, K: {k}, Depth: {depth}")
    
    def load_data(self) -> DataLoader:
        """Load and prepare dual embedding data"""
        self.logger.info(f"Loading embeddings from: {self.config['data_path']}")
        
        semantic_file = os.path.join(
            self.config['data_path'],
            self.config.get('embedding_file', 'item_emb.parquet')
        )
        
        cf_file = None
        if self.config.get('cf_embedding_file'):
            # CF embedding file should be relative to data_path or absolute
            cf_path = self.config['cf_embedding_file']
            if not os.path.isabs(cf_path):
                cf_file = os.path.join(self.config['data_path'], cf_path)
            else:
                cf_file = cf_path
            self.logger.info(f"Loading collaborative embeddings from: {cf_file}")
        
        dataset = DualEmbeddingDataset(
            semantic_embedding_file=semantic_file,
            cf_embedding_file=cf_file,
            max_items=self.config.get('max_items')
        )
        
        # For HKM, we don't need batching during training
        # Just use the entire dataset
        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),  # Load all at once
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.logger.info(f"Loaded {len(dataset)} items")
        self.logger.info(f"  Semantic embedding dim: {dataset.semantic_embeddings.shape[1]}")
        if dataset.has_cf:
            self.logger.info(f"  CF embedding dim: {dataset.cf_embeddings.shape[1]}")
        
        return dataloader
    
    def train_hkm(
        self,
        embeddings: torch.Tensor,
        path_name: str
    ) -> HierarchicalKMeans:
        """
        Train a single HKM model on embeddings.
        
        Args:
            embeddings: Input embeddings (n_items, embedding_dim)
            path_name: Name of the path ('behavior' or 'semantic')
            
        Returns:
            Trained HKM model
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Training HKM for {path_name.upper()} path")
        self.logger.info(f"{'='*60}")
        
        # Create HKM model
        model = HierarchicalKMeans(
            k=self.config.get('hkm_k', 8),
            max_depth=self.config.get('hkm_depth', 3),
            random_state=self.config.get('random_state', 42),
            n_init=self.config.get('hkm_n_init', 10),
            max_iter=self.config.get('hkm_max_iter', 300),
            device=self.device
        )
        
        # Train the model
        model.fit(embeddings)
        
        # Get collision statistics
        stats = model.get_collision_stats()
        
        self.logger.info(f"\n{path_name.upper()} Path Collision Statistics:")
        self.logger.info(f"  Total items: {stats['total_items']}")
        self.logger.info(f"  Branching factor (K): {stats['branching_factor']}")
        self.logger.info(f"  Tree depth: {stats['max_depth']}")
        
        for level, level_stats in stats['levels'].items():
            self.logger.info(f"\n  Level {level}:")
            self.logger.info(f"    Unique codes: {level_stats['unique_codes']}/{level_stats['theoretical_max']}")
            self.logger.info(f"    Collision rate: {level_stats['collision_rate']:.4f}")
            self.logger.info(f"    Max collision group: {level_stats['max_collision_group']} items")
            self.logger.info(f"    Duplicate items: {level_stats['duplicate_items']}")
        
        return model, stats
    
    def generate_semantic_ids(
        self,
        model: HierarchicalKMeans,
        item_ids: np.ndarray,
        path_name: str
    ) -> Dict[int, List[int]]:
        """
        Generate semantic IDs for all items.
        
        Args:
            model: Trained HKM model
            item_ids: Array of item IDs
            path_name: Name of the path ('behavior' or 'semantic')
            
        Returns:
            Dictionary mapping item_id to semantic ID (path)
        """
        self.logger.info(f"\nGenerating semantic IDs for {path_name} path...")
        
        item_to_codes = {}
        for item_idx, item_id in enumerate(item_ids):
            if item_idx in model.item_to_path:
                path = model.item_to_path[item_idx]
                item_to_codes[int(item_id)] = [int(code) for code in path]
        
        self.logger.info(f"  Generated {len(item_to_codes)} semantic IDs")
        
        return item_to_codes
    
    def train(self) -> Dict[str, Any]:
        """Main training loop for dual-path HKM"""
        self.logger.info("="*60)
        self.logger.info("EAGER Dual-Path HKM Tokenizer Training")
        self.logger.info("="*60)
        self.logger.info(f"\nConfiguration:")
        for key, value in self.config.items():
            if key != 'device':
                self.logger.info(f"  {key}: {value}")
        
        # Load data
        dataloader = self.load_data()
        batch = next(iter(dataloader))  # Get all data at once
        
        item_ids = batch['item_id'].numpy()
        semantic_embeddings = batch['semantic_embedding']
        has_cf = 'cf_embedding' in batch
        
        results = {
            'config': self.config,
            'collision_stats': {}
        }
        
        # Train semantic path HKM
        semantic_model, semantic_stats = self.train_hkm(
            semantic_embeddings.to(self.device),
            'semantic'
        )
        results['collision_stats']['semantic'] = semantic_stats
        
        # Generate and save semantic IDs
        semantic_ids = self.generate_semantic_ids(semantic_model, item_ids, 'semantic')
        semantic_path = os.path.join(self.config['output_dir'], 'semantic_id_mappings_semantic.json')
        with open(semantic_path, 'w') as f:
            json.dump(semantic_ids, f, indent=2)
        self.logger.info(f"  Saved to: {semantic_path}")
        
        # Save semantic model
        semantic_model_path = os.path.join(self.config['output_dir'], 'hkm_semantic.pt')
        semantic_model.save_model(semantic_model_path)
        
        # Train behavior path HKM (if CF embeddings available)
        if has_cf:
            cf_embeddings = batch['cf_embedding']
            behavior_model, behavior_stats = self.train_hkm(
                cf_embeddings.to(self.device),
                'behavior'
            )
            results['collision_stats']['behavior'] = behavior_stats
            
            # Generate and save behavior IDs
            behavior_ids = self.generate_semantic_ids(behavior_model, item_ids, 'behavior')
            behavior_path = os.path.join(self.config['output_dir'], 'semantic_id_mappings_behavior.json')
            with open(behavior_path, 'w') as f:
                json.dump(behavior_ids, f, indent=2)
            self.logger.info(f"  Saved to: {behavior_path}")
            
            # Save behavior model
            behavior_model_path = os.path.join(self.config['output_dir'], 'hkm_behavior.pt')
            behavior_model.save_model(behavior_model_path)
        else:
            self.logger.info("\n⚠ No collaborative embeddings provided, skipping behavior path")
        
        # Save collision statistics
        stats_path = os.path.join(self.config['output_dir'], 'collision_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(results['collision_stats'], f, indent=2)
        self.logger.info(f"\nCollision statistics saved to: {stats_path}")
        
        self.logger.info("\n" + "="*60)
        self.logger.info("✓ EAGER HKM training completed successfully!")
        self.logger.info("="*60)
        
        return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='EAGER Dual-Path HKM Tokenizer Training')
    
    # Data paths
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed dataset directory')
    parser.add_argument('--embedding_file', type=str, default='item_emb.parquet',
                       help='Semantic embedding file name')
    parser.add_argument('--cf_embedding_file', type=str, default=None,
                       help='Collaborative embedding file (e.g., lightgcn/item_embeddings_collab.npy)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for models and mappings')
    
    # HKM parameters
    parser.add_argument('--hkm_k', type=int, default=8,
                       help='Branching factor for hierarchical K-means')
    parser.add_argument('--hkm_depth', type=int, default=3,
                       help='Maximum depth of hierarchical tree (semantic ID length)')
    parser.add_argument('--hkm_n_init', type=int, default=10,
                       help='Number of K-means initializations')
    parser.add_argument('--hkm_max_iter', type=int, default=300,
                       help='Maximum K-means iterations')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # System
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum items to load (for testing)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert args to config dict
    config = vars(args)
    
    # Train EAGER HKM tokenizer
    trainer = EAGERTokenizerTrainer(config)
    results = trainer.train()
    
    return results


if __name__ == '__main__':
    main()
