#!/usr/bin/env python3
"""
Semantic ID Generation and Analysis Tool

Generate semantic IDs from trained PRISM model and analyze:
1. ID uniqueness and collision rates
2. Hierarchical overlap rates (layer-wise prefix sharing)
3. Tag prediction accuracy
4. Codebook usage statistics

Optionally apply Sinkhorn algorithm for collision-free ID reassignment.
"""

import os
import argparse
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

from PRISM import PRISM
from multimodal_dataset import PRISMDataset


class SemanticIDGenerator:
    """Generate and analyze semantic IDs from trained PRISM"""
    
    def __init__(
        self, 
        checkpoint_path: str,
        data_dir: str,
        device: str = 'cuda',
        output_dir: Optional[str] = None
    ):
        """
        Initialize ID generator.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            data_dir: Path to dataset directory
            device: Device to use
            output_dir: Output directory for results
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.data_dir = Path(data_dir)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if output_dir is None:
            output_dir = self.checkpoint_path.parent / 'semantic_ids'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load model and dataset
        self.load_model()
        self.load_dataset()
        
        self.logger.info("✓ Semantic ID Generator initialized")
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SemanticIDGenerator')
    
    def load_model(self):
        """Load trained PRISM model"""
        self.logger.info(f"Loading model from {self.checkpoint_path}...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Get number of classes from checkpoint
        # We need to reload dataset first to get this info
        # So we'll do it in load_dataset and create model there
        self.config = config
        self.checkpoint = checkpoint
        
        self.logger.info(f"✓ Checkpoint loaded (epoch {checkpoint['epoch']})")
    
    def load_dataset(self):
        """Load dataset"""
        self.logger.info(f"Loading dataset from {self.data_dir}...")
        
        self.dataset = PRISMDataset(
            data_dir=str(self.data_dir),
            max_items=self.config.get('max_items', None)
        )
        
        # Now create model with proper num_classes
        num_classes_per_layer = [
            self.dataset.tag_stats[f'n_L{i+2}'] + 1
            for i in range(self.config.get('n_layers', 3))
        ]
        
        from HID_VAE import create_prism_from_config
        self.model = create_prism_from_config(
            config=self.config,
            num_classes_per_layer=num_classes_per_layer
        )
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"✓ Model and dataset loaded")
        self.logger.info(f"  Items: {len(self.dataset)}")
        self.logger.info(f"  Layers: {self.config['n_layers']}")
        self.logger.info(f"  Codebook size: {self.config['n_embed']}")
    
    def generate_ids(self, batch_size: int = 512) -> Dict[str, np.ndarray]:
        """
        Generate semantic IDs for all items.
        
        Args:
            batch_size: Batch size for generation
            
        Returns:
            results: Dictionary with IDs and metadata
        """
        self.logger.info("Generating semantic IDs...")
        
        all_item_ids = []
        all_semantic_ids = []
        all_tag_ids = []
        all_predictions = []
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating IDs"):
                content_emb = batch['content_emb'].to(self.device)
                collab_emb = batch['collab_emb'].to(self.device)
                tag_ids = batch['tag_ids']  # Keep on CPU
                item_ids = batch['item_id']
                
                # Generate semantic IDs
                semantic_ids = self.model.generate_semantic_ids(
                    content_emb, collab_emb
                )
                
                # Get tag predictions if classifiers available
                if self.model.classifiers is not None:
                    outputs = self.model(
                        content_emb, collab_emb, return_codes=True
                    )
                    predictions = outputs['predictions']
                    pred_classes = [torch.argmax(p, dim=1).cpu() for p in predictions]
                    pred_classes = torch.stack(pred_classes, dim=1)  # (B, n_layers)
                else:
                    pred_classes = torch.zeros_like(semantic_ids)
                
                all_item_ids.append(item_ids.cpu().numpy())
                all_semantic_ids.append(semantic_ids.cpu().numpy())
                all_tag_ids.append(tag_ids.cpu().numpy())
                all_predictions.append(pred_classes.cpu().numpy())
        
        # Concatenate all batches
        results = {
            'item_ids': np.concatenate(all_item_ids, axis=0),
            'semantic_ids': np.concatenate(all_semantic_ids, axis=0),
            'tag_ids': np.concatenate(all_tag_ids, axis=0),
            'predictions': np.concatenate(all_predictions, axis=0)
        }
        
        self.logger.info(f"✓ Generated {len(results['item_ids'])} semantic IDs")
        
        return results
    
    def analyze_uniqueness(self, semantic_ids: np.ndarray) -> Dict:
        """
        Analyze ID uniqueness and collision rates.
        
        Args:
            semantic_ids: Array of semantic IDs (n_items, n_layers)
            
        Returns:
            stats: Dictionary with uniqueness statistics
        """
        self.logger.info("Analyzing ID uniqueness...")
        
        n_items = len(semantic_ids)
        n_layers = semantic_ids.shape[1]
        
        # Convert to tuples for hashing
        id_tuples = [tuple(sid) for sid in semantic_ids]
        
        # Count unique IDs
        unique_ids = set(id_tuples)
        n_unique = len(unique_ids)
        
        # Find collisions
        id_counts = Counter(id_tuples)
        collisions = {k: v for k, v in id_counts.items() if v > 1}
        n_collisions = len(collisions)
        n_items_with_collisions = sum(collisions.values())
        
        # Collision rate
        uniqueness_rate = n_unique / n_items
        collision_rate = n_collisions / n_items
        
        stats = {
            'n_items': n_items,
            'n_unique_ids': n_unique,
            'n_collisions': n_collisions,
            'n_items_with_collisions': n_items_with_collisions,
            'uniqueness_rate': uniqueness_rate,
            'collision_rate': collision_rate,
            'collisions': collisions
        }
        
        self.logger.info(f"  Total items: {n_items}")
        self.logger.info(f"  Unique IDs: {n_unique} ({uniqueness_rate:.2%})")
        self.logger.info(f"  Collisions: {n_collisions} ({collision_rate:.2%})")
        self.logger.info(f"  Items affected by collisions: {n_items_with_collisions}")
        
        return stats
    
    def analyze_hierarchical_overlap(self, semantic_ids: np.ndarray) -> Dict:
        """
        Analyze hierarchical overlap rates (prefix sharing).
        
        Args:
            semantic_ids: Array of semantic IDs (n_items, n_layers)
            
        Returns:
            overlap_stats: Dictionary with overlap statistics per layer
        """
        self.logger.info("Analyzing hierarchical overlap rates...")
        
        n_items = len(semantic_ids)
        n_layers = semantic_ids.shape[1]
        
        overlap_stats = {}
        
        for layer in range(1, n_layers + 1):
            # Get prefix up to this layer
            prefixes = [tuple(sid[:layer]) for sid in semantic_ids]
            
            # Count unique prefixes
            unique_prefixes = set(prefixes)
            n_unique = len(unique_prefixes)
            
            # Average items per prefix
            prefix_counts = Counter(prefixes)
            avg_items_per_prefix = n_items / n_unique
            
            # Overlap rate (lower is more diverse)
            overlap_rate = 1.0 - (n_unique / n_items)
            
            overlap_stats[f'layer_{layer}'] = {
                'unique_prefixes': n_unique,
                'avg_items_per_prefix': avg_items_per_prefix,
                'overlap_rate': overlap_rate
            }
            
            self.logger.info(f"  Layer {layer} prefix:")
            self.logger.info(f"    Unique: {n_unique} / {n_items}")
            self.logger.info(f"    Avg items/prefix: {avg_items_per_prefix:.2f}")
            self.logger.info(f"    Overlap rate: {overlap_rate:.2%}")
        
        return overlap_stats
    
    def analyze_tag_accuracy(
        self, 
        predictions: np.ndarray, 
        tag_ids: np.ndarray
    ) -> Dict:
        """
        Analyze tag prediction accuracy.
        
        Args:
            predictions: Predicted tag IDs (n_items, n_layers)
            tag_ids: Ground truth tag IDs (n_items, n_layers)
            
        Returns:
            accuracy_stats: Dictionary with accuracy per layer
        """
        self.logger.info("Analyzing tag prediction accuracy...")
        
        n_layers = predictions.shape[1]
        accuracy_stats = {}
        
        for layer in range(n_layers):
            pred = predictions[:, layer]
            target = tag_ids[:, layer]
            
            # Filter out PAD tokens (ID 0)
            valid_mask = target != 0
            
            if valid_mask.sum() > 0:
                correct = (pred[valid_mask] == target[valid_mask]).sum()
                total = valid_mask.sum()
                accuracy = correct / total
            else:
                accuracy = 0.0
                total = 0
            
            accuracy_stats[f'layer_{layer+1}'] = {
                'accuracy': float(accuracy),
                'n_valid': int(total)
            }
            
            self.logger.info(f"  Layer {layer+1} (L{layer+2}): {accuracy:.2%} ({total} valid)")
        
        return accuracy_stats
    
    def analyze_codebook_usage(self, semantic_ids: np.ndarray) -> Dict:
        """
        Analyze codebook usage statistics.
        
        Args:
            semantic_ids: Array of semantic IDs (n_items, n_layers)
            
        Returns:
            usage_stats: Dictionary with usage statistics per layer
        """
        self.logger.info("Analyzing codebook usage...")
        
        n_layers = semantic_ids.shape[1]
        n_embed = self.config['n_embed']
        
        usage_stats = {}
        
        for layer in range(n_layers):
            codes = semantic_ids[:, layer]
            
            # Count usage
            code_counts = Counter(codes)
            n_used = len(code_counts)
            usage_rate = n_used / n_embed
            
            # Most/least used
            most_common = code_counts.most_common(5)
            least_used = [i for i in range(n_embed) if i not in code_counts]
            
            usage_stats[f'layer_{layer+1}'] = {
                'n_used': n_used,
                'n_unused': n_embed - n_used,
                'usage_rate': usage_rate,
                'most_common': most_common,
                'n_least_used': len(least_used)
            }
            
            self.logger.info(f"  Layer {layer+1}:")
            self.logger.info(f"    Used: {n_used} / {n_embed} ({usage_rate:.2%})")
            self.logger.info(f"    Unused: {n_embed - n_used}")
        
        return usage_stats
    
    def save_results(
        self, 
        results: Dict[str, np.ndarray], 
        stats: Dict
    ):
        """
        Save semantic IDs and statistics.
        
        Args:
            results: Dictionary with IDs and predictions
            stats: Dictionary with all statistics
        """
        self.logger.info("Saving results...")
        
        # Save semantic IDs as parquet
        df = pd.DataFrame({
            'ItemID': results['item_ids'],
            'semantic_id': [tuple(sid) for sid in results['semantic_ids']],
            'tag_ids': [tuple(tid) for tid in results['tag_ids']],
            'predictions': [tuple(pid) for pid in results['predictions']]
        })
        
        # Add individual layer IDs as columns
        n_layers = results['semantic_ids'].shape[1]
        for i in range(n_layers):
            df[f'id_layer{i+1}'] = results['semantic_ids'][:, i]
            df[f'tag_layer{i+1}'] = results['tag_ids'][:, i]
            df[f'pred_layer{i+1}'] = results['predictions'][:, i]
        
        parquet_path = self.output_dir / 'semantic_ids.parquet'
        df.to_parquet(parquet_path, index=False)
        self.logger.info(f"  ✓ Semantic IDs saved: {parquet_path}")
        
        # Save as numpy
        npy_path = self.output_dir / 'semantic_ids.npy'
        np.save(npy_path, results['semantic_ids'])
        self.logger.info(f"  ✓ Numpy array saved: {npy_path}")
        
        # Save statistics as JSON
        stats_path = self.output_dir / 'id_statistics.json'
        
        # Remove large collision dict for JSON
        if 'uniqueness' in stats and 'collisions' in stats['uniqueness']:
            stats['uniqueness']['n_collision_groups'] = len(stats['uniqueness']['collisions'])
            del stats['uniqueness']['collisions']
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"  ✓ Statistics saved: {stats_path}")
        
        # Save collision details separately
        if 'uniqueness' in stats:
            collision_df = pd.DataFrame([
                {
                    'semantic_id': str(k),
                    'count': v,
                    'item_ids': str([
                        results['item_ids'][i] 
                        for i, sid in enumerate(results['semantic_ids']) 
                        if tuple(sid) == k
                    ][:10])  # First 10 items
                }
                for k, v in Counter([tuple(sid) for sid in results['semantic_ids']]).items()
                if v > 1
            ])
            
            if len(collision_df) > 0:
                collision_path = self.output_dir / 'collisions.csv'
                collision_df.to_csv(collision_path, index=False)
                self.logger.info(f"  ✓ Collision details saved: {collision_path}")
    
    def generate_and_analyze(self):
        """Main function: generate IDs and perform all analyses"""
        self.logger.info("=" * 80)
        self.logger.info("Semantic ID Generation and Analysis")
        self.logger.info("=" * 80)
        
        # Generate IDs
        results = self.generate_ids()
        
        # Analyze
        stats = {}
        
        # 1. Uniqueness
        stats['uniqueness'] = self.analyze_uniqueness(results['semantic_ids'])
        
        # 2. Hierarchical overlap
        stats['hierarchical_overlap'] = self.analyze_hierarchical_overlap(
            results['semantic_ids']
        )
        
        # 3. Tag accuracy
        stats['tag_accuracy'] = self.analyze_tag_accuracy(
            results['predictions'], 
            results['tag_ids']
        )
        
        # 4. Codebook usage
        stats['codebook_usage'] = self.analyze_codebook_usage(
            results['semantic_ids']
        )
        
        # Save results
        self.save_results(results, stats)
        
        self.logger.info("=" * 80)
        self.logger.info("✓ Analysis completed!")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)
        
        return results, stats


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate and analyze semantic IDs from trained PRISM'
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size for ID generation')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize generator
    generator = SemanticIDGenerator(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Generate and analyze
    generator.generate_and_analyze()


if __name__ == '__main__':
    main()

