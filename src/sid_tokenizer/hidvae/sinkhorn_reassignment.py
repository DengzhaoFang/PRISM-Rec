#!/usr/bin/env python3
"""
Sinkhorn-based ID Reassignment

Apply optimal transport to resolve ID collisions while preserving semantic structure.
Uses Sinkhorn-Knopp algorithm for efficient assignment.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import json


class SinkhornIDReassigner:
    """
    Reassign colliding semantic IDs using optimal transport.
    
    Strategy:
    1. Identify colliding ID groups
    2. For each group, compute pairwise semantic distances
    3. Use Sinkhorn to find optimal reassignment to nearby unused IDs
    4. Preserve hierarchical structure (prefer changing later layers)
    """
    
    def __init__(
        self,
        semantic_ids_path: str,
        checkpoint_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        device: str = 'cuda',
        output_dir: Optional[str] = None
    ):
        """
        Initialize reassigner.
        
        Args:
            semantic_ids_path: Path to semantic IDs parquet file
            checkpoint_path: Path to model checkpoint (for embeddings)
            data_dir: Path to dataset directory
            device: Device to use
            output_dir: Output directory for results
        """
        self.semantic_ids_path = Path(semantic_ids_path)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if output_dir is None:
            output_dir = self.semantic_ids_path.parent / 'reassigned'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load semantic IDs
        self.load_semantic_ids()
        
        # Load model if provided (for semantic distance computation)
        if checkpoint_path and data_dir:
            self.load_model(checkpoint_path, data_dir)
        else:
            self.model = None
            self.dataset = None
        
        self.logger.info("✓ Sinkhorn ID Reassigner initialized")
    
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SinkhornReassigner')
    
    def load_semantic_ids(self):
        """Load semantic IDs from parquet file"""
        self.logger.info(f"Loading semantic IDs from {self.semantic_ids_path}...")
        
        self.df = pd.read_parquet(self.semantic_ids_path)
        
        # Extract semantic IDs as numpy array
        n_layers = len([col for col in self.df.columns if col.startswith('id_layer')])
        self.semantic_ids = np.stack([
            self.df[f'id_layer{i+1}'].values
            for i in range(n_layers)
        ], axis=1)
        
        self.item_ids = self.df['ItemID'].values
        self.n_items = len(self.item_ids)
        self.n_layers = n_layers
        
        self.logger.info(f"  ✓ Loaded {self.n_items} items with {self.n_layers} layers")
    
    def load_model(self, checkpoint_path: str, data_dir: str):
        """Load model and dataset for semantic distance computation"""
        self.logger.info("Loading model for semantic distance computation...")
        
        from HID_VAE import create_hidvae_from_config
        from multimodal_dataset import HIDVAEDataset
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Load dataset
        self.dataset = HIDVAEDataset(data_dir=data_dir)
        
        # Create model
        num_classes_per_layer = [
            self.dataset.tag_stats[f'n_L{i+2}'] + 1
            for i in range(config.get('n_layers', 3))
        ]
        
        self.model = create_hidvae_from_config(
            config=config,
            num_classes_per_layer=num_classes_per_layer
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.logger.info("  ✓ Model loaded")
    
    def find_collisions(self) -> Dict[Tuple, List[int]]:
        """
        Find all colliding IDs.
        
        Returns:
            collisions: Dict mapping semantic_id -> list of item indices
        """
        self.logger.info("Finding ID collisions...")
        
        # Convert to tuples for hashing
        id_tuples = [tuple(sid) for sid in self.semantic_ids]
        
        # Group by ID
        collision_groups = {}
        for idx, sid in enumerate(id_tuples):
            if sid not in collision_groups:
                collision_groups[sid] = []
            collision_groups[sid].append(idx)
        
        # Filter to only colliding groups
        collisions = {
            sid: indices 
            for sid, indices in collision_groups.items() 
            if len(indices) > 1
        }
        
        n_collision_groups = len(collisions)
        n_items_affected = sum(len(indices) for indices in collisions.values())
        
        self.logger.info(f"  Found {n_collision_groups} collision groups")
        self.logger.info(f"  {n_items_affected} items affected")
        
        return collisions
    
    def sinkhorn_algorithm(
        self,
        cost_matrix: torch.Tensor,
        n_iters: int = 100,
        reg: float = 0.1
    ) -> torch.Tensor:
        """
        Sinkhorn-Knopp algorithm for optimal transport.
        
        Args:
            cost_matrix: Cost matrix (n, m)
            n_iters: Number of iterations
            reg: Regularization parameter (entropy)
            
        Returns:
            transport_plan: Optimal transport plan (n, m)
        """
        # Convert cost to log-space with regularization
        K = torch.exp(-cost_matrix / reg)
        
        # Initialize
        u = torch.ones(K.shape[0], device=K.device) / K.shape[0]
        v = torch.ones(K.shape[1], device=K.device) / K.shape[1]
        
        # Sinkhorn iterations
        for _ in range(n_iters):
            u = 1.0 / (K @ v + 1e-8)
            v = 1.0 / (K.T @ u + 1e-8)
        
        # Compute transport plan
        transport_plan = u[:, None] * K * v[None, :]
        
        return transport_plan
    
    def compute_id_distance(
        self,
        id1: np.ndarray,
        id2: np.ndarray,
        layer_weights: Optional[List[float]] = None
    ) -> float:
        """
        Compute distance between two semantic IDs.
        Prefer changing later layers (lower cost for later layer changes).
        
        Args:
            id1: First semantic ID (n_layers,)
            id2: Second semantic ID (n_layers,)
            layer_weights: Weights for each layer (higher = more important to preserve)
            
        Returns:
            distance: Weighted Hamming distance
        """
        if layer_weights is None:
            # Default: exponentially increasing weights (preserve early layers)
            layer_weights = [2 ** i for i in range(self.n_layers)]
        
        # Hamming distance with weights
        diff = (id1 != id2).astype(float)
        distance = np.dot(diff, layer_weights)
        
        return distance
    
    def find_nearby_unused_ids(
        self,
        target_id: Tuple[int, ...],
        n_candidates: int = 10,
        codebook_size: int = 256
    ) -> List[Tuple[int, ...]]:
        """
        Find unused IDs near the target ID.
        
        Args:
            target_id: Target semantic ID
            n_candidates: Number of candidates to return
            codebook_size: Size of each codebook
            
        Returns:
            candidates: List of nearby unused IDs
        """
        # Convert all IDs to set for fast lookup
        used_ids = set(tuple(sid) for sid in self.semantic_ids)
        
        candidates = []
        target_array = np.array(target_id)
        
        # Try modifying each layer, starting from the last
        for layer in range(self.n_layers - 1, -1, -1):
            # Try all possible values for this layer
            for new_value in range(codebook_size):
                if new_value == target_id[layer]:
                    continue
                
                # Create candidate ID
                candidate = list(target_id)
                candidate[layer] = new_value
                candidate_tuple = tuple(candidate)
                
                # Check if unused
                if candidate_tuple not in used_ids:
                    candidates.append(candidate_tuple)
                    
                    if len(candidates) >= n_candidates:
                        return candidates
        
        return candidates
    
    def reassign_collision_group(
        self,
        collision_id: Tuple[int, ...],
        item_indices: List[int],
        codebook_size: int = 256
    ) -> Dict[int, Tuple[int, ...]]:
        """
        Reassign IDs for a collision group using Sinkhorn.
        
        Args:
            collision_id: The colliding semantic ID
            item_indices: Indices of items with this ID
            codebook_size: Size of each codebook
            
        Returns:
            reassignment: Dict mapping item_index -> new_semantic_id
        """
        n_items = len(item_indices)
        
        # Keep one item with original ID, reassign others
        # Choose the "most representative" item to keep
        # For simplicity, keep the first one
        keep_idx = item_indices[0]
        reassign_indices = item_indices[1:]
        
        if len(reassign_indices) == 0:
            return {}
        
        # Find candidate IDs nearby
        n_candidates = min(n_items * 2, 50)
        candidates = self.find_nearby_unused_ids(
            collision_id, 
            n_candidates=n_candidates,
            codebook_size=codebook_size
        )
        
        if len(candidates) == 0:
            self.logger.warning(f"  No candidates found for {collision_id}, skipping")
            return {}
        
        # Compute cost matrix
        # Cost = distance from original ID
        cost_matrix = np.zeros((len(reassign_indices), len(candidates)))
        
        for i, _ in enumerate(reassign_indices):
            for j, cand_id in enumerate(candidates):
                cost_matrix[i, j] = self.compute_id_distance(
                    np.array(collision_id),
                    np.array(cand_id)
                )
        
        # Convert to torch
        cost_tensor = torch.tensor(cost_matrix, dtype=torch.float32, device=self.device)
        
        # Apply Sinkhorn (if we have more candidates than items)
        if len(candidates) >= len(reassign_indices):
            # Pad to square matrix
            n_pad = len(candidates) - len(reassign_indices)
            cost_tensor = F.pad(cost_tensor, (0, 0, 0, n_pad), value=1e6)
            
            transport_plan = self.sinkhorn_algorithm(cost_tensor, n_iters=100, reg=0.1)
            
            # Get assignment (greedy matching from transport plan)
            assignments = torch.argmax(transport_plan[:len(reassign_indices)], dim=1)
        else:
            # More items than candidates, use greedy assignment
            assignments = torch.argmin(cost_tensor, dim=1)
        
        # Create reassignment dictionary
        reassignment = {}
        for i, item_idx in enumerate(reassign_indices):
            if i < len(assignments):
                cand_idx = assignments[i].item()
                if cand_idx < len(candidates):
                    reassignment[item_idx] = candidates[cand_idx]
        
        return reassignment
    
    def reassign_all(self, codebook_size: int = 256) -> np.ndarray:
        """
        Reassign all colliding IDs.
        
        Args:
            codebook_size: Size of each codebook
            
        Returns:
            new_semantic_ids: Reassigned semantic IDs (n_items, n_layers)
        """
        self.logger.info("Starting ID reassignment...")
        
        # Find collisions
        collisions = self.find_collisions()
        
        if len(collisions) == 0:
            self.logger.info("  No collisions found, nothing to reassign")
            return self.semantic_ids.copy()
        
        # Initialize new IDs with original IDs
        new_semantic_ids = self.semantic_ids.copy()
        
        # Reassign each collision group
        total_reassigned = 0
        
        for collision_id, item_indices in tqdm(collisions.items(), desc="Reassigning"):
            reassignment = self.reassign_collision_group(
                collision_id, 
                item_indices,
                codebook_size
            )
            
            # Apply reassignment
            for item_idx, new_id in reassignment.items():
                new_semantic_ids[item_idx] = np.array(new_id)
                total_reassigned += 1
        
        self.logger.info(f"  ✓ Reassigned {total_reassigned} items")
        
        # Verify uniqueness
        new_id_tuples = [tuple(sid) for sid in new_semantic_ids]
        n_unique = len(set(new_id_tuples))
        uniqueness_rate = n_unique / len(new_id_tuples)
        
        self.logger.info(f"  Final uniqueness: {n_unique}/{len(new_id_tuples)} ({uniqueness_rate:.2%})")
        
        return new_semantic_ids
    
    def save_reassigned_ids(self, new_semantic_ids: np.ndarray):
        """
        Save reassigned semantic IDs.
        
        Args:
            new_semantic_ids: Reassigned semantic IDs
        """
        self.logger.info("Saving reassigned IDs...")
        
        # Create new dataframe
        df_new = self.df.copy()
        
        # Update semantic IDs
        df_new['semantic_id'] = [tuple(sid) for sid in new_semantic_ids]
        
        for i in range(self.n_layers):
            df_new[f'id_layer{i+1}'] = new_semantic_ids[:, i]
        
        # Save
        output_path = self.output_dir / 'semantic_ids_reassigned.parquet'
        df_new.to_parquet(output_path, index=False)
        self.logger.info(f"  ✓ Saved: {output_path}")
        
        # Save numpy
        npy_path = self.output_dir / 'semantic_ids_reassigned.npy'
        np.save(npy_path, new_semantic_ids)
        self.logger.info(f"  ✓ Saved: {npy_path}")
        
        # Save statistics
        original_unique = len(set(tuple(sid) for sid in self.semantic_ids))
        new_unique = len(set(tuple(sid) for sid in new_semantic_ids))
        
        stats = {
            'original_items': len(self.semantic_ids),
            'original_unique_ids': original_unique,
            'original_uniqueness_rate': original_unique / len(self.semantic_ids),
            'reassigned_unique_ids': new_unique,
            'reassigned_uniqueness_rate': new_unique / len(new_semantic_ids),
            'improvement': new_unique - original_unique
        }
        
        stats_path = self.output_dir / 'reassignment_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"  ✓ Statistics saved: {stats_path}")
    
    def run(self, codebook_size: int = 256):
        """Main function: reassign IDs and save results"""
        self.logger.info("=" * 80)
        self.logger.info("Sinkhorn-based ID Reassignment")
        self.logger.info("=" * 80)
        
        # Reassign
        new_semantic_ids = self.reassign_all(codebook_size=codebook_size)
        
        # Save
        self.save_reassigned_ids(new_semantic_ids)
        
        self.logger.info("=" * 80)
        self.logger.info("✓ Reassignment completed!")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Reassign colliding semantic IDs using Sinkhorn algorithm'
    )
    
    parser.add_argument('--semantic_ids', type=str, required=True,
                       help='Path to semantic IDs parquet file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint (optional)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Path to dataset directory (optional)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--codebook_size', type=int, default=256,
                       help='Codebook size')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Initialize reassigner
    reassigner = SinkhornIDReassigner(
        semantic_ids_path=args.semantic_ids,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Run reassignment
    reassigner.run(codebook_size=args.codebook_size)


if __name__ == '__main__':
    main()

