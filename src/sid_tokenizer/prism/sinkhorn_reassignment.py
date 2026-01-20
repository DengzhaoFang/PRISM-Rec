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
from collections import Counter, defaultdict

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
            # Default: output to same directory as input (overwrite)
            output_dir = self.semantic_ids_path.parent
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
        """Load semantic IDs from JSON or parquet file"""
        self.logger.info(f"Loading semantic IDs from {self.semantic_ids_path}...")
        
        if self.semantic_ids_path.suffix == '.json':
            # Load from JSON format (TIGER format)
            with open(self.semantic_ids_path, 'r') as f:
                id_mappings = json.load(f)
            
            # Convert to numpy array
            item_ids_list = []
            semantic_ids_list = []
            
            for item_id_str, codes in id_mappings.items():
                item_ids_list.append(int(item_id_str))
                semantic_ids_list.append(codes)
            
            self.item_ids = np.array(item_ids_list)
            self.semantic_ids = np.array(semantic_ids_list)
            self.n_items = len(self.item_ids)
            self.n_layers = len(semantic_ids_list[0]) if semantic_ids_list else 0
            
            # Create a simple dataframe for compatibility
            self.df = pd.DataFrame({
                'ItemID': self.item_ids,
                'semantic_id': [tuple(sid) for sid in self.semantic_ids]
            })
            for i in range(self.n_layers):
                self.df[f'id_layer{i+1}'] = self.semantic_ids[:, i]
            
            self.logger.info(f"  ✓ Loaded {self.n_items} items with {self.n_layers} layers from JSON")
        else:
            # Load from parquet file
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
            
            self.logger.info(f"  ✓ Loaded {self.n_items} items with {self.n_layers} layers from Parquet")
    
    def load_model(self, checkpoint_path: str, data_dir: str):
        """Load model and dataset for semantic distance computation"""
        self.logger.info("Loading model for semantic distance computation...")
        
        from PRISM import create_prism_from_config
        from multimodal_dataset import PRISMDataset
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']
        
        # Load dataset
        self.dataset = PRISMDataset(data_dir=data_dir)
        
        # Create model
        num_classes_per_layer = [
            self.dataset.tag_stats[f'n_L{i+2}'] + 1
            for i in range(config.get('n_layers', 3))
        ]
        
        self.model = create_prism_from_config(
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
        used_ids: set,
        n_candidates: int = 50,
        codebook_sizes: Optional[List[int]] = None,
        max_layers_to_modify: int = 2
    ) -> List[Tuple[int, ...]]:
        """
        Find unused IDs near the target ID.
        
        Args:
            target_id: Target semantic ID
            used_ids: Set of currently used IDs (updated in real-time)
            n_candidates: Number of candidates to return
            codebook_sizes: List of codebook sizes per layer (if None, uses uniform 256)
            max_layers_to_modify: Maximum number of layers to modify
            
        Returns:
            candidates: List of nearby unused IDs
        """
        if codebook_sizes is None:
            codebook_sizes = [256] * self.n_layers
        
        candidates = []
        target_array = np.array(target_id)
        
        # Strategy 1: Modify single layer (prefer later layers)
        for layer in range(self.n_layers - 1, -1, -1):
            codebook_size = codebook_sizes[layer]
            for new_value in range(codebook_size):
                if new_value == target_id[layer]:
                    continue
                
                candidate = list(target_id)
                candidate[layer] = new_value
                candidate_tuple = tuple(candidate)
                
                if candidate_tuple not in used_ids:
                    candidates.append(candidate_tuple)
                    if len(candidates) >= n_candidates:
                        return candidates
        
        # Strategy 2: Modify two layers (if single layer didn't find enough)
        # Limit search to avoid explosion: only try nearby values
        if len(candidates) < n_candidates and max_layers_to_modify >= 2:
            # Try modifying last two layers with nearby values
            for layer1 in range(self.n_layers - 1, max(-1, self.n_layers - 3), -1):
                codebook_size1 = codebook_sizes[layer1]
                search_radius1 = min(10, codebook_size1 // 4)
                
                for layer2 in range(layer1 - 1, -1, -1):
                    codebook_size2 = codebook_sizes[layer2]
                    search_radius2 = min(10, codebook_size2 // 4)
                    
                    # Try values near the original (faster than full search)
                    base_val1 = target_id[layer1]
                    base_val2 = target_id[layer2]
                    
                    for offset1 in range(-search_radius1, search_radius1 + 1):
                        val1 = (base_val1 + offset1) % codebook_size1
                        if val1 == base_val1:
                            continue
                        
                        for offset2 in range(-search_radius2, search_radius2 + 1):
                            val2 = (base_val2 + offset2) % codebook_size2
                            if val2 == base_val2:
                                continue
                            
                            candidate = list(target_id)
                            candidate[layer1] = val1
                            candidate[layer2] = val2
                            candidate_tuple = tuple(candidate)
                            
                            if candidate_tuple not in used_ids:
                                candidates.append(candidate_tuple)
                                if len(candidates) >= n_candidates:
                                    return candidates
        
        # Strategy 3: Modify all layers (last resort)
        if len(candidates) < n_candidates:
            # Try random combinations
            import random
            attempts = 0
            max_attempts = 1000
            while len(candidates) < n_candidates and attempts < max_attempts:
                candidate = list(target_id)
                # Modify random layer
                layer = random.randint(0, self.n_layers - 1)
                codebook_size = codebook_sizes[layer]
                new_value = random.randint(0, codebook_size - 1)
                if new_value != target_id[layer]:
                    candidate[layer] = new_value
                    candidate_tuple = tuple(candidate)
                    
                    if candidate_tuple not in used_ids:
                        candidates.append(candidate_tuple)
                attempts += 1
        
        return candidates
    
    def reassign_collision_group(
        self,
        collision_id: Tuple[int, ...],
        item_indices: List[int],
        used_ids: set,
        codebook_sizes: Optional[List[int]] = None
    ) -> Dict[int, Tuple[int, ...]]:
        """
        Reassign IDs for a collision group using Sinkhorn.
        
        Args:
            collision_id: The colliding semantic ID
            item_indices: Indices of items with this ID
            used_ids: Set of currently used IDs (updated in real-time)
            codebook_sizes: List of codebook sizes per layer (if None, uses uniform 256)
            
        Returns:
            reassignment: Dict mapping item_index -> new_semantic_id
        """
        if codebook_sizes is None:
            codebook_sizes = [256] * self.n_layers
        
        n_items = len(item_indices)
        
        # Keep one item with original ID, reassign others
        keep_idx = item_indices[0]
        reassign_indices = item_indices[1:]
        
        if len(reassign_indices) == 0:
            return {}
        
        # Find candidate IDs nearby (use real-time used_ids)
        n_candidates = max(n_items * 3, 100)  # More candidates to ensure success
        candidates = self.find_nearby_unused_ids(
            collision_id, 
            used_ids=used_ids,
            n_candidates=n_candidates,
            codebook_sizes=codebook_sizes,
            max_layers_to_modify=2
        )
        
        if len(candidates) < len(reassign_indices):
            # Not enough candidates, try more aggressive search
            self.logger.warning(
                f"  Only found {len(candidates)} candidates for {len(reassign_indices)} items, "
                f"expanding search..."
            )
            # Try modifying more layers
            candidates = self.find_nearby_unused_ids(
                collision_id,
                used_ids=used_ids,
                n_candidates=n_candidates * 2,
                codebook_sizes=codebook_sizes,
                max_layers_to_modify=3
            )
        
        if len(candidates) == 0:
            self.logger.error(f"  No candidates found for {collision_id}, cannot reassign!")
            # Fallback: assign sequential IDs from unused pool
            return self._fallback_reassign(collision_id, reassign_indices, used_ids, codebook_size)
        
        # Ensure we have enough candidates
        if len(candidates) < len(reassign_indices):
            # Use greedy assignment: assign closest candidates first
            reassignment = {}
            remaining_candidates = list(candidates)
            
            for item_idx in reassign_indices:
                if len(remaining_candidates) == 0:
                    # No more candidates, use fallback
                    fallback_id = self._find_any_unused_id(used_ids, codebook_size)
                    if fallback_id:
                        reassignment[item_idx] = fallback_id
                        used_ids.add(fallback_id)
                    continue
                
                # Find closest candidate
                best_cand = None
                best_dist = float('inf')
                for cand in remaining_candidates:
                    dist = self.compute_id_distance(
                        np.array(collision_id),
                        np.array(cand)
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_cand = cand
                
                if best_cand:
                    reassignment[item_idx] = best_cand
                    remaining_candidates.remove(best_cand)
                    used_ids.add(best_cand)
            
            return reassignment
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(reassign_indices), len(candidates)))
        
        for i, _ in enumerate(reassign_indices):
            for j, cand_id in enumerate(candidates):
                cost_matrix[i, j] = self.compute_id_distance(
                    np.array(collision_id),
                    np.array(cand_id)
                )
        
        # Convert to torch
        cost_tensor = torch.tensor(cost_matrix, dtype=torch.float32, device=self.device)
        
        # Apply Sinkhorn
        if len(candidates) >= len(reassign_indices):
            # Pad to square matrix for Sinkhorn
            n_pad = len(candidates) - len(reassign_indices)
            if n_pad > 0:
                cost_tensor = F.pad(cost_tensor, (0, 0, 0, n_pad), value=1e6)
            
            transport_plan = self.sinkhorn_algorithm(cost_tensor, n_iters=100, reg=0.1)
            
            # Get assignment (greedy matching from transport plan)
            assignments = torch.argmax(transport_plan[:len(reassign_indices)], dim=1)
        else:
            # More items than candidates (shouldn't happen now)
            assignments = torch.argmin(cost_tensor, dim=1)
        
        # Create reassignment dictionary and update used_ids
        reassignment = {}
        assigned_candidates = set()
        failed_items = []
        
        for i, item_idx in enumerate(reassign_indices):
            new_id = None
            
            if i < len(assignments):
                cand_idx = assignments[i].item()
                if cand_idx < len(candidates):
                    candidate_id = candidates[cand_idx]
                    
                    # Check if candidate is available
                    if candidate_id not in assigned_candidates and candidate_id not in used_ids:
                        new_id = candidate_id
                        assigned_candidates.add(candidate_id)
            
            # If Sinkhorn assignment failed, try to find alternative
            if new_id is None:
                # Try to find nearby alternative
                alt_id = self._find_alternative_id(
                    collision_id, used_ids, assigned_candidates, codebook_sizes
                )
                if alt_id:
                    new_id = alt_id
                    assigned_candidates.add(alt_id)
            
            # If still no ID found, mark for fallback
            if new_id is None:
                failed_items.append(item_idx)
            else:
                reassignment[item_idx] = new_id
                used_ids.add(new_id)  # Update used_ids immediately
        
        # Handle failed items with fallback
        if len(failed_items) > 0:
            self.logger.warning(
                f"  {len(failed_items)} items failed Sinkhorn assignment, using fallback"
            )
            for item_idx in failed_items:
                fallback_id = self._find_any_unused_id(used_ids, codebook_sizes)
                if fallback_id:
                    reassignment[item_idx] = fallback_id
                    used_ids.add(fallback_id)
                else:
                    self.logger.error(f"  Cannot find unused ID for item {item_idx}!")
        
        return reassignment
    
    def _find_alternative_id(
        self,
        target_id: Tuple[int, ...],
        used_ids: set,
        assigned_candidates: set,
        codebook_sizes: Optional[List[int]] = None
    ) -> Optional[Tuple[int, ...]]:
        """Find an alternative unused ID."""
        if codebook_sizes is None:
            codebook_sizes = [256] * self.n_layers
        
        # Try modifying each layer
        for layer in range(self.n_layers - 1, -1, -1):
            codebook_size = codebook_sizes[layer]
            for new_value in range(codebook_size):
                if new_value == target_id[layer]:
                    continue
                
                candidate = list(target_id)
                candidate[layer] = new_value
                candidate_tuple = tuple(candidate)
                
                if candidate_tuple not in used_ids and candidate_tuple not in assigned_candidates:
                    return candidate_tuple
        
        return None
    
    def _find_any_unused_id(
        self,
        used_ids: set,
        codebook_sizes: Optional[List[int]] = None
    ) -> Optional[Tuple[int, ...]]:
        """
        Find any unused ID (fallback).
        Uses systematic search if random search fails.
        """
        if codebook_sizes is None:
            codebook_sizes = [256] * self.n_layers
        
        import random
        
        # Strategy 1: Random search (fast)
        max_random_attempts = 10000
        for _ in range(max_random_attempts):
            candidate = tuple(random.randint(0, codebook_sizes[i] - 1) for i in range(self.n_layers))
            if candidate not in used_ids:
                return candidate
        
        # Strategy 2: Systematic search (slower but guaranteed to find if exists)
        self.logger.warning("  Random search failed, trying systematic search...")
        
        # Calculate total combinations
        total_combinations = 1
        for size in codebook_sizes:
            total_combinations *= size
        
        # Try all combinations systematically
        # Start from a random offset to avoid always using the same IDs
        offset = random.randint(0, min(codebook_sizes) - 1)
        max_systematic = min(100000, total_combinations)  # Limit systematic search
        
        for attempt in range(max_systematic):
            # Generate candidate using mixed-radix base conversion
            num = (offset + attempt) % total_combinations
            candidate = []
            temp = num
            for layer_idx in range(self.n_layers):
                candidate.append(temp % codebook_sizes[layer_idx])
                temp //= codebook_sizes[layer_idx]
            candidate_tuple = tuple(candidate)
            
            if candidate_tuple not in used_ids:
                return candidate_tuple
        
        # Strategy 3: Last resort - increment from a known ID
        self.logger.warning("  Systematic search limited, trying increment strategy...")
        
        # Try incrementing from existing IDs
        for base_id in list(used_ids)[:1000]:  # Try first 1000 used IDs
            base_list = list(base_id)
            for layer in range(self.n_layers):
                codebook_size = codebook_sizes[layer]
                for offset in [1, -1, 2, -2, 3, -3]:
                    new_val = (base_list[layer] + offset) % codebook_size
                    if new_val != base_list[layer]:
                        candidate = base_list.copy()
                        candidate[layer] = new_val
                        candidate_tuple = tuple(candidate)
                        if candidate_tuple not in used_ids:
                            return candidate_tuple
        
        # If all strategies fail, return None (shouldn't happen in practice)
        self.logger.error(f"  Cannot find unused ID! Used IDs: {len(used_ids)}, Total possible: {total_combinations}")
        return None
    
    def _fallback_reassign(
        self,
        collision_id: Tuple[int, ...],
        item_indices: List[int],
        used_ids: set,
        codebook_sizes: Optional[List[int]] = None
    ) -> Dict[int, Tuple[int, ...]]:
        """Fallback reassignment when no nearby candidates found."""
        if codebook_sizes is None:
            codebook_sizes = [256] * self.n_layers
        
        reassignment = {}
        
        for item_idx in item_indices:
            new_id = self._find_any_unused_id(used_ids, codebook_sizes)
            if new_id:
                reassignment[item_idx] = new_id
                used_ids.add(new_id)
            else:
                self.logger.error(f"  Cannot find unused ID for item {item_idx}!")
        
        return reassignment
    
    def reassign_all(self, codebook_sizes: Optional[List[int]] = None, max_iterations: int = 10) -> np.ndarray:
        """
        Reassign all colliding IDs iteratively until 100% uniqueness.
        
        Args:
            codebook_sizes: List of codebook sizes per layer (if None, uses uniform 256)
            max_iterations: Maximum number of iterations
            
        Returns:
            new_semantic_ids: Reassigned semantic IDs (n_items, n_layers)
        """
        if codebook_sizes is None:
            codebook_sizes = [256] * self.n_layers
        
        self.logger.info("Starting ID reassignment (iterative until 100% unique)...")
        self.logger.info(f"  Codebook sizes: {codebook_sizes}")
        
        # Initialize new IDs with original IDs
        new_semantic_ids = self.semantic_ids.copy()
        
        # Iterate until 100% uniqueness
        iteration = 0
        total_reassigned = 0
        
        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"\n  Iteration {iteration}/{max_iterations}:")
            
            # Recompute used_ids from current state (important!)
            id_tuples = [tuple(sid) for sid in new_semantic_ids]
            id_counts = Counter(id_tuples)
            
            # Find collisions in current state
            collisions = {
                sid: [i for i, tid in enumerate(id_tuples) if tid == sid]
                for sid, count in id_counts.items()
                if count > 1
            }
            
            if len(collisions) == 0:
                self.logger.info(f"  ✓ No collisions found! 100% unique achieved.")
                break
            
            n_collision_groups = len(collisions)
            n_items_in_collisions = sum(len(indices) for indices in collisions.values())
            
            self.logger.info(f"  Found {n_collision_groups} collision groups ({n_items_in_collisions} items)")
            
            # Build real-time used_ids set (all currently used IDs)
            used_ids = set(id_tuples)
            
            # Build mapping of ID to items (for efficient removal)
            id_to_items = defaultdict(list)
            for idx, sid_tuple in enumerate(id_tuples):
                id_to_items[sid_tuple].append(idx)
            
            # Reassign each collision group
            iteration_reassigned = 0
            
            # Sort by collision size (largest first) to handle big groups first
            sorted_collisions = sorted(
                collisions.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )
            
            for collision_id, item_indices in tqdm(
                sorted_collisions,
                desc=f"  Reassigning iteration {iteration}",
                leave=False
            ):
                # Items to reassign (keep first item with original ID)
                items_to_reassign = item_indices[1:]
                
                # Temporarily remove items being reassigned from used_ids
                # (they will get new IDs)
                for idx in items_to_reassign:
                    old_id = tuple(new_semantic_ids[idx])
                    # Check if this ID is only used by this item
                    if len(id_to_items[old_id]) == 1:
                        # This ID will become unused after reassignment
                        used_ids.discard(old_id)
                        del id_to_items[old_id]
                    else:
                        # Other items still use this ID, just remove this item
                        id_to_items[old_id].remove(idx)
                
                # Reassign
                reassignment = self.reassign_collision_group(
                    collision_id,
                    item_indices,
                    used_ids,  # Pass real-time used_ids
                    codebook_sizes
                )
                
                # Apply reassignment and update data structures
                for item_idx, new_id in reassignment.items():
                    # Update semantic ID
                    old_id = tuple(new_semantic_ids[item_idx])
                    new_semantic_ids[item_idx] = np.array(new_id)
                    
                    # Update used_ids
                    used_ids.add(new_id)
                    
                    # Update id_to_items mapping
                    id_to_items[new_id].append(item_idx)
                    
                    iteration_reassigned += 1
                    total_reassigned += 1
            
            self.logger.info(f"  Reassigned {iteration_reassigned} items in this iteration")
            
            # Verify current uniqueness
            new_id_tuples = [tuple(sid) for sid in new_semantic_ids]
            n_unique = len(set(new_id_tuples))
            uniqueness_rate = n_unique / len(new_id_tuples)
            
            self.logger.info(f"  Current uniqueness: {n_unique}/{len(new_id_tuples)} ({uniqueness_rate:.2%})")
            
            # Check if 100% unique
            if n_unique == len(new_semantic_ids):
                self.logger.info(f"  ✓ 100% uniqueness achieved!")
                break
        
        # Final verification
        final_id_tuples = [tuple(sid) for sid in new_semantic_ids]
        final_unique = len(set(final_id_tuples))
        final_uniqueness_rate = final_unique / len(final_id_tuples)
        
        if final_unique < len(new_semantic_ids):
            remaining_collisions = len(final_id_tuples) - final_unique
            self.logger.warning(
                f"  ⚠️  Still {remaining_collisions} collisions after {iteration} iterations!"
            )
            self.logger.warning(f"  Final uniqueness: {final_uniqueness_rate:.2%}")
            
            # One more aggressive pass
            self.logger.info("  Attempting final aggressive reassignment...")
            new_semantic_ids = self._aggressive_reassign(
                new_semantic_ids, codebook_sizes
            )
            
            # Final check
            final_id_tuples = [tuple(sid) for sid in new_semantic_ids]
            final_unique = len(set(final_id_tuples))
            final_uniqueness_rate = final_unique / len(final_id_tuples)
            
            if final_unique == len(new_semantic_ids):
                self.logger.info(f"  ✓ 100% uniqueness achieved after aggressive reassignment!")
            else:
                self.logger.error(
                    f"  ❌ Failed to achieve 100% uniqueness: {final_uniqueness_rate:.2%}"
                )
        else:
            self.logger.info(f"  ✓ Final uniqueness: {final_unique}/{len(new_semantic_ids)} (100%)")
        
        self.logger.info(f"\n  Total items reassigned: {total_reassigned}")
        
        return new_semantic_ids
    
    def _aggressive_reassign(
        self,
        semantic_ids: np.ndarray,
        codebook_sizes: Optional[List[int]] = None
    ) -> np.ndarray:
        """Aggressive reassignment: assign any unused IDs."""
        if codebook_sizes is None:
            codebook_sizes = [256] * self.n_layers
        
        new_semantic_ids = semantic_ids.copy()
        used_ids = set(tuple(sid) for sid in new_semantic_ids)
        
        # Find all collisions
        id_tuples = [tuple(sid) for sid in new_semantic_ids]
        id_counts = Counter(id_tuples)
        collisions = {
            sid: [i for i, tid in enumerate(id_tuples) if tid == sid]
            for sid, count in id_counts.items()
            if count > 1
        }
        
        self.logger.info(f"  Aggressive reassignment: {len(collisions)} collision groups")
        
        for collision_id, item_indices in collisions.items():
            # Keep first item, reassign others
            for item_idx in item_indices[1:]:
                # Find any unused ID
                new_id = self._find_any_unused_id(used_ids, codebook_sizes)
                if new_id:
                    new_semantic_ids[item_idx] = np.array(new_id)
                    used_ids.add(new_id)
                else:
                    self.logger.error(f"  Cannot find unused ID for item {item_idx}!")
        
        return new_semantic_ids
    
    def save_reassigned_ids(
        self, 
        new_semantic_ids: np.ndarray,
        original_semantic_ids: Optional[np.ndarray] = None
    ):
        """
        Save reassigned semantic IDs in TIGER format (JSON).
        
        Args:
            new_semantic_ids: Reassigned semantic IDs
            original_semantic_ids: Original semantic IDs (for comparison)
        """
        self.logger.info("Saving reassigned IDs...")
        
        if original_semantic_ids is None:
            original_semantic_ids = self.semantic_ids
        
        # Save in TIGER format (JSON)
        semantic_id_mappings = {}
        for i in range(self.n_items):
            item_id = str(self.item_ids[i])
            semantic_codes = new_semantic_ids[i].tolist()
            semantic_id_mappings[item_id] = semantic_codes
        
        # Overwrite original file with reassigned IDs
        output_path = self.output_dir / 'semantic_id_mappings.json'
        with open(output_path, 'w') as f:
            json.dump(semantic_id_mappings, f, indent=2)
        self.logger.info(f"  ✓ Saved: {output_path}")
        
        # Save numpy
        npy_path = self.output_dir / 'semantic_ids.npy'
        np.save(npy_path, new_semantic_ids)
        self.logger.info(f"  ✓ Saved: {npy_path}")
        
        # Detailed before/after analysis
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Before/After Sinkhorn Reassignment Analysis")
        self.logger.info("=" * 80)
        
        # Original analysis
        original_id_tuples = [tuple(sid) for sid in original_semantic_ids]
        original_unique = len(set(original_id_tuples))
        original_unique_rate = original_unique / len(original_id_tuples)
        
        original_counts = Counter(original_id_tuples)
        original_collisions = {sid: count for sid, count in original_counts.items() if count > 1}
        original_n_collision_groups = len(original_collisions)
        original_n_items_in_collisions = sum(count for count in original_collisions.values())
        
        self.logger.info(f"\n📊 BEFORE Sinkhorn Reassignment:")
        self.logger.info(f"   Total items: {len(original_semantic_ids)}")
        self.logger.info(f"   Unique IDs: {original_unique}")
        self.logger.info(f"   Uniqueness rate: {original_unique_rate:.4f} ({original_unique_rate:.2%})")
        self.logger.info(f"   Collision groups: {original_n_collision_groups}")
        self.logger.info(f"   Items in collisions: {original_n_items_in_collisions} ({original_n_items_in_collisions/len(original_semantic_ids):.2%})")
        
        # Reassigned analysis
        new_id_tuples = [tuple(sid) for sid in new_semantic_ids]
        new_unique = len(set(new_id_tuples))
        new_unique_rate = new_unique / len(new_semantic_ids)
        
        new_counts = Counter(new_id_tuples)
        new_collisions = {sid: count for sid, count in new_counts.items() if count > 1}
        new_n_collision_groups = len(new_collisions)
        new_n_items_in_collisions = sum(count for count in new_collisions.values())
        
        self.logger.info(f"\n📊 AFTER Sinkhorn Reassignment:")
        self.logger.info(f"   Total items: {len(new_semantic_ids)}")
        self.logger.info(f"   Unique IDs: {new_unique}")
        self.logger.info(f"   Uniqueness rate: {new_unique_rate:.4f} ({new_unique_rate:.2%})")
        self.logger.info(f"   Collision groups: {new_n_collision_groups}")
        self.logger.info(f"   Items in collisions: {new_n_items_in_collisions} ({new_n_items_in_collisions/len(new_semantic_ids):.2%})")
        
        # Improvement
        improvement = new_unique - original_unique
        improvement_rate = new_unique_rate - original_unique_rate
        
        self.logger.info(f"\n📈 IMPROVEMENT:")
        self.logger.info(f"   Unique IDs improvement: +{improvement}")
        self.logger.info(f"   Uniqueness rate improvement: +{improvement_rate:.4f} ({improvement_rate:.2%})")
        self.logger.info(f"   Collision groups reduced: {original_n_collision_groups - new_n_collision_groups}")
        self.logger.info(f"   Items in collisions reduced: {original_n_items_in_collisions - new_n_items_in_collisions}")
        
        if new_unique == len(new_semantic_ids):
            self.logger.info(f"\n   ✅ 100% uniqueness achieved!")
        else:
            self.logger.warning(f"\n   ⚠️  Still {len(new_semantic_ids) - new_unique} collisions remaining")
        
        # Save statistics
        stats = {
            'before': {
                'total_items': int(len(original_semantic_ids)),
                'unique_ids': int(original_unique),
                'uniqueness_rate': float(original_unique_rate),
                'collision_groups': int(original_n_collision_groups),
                'items_in_collisions': int(original_n_items_in_collisions)
            },
            'after': {
                'total_items': int(len(new_semantic_ids)),
                'unique_ids': int(new_unique),
                'uniqueness_rate': float(new_unique_rate),
                'collision_groups': int(new_n_collision_groups),
                'items_in_collisions': int(new_n_items_in_collisions)
            },
            'improvement': {
                'unique_ids_improvement': int(improvement),
                'uniqueness_rate_improvement': float(improvement_rate),
                'collision_groups_reduced': int(original_n_collision_groups - new_n_collision_groups),
                'items_in_collisions_reduced': int(original_n_items_in_collisions - new_n_items_in_collisions),
                'achieved_100_percent': (new_unique == len(new_semantic_ids))
            }
        }
        
        stats_path = self.output_dir / 'reassignment_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        self.logger.info(f"\n  ✓ Statistics saved: {stats_path}")
        self.logger.info("=" * 80)
    
    def run(self, codebook_sizes: Optional[List[int]] = None, max_iterations: int = 10):
        """Main function: reassign IDs and save results"""
        self.logger.info("=" * 80)
        self.logger.info("Sinkhorn-based ID Reassignment")
        self.logger.info("=" * 80)
        
        # Save original IDs for comparison
        original_semantic_ids = self.semantic_ids.copy()
        
        # Reassign
        new_semantic_ids = self.reassign_all(
            codebook_sizes=codebook_sizes,
            max_iterations=max_iterations
        )
        
        # Save (with original for comparison)
        self.save_reassigned_ids(new_semantic_ids, original_semantic_ids)
        
        self.logger.info("=" * 80)
        self.logger.info("✓ Reassignment completed!")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)
        
        return new_semantic_ids


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
    parser.add_argument('--max_iterations', type=int, default=10,
                       help='Maximum iterations for reassignment')
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
    reassigner.run(
        codebook_size=args.codebook_size,
        max_iterations=args.max_iterations
    )


if __name__ == '__main__':
    main()

