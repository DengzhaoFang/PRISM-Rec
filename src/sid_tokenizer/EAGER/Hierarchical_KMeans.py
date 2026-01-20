"""
Hierarchical K-Means (HKM) for EAGER Semantic ID Generation

This module implements hierarchical K-means clustering for generating semantic IDs
in the EAGER dual-stream recommender system. Each item gets a path from root to leaf
in a hierarchical clustering tree, which serves as its semantic ID.

Reference: EAGER paper Section 3.1 - Dual Codes generation
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
import logging


class TreeNode:
    """
    Node in the hierarchical K-means tree.
    Each node represents a cluster and can have K children.
    """
    
    def __init__(self, node_id: int, depth: int, center: Optional[np.ndarray] = None):
        """
        Initialize a tree node.
        
        Args:
            node_id: Unique identifier for this node (cluster ID at this level)
            depth: Depth of this node in the tree (0 = root)
            center: Cluster center (mean of assigned points)
        """
        self.node_id = node_id
        self.depth = depth
        self.center = center
        self.children: List[TreeNode] = []
        self.item_indices: List[int] = []  # Item indices assigned to this cluster
        
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0
    
    def add_child(self, child: 'TreeNode'):
        """Add a child node"""
        self.children.append(child)
        
    def assign_items(self, indices: List[int]):
        """Assign item indices to this node"""
        self.item_indices = indices


class HierarchicalKMeans(nn.Module):
    """
    Hierarchical K-Means for semantic ID generation.
    
    Builds a K-ary tree by recursively applying K-means clustering.
    Each item's semantic ID is the path from root to leaf.
    
    Example with K=4, depth=3:
        - Level 0: 1 cluster (root)
        - Level 1: 4 clusters (IDs 0-3)
        - Level 2: 16 clusters (IDs 0-3 per parent)
        - Level 3: 64 clusters (IDs 0-3 per parent)
        - Semantic ID: [c1, c2, c3] where c_i ∈ {0, 1, 2, 3}
    """
    
    def __init__(
        self,
        k: int = 8,
        max_depth: int = 3,
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        device: Optional[str] = None
    ):
        """
        Initialize Hierarchical K-Means.
        
        Args:
            k: Branching factor (number of clusters at each level)
            max_depth: Maximum depth of the tree (length of semantic ID)
            random_state: Random seed for reproducibility
            n_init: Number of K-means initializations
            max_iter: Maximum iterations for K-means
            device: Device for PyTorch operations
        """
        super().__init__()
        self.k = k
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.root: Optional[TreeNode] = None
        self.is_trained = False
        
        # For efficient encoding: map item index to path
        self.item_to_path: Dict[int, List[int]] = {}
        
        self.logger = logging.getLogger(__name__)
        
    def _build_tree_recursive(
        self,
        data: np.ndarray,
        item_indices: List[int],
        parent: Optional[TreeNode],
        depth: int
    ) -> TreeNode:
        """
        Recursively build the hierarchical tree using K-means.
        
        Args:
            data: Data points for this subtree (n_samples, n_features)
            item_indices: Original indices of items in this subtree
            parent: Parent node (None for root)
            depth: Current depth in the tree
            
        Returns:
            Root node of the subtree
        """
        n_samples = data.shape[0]
        
        # Base case: reached max depth
        if depth >= self.max_depth:
            # Create leaf node
            node = TreeNode(node_id=0, depth=depth, center=data.mean(axis=0))
            node.assign_items(item_indices)
            return node
        
        # Base case: too few samples to cluster (need at least 2 for K-means)
        if n_samples < 2:
            # Create leaf node
            node = TreeNode(node_id=0, depth=depth, center=data.mean(axis=0))
            node.assign_items(item_indices)
            return node
        
        # Adaptive K: use fewer clusters if we don't have enough samples
        # Ensure at least 2 clusters (for tree structure) and at least 1 sample per cluster
        effective_k = min(self.k, max(2, n_samples))
        
        # Apply K-means clustering with adaptive K
        kmeans = KMeans(
            n_clusters=effective_k,
            random_state=self.random_state + depth,  # Different seed per level
            n_init=self.n_init,
            max_iter=self.max_iter,
            algorithm='lloyd'
        )
        labels = kmeans.fit_predict(data)
        
        # Create root node for this subtree
        root = TreeNode(node_id=0, depth=depth, center=data.mean(axis=0))
        
        
        # Recursively build children
        for cluster_id in range(effective_k):
            # Get items assigned to this cluster
            cluster_mask = labels == cluster_id
            cluster_data = data[cluster_mask]
            cluster_item_indices = [item_indices[i] for i in range(len(item_indices)) if cluster_mask[i]]
            
            if len(cluster_data) == 0:
                continue
                
            # Create child node
            child = TreeNode(
                node_id=cluster_id,
                depth=depth + 1,
                center=kmeans.cluster_centers_[cluster_id]
            )
            
            # Recursively build subtree if not at max depth
            # FIXED: Changed from 'depth + 1 < self.max_depth' to 'depth < self.max_depth'
            # This ensures we build the tree to the full specified depth
            if depth < self.max_depth:
                child = self._build_tree_recursive(
                    cluster_data,
                    cluster_item_indices,
                    root,
                    depth + 1
                )
                child.node_id = cluster_id  # Ensure correct cluster ID
            else:
                child.assign_items(cluster_item_indices)
            
            root.add_child(child)
        
        return root
    
    def _extract_paths(self, node: TreeNode, current_path: List[int]):
        """
        Extract paths from root to all leaves (DFS traversal).
        
        Args:
            node: Current node
            current_path: Path from root to current node
        """
        if node.is_leaf():
            # Assign this path to all items in this leaf (without padding yet)
            for item_idx in node.item_indices:
                self.item_to_path[item_idx] = current_path.copy()
        else:
            # Recursively traverse children
            for child in node.children:
                self._extract_paths(child, current_path + [child.node_id])
    
    def _apply_deduplication_padding(self):
        """
        Apply padding with deduplication to ensure all semantic IDs are unique.
        
        Strategy:
        1. For paths shorter than max_depth: pad with 0s, then assign unique values to last position
        2. For paths at max_depth with collisions: increment last position for duplicates
        
        This ensures all IDs have exactly max_depth layers and are unique.
        """
        # Group items by their current path
        path_groups = defaultdict(list)
        for item_idx, path in self.item_to_path.items():
            path_tuple = tuple(path)
            path_groups[path_tuple].append(item_idx)
        
        # Track used codes to avoid new collisions
        used_codes = set()
        
        # Apply padding with deduplication
        for path_tuple, item_indices in path_groups.items():
            path_list = list(path_tuple)
            current_depth = len(path_list)
            
            # Pad to max_depth
            while len(path_list) < self.max_depth:
                path_list.append(0)
            
            # Handle collisions
            if len(item_indices) == 1:
                # No collision, use as is
                item_idx = item_indices[0]
                final_path = tuple(path_list)
                
                # Check if this code is already used (shouldn't happen, but be safe)
                counter = 0
                while final_path in used_codes:
                    path_list[-1] = counter
                    final_path = tuple(path_list)
                    counter += 1
                
                self.item_to_path[item_idx] = list(final_path)
                used_codes.add(final_path)
            else:
                # Collision: assign unique codes by incrementing last position
                for idx, item_idx in enumerate(sorted(item_indices)):
                    unique_path = path_list.copy()
                    
                    # Try incrementing the last position until we find an unused code
                    counter = idx
                    final_path = tuple(unique_path[:-1] + [unique_path[-1] + counter])
                    
                    while final_path in used_codes:
                        counter += 1
                        final_path = tuple(unique_path[:-1] + [unique_path[-1] + counter])
                    
                    self.item_to_path[item_idx] = list(final_path)
                    used_codes.add(final_path)
    
    def fit(self, data: torch.Tensor) -> 'HierarchicalKMeans':
        """
        Build the hierarchical K-means tree.
        
        Args:
            data: Input embeddings of shape (n_items, embedding_dim)
            
        Returns:
            self
        """
        self.logger.info(f"Building Hierarchical K-Means tree (K={self.k}, depth={self.max_depth})...")
        
        # Convert to numpy for sklearn
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = np.array(data)
        
        n_items = data_np.shape[0]
        item_indices = list(range(n_items))
        
        # Build tree recursively
        self.root = self._build_tree_recursive(data_np, item_indices, None, depth=0)
        
        # Extract paths for all items (without padding)
        self.item_to_path = {}
        self._extract_paths(self.root, [])
        
        # Apply padding with deduplication to ensure uniqueness
        self._apply_deduplication_padding()
        
        self.is_trained = True
        
        # Log statistics
        unique_paths = len(set(tuple(p) for p in self.item_to_path.values()))
        self.logger.info(f"✓ HKM tree built successfully")
        self.logger.info(f"  Total items: {n_items}")
        self.logger.info(f"  Unique paths: {unique_paths}")
        if unique_paths != n_items:
            self.logger.warning(f"  ⚠ Warning: {n_items - unique_paths} collisions detected even after deduplication!")
        
        return self
    
    def encode(self, data: torch.Tensor, item_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Encode data to semantic IDs (paths in the tree).
        
        Args:
            data: Input embeddings (n_items, embedding_dim)
            item_indices: Original item indices (if known from training)
                         If None, will find nearest leaf in the tree
            
        Returns:
            codes: Semantic ID codes of shape (n_items, max_depth)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained (fit) before encoding")
        
        n_items = data.shape[0]
        codes = torch.zeros(n_items, self.max_depth, dtype=torch.long)
        
        if item_indices is not None:
            # Use pre-computed paths from training
            for i, item_idx in enumerate(item_indices):
                if item_idx in self.item_to_path:
                    path = self.item_to_path[item_idx]
                    codes[i, :len(path)] = torch.tensor(path, dtype=torch.long)
        else:
            # Find nearest path for new data (inference mode)
            data_np = data.cpu().numpy() if isinstance(data, torch.Tensor) else np.array(data)
            
            for i in range(n_items):
                path = self._find_nearest_path(data_np[i])
                codes[i, :len(path)] = torch.tensor(path, dtype=torch.long)
        
        return codes
    
    def _find_nearest_path(self, data_point: np.ndarray) -> List[int]:
        """
        Find the path to the nearest leaf for a data point (inference).
        
        Args:
            data_point: Single data point (embedding_dim,)
            
        Returns:
            path: Path from root to nearest leaf
        """
        path = []
        current_node = self.root
        
        while not current_node.is_leaf():
            # Find nearest child
            min_dist = float('inf')
            nearest_child = None
            
            for child in current_node.children:
                dist = np.linalg.norm(data_point - child.center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_child = child
            
            if nearest_child is None:
                break
                
            path.append(nearest_child.node_id)
            current_node = nearest_child
        
        return path
    
    def get_collision_stats(self) -> Dict[str, any]:
        """
        Compute collision statistics at each hierarchical level.
        
        Returns:
            stats: Dictionary with collision rates and counts per level
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before computing statistics")
        
        total_items = len(self.item_to_path)
        stats = {
            'total_items': total_items,
            'max_depth': self.max_depth,
            'branching_factor': self.k,
            'levels': {}
        }
        
        # Analyze each level
        for level in range(1, self.max_depth + 1):
            # Group items by prefix (first 'level' codes)
            prefix_groups = defaultdict(list)
            for item_idx, path in self.item_to_path.items():
                prefix = tuple(path[:level])
                prefix_groups[prefix].append(item_idx)
            
            unique_prefixes = len(prefix_groups)
            duplicate_items = total_items - unique_prefixes
            collision_rate = duplicate_items / total_items if total_items > 0 else 0.0
            
            # Find max collision group
            max_group_size = max(len(items) for items in prefix_groups.values()) if prefix_groups else 0
            
            stats['levels'][level] = {
                'unique_codes': unique_prefixes,
                'duplicate_items': duplicate_items,
                'collision_rate': collision_rate,
                'max_collision_group': max_group_size,
                'theoretical_max': self.k ** level
            }
        
        return stats
    
    def save_model(self, path: str):
        """Save the trained HKM model"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        # We only need to save the item_to_path mapping
        # The tree structure can be reconstructed if needed
        torch.save({
            'k': self.k,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'item_to_path': self.item_to_path,
            'is_trained': self.is_trained
        }, path)
        
        self.logger.info(f"HKM model saved to: {path}")
    
    def load_model(self, path: str):
        """Load a trained HKM model"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        self.k = checkpoint['k']
        self.max_depth = checkpoint['max_depth']
        self.random_state = checkpoint['random_state']
        self.item_to_path = checkpoint['item_to_path']
        self.is_trained = checkpoint['is_trained']
        
        self.logger.info(f"HKM model loaded from: {path}")
        return self
