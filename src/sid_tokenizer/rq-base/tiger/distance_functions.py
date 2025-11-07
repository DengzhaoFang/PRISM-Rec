"""
Distance Functions for Vector Quantization

Implements efficient distance computation with optional batching for memory efficiency.
"""

import torch
from typing import Optional


class SquaredEuclideanDistance:
    """
    Compute squared Euclidean distances efficiently.
    
    Supports optional batching for memory-constrained scenarios.
    """
    
    @staticmethod
    def compute(
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute squared Euclidean distances between rows of x and rows of y.
        
        Uses the identity: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x @ y.T
        
        Args:
            x: Data points of shape (n1, d)
            y: Centroids of shape (n2, d)
            batch_size: Optional batch size for x-axis batching (for large n1)
            
        Returns:
            Squared distances of shape (n1, n2)
        """
        assert x.dim() == 2, f"x must be 2D, got {x.dim()} dimensions"
        assert y.dim() == 2, f"y must be 2D, got {y.dim()} dimensions"
        assert x.size(1) == y.size(1), "x and y must have same feature dimension"
        
        n1, d = x.shape
        n2, _ = y.shape
        
        if batch_size is None or batch_size >= n1:
            # No batching: compute directly
            x_sq = torch.sum(x ** 2, dim=1, keepdim=True)  # (n1, 1)
            y_sq = torch.sum(y ** 2, dim=1)  # (n2,)
            xy = torch.matmul(x, y.t())  # (n1, n2)
            
            distances = x_sq + y_sq - 2 * xy
            return distances
        else:
            # Batching: process x in chunks
            all_distances = []
            num_batches = (n1 + batch_size - 1) // batch_size
            
            # Precompute y squared norms
            y_sq = torch.sum(y ** 2, dim=1)  # (n2,)
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n1)
                x_batch = x[start_idx:end_idx]  # (batch_size, d)
                
                # Compute for this batch
                x_batch_sq = torch.sum(x_batch ** 2, dim=1, keepdim=True)  # (batch_size, 1)
                xy_batch = torch.matmul(x_batch, y.t())  # (batch_size, n2)
                
                distances_batch = x_batch_sq + y_sq - 2 * xy_batch
                all_distances.append(distances_batch)
            
            return torch.cat(all_distances, dim=0)


class CosineDistance:
    """
    Compute cosine distance (1 - cosine similarity).
    
    Useful for normalized embeddings.
    """
    
    @staticmethod
    def compute(
        x: torch.Tensor,
        y: torch.Tensor,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute cosine distance.
        
        Args:
            x: Data points of shape (n1, d)
            y: Centroids of shape (n2, d)
            batch_size: Optional batch size for batching
            
        Returns:
            Cosine distances of shape (n1, n2)
        """
        # Normalize
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
        
        # Compute cosine similarity
        if batch_size is None or batch_size >= x.size(0):
            similarity = torch.matmul(x_norm, y_norm.t())
        else:
            # Batched computation
            all_sim = []
            num_batches = (x.size(0) + batch_size - 1) // batch_size
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, x.size(0))
                x_batch = x_norm[start_idx:end_idx]
                
                sim_batch = torch.matmul(x_batch, y_norm.t())
                all_sim.append(sim_batch)
            
            similarity = torch.cat(all_sim, dim=0)
        
        # Convert to distance
        distance = 1.0 - similarity
        return distance

