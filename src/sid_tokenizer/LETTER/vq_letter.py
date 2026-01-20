"""
LETTER VectorQuantizer Implementation

Based on the original LETTER paper: "Learnable Item Tokenization for Generative Recommendation"

Key features:
1. Constrained K-Means for codebook initialization and clustering
2. Sinkhorn algorithm for soft assignment (optional)
3. Diversity loss using contrastive learning within clusters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional, Dict, List
from sklearn.cluster import KMeans


def sinkhorn_algorithm(distances: torch.Tensor, epsilon: float, sinkhorn_iterations: int) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm for optimal transport.
    
    Converts distances to a doubly-stochastic assignment matrix.
    
    Args:
        distances: Distance matrix (B, K) - batch_size x num_codes
        epsilon: Temperature parameter (smaller = harder assignment)
        sinkhorn_iterations: Number of iterations
        
    Returns:
        Q: Assignment matrix (B, K)
    """
    Q = torch.exp(-distances / epsilon)
    B = Q.shape[0]  # number of samples
    K = Q.shape[1]  # number of codes
    
    # Make the matrix sum to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q = Q / (sum_Q + 1e-10)
    
    for _ in range(sinkhorn_iterations):
        # Normalize columns
        Q = Q / (torch.sum(Q, dim=1, keepdim=True) + 1e-10)
        Q = Q / B
        # Normalize rows
        Q = Q / (torch.sum(Q, dim=0, keepdim=True) + 1e-10)
        Q = Q / K
    
    Q = Q * B  # columns must sum to 1
    return Q


class LETTERQuantizer(nn.Module):
    """
    Vector Quantizer with LETTER's diversity loss.
    
    This implements the exact diversity loss from the LETTER paper:
    - Uses constrained K-means to cluster codebook vectors
    - Diversity loss pulls quantized vectors toward same-cluster codes
    """
    
    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        mu: float = 0.25,  # commitment loss weight
        beta: float = 0.1,  # diversity loss weight
        n_clusters: int = 10,  # number of clusters for diversity loss
        kmeans_init: bool = True,
        kmeans_iters: int = 100,
        sk_epsilon: float = 0.0,  # Sinkhorn epsilon (0 = disabled)
        sk_iters: int = 100,
    ):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.mu = mu
        self.beta = beta
        self.n_clusters = n_clusters
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        
        # Codebook
        self.embedding = nn.Embedding(n_embed, embed_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()
        
        # Cluster labels for each code (updated periodically)
        self.register_buffer('cluster_labels', torch.zeros(n_embed, dtype=torch.long))
        # Indices list for each cluster
        self.cluster_indices: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
    
    def get_codebook(self) -> torch.Tensor:
        return self.embedding.weight
    
    def init_codebook(self, data: torch.Tensor):
        """Initialize codebook using constrained K-means clustering."""
        print(f"Initializing codebook with K-means (n_embed={self.n_embed}, data={len(data)})...")
        x = data.detach().cpu().numpy()
        
        # Use constrained K-means for better initialization
        try:
            from k_means_constrained import KMeansConstrained
            
            # Calculate size constraints based on data size
            n_samples = len(x)
            size_min = max(1, n_samples // (self.n_embed * 2))
            size_max = max(size_min + 1, n_samples // self.n_embed * 4)
            
            clf = KMeansConstrained(
                n_clusters=self.n_embed,
                size_min=size_min,
                size_max=size_max,
                max_iter=self.kmeans_iters,
                n_init=10,
                n_jobs=-1,
                verbose=False
            )
            clf.fit(x)
            centers = torch.from_numpy(clf.cluster_centers_).float()
            print(f"  ✓ Used constrained K-means (size_min={size_min}, size_max={size_max})")
        except ImportError:
            print("  Warning: k_means_constrained not installed, using regular KMeans")
            kmeans = KMeans(n_clusters=self.n_embed, max_iter=self.kmeans_iters, n_init=10)
            kmeans.fit(x)
            centers = torch.from_numpy(kmeans.cluster_centers_).float()
        
        self.embedding.weight.data.copy_(centers.to(self.embedding.weight.device))
        self.initted = True
        print(f"✓ Codebook initialized")
    
    def update_cluster_labels(self):
        """
        Update cluster assignments for codebook vectors.
        Uses constrained K-means to ensure balanced cluster sizes (following LETTER paper).
        """
        codebook = self.embedding.weight.detach().cpu().numpy()
        
        # Check if codebook is all zeros (not initialized)
        if np.allclose(codebook, 0):
            print("  Warning: Codebook is all zeros, skipping cluster update")
            return
        
        # Use constrained K-means for balanced clusters (as in original LETTER)
        try:
            from k_means_constrained import KMeansConstrained
            
            # Ensure balanced cluster sizes
            size_min = max(1, self.n_embed // (self.n_clusters * 2))
            size_max = min(self.n_embed, self.n_embed // self.n_clusters * 4)
            
            clf = KMeansConstrained(
                n_clusters=self.n_clusters,
                size_min=size_min,
                size_max=size_max,
                max_iter=100,
                n_init=10,
                n_jobs=-1,
                verbose=False
            )
            clf.fit(codebook)
            labels = clf.labels_
        except ImportError:
            print("  Warning: k_means_constrained not installed, using regular KMeans")
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, max_iter=100)
            labels = kmeans.fit_predict(codebook)
        
        self.cluster_labels = torch.from_numpy(labels).long().to(self.embedding.weight.device)
        
        # Build indices list for each cluster
        self.cluster_indices = {i: [] for i in range(self.n_clusters)}
        for idx, label in enumerate(labels):
            self.cluster_indices[int(label)].append(idx)
        
        # Log cluster sizes
        sizes = [len(self.cluster_indices[i]) for i in range(self.n_clusters)]
        print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        
        # Verify cluster_indices is populated
        total_codes = sum(len(v) for v in self.cluster_indices.values())
        print(f"  Total codes in clusters: {total_codes}/{self.n_embed}")
    
    def diversity_loss(
        self, 
        x_q: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diversity loss following LETTER paper.
        
        For each quantized vector x_q, we:
        1. Find which cluster its assigned code belongs to
        2. Sample a positive code from the same cluster (excluding self)
        3. Compute cross-entropy loss to pull x_q toward the positive
        
        Args:
            x_q: Quantized vectors (batch_size, embed_dim)
            indices: Assigned code indices (batch_size,)
            
        Returns:
            diversity_loss: Scalar loss
        """
        batch_size = x_q.shape[0]
        emb = self.embedding.weight  # (n_embed, embed_dim)
        temperature = 1.0
        
        # Get cluster labels for assigned codes
        indices_cluster = self.cluster_labels[indices].tolist()  # (batch_size,)
        
        # Sample positive codes from same cluster
        pos_samples = []
        valid_mask = []
        
        for idx in range(batch_size):
            code_idx = indices[idx].item()
            cluster_id = indices_cluster[idx]
            cluster_codes = self.cluster_indices.get(cluster_id, [])
            
            # Filter out self
            candidates = [c for c in cluster_codes if c != code_idx]
            
            if len(candidates) > 0:
                pos_idx = random.choice(candidates)
                pos_samples.append(pos_idx)
                valid_mask.append(True)
            else:
                pos_samples.append(0)  # placeholder
                valid_mask.append(False)
        
        if not any(valid_mask):
            return torch.tensor(0.0, device=x_q.device, requires_grad=True)
        
        # Compute similarities: x_q @ emb.T -> (batch_size, n_embed)
        sim = torch.matmul(x_q, emb.t()) / temperature
        
        # Mask out self-similarity (set to large negative)
        batch_indices = torch.arange(batch_size, device=x_q.device)
        sim[batch_indices, indices] = -1e12
        
        # Target: positive sample indices
        y_true = torch.tensor(pos_samples, device=x_q.device, dtype=torch.long)
        valid_mask = torch.tensor(valid_mask, device=x_q.device)
        
        # Compute cross-entropy only for valid samples
        if valid_mask.sum() > 0:
            loss = F.cross_entropy(sim[valid_mask], y_true[valid_mask])
        else:
            loss = torch.tensor(0.0, device=x_q.device, requires_grad=True)
        
        return loss
    
    @staticmethod
    def center_distance_for_constraint(distances: torch.Tensor) -> torch.Tensor:
        """Normalize distances for Sinkhorn algorithm."""
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        centered_distances = (distances - middle) / amplitude
        return centered_distances
    
    def forward(
        self, 
        x: torch.Tensor, 
        use_sk: bool = False,
        compute_diversity: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with quantization and diversity loss.
        
        Args:
            x: Input tensor (batch_size, embed_dim)
            use_sk: Whether to use Sinkhorn algorithm for soft assignment
            compute_diversity: Whether to compute diversity loss
            
        Returns:
            x_q: Quantized tensor (batch_size, embed_dim)
            loss: Total VQ loss (codebook + commitment + diversity)
            indices: Assigned code indices (batch_size,)
            diversity_loss: Diversity loss value
        """
        latent = x.view(-1, self.embed_dim)
        
        # Initialize codebook on first forward pass (if not already initialized)
        if not self.initted and self.training:
            self.init_codebook(latent)
            self.update_cluster_labels()
        
        # Ensure cluster_indices is populated (might be empty after loading)
        if not self.cluster_indices or all(len(v) == 0 for v in self.cluster_indices.values()):
            self.update_cluster_labels()
        
        # Compute L2 distances to codebook
        d = (torch.sum(latent ** 2, dim=1, keepdim=True) + 
             torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t() - 
             2 * torch.matmul(latent, self.embedding.weight.t()))
        
        # Quantization
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d_centered = self.center_distance_for_constraint(d)
            d_centered = d_centered.double()
            Q = sinkhorn_algorithm(d_centered, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Warning: Sinkhorn returned nan/inf, falling back to argmin")
                indices = torch.argmin(d, dim=-1)
            else:
                indices = torch.argmax(Q, dim=-1)
        
        # Get quantized vectors
        x_q = self.embedding(indices).view(x.shape)
        
        # Compute losses
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        
        # Diversity loss
        if compute_diversity and self.beta > 0 and self.training:
            div_loss = self.diversity_loss(x_q, indices)
        else:
            div_loss = torch.tensor(0.0, device=x.device)
        
        # Total loss
        loss = codebook_loss + self.mu * commitment_loss + self.beta * div_loss
        
        # Straight-through estimator
        x_q = x + (x_q - x).detach()
        
        return x_q, loss, indices, div_loss


class LETTERResidualQuantizer(nn.Module):
    """
    Residual Vector Quantizer with LETTER's diversity loss.
    
    Applies multiple layers of quantization, each operating on the residual
    from the previous layer.
    """
    
    def __init__(
        self,
        n_embed_list: List[int],
        embed_dim: int,
        mu: float = 0.25,
        beta: float = 0.1,
        n_clusters: int = 10,
        kmeans_init: bool = True,
        kmeans_iters: int = 100,
        sk_epsilons: Optional[List[float]] = None,
        sk_iters: int = 100,
    ):
        super().__init__()
        self.n_embed_list = n_embed_list
        self.embed_dim = embed_dim
        self.num_layers = len(n_embed_list)
        
        if sk_epsilons is None:
            sk_epsilons = [0.0] * self.num_layers
        
        self.quantizers = nn.ModuleList([
            LETTERQuantizer(
                n_embed=n_embed,
                embed_dim=embed_dim,
                mu=mu,
                beta=beta,
                n_clusters=n_clusters,
                kmeans_init=kmeans_init,
                kmeans_iters=kmeans_iters,
                sk_epsilon=sk_epsilon,
                sk_iters=sk_iters,
            )
            for n_embed, sk_epsilon in zip(n_embed_list, sk_epsilons)
        ])
    
    def get_codebook(self) -> torch.Tensor:
        """Get all codebooks stacked."""
        return torch.stack([q.get_codebook() for q in self.quantizers])
    
    def update_all_cluster_labels(self):
        """Update cluster labels for all quantizer layers."""
        for idx, quantizer in enumerate(self.quantizers):
            print(f"Updating clusters for layer {idx}...")
            quantizer.update_cluster_labels()
    
    def forward(
        self, 
        x: torch.Tensor, 
        use_sk: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual quantization.
        
        Args:
            x: Input tensor (batch_size, embed_dim)
            use_sk: Whether to use Sinkhorn algorithm
            
        Returns:
            x_q: Quantized tensor (batch_size, embed_dim)
            total_loss: Sum of losses from all layers
            all_indices: Indices from all layers (batch_size, num_layers)
            total_div_loss: Sum of diversity losses
        """
        all_losses = []
        all_indices = []
        all_div_losses = []
        
        x_q = torch.zeros_like(x)
        residual = x
        
        for quantizer in self.quantizers:
            x_res, loss, indices, div_loss = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)
            all_div_losses.append(div_loss)
        
        total_loss = torch.stack(all_losses).mean()
        total_div_loss = torch.stack(all_div_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        
        return x_q, total_loss, all_indices, total_div_loss
