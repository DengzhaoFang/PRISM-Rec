import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class RQKMeans(nn.Module):
    """
    Residual Quantization K-Means implementation.
    Applies iterative K-means clustering on residuals for hierarchical quantization.
    """
    
    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        n_layers: int = 4,
        max_iters: int = 100,
        tol: float = 1e-4,
        init_method: str = "kmeans++",
        device: Optional[str] = None
    ):
        """
        Initialize RQ-KMeans module.
        
        Args:
            n_clusters: Number of clusters per layer
            n_features: Feature dimension
            n_layers: Number of quantization layers
            max_iters: Maximum iterations for K-means
            tol: Tolerance for convergence
            init_method: Initialization method ("random" or "kmeans++")
            device: Device to use
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.n_layers = n_layers
        self.max_iters = max_iters
        self.tol = tol
        self.init_method = init_method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize codebooks for each layer 
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(n_clusters, n_features) * 0.1)
            for _ in range(n_layers)
        ])
        
        self.is_trained = False
        
    def _kmeans_plus_plus_init(self, data: torch.Tensor) -> torch.Tensor:
        """K-means++ initialization"""
        n_samples = data.shape[0]
        centroids = torch.zeros(self.n_clusters, self.n_features, device=data.device)
        
        # Choose first centroid randomly
        centroids[0] = data[torch.randint(0, n_samples, (1,))]
        
        for i in range(1, self.n_clusters):
            # Compute distances to nearest centroids
            distances = torch.cdist(data, centroids[:i])
            min_distances = torch.min(distances, dim=1)[0]
            
            # Choose next centroid with probability proportional to squared distance
            probs = min_distances / min_distances.sum()
            cumsum = torch.cumsum(probs, dim=0)
            r = torch.rand(1).item()
            next_idx = torch.searchsorted(cumsum, r)
            centroids[i] = data[next_idx]
            
        return centroids
    
    def _fit_layer(self, data: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Fit K-means for a single layer"""
        n_samples = data.shape[0]
        
        # Initialize centroids
        if self.init_method == "kmeans++":
            centroids = self._kmeans_plus_plus_init(data)
        else:
            centroids = data[torch.randperm(n_samples)[:self.n_clusters]]
            
        prev_loss = float('inf')
        
        for iteration in range(self.max_iters):
            # Assign points to nearest centroids
            distances = torch.cdist(data, centroids)
            assignments = torch.argmin(distances, dim=1)
            
            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    new_centroids[k] = data[mask].mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]  # Keep old centroid
                    
            # Check convergence
            loss = torch.sum((data - centroids[assignments]) ** 2)
            if abs(prev_loss - loss) < self.tol:
                break
                
            centroids = new_centroids
            prev_loss = loss
            
        return centroids
    
    def fit(self, data: torch.Tensor) -> 'RQKMeans':
        """
        Train the RQ-KMeans model.
        
        Args:
            data: Input data of shape (n_samples, n_features)
        """
        data = data.to(self.device)
        residuals = data.clone()
        
        for layer_idx in range(self.n_layers):
            # Fit K-means on current residuals
            centroids = self._fit_layer(residuals, layer_idx)
            self.codebooks[layer_idx].data = centroids
            
            # Compute quantized values and update residuals
            distances = torch.cdist(residuals, centroids)
            assignments = torch.argmin(distances, dim=1)
            quantized = centroids[assignments]
            residuals = residuals - quantized
            
        self.is_trained = True
        return self
    
    def encode(self, data: torch.Tensor) -> torch.Tensor:
        """
        Encode data to quantization codes.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            codes: Quantization codes of shape (n_samples, n_layers)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before encoding")
            
        data = data.to(self.device)
        codes = torch.zeros(data.shape[0], self.n_layers, dtype=torch.long, device=data.device)
        residuals = data.clone()
        
        for layer_idx in range(self.n_layers):
            # Find nearest centroids
            distances = torch.cdist(residuals, self.codebooks[layer_idx])
            assignments = torch.argmin(distances, dim=1)
            codes[:, layer_idx] = assignments
            
            # Update residuals
            quantized = self.codebooks[layer_idx][assignments]
            residuals = residuals - quantized
            
        return codes
    
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode quantization codes to reconstructed data.
        
        Args:
            codes: Quantization codes of shape (n_samples, n_layers)
            
        Returns:
            reconstructed: Reconstructed data of shape (n_samples, n_features)
        """
        codes = codes.to(self.device)
        reconstructed = torch.zeros(codes.shape[0], self.n_features, device=codes.device)
        
        for layer_idx in range(self.n_layers):
            layer_codes = codes[:, layer_idx]
            reconstructed += self.codebooks[layer_idx][layer_codes]
            
        return reconstructed
    
    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.
        
        Args:
            data: Input data of shape (n_samples, n_features)
            
        Returns:
            codes: Quantization codes
            reconstructed: Reconstructed data
        """
        codes = self.encode(data)
        reconstructed = self.decode(codes)
        return codes, reconstructed
    
    def get_reconstruction_loss(self, data: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss (MSE)"""
        _, reconstructed = self.forward(data)
        return F.mse_loss(reconstructed, data)
    
    def save_codebooks(self, path: str):
        """Save trained codebooks"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        torch.save({
            'codebooks': [cb.data.cpu() for cb in self.codebooks],
            'n_clusters': self.n_clusters,
            'n_features': self.n_features,
            'n_layers': self.n_layers
        }, path)
    
    def load_codebooks(self, path: str):
        """Load trained codebooks"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for i, cb_data in enumerate(checkpoint['codebooks']):
            self.codebooks[i].data = cb_data.to(self.device)
        self.is_trained = True
    
    def save_model(self, path: str):
        """Save model (alias for save_codebooks for unified interface)"""
        self.save_codebooks(path)
    
    def load_model(self, path: str):
        """Load model (alias for load_codebooks for unified interface)"""
        self.load_codebooks(path)
        return self