"""
Advanced Quantization Strategies for RQ-VAE

Implements various quantization methods from TIGER and related papers:
- Straight-Through Estimator (STE)
- Rotation Trick (more efficient gradient flow)
- Gumbel Softmax (differentiable sampling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from abc import ABC, abstractmethod


def sample_gumbel(shape: Tuple, device: torch.device, eps: float = 1e-20) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Draw a sample from the Gumbel-Softmax distribution.
    
    Args:
        logits: Input logits (unnormalized log probabilities)
        temperature: Sampling temperature
        
    Returns:
        Soft sample from the Gumbel-Softmax distribution
    """
    y = logits + sample_gumbel(logits.shape, logits.device)
    return F.softmax(y / temperature, dim=-1)


class QuantizationStrategy(ABC):
    """Base class for quantization strategies"""
    
    @abstractmethod
    def quantize(
        self,
        z: torch.Tensor,
        codebook: torch.Tensor,
        temperature: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input embeddings using the codebook.
        
        Args:
            z: Input embeddings (batch_size, embed_dim)
            codebook: Codebook embeddings (n_embed, embed_dim)
            temperature: Temperature for Gumbel-Softmax (if applicable)
            
        Returns:
            z_q: Quantized embeddings (for forward pass)
            z_q_loss: Quantized embeddings (for loss computation)
            encoding_indices: Codebook indices
        """
        pass


class STEQuantization(QuantizationStrategy):
    """
    Straight-Through Estimator (STE) quantization.
    
    Forward: Use quantized values from codebook
    Backward: Copy gradients straight through (no gradient for quantization)
    """
    
    def quantize(
        self,
        z: torch.Tensor,
        codebook: torch.Tensor,
        temperature: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize using STE.
        
        The gradient flows through as if quantization didn't happen.
        """
        # Compute distances to codebook
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook ** 2, dim=1) - \
            2 * torch.matmul(z, codebook.t())
        
        # Find nearest codebook entries
        encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(encoding_indices, codebook)
        
        # Straight-through estimator: forward uses z_q, backward uses z
        z_q_st = z + (z_q - z).detach()
        
        return z_q_st, z_q, encoding_indices


class RotationTrickQuantization(QuantizationStrategy):
    """
    Rotation Trick quantization for improved gradient flow.
    
    Based on "Rotation Trick: A New Way to Think About Vector Quantization"
    This method provides better gradients than STE by using a differentiable
    rotation and scaling transformation.
    """
    
    def rotate_and_scale(
        self,
        z: torch.Tensor,
        z_q: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply rotation and scaling trick to enable gradient flow.
        
        Args:
            z: Original embeddings (batch_size, embed_dim)
            z_q: Quantized embeddings (batch_size, embed_dim)
            
        Returns:
            Transformed embeddings with gradient flow (batch_size, embed_dim)
        """
        # Detach the quantized embeddings for the transformation
        z_q_detached = z_q.detach()
        z_detached = z.detach()
        
        # Compute norms
        z_q_norms = torch.linalg.vector_norm(z_q_detached, dim=-1, keepdim=True)  # (batch, 1)
        z_norms = torch.linalg.vector_norm(z_detached, dim=-1, keepdim=True)  # (batch, 1)
        
        # Compute scaling factor
        lambda_ = z_q_norms / (z_norms + 1e-8)  # (batch, 1)
        
        # Normalize vectors
        z_normalized = z_detached / (z_norms + 1e-8)  # (batch, embed_dim)
        z_q_normalized = z_q_detached / (z_q_norms + 1e-8)  # (batch, embed_dim)
        
        # Compute normalized sum and normalize it
        normalized_sum = F.normalize(z_normalized + z_q_normalized, p=2, dim=-1)  # (batch, embed_dim)
        
        # Compute projections using einsum to avoid broadcasting issues
        # sum_projection: project z onto normalized_sum direction, then scale by normalized_sum
        # This is: (z · normalized_sum) * normalized_sum
        z_dot_sum = torch.sum(z * normalized_sum, dim=-1, keepdim=True)  # (batch, 1)
        sum_projection = z_dot_sum * normalized_sum  # (batch, embed_dim)
        
        # rescaled: project z onto z_normalized direction, then scale by z_q_normalized
        # This is: (z · z_normalized) * z_q_normalized
        z_dot_z_norm = torch.sum(z * z_normalized, dim=-1, keepdim=True)  # (batch, 1)
        rescaled = z_dot_z_norm * z_q_normalized  # (batch, embed_dim)
        
        # Apply transformation: lambda * (z - 2*sum_projection + 2*rescaled)
        z_q_rotated = lambda_ * (z - 2 * sum_projection + 2 * rescaled)  # (batch, embed_dim)
        
        return z_q_rotated
    
    def quantize(
        self,
        z: torch.Tensor,
        codebook: torch.Tensor,
        temperature: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize using rotation trick.
        """
        # Compute distances to codebook
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook ** 2, dim=1) - \
            2 * torch.matmul(z, codebook.t())
        
        # Find nearest codebook entries
        encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(encoding_indices, codebook)
        
        # Apply rotation trick for gradient flow
        z_q_rotated = self.rotate_and_scale(z, z_q)
        
        return z_q_rotated, z_q, encoding_indices


class GumbelSoftmaxQuantization(QuantizationStrategy):
    """
    Gumbel-Softmax quantization for fully differentiable training.
    
    Uses Gumbel-Softmax to create a soft approximation of the categorical
    distribution over codebook entries.
    """
    
    def quantize(
        self,
        z: torch.Tensor,
        codebook: torch.Tensor,
        temperature: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize using Gumbel-Softmax.
        """
        # Compute distances (negative distances as logits)
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook ** 2, dim=1) - \
            2 * torch.matmul(z, codebook.t())
        
        # Use negative distances as logits (closer = higher probability)
        logits = -d
        
        # Sample from Gumbel-Softmax
        weights = gumbel_softmax_sample(logits, temperature)
        
        # Compute soft quantized embeddings
        z_q_soft = weights @ codebook
        
        # For inference/metrics, also get hard assignment
        encoding_indices = torch.argmin(d, dim=1)
        
        return z_q_soft, z_q_soft, encoding_indices


def create_quantization_strategy(strategy_name: str) -> QuantizationStrategy:
    """
    Factory function to create quantization strategies.
    
    Args:
        strategy_name: Name of strategy ('ste', 'rotation', 'gumbel')
        
    Returns:
        QuantizationStrategy instance
    """
    strategies = {
        'ste': STEQuantization,
        'rotation': RotationTrickQuantization,
        'gumbel': GumbelSoftmaxQuantization
    }
    
    if strategy_name not in strategies:
        raise ValueError(
            f"Unknown quantization strategy: {strategy_name}. "
            f"Choose from {list(strategies.keys())}"
        )
    
    return strategies[strategy_name]()

