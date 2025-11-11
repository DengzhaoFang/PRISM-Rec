"""
HID-VAE Loss Functions

Implements specialized loss functions for Hierarchical ID VAE:
1. Cosine Similarity Reconstruction Loss (multi-modal)
2. Tag Anchoring Loss (semantic guidance)
3. Codebook Balance Loss (prevent collapse)
4. Hierarchical Classification Loss (with masking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity-based reconstruction loss for multi-modal embeddings.
    Scale-invariant and suitable for embeddings of different dimensions.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 1 - cosine_similarity as loss.
        
        Args:
            pred: Predicted embedding (batch_size, dim)
            target: Target embedding (batch_size, dim)
            
        Returns:
            loss: Scalar loss value in range [0, 2]
        """
        # Normalize to unit vectors
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        
        # Cosine similarity: dot product of normalized vectors
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)
        
        # Loss: 1 - similarity (range [0, 2])
        loss = 1.0 - cosine_sim
        
        return loss.mean()


class MultiModalReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss for content and collaborative embeddings.
    Uses cosine similarity to avoid scale conflicts between different modalities.
    """
    
    def __init__(
        self, 
        lambda_content: float = 1.0, 
        lambda_collab: float = 1.0
    ):
        """
        Args:
            lambda_content: Weight for content embedding reconstruction
            lambda_collab: Weight for collaborative embedding reconstruction
        """
        super().__init__()
        self.lambda_content = lambda_content
        self.lambda_collab = lambda_collab
        self.cosine_loss = CosineSimilarityLoss()
        
    def forward(
        self,
        pred_content: torch.Tensor,
        target_content: torch.Tensor,
        pred_collab: torch.Tensor,
        target_collab: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-modal reconstruction loss.
        
        Args:
            pred_content: Predicted content embedding (batch_size, 768)
            target_content: Target content embedding (batch_size, 768)
            pred_collab: Predicted collaborative embedding (batch_size, 64)
            target_collab: Target collaborative embedding (batch_size, 64)
            
        Returns:
            loss: Combined loss scalar
            loss_dict: Dictionary with individual loss components
        """
        # Content reconstruction loss
        loss_content = self.cosine_loss(pred_content, target_content)
        
        # Collaborative reconstruction loss
        loss_collab = self.cosine_loss(pred_collab, target_collab)
        
        # Combined loss
        loss_total = (
            self.lambda_content * loss_content + 
            self.lambda_collab * loss_collab
        )
        
        loss_dict = {
            'recon_content': loss_content.item(),
            'recon_collab': loss_collab.item(),
            'recon_total': loss_total.item()
        }
        
        return loss_total, loss_dict


class TagAnchoringLoss(nn.Module):
    """
    Tag anchoring loss that aligns codebook vectors with tag semantic space.
    Uses projection layers to map tag embeddings to codebook dimension.
    """
    
    def __init__(
        self,
        tag_embed_dim: int = 768,
        codebook_dim: int = 32,
        n_layers: int = 3,
        beta_weights: Optional[List[float]] = None
    ):
        """
        Args:
            tag_embed_dim: Dimension of tag embeddings (768)
            codebook_dim: Dimension of codebook vectors (32)
            n_layers: Number of RQ layers
            beta_weights: Weights for each layer's anchor loss
        """
        super().__init__()
        self.tag_embed_dim = tag_embed_dim
        self.codebook_dim = codebook_dim
        self.n_layers = n_layers
        
        # Default weights if not provided
        if beta_weights is None:
            beta_weights = [1.0] * n_layers
        self.beta_weights = beta_weights
        
        # Projection layers: map 768D tag embeddings to 32D codebook space
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tag_embed_dim, codebook_dim * 2),
                nn.LayerNorm(codebook_dim * 2),
                nn.ReLU(),
                nn.Linear(codebook_dim * 2, codebook_dim)
            )
            for _ in range(n_layers)
        ])
        
    def forward(
        self,
        tag_embeddings_per_layer: List[torch.Tensor],  # List of (n_tags_l, 768)
        codebooks: List[torch.Tensor]  # List of (n_embed_l, 32)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute tag anchoring loss for all layers.
        
        Args:
            tag_embeddings_per_layer: List of tag embeddings for each layer
                [L2_tags (6, 768), L3_tags (37, 768), L4_tags (148, 768)]
            codebooks: List of codebook tensors for each layer
                [C1 (n_embed, 32), C2 (n_embed, 32), C3 (n_embed, 32)]
                
        Returns:
            loss: Combined anchor loss
            loss_dict: Dictionary with per-layer losses
        """
        total_loss = 0.0
        loss_dict = {}
        
        for layer_idx in range(self.n_layers):
            # Get tag embeddings and codebook for this layer
            tag_emb = tag_embeddings_per_layer[layer_idx]  # (n_tags, 768)
            codebook = codebooks[layer_idx]  # (n_embed, 32)
            
            if tag_emb is None or len(tag_emb) == 0:
                continue
                
            # Project tag embeddings to codebook space
            proj_tag_emb = self.projections[layer_idx](tag_emb)  # (n_tags, 32)
            
            # Quantize projected tags using current codebook
            # Compute distances to all codebook vectors
            # proj_tag_emb: (n_tags, 32), codebook: (n_embed, 32)
            distances = torch.cdist(
                proj_tag_emb.unsqueeze(0), 
                codebook.unsqueeze(0), 
                p=2
            ).squeeze(0)  # (n_tags, n_embed)
            
            # Find nearest codebook vector for each tag
            min_encoding_indices = torch.argmin(distances, dim=1)  # (n_tags,)
            quantized_tags = codebook[min_encoding_indices]  # (n_tags, 32)
            
            # Anchor loss: MSE between projected tags and quantized tags
            layer_loss = F.mse_loss(proj_tag_emb, quantized_tags)
            
            # Weighted sum
            total_loss += self.beta_weights[layer_idx] * layer_loss
            loss_dict[f'anchor_layer{layer_idx+1}'] = layer_loss.item()
        
        loss_dict['anchor_total'] = total_loss.item()
        return total_loss, loss_dict


class CodebookBalanceLoss(nn.Module):
    """
    Codebook balance loss using KL divergence to encourage uniform usage.
    Prevents codebook collapse by penalizing non-uniform usage distributions.
    """
    
    def __init__(
        self,
        n_layers: int = 3,
        gamma_weights: Optional[List[float]] = None,
        eps: float = 1e-10
    ):
        """
        Args:
            n_layers: Number of RQ layers
            gamma_weights: Weights for each layer's balance loss
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.n_layers = n_layers
        self.eps = eps
        
        if gamma_weights is None:
            gamma_weights = [1.0] * n_layers
        self.gamma_weights = gamma_weights
        
    def forward(
        self,
        encoding_indices_per_layer: List[torch.Tensor],  # List of (batch_size,)
        n_embed_per_layer: List[int]  # List of codebook sizes
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute codebook balance loss for all layers.
        
        Args:
            encoding_indices_per_layer: List of selected codebook indices per layer
            n_embed_per_layer: List of codebook sizes for each layer
            
        Returns:
            loss: Combined balance loss
            loss_dict: Dictionary with per-layer losses and usage stats
        """
        total_loss = 0.0
        loss_dict = {}
        
        for layer_idx in range(self.n_layers):
            indices = encoding_indices_per_layer[layer_idx]  # (batch_size,)
            n_embed = n_embed_per_layer[layer_idx]
            
            # Compute observed usage distribution
            bincount = torch.bincount(indices, minlength=n_embed).float()
            p_obs = bincount / (bincount.sum() + self.eps)  # Normalize to probability
            
            # Uniform target distribution
            p_uniform = torch.ones_like(p_obs) / n_embed
            
            # KL divergence: KL(p_obs || p_uniform)
            # KL = sum(p_obs * log(p_obs / p_uniform))
            kl_div = F.kl_div(
                torch.log(p_obs + self.eps),
                p_uniform,
                reduction='sum',
                log_target=False
            )
            
            # Weighted sum
            total_loss += self.gamma_weights[layer_idx] * kl_div
            
            # Track usage statistics
            used_codes = (bincount > 0).sum().item()
            usage_ratio = used_codes / n_embed
            
            loss_dict[f'balance_layer{layer_idx+1}'] = kl_div.item()
            loss_dict[f'usage_layer{layer_idx+1}'] = usage_ratio
        
        loss_dict['balance_total'] = total_loss.item()
        return total_loss, loss_dict


class HierarchicalClassificationLoss(nn.Module):
    """
    Hierarchical classification loss with masking for variable-length tag sequences.
    Each layer predicts its corresponding tag category.
    """
    
    def __init__(
        self,
        n_layers: int = 3,
        delta_weight: float = 1.0,
        ignore_index: int = 0  # PAD token index
    ):
        """
        Args:
            n_layers: Number of classification layers
            delta_weight: Weight for classification loss
            ignore_index: Index to ignore in loss computation (PAD token)
        """
        super().__init__()
        self.n_layers = n_layers
        self.delta_weight = delta_weight
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(
        self,
        predictions_per_layer: List[torch.Tensor],  # List of (batch, n_classes_l)
        targets_per_layer: List[torch.Tensor],  # List of (batch,)
        masks_per_layer: Optional[List[torch.Tensor]] = None  # List of (batch,)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hierarchical classification loss.
        
        Args:
            predictions_per_layer: List of logits for each layer
            targets_per_layer: List of target tag IDs for each layer
            masks_per_layer: Optional masks (1 for valid, 0 for PAD)
            
        Returns:
            loss: Combined classification loss
            loss_dict: Dictionary with per-layer losses and accuracies
        """
        total_loss = 0.0
        loss_dict = {}
        num_valid_layers = 0
        
        for layer_idx in range(self.n_layers):
            pred = predictions_per_layer[layer_idx]  # (batch, n_classes)
            target = targets_per_layer[layer_idx]  # (batch,)
            
            # Compute cross-entropy loss (automatically handles ignore_index)
            layer_loss = self.ce_loss(pred, target)
            
            # Count valid samples (non-PAD)
            if masks_per_layer is not None:
                mask = masks_per_layer[layer_idx]
                valid_count = mask.sum()
            else:
                valid_count = (target != self.ignore_index).sum()
            
            # Only add loss if there are valid samples
            if valid_count > 0:
                total_loss += layer_loss
                num_valid_layers += 1
                
                # Compute accuracy for valid samples
                with torch.no_grad():
                    pred_labels = torch.argmax(pred, dim=1)
                    if masks_per_layer is not None:
                        mask = masks_per_layer[layer_idx]
                        correct = ((pred_labels == target) & mask.bool()).sum()
                        accuracy = correct.float() / valid_count
                    else:
                        valid_mask = target != self.ignore_index
                        correct = ((pred_labels == target) & valid_mask).sum()
                        accuracy = correct.float() / valid_count
                    
                loss_dict[f'class_layer{layer_idx+1}'] = layer_loss.item()
                loss_dict[f'acc_layer{layer_idx+1}'] = accuracy.item()
        
        # Average over valid layers
        if num_valid_layers > 0:
            total_loss = total_loss / num_valid_layers
        
        # Apply weight
        weighted_loss = self.delta_weight * total_loss
        loss_dict['class_total'] = weighted_loss.item()
        
        return weighted_loss, loss_dict


class HIDVAETotalLoss(nn.Module):
    """
    Combined loss function for HID-VAE training.
    Integrates reconstruction, anchoring, balance, classification, and commitment losses.
    """
    
    def __init__(
        self,
        # Reconstruction loss params
        lambda_content: float = 1.0,
        lambda_collab: float = 1.0,
        # Anchor loss params
        tag_embed_dim: int = 768,
        codebook_dim: int = 32,
        beta_weights: Optional[List[float]] = None,
        # Balance loss params
        gamma_weights: Optional[List[float]] = None,
        # Classification loss params
        delta_weight: float = 1.0,
        # Commitment loss param
        commitment_weight: float = 0.25,
        # General params
        n_layers: int = 3,
        ignore_index: int = 0
    ):
        """
        Initialize combined loss function.
        """
        super().__init__()
        
        # Initialize sub-losses
        self.recon_loss = MultiModalReconstructionLoss(
            lambda_content=lambda_content,
            lambda_collab=lambda_collab
        )
        
        self.anchor_loss = TagAnchoringLoss(
            tag_embed_dim=tag_embed_dim,
            codebook_dim=codebook_dim,
            n_layers=n_layers,
            beta_weights=beta_weights
        )
        
        self.balance_loss = CodebookBalanceLoss(
            n_layers=n_layers,
            gamma_weights=gamma_weights
        )
        
        self.class_loss = HierarchicalClassificationLoss(
            n_layers=n_layers,
            delta_weight=delta_weight,
            ignore_index=ignore_index
        )
        
        self.commitment_weight = commitment_weight
        
    def forward(
        self,
        # Reconstruction inputs
        pred_content: torch.Tensor,
        target_content: torch.Tensor,
        pred_collab: torch.Tensor,
        target_collab: torch.Tensor,
        # Anchor inputs
        tag_embeddings_per_layer: List[torch.Tensor],
        codebooks: List[torch.Tensor],
        # Balance inputs
        encoding_indices_per_layer: List[torch.Tensor],
        n_embed_per_layer: List[int],
        # Classification inputs
        predictions_per_layer: List[torch.Tensor],
        targets_per_layer: List[torch.Tensor],
        masks_per_layer: Optional[List[torch.Tensor]] = None,
        # Commitment loss
        commitment_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total HID-VAE loss.
        
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with all loss components
        """
        # Compute individual losses
        loss_recon, dict_recon = self.recon_loss(
            pred_content, target_content, pred_collab, target_collab
        )
        
        loss_anchor, dict_anchor = self.anchor_loss(
            tag_embeddings_per_layer, codebooks
        )
        
        loss_balance, dict_balance = self.balance_loss(
            encoding_indices_per_layer, n_embed_per_layer
        )
        
        loss_class, dict_class = self.class_loss(
            predictions_per_layer, targets_per_layer, masks_per_layer
        )
        
        # Combine all losses
        total_loss = loss_recon + loss_anchor + loss_balance + loss_class
        
        # Add commitment loss if provided
        if commitment_loss is not None:
            weighted_commit = self.commitment_weight * commitment_loss
            total_loss += weighted_commit
            dict_commit = {'commitment': commitment_loss.item()}
        else:
            dict_commit = {}
        
        # Merge all dictionaries
        loss_dict = {
            **dict_recon,
            **dict_anchor,
            **dict_balance,
            **dict_class,
            **dict_commit,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict

