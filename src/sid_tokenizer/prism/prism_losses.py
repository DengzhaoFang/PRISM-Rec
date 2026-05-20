"""
PRISM Loss Functions

Implements loss functions for Hierarchical ID VAE:
1. Cosine Similarity Reconstruction Loss (multi-modal)
2. Multi-Modal Reconstruction Loss
3. Gate Supervision Loss
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
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)
        loss = 1.0 - cosine_sim
        return loss.mean()


class GateSupervisionLoss(nn.Module):
    """
    Gate supervision loss to align gate values with item popularity.

    Encourages:
    - Long-tail items (low popularity) -> low gate (don't trust noisy collab signal)
    - Popular items (high popularity) -> high gate (trust reliable collab signal)
    """

    def __init__(
        self,
        weight: float = 0.1,
        diversity_weight: float = 0.5,
        target_std: float = 0.2
    ):
        super().__init__()
        self.weight = weight
        self.diversity_weight = diversity_weight
        self.target_var = target_std ** 2

    def forward(
        self,
        gate_values: torch.Tensor,      # (batch_size, 768)
        popularity_scores: torch.Tensor  # (batch_size,)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        gate_mean = gate_values.mean(dim=1)  # (batch_size,)
        supervision_loss = F.mse_loss(gate_mean, popularity_scores)
        gate_var = gate_mean.var()
        diversity_loss = F.relu(self.target_var - gate_var)
        total_loss = supervision_loss + self.diversity_weight * diversity_loss

        loss_dict = {
            'gate_supervision': supervision_loss.item(),
            'gate_diversity': diversity_loss.item(),
            'gate_variance': gate_var.item(),
            'gate_mean': gate_mean.mean().item(),
            'gate_std': gate_mean.std().item()
        }

        return self.weight * total_loss, loss_dict


class MultiModalReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss for content and collaborative embeddings.
    Uses cosine similarity to avoid scale conflicts between different modalities.

    Supports two modes:
    1. Dual decoder mode (DHR): Separate losses for content and collab
    2. Single decoder mode: Loss on concatenated embedding
    """

    def __init__(
        self,
        lambda_content: float = 1.0,
        lambda_collab: float = 1.0,
        use_dual_decoder: bool = True
    ):
        super().__init__()
        self.lambda_content = lambda_content
        self.lambda_collab = lambda_collab
        self.use_dual_decoder = use_dual_decoder
        self.cosine_loss = CosineSimilarityLoss()

    def forward(
        self,
        pred_content: torch.Tensor,
        target_content: torch.Tensor,
        pred_collab: torch.Tensor,
        target_collab: torch.Tensor,
        weighted_collab_target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self.use_dual_decoder:
            loss_content = self.cosine_loss(pred_content, target_content)

            if weighted_collab_target is not None:
                collab_target = weighted_collab_target
            else:
                collab_target = target_collab

            loss_collab = self.cosine_loss(pred_collab, collab_target)

            loss_total = (
                self.lambda_content * loss_content +
                self.lambda_collab * loss_collab
            )

            loss_dict = {
                'recon_content': loss_content.item(),
                'recon_collab': loss_collab.item(),
                'recon_total': loss_total.item()
            }
        else:
            pred_concat = torch.cat([pred_content, pred_collab], dim=1)
            target_concat = torch.cat([target_content, target_collab], dim=1)
            loss_concat = self.cosine_loss(pred_concat, target_concat)
            loss_total = loss_concat

            with torch.no_grad():
                loss_content = self.cosine_loss(pred_content, target_content)
                loss_collab = self.cosine_loss(pred_collab, target_collab)

            loss_dict = {
                'recon_content': loss_content.item(),
                'recon_collab': loss_collab.item(),
                'recon_concat': loss_concat.item(),
                'recon_total': loss_total.item()
            }

        return loss_total, loss_dict


class PRISMTotalLoss(nn.Module):
    """
    Combined loss function for PRISM training.
    Integrates reconstruction and commitment losses.
    """

    def __init__(
        self,
        # Reconstruction loss params
        lambda_content: float = 1.0,
        lambda_collab: float = 1.0,
        use_dual_decoder: bool = True,
        # Commitment loss param
        commitment_weight: float = 0.25,
        # Gate supervision params
        use_gate_supervision: bool = False,
        gate_supervision_weight: float = 0.1,
        gate_diversity_weight: float = 0.5,
        gate_target_std: float = 0.2,
    ):
        super().__init__()

        self.recon_loss = MultiModalReconstructionLoss(
            lambda_content=lambda_content,
            lambda_collab=lambda_collab,
            use_dual_decoder=use_dual_decoder
        )

        self.commitment_weight = commitment_weight

        self.use_gate_supervision = use_gate_supervision
        if use_gate_supervision:
            self.gate_supervision_loss = GateSupervisionLoss(
                weight=gate_supervision_weight,
                diversity_weight=gate_diversity_weight,
                target_std=gate_target_std
            )
        else:
            self.gate_supervision_loss = None

    def forward(
        self,
        pred_content: torch.Tensor,
        target_content: torch.Tensor,
        pred_collab: torch.Tensor,
        target_collab: torch.Tensor,
        commitment_loss: Optional[torch.Tensor] = None,
        gate_values: Optional[torch.Tensor] = None,
        popularity_scores: Optional[torch.Tensor] = None,
        weighted_collab_target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total PRISM loss."""
        loss_recon, dict_recon = self.recon_loss(
            pred_content, target_content, pred_collab, target_collab,
            weighted_collab_target=weighted_collab_target
        )

        total_loss = loss_recon

        if commitment_loss is not None:
            weighted_commit = self.commitment_weight * commitment_loss
            total_loss += weighted_commit
            dict_commit = {'commitment': commitment_loss.item()}
        else:
            dict_commit = {}

        if self.gate_supervision_loss is not None and gate_values is not None and popularity_scores is not None:
            loss_gate_sup, dict_gate_sup = self.gate_supervision_loss(
                gate_values, popularity_scores
            )
            total_loss += loss_gate_sup
        else:
            dict_gate_sup = {}

        loss_dict = {
            **dict_recon,
            **dict_commit,
            **dict_gate_sup,
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict
