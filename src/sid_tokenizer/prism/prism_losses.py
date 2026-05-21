"""
PRISM Loss Functions

Implements:
1. UPR Loss (Unified Purified Reconstruction): MSE(z_dec, z_clean.detach())
2. SACO Loss (Sequence-Aware Contrastive Objective)
3. Combined PRISM Total Loss

Removed: DHR (cosine-based dual-head reconstruction), Gate Supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class UPRLoss(nn.Module):
    """
    Unified Purified Reconstruction loss.

    Reconstructs the clean fused feature z_clean = [h_c_hat || h_t_hat] (256D)
    from the quantized latent z_q. Target is detached to prevent the decoder
    gradient from corrupting the MCD denoising module.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_dec: torch.Tensor, z_clean: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(z_dec, z_clean.detach())


class SACOLoss(nn.Module):
    """
    Sequence-Aware Contrastive Objective.

    L_SAC = -Σ log( exp(cos(z_a, z_b)/τ) / Σ_N exp(cos(z_a, z_n)/τ) )

    Pulls together latent representations of co-occurring items,
    pushes apart non-co-occurring items.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z: torch.Tensor,
        pos_indices_a: torch.Tensor,
        pos_indices_b: torch.Tensor,
    ) -> torch.Tensor:
        if pos_indices_a.numel() == 0 or pos_indices_b.numel() == 0:
            return torch.tensor(0.0, device=z.device)

        mask = (pos_indices_a >= 0) & (pos_indices_b >= 0)
        pos_indices_a = pos_indices_a[mask]
        pos_indices_b = pos_indices_b[mask]

        if pos_indices_a.numel() == 0:
            return torch.tensor(0.0, device=z.device)

        z_norm = F.normalize(z, p=2, dim=-1)
        z_a = z_norm[pos_indices_a]
        z_b = z_norm[pos_indices_b]

        sim_matrix = torch.matmul(z_a, z_norm.T) / self.temperature
        pos_logits = sim_matrix[torch.arange(len(pos_indices_a), device=z.device), pos_indices_b]

        loss = -pos_logits + torch.logsumexp(sim_matrix, dim=-1)
        return loss.mean()


class PRISMTotalLoss(nn.Module):
    """
    Combined loss for PRISM training.

    L_stage1 = L_UPR + β * L_commit + λ_sac * L_SACO
    """

    def __init__(
        self,
        commitment_weight: float = 0.25,
        use_saco: bool = False,
        lambda_sac: float = 0.1,
        saco_temperature: float = 0.07,
    ):
        super().__init__()

        self.upr_loss = UPRLoss()
        self.commitment_weight = commitment_weight

        self.use_saco = use_saco
        if use_saco:
            self.saco_loss = SACOLoss(temperature=saco_temperature)
        else:
            self.saco_loss = None

        self.lambda_sac = lambda_sac

    def forward(
        self,
        z_dec: torch.Tensor,
        z_clean: torch.Tensor,
        commitment_loss: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        pos_indices_a: Optional[torch.Tensor] = None,
        pos_indices_b: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_dec: Decoded reconstruction (B, output_dim)
            z_clean: Clean fused target [h_c_hat || h_t_hat] (B, output_dim)
            commitment_loss: RQ-VAE codebook + commitment loss (scalar)
            z: Latent representations (B, latent_dim) for SACO
            pos_indices_a/b: Positive pair indices for SACO
        """
        loss_upr = self.upr_loss(z_dec, z_clean)
        total_loss = loss_upr

        dict_commit = {}
        if commitment_loss is not None:
            weighted_commit = self.commitment_weight * commitment_loss
            total_loss += weighted_commit
            dict_commit = {'commitment': commitment_loss.item()}

        dict_saco = {}
        if self.saco_loss is not None and z is not None and pos_indices_a is not None:
            loss_saco = self.saco_loss(z, pos_indices_a, pos_indices_b)
            weighted_saco = self.lambda_sac * loss_saco
            total_loss += weighted_saco
            dict_saco = {'saco': loss_saco.item()}

        loss_dict = {
            'upr': loss_upr.item(),
            **dict_commit,
            **dict_saco,
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict
