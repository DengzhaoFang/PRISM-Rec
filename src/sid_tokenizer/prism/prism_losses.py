"""
PRISM Loss Functions

Implements:
1. UPR Loss (Unified Purified Reconstruction): MSE(z_dec, z_clean.detach())
2. CMA Loss (Cross-Modal Alignment): InfoNCE between h_t and h_c
3. SACO Loss (Sequence-Aware Contrastive Objective) on paired latent z
4. Combined PRISM Total Loss
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


class CMALoss(nn.Module):
    """
    Cross-Modal Alignment: bidirectional InfoNCE between h_t and h_c.

    For each item i, (h_t_i, h_c_i) is the positive pair;
    (h_t_i, h_c_j) for j≠i are negatives.

    Fixes the cos(h_t, h_c) ≈ 0 problem caused by IDE projecting
    modalities into orthogonal subspaces without alignment supervision.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_t: torch.Tensor, h_c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_t: (B, d) — raw IDE text projections (pre-MCD)
            h_c: (B, d) — raw IDE collab projections (pre-MCD)
        """
        h_t_norm = F.normalize(h_t, p=2, dim=-1)
        h_c_norm = F.normalize(h_c, p=2, dim=-1)

        sim = torch.matmul(h_t_norm, h_c_norm.T) / self.temperature  # (B, B)
        labels = torch.arange(h_t.size(0), device=h_t.device)

        loss_t2c = F.cross_entropy(sim, labels)
        loss_c2t = F.cross_entropy(sim.T, labels)
        return (loss_t2c + loss_c2t) / 2


class SACOLoss(nn.Module):
    """
    Sequence-Aware Contrastive Objective on paired latent representations.

    For each anchor i, positive is z_pos[i] (a co-occurring item);
    negatives are all z_pos[j] for j≠i.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_anchor: torch.Tensor,
        z_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_anchor: (B, d) anchor latent representations
            z_pos: (B, d) positive (co-occurring) latent representations
        """
        z_a_norm = F.normalize(z_anchor, p=2, dim=-1)
        z_p_norm = F.normalize(z_pos, p=2, dim=-1)

        sim = torch.matmul(z_a_norm, z_p_norm.T) / self.temperature  # (B, B)
        labels = torch.arange(z_anchor.size(0), device=z_anchor.device)

        return F.cross_entropy(sim, labels)


class PRISMTotalLoss(nn.Module):
    """
    Combined loss for PRISM training.

    L = L_UPR + β * L_commit + λ_cma * L_CMA + λ_sac * L_SACO
    """

    def __init__(
        self,
        commitment_weight: float = 0.25,
        use_saco: bool = False,
        lambda_sac: float = 0.1,
        saco_temperature: float = 0.07,
        use_cma: bool = True,
        lambda_cma: float = 0.1,
        cma_temperature: float = 0.07,
    ):
        super().__init__()

        self.upr_loss = UPRLoss()
        self.commitment_weight = commitment_weight

        self.use_cma = use_cma
        if use_cma:
            self.cma_loss = CMALoss(temperature=cma_temperature)
        else:
            self.cma_loss = None
        self.lambda_cma = lambda_cma

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
        h_t: Optional[torch.Tensor] = None,
        h_c: Optional[torch.Tensor] = None,
        z_anchor: Optional[torch.Tensor] = None,
        z_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_upr = self.upr_loss(z_dec, z_clean)

        extra_losses = torch.tensor(0.0, device=z_dec.device)

        dict_commit = {}
        if commitment_loss is not None:
            extra_losses = extra_losses + self.commitment_weight * commitment_loss
            dict_commit = {'commitment': commitment_loss.item()}

        dict_cma = {}
        if self.cma_loss is not None and h_t is not None and h_c is not None:
            loss_cma = self.cma_loss(h_t, h_c)
            extra_losses = extra_losses + self.lambda_cma * loss_cma
            dict_cma = {'cma': loss_cma.item()}

        dict_saco = {}
        if self.saco_loss is not None and z_anchor is not None and z_pos is not None:
            loss_saco = self.saco_loss(z_anchor, z_pos)
            extra_losses = extra_losses + self.lambda_sac * loss_saco
            dict_saco = {'saco': loss_saco.item()}

        total_loss = loss_upr + extra_losses

        loss_dict = {
            'upr': loss_upr.item(),
            **dict_commit,
            **dict_cma,
            **dict_saco,
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict
