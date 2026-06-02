"""
PRISM Loss Functions

Implements:
1. UPR Loss (Unified Purified Reconstruction): MSE(z_dec, z_clean.detach())
2. CMA Loss (Cross-Modal Alignment): InfoNCE between h_t and h_c
3. Combined PRISM Total Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class UPRLoss(nn.Module):
    """Unified Purified Reconstruction loss.

    Reconstructs the clean fused feature z_clean = [h_c || h_t] (256D)
    from the quantized latent z_q. Target is detached to prevent the
    decoder gradient from corrupting the IDE module.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_dec: torch.Tensor, z_clean: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(z_dec, z_clean.detach())


class CMALoss(nn.Module):
    """Cross-Modal Alignment: bidirectional InfoNCE between h_t and h_c.

    For each item i, (h_t_i, h_c_i) is the positive pair;
    (h_t_i, h_c_j) for j≠i are negatives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, h_t: torch.Tensor, h_c: torch.Tensor) -> torch.Tensor:
        h_t_norm = F.normalize(h_t, p=2, dim=-1)
        h_c_norm = F.normalize(h_c, p=2, dim=-1)

        sim = torch.matmul(h_t_norm, h_c_norm.T) / self.temperature
        labels = torch.arange(h_t.size(0), device=h_t.device)

        loss_t2c = F.cross_entropy(sim, labels)
        loss_c2t = F.cross_entropy(sim.T, labels)
        return (loss_t2c + loss_c2t) / 2


class PRISMTotalLoss(nn.Module):
    """Combined loss for PRISM training.

    L = L_UPR + commit_weight * L_commit + λ_cma * L_CMA
    """

    def __init__(
        self,
        commit_weight: float = 0.0625,
        use_cma: bool = True,
        lambda_cma: float = 0.1,
        cma_temperature: float = 0.07,
    ):
        super().__init__()

        self.upr_loss = UPRLoss()
        self.commit_weight = commit_weight

        self.use_cma = use_cma
        if use_cma:
            self.cma_loss = CMALoss(temperature=cma_temperature)
        else:
            self.cma_loss = None
        self.lambda_cma = lambda_cma

    def forward(
        self,
        z_dec: torch.Tensor,
        z_clean: torch.Tensor,
        commitment_loss: Optional[torch.Tensor] = None,
        h_t: Optional[torch.Tensor] = None,
        h_c: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_upr = self.upr_loss(z_dec, z_clean)

        extra_losses = torch.tensor(0.0, device=z_dec.device)

        dict_commit = {}
        if commitment_loss is not None:
            extra_losses = extra_losses + self.commit_weight * commitment_loss
            dict_commit = {'commitment': commitment_loss.item()}

        dict_cma = {}
        if self.cma_loss is not None and h_t is not None and h_c is not None:
            loss_cma = self.cma_loss(h_t, h_c)
            extra_losses = extra_losses + self.lambda_cma * loss_cma
            dict_cma = {'cma': loss_cma.item()}

        total_loss = loss_upr + extra_losses

        loss_dict = {
            'upr': loss_upr.item(),
            **dict_commit,
            **dict_cma,
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict
