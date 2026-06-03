"""
PA-SCL Stage 4: Dual-Head Weighted Reconstruction

Replaces the single UnifiedDecoder (which reconstructs z_clean = [h_c||h_t])
with two independent decoder heads:

  Dec_t(z_q) → ĥ_t  (reconstructs text IDE projection)
  Dec_c(z_q) → ĥ_c  (reconstructs collab IDE projection)

Confidence-weighted MSE loss:

  L_UPR = w_t * MSE(ĥ_t, sg(h_t)) + w_c * MSE(ĥ_c, sg(h_c))

where w_t, w_c are modality reliability weights (constants, no grad):
  - w_t = 1.0 (text always reliable)
  - w_c = sigmoid(log(pop+1)-shift) (collab reliability ∝ popularity)

For cold-start items (w_c ≈ 0): the decoder is NOT penalised for failing to
reconstruct noisy h_c — it can focus codebook capacity on the clean text
signal.  For popular items (w_c ≈ 1): both modalities contribute equally.

Ablation: --use_dual_head (default False).  When disabled, falls back to
the original UnifiedDecoder reconstructing z_clean = [h_c||h_t].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict


# ═══════════════════════════════════════════════════════════════════
# Dual-Head Decoder
# ═══════════════════════════════════════════════════════════════════

class DualHeadDecoder(nn.Module):
    """
    Two-headed decoder: shared trunk → two independent output heads.

    Args:
        latent_dim:   z_q dimension (default 32).
        output_dim:   per-modality output dimension (default 128 = ide_dim).
        hidden_dims:  hidden layer sizes for the shared trunk.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        output_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]

        # Shared trunk (same architecture as UnifiedDecoder trunk)
        shared_layers = []
        prev_dim = latent_dim
        for hd in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hd),
                nn.LayerNorm(hd),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hd
        self.shared = nn.Sequential(*shared_layers)

        # Per-modality output heads (same architecture as UnifiedDecoder head)
        self.head_t = self._make_head(prev_dim, output_dim)
        self.head_c = self._make_head(prev_dim, output_dim)

    @staticmethod
    def _make_head(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, z_q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            h_t_hat: reconstructed text IDE projection  (B, output_dim)
            h_c_hat: reconstructed collab IDE projection (B, output_dim)
        """
        shared = self.shared(z_q)
        return {
            'h_t_hat': self.head_t(shared),
            'h_c_hat': self.head_c(shared),
        }


# ═══════════════════════════════════════════════════════════════════
# Confidence-Weighted Dual-Head UPR Loss
# ═══════════════════════════════════════════════════════════════════

class DualHeadUPRLoss(nn.Module):
    """
    Confidence-weighted MSE for dual-head decoder outputs.

    Args:
        use_pop_weight: If True, w_c is derived from item popularity.
                        If False, w_c = 1.0 (uniform weighting).
    """

    def __init__(self, use_pop_weight: bool = True):
        super().__init__()
        self.use_pop_weight = use_pop_weight

    def forward(
        self,
        h_t_hat: torch.Tensor,
        h_c_hat: torch.Tensor,
        h_t: torch.Tensor,
        h_c: torch.Tensor,
        item_popularities: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            h_t_hat, h_c_hat: decoder outputs  (B, d)
            h_t, h_c:         IDE targets       (B, d), detached
            item_popularities: interaction counts (B,), optional

        Returns:
            total_loss: MSE_t + w_c * MSE_c
            loss_dict:  per-component values
        """
        loss_t = F.mse_loss(h_t_hat, h_t.detach())

        if self.use_pop_weight and item_popularities is not None:
            w_c = self._compute_collab_weights(item_popularities, h_t.device)
            # Per-sample weighted MSE for collab
            per_sample_c = ((h_c_hat - h_c.detach()) ** 2).mean(dim=-1)  # (B,)
            loss_c = (w_c * per_sample_c).mean()
        else:
            w_c = torch.ones(1, device=h_t.device)
            loss_c = F.mse_loss(h_c_hat, h_c.detach())

        total = loss_t + loss_c

        loss_dict = {
            'upr_t': loss_t.item(),
            'upr_c': loss_c.item(),
            'upr': total.item(),
            'w_c_mean': w_c.mean().item(),
        }
        return total, loss_dict

    @staticmethod
    def _compute_collab_weights(
        popularities: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        w_c = sigmoid(log(pop + 1) - shift), centred at median.

        Cold-start items (low pop) → w_c ≈ 0 → minimal collab
        reconstruction penalty.  Hot items → w_c ≈ 1 → full penalty.
        """
        pop = popularities.float().to(device)
        log_pop = torch.log(pop + 1.0)
        shift = log_pop.median()
        return torch.sigmoid(log_pop - shift)
