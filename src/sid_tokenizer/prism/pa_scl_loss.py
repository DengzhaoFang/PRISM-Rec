"""
PA-SCL Stage 3: Popularity-Aware Soft Contrastive Loss

Replaces standard CMA (symmetric InfoNCE with hard [0,1] labels) with an
asymmetric KL-divergence loss that uses:

  1. Soft target matrix T(i,j) ∈ [0,1] from Stage 2 (topology-semantic
     prior), which distinguishes genuine negatives from false negatives
     (complements like "mouse pad" + "graphics card").

  2. Popularity-aware asymmetric weights W_ij = w_j * (1 - w_i), where
     w_i = sigmoid(log(pop_i + 1)).  Cold items (w_i ≈ 0) are pushed
     toward hot items (w_j ≈ 1); hot items are PROTECTED from being
     pulled toward noisy cold-item features.

Mathematical form (per batch of size B):

  sim(i,j) = h_t_i · h_c_j / τ               similarity matrix (B,B)
  P(i,j)   = softmax_j(sim(i,j))             row-normalised distribution
  Q(i,j)   = T(i,j) / Σ_k T(i,k)             target distribution from prior
  KL_i     = Σ_j Q(i,j) · log(Q(i,j)/P(i,j)) per-anchor KL divergence
  L        = (1/B) Σ_i Σ_j W_ij · Q(i,j) · log(Q(i,j)/P(i,j))

Gradient flows ONLY through h_t and h_c → IDE parameters (W_t, W_c).
T(i,j) and W_ij are fully detached structural priors.

Mutual exclusivity: PA-SCL and CMA must NOT be active simultaneously.
The caller (train_prism.py) should check this constraint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class PA_SCL_Loss(nn.Module):
    """
    Popularity-Aware Soft Contrastive Learning loss.

    Args:
        temperature:     τ for softmax sharpness (default 0.07, same as CMA).
        eps:             numerical stability for log/div (default 1e-8).
    """

    def __init__(self, temperature: float = 0.07, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    # ── Public API ──────────────────────────────────────────────────

    def forward(
        self,
        h_t: torch.Tensor,
        h_c: torch.Tensor,
        T: torch.Tensor,
        item_popularities: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            h_t:              IDE text projections   (B, d), L2-normalised.
            h_c:              IDE collab projections (B, d), L2-normalised.
            T:                Soft target matrix     (B, B), T[i,j] ∈ [0,1],
                              diagonal = 1.0.  Fully detached prior from
                              TopologySemanticPrior.compute_T().
            item_popularities: Raw interaction counts (B,), used to derive
                              asymmetric weights.  Not detached here but
                              treated as constants (no grad through w_i).

        Returns:
            loss:      Scalar PA-SCL loss.
            loss_dict: Per-component values for logging.
        """
        B = h_t.size(0)
        device = h_t.device

        # 1. Similarity matrix → log-softmax for KL
        sim = (h_t @ h_c.T) / self.temperature  # (B, B)
        log_P = F.log_softmax(sim, dim=-1)  # (B, B)

        # 2. Target distribution Q (row-normalised T)
        T_clamped = T.clamp(min=self.eps)
        Q = T_clamped / (T_clamped.sum(dim=-1, keepdim=True) + self.eps)  # (B, B)

        # 3. Per-pair KL: KL(Q||P) = Σ_j Q[j] * (log Q[j] - log P[j])
        #    F.kl_div(log_P, Q, reduction='none') = Q * (log Q - log P)
        kl_per_pair = F.kl_div(log_P, Q, reduction='none')  # (B, B)

        # 5. Asymmetric popularity weights: W_ij = w_j * (1 - w_i)
        w = self._compute_pop_weights(item_popularities, device)  # (B,)
        W = w.unsqueeze(0) * (1.0 - w.unsqueeze(1))  # (B, B)

        # 6. Weighted sum
        weighted_kl = W * kl_per_pair  # (B, B)
        loss = weighted_kl.sum() / B

        # ── Diagnostics ──
        with torch.no_grad():
            # Mean KL per anchor (unweighted)
            kl_per_anchor = kl_per_pair.sum(dim=-1)  # (B,)
            mean_kl = kl_per_anchor.mean().item()

            # Asymmetry ratio: avg W for cold→hot vs hot→cold
            cold_mask = w <= w.median()
            hot_mask = w > w.median()
            if cold_mask.any() and hot_mask.any():
                w_cold_to_hot = W[cold_mask][:, hot_mask].mean().item()
                w_hot_to_cold = W[hot_mask][:, cold_mask].mean().item()
            else:
                w_cold_to_hot = 0.0
                w_hot_to_cold = 0.0

            # Effective target entropy (how spread is Q?)
            q_entropy = -(Q * torch.log(Q + self.eps)).sum(dim=-1).mean().item()

            # Top-1 agreement: does argmax of P match argmax of Q?
            top1_match = (P.argmax(dim=-1) == Q.argmax(dim=-1)).float().mean().item()

        loss_dict = {
            'pa_scl': loss.item(),
            'mean_kl': mean_kl,
            'w_cold2hot': w_cold_to_hot,
            'w_hot2cold': w_hot_to_cold,
            'q_entropy': q_entropy,
            'top1_match': top1_match,
            'w_mean': w.mean().item(),
        }

        return loss, loss_dict

    # ── Internal ────────────────────────────────────────────────────

    @staticmethod
    def _compute_pop_weights(
        popularities: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        w_i = sigmoid(log(pop_i + 1) - shift)

        The shift (log of median popularity) centres the sigmoid so that
        the median item gets w ≈ 0.5, providing a balanced asymmetry.
        """
        pop = popularities.float().to(device)
        log_pop = torch.log(pop + 1.0)
        # Centre at median so ~half the batch gets w>0.5
        shift = log_pop.median()
        return torch.sigmoid(log_pop - shift)


# ═══════════════════════════════════════════════════════════════════
# Compatibility guard
# ═══════════════════════════════════════════════════════════════════

def validate_mutual_exclusivity(use_pa_scl: bool, use_cma: bool):
    """Raise ValueError if both PA-SCL and CMA are enabled."""
    if use_pa_scl and use_cma:
        raise ValueError(
            "PA-SCL and CMA are mutually exclusive. "
            "PA-SCL replaces CMA with an asymmetric, topology-aware "
            "soft contrastive loss.  Set use_cma=False when use_pa_scl=True."
        )
