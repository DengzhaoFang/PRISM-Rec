"""
Information Density Equalization (IDE) and Mutual Cross-modal Denoising (MCD)

IDE: Projects heterogeneous modalities to a common dimension d=128 with LayerNorm,
     preventing the 768D text gradient from overwhelming the 64D collaborative signal.

MCD: Asymmetric cross-modal denoising that replaces popularity-based confidence:
     - Uses cross-modal consistency score to gate noisy collaborative features
     - Uses collaborative signal to suppress recommendation-irrelevant text dimensions
     - Safety fallback: when collab is highly untrusted, falls back to stable text semantics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class IDEEqualizer(nn.Module):
    """
    Information Density Equalization.

    Projects both modalities to a shared dimension d with LayerNorm,
    ensuring equal gradient flow and numerical scale alignment.
    """

    def __init__(self, content_dim: int = 768, collab_dim: int = 64, d: int = 128):
        super().__init__()
        self.content_dim = content_dim
        self.collab_dim = collab_dim
        self.d = d

        self.W_t = nn.Linear(content_dim, d, bias=False)
        self.W_c = nn.Linear(collab_dim, d, bias=False)
        self.ln_t = nn.LayerNorm(d)
        self.ln_c = nn.LayerNorm(d)

    def forward(self, e_t: torch.Tensor, e_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            e_t: Text embeddings (B, content_dim)
            e_c: Collaborative embeddings (B, collab_dim)

        Returns:
            h_t: Equalized text features (B, d)
            h_c: Equalized collaborative features (B, d)
        """
        h_t = self.ln_t(self.W_t(e_t))
        h_c = self.ln_c(self.W_c(e_c))
        return h_t, h_c


class MCDModule(nn.Module):
    """
    Mutual Cross-modal Denoising.

    Operation A (collab denoising): Uses cross-modal consistency to gate collaborative
    features. When collab disagrees with text (low consistency), the safety fallback
    blends toward the stable text representation.

    Operation B (text denoising): Uses collaborative signal as context to generate a
    gate that suppresses text dimensions irrelevant to recommendation.
    """

    def __init__(self, d: int = 128, enabled: bool = True):
        super().__init__()
        self.d = d
        self.enabled = enabled

        # Operation A: collab reliability gate
        # Input: [h_c || s] where s is the consistency scalar
        self.W_gc = nn.Linear(d + 1, d)
        self.b_gc = nn.Parameter(torch.zeros(d))

        # Operation B: text relevance gate
        # Input: h_c (collab acts as context to filter text)
        self.W_gt = nn.Linear(d, d, bias=False)
        self.b_gt = nn.Parameter(torch.zeros(d))

    def _consistency_score(self, h_t: torch.Tensor, h_c: torch.Tensor) -> torch.Tensor:
        """Compute per-item cross-modal consistency: s = 0.5 * (cos(h_t, h_c) + 1)"""
        cos_sim = F.cosine_similarity(h_t, h_c, dim=-1)
        return 0.5 * (cos_sim + 1.0)

    def forward(
        self, h_t: torch.Tensor, h_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: Equalized text features (B, d)
            h_c: Equalized collaborative features (B, d)

        Returns:
            h_t_hat: Denoised text features (B, d)
            h_c_hat: Denoised collaborative features (B, d)
            s: Cross-modal consistency scores (B,)
        """
        if not self.enabled:
            return h_t, h_c, torch.zeros(h_t.size(0), device=h_t.device)

        s = self._consistency_score(h_t, h_c)  # (B,)

        # Operation A: pure collaborative gating (no text leakage)
        # Removed safety fallback (1-g_c)*h_t — collaborative signal is the
        # primary driver for recommendation and must not be diluted by text.
        # The gate suppresses noisy collab dimensions in-place without substitution.
        s_expanded = s.unsqueeze(-1)  # (B, 1)
        g_c = torch.sigmoid(self.W_gc(torch.cat([h_c, s_expanded], dim=-1)) + self.b_gc)
        h_c_hat = g_c * h_c

        # Operation B: text relevance suppression
        # Uses denoised h_c_hat as context — noisy collab would produce
        # unreliable gates, defeating the purpose of cross-modal guidance.
        g_t = torch.sigmoid(self.W_gt(h_c_hat) + self.b_gt)
        h_t_hat = g_t * h_t

        return h_t_hat, h_c_hat, s
