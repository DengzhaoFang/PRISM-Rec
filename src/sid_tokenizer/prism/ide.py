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
from typing import Tuple


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
