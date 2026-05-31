"""Cross-Modal Alignment (CMA) module — lightweight projection heads + InfoNCE loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CMAHeads(nn.Module):
    """Projection heads that map IDE outputs (h_t, h_c) to a contrastive space."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.head_t = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )
        self.head_c = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim),
        )

    def forward(self, h_t: torch.Tensor, h_c: torch.Tensor):
        return self.head_t(h_t), self.head_c(h_c)


class CMALoss(nn.Module):
    """Symmetric InfoNCE between CMA projections of the two modalities."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, g_t: torch.Tensor, g_c: torch.Tensor) -> torch.Tensor:
        a = F.normalize(g_t, p=2, dim=-1)
        b = F.normalize(g_c, p=2, dim=-1)
        sim = torch.matmul(a, b.T) / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
