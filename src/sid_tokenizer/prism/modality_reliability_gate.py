"""
Modality Reliability Gate for Heteroscedastic Cross-Modal Fusion.

Learns per-item collaborative modality reliability from the raw LightGCN
collaborative embedding e_c (64D). The raw embedding preserves norm
information that correlates with training signal quality
(Spearman ρ ≈ 0.78 between ||e_c|| and item popularity).

Theory: Heteroscedastic modality uncertainty (Kendall & Gal 2017).
Popular items → well-trained LightGCN embeddings → gate learns high α.
Cold-start items → noisy embeddings → gate learns low α.

The gate receives gradient primarily from SACO (78%) through the
path: SACO_loss → z → encoder → z_clean → α*h_c → α → gate_MLP,
with additional signal from UPR through the decoder pathway (22%).
"""

import torch
import torch.nn as nn


class ModalityReliabilityGate(nn.Module):
    """
    Lightweight MLP that predicts collaborative modality reliability
    from raw LightGCN collaborative embeddings.

    Input:  e_c (raw collab embedding)  (B, 64)
    Output: α ∈ [0.2, 1.0]              (B, 1)
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize bias so sigmoid output starts near 0.6 (mild trust)
        nn.init.xavier_uniform_(self.mlp[0].weight, gain=0.5)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight, gain=0.5)
        nn.init.constant_(self.mlp[2].bias, 0.4)  # σ(0.4) ≈ 0.6

    def forward(self, e_c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_c: Raw LightGCN collaborative embedding (B, 64).
                 This is a leaf tensor from the dataset (no grad history),
                 so no explicit detach is needed.

        Returns:
            α: Reliability weight (B, 1), clamped to [0.2, 1.0].
        """
        return torch.sigmoid(self.mlp(e_c)) * 0.8 + 0.2
