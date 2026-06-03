"""
PA-SCL Stage 2: Topology-Semantic Soft Target Matrix T(i,j)

GPU-accelerated: all tensors reside on GPU, zero Python for-loops in
the hot path.  Per-batch T computation is pure tensor indexing + matmul.
"""

import torch
import numpy as np
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict


class TopologySemanticPrior:
    """
    All tensors pre-loaded to GPU.  Per-batch compute_T() does pure GPU
    tensor ops — no numpy, no Python loops.

    Hyperparameters (exposed for ablation):
        text_percentile_lo : lower percentile for text min-max norm (default 1)
        text_percentile_hi : upper percentile for text min-max norm (default 99)
        text_sharpen_gamma : power-law exponent for text sharpening (default 3)
        graph_scale_beta   : cooc value treated as "definite co-occurrence"
                             (default 0.05 — items sharing 5% of neighbors)
    """

    def __init__(
        self,
        raw_text_emb: np.ndarray,
        item_ids: np.ndarray,
        device: torch.device,
        cooc_counts: Optional[Dict[Tuple[int, int], int]] = None,
        text_percentile_lo: float = 1.0,
        text_percentile_hi: float = 99.0,
        text_sharpen_gamma: float = 3.0,
        graph_scale_beta: float = 0.05,
    ):
        self.device = device
        self.text_p_lo = text_percentile_lo
        self.text_p_hi = text_percentile_hi
        self.text_gamma = text_sharpen_gamma
        self.graph_beta = graph_scale_beta

        # ── Text embeddings → GPU ──
        self.raw_text_emb = torch.tensor(
            raw_text_emb, dtype=torch.float32, device=self.device)
        self.text_norm = torch.nn.functional.normalize(
            self.raw_text_emb, p=2, dim=-1)

        # ── GPU index mapping: item_id → array index, zero Python overhead ──
        self.item_id_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}
        max_id = max(self.item_id_to_idx.keys())
        self.id_to_idx_tensor = torch.zeros(
            max_id + 1, dtype=torch.long, device=self.device)
        for iid, idx in self.item_id_to_idx.items():
            self.id_to_idx_tensor[iid] = idx

        # ── Graph cooc matrix → GPU (~585 MB float32) ──
        self._S_graph: Optional[torch.Tensor] = None
        if cooc_counts is not None:
            n_items = len(item_ids)
            self._cooc_max = max(cooc_counts.values()) if cooc_counts else 1
            idx_i, idx_j, vals = [], [], []
            for (a, b), cnt in cooc_counts.items():
                ia, ib = self.item_id_to_idx.get(a), self.item_id_to_idx.get(b)
                if ia is not None and ib is not None:
                    idx_i.extend([ia, ib])
                    idx_j.extend([ib, ia])
                    vals.extend([cnt, cnt])
            if vals:
                indices = torch.tensor([idx_i, idx_j], dtype=torch.long)
                values = torch.tensor(
                    vals, dtype=torch.float32) / self._cooc_max
                sparse = torch.sparse_coo_tensor(
                    indices, values, (n_items, n_items)).coalesce()
                self._S_graph = sparse.to_dense().to(self.device)

        # ── Global text percentiles (GPU, O(1) at runtime) ──
        sample_n = min(2000, len(self.text_norm))
        sample_idx = torch.randperm(
            len(self.text_norm), device=self.device)[:sample_n]
        S_sample = self.text_norm[sample_idx] @ self.text_norm[sample_idx].T
        off = S_sample[~torch.eye(sample_n, dtype=torch.bool, device=self.device)]
        self._text_lo = torch.quantile(off, self.text_p_lo / 100.0).item()
        self._text_hi = torch.quantile(off, self.text_p_hi / 100.0).item()
        self._text_denom = max(self._text_hi - self._text_lo, 1e-8)
        del S_sample, off, sample_idx

    # ── Public API ──────────────────────────────────────────────────

    @torch.no_grad()
    def compute_T(self, batch_item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch_item_ids: (B,) int64 GPU tensor of item IDs.
        Returns:
            T: (B, B) float32 GPU tensor, T[i,j] ∈ [0,1], diagonal = 1.0.
        """
        indices = self.id_to_idx_tensor[batch_item_ids]  # (B,) GPU tensor

        # Text similarity (pure GPU matmul)
        emb = self.text_norm[indices]                     # (B, D)
        S = emb @ emb.T                                    # (B, B)
        S_text = ((S - self._text_lo) / self._text_denom).clamp(0.0, 1.0)
        if self.text_gamma != 1.0:
            S_text = S_text ** self.text_gamma

        # Graph similarity (pure GPU index_select)
        S_graph = torch.zeros_like(S_text)
        if self._S_graph is not None:
            S_graph = self._S_graph[indices][:, indices]
            if self.graph_beta > 0:
                S_graph = (S_graph / self.graph_beta).clamp(0.0, 1.0)

        T = torch.maximum(S_text, S_graph)
        T.fill_diagonal_(1.0)
        return T


# ═══════════════════════════════════════════════════════════════════

def build_item_neighbor_graph(
    train_sequences: List[List[int]],
    min_cooc: int = 0,
) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, int], int]]:
    """Build item→neighbors dict AND cooc-count dict from sequences."""
    cooc = defaultdict(lambda: defaultdict(int))
    for seq in train_sequences:
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                a, b = seq[i], seq[j]
                if a != b:
                    cooc[a][b] += 1
                    cooc[b][a] += 1
    neighbors = {}
    cooc_dict = {}
    for item, nbrs in cooc.items():
        if min_cooc > 0:
            neighbors[item] = {n for n, cnt in nbrs.items() if cnt >= min_cooc}
        else:
            neighbors[item] = set(nbrs.keys())
        for nbr, cnt in nbrs.items():
            if cnt >= min_cooc:
                key = (min(item, nbr), max(item, nbr))
                cooc_dict[key] = cnt
    return neighbors, cooc_dict
