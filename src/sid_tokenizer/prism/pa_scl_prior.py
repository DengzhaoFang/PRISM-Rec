"""
PA-SCL Stage 2: Topology-Semantic Soft Target Matrix T(i,j)

Computes a gradient-free prior combining calibrated text semantics and
amplified graph topology.  Optimised with numpy neighbour arrays and
LRU caching — graph Jaccard is O(B²) set intersections on first epoch
but amortised O(1) lookups thereafter.

Pipeline (per batch):
  1. S_text_raw  = cosine_similarity(e_t_i, e_t_j)         [-1, 1]
  2. S_text_cal  = percentile_norm(S_text) ^ γ               [0, 1]
  3. S_graph     = Jaccard(neighbors(i), neighbors(j))       [0, 1]
  4. S_graph_amp = min(1.0, S_graph / β)                    [0, 1]
  5. T = max(S_text_cal, S_graph_amp), diag=1.0             [0, 1]

All computations are detached — T is a fixed structural prior.
"""

import torch
import numpy as np
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
from functools import lru_cache


class TopologySemanticPrior:
    """
    Precomputes per-item data and constructs batch-level soft target
    matrices with calibrated text and graph similarity.

    Hyperparameters (exposed for ablation):
        text_percentile_lo : lower percentile for text min-max norm (default 1)
        text_percentile_hi : upper percentile for text min-max norm (default 99)
        text_sharpen_gamma : power-law exponent for text sharpening (default 3)
        graph_scale_beta   : Jaccard value treated as "definite co-occurrence"
                             (default 0.05 — items sharing 5% of neighbors)
    """

    def __init__(
        self,
        raw_text_emb: np.ndarray,
        item_ids: np.ndarray,
        user_item_graph: Optional[Dict[int, Set[int]]] = None,
        cooc_counts: Optional[Dict[Tuple[int, int], int]] = None,
        text_percentile_lo: float = 1.0,
        text_percentile_hi: float = 99.0,
        text_sharpen_gamma: float = 3.0,
        graph_scale_beta: float = 0.05,
    ):
        self.raw_text_emb = torch.tensor(raw_text_emb, dtype=torch.float32)
        self.item_ids = item_ids
        self.item_id_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}

        self.text_p_lo = text_percentile_lo
        self.text_p_hi = text_percentile_hi
        self.text_gamma = text_sharpen_gamma
        self.graph_beta = graph_scale_beta

        # ── Fast vectorised graph lookup via sparse tensor ──
        # Build a sparse (N_items × N_items) tensor of normalised cooc counts.
        # Per-batch: index_select gives the B×B submatrix — fully vectorised.
        self._cooc_max = 1
        self._S_graph_sparse: Optional[torch.Tensor] = None
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
                values = torch.tensor(vals, dtype=torch.float32) / self._cooc_max
                self._S_graph_sparse = torch.sparse_coo_tensor(
                    indices, values, (n_items, n_items)).coalesce()

        # Cache L2-normalised text embeddings for fast cosine
        self.text_norm = torch.nn.functional.normalize(
            self.raw_text_emb, p=2, dim=-1)

    # ── Public API ──────────────────────────────────────────────────

    @torch.no_grad()
    def compute_T(self, batch_item_ids: np.ndarray) -> torch.Tensor:
        indices = self._ids_to_indices(batch_item_ids)
        S_text = self._compute_text_calibrated(indices)
        S_graph = self._compute_graph_amplified(batch_item_ids)
        T = torch.maximum(S_text, S_graph)
        T.fill_diagonal_(1.0)
        return T

    @torch.no_grad()
    def compute_S_text_raw(self, batch_item_ids: np.ndarray) -> torch.Tensor:
        return self._compute_S_text_raw(self._ids_to_indices(batch_item_ids))

    @torch.no_grad()
    def compute_S_text_calibrated(self, batch_item_ids: np.ndarray) -> torch.Tensor:
        return self._compute_text_calibrated(self._ids_to_indices(batch_item_ids))

    @torch.no_grad()
    def compute_S_graph_raw(self, batch_item_ids: np.ndarray) -> torch.Tensor:
        return self._compute_S_graph_raw(batch_item_ids)

    @torch.no_grad()
    def compute_S_graph_amplified(self, batch_item_ids: np.ndarray) -> torch.Tensor:
        return self._compute_graph_amplified(batch_item_ids)

    # ── Internal: text (unchanged, already vectorised) ──────────────

    def _ids_to_indices(self, item_ids: np.ndarray) -> List[int]:
        return [self.item_id_to_idx[int(iid)] for iid in item_ids]

    @torch.no_grad()
    def _compute_S_text_raw(self, indices: List[int]) -> torch.Tensor:
        emb = self.text_norm[indices]
        return emb @ emb.T

    @torch.no_grad()
    def _compute_text_calibrated(self, indices: List[int]) -> torch.Tensor:
        S = self._compute_S_text_raw(indices)
        B = S.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=S.device)
        off_diag = S[mask]
        lo = torch.quantile(off_diag, self.text_p_lo / 100.0)
        hi = torch.quantile(off_diag, self.text_p_hi / 100.0)
        denom = hi - lo
        if denom < 1e-8:
            denom = 1.0
        S_norm = (S - lo) / denom
        S_norm = S_norm.clamp(0.0, 1.0)
        if self.text_gamma != 1.0:
            S_norm = S_norm ** self.text_gamma
        return S_norm

    # ── Internal: graph (vectorised sparse index_select) ────────────

    @torch.no_grad()
    def _compute_S_graph_raw(self, batch_item_ids: np.ndarray) -> torch.Tensor:
        """
        Extract B×B submatrix from precomputed sparse cooc tensor.
        Fully vectorised — zero Python for-loops.
        """
        if self._S_graph_sparse is None:
            return torch.zeros(len(batch_item_ids), len(batch_item_ids))
        indices = torch.tensor(
            [self.item_id_to_idx[int(iid)] for iid in batch_item_ids],
            dtype=torch.long)
        # index_select both dimensions of the sparse matrix
        sub = self._S_graph_sparse.index_select(0, indices).index_select(1, indices)
        return sub.to_dense()

    @torch.no_grad()
    def _compute_graph_amplified(self, batch_item_ids: np.ndarray) -> torch.Tensor:
        S = self._compute_S_graph_raw(batch_item_ids)
        if self.graph_beta > 0:
            S = (S / self.graph_beta).clamp(0.0, 1.0)
        return S


# ═══════════════════════════════════════════════════════════════════

def build_item_neighbor_graph(
    train_sequences: List[List[int]],
    min_cooc: int = 0,
) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, int], int]]:
    """
    Build item→neighbors dict AND cooc-count dict from sequences.

    Returns:
        neighbors:  Dict[item_id, Set[neighbor_ids]]
        cooc_dict:  Dict[(min_id, max_id), raw_cooc_count]
    """
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
