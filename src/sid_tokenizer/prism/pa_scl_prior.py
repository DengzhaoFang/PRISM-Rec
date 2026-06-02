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

        # ── Convert Python sets → sorted numpy int32 arrays (much faster) ──
        self.neighbor_arr: Dict[int, np.ndarray] = {}
        if user_item_graph is not None:
            for iid in item_ids:
                nbrs = user_item_graph.get(int(iid), set())
                self.neighbor_arr[int(iid)] = np.fromiter(
                    nbrs, dtype=np.int32, count=len(nbrs))
                self.neighbor_arr[int(iid)].sort()
        self._jaccard_cache: Dict[Tuple[int, int], float] = {}

        # Cache L2-normalised text embeddings for fast cosine
        self.text_norm = torch.nn.functional.normalize(
            self.raw_text_emb, p=2, dim=-1)

        self._cache_hits = 0
        self._cache_misses = 0

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

    @property
    def cache_stats(self) -> Dict[str, int]:
        return {'hits': self._cache_hits, 'misses': self._cache_misses}

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

    # ── Internal: graph (optimised with numpy + LRU cache) ──────────

    def _get_jaccard(self, id_i: int, id_j: int) -> float:
        """Cached Jaccard lookup using numpy intersect1d."""
        key = (min(id_i, id_j), max(id_i, id_j))
        val = self._jaccard_cache.get(key)
        if val is not None:
            self._cache_hits += 1
            return val

        self._cache_misses += 1
        arr_i = self.neighbor_arr.get(id_i)
        arr_j = self.neighbor_arr.get(id_j)
        if arr_i is None or arr_j is None or len(arr_i) == 0 or len(arr_j) == 0:
            val = 0.0
        else:
            inter = np.intersect1d(arr_i, arr_j, assume_unique=True).size
            union = arr_i.size + arr_j.size - inter
            val = inter / union if union > 0 else 0.0
        self._jaccard_cache[key] = val
        return val

    @torch.no_grad()
    def _compute_S_graph_raw(self, batch_item_ids: np.ndarray) -> torch.Tensor:
        """Jaccard similarity with numpy arrays + LRU cache."""
        B = len(batch_item_ids)
        S = torch.zeros(B, B)
        if not self.neighbor_arr:
            return S
        ids = [int(x) for x in batch_item_ids]
        for i in range(B):
            for j in range(i + 1, B):
                val = self._get_jaccard(ids[i], ids[j])
                if val > 0:
                    S[i, j] = val
                    S[j, i] = val
        return S

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
) -> Dict[int, Set[int]]:
    """Build item→neighbors dict from user interaction sequences."""
    cooc = defaultdict(lambda: defaultdict(int))
    for seq in train_sequences:
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                a, b = seq[i], seq[j]
                if a != b:
                    cooc[a][b] += 1
                    cooc[b][a] += 1
    neighbors = {}
    for item, nbrs in cooc.items():
        if min_cooc > 0:
            neighbors[item] = {n for n, cnt in nbrs.items() if cnt >= min_cooc}
        else:
            neighbors[item] = set(nbrs.keys())
    return neighbors
