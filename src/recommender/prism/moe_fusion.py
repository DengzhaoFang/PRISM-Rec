"""
MoE Fusion for DSI: 3-way purified Dynamic Semantic Integration.

Sources (all projected to d_model):
  - id_emb:            (B, L, d_model)  — sequence structure
  - purified_content:  (B, L, 128)      — MCD-denoised semantics
  - purified_collab:   (B, L, 128)      — MCD-denoised behavior

Router selects top-k experts based on concatenated features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.LayerNorm(hidden_dim // 4),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x): return self.net(x)


class DenseRouter(nn.Module):
    """
    Dense softmax router for modality-specific dynamic fusion.

    Produces continuous weights w ∈ [0,1]³ with Σw = 1 via softmax.
    No top-k truncation, no load balancing — all 3 modality experts
    contribute at every time step, eliminating modality dropout.
    """

    def __init__(self, input_dim: int, num_experts: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts
        hd = max(input_dim // 2, 128)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hd), nn.LayerNorm(hd), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hd, hd // 2), nn.ReLU(), nn.Linear(hd // 2, num_experts),
        )
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)  # (B, L, num_experts)
        return F.softmax(logits, dim=-1)


class Router(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, use_load_balancing=True,
                 load_balance_weight=0.001, noise_std=0.05, use_noisy_gating=True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.use_load_balancing = use_load_balancing
        self.load_balance_weight = load_balance_weight
        self.noise_std = noise_std
        self.use_noisy_gating = use_noisy_gating

        hd = max(input_dim // 2, 128)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hd), nn.LayerNorm(hd), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hd, hd // 2), nn.ReLU(), nn.Linear(hd // 2, num_experts),
        )
        if use_noisy_gating:
            self.noise_weight = nn.Linear(input_dim, num_experts)
            nn.init.zeros_(self.noise_weight.weight); nn.init.zeros_(self.noise_weight.bias)
        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5); nn.init.zeros_(m.bias)

    def forward(self, x, return_stats=False):
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        clean_logits = self.gate(x)
        if self.training and self.use_noisy_gating:
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * F.softplus(self.noise_weight(x))
        else:
            noisy_logits = clean_logits

        all_probs = F.softmax(noisy_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(all_probs, self.top_k, dim=-1)
        expert_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        stats = None
        if return_stats or self.use_load_balancing:
            expert_usage = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                expert_usage[i] = (top_k_indices == i).float().sum()
            lb_loss = None
            if self.use_load_balancing:
                nt = x.size(0) * x.size(1)
                f = expert_usage / nt
                P = all_probs.mean(dim=(0, 1))
                imp = self.num_experts * (f * P).sum()
                ent = -(P * (P + 1e-8).log()).sum()
                max_ent = torch.log(torch.tensor(float(self.num_experts), device=x.device))
                lb_loss = (imp + (max_ent - ent) / max_ent) * self.load_balance_weight
            stats = {'expert_usage': expert_usage.cpu(), 'load_balance_loss': lb_loss}

        return top_k_indices, expert_weights, stats


class MoEFusion(nn.Module):
    """
    3-way MoE fusion with purified features.

    Supports two router types:
      - "sparse": top-k sparse gating with load balancing (original)
      - "dense":  softmax gating, all experts contribute, no truncation
    """

    def __init__(
        self,
        d_model: int,
        purified_dim: int = 128,
        num_experts: int = 3,
        expert_hidden_dim: int = 256,
        top_k: int = 2,
        use_load_balancing: bool = False,
        load_balance_weight: float = 0.001,
        dropout: float = 0.1,
        use_residual: bool = True,
        router_type: str = "sparse",
    ):
        super().__init__()
        self.d_model = d_model
        self.use_residual = use_residual
        self.router_type = router_type
        self.num_experts = num_experts

        self.content_proj = nn.Linear(purified_dim, d_model)
        self.collab_proj = nn.Linear(purified_dim, d_model)
        nn.init.xavier_uniform_(self.content_proj.weight, gain=0.5)
        nn.init.zeros_(self.content_proj.bias)
        nn.init.xavier_uniform_(self.collab_proj.weight, gain=0.5)
        nn.init.zeros_(self.collab_proj.bias)
        self.content_norm = nn.LayerNorm(d_model)
        self.collab_norm = nn.LayerNorm(d_model)

        concat_dim = d_model * 3  # id + content + collab

        # Dense mode: modality-specialized experts (128D each)
        # Sparse mode: homogeneous experts all seeing full concat (384D)
        expert_input_dim = d_model if router_type == "dense" else concat_dim
        self.experts = nn.ModuleList([
            Expert(expert_input_dim, expert_hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        expert_output_dim = expert_hidden_dim // 4
        self.output_proj = nn.Linear(expert_output_dim, d_model)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.output_proj.bias)
        self.output_norm = nn.LayerNorm(d_model)

        if router_type == "dense":
            self.dense_router = DenseRouter(concat_dim, num_experts, dropout)
            self.router = None
        else:
            self.router = Router(concat_dim, num_experts, top_k, use_load_balancing,
                                 load_balance_weight, noise_std=0.05, use_noisy_gating=True)
            self.dense_router = None

        if use_residual:
            self.fusion_alpha = nn.Parameter(torch.tensor(-2.0))

        self.dropout = nn.Dropout(dropout)
        tag = "Dense Softmax" if router_type == "dense" else f"Top-{top_k} Sparse"
        logger.info(f"MoE Fusion [{tag}]: {num_experts} experts, hidden={expert_hidden_dim}, concat={concat_dim}D")

    def forward(
        self,
        id_emb: torch.Tensor,
        purified_content: torch.Tensor,
        purified_collab: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        if self.router_type == "dense":
            return self._dense_forward(id_emb, purified_content, purified_collab,
                                       attention_mask, return_stats)
        return self._sparse_forward(id_emb, purified_content, purified_collab,
                                     attention_mask, return_stats)

    def _dense_forward(self, id_emb, purified_content, purified_collab,
                        attention_mask, return_stats):
        B, seq_len, _ = id_emb.shape

        content_proj = self.content_norm(self.content_proj(purified_content))
        collab_proj = self.collab_norm(self.collab_proj(purified_collab))
        concat = torch.cat([id_emb, content_proj, collab_proj], dim=-1)  # (B, L, concat_dim)

        # Softmax weights: w0 + w1 + w2 = 1, continuous, no truncation
        # Router sees full 384D context to make global routing decisions
        weights = self.dense_router(concat)  # (B, L, num_experts)

        # Modality-specialized experts: each only sees its own modality (128D)
        #   E0: ID specialist    (sequence structure)
        #   E1: Content specialist (text semantics)
        #   E2: Collab specialist  (behavioral patterns)
        e0_out = self.experts[0](id_emb)          # (B, L, D)
        e1_out = self.experts[1](content_proj)    # (B, L, D)
        e2_out = self.experts[2](collab_proj)     # (B, L, D)
        expert_outputs = torch.stack([e0_out, e1_out, e2_out], dim=2)  # (B, L, 3, D)
        combined = (expert_outputs * weights.unsqueeze(-1)).sum(dim=2)  # (B, L, D)

        fused = self.output_norm(self.output_proj(combined))

        if self.use_residual:
            alpha = torch.sigmoid(self.fusion_alpha)
            output = id_emb + alpha * (fused - id_emb)
        else:
            output = fused

        stats = None
        if return_stats:
            avg_weights = weights.mean(dim=(0, 1))  # (num_experts,)
            stats = {
                'expert_usage': avg_weights.detach().cpu(),
                'expert_weights': avg_weights.detach().cpu(),
                'fusion_alpha': alpha.item() if self.use_residual else None,
            }

        return output, stats

    def _sparse_forward(self, id_emb, purified_content, purified_collab,
                         attention_mask, return_stats):
        B, seq_len, _ = id_emb.shape

        content_proj = self.content_norm(self.content_proj(purified_content))
        collab_proj = self.collab_norm(self.collab_proj(purified_collab))
        concat = torch.cat([id_emb, content_proj, collab_proj], dim=-1)  # (B, L, concat_dim)

        # 1. Router: decide which experts handle which tokens
        expert_indices, expert_weights, router_stats = self.router(concat, return_stats=return_stats)

        # 2. Flatten to token level
        N = B * seq_len
        flat_concat = concat.reshape(N, -1)
        flat_indices = expert_indices.reshape(N, -1)
        flat_weights = expert_weights.reshape(N, -1)

        # 3. True sparse: each expert only processes tokens dispatched to it
        fused = torch.zeros(N, self.d_model, device=id_emb.device)

        for e in range(len(self.experts)):
            token_mask = (flat_indices == e)
            if not token_mask.any():
                continue

            token_rows, rank_cols = token_mask.nonzero(as_tuple=True)
            expert_input = flat_concat[token_rows]
            expert_out = self.experts[e](expert_input)
            projected = self.output_proj(expert_out)
            w = flat_weights[token_rows, rank_cols].unsqueeze(-1)

            fused.index_add_(0, token_rows, w * projected)

        fused = self.output_norm(fused)
        fused = fused.reshape(B, seq_len, self.d_model)

        # 4. Residual connection
        if self.use_residual:
            alpha = torch.sigmoid(self.fusion_alpha)
            output = id_emb + alpha * (fused - id_emb)
        else:
            output = fused

        stats = None
        if return_stats and router_stats is not None:
            stats = router_stats
            if self.use_residual:
                stats['fusion_alpha'] = alpha.item()

        return output, stats
