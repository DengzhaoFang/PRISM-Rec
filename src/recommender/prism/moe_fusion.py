"""
MoE Fusion for DSI: 3-way purified Dynamic Semantic Integration.

Sources (all projected to d_model):
  - id_emb:            (B, L, d_model)  — sequence structure
  - purified_content:  (B, L, 128)      — MCD-denoised semantics
  - purified_collab:   (B, L, 128)      — MCD-denoised behavior
  - teacher:           (B, teacher_dim) — stage1 recommendation prototype (optional)

Router selects top-k experts based on concatenated features.
When teacher is provided, routing is conditioned on the teacher prototype,
enabling item-level modality reliability estimation.
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

    Includes an entropy regularization term that penalises weight
    concentration onto a single expert, preventing modality collapse.
    """

    def __init__(self, input_dim: int, num_experts: int = 3, dropout: float = 0.1,
                 entropy_reg_weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.entropy_reg_weight = entropy_reg_weight
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
        weights = F.softmax(logits, dim=-1)

        if self.training and self.entropy_reg_weight > 0:
            avg_w = weights.mean(dim=(0, 1))  # (num_experts,)
            entropy = -(avg_w * (avg_w + 1e-8).log()).sum()
            max_entropy = torch.log(torch.tensor(float(self.num_experts), device=x.device))
            self._entropy_penalty = (max_entropy - entropy) / (max_entropy + 1e-8) * self.entropy_reg_weight
        else:
            self._entropy_penalty = torch.tensor(0.0, device=x.device)

        return weights


class TeacherConditionedRouter(nn.Module):
    """
    Item-level teacher-conditioned modality router.

    Uses the stage1 teacher prototype (which encodes "what recommendation
    context this item belongs to") to produce item-specific modality weights.
    This enables head/tail-aware routing: items with rich collaborative context
    naturally get different modality emphasis than cold-start items.

    Architecture:
      teacher_proj:  teacher_dim → gate_hidden
      gate:          [concat_modalities, teacher_proj] → softmax weights
      modality_temp: learnable per-modality temperature
    """

    def __init__(self, input_dim: int, teacher_dim: int = 832,
                 num_experts: int = 3, dropout: float = 0.1,
                 entropy_reg_weight: float = 0.05):
        super().__init__()
        self.num_experts = num_experts
        self.entropy_reg_weight = entropy_reg_weight

        # Teacher → routing context
        gate_hidden = 128
        self.teacher_proj = nn.Sequential(
            nn.Linear(teacher_dim, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.GELU(),
        )

        # Gate: [modality_concat + teacher_context] → expert weights
        gate_input_dim = input_dim + gate_hidden
        hd = max(gate_input_dim // 2, 128)
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hd), nn.LayerNorm(hd), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hd, hd // 2), nn.ReLU(),
            nn.Linear(hd // 2, num_experts),
        )
        # Learnable per-modality temperature (initialised to 1, sharpens adaptively)
        self.modality_temp = nn.Parameter(torch.ones(num_experts))

        for m in self.gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, input_dim) — concat of [id_emb, content_proj, collab_proj]
            teacher: (B, teacher_dim) — stage1 teacher prototype
        Returns:
            weights: (B, L, num_experts) — softmax modality weights
        """
        B, L, _ = x.shape

        t_proj = self.teacher_proj(teacher)           # (B, gate_hidden)
        t_proj = t_proj.unsqueeze(1).expand(B, L, -1)  # (B, L, gate_hidden)

        gate_input = torch.cat([x, t_proj], dim=-1)    # (B, L, input_dim + gate_hidden)
        logits = self.gate(gate_input)

        # Temperature-scaled softmax (temp always positive via abs)
        weights = F.softmax(logits / self.modality_temp.abs().clamp(min=0.1), dim=-1)

        # Entropy regularization
        if self.training and self.entropy_reg_weight > 0:
            avg_w = weights.mean(dim=(0, 1))
            entropy = -(avg_w * (avg_w + 1e-8).log()).sum()
            max_ent = torch.log(torch.tensor(float(self.num_experts), device=x.device))
            self._entropy_penalty = (max_ent - entropy) / (max_ent + 1e-8) * self.entropy_reg_weight
        else:
            self._entropy_penalty = torch.tensor(0.0, device=x.device)

        return weights


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
    3-way MoE fusion with purified features and optional teacher conditioning.

    Supports two router types:
      - "sparse": top-k sparse gating with load balancing (original)
      - "dense":  softmax gating, all experts contribute, no truncation

    When use_teacher_gate=True, the dense router is replaced with
    TeacherConditionedRouter which uses the stage1 teacher prototype
    for item-level modality reliability estimation.
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
        use_teacher_gate: bool = False,
        teacher_dim: int = 832,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_residual = use_residual
        self.router_type = router_type
        self.num_experts = num_experts
        self.use_teacher_gate = use_teacher_gate

        self.content_proj = nn.Linear(purified_dim, d_model)
        self.collab_proj = nn.Linear(purified_dim, d_model)
        nn.init.xavier_uniform_(self.content_proj.weight, gain=0.5)
        nn.init.zeros_(self.content_proj.bias)
        nn.init.xavier_uniform_(self.collab_proj.weight, gain=0.5)
        nn.init.zeros_(self.collab_proj.bias)
        self.content_norm = nn.LayerNorm(d_model)
        self.collab_norm = nn.LayerNorm(d_model)

        concat_dim = d_model * 3  # id + content + collab

        # Dense mode: each expert sees only its own modality (d_model-D).
        # Sparse mode: experts see the full concat (top-k router selects).
        if router_type == "dense":
            expert_input_dim = d_model
        else:
            expert_input_dim = concat_dim
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
            if use_teacher_gate:
                self.tc_router = TeacherConditionedRouter(
                    concat_dim, teacher_dim=teacher_dim,
                    num_experts=num_experts, dropout=dropout,
                    entropy_reg_weight=0.05)
                self.dense_router = None
            else:
                self.dense_router = DenseRouter(concat_dim, num_experts, dropout,
                                                entropy_reg_weight=0.0)
                self.tc_router = None
            self.router = None
        else:
            self.router = Router(concat_dim, num_experts, top_k, use_load_balancing,
                                 load_balance_weight, noise_std=0.05, use_noisy_gating=True)
            self.dense_router = None
            self.tc_router = None

        if use_residual:
            self.fusion_alpha = nn.Parameter(torch.tensor(-2.0))

        self.dropout = nn.Dropout(dropout)
        tag_parts = ["Dense Softmax"]
        if use_teacher_gate:
            tag_parts.append("+TeacherGate")
        tag = " ".join(tag_parts)
        logger.info(f"MoE Fusion [{tag}]: {num_experts} experts, hidden={expert_hidden_dim}, concat={concat_dim}D")

    def forward(
        self,
        id_emb: torch.Tensor,
        purified_content: torch.Tensor,
        purified_collab: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        teacher: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        if self.router_type == "dense":
            return self._dense_forward(id_emb, purified_content, purified_collab,
                                       attention_mask, return_stats, teacher=teacher)
        return self._sparse_forward(id_emb, purified_content, purified_collab,
                                     attention_mask, return_stats)

    def _dense_forward(self, id_emb, purified_content, purified_collab,
                        attention_mask, return_stats, teacher=None):
        B, seq_len, _ = id_emb.shape

        content_proj = self.content_norm(self.content_proj(purified_content))
        collab_proj = self.collab_norm(self.collab_proj(purified_collab))
        concat = torch.cat([id_emb, content_proj, collab_proj], dim=-1)  # (B, L, concat_dim)

        # Routing: teacher-conditioned (item-specific) or universal
        if self.tc_router is not None and teacher is not None:
            weights = self.tc_router(concat, teacher)
            ent_penalty = self.tc_router._entropy_penalty
        elif self.dense_router is not None:
            weights = self.dense_router(concat)
            ent_penalty = self.dense_router._entropy_penalty
        else:
            # Teacher gate enabled but teacher not provided — use uniform weights
            weights = torch.ones(B, seq_len, self.num_experts, device=id_emb.device) / self.num_experts
            ent_penalty = torch.tensor(0.0, device=id_emb.device)

        # Modality-specialized experts: each sees only its own modality.
        # Teacher guidance is applied at the routing level, not inside experts.
        e0_out = self.experts[0](id_emb)
        e1_out = self.experts[1](content_proj)
        e2_out = self.experts[2](collab_proj)

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
            avg_weights = weights.mean(dim=(0, 1))
            stats = {
                'expert_usage': avg_weights.detach().cpu(),
                'expert_weights': avg_weights.detach().cpu(),
                'fusion_alpha': alpha.item() if self.use_residual else None,
                'entropy_penalty': ent_penalty,
            }

        return output, stats

    def _sparse_forward(self, id_emb, purified_content, purified_collab,
                         attention_mask, return_stats):
        B, seq_len, _ = id_emb.shape

        content_proj = self.content_norm(self.content_proj(purified_content))
        collab_proj = self.collab_norm(self.collab_proj(purified_collab))
        concat = torch.cat([id_emb, content_proj, collab_proj], dim=-1)

        expert_indices, expert_weights, router_stats = self.router(concat, return_stats=return_stats)

        N = B * seq_len
        flat_concat = concat.reshape(N, -1)
        flat_indices = expert_indices.reshape(N, -1)
        flat_weights = expert_weights.reshape(N, -1)

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
