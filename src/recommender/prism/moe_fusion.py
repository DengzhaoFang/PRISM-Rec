"""
MoE Fusion for DSI: 3-way purified Dynamic Semantic Integration.

Sources (all equal-dimensional after projection):
  - id_emb:            (B, L, d_model)  — sequence structure
  - purified_content:  (B, L, 128)      — MCD-denoised semantics
  - purified_collab:   (B, L, 128)      — MCD-denoised behavior

Router selects top-k experts based on concatenated features,
each expert processes the full 3-source context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Expert(nn.Module):
    """Single expert for 3-way fusion."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, expert_id: int = 0):
        super().__init__()
        self.expert_id = expert_id
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
        )
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                gain = 0.3 + (expert_id * 0.2)
                nn.init.xavier_uniform_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    """Noisy Top-K router for expert selection."""

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2,
                 use_load_balancing: bool = True, load_balance_weight: float = 0.001,
                 noise_std: float = 0.05, use_noisy_gating: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.use_load_balancing = use_load_balancing
        self.load_balance_weight = load_balance_weight
        self.noise_std = noise_std
        self.use_noisy_gating = use_noisy_gating

        hidden_dim = max(input_dim // 2, 128)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts),
        )

        if use_noisy_gating:
            self.noise_weight = nn.Linear(input_dim, num_experts)
            nn.init.zeros_(self.noise_weight.weight)
            nn.init.zeros_(self.noise_weight.bias)

        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_stats: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        clean_logits = self.gate(x)

        if self.training and self.use_noisy_gating:
            noise_logits = self.noise_weight(x)
            noise = torch.randn_like(clean_logits) * F.softplus(noise_logits)
            noisy_logits = clean_logits + noise
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

            load_balance_loss = None
            if self.use_load_balancing:
                num_tokens = x.size(0) * x.size(1)
                f = expert_usage / num_tokens
                P = all_probs.mean(dim=(0, 1))
                importance_loss = self.num_experts * (f * P).sum()
                entropy = -(P * (P + 1e-8).log()).sum()
                max_entropy = torch.log(torch.tensor(float(self.num_experts), device=x.device))
                entropy_loss = (max_entropy - entropy) / max_entropy
                load_balance_loss = (importance_loss + entropy_loss) * self.load_balance_weight

            stats = {
                'expert_usage': expert_usage.cpu(),
                'load_balance_loss': load_balance_loss,
            }

        return top_k_indices, expert_weights, stats


class MoEFusion(nn.Module):
    """
    3-way MoE fusion for DSI.

    Input: id_emb (d_model) + purified_content (128D) + purified_collab (128D)
    All projected to d_model, concatenated → router + experts.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_residual = use_residual

        # Project purified modalities to d_model
        self.content_proj = nn.Linear(purified_dim, d_model)
        self.collab_proj = nn.Linear(purified_dim, d_model)
        nn.init.xavier_uniform_(self.content_proj.weight, gain=0.5)
        nn.init.zeros_(self.content_proj.bias)
        nn.init.xavier_uniform_(self.collab_proj.weight, gain=0.5)
        nn.init.zeros_(self.collab_proj.bias)
        self.content_norm = nn.LayerNorm(d_model)
        self.collab_norm = nn.LayerNorm(d_model)

        # Concat: id(d_model) + content(d_model) + collab(d_model) = 3*d_model
        concat_dim = d_model * 3

        self.experts = nn.ModuleList([
            Expert(concat_dim, expert_hidden_dim, dropout, expert_id=i)
            for i in range(num_experts)
        ])

        expert_output_dim = expert_hidden_dim // 4
        self.output_proj = nn.Linear(expert_output_dim, d_model)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.output_proj.bias)
        self.output_norm = nn.LayerNorm(d_model)

        self.router = Router(
            input_dim=concat_dim, num_experts=num_experts, top_k=top_k,
            use_load_balancing=use_load_balancing,
            load_balance_weight=load_balance_weight,
            noise_std=0.05, use_noisy_gating=True,
        )

        if use_residual:
            self.fusion_alpha = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2.0) ≈ 0.12
            logger.info(f"MoE residual alpha init: {torch.sigmoid(self.fusion_alpha).item():.4f}")

        self.dropout = nn.Dropout(dropout)
        logger.info(f"MoE Fusion: {num_experts} experts, Top-{top_k}, hidden={expert_hidden_dim}, concat={concat_dim}D")

    def forward(
        self,
        id_emb: torch.Tensor,
        purified_content: torch.Tensor,
        purified_collab: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        batch_size, seq_len, _ = id_emb.shape

        content_proj = self.content_norm(self.content_proj(purified_content))
        collab_proj = self.collab_norm(self.collab_proj(purified_collab))

        concat = torch.cat([id_emb, content_proj, collab_proj], dim=-1)  # (B, L, 3*d_model)

        expert_indices, expert_weights, router_stats = self.router(
            concat, return_stats=return_stats or self.router.use_load_balancing
        )

        expert_outputs = torch.stack([expert(concat) for expert in self.experts], dim=2)  # (B, L, E, expert_dim)

        batch_idx = torch.arange(batch_size, device=id_emb.device).view(-1, 1, 1)
        seq_idx = torch.arange(seq_len, device=id_emb.device).view(1, -1, 1)
        selected = expert_outputs[batch_idx, seq_idx, expert_indices]  # (B, L, top_k, expert_dim)
        expert_combined = (selected * expert_weights.unsqueeze(-1)).sum(dim=2)  # (B, L, expert_dim)

        fused = self.output_norm(self.output_proj(expert_combined))

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

    def get_routing_stats(self, id_emb, purified_content, purified_collab, attention_mask=None) -> Dict:
        with torch.no_grad():
            _, stats = self.forward(id_emb, purified_content, purified_collab,
                                     attention_mask=attention_mask, return_stats=True)
            return stats if stats is not None else {}
