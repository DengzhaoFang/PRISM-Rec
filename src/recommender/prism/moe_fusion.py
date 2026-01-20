"""
Mixture of Experts (MoE) Fusion Module.

Provides non-linear multi-source embedding fusion using expert networks.
Each expert specializes in different fusion patterns, and a router dynamically
selects the best experts for each input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Expert(nn.Module):
    """Single expert network for MoE fusion.
    
    Each expert learns a different way to combine ID, content, and collab embeddings.
    Uses diverse initialization to encourage specialization.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1, expert_id: int = 0):
        """Initialize expert network.
        
        Args:
            input_dim: Input dimension (concatenated embeddings)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            expert_id: Expert identifier for diverse initialization
        """
        super().__init__()
        
        self.expert_id = expert_id
        
        # Deeper network for more expressive power
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),  # Output to d_model will be handled by projection
            nn.LayerNorm(hidden_dim // 4)
        )
        
        # Diverse initialization: each expert starts with different biases
        # This encourages experts to specialize in different fusion patterns
        for i, module in enumerate(self.net.modules()):
            if isinstance(module, nn.Linear):
                # Use different gains for different experts
                gain = 0.3 + (expert_id * 0.2)  # 0.3, 0.5, 0.7, 0.9, ...
                nn.init.xavier_uniform_(module.weight, gain=gain)
                
                # Add small expert-specific bias to encourage diversity
                if i == 0:  # First layer
                    nn.init.uniform_(module.bias, -0.1 * (expert_id + 1), 0.1 * (expert_id + 1))
                else:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Concatenated embeddings (B, L, input_dim)
        
        Returns:
            Expert output (B, L, hidden_dim // 4)
        """
        return self.net(x)


class Router(nn.Module):
    """Router network for MoE with Noisy Top-K routing.
    
    Dynamically selects which experts to use for each input based on context.
    Uses noise injection to promote exploration and prevent expert collapse.
    Supports Top-K routing with enhanced load balancing.
    """
    
    def __init__(
        self,
        input_dim: int,  # Changed from d_model to input_dim
        num_experts: int,
        top_k: int = 2,
        use_load_balancing: bool = True,
        load_balance_weight: float = 0.1,
        noise_std: float = 0.1,
        use_noisy_gating: bool = True
    ):
        """Initialize router.
        
        Args:
            input_dim: Input dimension (concatenated embeddings)
            num_experts: Number of experts
            top_k: Number of experts to select per input
            use_load_balancing: Whether to use load balancing loss
            load_balance_weight: Weight for load balancing loss (increased default)
            noise_std: Standard deviation for input noise
            use_noisy_gating: Whether to use noisy top-k gating
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.use_load_balancing = use_load_balancing
        self.load_balance_weight = load_balance_weight
        self.noise_std = noise_std
        self.use_noisy_gating = use_noisy_gating
        
        # Router network: maps input to expert scores
        # Deeper network for better routing decisions
        hidden_dim = max(input_dim // 2, 128)  # Adaptive hidden dimension
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Learnable noise parameters for noisy top-k gating
        if use_noisy_gating:
            self.noise_weight = nn.Linear(input_dim, num_experts)
            nn.init.zeros_(self.noise_weight.weight)
            nn.init.zeros_(self.noise_weight.bias)
        
        # Initialize with small weights for stable training
        for module in self.gate.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """Route inputs to experts with noisy top-k gating.
        
        Args:
            x: Concatenated embeddings (B, L, input_dim)
            return_stats: Whether to return routing statistics
        
        Returns:
            - expert_indices: Selected expert indices (B, L, top_k)
            - expert_weights: Weights for selected experts (B, L, top_k)
            - stats: Optional routing statistics
        """
        batch_size, seq_len, _ = x.shape
        
        # Add small input noise during training to prevent overfitting
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        
        # Compute routing scores (clean logits)
        clean_logits = self.gate(x)  # (B, L, num_experts)
        
        # Noisy Top-K Gating (during training only)
        if self.training and self.use_noisy_gating:
            # Add learnable noise to logits
            noise_logits = self.noise_weight(x)
            noise = torch.randn_like(clean_logits) * F.softplus(noise_logits)
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
        
        # Softmax over all experts (not sigmoid) for proper probability distribution
        all_probs = F.softmax(noisy_logits, dim=-1)  # (B, L, num_experts)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(all_probs, self.top_k, dim=-1)  # (B, L, top_k)
        
        # Renormalize selected expert weights to sum to 1
        expert_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Compute statistics for monitoring and load balancing
        stats = None
        if return_stats or self.use_load_balancing:
            # Expert usage: count how many times each expert is selected (for monitoring)
            expert_usage = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                expert_usage[i] = (top_k_indices == i).float().sum()
            
            # Normalize to get distribution
            expert_dist = expert_usage / (expert_usage.sum() + 1e-8)
            
            # Enhanced load balancing loss (FIXED: use differentiable probabilities)
            load_balance_loss = None
            if self.use_load_balancing:
                # CRITICAL FIX: Use all_probs (differentiable) instead of expert_usage (discrete)
                # This ensures gradients can flow back to the router
                
                # Method 1: Importance loss (from Switch Transformer paper)
                # Encourages uniform distribution of routing probabilities
                # f_i = fraction of tokens routed to expert i
                # P_i = average routing probability to expert i
                batch_size, seq_len, _ = all_probs.shape
                num_tokens = batch_size * seq_len
                
                # Fraction of tokens routed to each expert (based on top-k selection)
                # This is still discrete but used only for the importance term
                f = expert_usage / num_tokens  # (num_experts,)
                
                # Average routing probability to each expert (differentiable!)
                P = all_probs.mean(dim=(0, 1))  # (num_experts,)
                
                # Importance loss: encourages f_i and P_i to be balanced
                # loss = num_experts * sum(f_i * P_i)
                # When balanced, each expert gets 1/num_experts of tokens
                importance_loss = self.num_experts * (f * P).sum()
                
                # Method 2: Entropy-based regularization (encourage high entropy)
                # Higher entropy = more uniform distribution
                # Use P (differentiable) instead of expert_dist (discrete)
                entropy = -(P * (P + 1e-8).log()).sum()
                max_entropy = torch.log(torch.tensor(self.num_experts, dtype=torch.float32, device=x.device))
                entropy_loss = (max_entropy - entropy) / max_entropy
                
                # Combine both losses
                load_balance_loss = (importance_loss + entropy_loss) * self.load_balance_weight
                
                # For monitoring: also compute CV^2 (discrete, not used in loss)
                mean_usage = expert_usage.mean()
                var_usage = expert_usage.var()
                cv_squared = var_usage / (mean_usage ** 2 + 1e-8)
            
            stats = {
                'expert_usage': expert_usage.cpu(),
                'expert_dist': expert_dist.cpu(),
                'load_balance_loss': load_balance_loss,
                'avg_weights': expert_weights.mean(dim=(0, 1)).cpu(),
                'cv_squared': cv_squared.item() if self.use_load_balancing else 0.0,
                'entropy': entropy.item() if self.use_load_balancing else 0.0,
                'importance_loss': importance_loss.item() if self.use_load_balancing else 0.0,
                'routing_probs': P.cpu() if self.use_load_balancing else None
            }
        
        return top_k_indices, expert_weights, stats


class MoEFusion(nn.Module):
    """Mixture of Experts fusion for multi-source embeddings.
    
    Uses multiple expert networks to capture non-linear interactions between
    ID, content, and collaborative embeddings. A router dynamically selects
    the best experts for each input.
    """
    
    def __init__(
        self,
        d_model: int,
        content_dim: int = 768,
        collab_dim: int = 64,
        num_experts: int = 4,
        expert_hidden_dim: int = 512,
        top_k: int = 2,
        use_load_balancing: bool = True,
        load_balance_weight: float = 0.01,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_improved_projection: bool = False,
        codebook_dim: int = 32
    ):
        """Initialize MoE fusion module.
        
        Args:
            d_model: Model dimension (T5's d_model)
            content_dim: Content embedding dimension
            collab_dim: Collaborative embedding dimension
            num_experts: Number of expert networks
            expert_hidden_dim: Hidden dimension for each expert
            top_k: Number of experts to select per input
            use_load_balancing: Whether to use load balancing loss
            load_balance_weight: Weight for load balancing loss
            dropout: Dropout rate
            use_residual: Whether to use residual connection with learnable alpha
            use_improved_projection: Whether to use improved projection mechanism
            codebook_dim: Codebook embedding dimension (for improved projection)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_residual = use_residual
        self.use_improved_projection = use_improved_projection
        self.codebook_dim = codebook_dim
        
        # Validation: improved projection requires codebook embeddings
        if use_improved_projection:
            logger.info(
                "⚠️ Improved projection enabled. Ensure multimodal fusion is enabled "
                "and codebook vectors are provided in the dataset!"
            )
        
        # Input normalization
        self.content_input_norm = nn.LayerNorm(content_dim)
        self.collab_input_norm = nn.LayerNorm(collab_dim)
        
        if use_improved_projection:
            # Improved projection mechanism:
            # Content: 768 → 256
            # ID: 128 → 128 (no projection needed if d_model=128)
            # Collab: 64 → 64 (no projection needed)
            # Codebook: 32 (will be concatenated)
            
            self.content_proj_dim = 256
            self.id_proj_dim = d_model  # Keep ID at d_model
            self.collab_proj_dim = 64
            
            # Content projection: 768 → 256
            self.content_proj = nn.Linear(content_dim, self.content_proj_dim)
            nn.init.xavier_uniform_(self.content_proj.weight, gain=0.5)
            nn.init.zeros_(self.content_proj.bias)
            self.content_norm = nn.LayerNorm(self.content_proj_dim)
            
            # Collab projection: 64 → 64 (identity-like, but still learnable)
            self.collab_proj = nn.Linear(collab_dim, self.collab_proj_dim)
            nn.init.xavier_uniform_(self.collab_proj.weight, gain=0.5)
            nn.init.zeros_(self.collab_proj.bias)
            self.collab_norm = nn.LayerNorm(self.collab_proj_dim)
            
            # Total concatenated dimension: 256 + 128 + 64 + 32 = 480
            concat_dim = self.content_proj_dim + self.id_proj_dim + self.collab_proj_dim + codebook_dim
            
            logger.info(
                f"MoE Fusion with IMPROVED projection: "
                f"Content({content_dim}→{self.content_proj_dim}) + "
                f"ID({d_model}→{self.id_proj_dim}) + "
                f"Collab({collab_dim}→{self.collab_proj_dim}) + "
                f"Codebook({codebook_dim}) = {concat_dim}D"
            )
        
        else:
            # Original projection mechanism: all to d_model
            self.content_proj = nn.Linear(content_dim, d_model)
            self.collab_proj = nn.Linear(collab_dim, d_model)
            
            nn.init.xavier_uniform_(self.content_proj.weight, gain=0.5)
            nn.init.zeros_(self.content_proj.bias)
            nn.init.xavier_uniform_(self.collab_proj.weight, gain=0.5)
            nn.init.zeros_(self.collab_proj.bias)
            
            # Layer norms after projection
            self.content_norm = nn.LayerNorm(d_model)
            self.collab_norm = nn.LayerNorm(d_model)
            
            concat_dim = d_model * 3
            
            logger.info(
                f"MoE Fusion with ORIGINAL projection: "
                f"Content({content_dim}→{d_model}) + "
                f"ID({d_model}) + "
                f"Collab({collab_dim}→{d_model}) = {concat_dim}D"
            )
        
        # Create expert networks with diverse initialization
        self.experts = nn.ModuleList([
            Expert(concat_dim, expert_hidden_dim, dropout, expert_id=i)
            for i in range(num_experts)
        ])
        
        # Output projection: expert output → d_model
        expert_output_dim = expert_hidden_dim // 4
        self.output_proj = nn.Linear(expert_output_dim, d_model)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.5)
        nn.init.zeros_(self.output_proj.bias)
        self.output_norm = nn.LayerNorm(d_model)
        
        # Router network with enhanced features
        self.router = Router(
            input_dim=concat_dim,  # Use concat_dim instead of d_model
            num_experts=num_experts,
            top_k=top_k,
            use_load_balancing=use_load_balancing,
            load_balance_weight=load_balance_weight,
            noise_std=0.05,  # Small input noise for robustness
            use_noisy_gating=True  # Enable noisy top-k gating
        )
        
        # Learnable fusion strength (alpha) for residual connection
        if use_residual:
            # Initialize to -2.0, which gives sigmoid(-2.0) ≈ 0.12
            self.fusion_alpha = nn.Parameter(torch.tensor(-2.0))
            logger.info(f"MoE fusion with learnable alpha (starts at ~0.12)")
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(
            f"MoE Fusion initialized: {num_experts} experts, "
            f"Top-{top_k}, hidden_dim={expert_hidden_dim}, "
            f"load_balancing={use_load_balancing}"
        )
    
    def forward(
        self,
        id_emb: torch.Tensor,
        content_emb: torch.Tensor,
        collab_emb: torch.Tensor,
        codebook_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Fuse multi-source embeddings using MoE.
        
        Args:
            id_emb: ID embeddings (B, L, d_model)
            content_emb: Content embeddings (B, L, content_dim)
            collab_emb: Collaborative embeddings (B, L, collab_dim)
            codebook_emb: Codebook embeddings (B, L, codebook_dim), optional
            attention_mask: Attention mask (B, L)
            return_stats: Whether to return routing statistics
        
        Returns:
            - Fused embeddings (B, L, d_model)
            - Optional statistics dictionary
        """
        batch_size, seq_len, _ = id_emb.shape
        
        # Normalize and project inputs
        content_emb = self.content_input_norm(content_emb)
        collab_emb = self.collab_input_norm(collab_emb)
        
        content_proj = self.content_norm(self.content_proj(content_emb))
        collab_proj = self.collab_norm(self.collab_proj(collab_emb))
        
        # Concatenate all sources for routing and expert processing
        if self.use_improved_projection:
            # Improved projection: Content(256) + ID(128) + Collab(64) + Codebook(32)
            if codebook_emb is not None:
                concat = torch.cat([content_proj, id_emb, collab_proj, codebook_emb], dim=-1)
            else:
                # CRITICAL: If no codebook embedding provided, use zeros
                # This must be consistent between training and inference!
                # Log warning once
                if not hasattr(self, '_codebook_warning_logged'):
                    logger.warning(
                        "⚠️ Improved projection mode enabled but no codebook_emb provided. "
                        "Using zero padding. Ensure this is consistent between training and inference!"
                    )
                    self._codebook_warning_logged = True
                
                zero_codebook = torch.zeros(
                    batch_size, seq_len, self.codebook_dim,
                    device=id_emb.device, dtype=id_emb.dtype
                )
                concat = torch.cat([content_proj, id_emb, collab_proj, zero_codebook], dim=-1)
        else:
            # Original projection: all projected to d_model, then concatenated
            concat = torch.cat([id_emb, content_proj, collab_proj], dim=-1)  # (B, L, 3*d_model)
        
        # Route to experts
        expert_indices, expert_weights, router_stats = self.router(
            concat, return_stats=return_stats or self.router.use_load_balancing
        )  # (B, L, top_k), (B, L, top_k)
        
        # Process with selected experts
        # For efficiency, we process all inputs with all experts, then select
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(concat)  # (B, L, expert_output_dim)
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=2)  # (B, L, num_experts, expert_output_dim)
        
        # Gather selected expert outputs
        # expert_indices: (B, L, top_k)
        # expert_outputs: (B, L, num_experts, expert_output_dim)
        batch_indices = torch.arange(batch_size, device=id_emb.device).view(-1, 1, 1)
        seq_indices = torch.arange(seq_len, device=id_emb.device).view(1, -1, 1)
        
        selected_outputs = expert_outputs[
            batch_indices, seq_indices, expert_indices
        ]  # (B, L, top_k, expert_output_dim)
        
        # Weighted combination of selected experts
        expert_weights_expanded = expert_weights.unsqueeze(-1)  # (B, L, top_k, 1)
        expert_combined = (selected_outputs * expert_weights_expanded).sum(dim=2)  # (B, L, expert_output_dim)
        
        # Project to d_model
        fused = self.output_norm(self.output_proj(expert_combined))  # (B, L, d_model)
        
        # Apply residual connection with learnable alpha
        if self.use_residual:
            alpha = torch.sigmoid(self.fusion_alpha)
            output = id_emb + alpha * (fused - id_emb)
        else:
            output = fused
        
        # Prepare statistics
        stats = None
        if return_stats and router_stats is not None:
            stats = router_stats
            if self.use_residual:
                stats['fusion_alpha'] = alpha.item()
        
        return output, stats
    
    def get_routing_stats(
        self,
        id_emb: torch.Tensor,
        content_emb: torch.Tensor,
        collab_emb: torch.Tensor,
        codebook_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, any]:
        """Get routing statistics for monitoring.
        
        Args:
            id_emb: ID embeddings (B, L, d_model)
            content_emb: Content embeddings (B, L, content_dim)
            collab_emb: Collaborative embeddings (B, L, collab_dim)
            codebook_emb: Codebook embeddings (B, L, codebook_dim), optional
            attention_mask: Attention mask (B, L)
        
        Returns:
            Dictionary with routing statistics
        """
        with torch.no_grad():
            _, stats = self.forward(
                id_emb, content_emb, collab_emb,
                codebook_emb=codebook_emb,
                attention_mask=attention_mask,
                return_stats=True
            )
            return stats if stats is not None else {}
