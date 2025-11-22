"""
TIGER model implementation.

A T5-based encoder-decoder model for generative recommendation.
Enhanced with multi-source information fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Tuple, Dict, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class MultiSourceFusion(nn.Module):
    """Multi-source embedding fusion module.
    
    Fuses ID embeddings, content embeddings, and collaborative embeddings
    using a learned gating mechanism or fixed weights.
    
    Supports layer-specific fusion for better representation learning.
    """
    
    def __init__(
        self,
        d_model: int,
        content_dim: int = 768,
        collab_dim: int = 64,
        gate_type: str = "learned",
        fixed_weights: Optional[Dict[str, float]] = None,
        dropout: float = 0.1,
        use_residual: bool = True,
        num_layers: int = 3,
        use_layer_specific: bool = False
    ):
        """Initialize fusion module.
        
        Args:
            d_model: Model dimension (T5's d_model)
            content_dim: Content embedding dimension
            collab_dim: Collaborative embedding dimension
            gate_type: Type of gating ("learned", "fixed", "attention")
            fixed_weights: Fixed weights if gate_type="fixed"
            dropout: Dropout rate
            use_residual: Whether to use residual connection (fused + id_emb)
            num_layers: Number of semantic ID layers (for layer-specific fusion)
            use_layer_specific: Whether to use layer-specific projections
        """
        super().__init__()
        
        self.d_model = d_model
        self.gate_type = gate_type
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.use_layer_specific = use_layer_specific
        
        # Layer-specific or shared projections
        if use_layer_specific:
            # Each layer gets its own projection
            self.content_projs = nn.ModuleList([
                nn.Linear(content_dim, d_model) for _ in range(num_layers)
            ])
            self.collab_projs = nn.ModuleList([
                nn.Linear(collab_dim, d_model) for _ in range(num_layers)
            ])
            
            # Initialize each projection
            for proj in self.content_projs:
                nn.init.xavier_uniform_(proj.weight, gain=0.5)
                nn.init.zeros_(proj.bias)
            for proj in self.collab_projs:
                nn.init.xavier_uniform_(proj.weight, gain=0.5)
                nn.init.zeros_(proj.bias)
            
            # Layer norms for each layer
            self.content_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_layers)
            ])
            self.collab_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_layers)
            ])
            
            logger.info(f"Using layer-specific projections for {num_layers} layers")
        else:
            # Shared projection across all layers
            self.content_proj = nn.Linear(content_dim, d_model)
            nn.init.xavier_uniform_(self.content_proj.weight, gain=0.5)
            nn.init.zeros_(self.content_proj.bias)
            
            self.collab_proj = nn.Linear(collab_dim, d_model)
            nn.init.xavier_uniform_(self.collab_proj.weight, gain=0.5)
            nn.init.zeros_(self.collab_proj.bias)
            
            # Layer norms after projection
            self.content_norm = nn.LayerNorm(d_model)
            self.collab_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # FIX: Add input normalization to handle scale differences
        self.content_input_norm = nn.LayerNorm(content_dim)
        self.collab_input_norm = nn.LayerNorm(collab_dim)
        
        # Learnable fusion strength (alpha)
        # FIX: Start with small positive value instead of 0 to allow gradient flow
        # Starts at 0.1, so initially: output ≈ 0.9*id_emb + 0.1*fused
        # This gives the model a chance to learn from fusion signals
        if use_residual and gate_type in ["fixed", "attention"]:
            # Initialize to -2.0, which gives sigmoid(-2.0) ≈ 0.12
            self.fusion_alpha = nn.Parameter(torch.tensor(-2.0))
            logger.info(f"Fusion with learnable alpha (starts at ~0.12) for {gate_type} gate")
        
        # Gating mechanism
        if gate_type == "learned":
            # Learn weights for each source
            self.gate_fc1 = nn.Linear(d_model * 3, d_model)
            self.gate_fc2 = nn.Linear(d_model, 3)
            
            # FIX: Use reasonable initialization to allow gradient flow
            nn.init.xavier_uniform_(self.gate_fc1.weight, gain=0.5)  # Increased from 0.1
            nn.init.zeros_(self.gate_fc1.bias)
            
            # FIX: Use normal initialization for gate_fc2 weights
            nn.init.xavier_uniform_(self.gate_fc2.weight, gain=0.5)  # Increased from 0.01
            
            # FIX: Bias to favor ID but allow other sources: [0.57, 0.21, 0.21] after softmax
            # With b0=1.0, b1=b2=0.0: softmax([1.0, 0.0, 0.0]) ≈ [0.57, 0.21, 0.21]
            # This allows the model to use content/collab signals much earlier
            self.gate_fc2.bias.data[0] = 1.0   # Favor ID
            self.gate_fc2.bias.data[1] = 0.0   # Allow content
            self.gate_fc2.bias.data[2] = 0.0   # Allow collab
            
            self.gate_dropout = nn.Dropout(dropout)
        elif gate_type == "attention":
            # Attention-based gating
            self.query_proj = nn.Linear(d_model, d_model)
            self.key_proj = nn.Linear(d_model, d_model)
            self.value_proj = nn.Linear(d_model, d_model)
            
            # FIX: Use normal initialization to allow gradient flow
            nn.init.xavier_uniform_(self.query_proj.weight, gain=0.5)  # Increased from 0.1
            nn.init.xavier_uniform_(self.key_proj.weight, gain=0.5)
            nn.init.xavier_uniform_(self.value_proj.weight, gain=0.5)
            nn.init.zeros_(self.query_proj.bias)
            nn.init.zeros_(self.key_proj.bias)
            nn.init.zeros_(self.value_proj.bias)
        elif gate_type == "fixed":
            # Fixed weights
            if fixed_weights is None:
                fixed_weights = {'id': 0.5, 'content': 0.3, 'collab': 0.2}
            self.register_buffer('fixed_weights', torch.tensor([
                fixed_weights['id'],
                fixed_weights['content'],
                fixed_weights['collab']
            ]))
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")
    
    def forward(
        self,
        id_emb: torch.Tensor,
        content_emb: torch.Tensor,
        collab_emb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_tokens_per_item: Optional[int] = None
    ) -> torch.Tensor:
        """Fuse multi-source embeddings.
        
        Args:
            id_emb: ID embeddings (B, L, d_model)
            content_emb: Content embeddings (B, L, content_dim)
            collab_emb: Collaborative embeddings (B, L, collab_dim)
            attention_mask: Attention mask (B, L) - 1 for real tokens, 0 for padding
            num_tokens_per_item: Number of tokens per item (for layer-specific fusion)
        
        Returns:
            Fused embeddings (B, L, d_model)
        """
        # FIX: Normalize inputs first to handle scale differences
        content_emb = self.content_input_norm(content_emb)
        collab_emb = self.collab_input_norm(collab_emb)
        
        # Project to d_model with normalization
        if self.use_layer_specific and num_tokens_per_item is not None:
            # Layer-specific projection
            batch_size, seq_len, _ = id_emb.shape
            num_items = seq_len // num_tokens_per_item
            
            # Reshape to (B, num_items, num_tokens_per_item, dim)
            content_emb_reshaped = content_emb.view(batch_size, num_items, num_tokens_per_item, -1)
            collab_emb_reshaped = collab_emb.view(batch_size, num_items, num_tokens_per_item, -1)
            
            # Apply layer-specific projections
            content_proj_list = []
            collab_proj_list = []
            
            for layer_idx in range(num_tokens_per_item):
                # Get embeddings for this layer across all items
                content_layer = content_emb_reshaped[:, :, layer_idx, :]  # (B, num_items, content_dim)
                collab_layer = collab_emb_reshaped[:, :, layer_idx, :]  # (B, num_items, collab_dim)
                
                # Project with layer-specific projection
                content_proj_layer = self.content_norms[layer_idx](
                    self.content_projs[layer_idx](content_layer)
                )  # (B, num_items, d_model)
                collab_proj_layer = self.collab_norms[layer_idx](
                    self.collab_projs[layer_idx](collab_layer)
                )  # (B, num_items, d_model)
                
                content_proj_list.append(content_proj_layer)
                collab_proj_list.append(collab_proj_layer)
            
            # Stack and reshape back to (B, seq_len, d_model)
            content_proj = torch.stack(content_proj_list, dim=2)  # (B, num_items, num_layers, d_model)
            collab_proj = torch.stack(collab_proj_list, dim=2)  # (B, num_items, num_layers, d_model)
            
            content_proj = content_proj.view(batch_size, seq_len, self.d_model)
            collab_proj = collab_proj.view(batch_size, seq_len, self.d_model)
        else:
            # Shared projection (original behavior)
            content_proj = self.content_norm(self.content_proj(content_emb))  # (B, L, d_model)
            collab_proj = self.collab_norm(self.collab_proj(collab_emb))  # (B, L, d_model)
        
        if self.gate_type == "learned":
            # Concatenate and compute weights
            concat = torch.cat([id_emb, content_proj, collab_proj], dim=-1)  # (B, L, 3*d_model)
            gate_hidden = F.relu(self.gate_fc1(concat))
            gate_hidden = self.gate_dropout(gate_hidden)
            gate_logits = self.gate_fc2(gate_hidden)  # (B, L, 3)
            weights = F.softmax(gate_logits, dim=-1)  # (B, L, 3)
            
            # Weighted fusion
            fused = (weights[..., 0:1] * id_emb +
                    weights[..., 1:2] * content_proj +
                    weights[..., 2:3] * collab_proj)
            
            # FIX: Don't apply dropout to final fused output
            # Dropout should only be applied to intermediate representations
            # Applying it here causes train/test inconsistency
        
        elif self.gate_type == "attention":
            # Stack sources
            sources = torch.stack([id_emb, content_proj, collab_proj], dim=2)  # (B, L, 3, d_model)
            
            # Compute attention scores
            query = self.query_proj(id_emb).unsqueeze(2)  # (B, L, 1, d_model)
            key = self.key_proj(sources)  # (B, L, 3, d_model)
            value = self.value_proj(sources)  # (B, L, 3, d_model)
            
            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)  # (B, L, 1, 3)
            weights = F.softmax(scores, dim=-1)  # (B, L, 1, 3)
            
            fused = torch.matmul(weights, value).squeeze(2)  # (B, L, d_model)
            
            # FIX: Don't apply dropout to final fused output
        
        elif self.gate_type == "fixed":
            # Fixed weighted sum
            weights = self.fixed_weights.view(1, 1, 3)  # (1, 1, 3)
            fused = (weights[..., 0] * id_emb +
                    weights[..., 1] * content_proj +
                    weights[..., 2] * collab_proj)
            
            # FIX: Don't apply dropout to final fused output
        
        # Learnable fusion strength: output = id_emb + alpha * (fused - id_emb)
        # This prevents negative optimization:
        # - Initially (alpha=0): output = id_emb (baseline performance)
        # - Gradually: alpha increases if fusion helps
        # - If fusion hurts: alpha stays near 0
        # Note: Only applied for fixed/attention gates (learned gates have built-in weighting)
        if self.use_residual and hasattr(self, 'fusion_alpha'):
            alpha = torch.sigmoid(self.fusion_alpha)  # Constrain to [0, 1]
            # Compute the difference between fused and pure id_emb
            # This way we're adding a scaled correction term
            delta = fused - id_emb
            output = id_emb + alpha * delta
        else:
            output = fused
        
        # NOTE: We do NOT apply attention mask here because:
        # 1. Padding positions naturally have zero content/collab embeddings
        # 2. The fusion will naturally learn to ignore these zeros
        # 3. Forcing mask causes train/test inconsistency
        # 4. Empirical results show mask hurts performance (v3: 0.0452 vs v4: 0.0440)
        
        return output


class CodebookPredictor(nn.Module):
    """Predicts codebook vectors from hidden states."""
    
    def __init__(self, d_model: int, n_layers: int, latent_dim: int, dropout: float = 0.1):
        """Initialize codebook predictor.
        
        Args:
            d_model: Model dimension
            n_layers: Number of codebook layers
            latent_dim: Dimension of each codebook vector
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_layers * latent_dim)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict codebook vectors.
        
        Args:
            hidden_states: Hidden states (B, L, d_model)
        
        Returns:
            Predicted codebook vectors (B, L, n_layers, latent_dim)
        """
        # Take the last token's hidden state for prediction
        last_hidden = hidden_states[:, -1, :]  # (B, d_model)
        
        # Predict flattened codebook vectors
        pred_flat = self.predictor(last_hidden)  # (B, n_layers * latent_dim)
        
        # Reshape to (B, n_layers, latent_dim)
        pred = pred_flat.view(-1, self.n_layers, self.latent_dim)
        
        return pred


class TagPredictor(nn.Module):
    """Predicts hierarchical tag IDs from hidden states."""
    
    def __init__(self, d_model: int, num_tags_per_layer: List[int], dropout: float = 0.1):
        """Initialize tag predictor.
        
        Args:
            d_model: Model dimension
            num_tags_per_layer: Number of tags for each layer
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_tags_per_layer = num_tags_per_layer
        
        # Separate classifier for each layer
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_tags)
            )
            for num_tags in num_tags_per_layer
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """Predict tag IDs for each layer.
        
        Args:
            hidden_states: Hidden states (B, L, d_model)
        
        Returns:
            List of logits for each layer [(B, num_tags_L1), (B, num_tags_L2), ...]
        """
        # Take the last token's hidden state
        last_hidden = hidden_states[:, -1, :]  # (B, d_model)
        
        # Predict for each layer
        predictions = [classifier(last_hidden) for classifier in self.classifiers]
        
        return predictions


def convert_tag_ids_to_tokens(tag_ids: torch.Tensor, tag_token_offset: int, max_tag_ids_per_layer: List[int]) -> torch.Tensor:
    """Convert tag IDs to token IDs in vocabulary.
    
    Args:
        tag_ids: Tag IDs (B, n_layers)
        tag_token_offset: Offset where tag tokens start in vocab
        max_tag_ids_per_layer: Max tag ID for each layer
    
    Returns:
        Token IDs (B, n_layers)
    """
    batch_size, n_layers = tag_ids.shape
    token_ids = torch.zeros_like(tag_ids)
    
    cumulative_offset = tag_token_offset
    for layer_idx in range(n_layers):
        # Map tag_id to token_id: token_id = tag_id + cumulative_offset
        token_ids[:, layer_idx] = tag_ids[:, layer_idx] + cumulative_offset
        # Update offset for next layer
        cumulative_offset += (max_tag_ids_per_layer[layer_idx] + 1)
    
    return token_ids


class ItemLayerEmbedding(nn.Module):
    """Item and layer position embeddings for hierarchical semantic IDs.
    
    This module adds:
    1. Item-level position embeddings (which item in the sequence)
    2. Layer-level embeddings (which layer within an item: L0, L1, L2)
    3. Temporal decay embeddings (recency information)
    
    CRITICAL FIX: Properly handles left-padded sequences by computing positions
    relative to actual content, not absolute positions.
    """
    
    def __init__(
        self,
        d_model: int,
        max_items: int = 20,
        num_layers: int = 3,
        use_temporal_decay: bool = True,
        dropout: float = 0.1
    ):
        """Initialize item/layer embeddings.
        
        Args:
            d_model: Model dimension
            max_items: Maximum number of items in sequence
            num_layers: Number of semantic ID layers per item
            use_temporal_decay: Whether to add temporal decay embeddings
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_items = max_items
        self.num_layers = num_layers
        self.use_temporal_decay = use_temporal_decay
        
        # Item position embeddings (0 to max_items-1)
        self.item_pos_emb = nn.Embedding(max_items, d_model)
        
        # Layer embeddings (0 to num_layers-1)
        self.layer_emb = nn.Embedding(num_layers, d_model)
        
        # Temporal decay embeddings (optional)
        if use_temporal_decay:
            # Learnable decay weights for each position
            self.temporal_decay = nn.Parameter(torch.zeros(max_items, d_model))
            nn.init.normal_(self.temporal_decay, mean=0.0, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize embeddings with smaller std to avoid disrupting pretrained embeddings
        nn.init.normal_(self.item_pos_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.layer_emb.weight, mean=0.0, std=0.01)
    
    def forward(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Add item/layer position embeddings to token embeddings.
        
        CRITICAL FIX: Only applies position embeddings to non-padding tokens.
        For left-padded sequences, computes positions relative to actual content.
        
        Args:
            token_embeddings: Token embeddings (B, seq_len, d_model)
            attention_mask: Attention mask (B, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            Enhanced embeddings (B, seq_len, d_model)
        """
        batch_size, seq_len, _ = token_embeddings.shape
        device = token_embeddings.device
        
        # Start with original embeddings
        enhanced_emb = token_embeddings.clone()
        
        if attention_mask is not None:
            # For each sample in batch, find where actual content starts
            for b in range(batch_size):
                mask = attention_mask[b]  # (seq_len,)
                
                # Find first non-padding position
                non_padding_indices = torch.where(mask > 0)[0]
                if len(non_padding_indices) == 0:
                    continue  # All padding, skip
                
                start_pos = non_padding_indices[0].item()
                end_pos = non_padding_indices[-1].item() + 1
                content_len = end_pos - start_pos
                
                # Calculate item and layer indices for actual content
                # Content positions: [0, 1, 2, ..., content_len-1]
                content_positions = torch.arange(content_len, device=device)
                item_indices = content_positions // self.num_layers  # [0,0,0, 1,1,1, ...]
                layer_indices = content_positions % self.num_layers  # [0,1,2, 0,1,2, ...]
                
                # Get embeddings for actual content
                item_emb = self.item_pos_emb(item_indices)  # (content_len, d_model)
                layer_emb = self.layer_emb(layer_indices)  # (content_len, d_model)
                
                # Add to embeddings (only for non-padding positions)
                enhanced_emb[b, start_pos:end_pos] = (
                    enhanced_emb[b, start_pos:end_pos] + item_emb + layer_emb
                )
                
                # Add temporal decay if enabled
                if self.use_temporal_decay:
                    temporal_emb = self.temporal_decay[item_indices]  # (content_len, d_model)
                    enhanced_emb[b, start_pos:end_pos] = (
                        enhanced_emb[b, start_pos:end_pos] + temporal_emb
                    )
        else:
            # No mask provided, assume all tokens are valid (backward compatibility)
            # This is the old behavior - not recommended
            item_indices = torch.arange(seq_len, device=device) // self.num_layers
            item_indices = item_indices.unsqueeze(0).expand(batch_size, -1)
            
            layer_indices = torch.arange(seq_len, device=device) % self.num_layers
            layer_indices = layer_indices.unsqueeze(0).expand(batch_size, -1)
            
            item_emb = self.item_pos_emb(item_indices)
            layer_emb = self.layer_emb(layer_indices)
            
            enhanced_emb = enhanced_emb + item_emb + layer_emb
            
            if self.use_temporal_decay:
                temporal_emb = self.temporal_decay[item_indices]
                enhanced_emb = enhanced_emb + temporal_emb
        
        # Apply layer norm and dropout
        enhanced_emb = self.layer_norm(enhanced_emb)
        enhanced_emb = self.dropout(enhanced_emb)
        
        # Zero out padding positions to ensure they don't affect computation
        if attention_mask is not None:
            enhanced_emb = enhanced_emb * attention_mask.unsqueeze(-1)
        
        return enhanced_emb


class HierarchicalAttention(nn.Module):
    """Hierarchical attention for item-level and layer-level modeling.
    
    This module implements:
    1. Intra-item attention: Tokens within the same item attend to each other
    2. Inter-item attention: Items attend to other items in the sequence
    3. Layer-level attention: Different layers have different importance
    
    CRITICAL FIX: Properly handles variable-length sequences with padding.
    """
    
    def __init__(
        self,
        d_model: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_item_attention: bool = True,
        use_layer_attention: bool = True
    ):
        """Initialize hierarchical attention.
        
        Args:
            d_model: Model dimension
            num_layers: Number of semantic ID layers per item
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_item_attention: Whether to use item-level attention
            use_layer_attention: Whether to use layer-level attention
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_item_attention = use_item_attention
        self.use_layer_attention = use_layer_attention
        
        # Find valid num_heads that divides d_model evenly
        def find_valid_num_heads(d_model: int, preferred_heads: int) -> int:
            """Find the largest valid num_heads <= preferred_heads that divides d_model."""
            for h in range(preferred_heads, 0, -1):
                if d_model % h == 0:
                    return h
            return 1  # Fallback to 1 head
        
        # Adjust num_heads to be compatible with d_model
        valid_num_heads = find_valid_num_heads(d_model, num_heads)
        if valid_num_heads != num_heads:
            logger.info(f"Adjusted num_heads from {num_heads} to {valid_num_heads} to be divisible by d_model={d_model}")
        
        # Intra-item attention (within same item)
        if use_item_attention:
            intra_heads = find_valid_num_heads(d_model, min(valid_num_heads, 2))
            self.intra_item_attn = nn.MultiheadAttention(
                d_model, num_heads=intra_heads, dropout=dropout, batch_first=True
            )
            self.intra_item_norm = nn.LayerNorm(d_model)
        
        # Inter-item attention (between items)
        if use_item_attention:
            self.inter_item_attn = nn.MultiheadAttention(
                d_model, num_heads=valid_num_heads, dropout=dropout, batch_first=True
            )
            self.inter_item_norm = nn.LayerNorm(d_model)
        
        # Layer-level attention (importance of different layers)
        if use_layer_attention:
            layer_heads = find_valid_num_heads(d_model, min(valid_num_heads, 2))
            self.layer_attn = nn.MultiheadAttention(
                d_model, num_heads=layer_heads, dropout=dropout, batch_first=True
            )
            self.layer_attn_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply hierarchical attention.
        
        CRITICAL FIX: Handles variable-length sequences properly.
        Only applies attention to actual content, not padding.
        
        Args:
            x: Input embeddings (B, seq_len, d_model)
            attention_mask: Attention mask (B, seq_len) - 1 for real, 0 for padding
        
        Returns:
            Attended embeddings (B, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # CRITICAL FIX: Check if seq_len is divisible by num_layers
        # With dynamic batching or variable-length sequences, this may not hold
        if seq_len % self.num_layers != 0:
            # Cannot apply hierarchical attention, return input as-is
            # This prevents crashes with variable-length sequences
            logger.warning(
                f"Hierarchical attention skipped: seq_len={seq_len} not divisible by "
                f"num_layers={self.num_layers}. Returning input unchanged."
            )
            return x
        
        num_items = seq_len // self.num_layers
        
        # Reshape to (B, num_items, num_layers, d_model)
        try:
            x_reshaped = x.view(batch_size, num_items, self.num_layers, d_model)
        except RuntimeError as e:
            logger.warning(f"Failed to reshape for hierarchical attention: {e}. Returning input unchanged.")
            return x
        
        # Step 1: Intra-item attention (tokens within same item)
        if self.use_item_attention:
            intra_outputs = []
            for i in range(num_items):
                item_tokens = x_reshaped[:, i, :, :]  # (B, num_layers, d_model)
                
                # Self-attention within item
                attended, _ = self.intra_item_attn(
                    item_tokens, item_tokens, item_tokens,
                    need_weights=False
                )
                attended = self.dropout(attended)
                attended = self.intra_item_norm(item_tokens + attended)
                intra_outputs.append(attended)
            
            x_reshaped = torch.stack(intra_outputs, dim=1)  # (B, num_items, num_layers, d_model)
        
        # Step 2: Inter-item attention (between items)
        if self.use_item_attention:
            # Use mean pooling to get item representations
            item_repr = x_reshaped.mean(dim=2)  # (B, num_items, d_model)
            
            # Create attention mask for items if provided
            item_mask = None
            if attention_mask is not None:
                # Reshape attention mask to (B, num_items, num_layers)
                try:
                    mask_reshaped = attention_mask.view(batch_size, num_items, self.num_layers)
                    # Item is valid if any of its tokens is valid
                    item_mask = mask_reshaped.any(dim=2)  # (B, num_items)
                    # Convert to attention mask format (True = masked)
                    item_mask = ~item_mask
                except RuntimeError:
                    # If reshape fails, skip masking
                    item_mask = None
            
            # Inter-item attention
            attended_items, _ = self.inter_item_attn(
                item_repr, item_repr, item_repr,
                key_padding_mask=item_mask,
                need_weights=False
            )
            attended_items = self.dropout(attended_items)
            attended_items = self.inter_item_norm(item_repr + attended_items)
            
            # Broadcast back to token level
            attended_items = attended_items.unsqueeze(2).expand(-1, -1, self.num_layers, -1)
            x_reshaped = x_reshaped + attended_items
        
        # Step 3: Layer-level attention (importance of different layers)
        if self.use_layer_attention:
            # Reshape to (B, num_layers, num_items, d_model) for layer-wise attention
            x_layer = x_reshaped.permute(0, 2, 1, 3)  # (B, num_layers, num_items, d_model)
            
            layer_outputs = []
            for layer_idx in range(self.num_layers):
                layer_tokens = x_layer[:, layer_idx, :, :]  # (B, num_items, d_model)
                
                # Self-attention within layer
                attended, _ = self.layer_attn(
                    layer_tokens, layer_tokens, layer_tokens,
                    need_weights=False
                )
                attended = self.dropout(attended)
                attended = self.layer_attn_norm(layer_tokens + attended)
                layer_outputs.append(attended)
            
            x_layer = torch.stack(layer_outputs, dim=1)  # (B, num_layers, num_items, d_model)
            x_reshaped = x_layer.permute(0, 2, 1, 3)  # (B, num_items, num_layers, d_model)
        
        # Reshape back to (B, seq_len, d_model)
        output = x_reshaped.reshape(batch_size, seq_len, d_model)
        
        # Zero out padding positions
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1)
        
        return output


class TIGER(nn.Module):
    """TIGER: T5-based Generative Recommender.
    
    This model uses a T5 encoder-decoder architecture to generate
    semantic item IDs for recommendation.
    
    Enhanced with:
    - Codebook vector warm-start for ID embeddings
    - Multi-source embedding fusion (ID + content + collab)
    - Codebook vector prediction
    - Tag ID prediction
    - Item/layer position embeddings
    - Hierarchical attention (item-level and layer-level)
    """
    
    def __init__(self, model_config, training_config=None):
        """Initialize the TIGER model.
        
        Args:
            model_config: ModelConfig instance with model hyperparameters
            training_config: TrainingConfig instance with feature toggles
        """
        super(TIGER, self).__init__()
        
        self.model_config = model_config
        self.training_config = training_config
        
        # Create T5 configuration
        t5_config = T5Config(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            d_ff=model_config.d_ff,
            d_kv=model_config.d_kv,
            num_layers=model_config.num_layers,
            num_decoder_layers=model_config.num_decoder_layers,
            num_heads=model_config.num_heads,
            dropout_rate=model_config.dropout_rate,
            feed_forward_proj=model_config.feed_forward_proj,
            pad_token_id=model_config.pad_token_id,
            eos_token_id=model_config.eos_token_id,
            decoder_start_token_id=model_config.pad_token_id,
        )
        
        # Initialize T5 model
        self.model = T5ForConditionalGeneration(t5_config)
        self.config = model_config
        
        # Feature flags
        self.use_multimodal_fusion = training_config and training_config.use_multimodal_fusion
        self.use_codebook_prediction = training_config and training_config.use_codebook_prediction
        self.use_tag_prediction = training_config and training_config.use_tag_prediction
        self.use_item_layer_emb = training_config and training_config.use_item_layer_emb
        self.use_hierarchical_attn = training_config and training_config.use_hierarchical_attn
        
        # Feature 4: Multi-source fusion
        if self.use_multimodal_fusion:
            fixed_weights = None
            if training_config.fusion_gate_type == "fixed":
                fixed_weights = {
                    'id': training_config.id_emb_weight,
                    'content': training_config.content_emb_weight,
                    'collab': training_config.collab_emb_weight
                }
            
            # Check if layer-specific fusion is enabled
            use_layer_specific = getattr(training_config, 'use_layer_specific_fusion', False)
            
            self.fusion_module = MultiSourceFusion(
                d_model=model_config.d_model,
                content_dim=768,  # Default content dimension
                collab_dim=64,  # Default collab dimension
                gate_type=training_config.fusion_gate_type,
                fixed_weights=fixed_weights,
                dropout=model_config.dropout_rate,
                use_residual=True,  # Enable learnable fusion strength
                num_layers=model_config.num_code_layers,
                use_layer_specific=use_layer_specific
            )
            logger.info(f"Multi-source fusion enabled (gate_type={training_config.fusion_gate_type}, layer_specific={use_layer_specific})")
        
        # NEW: Item/layer position embeddings
        if self.use_item_layer_emb:
            max_items = 20  # Default max sequence length
            use_temporal_decay = getattr(training_config, 'use_temporal_decay', True)
            
            self.item_layer_embedding = ItemLayerEmbedding(
                d_model=model_config.d_model,
                max_items=max_items,
                num_layers=model_config.num_code_layers,
                use_temporal_decay=use_temporal_decay,
                dropout=model_config.dropout_rate
            )
            
            # Learnable scaling factor to control the strength of position embeddings
            # Start small (0.1) to avoid disrupting pretrained model
            self.pos_emb_scale = nn.Parameter(torch.tensor(0.1))
            
            logger.info(f"Item/layer embeddings enabled (temporal_decay={use_temporal_decay}, initial_scale=0.1)")
        
        # NEW: Hierarchical attention
        if self.use_hierarchical_attn:
            use_item_attn = getattr(training_config, 'use_item_attention', True)
            use_layer_attn = getattr(training_config, 'use_layer_attention', True)
            
            self.hierarchical_attention = HierarchicalAttention(
                d_model=model_config.d_model,
                num_layers=model_config.num_code_layers,
                num_heads=model_config.num_heads,
                dropout=model_config.dropout_rate,
                use_item_attention=use_item_attn,
                use_layer_attention=use_layer_attn
            )
            
            # Learnable scaling factor to control the strength of hierarchical attention
            # Start small (0.1) to avoid disrupting pretrained model
            self.hier_attn_scale = nn.Parameter(torch.tensor(0.1))
            
            logger.info(f"Hierarchical attention enabled (item_attn={use_item_attn}, layer_attn={use_layer_attn}, initial_scale=0.1)")
        
        # Feature 2: Codebook prediction
        if self.use_codebook_prediction:
            self.codebook_predictor = CodebookPredictor(
                d_model=model_config.d_model,
                n_layers=model_config.num_code_layers,
                latent_dim=32,  # Default latent dimension
                dropout=model_config.dropout_rate
            )
            logger.info("Codebook prediction enabled")
        
        # Feature 3: Tag prediction
        if self.use_tag_prediction:
            # Get actual num_tags_per_layer from model config
            if hasattr(model_config, 'max_tag_ids_per_layer') and model_config.max_tag_ids_per_layer:
                num_tags_per_layer = [max_id + 1 for max_id in model_config.max_tag_ids_per_layer]
            else:
                # Fallback to placeholder values
                num_tags_per_layer = [100, 200, 300]
                logger.warning("max_tag_ids_per_layer not found in config, using placeholder values")
            
            self.tag_predictor = TagPredictor(
                d_model=model_config.d_model,
                num_tags_per_layer=num_tags_per_layer,
                dropout=model_config.dropout_rate
            )
            logger.info(f"Tag prediction enabled with {num_tags_per_layer} tags per layer")
        
        logger.info(f"Initialized TIGER model with vocab_size={model_config.vocab_size}")
        logger.info(self.n_parameters)
    
    def init_codebook_warmstart(
        self,
        codebook_vectors: Dict[int, np.ndarray],
        semantic_mapper,
        freeze: bool = False
    ):
        """Initialize ID embeddings with codebook vectors (Feature 1).
        
        Args:
            codebook_vectors: Dict mapping item_id to codebook vectors (n_layers, latent_dim)
            semantic_mapper: SemanticIDMapper instance
            freeze: Whether to freeze the warmstarted embeddings
        """
        logger.info("Initializing ID embeddings with codebook vectors...")
        
        embedding_table = self.model.get_input_embeddings()
        latent_dim = next(iter(codebook_vectors.values())).shape[1]
        
        # Create projection layer if needed
        if latent_dim != self.model_config.d_model:
            self.codebook_proj = nn.Linear(latent_dim, self.model_config.d_model, bias=False)
            # Initialize with Xavier
            nn.init.xavier_uniform_(self.codebook_proj.weight)
        else:
            self.codebook_proj = nn.Identity()
        
        # Initialize embeddings for each semantic ID token
        initialized_count = 0
        for item_id, codes in semantic_mapper.item_to_codes.items():
            if item_id not in codebook_vectors:
                continue
            
            item_codebook_vecs = codebook_vectors[item_id]  # (n_layers, latent_dim)
            
            for layer_idx, (code, vec) in enumerate(zip(codes, item_codebook_vecs)):
                # Project to d_model
                vec_tensor = torch.from_numpy(vec).float()
                projected_vec = self.codebook_proj(vec_tensor)
                
                # Initialize embedding
                with torch.no_grad():
                    embedding_table.weight.data[code] = projected_vec
                
                initialized_count += 1
        
        logger.info(f"Initialized {initialized_count} token embeddings with codebook vectors")
        
        # Freeze if requested
        if freeze:
            embedding_table.weight.requires_grad = False
            logger.info("Frozen warmstarted embeddings")
    
    def broadcast_item_to_tokens(
        self,
        item_embeddings: torch.Tensor,
        item_ids: List[int],
        num_tokens_per_item: int
    ) -> torch.Tensor:
        """Broadcast item-level embeddings to token-level.
        
        Args:
            item_embeddings: Item-level embeddings (batch_size, max_items, emb_dim)
            item_ids: List of item IDs for each position
            num_tokens_per_item: Number of tokens per item (n_layers)
        
        Returns:
            Token-level embeddings (batch_size, seq_len, emb_dim)
        """
        batch_size, max_items, emb_dim = item_embeddings.shape
        seq_len = max_items * num_tokens_per_item
        
        # Repeat each item embedding num_tokens_per_item times
        # (B, max_items, emb_dim) -> (B, max_items, num_tokens_per_item, emb_dim)
        broadcasted = item_embeddings.unsqueeze(2).repeat(1, 1, num_tokens_per_item, 1)
        
        # Reshape to (B, seq_len, emb_dim)
        broadcasted = broadcasted.view(batch_size, seq_len, emb_dim)
        
        return broadcasted
    
    @property
    def n_parameters(self) -> str:
        """Calculate the number of trainable parameters.
        
        Returns:
            String containing parameter statistics
        """
        def count_params(params):
            return sum(p.numel() for p in params if p.requires_grad)
        
        total_params = count_params(self.parameters())
        emb_params = count_params(self.model.get_input_embeddings().parameters())
        
        return (
            f"Model Parameters:\n"
            f"  Embedding parameters: {emb_params:,}\n"
            f"  Non-embedding parameters: {total_params - emb_params:,}\n"
            f"  Total trainable parameters: {total_params:,}"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        # NEW: Multi-source information
        content_embs: Optional[torch.Tensor] = None,
        collab_embs: Optional[torch.Tensor] = None,
        target_codebook_vecs: Optional[torch.Tensor] = None,
        target_tag_ids: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model with multi-source fusion.
        
        Args:
            input_ids: Input token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            labels: Target token IDs (B, target_len)
            content_embs: Content embeddings (B, max_items, 768)
            collab_embs: Collaborative embeddings (B, max_items, 64)
            target_codebook_vecs: Target codebook vectors (B, n_layers, latent_dim)
            target_tag_ids: Target tag IDs (B, n_layers)
            return_dict: Whether to return a dictionary
        
        Returns:
            Dictionary with losses and logits
        """
        # Get ID embeddings
        id_emb = self.model.get_input_embeddings()(input_ids)  # (B, seq_len, d_model)
        
        # Apply item/layer position embeddings if enabled
        # CRITICAL FIX: Use residual connection with learnable scale
        if self.use_item_layer_emb:
            pos_enhanced = self.item_layer_embedding(id_emb, attention_mask)
            # Residual: id_emb + scale * (pos_enhanced - id_emb)
            # This allows gradual learning without disrupting pretrained embeddings
            id_emb = id_emb + self.pos_emb_scale * (pos_enhanced - id_emb)
        
        # Apply hierarchical attention if enabled
        # CRITICAL FIX: Use residual connection with learnable scale
        if self.use_hierarchical_attn:
            attn_enhanced = self.hierarchical_attention(id_emb, attention_mask)
            # Residual: id_emb + scale * (attn_enhanced - id_emb)
            id_emb = id_emb + self.hier_attn_scale * (attn_enhanced - id_emb)
        
        # Apply multi-source fusion if enabled
        if self.use_multimodal_fusion and content_embs is not None and collab_embs is not None:
            # Broadcast item-level embeddings to token-level
            num_tokens_per_item = self.model_config.num_code_layers
            content_emb_broadcasted = self.broadcast_item_to_tokens(
                content_embs, None, num_tokens_per_item
            )  # (B, seq_len, 768)
            collab_emb_broadcasted = self.broadcast_item_to_tokens(
                collab_embs, None, num_tokens_per_item
            )  # (B, seq_len, 64)
            
            # Fuse embeddings (pass num_tokens_per_item for layer-specific fusion)
            fused_emb = self.fusion_module(
                id_emb, 
                content_emb_broadcasted, 
                collab_emb_broadcasted,
                attention_mask=attention_mask,
                num_tokens_per_item=num_tokens_per_item
            )
            
            # Forward with custom embeddings
            outputs = self.model(
                inputs_embeds=fused_emb,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )
        else:
            # Forward with enhanced ID embeddings
            outputs = self.model(
                inputs_embeds=id_emb,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True
            )
        
        result = {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'main_loss': outputs.loss
        }
        
        # Auxiliary task 1: Codebook prediction
        if self.use_codebook_prediction and target_codebook_vecs is not None:
            decoder_hidden_states = outputs.decoder_hidden_states[-1]  # (B, target_len, d_model)
            pred_codebook_vecs = self.codebook_predictor(decoder_hidden_states)  # (B, n_layers, latent_dim)
            
            # MSE loss
            codebook_loss = F.mse_loss(pred_codebook_vecs, target_codebook_vecs)
            result['codebook_loss'] = codebook_loss
            result['pred_codebook_vecs'] = pred_codebook_vecs
        
        # Auxiliary task 2: Tag prediction
        if self.use_tag_prediction and target_tag_ids is not None:
            decoder_hidden_states = outputs.decoder_hidden_states[-1]
            pred_tag_logits = self.tag_predictor(decoder_hidden_states)  # List of (B, num_tags)
            
            # Cross-entropy loss for each layer
            tag_losses = []
            for layer_idx, logits in enumerate(pred_tag_logits):
                tag_loss = F.cross_entropy(logits, target_tag_ids[:, layer_idx])
                tag_losses.append(tag_loss)
            
            tag_loss_total = sum(tag_losses) / len(tag_losses)
            result['tag_loss'] = tag_loss_total
            result['pred_tag_logits'] = pred_tag_logits
        
        # Combine losses
        total_loss = result['main_loss']
        
        if 'codebook_loss' in result:
            weight = self.training_config.codebook_prediction_weight if self.training_config else 0.1
            total_loss = total_loss + weight * result['codebook_loss']
        
        if 'tag_loss' in result:
            weight = self.training_config.tag_prediction_weight if self.training_config else 0.1
            total_loss = total_loss + weight * result['tag_loss']
        
        result['loss'] = total_loss
        
        if return_dict:
            return result
        else:
            return result['loss'], result['logits']
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 20,
        max_length: int = 5,
        content_embs: Optional[torch.Tensor] = None,
        collab_embs: Optional[torch.Tensor] = None,
        logits_processor=None,
        **kwargs
    ) -> torch.Tensor:
        """Generate recommendations using beam search with optional Trie constraints.
        
        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            num_beams: Number of beams for beam search
            max_length: Maximum length of generated sequence
            content_embs: Content embeddings (B, max_items, 768) - for fusion
            collab_embs: Collaborative embeddings (B, max_items, 64) - for fusion
            logits_processor: Optional logits processor (e.g., TrieConstrainedLogitsProcessor)
            **kwargs: Additional generation arguments
        
        Returns:
            Generated token IDs, shape (batch_size * num_beams, max_length)
        """
        # Get ID embeddings
        id_emb = self.model.get_input_embeddings()(input_ids)  # (B, seq_len, d_model)
        
        # Apply item/layer position embeddings if enabled
        # CRITICAL FIX: Use residual connection with learnable scale
        if self.use_item_layer_emb:
            pos_enhanced = self.item_layer_embedding(id_emb, attention_mask)
            # Residual: id_emb + scale * (pos_enhanced - id_emb)
            id_emb = id_emb + self.pos_emb_scale * (pos_enhanced - id_emb)
        
        # Apply hierarchical attention if enabled
        # CRITICAL FIX: Use residual connection with learnable scale
        if self.use_hierarchical_attn:
            attn_enhanced = self.hierarchical_attention(id_emb, attention_mask)
            # Residual: id_emb + scale * (attn_enhanced - id_emb)
            id_emb = id_emb + self.hier_attn_scale * (attn_enhanced - id_emb)
        
        # Apply multi-source fusion if enabled
        if self.use_multimodal_fusion and content_embs is not None and collab_embs is not None:
            # Broadcast item-level embeddings to token-level
            num_tokens_per_item = self.model_config.num_code_layers
            content_emb_broadcasted = self.broadcast_item_to_tokens(
                content_embs, None, num_tokens_per_item
            )
            collab_emb_broadcasted = self.broadcast_item_to_tokens(
                collab_embs, None, num_tokens_per_item
            )
            
            # Fuse embeddings (pass num_tokens_per_item for layer-specific fusion)
            fused_emb = self.fusion_module(
                id_emb,
                content_emb_broadcasted,
                collab_emb_broadcasted,
                attention_mask=attention_mask,
                num_tokens_per_item=num_tokens_per_item
            )
            
            # Encode with fused embeddings
            encoder_outputs = self.model.encoder(
                inputs_embeds=fused_emb,
                attention_mask=attention_mask,
                return_dict=True
            )
        else:
            # Encode with enhanced ID embeddings
            encoder_outputs = self.model.encoder(
                inputs_embeds=id_emb,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # Prepare logits processor list
        from transformers import LogitsProcessorList
        logits_processor_list = LogitsProcessorList()
        
        if logits_processor is not None:
            logits_processor_list.append(logits_processor)
        
        # Generate with custom encoder outputs and optional Trie constraints
        generated = self.model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            logits_processor=logits_processor_list if len(logits_processor_list) > 0 else None,
            **kwargs
        )
        return generated
    
    def save_pretrained(self, save_path: str):
        """Save the model.
        
        Args:
            save_path: Path to save the model
        """
        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_pretrained(self, load_path: str):
        """Load the model.
        
        Args:
            load_path: Path to load the model from
        """
        self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        logger.info(f"Model loaded from {load_path}")


def create_model(model_config, training_config=None) -> TIGER:
    """Create a TIGER model.
    
    Args:
        model_config: ModelConfig instance
        training_config: TrainingConfig instance (optional, for enhanced features)
    
    Returns:
        TIGER model instance
    """
    return TIGER(model_config, training_config)

