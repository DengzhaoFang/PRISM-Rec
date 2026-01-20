"""
PRISM: Hierarchical ID VAE with Multi-Modal Inputs

Main model architecture combining:
1. Multi-modal encoder (content + collaborative embeddings)
2. RQ-VAE quantization with hierarchical codebooks
3. Multi-modal decoders (separate for content and collaborative)
4. Hierarchical classifiers for tag prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from RQ_VAE import RQVAEQuantizer, QuantizeMode
from hierarchical_classifiers import HierarchicalClassifiers


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder with Gated Concatenation Fusion.
    
    Solves two critical problems:
    1. Dominance Problem: Prevents content embedding from dominating due to dimension imbalance
    2. Noise Pollution: Dynamically filters noisy collaborative embeddings (especially for long-tail items)
    
    Architecture:
    1. Learn a dynamic "trust gate" (0-1) to weight collaborative signal
    2. Fuse: fused_emb = Concat(content_emb, gate * collab_emb)
    3. Encode fused_emb (768 + 64 D) -> latent_dim (32D)
    """
    
    def __init__(
        self,
        content_dim: int = 768,
        collab_dim: int = 64,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        use_gated_fusion: bool = True
    ):
        """
        Args:
            content_dim: Dimension of content embeddings (768)
            collab_dim: Dimension of collaborative embeddings (64)
            latent_dim: Dimension of latent space (32)
            hidden_dims: Hidden layer dimensions for main encoder
            use_gated_fusion: Use gated fusion (recommended) vs simple concatenation
        """
        super().__init__()
        
        self.content_dim = content_dim
        self.collab_dim = collab_dim
        self.latent_dim = latent_dim
        self.use_gated_fusion = use_gated_fusion
        
        if use_gated_fusion:
            # === Gated Concatenation Fusion Components ===
            
            # Gate network: learns dynamic "trust score" for collaborative signal
            # Input: original collab_emb (64D) -> Output: gate weights (64D)
            # Simplified architecture for better gradient flow
            self.gate_network = nn.Sequential(
                nn.Linear(collab_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, collab_dim),
                nn.Sigmoid()  # Compress to (0, 1) range
            )
            
            # Fusion normalization: stabilize after concatenation
            self.fusion_norm = nn.LayerNorm(content_dim + collab_dim)
            
            # Main encoder input dimension is sum of dimensions
            input_dim = content_dim + collab_dim
        else:
            # Fallback: simple concatenation
            input_dim = content_dim + collab_dim
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        # Build main encoder network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final projection to latent space
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(
        self, 
        content_emb: torch.Tensor, 
        collab_emb: torch.Tensor,
        return_weighted_collab: bool = False
    ) -> torch.Tensor:
        """
        Encode multi-modal inputs to latent space with gated fusion.
        
        Args:
            content_emb: Content embeddings (batch_size, 768)
            collab_emb: Collaborative embeddings (batch_size, 64)
            return_weighted_collab: If True, also return the gate-weighted collab embedding
                                   for consistent reconstruction target in DHR mode
            
        Returns:
            z: Latent representations (batch_size, latent_dim)
            weighted_collab_emb: (optional) Gate-weighted collab embedding (batch_size, 64)
        """
        if self.use_gated_fusion:
            # === Gated Concatenation Fusion ===
            
            # Step 1: Compute dynamic trust gate
            # For noisy long-tail items, gate -> 0 (ignore collab signal)
            # For high-quality popular items, gate -> 1 (trust collab signal)
            gate = self.gate_network(collab_emb)  # (B, 64), values in (0, 1)
            
            # Step 2: Weighted collaborative embedding (denoised)
            weighted_collab_emb = gate * collab_emb  # (B, 64)
            
            # Step 3: Concatenate
            fused_emb = torch.cat([content_emb, weighted_collab_emb], dim=-1)  # (B, 768+64)
            
            # Step 4: Normalize fused embedding
            fused_emb = self.fusion_norm(fused_emb)
            
            # Step 5: Encode to latent space
            z = self.encoder(fused_emb)  # (B, latent_dim)
            
            if return_weighted_collab:
                return z, weighted_collab_emb
        else:
            # Fallback: simple concatenation
            x = torch.cat([content_emb, collab_emb], dim=-1)  # (B, 832)
            z = self.encoder(x)  # (B, latent_dim)
            
            if return_weighted_collab:
                # No gating, return original collab
                return z, collab_emb
        
        return z
    
    def get_gate_statistics(self, collab_emb: torch.Tensor) -> Dict[str, float]:
        """
        Get gate statistics for analysis.
        
        Args:
            collab_emb: Collaborative embeddings (batch_size, 64)
            
        Returns:
            stats: Dictionary with gate statistics
        """
        if not self.use_gated_fusion:
            return {}
        
        with torch.no_grad():
            gate = self.gate_network(collab_emb)  # (B, 768)
            mean_gate = gate.mean(dim=1)  # (B,) - average trust per item
            
            stats = {
                'mean': mean_gate.mean().item(),
                'std': mean_gate.std().item(),
                'min': mean_gate.min().item(),
                'max': mean_gate.max().item(),
                'median': mean_gate.median().item(),
                'q25': mean_gate.quantile(0.25).item(),
                'q75': mean_gate.quantile(0.75).item()
            }
            
            return stats


class MultiModalDecoder(nn.Module):
    """
    Multi-modal decoder with separate heads for content and collaborative embeddings.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        content_dim: int = 768,
        collab_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        use_dual_decoder: bool = True
    ):
        """
        Args:
            latent_dim: Dimension of latent/quantized space (32)
            content_dim: Dimension of content embeddings (768)
            collab_dim: Dimension of collaborative embeddings (64)
            hidden_dims: Shared hidden layer dimensions
            use_dual_decoder: If True, use separate heads for content and collab.
                            If False, use single head to decode concatenated embedding.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.content_dim = content_dim
        self.collab_dim = collab_dim
        self.use_dual_decoder = use_dual_decoder
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [128, 256, 512]
        
        # Shared decoder backbone
        shared_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_decoder = nn.Sequential(*shared_layers)
        
        if use_dual_decoder:
            # Separate heads for content and collaborative
            # DHR Design: Both heads use symmetric 2-layer structure (same depth)
            # Width differs based on output dimension (Task-specific Projection Heads)
            self.content_head = nn.Sequential(
                nn.Linear(prev_dim, content_dim * 2),
                nn.LayerNorm(content_dim * 2),
                nn.ReLU(),
                nn.Linear(content_dim * 2, content_dim)
            )
            
            # Collab head: same 2-layer depth as content head
            # Use 256 as intermediate dim (4x collab_dim) for sufficient expressiveness
            collab_hidden = max(256, collab_dim * 4)
            self.collab_head = nn.Sequential(
                nn.Linear(prev_dim, collab_hidden),
                nn.LayerNorm(collab_hidden),
                nn.ReLU(),
                nn.Linear(collab_hidden, collab_dim)
            )
            self.concat_head = None
        else:
            # Single head for concatenated embedding
            concat_dim = content_dim + collab_dim  # 768 + 64 = 832
            self.concat_head = nn.Sequential(
                nn.Linear(prev_dim, concat_dim * 2),
                nn.LayerNorm(concat_dim * 2),
                nn.ReLU(),
                nn.Linear(concat_dim * 2, concat_dim)
            )
            self.content_head = None
            self.collab_head = None
    
    def forward(
        self, 
        z_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode quantized latents to content and collaborative embeddings.
        
        Args:
            z_q: Quantized latents (batch_size, latent_dim)
            
        Returns:
            content_recon: Reconstructed content embeddings (batch_size, 768)
            collab_recon: Reconstructed collaborative embeddings (batch_size, 64)
        """
        # Shared decoding
        shared_features = self.shared_decoder(z_q)  # (B, hidden_dim)
        
        if self.use_dual_decoder:
            # Separate reconstruction heads
            content_recon = self.content_head(shared_features)  # (B, 768)
            collab_recon = self.collab_head(shared_features)  # (B, 64)
        else:
            # Single head for concatenated embedding
            concat_recon = self.concat_head(shared_features)  # (B, 832)
            # Split into content and collab
            content_recon = concat_recon[:, :self.content_dim]  # (B, 768)
            collab_recon = concat_recon[:, self.content_dim:]  # (B, 64)
        
        return content_recon, collab_recon


class PRISM(nn.Module):
    """
    Hierarchical ID VAE with multi-modal inputs and tag-guided learning.
    
    Architecture:
    1. Multi-modal encoder with Gated Additive Fusion:
       - Projects collab (64D) -> (768D)
       - Learns dynamic gate to filter noisy collaborative signals
       - Fuses: fused = content + gate * proj_collab
       - Encodes: fused (768D) -> z (32D)
    2. RQ-VAE: z -> z_q (hierarchical quantization)
    3. Multi-modal decoder: z_q -> (content_recon, collab_recon)
    4. Hierarchical classifiers: quantized codes -> tag predictions
    
    Key Innovation:
    - Gated fusion solves dimension imbalance and noise pollution
    - Automatically adapts to item popularity (long-tail vs popular)
    """
    
    def __init__(
        self,
        # Input/output dimensions
        content_dim: int = 768,
        collab_dim: int = 64,
        latent_dim: int = 32,
        # RQ-VAE parameters
        n_layers: int = 3,
        n_embed: int = 256,  # Default uniform size (can be overridden by n_embed_per_layer)
        n_embed_per_layer: Optional[List[int]] = None,  # Variable codebook sizes per layer
        # Encoder/decoder architecture
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        use_gated_fusion: bool = True,  # Use gated additive fusion (recommended)
        use_dual_decoder: bool = True,  # Use dual decoder heads (recommended)
        # Classification parameters
        num_classes_per_layer: Optional[List[int]] = None,
        # Quantization parameters
        use_ema: bool = True,
        ema_decay: float = 0.99,
        beta: float = 0.25,
        quantize_mode: QuantizeMode = QuantizeMode.ROTATION,
        # Other parameters
        dropout: float = 0.1
    ):
        """
        Initialize PRISM.
        
        Args:
            content_dim: Content embedding dimension
            collab_dim: Collaborative embedding dimension
            latent_dim: Latent/codebook dimension
            n_layers: Number of RQ layers
            n_embed: Default codebook size per layer (used if n_embed_per_layer is None)
            n_embed_per_layer: Variable codebook sizes per layer [size_L1, size_L2, size_L3]
                              If None, uses n_embed for all layers
            encoder_hidden_dims: Hidden dimensions for encoder
            decoder_hidden_dims: Hidden dimensions for decoder
            use_gated_fusion: Use gated additive fusion for multi-modal encoding (recommended)
            use_dual_decoder: If True, use separate decoder heads for content and collab.
                            If False, use single decoder head for concatenated embedding.
            num_classes_per_layer: Number of tag classes per layer [n_L2, n_L3, n_L4]
            use_ema: Use EMA for codebook updates
            ema_decay: EMA decay rate
            beta: Commitment loss weight
            quantize_mode: Quantization mode (STE, ROTATION, GUMBEL_SOFTMAX)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.content_dim = content_dim
        self.collab_dim = collab_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.use_dual_decoder = use_dual_decoder
        
        # Setup variable codebook sizes
        if n_embed_per_layer is None:
            self.n_embed_per_layer = [n_embed] * n_layers
        else:
            assert len(n_embed_per_layer) == n_layers, \
                f"n_embed_per_layer must have {n_layers} elements, got {len(n_embed_per_layer)}"
            self.n_embed_per_layer = n_embed_per_layer
        
        self.n_embed = n_embed  # Keep for backward compatibility
        self.beta = beta
        
        # Multi-modal encoder with gated fusion
        self.encoder = MultiModalEncoder(
            content_dim=content_dim,
            collab_dim=collab_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            use_gated_fusion=use_gated_fusion
        )
        
        # RQ-VAE quantizers (one per layer with variable codebook sizes)
        self.quantizers = nn.ModuleList([
            RQVAEQuantizer(
                n_embed=self.n_embed_per_layer[i],
                embed_dim=latent_dim,
                beta=beta,
                use_ema=use_ema,
                decay=ema_decay,
                quantize_mode=quantize_mode
            )
            for i in range(n_layers)
        ])
        
        # Multi-modal decoder
        self.decoder = MultiModalDecoder(
            latent_dim=latent_dim,
            content_dim=content_dim,
            collab_dim=collab_dim,
            hidden_dims=decoder_hidden_dims,
            use_dual_decoder=use_dual_decoder
        )
        
        # Hierarchical classifiers (if num_classes provided)
        if num_classes_per_layer is not None:
            self.classifiers = HierarchicalClassifiers(
                codebook_dim=latent_dim,
                num_classes_per_layer=num_classes_per_layer,
                n_layers=n_layers,
                dropout=dropout
            )
        else:
            self.classifiers = None
    
    def encode(
        self, 
        content_emb: torch.Tensor, 
        collab_emb: torch.Tensor,
        return_weighted_collab: bool = False
    ):
        """Encode inputs to latent space."""
        return self.encoder(content_emb, collab_emb, return_weighted_collab=return_weighted_collab)
    
    def quantize(
        self, 
        z: torch.Tensor,
        temperature: float = 0.2
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], 
               torch.Tensor, List[int]]:
        """
        Hierarchical quantization with RQ-VAE.
        
        Args:
            z: Latent representations (batch_size, latent_dim)
            temperature: Temperature for Gumbel-Softmax
            
        Returns:
            z_q: Quantized latent (sum of all layers)
            quantized_codes: List of quantized codes per layer
            encoding_indices: List of codebook indices per layer
            total_loss: Combined codebook + commitment loss
            perplexities: List of perplexities per layer
        """
        residual = z
        z_q_layers = []
        quantized_codes = []
        encoding_indices = []
        total_codebook_loss = 0.0
        total_commitment_loss = 0.0
        perplexities = []
        
        for layer_idx, quantizer in enumerate(self.quantizers):
            # Quantize residual
            # RQVAEQuantizer returns: (z_q, codebook_loss, commitment_loss, indices, unused_codes)
            z_q_layer, codebook_loss, commitment_loss, indices, unused_codes = quantizer(
                residual, 
                temperature=temperature
            )
            
            # Update residual for next layer
            residual = residual - z_q_layer
            
            z_q_layers.append(z_q_layer)
            quantized_codes.append(z_q_layer)
            encoding_indices.append(indices)
            total_codebook_loss += codebook_loss
            total_commitment_loss += commitment_loss
            
            # Calculate perplexity from unused codes
            # FIX 2.1.A: Use layer-specific codebook size instead of uniform size
            n_embed = self.n_embed_per_layer[layer_idx]
            used_codes = n_embed - unused_codes
            perplexity = used_codes  # Simple perplexity estimate
            perplexities.append(perplexity)
        
        # Sum all quantized layers
        z_q = torch.stack(z_q_layers, dim=0).sum(dim=0)
        
        # Combine losses
        # FIX 2.2.A: Add beta weighting to commitment loss
        total_loss = total_codebook_loss + self.beta * total_commitment_loss
        
        return z_q, quantized_codes, encoding_indices, total_loss, perplexities
    
    def decode(
        self, 
        z_q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode quantized latents to multi-modal outputs."""
        return self.decoder(z_q)
    
    def classify(
        self, 
        quantized_codes: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Predict tags from quantized codes."""
        if self.classifiers is None:
            raise ValueError("Classifiers not initialized")
        return self.classifiers(quantized_codes)
    
    def forward(
        self,
        content_emb: torch.Tensor,
        collab_emb: torch.Tensor,
        temperature: float = 0.2,
        return_codes: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PRISM.
        
        Args:
            content_emb: Content embeddings (batch_size, 768)
            collab_emb: Collaborative embeddings (batch_size, 64)
            temperature: Temperature for quantization
            return_codes: Whether to return quantized codes
            
        Returns:
            output_dict: Dictionary containing:
                - content_recon: Reconstructed content
                - collab_recon: Reconstructed collaborative (targets weighted_collab in DHR mode)
                - weighted_collab_emb: Gate-weighted collab embedding (for DHR reconstruction target)
                - z_q: Quantized latent (if return_codes=True)
                - quantized_codes: List of codes per layer (if return_codes=True)
                - encoding_indices: List of indices per layer (if return_codes=True)
                - codebook_loss: Codebook loss
                - perplexities: List of perplexities
                - predictions: Tag predictions (if classifiers available)
        """
        # Encode (get weighted_collab for DHR reconstruction target)
        encode_result = self.encode(content_emb, collab_emb, return_weighted_collab=True)
        z, weighted_collab_emb = encode_result
        
        # Quantize
        z_q, quantized_codes, encoding_indices, codebook_loss, perplexities = \
            self.quantize(z, temperature)
        
        # Decode
        content_recon, collab_recon = self.decode(z_q)
        
        # Build output dictionary
        output_dict = {
            'content_recon': content_recon,
            'collab_recon': collab_recon,
            'weighted_collab_emb': weighted_collab_emb,  # DHR target: denoised collab
            'codebook_loss': codebook_loss,
            'perplexities': perplexities
        }
        
        # Add codes if requested
        if return_codes:
            output_dict['z_q'] = z_q
            output_dict['quantized_codes'] = quantized_codes
            output_dict['encoding_indices'] = encoding_indices
        
        # Add tag predictions if classifiers available
        if self.classifiers is not None:
            predictions = self.classify(quantized_codes)
            output_dict['predictions'] = predictions
        
        return output_dict
    
    def get_codebooks(self) -> List[torch.Tensor]:
        """
        Get all codebook tensors for anchor loss computation.
        
        Returns:
            codebooks: List of codebook tensors [C1, C2, C3]
        """
        codebooks = []
        for quantizer in self.quantizers:
            if hasattr(quantizer, 'embedding'):
                if isinstance(quantizer.embedding, nn.Parameter):
                    codebooks.append(quantizer.embedding)
                elif isinstance(quantizer.embedding, nn.Embedding):
                    codebooks.append(quantizer.embedding.weight)
                else:
                    raise ValueError(f"Unknown embedding type: {type(quantizer.embedding)}")
            else:
                raise ValueError("Quantizer has no embedding attribute")
        return codebooks
    
    def generate_semantic_ids(
        self,
        content_emb: torch.Tensor,
        collab_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate semantic IDs for items.
        
        Args:
            content_emb: Content embeddings (batch_size, 768)
            collab_emb: Collaborative embeddings (batch_size, 64)
            
        Returns:
            semantic_ids: Hierarchical IDs (batch_size, n_layers)
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(content_emb, collab_emb, return_weighted_collab=False)
            _, _, encoding_indices, _, _ = self.quantize(z)
            
            # Stack indices to form IDs
            semantic_ids = torch.stack(encoding_indices, dim=1)  # (B, n_layers)
        
        return semantic_ids
    
    def get_codebook_sizes(self) -> List[int]:
        """
        Get codebook sizes for each layer.
        
        Returns:
            codebook_sizes: List of codebook sizes per layer
        """
        return self.n_embed_per_layer


def create_prism_from_config(
    config: Dict,
    num_classes_per_layer: Optional[List[int]] = None
) -> PRISM:
    """
    Factory function to create PRISM from configuration dictionary.
    
    Args:
        config: Configuration dictionary with model parameters
        num_classes_per_layer: Number of tag classes per layer
        
    Returns:
        model: Initialized PRISM model
    """
    return PRISM(
        content_dim=config.get('content_dim', 768),
        collab_dim=config.get('collab_dim', 64),
        latent_dim=config.get('latent_dim', 32),
        n_layers=config.get('n_layers', 3),
        n_embed=config.get('n_embed', 256),
        n_embed_per_layer=config.get('n_embed_per_layer'),  # Support variable sizes
        encoder_hidden_dims=config.get('encoder_hidden_dims'),
        decoder_hidden_dims=config.get('decoder_hidden_dims'),
        use_gated_fusion=config.get('use_gated_fusion', True),  # Default: use gated fusion
        use_dual_decoder=config.get('use_dual_decoder', True),  # Default: use dual decoder
        num_classes_per_layer=num_classes_per_layer,
        use_ema=config.get('use_ema', True),
        ema_decay=config.get('ema_decay', 0.99),
        beta=config.get('beta', 0.25),
        quantize_mode=QuantizeMode(config.get('quantize_mode', 'rotation')),
        dropout=config.get('dropout', 0.1)
    )

