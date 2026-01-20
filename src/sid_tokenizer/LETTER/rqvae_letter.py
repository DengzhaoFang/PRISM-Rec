"""
LETTER RQ-VAE Implementation

Based on the original LETTER paper: "Learnable Item Tokenization for Generative Recommendation"

This is a faithful reproduction of the original implementation with:
1. MLP encoder/decoder
2. Residual quantization with diversity loss
3. CF (Collaborative Filtering) loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List

# Handle both direct execution and module import
try:
    from .vq_letter import LETTERResidualQuantizer
except ImportError:
    from vq_letter import LETTERResidualQuantizer


class MLPLayers(nn.Module):
    """MLP layers following LETTER's original implementation."""
    
    def __init__(
        self, 
        layers: List[int], 
        dropout: float = 0.0, 
        activation: str = "relu",
        bn: bool = False
    ):
        super().__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            # Add activation except for last layer
            if idx != len(layers) - 2:
                if activation.lower() == "relu":
                    mlp_modules.append(nn.ReLU())
                elif activation.lower() == "tanh":
                    mlp_modules.append(nn.Tanh())
                elif activation.lower() == "sigmoid":
                    mlp_modules.append(nn.Sigmoid())
                elif activation.lower() == "leakyrelu":
                    mlp_modules.append(nn.LeakyReLU())
                elif activation.lower() == "silu":
                    mlp_modules.append(nn.SiLU())
        
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_layers(x)


class LETTER_RQVAE(nn.Module):
    """
    LETTER RQ-VAE model.
    
    Faithful reproduction of the original LETTER implementation with:
    - MLP encoder/decoder
    - Residual quantization with diversity loss
    - CF (Collaborative Filtering) loss for alignment with collaborative embeddings
    """
    
    def __init__(
        self,
        in_dim: int = 768,
        num_emb_list: List[int] = None,
        e_dim: int = 32,
        layers: List[int] = None,
        dropout_prob: float = 0.0,
        bn: bool = False,
        loss_type: str = "mse",
        quant_loss_weight: float = 1.0,
        kmeans_init: bool = True,
        kmeans_iters: int = 100,
        sk_epsilons: List[float] = None,
        sk_iters: int = 100,
        alpha: float = 0.1,  # CF loss weight
        beta: float = 0.1,   # Diversity loss weight (in quantizer)
        mu: float = 0.25,    # Commitment loss weight
        n_clusters: int = 10,
    ):
        super().__init__()
        
        if num_emb_list is None:
            num_emb_list = [256, 256, 256]
        if layers is None:
            layers = [512, 256, 128, 64]
        if sk_epsilons is None:
            sk_epsilons = [0.0] * len(num_emb_list)
        
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.n_clusters = n_clusters
        self.n_layers = len(num_emb_list)
        self.n_embed = num_emb_list[0]  # Assume all layers have same size
        
        # Encoder: in_dim -> layers -> e_dim
        encode_layer_dims = [in_dim] + layers + [e_dim]
        self.encoder = MLPLayers(
            layers=encode_layer_dims,
            dropout=dropout_prob,
            bn=bn
        )
        
        # Residual Quantizer
        self.rq = LETTERResidualQuantizer(
            n_embed_list=num_emb_list,
            embed_dim=e_dim,
            mu=mu,
            beta=beta,
            n_clusters=n_clusters,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
        )
        
        # Decoder: e_dim -> layers (reversed) -> in_dim
        decode_layer_dims = encode_layer_dims[::-1]
        self.decoder = MLPLayers(
            layers=decode_layer_dims,
            dropout=dropout_prob,
            bn=bn
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation."""
        return self.decoder(z)
    
    def cf_loss(
        self, 
        quantized_rep: torch.Tensor, 
        cf_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CF (Collaborative Filtering) loss.
        
        This aligns the quantized representation with collaborative embeddings
        using a contrastive loss (InfoNCE-style).
        
        Args:
            quantized_rep: Quantized representations (batch_size, e_dim)
            cf_embedding: Collaborative embeddings (batch_size, cf_dim)
            
        Returns:
            cf_loss: Scalar loss
        """
        batch_size = quantized_rep.size(0)
        labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
        
        # Project CF embedding to same dimension as quantized_rep if needed
        if cf_embedding.shape[1] != quantized_rep.shape[1]:
            # Lazy initialization of projection layer
            if not hasattr(self, 'cf_projection') or self.cf_projection is None:
                self.cf_projection = nn.Linear(
                    cf_embedding.shape[1], 
                    quantized_rep.shape[1], 
                    bias=False
                ).to(quantized_rep.device)
                # Initialize with xavier
                nn.init.xavier_normal_(self.cf_projection.weight)
            cf_embedding = self.cf_projection(cf_embedding)
        
        # Normalize for cosine similarity
        quantized_rep = F.normalize(quantized_rep, dim=-1)
        cf_embedding = F.normalize(cf_embedding, dim=-1)
        
        # Compute similarity matrix
        similarities = torch.matmul(quantized_rep, cf_embedding.transpose(0, 1))
        
        # Temperature scaling (use higher temp for more stable training)
        temperature = 0.1
        similarities = similarities / temperature
        
        # Cross-entropy loss (diagonal should be highest)
        cf_loss = F.cross_entropy(similarities, labels)
        return cf_loss
    
    def update_cluster_labels(self):
        """Update cluster labels for all quantizer layers."""
        self.rq.update_all_cluster_labels()
    
    def forward(
        self, 
        x: torch.Tensor, 
        cf_emb: Optional[torch.Tensor] = None,
        use_sk: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, in_dim)
            cf_emb: Collaborative embeddings (batch_size, cf_dim), optional
            use_sk: Whether to use Sinkhorn algorithm
            
        Returns:
            Dictionary with outputs and losses
        """
        # Encode
        z = self.encode(x)
        
        # Quantize
        z_q, quant_loss, indices, div_loss = self.rq(z, use_sk=use_sk)
        
        # Decode
        x_recon = self.decode(z_q)
        
        # Reconstruction loss
        if self.loss_type == 'mse':
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        elif self.loss_type == 'l1':
            recon_loss = F.l1_loss(x_recon, x, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Base loss (reconstruction + quantization)
        base_loss = recon_loss + self.quant_loss_weight * quant_loss
        
        # CF loss
        cf_loss_val = torch.tensor(0.0, device=x.device)
        if self.alpha > 0 and cf_emb is not None:
            cf_loss_val = self.cf_loss(z_q, cf_emb)
        
        # Total loss
        total_loss = base_loss + self.alpha * cf_loss_val
        
        # Calculate duplicate rate
        with torch.no_grad():
            codes_tuple = [tuple(row.tolist()) for row in indices]
            unique_codes = len(set(codes_tuple))
            duplicate_rate = 1.0 - unique_codes / len(codes_tuple)
        
        return {
            'x_recon': x_recon,
            'z': z,
            'z_q': z_q,
            'codes': indices,
            'recon_loss': recon_loss,
            'quant_loss': quant_loss,
            'diversity_loss': div_loss,
            'cf_loss': cf_loss_val,
            'total_loss': total_loss,
            'duplicate_rate': duplicate_rate,
        }
    
    @torch.no_grad()
    def get_indices(self, x: torch.Tensor, use_sk: bool = False) -> torch.Tensor:
        """Get quantization indices for input."""
        z = self.encode(x)
        _, _, indices, _ = self.rq(z, use_sk=use_sk)
        return indices
    
    def apply_post_id_deduplication(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Apply post-ID method to ensure unique semantic IDs.
        Adds a 4th code for items that share the same first 3 codewords.
        """
        batch_size = codes.shape[0]
        codes_np = codes.cpu().numpy()
        
        tuple_counts = {}
        unique_codes = np.zeros((batch_size, codes.shape[1] + 1), dtype=np.int64)
        
        for i, code_seq in enumerate(codes_np):
            code_tuple = tuple(code_seq)
            
            if code_tuple not in tuple_counts:
                tuple_counts[code_tuple] = 0
            else:
                tuple_counts[code_tuple] += 1
            
            unique_codes[i, :-1] = code_seq
            unique_codes[i, -1] = tuple_counts[code_tuple]
        
        return torch.tensor(unique_codes, device=codes.device, dtype=torch.long)
