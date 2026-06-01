"""
PRISM: Hierarchical ID VAE with IDE + UPR + SACO

Architecture:
1. IDE: projects text (768D) and collab (64D) -> shared 128D
2. 
2. Fusion: z_clean = [h_c || h_t]  (256D clean fused feature)
3. Encoder: z_clean -> MLP -> z_latent (latent_dim)
4. RQ-VAE: hierarchical quantization of z_latent
5. UnifiedDecoder: z_q -> z_dec (256D, reconstructs z_clean)

Loss: L_UPR (MSE on z_clean) + beta * L_commit + lambda_sac * L_SACO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from RQ_VAE import RQVAEQuantizer, QuantizeMode
from ide import IDEEqualizer


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder with IDE + MCD pipeline.

    Pipeline:
    1. IDE: Project text (768D) and collab (64D) to shared dimension d=128
    
    2. Fusion: z_clean = [h_c || h_t]  (256D clean feature)
    4. Encode: MLP projects z_clean -> z_latent (latent_dim)
    """

    def __init__(
        self,
        content_dim: int = 768,
        collab_dim: int = 64,
        latent_dim: int = 32,
        hidden_dims: Optional[List[int]] = None,
        use_ide: bool = True,
        ide_dim: int = 128,
    ):
        super().__init__()

        self.content_dim = content_dim
        self.collab_dim = collab_dim
        self.latent_dim = latent_dim
        self.use_ide = use_ide
        self.ide_dim = ide_dim

        if use_ide:
            self.ide = IDEEqualizer(
                content_dim=content_dim,
                collab_dim=collab_dim,
                d=ide_dim,
            )
            fusion_dim = ide_dim * 2  # 256
        else:
            self.ide = None
            fusion_dim = content_dim + collab_dim

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        prev_dim = fusion_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(
        self,
        content_emb: torch.Tensor,
        collab_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.use_ide:
            h_t, h_c = self.ide(content_emb, collab_emb)
        else:
            h_t, h_c = content_emb, collab_emb

        z_clean = torch.cat([h_c, h_t], dim=-1)
        z = self.encoder(z_clean)

        return {
            'z': z,
            'z_clean': z_clean,
            'h_t': h_t,
            'h_c': h_c,
        }


class UnifiedDecoder(nn.Module):
    """
    Unified Purified Decoder: reconstructs z_clean (256D) from quantized latent.

    Replaces the old dual-head decoder. Since MCD already produces high-quality
    256D fused features with equalized dimensions, the decoder only needs to
    faithfully reconstruct this clean target — no need to project back to the
    asymmetric, noisy original spaces.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        output_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256, 512]

        shared_layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        self.shared_decoder = nn.Sequential(*shared_layers)

        # Single clean head: shared_features -> output_dim (256D)
        self.output_head = nn.Sequential(
            nn.Linear(prev_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        shared_features = self.shared_decoder(z_q)
        return self.output_head(shared_features)


class PRISM(nn.Module):
    """
    PRISM with IDE + MCD + UPR pipeline.

    Architecture:
    1. IDE: projects text (768D) and collab (64D) -> shared 128D
    2. Fusion: z_clean = [h_c || h_t]  (256D)
    3. Encoder: z_clean -> MLP -> z_latent (latent_dim)
    4. RQ-VAE: hierarchical quantization of z_latent -> z_q
    5. UnifiedDecoder: z_q -> z_dec (256D, targets z_clean.detach())

    Loss: L_UPR = MSE(z_dec, z_clean.detach())
    """

    def __init__(
        self,
        content_dim: int = 768,
        collab_dim: int = 64,
        latent_dim: int = 32,
        n_layers: int = 3,
        n_embed: int = 256,
        n_embed_per_layer: Optional[List[int]] = None,
        encoder_hidden_dims: Optional[List[int]] = None,
        decoder_hidden_dims: Optional[List[int]] = None,
        use_ide: bool = True,
        ide_dim: int = 128,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        beta: float = 0.25,
        quantize_mode: QuantizeMode = QuantizeMode.ROTATION,
    ):
        super().__init__()

        self.content_dim = content_dim
        self.collab_dim = collab_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        if n_embed_per_layer is None:
            self.n_embed_per_layer = [n_embed] * n_layers
        else:
            assert len(n_embed_per_layer) == n_layers, \
                f"n_embed_per_layer must have {n_layers} elements, got {len(n_embed_per_layer)}"
            self.n_embed_per_layer = n_embed_per_layer

        self.n_embed = n_embed
        self.beta = beta
        output_dim = ide_dim * 2 if use_ide else content_dim + collab_dim

        self.encoder = MultiModalEncoder(
            content_dim=content_dim,
            collab_dim=collab_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            use_ide=use_ide,
            ide_dim=ide_dim,
        )

        self.quantizers = nn.ModuleList([
            RQVAEQuantizer(
                n_embed=self.n_embed_per_layer[i],
                embed_dim=latent_dim,
                beta=beta,
                use_ema=use_ema,
                decay=ema_decay,
                quantize_mode=quantize_mode,
            )
            for i in range(n_layers)
        ])

        self.decoder = UnifiedDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=decoder_hidden_dims,
        )

    def encode(self, content_emb, collab_emb):
        return self.encoder(content_emb, collab_emb)

    def quantize(self, z, temperature=0.2):
        residual = z
        z_q_layers = []
        quantized_codes = []
        encoding_indices = []
        total_codebook_loss = 0.0
        total_commitment_loss = 0.0
        perplexities = []

        for layer_idx, quantizer in enumerate(self.quantizers):
            z_q_layer, codebook_loss, commitment_loss, indices, unused_codes = quantizer(
                residual, temperature=temperature
            )
            residual = residual - z_q_layer
            z_q_layers.append(z_q_layer)
            quantized_codes.append(z_q_layer)
            encoding_indices.append(indices)
            total_codebook_loss += codebook_loss
            total_commitment_loss += commitment_loss

            n_embed = self.n_embed_per_layer[layer_idx]
            perplexities.append(n_embed - unused_codes)

        z_q = torch.stack(z_q_layers, dim=0).sum(dim=0)
        # Pass raw codebook + commitment to loss function.
        # The effective commitment weight is controlled by --commit_weight (default 0.0625).
        total_loss = total_codebook_loss + total_commitment_loss

        return z_q, quantized_codes, encoding_indices, total_loss, perplexities

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(
        self,
        content_emb: torch.Tensor,
        collab_emb: torch.Tensor,
        temperature: float = 0.2,
        return_codes: bool = False,
    ) -> Dict[str, torch.Tensor]:
        enc_outputs = self.encode(content_emb, collab_emb)
        z = enc_outputs['z']
        z_clean = enc_outputs['z_clean']

        z_q, quantized_codes, encoding_indices, codebook_loss, perplexities = \
            self.quantize(z, temperature)

        z_dec = self.decode(z_q)

        output_dict = {
            'z_dec': z_dec,
            'z_clean': z_clean,
            'z': z,
            'h_t': enc_outputs['h_t'],
            'h_c': enc_outputs['h_c'],
            'codebook_loss': codebook_loss,
            'perplexities': perplexities,
        }

        if return_codes:
            output_dict['z_q'] = z_q
            output_dict['quantized_codes'] = quantized_codes
            output_dict['encoding_indices'] = encoding_indices

        return output_dict

    def get_codebooks(self) -> List[torch.Tensor]:
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
        collab_emb: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            enc_outputs = self.encode(content_emb, collab_emb)
            _, _, encoding_indices, _, _ = self.quantize(enc_outputs['z'])
            semantic_ids = torch.stack(encoding_indices, dim=1)
        return semantic_ids

    def get_codebook_sizes(self) -> List[int]:
        return self.n_embed_per_layer


def create_prism_from_config(config: Dict) -> PRISM:
    """Factory function to create PRISM from configuration dictionary."""
    return PRISM(
        content_dim=config.get('content_dim', 768),
        collab_dim=config.get('collab_dim', 64),
        latent_dim=config.get('latent_dim', 32),
        n_layers=config.get('n_layers', 3),
        n_embed=config.get('n_embed', 256),
        n_embed_per_layer=config.get('n_embed_per_layer'),
        encoder_hidden_dims=config.get('encoder_hidden_dims'),
        decoder_hidden_dims=config.get('decoder_hidden_dims'),
        use_ide=config.get('use_ide', True),
        ide_dim=config.get('ide_dim', 128),
        use_ema=config.get('use_ema', True),
        ema_decay=config.get('ema_decay', 0.99),
        beta=config.get('beta', 0.25),
        quantize_mode=QuantizeMode(config.get('quantize_mode', 'rotation')),
    )
