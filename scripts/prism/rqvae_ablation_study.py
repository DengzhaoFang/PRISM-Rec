#!/usr/bin/env python3
"""
RQ-VAE Ablation Study: Comparing Different Input Modes

This script implements five RQ-VAE training modes to study the effect of
different input embeddings on semantic ID generation:

Mode 1 (semantic_only): RQ-VAE input is only semantic embedding
Mode 2 (collab_only): RQ-VAE input is only collaborative embedding  
Mode 3 (concat): RQ-VAE input is concatenation of semantic and collaborative embeddings
Mode 4 (contrastive): RQ-VAE input is semantic embedding, but trained with contrastive
                      loss between quantized embedding z_hat and collaborative embedding
Mode 5 (gated_dual): RQ-VAE input is semantic + gated(denoised) collaborative embedding,
                     with dual reconstruction heads for semantic and denoised collab

For each mode, after training converges, the script saves the quantized embeddings
(z_hat = sum of selected code embeddings) for each item.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'sid_tokenizer' / 'rq-base'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src' / 'sid_tokenizer' / 'rq-base' / 'tiger'))

from tiger.RQ_VAE import RQVAE, RQVAEEncoder, RQVAEDecoder, RQVAEQuantizer, QuantizeMode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Classes
# ============================================================================
class AblationDataset(Dataset):
    """Dataset for ablation study with semantic and collaborative embeddings"""
    
    def __init__(
        self,
        semantic_emb_path: str,
        collab_emb_path: Optional[str] = None,
        popularity_path: Optional[str] = None,  # Kept for backward compatibility, but not used
        max_items: Optional[int] = None
    ):
        """
        Args:
            semantic_emb_path: Path to semantic embeddings (parquet), also contains popularity_score
            collab_emb_path: Path to collaborative embeddings (npy), optional
            popularity_path: Deprecated, popularity_score is loaded from semantic_emb_path
            max_items: Maximum number of items to load
        """
        # Load semantic embeddings (also contains popularity_score)
        logger.info(f"Loading semantic embeddings from {semantic_emb_path}")
        df = pd.read_parquet(semantic_emb_path)
        
        if max_items is not None:
            df = df.head(max_items)
        
        # Get embedding column
        emb_col = 'embedding' if 'embedding' in df.columns else 'attribute_embedding'
        self.semantic_embs = torch.stack([
            torch.tensor(emb, dtype=torch.float32) for emb in df[emb_col]
        ])
        self.item_ids = df['ItemID'].values
        
        logger.info(f"  Loaded {len(self.item_ids)} items, semantic dim: {self.semantic_embs.shape[1]}")
        
        # Load collaborative embeddings if provided
        self.collab_embs = None
        if collab_emb_path and os.path.exists(collab_emb_path):
            logger.info(f"Loading collaborative embeddings from {collab_emb_path}")
            collab_all = np.load(collab_emb_path)
            
            # Map item IDs to collaborative embeddings
            # Assuming item_ids are 0-indexed internal IDs
            self.collab_embs = torch.tensor(collab_all[self.item_ids], dtype=torch.float32)
            logger.info(f"  Collaborative dim: {self.collab_embs.shape[1]}")
        
        # Load popularity scores from the same parquet file (like PRISM does)
        self.popularity_scores = None
        if 'popularity_score' in df.columns:
            self.popularity_scores = torch.tensor(
                df['popularity_score'].values, 
                dtype=torch.float32
            )
            logger.info(f"  ✓ Popularity scores loaded from item_emb.parquet")
            logger.info(f"    Range: [{self.popularity_scores.min():.4f}, {self.popularity_scores.max():.4f}]")
            logger.info(f"    Mean: {self.popularity_scores.mean():.4f}")
        else:
            logger.warning("  ⚠ No popularity_score column found in item_emb.parquet")
            logger.warning("    Gate supervision will use diversity regularization only")
        
    def __len__(self):
        return len(self.item_ids)
    
    def __getitem__(self, idx):
        result = {
            'item_id': self.item_ids[idx],
            'semantic_emb': self.semantic_embs[idx]
        }
        if self.collab_embs is not None:
            result['collab_emb'] = self.collab_embs[idx]
        if self.popularity_scores is not None:
            result['popularity_score'] = self.popularity_scores[idx]
        return result


# ============================================================================
# RQ-VAE with Contrastive Learning
# ============================================================================
class RQVAEContrastive(RQVAE):
    """
    RQ-VAE with contrastive learning between quantized embedding and collaborative embedding.
    
    The contrastive loss encourages:
    - Positive pairs: (z_hat_i, h_i) - same item's quantized and CF embeddings
    - Negative pairs: (z_hat_i, h_j) - different items' embeddings within batch
    """
    
    def __init__(
        self,
        input_dim: int,
        collab_dim: int = 64,
        latent_dim: int = 64,  # Match collab_dim for contrastive learning
        n_embed: int = 256,
        n_layers: int = 3,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        commitment_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        temperature: float = 0.07,
        quantize_mode: QuantizeMode = QuantizeMode.GUMBEL_SOFTMAX,
        normalize_residuals: bool = True
    ):
        # Initialize parent with latent_dim matching collab_dim
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            n_embed=n_embed,
            n_layers=n_layers,
            beta=beta,
            use_ema=use_ema,
            decay=decay,
            commitment_weight=commitment_weight,
            reconstruction_weight=reconstruction_weight,
            quantize_mode=quantize_mode,
            normalize_residuals=normalize_residuals
        )
        
        self.collab_dim = collab_dim
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = temperature
        
        # Projection head for collaborative embeddings (if dimensions don't match)
        if collab_dim != latent_dim:
            self.collab_projector = nn.Linear(collab_dim, latent_dim, bias=False)
        else:
            self.collab_projector = nn.Identity()
    
    def info_nce_loss(
        self,
        z_hat: torch.Tensor,
        h: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            z_hat: Quantized embeddings (batch_size, latent_dim)
            h: Collaborative embeddings (batch_size, latent_dim)
            temperature: Temperature for softmax
            
        Returns:
            InfoNCE loss
        """
        batch_size = z_hat.shape[0]
        
        # L2 normalize embeddings
        z_hat_norm = F.normalize(z_hat, dim=-1)
        h_norm = F.normalize(h, dim=-1)
        
        # Compute similarity matrix: (batch_size, batch_size)
        # sim[i, j] = cosine_similarity(z_hat_i, h_j)
        sim_matrix = torch.matmul(z_hat_norm, h_norm.t()) / temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=z_hat.device)
        
        # Cross entropy loss (treating as classification)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def forward_contrastive(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        gumbel_temperature: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive learning.
        
        Args:
            x: Semantic embeddings (batch_size, input_dim)
            h: Collaborative embeddings (batch_size, collab_dim)
            gumbel_temperature: Temperature for Gumbel-Softmax quantization
            
        Returns:
            Dictionary with outputs, losses, and statistics
        """
        # Standard RQ-VAE forward
        outputs = self.forward(x, temperature=gumbel_temperature)
        
        # Get quantized embedding z_hat
        z_hat = outputs['z_q']  # (batch_size, latent_dim)
        
        # Project collaborative embeddings
        h_proj = self.collab_projector(h)  # (batch_size, latent_dim)
        
        # Compute contrastive loss
        contrastive_loss = self.info_nce_loss(
            z_hat, h_proj, self.contrastive_temperature
        )
        
        # Update total loss
        outputs['contrastive_loss'] = contrastive_loss
        outputs['total_loss'] = (
            outputs['total_loss'] + 
            self.contrastive_weight * contrastive_loss
        )
        
        return outputs


# ============================================================================
# RQ-VAE with Gated Fusion and Dual Reconstruction (Mode 5)
# ============================================================================
class GateSupervisionLoss(nn.Module):
    """
    Gate supervision loss to align gate values with item popularity.
    
    Encourages:
    - Long-tail items (low popularity) → low gate (don't trust noisy collab signal)
    - Popular items (high popularity) → high gate (trust reliable collab signal)
    
    Also includes diversity regularization to prevent gate collapse.
    """
    
    def __init__(
        self, 
        weight: float = 0.1,
        diversity_weight: float = 0.5,
        target_std: float = 0.2
    ):
        """
        Args:
            weight: Overall weight for gate supervision loss
            diversity_weight: Weight for diversity regularization
            target_std: Target standard deviation for gate values
        """
        super().__init__()
        self.weight = weight
        self.diversity_weight = diversity_weight
        self.target_var = target_std ** 2
        
    def forward(
        self, 
        gate_values: torch.Tensor,      # (batch_size, collab_dim)
        popularity_scores: Optional[torch.Tensor] = None  # (batch_size,)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute gate supervision loss.
        
        Args:
            gate_values: Gate values from gate network (batch_size, collab_dim)
            popularity_scores: Ground truth popularity scores (batch_size,)
            
        Returns:
            loss: Supervision loss
            loss_dict: Dictionary with loss components
        """
        # Average gate value per item
        gate_mean = gate_values.mean(dim=1)  # (batch_size,)
        
        # Supervision loss (only if popularity scores available)
        if popularity_scores is not None:
            supervision_loss = F.mse_loss(gate_mean, popularity_scores)
        else:
            # Without popularity, don't use supervision loss
            # Only rely on diversity regularization
            supervision_loss = torch.tensor(0.0, device=gate_values.device)
        
        # Diversity regularization (encourage larger variance, prevent collapse)
        # This is the key to preventing gate collapse
        gate_var = gate_mean.var()
        diversity_loss = F.relu(self.target_var - gate_var)
        
        # Also add a penalty for gate values being too close to 0 or 1
        # Encourage gates to be in a reasonable range (not collapsed)
        gate_entropy = -(gate_mean * torch.log(gate_mean + 1e-8) + 
                        (1 - gate_mean) * torch.log(1 - gate_mean + 1e-8)).mean()
        # Max entropy is at gate=0.5, which gives entropy ≈ 0.693
        # We want to encourage higher entropy (more diverse gates)
        entropy_loss = F.relu(0.5 - gate_entropy)  # Penalize if entropy < 0.5
        
        # Combined loss
        total_loss = supervision_loss + self.diversity_weight * (diversity_loss + entropy_loss)
        
        loss_dict = {
            'gate_supervision': supervision_loss.item() if torch.is_tensor(supervision_loss) else supervision_loss,
            'gate_diversity': diversity_loss.item(),
            'gate_entropy': gate_entropy.item(),
            'gate_variance': gate_var.item(),
            'gate_mean': gate_mean.mean().item(),
            'gate_std': gate_mean.std().item()
        }
        
        return self.weight * total_loss, loss_dict


class GateNetwork(nn.Module):
    """
    Gate network for denoising collaborative embeddings.
    
    Learns a dynamic "trust score" (0-1) for each dimension of the collaborative
    embedding. For noisy long-tail items, gate -> 0 (ignore collab signal).
    For high-quality popular items, gate -> 1 (trust collab signal).
    """
    
    def __init__(self, collab_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(collab_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, collab_dim),
            nn.Sigmoid()  # Output in (0, 1)
        )
    
    def forward(self, collab_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            collab_emb: Collaborative embeddings (batch_size, collab_dim)
            
        Returns:
            weighted_collab: Denoised collaborative embeddings
            gate_values: Gate values for analysis
        """
        gate_values = self.gate(collab_emb)  # (B, collab_dim)
        weighted_collab = gate_values * collab_emb  # Element-wise gating
        return weighted_collab, gate_values


class RQVAEGatedDual(nn.Module):
    """
    RQ-VAE with Gated Fusion and Dual Reconstruction Heads.
    
    Architecture:
    1. Gate network: collab_emb -> denoised_collab (via learned gate)
    2. Encoder: concat(semantic, denoised_collab) -> z
    3. RQ-VAE quantization: z -> z_q
    4. Dual decoders:
       - Semantic head: z_q -> semantic_recon
       - Collab head: z_q -> collab_recon (targets denoised_collab, not original)
    
    This follows PRISM's approach of using gated fusion to filter noisy
    collaborative signals, especially for long-tail items.
    
    Gate Supervision (optional):
    - Aligns gate values with item popularity scores
    - Adds diversity regularization to prevent gate collapse
    """
    
    def __init__(
        self,
        semantic_dim: int = 768,
        collab_dim: int = 64,
        latent_dim: int = 64,
        n_embed: int = 256,
        n_layers: int = 3,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        semantic_recon_weight: float = 1.0,
        collab_recon_weight: float = 1.0,
        # Gate supervision parameters
        use_gate_supervision: bool = True,
        gate_supervision_weight: float = 0.1,
        gate_diversity_weight: float = 0.5,
        gate_target_std: float = 0.2,
        quantize_mode: QuantizeMode = QuantizeMode.GUMBEL_SOFTMAX,
        normalize_residuals: bool = True
    ):
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.collab_dim = collab_dim
        self.latent_dim = latent_dim
        self.n_embed = n_embed
        self.n_layers = n_layers
        self.beta = beta
        self.use_ema = use_ema
        self.semantic_recon_weight = semantic_recon_weight
        self.collab_recon_weight = collab_recon_weight
        self.normalize_residuals = normalize_residuals
        self.use_gate_supervision = use_gate_supervision
        
        # Input dimension is concatenation of semantic and collab
        self.input_dim = semantic_dim + collab_dim
        
        # Gate network for denoising collaborative embeddings
        self.gate_network = GateNetwork(collab_dim)
        
        # Gate supervision loss (optional)
        if use_gate_supervision:
            self.gate_supervision_loss = GateSupervisionLoss(
                weight=gate_supervision_weight,
                diversity_weight=gate_diversity_weight,
                target_std=gate_target_std
            )
        else:
            self.gate_supervision_loss = None
        
        # Fusion layer normalization
        self.fusion_norm = nn.LayerNorm(self.input_dim)
        
        # Encoder: concat(semantic, denoised_collab) -> latent
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512, bias=False),
            nn.SiLU(),
            nn.Linear(512, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, latent_dim, bias=False)
        )
        
        # Shared decoder backbone
        self.shared_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, 256, bias=False),
            nn.SiLU(),
            nn.Linear(256, 512, bias=False),
            nn.SiLU()
        )
        
        # Dual reconstruction heads
        # Semantic head: 512 -> semantic_dim
        self.semantic_head = nn.Sequential(
            nn.Linear(512, semantic_dim * 2),
            nn.LayerNorm(semantic_dim * 2),
            nn.ReLU(),
            nn.Linear(semantic_dim * 2, semantic_dim)
        )
        
        # Collab head: 512 -> collab_dim
        collab_hidden = max(256, collab_dim * 4)
        self.collab_head = nn.Sequential(
            nn.Linear(512, collab_hidden),
            nn.LayerNorm(collab_hidden),
            nn.ReLU(),
            nn.Linear(collab_hidden, collab_dim)
        )
        
        # RQ-VAE quantizers
        self.quantizers = nn.ModuleList([
            RQVAEQuantizer(
                n_embed=n_embed,
                embed_dim=latent_dim,
                beta=beta,
                use_ema=use_ema,
                decay=decay,
                quantize_mode=quantize_mode
            )
            for _ in range(n_layers)
        ])
    
    def encode(
        self, 
        semantic_emb: torch.Tensor, 
        collab_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode with gated fusion.
        
        Returns:
            z: Latent representation
            denoised_collab: Gate-weighted collaborative embedding
            gate_values: Gate values for analysis
        """
        # Apply gate to denoise collaborative embedding
        denoised_collab, gate_values = self.gate_network(collab_emb)
        
        # Concatenate semantic and denoised collab
        fused = torch.cat([semantic_emb, denoised_collab], dim=-1)
        fused = self.fusion_norm(fused)
        
        # Encode to latent space
        z = self.encoder(fused)
        
        return z, denoised_collab, gate_values
    
    def decode(self, z_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode with dual heads.
        
        Returns:
            semantic_recon: Reconstructed semantic embedding
            collab_recon: Reconstructed (denoised) collaborative embedding
        """
        shared_features = self.shared_decoder(z_q)
        semantic_recon = self.semantic_head(shared_features)
        collab_recon = self.collab_head(shared_features)
        return semantic_recon, collab_recon
    
    def quantize(
        self, 
        z: torch.Tensor, 
        temperature: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Apply residual quantization."""
        # Initialize codebooks on first forward pass
        if self.training and not all(q._initialized for q in self.quantizers):
            self._init_all_codebooks(z, temperature)
        
        residual = z
        z_q_total = torch.zeros_like(z)
        codebook_loss_total = 0.0
        commitment_loss_total = 0.0
        total_unused = 0
        codes = []
        
        for quantizer in self.quantizers:
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)
            
            z_q_layer, cb_loss, cm_loss, layer_codes, unused = quantizer(residual, temperature)
            z_q_total += z_q_layer
            residual = residual - z_q_layer
            codebook_loss_total += cb_loss
            commitment_loss_total += cm_loss
            total_unused += unused
            codes.append(layer_codes)
        
        codes = torch.stack(codes, dim=1)
        return z_q_total, codebook_loss_total, commitment_loss_total, codes, total_unused
    
    def _init_all_codebooks(self, z: torch.Tensor, temperature: float = 0.2):
        """Initialize all codebook layers with k-means."""
        print("=" * 60)
        print("Initializing all RQ-VAE codebooks (Gated Dual mode)...")
        print("=" * 60)
        
        residual = z
        for layer_idx, quantizer in enumerate(self.quantizers):
            if quantizer._initialized:
                continue
            
            if self.normalize_residuals:
                residual_norm = F.normalize(residual, dim=-1)
            else:
                residual_norm = residual
            
            print(f"\nLayer {layer_idx}:")
            quantizer._kmeans_init(residual_norm)
            
            with torch.no_grad():
                z_q_layer, _, _, _, _ = quantizer(residual, temperature)
                residual = residual - z_q_layer
        
        print("=" * 60)
        print("✓ All codebooks initialized!")
        print("=" * 60)
    
    def forward(
        self,
        semantic_emb: torch.Tensor,
        collab_emb: torch.Tensor,
        popularity_scores: Optional[torch.Tensor] = None,
        temperature: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with gated fusion and dual reconstruction.
        
        Args:
            semantic_emb: Semantic embeddings (batch_size, semantic_dim)
            collab_emb: Collaborative embeddings (batch_size, collab_dim)
            popularity_scores: Optional popularity scores for gate supervision (batch_size,)
            temperature: Temperature for Gumbel-Softmax quantization
        """
        # Encode with gating
        z, denoised_collab, gate_values = self.encode(semantic_emb, collab_emb)
        
        # Quantize
        z_q, codebook_loss, commitment_loss, codes, total_unused = self.quantize(z, temperature)
        
        # Decode with dual heads
        semantic_recon, collab_recon = self.decode(z_q)
        
        # Reconstruction losses
        # Semantic reconstruction targets original semantic embedding
        semantic_recon_loss = ((semantic_recon - semantic_emb) ** 2).sum(dim=-1).mean()
        
        # Collab reconstruction targets DENOISED collab (not original)
        # This is key: we reconstruct the gated/denoised version
        collab_recon_loss = ((collab_recon - denoised_collab) ** 2).sum(dim=-1).mean()
        
        # Combined reconstruction loss
        recon_loss = (self.semantic_recon_weight * semantic_recon_loss + 
                     self.collab_recon_weight * collab_recon_loss)
        
        # VQ loss
        vq_loss = codebook_loss + self.beta * commitment_loss
        
        # Total loss (before gate supervision)
        total_loss = recon_loss + vq_loss
        
        # Gate supervision loss (if enabled)
        gate_sup_loss = 0.0
        gate_sup_dict = {}
        if self.gate_supervision_loss is not None:
            gate_sup_loss, gate_sup_dict = self.gate_supervision_loss(
                gate_values, popularity_scores
            )
            total_loss = total_loss + gate_sup_loss
        
        # Codebook usage
        codebook_usage = 1.0 - (total_unused / (self.n_layers * self.n_embed))
        
        # Duplicate rate
        codes_np = codes.detach().cpu().numpy()
        unique_codes = len(set(tuple(c) for c in codes_np))
        duplicate_rate_pre = 1.0 - unique_codes / len(codes_np) if len(codes_np) > 0 else 0.0
        
        # Gate statistics (from gate_sup_dict if available, else compute)
        if gate_sup_dict:
            mean_gate = gate_sup_dict.get('gate_mean', gate_values.mean().item())
            gate_std = gate_sup_dict.get('gate_std', gate_values.mean(dim=1).std().item())
            gate_var = gate_sup_dict.get('gate_variance', gate_values.mean(dim=1).var().item())
        else:
            mean_gate = gate_values.mean().item()
            gate_std = gate_values.mean(dim=1).std().item()
            gate_var = gate_values.mean(dim=1).var().item()
        
        result = {
            'semantic_recon': semantic_recon,
            'collab_recon': collab_recon,
            'denoised_collab': denoised_collab,
            'gate_values': gate_values,
            'z': z,
            'z_q': z_q,
            'codes': codes,
            'recon_loss': recon_loss,
            'semantic_recon_loss': semantic_recon_loss,
            'collab_recon_loss': collab_recon_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'vq_loss': vq_loss,
            'total_loss': total_loss,
            'codebook_usage': codebook_usage,
            'duplicate_rate_pre': duplicate_rate_pre,
            'mean_gate': mean_gate,
            'gate_std': gate_std,
            'gate_var': gate_var
        }
        
        # Add gate supervision metrics if available
        if gate_sup_dict:
            result['gate_supervision_loss'] = gate_sup_dict.get('gate_supervision', 0.0)
            result['gate_diversity_loss'] = gate_sup_dict.get('gate_diversity', 0.0)
        
        return result


# ============================================================================
# Training Functions
# ============================================================================
class AblationTrainer:
    """Trainer for RQ-VAE ablation study"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging"""
        log_dir = self.config['output_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode = self.config['mode']
        log_path = os.path.join(log_dir, f'{timestamp}_{mode}.log')
        
        # Add file handler
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to: {log_path}")
        logger.info(f"Mode: {mode}")
        logger.info(f"Config: {json.dumps(self.config, indent=2, default=str)}")
    
    def load_data(self) -> DataLoader:
        """Load dataset"""
        dataset = AblationDataset(
            semantic_emb_path=self.config['semantic_emb_path'],
            collab_emb_path=self.config.get('collab_emb_path'),
            max_items=self.config.get('max_items')
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return dataloader, dataset
    
    def create_model(self, semantic_dim: int, collab_dim: Optional[int] = None) -> nn.Module:
        """Create model based on mode"""
        mode = self.config['mode']
        
        if mode == 'semantic_only':
            # Mode 1: Only semantic embedding
            input_dim = semantic_dim
            model = RQVAE(
                input_dim=input_dim,
                latent_dim=self.config.get('latent_dim', 64),
                n_embed=self.config.get('n_embed', 256),
                n_layers=self.config.get('n_layers', 3),
                beta=self.config.get('beta', 0.25),
                use_ema=self.config.get('use_ema', True),
                decay=self.config.get('ema_decay', 0.99),
                quantize_mode=QuantizeMode.GUMBEL_SOFTMAX
            )
            
        elif mode == 'collab_only':
            # Mode 2: Only collaborative embedding
            if collab_dim is None:
                raise ValueError("collab_emb_path required for collab_only mode")
            input_dim = collab_dim
            model = RQVAE(
                input_dim=input_dim,
                latent_dim=self.config.get('latent_dim', 64),
                n_embed=self.config.get('n_embed', 256),
                n_layers=self.config.get('n_layers', 3),
                beta=self.config.get('beta', 0.25),
                use_ema=self.config.get('use_ema', True),
                decay=self.config.get('ema_decay', 0.99),
                quantize_mode=QuantizeMode.GUMBEL_SOFTMAX
            )
            
        elif mode == 'concat':
            # Mode 3: Concatenation of semantic and collaborative
            if collab_dim is None:
                raise ValueError("collab_emb_path required for concat mode")
            input_dim = semantic_dim + collab_dim
            model = RQVAE(
                input_dim=input_dim,
                latent_dim=self.config.get('latent_dim', 64),
                n_embed=self.config.get('n_embed', 256),
                n_layers=self.config.get('n_layers', 3),
                beta=self.config.get('beta', 0.25),
                use_ema=self.config.get('use_ema', True),
                decay=self.config.get('ema_decay', 0.99),
                quantize_mode=QuantizeMode.GUMBEL_SOFTMAX
            )
            
        elif mode == 'contrastive':
            # Mode 4: Semantic input with contrastive learning
            if collab_dim is None:
                raise ValueError("collab_emb_path required for contrastive mode")
            model = RQVAEContrastive(
                input_dim=semantic_dim,
                collab_dim=collab_dim,
                latent_dim=self.config.get('latent_dim', 64),
                n_embed=self.config.get('n_embed', 256),
                n_layers=self.config.get('n_layers', 3),
                beta=self.config.get('beta', 0.25),
                use_ema=self.config.get('use_ema', True),
                decay=self.config.get('ema_decay', 0.99),
                contrastive_weight=self.config.get('contrastive_weight', 1.0),
                temperature=self.config.get('contrastive_temperature', 0.07),
                quantize_mode=QuantizeMode.GUMBEL_SOFTMAX
            )
            
        elif mode == 'gated_dual':
            # Mode 5: Gated fusion with dual reconstruction heads
            if collab_dim is None:
                raise ValueError("collab_emb_path required for gated_dual mode")
            model = RQVAEGatedDual(
                semantic_dim=semantic_dim,
                collab_dim=collab_dim,
                latent_dim=self.config.get('latent_dim', 64),
                n_embed=self.config.get('n_embed', 256),
                n_layers=self.config.get('n_layers', 3),
                beta=self.config.get('beta', 0.25),
                use_ema=self.config.get('use_ema', True),
                decay=self.config.get('ema_decay', 0.99),
                semantic_recon_weight=self.config.get('semantic_recon_weight', 1.0),
                collab_recon_weight=self.config.get('collab_recon_weight', 1.0),
                # Gate supervision parameters
                use_gate_supervision=self.config.get('use_gate_supervision', True),
                gate_supervision_weight=self.config.get('gate_supervision_weight', 0.1),
                gate_diversity_weight=self.config.get('gate_diversity_weight', 0.5),
                gate_target_std=self.config.get('gate_target_std', 0.2),
                quantize_mode=QuantizeMode.GUMBEL_SOFTMAX
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        logger.info(f"Created model for mode '{mode}'")
        if mode in ['semantic_only', 'collab_only', 'concat']:
            logger.info(f"  Input dim: {input_dim}")
        elif mode == 'contrastive':
            logger.info(f"  Input dim: {semantic_dim} (semantic)")
        elif mode == 'gated_dual':
            logger.info(f"  Input dim: {semantic_dim} (semantic) + {collab_dim} (collab, gated)")
        logger.info(f"  Latent dim: {model.latent_dim}")
        logger.info(f"  Codebook: {model.n_layers} layers x {model.n_embed} codes")
        
        return model.to(self.device)
    
    def get_input_embedding(self, batch: Dict, mode: str) -> torch.Tensor:
        """Get input embedding based on mode"""
        if mode == 'semantic_only':
            return batch['semantic_emb'].to(self.device)
        elif mode == 'collab_only':
            return batch['collab_emb'].to(self.device)
        elif mode == 'concat':
            semantic = batch['semantic_emb'].to(self.device)
            collab = batch['collab_emb'].to(self.device)
            return torch.cat([semantic, collab], dim=-1)
        elif mode == 'contrastive':
            return batch['semantic_emb'].to(self.device)
        elif mode == 'gated_dual':
            # For gated_dual, we return both embeddings separately
            # The model handles the fusion internally
            return None  # Special case, handled in train loop
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("Starting training...")
        
        # Load data
        dataloader, dataset = self.load_data()
        
        # Get dimensions
        first_batch = next(iter(dataloader))
        semantic_dim = first_batch['semantic_emb'].shape[1]
        collab_dim = first_batch['collab_emb'].shape[1] if 'collab_emb' in first_batch else None
        
        logger.info(f"Semantic dim: {semantic_dim}, Collab dim: {collab_dim}")
        
        # Create model
        model = self.create_model(semantic_dim, collab_dim)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Training settings
        mode = self.config['mode']
        epochs = self.config['epochs']
        init_temperature = self.config.get('init_temperature', 1.0)
        min_temperature = self.config.get('min_temperature', 0.2)
        total_steps = epochs * len(dataloader)
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('early_stop_patience', 50)
        best_model_state = None
        
        training_stats = []
        global_step = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_stats = defaultdict(float)
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in progress_bar:
                # Calculate temperature (cosine annealing)
                progress = global_step / total_steps
                temperature = min_temperature + 0.5 * (init_temperature - min_temperature) * (1 + np.cos(np.pi * progress))
                
                optimizer.zero_grad()
                
                # Forward pass based on mode
                if mode == 'contrastive':
                    x = self.get_input_embedding(batch, mode)
                    h = batch['collab_emb'].to(self.device)
                    outputs = model.forward_contrastive(x, h, gumbel_temperature=temperature)
                elif mode == 'gated_dual':
                    semantic = batch['semantic_emb'].to(self.device)
                    collab = batch['collab_emb'].to(self.device)
                    # Get popularity scores if available
                    popularity = batch.get('popularity_score')
                    if popularity is not None:
                        popularity = popularity.to(self.device)
                    outputs = model(semantic, collab, popularity_scores=popularity, temperature=temperature)
                else:
                    x = self.get_input_embedding(batch, mode)
                    outputs = model(x, temperature=temperature)
                
                loss = outputs['total_loss']
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 1.0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'])
                
                optimizer.step()
                global_step += 1
                
                # Accumulate stats
                for key in ['total_loss', 'recon_loss', 'vq_loss', 'codebook_loss', 
                           'commitment_loss', 'codebook_usage', 'duplicate_rate_pre']:
                    if key in outputs:
                        val = outputs[key].item() if torch.is_tensor(outputs[key]) else outputs[key]
                        epoch_stats[key] += val
                
                if 'contrastive_loss' in outputs:
                    epoch_stats['contrastive_loss'] += outputs['contrastive_loss'].item()
                
                # Mode 5 specific stats
                if 'semantic_recon_loss' in outputs:
                    epoch_stats['semantic_recon_loss'] += outputs['semantic_recon_loss'].item()
                if 'collab_recon_loss' in outputs:
                    epoch_stats['collab_recon_loss'] += outputs['collab_recon_loss'].item()
                if 'mean_gate' in outputs:
                    epoch_stats['mean_gate'] += outputs['mean_gate']
                if 'gate_std' in outputs:
                    epoch_stats['gate_std'] += outputs['gate_std']
                if 'gate_var' in outputs:
                    epoch_stats['gate_var'] += outputs['gate_var']
                if 'gate_supervision_loss' in outputs:
                    epoch_stats['gate_supervision_loss'] += outputs['gate_supervision_loss']
                if 'gate_diversity_loss' in outputs:
                    epoch_stats['gate_diversity_loss'] += outputs['gate_diversity_loss']
                
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f"{outputs['total_loss'].item():.4f}",
                    'Recon': f"{outputs['recon_loss'].item():.4f}",
                    'Usage': f"{outputs['codebook_usage']:.3f}"
                })
            
            # Average stats
            for key in epoch_stats:
                epoch_stats[key] /= num_batches
            epoch_stats['epoch'] = epoch
            epoch_stats['temperature'] = temperature
            
            training_stats.append(dict(epoch_stats))
            
            # Log
            log_msg = (f"Epoch {epoch+1}: Loss={epoch_stats['total_loss']:.4f}, "
                      f"Recon={epoch_stats['recon_loss']:.4f}, "
                      f"Usage={epoch_stats['codebook_usage']:.4f}")
            if 'contrastive_loss' in epoch_stats:
                log_msg += f", Contrastive={epoch_stats['contrastive_loss']:.4f}"
            if 'semantic_recon_loss' in epoch_stats and 'collab_recon_loss' in epoch_stats:
                log_msg += f", SemRecon={epoch_stats['semantic_recon_loss']:.4f}"
                log_msg += f", ColRecon={epoch_stats['collab_recon_loss']:.4f}"
            if 'mean_gate' in epoch_stats:
                log_msg += f", Gate={epoch_stats['mean_gate']:.4f}"
            if 'gate_std' in epoch_stats:
                log_msg += f", GateStd={epoch_stats['gate_std']:.4f}"
            if 'gate_supervision_loss' in epoch_stats:
                log_msg += f", GateSup={epoch_stats['gate_supervision_loss']:.4f}"
            logger.info(log_msg)
            
            # Early stopping
            current_loss = epoch_stats['total_loss']
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info(f"  ✓ New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"\n⚠ Early stopping at epoch {epoch+1}")
                    break
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.get('save_every', 50) == 0:
                self.save_checkpoint(model, optimizer, epoch, epoch_stats)
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info("Loaded best model state")
        
        # Save final model and generate embeddings
        self.save_model(model)
        self.generate_quantized_embeddings(model, dataloader, mode)
        
        return {'training_stats': training_stats, 'best_loss': best_loss}
    
    def save_checkpoint(self, model, optimizer, epoch, stats):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(
            self.config['output_dir'], 
            f"checkpoint_epoch_{epoch+1}_loss_{stats['total_loss']:.4f}.pt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': stats,
            'config': self.config
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_model(self, model):
        """Save final model"""
        model_path = os.path.join(self.config['output_dir'], 'final_model.pt')
        
        save_dict = {
            'state_dict': model.state_dict(),
            'config': self.config,
            'input_dim': model.input_dim,
            'latent_dim': model.latent_dim,
            'n_embed': model.n_embed,
            'n_layers': model.n_layers
        }
        
        if hasattr(model, 'collab_dim'):
            save_dict['collab_dim'] = model.collab_dim
        
        torch.save(save_dict, model_path)
        logger.info(f"Final model saved: {model_path}")
    
    def generate_quantized_embeddings(self, model, dataloader, mode):
        """
        Generate and save quantized embeddings (z_hat) for all items.
        
        z_hat is computed as the sum of selected code embeddings from each layer.
        """
        logger.info("Generating quantized embeddings for all items...")
        
        model.eval()
        
        item_ids_all = []
        z_hat_all = []
        codes_all = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                item_ids = batch['item_id'].numpy()
                
                # Get z and z_q based on mode
                if mode == 'gated_dual':
                    semantic = batch['semantic_emb'].to(self.device)
                    collab = batch['collab_emb'].to(self.device)
                    z, _, _ = model.encode(semantic, collab)
                    z_q, _, _, codes, _ = model.quantize(z, temperature=0.2)
                else:
                    x = self.get_input_embedding(batch, mode)
                    z = model.encode(x)
                    z_q, _, _, codes, _ = model.quantize(z, temperature=0.2)
                
                item_ids_all.extend(item_ids.tolist())
                z_hat_all.append(z_q.cpu())
                codes_all.append(codes.cpu())
        
        # Concatenate
        z_hat_all = torch.cat(z_hat_all, dim=0).numpy()
        codes_all = torch.cat(codes_all, dim=0).numpy()
        
        # Create item_id to z_hat mapping
        item_to_z_hat = {int(item_id): z_hat_all[i].tolist() 
                         for i, item_id in enumerate(item_ids_all)}
        
        # Create item_id to codes mapping
        item_to_codes = {int(item_id): codes_all[i].tolist()
                         for i, item_id in enumerate(item_ids_all)}
        
        # Save z_hat embeddings
        z_hat_path = os.path.join(self.config['output_dir'], 'quantized_embeddings.npy')
        np.save(z_hat_path, z_hat_all)
        logger.info(f"Quantized embeddings saved: {z_hat_path}")
        logger.info(f"  Shape: {z_hat_all.shape}")
        
        # Save item_id to z_hat mapping (JSON)
        mapping_path = os.path.join(self.config['output_dir'], 'item_to_z_hat.json')
        with open(mapping_path, 'w') as f:
            json.dump(item_to_z_hat, f)
        logger.info(f"Item to z_hat mapping saved: {mapping_path}")
        
        # Save semantic ID codes
        codes_path = os.path.join(self.config['output_dir'], 'semantic_id_mappings.json')
        with open(codes_path, 'w') as f:
            json.dump(item_to_codes, f, indent=2)
        logger.info(f"Semantic ID mappings saved: {codes_path}")
        
        # Save item_ids order
        item_ids_path = os.path.join(self.config['output_dir'], 'item_ids.npy')
        np.save(item_ids_path, np.array(item_ids_all))
        logger.info(f"Item IDs saved: {item_ids_path}")
        
        # Compute and log statistics
        logger.info("\nQuantized embedding statistics:")
        logger.info(f"  Mean: {z_hat_all.mean():.4f}")
        logger.info(f"  Std: {z_hat_all.std():.4f}")
        logger.info(f"  Min: {z_hat_all.min():.4f}")
        logger.info(f"  Max: {z_hat_all.max():.4f}")
        
        # Compute unique codes
        unique_codes = len(set(tuple(c) for c in codes_all))
        logger.info(f"\nSemantic ID statistics:")
        logger.info(f"  Total items: {len(codes_all)}")
        logger.info(f"  Unique codes: {unique_codes}")
        logger.info(f"  Collision rate: {1 - unique_codes/len(codes_all):.4f}")


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='RQ-VAE Ablation Study')
    
    # Data paths
    parser.add_argument('--semantic_emb_path', type=str, required=True,
                       help='Path to semantic embeddings (parquet), also contains popularity_score')
    parser.add_argument('--collab_emb_path', type=str, default=None,
                       help='Path to collaborative embeddings (npy)')
    parser.add_argument('--popularity_path', type=str, default=None,
                       help='[DEPRECATED] popularity_score is now loaded from semantic_emb_path')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=['semantic_only', 'collab_only', 'concat', 'contrastive', 'gated_dual'],
                       help='Training mode')
    
    # Model hyperparameters
    parser.add_argument('--latent_dim', type=int, default=64,
                       help='Latent dimension (default: 64)')
    parser.add_argument('--n_embed', type=int, default=256,
                       help='Number of codes per layer (default: 256)')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of quantization layers (default: 3)')
    parser.add_argument('--beta', type=float, default=0.25,
                       help='Commitment loss weight (default: 0.25)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping (default: 1.0)')
    
    # Temperature scheduling
    parser.add_argument('--init_temperature', type=float, default=1.0,
                       help='Initial temperature (default: 1.0)')
    parser.add_argument('--min_temperature', type=float, default=0.2,
                       help='Minimum temperature (default: 0.2)')
    
    # Contrastive learning (mode 4)
    parser.add_argument('--contrastive_weight', type=float, default=1.0,
                       help='Contrastive loss weight (default: 1.0)')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                       help='Contrastive temperature (default: 0.07)')
    
    # Gated dual mode (mode 5)
    parser.add_argument('--semantic_recon_weight', type=float, default=1.0,
                       help='Semantic reconstruction weight (default: 1.0)')
    parser.add_argument('--collab_recon_weight', type=float, default=1.0,
                       help='Collab reconstruction weight (default: 1.0)')
    
    # Gate supervision (mode 5)
    parser.add_argument('--use_gate_supervision', action='store_true', default=True,
                       help='Use gate supervision loss (default: True)')
    parser.add_argument('--no_gate_supervision', action='store_false', dest='use_gate_supervision',
                       help='Disable gate supervision')
    parser.add_argument('--gate_supervision_weight', type=float, default=0.1,
                       help='Gate supervision loss weight (default: 0.1)')
    parser.add_argument('--gate_diversity_weight', type=float, default=0.5,
                       help='Gate diversity regularization weight (default: 0.5)')
    parser.add_argument('--gate_target_std', type=float, default=0.2,
                       help='Target std for gate values (default: 0.2)')
    
    # EMA settings
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Use EMA for codebook (default: True)')
    parser.add_argument('--ema_decay', type=float, default=0.99,
                       help='EMA decay rate (default: 0.99)')
    
    # Other
    parser.add_argument('--early_stop_patience', type=int, default=50,
                       help='Early stopping patience (default: 50)')
    parser.add_argument('--save_every', type=int, default=50,
                       help='Save checkpoint every N epochs (default: 50)')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Max items for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers (default: 4)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build config (popularity_score is loaded from semantic_emb_path directly)
    config = {
        'semantic_emb_path': args.semantic_emb_path,
        'collab_emb_path': args.collab_emb_path,
        'output_dir': args.output_dir,
        'mode': args.mode,
        'latent_dim': args.latent_dim,
        'n_embed': args.n_embed,
        'n_layers': args.n_layers,
        'beta': args.beta,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'init_temperature': args.init_temperature,
        'min_temperature': args.min_temperature,
        'contrastive_weight': args.contrastive_weight,
        'contrastive_temperature': args.contrastive_temperature,
        'semantic_recon_weight': args.semantic_recon_weight,
        'collab_recon_weight': args.collab_recon_weight,
        # Gate supervision parameters
        'use_gate_supervision': args.use_gate_supervision,
        'gate_supervision_weight': args.gate_supervision_weight,
        'gate_diversity_weight': args.gate_diversity_weight,
        'gate_target_std': args.gate_target_std,
        'use_ema': args.use_ema,
        'ema_decay': args.ema_decay,
        'early_stop_patience': args.early_stop_patience,
        'save_every': args.save_every,
        'max_items': args.max_items,
        'num_workers': args.num_workers,
        'device': args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'),
        'seed': args.seed
    }
    
    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    trainer = AblationTrainer(config)
    results = trainer.train()
    
    # Save results
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nTraining completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
