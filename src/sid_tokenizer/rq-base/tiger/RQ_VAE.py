import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans

# Import optimized components
try:
    from quantization_strategies import (
        STEQuantization, 
        RotationTrickQuantization, 
        GumbelSoftmaxQuantization
    )
    from distance_functions import SquaredEuclideanDistance
    ADVANCED_QUANT_AVAILABLE = True
except ImportError:
    ADVANCED_QUANT_AVAILABLE = False


class QuantizeMode(Enum):
    """Quantization forward pass modes"""
    STE = "ste"  # Straight-Through Estimator
    ROTATION = "rotation"  # Rotation Trick (better gradients)
    GUMBEL_SOFTMAX = "gumbel_softmax"  # Gumbel Softmax (fully differentiable)


def sample_gumbel(shape: Tuple, device: torch.device, eps: float = 1e-20) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution"""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float, device: torch.device) -> torch.Tensor:
    """Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, device)
    sample = F.softmax(y / temperature, dim=-1)
    return sample


class RQVAEQuantizer(nn.Module):
    """
    Vector quantizer for RQ-VAE with EMA or standard VQ.
    Supports both EMA updates (more stable) and standard VQ with stop-gradient.
    Uses advanced quantization strategies when available.
    """
    
    def __init__(
        self,
        n_embed: int,
        embed_dim: int,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        eps: float = 1e-5,
        quantize_mode: QuantizeMode = QuantizeMode.ROTATION
    ):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.use_ema = use_ema
        self.decay = decay
        self.eps = eps
        self.quantize_mode = quantize_mode
        
        # Initialize advanced quantization strategy if available
        if ADVANCED_QUANT_AVAILABLE:
            if quantize_mode == QuantizeMode.STE:
                self.quant_strategy = STEQuantization()
            elif quantize_mode == QuantizeMode.ROTATION:
                self.quant_strategy = RotationTrickQuantization()
            elif quantize_mode == QuantizeMode.GUMBEL_SOFTMAX:
                self.quant_strategy = GumbelSoftmaxQuantization()
            else:
                self.quant_strategy = None
        else:
            self.quant_strategy = None
        
        if use_ema:
            # EMA codebook (non-trainable, updated with EMA)
            embed = torch.zeros(n_embed, embed_dim)
            self.embedding = nn.Parameter(embed, requires_grad=False)
            nn.init.xavier_normal_(self.embedding.data)
            
            # EMA buffers
            self.register_buffer("embed_avg", embed.clone())
            self.register_buffer("cluster_size", torch.ones(n_embed))
        else:
            # Standard trainable codebook
            self.embedding = nn.Embedding(n_embed, embed_dim)
            nn.init.uniform_(self.embedding.weight.data, -1.0/n_embed, 1.0/n_embed)
        
        # Track initialization state for k-means initialization
        self.register_buffer('_initialized', torch.tensor(False))
        
    def _kmeans_init(self, z: torch.Tensor):
        """
        Initialize codebook using k-means clustering on first training batch.
        Following TIGER paper approach to prevent codebook collapse.
        
        Args:
            z: First batch of input embeddings (batch_size, embed_dim)
        """
        print(f"Initializing codebook with k-means clustering...")
        
        # Convert to numpy for sklearn k-means
        z_np = z.detach().cpu().numpy()
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=self.n_embed, random_state=42, n_init=10)
        kmeans.fit(z_np)
        
        # Use centroids as codebook initialization
        centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=z.device)
        
        # Initialize codebook with k-means centroids
        with torch.no_grad():
            if self.use_ema:
                self.embedding.data.copy_(centroids)
                self.embed_avg.data.copy_(centroids)
                self.cluster_size.data.fill_(1.0)
            else:
                self.embedding.weight.data.copy_(centroids)
            
        print(f"✓ Codebook initialized with k-means centroids")
        print(f"  Codebook range: [{centroids.min().item():.4f}, {centroids.max().item():.4f}]")
        print(f"  Input range: [{z.min().item():.4f}, {z.max().item():.4f}]")
        
        # Mark as initialized
        self._initialized.fill_(True)
        
    def forward(
        self, 
        z: torch.Tensor, 
        temperature: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Forward pass of vector quantizer with EMA or stop-gradient.
        Applies k-means initialization on first batch.
        Uses advanced quantization strategies when available.
        
        Args:
            z: Input tensor of shape (batch_size, embed_dim)
            temperature: Temperature for Gumbel-Softmax (only used in GUMBEL_SOFTMAX mode)
            
        Returns:
            z_q: Quantized tensor (with gradient flow)
            codebook_loss: Codebook loss
            commitment_loss: Commitment loss
            encoding_indices: Quantization indices
            unused_codes: Number of unused codebook entries
        """
        # K-means initialization on first training batch
        if self.training and not self._initialized:
            self._kmeans_init(z)
            
        batch_size = z.shape[0]
        z_flattened = z.view(-1, self.embed_dim)
        
        # Get codebook weights
        if self.use_ema:
            code_embs = self.embedding
        else:
            code_embs = self.embedding.weight
        
        # Use advanced quantization strategy if available
        if self.quant_strategy is not None and not self.use_ema:
            # Advanced quantization with better gradient flow
            z_q_grad, z_q_loss, encoding_indices = self.quant_strategy.quantize(
                z_flattened, code_embs, temperature
            )
            z_q_grad = z_q_grad.view(batch_size, self.embed_dim)
            z_q_loss = z_q_loss.view(batch_size, self.embed_dim)
            
            # Count unused codes
            embed_onehot = F.one_hot(encoding_indices, self.n_embed)
            embed_onehot_sum = embed_onehot.sum(0)
            unused_codes = (embed_onehot_sum == 0).sum().item()
            
            # Compute losses
            codebook_loss = F.mse_loss(z_q_loss, z.detach())
            commitment_loss = F.mse_loss(z, z_q_loss.detach())
            
            return z_q_grad, codebook_loss, commitment_loss, encoding_indices, unused_codes
        else:
            # Standard quantization with EMA or basic STE
            # Calculate L2 distances to find nearest codebook vectors
            if ADVANCED_QUANT_AVAILABLE:
                # Use optimized distance function
                d = SquaredEuclideanDistance.compute(z_flattened, code_embs)
            else:
                # Fallback to original implementation
                d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                    torch.sum(code_embs ** 2, dim=1) - \
                    2 * torch.matmul(z_flattened, code_embs.t())
            
            # Find closest embedding indices
            encoding_indices = torch.argmin(d, dim=1)
            
            # Count unused codes
            embed_onehot = F.one_hot(encoding_indices, self.n_embed)
            embed_onehot_sum = embed_onehot.sum(0)
            unused_codes = (embed_onehot_sum == 0).sum().item()
            
            # Get quantized embeddings
            z_q = F.embedding(encoding_indices, code_embs).view(batch_size, self.embed_dim)
            
            if self.use_ema and self.training:
                # EMA update of codebook
                embed_onehot = embed_onehot.float()
                embed_sum = embed_onehot.t() @ z_flattened
                
                self.cluster_size.data.mul_(self.decay).add_(
                    embed_onehot_sum.float(), alpha=1 - self.decay
                )
                self.embed_avg.data.mul_(self.decay).add_(
                    embed_sum, alpha=1 - self.decay
                )
                
                # Normalize
                n = self.cluster_size.sum()
                norm_w = (
                    n * (self.cluster_size + self.eps) / (n + self.n_embed * self.eps)
                )
                embed_normalized = self.embed_avg / norm_w.unsqueeze(1)
                self.embedding.data.copy_(embed_normalized)
                
                # Loss: only commitment loss for EMA
                codebook_loss = torch.tensor(0.0, device=z.device)
                commitment_loss = F.mse_loss(z, z_q.detach())
            else:
                # Standard VQ loss
                codebook_loss = F.mse_loss(z_q, z.detach())
                commitment_loss = F.mse_loss(z, z_q.detach())
            
            # Straight-through estimator
            z_q = z + (z_q - z).detach()
            
            return z_q, codebook_loss, commitment_loss, encoding_indices, unused_codes


class RQVAEEncoder(nn.Module):
    """Encoder for RQ-VAE - optimized architecture with SiLU activation"""
    
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        
        # Optimized architecture: no bias, SiLU activation for smoother gradients
        # Input -> 512 -> 256 -> 128 -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512, bias=False),
            nn.SiLU(),  # Smoother than ReLU, helps prevent dead neurons
            nn.Linear(512, 256, bias=False), 
            nn.SiLU(),
            nn.Linear(256, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, latent_dim, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class RQVAEDecoder(nn.Module):
    """Decoder for RQ-VAE - optimized architecture with SiLU activation"""
    
    def __init__(self, latent_dim: int = 32, output_dim: int = 768):
        super().__init__()
        
        # Symmetric to encoder: latent_dim -> 128 -> 256 -> 512 -> output_dim
        # No bias, SiLU activation for smoother gradients
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128, bias=False),
            nn.SiLU(),
            nn.Linear(128, 256, bias=False),
            nn.SiLU(), 
            nn.Linear(256, 512, bias=False),
            nn.SiLU(),
            nn.Linear(512, output_dim, bias=False)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class RQVAE(nn.Module):
    """
    Residual Quantized Variational AutoEncoder (RQ-VAE).
    Following TIGER paper implementation with stop-gradient training.
    Uses 3-level residual quantization with 256-dim codebooks.
    
    Key improvements based on src-old:
    - Residual normalization before each quantization layer (critical for stability)
    - Proper EMA cluster tracking
    - Dead code detection and monitoring
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        n_embed: int = 256, 
        n_layers: int = 3,
        beta: float = 0.25,
        use_ema: bool = True,
        decay: float = 0.99,
        commitment_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        quantize_mode: QuantizeMode = QuantizeMode.GUMBEL_SOFTMAX,
        normalize_residuals: bool = True  # EBODA analysis: can be False, but True is safer
    ):
        """
        Initialize RQ-VAE with optimized settings.
        
        Args:
            input_dim: Input feature dimension (e.g., 768 for sentence-t5)
            latent_dim: Latent space dimension (32 recommended)
            n_embed: Number of embeddings per codebook (256 recommended)  
            n_layers: Number of quantization layers (3 recommended)
            beta: Commitment loss weight (0.25 recommended)
            use_ema: Use EMA update for codebook (True recommended, more stable)
            decay: EMA decay rate (0.99 recommended)
            commitment_weight: Weight for total commitment loss
            reconstruction_weight: Weight for reconstruction loss
            quantize_mode: Quantization mode (for non-EMA only)
            normalize_residuals: Whether to L2-normalize residuals before each layer (True recommended)
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_embed = n_embed
        self.n_layers = n_layers
        self.beta = beta
        self.use_ema = use_ema
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.reconstruction_weight = reconstruction_weight
        self.quantize_mode = quantize_mode
        self.normalize_residuals = normalize_residuals
        
        # Encoder and decoder with optimized architecture
        self.encoder = RQVAEEncoder(input_dim, latent_dim)
        self.decoder = RQVAEDecoder(latent_dim, input_dim)
        
        # Residual quantizers
        self.quantizers = nn.ModuleList([
            RQVAEQuantizer(n_embed, latent_dim, beta, use_ema, decay, quantize_mode=quantize_mode)
            for _ in range(n_layers)
        ])
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction"""
        return self.decoder(z)
    
    def quantize(
        self, 
        z: torch.Tensor, 
        temperature: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Apply residual quantization with optional residual normalization.
        
         K-means initialization for all layers on first forward pass.
        Each layer is initialized with its actual input distribution (normalized residuals).
        
        Args:
            z: Latent representation
            temperature: Temperature for Gumbel-Softmax
            
        Returns:
            z_q: Quantized representation  
            codebook_loss: Total codebook loss
            commitment_loss: Total commitment loss
            codes: Quantization codes for each layer (batch_size, n_layers)
            total_unused: Total number of unused codes across all layers
        """
        #  Initialize all layers on first forward pass
        if self.training and not all(q._initialized for q in self.quantizers):
            self._init_all_codebooks(z, temperature)
        
        residual = z
        z_q_total = torch.zeros_like(z)
        codebook_loss_total = 0.0
        commitment_loss_total = 0.0
        total_unused = 0
        codes = []
        
        for quantizer in self.quantizers:
            # CRITICAL: Normalize residuals before quantization
            # This prevents codebook collapse and improves training stability
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)  # L2 normalize along feature dimension
            
            z_q_layer, cb_loss, cm_loss, layer_codes, unused = quantizer(residual, temperature)
            z_q_total += z_q_layer
            residual = residual - z_q_layer
            codebook_loss_total += cb_loss
            commitment_loss_total += cm_loss
            total_unused += unused
            codes.append(layer_codes)
            
        codes = torch.stack(codes, dim=1)  # (batch_size, n_layers)
        return z_q_total, codebook_loss_total, commitment_loss_total, codes, total_unused
    
    def _init_all_codebooks(self, z: torch.Tensor, temperature: float = 0.2):
        """
         Initialize all codebook layers at once on first forward pass.
        Each layer is initialized with k-means on its actual input (normalized residuals).
        
        Args:
            z: Initial latent representation
            temperature: Temperature for forward pass
        """
        print("="*60)
        print("Initializing all RQ-VAE codebooks ...")
        print("="*60)
        
        residual = z
        
        for layer_idx, quantizer in enumerate(self.quantizers):
            if quantizer._initialized:
                continue
            
            # Normalize residual if configured
            if self.normalize_residuals:
                residual_norm = F.normalize(residual, dim=-1)
            else:
                residual_norm = residual
            
            # Initialize this layer's codebook
            print(f"\nLayer {layer_idx}:")
            quantizer._kmeans_init(residual_norm)
            
            # Compute quantized output to get residual for next layer
            with torch.no_grad():
                z_q_layer, _, _, _, _ = quantizer(residual, temperature)
                residual = residual - z_q_layer
        
        print("="*60)
        print("✓ All codebooks initialized!")
        print("="*60)
    
    def calculate_duplicate_rate(self, codes: torch.Tensor) -> float:
        """
        Calculate the rate of duplicate semantic IDs before post-ID processing.
        
        Args:
            codes: Quantization codes of shape (batch_size, n_layers)
            
        Returns:
            duplicate_rate: Fraction of duplicate 3-tuple IDs
        """
        # Convert codes to tuples for uniqueness checking
        codes_np = codes.cpu().numpy()
        unique_codes = set()
        duplicates = 0
        
        for code_seq in codes_np:
            code_tuple = tuple(code_seq)
            if code_tuple in unique_codes:
                duplicates += 1
            else:
                unique_codes.add(code_tuple)
                
        return duplicates / len(codes_np) if len(codes_np) > 0 else 0.0
    
    def apply_post_id_deduplication(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Apply post-ID method to ensure unique 4-tuple semantic IDs.
        Adds a 4th code for items that share the same first 3 codewords.
        
        Args:
            codes: Original 3-layer codes (batch_size, 3)
            
        Returns:
            unique_codes: 4-layer codes with unique IDs (batch_size, 4)
        """
        batch_size = codes.shape[0]
        codes_np = codes.cpu().numpy()
        
        # Track occurrence count for each 3-tuple
        tuple_counts = {}
        unique_codes = np.zeros((batch_size, 4), dtype=np.int64)
        
        for i, code_seq in enumerate(codes_np):
            # Get first 3 codes as tuple
            code_tuple = tuple(code_seq[:3])
            
            # Count occurrences and assign 4th code
            if code_tuple not in tuple_counts:
                tuple_counts[code_tuple] = 0
            else:
                tuple_counts[code_tuple] += 1
            
            # Set first 3 codes and 4th deduplication code
            unique_codes[i, :3] = code_seq[:3]
            unique_codes[i, 3] = tuple_counts[code_tuple]
            
        return torch.tensor(unique_codes, device=codes.device, dtype=torch.long)
    
    def get_codebook_usage_stats(self) -> Dict[str, float]:
        """
        Get detailed usage statistics for each codebook.
        
        Returns:
            stats: Dictionary with usage statistics
        """
        stats = {}
        total_usage = 0
        
        for i, quantizer in enumerate(self.quantizers):
            # Count parameters that have been significantly updated from initialization
            codebook_weights = quantizer.embedding.weight.data
            init_scale = 1.0 / quantizer.n_embed
            
            # Simple heuristic: codes with weights significantly different from uniform init
            used_codes = torch.abs(codebook_weights).mean(dim=1) > init_scale * 0.5
            usage_rate = used_codes.float().mean().item()
            
            stats[f'layer_{i}_usage'] = usage_rate
            total_usage += usage_rate
            
        stats['average_usage'] = total_usage / self.n_layers
        return stats
    
    def forward(
        self, 
        x: torch.Tensor, 
        temperature: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optimized training procedure.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            temperature: Temperature for Gumbel-Softmax (lower = more discrete)
            
        Returns:
            Dictionary with outputs, losses, and statistics
        """
        # Encode to latent space
        z = self.encode(x)
        
        # Apply residual quantization with temperature
        z_q, codebook_loss, commitment_loss, codes, total_unused = self.quantize(z, temperature)
        
        # Decode back to input space
        x_recon = self.decode(z_q)
        
        # Reconstruction loss (MSE, sum over features for per-sample loss)
        recon_loss = ((x_recon - x) ** 2).sum(dim=-1).mean()
        
        # Total VQ loss (weighted combination of codebook and commitment)
        vq_loss = codebook_loss + self.beta * commitment_loss
        
        # Total loss (weighted combination)
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.commitment_weight * vq_loss)
        
        # Calculate codebook usage
        codebook_usage = 1.0 - (total_unused / (self.n_layers * self.n_embed))
        
        # Calculate duplicate rate before post-ID processing
        with torch.no_grad():
            duplicate_rate_pre = self.calculate_duplicate_rate(codes)
            
            # Apply post-ID deduplication
            codes_unique = self.apply_post_id_deduplication(codes)
            duplicate_rate_post = self.calculate_duplicate_rate(codes_unique)
        
        return {
            'x_recon': x_recon,
            'z': z,
            'z_q': z_q,
            'codes': codes,  # Original 3-layer codes
            'codes_unique': codes_unique,  # 4-layer unique codes
            'recon_loss': recon_loss,
            'codebook_loss': codebook_loss,
            'commitment_loss': commitment_loss,
            'vq_loss': vq_loss,
            'total_loss': total_loss,
            'codebook_usage': codebook_usage,
            'duplicate_rate_pre': duplicate_rate_pre,
            'duplicate_rate_post': duplicate_rate_post
        }
    
    def encode_to_codes(
        self, 
        x: torch.Tensor, 
        apply_post_id: bool = True,
        temperature: float = 0.2  # Use same temperature as training min_temperature
    ) -> torch.Tensor:
        """
        Encode input directly to quantization codes.
        
        Args:
            x: Input tensor
            apply_post_id: Whether to apply post-ID deduplication
            temperature: Temperature for quantization (should match training min_temperature for consistency)
            
        Returns:
            codes: Quantization codes (3-layer if apply_post_id=False, 4-layer if True)
        """
        z = self.encode(x)
        _, _, _, codes, _ = self.quantize(z, temperature)
        
        if apply_post_id:
            codes = self.apply_post_id_deduplication(codes)
            
        return codes
    
    def decode_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Decode from quantization codes to reconstruction.
        
        Args:
            codes: Quantization codes (batch_size, n_layers)
                  Can be 3-layer or 4-layer (ignores 4th layer if present)
        """
        batch_size = codes.shape[0]
        z_q = torch.zeros(batch_size, self.latent_dim, device=codes.device)
        
        # Use only first 3 layers (ignore 4th deduplication code)
        n_layers_to_use = min(codes.shape[1], self.n_layers)
        
        for layer_idx in range(n_layers_to_use):
            layer_codes = codes[:, layer_idx]
            layer_embeddings = self.quantizers[layer_idx].embedding(layer_codes)
            z_q += layer_embeddings
            
        return self.decode(z_q)
    
    def save_model(self, path: str):
        """Save the complete model with TIGER paper configuration"""
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'n_embed': self.n_embed,
            'n_layers': self.n_layers,
            'beta': self.beta,
            'use_ema': self.use_ema,
            'decay': self.decay,
            'normalize_residuals': self.normalize_residuals,
        }, path)
    
    def load_model(self, path: str):
        """Load the complete model"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        return self