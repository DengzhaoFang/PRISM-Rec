"""
PRISM Loss Functions

Implements specialized loss functions for Hierarchical ID VAE:
1. Cosine Similarity Reconstruction Loss (multi-modal)
2. Tag Anchoring Loss (contrastive learning for semantic guidance)
3. Codebook Balance Loss (contrastive learning to prevent collapse)
4. Hierarchical Classification Loss (with masking)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


class CosineSimilarityLoss(nn.Module):
    """
    Cosine similarity-based reconstruction loss for multi-modal embeddings.
    Scale-invariant and suitable for embeddings of different dimensions.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 1 - cosine_similarity as loss.
        
        Args:
            pred: Predicted embedding (batch_size, dim)
            target: Target embedding (batch_size, dim)
            
        Returns:
            loss: Scalar loss value in range [0, 2]
        """
        # Normalize to unit vectors
        pred_norm = F.normalize(pred, p=2, dim=-1)
        target_norm = F.normalize(target, p=2, dim=-1)
        
        # Cosine similarity: dot product of normalized vectors
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)
        
        # Loss: 1 - similarity (range [0, 2])
        loss = 1.0 - cosine_sim
        
        return loss.mean()


class GateSupervisionLoss(nn.Module):
    """
    Gate supervision loss to align gate values with item popularity.
    
    Encourages:
    - Long-tail items (low popularity) → low gate (don't trust noisy collab signal)
    - Popular items (high popularity) → high gate (trust reliable collab signal)
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
        gate_values: torch.Tensor,      # (batch_size, 768)
        popularity_scores: torch.Tensor  # (batch_size,)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute gate supervision loss.
        
        Args:
            gate_values: Gate values from gate network (batch_size, 768)
            popularity_scores: Ground truth popularity scores (batch_size,)
            
        Returns:
            loss: Supervision loss
            loss_dict: Dictionary with loss components
        """
        # Average gate value per item
        gate_mean = gate_values.mean(dim=1)  # (batch_size,)
        
        # MSE loss between gate and popularity
        supervision_loss = F.mse_loss(gate_mean, popularity_scores)
        
        # Diversity regularization (encourage larger variance)
        gate_var = gate_mean.var()
        diversity_loss = F.relu(self.target_var - gate_var)
        
        # Combined loss
        total_loss = supervision_loss + self.diversity_weight * diversity_loss
        
        loss_dict = {
            'gate_supervision': supervision_loss.item(),
            'gate_diversity': diversity_loss.item(),
            'gate_variance': gate_var.item(),
            'gate_mean': gate_mean.mean().item(),
            'gate_std': gate_mean.std().item()
        }
        
        return self.weight * total_loss, loss_dict


class MultiModalReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss for content and collaborative embeddings.
    Uses cosine similarity to avoid scale conflicts between different modalities.
    
    Supports two modes:
    1. Dual decoder mode (DHR): Separate losses for content and collab
       - Content: reconstruct original content embedding
       - Collab: reconstruct gate-weighted (denoised) collab embedding
    2. Single decoder mode: Loss on concatenated embedding
    """
    
    def __init__(
        self, 
        lambda_content: float = 1.0, 
        lambda_collab: float = 1.0,
        use_dual_decoder: bool = True
    ):
        """
        Args:
            lambda_content: Weight for content embedding reconstruction
            lambda_collab: Weight for collaborative embedding reconstruction
                          (default 1.0, but effective weight is higher due to gradient balancing)
            use_dual_decoder: If True, compute separate losses for content and collab.
                            If False, compute loss on concatenated embedding.
        """
        super().__init__()
        self.lambda_content = lambda_content
        self.lambda_collab = lambda_collab
        self.use_dual_decoder = use_dual_decoder
        self.cosine_loss = CosineSimilarityLoss()
        
    def forward(
        self,
        pred_content: torch.Tensor,
        target_content: torch.Tensor,
        pred_collab: torch.Tensor,
        target_collab: torch.Tensor,
        weighted_collab_target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-modal reconstruction loss.
        
        Args:
            pred_content: Predicted content embedding (batch_size, 768)
            target_content: Target content embedding (batch_size, 768)
            pred_collab: Predicted collaborative embedding (batch_size, 64)
            target_collab: Target collaborative embedding (batch_size, 64) - original collab
            weighted_collab_target: Gate-weighted collab embedding (batch_size, 64)
                                   Used as reconstruction target in DHR mode for consistency
            
        Returns:
            loss: Combined loss scalar
            loss_dict: Dictionary with individual loss components
        """
        if self.use_dual_decoder:
            # Dual decoder mode (DHR): separate losses
            # Content reconstruction loss (reconstruct original content)
            loss_content = self.cosine_loss(pred_content, target_content)
            
            # Collaborative reconstruction loss
            # KEY FIX: Use weighted_collab_target (denoised) as target for consistency
            # The encoder sees weighted_collab, so decoder should reconstruct weighted_collab
            if weighted_collab_target is not None:
                collab_target = weighted_collab_target
            else:
                # Fallback to original collab if weighted not provided
                collab_target = target_collab
            
            loss_collab = self.cosine_loss(pred_collab, collab_target)
            
            # Combined loss with gradient balancing
            # Collab head has fewer parameters, so we boost its weight slightly
            # to ensure balanced gradient flow through shared decoder
            loss_total = (
                self.lambda_content * loss_content + 
                self.lambda_collab * loss_collab
            )
            
            loss_dict = {
                'recon_content': loss_content.item(),
                'recon_collab': loss_collab.item(),
                'recon_total': loss_total.item()
            }
        else:
            # Single decoder mode: loss on concatenated embedding
            # Concatenate predictions and targets
            pred_concat = torch.cat([pred_content, pred_collab], dim=1)  # (B, 832)
            target_concat = torch.cat([target_content, target_collab], dim=1)  # (B, 832)
            
            # Compute loss on concatenated embedding (THIS IS WHAT GETS BACKPROPAGATED)
            loss_concat = self.cosine_loss(pred_concat, target_concat)
            loss_total = loss_concat
            
            # For logging, also compute individual losses (NOT used in backprop, just for monitoring)
            with torch.no_grad():
                loss_content = self.cosine_loss(pred_content, target_content)
                loss_collab = self.cosine_loss(pred_collab, target_collab)
            
            loss_dict = {
                'recon_content': loss_content.item(),  # For monitoring only
                'recon_collab': loss_collab.item(),    # For monitoring only
                'recon_concat': loss_concat.item(),    # ACTUAL loss used for backprop
                'recon_total': loss_total.item()
            }
        
        return loss_total, loss_dict


class SoftAnchorLoss(nn.Module):
    """
    Soft anchoring loss that aligns codebooks with tag semantics while avoiding aggressive
    assignments that destabilise the codebook.
    
    The loss combines:
    - Soft reconstruction: align projected tag embeddings with the weighted average of codes
    - Distribution consistency: keep assignment distributions close to EMA-stabilised targets
    - Entropy-aware adaptive weighting: prevents over-confident assignments that collapse usage
    """
    
    def __init__(
        self,
        tag_embed_dim: int = 768,
        codebook_dim: int = 32,
        n_layers: int = 3,
        beta_weights: Optional[List[float]] = None,
        temperature: float = 0.15,
        ema_decay: float = 0.98,
        distance_weight: float = 1.0,
        distribution_weight: float = 0.5,
        target_entropy_ratio: float = 0.35,
        min_scale: float = 0.25,
        max_scale: float = 1.5,
        eps: float = 1e-6
    ):
        super().__init__()
        self.tag_embed_dim = tag_embed_dim
        self.codebook_dim = codebook_dim
        self.n_layers = n_layers
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.distance_weight = distance_weight
        self.distribution_weight = distribution_weight
        self.target_entropy_ratio = target_entropy_ratio
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.eps = eps
        
        if beta_weights is None:
            beta_weights = [1.0] * n_layers
        self.beta_weights = beta_weights
        
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tag_embed_dim, codebook_dim * 2),
                nn.LayerNorm(codebook_dim * 2),
                nn.ReLU(),
                nn.Linear(codebook_dim * 2, codebook_dim)
            )
            for _ in range(n_layers)
        ])
        
        self._initialized = False
        self._ema_assignments: List[Optional[torch.Tensor]] = []
        self._entropy_targets: List[float] = []
    
    def _maybe_initialize_buffers(
        self,
        tag_embeddings_per_layer: List[torch.Tensor],
        codebooks: List[torch.Tensor]
    ):
        if self._initialized:
            return
        
        device = codebooks[0].device if codebooks else torch.device('cpu')
        self._ema_assignments = []
        self._entropy_targets = []
        
        for layer_idx in range(self.n_layers):
            tag_emb = tag_embeddings_per_layer[layer_idx]
            codebook = codebooks[layer_idx]
            
            if tag_emb is None or codebook is None or len(tag_emb) == 0:
                self._ema_assignments.append(None)
                self._entropy_targets.append(0.0)
                continue
            
            n_tags = tag_emb.size(0)
            n_embed = codebook.size(0)
            
            buffer = torch.zeros(n_tags, n_embed, device=device)
            buffer_name = f'_ema_assign_layer{layer_idx}'
            self.register_buffer(buffer_name, buffer)
            self._ema_assignments.append(getattr(self, buffer_name))
            
            target_entropy = torch.log(torch.tensor(float(n_embed), device=device)).item()
            self._entropy_targets.append(target_entropy * self.target_entropy_ratio)
        
        self._initialized = True
    
    def forward(
        self,
        tag_embeddings_per_layer: List[torch.Tensor],
        codebooks: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        self._maybe_initialize_buffers(tag_embeddings_per_layer, codebooks)
        
        total_loss = torch.tensor(0.0, device=codebooks[0].device if codebooks else 'cpu')
        log_dict: Dict[str, float] = {}
        
        for layer_idx in range(self.n_layers):
            tag_emb = tag_embeddings_per_layer[layer_idx]
            codebook = codebooks[layer_idx]
            
            if tag_emb is None or codebook is None or len(tag_emb) == 0:
                continue
            
            proj_tag_emb = self.projections[layer_idx](tag_emb)
            distances = torch.cdist(proj_tag_emb, codebook, p=2)
            logits = -distances / self.temperature
            assign_probs = torch.softmax(logits, dim=-1)
            
            # EMA-stabilised targets
            ema_buffer = self._ema_assignments[layer_idx]
            if ema_buffer is not None:
                with torch.no_grad():
                    if torch.count_nonzero(ema_buffer).item() == 0:
                        ema_buffer.copy_(assign_probs.detach())
                    else:
                        ema_buffer.mul_(self.ema_decay).add_(
                            assign_probs.detach(), alpha=1.0 - self.ema_decay
                        )
                target_probs = ema_buffer / (ema_buffer.sum(dim=-1, keepdim=True) + self.eps)
            else:
                target_probs = assign_probs.detach()
            
            target_probs = torch.clamp(target_probs, min=self.eps)
            
            soft_positive = torch.matmul(assign_probs, codebook)
            distance_loss = F.smooth_l1_loss(soft_positive, proj_tag_emb)
            kl_loss = F.kl_div(
                torch.log(assign_probs + self.eps),
                target_probs,
                reduction='batchmean'
            )
            
            entropy = -(assign_probs * torch.log(assign_probs + self.eps)).sum(dim=-1).mean()
            target_entropy = torch.tensor(
                self._entropy_targets[layer_idx],
                device=entropy.device
            )
            entropy_scale = torch.clamp(
                target_entropy / (entropy + self.eps),
                min=self.min_scale,
                max=self.max_scale
            ).detach()
            
            layer_loss = (
                self.distance_weight * distance_loss +
                self.distribution_weight * kl_loss
            )
            weighted_layer_loss = self.beta_weights[layer_idx] * entropy_scale * layer_loss
            
            total_loss = total_loss + weighted_layer_loss
            
            log_dict[f'anchor_layer{layer_idx+1}_distance'] = distance_loss.item()
            log_dict[f'anchor_layer{layer_idx+1}_kl'] = kl_loss.item()
            log_dict[f'anchor_layer{layer_idx+1}_entropy'] = entropy.item()
            log_dict[f'anchor_layer{layer_idx+1}_scale'] = entropy_scale.item()
        
        log_dict['anchor_total'] = total_loss.item()
        return total_loss, log_dict


class CodebookBalanceLoss(nn.Module):
    """
    Codebook balance loss using contrastive learning to encourage diversity.
    Prevents codebook collapse by pushing different codebook vectors apart in embedding space.
    Uses contrastive learning: pull together instances of the same code, push apart different codes.
    """
    
    def __init__(
        self,
        n_layers: int = 3,
        gamma_weights: Optional[List[float]] = None,
        temperature: float = 0.07,
        eps: float = 1e-10
    ):
        """
        Args:
            n_layers: Number of RQ layers
            gamma_weights: Weights for each layer's balance loss
            temperature: Temperature parameter for contrastive loss
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.n_layers = n_layers
        self.temperature = temperature
        self.eps = eps
        
        if gamma_weights is None:
            gamma_weights = [1.0] * n_layers
        self.gamma_weights = gamma_weights
        
    def contrastive_balance_loss(
        self, 
        codebook: torch.Tensor, 
        encoding_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss to encourage codebook diversity.
        
        Args:
            codebook: Codebook vectors (n_embed, dim)
            encoding_indices: Selected codebook indices (batch_size,)
            
        Returns:
            loss: Contrastive balance loss
        """
        n_embed = codebook.size(0)
        batch_size = encoding_indices.size(0)
        
        # Normalize codebook vectors
        codebook_norm = F.normalize(codebook, p=2, dim=-1)  # (n_embed, dim)
        
        # Get used codebook vectors
        used_indices = torch.unique(encoding_indices)
        n_used = used_indices.size(0)
        
        if n_used < 2:
            # Need at least 2 different codes for contrastive learning
            return torch.tensor(0.0, device=codebook.device)
        
        # Get embeddings of used codes
        used_codes = codebook_norm[used_indices]  # (n_used, dim)
        
        # Compute pairwise similarities between all used codes
        # Similarity matrix: (n_used, n_used)
        similarity_matrix = torch.matmul(used_codes, used_codes.t())  # (n_used, n_used)
        
        # Create labels: diagonal is positive (same code), off-diagonal is negative (different codes)
        # For contrastive learning, we want to minimize similarity between different codes
        # and maximize similarity of codes with themselves (which is always 1.0 after normalization)
        
        # Create mask for positive pairs (diagonal) and negative pairs (off-diagonal)
        identity_mask = torch.eye(n_used, device=codebook.device, dtype=torch.bool)
        
        # Positive pairs: same code with itself (diagonal, should be 1.0)
        # Negative pairs: different codes (off-diagonal, should be low)
        
        # For contrastive learning, we want:
        # - Positive pairs (same code): high similarity (already 1.0 after normalization)
        # - Negative pairs (different codes): low similarity
        
        # Extract negative similarities (off-diagonal)
        negative_mask = ~identity_mask
        negative_similarities = similarity_matrix[negative_mask]  # (n_used * (n_used - 1),)
        
        # Loss: maximize distance between different codes
        # We want negative similarities to be as low as possible
        # Use a hinge-like loss: max(0, similarity - margin)
        # Or use InfoNCE-style: -log(exp(-sim_neg / temp) / sum(exp(-sim_all / temp)))
        
        # Simple approach: penalize high similarities between different codes
        # Loss = mean of negative similarities (we want this to be low)
        # But we want to push them apart, so we maximize the negative of similarities
        # Or use: loss = mean(negative_similarities) - we want this to be minimized
        
        # Alternative: use a margin-based loss
        margin = 0.5  # Target margin between different codes
        negative_loss = F.relu(negative_similarities - margin).mean()
        
        # Also encourage uniform usage by penalizing over-used codes
        # Count usage frequency
        bincount = torch.bincount(encoding_indices, minlength=n_embed).float()
        usage_freq = bincount[used_indices] / (bincount.sum() + self.eps)  # (n_used,)
        
        # Uniform target: 1 / n_used for each used code
        uniform_target = torch.ones_like(usage_freq) / n_used
        
        # Usage diversity loss: encourage uniform distribution
        usage_loss = F.mse_loss(usage_freq, uniform_target)
        
        # Combine contrastive loss and usage diversity loss
        total_loss = negative_loss + 0.1 * usage_loss
        
        return total_loss
        
    def forward(
        self,
        encoding_indices_per_layer: List[torch.Tensor],  # List of (batch_size,)
        codebooks: List[torch.Tensor],  # List of (n_embed, dim) - need codebooks for contrastive learning
        n_embed_per_layer: List[int]  # List of codebook sizes
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute codebook balance loss for all layers using contrastive learning.
        
        Args:
            encoding_indices_per_layer: List of selected codebook indices per layer
            codebooks: List of codebook tensors for each layer (needed for contrastive learning)
            n_embed_per_layer: List of codebook sizes for each layer
            
        Returns:
            loss: Combined balance loss
            loss_dict: Dictionary with per-layer losses and usage stats
        """
        total_loss = 0.0
        loss_dict = {}
        
        for layer_idx in range(self.n_layers):
            indices = encoding_indices_per_layer[layer_idx]  # (batch_size,)
            codebook = codebooks[layer_idx]  # (n_embed, dim)
            n_embed = n_embed_per_layer[layer_idx]
            
            # Compute contrastive balance loss
            layer_loss = self.contrastive_balance_loss(codebook, indices)
            
            # Track usage statistics
            bincount = torch.bincount(indices, minlength=n_embed).float()
            used_codes = (bincount > 0).sum().item()
            usage_ratio = used_codes / n_embed
            
            # Weighted sum
            total_loss += self.gamma_weights[layer_idx] * layer_loss
            
            loss_dict[f'balance_layer{layer_idx+1}'] = layer_loss.item()
            loss_dict[f'usage_layer{layer_idx+1}'] = usage_ratio
        
        loss_dict['balance_total'] = total_loss.item()
        return total_loss, loss_dict


class HierarchicalClassificationLoss(nn.Module):
    """
    Hierarchical classification loss with masking for variable-length tag sequences.
    Each layer predicts its corresponding tag category.
    """
    
    def __init__(
        self,
        n_layers: int = 3,
        delta_weight: float = 1.0,
        ignore_index: int = 0  # PAD token index
    ):
        """
        Args:
            n_layers: Number of classification layers
            delta_weight: Weight for classification loss
            ignore_index: Index to ignore in loss computation (PAD token)
        """
        super().__init__()
        self.n_layers = n_layers
        self.delta_weight = delta_weight
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(
        self,
        predictions_per_layer: List[torch.Tensor],  # List of (batch, n_classes_l)
        targets_per_layer: List[torch.Tensor],  # List of (batch,)
        masks_per_layer: Optional[List[torch.Tensor]] = None  # List of (batch,)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hierarchical classification loss.
        
        Args:
            predictions_per_layer: List of logits for each layer
            targets_per_layer: List of target tag IDs for each layer
            masks_per_layer: Optional masks (1 for valid, 0 for PAD)
            
        Returns:
            loss: Combined classification loss
            loss_dict: Dictionary with per-layer losses and accuracies
        """
        total_loss = 0.0
        loss_dict = {}
        num_valid_layers = 0
        
        for layer_idx in range(self.n_layers):
            pred = predictions_per_layer[layer_idx]  # (batch, n_classes)
            target = targets_per_layer[layer_idx]  # (batch,)
            
            # Compute cross-entropy loss (automatically handles ignore_index)
            layer_loss = self.ce_loss(pred, target)
            
            # Count valid samples (non-PAD)
            if masks_per_layer is not None:
                mask = masks_per_layer[layer_idx]
                valid_count = mask.sum()
            else:
                valid_count = (target != self.ignore_index).sum()
            
            # Only add loss if there are valid samples
            if valid_count > 0:
                total_loss += layer_loss
                num_valid_layers += 1
                
                # Compute accuracy for valid samples
                with torch.no_grad():
                    pred_labels = torch.argmax(pred, dim=1)
                    if masks_per_layer is not None:
                        mask = masks_per_layer[layer_idx]
                        correct = ((pred_labels == target) & mask.bool()).sum()
                        accuracy = correct.float() / valid_count
                    else:
                        valid_mask = target != self.ignore_index
                        correct = ((pred_labels == target) & valid_mask).sum()
                        accuracy = correct.float() / valid_count
                    
                loss_dict[f'class_layer{layer_idx+1}'] = layer_loss.item()
                loss_dict[f'acc_layer{layer_idx+1}'] = accuracy.item()
        
        # Average over valid layers
        if num_valid_layers > 0:
            total_loss = total_loss / num_valid_layers
        
        # Apply weight
        weighted_loss = self.delta_weight * total_loss
        loss_dict['class_total'] = weighted_loss.item()
        
        return weighted_loss, loss_dict


class PRISMTotalLoss(nn.Module):
    """
    Combined loss function for PRISM training.
    Integrates reconstruction, anchoring, balance, classification, and commitment losses.
    
    Note:
        - Tag Anchoring Loss: Uses contrastive learning (InfoNCE) to align codebook vectors with tag semantics
        - Codebook Balance Loss: Uses contrastive learning to encourage diversity among codebook vectors
    """
    
    def __init__(
        self,
        # Reconstruction loss params
        lambda_content: float = 1.0,
        lambda_collab: float = 1.0,
        use_dual_decoder: bool = True,
        # Anchor loss params
        tag_embed_dim: int = 768,
        codebook_dim: int = 32,
        beta_weights: Optional[List[float]] = None,
        # Balance loss params
        gamma_weights: Optional[List[float]] = None,
        # Classification loss params
        delta_weight: float = 1.0,
        # Commitment loss param
        commitment_weight: float = 0.25,
        # Gate supervision params
        use_gate_supervision: bool = False,
        gate_supervision_weight: float = 0.1,
        gate_diversity_weight: float = 0.5,
        gate_target_std: float = 0.2,
        # General params
        n_layers: int = 3,
        ignore_index: int = 0
    ):
        """
        Initialize combined loss function.
        """
        super().__init__()
        
        # Initialize sub-losses
        self.recon_loss = MultiModalReconstructionLoss(
            lambda_content=lambda_content,
            lambda_collab=lambda_collab,
            use_dual_decoder=use_dual_decoder
        )
        
        self.anchor_loss = SoftAnchorLoss(
            tag_embed_dim=tag_embed_dim,
            codebook_dim=codebook_dim,
            n_layers=n_layers,
            beta_weights=beta_weights
        )
        
        self.balance_loss = CodebookBalanceLoss(
            n_layers=n_layers,
            gamma_weights=gamma_weights
        )
        
        self.class_loss = HierarchicalClassificationLoss(
            n_layers=n_layers,
            delta_weight=delta_weight,
            ignore_index=ignore_index
        )
        
        self.commitment_weight = commitment_weight
        
        # Gate supervision loss
        self.use_gate_supervision = use_gate_supervision
        if use_gate_supervision:
            self.gate_supervision_loss = GateSupervisionLoss(
                weight=gate_supervision_weight,
                diversity_weight=gate_diversity_weight,
                target_std=gate_target_std
            )
        else:
            self.gate_supervision_loss = None
        
    def forward(
        self,
        # Reconstruction inputs
        pred_content: torch.Tensor,
        target_content: torch.Tensor,
        pred_collab: torch.Tensor,
        target_collab: torch.Tensor,
        # Anchor inputs
        tag_embeddings_per_layer: List[torch.Tensor],
        codebooks: List[torch.Tensor],
        # Balance inputs
        encoding_indices_per_layer: List[torch.Tensor],
        n_embed_per_layer: List[int],
        # Classification inputs
        predictions_per_layer: List[torch.Tensor],
        targets_per_layer: List[torch.Tensor],
        masks_per_layer: Optional[List[torch.Tensor]] = None,
        # Commitment loss
        commitment_loss: Optional[torch.Tensor] = None,
        # Gate supervision inputs
        gate_values: Optional[torch.Tensor] = None,
        popularity_scores: Optional[torch.Tensor] = None,
        # DHR reconstruction target
        weighted_collab_target: Optional[torch.Tensor] = None,
        # Optional curriculum weights
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total PRISM loss.
        
        Args:
            pred_content: Predicted content embeddings
            target_content: Target content embeddings
            pred_collab: Predicted collaborative embeddings
            target_collab: Target collaborative embeddings (original)
            tag_embeddings_per_layer: List of tag embeddings for each layer
            codebooks: List of codebook tensors for each layer (used by both anchor and balance losses)
            encoding_indices_per_layer: List of selected codebook indices per layer
            n_embed_per_layer: List of codebook sizes for each layer
            predictions_per_layer: List of classification predictions per layer
            targets_per_layer: List of classification targets per layer
            masks_per_layer: Optional masks for classification
            commitment_loss: Optional commitment loss from VQ
            weighted_collab_target: Gate-weighted collab embedding for DHR reconstruction
            
        Returns:
            total_loss: Combined loss scalar
            loss_dict: Dictionary with all loss components
        """
        # Compute individual losses
        loss_recon, dict_recon = self.recon_loss(
            pred_content, target_content, pred_collab, target_collab,
            weighted_collab_target=weighted_collab_target
        )
        
        loss_anchor, dict_anchor = self.anchor_loss(
            tag_embeddings_per_layer, codebooks
        )
        
        loss_balance, dict_balance = self.balance_loss(
            encoding_indices_per_layer, codebooks, n_embed_per_layer
        )
        
        loss_class, dict_class = self.class_loss(
            predictions_per_layer, targets_per_layer, masks_per_layer
        )
        
        # Curriculum / adaptive weights
        weights = loss_weights or {}
        recon_scale = weights.get('recon', 1.0)
        anchor_scale = weights.get('anchor', 1.0)
        balance_scale = weights.get('balance', 1.0)
        class_scale = weights.get('class', 1.0)
        commit_scale = weights.get('commitment', 1.0)
        
        scaled_recon = recon_scale * loss_recon
        scaled_anchor = anchor_scale * loss_anchor
        scaled_balance = balance_scale * loss_balance
        scaled_class = class_scale * loss_class
        
        # Combine all losses
        total_loss = scaled_recon + scaled_anchor + scaled_balance + scaled_class
        
        # Add commitment loss if provided
        if commitment_loss is not None:
            weighted_commit = commit_scale * self.commitment_weight * commitment_loss
            total_loss += weighted_commit
            dict_commit = {'commitment': commitment_loss.item()}
        else:
            dict_commit = {}
        
        # Add gate supervision loss if enabled
        if self.gate_supervision_loss is not None and gate_values is not None and popularity_scores is not None:
            loss_gate_sup, dict_gate_sup = self.gate_supervision_loss(
                gate_values, popularity_scores
            )
            gate_scale = weights.get('gate_supervision', 1.0)
            total_loss += gate_scale * loss_gate_sup
        else:
            dict_gate_sup = {}
        
        # Merge all dictionaries
        loss_dict = {
            **dict_recon,
            **dict_anchor,
            **dict_balance,
            **dict_class,
            **dict_commit,
            **dict_gate_sup,
            'total_loss': total_loss.item(),
            'scale_recon': recon_scale,
            'scale_anchor': anchor_scale,
            'scale_balance': balance_scale,
            'scale_class': class_scale,
            'scale_commitment': commit_scale
        }
        
        return total_loss, loss_dict

