"""
Adaptive Temperature Scaling for Hard Negative Mining.

Implements dynamic temperature adjustment based on semantic ID branch density.
The key insight: items sharing longer prefixes are harder to distinguish (Hard Negatives),
requiring smaller temperature to increase penalty on incorrect predictions.

Key Features:
- Pre-computes branch density from Trie structure
- Generates item-specific temperature sequences
- Only applied during training, not inference
- Minimal computational overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AdaptiveTemperatureScaler:
    """Adaptive temperature scaling based on semantic ID branch density.
    """
    
    def __init__(
        self,
        trie,
        semantic_mapper,
        alpha: float = 0.5,
        tau_min: float = 0.1,
        tau_max: float = 2.0,
        mean_center: bool = True,
        k_ref: float = 50.0,
        start_layer: int = 0
    ):
        """Initialize adaptive temperature scaler.
        
        Args:
            trie: SemanticIDTrie instance (already built)
            semantic_mapper: SemanticIDMapper with item_to_codes mapping
            alpha: Sensitivity to branch density (higher = more aggressive)
            tau_min: Minimum temperature (for dense branches, hard negatives)
            tau_max: Maximum temperature (for sparse branches, easy cases)
            mean_center: Whether to scale temperatures to have mean=1.0
            k_ref: Reference density for normalization (dataset-dependent)
            start_layer: Start applying adaptive temperature from this layer (0=all, 1=skip Layer 0)
        """
        self.trie = trie
        self.semantic_mapper = semantic_mapper
        self.alpha = alpha
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.mean_center = mean_center
        self.k_ref = k_ref
        self.start_layer = start_layer
        
        # Pre-compute branch densities and temperature sequences
        self.item_temperatures = {}  # item_id -> List[float] (temperature per layer)
        self._precompute_temperatures()
        
        logger.info(
            f"AdaptiveTemperatureScaler initialized: "
            f"alpha={alpha}, tau_range=[{tau_min}, {tau_max}], "
            f"mean_center={mean_center}, k_ref={k_ref}, start_layer={start_layer}"
        )
    
    def _compute_branch_density(self, prefix: List[int]) -> int:
        """Compute branch density (number of unique tokens following prefix).
        
        Args:
            prefix: List of token IDs representing a prefix
        
        Returns:
            Number of unique tokens that can follow this prefix
        """
        node = self.trie.root
        
        # Navigate to the node for this prefix
        for token_id in prefix:
            node = node.get_child(token_id)
            if node is None:
                return 0  # Invalid prefix
        
        # Return number of children (branch density)
        return len(node.children)
    
    def _compute_temperature(self, branch_density: int) -> float:
        """Compute temperature from branch density.
        Args:
            branch_density: Number of unique tokens following a prefix
        
        Returns:
            Temperature value in [tau_min, tau_max]
        """
        if branch_density == 0:
            # No branching, use max temperature (easiest case)
            return self.tau_max
        
        # Use configurable reference density
        # For Beauty dataset with ~12K items and 256 codebook size, 
        # typical branch density ranges from 1 to ~100
        # Default K_ref = 50.0 works well
        
        # Compute temperature with exponential decay from max to min
        # When K=0: exp(0) = 1 → τ = τ_max
        # When K=K_ref: exp(-α) ≈ 0.6 (for α=0.5) → τ ≈ middle
        # When K→∞: exp(-∞) = 0 → τ = τ_min
        tau_range = self.tau_max - self.tau_min
        decay_factor = np.exp(-self.alpha * branch_density / self.k_ref)
        tau = self.tau_min + tau_range * decay_factor
        
        # Clamp to valid range (should already be in range, but just in case)
        tau = np.clip(tau, self.tau_min, self.tau_max)
        
        return tau
    
    def _precompute_temperatures(self):
        """Pre-compute temperature sequences for all items.
        """
        logger.info("Pre-computing adaptive temperatures for all items...")
        
        num_items = len(self.semantic_mapper.item_to_codes)
        
        for item_id, token_sequence in self.semantic_mapper.item_to_codes.items():
            # Filter out padding tokens
            valid_tokens = [t for t in token_sequence if t != 0]
            
            if not valid_tokens:
                continue
            
            # Compute temperature for each layer
            temperatures = []
            
            for layer_idx in range(len(valid_tokens)):
                # Skip adaptive temperature for early layers if start_layer > 0
                if layer_idx < self.start_layer:
                    # Use standard temperature 1.0 (no scaling)
                    temperatures.append(1.0)
                else:
                    # Get prefix up to (but not including) current layer
                    prefix = valid_tokens[:layer_idx]
                    
                    # Compute branch density at this prefix
                    branch_density = self._compute_branch_density(prefix)
                    
                    # Compute temperature
                    tau = self._compute_temperature(branch_density)
                    temperatures.append(tau)
            
            self.item_temperatures[item_id] = temperatures
        
        # Compute statistics before mean centering
        all_temps = [t for temps in self.item_temperatures.values() for t in temps]
        if all_temps:
            mean_tau_before = np.mean(all_temps)
            std_tau_before = np.std(all_temps)
            min_tau_before = np.min(all_temps)
            max_tau_before = np.max(all_temps)
            
            logger.info(
                f"Temperature statistics (before mean centering): "
                f"mean={mean_tau_before:.3f}, std={std_tau_before:.3f}, "
                f"min={min_tau_before:.3f}, max={max_tau_before:.3f}"
            )
            
            # Log per-layer statistics if start_layer > 0
            if self.start_layer > 0:
                for layer_idx in range(max(len(temps) for temps in self.item_temperatures.values())):
                    layer_temps = [temps[layer_idx] for temps in self.item_temperatures.values() 
                                   if layer_idx < len(temps)]
                    if layer_temps:
                        logger.info(
                            f"  Layer {layer_idx}: mean={np.mean(layer_temps):.3f}, "
                            f"min={np.min(layer_temps):.3f}, max={np.max(layer_temps):.3f}"
                        )
            
            # Apply mean centering if enabled
            if self.mean_center:
                # Scale all temperatures so mean = 1.0
                scaling_factor = 1.0 / mean_tau_before
                
                for item_id in self.item_temperatures:
                    self.item_temperatures[item_id] = [
                        t * scaling_factor for t in self.item_temperatures[item_id]
                    ]
                
                # Recompute statistics after mean centering
                all_temps_centered = [t for temps in self.item_temperatures.values() for t in temps]
                mean_tau_after = np.mean(all_temps_centered)
                std_tau_after = np.std(all_temps_centered)
                min_tau_after = np.min(all_temps_centered)
                max_tau_after = np.max(all_temps_centered)
                
                logger.info(
                    f"Applied mean centering with scaling factor: {scaling_factor:.4f}"
                )
                logger.info(
                    f"Temperature statistics (after mean centering): "
                    f"mean={mean_tau_after:.3f}, std={std_tau_after:.3f}, "
                    f"min={min_tau_after:.3f}, max={max_tau_after:.3f}"
                )
                
                # Log per-layer statistics after mean centering if start_layer > 0
                if self.start_layer > 0:
                    for layer_idx in range(max(len(temps) for temps in self.item_temperatures.values())):
                        layer_temps = [temps[layer_idx] for temps in self.item_temperatures.values() 
                                       if layer_idx < len(temps)]
                        if layer_temps:
                            logger.info(
                                f"  Layer {layer_idx} (centered): mean={np.mean(layer_temps):.3f}, "
                                f"min={np.min(layer_temps):.3f}, max={np.max(layer_temps):.3f}"
                            )
            
            logger.info(f"Pre-computed temperatures for {num_items} items")
    
    def get_temperatures_for_items(
        self,
        item_ids: List[int],
        num_layers: int,
        device: torch.device
    ) -> torch.Tensor:
        """Get temperature sequences for a batch of items.
        
        Args:
            item_ids: List of item IDs (batch_size,)
            num_layers: Number of semantic ID layers
            device: Device for output tensor
        
        Returns:
            Temperature tensor (batch_size, num_layers)
        """
        batch_size = len(item_ids)
        # Default to middle of range for unknown items
        default_temp = (self.tau_min + self.tau_max) / 2.0
        temperatures = torch.ones(batch_size, num_layers, device=device) * default_temp
        
        for batch_idx, item_id in enumerate(item_ids):
            if item_id in self.item_temperatures:
                item_temps = self.item_temperatures[item_id]
                # Pad or truncate to num_layers
                for layer_idx in range(min(len(item_temps), num_layers)):
                    temperatures[batch_idx, layer_idx] = item_temps[layer_idx]
        
        return temperatures
    
    def get_temperature_stats(self) -> Dict[str, float]:
        """Get statistics about temperature distribution.
        
        Returns:
            Dictionary with temperature statistics
        """
        all_temps = [t for temps in self.item_temperatures.values() for t in temps]
        
        if not all_temps:
            return {}
        
        return {
            'mean': float(np.mean(all_temps)),
            'std': float(np.std(all_temps)),
            'min': float(np.min(all_temps)),
            'max': float(np.max(all_temps)),
            'median': float(np.median(all_temps))
        }


class TemperatureScaledCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with adaptive temperature scaling.
    """
    
    def __init__(
        self,
        temperature_scaler: Optional[AdaptiveTemperatureScaler] = None,
        ignore_index: int = -100
    ):
        """Initialize temperature-scaled loss.
        
        Args:
            temperature_scaler: AdaptiveTemperatureScaler instance (optional)
            ignore_index: Index to ignore in loss computation (padding)
        """
        super().__init__()
        self.temperature_scaler = temperature_scaler
        self.ignore_index = ignore_index
        
        if temperature_scaler is not None:
            logger.info("Temperature-scaled cross-entropy loss enabled")
        else:
            logger.info("Standard cross-entropy loss (no temperature scaling)")
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        item_ids: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Compute temperature-scaled cross-entropy loss.
        
        Args:
            logits: Model logits (batch_size, seq_len, vocab_size)
            labels: Target labels (batch_size, seq_len)
            item_ids: Item IDs for temperature lookup (batch_size,)
        
        Returns:
            Scalar loss value
        """
        # If no temperature scaler, use standard cross-entropy
        if self.temperature_scaler is None or item_ids is None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.ignore_index
            )
        
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        
        # Get temperature sequences for this batch
        temperatures = self.temperature_scaler.get_temperatures_for_items(
            item_ids, seq_len, device
        )  # (batch_size, seq_len)
        
        # Apply temperature scaling to logits
        # Expand temperatures to match logits shape
        temperatures_expanded = temperatures.unsqueeze(-1)  # (batch_size, seq_len, 1)
        scaled_logits = logits / temperatures_expanded  # (batch_size, seq_len, vocab_size)
        
        # Compute cross-entropy loss with scaled logits
        loss = F.cross_entropy(
            scaled_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=self.ignore_index
        )
        
        return loss


def create_adaptive_temperature_scaler(
    trie,
    semantic_mapper,
    alpha: float = 0.5,
    tau_min: float = 0.1,
    tau_max: float = 2.0,
    mean_center: bool = True,
    k_ref: float = 50.0,
    start_layer: int = 0
) -> AdaptiveTemperatureScaler:
    """Factory function to create adaptive temperature scaler.
    """
    return AdaptiveTemperatureScaler(
        trie=trie,
        semantic_mapper=semantic_mapper,
        alpha=alpha,
        tau_min=tau_min,
        tau_max=tau_max,
        mean_center=mean_center,
        k_ref=k_ref,
        start_layer=start_layer
    )
