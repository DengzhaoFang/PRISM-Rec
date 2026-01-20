"""
Hierarchical Classifiers for PRISM

Implements layer-wise classifiers that predict tag categories from quantized codes.
Each classifier takes concatenated codebook vectors from all previous layers.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict


class LayerClassifier(nn.Module):
    """
    Single-layer classifier for tag prediction.
    Takes concatenated codebook vectors and predicts tag category.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input (concatenated codebook vectors)
            num_classes: Number of tag classes for this layer (including PAD)
            hidden_dim: Hidden layer dimension (default: 2 * input_dim)
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(input_dim * 2, 128)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            
        Returns:
            logits: Class logits (batch_size, num_classes)
        """
        return self.classifier(x)


class HierarchicalClassifiers(nn.Module):
    """
    Hierarchical classifiers for multi-layer tag prediction.
    
    Layer 1: Predicts L2 tags from c1 (codebook dim: 32)
    Layer 2: Predicts L3 tags from [c1; c2] (codebook dim: 64)
    Layer 3: Predicts L4 tags from [c1; c2; c3] (codebook dim: 96)
    """
    
    def __init__(
        self,
        codebook_dim: int,
        num_classes_per_layer: List[int],  # [n_L2_tags, n_L3_tags, n_L4_tags]
        n_layers: int = 3,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            codebook_dim: Dimension of each codebook vector (e.g., 32)
            num_classes_per_layer: Number of classes for each layer (including PAD)
            n_layers: Number of hierarchical layers
            hidden_dim: Hidden dimension for each classifier
            dropout: Dropout probability
        """
        super().__init__()
        self.codebook_dim = codebook_dim
        self.num_classes_per_layer = num_classes_per_layer
        self.n_layers = n_layers
        
        # Create classifiers for each layer
        self.classifiers = nn.ModuleList()
        
        for layer_idx in range(n_layers):
            # Input is concatenation of all codebook vectors up to this layer
            input_dim = codebook_dim * (layer_idx + 1)
            num_classes = num_classes_per_layer[layer_idx]
            
            classifier = LayerClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            
            self.classifiers.append(classifier)
    
    def forward(
        self, 
        quantized_codes: List[torch.Tensor]  # List of (batch_size, codebook_dim)
    ) -> List[torch.Tensor]:
        """
        Forward pass through all hierarchical classifiers.
        
        Args:
            quantized_codes: List of quantized codebook vectors per layer
                [c1 (B, 32), c2 (B, 32), c3 (B, 32)]
                
        Returns:
            predictions: List of class logits per layer
                [logits_L2 (B, n_L2), logits_L3 (B, n_L3), logits_L4 (B, n_L4)]
        """
        predictions = []
        
        for layer_idx in range(self.n_layers):
            # Concatenate all codebook vectors up to this layer
            concat_input = torch.cat(
                quantized_codes[:layer_idx + 1], 
                dim=-1
            )  # (batch_size, codebook_dim * (layer_idx + 1))
            
            # Predict tag for this layer
            logits = self.classifiers[layer_idx](concat_input)
            predictions.append(logits)
        
        return predictions
    
    def predict(
        self, 
        quantized_codes: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Predict tag classes with confidence scores.
        
        Args:
            quantized_codes: List of quantized codebook vectors per layer
            
        Returns:
            predicted_classes: List of predicted class indices per layer
            confidence_scores: List of confidence scores (probabilities) per layer
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(quantized_codes)
            
            predicted_classes = []
            confidence_scores = []
            
            for logits in predictions:
                probs = torch.softmax(logits, dim=-1)
                pred_class = torch.argmax(logits, dim=-1)
                confidence = probs.gather(1, pred_class.unsqueeze(1)).squeeze(1)
                
                predicted_classes.append(pred_class)
                confidence_scores.append(confidence)
        
        return predicted_classes, confidence_scores


def create_hierarchical_classifiers(
    codebook_dim: int,
    tag_stats: Dict[str, int],  # {'n_L2': 6, 'n_L3': 37, 'n_L4': 148}
    n_layers: int = 3,
    include_pad: bool = True,
    **kwargs
) -> HierarchicalClassifiers:
    """
    Factory function to create hierarchical classifiers from tag statistics.
    
    Args:
        codebook_dim: Dimension of codebook vectors
        tag_stats: Dictionary with number of tags per level
        n_layers: Number of layers
        include_pad: Whether to include PAD token in class count
        **kwargs: Additional arguments for HierarchicalClassifiers
        
    Returns:
        classifiers: Initialized HierarchicalClassifiers module
    """
    # Extract number of classes per layer (add 1 for PAD token)
    num_classes_per_layer = [
        tag_stats.get(f'n_L{i+2}', 0) + (1 if include_pad else 0)
        for i in range(n_layers)
    ]
    
    return HierarchicalClassifiers(
        codebook_dim=codebook_dim,
        num_classes_per_layer=num_classes_per_layer,
        n_layers=n_layers,
        **kwargs
    )

