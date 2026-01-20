"""
ActionPiece model implementation.

A T5-based encoder-decoder model for generative recommendation with
ActionPiece tokenization and inference-time ensemble.
"""

import collections
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import numpy as np
from transformers import T5ForConditionalGeneration, T5Config
import logging

logger = logging.getLogger(__name__)


class ActionPieceModel(nn.Module):
    """ActionPiece: T5-based Generative Recommender with ActionPiece tokenization.
    
    This model uses a T5 encoder-decoder architecture with:
    - ActionPiece tokenization for input sequences
    - Inference-time ensemble with multiple SPR augmentations
    - nDCG-based score aggregation for ensemble
    """
    
    def __init__(self, model_config, actionpiece_mapper):
        """Initialize the ActionPiece model.
        
        Args:
            model_config: ModelConfig instance with model hyperparameters
            actionpiece_mapper: ActionPieceMapper instance for tokenization
        """
        super(ActionPieceModel, self).__init__()
        
        self.mapper = actionpiece_mapper
        self.config = model_config
        
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
            eos_token_id=actionpiece_mapper.eos_token,
            decoder_start_token_id=0,
        )
        
        # Initialize T5 model
        self.model = T5ForConditionalGeneration(t5_config)
        
        # Inference ensemble settings
        self.n_inference_ensemble = model_config.n_inference_ensemble
        
        logger.info(f"Initialized ActionPiece model with vocab_size={model_config.vocab_size}")
        logger.info(f"Inference ensemble: {self.n_inference_ensemble}")
        logger.info(self.n_parameters)
    
    @property
    def n_parameters(self) -> str:
        """Calculate the number of trainable parameters."""
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
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            labels: Target token IDs, shape (batch_size, target_len)
        
        Returns:
            Dictionary with 'loss' and 'logits'
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }
    
    def generate_single(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 30,
        max_length: int = 6,
        num_return_sequences: int = 30,
        **kwargs
    ) -> torch.Tensor:
        """Generate recommendations without ensemble.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            num_beams: Number of beams for beam search
            max_length: Maximum generation length
            num_return_sequences: Number of sequences to return
        
        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            **kwargs
        )
    
    def generate_with_ensemble(
        self,
        batch: Dict[str, torch.Tensor],
        num_beams: int = 30,
        num_return_sequences: int = 30,
        n_ensemble: Optional[int] = None
    ) -> torch.Tensor:
        """Generate recommendations with inference-time ensemble.
        
        Uses multiple SPR augmentations and aggregates scores using nDCG weighting.
        
        Args:
            batch: Batch dictionary with 'input_ids', 'attention_mask'
            num_beams: Number of beams for beam search
            num_return_sequences: Number of sequences to return per sample
            n_ensemble: Number of ensemble runs (uses config default if None)
        
        Returns:
            Final predictions, shape (batch_size, num_return_sequences, n_categories)
        """
        if n_ensemble is None:
            n_ensemble = self.n_inference_ensemble
        
        device = batch['input_ids'].device
        batch_size = batch['input_ids'].shape[0]
        n_categories = self.mapper.n_categories
        
        # If ensemble is disabled, use single generation
        if n_ensemble <= 1:
            outputs = self.generate_single(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                num_beams=num_beams,
                max_length=n_categories + 1,
                num_return_sequences=num_return_sequences
            )
            # Decode and reshape
            return self._decode_outputs(outputs, batch_size, 1, num_return_sequences, device)
        
        # For ensemble, we need to re-encode with different SPR augmentations
        # This requires access to the original item sequences
        # For now, we'll use the provided input_ids multiple times
        # In practice, the dataloader should provide multiple augmented versions
        
        all_outputs = []
        for _ in range(n_ensemble):
            outputs = self.generate_single(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                num_beams=num_beams,
                max_length=n_categories + 1,
                num_return_sequences=num_return_sequences
            )
            all_outputs.append(outputs)
        
        # Stack outputs: (n_ensemble, batch_size * num_return_sequences, seq_len)
        all_outputs = torch.stack(all_outputs, dim=0)
        
        # Decode and aggregate
        return self._decode_and_aggregate(
            all_outputs, batch_size, n_ensemble, num_return_sequences, device
        )
    
    def _decode_outputs(
        self,
        outputs: torch.Tensor,
        batch_size: int,
        n_ensemble: int,
        num_return_sequences: int,
        device: torch.device
    ) -> torch.Tensor:
        """Decode generated outputs to feature states.
        
        Args:
            outputs: Generated token IDs
            batch_size: Batch size
            n_ensemble: Number of ensemble runs
            num_return_sequences: Number of sequences per sample
            device: Target device
        
        Returns:
            Decoded states, shape (batch_size, num_return_sequences, n_categories)
        """
        n_categories = self.mapper.n_categories
        
        # Remove decoder start token
        outputs = outputs[:, 1:]
        
        decoded_outputs = []
        for output in outputs.cpu().tolist():
            # Remove EOS token if present
            if self.mapper.eos_token in output:
                idx = output.index(self.mapper.eos_token)
                output = output[:idx]
            else:
                output = output[:n_categories]
            
            # Decode to single state
            decoded = self.mapper.decode_tokens(output)
            if decoded is None:
                decoded_outputs.append([-1] * n_categories)
            else:
                # Convert to token indices
                decoded_outputs.append([self.mapper.actionpiece.rank[f] for f in decoded])
        
        # Reshape to (batch_size, num_return_sequences, n_categories)
        decoded_tensor = torch.tensor(decoded_outputs, dtype=torch.long, device=device)
        return decoded_tensor.view(batch_size, num_return_sequences, n_categories)
    
    def _decode_and_aggregate(
        self,
        all_outputs: torch.Tensor,
        batch_size: int,
        n_ensemble: int,
        num_return_sequences: int,
        device: torch.device
    ) -> torch.Tensor:
        """Decode and aggregate ensemble outputs using nDCG weighting.
        
        Args:
            all_outputs: All generated outputs, shape (n_ensemble, batch_size * num_return_sequences, seq_len)
            batch_size: Batch size
            n_ensemble: Number of ensemble runs
            num_return_sequences: Number of sequences per sample
            device: Target device
        
        Returns:
            Aggregated predictions, shape (batch_size, num_return_sequences, n_categories)
        """
        n_categories = self.mapper.n_categories
        
        # Decode all outputs
        decoded_all = []
        for ensemble_idx in range(n_ensemble):
            outputs = all_outputs[ensemble_idx]
            decoded = self._decode_outputs(
                outputs, batch_size, 1, num_return_sequences, device
            )
            decoded_all.append(decoded)
        
        # Stack: (batch_size, n_ensemble, num_return_sequences, n_categories)
        decoded_all = torch.stack(decoded_all, dim=1)
        
        # Aggregate using nDCG weighting
        final_outputs = torch.full(
            (batch_size, num_return_sequences, n_categories),
            -1,
            dtype=torch.long,
            device=device
        )
        
        for bid in range(batch_size):
            pred2score = collections.defaultdict(float)
            
            for ens_idx in range(n_ensemble):
                for rank_idx in range(num_return_sequences):
                    pred = tuple(decoded_all[bid, ens_idx, rank_idx].tolist())
                    if pred[0] != -1:  # Valid prediction
                        # nDCG-style weighting: 1 / log2(rank + 2)
                        pred2score[pred] += 1 / np.log2(rank_idx + 2)
            
            # Sort by aggregated score
            all_scores = [(pred, score) for pred, score in pred2score.items()]
            all_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Fill final outputs
            for j in range(min(num_return_sequences, len(all_scores))):
                final_outputs[bid, j] = torch.tensor(all_scores[j][0], dtype=torch.long)
        
        return final_outputs
    
    def save_pretrained(self, save_path: str):
        """Save the model."""
        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_pretrained(self, load_path: str):
        """Load the model."""
        self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        logger.info(f"Model loaded from {load_path}")


def create_actionpiece_model(model_config, actionpiece_mapper) -> ActionPieceModel:
    """Create an ActionPiece model.
    
    Args:
        model_config: ModelConfig instance
        actionpiece_mapper: ActionPieceMapper instance
    
    Returns:
        ActionPieceModel instance
    """
    return ActionPieceModel(model_config, actionpiece_mapper)
