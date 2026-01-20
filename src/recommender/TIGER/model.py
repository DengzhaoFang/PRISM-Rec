"""
TIGER model implementation.

A T5-based encoder-decoder model for generative recommendation.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class TIGER(nn.Module):
    """TIGER: T5-based Generative Recommender.
    
    This model uses a T5 encoder-decoder architecture to generate
    semantic item IDs for recommendation.
    """
    
    def __init__(self, model_config):
        """Initialize the TIGER model.
        
        Args:
            model_config: ModelConfig instance with model hyperparameters
        """
        super(TIGER, self).__init__()
        
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
        
        logger.info(f"Initialized TIGER model with vocab_size={model_config.vocab_size}")
        logger.info(self.n_parameters)
    
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
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            labels: Target token IDs, shape (batch_size, target_len)
        
        Returns:
            Tuple of (loss, logits)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs.loss, outputs.logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 20,
        max_length: int = 5,
        **kwargs
    ) -> torch.Tensor:
        """Generate recommendations using beam search.
        
        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)
            num_beams: Number of beams for beam search
            max_length: Maximum length of generated sequence
            **kwargs: Additional generation arguments
        
        Returns:
            Generated token IDs, shape (batch_size * num_beams, max_length)
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs
        )
    
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


def create_model(model_config) -> TIGER:
    """Create a TIGER model.
    
    Args:
        model_config: ModelConfig instance
    
    Returns:
        TIGER model instance
    """
    return TIGER(model_config)

