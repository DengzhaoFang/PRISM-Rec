"""
EAGER model implementation.

A Two-Stream Generative Recommender with Behavior-Semantic Collaboration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class EAGER(nn.Module):
    """EAGER: Two-Stream Generative Recommender.
    
    Architecture:
    1. Shared Encoder: Encodes item ID sequences (User History).
    2. Dual Decoders: Behavior and Semantic decoders for code generation.
    3. Global Contrastive Task (GCT): Aligns summary tokens with item embeddings.
    4. Semantic-Guided Transfer Task (STT): Transfers semantic knowledge to behavior.
    """
    
    def __init__(self, model_config, behavior_emb_path=None, semantic_emb_path=None):
        super(EAGER, self).__init__()
        self.config = model_config
        
        # 1. Embeddings
        self.item_embedding = nn.Embedding(
            model_config.num_items + 1,
            model_config.d_model,
            padding_idx=model_config.pad_token_id
        )
        
        self.token_embedding = nn.Embedding(
            model_config.vocab_size,
            model_config.d_model,
            padding_idx=model_config.pad_token_id
        )
        
        # Fixed embeddings for GCT
        self.behavior_emb_fixed = None
        self.semantic_emb_fixed = None
        self.behavior_projection = None
        self.semantic_projection = None
        
        if behavior_emb_path is not None:
            self._load_behavior_embeddings(behavior_emb_path, model_config)
        
        if semantic_emb_path is not None:
            self._load_semantic_embeddings(semantic_emb_path, model_config)
        
        # 2. T5 Configuration
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
            output_hidden_states=True
        )

        # 3. Dual-Stream Architecture
        self.behavior_model = T5ForConditionalGeneration(t5_config)
        self.semantic_model = T5ForConditionalGeneration(t5_config)
        
        # Share encoder
        self.semantic_model.encoder = self.behavior_model.encoder
        
        # Set embeddings
        self.behavior_model.encoder.set_input_embeddings(self.item_embedding)
        self.behavior_model.decoder.set_input_embeddings(self.token_embedding)
        self.semantic_model.decoder.set_input_embeddings(self.token_embedding)
        
        # Tie LM head weights
        self.behavior_model.lm_head.weight = self.token_embedding.weight
        self.semantic_model.lm_head.weight = self.token_embedding.weight

        # 4. STT Module
        self._init_stt_module(model_config)
        
        logger.info(f"Initialized EAGER model with num_items={model_config.num_items}, vocab_size={model_config.vocab_size}")
    
    def _init_stt_module(self, model_config):
        """Initialize Semantic-Guided Transfer Task module."""
        stt_heads = model_config.num_heads
        if model_config.d_model % stt_heads != 0:
            stt_heads = 4
            logger.warning(f"Adjusting STT heads to {stt_heads} for d_model {model_config.d_model}")

        stt_layer = nn.TransformerDecoderLayer(
            d_model=model_config.d_model,
            nhead=stt_heads,
            dim_feedforward=model_config.d_ff,
            dropout=model_config.dropout_rate,
            batch_first=True,
            norm_first=True
        )
        self.stt_decoder = nn.TransformerDecoder(stt_layer, num_layers=2)
        
        # [CLS] token embedding for STT (learnable)
        self.stt_cls_embedding = nn.Parameter(torch.randn(1, 1, model_config.d_model) * 0.02)
        
        # Output projection for reconstruction (project to embedding space)
        self.stt_output_proj = nn.Linear(model_config.d_model, model_config.d_model)
        
        # Classifier for recognition
        self.stt_classifier = nn.Linear(model_config.d_model, 1)
    
    def _load_behavior_embeddings(self, emb_path, model_config):
        """Load behavior embeddings for GCT."""
        logger.info(f"Loading fixed behavior embeddings from {emb_path}")
        emb_tensor, orig_dim = self._load_embedding_file(
            emb_path, model_config.num_items, model_config.d_model
        )
        self.behavior_emb_fixed = nn.Embedding.from_pretrained(
            emb_tensor, freeze=True, padding_idx=model_config.pad_token_id
        )
        logger.info(f"Loaded behavior embeddings: shape={emb_tensor.shape}")
        
        if orig_dim is not None:
            self.behavior_projection = nn.Linear(orig_dim, model_config.d_model, bias=False)
            logger.info(f"Created behavior projection: {orig_dim} -> {model_config.d_model}")
    
    def _load_semantic_embeddings(self, emb_path, model_config):
        """Load semantic embeddings for GCT."""
        logger.info(f"Loading fixed semantic embeddings from {emb_path}")
        emb_tensor, orig_dim = self._load_embedding_file(
            emb_path, model_config.num_items, model_config.d_model
        )
        self.semantic_emb_fixed = nn.Embedding.from_pretrained(
            emb_tensor, freeze=True, padding_idx=model_config.pad_token_id
        )
        logger.info(f"Loaded semantic embeddings: shape={emb_tensor.shape}")
        
        if orig_dim is not None:
            self.semantic_projection = nn.Linear(orig_dim, model_config.d_model, bias=False)
            logger.info(f"Created semantic projection: {orig_dim} -> {model_config.d_model}")
    
    def _load_embedding_file(self, emb_path: str, num_items: int, d_model: int):
        """Load embedding file and return tensor with optional original dimension."""
        import pandas as pd
        import numpy as np
        
        original_dim = None
        
        if emb_path.endswith('.npy'):
            emb_array = np.load(emb_path)
            logger.info(f"Loaded .npy embeddings: shape={emb_array.shape}")
            if emb_array.shape[0] == num_items + 1:
                emb_array = emb_array[1:]
            if emb_array.shape[1] != d_model:
                original_dim = emb_array.shape[1]
                
        elif emb_path.endswith('.parquet'):
            df = pd.read_parquet(emb_path)
            logger.info(f"Loaded .parquet embeddings: shape={df.shape}")
            emb_list = df['embedding'].tolist()
            emb_array = np.array(emb_list, dtype=np.float32)
            if emb_array.shape[1] != d_model:
                original_dim = emb_array.shape[1]
        else:
            raise ValueError(f"Unsupported embedding file format: {emb_path}")
        
        emb_dim = original_dim if original_dim is not None else d_model
        emb_tensor = torch.zeros((num_items + 1, emb_dim), dtype=torch.float32)
        emb_tensor[1:num_items+1] = torch.from_numpy(emb_array[:num_items])
        
        return emb_tensor, original_dim

    def forward(
        self,
        history_item_ids: torch.Tensor,
        target_behavior_codes: torch.Tensor,
        target_semantic_codes: torch.Tensor,
        target_item_id: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lambda_1: float = 1.0,
        lambda_2: float = 1.0,
        mask_ratio_recon: float = 0.5,
        mask_ratio_recog: float = 0.5,
        num_negatives: int = 64
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass computing all losses."""
        device = history_item_ids.device
        B, S = target_behavior_codes.shape
        
        if attention_mask is None:
            attention_mask = (history_item_ids != self.config.pad_token_id).long()
        
        # 1. Shared Encoder
        encoder_outputs = self.behavior_model.encoder(
            input_ids=history_item_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 2. Dual Decoder Forward (Generation Loss)
        outputs_b = self.behavior_model(
            encoder_outputs=encoder_outputs,
            labels=target_behavior_codes,
            return_dict=True
        )
        outputs_s = self.semantic_model(
            encoder_outputs=encoder_outputs,
            labels=target_semantic_codes,
            return_dict=True
        )
        
        loss_gen = outputs_b.loss + outputs_s.loss
        
        # 3. Global Contrastive Task (GCT)
        summary_b = outputs_b.decoder_hidden_states[-1][:, -1, :]  # (B, D)
        summary_s = outputs_s.decoder_hidden_states[-1][:, -1, :]  # (B, D)
        
        target_behavior_emb = self._get_target_embedding(target_item_id, 'behavior')
        target_semantic_emb = self._get_target_embedding(target_item_id, 'semantic')
        
        loss_con = (F.smooth_l1_loss(summary_b, target_behavior_emb) + 
                    F.smooth_l1_loss(summary_s, target_semantic_emb))
        
        # 4. Semantic-Guided Transfer Task (STT)
        loss_recon, loss_recog = self._compute_stt_loss(
            target_behavior_codes, summary_s, device, B, S,
            mask_ratio_recon, mask_ratio_recog, num_negatives
        )
        
        loss_stt = loss_recon + loss_recog
        total_loss = loss_gen + lambda_1 * loss_con + lambda_2 * loss_stt
        
        return total_loss, {
            "loss_gen": loss_gen.item(),
            "loss_con": loss_con.item(),
            "loss_stt": loss_stt.item(),
            "loss_recon": loss_recon.item(),
            "loss_recog": loss_recog.item()
        }
    
    def _get_target_embedding(self, target_item_id, stream_type):
        """Get target embedding for GCT."""
        if stream_type == 'behavior' and self.behavior_emb_fixed is not None:
            emb = self.behavior_emb_fixed(target_item_id)
            if self.behavior_projection is not None:
                emb = self.behavior_projection(emb)
            return emb
        elif stream_type == 'semantic' and self.semantic_emb_fixed is not None:
            emb = self.semantic_emb_fixed(target_item_id)
            if self.semantic_projection is not None:
                emb = self.semantic_projection(emb)
            return emb
        else:
            return self.item_embedding(target_item_id)
    
    def _compute_stt_loss(self, target_behavior_codes, summary_s, device, B, S,
                          mask_ratio_recon, mask_ratio_recog, num_negatives):
        """Compute STT losses: Reconstruction (InfoNCE) and Recognition (BCE)."""
        
        # ========== Reconstruction Task (Eq. 4 in paper) ==========
        # Add [CLS] token at the beginning: {y^b_[cls], y^b_1, ..., y^b_l}
        # Then mask some tokens and reconstruct using contrastive loss
        
        MASK_TOKEN_ID = self.config.mask_token_id
        
        # Create mask (exclude first position which will be [CLS])
        rand_matrix = torch.rand((B, S), device=device)
        mask_indices = rand_matrix < mask_ratio_recon
        
        # Prepare input with [CLS] token
        stt_input_ids = target_behavior_codes.clone()
        stt_input_ids[mask_indices] = MASK_TOKEN_ID
        stt_input_emb = self.token_embedding(stt_input_ids)  # (B, S, D)
        
        # Prepend [CLS] token
        cls_emb = self.stt_cls_embedding.expand(B, -1, -1)  # (B, 1, D)
        stt_input_with_cls = torch.cat([cls_emb, stt_input_emb], dim=1)  # (B, S+1, D)
        
        # Context: Semantic Summary
        context = summary_s.unsqueeze(1)  # (B, 1, D)
        
        # Run bidirectional STT decoder
        S_with_cls = S + 1
        tgt_mask = torch.zeros((S_with_cls, S_with_cls), device=device, dtype=torch.bool)
        stt_output = self.stt_decoder(
            tgt=stt_input_with_cls,
            memory=context,
            tgt_mask=tgt_mask
        )  # (B, S+1, D)
        
        # Extract outputs for masked positions (skip [CLS] at position 0)
        stt_output_tokens = stt_output[:, 1:, :]  # (B, S, D)
        
        # Project to embedding space
        stt_output_proj = self.stt_output_proj(stt_output_tokens)  # (B, S, D)
        
        # Compute InfoNCE loss for reconstruction (Eq. 4)
        loss_recon = self._compute_infonce_loss(
            stt_output_proj, target_behavior_codes, mask_indices, num_negatives, device
        )
        
        # ========== Recognition Task (Eq. 5 in paper) ==========
        # Construct positive and negative samples
        # Positive: original behavior codes with matching semantic summary
        # Negative: corrupted behavior codes (random token replacement)
        
        loss_recog = self._compute_recognition_loss(
            target_behavior_codes, summary_s, context, device, B, S, mask_ratio_recog
        )
        
        return loss_recon, loss_recog
    
    def _compute_infonce_loss(self, output_proj, target_codes, mask_indices, num_negatives, device):
        """Compute InfoNCE-style contrastive loss for reconstruction (Eq. 4)."""
        if not mask_indices.any():
            return torch.tensor(0.0, device=device)
        
        # Get masked positions' outputs and ground truth
        masked_outputs = output_proj[mask_indices]  # (N_masked, D)
        masked_targets = target_codes[mask_indices]  # (N_masked,)
        
        N_masked = masked_outputs.shape[0]
        if N_masked == 0:
            return torch.tensor(0.0, device=device)
        
        # Get ground truth token embeddings
        gt_embeddings = self.token_embedding(masked_targets)  # (N_masked, D)
        
        # Sample negative tokens (exclude special tokens 0, 1, 2)
        neg_tokens = torch.randint(
            low=3, high=self.config.vocab_size,
            size=(N_masked, num_negatives), device=device
        )
        neg_embeddings = self.token_embedding(neg_tokens)  # (N_masked, num_neg, D)
        
        # Compute similarities
        # Positive: dot product between output and ground truth
        pos_sim = (masked_outputs * gt_embeddings).sum(dim=-1)  # (N_masked,)
        
        # Negative: dot product between output and negatives
        neg_sim = torch.bmm(
            neg_embeddings, masked_outputs.unsqueeze(-1)
        ).squeeze(-1)  # (N_masked, num_neg)
        
        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (N_masked, 1+num_neg)
        labels = torch.zeros(N_masked, dtype=torch.long, device=device)  # Positive is at index 0
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
    def _compute_recognition_loss(self, target_behavior_codes, summary_s, context, 
                                   device, B, S, mask_ratio_recog):
        """Compute recognition loss (Eq. 5): binary classification."""
        # Create positive samples (original codes)
        pos_input_ids = target_behavior_codes.clone()
        pos_input_emb = self.token_embedding(pos_input_ids)
        
        # Create negative samples (corrupted codes)
        neg_input_ids = target_behavior_codes.clone()
        corrupt_mask = torch.rand((B, S), device=device) < mask_ratio_recog
        if corrupt_mask.any():
            num_corrupt = corrupt_mask.sum().item()
            random_tokens = torch.randint(
                low=3, high=self.config.vocab_size,
                size=(num_corrupt,), device=device, dtype=torch.long
            )
            neg_input_ids[corrupt_mask] = random_tokens
        neg_input_emb = self.token_embedding(neg_input_ids)
        
        # Add [CLS] token
        cls_emb = self.stt_cls_embedding.expand(B, -1, -1)
        pos_with_cls = torch.cat([cls_emb, pos_input_emb], dim=1)
        neg_with_cls = torch.cat([cls_emb, neg_input_emb], dim=1)
        
        S_with_cls = S + 1
        tgt_mask = torch.zeros((S_with_cls, S_with_cls), device=device, dtype=torch.bool)
        
        # Run STT decoder for positive samples
        pos_output = self.stt_decoder(
            tgt=pos_with_cls, memory=context, tgt_mask=tgt_mask
        )
        pos_cls_output = pos_output[:, 0, :]  # [CLS] output
        pos_logits = self.stt_classifier(pos_cls_output).squeeze(-1)
        
        # Run STT decoder for negative samples
        neg_output = self.stt_decoder(
            tgt=neg_with_cls, memory=context, tgt_mask=tgt_mask
        )
        neg_cls_output = neg_output[:, 0, :]
        neg_logits = self.stt_classifier(neg_cls_output).squeeze(-1)
        
        # BCE loss: -log(s+) - log(1 - s-)
        pos_labels = torch.ones(B, device=device)
        neg_labels = torch.zeros(B, device=device)
        
        loss_pos = F.binary_cross_entropy_with_logits(pos_logits, pos_labels)
        loss_neg = F.binary_cross_entropy_with_logits(neg_logits, neg_labels)
        
        return loss_pos + loss_neg

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 20,
        max_length: int = 5,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate recommendations using dual-stream beam search."""
        b_out = self.behavior_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length,
            num_return_sequences=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=False,
            **kwargs
        )
        
        s_out = self.semantic_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length,
            num_return_sequences=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=False,
            **kwargs
        )
        
        return b_out.sequences, b_out.sequences_scores, s_out.sequences, s_out.sequences_scores


def create_model(model_config, behavior_emb_path=None, semantic_emb_path=None) -> EAGER:
    """Create an EAGER model."""
    return EAGER(model_config, behavior_emb_path=behavior_emb_path, semantic_emb_path=semantic_emb_path)
