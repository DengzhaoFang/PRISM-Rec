"""
T5-based encoder-decoder model for generative recommendation.

DSI (Dynamic Semantic Integration): 3-way MoE fusion of
  1. ID embeddings (sequence structural signal)
  2. Purified content h_t_hat (128D, MCD-denoised semantics)
  3. Purified collab h_c_hat (128D, MCD-denoised behavior)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Tuple, Dict, List
import logging
import numpy as np

from .moe_fusion import MoEFusion
from .adaptive_temperature import AdaptiveTemperatureScaler, TemperatureScaledCrossEntropyLoss

logger = logging.getLogger(__name__)


class PurifiedSemanticPredictor(nn.Module):
    """Predict target item's z_clean (256D) from decoder hidden states.

    Auxiliary MSE regularization: the decoder's internal representation
    of the generated semantic codes must be predictive of the item's
    underlying purified multimodal features, encouraging semantic grounding.
    """

    def __init__(self, d_model: int = 128, purified_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, purified_dim),
        )

    def forward(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        """decoder_hidden: (B, num_code_tokens, d_model) last-layer decoder states."""
        pooled = decoder_hidden.mean(dim=1)  # (B, d_model)
        return self.predictor(pooled)


class MultiSourceFusion(nn.Module):
    """
    3-way purified fusion for DSI.

    Sources:
      - id_emb:            (B, L, d_model)   sequence structure
      - purified_content:  (B, L, 128)       MCD-denoised semantics
      - purified_collab:   (B, L, 128)       MCD-denoised behavior

    All projected to d_model, fused via learned/attention/fixed gating.
    """

    def __init__(
        self,
        d_model: int,
        purified_dim: int = 128,
        gate_type: str = "learned",
        fixed_weights: Optional[Dict[str, float]] = None,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.gate_type = gate_type
        self.use_residual = use_residual

        self.content_proj = nn.Linear(purified_dim, d_model)
        self.collab_proj = nn.Linear(purified_dim, d_model)
        nn.init.xavier_uniform_(self.content_proj.weight, gain=0.5)
        nn.init.zeros_(self.content_proj.bias)
        nn.init.xavier_uniform_(self.collab_proj.weight, gain=0.5)
        nn.init.zeros_(self.collab_proj.bias)
        self.content_norm = nn.LayerNorm(d_model)
        self.collab_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        if use_residual:
            self.fusion_alpha = nn.Parameter(torch.tensor(-2.0))

        if gate_type == "learned":
            self.gate_fc1 = nn.Linear(d_model * 3, d_model)
            self.gate_fc2 = nn.Linear(d_model, 3)
            nn.init.xavier_uniform_(self.gate_fc1.weight, gain=0.5)
            nn.init.zeros_(self.gate_fc1.bias)
            nn.init.xavier_uniform_(self.gate_fc2.weight, gain=0.5)
            self.gate_fc2.bias.data[0] = 1.0
            self.gate_fc2.bias.data[1] = 0.0
            self.gate_fc2.bias.data[2] = 0.0
            self.gate_dropout = nn.Dropout(dropout)
        elif gate_type == "attention":
            self.query_proj = nn.Linear(d_model, d_model)
            self.key_proj = nn.Linear(d_model, d_model)
            self.value_proj = nn.Linear(d_model, d_model)
            nn.init.xavier_uniform_(self.query_proj.weight, gain=0.5)
            nn.init.xavier_uniform_(self.key_proj.weight, gain=0.5)
            nn.init.xavier_uniform_(self.value_proj.weight, gain=0.5)
        elif gate_type == "fixed":
            if fixed_weights is None:
                fixed_weights = {'id': 0.5, 'content': 0.25, 'collab': 0.25}
            self.register_buffer('fixed_weights', torch.tensor([
                fixed_weights['id'], fixed_weights['content'], fixed_weights['collab']
            ]))

    def forward(
        self,
        id_emb: torch.Tensor,
        purified_content: torch.Tensor,
        purified_collab: torch.Tensor,
    ) -> torch.Tensor:
        content_proj = self.content_norm(self.content_proj(purified_content))
        collab_proj = self.collab_norm(self.collab_proj(purified_collab))

        if self.gate_type == "learned":
            concat = torch.cat([id_emb, content_proj, collab_proj], dim=-1)
            gate_hidden = F.relu(self.gate_fc1(concat))
            gate_hidden = self.gate_dropout(gate_hidden)
            gates = torch.sigmoid(self.gate_fc2(gate_hidden))
            fused = (gates[..., 0:1] * id_emb + gates[..., 1:2] * content_proj +
                     gates[..., 2:3] * collab_proj)
        elif self.gate_type == "attention":
            sources = torch.stack([id_emb, content_proj, collab_proj], dim=2)
            query = self.query_proj(id_emb).unsqueeze(2)
            key = self.key_proj(sources)
            value = self.value_proj(sources)
            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_model ** 0.5)
            fused = torch.matmul(torch.sigmoid(scores), value).squeeze(2)
        elif self.gate_type == "fixed":
            w = self.fixed_weights.view(1, 1, 3)
            fused = w[..., 0]*id_emb + w[..., 1]*content_proj + w[..., 2]*collab_proj
        else:
            fused = id_emb + content_proj + collab_proj

        if self.use_residual and hasattr(self, 'fusion_alpha'):
            alpha = torch.sigmoid(self.fusion_alpha)
            return id_emb + alpha * (fused - id_emb)
        return fused

    def get_gate_weights_stats(self, id_emb, purified_content, purified_collab) -> Dict[str, float]:
        with torch.no_grad():
            content_proj = self.content_norm(self.content_proj(purified_content))
            collab_proj = self.collab_norm(self.collab_proj(purified_collab))

            if self.gate_type == "learned":
                concat = torch.cat([id_emb, content_proj, collab_proj], dim=-1)
                gate_hidden = F.relu(self.gate_fc1(concat))
                gate_logits = self.gate_fc2(gate_hidden)
                raw_weights = torch.sigmoid(gate_logits)
                weights = raw_weights / (raw_weights.sum(dim=-1, keepdim=True) + 1e-8)
                avg = weights.mean(dim=(0, 1))
                return {'id': avg[0].item(), 'content': avg[1].item(), 'collab': avg[2].item()}
            elif self.gate_type == "fixed":
                return {'id': self.fixed_weights[0].item(),
                        'content': self.fixed_weights[1].item(),
                        'collab': self.fixed_weights[2].item()}
        return {}


class ItemLayerEmbedding(nn.Module):
    """Item and layer position embeddings for hierarchical semantic IDs."""

    def __init__(self, d_model: int, max_items: int = 20, num_layers: int = 3,
                 use_temporal_decay: bool = True, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_items = max_items
        self.num_layers = num_layers
        self.use_temporal_decay = use_temporal_decay

        self.item_pos_emb = nn.Embedding(max_items, d_model)
        self.layer_emb = nn.Embedding(num_layers, d_model)
        if use_temporal_decay:
            self.temporal_decay = nn.Parameter(torch.zeros(max_items, d_model))
            nn.init.normal_(self.temporal_decay, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.item_pos_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.layer_emb.weight, mean=0.0, std=0.01)

    def forward(self, token_embeddings: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = token_embeddings.shape
        device = token_embeddings.device
        enhanced_emb = token_embeddings.clone()

        if attention_mask is not None:
            for b in range(batch_size):
                mask = attention_mask[b]
                non_padding = torch.where(mask > 0)[0]
                if len(non_padding) == 0:
                    continue
                start_pos = non_padding[0].item()
                end_pos = non_padding[-1].item() + 1
                content_len = end_pos - start_pos
                content_positions = torch.arange(content_len, device=device)
                item_indices = content_positions // self.num_layers
                layer_indices = content_positions % self.num_layers
                item_emb = self.item_pos_emb(item_indices)
                layer_emb = self.layer_emb(layer_indices)
                enhanced_emb[b, start_pos:end_pos] = enhanced_emb[b, start_pos:end_pos] + item_emb + layer_emb
                if self.use_temporal_decay:
                    enhanced_emb[b, start_pos:end_pos] = enhanced_emb[b, start_pos:end_pos] + self.temporal_decay[item_indices]

        enhanced_emb = self.layer_norm(enhanced_emb)
        enhanced_emb = self.dropout(enhanced_emb)
        if attention_mask is not None:
            enhanced_emb = enhanced_emb * attention_mask.unsqueeze(-1)
        return enhanced_emb


class TIGER(nn.Module):
    """TIGER: T5-based Generative Recommender with DSI purified fusion."""

    def __init__(self, model_config, training_config=None):
        super(TIGER, self).__init__()
        self.model_config = model_config
        self.training_config = training_config

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

        self.model = T5ForConditionalGeneration(t5_config)
        self.config = model_config

        self.use_multimodal_fusion = training_config and training_config.use_multimodal_fusion
        self.use_item_layer_emb = training_config and training_config.use_item_layer_emb
        self.use_adaptive_temperature = training_config and training_config.use_adaptive_temperature

        self.temperature_scaler = None
        self.temperature_loss = None

        # DSI: 3-way purified fusion
        if self.use_multimodal_fusion:
            fusion_gate_type = training_config.fusion_gate_type
            purified_dim = getattr(training_config, 'purified_dim', 128)

            if fusion_gate_type in ("moe", "dense"):
                num_experts = getattr(training_config, 'moe_num_experts', 3)
                expert_hidden_dim = getattr(training_config, 'moe_expert_hidden_dim', 256)
                top_k = getattr(training_config, 'moe_top_k', 2)
                use_load_balancing = getattr(training_config, 'moe_use_load_balancing', False)
                load_balance_weight = getattr(training_config, 'moe_load_balance_weight', 0.001)
                router_type = "dense" if fusion_gate_type == "dense" else "sparse"
                use_teacher_gate = getattr(training_config, 'use_teacher_gate', False)
                teacher_dim = getattr(training_config, 'teacher_dim', 832)

                self.fusion_module = MoEFusion(
                    d_model=model_config.d_model,
                    purified_dim=purified_dim,
                    num_experts=num_experts,
                    expert_hidden_dim=expert_hidden_dim,
                    top_k=top_k,
                    use_load_balancing=use_load_balancing,
                    load_balance_weight=load_balance_weight,
                    dropout=model_config.dropout_rate,
                    use_residual=True,
                    router_type=router_type,
                    use_teacher_gate=use_teacher_gate,
                    teacher_dim=teacher_dim,
                )
                tag = "Dense Softmax" if router_type == "dense" else f"Sparse Top-{top_k}"
                if use_teacher_gate:
                    tag += " +TeacherGate"
                logger.info(f"DSI MoE [{tag}]: {num_experts} experts, hidden={expert_hidden_dim}")
            else:
                fixed_weights = None
                if fusion_gate_type == "fixed":
                    fixed_weights = {'id': 0.5, 'content': 0.25, 'collab': 0.25}

                self.fusion_module = MultiSourceFusion(
                    d_model=model_config.d_model,
                    purified_dim=purified_dim,
                    gate_type=fusion_gate_type,
                    fixed_weights=fixed_weights,
                    dropout=model_config.dropout_rate,
                    use_residual=True,
                )
                logger.info(f"DSI Fusion: gate_type={fusion_gate_type}")

        if self.use_item_layer_emb:
            max_items = 20
            use_temporal_decay = getattr(training_config, 'use_temporal_decay', True)
            self.item_layer_embedding = ItemLayerEmbedding(
                d_model=model_config.d_model, max_items=max_items,
                num_layers=model_config.num_code_layers,
                use_temporal_decay=use_temporal_decay, dropout=model_config.dropout_rate
            )
            self.pos_emb_scale = nn.Parameter(torch.tensor(0.1))
            logger.info(f"Item/layer embeddings enabled")

        # Teacher alignment: carry stage1 recommendation signal into stage2
        self.lambda_align = getattr(training_config, 'lambda_align', 0.0)
        self.use_teacher = (use_teacher_gate if 'use_teacher_gate' in dir() else False) or self.lambda_align > 0
        if self.use_teacher:
            teacher_dim = getattr(training_config, 'teacher_dim', 832)
            self.teacher_to_model = nn.Sequential(
                nn.Linear(teacher_dim, model_config.d_model),
                nn.LayerNorm(model_config.d_model),
                nn.GELU(),
            )
            if self.lambda_align > 0:
                self.teacher_align_proj = nn.Linear(model_config.d_model, model_config.d_model)
            logger.info(f"Teacher alignment: lambda_align={self.lambda_align}, teacher_dim={teacher_dim}")

        self._init_purified_predictor(training_config)

        logger.info(f"Initialized TIGER: vocab={model_config.vocab_size}, d_model={model_config.d_model}")
        logger.info(self.n_parameters)

    def _init_purified_predictor(self, training_config):
        self.use_purified_predictor = getattr(training_config, 'use_purified_predictor', False)
        if self.use_purified_predictor:
            p_dim = getattr(training_config, 'purified_dim', 128)
            z_clean_dim = p_dim * 2  # z_clean = [h_t_hat || h_c_hat]
            self.purified_predictor = PurifiedSemanticPredictor(
                d_model=self.model_config.d_model,
                purified_dim=z_clean_dim,
                hidden_dim=z_clean_dim,
                dropout=self.model_config.dropout_rate,
            )
            self.purified_predictor_weight = getattr(training_config, 'purified_predictor_weight', 0.1)
            logger.info(f"PurifiedSemanticPredictor: d_model={self.model_config.d_model} → {z_clean_dim}D, weight={self.purified_predictor_weight}")

    def init_adaptive_temperature(self, trie, semantic_mapper, alpha=0.5, tau_min=0.1, tau_max=2.0,
                                   mean_center=True, k_ref=50.0, start_layer=0):
        if not self.use_adaptive_temperature:
            return
        logger.info("Initializing adaptive temperature scaler...")
        self.temperature_scaler = AdaptiveTemperatureScaler(
            trie=trie, semantic_mapper=semantic_mapper, alpha=alpha,
            tau_min=tau_min, tau_max=tau_max, mean_center=mean_center,
            k_ref=k_ref, start_layer=start_layer
        )
        self.temperature_loss = TemperatureScaledCrossEntropyLoss(
            temperature_scaler=self.temperature_scaler, ignore_index=-100
        )

    def broadcast_item_to_tokens(self, item_embeddings, item_ids, num_tokens_per_item):
        batch_size, max_items, emb_dim = item_embeddings.shape
        broadcasted = item_embeddings.unsqueeze(2).expand(-1, -1, num_tokens_per_item, -1)
        return broadcasted.reshape(batch_size, max_items * num_tokens_per_item, emb_dim)

    @property
    def n_parameters(self) -> str:
        def count(p): return sum(p.numel() for p in p if p.requires_grad)
        total = count(self.parameters())
        emb = count(self.model.get_input_embeddings().parameters())
        return f"  Embedding params: {emb:,}\n  Non-embedding: {total - emb:,}\n  Total: {total:,}"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        purified_content: Optional[torch.Tensor] = None,
        purified_collab: Optional[torch.Tensor] = None,
        codebook_zq: Optional[torch.Tensor] = None,
        target_z_clean: Optional[torch.Tensor] = None,
        item_ids: Optional[List[int]] = None,
        teacher: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> Dict[str, torch.Tensor]:
        id_emb = self.model.get_input_embeddings()(input_ids)

        if self.use_item_layer_emb:
            pos_enhanced = self.item_layer_embedding(id_emb, attention_mask)
            id_emb = id_emb + self.pos_emb_scale * (pos_enhanced - id_emb)

        fusion_stats = None
        if self.use_multimodal_fusion and purified_content is not None and purified_collab is not None:
            num_tokens = self.model_config.num_code_layers
            content_bc = self.broadcast_item_to_tokens(purified_content, None, num_tokens)
            collab_bc = self.broadcast_item_to_tokens(purified_collab, None, num_tokens)
            codebook_bc = None
            if codebook_zq is not None:
                codebook_bc = self.broadcast_item_to_tokens(codebook_zq, None, num_tokens)

            if isinstance(self.fusion_module, MoEFusion):
                fused_emb, fusion_stats = self.fusion_module(
                    id_emb, content_bc, collab_bc,
                    attention_mask=attention_mask, return_stats=True,
                    teacher=teacher, codebook_emb=codebook_bc,
                )
            else:
                fused_emb = self.fusion_module(id_emb, content_bc, collab_bc)

            outputs = self.model(inputs_embeds=fused_emb, attention_mask=attention_mask,
                                 labels=labels, output_hidden_states=self.use_purified_predictor)
        else:
            outputs = self.model(inputs_embeds=id_emb, attention_mask=attention_mask,
                                 labels=labels, output_hidden_states=self.use_purified_predictor)

        if self.use_adaptive_temperature and self.temperature_loss is not None and labels is not None and item_ids is not None:
            logits = outputs.logits
            main_loss = self.temperature_loss(logits, labels, item_ids)
        else:
            main_loss = outputs.loss

        total_loss = main_loss

        # Purified Semantic Predictor auxiliary loss (cosine, direction only)
        pred_loss = None
        if self.use_purified_predictor and target_z_clean is not None and labels is not None:
            decoder_hidden = outputs.decoder_hidden_states[-1]  # (B, num_code_tokens, d_model)
            pred_z_clean = self.purified_predictor(decoder_hidden)
            pred_loss = 1.0 - F.cosine_similarity(pred_z_clean, target_z_clean, dim=-1).mean()
            total_loss = total_loss + self.purified_predictor_weight * pred_loss

        # MoE load balancing & entropy regularization (anti-collapse)
        if fusion_stats is not None:
            if 'load_balance_loss' in fusion_stats:
                lb_loss = fusion_stats['load_balance_loss']
                if lb_loss is not None:
                    total_loss = total_loss + lb_loss
            if 'entropy_penalty' in fusion_stats:
                ent_pen = fusion_stats['entropy_penalty']
                if ent_pen is not None:
                    total_loss = total_loss + ent_pen

        # Teacher alignment loss: keep stage2 fused representation aligned with stage1 teacher
        teacher_align_loss = None
        if (self.lambda_align > 0 and teacher is not None
                and hasattr(self, 'teacher_align_proj') and fusion_stats is not None):
            num_ct = self.model_config.num_code_layers
            item_repr = fused_emb[:, -num_ct:, :].mean(dim=1)  # (B, d_model)
            item_aligned = self.teacher_align_proj(item_repr)
            teacher_aligned = self.teacher_to_model(teacher)
            teacher_align_loss = 1.0 - F.cosine_similarity(
                item_aligned, teacher_aligned.detach(), dim=-1
            ).mean()
            total_loss = total_loss + self.lambda_align * teacher_align_loss

        result = {'loss': total_loss, 'logits': outputs.logits, 'main_loss': main_loss}
        if pred_loss is not None:
            result['pred_loss'] = pred_loss.item()
        if teacher_align_loss is not None:
            result['teacher_align_loss'] = teacher_align_loss.item()
        if fusion_stats is not None:
            result['fusion_stats'] = fusion_stats
            if 'load_balance_loss' in fusion_stats and fusion_stats['load_balance_loss'] is not None:
                result['moe_load_balance_loss'] = fusion_stats['load_balance_loss'].item()
            if 'entropy_penalty' in fusion_stats:
                ep = fusion_stats['entropy_penalty']
                result['moe_entropy_penalty'] = float(ep.item() if hasattr(ep, 'item') else ep)
            if 'expert_usage' in fusion_stats:
                result['expert_usage'] = fusion_stats['expert_usage'].detach()

        if return_dict:
            return result
        return result['loss'], result['logits']

    def generate(
        self,
        input_ids, attention_mask=None, num_beams=20, max_length=5,
        purified_content=None, purified_collab=None,
        logits_processor=None, **kwargs
    ) -> torch.Tensor:
        id_emb = self.model.get_input_embeddings()(input_ids)

        if self.use_item_layer_emb:
            pos_enhanced = self.item_layer_embedding(id_emb, attention_mask)
            id_emb = id_emb + self.pos_emb_scale * (pos_enhanced - id_emb)

        if self.use_multimodal_fusion and purified_content is not None and purified_collab is not None:
            num_tokens = self.model_config.num_code_layers
            content_bc = self.broadcast_item_to_tokens(purified_content, None, num_tokens)
            collab_bc = self.broadcast_item_to_tokens(purified_collab, None, num_tokens)

            if isinstance(self.fusion_module, MoEFusion):
                fused_emb, _ = self.fusion_module(id_emb, content_bc, collab_bc,
                                                   attention_mask=attention_mask, return_stats=False)
            else:
                fused_emb = self.fusion_module(id_emb, content_bc, collab_bc)

            encoder_outputs = self.model.encoder(inputs_embeds=fused_emb, attention_mask=attention_mask, return_dict=True)
        else:
            encoder_outputs = self.model.encoder(inputs_embeds=id_emb, attention_mask=attention_mask, return_dict=True)

        from transformers import LogitsProcessorList
        logits_processor_list = LogitsProcessorList()
        if logits_processor is not None:
            logits_processor_list.append(logits_processor)

        generated = self.model.generate(
            encoder_outputs=encoder_outputs, attention_mask=attention_mask,
            max_length=max_length, num_beams=num_beams, num_return_sequences=num_beams,
            logits_processor=logits_processor_list if len(logits_processor_list) > 0 else None,
            **kwargs
        )
        return generated

    def save_pretrained(self, save_path: str):
        self.model.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    def load_pretrained(self, load_path: str):
        self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        logger.info(f"Model loaded from {load_path}")


def create_model(model_config, training_config=None) -> TIGER:
    return TIGER(model_config, training_config)
