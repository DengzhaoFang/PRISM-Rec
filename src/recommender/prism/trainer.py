"""
Training utilities for the recommender model.

Provides trainer class and training loop implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    LambdaLR, 
    ReduceLROnPlateau, 
    ExponentialLR, 
    StepLR,
    CosineAnnealingLR
)
from tqdm import tqdm
import logging
import json
import os
from pathlib import Path
from typing import Dict, Optional
import time
import math

from .metrics import MetricsCalculator, format_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for the TIGER model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        valid_loader,
        test_loader,
        config: Dict,
        device: str = "cuda",
        semantic_mapper=None
    ):
        """Initialize the trainer.
        
        Args:
            model: TIGER model instance
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
            device: Device to use for training
            semantic_mapper: SemanticIDMapper instance (for Trie-constrained decoding)
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.semantic_mapper = semantic_mapper
        
        # Move model to device
        self.model.to(self.device)
        
        # Training config
        self.training_config = config['training']
        self.num_epochs = self.training_config.num_epochs
        self.learning_rate = self.training_config.learning_rate
        self.gradient_clip = self.training_config.gradient_clip
        
        # Setup optimizer
        param_groups = []
        
        # Get embedding parameters
        embedding_table = self.model.model.get_input_embeddings()
        embedding_params = [embedding_table.weight]
        
        param_groups.append({
            'params': embedding_params,
            'lr': self.learning_rate,
            'weight_decay': self.training_config.weight_decay,
            'name': 'embeddings'
        })
        
        # Collect non-embedding parameters (excluding fusion_alpha)
        fusion_alpha_params = []
        non_embedding_params = []
        embedding_param = embedding_table.weight
        
        for name, param in self.model.named_parameters():
            # Skip embedding parameters (already handled above)
            if param is embedding_param:
                continue
            # Handle fusion_alpha separately
            elif hasattr(self.model, 'fusion_module') and hasattr(self.model.fusion_module, 'fusion_alpha'):
                if param is self.model.fusion_module.fusion_alpha:
                    fusion_alpha_params.append(param)
                    continue
            non_embedding_params.append(param)
        
        # Add non-embedding parameters
        if non_embedding_params:
            param_groups.append({
                'params': non_embedding_params,
                'lr': self.learning_rate,
                'weight_decay': self.training_config.weight_decay,
                'name': 'non_embedding'
            })
        
        # Add fusion_alpha with higher LR
        if fusion_alpha_params:
            param_groups.append({
                'params': fusion_alpha_params,
                'lr': self.learning_rate * 10,
                'weight_decay': 0.0,
                'name': 'fusion_alpha'
            })
            logger.info(f"Using separate learning rate for fusion_alpha: {self.learning_rate * 10:.6f}")
        
        self.optimizer = optim.Adam(param_groups)
        # Track global step inside model for aux weight scheduling
        self.model.current_step = 0
        
        # Setup learning rate scheduler
        self.scheduler = self._get_lr_scheduler()
        self.scheduler_type = self.training_config.lr_scheduler
        
        # Log scheduler info
        if self.scheduler_type != 'none':
            logger.info(f"Learning rate scheduler: {self.scheduler_type}")
            if self.scheduler_type == 'warmup_cosine':
                total_steps = len(self.train_loader) * self.num_epochs
                warmup_steps = int(total_steps * self.training_config.warmup_ratio)
                logger.info(f"  Total steps: {total_steps}")
                logger.info(f"  Warmup steps: {warmup_steps}")
                logger.info(f"  Initial LR: {self.learning_rate}")
                logger.info(f"  Min LR: {self.training_config.min_lr}")
            elif self.scheduler_type == 'reduce_on_plateau':
                logger.info(f"  Factor: {self.training_config.lr_decay_factor}")
                logger.info(f"  Patience: {self.training_config.lr_patience} epochs")
                logger.info(f"  Min LR: {self.training_config.min_lr}")
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(
            topk_list=self.training_config.topk_list,
            num_layers=config['model'].num_code_layers
        )
        
        # Trie-constrained decoding (if enabled)
        self.use_trie_constraints = self.training_config.use_trie_constraints if hasattr(self.training_config, 'use_trie_constraints') else False
        self.trie_logits_processor = None
        
        if self.use_trie_constraints and semantic_mapper is not None:
            from .trie_constrained_decoder import SemanticIDTrie, TrieConstrainedLogitsProcessor
            
            logger.info("Building Trie for constrained decoding...")
            self.trie = SemanticIDTrie.from_semantic_mapper(semantic_mapper)
            self.trie_logits_processor = TrieConstrainedLogitsProcessor(
                trie=self.trie,
                pad_token_id=config['model'].pad_token_id,
                eos_token_id=config['model'].eos_token_id,
                num_beams=self.training_config.beam_size
            )
            logger.info("Trie-constrained decoding enabled")
        else:
            self.trie = None
        
        # Adaptive temperature scaling (if enabled)
        self.use_adaptive_temperature = self.training_config.use_adaptive_temperature if hasattr(self.training_config, 'use_adaptive_temperature') else False
        
        if self.use_adaptive_temperature and semantic_mapper is not None:
            # Build Trie if not already built
            if self.trie is None:
                from .trie_constrained_decoder import SemanticIDTrie
                logger.info("Building Trie for adaptive temperature...")
                self.trie = SemanticIDTrie.from_semantic_mapper(semantic_mapper)
            
            # Initialize adaptive temperature in model
            tau_alpha = getattr(self.training_config, 'tau_alpha', 0.5)
            tau_min = getattr(self.training_config, 'tau_min', 0.1)
            tau_max = getattr(self.training_config, 'tau_max', 2.0)
            tau_mean_center = getattr(self.training_config, 'tau_mean_center', True)
            tau_k_ref = getattr(self.training_config, 'tau_k_ref', 50.0)
            tau_start_layer = getattr(self.training_config, 'tau_start_layer', 0)
            
            self.model.init_adaptive_temperature(
                trie=self.trie,
                semantic_mapper=semantic_mapper,
                alpha=tau_alpha,
                tau_min=tau_min,
                tau_max=tau_max,
                mean_center=tau_mean_center,
                k_ref=tau_k_ref,
                start_layer=tau_start_layer
            )
            logger.info(f"Adaptive temperature scaling enabled (start_layer={tau_start_layer})")
        
        # Store semantic mapper for item_id lookup
        self.semantic_mapper = semantic_mapper
        
        # Output directories
        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.early_stop_counter = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'valid_metrics': [],
            'test_metrics': [],
            'best_epoch': 0
        }
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _get_lr_scheduler(self):
        """Get learning rate scheduler based on configuration.
        
        Returns:
            Learning rate scheduler or None
        """
        scheduler_type = self.training_config.lr_scheduler
        
        if scheduler_type == 'none':
            return None
        
        elif scheduler_type == 'warmup_cosine':
            # Warmup + Cosine Annealing (Recommended)
            # This is widely used in transformer training
            total_steps = len(self.train_loader) * self.num_epochs
            warmup_steps = int(total_steps * self.training_config.warmup_ratio)
            
            def lr_lambda(step):
                if step < warmup_steps:
                    # Linear warmup
                    return float(step) / float(max(1, warmup_steps))
                else:
                    # Cosine annealing
                    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                    min_lr_ratio = self.training_config.min_lr / self.learning_rate
                    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        elif scheduler_type == 'reduce_on_plateau':
            # Reduce learning rate when validation metric plateaus
            # This is adaptive and doesn't require knowing total steps
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',  # For metrics like NDCG (higher is better)
                factor=self.training_config.lr_decay_factor,
                patience=self.training_config.lr_patience,
                verbose=True,
                min_lr=self.training_config.min_lr
            )
        
        elif scheduler_type == 'exponential':
            # Exponential decay
            return ExponentialLR(
                self.optimizer,
                gamma=self.training_config.lr_gamma
            )
        
        elif scheduler_type == 'step':
            # Step decay (reduce LR every N epochs)
            return StepLR(
                self.optimizer,
                step_size=self.training_config.lr_step_size,
                gamma=self.training_config.lr_decay_factor
            )
        
        else:
            raise ValueError(
                f"Unknown lr_scheduler type: {scheduler_type}. "
                f"Available types: 'none', 'warmup_cosine', 'reduce_on_plateau', 'exponential', 'step'"
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_codebook_loss = 0.0
        total_tag_loss = 0.0
        num_batches = 0
        
        # Reset MOE load balance loss counter for this epoch
        self.total_moe_lb_loss = 0.0
        
        # Check if using multi-modal features
        use_multimodal = self.training_config.use_multimodal_fusion
        use_codebook_pred = self.training_config.use_codebook_prediction
        use_tag_pred = self.training_config.use_tag_prediction
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Expose global step for aux weight scheduling
            if hasattr(self.model, 'current_step'):
                self.model.current_step = self.global_step
            
            # Move batch to device
            input_ids = batch['history'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['target'].to(self.device)
            
            # Get item IDs for adaptive temperature
            item_ids = None
            if self.use_adaptive_temperature and 'target_item_id' in batch:
                # target_item_id is the target item ID
                item_ids = batch['target_item_id'].tolist() if torch.is_tensor(batch['target_item_id']) else batch['target_item_id']
            
            # Prepare multi-modal inputs if enabled
            content_embs = None
            collab_embs = None
            history_codebook_vecs = None
            target_codebook_vecs = None
            target_tag_ids = None
            
            if use_multimodal and 'history_content_embs' in batch:
                content_embs = batch['history_content_embs'].to(self.device)
                collab_embs = batch['history_collab_embs'].to(self.device)
                # Add history_codebook_vecs if available
                if 'history_codebook_vecs' in batch:
                    history_codebook_vecs = batch['history_codebook_vecs'].to(self.device)
            
            if use_codebook_pred and 'target_codebook_vecs' in batch:
                target_codebook_vecs = batch['target_codebook_vecs'].to(self.device)
            
            if use_tag_pred and 'target_tag_ids' in batch:
                target_tag_ids = torch.tensor(batch['target_tag_ids']).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model output (returns dict if using enhanced features)
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                content_embs=content_embs,
                collab_embs=collab_embs,
                history_codebook_vecs=history_codebook_vecs,
                target_codebook_vecs=target_codebook_vecs,
                target_tag_ids=target_tag_ids,
                item_ids=item_ids,
                return_dict=True
            )
            
            # Extract losses
            if isinstance(output, dict):
                loss = output['loss']
                main_loss = output.get('main_loss', loss)
                codebook_loss = output.get('codebook_loss', 0.0)
                tag_loss = output.get('tag_loss', 0.0)
                moe_load_balance_loss = output.get('moe_load_balance_loss', 0.0)
            else:
                loss, _ = output
                main_loss = loss
                codebook_loss = 0.0
                tag_loss = 0.0
                moe_load_balance_loss = 0.0
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Step-wise scheduler update (for warmup_cosine, exponential, etc.)
            # Note: ReduceLROnPlateau is stepped after validation
            if self.scheduler is not None and self.scheduler_type not in ['reduce_on_plateau']:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_main_loss += main_loss.item() if isinstance(main_loss, torch.Tensor) else main_loss
            if isinstance(codebook_loss, torch.Tensor):
                total_codebook_loss += codebook_loss.item()
            if isinstance(tag_loss, torch.Tensor):
                total_tag_loss += tag_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Track MOE load balance loss separately for monitoring
            total_moe_lb_loss = getattr(self, 'total_moe_lb_loss', 0.0)
            if isinstance(moe_load_balance_loss, (int, float)) and moe_load_balance_loss > 0:
                total_moe_lb_loss += moe_load_balance_loss
                self.total_moe_lb_loss = total_moe_lb_loss
            
            # Update progress bar
            postfix = {
                'loss': loss.item(),
                'avg': total_loss / num_batches
            }
            if use_codebook_pred and total_codebook_loss > 0:
                postfix['cb'] = total_codebook_loss / num_batches
            if use_tag_pred and total_tag_loss > 0:
                postfix['tag'] = total_tag_loss / num_batches
            
            # Show MOE load balance loss if using MOE fusion
            if use_multimodal and hasattr(self, 'total_moe_lb_loss') and self.total_moe_lb_loss > 0:
                postfix['moe_lb'] = self.total_moe_lb_loss / num_batches
            
            # Show fusion alpha if using multimodal fusion with residual
            if use_multimodal and hasattr(self.model, 'fusion_module') and hasattr(self.model.fusion_module, 'fusion_alpha'):
                alpha = torch.sigmoid(self.model.fusion_module.fusion_alpha).item()
                postfix['α'] = f"{alpha:.3f}"
            
            progress_bar.set_postfix(postfix)
            
            # Log periodically
            if self.global_step % self.training_config.log_every_n_steps == 0:
                log_msg = f"Step {self.global_step}: loss={loss.item():.4f}, avg_loss={total_loss / num_batches:.4f}"
                if use_codebook_pred and total_codebook_loss > 0:
                    log_msg += f", codebook_loss={total_codebook_loss / num_batches:.4f}"
                if use_tag_pred and total_tag_loss > 0:
                    log_msg += f", tag_loss={total_tag_loss / num_batches:.4f}"
                # Log MOE load balance loss if using MOE fusion
                if use_multimodal and hasattr(self, 'total_moe_lb_loss') and self.total_moe_lb_loss > 0:
                    log_msg += f", moe_lb_loss={self.total_moe_lb_loss / num_batches:.6f}"
                logger.info(log_msg)
                
                # Log gate weights for learned/attention fusion modes
                if use_multimodal and hasattr(self.model, 'fusion_module'):
                    from src.recommender.prism.moe_fusion import MoEFusion
                    
                    if isinstance(self.model.fusion_module, MoEFusion):
                        # Log MOE routing statistics
                        try:
                            id_emb = self.model.model.get_input_embeddings()(input_ids)
                            if content_embs is not None and collab_embs is not None:
                                # Broadcast to token level
                                num_tokens_per_item = self.model.model_config.num_code_layers
                                content_emb_broadcasted = self.model.broadcast_item_to_tokens(
                                    content_embs, None, num_tokens_per_item
                                )
                                collab_emb_broadcasted = self.model.broadcast_item_to_tokens(
                                    collab_embs, None, num_tokens_per_item
                                )
                                
                                # Broadcast codebook vectors if using improved projection
                                codebook_emb_broadcasted = None
                                if self.model.fusion_module.use_improved_projection and history_codebook_vecs is not None:
                                    batch_size, max_items, n_layers, latent_dim = history_codebook_vecs.shape
                                    codebook_emb_broadcasted = history_codebook_vecs.view(
                                        batch_size, max_items * n_layers, latent_dim
                                    )
                                
                                # Get MOE routing statistics
                                moe_stats = self.model.fusion_module.get_routing_stats(
                                    id_emb, content_emb_broadcasted, collab_emb_broadcasted,
                                    codebook_emb=codebook_emb_broadcasted,
                                    attention_mask=attention_mask
                                )
                                
                                if moe_stats and 'expert_dist' in moe_stats:
                                    expert_dist = moe_stats['expert_dist']
                                    dist_str = ", ".join([f"E{i}={expert_dist[i]:.3f}" for i in range(len(expert_dist))])
                                    logger.info(f"  MOE expert usage: {dist_str}")
                                    
                                    if 'fusion_alpha' in moe_stats:
                                        logger.info(f"  MOE fusion alpha: {moe_stats['fusion_alpha']:.4f}")
                        except Exception as e:
                            logger.warning(f"Failed to compute MOE stats: {e}")  # Changed from debug to warning
                    
                    elif hasattr(self.model.fusion_module, 'gate_type'):
                        fusion_gate_type = self.model.fusion_module.gate_type
                        if fusion_gate_type in ["learned", "attention"]:
                            # Get a sample batch to compute weight statistics
                            try:
                                id_emb = self.model.model.get_input_embeddings()(input_ids)
                                if content_embs is not None and collab_embs is not None:
                                    # Broadcast to token level
                                    num_tokens_per_item = self.model.model_config.num_code_layers
                                    content_emb_broadcasted = self.model.broadcast_item_to_tokens(
                                        content_embs, None, num_tokens_per_item
                                    )
                                    collab_emb_broadcasted = self.model.broadcast_item_to_tokens(
                                        collab_embs, None, num_tokens_per_item
                                    )
                                    
                                    # Get weight statistics
                                    weight_stats = self.model.fusion_module.get_gate_weights_stats(
                                        id_emb, content_emb_broadcasted, collab_emb_broadcasted, attention_mask
                                    )
                                    
                                    logger.info(
                                        f"  Gate weights [{fusion_gate_type}]: "
                                        f"ID={weight_stats['id']:.3f}, "
                                        f"Content={weight_stats['content']:.3f}, "
                                        f"Collab={weight_stats['collab']:.3f}"
                                    )
                            except Exception as e:
                                logger.debug(f"Failed to compute gate weights: {e}")
        
        # Return all losses
        result = {
            'total_loss': total_loss / num_batches,
            'main_loss': total_main_loss / num_batches
        }
        if use_codebook_pred and total_codebook_loss > 0:
            result['codebook_loss'] = total_codebook_loss / num_batches
        if use_tag_pred and total_tag_loss > 0:
            result['tag_loss'] = total_tag_loss / num_batches
        
        return result
    
    def evaluate(self, data_loader, split_name: str = "Valid") -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            data_loader: Data loader for evaluation
            split_name: Name of the split (for logging)
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        # Check if using auxiliary tasks
        use_codebook_pred = self.training_config.use_codebook_prediction
        use_tag_pred = self.training_config.use_tag_prediction
        use_multimodal = self.training_config.use_multimodal_fusion
        
        # Accumulators for auxiliary task metrics
        total_codebook_mse = 0.0
        total_tag_acc = 0.0
        num_codebook_samples = 0
        num_tag_samples = 0
        
        verbose = self.training_config.verbose
        
        # For verbose logging: collect all batches first to randomly sample
        all_batches = []
        if verbose:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"VERBOSE MODE: Collecting batches for random sampling")
            logger.info(f"{'=' * 80}")
        
        progress_bar = tqdm(
            data_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [{split_name}]"
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['history'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['target'].to(self.device)
                
                # Get item IDs if available
                item_ids = None
                if 'target_item_id' in batch:
                    item_ids = batch['target_item_id']
                
                # Prepare multi-modal inputs if enabled
                content_embs = None
                collab_embs = None
                history_codebook_vecs = None
                target_codebook_vecs = None
                target_tag_ids = None
                
                if use_multimodal and 'history_content_embs' in batch:
                    content_embs = batch['history_content_embs'].to(self.device)
                    collab_embs = batch['history_collab_embs'].to(self.device)
                    # Add history_codebook_vecs if available
                    if 'history_codebook_vecs' in batch:
                        history_codebook_vecs = batch['history_codebook_vecs'].to(self.device)
                
                if use_codebook_pred and 'target_codebook_vecs' in batch:
                    target_codebook_vecs = batch['target_codebook_vecs'].to(self.device)
                
                if use_tag_pred and 'target_tag_ids' in batch:
                    target_tag_ids = torch.tensor(batch['target_tag_ids']).to(self.device)
                
                # Forward pass for auxiliary tasks
                if use_codebook_pred or use_tag_pred:
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        content_embs=content_embs,
                        collab_embs=collab_embs,
                        history_codebook_vecs=history_codebook_vecs,
                        target_codebook_vecs=target_codebook_vecs,
                        target_tag_ids=target_tag_ids,
                        item_ids=item_ids,  # Pass item_ids for adaptive temperature
                        return_dict=True
                    )
                    
                    # Compute codebook MSE
                    if use_codebook_pred and 'pred_codebook_vecs' in output and target_codebook_vecs is not None:
                        pred_cb = output['pred_codebook_vecs']
                        mse = torch.nn.functional.mse_loss(pred_cb, target_codebook_vecs)
                        total_codebook_mse += mse.item() * pred_cb.size(0)
                        num_codebook_samples += pred_cb.size(0)
                    
                    # Compute tag accuracy
                    if use_tag_pred and 'pred_tag_logits' in output and target_tag_ids is not None:
                        pred_tag_logits = output['pred_tag_logits']
                        # Average accuracy across all layers
                        layer_accs = []
                        for layer_idx, logits in enumerate(pred_tag_logits):
                            pred_tags = torch.argmax(logits, dim=-1)
                            acc = (pred_tags == target_tag_ids[:, layer_idx]).float().mean()
                            layer_accs.append(acc.item())
                        total_tag_acc += sum(layer_accs) / len(layer_accs) * target_tag_ids.size(0)
                        num_tag_samples += target_tag_ids.size(0)
                
                # Generate predictions for main task
                # max_length = num_code_layers + 1 (for decoder start token that will be removed)
                max_gen_length = self.model.model_config.num_code_layers + 1
                
                # CRITICAL FIX: Pass fusion embeddings and Trie constraints to generate
                preds = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=self.training_config.beam_size,
                    max_length=max_gen_length,
                    content_embs=content_embs,
                    collab_embs=collab_embs,
                    history_codebook_vecs=history_codebook_vecs,
                    logits_processor=self.trie_logits_processor
                )
                
                # Remove start token
                preds = preds[:, 1:]
                
                # Store batch info for verbose logging
                if verbose:
                    batch_info = {
                        'input_ids': input_ids.cpu(),
                        'preds': preds.cpu(),
                        'labels': labels.cpu(),
                        'item_ids': item_ids.cpu() if item_ids is not None else None,
                    }
                    all_batches.append(batch_info)
                
                # Update metrics
                self.metrics_calculator.update(
                    preds, labels, self.training_config.beam_size
                )
        
        # Compute metrics
        metrics = self.metrics_calculator.compute()
        
        # Add auxiliary task metrics
        if use_codebook_pred and num_codebook_samples > 0:
            metrics['codebook_mse'] = total_codebook_mse / num_codebook_samples
        
        if use_tag_pred and num_tag_samples > 0:
            metrics['tag_accuracy'] = total_tag_acc / num_tag_samples
        
        # Verbose logging: randomly sample and print 10 samples
        if verbose and len(all_batches) > 0:
            self._print_verbose_samples(all_batches, split_name)
        
        return metrics
    
    def _print_verbose_samples(self, all_batches, split_name: str):
        """Print randomly sampled examples for verbose logging.
        
        Args:
            all_batches: List of batch information dictionaries
            split_name: Name of the split (for logging)
        """
        import random
        
        # Flatten all samples from all batches
        all_samples = []
        for batch_info in all_batches:
            # Get the minimum batch size to handle potential size mismatches
            batch_size = min(
                batch_info['preds'].size(0),
                batch_info['input_ids'].size(0),
                batch_info['labels'].size(0)
            )
            
            for i in range(batch_size):
                sample = {
                    'input_ids': batch_info['input_ids'][i],
                    'preds': batch_info['preds'][i],
                    'labels': batch_info['labels'][i],
                    'item_id': batch_info['item_ids'][i] if batch_info['item_ids'] is not None else None,
                }
                all_samples.append(sample)
        
        # Randomly sample up to 10 samples
        num_samples = min(10, len(all_samples))
        sampled_indices = random.sample(range(len(all_samples)), num_samples)
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"VERBOSE SAMPLES - {split_name} (Epoch {self.current_epoch + 1})")
        logger.info(f"Randomly sampled {num_samples} examples from {len(all_samples)} total samples")
        logger.info(f"{'=' * 80}\n")
        
        for idx, sample_idx in enumerate(sampled_indices, 1):
            sample = all_samples[sample_idx]
            
            logger.info(f"Sample {idx}:")
            logger.info(f"-" * 40)
            
            # Print item ID if available
            if sample['item_id'] is not None:
                logger.info(f"  Item ID: {sample['item_id'].item()}")
            
            # Print input semantic IDs (history sequence)
            input_seq = sample['input_ids'].tolist()
            # Remove padding tokens (assuming 0 is padding)
            input_seq = [x for x in input_seq if x != 0]
            logger.info(f"  Input (history): {input_seq}")
            
            # Print predictions and ground truth
            pred_seq = sample['preds'].tolist()
            label_seq = sample['labels'].tolist()
            logger.info(f"  Predicted:       {pred_seq}")
            logger.info(f"  Ground Truth:    {label_seq}")
            
            # Check if prediction matches
            is_correct = pred_seq == label_seq
            logger.info(f"  Match: {'✓ YES' if is_correct else '✗ NO'}")
            
            logger.info("")  # Empty line between samples
        
        logger.info(f"{'=' * 80}\n")
    
    def _save_best_model(self, metrics: Dict[str, float]):
        """Save the best model checkpoint.
        
        This method is called immediately when a new best model is found,
        ensuring the best model is always saved regardless of save_every_n_epochs.
        
        Args:
            metrics: Current metrics
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        best_path = self.checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Best model saved to {best_path}")
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save a checkpoint.
        
        Args:
            metrics: Current metrics
            is_best: Whether this is the best checkpoint (kept for backward compatibility,
                     but best model is now saved immediately via _save_best_model)
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # NOTE: Best model is now saved immediately when new best is found
        # via _save_best_model(), so we don't need to save it here again.
        # The is_best parameter is kept for backward compatibility.
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        if self.training_config.keep_last_n_checkpoints <= 0:
            return
        
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-self.training_config.keep_last_n_checkpoints]:
            checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        logger.info(f"Checkpoint loaded. Resuming from epoch {self.current_epoch + 1}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    def train(self):
        """Run the complete training loop."""
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.valid_loader.dataset)}")
        logger.info(f"Test samples: {len(self.test_loader.dataset)}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            logger.info("-" * 80)
            
            # Train for one epoch
            train_losses = self.train_epoch()
            
            # Handle both dict and float return types (backward compatibility)
            if isinstance(train_losses, dict):
                train_loss = train_losses['total_loss']
                self.history['train_loss'].append(train_loss)
                
                # Log all losses
                current_lr = self.optimizer.param_groups[0]['lr']
                log_msg = f"Training - Total: {train_loss:.4f}, Main: {train_losses['main_loss']:.4f}"
                if 'codebook_loss' in train_losses:
                    log_msg += f", Codebook: {train_losses['codebook_loss']:.4f}"
                if 'tag_loss' in train_losses:
                    log_msg += f", Tag: {train_losses['tag_loss']:.4f}"
                log_msg += f" | LR: {current_lr:.6f}"
                logger.info(log_msg)
            else:
                # Backward compatibility
                train_loss = train_losses
                self.history['train_loss'].append(train_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Training loss: {train_loss:.4f} | LR: {current_lr:.6f}")
            
            # Evaluate on validation set
            if (epoch + 1) % self.training_config.eval_every_n_epochs == 0:
                valid_metrics = self.evaluate(self.valid_loader, "Valid")
                self.history['valid_metrics'].append({
                    'epoch': epoch + 1,
                    **valid_metrics
                })
                
                logger.info(f"Validation metrics:")
                logger.info(format_metrics(valid_metrics))
                
                # Log auxiliary task metrics if available
                if 'codebook_mse' in valid_metrics:
                    logger.info(f"  Codebook MSE: {valid_metrics['codebook_mse']:.6f}")
                if 'tag_accuracy' in valid_metrics:
                    logger.info(f"  Tag Accuracy: {valid_metrics['tag_accuracy']:.4f}")
                
                # Log epoch-level gate weight statistics for learned/attention modes
                use_multimodal = self.training_config.use_multimodal_fusion
                if use_multimodal and hasattr(self.model, 'fusion_module'):
                    from src.recommender.prism.moe_fusion import MoEFusion
                    
                    if isinstance(self.model.fusion_module, MoEFusion):
                        # MOE fusion: log fusion alpha
                        logger.info(f"\n  MOE fusion summary (epoch {epoch + 1}):")
                        if hasattr(self.model.fusion_module, 'fusion_alpha'):
                            alpha = torch.sigmoid(self.model.fusion_module.fusion_alpha).item()
                            logger.info(f"    Fusion alpha: {alpha:.4f}")
                    elif hasattr(self.model.fusion_module, 'gate_type'):
                        # Standard fusion: log gate type and alpha
                        fusion_gate_type = self.model.fusion_module.gate_type
                        if fusion_gate_type in ["learned", "attention"]:
                            logger.info(f"\n  Fusion gate weights summary (epoch {epoch + 1}):")
                            logger.info(f"    Gate type: {fusion_gate_type}")
                            # The weights are already logged during training steps
                            # Here we just provide a summary reminder
                            if hasattr(self.model.fusion_module, 'fusion_alpha'):
                                alpha = torch.sigmoid(self.model.fusion_module.fusion_alpha).item()
                                logger.info(f"    Fusion alpha: {alpha:.4f}")
                
                # Check if this is the best model
                current_metric = valid_metrics[self.training_config.early_stopping_metric]
                is_best = current_metric > self.best_metric
                
                # Step ReduceLROnPlateau scheduler based on validation metric
                if self.scheduler is not None and self.scheduler_type == 'reduce_on_plateau':
                    self.scheduler.step(current_metric)
                    # Log current learning rate
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f"Current learning rate: {current_lr:.6f}")
                
                if is_best:
                    self.best_metric = current_metric
                    self.early_stop_counter = 0
                    self.history['best_epoch'] = epoch + 1
                    
                    # Evaluate on test set
                    test_metrics = self.evaluate(self.test_loader, "Test")
                    self.history['test_metrics'].append({
                        'epoch': epoch + 1,
                        **test_metrics
                    })
                    
                    logger.info(f"✓ New best {self.training_config.early_stopping_metric}: {current_metric:.4f}")
                    logger.info(f"Test metrics:")
                    logger.info(format_metrics(test_metrics))
                    
                    # Log auxiliary task metrics if available
                    if 'codebook_mse' in test_metrics:
                        logger.info(f"  Codebook MSE: {test_metrics['codebook_mse']:.6f}")
                    if 'tag_accuracy' in test_metrics:
                        logger.info(f"  Tag Accuracy: {test_metrics['tag_accuracy']:.4f}")
                    
                    # CRITICAL FIX: Save best model immediately when new best is found
                    # Don't wait for the next save_every_n_epochs checkpoint
                    self._save_best_model(valid_metrics)
                else:
                    self.early_stop_counter += 1
                    logger.info(
                        f"No improvement. Early stop counter: "
                        f"{self.early_stop_counter}/{self.training_config.early_stopping_patience}"
                    )
                
                # Save checkpoint
                # CRITICAL FIX: Always save best model immediately when is_best=True
                # Regular checkpoints are saved every N epochs
                if is_best:
                    self.save_checkpoint(valid_metrics, is_best=True)
                elif (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                    self.save_checkpoint(valid_metrics, is_best=False)
                
                # Early stopping
                if self.early_stop_counter >= self.training_config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Training completed
        elapsed_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed_time / 3600:.2f} hours")
        logger.info(f"Best epoch: {self.history['best_epoch']}")
        logger.info(f"Best {self.training_config.early_stopping_metric}: {self.best_metric:.4f}")
        
        # Save final history
        self.save_history()
        
        # Load best model and evaluate on test set
        best_model_path = self.checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            self.load_checkpoint(best_model_path)
            final_test_metrics = self.evaluate(self.test_loader, "Test (Final)")
            logger.info(f"\nFinal test metrics (best model):")
            logger.info(format_metrics(final_test_metrics))
            
            # Save final test metrics
            final_metrics_path = self.output_dir / "final_test_metrics.json"
            with open(final_metrics_path, 'w') as f:
                json.dump(final_test_metrics, f, indent=2)
            logger.info(f"Final test metrics saved to {final_metrics_path}")

