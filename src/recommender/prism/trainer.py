"""
Training utilities for the recommender model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, ExponentialLR, StepLR, CosineAnnealingLR
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
    """Trainer for the TIGER model with DSI purified fusion."""

    def __init__(self, model, train_loader, valid_loader, test_loader, config, device="cuda", semantic_mapper=None):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.semantic_mapper = semantic_mapper

        self.model.to(self.device)
        self.training_config = config['training']
        self.num_epochs = self.training_config.num_epochs
        self.learning_rate = self.training_config.learning_rate
        self.gradient_clip = self.training_config.gradient_clip

        # Optimizer
        embedding_table = self.model.model.get_input_embeddings()
        embedding_params = [embedding_table.weight]
        param_groups = [{'params': embedding_params, 'lr': self.learning_rate,
                          'weight_decay': self.training_config.weight_decay, 'name': 'embeddings'}]

        fusion_alpha_params = []
        non_embedding_params = []
        for name, param in self.model.named_parameters():
            if param is embedding_table.weight:
                continue
            elif hasattr(self.model, 'fusion_module') and hasattr(self.model.fusion_module, 'fusion_alpha'):
                if param is self.model.fusion_module.fusion_alpha:
                    fusion_alpha_params.append(param)
                    continue
            non_embedding_params.append(param)

        if non_embedding_params:
            param_groups.append({'params': non_embedding_params, 'lr': self.learning_rate,
                                  'weight_decay': self.training_config.weight_decay, 'name': 'non_embedding'})
        if fusion_alpha_params:
            param_groups.append({'params': fusion_alpha_params, 'lr': self.learning_rate * 10,
                                  'weight_decay': 0.0, 'name': 'fusion_alpha'})
            logger.info(f"Fusion alpha LR: {self.learning_rate * 10:.6f}")

        self.optimizer = optim.Adam(param_groups)
        self.model.current_step = 0
        self.scheduler = self._get_lr_scheduler()
        self.scheduler_type = self.training_config.lr_scheduler

        self.metrics_calculator = MetricsCalculator(
            topk_list=self.training_config.topk_list,
            num_layers=config['model'].num_code_layers
        )

        # Trie-constrained decoding
        self.use_trie_constraints = getattr(self.training_config, 'use_trie_constraints', False)
        self.trie_logits_processor = None
        if self.use_trie_constraints and semantic_mapper is not None:
            from .trie_constrained_decoder import SemanticIDTrie, TrieConstrainedLogitsProcessor
            logger.info("Building Trie for constrained decoding...")
            self.trie = SemanticIDTrie.from_semantic_mapper(semantic_mapper)
            self.trie_logits_processor = TrieConstrainedLogitsProcessor(
                trie=self.trie, pad_token_id=config['model'].pad_token_id,
                eos_token_id=config['model'].eos_token_id, num_beams=self.training_config.beam_size
            )
        else:
            self.trie = None

        # Adaptive temperature
        self.use_adaptive_temperature = getattr(self.training_config, 'use_adaptive_temperature', False)
        if self.use_adaptive_temperature and semantic_mapper is not None:
            if self.trie is None:
                from .trie_constrained_decoder import SemanticIDTrie
                self.trie = SemanticIDTrie.from_semantic_mapper(semantic_mapper)
            self.model.init_adaptive_temperature(
                trie=self.trie, semantic_mapper=semantic_mapper,
                alpha=getattr(self.training_config, 'tau_alpha', 0.5),
                tau_min=getattr(self.training_config, 'tau_min', 0.1),
                tau_max=getattr(self.training_config, 'tau_max', 2.0),
                mean_center=getattr(self.training_config, 'tau_mean_center', True),
                k_ref=getattr(self.training_config, 'tau_k_ref', 50.0),
                start_layer=getattr(self.training_config, 'tau_start_layer', 0)
            )

        self.semantic_mapper = semantic_mapper
        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.early_stop_counter = 0
        self.history = {'train_loss': [], 'valid_metrics': [], 'test_metrics': [], 'best_epoch': 0}
        logger.info(f"Trainer initialized on {self.device}")

    def _get_lr_scheduler(self):
        scheduler_type = self.training_config.lr_scheduler
        if scheduler_type == 'none':
            return None
        elif scheduler_type == 'warmup_cosine':
            total_steps = len(self.train_loader) * self.num_epochs
            # Use explicit warmup_steps if set (>0), otherwise compute from warmup_ratio
            if self.training_config.warmup_steps > 0:
                warmup_steps = self.training_config.warmup_steps
            else:
                warmup_steps = int(total_steps * self.training_config.warmup_ratio)
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                min_lr_ratio = self.training_config.min_lr / self.learning_rate
                return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            return LambdaLR(self.optimizer, lr_lambda)
        elif scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(self.optimizer, mode='max', factor=self.training_config.lr_decay_factor,
                                      patience=self.training_config.lr_patience, verbose=True,
                                      min_lr=self.training_config.min_lr)
        elif scheduler_type == 'exponential':
            return ExponentialLR(self.optimizer, gamma=self.training_config.lr_gamma)
        elif scheduler_type == 'step':
            return StepLR(self.optimizer, step_size=self.training_config.lr_step_size,
                           gamma=self.training_config.lr_decay_factor)
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_pred_loss = 0.0
        num_batches = 0
        self.total_moe_lb_loss = 0.0
        use_multimodal = self.training_config.use_multimodal_fusion
        use_predictor = getattr(self.training_config, 'use_purified_predictor', False)

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]")

        for batch in progress_bar:
            if hasattr(self.model, 'current_step'):
                self.model.current_step = self.global_step

            input_ids = batch['history'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['target'].to(self.device)

            item_ids = None
            if self.use_adaptive_temperature and 'target_item_id' in batch:
                item_ids = batch['target_item_id'].tolist() if torch.is_tensor(batch['target_item_id']) else batch['target_item_id']

            purified_content = None
            purified_collab = None
            target_z_clean = None
            teacher = None
            if use_multimodal and 'history_purified_content' in batch:
                purified_content = batch['history_purified_content'].to(self.device)
                purified_collab = batch['history_purified_collab'].to(self.device)
            codebook_zq = None
            if 'history_codebook_zq' in batch:
                codebook_zq = batch['history_codebook_zq'].to(self.device)
            if use_predictor and 'target_z_clean' in batch:
                target_z_clean = batch['target_z_clean'].to(self.device)
            if 'target_teacher' in batch:
                teacher = batch['target_teacher'].to(self.device)

            self.optimizer.zero_grad()

            output = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                purified_content=purified_content, purified_collab=purified_collab,
                codebook_zq=codebook_zq, target_z_clean=target_z_clean,
                item_ids=item_ids, teacher=teacher, return_dict=True
            )

            if isinstance(output, dict):
                loss = output['loss']
                main_loss = output.get('main_loss', loss)
                pred_loss = output.get('pred_loss', None)
                moe_lb_loss = output.get('moe_load_balance_loss', 0.0)
                teacher_align_loss = output.get('teacher_align_loss', None)
            else:
                loss, _ = output
                main_loss = loss
                pred_loss = None
                moe_lb_loss = 0.0
                teacher_align_loss = None

            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            if self.scheduler is not None and self.scheduler_type not in ['reduce_on_plateau']:
                self.scheduler.step()

            total_loss += loss.item()
            total_main_loss += main_loss.item() if isinstance(main_loss, torch.Tensor) else main_loss
            if pred_loss is not None:
                total_pred_loss += pred_loss
            num_batches += 1
            self.global_step += 1

            if isinstance(moe_lb_loss, (int, float)) and moe_lb_loss > 0:
                self.total_moe_lb_loss += moe_lb_loss

            postfix = {'loss': loss.item(), 'avg': total_loss / num_batches}

            if use_predictor and total_pred_loss > 0:
                postfix['pred'] = total_pred_loss / num_batches

            if use_multimodal and hasattr(self, 'total_moe_lb_loss') and self.total_moe_lb_loss > 0:
                postfix['moe_lb'] = self.total_moe_lb_loss / num_batches

            if use_multimodal and hasattr(self.model, 'fusion_module') and hasattr(self.model.fusion_module, 'fusion_alpha'):
                alpha = torch.sigmoid(self.model.fusion_module.fusion_alpha).item()
                postfix['a'] = f"{alpha:.3f}"

            if use_multimodal and hasattr(self.model, 'fusion_module') and \
               getattr(self.model.fusion_module, 'router_type', None) == 'dense':
                ew = output.get('expert_usage')
                if ew is not None:
                    postfix['mod'] = f"{ew[0].item():.2f}/{ew[1].item():.2f}/{ew[2].item():.2f}"

            progress_bar.set_postfix(postfix)

            # Periodic MoE expert distribution logging
            if use_multimodal and self.global_step % self.training_config.log_every_n_steps == 0:
                expert_usage = output.get('expert_usage')
                if expert_usage is not None:
                    eu = expert_usage.detach().cpu()
                    is_dense = (hasattr(self.model, 'fusion_module') and
                                getattr(self.model.fusion_module, 'router_type', None) == 'dense')
                    if is_dense:
                        usage_str = " ".join([f"E{i}={eu[i].item():.3f}" for i in range(len(eu))])
                        logger.info(f"  Step {self.global_step} Modality weights: {usage_str}")
                    else:
                        usage_str = " ".join([f"E{i}={eu[i].item():.0f}" for i in range(len(eu))])
                        logger.info(f"  Step {self.global_step} MoE usage: {usage_str}")
                    if hasattr(self.model, 'fusion_module') and hasattr(self.model.fusion_module, 'fusion_alpha'):
                        alpha = torch.sigmoid(self.model.fusion_module.fusion_alpha).item()
                        logger.info(f"  Fusion alpha: {alpha:.4f}")
                    ent_pen = output.get('moe_entropy_penalty')
                    if ent_pen is not None and ent_pen > 0:
                        logger.info(f"  Entropy penalty: {ent_pen:.6f}")
                    if teacher_align_loss is not None and teacher_align_loss > 0:
                        logger.info(f"  Teacher align loss: {teacher_align_loss:.6f}")

        metrics = {'total_loss': total_loss / num_batches, 'main_loss': total_main_loss / num_batches}
        if total_pred_loss > 0:
            metrics['pred_loss'] = total_pred_loss / num_batches
        return metrics

    def evaluate(self, data_loader, split_name="Valid") -> Dict[str, float]:
        self.model.eval()
        self.metrics_calculator.reset()
        use_multimodal = self.training_config.use_multimodal_fusion
        verbose = self.training_config.verbose
        all_batches = []

        progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [{split_name}]")

        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['history'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['target'].to(self.device)
                item_ids = batch.get('target_item_id')

                purified_content = None
                purified_collab = None
                if use_multimodal and 'history_purified_content' in batch:
                    purified_content = batch['history_purified_content'].to(self.device)
                    purified_collab = batch['history_purified_collab'].to(self.device)

                max_gen_length = self.model.model_config.num_code_layers + 1
                preds = self.model.generate(
                    input_ids=input_ids, attention_mask=attention_mask,
                    num_beams=self.training_config.beam_size, max_length=max_gen_length,
                    purified_content=purified_content, purified_collab=purified_collab,
                    logits_processor=self.trie_logits_processor
                )
                preds = preds[:, 1:]

                if verbose:
                    all_batches.append({
                        'input_ids': input_ids.cpu(), 'preds': preds.cpu(),
                        'labels': labels.cpu(),
                        'item_ids': item_ids.cpu() if item_ids is not None else None,
                    })

                self.metrics_calculator.update(preds, labels, self.training_config.beam_size)

        metrics = self.metrics_calculator.compute()

        if verbose and all_batches:
            self._print_verbose_samples(all_batches, split_name)

        return metrics

    def _print_verbose_samples(self, all_batches, split_name):
        import random
        all_samples = []
        for batch_info in all_batches:
            bsize = min(batch_info['preds'].size(0), batch_info['input_ids'].size(0), batch_info['labels'].size(0))
            for i in range(bsize):
                all_samples.append({
                    'input_ids': batch_info['input_ids'][i], 'preds': batch_info['preds'][i],
                    'labels': batch_info['labels'][i],
                    'item_id': batch_info['item_ids'][i] if batch_info['item_ids'] is not None else None,
                })

        num = min(10, len(all_samples))
        sampled = random.sample(range(len(all_samples)), num)
        logger.info(f"\n{'='*80}\nVERBOSE: {split_name} (Epoch {self.current_epoch + 1})\n{'-'*40}")
        for idx, si in enumerate(sampled, 1):
            s = all_samples[si]
            logger.info(f"Sample {idx}: item={s['item_id'].item() if s['item_id'] is not None else '?'}")
            logger.info(f"  History: {[x for x in s['input_ids'].tolist() if x != 0]}")
            logger.info(f"  Pred:    {s['preds'].tolist()}")
            logger.info(f"  GT:      {s['labels'].tolist()}")
            logger.info(f"  Match:   {'YES' if s['preds'].tolist() == s['labels'].tolist() else 'NO'}")
        logger.info(f"{'='*80}\n")

    def _save_best_model(self, metrics):
        checkpoint = {
            'epoch': self.current_epoch, 'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric, 'metrics': metrics, 'config': self.config
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, self.checkpoint_dir / "best_model.pt")
        logger.info(f"Best model saved to {self.checkpoint_dir / 'best_model.pt'}")

    def save_checkpoint(self, metrics, is_best=False):
        checkpoint = {
            'epoch': self.current_epoch, 'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric, 'metrics': metrics, 'config': self.config
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt")
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        if self.training_config.keep_last_n_checkpoints <= 0:
            return
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
        for cp in checkpoints[:-self.training_config.keep_last_n_checkpoints]:
            cp.unlink()

    def load_checkpoint(self, checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']

    def save_history(self):
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

    def train(self):
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        start_time = time.time()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            logger.info("-" * 80)

            train_losses = self.train_epoch()

            if isinstance(train_losses, dict):
                self.history['train_loss'].append(train_losses['total_loss'])
                current_lr = self.optimizer.param_groups[0]['lr']
                parts = f"Training - Total: {train_losses['total_loss']:.4f}, Main: {train_losses['main_loss']:.4f}"
                if 'pred_loss' in train_losses:
                    parts += f", Pred: {train_losses['pred_loss']:.4f}"
                parts += f" | LR: {current_lr:.6f}"
                logger.info(parts)
            else:
                self.history['train_loss'].append(train_losses)
                logger.info(f"Training loss: {train_losses:.4f}")

            if (epoch + 1) % self.training_config.eval_every_n_epochs == 0:
                valid_metrics = self.evaluate(self.valid_loader, "Valid")
                self.history['valid_metrics'].append({'epoch': epoch + 1, **valid_metrics})
                logger.info(f"Validation: {format_metrics(valid_metrics)}")

                current_metric = valid_metrics[self.training_config.early_stopping_metric]
                is_best = current_metric > self.best_metric

                if self.scheduler is not None and self.scheduler_type == 'reduce_on_plateau':
                    self.scheduler.step(current_metric)

                if is_best:
                    self.best_metric = current_metric
                    self.history['best_epoch'] = epoch + 1
                    test_metrics = self.evaluate(self.test_loader, "Test")
                    self.history['test_metrics'].append({'epoch': epoch + 1, **test_metrics})
                    logger.info(f"New best {self.training_config.early_stopping_metric}: {current_metric:.4f}")
                    logger.info(f"Test: {format_metrics(test_metrics)}")
                    self._save_best_model(valid_metrics)
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    logger.info(f"No improvement. Counter: {self.early_stop_counter}/{self.training_config.early_stopping_patience}")

                if is_best:
                    self.save_checkpoint(valid_metrics, is_best=True)
                elif (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                    self.save_checkpoint(valid_metrics, is_best=False)

                if self.early_stop_counter >= self.training_config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"TRAINING COMPLETED ({elapsed / 3600:.2f}h)")
        logger.info(f"Best epoch: {self.history['best_epoch']}, Best {self.training_config.early_stopping_metric}: {self.best_metric:.4f}")
        self.save_history()

        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            self.load_checkpoint(best_path)
            final_metrics = self.evaluate(self.test_loader, "Test (Final)")
            logger.info(f"Final test: {format_metrics(final_metrics)}")
            with open(self.output_dir / "final_test_metrics.json", 'w') as f:
                json.dump(final_metrics, f, indent=2)
