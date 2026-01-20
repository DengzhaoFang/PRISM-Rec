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
        device: str = "cuda"
    ):
        """Initialize the trainer.
        
        Args:
            model: TIGER model instance
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Training config
        self.training_config = config['training']
        self.num_epochs = self.training_config.num_epochs
        self.learning_rate = self.training_config.learning_rate
        self.gradient_clip = self.training_config.gradient_clip
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
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
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['history'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
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
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / num_batches
            })
            
            # Log periodically
            if self.global_step % self.training_config.log_every_n_steps == 0:
                logger.info(
                    f"Step {self.global_step}: loss={loss.item():.4f}, "
                    f"avg_loss={total_loss / num_batches:.4f}"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
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
                if 'item_id' in batch:
                    item_ids = batch['item_id']
                
                # Generate predictions
                # max_length = num_code_layers + 1 (for decoder start token that will be removed)
                max_gen_length = self.model.config.num_code_layers + 1
                preds = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=self.training_config.beam_size,
                    max_length=max_gen_length
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
            is_best: Whether this is the best checkpoint (kept for backward compatibility)
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
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Log current learning rate
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
                    
                    # CRITICAL FIX: Save best model immediately when new best is found
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

