"""
Training utilities for ActionPiece recommender model.

Provides trainer class with:
- Dynamic SPR augmentation during training
- Inference-time ensemble evaluation
- nDCG-based score aggregation
"""

import collections
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import json
import os
from typing import Dict, Optional, List
import time
import math
import numpy as np

# Import local ActionPieceCore
from src.sid_tokenizer.ActionPiece.actionpiece_core import ActionPieceCore

# Import metrics from TIGER (shared utilities)
from src.recommender.TIGER.metrics import MetricsCalculator, format_metrics
from .actionpiece_dataset import collate_fn_actionpiece, ActionPieceEnsembleDataset, collate_fn_ensemble

logger = logging.getLogger(__name__)


class ActionPieceTrainer:
    """Trainer for ActionPiece model with dynamic augmentation."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        actionpiece_mapper,
        device: str = "cuda"
    ):
        """Initialize the trainer.
        
        Args:
            model: ActionPiece model instance
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            config: Configuration dictionary
            actionpiece_mapper: ActionPieceMapper for tokenization
            device: Device to use for training
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.mapper = actionpiece_mapper
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Training config
        self.training_config = config['training']
        self.num_epochs = self.training_config.num_epochs
        self.learning_rate = self.training_config.learning_rate
        self.gradient_clip = self.training_config.gradient_clip
        
        # ActionPiece specific settings
        self.n_inference_ensemble = config['model'].n_inference_ensemble
        
        # Setup optimizer (AdamW as in paper)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Setup learning rate scheduler (cosine with warmup)
        self.scheduler = self._get_lr_scheduler()
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(
            topk_list=self.training_config.topk_list,
            num_layers=self.mapper.n_categories
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
        logger.info(f"Inference ensemble: {self.n_inference_ensemble}")
    
    def _get_lr_scheduler(self):
        """Get learning rate scheduler (cosine with warmup)."""
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.training_config.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                min_lr_ratio = self.training_config.min_lr / self.learning_rate
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self) -> float:
        """Train for one epoch with dynamic SPR augmentation.
        
        Note: SPR augmentation happens in the dataset's __getitem__ method,
        so each epoch sees different tokenizations of the same sequences.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]"
        )
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
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
        
        return total_loss / num_batches
    
    def evaluate(
        self, 
        data_loader: DataLoader, 
        split_name: str = "Valid",
        use_ensemble: bool = False
    ) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            data_loader: Data loader for evaluation
            split_name: Name of the split (for logging)
            use_ensemble: Whether to use inference-time ensemble
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.metrics_calculator.reset()
        
        n_ensemble = self.n_inference_ensemble if use_ensemble else 1
        
        progress_bar = tqdm(
            data_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [{split_name}]"
        )
        
        first_batch = True
        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target_states = batch['target_states'].to(self.device)
                
                # Check if this is ensemble data (from collate_fn_actionpiece_test)
                batch_n_ensemble = batch.get('n_ensemble', 1)
                original_batch_size = batch.get('batch_size', target_states.shape[0])
                
                if use_ensemble and batch_n_ensemble > 1:
                    # Ensemble evaluation: input_ids is (batch_size * n_ensemble, seq_len)
                    # target_states is (batch_size, n_categories)
                    preds = self._evaluate_with_ensemble(
                        input_ids, attention_mask, original_batch_size, batch_n_ensemble
                    )
                else:
                    # Single evaluation
                    preds = self._evaluate_single(input_ids, attention_mask, debug=first_batch)
                
                # Debug: print first batch info
                if first_batch:
                    logger.info(f"[DEBUG] First batch - input_ids shape: {input_ids.shape}")
                    logger.info(f"[DEBUG] First batch - target_states shape: {target_states.shape}")
                    logger.info(f"[DEBUG] First batch - preds shape: {preds.shape}")
                    logger.info(f"[DEBUG] First target: {target_states[0].tolist()}")
                    logger.info(f"[DEBUG] First pred (top-1): {preds[0].tolist()}")
                    if use_ensemble and batch_n_ensemble > 1:
                        logger.info(f"[DEBUG] Ensemble mode: n_ensemble={batch_n_ensemble}, original_batch_size={original_batch_size}")
                    first_batch = False
                
                # Update metrics
                # preds: (batch_size * beam_size, n_categories)
                # target_states: (batch_size, n_categories)
                self.metrics_calculator.update(
                    preds, target_states, self.training_config.beam_size
                )
        
        return self.metrics_calculator.compute()
    
    def _evaluate_single(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        debug: bool = False
    ) -> torch.Tensor:
        """Single evaluation without ensemble."""
        n_categories = self.mapper.n_categories
        beam_size = self.training_config.beam_size
        
        # Generate predictions
        outputs = self.model.generate_single(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            max_length=n_categories + 2,  # +2 for decoder_start_token and potential EOS
            num_return_sequences=beam_size
        )
        
        if debug:
            logger.debug(f"Generated outputs shape: {outputs.shape}")
            logger.debug(f"First output: {outputs[0].tolist()}")
        
        # Remove decoder start token
        outputs = outputs[:, 1:]
        
        # Decode outputs to feature states
        decoded = self._decode_outputs(outputs, debug=debug)
        
        return decoded
    
    def _evaluate_with_ensemble(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        n_ensemble: int
    ) -> torch.Tensor:
        """Evaluation with inference-time ensemble (following original ActionPiece implementation).
        
        This method handles input_ids that contain n_ensemble different SPR encodings
        per sample (from ActionPieceEnsembleDataset).
        
        Args:
            input_ids: Shape (batch_size * n_ensemble, seq_len)
            attention_mask: Shape (batch_size * n_ensemble, seq_len)
            batch_size: Original batch size (before ensemble expansion)
            n_ensemble: Number of ensemble runs per sample
        
        Returns:
            Aggregated predictions, shape (batch_size * beam_size, n_categories)
        """
        n_categories = self.mapper.n_categories
        beam_size = self.training_config.beam_size
        device = input_ids.device
        
        # Generate predictions for all ensemble inputs at once
        # max_length should be n_categories + 2 to account for decoder_start_token and potential EOS
        outputs = self.model.generate_single(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            max_length=n_categories + 2,
            num_return_sequences=beam_size,
            early_stopping=False  # Don't stop early, generate full sequence
        )
        
        # Remove decoder start token (first token)
        outputs = outputs[:, 1:]
        
        # Decode all outputs - this returns (num_sequences, n_categories)
        decoded = self._decode_outputs(outputs)
        
        # Expected shape: (batch_size * n_ensemble * beam_size, n_categories)
        expected_size = batch_size * n_ensemble * beam_size
        actual_size = decoded.shape[0]
        
        if actual_size != expected_size:
            logger.warning(f"Decoded shape mismatch: expected {expected_size}, got {actual_size}")
            # Pad or truncate if necessary
            if actual_size < expected_size:
                padding = torch.full(
                    (expected_size - actual_size, n_categories),
                    -1, dtype=torch.long, device=device
                )
                decoded = torch.cat([decoded, padding], dim=0)
            else:
                decoded = decoded[:expected_size]
        
        # Reshape: (batch_size * n_ensemble * beam_size, n_categories) -> (batch_size, n_ensemble, beam_size, n_categories)
        decoded = decoded.view(batch_size, n_ensemble, beam_size, n_categories)
        
        # Aggregate using nDCG weighting
        final_outputs = self._aggregate_ensemble(decoded, batch_size, beam_size, n_categories, device)
        
        # Reshape to (batch_size * beam_size, n_categories)
        return final_outputs.view(-1, n_categories)
    
    def _decode_outputs(self, outputs: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Decode generated outputs to feature states (token indices).
        
        Args:
            outputs: Generated token IDs, shape (num_sequences, seq_len)
            debug: Whether to print debug info
            
        Returns:
            Decoded states as token indices, shape (num_sequences, n_categories)
            Each row contains [rank[(0, feat0)], rank[(1, feat1)], ...] format
            to match target_states format.
        """
        n_categories = self.mapper.n_categories
        device = outputs.device
        
        decoded_outputs = []
        valid_count = 0
        invalid_count = 0
        
        for i, output in enumerate(outputs.cpu().tolist()):
            # Remove EOS token if present
            if self.mapper.eos_token in output:
                idx = output.index(self.mapper.eos_token)
                output = output[:idx]
            
            # Truncate to n_categories
            output = output[:n_categories]
            
            # Decode to single state: List[(category_idx, feature_idx)]
            decoded = self.mapper.decode_tokens(output)
            if decoded is None:
                # Invalid decoding, fill with -1
                decoded_outputs.append([-1] * n_categories)
                invalid_count += 1
                if debug and i < 5:
                    logger.debug(f"Invalid decode for output {i}: {output}")
            else:
                # Convert (category, feature) tuples to token indices
                # decoded is List[(category_idx, feature_idx)]
                # We need to convert back to rank[(cat, feat)] format
                token_indices = [-1] * n_categories
                for cat_idx, feat_idx in decoded:
                    if 0 <= cat_idx < n_categories:
                        # Get the token index for this (category, feature) pair
                        token_idx = self.mapper.actionpiece.rank.get((cat_idx, feat_idx), -1)
                        token_indices[cat_idx] = token_idx
                decoded_outputs.append(token_indices)
                valid_count += 1
                if debug and i < 5:
                    logger.debug(f"Valid decode for output {i}: {output} -> {decoded} -> {token_indices}")
        
        if debug:
            logger.debug(f"Decode stats: valid={valid_count}, invalid={invalid_count}")
        
        return torch.tensor(decoded_outputs, dtype=torch.long, device=device)
    
    def _aggregate_ensemble(
        self,
        all_decoded: torch.Tensor,
        batch_size: int,
        beam_size: int,
        n_categories: int,
        device: torch.device
    ) -> torch.Tensor:
        """Aggregate ensemble predictions using nDCG weighting."""
        n_ensemble = all_decoded.shape[1]
        
        final_outputs = torch.full(
            (batch_size, beam_size, n_categories),
            -1,
            dtype=torch.long,
            device=device
        )
        
        for bid in range(batch_size):
            pred2score = collections.defaultdict(float)
            
            for ens_idx in range(n_ensemble):
                for rank_idx in range(beam_size):
                    pred = tuple(all_decoded[bid, ens_idx, rank_idx].tolist())
                    if pred[0] != -1:  # Valid prediction
                        # nDCG-style weighting
                        pred2score[pred] += 1 / np.log2(rank_idx + 2)
            
            # Sort by aggregated score
            all_scores = [(pred, score) for pred, score in pred2score.items()]
            all_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Fill final outputs
            for j in range(min(beam_size, len(all_scores))):
                final_outputs[bid, j] = torch.tensor(all_scores[j][0], dtype=torch.long)
        
        return final_outputs
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints."""
        if self.training_config.keep_last_n_checkpoints <= 0:
            return
        
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime
        )
        
        for checkpoint in checkpoints[:-self.training_config.keep_last_n_checkpoints]:
            checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint with validation checks.
        
        Performs the following checks before loading:
        1. Checkpoint file exists
        2. Dataset name matches
        3. Model architecture is compatible (vocab size, n_categories)
        4. Tokenizer path matches (warning if different)
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is incompatible with current config
        """
        logger.info("=" * 80)
        logger.info("LOADING CHECKPOINT")
        logger.info("=" * 80)
        logger.info(f"Checkpoint path: {checkpoint_path}")
        
        # Check 1: File exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Check 2: Required keys exist
        required_keys = ['epoch', 'global_step', 'model_state_dict', 'optimizer_state_dict', 
                         'scheduler_state_dict', 'best_metric', 'config']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
        
        ckpt_config = checkpoint['config']
        
        # Check 3: Dataset name matches
        ckpt_dataset = ckpt_config['data'].dataset_name
        current_dataset = self.config['data'].dataset_name
        if ckpt_dataset != current_dataset:
            raise ValueError(
                f"Dataset mismatch! Checkpoint was trained on '{ckpt_dataset}', "
                f"but current config is for '{current_dataset}'. "
                f"Please use the correct dataset configuration."
            )
        logger.info(f"✓ Dataset check passed: {ckpt_dataset}")
        
        # Check 4: Model architecture compatibility
        ckpt_vocab_size = ckpt_config['model'].vocab_size
        current_vocab_size = self.config['model'].vocab_size
        if ckpt_vocab_size != current_vocab_size:
            raise ValueError(
                f"Vocab size mismatch! Checkpoint: {ckpt_vocab_size}, "
                f"Current: {current_vocab_size}. "
                f"This usually means different tokenizers were used."
            )
        logger.info(f"✓ Vocab size check passed: {ckpt_vocab_size}")
        
        ckpt_n_categories = ckpt_config['model'].n_categories
        current_n_categories = self.config['model'].n_categories
        if ckpt_n_categories != current_n_categories:
            raise ValueError(
                f"Number of categories mismatch! Checkpoint: {ckpt_n_categories}, "
                f"Current: {current_n_categories}."
            )
        logger.info(f"✓ N_categories check passed: {ckpt_n_categories}")
        
        # Check 5: Tokenizer path (warning only, not fatal)
        ckpt_tokenizer = ckpt_config['data'].tokenizer_path
        current_tokenizer = self.config['data'].tokenizer_path
        if ckpt_tokenizer != current_tokenizer:
            logger.warning(
                f"⚠ Tokenizer path differs! Checkpoint: {ckpt_tokenizer}, "
                f"Current: {current_tokenizer}. "
                f"Make sure they are equivalent tokenizers."
            )
        else:
            logger.info(f"✓ Tokenizer path check passed")
        
        # Check 6: Model state dict keys match
        model_state_keys = set(self.model.state_dict().keys())
        ckpt_state_keys = set(checkpoint['model_state_dict'].keys())
        if model_state_keys != ckpt_state_keys:
            missing_in_ckpt = model_state_keys - ckpt_state_keys
            extra_in_ckpt = ckpt_state_keys - model_state_keys
            if missing_in_ckpt:
                logger.warning(f"⚠ Keys in model but not in checkpoint: {missing_in_ckpt}")
            if extra_in_ckpt:
                logger.warning(f"⚠ Keys in checkpoint but not in model: {extra_in_ckpt}")
            raise ValueError(
                "Model architecture mismatch! The checkpoint was trained with a different model structure."
            )
        logger.info(f"✓ Model architecture check passed")
        
        # All checks passed, load the checkpoint
        logger.info("-" * 80)
        logger.info("All compatibility checks passed. Loading checkpoint...")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        # Log checkpoint info
        ckpt_metrics = checkpoint.get('metrics', {})
        logger.info(f"✓ Checkpoint loaded successfully!")
        logger.info(f"  - Trained epochs: {checkpoint['epoch'] + 1}")
        logger.info(f"  - Resuming from epoch: {self.current_epoch + 1}")
        logger.info(f"  - Global step: {self.global_step}")
        logger.info(f"  - Best metric ({self.training_config.early_stopping_metric}): {self.best_metric:.6f}")
        if ckpt_metrics:
            logger.info(f"  - Last checkpoint metrics:")
            for k, v in ckpt_metrics.items():
                if isinstance(v, float):
                    logger.info(f"      {k}: {v:.6f}")
        logger.info("=" * 80)
    
    def save_history(self):
        """Save training history."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")
    
    def train(self):
        """Run the complete training loop."""
        logger.info("=" * 80)
        logger.info("STARTING ACTIONPIECE TRAINING")
        logger.info("=" * 80)
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.valid_loader.dataset)}")
        logger.info(f"Test samples: {len(self.test_loader.dataset)}")
        logger.info(f"Inference ensemble: {self.n_inference_ensemble}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            logger.info("-" * 80)
            
            # Train for one epoch (with dynamic SPR augmentation)
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Training loss: {train_loss:.4f} | LR: {current_lr:.6f}")
            
            # Evaluate on validation set
            if (epoch + 1) % self.training_config.eval_every_n_epochs == 0:
                # Use ensemble for validation if configured
                use_ensemble = self.n_inference_ensemble > 1
                valid_metrics = self.evaluate(self.valid_loader, "Valid", use_ensemble=False)
                self.history['valid_metrics'].append({
                    'epoch': epoch + 1,
                    **valid_metrics
                })
                
                logger.info(f"Validation metrics:")
                logger.info(format_metrics(valid_metrics))
                
                # Check if this is the best model
                current_metric = valid_metrics[self.training_config.early_stopping_metric]
                is_best = current_metric > self.best_metric
                
                if is_best:
                    self.best_metric = current_metric
                    self.early_stop_counter = 0
                    self.history['best_epoch'] = epoch + 1
                    
                    # Evaluate on test set with ensemble (following original implementation)
                    test_metrics = self.evaluate(
                        self.test_loader, "Test", 
                        use_ensemble=use_ensemble
                    )
                    self.history['test_metrics'].append({
                        'epoch': epoch + 1,
                        **test_metrics
                    })
                    
                    logger.info(f"✓ New best {self.training_config.early_stopping_metric}: {current_metric:.4f}")
                    logger.info(f"Test metrics (with ensemble={use_ensemble}):")
                    logger.info(format_metrics(test_metrics))
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
        
        # Load best model and evaluate on test set with full ensemble
        best_model_path = self.checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            self.load_checkpoint(best_model_path)
            final_test_metrics = self.evaluate(
                self.test_loader, "Test (Final)", 
                use_ensemble=(self.n_inference_ensemble > 1)
            )
            logger.info(f"\nFinal test metrics (best model, ensemble={self.n_inference_ensemble > 1}):")
            logger.info(format_metrics(final_test_metrics))
            
            # Save final test metrics
            final_metrics_path = self.output_dir / "final_test_metrics.json"
            with open(final_metrics_path, 'w') as f:
                json.dump(final_test_metrics, f, indent=2)
            logger.info(f"Final test metrics saved to {final_metrics_path}")
