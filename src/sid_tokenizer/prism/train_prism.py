#!/usr/bin/env python3
"""
PRISM Training Script

Train Hierarchical ID VAE with multi-modal inputs and tag-guided learning.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans

# Try to import matplotlib (optional, for visualizations)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import PRISM components
from PRISM import PRISM, create_prism_from_config
from multimodal_dataset import PRISMDataset, create_dataloaders
from hid_losses import PRISMTotalLoss
from hierarchical_classifiers import create_hierarchical_classifiers

# Import schedulers
try:
    from schedulers import WarmupCosineScheduler, ExponentialSchedulerWithWarmup
    SCHEDULERS_AVAILABLE = True
except ImportError:
    SCHEDULERS_AVAILABLE = False


class PRISMTrainer:
    """Main trainer class for PRISM"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.logger.info("Initializing PRISM trainer...")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # Load dataset
        self.setup_data()
        
        # Initialize model
        self.setup_model()
        
        # Initialize optimizer and scheduler
        self.setup_optimizer()
        
        # Initialize loss function
        self.setup_loss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.perplexity_collapse_epochs = 0
        
        # Metrics tracking
        self.train_history = defaultdict(list)
        self.prev_epoch_metrics: Optional[Dict[str, float]] = None
        self.last_loss_weights = {
            'recon': 1.0,
            'anchor': 0.0,
            'balance': 0.0,
            'class': 0.0,
            'commitment': 1.0
        }
        self.curriculum_stage: Optional[str] = None
        self.curriculum_settings = self._build_curriculum_settings()
        self.best_metrics = {
            'total_loss': float('inf'),
            'class_total': float('inf')
        }
        
        self.logger.info("✓ PRISM trainer initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        
        # Create logger
        self.logger = logging.getLogger('PRISMTrainer')
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"Logging to {log_file}")
    
    def setup_data(self):
        """Setup datasets and dataloaders"""
        self.logger.info("Loading dataset...")
        
        data_dir = self.config['data_path']
        batch_size = self.config.get('batch_size', 256)
        num_workers = self.config.get('num_workers', 4)
        max_items = self.config.get('max_items', None)
        n_layers = self.config.get('n_layers', 4)  # Get n_layers early for dataset
        
        # Create dataloader
        self.train_loader, self.dataset = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            max_items=max_items,
            n_layers=n_layers  # Pass n_layers to dataset
        )
        
        self.logger.info(f"✓ Dataset loaded: {len(self.dataset)} items")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Number of batches: {len(self.train_loader)}")
        self.logger.info(f"  Tag statistics: {self.dataset.tag_stats}")
        
        # Get tag embeddings for anchor loss (fixed throughout training)
        self.tag_embeddings_per_layer = self.dataset.get_tag_embeddings_per_level(n_layers)
        
        # Move tag embeddings to device
        self.tag_embeddings_per_layer = [
            emb.to(self.device) for emb in self.tag_embeddings_per_layer
        ]
        
        self.logger.info(f"✓ Tag embeddings loaded:")
        for i, emb in enumerate(self.tag_embeddings_per_layer):
            self.logger.info(f"  Layer {i+1} (L{i+2}): {emb.shape[0]} tags")
    
    def setup_model(self):
        """Initialize PRISM model"""
        self.logger.info("Initializing PRISM model...")
        
        # Get number of classes per layer (add 1 for PAD token)
        num_classes_per_layer = [
            self.dataset.tag_stats[f'n_L{i+2}'] + 1  # +1 for PAD
            for i in range(self.config.get('n_layers', 3))
        ]
        
        self.logger.info(f"Number of classes per layer: {num_classes_per_layer}")
        
        # Create model
        self.model = create_prism_from_config(
            config=self.config,
            num_classes_per_layer=num_classes_per_layer
        )
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"✓ Model initialized")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        # Log codebook sizes
        codebook_sizes = self.model.get_codebook_sizes()
        self.logger.info(f"  Codebook sizes per layer: {codebook_sizes}")
        for i, size in enumerate(codebook_sizes):
            self.logger.info(f"    Layer {i+1}: {size} codes")
        
        self._initialize_codebooks_hierarchical()
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler"""
        self.logger.info("Initializing optimizer and scheduler...")
        
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.0)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        if self.config.get('use_scheduler', False) and SCHEDULERS_AVAILABLE:
            scheduler_type = self.config.get('scheduler_type', 'warmup_cosine')
            
            total_steps = self.config['epochs'] * len(self.train_loader)
            warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
            
            if scheduler_type == 'warmup_cosine':
                self.scheduler = WarmupCosineScheduler(
                    optimizer=self.optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    min_lr_ratio=0.01  # 1% of initial LR
                )
            elif scheduler_type == 'exponential':
                self.scheduler = ExponentialSchedulerWithWarmup(
                    optimizer=self.optimizer,
                    warmup_steps=warmup_steps,
                    decay_rate=0.95,
                    decay_steps=len(self.train_loader),
                    min_lr=lr * 0.01
                )
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
            self.logger.info(f"✓ Scheduler: {scheduler_type}")
            self.logger.info(f"  Warmup steps: {warmup_steps}")
            self.logger.info(f"  Total steps: {total_steps}")
        else:
            self.scheduler = None
            self.logger.info("No scheduler used")
        
        self.logger.info(f"✓ Optimizer: AdamW (lr={lr}, wd={weight_decay})")
    
    def setup_loss(self):
        """Initialize loss function"""
        self.logger.info("Initializing loss function...")
        
        # Get loss weights
        gamma_weight = self.config.get('gamma_weight', 1.0)
        beta_anchor_weight = self.config.get('beta_anchor_weight', 0.1)
        n_layers = self.config.get('n_layers', 3)
        
        # Create weight lists for each layer (uniform for now)
        gamma_weights = [gamma_weight] * n_layers
        beta_weights = [beta_anchor_weight] * n_layers
        
        self.loss_fn = PRISMTotalLoss(
            # Reconstruction loss params
            lambda_content=self.config.get('lambda_content', 1.0),
            lambda_collab=self.config.get('lambda_collab', 1.0),
            use_dual_decoder=self.config.get('use_dual_decoder', True),
            # Anchor loss params
            tag_embed_dim=768,
            codebook_dim=self.config.get('latent_dim', 32),
            beta_weights=beta_weights,
            # Balance loss params
            gamma_weights=gamma_weights,
            # Classification loss params
            delta_weight=self.config.get('delta_weight', 1.0),
            # Commitment loss param
            commitment_weight=self.config.get('beta', 0.25),
            # Gate supervision params
            use_gate_supervision=self.config.get('use_gate_supervision', False),
            gate_supervision_weight=self.config.get('gate_supervision_weight', 0.1),
            gate_diversity_weight=self.config.get('gate_diversity_weight', 0.5),
            gate_target_std=self.config.get('gate_target_std', 0.2),
            # General params
            n_layers=n_layers,
            ignore_index=0  # PAD token
        )
        
        self.loss_fn = self.loss_fn.to(self.device)
        
        self.logger.info("✓ Loss function initialized")
        self.logger.info(f"  Decoder mode: {'Dual' if self.config.get('use_dual_decoder', True) else 'Single'}")
        self.logger.info(f"  Lambda content: {self.config.get('lambda_content', 1.0)}")
        self.logger.info(f"  Lambda collab: {self.config.get('lambda_collab', 1.0)}")
        self.logger.info(f"  Delta (classification): {self.config.get('delta_weight', 1.0)}")
        self.logger.info(f"  Beta (commitment): {self.config.get('beta', 0.25)}")
        self.logger.info(f"  Gamma (balance): {gamma_weight}")
        self.logger.info(f"  Beta anchor (tag alignment): {beta_anchor_weight}")
        if self.config.get('use_gate_supervision', False):
            self.logger.info(f"  Gate supervision: ENABLED")
            self.logger.info(f"    Weight: {self.config.get('gate_supervision_weight', 0.1)}")
            self.logger.info(f"    Target std: {self.config.get('gate_target_std', 0.2)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            metrics: Dictionary of averaged metrics
        """
        self.model.train()
        epoch_metrics = defaultdict(float)
        loss_weights = self._get_loss_weights(epoch)
        self.last_loss_weights = loss_weights
        stage_label = loss_weights.get('stage')
        if stage_label != self.curriculum_stage:
            pretty_stage = stage_label or "full_objective"
            self.logger.info(f"[Curriculum] Entering stage: {pretty_stage} (epoch {epoch})")
            self.curriculum_stage = stage_label
        
        # Track gate statistics if using gated fusion
        gate_stats = defaultdict(float)
        n_gate_samples = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch}/{self.config['epochs']}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            content_emb = batch['content_emb'].to(self.device)
            collab_emb = batch['collab_emb'].to(self.device)
            tag_ids = batch['tag_ids'].to(self.device)  # (B, 3)
            tag_mask = batch['tag_mask'].to(self.device)  # (B, 3)
            popularity_scores = batch['popularity_score'].to(self.device)  # (B,)
            
            # Collect gate statistics (if using gated fusion)
            if self.config.get('use_gated_fusion', True) and hasattr(self.model.encoder, 'gate_network'):
                with torch.no_grad():
                    gate = self.model.encoder.gate_network(collab_emb)  # (B, 768)
                    mean_gate = gate.mean(dim=1)  # (B,) - average trust per item
                    
                    gate_stats['mean'] += mean_gate.mean().item()
                    gate_stats['std'] += mean_gate.std().item()
                    gate_stats['min'] += mean_gate.min().item()
                    gate_stats['max'] += mean_gate.max().item()
                    gate_stats['median'] += mean_gate.median().item()
                    n_gate_samples += 1
            
            # Forward pass
            outputs = self.model(
                content_emb=content_emb,
                collab_emb=collab_emb,
                temperature=self._get_temperature(epoch),
                return_codes=True
            )
            
            # Extract outputs
            content_recon = outputs['content_recon']
            collab_recon = outputs['collab_recon']
            weighted_collab_emb = outputs.get('weighted_collab_emb', None)  # DHR target
            quantized_codes = outputs['quantized_codes']
            encoding_indices = outputs['encoding_indices']
            vq_loss = outputs['codebook_loss']  # This is codebook + commitment loss
            predictions = outputs.get('predictions', None)
            
            # Get codebooks for anchor loss
            codebooks = self.model.get_codebooks()
            
            # Prepare tag targets (split by layer)
            targets_per_layer = [tag_ids[:, i] for i in range(self.config['n_layers'])]
            masks_per_layer = [tag_mask[:, i] for i in range(self.config['n_layers'])]
            
            # Prepare codebook sizes (use variable sizes if available)
            n_embed_per_layer = self.model.get_codebook_sizes()
            
            # Get gate values for supervision (if enabled)
            if self.config.get('use_gate_supervision', False) and hasattr(self.model.encoder, 'gate_network'):
                gate_values = self.model.encoder.gate_network(collab_emb)  # (B, 768)
            else:
                gate_values = None
            
            # Compute total loss
            total_loss, loss_dict = self.loss_fn(
                # Reconstruction inputs
                pred_content=content_recon,
                target_content=content_emb,
                pred_collab=collab_recon,
                target_collab=collab_emb,
                # Anchor inputs
                tag_embeddings_per_layer=self.tag_embeddings_per_layer,
                codebooks=codebooks,
                # Balance inputs
                encoding_indices_per_layer=encoding_indices,
                n_embed_per_layer=n_embed_per_layer,
                # Classification inputs
                predictions_per_layer=predictions if predictions else [],
                targets_per_layer=targets_per_layer,
                masks_per_layer=masks_per_layer,
                # VQ loss (codebook + commitment)
                commitment_loss=vq_loss,
                # Gate supervision inputs
                gate_values=gate_values,
                popularity_scores=popularity_scores,
                # DHR reconstruction target (gate-weighted collab)
                weighted_collab_target=weighted_collab_emb,
                # Curriculum weights
                loss_weights=loss_weights
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update metrics
            for key, value in loss_dict.items():
                epoch_metrics[key] += value
            
            # Add perplexity metrics
            for i, perp in enumerate(outputs['perplexities']):
                epoch_metrics[f'perplexity_layer{i+1}'] += perp
            
            # Update progress bar
            postfix_dict = {
                'loss': loss_dict['total_loss'],
                'recon': loss_dict['recon_total'],
                'class': loss_dict.get('class_total', 0),
                'anchor_w': loss_weights['anchor'],
                'lr': self.optimizer.param_groups[0]['lr']
            }
            if 'gate_supervision' in loss_dict:
                postfix_dict['gate_sup'] = loss_dict['gate_supervision']
            progress_bar.set_postfix(postfix_dict)
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        epoch_metrics['loss_weight_recon'] = loss_weights['recon']
        epoch_metrics['loss_weight_anchor'] = loss_weights['anchor']
        epoch_metrics['loss_weight_balance'] = loss_weights['balance']
        epoch_metrics['loss_weight_class'] = loss_weights['class']
        epoch_metrics['loss_weight_commitment'] = loss_weights['commitment']
        
        # Add gate statistics to metrics
        if n_gate_samples > 0:
            for key in gate_stats:
                epoch_metrics[f'gate_{key}'] = gate_stats[key] / n_gate_samples
        
        self.prev_epoch_metrics = epoch_metrics
        
        return epoch_metrics
    
    def _get_temperature(self, epoch: int) -> float:
        """
        Get temperature for Gumbel-Softmax (if used).
        Gradually anneal from init_temp to min_temp.
        """
        if self.config.get('quantize_mode', 'rotation') != 'gumbel_softmax':
            return 0.2  # Not used for other modes
        
        init_temp = self.config.get('init_temp', 1.0)
        min_temp = self.config.get('min_temp', 0.1)
        anneal_rate = self.config.get('anneal_rate', 0.00003)
        
        return max(min_temp, init_temp * np.exp(-anneal_rate * epoch))

    def _build_curriculum_settings(self) -> Dict[str, Any]:
        """
        Define curriculum phases and thresholds.
        
        Supports two curriculum strategies:
        1. 'default': Original order (class -> balance -> anchor)
        2. 'similarity_first': Learn similarity first, then distinctiveness
           - Phase 1: Recon + Class + Anchor (capture similarity)
           - Phase 2: Add Balance (increase distinctiveness)
        """
        enabled = not self.config.get('no_curriculum', False)
        epochs = max(1, self.config.get('epochs', 100))
        strategy = self.config.get('curriculum_strategy', 'default')
        
        # Get base parameters
        warmup_epochs = self.config.get('curriculum_warmup_epochs', max(3, epochs // 10))
        
        if strategy == 'similarity_first':
            # Strategy: Learn similarity first, then distinctiveness
            # Phase 1 (Similarity): Recon + Class + Anchor
            # Phase 2 (Distinctiveness): Add Balance, reduce Anchor
            
            phase1_duration = self.config.get('curriculum_phase1_duration', epochs // 3)
            phase2_start = warmup_epochs + phase1_duration
            
            class_delay = warmup_epochs
            class_ramp = self.config.get('curriculum_class_ramp', max(10, epochs // 8))
            
            anchor_delay = warmup_epochs + max(5, phase1_duration // 4)
            anchor_ramp = self.config.get('curriculum_anchor_ramp', max(15, phase1_duration // 2))
            
            balance_delay = phase2_start  # Start in Phase 2
            balance_ramp = self.config.get('curriculum_balance_ramp', max(20, epochs // 4))
            
            # In Phase 2, reduce anchor weight
            anchor_decay_start = phase2_start
            anchor_decay_rate = self.config.get('curriculum_anchor_decay', 0.5)
            
            return {
                'enabled': enabled,
                'strategy': 'similarity_first',
                'warmup_epochs': warmup_epochs,
                'phase1_duration': phase1_duration,
                'phase2_start': phase2_start,
                'class_delay': class_delay,
                'class_ramp_epochs': class_ramp,
                'anchor_delay': anchor_delay,
                'anchor_ramp_epochs': anchor_ramp,
                'balance_delay': balance_delay,
                'balance_ramp_epochs': balance_ramp,
                'anchor_decay_start': anchor_decay_start,
                'anchor_decay_rate': anchor_decay_rate,
                'target_class_weight': self.config.get('curriculum_class_target', 1.0),
                'target_anchor_weight': self.config.get('curriculum_anchor_target', 1.0),
                'target_balance_weight': self.config.get('curriculum_balance_target', 1.0),
                'anchor_over_total_ratio': self.config.get('curriculum_anchor_ratio', 0.55),
                'anchor_penalty_factor': self.config.get('curriculum_anchor_penalty', 0.7),
                'min_perplexity_ratio': self.config.get('curriculum_min_perplexity_ratio', 0.35),
                'perplexity_patience': self.config.get('perplexity_collapse_patience', 3),
                'early_stop_cooldown': self.config.get('early_stop_cooldown', 3)
            }
        else:
            # Default strategy: Original order
            class_ramp = max(1, self.config.get('curriculum_class_ramp', max(5, epochs // 6)))
            anchor_delay = self.config.get('curriculum_anchor_delay', warmup_epochs + max(1, epochs // 12))
            anchor_ramp = max(1, self.config.get('curriculum_anchor_ramp', max(5, epochs // 5)))
            balance_delay = self.config.get('curriculum_balance_delay', warmup_epochs)
            balance_ramp = max(1, self.config.get('curriculum_balance_ramp', max(3, epochs // 8)))
            
            return {
                'enabled': enabled,
                'strategy': 'default',
                'warmup_epochs': warmup_epochs,
                'class_ramp_epochs': class_ramp,
                'anchor_delay': anchor_delay,
                'anchor_ramp_epochs': anchor_ramp,
                'balance_delay': balance_delay,
                'balance_ramp_epochs': balance_ramp,
                'target_class_weight': self.config.get('curriculum_class_target', 1.0),
                'target_anchor_weight': self.config.get('curriculum_anchor_target', 1.0),
                'target_balance_weight': self.config.get('curriculum_balance_target', 1.0),
                'anchor_over_total_ratio': self.config.get('curriculum_anchor_ratio', 0.55),
                'anchor_penalty_factor': self.config.get('curriculum_anchor_penalty', 0.7),
                'min_perplexity_ratio': self.config.get('curriculum_min_perplexity_ratio', 0.35),
                'perplexity_patience': self.config.get('perplexity_collapse_patience', 3),
                'early_stop_cooldown': self.config.get('early_stop_cooldown', 3)
            }

    def _get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """Compute curriculum-aware loss scales for the given epoch."""
        settings = self.curriculum_settings
        if not settings.get('enabled', True):
            return {
                'recon': 1.0,
                'class': 1.0,
                'anchor': 1.0,
                'balance': 1.0,
                'commitment': 1.0,
                'stage': 'disabled'
            }
        
        strategy = settings.get('strategy', 'default')
        
        if strategy == 'similarity_first':
            return self._get_loss_weights_similarity_first(epoch, settings)
        else:
            return self._get_loss_weights_default(epoch, settings)
    
    def _get_loss_weights_similarity_first(self, epoch: int, settings: Dict) -> Dict[str, float]:
        """
        Similarity-first curriculum strategy.
        
        Phase 1 (Warmup + Similarity Learning):
          - Epochs 1-warmup: Recon + Commitment only
          - Epochs warmup-phase2: Recon + Class + Anchor (learn similarity)
        
        Phase 2 (Distinctiveness Learning):
          - Epochs phase2+: Add Balance, reduce Anchor (increase diversity)
        """
        recon_weight = 1.0
        commitment_weight = 1.0
        class_weight = 0.0
        anchor_weight = 0.0
        balance_weight = 0.0
        stage = 'warmup'
        
        warmup_epochs = settings['warmup_epochs']
        phase2_start = settings['phase2_start']
        
        # Phase 1: Warmup
        if epoch <= warmup_epochs:
            stage = 'warmup'
        
        # Phase 1: Similarity Learning (Class + Anchor)
        elif epoch <= phase2_start:
            # Classification
            class_delay = settings.get('class_delay', warmup_epochs)
            if epoch > class_delay:
                class_progress = min(1.0, (epoch - class_delay) / settings['class_ramp_epochs'])
                class_weight = settings['target_class_weight'] * class_progress
            
            # Anchor (learn tag alignment)
            anchor_delay = settings['anchor_delay']
            if epoch > anchor_delay:
                anchor_progress = min(1.0, (epoch - anchor_delay) / settings['anchor_ramp_epochs'])
                anchor_weight = settings['target_anchor_weight'] * anchor_progress
            
            stage = 'phase1_similarity'
        
        # Phase 2: Distinctiveness Learning (Balance + reduced Anchor)
        else:
            # Keep classification
            class_weight = settings['target_class_weight']
            
            # Reduce anchor weight (decay over time)
            anchor_decay_rate = settings.get('anchor_decay_rate', 0.5)
            anchor_weight = settings['target_anchor_weight'] * anchor_decay_rate
            
            # Ramp up balance
            balance_progress = min(1.0, (epoch - phase2_start) / settings['balance_ramp_epochs'])
            balance_weight = settings['target_balance_weight'] * balance_progress
            
            stage = 'phase2_distinctiveness'
        
        # Adaptive adjustments
        if self.prev_epoch_metrics:
            anchor_prev = self.prev_epoch_metrics.get('anchor_total')
            total_prev = self.prev_epoch_metrics.get('total_loss')
            if anchor_prev is not None and total_prev is not None and total_prev > 0:
                ratio = anchor_prev / total_prev
                if ratio > settings['anchor_over_total_ratio']:
                    anchor_weight *= settings['anchor_penalty_factor']
                    balance_weight = max(balance_weight, settings['target_balance_weight'])
                    stage = 'anchor_regulated'
            
            # Encourage balance when usage collapsed
            for i in range(self.config.get('n_layers', 3)):
                usage_key = f'usage_layer{i+1}'
                if usage_key in self.prev_epoch_metrics and self.prev_epoch_metrics[usage_key] < 0.15:
                    balance_weight = max(balance_weight, settings['target_balance_weight'])
                    anchor_weight *= settings['anchor_penalty_factor']
                    stage = 'balance_recovery'
                    break
        
        # Perplexity collapse guard
        if self.perplexity_collapse_epochs > 0:
            decay = 0.7 ** self.perplexity_collapse_epochs
            anchor_weight *= decay
            balance_weight *= (1.0 + 0.4 * self.perplexity_collapse_epochs)
            stage = 'perplexity_recovery'
        
        return {
            'recon': float(recon_weight),
            'class': float(class_weight),
            'anchor': float(anchor_weight),
            'balance': float(balance_weight),
            'commitment': float(commitment_weight),
            'stage': stage
        }
    
    def _get_loss_weights_default(self, epoch: int, settings: Dict) -> Dict[str, float]:
        """Default curriculum strategy (original implementation)."""
        recon_weight = 1.0
        commitment_weight = 1.0
        class_weight = 0.0
        anchor_weight = 0.0
        balance_weight = 0.0
        stage = 'recon_warmup'
        
        warmup_epochs = settings['warmup_epochs']
        if epoch > warmup_epochs:
            class_progress = min(1.0, (epoch - warmup_epochs) / settings['class_ramp_epochs'])
            class_weight = settings['target_class_weight'] * class_progress
            stage = 'class_ramp' if class_progress < 1.0 else 'class_full'
        
        if epoch > settings['balance_delay']:
            balance_progress = min(1.0, (epoch - settings['balance_delay']) / settings['balance_ramp_epochs'])
            balance_weight = settings['target_balance_weight'] * balance_progress
            if balance_progress < 1.0:
                stage = 'balance_ramp'
            else:
                stage = 'balance_full'
        
        if epoch > settings['anchor_delay']:
            anchor_progress = min(1.0, (epoch - settings['anchor_delay']) / settings['anchor_ramp_epochs'])
            anchor_weight = settings['target_anchor_weight'] * anchor_progress
            if anchor_progress < 1.0:
                stage = 'anchor_ramp'
            else:
                stage = 'full_objective'
        
        # Adaptive adjustments based on previous epoch
        if self.prev_epoch_metrics:
            anchor_prev = self.prev_epoch_metrics.get('anchor_total')
            total_prev = self.prev_epoch_metrics.get('total_loss')
            if anchor_prev is not None and total_prev is not None and total_prev > 0:
                ratio = anchor_prev / total_prev
                if ratio > settings['anchor_over_total_ratio']:
                    anchor_weight *= settings['anchor_penalty_factor']
                    balance_weight = max(balance_weight, settings['target_balance_weight'])
                    stage = 'anchor_regulated'
            
            # Encourage balance when usage collapsed
            for i in range(self.config.get('n_layers', 3)):
                usage_key = f'usage_layer{i+1}'
                if usage_key in self.prev_epoch_metrics and self.prev_epoch_metrics[usage_key] < 0.15:
                    balance_weight = max(balance_weight, settings['target_balance_weight'])
                    anchor_weight *= settings['anchor_penalty_factor']
                    stage = 'balance_recovery'
                    break
        
        # Perplexity collapse guard
        if self.perplexity_collapse_epochs > 0:
            decay = 0.7 ** self.perplexity_collapse_epochs
            anchor_weight *= decay
            balance_weight *= (1.0 + 0.4 * self.perplexity_collapse_epochs)
            stage = 'perplexity_recovery'
        
        return {
            'recon': float(recon_weight),
            'class': float(class_weight),
            'anchor': float(anchor_weight),
            'balance': float(balance_weight),
            'commitment': float(commitment_weight),
            'stage': stage
        }

    def _update_early_stopping(self, metrics: Dict[str, float], epoch: int) -> Tuple[bool, bool]:
        """Composite early stopping with cooldown and auxiliary metric tracking."""
        patience = self.config.get('early_stop_patience', float('inf'))
        if not np.isfinite(patience):
            self._update_perplexity_guard(metrics)
            return False, False
        
        min_delta = self.config.get('early_stop_min_delta', 1e-4)
        total_loss = metrics.get('total_loss', float('inf'))
        improved = total_loss < (self.best_loss - min_delta)
        
        cooldown = self.curriculum_settings.get('early_stop_cooldown', 3)
        warmup_limit = self.curriculum_settings.get('warmup_epochs', 0) + cooldown
        
        if improved:
            self.best_loss = total_loss
            self.best_metrics['total_loss'] = total_loss
            self.patience_counter = 0
        else:
            class_loss = metrics.get('class_total', float('inf'))
            if class_loss < (self.best_metrics.get('class_total', float('inf')) - min_delta):
                self.best_metrics['class_total'] = class_loss
                self.patience_counter = max(0, self.patience_counter - 1)
            elif epoch > warmup_limit:
                self.patience_counter += 1
            else:
                self.patience_counter = 0
        
        self._update_perplexity_guard(metrics)
        
        should_stop = self.patience_counter >= patience
        if self.perplexity_collapse_epochs >= self.curriculum_settings.get('perplexity_patience', 3):
            # Delay early stopping when codebook is unstable
            should_stop = False
            self.patience_counter = max(0, self.patience_counter - 1)
        
        return should_stop, improved

    def _update_perplexity_guard(self, metrics: Dict[str, float]) -> None:
        """Track consecutive epochs with low perplexity to prevent collapse."""
        perps = []
        for i in range(self.config.get('n_layers', 3)):
            key = f'perplexity_layer{i+1}'
            if key in metrics:
                perps.append(metrics[key])
        if not perps:
            self.perplexity_collapse_epochs = 0
            return
        
        avg_perp = float(sum(perps)) / len(perps)
        n_embed = max(1.0, float(self.config.get('n_embed', 256)))
        ratio = avg_perp / n_embed
        
        if ratio < self.curriculum_settings.get('min_perplexity_ratio', 0.35):
            self.perplexity_collapse_epochs += 1
        else:
            self.perplexity_collapse_epochs = 0

    def _initialize_codebooks_hierarchical(self) -> None:
        """Hierarchical K-means initialization to encourage coarse-to-fine structure."""
        if self.config.get('no_hierarchical_kmeans_init', False):
            self.logger.info("Skipping hierarchical k-means initialization (disabled).")
            return
        
        quantizers = getattr(self.model, 'quantizers', [])
        if not quantizers:
            return
        
        already_initialized = all(
            bool(getattr(q, '_initialized', torch.tensor(False)).item()) for q in quantizers
        )
        if already_initialized:
            self.logger.info("Codebooks already initialized, skipping hierarchical k-means.")
            return
        
        sample_size = min(
            self.config.get('kmeans_init_samples', 8192),
            len(self.dataset)
        )
        if sample_size < self.config.get('n_embed', 256):
            self.logger.warning("Not enough samples for hierarchical k-means initialization; skipping.")
            return
        
        self.logger.info(f"Running hierarchical k-means initialization with {sample_size} samples...")
        indices = torch.randperm(len(self.dataset))[:sample_size]
        
        latent_batches = []
        batch_size = min(self.config.get('kmeans_batch_size', 1024), sample_size)
        for start in range(0, sample_size, batch_size):
            batch_idx = indices[start:start + batch_size]
            content_batch = self.dataset.content_embeddings[batch_idx].to(self.device)
            collab_batch = self.dataset.collab_embeddings[batch_idx].to(self.device)
            with torch.no_grad():
                latents = self.model.encode(content_batch, collab_batch)
            latent_batches.append(latents.cpu())
        
        if not latent_batches:
            self.logger.warning("Failed to collect latent samples for k-means initialization.")
            return
        
        latents_cpu = torch.cat(latent_batches, dim=0)
        residual_cpu = latents_cpu.clone()
        
        codebook_sizes = self.model.get_codebook_sizes()
        for layer_idx, quantizer in enumerate(quantizers):
            n_clusters = codebook_sizes[layer_idx]
            if residual_cpu.size(0) < n_clusters:
                self.logger.warning(f"Layer {layer_idx+1}: insufficient samples for k-means ({residual_cpu.size(0)} < {n_clusters}).")
                break
            
            try:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    random_state=self.config.get('kmeans_random_state', 42)
                )
                kmeans.fit(residual_cpu.numpy())
                centroids_cpu = torch.tensor(
                    kmeans.cluster_centers_,
                    dtype=latents_cpu.dtype
                )
            except Exception as exc:
                self.logger.warning(f"K-means initialization failed at layer {layer_idx+1}: {exc}")
                break
            
            centroids = centroids_cpu.to(self.device)
            if getattr(quantizer, 'use_ema', False):
                quantizer.embedding.data.copy_(centroids)
                if hasattr(quantizer, 'embed_avg'):
                    quantizer.embed_avg.data.copy_(centroids)
                if hasattr(quantizer, 'cluster_size'):
                    quantizer.cluster_size.data.fill_(1.0)
            else:
                quantizer.embedding.weight.data.copy_(centroids)
            
            if hasattr(quantizer, '_initialized'):
                quantizer._initialized.fill_(True)
            
            assignments = torch.tensor(kmeans.labels_, dtype=torch.long)
            residual_cpu = residual_cpu - centroids_cpu[assignments]
            self.logger.info(
                f"  Layer {layer_idx+1}: k-means initialized (inertia={kmeans.inertia_:.4f})"
            )
        
        self.logger.info("Hierarchical k-means initialization completed.")
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model (currently same as train since no separate validation set).
        Can be extended to use validation split.
        """
        # For now, just return train metrics
        # TODO: Implement proper validation split
        return {}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, save_regular: bool = True):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint (only when explicitly requested, e.g., every N epochs)
        if save_regular:
            checkpoint_path = self.output_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"✓ Best model saved: {best_path}")
        
        # Always save latest checkpoint (overwrite)
        latest_path = self.output_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"✓ Checkpoint loaded (epoch {self.current_epoch})")
    
    def save_item_codebook_mappings(self):
        """
        Save detailed item information including:
        - Original item ID
        - Codebook vectors for each layer
        - Predicted tags for each layer
        
        Automatically called after training completion.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Saving Item-Codebook Mappings")
        self.logger.info("=" * 80)
        
        # Load best model
        best_model_path = self.output_dir / 'best_model.pt'
        if not best_model_path.exists():
            self.logger.warning(f"Best model not found at {best_model_path}, using current model")
        else:
            self.logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        # Get codebooks from model
        codebooks = self.model.get_codebooks()  # List of tensors [C1, C2, C3]
        
        # Collect data for all items
        self.logger.info("Processing all items...")
        item_mappings = {}
        
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Processing items"):
                content_emb = batch['content_emb'].to(self.device)
                collab_emb = batch['collab_emb'].to(self.device)
                item_ids = batch['item_id'].cpu().numpy()
                
                # Forward pass to get codes and predictions
                outputs = self.model(
                    content_emb=content_emb,
                    collab_emb=collab_emb,
                    return_codes=True
                )
                
                # Extract information
                encoding_indices = outputs['encoding_indices']  # List of (B,) tensors
                predictions = outputs.get('predictions', None)  # List of (B, num_classes) tensors
                
                # Process each item in batch
                batch_size = content_emb.size(0)
                for i in range(batch_size):
                    item_id = str(item_ids[i])
                    
                    # Get codebook indices for this item
                    item_indices = [indices[i].item() for indices in encoding_indices]
                    
                    # Get codebook vectors for this item
                    item_codebook_vectors = []
                    for layer_idx, (codebook, code_idx) in enumerate(zip(codebooks, item_indices)):
                        # codebook shape: (n_embed, latent_dim)
                        vector = codebook[code_idx].cpu().numpy().tolist()
                        item_codebook_vectors.append(vector)
                    
                    # Get predicted tags for this item
                    item_predicted_tags = []
                    if predictions is not None:
                        for layer_pred in predictions:
                            # layer_pred shape: (B, num_classes)
                            pred_tag = torch.argmax(layer_pred[i]).item()
                            item_predicted_tags.append(pred_tag)
                    
                    # Store all information
                    item_mappings[item_id] = {
                        'item_id': item_id,
                        'codebook_indices': item_indices,
                        'codebook_vectors': item_codebook_vectors,
                        'predicted_tags': item_predicted_tags
                    }
        
        # Save to JSON file
        output_file = self.output_dir / 'item_codebook_mappings.json'
        with open(output_file, 'w') as f:
            json.dump(item_mappings, f, indent=2)
        
        self.logger.info(f"✓ Saved {len(item_mappings)} item mappings to: {output_file}")
        
        # Also save in a more compact numpy format for faster loading
        npz_file = self.output_dir / 'item_codebook_mappings.npz'
        
        # Prepare numpy arrays
        n_items = len(item_mappings)
        n_layers = len(codebooks)
        latent_dim = codebooks[0].size(1)
        
        item_ids_array = np.array([int(k) for k in item_mappings.keys()])
        indices_array = np.array([item_mappings[str(iid)]['codebook_indices'] for iid in item_ids_array])
        
        # Stack all codebook vectors
        vectors_list = [item_mappings[str(iid)]['codebook_vectors'] for iid in item_ids_array]
        vectors_array = np.array(vectors_list)  # Shape: (n_items, n_layers, latent_dim)
        
        # Stack predicted tags
        if predictions is not None:
            tags_list = [item_mappings[str(iid)]['predicted_tags'] for iid in item_ids_array]
            tags_array = np.array(tags_list)  # Shape: (n_items, n_layers)
        else:
            tags_array = np.array([])
        
        np.savez(
            npz_file,
            item_ids=item_ids_array,
            codebook_indices=indices_array,
            codebook_vectors=vectors_array,
            predicted_tags=tags_array
        )
        
        self.logger.info(f"✓ Saved numpy format to: {npz_file}")
        self.logger.info(f"  - item_ids: {item_ids_array.shape}")
        self.logger.info(f"  - codebook_indices: {indices_array.shape}")
        self.logger.info(f"  - codebook_vectors: {vectors_array.shape}")
        if predictions is not None:
            self.logger.info(f"  - predicted_tags: {tags_array.shape}")
        
        self.logger.info("=" * 80)
    
    def generate_semantic_ids_and_analyze(self):
        """
        Generate semantic IDs for all items and perform comprehensive analysis.
        Automatically called after training completion.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Generating Semantic IDs and Analysis")
        self.logger.info("=" * 80)
        
        # Load best model
        best_model_path = self.output_dir / 'best_model.pt'
        if not best_model_path.exists():
            self.logger.warning(f"Best model not found at {best_model_path}, using current model")
        else:
            self.logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        # Generate IDs
        self.logger.info("Generating semantic IDs for all items...")
        all_item_ids = []
        all_semantic_ids = []
        all_tag_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Generating IDs"):
                content_emb = batch['content_emb'].to(self.device)
                collab_emb = batch['collab_emb'].to(self.device)
                item_ids = batch['item_id']
                tag_ids = batch['tag_ids']
                
                # Generate semantic IDs
                semantic_ids = self.model.generate_semantic_ids(content_emb, collab_emb)
                
                all_item_ids.append(item_ids.cpu().numpy())
                all_semantic_ids.append(semantic_ids.cpu().numpy())
                all_tag_ids.append(tag_ids.cpu().numpy())
        
        # Concatenate all batches
        all_item_ids = np.concatenate(all_item_ids, axis=0)
        all_semantic_ids = np.concatenate(all_semantic_ids, axis=0)
        all_tag_ids = np.concatenate(all_tag_ids, axis=0)
        
        n_items = len(all_item_ids)
        n_layers = all_semantic_ids.shape[1]
        
        self.logger.info(f"✓ Generated {n_items} semantic IDs with {n_layers} layers")
        
        # Analyze
        self.logger.info("\n" + "-" * 80)
        self.logger.info("Semantic ID Analysis")
        self.logger.info("-" * 80)
        
        # 1. Overall uniqueness
        id_tuples = [tuple(sid) for sid in all_semantic_ids]
        unique_ids = set(id_tuples)
        n_unique = len(unique_ids)
        uniqueness_rate = n_unique / n_items
        
        self.logger.info(f"\n1. Overall Uniqueness:")
        self.logger.info(f"   Total items: {n_items}")
        self.logger.info(f"   Unique IDs: {n_unique}")
        self.logger.info(f"   Uniqueness rate: {uniqueness_rate:.2%}")
        
        # 2. Collision analysis
        from collections import Counter
        id_counts = Counter(id_tuples)
        collisions = {k: v for k, v in id_counts.items() if v > 1}
        n_collision_groups = len(collisions)
        n_items_in_collisions = sum(collisions.values())
        
        self.logger.info(f"\n2. Collision Analysis:")
        self.logger.info(f"   Collision groups: {n_collision_groups}")
        self.logger.info(f"   Items in collisions: {n_items_in_collisions} ({n_items_in_collisions/n_items:.2%})")
        
        if n_collision_groups > 0:
            # Show top collisions
            top_collisions = sorted(collisions.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info(f"   Top 5 collisions:")
            for sid, count in top_collisions:
                self.logger.info(f"     ID {sid}: {count} items")
        
        # 3. Hierarchical overlap analysis (MOST IMPORTANT!)
        self.logger.info(f"\n3. Hierarchical Overlap Analysis:")
        self.logger.info(f"   (Lower overlap rate = higher diversity)")
        
        for layer in range(1, n_layers + 1):
            # Get prefix up to this layer
            prefixes = [tuple(sid[:layer]) for sid in all_semantic_ids]
            unique_prefixes = set(prefixes)
            n_unique_prefix = len(unique_prefixes)
            
            # Overlap rate: how many items share the same prefix
            overlap_rate = 1.0 - (n_unique_prefix / n_items)
            avg_items_per_prefix = n_items / n_unique_prefix
            
            self.logger.info(f"\n   Layer {layer} Prefix:")
            self.logger.info(f"     Unique prefixes: {n_unique_prefix} / {n_items}")
            self.logger.info(f"     Overlap rate: {overlap_rate:.4f}")
            self.logger.info(f"     Avg items per prefix: {avg_items_per_prefix:.2f}")
            
            # Show distribution
            prefix_counts = Counter(prefixes)
            singleton_count = sum(1 for count in prefix_counts.values() if count == 1)
            self.logger.info(f"     Singleton prefixes: {singleton_count} ({singleton_count/n_unique_prefix:.2%})")
        
        # 4. Codebook usage per layer
        self.logger.info(f"\n4. Codebook Usage per Layer:")
        codebook_sizes = self.model.get_codebook_sizes()
        
        for layer in range(n_layers):
            codes = all_semantic_ids[:, layer]
            unique_codes = len(set(codes))
            n_embed = codebook_sizes[layer]
            usage_rate = unique_codes / n_embed
            
            # Most/least used
            code_counts = Counter(codes)
            most_common = code_counts.most_common(3)
            unused_codes = n_embed - unique_codes
            
            self.logger.info(f"\n   Layer {layer + 1}:")
            self.logger.info(f"     Codebook size: {n_embed}")
            self.logger.info(f"     Used: {unique_codes} / {n_embed} ({usage_rate:.2%})")
            self.logger.info(f"     Unused: {unused_codes}")
            self.logger.info(f"     Most common codes: {[f'{code}({count})' for code, count in most_common]}")
        
        # 5. Tag alignment analysis
        self.logger.info(f"\n5. Tag Alignment Analysis:")
        self.logger.info(f"   (Measuring how well semantic IDs align with category tags)")
        
        for layer in range(min(n_layers, all_tag_ids.shape[1])):
            # Group items by tag
            tag_to_ids = defaultdict(list)
            for i in range(n_items):
                tag = all_tag_ids[i, layer]
                if tag > 0:  # Ignore PAD
                    sid_prefix = tuple(all_semantic_ids[i, :layer+1])
                    tag_to_ids[tag].append(sid_prefix)
            
            # Compute purity for each tag
            purities = []
            for tag, sid_list in tag_to_ids.items():
                if len(sid_list) > 1:
                    most_common_sid = Counter(sid_list).most_common(1)[0]
                    purity = most_common_sid[1] / len(sid_list)
                    purities.append(purity)
            
            if purities:
                avg_purity = np.mean(purities)
                self.logger.info(f"   Layer {layer + 1}: Avg prefix purity = {avg_purity:.4f}")
                self.logger.info(f"     (Same tag -> same prefix: {avg_purity:.2%})")
        
        # Save results in TIGER format (JSON)
        semantic_id_mappings = {}
        for i in range(n_items):
            item_id = str(all_item_ids[i])
            semantic_codes = all_semantic_ids[i].tolist()
            semantic_id_mappings[item_id] = semantic_codes
        
        output_file = self.output_dir / 'semantic_id_mappings.json'
        with open(output_file, 'w') as f:
            json.dump(semantic_id_mappings, f, indent=2)
        self.logger.info(f"\n✓ Semantic ID mappings saved to: {output_file}")
        
        # Also save as numpy for faster loading
        npy_file = self.output_dir / 'semantic_ids.npy'
        np.save(npy_file, all_semantic_ids)
        self.logger.info(f"✓ Semantic IDs (numpy) saved to: {npy_file}")
        
        # Save detailed report
        report = {
            'n_items': int(n_items),
            'n_layers': int(n_layers),
            'uniqueness': {
                'unique_ids': int(n_unique),
                'uniqueness_rate': float(uniqueness_rate),
                'collision_groups': int(n_collision_groups),
                'items_in_collisions': int(n_items_in_collisions)
            },
            'hierarchical_overlap': {},
            'codebook_usage': {}
        }
        
        # Add hierarchical overlap
        for layer in range(1, n_layers + 1):
            prefixes = [tuple(sid[:layer]) for sid in all_semantic_ids]
            unique_prefixes = len(set(prefixes))
            overlap_rate = 1.0 - (unique_prefixes / n_items)
            
            report['hierarchical_overlap'][f'layer_{layer}'] = {
                'unique_prefixes': int(unique_prefixes),
                'overlap_rate': float(overlap_rate),
                'avg_items_per_prefix': float(n_items / unique_prefixes)
            }
        
        # Add codebook usage
        codebook_sizes = self.model.get_codebook_sizes()
        for layer in range(n_layers):
            codes = all_semantic_ids[:, layer]
            unique_codes = len(set(codes))
            n_embed = codebook_sizes[layer]
            usage_rate = unique_codes / n_embed
            
            report['codebook_usage'][f'layer_{layer+1}'] = {
                'codebook_size': int(n_embed),
                'used': int(unique_codes),
                'total': int(n_embed),
                'usage_rate': float(usage_rate),
                'unused': int(n_embed - unique_codes)
            }
        
        report_file = self.output_dir / 'semantic_id_analysis.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"✓ Analysis report saved to: {report_file}")
        self.logger.info("=" * 80)
        
        # Auto-apply Sinkhorn algorithm to eliminate collisions
        self.apply_sinkhorn_reassignment()
    
    def analyze_gate_weights(self):
        """
        Analyze and visualize gate weights from trained model.
        Automatically called after training completion if using gated fusion.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Analyzing Gate Weights")
        self.logger.info("=" * 80)
        
        # Check if gated fusion is enabled
        if not self.config.get('use_gated_fusion', True):
            self.logger.info("Gated fusion not enabled, skipping gate analysis")
            return
        
        # Load best model
        best_model_path = self.output_dir / 'best_model.pt'
        if not best_model_path.exists():
            self.logger.warning(f"Best model not found at {best_model_path}, using current model")
        else:
            self.logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        # Create output directory for gate analysis
        gate_output_dir = self.output_dir / 'gate_analysis'
        gate_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect gate statistics
        self.logger.info("Collecting gate statistics...")
        
        all_gate_means = []
        all_item_ids = []
        
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Processing batches"):
                collab_emb = batch['collab_emb'].to(self.device)
                item_ids = batch['item_id'].cpu().numpy()
                
                # Get gate values
                gate = self.model.encoder.gate_network(collab_emb)  # (B, 768)
                mean_gate = gate.mean(dim=1).cpu().numpy()  # (B,)
                
                all_gate_means.append(mean_gate)
                all_item_ids.append(item_ids)
        
        # Concatenate results
        all_gate_means = np.concatenate(all_gate_means)
        all_item_ids = np.concatenate(all_item_ids)
        
        self.logger.info(f"✓ Collected gate statistics for {len(all_gate_means)} items")
        
        # Compute statistics
        self.logger.info("\n" + "-" * 80)
        self.logger.info("Gate Statistics")
        self.logger.info("-" * 80)
        self.logger.info(f"Mean:   {all_gate_means.mean():.4f}")
        self.logger.info(f"Median: {np.median(all_gate_means):.4f}")
        self.logger.info(f"Std:    {all_gate_means.std():.4f}")
        self.logger.info(f"Min:    {all_gate_means.min():.4f}")
        self.logger.info(f"Max:    {all_gate_means.max():.4f}")
        self.logger.info(f"Q25:    {np.percentile(all_gate_means, 25):.4f}")
        self.logger.info(f"Q75:    {np.percentile(all_gate_means, 75):.4f}")
        
        # Categorize items
        low_gate = (all_gate_means < 0.3).sum()
        mid_gate = ((all_gate_means >= 0.3) & (all_gate_means <= 0.7)).sum()
        high_gate = (all_gate_means > 0.7).sum()
        
        self.logger.info(f"\nGate Distribution:")
        self.logger.info(f"  Low trust (< 0.3):  {low_gate:6d} ({low_gate/len(all_gate_means)*100:.1f}%)")
        self.logger.info(f"  Medium (0.3-0.7):   {mid_gate:6d} ({mid_gate/len(all_gate_means)*100:.1f}%)")
        self.logger.info(f"  High trust (> 0.7): {high_gate:6d} ({high_gate/len(all_gate_means)*100:.1f}%)")
        
        # Create visualizations
        self.logger.info(f"\nCreating visualizations in {gate_output_dir}...")
        
        if MATPLOTLIB_AVAILABLE:
            try:
                # Plot 1: Histogram
                plt.figure(figsize=(10, 6))
                plt.hist(all_gate_means, bins=50, edgecolor='black', alpha=0.7)
                plt.axvline(all_gate_means.mean(), color='red', linestyle='--', 
                            label=f'Mean: {all_gate_means.mean():.3f}')
                plt.axvline(np.median(all_gate_means), color='green', linestyle='--',
                            label=f'Median: {np.median(all_gate_means):.3f}')
                plt.xlabel('Mean Gate Value (Trust Score)')
                plt.ylabel('Number of Items')
                plt.title('Distribution of Gate Values Across Items')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(gate_output_dir / 'gate_distribution.png', dpi=150)
                self.logger.info(f"  ✓ Saved: gate_distribution.png")
                plt.close()
                
                # Plot 2: CDF
                plt.figure(figsize=(10, 6))
                sorted_gates = np.sort(all_gate_means)
                cdf = np.arange(1, len(sorted_gates) + 1) / len(sorted_gates)
                plt.plot(sorted_gates, cdf, linewidth=2)
                plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50th percentile')
                plt.axvline(0.3, color='orange', linestyle='--', alpha=0.5, label='Low trust threshold')
                plt.axvline(0.7, color='green', linestyle='--', alpha=0.5, label='High trust threshold')
                plt.xlabel('Mean Gate Value (Trust Score)')
                plt.ylabel('Cumulative Probability')
                plt.title('Cumulative Distribution of Gate Values')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(gate_output_dir / 'gate_cdf.png', dpi=150)
                self.logger.info(f"  ✓ Saved: gate_cdf.png")
                plt.close()
                
                # Plot 3: Box plot
                plt.figure(figsize=(8, 6))
                plt.boxplot(all_gate_means, vert=True)
                plt.ylabel('Mean Gate Value (Trust Score)')
                plt.title('Box Plot of Gate Values')
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(gate_output_dir / 'gate_boxplot.png', dpi=150)
                self.logger.info(f"  ✓ Saved: gate_boxplot.png")
                plt.close()
                
            except Exception as e:
                self.logger.error(f"Error creating visualizations: {e}")
        else:
            self.logger.warning("matplotlib not available, skipping visualizations")
        
        # Save statistics to file
        stats_file = gate_output_dir / 'gate_statistics.txt'
        with open(stats_file, 'w') as f:
            f.write("Gate Weight Statistics\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total items: {len(all_gate_means)}\n\n")
            f.write(f"Mean:   {all_gate_means.mean():.6f}\n")
            f.write(f"Median: {np.median(all_gate_means):.6f}\n")
            f.write(f"Std:    {all_gate_means.std():.6f}\n")
            f.write(f"Min:    {all_gate_means.min():.6f}\n")
            f.write(f"Max:    {all_gate_means.max():.6f}\n")
            f.write(f"Q25:    {np.percentile(all_gate_means, 25):.6f}\n")
            f.write(f"Q75:    {np.percentile(all_gate_means, 75):.6f}\n\n")
            f.write("Gate Distribution:\n")
            f.write(f"  Low trust (< 0.3):  {low_gate:6d} ({low_gate/len(all_gate_means)*100:.2f}%)\n")
            f.write(f"  Medium (0.3-0.7):   {mid_gate:6d} ({mid_gate/len(all_gate_means)*100:.2f}%)\n")
            f.write(f"  High trust (> 0.7): {high_gate:6d} ({high_gate/len(all_gate_means)*100:.2f}%)\n")
        
        self.logger.info(f"  ✓ Saved: gate_statistics.txt")
        
        # Save gate values for each item (for further analysis)
        gate_data_file = gate_output_dir / 'gate_values.npz'
        np.savez(
            gate_data_file,
            item_ids=all_item_ids,
            gate_means=all_gate_means
        )
        self.logger.info(f"  ✓ Saved: gate_values.npz")
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("✓ Gate analysis completed!")
        self.logger.info(f"Output directory: {gate_output_dir}")
        self.logger.info("=" * 80)
    
    def apply_sinkhorn_reassignment(self):
        """
        Automatically apply Sinkhorn algorithm to eliminate ID collisions.
        Called after semantic ID generation.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Applying Sinkhorn Algorithm for Collision Elimination")
        self.logger.info("=" * 80)
        
        # Check if semantic_id_mappings.json exists
        semantic_ids_file = self.output_dir / 'semantic_id_mappings.json'
        if not semantic_ids_file.exists():
            self.logger.warning(f"Semantic IDs file not found: {semantic_ids_file}")
            self.logger.warning("Skipping Sinkhorn reassignment")
            return
        
        # Backup original file
        original_backup = self.output_dir / 'semantic_id_mappings_original.json'
        if not original_backup.exists():
            import shutil
            shutil.copy2(semantic_ids_file, original_backup)
            self.logger.info(f"  ✓ Backed up original IDs to: {original_backup}")
        
        # Import Sinkhorn reassigner
        try:
            from sinkhorn_reassignment import SinkhornIDReassigner
        except ImportError:
            self.logger.error("Cannot import SinkhornIDReassigner, skipping reassignment")
            return
        
        # Initialize reassigner
        try:
            reassigner = SinkhornIDReassigner(
                semantic_ids_path=str(semantic_ids_file),
                checkpoint_path=str(self.output_dir / 'best_model.pt'),
                data_dir=self.config['data_path'],
                device=self.config.get('device', 'cuda'),
                output_dir=str(self.output_dir)
            )
            
            # Run reassignment with variable codebook sizes
            codebook_sizes = self.model.get_codebook_sizes()
            new_semantic_ids = reassigner.run(codebook_sizes=codebook_sizes, max_iterations=10)
            
            # Verify final uniqueness
            from collections import Counter
            id_tuples = [tuple(sid) for sid in new_semantic_ids]
            n_unique = len(set(id_tuples))
            uniqueness_rate = n_unique / len(id_tuples)
            
            if n_unique == len(new_semantic_ids):
                self.logger.info("\n" + "=" * 80)
                self.logger.info("✅ SUCCESS: 100% uniqueness achieved after Sinkhorn reassignment!")
                self.logger.info("=" * 80)
            else:
                self.logger.warning(f"\n⚠️  Warning: {len(new_semantic_ids) - n_unique} collisions still remain")
                self.logger.warning(f"   Final uniqueness: {uniqueness_rate:.2%}")
            
        except Exception as e:
            self.logger.error(f"Error during Sinkhorn reassignment: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.warning("Continuing without Sinkhorn reassignment")
    
    def train(self):
        """Main training loop"""
        self.logger.info("=" * 80)
        self.logger.info("Starting PRISM training")
        self.logger.info("=" * 80)
        
        start_epoch = self.current_epoch + 1
        end_epoch = self.config['epochs']
        
        for epoch in range(start_epoch, end_epoch + 1):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Log metrics
            self.logger.info(f"\nEpoch {epoch} Summary:")
            self.logger.info(f"  Total Loss: {train_metrics['total_loss']:.4f}")
            
            # Reconstruction loss - different format for single vs dual decoder
            if self.config.get('use_dual_decoder', True):
                # Dual decoder mode: show separate losses
                self.logger.info(f"  Recon Loss: {train_metrics['recon_total']:.4f} "
                               f"(content: {train_metrics['recon_content']:.4f}, "
                               f"collab: {train_metrics['recon_collab']:.4f})")
            else:
                # Single decoder mode: emphasize concat loss (what's actually backpropagated)
                self.logger.info(f"  Recon Loss: {train_metrics['recon_total']:.4f} "
                               f"(concat: {train_metrics.get('recon_concat', train_metrics['recon_total']):.4f})")
                self.logger.info(f"    [Monitor only - content: {train_metrics['recon_content']:.4f}, "
                               f"collab: {train_metrics['recon_collab']:.4f}]")
            
            if 'class_total' in train_metrics:
                self.logger.info(f"  Class Loss: {train_metrics['class_total']:.4f}")
                for i in range(self.config['n_layers']):
                    if f'acc_layer{i+1}' in train_metrics:
                        self.logger.info(f"    Layer {i+1} Acc: {train_metrics[f'acc_layer{i+1}']:.4f}")
            
            if 'anchor_total' in train_metrics:
                self.logger.info(f"  Anchor Loss: {train_metrics['anchor_total']:.4f}")
            
            if 'balance_total' in train_metrics:
                self.logger.info(f"  Balance Loss: {train_metrics['balance_total']:.4f}")
                for i in range(self.config['n_layers']):
                    if f'usage_layer{i+1}' in train_metrics:
                        self.logger.info(f"    Layer {i+1} Usage: {train_metrics[f'usage_layer{i+1}']:.4f}")
            
            # Perplexity
            for i in range(self.config['n_layers']):
                if f'perplexity_layer{i+1}' in train_metrics:
                    self.logger.info(f"  Perplexity Layer {i+1}: {train_metrics[f'perplexity_layer{i+1}']:.2f}")
            
            # Gate statistics (if using gated fusion)
            if 'gate_mean' in train_metrics:
                self.logger.info(
                    f"  Gate Stats: "
                    f"mean={train_metrics['gate_mean']:.3f}, "
                    f"median={train_metrics['gate_median']:.3f}, "
                    f"range=[{train_metrics['gate_min']:.3f}, {train_metrics['gate_max']:.3f}]"
                )
            
            # Gate supervision loss (if enabled)
            if 'gate_supervision' in train_metrics:
                self.logger.info(
                    f"  Gate Supervision: "
                    f"loss={train_metrics['gate_supervision']:.4f}, "
                    f"diversity={train_metrics.get('gate_diversity', 0):.4f}, "
                    f"variance={train_metrics.get('gate_variance', 0):.4f}, "
                    f"std={train_metrics.get('gate_std', 0):.3f}"
                )
            
            self.logger.info(
                "  Loss scales: "
                f"recon={train_metrics.get('loss_weight_recon', 1.0):.2f}, "
                f"class={train_metrics.get('loss_weight_class', 0.0):.2f}, "
                f"anchor={train_metrics.get('loss_weight_anchor', 0.0):.2f}, "
                f"balance={train_metrics.get('loss_weight_balance', 0.0):.2f}, "
                f"commit={train_metrics.get('loss_weight_commitment', 1.0):.2f}"
            )
            
            # Track history
            for key, value in train_metrics.items():
                self.train_history[key].append(value)
            
            # Check for improvement
            should_stop, primary_improved = self._update_early_stopping(train_metrics, epoch)
            if primary_improved:
                self.logger.info(f"  ✓ New best loss: {self.best_loss:.4f}")
            else:
                self.logger.info(
                    f"  Patience: {self.patience_counter}/"
                    f"{self.config.get('early_stop_patience', float('inf'))}"
                )
            
            # Save checkpoint
            should_save_regular = (epoch % self.config.get('save_every', 50) == 0)
            if should_save_regular or primary_improved:
                self.save_checkpoint(epoch, train_metrics, primary_improved, save_regular=should_save_regular)
            
            # Early stopping
            if should_stop:
                self.logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Final save (always save as regular checkpoint)
        self.save_checkpoint(epoch, train_metrics, is_best=False, save_regular=True)
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        self.logger.info("=" * 80)
        self.logger.info("Training completed!")
        self.logger.info(f"Best loss: {self.best_loss:.4f}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)
        
        # Auto-analyze gate weights if using gated fusion
        if self.config.get('use_gated_fusion', True):
            self.analyze_gate_weights()
        
        # Save item-codebook mappings (NEW FEATURE)
        self.save_item_codebook_mappings()
        
        # Auto-generate semantic IDs and analysis
        self.generate_semantic_ids_and_analyze()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PRISM')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum number of items (for testing)')
    
    # Model arguments
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of RQ layers')
    parser.add_argument('--n_embed', type=int, default=256,
                       help='Default codebook size per layer (used if n_embed_per_layer not specified)')
    parser.add_argument('--n_embed_per_layer', type=str, default=None,
                       help='Variable codebook sizes per layer (comma-separated, e.g., "128,256,512")')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent/codebook dimension')
    parser.add_argument('--content_dim', type=int, default=768,
                       help='Content embedding dimension')
    parser.add_argument('--collab_dim', type=int, default=64,
                       help='Collaborative embedding dimension')
    parser.add_argument('--no_gated_fusion', action='store_true',
                       help='Disable gated additive fusion (use simple concatenation instead)')
    parser.add_argument('--no_dual_decoder', action='store_true',
                       help='Disable dual decoder heads (use single decoder for concatenated embedding instead)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=500,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm')
    
    # Loss weights
    parser.add_argument('--lambda_content', type=float, default=1.0,
                       help='Weight for content reconstruction')
    parser.add_argument('--lambda_collab', type=float, default=1.0,
                       help='Weight for collaborative reconstruction')
    parser.add_argument('--delta_weight', type=float, default=1.0,
                       help='Weight for classification loss')
    parser.add_argument('--beta', type=float, default=0.25,
                       help='Commitment loss weight')
    parser.add_argument('--gamma_weight', type=float, default=1.0,
                       help='Balance loss weight (recommend: 0.01-0.1)')
    parser.add_argument('--beta_anchor_weight', type=float, default=0.1,
                       help='Anchor loss weight (recommend: 0.05-0.2)')
    
    # Quantization arguments
    parser.add_argument('--use_ema', action='store_true',
                       help='Use EMA for codebook updates')
    parser.add_argument('--ema_decay', type=float, default=0.99,
                       help='EMA decay rate')
    parser.add_argument('--quantize_mode', type=str, default='rotation',
                       choices=['ste', 'rotation', 'gumbel_softmax'],
                       help='Quantization mode')
    
    # Gate supervision arguments
    parser.add_argument('--use_gate_supervision', action='store_true',
                       help='Use popularity-based gate supervision to improve gate diversity')
    parser.add_argument('--gate_supervision_weight', type=float, default=0.1,
                       help='Weight for gate supervision loss')
    parser.add_argument('--gate_diversity_weight', type=float, default=0.5,
                       help='Weight for gate diversity regularization')
    parser.add_argument('--gate_target_std', type=float, default=0.2,
                       help='Target standard deviation for gate values')
    
    # Scheduler arguments
    parser.add_argument('--use_scheduler', action='store_true',
                       help='Use learning rate scheduler')
    parser.add_argument('--scheduler_type', type=str, default='warmup_cosine',
                       choices=['warmup_cosine', 'exponential'],
                       help='Scheduler type')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    
    # Early stopping
    parser.add_argument('--early_stop_patience', type=int, default=200,
                       help='Early stopping patience')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                       help='Minimum delta for early stopping')
    
    # Other arguments
    parser.add_argument('--save_every', type=int, default=50,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Curriculum learning arguments
    parser.add_argument('--no_curriculum', action='store_true',
                       help='Disable curriculum learning for loss weights')
    parser.add_argument('--curriculum_strategy', type=str, default='default',
                       choices=['default', 'similarity_first'],
                       help='Curriculum learning strategy: '
                            'default (class->balance->anchor) or '
                            'similarity_first (anchor->balance, learn similarity then distinctiveness)')
    parser.add_argument('--curriculum_warmup_epochs', type=int, default=5,
                       help='Epochs to focus on reconstruction before turning on other losses')
    parser.add_argument('--curriculum_phase1_duration', type=int, default=None,
                       help='Duration of Phase 1 for similarity_first strategy (default: epochs/3)')
    parser.add_argument('--curriculum_class_ramp', type=int, default=15,
                       help='Number of epochs to ramp classification loss to full weight')
    parser.add_argument('--curriculum_anchor_delay', type=int, default=15,
                       help='Epoch to start introducing anchor loss')
    parser.add_argument('--curriculum_anchor_ramp', type=int, default=20,
                       help='Number of epochs to ramp anchor loss to full weight')
    parser.add_argument('--curriculum_anchor_decay', type=float, default=0.5,
                       help='Anchor weight decay factor in Phase 2 (similarity_first strategy)')
    parser.add_argument('--curriculum_balance_delay', type=int, default=10,
                       help='Epoch to start encouraging balance loss')
    parser.add_argument('--curriculum_balance_ramp', type=int, default=12,
                       help='Number of epochs to ramp balance loss to full weight')
    parser.add_argument('--curriculum_class_target', type=float, default=1.0,
                       help='Target scale for classification loss after ramp-up')
    parser.add_argument('--curriculum_anchor_target', type=float, default=1.0,
                       help='Target scale for anchor loss after ramp-up')
    parser.add_argument('--curriculum_balance_target', type=float, default=1.0,
                       help='Target scale for balance loss after ramp-up')
    parser.add_argument('--curriculum_anchor_ratio', type=float, default=0.55,
                       help='Max allowed ratio of anchor loss to total loss before penalizing')
    parser.add_argument('--curriculum_anchor_penalty', type=float, default=0.7,
                       help='Multiplicative penalty applied when anchor loss dominates')
    parser.add_argument('--curriculum_min_perplexity_ratio', type=float, default=0.35,
                       help='Minimum acceptable perplexity ratio before triggering collapse guard')
    parser.add_argument('--perplexity_collapse_patience', type=int, default=3,
                       help='Number of consecutive low-perplexity epochs before intervention')
    parser.add_argument('--early_stop_cooldown', type=int, default=3,
                       help='Extra epochs after warmup before early stopping can trigger')
    
    # Hierarchical k-means initialization arguments
    parser.add_argument('--no_hierarchical_kmeans_init', action='store_true',
                       help='Disable hierarchical k-means codebook initialization')
    parser.add_argument('--kmeans_init_samples', type=int, default=8192,
                       help='Number of items to sample for k-means initialization')
    parser.add_argument('--kmeans_batch_size', type=int, default=1024,
                       help='Batch size when encoding samples for k-means')
    parser.add_argument('--kmeans_random_state', type=int, default=42,
                       help='Random seed for k-means clustering')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Set use_gated_fusion (default True unless --no_gated_fusion is specified)
    config['use_gated_fusion'] = not args.no_gated_fusion
    
    # Set use_dual_decoder (default True unless --no_dual_decoder is specified)
    config['use_dual_decoder'] = not args.no_dual_decoder
    
    # Parse n_embed_per_layer if provided
    if args.n_embed_per_layer is not None:
        try:
            n_embed_per_layer = [int(x.strip()) for x in args.n_embed_per_layer.split(',')]
            if len(n_embed_per_layer) != args.n_layers:
                raise ValueError(
                    f"n_embed_per_layer must have {args.n_layers} values, "
                    f"got {len(n_embed_per_layer)}"
                )
            config['n_embed_per_layer'] = n_embed_per_layer
            print(f"Using variable codebook sizes: {n_embed_per_layer}")
        except Exception as e:
            print(f"Error parsing n_embed_per_layer: {e}")
            print(f"Using uniform codebook size: {args.n_embed}")
            config['n_embed_per_layer'] = None
    else:
        config['n_embed_per_layer'] = None
    
    # Initialize trainer
    trainer = PRISMTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()

