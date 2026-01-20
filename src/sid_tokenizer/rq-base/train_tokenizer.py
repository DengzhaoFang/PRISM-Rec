#!/usr/bin/env python3
"""
Semantic ID Tokenizer Training Script

Supports multiple tokenization modes for recommendation systems:
- TIGER: Uses RQ-VAE for hierarchical semantic ID generation
- KMEANS: Uses RQ-KMeans for clustering-based tokenization

This script trains tokenizers on processed Amazon dataset with item embeddings.
"""

import os
import argparse
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

from tiger.RQ_VAE import RQVAE, QuantizeMode
from rqkmeans.RQ_KMeans import RQKMeans

# Import schedulers
try:
    from schedulers import WarmupCosineScheduler, ExponentialSchedulerWithWarmup
    SCHEDULERS_AVAILABLE = True
except ImportError:
    SCHEDULERS_AVAILABLE = False


class ItemEmbeddingDataset(Dataset):
    """Dataset for loading item embeddings from processed data"""
    
    def __init__(self, embedding_file: str, max_items: Optional[int] = None):
        """
        Initialize dataset.
        
        Args:
            embedding_file: Path to item_emb_all.parquet or item_emb.parquet file
            max_items: Maximum number of items to load (for testing)
        """
        self.embeddings_df = pd.read_parquet(embedding_file)
        
        if max_items is not None:
            self.embeddings_df = self.embeddings_df.head(max_items)
        
        # Convert embeddings from list to tensor
        self.embeddings = torch.stack([
            torch.tensor(emb, dtype=torch.float32) 
            for emb in self.embeddings_df['embedding']
        ])
        
        self.item_ids = self.embeddings_df['ItemID'].values
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'item_id': self.item_ids[idx],
            'embedding': self.embeddings[idx]
        }


class TokenizerTrainer:
    """Main trainer class for semantic ID tokenizers"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.setup_logging()
        
    def _calculate_temperature(
        self,
        step: int,
        schedule: str,
        init_temp: float,
        min_temp: float,
        warmup_steps: int,
        total_steps: int
    ) -> float:
        """
        Calculate temperature using various scheduling strategies.
        
        Args:
            step: Current training step
            schedule: Schedule type ('cosine', 'exponential', 'constant', 'piecewise')
            init_temp: Initial temperature
            min_temp: Minimum temperature
            warmup_steps: Number of warmup steps (for cosine/piecewise)
            total_steps: Total training steps
            
        Returns:
            Current temperature value
        """
        if schedule == 'constant':
            #  fixed temperature
            return min_temp
        
        elif schedule == 'cosine':
            # Cosine annealing with warmup
            if step < warmup_steps:
                # Linear warmup from init_temp to init_temp
                return init_temp
            else:
                # Cosine decay from init_temp to min_temp
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                progress = min(progress, 1.0)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                return min_temp + (init_temp - min_temp) * cosine_decay
        
        elif schedule == 'piecewise':
            # Piecewise constant: high temp for exploration, then low temp for exploitation
            if step < total_steps * 0.3:
                return init_temp
            elif step < total_steps * 0.7:
                return (init_temp + min_temp) / 2
            else:
                return min_temp
        
        elif schedule == 'exponential':
            # Original exponential decay (less stable)
            anneal_rate = self.config.get('temperature_anneal_rate', 0.00003)
            return max(min_temp, init_temp * np.exp(-anneal_rate * step))
        
        else:
            raise ValueError(f"Unknown temperature schedule: {schedule}")
    
    def setup_logging(self):
        """Setup logging configuration with timestamped log files in output directory"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        # Use output directory for logs (same as model files)
        logs_dir = self.config['output_dir']
        os.makedirs(logs_dir, exist_ok=True)
        
        # Extract dataset name from data path
        data_path = self.config['data_path']
        dataset_name = os.path.basename(data_path.rstrip('/'))
        if not dataset_name:  # Handle case where path ends with '/'
            dataset_name = os.path.basename(os.path.dirname(data_path))
        
        # Generate timestamped log filename with key information
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mode = self.config.get('mode', 'unknown')
        epochs = self.config.get('epochs', 'unknown')
        batch_size = self.config.get('batch_size', 'unknown')
        
        log_filename = f"{timestamp}_{dataset_name}_{mode}_ep{epochs}_bs{batch_size}.log"
        log_path = os.path.join(logs_dir, log_filename)
        
        # Clear any existing handlers to avoid duplication
        logging.getLogger().handlers.clear()
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, mode='w', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log the log file location for reference
        self.logger.info(f"Logging to file: {log_path}")
        self.logger.info(f"Dataset: {dataset_name}, Mode: {mode}, Epochs: {epochs}, Batch size: {batch_size}")
        
    def load_data(self) -> DataLoader:
        """Load and prepare data for training"""
        self.logger.info(f"Loading embeddings from: {self.config['data_path']}")
        
        dataset = ItemEmbeddingDataset(
            embedding_file=os.path.join(self.config['data_path'], self.config.get('embedding_file', 'item_emb_all.parquet')),
            max_items=self.config.get('max_items')
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.logger.info(f"Loaded {len(dataset)} items")
        return dataloader
    
    def create_model(self, embedding_dim: int) -> nn.Module:
        """Create model based on training mode"""
        mode = self.config['mode'].lower()
        
        if mode == 'tiger':
            # Get quantize mode
            quantize_mode_str = self.config.get('quantize_mode', 'gumbel_softmax')
            quantize_mode = (QuantizeMode.GUMBEL_SOFTMAX if quantize_mode_str == 'gumbel_softmax' 
                           else QuantizeMode.STE)
            
            # Optimized RQ-VAE configuration
            model = RQVAE(
                input_dim=embedding_dim,
                latent_dim=self.config.get('latent_dim', 32),
                n_embed=self.config.get('n_embed', 256),
                n_layers=self.config.get('n_layers', 3),
                beta=self.config.get('beta', 0.25),
                use_ema=self.config.get('use_ema', True),
                decay=self.config.get('ema_decay', 0.99),
                commitment_weight=self.config.get('commitment_weight', 1.0),
                reconstruction_weight=self.config.get('reconstruction_weight', 1.0),
                quantize_mode=quantize_mode
            )
            self.logger.info("Created optimized RQ-VAE model")
            self.logger.info(f"  Architecture: {embedding_dim} -> 512 -> 256 -> 128 -> {model.latent_dim}")
            self.logger.info(f"  Quantization: {model.n_layers} layers, {model.n_embed} codes per layer")
            self.logger.info(f"  Beta coefficient: {model.beta}")
            self.logger.info(f"  Use EMA: {model.use_ema}")
            if model.use_ema:
                self.logger.info(f"  EMA decay: {model.decay}")
                self.logger.info(f"  Quantization: STE (Straight-Through Estimator)")
            else:
                self.logger.info(f"  Quantization: {quantize_mode.value}")
            self.logger.info(f"  Normalize residuals: {model.normalize_residuals} {'(CRITICAL for stability!)' if model.normalize_residuals else '(WARNING: May cause collapse!)'}")
            self.logger.info(f"  Activation: SiLU (optimized)")
            
        elif mode == 'kmeans':
            model = RQKMeans(
                n_clusters=self.config.get('n_clusters', 512),
                n_features=embedding_dim,
                n_layers=self.config.get('n_layers', 4),
                max_iters=self.config.get('max_iters', 100),
                init_method=self.config.get('init_method', 'kmeans++')
            )
            self.logger.info("Created RQ-KMeans model")
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        return model.to(self.device)
    
    def train_tiger_mode(self, model: RQVAE, dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """Train TIGER mode using RQ-VAE with optimized settings and EBODA-style evaluation"""
        # Use AdamW optimizer (more stable than Adagrad)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),  # Lower learning rate for stability
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        self.logger.info(f"Using AdamW optimizer with lr={self.config.get('learning_rate', 1e-4)}")
        
        # Advanced learning rate scheduler
        scheduler = None
        if self.config.get('use_scheduler', False):
            if SCHEDULERS_AVAILABLE and self.config.get('scheduler_type') == 'warmup_cosine':
                # Calculate total steps
                total_steps = self.config['epochs'] * len(dataloader)
                warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
                
                scheduler = WarmupCosineScheduler(
                    optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    min_lr_ratio=self.config.get('min_lr_ratio', 0.1),
                    num_cycles=0.5
                )
                self.logger.info(f"Using WarmupCosineScheduler (warmup: {warmup_steps}, total: {total_steps})")
            else:
                # Fallback to standard cosine annealing
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config['epochs'],
                    eta_min=self.config.get('min_lr', 1e-6)
                )
                self.logger.info("Using CosineAnnealingLR scheduler")
        
        # Temperature scheduling for Gumbel-Softmax
        # Use cosine annealing with warmup (more stable than exponential decay)
        init_temperature = self.config.get('init_temperature', 1.0)
        min_temperature = self.config.get('min_temperature', 0.2)
        temperature_schedule = self.config.get('temperature_schedule', 'cosine')  # 'cosine', 'exponential', or 'constant'
        warmup_steps = self.config.get('temperature_warmup_steps', 1000)
        total_steps = self.config['epochs'] * len(dataloader)
        
        self.logger.info(f"Temperature schedule: {temperature_schedule}")
        self.logger.info(f"  Initial: {init_temperature}, Minimum: {min_temperature}")
        if temperature_schedule == 'cosine':
            self.logger.info(f"  Warmup steps: {warmup_steps}, Total steps: {total_steps}")
        
        model.train()
        training_stats = []
        global_step = 0
        
        # Early stopping state
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = self.config.get('early_stop_patience', 50)
        early_stop_min_delta = self.config.get('early_stop_min_delta', 1e-4)
        
        for epoch in range(self.config['epochs']):
            epoch_stats = {
                'epoch': epoch, 'total_loss': 0, 'recon_loss': 0, 'vq_loss': 0,
                'codebook_loss': 0, 'commitment_loss': 0, 'codebook_usage': 0,
                'duplicate_rate_pre': 0, 'duplicate_rate_post': 0
            }
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
            
            for batch in progress_bar:
                embeddings = batch['embedding'].to(self.device)
                
                # Calculate current temperature using configured schedule
                temperature = self._calculate_temperature(
                    global_step, 
                    temperature_schedule,
                    init_temperature,
                    min_temperature,
                    warmup_steps,
                    total_steps
                )
                
                optimizer.zero_grad()
                outputs = model(embeddings, temperature=temperature)
                
                loss = outputs['total_loss']
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'])
                
                optimizer.step()
                
                # Step scheduler (per-batch for step-based schedulers like WarmupCosine)
                if scheduler and isinstance(scheduler, (WarmupCosineScheduler if SCHEDULERS_AVAILABLE else type(None))):
                    scheduler.step()
                
                global_step += 1
                
                # Update stats
                epoch_stats['total_loss'] += outputs['total_loss'].item()
                epoch_stats['recon_loss'] += outputs['recon_loss'].item()
                epoch_stats['vq_loss'] += outputs['vq_loss'].item()
                epoch_stats['codebook_loss'] += outputs['codebook_loss'].item()
                epoch_stats['commitment_loss'] += outputs['commitment_loss'].item()
                epoch_stats['codebook_usage'] += outputs['codebook_usage']
                epoch_stats['duplicate_rate_pre'] += outputs['duplicate_rate_pre']
                epoch_stats['duplicate_rate_post'] += outputs['duplicate_rate_post']
                
                # EBODA-style detailed monitoring (computed per batch)
                with torch.no_grad():
                    codes = outputs['codes']  # (batch_size, n_layers)
                    
                    # Per-layer codebook usage
                    for layer_idx in range(model.n_layers):
                        layer_codes = codes[:, layer_idx]
                        unique_codes = torch.unique(layer_codes)
                        usage = len(unique_codes) / model.n_embed
                        key = f'codebook_usage_layer_{layer_idx}'
                        if key not in epoch_stats:
                            epoch_stats[key] = 0
                        epoch_stats[key] += usage
                
                num_batches += 1
                
                progress_bar.set_postfix({
                    'Loss': f"{outputs['total_loss'].item():.4f}",
                    'Recon': f"{outputs['recon_loss'].item():.4f}",
                    'CB': f"{outputs['codebook_loss'].item():.4f}",
                    'CM': f"{outputs['commitment_loss'].item():.4f}",
                    'Usage': f"{outputs['codebook_usage']:.3f}",
                    'Dup': f"{outputs['duplicate_rate_pre']:.3f}"
                })
            
            # Average stats
            for key in epoch_stats:
                if key != 'epoch':
                    epoch_stats[key] /= num_batches
            
            # Record current temperature
            epoch_stats['temperature'] = temperature
            
            training_stats.append(epoch_stats)
            
            # Step scheduler (per-epoch for epoch-based schedulers)
            if scheduler and not isinstance(scheduler, (WarmupCosineScheduler if SCHEDULERS_AVAILABLE else type(None))):
                scheduler.step()
            
            #  Compute global diversity metrics after epoch
            with torch.no_grad():
                # Collect all codes from the epoch
                all_codes = []
                for batch in dataloader:
                    embeddings = batch['embedding'].to(self.device)
                    codes_batch = model.encode_to_codes(embeddings, apply_post_id=False, temperature=temperature)
                    all_codes.append(codes_batch)
                    if len(all_codes) >= 10:  # Sample first 10 batches for efficiency
                        break
                
                if all_codes:
                    all_codes = torch.cat(all_codes, dim=0)
                    
                    # Unique ID proportion (p_unique_ids)
                    unique_ids = torch.unique(all_codes, dim=0)
                    p_unique_ids = len(unique_ids) / len(all_codes)
                    epoch_stats['p_unique_ids'] = p_unique_ids
                    
                    # RQ-VAE entropy
                    _, counts = torch.unique(all_codes, dim=0, return_counts=True)
                    p = counts.float() / all_codes.shape[0]
                    entropy = -(p * torch.log(p + 1e-10)).sum()
                    epoch_stats['rqvae_entropy'] = entropy.item()
            
            # Log epoch stats
            self.logger.info(
                f"Epoch {epoch+1}: Loss={epoch_stats['total_loss']:.4f}, "
                f"Recon={epoch_stats['recon_loss']:.4f}, "
                f"CB={epoch_stats['codebook_loss']:.4f}, "
                f"CM={epoch_stats['commitment_loss']:.4f}"
            )
            self.logger.info(
                f"  Codebook usage (avg): {epoch_stats['codebook_usage']:.4f}, "
                f"Temp={epoch_stats['temperature']:.4f}"
            )
            
            # Log per-layer codebook usage (EBODA-style)
            usage_str = ", ".join([
                f"L{i}={epoch_stats.get(f'codebook_usage_layer_{i}', 0):.4f}"
                for i in range(model.n_layers)
            ])
            self.logger.info(f"  Per-layer usage: {usage_str}")
            
            self.logger.info(
                f"  Duplicate rates - Pre: {epoch_stats['duplicate_rate_pre']:.4f}, "
                f"Post: {epoch_stats['duplicate_rate_post']:.4f}"
            )
            
            # Log EBODA-style diversity metrics
            if 'p_unique_ids' in epoch_stats:
                self.logger.info(
                    f"  Unique IDs: {epoch_stats['p_unique_ids']:.4f}, "
                    f"Entropy: {epoch_stats.get('rqvae_entropy', 0):.4f}"
                )
            
            #  Evaluate on validation set
            eval_stats = {}
            if eval_dataloader and (epoch + 1) % self.config.get('eval_every', 10) == 0:
                self.logger.info(f"\nRunning validation at epoch {epoch+1}...")
                eval_stats = self.evaluate(model, eval_dataloader, temperature)
                
                self.logger.info(f"Validation metrics:")
                self.logger.info(f"  Loss: {eval_stats['loss']:.4f}")
                self.logger.info(f"  Recon: {eval_stats['recon_loss']:.4f}")
                self.logger.info(f"  Codebook usage: {eval_stats['codebook_usage']:.4f}")
                self.logger.info(f"  Duplicate rate: {eval_stats['duplicate_rate_pre']:.4f}")
                if 'p_unique_ids' in eval_stats:
                    self.logger.info(f"  Unique IDs: {eval_stats['p_unique_ids']:.4f}")
                    self.logger.info(f"  Entropy: {eval_stats['entropy']:.4f}")
                
                # Log per-layer usage
                layer_usage = [eval_stats.get(f'layer_{i}_usage', 0) for i in range(model.n_layers)]
                usage_str = ", ".join([f"L{i}={u:.3f}" for i, u in enumerate(layer_usage)])
                self.logger.info(f"  Layer usage: {usage_str}")
            
            # Early stopping check (use validation loss if available, else training loss)
            check_loss = eval_stats.get('loss', epoch_stats['total_loss'])
            if early_stop_patience > 0:
                if best_loss - check_loss > early_stop_min_delta:
                    best_loss = check_loss
                    patience_counter = 0
                    self.logger.info(f"  ✓ New best loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        self.logger.info(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                        self.logger.info(f"  Loss has not improved for {early_stop_patience} epochs")
                        break
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                # Combine train and eval stats
                combined_stats = {**epoch_stats, **{f'eval_{k}': v for k, v in eval_stats.items()}}
                self.save_checkpoint(model, optimizer, epoch, combined_stats)
        
        return {'training_stats': training_stats}
    
    def train_kmeans_mode(self, model: RQKMeans, dataloader: DataLoader) -> Dict[str, Any]:
        """Train K-means mode using RQ-KMeans"""
        # Collect all embeddings for batch training
        self.logger.info("Collecting embeddings for K-means training...")
        all_embeddings = []
        
        for batch in tqdm(dataloader, desc="Loading data"):
            embeddings = batch['embedding']
            all_embeddings.append(embeddings)
            
        all_embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
        self.logger.info(f"Training K-means on {all_embeddings.shape[0]} embeddings")
        
        # Train model
        model.fit(all_embeddings)
        
        # Compute reconstruction loss for evaluation
        codes, reconstructed = model(all_embeddings)
        recon_loss = model.get_reconstruction_loss(all_embeddings)
        
        self.logger.info(f"Training completed. Reconstruction loss: {recon_loss:.4f}")
        
        return {
            'reconstruction_loss': recon_loss.item(),
            'n_samples': all_embeddings.shape[0]
        }
    
    def evaluate(self, model, dataloader, temperature: float = 0.2) -> Dict[str, float]:
        """
         Comprehensive evaluation on validation set.
        
        Args:
            model: RQ-VAE model
            dataloader: Validation dataloader
            temperature: Temperature for quantization
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        eval_stats = {
            'loss': 0, 'recon_loss': 0, 'vq_loss': 0,
            'codebook_loss': 0, 'commitment_loss': 0,
            'codebook_usage': 0, 'duplicate_rate_pre': 0
        }
        num_batches = 0
        
        all_codes = []
        
        with torch.no_grad():
            for batch in dataloader:
                embeddings = batch['embedding'].to(self.device)
                outputs = model(embeddings, temperature=temperature)
                
                # Accumulate losses
                eval_stats['loss'] += outputs['total_loss'].item()
                eval_stats['recon_loss'] += outputs['recon_loss'].item()
                eval_stats['vq_loss'] += outputs['vq_loss'].item()
                eval_stats['codebook_loss'] += outputs['codebook_loss'].item()
                eval_stats['commitment_loss'] += outputs['commitment_loss'].item()
                eval_stats['codebook_usage'] += outputs['codebook_usage']
                eval_stats['duplicate_rate_pre'] += outputs['duplicate_rate_pre']
                
                # Collect codes for diversity metrics
                all_codes.append(outputs['codes'])
                num_batches += 1
                
                # Limit evaluation batches for speed
                if num_batches >= 50:
                    break
        
        # Average metrics
        for key in eval_stats:
            eval_stats[key] /= num_batches
        
        # Compute global diversity metrics
        if all_codes:
            all_codes_concat = torch.cat(all_codes, dim=0)
            
            # Unique ID proportion
            unique_ids = torch.unique(all_codes_concat, dim=0)
            p_unique_ids = len(unique_ids) / len(all_codes_concat)
            eval_stats['p_unique_ids'] = p_unique_ids
            
            # Entropy
            _, counts = torch.unique(all_codes_concat, dim=0, return_counts=True)
            p = counts.float() / all_codes_concat.shape[0]
            entropy = -(p * torch.log(p + 1e-10)).sum()
            eval_stats['entropy'] = entropy.item()
            
            # Per-layer codebook usage
            for layer_idx in range(model.n_layers):
                layer_codes = all_codes_concat[:, layer_idx]
                unique_layer_codes = torch.unique(layer_codes)
                usage = len(unique_layer_codes) / model.n_embed
                eval_stats[f'layer_{layer_idx}_usage'] = usage
        
        return eval_stats
    
    def train(self) -> Dict[str, Any]:
        """Main training loop with EBODA-style evaluation"""
        self.logger.info("Starting tokenizer training...")
        self.logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Load data
        dataloader = self.load_data()
        
        # Load validation data if configured
        eval_dataloader = None
        if self.config.get('eval_split', False):
            self.logger.info("Loading validation data...")
            eval_dataset = ItemEmbeddingDataset(
                embedding_file=os.path.join(self.config['data_path'], self.config.get('embedding_file', 'item_emb_all.parquet')),
                max_items=self.config.get('max_eval_items', 5000)
            )
            eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config.get('num_workers', 4),
                pin_memory=True
            )
            self.logger.info(f"Loaded {len(eval_dataset)} validation items")
        
        # Get embedding dimension from first batch
        first_batch = next(iter(dataloader))
        embedding_dim = first_batch['embedding'].shape[1]
        self.logger.info(f"Embedding dimension: {embedding_dim}")
        
        # Create model
        model = self.create_model(embedding_dim)
        
        # Load checkpoint if provided
        if self.config.get('load_checkpoint'):
            checkpoint_path = self.config['load_checkpoint']
            self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
            model.load_model(checkpoint_path)
            self.logger.info("Checkpoint loaded successfully!")
        
        # Check if we should skip training (only generate semantic IDs)
        if self.config['epochs'] == 0:
            self.logger.info("Epochs=0, skipping training and generating semantic IDs only...")
            results = {'training_stats': [], 'mode': 'inference_only'}
        else:
            # Train based on mode
            if self.config['mode'].lower() == 'tiger':
                results = self.train_tiger_mode(model, dataloader, eval_dataloader)
            elif self.config['mode'].lower() == 'kmeans':
                results = self.train_kmeans_mode(model, dataloader)
            else:
                raise ValueError(f"Unknown mode: {self.config['mode']}")
            
            # Save final model
            self.save_model(model)
        
        # Generate and save codebooks/mappings
        self.generate_semantic_ids(model, dataloader)
        
        self.logger.info("Training completed successfully!")
        return results
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       epoch: int, stats: Dict[str, Any]):
        """Save training checkpoint with loss and duplicate rate in filename"""
        # Create filename with loss and duplicate rate
        loss = stats['total_loss']
        dup_rate = stats.get('duplicate_rate_pre', 0.0)  # Use pre-post-ID rate
        
        checkpoint_name = f'checkpoint_epoch_{epoch+1}_loss_{loss:.4f}_dup_{dup_rate:.4f}.pt'
        checkpoint_path = os.path.join(self.config['output_dir'], checkpoint_name)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'stats': stats,
            'config': self.config
        }, checkpoint_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.logger.info(f"  Stats - Loss: {loss:.4f}, Duplicate rate: {dup_rate:.4f}")
    
    def save_model(self, model: nn.Module):
        """Save final trained model"""
        model_path = os.path.join(self.config['output_dir'], 'final_model.pt')
        
        if hasattr(model, 'save_model'):
            model.save_model(model_path)
        else:
            torch.save({
                'state_dict': model.state_dict(),
                'config': self.config
            }, model_path)
            
        self.logger.info(f"Final model saved: {model_path}")
    
    def generate_semantic_ids(self, model: nn.Module, dataloader: DataLoader):
        """Generate semantic IDs with GLOBAL post-ID deduplication"""
        self.logger.info("Generating semantic IDs for all items...")
        
        # Keep model in train() mode to use same quantization as training
        # Use torch.no_grad() to prevent gradient computation
        model.train()
        
        if self.config['mode'].lower() == 'tiger':
            # Step 1: Generate 3-layer codes for all items
            self.logger.info("Step 1: Generating 3-layer codes for all items...")
            item_to_3layer = {}
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Generating 3-layer codes"):
                    item_ids = batch['item_id'].numpy()
                    embeddings = batch['embedding'].to(self.device)
                    
                    codes_3layer = model.encode_to_codes(embeddings, apply_post_id=False)
                    
                    for item_id, code in zip(item_ids, codes_3layer.cpu().numpy()):
                        item_to_3layer[int(item_id)] = tuple(code)
            
            # Step 2: Apply GLOBAL post-ID deduplication
            self.logger.info("Step 2: Applying GLOBAL post-ID deduplication...")
            tuple_to_items = defaultdict(list)
            for item_id, code_tuple in item_to_3layer.items():
                tuple_to_items[code_tuple].append(item_id)
            
            # Step 3: Assign unique 4th codes
            self.logger.info("Step 3: Assigning unique 4th layer codes...")
            item_to_codes = {}
            for code_tuple, item_ids in tuple_to_items.items():
                for idx, item_id in enumerate(sorted(item_ids)):
                    # Convert numpy types to Python native types for JSON serialization
                    code_4layer = [int(c) for c in code_tuple] + [int(idx)]
                    item_to_codes[int(item_id)] = code_4layer
            
            # Calculate duplicate rates
            total_items = len(item_to_codes)
            unique_3layer = len(tuple_to_items)
            dup_rate_pre = 1.0 - (unique_3layer / total_items)
            dup_rate_post = 0.0  # Should be 0 with global deduplication
            
            self.logger.info(f"Final duplicate rates:")
            self.logger.info(f"  Before post-ID: {dup_rate_pre:.4f} ({total_items - unique_3layer}/{total_items} duplicates)")
            self.logger.info(f"  After post-ID: {dup_rate_post:.4f} (0/{total_items} duplicates)")
            
            # Step 4: Calculate and log hierarchical duplicate rates
            self.logger.info("\n" + "="*60)
            self.logger.info("SEMANTIC ID HIERARCHICAL DUPLICATE ANALYSIS")
            self.logger.info("="*60)
            self._analyze_hierarchical_duplicates(item_to_codes, total_items)
            
        else:  # kmeans mode
            self.logger.info("Generating codes for KMeans mode...")
            item_to_codes = {}
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Generating codes"):
                    item_ids = batch['item_id'].numpy()
                    embeddings = batch['embedding'].to(self.device)
                    codes = model.encode(embeddings).cpu().numpy()
                    
                    for item_id, code in zip(item_ids, codes):
                        item_to_codes[int(item_id)] = code.tolist()
        
        # Save semantic ID mappings
        mappings_path = os.path.join(self.config['output_dir'], 'semantic_id_mappings.json')
        with open(mappings_path, 'w') as f:
            json.dump(item_to_codes, f, indent=2)
        
        self.logger.info(f"\nSemantic ID mappings saved: {mappings_path}")
        self.logger.info(f"Generated {len(item_to_codes)} unique semantic IDs")
    
    def _analyze_hierarchical_duplicates(self, item_to_codes: dict, total_items: int):
        """Analyze and log hierarchical duplicate rates"""
        # Get number of hierarchies (assuming all codes have same length)
        n_hierarchies = len(next(iter(item_to_codes.values())))
        
        for level in range(1, n_hierarchies + 1):
            # Group items by first 'level' codes
            prefix_groups = defaultdict(list)
            for item_id, codes in item_to_codes.items():
                prefix = tuple(codes[:level])
                prefix_groups[prefix].append(item_id)
            
            unique_prefixes = len(prefix_groups)
            duplicate_count = total_items - unique_prefixes
            duplicate_rate = duplicate_count / total_items if total_items > 0 else 0.0
            
            # Find maximum group size
            max_group_size = max(len(items) for items in prefix_groups.values())
            
            self.logger.info(f"\nLayer {level} (first {level} code{'s' if level > 1 else ''}):")
            self.logger.info(f"  Unique combinations: {unique_prefixes}/{total_items}")
            self.logger.info(f"  Duplicate items: {duplicate_count}/{total_items} ({duplicate_rate:.4f})")
            self.logger.info(f"  Max items sharing same prefix: {max_group_size}")
            
            # Show some examples of duplicates if they exist
            if duplicate_count > 0 and level < n_hierarchies:
                # Find groups with more than 1 item
                duplicate_groups = [(prefix, items) for prefix, items in prefix_groups.items() if len(items) > 1]
                if duplicate_groups:
                    # Sort by group size and show top 3
                    duplicate_groups.sort(key=lambda x: len(x[1]), reverse=True)
                    self.logger.info(f"  Top duplicate groups (showing up to 3):")
                    for i, (prefix, items) in enumerate(duplicate_groups[:3]):
                        self.logger.info(f"    Group {i+1}: Prefix {prefix} -> {len(items)} items")
        
        self.logger.info("\n" + "="*60)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Semantic ID Tokenizer")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed dataset directory (containing item_emb.parquet)')
    parser.add_argument('--embedding_file', type=str, default='item_emb_all.parquet',
                       help='Embedding file name (default: item_emb_all.parquet for all items, use item_emb.parquet for filtered)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for trained models and results')
    
    # Training arguments
    parser.add_argument('--mode', type=str, default='tiger', choices=['tiger', 'kmeans'],
                       help='Tokenization mode (default: tiger)')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Number of training epochs (default: 2000, optimized)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256, optimized for stability)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4, optimized for AdamW)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda, default: auto)')
    
    # Model arguments (TIGER paper defaults)
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of quantization layers (default: 3, TIGER paper)')
    parser.add_argument('--n_embed', type=int, default=256,
                       help='Number of embeddings per layer for TIGER mode (default: 256, TIGER paper)')
    parser.add_argument('--n_clusters', type=int, default=512,
                       help='Number of clusters per layer for K-means mode (default: 512)')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension for TIGER mode (default: 32, TIGER paper)')
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Use EMA update for codebook (default: True, more stable)')
    parser.add_argument('--no_ema', action='store_false', dest='use_ema',
                       help='Disable EMA update (use standard VQ)')
    parser.add_argument('--ema_decay', type=float, default=0.99,
                       help='EMA decay rate (default: 0.99)')
    parser.add_argument('--beta', type=float, default=0.25,
                       help='Commitment loss weight (default: 0.25)')
    
    # Early stopping
    parser.add_argument('--early_stop_patience', type=int, default=50,
                       help='Stop if loss does not improve for N epochs (0=disable, default: 50)')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                       help='Minimum change in loss to qualify as improvement (default: 1e-4)')
    
    # Scheduler arguments (new optimizations)
    parser.add_argument('--use_scheduler', action='store_true', default=False,
                       help='Enable learning rate scheduler (default: False)')
    parser.add_argument('--scheduler_type', type=str, default='warmup_cosine',
                       choices=['warmup_cosine', 'standard'],
                       help='Scheduler type: warmup_cosine or standard (default: warmup_cosine)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio of total steps (default: 0.1)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1,
                       help='Minimum LR ratio of initial LR (default: 0.1)')
    
    # Training optimizations
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping value (default: 1.0, 0=disable)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='AdamW weight decay (default: 0.01)')
    
    # Temperature scheduling (for Gumbel-Softmax)
    parser.add_argument('--init_temperature', type=float, default=1.0,
                       help='Initial temperature for Gumbel-Softmax (default: 1.0)')
    parser.add_argument('--min_temperature', type=float, default=0.2,
                       help='Minimum temperature (default: 0.2)')
    parser.add_argument('--temperature_anneal_rate', type=float, default=0.00003,
                       help='Temperature annealing rate (default: 0.00003)')
    
    # Other arguments
    parser.add_argument('--max_items', type=int, default=None,
                       help='Maximum number of items to use (for testing)')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    parser.add_argument('--load_checkpoint', type=str, default=None,
                       help='Path to checkpoint to load (if epochs=0, only generates semantic IDs)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare optimized config
    config = {
        'data_path': args.data_path,
        'embedding_file': args.embedding_file,
        'output_dir': args.output_dir,
        'mode': args.mode,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'device': args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'),
        'n_layers': args.n_layers,
        'n_embed': args.n_embed,
        'n_clusters': args.n_clusters,
        'latent_dim': args.latent_dim,
        'max_items': args.max_items,
        'save_every': args.save_every,
        'log_level': args.log_level,
        'load_checkpoint': args.load_checkpoint,
        'num_workers': 4,
        'grad_clip': args.grad_clip,
        'weight_decay': args.weight_decay,
        'use_scheduler': args.use_scheduler,
        'scheduler_type': args.scheduler_type,
        'warmup_ratio': args.warmup_ratio,
        'min_lr_ratio': args.min_lr_ratio,
        'beta': args.beta,
        'use_ema': args.use_ema,
        'ema_decay': args.ema_decay,
        'quantize_mode': 'gumbel_softmax',
        'init_temperature': args.init_temperature,
        'min_temperature': args.min_temperature,
        'temperature_anneal_rate': args.temperature_anneal_rate,
        'early_stop_patience': args.early_stop_patience,
        'early_stop_min_delta': args.early_stop_min_delta
    }
    
    # Initialize trainer and start training
    trainer = TokenizerTrainer(config)
    results = trainer.train()
    
    # Save results
    results_path = os.path.join(args.output_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Training completed! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()