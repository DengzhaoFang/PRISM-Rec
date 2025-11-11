#!/usr/bin/env python3
"""
HID-VAE Training Script

Train Hierarchical ID VAE with multi-modal inputs and tag-guided learning.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import HID-VAE components
from HID_VAE import HIDVAE, create_hidvae_from_config
from multimodal_dataset import HIDVAEDataset, create_dataloaders
from hid_losses import HIDVAETotalLoss
from hierarchical_classifiers import create_hierarchical_classifiers

# Import schedulers
try:
    from schedulers import WarmupCosineScheduler, ExponentialSchedulerWithWarmup
    SCHEDULERS_AVAILABLE = True
except ImportError:
    SCHEDULERS_AVAILABLE = False


class HIDVAETrainer:
    """Main trainer class for HID-VAE"""
    
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
        self.logger.info("Initializing HID-VAE trainer...")
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
        
        # Metrics tracking
        self.train_history = defaultdict(list)
        
        self.logger.info("✓ HID-VAE trainer initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        
        # Create logger
        self.logger = logging.getLogger('HIDVAETrainer')
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
        
        # Create dataloader
        self.train_loader, self.dataset = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            max_items=max_items
        )
        
        self.logger.info(f"✓ Dataset loaded: {len(self.dataset)} items")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Number of batches: {len(self.train_loader)}")
        self.logger.info(f"  Tag statistics: {self.dataset.tag_stats}")
        
        # Get tag embeddings for anchor loss (fixed throughout training)
        n_layers = self.config.get('n_layers', 3)
        self.tag_embeddings_per_layer = self.dataset.get_tag_embeddings_per_level(n_layers)
        
        # Move tag embeddings to device
        self.tag_embeddings_per_layer = [
            emb.to(self.device) for emb in self.tag_embeddings_per_layer
        ]
        
        self.logger.info(f"✓ Tag embeddings loaded:")
        for i, emb in enumerate(self.tag_embeddings_per_layer):
            self.logger.info(f"  Layer {i+1} (L{i+2}): {emb.shape[0]} tags")
    
    def setup_model(self):
        """Initialize HID-VAE model"""
        self.logger.info("Initializing HID-VAE model...")
        
        # Get number of classes per layer (add 1 for PAD token)
        num_classes_per_layer = [
            self.dataset.tag_stats[f'n_L{i+2}'] + 1  # +1 for PAD
            for i in range(self.config.get('n_layers', 3))
        ]
        
        self.logger.info(f"Number of classes per layer: {num_classes_per_layer}")
        
        # Create model
        self.model = create_hidvae_from_config(
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
        
        self.loss_fn = HIDVAETotalLoss(
            # Reconstruction loss params
            lambda_content=self.config.get('lambda_content', 1.0),
            lambda_collab=self.config.get('lambda_collab', 1.0),
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
            # General params
            n_layers=n_layers,
            ignore_index=0  # PAD token
        )
        
        self.loss_fn = self.loss_fn.to(self.device)
        
        self.logger.info("✓ Loss function initialized")
        self.logger.info(f"  Lambda content: {self.config.get('lambda_content', 1.0)}")
        self.logger.info(f"  Lambda collab: {self.config.get('lambda_collab', 1.0)}")
        self.logger.info(f"  Delta (classification): {self.config.get('delta_weight', 1.0)}")
        self.logger.info(f"  Beta (commitment): {self.config.get('beta', 0.25)}")
        self.logger.info(f"  Gamma (balance): {gamma_weight}")
        self.logger.info(f"  Beta anchor (tag alignment): {beta_anchor_weight}")
    
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
            quantized_codes = outputs['quantized_codes']
            encoding_indices = outputs['encoding_indices']
            vq_loss = outputs['codebook_loss']  # This is codebook + commitment loss
            predictions = outputs.get('predictions', None)
            
            # Get codebooks for anchor loss
            codebooks = self.model.get_codebooks()
            
            # Prepare tag targets (split by layer)
            targets_per_layer = [tag_ids[:, i] for i in range(self.config['n_layers'])]
            masks_per_layer = [tag_mask[:, i] for i in range(self.config['n_layers'])]
            
            # Prepare codebook sizes
            n_embed_per_layer = [self.config['n_embed']] * self.config['n_layers']
            
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
                commitment_loss=vq_loss
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
            progress_bar.set_postfix({
                'loss': loss_dict['total_loss'],
                'recon': loss_dict['recon_total'],
                'class': loss_dict.get('class_total', 0),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            self.global_step += 1
        
        # Average metrics
        num_batches = len(self.train_loader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
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
        n_embed = self.config['n_embed']
        
        for layer in range(n_layers):
            codes = all_semantic_ids[:, layer]
            unique_codes = len(set(codes))
            usage_rate = unique_codes / n_embed
            
            # Most/least used
            code_counts = Counter(codes)
            most_common = code_counts.most_common(3)
            unused_codes = n_embed - unique_codes
            
            self.logger.info(f"\n   Layer {layer + 1}:")
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
        for layer in range(n_layers):
            codes = all_semantic_ids[:, layer]
            unique_codes = len(set(codes))
            usage_rate = unique_codes / n_embed
            
            report['codebook_usage'][f'layer_{layer+1}'] = {
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
    
    def train(self):
        """Main training loop"""
        self.logger.info("=" * 80)
        self.logger.info("Starting HID-VAE training")
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
            self.logger.info(f"  Recon Loss: {train_metrics['recon_total']:.4f} "
                           f"(content: {train_metrics['recon_content']:.4f}, "
                           f"collab: {train_metrics['recon_collab']:.4f})")
            
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
            
            # Track history
            for key, value in train_metrics.items():
                self.train_history[key].append(value)
            
            # Check for improvement
            current_loss = train_metrics['total_loss']
            is_best = current_loss < self.best_loss
            
            if is_best:
                self.best_loss = current_loss
                self.patience_counter = 0
                self.logger.info(f"  ✓ New best loss: {self.best_loss:.4f}")
            else:
                self.patience_counter += 1
                self.logger.info(f"  Patience: {self.patience_counter}/"
                               f"{self.config.get('early_stop_patience', float('inf'))}")
            
            # Save checkpoint
            should_save_regular = (epoch % self.config.get('save_every', 50) == 0)
            if should_save_regular or is_best:
                self.save_checkpoint(epoch, train_metrics, is_best, save_regular=should_save_regular)
            
            # Early stopping
            if self.patience_counter >= self.config.get('early_stop_patience', float('inf')):
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
        
        # Auto-generate semantic IDs and analysis
        self.generate_semantic_ids_and_analyze()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train HID-VAE')
    
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
                       help='Codebook size per layer')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent/codebook dimension')
    parser.add_argument('--content_dim', type=int, default=768,
                       help='Content embedding dimension')
    parser.add_argument('--collab_dim', type=int, default=64,
                       help='Collaborative embedding dimension')
    
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
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Initialize trainer
    trainer = HIDVAETrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()

