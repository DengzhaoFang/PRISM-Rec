#!/usr/bin/env python3
"""
PRISM Training Script

Train Hierarchical ID VAE with IDE + MCD pipeline and optional SACO loss.
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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from PRISM import PRISM, create_prism_from_config
from multimodal_dataset import PRISMDataset, create_dataloaders
from prism_losses import PRISMTotalLoss

try:
    from schedulers import WarmupCosineScheduler, ExponentialSchedulerWithWarmup
    SCHEDULERS_AVAILABLE = True
except ImportError:
    SCHEDULERS_AVAILABLE = False


class PRISMTrainer:
    """Main trainer class for PRISM with IDE + MCD + SACO support."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.setup_logging()
        self.logger.info("Initializing PRISM trainer...")
        self.logger.info(f"Configuration: {json.dumps(config, indent=2)}")

        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_loss()

        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.perplexity_collapse_epochs = 0

        self.train_history = defaultdict(list)
        self.prev_epoch_metrics: Optional[Dict[str, float]] = None
        self.best_metrics = {'total_loss': float('inf')}

        self.logger.info("PRISM trainer initialized successfully")

    def setup_logging(self):
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        # Suppress root logger and sub-module loggers to prevent duplicate output
        logging.getLogger().handlers.clear()
        logging.getLogger().setLevel(logging.WARNING)
        for sub in ['SinkhornReassigner']:
            logging.getLogger(sub).handlers.clear()
            logging.getLogger(sub).setLevel(logging.WARNING)

        self.logger = logging.getLogger('PRISMTrainer')
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()
        self.logger.propagate = False

        log_file = self.output_dir / 'training.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)

    def setup_data(self):
        self.logger.info("Loading dataset...")
        data_dir = self.config['data_path']
        batch_size = self.config.get('batch_size', 256)
        num_workers = self.config.get('num_workers', 4)
        max_items = self.config.get('max_items', None)

        self.train_loader, self.dataset = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            max_items=max_items,
        )

        self.logger.info(f"Dataset loaded: {len(self.dataset)} items")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Number of batches: {len(self.train_loader)}")
        if self.dataset.has_cooc:
            self.logger.info(f"  Co-occurrence graph: ENABLED")
        else:
            self.logger.info(f"  Co-occurrence graph: DISABLED (SACO will be skipped)")

    def setup_model(self):
        self.logger.info("Initializing PRISM model...")
        self.model = create_prism_from_config(config=self.config)
        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model initialized")
        self.logger.info(f"  Total parameters: {total_params:,}")
        self.logger.info(f"  Trainable parameters: {trainable_params:,}")
        self.logger.info(f"  IDE: {'ENABLED' if self.config.get('use_ide', True) else 'DISABLED'}")
        self.logger.info(f"  MCD: {'ENABLED' if self.config.get('use_mcd', True) else 'DISABLED'}")

        codebook_sizes = self.model.get_codebook_sizes()
        self.logger.info(f"  Codebook sizes per layer: {codebook_sizes}")
        for i, size in enumerate(codebook_sizes):
            self.logger.info(f"    Layer {i+1}: {size} codes")

        self._initialize_codebooks_hierarchical()

    def setup_optimizer(self):
        self.logger.info("Initializing optimizer and scheduler...")
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.0)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )

        if self.config.get('use_scheduler', False) and SCHEDULERS_AVAILABLE:
            scheduler_type = self.config.get('scheduler_type', 'warmup_cosine')
            total_steps = self.config['epochs'] * len(self.train_loader)
            warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))

            if scheduler_type == 'warmup_cosine':
                self.scheduler = WarmupCosineScheduler(
                    optimizer=self.optimizer,
                    warmup_steps=warmup_steps,
                    total_steps=total_steps,
                    min_lr_ratio=0.01
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
            self.logger.info(f"  Scheduler: {scheduler_type}, warmup steps: {warmup_steps}, total: {total_steps}")
        else:
            self.scheduler = None
            self.logger.info("  No scheduler used")

        self.logger.info(f"  Optimizer: AdamW (lr={lr}, wd={weight_decay})")

    def setup_loss(self):
        self.logger.info("Initializing loss function...")

        use_saco = (
            self.config.get('use_saco', False) and self.dataset.has_cooc
        )
        use_cma = self.config.get('use_ide', True)

        commit_w = self.config.get('commit_weight', 0.0625)
        self.loss_fn = PRISMTotalLoss(
            commit_weight=commit_w,
            use_saco=use_saco,
            lambda_sac=self.config.get('lambda_sac', 0.1),
            saco_temperature=self.config.get('saco_temperature', 0.07),
            use_cma=use_cma,
            lambda_cma=self.config.get('lambda_cma', 0.1),
            cma_temperature=self.config.get('cma_temperature', 0.07),
        )

        self.loss_fn = self.loss_fn.to(self.device)
        self.logger.info("Loss function initialized")
        self.logger.info(f"  UPR: MSE(z_dec, z_clean.detach()) — 256D unified reconstruction")
        self.logger.info(f"  Commit weight={commit_w}")
        self.logger.info(f"  Loss: UPR + {commit_w}*commit + SACO + CMA")
        self.logger.info(f"  CMA: {'ENABLED' if use_cma else 'DISABLED'}")
        if use_cma:
            self.logger.info(f"    λ_cma={self.config.get('lambda_cma', 0.1)}")
            self.logger.info(f"    τ_cma={self.config.get('cma_temperature', 0.07)}")
        self.logger.info(f"  SACO: {'ENABLED' if use_saco else 'DISABLED'}")
        if use_saco:
            self.logger.info(f"    λ_sac={self.config.get('lambda_sac', 0.1)}")
            self.logger.info(f"    τ_saco={self.config.get('saco_temperature', 0.07)}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = defaultdict(float)

        consistency_stats = defaultdict(float)
        n_consistency_samples = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config['epochs']}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Anchor forward pass (full pipeline)
            content_emb = batch['content_emb'].to(self.device)
            collab_emb = batch['collab_emb'].to(self.device)

            outputs = self.model(
                content_emb=content_emb,
                collab_emb=collab_emb,
                temperature=self._get_temperature(epoch),
                return_codes=True,
            )

            z_dec = outputs['z_dec']
            z_clean = outputs['z_clean']
            vq_loss = outputs['codebook_loss']
            z_anchor = outputs['z']

            # CMA inputs (raw IDE projections, pre-MCD)
            h_t = outputs.get('h_t')
            h_c = outputs.get('h_c')

            # Positive encode (latent only, for SACO)
            z_pos = None
            if self.loss_fn.use_saco:
                pos_content = batch['pos_content_emb'].to(self.device)
                pos_collab = batch['pos_collab_emb'].to(self.device)
                pos_enc = self.model.encode(pos_content, pos_collab)
                z_pos = pos_enc['z']

            # Track MCD consistency (computed from raw IDE outputs)
            consistency = outputs.get('consistency')
            if consistency is not None:
                consistency_stats['mean'] += consistency.mean().item()
                consistency_stats['std'] += consistency.std().item()
                consistency_stats['min'] += consistency.min().item()
                consistency_stats['max'] += consistency.max().item()
                n_consistency_samples += 1

            total_loss, loss_dict = self.loss_fn(
                z_dec=z_dec,
                z_clean=z_clean,
                commitment_loss=vq_loss,
                h_t=h_t,
                h_c=h_c,
                z_anchor=z_anchor,
                z_pos=z_pos,
            )

            self.optimizer.zero_grad()
            total_loss.backward()

            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            for key, value in loss_dict.items():
                epoch_metrics[key] += value

            for i, perp in enumerate(outputs['perplexities']):
                epoch_metrics[f'perplexity_layer{i+1}'] += perp

            postfix_dict = {
                'loss': loss_dict['total_loss'],
                'upr': loss_dict['upr'],
                'lr': self.optimizer.param_groups[0]['lr'],
            }
            if 'cma' in loss_dict:
                postfix_dict['cma'] = loss_dict['cma']
            if 'saco' in loss_dict:
                postfix_dict['saco'] = loss_dict['saco']
            if n_consistency_samples > 0:
                postfix_dict['cons'] = consistency_stats['mean'] / n_consistency_samples
            progress_bar.set_postfix(postfix_dict)

            self.global_step += 1

        num_batches = len(self.train_loader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}

        if n_consistency_samples > 0:
            for key in consistency_stats:
                epoch_metrics[f'consistency_{key}'] = consistency_stats[key] / n_consistency_samples

        self.prev_epoch_metrics = epoch_metrics
        return epoch_metrics

    def _get_temperature(self, epoch: int) -> float:
        if self.config.get('quantize_mode', 'rotation') != 'gumbel_softmax':
            return 0.2
        init_temp = self.config.get('init_temp', 1.0)
        min_temp = self.config.get('min_temp', 0.1)
        anneal_rate = self.config.get('anneal_rate', 0.00003)
        return max(min_temp, init_temp * np.exp(-anneal_rate * epoch))

    def _update_early_stopping(self, metrics: Dict[str, float], epoch: int) -> Tuple[bool, bool]:
        patience = self.config.get('early_stop_patience', float('inf'))
        if not np.isfinite(patience):
            self._update_perplexity_guard(metrics)
            return False, False

        min_delta = self.config.get('early_stop_min_delta', 1e-4)
        total_loss = metrics.get('total_loss', float('inf'))
        improved = total_loss < (self.best_loss - min_delta)

        cooldown = self.config.get('early_stop_cooldown', 3)
        warmup_epochs = self.config.get('early_stop_warmup_epochs', 5)
        warmup_limit = warmup_epochs + cooldown

        if improved:
            self.best_loss = total_loss
            self.best_metrics['total_loss'] = total_loss
            self.patience_counter = 0
        elif epoch > warmup_limit:
            self.patience_counter += 1
        else:
            self.patience_counter = 0

        self._update_perplexity_guard(metrics)

        should_stop = self.patience_counter >= patience
        if self.perplexity_collapse_epochs >= self.config.get('perplexity_collapse_patience', 3):
            should_stop = False
            self.patience_counter = max(0, self.patience_counter - 1)

        return should_stop, improved

    def _update_perplexity_guard(self, metrics: Dict[str, float]) -> None:
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

        if ratio < self.config.get('perplexity_collapse_ratio', 0.35):
            self.perplexity_collapse_epochs += 1
        else:
            self.perplexity_collapse_epochs = 0

    def _initialize_codebooks_hierarchical(self) -> None:
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
                # Skip MCD during k-means: untrained denoiser poisons centroids
                mcd = getattr(self.model.encoder, 'mcd', None)
                if mcd is not None:
                    self.model.encoder.mcd = None
                enc_outputs = self.model.encode(content_batch, collab_batch)
                if mcd is not None:
                    self.model.encoder.mcd = mcd
                latents = enc_outputs['z']
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
                self.logger.warning(f"Layer {layer_idx+1}: insufficient samples for k-means "
                                    f"({residual_cpu.size(0)} < {n_clusters}).")
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
        return {}

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False,
                        save_regular: bool = True):
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

        if save_regular:
            checkpoint_path = self.output_dir / f'checkpoint_epoch{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")

        latest_path = self.output_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.logger.info(f"Checkpoint loaded (epoch {self.current_epoch})")

    def save_item_codebook_mappings(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Saving Item-Codebook Mappings")
        self.logger.info("=" * 80)

        best_model_path = self.output_dir / 'best_model.pt'
        if not best_model_path.exists():
            self.logger.warning(f"Best model not found at {best_model_path}, using current model")
        else:
            self.logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()
        codebooks = self.model.get_codebooks()

        self.logger.info("Processing all items...")
        item_mappings = {}

        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Processing items"):
                content_emb = batch['content_emb'].to(self.device)
                collab_emb = batch['collab_emb'].to(self.device)
                item_ids = batch['item_id'].cpu().numpy()

                outputs = self.model(
                    content_emb=content_emb,
                    collab_emb=collab_emb,
                    return_codes=True,
                )

                encoding_indices = outputs['encoding_indices']
                batch_size = content_emb.size(0)

                for i in range(batch_size):
                    item_id = str(item_ids[i])
                    item_indices = [indices[i].item() for indices in encoding_indices]

                    item_codebook_vectors = []
                    for layer_idx, (codebook, code_idx) in enumerate(zip(codebooks, item_indices)):
                        vector = codebook[code_idx].cpu().numpy().tolist()
                        item_codebook_vectors.append(vector)

                    item_mappings[item_id] = {
                        'item_id': item_id,
                        'codebook_indices': item_indices,
                        'codebook_vectors': item_codebook_vectors,
                    }

        output_file = self.output_dir / 'item_codebook_mappings.json'
        with open(output_file, 'w') as f:
            json.dump(item_mappings, f, indent=2)
        self.logger.info(f"Saved {len(item_mappings)} item mappings to: {output_file}")

        npz_file = self.output_dir / 'item_codebook_mappings.npz'
        n_items = len(item_mappings)
        n_layers = len(codebooks)

        item_ids_array = np.array([int(k) for k in item_mappings.keys()])
        indices_array = np.array([item_mappings[str(iid)]['codebook_indices'] for iid in item_ids_array])
        vectors_list = [item_mappings[str(iid)]['codebook_vectors'] for iid in item_ids_array]
        vectors_array = np.array(vectors_list)

        np.savez(
            npz_file,
            item_ids=item_ids_array,
            codebook_indices=indices_array,
            codebook_vectors=vectors_array,
        )
        self.logger.info(f"Saved numpy format to: {npz_file}")
        self.logger.info("=" * 80)

    def generate_semantic_ids_and_analyze(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Generating Semantic IDs and Analysis")
        self.logger.info("=" * 80)

        best_model_path = self.output_dir / 'best_model.pt'
        if not best_model_path.exists():
            self.logger.warning(f"Best model not found at {best_model_path}, using current model")
        else:
            self.logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.eval()

        self.logger.info("Generating semantic IDs for all items...")
        all_item_ids = []
        all_semantic_ids = []

        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Generating IDs"):
                content_emb = batch['content_emb'].to(self.device)
                collab_emb = batch['collab_emb'].to(self.device)
                item_ids = batch['item_id']

                semantic_ids = self.model.generate_semantic_ids(content_emb, collab_emb)

                all_item_ids.append(item_ids.cpu().numpy())
                all_semantic_ids.append(semantic_ids.cpu().numpy())

        all_item_ids = np.concatenate(all_item_ids, axis=0)
        all_semantic_ids = np.concatenate(all_semantic_ids, axis=0)
        n_items = len(all_item_ids)
        n_layers = all_semantic_ids.shape[1]

        self.logger.info(f"Generated {n_items} semantic IDs with {n_layers} layers")

        self.logger.info("\n" + "-" * 80)
        self.logger.info("Semantic ID Analysis")
        self.logger.info("-" * 80)

        id_tuples = [tuple(sid) for sid in all_semantic_ids]
        unique_ids = set(id_tuples)
        n_unique = len(unique_ids)
        uniqueness_rate = n_unique / n_items

        self.logger.info(f"\n1. Overall Uniqueness:")
        self.logger.info(f"   Total items: {n_items}")
        self.logger.info(f"   Unique IDs: {n_unique}")
        self.logger.info(f"   Uniqueness rate: {uniqueness_rate:.2%}")

        from collections import Counter
        id_counts = Counter(id_tuples)
        collisions = {k: v for k, v in id_counts.items() if v > 1}
        n_collision_groups = len(collisions)
        n_items_in_collisions = sum(collisions.values())

        self.logger.info(f"\n2. Collision Analysis:")
        self.logger.info(f"   Collision groups: {n_collision_groups}")
        self.logger.info(f"   Items in collisions: {n_items_in_collisions} ({n_items_in_collisions/n_items:.2%})")

        if n_collision_groups > 0:
            top_collisions = sorted(collisions.items(), key=lambda x: x[1], reverse=True)[:5]
            self.logger.info(f"   Top 5 collisions:")
            for sid, count in top_collisions:
                self.logger.info(f"     ID {sid}: {count} items")

        self.logger.info(f"\n3. Hierarchical Overlap Analysis:")
        for layer in range(1, n_layers + 1):
            prefixes = [tuple(sid[:layer]) for sid in all_semantic_ids]
            unique_prefixes = set(prefixes)
            n_unique_prefix = len(unique_prefixes)
            overlap_rate = 1.0 - (n_unique_prefix / n_items)
            avg_items_per_prefix = n_items / n_unique_prefix

            self.logger.info(f"\n   Layer {layer} Prefix:")
            self.logger.info(f"     Unique prefixes: {n_unique_prefix} / {n_items}")
            self.logger.info(f"     Overlap rate: {overlap_rate:.4f}")
            self.logger.info(f"     Avg items per prefix: {avg_items_per_prefix:.2f}")

            prefix_counts = Counter(prefixes)
            singleton_count = sum(1 for count in prefix_counts.values() if count == 1)
            self.logger.info(f"     Singleton prefixes: {singleton_count} ({singleton_count/n_unique_prefix:.2%})")

        self.logger.info(f"\n4. Codebook Usage per Layer:")
        codebook_sizes = self.model.get_codebook_sizes()
        for layer in range(n_layers):
            codes = all_semantic_ids[:, layer]
            unique_codes = len(set(codes))
            n_embed = codebook_sizes[layer]
            usage_rate = unique_codes / n_embed

            code_counts = Counter(codes)
            most_common = code_counts.most_common(3)
            unused_codes = n_embed - unique_codes

            self.logger.info(f"\n   Layer {layer + 1}:")
            self.logger.info(f"     Codebook size: {n_embed}")
            self.logger.info(f"     Used: {unique_codes} / {n_embed} ({usage_rate:.2%})")
            self.logger.info(f"     Unused: {unused_codes}")
            self.logger.info(f"     Most common codes: {[f'{code}({count})' for code, count in most_common]}")

        # Save results
        semantic_id_mappings = {}
        for i in range(n_items):
            item_id = str(all_item_ids[i])
            semantic_codes = all_semantic_ids[i].tolist()
            semantic_id_mappings[item_id] = semantic_codes

        output_file = self.output_dir / 'semantic_id_mappings.json'
        with open(output_file, 'w') as f:
            json.dump(semantic_id_mappings, f, indent=2)
        self.logger.info(f"\nSemantic ID mappings saved to: {output_file}")

        npy_file = self.output_dir / 'semantic_ids.npy'
        np.save(npy_file, all_semantic_ids)
        self.logger.info(f"Semantic IDs (numpy) saved to: {npy_file}")

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

        for layer in range(1, n_layers + 1):
            prefixes = [tuple(sid[:layer]) for sid in all_semantic_ids]
            unique_prefixes = len(set(prefixes))
            overlap_rate = 1.0 - (unique_prefixes / n_items)
            report['hierarchical_overlap'][f'layer_{layer}'] = {
                'unique_prefixes': int(unique_prefixes),
                'overlap_rate': float(overlap_rate),
                'avg_items_per_prefix': float(n_items / unique_prefixes)
            }

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
        self.logger.info(f"Analysis report saved to: {report_file}")
        self.logger.info("=" * 80)

        self.apply_sinkhorn_reassignment()

    def export_purified_embeddings(self):
        """Export MCD-purified h_t_hat and h_c_hat for Stage 2 DSI."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Exporting Purified Embeddings for Stage 2")
        self.logger.info("=" * 80)

        best_model_path = self.output_dir / 'best_model.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded best model from {best_model_path}")

        self.model.eval()
        n_items = len(self.dataset)
        ide_dim = self.config.get('ide_dim', 128)

        purified_content = np.zeros((n_items, ide_dim), dtype=np.float32)
        purified_collab = np.zeros((n_items, ide_dim), dtype=np.float32)
        item_ids_out = np.zeros(n_items, dtype=np.int64)

        idx = 0
        with torch.no_grad():
            for batch in tqdm(self.train_loader, desc="Exporting purified features"):
                content_emb = batch['content_emb'].to(self.device)
                collab_emb = batch['collab_emb'].to(self.device)
                item_ids = batch['item_id'].cpu().numpy()

                enc_outputs = self.model.encode(content_emb, collab_emb)
                h_t_hat = enc_outputs['h_t_hat'].cpu().numpy()
                h_c_hat = enc_outputs['h_c_hat'].cpu().numpy()

                b = len(item_ids)
                purified_content[idx:idx + b] = h_t_hat
                purified_collab[idx:idx + b] = h_c_hat
                item_ids_out[idx:idx + b] = item_ids
                idx += b

        np.save(self.output_dir / 'item_purified_content.npy', purified_content)
        np.save(self.output_dir / 'item_purified_collab.npy', purified_collab)
        np.save(self.output_dir / 'item_purified_ids.npy', item_ids_out)

        # Also export codebook z_q (sum of all layer codebook vectors, 32D)
        latent_dim = self.config.get('latent_dim', 32)
        codebook_zq = np.zeros((n_items, latent_dim), dtype=np.float32)
        with torch.no_grad():
            idx = 0
            for batch in tqdm(self.train_loader, desc="Exporting codebook z_q"):
                content_emb = batch['content_emb'].to(self.device)
                collab_emb = batch['collab_emb'].to(self.device)
                b = len(batch['item_id'])
                z = self.model.encode(content_emb, collab_emb)['z']
                z_q, _, _, _, _ = self.model.quantize(z)
                codebook_zq[idx:idx + b] = z_q.cpu().numpy()
                idx += b
        np.save(self.output_dir / 'item_codebook_zq.npy', codebook_zq)

        self.logger.info(f"Purified features exported: {n_items} items, {ide_dim}D each")
        self.logger.info(f"Codebook z_q exported: {n_items} items, {latent_dim}D each")
        self.logger.info("=" * 80)

    def apply_sinkhorn_reassignment(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Applying Sinkhorn Algorithm for Collision Elimination")
        self.logger.info("=" * 80)

        semantic_ids_file = self.output_dir / 'semantic_id_mappings.json'
        if not semantic_ids_file.exists():
            self.logger.warning(f"Semantic IDs file not found: {semantic_ids_file}")
            self.logger.warning("Skipping Sinkhorn reassignment")
            return

        original_backup = self.output_dir / 'semantic_id_mappings_original.json'
        if not original_backup.exists():
            import shutil
            shutil.copy2(semantic_ids_file, original_backup)
            self.logger.info(f"  Backed up original IDs to: {original_backup}")

        try:
            from sinkhorn_reassignment import SinkhornIDReassigner
        except ImportError:
            self.logger.error("Cannot import SinkhornIDReassigner, skipping reassignment")
            return

        try:
            reassigner = SinkhornIDReassigner(
                semantic_ids_path=str(semantic_ids_file),
                checkpoint_path=str(self.output_dir / 'best_model.pt'),
                data_dir=self.config['data_path'],
                device=self.config.get('device', 'cuda'),
                output_dir=str(self.output_dir)
            )

            codebook_sizes = self.model.get_codebook_sizes()
            new_semantic_ids = reassigner.run(codebook_sizes=codebook_sizes, max_iterations=10)

            from collections import Counter
            id_tuples = [tuple(sid) for sid in new_semantic_ids]
            n_unique = len(set(id_tuples))

            if n_unique == len(new_semantic_ids):
                self.logger.info("\n" + "=" * 80)
                self.logger.info("SUCCESS: 100% uniqueness achieved after Sinkhorn reassignment!")
                self.logger.info("=" * 80)
            else:
                self.logger.warning(f"\nWarning: {len(new_semantic_ids) - n_unique} collisions still remain")
                self.logger.warning(f"   Final uniqueness: {n_unique / len(id_tuples):.2%}")

        except Exception as e:
            self.logger.error(f"Error during Sinkhorn reassignment: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.logger.warning("Continuing without Sinkhorn reassignment")

    def train(self):
        self.logger.info("=" * 80)
        self.logger.info("Starting PRISM training")
        self.logger.info("=" * 80)

        start_epoch = self.current_epoch + 1
        end_epoch = self.config['epochs']

        for epoch in range(start_epoch, end_epoch + 1):
            self.current_epoch = epoch
            train_metrics = self.train_epoch(epoch)

            self.logger.info(f"\nEpoch {epoch} Summary:")
            self.logger.info(f"  Total Loss: {train_metrics['total_loss']:.4f}")
            self.logger.info(f"  UPR Loss: {train_metrics['upr']:.4f}")

            for i in range(self.config['n_layers']):
                if f'perplexity_layer{i+1}' in train_metrics:
                    self.logger.info(f"  Perplexity Layer {i+1}: {train_metrics[f'perplexity_layer{i+1}']:.2f}")

            if 'consistency_mean' in train_metrics:
                self.logger.info(
                    f"  MCD Consistency: "
                    f"mean={train_metrics['consistency_mean']:.3f}, "
                    f"std={train_metrics['consistency_std']:.3f}, "
                    f"range=[{train_metrics['consistency_min']:.3f}, {train_metrics['consistency_max']:.3f}]"
                )

            if 'cma' in train_metrics:
                self.logger.info(f"  CMA Loss: {train_metrics['cma']:.4f}")
            if 'saco' in train_metrics:
                self.logger.info(f"  SACO Loss: {train_metrics['saco']:.4f}")

            for key, value in train_metrics.items():
                self.train_history[key].append(value)

            should_stop, primary_improved = self._update_early_stopping(train_metrics, epoch)
            if primary_improved:
                self.logger.info(f"  New best loss: {self.best_loss:.4f}")
            else:
                self.logger.info(
                    f"  Patience: {self.patience_counter}/"
                    f"{self.config.get('early_stop_patience', float('inf'))}"
                )

            should_save_regular = (epoch % self.config.get('save_every', 50) == 0)
            if should_save_regular or primary_improved:
                self.save_checkpoint(epoch, train_metrics, primary_improved, save_regular=should_save_regular)

            if should_stop:
                self.logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                break

        self.save_checkpoint(epoch, train_metrics, is_best=False, save_regular=True)

        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)

        self.logger.info("=" * 80)
        self.logger.info("Training completed!")
        self.logger.info(f"Best loss: {self.best_loss:.4f}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 80)

        for step_name, step_fn in [
            ("save_item_codebook_mappings", self.save_item_codebook_mappings),
            ("generate_semantic_ids_and_analyze", self.generate_semantic_ids_and_analyze),
            ("export_purified_embeddings", self.export_purified_embeddings),
            ("_analyze_embedding_quality", self._analyze_embedding_quality),
        ]:
            try:
                step_fn()
            except Exception as e:
                self.logger.error(f"Post-training step '{step_name}' failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

    def _analyze_embedding_quality(self):
        """Deep embedding quality analysis — norms, variance, codebook stats."""
        self.model.eval()
        all_z, all_zc, all_ht, all_hc, all_zq = [], [], [], [], []
        with torch.no_grad():
            for batch in self.train_loader:
                ce = batch['content_emb'].to(self.device)
                cole = batch['collab_emb'].to(self.device)
                enc = self.model.encode(ce, cole)
                all_z.append(enc['z'].cpu()); all_zc.append(enc['z_clean'].cpu())
                all_ht.append(enc.get('h_t_hat', enc['h_t']).cpu())
                all_hc.append(enc.get('h_c_hat', enc['h_c']).cpu())
                all_zq.append(self.model.quantize(enc['z'])[0].cpu())

        z = torch.cat(all_z, dim=0); zc = torch.cat(all_zc, dim=0)
        ht = torch.cat(all_ht, dim=0); hc = torch.cat(all_hc, dim=0); zq = torch.cat(all_zq, dim=0)
        N = len(z); mask = ~torch.eye(N, dtype=torch.bool)
        sim_z = torch.nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)
        sim_zc = torch.nn.functional.cosine_similarity(zc.unsqueeze(1), zc.unsqueeze(0), dim=-1)
        commit = (z - zq).norm(dim=-1)

        self.logger.info("=" * 70)
        self.logger.info("DEEP EMBEDDING QUALITY ANALYSIS")
        self.logger.info("[1] Pre-encoder: z_clean={:.2f}+/-{:.2f} h_t={:.2f} h_c={:.2f} zc_inter_cos={:.4f} zc_neg%={:.1f}".format(
            zc.norm(dim=-1).mean(), zc.norm(dim=-1).std(), ht.norm(dim=-1).mean(), hc.norm(dim=-1).mean(),
            sim_zc[mask].mean(), (sim_zc[mask]<0).float().mean()*100))
        self.logger.info("[2] Latent z: norm={:.2f}+/-{:.2f} var={:.2f} inter_cos={:.4f} neg%={:.1f} ratio={:.2f}".format(
            z.norm(dim=-1).mean(), z.norm(dim=-1).std(), z.var(dim=0).sum(),
            sim_z[mask].mean(), (sim_z[mask]<0).float().mean()*100, (z.norm(dim=-1).mean()/zc.norm(dim=-1).mean())))
        self.logger.info("[3] Quantized: zq_norm={:.2f} commit|z-zq|={:.3f} commit_ratio={:.3f}".format(
            zq.norm(dim=-1).mean(), commit.mean(), commit.mean()/z.norm(dim=-1).mean()))
        cbs = self.model.get_codebooks()
        for li, cb in enumerate(cbs):
            cn = cb.norm(dim=-1); dead = (cn < 1e-4).sum().item()
            cb_sim = torch.nn.functional.cosine_similarity(cb.unsqueeze(1), cb.unsqueeze(0), dim=-1)
            cbm = ~torch.eye(len(cb), dtype=torch.bool)
            self.logger.info("[4] Codebook L{}: norm={:.2f}+/-{:.2f} dead={} inter_cos={:.4f}".format(
                li+1, cn.mean(), cn.std(), dead, cb_sim[cbm].mean()))
        # Encoder/Decoder weight norms — OLD structure: self.model.encoder.encoder (MLP), self.model.decoder (Sequential)
        enc_layers = list(self.model.encoder.encoder)
        dec_layers = list(self.model.decoder.shared_decoder) if hasattr(self.model.decoder, 'shared_decoder') else []
        if dec_layers:
            self.logger.info("[5] Weights: enc_first={:.2f} enc_last={:.2f} dec_first={:.2f} dec_last={:.2f}".format(
                enc_layers[0].weight.data.norm(), enc_layers[-1].weight.data.norm() if hasattr(enc_layers[-1], 'weight') else 0,
                dec_layers[0].weight.data.norm(), dec_layers[-1].weight.data.norm() if hasattr(dec_layers[-1], 'weight') else 0))
        self.logger.info("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description='Train PRISM with IDE + MCD + SACO')

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
                        help='Default codebook size per layer')
    parser.add_argument('--n_embed_per_layer', type=str, default=None,
                        help='Variable codebook sizes per layer (comma-separated)')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent/codebook dimension')
    parser.add_argument('--content_dim', type=int, default=768,
                        help='Content embedding dimension')
    parser.add_argument('--collab_dim', type=int, default=64,
                        help='Collaborative embedding dimension')
    # IDE + MCD arguments
    parser.add_argument('--ide', type=str, default='on', choices=['on', 'off'],
                        help='Enable/disable IDE (default: on). Ablation: --ide off')
    parser.add_argument('--mcd', type=str, default='on', choices=['on', 'off'],
                        help='Enable/disable MCD (default: on). Ablation: --mcd off')
    parser.add_argument('--ide_dim', type=int, default=128,
                        help='IDE projection dimension d (default: 128)')

    # SACO arguments
    parser.add_argument('--use_saco', action='store_true',
                        help='Enable Sequence-Aware Contrastive Objective')
    parser.add_argument('--lambda_sac', type=float, default=0.1,
                        help='Weight for SACO loss')
    parser.add_argument('--saco_temperature', type=float, default=0.07,
                        help='Temperature for SACO contrastive loss')

    # CMA arguments
    parser.add_argument('--lambda_cma', type=float, default=0.1,
                        help='Weight for Cross-Modal Alignment loss')
    parser.add_argument('--cma_temperature', type=float, default=0.07,
                        help='Temperature for CMA InfoNCE loss')

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
    parser.add_argument('--beta', type=float, default=0.25,
                        help='VQ beta (quantizer internal, mostly unused with EMA)')
    parser.add_argument('--commit_weight', type=float, default=0.0625,
                        help='Effective commitment loss weight (0.0625 = OLD default)')

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
    parser.add_argument('--early_stop_cooldown', type=int, default=3,
                        help='Extra epochs after warmup before early stopping can trigger')
    parser.add_argument('--early_stop_warmup_epochs', type=int, default=5,
                        help='Epochs to wait before early stopping can trigger')
    parser.add_argument('--perplexity_collapse_patience', type=int, default=3,
                        help='Number of consecutive low-perplexity epochs before intervention')
    parser.add_argument('--perplexity_collapse_ratio', type=float, default=0.35,
                        help='Minimum acceptable perplexity ratio before triggering collapse guard')

    # Hierarchical k-means initialization arguments
    parser.add_argument('--no_hierarchical_kmeans_init', action='store_true',
                        help='Disable hierarchical k-means codebook initialization')
    parser.add_argument('--kmeans_init_samples', type=int, default=8192,
                        help='Number of items to sample for k-means initialization')
    parser.add_argument('--kmeans_batch_size', type=int, default=1024,
                        help='Batch size when encoding samples for k-means')
    parser.add_argument('--kmeans_random_state', type=int, default=42,
                        help='Random seed for k-means clustering')

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
    args = parse_args()
    config = vars(args)

    config['use_ide'] = (args.ide == 'on')
    config['use_mcd'] = (args.mcd == 'on')

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

    trainer = PRISMTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()


if __name__ == '__main__':
    main()
