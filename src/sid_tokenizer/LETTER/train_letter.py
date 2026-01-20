#!/usr/bin/env python3
"""
LETTER Tokenizer Training Script

Faithful reproduction of the original LETTER paper implementation.
"""

import os
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

# Handle both direct execution and module import
try:
    from .rqvae_letter import LETTER_RQVAE
except ImportError:
    from rqvae_letter import LETTER_RQVAE


class ItemEmbeddingDataset(Dataset):
    """Dataset for loading item embeddings."""
    
    def __init__(
        self, 
        embedding_file: str, 
        cf_embedding_file: str = None,
        max_items: int = None
    ):
        self.embeddings_df = pd.read_parquet(embedding_file)
        
        if max_items is not None:
            self.embeddings_df = self.embeddings_df.head(max_items)
        
        # Get embedding column
        embedding_col = 'attribute_embedding' if 'attribute_embedding' in self.embeddings_df.columns else 'embedding'
        self.embeddings = torch.stack([
            torch.tensor(emb, dtype=torch.float32) 
            for emb in self.embeddings_df[embedding_col]
        ])
        
        self.item_ids = self.embeddings_df['ItemID'].values
        self.dim = self.embeddings.shape[1]
        
        # Load CF embeddings if provided
        self.cf_embeddings = None
        if cf_embedding_file is not None and os.path.exists(cf_embedding_file):
            print(f"Loading CF embeddings from {cf_embedding_file}...")
            if cf_embedding_file.endswith('.npy'):
                cf_emb_all = np.load(cf_embedding_file)
            elif cf_embedding_file.endswith('.pt'):
                cf_data = torch.load(cf_embedding_file, weights_only=False)
                if isinstance(cf_data, torch.Tensor):
                    cf_emb_all = cf_data.squeeze().numpy()
                else:
                    cf_emb_all = cf_data
            else:
                cf_emb_all = np.load(cf_embedding_file)
            
            self.cf_embeddings = torch.stack([
                torch.tensor(cf_emb_all[item_id], dtype=torch.float32)
                for item_id in self.item_ids
            ])
            print(f"  ✓ Loaded CF embeddings: {self.cf_embeddings.shape}")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        item = {
            'item_id': self.item_ids[idx],
            'embedding': self.embeddings[idx],
            'idx': idx
        }
        if self.cf_embeddings is not None:
            item['cf_emb'] = self.cf_embeddings[idx]
        return item


class LETTERTrainer:
    """Trainer for LETTER tokenizer."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.args.output_dir, f'{timestamp}_training.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode='w', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging to: {log_file}")
    
    def load_data(self) -> DataLoader:
        """Load dataset."""
        embedding_file = os.path.join(self.args.data_path, self.args.embedding_file)
        cf_file = os.path.join(self.args.data_path, self.args.cf_emb) if self.args.cf_emb else None
        
        dataset = ItemEmbeddingDataset(
            embedding_file=embedding_file,
            cf_embedding_file=cf_file,
            max_items=self.args.max_items
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        self.logger.info(f"Loaded {len(dataset)} items, dim={dataset.dim}")
        return dataloader, dataset
    
    def create_model(self, in_dim: int) -> LETTER_RQVAE:
        """Create LETTER model."""
        model = LETTER_RQVAE(
            in_dim=in_dim,
            num_emb_list=self.args.num_emb_list,
            e_dim=self.args.e_dim,
            layers=self.args.layers,
            dropout_prob=self.args.dropout_prob,
            bn=self.args.bn,
            loss_type=self.args.loss_type,
            quant_loss_weight=self.args.quant_loss_weight,
            kmeans_init=self.args.kmeans_init,
            kmeans_iters=self.args.kmeans_iters,
            sk_epsilons=self.args.sk_epsilons,
            sk_iters=self.args.sk_iters,
            alpha=self.args.alpha,
            beta=self.args.beta,
            mu=self.args.mu,
            n_clusters=self.args.n_clusters,
        )
        
        self.logger.info(f"Created LETTER model:")
        self.logger.info(f"  Input dim: {in_dim}")
        self.logger.info(f"  Latent dim: {self.args.e_dim}")
        self.logger.info(f"  Codebook sizes: {self.args.num_emb_list}")
        self.logger.info(f"  Encoder layers: {self.args.layers}")
        self.logger.info(f"  Alpha (CF loss): {self.args.alpha}")
        self.logger.info(f"  Beta (Diversity loss): {self.args.beta}")
        self.logger.info(f"  Mu (Commitment loss): {self.args.mu}")
        self.logger.info(f"  N clusters: {self.args.n_clusters}")
        
        return model.to(self.device)
    
    def initialize_codebooks(self, model, dataset):
        """Initialize codebooks using all data."""
        self.logger.info("Initializing codebooks with all data...")
        
        # Get all embeddings
        all_embeddings = dataset.embeddings.to(self.device)
        
        # Encode all data
        with torch.no_grad():
            all_latent = model.encode(all_embeddings)
        
        # Initialize each quantizer layer with residuals
        residual = all_latent
        for idx, quantizer in enumerate(model.rq.quantizers):
            self.logger.info(f"  Layer {idx}: Initializing with {len(residual)} samples...")
            quantizer.init_codebook(residual)
            quantizer.update_cluster_labels()
            
            # Compute residual for next layer
            with torch.no_grad():
                d = (torch.sum(residual ** 2, dim=1, keepdim=True) + 
                     torch.sum(quantizer.embedding.weight ** 2, dim=1, keepdim=True).t() - 
                     2 * torch.matmul(residual, quantizer.embedding.weight.t()))
                indices = torch.argmin(d, dim=-1)
                x_q = quantizer.embedding(indices)
                residual = residual - x_q
        
        self.logger.info("✓ All codebooks initialized")
    
    def train(self):
        """Main training loop."""
        dataloader, dataset = self.load_data()
        model = self.create_model(dataset.dim)
        
        # Initialize codebooks with all data BEFORE training
        self.initialize_codebooks(model, dataset)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        
        self.logger.info(f"Starting training for {self.args.epochs} epochs...")
        self.logger.info(f"Config: {vars(self.args)}")
        
        best_loss = float('inf')
        best_dup_rate = 1.0
        patience_counter = 0
        
        for epoch in range(self.args.epochs):
            model.train()
            epoch_stats = defaultdict(float)
            num_batches = 0
            
            progress = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
            
            for batch in progress:
                embeddings = batch['embedding'].to(self.device)
                cf_emb = batch.get('cf_emb')
                if cf_emb is not None:
                    cf_emb = cf_emb.to(self.device)
                
                optimizer.zero_grad()
                # Use Sinkhorn if any sk_epsilon > 0
                use_sk = any(eps > 0 for eps in self.args.sk_epsilons)
                outputs = model(embeddings, cf_emb=cf_emb, use_sk=use_sk)
                
                loss = outputs['total_loss']
                loss.backward()
                
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_clip)
                
                optimizer.step()
                
                # Track stats
                epoch_stats['total_loss'] += outputs['total_loss'].item()
                epoch_stats['recon_loss'] += outputs['recon_loss'].item()
                epoch_stats['quant_loss'] += outputs['quant_loss'].item()
                epoch_stats['diversity_loss'] += outputs['diversity_loss'].item()
                epoch_stats['cf_loss'] += outputs['cf_loss'].item()
                epoch_stats['duplicate_rate'] += outputs['duplicate_rate']
                num_batches += 1
                
                # Track codebook usage per layer
                codes = outputs['codes']
                for layer_idx in range(codes.shape[1]):
                    layer_codes = codes[:, layer_idx]
                    unique_codes = len(torch.unique(layer_codes))
                    key = f'layer_{layer_idx}_usage'
                    if key not in epoch_stats:
                        epoch_stats[key] = 0
                    epoch_stats[key] += unique_codes / model.n_embed
                
                progress.set_postfix({
                    'Loss': f"{outputs['total_loss'].item():.4f}",
                    'Recon': f"{outputs['recon_loss'].item():.4f}",
                    'Div': f"{outputs['diversity_loss'].item():.4f}",
                    'Dup': f"{outputs['duplicate_rate']:.3f}"
                })
            
            # Average stats
            for key in epoch_stats:
                epoch_stats[key] /= num_batches
            
            # Update clusters periodically
            if (epoch + 1) % self.args.cluster_every == 0:
                self.logger.info(f"  Updating cluster assignments...")
                model.update_cluster_labels()
            
            # Log
            self.logger.info(
                f"Epoch {epoch+1}: Loss={epoch_stats['total_loss']:.4f}, "
                f"Recon={epoch_stats['recon_loss']:.4f}, "
                f"Quant={epoch_stats['quant_loss']:.4f}, "
                f"Div={epoch_stats['diversity_loss']:.4f}, "
                f"CF={epoch_stats['cf_loss']:.4f}, "
                f"Dup={epoch_stats['duplicate_rate']:.4f}"
            )
            
            # Log per-layer codebook usage
            usage_str = ", ".join([
                f"L{i}={epoch_stats.get(f'layer_{i}_usage', 0):.3f}"
                for i in range(model.n_layers)
            ])
            self.logger.info(f"  Codebook usage: {usage_str}")
            
            # Early stopping
            if epoch_stats['total_loss'] < best_loss - self.args.early_stop_min_delta:
                best_loss = epoch_stats['total_loss']
                best_dup_rate = epoch_stats['duplicate_rate']
                patience_counter = 0
                self.logger.info(f"  ✓ New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= self.args.early_stop_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self.save_checkpoint(model, epoch, epoch_stats)
        
        # Save final model
        self.save_checkpoint(model, epoch, epoch_stats, final=True)
        
        # Generate semantic IDs
        self.generate_semantic_ids(model, dataloader, dataset)
        
        self.logger.info(f"Training completed! Best loss: {best_loss:.4f}, Best dup rate: {best_dup_rate:.4f}")
        
        return {'best_loss': best_loss, 'best_dup_rate': best_dup_rate}
    
    def save_checkpoint(self, model, epoch, stats, final=False):
        """Save model checkpoint."""
        filename = 'final_model.pt' if final else f'checkpoint_epoch_{epoch+1}.pt'
        path = os.path.join(self.args.output_dir, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'stats': dict(stats),
            'args': vars(self.args)
        }, path)
        
        self.logger.info(f"Saved checkpoint: {path}")
    
    def generate_semantic_ids(self, model, dataloader, dataset):
        """Generate semantic IDs for all items."""
        self.logger.info("Generating semantic IDs...")
        
        model.eval()
        all_codes = []
        all_item_ids = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating codes"):
                embeddings = batch['embedding'].to(self.device)
                item_ids = batch['item_id'].numpy()
                
                indices = model.get_indices(embeddings, use_sk=False)
                all_codes.append(indices.cpu())
                all_item_ids.extend(item_ids)
        
        all_codes = torch.cat(all_codes, dim=0)
        
        # Apply post-ID deduplication
        unique_codes = model.apply_post_id_deduplication(all_codes)
        
        # Create mapping
        item_to_codes = {}
        for item_id, codes in zip(all_item_ids, unique_codes.numpy()):
            item_to_codes[int(item_id)] = codes.tolist()
        
        # Analyze duplicates
        self._analyze_duplicates(all_codes, unique_codes)
        
        # Save
        mappings_path = os.path.join(self.args.output_dir, 'semantic_id_mappings.json')
        with open(mappings_path, 'w') as f:
            json.dump(item_to_codes, f, indent=2)
        
        self.logger.info(f"Saved semantic IDs to: {mappings_path}")
    
    def _analyze_duplicates(self, codes_3layer, codes_4layer):
        """Analyze duplicate rates at each layer."""
        total = len(codes_3layer)
        n_layers = codes_3layer.shape[1]
        
        self.logger.info("=" * 60)
        self.logger.info("SEMANTIC ID HIERARCHICAL ANALYSIS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total items: {total}")
        
        # Analyze each layer prefix
        for layer in range(1, n_layers + 1):
            prefix_codes = codes_3layer[:, :layer]
            prefix_set = set(tuple(c.tolist()) for c in prefix_codes)
            unique_count = len(prefix_set)
            dup_rate = 1.0 - unique_count / total
            
            # Find max collision group
            from collections import Counter
            prefix_counter = Counter(tuple(c.tolist()) for c in prefix_codes)
            max_collision = max(prefix_counter.values())
            
            self.logger.info(f"Layer {layer} (first {layer} codes):")
            self.logger.info(f"  Unique: {unique_count}/{total}")
            self.logger.info(f"  Duplicate rate: {dup_rate*100:.2f}%")
            self.logger.info(f"  Max collision group: {max_collision} items")
        
        # 4-layer (after post-ID)
        codes_4_set = set(tuple(c.tolist()) for c in codes_4layer)
        unique_4 = len(codes_4_set)
        self.logger.info(f"Layer 4 (after post-ID):")
        self.logger.info(f"  Unique: {unique_4}/{total}")
        self.logger.info(f"  Duplicate rate: {(1-unique_4/total)*100:.2f}%")
        
        self.logger.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="LETTER Tokenizer Training")
    
    # Data
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--embedding_file', type=str, default='item_emb.parquet')
    parser.add_argument('--cf_emb', type=str, default=None, help='CF embedding file')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_items', type=int, default=None)
    
    # Training
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Model
    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256, 256, 256])
    parser.add_argument('--e_dim', type=int, default=32)
    parser.add_argument('--layers', type=int, nargs='+', default=[512, 256, 128, 64])
    parser.add_argument('--dropout_prob', type=float, default=0.0)
    parser.add_argument('--bn', action='store_true', default=False)
    parser.add_argument('--loss_type', type=str, default='mse')
    
    # Quantization
    parser.add_argument('--quant_loss_weight', type=float, default=1.0)
    parser.add_argument('--kmeans_init', action='store_true', default=True)
    parser.add_argument('--kmeans_iters', type=int, default=100)
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0])
    parser.add_argument('--sk_iters', type=int, default=100)
    
    # LETTER losses
    parser.add_argument('--alpha', type=float, default=0.1, help='CF loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='Diversity loss weight')
    parser.add_argument('--mu', type=float, default=0.25, help='Commitment loss weight')
    parser.add_argument('--n_clusters', type=int, default=10)
    parser.add_argument('--cluster_every', type=int, default=10)
    
    # Early stopping
    parser.add_argument('--early_stop_patience', type=int, default=100)
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4)
    parser.add_argument('--save_every', type=int, default=50)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = LETTERTrainer(args)
    trainer.train()
