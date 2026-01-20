"""
Trainer for LightGCN model
"""

import torch
import torch.optim as optim
from pathlib import Path
from typing import Optional, Dict, List
import logging
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import LightGCN
from dataset import BeautyDataset, BPRSampler
from evaluation import evaluate_model, print_metrics

logger = logging.getLogger(__name__)


class LightGCNTrainer:
    """
    Trainer for LightGCN model with BPR loss.
    """
    
    def __init__(
        self,
        model: LightGCN,
        dataset: BeautyDataset,
        valid_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        learning_rate: float = 0.001,
        reg_weight: float = 1e-4,
        batch_size: int = 2048,
        eval_batch_size: int = 256,
        device: str = 'cuda'
    ):
        """
        Args:
            model: LightGCN model
            dataset: BeautyDataset instance (training data)
            valid_data: Validation data (pandas DataFrame)
            test_data: Test data (pandas DataFrame)
            learning_rate: Learning rate for optimizer
            reg_weight: L2 regularization weight
            batch_size: Batch size for training
            eval_batch_size: Batch size for evaluation
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.dataset = dataset
        self.valid_data = valid_data
        self.test_data = test_data
        self.reg_weight = reg_weight
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Move graph to device
        graph = dataset.get_sparse_graph().to(self.device)
        self.model.set_graph(graph)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create BPR sampler
        self.sampler = BPRSampler(dataset, batch_size=batch_size, shuffle=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            metrics: Dictionary containing loss metrics
        """
        self.model.train()
        
        total_bpr_loss = 0.0
        total_reg_loss = 0.0
        total_loss = 0.0
        n_batches = 0
        
        # Create progress bar
        pbar = tqdm(self.sampler, desc="Training", leave=False)
        
        for batch_users, batch_pos_items, batch_neg_items in pbar:
            # Move to device
            batch_users = batch_users.to(self.device)
            batch_pos_items = batch_pos_items.to(self.device)
            batch_neg_items = batch_neg_items.to(self.device)
            
            # Forward pass
            bpr_loss, reg_loss = self.model.bpr_loss(
                batch_users, batch_pos_items, batch_neg_items, self.reg_weight
            )
            
            # Total loss
            loss = bpr_loss + self.reg_weight * reg_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses
            total_bpr_loss += bpr_loss.item()
            total_reg_loss += reg_loss.item()
            total_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bpr': f'{bpr_loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}'
            })
        
        # Average losses
        avg_bpr_loss = total_bpr_loss / n_batches
        avg_reg_loss = total_reg_loss / n_batches
        avg_loss = total_loss / n_batches
        
        return {
            'loss': avg_loss,
            'bpr_loss': avg_bpr_loss,
            'reg_loss': avg_reg_loss
        }
    
    def evaluate(self, k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate model on validation set
        
        Args:
            k_list: List of K values for evaluation
        
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        if self.valid_data is None:
            logger.warning("No validation data provided, skipping evaluation")
            return {}
        
        metrics = evaluate_model(
            self.model,
            self.dataset,
            self.valid_data,
            k_list=k_list,
            batch_size=self.eval_batch_size,
            device=str(self.device)
        )
        
        return metrics
    
    def train(
        self,
        n_epochs: int = 100,
        save_dir: Optional[str] = None,
        save_every: int = 10,
        eval_every: int = 1,
        early_stop_patience: int = 10,
        early_stop_metric: str = 'Recall@20',
        k_list: List[int] = [5, 10, 20]
    ) -> Dict[str, list]:
        """
        Train the model for multiple epochs with validation.
        
        Args:
            n_epochs: Number of training epochs
            save_dir: Directory to save checkpoints (None = don't save)
            save_every: Save checkpoint every N epochs
            eval_every: Evaluate on validation set every N epochs
            early_stop_patience: Stop if no improvement for N epochs
            early_stop_metric: Metric to use for early stopping (e.g., 'Recall@20')
            k_list: List of K values for evaluation
        
        Returns:
            history: Dictionary containing training history
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'loss': [],
            'bpr_loss': [],
            'reg_loss': []
        }
        
        # Add validation metrics to history
        if self.valid_data is not None:
            for k in k_list:
                history[f'val_Recall@{k}'] = []
                history[f'val_NDCG@{k}'] = []
        
        best_metric = 0.0 if 'Recall' in early_stop_metric or 'NDCG' in early_stop_metric else float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {n_epochs} epochs")
        if self.valid_data is not None:
            logger.info(f"Validation: enabled (every {eval_every} epochs)")
            logger.info(f"Early stopping: {early_stop_metric} with patience {early_stop_patience}")
        
        for epoch in range(1, n_epochs + 1):
            start_time = time.time()
            
            # Train one epoch
            metrics = self.train_epoch()
            
            # Update history
            for key, value in metrics.items():
                history[key].append(value)
            
            epoch_time = time.time() - start_time
            
            # Log progress
            log_msg = (
                f"Epoch {epoch}/{n_epochs} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"BPR: {metrics['bpr_loss']:.4f} | "
                f"Reg: {metrics['reg_loss']:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Evaluate on validation set
            val_metrics = {}
            if self.valid_data is not None and epoch % eval_every == 0:
                logger.info("Running validation...")
                val_metrics = self.evaluate(k_list=k_list)
                
                # Update history
                for key, value in val_metrics.items():
                    history[f'val_{key}'].append(value)
                
                # Print validation metrics
                print_metrics(val_metrics, prefix="Validation")
            
            logger.info(log_msg)
            
            # Save checkpoint
            if save_dir and (epoch % save_every == 0 or epoch == n_epochs):
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pt'
                combined_metrics = {**metrics, **val_metrics}
                self.save_checkpoint(checkpoint_path, epoch, combined_metrics)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Early stopping check (based on validation metric if available)
            if self.valid_data is not None and epoch % eval_every == 0 and early_stop_metric in val_metrics:
                current_metric = val_metrics[early_stop_metric]
                
                # For Recall/NDCG, higher is better
                if current_metric > best_metric:
                    best_metric = current_metric
                    patience_counter = 0
                    
                    # Save best model
                    if save_dir:
                        best_path = save_dir / 'best_model.pt'
                        combined_metrics = {**metrics, **val_metrics}
                        self.save_checkpoint(best_path, epoch, combined_metrics)
                        logger.info(f"New best {early_stop_metric}: {best_metric:.4f}")
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stop_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        logger.info(f"Best {early_stop_metric}: {best_metric:.4f}")
                        break
            elif self.valid_data is None:
                # Fallback to training loss
                if metrics['loss'] < best_metric if best_metric != 0.0 else True:
                    best_metric = metrics['loss']
                    patience_counter = 0
                    
                    if save_dir:
                        best_path = save_dir / 'best_model.pt'
                        self.save_checkpoint(best_path, epoch, metrics)
                else:
                    patience_counter += 1
                    
                    if patience_counter >= early_stop_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
        
        logger.info("Training completed!")
        
        # Final evaluation on test set
        if self.test_data is not None:
            logger.info("="*80)
            logger.info("Final evaluation on test set...")
            test_metrics = evaluate_model(
                self.model,
                self.dataset,
                self.test_data,
                k_list=k_list,
                batch_size=self.eval_batch_size,
                device=str(self.device)
            )
            print_metrics(test_metrics, prefix="Test")
            
            # Add to history
            for key, value in test_metrics.items():
                history[f'test_{key}'] = value
            
            logger.info("="*80)
        
        return history
    
    def save_checkpoint(self, path: Path, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': {
                'n_users': self.model.n_users,
                'n_items': self.model.n_items,
                'embedding_dim': self.model.embedding_dim,
                'n_layers': self.model.n_layers,
                'dropout': self.model.dropout,
                'keep_prob': self.model.keep_prob
            }
        }, path)
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    
    def extract_item_embeddings(self) -> np.ndarray:
        """
        Extract final item embeddings for use in Prism.
        
        Returns:
            item_embeddings: numpy array [n_items, embedding_dim]
        """
        self.model.eval()
        with torch.no_grad():
            item_embeddings = self.model.get_item_embeddings()
            item_embeddings = item_embeddings.cpu().numpy()
        
        logger.info(f"Extracted item embeddings: shape {item_embeddings.shape}")
        return item_embeddings
    
    def save_embeddings(self, save_path: str):
        """
        Extract and save item embeddings to file.
        
        Args:
            save_path: Path to save embeddings (.npy file)
        """
        embeddings = self.extract_item_embeddings()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.save(save_path, embeddings)
        logger.info(f"Saved item embeddings to {save_path}")
        
        return embeddings

