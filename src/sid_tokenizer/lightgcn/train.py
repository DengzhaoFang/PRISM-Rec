"""
Training script for LightGCN - Stage 0 of Prism
"""

import argparse
import os
import torch
import pandas as pd
from pathlib import Path
import sys
import logging

from dataset import BeautyDataset
from model import LightGCN
from trainer import LightGCNTrainer
from utils import setup_logger, save_config, print_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LightGCN for Stage 0')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--use_val', action='store_true',
                       help='Include validation set in training')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of GCN layers')
    parser.add_argument('--dropout', action='store_true',
                       help='Use dropout during training')
    parser.add_argument('--keep_prob', type=float, default=0.6,
                       help='Keep probability for dropout')
    
    # Training arguments
    parser.add_argument('--n_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--reg_weight', type=float, default=1e-4,
                       help='L2 regularization weight')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Evaluation arguments
    parser.add_argument('--eval_every', type=int, default=1,
                       help='Evaluate on validation set every N epochs')
    parser.add_argument('--eval_batch_size', type=int, default=256,
                       help='Batch size for evaluation')
    parser.add_argument('--early_stop_metric', type=str, default='Recall@20',
                       choices=['Recall@5', 'Recall@10', 'Recall@20', 'NDCG@5', 'NDCG@10', 'NDCG@20'],
                       help='Metric to use for early stopping')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20],
                       help='K values for evaluation metrics')
    
    # Save arguments
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save model and embeddings')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--exp_name', type=str, default='lightgcn_stage0',
                       help='Experiment name')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--gpu_id', type=int, default=None,
                       help='GPU ID to use (sets CUDA_VISIBLE_DEVICES). If None, use default GPU.')
    
    args = parser.parse_args()
    return args


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Set GPU if specified (must be done before importing torch.cuda)
    if args.gpu_id is not None and args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    log_file = output_dir / 'train.log'
    logger = setup_logger(log_file=str(log_file))
    
    logger.info("="*80)
    logger.info("LightGCN Training - Stage 0: Cold-start Initialization")
    logger.info("="*80)
    
    # Print configuration
    config = vars(args)
    print_config(config)
    
    # Save configuration
    save_config(config, output_dir / 'config.json')
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    data_path = Path(args.data_dir)
    
    # Load training dataset
    dataset = BeautyDataset(args.data_dir, use_val=args.use_val)
    
    logger.info(f"Training dataset loaded:")
    logger.info(f"  - Users: {dataset.n_users}")
    logger.info(f"  - Items: {dataset.n_items}")
    logger.info(f"  - Interactions: {dataset.n_train}")
    
    # Load validation and test data for evaluation
    logger.info("Loading validation and test data...")
    valid_data = pd.read_parquet(data_path / 'valid.parquet')
    test_data = pd.read_parquet(data_path / 'test.parquet')
    
    logger.info(f"  - Validation samples: {len(valid_data)}")
    logger.info(f"  - Test samples: {len(test_data)}")
    
    # Create model
    logger.info("Creating LightGCN model")
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        keep_prob=args.keep_prob
    )
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = LightGCNTrainer(
        model=model,
        dataset=dataset,
        valid_data=valid_data,
        test_data=test_data,
        learning_rate=args.lr,
        reg_weight=args.reg_weight,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        device=args.device
    )
    
    # Train model
    logger.info("Starting training")
    history = trainer.train(
        n_epochs=args.n_epochs,
        save_dir=str(output_dir / 'checkpoints'),
        save_every=args.save_every,
        eval_every=args.eval_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_metric=args.early_stop_metric,
        k_list=args.k_values
    )
    
    # Extract and save item embeddings
    logger.info("Extracting item embeddings")
    embeddings_path = output_dir / 'item_embeddings_collab.npy'
    trainer.save_embeddings(str(embeddings_path))
    
    logger.info("="*80)
    logger.info("Training completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Item embeddings saved to: {embeddings_path}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

