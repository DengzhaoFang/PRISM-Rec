"""
Main training script for ActionPiece recommender model.

Usage:
    python -m src.recommender.ActionPiece.actionpiece_train --config beauty --device cuda:0
    python -m src.recommender.ActionPiece.actionpiece_train --config beauty --device cuda:0 --n_ensemble 5
"""

import argparse
import logging
import random
import numpy as np
import torch
import sys
from pathlib import Path
from functools import partial

from torch.utils.data import DataLoader

from .actionpiece_config import get_actionpiece_config
from .actionpiece_dataset import (
    create_actionpiece_datasets, 
    collate_fn_actionpiece
)
from .actionpiece_model import create_actionpiece_model
from .actionpiece_trainer import ActionPieceTrainer


def setup_logging(output_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(console_formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging to {log_file}")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ActionPiece recommender model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python -m src.recommender.ActionPiece.actionpiece_train --config beauty --device cuda:0

  # With custom ensemble size
  python -m src.recommender.ActionPiece.actionpiece_train --config beauty --device cuda:0 --n_ensemble 5

  # Custom paths
  python -m src.recommender.ActionPiece.actionpiece_train --config beauty \\
      --tokenizer_path path/to/actionpiece.json \\
      --item2feat_path path/to/item2feat.json
        """
    )
    
    # Essential arguments
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        default=None,
        choices=['beauty', 'sports', 'toys', 'cds'],
        help='Dataset configuration. Not required when resuming from checkpoint.'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for training'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    
    # ActionPiece specific
    parser.add_argument(
        '--n_ensemble',
        type=int,
        default=5,
        help='Number of inference ensemble runs (default: 5, paper setting)'
    )
    
    parser.add_argument(
        '--train_shuffle',
        type=str,
        default='feature',
        choices=['feature', 'none', 'token'],
        help='Shuffle strategy for training (default: feature for SPR)'
    )
    
    # Path overrides
    parser.add_argument(
        '--sequence_data_path',
        type=str,
        default=None,
        help='Override sequence data path'
    )
    
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default=None,
        help='Override ActionPiece tokenizer path'
    )
    
    parser.add_argument(
        '--item2feat_path',
        type=str,
        default=None,
        help='Override item2feat path'
    )
    
    # Optional overrides
    parser.add_argument(
        '--model_type',
        type=str,
        default='t5-tiny-2',
        choices=['t5-tiny-2', 'actionpiece-paper', 't5-small'],
        help='Model type'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    
    parser.add_argument(
        '--output_keywords',
        type=str,
        default=None,
        help='Keywords to append to output directory name'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    # Training overrides
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--beam_size',
        type=int,
        default=30,
        help='Beam size for generation (default: 30)'
    )
    
    return parser.parse_args()


def _build_config_kwargs(args) -> dict:
    """Build configuration kwargs from command line arguments."""
    config_kwargs = {}
    
    if args.model_type is not None:
        config_kwargs['model_type'] = args.model_type
    
    if args.output_dir is not None:
        config_kwargs['output_dir'] = args.output_dir
    
    if args.output_keywords is not None:
        config_kwargs['output_keywords'] = args.output_keywords
    
    if args.sequence_data_path is not None:
        config_kwargs['sequence_data_path'] = args.sequence_data_path
    
    if args.tokenizer_path is not None:
        config_kwargs['tokenizer_path'] = args.tokenizer_path
    
    if args.item2feat_path is not None:
        config_kwargs['item2feat_path'] = args.item2feat_path
    
    # Training overrides
    if args.batch_size is not None:
        config_kwargs['batch_size'] = args.batch_size
    
    if args.learning_rate is not None:
        config_kwargs['learning_rate'] = args.learning_rate
    
    if args.num_epochs is not None:
        config_kwargs['num_epochs'] = args.num_epochs
    
    config_kwargs['beam_size'] = args.beam_size
    config_kwargs['n_inference_ensemble'] = args.n_ensemble
    config_kwargs['train_shuffle'] = args.train_shuffle
    config_kwargs['device'] = args.device
    config_kwargs['num_workers'] = args.num_workers
    
    return config_kwargs


def _log_config(logger, config: dict, model_type: str):
    """Log configuration information."""
    logger.info("=" * 80)
    logger.info("ACTIONPIECE RECOMMENDER TRAINING")
    logger.info("=" * 80)
    
    logger.info("\nConfiguration:")
    logger.info(f"  Dataset: {config['data'].dataset_name}")
    logger.info(f"  Sequence data: {config['data'].sequence_data_path}")
    logger.info(f"  Tokenizer: {config['data'].tokenizer_path}")
    logger.info(f"  Item2feat: {config['data'].item2feat_path}")
    logger.info(f"  Output directory: {config['output_dir']}")
    logger.info(f"  Max sequence length: {config['data'].max_seq_length}")
    logger.info(f"  Train shuffle: {config['data'].train_shuffle}")
    
    logger.info(f"\nModel configuration:")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Vocab size: {config['model'].vocab_size}")
    logger.info(f"  d_model: {config['model'].d_model}")
    logger.info(f"  d_ff: {config['model'].d_ff}")
    logger.info(f"  Num layers: {config['model'].num_layers}")
    logger.info(f"  Num heads: {config['model'].num_heads}")
    logger.info(f"  Inference ensemble: {config['model'].n_inference_ensemble}")
    
    logger.info(f"\nTraining configuration:")
    logger.info(f"  Batch size: {config['training'].batch_size}")
    logger.info(f"  Num epochs: {config['training'].num_epochs}")
    logger.info(f"  Learning rate: {config['training'].learning_rate}")
    logger.info(f"  Weight decay: {config['training'].weight_decay}")
    logger.info(f"  Beam size: {config['training'].beam_size}")
    logger.info(f"  Device: {config['training'].device}")


def main():
    """Main training function."""
    args = parse_args()
    
    # Check if resuming from checkpoint
    if args.resume:
        # Load checkpoint to extract config
        logger_temp = logging.getLogger(__name__)
        logger_temp.info(f"Loading checkpoint from {args.resume} to extract configuration...")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        
        if 'config' not in checkpoint:
            raise ValueError(f"Checkpoint {args.resume} does not contain config. Cannot resume training.")
        
        # Use config from checkpoint
        config = checkpoint['config']
        
        # Only override device and num_workers from command line
        config['training'].device = args.device
        config['training'].num_workers = args.num_workers
        
        logger_temp.info("Configuration loaded from checkpoint:")
        logger_temp.info(f"  Dataset: {config['data'].dataset_name}")
        logger_temp.info(f"  Model type: {config['model_type']}")
        logger_temp.info(f"  Epoch to resume from: {checkpoint['epoch'] + 1}")
        logger_temp.info(f"  Best metric so far: {checkpoint['best_metric']:.4f}")
        logger_temp.info(f"  Device (overridden): {config['training'].device}")
        logger_temp.info(f"  Num workers (overridden): {config['training'].num_workers}")
    else:
        # Validate that config is provided for new training
        if args.config is None:
            raise ValueError("--config is required when starting new training (not resuming from checkpoint)")
        
        # Build configuration
        config_kwargs = _build_config_kwargs(args)
        config = get_actionpiece_config(args.config, **config_kwargs)
    
    # Setup logging
    setup_logging(Path(config['output_dir']), args.log_level)
    logger = logging.getLogger(__name__)
    
    # Log configuration
    _log_config(logger, config, config['model_type'])
    
    # Set random seed
    set_seed(config['training'].seed)
    
    # Load datasets
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATASETS")
    logger.info("=" * 80)
    
    train_dataset, valid_dataset, test_dataset, mapper = create_actionpiece_datasets(
        sequence_data_dir=config['data'].sequence_data_path,
        tokenizer_path=config['data'].tokenizer_path,
        item2feat_path=config['data'].item2feat_path,
        max_len=config['data'].max_seq_length,
        train_shuffle=config['data'].train_shuffle,
        pad_token_id=config['model'].pad_token_id
    )
    
    # Update model config with actual vocab size
    config['model'].set_vocab_size(mapper.vocab_size)
    config['model'].n_categories = mapper.n_categories
    
    logger.info(f"Vocabulary size: {mapper.vocab_size}")
    logger.info(f"Number of categories: {mapper.n_categories}")
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Valid samples: {len(valid_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    collate_fn = partial(
        collate_fn_actionpiece,
        pad_token_id=config['model'].pad_token_id,
        n_categories=mapper.n_categories
    )
    
    # Test collate function with ensemble (following original implementation)
    from .actionpiece_dataset import collate_fn_actionpiece_test
    collate_fn_test = partial(
        collate_fn_actionpiece_test,
        mapper=mapper,
        n_ensemble=config['model'].n_inference_ensemble,
        pad_token_id=config['model'].pad_token_id,
        n_categories=mapper.n_categories
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training'].batch_size,
        shuffle=True,
        num_workers=config['training'].num_workers,
        collate_fn=collate_fn
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['training'].eval_batch_size,
        shuffle=False,
        num_workers=config['training'].num_workers,
        collate_fn=collate_fn
    )
    
    # Test loader uses smaller batch size due to ensemble expansion
    # batch_size * n_ensemble should fit in memory
    test_batch_size = config['training'].eval_batch_size // config['model'].n_inference_ensemble
    test_batch_size = max(1, test_batch_size)  # At least 1
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=config['training'].num_workers,
        collate_fn=collate_fn_test
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Valid batches: {len(valid_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("\n" + "=" * 80)
    logger.info("CREATING MODEL")
    logger.info("=" * 80)
    
    model = create_actionpiece_model(config['model'], mapper)
    
    # Create trainer
    trainer = ActionPieceTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config=config,
        actionpiece_mapper=mapper,
        device=config['training'].device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINISHED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
