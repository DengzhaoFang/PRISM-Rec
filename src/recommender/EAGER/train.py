"""
Main training script for the recommender model.

All hyperparameters are configured in config.py.
Only specify essential runtime parameters (config, device, num_workers).

Usage:
    python -m src.recommender.train --config beauty --device cuda:0 --num_workers 4
    python -m src.recommender.train --config beauty --device cuda:0 --model_type t5-small-raw
"""

import argparse
import logging
import random
import numpy as np
import torch
import sys
from pathlib import Path

from .config import get_config
from .dataset import create_datasets
from .dataloader import create_dataloaders
from .model import create_model
from .trainer import Trainer


def setup_logging(output_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration.
    
    Args:
        output_dir: Directory to save log files
        log_level: Logging level
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(file_formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(console_formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging to {log_file}")


def set_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")


def parse_args():
    """Parse command line arguments.
    
    All hyperparameters are configured in config.py.
    Only specify essential runtime parameters here.
    """
    parser = argparse.ArgumentParser(
        description="Train TIGER recommender model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python -m src.recommender.train --config beauty --device cuda:0

  # Use T5-small model
  python -m src.recommender.train --config beauty --device cuda:0 --model_type t5-small-raw

  # Custom output directory
  python -m src.recommender.train --config beauty --device cuda:0 --output_dir ./my_output
        """
    )
    
    # Essential arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        choices=['beauty', 'sports', 'toys', 'cds'],
        help='Dataset configuration (all hyperparameters are in config.py)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for training (e.g., cuda, cuda:0, cpu)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers (default: 4)'
    )
    
    # Optional overrides
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        choices=['t5-pico', 't5-nano', 't5-micro', 't5-tiny', 't5-tiny-2', 't5-small'],
        help='Override model type from config (optional)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Override output directory from config (optional)'
    )
    
    parser.add_argument(
        '--output_keywords',
        type=str,
        default=None,
        help='Keywords to append to output directory name (e.g., "baseline-experiment")'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from (optional)'
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    # Verbose logging
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Enable verbose sample printing during validation and testing'
    )
    
    # Learning rate scheduler
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default=None,
        choices=['none', 'warmup_cosine', 'reduce_on_plateau', 'exponential', 'step'],
        help='Learning rate scheduler type. Options: none (disable), warmup_cosine (recommended), reduce_on_plateau, exponential, step'
    )
    
    # EAGER specific arguments (optional, will use config defaults if not provided)
    parser.add_argument(
        '--behavior_mapping_path',
        type=str,
        default=None,
        nargs='?',
        const=None,
        help='Path to behavior semantic ID mapping (for EAGER). If not provided, uses config default.'
    )
    
    parser.add_argument(
        '--lambda_1',
        type=float,
        default=None,
        nargs='?',
        const=None,
        help='Weight for Global Contrastive Task (GCT). If not provided, uses config default.'
    )
    
    parser.add_argument(
        '--lambda_2',
        type=float,
        default=None,
        nargs='?',
        const=None,
        help='Weight for Semantic-Guided Transfer Task (STT). If not provided, uses config default.'
    )
    
    parser.add_argument(
        '--mask_ratio_recon',
        type=float,
        default=None,
        nargs='?',
        const=None,
        help='Masking ratio for STT reconstruction. If not provided, uses config default.'
    )
    
    parser.add_argument(
        '--mask_ratio_recog',
        type=float,
        default=None,
        nargs='?',
        const=None,
        help='Masking ratio for STT recognition. If not provided, uses config default.'
    )
    
    return parser.parse_args()


def _load_data(logger, config: dict):
    """Load datasets and create dataloaders.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, valid_loader, test_loader, semantic_mapper)
    """
    logger.info("\n" + "=" * 80)
    logger.info("LOADING DATASETS")
    logger.info("=" * 80)
    
    train_dataset, valid_dataset, test_dataset, semantic_mapper = create_datasets(
        sequence_data_dir=config['data'].sequence_data_path,
        semantic_mapping_path=config['data'].semantic_mapping_path,
        behavior_mapping_path=config['data'].behavior_mapping_path if hasattr(config['data'], 'behavior_mapping_path') else None,
        max_len=config['data'].max_seq_length,
        codebook_size=config['model'].codebook_size,
        num_layers=config['model'].num_code_layers,
        pad_token_id=config['model'].pad_token_id,
        model_config=config['model']
    )
    
    # Log vocabulary statistics
    layer_stats = semantic_mapper.get_layer_stats()
    logger.info(f"\nSemantic ID Layer Statistics:")
    if 'layer_max_values' in layer_stats:
        for i in range(layer_stats['num_layers']):
            logger.info(f"  Layer {i}: max_value={layer_stats['layer_max_values'][i]}")
        logger.info(f"  Actual vocab size: {layer_stats['actual_vocab_size']}")
        logger.info(f"  Theoretical vocab size: {layer_stats['theoretical_vocab_size']}")
        logger.info(f"  Embedding parameter savings: {layer_stats['savings']} tokens")
    elif 'behavior_stats' in layer_stats:
        logger.info("  Behavior Stream:")
        b_stats = layer_stats['behavior_stats']
        for i in range(b_stats['num_layers']):
            logger.info(f"    Layer {i}: max_value={b_stats['layer_max_values'][i]}")
        
        logger.info("  Semantic Stream:")
        s_stats = layer_stats['semantic_stats']
        for i in range(s_stats['num_layers']):
            logger.info(f"    Layer {i}: max_value={s_stats['layer_max_values'][i]}")
            
        logger.info(f"  Combined vocab size: {layer_stats['combined_vocab_size']}")
    
    # Create dataloaders
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset,
        valid_dataset,
        test_dataset,
        batch_size=config['training'].batch_size,
        eval_batch_size=config['training'].eval_batch_size,
        num_workers=config['training'].num_workers,
        pad_token_id=config['model'].pad_token_id
    )
    
    return train_loader, valid_loader, test_loader, semantic_mapper


def _create_model(logger, config: dict):
    """Create EAGER model.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        
    Returns:
        EAGER model instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING MODEL")
    logger.info("=" * 80)
    
    # Get embedding paths from config (for GCT - Global Contrastive Task)
    behavior_emb_path = None
    semantic_emb_path = None
    
    # Check if this is EAGER (dual-stream) mode
    if hasattr(config['data'], 'behavior_mapping_path') and config['data'].behavior_mapping_path is not None:
        # EAGER mode - use embeddings from config
        logger.info("EAGER mode detected - loading original embeddings for GCT")
        
        # Get paths from config (already validated in DataConfig.__post_init__)
        behavior_emb_path = config['data'].behavior_emb_path
        semantic_emb_path = config['data'].semantic_emb_path
        
        logger.info(f"  Behavior embeddings: {behavior_emb_path}")
        logger.info(f"  Semantic embeddings: {semantic_emb_path}")
    
    return create_model(config['model'], behavior_emb_path=behavior_emb_path, semantic_emb_path=semantic_emb_path)


def _log_config(logger, config: dict, model_type: str):
    """Log configuration information.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        model_type: Model type string
    """
    logger.info("=" * 80)
    logger.info("TIGER RECOMMENDER TRAINING")
    logger.info("=" * 80)
    
    logger.info("\nConfiguration:")
    logger.info(f"  Dataset: {config['data'].dataset_name}")
    logger.info(f"  Sequence data: {config['data'].sequence_data_path}")
    logger.info(f"  Semantic mapping: {config['data'].semantic_mapping_path}")
    logger.info(f"  Output directory: {config['output_dir']}")
    logger.info(f"  Checkpoint directory: {config['checkpoint_dir']}")
    logger.info(f"  Max sequence length: {config['data'].max_seq_length}")
    logger.info(f"  Seed: {config['training'].seed}")
    logger.info(f"  Device: {config['training'].device}")
    logger.info(f"  Num workers: {config['training'].num_workers}")
    
    logger.info(f"\nModel configuration:")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Vocab size: {config['model'].vocab_size}")
    logger.info(f"  d_model: {config['model'].d_model}")
    logger.info(f"  d_ff: {config['model'].d_ff}")
    logger.info(f"  Num layers: {config['model'].num_layers}")
    logger.info(f"  Num decoder layers: {config['model'].num_decoder_layers}")
    logger.info(f"  Num heads: {config['model'].num_heads}")
    
    logger.info(f"\nTraining configuration:")
    logger.info(f"  Batch size: {config['training'].batch_size}")
    logger.info(f"  Num epochs: {config['training'].num_epochs}")
    logger.info(f"  Learning rate: {config['training'].learning_rate}")
    logger.info(f"  LR scheduler: {config['training'].lr_scheduler}")
    logger.info(f"  Beam size: {config['training'].beam_size}")
    logger.info(f"  Eval every N epochs: {config['training'].eval_every_n_epochs}")
    logger.info(f"  Device: {config['training'].device}")


def _build_config_kwargs(args) -> dict:
    """Build configuration kwargs from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary of configuration overrides
    """
    config_kwargs = {}
    
    # Add overrides only if specified
    if args.model_type is not None:
        config_kwargs['model_type'] = args.model_type
    
    if args.output_dir is not None:
        config_kwargs['output_dir'] = args.output_dir
    
    if args.output_keywords is not None:
        config_kwargs['output_keywords'] = args.output_keywords
    
    # Always override these runtime parameters
    config_kwargs['device'] = args.device
    config_kwargs['num_workers'] = args.num_workers
    
    # Verbose logging
    if args.verbose:
        config_kwargs['verbose'] = True
    
    # Learning rate scheduler
    if args.lr_scheduler is not None:
        config_kwargs['lr_scheduler'] = args.lr_scheduler
        
    # EAGER overrides
    if args.behavior_mapping_path is not None:
        config_kwargs['behavior_mapping_path'] = args.behavior_mapping_path
        
    if args.lambda_1 is not None:
        config_kwargs['lambda_1'] = args.lambda_1
        
    if args.lambda_2 is not None:
        config_kwargs['lambda_2'] = args.lambda_2
        
    if args.mask_ratio_recon is not None:
        config_kwargs['mask_ratio_recon'] = args.mask_ratio_recon
        
    if args.mask_ratio_recog is not None:
        config_kwargs['mask_ratio_recog'] = args.mask_ratio_recog
    
    return config_kwargs


def main():
    """Main training function."""
    args = parse_args()
    
    # Build configuration with optional overrides
    config_kwargs = _build_config_kwargs(args)
    config = get_config(args.config, **config_kwargs)
    
    # Setup logging
    setup_logging(Path(config['output_dir']), args.log_level)
    logger = logging.getLogger(__name__)
    
    # Log configuration
    _log_config(logger, config, config['model_type'])
    
    # Set random seed
    set_seed(config['training'].seed)
    
    # Load data
    train_loader, valid_loader, test_loader, semantic_mapper = _load_data(logger, config)
    
    # Create model
    model = _create_model(logger, config)
    
    # Create and run trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config=config,
        device=config['training'].device,
        semantic_mapper=semantic_mapper
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING FINISHED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

