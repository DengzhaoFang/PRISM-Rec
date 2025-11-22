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
        choices=['beauty', 'sports', 'toys'],
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
    
    # ============================================================
    # NEW FEATURES: Multi-source Information Fusion
    # ============================================================
    
    # Feature 1: Codebook Vector Warm-start
    parser.add_argument(
        '--use_codebook_warmstart',
        action='store_true',
        help='Initialize ID embeddings with codebook vectors from HID-VAE'
    )
    parser.add_argument(
        '--codebook_warmstart_freeze',
        action='store_true',
        help='Freeze warmstarted ID embeddings (no fine-tuning)'
    )
    
    # Feature 2: Codebook Vector Prediction
    parser.add_argument(
        '--use_codebook_prediction',
        action='store_true',
        help='Add auxiliary task to predict target codebook vectors'
    )
    parser.add_argument(
        '--codebook_prediction_weight',
        type=float,
        default=0.1,
        help='Loss weight for codebook prediction (default: 0.1)'
    )
    
    # Feature 3: Tag ID Prediction
    parser.add_argument(
        '--use_tag_prediction',
        action='store_true',
        help='Add auxiliary task to predict hierarchical tag IDs'
    )
    parser.add_argument(
        '--tag_prediction_weight',
        type=float,
        default=0.1,
        help='Loss weight for tag prediction (default: 0.1)'
    )
    parser.add_argument(
        '--predict_tags_first',
        action='store_true',
        help='Predict tags before semantic IDs in output sequence'
    )
    
    # Feature 4: Multi-source Embedding Fusion
    parser.add_argument(
        '--use_multimodal_fusion',
        action='store_true',
        help='Fuse ID, content, and collaborative embeddings'
    )
    parser.add_argument(
        '--fusion_gate_type',
        type=str,
        default='learned',
        choices=['learned', 'fixed', 'attention'],
        help='Type of fusion gating mechanism (default: learned)'
    )
    parser.add_argument(
        '--content_emb_weight',
        type=float,
        default=0.5,
        help='Fixed weight for content embeddings (used if fusion_gate_type=fixed)'
    )
    parser.add_argument(
        '--collab_emb_weight',
        type=float,
        default=0.3,
        help='Fixed weight for collaborative embeddings (used if fusion_gate_type=fixed)'
    )
    parser.add_argument(
        '--id_emb_weight',
        type=float,
        default=0.2,
        help='Fixed weight for ID embeddings (used if fusion_gate_type=fixed)'
    )
    parser.add_argument(
        '--freeze_content_emb',
        action='store_true',
        default=True,
        help='Freeze content embeddings (default: True)'
    )
    parser.add_argument(
        '--freeze_collab_emb',
        action='store_true',
        default=True,
        help='Freeze collaborative embeddings (default: True)'
    )
    
    # Collaborative embedding path
    parser.add_argument(
        '--collab_embedding_path',
        type=str,
        default=None,
        help='Path to collaborative embeddings NPZ file (optional override)'
    )
    
    # Layer-specific fusion
    parser.add_argument(
        '--use_layer_specific_fusion',
        action='store_true',
        help='Use layer-specific projections for fusion (recommended for better performance)'
    )
    
    # ============================================================
    # NEW FEATURES: Structural Improvements
    # ============================================================
    
    # Feature 5: Dynamic Batching
    parser.add_argument(
        '--use_dynamic_batching',
        action='store_true',
        help='Use dynamic batching to reduce padding waste (pads to max length in batch)'
    )
    
    # Feature 6: Item/Layer Position Embeddings
    parser.add_argument(
        '--use_item_layer_emb',
        action='store_true',
        help='Add item/layer position embeddings to help model recognize item boundaries'
    )
    parser.add_argument(
        '--use_temporal_decay',
        action='store_true',
        default=True,
        help='Add temporal decay embeddings for recency information (default: True)'
    )
    
    # Feature 7: Hierarchical Attention
    parser.add_argument(
        '--use_hierarchical_attn',
        action='store_true',
        help='Use hierarchical attention (item-level and layer-level)'
    )
    parser.add_argument(
        '--use_item_attention',
        action='store_true',
        default=True,
        help='Use item-level attention (inter-item and intra-item) (default: True)'
    )
    parser.add_argument(
        '--use_layer_attention',
        action='store_true',
        default=True,
        help='Use layer-level attention (default: True)'
    )
    
    # Feature 8: Trie-Constrained Decoding
    parser.add_argument(
        '--use_trie_constraints',
        action='store_true',
        help='Enable Trie-constrained decoding (ensures all generated paths lead to real items)'
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
    
    # Get codebook_sizes if available (for variable-length codebooks)
    codebook_sizes = config['model'].__dict__.get('codebook_sizes', None)
    
    # Check if multimodal fusion is enabled
    use_multimodal = config['training'].use_multimodal_fusion
    collab_embedding_path = config['data'].collab_embedding_path
    
    train_dataset, valid_dataset, test_dataset, semantic_mapper = create_datasets(
        sequence_data_dir=config['data'].sequence_data_path,
        semantic_mapping_path=config['data'].semantic_mapping_path,
        max_len=config['data'].max_seq_length,
        codebook_size=config['model'].codebook_size,
        num_layers=config['model'].num_code_layers,
        pad_token_id=config['model'].pad_token_id,
        model_config=config['model'],
        codebook_sizes=codebook_sizes,
        collab_embedding_path=collab_embedding_path,
        use_multimodal=use_multimodal
    )
    
    # Log vocabulary statistics
    layer_stats = semantic_mapper.get_layer_stats()
    logger.info(f"\nSemantic ID Layer Statistics:")
    logger.info(f"  Codebook sizes: {layer_stats['codebook_sizes']}")
    for i in range(layer_stats['num_layers']):
        logger.info(f"  Layer {i}: codebook_size={layer_stats['codebook_sizes'][i]}, max_value={layer_stats['layer_max_values'][i]}")
    logger.info(f"  Actual vocab size: {layer_stats['actual_vocab_size']}")
    logger.info(f"  Theoretical vocab size: {layer_stats['theoretical_vocab_size']}")
    logger.info(f"  Embedding parameter savings: {layer_stats['savings']} tokens")
    
    # Create dataloaders
    use_dynamic_batching = config['training'].use_dynamic_batching
    train_loader, valid_loader, test_loader = create_dataloaders(
        train_dataset,
        valid_dataset,
        test_dataset,
        batch_size=config['training'].batch_size,
        eval_batch_size=config['training'].eval_batch_size,
        num_workers=config['training'].num_workers,
        pad_token_id=config['model'].pad_token_id,
        use_dynamic_batching=use_dynamic_batching
    )
    
    return train_loader, valid_loader, test_loader, semantic_mapper


def _create_model(logger, config: dict):
    """Create TIGER model.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        
    Returns:
        TIGER model instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING MODEL")
    logger.info("=" * 80)
    
    # Create model with training config for enhanced features
    model = create_model(config['model'], config['training'])
    
    # Feature 1: Codebook warmstart
    if config['training'].use_codebook_warmstart:
        logger.info("\nInitializing codebook warmstart...")
        # TODO: Load codebook vectors and semantic mapper
        # model.init_codebook_warmstart(codebook_vectors, semantic_mapper, freeze=...)
        logger.info("Codebook warmstart will be initialized in trainer")
    
    return model


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
    
    # NEW FEATURES: Multi-source Information Fusion
    if args.use_codebook_warmstart:
        config_kwargs['use_codebook_warmstart'] = True
        config_kwargs['codebook_warmstart_freeze'] = args.codebook_warmstart_freeze
    
    if args.use_codebook_prediction:
        config_kwargs['use_codebook_prediction'] = True
        config_kwargs['codebook_prediction_weight'] = args.codebook_prediction_weight
    
    if args.use_tag_prediction:
        config_kwargs['use_tag_prediction'] = True
        config_kwargs['tag_prediction_weight'] = args.tag_prediction_weight
        config_kwargs['predict_tags_first'] = args.predict_tags_first
    
    if args.use_multimodal_fusion:
        config_kwargs['use_multimodal_fusion'] = True
        config_kwargs['fusion_gate_type'] = args.fusion_gate_type
        config_kwargs['content_emb_weight'] = args.content_emb_weight
        config_kwargs['collab_emb_weight'] = args.collab_emb_weight
        config_kwargs['id_emb_weight'] = args.id_emb_weight
        config_kwargs['freeze_content_emb'] = args.freeze_content_emb
        config_kwargs['freeze_collab_emb'] = args.freeze_collab_emb
        
        # Layer-specific fusion
        if args.use_layer_specific_fusion:
            config_kwargs['use_layer_specific_fusion'] = True
    
    if args.collab_embedding_path is not None:
        config_kwargs['collab_embedding_path'] = args.collab_embedding_path
    
    # NEW FEATURES: Structural Improvements
    if args.use_dynamic_batching:
        config_kwargs['use_dynamic_batching'] = True
    
    if args.use_item_layer_emb:
        config_kwargs['use_item_layer_emb'] = True
        config_kwargs['use_temporal_decay'] = args.use_temporal_decay
    
    if args.use_hierarchical_attn:
        config_kwargs['use_hierarchical_attn'] = True
        config_kwargs['use_item_attention'] = args.use_item_attention
        config_kwargs['use_layer_attention'] = args.use_layer_attention
    
    # Feature 8: Trie-Constrained Decoding
    if args.use_trie_constraints:
        config_kwargs['use_trie_constraints'] = True
    
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

