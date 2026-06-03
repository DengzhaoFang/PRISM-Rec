"""
Main training script for the recommender model.

All hyperparameters are configured in config.py.
Only specify essential runtime parameters (config, device, num_workers).

Usage:
    python -m src.recommender.train --config beauty --device cuda:0 --num_workers 4
    python -m src.recommender.train --config beauty --device cuda:0 --model_type t5-small-raw
"""

import argparse
import json
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


def load_fast_dev_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def apply_fast_dev_overrides(config: dict, fast_dev: dict):
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("APPLYING FAST DEV OVERRIDES")
    logger.info("=" * 80)

    training_overrides = fast_dev.get('training', {})
    model_overrides = fast_dev.get('model', {})
    data_overrides = fast_dev.get('data', {})

    for key, value in model_overrides.items():
        if hasattr(config['model'], key):
            setattr(config['model'], key, value)
            logger.info(f"  model.{key} = {value}")

    for key, value in training_overrides.items():
        if hasattr(config['training'], key):
            setattr(config['training'], key, value)
            logger.info(f"  training.{key} = {value}")

    fast_dev_state = {
        'enabled': True,
        'train_sample_limit': data_overrides.get('train_sample_limit'),
        'valid_sample_limit': data_overrides.get('valid_sample_limit'),
        'test_sample_limit': data_overrides.get('test_sample_limit'),
    }
    config['fast_dev'] = fast_dev_state
    logger.info(
        "  data caps: "
        f"train={fast_dev_state['train_sample_limit']}, "
        f"valid={fast_dev_state['valid_sample_limit']}, "
        f"test={fast_dev_state['test_sample_limit']}"
    )


def setup_logging(output_dir: Path, log_level: str = "INFO", clean: bool = False):
    """Setup logging configuration.

    Args:
        output_dir: Directory to save log files
        log_level: Logging level
        clean: If True, skip console handler and use minimal format (for batch runs)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    # Create formatters
    if clean:
        file_formatter = logging.Formatter('%(message)s')
    else:
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

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)

    if not clean:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(console_formatter)
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
        required=False,
        default=None,
        choices=['beauty', 'sports', 'toys', 'cds'],
        help='Dataset configuration (all hyperparameters are in config.py). Not required when resuming from checkpoint.'
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
        '--fast_dev_config',
        type=str,
        default=None,
        help='Path to an external fast-dev JSON config with short-run overrides'
    )

    # Stage 1 path overrides (for hyperparameter sweep)
    parser.add_argument('--semantic_mapping_path', type=str, default=None,
                        help='Override semantic_id_mappings.json path from Stage 1')
    parser.add_argument('--purified_content_path', type=str, default=None,
                        help='Override item_purified_content.npy path from Stage 1')
    parser.add_argument('--purified_collab_path', type=str, default=None,
                        help='Override item_purified_collab.npy path from Stage 1')
    parser.add_argument('--purified_dim', type=int, default=None,
                        help='Override purified feature dimension from Stage 1 (default: 128)')

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

    parser.add_argument(
        '--log_clean',
        action='store_true',
        default=False,
        help='Clean logging: no timestamps, no console handler (for batch runs)'
    )

    parser.add_argument('--eval_every_n_epochs', type=int, default=None,
                        help='Evaluate every N epochs (default: 3)')

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
    
    # Feature: Multi-source DSI Fusion (Stage 1 purified features)
    parser.add_argument(
        '--use_multimodal_fusion',
        action='store_true',
        help='Enable DSI: 3-way fusion of ID + purified_content + purified_collab'
    )
    parser.add_argument(
        '--fusion_gate_type',
        type=str, default='moe', choices=['learned', 'fixed', 'attention', 'moe', 'dense'],
        help='Fusion gating mechanism (default: moe)'
    )

    # MoE parameters
    parser.add_argument('--moe_num_experts', type=int, default=3,
                        help='Number of MoE experts (default: 3)')
    parser.add_argument('--moe_expert_hidden_dim', type=int, default=256,
                        help='Expert hidden dimension (default: 256)')
    parser.add_argument('--moe_top_k', type=int, default=2,
                        help='Top-K experts per input (default: 2)')
    parser.add_argument('--moe_use_load_balancing', action='store_true',
                        help='Use MoE load balancing loss')
    parser.add_argument('--moe_load_balance_weight', type=float, default=0.001,
                        help='Load balancing loss weight (default: 0.001)')

    # TCAF: Teacher-Conditioned Adaptive Fusion
    parser.add_argument('--use_teacher_gate', action='store_true',
                        help='Use teacher-conditioned modality routing (TCAF)')
    parser.add_argument('--lambda_align', type=float, default=0.0,
                        help='Teacher alignment loss weight (default: 0.0)')
    parser.add_argument('--teacher_dim', type=int, default=832,
                        help='Teacher prototype dimension (default: 832)')
    parser.add_argument('--teacher_path', type=str, default=None,
                        help='Path to teacher_prototypes.npy')

    # ============================================================
    # NEW FEATURES: Structural Improvements
    # ============================================================
    
    # Feature 4: Dynamic Batching
    parser.add_argument(
        '--use_dynamic_batching',
        action='store_true',
        help='Use dynamic batching to reduce padding waste (pads to max length in batch)'
    )
    
    # Feature 5: Item/Layer Position Embeddings
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
    
    # Feature 6: Trie-Constrained Decoding
    parser.add_argument(
        '--use_trie_constraints',
        action='store_true',
        help='Enable Trie-constrained decoding (ensures all generated paths lead to real items)'
    )
    
    # Feature 8: Purified Semantic Predictor
    parser.add_argument(
        '--use_purified_predictor',
        action='store_true',
        help='Enable PurifiedSemanticPredictor: auxiliary MSE loss predicting target z_clean from decoder hidden states'
    )
    parser.add_argument(
        '--purified_predictor_weight',
        type=float, default=0.1,
        help='Weight for purified predictor loss (default: 0.1)'
    )

    # Feature 7: Adaptive Temperature Scaling
    parser.add_argument(
        '--use_adaptive_temperature',
        action='store_true',
        help='Enable adaptive temperature scaling for hard negative mining'
    )
    parser.add_argument(
        '--tau_alpha',
        type=float,
        default=None,
        help='Sensitivity to branch density (default: 0.5, range: 0.3-0.8)'
    )
    parser.add_argument(
        '--tau_min',
        type=float,
        default=None,
        help='Minimum temperature for dense branches (default: 0.1)'
    )
    parser.add_argument(
        '--tau_max',
        type=float,
        default=None,
        help='Maximum temperature for sparse branches (default: 2.0)'
    )
    parser.add_argument(
        '--tau_start_layer',
        type=int,
        default=None,
        help='Start applying adaptive temperature from this layer (0=all layers, 1=skip Layer 0)'
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
    
    # Check if multimodal fusion or purified predictor is enabled
    use_multimodal = config['training'].use_multimodal_fusion
    use_purified_predictor = getattr(config['training'], 'use_purified_predictor', False)
    purified_content_path = config['data'].purified_content_path
    purified_collab_path = config['data'].purified_collab_path
    fast_dev = config.get('fast_dev', {})

    teacher_path = config['training'].teacher_path
    train_dataset, valid_dataset, test_dataset, semantic_mapper = create_datasets(
        sequence_data_dir=config['data'].sequence_data_path,
        semantic_mapping_path=config['data'].semantic_mapping_path,
        max_len=config['data'].max_seq_length,
        codebook_size=config['model'].codebook_size,
        num_layers=config['model'].num_code_layers,
        pad_token_id=config['model'].pad_token_id,
        model_config=config['model'],
        codebook_sizes=codebook_sizes,
        purified_content_path=purified_content_path,
        purified_collab_path=purified_collab_path,
        use_multimodal=use_multimodal or use_purified_predictor,
        train_sample_limit=fast_dev.get('train_sample_limit'),
        valid_sample_limit=fast_dev.get('valid_sample_limit'),
        test_sample_limit=fast_dev.get('test_sample_limit'),
        teacher_path=teacher_path,
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


def _create_model(logger, config: dict, semantic_mapper=None):
    """Create TIGER model.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        semantic_mapper: SemanticIDMapper instance
        
    Returns:
        TIGER model instance
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREATING MODEL")
    logger.info("=" * 80)
    
    # Create model with training config for enhanced features
    model = create_model(config['model'], config['training'])
    
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

    if args.eval_every_n_epochs is not None:
        config_kwargs['eval_every_n_epochs'] = args.eval_every_n_epochs
    
    # Learning rate scheduler
    if args.lr_scheduler is not None:
        config_kwargs['lr_scheduler'] = args.lr_scheduler
    
    if args.use_multimodal_fusion:
        config_kwargs['use_multimodal_fusion'] = True
        config_kwargs['fusion_gate_type'] = args.fusion_gate_type
        if args.fusion_gate_type in ('moe', 'dense'):
            config_kwargs['moe_num_experts'] = args.moe_num_experts
            config_kwargs['moe_expert_hidden_dim'] = args.moe_expert_hidden_dim
            config_kwargs['moe_top_k'] = args.moe_top_k
            config_kwargs['moe_use_load_balancing'] = args.moe_use_load_balancing
            config_kwargs['moe_load_balance_weight'] = args.moe_load_balance_weight

    # NEW FEATURES: Structural Improvements
    if args.use_dynamic_batching:
        config_kwargs['use_dynamic_batching'] = True
    
    if args.use_item_layer_emb:
        config_kwargs['use_item_layer_emb'] = True
        config_kwargs['use_temporal_decay'] = args.use_temporal_decay
    
    # Feature 6: Trie-Constrained Decoding
    if args.use_trie_constraints:
        config_kwargs['use_trie_constraints'] = True
    
    # Feature 7: Adaptive Temperature Scaling
    if args.use_adaptive_temperature:
        config_kwargs['use_adaptive_temperature'] = True
        if args.tau_alpha is not None:
            config_kwargs['tau_alpha'] = args.tau_alpha
        if args.tau_min is not None:
            config_kwargs['tau_min'] = args.tau_min
        if args.tau_max is not None:
            config_kwargs['tau_max'] = args.tau_max
        if args.tau_start_layer is not None:
            config_kwargs['tau_start_layer'] = args.tau_start_layer

    # Feature 8: Purified Semantic Predictor
    if args.use_purified_predictor:
        config_kwargs['use_purified_predictor'] = True
        config_kwargs['purified_predictor_weight'] = args.purified_predictor_weight

    # TCAF: Teacher-Conditioned Adaptive Fusion
    if args.use_teacher_gate:
        config_kwargs['use_teacher_gate'] = True
    if args.lambda_align > 0:
        config_kwargs['lambda_align'] = args.lambda_align
    if args.teacher_dim is not None:
        config_kwargs['teacher_dim'] = args.teacher_dim
    if args.teacher_path is not None:
        config_kwargs['teacher_path'] = args.teacher_path

    # Stage 1 path overrides (for hyperparameter sweep)
    if args.semantic_mapping_path is not None:
        config_kwargs['semantic_mapping_path'] = args.semantic_mapping_path
    if args.purified_content_path is not None:
        config_kwargs['purified_content_path'] = args.purified_content_path
    if args.purified_collab_path is not None:
        config_kwargs['purified_collab_path'] = args.purified_collab_path
    if args.purified_dim is not None:
        config_kwargs['purified_dim'] = args.purified_dim

    return config_kwargs


def main():
    """Main training function."""
    args = parse_args()

    # Enable TF32 for faster matmul/conv on Ampere+ GPUs (zero accuracy loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
        
        # Build configuration with optional overrides
        config_kwargs = _build_config_kwargs(args)
        config = get_config(args.config, **config_kwargs)

    if args.fast_dev_config is not None:
        fast_dev = load_fast_dev_config(args.fast_dev_config)
        apply_fast_dev_overrides(config, fast_dev)

    # Setup logging
    setup_logging(Path(config['output_dir']), args.log_level, clean=args.log_clean)
    logger = logging.getLogger(__name__)
    
    # Log configuration
    _log_config(logger, config, config['model_type'])
    
    # Set random seed
    set_seed(config['training'].seed)
    
    # Load data
    train_loader, valid_loader, test_loader, semantic_mapper = _load_data(logger, config)
    
    # Create model
    model = _create_model(logger, config, semantic_mapper)
    
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
