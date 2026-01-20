"""
Configuration management for ActionPiece recommender system.

Provides dataset-specific configurations and model hyperparameters
following the ActionPiece paper settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os
from datetime import datetime


@dataclass
class ActionPieceModelConfig:
    """ActionPiece T5 model configuration (paper settings)."""
    # T5 architecture (from paper: 4 layers, 6 heads, d_model=128)
    num_layers: int = 4
    num_decoder_layers: int = 4
    d_model: int = 128
    d_ff: int = 1024
    num_heads: int = 6
    d_kv: int = 64
    dropout_rate: float = 0.1
    feed_forward_proj: str = "relu"
    
    # Vocabulary size (will be set dynamically from ActionPiece tokenizer)
    _vocab_size: Optional[int] = None
    
    # ActionPiece specific
    n_categories: int = 5  # 4 PQ codes + 1 hash bucket
    n_inference_ensemble: int = 5  # q=5 in paper
    
    def set_vocab_size(self, vocab_size: int):
        """Set vocabulary size from ActionPiece tokenizer."""
        self._vocab_size = vocab_size
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._vocab_size is not None:
            return self._vocab_size
        return 40002  # Default: 40000 + BOS + EOS
    
    @property
    def pad_token_id(self) -> int:
        return 0
    
    @property
    def eos_token_id(self) -> int:
        return self._vocab_size - 1 if self._vocab_size else 40001


@dataclass
class ActionPieceDataConfig:
    """Data configuration for ActionPiece."""
    dataset_name: str
    sequence_data_path: str
    tokenizer_path: str  # Path to actionpiece.json
    item2feat_path: str  # Path to item2feat.json
    max_seq_length: int = 20  # Paper setting
    train_shuffle: str = 'feature'  # SPR augmentation
    
    def __post_init__(self):
        """Validate paths exist."""
        if not os.path.exists(self.sequence_data_path):
            raise ValueError(f"Sequence data path does not exist: {self.sequence_data_path}")
        if not os.path.exists(self.tokenizer_path):
            raise ValueError(f"Tokenizer path does not exist: {self.tokenizer_path}")
        if not os.path.exists(self.item2feat_path):
            raise ValueError(f"Item2feat path does not exist: {self.item2feat_path}")


@dataclass
class ActionPieceTrainingConfig:
    """Training configuration for ActionPiece (paper settings)."""
    batch_size: int = 128  # Paper setting
    eval_batch_size: int = 128
    num_epochs: int = 300  # Paper: 200 with early stop
    learning_rate: float = 0.001  # Paper: 0.001 for Beauty
    warmup_steps: int = 10000  # Paper setting
    warmup_ratio: float = 0.1
    weight_decay: float = 0.15 # 0.07 for cds, 0.15 for others
    gradient_clip: float = 1.0
    min_lr: float = 1e-6
    
    # Evaluation
    eval_every_n_epochs: int = 1
    topk_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    beam_size: int = 30  # Aligned with our framework (paper uses 50)
    
    # Early stopping
    early_stopping_patience: int = 8
    early_stopping_metric: str = "NDCG@20"
    
    # Checkpointing
    save_every_n_epochs: int = 10
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 100
    verbose: bool = False
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    
    # Reproducibility
    seed: int = 42


def get_actionpiece_model_config(model_type: str = "t5-tiny-2") -> ActionPieceModelConfig:
    """Get model configuration by type.
    
    Args:
        model_type: Type of model configuration
    
    Returns:
        ActionPieceModelConfig instance
    """
    if model_type == "t5-tiny-2":
        # Our framework's t5-tiny-2 aligned with ActionPiece paper
        return ActionPieceModelConfig(
            num_layers=4,
            num_decoder_layers=4,
            d_model=128,
            d_ff=1024,
            num_heads=6,
            d_kv=64,
            dropout_rate=0.1,
            feed_forward_proj="relu",
            n_inference_ensemble=5
        )
    elif model_type == "actionpiece-paper":
        return ActionPieceModelConfig(
            num_layers=4,
            num_decoder_layers=4,
            d_model=256,
            d_ff=2048,
            num_heads=6,
            d_kv=64,
            dropout_rate=0.1,
            feed_forward_proj="relu",
            n_inference_ensemble=5
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_actionpiece_beauty_config(
    sequence_data_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    item2feat_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-tiny-2",
    **kwargs
) -> dict:
    """Get configuration for Beauty dataset with ActionPiece.
    
    Args:
        sequence_data_path: Path to sequence data directory
        tokenizer_path: Path to actionpiece.json
        item2feat_path: Path to item2feat.json
        output_dir: Directory to save outputs
        checkpoint_dir: Directory to save checkpoints
        model_type: Type of model configuration
        **kwargs: Additional overrides
    
    Returns:
        Configuration dictionary
    """
    # Default paths
    default_seq_path = "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
    default_tokenizer_path = "scripts/output/actionpiece_tokenizer/beauty/actionpiece.json"
    default_item2feat_path = "scripts/output/actionpiece_tokenizer/beauty/item2feat.json"
    
    sequence_data_path = sequence_data_path or default_seq_path
    tokenizer_path = tokenizer_path or default_tokenizer_path
    item2feat_path = item2feat_path or default_item2feat_path
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_keywords = kwargs.get('output_keywords', '')
        if output_keywords:
            output_dir = f"scripts/output/recommender/actionpiece/beauty/{timestamp}_{output_keywords}"
        else:
            output_dir = f"scripts/output/recommender/actionpiece/beauty/{timestamp}"
    
    checkpoint_dir = checkpoint_dir or output_dir
    
    # Get model config
    model_config = get_actionpiece_model_config(model_type)
    
    # Data config
    data_config = ActionPieceDataConfig(
        dataset_name="beauty",
        sequence_data_path=sequence_data_path,
        tokenizer_path=tokenizer_path,
        item2feat_path=item2feat_path,
        max_seq_length=20,
        train_shuffle='feature'
    )
    
    # Training config (Beauty paper settings)
    training_config = ActionPieceTrainingConfig(
        batch_size=128,
        eval_batch_size=128,
        num_epochs=200,
        learning_rate=1e-3,  
        warmup_steps=1000,
        gradient_clip=1.0,
        beam_size=30,  # Aligned with our framework
        early_stopping_patience=20
    )
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
        elif hasattr(data_config, key):
            setattr(data_config, key, value)
        elif hasattr(training_config, key):
            setattr(training_config, key, value)
    
    return {
        "model": model_config,
        "data": data_config,
        "training": training_config,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "model_type": model_type
    }


def get_actionpiece_sports_config(
    sequence_data_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    item2feat_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-tiny-2",
    **kwargs
) -> dict:
    """Get configuration for Sports dataset with ActionPiece."""
    default_seq_path = "dataset/Amazon-Sports/processed/sports-tiger-sentenceT5base/Sports"
    default_tokenizer_path = "scripts/output/actionpiece_tokenizer/sports/actionpiece.json"
    default_item2feat_path = "scripts/output/actionpiece_tokenizer/sports/item2feat.json"
    
    sequence_data_path = sequence_data_path or default_seq_path
    tokenizer_path = tokenizer_path or default_tokenizer_path
    item2feat_path = item2feat_path or default_item2feat_path
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_keywords = kwargs.get('output_keywords', '')
        if output_keywords:
            output_dir = f"scripts/output/recommender/actionpiece/sports/{timestamp}_{output_keywords}"
        else:
            output_dir = f"scripts/output/recommender/actionpiece/sports/{timestamp}"
    
    checkpoint_dir = checkpoint_dir or output_dir
    
    model_config = get_actionpiece_model_config(model_type)
    
    data_config = ActionPieceDataConfig(
        dataset_name="sports",
        sequence_data_path=sequence_data_path,
        tokenizer_path=tokenizer_path,
        item2feat_path=item2feat_path,
        max_seq_length=20,
        train_shuffle='feature'
    )
    
    # Training config (Sports paper settings)
    training_config = ActionPieceTrainingConfig(
        batch_size=128,
        eval_batch_size=128,
        num_epochs=200,
        learning_rate=5e-3,  
        warmup_steps=1000,
        gradient_clip=1.0,
        beam_size=30,
        early_stopping_patience=20
    )
    
    for key, value in kwargs.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
        elif hasattr(data_config, key):
            setattr(data_config, key, value)
        elif hasattr(training_config, key):
            setattr(training_config, key, value)
    
    return {
        "model": model_config,
        "data": data_config,
        "training": training_config,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "model_type": model_type
    }


def get_actionpiece_toys_config(
    sequence_data_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    item2feat_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-tiny-2",
    **kwargs
) -> dict:
    """Get configuration for Toys dataset with ActionPiece."""
    default_seq_path = "dataset/Amazon-Toys/processed/toys-tiger-sentenceT5base/Toys"
    default_tokenizer_path = "scripts/output/actionpiece_tokenizer/toys/actionpiece.json"
    default_item2feat_path = "scripts/output/actionpiece_tokenizer/toys/item2feat.json"
    
    sequence_data_path = sequence_data_path or default_seq_path
    tokenizer_path = tokenizer_path or default_tokenizer_path
    item2feat_path = item2feat_path or default_item2feat_path
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_keywords = kwargs.get('output_keywords', '')
        if output_keywords:
            output_dir = f"scripts/output/recommender/actionpiece/toys/{timestamp}_{output_keywords}"
        else:
            output_dir = f"scripts/output/recommender/actionpiece/toys/{timestamp}"
    
    checkpoint_dir = checkpoint_dir or output_dir
    
    model_config = get_actionpiece_model_config(model_type)
    
    data_config = ActionPieceDataConfig(
        dataset_name="toys",
        sequence_data_path=sequence_data_path,
        tokenizer_path=tokenizer_path,
        item2feat_path=item2feat_path,
        max_seq_length=20,
        train_shuffle='feature'
    )
    
    training_config = ActionPieceTrainingConfig(
        batch_size=128,
        eval_batch_size=128,
        num_epochs=200,
        learning_rate=1e-3,
        warmup_steps=1000,
        gradient_clip=1.0,
        beam_size=30,
        early_stopping_patience=20
    )
    
    for key, value in kwargs.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
        elif hasattr(data_config, key):
            setattr(data_config, key, value)
        elif hasattr(training_config, key):
            setattr(training_config, key, value)
    
    return {
        "model": model_config,
        "data": data_config,
        "training": training_config,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "model_type": model_type
    }


def get_actionpiece_cds_config(
    sequence_data_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    item2feat_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-tiny-2",
    **kwargs
) -> dict:
    """Get configuration for CDs dataset with ActionPiece."""
    default_seq_path = "dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs"
    default_tokenizer_path = "scripts/output/actionpiece_tokenizer/cds/actionpiece.json"
    default_item2feat_path = "scripts/output/actionpiece_tokenizer/cds/item2feat.json"
    
    sequence_data_path = sequence_data_path or default_seq_path
    tokenizer_path = tokenizer_path or default_tokenizer_path
    item2feat_path = item2feat_path or default_item2feat_path
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_keywords = kwargs.get('output_keywords', '')
        if output_keywords:
            output_dir = f"scripts/output/recommender/actionpiece/cds/{timestamp}_{output_keywords}"
        else:
            output_dir = f"scripts/output/recommender/actionpiece/cds/{timestamp}"
    
    checkpoint_dir = checkpoint_dir or output_dir
    
    model_config = get_actionpiece_model_config(model_type)
    
    data_config = ActionPieceDataConfig(
        dataset_name="cds",
        sequence_data_path=sequence_data_path,
        tokenizer_path=tokenizer_path,
        item2feat_path=item2feat_path,
        max_seq_length=20,
        train_shuffle='feature'
    )
    
    training_config = ActionPieceTrainingConfig(
        batch_size=128,
        eval_batch_size=128,
        num_epochs=200,
        learning_rate=1e-3,
        warmup_steps=1000,
        gradient_clip=1.0,
        beam_size=30,
        early_stopping_patience=20
    )
    
    for key, value in kwargs.items():
        if hasattr(model_config, key):
            setattr(model_config, key, value)
        elif hasattr(data_config, key):
            setattr(data_config, key, value)
        elif hasattr(training_config, key):
            setattr(training_config, key, value)
    
    return {
        "model": model_config,
        "data": data_config,
        "training": training_config,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
        "model_type": model_type
    }


# Config registry
ACTIONPIECE_CONFIG_REGISTRY = {
    "beauty": get_actionpiece_beauty_config,
    "sports": get_actionpiece_sports_config,
    "toys": get_actionpiece_toys_config,
    "cds": get_actionpiece_cds_config
}


def get_actionpiece_config(config_name: str, **kwargs) -> dict:
    """Get ActionPiece configuration by name.
    
    Args:
        config_name: Name of the configuration ("beauty", "sports", or "toys")
        **kwargs: Configuration parameters
    
    Returns:
        Configuration dictionary
    """
    if config_name not in ACTIONPIECE_CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown config name: {config_name}. "
            f"Available configs: {list(ACTIONPIECE_CONFIG_REGISTRY.keys())}"
        )
    
    return ACTIONPIECE_CONFIG_REGISTRY[config_name](**kwargs)
