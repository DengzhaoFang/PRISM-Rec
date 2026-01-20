"""
Configuration management for the recommender system.

Provides dataset-specific configurations and model hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ModelConfig:
    """T5 Nano model configuration."""
    # 两个标准T5缩放规则：规则 1: $d_{model} = num_{heads} \times d_{kv}$ 
    #                     规则 2: $d_{ff} = 4 \times d_{model}$。
    num_layers: int = 1  # Shared encoder: 1 layer (EAGER paper)
    num_decoder_layers: int = 4
    d_model: int = 64
    d_ff: int = 256
    num_heads: int = 1
    d_kv: int = 64
    dropout_rate: float = 0.1
    feed_forward_proj: str = "relu"
    
    # Semantic code parameters
    num_code_layers: int = 4  # Number of RQ-VAE layers
    codebook_size: int = 256  # Size of each codebook
    
    # Vocabulary size (will be set dynamically based on actual data)
    _vocab_size: Optional[int] = None
    _num_items: Optional[int] = None  # Number of items for encoder embedding

    
    def set_vocab_size(self, vocab_size: int, num_items: Optional[int] = None):
        """Set vocabulary size based on actual data.
        
        Args:
            vocab_size: Actual vocabulary size from SemanticIDMapper
            num_items: Number of items (for encoder embedding)
        """
        self._vocab_size = vocab_size
        if num_items is not None:
            self._num_items = num_items

    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size.
        
        Returns actual vocab size if set, otherwise returns theoretical maximum.
        """
        if self._vocab_size is not None:
            return self._vocab_size
        else:
            # Theoretical maximum (may waste some embedding space)
            return self.num_code_layers * self.codebook_size + 1
    
    @property
    def pad_token_id(self) -> int:
        """Padding token ID."""
        return 0
    
    @property
    def eos_token_id(self) -> int:
        """End of sequence token ID."""
        return 1  # Changed from 0 to avoid collision with pad_token_id
    
    @property
    def mask_token_id(self) -> int:
        """Mask token ID for STT (Semantic-Guided Transfer Task)."""
        return 2  # Reserved for masking in STT reconstruction task
        
    @property
    def num_items(self) -> int:
        """Get number of items."""
        if self._num_items is not None:
            return self._num_items
        else:
            return 10000  # Default fallback



def get_model_config(model_type: str = "default") -> ModelConfig:
    """Get model configuration by type.
    
    Args:
        model_type: Type of model configuration
    
    Returns:
        ModelConfig instance
    """
    if model_type == "t5-pico":
        return ModelConfig(
            num_layers=1,  # Shared encoder: 1 layer
            num_decoder_layers=2,
            d_model=32,
            d_ff=128,
            num_heads=1,
            d_kv=32,
            dropout_rate=0.1,
            feed_forward_proj="relu"
        )
    elif model_type == "t5-nano":
        return ModelConfig()
    elif model_type == "t5-micro":
        return ModelConfig(
            num_layers=1,  # Shared encoder: 1 layer
            num_decoder_layers=4,
            d_model=128,
            d_ff=512,
            num_heads=2,
            d_kv=64,
            dropout_rate=0.2,
            feed_forward_proj="relu"
        )
    elif model_type == "t5-tiny":
        return ModelConfig(
            num_layers=1,  # Shared encoder: 1 layer
            num_decoder_layers=4,
            d_model=64,
            d_ff=1024,
            num_heads=6,
            d_kv=64,
            dropout_rate=0.1,
            feed_forward_proj="relu"
        )
    elif model_type == "t5-tiny-2":
        return ModelConfig(
            num_layers=1,  # Shared encoder: 1 layer
            num_decoder_layers=4,
            d_model=128,
            d_ff=1024,
            num_heads=6,
            d_kv=64,
            dropout_rate=0.1,
            feed_forward_proj="relu"
        )
    elif model_type == "t5-small":
        return ModelConfig(
            num_layers=1,  # Shared encoder: 1 layer
            num_decoder_layers=6,
            d_model=512,
            d_ff=2048,
            num_heads=8,
            d_kv=64,
            dropout_rate=0.15, 
            feed_forward_proj="relu"
        )

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available types: 't5-pico', 't5-nano', 't5-micro',  't5-tiny', 't5-tiny-2', 't5-small'"
        )


@dataclass
class DataConfig:
    """Data configuration for different datasets."""
    dataset_name: str
    sequence_data_path: str
    semantic_mapping_path: str  # Semantic path (content embeddings)
    behavior_mapping_path: Optional[str] = None  # Behavior path (collaborative embeddings, for EAGER)
    max_seq_length: int = 20
    
    # GCT (Global Contrastive Task) embedding paths for EAGER
    behavior_emb_path: Optional[str] = None  # Path to behavior embeddings (.npy)
    semantic_emb_path: Optional[str] = None  # Path to semantic embeddings (.parquet)
    
    def __post_init__(self):
        """Validate paths exist."""
        if not os.path.exists(self.sequence_data_path):
            raise ValueError(f"Sequence data path does not exist: {self.sequence_data_path}")
        if not os.path.exists(self.semantic_mapping_path):
            raise ValueError(f"Semantic mapping path does not exist: {self.semantic_mapping_path}")
        if self.behavior_mapping_path is not None and not os.path.exists(self.behavior_mapping_path):
            raise ValueError(f"Behavior mapping path does not exist: {self.behavior_mapping_path}")
        
        # Validate GCT embedding paths for EAGER (dual-stream mode)
        if self.behavior_mapping_path is not None:
            # This is EAGER mode, GCT embeddings are required
            if self.behavior_emb_path is None:
                raise ValueError(
                    f"EAGER mode requires behavior_emb_path for GCT (Global Contrastive Task). "
                    f"Please specify the path to behavior embeddings (.npy file from collaborative filtering model)."
                )
            if self.semantic_emb_path is None:
                raise ValueError(
                    f"EAGER mode requires semantic_emb_path for GCT (Global Contrastive Task). "
                    f"Please specify the path to semantic embeddings (.parquet file with content features)."
                )
            
            # Check if files exist
            if not os.path.exists(self.behavior_emb_path):
                raise ValueError(
                    f"Behavior embedding file does not exist: {self.behavior_emb_path}\n"
                    f"This file is required for GCT (Global Contrastive Task) in EAGER.\n"
                    f"Expected: .npy file containing collaborative filtering embeddings (e.g., from LightGCN)."
                )
            if not os.path.exists(self.semantic_emb_path):
                raise ValueError(
                    f"Semantic embedding file does not exist: {self.semantic_emb_path}\n"
                    f"This file is required for GCT (Global Contrastive Task) in EAGER.\n"
                    f"Expected: .parquet file containing content-based embeddings (e.g., from Sentence-T5)."
                )


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 128
    eval_batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 5e-4  # Changed from 1e-4 to 5e-4 based on experimental results
    warmup_steps: int = 0
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    
    # Learning rate scheduler
    # Options: 'none', 'warmup_cosine', 'reduce_on_plateau', 'exponential', 'step'
    lr_scheduler: str = 'warmup_cosine'  
    warmup_ratio: float = 0.1  # Warmup as % of total training steps (used by warmup_cosine)
    min_lr: float = 1e-6  # Minimum learning rate for cosine annealing
    lr_decay_factor: float = 0.5  # Factor for ReduceLROnPlateau and StepLR
    lr_patience: int = 5  # Patience for ReduceLROnPlateau (epochs without improvement)
    lr_step_size: int = 50  # Step size for StepLR (every N epochs)
    lr_gamma: float = 0.95  # Gamma for ExponentialLR
    
    # EAGER-specific hyperparameters
    lambda_1: float = 1.0  # Weight for Global Contrastive Task (GCT)
    lambda_2: float = 1.0  # Weight for Semantic-Guided Transfer Task (STT)
    mask_ratio_recon: float = 0.5  # Masking ratio for STT reconstruction
    mask_ratio_recog: float = 0.5  # Masking ratio for STT recognition
    num_negatives: int = 64  # Number of negative samples for InfoNCE loss
    
    # Evaluation
    eval_every_n_epochs: int = 1
    topk_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    beam_size: int = 30
    
    # Early stopping
    early_stopping_patience: int = 15  # Increased from 10 to 15 for better convergence
    early_stopping_metric: str = "NDCG@20"
    
    # Checkpointing
    save_every_n_epochs: int = 10
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 100
    verbose: bool = False  # Enable verbose sample printing during eval/test
    
    # Device
    device: str = "cuda"
    num_workers: int = 4
    
    # Reproducibility
    seed: int = 42


def _create_dataset_config(
    dataset_name: str,
    sequence_data_path: Optional[str],
    semantic_mapping_path: Optional[str],
    output_dir: Optional[str],
    checkpoint_dir: Optional[str],
    model_type: str,
    default_paths: dict,
    **kwargs
) -> dict:
    """Helper function to create dataset configuration (DRY principle).
    
    Args:
        dataset_name: Name of dataset
        sequence_data_path: Path to sequence data
        semantic_mapping_path: Path to semantic mapping
        output_dir: Output directory
        checkpoint_dir: Checkpoint directory
        model_type: Model type
        default_paths: Dictionary with default paths for this dataset
        **kwargs: Additional overrides
    
    Returns:
        Configuration dictionary
    """
    from datetime import datetime
    
    # Set default paths
    sequence_data_path = sequence_data_path or default_paths['sequence_data_path']
    semantic_mapping_path = semantic_mapping_path or default_paths['semantic_mapping_path']
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # Add output_keywords to directory name if provided
        output_keywords = kwargs.get('output_keywords', None)
        if output_keywords:
            output_dir = f"scripts/output/recommender/eager/{dataset_name}/{timestamp}_{output_keywords}"
        else:
            output_dir = f"scripts/output/recommender/eager/{dataset_name}/{timestamp}"
    
    checkpoint_dir = checkpoint_dir or output_dir
    
    # Get model config
    model_config = get_model_config(model_type)
    
    # Data config
    data_config = DataConfig(
        dataset_name=dataset_name,
        sequence_data_path=sequence_data_path,
        semantic_mapping_path=semantic_mapping_path,
        behavior_mapping_path=kwargs.get('behavior_mapping_path', default_paths.get('behavior_mapping_path', None)),
        max_seq_length=20,
        # GCT embedding paths
        behavior_emb_path=kwargs.get('behavior_emb_path', default_paths.get('behavior_emb_path', None)),
        semantic_emb_path=kwargs.get('semantic_emb_path', default_paths.get('semantic_emb_path', None))
    )
    
    # Training config (adjust based on model type)
    if model_type in ["t5-small"]:
         training_config = TrainingConfig(
            batch_size=64,
            eval_batch_size=64,
            num_epochs=200,
            learning_rate=1e-4,
            warmup_steps=1000,
            weight_decay=0.01,
            gradient_clip=1.0,
            beam_size=30,
            early_stopping_patience=10,
            eval_every_n_epochs=2
        )
    else:
        training_config = TrainingConfig(
            batch_size=128,
            eval_batch_size=128,
            num_epochs=300,
            learning_rate=1e-3,  # Paper recommends 0.001
            warmup_steps=1000,
            gradient_clip=1.0,
            beam_size=30,
            early_stopping_patience=15,
            eval_every_n_epochs=3
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


def get_beauty_config(
    sequence_data_path: Optional[str] = None,
    semantic_mapping_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-tiny",
    **kwargs
) -> dict:
    """Get configuration for Beauty dataset.
    
    Args:
        sequence_data_path: Path to sequence data directory
        semantic_mapping_path: Path to semantic ID mapping JSON file
        output_dir: Directory to save outputs
        checkpoint_dir: Directory to save checkpoints
        model_type: Type of model configuration
        **kwargs: Additional overrides
    
    Returns:
        Dictionary containing all configurations
    """
    default_paths = {
        'sequence_data_path': "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty",
        'semantic_mapping_path': "scripts/output/eager_tokenizer/beauty/hkm_k8_d2/semantic_id_mappings_semantic.json",
        'behavior_mapping_path': "scripts/output/eager_tokenizer/beauty/hkm_k8_d2/semantic_id_mappings_behavior.json",
        # GCT embedding paths for EAGER
        'behavior_emb_path': "dataset/Amazon-Beauty/processed/beauty-prism-sentenceT5base/Beauty/lightgcn/item_embeddings_collab.npy",
        'semantic_emb_path': "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty/item_emb.parquet"
    }
    return _create_dataset_config(
        dataset_name="beauty",
        sequence_data_path=sequence_data_path,
        semantic_mapping_path=semantic_mapping_path,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        default_paths=default_paths,
        **kwargs
    )


def get_sports_config(
    sequence_data_path: Optional[str] = None,
    semantic_mapping_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-nano",
    **kwargs
) -> dict:
    """Get configuration for Sports dataset."""
    default_paths = {
        'sequence_data_path': "dataset/Amazon-Sports/processed/sports-tiger-sentenceT5base/Sports",
        'semantic_mapping_path': "scripts/output/eager_tokenizer/sports/hkm_k8_d2/semantic_id_mappings_semantic.json",
        'behavior_mapping_path': "scripts/output/eager_tokenizer/sports/hkm_k8_d2/semantic_id_mappings_behavior.json",
        # GCT embedding paths for EAGER (if using dual-stream mode)
        'behavior_emb_path': "dataset/Amazon-Sports/processed/sports-prism-sentenceT5base/Sports/lightgcn/item_embeddings_collab.npy",
        'semantic_emb_path': "dataset/Amazon-Sports/processed/sports-tiger-sentenceT5base/Sports/item_emb.parquet"
    }
    
    return _create_dataset_config(
        dataset_name="sports",
        sequence_data_path=sequence_data_path,
        semantic_mapping_path=semantic_mapping_path,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        default_paths=default_paths,
        **kwargs
    )


def get_toys_config(
    sequence_data_path: Optional[str] = None,
    semantic_mapping_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-nano",
    **kwargs
) -> dict:
    """Get configuration for Toys dataset."""
    default_paths = {
        'sequence_data_path': "dataset/Amazon-Toys/processed/toys-tiger-sentenceT5base/Toys",
        'semantic_mapping_path': "scripts/output/eager_tokenizer/toys/hkm_k8_d2/semantic_id_mappings_semantic.json",
        'behavior_mapping_path': "scripts/output/eager_tokenizer/toys/hkm_k8_d2/semantic_id_mappings_behavior.json",
        # GCT embedding paths for EAGER (if using dual-stream mode)
        'behavior_emb_path': "dataset/Amazon-Toys/processed/toys-prism-sentenceT5base/Toys/lightgcn/item_embeddings_collab.npy",
        'semantic_emb_path': "dataset/Amazon-Toys/processed/toys-tiger-sentenceT5base/Toys/item_emb.parquet"
    }
    
    return _create_dataset_config(
        dataset_name="toys",
        sequence_data_path=sequence_data_path,
        semantic_mapping_path=semantic_mapping_path,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        default_paths=default_paths,
        **kwargs
    )


def get_cds_config(
    sequence_data_path: Optional[str] = None,
    semantic_mapping_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-nano",
    **kwargs
) -> dict:
    """Get configuration for CDs dataset."""
    default_paths = {
        'sequence_data_path': "dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs",
        'semantic_mapping_path': "scripts/output/eager_tokenizer/cds/hkm_k8_d2/semantic_id_mappings_semantic.json",
        'behavior_mapping_path': "scripts/output/eager_tokenizer/cds/hkm_k8_d2/semantic_id_mappings_behavior.json",
        # GCT embedding paths for EAGER (if using dual-stream mode)
        'behavior_emb_path': "dataset/Amazon-CDs/processed/cds-prism-sentenceT5base/CDs/lightgcn/item_embeddings_collab.npy",
        'semantic_emb_path': "dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs/item_emb.parquet"
    }
    
    return _create_dataset_config(
        dataset_name="cds",
        sequence_data_path=sequence_data_path,
        semantic_mapping_path=semantic_mapping_path,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        model_type=model_type,
        default_paths=default_paths,
        **kwargs
    )


# Mapping from config name to config function
CONFIG_REGISTRY = {
    "beauty": get_beauty_config,
    "sports": get_sports_config,
    "toys": get_toys_config,
    "cds": get_cds_config
}


def get_config(config_name: str, **kwargs) -> dict:
    """Get configuration by name.
    
    Args:
        config_name: Name of the configuration ("beauty", "sports", or "toys")
        **kwargs: Configuration parameters
    
    Returns:
        Configuration dictionary
    
    Raises:
        ValueError: If config_name is not recognized
    """
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown config name: {config_name}. "
            f"Available configs: {list(CONFIG_REGISTRY.keys())}"
        )
    
    return CONFIG_REGISTRY[config_name](**kwargs)

