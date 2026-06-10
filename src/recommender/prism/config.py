"""
Configuration management for the recommender system.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class ModelConfig:
    num_layers: int = 4
    num_decoder_layers: int = 4
    d_model: int = 64
    d_ff: int = 256
    num_heads: int = 1
    d_kv: int = 64
    dropout_rate: float = 0.1
    feed_forward_proj: str = "relu"

    num_code_layers: int = 3
    codebook_size: int = 256
    codebook_sizes: Optional[List[int]] = None

    _vocab_size: Optional[int] = None

    def set_vocab_size(self, vocab_size: int):
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is not None:
            return self._vocab_size
        if self.codebook_sizes is not None:
            return 1 + sum(self.codebook_sizes)
        return self.num_code_layers * self.codebook_size + 1

    @property
    def pad_token_id(self) -> int: return 0

    @property
    def eos_token_id(self) -> int: return 0


def get_model_config(model_type: str = "default") -> ModelConfig:
    if model_type == "t5-pico":
        return ModelConfig(num_layers=2, num_decoder_layers=2, d_model=32, d_ff=128, num_heads=1, d_kv=32)
    elif model_type == "t5-nano":
        return ModelConfig()
    elif model_type == "t5-micro":
        return ModelConfig(num_layers=4, num_decoder_layers=4, d_model=128, d_ff=512, num_heads=2, d_kv=64, dropout_rate=0.2)
    elif model_type == "t5-tiny":
        return ModelConfig(num_layers=4, num_decoder_layers=4, d_model=64, d_ff=1024, num_heads=6, d_kv=64)
    elif model_type == "t5-tiny-2":
        return ModelConfig(num_layers=4, num_decoder_layers=4, d_model=128, d_ff=1024, num_heads=6, d_kv=64)
    elif model_type == "t5-small":
        return ModelConfig(num_layers=6, num_decoder_layers=6, d_model=512, d_ff=2048, num_heads=8, d_kv=64)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@dataclass
class DataConfig:
    """Data configuration. Purified features loaded from Stage 1 output dir."""
    dataset_name: str
    sequence_data_path: str
    semantic_mapping_path: str
    purified_content_path: Optional[str] = None   # h_t_hat (128D)
    purified_collab_path: Optional[str] = None     # h_c_hat (128D)
    collab_embedding_path: Optional[str] = None    # LightGCN collab embeddings
    max_seq_length: int = 20

    def __post_init__(self):
        if not os.path.exists(self.sequence_data_path):
            raise ValueError(f"Sequence data path does not exist: {self.sequence_data_path}")
        if not os.path.exists(self.semantic_mapping_path):
            raise ValueError(f"Semantic mapping path does not exist: {self.semantic_mapping_path}")


@dataclass
class TrainingConfig:
    batch_size: int = 128
    eval_batch_size: int = 128
    num_epochs: int = 200
    learning_rate: float = 5e-4
    warmup_steps: int = 0
    weight_decay: float = 0.0
    gradient_clip: float = 1.0

    lr_scheduler: str = 'warmup_cosine'
    warmup_ratio: float = 0.1
    min_lr: float = 1e-6
    lr_decay_factor: float = 0.5
    lr_patience: int = 5
    lr_step_size: int = 50
    lr_gamma: float = 0.95

    eval_every_n_epochs: int = 3
    topk_list: List[int] = field(default_factory=lambda: [5, 10, 20])
    beam_size: int = 30

    early_stopping_patience: int = 15
    early_stopping_metric: str = "NDCG@20"

    save_every_n_epochs: int = 10
    keep_last_n_checkpoints: int = 3

    log_every_n_steps: int = 100
    verbose: bool = False

    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42

    # DSI: 3-way purified MoE fusion
    use_multimodal_fusion: bool = False
    fusion_gate_type: str = "moe"           # "moe", "dense", "learned", "attention", "fixed"
    purified_dim: int = 128                  # MCD-denoised feature dim from Stage 1

    # Purified Semantic Predictor: auxiliary MSE on target z_clean (256D)
    use_purified_predictor: bool = False
    purified_predictor_weight: float = 0.1

    # MoE parameters
    moe_num_experts: int = 3
    moe_expert_hidden_dim: int = 256
    moe_top_k: int = 2
    moe_use_load_balancing: bool = True
    moe_load_balance_weight: float = 0.01

    # Structural features
    use_dynamic_batching: bool = False
    use_item_layer_emb: bool = False
    use_temporal_decay: bool = True

    # Trie-Constrained Decoding
    use_trie_constraints: bool = False

    # Adaptive Temperature Scaling
    use_adaptive_temperature: bool = False
    tau_alpha: float = 0.5
    tau_min: float = 0.1
    tau_max: float = 2.0
    tau_mean_center: bool = False
    tau_k_ref: float = 50.0
    tau_start_layer: int = 0


def _create_dataset_config(
    dataset_name: str, sequence_data_path: Optional[str],
    semantic_mapping_path: Optional[str], output_dir: Optional[str],
    checkpoint_dir: Optional[str], model_type: str,
    default_paths: dict, **kwargs
) -> dict:
    from datetime import datetime

    sequence_data_path = sequence_data_path or default_paths['sequence_data_path']
    semantic_mapping_path = semantic_mapping_path or default_paths['semantic_mapping_path']
    purified_content_path = kwargs.get('purified_content_path') or default_paths.get('purified_content_path')
    purified_collab_path = kwargs.get('purified_collab_path') or default_paths.get('purified_collab_path')

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_keywords = kwargs.get('output_keywords', None)
        output_dir = f"scripts/output/recommender/prism/{dataset_name}/{timestamp}{'_' + output_keywords if output_keywords else ''}"

    checkpoint_dir = checkpoint_dir or output_dir
    model_config = get_model_config(model_type)

    collab_path = os.path.join(os.path.dirname(sequence_data_path), 'lightgcn', 'item_embeddings_collab.npy')
    data_config = DataConfig(
        dataset_name=dataset_name, sequence_data_path=sequence_data_path,
        semantic_mapping_path=semantic_mapping_path,
        purified_content_path=purified_content_path,
        purified_collab_path=purified_collab_path,
        collab_embedding_path=collab_path,
        max_seq_length=20
    )

    if model_type in ["t5-small"]:
        training_config = TrainingConfig(
            batch_size=128, eval_batch_size=128, num_epochs=200,
            learning_rate=1e-3, warmup_steps=1000, weight_decay=0.001,
            gradient_clip=1.0, beam_size=30, early_stopping_patience=10,
            eval_every_n_epochs=2
        )
    else:
        training_config = TrainingConfig(
            batch_size=128, eval_batch_size=128, num_epochs=300,
            learning_rate=5e-4, warmup_steps=1000, gradient_clip=1.0,
            beam_size=30, early_stopping_patience=10, eval_every_n_epochs=3
        )

    for key, value in kwargs.items():
        if value is None: continue
        if hasattr(model_config, key): setattr(model_config, key, value)
        elif hasattr(data_config, key): setattr(data_config, key, value)
        elif hasattr(training_config, key): setattr(training_config, key, value)

    return {
        "model": model_config, "data": data_config, "training": training_config,
        "output_dir": output_dir, "checkpoint_dir": checkpoint_dir, "model_type": model_type
    }


def get_beauty_config(
    sequence_data_path: Optional[str] = None,
    semantic_mapping_path: Optional[str] = None,
    purified_content_path: Optional[str] = None,
    purified_collab_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-tiny-2",
    **kwargs
) -> dict:
    tokenizer_dir = "scripts/output/prism_tokenizer/beauty/hparam_stage1_PASCL/pa_scl_text_dominant"
    default_paths = {
        'sequence_data_path': "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty",
        'semantic_mapping_path': f"{tokenizer_dir}/semantic_id_mappings.json",
        'purified_content_path': f"{tokenizer_dir}/item_purified_content.npy",
        'purified_collab_path': f"{tokenizer_dir}/item_purified_collab.npy",
    }
    return _create_dataset_config(
        dataset_name="beauty", sequence_data_path=sequence_data_path,
        semantic_mapping_path=semantic_mapping_path,
        output_dir=output_dir, checkpoint_dir=checkpoint_dir,
        model_type=model_type, default_paths=default_paths,
        purified_content_path=purified_content_path,
        purified_collab_path=purified_collab_path,
        **kwargs
    )


def get_sports_config(
    sequence_data_path: Optional[str] = None,
    semantic_mapping_path: Optional[str] = None,
    purified_content_path: Optional[str] = None,
    purified_collab_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-nano",
    **kwargs
) -> dict:
    default_paths = {
        'sequence_data_path': "dataset/Amazon-Sports/processed/sports-tiger-sentenceT5base/Sports",
        'semantic_mapping_path': "scripts/output/prism_tokenizer/sports/3-256-32-ema-only-5-core-items/semantic_id_mappings.json",
        'purified_content_path': None, 'purified_collab_path': None,
    }
    return _create_dataset_config(
        dataset_name="sports", sequence_data_path=sequence_data_path,
        semantic_mapping_path=semantic_mapping_path,
        output_dir=output_dir, checkpoint_dir=checkpoint_dir,
        model_type=model_type, default_paths=default_paths,
        purified_content_path=purified_content_path,
        purified_collab_path=purified_collab_path,
        **kwargs
    )


def get_toys_config(
    sequence_data_path: Optional[str] = None,
    semantic_mapping_path: Optional[str] = None,
    purified_content_path: Optional[str] = None,
    purified_collab_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-tiny-2",
    **kwargs
) -> dict:
    default_paths = {
        'sequence_data_path': "dataset/Amazon-Toys/processed/toys-tiger-sentenceT5base/Toys",
        'semantic_mapping_path': "scripts/output/prism_tokenizer/toys/3-256-32-ema-only-5-core-items/semantic_id_mappings.json",
        'purified_content_path': None, 'purified_collab_path': None,
    }
    return _create_dataset_config(
        dataset_name="toys", sequence_data_path=sequence_data_path,
        semantic_mapping_path=semantic_mapping_path,
        output_dir=output_dir, checkpoint_dir=checkpoint_dir,
        model_type=model_type, default_paths=default_paths,
        purified_content_path=purified_content_path,
        purified_collab_path=purified_collab_path,
        **kwargs
    )


def get_cds_config(
    sequence_data_path: Optional[str] = None,
    semantic_mapping_path: Optional[str] = None,
    purified_content_path: Optional[str] = None,
    purified_collab_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    model_type: str = "t5-tiny-2",
    **kwargs
) -> dict:
    default_paths = {
        'sequence_data_path': "dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs",
        'semantic_mapping_path': "scripts/output/prism_tokenizer/cds/3-256-32-ema-only-5-core-items/semantic_id_mappings.json",
        'purified_content_path': None, 'purified_collab_path': None,
    }
    return _create_dataset_config(
        dataset_name="cds", sequence_data_path=sequence_data_path,
        semantic_mapping_path=semantic_mapping_path,
        output_dir=output_dir, checkpoint_dir=checkpoint_dir,
        model_type=model_type, default_paths=default_paths,
        purified_content_path=purified_content_path,
        purified_collab_path=purified_collab_path,
        **kwargs
    )


CONFIG_REGISTRY = {
    "beauty": get_beauty_config, "sports": get_sports_config,
    "toys": get_toys_config, "cds": get_cds_config
}


def get_config(config_name: str, **kwargs) -> dict:
    if config_name not in CONFIG_REGISTRY:
        raise ValueError(f"Unknown config name: {config_name}. Available: {list(CONFIG_REGISTRY.keys())}")
    return CONFIG_REGISTRY[config_name](**kwargs)
