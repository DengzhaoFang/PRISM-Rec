"""
Efficiency Benchmark for Generative Recommender Models.
"""

import argparse
import logging
import json
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)

# Dataset configurations
DATASET_CONFIGS = {
    'beauty': {
        'test_data': 'dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty/test.parquet',
        'checkpoints': {
            'TIGER': 'scripts/output/recommender/tiger/beauty/2026-01-06-22-02-28_3layer-tiger/best_model.pt',
            'LETTER': 'scripts/output/recommender/letter/beauty/2025-12-13-08-16-06_3layer-letter/best_model.pt',
            'ActionPiece': 'scripts/output/recommender/actionpiece/beauty/2025-12-03-23-55-37_actionpiece-spr/best_model.pt',
            'Prism': 'scripts/output/recommender/prism/beauty/2026-01-06-21-58-26_3layer-prism/best_model.pt',
            'EAGER': 'scripts/output/recommender/eager/beauty/2025-11-30-20-26-56_eager/checkpoint_epoch_90.pt',
            'SASRec': None,
        },
        'display_name': 'Beauty',
        'catalog_size': 12101,  # Will be computed if not set
    },
    'cds': {
        'test_data': 'dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs/test.parquet',
        'checkpoints': {
            'TIGER': 'scripts/output/recommender/tiger/cds/2025-12-13-00-29-43_3layer-tiger/best_model.pt',
            'LETTER': 'scripts/output/recommender/letter/cds/2025-12-13-17-49-46_3layer-letter/best_model.pt',
            'ActionPiece': 'scripts/output/recommender/actionpiece/cds/2026-01-09-20-26-12_actionpiece-large/best_model.pt',
            'Prism': 'scripts/output/recommender/prism/cds/2025-12-29-01-59-37_3layer-prism-tile/best_model.pt',
            'EAGER': 'scripts/output/recommender/eager/cds/2025-12-14-12-55-05_eager-dual-stream/best_model.pt',
            'SASRec': None,
        },
        'display_name': 'CDs',
        'catalog_size': 64443,  # Will be computed if not set
    },
}

# For backward compatibility
MODEL_CHECKPOINTS = DATASET_CONFIGS['beauty']['checkpoints']
TEST_DATA_PATH = DATASET_CONFIGS['beauty']['test_data']


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters.
    
    Returns:
        Dict with total, trainable, embedding, and non-embedding params
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count embedding parameters
    embedding_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_params += sum(p.numel() for p in module.parameters())
    
    return {
        'total': total,
        'trainable': trainable,
        'embedding': embedding_params,
        'non_embedding': total - embedding_params
    }


def count_auxiliary_task_parameters(model: nn.Module, model_name: str) -> int:
    """Count parameters used only for auxiliary tasks (training only, not inference).
    """
    auxiliary_params = 0
    
    if model_name == 'Prism':
        # Count codebook_predictor parameters
        if hasattr(model, 'codebook_predictor') and model.codebook_predictor is not None:
            codebook_params = sum(p.numel() for p in model.codebook_predictor.parameters())
            auxiliary_params += codebook_params
            logger.info(f"  Codebook predictor params (training only): {codebook_params:,}")
        
        # Count tag_predictor parameters
        if hasattr(model, 'tag_predictor') and model.tag_predictor is not None:
            tag_params = sum(p.numel() for p in model.tag_predictor.parameters())
            auxiliary_params += tag_params
            logger.info(f"  Tag predictor params (training only): {tag_params:,}")
    
    return auxiliary_params


def count_activated_parameters(model: nn.Module, model_name: str, config: dict) -> int:
    """Count activated parameters during inference.
    """
    total_params = count_parameters(model)
    
    # Start with total parameters
    activated_params = total_params['total']
    
    # Subtract auxiliary task parameters (training only, not used in inference)
    auxiliary_params = count_auxiliary_task_parameters(model, model_name)
    if auxiliary_params > 0:
        logger.info(f"  Total auxiliary params (excluded): {auxiliary_params:,}")
        activated_params -= auxiliary_params
    
    # Check if model uses MOE
    if model_name == 'Prism':
        training_config = config.get('training', None)
        if training_config and hasattr(training_config, 'fusion_gate_type'):
            if training_config.fusion_gate_type == 'moe':
                # MOE model: only top-k experts are activated
                num_experts = getattr(training_config, 'moe_num_experts', 4)
                top_k = getattr(training_config, 'moe_top_k', 2)
                
                # Count MOE expert parameters
                moe_expert_params = 0
                for name, module in model.named_modules():
                    # Match expert layers in MoE fusion module
                    if 'fusion_module' in name and 'expert' in name:
                        moe_expert_params += sum(p.numel() for p in module.parameters())
                
                if moe_expert_params > 0:
                    # Calculate inactive expert parameters
                    inactive_ratio = (num_experts - top_k) / num_experts
                    inactive_moe_params = int(moe_expert_params * inactive_ratio)
                    
                    logger.info(f"  MOE detected: {num_experts} experts, top-{top_k}")
                    logger.info(f"  Total MOE expert params: {moe_expert_params:,}")
                    logger.info(f"  Inactive MOE params: {inactive_moe_params:,}")
                    
                    activated_params -= inactive_moe_params
    
    return activated_params


def pad_or_truncate(sequence: List[int], max_len: int, pad_token_id: int = 0) -> List[int]:
    """Pad or truncate a sequence."""
    if len(sequence) > max_len:
        return sequence[-max_len:]
    else:
        return [pad_token_id] * (max_len - len(sequence)) + sequence


def prepare_test_samples(
    test_path: str,
    num_samples: int = 500,
    max_len: int = 20,
    seed: int = 42
) -> List[Dict]:
    """Prepare fixed test samples for all models.
    """
    df = pd.read_parquet(test_path)
    
    # Sample fixed subset
    np.random.seed(seed)
    indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
    
    samples = []
    for idx in indices:
        row = df.iloc[idx]
        history = list(row['history'])
        target = row['target']
        
        # Pad/truncate history
        history_padded = pad_or_truncate(history, max_len, 0)
        
        samples.append({
            'history': history_padded,
            'target': target
        })
    
    logger.info(f"Prepared {len(samples)} test samples (max_len={max_len})")
    return samples



def load_model(model_name: str, checkpoint_path: str, device: str) -> Tuple[nn.Module, dict, any]:
    """Load a model from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # IMPORTANT: Do NOT override vocab_size from checkpoint!
    # The checkpoint's config already has the correct vocab_size used during training.
    
    if model_name == 'TIGER':
        from src.recommender.TIGER.dataset import SemanticIDMapper
        from src.recommender.TIGER.model import create_model
        
        # Use semantic mapping path from checkpoint config
        semantic_mapping_path = config['data'].semantic_mapping_path
        
        semantic_mapper = SemanticIDMapper(
            semantic_mapping_path,
            codebook_size=config['model'].codebook_size,
            num_layers=config['model'].num_code_layers
        )
        # Don't override vocab_size - use checkpoint's value
        model = create_model(config['model'])
        
    elif model_name == 'LETTER':
        from src.recommender.LETTER.dataset import SemanticIDMapper
        from src.recommender.LETTER.model import create_model
        
        # Use semantic mapping path from checkpoint config
        semantic_mapping_path = config['data'].semantic_mapping_path
        
        semantic_mapper = SemanticIDMapper(
            semantic_mapping_path,
            codebook_size=config['model'].codebook_size,
            num_layers=config['model'].num_code_layers
        )
        # Don't override vocab_size - use checkpoint's value
        model = create_model(config['model'])
        
    elif model_name == 'ActionPiece':
        from src.recommender.ActionPiece.actionpiece_dataset import ActionPieceMapper
        from src.recommender.ActionPiece.actionpiece_model import ActionPieceModel
        
        # ActionPiece stores paths differently in config (no semantic_mapping_path)
        tokenizer_path = config['data'].tokenizer_path
        item2feat_path = config['data'].item2feat_path
        
        semantic_mapper = ActionPieceMapper(
            tokenizer_path=tokenizer_path,
            item2feat_path=item2feat_path
        )
        # ActionPiece needs vocab_size from mapper since it's not in checkpoint
        config['model'].set_vocab_size(semantic_mapper.vocab_size)
        config['model'].num_code_layers = semantic_mapper.n_categories
        model = ActionPieceModel(config['model'], semantic_mapper)
        
    elif model_name == 'Prism':
        from src.recommender.prism.dataset import SemanticIDMapper
        from src.recommender.prism.model import create_model
        
        # Use semantic mapping path from checkpoint config
        semantic_mapping_path = config['data'].semantic_mapping_path
        
        semantic_mapper = SemanticIDMapper(
            semantic_mapping_path,
            codebook_size=config['model'].codebook_size,
            num_layers=config['model'].num_code_layers
        )
        # Don't override vocab_size - use checkpoint's value
        model = create_model(config['model'], config.get('training', None))
        
    elif model_name == 'EAGER':
        from src.recommender.EAGER.dataset import DualSemanticIDMapper
        from src.recommender.EAGER.model import create_model
        
        # EAGER uses dual semantic ID mapper (behavior + semantic)
        behavior_mapping_path = config['data'].behavior_mapping_path
        semantic_mapping_path = config['data'].semantic_mapping_path
        
        semantic_mapper = DualSemanticIDMapper(
            behavior_mapping_path=behavior_mapping_path,
            semantic_mapping_path=semantic_mapping_path,
            codebook_size=config['model'].codebook_size,
            num_layers=config['model'].num_code_layers
        )
        
        # For inference/benchmark, we don't need GCT embeddings (behavior_emb_fixed, semantic_emb_fixed)
        # These are only used during training for contrastive loss computation.
        # Create model without embedding paths, then load weights with strict=False
        model = create_model(config['model'], behavior_emb_path=None, semantic_emb_path=None)
        
        # Load state dict with strict=False to skip GCT embedding weights
        # (behavior_emb_fixed, semantic_emb_fixed, behavior_projection, semantic_projection)
        # These modules are not created when embedding paths are None
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        model.eval()
        
        return model, config, semantic_mapper
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config, semantic_mapper


def benchmark_inference(
    model: nn.Module,
    model_name: str,
    samples: List[Dict],
    semantic_mapper,
    device: str,
    beam_size: int = 20,
    warmup_runs: int = 10,
    config: dict = None,
    content_embeddings: Dict[int, np.ndarray] = None,
    collab_embeddings: Dict[int, np.ndarray] = None,
    codebook_vectors: Dict[int, np.ndarray] = None
) -> Dict[str, float]:
    """Benchmark inference speed.
    """
    device_obj = torch.device(device)
    
    # Check if Prism needs multimodal inputs
    use_multimodal = False
    if model_name == 'Prism' and config:
        training_config = config.get('training', None)
        if training_config and training_config.use_multimodal_fusion:
            use_multimodal = True
            logger.info("  Using multimodal inputs for Prism")
    
    # Determine dimensions for multimodal
    content_dim = 768
    collab_dim = 64
    num_code_layers = semantic_mapper.num_layers if hasattr(semantic_mapper, 'num_layers') else 3
    latent_dim = 32
    
    if content_embeddings:
        content_dim = next(iter(content_embeddings.values())).shape[0]
    if collab_embeddings:
        collab_dim = next(iter(collab_embeddings.values())).shape[0]
    if codebook_vectors:
        sample_cb = next(iter(codebook_vectors.values()))
        num_code_layers = sample_cb.shape[0]
        latent_dim = sample_cb.shape[1]
    
    # Prepare inputs based on model type
    all_inputs = []
    all_item_ids = []  # For EAGER (uses item IDs directly)
    all_multimodal = []  # For Prism
    
    if model_name == 'ActionPiece':
        # ActionPiece uses encode_sequence
        for sample in samples:
            # Filter out padding (0s)
            history = [x for x in sample['history'] if x != 0]
            tokens = semantic_mapper.encode_sequence(history, shuffle='none')
            all_inputs.append(tokens)
        num_layers = semantic_mapper.n_categories
    elif model_name == 'EAGER':
        # EAGER uses item IDs directly (encoder has item_embedding)
        for sample in samples:
            all_item_ids.append(sample['history'])
        num_layers = semantic_mapper.num_layers
    else:
        # Standard models use get_codes
        for sample in samples:
            history_codes = []
            for item_id in sample['history']:
                codes = semantic_mapper.get_codes(item_id)
                history_codes.extend(codes)
            all_inputs.append(history_codes)
            
            # Prepare multimodal inputs for Prism
            if use_multimodal:
                history_padded = sample['history']
                content_embs = []
                collab_embs = []
                codebook_vecs = []
                
                for item_id in history_padded:
                    if content_embeddings and item_id in content_embeddings:
                        content_embs.append(content_embeddings[item_id])
                    else:
                        content_embs.append(np.zeros(content_dim, dtype=np.float32))
                    
                    if collab_embeddings and item_id in collab_embeddings:
                        collab_embs.append(collab_embeddings[item_id])
                    else:
                        collab_embs.append(np.zeros(collab_dim, dtype=np.float32))
                    
                    if codebook_vectors and item_id in codebook_vectors:
                        codebook_vecs.append(codebook_vectors[item_id])
                    else:
                        codebook_vecs.append(np.zeros((num_code_layers, latent_dim), dtype=np.float32))
                
                all_multimodal.append({
                    'content': np.array(content_embs, dtype=np.float32),
                    'collab': np.array(collab_embs, dtype=np.float32),
                    'codebook': np.array(codebook_vecs, dtype=np.float32)
                })
        
        num_layers = semantic_mapper.num_layers
    
    # Warmup
    logger.info(f"  Warming up ({warmup_runs} runs)...")
    for i in range(warmup_runs):
        if model_name == 'EAGER':
            input_ids = torch.tensor([all_item_ids[i % len(all_item_ids)]], dtype=torch.long, device=device_obj)
        else:
            input_ids = torch.tensor([all_inputs[i % len(all_inputs)]], dtype=torch.long, device=device_obj)
        attention_mask = (input_ids != 0).long()
        with torch.no_grad():
            if model_name == 'ActionPiece':
                _ = model.generate_single(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=num_layers + 1,
                    num_return_sequences=beam_size
                )
            elif model_name == 'EAGER':
                # EAGER returns 4 tensors: b_seqs, b_scores, s_seqs, s_scores
                _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=num_layers + 1
                )
            elif use_multimodal and all_multimodal:
                mm = all_multimodal[i % len(all_multimodal)]
                content_tensor = torch.tensor(mm['content'], dtype=torch.float32, device=device_obj).unsqueeze(0)
                collab_tensor = torch.tensor(mm['collab'], dtype=torch.float32, device=device_obj).unsqueeze(0)
                codebook_tensor = torch.tensor(mm['codebook'], dtype=torch.float32, device=device_obj).unsqueeze(0)
                _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=num_layers + 1,
                    content_embs=content_tensor,
                    collab_embs=collab_tensor,
                    history_codebook_vecs=codebook_tensor
                )
            else:
                _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=num_layers + 1
                )
    
    # Synchronize GPU
    if device_obj.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    logger.info(f"  Running benchmark ({len(samples)} samples)...")
    times = []
    
    # Determine which input list to iterate over
    if model_name == 'EAGER':
        input_list = all_item_ids
    else:
        input_list = all_inputs
    
    for idx, sample_input in enumerate(tqdm(input_list, desc=f"  {model_name}", leave=False)):
        input_ids = torch.tensor([sample_input], dtype=torch.long, device=device_obj)
        attention_mask = (input_ids != 0).long()
        
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            if model_name == 'ActionPiece':
                _ = model.generate_single(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=num_layers + 1,
                    num_return_sequences=beam_size
                )
            elif model_name == 'EAGER':
                # EAGER returns 4 tensors: b_seqs, b_scores, s_seqs, s_scores
                _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=num_layers + 1
                )
            elif use_multimodal and all_multimodal:
                mm = all_multimodal[idx]
                content_tensor = torch.tensor(mm['content'], dtype=torch.float32, device=device_obj).unsqueeze(0)
                collab_tensor = torch.tensor(mm['collab'], dtype=torch.float32, device=device_obj).unsqueeze(0)
                codebook_tensor = torch.tensor(mm['codebook'], dtype=torch.float32, device=device_obj).unsqueeze(0)
                _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=num_layers + 1,
                    content_embs=content_tensor,
                    collab_embs=collab_tensor,
                    history_codebook_vecs=codebook_tensor
                )
            else:
                _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=num_layers + 1
                )
        
        if device_obj.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times)
    
    return {
        'mean_ms': times.mean() * 1000,
        'std_ms': times.std() * 1000,
        'median_ms': np.median(times) * 1000,
        'p95_ms': np.percentile(times, 95) * 1000,
        'throughput': 1.0 / times.mean(),  # samples per second
    }


def plot_efficiency_results(
    results: Dict[str, Dict],
    output_path: str
):
    """Plot efficiency benchmark results in publication-quality style (single dataset).
    """
    # Call multi-dataset version with single dataset
    plot_multi_dataset_efficiency({'single': results}, output_path)


def plot_multi_dataset_efficiency(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: str
):
    """Plot efficiency benchmark results for multiple datasets.
    """
    # Configure Linux Libertine font
    import matplotlib.font_manager as fm
    import os
    
    libertine_font_dir = '/home/fangdengzhao/Fonts/libertine/opentype'
    libertine_fonts = [
        f'{libertine_font_dir}/LinLibertine_R.otf',
        f'{libertine_font_dir}/LinLibertine_RI.otf',
        f'{libertine_font_dir}/LinLibertine_RB.otf',
        f'{libertine_font_dir}/LinLibertine_RBI.otf',
    ]
    
    for font_file in libertine_fonts:
        if os.path.exists(font_file):
            fm.fontManager.addfont(font_file)
    
    # Publication-quality style - larger fonts for top-venue readability
    plt.rcParams.update({
        'font.family': 'Linux Libertine O',
        'font.weight': 'normal',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Linux Libertine O',
        'mathtext.it': 'Linux Libertine O:italic',
        'mathtext.bf': 'Linux Libertine O:bold',
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
    })
    
    # Display name mapping
    display_names = {
        'TIGER': 'TIGER',
        'LETTER': 'LETTER',
        'ActionPiece': 'ActionPiece',
        'Prism': 'PRISM',
        'EAGER': 'EAGER',
    }
    
    # Dataset display names
    dataset_info = {
        'beauty': {'display': 'Beauty'},
        'cds': {'display': 'CDs'},
        'single': {'display': ''},
    }
    
    # Refined color palette - more sophisticated and harmonious
    model_colors = {
        'TIGER': '#4C72B0',       # Steel blue
        'LETTER': '#DD8452',      # Coral orange
        'ActionPiece': '#55A868', # Sage green
        'EAGER': '#C44E52',       # Muted red
        'PRISM': '#8172B3',       # Soft purple (our method - distinctive)
    }
    
    # Define desired order: generative models only (exclude SASRec)
    desired_order = ['TIGER', 'LETTER', 'ActionPiece', 'EAGER', 'PRISM']
    
    datasets = list(all_results.keys())
    n_datasets = len(datasets)
    
    # Filter out SASRec and get all generative models across all datasets
    all_models = set()
    for dataset_results in all_results.values():
        for model_name in dataset_results.keys():
            if model_name != 'SASRec':  # Exclude SASRec
                all_models.add(model_name)
    
    # Sort models according to desired order
    model_names = [m for m in desired_order if m in all_models]
    model_names.extend([m for m in all_models if m not in model_names])
    n_models = len(model_names)
    
    display_model_names = [display_names.get(m, m) for m in model_names]
    
    # Check if single dataset mode
    is_single_dataset = n_datasets == 1 and 'single' in datasets
    
    if is_single_dataset:
        # Single dataset mode
        results = {k: v for k, v in all_results['single'].items() if k != 'SASRec'}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        
        x = np.arange(n_models)
        bar_width = 0.65
        bar_colors = [model_colors.get(display_names.get(m, m), '#888888') for m in model_names]
        
        # Subplot 1: Activated Parameters
        ax1 = axes[0]
        params = [results[m]['activated_params'] / 1e6 for m in model_names if m in results]
        
        bars1 = ax1.bar(x, params, bar_width, color=bar_colors, 
                       edgecolor='white', linewidth=1.2, zorder=3)
        
        for bar, val in zip(bars1, params):
            ax1.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14, fontweight='normal')
        
        ax1.set_ylabel('Activated Parameters (M)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_model_names, rotation=25, ha='right')
        ax1.set_ylim(0, max(params) * 1.20)
        ax1.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.6, zorder=0)
        ax1.set_axisbelow(True)
        
        # Subplot 2: Inference Latency
        ax2 = axes[1]
        latencies = [results[m]['timing']['mean_ms'] for m in model_names if m in results]
        errors = [results[m]['timing']['std_ms'] for m in model_names if m in results]
        
        bars2 = ax2.bar(x, latencies, bar_width, yerr=errors, capsize=3,
                       color=bar_colors, edgecolor='white', linewidth=1.2, 
                       error_kw={'linewidth': 1.0, 'capthick': 1.0}, zorder=3)
        
        for bar, val in zip(bars2, latencies):
            ax2.annotate(f'{val:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14, fontweight='normal')
        
        ax2.set_ylabel('Inference Latency (ms)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_model_names, rotation=25, ha='right')
        ax2.set_ylim(0, max(latencies) * 1.25)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.6, zorder=0)
        ax2.set_axisbelow(True)
        
    else:
        # Multi-dataset mode - grouped bar chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.arange(n_models)
        bar_width = 0.38
        
        # Dataset visual styles - solid vs hatched with refined aesthetics
        dataset_styles = [
            {'alpha': 1.0, 'hatch': '', 'edgecolor': 'white'},
            {'alpha': 0.85, 'hatch': '///', 'edgecolor': '#333333'},
        ]
        
        # Subplot 1: Activated Parameters (grouped by dataset)
        ax1 = axes[0]
        
        for i, dataset in enumerate(datasets):
            results = {k: v for k, v in all_results[dataset].items() if k != 'SASRec'}
            params = []
            for m in model_names:
                if m in results:
                    params.append(results[m]['activated_params'] / 1e6)
                else:
                    params.append(0)
            
            offset = (i - (n_datasets - 1) / 2) * bar_width
            bar_colors = [model_colors.get(display_names.get(m, m), '#888888') for m in model_names]
            
            bars = ax1.bar(x + offset, params, bar_width, 
                          color=bar_colors, 
                          edgecolor=dataset_styles[i]['edgecolor'], 
                          linewidth=1.2, 
                          alpha=dataset_styles[i]['alpha'],
                          hatch=dataset_styles[i]['hatch'],
                          label=dataset_info[dataset]['display'],
                          zorder=3)
        
        ax1.set_ylabel('Activated Parameters (M)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_model_names, rotation=25, ha='right')
        
        # Get max params for y-axis
        max_params = 0
        for dataset in datasets:
            for m in model_names:
                if m in all_results[dataset] and m != 'SASRec':
                    max_params = max(max_params, all_results[dataset][m]['activated_params'] / 1e6)
        ax1.set_ylim(0, max_params * 1.18)
        ax1.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.6, zorder=0)
        ax1.set_axisbelow(True)
        
        # Subplot 2: Inference Latency (grouped by dataset)
        ax2 = axes[1]
        
        for i, dataset in enumerate(datasets):
            results = {k: v for k, v in all_results[dataset].items() if k != 'SASRec'}
            latencies = []
            errors = []
            for m in model_names:
                if m in results:
                    latencies.append(results[m]['timing']['mean_ms'])
                    errors.append(results[m]['timing']['std_ms'])
                else:
                    latencies.append(0)
                    errors.append(0)
            
            offset = (i - (n_datasets - 1) / 2) * bar_width
            bar_colors = [model_colors.get(display_names.get(m, m), '#888888') for m in model_names]
            
            bars = ax2.bar(x + offset, latencies, bar_width,
                          yerr=errors, capsize=3,
                          color=bar_colors, 
                          edgecolor=dataset_styles[i]['edgecolor'], 
                          linewidth=1.2,
                          alpha=dataset_styles[i]['alpha'],
                          hatch=dataset_styles[i]['hatch'],
                          error_kw={'linewidth': 0.8, 'capthick': 0.8},
                          label=dataset_info[dataset]['display'],
                          zorder=3)
        
        ax2.set_ylabel('Inference Latency (ms)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_model_names, rotation=25, ha='right')
        
        # Get max latency for y-axis
        max_latency = 0
        for dataset in datasets:
            for m in model_names:
                if m in all_results[dataset] and m != 'SASRec':
                    max_latency = max(max_latency, all_results[dataset][m]['timing']['mean_ms'])
        ax2.set_ylim(0, max_latency * 1.22)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.6, zorder=0)
        ax2.set_axisbelow(True)
        
        # Add legend for datasets (using custom patches) - refined style
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#666666', edgecolor=dataset_styles[i]['edgecolor'], 
                  alpha=dataset_styles[i]['alpha'], hatch=dataset_styles[i]['hatch'], 
                  linewidth=1.2, label=dataset_info[datasets[i]]['display'])
            for i in range(n_datasets)
        ]
        ax2.legend(handles=legend_elements, loc='upper right', frameon=True, 
                  framealpha=0.95, fontsize=13, edgecolor='#cccccc',
                  fancybox=False)
    
    plt.tight_layout(pad=1.2)
    
    # Save
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', bbox_inches='tight', 
                pad_inches=0.05, dpi=300)
    logger.info(f"Figure saved to {output_path}")
    plt.close()



def setup_logging(output_dir: Path, log_level: str = "INFO"):
    """Setup logging configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "efficiency_benchmark.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Efficiency benchmark for generative recommender models"
    )
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of test samples')
    parser.add_argument('--beam_size', type=int, default=20,
                       help='Beam size for generation')
    parser.add_argument('--output_dir', type=str, 
                       default='scripts/output/efficiency_benchmark',
                       help='Output directory')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['TIGER', 'LETTER', 'ActionPiece', 'EAGER', 'Prism', 'SASRec'],
                       help='Models to benchmark')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['beauty'],
                       choices=['beauty', 'cds'],
                       help='Datasets to benchmark (beauty, cds)')
    parser.add_argument('--plot_only', action='store_true',
                       help='Skip benchmarking, only plot from existing results file')
    parser.add_argument('--only_AP_CDs', action='store_true',
                       help='Only test ActionPiece on CDs dataset and update JSON')
    
    return parser.parse_args()


def benchmark_dataset(
    dataset_name: str,
    models: List[str],
    num_samples: int,
    beam_size: int,
    device: str
) -> Dict[str, Dict]:
    """Benchmark all models on a single dataset.
    
    Args:
        dataset_name: Name of dataset (beauty, cds)
        models: List of model names to benchmark
        num_samples: Number of test samples
        beam_size: Beam size for generation
        device: Device to use
        
    Returns:
        Dict mapping model_name to results
    """
    config = DATASET_CONFIGS[dataset_name]
    test_data_path = config['test_data']
    checkpoints = config['checkpoints']
    
    logger.info(f"\n{'#' * 60}")
    logger.info(f"# Dataset: {config['display_name']} (catalog: {config['catalog_size']:,} items)")
    logger.info(f"{'#' * 60}")
    
    # Prepare test samples
    logger.info(f"\nPreparing test samples from {test_data_path}...")
    samples = prepare_test_samples(
        test_data_path,
        num_samples=num_samples,
        max_len=20,
        seed=42
    )
    
    results = {}
    
    for model_name in models:
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Benchmarking {model_name} on {dataset_name}")
        logger.info(f"{'=' * 40}")
        
        # Special handling for SASRec (simulated discriminative model)
        if model_name == 'SASRec':
            try:
                from src.recommender.prism.sasrec_simulator import create_sasrec_simulator
                
                # Get catalog size from test data
                df = pd.read_parquet(test_data_path)
                all_items = set()
                for history in df['history']:
                    all_items.update(history)
                all_items.update(df['target'].tolist())
                num_items = max(all_items)
                
                logger.info(f"Creating SASRec simulator (catalog size: {num_items:,} items)")
                
                # Create simulator
                sasrec = create_sasrec_simulator(
                    num_items=num_items,
                    hidden_dim=64,
                    num_layers=2,
                    num_heads=2,
                    device=device
                )
                
                # Count parameters
                param_counts = sasrec.count_parameters()
                activated_params = param_counts['activated']
                
                logger.info(f"Parameters:")
                logger.info(f"  Total: {param_counts['total']:,}")
                logger.info(f"  Embedding: {param_counts['embedding']:,}")
                logger.info(f"  Activated: {activated_params:,}")
                
                # Benchmark inference
                timing = sasrec.benchmark(
                    samples, top_k=beam_size, warmup_runs=10
                )
                
                logger.info(f"Timing:")
                logger.info(f"  Mean: {timing['mean_ms']:.2f} ms")
                logger.info(f"  Encoding: {timing['encoding_mean_ms']:.2f} ms")
                logger.info(f"  ANN search: {timing['ann_mean_ms']:.2f} ms ({timing['ann_percentage']:.1f}%)")
                logger.info(f"  ⚠️  Bottleneck: ANN search over {num_items:,} items")
                
                results[model_name] = {
                    'total_params': param_counts['total'],
                    'activated_params': activated_params,
                    'embedding_params': param_counts['embedding'],
                    'timing': timing,
                    'catalog_size': num_items
                }
                
                del sasrec
                torch.cuda.empty_cache() if device.startswith('cuda') else None
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Handle generative models
        else:
            checkpoint_path = checkpoints.get(model_name)
            if not checkpoint_path or not Path(checkpoint_path).exists():
                logger.warning(f"Checkpoint not found for {model_name} on {dataset_name}, skipping...")
                continue
            
            try:
                # Load model
                logger.info(f"Loading model from {checkpoint_path}")
                model, model_config, semantic_mapper = load_model(
                    model_name, checkpoint_path, device
                )
                
                # Count parameters
                param_counts = count_parameters(model)
                activated_params = count_activated_parameters(model, model_name, model_config)
                
                logger.info(f"Parameters:")
                logger.info(f"  Total: {param_counts['total']:,}")
                logger.info(f"  Activated: {activated_params:,}")
                
                # Load multimodal embeddings for Prism
                content_embeddings = None
                collab_embeddings = None
                codebook_vectors = None
                
                if model_name == 'Prism':
                    training_config = model_config.get('training', None)
                    if training_config and training_config.use_multimodal_fusion:
                        logger.info("Loading multimodal embeddings for Prism...")
                        
                        from src.recommender.prism.dataset import (
                            load_content_embeddings,
                            load_collab_embeddings,
                            load_codebook_mappings
                        )
                        
                        # Determine data directory based on dataset
                        if dataset_name == 'beauty':
                            data_dir = 'dataset/Amazon-Beauty/processed/beauty-prism-sentenceT5base/Beauty'
                        elif dataset_name == 'cds':
                            data_dir = 'dataset/Amazon-CDs/processed/cds-prism-sentenceT5base/CDs'
                        else:
                            data_dir = None
                        
                        if data_dir:
                            content_embeddings = load_content_embeddings(data_dir)
                            logger.info(f"  Loaded content embeddings for {len(content_embeddings)} items")
                            
                            collab_path = Path(data_dir) / 'lightgcn' / 'item_embeddings_collab.npy'
                            if collab_path.exists():
                                collab_embeddings = load_collab_embeddings(str(collab_path))
                                logger.info(f"  Loaded collab embeddings for {len(collab_embeddings)} items")
                            
                            tokenizer_dir = Path(model_config['data'].semantic_mapping_path).parent
                            codebook_vectors, _ = load_codebook_mappings(str(tokenizer_dir))
                            logger.info(f"  Loaded codebook vectors for {len(codebook_vectors)} items")
                
                # Benchmark inference
                timing = benchmark_inference(
                    model, model_name, samples, semantic_mapper,
                    device, beam_size,
                    config=model_config,
                    content_embeddings=content_embeddings,
                    collab_embeddings=collab_embeddings,
                    codebook_vectors=codebook_vectors
                )
                
                logger.info(f"Timing:")
                logger.info(f"  Mean: {timing['mean_ms']:.2f} ms")
                logger.info(f"  Throughput: {timing['throughput']:.2f} samples/s")
                
                results[model_name] = {
                    'total_params': param_counts['total'],
                    'activated_params': activated_params,
                    'embedding_params': param_counts['embedding'],
                    'timing': timing
                }
                
                del model
                torch.cuda.empty_cache() if device.startswith('cuda') else None
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return results


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    setup_logging(output_dir)
    
    logger.info("=" * 60)
    logger.info("EFFICIENCY BENCHMARK FOR RECOMMENDER MODELS")
    logger.info("=" * 60)
    
    # Plot-only mode: load existing results and regenerate plots
    if args.plot_only:
        results_path = output_dir / 'efficiency_results.json'
        if not results_path.exists():
            logger.error(f"Results file not found: {results_path}")
            logger.error("Run without --plot_only first to generate results.")
            return
        
        logger.info(f"Plot-only mode: loading results from {results_path}")
        with open(results_path, 'r') as f:
            all_results = json.load(f)
        
        logger.info(f"Loaded results for datasets: {list(all_results.keys())}")
        
        # Plot results
        plot_path = str(output_dir / 'efficiency_comparison.pdf')
        plot_multi_dataset_efficiency(all_results, plot_path)
        
        logger.info("\n" + "=" * 60)
        logger.info("PLOT REGENERATED")
        logger.info("=" * 60)
        return
    
    # Only ActionPiece on CDs mode: update only that specific result
    if args.only_AP_CDs:
        results_path = output_dir / 'efficiency_results.json'
        
        # Load existing results
        if results_path.exists():
            logger.info(f"Loading existing results from {results_path}")
            with open(results_path, 'r') as f:
                all_results = json.load(f)
        else:
            logger.warning(f"No existing results found, creating new file")
            all_results = {}
        
        logger.info("=" * 60)
        logger.info("ONLY TESTING: ActionPiece on CDs dataset")
        logger.info("=" * 60)
        
        # Benchmark only ActionPiece on CDs
        results = benchmark_dataset(
            dataset_name='cds',
            models=['ActionPiece'],
            num_samples=args.num_samples,
            beam_size=args.beam_size,
            device=args.device
        )
        
        # Update only the ActionPiece result in CDs dataset
        if 'cds' not in all_results:
            all_results['cds'] = {}
        
        all_results['cds']['ActionPiece'] = {
            'total_params': results['ActionPiece']['total_params'],
            'activated_params': results['ActionPiece']['activated_params'],
            'embedding_params': results['ActionPiece']['embedding_params'],
            'timing': results['ActionPiece']['timing']
        }
        
        # Save updated results
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\n✓ Updated ActionPiece results for CDs dataset in {results_path}")
        logger.info("=" * 60)
        logger.info("UPDATE COMPLETED")
        logger.info("=" * 60)
        return
    
    # Normal benchmark mode
    logger.info(f"Device: {args.device}")
    logger.info(f"Num samples: {args.num_samples}")
    logger.info(f"Beam size: {args.beam_size}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Datasets: {args.datasets}")
    
    # Benchmark each dataset
    all_results = {}
    
    for dataset_name in args.datasets:
        results = benchmark_dataset(
            dataset_name=dataset_name,
            models=args.models,
            num_samples=args.num_samples,
            beam_size=args.beam_size,
            device=args.device
        )
        all_results[dataset_name] = results
    
    # Save results
    results_path = output_dir / 'efficiency_results.json'
    
    # Convert to serializable format
    serializable_results = {}
    for dataset_name, results in all_results.items():
        serializable_results[dataset_name] = {}
        for model_name, data in results.items():
            serializable_results[dataset_name][model_name] = {
                'total_params': data['total_params'],
                'activated_params': data['activated_params'],
                'embedding_params': data['embedding_params'],
                'timing': data['timing']
            }
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")
    
    # Plot results
    if len(all_results) > 0:
        plot_path = str(output_dir / 'efficiency_comparison.pdf')
        plot_multi_dataset_efficiency(all_results, plot_path)
    
    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    for dataset_name, results in all_results.items():
        logger.info(f"\n--- {dataset_name.upper()} ---")
        logger.info(f"{'Model':<15} {'Activated Params':<18} {'Latency (ms)':<15} {'Throughput':<12}")
        logger.info("-" * 60)
        for model_name, data in results.items():
            params_m = data['activated_params'] / 1e6
            latency = data['timing']['mean_ms']
            throughput = data['timing']['throughput']
            logger.info(f"{model_name:<15} {params_m:>14.2f}M   {latency:>11.2f}    {throughput:>8.2f}/s")
    
    # Highlight SASRec scaling if multiple datasets
    if len(args.datasets) > 1 and 'SASRec' in args.models:
        logger.info("\n" + "=" * 70)
        logger.info("KEY INSIGHT: SASRec (Discriminative) Scaling")
        logger.info("=" * 70)
        
        sasrec_latencies = []
        catalog_sizes = []
        for dataset_name in args.datasets:
            if 'SASRec' in all_results.get(dataset_name, {}):
                sasrec_latencies.append(all_results[dataset_name]['SASRec']['timing']['mean_ms'])
                catalog_sizes.append(all_results[dataset_name]['SASRec'].get('catalog_size', 
                                     DATASET_CONFIGS[dataset_name]['catalog_size']))
        
        if len(sasrec_latencies) == 2:
            speedup = sasrec_latencies[1] / sasrec_latencies[0]
            catalog_ratio = catalog_sizes[1] / catalog_sizes[0]
            logger.info(f"  Catalog size: {catalog_sizes[0]:,} → {catalog_sizes[1]:,} ({catalog_ratio:.1f}× larger)")
            logger.info(f"  SASRec latency: {sasrec_latencies[0]:.2f}ms → {sasrec_latencies[1]:.2f}ms ({speedup:.1f}× slower)")
            logger.info(f"  ⚠️  Discriminative models scale with catalog size!")
            logger.info(f"  ✓  Generative models maintain constant latency regardless of catalog size")
    
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
