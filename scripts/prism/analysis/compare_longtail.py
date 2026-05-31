#!/usr/bin/env python3
"""
Long-tail Comparison: TIGER vs PRISM

Evaluates both models on multiple datasets with unified popularity grouping,
then generates publication-quality figures for comparison.

Supports: Beauty, CDs datasets
Output: Side-by-side comparison (Beauty | CDs), each showing TIGER vs PRISM
"""

import argparse
import json
import logging
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dataset configurations
DATASET_CONFIGS = {
    'beauty': {
        'unified_item_emb_path': "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty/item_emb.parquet",
        'unified_test_path': "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty/test.parquet",
        'tiger': {
            'sequence_data_path': "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty",
            'semantic_mapping_path': "scripts/output/tiger_tokenizer/beauty/3-256-32-ema-only-5-core-items/semantic_id_mappings.json",
        },
        'prism': {
            'sequence_data_path': "dataset/Amazon-Beauty/processed/beauty-prism-sentenceT5base/Beauty",
            'semantic_mapping_path': "scripts/output/prism_tokenizer/beauty/3-256-32-ema-only-5-core-items/semantic_id_mappings.json",
        },
        'actionpiece': {
            'sequence_data_path': "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty",
            'tokenizer_path': "scripts/output/actionpiece_tokenizer/beauty/actionpiece.json",
            'item2feat_path': "scripts/output/actionpiece_tokenizer/beauty/item2feat.json",
        },
        'display_name': 'Beauty'
    },
    'cds': {
        'unified_item_emb_path': "dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs/item_emb.parquet",
        'unified_test_path': "dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs/test.parquet",
        'tiger': {
            'sequence_data_path': "dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs",
            'semantic_mapping_path': "scripts/output/tiger_tokenizer/cds/3-256-32-ema-only-5-core-items/semantic_id_mappings.json",
        },
        'prism': {
            'sequence_data_path': "dataset/Amazon-CDs/processed/cds-prism-sentenceT5base/CDs",
            'semantic_mapping_path': "scripts/output/prism_tokenizer/cds/3-256-32-ema-only-5-core-items/semantic_id_mappings.json",
        },
        'actionpiece': {
            'sequence_data_path': "dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs",
            'tokenizer_path': "scripts/output/actionpiece_tokenizer/cds/actionpiece.json",
            'item2feat_path': "scripts/output/actionpiece_tokenizer/cds/item2feat.json",
        },
        'display_name': 'CDs'
    }
}


class PopularityGrouper:
    """Groups items by popularity score using rank-based assignment."""
    
    def __init__(self, item_emb_path: str, num_groups: int = 3):
        self.num_groups = num_groups
        
        logger.info(f"Loading item popularity from {item_emb_path}")
        df = pd.read_parquet(item_emb_path)
        
        if 'popularity_score' not in df.columns:
            raise ValueError("popularity_score column not found")
        
        self.item_popularity = dict(zip(df['ItemID'], df['popularity_score']))
        self._compute_groups_by_rank(df)
    
    def _compute_groups_by_rank(self, df: pd.DataFrame):
        df_sorted = df.sort_values('popularity_score', ascending=False).reset_index(drop=True)
        n_items = len(df_sorted)
        items_per_group = n_items // self.num_groups
        
        self.item_to_group = {}
        for idx, row in df_sorted.iterrows():
            group = min(idx // items_per_group, self.num_groups - 1)
            self.item_to_group[row['ItemID']] = group
        
        logger.info(f"Rank-based grouping: ~{items_per_group} items per group")
    
    def get_group(self, item_id: int) -> int:
        return self.item_to_group.get(item_id, self.num_groups - 1)
    
    def get_group_names(self) -> List[str]:
        if self.num_groups == 3:
            return ['Popular', 'Medium', 'Long-tail']
        elif self.num_groups == 5:
            return ['Popular', 'Mid-High', 'Medium', 'Mid-Low', 'Long-tail']
        return [f'Group {i+1}' for i in range(self.num_groups)]


def load_test_data(test_path: str) -> List[Dict]:
    """Load test data from parquet file."""
    df = pd.read_parquet(test_path)
    samples = []
    for _, row in df.iterrows():
        samples.append({
            'history': list(row['history']),
            'target': row['target']
        })
    logger.info(f"Loaded {len(samples)} test samples from {test_path}")
    return samples


def pad_or_truncate(sequence: List[int], max_len: int, pad_token_id: int = 0) -> List[int]:
    """Pad or truncate sequence to max_len."""
    if len(sequence) > max_len:
        return sequence[-max_len:]
    return [pad_token_id] * (max_len - len(sequence)) + sequence


def load_model_and_mapper(model_type: str, checkpoint_path: str, device: str):
    """Load model and semantic mapper."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    if model_type == 'tiger':
        from src.recommender.TIGER.dataset import SemanticIDMapper
        from src.recommender.TIGER.model import create_model
        
        semantic_mapper = SemanticIDMapper(
            config['data'].semantic_mapping_path,
            codebook_size=config['model'].codebook_size,
            num_layers=config['model'].num_code_layers
        )
        model = create_model(config['model'])
    elif model_type == 'prism':
        from src.recommender.prism.dataset import SemanticIDMapper
        from src.recommender.prism.model import create_model
        
        semantic_mapper = SemanticIDMapper(
            config['data'].semantic_mapping_path,
            codebook_size=config['model'].codebook_size,
            num_layers=config['model'].num_code_layers
        )
        model = create_model(config['model'], config.get('training', None))
    elif model_type == 'actionpiece':
        from src.recommender.ActionPiece.actionpiece_dataset import ActionPieceMapper
        from src.recommender.ActionPiece.actionpiece_model import create_actionpiece_model
        
        # ActionPiece uses a different mapper
        semantic_mapper = ActionPieceMapper(
            config['data'].tokenizer_path,
            config['data'].item2feat_path
        )
        model = create_actionpiece_model(config['model'], semantic_mapper)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded {model_type.upper()} from epoch {checkpoint.get('epoch', '?')}")
    
    return model, semantic_mapper, config


def load_prism_multimodal_data(config) -> Tuple[Dict, Dict, Dict]:
    """Load multimodal data for PRISM model."""
    from src.recommender.prism.dataset import (
        load_content_embeddings, load_collab_embeddings, load_codebook_mappings
    )
    
    data_config = config['data']
    semantic_mapping_dir = Path(data_config.semantic_mapping_path).parent
    
    codebook_vectors, _ = load_codebook_mappings(str(semantic_mapping_dir))
    logger.info(f"Loaded codebook vectors for {len(codebook_vectors)} items")
    
    content_embeddings = load_content_embeddings(data_config.sequence_data_path)
    logger.info(f"Loaded content embeddings for {len(content_embeddings)} items")
    
    collab_path = getattr(data_config, 'collab_embedding_path', None)
    if collab_path:
        collab_embeddings = load_collab_embeddings(collab_path)
    else:
        default_collab = Path(data_config.sequence_data_path) / 'lightgcn' / 'item_embeddings_collab.npy'
        if default_collab.exists():
            collab_embeddings = load_collab_embeddings(str(default_collab))
        else:
            collab_embeddings = {}
    logger.info(f"Loaded collab embeddings for {len(collab_embeddings)} items")
    
    return content_embeddings, collab_embeddings, codebook_vectors


def evaluate_model(
    model,
    model_type: str,
    semantic_mapper,
    config: dict,
    test_samples: List[Dict],
    popularity_grouper: PopularityGrouper,
    device: str,
    beam_size: int = 30,
    topk_list: List[int] = None,
    content_embeddings: Dict = None,
    collab_embeddings: Dict = None,
    codebook_vectors: Dict = None,
    max_samples_per_group: int = None,  # Limit samples per group for quick testing
) -> Dict:
    """Evaluate model on test samples."""
    topk_list = topk_list or [5, 10, 20]
    model.to(device)
    model.eval()
    
    model_config = config['model']
    data_config = config['data']
    training_config = config.get('training', None)
    max_len = data_config.max_seq_length
    
    # Handle different model types
    if model_type == 'actionpiece':
        num_code_layers = model_config.n_categories
    else:
        num_code_layers = model_config.num_code_layers
    
    use_multimodal = False
    trie_logits_processor = None
    
    if model_type == 'prism' and training_config:
        use_multimodal = getattr(training_config, 'use_multimodal_fusion', False)
        use_trie = getattr(training_config, 'use_trie_constraints', False)
        
        if use_trie:
            from src.recommender.prism.trie_constrained_decoder import (
                SemanticIDTrie, TrieConstrainedLogitsProcessor
            )
            trie = SemanticIDTrie.from_semantic_mapper(semantic_mapper)
            trie_logits_processor = TrieConstrainedLogitsProcessor(
                trie=trie,
                pad_token_id=model_config.pad_token_id,
                eos_token_id=model_config.eos_token_id,
                num_beams=beam_size
            )
            logger.info("Trie-constrained decoding enabled")
        
        if use_multimodal:
            logger.info("Multimodal fusion enabled")
    
    content_dim = 768
    collab_dim = 64
    latent_dim = 32
    
    if content_embeddings:
        sample_emb = next(iter(content_embeddings.values()))
        content_dim = sample_emb.shape[0]
    if collab_embeddings:
        sample_emb = next(iter(collab_embeddings.values()))
        collab_dim = sample_emb.shape[0]
    
    num_groups = popularity_grouper.num_groups
    group_metrics = {
        g: {f'Recall@{k}': [] for k in topk_list} | {f'NDCG@{k}': [] for k in topk_list}
        for g in range(num_groups)
    }
    group_counts = [0] * num_groups
    group_sample_counts = [0] * num_groups  # Track samples evaluated per group
    
    logger.info(f"Evaluating {len(test_samples)} samples...")
    if max_samples_per_group:
        logger.info(f"  (Limited to {max_samples_per_group} samples per group)")
    
    with torch.no_grad():
        for sample in tqdm(test_samples, desc=f"Evaluating {model_type.upper()}"):
            history = sample['history']
            target_item = sample['target']
            
            group = popularity_grouper.get_group(target_item)
            
            # Skip if we've reached the limit for this group
            if max_samples_per_group and group_sample_counts[group] >= max_samples_per_group:
                continue
            
            group_counts[group] += 1
            group_sample_counts[group] += 1
            
            history_padded = pad_or_truncate(history, max_len, 0)
            
            # Get codes based on model type
            if model_type == 'actionpiece':
                # ActionPiece: check if ensemble is needed
                if model_config.n_inference_ensemble > 1:
                    # Generate multiple SPR augmentations for ensemble
                    history_codes_list = []
                    max_code_len = 0
                    for _ in range(model_config.n_inference_ensemble):
                        # Each call with shuffle='feature' gives different SPR augmentation
                        codes = semantic_mapper.encode_sequence(history_padded, shuffle='feature')
                        # Add BOS and EOS tokens (like training)
                        input_tokens = [semantic_mapper.bos_token] + codes + [semantic_mapper.eos_token]
                        history_codes_list.append(input_tokens)
                        max_code_len = max(max_code_len, len(input_tokens))
                    
                    # Pad all sequences to the same length
                    padded_codes_list = []
                    for codes in history_codes_list:
                        if len(codes) < max_code_len:
                            codes = codes + [0] * (max_code_len - len(codes))
                        padded_codes_list.append(codes)
                    
                    # Stack all augmentations
                    input_ids = torch.tensor(padded_codes_list, dtype=torch.long, device=device)
                    attention_mask = (input_ids != 0).long()
                else:
                    # Single generation without ensemble
                    history_codes = semantic_mapper.encode_sequence(history_padded, shuffle='none')
                    # Add BOS and EOS tokens (like training)
                    input_tokens = [semantic_mapper.bos_token] + history_codes + [semantic_mapper.eos_token]
                    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=device)
                    attention_mask = (input_ids != 0).long()
                
                # Get target codes (raw state = token indices)
                target_codes = semantic_mapper.get_raw_state(target_item)
                if target_codes is None:
                    # Skip this sample if target not in vocabulary
                    continue
                target_codes = target_codes.tolist()
            else:
                history_codes = []
                for item_id in history_padded:
                    codes = semantic_mapper.get_codes(item_id)
                    history_codes.extend(codes)
                target_codes = semantic_mapper.get_codes(target_item)
                
                input_ids = torch.tensor([history_codes], dtype=torch.long, device=device)
                attention_mask = (input_ids != 0).long()
            
            content_embs = None
            collab_embs = None
            history_codebook_vecs = None
            
            if use_multimodal and content_embeddings and collab_embeddings and codebook_vectors:
                hist_content = []
                hist_collab = []
                hist_codebook = []
                
                for item_id in history_padded:
                    if item_id in content_embeddings:
                        hist_content.append(content_embeddings[item_id])
                    else:
                        hist_content.append(np.zeros(content_dim, dtype=np.float32))
                    
                    if item_id in collab_embeddings:
                        hist_collab.append(collab_embeddings[item_id])
                    else:
                        hist_collab.append(np.zeros(collab_dim, dtype=np.float32))
                    
                    if item_id in codebook_vectors:
                        hist_codebook.append(codebook_vectors[item_id])
                    else:
                        hist_codebook.append(np.zeros((num_code_layers, latent_dim), dtype=np.float32))
                
                content_embs = torch.tensor(np.array(hist_content), dtype=torch.float32, device=device).unsqueeze(0)
                collab_embs = torch.tensor(np.array(hist_collab), dtype=torch.float32, device=device).unsqueeze(0)
                history_codebook_vecs = torch.tensor(np.array(hist_codebook), dtype=torch.float32, device=device).unsqueeze(0)
            
            max_gen_length = num_code_layers + 1
            
            # Generate predictions based on model type
            if model_type == 'actionpiece':
                # ActionPiece: use ensemble generation if n_inference_ensemble > 1
                if model_config.n_inference_ensemble > 1:
                    # We already prepared n_ensemble different SPR augmentations
                    # input_ids shape: (n_ensemble, seq_len)
                    # Generate for all augmentations at once using batch processing (like training)
                    n_ensemble = input_ids.shape[0]
                    
                    # Generate all at once - this is how training does it
                    outputs = model.generate_single(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_beams=beam_size,
                        max_length=max_gen_length + 1,  # +1 for potential EOS
                        num_return_sequences=beam_size,
                        early_stopping=False
                    )
                    
                    # Remove decoder start token
                    outputs = outputs[:, 1:]
                    
                    # Decode outputs to feature states (like _decode_outputs in trainer)
                    decoded_outputs = []
                    for output in outputs.cpu().tolist():
                        # Remove EOS if present
                        if semantic_mapper.eos_token in output:
                            idx = output.index(semantic_mapper.eos_token)
                            output = output[:idx]
                        output = output[:semantic_mapper.n_categories]
                        
                        # Decode to feature state
                        decoded = semantic_mapper.decode_tokens(output)
                        if decoded is None:
                            decoded_outputs.append([-1] * semantic_mapper.n_categories)
                        else:
                            # Convert (category, feature) tuples to token indices
                            token_indices = [-1] * semantic_mapper.n_categories
                            for cat_idx, feat_idx in decoded:
                                if 0 <= cat_idx < semantic_mapper.n_categories:
                                    token_idx = semantic_mapper.actionpiece.rank.get((cat_idx, feat_idx), -1)
                                    token_indices[cat_idx] = token_idx
                            decoded_outputs.append(token_indices)
                    
                    # Reshape: (n_ensemble * beam_size, n_categories) -> (n_ensemble, beam_size, n_categories)
                    decoded = torch.tensor(decoded_outputs, dtype=torch.long)
                    decoded = decoded.view(n_ensemble, beam_size, semantic_mapper.n_categories)
                    
                    # Aggregate using nDCG weighting (like _aggregate_ensemble in trainer)
                    import collections
                    pred2score = collections.defaultdict(float)
                    
                    for ens_idx in range(n_ensemble):
                        for rank_idx in range(beam_size):
                            pred = tuple(decoded[ens_idx, rank_idx].tolist())
                            if pred[0] != -1:  # Valid prediction
                                pred2score[pred] += 1 / np.log2(rank_idx + 2)
                    
                    # Sort by aggregated score
                    sorted_preds = sorted(pred2score.items(), key=lambda x: x[1], reverse=True)
                    
                    # Take top beam_size predictions
                    final_preds = []
                    for pred_tuple, _ in sorted_preds[:beam_size]:
                        final_preds.append(list(pred_tuple))
                    
                    # Pad if needed
                    while len(final_preds) < beam_size:
                        final_preds.append([-1] * semantic_mapper.n_categories)
                    
                    preds = torch.tensor(final_preds[:beam_size], dtype=torch.long)
                else:
                    preds = model.generate_single(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_beams=beam_size,
                        max_length=max_gen_length,
                        num_return_sequences=beam_size
                    )
            elif model_type == 'prism' and use_multimodal:
                preds = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=max_gen_length,
                    content_embs=content_embs,
                    collab_embs=collab_embs,
                    history_codebook_vecs=history_codebook_vecs,
                    logits_processor=trie_logits_processor
                )
            elif model_type == 'prism':
                preds = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=max_gen_length,
                    logits_processor=trie_logits_processor
                )
            else:
                preds = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=beam_size,
                    max_length=max_gen_length
                )
            
            # Process predictions - unified handling for ActionPiece
            if model_type == 'actionpiece':
                # For ensemble mode, preds is already decoded to token indices
                # For non-ensemble, need to decode the generated tokens
                if model_config.n_inference_ensemble <= 1:
                    preds = preds[:, 1:].cpu()  # Remove decoder start token
                    
                    # Decode outputs to token indices (like ensemble mode)
                    decoded_preds = []
                    for output in preds.tolist():
                        # Remove EOS if present
                        if semantic_mapper.eos_token in output:
                            idx = output.index(semantic_mapper.eos_token)
                            output = output[:idx]
                        output = output[:semantic_mapper.n_categories]
                        
                        # Decode to feature state
                        decoded = semantic_mapper.decode_tokens(output)
                        if decoded is None:
                            decoded_preds.append([-1] * semantic_mapper.n_categories)
                        else:
                            # Convert (category, feature) tuples to token indices
                            token_indices = [-1] * semantic_mapper.n_categories
                            for cat_idx, feat_idx in decoded:
                                if 0 <= cat_idx < semantic_mapper.n_categories:
                                    token_idx = semantic_mapper.actionpiece.rank.get((cat_idx, feat_idx), -1)
                                    token_indices[cat_idx] = token_idx
                            decoded_preds.append(token_indices)
                    preds = decoded_preds
                
                pos_index = torch.zeros(len(preds) if isinstance(preds, list) else preds.shape[0], dtype=torch.bool)
                for j in range(len(preds) if isinstance(preds, list) else preds.shape[0]):
                    pred_tokens = preds[j].tolist() if isinstance(preds[j], torch.Tensor) else preds[j]
                    
                    # Compare directly with target_codes (both are token indices)
                    if len(pred_tokens) == len(target_codes) and pred_tokens == target_codes:
                        pos_index[j] = True
                        break
            else:
                # TIGER/PRISM: standard processing
                preds = preds[:, 1:].cpu()
                pos_index = torch.zeros(preds.shape[0], dtype=torch.bool)
                for j in range(preds.shape[0]):
                    if preds[j].tolist() == target_codes:
                        pos_index[j] = True
                        break
            
            for k in topk_list:
                recall = pos_index[:k].sum().float().item()
                ranks = torch.arange(1, len(pos_index) + 1, dtype=torch.float32)
                dcg = torch.where(pos_index, 1.0 / torch.log2(ranks + 1), torch.tensor(0.0))
                ndcg = dcg[:k].sum().item()
                
                group_metrics[group][f'Recall@{k}'].append(recall)
                group_metrics[group][f'NDCG@{k}'].append(ndcg)
    
    results = {'per_group': {}, 'overall': {}, 'group_counts': group_counts}
    group_names = popularity_grouper.get_group_names()
    
    overall_metrics = {f'Recall@{k}': [] for k in topk_list} | {f'NDCG@{k}': [] for k in topk_list}
    
    for g in range(num_groups):
        results['per_group'][group_names[g]] = {}
        for metric_name, values in group_metrics[g].items():
            if values:
                avg_value = np.mean(values)
                results['per_group'][group_names[g]][metric_name] = avg_value
                overall_metrics[metric_name].extend(values)
            else:
                results['per_group'][group_names[g]][metric_name] = 0.0
    
    for metric_name, values in overall_metrics.items():
        results['overall'][metric_name] = np.mean(values) if values else 0.0
    
    return results


def evaluate_dataset(
    dataset_name: str,
    tiger_checkpoint: str,
    prism_checkpoint: str,
    actionpiece_checkpoint: str,
    device: str,
    num_groups: int,
    beam_size: int,
    max_samples_per_group: int = None
) -> Tuple[Dict, Dict, Dict]:
    """Evaluate all three models on a single dataset."""
    config = DATASET_CONFIGS[dataset_name]
    
    # Load test data
    test_samples = load_test_data(config['unified_test_path'])
    
    # Create popularity grouper
    popularity_grouper = PopularityGrouper(config['unified_item_emb_path'], num_groups=num_groups)
    
    # Evaluate TIGER
    logger.info(f"\nEvaluating TIGER on {dataset_name}...")
    tiger_model, tiger_mapper, tiger_config = load_model_and_mapper('tiger', tiger_checkpoint, device)
    tiger_results = evaluate_model(
        tiger_model, 'tiger', tiger_mapper, tiger_config,
        test_samples, popularity_grouper, device, beam_size,
        max_samples_per_group=max_samples_per_group
    )
    del tiger_model
    torch.cuda.empty_cache()
    
    # Evaluate PRISM
    logger.info(f"\nEvaluating PRISM on {dataset_name}...")
    prism_model, prism_mapper, prism_config = load_model_and_mapper('prism', prism_checkpoint, device)
    
    content_embs, collab_embs, codebook_vecs = None, None, None
    training_config = prism_config.get('training', None)
    if training_config and getattr(training_config, 'use_multimodal_fusion', False):
        content_embs, collab_embs, codebook_vecs = load_prism_multimodal_data(prism_config)
    
    prism_results = evaluate_model(
        prism_model, 'prism', prism_mapper, prism_config,
        test_samples, popularity_grouper, device, beam_size,
        content_embeddings=content_embs,
        collab_embeddings=collab_embs,
        codebook_vectors=codebook_vecs,
        max_samples_per_group=max_samples_per_group
    )
    del prism_model
    torch.cuda.empty_cache()
    
    # Evaluate ActionPiece
    logger.info(f"\nEvaluating ActionPiece on {dataset_name}...")
    actionpiece_model, actionpiece_mapper, actionpiece_config = load_model_and_mapper('actionpiece', actionpiece_checkpoint, device)
    actionpiece_results = evaluate_model(
        actionpiece_model, 'actionpiece', actionpiece_mapper, actionpiece_config,
        test_samples, popularity_grouper, device, beam_size,
        max_samples_per_group=max_samples_per_group
    )
    del actionpiece_model
    torch.cuda.empty_cache()
    
    # Print results immediately after evaluation
    logger.info("\n" + "=" * 90)
    logger.info(f"LONG-TAIL COMPARISON: {dataset_name.upper()} (TIGER vs PRISM vs ActionPiece)")
    logger.info("=" * 90)
    
    group_names = popularity_grouper.get_group_names()
    for group_idx, group_name in enumerate(group_names):
        logger.info(f"\n{group_name} (n={tiger_results['group_counts'][group_idx]}):")
        logger.info("-" * 80)
        
        for metric in ['Recall@10', 'NDCG@10']:
            tiger_val = tiger_results['per_group'][group_name][metric]
            prism_val = prism_results['per_group'][group_name][metric]
            ap_val = actionpiece_results['per_group'][group_name][metric]
            
            prism_imp = ((prism_val - tiger_val) / tiger_val * 100) if tiger_val > 0 else 0
            ap_imp = ((ap_val - tiger_val) / tiger_val * 100) if tiger_val > 0 else 0
            
            logger.info(f"{metric:12}: TIGER={tiger_val:.4f}, PRISM={prism_val:.4f} ({prism_imp:+.1f}%), ActionPiece={ap_val:.4f} ({ap_imp:+.1f}%)")
    
    # Overall results
    logger.info(f"\nOverall:")
    logger.info("-" * 80)
    for metric in ['Recall@10', 'NDCG@10']:
        tiger_val = tiger_results['overall'][metric]
        prism_val = prism_results['overall'][metric]
        ap_val = actionpiece_results['overall'][metric]
        
        prism_imp = ((prism_val - tiger_val) / tiger_val * 100) if tiger_val > 0 else 0
        ap_imp = ((ap_val - tiger_val) / tiger_val * 100) if tiger_val > 0 else 0
        
        logger.info(f"{metric:12}: TIGER={tiger_val:.4f}, PRISM={prism_val:.4f} ({prism_imp:+.1f}%), ActionPiece={ap_val:.4f} ({ap_imp:+.1f}%)")
    
    logger.info("=" * 90)
    
    return tiger_results, prism_results, actionpiece_results


def plot_multi_dataset_comparison(
    all_results: Dict[str, Tuple[Dict, Dict]],
    output_path: str,
    metrics: List[str] = None
):
    """
    Generate publication-quality figure with multiple datasets.
    Shows absolute values with relative improvement annotations.
    
    Figure size: 9.0 x 4.5 inches (larger for better readability)
    """
    metrics = metrics or ['Recall@10', 'NDCG@10']
    
    # Publication-quality settings - LARGER fonts for readability
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 13,
        'axes.labelsize': 13,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    
    dataset_names = list(all_results.keys())
    num_datasets = len(dataset_names)
    
    # LARGER figure size
    fig = plt.figure(figsize=(9.0, 4.5), dpi=300)
    fig.patch.set_facecolor('white')
    
    # Use GridSpec for precise layout control - legend on top row
    # Reduced wspace and added left/right margins to maximize subplot width
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, num_datasets, figure=fig, height_ratios=[0.10, 1], 
                  hspace=0.22, wspace=0.15,
                  left=0.06, right=0.98, top=0.92, bottom=0.10)
    
    # Top row: shared legend spanning all columns
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')
    
    # Bottom row: subplots for each dataset
    axes = []
    for i in range(num_datasets):
        ax = fig.add_subplot(gs[1, i])
        axes.append(ax)
    
    # Color scheme
    colors = {
        ('TIGER', 'Recall'): '#4E79A7',
        ('TIGER', 'NDCG'): '#76B7B2',
        ('PRISM', 'Recall'): '#E15759',
        ('PRISM', 'NDCG'): '#F28E2B',
    }
    
    hatches = {
        ('TIGER', 'Recall'): '',
        ('TIGER', 'NDCG'): '',
        ('PRISM', 'Recall'): '//',
        ('PRISM', 'NDCG'): '//',
    }
    
    # Get global max for consistent y-axis
    all_values = []
    for dataset_name, (tiger_res, prism_res) in all_results.items():
        group_names = list(tiger_res['per_group'].keys())
        for g in group_names:
            for m in metrics:
                all_values.append(tiger_res['per_group'][g].get(m, 0))
                all_values.append(prism_res['per_group'][g].get(m, 0))
    y_max = 0.18
    
    for ax_idx, dataset_name in enumerate(dataset_names):
        ax = axes[ax_idx]
        tiger_results, prism_results = all_results[dataset_name]
        
        group_names = list(tiger_results['per_group'].keys())
        num_groups = len(group_names)
        num_metrics = len(metrics)
        
        x = np.arange(num_groups) * 1.7
        total_bars = 2 * num_metrics
        bar_width = 0.20
        gap_between_bars = 0.15
        
        bar_idx = 0
        legend_handles = []
        legend_labels = []
        
        for model_name, results in [('TIGER', tiger_results), ('PRISM', prism_results)]:
            for metric in metrics:
                values = [results['per_group'][g].get(metric, 0) for g in group_names]
                base_offset = (bar_idx - (total_bars - 1) / 2) * (bar_width + gap_between_bars)
                
                metric_type = 'Recall' if 'Recall' in metric else 'NDCG'
                color = colors[(model_name, metric_type)]
                hatch = hatches[(model_name, metric_type)]
                
                bars = ax.bar(
                    x + base_offset, values, bar_width,
                    color=color,
                    edgecolor='#222222',
                    linewidth=0.5,
                    hatch=hatch,
                    zorder=3
                )
                
                # Collect legend handles from first subplot only
                if ax_idx == 0:
                    legend_handles.append(bars[0])
                    legend_labels.append(f'{model_name} {metric_type}')
                
                # Add value labels - LARGER fonts
                for i, (bar, val) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    bar_x = bar.get_x() + bar.get_width() / 2
                    
                    if model_name == 'PRISM':
                        tiger_val = tiger_results['per_group'][group_names[i]].get(metric, 0)
                        
                        # Absolute value - LARGER font, BOLD
                        ax.text(
                            bar_x, height + 0.002,
                            f'{val:.3f}',
                            ha='center', va='bottom',
                            fontsize=10,
                            color='#000000',
                            fontweight='bold',
                            rotation=90
                        )
                        
                        # Improvement label - LARGER font
                        if tiger_val > 0:
                            improvement = (val - tiger_val) / tiger_val * 100
                            imp_color = '#D62828' if improvement > 0 else '#2A9D8F'
                            imp_y = height + 0.034
                            
                            ax.text(
                                bar_x, imp_y,
                                f'{improvement:+.0f}%',
                                ha='center', va='bottom',
                                fontsize=9,
                                color=imp_color,
                                fontweight='bold',
                                rotation=90,
                                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', 
                                         edgecolor=imp_color, linewidth=0.6, alpha=0.95)
                            )
                    else:
                        # TIGER: LARGER font, BOLD
                        ax.text(
                            bar_x, height + 0.002,
                            f'{val:.3f}',
                            ha='center', va='bottom',
                            fontsize=10,
                            color='#000000',
                            fontweight='bold',
                            rotation=90
                        )
                
                bar_idx += 1
        
        # Subplot styling - LARGER fonts, more padding for title
        display_name = DATASET_CONFIGS[dataset_name]['display_name']
        ax.set_title(f'{display_name}', fontsize=15, fontweight='normal', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=0, fontsize=12)
        ax.set_ylim(0, y_max)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, linewidth=0.4, zorder=0)
        
        # Store handles for legend
        if ax_idx == 0:
            all_handles = legend_handles
            all_labels = legend_labels
    
    # Add legend to the dedicated legend axis (no overlap with titles)
    ax_legend.legend(
        all_handles, all_labels,
        loc='center',
        ncol=4,
        frameon=False,
        fontsize=13,
        handletextpad=0.5,
        columnspacing=2.2,
    )
    
    # No tight_layout() - we use GridSpec with explicit margins instead
    
    # Save in multiple formats
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    
    # PNG
    png_path = f"{base_path}.png"
    plt.savefig(png_path, dpi=300, facecolor='white', pad_inches=0.01)
    logger.info(f"Saved PNG: {png_path}")
    
    # PDF (vector format for LaTeX)
    pdf_path = f"{base_path}.pdf"
    plt.savefig(pdf_path, format='pdf', facecolor='white', pad_inches=0.01)
    logger.info(f"Saved PDF: {pdf_path}")
    
    # SVG (vector format for editing)
    svg_path = f"{base_path}.svg"
    plt.savefig(svg_path, format='svg', facecolor='white', pad_inches=0.01)
    logger.info(f"Saved SVG: {svg_path}")
    
    plt.close()


def plot_relative_improvement(
    all_results: Dict[str, Tuple[Dict, Dict]],
    output_path: str,
    metrics: List[str] = None
):
    """
    Plot relative improvement of PRISM over TIGER.
    This helps highlight PRISM's advantages, especially on long-tail items.
    
    Figure size: 6.5 x 3.2 inches (EXACTLY matching visualize_codebook_comparison.py)
    """
    metrics = metrics or ['Recall@10', 'NDCG@10']
    
    # Publication-quality settings (NeurIPS/KDD standard)
    # EXACTLY matching visualize_codebook_comparison.py
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    
    dataset_names = list(all_results.keys())
    num_datasets = len(dataset_names)
    
    # Fixed figure size: 6.5 x 3.2 inches (EXACTLY matching visualize_codebook_comparison.py)
    fig = plt.figure(figsize=(6.5, 3.2), dpi=300)
    fig.patch.set_facecolor('white')
    
    # Create subplots
    axes = []
    for i in range(num_datasets):
        ax = fig.add_subplot(1, num_datasets, i + 1)
        axes.append(ax)
    
    colors = ['#59A14F', '#76B7B2']  # Green shades for improvement
    
    for ax_idx, dataset_name in enumerate(dataset_names):
        ax = axes[ax_idx]
        tiger_results, prism_results = all_results[dataset_name]
        
        group_names = list(tiger_results['per_group'].keys())
        num_groups = len(group_names)
        
        x = np.arange(num_groups)
        bar_width = 0.35
        
        for m_idx, metric in enumerate(metrics):
            improvements = []
            for g in group_names:
                tiger_val = tiger_results['per_group'][g].get(metric, 0)
                prism_val = prism_results['per_group'][g].get(metric, 0)
                if tiger_val > 0:
                    rel_imp = (prism_val - tiger_val) / tiger_val * 100
                else:
                    rel_imp = 0
                improvements.append(rel_imp)
            
            offset = (m_idx - 0.5) * bar_width
            bars = ax.bar(
                x + offset, improvements, bar_width,
                label=metric,
                color=colors[m_idx % len(colors)],
                edgecolor='#222222',
                linewidth=0.5,
                zorder=3
            )
            
            # Value labels
            for bar, val in zip(bars, improvements):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset_y = 2 if height >= 0 else -2
                ax.annotate(
                    f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset_y),
                    textcoords="offset points",
                    ha='center', va=va,
                    fontsize=7,
                    color='#000000',
                    fontweight='normal'
                )
        
        ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1, zorder=2)
        
        display_name = DATASET_CONFIGS[dataset_name]['display_name']
        ax.set_title(f'{display_name}', fontsize=10, fontweight='normal', pad=4)
        ax.set_xlabel('Popularity Group')
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=0)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, linewidth=0.4, zorder=0)
    
    axes[0].set_ylabel('Relative Improvement (%)\n(PRISM vs TIGER)')
    
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=9,
        handletextpad=0.3,
        columnspacing=1.5,
    )
    
    plt.tight_layout()
    
    # Save in multiple formats (matching visualize_codebook_comparison.py)
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    
    # PNG
    png_path = f"{base_path}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    logger.info(f"Saved PNG: {png_path}")
    
    # PDF (vector format for LaTeX)
    pdf_path = f"{base_path}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.02)
    logger.info(f"Saved PDF: {pdf_path}")
    
    # SVG (vector format for editing)
    svg_path = f"{base_path}.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white', pad_inches=0.02)
    logger.info(f"Saved SVG: {svg_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Long-tail comparison: TIGER vs PRISM vs ActionPiece")
    
    # Beauty checkpoints
    parser.add_argument('--tiger_checkpoint_beauty', type=str, default=None)
    parser.add_argument('--prism_checkpoint_beauty', type=str, default=None)
    parser.add_argument('--actionpiece_checkpoint_beauty', type=str, default=None)
    
    # CDs checkpoints (optional)
    parser.add_argument('--tiger_checkpoint_cds', type=str, default=None)
    parser.add_argument('--prism_checkpoint_cds', type=str, default=None)
    parser.add_argument('--actionpiece_checkpoint_cds', type=str, default=None)
    
    parser.add_argument('--output_dir', type=str, default='scripts/output/longtail_comparison')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_groups', type=int, default=3)
    parser.add_argument('--beam_size', type=int, default=30)
    parser.add_argument('--metrics', type=str, nargs='+', default=['Recall@10', 'NDCG@10'])
    parser.add_argument('--max_samples_per_group', type=int, default=None,
                       help='Limit samples per group for quick testing (e.g., 100)')
    parser.add_argument('--plot_only', action='store_true',
                       help='Skip evaluation and only generate plots from existing JSON files')
    parser.add_argument('--eval_actionpiece_only', action='store_true',
                       help='Only evaluate ActionPiece, load TIGER/PRISM results from existing JSON files')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    if args.plot_only:
        # Load results from JSON files
        logger.info("Plot-only mode: Loading results from JSON files...")
        
        # Load Beauty results
        beauty_tiger_path = output_dir / 'tiger_results_beauty.json'
        beauty_prism_path = output_dir / 'prism_results_beauty.json'
        beauty_ap_path = output_dir / 'actionpiece_results_beauty.json'
        
        if beauty_tiger_path.exists() and beauty_prism_path.exists() and beauty_ap_path.exists():
            with open(beauty_tiger_path, 'r') as f:
                tiger_beauty = json.load(f)
            with open(beauty_prism_path, 'r') as f:
                prism_beauty = json.load(f)
            with open(beauty_ap_path, 'r') as f:
                ap_beauty = json.load(f)
            all_results['beauty'] = (tiger_beauty, prism_beauty, ap_beauty)
            logger.info(f"Loaded Beauty results from {output_dir}")
        else:
            logger.error(f"Beauty results not found in {output_dir}")
            return
        
        # Load CDs results if available
        cds_tiger_path = output_dir / 'tiger_results_cds.json'
        cds_prism_path = output_dir / 'prism_results_cds.json'
        cds_ap_path = output_dir / 'actionpiece_results_cds.json'
        
        if cds_tiger_path.exists() and cds_prism_path.exists() and cds_ap_path.exists():
            with open(cds_tiger_path, 'r') as f:
                tiger_cds = json.load(f)
            with open(cds_prism_path, 'r') as f:
                prism_cds = json.load(f)
            with open(cds_ap_path, 'r') as f:
                ap_cds = json.load(f)
            all_results['cds'] = (tiger_cds, prism_cds, ap_cds)
            logger.info(f"Loaded CDs results from {output_dir}")
        else:
            logger.info("CDs results not found, skipping")
    
    elif args.eval_actionpiece_only:
        # Only evaluate ActionPiece, load TIGER/PRISM from existing files
        logger.info("ActionPiece-only evaluation mode: Loading TIGER/PRISM results from JSON files...")
        
        if not args.actionpiece_checkpoint_beauty:
            logger.error("ActionPiece Beauty checkpoint required for eval_actionpiece_only mode.")
            return
        
        # Load existing TIGER/PRISM results for Beauty
        beauty_tiger_path = output_dir / 'tiger_results_beauty.json'
        beauty_prism_path = output_dir / 'prism_results_beauty.json'
        
        if not beauty_tiger_path.exists() or not beauty_prism_path.exists():
            logger.error(f"TIGER/PRISM Beauty results not found in {output_dir}. Please run full evaluation first.")
            return
        
        with open(beauty_tiger_path, 'r') as f:
            tiger_beauty = json.load(f)
        with open(beauty_prism_path, 'r') as f:
            prism_beauty = json.load(f)
        logger.info(f"Loaded existing TIGER/PRISM Beauty results")
        
        # Evaluate ActionPiece on Beauty
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING ACTIONPIECE ON BEAUTY DATASET")
        logger.info("=" * 60)
        
        config = DATASET_CONFIGS['beauty']
        test_samples = load_test_data(config['unified_test_path'])
        popularity_grouper = PopularityGrouper(config['unified_item_emb_path'], num_groups=args.num_groups)
        
        actionpiece_model, actionpiece_mapper, actionpiece_config = load_model_and_mapper(
            'actionpiece', args.actionpiece_checkpoint_beauty, args.device
        )
        ap_beauty = evaluate_model(
            actionpiece_model, 'actionpiece', actionpiece_mapper, actionpiece_config,
            test_samples, popularity_grouper, args.device, args.beam_size,
            max_samples_per_group=args.max_samples_per_group
        )
        del actionpiece_model
        torch.cuda.empty_cache()
        
        all_results['beauty'] = (tiger_beauty, prism_beauty, ap_beauty)
        
        # Save ActionPiece Beauty results
        with open(output_dir / 'actionpiece_results_beauty.json', 'w') as f:
            json.dump(ap_beauty, f, indent=2)
        logger.info(f"Saved ActionPiece Beauty results")
        
        # Print Beauty results immediately
        logger.info("\n" + "=" * 90)
        logger.info("ACTIONPIECE BEAUTY RESULTS (Immediate)")
        logger.info("=" * 90)
        group_names = popularity_grouper.get_group_names()
        for group_idx, group_name in enumerate(group_names):
            logger.info(f"\n{group_name} (n={ap_beauty['group_counts'][group_idx]}):")
            logger.info("-" * 80)
            for metric in ['Recall@10', 'NDCG@10']:
                tiger_val = tiger_beauty['per_group'][group_name][metric]
                prism_val = prism_beauty['per_group'][group_name][metric]
                ap_val = ap_beauty['per_group'][group_name][metric]
                prism_imp = ((prism_val - tiger_val) / tiger_val * 100) if tiger_val > 0 else 0
                ap_imp = ((ap_val - tiger_val) / tiger_val * 100) if tiger_val > 0 else 0
                logger.info(f"{metric:12}: TIGER={tiger_val:.4f}, PRISM={prism_val:.4f} ({prism_imp:+.1f}%), ActionPiece={ap_val:.4f} ({ap_imp:+.1f}%)")
        logger.info(f"\nOverall:")
        logger.info("-" * 80)
        for metric in ['Recall@10', 'NDCG@10']:
            tiger_val = tiger_beauty['overall'][metric]
            prism_val = prism_beauty['overall'][metric]
            ap_val = ap_beauty['overall'][metric]
            prism_imp = ((prism_val - tiger_val) / tiger_val * 100) if tiger_val > 0 else 0
            ap_imp = ((ap_val - tiger_val) / tiger_val * 100) if tiger_val > 0 else 0
            logger.info(f"{metric:12}: TIGER={tiger_val:.4f}, PRISM={prism_val:.4f} ({prism_imp:+.1f}%), ActionPiece={ap_val:.4f} ({ap_imp:+.1f}%)")
        logger.info("=" * 90)
        
        # Evaluate CDs if checkpoint provided
        if args.actionpiece_checkpoint_cds:
            # Load existing TIGER/PRISM results for CDs
            cds_tiger_path = output_dir / 'tiger_results_cds.json'
            cds_prism_path = output_dir / 'prism_results_cds.json'
            
            if not cds_tiger_path.exists() or not cds_prism_path.exists():
                logger.warning(f"TIGER/PRISM CDs results not found in {output_dir}. Skipping CDs.")
            else:
                with open(cds_tiger_path, 'r') as f:
                    tiger_cds = json.load(f)
                with open(cds_prism_path, 'r') as f:
                    prism_cds = json.load(f)
                logger.info(f"Loaded existing TIGER/PRISM CDs results")
                
                # Evaluate ActionPiece on CDs
                logger.info("\n" + "=" * 60)
                logger.info("EVALUATING ACTIONPIECE ON CDS DATASET")
                logger.info("=" * 60)
                
                config = DATASET_CONFIGS['cds']
                test_samples = load_test_data(config['unified_test_path'])
                popularity_grouper = PopularityGrouper(config['unified_item_emb_path'], num_groups=args.num_groups)
                
                actionpiece_model, actionpiece_mapper, actionpiece_config = load_model_and_mapper(
                    'actionpiece', args.actionpiece_checkpoint_cds, args.device
                )
                ap_cds = evaluate_model(
                    actionpiece_model, 'actionpiece', actionpiece_mapper, actionpiece_config,
                    test_samples, popularity_grouper, args.device, args.beam_size,
                    max_samples_per_group=args.max_samples_per_group
                )
                del actionpiece_model
                torch.cuda.empty_cache()
                
                all_results['cds'] = (tiger_cds, prism_cds, ap_cds)
                
                # Save ActionPiece CDs results
                with open(output_dir / 'actionpiece_results_cds.json', 'w') as f:
                    json.dump(ap_cds, f, indent=2)
                logger.info(f"Saved ActionPiece CDs results")
    
    else:
        # Normal evaluation mode
        if not args.tiger_checkpoint_beauty or not args.prism_checkpoint_beauty or not args.actionpiece_checkpoint_beauty:
            logger.error("All three model checkpoints required for evaluation mode. Use --plot_only to skip evaluation.")
            return
        
        # Evaluate Beauty
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATING BEAUTY DATASET")
        logger.info("=" * 60)
        tiger_beauty, prism_beauty, ap_beauty = evaluate_dataset(
            'beauty',
            args.tiger_checkpoint_beauty,
            args.prism_checkpoint_beauty,
            args.actionpiece_checkpoint_beauty,
            args.device,
            args.num_groups,
            args.beam_size,
            max_samples_per_group=args.max_samples_per_group
        )
        all_results['beauty'] = (tiger_beauty, prism_beauty, ap_beauty)
        
        # Save Beauty results
        with open(output_dir / 'tiger_results_beauty.json', 'w') as f:
            json.dump(tiger_beauty, f, indent=2)
        with open(output_dir / 'prism_results_beauty.json', 'w') as f:
            json.dump(prism_beauty, f, indent=2)
        with open(output_dir / 'actionpiece_results_beauty.json', 'w') as f:
            json.dump(ap_beauty, f, indent=2)
        
        # Evaluate CDs if checkpoints provided
        if args.tiger_checkpoint_cds and args.prism_checkpoint_cds and args.actionpiece_checkpoint_cds:
            logger.info("\n" + "=" * 60)
            logger.info("EVALUATING CDS DATASET")
            logger.info("=" * 60)
            tiger_cds, prism_cds, ap_cds = evaluate_dataset(
                'cds',
                args.tiger_checkpoint_cds,
                args.prism_checkpoint_cds,
                args.actionpiece_checkpoint_cds,
                args.device,
                args.num_groups,
                args.beam_size,
                max_samples_per_group=args.max_samples_per_group
            )
            all_results['cds'] = (tiger_cds, prism_cds, ap_cds)
            
            # Save CDs results
            with open(output_dir / 'tiger_results_cds.json', 'w') as f:
                json.dump(tiger_cds, f, indent=2)
            with open(output_dir / 'prism_results_cds.json', 'w') as f:
                json.dump(prism_cds, f, indent=2)
            with open(output_dir / 'actionpiece_results_cds.json', 'w') as f:
                json.dump(ap_cds, f, indent=2)
    
    # Import new plotting functions
    from plot_longtail_three_models import (
        plot_multi_dataset_comparison_three_models,
        print_results_table_three_models
    )
    
    # Print results
    print_results_table_three_models(all_results, args.metrics)
    
    # Generate figure
    plot_multi_dataset_comparison_three_models(
        all_results,
        str(output_dir / 'longtail_comparison.pdf'),
        metrics=args.metrics,
        dataset_configs=DATASET_CONFIGS
    )
    
    logger.info(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
