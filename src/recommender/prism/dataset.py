"""
Dataset implementation for generative recommendation.

Handles loading sequence data, semantic ID mappings, and purified features.
"""

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


def load_purified_embeddings(content_path: str, collab_path: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Load Stage 1 purified embeddings.

    Expects:
      - item_purified_ids.npy        (n_items,) int64 item IDs
      - item_purified_content.npy    (n_items, 128) h_t
      - item_purified_collab.npy     (n_items, 128) h_c
      - item_purified_z_clean.npy    (n_items, 256) [h_c || h_t]
      - item_codebook_zq.npy         (n_items, 32)  z_q (optional)
    """
    content_dir = os.path.dirname(content_path)
    ids_path = os.path.join(content_dir, 'item_purified_ids.npy')
    z_clean_path = os.path.join(content_dir, 'item_purified_z_clean.npy')
    zq_path = os.path.join(content_dir, 'item_codebook_zq.npy')

    if not os.path.exists(content_path) or not os.path.exists(collab_path):
        logger.warning(f"Purified embeddings not found, DSI fusion will be disabled")
        return {}, {}, {}

    logger.info(f"Loading purified content: {content_path}")
    logger.info(f"Loading purified collab: {collab_path}")

    content_arr = np.load(content_path)   # (n_items, dim)
    collab_arr = np.load(collab_path)     # (n_items, dim)
    z_clean_arr = np.load(z_clean_path) if os.path.exists(z_clean_path) else None

    if os.path.exists(ids_path):
        item_ids = np.load(ids_path)
    else:
        item_ids = np.arange(len(content_arr))

    content_dict = {int(item_ids[i]): content_arr[i].astype(np.float32) for i in range(len(item_ids))}
    collab_dict = {int(item_ids[i]): collab_arr[i].astype(np.float32) for i in range(len(item_ids))}
    z_clean_dict = {}
    if z_clean_arr is not None:
        z_clean_dict = {int(item_ids[i]): z_clean_arr[i].astype(np.float32) for i in range(len(item_ids))}
    else:
        logger.warning("item_purified_z_clean.npy not found; falling back to [content || collab] concat")
        z_clean_dict = {
            int(item_ids[i]): np.concatenate([content_arr[i], collab_arr[i]]).astype(np.float32)
            for i in range(len(item_ids))
        }

    # Optional: codebook z_q (32D quantized latent from Stage 1 RQ-VAE)
    codebook_dict = {}
    if os.path.exists(zq_path):
        zq_arr = np.load(zq_path)
        codebook_dict = {int(item_ids[i]): zq_arr[i].astype(np.float32) for i in range(len(item_ids))}

    logger.info(f"Loaded purified features for {len(content_dict)} items")
    logger.info(f"  Purified content dim: {content_arr.shape[1]}")
    logger.info(f"  Purified collab dim: {collab_arr.shape[1]}")
    if z_clean_arr is not None:
        logger.info(f"  Purified z_clean dim: {z_clean_arr.shape[1]}")
    if codebook_dict:
        logger.info(f"  Codebook z_q dim: {zq_arr.shape[1]}")

    return content_dict, collab_dict, z_clean_dict, codebook_dict


def load_content_embeddings(data_dir: str) -> Dict[int, np.ndarray]:
    """Load raw content embeddings (768D) from item_emb.parquet."""
    parquet_path = os.path.join(data_dir, 'item_emb.parquet')
    if not os.path.exists(parquet_path):
        logger.warning(f"Content embeddings not found at {parquet_path}")
        return {}
    import pandas as pd
    item_df = pd.read_parquet(parquet_path)
    item_ids = item_df['ItemID'].values
    emb_col = 'attribute_embedding' if 'attribute_embedding' in item_df.columns else 'embedding'
    embeddings = np.stack([np.array(emb) for emb in item_df[emb_col]])
    content_dict = {int(iid): embeddings[i].astype(np.float32) for i, iid in enumerate(item_ids)}
    logger.info(f"Loaded raw content embeddings: {len(content_dict)} items, dim={embeddings.shape[1]}")
    return content_dict


def load_codebook_mappings(codebook_dir: str) -> Tuple[Dict[int, np.ndarray], int]:
    """Load codebook z_q embeddings from Stage 1 output directory."""
    zq_path = os.path.join(codebook_dir, 'item_codebook_zq.npy')
    ids_path = os.path.join(codebook_dir, 'item_purified_ids.npy')
    if os.path.exists(zq_path) and os.path.exists(ids_path):
        zq_arr = np.load(zq_path)
        item_ids = np.load(ids_path)
        return {int(item_ids[i]): zq_arr[i].astype(np.float32) for i in range(len(item_ids))}, zq_arr.shape[1]
    return {}, 0


def load_collab_embeddings(file_path: str, data_dir: str = None) -> Dict[int, np.ndarray]:
    """Load raw collaborative embeddings (64D) from .npy file.

    For .npy files, item IDs are resolved from item_emb.parquet in data_dir.
    For .npz files, item_ids are stored within the archive.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Collab embeddings not found at {file_path}")
        return {}
    embeddings = np.load(file_path, allow_pickle=True)
    if file_path.endswith('.npz'):
        data = np.load(file_path, allow_pickle=True)
        item_ids = data['item_ids']
        embeddings = data['embeddings']
        collab_dict = {int(iid): embeddings[i].astype(np.float32) for i, iid in enumerate(item_ids)}
    else:
        import pandas as pd
        if data_dir:
            parquet_path = os.path.join(data_dir, 'item_emb.parquet')
            if os.path.exists(parquet_path):
                item_ids = pd.read_parquet(parquet_path)['ItemID'].values
            else:
                item_ids = np.arange(len(embeddings))
        else:
            item_ids = np.arange(len(embeddings))
        collab_dict = {int(iid): embeddings[i].astype(np.float32) for i, iid in enumerate(item_ids)}
    logger.info(f"Loaded raw collab embeddings: {len(collab_dict)} items, dim={embeddings.shape[1]}")
    return collab_dict


class SemanticIDMapper:
    """Manages the mapping from item IDs to semantic codes."""

    def __init__(self, mapping_path: str, codebook_size: int = 256, num_layers: int = 3, pad_token_id: int = 0, codebook_sizes: Optional[List[int]] = None):
        self.mapping_path = mapping_path
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.codebook_sizes = codebook_sizes

        logger.info(f"Loading semantic ID mapping from {mapping_path}")
        with open(mapping_path, 'r') as f:
            raw_mapping = json.load(f)

        if 'item_to_codes' in raw_mapping:
            logger.info("Detected Prism format with metadata")
            raw_mapping = raw_mapping['item_to_codes']

        actual_max_layers = max(len(codes) for codes in raw_mapping.values())
        if actual_max_layers != num_layers:
            logger.warning(f"Auto-adjusting num_layers: {num_layers} -> {actual_max_layers}")
            num_layers = actual_max_layers
            self.num_layers = num_layers

        if self.codebook_sizes is None:
            detected_sizes = self._detect_codebook_sizes(raw_mapping, num_layers)
            if detected_sizes is not None:
                self.codebook_sizes = detected_sizes
                logger.info(f"Auto-detected codebook sizes: {detected_sizes}")
            else:
                self.codebook_sizes = [codebook_size] * num_layers
        else:
            if len(self.codebook_sizes) != num_layers:
                if len(self.codebook_sizes) < num_layers:
                    self.codebook_sizes = self.codebook_sizes + [self.codebook_sizes[-1]] * (num_layers - len(self.codebook_sizes))
                else:
                    self.codebook_sizes = self.codebook_sizes[:num_layers]

        layer_max_values = [0] * num_layers
        for codes in raw_mapping.values():
            if len(codes) < num_layers:
                codes = codes + [pad_token_id] * (num_layers - len(codes))
            for i, code in enumerate(codes):
                if code != pad_token_id:
                    layer_max_values[i] = max(layer_max_values[i], code)

        self._layer_max_values = layer_max_values

        self.item_to_codes = {}
        for item_id_str, codes in raw_mapping.items():
            item_id = int(item_id_str)
            if len(codes) < num_layers:
                codes = codes + [pad_token_id] * (num_layers - len(codes))
            offset_codes = self._apply_offset(codes)
            self.item_to_codes[item_id] = offset_codes

        self._actual_vocab_size = self._compute_vocab_size(layer_max_values)
        self.pad_codes = [pad_token_id] * num_layers
        logger.info(f"Loaded {len(self.item_to_codes)} item-to-code mappings, vocab_size={self._actual_vocab_size}")

    def _detect_codebook_sizes(self, raw_mapping: dict, num_layers: int) -> Optional[List[int]]:
        layer_max_values = [0] * num_layers
        for codes in raw_mapping.values():
            for i, code in enumerate(codes[:num_layers]):
                if code != self.pad_token_id:
                    layer_max_values[i] = max(layer_max_values[i], code)
        if len(set(layer_max_values)) > 1:
            max_val, min_val = max(layer_max_values), min(v for v in layer_max_values if v > 0)
            if max_val > min_val * 1.2:
                common_sizes = [64, 128, 256, 512, 1024, 2048]
                detected_sizes = [next((s for s in common_sizes if s > v), v + 1) for v in layer_max_values]
                return detected_sizes
        return None

    def _apply_offset(self, codes: List[int]) -> List[int]:
        offset_codes = []
        cumulative_offset = 1
        for i, code in enumerate(codes):
            offset_codes.append(code + cumulative_offset)
            cumulative_offset += (self._layer_max_values[i] + 1)
        return offset_codes

    def get_codes(self, item_id: int) -> List[int]:
        return self.item_to_codes.get(item_id, self.pad_codes)

    def get_codes_batch(self, item_ids: List[int]) -> List[List[int]]:
        return [self.get_codes(item_id) for item_id in item_ids]

    def __len__(self) -> int:
        return len(self.item_to_codes)

    def _compute_vocab_size(self, layer_max_values: List[int]) -> int:
        max_token_id = 0
        cumulative_offset = 1
        for i, layer_max in enumerate(layer_max_values):
            max_token_id = max(max_token_id, layer_max + cumulative_offset)
            cumulative_offset += (layer_max + 1)
        return max_token_id + 1

    def get_vocab_size(self, use_actual: bool = True) -> int:
        if use_actual:
            return self._actual_vocab_size
        return 1 + sum(self.codebook_sizes)

    def get_layer_stats(self) -> Dict:
        return {
            'num_layers': self.num_layers,
            'codebook_size': self.codebook_size,
            'codebook_sizes': self.codebook_sizes,
            'layer_max_values': self._layer_max_values,
            'actual_vocab_size': self._actual_vocab_size,
            'theoretical_vocab_size': 1 + sum(self.codebook_sizes),
            'savings': (1 + sum(self.codebook_sizes)) - self._actual_vocab_size
        }


def process_sequence_data(file_path: str, mode: str, max_len: int, pad_token_id: int = 0) -> List[Dict]:
    logger.info(f"Processing sequence data from {file_path} in {mode} mode")
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} sequences")

    processed_data = []
    if mode == 'train':
        for _, row in df.iterrows():
            history = list(row['history'])
            target = row['target']
            sequence = history + [target]
            for i in range(1, len(sequence)):
                processed_data.append({'history': sequence[:i], 'target': sequence[i]})
    elif mode == 'evaluation':
        for _, row in df.iterrows():
            processed_data.append({'history': list(row['history']), 'target': row['target']})
    else:
        raise ValueError(f"Invalid mode: {mode}")

    logger.info(f"Generated {len(processed_data)} samples in {mode} mode")
    return processed_data


def pad_or_truncate(sequence: List[int], max_len: int, pad_token_id: int = 0) -> List[int]:
    if len(sequence) > max_len:
        return sequence[-max_len:]
    return [pad_token_id] * (max_len - len(sequence)) + sequence


class GenRecDataset(Dataset):
    """Dataset for generative recommendation with purified features."""

    def __init__(
        self,
        sequence_file: str,
        semantic_mapper: SemanticIDMapper,
        mode: str = 'train',
        max_len: int = 20,
        pad_token_id: int = 0,
        purified_content: Optional[Dict[int, np.ndarray]] = None,
        purified_collab: Optional[Dict[int, np.ndarray]] = None,
        purified_z_clean: Optional[Dict[int, np.ndarray]] = None,
        purified_codebook: Optional[Dict[int, np.ndarray]] = None,
        use_multimodal: bool = False,
        sample_limit: Optional[int] = None,
    ):
        self.sequence_file = sequence_file
        self.semantic_mapper = semantic_mapper
        self.mode = mode
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.purified_content = purified_content or {}
        self.purified_collab = purified_collab or {}
        self.purified_z_clean = purified_z_clean or {}
        self.purified_codebook = purified_codebook or {}
        self.use_multimodal = use_multimodal
        self.codebook_dim = 32

        self.purified_dim = 128
        if self.purified_content:
            sample = next(iter(self.purified_content.values()))
            self.purified_dim = sample.shape[0]

        self.data = process_sequence_data(sequence_file, mode, max_len, pad_token_id)
        if sample_limit is not None:
            self.data = self.data[:sample_limit]
            logger.info(f"Fast-dev: capped {mode} samples to {len(self.data)} before code conversion")
        self._convert_to_codes()

        logger.info(f"Dataset initialized with {len(self.data)} samples")
        if use_multimodal and self.purified_content:
            logger.info(f"  Purified DSI: dim={self.purified_dim}D")

    def _convert_to_codes(self):
        logger.info("Converting item IDs to semantic codes...")
        missing_items = set()
        for item in self.data:
            history_ids = pad_or_truncate(item['history'], self.max_len, self.pad_token_id)
            history_codes = []
            for item_id in history_ids:
                codes = self.semantic_mapper.get_codes(item_id)
                if item_id != self.pad_token_id and codes == self.semantic_mapper.pad_codes:
                    missing_items.add(item_id)
                history_codes.extend(codes)

            target_id = item['target']
            target_codes = self.semantic_mapper.get_codes(target_id)
            if target_codes == self.semantic_mapper.pad_codes:
                missing_items.add(target_id)

            item['history_codes'] = history_codes
            item['target_codes'] = target_codes

        if missing_items:
            logger.warning(f"Found {len(missing_items)} items without semantic mappings")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        history_item_ids = item['history']
        target_item_id = item['target']

        history_item_ids_padded = pad_or_truncate(history_item_ids, self.max_len, self.pad_token_id)
        n_layers = self.semantic_mapper.num_layers

        result = {
            'history': item['history_codes'],
            'target': item['target_codes'],
            'history_item_ids': history_item_ids_padded,
            'target_item_id': target_item_id
        }

        if self.use_multimodal:
            history_purified_content = np.zeros((self.max_len, self.purified_dim), dtype=np.float32)
            for i, iid in enumerate(history_item_ids_padded):
                if iid in self.purified_content:
                    history_purified_content[i] = self.purified_content[iid]

            history_purified_collab = np.zeros((self.max_len, self.purified_dim), dtype=np.float32)
            for i, iid in enumerate(history_item_ids_padded):
                if iid in self.purified_collab:
                    history_purified_collab[i] = self.purified_collab[iid]

            result.update({
                'history_purified_content': history_purified_content,
                'history_purified_collab': history_purified_collab,
            })

            # Codebook z_q (32D quantized latent)
            if self.purified_codebook:
                history_codebook = np.zeros((self.max_len, self.codebook_dim), dtype=np.float32)
                for i, iid in enumerate(history_item_ids_padded):
                    if iid in self.purified_codebook:
                        history_codebook[i] = self.purified_codebook[iid]
                result['history_codebook_zq'] = history_codebook

            # Stage 1 exports z_clean as [h_t_hat || h_c_hat]; use it directly.
            target_z_clean = np.zeros(self.purified_dim * 2, dtype=np.float32)
            if target_item_id in self.purified_z_clean:
                target_z_clean = self.purified_z_clean[target_item_id]
            result['target_z_clean'] = target_z_clean

        return result

    def get_stats(self) -> Dict:
        return {
            'num_samples': len(self),
            'mode': self.mode,
            'max_len': self.max_len,
            'num_layers': self.semantic_mapper.num_layers,
            'sequence_length': self.max_len * self.semantic_mapper.num_layers,
            'vocab_size': self.semantic_mapper.get_vocab_size()
        }


def create_datasets(
    sequence_data_dir: str,
    semantic_mapping_path: str,
    max_len: int = 20,
    codebook_size: int = 256,
    num_layers: int = 3,
    pad_token_id: int = 0,
    model_config: Optional[any] = None,
    codebook_sizes: Optional[List[int]] = None,
    purified_content_path: Optional[str] = None,
    purified_collab_path: Optional[str] = None,
    use_multimodal: bool = False,
    train_sample_limit: Optional[int] = None,
    valid_sample_limit: Optional[int] = None,
    test_sample_limit: Optional[int] = None,
) -> Tuple[GenRecDataset, GenRecDataset, GenRecDataset, SemanticIDMapper]:
    """Create train, validation, and test datasets."""

    semantic_mapper = SemanticIDMapper(
        semantic_mapping_path, codebook_size=codebook_size,
        num_layers=num_layers, pad_token_id=pad_token_id,
        codebook_sizes=codebook_sizes
    )

    purified_content_dict = {}
    purified_collab_dict = {}
    purified_z_clean_dict = {}
    purified_codebook_dict = {}
    if use_multimodal and purified_content_path and purified_collab_path:
        logger.info("Loading Stage 1 purified features for DSI fusion...")
        purified_content_dict, purified_collab_dict, purified_z_clean_dict, purified_codebook_dict = load_purified_embeddings(
            purified_content_path, purified_collab_path
        )

    if semantic_mapper.num_layers != num_layers:
        logger.warning(f"num_layers auto-adjusted: {num_layers} -> {semantic_mapper.num_layers}")
        num_layers = semantic_mapper.num_layers

    if model_config is not None:
        model_config.set_vocab_size(semantic_mapper.get_vocab_size(use_actual=True))
        model_config.num_code_layers = num_layers
        logger.info(f"Updated vocab_size={model_config.vocab_size}, num_code_layers={num_layers}")

    data_dir = Path(sequence_data_dir)
    dataset_kwargs = dict(
        semantic_mapper=semantic_mapper, max_len=max_len, pad_token_id=pad_token_id,
        purified_content=purified_content_dict, purified_collab=purified_collab_dict,
        purified_z_clean=purified_z_clean_dict,
        purified_codebook=purified_codebook_dict,
        use_multimodal=use_multimodal,
    )

    train_dataset = GenRecDataset(
        str(data_dir / "train.parquet"),
        mode='train',
        sample_limit=train_sample_limit,
        **dataset_kwargs,
    )
    valid_dataset = GenRecDataset(
        str(data_dir / "valid.parquet"),
        mode='evaluation',
        sample_limit=valid_sample_limit,
        **dataset_kwargs,
    )
    test_dataset = GenRecDataset(
        str(data_dir / "test.parquet"),
        mode='evaluation',
        sample_limit=test_sample_limit,
        **dataset_kwargs,
    )

    return train_dataset, valid_dataset, test_dataset, semantic_mapper
