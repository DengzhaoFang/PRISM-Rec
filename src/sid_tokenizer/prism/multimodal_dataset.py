"""
Multi-Modal Dataset for PRISM Training

Loads and combines:
1. Content embeddings (768D from TIGER-format item_emb.parquet)
2. Collaborative embeddings (64D from LightGCN)
3. Co-occurrence graph from user sequences (for SACO loss)

Returns paired data (anchor, positive) when co-occurrence graph is available,
ensuring every anchor has a guaranteed hard positive for SACO contrastive learning.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PRISMDataset(Dataset):
    """
    Multi-modal dataset for PRISM training.

    Combines item content embeddings, collaborative embeddings,
    and co-occurrence graph for sequence-aware contrastive learning.

    Returns each item paired with a randomly-sampled co-occurring positive,
    guaranteeing every anchor has a hard positive for SACO.
    """

    def __init__(
        self,
        data_dir: str,
        embedding_file: str = 'item_emb.parquet',
        collab_embedding_file: str = 'lightgcn/item_embeddings_collab.npy',
        max_items: Optional[int] = None,
        train_seq_file: Optional[str] = 'train.parquet',
        cooc_window: int = 4,
    ):
        self.data_dir = Path(data_dir)
        self.cooc_window = cooc_window

        print(f"Loading item embeddings from {embedding_file}...")
        item_df = pd.read_parquet(self.data_dir / embedding_file)

        if max_items is not None:
            item_df = item_df.head(max_items)

        self.item_ids = item_df['ItemID'].values
        self.num_items = len(item_df)

        self.content_embeddings = torch.stack([
            torch.tensor(emb, dtype=torch.float32)
            for emb in item_df['embedding']
        ])

        print(f"Loading collaborative embeddings from {collab_embedding_file}...")
        collab_emb_path = self.data_dir / collab_embedding_file
        collab_emb_all = np.load(collab_emb_path)

        self.collab_embeddings = torch.stack([
            torch.tensor(collab_emb_all[item_id], dtype=torch.float32)
            for item_id in self.item_ids
        ])

        # Build item_id -> dataset_index mapping
        self.item_id_to_idx = {int(item_id): idx for idx, item_id in enumerate(self.item_ids)}

        # Load co-occurrence graph from training sequences
        self.cooc_graph = {}
        self.has_cooc = False
        train_seq_path = self.data_dir / train_seq_file
        if train_seq_file and train_seq_path.exists():
            self._build_cooc_graph(train_seq_path)
        else:
            print(f"  No train sequence file found at {train_seq_path}, SACO will be disabled")

        print(f"Dataset loaded: {self.num_items} items")
        print(f"  Content embedding dim: {self.content_embeddings.shape[1]}")
        print(f"  Collab embedding dim: {self.collab_embeddings.shape[1]}")
        if self.has_cooc:
            print(f"  Co-occurrence graph: {len(self.cooc_graph)} items, "
                  f"{sum(len(v) for v in self.cooc_graph.values())} edges")
        else:
            print(f"  Co-occurrence graph: DISABLED")

    def _build_cooc_graph(self, train_seq_path: Path) -> None:
        """
        Build item-level co-occurrence graph from user interaction sequences.

        For each user sequence, all item pairs within a sliding window
        are considered co-occurring (positive pairs for SACO).
        """
        print(f"Building co-occurrence graph from {train_seq_path}...")
        df = pd.read_parquet(train_seq_path)

        self.cooc_graph = defaultdict(list)

        for _, row in df.iterrows():
            seq = list(row['history']) + [row['target']]
            # Only keep items that exist in our embedding set
            seq = [item_id for item_id in seq if item_id in self.item_id_to_idx]

            for i in range(len(seq)):
                for j in range(i + 1, min(i + self.cooc_window + 1, len(seq))):
                    a, b = seq[i], seq[j]
                    if a != b:
                        self.cooc_graph[a].append(b)
                        self.cooc_graph[b].append(a)

        self.cooc_graph = dict(self.cooc_graph)
        self.has_cooc = len(self.cooc_graph) > 0
        print(f"  Co-occurrence graph built: {len(self.cooc_graph)} items with edges")

    def _sample_positive(self, item_id: int) -> int:
        """
        Randomly sample a co-occurring positive item.

        Returns the item itself (identity fallback) for cold items with no co-occurrences.
        The identity pair provides near-zero gradient in SACO InfoNCE,
        effectively skipping the contrastive loss for that anchor.
        """
        cooc_list = self.cooc_graph.get(item_id, [])
        if len(cooc_list) == 0:
            return item_id
        return int(random.choice(cooc_list))

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        anchor_item_id = int(self.item_ids[idx])

        if self.has_cooc:
            pos_item_id = self._sample_positive(anchor_item_id)
            pos_idx = self.item_id_to_idx[pos_item_id]
        else:
            pos_item_id = anchor_item_id
            pos_idx = idx

        return {
            'item_id': anchor_item_id,
            'content_emb': self.content_embeddings[idx],
            'collab_emb': self.collab_embeddings[idx],
            'pos_content_emb': self.content_embeddings[pos_idx],
            'pos_collab_emb': self.collab_embeddings[pos_idx],
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    max_items: Optional[int] = None,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, PRISMDataset]:
    dataset = PRISMDataset(
        data_dir=data_dir,
        max_items=max_items,
        **dataset_kwargs
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader, dataset


def collate_prism_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        'item_id': torch.tensor([item['item_id'] for item in batch]),
        'content_emb': torch.stack([item['content_emb'] for item in batch]),
        'collab_emb': torch.stack([item['collab_emb'] for item in batch]),
        'pos_content_emb': torch.stack([item['pos_content_emb'] for item in batch]),
        'pos_collab_emb': torch.stack([item['pos_collab_emb'] for item in batch]),
    }
