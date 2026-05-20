"""
Multi-Modal Dataset for PRISM Training

Loads and combines:
1. Content embeddings (768D from TIGER-format item_emb.parquet)
2. Collaborative embeddings (64D from LightGCN)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PRISMDataset(Dataset):
    """
    Multi-modal dataset for PRISM training.

    Combines item content embeddings and collaborative embeddings.
    """

    def __init__(
        self,
        data_dir: str,
        embedding_file: str = 'item_emb.parquet',
        collab_embedding_file: str = 'lightgcn/item_embeddings_collab.npy',
        max_items: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)

        print(f"Loading item embeddings from {embedding_file}...")
        item_df = pd.read_parquet(self.data_dir / embedding_file)

        if max_items is not None:
            item_df = item_df.head(max_items)

        self.item_ids = item_df['ItemID'].values
        self.num_items = len(item_df)

        if 'popularity_score' in item_df.columns:
            self.has_popularity = True
            self.popularity_scores = torch.tensor(
                item_df['popularity_score'].values,
                dtype=torch.float32
            )
            print(f"  Popularity scores loaded (mean: {self.popularity_scores.mean():.3f})")
        else:
            self.has_popularity = False
            self.popularity_scores = torch.zeros(self.num_items, dtype=torch.float32)
            print(f"  No popularity_score found, using zeros")

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

        print(f"Dataset loaded: {self.num_items} items")
        print(f"  Content embedding dim: {self.content_embeddings.shape[1]}")
        print(f"  Collab embedding dim: {self.collab_embeddings.shape[1]}")

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'item_id': self.item_ids[idx],
            'content_emb': self.content_embeddings[idx],
            'collab_emb': self.collab_embeddings[idx],
            'popularity_score': self.popularity_scores[idx],
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
        drop_last=False
    )

    return dataloader, dataset


def collate_prism_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    return {
        'item_id': torch.tensor([item['item_id'] for item in batch]),
        'content_emb': torch.stack([item['content_emb'] for item in batch]),
        'collab_emb': torch.stack([item['collab_emb'] for item in batch]),
        'popularity_score': torch.tensor([item['popularity_score'] for item in batch]),
    }
