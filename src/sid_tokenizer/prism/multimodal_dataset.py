"""
Multi-Modal Dataset for PRISM Training

Loads and combines:
1. Content embeddings (768D from attribute_embedding)
2. Collaborative embeddings (64D from LightGCN)
3. Hierarchical tag information (category_tag_ids, category_tag_texts)
4. Tag embeddings (768D for each tag)
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
    
    Combines item content embeddings, collaborative embeddings, and hierarchical tags.
    """
    
    def __init__(
        self,
        data_dir: str,
        embedding_file: str = 'item_emb.parquet',
        collab_embedding_file: str = 'lightgcn/item_embeddings_collab.npy',
        tag_embedding_file: str = 'tag_embeddings.parquet',
        tag_mapping_file: str = 'tag_mapping.npy',
        max_items: Optional[int] = None,
        pad_token_id: int = 0,
        n_layers: int = 4
    ):
        """
        Initialize multi-modal dataset.
        
        Args:
            data_dir: Directory containing all data files
            embedding_file: Parquet file with item embeddings and tags
            collab_embedding_file: Numpy file with collaborative embeddings
            tag_embedding_file: Parquet file with tag embeddings
            tag_mapping_file: Numpy file with tag name to ID mapping
            max_items: Maximum number of items to load (for testing)
            pad_token_id: ID for PAD token
            n_layers: Number of RQ layers to use (excluding L1)
        """
        self.data_dir = Path(data_dir)
        self.pad_token_id = pad_token_id
        self.n_layers = n_layers  # Store n_layers for dynamic processing
        
        # Load item data (content embeddings + tags)
        print(f"Loading item embeddings from {embedding_file}...")
        item_df = pd.read_parquet(self.data_dir / embedding_file)
        
        if max_items is not None:
            item_df = item_df.head(max_items)
        
        self.item_ids = item_df['ItemID'].values
        self.num_items = len(item_df)
        
        # Store item_df for accessing popularity scores
        self.item_df = item_df
        
        # Check if popularity_score exists
        if 'popularity_score' in item_df.columns:
            self.has_popularity = True
            self.popularity_scores = torch.tensor(
                item_df['popularity_score'].values, 
                dtype=torch.float32
            )
            print(f"  ✓ Popularity scores loaded (mean: {self.popularity_scores.mean():.3f})")
        else:
            self.has_popularity = False
            self.popularity_scores = torch.zeros(self.num_items, dtype=torch.float32)
            print(f"  ⚠ No popularity_score found, using zeros")
        
        # Extract content embeddings (attribute_embedding: 768D)
        self.content_embeddings = torch.stack([
            torch.tensor(emb, dtype=torch.float32)
            for emb in item_df['attribute_embedding']
        ])
        
        # Load collaborative embeddings (64D from LightGCN)
        print(f"Loading collaborative embeddings from {collab_embedding_file}...")
        collab_emb_path = self.data_dir / collab_embedding_file
        collab_emb_all = np.load(collab_emb_path)  # (n_items + 1, 64)
        
        # Match collaborative embeddings to items (handle index offset)
        # Assuming ItemID starts from 1, collab embedding index 0 might be padding
        self.collab_embeddings = torch.stack([
            torch.tensor(collab_emb_all[item_id], dtype=torch.float32)
            for item_id in self.item_ids
        ])
        
        # Extract hierarchical tag information
        print("Processing hierarchical tags...")
        self.tag_ids_per_item = []  # List of lists: [[L1, L2, L3, ...], ...]
        self.tag_texts_per_item = []
        self.tag_levels_per_item = []  # Number of valid tags per item
        
        for _, row in item_df.iterrows():
            tag_ids = row['category_tag_ids']
            tag_texts = row['category_tag_texts']
            num_categories = row['num_categories']
            
            # Filter out PAD tokens (ID 0 or text '<PAD>')
            valid_tag_ids = []
            valid_tag_texts = []
            
            for tid, ttext in zip(tag_ids, tag_texts):
                if tid != pad_token_id and ttext != '<PAD>':
                    valid_tag_ids.append(tid)
                    valid_tag_texts.append(ttext)
            
            self.tag_ids_per_item.append(valid_tag_ids)
            self.tag_texts_per_item.append(valid_tag_texts)
            self.tag_levels_per_item.append(len(valid_tag_ids))
        
        # Load tag embeddings
        print(f"Loading tag embeddings from {tag_embedding_file}...")
        tag_emb_df = pd.read_parquet(self.data_dir / tag_embedding_file)
        
        # Create tag_id -> embedding mapping
        self.tag_id_to_embedding = {}
        for _, row in tag_emb_df.iterrows():
            tag_id = row['tag_id']
            tag_emb = torch.tensor(row['tag_embedding'], dtype=torch.float32)
            self.tag_id_to_embedding[tag_id] = tag_emb
        
        # Load tag mapping (tag_text -> tag_id)
        print(f"Loading tag mapping from {tag_mapping_file}...")
        self.tag_mapping = np.load(
            self.data_dir / tag_mapping_file, 
            allow_pickle=True
        ).item()
        
        # Analyze tag statistics
        self._compute_tag_statistics()
        
        # Build tag ID to class index mapping for each layer
        self._build_tag_to_class_mapping()
        
        print(f"✓ Dataset loaded: {self.num_items} items")
        print(f"  Content embedding dim: {self.content_embeddings.shape[1]}")
        print(f"  Collab embedding dim: {self.collab_embeddings.shape[1]}")
        print(f"  Configured n_layers: {self.n_layers} (excluding L1)")
        print(f"  Available tag levels: {self.max_tag_level}")
        print(f"  Tag counts per level: {self.num_tags_per_level}")
    
    def _compute_tag_statistics(self):
        """Compute tag statistics for dataset initialization."""
        # Find maximum tag level (excluding L1 'Beauty')
        self.max_tag_level = max(self.tag_levels_per_item)
        
        # Count unique tags at each level (L2, L3, L4, ...)
        # L1 is always 'Beauty', so we start from level 2
        tag_sets_per_level = [set() for _ in range(self.max_tag_level)]
        
        for tag_ids in self.tag_ids_per_item:
            for level_idx, tag_id in enumerate(tag_ids):
                if level_idx > 0:  # Skip L1 (index 0)
                    tag_sets_per_level[level_idx].add(tag_id)
        
        # Number of unique tags per level (excluding L1)
        self.num_tags_per_level = [
            len(tag_sets_per_level[i]) 
            for i in range(1, self.max_tag_level)  # Start from L2
        ]
        
        # Store as dict for easy access
        self.tag_stats = {
            f'n_L{i+2}': self.num_tags_per_level[i]
            for i in range(len(self.num_tags_per_level))
        }
    
    def _build_tag_to_class_mapping(self):
        """
        Build mapping from global tag IDs to per-layer class indices.
        Class index 0 is reserved for PAD token.
        """
        # Collect unique tag IDs at each level (starting from L2)
        tag_sets_per_level = [set() for _ in range(self.max_tag_level)]
        
        for tag_ids in self.tag_ids_per_item:
            for level_idx, tag_id in enumerate(tag_ids):
                if level_idx > 0:  # Skip L1
                    tag_sets_per_level[level_idx].add(tag_id)
        
        # Create mappings for each layer (L2, L3, L4, ...)
        # Map: global_tag_id -> local_class_index (1-indexed, 0 is PAD)
        self.tag_to_class_maps = []
        
        for level_idx in range(1, self.max_tag_level):  # Start from L2
            unique_tags = sorted(list(tag_sets_per_level[level_idx]))
            
            # Create mapping: tag_id -> class_index (starting from 1, 0 is PAD)
            tag_to_class = {tag_id: idx + 1 for idx, tag_id in enumerate(unique_tags)}
            tag_to_class[self.pad_token_id] = 0  # PAD token maps to 0
            
            self.tag_to_class_maps.append(tag_to_class)
        
        print(f"✓ Tag to class mapping built:")
        for i, mapping in enumerate(self.tag_to_class_maps[:3]):  # Show first 3 layers
            print(f"  Layer {i+2}: {len(mapping)-1} unique tags (+PAD)")
    
    def get_tag_embeddings_per_level(
        self, 
        n_layers: int = 3
    ) -> List[torch.Tensor]:
        """
        Get all unique tag embeddings organized by level, ordered by mapped class indices.
        
        Args:
            n_layers: Number of RQ layers (typically 3)
            
        Returns:
            tag_embeddings_per_layer: List of tensors
                [L2_tags (n_L2, 768), L3_tags (n_L3, 768), L4_tags (n_L4, 768)]
                Each tensor is ordered by class index (1 to n_classes)
        """
        tag_embeddings_per_layer = []
        
        for layer_idx in range(min(n_layers, len(self.tag_to_class_maps))):
            # Get the mapping for this layer
            tag_to_class = self.tag_to_class_maps[layer_idx]
            
            # Create list of (class_idx, tag_id) pairs, excluding PAD
            tag_class_pairs = [
                (class_idx, tag_id) 
                for tag_id, class_idx in tag_to_class.items() 
                if class_idx > 0  # Exclude PAD (class_idx=0)
            ]
            
            # Sort by class index
            tag_class_pairs.sort(key=lambda x: x[0])
            
            # Get embeddings in sorted order
            if len(tag_class_pairs) > 0:
                tag_embs = torch.stack([
                    self.tag_id_to_embedding[tag_id]
                    for class_idx, tag_id in tag_class_pairs
                    if tag_id in self.tag_id_to_embedding
                ])
                tag_embeddings_per_layer.append(tag_embs)
            else:
                # Empty tensor if no tags at this level
                tag_embeddings_per_layer.append(torch.empty(0, 768))
        
        # Fill remaining layers with empty tensors if needed
        while len(tag_embeddings_per_layer) < n_layers:
            tag_embeddings_per_layer.append(torch.empty(0, 768))
        
        return tag_embeddings_per_layer
    
    def __len__(self) -> int:
        return self.num_items
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item with all modalities.
        
        Returns:
            data_dict: Dictionary containing:
                - item_id: Item identifier
                - content_emb: Content embedding (768D)
                - collab_emb: Collaborative embedding (64D)
                - tag_ids: Mapped class indices at each level (padded to n_layers)
                - tag_mask: Binary mask (1 for valid tags, 0 for padding)
                - num_tags: Number of valid tags
        """
        # Get embeddings
        content_emb = self.content_embeddings[idx]
        collab_emb = self.collab_embeddings[idx]
        
        # Get tags (excluding L1 'Beauty')
        global_tag_ids = self.tag_ids_per_item[idx][1:]  # Skip L1
        num_tags = len(global_tag_ids)
        
        # Use configured n_layers (dynamic, not hardcoded)
        # Truncate if more layers available than needed, pad if less
        if num_tags > self.n_layers:
            # Truncate to n_layers
            global_tag_ids = global_tag_ids[:self.n_layers]
            num_tags = self.n_layers
        
        # Convert global tag IDs to per-layer class indices
        mapped_tag_ids = []
        for layer_idx, global_tag_id in enumerate(global_tag_ids):
            if layer_idx < len(self.tag_to_class_maps):
                # Map global ID to local class index
                local_class_idx = self.tag_to_class_maps[layer_idx].get(
                    global_tag_id, 
                    0  # Default to PAD if not found
                )
                mapped_tag_ids.append(local_class_idx)
            else:
                mapped_tag_ids.append(0)  # PAD for extra layers
        
        # Pad to n_layers
        padded_tag_ids = mapped_tag_ids + [0] * (self.n_layers - len(mapped_tag_ids))
        tag_mask = [1] * num_tags + [0] * (self.n_layers - num_tags)
        
        # Get popularity score
        popularity_score = self.popularity_scores[idx]
        
        return {
            'item_id': self.item_ids[idx],
            'content_emb': content_emb,
            'collab_emb': collab_emb,
            'tag_ids': torch.tensor(padded_tag_ids, dtype=torch.long),
            'tag_mask': torch.tensor(tag_mask, dtype=torch.float32),
            'num_tags': num_tags,
            'popularity_score': popularity_score
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    num_workers: int = 4,
    max_items: Optional[int] = None,
    **dataset_kwargs
) -> Tuple[torch.utils.data.DataLoader, PRISMDataset]:
    """
    Create dataloader for PRISM training.
    
    Args:
        data_dir: Directory containing dataset files
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_items: Maximum number of items (for testing)
        **dataset_kwargs: Additional arguments for PRISMDataset
        
    Returns:
        dataloader: DataLoader for training
        dataset: Dataset instance (for accessing metadata)
    """
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


# Utility function to collate variable-length sequences
def collate_prism_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching PRISM data.
    
    Args:
        batch: List of data dictionaries from __getitem__
        
    Returns:
        batched_data: Dictionary with batched tensors
    """
    batched_data = {
        'item_id': torch.tensor([item['item_id'] for item in batch]),
        'content_emb': torch.stack([item['content_emb'] for item in batch]),
        'collab_emb': torch.stack([item['collab_emb'] for item in batch]),
        'tag_ids': torch.stack([item['tag_ids'] for item in batch]),
        'tag_mask': torch.stack([item['tag_mask'] for item in batch]),
        'num_tags': torch.tensor([item['num_tags'] for item in batch])
    }
    
    return batched_data

