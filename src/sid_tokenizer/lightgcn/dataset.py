"""
Dataset class for LightGCN training on Beauty dataset
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class BeautyDataset(Dataset):
    """
    Dataset for LightGCN training on Beauty dataset.
    Builds user-item interaction graph from parquet files.
    """
    
    def __init__(self, data_dir: str, use_val: bool = False):
        """
        Args:
            data_dir: Path to the Beauty dataset directory
            use_val: Whether to include validation set for training (default: False)
        """
        self.data_dir = Path(data_dir)
        self.use_val = use_val
        
        # Load data
        self._load_data()
        
        # Build interaction matrix
        self._build_interaction_matrix()
        
        # Build adjacency graph for LightGCN
        self._build_sparse_graph()
        
        logger.info(f"Dataset loaded: {self.n_users} users, {self.n_items} items, "
                   f"{self.n_train} interactions")
    
    def _load_data(self):
        """Load train (and optionally validation) data from parquet files"""
        train_df = pd.read_parquet(self.data_dir / 'train.parquet')
        
        if self.use_val:
            val_df = pd.read_parquet(self.data_dir / 'valid.parquet')
            combined_df = pd.concat([train_df, val_df], ignore_index=True)
        else:
            combined_df = train_df
        
        # Extract all user-item interactions
        # Each row has: user, history (list of items), target
        user_item_pairs = []
        
        for idx, row in combined_df.iterrows():
            user = row['user']
            # Add interactions from history
            for item in row['history']:
                user_item_pairs.append((user, item))
            # Add target interaction
            user_item_pairs.append((user, row['target']))
        
        # Remove duplicates (same user-item pair might appear multiple times)
        user_item_pairs = list(set(user_item_pairs))
        
        # Convert to arrays
        self.train_user = np.array([pair[0] for pair in user_item_pairs])
        self.train_item = np.array([pair[1] for pair in user_item_pairs])
        
        # Get unique users and items
        self.n_users = int(self.train_user.max()) + 1
        self.n_items = int(self.train_item.max()) + 1
        self.n_train = len(user_item_pairs)
        
        logger.info(f"Loaded {len(user_item_pairs)} unique user-item interactions")
    
    def _build_interaction_matrix(self):
        """Build user-item interaction matrix in sparse format"""
        # Create a binary interaction matrix (1 if user interacted with item)
        data = np.ones(len(self.train_user))
        self.interaction_matrix = sp.csr_matrix(
            (data, (self.train_user, self.train_item)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32
        )
        
        # Create dict: user -> list of positive items
        self.user_pos_items = {}
        for user, item in zip(self.train_user, self.train_item):
            if user not in self.user_pos_items:
                self.user_pos_items[user] = []
            self.user_pos_items[user].append(item)
        
        # Convert to numpy arrays for faster access
        for user in self.user_pos_items:
            self.user_pos_items[user] = np.array(self.user_pos_items[user])
    
    def _build_sparse_graph(self):
        """
        Build the adjacency matrix for LightGCN.
        
        The graph structure is:
        [0,      R    ]
        [R^T,    0    ]
        
        where R is the user-item interaction matrix.
        This creates a bipartite graph between users and items.
        """
        n_nodes = self.n_users + self.n_items
        
        # Create adjacency matrix entries
        # User-Item edges
        user_indices = self.train_user
        item_indices = self.train_item + self.n_users  # Offset item IDs
        
        # Create edges in both directions (undirected graph)
        row_indices = np.concatenate([user_indices, item_indices])
        col_indices = np.concatenate([item_indices, user_indices])
        
        # Create adjacency matrix
        data = np.ones(len(row_indices), dtype=np.float32)
        adj_mat = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32
        )
        
        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        # This is the standard normalization for GCN
        rowsum = np.array(adj_mat.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # Normalized adjacency matrix
        norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt)
        
        # Convert to COO format for PyTorch
        norm_adj = norm_adj.tocoo()
        
        # Convert to PyTorch sparse tensor
        indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        
        self.graph = torch.sparse.FloatTensor(indices, values, shape)
        
        logger.info(f"Built sparse graph: {n_nodes} nodes, {len(norm_adj.data)} edges")
    
    def get_sparse_graph(self) -> torch.sparse.FloatTensor:
        """Return the normalized adjacency matrix as PyTorch sparse tensor"""
        return self.graph
    
    def get_user_pos_items(self, users: np.ndarray) -> Dict[int, np.ndarray]:
        """Get positive items for given users"""
        return {user: self.user_pos_items.get(user, np.array([])) for user in users}
    
    def __len__(self) -> int:
        """Return number of training interactions"""
        return self.n_train
    
    def __getitem__(self, idx: int) -> Tuple[int, int]:
        """Get a user-item pair"""
        return self.train_user[idx], self.train_item[idx]


class BPRSampler:
    """
    Bayesian Personalized Ranking sampler for training.
    Samples (user, positive_item, negative_item) triplets.
    """
    
    def __init__(self, dataset: BeautyDataset, batch_size: int = 2048, shuffle: bool = True):
        """
        Args:
            dataset: BeautyDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle training data
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.n_users = dataset.n_users
        self.n_items = dataset.n_items
        self.train_user = dataset.train_user
        self.train_item = dataset.train_item
        self.user_pos_items = dataset.user_pos_items
        
        # Create training indices
        self.n_samples = len(dataset)
        self.indices = np.arange(self.n_samples)
    
    def __len__(self) -> int:
        """Number of batches"""
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches"""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        for start_idx in range(0, self.n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.n_samples)
            batch_indices = self.indices[start_idx:end_idx]
            
            # Get batch users and positive items
            batch_users = self.train_user[batch_indices]
            batch_pos_items = self.train_item[batch_indices]
            
            # Sample negative items
            batch_neg_items = self._sample_negative_items(batch_users)
            
            yield (
                torch.LongTensor(batch_users),
                torch.LongTensor(batch_pos_items),
                torch.LongTensor(batch_neg_items)
            )
    
    def _sample_negative_items(self, users: np.ndarray) -> np.ndarray:
        """Sample negative items for each user (items they haven't interacted with)"""
        neg_items = np.zeros(len(users), dtype=np.int64)
        
        for i, user in enumerate(users):
            # Get positive items for this user
            pos_items = self.user_pos_items.get(user, np.array([]))
            pos_items_set = set(pos_items)
            
            # Sample a negative item
            while True:
                neg_item = np.random.randint(0, self.n_items)
                if neg_item not in pos_items_set:
                    neg_items[i] = neg_item
                    break
        
        return neg_items

