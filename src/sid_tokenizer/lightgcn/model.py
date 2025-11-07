"""
LightGCN model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    
    Paper: https://arxiv.org/abs/2002.02126
    
    Key idea: Remove feature transformation and nonlinear activation in GCN,
    only keep neighbor aggregation for collaborative filtering.
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        n_layers: int = 3,
        dropout: bool = False,
        keep_prob: float = 0.6
    ):
        """
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of embeddings
            n_layers: Number of graph convolution layers
            dropout: Whether to use dropout during training
            keep_prob: Keep probability for dropout
        """
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.keep_prob = keep_prob
        
        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialize with normal distribution
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Graph will be set later
        self.graph = None
        
        logger.info(f"LightGCN initialized: {n_users} users, {n_items} items, "
                   f"{embedding_dim} dim, {n_layers} layers")
    
    def set_graph(self, graph: torch.sparse.FloatTensor):
        """Set the normalized adjacency matrix"""
        self.graph = graph
    
    def _dropout_sparse(self, x: torch.sparse.FloatTensor, keep_prob: float) -> torch.sparse.FloatTensor:
        """Apply dropout to sparse tensor"""
        size = x.size()
        index = x.indices().t()
        values = x.values()
        
        # Random mask
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        
        # Apply mask
        index = index[random_index]
        values = values[random_index] / keep_prob
        
        # Reconstruct sparse tensor
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def compute_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Graph convolution to compute final user and item embeddings.
        
        Returns:
            user_embeddings: Final user embeddings [n_users, embedding_dim]
            item_embeddings: Final item embeddings [n_items, embedding_dim]
        """
        # Get initial embeddings
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        
        # Concatenate user and item embeddings
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        
        # Store embeddings from each layer
        embs = [all_emb]
        
        # Apply dropout if enabled
        if self.dropout and self.training:
            graph = self._dropout_sparse(self.graph, self.keep_prob)
        else:
            graph = self.graph
        
        # Graph convolution layers
        for layer in range(self.n_layers):
            # Message passing: aggregate neighbor embeddings
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        
        # Stack embeddings from all layers
        embs = torch.stack(embs, dim=1)
        
        # Average pooling across layers
        light_out = torch.mean(embs, dim=1)
        
        # Split back to user and item embeddings
        users, items = torch.split(light_out, [self.n_users, self.n_items], dim=0)
        
        return users, items
    
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute scores for user-item pairs.
        
        Args:
            users: User indices [batch_size]
            items: Item indices [batch_size]
        
        Returns:
            scores: Predicted scores [batch_size]
        """
        # Get final embeddings
        all_users, all_items = self.compute_embeddings()
        
        # Get embeddings for batch
        users_emb = all_users[users]
        items_emb = all_items[items]
        
        # Compute inner product
        scores = torch.sum(users_emb * items_emb, dim=1)
        
        return scores
    
    def get_user_rating(self, users: torch.Tensor) -> torch.Tensor:
        """
        Get predicted ratings for all items for given users.
        
        Args:
            users: User indices [batch_size]
        
        Returns:
            ratings: Predicted ratings [batch_size, n_items]
        """
        all_users, all_items = self.compute_embeddings()
        users_emb = all_users[users]
        items_emb = all_items
        
        # Matrix multiplication for all items
        ratings = torch.matmul(users_emb, items_emb.t())
        
        return ratings
    
    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        reg_weight: float = 1e-4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bayesian Personalized Ranking loss.
        
        Args:
            users: User indices [batch_size]
            pos_items: Positive item indices [batch_size]
            neg_items: Negative item indices [batch_size]
            reg_weight: L2 regularization weight
        
        Returns:
            bpr_loss: BPR loss
            reg_loss: Regularization loss
        """
        # Get final embeddings after graph convolution
        all_users, all_items = self.compute_embeddings()
        
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        # Get initial embeddings (for regularization)
        users_emb_0 = self.user_embedding(users)
        pos_emb_0 = self.item_embedding(pos_items)
        neg_emb_0 = self.item_embedding(neg_items)
        
        # BPR loss: maximize difference between positive and negative scores
        pos_scores = torch.sum(users_emb * pos_emb, dim=1)
        neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))
        
        # L2 regularization on initial embeddings
        reg_loss = (1/2) * (
            users_emb_0.norm(2).pow(2) +
            pos_emb_0.norm(2).pow(2) +
            neg_emb_0.norm(2).pow(2)
        ) / float(len(users))
        
        return bpr_loss, reg_loss
    
    def get_item_embeddings(self) -> torch.Tensor:
        """
        Get final item embeddings after graph convolution.
        This is used for extracting collaborative embeddings in Stage 0.
        
        Returns:
            item_embeddings: [n_items, embedding_dim]
        """
        with torch.no_grad():
            _, item_embeddings = self.compute_embeddings()
        return item_embeddings
    
    def get_user_embeddings(self) -> torch.Tensor:
        """
        Get final user embeddings after graph convolution.
        
        Returns:
            user_embeddings: [n_users, embedding_dim]
        """
        with torch.no_grad():
            user_embeddings, _ = self.compute_embeddings()
        return user_embeddings

