"""
Embedding Space Geometry Analysis: PRISM vs TIGER

This script compares the embedding space geometry of PRISM and TIGER models
by analyzing:
1. Intra-Category Distance 
2. Inter-Category Distance 
3. Silhouette Score 
4. t-SNE/UMAP Visualization
"""

import argparse
import json
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Publication-quality settings - SAME as codebook comparison
# Configure Linux Libertine font
import matplotlib.font_manager as fm
import os

# Add Linux Libertine fonts from the opentype directory
libertine_font_dir = '/home/fangdengzhao/Fonts/libertine/opentype'
libertine_fonts = [
    f'{libertine_font_dir}/LinLibertine_R.otf',      # Regular
    f'{libertine_font_dir}/LinLibertine_RI.otf',     # Italic
    f'{libertine_font_dir}/LinLibertine_RB.otf',     # Bold
    f'{libertine_font_dir}/LinLibertine_RBI.otf',    # Bold Italic
]

for font_file in libertine_fonts:
    if os.path.exists(font_file):
        fm.fontManager.addfont(font_file)

plt.rcParams.update({
    'font.family': 'Linux Libertine O',
    'font.weight': 'normal',  # Ensure no bold
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Linux Libertine O',
    'mathtext.it': 'Linux Libertine O:italic',
    'mathtext.bf': 'Linux Libertine O:bold',
    'font.size': 12,           # Base font size for publication
    'axes.labelsize': 12,      # Axis labels
    'axes.titlesize': 14,      # Subplot titles
    'xtick.labelsize': 11,     # X-axis tick labels
    'ytick.labelsize': 11,     # Y-axis tick labels
    'legend.fontsize': 7,     # Legend text
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Import model modules
import sys
sys.path.append('.')
from src.recommender.prism.model import TIGER as PRISM  # PRISM is TIGER with enhancements
from src.recommender.prism.config import get_config as get_prism_config
from src.recommender.TIGER.model import TIGER
from src.recommender.TIGER.config import get_config as get_tiger_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingAnalyzer:
    """Analyzer for comparing embedding space geometry."""
    
    def __init__(
        self,
        prism_checkpoint: str,
        tiger_checkpoint: str,
        dataset_name: str,
        device: str = 'cuda'
    ):
        """Initialize the analyzer.
        
        Args:
            prism_checkpoint: Path to PRISM model checkpoint
            tiger_checkpoint: Path to TIGER model checkpoint
            dataset_name: Name of dataset (beauty, sports, toys, cds)
            device: Device to use for computation
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load models
        logger.info("Loading PRISM model...")
        self.prism_model = self._load_prism_model(prism_checkpoint, dataset_name)
        
        logger.info("Loading TIGER model...")
        self.tiger_model = self._load_tiger_model(tiger_checkpoint, dataset_name)
        
    def _load_prism_model(self, checkpoint_path: str, dataset_name: str) -> PRISM:
        """Load PRISM model from checkpoint."""
        # Load checkpoint first to get the saved config
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Try to get config from checkpoint, otherwise use default
        if 'config' in checkpoint:
            config = checkpoint['config']
            logger.info("Loaded config from checkpoint")
        else:
            # Infer model type from checkpoint structure
            config = get_prism_config(dataset_name)
            
            # Try to infer d_model from embedding shape
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            if 'model.shared.weight' in state_dict:
                vocab_size, d_model = state_dict['model.shared.weight'].shape
                logger.info(f"Inferred from checkpoint: vocab_size={vocab_size}, d_model={d_model}")
                
                # Update config with inferred values
                config['model'].set_vocab_size(vocab_size)
                config['model'].d_model = d_model
                
                # Adjust other dimensions proportionally
                if d_model == 128:
                    config['model'].d_ff = 1024
                    config['model'].num_heads = 6
                    config['model'].d_kv = 64
                elif d_model == 64:
                    config['model'].d_ff = 1024
                    config['model'].num_heads = 6
                    config['model'].d_kv = 64
        
        model = PRISM(config['model'], config['training'])
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out incompatible codebook_predictor keys (old format vs new format)
        # The codebook_predictor is not needed for embedding analysis
        filtered_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if 'codebook_predictor' in key:
                skipped_keys.append(key)
            else:
                filtered_state_dict[key] = value
        
        if skipped_keys:
            logger.info(f"Skipped {len(skipped_keys)} codebook_predictor keys (not needed for embedding analysis)")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        logger.info(f"Loaded PRISM model: d_model={model.config.d_model}, vocab_size={model.config.vocab_size}")
        return model
    
    def _load_tiger_model(self, checkpoint_path: str, dataset_name: str) -> TIGER:
        """Load TIGER model from checkpoint."""
        # Load checkpoint first to get the saved config
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Try to get config from checkpoint, otherwise use default
        if 'config' in checkpoint:
            config = checkpoint['config']
            logger.info("Loaded config from checkpoint")
        else:
            # Infer model type from checkpoint structure
            config = get_tiger_config(dataset_name)
            
            # Try to infer d_model from embedding shape
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            if 'model.shared.weight' in state_dict:
                vocab_size, d_model = state_dict['model.shared.weight'].shape
                logger.info(f"Inferred from checkpoint: vocab_size={vocab_size}, d_model={d_model}")
                
                # Update config with inferred values
                config['model'].set_vocab_size(vocab_size)
                config['model'].d_model = d_model
                
                # Adjust other dimensions proportionally
                if d_model == 128:
                    config['model'].d_ff = 1024
                    config['model'].num_heads = 6
                    config['model'].d_kv = 64
                elif d_model == 64:
                    config['model'].d_ff = 1024
                    config['model'].num_heads = 6
                    config['model'].d_kv = 64
        
        model = TIGER(config['model'])
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out incompatible codebook_predictor keys (old format vs new format)
        # The codebook_predictor is not needed for embedding analysis
        filtered_state_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if 'codebook_predictor' in key:
                skipped_keys.append(key)
            else:
                filtered_state_dict[key] = value
        
        if skipped_keys:
            logger.info(f"Skipped {len(skipped_keys)} codebook_predictor keys (not needed for embedding analysis)")
        
        model.load_state_dict(filtered_state_dict, strict=False)
        
        model.to(self.device)
        model.eval()
        logger.info(f"Loaded TIGER model: d_model={model.config.d_model}, vocab_size={model.config.vocab_size}")
        return model
    
    def _load_metadata(self, meta_file: str) -> Dict:
        """Load item metadata from JSON file.
        
        Handles both standard JSON and Python dict format (with single quotes).
        """
        metadata = {}
        
        if meta_file.endswith('.gz'):
            import gzip
            open_func = lambda f: gzip.open(f, 'rt', encoding='utf-8')
        else:
            open_func = lambda f: open(f, 'r', encoding='utf-8')
        
        with open_func(meta_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Try standard JSON first
                    item = json.loads(line)
                except json.JSONDecodeError:
                    try:
                        # Try Python literal eval for single-quoted dicts
                        import ast
                        item = ast.literal_eval(line)
                    except (ValueError, SyntaxError) as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue
                
                if 'asin' in item:
                    metadata[item['asin']] = item
        
        logger.info(f"Loaded metadata for {len(metadata)} items")
        return metadata
    
    def _extract_categories(self) -> Dict[str, str]:
        """Extract category information for each item.
        
        Returns:
            Dictionary mapping item_id to category string
        """
        item_to_category = {}
        
        for item_id, meta in self.item_metadata.items():
            if 'categories' in meta and len(meta['categories']) > 0:
                # Use the first category path
                category_path = meta['categories'][0]
                # Use the second level category (more specific than top-level)
                if len(category_path) >= 2:
                    category = category_path[1]
                else:
                    category = category_path[0]
                item_to_category[item_id] = category
        
        return item_to_category
    
    def extract_embeddings(
        self,
        item_ids: List[str],
        semantic_mapper,
        content_embs_dict: Dict,
        collab_embs_dict: Dict,
        codebook_embs_dict: Dict,
        model_type: str = 'prism'
    ) -> torch.Tensor:
        """Extract embeddings for given items.
        
        For PRISM: Extract MOE-fused embeddings (after fusion, before T5 encoder)
        For TIGER: Extract pure ID embeddings (from embedding layer)
        
        Args:
            item_ids: List of item IDs (as strings)
            semantic_mapper: SemanticIDMapper instance
            content_embs_dict: Dictionary of content embeddings
            collab_embs_dict: Dictionary of collaborative embeddings
            model_type: 'prism' or 'tiger'
        
        Returns:
            Tensor of shape (num_items, embedding_dim)
        """
        model = self.prism_model if model_type == 'prism' else self.tiger_model
        
        embeddings = []
        valid_item_ids = []
        
        with torch.no_grad():
            for item_id in tqdm(item_ids, desc=f"Extracting {model_type.upper()} embeddings"):
                # Convert item_id to semantic token IDs
                try:
                    # Convert to int first
                    item_id_int = int(item_id)
                    
                    # Get semantic codes (already with offset applied by SemanticIDMapper)
                    token_ids = semantic_mapper.get_codes(item_id_int)
                    if token_ids is None or token_ids == semantic_mapper.pad_codes:
                        continue
                    
                    # Convert to tensor
                    token_ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                    
                    # Get ID embeddings
                    id_emb = model.model.get_input_embeddings()(token_ids_tensor)  # (1, num_layers, d_model)
                    
                    if model_type == 'prism' and model.use_multimodal_fusion:
                        # For PRISM: apply fusion to get final embedding
                        # Get content and collab embeddings
                        if item_id_int in content_embs_dict and item_id_int in collab_embs_dict:
                            content_emb = torch.tensor(
                                content_embs_dict[item_id_int],
                                dtype=torch.float32,
                                device=self.device
                            ).unsqueeze(0).unsqueeze(0)  # (1, 1, 768)
                            
                            collab_emb = torch.tensor(
                                collab_embs_dict[item_id_int],
                                dtype=torch.float32,
                                device=self.device
                            ).unsqueeze(0).unsqueeze(0)  # (1, 1, 64)
                            
                            # Broadcast to match token sequence length
                            num_tokens = id_emb.shape[1]
                            content_emb = content_emb.expand(-1, num_tokens, -1)
                            collab_emb = collab_emb.expand(-1, num_tokens, -1)
                            
                            # Get codebook embedding if available
                            codebook_emb = None
                            if item_id_int in codebook_embs_dict:
                                # codebook_embs_dict[item_id]: (n_layers, latent_dim)
                                codebook_vec = codebook_embs_dict[item_id_int]
                                # Shape: (n_layers, latent_dim), e.g., (3, 32)
                                
                                # We need (1, num_tokens, latent_dim)
                                codebook_emb = torch.tensor(
                                    codebook_vec,
                                    dtype=torch.float32,
                                    device=self.device
                                ).unsqueeze(0)  # (1, n_layers, latent_dim)
                                
                                # If num_tokens != n_layers, we need to handle it
                                if codebook_emb.shape[1] != num_tokens:
                                    logger.warning(f"Codebook layers ({codebook_emb.shape[1]}) != num_tokens ({num_tokens}), skipping codebook")
                                    codebook_emb = None
                            
                            # Apply fusion
                            from src.recommender.prism.moe_fusion import MoEFusion
                            if isinstance(model.fusion_module, MoEFusion):
                                fused_emb, _ = model.fusion_module(
                                    id_emb, content_emb, collab_emb,
                                    codebook_emb=codebook_emb,
                                    return_stats=False
                                )
                            else:
                                fused_emb = model.fusion_module(
                                    id_emb, content_emb, collab_emb
                                )
                            
                            # Average over sequence dimension to get item-level embedding
                            item_emb = fused_emb.mean(dim=1).squeeze(0)  # (d_model,)
                        else:
                            # Fallback to ID embedding if content/collab not available
                            item_emb = id_emb.mean(dim=1).squeeze(0)
                    else:
                        # For TIGER or PRISM without fusion: use pure ID embedding
                        # Average over sequence dimension to get item-level embedding
                        item_emb = id_emb.mean(dim=1).squeeze(0)  # (d_model,)
                    
                    embeddings.append(item_emb)
                    valid_item_ids.append(item_id)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for item {item_id}: {e}")
                    continue
        
        if len(embeddings) == 0:
            raise ValueError("No valid embeddings extracted!")
        
        logger.info(f"Extracted {len(embeddings)} valid embeddings out of {len(item_ids)} items")
        return torch.stack(embeddings), valid_item_ids
    
    def assign_categories_by_semantic_prefix(
        self,
        item_ids: List[str],
        semantic_mapper,
        debug: bool = False
    ) -> Dict[str, str]:
        """Assign categories based on semantic ID prefixes.
        
        Logic:
        - First code different → different major category
        - First two codes same, last different → same major category, different items
        
        Args:
            item_ids: List of item IDs
            semantic_mapper: SemanticIDMapper instance
            debug: If True, print detailed category information
        
        Returns:
            Dictionary mapping item_id to category label
        """
        item_to_category = {}
        category_examples = {}  # For debugging: store examples per category
        
        for item_id in item_ids:
            try:
                item_id_int = int(item_id)
                codes = semantic_mapper.get_codes(item_id_int)
                
                if codes is None or codes == semantic_mapper.pad_codes:
                    continue
                
                # Use first code as major category
                # Format: "Cat_X" where X is the first code
                if len(codes) >= 1:
                    category = f"Cat_{codes[0]}"
                    item_to_category[item_id] = category
                    
                    # Store examples for debugging
                    if debug and category not in category_examples:
                        category_examples[category] = {
                            'item_id': item_id,
                            'codes': codes
                        }
                    
            except Exception as e:
                logger.warning(f"Failed to assign category for item {item_id}: {e}")
                continue
        
        # Print debug information
        if debug and category_examples:
            logger.info("\n" + "="*80)
            logger.info("CATEGORY ASSIGNMENT DEBUG INFO")
            logger.info("="*80)
            for cat in sorted(category_examples.keys()):
                example = category_examples[cat]
                logger.info(f"{cat}:")
                logger.info(f"  Example item_id: {example['item_id']}")
                logger.info(f"  Semantic codes: {example['codes']}")
            logger.info("="*80 + "\n")
        
        return item_to_category
    
    def compute_embedding_similarity(
        self,
        embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """Compute embedding quality metrics without category information.
        
        Args:
            embeddings: Tensor of shape (num_items, embedding_dim)
        
        Returns:
            Dictionary with metrics
        """
        # Compute pairwise cosine similarities
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Remove diagonal (self-similarity)
        mask = ~torch.eye(len(embeddings), dtype=torch.bool, device=embeddings.device)
        similarities = similarity_matrix[mask]
        
        # Compute statistics
        mean_sim = similarities.mean().item()
        std_sim = similarities.std().item()
        max_sim = similarities.max().item()
        min_sim = similarities.min().item()
        
        # Compute diversity (lower similarity = higher diversity)
        diversity = 1 - mean_sim
        
        return {
            'mean_similarity': mean_sim,
            'std_similarity': std_sim,
            'max_similarity': max_sim,
            'min_similarity': min_sim,
            'diversity': diversity
        }
    
    def visualize_embeddings(
        self,
        prism_embeddings: torch.Tensor,
        tiger_embeddings: torch.Tensor,
        item_ids: List[str],
        semantic_mapper,
        output_dir: str,
        method: str = 'tsne',
        item_metadata: Dict = None,
        prism_semantic_mapper=None,
        tiger_semantic_mapper=None
    ):
        """Visualize embeddings using t-SNE or UMAP with semantic ID-based categories.
        
        Generates a SINGLE publication-quality figure with:
        - Legend on top (2 rows for 10 categories)
        - TIGER Recommendation (left) and PRISM Recommendation (right)
        
        Output formats: PNG, PDF, SVG
        
        Args:
            prism_embeddings: PRISM embeddings
            tiger_embeddings: TIGER embeddings
            item_ids: List of item IDs
            semantic_mapper: SemanticIDMapper for category assignment
            output_dir: Directory to save the visualizations
            method: 'tsne' or 'umap'
        """
        # Prepare data
        prism_np = prism_embeddings.cpu().numpy()
        tiger_np = tiger_embeddings.cpu().numpy()
        
        # Assign categories based on semantic ID prefixes (using PRISM's mapper)
        logger.info("Assigning categories based on PRISM semantic ID prefixes...")
        logger.info("(This ensures consistent category labels for both models)")
        item_to_category = self.assign_categories_by_semantic_prefix(
            item_ids, semantic_mapper, debug=False
        )
        
        # Get category labels for valid items
        labels = []
        valid_indices = []
        
        for i, item_id in enumerate(item_ids):
            if item_id in item_to_category:
                labels.append(item_to_category[item_id])
                valid_indices.append(i)
        
        prism_np = prism_np[valid_indices]
        tiger_np = tiger_np[valid_indices]
        
        # Select top 10 categories by frequency
        from collections import Counter
        category_counts = Counter(labels)
        top_categories = [cat for cat, _ in category_counts.most_common(10)]
        
        # Filter to only include top categories
        top_indices = [i for i, label in enumerate(labels) if label in top_categories]
        prism_np = prism_np[top_indices]
        tiger_np = tiger_np[top_indices]
        labels = [labels[i] for i in top_indices]
        final_item_ids = [item_ids[valid_indices[i]] for i in top_indices]
        
        logger.info(f"Visualizing {len(top_categories)} most frequent semantic categories")
        logger.info(f"Top categories: {top_categories}")
        
        # Extended color-blind friendly palette for 10 categories
        UNIFIED_PALETTE = [
            '#009E73',  # Teal
            '#D55E00',  # Orange
            '#0072B2',  # Blue
            '#F0E442',  # Yellow
            '#CC79A7',  # Pink
            '#56B4E9',  # Sky Blue
            '#E69F00',  # Orange-Yellow
            '#8B4513',  # Saddle Brown
            '#2E8B57',  # Sea Green
            '#9370DB',  # Medium Purple
        ]
        palette = UNIFIED_PALETTE[:len(top_categories)]
        
        # Map categories to indices and colors
        category_to_idx = {cat: i for i, cat in enumerate(top_categories)}
        category_to_color = {cat: palette[i] for i, cat in enumerate(top_categories)}
        
        # Dimensionality reduction
        logger.info(f"Performing {method.upper()} dimensionality reduction...")
        
        if method == 'tsne':
            prism_reducer = TSNE(
                n_components=2, random_state=42, 
                perplexity=min(60, len(prism_np)-1),
                max_iter=3000, init='pca', learning_rate='auto'
            )
            tiger_reducer = TSNE(
                n_components=2, random_state=42, 
                perplexity=min(60, len(tiger_np)-1),
                max_iter=3000, init='pca', learning_rate='auto'
            )
        else:
            prism_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(prism_np)-1), min_dist=0.1)
            tiger_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(tiger_np)-1), min_dist=0.1)
        
        logger.info(f"Performing {method.upper()} for PRISM...")
        prism_2d = prism_reducer.fit_transform(prism_np)
        
        logger.info(f"Performing {method.upper()} for TIGER...")
        tiger_2d = tiger_reducer.fit_transform(tiger_np)
        
        # Remove outliers
        def filter_outliers(embeddings_2d, percentile=1.0):
            x_min, x_max = np.percentile(embeddings_2d[:, 0], [percentile, 100-percentile])
            y_min, y_max = np.percentile(embeddings_2d[:, 1], [percentile, 100-percentile])
            mask = (
                (embeddings_2d[:, 0] >= x_min) & (embeddings_2d[:, 0] <= x_max) &
                (embeddings_2d[:, 1] >= y_min) & (embeddings_2d[:, 1] <= y_max)
            )
            return mask
        
        tiger_mask = filter_outliers(tiger_2d, percentile=1.0)
        prism_mask = filter_outliers(prism_2d, percentile=1.0)
        valid_mask = tiger_mask & prism_mask
        
        tiger_2d = tiger_2d[valid_mask]
        prism_2d = prism_2d[valid_mask]
        labels = [labels[i] for i, v in enumerate(valid_mask) if v]
        
        logger.info(f"Filtered {(~valid_mask).sum()} outliers ({(~valid_mask).sum()/len(valid_mask)*100:.1f}%)")
        
        # Convert labels to numeric indices
        label_indices = np.array([category_to_idx[label] for label in labels])
        
        # Visual parameters - optimized for publication quality
        point_size = 12
        edge_width = 0.3
        point_alpha = 0.7
        
        logger.info(f"Plotting {len(tiger_2d)} points...")
        
        # Create combined figure with GridSpec
        # CRITICAL: Use EXACT same figsize as codebook to ensure identical scaling in LaTeX
        from matplotlib.gridspec import GridSpec
        from matplotlib.lines import Line2D
        
        # figsize=(4.5, 2.2) - Reduced height since no legend
        fig = plt.figure(figsize=(4.5, 2.2), dpi=300)
        fig.patch.set_facecolor('white')
        
        # GridSpec: 1 row, 2 columns (TIGER + PRISM) - no legend row
        gs = GridSpec(1, 2, figure=fig, wspace=0.12)
        
        # TIGER (left) and PRISM (right)
        ax_tiger = fig.add_subplot(gs[0, 0])
        ax_prism = fig.add_subplot(gs[0, 1])
        
        def plot_scatter(ax, embeddings_2d, labels_list, title):
            ax.set_facecolor('white')
            
            # Shuffle to prevent z-order occlusion
            indices = np.arange(len(embeddings_2d))
            np.random.seed(42)
            np.random.shuffle(indices)
            embeddings_2d = embeddings_2d[indices]
            labels_shuffled = [labels_list[i] for i in indices]
            
            colors = [category_to_color[label] for label in labels_shuffled]
            
            ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=colors,
                alpha=point_alpha,
                s=point_size,
                edgecolors='white',
                linewidth=edge_width,
                rasterized=False
            )
            
            # Force square aspect ratio
            ax.set_aspect('equal', adjustable='box')
            
            # Set equal axis limits with padding
            x_range = embeddings_2d[:, 0].max() - embeddings_2d[:, 0].min()
            y_range = embeddings_2d[:, 1].max() - embeddings_2d[:, 1].min()
            max_range = max(x_range, y_range)
            
            x_center = (embeddings_2d[:, 0].max() + embeddings_2d[:, 0].min()) / 2
            y_center = (embeddings_2d[:, 1].max() + embeddings_2d[:, 1].min()) / 2
            margin = max_range * 0.08
            
            ax.set_xlim(x_center - max_range/2 - margin, x_center + max_range/2 + margin)
            ax.set_ylim(y_center - max_range/2 - margin, y_center + max_range/2 + margin)
            
            # Clean style: no border, no ticks (common in top venues)
            ax.axis('off')
            
            # Add title below the plot
            ax.set_title(title, fontsize=7, fontweight='normal', pad=2, y=0)
        
        # Plot both
        plot_scatter(ax_tiger, tiger_2d, labels, 'TIGER Recommendation')
        plot_scatter(ax_prism, prism_2d, labels, 'PRISM Recommendation')
        
        plt.tight_layout()
        
        # Save in multiple formats
        base_path = f"{output_dir}/embedding_{method}_comparison"
        
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
        
        logger.info(f"\nGenerated combined visualization:")
        logger.info(f"  - {base_path}.png")
        logger.info(f"  - {base_path}.pdf")
        logger.info(f"  - {base_path}.svg")
    
    def create_metrics_comparison_plot(
        self,
        prism_metrics: Dict[str, float],
        tiger_metrics: Dict[str, float],
        output_path: str
    ):
        """Create a publication-quality comparison plot of embedding quality metrics.
        
        Args:
            prism_metrics: PRISM embedding metrics
            tiger_metrics: TIGER embedding metrics
            output_path: Path to save the plot
        """
        plt.style.use('seaborn-v0_8-paper')
        sns.set_context("paper", font_scale=1.3)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.patch.set_facecolor('white')
        
        # Color scheme
        tiger_color = '#FF6B6B'  # Red
        prism_color = '#4ECDC4'  # Teal
        
        # 1. Mean Similarity
        ax = axes[0]
        metrics = ['Mean\nSimilarity']
        tiger_vals = [tiger_metrics['mean_similarity']]
        prism_vals = [prism_metrics['mean_similarity']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, tiger_vals, width, label='TIGER', color=tiger_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, prism_vals, width, label='PRISM', color=prism_color, alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        ax.set_ylabel('Similarity', fontsize=7, fontweight='bold')
        ax.set_title('Mean Pairwise Similarity\n(Lower = More Diverse)', fontsize=7, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=7)
        ax.legend(fontsize=7, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.set_facecolor('#f8f8f8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 2. Diversity
        ax = axes[1]
        metrics = ['Diversity']
        tiger_vals = [tiger_metrics['diversity']]
        prism_vals = [prism_metrics['diversity']]
        
        x = np.arange(len(metrics))
        
        bars1 = ax.bar(x - width/2, tiger_vals, width, label='TIGER', color=tiger_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, prism_vals, width, label='PRISM', color=prism_color, alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        ax.set_ylabel('Score', fontsize=7, fontweight='bold')
        ax.set_title('Embedding Diversity\n(Higher is Better)', fontsize=7, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=7)
        ax.legend(fontsize=7, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.set_facecolor('#f8f8f8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Calculate improvement
        improvement = (prism_metrics['diversity'] - tiger_metrics['diversity']) / tiger_metrics['diversity'] * 100
        ax.text(0, max(tiger_vals + prism_vals) * 1.05,
               f'Improvement: {improvement:+.1f}%',
               ha='center', fontsize=7, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        # 3. Std Similarity
        ax = axes[2]
        metrics = ['Std\nSimilarity']
        tiger_vals = [tiger_metrics['std_similarity']]
        prism_vals = [prism_metrics['std_similarity']]
        
        x = np.arange(len(metrics))
        
        bars1 = ax.bar(x - width/2, tiger_vals, width, label='TIGER', color=tiger_color, alpha=0.8)
        bars2 = ax.bar(x + width/2, prism_vals, width, label='PRISM', color=prism_color, alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        ax.set_ylabel('Standard Deviation', fontsize=7, fontweight='bold')
        ax.set_title('Similarity Std Dev\n(Higher = More Varied)', fontsize=7, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=7)
        ax.legend(fontsize=7, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.2, axis='y', linestyle='--')
        ax.set_facecolor('#f8f8f8')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save in multiple formats
        base_path = output_path.rsplit('.', 1)[0]
        plt.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f"{base_path}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Metrics comparison plot saved to {base_path}.png and {base_path}.pdf")
        
        plt.close()
        plt.style.use('default')


def main():
    parser = argparse.ArgumentParser(description='Analyze embedding space geometry')
    parser.add_argument('--prism_checkpoint', type=str, required=True,
                        help='Path to PRISM model checkpoint')
    parser.add_argument('--tiger_checkpoint', type=str, required=True,
                        help='Path to TIGER model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['beauty', 'sports', 'toys', 'cds'],
                        help='Dataset name')
    parser.add_argument('--output_dir', type=str, default='scripts/prism/embedding_analysis',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--num_samples', type=str, default='5000',
                        help='Number of items to sample for analysis')
    parser.add_argument('--vis_method', type=str, default='tsne',
                        choices=['tsne', 'umap'],
                        help='Visualization method')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = EmbeddingAnalyzer(
        prism_checkpoint=args.prism_checkpoint,
        tiger_checkpoint=args.tiger_checkpoint,
        dataset_name=args.dataset,
        device=args.device
    )
    
    # Load semantic mapper and embeddings
    logger.info("Loading semantic mappers and embeddings...")
    prism_config = get_prism_config(args.dataset)
    tiger_config = get_tiger_config(args.dataset)
    
    # Load item metadata for displaying text information
    logger.info("Loading item metadata...")
    metadata_path = Path(prism_config['data'].sequence_data_path) / f"{args.dataset.capitalize()}_metadata.json"
    item_metadata = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    if 'asin' in item:
                        item_metadata[item['asin']] = item
            logger.info(f"  Loaded metadata for {len(item_metadata)} items")
        except Exception as e:
            logger.warning(f"  Failed to load metadata: {e}")
    else:
        logger.warning(f"  Metadata file not found: {metadata_path}")
    
    # Load semantic mappers (SEPARATE for PRISM and TIGER!)
    from src.recommender.prism.dataset import SemanticIDMapper, load_content_embeddings, load_collab_embeddings, load_codebook_mappings
    
    logger.info("Loading PRISM semantic mapper...")
    prism_semantic_mapper = SemanticIDMapper(
        mapping_path=prism_config['data'].semantic_mapping_path,
        codebook_size=prism_config['model'].codebook_size,
        num_layers=prism_config['model'].num_code_layers,
        pad_token_id=prism_config['model'].pad_token_id
    )
    logger.info(f"  PRISM mapper: {len(prism_semantic_mapper.item_to_codes)} items")
    
    logger.info("Loading TIGER semantic mapper...")
    tiger_semantic_mapper = SemanticIDMapper(
        mapping_path=tiger_config['data'].semantic_mapping_path,
        codebook_size=tiger_config['model'].codebook_size,
        num_layers=tiger_config['model'].num_code_layers,
        pad_token_id=tiger_config['model'].pad_token_id
    )
    logger.info(f"  TIGER mapper: {len(tiger_semantic_mapper.item_to_codes)} items")
    
    # Load content and collab embeddings (shared between models)
    data_dir = prism_config['data'].sequence_data_path
    content_embs_dict = load_content_embeddings(data_dir)
    collab_embs_dict = load_collab_embeddings(prism_config['data'].collab_embedding_path)
    
    # Load codebook embeddings (for improved projection mode)
    semantic_mapping_dir = Path(prism_config['data'].semantic_mapping_path).parent
    codebook_embs_dict, _ = load_codebook_mappings(str(semantic_mapping_dir))
    
    logger.info(f"Loaded {len(content_embs_dict)} content embeddings")
    logger.info(f"Loaded {len(collab_embs_dict)} collab embeddings")
    logger.info(f"Loaded {len(codebook_embs_dict)} codebook embeddings")
    
    # Get all item IDs from BOTH semantic mappers
    prism_item_ids = set(prism_semantic_mapper.item_to_codes.keys())
    tiger_item_ids = set(tiger_semantic_mapper.item_to_codes.keys())
    
    # Use intersection of items that exist in both mappers
    common_item_ids = prism_item_ids & tiger_item_ids
    logger.info(f"Found {len(common_item_ids)} items in both PRISM and TIGER mappers")
    
    # Debug: Show semantic ID differences for a few sample items
    logger.info("\n" + "="*80)
    logger.info("SEMANTIC ID COMPARISON: PRISM vs TIGER (Sample)")
    logger.info("="*80)
    sample_items = sorted(list(common_item_ids))[:3]  # Only show 3 samples
    for item_id in sample_items:
        prism_codes = prism_semantic_mapper.get_codes(item_id)
        tiger_codes = tiger_semantic_mapper.get_codes(item_id)
        same = "✓ SAME" if prism_codes == tiger_codes else "✗ DIFFERENT"
        logger.info(f"Item {item_id}: PRISM={prism_codes}, TIGER={tiger_codes} [{same}]")
    logger.info("="*80 + "\n")
    
    # Filter to only items that have both content and collab embeddings
    valid_item_ids = [
        str(item_id) for item_id in common_item_ids
        if item_id in content_embs_dict and item_id in collab_embs_dict
    ]
    
    logger.info(f"Found {len(valid_item_ids)} items with complete embedding info")
    
    # Sample items for analysis
    num_samples = int(args.num_samples)
    if len(valid_item_ids) > num_samples:
        np.random.seed(42)
        sampled_item_ids = np.random.choice(valid_item_ids, num_samples, replace=False).tolist()
    else:
        sampled_item_ids = valid_item_ids
    
    logger.info(f"Analyzing {len(sampled_item_ids)} items...")
    
    # Extract embeddings
    logger.info("Extracting embeddings...")
    prism_embeddings, prism_valid_ids = analyzer.extract_embeddings(
        sampled_item_ids, prism_semantic_mapper, content_embs_dict, collab_embs_dict, codebook_embs_dict, 'prism'
    )
    tiger_embeddings, tiger_valid_ids = analyzer.extract_embeddings(
        sampled_item_ids, tiger_semantic_mapper, content_embs_dict, collab_embs_dict, codebook_embs_dict, 'tiger'
    )
    
    # Use intersection of valid IDs
    valid_ids = list(set(prism_valid_ids) & set(tiger_valid_ids))
    logger.info(f"Using {len(valid_ids)} items with valid embeddings from both models")
    
    # Filter embeddings to only include valid IDs
    prism_indices = [prism_valid_ids.index(iid) for iid in valid_ids]
    tiger_indices = [tiger_valid_ids.index(iid) for iid in valid_ids]
    
    prism_embeddings = prism_embeddings[prism_indices]
    tiger_embeddings = tiger_embeddings[tiger_indices]
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    
    prism_metrics = analyzer.compute_embedding_similarity(prism_embeddings)
    tiger_metrics = analyzer.compute_embedding_similarity(tiger_embeddings)
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("EMBEDDING QUALITY ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"\nMean Pairwise Similarity (lower = more diverse):")
    logger.info(f"  TIGER: {tiger_metrics['mean_similarity']:.4f}")
    logger.info(f"  PRISM: {prism_metrics['mean_similarity']:.4f}")
    
    logger.info(f"\nEmbedding Diversity (higher is better):")
    logger.info(f"  TIGER: {tiger_metrics['diversity']:.4f}")
    logger.info(f"  PRISM: {prism_metrics['diversity']:.4f}")
    improvement = (prism_metrics['diversity'] - tiger_metrics['diversity']) / tiger_metrics['diversity'] * 100
    logger.info(f"  Improvement: {improvement:+.2f}%")
    
    logger.info(f"\nSimilarity Std Dev (higher = more varied):")
    logger.info(f"  TIGER: {tiger_metrics['std_similarity']:.4f}")
    logger.info(f"  PRISM: {prism_metrics['std_similarity']:.4f}")
    
    # Save results
    results = {
        'num_items_analyzed': len(valid_ids),
        'prism_metrics': prism_metrics,
        'tiger_metrics': tiger_metrics,
        'improvement_diversity_pct': float(improvement)
    }
    
    results_file = output_dir / 'metrics.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")
    
    # Visualize embeddings (generates TWO separate images: TIGER and PRISM)
    logger.info("\nGenerating visualizations...")
    analyzer.visualize_embeddings(
        prism_embeddings,
        tiger_embeddings,
        valid_ids,
        prism_semantic_mapper,  # Use PRISM's mapper for category assignment (content-based)
        str(output_dir),
        method=args.vis_method,
        item_metadata=item_metadata,
        prism_semantic_mapper=prism_semantic_mapper,
        tiger_semantic_mapper=tiger_semantic_mapper
    )
    
    # Create metrics comparison plot
    logger.info("\nGenerating metrics comparison plot...")
    metrics_plot_file = output_dir / 'metrics_comparison.png'
    analyzer.create_metrics_comparison_plot(
        prism_metrics,
        tiger_metrics,
        str(metrics_plot_file)
    )
    
    logger.info("\n" + "="*80)
    logger.info("Analysis complete!")
    logger.info("="*80)
    logger.info(f"\nGenerated files:")
    logger.info(f"  - embedding_{args.vis_method}_comparison.png: Combined visualization (PNG)")
    logger.info(f"  - embedding_{args.vis_method}_comparison.pdf: Combined visualization (PDF for LaTeX)")
    logger.info(f"  - embedding_{args.vis_method}_comparison.svg: Combined visualization (SVG)")
    logger.info(f"  - metrics_comparison.png: Metrics comparison plot")
    logger.info(f"  - metrics.json: Quantitative metrics")
    logger.info(f"\nAll files saved to: {output_dir}")


if __name__ == '__main__':
    main()
