# -*- coding: utf-8 -*-

import json
import gzip
import os
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from collections import Counter

# Available embedding models with their remote IDs
EMBEDDING_MODELS = {
    'sentence-t5': {
        'local_path': './sentence-t5-base',
        'remote_id': 'sentence-transformers/sentence-t5-base'
    },
    # Add more models here as needed
}

def setup_logging(output_dir, dataset_name):
    """Setup logging configuration to save to file"""
    log_file = os.path.join(output_dir, f"{dataset_name}_preprocessing.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Process Amazon dataset for recommendation")
    
    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='Beauty', 
                       help='Dataset name (e.g., Beauty, Sports, Toys)')
    parser.add_argument('--review_path', type=str, required=True,
                       help='Path to reviews_*.json.gz file')
    parser.add_argument('--meta_path', type=str, required=True,
                       help='Path to meta_*.json.gz file')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for processed data')
    
    # Filtering configuration
    parser.add_argument('--min_interactions', type=int, default=5,
                       help='Minimum interactions per user and per item (5-core filtering)')
    
    # Long-tail experiment configuration
    parser.add_argument('--long_tail_mode', action='store_true',
                       help='Enable long-tail analysis mode with grouped validation/test sets')
    parser.add_argument('--tail_thresholds', type=int, nargs='+', default=[5, 10, 20, 30],
                       help='Thresholds for long-tail grouping (default: [5, 10, 20, 30])')
    
    # Embedding configuration
    parser.add_argument('--embed_mode', type=str, default='tiger',
                       choices=['tiger', 'prism'],
                       help='Embedding mode: tiger (concat all) or prism (separate categories)')
    parser.add_argument('--embed_model', type=str, default='sentence-t5',
                       choices=list(EMBEDDING_MODELS.keys()),
                       help='Embedding model to use')
    parser.add_argument('--model_source', type=str, default='modelscope',
                       choices=['huggingface', 'modelscope'],
                       help='Model download source: huggingface or modelscope')
    parser.add_argument('--max_tags', type=int, default=5,
                       help='Maximum number of category tags for prism mode')
    parser.add_argument('--generate_all_embeddings', action='store_true',
                       help='Generate embeddings for ALL items (item_emb_all.parquet) in addition to filtered items. By default, only filtered items are processed.')
    
    # Debug options
    parser.add_argument('--print_samples', type=int, default=100,
                       help='Number of samples to print for inspection')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for embedding generation (auto/cpu/cuda/cuda:0/cuda:1/cuda:2, default: auto uses cuda:2 if available, else cpu)')
    
    return parser.parse_args()

# --- 1. Convert Raw Data to 'Strict' JSON ---
def convert_raw_to_json(args, logger):
    """Convert gzipped raw data to standard JSON format"""
    logger.info("="*80)
    logger.info("SECTION 1: Converting raw data to standard JSON format")
    logger.info("="*80)
    
    dataset_name = args.dataset
    output_path = os.path.join(args.output_dir, dataset_name)
    os.makedirs(output_path, exist_ok=True)

    def parse(path):
        """Generator function to parse gzipped files"""
        g = gzip.open(path, 'r')
        for l in g:
            yield json.dumps(eval(l))
    
    # Convert reviews
    review_json_path = os.path.join(output_path, f"{dataset_name}.json")
    logger.info(f"Parsing reviews from: {args.review_path}")
    logger.info(f"Output to: {review_json_path}")
    
    with open(review_json_path, 'w') as f:
        for l in parse(args.review_path):
            f.write(l + '\n')
    logger.info("✓ Review data conversion complete")
    
    # Inspect the converted data
    with open(review_json_path, 'r') as data:
        num_lines = sum(1 for _ in data)
        logger.info(f"  Total reviews: {num_lines:,}")
        data.seek(0)
        first_line = data.readline().strip()
        logger.info(f"  First review: {first_line[:150]}...")
    
    return output_path

# --- 2. Preprocess and Split Interaction Data ---
def apply_kcore_filter(interactions, min_interactions=5):
    """Apply k-core filtering to ensure each user and item has at least k interactions"""
    # Count interactions per user and item
    user_counts = Counter(interactions['userID'])
    item_counts = Counter(interactions['itemID'])
    
    # Initial stats
    original_users = len(user_counts)
    original_items = len(item_counts)
    original_interactions = len(interactions)
    
    # Iteratively filter until convergence
    prev_interactions = 0
    iteration = 0
    
    while len(interactions) != prev_interactions:
        prev_interactions = len(interactions)
        iteration += 1
        
        # Filter users with insufficient interactions
        valid_users = {user for user, count in user_counts.items() if count >= min_interactions}
        interactions = interactions[interactions['userID'].isin(valid_users)]
        
        # Filter items with insufficient interactions
        valid_items = {item for item, count in item_counts.items() if count >= min_interactions}
        interactions = interactions[interactions['itemID'].isin(valid_items)]
        
        # Recount after filtering
        user_counts = Counter(interactions['userID'])
        item_counts = Counter(interactions['itemID'])
    
    # Final stats
    final_users = len(user_counts)
    final_items = len(item_counts)
    final_interactions = len(interactions)
    
    return interactions, {
        'original': (original_users, original_items, original_interactions),
        'filtered': (final_users, final_items, final_interactions),
        'iterations': iteration
    }

def preprocess_interactions(args, output_path, logger):
    """Process interaction data with 5-core filtering and split into train/valid/test sets"""
    logger.info("\n" + "="*80)
    logger.info("SECTION 2: Preprocessing and splitting interaction data (5-core filtering)")
    logger.info("="*80)
    logger.info(f"Minimum interactions per user/item: {args.min_interactions}")

    dataset_name = args.dataset
    review_json_path = os.path.join(output_path, f"{dataset_name}.json")
    
    # Load all interactions into DataFrame
    logger.info(f"Loading interaction data from: {review_json_path}")
    interactions = []
    
    with open(review_json_path, 'r') as data:
        for line in data:
            review = json.loads(line.strip())
            interactions.append({
                'userID': review['reviewerID'],
                'itemID': review['asin'],
                'timestamp': review['unixReviewTime']
            })
    
    interactions_df = pd.DataFrame(interactions)
    logger.info(f"✓ Loaded {len(interactions_df):,} interactions")
    
    # Apply k-core filtering
    logger.info(f"\nApplying {args.min_interactions}-core filtering...")
    filtered_df, filter_stats = apply_kcore_filter(interactions_df, args.min_interactions)
    
    original_users, original_items, original_interactions = filter_stats['original']
    final_users, final_items, final_interactions = filter_stats['filtered']
    
    logger.info(f"✓ Filtering completed in {filter_stats['iterations']} iterations:")
    logger.info(f"  Users: {original_users:,} → {final_users:,} ({final_users/original_users:.1%})")
    logger.info(f"  Items: {original_items:,} → {final_items:,} ({final_items/original_items:.1%})")
    logger.info(f"  Interactions: {original_interactions:,} → {final_interactions:,} ({final_interactions/original_interactions:.1%})")
    
    # Create mappings for filtered data
    unique_users = sorted(filtered_df['userID'].unique())
    unique_items = sorted(filtered_df['itemID'].unique())
    
    userID_mapping = {user: idx + 1 for idx, user in enumerate(unique_users)}
    itemID_mapping = {item: idx + 1 for idx, item in enumerate(unique_items)}
    
    # Save mapping dictionaries
    user_mapping_path = os.path.join(output_path, 'user_mapping.npy')
    item_mapping_path = os.path.join(output_path, 'item_mapping.npy')
    
    np.save(user_mapping_path, userID_mapping)
    np.save(item_mapping_path, itemID_mapping)
    
    logger.info(f"\n✓ Created mappings:")
    logger.info(f"  Users: {len(userID_mapping):,}")
    logger.info(f"  Items: {len(itemID_mapping):,}")
    
    # Group interactions by user and sort by timestamp
    logger.info("\nGrouping interactions by user...")
    user_sequences = {}
    
    for _, row in filtered_df.iterrows():
        userID = userID_mapping[row['userID']]
        itemID = itemID_mapping[row['itemID']]
        timestamp = row['timestamp']
        
        if userID not in user_sequences:
            user_sequences[userID] = []
        user_sequences[userID].append((itemID, timestamp))
    
    # Sort items for each user by timestamp
    for userID in user_sequences:
        user_sequences[userID].sort(key=lambda x: x[1])
        user_sequences[userID] = [item[0] for item in user_sequences[userID]]
    
    logger.info(f"✓ Created sequences for {len(user_sequences):,} users")
    
    # Split data: last item for test, second-to-last for validation, rest for training
    train_data = []
    val_data = []
    test_data = []
    
    for userID, item_sequence in user_sequences.items():
        seq_len = len(item_sequence)
        if seq_len < 3:  # Need at least 3 items for train/val/test split
            continue
            
        # Training: use all but last 2 items as history, predict third-to-last item
        if seq_len > 3:  # Need at least 4 items to have training data
            train_history = item_sequence[:-3]
            train_target = item_sequence[-3]
            if len(train_history) > 0:  # Ensure we have history
                train_data.append({'user': userID, 'history': train_history, 'target': train_target})
        
        # Validation: use all but last 2 items as history, predict second-to-last item
        val_history = item_sequence[:-2]
        val_target = item_sequence[-2]  # second-to-last
        if len(val_history) > 0:  # Ensure we have history
            val_data.append({'user': userID, 'history': val_history, 'target': val_target})
        
        # Test: use all but last item as history, predict last item
        test_history = item_sequence[:-1]
        test_target = item_sequence[-1]  # last item
        test_data.append({'user': userID, 'history': test_history, 'target': test_target})
    
    # Create DataFrames
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    test_df = pd.DataFrame(test_data)
    
    logger.info(f"\n✓ Data split completed:")
    logger.info(f"  Train: {train_df.shape[0]:,} samples")
    logger.info(f"  Valid: {val_df.shape[0]:,} samples")
    logger.info(f"  Test:  {test_df.shape[0]:,} samples")
    
    # Show sequence length statistics
    seq_lengths = [len(seq) for seq in user_sequences.values()]
    logger.info(f"\n  User sequence length statistics:")
    logger.info(f"    Min: {min(seq_lengths)}")
    logger.info(f"    Max: {max(seq_lengths)}")
    logger.info(f"    Mean: {np.mean(seq_lengths):.1f}")
    logger.info(f"    Median: {np.median(seq_lengths):.0f}")
    logger.info(f"  Note: These are raw sequence counts (will be expanded using sliding window during training)")
    
    # Show example sequences
    if len(user_sequences) > 0:
        example_user, example_seq = next(iter(user_sequences.items()))
        logger.info(f"  Example user {example_user} sequence (first 10): {example_seq[:10]}")
    
    # Long-tail analysis if enabled
    if args.long_tail_mode:
        logger.info(f"\n" + "="*60)
        logger.info("LONG-TAIL ANALYSIS MODE ENABLED")
        logger.info("="*60)
        
        # Count item interactions from all data (for accurate popularity measurement)
        item_popularity = Counter()
        for _, row in filtered_df.iterrows():
            item_id = itemID_mapping[row['itemID']]
            item_popularity[item_id] += 1
        
        # Create long-tail splits
        train_df, val_groups, test_groups = create_longtail_splits(
            train_data, val_data, test_data, item_popularity, args.tail_thresholds, logger
        )
        
        # Save training data (includes all data)
        train_df.to_parquet(os.path.join(output_path, 'train.parquet'), index=False)
        
        # Save grouped validation and test sets
        for group_name, group_df in val_groups.items():
            filename = f'valid_{group_name}.parquet'
            group_df.to_parquet(os.path.join(output_path, filename), index=False)
            logger.info(f"  Saved {filename}: {len(group_df):,} samples")
            
        for group_name, group_df in test_groups.items():
            filename = f'test_{group_name}.parquet'
            group_df.to_parquet(os.path.join(output_path, filename), index=False)
            logger.info(f"  Saved {filename}: {len(group_df):,} samples")
    else:
        # Standard mode: save normal train/val/test splits
        train_df.to_parquet(os.path.join(output_path, 'train.parquet'), index=False)
        val_df.to_parquet(os.path.join(output_path, 'valid.parquet'), index=False)
        test_df.to_parquet(os.path.join(output_path, 'test.parquet'), index=False)
    
    logger.info(f"✓ Data saved to parquet files in: {output_path}")
    
    # Calculate item interaction counts (for popularity score)
    item_interaction_counts = Counter()
    for _, row in filtered_df.iterrows():
        item_id = itemID_mapping[row['itemID']]
        item_interaction_counts[item_id] += 1
    
    logger.info(f"\n✓ Item interaction statistics:")
    counts = list(item_interaction_counts.values())
    logger.info(f"  Min: {min(counts)}")
    logger.info(f"  Max: {max(counts)}")
    logger.info(f"  Mean: {np.mean(counts):.1f}")
    logger.info(f"  Median: {np.median(counts):.0f}")
    
    return itemID_mapping, item_interaction_counts

def create_longtail_splits(train_data, val_data, test_data, item_popularity, thresholds, logger):
    """Create long-tail splits based on target item popularity"""
    
    # Create training DataFrame (includes all data)
    train_df = pd.DataFrame(train_data)
    
    # Define popularity groups based on thresholds
    # thresholds = [5, 10, 20, 30] creates groups: (0-5], (5-10], (10-20], (20-30], (30+]
    group_ranges = []
    prev_threshold = 0
    for threshold in thresholds:
        group_ranges.append((prev_threshold, threshold))
        prev_threshold = threshold
    group_ranges.append((prev_threshold, float('inf')))  # Last group: 30+
    
    # Create group names
    group_names = []
    for i, (start, end) in enumerate(group_ranges):
        if end == float('inf'):
            group_names.append(f"tail_{start}plus")
        else:
            group_names.append(f"tail_{start}-{end}")
    
    logger.info(f"\nCreating long-tail groups based on target item popularity:")
    for name, (start, end) in zip(group_names, group_ranges):
        if end == float('inf'):
            logger.info(f"  {name}: items with >{start} interactions")
        else:
            logger.info(f"  {name}: items with ({start}-{end}] interactions")
    
    # Group validation data
    val_groups = {}
    for group_name in group_names:
        val_groups[group_name] = []
    
    for sample in val_data:
        target_item = sample['target']
        popularity = item_popularity.get(target_item, 0)
        
        # Find which group this item belongs to
        for group_name, (start, end) in zip(group_names, group_ranges):
            if start < popularity <= end:
                val_groups[group_name].append(sample)
                break
    
    # Group test data
    test_groups = {}
    for group_name in group_names:
        test_groups[group_name] = []
    
    for sample in test_data:
        target_item = sample['target']
        popularity = item_popularity.get(target_item, 0)
        
        # Find which group this item belongs to
        for group_name, (start, end) in zip(group_names, group_ranges):
            if start < popularity <= end:
                test_groups[group_name].append(sample)
                break
    
    # Convert to DataFrames and log statistics
    val_groups_df = {}
    test_groups_df = {}
    
    logger.info(f"\n✓ Long-tail grouping statistics:")
    
    total_val_samples = 0
    total_test_samples = 0
    
    for group_name in group_names:
        val_df = pd.DataFrame(val_groups[group_name])
        test_df = pd.DataFrame(test_groups[group_name])
        
        val_groups_df[group_name] = val_df
        test_groups_df[group_name] = test_df
        
        total_val_samples += len(val_df)
        total_test_samples += len(test_df)
        
        logger.info(f"  {group_name}:")
        logger.info(f"    Validation: {len(val_df):,} samples")
        logger.info(f"    Test: {len(test_df):,} samples")
    
    logger.info(f"\n  Total validation samples: {total_val_samples:,}")
    logger.info(f"  Total test samples: {total_test_samples:,}")
    logger.info(f"  Training samples: {len(train_df):,} (includes all data)")
    
    return train_df, val_groups_df, test_groups_df

def flatten_categories(categories):
    """Recursively flatten nested category lists to handle arbitrary nesting levels"""
    result = []
    if not categories:
        return result
    
    if isinstance(categories, list):
        for item in categories:
            if isinstance(item, list):
                # Recursively flatten nested lists
                result.extend(flatten_categories(item))
            elif item and str(item).strip():  # Only add non-empty items (no empty strings or whitespace-only)
                result.append(str(item))
    elif categories and str(categories).strip():  # Single non-list item (no empty strings or whitespace-only)
        result.append(str(categories))
    
    return result

def select_device(device_arg, logger):
    """
    Select device for embedding generation
    Priority: cuda:2 > cuda:1 > cuda:0 > cpu
    """
    if device_arg == 'auto':
        # Check available GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            logger.info(f"Found {num_gpus} CUDA device(s)")
            
            # Prefer cuda:2 (third GPU), then cuda:1, then cuda:0
            if num_gpus >= 3:
                device = 'cuda:2'
                logger.info(f"✓ Using device: {device} (third GPU, preferred)")
            elif num_gpus >= 2:
                device = 'cuda:1'
                logger.info(f"✓ Using device: {device} (second GPU)")
            else:
                device = 'cuda:0'
                logger.info(f"✓ Using device: {device} (first GPU)")
            
            # Print GPU info
            gpu_idx = int(device.split(':')[1]) if ':' in device else 0
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            gpu_memory = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
            logger.info(f"  GPU Name: {gpu_name}")
            logger.info(f"  GPU Memory: {gpu_memory:.2f} GB")
        else:
            device = 'cpu'
            logger.info(f"✓ CUDA not available, using device: {device}")
    else:
        device = device_arg
        logger.info(f"✓ Using specified device: {device}")
    
    return device

def download_model_if_needed(model_name, model_source, logger):
    """Download model if not exists locally, supporting both HuggingFace and ModelScope"""
    model_config = EMBEDDING_MODELS[model_name]
    local_path = model_config['local_path']
    remote_id = model_config['remote_id']
    
    # Check if model exists locally
    if os.path.exists(local_path) and os.listdir(local_path):
        logger.info(f"✓ Model found locally at: {local_path}")
        return local_path
    
    logger.info(f"Model not found locally. Downloading from {model_source}...")
    
    try:
        if model_source == 'huggingface':
            # Use SentenceTransformer's built-in download from HuggingFace
            logger.info(f"Downloading {remote_id} from HuggingFace...")
            model = SentenceTransformer(remote_id, cache_folder=os.path.dirname(local_path))
            # Save to specific local path
            model.save(local_path)
            logger.info(f"✓ Model downloaded and saved to: {local_path}")
            
        elif model_source == 'modelscope':
            # Download from ModelScope
            logger.info(f"Downloading {remote_id} from ModelScope...")
            try:
                from modelscope.hub.snapshot_download import snapshot_download
                
                # Download to local path
                model_dir = snapshot_download(
                    model_id=remote_id,
                    local_dir=local_path,
                    cache_dir=None  # Use local_dir directly
                )
                logger.info(f"✓ Model downloaded from ModelScope to: {model_dir}")
                
            except ImportError:
                logger.error("✗ ModelScope not installed. Please install: pip install modelscope")
                logger.info("Falling back to HuggingFace download...")
                model = SentenceTransformer(remote_id, cache_folder=os.path.dirname(local_path))
                model.save(local_path)
                logger.info(f"✓ Model downloaded from HuggingFace and saved to: {local_path}")
        
        return local_path
        
    except Exception as e:
        logger.error(f"✗ Failed to download model: {e}")
        logger.info("Please manually download the model or check your internet connection.")
        raise e


def generate_tiger_embeddings(item_info, model, logger):
    """TIGER mode: Concatenate all attributes into one text and embed"""
    item_embeddings = []
    
    for itemID, info in sorted(item_info.items()):
        # Combine all fields into single text
        text_parts = [
            f"title: {info.get('title', '')}",
            f"price: {info.get('price', '')}",
            f"salesRank: {info.get('salesRank', '')}",
            f"brand: {info.get('brand', '')}",
            f"categories: {info.get('categories', '')}"
        ]
        semantics = '\n'.join(text_parts)
        
        embedding = model.encode(semantics)
        item_embeddings.append({
            'ItemID': itemID,
            'embedding': embedding.tolist()
        })
    
    logger.info(f"  Generated {len(item_embeddings)} embeddings (single embedding per item)")
    return item_embeddings

def generate_prism_embeddings(item_info, model, max_tags, logger):
    """PRISM mode: Separate embeddings for categories and other attributes
    
    Optimized version:
    1. Collect all unique tags first
    2. Assign tag IDs and compute embeddings only once per unique tag
    3. Store tag IDs for each item instead of duplicate embeddings
    """
    # Step 1: Collect all unique tags from all items
    logger.info("  Step 1: Collecting all unique tags...")
    all_tags = set()
    item_tags = {}  # itemID -> list of tag texts
    
    for itemID, info in sorted(item_info.items()):
        categories = info.get('categories', [])
        category_tags = flatten_categories(categories)
        
        # Limit to max_tags
        category_tags = category_tags[:max_tags]
        item_tags[itemID] = category_tags
        
        # Add to unique tags set
        for tag in category_tags:
            if tag:  # Skip empty tags
                all_tags.add(str(tag))
    
    logger.info(f"    Found {len(all_tags):,} unique tags across all items")
    
    # Step 2: Create tag ID mapping (1-indexed, 0 reserved for padding)
    logger.info("  Step 2: Creating tag ID mapping...")
    sorted_tags = sorted(all_tags)
    tag_to_id = {tag: idx + 1 for idx, tag in enumerate(sorted_tags)}
    tag_to_id['<PAD>'] = 0  # Padding tag
    
    logger.info(f"    Assigned IDs to {len(tag_to_id):,} tags (including padding)")
    
    # Step 3: Compute embeddings for all unique tags (batch processing)
    logger.info("  Step 3: Computing embeddings for unique tags...")
    tag_embeddings_dict = {}
    embedding_dim = 768  # Default dimension
    
    # Encode all unique tags in batch for efficiency
    if sorted_tags:
        tag_embeddings_list = model.encode(sorted_tags, show_progress_bar=True, batch_size=256)
        if len(tag_embeddings_list) > 0:
            embedding_dim = len(tag_embeddings_list[0])
            for tag, embedding in zip(sorted_tags, tag_embeddings_list):
                tag_id = tag_to_id[tag]
                tag_embeddings_dict[tag_id] = embedding.tolist()
    
    # Add padding embedding (all zeros)
    tag_embeddings_dict[0] = [0.0] * embedding_dim
    
    logger.info(f"    Computed embeddings for {len(sorted_tags):,} unique tags")
    logger.info(f"    Embedding dimension: {embedding_dim}")
    
    # Step 4: Generate item embeddings with tag IDs
    logger.info("  Step 4: Generating item embeddings...")
    item_embeddings = []
    
    for itemID, info in sorted(item_info.items()):
        tags = item_tags[itemID]
        
        # Convert tags to tag IDs
        tag_ids = [tag_to_id[str(tag)] for tag in tags if tag]
        
        # Pad tag IDs to max_tags
        padded_tag_ids = tag_ids[:max_tags]  # Truncate if needed
        while len(padded_tag_ids) < max_tags:
            padded_tag_ids.append(0)  # Pad with 0
        
        # Get tag embeddings by looking up tag IDs
        category_embeddings = [tag_embeddings_dict[tag_id] for tag_id in padded_tag_ids]
        
        # Generate attribute embedding (excluding categories)
        attr_parts = [
            f"title: {info.get('title', '')}",
            f"price: {info.get('price', '')}",
            f"salesRank: {info.get('salesRank', '')}",
            f"brand: {info.get('brand', '')}"
        ]
        attr_text = '\n'.join(attr_parts)
        attr_embedding = model.encode(attr_text)
        
        # Store item embedding with metadata
        item_embeddings.append({
            'ItemID': itemID,
            'title': info.get('title', ''),  # Add item title
            'brand': info.get('brand', ''),  # Add brand info
            'category_tag_ids': padded_tag_ids,  # Store tag IDs instead of embeddings
            'category_tag_texts': [str(tag) for tag in tags[:max_tags]] + [''] * (max_tags - len(tags[:max_tags])),  # Tag texts for reference
            'category_embeddings': category_embeddings,  # Embeddings (for backward compatibility)
            'attribute_embedding': attr_embedding.tolist(),
            'num_categories': len(tags)
        })
    
    logger.info(f"  Generated {len(item_embeddings)} items with:")
    logger.info(f"    - {max_tags} category tag IDs per item (padded)")
    logger.info(f"    - {max_tags} category embeddings per item (padded)")
    logger.info(f"    - 1 attribute embedding per item")
    logger.info(f"    - Item metadata (title, brand, tag texts)")
    
    # Prepare tag info for return
    tag_info = {
        'tag_to_id': tag_to_id,
        'tag_embeddings': tag_embeddings_dict,
        'embedding_dim': embedding_dim,
        'num_unique_tags': len(sorted_tags)
    }
    
    return item_embeddings, tag_info

# --- 3a. Generate Embeddings for ALL Items (for Tokenizer Training) ---
def generate_all_item_embeddings(args, output_path, logger):
    """Generate semantic embeddings for ALL items (no filtering) - for tokenizer training"""
    logger.info("\n" + "="*80)
    logger.info("SECTION 3a: Generating embeddings for ALL items (tokenizer training)")
    logger.info("="*80)
    logger.info(f"Embedding mode: {args.embed_mode}")
    logger.info(f"Embedding model: {args.embed_model}")
    logger.info("Note: This includes ALL items from metadata, without 5-core filtering")
    
    dataset_name = args.dataset
    
    def parse(path):
        """Generator function to parse gzipped files"""
        g = gzip.open(path, 'r')
        for l in g:
            yield json.dumps(eval(l))
    
    # Convert metadata
    meta_json_path = os.path.join(output_path, f"{dataset_name}_metadata.json")
    logger.info(f"\nParsing metadata from: {args.meta_path}")
    logger.info(f"Output to: {meta_json_path}")
    
    with open(meta_json_path, 'w') as f:
        for l in parse(args.meta_path):
            f.write(l + '\n')
    logger.info("✓ Metadata conversion complete")
    
    # Process ALL metadata items (no filtering)
    logger.info("\nProcessing ALL items from metadata...")
    all_items_asin = []
    item_info_dict = {}
    
    with open(meta_json_path, 'r') as metadata_file:
        for line in metadata_file:
            metadata = json.loads(line.strip())
            asin = metadata.get('asin')
            
            if asin:  # As long as asin exists, we include it
                all_items_asin.append(asin)
                item_info_dict[asin] = {
                    'title': metadata.get('title'),
                    'price': metadata.get('price'),
                    'salesRank': metadata.get('salesRank'),
                    'brand': metadata.get('brand'),
                    'categories': metadata.get('categories', []),
                }
    
    logger.info(f"✓ Found {len(all_items_asin):,} unique items in metadata")
    
    # Create mapping for ALL items (1-indexed)
    unique_items = sorted(set(all_items_asin))
    itemID_mapping_all = {asin: idx + 1 for idx, asin in enumerate(unique_items)}
    
    # Save mapping for all items
    item_mapping_all_path = os.path.join(output_path, 'item_mapping_all.npy')
    np.save(item_mapping_all_path, itemID_mapping_all)
    logger.info(f"✓ Created mapping for ALL items: {len(itemID_mapping_all):,} items")
    logger.info(f"  Saved to: {item_mapping_all_path}")
    
    # Convert to ItemID-based dictionary
    item_info = {}
    for asin, itemID in itemID_mapping_all.items():
        if asin in item_info_dict:
            item_info[itemID] = item_info_dict[asin]
    
    logger.info(f"✓ Extracted metadata for {len(item_info):,} items")
    
    # Print sample metadata
    logger.info(f"\n--- Sample metadata (first {min(args.print_samples, len(item_info))} items) ---")
    sorted_item_info = sorted(item_info.items())
    for idx, (itemID, info) in enumerate(sorted_item_info[:args.print_samples]):
        logger.info(f"\nItem {itemID}:")
        logger.info(f"  Title: {info.get('title')}")
        logger.info(f"  Brand: {info.get('brand')}")
        logger.info(f"  Price: {info.get('price')}")
        logger.info(f"  SalesRank: {info.get('salesRank')}")
        logger.info(f"  Categories: {info.get('categories')}")
    
    # Select and set device
    logger.info(f"\n✓ Selecting device for embedding generation...")
    device = select_device(args.device, logger)
    
    # Download and load embedding model
    logger.info(f"\n✓ Preparing embedding model: {args.embed_model}")
    logger.info(f"Model source: {args.model_source}")
    
    try:
        model_path = download_model_if_needed(args.embed_model, args.model_source, logger)
        model = SentenceTransformer(model_path, device=device)
        logger.info(f"✓ Model loaded successfully from: {model_path}")
        logger.info(f"✓ Model running on device: {model.device}")
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        logger.error(f"  Failed to load model {args.embed_model}")
        exit(1)
    
    # Generate embeddings based on mode
    logger.info(f"\n✓ Generating embeddings in '{args.embed_mode}' mode...")
    
    tag_info = None  # Only used for prism mode
    if args.embed_mode == 'tiger':
        item_embeddings = generate_tiger_embeddings(item_info, model, logger)
    elif args.embed_mode == 'prism':
        item_embeddings, tag_info = generate_prism_embeddings(item_info, model, args.max_tags, logger)
    else:
        raise ValueError(f"Unknown embedding mode: {args.embed_mode}")
    
    # Save embeddings for ALL items
    item_emb_df = pd.DataFrame(item_embeddings)
    output_file = os.path.join(output_path, 'item_emb_all.parquet')
    item_emb_df.to_parquet(output_file, index=False)
    
    logger.info(f"\n✓ ALL items embeddings saved (for tokenizer training):")
    logger.info(f"  File: {output_file}")
    logger.info(f"  Shape: {item_emb_df.shape}")
    logger.info(f"  Columns: {list(item_emb_df.columns)}")
    logger.info(f"  Total items: {len(item_emb_df):,}")
    
    # Save tag information for prism mode
    if args.embed_mode == 'prism' and tag_info is not None:
        logger.info(f"\n✓ Saving tag information...")
        
        # Save tag-to-id mapping
        tag_mapping_file = os.path.join(output_path, 'tag_mapping.npy')
        np.save(tag_mapping_file, tag_info['tag_to_id'])
        logger.info(f"  Tag mapping saved: {tag_mapping_file}")
        logger.info(f"    Total unique tags: {tag_info['num_unique_tags']:,}")
        
        # Save tag embeddings as parquet for easier inspection
        tag_data = []
        for tag_text, tag_id in sorted(tag_info['tag_to_id'].items(), key=lambda x: x[1]):
            tag_data.append({
                'tag_id': tag_id,
                'tag_text': tag_text,
                'tag_embedding': tag_info['tag_embeddings'][tag_id]
            })
        
        tag_emb_df = pd.DataFrame(tag_data)
        tag_emb_file = os.path.join(output_path, 'tag_embeddings.parquet')
        tag_emb_df.to_parquet(tag_emb_file, index=False)
        logger.info(f"  Tag embeddings saved: {tag_emb_file}")
        logger.info(f"    Shape: {tag_emb_df.shape}")
        logger.info(f"    Embedding dimension: {tag_info['embedding_dim']}")
        
        # Show sample tags
        logger.info(f"\n  Sample tags (first 10):")
        for i, row in tag_emb_df.head(10).iterrows():
            logger.info(f"    Tag ID {row['tag_id']}: {row['tag_text']}")
    
    return item_emb_df, itemID_mapping_all

# --- 3b. Generate Embeddings for Filtered Items Directly ---
def generate_filtered_embeddings_directly(args, output_path, itemID_mapping_filtered, item_interaction_counts, logger):
    """Generate embeddings directly for filtered items only (after 5-core filter)"""
    logger.info(f"Embedding mode: {args.embed_mode}")
    logger.info(f"Embedding model: {args.embed_model}")
    logger.info(f"Processing {len(itemID_mapping_filtered):,} filtered items")
    
    dataset_name = args.dataset
    
    def parse(path):
        """Generator function to parse gzipped files"""
        g = gzip.open(path, 'r')
        for l in g:
            yield json.dumps(eval(l))
    
    # Convert metadata if not already done
    meta_json_path = os.path.join(output_path, f"{dataset_name}_metadata.json")
    if not os.path.exists(meta_json_path):
        logger.info(f"\nParsing metadata from: {args.meta_path}")
        logger.info(f"Output to: {meta_json_path}")
        
        with open(meta_json_path, 'w') as f:
            for l in parse(args.meta_path):
                f.write(l + '\n')
        logger.info("✓ Metadata conversion complete")
    else:
        logger.info(f"\n✓ Using existing metadata: {meta_json_path}")
    
    # Get reverse mapping: itemID -> asin
    reverse_mapping = {v: k for k, v in itemID_mapping_filtered.items()}
    
    # Load metadata only for filtered items
    logger.info("\nLoading metadata for filtered items...")
    item_info_dict = {}
    
    with open(meta_json_path, 'r') as metadata_file:
        for line in metadata_file:
            metadata = json.loads(line.strip())
            asin = metadata.get('asin')
            
            # Only process if this item is in filtered set
            if asin in itemID_mapping_filtered:
                item_info_dict[asin] = {
                    'title': metadata.get('title'),
                    'price': metadata.get('price'),
                    'salesRank': metadata.get('salesRank'),
                    'brand': metadata.get('brand'),
                    'categories': metadata.get('categories', []),
                }
    
    logger.info(f"✓ Loaded metadata for {len(item_info_dict):,} / {len(itemID_mapping_filtered):,} filtered items")
    
    # Convert to ItemID-based dictionary
    item_info = {}
    for itemID, asin in reverse_mapping.items():
        if asin in item_info_dict:
            item_info[itemID] = item_info_dict[asin]
    
    logger.info(f"✓ Prepared metadata for {len(item_info):,} items")
    
    # Print sample metadata
    logger.info(f"\n--- Sample metadata (first {min(args.print_samples, len(item_info))} items) ---")
    sorted_item_info = sorted(item_info.items())
    for idx, (itemID, info) in enumerate(sorted_item_info[:args.print_samples]):
        logger.info(f"\nItem {itemID}:")
        logger.info(f"  Title: {info.get('title')}")
        logger.info(f"  Brand: {info.get('brand')}")
        logger.info(f"  Price: {info.get('price')}")
        logger.info(f"  SalesRank: {info.get('salesRank')}")
        logger.info(f"  Categories: {info.get('categories')}")
    
    # Select and set device
    logger.info(f"\n✓ Selecting device for embedding generation...")
    device = select_device(args.device, logger)
    
    # Download and load embedding model
    logger.info(f"\n✓ Preparing embedding model: {args.embed_model}")
    logger.info(f"Model source: {args.model_source}")
    
    try:
        model_path = download_model_if_needed(args.embed_model, args.model_source, logger)
        model = SentenceTransformer(model_path, device=device)
        logger.info(f"✓ Model loaded successfully from: {model_path}")
        logger.info(f"✓ Model running on device: {model.device}")
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        logger.error(f"  Failed to load model {args.embed_model}")
        exit(1)
    
    # Generate embeddings based on mode
    logger.info(f"\n✓ Generating embeddings in '{args.embed_mode}' mode...")
    
    tag_info = None  # Only used for prism mode
    if args.embed_mode == 'tiger':
        item_embeddings = generate_tiger_embeddings(item_info, model, logger)
    elif args.embed_mode == 'prism':
        item_embeddings, tag_info = generate_prism_embeddings(item_info, model, args.max_tags, logger)
    else:
        raise ValueError(f"Unknown embedding mode: {args.embed_mode}")
    
    # Save embeddings for filtered items
    item_emb_df = pd.DataFrame(item_embeddings)
    
    # Add popularity scores based on interaction counts
    logger.info(f"\n✓ Adding popularity scores...")
    
    # Get interaction counts for each item
    interaction_counts = []
    for _, row in item_emb_df.iterrows():
        item_id = row['ItemID']
        count = item_interaction_counts.get(item_id, 0)
        interaction_counts.append(count)
    
    item_emb_df['interaction_count'] = interaction_counts
    
    # Calculate log-transformed popularity (for better distribution)
    item_emb_df['popularity_log'] = np.log1p(item_emb_df['interaction_count'])
    
    # Normalize to [0, 1] range for popularity_score
    max_log = item_emb_df['popularity_log'].max()
    min_log = item_emb_df['popularity_log'].min()
    if max_log > min_log:
        item_emb_df['popularity_score'] = (item_emb_df['popularity_log'] - min_log) / (max_log - min_log)
    else:
        item_emb_df['popularity_score'] = 0.5  # All items have same popularity
    
    logger.info(f"  Interaction count range: [{item_emb_df['interaction_count'].min()}, {item_emb_df['interaction_count'].max()}]")
    logger.info(f"  Popularity score range: [{item_emb_df['popularity_score'].min():.3f}, {item_emb_df['popularity_score'].max():.3f}]")
    logger.info(f"  Popularity score mean: {item_emb_df['popularity_score'].mean():.3f}")
    
    output_file = os.path.join(output_path, 'item_emb.parquet')
    item_emb_df.to_parquet(output_file, index=False)
    
    logger.info(f"\n✓ Filtered items embeddings saved:")
    logger.info(f"  File: {output_file}")
    logger.info(f"  Shape: {item_emb_df.shape}")
    logger.info(f"  Columns: {list(item_emb_df.columns)}")
    logger.info(f"  Total items: {len(item_emb_df):,}")
    
    # Save tag information for prism mode
    if args.embed_mode == 'prism' and tag_info is not None:
        logger.info(f"\n✓ Saving tag information...")
        
        # Save tag-to-id mapping
        tag_mapping_file = os.path.join(output_path, 'tag_mapping.npy')
        np.save(tag_mapping_file, tag_info['tag_to_id'])
        logger.info(f"  Tag mapping saved: {tag_mapping_file}")
        logger.info(f"    Total unique tags: {tag_info['num_unique_tags']:,}")
        
        # Save tag embeddings as parquet for easier inspection
        tag_data = []
        for tag_text, tag_id in sorted(tag_info['tag_to_id'].items(), key=lambda x: x[1]):
            tag_data.append({
                'tag_id': tag_id,
                'tag_text': tag_text,
                'tag_embedding': tag_info['tag_embeddings'][tag_id]
            })
        
        tag_emb_df = pd.DataFrame(tag_data)
        tag_emb_file = os.path.join(output_path, 'tag_embeddings.parquet')
        tag_emb_df.to_parquet(tag_emb_file, index=False)
        logger.info(f"  Tag embeddings saved: {tag_emb_file}")
        logger.info(f"    Shape: {tag_emb_df.shape}")
        logger.info(f"    Embedding dimension: {tag_info['embedding_dim']}")
        
        # Show sample tags
        logger.info(f"\n  Sample tags (first 10):")
        for i, row in tag_emb_df.head(10).iterrows():
            logger.info(f"    Tag ID {row['tag_id']}: {row['tag_text']}")
    
    return item_emb_df

# --- 3c. Generate Filtered Embeddings (for Recommender Training) ---
def generate_filtered_item_embeddings(args, output_path, itemID_mapping_filtered, item_emb_all_df, itemID_mapping_all, logger):
    """Generate embeddings for filtered items only (after 5-core filter) - for recommender training"""
    logger.info("\n" + "="*80)
    logger.info("SECTION 3b: Generating embeddings for FILTERED items (recommender training)")
    logger.info("="*80)
    logger.info(f"Filtering based on 5-core interaction data")
    
    # Create reverse mapping: filtered itemID -> original asin
    reverse_mapping_filtered = {v: k for k, v in itemID_mapping_filtered.items()}
    reverse_mapping_all = {v: k for k, v in itemID_mapping_all.items()}
    
    # Find which items from item_emb_all are in the filtered set
    filtered_embeddings = []
    matched_count = 0
    
    for filtered_itemID, asin in reverse_mapping_filtered.items():
        # Find this asin in the all items mapping
        if asin in itemID_mapping_all:
            all_itemID = itemID_mapping_all[asin]
            # Find embedding in item_emb_all_df
            emb_row = item_emb_all_df[item_emb_all_df['ItemID'] == all_itemID]
            if not emb_row.empty:
                # Create new row with filtered ItemID
                new_row = emb_row.iloc[0].to_dict()
                new_row['ItemID'] = filtered_itemID  # Use the filtered ItemID
                filtered_embeddings.append(new_row)
                matched_count += 1
    
    logger.info(f"✓ Matched {matched_count:,} / {len(itemID_mapping_filtered):,} filtered items with embeddings")
    
    # Create DataFrame and save
    item_emb_filtered_df = pd.DataFrame(filtered_embeddings)
    output_file = os.path.join(output_path, 'item_emb.parquet')
    item_emb_filtered_df.to_parquet(output_file, index=False)
    
    logger.info(f"\n✓ FILTERED items embeddings saved (for recommender training):")
    logger.info(f"  File: {output_file}")
    logger.info(f"  Shape: {item_emb_filtered_df.shape}")
    logger.info(f"  Columns: {list(item_emb_filtered_df.columns)}")
    logger.info(f"  Total items: {len(item_emb_filtered_df):,}")
    
    return item_emb_filtered_df

# --- Main Execution ---
def main():
    """Main execution function"""
    args = parse_args()
    
    # Setup logging first (create output directory if needed)
    dataset_name = args.dataset
    output_base = os.path.join(args.output_dir, dataset_name)
    os.makedirs(output_base, exist_ok=True)
    
    logger = setup_logging(output_base, dataset_name)
    
    logger.info("="*80)
    logger.info("AMAZON DATASET PREPROCESSING PIPELINE")
    logger.info("="*80)
    log_file_path = os.path.join(output_base, f"{dataset_name}_preprocessing.log")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Review path: {args.review_path}")
    logger.info(f"Meta path: {args.meta_path}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Min interactions: {args.min_interactions}")
    logger.info(f"Long-tail mode: {args.long_tail_mode}")
    if args.long_tail_mode:
        logger.info(f"Tail thresholds: {args.tail_thresholds}")
    logger.info(f"Embedding mode: {args.embed_mode}")
    logger.info(f"Embedding model: {args.embed_model}")
    logger.info(f"Model source: {args.model_source}")
    logger.info(f"Device: {args.device}")
    logger.info("")
    
    # Determine strategy based on --generate_all_embeddings flag
    if args.generate_all_embeddings:
        logger.info("STRATEGY: Generate embeddings for ALL items (tokenizer training),")
        logger.info("          then apply filtering only for interaction data (recommender training)")
        logger.info("          Will generate both item_emb_all.parquet and item_emb.parquet")
        skip_all_items = False
    else:
        logger.info("STRATEGY: Only generate embeddings for filtered items (item_emb.parquet)")
        logger.info("          Skipping item_emb_all.parquet to save computation time")
        logger.info("          Use --generate_all_embeddings flag if you need ALL items embeddings")
        skip_all_items = True
    
    # Step 1: Convert raw data
    output_path = convert_raw_to_json(args, logger)
    
    # Step 2: Preprocess interactions with filtering (for recommender training)
    itemID_mapping_filtered, item_interaction_counts = preprocess_interactions(args, output_path, logger)
    
    # Step 3: Generate embeddings based on strategy
    if skip_all_items:
        # Only generate embeddings for filtered items
        logger.info("\n" + "="*80)
        logger.info("GENERATING EMBEDDINGS FOR FILTERED ITEMS ONLY")
        logger.info("="*80)
        
        # Load metadata and generate embeddings for filtered items only
        item_emb_filtered_df = generate_filtered_embeddings_directly(args, output_path, itemID_mapping_filtered, item_interaction_counts, logger)
    else:
        # Generate embeddings for ALL items first, then filter
        item_emb_all_df, itemID_mapping_all = generate_all_item_embeddings(args, output_path, logger)
        item_emb_filtered_df = generate_filtered_item_embeddings(args, output_path, itemID_mapping_filtered, item_emb_all_df, itemID_mapping_all, logger)
    
    logger.info("\n" + "="*80)
    logger.info("✓ PREPROCESSING COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Output directory: {output_path}")
    logger.info("\nGenerated files:")
    logger.info(f"  - {args.dataset}.json (reviews)")
    logger.info(f"  - {args.dataset}_metadata.json (metadata)")
    logger.info(f"")
    logger.info(f"  Mapping files:")
    logger.info(f"    - user_mapping.npy (user ID mapping, filtered)")
    logger.info(f"    - item_mapping.npy (item ID mapping, filtered, for recommender)")
    if not skip_all_items:
        logger.info(f"    - item_mapping_all.npy (item ID mapping, ALL items, for tokenizer)")
    logger.info(f"")
    logger.info(f"  Interaction data (filtered, for recommender training):")
    logger.info(f"    - train.parquet (training data)")
    
    if args.long_tail_mode:
        logger.info("    - Long-tail validation sets:")
        for threshold in args.tail_thresholds:
            prev_threshold = args.tail_thresholds[args.tail_thresholds.index(threshold) - 1] if threshold != args.tail_thresholds[0] else 0
            logger.info(f"      - valid_tail_{prev_threshold}-{threshold}.parquet")
        logger.info(f"      - valid_tail_{args.tail_thresholds[-1]}plus.parquet")
        
        logger.info("    - Long-tail test sets:")
        for threshold in args.tail_thresholds:
            prev_threshold = args.tail_thresholds[args.tail_thresholds.index(threshold) - 1] if threshold != args.tail_thresholds[0] else 0
            logger.info(f"      - test_tail_{prev_threshold}-{threshold}.parquet")
        logger.info(f"      - test_tail_{args.tail_thresholds[-1]}plus.parquet")
    else:
        logger.info(f"    - valid.parquet (validation data)")
        logger.info(f"    - test.parquet (test data)")
    
    logger.info(f"")
    logger.info(f"  Embedding files:")
    if skip_all_items:
        logger.info(f"    - item_emb.parquet (filtered items embeddings, for recommender training)")
    else:
        logger.info(f"    - item_emb_all.parquet (ALL items embeddings, for tokenizer training)")
        logger.info(f"    - item_emb.parquet (filtered items embeddings, for recommender training)")
    
    if args.embed_mode == 'prism':
        logger.info(f"")
        logger.info(f"  Tag files (prism mode only):")
        logger.info(f"    - tag_mapping.npy (tag text -> tag ID mapping)")
        logger.info(f"    - tag_embeddings.parquet (tag ID, tag text, tag embedding)")
    
    logger.info(f"")
    logger.info(f"  - {dataset_name}_preprocessing.log (this log file)")
    


if __name__ == "__main__":
    main()