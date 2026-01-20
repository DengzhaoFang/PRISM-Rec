#!/usr/bin/env python3
"""
Add popularity_score to existing item_emb.parquet files.

This script calculates item interaction counts from train.parquet and adds
popularity-related columns to item_emb.parquet:
- interaction_count: raw count of interactions
- popularity_log: log1p(interaction_count)
- popularity_score: normalized to [0, 1] range

Usage:
    python add_popularity_score.py --data_dir dataset/Amazon-Sports/processed/sports-prism-sentenceT5base/Sports
"""

import argparse
import os
import numpy as np
import pandas as pd
from collections import Counter


def add_popularity_score(data_dir: str):
    """Add popularity_score to item_emb.parquet based on train.parquet interactions."""
    
    # Paths
    train_path = os.path.join(data_dir, 'train.parquet')
    item_emb_path = os.path.join(data_dir, 'item_emb.parquet')
    
    # Check files exist
    if not os.path.exists(train_path):
        print(f"Error: train.parquet not found at {train_path}")
        return False
    
    if not os.path.exists(item_emb_path):
        print(f"Error: item_emb.parquet not found at {item_emb_path}")
        return False
    
    print(f"Loading train.parquet from {train_path}...")
    train_df = pd.read_parquet(train_path)
    print(f"  Loaded {len(train_df)} training samples")
    
    # Count item interactions from training data
    # Each sample has 'history' (list of items) and 'target' (single item)
    item_counts = Counter()
    
    for _, row in train_df.iterrows():
        # Count items in history
        history = row['history']
        if isinstance(history, list):
            for item_id in history:
                item_counts[item_id] += 1
        
        # Count target item
        target = row['target']
        item_counts[target] += 1
    
    print(f"  Found {len(item_counts)} unique items in training data")
    
    # Load item_emb.parquet
    print(f"\nLoading item_emb.parquet from {item_emb_path}...")
    item_emb_df = pd.read_parquet(item_emb_path)
    print(f"  Loaded {len(item_emb_df)} items")
    print(f"  Existing columns: {item_emb_df.columns.tolist()}")
    
    # Check if popularity_score already exists
    if 'popularity_score' in item_emb_df.columns:
        print(f"\n⚠ popularity_score already exists!")
        print(f"  Mean: {item_emb_df['popularity_score'].mean():.3f}")
        overwrite = input("  Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("  Skipping...")
            return False
    
    # Add interaction counts
    print(f"\nAdding popularity scores...")
    interaction_counts = []
    for _, row in item_emb_df.iterrows():
        item_id = row['ItemID']
        count = item_counts.get(item_id, 0)
        interaction_counts.append(count)
    
    item_emb_df['interaction_count'] = interaction_counts
    
    # Calculate log-transformed popularity
    item_emb_df['popularity_log'] = np.log1p(item_emb_df['interaction_count'])
    
    # Normalize to [0, 1] range
    max_log = item_emb_df['popularity_log'].max()
    min_log = item_emb_df['popularity_log'].min()
    if max_log > min_log:
        item_emb_df['popularity_score'] = (item_emb_df['popularity_log'] - min_log) / (max_log - min_log)
    else:
        item_emb_df['popularity_score'] = 0.5
    
    # Print statistics
    print(f"\n✓ Popularity statistics:")
    print(f"  Interaction count range: [{item_emb_df['interaction_count'].min()}, {item_emb_df['interaction_count'].max()}]")
    print(f"  Interaction count mean: {item_emb_df['interaction_count'].mean():.1f}")
    print(f"  Popularity score range: [{item_emb_df['popularity_score'].min():.3f}, {item_emb_df['popularity_score'].max():.3f}]")
    print(f"  Popularity score mean: {item_emb_df['popularity_score'].mean():.3f}")
    
    # Save updated file
    print(f"\nSaving updated item_emb.parquet...")
    item_emb_df.to_parquet(item_emb_path, index=False)
    print(f"✓ Saved to {item_emb_path}")
    print(f"  New columns: {item_emb_df.columns.tolist()}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Add popularity_score to item_emb.parquet")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing train.parquet and item_emb.parquet')
    
    args = parser.parse_args()
    
    success = add_popularity_score(args.data_dir)
    
    if success:
        print("\n✓ Done!")
    else:
        print("\n✗ Failed!")


if __name__ == "__main__":
    main()
