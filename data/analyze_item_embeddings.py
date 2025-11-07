#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析 item_emb.parquet 文件：
1. 统计独立 tag id 数量
2. 随机打印 100 行完整信息
3. 统计 tag 层数信息
"""

import pandas as pd
import numpy as np
import argparse
from collections import Counter
import random
import os

def analyze_item_embeddings(parquet_path, num_samples=100, seed=42, tag_mapping_path=None):
    """分析 item embeddings parquet 文件"""
    
    print("="*80)
    print("ITEM EMBEDDINGS ANALYSIS")
    print("="*80)
    print(f"\nLoading file: {parquet_path}")
    
    # 加载数据
    df = pd.read_parquet(parquet_path)
    
    # 尝试加载 tag 映射（如果提供了路径）
    tag_to_id = None
    id_to_tag = None
    if tag_mapping_path and os.path.exists(tag_mapping_path):
        print(f"\nLoading tag mapping from: {tag_mapping_path}")
        tag_to_id = np.load(tag_mapping_path, allow_pickle=True).item()
        id_to_tag = {v: k for k, v in tag_to_id.items()}
        print(f"✓ Loaded {len(tag_to_id)} tag mappings")
    else:
        # 尝试从同一目录自动查找
        base_dir = os.path.dirname(parquet_path)
        auto_tag_path = os.path.join(base_dir, 'tag_mapping.npy')
        if os.path.exists(auto_tag_path):
            print(f"\nAuto-loading tag mapping from: {auto_tag_path}")
            tag_to_id = np.load(auto_tag_path, allow_pickle=True).item()
            id_to_tag = {v: k for k, v in tag_to_id.items()}
            print(f"✓ Loaded {len(tag_to_id)} tag mappings")
    
    print(f"✓ Loaded {len(df):,} items")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    
    # 1. 统计独立 tag id
    print("\n" + "="*80)
    print("1. STATISTICS: Unique Tag IDs")
    print("="*80)
    
    all_tag_ids = set()
    tag_id_counter = Counter()
    
    for idx, row in df.iterrows():
        tag_ids = row['category_tag_ids']
        for tag_id in tag_ids:
            if tag_id > 0:  # 排除 padding (0)
                all_tag_ids.add(tag_id)
                tag_id_counter[tag_id] += 1
    
    print(f"Total unique tag IDs (excluding padding): {len(all_tag_ids):,}")
    print(f"Tag ID range: {min(all_tag_ids)} - {max(all_tag_ids)}")
    print(f"\nMost frequently used tags (top 10):")
    for tag_id, count in tag_id_counter.most_common(10):
        # 优先使用 id_to_tag 映射
        if id_to_tag and tag_id in id_to_tag:
            tag_text = id_to_tag[tag_id]
        else:
            # 回退到从数据中查找
            sample_row = df[df['category_tag_ids'].apply(lambda x: tag_id in x)]
            if not sample_row.empty:
                tag_ids = sample_row.iloc[0]['category_tag_ids']
                tag_texts = sample_row.iloc[0]['category_tag_texts']
                tag_text_idx = next((i for i, tid in enumerate(tag_ids) if tid == tag_id), None)
                if tag_text_idx is not None and tag_text_idx < len(tag_texts):
                    tag_text = tag_texts[tag_text_idx]
                else:
                    tag_text = "Unknown"
            else:
                tag_text = "Unknown"
        print(f"  Tag ID {tag_id}: '{tag_text}' - used in {count:,} items")
    
    # 2. 随机打印 100 行完整信息
    print("\n" + "="*80)
    print(f"2. SAMPLE: Random {num_samples} rows (complete columns)")
    print("="*80)
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 随机选择行
    sample_indices = random.sample(range(len(df)), min(num_samples, len(df)))
    sample_df = df.iloc[sample_indices].copy()
    
    # 按 ItemID 排序以便阅读
    sample_df = sample_df.sort_values('ItemID')
    
    print(f"\nPrinting {len(sample_df)} random samples:\n")
    
    for idx, (row_idx, row) in enumerate(sample_df.iterrows(), 1):
        print(f"{'='*80}")
        print(f"Sample {idx}/{len(sample_df)} - Row Index: {row_idx}")
        print(f"{'='*80}")
        print(f"ItemID: {row['ItemID']}")
        print(f"Title: {row['title']}")
        print(f"Brand: {row['brand']}")
        print(f"Num Categories: {row['num_categories']}")
        print(f"\nCategory Tag IDs: {row['category_tag_ids']}")
        print(f"Category Tag Texts:")
        for i, (tag_id, tag_text) in enumerate(zip(row['category_tag_ids'], row['category_tag_texts'])):
            if tag_id > 0:  # 只显示非 padding
                # 如果 tag_text 为空，尝试从 id_to_tag 获取
                if not tag_text and id_to_tag and tag_id in id_to_tag:
                    tag_text = id_to_tag[tag_id]
                print(f"  [{i+1}] Tag ID {tag_id}: '{tag_text}'")
        print(f"\nEmbedding Dimensions:")
        print(f"  Category Embeddings: {len(row['category_embeddings'])} vectors")
        if len(row['category_embeddings']) > 0:
            print(f"    - Each vector dimension: {len(row['category_embeddings'][0])}")
        print(f"  Attribute Embedding: 1 vector")
        print(f"    - Vector dimension: {len(row['attribute_embedding'])}")
        print()
    
    # 3. 统计 tag 层数信息
    print("="*80)
    print("3. STATISTICS: Tag Hierarchy Information")
    print("="*80)
    
    # 从 tag texts 推断层级信息
    # 通常 Amazon 的 categories 格式可能是：["Beauty", "Health & Personal Care", ...]
    # 或者可能是嵌套结构，但我们已经 flatten 了
    # 我们可以统计：
    #   - 每个 item 的 tag 数量分布
    #   - tag 文本长度分布
    #   - tag 文本中是否包含层级分隔符（如 ">"）
    
    tag_count_dist = Counter()
    tag_text_lengths = []
    hierarchy_separators = Counter()
    
    for idx, row in df.iterrows():
        tag_count = row['num_categories']
        tag_count_dist[tag_count] += 1
        
        for tag_text in row['category_tag_texts']:
            if tag_text and tag_text.strip():  # 非空
                tag_text_lengths.append(len(tag_text))
                
                # 检查层级分隔符
                if ' > ' in tag_text or '>' in tag_text:
                    hierarchy_separators['>'] += 1
                if ' / ' in tag_text or '/' in tag_text:
                    hierarchy_separators['/'] += 1
                if ' | ' in tag_text or '|' in tag_text:
                    hierarchy_separators['|'] += 1
    
    print(f"\nTag count per item distribution:")
    for count in sorted(tag_count_dist.keys()):
        num_items = tag_count_dist[count]
        percentage = (num_items / len(df)) * 100
        print(f"  {count} tags: {num_items:,} items ({percentage:.1f}%)")
    
    if tag_text_lengths:
        print(f"\nTag text length statistics:")
        print(f"  Min: {min(tag_text_lengths)} characters")
        print(f"  Max: {max(tag_text_lengths)} characters")
        print(f"  Mean: {np.mean(tag_text_lengths):.1f} characters")
        print(f"  Median: {np.median(tag_text_lengths):.0f} characters")
    
    if hierarchy_separators:
        print(f"\nHierarchy separator usage:")
        for sep, count in hierarchy_separators.items():
            print(f"  '{sep}': found in {count:,} tags")
    
    # 尝试从原始 tag texts 推断层级深度
    # 如果 tag 文本本身不包含层级信息，我们统计每个 item 的平均层级深度
    # 这里我们假设每个 tag 可能代表一个层级（但这取决于原始数据的结构）
    
    print(f"\nTag structure analysis:")
    print(f"  - Items with at least 1 tag: {(df['num_categories'] > 0).sum():,} ({(df['num_categories'] > 0).sum() / len(df) * 100:.1f}%)")
    print(f"  - Items with max tags ({max(tag_count_dist.keys())}): {tag_count_dist[max(tag_count_dist.keys())]:,}")
    print(f"  - Average tags per item: {df['num_categories'].mean():.2f}")
    print(f"  - Median tags per item: {df['num_categories'].median():.0f}")
    
    # 如果可能的话，尝试从 metadata 加载原始 categories 来统计真实层级
    print(f"\nNote: Original category hierarchy structure was flattened during processing.")
    print(f"      Each tag in 'category_tag_texts' represents a flattened category label.")
    print(f"      The 'num_categories' field shows how many tags each item has.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Analyze item_emb.parquet file")
    parser.add_argument('--parquet_path', type=str, required=True,
                       help='Path to item_emb.parquet file')
    parser.add_argument('--tag_mapping_path', type=str, default=None,
                       help='Path to tag_mapping.npy file (auto-detected if not provided)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of random samples to print (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling (default: 42)')
    
    args = parser.parse_args()
    
    analyze_item_embeddings(args.parquet_path, args.num_samples, args.seed, args.tag_mapping_path)

if __name__ == "__main__":
    main()

