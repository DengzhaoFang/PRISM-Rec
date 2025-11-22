#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析tag的层次化重复率：
1. 加载tag_mapping和item_embeddings
2. 分析tag序列的重复情况
3. 计算不同层次的重复率
"""

import pandas as pd
import numpy as np
import argparse
from collections import Counter, defaultdict
import os
from typing import Dict, List, Tuple, Set


def load_data(tag_mapping_path: str, item_emb_path: str):
    """加载tag映射和item embeddings"""
    print("="*80)
    print("加载数据...")
    print("="*80)
    
    # 加载tag映射
    print(f"\n加载tag映射: {tag_mapping_path}")
    tag_to_id = np.load(tag_mapping_path, allow_pickle=True).item()
    id_to_tag = {v: k for k, v in tag_to_id.items()}
    print(f"✓ 加载了 {len(tag_to_id)} 个tag映射")
    
    # 加载item embeddings
    print(f"\n加载item embeddings: {item_emb_path}")
    df = pd.read_parquet(item_emb_path)
    print(f"✓ 加载了 {len(df):,} 个items")
    print(f"  列: {list(df.columns)}")
    
    return tag_to_id, id_to_tag, df


def analyze_tag_sequences(df: pd.DataFrame, id_to_tag: Dict[int, str]) -> Dict:
    """分析tag序列的重复情况"""
    print("\n" + "="*80)
    print("分析tag序列...")
    print("="*80)
    
    # 收集所有item的tag序列
    tag_sequences = []  # 每个item的tag序列（文本）
    tag_id_sequences = []  # 每个item的tag序列（ID）
    sequence_to_items = defaultdict(list)  # tag序列 -> item列表
    
    for idx, row in df.iterrows():
        tag_ids = row['category_tag_ids']
        # 过滤掉padding (0)
        valid_tag_ids = [tid for tid in tag_ids if tid > 0]
        
        # 转换为tag文本
        tag_texts = [id_to_tag.get(tid, f"<UNK:{tid}>") for tid in valid_tag_ids]
        
        # 创建序列（使用tuple以便作为dict的key）
        tag_id_tuple = tuple(valid_tag_ids)
        tag_text_tuple = tuple(tag_texts)
        
        tag_sequences.append(tag_text_tuple)
        tag_id_sequences.append(tag_id_tuple)
        
        # 记录哪些items有相同的tag序列
        sequence_to_items[tag_id_tuple].append(row['ItemID'])
    
    print(f"\n✓ 收集了 {len(tag_sequences):,} 个tag序列")
    print(f"✓ 发现 {len(sequence_to_items):,} 个唯一的tag序列")
    
    return {
        'tag_sequences': tag_sequences,
        'tag_id_sequences': tag_id_sequences,
        'sequence_to_items': sequence_to_items
    }


def calculate_repetition_rates(sequence_data: Dict, df: pd.DataFrame, id_to_tag: Dict[int, str]) -> Dict:
    """计算层次化重复率"""
    print("\n" + "="*80)
    print("计算层次化重复率...")
    print("="*80)
    
    tag_sequences = sequence_data['tag_sequences']
    tag_id_sequences = sequence_data['tag_id_sequences']
    sequence_to_items = sequence_data['sequence_to_items']
    
    results = {}
    
    # 1. 完整序列重复率
    print("\n1. 完整tag序列重复率")
    print("-" * 80)
    sequence_counts = Counter(tag_id_sequences)
    total_items = len(tag_sequences)
    unique_sequences = len(sequence_counts)
    
    # 计算重复的序列数量（出现次数>1的序列）
    repeated_sequences = sum(1 for count in sequence_counts.values() if count > 1)
    items_with_repeated_sequences = sum(count for count in sequence_counts.values() if count > 1)
    
    sequence_repetition_rate = (items_with_repeated_sequences / total_items) * 100
    unique_sequence_rate = (unique_sequences / total_items) * 100
    
    print(f"  总items数: {total_items:,}")
    print(f"  唯一序列数: {unique_sequences:,}")
    print(f"  重复序列数（出现>1次）: {repeated_sequences:,}")
    print(f"  使用重复序列的items数: {items_with_repeated_sequences:,}")
    print(f"  序列重复率: {sequence_repetition_rate:.2f}%")
    print(f"  序列唯一率: {unique_sequence_rate:.2f}%")
    
    # 显示最常见的重复序列
    print(f"\n  最常见的重复序列（Top 10）:")
    for seq_tuple, count in sequence_counts.most_common(10):
        if count > 1:
            seq_texts = [id_to_tag.get(tid, f"<UNK:{tid}>") for tid in seq_tuple]
            print(f"    出现{count}次: {' > '.join(seq_texts)}")
    
    results['sequence_repetition'] = {
        'total_items': total_items,
        'unique_sequences': unique_sequences,
        'repeated_sequences': repeated_sequences,
        'items_with_repeated_sequences': items_with_repeated_sequences,
        'sequence_repetition_rate': sequence_repetition_rate,
        'unique_sequence_rate': unique_sequence_rate
    }
    
    # 2. 前缀序列重复率（层次化的前半部分）
    print("\n2. 前缀序列重复率（层次化分析）")
    print("-" * 80)
    
    prefix_stats = defaultdict(int)  # prefix -> 出现次数
    prefix_lengths = []  # 记录每个item使用的前缀长度
    
    for seq_tuple in tag_id_sequences:
        # 对于每个序列，统计所有可能的前缀
        for prefix_len in range(1, len(seq_tuple) + 1):
            prefix = seq_tuple[:prefix_len]
            prefix_stats[prefix] += 1
        prefix_lengths.append(len(seq_tuple))
    
    # 计算不同长度前缀的重复率
    print(f"\n  前缀长度分布:")
    prefix_length_dist = Counter(prefix_lengths)
    for length in sorted(prefix_length_dist.keys()):
        count = prefix_length_dist[length]
        print(f"    长度{length}: {count:,} items ({count/len(prefix_lengths)*100:.1f}%)")
    
    # 分析不同层级的前缀重复
    max_depth = max(prefix_lengths) if prefix_lengths else 0
    print(f"\n  各层级前缀重复率（最多{max_depth}层）:")
    
    level_repetition_rates = {}
    for level in range(1, min(max_depth + 1, 6)):  # 最多分析5层
        level_prefixes = defaultdict(int)
        for seq_tuple in tag_id_sequences:
            if len(seq_tuple) >= level:
                prefix = seq_tuple[:level]
                level_prefixes[prefix] += 1
        
        total_prefixes = sum(level_prefixes.values())
        unique_prefixes = len(level_prefixes)
        repeated_prefixes = sum(1 for count in level_prefixes.values() if count > 1)
        items_with_repeated = sum(count for count in level_prefixes.values() if count > 1)
        repetition_rate = (items_with_repeated / total_prefixes) * 100 if total_prefixes > 0 else 0
        
        level_repetition_rates[level] = {
            'total': total_prefixes,
            'unique': unique_prefixes,
            'repeated': repeated_prefixes,
            'items_with_repeated': items_with_repeated,
            'repetition_rate': repetition_rate
        }
        
        print(f"    第{level}层:")
        print(f"      总前缀数: {total_prefixes:,}")
        print(f"      唯一前缀数: {unique_prefixes:,}")
        print(f"      重复前缀数: {repeated_prefixes:,}")
        print(f"      使用重复前缀的items数: {items_with_repeated:,}")
        print(f"      重复率: {repetition_rate:.2f}%")
        
        # 显示最常见的重复前缀
        if repeated_prefixes > 0:
            most_common = Counter(level_prefixes).most_common(3)
            print(f"      最常见的重复前缀（Top 3）:")
            for prefix_tuple, count in most_common:
                if count > 1:
                    prefix_texts = [id_to_tag.get(tid, f"<UNK:{tid}>") for tid in prefix_tuple]
                    print(f"        出现{count}次: {' > '.join(prefix_texts)}")
    
    results['prefix_repetition'] = level_repetition_rates
    
    # 3. 单个tag的重复使用率
    print("\n3. 单个tag的重复使用率")
    print("-" * 80)
    
    tag_usage = Counter()
    for seq_tuple in tag_id_sequences:
        for tag_id in seq_tuple:
            tag_usage[tag_id] += 1
    
    total_tag_occurrences = sum(tag_usage.values())
    unique_tags = len(tag_usage)
    avg_tags_per_item = np.mean(prefix_lengths) if prefix_lengths else 0
    
    print(f"  总tag出现次数: {total_tag_occurrences:,}")
    print(f"  唯一tag数: {unique_tags:,}")
    print(f"  平均每个item的tag数: {avg_tags_per_item:.2f}")
    print(f"  平均每个tag被使用次数: {total_tag_occurrences/unique_tags:.2f}")
    
    # 计算tag的重复使用率（使用次数>1的tag）
    repeated_tags = sum(1 for count in tag_usage.values() if count > 1)
    tag_repetition_rate = (repeated_tags / unique_tags) * 100 if unique_tags > 0 else 0
    
    print(f"  重复使用的tag数（使用>1次）: {repeated_tags:,}")
    print(f"  tag重复使用率: {tag_repetition_rate:.2f}%")
    
    # 显示最常用的tag
    print(f"\n  最常用的tag（Top 10）:")
    for tag_id, count in tag_usage.most_common(10):
        tag_text = id_to_tag.get(tag_id, f"<UNK:{tag_id}>")
        print(f"    '{tag_text}': 使用{count:,}次")
    
    results['tag_usage'] = {
        'total_occurrences': total_tag_occurrences,
        'unique_tags': unique_tags,
        'repeated_tags': repeated_tags,
        'tag_repetition_rate': tag_repetition_rate,
        'avg_tags_per_item': avg_tags_per_item
    }
    
    # 4. 层次化路径分析（不同长度的路径重复率）
    print("\n4. 层次化路径重复率（按路径长度）")
    print("-" * 80)
    
    path_length_stats = defaultdict(lambda: {'total': 0, 'unique': set(), 'counts': Counter()})
    
    for seq_tuple in tag_id_sequences:
        path_len = len(seq_tuple)
        path_length_stats[path_len]['total'] += 1
        path_length_stats[path_len]['unique'].add(seq_tuple)
        path_length_stats[path_len]['counts'][seq_tuple] += 1
    
    print(f"\n  各路径长度的重复情况:")
    for path_len in sorted(path_length_stats.keys()):
        stats = path_length_stats[path_len]
        total = stats['total']
        unique = len(stats['unique'])
        repeated = sum(1 for count in stats['counts'].values() if count > 1)
        items_with_repeated = sum(count for count in stats['counts'].values() if count > 1)
        repetition_rate = (items_with_repeated / total) * 100 if total > 0 else 0
        
        print(f"    路径长度{path_len}:")
        print(f"      总items: {total:,}")
        print(f"      唯一路径数: {unique:,}")
        print(f"      重复路径数: {repeated:,}")
        print(f"      使用重复路径的items数: {items_with_repeated:,}")
        print(f"      重复率: {repetition_rate:.2f}%")
    
    results['path_length_repetition'] = {
        length: {
            'total': stats['total'],
            'unique': len(stats['unique']),
            'repeated': sum(1 for count in stats['counts'].values() if count > 1),
            'repetition_rate': (sum(count for count in stats['counts'].values() if count > 1) / stats['total']) * 100 if stats['total'] > 0 else 0
        }
        for length, stats in path_length_stats.items()
    }
    
    return results


def print_summary(results: Dict):
    """打印总结"""
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    seq_rep = results.get('sequence_repetition', {})
    tag_usage = results.get('tag_usage', {})
    
    print(f"\n关键指标:")
    print(f"  1. 完整序列重复率: {seq_rep.get('sequence_repetition_rate', 0):.2f}%")
    print(f"  2. 序列唯一率: {seq_rep.get('unique_sequence_rate', 0):.2f}%")
    print(f"  3. Tag重复使用率: {tag_usage.get('tag_repetition_rate', 0):.2f}%")
    print(f"  4. 平均每个item的tag数: {tag_usage.get('avg_tags_per_item', 0):.2f}")
    
    prefix_rep = results.get('prefix_repetition', {})
    if prefix_rep:
        print(f"\n各层级前缀重复率:")
        for level in sorted(prefix_rep.keys()):
            rate = prefix_rep[level].get('repetition_rate', 0)
            print(f"  第{level}层: {rate:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="分析tag的层次化重复率")
    parser.add_argument('--tag_mapping_path', type=str, 
                       default='dataset/Amazon-Beauty/processed/beauty-hidvae-sentenceT5base/Beauty/tag_mapping.npy',
                       help='tag_mapping.npy文件路径')
    parser.add_argument('--item_emb_path', type=str,
                       default='dataset/Amazon-Beauty/processed/beauty-hidvae-sentenceT5base/Beauty/item_emb.parquet',
                       help='item_emb.parquet文件路径')
    
    args = parser.parse_args()
    
    # 加载数据
    tag_to_id, id_to_tag, df = load_data(args.tag_mapping_path, args.item_emb_path)
    
    # 分析tag序列
    sequence_data = analyze_tag_sequences(df, id_to_tag)
    
    # 计算重复率
    results = calculate_repetition_rates(sequence_data, df, id_to_tag)
    
    # 打印总结
    print_summary(results)
    
    print("\n" + "="*80)
    print("分析完成")
    print("="*80)


if __name__ == "__main__":
    main()













