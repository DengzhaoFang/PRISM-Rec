"""
抽取不同类别的样本并打印元信息
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SemanticIDMapper:
    """简化版的语义ID映射器"""
    
    def __init__(self, mapping_path: str):
        self.item_to_codes = {}
        self._load_mappings(mapping_path)
    
    def _load_mappings(self, mapping_path: str):
        """加载语义ID映射"""
        with open(mapping_path, 'r') as f:
            data = json.load(f)
            for item_id_str, codes in data.items():
                item_id = int(item_id_str)
                self.item_to_codes[item_id] = codes
        logger.info(f"加载了 {len(self.item_to_codes)} 个item的语义映射")
    
    def get_codes(self, item_id: int):
        """获取item的语义codes"""
        return self.item_to_codes.get(item_id)


def load_metadata(meta_file: str) -> Dict:
    """加载item元数据"""
    metadata = {}
    
    if not Path(meta_file).exists():
        logger.warning(f"元数据文件不存在: {meta_file}")
        return metadata
    
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                try:
                    import ast
                    item = ast.literal_eval(line)
                except (ValueError, SyntaxError) as e:
                    logger.warning(f"解析失败 line {line_num}: {e}")
                    continue
            
            if 'asin' in item:
                metadata[item['asin']] = item
    
    logger.info(f"加载了 {len(metadata)} 个item的元数据")
    return metadata


def load_item_mapping(mapping_file: str) -> Dict[int, str]:
    """
    加载item_id到ASIN的映射
    
    Returns:
        字典: item_id -> asin (注意：这里需要反转原始映射)
    """
    import numpy as np
    
    if not Path(mapping_file).exists():
        logger.warning(f"映射文件不存在: {mapping_file}")
        return {}
    
    # 加载映射 (asin -> item_id)
    asin_to_id = np.load(mapping_file, allow_pickle=True).item()
    
    # 反转映射 (item_id -> asin)
    id_to_asin = {v: k for k, v in asin_to_id.items()}
    
    logger.info(f"加载了 {len(id_to_asin)} 个item的映射关系")
    return id_to_asin


def assign_categories(semantic_mapper: SemanticIDMapper) -> Dict[str, List[int]]:
    """
    根据语义ID的第一个code分配类别
    
    Returns:
        字典: category_name -> [item_id1, item_id2, ...]
    """
    category_to_items = defaultdict(list)
    
    for item_id, codes in semantic_mapper.item_to_codes.items():
        if codes and len(codes) >= 1:
            # 使用第一个code作为类别标识
            category = f"Cat_{codes[0]}"
            category_to_items[category].append(item_id)
    
    return dict(category_to_items)


def print_item_info(item_id: int, metadata: Dict, id_to_asin: Dict, semantic_mapper: SemanticIDMapper):
    """打印item的文本信息"""
    # 将item_id转换为asin
    asin = id_to_asin.get(item_id)
    if not asin:
        logger.warning(f"  Item {item_id}: 未找到对应的ASIN")
        return
    
    # 转换ASIN为字符串（可能是整数）
    asin_str = str(asin)
    
    meta = metadata.get(asin_str)
    if not meta:
        logger.warning(f"  Item {item_id} (ASIN: {asin_str}): 未找到元数据")
        return
    
    # 获取语义codes
    codes = semantic_mapper.get_codes(item_id)
    
    # 打印关键信息
    print(f"\n  【样本 {item_id}】")
    print(f"  ASIN: {asin_str}")
    print(f"  语义Codes: {codes}")
    
    if 'title' in meta:
        title = meta['title']
        if len(title) > 80:
            title = title[:80] + "..."
        print(f"  标题: {title}")
    
    if 'categories' in meta and meta['categories']:
        categories = meta['categories'][0] if meta['categories'] else []
        if categories:
            category_path = ' > '.join(categories[:3])  # 显示前3级
            print(f"  类别路径: {category_path}")
    
    if 'brand' in meta:
        print(f"  品牌: {meta['brand']}")
    
    if 'price' in meta:
        print(f"  价格: {meta['price']}")


def main():
    parser = argparse.ArgumentParser(description='抽取不同类别的样本并打印元信息')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['beauty', 'sports', 'toys', 'cds'],
                        help='数据集名称')
    parser.add_argument('--semantic_mapping', type=str, required=True,
                        help='语义ID映射文件路径')
    parser.add_argument('--metadata', type=str, required=True,
                        help='元数据文件路径')
    parser.add_argument('--item_mapping', type=str, required=True,
                        help='item_mapping.npy文件路径')
    parser.add_argument('--min_samples', type=int, default=10,
                        help='类别最小样本数阈值（少于此数量的类别会被过滤）')
    parser.add_argument('--samples_per_category', type=int, default=2,
                        help='每个类别抽取的样本数')
    
    args = parser.parse_args()
    
    # 加载语义映射
    logger.info(f"加载语义映射: {args.semantic_mapping}")
    semantic_mapper = SemanticIDMapper(args.semantic_mapping)
    
    # 加载元数据
    logger.info(f"加载元数据: {args.metadata}")
    metadata = load_metadata(args.metadata)
    
    # 加载item_id到ASIN的映射
    logger.info(f"加载item映射: {args.item_mapping}")
    id_to_asin = load_item_mapping(args.item_mapping)
    
    # 分配类别
    logger.info("根据语义ID分配类别...")
    category_to_items = assign_categories(semantic_mapper)
    
    # 过滤样本数过少的类别
    filtered_categories = {
        cat: items for cat, items in category_to_items.items()
        if len(items) >= args.min_samples
    }
    
    logger.info(f"\n总类别数: {len(category_to_items)}")
    logger.info(f"过滤后类别数: {len(filtered_categories)} (最小样本数: {args.min_samples})")
    
    # 按类别名称排序
    sorted_categories = sorted(filtered_categories.items(), key=lambda x: x[0])
    
    # 打印每个类别的样本信息
    print("\n" + "="*80)
    print("类别样本信息")
    print("="*80)
    
    for category, items in sorted_categories:
        print(f"\n【{category}】 (共 {len(items)} 个样本)")
        print("-" * 80)
        
        # 抽取指定数量的样本
        sample_items = items[:args.samples_per_category]
        
        for item_id in sample_items:
            print_item_info(item_id, metadata, id_to_asin, semantic_mapper)
    
    print("\n" + "="*80)
    print(f"总结: 显示了 {len(filtered_categories)} 个类别的样本")
    print("="*80)
    
    # 输出类别统计
    print("\n类别统计 (前20个):")
    for category, items in sorted_categories[:20]:
        print(f"  {category}: {len(items)} 个样本")
    if len(sorted_categories) > 20:
        print(f"  ... (还有 {len(sorted_categories) - 20} 个类别)")


if __name__ == '__main__':
    main()
