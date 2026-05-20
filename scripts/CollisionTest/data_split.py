#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义密度分析系统

该脚本用于分析商品在语义空间中的密度分布，通过KNN距离计算语义密度，
并将商品按密度划分为多个桶，为碰撞实验提供数据基础。

使用示例:
    python scripts/CollisionTest/data_split.py \
        --dataset_path dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty \
        --num_buckets 10 \
        --k_neighbors 20
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# 1. 命令行接口模块
# ============================================================================

class CLIParser:
    """命令行参数解析器"""
    
    @staticmethod
    def parse_args() -> argparse.Namespace:
        """解析命令行参数"""
        parser = argparse.ArgumentParser(
            description="语义密度分析系统 - 计算商品语义密度并分桶",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  # 基本使用
  python %(prog)s --dataset_path dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty 

  # 指定桶数量和K值
  python %(prog)s --dataset_path <path> --num_buckets 20 --k_neighbors 30
  
  # 打印样本
  python %(prog)s --dataset_path <path> --print_samples
            """
        )
        
        # 必需参数
        parser.add_argument(
            '--dataset_path',
            type=str,
            required=True,
            help='数据集路径（包含item_emb.parquet等文件的目录）'
        )
        
        # 可选参数
        parser.add_argument(
            '--num_buckets',
            type=int,
            default=10,
            help='桶数量（默认: 10）'
        )
        
        parser.add_argument(
            '--k_neighbors',
            type=int,
            default=20,
            help='KNN中的K值（默认: 20）'
        )
        
        parser.add_argument(
            '--print_samples',
            action='store_true',
            help='是否打印每个桶的随机样本'
        )
        
        parser.add_argument(
            '--random_seed',
            type=int,
            default=42,
            help='随机种子（默认: 42）'
        )
        
        parser.add_argument(
            '--output_dir',
            type=str,
            default='scripts/CollisionTest/output',
            help='输出目录（默认: scripts/CollisionTest/output）'
        )
        
        parser.add_argument(
            '--batch_size',
            type=int,
            default=1000,
            help='批处理大小（默认: 1000）'
        )
        
        args = parser.parse_args()
        return args
    
    @staticmethod
    def validate_args(args: argparse.Namespace) -> None:
        """验证参数有效性"""
        # 验证数据集路径
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(
                f"数据集路径不存在: {args.dataset_path}\n"
                f"请检查 --dataset_path 参数是否正确"
            )
        
        # 验证必需文件
        required_files = ['item_emb.parquet', 'item_mapping.npy']
        for filename in required_files:
            filepath = os.path.join(args.dataset_path, filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"缺少必需文件: {filepath}\n"
                    f"请确保数据集已经过预处理"
                )
        
        # 验证参数范围
        if args.num_buckets < 1:
            raise ValueError(f"num_buckets 必须 >= 1, 当前值: {args.num_buckets}")
        
        if args.k_neighbors < 1:
            raise ValueError(f"k_neighbors 必须 >= 1, 当前值: {args.k_neighbors}")
        
        if args.batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1, 当前值: {args.batch_size}")


# ============================================================================
# 2. 数据加载模块
# ============================================================================

class DataLoader:
    """数据加载器 - 加载商品向量和元数据"""
    
    def __init__(self, dataset_path: str):
        """
        初始化数据加载器
        
        Args:
            dataset_path: 数据集目录路径
        """
        self.dataset_path = dataset_path
        self.embeddings = None
        self.item_mapping = None
        self.metadata = None
        self.item_ids = None  # ItemID数组
        self.item_df = None  # 完整的DataFrame（包含title等信息）
        
    def load_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        加载商品向量数据
        
        Returns:
            embeddings: shape (N, 768) 的numpy数组
            item_ids: shape (N,) 的ItemID数组
        """
        parquet_path = os.path.join(self.dataset_path, 'item_emb.parquet')
        
        try:
            self.item_df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise IOError(f"无法加载 {parquet_path}: {str(e)}")
        
        # 提取embedding列（自动检测列名）
        # 支持 'attribute_embedding' (prism数据集) 和 'embedding' (tiger数据集)
        embedding_col = None
        if 'attribute_embedding' in self.item_df.columns:
            embedding_col = 'attribute_embedding'
        elif 'embedding' in self.item_df.columns:
            embedding_col = 'embedding'
        else:
            raise ValueError(
                f"item_emb.parquet 缺少 embedding 列\n"
                f"需要 'attribute_embedding' 或 'embedding' 列\n"
                f"可用列: {self.item_df.columns.tolist()}"
            )
        
        print(f"  使用embedding列: {embedding_col}")
        
        # 转换为numpy数组
        embeddings_list = self.item_df[embedding_col].tolist()
        self.embeddings = np.array(embeddings_list, dtype=np.float32)
        
        # 提取ItemID
        if 'ItemID' not in self.item_df.columns:
            raise ValueError(
                f"item_emb.parquet 缺少 'ItemID' 列\n"
                f"可用列: {self.item_df.columns.tolist()}"
            )
        
        self.item_ids = self.item_df['ItemID'].values
        
        return self.embeddings, self.item_ids
    
    def load_item_mapping(self) -> Dict[str, int]:
        """
        加载商品ID映射
        
        Returns:
            item_mapping: {ASIN: ItemID} 的字典
        """
        mapping_path = os.path.join(self.dataset_path, 'item_mapping.npy')
        
        try:
            self.item_mapping = np.load(mapping_path, allow_pickle=True).item()
        except Exception as e:
            raise IOError(f"无法加载 {mapping_path}: {str(e)}")
        
        return self.item_mapping
    
    def load_metadata(self) -> Optional[Dict]:
        """
        加载商品元数据（可选）
        
        注意：现在优先使用item_df中的title和brand信息
        
        Returns:
            metadata: {ItemID: {title, brand, ...}} 的字典，如果没有则返回None
        """
        # 如果已经加载了item_df，直接从中提取元数据
        if self.item_df is not None:
            metadata = {}
            for _, row in self.item_df.iterrows():
                item_id = row['ItemID']
                metadata[item_id] = {
                    'title': row.get('title', '[No title]'),
                    'brand': row.get('brand', '[No brand]')
                }
            self.metadata = metadata
            return metadata
        
        # 备选：尝试加载Beauty_metadata.json（旧版本兼容）
        metadata_path = os.path.join(self.dataset_path, 'Beauty_metadata.json')
        
        if not os.path.exists(metadata_path):
            print(f"  ⚠ 元数据文件不存在: {metadata_path}")
            print(f"  将使用item_emb.parquet中的信息")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except Exception as e:
            print(f"  ⚠ 无法加载元数据: {str(e)}")
            return None
        
        return self.metadata
    
    def get_item_count(self) -> int:
        """获取商品总数"""
        return len(self.item_ids) if self.item_ids is not None else 0
    
    def get_embedding_dim(self) -> int:
        """获取向量维度"""
        return self.embeddings.shape[1] if self.embeddings is not None else 0


# ============================================================================
# 3. 密度计算模块
# ============================================================================

class DensityCalculator:
    """密度计算器 - 使用KNN算法计算语义密度"""
    
    def __init__(self, k_neighbors: int = 20):
        """
        初始化密度计算器
        
        Args:
            k_neighbors: KNN中的K值
        """
        self.k_neighbors = k_neighbors
        self.embeddings = None
        self.n_items = 0
        
    def fit(self, embeddings: np.ndarray) -> None:
        """
        准备数据
        
        Args:
            embeddings: shape (N, D) 的numpy数组
        """
        self.embeddings = embeddings
        self.n_items = len(embeddings)
        
        # 验证K值
        if self.k_neighbors >= self.n_items:
            print(f"  ⚠ K值 ({self.k_neighbors}) >= 商品数量 ({self.n_items})")
            self.k_neighbors = max(1, self.n_items - 1)
            print(f"  自动调整K值为: {self.k_neighbors}")
    
    def calculate_density(self, batch_size: int = 1000) -> np.ndarray:
        """
        计算所有商品的语义密度（KNN距离）
        
        Args:
            batch_size: 批处理大小
        
        Returns:
            densities: shape (N,) 的numpy数组，表示每个商品的平均KNN距离
        """
        print(f"  使用KNN算法 (K={self.k_neighbors})")
        
        # 尝试使用FAISS（如果可用）
        try:
            import faiss
            return self._calculate_with_faiss()
        except ImportError:
            print(f"  FAISS不可用，使用sklearn")
            return self._calculate_with_sklearn(batch_size)
    
    def _calculate_with_faiss(self) -> np.ndarray:
        """使用FAISS计算KNN距离"""
        import faiss
        
        # 构建索引
        d = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(self.embeddings)
        
        # 搜索K+1个近邻（包括自己）
        distances, _ = index.search(self.embeddings, self.k_neighbors + 1)
        
        # 排除自己（第一个），计算平均距离
        # FAISS返回的是平方距离，需要开方
        knn_distances = np.sqrt(distances[:, 1:])  # 排除第一列（自己）
        avg_distances = np.mean(knn_distances, axis=1)
        
        return avg_distances
    
    def _calculate_with_sklearn(self, batch_size: int) -> np.ndarray:
        """使用sklearn计算KNN距离"""
        from sklearn.neighbors import NearestNeighbors
        
        # 构建KNN模型
        nbrs = NearestNeighbors(
            n_neighbors=self.k_neighbors + 1,  # +1 因为包括自己
            algorithm='auto',
            metric='euclidean',
            n_jobs=-1
        )
        nbrs.fit(self.embeddings)
        
        # 批处理计算
        densities = []
        n_batches = (self.n_items + batch_size - 1) // batch_size
        
        with tqdm(total=self.n_items, desc="  计算密度", unit="items") as pbar:
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, self.n_items)
                batch = self.embeddings[start_idx:end_idx]
                
                # 查询K+1个近邻
                distances, _ = nbrs.kneighbors(batch)
                
                # 排除自己（第一个），计算平均距离
                knn_distances = distances[:, 1:]  # 排除第一列（自己）
                avg_distances = np.mean(knn_distances, axis=1)
                
                densities.append(avg_distances)
                pbar.update(len(batch))
        
        return np.concatenate(densities)
    
    def get_statistics(self, densities: np.ndarray) -> Dict:
        """
        计算密度统计信息
        
        Args:
            densities: 密度数组
        
        Returns:
            stats: 统计信息字典
        """
        return {
            'min': float(np.min(densities)),
            'max': float(np.max(densities)),
            'mean': float(np.mean(densities)),
            'median': float(np.median(densities)),
            'std': float(np.std(densities))
        }



# ============================================================================
# 4. 分桶模块
# ============================================================================

class BucketingEngine:
    """分桶引擎 - 按语义密度将商品划分为多个桶"""
    
    def __init__(self, num_buckets: int = 10):
        """
        初始化分桶引擎
        
        Args:
            num_buckets: 桶数量
        """
        self.num_buckets = num_buckets
        self.buckets = None
        
    def create_buckets(
        self, 
        item_ids: np.ndarray, 
        densities: np.ndarray
    ) -> Dict[int, Dict]:
        """
        创建密度桶
        
        Args:
            item_ids: 商品ID数组
            densities: 密度数组
        
        Returns:
            buckets: {
                bucket_id: {
                    'items': [item_ids],
                    'densities': [densities],
                    'density_range': (min, max),
                    'count': int
                }
            }
        """
        n_items = len(item_ids)
        
        # 按密度排序（从小到大，即从高密度到低密度）
        sorted_indices = np.argsort(densities)
        sorted_item_ids = item_ids[sorted_indices]
        sorted_densities = densities[sorted_indices]
        
        # 均匀分桶
        self.buckets = {}
        items_per_bucket = n_items // self.num_buckets
        remainder = n_items % self.num_buckets
        
        start_idx = 0
        for bucket_id in range(self.num_buckets):
            # 处理余数：前remainder个桶多分配1个商品
            bucket_size = items_per_bucket + (1 if bucket_id < remainder else 0)
            end_idx = start_idx + bucket_size
            
            # 提取该桶的商品和密度
            bucket_items = sorted_item_ids[start_idx:end_idx]
            bucket_densities = sorted_densities[start_idx:end_idx]
            
            self.buckets[bucket_id] = {
                'items': bucket_items.tolist(),
                'densities': bucket_densities.tolist(),
                'density_range': (
                    float(bucket_densities.min()),
                    float(bucket_densities.max())
                ),
                'count': len(bucket_items)
            }
            
            start_idx = end_idx
        
        return self.buckets
    
    def get_bucket_statistics(self) -> List[Dict]:
        """
        获取每个桶的统计信息
        
        Returns:
            stats: 统计信息列表
        """
        if self.buckets is None:
            return []
        
        stats = []
        for bucket_id in sorted(self.buckets.keys()):
            bucket = self.buckets[bucket_id]
            densities = np.array(bucket['densities'])
            
            stats.append({
                'bucket_id': bucket_id,
                'count': bucket['count'],
                'density_range': bucket['density_range'],
                'mean_density': float(np.mean(densities)),
                'median_density': float(np.median(densities))
            })
        
        return stats


# ============================================================================
# 5. 随机抽样模块
# ============================================================================

class Sampler:
    """抽样器 - 从每个桶中随机抽取商品样本"""
    
    def __init__(self, random_seed: int = 42):
        """
        初始化抽样器
        
        Args:
            random_seed: 随机种子
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def sample_from_buckets(
        self,
        buckets: Dict[int, Dict],
        item_mapping: Dict[str, int],
        metadata: Optional[Dict],
        sample_size: int = 10
    ) -> Dict[int, List[Dict]]:
        """
        从每个桶中随机抽样
        
        Args:
            buckets: 分桶结果
            item_mapping: 商品ID映射 {ASIN: ItemID}
            metadata: 商品元数据 {ItemID: {title, brand, ...}}
            sample_size: 每个桶的样本数量
        
        Returns:
            samples: {
                bucket_id: [
                    {
                        'item_id': int,
                        'asin': str,
                        'title': str,
                        'brand': str,
                        'density': float
                    }
                ]
            }
        """
        # 创建反向映射 {ItemID: ASIN}
        reverse_mapping = {v: k for k, v in item_mapping.items()}
        
        samples = {}
        for bucket_id, bucket in buckets.items():
            items = bucket['items']
            densities = bucket['densities']
            
            # 确定样本数量
            n_samples = min(sample_size, len(items))
            
            # 随机抽样
            sample_indices = np.random.choice(
                len(items), 
                size=n_samples, 
                replace=False
            )
            
            bucket_samples = []
            for idx in sample_indices:
                item_id = items[idx]
                density = densities[idx]
                
                # 获取ASIN
                asin = reverse_mapping.get(item_id, f"Unknown_{item_id}")
                
                # 获取元数据（优先使用metadata字典，它现在是以ItemID为key）
                title = "[Metadata not available]"
                brand = "[Unknown]"
                
                if metadata and item_id in metadata:
                    title = metadata[item_id].get('title', title)
                    brand = metadata[item_id].get('brand', brand)
                
                bucket_samples.append({
                    'item_id': item_id,
                    'asin': asin,
                    'title': title,
                    'brand': brand,
                    'density': density
                })
            
            samples[bucket_id] = bucket_samples
        
        return samples
    
    def print_samples(self, samples: Dict[int, List[Dict]]) -> None:
        """
        打印样本到控制台
        
        Args:
            samples: 抽样结果
        """
        print("\n" + "=" * 80)
        print("随机样本展示")
        print("=" * 80)
        
        for bucket_id in sorted(samples.keys()):
            bucket_samples = samples[bucket_id]
            
            print(f"\n桶 {bucket_id} (共 {len(bucket_samples)} 个样本):")
            print("-" * 80)
            
            for i, sample in enumerate(bucket_samples, 1):
                print(f"\n  样本 {i}:")
                print(f"    ItemID: {sample['item_id']}")
                print(f"    ASIN: {sample['asin']}")
                print(f"    密度: {sample['density']:.4f}")
                print(f"    品牌: {sample.get('brand', '[Unknown]')}")
                print(f"    标题: {sample['title']}")  # 不限制长度，完整显示
        
        print("\n" + "=" * 80)


# ============================================================================
# 6. 结果输出模块
# ============================================================================

class OutputWriter:
    """输出写入器 - 保存分析结果到文件"""
    
    def __init__(self, output_dir: str):
        """
        初始化输出写入器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def write_density_csv(
        self,
        item_ids: np.ndarray,
        densities: np.ndarray,
        bucket_ids: np.ndarray,
        dataset_name: str,
        k_neighbors: int,
        num_buckets: int
    ) -> str:
        """
        写入密度CSV文件
        
        Args:
            item_ids: 商品ID数组
            densities: 密度数组
            bucket_ids: 桶ID数组
            dataset_name: 数据集名称
            k_neighbors: K值
            num_buckets: 桶数量
        
        Returns:
            output_path: 输出文件路径
        """
        # 构建DataFrame
        df = pd.DataFrame({
            'ItemID': item_ids,
            'SemanticDensity': densities,
            'BucketID': bucket_ids
        })
        
        # 生成文件名
        filename = f"{dataset_name}_density_k{k_neighbors}_buckets{num_buckets}.csv"
        output_path = os.path.join(self.output_dir, filename)
        
        # 保存
        df.to_csv(output_path, index=False)
        
        return output_path
    
    def write_bucket_stats_json(
        self,
        bucket_stats: List[Dict],
        dataset_name: str,
        k_neighbors: int,
        num_buckets: int,
        total_items: int
    ) -> str:
        """
        写入分桶统计JSON文件
        
        Args:
            bucket_stats: 分桶统计信息
            dataset_name: 数据集名称
            k_neighbors: K值
            num_buckets: 桶数量
            total_items: 商品总数
        
        Returns:
            output_path: 输出文件路径
        """
        # 构建JSON数据
        data = {
            'dataset_name': dataset_name,
            'num_buckets': num_buckets,
            'k_neighbors': k_neighbors,
            'total_items': total_items,
            'buckets': bucket_stats
        }
        
        # 生成文件名
        filename = f"{dataset_name}_bucket_stats_k{k_neighbors}_buckets{num_buckets}.json"
        output_path = os.path.join(self.output_dir, filename)
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path


# ============================================================================
# 7. 主流程
# ============================================================================

def main():
    """主函数"""
    start_time = time.time()
    
    # 解析参数
    args = CLIParser.parse_args()
    
    print("\n" + "=" * 80)
    print("语义密度分析系统")
    print("=" * 80)
    
    # 验证参数
    try:
        CLIParser.validate_args(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n错误: {str(e)}")
        sys.exit(1)
    
    # 打印配置
    print("\n配置参数:")
    print(f"  数据集路径: {args.dataset_path}")
    print(f"  桶数量: {args.num_buckets}")
    print(f"  K值: {args.k_neighbors}")
    print(f"  随机种子: {args.random_seed}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  批处理大小: {args.batch_size}")
    print(f"  打印样本: {args.print_samples}")
    
    # 提取数据集名称
    dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
    
    # ========================================================================
    # 步骤1: 加载数据
    # ========================================================================
    print("\n[1/5] 加载数据...")
    loader = DataLoader(args.dataset_path)
    
    try:
        embeddings, item_ids = loader.load_embeddings()
        print(f"  ✓ 加载商品向量: {len(item_ids)}个商品, {embeddings.shape[1]}维")
        
        item_mapping = loader.load_item_mapping()
        print(f"  ✓ 加载商品映射: {len(item_mapping)}个映射")
        
        metadata = loader.load_metadata()
        if metadata:
            print(f"  ✓ 加载元数据: {len(metadata)}个商品")
    except Exception as e:
        print(f"\n错误: 数据加载失败")
        print(f"  {str(e)}")
        sys.exit(1)
    
    # ========================================================================
    # 步骤2: 计算语义密度
    # ========================================================================
    print("\n[2/5] 计算语义密度...")
    calculator = DensityCalculator(k_neighbors=args.k_neighbors)
    calculator.fit(embeddings)
    
    try:
        densities = calculator.calculate_density(batch_size=args.batch_size)
        print(f"  ✓ 完成")
    except Exception as e:
        print(f"\n错误: 密度计算失败")
        print(f"  {str(e)}")
        sys.exit(1)
    
    # ========================================================================
    # 步骤3: 密度统计
    # ========================================================================
    print("\n[3/5] 密度统计:")
    stats = calculator.get_statistics(densities)
    print(f"  最小值: {stats['min']:.4f}")
    print(f"  最大值: {stats['max']:.4f}")
    print(f"  平均值: {stats['mean']:.4f}")
    print(f"  中位数: {stats['median']:.4f}")
    print(f"  标准差: {stats['std']:.4f}")
    
    # ========================================================================
    # 步骤4: 创建密度桶
    # ========================================================================
    print("\n[4/5] 创建密度桶...")
    bucketing = BucketingEngine(num_buckets=args.num_buckets)
    buckets = bucketing.create_buckets(item_ids, densities)
    bucket_stats = bucketing.get_bucket_statistics()
    
    print(f"  ✓ 创建{args.num_buckets}个桶")
    for stat in bucket_stats:
        print(f"  桶{stat['bucket_id']}: {stat['count']}个商品, "
              f"密度范围 [{stat['density_range'][0]:.4f}, {stat['density_range'][1]:.4f}]")
    
    # ========================================================================
    # 步骤4.5: 随机抽样（可选）
    # ========================================================================
    if args.print_samples:
        print("\n[4.5/5] 随机抽样...")
        sampler = Sampler(random_seed=args.random_seed)
        samples = sampler.sample_from_buckets(
            buckets, 
            item_mapping, 
            metadata, 
            sample_size=10
        )
        sampler.print_samples(samples)
    
    # ========================================================================
    # 步骤5: 保存结果
    # ========================================================================
    print("\n[5/5] 保存结果...")
    writer = OutputWriter(args.output_dir)
    
    # 创建桶ID数组（与item_ids对应）
    bucket_id_array = np.zeros(len(item_ids), dtype=int)
    for bucket_id, bucket in buckets.items():
        for item_id in bucket['items']:
            idx = np.where(item_ids == item_id)[0][0]
            bucket_id_array[idx] = bucket_id
    
    # 写入CSV
    csv_path = writer.write_density_csv(
        item_ids,
        densities,
        bucket_id_array,
        dataset_name,
        args.k_neighbors,
        args.num_buckets
    )
    print(f"  ✓ CSV文件: {csv_path}")
    
    # 写入JSON
    json_path = writer.write_bucket_stats_json(
        bucket_stats,
        dataset_name,
        args.k_neighbors,
        args.num_buckets,
        len(item_ids)
    )
    print(f"  ✓ JSON文件: {json_path}")
    
    # ========================================================================
    # 完成
    # ========================================================================
    elapsed_time = time.time() - start_time
    print(f"\n总耗时: {elapsed_time:.1f}秒")
    print("分析完成！")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
