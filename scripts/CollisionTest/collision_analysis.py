#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
碰撞率分析系统 (Collision Rate Analysis System)

该脚本实现完整的碰撞率分析流程：
1. 阶段1: 训练RQ-VAE Tokenizer，生成语义ID映射
2. 阶段2: 基于密度分桶计算碰撞率，验证"语义冷门商品更容易碰撞"的理论假设

使用示例:
    # 完整流程（训练 + 分析）
    python scripts/CollisionTest/collision_analysis.py \
    --mode full \
    --dataset_path dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty \
    --density_file scripts/CollisionTest/output/Beauty_density_k20_buckets10.csv \
    --output_dir scripts/CollisionTest/output/tokenizer

    
    # 仅训练tokenizer
    python scripts/CollisionTest/collision_analysis.py \
        --mode train \
        --dataset_path <path> \
        --output_dir <output>
    
    # 仅分析碰撞率（需要已有的语义ID映射）
    python scripts/CollisionTest/collision_analysis.py \
        --mode analyze \
        --density_file <density_csv> \
        --mappings_file <mappings_json> \
        --output_dir <output>
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm

# 添加src路径以导入RQ-VAE模型
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/sid_tokenizer/rq-base'))

from tiger.RQ_VAE import RQVAE, QuantizeMode


# ============================================================================
# 1. 数据加载模块
# ============================================================================

class ItemEmbeddingDataset(Dataset):
    """商品embedding数据集"""
    
    def __init__(self, embedding_file: str, max_items: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            embedding_file: item_emb.parquet文件路径
            max_items: 最大商品数（用于测试）
        """
        self.embeddings_df = pd.read_parquet(embedding_file)
        
        if max_items is not None:
            self.embeddings_df = self.embeddings_df.head(max_items)
        
        # 转换embeddings为tensor
        # 使用attribute_embedding列（beauty-prism数据集的列名）
        embedding_col = 'attribute_embedding' if 'attribute_embedding' in self.embeddings_df.columns else 'embedding'
        self.embeddings = torch.stack([
            torch.tensor(emb, dtype=torch.float32) 
            for emb in self.embeddings_df[embedding_col]
        ])
        
        self.item_ids = self.embeddings_df['ItemID'].values
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'item_id': self.item_ids[idx],
            'embedding': self.embeddings[idx]
        }


# ============================================================================
# 2. Tokenizer训练模块
# ============================================================================

class TokenizerTrainer:
    """RQ-VAE Tokenizer训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config: 训练配置字典，包含：
                - dataset_path: 数据集路径
                - output_dir: 输出目录
                - n_layers: RQ-VAE层数（固定为3）
                - n_embed: 每层码本大小（固定为256）
                - latent_dim: 潜在向量维度
                - epochs: 训练轮数
                - batch_size: 批大小
                - learning_rate: 学习率
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        log_dir = self.config['output_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode='w', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志文件: {log_file}")
        
    def load_data(self) -> Tuple[DataLoader, List[str], torch.Tensor]:
        """
        加载数据
        
        Returns:
            dataloader: 数据加载器
            item_ids: 商品ID列表
            embeddings: 所有embeddings的tensor
        """
        self.logger.info(f"加载数据: {self.config['dataset_path']}")
        
        embedding_file = os.path.join(
            self.config['dataset_path'], 
            'item_emb.parquet'
        )
        
        dataset = ItemEmbeddingDataset(
            embedding_file=embedding_file,
            max_items=self.config.get('max_items')
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.logger.info(f"加载了 {len(dataset)} 个商品")
        
        return dataloader, dataset.item_ids, dataset.embeddings


    def train(self) -> Tuple[RQVAE, Dict[str, List[int]]]:
        """
        训练RQ-VAE模型
        
        Returns:
            model: 训练好的模型
            semantic_mappings: ItemID -> [code1, code2, code3, post_id]
        """
        # 加载数据
        dataloader, item_ids, all_embeddings = self.load_data()
        
        # 初始化模型
        input_dim = all_embeddings.shape[1]  # 768
        self.logger.info(f"初始化RQ-VAE模型: input_dim={input_dim}, latent_dim={self.config['latent_dim']}")
        
        # 获取quantize_mode（与TIGER训练脚本对齐）
        quantize_mode_str = self.config.get('quantize_mode', 'gumbel_softmax')
        if quantize_mode_str == 'gumbel_softmax':
            quantize_mode = QuantizeMode.GUMBEL_SOFTMAX
        elif quantize_mode_str == 'ste':
            quantize_mode = QuantizeMode.STE
        else:
            quantize_mode = QuantizeMode.ROTATION
        
        model = RQVAE(
            input_dim=input_dim,
            latent_dim=self.config['latent_dim'],
            n_layers=self.config['n_layers'],
            n_embed=self.config['n_embed'],
            beta=self.config.get('beta', 0.25),
            use_ema=self.config.get('use_ema', True),
            decay=self.config.get('ema_decay', 0.99),
            commitment_weight=self.config.get('commitment_weight', 1.0),
            reconstruction_weight=self.config.get('reconstruction_weight', 1.0),
            quantize_mode=quantize_mode,
            normalize_residuals=self.config.get('normalize_residuals', True)  # 关键参数！
        ).to(self.device)
        
        # 使用AdamW优化器（与TIGER训练脚本对齐）
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.999)
        )
        
        self.logger.info(f"模型配置:")
        self.logger.info(f"  - n_layers: {self.config['n_layers']}")
        self.logger.info(f"  - n_embed: {self.config['n_embed']}")
        self.logger.info(f"  - latent_dim: {self.config['latent_dim']}")
        self.logger.info(f"  - beta: {self.config.get('beta', 0.25)}")
        self.logger.info(f"  - use_ema: {self.config.get('use_ema', True)}")
        self.logger.info(f"  - ema_decay: {self.config.get('ema_decay', 0.99)}")
        self.logger.info(f"  - quantize_mode: {quantize_mode_str}")
        self.logger.info(f"  - normalize_residuals: {self.config.get('normalize_residuals', True)}")
        self.logger.info(f"  - optimizer: AdamW")
        self.logger.info(f"  - learning_rate: {self.config['learning_rate']}")
        self.logger.info(f"  - batch_size: {self.config['batch_size']}")
        self.logger.info(f"  - epochs: {self.config['epochs']}")
        
        # 温度调度参数（与TIGER训练脚本对齐）
        init_temperature = self.config.get('init_temperature', 1.0)
        min_temperature = self.config.get('min_temperature', 0.2)
        temperature_schedule = self.config.get('temperature_schedule', 'cosine')
        total_steps = self.config['epochs'] * len(dataloader)
        warmup_steps = self.config.get('temperature_warmup_steps', 1000)
        
        # 训练循环
        self.logger.info(f"开始训练: {self.config['epochs']} epochs")
        model.train()
        
        global_step = 0
        
        # 早停策略状态（与TIGER训练脚本对齐）
        best_loss = float('inf')
        patience_counter = 0
        early_stop_patience = self.config.get('early_stop_patience', 30)
        early_stop_min_delta = self.config.get('early_stop_min_delta', 1e-5)
        
        for epoch in range(self.config['epochs']):
            total_loss = 0
            total_recon_loss = 0
            total_codebook_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for batch in pbar:
                embeddings = batch['embedding'].to(self.device)
                
                # 计算当前温度（与TIGER训练脚本对齐）
                if temperature_schedule == 'cosine':
                    if global_step < warmup_steps:
                        temperature = init_temperature
                    else:
                        progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                        progress = min(progress, 1.0)
                        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                        temperature = min_temperature + (init_temperature - min_temperature) * cosine_decay
                elif temperature_schedule == 'constant':
                    temperature = min_temperature
                else:
                    # exponential decay
                    anneal_rate = self.config.get('temperature_anneal_rate', 0.00003)
                    temperature = max(min_temperature, init_temperature * np.exp(-anneal_rate * global_step))
                
                # 前向传播 - 返回字典
                outputs = model(embeddings, temperature=temperature)
                
                loss = outputs['total_loss']
                recon_loss = outputs['recon_loss']
                codebook_loss = outputs['codebook_loss']
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪（与TIGER训练脚本对齐）
                if self.config.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'])
                
                optimizer.step()
                
                global_step += 1
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_codebook_loss += codebook_loss.item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}',
                    'codebook': f'{codebook_loss.item():.4f}'
                })
            
            avg_loss = total_loss / len(dataloader)
            avg_recon = total_recon_loss / len(dataloader)
            avg_codebook = total_codebook_loss / len(dataloader)
            
            self.logger.info(
                f"Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                f"Recon={avg_recon:.4f}, Codebook={avg_codebook:.4f}"
            )
            
            # 早停检查（与TIGER训练脚本对齐）
            if early_stop_patience > 0:
                if best_loss - avg_loss > early_stop_min_delta:
                    best_loss = avg_loss
                    patience_counter = 0
                    self.logger.info(f"  ✓ 新的最佳loss: {best_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        self.logger.info(f"\n⚠ 早停触发，在第 {epoch+1} 轮停止训练")
                        self.logger.info(f"  训练loss在 {early_stop_patience} 轮内没有改善")
                        break
            
            # 定期保存checkpoint
            if (epoch + 1) % 100 == 0:
                checkpoint_path = os.path.join(
                    self.config['output_dir'], 
                    f'checkpoint_epoch_{epoch+1}.pt'
                )
                self.save_model(model, checkpoint_path)
                self.logger.info(f"保存checkpoint: {checkpoint_path}")
        
        # 生成语义ID映射
        self.logger.info("生成语义ID映射...")
        semantic_mappings = self.generate_semantic_mappings(model, item_ids, all_embeddings)
        
        return model, semantic_mappings
    
    def generate_semantic_mappings(
        self, 
        model: RQVAE, 
        item_ids: List[str], 
        embeddings: torch.Tensor
    ) -> Dict[str, List[int]]:
        """
        生成语义ID映射（使用GLOBAL post-ID deduplication，与TIGER训练脚本一致）
        
        Args:
            model: 训练好的模型
            item_ids: 商品ID列表
            embeddings: 商品embeddings
        
        Returns:
            semantic_mappings: ItemID -> [code1, code2, code3, post_id]
        """
        model.eval()
        
        with torch.no_grad():
            embeddings = embeddings.to(self.device)
            
            # Step 1: 生成3层codes for all items
            self.logger.info("Step 1: 生成3层codes...")
            item_to_3layer = {}
            batch_size = 256
            for i in tqdm(range(0, len(embeddings), batch_size), desc="生成3层codes"):
                batch_emb = embeddings[i:i+batch_size]
                
                # 使用encode_to_codes方法获取量化码（不应用post-ID）
                codes = model.encode_to_codes(batch_emb, apply_post_id=False)
                
                # codes的shape: (batch_size, n_layers)
                for j in range(len(batch_emb)):
                    item_id = str(item_ids[i + j])  # 转换为字符串
                    # 提取3层code并转换为tuple
                    code_tuple = tuple(codes[j].cpu().tolist())
                    item_to_3layer[item_id] = code_tuple
            
            # Step 2: 应用GLOBAL post-ID deduplication
            self.logger.info("Step 2: 应用GLOBAL post-ID去重...")
            tuple_to_items = defaultdict(list)
            for item_id, code_tuple in item_to_3layer.items():
                tuple_to_items[code_tuple].append(item_id)
            
            # Step 3: 分配唯一的第4层codes
            self.logger.info("Step 3: 分配唯一的第4层codes...")
            semantic_mappings = {}
            for code_tuple, item_ids_group in tuple_to_items.items():
                for idx, item_id in enumerate(sorted(item_ids_group)):
                    # 转换为Python native types for JSON serialization
                    code_4layer = [int(c) for c in code_tuple] + [int(idx)]
                    semantic_mappings[item_id] = code_4layer
            
            # 计算去重率
            total_items = len(semantic_mappings)
            unique_3layer = len(tuple_to_items)
            dup_rate_pre = 1.0 - (unique_3layer / total_items)
            dup_rate_post = 0.0  # 应该为0（全局去重后）
            
            self.logger.info(f"最终去重率:")
            self.logger.info(f"  去重前: {dup_rate_pre:.4f} ({total_items - unique_3layer}/{total_items} 重复)")
            self.logger.info(f"  去重后: {dup_rate_post:.4f} (0/{total_items} 重复)")
        
        self.logger.info(f"生成了 {len(semantic_mappings)} 个商品的语义ID映射")
        return semantic_mappings
    
    def save_model(self, model: RQVAE, save_path: str):
        """保存模型"""
        torch.save(model.state_dict(), save_path)
        self.logger.info(f"模型已保存: {save_path}")
    
    def save_mappings(self, mappings: Dict[str, List[int]], save_path: str):
        """保存语义ID映射"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        self.logger.info(f"语义ID映射已保存: {save_path}")


# ============================================================================
# 3. 碰撞率分析模块
# ============================================================================

class CollisionAnalyzer:
    """碰撞率分析器"""
    
    def __init__(
        self,
        density_buckets_path: str,
        semantic_mappings_path: str,
        output_dir: str
    ):
        """
        初始化分析器
        
        Args:
            density_buckets_path: 密度分桶CSV文件路径
            semantic_mappings_path: 语义ID映射JSON文件路径
            output_dir: 输出目录
        """
        self.density_buckets_path = density_buckets_path
        self.semantic_mappings_path = semantic_mappings_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.output_dir, f'analysis_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file, mode='w', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志文件: {log_file}")


    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, List[int]], Dict[str, Any]]:
        """
        加载数据
        
        Returns:
            density_df: 密度分桶DataFrame
            semantic_mappings: 语义ID映射
            bucket_stats: 桶统计信息
        """
        self.logger.info("加载数据...")
        
        # 加载密度分桶
        self.logger.info(f"加载密度分桶: {self.density_buckets_path}")
        density_df = pd.read_csv(self.density_buckets_path)
        
        # 加载语义ID映射
        self.logger.info(f"加载语义ID映射: {self.semantic_mappings_path}")
        with open(self.semantic_mappings_path, 'r', encoding='utf-8') as f:
            semantic_mappings = json.load(f)
        
        # 加载桶统计信息（如果存在）
        bucket_stats_path = self.density_buckets_path.replace('_density_', '_bucket_stats_').replace('.csv', '.json')
        if os.path.exists(bucket_stats_path):
            self.logger.info(f"加载桶统计: {bucket_stats_path}")
            with open(bucket_stats_path, 'r', encoding='utf-8') as f:
                bucket_stats = json.load(f)
        else:
            self.logger.warning(f"桶统计文件不存在: {bucket_stats_path}")
            bucket_stats = {}
        
        self.logger.info(f"加载完成: {len(density_df)} 个商品, {len(semantic_mappings)} 个映射")
        
        return density_df, semantic_mappings, bucket_stats
    
    def calculate_collision_rate(
        self,
        bucket_items: List[str],
        semantic_mappings: Dict[str, List[int]],
        layer: int = 3
    ) -> Dict[str, Any]:
        """
        计算指定桶的碰撞率
        
        Args:
            bucket_items: 桶内商品ID列表
            semantic_mappings: 语义ID映射
            layer: 计算前N层的碰撞率（1-4）
        
        Returns:
            碰撞率统计字典
        """
        total_items = len(bucket_items)
        collision_groups = defaultdict(int)
        missing_items = []
        
        # 统计每个code的出现次数
        for item_id in bucket_items:
            # 确保item_id是字符串
            item_id_str = str(item_id)
            
            if item_id_str not in semantic_mappings:
                missing_items.append(item_id)
                continue
            
            codes = semantic_mappings[item_id_str]
            code_prefix = codes[:layer]
            code_str = '-'.join(map(str, code_prefix))
            collision_groups[code_str] += 1
        
        if missing_items:
            self.logger.warning(f"缺失 {len(missing_items)} 个商品的语义ID映射")
        
        # 计算统计指标
        valid_items = total_items - len(missing_items)
        unique_codes = len(collision_groups)
        collision_rate = 1 - (unique_codes / valid_items) if valid_items > 0 else 0
        max_collision_group_size = max(collision_groups.values()) if collision_groups else 0
        
        # 找出实际碰撞组（count > 1）
        actual_collision_groups = {
            code: count for code, count in collision_groups.items() if count > 1
        }
        
        return {
            'total_items': total_items,
            'valid_items': valid_items,
            'missing_items': len(missing_items),
            'unique_codes': unique_codes,
            'collision_rate': collision_rate,
            'max_collision_group_size': max_collision_group_size,
            'num_collision_groups': len(actual_collision_groups),
            'collision_groups': actual_collision_groups
        }
    
    def calculate_layer_collision_rates(
        self,
        bucket_items: List[str],
        semantic_mappings: Dict[str, List[int]]
    ) -> List[float]:
        """
        计算每一层的碰撞率
        
        Args:
            bucket_items: 桶内商品ID列表
            semantic_mappings: 语义ID映射
        
        Returns:
            每层的碰撞率列表 [layer1, layer2, layer3, layer4]
        """
        layer_rates = []
        for layer in range(1, 5):
            stats = self.calculate_collision_rate(bucket_items, semantic_mappings, layer)
            layer_rates.append(stats['collision_rate'])
        return layer_rates
    
    def analyze_all_buckets(self) -> Dict[str, Any]:
        """
        分析所有桶的碰撞率
        
        Returns:
            完整的碰撞率统计结果
        """
        density_df, semantic_mappings, bucket_stats = self.load_data()
        
        self.logger.info("开始分析碰撞率...")
        
        # 检查列名并标准化
        if 'BucketID' in density_df.columns:
            density_df = density_df.rename(columns={'BucketID': 'bucket_id'})
        
        # 按桶分组
        buckets = density_df.groupby('bucket_id')
        
        results = {
            'dataset': bucket_stats.get('dataset_name', 'Unknown'),
            'num_buckets': bucket_stats.get('num_buckets', len(buckets)),
            'total_items': len(density_df),
            'buckets': []
        }
        
        for bucket_id, group in buckets:
            self.logger.info(f"分析桶 {bucket_id}...")
            
            bucket_items = group['ItemID'].tolist()
            
            # 计算3层和4层碰撞率
            stats_3layer = self.calculate_collision_rate(bucket_items, semantic_mappings, layer=3)
            stats_4layer = self.calculate_collision_rate(bucket_items, semantic_mappings, layer=4)
            
            # 计算每层碰撞率
            layer_rates = self.calculate_layer_collision_rates(bucket_items, semantic_mappings)
            
            # 获取密度信息
            bucket_info = None
            if 'buckets' in bucket_stats:
                bucket_info = next((b for b in bucket_stats['buckets'] if b['bucket_id'] == bucket_id), None)
            
            bucket_result = {
                'bucket_id': int(bucket_id),
                'total_items': stats_3layer['total_items'],
                'valid_items': stats_3layer['valid_items'],
                'missing_items': stats_3layer['missing_items'],
                'unique_3layer_codes': stats_3layer['unique_codes'],
                'unique_4layer_codes': stats_4layer['unique_codes'],
                'collision_rate_3layer': stats_3layer['collision_rate'],
                'collision_rate_4layer': stats_4layer['collision_rate'],
                'max_collision_group_size': stats_3layer['max_collision_group_size'],
                'num_collision_groups': stats_3layer['num_collision_groups'],
                'layer_collision_rates': layer_rates,
                'top_collision_groups': dict(sorted(
                    stats_3layer['collision_groups'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10])  # 只保留前10个最大碰撞组
            }
            
            # 添加密度信息
            if bucket_info:
                bucket_result['density_range'] = bucket_info['density_range']
                bucket_result['mean_density'] = bucket_info['mean_density']
                bucket_result['median_density'] = bucket_info['median_density']
            
            results['buckets'].append(bucket_result)
            
            self.logger.info(
                f"桶 {bucket_id}: 碰撞率(3层)={stats_3layer['collision_rate']:.4f}, "
                f"碰撞率(4层)={stats_4layer['collision_rate']:.4f}"
            )
        
        return results


    def generate_report(self, stats: Dict[str, Any]):
        """
        生成碰撞率对比报告
        
        Args:
            stats: 碰撞率统计结果
        """
        report_path = os.path.join(self.output_dir, 'collision_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("碰撞率分析报告 (Collision Rate Analysis Report)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"数据集: {stats['dataset']}\n")
            f.write(f"总商品数: {stats['total_items']}\n")
            f.write(f"桶数量: {stats['num_buckets']}\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("桶级碰撞率统计\n")
            f.write("-" * 80 + "\n\n")
            
            # 表头
            f.write(f"{'桶ID':^6} | {'商品数':^8} | {'唯一Code':^10} | {'碰撞率(3层)':^12} | "
                   f"{'碰撞率(4层)':^12} | {'最大碰撞组':^10} | {'平均密度':^10}\n")
            f.write("-" * 80 + "\n")
            
            # 数据行
            for bucket in stats['buckets']:
                f.write(
                    f"{bucket['bucket_id']:^6} | "
                    f"{bucket['total_items']:^8} | "
                    f"{bucket['unique_3layer_codes']:^10} | "
                    f"{bucket['collision_rate_3layer']:^12.4f} | "
                    f"{bucket['collision_rate_4layer']:^12.4f} | "
                    f"{bucket['max_collision_group_size']:^10} | "
                    f"{bucket.get('mean_density', 0):^10.4f}\n"
                )
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("层级碰撞率分析\n")
            f.write("-" * 80 + "\n\n")
            
            f.write(f"{'桶ID':^6} | {'Layer1':^10} | {'Layer2':^10} | {'Layer3':^10} | {'Layer4':^10}\n")
            f.write("-" * 80 + "\n")
            
            for bucket in stats['buckets']:
                layer_rates = bucket['layer_collision_rates']
                f.write(
                    f"{bucket['bucket_id']:^6} | "
                    f"{layer_rates[0]:^10.4f} | "
                    f"{layer_rates[1]:^10.4f} | "
                    f"{layer_rates[2]:^10.4f} | "
                    f"{layer_rates[3]:^10.4f}\n"
                )
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("理论假设验证\n")
            f.write("=" * 80 + "\n\n")
            
            # 验证假设：密度低的桶碰撞率更高
            bucket_0 = stats['buckets'][0]
            bucket_last = stats['buckets'][-1]
            
            f.write(f"假设: 语义密度低的商品（冷门商品）更容易发生碰撞\n\n")
            
            f.write(f"桶0（最低密度）:\n")
            f.write(f"  - 平均密度: {bucket_0.get('mean_density', 0):.4f}\n")
            f.write(f"  - 碰撞率(3层): {bucket_0['collision_rate_3layer']:.4f}\n\n")
            
            f.write(f"桶{bucket_last['bucket_id']}（最高密度）:\n")
            f.write(f"  - 平均密度: {bucket_last.get('mean_density', 0):.4f}\n")
            f.write(f"  - 碰撞率(3层): {bucket_last['collision_rate_3layer']:.4f}\n\n")
            
            if bucket_0['collision_rate_3layer'] > bucket_last['collision_rate_3layer']:
                f.write("✓ 结论: 假设成立！密度低的桶碰撞率更高。\n")
            else:
                f.write("✗ 结论: 假设不成立。密度低的桶碰撞率并未更高。\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Top碰撞组示例（桶0）\n")
            f.write("=" * 80 + "\n\n")
            
            for code, count in list(bucket_0['top_collision_groups'].items())[:5]:
                f.write(f"Code: {code} -> {count} 个商品\n")
        
        self.logger.info(f"报告已生成: {report_path}")
        
        # 同时打印到控制台
        with open(report_path, 'r', encoding='utf-8') as f:
            print("\n" + f.read())
    
    def save_stats(self, stats: Dict[str, Any]):
        """保存统计结果为JSON"""
        stats_path = os.path.join(self.output_dir, 'collision_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        self.logger.info(f"统计结果已保存: {stats_path}")
    
    def visualize(self, stats: Dict[str, Any]):
        """
        生成可视化图表
        
        Args:
            stats: 碰撞率统计结果
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
        except ImportError:
            self.logger.warning("matplotlib未安装，跳过可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        bucket_ids = [b['bucket_id'] for b in stats['buckets']]
        collision_rates_3layer = [b['collision_rate_3layer'] for b in stats['buckets']]
        mean_densities = [b.get('mean_density', 0) for b in stats['buckets']]
        
        # 图1: 碰撞率 vs 桶ID
        axes[0, 0].bar(bucket_ids, collision_rates_3layer, color='steelblue')
        axes[0, 0].set_xlabel('Bucket ID')
        axes[0, 0].set_ylabel('Collision Rate (3-layer)')
        axes[0, 0].set_title('Collision Rate by Bucket')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 图2: 碰撞率 vs 密度
        # 检查密度数据是否有效
        valid_densities = [d for d in mean_densities if d != 0.0]
        
        if len(valid_densities) > 1 and len(set(valid_densities)) > 1:
            # 有有效的密度数据且不全相同，绘制散点图和趋势线
            axes[0, 1].scatter(mean_densities, collision_rates_3layer, color='coral', s=100)
            axes[0, 1].set_xlabel('Mean Density')
            axes[0, 1].set_ylabel('Collision Rate (3-layer)')
            axes[0, 1].set_title('Collision Rate vs Density')
            axes[0, 1].grid(alpha=0.3)
            
            # 添加趋势线（只在有有效数据时）
            try:
                z = np.polyfit(mean_densities, collision_rates_3layer, 1)
                p = np.poly1d(z)
                axes[0, 1].plot(mean_densities, p(mean_densities), "r--", alpha=0.8, label='Trend')
                axes[0, 1].legend()
            except (np.linalg.LinAlgError, ValueError) as e:
                self.logger.warning(f"无法拟合趋势线: {str(e)}")
        else:
            # 密度数据无效或全为0，改为绘制碰撞率 vs 桶ID
            axes[0, 1].plot(bucket_ids, collision_rates_3layer, marker='o', color='coral', linewidth=2)
            axes[0, 1].set_xlabel('Bucket ID (Low Density → High Density)')
            axes[0, 1].set_ylabel('Collision Rate (3-layer)')
            axes[0, 1].set_title('Collision Rate Trend Across Buckets')
            axes[0, 1].grid(alpha=0.3)
            if len(valid_densities) == 0:
                self.logger.warning("密度数据全为0，使用桶ID代替")
            else:
                self.logger.warning("密度数据缺失或无效，使用桶ID代替")
        
        # 图3: 层级碰撞率对比
        layer_labels = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
        x = np.arange(len(layer_labels))
        width = 0.15
        
        for i, bucket in enumerate(stats['buckets'][:5]):  # 只显示前5个桶
            offset = (i - 2) * width
            axes[1, 0].bar(
                x + offset, 
                bucket['layer_collision_rates'], 
                width, 
                label=f'Bucket {bucket["bucket_id"]}'
            )
        
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Collision Rate')
        axes[1, 0].set_title('Layer-wise Collision Rates (First 5 Buckets)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(layer_labels)
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 图4: 唯一Code数量
        unique_codes = [b['unique_3layer_codes'] for b in stats['buckets']]
        total_items = [b['total_items'] for b in stats['buckets']]
        
        axes[1, 1].plot(bucket_ids, unique_codes, marker='o', label='Unique Codes', color='green')
        axes[1, 1].plot(bucket_ids, total_items, marker='s', label='Total Items', color='orange')
        axes[1, 1].set_xlabel('Bucket ID')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Unique Codes vs Total Items')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'collision_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"可视化图表已保存: {plot_path}")


# ============================================================================
# 4. 主流程控制
# ============================================================================

class CollisionRateAnalysis:
    """碰撞率分析主控制器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化分析系统
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        运行完整流程：训练 + 分析
        
        Returns:
            碰撞率统计结果
        """
        print("\n" + "=" * 80)
        print("碰撞率分析系统 - 完整流程")
        print("=" * 80 + "\n")
        
        # 阶段1: 训练Tokenizer
        print("阶段1: 训练RQ-VAE Tokenizer")
        print("-" * 80)
        
        trainer = TokenizerTrainer(self.config)
        model, semantic_mappings = trainer.train()
        
        # 保存模型和映射
        model_path = os.path.join(self.config['output_dir'], 'tokenizer.pt')
        mappings_path = os.path.join(self.config['output_dir'], 'semantic_id_mappings.json')
        
        trainer.save_model(model, model_path)
        trainer.save_mappings(semantic_mappings, mappings_path)
        
        print("\n✓ 阶段1完成\n")
        
        # 阶段2: 碰撞率分析
        print("阶段2: 碰撞率分析")
        print("-" * 80)
        
        analyzer = CollisionAnalyzer(
            density_buckets_path=self.config['density_file'],
            semantic_mappings_path=mappings_path,
            output_dir=self.config['output_dir']
        )
        
        stats = analyzer.analyze_all_buckets()
        analyzer.save_stats(stats)
        analyzer.generate_report(stats)
        analyzer.visualize(stats)
        
        print("\n✓ 阶段2完成\n")
        print("=" * 80)
        print("分析完成！")
        print("=" * 80 + "\n")
        
        return stats


    def run_train_only(self) -> Tuple[RQVAE, Dict[str, List[int]]]:
        """
        仅运行训练阶段
        
        Returns:
            model: 训练好的模型
            semantic_mappings: 语义ID映射
        """
        print("\n" + "=" * 80)
        print("碰撞率分析系统 - 仅训练Tokenizer")
        print("=" * 80 + "\n")
        
        trainer = TokenizerTrainer(self.config)
        model, semantic_mappings = trainer.train()
        
        # 保存模型和映射
        model_path = os.path.join(self.config['output_dir'], 'tokenizer.pt')
        mappings_path = os.path.join(self.config['output_dir'], 'semantic_id_mappings.json')
        
        trainer.save_model(model, model_path)
        trainer.save_mappings(semantic_mappings, mappings_path)
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80 + "\n")
        
        return model, semantic_mappings
    
    def run_analyze_only(self) -> Dict[str, Any]:
        """
        仅运行分析阶段（需要已有的语义ID映射）
        
        Returns:
            碰撞率统计结果
        """
        print("\n" + "=" * 80)
        print("碰撞率分析系统 - 仅分析碰撞率")
        print("=" * 80 + "\n")
        
        analyzer = CollisionAnalyzer(
            density_buckets_path=self.config['density_file'],
            semantic_mappings_path=self.config['mappings_file'],
            output_dir=self.config['output_dir']
        )
        
        stats = analyzer.analyze_all_buckets()
        analyzer.save_stats(stats)
        analyzer.generate_report(stats)
        analyzer.visualize(stats)
        
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80 + "\n")
        
        return stats


# ============================================================================
# 5. 命令行接口
# ============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="碰撞率分析系统 - 验证语义冷门商品碰撞假设",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（训练 + 分析）
  python %(prog)s --mode full \\
      --dataset_path dataset/Amazon-Beauty/processed/beauty-prism-sentenceT5base/Beauty \\
      --density_file scripts/CollisionTest/output/Beauty_density_k20_buckets10.csv \\
      --output_dir scripts/CollisionTest/output/tokenizer
  
  # 仅训练tokenizer
  python %(prog)s --mode train \\
      --dataset_path dataset/Amazon-Beauty/processed/beauty-prism-sentenceT5base/Beauty \\
      --output_dir scripts/CollisionTest/output/tokenizer
  
  # 仅分析碰撞率
  python %(prog)s --mode analyze \\
      --density_file scripts/CollisionTest/output/Beauty_density_k20_buckets10.csv \\
      --mappings_file scripts/CollisionTest/output/tokenizer/semantic_id_mappings.json \\
      --output_dir scripts/CollisionTest/output
        """
    )
    
    # 运行模式
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'train', 'analyze'],
        default='full',
        help='运行模式: full=完整流程, train=仅训练, analyze=仅分析'
    )
    
    # 数据路径
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='数据集路径（包含item_emb.parquet的目录）'
    )
    
    parser.add_argument(
        '--density_file',
        type=str,
        help='密度分桶CSV文件路径'
    )
    
    parser.add_argument(
        '--mappings_file',
        type=str,
        help='语义ID映射JSON文件路径（仅analyze模式需要）'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='scripts/CollisionTest/output/tokenizer',
        help='输出目录（默认: scripts/CollisionTest/output/tokenizer）'
    )
    
    # 训练参数
    parser.add_argument(
        '--n_layers',
        type=int,
        default=3,
        help='RQ-VAE层数（默认: 3）'
    )
    
    parser.add_argument(
        '--n_embed',
        type=int,
        default=256,
        help='每层码本大小（默认: 256）'
    )
    
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=32,
        help='潜在向量维度（默认: 32）'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='训练轮数（默认: 500，与TIGER训练脚本对齐）'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='批大小（默认: 512，与TIGER训练脚本对齐）'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='学习率（默认: 1e-4，与TIGER训练脚本对齐）'
    )
    
    parser.add_argument(
        '--max_items',
        type=int,
        default=None,
        help='最大商品数（用于测试，默认: None=全部）'
    )
    
    # 额外的训练参数（与TIGER训练脚本对齐）
    parser.add_argument(
        '--beta',
        type=float,
        default=0.25,
        help='VQ-VAE beta系数（默认: 0.25）'
    )
    
    parser.add_argument(
        '--use_ema',
        action='store_true',
        default=True,
        help='使用EMA更新码本（默认: True）'
    )
    
    parser.add_argument(
        '--ema_decay',
        type=float,
        default=0.99,
        help='EMA衰减率（默认: 0.99）'
    )
    
    parser.add_argument(
        '--quantize_mode',
        type=str,
        default='gumbel_softmax',
        choices=['gumbel_softmax', 'ste', 'rotation'],
        help='量化模式（默认: gumbel_softmax，与TIGER训练脚本对齐）'
    )
    
    parser.add_argument(
        '--commitment_weight',
        type=float,
        default=1.0,
        help='Commitment loss权重（默认: 1.0）'
    )
    
    parser.add_argument(
        '--reconstruction_weight',
        type=float,
        default=1.0,
        help='Reconstruction loss权重（默认: 1.0）'
    )
    
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
        help='AdamW权重衰减（默认: 0.01）'
    )
    
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=1.0,
        help='梯度裁剪阈值（默认: 1.0）'
    )
    
    parser.add_argument(
        '--init_temperature',
        type=float,
        default=1.0,
        help='初始温度（默认: 1.0）'
    )
    
    parser.add_argument(
        '--min_temperature',
        type=float,
        default=0.2,
        help='最小温度（默认: 0.2）'
    )
    
    parser.add_argument(
        '--temperature_schedule',
        type=str,
        default='cosine',
        choices=['cosine', 'exponential', 'constant'],
        help='温度调度策略（默认: cosine）'
    )
    
    parser.add_argument(
        '--temperature_warmup_steps',
        type=int,
        default=1000,
        help='温度预热步数（默认: 1000）'
    )
    
    # 早停策略参数（与TIGER训练脚本对齐）
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=30,
        help='早停耐心值：loss在N轮内不改善则停止训练（默认: 30，与TIGER对齐）'
    )
    
    parser.add_argument(
        '--early_stop_min_delta',
        type=float,
        default=1e-5,
        help='早停最小改善阈值（默认: 1e-5，与TIGER对齐）'
    )
    
    parser.add_argument(
        '--normalize_residuals',
        action='store_true',
        default=True,
        help='是否对残差进行L2归一化（默认: True，CRITICAL for stability!）'
    )
    
    parser.add_argument(
        '--no_normalize_residuals',
        action='store_false',
        dest='normalize_residuals',
        help='禁用残差归一化'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 验证参数
    if args.mode in ['full', 'train']:
        if not args.dataset_path:
            print("错误: --dataset_path 是必需的（full/train模式）")
            sys.exit(1)
    
    if args.mode in ['full', 'analyze']:
        if not args.density_file:
            print("错误: --density_file 是必需的（full/analyze模式）")
            sys.exit(1)
    
    if args.mode == 'analyze':
        if not args.mappings_file:
            print("错误: --mappings_file 是必需的（analyze模式）")
            sys.exit(1)
    
    # 构建配置
    config = {
        'dataset_path': args.dataset_path,
        'density_file': args.density_file,
        'mappings_file': args.mappings_file,
        'output_dir': args.output_dir,
        'n_layers': args.n_layers,
        'n_embed': args.n_embed,
        'latent_dim': args.latent_dim,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_items': args.max_items,
        # 额外的训练参数
        'beta': args.beta,
        'use_ema': args.use_ema,
        'ema_decay': args.ema_decay,
        'quantize_mode': args.quantize_mode,
        'commitment_weight': args.commitment_weight,
        'reconstruction_weight': args.reconstruction_weight,
        'weight_decay': args.weight_decay,
        'grad_clip': args.grad_clip,
        'init_temperature': args.init_temperature,
        'min_temperature': args.min_temperature,
        'temperature_schedule': args.temperature_schedule,
        'temperature_warmup_steps': args.temperature_warmup_steps,
        # 早停策略参数
        'early_stop_patience': args.early_stop_patience,
        'early_stop_min_delta': args.early_stop_min_delta,
        # RQ-VAE架构参数
        'normalize_residuals': args.normalize_residuals
    }
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行分析
    analyzer = CollisionRateAnalysis(config)
    
    if args.mode == 'full':
        results = analyzer.run_full_pipeline()
    elif args.mode == 'train':
        model, mappings = analyzer.run_train_only()
    elif args.mode == 'analyze':
        results = analyzer.run_analyze_only()


if __name__ == '__main__':
    main()
