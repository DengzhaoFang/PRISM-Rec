# ActionPiece Tokenizer

ActionPiece 是一种针对生成式推荐系统的上下文感知分词方法。

## 论文

**ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation**

## 核心思想

ActionPiece 将用户行为序列视为"特征集合序列"，通过类似 BPE 的算法学习上下文感知的词表：

1. **特征提取**: 使用 PQ (Product Quantization) 将商品文本嵌入量化为 m 个离散码
2. **Hash Bucket**: 添加 hash bucket 确保每个 item 的语义 ID 全局唯一（1-to-1 映射）
3. **词表构建**: 使用加权共现统计和中间节点机制学习合并规则
4. **分词**: 使用 SPR (Set Permutation Regularization) 进行随机分词增强

## 语义 ID 的唯一性保证

ActionPiece 通过 **hash bucket** 机制确保每个 item 都有唯一的语义 ID：

- OPQ 量化后，不同 item 可能有相同的 PQ codes
- 添加 hash bucket 作为第 5 个特征，确保全局唯一
- 最终语义 ID 格式：`[code1, code2, code3, code4, hash_bucket]`

## 使用方法

### 1. 训练 ActionPiece Tokenizer

```bash
cd scripts/ActionPiece
bash train_tokenizer.sh
```

或者直接运行 Python 脚本：

```bash
python -m src.sid_tokenizer.ActionPiece.train_tokenizer \
    --data_path dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty \
    --embedding_file item_emb.parquet \
    --output_dir scripts/output/actionpiece_tokenizer/beauty \
    --pq_n_codebooks 4 \
    --pq_codebook_size 256 \
    --n_hash_buckets 128 \
    --vocab_size 40000
```

### 2. 输出文件

训练完成后，输出目录包含：

- `semantic_id_mappings.json`: 商品到语义 ID 的映射（与框架兼容）
- `actionpiece.json`: ActionPiece 分词器（包含词表和合并规则）
- `item2feat.json`: 商品到特征的映射（与 semantic_id_mappings.json 相同）
- `item2sem_ids.json`: 商品到原始 PQ codes 的映射（不含 hash bucket，仅供参考）
- `stats.json`: 统计信息
- `config.json`: 训练配置

## 参数说明

### PQ 参数（论文设置）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pq_n_codebooks` | 4 | PQ 码本数量 (m=4) |
| `pq_codebook_size` | 256 | 每个码本的大小 |
| `n_hash_buckets` | 128 | 用于处理碰撞的 hash bucket 数量 |

### ActionPiece 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocab_size` | 40000 | 目标词表大小 |
| `n_threads` | 4 | Faiss 使用的线程数 |

## 与 TIGER 的区别

| 特性 | TIGER | ActionPiece |
|------|-------|-------------|
| 量化方法 | RQ-VAE | Faiss OPQ (Optimized Product Quantization) |
| 语义 ID 结构 | 层次化 (hierarchical) | 无序集合 (unordered set) + hash bucket |
| 分词方法 | 固定偏移 | BPE-like 合并 |
| 数据增强 | 无 | SPR (随机排列) |
| 唯一性保证 | 第 4 层 post-ID | hash bucket |

## 依赖

- faiss-cpu 或 faiss-gpu (用于 OPQ 量化)
- numpy
- pandas
- tqdm

## 参考

论文原始代码位于 `src/action_piece/genrec/models/ActionPiece/`
