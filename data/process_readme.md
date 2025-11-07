# Amazon Dataset Preprocessing Guide

## Overview

This guide explains how to use the parameterized preprocessing script `process.py` to prepare Amazon review datasets for TIGER and other recommendation models with 5-core filtering and proper data splitting.

## Features

✅ **Parameterized execution** - Configure datasets via command-line arguments  
✅ **5-core filtering** - Ensure data quality by filtering users/items with <5 interactions  
✅ **Proper data splitting** - Last item for test, second-to-last for validation  
✅ **Multiple embedding modes** - Support for TIGER and HID-VAE modes  
✅ **Flexible model selection** - Choose from available embedding models  
✅ **Comprehensive logging** - All processing details saved to log files  
✅ **Sample inspection** - Print metadata samples for verification  
✅ **Multi-dataset support** - Process Beauty, Sports, Toys, and other Amazon datasets  

---

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--review_path` | Path to reviews_*.json.gz file | `path/to/reviews_Beauty.json.gz` |
| `--meta_path` | Path to meta_*.json.gz file | `path/to/meta_Beauty.json.gz` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `Beauty` | Dataset name (e.g., Beauty, Sports, Toys) |
| `--output_dir` | `.` | Output directory for processed data |
| `--min_interactions` | `5` | Minimum interactions per user/item (5-core filtering) |
| `--long_tail_mode` | `False` | Enable long-tail analysis with grouped validation/test sets |
| `--tail_thresholds` | `[5,10,20,30]` | Thresholds for long-tail grouping |
| `--embed_mode` | `tiger` | Embedding mode: `tiger` or `hid-vae` |
| `--embed_model` | `sentence-t5` | Embedding model to use |
| `--model_source` | `modelscope` | Model download source: `huggingface` or `modelscope` |
| `--max_tags` | `5` | Max category tags for hid-vae mode |
| `--print_samples` | `100` | Number of samples to print |

---

## 5-Core Filtering

The preprocessing pipeline implements **5-core filtering** to ensure data quality:

### What is 5-Core Filtering?

5-core filtering ensures that:
- Each **user** has at least 5 interactions (reviews)
- Each **item** has at least 5 interactions (reviews)


### Benefits

- **Quality Assurance**: Removes sparse users and items
- **Model Performance**: Improves recommendation accuracy
- **Training Stability**: Reduces noise in the dataset

This is a iterative process, all the items and users fit the condition after multi-rounds.

**Tips: 5-Core filting is compatible with long-tail mode, long-tail mode will process the filtered data by 5-core filting. 
However, if min_interactions is set to 5, the 0-5 test group in long-tail mode will be empty.
So you can set min_interactions to 1 when you want to execute standard long-tail experiment.**



## Long-Tail Analysis Mode

The script supports **long-tail analysis mode** for studying recommendation performance on items with different popularity levels.

### What is Long-Tail Mode?

Long-tail mode creates separate validation and test sets grouped by target item popularity:
- **Training Data**: Includes ALL interactions (both popular and long-tail items)
- **Validation/Test Data**: Grouped by target item interaction counts

### Popularity Groups

Default thresholds `[5, 10, 20, 30]` create these groups:
- `tail_0-5`: Items with 1-5 interactions (very long-tail)
- `tail_5-10`: Items with 6-10 interactions (long-tail)
- `tail_10-20`: Items with 11-20 interactions (medium popularity)
- `tail_20-30`: Items with 21-30 interactions (popular)
- `tail_30plus`: Items with 30+ interactions (very popular)

### Custom Thresholds

You can customize grouping thresholds:

```bash
python process.py \
    --long_tail_mode \
    --tail_thresholds 3 8 15 25 50 \  # Custom thresholds
    ...
```

This creates groups: `(0-3]`, `(3-8]`, `(8-15]`, `(15-25]`, `(25-50]`, `(50+)`

---

## Embedding Modes


## Model Download and Management

The preprocessing script automatically handles embedding model download and management:

### Automatic Model Download

1. **Local Check**: First checks if model exists at local path (e.g., `./sentence-t5-base/`)
2. **Auto Download**: If not found locally, downloads from specified source
3. **Source Selection**: Choose between HuggingFace and ModelScope via `--model_source`

### 1. TIGER Mode (Default)

**Description:** Concatenates all item attributes into a single text and generates one embedding per item.

**Format:**
```
title: [item title]
price: [price]
salesRank: [sales rank]
brand: [brand]
categories: [categories list]
```

**Output:** Single `embedding` column in `item_emb.parquet`


### 2. HID-VAE Mode

**Description:** Separates category tags and other attributes into different embeddings.

**Output Structure:**
- `category_embeddings`: List of up to `max_tags` category embeddings (padded with zeros)
- `attribute_embedding`: Single embedding for title, price, salesRank, brand
- `num_categories`: Actual number of categories (before padding)
---



## Usage Examples

### Example 1: Process Beauty Dataset (TIGER mode with 5-core filtering)

```bash
python process.py \
    --dataset Beauty \
    --review_path "../dataset/Amazon-Beauty/reviews_Beauty.json.gz" \
    --meta_path "../dataset/Amazon-Beauty/meta_Beauty.json.gz" \
    --output_dir "../dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base" \
    --min_interactions 5 \
    --embed_mode tiger \
    --embed_model sentence-t5 \
    --model_source modelscope \
    --print_samples 30
```

**Output**: All processing details will be saved to `Beauty_preprocessing.log`

### Example 2: Process Sports Dataset (HID-VAE mode with strict filtering)

```bash
python process.py \
    --dataset Sports \
    --review_path "path/to/reviews_Sports_and_Outdoors.json.gz" \
    --meta_path "path/to/meta_Sports_and_Outdoors.json.gz" \
    --min_interactions 10 \
    --embed_mode hid-vae \
    --model_source modelscope \
    --max_tags 5
```

**Output**: Stricter 10-core filtering with detailed logging in `Sports_preprocessing.log`

### Example 3: Long-Tail Analysis Mode

```bash
python process.py \
    --dataset Beauty \
    --review_path "../dataset/Amazon-Beauty/reviews_Beauty.json.gz" \
    --meta_path "../dataset/Amazon-Beauty/meta_Beauty.json.gz" \
    --min_interactions 1 \
    --long_tail_mode \
    --tail_thresholds 5 10 20 30 \
    --embed_mode tiger
```

**Output Files**:
- `train.parquet` (includes all data)
- `valid_tail_0-5.parquet`, `valid_tail_5-10.parquet`, etc.
- `test_tail_0-5.parquet`, `test_tail_5-10.parquet`, etc.
- Complete statistics in `Beauty_preprocessing.log`



## Data Split Strategy

The script uses a **sequential leave-one-out** strategy designed for sequential recommendation:

### Split Logic

For a user with interaction sequence `[1, 2, 3, 4, 5]`:

- **Training:** Use `[1, 2]` to predict `3` (third-to-last item)
- **Validation:** Use `[1, 2, 3]` to predict `4` (second-to-last item)  
- **Test:** Use `[1, 2, 3, 4]` to predict `5` (last item)

### Key Features

- **Temporal Order**: Respects chronological order of interactions
- **No Future Leakage**: Each split only uses past information
- **Progressive Evaluation**: Training → Validation → Test in chronological order
- **Minimum Length**: Requires ≥3 interactions per user (≥4 for training data)

---

## Adding New Embedding Models

To add a new embedding model:

1. Edit `process.py` and add to `EMBEDDING_MODELS` dictionary:

```python
EMBEDDING_MODELS = {
    'sentence-t5': './sentence-t5-base',
    'your-model': './path/to/your-model',
}
```

2. Use the new model:

```bash
python process.py \
    --embed_model your-model \
    ...
```

---

## Supported Amazon Datasets

The script supports any Amazon review dataset with the standard format:

- ✅ Beauty
- ✅ Sports and Outdoors
- ✅ Toys and Games
- ✅ Electronics
- ✅ Books
- ✅ Movies and TV
- ✅ Clothing, Shoes and Jewelry
- ✅ Home and Kitchen
- ✅ And more...

Just specify the correct paths and dataset name!

---

## Troubleshooting

### Error: Model not found

**Problem:** `Error loading model: [Errno 2] No such file or directory`

**Solution:** Ensure the embedding model is downloaded:
```bash
# For sentence-t5-base
python down_model.py
```

### Error: Review file not found

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:** Verify the paths to `.json.gz` files are correct. Use absolute paths if needed.

### Memory Issues

**Problem:** Out of memory when processing large datasets

**Solution:** Process embeddings in batches or use a machine with more RAM.

---


## References

- [Amazon Review Dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- [TIGER Paper](https://arxiv.org/pdf/2305.05065)
- [Sentence Transformers](https://www.sbert.net/)

