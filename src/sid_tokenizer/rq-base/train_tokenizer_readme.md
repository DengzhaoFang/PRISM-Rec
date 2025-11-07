# Semantic ID Tokenizer Training

A comprehensive framework for training semantic ID tokenizers for recommendation systems, supporting multiple quantization approaches including TIGER and K-means clustering.

## Overview

This framework provides tools to train semantic ID tokenizers that convert item embeddings into discrete tokens for efficient recommendation model training. It supports two main approaches:

- **TIGER Mode**: Uses Residual Quantized VAE (RQ-VAE) for hierarchical semantic ID generation
- **K-means Mode**: Uses Residual Quantized K-means (RQ-KMeans) for clustering-based tokenization

## Features

✅ **Multiple Training Modes** - Support for TIGER and K-means approaches  
✅ **Hierarchical Quantization** - Multi-layer residual quantization for rich representations  
✅ **Efficient Training** - Optimized training loops with checkpointing and monitoring  
✅ **Flexible Configuration** - Command-line interface with extensive customization  
✅ **Automatic Evaluation** - Built-in reconstruction loss computation and model evaluation  
✅ **Export Capabilities** - Generate semantic ID mappings for downstream use  

## Quick Start

### Basic Usage

```bash
# Train TIGER mode tokenizer
python train_tokenizer.py \
    --data_path ../dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty \
    --output_dir ./output/tiger_tokenizer \
    --mode tiger \
    --n_layers 3 \
    --n_embed 512 \
    --latent_dim 32 \
    --epochs 300 \
    --batch_size 1024 \
    --learning_rate 4e-1 \
    --save_every 20 \
    --device cuda \
    --log_level INFO

# Train K-means mode tokenizer  
python train_tokenizer.py \
    --data_path ../dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty \
    --output_dir ./output/kmeans_tokenizer \
    --mode kmeans \
    --n_clusters 512
```

### Example Output Structure

```
output/tiger_tokenizer/
├── final_model.pt              # Trained model
├── semantic_id_mappings.json   # Item ID -> Semantic ID mappings
├── training_results.json       # Training statistics
└── checkpoint_epoch_*.pt      # Training checkpoints

logs/
└── 20241014_153025_Beauty_tiger_ep100_bs1024.log  # Timestamped training logs
```

## Command Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data_path` | Path to processed dataset directory | `./Beauty/` |
| `--output_dir` | Output directory for results | `./output/` |

### Training Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `tiger` | Training mode (`tiger` or `kmeans`) |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `512` | Training batch size |
| `--learning_rate` | `1e-3` | Learning rate for TIGER mode |
| `--device` | `auto` | Device to use (`auto`/`cpu`/`cuda`) |

### Model Architecture

| Argument | Default | Description |
|----------|---------|-------------|
| `--n_layers` | `4` | Number of quantization layers |
| `--n_embed` | `512` | Embeddings per layer (TIGER) |
| `--n_clusters` | `512` | Clusters per layer (K-means) |
| `--latent_dim` | `128` | Latent dimension (TIGER) |

### Utility Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_items` | `None` | Max items for testing |
| `--save_every` | `10` | Checkpoint frequency |
| `--log_level` | `INFO` | Logging verbosity |

## Training Modes

### TIGER Mode (RQ-VAE)

Uses a Residual Quantized Variational AutoEncoder for hierarchical representation learning:

**Architecture:**
- **Encoder**: Multi-layer neural network that maps embeddings to latent space
- **Quantizers**: Multiple VQ layers applied residually for hierarchical coding
- **Decoder**: Reconstruction network for training stability

## Architecture Details

### Model Architecture
```
Input (768-dim sentence-t5 embeddings)
    ↓
Encoder: 768 → 512 → 256 → 128 → 32
    ↓
3-Layer Residual Quantization:
  - Layer 0: 256 codebook vectors (32-dim each)
  - Layer 1: 256 codebook vectors (32-dim each)  
  - Layer 2: 256 codebook vectors (32-dim each)
    ↓
Decoder: 32 → 128 → 256 → 512 → 768
    ↓
Reconstructed embeddings (768-dim)
```

### Loss Function
Following TIGER paper formulation:
```
L_rqvae = Σ[||sg[r_i] - e_{c_i}||² + β ||r_i - sg[e_{c_i}]||²]
```
Where:
- `sg[]` = stop-gradient operation
- `r_i` = residual at layer i
- `e_{c_i}` = selected codebook vector
- `β = 0.25` (commitment loss weight)

**Key Features:**
- K-means clustering initialization (TIGER paper method)
- EMA updates （Codebook update：through EMA （dont need gradients） so codebook loss=0 is normal）
- Commitment loss for encoder-quantizer alignment
- Hierarchical semantic representations
- Post-ID deduplication for unique semantic IDs

**Use Case**: Best for complex semantic relationships and fine-grained item representations.

**Example Configuration:**
```bash
python train_tokenizer.py \
    --mode tiger \
    --n_layers 4 \
    --n_embed 512 \
    --latent_dim 128 \
    --epochs 100 \
    --learning_rate 1e-3
```

**Optimization**:

**EMA Codebook Update (Recommended):**
- Uses Exponential Moving Average to update codebook
- More stable than gradient-based updates
- Prevents codebook collapse
- Always uses STE (Straight-Through Estimator) for quantization
- Set with `--use_ema` (enabled by default)

**Alternative: Standard VQ with Gumbel-Softmax:**
- For non-EMA mode (`--no_ema`)
- Uses gradient descent to update codebook
- Optional Gumbel-Softmax for smoother gradients
- Less stable but more flexible

**Other Improvements:**
- AdamW Optimizer (lr=1e-3 for EMA, 1e-4 for standard)
- SiLU Activation (smoother gradients, no dead neurons)
- K-means initialization (prevents initial collapse)
- Early stopping (prevents over-training)
- Batch size: 128-256 (more frequent updates)
- Architecture: Removed bias terms for better scaling



### K-means Mode (RQ-KMeans)

Uses Residual Quantized K-means clustering for efficient tokenization:

**Process:**
1. Apply K-means clustering to embeddings
2. Compute residuals after quantization
3. Repeat for multiple layers hierarchically
4. Generate discrete codes for each layer

**Key Features:**
- K-means++ initialization for better convergence
- Iterative residual quantization
- Fast training with no backpropagation needed

**Use Case**: Ideal for large-scale datasets requiring fast training and inference.

**Example Configuration:**
```bash
python train_tokenizer.py \
    --mode kmeans \
    --n_layers 4 \
    --n_clusters 512 \
    --max_iters 100
```

## Data Requirements

### Input Format

The training script expects a processed dataset directory containing:

```
data_path/
├── item_emb.parquet    # Item embeddings (required)
├── train.parquet       # Training data (optional)
├── valid.parquet       # Validation data (optional)
└── test.parquet        # Test data (optional)
```

### Embedding File Structure

The `item_emb.parquet` file should contain:
- `ItemID`: Integer item identifiers
- `embedding`: List of float values representing item embeddings

Example:
```python
import pandas as pd
df = pd.read_parquet('item_emb.parquet')
print(df.head())
#    ItemID                    embedding
# 0       1  [0.1, 0.2, 0.3, ..., 0.8]
# 1       2  [0.4, 0.1, 0.7, ..., 0.2]
```

## Output Files

### Model Files

- **`final_model.pt`**: Complete trained model with all parameters
- **`checkpoint_epoch_*.pt`**: Training checkpoints for resuming

### Mappings and Results

- **`semantic_id_mappings.json`**: Maps item IDs to semantic ID sequences
- **`training_results.json`**: Training statistics and metrics

### Log Files

Log files are automatically saved in the `logs/` directory with timestamped names containing key training information:

**Format**: `{timestamp}_{dataset}_{mode}_ep{epochs}_bs{batch_size}.log`

**Examples**:
- `20241014_153025_Beauty_tiger_ep20000_bs1024.log`
- `20241014_160342_Sports_kmeans_ep100_bs512.log`

**Log contents include**:
- Training progress and loss curves
- Duplicate rate statistics (for TIGER mode)
- Codebook usage statistics  
- Model architecture details
- Hyperparameter configuration

### Semantic ID Mappings Format

```json
{
  "1": [45, 123, 67, 289],     # Item 1 -> 4-layer semantic ID
  "2": [12, 456, 78, 134],     # Item 2 -> 4-layer semantic ID
  ...
}
```

## Advanced Usage

### Custom Training Configuration

```python
# Example: Fine-tuned TIGER training
python train_tokenizer.py \
    --data_path ./Beauty/ \
    --output_dir ./output/custom_tiger \
    --mode tiger \
    --n_layers 6 \
    --n_embed 1024 \
    --latent_dim 256 \
    --epochs 200 \
    --batch_size 256 \
    --learning_rate 5e-4 \
    --save_every 20
```

### Testing with Limited Data

```bash
# Quick test with 1000 items
python train_tokenizer.py \
    --data_path ./Beauty/ \
    --output_dir ./test_output \
    --max_items 1000 \
    --epochs 10
```



## Performance Considerations

### Memory Usage

- **TIGER Mode**: Higher memory usage due to neural network parameters
- **K-means Mode**: Lower memory, but requires loading all embeddings simultaneously

### Training Speed

- **TIGER Mode**: Slower convergence, benefits from GPU acceleration
- **K-means Mode**: Fast training, mostly CPU-based operations

### Scalability

- **Small datasets** (< 10K items): Both modes work well
- **Medium datasets** (10K-100K items): TIGER recommended for quality
- **Large datasets** (> 100K items): K-means recommended for speed



### Validation

Check model quality by examining reconstruction loss in training logs:
```
INFO - Epoch 50: Loss=0.1234, Recon=0.0567, VQ=0.0667
```

Lower reconstruction loss indicates better item representation quality.

## Integration with Recommendation Models

The generated semantic IDs can be used in various recommendation architectures:

1. **Sequence Models**: Use semantic IDs as input tokens for transformer-based models
2. **Embedding Models**: Replace item embeddings with learned semantic representations  
3. **Generative Models**: Train language models to predict next semantic IDs


## References

- [TIGER: A Neural Approach for Text-to-Image Generation with Rich Attributes](https://arxiv.org/pdf/2305.05065)
- [Neural Discrete Representation Learning (VQ-VAE)](https://arxiv.org/abs/1711.00937)
- [Mini-Batch K-Means: A Faster Alternative to K-Means](https://dl.acm.org/doi/abs/10.1145/1772690.1772862)
