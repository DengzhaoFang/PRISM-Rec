# HID-VAE: Hierarchical ID VAE with Tag-Guided Learning

## Overview

HID-VAE (Hierarchical ID VAE) is an advanced semantic ID generation model that addresses the key limitations of standard RQ-VAE:

- **Low codebook utilization**
- **Codebook collapse**
- **ID collisions**

### Key Innovations

1. **Multi-Modal Inputs**: Combines content embeddings (768D) and collaborative embeddings (64D)
2. **Tag-Guided Learning**: Uses hierarchical category tags as supervision signals
3. **Multiple Loss Components**:
   - Cosine similarity reconstruction loss (scale-invariant)
   - Tag anchoring loss (semantic guidance)
   - Codebook balance loss (prevent collapse)
   - Hierarchical classification loss (tag prediction)
4. **Collision Resolution**: Sinkhorn algorithm for post-training ID reassignment

---

## Architecture

### Components

```
Input: Content Emb (768D) + Collaborative Emb (64D)
   ↓
Encoder: Multi-modal fusion → Latent (32D)
   ↓
RQ-VAE: 3-layer hierarchical quantization
   ├─ Layer 1 (Codebook 1: 256 codes) → c1
   ├─ Layer 2 (Codebook 2: 256 codes) → c2
   └─ Layer 3 (Codebook 3: 256 codes) → c3
   ↓
Quantized: z_q = c1 + c2 + c3
   ↓
Decoder: Multi-modal reconstruction
   ├─ Content Decoder → Content Recon (768D)
   └─ Collab Decoder → Collab Recon (64D)
   ↓
Classifiers: Hierarchical tag prediction
   ├─ Classifier 1: c1 → L2 tag (6 classes)
   ├─ Classifier 2: [c1; c2] → L3 tag (37 classes)
   └─ Classifier 3: [c1; c2; c3] → L4 tag (148 classes)
```

### Loss Functions

**Total Loss**:
```
L_total = L_recon + L_anchor + L_balance + L_class + L_commit
```

1. **Reconstruction Loss** (L_recon):
   - Content: `1 - CosineSimilarity(pred_content, target_content)`
   - Collab: `1 - CosineSimilarity(pred_collab, target_collab)`
   - Weight: λ_content=1.0, λ_collab=1.0

2. **Tag Anchoring Loss** (L_anchor):
   - Projects tag embeddings (768D) to codebook space (32D)
   - MSE between projected tags and quantized tags
   - Guides codebooks to align with semantic categories

3. **Codebook Balance Loss** (L_balance):
   - KL divergence between observed and uniform distribution
   - Prevents codebook collapse by encouraging uniform usage

4. **Classification Loss** (L_class):
   - Cross-entropy for hierarchical tag prediction
   - Masked for variable-length tag sequences
   - Weight: δ=1.0

5. **Commitment Loss** (L_commit):
   - Standard VQ commitment loss
   - Weight: β=0.25

---

## File Structure

```
src/sid_tokenizer/hidvae/
├── HID_VAE.py                    # Main model architecture
├── hid_losses.py                 # Loss function implementations
├── hierarchical_classifiers.py   # Tag classifiers
├── multimodal_dataset.py         # Dataset loader
├── train_hidvae.py              # Training script
├── generate_semantic_ids.py     # ID generation and analysis
├── sinkhorn_reassignment.py     # Collision resolution
└── HID_VAE_README.md            # This file

scripts/hidvae/
└── train_hidvae_beauty.sh       # Training script for Beauty dataset
```

---

## Usage

### 1. Training

#### Basic Training
```bash
cd /home/fangdengzhao/SID-GR
source ./.venv/bin/activate

python src/sid_tokenizer/hidvae/train_hidvae.py \
    --data_path dataset/Amazon-Beauty/processed/beauty-hidvae-sentenceT5base/Beauty \
    --output_dir scripts/output/hidvae_tokenizer/beauty/test_run \
    --n_layers 3 \
    --n_embed 256 \
    --latent_dim 32 \
    --epochs 500 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --lambda_content 1.0 \
    --lambda_collab 1.0 \
    --delta_weight 1.0 \
    --beta 0.25 \
    --use_ema \
    --ema_decay 0.99 \
    --use_scheduler \
    --early_stop_patience 200 \
    --device cuda
```

#### Using Bash Script
```bash
cd /home/fangdengzhao/SID-GR
source ./.venv/bin/activate
bash scripts/hidvae/train_hidvae_beauty.sh
```

### 2. Generate Semantic IDs

After training, generate semantic IDs for all items:

```bash
python src/sid_tokenizer/hidvae/generate_semantic_ids.py \
    --checkpoint scripts/output/hidvae_tokenizer/beauty/test_run/best_model.pt \
    --data_dir dataset/Amazon-Beauty/processed/beauty-hidvae-sentenceT5base/Beauty \
    --output_dir scripts/output/hidvae_tokenizer/beauty/test_run/semantic_ids \
    --device cuda \
    --batch_size 512
```

**Outputs**:
- `semantic_ids.parquet`: All semantic IDs with metadata
- `semantic_ids.npy`: Numpy array of IDs
- `id_statistics.json`: Comprehensive statistics
- `collisions.csv`: Details of colliding IDs

**Statistics Computed**:
- ID uniqueness rate
- Collision rate
- Hierarchical overlap rates (prefix sharing)
- Tag prediction accuracy
- Codebook usage statistics

### 3. Resolve Collisions (Optional)

If ID collisions exist, use Sinkhorn algorithm to reassign:

```bash
python src/sid_tokenizer/hidvae/sinkhorn_reassignment.py \
    --semantic_ids scripts/output/hidvae_tokenizer/beauty/test_run/semantic_ids/semantic_ids.parquet \
    --checkpoint scripts/output/hidvae_tokenizer/beauty/test_run/best_model.pt \
    --data_dir dataset/Amazon-Beauty/processed/beauty-hidvae-sentenceT5base/Beauty \
    --codebook_size 256 \
    --device cuda
```

**Outputs**:
- `semantic_ids_reassigned.parquet`: Collision-free IDs
- `semantic_ids_reassigned.npy`: Numpy array
- `reassignment_stats.json`: Before/after statistics

---

## Configuration Parameters

### Model Architecture
- `n_layers`: Number of RQ layers (default: 3)
- `n_embed`: Codebook size per layer (default: 256)
- `latent_dim`: Dimension of codebook vectors (default: 32)
- `content_dim`: Content embedding dimension (default: 768)
- `collab_dim`: Collaborative embedding dimension (default: 64)

### Loss Weights
- `lambda_content`: Content reconstruction weight (default: 1.0)
- `lambda_collab`: Collaborative reconstruction weight (default: 1.0)
- `delta_weight`: Classification loss weight (default: 1.0)
- `beta`: Commitment loss weight (default: 0.25)

### Quantization
- `use_ema`: Use EMA for codebook updates (recommended: True)
- `ema_decay`: EMA decay rate (default: 0.99)
- `quantize_mode`: Quantization method (choices: 'ste', 'rotation', 'gumbel_softmax')

### Training
- `epochs`: Number of epochs (default: 500)
- `batch_size`: Batch size (default: 256)
- `learning_rate`: Initial learning rate (default: 1e-3)
- `grad_clip`: Gradient clipping norm (default: 1.0)

### Scheduler
- `use_scheduler`: Enable LR scheduler
- `scheduler_type`: Type ('warmup_cosine' or 'exponential')
- `warmup_ratio`: Warmup ratio (default: 0.1)

### Early Stopping
- `early_stop_patience`: Patience epochs (default: 200)
- `early_stop_min_delta`: Minimum improvement (default: 1e-4)

---

## Expected Results

### Training Metrics

Monitor these metrics during training:

1. **Reconstruction Losses**:
   - Content reconstruction: Should converge to 0.1-0.3
   - Collab reconstruction: Should converge to 0.1-0.3

2. **Classification Accuracy**:
   - Layer 1 (L2 tags): 70-90%
   - Layer 2 (L3 tags): 60-80%
   - Layer 3 (L4 tags): 50-70%

3. **Codebook Usage**:
   - Usage rate per layer: >80% (avoid collapse)
   - Perplexity: Higher is better (>100 is good)

4. **Balance Loss**:
   - Should decrease over training
   - Low values indicate uniform usage

### Semantic ID Quality

After generation:

1. **Uniqueness**:
   - Target: >95% unique IDs
   - Collision rate: <5%

2. **Hierarchical Overlap**:
   - Layer 1 prefix: Lower overlap (more diversity)
   - Layer 2 prefix: Moderate overlap
   - Layer 3 full ID: Highest diversity

3. **Tag Alignment**:
   - Items with same tags should share prefixes
   - Hierarchical structure should be preserved

---

## Troubleshooting

### Problem: Codebook Collapse

**Symptoms**: Few codes used, low perplexity, high balance loss

**Solutions**:
1. Increase `gamma_weights` for balance loss
2. Lower EMA decay (try 0.95)
3. Increase batch size
4. Check k-means initialization

### Problem: High Reconstruction Loss

**Symptoms**: Content/collab loss doesn't converge

**Solutions**:
1. Check embedding quality (visualize distributions)
2. Adjust `lambda_content` and `lambda_collab` weights
3. Increase model capacity (hidden dimensions)
4. Lower learning rate

### Problem: Poor Classification Accuracy

**Symptoms**: Tag prediction accuracy <50%

**Solutions**:
1. Increase `delta_weight` for classification loss
2. Check tag data quality
3. Add dropout to classifiers
4. Increase classifier hidden dimensions

### Problem: High ID Collision Rate

**Symptoms**: >10% collision rate

**Solutions**:
1. Increase codebook size (`n_embed`)
2. Add more RQ layers
3. Increase `latent_dim`
4. Apply Sinkhorn reassignment

---

## Advanced Tips

### 1. Hyperparameter Tuning

**Priority order**:
1. `delta_weight`: Start with 1.0, increase if classification is poor
2. `lambda_collab`: Adjust if collaborative info is weak
3. `beta`: Standard value (0.25) usually works well
4. Learning rate: 1e-3 is a good starting point

### 2. Monitoring Training

Watch for:
- Steadily decreasing total loss
- Stable codebook usage (>80%)
- Increasing classification accuracy
- No sudden spikes (indicates instability)

### 3. Model Variants

**For larger datasets**:
- Increase `n_embed` to 512 or 1024
- Add more layers (n_layers=4 or 5)
- Increase model capacity

**For faster training**:
- Reduce batch size
- Use 'ste' quantization mode
- Disable scheduler

**For better quality**:
- Use 'rotation' quantization mode
- Increase training epochs
- Fine-tune loss weights

---

## Citation

If you use HID-VAE in your research, please cite:

```
@article{hidvae2025,
  title={HID-VAE: Hierarchical ID VAE with Tag-Guided Learning for Semantic ID Generation},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## References

1. **TIGER**: Rajput et al. "Recommender Systems with Generative Retrieval" (NeurIPS 2023)
2. **RQ-VAE**: Lee et al. "Autoregressive Image Generation using Residual Quantization" (CVPR 2022)
3. **Sinkhorn Algorithm**: Cuturi "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (NeurIPS 2013)

---

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

