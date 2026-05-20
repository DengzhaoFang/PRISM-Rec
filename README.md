# PRISM: Purified Representation and Integrated Semantic Modeling

A novel generative sequential recommendation framework that addresses tokenization quality and information loss through purified semantic quantization and integrated semantic modeling.

## Quick Start

### 1. Data Preprocessing

Process raw Amazon review data with 5-core filtering and generate semantic embeddings:

```bash
cd data
python process_amazon.py \
    --dataset Beauty \
    --review_path ../dataset/Amazon-Beauty/reviews_Beauty.json.gz \
    --meta_path ../dataset/Amazon-Beauty/meta_Beauty.json.gz \
    --output_dir ../dataset/Amazon-Beauty/processed/beauty-prism-sentenceT5base \
    --min_interactions 5 \
    --embed_mode prism \
    --embed_model sentence-t5 \
    --device cuda:0
```

**Other datasets**: Replace `Beauty` with `CDs`, `Sports`, or `Toys` and adjust paths accordingly.

### 2. Training PRISM

#### (1) Train Tokenizer (Purified Semantic Quantizer)

```bash
cd src/sid_tokenizer/prism
python train_prism.py \
    --data_path ../../../dataset/Amazon-Beauty/processed/beauty-prism-sentenceT5base/Beauty \
    --output_dir ../../../scripts/output/prism_tokenizer/beauty \
    --n_layers 3 \
    --n_embed_per_layer "256,256,256" \
    --latent_dim 32 \
    --epochs 500 \
    --batch_size 512 \
    --learning_rate 1e-4 \
    --use_ema \
    --device cuda
```

Or use the provided script:
```bash
bash scripts/prism/train_prism_beauty.sh
```

#### (2) Train Recommender (Integrated Semantic Recommender)

```bash
cd <project_root>
python -m src.recommender.prism.train \
    --config beauty \
    --device cuda:0 \
    --model_type t5-tiny-2 \
    --use_multimodal_fusion \
    --use_codebook_prediction \
    --use_trie_constraints
```

Or use the provided script:
```bash
bash scripts/prism/train_recommender.sh
```

### 3. Running Baselines

Example: Training TIGER baseline

#### (1) Train TIGER Tokenizer
```bash
cd src/sid_tokenizer/rq-base
python train_tokenizer.py \
    --data_path ../../../dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty \
    --output_dir ../../../scripts/output/tiger_tokenizer/beauty \
    --mode tiger \
    --n_layers 3 \
    --n_embed 256 \
    --latent_dim 32 \
    --epochs 500 \
    --batch_size 512 \
    --use_ema \
    --device cuda
```

Or use script: `bash scripts/TIGER/train_tiger_beauty.sh`

#### (2) Train TIGER Recommender
```bash
cd <project_root>
python -m src.recommender.TIGER.train \
    --config beauty \
    --device cuda:0 \
    --model_type t5-tiny-2
```

Or use script: `bash scripts/TIGER/train_recommender.sh`

**Other baselines**: Replace `TIGER` with `EAGER`, `LETTER`, or `ActionPiece` in the paths above.

## Project Structure

```
├── data/                    # Data preprocessing scripts
├── dataset/                 # Raw and processed datasets
├── src/
│   ├── sid_tokenizer/      # Tokenizer implementations
│   │   ├── prism/          # PRISM tokenizer
│   │   ├── rq-base/        # Baseline tokenizers (TIGER, etc.)
│   │   └── ...
│   └── recommender/        # Recommender implementations
│       ├── prism/          # PRISM recommender
│       └── ...
└── scripts/                # Training scripts
    ├── prism/
    ├── TIGER/
    └── ...
```

