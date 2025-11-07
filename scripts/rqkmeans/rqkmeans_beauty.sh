#!/bin/bash
# RQ-KMeans training script (more stable than RQ-VAE)
# No codebook collapse issues

cd ../../src/sid_tokenizer

python train_tokenizer.py \
    --data_path ../../dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty \
    --embedding_file item_emb_all.parquet \
    --output_dir ../../scripts/output/kmeans_tokenizer/beauty/3-256-32 \
    --mode kmeans \
    --n_layers 3 \
    --n_clusters 256 \
    --device cuda \
    --log_level INFO

echo "RQ-KMeans training completed!"

