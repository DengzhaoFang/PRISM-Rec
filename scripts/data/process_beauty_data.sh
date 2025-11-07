#!/bin/bash
# Process Beauty dataset with ALL items for tokenizer training
# Two-stage approach:
#   1. Generate embeddings for ALL items (for tokenizer)
#   2. Apply 5-core filter only for interaction data (for recommender)

cd ../../data

python process.py \
    --dataset Beauty \
    --review_path ../dataset/Amazon-Beauty/reviews_Beauty.json.gz \
    --meta_path ../dataset/Amazon-Beauty/meta_Beauty.json.gz \
    --output_dir ../dataset/Amazon-Beauty/processed/beauty-hidvae-sentenceT5base \
    --min_interactions 5 \
    --embed_mode hid-vae \
    --embed_model sentence-t5 \
    --model_source modelscope \
    --device auto \
    --print_samples 10

echo ""
echo "=================================="
echo "Data processing completed!"


