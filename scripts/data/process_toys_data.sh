#!/bin/bash
# Process Toys dataset with 5-core filtering
# Only generate embeddings for filtered items (default behavior)
# Add --generate_all_embeddings flag if you need ALL items embeddings for tokenizer training

cd ../../data

python process_amazon.py \
    --dataset Toys \
    --review_path ../dataset/Amazon-Toys/reviews_Toys_and_Games.json.gz \
    --meta_path ../dataset/Amazon-Toys/meta_Toys_and_Games.json.gz \
    --output_dir ../dataset/Amazon-Toys/processed/toys-prism-sentenceT5base \
    --min_interactions 5 \
    --embed_mode prism \
    --embed_model sentence-t5 \
    --model_source modelscope \
    --device cuda:3 \
    --print_samples 10

echo ""
echo "=================================="
echo "Data processing completed!"


