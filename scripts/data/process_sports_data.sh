#!/bin/bash
# Process Sports dataset with 5-core filtering
# Only generate embeddings for filtered items (default behavior)
# Add --generate_all_embeddings flag if you need ALL items embeddings for tokenizer training

cd ../../data

python process_amazon.py \
    --dataset Sports \
    --review_path ../dataset/Amazon-Sports/reviews_Sports_and_Outdoors.json.gz \
    --meta_path ../dataset/Amazon-Sports/meta_Sports_and_Outdoors.json.gz \
    --output_dir ../dataset/Amazon-Sports/processed/sports-prism-sentenceT5base \
    --min_interactions 5 \
    --embed_mode prism \
    --embed_model sentence-t5 \
    --model_source modelscope \
    --device auto \
    --print_samples 10

echo ""
echo "=================================="
echo "Data processing completed!"


