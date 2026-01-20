#!/bin/bash
# Process CDs dataset with 5-core filtering
# Only generate embeddings for filtered items (min_interactions >= 5)

cd ../../data

python process_amazon.py \
    --dataset CDs \
    --review_path ../dataset/Amazon-CDs/reviews_CDs_and_Vinyl.json.gz \
    --meta_path ../dataset/Amazon-CDs/meta_CDs_and_Vinyl.json.gz \
    --output_dir ../dataset/Amazon-CDs/processed/cds-prism-sentenceT5base \
    --min_interactions 5 \
    --embed_mode prism \
    --embed_model sentence-t5 \
    --model_source modelscope \
    --device auto \
    --print_samples 10

echo ""
echo "=================================="
echo "Data processing completed!"


