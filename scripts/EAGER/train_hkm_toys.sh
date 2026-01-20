#!/bin/bash
# EAGER Dual-Path Hierarchical K-Means Training Script
# Generates semantic IDs for both behavior and semantic paths

cd ../../src/sid_tokenizer/EAGER


python train_tokenizer.py \
    --data_path ../../../dataset/Amazon-Toys/processed/toys-prism-sentenceT5base/Toys \
    --embedding_file item_emb.parquet \
    --cf_embedding_file lightgcn/item_embeddings_collab.npy \
    --output_dir ../../../scripts/output/eager_tokenizer/toys/hkm_k8_d2 \
    --hkm_k 256 \
    --hkm_depth 2 \
    --hkm_n_init 10 \
    --hkm_max_iter 300 \
    --random_state 42 \
    --device cuda \
    --log_level INFO

echo ""
echo "=========================================="
echo "✓ EAGER HKM Training Completed!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - semantic_id_mappings_semantic.json"
echo "  - semantic_id_mappings_behavior.json"
echo "  - collision_stats.json"
echo "  - hkm_semantic.pt"
echo "  - hkm_behavior.pt"
echo ""
echo "Check collision rates in the log file and collision_stats.json"
echo ""



