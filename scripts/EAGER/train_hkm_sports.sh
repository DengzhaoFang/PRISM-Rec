#!/bin/bash
# EAGER Dual-Path Hierarchical K-Means Training Script
# Generates semantic IDs for both behavior and semantic paths

cd ../../src/sid_tokenizer/EAGER

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python train_tokenizer.py \
    --data_path ../../../dataset/Amazon-Sports/processed/sports-prism-sentenceT5base/Sports \
    --embedding_file item_emb.parquet \
    --cf_embedding_file lightgcn/item_embeddings_collab.npy \
    --output_dir ../../../scripts/output/eager_tokenizer/sports/hkm_k8_d2 \
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



