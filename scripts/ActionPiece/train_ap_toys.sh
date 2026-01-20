#!/bin/bash

# ActionPiece Tokenizer Training Script
# 
# 1. OPQ (Optimized Product Quantization) for feature extraction
# 2. ActionPiece BPE-like algorithm for vocabulary construction


cd ../..

echo "=================================================="
echo "Training ActionPiece Tokenizer"
echo "=================================================="
echo ""

# ============================================================
# Configuration
# ============================================================
DATASET="toys"  # Options: beauty, sports, toys, cds

# Data paths (adjust based on dataset)
case $DATASET in
    "beauty")
        DATA_PATH="dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
        OUTPUT_DIR="scripts/output/actionpiece_tokenizer/beauty"
        ;;
    "sports")
        DATA_PATH="dataset/Amazon-Sports/processed/sports-tiger-sentenceT5base/Sports"
        OUTPUT_DIR="scripts/output/actionpiece_tokenizer/sports"
        ;;
    "toys")
        DATA_PATH="dataset/Amazon-Toys/processed/toys-tiger-sentenceT5base/Toys"
        OUTPUT_DIR="scripts/output/actionpiece_tokenizer/toys"
        ;;
    "cds")
        DATA_PATH="dataset/Amazon-CDs/processed/cds-tiger-sentenceT5base/CDs"
        OUTPUT_DIR="scripts/output/actionpiece_tokenizer/cds"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

# OPQ parameters (paper settings)
PQ_N_CODEBOOKS=4      # m=4 in paper
PQ_CODEBOOK_SIZE=256  # 256 codes per codebook
N_HASH_BUCKETS=128    # For collision handling (ensures unique semantic IDs)

# ActionPiece vocabulary size (paper setting)
VOCAB_SIZE=40000

# Faiss threads
N_THREADS=4

# Embedding file
EMBEDDING_FILE="item_emb.parquet"

# Random seed
SEED=42

# ============================================================
# Training
# ============================================================

echo "✅ Dataset: ${DATASET}"
echo "   Data path: ${DATA_PATH}"
echo "   Output directory: ${OUTPUT_DIR}"
echo ""
echo "✅ OPQ Parameters:"
echo "   Codebooks: ${PQ_N_CODEBOOKS}"
echo "   Codebook size: ${PQ_CODEBOOK_SIZE}"
echo "   Hash buckets: ${N_HASH_BUCKETS}"
echo ""
echo "✅ ActionPiece vocabulary size: ${VOCAB_SIZE}"
echo ""
echo "=================================================="

python -m src.sid_tokenizer.ActionPiece.train_tokenizer \
    --data_path ${DATA_PATH} \
    --embedding_file ${EMBEDDING_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --pq_n_codebooks ${PQ_N_CODEBOOKS} \
    --pq_codebook_size ${PQ_CODEBOOK_SIZE} \
    --n_hash_buckets ${N_HASH_BUCKETS} \
    --vocab_size ${VOCAB_SIZE} \
    --n_threads ${N_THREADS} \
    --seed ${SEED} \
    --log_level INFO

echo ""
echo "=================================================="
echo "✓ ActionPiece tokenizer training completed!"
echo "  Output files:"
echo "    - ${OUTPUT_DIR}/semantic_id_mappings.json (item -> semantic ID)"
echo "    - ${OUTPUT_DIR}/actionpiece.json (tokenizer)"
echo "    - ${OUTPUT_DIR}/item2feat.json (item features)"
echo "    - ${OUTPUT_DIR}/stats.json (statistics)"
echo "=================================================="
