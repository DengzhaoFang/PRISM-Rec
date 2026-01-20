#!/bin/bash

# Training script for LightGCN 
# This script trains LightGCN on the user-item interaction graph
# to obtain initial collaborative embeddings for items

# Change to project root
cd ../../src/sid_tokenizer/lightgcn


# Set dataset path
DATA_DIR="../../../dataset/Amazon-Sports/processed/sports-prism-sentenceT5base/Sports"

# Set output directory
OUTPUT_DIR=$DATA_DIR/lightgcn


# Model hyperparameters
EMBEDDING_DIM=64
N_LAYERS=3
DROPOUT=false
KEEP_PROB=0.6

# Training hyperparameters
N_EPOCHS=500
BATCH_SIZE=2048
LR=0.001
REG_WEIGHT=0.0001
EARLY_STOP_PATIENCE=20

# Evaluation settings
EVAL_EVERY=2
EVAL_BATCH_SIZE=1024
EARLY_STOP_METRIC="Recall@20"
K_VALUES="5 10 20"

# Save settings
SAVE_EVERY=10

# Device
DEVICE="cuda"
GPU_ID=2  

# Additional flags
USE_VAL=""  # Add --use_val to include validation set in training (WARNING: causes data leakage!)

echo "=================================="
echo "Training LightGCN "
echo "=================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Embedding dimension: $EMBEDDING_DIM"
echo "Number of layers: $N_LAYERS"
echo "Number of epochs: $N_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LR"
echo "=================================="

# Run training
python train.py \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --exp_name "$EXP_NAME" \
    --embedding_dim $EMBEDDING_DIM \
    --n_layers $N_LAYERS \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --reg_weight $REG_WEIGHT \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --eval_every $EVAL_EVERY \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --early_stop_metric $EARLY_STOP_METRIC \
    --k_values $K_VALUES \
    --save_every $SAVE_EVERY \
    --device $DEVICE \
    ${GPU_ID:+--gpu_id $GPU_ID} \
    $USE_VAL

echo "=================================="
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================="
