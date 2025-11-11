# Optimized RQ-VAE Training with EMA
# - Uses EMA update for stable codebook (more robust than Gumbel-Softmax)
# - Early stopping to prevent collapse
# - Proper loss monitoring (codebook + commitment)
cd ../../src/sid_tokenizer/hidvae

python train_tokenizer.py \
    --data_path ../../../dataset/Amazon-Beauty/processed/beauty-hidvae-sentenceT5base/Beauty \
    --embedding_file item_emb.parquet \
    --output_dir ../../../scripts/output/hidvae_tokenizer/beauty/3-256-32-ema-only-5-core-items-666 \
    --mode tiger \
    --n_layers 3 \
    --n_embed 256 \
    --latent_dim 32 \
    --epochs 500 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --use_ema \
    --ema_decay 0.99 \
    --beta 0.25 \
    --early_stop_patience 200 \
    --early_stop_min_delta 1e-4 \
    --save_every 50 \
    --device cuda \
    --log_level INFO \
    --use_scheduler \
    --scheduler_type warmup_cosine \
    --warmup_ratio 0.1 \
    --grad_clip 1.0 



    
echo "✓ Training completed! Check codebook usage and duplicate rates."






