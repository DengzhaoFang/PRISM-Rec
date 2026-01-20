

cd ../../src/sid_tokenizer/LETTER

python train_letter.py \
    --data_path ../../../dataset/Amazon-Toys/processed/toys-prism-sentenceT5base/Toys \
    --embedding_file item_emb.parquet \
    --cf_emb lightgcn/item_embeddings_collab.npy \
    --output_dir ../../../scripts/output/letter_tokenizer/toys/3-256-32 \
    --epochs 500 \
    --batch_size 512 \
    --lr 1e-3 \
    --num_emb_list 256 256 256 \
    --e_dim 32 \
    --layers 512 256 128 64 \
    --alpha 0.02 \
    --beta 1e-4 \
    --mu 0.25 \
    --n_clusters 25 \
    --cluster_every 20 \
    --kmeans_init \
    --kmeans_iters 100 \
    --sk_epsilons 0.05 0.05 0.05 \
    --sk_iters 100 \
    --quant_loss_weight 1.0 \
    --early_stop_patience 300 \
    --early_stop_min_delta 1e-5 \
    --save_every 100 \
    --grad_clip 1.0 \
    --device cuda

echo ""
echo "✓ Training completed!"
echo ""

