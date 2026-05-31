"""Compute teacher prototypes for the Amazon Beauty dataset."""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'src/sid_tokenizer/prism'))

from recommendation_teacher import RecommendationTeacher

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty')

# Load item embeddings
print("Loading item embeddings...")
item_df = pd.read_parquet(os.path.join(DATA_DIR, 'item_emb.parquet'))
content_emb = np.stack([np.array(emb, dtype=np.float32) for emb in item_df['embedding']])
item_ids = item_df['ItemID'].values

print("Loading collab embeddings...")
collab_emb_all = np.load(os.path.join(DATA_DIR, 'lightgcn/item_embeddings_collab.npy'))
collab_emb = np.stack([collab_emb_all[iid].astype(np.float32) for iid in item_ids])

item_id_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}
num_items = len(item_ids)
print(f"  Items: {num_items}, content_dim={content_emb.shape[1]}, collab_dim={collab_emb.shape[1]}")

# Compute teacher
print("\nComputing teacher prototypes...")
teacher_obj = RecommendationTeacher(
    data_dir=DATA_DIR,
    content_embeddings=content_emb,
    collab_embeddings=collab_emb,
    item_id_to_idx=item_id_to_idx,
    recency_gamma=0.5,
    backoff_tau=5.0,
    teacher_dim=256,
)
teacher_matrix = teacher_obj.build()
teacher_obj.save(DATA_DIR)

print(f"\nTeacher matrix: {teacher_matrix.shape}")
print(f"Items with signal: {(teacher_obj.n_ctx > 0).sum()} / {num_items}")
print(f"Alpha: mean={teacher_obj.alpha.mean():.3f}, median={float(np.median(teacher_obj.alpha)):.3f}")
print("Done!")
