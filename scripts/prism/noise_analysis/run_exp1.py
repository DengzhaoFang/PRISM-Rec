#!/usr/bin/env python3
"""
Experiment 1: PCA Truncation Analysis for Information Density
=============================================================

Validates the "relevance noise" hypothesis:
  - Text embeddings (768D): full of redundancy, PCA compression to 64-128D doesn't hurt
  - Collab embeddings (64D):  information-dense,  PCA compression causes steep degradation

Method:
  PCA → SASRec sequential recommender → NDCG@10 evaluation
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from collections import defaultdict
from copy import deepcopy

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
OUTPUT_DIR = "scripts/prism/noise_analysis"
OUTPUT_DATA = os.path.join(OUTPUT_DIR, "exp1_results.json")

DIMS_TEXT = [8, 16, 32, 64, 128, 256, 512, 768]
DIMS_COLLAB = [8, 16, 32, 64]

SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 20
NUM_WORKERS = 0

# SASRec hyperparams
D_MODEL = 128
NUM_HEADS = 2
NUM_BLOCKS = 2
DROPOUT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Data Loading ────────────────────────────────────────────────────────
def load_data():
    print("Loading data...")
    emb_df = pd.read_parquet(os.path.join(DATA_DIR, "item_emb.parquet"))
    item_ids = emb_df["ItemID"].values
    id_to_idx = {int(iid): idx for idx, iid in enumerate(item_ids)}
    n_items = len(item_ids)

    # Content embeddings (768D)
    content_embs = np.stack([np.array(e, dtype=np.float32) for e in emb_df["embedding"]])

    # Collab embeddings (64D)
    collab_path = os.path.join(DATA_DIR, "lightgcn/item_embeddings_collab.npy")
    collab_all = np.load(collab_path).astype(np.float32)  # (n_items+1, 64)
    collab_embs = np.stack([collab_all[iid] for iid in item_ids])

    # Interactions
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    valid_df = pd.read_parquet(os.path.join(DATA_DIR, "valid.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

    def build_sequences(df):
        users, seqs, targets = [], [], []
        for _, row in df.iterrows():
            hist = row["history"]
            if len(hist) < 2:
                continue
            hist = hist[-MAX_SEQ_LEN:]
            for i in range(1, len(hist)):
                prefix = hist[:i]
                mapped = [id_to_idx.get(int(x), 0) for x in prefix]
                padded = [0] * (MAX_SEQ_LEN - len(mapped)) + mapped
                if len(padded) > MAX_SEQ_LEN:
                    padded = padded[-MAX_SEQ_LEN:]
                target = id_to_idx.get(int(hist[i]), 0)
                users.append(row["user"])
                seqs.append(padded)
                targets.append(target)
        return np.array(seqs, dtype=np.int64), np.array(targets, dtype=np.int64)

    train_seqs, train_targets = build_sequences(train_df)
    valid_seqs, valid_targets = build_sequences(valid_df)
    test_seqs, test_targets = build_sequences(test_df)

    print(f"  Items: {n_items}, Train seqs: {len(train_seqs)}, "
          f"Valid: {len(valid_seqs)}, Test: {len(test_seqs)}")
    return (item_ids, n_items, content_embs, collab_embs,
            train_seqs, train_targets, valid_seqs, valid_targets,
            test_seqs, test_targets)


# ── PCA ─────────────────────────────────────────────────────────────────
def apply_pca(embs, n_components):
    if n_components >= embs.shape[1]:
        return embs.copy()
    pca = PCA(n_components=n_components, random_state=SEED)
    reduced = pca.fit_transform(embs)
    return reduced.astype(np.float32)


# ── SASRec Model ────────────────────────────────────────────────────────
class SASRec(nn.Module):
    """Lightweight SASRec for comparing embedding quality."""

    def __init__(self, n_items, emb_dim, d_model=128, num_heads=2, num_blocks=2, dropout=0.2, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.max_len = max_len

        self.pos_emb = nn.Embedding(max_len, d_model)
        self.item_proj = nn.Linear(emb_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.out_ln = nn.LayerNorm(d_model)

        # Register pre-computed item embeddings (frozen, used for scoring)
        self.register_buffer("_item_embs_init", torch.zeros(1))

    def set_item_embeddings(self, item_embs: torch.Tensor):
        """Set item embeddings used for final scoring. Shape: (n_items, emb_dim)."""
        self.register_buffer("item_embs", item_embs)

    def forward(self, seqs, item_embs=None):
        B, L = seqs.shape
        if item_embs is None:
            item_embs = self.item_embs

        item_feats = item_embs[seqs]             # (B, L, emb_dim)
        h = self.item_proj(item_feats)           # (B, L, d_model)

        pos = torch.arange(L, device=seqs.device).unsqueeze(0)
        h = h + self.pos_emb(pos)
        h = self.dropout(self.ln(h))

        mask = nn.Transformer.generate_square_subsequent_mask(L, device=seqs.device)
        h = self.transformer(h, mask=mask)
        h = self.out_ln(h)                       # (B, L, d_model)

        # Use last position output as user representation
        user_repr = h[:, -1, :]                  # (B, d_model)

        # Score against ALL items using dot-product
        item_feats_all = item_embs               # (n_items, emb_dim)
        item_repr = self.item_proj(item_feats_all)  # (n_items, d_model)
        scores = torch.matmul(user_repr, item_repr.T)  # (B, n_items)
        return scores


# ── Training ────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, seqs, targets):
        self.seqs = torch.from_numpy(seqs).long()
        self.targets = torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]


def compute_ndcg(scores, targets, k=10):
    """Compute NDCG@k. scores: (B, n_items), targets: (B,)."""
    batch_size = scores.size(0)
    ndcg_sum = 0.0
    for i in range(batch_size):
        target = targets[i].item()
        rel = torch.zeros(scores.size(1), device=scores.device)
        rel[target] = 1.0

        _, topk_idx = torch.topk(scores[i], k=k)
        dcg = 0.0
        for j, idx in enumerate(topk_idx):
            if rel[idx] > 0:
                dcg += 1.0 / np.log2(j + 2)
        idcg = 1.0 / np.log2(2.0)  # single relevant item at rank 1
        ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0
    return ndcg_sum / batch_size


def evaluate(model, dataloader, item_embs_tensor):
    model.eval()
    total_ndcg = 0.0
    with torch.no_grad():
        for seqs, targets in dataloader:
            seqs, targets = seqs.to(DEVICE), targets.to(DEVICE)
            scores = model(seqs, item_embs_tensor)
            total_ndcg += compute_ndcg(scores, targets, k=10) * seqs.size(0)
    return total_ndcg / len(dataloader.dataset)


def train_and_eval(item_embs, train_seqs, train_targets, valid_seqs, valid_targets,
                   test_seqs, test_targets, n_items, label="", verbose=True):
    emb_dim = item_embs.shape[1]
    model = SASRec(n_items, emb_dim, d_model=D_MODEL, num_heads=NUM_HEADS,
                   num_blocks=NUM_BLOCKS, dropout=DROPOUT, max_len=MAX_SEQ_LEN)
    model.to(DEVICE)

    item_embs_tensor = torch.from_numpy(item_embs).float().to(DEVICE)
    model.set_item_embeddings(item_embs_tensor)

    train_ds = SeqDataset(train_seqs, train_targets)
    valid_ds = SeqDataset(valid_seqs, valid_targets)
    test_ds = SeqDataset(test_seqs, test_targets)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_valid_ndcg = 0.0
    best_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            scores = model(seqs, item_embs_tensor)
            loss = criterion(scores, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        valid_ndcg = evaluate(model, valid_loader, item_embs_tensor)

        if valid_ndcg > best_valid_ndcg + 1e-5:
            best_valid_ndcg = valid_ndcg
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch >= 10:
            break

    model.load_state_dict(best_state)
    test_ndcg = evaluate(model, test_loader, item_embs_tensor)

    if verbose:
        print(f"  [{label}] test NDCG@10 = {test_ndcg:.4f}  (best valid = {best_valid_ndcg:.4f})")
    return test_ndcg


# ── Main ────────────────────────────────────────────────────────────────
def main():
    (item_ids, n_items, content_embs, collab_embs,
     train_seqs, train_targets, valid_seqs, valid_targets,
     test_seqs, test_targets) = load_data()

    results = {
        "config": {
            "DIMS_TEXT": DIMS_TEXT,
            "DIMS_COLLAB": DIMS_COLLAB,
            "BATCH_SIZE": BATCH_SIZE,
            "NUM_EPOCHS": NUM_EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "D_MODEL": D_MODEL,
            "NUM_HEADS": NUM_HEADS,
            "NUM_BLOCKS": NUM_BLOCKS,
            "DROPOUT": DROPOUT,
            "SEED": SEED,
        },
        "text": [],
        "collab": [],
    }

    # ── Text Embeddings ──
    print("\n" + "=" * 70)
    print("TEXT EMBEDDINGS: PCA truncation analysis")
    print("=" * 70)
    for d in DIMS_TEXT:
        label = f"text_d={d}"
        print(f"\n>>> {label} <<<")
        t0 = time.time()
        reduced = apply_pca(content_embs, d)
        ndcg = train_and_eval(reduced, train_seqs, train_targets, valid_seqs, valid_targets,
                              test_seqs, test_targets, n_items, label=label)
        elapsed = time.time() - t0
        results["text"].append({"dim": d, "ndcg@10": round(ndcg, 6), "time_s": round(elapsed, 1)})
        print(f"  Time: {elapsed:.1f}s")

    # ── Collab Embeddings ──
    print("\n" + "=" * 70)
    print("COLLAB EMBEDDINGS: PCA truncation analysis")
    print("=" * 70)
    for d in DIMS_COLLAB:
        label = f"collab_d={d}"
        print(f"\n>>> {label} <<<")
        t0 = time.time()
        reduced = apply_pca(collab_embs, d)
        ndcg = train_and_eval(reduced, train_seqs, train_targets, valid_seqs, valid_targets,
                              test_seqs, test_targets, n_items, label=label)
        elapsed = time.time() - t0
        results["collab"].append({"dim": d, "ndcg@10": round(ndcg, 6), "time_s": round(elapsed, 1)})
        print(f"  Time: {elapsed:.1f}s")

    # ── Save Results ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_DATA, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DATA}")


if __name__ == "__main__":
    main()
