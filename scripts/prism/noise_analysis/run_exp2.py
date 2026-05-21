#!/usr/bin/env python3
"""
Experiment 2: Popularity-Bucketed Sequence Prediction
======================================================

Validates the "reliability noise" hypothesis:
  - Collab embeddings: excellent for popular items, fail on long-tail (steep curve)
  - Text embeddings:  stable across popularity, robust on long-tail (flat curve)

Method:
  SASRec trained with text-only or collab-only embeddings
  → Recall@10 per popularity bucket
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
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from copy import deepcopy

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = "dataset/Amazon-Beauty/processed/beauty-tiger-sentenceT5base/Beauty"
OUTPUT_DIR = "scripts/prism/noise_analysis"
OUTPUT_DATA = os.path.join(OUTPUT_DIR, "exp2_results.json")

NUM_BUCKETS = 8
SEED = 42
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 20
NUM_WORKERS = 0

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

    # Text embeddings (768D)
    text_embs = np.stack([np.array(e, dtype=np.float32) for e in emb_df["embedding"]])

    # Collab embeddings (64D)
    collab_path = os.path.join(DATA_DIR, "lightgcn/item_embeddings_collab.npy")
    collab_all = np.load(collab_path).astype(np.float32)
    collab_embs = np.stack([collab_all[iid] for iid in item_ids])

    # Item popularity: compute TRUE interaction counts from train.parquet
    # (the 'interaction_count' field in item_emb.parquet is pre-5core-filter, unreliable!)
    print("Computing true popularity from train.parquet...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    pop = np.zeros(n_items, dtype=np.float32)
    for _, row in train_df.iterrows():
        for h in row["history"]:
            idx = id_to_idx.get(int(h), -1)
            if idx >= 0:
                pop[idx] += 1
        idx = id_to_idx.get(int(row["target"]), -1)
        if idx >= 0:
            pop[idx] += 1
    print(f"  True popularity: min={pop.min():.0f}, median={np.median(pop):.0f}, "
          f"mean={pop.mean():.1f}, zero={int((pop==0).sum())}")

    test_df = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))
    train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
    valid_df = pd.read_parquet(os.path.join(DATA_DIR, "valid.parquet"))

    def build_sequences(df):
        seqs, targets = [], []
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
                seqs.append(padded)
                targets.append(target)
        return np.array(seqs, dtype=np.int64), np.array(targets, dtype=np.int64)

    train_seqs, train_targets = build_sequences(train_df)
    valid_seqs, valid_targets = build_sequences(valid_df)
    test_seqs, test_targets = build_sequences(test_df)

    # Map test target indices back to item IDs for bucket assignment
    test_target_iids = []
    for t in test_targets:
        if t < len(item_ids):
            test_target_iids.append(int(item_ids[t]))
        else:
            test_target_iids.append(None)

    test_target_iids = np.array(test_target_iids)

    print(f"  Items: {n_items}, Train: {len(train_seqs)}, Valid: {len(valid_seqs)}, Test: {len(test_seqs)}")
    return (n_items, text_embs, collab_embs, pop, item_ids, id_to_idx,
            train_seqs, train_targets, valid_seqs, valid_targets,
            test_seqs, test_targets, test_target_iids)


# ── Popularity Bucketing (Test-Sample-Count-Based) ──────────────────────
def build_popularity_buckets(pop, test_target_iids, n_buckets=NUM_BUCKETS):
    """Build buckets with roughly EQUAL test samples per bucket.

    Sorts all test targets by their item's true popularity, then splits
    into n_buckets equal-sized chunks. This ensures each bucket's Recall
    is computed on a stable (similar-sized) sample set.
    """
    # Get true popularity for each test target
    test_pops = np.array([pop[iid] if iid is not None and iid < len(pop) else -1
                           for iid in test_target_iids])

    # Filter out negative (invalid) targets
    valid_mask = test_pops >= 0
    valid_pops = test_pops[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Sort test samples by item popularity (ascending: long-tail first)
    sort_order = np.argsort(valid_pops)
    sorted_pops = valid_pops[sort_order]
    sorted_indices = valid_indices[sort_order]

    # Split into n_buckets equal-sized chunks
    chunk_size = len(sorted_pops) // n_buckets
    bucket_boundaries = []      # popularity range for each bucket
    bucket_test_indices = []    # which test sample indices belong to each bucket
    bucket_stats = []           # per-bucket metadata

    for i in range(n_buckets):
        start = i * chunk_size
        if i == n_buckets - 1:
            end = len(sorted_pops)
        else:
            end = (i + 1) * chunk_size

        chunk_pops = sorted_pops[start:end]
        chunk_idx = sorted_indices[start:end]

        lo = int(chunk_pops.min())
        hi = int(chunk_pops.max())
        n_test = len(chunk_idx)

        # Count items in this popularity range
        n_items_in_range = int(((pop >= lo) & (pop <= hi)).sum())

        bucket_boundaries.append((lo, hi))
        bucket_test_indices.append(chunk_idx)
        bucket_stats.append({
            "label": f"[{lo}-{hi}]",
            "pop_min": lo,
            "pop_max": hi,
            "n_test_samples": n_test,
            "n_items": n_items_in_range,
        })

    # Build a per-sample bucket assignment for evaluation
    test_sample_to_bucket = -np.ones(len(test_target_iids), dtype=int)
    for b, indices in enumerate(bucket_test_indices):
        test_sample_to_bucket[indices] = b

    return bucket_stats, test_sample_to_bucket


# ── Fusion Module (IDE: Information Density Equalization) ───────────────
FUSION_DIM = 128


class FusedSASRec(nn.Module):
    """SASRec with learnable IDE fusion of text and collab embeddings.

    Text (768D) --Linear+LN--> 128D  \
                                       |--concat--> 256D --> SASRec
    Collab (64D) --Linear+LN--> 128D /

    Source embeddings stored as buffers (no grad on raw data).
    Projection weights learn end-to-end; fused embeddings recomputed each forward
    (fresh computation graph, avoids backward-second-time error).
    """

    def __init__(self, n_items, text_dim=768, collab_dim=64, fusion_dim=FUSION_DIM,
                 d_model=128, num_heads=2, num_blocks=2, dropout=0.2, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = fusion_dim * 2  # 256D

        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.collab_proj = nn.Linear(collab_dim, fusion_dim)
        self.text_ln = nn.LayerNorm(fusion_dim)
        self.collab_ln = nn.LayerNorm(fusion_dim)

        self.pos_emb = nn.Embedding(max_len, d_model)
        self.item_proj = nn.Linear(self.emb_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.out_ln = nn.LayerNorm(d_model)

    def set_source_embeddings(self, text_embs: torch.Tensor, collab_embs: torch.Tensor):
        """Store raw source embeddings (fixed, no grad). Projections re-applied each forward."""
        self.register_buffer("_text_src", text_embs.detach().clone())
        self.register_buffer("_collab_src", collab_embs.detach().clone())

    def _get_fused(self):
        """Recompute fused item embeddings (fresh graph each call)."""
        h_t = self.text_ln(self.text_proj(self._text_src))
        h_c = self.collab_ln(self.collab_proj(self._collab_src))
        return torch.cat([h_t, h_c], dim=-1)

    def forward(self, seqs, item_embs=None):
        B, L = seqs.shape
        fused = self._get_fused()
        item_feats = fused[seqs]
        h = self.item_proj(item_feats)
        pos = torch.arange(L, device=seqs.device).unsqueeze(0)
        h = h + self.pos_emb(pos)
        h = self.dropout(self.ln(h))
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=seqs.device)
        h = self.transformer(h, mask=mask)
        h = self.out_ln(h)
        user_repr = h[:, -1, :]
        item_repr = self.item_proj(fused)
        scores = torch.matmul(user_repr, item_repr.T)
        return scores


# ── SASRec ──────────────────────────────────────────────────────────────
class SASRec(nn.Module):
    def __init__(self, n_items, emb_dim, d_model=128, num_heads=2, num_blocks=2, dropout=0.2, max_len=MAX_SEQ_LEN):
        super().__init__()
        self.n_items = n_items
        self.emb_dim = emb_dim

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

    def set_item_embeddings(self, item_embs: torch.Tensor):
        self.register_buffer("item_embs", item_embs)

    def forward(self, seqs, item_embs=None):
        B, L = seqs.shape
        if item_embs is None:
            item_embs = self.item_embs

        item_feats = item_embs[seqs]
        h = self.item_proj(item_feats)

        pos = torch.arange(L, device=seqs.device).unsqueeze(0)
        h = h + self.pos_emb(pos)
        h = self.dropout(self.ln(h))

        mask = nn.Transformer.generate_square_subsequent_mask(L, device=seqs.device)
        h = self.transformer(h, mask=mask)
        h = self.out_ln(h)

        user_repr = h[:, -1, :]
        item_repr = self.item_proj(item_embs)
        scores = torch.matmul(user_repr, item_repr.T)
        return scores


# ── Dataset ─────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, seqs, targets):
        self.seqs = torch.from_numpy(seqs).long()
        self.targets = torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.targets[idx]


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_recall(scores, targets, k=10):
    batch_size = scores.size(0)
    hit = 0
    for i in range(batch_size):
        _, topk = torch.topk(scores[i], k=k)
        if targets[i] in topk:
            hit += 1
    return hit / batch_size


def compute_recall_per_bucket(scores, targets, target_iids, bucket_ids, k=10):
    """Compute Recall@k for each popularity bucket."""
    bucket_recalls = defaultdict(list)
    for i in range(len(targets)):
        iid = target_iids[i]
        if iid is None:
            continue
        bucket = int(bucket_ids[iid]) if iid < len(bucket_ids) else -1
        if bucket < 0:
            continue
        _, topk = torch.topk(scores[i], k=k)
        hit = 1.0 if targets[i] in topk else 0.0
        bucket_recalls[bucket].append(hit)
    return {b: np.mean(v) for b, v in bucket_recalls.items()}


def evaluate(model, dataloader, item_embs_tensor, test_sample_to_bucket):
    model.eval()
    all_bucket_hits = defaultdict(list)
    total_recall = 0.0
    n = 0
    with torch.no_grad():
        for seqs, targets in dataloader:
            seqs = seqs.to(DEVICE)
            targets_dev = targets.to(DEVICE)
            scores = model(seqs, item_embs_tensor)

            total_recall += compute_recall(scores, targets_dev, k=10) * seqs.size(0)
            n += seqs.size(0)

            start = n - seqs.size(0)
            for i in range(seqs.size(0)):
                idx = start + i
                if idx >= len(test_sample_to_bucket):
                    break
                bucket = int(test_sample_to_bucket[idx])
                if bucket < 0:
                    continue
                _, topk = torch.topk(scores[i], k=10)
                hit = 1.0 if targets_dev[i] in topk else 0.0
                all_bucket_hits[bucket].append(hit)

    bucket_recall = {int(b): np.mean(hits) for b, hits in all_bucket_hits.items()}
    return total_recall / n, bucket_recall


# ── Training ────────────────────────────────────────────────────────────
def train_and_eval(item_embs, n_items, train_seqs, train_targets, valid_seqs, valid_targets,
                   test_seqs, test_targets, test_sample_to_bucket, label="", verbose=True):
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

    best_valid_recall = 0.0
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

        # Validate (overall recall)
        valid_recall = 0.0
        model.eval()
        with torch.no_grad():
            for seqs, targets in valid_loader:
                seqs, targets = seqs.to(DEVICE), targets.to(DEVICE)
                scores = model(seqs, item_embs_tensor)
                valid_recall += compute_recall(scores, targets, k=10) * seqs.size(0)
        valid_recall /= len(valid_ds)

        if valid_recall > best_valid_recall + 1e-5:
            best_valid_recall = valid_recall
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch >= 10:
            break

    model.load_state_dict(best_state)
    test_recall, bucket_recall = evaluate(model, test_loader, item_embs_tensor,
                                          test_sample_to_bucket)

    if verbose:
        print(f"  [{label}] test Recall@10 = {test_recall:.4f}  (best valid = {best_valid_recall:.4f})")
        print(f"    Bucket recalls: ", end="")
        for b in sorted(bucket_recall.keys()):
            print(f"B{b}={bucket_recall[b]:.3f} ", end="")
        print()
    return test_recall, bucket_recall


def train_and_eval_fused(text_embs, collab_embs, n_items,
                         train_seqs, train_targets, valid_seqs, valid_targets,
                         test_seqs, test_targets, test_sample_to_bucket,
                         label="fused", verbose=True):
    model = FusedSASRec(n_items, text_dim=768, collab_dim=64, fusion_dim=FUSION_DIM,
                        d_model=D_MODEL, num_heads=NUM_HEADS, num_blocks=NUM_BLOCKS,
                        dropout=DROPOUT, max_len=MAX_SEQ_LEN)
    model.to(DEVICE)

    text_t = torch.from_numpy(text_embs).float().to(DEVICE)
    collab_t = torch.from_numpy(collab_embs).float().to(DEVICE)
    model.set_source_embeddings(text_t, collab_t)

    train_ds = SeqDataset(train_seqs, train_targets)
    valid_ds = SeqDataset(valid_seqs, valid_targets)
    test_ds = SeqDataset(test_seqs, test_targets)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_valid_recall = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            scores = model(seqs)
            loss = criterion(scores, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        valid_recall = 0.0
        model.eval()
        with torch.no_grad():
            for seqs, targets in valid_loader:
                seqs, targets = seqs.to(DEVICE), targets.to(DEVICE)
                scores = model(seqs)
                valid_recall += compute_recall(scores, targets, k=10) * seqs.size(0)
        valid_recall /= len(valid_ds)

        if valid_recall > best_valid_recall + 1e-5:
            best_valid_recall = valid_recall
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 5 and epoch >= 10:
            break

    model.load_state_dict(best_state)
    fused_embs = model._get_fused()
    test_recall, bucket_recall = evaluate(model, test_loader, fused_embs,
                                          test_sample_to_bucket)

    if verbose:
        print(f"  [{label}] test Recall@10 = {test_recall:.4f}  (best valid = {best_valid_recall:.4f})")
        print(f"    Bucket recalls: ", end="")
        for b in sorted(bucket_recall.keys()):
            print(f"B{b}={bucket_recall[b]:.3f} ", end="")
        print()
    return test_recall, bucket_recall


# ── Main ────────────────────────────────────────────────────────────────
def main():
    (n_items, text_embs, collab_embs, pop, item_ids, id_to_idx,
     train_seqs, train_targets, valid_seqs, valid_targets,
     test_seqs, test_targets, test_target_iids) = load_data()

    # Build buckets: roughly equal test samples per bucket
    bucket_stats, test_sample_to_bucket = build_popularity_buckets(
        pop, test_target_iids, NUM_BUCKETS
    )

    results = {
        "config": {
            "NUM_BUCKETS": NUM_BUCKETS,
            "BATCH_SIZE": BATCH_SIZE,
            "NUM_EPOCHS": NUM_EPOCHS,
            "LEARNING_RATE": LEARNING_RATE,
            "D_MODEL": D_MODEL,
            "NUM_HEADS": NUM_HEADS,
            "NUM_BLOCKS": NUM_BLOCKS,
            "DROPOUT": DROPOUT,
            "SEED": SEED,
            "bucketing_method": "test-sample-count-based",
        },
        "buckets": {str(i): s for i, s in enumerate(bucket_stats)},
        "text": {},
        "collab": {},
    }

    # ── Text-only ──
    print("\n" + "=" * 70)
    print("TEXT-ONLY SASRec")
    print("=" * 70)
    t0 = time.time()
    text_recall, text_bucket = train_and_eval(
        text_embs, n_items, train_seqs, train_targets, valid_seqs, valid_targets,
        test_seqs, test_targets, test_sample_to_bucket, label="text-only"
    )
    results["text"] = {
        "recall@10": round(text_recall, 6),
        "bucket_recall": {str(b): round(v, 6) for b, v in text_bucket.items()},
        "time_s": round(time.time() - t0, 1),
    }

    # ── Collab-only ──
    print("\n" + "=" * 70)
    print("COLLAB-ONLY SASRec")
    print("=" * 70)
    t0 = time.time()
    collab_recall, collab_bucket = train_and_eval(
        collab_embs, n_items, train_seqs, train_targets, valid_seqs, valid_targets,
        test_seqs, test_targets, test_sample_to_bucket, label="collab-only"
    )
    results["collab"] = {
        "recall@10": round(collab_recall, 6),
        "bucket_recall": {str(b): round(v, 6) for b, v in collab_bucket.items()},
        "time_s": round(time.time() - t0, 1),
    }

    # ── Fused (IDE) ──
    print("\n" + "=" * 70)
    print("FUSED SASRec (IDE: Text 768→128 + Collab 64→128 → Concat 256D)")
    print("=" * 70)
    t0 = time.time()
    fused_recall, fused_bucket = train_and_eval_fused(
        text_embs, collab_embs, n_items,
        train_seqs, train_targets, valid_seqs, valid_targets,
        test_seqs, test_targets, test_sample_to_bucket, label="fused"
    )
    results["fused"] = {
        "recall@10": round(fused_recall, 6),
        "bucket_recall": {str(b): round(v, 6) for b, v in fused_bucket.items()},
        "time_s": round(time.time() - t0, 1),
    }

    # ── Save ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_DATA, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DATA}")


if __name__ == "__main__":
    main()
