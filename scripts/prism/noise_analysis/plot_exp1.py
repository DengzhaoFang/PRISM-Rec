#!/usr/bin/env python3
"""
Plot Experiment 1: Information Density via PCA Truncation
=========================================================

Reads exp1_results.json and produces Figure 1 as specified in exp1.md.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

OUTPUT_DIR = "scripts/prism/noise_analysis"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "exp1_results.json")

# ── Load results ────────────────────────────────────────────────────────
def load_results():
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    return data


def plot_figure1(data):
    text_dims = [p["dim"] for p in data["text"]]
    text_ndcg = [p["ndcg@10"] for p in data["text"]]
    collab_dims = [p["dim"] for p in data["collab"]]
    collab_ndcg = [p["ndcg@10"] for p in data["collab"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Color scheme
    text_color = "#E07050"   # warm red-orange
    collab_color = "#4070C0" # blue

    # Text curve
    ax.plot(text_dims, text_ndcg, "o-", color=text_color, linewidth=2.2,
            markersize=8, markeredgewidth=0, label="Text Embedding (768D)",
            zorder=5)

    # Collab curve
    ax.plot(collab_dims, collab_ndcg, "s--", color=collab_color, linewidth=2.2,
            markersize=8, markeredgewidth=0, label="Collab Embedding (64D)",
            zorder=5)

    # Annotations
    # Mark "optimal" text compression zone
    text_768 = text_ndcg[-1]
    for i, (d, v) in enumerate(zip(text_dims, text_ndcg)):
        if v > text_768 * 1.01:  # >1% improvement over 768D
            ax.annotate(f"d={d}", (d, v), textcoords="offset points",
                        xytext=(0, 12), fontsize=8, ha="center", color=text_color, fontweight="bold")

    # Mark collab cliff
    collab_64 = collab_ndcg[-1]
    for i, (d, v) in enumerate(zip(collab_dims, collab_ndcg)):
        if v < collab_64 * 0.5:
            ax.annotate(f"d={d}", (d, v), textcoords="offset points",
                        xytext=(0, 12), fontsize=8, ha="center", color=collab_color, fontweight="bold")
            break

    ax.set_xscale("log")
    ax.set_xlabel("Retained PCA Dimension $d$ (log scale)", fontsize=12, fontweight="bold")
    ax.set_ylabel("NDCG@10", fontsize=12, fontweight="bold")
    ax.set_title("Figure 1: Information Density via PCA Truncation\n"
                 "Text embeddings tolerate extreme compression; Collab embeddings collapse sharply",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)

    # Custom log tick labels
    ax.set_xticks([8, 16, 32, 64, 128, 256, 512, 768])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()

    # Save
    for fmt in ["pdf", "png", "svg"]:
        path = os.path.join(OUTPUT_DIR, f"exp1_figure1.{fmt}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()
    print("Done.")


# ── Also dump a text summary ────────────────────────────────────────────
def print_summary(data):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 SUMMARY")
    print("=" * 70)
    print("\nText Embedding (768D):")
    print(f"  {'Dim':>6s}  {'NDCG@10':>10s}  {'Δ vs 768D':>10s}")
    baseline = data["text"][-1]["ndcg@10"]
    for p in data["text"]:
        delta = p["ndcg@10"] - baseline
        print(f"  {p['dim']:>6d}  {p['ndcg@10']:>10.6f}  {delta:>+10.6f}")

    print("\nCollab Embedding (64D):")
    print(f"  {'Dim':>6s}  {'NDCG@10':>10s}  {'Δ vs 64D':>10s}")
    baseline = data["collab"][-1]["ndcg@10"]
    for p in data["collab"]:
        delta = p["ndcg@10"] - baseline
        print(f"  {p['dim']:>6d}  {p['ndcg@10']:>10.6f}  {delta:>+10.6f}")


if __name__ == "__main__":
    data = load_results()
    print_summary(data)
    plot_figure1(data)
