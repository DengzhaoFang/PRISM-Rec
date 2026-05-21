#!/usr/bin/env python3
"""
Plot Experiment 2: Popularity-Bucketed Performance (Reliability Noise)
=======================================================================

Reads exp2_results.json and produces Figure 2 as specified in exp2.md.
Includes the IDE-fused condition (Text→128 + Collab→128 → Concat 256D).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "scripts/prism/noise_analysis"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "exp2_results.json")


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def plot_figure2(data):
    text_color = "#E07050"
    collab_color = "#4070C0"
    fused_color = "#2CA02C"  # green

    bucket_keys = sorted([int(k) for k in data["text"]["bucket_recall"].keys()])
    n_buckets = len(bucket_keys)

    x_labels = []
    text_recalls = []
    collab_recalls = []
    fused_recalls = []
    n_samples = []

    for b in bucket_keys:
        bk = str(b)
        info = data["buckets"][bk]
        x_labels.append(f"B{b}\n[{info['pop_min']}-{info['pop_max']}]")
        text_recalls.append(data["text"]["bucket_recall"][bk])
        collab_recalls.append(data["collab"]["bucket_recall"][bk])
        fused_recalls.append(data["fused"]["bucket_recall"][bk])
        n_samples.append(info["n_test_samples"])

    x = np.arange(n_buckets)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Fused curve (on top, thicker)
    ax.plot(x, fused_recalls, "D-", color=fused_color, linewidth=2.8, markersize=9,
            markeredgewidth=0, label="Fused (IDE: 768→128 + 64→128 → 256D)", zorder=10)

    ax.plot(x, text_recalls, "o-", color=text_color, linewidth=2.2, markersize=8,
            markeredgewidth=0, label="Text-only (768D)", zorder=5)
    ax.plot(x, collab_recalls, "s--", color=collab_color, linewidth=2.2, markersize=8,
            markeredgewidth=0, label="Collab-only (64D)", zorder=5)

    # Error bands for all 3
    for i in range(n_buckets):
        for arr, color in [(text_recalls, text_color), (collab_recalls, collab_color),
                           (fused_recalls, fused_color)]:
            p = arr[i]
            se = np.sqrt(p * (1 - p) / max(n_samples[i], 1))
            ax.plot([x[i], x[i]], [p - se, p + se], color=color, linewidth=1.2, alpha=0.35)

    # Sample count annotation
    for i in range(n_buckets):
        ax.text(x[i], -0.06, f"n={n_samples[i]}", transform=ax.get_xaxis_transform(),
                fontsize=7, ha="center", color="gray", fontstyle="italic")

    # Long-tail vs Popular shading
    ax.axvspan(-0.5, n_buckets // 2 - 0.5, alpha=0.03, color="orange")
    ax.axvspan(n_buckets // 2 - 0.5, n_buckets - 0.5, alpha=0.03, color="blue")
    ax.text((n_buckets // 4) - 0.5, ax.get_ylim()[1] * 0.95,
            "Long-tail", fontsize=9, ha="center", color="orange", alpha=0.6)
    ax.text((3 * n_buckets // 4) - 0.5, ax.get_ylim()[1] * 0.95,
            "Popular", fontsize=9, ha="center", color="blue", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7.5)
    ax.set_xlabel("Item Popularity Bucket (Long-tail → Popular)\n(popularity range + test sample count below)",
                  fontsize=11, fontweight="bold")
    ax.set_ylabel("Recall@10", fontsize=12, fontweight="bold")
    ax.set_title("Figure 2: Reliability Noise + IDE Fusion — Performance by Item Popularity\n"
                 "IDE: Text 768→128 + Collab 64→128 → Concat 256D. Error bars: ±1 SE.",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.set_ylim(bottom=0)

    # Overall recall box
    text_overall = data["text"]["recall@10"]
    collab_overall = data["collab"]["recall@10"]
    fused_overall = data["fused"]["recall@10"]
    ax.text(0.98, 0.22,
            f"Overall: Text={text_overall:.4f}  Collab={collab_overall:.4f}  Fused={fused_overall:.4f}",
            transform=ax.transAxes, fontsize=8, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()

    for fmt in ["pdf", "png", "svg"]:
        path = os.path.join(OUTPUT_DIR, f"exp2_figure2.{fmt}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close()
    print("Done.")


def print_summary(data):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 SUMMARY")
    print("=" * 70)
    print(f"  Text-only  Recall@10: {data['text']['recall@10']:.4f}")
    print(f"  Collab-only Recall@10: {data['collab']['recall@10']:.4f}")
    print(f"  Fused (IDE) Recall@10: {data['fused']['recall@10']:.4f}")

    bucket_keys = sorted([int(k) for k in data["text"]["bucket_recall"].keys()])
    print(f"\n  {'Bucket':<30s} {'n_test':>7s} {'Text':>10s} {'Collab':>10s} "
          f"{'Fused':>10s} {'Fused-Text':>10s} {'Fused-Collab':>12s}")
    print("  " + "-" * 95)
    for b in bucket_keys:
        bk = str(b)
        tr = data["text"]["bucket_recall"][bk]
        cr = data["collab"]["bucket_recall"][bk]
        fr = data["fused"]["bucket_recall"][bk]
        info = data["buckets"][bk]
        n_test = info["n_test_samples"]
        label = f"B{b} [{info['pop_min']}-{info['pop_max']}]"
        print(f"  {label:<30s} {n_test:>7d} {tr:>10.4f} {cr:>10.4f} "
              f"{fr:>10.4f} {fr-tr:>+10.4f} {fr-cr:>+12.4f}")


if __name__ == "__main__":
    data = load_results()
    print_summary(data)
    plot_figure2(data)
