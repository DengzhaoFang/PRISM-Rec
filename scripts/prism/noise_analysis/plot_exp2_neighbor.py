#!/usr/bin/env python3
"""Plot 2a: Cross-Modal Neighbor Agreement by Popularity."""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── 1. Load ───────────────────────────────────────────────────────────────
DATA_DIR = "scripts/prism/noise_analysis"
with open(os.path.join(DATA_DIR, "exp2_neighbor_agreement.json")) as f:
    data = json.load(f)

buckets = data["buckets"]
valid = [int(b) for b in buckets if buckets[b]["n_items"] > 0]
labels = [buckets[str(b)]["label"] for b in valid]
jaccard = [data["jaccard"][b] for b in valid]
tp = [data["text_precision"][b] for b in valid]
cp = [data["collab_precision"][b] for b in valid]

x = np.arange(len(valid))

# ── 2. Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
pal = sns.color_palette("deep")

fig, ax = plt.subplots(figsize=(5.8, 3.8))

# ── 3. Bar + line ─────────────────────────────────────────────────────────
w = 0.35
ax.bar(x - w/2, tp, w, color=pal[0], alpha=0.85, label="Text→Collab (text neighbors found in collab)")
ax.bar(x + w/2, cp, w, color=pal[3], alpha=0.85, label="Collab→Text (collab neighbors found in text)")

# ── 4. Jaccard overlay ────────────────────────────────────────────────────
ax2 = ax.twinx()
ax2.plot(x, jaccard, "D-", color="gray", lw=1.5, ms=5, label="Jaccard overlap", zorder=5)
ax2.set_ylabel("Jaccard Overlap", color="gray")
ax2.tick_params(axis="y", colors="gray")

# ── 5. Annotation ─────────────────────────────────────────────────────────
ratio = data["hot_cold_ratio"]
ax.annotate(f"Hot/Cold = {ratio:.1f}×", xy=(0.95, 0.92), xycoords="axes fraction",
            ha="right", fontsize=8.5, fontweight="bold", color="gray",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

# ── 6. Axes ───────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_xlabel("Item Popularity Bucket (cold → hot)")
ax.set_ylabel(f"Neighbor Precision@{data['config']['K']}")
ax.spines["top"].set_visible(False)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left", fontsize=8.5)

plt.tight_layout()
out_path = os.path.join(DATA_DIR, "fig2a_neighbor_agreement.pdf")
fig.savefig(out_path)
print(f"Saved: {out_path}")
plt.close()
