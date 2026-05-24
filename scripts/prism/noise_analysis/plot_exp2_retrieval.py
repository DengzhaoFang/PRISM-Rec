#!/usr/bin/env python3
"""Plot 2: Bucket-wise Recall@10 — modality stability under popularity shift."""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── 1. Load ───────────────────────────────────────────────────────────────
DATA_DIR = "scripts/prism/noise_analysis"
with open(os.path.join(DATA_DIR, "exp2_retrieval_results.json")) as f:
    data = json.load(f)

buckets = data["buckets"]
# Filter out empty buckets
valid = [int(b) for b in buckets if data["text"]["bucket_recall"][b] > 0 or data["collab"]["bucket_recall"][b] > 0]
labels = [buckets[str(b)]["label"] for b in valid]
text_recall = [data["text"]["bucket_recall"][str(b)] for b in valid]
collab_recall = [data["collab"]["bucket_recall"][str(b)] for b in valid]

x = np.arange(len(valid))
K = data["config"]["K"]

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

# ── 3. Plot ───────────────────────────────────────────────────────────────
ax.plot(x, text_recall, "o-", color=pal[0], lw=1.8, ms=5, label="Text (cosine)")
ax.plot(x, collab_recall, "s--", color=pal[3], lw=1.8, ms=5, label="Collab (cosine)")

# ── 4. Annotations ────────────────────────────────────────────────────────
tc = data["text"]["cold_mean"]; th = data["text"]["hot_mean"]
cc = data["collab"]["cold_mean"]; ch = data["collab"]["hot_mean"]
ax.annotate(f"Text C/H={tc/th:.2f}", xy=(0.1, 0.92), xycoords="axes fraction",
            fontsize=7.5, color=pal[0])
ax.annotate(f"Collab C/H={cc/ch:.2f}", xy=(0.1, 0.87), xycoords="axes fraction",
            fontsize=7.5, color=pal[3])

# ── 5. Axes ───────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_xlabel("Popularity Bucket (cold → hot)")
ax.set_ylabel(f"Recall@{K}")
ax.legend(frameon=False, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

plt.tight_layout()
out_path = os.path.join(DATA_DIR, "fig2_bucket_recall.pdf")
fig.savefig(out_path)
print(f"Saved: {out_path}")
plt.close()
