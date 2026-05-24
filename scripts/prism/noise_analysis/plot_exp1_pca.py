#!/usr/bin/env python3
"""Plot 1: PCA Cumulative Explained Variance — information density comparison."""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── 1. Load data ──────────────────────────────────────────────────────────
DATA_DIR = "scripts/prism/noise_analysis"
with open(os.path.join(DATA_DIR, "exp1_pca_results.json")) as f:
    data = json.load(f)

text_cumvar = np.array(data["text"]["cumulative_variance"])
collab_cumvar = np.array(data["collab"]["cumulative_variance"])
text_90 = data["text"]["components_for_90pct"]
collab_90 = data["collab"]["components_for_90pct"]

# ── 2. Style setup ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 10,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
palette = sns.color_palette("deep")

fig, ax = plt.subplots(figsize=(5.5, 3.8))

# ── 3. Plot curves ────────────────────────────────────────────────────────
x_text = np.arange(1, len(text_cumvar) + 1)
x_collab = np.arange(1, len(collab_cumvar) + 1)

ax.plot(x_text, text_cumvar, color=palette[0], linewidth=1.8, label="Text (768D)")
ax.plot(x_collab, collab_cumvar, color=palette[3], linewidth=1.8, label="Collab (64D)")

# ── 4. 90% threshold line + annotations ───────────────────────────────────
ax.axhline(y=0.90, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
ax.annotate("90% variance", xy=(max(x_text) * 0.75, 0.905), fontsize=8,
            color="gray", ha="left", va="bottom")

# Marker + label for Text @ 90%
ax.plot(text_90, 0.90, "o", color=palette[0], markersize=5, zorder=5)
ax.annotate(f"Text: {text_90} dims\n({text_90/768*100:.0f}% of 768D)",
            xy=(text_90, 0.90), xytext=(text_90 + 30, 0.82),
            fontsize=8, color=palette[0], ha="left",
            arrowprops=dict(arrowstyle="->", color=palette[0], lw=0.8))

# Marker + label for Collab @ 90%
ax.plot(collab_90, 0.90, "o", color=palette[3], markersize=5, zorder=5)
ax.annotate(f"Collab: {collab_90} dims\n({collab_90/64*100:.0f}% of 64D)",
            xy=(collab_90, 0.90), xytext=(collab_90 + 8, 0.72),
            fontsize=8, color=palette[3], ha="left",
            arrowprops=dict(arrowstyle="->", color=palette[3], lw=0.8))

# ── 5. Axes & layout ──────────────────────────────────────────────────────
ax.set_xlim(0, 256)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Explained Variance Ratio")
ax.legend(frameon=False, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

plt.tight_layout()
out_path = os.path.join(DATA_DIR, "fig1_pca_variance.pdf")
fig.savefig(out_path)
print(f"Saved: {out_path}")
plt.close()
