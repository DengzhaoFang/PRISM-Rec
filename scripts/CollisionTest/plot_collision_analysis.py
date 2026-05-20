# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# plot_collision_analysis.py

# Clean, publication-ready dual-axis visualization for Semantic ID Collision Rate.
# Optimized for Top-Tier Academic Conferences (ICLR / KDD / SIGIR).
# Features: Zero-padding, large readable fonts, borderless legends.
# """

# import os
# import sys
# import json
# import argparse
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.ticker as mticker
# from matplotlib.gridspec import GridSpec

# # ---------------------------------------------------------------------------
# # 0. Aesthetics — Top-Tier Academic Style (Elegant, Minimalist, High-Contrast)
# # ---------------------------------------------------------------------------

# mpl.rcParams.update({
#     "font.family":            "serif",
#     "font.serif":             ["Times New Roman", "Times", "DejaVu Serif"],
#     "font.size":              15,      # Base font size increased for better readability
#     "axes.titlesize":         16,
#     "axes.labelsize":         16,      # Keep axis labels at 16
#     "xtick.labelsize":        16,      # Increased tick labels
#     "ytick.labelsize":        16,      # Increased tick labels
#     "legend.fontsize":        16,      # Larger legend text
#     "axes.linewidth":         1.2,     # Thicker axes for professional look
#     "axes.edgecolor":         "#2C3E50",  # Darker, more professional edge color
#     "xtick.color":            "#2C3E50",
#     "ytick.color":            "#2C3E50",
#     "xtick.major.width":      1.2,
#     "ytick.major.width":      1.2,
#     "xtick.major.size":       6,
#     "ytick.major.size":       6,
#     "xtick.direction":        "in",    # Inward ticks (classic academic style)
#     "ytick.direction":        "in",
#     "figure.dpi":             150,
#     "savefig.dpi":            300,
#     "savefig.bbox":           "tight",
#     "savefig.pad_inches":     0.05,    # Minimal padding for professional appearance
# })

# # Refined Academic Color Palette
# COLOR_BAR       = "#7B9EBD"  # Elegant Slate Blue
# COLOR_BAR_EDGE  = "#4A6D8A"  # Darker edge for crisp rendering
# COLOR_LINE      = "#B22222"  # Firebrick Red (Strong contrast, professional)
# COLOR_TEXT      = "#2C3E50"  # Dark Charcoal for annotations
# LAYER_COLS      = ["#4C72B0", "#DD8452", "#B22222"] # Muted Blue, Muted Orange, Firebrick


# # ---------------------------------------------------------------------------
# # 1. Data loading
# # ---------------------------------------------------------------------------

# def load_stats(stats_file: str) -> dict:
#     with open(stats_file, "r", encoding="utf-8") as f:
#         return json.load(f)

# def extract_arrays(stats: dict):
#     buckets     = sorted(stats["buckets"], key=lambda b: b["bucket_id"])
#     ids         = np.array([b["bucket_id"]                for b in buckets])
#     n_items     = np.array([b["total_items"]               for b in buckets])
#     cr3         = np.array([b["collision_rate_3layer"]     for b in buckets])
#     layer_rates = np.array([b["layer_collision_rates"]     for b in buckets])
#     return ids, n_items, cr3, layer_rates

# # ---------------------------------------------------------------------------
# # 2. Main dual-axis figure
# # ---------------------------------------------------------------------------

# def plot_main_figure(stats: dict, output_path: str):
#     ids, n_items, cr3, layer_rates = extract_arrays(stats)

#     n  = len(ids)
#     x  = np.arange(n)
#     x_labels = [f"D{i+1}" for i in range(n)] 

#     # ── Figure & layout ──────────────────────────────────────────────────────
#     # layout="constrained" smartly handles spacing to prevent overlaps
#     fig = plt.figure(figsize=(13.5, 4.5), layout="constrained") 
#     gs  = GridSpec(1, 2, figure=fig, width_ratios=[2.0, 1.2], wspace=0.14)  # Increased right subplot width for better spacing
    
#     ax_main = fig.add_subplot(gs[0])
#     ax_side = fig.add_subplot(gs[1])

#     # =========================================================================
#     # PANEL A — Bar (item count) + Line (collision rate)
#     # =========================================================================

#     bar_kw = dict(width=0.68, align="center", zorder=2, color=COLOR_BAR, edgecolor=COLOR_BAR_EDGE, linewidth=1.3)
#     bars = ax_main.bar(x, n_items, **bar_kw)

#     # Count labels on top of bars
#     y_max = n_items.max() * 1.32 # Extra room for the legend and labels
#     for bar, val in zip(bars, n_items):
#         ax_main.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height() + y_max * 0.015,
#             f"{val:,}",
#             ha="center", va="bottom",
#             fontsize=12, color=COLOR_TEXT, fontweight="600"  # Keep original size for bar labels
#         )

#     ax_main.set_xlim(-0.6, n - 0.4)
#     ax_main.set_ylim(0, y_max)
#     ax_main.set_xticks(x)
#     ax_main.set_xticklabels(x_labels)
#     ax_main.set_xlabel("Semantic Density Decile", fontweight="bold")
#     ax_main.set_ylabel("Number of Items", color=COLOR_BAR_EDGE, fontweight="bold")
#     ax_main.tick_params(axis="y", labelcolor=COLOR_BAR_EDGE)
    
#     # Clean up grid and spines
#     ax_main.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35, color='#CCCCCC')
#     ax_main.grid(axis='x', visible=False)
#     ax_main.spines["top"].set_visible(False)
#     ax_main.spines["right"].set_visible(False)  # Remove right spine for cleaner look

#     # ── Twin axis: collision rate ─────────────────────────────────────────────
#     ax_cr = ax_main.twinx()

#     # Direct line plot (No Gaussian smoothing)
#     ax_cr.plot(x, cr3, linestyle="-", color=COLOR_LINE, linewidth=3.2, zorder=5)
#     ax_cr.plot(x, cr3, "o", color=COLOR_LINE, markersize=9, markeredgecolor="white", markeredgewidth=2.0, zorder=6)

#     cr_max = max(cr3.max() * 1.5, 0.1)
#     ax_cr.set_ylim(0, cr_max)
#     ax_cr.set_ylabel("3-Layer Collision Rate", color=COLOR_LINE, fontweight="bold")
#     ax_cr.tick_params(axis="y", labelcolor=COLOR_LINE, width=1.2)
#     ax_cr.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
#     ax_cr.spines["top"].set_visible(False)
#     ax_cr.spines["left"].set_visible(False)  # Remove left spine (belongs to main axis)

#     # Legend inside the plot (Top Right) - BORDERLESS
#     bar_patch  = mpatches.Patch(facecolor=COLOR_BAR, edgecolor=COLOR_BAR_EDGE, linewidth=1.3, label="# Items in Decile")
#     line_patch = mpl.lines.Line2D([], [], color=COLOR_LINE, linewidth=3.2, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2.0, label="Collision Rate")
#     ax_main.legend(
#         handles=[bar_patch, line_patch],
#         loc="upper right",
#         frameon=False,        # <--- BORDERLESS LEGEND (Top-Tier Style)
#         ncol=2,
#         columnspacing=1.5,    # More space between legend columns
#         handlelength=2.0,     # Longer legend handles for clarity
#         borderaxespad=0.3
#     )

#     # =========================================================================
#     # PANEL B — Layer-wise collision rate
#     # =========================================================================

#     layer_cfg = [
#         ("Layer 1", "-s", LAYER_COLS[0], 0),
#         ("Layer 2", "-^", LAYER_COLS[1], 1),
#         ("Layer 3", "-o", LAYER_COLS[2], 2),
#     ]

#     for lname, lstyle, lcol, li in layer_cfg:
#         y_vals = layer_rates[:, li]
#         ax_side.plot(
#             x, y_vals, lstyle, color=lcol,
#             linewidth=2.8, markersize=8,
#             markeredgecolor="white", markeredgewidth=1.5,
#             label=lname, zorder=4,
#         )

#     ax_side.set_xlim(-0.5, n - 0.5)
#     ax_side.set_xticks(x)
#     ax_side.set_xticklabels(x_labels)  # Use default fontsize (16 from rcParams)
#     ax_side.set_xlabel("Semantic Density Decile", fontweight="bold")
#     # Increase spacing between x-tick labels
#     ax_side.tick_params(axis='x', pad=8)
#     ax_side.set_ylabel("Collision Rate", fontweight="bold")
#     ax_side.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    
#     # Clean up grid and spines for side panel
#     ax_side.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35, color='#CCCCCC')
#     ax_side.grid(axis='x', visible=False)
#     ax_side.spines["top"].set_visible(False)
#     ax_side.spines["right"].set_visible(False)
    
#     # Ensure y-axis starts at 0 for proper visual comparison
#     ax_side.set_ylim(bottom=0)
    
#     ax_side.legend(
#         loc="upper right", 
#         frameon=False,        # <--- BORDERLESS LEGEND
#         bbox_to_anchor=(0.98, 0.90),  # Move legend down slightly to avoid line overlap
#         handlelength=2.5,     # Longer legend handles for better visibility
#         borderaxespad=0.3
#     )

#     # ── Save ─────────────────────────────────────────────────────────────────
#     for ext in ["pdf", "png"]:
#         p = os.path.splitext(output_path)[0] + f".{ext}"
#         fig.savefig(p, dpi=300, format=ext)
#         print(f"[✓] Saved zero-padding academic figure: {p}")
#     plt.close(fig)

# # ---------------------------------------------------------------------------
# # 3. CLI
# # ---------------------------------------------------------------------------

# def parse_args():
#     parser = argparse.ArgumentParser(description="Publication-ready semantic ID collision figure generator.")
#     parser.add_argument("--stats_file", type=str, default="./output/tokenizer/collision_stats.json")
#     parser.add_argument("--output_dir", type=str, default="./output/figures")
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     if not os.path.exists(args.stats_file):
#         print(f"[✗] stats_file not found: {args.stats_file}")
#         sys.exit(1)

#     os.makedirs(args.output_dir, exist_ok=True)
#     stats = load_stats(args.stats_file)
    
#     main_path = os.path.join(args.output_dir, "fig1_collision_dual_axis.pdf")
#     plot_main_figure(stats, main_path)

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_collision_analysis.py

Clean, publication-ready dual-axis visualization for Semantic ID Collision Rate.
Optimized for Top-Tier Academic Conferences (ICLR / KDD / SIGIR).
Features: Zero-padding, large readable fonts, borderless legends.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ---------------------------------------------------------------------------
# 0. Aesthetics — Top-Tier Academic Style (Linux Libertine + High-Contrast)
# ---------------------------------------------------------------------------

# Add Linux Libertine fonts from the opentype directory
libertine_font_dir = '/home/fangdengzhao/Fonts/libertine/opentype'
libertine_fonts = [
    f'{libertine_font_dir}/LinLibertine_R.otf',      # Regular
    f'{libertine_font_dir}/LinLibertine_RI.otf',     # Italic
    f'{libertine_font_dir}/LinLibertine_RB.otf',     # Bold
    f'{libertine_font_dir}/LinLibertine_RBI.otf',    # Bold Italic
]

for font_file in libertine_fonts:
    if os.path.exists(font_file):
        fm.fontManager.addfont(font_file)

mpl.rcParams.update({
    # --- Font Configuration (Linux Libertine) ---
    "font.family":        "Linux Libertine O",
    "font.weight":        "normal",
    "mathtext.fontset":   "custom",
    "mathtext.rm":        "Linux Libertine O",
    "mathtext.it":        "Linux Libertine O:italic",
    "mathtext.bf":        "Linux Libertine O:bold",

    # --- Base Font Sizes (Preserved your larger target sizes) ---
    "font.size":          15,      # Base font size increased for better readability
    "axes.titlesize":     16,
    "axes.labelsize":     16,      # Keep axis labels at 16
    "xtick.labelsize":    16,      # Increased tick labels
    "ytick.labelsize":    16,      # Increased tick labels
    "legend.fontsize":    16,      # Larger legend text

    # --- Axes and Ticks Styling (Minimalist, High-Contrast) ---
    "axes.linewidth":     1.2,     # Thicker axes for professional look
    "axes.edgecolor":     "#2C3E50",  # Darker, more professional edge color
    "xtick.color":        "#2C3E50",
    "ytick.color":        "#2C3E50",
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "xtick.major.size":   6,
    "ytick.major.size":   6,
    "xtick.direction":    "in",    # Inward ticks (classic academic style)
    "ytick.direction":    "in",

    # --- Figure and Export Settings ---
    "figure.dpi":         300,     # Upgraded to 300 for crisp inline rendering in notebooks
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,    # Minimal padding for professional appearance
})

# Refined Academic Color Palette (Preserved)
COLOR_BAR       = "#7B9EBD"  # Elegant Slate Blue
COLOR_BAR_EDGE  = "#4A6D8A"  # Darker edge for crisp rendering
COLOR_LINE      = "#B22222"  # Firebrick Red (Strong contrast, professional)
COLOR_TEXT      = "#2C3E50"  # Dark Charcoal for annotations
LAYER_COLS      = ["#4C72B0", "#DD8452", "#B22222"] # Muted Blue, Muted Orange, Firebrick

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_stats(stats_file: str) -> dict:
    with open(stats_file, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_arrays(stats: dict):
    buckets     = sorted(stats["buckets"], key=lambda b: b["bucket_id"])
    ids         = np.array([b["bucket_id"]                for b in buckets])
    n_items     = np.array([b["total_items"]               for b in buckets])
    cr3         = np.array([b["collision_rate_3layer"]     for b in buckets])
    layer_rates = np.array([b["layer_collision_rates"]     for b in buckets])
    return ids, n_items, cr3, layer_rates

# ---------------------------------------------------------------------------
# 2. Main dual-axis figure
# ---------------------------------------------------------------------------

def plot_main_figure(stats: dict, output_path: str):
    ids, n_items, cr3, layer_rates = extract_arrays(stats)

    n  = len(ids)
    x  = np.arange(n)
    x_labels = [f"D{i+1}" for i in range(n)] 

    # ── Figure & layout ──────────────────────────────────────────────────────
    # layout="constrained" smartly handles spacing to prevent overlaps
    fig = plt.figure(figsize=(13.5, 4.5), layout="constrained") 
    
    # MODIFICATION HERE: Changed width_ratios from [2.0, 1.2] to [1.0, 1.0] for symmetry
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0], wspace=0.14) 
    
    ax_main = fig.add_subplot(gs[0])
    ax_side = fig.add_subplot(gs[1])

    # =========================================================================#
    # PANEL A — Bar (item count) + Line (collision rate)
    # =========================================================================

    bar_kw = dict(width=0.68, align="center", zorder=2, color=COLOR_BAR, edgecolor=COLOR_BAR_EDGE, linewidth=1.3)
    bars = ax_main.bar(x, n_items, **bar_kw)

    # Count labels on top of bars
    y_max = n_items.max() * 1.32 # Extra room for the legend and labels
    # for bar, val in zip(bars, n_items):
    #     ax_main.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         bar.get_height() + y_max * 0.015,
    #         f"{val:,}",
    #         ha="center", va="bottom",
    #         fontsize=12, color=COLOR_TEXT, fontweight="600"  # Keep original size for bar labels
    #     )

    ax_main.set_xlim(-0.6, n - 0.4)
    ax_main.set_ylim(0, y_max)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(x_labels)
    ax_main.set_xlabel("Semantic Density Decile", fontweight="bold")
    ax_main.set_ylabel("Number of Items", color=COLOR_BAR_EDGE, fontweight="bold")
    ax_main.tick_params(axis="y", labelcolor=COLOR_BAR_EDGE)
    
    # Clean up grid and spines
    ax_main.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35, color='#CCCCCC')
    ax_main.grid(axis='x', visible=False)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)  # Remove right spine for cleaner look

    # ── Twin axis: collision rate ─────────────────────────────────────────────
    ax_cr = ax_main.twinx()

    # Direct line plot (No Gaussian smoothing)
    ax_cr.plot(x, cr3, linestyle="-", color=COLOR_LINE, linewidth=3.2, zorder=5)
    ax_cr.plot(x, cr3, "o", color=COLOR_LINE, markersize=9, markeredgecolor="white", markeredgewidth=2.0, zorder=6)

    cr_max = max(cr3.max() * 1.5, 0.1)
    ax_cr.set_ylim(0, cr_max)
    ax_cr.set_ylabel("3-Layer Collision Rate", color=COLOR_LINE, fontweight="bold")
    ax_cr.tick_params(axis="y", labelcolor=COLOR_LINE, width=1.2)
    ax_cr.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_cr.spines["top"].set_visible(False)
    ax_cr.spines["left"].set_visible(False)  # Remove left spine (belongs to main axis)

    # Legend inside the plot (Top Right) - BORDERLESS
    bar_patch  = mpatches.Patch(facecolor=COLOR_BAR, edgecolor=COLOR_BAR_EDGE, linewidth=1.3, label="Item Count")
    line_patch = mpl.lines.Line2D([], [], color=COLOR_LINE, linewidth=3.2, marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2.0, label="Collision Rate")
    ax_main.legend(
        handles=[bar_patch, line_patch],
        loc="upper center",
        frameon=False,        # <--- BORDERLESS LEGEND (Top-Tier Style)
        ncol=2,
        columnspacing=1.5,    # More space between legend columns
        handlelength=2.0,     # Longer legend handles for clarity
        borderaxespad=0.3
    )

    # =========================================================================
    # PANEL B — Layer-wise collision rate
    # =========================================================================

    layer_cfg = [
        ("Layer 1", "-s", LAYER_COLS[0], 0),
        ("Layer 2", "-^", LAYER_COLS[1], 1),
        ("Layer 3", "-o", LAYER_COLS[2], 2),
    ]

    for lname, lstyle, lcol, li in layer_cfg:
        y_vals = layer_rates[:, li]
        ax_side.plot(
            x, y_vals, lstyle, color=lcol,
            linewidth=2.8, markersize=8,
            markeredgecolor="white", markeredgewidth=1.5,
            label=lname, zorder=4,
        )

    ax_side.set_xlim(-0.5, n - 0.5)
    ax_side.set_xticks(x)
    ax_side.set_xticklabels(x_labels)  # Use default fontsize (16 from rcParams)
    ax_side.set_xlabel("Semantic Density Decile", fontweight="bold")
    # Increase spacing between x-tick labels
    ax_side.tick_params(axis='x', pad=8)
    ax_side.set_ylabel("Collision Rate", fontweight="bold")
    ax_side.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    
    # Clean up grid and spines for side panel
    ax_side.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.35, color='#CCCCCC')
    ax_side.grid(axis='x', visible=False)
    ax_side.spines["top"].set_visible(False)
    ax_side.spines["right"].set_visible(False)
    
    # Ensure y-axis starts at 0 for proper visual comparison
    ax_side.set_ylim(bottom=0)
    
    ax_side.legend(
        loc="upper right", 
        frameon=False,        # <--- BORDERLESS LEGEND
        bbox_to_anchor=(0.98, 0.90),  # Move legend down slightly to avoid line overlap
        handlelength=2.5,     # Longer legend handles for better visibility
        borderaxespad=0.3
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    for ext in ["pdf", "png"]:
        p = os.path.splitext(output_path)[0] + f".{ext}"
        fig.savefig(p, dpi=300, format=ext)
        print(f"[✓] Saved zero-padding academic figure: {p}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Publication-ready semantic ID collision figure generator.")
    parser.add_argument("--stats_file", type=str, default="./output/tokenizer/collision_stats.json")
    parser.add_argument("--output_dir", type=str, default="./output/figures")
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.stats_file):
        print(f"[✗] stats_file not found: {args.stats_file}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    stats = load_stats(args.stats_file)
    
    main_path = os.path.join(args.output_dir, "fig1_collision_dual_axis.pdf")
    plot_main_figure(stats, main_path)

if __name__ == "__main__":
    main()