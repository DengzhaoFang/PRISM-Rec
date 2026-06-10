import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ==========================================
# 1. Font Configuration: Linux Libertine O
# ==========================================
libertine_font_dir = '/home/fangdengzhao/Fonts/libertine/opentype'
libertine_fonts = [
    f'{libertine_font_dir}/LinLibertine_R.otf',    # Regular
    f'{libertine_font_dir}/LinLibertine_RI.otf',   # Italic
    f'{libertine_font_dir}/LinLibertine_RB.otf',   # Bold
    f'{libertine_font_dir}/LinLibertine_RBI.otf',  # Bold Italic
]

for font_file in libertine_fonts:
    if os.path.exists(font_file):
        fm.fontManager.addfont(font_file)

plt.rcParams.update({
    'font.family': 'Linux Libertine O',
    'font.weight': 'normal',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Linux Libertine O',
    'mathtext.it': 'Linux Libertine O:italic',
    'mathtext.bf': 'Linux Libertine O:bold',

    # 字体整体稍增大，同时依靠更紧凑布局提升“视觉字号”
    'font.size': 14,
    'axes.labelsize': 24,
    'axes.titlesize': 22,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 22,

    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',

    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,

    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.major.size': 4,
    'ytick.major.size': 4,

    'grid.linewidth': 0.7,
    'grid.alpha': 0.28,

    # 可编辑矢量字体
    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    # hatch 更明显，便于区分低柱子
    'hatch.linewidth': 1.05,
})

# ==========================================
# 2. Elegant, low-saturation palette
# 更低饱和度、更精致，同时依赖边框和花纹增强低值柱子的辨识度
# ==========================================
COLORS = {
    'TIGER':   '#DCEAF4',   # pale mist blue
    'ADC-SID': '#EAE3D3',   # muted parchment
    'ADSA':    '#E9D9DF',   # dusty blush
}

# 两种明显且适合小柱子的花纹：
# - 斜线 ////// ：线性强，低柱子也容易看出来
# - 交叉 xxxx   ：与斜线区分度明显
HATCHES = {
    'Recall@10': '////',
    'NDCG@10': 'xxxx',
}

EDGE_COLOR = '#6A6A6A'
GRID_COLOR = '#DCDCDC'
FACE_COLOR = '#FFFFFF'

# ==========================================
# 3. Data (unchanged)
# ==========================================
DATA = {
    'Beauty': {
        'groups': ['Popular', 'Medium', 'Long-tail'],
        'counts': [12678, 5188, 4497],
        'metrics': {
            'TIGER':   {'Recall@10': [0.088, 0.012, 0.010], 'NDCG@10': [0.052, 0.006, 0.005]},
            'ADC-SID': {'Recall@10': [0.095, 0.018, 0.015], 'NDCG@10': [0.056, 0.011, 0.009]},
            'ADSA':    {'Recall@10': [0.118, 0.024, 0.022], 'NDCG@10': [0.060, 0.014, 0.012]}
        }
    },
    'CDs': {
        'groups': ['Popular', 'Medium', 'Long-tail'],
        'counts': [46796, 14963, 13499],
        'metrics': {
            'TIGER':   {'Recall@10': [0.078, 0.007, 0.003], 'NDCG@10': [0.042, 0.004, 0.002]},
            'ADC-SID': {'Recall@10': [0.108, 0.016, 0.014], 'NDCG@10': [0.055, 0.008, 0.006]},
            'ADSA':    {'Recall@10': [0.119, 0.019, 0.018], 'NDCG@10': [0.059, 0.010, 0.008]}
        }
    }
}

from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ==========================================
# Single-column friendly plotting function
# 适合 LaTeX 中使用:
# \includegraphics[width=\columnwidth]{longtail_comparison_singlecol.pdf}
# ==========================================
def plot_longtail_performance(output_filename="longtail_comparison_singlecol"):
    datasets = list(DATA.keys())
    models = ['TIGER', 'ADC-SID', 'ADSA']
    metric_names = ['Recall@10', 'NDCG@10']

    # 关键：单栏图不要用 17 inch 宽的大画布
    # 这里的宽度接近双栏论文的 column width
    fig, axes = plt.subplots(
        2, 1,
        figsize=(4.2, 6.2),
        sharey=True
    )

    # 单栏下需要更紧凑的横向布局
    bar_width = 0.185
    inner_gap = 0.006
    metric_gap = 0.040
    group_gap = 0.125

    cluster_width = (3 * bar_width) + (2 * inner_gap)
    group_width = (2 * cluster_width) + metric_gap
    stride = group_width + group_gap

    global_max = 0.0

    ylabel_map = {
    'Beauty': 'Beauty Dataset Performance',
    'CDs': 'CDs Dataset Performance'
    }

    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        d_info = DATA[dataset]
        groups = d_info['groups']
        counts = d_info['counts']
        n_groups = len(groups)

        x_centers = np.arange(n_groups) * stride

        for g_idx in range(n_groups):
            group_center = x_centers[g_idx]

            for m_idx, metric in enumerate(metric_names):
                if m_idx == 0:
                    cluster_offset = -(group_width / 2) + (cluster_width / 2)
                else:
                    cluster_offset =  (group_width / 2) - (cluster_width / 2)

                for mod_idx, model in enumerate(models):
                    val = d_info['metrics'][model][metric][g_idx]
                    global_max = max(global_max, val)

                    bar_offset = (mod_idx - 1) * (bar_width + inner_gap)
                    final_x = group_center + cluster_offset + bar_offset

                    ax.bar(
                        final_x,
                        val,
                        width=bar_width,
                        color=COLORS[model],
                        edgecolor=EDGE_COLOR,
                        linewidth=0.85,
                        hatch=HATCHES[metric],
                        zorder=3
                    )

        # 数据集标题：不加粗，但字号比 tick 大
        # ax.set_title(
        #     f'{dataset} Dataset',
        #     fontsize=10.8,
        #     fontweight='normal',
        #     pad=4
        # )

        ax.set_xticks(x_centers)

        # 单栏图里不要写太长，去掉括号可节省空间
        xlabels = [f'{g}\nn={c:,}' for g, c in zip(groups, counts)]
        ax.set_xticklabels(
            xlabels,
            fontsize=9.4,
            linespacing=0.92
        )
        ax.set_ylabel(
        ylabel_map[dataset],
        fontsize=10.0,
        labelpad=8
        )

        ax.tick_params(axis='x', pad=1.5)
        ax.tick_params(axis='y', labelsize=9.4, pad=2)

        ax.grid(True, axis='y', color=GRID_COLOR, linestyle='-', zorder=0)
        ax.set_axisbelow(True)

        ax.set_facecolor(FACE_COLOR)
        ax.spines['left'].set_color(EDGE_COLOR)
        ax.spines['bottom'].set_color(EDGE_COLOR)

        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # 关键：手动收紧左右空白
        left_bound = x_centers[0] - group_width / 2 - 0.035
        right_bound = x_centers[-1] + group_width / 2 + 0.035
        ax.set_xlim(left_bound, right_bound)

    # y 轴上限不要放太高，否则长尾柱子更矮
    # 当前最大值约 0.119，用 0.13 比 0.14/0.15 更紧凑
    for ax in axes:
        ax.set_ylim(0, 0.13)

    # 公共 y 轴标题，避免每个子图都占空间
    # fig.text(
    #     0.015, 0.50,
    #     'Performance',
    #     va='center',
    #     rotation='vertical',
    #     fontsize=10.0
    # )

    # ==========================================
    # Compact legend
    # ==========================================
    # model_patches = [
    #     Patch(
    #         facecolor=COLORS[m],
    #         edgecolor=EDGE_COLOR,
    #         linewidth=0.85,
    #         label=m
    #     )
    #     for m in models
    # ]

    # metric_patches = [
    #     Patch(
    #         facecolor='#F7F7F7',
    #         edgecolor=EDGE_COLOR,
    #         linewidth=0.85,
    #         hatch=HATCHES[m],
    #         label=m
    #     )
    #     for m in metric_names
    # ]

    # legend_elements = model_patches + metric_patches

    # # 单栏里 5 个 legend 横排会太小，所以用 3 列，两行
    # fig.legend(
    #     handles=legend_elements,
    #     loc='upper center',
    #     bbox_to_anchor=(0.53, 0.995),
    #     ncol=3,
    #     frameon=False,
    #     fontsize=8.6,
    #     columnspacing=0.75,
    #     handlelength=1.15,
    #     handleheight=0.75,
    #     handletextpad=0.35,
    #     borderaxespad=0.0
    # )
    # ==========================================
# Compact two-line centered legend
# ==========================================
    model_patches = [
        Patch(facecolor=COLORS['TIGER'], edgecolor=EDGE_COLOR, linewidth=0.85, label='TIGER'),
        Patch(facecolor=COLORS['ADC-SID'], edgecolor=EDGE_COLOR, linewidth=0.85, label='ADC-SID'),
        Patch(facecolor=COLORS['ADSA'], edgecolor=EDGE_COLOR, linewidth=0.85, label='ADSA'),
    ]

    metric_patches = [
        Patch(
            facecolor='#F7F7F7',
            edgecolor=EDGE_COLOR,
            linewidth=0.85,
            hatch=HATCHES['Recall@10'],
            label='Recall@10'
        ),
        Patch(
            facecolor='#F7F7F7',
            edgecolor=EDGE_COLOR,
            linewidth=0.85,
            hatch=HATCHES['NDCG@10'],
            label='NDCG@10'
        ),
    ]

    # 第一行：三个模型，整体居中
    legend_models = fig.legend(
        handles=model_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.995),
        bbox_transform=fig.transFigure,
        ncol=3,
        frameon=False,
        fontsize=8.8,
        columnspacing=0.95,
        handlelength=1.15,
        handleheight=0.75,
        handletextpad=0.35,
        borderaxespad=0.0
    )

    # 第二行：两个指标，整体居中
    legend_metrics = fig.legend(
        handles=metric_patches,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.955),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        fontsize=8.8,
        columnspacing=0.95,
        handlelength=1.15,
        handleheight=0.75,
        handletextpad=0.35,
        borderaxespad=0.0
    )

    fig.add_artist(legend_models)

    png_path = f"{output_filename}.png"
    pdf_path = f"{output_filename}.pdf"

    plt.savefig(png_path, bbox_inches='tight', pad_inches=0.01)
    plt.savefig(pdf_path, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)

    print(f"Success! Single-column optimized plots saved as '{png_path}' and '{pdf_path}'.")

if __name__ == "__main__":
    plot_longtail_performance()