import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from matplotlib.lines import Line2D

libertine_font_dir = '/home/fangdengzhao/Fonts/libertine/opentype'
libertine_fonts = [
    f'{libertine_font_dir}/LinLibertine_R.otf',
    f'{libertine_font_dir}/LinLibertine_RI.otf',
    f'{libertine_font_dir}/LinLibertine_RB.otf',
    f'{libertine_font_dir}/LinLibertine_RBI.otf',
]

for font_file in libertine_fonts:
    if os.path.exists(font_file):
        fm.fontManager.addfont(font_file)

plt.rcParams.update({
    'font.family': 'Linux Libertine O',
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'figure.titleweight': 'normal',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Linux Libertine O',
    'mathtext.it': 'Linux Libertine O:italic',
    'mathtext.bf': 'Linux Libertine O:bold',
    'font.cursive': ['Linux Libertine O'],
    'font.size': 28,           # 基础字体继续放大
    'axes.labelsize': 30,      # 坐标轴标签继续放大
    'axes.titlesize': 30,
    'xtick.labelsize': 28,     # 刻度标签继续放大
    'ytick.labelsize': 28,
    'legend.fontsize': 28,
    'lines.linewidth': 3.0,    # 线条更粗
    'lines.markersize': 12,    # 标记点更大
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

output_dir = 'hyperparam_plots'
os.makedirs(output_dir, exist_ok=True)

# uniform_x: 是否使用均匀间隔的x轴（对于非等间隔数据如32,64,128）
datasets = [
    # L: 等间隔
    ([2, 3, 4], [0.0697, 0.0713, 0.0718], [0.0371, 0.0387, 0.0375], 
     r'$L$', 'fig_layer', None, False, False),
    # d_cb: 32,64,128 非等间隔，需要均匀化
    ([32, 64, 128], [0.0713, 0.0697, 0.0696], [0.0387, 0.0366, 0.0380], 
     r'$d_{cb}$', 'fig_dim', None, False, True),
    # lambda_1: 添加0和1.2的数据点
    ([0, 0.4, 0.8, 1.2, 1.6], 
     [0.0691, 0.0684, 0.0713, 0.0682, 0.0682], 
     [0.0375, 0.0364, 0.0387, 0.0366, 0.0363], 
     r'$\lambda_1$', 'fig_acd', None, False, False),
    # lambda_2: 添加0和0.3的数据点
    ([0, 0.1, 0.2, 0.3, 0.4], 
     [0.0652, 0.0696, 0.0713, 0.0694, 0.0682], 
     [0.0341, 0.0366, 0.0387, 0.0372, 0.0351], 
     r'$\lambda_2$', 'fig_hsa', None, False, False),
    # gamma: 添加0和0.001的数据点，使用科学记数法
    ([0, 0.0003, 0.0005, 0.0007, 0.001], 
     [0.0701, 0.0678, 0.0713, 0.0681, 0.0689], 
     [0.0381, 0.0366, 0.0387, 0.0358, 0.0365], 
     r'$\gamma$', 'fig_ssa', None, True, False),
    # d_moe: 128,256,512 非等间隔，需要均匀化
    ([128, 256, 512], [0.0687, 0.0713, 0.0703], [0.0371, 0.0387, 0.0378], 
     r'$d_{moe}$', 'fig_moe', None, False, True)
]

COLOR_R10 = '#0072B2'
COLOR_N10 = '#D55E00'


def plot_dual_axis(x, y1, y2, xlabel, filename, xticks=None, use_scientific=False, uniform_x=False):
    """绑制单个超参数图（不含图例）"""
    fig, ax1 = plt.subplots(figsize=(4, 3.2))
    
    # 如果需要均匀间隔，使用索引作为x坐标
    if uniform_x:
        x_plot = list(range(len(x)))
        x_labels = [str(v) for v in x]
    else:
        x_plot = x
        x_labels = None
    
    y1_min, y1_max = min(y1), max(y1)
    y1_span = y1_max - y1_min if y1_max != y1_min else y1_max * 0.1
    y2_min, y2_max = min(y2), max(y2)
    y2_span = y2_max - y2_min if y2_max != y2_min else y2_max * 0.1

    ax1.set_ylim(y1_min - y1_span * 0.8, y1_max + y1_span * 0.2)
    ax2 = ax1.twinx()
    ax2.set_ylim(y2_min - y2_span * 0.2, y2_max + y2_span * 0.8)

    ax1.plot(x_plot, y1, color=COLOR_R10, marker='o', linestyle='-')
    ax2.plot(x_plot, y2, color=COLOR_N10, marker='s', linestyle='--')

    # Hide x-axis label (the symbol like λ, γ, etc.)
    # ax1.set_xlabel(xlabel, fontsize=30)
    ax1.tick_params(axis='y', labelcolor=COLOR_R10)
    ax2.tick_params(axis='y', labelcolor=COLOR_N10)
    ax1.locator_params(axis='y', nbins=5)
    ax2.locator_params(axis='y', nbins=5)

    # 在左侧y轴上方添加 Recall@10 标签
    ax1.text(-0.15, 1.05, 'R@10', transform=ax1.transAxes, 
             fontsize=28, color=COLOR_R10, ha='center', va='bottom')
    
    # 在右侧y轴上方添加 NDCG@10 标签
    ax2.text(1.15, 1.05, 'N@10', transform=ax2.transAxes, 
             fontsize=28, color=COLOR_N10, ha='center', va='bottom')

    ax1.set_xticks(x_plot)
    
    if uniform_x:
        ax1.set_xticklabels(x_labels)
    elif use_scientific:
        scale = 1e4
        x_scaled = [v * scale for v in x]
        ax1.set_xticklabels([f'{int(v)}' for v in x_scaled])
        # 指数放在横坐标右侧，与刻度标签严格在同一水平线上，字体大小与刻度标签一致
        ax1.annotate(r'$\times 10^{-4}$', xy=(1.0, 0), xycoords='axes fraction',
                     xytext=(3, -16), textcoords='offset points',
                     ha='left', va='center', fontsize=28)
    elif xticks:
        ax1.set_xticklabels(xticks)
    elif isinstance(x[0], float) and x[0] < 0.01:
        ax1.set_xticklabels([f'{v:.4f}' for v in x])
    elif isinstance(x[0], float):
        ax1.set_xticklabels([f'{v:.1f}' for v in x])
    else:
        ax1.set_xticklabels([str(v) for v in x])

    plt.savefig(os.path.join(output_dir, filename + '.pdf'), dpi=300)
    plt.savefig(os.path.join(output_dir, filename + '.png'), dpi=300)
    print(f"Saved {filename}")
    plt.close()


# 直接生成所有超参数图
for data in datasets:
    x, y1, y2, xlabel, filename, xticks, use_scientific, uniform_x = data
    plot_dual_axis(x, y1, y2, xlabel, filename, xticks, use_scientific, uniform_x)

print("All plots generated successfully!")
