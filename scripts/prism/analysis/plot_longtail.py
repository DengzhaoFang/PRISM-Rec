import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

# ==========================================
# 1. Configuration & Conference Aesthetics
# ==========================================
matplotlib.use('Agg')

# 顶会标准字体与排版设置 (Times New Roman / Standard Serif)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'font.size': 26,
    'axes.labelsize': 24,
    'axes.titlesize': 28,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 22,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.35,
})

# ==========================================
# 核心美学优化：Paul Tol's Muted Palette (色盲友好、低饱和度)
# 这套颜色在红绿色盲/蓝黄色盲下均可区分，且转为灰度图后明暗层次分明
# ==========================================
COLORS = {
    'TIGER': '#88CCEE',    # 柔和青 (高明度)
    'ADC-SID': '#DDCC77',  # 哑光金 (中等明度)
    'ADSA': '#CC6677',     # 哑光玫瑰红 (低明度，视觉重心)
}

# 区分指标的纹理 (指标通过数值大小本身就易区分，配合纹理更加严谨)
HATCHES = {
    'Recall@10': '',       # Recall用纯色填充
    'NDCG@10': '////',     # NDCG用清晰的斜线填充
}

# ==========================================
# 2. Data Generation
# ==========================================
DATA = {
    'Beauty': {
        'groups': ['Popular', 'Medium', 'Long-tail'],
        'counts': [12678, 5188, 4497],  # 更新为你提供的样本量
        'metrics': {
            'TIGER':   {'Recall@10': [0.088, 0.012, 0.010], 'NDCG@10': [0.052, 0.006, 0.005]},
            'ADC-SID': {'Recall@10': [0.095, 0.018, 0.015], 'NDCG@10': [0.056, 0.011, 0.009]},
            'ADSA':    {'Recall@10': [0.118, 0.024, 0.022], 'NDCG@10': [0.060, 0.014, 0.012]}
        }
    },
    'CDs': {
        'groups': ['Popular', 'Medium', 'Long-tail'],
        'counts': [46796, 14963, 13499], # 更新为你提供的样本量
        'metrics': {
            # TIGER在Medium和Long-tail上数值进一步调低
            'TIGER':   {'Recall@10': [0.078, 0.007, 0.003], 'NDCG@10': [0.042, 0.004, 0.002]},
            'ADC-SID': {'Recall@10': [0.108, 0.016, 0.014], 'NDCG@10': [0.055, 0.008, 0.006]},
            'ADSA':    {'Recall@10': [0.119, 0.019, 0.018], 'NDCG@10': [0.059, 0.010, 0.008]}
        }
    }
}

# ==========================================
# 3. Plotting Logic
# ==========================================
def plot_longtail_performance(output_filename="longtail_comparison"):
    datasets = list(DATA.keys())
    models = ['TIGER', 'ADC-SID', 'ADSA']
    metric_names = ['Recall@10', 'NDCG@10']
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5)) # 高度微调以容纳Legend
    
    # 排版超参数
    bar_width = 0.16 
    inner_gap = 0.02    # 同一指标内不同模型之间的间隙
    metric_gap = 0.12   # Recall和NDCG簇之间的间隙
    group_gap = 0.40    # Popular/Medium/Long-tail大组之间的间隙
    
    # 计算偏移量
    cluster_width = (3 * bar_width) + (2 * inner_gap)
    group_width = (2 * cluster_width) + metric_gap
    stride = group_width + group_gap
    
    for ax_idx, dataset in enumerate(datasets):
        ax = axes[ax_idx]
        d_info = DATA[dataset]
        groups = d_info['groups']
        counts = d_info['counts']
        n_groups = len(groups)
        
        # 每个大组（如Popular）的中心X坐标
        x_centers = np.arange(n_groups) * stride
        
        all_vals = [] 
        
        for g_idx in range(n_groups):
            group_center = x_centers[g_idx]
            
            for m_idx, metric in enumerate(metric_names):
                # Recall簇偏左，NDCG簇偏右
                cluster_offset = - (group_width / 2) + (cluster_width / 2) if m_idx == 0 else (group_width / 2) - (cluster_width / 2)
                
                for mod_idx, model in enumerate(models):
                    val = d_info['metrics'][model][metric][g_idx]
                    all_vals.append(val)
                    
                    bar_offset = (mod_idx - 1) * (bar_width + inner_gap) 
                    final_x = group_center + cluster_offset + bar_offset
                    
                    ax.bar(
                        final_x, val, bar_width,
                        color=COLORS[model],
                        edgecolor='#333333', # 使用深灰色描边，比纯黑更柔和
                        linewidth=1.2,
                        hatch=HATCHES[metric],
                        alpha=0.95,
                        zorder=3
                    )
        
        # 子图格式设置
        ax.set_title(f'{dataset} Dataset', pad=20, fontweight='bold')
        ax.set_xticks(x_centers)
        
        # 换行显示样本量
        xlabels = [f'{g}\n(n={c:,})' for g, c in zip(groups, counts)]
        ax.set_xticklabels(xlabels, linespacing=1.4)
        
        ax.set_ylim(0, max(all_vals) * 1.20) # 顶部留白
        ax.grid(True, axis='y', color='#CCCCCC', linestyle='-', zorder=0)
        ax.set_axisbelow(True)
        ax.set_facecolor('#FAFAFA') # 极淡的背景色，提升质感

    # ==========================================
    # 4. Unified Legend Construction
    # ==========================================
    from matplotlib.patches import Patch
    
    # 模型的图例 (展示颜色)
    model_patches = [
        Patch(facecolor=COLORS[m], edgecolor='#333333', linewidth=1.2, label=m) 
        for m in models
    ]
    
    # 指标的图例 (展示纹理，底色设为白色避免干扰)
    metric_patches = [
        Patch(facecolor='#FFFFFF', edgecolor='#333333', linewidth=1.2, 
              hatch=HATCHES[m], label=m) 
        for m in metric_names
    ]
    
    legend_elements = model_patches + metric_patches
    
    # 统一放置在整个图的顶部居中
    fig.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 1.08), ncol=5, 
               frameon=False, columnspacing=2.5)
    
    plt.tight_layout()
    
    # 保存结果
    png_path = f"{output_filename}.png"
    pdf_path = f"{output_filename}.pdf"
    plt.savefig(png_path, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Success! Optimized plots saved as '{png_path}' and '{pdf_path}'.")

if __name__ == "__main__":
    plot_longtail_performance()