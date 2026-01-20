"""
RQ-VAE Ablation Study: 同类别 vs 不同类别 距离分布可视化

对每个变体，计算同类别item对的embedding距离分布（应该小）
和不同类别item对的embedding距离分布（应该大），用violin plot展示。

好的方法：两个分布分离度高（同类别距离小，不同类别距离大）

Pairwise Distance: 两个item的embedding之间的欧氏距离
- 同类别距离小 → 保持语义相似性
- 不同类别距离大 → 保持区分性
- 分离度高 → 既有相似性又有区分性

Separation Score: (inter_mean - intra_mean) / (intra_std + inter_std)
- 越高越好，表示两个分布分离度越高
"""

import argparse
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist
import os
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置 Linux Libertine 字体
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

COLORS = {
    'intra': '#5B9BD5',   # 柔和蓝色 - 同类别 (相似性)
    'inter': '#ED7D31',   # 柔和橙色 - 不同类别 (区分性)
}

# 变体显示名称
VARIANT_NAMES = {
    'semantic_only': 'Semantic\nOnly',
    'collab_only': 'Collab\nOnly',
    'concat': 'Concat',
    'contrastive': 'Contrastive',
    'gated_dual': 'Gated\nDual (Ours)',
}

# 变体顺序
VARIANT_ORDER = ['semantic_only', 'collab_only', 'concat', 'contrastive', 'gated_dual']


def load_embeddings(variant_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """加载量化后的embeddings和item_ids"""
    emb_path = variant_dir / 'quantized_embeddings.npy'
    ids_path = variant_dir / 'item_ids.npy'
    
    embeddings = np.load(emb_path)
    item_ids = np.load(ids_path)
    
    logger.info(f"  加载 {variant_dir.name}: {embeddings.shape}")
    return embeddings, item_ids


def load_semantic_mappings(variant_dir: Path) -> Dict[int, List[int]]:
    """加载语义ID映射"""
    mapping_path = variant_dir / 'semantic_id_mappings.json'
    with open(mapping_path, 'r') as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def assign_categories(semantic_mappings: Dict[int, List[int]], min_samples: int = 20) -> Dict[str, List[int]]:
    """根据语义ID的第一个code分配类别"""
    category_to_items = defaultdict(list)
    
    for item_id, codes in semantic_mappings.items():
        if codes and len(codes) >= 1:
            category = f"Cat_{codes[0]}"
            category_to_items[category].append(item_id)
    
    # 过滤样本数过少的类别
    filtered = {cat: items for cat, items in category_to_items.items() 
                if len(items) >= min_samples}
    
    logger.info(f"  类别数: {len(category_to_items)} -> 过滤后: {len(filtered)}")
    return filtered



def compute_distance_distributions(
    embeddings: np.ndarray,
    item_ids: np.ndarray,
    category_to_items: Dict[str, List[int]],
    n_intra_samples: int = 10000,
    n_inter_samples: int = 10000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算同类别和不同类别的距离分布
    
    Args:
        embeddings: (N, D) 量化后的embeddings
        item_ids: (N,) item IDs
        category_to_items: 类别到item列表的映射
        n_intra_samples: 同类别采样对数
        n_inter_samples: 不同类别采样对数
        seed: 随机种子
    
    Returns:
        intra_distances: 同类别距离数组
        inter_distances: 不同类别距离数组
    """
    np.random.seed(seed)
    
    # 建立item_id到embedding索引的映射
    id_to_idx = {int(item_id): idx for idx, item_id in enumerate(item_ids)}
    
    # 获取所有有效类别
    categories = list(category_to_items.keys())
    
    # 计算同类别距离 (intra-class)
    intra_distances = []
    samples_per_category = max(1, n_intra_samples // len(categories))
    
    for cat, items in category_to_items.items():
        valid_items = [item for item in items if item in id_to_idx]
        if len(valid_items) < 2:
            continue
        
        # 随机采样pairs
        n_pairs = min(samples_per_category, len(valid_items) * (len(valid_items) - 1) // 2)
        for _ in range(n_pairs):
            i, j = np.random.choice(len(valid_items), 2, replace=False)
            idx_i = id_to_idx[valid_items[i]]
            idx_j = id_to_idx[valid_items[j]]
            dist = np.linalg.norm(embeddings[idx_i] - embeddings[idx_j])
            intra_distances.append(dist)
    
    # 计算不同类别距离 (inter-class)
    inter_distances = []
    for _ in range(n_inter_samples):
        # 随机选择两个不同类别
        cat1, cat2 = np.random.choice(categories, 2, replace=False)
        items1 = [item for item in category_to_items[cat1] if item in id_to_idx]
        items2 = [item for item in category_to_items[cat2] if item in id_to_idx]
        
        if len(items1) == 0 or len(items2) == 0:
            continue
        
        item1 = np.random.choice(items1)
        item2 = np.random.choice(items2)
        
        idx1 = id_to_idx[item1]
        idx2 = id_to_idx[item2]
        dist = np.linalg.norm(embeddings[idx1] - embeddings[idx2])
        inter_distances.append(dist)
    
    return np.array(intra_distances), np.array(inter_distances)


def compute_separation_score(intra_distances: np.ndarray, inter_distances: np.ndarray) -> float:
    """
    计算分离度得分 (越高越好)
    使用 (inter_mean - intra_mean) / (intra_std + inter_std)
    """
    intra_mean = np.mean(intra_distances)
    inter_mean = np.mean(inter_distances)
    intra_std = np.std(intra_distances)
    inter_std = np.std(inter_distances)
    
    separation = (inter_mean - intra_mean) / (intra_std + inter_std + 1e-8)
    return separation



def plot_distance_distributions(
    all_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str,
    figsize: Tuple[float, float] = (7.5, 3.0)
):
    """
    绘制所有变体的距离分布对比图
    
    使用split violin plot，左半边是同类别距离，右半边是不同类别距离
    """
    # 设置顶会风格 - 增大字体以适应论文排版
    plt.rcParams.update({
        'font.family': 'Linux Libertine O',
        'font.weight': 'normal',
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Linux Libertine O',
        'mathtext.it': 'Linux Libertine O:italic',
        'mathtext.bf': 'Linux Libertine O:bold',
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 1.0,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    positions = np.arange(len(VARIANT_ORDER))
    
    # 收集数据用于violin plot
    intra_data = []
    inter_data = []
    separation_scores = []
    
    for variant in VARIANT_ORDER:
        intra_dist, inter_dist = all_results[variant]
        intra_data.append(intra_dist)
        inter_data.append(inter_dist)
        sep_score = compute_separation_score(intra_dist, inter_dist)
        separation_scores.append(sep_score)
        logger.info(f"  {variant}: intra_mean={np.mean(intra_dist):.3f}, "
                   f"inter_mean={np.mean(inter_dist):.3f}, separation={sep_score:.3f}")
    
    # 绘制split violin plot
    def draw_half_violin(data, positions, side='left', color='blue', alpha=0.7):
        """绘制半边violin"""
        parts = ax.violinplot(data, positions=positions, showmeans=False, 
                             showmedians=False, showextrema=False, widths=0.75)
        
        for i, pc in enumerate(parts['bodies']):
            # 获取violin的路径
            m = np.mean(pc.get_paths()[0].vertices[:, 0])
            if side == 'left':
                pc.get_paths()[0].vertices[:, 0] = np.clip(
                    pc.get_paths()[0].vertices[:, 0], -np.inf, m)
            else:
                pc.get_paths()[0].vertices[:, 0] = np.clip(
                    pc.get_paths()[0].vertices[:, 0], m, np.inf)
            pc.set_facecolor(color)
            pc.set_edgecolor('white')
            pc.set_alpha(alpha)
            pc.set_linewidth(0.5)
    
    # 绘制左半边 (同类别 - 柔和蓝色)
    draw_half_violin(intra_data, positions, side='left', color=COLORS['intra'], alpha=0.75)
    
    # 绘制右半边 (不同类别 - 柔和橙色)
    draw_half_violin(inter_data, positions, side='right', color=COLORS['inter'], alpha=0.75)
    
    # 添加均值标记 (小圆点) - 增大标记以便在论文中清晰可见
    for i, (intra, inter) in enumerate(zip(intra_data, inter_data)):
        # 同类别均值 (左侧)
        ax.scatter(positions[i] - 0.12, np.mean(intra), color='white', 
                  edgecolor=COLORS['intra'], s=35, zorder=5, linewidth=1.5, marker='o')
        # 不同类别均值 (右侧)
        ax.scatter(positions[i] + 0.12, np.mean(inter), color='white',
                  edgecolor=COLORS['inter'], s=35, zorder=5, linewidth=1.5, marker='o')
    
    # 设置x轴
    ax.set_xticks(positions)
    ax.set_xticklabels([VARIANT_NAMES[v] for v in VARIANT_ORDER])
    
    # 设置y轴
    ax.set_ylabel('Pairwise Distance', labelpad=6)
    ax.set_ylim(bottom=0, top=2.6)
    
    # 添加分离度得分标注 (相对于各自violin顶部的固定偏移)
    offset_above_violin = 0.08  # 相对于violin顶部的偏移量
    for i, score in enumerate(separation_scores):
        # 计算该位置的violin顶部（取95分位数）
        y_top = max(np.percentile(intra_data[i], 98), np.percentile(inter_data[i], 98))
        y_text = y_top + offset_above_violin
        
        # 高亮最佳结果
        if i == len(separation_scores) - 1:  # gated_dual
            ax.text(positions[i], y_text, f'Sep: {score:.2f}', 
                   ha='center', va='bottom', fontsize=11, 
                   color='#2E7D32', fontweight='bold')  # 柔和的深绿色
        else:
            ax.text(positions[i], y_text, f'Sep: {score:.2f}', 
                   ha='center', va='bottom', fontsize=10.5, color='#757575')  # 柔和的灰色
    
    # 添加图例 (左上角，避免与右侧的sep标注重叠)
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['intra'], alpha=0.75, 
                      edgecolor='white', linewidth=0.5, label='Intra-category'),
        mpatches.Patch(facecolor=COLORS['inter'], alpha=0.75,
                      edgecolor='white', linewidth=0.5, label='Inter-category'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False,
             handlelength=1.0, handletextpad=0.4, borderpad=0.2, columnspacing=0.8)
    
    # 添加淡网格线
    ax.yaxis.grid(True, linestyle='--', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 紧凑布局
    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, bbox_inches='tight', dpi=300, 
               facecolor='white', edgecolor='none', pad_inches=0.05)
    plt.savefig(output_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=300,
               facecolor='white', edgecolor='none', pad_inches=0.05)
    logger.info(f"图片已保存: {output_path}")
    plt.close()



def get_project_root() -> Path:
    """自动检测项目根目录（包含pyproject.toml的目录）"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    # 如果找不到，返回当前工作目录
    return Path.cwd()


def main():
    parser = argparse.ArgumentParser(description='RQ-VAE Ablation: 距离分布可视化')
    parser.add_argument('--ablation_dir', type=str, 
                        default='scripts/output/rqvae_ablation/beauty',
                        help='ablation study输出目录（相对于项目根目录）')
    parser.add_argument('--output', type=str,
                        default='scripts/output/rqvae_ablation/beauty/distance_distribution.pdf',
                        help='输出图片路径（相对于项目根目录）')
    parser.add_argument('--n_intra_samples', type=int, default=50000,
                        help='同类别采样对数')
    parser.add_argument('--n_inter_samples', type=int, default=50000,
                        help='不同类别采样对数')
    parser.add_argument('--min_category_samples', type=int, default=20,
                        help='类别最小样本数阈值')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 自动检测项目根目录
    project_root = get_project_root()
    logger.info(f"项目根目录: {project_root}")
    
    # 转换为绝对路径
    ablation_dir = project_root / args.ablation_dir
    if not ablation_dir.exists():
        # 如果相对路径不存在，尝试作为绝对路径
        ablation_dir = Path(args.ablation_dir)
    
    logger.info(f"Ablation目录: {ablation_dir}")
    
    # 存储所有变体的结果
    all_results = {}
    
    # 使用第一个变体的语义映射来定义类别（所有变体应该有相同的item集合）
    reference_variant = 'gated_dual'
    ref_dir = ablation_dir / reference_variant
    semantic_mappings = load_semantic_mappings(ref_dir)
    category_to_items = assign_categories(semantic_mappings, args.min_category_samples)
    
    logger.info(f"\n使用 {reference_variant} 的语义映射定义类别")
    logger.info(f"有效类别数: {len(category_to_items)}")
    
    # 处理每个变体
    for variant in VARIANT_ORDER:
        variant_dir = ablation_dir / variant
        if not variant_dir.exists():
            logger.warning(f"变体目录不存在: {variant_dir}")
            continue
        
        logger.info(f"\n处理变体: {variant}")
        
        # 加载embeddings
        embeddings, item_ids = load_embeddings(variant_dir)
        
        # 计算距离分布
        intra_dist, inter_dist = compute_distance_distributions(
            embeddings, item_ids, category_to_items,
            n_intra_samples=args.n_intra_samples,
            n_inter_samples=args.n_inter_samples,
            seed=args.seed
        )
        
        all_results[variant] = (intra_dist, inter_dist)
        logger.info(f"  同类别样本对: {len(intra_dist)}, 不同类别样本对: {len(inter_dist)}")
    
    # 处理输出路径
    output_path = project_root / args.output
    if not output_path.parent.exists():
        # 如果相对路径的父目录不存在，尝试作为绝对路径
        output_path = Path(args.output)
    
    # 绘制对比图
    logger.info("\n绘制距离分布对比图...")
    plot_distance_distributions(all_results, str(output_path))
    
    # 打印总结
    print("\n" + "="*60)
    print("分离度得分总结 (越高越好)")
    print("="*60)
    for variant in VARIANT_ORDER:
        if variant in all_results:
            intra, inter = all_results[variant]
            sep = compute_separation_score(intra, inter)
            print(f"  {VARIANT_NAMES[variant].replace(chr(10), ' ')}: {sep:.3f}")
    print("="*60)


if __name__ == '__main__':
    main()
