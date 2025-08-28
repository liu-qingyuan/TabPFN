#!/usr/bin/env python3
"""
Feature Number Performance Visualization Script

根据feature_number_comparison.csv生成高质量PDF格式的性能对比图表
用于学术论文发表，DPI设置为900-1200

依赖包安装:
pip install pandas matplotlib numpy seaborn

作者: TabPFN+TCA Medical Research Team
日期: 2025-08-22
"""

import sys
from pathlib import Path
import warnings

# 检查依赖包
required_packages = {
    'pandas': 'pandas',
    'matplotlib.pyplot': 'matplotlib', 
    'numpy': 'numpy',
    'seaborn': 'seaborn'
}

missing_packages = []
for module, package in required_packages.items():
    try:
        __import__(module)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("❌ 缺少必要的依赖包:")
    for package in missing_packages:
        print(f"  - {package}")
    print("\n💡 安装命令:")
    print(f"pip install {' '.join(missing_packages)}")
    print("\n或者使用conda:")
    print(f"conda install {' '.join(missing_packages)}")
    sys.exit(1)

# 导入所有依赖包
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings('ignore')

# 设置Nature期刊级别的图形参数
plt.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'text.usetex': False,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8
})

def load_and_analyze_data(csv_path):
    """加载并分析CSV数据"""
    print(f"📊 加载数据文件: {csv_path}")
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功加载数据，共 {len(df)} 行记录")
        
        # 显示数据基本信息
        print(f"📋 数据列: {list(df.columns)}")
        print(f"📈 特征数量范围: {df['n_features'].min()} - {df['n_features'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        raise

def create_performance_comparison_plot(df, output_path, dpi=1200):
    """
    创建性能对比图表
    
    参数:
    - df: 数据框
    - output_path: 输出路径
    - dpi: 图片分辨率 (900-1200)
    """
    print(f"🎨 生成性能对比图表 (DPI: {dpi})")
    
    # 创建图形和子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # 删除主标题以符合投稿要求
    # fig.suptitle('Feature Number vs. Classification Performance (TabPFN+TCA Medical Cross-Domain Analysis)', 
    #              fontsize=12, fontweight='bold', y=0.96)
    
    # 定义Nature期刊标准配色 - 四色方案
    colors = {
        'accuracy': '#9BDCFC',     # 新浅蓝色
        'auc': '#C9EFBE',          # 新浅绿色
        'f1': '#CAC8EF',           # 新浅紫色
        'cost_eff': '#F0CFEA',     # 新浅粉色（Cost-Effectiveness）
        'dual_1': '#9BDCFC',       # 双色方案1（复用）
        'dual_2': '#F0CFEA'        # 双色方案2（复用）
    }
    
    # 子图1: Accuracy vs Feature Number
    ax1 = axes[0, 0]
    ax1.plot(df['n_features'], df['mean_accuracy'], 'o-', 
             color=colors['accuracy'], linewidth=2, markersize=5, alpha=0.9)
    ax1.fill_between(df['n_features'], 
                     df['mean_accuracy'] - df['std_accuracy'],
                     df['mean_accuracy'] + df['std_accuracy'],
                     alpha=0.25, color=colors['accuracy'])
    ax1.set_title('a', fontweight='bold', fontsize=24, pad=20, loc='left')
    ax1.text(0.02, 0.95, 'Accuracy vs. Number of Features', transform=ax1.transAxes, 
             fontsize=11, fontweight='normal', va='top')
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('Mean Accuracy')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0.60, 0.95])
    
    # 添加最佳性能点标注
    best_acc_idx = df['mean_accuracy'].idxmax()
    best_acc_features = df.iloc[best_acc_idx]['n_features']
    best_acc_value = df.iloc[best_acc_idx]['mean_accuracy']
    ax1.annotate(f'Peak: {best_acc_features} features',
                xy=(best_acc_features, best_acc_value),
                xytext=(best_acc_features + 8, best_acc_value - 0.02),
                arrowprops=dict(arrowstyle='->', color='#333333', alpha=0.8, lw=1),
                fontsize=9, ha='left',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))
    
    # 子图2: AUC vs Feature Number
    ax2 = axes[0, 1]
    ax2.plot(df['n_features'], df['mean_auc'], 'o-', 
             color=colors['auc'], linewidth=2, markersize=5, alpha=0.9)
    ax2.fill_between(df['n_features'], 
                     df['mean_auc'] - df['std_auc'],
                     df['mean_auc'] + df['std_auc'],
                     alpha=0.25, color=colors['auc'])
    ax2.set_title('b', fontweight='bold', fontsize=24, pad=20, loc='left')
    ax2.text(0.02, 0.95, 'AUC vs. Number of Features', transform=ax2.transAxes, 
             fontsize=11, fontweight='normal', va='top')
    ax2.set_xlabel('Number of Features')
    ax2.set_ylabel('Mean AUC')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0.60, 0.95])
    
    # 添加最佳AUC点标注
    best_auc_idx = df['mean_auc'].idxmax()
    best_auc_features = df.iloc[best_auc_idx]['n_features']
    best_auc_value = df.iloc[best_auc_idx]['mean_auc']
    ax2.annotate(f'Peak: {best_auc_features} features',
                xy=(best_auc_features, best_auc_value),
                xytext=(best_auc_features + 8, best_auc_value - 0.02),
                arrowprops=dict(arrowstyle='->', color='#333333', alpha=0.8, lw=1),
                fontsize=9, ha='left',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))
    
    # 子图3: F1 Score vs Feature Number
    ax3 = axes[1, 0]
    ax3.plot(df['n_features'], df['mean_f1'], 'o-', 
             color=colors['f1'], linewidth=2, markersize=5, alpha=0.9)
    ax3.fill_between(df['n_features'], 
                     df['mean_f1'] - df['std_f1'],
                     df['mean_f1'] + df['std_f1'],
                     alpha=0.25, color=colors['f1'])
    ax3.set_title('c', fontweight='bold', fontsize=24, pad=20, loc='left')
    ax3.text(0.02, 0.95, 'F1-Score vs. Number of Features', transform=ax3.transAxes, 
             fontsize=11, fontweight='normal', va='top')
    ax3.set_xlabel('Number of Features')
    ax3.set_ylabel('Mean F1-Score')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim([0.60, 0.95])
    
    # 添加最佳F1点标注
    best_f1_idx = df['mean_f1'].idxmax()
    best_f1_features = df.iloc[best_f1_idx]['n_features']
    best_f1_value = df.iloc[best_f1_idx]['mean_f1']
    ax3.annotate(f'Peak: {best_f1_features} features',
                xy=(best_f1_features, best_f1_value),
                xytext=(best_f1_features + 8, best_f1_value - 0.02),
                arrowprops=dict(arrowstyle='->', color='#333333', alpha=0.8, lw=1),
                fontsize=9, ha='left',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))
    
    # 子图4: 综合性价比指标
    ax4 = axes[1, 1]
    
    # 计算综合性价比指标
    # 1. 性能得分 (0-1, 越高越好)
    performance_score = (df['mean_auc'] * 0.5 + df['mean_accuracy'] * 0.3 + df['mean_f1'] * 0.2)
    
    # 2. 效率得分 (0-1, 训练时间越短越好)
    max_time = df['mean_time'].max()
    min_time = df['mean_time'].min()
    efficiency_score = 1 - (df['mean_time'] - min_time) / (max_time - min_time)
    
    # 3. 稳定性得分 (0-1, 标准差越小越好) 
    avg_std = (df['std_auc'] + df['std_accuracy'] + df['std_f1']) / 3
    max_std = avg_std.max()
    min_std = avg_std.min()
    stability_score = 1 - (avg_std - min_std) / (max_std - min_std) if max_std > min_std else 1
    
    # 4. 简洁性得分 (0-1, 符合奥卡姆剃刀原则，越简单越好)
    # 使用纯数学的指数衰减函数，体现"越简单越好"的奥卡姆剃刀原理
    alpha = 0.015  # 复杂度惩罚系数
    simplicity_score = np.exp(-alpha * df['n_features'])
    
    # 综合评价指标 (理论驱动的权重分配，性能优先，简洁度适度增强)
    # 优化权重: Performance=0.45, Efficiency=0.15, Stability=0.15, Simplicity=0.25
    cost_effectiveness = (performance_score * 0.45 + 
                         efficiency_score * 0.15 + 
                         stability_score * 0.15 + 
                         simplicity_score * 0.25)
    
    # 绘制综合指标
    ax4.plot(df['n_features'], cost_effectiveness, 'o-', 
             color=colors['cost_eff'], linewidth=2.5, markersize=6, alpha=0.9)
    ax4.fill_between(df['n_features'], 
                     cost_effectiveness - cost_effectiveness.std()*0.1,
                     cost_effectiveness + cost_effectiveness.std()*0.1,
                     alpha=0.2, color=colors['cost_eff'])
    
    # 为d图添加9-13特征重要区间背景
    ax4.axvspan(9, 13, alpha=0.25, color=colors['cost_eff'], 
                label='Key Range (9-13 features)', zorder=0)
    
    ax4.set_title('d', fontweight='bold', fontsize=24, pad=20, loc='left')
    ax4.text(0.98, 0.95, 'Cost-Effectiveness Index', transform=ax4.transAxes, 
             fontsize=11, fontweight='normal', va='top', ha='right')
    ax4.set_xlabel('Number of Features')
    ax4.set_ylabel('Comprehensive Score')
    ax4.grid(True, alpha=0.3, linestyle='--')
    # 调整y轴范围以更好地显示标准差
    y_min = min(cost_effectiveness.min() - cost_effectiveness.std()*0.15, 
                cost_effectiveness.min() - 0.1)
    y_max = min(1.0, cost_effectiveness.max() + cost_effectiveness.std()*0.15)
    ax4.set_ylim([y_min, y_max])
    
    # 添加最佳性价比点标注
    best_ce_idx = cost_effectiveness.idxmax()
    best_ce_features = df.iloc[best_ce_idx]['n_features']
    best_ce_value = cost_effectiveness.iloc[best_ce_idx]
    ax4.annotate(f'Optimal: {best_ce_features} features',
                xy=(best_ce_features, best_ce_value),
                xytext=(best_ce_features + 8, best_ce_value - 0.05),
                arrowprops=dict(arrowstyle='->', color='#333333', alpha=0.8, lw=1),
                fontsize=9, ha='left',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, 
                         edgecolor='gray', linewidth=0.5))
    
    # 添加图例说明综合指标的构成
    legend_text = ('Performance×0.45 + Simplicity×0.25\n'
                   '+ Efficiency×0.15 + Stability×0.15')
    ax4.text(0.02, 0.02, legend_text, transform=ax4.transAxes, 
             fontsize=8, ha='left', va='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', 
                      alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    # 调整子图间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # 保存为PDF
    print(f"💾 保存PDF图表到: {output_path}")
    plt.savefig(output_path, format='pdf', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                metadata={'Title': 'Feature Number Performance Analysis',
                         'Author': 'TabPFN+TCA Medical Research Team',
                         'Subject': 'Cross-Domain Medical Classification',
                         'Creator': 'Python matplotlib'})
    
    # 同时保存高分辨率PNG用于预览
    png_path = output_path.with_suffix('.png')
    print(f"💾 保存PNG预览图到: {png_path}")
    plt.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.show()
    
    # 返回最佳性能统计
    best_performance = {
        'best_accuracy': {'features': best_acc_features, 'value': best_acc_value},
        'best_auc': {'features': best_auc_features, 'value': best_auc_value},
        'best_f1': {'features': best_f1_features, 'value': best_f1_value}
    }
    
    return best_performance

def create_comprehensive_analysis_plot(df, output_path, dpi=1200):
    """
    创建综合分析图表，包含更多细节分析
    """
    print(f"🎨 生成综合分析图表 (DPI: {dpi})")
    
    # 创建更大的图形用于综合分析 - 调整为更高的图形以适应新布局
    fig = plt.figure(figsize=(16, 20))
    
    # 使用GridSpec进行复杂布局 - 修改为5行2列的布局
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(5, 2, figure=fig, height_ratios=[2, 1, 1, 1, 1], width_ratios=[1, 1])
    
    # 定义Nature期刊标准配色 - 四色方案
    colors = {
        'accuracy': '#9BDCFC',     # 新浅蓝色
        'auc': '#C9EFBE',          # 新浅绿色
        'f1': '#CAC8EF',           # 新浅紫色
        'cost_eff': '#F0CFEA',     # 新浅粉色（Cost-Effectiveness）
        'dual_1': '#9BDCFC',       # 双色方案1（复用）
        'dual_2': '#F0CFEA'        # 双色方案2（复用）
    }
    
    # 删除主标题以符合投稿要求
    # fig.suptitle('Comprehensive Feature Number Analysis for Medical Cross-Domain Classification\n(TabPFN+TCA Framework)', 
    #              fontsize=18, fontweight='bold', y=0.95)
    
    # 主要性能指标趋势图 (占据第一行的整个宽度)
    ax_main = fig.add_subplot(gs[0, :])
    
    # 绘制多条性能曲线
    metrics = ['mean_accuracy', 'mean_auc', 'mean_f1']
    metric_labels = ['Accuracy', 'AUC', 'F1-Score']
    metric_colors = [colors['accuracy'], colors['auc'], colors['f1']]
    
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
        ax_main.plot(df['n_features'], df[metric], 'o-', 
                     color=color, linewidth=2.5, markersize=7, 
                     label=label, alpha=0.9)
        
        # 添加误差带
        std_metric = f'std_{metric.split("_")[1]}'
        ax_main.fill_between(df['n_features'], 
                            df[metric] - df[std_metric],
                            df[metric] + df[std_metric],
                            alpha=0.15, color=color)
    
    ax_main.set_title('a', fontweight='bold', fontsize=24, pad=20, loc='left')
    ax_main.text(0.02, 0.95, 'Performance Metrics vs. Number of Features', transform=ax_main.transAxes, 
                 fontsize=11, fontweight='normal', va='top')
    ax_main.set_xlabel('Number of Features', fontsize=12)
    ax_main.set_ylabel('Performance Score', fontsize=12)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.legend(loc='lower right', fontsize=11)
    ax_main.set_ylim([0.60, 0.95])
    
    # 添加9-13特征重要区间的背景色标识
    ax_main.axvspan(9, 13, alpha=0.25, color=colors['cost_eff'], 
                    label='Key Range (9-13 features)', zorder=0)
    
    # 添加峰值点标注
    # 找到每个指标的峰值
    best_acc_idx = df['mean_accuracy'].idxmax()
    best_auc_idx = df['mean_auc'].idxmax()  
    best_f1_idx = df['mean_f1'].idxmax()
    
    best_acc_features = df.iloc[best_acc_idx]['n_features']
    best_auc_features = df.iloc[best_auc_idx]['n_features']
    best_f1_features = df.iloc[best_f1_idx]['n_features']
    
    best_acc_value = df.iloc[best_acc_idx]['mean_accuracy']
    best_auc_value = df.iloc[best_auc_idx]['mean_auc']
    best_f1_value = df.iloc[best_f1_idx]['mean_f1']
    
    # 标注峰值点 - 重新设计避免重叠
    # Acc Peak: 13 - 从右侧指向峰值
    ax_main.annotate(f'Acc Peak: {best_acc_features}',
                    xy=(best_acc_features, best_acc_value),
                    xytext=(best_acc_features + 10, best_acc_value),
                    arrowprops=dict(arrowstyle='->', color=colors['accuracy'], alpha=0.8, lw=1.5),
                    fontsize=8, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, 
                             edgecolor=colors['accuracy'], linewidth=0.5))
    
    # AUC Peak: 15 - 从右侧指向峰值
    ax_main.annotate(f'AUC Peak: {best_auc_features}',
                    xy=(best_auc_features, best_auc_value),
                    xytext=(best_auc_features + 8, best_auc_value),
                    arrowprops=dict(arrowstyle='->', color=colors['auc'], alpha=0.8, lw=1.5),
                    fontsize=8, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, 
                             edgecolor=colors['auc'], linewidth=0.5))
    
    # F1 Peak: 13 - 从右侧指向峰值，避开Acc标注
    ax_main.annotate(f'F1 Peak: {best_f1_features}',
                    xy=(best_f1_features, best_f1_value),
                    xytext=(best_f1_features + 12, best_f1_value),
                    arrowprops=dict(arrowstyle='->', color=colors['f1'], alpha=0.8, lw=1.5),
                    fontsize=8, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, 
                             edgecolor=colors['f1'], linewidth=0.5))
    
    # 更新图例包含重要区间
    ax_main.legend(loc='lower right', fontsize=10)
    
    # 类别特异性性能 (第二行左侧)
    ax_class = fig.add_subplot(gs[1, 0])
    ax_class.plot(df['n_features'], df['mean_acc_0'], 'o-', 
                  color=colors['dual_1'], linewidth=2, markersize=5, 
                  label='Class 0 (Benign)', alpha=0.8)
    ax_class.plot(df['n_features'], df['mean_acc_1'], 'o-', 
                  color=colors['dual_2'], linewidth=2, markersize=5, 
                  label='Class 1 (Malignant)', alpha=0.8)
    ax_class.set_title('b', fontweight='bold', fontsize=24, pad=20, loc='left')
    ax_class.text(0.98, 0.02, 'Class-Specific Accuracy', transform=ax_class.transAxes, 
                  fontsize=11, fontweight='normal', va='bottom', ha='right')
    ax_class.set_xlabel('Features')
    ax_class.set_ylabel('Accuracy')
    ax_class.grid(True, alpha=0.3)
    ax_class.legend(fontsize=9)
    
    # 为comprehensive版b图添加9-13特征重要区间背景
    ax_class.axvspan(9, 13, alpha=0.25, color=colors['cost_eff'], 
                    label='Key Range (9-13 features)', zorder=0)
    # 更新图例包含重要区间
    ax_class.legend(fontsize=9)
    
    # 训练时间分析 (第二行右侧)
    ax_time = fig.add_subplot(gs[1, 1])
    ax_time.plot(df['n_features'], df['mean_time'], 'o-', 
                 color='#666666', linewidth=2, markersize=5, alpha=0.8)
    ax_time.fill_between(df['n_features'], 
                        df['mean_time'] - df['std_time'],
                        df['mean_time'] + df['std_time'],
                        alpha=0.2, color='#666666')
    ax_time.set_title('c', fontweight='bold', fontsize=24, pad=20, loc='left')
    ax_time.text(0.02, 0.95, 'Training Time Complexity', transform=ax_time.transAxes, 
                 fontsize=11, fontweight='normal', va='top')
    ax_time.set_xlabel('Features')
    ax_time.set_ylabel('Time (s)')
    ax_time.grid(True, alpha=0.3)
    
    # 性能稳定性分析 (第三行整行)
    ax_stability = fig.add_subplot(gs[2, :])
    
    # 计算变异系数 (CV = std/mean)
    cv_accuracy = df['std_accuracy'] / df['mean_accuracy']
    cv_auc = df['std_auc'] / df['mean_auc']
    cv_f1 = df['std_f1'] / df['mean_f1']
    
    ax_stability.plot(df['n_features'], cv_accuracy, 'o-', 
                     color=colors['accuracy'], linewidth=2, markersize=5, 
                     label='Accuracy CV', alpha=0.8)
    ax_stability.plot(df['n_features'], cv_auc, 'o-', 
                     color=colors['auc'], linewidth=2, markersize=5, 
                     label='AUC CV', alpha=0.8)
    ax_stability.plot(df['n_features'], cv_f1, 'o-', 
                     color=colors['f1'], linewidth=2, markersize=5, 
                     label='F1 CV', alpha=0.8)
    
    ax_stability.set_title('d', fontweight='bold', fontsize=24, pad=20, loc='left')
    ax_stability.text(0.98, 0.02, 'Performance Stability (Coefficient of Variation)', transform=ax_stability.transAxes, 
                      fontsize=11, fontweight='normal', va='bottom', ha='right')
    ax_stability.set_xlabel('Number of Features')
    ax_stability.set_ylabel('CV (std/mean)')
    ax_stability.grid(True, alpha=0.3)
    ax_stability.legend(fontsize=10)
    
    # 为comprehensive版d图添加9-13特征重要区间背景
    ax_stability.axvspan(9, 13, alpha=0.25, color=colors['cost_eff'], 
                        label='Key Range (9-13 features)', zorder=0)
    # 更新图例包含重要区间
    ax_stability.legend(fontsize=10)
    
    # Cost-Effectiveness Index 分析 (第四行整行)
    ax_optimal = fig.add_subplot(gs[3, :])
    
    # 计算Cost-Effectiveness Index (与标准版本相同的算法)
    # 1. 性能得分 (0-1, 越高越好)
    performance_score = (df['mean_auc'] * 0.5 + df['mean_accuracy'] * 0.3 + df['mean_f1'] * 0.2)
    
    # 2. 效率得分 (0-1, 训练时间越短越好)
    max_time = df['mean_time'].max()
    min_time = df['mean_time'].min()
    efficiency_score = 1 - (df['mean_time'] - min_time) / (max_time - min_time)
    
    # 3. 稳定性得分 (0-1, 标准差越小越好) 
    avg_std = (df['std_auc'] + df['std_accuracy'] + df['std_f1']) / 3
    max_std = avg_std.max()
    min_std = avg_std.min()
    stability_score = 1 - (avg_std - min_std) / (max_std - min_std) if max_std > min_std else 1
    
    # 4. 简洁性得分 (0-1, 符合奥卡姆剃刀原则，越简单越好)
    # 使用纯数学的指数衰减函数，体现"越简单越好"的奥卡姆剃刀原理
    alpha = 0.015  # 复杂度惩罚系数
    simplicity_score = np.exp(-alpha * df['n_features'])
    
    # 综合评价指标 (理论驱动的权重分配，性能优先，简洁度适度增强)
    # 优化权重: Performance=0.45, Efficiency=0.15, Stability=0.15, Simplicity=0.25
    cost_effectiveness = (performance_score * 0.45 + 
                         efficiency_score * 0.15 + 
                         stability_score * 0.15 + 
                         simplicity_score * 0.25)
    
    # 绘制综合指标
    ax_optimal.plot(df['n_features'], cost_effectiveness, 'o-', 
                   color='#F5A889', linewidth=2.5, markersize=6, alpha=0.9)
    ax_optimal.fill_between(df['n_features'], 
                           cost_effectiveness - cost_effectiveness.std()*0.1,
                           cost_effectiveness + cost_effectiveness.std()*0.1,
                           alpha=0.2, color='#F5A889')
    
    ax_optimal.set_title('e', fontweight='bold', fontsize=24, pad=20, loc='left')
    ax_optimal.text(0.98, 0.95, 'Cost-Effectiveness Index', transform=ax_optimal.transAxes, 
                    fontsize=11, fontweight='normal', va='top', ha='right')
    ax_optimal.set_xlabel('Number of Features')
    ax_optimal.set_ylabel('Comprehensive Score')
    ax_optimal.grid(True, alpha=0.3, linestyle='--')
    ax_optimal.set_ylim([cost_effectiveness.min() - 0.05, 
                        min(1.0, cost_effectiveness.max() + 0.05)])
    
    # 添加最佳性价比点标注
    best_ce_idx = cost_effectiveness.idxmax()
    best_ce_features = df.iloc[best_ce_idx]['n_features']
    best_ce_value = cost_effectiveness.iloc[best_ce_idx]
    ax_optimal.annotate(f'Optimal: {best_ce_features} features',
                       xy=(best_ce_features, best_ce_value),
                       xytext=(best_ce_features + 8, best_ce_value - 0.05),
                       arrowprops=dict(arrowstyle='->', color='#333333', alpha=0.8, lw=1),
                       fontsize=9, ha='left',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, 
                                edgecolor='gray', linewidth=0.5))
    
    # 添加综合指标构成说明
    formula_text = ('Performance×0.45 + Simplicity×0.25\n'
                   '+ Efficiency×0.15 + Stability×0.15')
    ax_optimal.text(0.02, 0.02, formula_text, transform=ax_optimal.transAxes, 
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', 
                             alpha=0.8, edgecolor='gray', linewidth=0.5))
    
    # 调整布局 - 为新的5行布局增加垂直间距
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.3, wspace=0.2)
    
    # 保存综合分析图
    comprehensive_path = output_path.parent / f"{output_path.stem}_comprehensive.pdf"
    print(f"💾 保存综合分析PDF到: {comprehensive_path}")
    plt.savefig(comprehensive_path, format='pdf', dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # 保存PNG预览
    png_path = comprehensive_path.with_suffix('.png')
    plt.savefig(png_path, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.show()
    
    return {
        'best_cost_effectiveness': {'features': best_ce_features, 'value': best_ce_value}
    }

def print_summary_statistics(df, best_performance):
    """打印总结统计信息"""
    print(f"\n📊 特征数量性能分析总结")
    print("=" * 50)
    
    print(f"📈 最佳性能指标:")
    print(f"  最佳准确率: {best_performance['best_accuracy']['value']:.3f} (特征数: {best_performance['best_accuracy']['features']})")
    print(f"  最佳AUC: {best_performance['best_auc']['value']:.3f} (特征数: {best_performance['best_auc']['features']})")
    print(f"  最佳F1分数: {best_performance['best_f1']['value']:.3f} (特征数: {best_performance['best_f1']['features']})")
    
    print(f"\n⚡ 效率分析:")
    min_time_idx = df['mean_time'].idxmin()
    min_time_features = df.iloc[min_time_idx]['n_features']
    min_time_value = df.iloc[min_time_idx]['mean_time']
    print(f"  最快训练时间: {min_time_value:.2f}s (特征数: {min_time_features})")
    
    max_time_idx = df['mean_time'].idxmax()
    max_time_features = df.iloc[max_time_idx]['n_features']
    max_time_value = df.iloc[max_time_idx]['mean_time']
    print(f"  最慢训练时间: {max_time_value:.2f}s (特征数: {max_time_features})")
    
    print(f"\n🎯 推荐配置:")
    # 找到平衡点：性能好且效率高
    df['efficiency_score'] = (df['mean_auc'] * 0.6 + df['mean_f1'] * 0.4) / (df['mean_time'] / df['mean_time'].min())
    best_efficiency_idx = df['efficiency_score'].idxmax()
    best_efficiency_features = df.iloc[best_efficiency_idx]['n_features']
    best_efficiency_auc = df.iloc[best_efficiency_idx]['mean_auc']
    best_efficiency_time = df.iloc[best_efficiency_idx]['mean_time']
    
    print(f"  推荐特征数: {best_efficiency_features} (平衡性能与效率)")
    print(f"  对应AUC: {best_efficiency_auc:.3f}")
    print(f"  对应训练时间: {best_efficiency_time:.2f}s")

def main():
    """主函数"""
    print("🎯 特征数量性能分析 - 学术论文级PDF生成器")
    print("=" * 60)
    
    # 文件路径配置
    csv_path = Path("/Users/lqy/work/TabPFN/uda_medical_imbalance_project/results/feature_number_evaluation/feature_number_comparison.csv")
    output_path = Path("/Users/lqy/work/TabPFN/uda_medical_imbalance_project/results/feature_number_evaluation/performance_comparison.pdf")
    
    # 检查输入文件
    if not csv_path.exists():
        print(f"❌ 输入文件不存在: {csv_path}")
        return
    
    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 加载数据
        df = load_and_analyze_data(csv_path)
        
        # 2. 生成标准性能对比图 (DPI: 1200)
        print(f"\n🎨 生成标准性能对比图...")
        best_performance = create_performance_comparison_plot(df, output_path, dpi=1200)
        
        # 3. 生成综合分析图 (DPI: 900, 更复杂的图表使用稍低DPI以平衡文件大小)
        print(f"\n🎨 生成综合分析图...")
        analysis_results = create_comprehensive_analysis_plot(df, output_path, dpi=900)
        
        # 4. 打印总结统计
        print_summary_statistics(df, best_performance)
        
        print(f"\n✅ PDF图表生成完成!")
        print(f"📁 输出文件:")
        print(f"  标准版: {output_path}")
        print(f"  综合版: {output_path.parent / f'{output_path.stem}_comprehensive.pdf'}")
        print(f"  PNG预览: {output_path.with_suffix('.png')}")
        
        print(f"\n💡 使用说明:")
        print(f"  - 标准版适合论文正文插入")
        print(f"  - 综合版适合补充材料或详细分析")
        print(f"  - 所有图表DPI≥900，符合学术期刊要求")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()