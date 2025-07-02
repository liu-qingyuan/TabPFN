"""
全面测试不平衡处理器功能

强制执行所有不平衡处理方法并生成完整的可视化对比
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from data.loader import MedicalDataLoader
from preprocessing.imbalance_handler import ImbalanceHandler
from config.settings import get_categorical_indices
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def force_test_all_methods():
    """强制测试所有不平衡处理方法"""
    print("=" * 80)
    print("强制测试所有不平衡处理方法")
    print("=" * 80)
    
    # 加载数据
    loader = MedicalDataLoader()
    data_a = loader.load_dataset('A', feature_type='best10')
    
    X_a = data_a['X']
    y_a = data_a['y']
    
    print(f"原始数据形状: {X_a.shape}")
    print(f"原始类别分布: {Counter(y_a)}")
    
    # 定义所有要测试的方法 - 自动使用SMOTENC处理类别特征
    methods_to_test = {
        'none': {},
        'smote': {'k_neighbors': 5, 'feature_type': 'best10'},
        'smotenc': {'k_neighbors': 5, 'feature_type': 'best10'},
        'borderline_smote': {'k_neighbors': 5, 'feature_type': 'best10'},
        'kmeans_smote': {'k_neighbors': 5, 'feature_type': 'best10'},
        'svm_smote': {'k_neighbors': 5, 'feature_type': 'best10'},
        'adasyn': {'n_neighbors': 5, 'feature_type': 'best10'},
        'smote_tomek': {
            'smote_config': {'k_neighbors': 5},
            'tomek_config': {'sampling_strategy': 'all'},
            'feature_type': 'best10'
        },
        'smote_enn': {
            'smote_config': {'k_neighbors': 5},
            'enn_config': {'sampling_strategy': 'all', 'n_neighbors': 3},
            'feature_type': 'best10'
        },
        'random_under': {'sampling_strategy': 'auto'}
    }
    
    results = {}
    
    for method_name, params in methods_to_test.items():
        try:
            print(f"\n强制测试方法: {method_name}")
            
            # 创建处理器
            handler = ImbalanceHandler(
                method=method_name,
                random_state=42,
                **params
            )
            
            # 强制执行重采样（跳过平衡检查）
            if method_name == 'none':
                X_resampled, y_resampled = X_a.copy(), y_a.copy()
            else:
                # 直接调用采样器进行重采样
                if handler.sampler is not None:
                    X_resampled, y_resampled = handler.sampler.fit_resample(X_a, y_a)
                else:
                    X_resampled, y_resampled = X_a.copy(), y_a.copy()
            
            distribution = Counter(y_resampled)
            results[method_name] = {
                'original_shape': X_a.shape,
                'resampled_shape': X_resampled.shape,
                'original_distribution': Counter(y_a),
                'resampled_distribution': distribution,
                'total_samples': len(y_resampled),
                'X_original': X_a,
                'y_original': y_a,
                'X_resampled': X_resampled,
                'y_resampled': y_resampled
            }
            
            print(f"  原始形状: {X_a.shape}")
            print(f"  重采样后形状: {X_resampled.shape}")
            print(f"  原始分布: {Counter(y_a)}")
            print(f"  重采样后分布: {distribution}")
            
        except Exception as e:
            print(f"  {method_name} 失败: {e}")
            results[method_name] = {'error': str(e)}
    
    return results

def create_dimensionality_reduction_visualization(results):
    """创建PCA和t-SNE降维可视化"""
    print("\n" + "=" * 80)
    print("生成PCA和t-SNE降维可视化")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = "tests/imgs/imbalance_handler"
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取有效结果
    valid_results = {}
    for method, result in results.items():
        if 'error' not in result and 'X_resampled' in result:
            valid_results[method] = result
    
    if not valid_results:
        print("没有有效的结果用于降维可视化")
        return
    
    # 可视化所有有效的方法
    methods_to_visualize = list(valid_results.keys())
    
    print(f"将可视化以下方法: {methods_to_visualize}")
    
    # 为每种方法创建PCA和t-SNE可视化
    for method in methods_to_visualize:
        result = valid_results[method]
        X_original = result['X_original']
        y_original = result['y_original']
        X_resampled = result['X_resampled']
        y_resampled = result['y_resampled']
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Dimensionality Reduction Visualization: {method.upper()}', fontsize=16)
        
        # 标准化数据（对于t-SNE很重要）
        scaler_orig = StandardScaler()
        X_orig_scaled = scaler_orig.fit_transform(X_original)
        
        scaler_resamp = StandardScaler()
        X_resamp_scaled = scaler_resamp.fit_transform(X_resampled)
        
        # PCA - 原始数据
        pca_orig = PCA(n_components=2, random_state=42)
        X_orig_pca = pca_orig.fit_transform(X_orig_scaled)
        
        axes[0, 0].scatter(X_orig_pca[y_original == 0, 0], X_orig_pca[y_original == 0, 1], 
                          c='lightblue', alpha=0.7, label='Class 0', s=30)
        axes[0, 0].scatter(X_orig_pca[y_original == 1, 0], X_orig_pca[y_original == 1, 1], 
                          c='lightcoral', alpha=0.7, label='Class 1', s=30)
        axes[0, 0].set_title(f'PCA - Original Data\n({len(y_original)} samples)')
        axes[0, 0].set_xlabel(f'PC1 ({pca_orig.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 0].set_ylabel(f'PC2 ({pca_orig.explained_variance_ratio_[1]:.2%} variance)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PCA - 重采样后数据
        pca_resamp = PCA(n_components=2, random_state=42)
        X_resamp_pca = pca_resamp.fit_transform(X_resamp_scaled)
        
        axes[0, 1].scatter(X_resamp_pca[y_resampled == 0, 0], X_resamp_pca[y_resampled == 0, 1], 
                          c='lightblue', alpha=0.7, label='Class 0', s=30)
        axes[0, 1].scatter(X_resamp_pca[y_resampled == 1, 0], X_resamp_pca[y_resampled == 1, 1], 
                          c='lightcoral', alpha=0.7, label='Class 1', s=30)
        axes[0, 1].set_title(f'PCA - Resampled Data\n({len(y_resampled)} samples)')
        axes[0, 1].set_xlabel(f'PC1 ({pca_resamp.explained_variance_ratio_[0]:.2%} variance)')
        axes[0, 1].set_ylabel(f'PC2 ({pca_resamp.explained_variance_ratio_[1]:.2%} variance)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # t-SNE - 原始数据
        if len(X_original) > 5:  # t-SNE需要足够的样本
            tsne_orig = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_original)//4))
            X_orig_tsne = tsne_orig.fit_transform(X_orig_scaled)
            
            axes[1, 0].scatter(X_orig_tsne[y_original == 0, 0], X_orig_tsne[y_original == 0, 1], 
                              c='lightblue', alpha=0.7, label='Class 0', s=30)
            axes[1, 0].scatter(X_orig_tsne[y_original == 1, 0], X_orig_tsne[y_original == 1, 1], 
                              c='lightcoral', alpha=0.7, label='Class 1', s=30)
            axes[1, 0].set_title(f't-SNE - Original Data\n({len(y_original)} samples)')
            axes[1, 0].set_xlabel('t-SNE Component 1')
            axes[1, 0].set_ylabel('t-SNE Component 2')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Not enough samples\nfor t-SNE', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('t-SNE - Original Data')
        
        # t-SNE - 重采样后数据
        if len(X_resampled) > 5:  # t-SNE需要足够的样本
            tsne_resamp = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_resampled)//4))
            X_resamp_tsne = tsne_resamp.fit_transform(X_resamp_scaled)
            
            axes[1, 1].scatter(X_resamp_tsne[y_resampled == 0, 0], X_resamp_tsne[y_resampled == 0, 1], 
                              c='lightblue', alpha=0.7, label='Class 0', s=30)
            axes[1, 1].scatter(X_resamp_tsne[y_resampled == 1, 0], X_resamp_tsne[y_resampled == 1, 1], 
                              c='lightcoral', alpha=0.7, label='Class 1', s=30)
            axes[1, 1].set_title(f't-SNE - Resampled Data\n({len(y_resampled)} samples)')
            axes[1, 1].set_xlabel('t-SNE Component 1')
            axes[1, 1].set_ylabel('t-SNE Component 2')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Not enough samples\nfor t-SNE', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('t-SNE - Resampled Data')
        
        plt.tight_layout()
        
        # 保存图片
        filename = f"dimensionality_reduction_{method}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"保存降维可视化: {filepath}")
        plt.close()
    
    # 创建所有方法的PCA对比图
    create_pca_comparison(valid_results, output_dir)

def create_pca_comparison(valid_results, output_dir):
    """创建所有方法的PCA对比图"""
    print("\n创建PCA对比图...")
    
    methods = list(valid_results.keys())
    n_methods = len(methods)
    
    if n_methods == 0:
        return
    
    # 计算子图布局
    n_cols = min(4, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    fig.suptitle('PCA Comparison: Before vs After Resampling', fontsize=16)
    
    for i, method in enumerate(methods):
        result = valid_results[method]
        X_original = result['X_original']
        y_original = result['y_original']
        X_resampled = result['X_resampled']
        y_resampled = result['y_resampled']
        
        # 标准化数据
        scaler = StandardScaler()
        X_orig_scaled = scaler.fit_transform(X_original)
        X_resamp_scaled = scaler.fit_transform(X_resampled)
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_orig_pca = pca.fit_transform(X_orig_scaled)
        X_resamp_pca = pca.fit_transform(X_resamp_scaled)
        
        ax = axes[i]
        
        # 绘制原始数据（较小的点，透明度高）
        ax.scatter(X_orig_pca[y_original == 0, 0], X_orig_pca[y_original == 0, 1], 
                  c='blue', alpha=0.3, label='Original Class 0', s=20, marker='o')
        ax.scatter(X_orig_pca[y_original == 1, 0], X_orig_pca[y_original == 1, 1], 
                  c='red', alpha=0.3, label='Original Class 1', s=20, marker='o')
        
        # 绘制重采样后数据（较大的点，透明度低）
        ax.scatter(X_resamp_pca[y_resampled == 0, 0], X_resamp_pca[y_resampled == 0, 1], 
                  c='lightblue', alpha=0.7, label='Resampled Class 0', s=30, marker='s')
        ax.scatter(X_resamp_pca[y_resampled == 1, 0], X_resamp_pca[y_resampled == 1, 1], 
                  c='lightcoral', alpha=0.7, label='Resampled Class 1', s=30, marker='s')
        
        ax.set_title(f'{method.upper()}\nOrig: {len(y_original)}, Resamp: {len(y_resampled)}')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.grid(True, alpha=0.3)
        
        if i == 0:  # 只在第一个子图显示图例
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 隐藏多余的子图
    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    filename = "pca_comparison_all_methods.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"保存PCA对比图: {filepath}")
    plt.close()

def create_comprehensive_visualization(results):
    """创建全面的可视化结果"""
    print("\n" + "=" * 80)
    print("生成全面可视化结果")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = "tests/imgs/imbalance_handler"
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取有效结果
    valid_results = {}
    for method, result in results.items():
        if 'error' not in result and 'resampled_distribution' in result:
            valid_results[method] = result
    
    if not valid_results:
        print("没有有效的结果用于可视化")
        return
    
    # 准备数据
    methods = list(valid_results.keys())
    original_counts = []
    resampled_counts = []
    class_0_original = []
    class_1_original = []
    class_0_resampled = []
    class_1_resampled = []
    balance_ratios_original = []
    balance_ratios_resampled = []
    
    for method in methods:
        result = valid_results[method]
        
        # 原始分布
        orig_dist = result['original_distribution']
        orig_total = sum(orig_dist.values())
        orig_c0 = orig_dist.get(0, 0)
        orig_c1 = orig_dist.get(1, 0)
        
        # 重采样后分布
        resamp_dist = result['resampled_distribution']
        resamp_total = sum(resamp_dist.values())
        resamp_c0 = resamp_dist.get(0, 0)
        resamp_c1 = resamp_dist.get(1, 0)
        
        original_counts.append(orig_total)
        resampled_counts.append(resamp_total)
        class_0_original.append(orig_c0)
        class_1_original.append(orig_c1)
        class_0_resampled.append(resamp_c0)
        class_1_resampled.append(resamp_c1)
        
        # 计算平衡比例
        if orig_c0 > 0 and orig_c1 > 0:
            balance_ratios_original.append(min(orig_c0, orig_c1) / max(orig_c0, orig_c1))
        else:
            balance_ratios_original.append(0)
            
        if resamp_c0 > 0 and resamp_c1 > 0:
            balance_ratios_resampled.append(min(resamp_c0, resamp_c1) / max(resamp_c0, resamp_c1))
        else:
            balance_ratios_resampled.append(0)
    
    # 创建综合对比图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 子图1: 总样本数对比（原始 vs 重采样）
    x = np.arange(len(methods))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, original_counts, width, label='Original', alpha=0.7, color='lightblue')
    axes[0, 0].bar(x + width/2, resampled_counts, width, label='Resampled', alpha=0.7, color='lightcoral')
    axes[0, 0].set_title('Total Samples: Original vs Resampled')
    axes[0, 0].set_xlabel('Resampling Method')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods, rotation=45)
    axes[0, 0].legend()
    
    # 子图2: 原始类别分布
    axes[0, 1].bar(x - width/2, class_0_original, width, label='Class 0', alpha=0.7, color='lightgreen')
    axes[0, 1].bar(x + width/2, class_1_original, width, label='Class 1', alpha=0.7, color='orange')
    axes[0, 1].set_title('Original Class Distribution')
    axes[0, 1].set_xlabel('Resampling Method')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(methods, rotation=45)
    axes[0, 1].legend()
    
    # 子图3: 重采样后类别分布
    axes[0, 2].bar(x - width/2, class_0_resampled, width, label='Class 0', alpha=0.7, color='lightgreen')
    axes[0, 2].bar(x + width/2, class_1_resampled, width, label='Class 1', alpha=0.7, color='orange')
    axes[0, 2].set_title('Resampled Class Distribution')
    axes[0, 2].set_xlabel('Resampling Method')
    axes[0, 2].set_ylabel('Number of Samples')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(methods, rotation=45)
    axes[0, 2].legend()
    
    # 子图4: 平衡比例对比
    axes[1, 0].bar(x - width/2, balance_ratios_original, width, label='Original', alpha=0.7, color='skyblue')
    axes[1, 0].bar(x + width/2, balance_ratios_resampled, width, label='Resampled', alpha=0.7, color='salmon')
    axes[1, 0].set_title('Class Balance Ratio: Original vs Resampled')
    axes[1, 0].set_xlabel('Resampling Method')
    axes[1, 0].set_ylabel('Balance Ratio (Min/Max)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods, rotation=45)
    axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Balance')
    axes[1, 0].legend()
    
    # 子图5: 采样效果（增加的样本数）
    sample_changes = [resamp - orig for orig, resamp in zip(original_counts, resampled_counts)]
    colors = ['green' if change >= 0 else 'red' for change in sample_changes]
    
    axes[1, 1].bar(methods, sample_changes, color=colors, alpha=0.7)
    axes[1, 1].set_title('Sample Count Change After Resampling')
    axes[1, 1].set_xlabel('Resampling Method')
    axes[1, 1].set_ylabel('Change in Sample Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 添加数值标签
    for i, change in enumerate(sample_changes):
        axes[1, 1].text(i, change + (5 if change >= 0 else -10), 
                       f'{change:+d}', ha='center', va='bottom' if change >= 0 else 'top')
    
    # 子图6: 方法效果评分（基于平衡改善程度）
    improvement_scores = []
    for i in range(len(methods)):
        if balance_ratios_original[i] > 0:
            improvement = (balance_ratios_resampled[i] - balance_ratios_original[i]) / balance_ratios_original[i]
        else:
            improvement = balance_ratios_resampled[i]
        improvement_scores.append(improvement)
    
    colors = plt.cm.RdYlGn([0.5 + 0.5 * min(1, max(-1, score)) for score in improvement_scores])
    bars = axes[1, 2].bar(methods, improvement_scores, color=colors, alpha=0.8)
    axes[1, 2].set_title('Balance Improvement Score')
    axes[1, 2].set_xlabel('Resampling Method')
    axes[1, 2].set_ylabel('Improvement Score')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 添加数值标签
    for i, score in enumerate(improvement_scores):
        axes[1, 2].text(i, score + (0.02 if score >= 0 else -0.05), 
                       f'{score:.2f}', ha='center', va='bottom' if score >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_resampling_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建详细统计表
    create_statistics_table(valid_results, output_dir)
    
    print(f"全面可视化结果已保存到:")
    print(f"  - {output_dir}/comprehensive_resampling_analysis.png")
    print(f"  - {output_dir}/resampling_statistics_table.png")

def create_statistics_table(valid_results, output_dir):
    """创建详细的统计表格"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    headers = [
        'Method', 'Original\nTotal', 'Original\nClass 0', 'Original\nClass 1', 
        'Original\nBalance', 'Resampled\nTotal', 'Resampled\nClass 0', 
        'Resampled\nClass 1', 'Resampled\nBalance', 'Sample\nChange', 'Balance\nImprovement'
    ]
    
    table_data = []
    for method, result in valid_results.items():
        orig_dist = result['original_distribution']
        resamp_dist = result['resampled_distribution']
        
        orig_total = sum(orig_dist.values())
        orig_c0 = orig_dist.get(0, 0)
        orig_c1 = orig_dist.get(1, 0)
        
        resamp_total = sum(resamp_dist.values())
        resamp_c0 = resamp_dist.get(0, 0)
        resamp_c1 = resamp_dist.get(1, 0)
        
        # 计算平衡比例
        orig_balance = min(orig_c0, orig_c1) / max(orig_c0, orig_c1) if orig_c0 > 0 and orig_c1 > 0 else 0
        resamp_balance = min(resamp_c0, resamp_c1) / max(resamp_c0, resamp_c1) if resamp_c0 > 0 and resamp_c1 > 0 else 0
        
        sample_change = resamp_total - orig_total
        balance_improvement = resamp_balance - orig_balance
        
        row = [
            method,
            orig_total,
            orig_c0,
            orig_c1,
            f'{orig_balance:.3f}',
            resamp_total,
            resamp_c0,
            resamp_c1,
            f'{resamp_balance:.3f}',
            f'{sample_change:+d}',
            f'{balance_improvement:+.3f}'
        ]
        table_data.append(row)
    
    # 创建表格
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # 设置表格样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 为不同类型的方法设置不同的背景色
    method_colors = {
        'none': '#FFEBEE',
        'smote': '#E3F2FD',
        'smotenc': '#E3F2FD',
        'borderline_smote': '#E3F2FD',
        'kmeans_smote': '#E8F5E8',
        'svm_smote': '#E8F5E8',
        'adasyn': '#FFF3E0',
        'smote_tomek': '#F3E5F5',
        'smote_enn': '#F3E5F5',
        'random_under': '#FFEBEE'
    }
    
    for i, (method, _) in enumerate(valid_results.items()):
        color = method_colors.get(method, '#FFFFFF')
        for j in range(len(headers)):
            table[(i + 1, j)].set_facecolor(color)
    
    ax.set_title('Detailed Resampling Statistics Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(f"{output_dir}/resampling_statistics_table.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """主测试函数"""
    print("开始全面测试不平衡处理器...")
    
    # 强制测试所有方法
    results = force_test_all_methods()
    
    # 生成降维可视化
    create_dimensionality_reduction_visualization(results)
    
    # 生成综合可视化
    create_comprehensive_visualization(results)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    successful_methods = []
    failed_methods = []
    
    for method, result in results.items():
        if 'error' in result:
            failed_methods.append(method)
        else:
            successful_methods.append(method)
    
    print(f"成功的方法 ({len(successful_methods)}): {successful_methods}")
    if failed_methods:
        print(f"失败的方法 ({len(failed_methods)}): {failed_methods}")
        for method in failed_methods:
            print(f"  {method}: {results[method]['error']}")
    
    print("\n全面测试完成！")

if __name__ == "__main__":
    main() 