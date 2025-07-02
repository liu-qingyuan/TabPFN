"""
测试不平衡处理器功能

测试各种不平衡处理方法在真实医疗数据上的效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from data.loader import MedicalDataLoader
from preprocessing.imbalance_handler import ImbalanceHandler, create_imbalance_handler
from config.settings import get_categorical_indices, get_features_by_type
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_methods():
    """测试基本的不平衡处理方法"""
    print("=" * 60)
    print("测试基本不平衡处理方法")
    print("=" * 60)
    
    # 加载数据
    loader = MedicalDataLoader()
    data_a = loader.load_dataset('A', feature_type='best7')
    
    X_a = data_a['X']
    y_a = data_a['y']
    
    print(f"原始数据形状: {X_a.shape}")
    print(f"原始类别分布: {Counter(y_a)}")
    
    # 测试基本方法
    basic_methods = ['none', 'smote', 'borderline_smote', 'adasyn', 'random_under']
    
    results = {}
    
    for method in basic_methods:
        try:
            print(f"\n测试方法: {method}")
            handler = create_imbalance_handler(method=method, random_state=42)
            X_resampled, y_resampled = handler.fit_transform(X_a, y_a)
            
            distribution = Counter(y_resampled)
            results[method] = {
                'shape': X_resampled.shape,
                'distribution': distribution,
                'total_samples': len(y_resampled)
            }
            
            print(f"  重采样后形状: {X_resampled.shape}")
            print(f"  类别分布: {distribution}")
            
        except Exception as e:
            print(f"  {method} 失败: {e}")
            results[method] = {'error': str(e)}
    
    return results

def test_smotenc_method():
    """测试SMOTENC方法（处理混合数据类型）"""
    print("\n" + "=" * 60)
    print("测试SMOTENC方法（混合数据类型）")
    print("=" * 60)
    
    # 加载数据
    loader = MedicalDataLoader()
    data_a = loader.load_dataset('A', feature_type='best10')
    
    X_a = data_a['X']
    y_a = data_a['y']
    
    # 获取类别特征索引
    categorical_indices = get_categorical_indices('best10')
    
    print(f"数据形状: {X_a.shape}")
    print(f"类别特征索引: {categorical_indices}")
    print(f"类别分布: {Counter(y_a)}")
    
    try:
        # 测试SMOTENC
        handler = create_imbalance_handler(
            method='smotenc',
            categorical_features=categorical_indices,
            random_state=42
        )
        X_resampled, y_resampled = handler.fit_transform(X_a, y_a)
        
        print(f"SMOTENC重采样后形状: {X_resampled.shape}")
        print(f"SMOTENC类别分布: {Counter(y_resampled)}")
        
        return {
            'shape': X_resampled.shape,
            'distribution': Counter(y_resampled),
            'categorical_indices': categorical_indices
        }
        
    except Exception as e:
        print(f"SMOTENC测试失败: {e}")
        return {'error': str(e)}

def test_advanced_smote_variants():
    """测试高级SMOTE变体"""
    print("\n" + "=" * 60)
    print("测试高级SMOTE变体")
    print("=" * 60)
    
    # 加载数据
    loader = MedicalDataLoader()
    data_a = loader.load_dataset('A', feature_type='best8')
    
    X_a = data_a['X']
    y_a = data_a['y']
    
    print(f"数据形状: {X_a.shape}")
    print(f"类别分布: {Counter(y_a)}")
    
    # 测试高级变体
    advanced_methods = ['kmeans_smote', 'svm_smote']
    results = {}
    
    for method in advanced_methods:
        try:
            print(f"\n测试方法: {method}")
            handler = create_imbalance_handler(method=method, random_state=42)
            X_resampled, y_resampled = handler.fit_transform(X_a, y_a)
            
            distribution = Counter(y_resampled)
            results[method] = {
                'shape': X_resampled.shape,
                'distribution': distribution
            }
            
            print(f"  重采样后形状: {X_resampled.shape}")
            print(f"  类别分布: {distribution}")
            
        except Exception as e:
            print(f"  {method} 失败: {e}")
            results[method] = {'error': str(e)}
    
    return results

def test_combination_methods():
    """测试组合方法（SMOTETomek和SMOTEENN）"""
    print("\n" + "=" * 60)
    print("测试组合方法（过采样+欠采样）")
    print("=" * 60)
    
    # 加载数据
    loader = MedicalDataLoader()
    data_a = loader.load_dataset('A', feature_type='best7')
    
    X_a = data_a['X']
    y_a = data_a['y']
    
    print(f"数据形状: {X_a.shape}")
    print(f"类别分布: {Counter(y_a)}")
    
    # 测试组合方法
    combination_methods = ['smote_tomek', 'smote_enn']
    results = {}
    
    for method in combination_methods:
        try:
            print(f"\n测试方法: {method}")
            
            # 配置组合方法
            if method == 'smote_tomek':
                handler = create_imbalance_handler(
                    method=method,
                    random_state=42,
                    smote_config={'k_neighbors': 5},
                    tomek_config={'sampling_strategy': 'all'}
                )
            else:  # smote_enn
                handler = create_imbalance_handler(
                    method=method,
                    random_state=42,
                    smote_config={'k_neighbors': 5},
                    enn_config={'sampling_strategy': 'all', 'n_neighbors': 3}
                )
            
            X_resampled, y_resampled = handler.fit_transform(X_a, y_a)
            
            distribution = Counter(y_resampled)
            results[method] = {
                'shape': X_resampled.shape,
                'distribution': distribution
            }
            
            print(f"  重采样后形状: {X_resampled.shape}")
            print(f"  类别分布: {distribution}")
            
            # 获取采样信息
            sampling_info = handler.get_sampling_info()
            print(f"  采样信息: {sampling_info['method']}")
            
        except Exception as e:
            print(f"  {method} 失败: {e}")
            results[method] = {'error': str(e)}
    
    return results

def test_cross_dataset():
    """测试跨数据集的不平衡处理"""
    print("\n" + "=" * 60)
    print("测试跨数据集不平衡处理")
    print("=" * 60)
    
    loader = MedicalDataLoader()
    results = {}
    
    # 测试不同数据集
    datasets = ['A', 'B']
    method = 'smote'
    
    for dataset in datasets:
        try:
            print(f"\n测试数据集: {dataset}")
            data = loader.load_dataset(dataset, feature_type='best7')
            
            X = data['X']
            y = data['y']
            
            print(f"  原始形状: {X.shape}")
            print(f"  原始分布: {Counter(y)}")
            
            handler = create_imbalance_handler(method=method, random_state=42)
            X_resampled, y_resampled = handler.fit_transform(X, y)
            
            distribution = Counter(y_resampled)
            results[dataset] = {
                'original_shape': X.shape,
                'resampled_shape': X_resampled.shape,
                'original_distribution': Counter(y),
                'resampled_distribution': distribution
            }
            
            print(f"  重采样后形状: {X_resampled.shape}")
            print(f"  重采样后分布: {distribution}")
            
        except Exception as e:
            print(f"  数据集 {dataset} 失败: {e}")
            results[dataset] = {'error': str(e)}
    
    return results

def create_comprehensive_visualization(results):
    """创建全面的可视化结果"""
    print("\n" + "=" * 60)
    print("生成全面可视化结果")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = "tests/imgs/imbalance_handler"
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取有效结果
    valid_results = {}
    for method, result in results.items():
        if 'error' not in result and 'distribution' in result:
            valid_results[method] = result
    
    if not valid_results:
        print("没有有效的结果用于可视化")
        return
    
    # 1. 样本数量和类别分布对比
    methods = list(valid_results.keys())
    sample_counts = []
    class_0_counts = []
    class_1_counts = []
    balance_ratios = []
    
    for method in methods:
        distribution = valid_results[method]['distribution']
        total = sum(distribution.values())
        class_0 = distribution.get(0, 0)
        class_1 = distribution.get(1, 0)
        
        sample_counts.append(total)
        class_0_counts.append(class_0)
        class_1_counts.append(class_1)
        
        # 计算平衡比例（最小类别/最大类别）
        if class_0 > 0 and class_1 > 0:
            balance_ratio = min(class_0, class_1) / max(class_0, class_1)
        else:
            balance_ratio = 0
        balance_ratios.append(balance_ratio)
    
    # 创建综合对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 子图1: 总样本数对比
    axes[0, 0].bar(methods, sample_counts, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Total Samples After Resampling')
    axes[0, 0].set_xlabel('Resampling Method')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 在柱状图上添加数值标签
    for i, count in enumerate(sample_counts):
        axes[0, 0].text(i, count + 5, str(count), ha='center', va='bottom')
    
    # 子图2: 类别分布对比
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = axes[0, 1].bar(x - width/2, class_0_counts, width, label='Class 0', alpha=0.7, color='lightcoral')
    bars2 = axes[0, 1].bar(x + width/2, class_1_counts, width, label='Class 1', alpha=0.7, color='lightgreen')
    
    axes[0, 1].set_title('Class Distribution After Resampling')
    axes[0, 1].set_xlabel('Resampling Method')
    axes[0, 1].set_ylabel('Number of Samples')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(methods, rotation=45)
    axes[0, 1].legend()
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 子图3: 平衡比例对比
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = axes[1, 0].bar(methods, balance_ratios, color=colors, alpha=0.7)
    axes[1, 0].set_title('Class Balance Ratio (Min/Max)')
    axes[1, 0].set_xlabel('Resampling Method')
    axes[1, 0].set_ylabel('Balance Ratio')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Balance')
    axes[1, 0].legend()
    
    # 添加数值标签
    for i, ratio in enumerate(balance_ratios):
        axes[1, 0].text(i, ratio + 0.02, f'{ratio:.3f}', ha='center', va='bottom')
    
    # 子图4: 采样效果统计表
    axes[1, 1].axis('tight')
    axes[1, 1].axis('off')
    
    # 创建统计表格数据
    table_data = []
    headers = ['Method', 'Total', 'Class 0', 'Class 1', 'Balance Ratio']
    
    for i, method in enumerate(methods):
        row = [
            method,
            sample_counts[i],
            class_0_counts[i],
            class_1_counts[i],
            f'{balance_ratios[i]:.3f}'
        ]
        table_data.append(row)
    
    table = axes[1, 1].table(cellText=table_data, colLabels=headers,
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('Resampling Statistics Summary')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_resampling_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 创建方法类型分组对比
    method_groups = {
        'Basic Over-sampling': ['smote', 'borderline_smote', 'adasyn'],
        'Advanced SMOTE': ['smotenc', 'kmeans_smote', 'svm_smote'],
        'Combination Methods': ['smote_tomek', 'smote_enn'],
        'Under-sampling': ['random_under', 'edited_nn'],
        'No Resampling': ['none']
    }
    
    # 按组绘制对比图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    group_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    x_pos = 0
    group_positions = []
    group_labels = []
    
    for i, (group_name, group_methods) in enumerate(method_groups.items()):
        group_methods_in_results = [m for m in group_methods if m in valid_results]
        if not group_methods_in_results:
            continue
            
        group_x_positions = []
        for method in group_methods_in_results:
            distribution = valid_results[method]['distribution']
            class_0 = distribution.get(0, 0)
            class_1 = distribution.get(1, 0)
            
            # 绘制堆叠柱状图
            ax.bar(x_pos, class_0, color=group_colors[i], alpha=0.7, label=f'{group_name} - Class 0' if method == group_methods_in_results[0] else "")
            ax.bar(x_pos, class_1, bottom=class_0, color=group_colors[i], alpha=0.4, label=f'{group_name} - Class 1' if method == group_methods_in_results[0] else "")
            
            # 添加方法名标签
            ax.text(x_pos, class_0 + class_1 + 10, method, rotation=45, ha='right', va='bottom', fontsize=8)
            
            group_x_positions.append(x_pos)
            x_pos += 1
        
        # 记录组的位置
        if group_x_positions:
            group_positions.append(np.mean(group_x_positions))
            group_labels.append(group_name)
        
        x_pos += 0.5  # 组间间隔
    
    ax.set_title('Resampling Methods Grouped by Type')
    ax.set_xlabel('Methods (Grouped by Type)')
    ax.set_ylabel('Number of Samples')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 添加组分隔线
    for i in range(len(group_positions) - 1):
        sep_pos = (group_positions[i] + group_positions[i + 1]) / 2
        ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/methods_by_type_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"综合可视化结果已保存到:")
    print(f"  - {output_dir}/comprehensive_resampling_analysis.png")
    print(f"  - {output_dir}/methods_by_type_comparison.png")

def main():
    """主测试函数"""
    print("开始测试不平衡处理器...")
    
    # 运行所有测试
    basic_results = test_basic_methods()
    smotenc_results = test_smotenc_method()
    advanced_results = test_advanced_smote_variants()
    combination_results = test_combination_methods()
    cross_dataset_results = test_cross_dataset()
    
    # 合并结果
    all_results = {**basic_results, **advanced_results, **combination_results}
    
    # 生成可视化
    create_comprehensive_visualization(all_results)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    successful_methods = []
    failed_methods = []
    
    for method, result in all_results.items():
        if 'error' in result:
            failed_methods.append(method)
        else:
            successful_methods.append(method)
    
    print(f"成功的方法 ({len(successful_methods)}): {successful_methods}")
    if failed_methods:
        print(f"失败的方法 ({len(failed_methods)}): {failed_methods}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main() 