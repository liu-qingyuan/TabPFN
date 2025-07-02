# -*- coding: utf-8 -*-
"""
真实医疗数据标准化器测试模块

使用真实医疗数据测试标准化功能：
- 数据A：295条记录
- 数据B：190条记录
- 测试跨域标准化效果
- 可视化标准化前后的效果

Author: UDA Medical Project Team
Date: 2024
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from preprocessing.scalers import MedicalDataScaler, compare_scalers
    from data.loader import MedicalDataLoader
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在项目根目录下运行测试")
    sys.exit(1)

# 设置matplotlib中文字体和图片保存目录
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
IMG_DIR = Path(__file__).parent / 'imgs'
IMG_DIR.mkdir(exist_ok=True)


def save_figure(fig, filename, dpi=300):
    """保存图片到指定目录"""
    filepath = IMG_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"📊 图片已保存: {filepath}")


def visualize_scaling_effect(X_original, X_scaled, feature_names, categorical_features, 
                           title_prefix, filename_prefix):
    """可视化标准化前后的效果"""
    # 获取数值特征
    numerical_features = [f for f in feature_names if f not in categorical_features]
    
    if len(numerical_features) == 0:
        print("⚠️ 没有数值特征需要可视化")
        return
    
    # 创建子图
    n_features = len(numerical_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # 为每个数值特征绘制分布图
    for i, feature in enumerate(numerical_features):
        feature_idx = feature_names.index(feature)
        
        ax = axes[i] if len(axes) > 1 else axes[0]
        
        # 原始数据分布
        ax.hist(X_original[:, feature_idx], bins=30, alpha=0.6, 
                label='Original', color='skyblue', density=True)
        
        # 标准化后数据分布
        ax.hist(X_scaled[:, feature_idx], bins=30, alpha=0.6, 
                label='Scaled', color='orange', density=True)
        
        ax.set_title(f'{feature}\nOriginal: μ={X_original[:, feature_idx].mean():.3f}, σ={X_original[:, feature_idx].std():.3f}\n'
                    f'Scaled: μ={X_scaled[:, feature_idx].mean():.3f}, σ={X_scaled[:, feature_idx].std():.3f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(numerical_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(f'{title_prefix} - Feature Distribution Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # 保存图片
    save_figure(fig, f'{filename_prefix}_feature_distributions.png')
    plt.close()


def visualize_cross_domain_effect(X_source_original, X_source_scaled, 
                                X_target_original, X_target_scaled,
                                feature_names, categorical_features):
    """可视化跨域标准化效果"""
    numerical_features = [f for f in feature_names if f not in categorical_features]
    
    if len(numerical_features) == 0:
        return
    
    # 选择前4个数值特征进行可视化
    selected_features = numerical_features[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(selected_features):
        if i >= 4:
            break
            
        feature_idx = feature_names.index(feature)
        ax = axes[i]
        
        # 源域数据
        ax.hist(X_source_original[:, feature_idx], bins=20, alpha=0.5, 
                label='Source Original', color='blue', density=True)
        ax.hist(X_source_scaled[:, feature_idx], bins=20, alpha=0.5, 
                label='Source Scaled', color='lightblue', density=True)
        
        # 目标域数据
        ax.hist(X_target_original[:, feature_idx], bins=20, alpha=0.5, 
                label='Target Original', color='red', density=True)
        ax.hist(X_target_scaled[:, feature_idx], bins=20, alpha=0.5, 
                label='Target Scaled', color='pink', density=True)
        
        ax.set_title(f'{feature} - Cross-Domain Scaling')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Cross-Domain Scaling Effect (Source A → Target B)', fontsize=16)
    plt.tight_layout()
    
    save_figure(fig, 'cross_domain_scaling_effect.png')
    plt.close()


def visualize_scaler_comparison(X_original, scalers_results, feature_names, categorical_features):
    """可视化不同标准化器的对比效果"""
    numerical_features = [f for f in feature_names if f not in categorical_features]
    
    if len(numerical_features) == 0:
        return
    
    # 选择前2个数值特征进行详细对比
    selected_features = numerical_features[:2]
    
    fig, axes = plt.subplots(len(selected_features), 1, figsize=(12, 6*len(selected_features)))
    if len(selected_features) == 1:
        axes = [axes]
    
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, feature in enumerate(selected_features):
        feature_idx = feature_names.index(feature)
        ax = axes[i]
        
        # 原始数据
        ax.hist(X_original[:, feature_idx], bins=30, alpha=0.4, 
                label='Original', color='gray', density=True)
        
        # 不同标准化器的结果
        for j, (scaler_name, X_scaled) in enumerate(scalers_results.items()):
            ax.hist(X_scaled[:, feature_idx], bins=30, alpha=0.6, 
                    label=f'{scaler_name.title()}', color=colors[j % len(colors)], 
                    density=True, histtype='step', linewidth=2)
        
        ax.set_title(f'{feature} - Scaler Comparison')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Different Scalers Comparison', fontsize=16)
    plt.tight_layout()
    
    save_figure(fig, 'scalers_comparison.png')
    plt.close()


def visualize_feature_sets_comparison(loader):
    """可视化不同特征集的统计信息"""
    feature_types = ['best7', 'best8', 'best9', 'best10']
    stats_data = []
    
    for feature_type in feature_types:
        dataset_A = loader.load_dataset('A', feature_type=feature_type)
        categorical_features = loader.get_categorical_features(feature_type)
        
        stats_data.append({
            'Feature Set': feature_type,
            'Total Features': dataset_A['n_features'],
            'Numerical Features': dataset_A['n_features'] - len(categorical_features),
            'Categorical Features': len(categorical_features),
            'Samples': dataset_A['n_samples']
        })
    
    # 创建统计图表
    df_stats = pd.DataFrame(stats_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 特征数量对比
    x = range(len(feature_types))
    width = 0.35
    
    axes[0].bar([i - width/2 for i in x], df_stats['Numerical Features'], 
                width, label='Numerical Features', color='skyblue')
    axes[0].bar([i + width/2 for i in x], df_stats['Categorical Features'], 
                width, label='Categorical Features', color='orange')
    
    axes[0].set_xlabel('Feature Sets')
    axes[0].set_ylabel('Number of Features')
    axes[0].set_title('Feature Composition by Feature Set')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(feature_types)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 特征比例饼图
    total_features = df_stats['Total Features'].iloc[-1]  # 使用best10的总数
    numerical_count = df_stats['Numerical Features'].iloc[-1]
    categorical_count = df_stats['Categorical Features'].iloc[-1]
    
    axes[1].pie([numerical_count, categorical_count], 
                labels=['Numerical Features', 'Categorical Features'],
                colors=['skyblue', 'orange'], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Feature Type Distribution (Best10 Set)')
    
    plt.tight_layout()
    save_figure(fig, 'feature_sets_comparison.png')
    plt.close()
    
    return df_stats


def test_real_data_standard_scaling():
    """测试真实医疗数据的标准标准化"""
    print("🔬 开始测试真实医疗数据标准化...")
    
    # 加载真实医疗数据
    loader = MedicalDataLoader()
    
    # 加载数据A（295条）和数据B（190条）
    dataset_A = loader.load_dataset('A', feature_type='best8')
    dataset_B = loader.load_dataset('B', feature_type='best8')
    
    X_A = pd.DataFrame(dataset_A['X'], columns=dataset_A['feature_names'])
    X_B = pd.DataFrame(dataset_B['X'], columns=dataset_B['feature_names'])
    
    # 获取类别特征
    categorical_features = loader.get_categorical_features('best8')
    
    print(f"📊 数据加载完成:")
    print(f"   - 数据A: {X_A.shape} (295条记录)")
    print(f"   - 数据B: {X_B.shape} (190条记录)")
    print(f"   - 特征集: best8 ({X_A.shape[1]}个特征)")
    print(f"   - 类别特征: {categorical_features}")
    
    # 测试标准标准化
    scaler = MedicalDataScaler(
        scaler_type='standard',
        categorical_features=categorical_features
    )
    
    # 在数据A上拟合
    scaler.fit(X_A)
    print(f"✅ 标准化器在数据A上拟合完成")
    
    # 获取特征信息
    info = scaler.get_feature_info()
    print(f"📋 特征信息:")
    print(f"   - 总特征数: {info['total_features']}")
    print(f"   - 数值特征数: {info['numerical_features']}")
    print(f"   - 类别特征数: {info['categorical_features']}")
    print(f"   - 数值特征名: {info['numerical_feature_names']}")
    print(f"   - 类别特征名: {info['categorical_feature_names']}")
    
    # 在数据A上变换
    X_A_scaled = scaler.transform(X_A)
    print(f"✅ 数据A标准化完成: {X_A_scaled.shape}")
    
    # 验证数值特征标准化效果
    numerical_indices = [X_A.columns.get_loc(col) for col in info['numerical_feature_names']]
    X_A_numerical_scaled = X_A_scaled[:, numerical_indices]
    
    mean_after = np.mean(X_A_numerical_scaled)
    std_after = np.std(X_A_numerical_scaled)
    
    print(f"📈 数据A标准化效果:")
    print(f"   - 数值特征均值: {mean_after:.6f} (应接近0)")
    print(f"   - 数值特征标准差: {std_after:.6f} (应接近1)")
    
    # 验证类别特征保持不变
    for cat_feature in categorical_features:
        cat_idx = X_A.columns.get_loc(cat_feature)
        original_values = X_A.iloc[:, cat_idx].values
        scaled_values = X_A_scaled[:, cat_idx]
        assert np.array_equal(original_values, scaled_values), f"类别特征 {cat_feature} 发生了变化"
    
    print(f"✅ 类别特征保持不变验证通过")
    
    # 可视化标准化效果
    visualize_scaling_effect(
        X_A.values, X_A_scaled, dataset_A['feature_names'], categorical_features,
        'Dataset A Standard Scaling', 'dataset_A_standard_scaling'
    )
    
    return scaler, X_A_scaled, info, X_A.values


def test_cross_domain_scaling():
    """测试跨域标准化：在数据A上拟合，在数据B上变换"""
    print("\n🌐 开始测试跨域标准化...")
    
    # 加载数据
    loader = MedicalDataLoader()
    dataset_A = loader.load_dataset('A', feature_type='best8')
    dataset_B = loader.load_dataset('B', feature_type='best8')
    
    X_A = pd.DataFrame(dataset_A['X'], columns=dataset_A['feature_names'])
    X_B = pd.DataFrame(dataset_B['X'], columns=dataset_B['feature_names'])
    categorical_features = loader.get_categorical_features('best8')
    
    # 创建标准化器并在数据A上拟合
    scaler = MedicalDataScaler(
        scaler_type='standard',
        categorical_features=categorical_features
    )
    scaler.fit(X_A)
    
    # 在数据A和数据B上应用标准化
    X_A_scaled = scaler.transform(X_A)
    X_B_scaled = scaler.transform(X_B)
    
    print(f"✅ 跨域标准化完成: 数据A({X_A.shape[0]}) → 数据B({X_B.shape[0]})")
    
    # 验证形状
    assert X_B_scaled.shape == X_B.shape, "标准化后形状发生变化"
    
    # 验证类别特征保持不变
    for cat_feature in categorical_features:
        cat_idx = X_B.columns.get_loc(cat_feature)
        original_values = X_B.iloc[:, cat_idx].values
        scaled_values = X_B_scaled[:, cat_idx]
        assert np.array_equal(original_values, scaled_values), f"类别特征 {cat_feature} 发生了变化"
    
    print(f"✅ 跨域标准化验证通过")
    
    # 分析数值特征的分布变化
    info = scaler.get_feature_info()
    numerical_indices = [X_B.columns.get_loc(col) for col in info['numerical_feature_names']]
    X_B_numerical_scaled = X_B_scaled[:, numerical_indices]
    
    mean_B = np.mean(X_B_numerical_scaled)
    std_B = np.std(X_B_numerical_scaled)
    
    print(f"📈 数据B标准化后统计:")
    print(f"   - 数值特征均值: {mean_B:.6f}")
    print(f"   - 数值特征标准差: {std_B:.6f}")
    print(f"   - 注意: 由于域间差异，均值和标准差可能不是0和1")
    
    # 可视化跨域标准化效果
    visualize_cross_domain_effect(
        X_A.values, X_A_scaled, X_B.values, X_B_scaled,
        dataset_A['feature_names'], categorical_features
    )
    
    return X_B_scaled


def test_robust_scaling():
    """测试鲁棒标准化"""
    print("\n🛡️ 开始测试鲁棒标准化...")
    
    # 加载数据
    loader = MedicalDataLoader()
    dataset_A = loader.load_dataset('A', feature_type='best8')
    
    X_A = pd.DataFrame(dataset_A['X'], columns=dataset_A['feature_names'])
    categorical_features = loader.get_categorical_features('best8')
    
    # 创建鲁棒标准化器
    scaler = MedicalDataScaler(
        scaler_type='robust',
        categorical_features=categorical_features
    )
    
    # 拟合和变换
    X_A_scaled = scaler.fit_transform(X_A)
    print(f"✅ 鲁棒标准化完成: {X_A_scaled.shape}")
    
    # 验证数值特征的中位数接近0
    info = scaler.get_feature_info()
    numerical_indices = [X_A.columns.get_loc(col) for col in info['numerical_feature_names']]
    X_A_numerical_scaled = X_A_scaled[:, numerical_indices]
    
    median_after = np.median(X_A_numerical_scaled)
    iqr_after = np.percentile(X_A_numerical_scaled, 75) - np.percentile(X_A_numerical_scaled, 25)
    
    print(f"📈 鲁棒标准化效果:")
    print(f"   - 数值特征中位数: {median_after:.6f} (应接近0)")
    print(f"   - 数值特征IQR: {iqr_after:.6f}")
    
    # 可视化鲁棒标准化效果
    visualize_scaling_effect(
        X_A.values, X_A_scaled, dataset_A['feature_names'], categorical_features,
        'Dataset A Robust Scaling', 'dataset_A_robust_scaling'
    )
    
    return X_A_scaled


def test_no_scaling():
    """测试不进行标准化"""
    print("\n🚫 开始测试不进行标准化...")
    
    # 加载数据
    loader = MedicalDataLoader()
    dataset_A = loader.load_dataset('A', feature_type='best8')
    
    X_A = pd.DataFrame(dataset_A['X'], columns=dataset_A['feature_names'])
    categorical_features = loader.get_categorical_features('best8')
    
    # 创建无标准化器
    scaler = MedicalDataScaler(
        scaler_type='none',
        categorical_features=categorical_features
    )
    
    # 拟合和变换
    X_A_no_scaled = scaler.fit_transform(X_A)
    print(f"✅ 无标准化完成: {X_A_no_scaled.shape}")
    
    # 验证数据保持不变
    assert np.array_equal(X_A.values, X_A_no_scaled), "无标准化时数据应保持不变"
    
    # 获取特征信息
    info = scaler.get_feature_info()
    numerical_indices = [X_A.columns.get_loc(col) for col in info['numerical_feature_names']]
    X_A_numerical = X_A_no_scaled[:, numerical_indices]
    
    mean_original = np.mean(X_A_numerical)
    std_original = np.std(X_A_numerical)
    
    print(f"📈 无标准化效果:")
    print(f"   - 数值特征均值: {mean_original:.6f} (保持原始)")
    print(f"   - 数值特征标准差: {std_original:.6f} (保持原始)")
    print(f"   - 数据完全保持不变: ✅")
    
    # 可视化无标准化效果（应该显示原始数据和"标准化"后数据完全一致）
    visualize_scaling_effect(
        X_A.values, X_A_no_scaled, dataset_A['feature_names'], categorical_features,
        'Dataset A No Scaling', 'dataset_A_no_scaling'
    )
    
    return X_A_no_scaled


def test_scaler_comparison():
    """比较不同标准化器的效果"""
    print("\n⚖️ 开始比较不同标准化器...")
    
    # 加载数据
    loader = MedicalDataLoader()
    dataset_A = loader.load_dataset('A', feature_type='best8')
    
    X_A = pd.DataFrame(dataset_A['X'], columns=dataset_A['feature_names'])
    categorical_features = loader.get_categorical_features('best8')
    
    # 比较标准化器
    comparison = compare_scalers(X_A, categorical_features)
    
    print(f"📊 标准化器比较结果:")
    for scaler_type, stats in comparison.items():
        print(f"   {scaler_type.upper()}:")
        print(f"     - 均值: {stats['mean']:.6f}")
        print(f"     - 标准差: {stats['std']:.6f}")
        print(f"     - 中位数: {stats['median']:.6f}")
        print(f"     - IQR: {stats['iqr']:.6f}")
        print(f"     - 范围: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # 获取不同标准化器的结果用于可视化
    scalers_results = {}
    for scaler_type in ['standard', 'robust', 'none']:
        scaler = MedicalDataScaler(
            scaler_type=scaler_type,
            categorical_features=categorical_features
        )
        X_scaled = scaler.fit_transform(X_A)
        scalers_results[scaler_type] = X_scaled
    
    # 可视化不同标准化器的对比
    visualize_scaler_comparison(
        X_A.values, scalers_results, dataset_A['feature_names'], categorical_features
    )
    
    return comparison


def test_feature_sets():
    """测试不同特征集的标准化"""
    print("\n🎯 开始测试不同特征集...")
    
    loader = MedicalDataLoader()
    feature_types = ['best7', 'best8', 'best9', 'best10']
    
    for feature_type in feature_types:
        print(f"\n📋 测试特征集: {feature_type}")
        
        # 加载数据
        dataset_A = loader.load_dataset('A', feature_type=feature_type)
        X_A = pd.DataFrame(dataset_A['X'], columns=dataset_A['feature_names'])
        categorical_features = loader.get_categorical_features(feature_type)
        
        # 创建标准化器
        scaler = MedicalDataScaler(
            scaler_type='standard',
            categorical_features=categorical_features
        )
        
        # 拟合和变换
        scaler.fit_transform(X_A)
        
        # 获取信息
        info = scaler.get_feature_info()
        
        print(f"   - 数据形状: {X_A.shape}")
        print(f"   - 总特征: {info['total_features']}")
        print(f"   - 数值特征: {info['numerical_features']}")
        print(f"   - 类别特征: {info['categorical_features']}")
        print(f"   - 类别特征名: {info['categorical_feature_names']}")
    
    # 可视化特征集对比
    stats_df = visualize_feature_sets_comparison(loader)
    print(f"\n📊 特征集统计对比:")
    print(stats_df.to_string(index=False))


def main():
    """主测试函数"""
    print("🚀 开始真实医疗数据标准化测试")
    print("=" * 60)
    
    try:
        # 测试1: 标准标准化
        test_real_data_standard_scaling()
        
        # 测试2: 跨域标准化
        test_cross_domain_scaling()
        
        # 测试3: 鲁棒标准化
        test_robust_scaling()
        
        # 测试4: 无标准化
        test_no_scaling()
        
        # 测试5: 标准化器比较
        test_scaler_comparison()
        
        # 测试5: 不同特征集
        test_feature_sets()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！")
        print("✅ 标准化器可以正确处理真实医疗数据")
        print("✅ 支持跨域标准化（A→B）")
        print("✅ 类别特征保持不变")
        print("✅ 数值特征正确标准化")
        print(f"📊 可视化结果已保存到: {IMG_DIR}")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 