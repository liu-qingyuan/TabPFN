#!/usr/bin/env python3
"""
UDA Medical Imbalance Project - 快速开始示例

本示例展示如何使用项目进行基础的UDA实验：
1. 加载和预处理数据
2. 应用不平衡处理
3. 运行UDA方法
4. 评估结果
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

print("=" * 60)
print("UDA Medical Imbalance Project - 快速开始示例")
print("=" * 60)

# 1. 创建模拟医疗数据集
print("\n1. 创建模拟医疗数据集...")

# 源域数据（假设为AI4health数据集）
X_source, y_source = make_classification(
    n_samples=800,
    n_features=10, 
    n_classes=2,
    weights=[0.7, 0.3],  # 轻微不平衡
    flip_y=0.05,
    random_state=42
)

# 目标域数据（假设为HenanCancerHospital数据集，存在域偏移）
X_target, y_target = make_classification(
    n_samples=600,
    n_features=10,
    n_classes=2, 
    weights=[0.8, 0.2],  # 更严重的不平衡
    flip_y=0.03,
    random_state=123
)

# 添加域偏移（目标域特征分布偏移）
X_target = X_target + np.random.normal(1.5, 0.5, X_target.shape)

print(f"源域数据形状: {X_source.shape}, 类别分布: {np.bincount(y_source)}")
print(f"目标域数据形状: {X_target.shape}, 类别分布: {np.bincount(y_target)}")

# 2. 特征选择（选择7个最重要的特征）
print("\n2. 特征选择（选择7个特征）...")

from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=7)
X_source_selected = selector.fit_transform(X_source, y_source)
X_target_selected = selector.transform(X_target)

print(f"特征选择后源域形状: {X_source_selected.shape}")
print(f"特征选择后目标域形状: {X_target_selected.shape}")

# 3. 数据标准化
print("\n3. 数据标准化...")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_source_scaled = scaler.fit_transform(X_source_selected)
X_target_scaled = scaler.transform(X_target_selected)

print("数据标准化完成")

# 4. 类别不平衡处理（SMOTE）
print("\n4. 类别不平衡处理（SMOTE）...")

try:
    from imblearn.over_sampling import SMOTE
    
    smote = SMOTE(random_state=42)
    X_source_resampled, y_source_resampled = smote.fit_transform(X_source_scaled, y_source)
    
    print(f"SMOTE前源域类别分布: {np.bincount(y_source)}")
    print(f"SMOTE后源域类别分布: {np.bincount(y_source_resampled)}")
    
except ImportError:
    print("imblearn库未安装，跳过SMOTE处理")
    X_source_resampled = X_source_scaled
    y_source_resampled = y_source

# 5. 基线方法（不使用域适应）
print("\n5. 基线方法评估（无域适应）...")

# 在源域训练，在目标域测试
baseline_model = LogisticRegression(random_state=42, max_iter=1000)
baseline_model.fit(X_source_resampled, y_source_resampled)

y_pred_baseline = baseline_model.predict(X_target_scaled)
y_pred_proba_baseline = baseline_model.predict_proba(X_target_scaled)[:, 1]

baseline_auc = roc_auc_score(y_target, y_pred_proba_baseline)
print(f"基线方法 AUC: {baseline_auc:.4f}")

# 6. 简单域适应方法（均值-标准差对齐）
print("\n6. 简单域适应方法（均值-标准差对齐）...")

# 计算源域和目标域的统计量
source_mean = np.mean(X_source_resampled, axis=0)
source_std = np.std(X_source_resampled, axis=0) + 1e-8
target_mean = np.mean(X_target_scaled, axis=0)
target_std = np.std(X_target_scaled, axis=0) + 1e-8

# 将源域数据适应到目标域分布
X_source_adapted = (X_source_resampled - source_mean) / source_std
X_source_adapted = X_source_adapted * target_std + target_mean

# 在适应后的源域数据上训练
adapted_model = LogisticRegression(random_state=42, max_iter=1000)
adapted_model.fit(X_source_adapted, y_source_resampled)

# 在目标域测试
y_pred_adapted = adapted_model.predict(X_target_scaled)
y_pred_proba_adapted = adapted_model.predict_proba(X_target_scaled)[:, 1]

adapted_auc = roc_auc_score(y_target, y_pred_proba_adapted)
print(f"域适应方法 AUC: {adapted_auc:.4f}")

# 7. 结果对比
print("\n7. 结果对比...")
print("=" * 40)
print(f"基线方法（无域适应）AUC: {baseline_auc:.4f}")
print(f"简单域适应方法 AUC:   {adapted_auc:.4f}")
print(f"性能提升: {adapted_auc - baseline_auc:+.4f}")
print("=" * 40)

# 8. 详细分类报告
print("\n8. 详细分类报告...")

print("\n基线方法分类报告:")
print(classification_report(y_target, y_pred_baseline, 
                          target_names=['阴性', '阳性']))

print("\n域适应方法分类报告:")
print(classification_report(y_target, y_pred_adapted,
                          target_names=['阴性', '阳性']))

# 9. 可视化（如果matplotlib可用）
print("\n9. 数据分布可视化...")

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('UDA Medical Data Analysis - Feature Distribution Comparison', 
                 fontsize=14, fontweight='bold')
    
    # 选择第一个特征进行可视化
    feature_idx = 0
    
    # 原始分布
    axes[0, 0].hist(X_source_scaled[:, feature_idx], alpha=0.7, 
                    label='Source Domain', bins=30, color='blue')
    axes[0, 0].hist(X_target_scaled[:, feature_idx], alpha=0.7, 
                    label='Target Domain', bins=30, color='red')
    axes[0, 0].set_title('Original Feature Distribution')
    axes[0, 0].set_xlabel('Feature Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 适应后分布
    axes[0, 1].hist(X_source_adapted[:, feature_idx], alpha=0.7,
                    label='Adapted Source', bins=30, color='green')
    axes[0, 1].hist(X_target_scaled[:, feature_idx], alpha=0.7,
                    label='Target Domain', bins=30, color='red')
    axes[0, 1].set_title('Adapted Feature Distribution')
    axes[0, 1].set_xlabel('Feature Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 性能对比柱状图
    methods = ['Baseline\n(No Adaptation)', 'Simple Domain\nAdaptation']
    aucs = [baseline_auc, adapted_auc]
    
    bars = axes[1, 0].bar(methods, aucs, color=['skyblue', 'lightgreen'], 
                         edgecolor='black', linewidth=1)
    axes[1, 0].set_title('Performance Comparison (AUC)')
    axes[1, 0].set_ylabel('AUC Score')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值标签
    for bar, auc in zip(bars, aucs):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{auc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 类别分布对比
    domains = ['Source\n(Original)', 'Source\n(Resampled)', 'Target']
    pos_ratios = [
        y_source.mean(),
        y_source_resampled.mean(),
        y_target.mean()
    ]
    
    bars = axes[1, 1].bar(domains, pos_ratios, 
                         color=['lightblue', 'lightcoral', 'lightgray'],
                         edgecolor='black', linewidth=1)
    axes[1, 1].set_title('Positive Class Ratio Comparison')
    axes[1, 1].set_ylabel('Positive Class Ratio')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, ratio in zip(bars, pos_ratios):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{ratio:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = "uda_medical_experiment_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到: {save_path}")
    
    # 显示图片（如果在交互环境中）
    try:
        plt.show()
    except:
        pass
        
except ImportError:
    print("matplotlib库未安装，跳过可视化")

print("\n" + "=" * 60)
print("快速开始示例完成!")
print("这个示例展示了UDA项目的核心工作流程：")
print("1. 数据加载和预处理")
print("2. 特征选择和标准化") 
print("3. 不平衡数据处理")
print("4. 简单域适应方法")
print("5. 性能评估和对比")
print("6. 结果可视化")
print("\n如需运行更复杂的UDA方法，请参考项目文档和其他示例。")
print("=" * 60) 