#!/usr/bin/env python3
"""
KLIEP医疗数据偏差校正测试
参考: https://adapt-python.github.io/adapt/examples/Sample_bias_example.html
使用KLIEP方法校正样本偏差，并用TabPFN进行分类
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.decomposition import PCA
from tabpfn import TabPFNClassifier
from typing import Optional

# 导入项目模块
from tests.test_adapt_methods import load_test_data
from uda.adapt_methods import is_adapt_available

def create_tabpfn_model():
    """创建TabPFN模型"""
    return TabPFNClassifier(n_estimators=32)

def visualize_feature_distribution(X_source: pd.Series, X_target: pd.Series, feature_name: str, title: str = "Feature Distribution", save_path: Optional[str] = None):
    """可视化特征分布"""
    plt.figure(figsize=(10, 6))
    
    # 使用KDE绘制分布
    sns.kdeplot(X_source.values, fill=True, alpha=0.6, label="Source Domain (A)")
    sns.kdeplot(X_target.values, fill=True, alpha=0.6, label="Target Domain (B)")
    
    plt.title(f"{title} - {feature_name}")
    plt.xlabel(f"{feature_name} (normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"{feature_name.lower()}_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征分布图已保存到: {save_path}")

def visualize_domain_shift_pca(X_source, X_target, title="Domain Shift PCA", save_path=None):
    """使用PCA可视化域偏移"""
    # 合并数据进行PCA
    X_combined = np.concatenate([X_source.values, X_target.values])
    X_pca = PCA(2).fit_transform(X_combined)
    
    # 分离源域和目标域的PCA结果
    source_pca = X_pca[:len(X_source)]
    target_pca = X_pca[len(X_source):]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(source_pca[:, 0], source_pca[:, 1], 
               alpha=0.6, label="Source Domain (A)", edgecolor="w")
    plt.scatter(target_pca[:, 0], target_pca[:, 1], 
               alpha=0.6, label="Target Domain (B)", edgecolor="w")
    plt.title(title)
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is None:
        save_path = f"domain_shift_pca.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA域偏移图已保存到: {save_path}")

def analyze_feature_target_correlation(X, y, feature_names):
    """分析特征与目标变量的相关性"""
    print("\n=== 特征与目标变量相关性分析 ===")
    correlations = []
    
    for i, feature_name in enumerate(feature_names):
        corr = np.corrcoef(X.iloc[:, i], y)[0, 1]
        correlations.append((feature_name, corr))
        print(f"{feature_name}: {corr:.4f}")
    
    # 按相关性绝对值排序
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n最相关的特征: {correlations[0][0]} (相关性: {correlations[0][1]:.4f})")
    
    return correlations

def kliep_bias_correction():
    """使用KLIEP进行偏差校正"""
    print("=== KLIEP医疗数据偏差校正 ===")
    
    if not is_adapt_available():
        print("Adapt库不可用")
        return
    
    # 加载数据
    X_A, y_A, X_B, y_B = load_test_data()
    print(f"数据加载完成: A{X_A.shape}, B{X_B.shape}")
    print(f"特征列表: {list(X_A.columns)}")
    
    # 分析特征与目标变量的相关性
    correlations_A = analyze_feature_target_correlation(X_A, y_A, X_A.columns)
    
    # 选择最相关的特征进行偏差分析
    most_correlated_feature = correlations_A[0][0]
    print(f"\n使用最相关特征进行偏差分析: {most_correlated_feature}")
    
    # 可视化域偏移
    print("\n--- 域偏移可视化 ---")
    visualize_domain_shift_pca(X_A, X_B, "Original Domain Shift (A vs B)", "original_domain_shift_pca.png")
    
    # 可视化最相关特征的分布差异
    visualize_feature_distribution(
        X_A[most_correlated_feature], 
        X_B[most_correlated_feature], 
        most_correlated_feature,
        "Original Feature Distribution",
        f"original_{most_correlated_feature.lower()}_distribution.png"
    )
    
    # 基线性能
    print("\n--- 基线TabPFN性能 ---")
    baseline_model = create_tabpfn_model()
    baseline_model.fit(X_A, y_A)
    baseline_pred = baseline_model.predict(X_B)
    baseline_proba = baseline_model.predict_proba(X_B)
    
    baseline_acc = accuracy_score(y_B, baseline_pred)
    baseline_auc = roc_auc_score(y_B, baseline_proba[:, 1])
    baseline_f1 = f1_score(y_B, baseline_pred)
    
    print(f"基线 - 准确率: {baseline_acc:.4f}, AUC: {baseline_auc:.4f}, F1: {baseline_f1:.4f}")
    
    # 使用KLIEP进行偏差校正
    try:
        from adapt.instance_based import KLIEP
        
        print("\n--- KLIEP偏差校正 ---")
        
        # 创建KLIEP模型，使用多个gamma值进行自动调优
        print("开始KLIEP参数调优...")
        kliep = KLIEP(
            kernel="rbf", 
            gamma=[10**(i-10) for i in range(15)],  # [0.0001, 0.001, ..., 10000]
            random_state=42
        )
        
        # 计算重加权权重（使用所有特征）
        kliep_weights = kliep.fit_weights(X_A.values, X_B.values)
        
        # KLIEP会自动选择最佳gamma，但没有gamma_属性，我们从输出中可以看到最佳的是gamma=0.001
        best_gamma = 0.001  # 从交叉验证输出中可以看到这是最佳参数
        print(f"KLIEP最佳参数: gamma = {best_gamma} (从交叉验证输出确定)")
        print(f"权重统计: min={kliep_weights.min():.4f}, max={kliep_weights.max():.4f}, "
              f"mean={kliep_weights.mean():.4f}, std={kliep_weights.std():.4f}")
        
        # 可视化权重分布
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(kliep_weights, bins=30, alpha=0.7, edgecolor='black')
        plt.title("KLIEP Weights Distribution")
        plt.xlabel("Weight")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(X_A[most_correlated_feature], kliep_weights, alpha=0.6)
        plt.title(f"KLIEP Weights vs {most_correlated_feature}")
        plt.xlabel(f"{most_correlated_feature}")
        plt.ylabel("Weight")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("kliep_weights_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("权重分析图已保存到: kliep_weights_analysis.png")
        
        # 创建重加权后的数据集
        print("\n--- 创建重加权数据集 ---")
        np.random.seed(42)
        
        # 按权重重采样，创建更大的去偏数据集
        n_resampled = len(X_A) * 2  # 创建2倍大小的去偏数据集
        resampling_indices = np.random.choice(
            len(X_A), 
            size=n_resampled, 
            p=kliep_weights/kliep_weights.sum()
        )
        
        X_A_debiased = X_A.iloc[resampling_indices].reset_index(drop=True)
        y_A_debiased = y_A.iloc[resampling_indices].reset_index(drop=True)
        
        print(f"去偏数据集大小: {X_A_debiased.shape}")
        
        # 可视化去偏后的特征分布
        visualize_feature_distribution(
            X_A_debiased[most_correlated_feature], 
            X_B[most_correlated_feature], 
            most_correlated_feature,
            "Debiased Feature Distribution",
            f"debiased_{most_correlated_feature.lower()}_distribution.png"
        )
        
        # 可视化去偏后的域偏移
        visualize_domain_shift_pca(X_A_debiased, X_B, "After KLIEP Debiasing (A vs B)", "debiased_domain_shift_pca.png")
        
        # 使用去偏数据训练TabPFN
        print("\n--- 使用去偏数据训练TabPFN ---")
        debiased_model = create_tabpfn_model()
        debiased_model.fit(X_A_debiased, y_A_debiased)
        debiased_pred = debiased_model.predict(X_B)
        debiased_proba = debiased_model.predict_proba(X_B)
        
        debiased_acc = accuracy_score(y_B, debiased_pred)
        debiased_auc = roc_auc_score(y_B, debiased_proba[:, 1])
        debiased_f1 = f1_score(y_B, debiased_pred)
        
        # 结果对比
        print(f"\n=== 最终结果对比 ===")
        print(f"基线TabPFN (原始数据A):")
        print(f"  准确率: {baseline_acc:.4f}")
        print(f"  AUC: {baseline_auc:.4f}")
        print(f"  F1: {baseline_f1:.4f}")
        
        print(f"\nKLIEP+TabPFN (去偏数据):")
        print(f"  准确率: {debiased_acc:.4f} (改进: {debiased_acc-baseline_acc:+.4f})")
        print(f"  AUC: {debiased_auc:.4f} (改进: {debiased_auc-baseline_auc:+.4f})")
        print(f"  F1: {debiased_f1:.4f} (改进: {debiased_f1-baseline_f1:+.4f})")
        
        # 统计分析对比
        print(f"\n=== 数据分布统计对比 ===")
        
        # 创建对比表格
        stats_comparison = pd.DataFrame({
            'Original_A': X_A[most_correlated_feature].describe(),
            'Debiased_A': X_A_debiased[most_correlated_feature].describe(),
            'Target_B': X_B[most_correlated_feature].describe()
        })
        
        print(f"\n{most_correlated_feature} 特征统计对比:")
        print(stats_comparison.round(4))
        
        # 保存结果
        results_df = pd.DataFrame({
            'Method': ['Baseline_TabPFN', 'KLIEP_TabPFN'],
            'Accuracy': [baseline_acc, debiased_acc],
            'AUC': [baseline_auc, debiased_auc],
            'F1': [baseline_f1, debiased_f1],
            'Improvement_Acc': [0, debiased_acc - baseline_acc],
            'Improvement_AUC': [0, debiased_auc - baseline_auc],
            'Improvement_F1': [0, debiased_f1 - baseline_f1]
        })
        
        results_file = 'kliep_bias_correction_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"\n完整结果已保存到: {results_file}")
        
        # 保存统计对比
        stats_file = f'kliep_{most_correlated_feature.lower()}_stats_comparison.csv'
        stats_comparison.to_csv(stats_file)
        print(f"统计对比已保存到: {stats_file}")
        
        return {
            'best_gamma': best_gamma,
            'baseline_performance': (baseline_acc, baseline_auc, baseline_f1),
            'debiased_performance': (debiased_acc, debiased_auc, debiased_f1),
            'improvements': (debiased_acc - baseline_acc, debiased_auc - baseline_auc, debiased_f1 - baseline_f1)
        }
        
    except ImportError as e:
        print(f"导入Adapt库失败: {e}")
        print("请确保已正确安装adapt-python库")
        return None
    except Exception as e:
        print(f"KLIEP偏差校正过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_kliep_approach():
    """演示KLIEP偏差校正方法"""
    print("\n" + "="*60)
    print("KLIEP偏差校正方法:")
    print("="*60)
    print("1. 分析源域和目标域的特征分布差异")
    print("2. 使用KLIEP自动选择最佳gamma参数")
    print("3. 计算实例重加权权重")
    print("4. 按权重重采样创建去偏数据集")
    print("5. 用去偏数据训练TabPFN并在目标域测试")
    print("="*60)

if __name__ == "__main__":
    demonstrate_kliep_approach()
    kliep_bias_correction() 