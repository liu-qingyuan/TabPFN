#!/usr/bin/env python3
"""
数据分布分析测试脚本

用于诊断跨域实验中测试集AUC低的问题，分析可能的原因：
1. 数据分布差异
2. 特征标准化问题
3. 类别不平衡
4. 特征选择问题
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config.settings import DATA_PATHS, LABEL_COL, get_features_by_type, get_categorical_indices

def load_datasets():
    """加载数据集A和B"""
    print("加载数据集...")
    
    # 加载数据集A (AI4Health)
    df_A = pd.read_excel(DATA_PATHS['A'])
    print(f"数据集A形状: {df_A.shape}")
    
    # 加载数据集B (Henan)
    df_B = pd.read_excel(DATA_PATHS['B'])
    print(f"数据集B形状: {df_B.shape}")
    
    return df_A, df_B

def analyze_basic_statistics(df_A, df_B, features):
    """分析基础统计信息"""
    print("\n" + "="*60)
    print("基础统计信息分析")
    print("="*60)
    
    # 提取特征和标签
    X_A = df_A[features]
    y_A = df_A[LABEL_COL]
    X_B = df_B[features]
    y_B = df_B[LABEL_COL]
    
    print(f"\n数据集A:")
    print(f"  样本数: {len(X_A)}")
    print(f"  特征数: {len(features)}")
    print(f"  标签分布: {dict(y_A.value_counts())}")
    print(f"  正样本比例: {y_A.mean():.3f}")
    
    print(f"\n数据集B:")
    print(f"  样本数: {len(X_B)}")
    print(f"  特征数: {len(features)}")
    print(f"  标签分布: {dict(y_B.value_counts())}")
    print(f"  正样本比例: {y_B.mean():.3f}")
    
    # 检查缺失值
    print(f"\n缺失值分析:")
    print(f"  数据集A缺失值: {X_A.isnull().sum().sum()}")
    print(f"  数据集B缺失值: {X_B.isnull().sum().sum()}")
    
    return X_A, y_A, X_B, y_B

def analyze_feature_distributions(X_A, X_B, features, categorical_indices):
    """分析特征分布差异"""
    print("\n" + "="*60)
    print("特征分布差异分析")
    print("="*60)
    
    # 分离数值特征和类别特征
    numerical_features = [f for i, f in enumerate(features) if i not in categorical_indices]
    categorical_features = [f for i, f in enumerate(features) if i in categorical_indices]
    
    print(f"\n数值特征数量: {len(numerical_features)}")
    print(f"类别特征数量: {len(categorical_features)}")
    
    # 分析数值特征
    if numerical_features:
        print(f"\n数值特征分布差异:")
        for feature in numerical_features:
            if feature in X_A.columns and feature in X_B.columns:
                # 计算统计量
                mean_A = X_A[feature].mean()
                std_A = X_A[feature].std()
                mean_B = X_B[feature].mean()
                std_B = X_B[feature].std()
                
                # 计算差异
                mean_diff = abs(mean_B - mean_A) / (std_A + 1e-8)
                std_ratio = std_B / (std_A + 1e-8)
                
                # KS检验
                ks_stat, ks_pval = stats.ks_2samp(X_A[feature].dropna(), X_B[feature].dropna())
                
                print(f"  {feature}:")
                print(f"    均值差异(标准化): {mean_diff:.3f}")
                print(f"    标准差比例: {std_ratio:.3f}")
                print(f"    KS统计量: {ks_stat:.3f} (p={ks_pval:.3e})")
                
                if ks_pval < 0.001:
                    print(f"    ⚠️  分布显著不同!")
    
    # 分析类别特征
    if categorical_features:
        print(f"\n类别特征分布差异:")
        for feature in categorical_features:
            if feature in X_A.columns and feature in X_B.columns:
                # 计算类别分布
                dist_A = X_A[feature].value_counts(normalize=True).sort_index()
                dist_B = X_B[feature].value_counts(normalize=True).sort_index()
                
                # 计算JS散度
                all_categories = set(dist_A.index) | set(dist_B.index)
                p = np.array([dist_A.get(cat, 0) for cat in all_categories])
                q = np.array([dist_B.get(cat, 0) for cat in all_categories])
                
                # JS散度计算
                m = 0.5 * (p + q)
                js_div = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
                
                print(f"  {feature}:")
                print(f"    JS散度: {js_div:.3f}")
                print(f"    A域类别数: {len(dist_A)}")
                print(f"    B域类别数: {len(dist_B)}")
                
                if js_div > 0.1:
                    print(f"    ⚠️  类别分布差异较大!")

def analyze_standardization_effect(X_A, X_B, features, categorical_indices):
    """分析标准化的效果"""
    print("\n" + "="*60)
    print("标准化效果分析")
    print("="*60)
    
    # 分离数值特征
    numerical_indices = [i for i in range(len(features)) if i not in categorical_indices]
    
    if not numerical_indices:
        print("没有数值特征需要标准化")
        return None, None
    
    # 原始数据
    X_A_num = X_A.iloc[:, numerical_indices].values
    X_B_num = X_B.iloc[:, numerical_indices].values
    
    # 标准化 (用A拟合)
    scaler = StandardScaler()
    X_A_scaled = scaler.fit_transform(X_A_num)
    X_B_scaled = scaler.transform(X_B_num)
    
    print(f"\n标准化前:")
    print(f"  A域数值特征均值范围: [{X_A_num.mean(axis=0).min():.3f}, {X_A_num.mean(axis=0).max():.3f}]")
    print(f"  A域数值特征标准差范围: [{X_A_num.std(axis=0).min():.3f}, {X_A_num.std(axis=0).max():.3f}]")
    print(f"  B域数值特征均值范围: [{X_B_num.mean(axis=0).min():.3f}, {X_B_num.mean(axis=0).max():.3f}]")
    print(f"  B域数值特征标准差范围: [{X_B_num.std(axis=0).min():.3f}, {X_B_num.std(axis=0).max():.3f}]")
    
    print(f"\n标准化后:")
    print(f"  A域数值特征均值范围: [{X_A_scaled.mean(axis=0).min():.3f}, {X_A_scaled.mean(axis=0).max():.3f}]")
    print(f"  A域数值特征标准差范围: [{X_A_scaled.std(axis=0).min():.3f}, {X_A_scaled.std(axis=0).max():.3f}]")
    print(f"  B域数值特征均值范围: [{X_B_scaled.mean(axis=0).min():.3f}, {X_B_scaled.mean(axis=0).max():.3f}]")
    print(f"  B域数值特征标准差范围: [{X_B_scaled.std(axis=0).min():.3f}, {X_B_scaled.std(axis=0).max():.3f}]")
    
    # 计算标准化后的分布差异
    mean_diffs_before = np.abs(X_A_num.mean(axis=0) - X_B_num.mean(axis=0))
    mean_diffs_after = np.abs(X_A_scaled.mean(axis=0) - X_B_scaled.mean(axis=0))
    
    print(f"\n均值差异改善:")
    print(f"  标准化前平均差异: {mean_diffs_before.mean():.3f}")
    print(f"  标准化后平均差异: {mean_diffs_after.mean():.3f}")
    print(f"  改善比例: {(1 - mean_diffs_after.mean()/mean_diffs_before.mean())*100:.1f}%")
    
    return X_A_scaled, X_B_scaled

def analyze_domain_gap(X_A, X_B, y_A, y_B):
    """分析域差距"""
    print("\n" + "="*60)
    print("域差距分析")
    print("="*60)
    
    # 合并数据进行PCA分析
    X_combined = np.vstack([X_A, X_B])
    domain_labels = np.hstack([np.zeros(len(X_A)), np.ones(len(X_B))])
    
    # PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_combined)
    
    print(f"PCA解释方差比例: {pca.explained_variance_ratio_}")
    print(f"前2个主成分累计解释方差: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 计算域间距离
    centroid_A = X_pca[domain_labels == 0].mean(axis=0)
    centroid_B = X_pca[domain_labels == 1].mean(axis=0)
    domain_distance = np.linalg.norm(centroid_A - centroid_B)
    
    print(f"域间质心距离 (PCA空间): {domain_distance:.3f}")
    
    # 分析类别条件分布
    print(f"\n类别条件分析:")
    for class_label in [0, 1]:
        A_class_mask = (domain_labels == 0) & (np.hstack([y_A, y_B]) == class_label)
        B_class_mask = (domain_labels == 1) & (np.hstack([y_A, y_B]) == class_label)
        
        if A_class_mask.sum() > 0 and B_class_mask.sum() > 0:
            centroid_A_class = X_pca[A_class_mask].mean(axis=0)
            centroid_B_class = X_pca[B_class_mask].mean(axis=0)
            class_distance = np.linalg.norm(centroid_A_class - centroid_B_class)
            
            print(f"  类别{class_label}域间距离: {class_distance:.3f}")
    
    return X_pca, domain_labels

def analyze_feature_importance_stability(X_A, y_A, X_B, y_B, features):
    """分析特征重要性的稳定性"""
    print("\n" + "="*60)
    print("特征重要性稳定性分析")
    print("="*60)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        # 随机森林特征重要性
        rf_A = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_A.fit(X_A, y_A)
        importance_A = rf_A.feature_importances_
        
        rf_B = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_B.fit(X_B, y_B)
        importance_B = rf_B.feature_importances_
        
        # 计算重要性相关性
        importance_corr = np.corrcoef(importance_A, importance_B)[0, 1]
        print(f"特征重要性相关性 (RF): {importance_corr:.3f}")
        
        # 互信息
        mi_A = mutual_info_classif(X_A, y_A, random_state=42)
        mi_B = mutual_info_classif(X_B, y_B, random_state=42)
        mi_corr = np.corrcoef(mi_A, mi_B)[0, 1]
        print(f"特征重要性相关性 (MI): {mi_corr:.3f}")
        
        # 打印top特征对比
        print(f"\nTop 5 重要特征对比:")
        top_A = np.argsort(importance_A)[-5:][::-1]
        top_B = np.argsort(importance_B)[-5:][::-1]
        
        print("数据集A Top 5:", [features[i] for i in top_A])
        print("数据集B Top 5:", [features[i] for i in top_B])
        
        # 计算重叠度
        overlap = len(set(top_A) & set(top_B))
        print(f"Top 5特征重叠数: {overlap}/5")
        
        if importance_corr < 0.5:
            print("⚠️  特征重要性在两个域间差异很大!")
        
    except ImportError:
        print("sklearn不可用，跳过特征重要性分析")

def generate_diagnostic_report(feature_type='best7'):
    """生成诊断报告"""
    print("="*80)
    print("跨域性能诊断报告")
    print("="*80)
    
    # 加载数据
    df_A, df_B = load_datasets()
    
    # 获取特征
    features = get_features_by_type(feature_type)
    categorical_indices = get_categorical_indices(feature_type)
    
    print(f"\n使用特征类型: {feature_type}")
    print(f"特征列表: {features}")
    print(f"类别特征索引: {categorical_indices}")
    
    # 基础统计分析
    X_A, y_A, X_B, y_B = analyze_basic_statistics(df_A, df_B, features)
    
    # 特征分布分析
    analyze_feature_distributions(X_A, X_B, features, categorical_indices)
    
    # 标准化效果分析
    X_A_scaled, X_B_scaled = analyze_standardization_effect(X_A, X_B, features, categorical_indices)
    
    # 域差距分析
    if X_A_scaled is not None and X_B_scaled is not None:
        # 重新组合标准化后的数据
        X_A_combined = X_A.copy()
        X_B_combined = X_B.copy()
        
        # 只替换数值特征
        numerical_features = [f for i, f in enumerate(features) if i not in categorical_indices]
        if numerical_features:
            X_A_combined[numerical_features] = X_A_scaled
            X_B_combined[numerical_features] = X_B_scaled
        
        analyze_domain_gap(X_A_combined.values, X_B_combined.values, y_A, y_B)
    else:
        analyze_domain_gap(X_A.values, X_B.values, y_A, y_B)
    
    # 特征重要性分析
    analyze_feature_importance_stability(X_A, y_A, X_B, y_B, features)
    
    # 生成建议
    print("\n" + "="*60)
    print("改进建议")
    print("="*60)
    
    print("\n基于分析结果的建议:")
    print("1. 🔧 确保数据标准化: 用A域数据拟合StandardScaler，然后应用到B域")
    print("2. 📊 检查类别不平衡: 考虑使用class_weight='balanced'")
    print("3. 🎯 特征选择: 选择在两个域间都稳定的特征")
    print("4. 🔄 域适应: 考虑使用MMD等域适应方法")
    print("5. 📈 模型选择: 尝试不同的模型类型和超参数")
    print("6. 🎲 数据增强: 考虑对少数类进行过采样")
    
    # 具体的数值建议
    pos_ratio_A = y_A.mean()
    pos_ratio_B = y_B.mean()
    ratio_diff = abs(pos_ratio_A - pos_ratio_B)
    
    if ratio_diff > 0.1:
        print(f"\n⚠️  类别分布差异较大 (A域: {pos_ratio_A:.3f}, B域: {pos_ratio_B:.3f})")
        print("   建议: 使用分层采样或调整class_weight")
    
    if len(X_A) < 500:
        print(f"\n⚠️  训练样本较少 (A域: {len(X_A)}样本)")
        print("   建议: 考虑数据增强或使用更简单的模型")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据分布分析诊断')
    parser.add_argument('--feature-type', type=str, default='best7',
                       choices=['all', 'best7'], help='特征类型')
    
    args = parser.parse_args()
    
    try:
        generate_diagnostic_report(args.feature_type)
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 