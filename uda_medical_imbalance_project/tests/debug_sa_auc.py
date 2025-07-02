#!/usr/bin/env python3
"""
调试SA方法的AUC计算问题
检查概率预测是否正确
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from tabpfn import TabPFNClassifier
import matplotlib.pyplot as plt

# 导入项目模块
from tests.test_adapt_methods import load_test_data, create_tabpfn_model, calculate_metrics
from uda.adapt_methods import create_adapt_method, is_adapt_available

def debug_sa_probability_prediction():
    """调试SA方法的概率预测"""
    print("=== 调试SA方法的概率预测问题 ===")
    
    if not is_adapt_available():
        print("Adapt库不可用")
        return
    
    # 加载数据
    X_A, y_A, X_B, y_B = load_test_data()
    print(f"数据加载完成: A{X_A.shape}, B{X_B.shape}")
    print(f"标签分布 - A: {dict(y_A.value_counts().sort_index())}")
    print(f"标签分布 - B: {dict(y_B.value_counts().sort_index())}")
    
    # 1. 基线TabPFN性能
    print("\n--- 基线TabPFN性能 ---")
    baseline_model = create_tabpfn_model()
    baseline_model.fit(X_A, y_A)
    baseline_pred = baseline_model.predict(X_B)
    baseline_proba = baseline_model.predict_proba(X_B)
    
    print(f"基线预测形状: {baseline_pred.shape}")
    print(f"基线概率形状: {baseline_proba.shape}")
    print(f"基线概率范围: [{baseline_proba.min():.4f}, {baseline_proba.max():.4f}]")
    print(f"基线概率[:, 1]范围: [{baseline_proba[:, 1].min():.4f}, {baseline_proba[:, 1].max():.4f}]")
    
    baseline_metrics = calculate_metrics(y_B, baseline_pred, baseline_proba)
    print(f"基线性能:")
    for metric, value in baseline_metrics.items():
        if not np.isnan(value):
            print(f"  {metric.upper()}: {value:.4f}")
    
    # 2. SA方法性能
    print("\n--- SA方法性能 ---")
    try:
        sa_method = create_adapt_method(
            method_name='SA',
            estimator=create_tabpfn_model(),
            n_components=None,
            verbose=0,
            random_state=42
        )
        
        # 拟合模型
        sa_method.fit(X_A, y_A, X_B)
        
        # 预测
        sa_pred = sa_method.predict(X_B.values)
        print(f"SA预测形状: {sa_pred.shape}")
        print(f"SA预测值: {np.unique(sa_pred, return_counts=True)}")
        
        # 尝试获取概率预测
        try:
            sa_proba = sa_method.predict_proba(X_B.values)
            if sa_proba is not None:
                print(f"SA概率形状: {sa_proba.shape}")
                print(f"SA概率范围: [{sa_proba.min():.4f}, {sa_proba.max():.4f}]")
                
                if sa_proba.ndim == 2 and sa_proba.shape[1] == 2:
                    print(f"SA概率[:, 0]范围: [{sa_proba[:, 0].min():.4f}, {sa_proba[:, 0].max():.4f}]")
                    print(f"SA概率[:, 1]范围: [{sa_proba[:, 1].min():.4f}, {sa_proba[:, 1].max():.4f}]")
                    print(f"SA概率行和检查: [{sa_proba.sum(axis=1).min():.4f}, {sa_proba.sum(axis=1).max():.4f}]")
                    
                    # 检查概率预测是否合理
                    prob_class1 = sa_proba[:, 1]
                    print(f"\n概率预测分析:")
                    print(f"  概率 > 0.5 的样本数: {(prob_class1 > 0.5).sum()}")
                    print(f"  概率 < 0.5 的样本数: {(prob_class1 < 0.5).sum()}")
                    print(f"  概率 = 0.5 的样本数: {(prob_class1 == 0.5).sum()}")
                    
                    # 检查预测标签与概率的一致性
                    pred_from_proba = (prob_class1 > 0.5).astype(int)
                    consistency = (sa_pred == pred_from_proba).mean()
                    print(f"  预测标签与概率一致性: {consistency:.4f}")
                    
                    # 手动计算AUC
                    manual_auc = roc_auc_score(y_B, prob_class1)
                    print(f"  手动计算AUC: {manual_auc:.4f}")
                    
                    # 检查是否存在概率反转问题
                    # 如果概率预测与实际标签完全相反，AUC会接近0
                    inverted_auc = roc_auc_score(y_B, 1 - prob_class1)
                    print(f"  反转概率AUC: {inverted_auc:.4f}")
                    
                    if inverted_auc > manual_auc:
                        print("  ⚠️  检测到概率可能被反转了！")
                        print("  建议使用 1 - prob_class1 作为正类概率")
                    
                else:
                    print(f"SA概率是1维数组: {sa_proba.shape}")
                    manual_auc = roc_auc_score(y_B, sa_proba)
                    print(f"  手动计算AUC: {manual_auc:.4f}")
            else:
                print("SA概率预测返回None")
                
        except Exception as e:
            print(f"获取SA概率预测失败: {e}")
            sa_proba = None
        
        # 计算指标
        sa_metrics = calculate_metrics(y_B, sa_pred, sa_proba)
        print(f"\nSA性能:")
        for metric, value in sa_metrics.items():
            if not np.isnan(value):
                print(f"  {metric.upper()}: {value:.4f}")
        
        # 详细分析混淆矩阵
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(y_B, sa_pred)
        print(f"\n混淆矩阵:")
        print(f"  真实\\预测  0    1")
        print(f"  0        {cm[0,0]:3d}  {cm[0,1]:3d}")
        print(f"  1        {cm[1,0]:3d}  {cm[1,1]:3d}")
        
        print(f"\n分类报告:")
        print(classification_report(y_B, sa_pred, target_names=['Class 0', 'Class 1']))
        
        # 3. 深入分析SA方法内部
        print("\n--- SA方法内部分析 ---")
        try:
            # 检查SA方法的内部结构
            print(f"SA方法类型: {type(sa_method.adapt_model)}")
            print(f"SA方法属性: {dir(sa_method.adapt_model)}")
            
            if hasattr(sa_method.adapt_model, 'estimator_'):
                estimator = sa_method.adapt_model.estimator_
                print(f"内部估计器类型: {type(estimator)}")
                print(f"内部估计器属性: {[attr for attr in dir(estimator) if not attr.startswith('_')]}")
                
                # 直接使用内部估计器预测
                if hasattr(estimator, 'predict_proba'):
                    # 需要先变换特征
                    if hasattr(sa_method.adapt_model, 'transform'):
                        X_B_transformed = sa_method.adapt_model.transform(sa_method.scaler.transform(X_B))
                        direct_proba = estimator.predict_proba(X_B_transformed)
                        print(f"直接使用内部估计器的概率预测形状: {direct_proba.shape}")
                        print(f"直接概率[:, 1]范围: [{direct_proba[:, 1].min():.4f}, {direct_proba[:, 1].max():.4f}]")
                        
                        direct_auc = roc_auc_score(y_B, direct_proba[:, 1])
                        print(f"直接计算AUC: {direct_auc:.4f}")
                        
        except Exception as e:
            print(f"SA内部分析失败: {e}")
        
        # 4. 可视化概率分布
        if sa_proba is not None and sa_proba.ndim == 2:
            print("\n--- 概率分布可视化 ---")
            try:
                plt.figure(figsize=(12, 4))
                
                # 子图1: 基线概率分布
                plt.subplot(1, 3, 1)
                plt.hist(baseline_proba[:, 1], bins=20, alpha=0.7, label='Baseline')
                plt.xlabel('Probability of Class 1')
                plt.ylabel('Frequency')
                plt.title(f'Baseline Probability Distribution\nAUC: {baseline_metrics["auc"]:.4f}')
                plt.legend()
                
                # 子图2: SA概率分布
                plt.subplot(1, 3, 2)
                plt.hist(sa_proba[:, 1], bins=20, alpha=0.7, label='SA', color='orange')
                plt.xlabel('Probability of Class 1')
                plt.ylabel('Frequency')
                plt.title(f'SA Probability Distribution\nAUC: {sa_metrics["auc"]:.4f}')
                plt.legend()
                
                # 子图3: 按真实标签分组的概率分布
                plt.subplot(1, 3, 3)
                class0_probs = sa_proba[y_B == 0, 1]
                class1_probs = sa_proba[y_B == 1, 1]
                plt.hist(class0_probs, bins=15, alpha=0.7, label='True Class 0', color='blue')
                plt.hist(class1_probs, bins=15, alpha=0.7, label='True Class 1', color='red')
                plt.xlabel('Probability of Class 1')
                plt.ylabel('Frequency')
                plt.title('SA Probability by True Class')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('sa_probability_debug.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("概率分布图已保存到: sa_probability_debug.png")
                
            except Exception as e:
                print(f"可视化失败: {e}")
        
    except Exception as e:
        print(f"SA方法测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_sa_probability_prediction() 