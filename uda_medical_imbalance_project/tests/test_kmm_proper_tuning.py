#!/usr/bin/env python3
"""
正确的KMM参数调优测试
参考: https://adapt-python.github.io/adapt/examples/Sample_bias_example.html
使用无监督评估指标进行参数选择
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from tabpfn import TabPFNClassifier
from typing import Optional

# 导入项目模块
from tests.test_adapt_methods import load_test_data
from uda.adapt_methods import is_adapt_available

def create_tabpfn_model():
    """创建TabPFN模型"""
    return TabPFNClassifier(n_estimators=32)

def visualize_domain_shift(X_source: pd.DataFrame, X_target: pd.DataFrame, title: str = "Domain Shift Visualization", save_path: Optional[str] = None):
    """可视化域偏移并保存为文件"""
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
    
    # 保存图像
    if save_path is None:
        save_path = f"domain_shift_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存到: {save_path}")

def proper_kmm_tuning():
    """正确的KMM参数调优 - 使用无监督评估"""
    print("=== 正确的KMM参数调优（无监督） ===")
    
    if not is_adapt_available():
        print("Adapt库不可用")
        return
    
    # 加载数据
    X_A, y_A, X_B, y_B = load_test_data()
    print(f"数据加载完成: A{X_A.shape}, B{X_B.shape}")
    
    # 可视化域偏移
    print("\n--- 域偏移可视化 ---")
    visualize_domain_shift(X_A, X_B, "Original Domain Shift (A vs B)", "original_domain_shift.png")
    
    # 基线性能（仅用于最终对比，不用于调参）
    print("\n--- 基线TabPFN性能 ---")
    baseline_model = create_tabpfn_model()
    baseline_model.fit(X_A, y_A)
    baseline_pred = baseline_model.predict(X_B)
    baseline_proba = baseline_model.predict_proba(X_B)
    
    baseline_acc = accuracy_score(y_B, baseline_pred)
    baseline_auc = roc_auc_score(y_B, baseline_proba[:, 1])
    baseline_f1 = f1_score(y_B, baseline_pred)
    
    print(f"基线 - 准确率: {baseline_acc:.4f}, AUC: {baseline_auc:.4f}, F1: {baseline_f1:.4f}")
    
    # 使用Adapt库进行无监督参数调优
    try:
        import adapt.metrics
        import importlib
        importlib.reload(adapt.metrics)
        from adapt.metrics import make_uda_scorer, neg_j_score
        from adapt.instance_based import KMM
        
        print("\n--- 无监督参数调优 ---")
        
        # 按照官方示例创建KMM模型
        kmm = KMM(
            estimator=create_tabpfn_model(),
            Xt=X_B.values,  
            kernel="rbf",  # Gaussian kernel
            gamma=0.,  # 初始值，会被网格搜索替换
            verbose=0,
            random_state=42
        )
        
        # 创建无监督评分函数
        score = make_uda_scorer(neg_j_score, X_A.values, X_B.values)
        
        # 网格搜索（按照官方示例）
        print("开始无监督网格搜索...")
        gs = GridSearchCV(
            kmm, 
            {"gamma": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1., 10., 100., 1000.]},
            scoring=score,
            return_train_score=True,  # 关键：需要返回训练分数
            cv=5, 
            verbose=1
        )
        
        # 拟合
        gs.fit(X_A.values, y_A.values)
        
        # 按照官方示例打印结果（neg_j_score在train_score中）
        keys = ["params", 'mean_train_score', 'std_train_score']
        results = [v for k, v in gs.cv_results_.items() if k in keys]
        best = results[1].argmin()  # neg_j_score越小越好
        
        print("Best Params %s -- Score %.6f (%.6f)" %
              (str(results[0][best]), results[1][best], results[2][best]))
        print("-" * 50)
        for p, mu, std in zip(*results):
            print("Params %s -- Score %.6f (%.6f)" % (str(p), mu, std))
        
        best_gamma = results[0][best]['gamma']
        best_score = results[1][best]
        
        print(f"\n最佳参数: gamma = {best_gamma}")
        print(f"最佳J-score: {best_score:.6f}")
        
        # 使用最佳参数创建最终模型（用TabPFN替换Ridge）
        print("\n--- 使用最佳参数测试性能 ---")
        
        best_kmm = KMM(
            estimator=create_tabpfn_model(),  
            Xt=X_B.values,
            kernel="rbf",
            gamma=best_gamma,
            verbose=0,
            random_state=42
        )
        
        # 拟合和预测
        best_kmm.fit(X_A.values, y_A.values)
        kmm_pred = best_kmm.predict(X_B.values)
        
        # 尝试获取概率预测
        try:
            if hasattr(best_kmm, 'predict_proba'):
                kmm_proba = best_kmm.predict_proba(X_B.values)
            elif hasattr(best_kmm, 'estimator_') and hasattr(best_kmm.estimator_, 'predict_proba'):
                kmm_proba = best_kmm.estimator_.predict_proba(X_B.values)
            else:
                kmm_proba = None
        except:
            kmm_proba = None
        
        # 计算性能指标
        kmm_acc = accuracy_score(y_B, kmm_pred)
        kmm_f1 = f1_score(y_B, kmm_pred)
        
        if kmm_proba is not None:
            if kmm_proba.ndim == 2 and kmm_proba.shape[1] == 2:
                kmm_auc = roc_auc_score(y_B, kmm_proba[:, 1])
            else:
                kmm_auc = roc_auc_score(y_B, kmm_proba)
        else:
            kmm_auc = np.nan
        
        print(f"\n=== 最终结果对比 ===")
        print(f"基线TabPFN:")
        print(f"  准确率: {baseline_acc:.4f}")
        print(f"  AUC: {baseline_auc:.4f}")
        print(f"  F1: {baseline_f1:.4f}")
        
        print(f"\nKMM+TabPFN (最佳参数 gamma={best_gamma}):")
        print(f"  准确率: {kmm_acc:.4f} (改进: {kmm_acc-baseline_acc:+.4f})")
        if not np.isnan(kmm_auc):
            print(f"  AUC: {kmm_auc:.4f} (改进: {kmm_auc-baseline_auc:+.4f})")
        else:
            print(f"  AUC: N/A")
        print(f"  F1: {kmm_f1:.4f} (改进: {kmm_f1-baseline_f1:+.4f})")
        
        # 可视化重加权后的效果
        try:
            # 获取重加权权重
            weights_kmm = KMM(
                estimator=create_tabpfn_model(),
                Xs=X_A.values,
                ys=y_A.values,
                Xt=X_B.values,
                kernel="rbf",
                gamma=best_gamma,
                verbose=0,
                epochs=100, 
                batch_size=100,
                random_state=42
            )
            weights = weights_kmm.fit_weights(X_A.values, X_B.values)
            
            # 创建重加权后的源域数据可视化
            print("\n--- 重加权效果可视化 ---")
            
            # 按权重重采样源域数据
            reweighted_indices = np.random.choice(
                X_A.index, 
                size=len(X_A), 
                p=weights/weights.sum()
            )
            X_A_reweighted = X_A.loc[reweighted_indices]
            
            # 可视化重加权后的域偏移
            reweight_filename = f"kmm_reweighted_gamma_{best_gamma}.png"
            visualize_domain_shift(X_A_reweighted, X_B, 
                                 f"After KMM Reweighting (gamma={best_gamma})",
                                 reweight_filename)
            
        except Exception as e:
            print(f"重加权可视化失败: {e}")
        
        # 保存结果
        results_file = 'kmm_proper_tuning_results.csv'
        results_df = pd.DataFrame({
            'gamma': [p['gamma'] for p in results[0]],
            'mean_train_score': results[1],
            'std_train_score': results[2]
        })
        results_df.to_csv(results_file, index=False)
        print(f"\n完整结果已保存到: {results_file}")
        
        return best_gamma, best_score
        
    except ImportError as e:
        print(f"导入Adapt库失败: {e}")
        print("请确保已正确安装adapt-python库")
        return None, None
    except Exception as e:
        print(f"参数调优过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def demonstrate_proper_approach():
    """演示正确的域适应调参方法"""
    print("\n" + "="*60)
    print("正确的域适应参数调优方法:")
    print("="*60)
    print("1. 使用无监督评估指标（如J-score）")
    print("2. 只使用源域标签，不使用目标域标签")
    print("3. 评估重加权后源域与目标域分布的匹配程度")
    print("4. 选择使分布匹配最好的参数")
    print("5. 最后用最佳参数在目标域上测试（仅用于验证）")
    print("="*60)

if __name__ == "__main__":
    demonstrate_proper_approach()
    proper_kmm_tuning() 