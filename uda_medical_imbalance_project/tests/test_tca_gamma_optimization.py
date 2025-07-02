"""
TCA gamma参数优化测试
演示如何为TCA方法手动进行gamma参数搜索
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from uda.adapt_methods import create_adapt_method
    from test_adapt_methods import load_test_data, create_tabpfn_model, calculate_metrics
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"依赖模块导入失败: {e}")
    DEPENDENCIES_AVAILABLE = False


def manual_tca_gamma_optimization():
    """手动为TCA进行gamma参数优化"""
    
    if not DEPENDENCIES_AVAILABLE:
        print("跳过TCA gamma优化测试：缺少必要依赖")
        return
        
    print("=== TCA 完整参数优化测试 ===")
    
    # 加载数据
    X_A, y_A, X_B, y_B = load_test_data()
    print(f"数据集A: {X_A.shape}, 数据集B: {X_B.shape}")
    
    # 定义搜索范围
    gamma_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    mu_values = [0.01, 0.1, 1.0, 10.0]  # TCA的关键参数
    n_components_values = [5, 10, 15, 20]  # 子空间维度
    
    results = {}
    
    print("\n=== 1. Linear核参数优化 ===")
    print("优化参数: mu, n_components")
    
    linear_results = {}
    best_linear_auc = 0
    best_linear_config = None
    
    for mu in mu_values:
        for n_comp in n_components_values:
            try:
                tca = create_adapt_method(
                    method_name='TCA',
                    estimator=create_tabpfn_model(),
                    kernel='linear',
                    mu=mu,
                    n_components=n_comp,
                    random_state=0,
                    verbose=0
                )
                
                tca.fit(X_A, y_A, X_B)
                y_pred = tca.predict(X_B)
                try:
                    y_pred_proba = tca.predict_proba(X_B)
                except:
                    y_pred_proba = None
                
                metrics = calculate_metrics(y_B, y_pred, y_pred_proba)
                
                config_key = f"mu={mu}_comp={n_comp}"
                linear_results[config_key] = {
                    'mu': mu,
                    'n_components': n_comp,
                    'metrics': metrics
                }
                
                auc = metrics['auc']
                print(f"  mu={mu:4.2f}, n_comp={n_comp:2d} - AUC: {auc:.4f}, Acc: {metrics['accuracy']:.4f}")
                
                if not np.isnan(auc) and auc > best_linear_auc:
                    best_linear_auc = auc
                    best_linear_config = linear_results[config_key]
                
            except Exception as e:
                print(f"  mu={mu:4.2f}, n_comp={n_comp:2d} - 失败: {e}")
    
    results['linear'] = linear_results
    
    print(f"\nLinear核最佳配置: mu={best_linear_config['mu']}, n_components={best_linear_config['n_components']}")
    print(f"最佳AUC: {best_linear_config['metrics']['auc']:.4f}")
    
    print("\n=== 2. RBF核参数优化 ===")
    print("优化参数: gamma, mu, n_components")
    
    rbf_results = {}
    best_rbf_auc = 0
    best_rbf_config = None
    
    # 为了节省时间，只测试几个重要的组合
    selected_combinations = [
        (0.001, 0.1, 10),
        (0.01, 0.1, 10), 
        (0.1, 0.1, 10),
        (1.0, 0.1, 10),
        (0.1, 1.0, 10),
        (0.1, 0.1, 5),
        (0.1, 0.1, 15)
    ]
    
    for gamma, mu, n_comp in selected_combinations:
        try:
            tca = create_adapt_method(
                method_name='TCA',
                estimator=create_tabpfn_model(),
                kernel='rbf',
                gamma=gamma,
                mu=mu,
                n_components=n_comp,
                random_state=0,
                verbose=0
            )
            
            tca.fit(X_A, y_A, X_B)
            y_pred = tca.predict(X_B)
            try:
                y_pred_proba = tca.predict_proba(X_B)
            except:
                y_pred_proba = None
            
            metrics = calculate_metrics(y_B, y_pred, y_pred_proba)
            
            config_key = f"gamma={gamma}_mu={mu}_comp={n_comp}"
            rbf_results[config_key] = {
                'gamma': gamma,
                'mu': mu,
                'n_components': n_comp,
                'metrics': metrics
            }
            
            auc = metrics['auc']
            print(f"  γ={gamma:5.3f}, μ={mu:4.2f}, comp={n_comp:2d} - AUC: {auc:.4f}, Acc: {metrics['accuracy']:.4f}")
            
            if not np.isnan(auc) and auc > best_rbf_auc:
                best_rbf_auc = auc
                best_rbf_config = rbf_results[config_key]
                
        except Exception as e:
            print(f"  γ={gamma:5.3f}, μ={mu:4.2f}, comp={n_comp:2d} - 失败: {e}")
    
    results['rbf'] = rbf_results
    
    if best_rbf_config:
        print(f"\nRBF核最佳配置: gamma={best_rbf_config['gamma']}, mu={best_rbf_config['mu']}, n_components={best_rbf_config['n_components']}")
        print(f"最佳AUC: {best_rbf_config['metrics']['auc']:.4f}")
    
    # 全局最佳对比
    print("\n=== 3. 全局最佳配置 ===")
    
    global_best_auc = max(best_linear_auc, best_rbf_auc if best_rbf_config else 0)
    
    if global_best_auc == best_linear_auc:
        print("✓ Linear核获得最佳性能")
        global_best = best_linear_config
        global_kernel = 'linear'
    else:
        print("✓ RBF核获得最佳性能")  
        global_best = best_rbf_config
        global_kernel = 'rbf'
    
    print(f"最佳核函数: {global_kernel}")
    if global_kernel == 'rbf':
        print(f"最佳gamma: {global_best['gamma']}")
    print(f"最佳mu: {global_best['mu']}")
    print(f"最佳n_components: {global_best['n_components']}")
    print(f"最佳AUC: {global_best['metrics']['auc']:.4f}")
    print(f"最佳准确率: {global_best['metrics']['accuracy']:.4f}")
    
    # 与默认配置对比
    print("\n=== 4. 与默认配置对比 ===")
    try:
        # 默认配置 (当前测试中使用的)
        default_tca = create_adapt_method(
            method_name='TCA',
            estimator=create_tabpfn_model(),
            kernel='linear',
            mu=1.0,  # 当前测试默认值
            n_components=20,  # TCA默认值
            random_state=0,
            verbose=0
        )
        
        default_tca.fit(X_A, y_A, X_B)
        default_pred = default_tca.predict(X_B)
        try:
            default_proba = default_tca.predict_proba(X_B)
        except:
            default_proba = None
        
        default_metrics = calculate_metrics(y_B, default_pred, default_proba)
        
        print(f"默认配置 (linear, mu=1.0, n_comp=20): AUC={default_metrics['auc']:.4f}")
        print(f"优化配置: AUC={global_best['metrics']['auc']:.4f}")
        
        auc_improvement = global_best['metrics']['auc'] - default_metrics['auc']
        print(f"AUC改进: {auc_improvement:+.4f}")
        
        if auc_improvement > 0.01:
            print("✓ 参数优化带来了显著改进！")
        elif auc_improvement > 0:
            print("→ 参数优化带来了轻微改进")
        else:
            print("→ 默认配置表现更好或相当")
            
    except Exception as e:
        print(f"默认配置对比失败: {e}")
        
    return results


def demonstrate_tca_vs_ulsif_optimization():
    """对比TCA和ULSIF的参数优化机制"""
    
    if not DEPENDENCIES_AVAILABLE:
        print("跳过对比测试：缺少必要依赖")
        return
        
    print("\n=== TCA vs ULSIF 参数优化对比 ===")
    
    X_A, y_A, X_B, y_B = load_test_data()
    
    # 1. TCA手动优化
    print("\n1. TCA (手动优化):")
    print("   - 需要用户手动搜索gamma参数")
    print("   - 没有内置的无监督评分机制")
    print("   - 适合子空间维度较低的情况")
    
    # 2. ULSIF自动优化
    print("\n2. ULSIF (内置优化):")
    print("   - 内置Leave-One-Out交叉验证")
    print("   - 自动选择最优gamma和lambda")
    print("   - 基于J-score进行无监督参数选择")
    
    try:
        # 演示ULSIF的自动优化
        ulsif = create_adapt_method(
            method_name='ULSIF',
            estimator=create_tabpfn_model(),
            gamma=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            lambdas=[0.001, 0.01, 0.1, 1.0, 10.0],
            kernel='rbf',
            max_centers=50,  # 减少计算量
            verbose=1,
            random_state=0
        )
        
        print("\nULSIF自动优化过程:")
        ulsif.fit(X_A, y_A, X_B)
        
        if hasattr(ulsif.adapt_model, 'best_params_'):
            best_params = ulsif.adapt_model.best_params_
            print(f"ULSIF最佳参数: {best_params}")
        
        ulsif_pred = ulsif.predict(X_B)
        try:
            ulsif_proba = ulsif.predict_proba(X_B)
        except:
            ulsif_proba = None
            
        ulsif_metrics = calculate_metrics(y_B, ulsif_pred, ulsif_proba)
        print(f"ULSIF优化后性能: AUC={ulsif_metrics['auc']:.4f}")
        
    except Exception as e:
        print(f"ULSIF测试失败: {e}")
    
    print("\n总结:")
    print("- TCA: 更适合确定性的特征对齐，参数相对稳健")
    print("- ULSIF: 更依赖参数调优，但有内置优化机制")
    print("- 建议: TCA优先尝试linear核，必要时手动调gamma")


if __name__ == "__main__":
    manual_tca_gamma_optimization()
    demonstrate_tca_vs_ulsif_optimization() 