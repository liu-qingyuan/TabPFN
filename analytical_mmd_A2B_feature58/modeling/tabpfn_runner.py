import numpy as np
import time
import os
import logging
from sklearn.model_selection import train_test_split
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Tuple

# 导入 MMD 相关模块
from ..preprocessing.mmd import mmd_transform, compute_mmd, compute_multiple_kernels_mmd
from ..metrics.evaluation import evaluate_metrics, optimize_threshold
from ..visualization.visualize_analytical_mmd_tsne import (
    visualize_tsne,
    visualize_feature_histograms,
    visualize_mmd_adaptation_results
)

def run_mmd_adaptation_experiment(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    cat_idx: Optional[List[int]] = None,
    method: str = 'linear',
    model_name: str = 'TabPFN-MMD',
    tabpfn_params: Optional[Dict[str, Any]] = None,
    base_path: str = './results_analytical_mmd_A2B',
    optimize_decision_threshold: bool = True,
    feature_names_for_plot: Optional[List[str]] = None,
    **mmd_kwargs
) -> Dict[str, Any]:
    """
    运行带有MMD域适应的TabPFN实验
    
    参数:
    - X_source: 源域特征
    - y_source: 源域标签
    - X_target: 目标域特征
    - y_target: 目标域标签
    - cat_idx: 类别特征索引列表
    - method: MMD对齐方法 ('linear', 'kpca', 'mean_std')
    - model_name: 模型名称
    - tabpfn_params: TabPFN参数
    - base_path: 结果保存路径
    - optimize_decision_threshold: 是否优化决策阈值
    - feature_names_for_plot: 特征名称列表，用于绘图
    - **mmd_kwargs: 传递给MMD方法的其他参数
    
    返回:
    - 评估指标字典
    """
    if tabpfn_params is None:
        tabpfn_params = {'device': 'cuda', 'max_time': 60, 'random_state': 42}
    
    if cat_idx is None:
        cat_idx = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22]  # TabPFN默认类别特征索引

    logging.info(f"\n=== {model_name} Model (MMD-{method.upper()}) ===")
    
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 数据标准化
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # 分析源域和目标域的特征分布差异
    logging.info("分析对齐前的域差异...")
    source_mean = np.mean(X_source_scaled, axis=0)
    target_mean = np.mean(X_target_scaled, axis=0)
    source_std = np.std(X_source_scaled, axis=0)
    target_std = np.std(X_target_scaled, axis=0)
    
    mean_diff = np.mean(np.abs(source_mean - target_mean))
    std_diff = np.mean(np.abs(source_std - target_std))
    logging.info(f"初始域差异: 均值差异={mean_diff:.6f}, 标准差差异={std_diff:.6f}")
    
    # 计算初始MMD
    initial_mmd = compute_mmd(X_source_scaled, X_target_scaled, kernel='rbf', gamma=1.0)
    logging.info(f"初始MMD: {initial_mmd:.6f}")
    
    # 在源域内划分训练集和测试集
    logging.info("将源域划分为训练集和验证集 (80/20)...")
    X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
        X_source_scaled, y_source, test_size=0.2, random_state=42, stratify=y_source
    )
    logging.info(f"源域 - 训练: {X_source_train.shape[0]} 样本, 验证: {X_source_val.shape[0]} 样本")
    
    # 初始化TabPFN模型
    logging.info("初始化TabPFN模型...")
    tabpfn_model = AutoTabPFNClassifier(**tabpfn_params)
    
    # 在源域训练数据上训练TabPFN
    logging.info("在源域训练数据上训练TabPFN...")
    start_time = time.time()
    tabpfn_model.fit(X_source_train, y_source_train)
    tabpfn_time = time.time() - start_time
    logging.info(f"TabPFN训练完成，耗时 {tabpfn_time:.2f} 秒")
    
    # 在源域验证集上评估TabPFN
    logging.info("\n在源域验证集上评估TabPFN...")
    y_source_val_pred = tabpfn_model.predict(X_source_val)
    y_source_val_proba = tabpfn_model.predict_proba(X_source_val)
    
    # 计算源域验证指标
    source_metrics = {
        'acc': accuracy_score(y_source_val, y_source_val_pred),
        'auc': roc_auc_score(y_source_val, y_source_val_proba[:, 1]),
        'f1': f1_score(y_source_val, y_source_val_pred)
    }
    
    # 计算混淆矩阵和每类准确率
    conf_matrix = confusion_matrix(y_source_val, y_source_val_pred)
    source_metrics['acc_0'] = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    source_metrics['acc_1'] = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    
    logging.info(f"源域验证 - 准确率: {source_metrics['acc']:.4f}, AUC: {source_metrics['auc']:.4f}, F1: {source_metrics['f1']:.4f}")
    logging.info(f"源域验证 - 类别0准确率: {source_metrics['acc_0']:.4f}, 类别1准确率: {source_metrics['acc_1']:.4f}")
    
    # 在目标域上进行直接评估（未对齐）
    logging.info("\n在目标域上直接评估TabPFN（未对齐）...")
    y_target_pred_direct = tabpfn_model.predict(X_target_scaled)
    y_target_proba_direct = tabpfn_model.predict_proba(X_target_scaled)
    
    # 分析预测分布
    unique_labels, counts = np.unique(y_target_pred_direct, return_counts=True)
    logging.info(f"直接预测分布: {dict(zip(unique_labels, counts))}")
    
    # 计算原始TabPFN在目标域上的性能
    direct_metrics = {
        'acc': accuracy_score(y_target, y_target_pred_direct),
        'auc': roc_auc_score(y_target, y_target_proba_direct[:, 1]),
        'f1': f1_score(y_target, y_target_pred_direct)
    }
    
    # 计算每类准确率
    conf_matrix = confusion_matrix(y_target, y_target_pred_direct)
    direct_metrics['acc_0'] = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    direct_metrics['acc_1'] = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    
    logging.info(f"直接预测 - 准确率: {direct_metrics['acc']:.4f}, AUC: {direct_metrics['auc']:.4f}, F1: {direct_metrics['f1']:.4f}")
    logging.info(f"直接预测 - 类别0准确率: {direct_metrics['acc_0']:.4f}, 类别1准确率: {direct_metrics['acc_1']:.4f}")
    
    # 使用MMD进行特征对齐
    logging.info(f"\n应用MMD变换 (方法: {method})...")
    start_time = time.time()
    X_target_aligned, mmd_info = mmd_transform(
        X_source_scaled, X_target_scaled, method=method, cat_idx=cat_idx, **mmd_kwargs
    )
    align_time = time.time() - start_time
    logging.info(f"MMD变换完成，耗时 {align_time:.2f} 秒")
    
    # 记录MMD信息
    logging.info(f"MMD对齐信息:")
    logging.info(f"  方法: {mmd_info['method']}")
    logging.info(f"  初始MMD: {mmd_info['initial_mmd']:.6f}")
    logging.info(f"  最终MMD: {mmd_info['final_mmd']:.6f}")
    logging.info(f"  MMD减少: {mmd_info['reduction']:.2f}%")
    
    # 分析对齐前后的特征差异
    mean_diff_after = np.mean(np.abs(np.mean(X_source_scaled, axis=0) - np.mean(X_target_aligned, axis=0)))
    std_diff_after = np.mean(np.abs(np.std(X_source_scaled, axis=0) - np.std(X_target_aligned, axis=0)))
    logging.info(f"对齐后: 均值差异={mean_diff_after:.6f}, 标准差差异={std_diff_after:.6f}")
    if mean_diff > 0:
        logging.info(f"差异减少: 均值: {(mean_diff-mean_diff_after)/mean_diff:.2%}, 标准差: {(std_diff-std_diff_after)/std_diff:.2%}")
    
    # 计算多个内核的MMD值（可选）
    if method in ['linear', 'kpca']:
        logging.info("\n计算多个内核的MMD值...")
        multi_mmd_before = compute_multiple_kernels_mmd(X_source_scaled, X_target_scaled)
        multi_mmd_after = compute_multiple_kernels_mmd(X_source_scaled, X_target_aligned)
        
        logging.info("对齐前的多内核MMD:")
        for kernel, value in multi_mmd_before.items():
            if not isinstance(value, str) and not np.isnan(value):
                logging.info(f"  {kernel}: {value:.6f}")
        
        logging.info("对齐后的多内核MMD:")
        for kernel, value in multi_mmd_after.items():
            if not isinstance(value, str) and not np.isnan(value):
                logging.info(f"  {kernel}: {value:.6f}")
    
    # 在目标域上进行评估
    logging.info(f"\n在目标域上评估模型（使用MMD-{method}对齐）...")
    
    # 目标域预测（使用MMD对齐）
    start_time = time.time()
    y_target_pred = tabpfn_model.predict(X_target_aligned)
    y_target_proba = tabpfn_model.predict_proba(X_target_aligned)
    inference_time = time.time() - start_time
    
    # 分析预测分布
    unique_labels, counts = np.unique(y_target_pred, return_counts=True)
    logging.info(f"MMD对齐后预测分布: {dict(zip(unique_labels, counts))}")
    
    # 计算目标域指标
    target_metrics = {
        'acc': accuracy_score(y_target, y_target_pred),
        'auc': roc_auc_score(y_target, y_target_proba[:, 1]),
        'f1': f1_score(y_target, y_target_pred)
    }
    
    # 计算混淆矩阵和每类准确率
    conf_matrix = confusion_matrix(y_target, y_target_pred)
    acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    
    target_metrics['acc_0'] = acc_0
    target_metrics['acc_1'] = acc_1
    
    optimal_threshold = 0.5  # 默认阈值
    # 优化决策阈值（可选）
    if optimize_decision_threshold:
        logging.info("\n使用Youden指数优化决策阈值...")
        optimal_threshold, optimal_metrics = optimize_threshold(y_target, y_target_proba[:, 1])
        
        logging.info(f"最优阈值: {optimal_threshold:.4f} (默认: 0.5)")
        logging.info(f"优化阈值后的指标:")
        logging.info(f"  准确率: {optimal_metrics['acc']:.4f} (原始: {target_metrics['acc']:.4f})")
        logging.info(f"  F1分数: {optimal_metrics['f1']:.4f} (原始: {target_metrics['f1']:.4f})")
        logging.info(f"  类别0准确率: {optimal_metrics['acc_0']:.4f} (原始: {target_metrics['acc_0']:.4f})")
        logging.info(f"  类别1准确率: {optimal_metrics['acc_1']:.4f} (原始: {target_metrics['acc_1']:.4f})")
        
        # 更新目标域指标为优化后的指标
        target_metrics.update(optimal_metrics)
    
    # 记录性能改进
    acc_improvement = target_metrics['acc'] - direct_metrics['acc']
    auc_improvement = target_metrics['auc'] - direct_metrics['auc']
    f1_improvement = target_metrics['f1'] - direct_metrics['f1']
    
    logging.info(f"\nMMD对齐性能改进:")
    logging.info(f"  准确率改进: {acc_improvement:+.4f} ({acc_improvement/direct_metrics['acc']:+.2%})")
    logging.info(f"  AUC改进: {auc_improvement:+.4f} ({auc_improvement/direct_metrics['auc']:+.2%})")
    logging.info(f"  F1改进: {f1_improvement:+.4f} ({f1_improvement/direct_metrics['f1']:+.2%})")
    
    # 生成可视化
    logging.info("\n生成可视化结果...")
    visualization_dir = os.path.join(base_path, f'MMD-{method}_visualizations')
    
    try:
        visualize_mmd_adaptation_results(
            X_source_scaled, X_target_scaled, X_target_aligned,
            source_labels=y_source, target_labels=y_target,
            output_dir=visualization_dir,
            feature_names=feature_names_for_plot,
            method_name=f'MMD-{method.upper()}'
        )
        logging.info(f"可视化结果已保存至: {visualization_dir}")
    except Exception as e:
        logging.warning(f"可视化生成失败: {str(e)}")
    
    # 保存ROC曲线
    try:
        plt.figure(figsize=(10, 8))
        
        # 直接预测的ROC曲线
        fpr_direct, tpr_direct, _ = roc_curve(y_target, y_target_proba_direct[:, 1])
        plt.plot(fpr_direct, tpr_direct, label=f'Direct (AUC = {direct_metrics["auc"]:.3f})', linewidth=2)
        
        # MMD对齐后的ROC曲线
        fpr_aligned, tpr_aligned, _ = roc_curve(y_target, y_target_proba[:, 1])
        plt.plot(fpr_aligned, tpr_aligned, label=f'MMD-{method.upper()} (AUC = {target_metrics["auc"]:.3f})', linewidth=2)
        
        # 对角线
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves Comparison - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        roc_path = os.path.join(base_path, f'roc_curves_comparison_{method}.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"ROC曲线已保存至: {roc_path}")
    except Exception as e:
        logging.warning(f"ROC曲线生成失败: {str(e)}")
    
    # 汇总结果
    results = {
        'model_name': model_name,
        'method': method,
        'source_metrics': source_metrics,
        'direct_metrics': direct_metrics,
        'target_metrics': target_metrics,
        'mmd_info': mmd_info,
        'optimal_threshold': optimal_threshold,
        'training_time': tabpfn_time,
        'alignment_time': align_time,
        'inference_time': inference_time,
        'improvements': {
            'acc': acc_improvement,
            'auc': auc_improvement,
            'f1': f1_improvement
        }
    }
    
    logging.info(f"\n=== {model_name} 实验完成 ===")
    logging.info(f"最终结果 - 准确率: {target_metrics['acc']:.4f}, AUC: {target_metrics['auc']:.4f}, F1: {target_metrics['f1']:.4f}")
    
    return results

def compare_mmd_methods(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    cat_idx: Optional[List[int]] = None,
    methods: List[str] = ['linear', 'kpca', 'mean_std'],
    base_path: str = './results_mmd_comparison',
    **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    比较不同MMD方法的性能
    
    参数:
    - X_source, y_source: 源域数据
    - X_target, y_target: 目标域数据
    - cat_idx: 类别特征索引
    - methods: 要比较的MMD方法列表
    - base_path: 结果保存路径
    - **kwargs: 传递给实验函数的其他参数
    
    返回:
    - 各方法的结果字典
    """
    logging.info(f"\n=== 比较MMD方法: {methods} ===")
    
    results = {}
    
    for method in methods:
        logging.info(f"\n--- 运行 {method.upper()} 方法 ---")
        
        method_path = os.path.join(base_path, f'MMD_{method}')
        
        try:
            result = run_mmd_adaptation_experiment(
                X_source, y_source, X_target, y_target,
                cat_idx=cat_idx,
                method=method,
                model_name=f'TabPFN-MMD-{method.upper()}',
                base_path=method_path,
                **kwargs
            )
            results[method] = result
            
        except Exception as e:
            logging.error(f"{method} 方法运行失败: {str(e)}")
            results[method] = {'error': str(e)}
    
    # 生成比较报告
    logging.info(f"\n=== MMD方法比较结果 ===")
    
    comparison_data = []
    for method, result in results.items():
        if 'error' not in result:
            comparison_data.append({
                'Method': method.upper(),
                'Accuracy': result['target_metrics']['acc'],
                'AUC': result['target_metrics']['auc'],
                'F1': result['target_metrics']['f1'],
                'MMD_Reduction': result['mmd_info']['reduction'],
                'Alignment_Time': result['alignment_time']
            })
    
    if comparison_data:
        import pandas as pd
        df_comparison = pd.DataFrame(comparison_data)
        
        # 按AUC排序
        df_comparison = df_comparison.sort_values('AUC', ascending=False)
        
        logging.info("\n方法性能排名:")
        for _, row in df_comparison.iterrows():
            logging.info(f"{row['Method']}: AUC={row['AUC']:.4f}, Acc={row['Accuracy']:.4f}, "
                        f"F1={row['F1']:.4f}, MMD减少={row['MMD_Reduction']:.1f}%, "
                        f"对齐时间={row['Alignment_Time']:.2f}s")
        
        # 保存比较结果
        comparison_path = os.path.join(base_path, 'mmd_methods_comparison.csv')
        df_comparison.to_csv(comparison_path, index=False)
        logging.info(f"比较结果已保存至: {comparison_path}")
    
    return results 