"""
跨域实验指标计算模块

提供跨域实验中的各种指标计算功能，包括：
- 外部数据集的交叉验证评估
- 单次评估指标计算
- 结果汇总和比较
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from .classification import calculate_basic_metrics


def evaluate_model_on_external_cv(model, X_external: np.ndarray, 
                                y_external: np.ndarray, n_folds: int = 10) -> Dict[str, Any]:
    """
    使用已训练的模型在外部数据集上进行K折交叉验证评估
    完全按照predict_healthcare_auto_external_adjust_parameter.py的实现
    
    参数:
    - model: 已训练的模型
    - X_external: 外部数据集特征
    - y_external: 外部数据集标签
    - n_folds: 交叉验证折数
    
    返回:
    - Dict[str, Any]: 包含详细结果的字典
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    all_preds = []
    all_probs = []
    all_true = []
    
    for fold, (_, test_idx) in enumerate(kf.split(X_external), 1):
        X_test_fold = X_external[test_idx]
        y_test_fold = y_external[test_idx]
        
        # 预测
        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)
        
        # 保存预测结果
        all_preds.extend(y_pred)
        all_probs.extend(y_proba[:, 1])
        all_true.extend(y_test_fold)
        
        # 计算指标
        fold_acc = accuracy_score(y_test_fold, y_pred)
        try:
            fold_auc = roc_auc_score(y_test_fold, y_proba[:, 1])
        except:
            fold_auc = 0.5  # 如果出现错误，设置为默认值
        fold_f1 = f1_score(y_test_fold, y_pred)
        
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_test_fold, y_pred)
        # 避免除以零错误
        if conf_matrix[0, 0] + conf_matrix[0, 1] > 0:
            fold_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        else:
            fold_acc_0 = 0
            
        if conf_matrix[1, 0] + conf_matrix[1, 1] > 0:
            fold_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        else:
            fold_acc_1 = 0
        
        fold_results.append({
            'fold': fold,
            'accuracy': fold_acc,
            'auc': fold_auc,
            'f1': fold_f1,
            'acc_0': fold_acc_0,
            'acc_1': fold_acc_1
        })
        
        logging.info(f"Fold {fold}: Acc={fold_acc:.4f}, AUC={fold_auc:.4f}, F1={fold_f1:.4f}")
    
    # 计算整体指标
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    overall_acc = accuracy_score(all_true, all_preds)
    overall_auc = roc_auc_score(all_true, all_probs)
    overall_f1 = f1_score(all_true, all_preds)
    
    # 计算整体混淆矩阵
    overall_cm = confusion_matrix(all_true, all_preds)
    overall_acc_0 = overall_cm[0, 0] / (overall_cm[0, 0] + overall_cm[0, 1]) if (overall_cm[0, 0] + overall_cm[0, 1]) > 0 else 0
    overall_acc_1 = overall_cm[1, 1] / (overall_cm[1, 0] + overall_cm[1, 1]) if (overall_cm[1, 0] + overall_cm[1, 1]) > 0 else 0
    
    # 计算平均和标准差
    metrics_df = pd.DataFrame(fold_results)
    
    return {
        'fold_results': fold_results,
        'overall': {
            'accuracy': overall_acc,
            'auc': overall_auc,
            'f1': overall_f1,
            'acc_0': overall_acc_0,
            'acc_1': overall_acc_1,
            'confusion_matrix': overall_cm.tolist()
        },
        'means': {
            'accuracy': metrics_df['accuracy'].mean(),
            'auc': metrics_df['auc'].mean(),
            'f1': metrics_df['f1'].mean(),
            'acc_0': metrics_df['acc_0'].mean(),
            'acc_1': metrics_df['acc_1'].mean(),
        },
        'stds': {
            'accuracy': metrics_df['accuracy'].std(),
            'auc': metrics_df['auc'].std(),
            'f1': metrics_df['f1'].std(),
            'acc_0': metrics_df['acc_0'].std(),
            'acc_1': metrics_df['acc_1'].std(),
        }
    }


def evaluate_single_external_dataset(model, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    在外部数据集上进行单次评估
    
    参数:
    - model: 模型类
    - X_train: 训练数据特征
    - y_train: 训练数据标签
    - X_test: 测试数据特征
    - y_test: 测试数据标签
    
    返回:
    - Dict[str, float]: 评估指标
    """
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # 计算指标
    return calculate_basic_metrics(y_test, y_pred, y_proba)


def log_external_cv_results(results: Dict[str, Any], prefix: str):
    """
    记录外部CV结果
    
    参数:
    - results: 评估结果字典
    - prefix: 日志前缀
    """
    means = results['means']
    stds = results['stds']
    logging.info(f"{prefix} - accuracy: {means['accuracy']:.4f} (±{stds['accuracy']:.4f})")
    logging.info(f"{prefix} - auc: {means['auc']:.4f} (±{stds['auc']:.4f})")
    logging.info(f"{prefix} - f1: {means['f1']:.4f} (±{stds['f1']:.4f})")
    logging.info(f"{prefix} - acc_0: {means['acc_0']:.4f} (±{stds['acc_0']:.4f})")
    logging.info(f"{prefix} - acc_1: {means['acc_1']:.4f} (±{stds['acc_1']:.4f})")


def calculate_cross_domain_improvement(results_before: Dict[str, Any], 
                                     results_after: Dict[str, Any]) -> Dict[str, float]:
    """
    计算跨域实验中的改进幅度
    
    参数:
    - results_before: 域适应前的结果
    - results_after: 域适应后的结果
    
    返回:
    - Dict[str, float]: 改进幅度
    """
    improvement = {}
    
    # 如果是CV结果，使用means
    if 'means' in results_before and 'means' in results_after:
        before_metrics = results_before['means']
        after_metrics = results_after['means']
    else:
        # 如果是单次评估结果，直接使用
        before_metrics = results_before
        after_metrics = results_after
    
    for metric in ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']:
        if metric in before_metrics and metric in after_metrics:
            improvement[metric] = after_metrics[metric] - before_metrics[metric]
    
    return improvement


def summarize_cross_domain_results(results: Dict[str, Any]) -> str:
    """
    汇总跨域实验结果为可读的字符串
    
    参数:
    - results: 实验结果字典
    
    返回:
    - str: 格式化的结果摘要
    """
    summary_lines = []
    
    # 数据集A的交叉验证结果
    if results.get('cross_validation_A') is not None:
        summary_lines.append("数据集A交叉验证结果:")
        cv_results = results['cross_validation_A']
        for metric, value in cv_results.items():
            if metric != 'detailed_scores':
                summary_lines.append(f"  {metric}: {value}")
        summary_lines.append("")
    
    # 外部验证结果
    for dataset in ['B', 'C']:
        key = f'external_validation_{dataset}'
        if key in results:
            summary_lines.append(f"数据集{dataset}外部验证结果:")
            ext_results = results[key]
            
            # 无域适应结果
            if 'without_domain_adaptation' in ext_results:
                summary_lines.append("  无域适应:")
                no_da_results = ext_results['without_domain_adaptation']
                if 'means' in no_da_results:
                    # CV结果
                    means = no_da_results['means']
                    stds = no_da_results['stds']
                    for metric in ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']:
                        if metric in means:
                            summary_lines.append(f"    {metric}: {means[metric]:.4f} (±{stds[metric]:.4f})")
                else:
                    # 单次评估结果
                    for metric, value in no_da_results.items():
                        if metric not in ['confusion_matrix', 'detailed_scores']:
                            if isinstance(value, (int, float)):
                                summary_lines.append(f"      {metric}: {value:.4f}")
                            else:
                                summary_lines.append(f"      {metric}: {value}")
            
            # 有域适应结果
            if 'with_domain_adaptation' in ext_results:
                summary_lines.append("  有域适应:")
                with_da_results = ext_results['with_domain_adaptation']
                if 'means' in with_da_results:
                    # CV结果
                    means = with_da_results['means']
                    stds = with_da_results['stds']
                    for metric in ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']:
                        if metric in means:
                            summary_lines.append(f"    {metric}: {means[metric]:.4f} (±{stds[metric]:.4f})")
                else:
                    # 单次评估结果
                    for metric, value in with_da_results.items():
                        if metric not in ['confusion_matrix', 'detailed_scores']:
                            if isinstance(value, (int, float)):
                                summary_lines.append(f"      {metric}: {value:.4f}")
                            else:
                                summary_lines.append(f"      {metric}: {value}")
            
            # 改进幅度
            if 'improvement' in ext_results:
                summary_lines.append("  改进幅度:")
                for metric, value in ext_results['improvement'].items():
                    # 检查值的类型，如果已经是字符串就直接使用，否则格式化
                    if isinstance(value, str):
                        summary_lines.append(f"    {metric}: {value}")
                    else:
                        summary_lines.append(f"    {metric}: {value:.4f}")
            
            summary_lines.append("")
    
    return "\n".join(summary_lines)