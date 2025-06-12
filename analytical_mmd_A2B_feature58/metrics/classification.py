"""
分类指标计算模块

提供各种分类任务的评估指标计算功能
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix,
    precision_score, recall_score
)
import logging

def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_proba: np.ndarray) -> Dict[str, float]:
    """
    计算基础分类指标
    
    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_proba: 预测概率 (N, 2)
    
    返回:
    - Dict[str, float]: 包含各种指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_proba[:, 1]),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }
    
    # 计算混淆矩阵和每类准确率
    conf_matrix = confusion_matrix(y_true, y_pred)
    if conf_matrix.shape == (2, 2):
        # 类别0的准确率 (特异性)
        metrics['acc_0'] = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
        # 类别1的准确率 (敏感性)
        metrics['acc_1'] = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
        metrics['confusion_matrix'] = conf_matrix.tolist()
    
    return metrics

def calculate_cv_summary(cv_scores: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    计算交叉验证结果的汇总统计
    
    参数:
    - cv_scores: 交叉验证各折的结果列表
    
    返回:
    - Dict[str, str]: 包含均值±标准差格式的汇总结果
    """
    cv_df = pd.DataFrame(cv_scores)
    
    summary = {}
    for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall', 'acc_0', 'acc_1']:
        if metric in cv_df.columns:
            mean_val = cv_df[metric].mean()
            std_val = cv_df[metric].std()
            summary[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
    
    return summary

def calculate_improvement(results_before: Dict[str, str], 
                        results_after: Dict[str, str]) -> Dict[str, str]:
    """
    计算改进幅度
    
    参数:
    - results_before: 改进前的结果
    - results_after: 改进后的结果
    
    返回:
    - Dict[str, str]: 改进幅度
    """
    improvement = {}
    
    for metric in ['accuracy', 'auc', 'f1']:
        if metric in results_before and metric in results_after:
            # 提取均值部分
            before_val = float(results_before[metric].split(' ± ')[0])
            after_val = float(results_after[metric].split(' ± ')[0])
            improvement[metric] = f"{after_val - before_val:+.4f}"
    
    return improvement

def log_metrics(metrics: Dict[str, Any], prefix: str = ""):
    """
    记录指标到日志
    
    参数:
    - metrics: 指标字典
    - prefix: 日志前缀
    """
    if prefix:
        prefix = f"{prefix} - "
    
    for metric, value in metrics.items():
        if metric not in ['confusion_matrix', 'detailed_scores', 'adaptation_info', 'threshold_info']:
            # 直接记录值，不进行额外的格式化
            # 因为改进幅度等值已经是格式化的字符串
            logging.info(f"{prefix}{metric}: {value}")

def format_metrics_for_display(metrics: Dict[str, Any]) -> str:
    """
    格式化指标用于显示
    
    参数:
    - metrics: 指标字典
    
    返回:
    - str: 格式化的字符串
    """
    display_lines = []
    
    for metric, value in metrics.items():
        if metric not in ['confusion_matrix', 'detailed_scores', 'adaptation_info', 'threshold_info']:
            display_lines.append(f"  {metric}: {value}")
    
    return "\n".join(display_lines)

def compare_metrics(metrics1: Dict[str, Any], metrics2: Dict[str, Any], 
                   labels: Optional[List[str]] = None) -> str:
    """
    比较两组指标
    
    参数:
    - metrics1: 第一组指标
    - metrics2: 第二组指标
    - labels: 标签列表
    
    返回:
    - str: 比较结果的格式化字符串
    """
    if labels is None:
        labels = ["Before", "After"]
    
    comparison_lines = []
    comparison_lines.append(f"{'Metric':<15} {labels[0]:<15} {labels[1]:<15} {'Improvement':<15}")
    comparison_lines.append("-" * 60)
    
    for metric in ['accuracy', 'auc', 'f1']:
        if metric in metrics1 and metric in metrics2:
            val1 = float(metrics1[metric].split(' ± ')[0]) if isinstance(metrics1[metric], str) else metrics1[metric]
            val2 = float(metrics2[metric].split(' ± ')[0]) if isinstance(metrics2[metric], str) else metrics2[metric]
            improvement = val2 - val1
            
            comparison_lines.append(
                f"{metric:<15} {val1:<15.4f} {val2:<15.4f} {improvement:<+15.4f}"
            )
    
    return "\n".join(comparison_lines) 