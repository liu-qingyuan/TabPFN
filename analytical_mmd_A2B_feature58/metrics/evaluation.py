#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分类评估模块

本模块包含分类任务的评估指标计算函数，包括：
- 准确率、AUC、F1分数等基础指标
- 阈值优化功能

度量函数（KL散度、Wasserstein距离、MMD等）已迁移到 metrics.discrepancy 模块。
"""

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
import logging
from typing import Dict, Tuple

# 导入统一的度量函数（为了向后兼容）
try:
    from .discrepancy import (
        calculate_kl_divergence, 
        calculate_wasserstein_distances, 
        compute_mmd_kernel, 
        compute_domain_discrepancy,
        detect_outliers
    )
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from analytical_mmd_A2B_feature58.metrics.discrepancy import (
            calculate_kl_divergence, 
            calculate_wasserstein_distances, 
            compute_mmd_kernel, 
            compute_domain_discrepancy,
            detect_outliers
        )
    except ImportError:
        # 如果都失败了，定义空函数以避免导入错误
        def calculate_kl_divergence(*args, **kwargs):
            return 0.0, {}
        def calculate_wasserstein_distances(*args, **kwargs):
            return 0.0, {}
        def compute_mmd_kernel(*args, **kwargs):
            return 0.0
        def compute_domain_discrepancy(*args, **kwargs):
            return {}
        def detect_outliers(*args, **kwargs):
            return [], [], [], []

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
    """
    计算所有评估指标
    
    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_pred_proba: 预测概率
    
    返回:
    - metrics: 包含各种评估指标的字典
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'auc': float(roc_auc_score(y_true, y_pred_proba)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'acc_0': float(conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0.0,
        'acc_1': float(conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0.0
    }

def print_metrics(dataset_name: str, metrics: Dict[str, float]):
    """
    打印评估指标
    
    参数:
    - dataset_name: 数据集名称
    - metrics: 评估指标字典
    """
    logging.info(f"{dataset_name}准确率 (Accuracy): {metrics['acc']:.4f}")
    logging.info(f"{dataset_name} AUC: {metrics['auc']:.4f}")
    logging.info(f"{dataset_name} F1分数: {metrics['f1']:.4f}")
    logging.info(f"{dataset_name}类别0准确率: {metrics['acc_0']:.4f}")
    logging.info(f"{dataset_name}类别1准确率: {metrics['acc_1']:.4f}")

def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    使用Youden指数(敏感性+特异性-1)寻找最佳决策阈值
    
    参数:
    - y_true: 真实标签
    - y_proba: 预测为正类的概率
    
    返回:
    - optimal_threshold: 最佳决策阈值
    - optimal_metrics: 在最佳阈值下的性能指标
    """
    # 计算ROC曲线上的各点
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # 计算每个阈值的Youden指数 (TPR + TNR - 1 = TPR - FPR)
    youden_index = tpr - fpr
    
    # 找到最大Youden指数对应的阈值
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = float(thresholds[optimal_idx])
    
    # 使用最佳阈值进行预测
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    # 计算在最佳阈值下的性能指标
    optimal_metrics = {
        'threshold': optimal_threshold,
        'acc': float(accuracy_score(y_true, y_pred_optimal)),
        'f1': float(f1_score(y_true, y_pred_optimal)),
        'sensitivity': float(tpr[optimal_idx]),  # TPR at optimal threshold
        'specificity': float(1 - fpr[optimal_idx])  # TNR at optimal threshold
    }
    
    # 计算混淆矩阵，获取每类准确率
    cm = confusion_matrix(y_true, y_pred_optimal)
    if cm.shape[0] == 2 and cm.shape[1] == 2:  # 确保是二分类
        optimal_metrics['acc_0'] = float(cm[0, 0] / (cm[0, 0] + cm[0, 1])) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
        optimal_metrics['acc_1'] = float(cm[1, 1] / (cm[1, 0] + cm[1, 1])) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0
    
    return optimal_threshold, optimal_metrics 