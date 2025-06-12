import numpy as np
import logging
from typing import Dict, Tuple, Any
from sklearn.metrics import roc_curve, accuracy_score, f1_score, confusion_matrix


def optimize_threshold_youden(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
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
    optimal_threshold = thresholds[optimal_idx]
    
    # 使用最佳阈值进行预测
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    # 计算在最佳阈值下的性能指标
    optimal_metrics = {
        'threshold': optimal_threshold,
        'acc': accuracy_score(y_true, y_pred_optimal),
        'f1': f1_score(y_true, y_pred_optimal),
        'sensitivity': tpr[optimal_idx],  # TPR at optimal threshold
        'specificity': 1 - fpr[optimal_idx],  # TNR at optimal threshold
        'youden_index': youden_index[optimal_idx]
    }
    
    # 计算混淆矩阵，获取每类准确率
    cm = confusion_matrix(y_true, y_pred_optimal)
    if cm.shape[0] == 2 and cm.shape[1] == 2:  # 确保是二分类
        optimal_metrics['acc_0'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        optimal_metrics['acc_1'] = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    else:
        optimal_metrics['acc_0'] = 0
        optimal_metrics['acc_1'] = 0
    
    logging.info(f"阈值优化完成: 最佳阈值 = {optimal_threshold:.4f}, Youden指数 = {youden_index[optimal_idx]:.4f}")
    
    return optimal_threshold, optimal_metrics


def apply_threshold_optimization(y_true: np.ndarray, y_pred_original: np.ndarray, 
                                y_proba: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    应用阈值优化并返回优化后的预测结果
    
    参数:
    - y_true: 真实标签
    - y_pred_original: 原始预测标签（使用默认阈值0.5）
    - y_proba: 预测概率
    
    返回:
    - y_pred_optimized: 优化后的预测标签
    - optimization_info: 优化信息
    """
    # 计算原始性能指标
    original_metrics = {
        'acc': accuracy_score(y_true, y_pred_original),
        'f1': f1_score(y_true, y_pred_original)
    }
    
    # 计算原始混淆矩阵
    cm_original = confusion_matrix(y_true, y_pred_original)
    if cm_original.shape[0] == 2 and cm_original.shape[1] == 2:
        original_metrics['acc_0'] = cm_original[0, 0] / (cm_original[0, 0] + cm_original[0, 1]) if (cm_original[0, 0] + cm_original[0, 1]) > 0 else 0
        original_metrics['acc_1'] = cm_original[1, 1] / (cm_original[1, 0] + cm_original[1, 1]) if (cm_original[1, 0] + cm_original[1, 1]) > 0 else 0
    else:
        original_metrics['acc_0'] = 0
        original_metrics['acc_1'] = 0
    
    # 优化阈值
    optimal_threshold, optimal_metrics = optimize_threshold_youden(y_true, y_proba)
    
    # 使用优化阈值生成新的预测
    y_pred_optimized = (y_proba >= optimal_threshold).astype(int)
    
    # 计算改进
    improvements = {}
    for metric in ['acc', 'f1', 'acc_0', 'acc_1']:
        improvements[metric] = optimal_metrics[metric] - original_metrics[metric]
    
    # 编译优化信息
    optimization_info = {
        'optimal_threshold': optimal_threshold,
        'default_threshold': 0.5,
        'original_metrics': original_metrics,
        'optimal_metrics': optimal_metrics,
        'improvements': improvements,
        'youden_index': optimal_metrics['youden_index']
    }
    
    # 记录优化结果
    logging.info(f"阈值优化结果:")
    logging.info(f"  默认阈值(0.5) -> 最佳阈值({optimal_threshold:.4f})")
    logging.info(f"  准确率: {original_metrics['acc']:.4f} -> {optimal_metrics['acc']:.4f} ({improvements['acc']:+.4f})")
    logging.info(f"  F1分数: {original_metrics['f1']:.4f} -> {optimal_metrics['f1']:.4f} ({improvements['f1']:+.4f})")
    logging.info(f"  类别0准确率: {original_metrics['acc_0']:.4f} -> {optimal_metrics['acc_0']:.4f} ({improvements['acc_0']:+.4f})")
    logging.info(f"  类别1准确率: {original_metrics['acc_1']:.4f} -> {optimal_metrics['acc_1']:.4f} ({improvements['acc_1']:+.4f})")
    
    return y_pred_optimized, optimization_info


def get_threshold_optimization_suffix(use_threshold_optimization: bool) -> str:
    """
    根据是否使用阈值优化返回路径后缀
    
    参数:
    - use_threshold_optimization: 是否使用阈值优化
    
    返回:
    - suffix: 路径后缀字符串
    """
    return "_threshold_optimized" if use_threshold_optimization else ""


def get_roc_curve_data(y_true: np.ndarray, y_proba: np.ndarray, 
                      optimal_threshold: float) -> Dict[str, Any]:
    """
    获取ROC曲线数据，用于可视化
    
    参数:
    - y_true: 真实标签
    - y_proba: 预测概率
    - optimal_threshold: 最佳阈值
    
    返回:
    - roc_data: ROC曲线相关数据
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    # 找到最佳阈值对应的点
    optimal_idx = None
    for i, t in enumerate(thresholds):
        if t <= optimal_threshold:
            optimal_idx = i
            break
    
    # 找到默认阈值0.5对应的点
    default_idx = None
    for i, t in enumerate(thresholds):
        if t <= 0.5:
            default_idx = i
            break
    
    roc_data = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'optimal_threshold': optimal_threshold,
        'optimal_idx': optimal_idx,
        'default_idx': default_idx
    }
    
    return roc_data 