#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一的分布差异度量模块

本模块包含所有用于计算分布差异的度量函数，包括：
- KL散度 (Kullback-Leibler Divergence)
- Wasserstein距离 (Earth Mover's Distance)
- MMD (Maximum Mean Discrepancy)
- 域间差异综合指标

这是所有度量函数的统一入口，避免代码重复。
"""

import numpy as np
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from typing import Dict, Any, Tuple
import logging

def calculate_kl_divergence(X_source: np.ndarray, X_target: np.ndarray, 
                           bins: int = 20, epsilon: float = 1e-10) -> Tuple[float, Dict[str, float]]:
    """
    使用直方图计算每个特征的KL散度，并返回平均值
    
    参数:
    - X_source: 源域特征 [n_samples_source, n_features]
    - X_target: 目标域特征 [n_samples_target, n_features]
    - bins: 直方图的箱数
    - epsilon: 平滑因子，防止除零错误
    
    返回:
    - kl_div: 平均KL散度
    - kl_per_feature: 每个特征的KL散度字典
    """
    n_features = X_source.shape[1]
    kl_per_feature: Dict[str, float] = {}
    
    for i in range(n_features):
        # 提取第i个特征
        x_s = X_source[:, i]
        x_t = X_target[:, i]
        
        # 确定共同的区间范围
        min_val = min(np.min(x_s), np.min(x_t))
        max_val = max(np.max(x_s), np.max(x_t))
        bin_range = (min_val, max_val)
        
        # 计算直方图
        hist_s, _ = np.histogram(x_s, bins=bins, range=bin_range, density=True)
        hist_t, _ = np.histogram(x_t, bins=bins, range=bin_range, density=True)
        
        # 平滑处理，防止除零
        hist_s = hist_s + epsilon
        hist_t = hist_t + epsilon
        
        # 归一化
        hist_s = hist_s / np.sum(hist_s)
        hist_t = hist_t / np.sum(hist_t)
        
        # 计算KL散度 (P || Q)
        kl_s_t = entropy(hist_s, hist_t)
        kl_t_s = entropy(hist_t, hist_s)
        
        # 保存结果（使用对称KL散度）
        kl_per_feature[f'feature_{i}'] = float((kl_s_t + kl_t_s) / 2)
    
    # 计算平均KL散度
    kl_div = float(np.mean(list(kl_per_feature.values()))) if kl_per_feature else 0.0
    
    return kl_div, kl_per_feature

def calculate_wasserstein_distances(X_source: np.ndarray, X_target: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """
    计算每个特征的Wasserstein距离（Earth Mover's Distance）
    
    参数:
    - X_source: 源域特征 [n_samples_source, n_features]
    - X_target: 目标域特征 [n_samples_target, n_features]
    
    返回:
    - avg_wasserstein: 平均Wasserstein距离
    - wasserstein_per_feature: 每个特征的Wasserstein距离字典
    """
    n_features = X_source.shape[1]
    wasserstein_per_feature: Dict[str, float] = {}
    
    for i in range(n_features):
        # 提取第i个特征
        x_s = X_source[:, i]
        x_t = X_target[:, i]
        
        # 计算Wasserstein距离
        w_dist = wasserstein_distance(x_s, x_t)
        wasserstein_per_feature[f'feature_{i}'] = float(w_dist)
    
    # 计算平均Wasserstein距离
    avg_wasserstein = float(np.mean(list(wasserstein_per_feature.values()))) if wasserstein_per_feature else 0.0
    
    return avg_wasserstein, wasserstein_per_feature

def compute_mmd_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    计算基于RBF核的MMD (Maximum Mean Discrepancy)
    
    参数:
    - X: 第一个分布的样本 [n_samples_x, n_features]
    - Y: 第二个分布的样本 [n_samples_y, n_features]
    - gamma: RBF核参数
    
    返回:
    - mmd_value: MMD值
    """
    # 样本数
    n_x = X.shape[0]
    n_y = Y.shape[0]

    if n_x == 0 or n_y == 0:  # Handle empty inputs
        return 0.0
    
    # 计算核矩阵
    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)
    
    # 计算MMD
    # Handle cases where n_x or n_y is 1 to avoid division by zero or (n_x - 1) = 0
    term_xx = (np.sum(K_xx) - np.trace(K_xx)) / (n_x * (n_x - 1)) if n_x > 1 else 0.0
    term_yy = (np.sum(K_yy) - np.trace(K_yy)) / (n_y * (n_y - 1)) if n_y > 1 else 0.0
    term_xy = 2 * np.mean(K_xy) if n_x > 0 and n_y > 0 else 0.0
    
    mmd_squared = term_xx + term_yy - term_xy
    
    return float(np.sqrt(max(mmd_squared, 0)))

# 为了兼容性，提供compute_mmd别名
def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """
    compute_mmd_kernel的别名，为了兼容性
    """
    return compute_mmd_kernel(X, Y, gamma)

def compute_domain_discrepancy(X_source: np.ndarray, X_target: np.ndarray) -> Dict[str, Any]:
    """
    计算源域和目标域之间的分布差异度量
    
    参数:
    - X_source: 源域特征
    - X_target: 目标域特征
    
    返回:
    - discrepancy_metrics: 包含多种差异度量的字典
    """
    # 检查输入数据是否为空
    if X_source.shape[0] == 0 or X_target.shape[0] == 0:
        logging.warning("Cannot compute domain discrepancy with empty input arrays.")
        return {
            'mean_distance': 0.0,
            'mean_difference': 0.0,
            'covariance_difference': 0.0,
            'kernel_mean_difference': 0.0,
            'mmd': 0.0,
            'kl_divergence': 0.0,
            'kl_per_feature': {},
            'wasserstein_distance': 0.0,
            'wasserstein_per_feature': {}
        }
    
    # 1. 平均距离
    mean_dist = float(np.mean(cdist(X_source, X_target))) if X_source.size > 0 and X_target.size > 0 else 0.0
    
    # 2. 均值差异
    mean_diff = float(np.linalg.norm(np.mean(X_source, axis=0) - np.mean(X_target, axis=0)))
    
    # 3. 协方差矩阵距离
    # 计算协方差时需要至少2个样本
    cov_source = np.cov(X_source, rowvar=False) if X_source.shape[0] > 1 else np.zeros((X_source.shape[1], X_source.shape[1]))
    cov_target = np.cov(X_target, rowvar=False) if X_target.shape[0] > 1 else np.zeros((X_target.shape[1], X_target.shape[1]))
    cov_diff = float(np.linalg.norm(cov_source - cov_target, 'fro'))
    
    # 4. 核均值差异
    X_s_mean = np.mean(X_source, axis=0, keepdims=True)
    X_t_mean = np.mean(X_target, axis=0, keepdims=True)
    kernel_mean_diff = float(np.exp(-0.5 * np.sum((X_s_mean - X_t_mean)**2)))
    
    # 5. MMD
    mmd_value = compute_mmd_kernel(X_source, X_target)
    
    # 6. KL散度
    kl_div, kl_per_feature = calculate_kl_divergence(X_source, X_target)
    
    # 7. Wasserstein距离
    wasserstein_dist, wasserstein_per_feature = calculate_wasserstein_distances(X_source, X_target)
    
    return {
        'mean_distance': mean_dist,
        'mean_difference': mean_diff,
        'covariance_difference': cov_diff,
        'kernel_mean_difference': kernel_mean_diff,
        'mmd': mmd_value,
        'kl_divergence': kl_div,
        'kl_per_feature': kl_per_feature,
        'wasserstein_distance': wasserstein_dist,
        'wasserstein_per_feature': wasserstein_per_feature
    }

def detect_outliers(X_source: np.ndarray, X_target: np.ndarray, 
                   percentile: int = 95) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    检测异常点
    
    参数:
    - X_source: 源域特征
    - X_target: 目标域特征
    - percentile: 异常点阈值百分位数
    
    返回:
    - source_outliers: 源域异常点索引
    - target_outliers: 目标域异常点索引
    - min_dist_source: 源域样本到目标域的最小距离
    - min_dist_target: 目标域样本到源域的最小距离
    """
    if X_source.shape[0] == 0 or X_target.shape[0] == 0:
        logging.warning("Cannot detect outliers with empty input arrays.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 计算每个源域样本到目标域的最小距离
    min_dist_source = np.min(cdist(X_source, X_target), axis=1)
    # 计算每个目标域样本到源域的最小距离
    min_dist_target = np.min(cdist(X_target, X_source), axis=1)

    # 根据百分位数确定异常点阈值
    source_threshold = np.percentile(min_dist_source, percentile)
    target_threshold = np.percentile(min_dist_target, percentile)

    # 找出异常点
    source_outliers = np.where(min_dist_source > source_threshold)[0]
    target_outliers = np.where(min_dist_target > target_threshold)[0]

    return source_outliers, target_outliers, min_dist_source, min_dist_target 