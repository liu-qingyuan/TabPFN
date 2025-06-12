#!/usr/bin/env python3
"""
Mean-Variance Alignment 域适应算法实现

Mean-Variance Alignment通过对齐源域和目标域的一阶统计量（均值）和二阶统计量（方差）
来减少域差距。这是一个简单而有效的统计对齐方法。

这种方法假设源域和目标域的数据分布可以通过简单的线性变换对齐。
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler


def compute_mean_variance_distance(X_s: np.ndarray, X_t: np.ndarray) -> Dict[str, Any]:
    """
    计算源域和目标域之间的均值和方差距离
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - X_t: 目标域特征 [n_samples_target, n_features]
    
    返回:
    - distances: 包含均值距离和方差距离的字典
    """
    # 计算均值
    mean_s = np.mean(X_s, axis=0)
    mean_t = np.mean(X_t, axis=0)
    
    # 计算方差
    var_s = np.var(X_s, axis=0)
    var_t = np.var(X_t, axis=0)
    
    # 计算距离
    mean_distance = np.linalg.norm(mean_s - mean_t)
    var_distance = np.linalg.norm(var_s - var_t)
    
    # 计算相对距离（归一化）
    mean_relative_distance = mean_distance / (np.linalg.norm(mean_s) + 1e-8)
    var_relative_distance = var_distance / (np.linalg.norm(var_s) + 1e-8)
    
    return {
        'mean_distance': mean_distance,
        'var_distance': var_distance,
        'mean_relative_distance': mean_relative_distance,
        'var_relative_distance': var_relative_distance,
        'total_distance': mean_distance + var_distance
    }


def mean_variance_transform(X_s: np.ndarray, X_t: np.ndarray,
                           align_mean: bool = True,
                           align_variance: bool = True,
                           standardize: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    使用均值-方差对齐算法对齐目标域到源域
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - X_t: 目标域特征 [n_samples_target, n_features]
    - align_mean: 是否对齐均值
    - align_variance: 是否对齐方差
    - standardize: 是否预先标准化特征
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    - mv_info: 均值-方差对齐信息
    """
    logging.info("开始Mean-Variance域适应...")
    logging.info(f"对齐设置: 均值={align_mean}, 方差={align_variance}")
    
    # 特征标准化（可选）
    if standardize:
        scaler = StandardScaler()
        X_s_scaled = scaler.fit_transform(X_s)
        X_t_scaled = scaler.transform(X_t)
        logging.info("已对特征进行标准化")
    else:
        X_s_scaled = X_s.copy()
        X_t_scaled = X_t.copy()
        scaler = None
    
    # 计算初始距离
    initial_distances = compute_mean_variance_distance(X_s_scaled, X_t_scaled)
    logging.info(f"初始均值距离: {initial_distances['mean_distance']:.6f}")
    logging.info(f"初始方差距离: {initial_distances['var_distance']:.6f}")
    
    # 计算统计量
    mean_s = np.mean(X_s_scaled, axis=0)
    mean_t = np.mean(X_t_scaled, axis=0)
    var_s = np.var(X_s_scaled, axis=0)
    var_t = np.var(X_t_scaled, axis=0)
    
    # 防止除零
    var_s = np.maximum(var_s, 1e-8)
    var_t = np.maximum(var_t, 1e-8)
    
    # 应用对齐变换
    X_t_aligned_scaled = X_t_scaled.copy()
    
    if align_variance:
        # 先对齐方差：X_t_new = (X_t - mean_t) * sqrt(var_s / var_t)
        scale_factor = np.sqrt(var_s / var_t)
        X_t_aligned_scaled = (X_t_aligned_scaled - mean_t) * scale_factor
        
        # 更新目标域均值（方差对齐后）
        mean_t_after_var_align = np.mean(X_t_aligned_scaled, axis=0)
        
        if align_mean:
            # 再对齐均值：X_t_final = X_t_new + (mean_s - mean_t_new)
            X_t_aligned_scaled = X_t_aligned_scaled + (mean_s - mean_t_after_var_align)
        else:
            # 如果不对齐均值，恢复原始均值
            X_t_aligned_scaled = X_t_aligned_scaled + mean_t
            
    elif align_mean:
        # 只对齐均值：X_t_new = X_t + (mean_s - mean_t)
        X_t_aligned_scaled = X_t_aligned_scaled + (mean_s - mean_t)
    
    # 如果使用了标准化，逆标准化回原始空间
    if standardize and scaler is not None:
        X_t_aligned = scaler.inverse_transform(X_t_aligned_scaled)
    else:
        X_t_aligned = X_t_aligned_scaled
    
    # 计算最终距离
    final_distances = compute_mean_variance_distance(X_s_scaled, X_t_aligned_scaled)
    
    # 计算改进百分比
    mean_improvement = 0.0
    var_improvement = 0.0
    total_improvement = 0.0
    
    if initial_distances['mean_distance'] > 1e-10:
        mean_improvement = (initial_distances['mean_distance'] - final_distances['mean_distance']) / initial_distances['mean_distance'] * 100
    
    if initial_distances['var_distance'] > 1e-10:
        var_improvement = (initial_distances['var_distance'] - final_distances['var_distance']) / initial_distances['var_distance'] * 100
    
    if initial_distances['total_distance'] > 1e-10:
        total_improvement = (initial_distances['total_distance'] - final_distances['total_distance']) / initial_distances['total_distance'] * 100
    
    logging.info(f"最终均值距离: {final_distances['mean_distance']:.6f}")
    logging.info(f"最终方差距离: {final_distances['var_distance']:.6f}")
    logging.info(f"均值距离减少: {mean_improvement:.2f}%")
    logging.info(f"方差距离减少: {var_improvement:.2f}%")
    logging.info(f"总体距离减少: {total_improvement:.2f}%")
    
    mv_info = {
        'method': 'mean_variance_alignment',
        'align_mean': align_mean,
        'align_variance': align_variance,
        'standardized': standardize,
        'initial_distances': initial_distances,
        'final_distances': final_distances,
        'mean_improvement_percent': mean_improvement,
        'var_improvement_percent': var_improvement,
        'total_improvement_percent': total_improvement
    }
    
    return X_t_aligned, mv_info


def adaptive_mean_variance_transform(X_s: np.ndarray, X_t: np.ndarray,
                                   standardize: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    自适应均值-方差对齐：自动选择最佳对齐策略
    
    参数:
    - X_s: 源域特征
    - X_t: 目标域特征
    - standardize: 是否标准化特征
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    - mv_info: 自适应均值-方差对齐信息
    """
    logging.info("开始自适应Mean-Variance域适应...")
    
    # 测试不同的对齐策略
    strategies = [
        {'align_mean': True, 'align_variance': True, 'name': 'mean+variance'},
        {'align_mean': True, 'align_variance': False, 'name': 'mean_only'},
        {'align_mean': False, 'align_variance': True, 'name': 'variance_only'},
        {'align_mean': False, 'align_variance': False, 'name': 'no_alignment'}
    ]
    
    best_strategy = None
    best_improvement = -float('inf')
    best_result = None
    best_info = None
    
    for strategy in strategies:
        X_t_candidate, info = mean_variance_transform(
            X_s, X_t, 
            align_mean=strategy['align_mean'],
            align_variance=strategy['align_variance'],
            standardize=standardize
        )
        
        total_improvement = info['total_improvement_percent']
        
        logging.info(f"策略 '{strategy['name']}': 总体改进 {total_improvement:.2f}%")
        
        if total_improvement > best_improvement:
            best_improvement = total_improvement
            best_strategy = strategy
            best_result = X_t_candidate
            best_info = info
    
    logging.info(f"最佳策略: '{best_strategy['name']}' (改进: {best_improvement:.2f}%)")
    
    # 更新信息
    best_info['method'] = 'adaptive_mean_variance_alignment'
    best_info['best_strategy'] = best_strategy
    best_info['all_strategies_tested'] = strategies
    
    return best_result, best_info


def class_conditional_mean_variance_transform(X_s: np.ndarray, y_s: np.ndarray,
                                            X_t: np.ndarray, y_t_pseudo: np.ndarray,
                                            align_mean: bool = True,
                                            align_variance: bool = True,
                                            standardize: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    类条件均值-方差对齐：分别对每个类别进行对齐
    
    参数:
    - X_s: 源域特征
    - y_s: 源域标签
    - X_t: 目标域特征
    - y_t_pseudo: 目标域伪标签
    - align_mean: 是否对齐均值
    - align_variance: 是否对齐方差
    - standardize: 是否标准化特征
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    - mv_info: 类条件均值-方差对齐信息
    """
    logging.info("开始类条件Mean-Variance域适应...")
    
    unique_classes = np.unique(y_s)
    logging.info(f"发现 {len(unique_classes)} 个类别: {unique_classes}")
    
    X_t_aligned = X_t.copy()
    class_distances = {}
    class_improvements = {}
    total_initial_distance = 0.0
    total_final_distance = 0.0
    
    for class_label in unique_classes:
        # 获取当前类别的样本
        s_mask = (y_s == class_label)
        t_mask = (y_t_pseudo == class_label)
        
        X_s_class = X_s[s_mask]
        X_t_class = X_t[t_mask]
        
        if len(X_s_class) < 2 or len(X_t_class) < 2:
            logging.warning(f"类别 {class_label} 样本数不足，跳过对齐")
            class_distances[class_label] = {'initial': 0.0, 'final': 0.0, 'improvement': 0.0}
            continue
        
        logging.info(f"类别 {class_label}: 源域 {len(X_s_class)} 样本, 目标域 {len(X_t_class)} 样本")
        
        # 对当前类别应用均值-方差对齐
        X_t_class_aligned, class_info = mean_variance_transform(
            X_s_class, X_t_class, align_mean, align_variance, standardize
        )
        
        # 更新对齐后的特征
        X_t_aligned[t_mask] = X_t_class_aligned
        
        # 记录类别级别的距离
        class_distances[class_label] = {
            'initial': class_info['initial_distances']['total_distance'],
            'final': class_info['final_distances']['total_distance'],
            'improvement': class_info['total_improvement_percent']
        }
        
        # 累计总距离（按样本数加权）
        weight = len(X_t_class) / len(X_t)
        total_initial_distance += class_info['initial_distances']['total_distance'] * weight
        total_final_distance += class_info['final_distances']['total_distance'] * weight
        
        logging.info(f"类别 {class_label} 总体距离: {class_info['initial_distances']['total_distance']:.6f} -> {class_info['final_distances']['total_distance']:.6f}")
    
    # 计算总体改进
    if total_initial_distance > 1e-10:
        total_improvement = (total_initial_distance - total_final_distance) / total_initial_distance * 100
    else:
        total_improvement = 0.0
    
    logging.info(f"类条件Mean-Variance总体改进: {total_improvement:.2f}%")
    
    mv_info = {
        'method': 'class_conditional_mean_variance_alignment',
        'align_mean': align_mean,
        'align_variance': align_variance,
        'standardized': standardize,
        'total_initial_distance': total_initial_distance,
        'total_final_distance': total_final_distance,
        'total_improvement_percent': total_improvement,
        'class_distances': class_distances,
        'num_classes': len(unique_classes)
    }
    
    return X_t_aligned, mv_info


def feature_wise_mean_variance_transform(X_s: np.ndarray, X_t: np.ndarray,
                                       feature_weights: Optional[np.ndarray] = None,
                                       align_mean: bool = True,
                                       align_variance: bool = True,
                                       standardize: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    特征级别的均值-方差对齐：对不同特征使用不同的对齐强度
    
    参数:
    - X_s: 源域特征
    - X_t: 目标域特征
    - feature_weights: 特征权重（0-1之间），控制每个特征的对齐强度
    - align_mean: 是否对齐均值
    - align_variance: 是否对齐方差
    - standardize: 是否标准化特征
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    - mv_info: 特征级别均值-方差对齐信息
    """
    logging.info("开始特征级别Mean-Variance域适应...")
    
    n_features = X_s.shape[1]
    
    # 如果没有提供特征权重，使用均匀权重
    if feature_weights is None:
        feature_weights = np.ones(n_features)
        logging.info("使用均匀特征权重")
    else:
        feature_weights = np.clip(feature_weights, 0.0, 1.0)  # 确保在[0,1]范围内
        logging.info(f"使用自定义特征权重，平均权重: {np.mean(feature_weights):.3f}")
    
    # 特征标准化（可选）
    if standardize:
        scaler = StandardScaler()
        X_s_scaled = scaler.fit_transform(X_s)
        X_t_scaled = scaler.transform(X_t)
        logging.info("已对特征进行标准化")
    else:
        X_s_scaled = X_s.copy()
        X_t_scaled = X_t.copy()
        scaler = None
    
    # 计算初始距离
    initial_distances = compute_mean_variance_distance(X_s_scaled, X_t_scaled)
    
    # 计算统计量
    mean_s = np.mean(X_s_scaled, axis=0)
    mean_t = np.mean(X_t_scaled, axis=0)
    var_s = np.var(X_s_scaled, axis=0)
    var_t = np.var(X_t_scaled, axis=0)
    
    # 防止除零
    var_s = np.maximum(var_s, 1e-8)
    var_t = np.maximum(var_t, 1e-8)
    
    # 应用加权对齐变换
    X_t_aligned_scaled = X_t_scaled.copy()
    
    if align_variance:
        # 特征级别的方差对齐
        scale_factor = np.sqrt(var_s / var_t)
        # 使用权重控制对齐强度：weight=0表示不对齐，weight=1表示完全对齐
        weighted_scale_factor = feature_weights * scale_factor + (1 - feature_weights) * 1.0
        
        X_t_aligned_scaled = (X_t_aligned_scaled - mean_t) * weighted_scale_factor
        
        # 更新目标域均值
        mean_t_after_var_align = np.mean(X_t_aligned_scaled, axis=0)
        
        if align_mean:
            # 特征级别的均值对齐
            mean_shift = mean_s - mean_t_after_var_align
            weighted_mean_shift = feature_weights * mean_shift
            X_t_aligned_scaled = X_t_aligned_scaled + weighted_mean_shift
        else:
            # 恢复原始均值
            X_t_aligned_scaled = X_t_aligned_scaled + mean_t
            
    elif align_mean:
        # 只进行特征级别的均值对齐
        mean_shift = mean_s - mean_t
        weighted_mean_shift = feature_weights * mean_shift
        X_t_aligned_scaled = X_t_aligned_scaled + weighted_mean_shift
    
    # 如果使用了标准化，逆标准化回原始空间
    if standardize and scaler is not None:
        X_t_aligned = scaler.inverse_transform(X_t_aligned_scaled)
    else:
        X_t_aligned = X_t_aligned_scaled
    
    # 计算最终距离
    final_distances = compute_mean_variance_distance(X_s_scaled, X_t_aligned_scaled)
    
    # 计算改进百分比
    total_improvement = 0.0
    if initial_distances['total_distance'] > 1e-10:
        total_improvement = (initial_distances['total_distance'] - final_distances['total_distance']) / initial_distances['total_distance'] * 100
    
    logging.info(f"特征级别Mean-Variance总体改进: {total_improvement:.2f}%")
    
    mv_info = {
        'method': 'feature_wise_mean_variance_alignment',
        'align_mean': align_mean,
        'align_variance': align_variance,
        'standardized': standardize,
        'feature_weights': feature_weights.tolist(),
        'initial_distances': initial_distances,
        'final_distances': final_distances,
        'total_improvement_percent': total_improvement
    }
    
    return X_t_aligned, mv_info 