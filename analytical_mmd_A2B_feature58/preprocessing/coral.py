#!/usr/bin/env python3
"""
CORAL (CORrelation ALignment) 域适应算法实现

CORAL通过对齐源域和目标域的二阶统计量（协方差矩阵）来减少域差距。
这是一个简单而有效的无监督域适应方法。

参考文献:
Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep domain adaptation. 
In European conference on computer vision (pp. 443-450). Springer.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler


def compute_coral_loss(X_s: np.ndarray, X_t: np.ndarray) -> float:
    """
    计算CORAL损失（Frobenius范数）
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - X_t: 目标域特征 [n_samples_target, n_features]
    
    返回:
    - coral_loss: CORAL损失值
    """
    # 中心化数据
    X_s_centered = X_s - np.mean(X_s, axis=0)
    X_t_centered = X_t - np.mean(X_t, axis=0)
    
    # 计算协方差矩阵
    n_s = X_s.shape[0]
    n_t = X_t.shape[0]
    
    C_s = np.cov(X_s_centered, rowvar=False)
    C_t = np.cov(X_t_centered, rowvar=False)
    
    # 计算Frobenius范数
    coral_loss = np.linalg.norm(C_s - C_t, 'fro') ** 2
    
    # 归一化
    coral_loss = coral_loss / (4 * X_s.shape[1] ** 2)
    
    return coral_loss


def coral_transform(X_s: np.ndarray, X_t: np.ndarray, 
                   regularization: float = 1e-6,
                   standardize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    使用CORAL算法对齐目标域到源域
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - X_t: 目标域特征 [n_samples_target, n_features]
    - regularization: 正则化参数，防止协方差矩阵奇异
    - standardize: 是否标准化特征
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    - coral_info: CORAL变换信息
    """
    logging.info("开始CORAL域适应...")
    
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
    
    # 计算初始CORAL损失
    initial_coral_loss = compute_coral_loss(X_s_scaled, X_t_scaled)
    logging.info(f"初始CORAL损失: {initial_coral_loss:.6f}")
    
    # 中心化数据
    X_s_mean = np.mean(X_s_scaled, axis=0)
    X_t_mean = np.mean(X_t_scaled, axis=0)
    
    X_s_centered = X_s_scaled - X_s_mean
    X_t_centered = X_t_scaled - X_t_mean
    
    # 计算协方差矩阵
    C_s = np.cov(X_s_centered, rowvar=False) + regularization * np.eye(X_s_scaled.shape[1])
    C_t = np.cov(X_t_centered, rowvar=False) + regularization * np.eye(X_t_scaled.shape[1])
    
    logging.info(f"源域协方差矩阵条件数: {np.linalg.cond(C_s):.2e}")
    logging.info(f"目标域协方差矩阵条件数: {np.linalg.cond(C_t):.2e}")
    
    try:
        # CORAL变换：X_t_aligned = (X_t - μ_t) * A + μ_s
        # 其中 A = C_t^(-1/2) * C_s^(1/2)
        
        # 计算C_t的逆平方根
        eigenvals_t, eigenvecs_t = np.linalg.eigh(C_t)
        eigenvals_t = np.maximum(eigenvals_t, regularization)  # 确保正定
        C_t_inv_sqrt = eigenvecs_t @ np.diag(1.0 / np.sqrt(eigenvals_t)) @ eigenvecs_t.T
        
        # 计算C_s的平方根
        eigenvals_s, eigenvecs_s = np.linalg.eigh(C_s)
        eigenvals_s = np.maximum(eigenvals_s, regularization)  # 确保正定
        C_s_sqrt = eigenvecs_s @ np.diag(np.sqrt(eigenvals_s)) @ eigenvecs_s.T
        
        # CORAL变换矩阵
        A = C_t_inv_sqrt @ C_s_sqrt
        
        # 应用变换
        X_t_aligned_scaled = X_t_centered @ A.T + X_s_mean
        
        # 如果使用了标准化，逆标准化回原始空间
        if standardize and scaler is not None:
            X_t_aligned = scaler.inverse_transform(X_t_aligned_scaled)
        else:
            X_t_aligned = X_t_aligned_scaled
        
        # 计算最终CORAL损失
        final_coral_loss = compute_coral_loss(X_s_scaled, X_t_aligned_scaled)
        
        # 计算改进百分比
        if initial_coral_loss > 1e-10:
            improvement = (initial_coral_loss - final_coral_loss) / initial_coral_loss * 100
        else:
            improvement = 0.0
        
        logging.info(f"最终CORAL损失: {final_coral_loss:.6f}")
        logging.info(f"CORAL损失减少: {improvement:.2f}%")
        
        coral_info = {
            'method': 'coral',
            'initial_coral_loss': initial_coral_loss,
            'final_coral_loss': final_coral_loss,
            'improvement_percent': improvement,
            'regularization': regularization,
            'standardized': standardize,
            'transformation_matrix_condition': np.linalg.cond(A),
            'success': True
        }
        
    except np.linalg.LinAlgError as e:
        logging.error(f"CORAL变换失败: {e}")
        logging.info("回退到恒等变换")
        
        X_t_aligned = X_t.copy()
        
        coral_info = {
            'method': 'coral',
            'initial_coral_loss': initial_coral_loss,
            'final_coral_loss': initial_coral_loss,
            'improvement_percent': 0.0,
            'regularization': regularization,
            'standardized': standardize,
            'error': str(e),
            'success': False
        }
    
    return X_t_aligned, coral_info


def class_conditional_coral_transform(X_s: np.ndarray, y_s: np.ndarray,
                                     X_t: np.ndarray, y_t_pseudo: np.ndarray,
                                     regularization: float = 1e-6,
                                     standardize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    类条件CORAL变换：分别对每个类别进行CORAL对齐
    
    参数:
    - X_s: 源域特征
    - y_s: 源域标签
    - X_t: 目标域特征
    - y_t_pseudo: 目标域伪标签（通过预训练模型获得）
    - regularization: 正则化参数
    - standardize: 是否标准化特征
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    - coral_info: 类条件CORAL信息
    """
    logging.info("开始类条件CORAL域适应...")
    
    unique_classes = np.unique(y_s)
    logging.info(f"发现 {len(unique_classes)} 个类别: {unique_classes}")
    
    X_t_aligned = X_t.copy()
    class_coral_losses = {}
    class_improvements = {}
    total_initial_loss = 0.0
    total_final_loss = 0.0
    
    for class_label in unique_classes:
        # 获取当前类别的样本
        s_mask = (y_s == class_label)
        t_mask = (y_t_pseudo == class_label)
        
        X_s_class = X_s[s_mask]
        X_t_class = X_t[t_mask]
        
        if len(X_s_class) < 2 or len(X_t_class) < 2:
            logging.warning(f"类别 {class_label} 样本数不足，跳过CORAL对齐")
            class_coral_losses[class_label] = {'initial': 0.0, 'final': 0.0, 'improvement': 0.0}
            continue
        
        logging.info(f"类别 {class_label}: 源域 {len(X_s_class)} 样本, 目标域 {len(X_t_class)} 样本")
        
        # 对当前类别应用CORAL变换
        X_t_class_aligned, class_info = coral_transform(
            X_s_class, X_t_class, regularization, standardize
        )
        
        # 更新对齐后的特征
        X_t_aligned[t_mask] = X_t_class_aligned
        
        # 记录类别级别的损失
        class_coral_losses[class_label] = {
            'initial': class_info['initial_coral_loss'],
            'final': class_info['final_coral_loss'],
            'improvement': class_info['improvement_percent']
        }
        
        # 累计总损失（按样本数加权）
        weight = len(X_t_class) / len(X_t)
        total_initial_loss += class_info['initial_coral_loss'] * weight
        total_final_loss += class_info['final_coral_loss'] * weight
        
        logging.info(f"类别 {class_label} CORAL损失: {class_info['initial_coral_loss']:.6f} -> {class_info['final_coral_loss']:.6f}")
    
    # 计算总体改进
    if total_initial_loss > 1e-10:
        total_improvement = (total_initial_loss - total_final_loss) / total_initial_loss * 100
    else:
        total_improvement = 0.0
    
    logging.info(f"类条件CORAL总体改进: {total_improvement:.2f}%")
    
    coral_info = {
        'method': 'class_conditional_coral',
        'total_initial_loss': total_initial_loss,
        'total_final_loss': total_final_loss,
        'total_improvement_percent': total_improvement,
        'class_losses': class_coral_losses,
        'num_classes': len(unique_classes),
        'regularization': regularization,
        'standardized': standardize
    }
    
    return X_t_aligned, coral_info


def adaptive_coral_transform(X_s: np.ndarray, X_t: np.ndarray,
                            regularization_range: Tuple[float, float] = (1e-8, 1e-3),
                            n_trials: int = 10,
                            standardize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    自适应CORAL变换：自动选择最佳正则化参数
    
    参数:
    - X_s: 源域特征
    - X_t: 目标域特征
    - regularization_range: 正则化参数搜索范围
    - n_trials: 搜索试验次数
    - standardize: 是否标准化特征
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    - coral_info: 自适应CORAL信息
    """
    logging.info("开始自适应CORAL域适应...")
    
    # 生成正则化参数候选值（对数空间）
    reg_candidates = np.logspace(
        np.log10(regularization_range[0]),
        np.log10(regularization_range[1]),
        n_trials
    )
    
    best_reg = reg_candidates[0]
    best_loss = float('inf')
    best_result = None
    best_info = None
    
    for reg in reg_candidates:
        try:
            X_t_candidate, info = coral_transform(X_s, X_t, reg, standardize)
            
            if info['success'] and info['final_coral_loss'] < best_loss:
                best_loss = info['final_coral_loss']
                best_reg = reg
                best_result = X_t_candidate
                best_info = info
                
        except Exception as e:
            logging.warning(f"正则化参数 {reg:.2e} 失败: {e}")
            continue
    
    if best_result is None:
        logging.error("所有正则化参数都失败，返回原始目标域特征")
        return X_t.copy(), {
            'method': 'adaptive_coral',
            'success': False,
            'error': 'All regularization parameters failed'
        }
    
    logging.info(f"最佳正则化参数: {best_reg:.2e}")
    logging.info(f"最佳CORAL损失: {best_loss:.6f}")
    
    # 更新信息
    best_info['method'] = 'adaptive_coral'
    best_info['best_regularization'] = best_reg
    best_info['regularization_candidates'] = reg_candidates.tolist()
    best_info['n_trials'] = n_trials
    
    return best_result, best_info


def coral_with_feature_selection(X_s: np.ndarray, X_t: np.ndarray,
                                feature_importance: Optional[np.ndarray] = None,
                                top_k_features: Optional[int] = None,
                                regularization: float = 1e-6,
                                standardize: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    结合特征选择的CORAL变换
    
    参数:
    - X_s: 源域特征
    - X_t: 目标域特征
    - feature_importance: 特征重要性分数（可选）
    - top_k_features: 选择前k个重要特征（可选）
    - regularization: 正则化参数
    - standardize: 是否标准化特征
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    - coral_info: 特征选择CORAL信息
    """
    logging.info("开始特征选择CORAL域适应...")
    
    n_features = X_s.shape[1]
    
    # 如果没有提供特征重要性，使用方差作为重要性指标
    if feature_importance is None:
        feature_importance = np.var(X_s, axis=0) + np.var(X_t, axis=0)
        logging.info("使用特征方差作为重要性指标")
    
    # 如果没有指定top_k，使用所有特征
    if top_k_features is None:
        top_k_features = n_features
    
    top_k_features = min(top_k_features, n_features)
    
    # 选择最重要的特征
    top_indices = np.argsort(feature_importance)[-top_k_features:]
    
    logging.info(f"选择前 {top_k_features} 个重要特征进行CORAL对齐")
    
    # 在选定特征上应用CORAL
    X_s_selected = X_s[:, top_indices]
    X_t_selected = X_t[:, top_indices]
    
    X_t_selected_aligned, coral_info = coral_transform(
        X_s_selected, X_t_selected, regularization, standardize
    )
    
    # 构建完整的对齐特征矩阵
    X_t_aligned = X_t.copy()
    X_t_aligned[:, top_indices] = X_t_selected_aligned
    
    # 更新信息
    coral_info['method'] = 'coral_with_feature_selection'
    coral_info['selected_features'] = top_indices.tolist()
    coral_info['n_selected_features'] = top_k_features
    coral_info['feature_importance_used'] = feature_importance is not None
    
    return X_t_aligned, coral_info 