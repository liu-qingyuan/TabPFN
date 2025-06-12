#!/usr/bin/env python3
"""
JDA (Joint Distribution Adaptation) 域适应算法实现

JDA是TCA的扩展，同时对齐边缘分布和条件分布（类条件）。
通过迭代优化，交替更新伪标签和投影矩阵，实现更精准的联合分布对齐。

参考文献:
Long, M., Wang, J., Ding, G., Sun, J., & Yu, P. S. (2013). 
Transfer feature learning with joint distribution adaptation. 
In Proceedings of the IEEE international conference on computer vision (pp. 2200-2207).
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from scipy.linalg import eigh


def compute_joint_mmd_matrix(X_s: np.ndarray, y_s: np.ndarray, 
                            X_t: np.ndarray, y_t_pseudo: np.ndarray,
                            gamma: float = 1.0, 
                            mu: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算联合MMD矩阵（边缘分布 + 条件分布）
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - y_s: 源域标签
    - X_t: 目标域特征 [n_samples_target, n_features]
    - y_t_pseudo: 目标域伪标签
    - gamma: RBF核参数
    - mu: 边缘分布和条件分布的权重平衡参数
    
    返回:
    - K: 核矩阵 [(n_s + n_t), (n_s + n_t)]
    - L: 联合MMD矩阵 [(n_s + n_t), (n_s + n_t)]
    """
    n_s = X_s.shape[0]
    n_t = X_t.shape[0]
    n_total = n_s + n_t
    
    # 合并数据
    X_combined = np.vstack([X_s, X_t])
    
    # 计算RBF核矩阵
    K = rbf_kernel(X_combined, gamma=gamma)
    
    # 初始化联合MMD矩阵
    L = np.zeros((n_total, n_total))
    
    # 1. 边缘分布MMD矩阵
    L_marginal = np.zeros((n_total, n_total))
    
    # 源域-源域块：1/(n_s^2)
    L_marginal[:n_s, :n_s] = 1.0 / (n_s * n_s)
    
    # 目标域-目标域块：1/(n_t^2)
    L_marginal[n_s:, n_s:] = 1.0 / (n_t * n_t)
    
    # 源域-目标域块：-1/(n_s * n_t)
    L_marginal[:n_s, n_s:] = -1.0 / (n_s * n_t)
    L_marginal[n_s:, :n_s] = -1.0 / (n_s * n_t)
    
    # 2. 条件分布MMD矩阵
    L_conditional = np.zeros((n_total, n_total))
    
    # 获取所有类别
    unique_classes = np.unique(np.concatenate([y_s, y_t_pseudo]))
    n_classes = len(unique_classes)
    
    if n_classes > 1:
        for c in unique_classes:
            # 源域中类别c的样本索引
            s_c_indices = np.where(y_s == c)[0]
            # 目标域中类别c的样本索引（相对于整个数据的索引）
            t_c_indices = np.where(y_t_pseudo == c)[0] + n_s
            
            n_s_c = len(s_c_indices)
            n_t_c = len(t_c_indices)
            
            if n_s_c > 0 and n_t_c > 0:
                # 源域类别c内部
                L_conditional[np.ix_(s_c_indices, s_c_indices)] += 1.0 / (n_s_c * n_s_c)
                
                # 目标域类别c内部
                L_conditional[np.ix_(t_c_indices, t_c_indices)] += 1.0 / (n_t_c * n_t_c)
                
                # 源域类别c与目标域类别c之间
                L_conditional[np.ix_(s_c_indices, t_c_indices)] -= 1.0 / (n_s_c * n_t_c)
                L_conditional[np.ix_(t_c_indices, s_c_indices)] -= 1.0 / (n_s_c * n_t_c)
    
    # 3. 联合MMD矩阵：加权组合（考虑类别样本占比的动态权重）
    if n_classes > 1:
        # 计算类别权重（基于样本占比）
        class_weights = []
        total_samples = len(y_s) + len(y_t_pseudo)
        for c in unique_classes:
            n_s_c = np.sum(y_s == c)
            n_t_c = np.sum(y_t_pseudo == c)
            class_weight = (n_s_c + n_t_c) / total_samples
            class_weights.append(class_weight)
        
        # 使用加权平均而不是简单除以类别数
        weighted_conditional = np.sum(class_weights) * L_conditional
        L = mu * L_marginal + (1 - mu) * weighted_conditional
    else:
        # 只有一个类别时，退化为边缘分布
        L = L_marginal
    
    return K, L


def jda_transform(X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
                 subspace_dim: int = 10,
                 gamma: Optional[float] = None,
                 mu: float = 0.5,
                 max_iterations: int = 10,
                 regularization: float = 1e-3,
                 standardize: bool = True,
                 center_kernel: bool = True,
                 use_sparse_solver: bool = True,
                 confidence_threshold: float = 0.7) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    使用JDA算法进行域适应
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - y_s: 源域标签
    - X_t: 目标域特征 [n_samples_target, n_features]
    - subspace_dim: 子空间维度
    - gamma: RBF核参数（如果为None，使用中值启发式）
    - mu: 边缘分布和条件分布的权重平衡参数
    - max_iterations: 最大迭代次数
    - regularization: 正则化参数
    - standardize: 是否标准化特征
    - center_kernel: 是否中心化核矩阵
    - use_sparse_solver: 是否使用稀疏特征值求解器
    - confidence_threshold: 伪标签置信度阈值
    
    返回:
    - X_s_transformed: 变换后的源域特征
    - X_t_transformed: 变换后的目标域特征
    - jda_info: JDA变换信息
    """
    logging.info("开始JDA域适应...")
    
    n_s = X_s.shape[0]
    n_t = X_t.shape[0]
    n_features = X_s.shape[1]
    
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
    
    # 自动计算gamma（如果未提供）
    if gamma is None:
        from .tca import median_heuristic_gamma
        X_combined = np.vstack([X_s_scaled, X_t_scaled])
        gamma = median_heuristic_gamma(X_combined)
        logging.info(f"使用中值启发式计算gamma: {gamma:.6f}")
    else:
        logging.info(f"使用指定gamma: {gamma:.6f}")
    
    # 限制子空间维度
    max_dim = min(n_s + n_t - 1, n_features)
    subspace_dim = min(subspace_dim, max_dim)
    logging.info(f"子空间维度: {subspace_dim}")
    logging.info(f"权重参数mu: {mu:.3f}")
    logging.info(f"最大迭代次数: {max_iterations}")
    
    # 初始化伪标签（使用简单的最近邻方法）
    from sklearn.neighbors import KNeighborsClassifier
    initial_clf = KNeighborsClassifier(n_neighbors=1)
    initial_clf.fit(X_s_scaled, y_s)
    y_t_pseudo = initial_clf.predict(X_t_scaled)
    
    logging.info(f"初始伪标签分布: {np.bincount(y_t_pseudo.astype(int))}")
    
    # 记录迭代历史
    iteration_history = []
    best_mmd = float('inf')
    best_result = None
    best_info = None
    
    try:
        for iteration in range(max_iterations):
            logging.info(f"JDA迭代 {iteration + 1}/{max_iterations}")
            
            # 计算联合MMD矩阵
            K, L = compute_joint_mmd_matrix(X_s_scaled, y_s, X_t_scaled, y_t_pseudo, gamma, mu)
            
            # 计算当前MMD损失
            current_mmd = np.trace(K @ L)
            logging.info(f"  当前MMD损失: {current_mmd:.6f}")
            
            # 核矩阵处理
            n_total = n_s + n_t
            if center_kernel:
                # 中心化核矩阵
                H = np.eye(n_total) - np.ones((n_total, n_total)) / n_total
                K_processed = H @ K @ H
            else:
                K_processed = K
            
            # 添加正则化
            K_reg = K_processed + regularization * np.eye(n_total)
            
            # 求解广义特征值问题
            try:
                # 计算 K_reg^(-1) @ L @ K
                K_inv = np.linalg.inv(K_reg)
                M = K_inv @ L @ K
                
                # 选择特征值求解方法
                if use_sparse_solver and subspace_dim < n_total // 2:
                    # 使用稀疏特征值求解器求最小的k个特征值
                    try:
                        from scipy.sparse.linalg import eigsh
                        # 增加迭代次数和放宽收敛条件以提高收敛性
                        eigenvals, eigenvecs = eigsh(M, k=subspace_dim, which='SM', 
                                                   maxiter=5000, tol=1e-4)
                        selected_indices = np.arange(subspace_dim)
                        logging.info("  使用稀疏特征值求解器成功")
                    except Exception as e:
                        logging.warning(f"  稀疏求解器失败，使用密集求解器: {e}")
                        eigenvals, eigenvecs = eigh(M)
                        sorted_indices = np.argsort(eigenvals)
                        selected_indices = sorted_indices[:subspace_dim]
                        logging.info("  使用密集特征值求解器成功")
                else:
                    # 使用密集特征值求解器
                    eigenvals, eigenvecs = eigh(M)
                    sorted_indices = np.argsort(eigenvals)
                    selected_indices = sorted_indices[:subspace_dim]
                
                # 投影矩阵
                A = eigenvecs[:, selected_indices]
                
            except np.linalg.LinAlgError:
                logging.warning("  矩阵求逆失败，使用伪逆")
                K_pinv = np.linalg.pinv(K_reg)
                M = K_pinv @ L @ K
                eigenvals, eigenvecs = eigh(M)
                sorted_indices = np.argsort(eigenvals)
                selected_indices = sorted_indices[:subspace_dim]
                A = eigenvecs[:, selected_indices]
            
            # 应用变换
            K_transformed = K @ A
            
            # 分离源域和目标域的变换结果
            X_s_transformed = K_transformed[:n_s, :]
            X_t_transformed = K_transformed[n_s:, :]
            
            # 计算变换后的MMD损失
            final_mmd = np.trace(K_transformed.T @ L @ K_transformed)
            logging.info(f"  变换后MMD损失: {final_mmd:.6f}")
            
            # 记录迭代信息
            iter_info = {
                'iteration': iteration + 1,
                'mmd_before': current_mmd,
                'mmd_after': final_mmd,
                'pseudo_label_distribution': np.bincount(y_t_pseudo.astype(int)).tolist(),
                'eigenvalues': eigenvals[selected_indices].tolist()
            }
            iteration_history.append(iter_info)
            
            # 更新最佳结果
            if final_mmd < best_mmd:
                best_mmd = final_mmd
                best_result = (X_s_transformed.copy(), X_t_transformed.copy())
                best_info = {
                    'best_iteration': iteration + 1,
                    'best_mmd': final_mmd,
                    'best_pseudo_labels': y_t_pseudo.copy()
                }
            
            # 在变换后的特征上重新训练分类器并更新伪标签
            if iteration < max_iterations - 1:  # 最后一次迭代不需要更新伪标签
                try:
                    # 使用更稳健的分类器（支持概率输出）
                    from sklearn.svm import SVC
                    try:
                        # 尝试使用SVM（如果数据量不太大）
                        if len(X_s_transformed) < 1000:
                            clf = SVC(probability=True, random_state=42)
                        else:
                            clf = LogisticRegression(random_state=42, max_iter=1000)
                    except:
                        clf = LogisticRegression(random_state=42, max_iter=1000)
                    
                    clf.fit(X_s_transformed, y_s)
                    
                    # 获取预测概率
                    y_t_proba = clf.predict_proba(X_t_transformed)
                    y_t_pred = clf.predict(X_t_transformed)
                    
                    # 计算置信度（最大概率）
                    confidence_scores = np.max(y_t_proba, axis=1)
                    
                    # 只更新高置信度的伪标签
                    high_confidence_mask = confidence_scores >= confidence_threshold
                    y_t_pseudo_new = y_t_pseudo.copy()
                    y_t_pseudo_new[high_confidence_mask] = y_t_pred[high_confidence_mask]
                    
                    # 检查伪标签是否收敛
                    label_change_ratio = np.mean(y_t_pseudo != y_t_pseudo_new)
                    high_conf_ratio = np.mean(high_confidence_mask)
                    logging.info(f"  伪标签变化比例: {label_change_ratio:.3f}")
                    logging.info(f"  高置信度样本比例: {high_conf_ratio:.3f}")
                    
                    y_t_pseudo = y_t_pseudo_new
                    logging.info(f"  更新后伪标签分布: {np.bincount(y_t_pseudo.astype(int))}")
                    
                    # 如果伪标签变化很小，提前停止
                    if label_change_ratio < 0.01:
                        logging.info("  伪标签收敛，提前停止迭代")
                        break
                        
                except Exception as e:
                    logging.warning(f"  更新伪标签失败: {e}")
                    break
        
        # 使用最佳结果
        if best_result is not None:
            X_s_final, X_t_final = best_result
            
            # 计算总体改进
            initial_mmd = iteration_history[0]['mmd_before'] if iteration_history else 0.0
            if abs(initial_mmd) > 1e-10:
                improvement = (initial_mmd - best_mmd) / abs(initial_mmd) * 100
            else:
                improvement = 0.0
            
            logging.info(f"JDA完成，最佳迭代: {best_info['best_iteration']}")
            logging.info(f"最终MMD损失: {best_mmd:.6f}")
            logging.info(f"MMD损失减少: {improvement:.2f}%")
            
            jda_info = {
                'method': 'jda',
                'subspace_dim': subspace_dim,
                'gamma': gamma,
                'mu': mu,
                'max_iterations': max_iterations,
                'actual_iterations': len(iteration_history),
                'regularization': regularization,
                'standardized': standardize,
                'initial_mmd': initial_mmd,
                'final_mmd': best_mmd,
                'improvement_percent': improvement,
                'best_iteration': best_info['best_iteration'],
                'iteration_history': iteration_history,
                'final_pseudo_labels': best_info['best_pseudo_labels'].tolist(),
                'success': True
            }
            
        else:
            raise ValueError("未找到有效的变换结果")
            
    except Exception as e:
        logging.error(f"JDA变换失败: {e}")
        logging.info("回退到恒等变换")
        
        # 回退策略
        if subspace_dim < n_features:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=subspace_dim)
            X_combined = np.vstack([X_s_scaled, X_t_scaled])
            X_combined_reduced = pca.fit_transform(X_combined)
            X_s_final = X_combined_reduced[:n_s, :]
            X_t_final = X_combined_reduced[n_s:, :]
        else:
            X_s_final = X_s_scaled
            X_t_final = X_t_scaled
        
        jda_info = {
            'method': 'jda',
            'subspace_dim': subspace_dim,
            'gamma': gamma,
            'mu': mu,
            'max_iterations': max_iterations,
            'actual_iterations': 0,
            'regularization': regularization,
            'standardized': standardize,
            'initial_mmd': 0.0,
            'final_mmd': 0.0,
            'improvement_percent': 0.0,
            'error': str(e),
            'success': False
        }
    
    return X_s_final, X_t_final, jda_info


def adaptive_jda_transform(X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
                          subspace_dim_range: Tuple[int, int] = (5, 20),
                          gamma_range: Tuple[float, float] = (0.1, 10.0),
                          mu_range: Tuple[float, float] = (0.1, 0.9),
                          n_trials: int = 12,
                          max_iterations: int = 5,
                          standardize: bool = True,
                          **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    自适应JDA变换：自动选择最佳参数
    
    参数:
    - X_s: 源域特征
    - y_s: 源域标签
    - X_t: 目标域特征
    - subspace_dim_range: 子空间维度搜索范围
    - gamma_range: gamma参数搜索范围
    - mu_range: mu参数搜索范围
    - n_trials: 搜索试验次数
    - max_iterations: 每次试验的最大迭代次数
    - standardize: 是否标准化特征
    
    返回:
    - X_s_transformed: 变换后的源域特征
    - X_t_transformed: 变换后的目标域特征
    - jda_info: 自适应JDA信息
    """
    logging.info("开始自适应JDA域适应...")
    
    # 生成参数候选值
    dim_candidates = np.linspace(subspace_dim_range[0], subspace_dim_range[1], 
                                max(3, n_trials // 4), dtype=int)
    gamma_candidates = np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[1]), 
                                  max(2, n_trials // 4))
    mu_candidates = np.linspace(mu_range[0], mu_range[1], max(2, n_trials // 4))
    
    best_params = None
    best_mmd = float('inf')
    best_result = None
    best_info = None
    
    # 网格搜索
    trial_count = 0
    for dim in dim_candidates:
        for gamma in gamma_candidates:
            for mu in mu_candidates:
                if trial_count >= n_trials:
                    break
                
                try:
                    logging.info(f"试验 {trial_count + 1}/{n_trials}: dim={dim}, gamma={gamma:.3f}, mu={mu:.3f}")
                    
                    X_s_cand, X_t_cand, info = jda_transform(
                        X_s, y_s, X_t, 
                        subspace_dim=dim, 
                        gamma=gamma, 
                        mu=mu,
                        max_iterations=max_iterations,
                        standardize=standardize
                    )
                    
                    if info['success'] and info['final_mmd'] < best_mmd:
                        best_mmd = info['final_mmd']
                        best_params = {'subspace_dim': dim, 'gamma': gamma, 'mu': mu}
                        best_result = (X_s_cand, X_t_cand)
                        best_info = info
                        
                    trial_count += 1
                    
                except Exception as e:
                    logging.warning(f"参数组合失败: {e}")
                    trial_count += 1
                    continue
            
            if trial_count >= n_trials:
                break
        if trial_count >= n_trials:
            break
    
    if best_result is None:
        logging.error("所有参数组合都失败，返回原始特征")
        return X_s.copy(), X_t.copy(), {
            'method': 'adaptive_jda',
            'success': False,
            'error': 'All parameter combinations failed'
        }
    
    logging.info(f"最佳参数: {best_params}")
    logging.info(f"最佳MMD损失: {best_mmd:.6f}")
    
    # 更新信息
    best_info['method'] = 'adaptive_jda'
    best_info['best_params'] = best_params
    best_info['param_candidates'] = {
        'subspace_dims': dim_candidates.tolist(),
        'gammas': gamma_candidates.tolist(),
        'mus': mu_candidates.tolist()
    }
    best_info['n_trials'] = trial_count
    
    return best_result[0], best_result[1], best_info


def jda_with_confidence(X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
                       subspace_dim: int = 10,
                       gamma: Optional[float] = None,
                       mu: float = 0.5,
                       max_iterations: int = 10,
                       confidence_threshold: float = 0.8,
                       standardize: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    带置信度的JDA变换：只使用高置信度的伪标签
    
    参数:
    - X_s: 源域特征
    - y_s: 源域标签
    - X_t: 目标域特征
    - subspace_dim: 子空间维度
    - gamma: RBF核参数
    - mu: 权重平衡参数
    - max_iterations: 最大迭代次数
    - confidence_threshold: 置信度阈值
    - standardize: 是否标准化特征
    
    返回:
    - X_s_transformed: 变换后的源域特征
    - X_t_transformed: 变换后的目标域特征
    - jda_info: 带置信度的JDA信息
    """
    logging.info("开始带置信度的JDA域适应...")
    logging.info(f"置信度阈值: {confidence_threshold:.3f}")
    
    # 首先运行标准JDA
    X_s_trans, X_t_trans, jda_info = jda_transform(
        X_s, y_s, X_t, subspace_dim, gamma, mu, max_iterations, standardize=standardize
    )
    
    if not jda_info['success']:
        return X_s_trans, X_t_trans, jda_info
    
    try:
        # 在最终变换后的特征上训练分类器
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_s_trans, y_s)
        
        # 获取目标域预测的概率
        y_t_proba = clf.predict_proba(X_t_trans)
        y_t_pred = clf.predict(X_t_trans)
        
        # 计算置信度（最大概率）
        confidence_scores = np.max(y_t_proba, axis=1)
        high_confidence_mask = confidence_scores >= confidence_threshold
        
        n_high_confidence = np.sum(high_confidence_mask)
        confidence_ratio = n_high_confidence / len(X_t_trans)
        
        logging.info(f"高置信度样本数: {n_high_confidence}/{len(X_t_trans)} ({confidence_ratio:.1%})")
        
        # 更新信息
        jda_info['method'] = 'jda_with_confidence'
        jda_info['confidence_threshold'] = confidence_threshold
        jda_info['high_confidence_samples'] = n_high_confidence
        jda_info['confidence_ratio'] = confidence_ratio
        jda_info['confidence_scores'] = confidence_scores.tolist()
        jda_info['final_predictions'] = y_t_pred.tolist()
        jda_info['prediction_probabilities'] = y_t_proba.tolist()
        
    except Exception as e:
        logging.warning(f"置信度评估失败: {e}")
        jda_info['confidence_error'] = str(e)
    
    return X_s_trans, X_t_trans, jda_info 