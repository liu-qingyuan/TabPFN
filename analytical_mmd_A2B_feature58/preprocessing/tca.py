#!/usr/bin/env python3
"""
TCA (Transfer Component Analysis) 域适应算法实现

TCA利用MMD作为目标，学习一个映射（投影矩阵），将源域和目标域映射到共同的低维子空间中，
从而缩小域间的差异。这是一个基于核方法的无监督域适应算法。

参考文献:
Pan, S. J., Tsang, I. W., Kwok, J. T., & Yang, Q. (2011). 
Domain adaptation via transfer component analysis. 
IEEE Transactions on Neural Networks, 22(2), 199-210.
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh


def compute_mmd_matrix(X_s: np.ndarray, X_t: np.ndarray, gamma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算MMD矩阵和核矩阵
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - X_t: 目标域特征 [n_samples_target, n_features]
    - gamma: RBF核参数
    
    返回:
    - K: 核矩阵 [(n_s + n_t), (n_s + n_t)]
    - L: MMD矩阵 [(n_s + n_t), (n_s + n_t)]
    """
    n_s = X_s.shape[0]
    n_t = X_t.shape[0]
    
    # 合并数据
    X_combined = np.vstack([X_s, X_t])
    
    # 计算RBF核矩阵
    K = rbf_kernel(X_combined, gamma=gamma)
    
    # 构造MMD矩阵L - 使用更简洁的向量化方式
    e = np.vstack((np.full((n_s, 1), 1.0/n_s), np.full((n_t, 1), -1.0/n_t)))
    L = e @ e.T
    
    # 检查矩阵的数值稳定性
    L_rank = np.linalg.matrix_rank(L)
    if L_rank < min(n_s + n_t - 1, 2):
        logging.warning(f"MMD矩阵L的秩过低: {L_rank}, 可能影响数值稳定性")
    
    return K, L


def median_heuristic_gamma(X: np.ndarray, sample_size: int = 1000) -> float:
    """
    使用中值启发式计算RBF核参数gamma
    
    参数:
    - X: 输入特征矩阵
    - sample_size: 采样大小（避免计算量过大）
    
    返回:
    - gamma: 计算得到的gamma值
    """
    if X.shape[0] > sample_size:
        # 随机采样
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # 计算成对距离
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(X_sample)
    
    # 取上三角矩阵（排除对角线）
    upper_tri_indices = np.triu_indices_from(distances, k=1)
    pairwise_distances = distances[upper_tri_indices]
    
    # 计算中值距离
    median_distance = np.median(pairwise_distances)
    
    # gamma = 1 / (2 * median_distance^2)
    gamma = 1.0 / (2 * median_distance ** 2) if median_distance > 0 else 1.0
    
    return gamma


def tca_transform(X_s: np.ndarray, X_t: np.ndarray,
                 subspace_dim: int = 10,
                 gamma: Optional[float] = None,
                 regularization: float = 1e-3,
                 standardize: bool = True,
                 center_kernel: bool = True,
                 use_sparse_solver: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    使用TCA算法进行域适应
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - X_t: 目标域特征 [n_samples_target, n_features]
    - subspace_dim: 子空间维度
    - gamma: RBF核参数（如果为None，使用中值启发式）
    - regularization: 正则化参数
    - standardize: 是否标准化特征
    - center_kernel: 是否中心化核矩阵
    - use_sparse_solver: 是否使用稀疏特征值求解器
    
    返回:
    - X_s_transformed: 变换后的源域特征
    - X_t_transformed: 变换后的目标域特征
    - tca_info: TCA变换信息
    """
    logging.info("开始TCA域适应...")
    
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
        X_combined = np.vstack([X_s_scaled, X_t_scaled])
        gamma = median_heuristic_gamma(X_combined)
        logging.info(f"使用中值启发式计算gamma: {gamma:.6f}")
    else:
        logging.info(f"使用指定gamma: {gamma:.6f}")
    
    # 限制子空间维度
    max_dim = min(n_s + n_t - 1, n_features)
    subspace_dim = min(subspace_dim, max_dim)
    logging.info(f"子空间维度: {subspace_dim}")
    
    try:
        # 计算核矩阵和MMD矩阵
        K, L = compute_mmd_matrix(X_s_scaled, X_t_scaled, gamma)
        
        # 计算初始MMD损失
        initial_mmd = np.trace(K @ L)
        logging.info(f"初始MMD损失: {initial_mmd:.6f}")
        
        # 核矩阵处理
        n_total = n_s + n_t
        if center_kernel:
            # 中心化核矩阵
            H = np.eye(n_total) - np.ones((n_total, n_total)) / n_total
            K_processed = H @ K @ H
            logging.info("已中心化核矩阵")
        else:
            K_processed = K
            logging.info("未中心化核矩阵")
        
        # 添加正则化
        K_reg = K_processed + regularization * np.eye(n_total)
        
        # 求解广义特征值问题: K_reg @ A = lambda @ L @ K @ A
        # 等价于: (K_reg)^(-1) @ L @ K @ A = (1/lambda) @ A
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
                    eigenvals, eigenvecs = eigsh(
                        M, 
                        k=subspace_dim, 
                        which='SM',  # 最小特征值
                        maxiter=5000,  # 大幅增加迭代次数
                        tol=1e-4,      # 放宽收敛容差
                        ncv=min(max(2*subspace_dim, 20), n_total-1)  # 优化Krylov子空间维度
                    )
                    selected_indices = np.arange(subspace_dim)
                    logging.info("使用稀疏特征值求解器成功")
                except Exception as e:
                    logging.warning(f"稀疏求解器失败，使用密集求解器: {e}")
                    eigenvals, eigenvecs = eigh(M)
                    sorted_indices = np.argsort(eigenvals)
                    selected_indices = sorted_indices[:subspace_dim]
                    logging.info("使用密集特征值求解器成功")
            else:
                # 使用密集特征值求解器
                eigenvals, eigenvecs = eigh(M)
                sorted_indices = np.argsort(eigenvals)
                selected_indices = sorted_indices[:subspace_dim]
                logging.info("使用密集特征值求解器")
            
            # 投影矩阵
            A = eigenvecs[:, selected_indices]
            
        except np.linalg.LinAlgError:
            logging.warning("矩阵求逆失败，使用伪逆")
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
        
        # 计算最终MMD损失
        final_mmd = np.trace(K_transformed.T @ L @ K_transformed)
        
        # 计算改进百分比
        if abs(initial_mmd) > 1e-10:
            improvement = (initial_mmd - final_mmd) / abs(initial_mmd) * 100
        else:
            improvement = 0.0
        
        logging.info(f"最终MMD损失: {final_mmd:.6f}")
        logging.info(f"MMD损失减少: {improvement:.2f}%")
        
        tca_info = {
            'method': 'tca',
            'subspace_dim': subspace_dim,
            'gamma': gamma,
            'regularization': regularization,
            'standardized': standardize,
            'initial_mmd': initial_mmd,
            'final_mmd': final_mmd,
            'improvement_percent': improvement,
            'eigenvalues': eigenvals[selected_indices].tolist(),
            'success': True
        }
        
    except Exception as e:
        logging.error(f"TCA变换失败: {e}")
        logging.info("回退到恒等变换")
        
        # 回退策略：返回原始特征（可能降维）
        if subspace_dim < n_features:
            # 使用PCA降维
            from sklearn.decomposition import PCA
            pca = PCA(n_components=subspace_dim)
            X_combined = np.vstack([X_s_scaled, X_t_scaled])
            X_combined_reduced = pca.fit_transform(X_combined)
            X_s_transformed = X_combined_reduced[:n_s, :]
            X_t_transformed = X_combined_reduced[n_s:, :]
        else:
            X_s_transformed = X_s_scaled
            X_t_transformed = X_t_scaled
        
        tca_info = {
            'method': 'tca',
            'subspace_dim': subspace_dim,
            'gamma': gamma,
            'regularization': regularization,
            'standardized': standardize,
            'initial_mmd': 0.0,
            'final_mmd': 0.0,
            'improvement_percent': 0.0,
            'error': str(e),
            'success': False
        }
    
    return X_s_transformed, X_t_transformed, tca_info


def adaptive_tca_transform(X_s: np.ndarray, X_t: np.ndarray,
                          subspace_dim_range: Tuple[int, int] = (5, 20),
                          gamma_range: Tuple[float, float] = (0.1, 10.0),
                          n_trials: int = 10,
                          standardize: bool = True,
                          **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    自适应TCA变换：自动选择最佳参数
    
    参数:
    - X_s: 源域特征
    - X_t: 目标域特征
    - subspace_dim_range: 子空间维度搜索范围
    - gamma_range: gamma参数搜索范围
    - n_trials: 搜索试验次数
    - standardize: 是否标准化特征
    
    返回:
    - X_s_transformed: 变换后的源域特征
    - X_t_transformed: 变换后的目标域特征
    - tca_info: 自适应TCA信息
    """
    logging.info("开始自适应TCA域适应...")
    
    # 生成参数候选值
    dim_candidates = np.linspace(subspace_dim_range[0], subspace_dim_range[1], 
                                max(3, n_trials // 3), dtype=int)
    gamma_candidates = np.logspace(np.log10(gamma_range[0]), np.log10(gamma_range[1]), 
                                  max(3, n_trials // 3))
    
    best_params = None
    best_mmd = float('inf')
    best_result = None
    best_info = None
    
    # 网格搜索
    for dim in dim_candidates:
        for gamma in gamma_candidates:
            try:
                X_s_cand, X_t_cand, info = tca_transform(
                    X_s, X_t, subspace_dim=dim, gamma=gamma, standardize=standardize
                )
                
                if info['success'] and info['final_mmd'] < best_mmd:
                    best_mmd = info['final_mmd']
                    best_params = {'subspace_dim': dim, 'gamma': gamma}
                    best_result = (X_s_cand, X_t_cand)
                    best_info = info
                    
            except Exception as e:
                logging.warning(f"参数组合 (dim={dim}, gamma={gamma:.3f}) 失败: {e}")
                continue
    
    if best_result is None:
        logging.error("所有参数组合都失败，返回原始特征")
        return X_s.copy(), X_t.copy(), {
            'method': 'adaptive_tca',
            'success': False,
            'error': 'All parameter combinations failed'
        }
    
    logging.info(f"最佳参数: {best_params}")
    logging.info(f"最佳MMD损失: {best_mmd:.6f}")
    
    # 更新信息
    best_info['method'] = 'adaptive_tca'
    best_info['best_params'] = best_params
    best_info['param_candidates'] = {
        'subspace_dims': dim_candidates.tolist(),
        'gammas': gamma_candidates.tolist()
    }
    best_info['n_trials'] = n_trials
    
    return best_result[0], best_result[1], best_info


def tca_with_validation(X_s: np.ndarray, y_s: np.ndarray, X_t: np.ndarray,
                       subspace_dim: int = 10,
                       gamma: Optional[float] = None,
                       validation_split: float = 0.2,
                       standardize: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    带验证的TCA变换：使用源域验证集评估变换效果
    
    参数:
    - X_s: 源域特征
    - y_s: 源域标签
    - X_t: 目标域特征
    - subspace_dim: 子空间维度
    - gamma: RBF核参数
    - validation_split: 验证集比例
    - standardize: 是否标准化特征
    
    返回:
    - X_s_transformed: 变换后的源域特征
    - X_t_transformed: 变换后的目标域特征
    - tca_info: 带验证的TCA信息
    """
    logging.info("开始带验证的TCA域适应...")
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    
    # 划分源域训练集和验证集
    X_s_train, X_s_val, y_s_train, y_s_val = train_test_split(
        X_s, y_s, test_size=validation_split, stratify=y_s, random_state=42
    )
    
    # 应用TCA变换
    X_s_transformed, X_t_transformed, tca_info = tca_transform(
        X_s, X_t, subspace_dim, gamma, standardize=standardize
    )
    
    if not tca_info['success']:
        return X_s_transformed, X_t_transformed, tca_info
    
    # 分离变换后的训练集和验证集
    X_s_train_transformed = X_s_transformed[:len(X_s_train)]
    X_s_val_transformed = X_s_transformed[len(X_s_train):]
    
    try:
        # 在变换后的特征上训练分类器
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_s_train_transformed, y_s_train)
        
        # 在验证集上评估
        y_val_pred = clf.predict(X_s_val_transformed)
        val_accuracy = accuracy_score(y_s_val, y_val_pred)
        
        logging.info(f"变换后验证集准确率: {val_accuracy:.4f}")
        
        # 更新信息
        tca_info['validation_accuracy'] = val_accuracy
        tca_info['validation_split'] = validation_split
        
    except Exception as e:
        logging.warning(f"验证评估失败: {e}")
        tca_info['validation_accuracy'] = 0.0
        tca_info['validation_error'] = str(e)
    
    return X_s_transformed, X_t_transformed, tca_info 