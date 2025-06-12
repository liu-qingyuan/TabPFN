import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel # For MMD
import logging

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """计算所有评估指标"""
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'acc_0': conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0,
        'acc_1': conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    }

def print_metrics(dataset_name: str, metrics: dict):
    """打印评估指标"""
    logging.info(f"{dataset_name}准确率 (Accuracy): {metrics['acc']:.4f}")
    logging.info(f"{dataset_name} AUC: {metrics['auc']:.4f}")
    logging.info(f"{dataset_name} F1分数: {metrics['f1']:.4f}")
    logging.info(f"{dataset_name}类别0准确率: {metrics['acc_0']:.4f}")
    logging.info(f"{dataset_name}类别1准确率: {metrics['acc_1']:.4f}")


# 添加阈值优化函数
def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, dict]:
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
        'specificity': 1 - fpr[optimal_idx]  # TNR at optimal threshold
    }
    
    # 计算混淆矩阵，获取每类准确率
    cm = confusion_matrix(y_true, y_pred_optimal)
    if cm.shape[0] == 2 and cm.shape[1] == 2:  # 确保是二分类
        optimal_metrics['acc_0'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        optimal_metrics['acc_1'] = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    
    return optimal_threshold, optimal_metrics

def calculate_kl_divergence(X_source: np.ndarray, X_target: np.ndarray, bins: int = 20, epsilon: float = 1e-10) -> tuple[float, dict]:
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
    kl_per_feature = {}
    
    for i in range(n_features):
        # 提取第i个特征
        x_s = X_source[:, i]
        x_t = X_target[:, i]
        
        # 确定共同的区间范围
        min_val = min(np.min(x_s), np.min(x_t))
        max_val = max(np.max(x_s), np.max(x_t))
        bin_range = (min_val, max_val)
        
        # 计算直方图
        hist_s, _ = np.histogram(x_s, bins=bins, range=bin_range, density=True) # bin_edges unused
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
        kl_per_feature[f'feature_{i}'] = (kl_s_t + kl_t_s) / 2
    
    # 计算平均KL散度
    kl_div = np.mean(list(kl_per_feature.values())) if kl_per_feature else 0.0
    
    return kl_div, kl_per_feature

def calculate_wasserstein_distances(X_source: np.ndarray, X_target: np.ndarray) -> tuple[float, dict]:
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
    wasserstein_per_feature = {}
    
    for i in range(n_features):
        # 提取第i个特征
        x_s = X_source[:, i]
        x_t = X_target[:, i]
        
        # 计算Wasserstein距离
        w_dist = wasserstein_distance(x_s, x_t)
        wasserstein_per_feature[f'feature_{i}'] = w_dist
    
    # 计算平均Wasserstein距离
    avg_wasserstein = np.mean(list(wasserstein_per_feature.values())) if wasserstein_per_feature else 0.0
    
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
    from sklearn.metrics.pairwise import rbf_kernel
    
    # 样本数
    n_x = X.shape[0]
    n_y = Y.shape[0]

    if n_x == 0 or n_y == 0: # Handle empty inputs
        return 0.0
    
    # 计算核矩阵
    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)
    
    # 计算MMD
    # Handle cases where n_x or n_y is 1 to avoid division by zero or (n_x - 1) = 0
    term_xx = (np.sum(K_xx) - np.trace(K_xx)) / (n_x * (n_x - 1)) if n_x > 1 else 0
    term_yy = (np.sum(K_yy) - np.trace(K_yy)) / (n_y * (n_y - 1)) if n_y > 1 else 0
    term_xy = 2 * np.mean(K_xy) if n_x > 0 and n_y > 0 else 0
    
    mmd_squared = term_xx + term_yy - term_xy
    
    return np.sqrt(max(mmd_squared, 0))

def compute_domain_discrepancy(X_source: np.ndarray, X_target: np.ndarray) -> dict:
    """
    计算源域和目标域之间的分布差异度量
    
    参数:
    - X_source: 源域特征
    - X_target: 目标域特征
    
    返回:
    - discrepancy_metrics: 包含多种差异度量的字典
    """
    # 1. 平均距离
    mean_dist = np.mean(cdist(X_source, X_target)) if X_source.size > 0 and X_target.size > 0 else 0.0
    
    # 2. 均值差异
    mean_diff = np.linalg.norm(np.mean(X_source, axis=0) - np.mean(X_target, axis=0)) if X_source.size > 0 and X_target.size > 0 else 0.0
    
    # 3. 协方差矩阵距离
    cov_source = np.cov(X_source, rowvar=False) if X_source.shape[0] > 1 else np.eye(X_source.shape[1])
    cov_target = np.cov(X_target, rowvar=False) if X_target.shape[0] > 1 else np.eye(X_target.shape[1])
    cov_diff = np.linalg.norm(cov_source - cov_target, 'fro')
    
    # 4. 最大平均差异(简化版)
    X_s_mean = np.mean(X_source, axis=0, keepdims=True) if X_source.size > 0 else np.zeros((1, X_source.shape[1]))
    X_t_mean = np.mean(X_target, axis=0, keepdims=True) if X_target.size > 0 else np.zeros((1, X_target.shape[1]))
    kernel_mean_diff = np.exp(-0.5 * np.sum((X_s_mean - X_t_mean)**2))
    
    # 5. Maximum Mean Discrepancy (MMD)
    mmd_value = compute_mmd_kernel(X_source, X_target)
    
    # 6. KL散度 (平均每个特征的)
    kl_div, _ = calculate_kl_divergence(X_source, X_target) # kl_per_feature unused
    
    # 7. Wasserstein距离 (平均每个特征的)
    wasserstein_dist, _ = calculate_wasserstein_distances(X_source, X_target) # wasserstein_per_feature unused
    
    return {
        'mean_distance': mean_dist,
        'mean_difference': mean_diff,
        'covariance_difference': cov_diff,
        'kernel_mean_difference': kernel_mean_diff,
        'mmd': mmd_value,
        'kl_divergence': kl_div,
        'wasserstein_distance': wasserstein_dist
    }

def detect_outliers(X_source: np.ndarray, X_target: np.ndarray, percentile: int = 95) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    检测跨分布的异常点
    
    参数:
    - X_source: 源域特征
    - X_target: 目标域特征
    - percentile: 认为是异常点的百分位数阈值
    
    返回:
    - source_outliers: 源域异常点索引
    - target_outliers: 目标域异常点索引
    - source_distances: 源域样本到最近目标域样本的距离
    - target_distances: 目标域样本到最近源域样本的距离
    """
    if X_source.size == 0 or X_target.size == 0: # Handle empty arrays
        return np.array([]), np.array([]), np.array([]), np.array([])

    # 计算每个源域样本到最近目标域样本的距离
    min_dist_source = np.min(cdist(X_source, X_target), axis=1)
    
    # 计算每个目标域样本到最近源域样本的距离
    min_dist_target = np.min(cdist(X_target, X_source), axis=1)
    
    # 根据百分位数确定阈值
    source_threshold = np.percentile(min_dist_source, percentile) if min_dist_source.size > 0 else 0
    target_threshold = np.percentile(min_dist_target, percentile) if min_dist_target.size > 0 else 0
    
    # 找出超过阈值的样本索引
    source_outliers = np.where(min_dist_source > source_threshold)[0] if min_dist_source.size > 0 else np.array([])
    target_outliers = np.where(min_dist_target > target_threshold)[0] if min_dist_target.size > 0 else np.array([])
    
    logging.info(f"源域异常点数量: {len(source_outliers)}/{len(X_source)} ({len(source_outliers)/len(X_source) if len(X_source) > 0 else 0:.2%})")
    logging.info(f"目标域异常点数量: {len(target_outliers)}/{len(X_target)} ({len(target_outliers)/len(X_target) if len(X_target) > 0 else 0:.2%})")
    
    return source_outliers, target_outliers, min_dist_source, min_dist_target