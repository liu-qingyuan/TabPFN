import numpy as np
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from typing import Dict, Any, Tuple
import logging

# Distribution difference metric calculation functions
def calculate_kl_divergence(X_source: np.ndarray, X_target: np.ndarray, bins: int = 20, epsilon: float = 1e-10) -> Tuple[float, Dict[str, float]]:
    """计算KL散度"""
    n_features = X_source.shape[1]
    kl_per_feature: Dict[str, float] = {}

    for i in range(n_features):
        x_s = X_source[:, i]
        x_t = X_target[:, i]

        min_val = min(np.min(x_s), np.min(x_t))
        max_val = max(np.max(x_s), np.max(x_t))
        bin_range = (min_val, max_val)

        hist_s, _ = np.histogram(x_s, bins=bins, range=bin_range, density=True)
        hist_t, _ = np.histogram(x_t, bins=bins, range=bin_range, density=True)

        hist_s = hist_s + epsilon
        hist_t = hist_t + epsilon

        hist_s = hist_s / np.sum(hist_s)
        hist_t = hist_t / np.sum(hist_t)

        kl_s_t = entropy(hist_s, hist_t)
        kl_t_s = entropy(hist_t, hist_s)

        kl_per_feature[f'feature_{i}'] = float((kl_s_t + kl_t_s) / 2) # Ensure float type

    kl_div = float(np.mean(list(kl_per_feature.values())))
    return kl_div, kl_per_feature

def calculate_wasserstein_distances(X_source: np.ndarray, X_target: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """计算Wasserstein距离"""
    n_features = X_source.shape[1]
    wasserstein_per_feature: Dict[str, float] = {}

    for i in range(n_features):
        x_s = X_source[:, i]
        x_t = X_target[:, i]

        w_dist = wasserstein_distance(x_s, x_t)
        wasserstein_per_feature[f'feature_{i}'] = float(w_dist) # Ensure float type

    avg_wasserstein = float(np.mean(list(wasserstein_per_feature.values())))
    return avg_wasserstein, wasserstein_per_feature

def compute_mmd_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
    """计算MMD"""
    n_x = X.shape[0]
    n_y = Y.shape[0]

    # Handle cases with insufficient samples
    if n_x <= 1 or n_y <= 1:
         return 0.0

    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)

    # Ensure terms are calculated only if there are enough samples for the denominator
    term_xx = (np.sum(K_xx) - np.trace(K_xx)) / (n_x * (n_x - 1)) if n_x > 1 else 0.0
    term_yy = (np.sum(K_yy) - np.trace(K_yy)) / (n_y * (n_y - 1)) if n_y > 1 else 0.0
    term_xy = 2 * np.mean(K_xy) if n_x > 0 and n_y > 0 else 0.0 # np.mean handles division by sample counts

    mmd_squared = term_xx + term_yy - term_xy

    return float(np.sqrt(max(mmd_squared, 0)))


def compute_domain_discrepancy(X_source: np.ndarray, X_target: np.ndarray) -> Dict[str, Any]:
    """计算域间差异指标"""
    # Ensure there are samples before calculating distances/means/covariances
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

    mean_dist = float(np.mean(cdist(X_source, X_target)))
    mean_diff = float(np.linalg.norm(np.mean(X_source, axis=0) - np.mean(X_target, axis=0)))

    # Compute covariance only if enough samples (at least 2 for cov)
    cov_source = np.cov(X_source, rowvar=False) if X_source.shape[0] > 1 else np.zeros((X_source.shape[1], X_source.shape[1]))
    cov_target = np.cov(X_target, rowvar=False) if X_target.shape[0] > 1 else np.zeros((X_target.shape[1], X_target.shape[1]))
    cov_diff = float(np.linalg.norm(cov_source - cov_target, 'fro'))

    X_s_mean = np.mean(X_source, axis=0, keepdims=True)
    X_t_mean = np.mean(X_target, axis=0, keepdims=True)
    kernel_mean_diff = float(np.exp(-0.5 * np.sum((X_s_mean - X_t_mean)**2)))

    mmd_value = compute_mmd_kernel(X_source, X_target)
    kl_div, kl_per_feature = calculate_kl_divergence(X_source, X_target)
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

# The detect_outliers function is not directly a metric calculation but a related utility.
# For now, we will keep it here.
def detect_outliers(X_source: np.ndarray, X_target: np.ndarray, percentile: int = 95) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """检测异常点"""
    if X_source.shape[0] == 0 or X_target.shape[0] == 0:
        logging.warning("Cannot detect outliers with empty input arrays.")
        return np.array([]), np.array([]), np.array([]), np.array([])

    min_dist_source = np.min(cdist(X_source, X_target), axis=1)
    min_dist_target = np.min(cdist(X_target, X_source), axis=1)

    source_threshold = np.percentile(min_dist_source, percentile)
    target_threshold = np.percentile(min_dist_target, percentile)

    source_outliers = np.where(min_dist_source > source_threshold)[0]
    target_outliers = np.where(min_dist_target > target_threshold)[0]

    return source_outliers, target_outliers, min_dist_source, min_dist_target 