import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import logging
import sys
import scipy.linalg
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import cdist

# 导入可视化模块
from visualize_analytical_CORAL_tsne import (
    visualize_tsne,
    visualize_feature_histograms 
    # compute_domain_discrepancy, # Unused
    # compare_before_after_adaptation # Unused
)

# 设置日志系统
# 先移除 root logger 里所有的 handler
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 创建两个 handler：
stdout_handler = logging.StreamHandler(sys.stdout)  # 处理 INFO 及以上的日志
stderr_handler = logging.StreamHandler(sys.stderr)  # 处理 WARNING 及以上的日志

# 设置不同的日志级别：
stdout_handler.setLevel(logging.INFO)   # 只处理 INFO及以上
stderr_handler.setLevel(logging.WARNING)  # 只处理 WARNING 及以上

# 设置日志格式
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
stdout_handler.setFormatter(formatter)
stderr_handler.setFormatter(formatter)

# 把 handler 添加到 root logger
logging.root.addHandler(stdout_handler)
logging.root.addHandler(stderr_handler)
logging.root.setLevel(logging.INFO)  # 让 root logger 处理 INFO 及以上的日志

def check_matrix_properties(X_data, data_name="Data", cat_idx_list=None, selected_features_names=None):
    """
    检查数据矩阵的属性，重点关注连续特征，以诊断协方差矩阵计算中的潜在问题。
    """
    if X_data is None or X_data.shape[0] == 0:
        logging.warning(f"{data_name} 没有提供数据进行属性检查。")
        return

    # 默认类别特征索引（与 coral_transform 保持一致）
    if cat_idx_list is None:
        cat_idx_list = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22]
    
    all_idx = list(range(X_data.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx_list]

    if not cont_idx:
        logging.info(f"{data_name} 没有连续特征需要检查")
        return

    # 提取连续特征
    X_cont = X_data[:, cont_idx]
    
    # 1. 基本统计信息
    logging.info(f"\n{data_name} 连续特征基本统计:")
    logging.info(f"形状: {X_cont.shape}")
    logging.info(f"数值范围: [{np.min(X_cont):.3f}, {np.max(X_cont):.3f}]")
    
    # 2. 检查是否存在无穷值
    inf_mask = np.isinf(X_cont)
    if np.any(inf_mask):
        inf_counts = np.sum(inf_mask, axis=0)
        inf_features = [(i, cont_idx[i], inf_counts[i]) for i in range(len(cont_idx)) if inf_counts[i] > 0]
        logging.error(f"发现无穷值! 特征索引 (原始,连续) 和计数: {inf_features}")
    
    # 3. 检查是否存在 NaN
    nan_mask = np.isnan(X_cont)
    if np.any(nan_mask):
        nan_counts = np.sum(nan_mask, axis=0)
        nan_features = [(i, cont_idx[i], nan_counts[i]) for i in range(len(cont_idx)) if nan_counts[i] > 0]
        logging.error(f"发现 NaN 值! 特征索引 (原始,连续) 和计数: {nan_features}")
    
    # 4. 检查方差
    variances = np.var(X_cont, axis=0)
    zero_var_idx = np.where(variances < 1e-10)[0]
    if len(zero_var_idx) > 0:
        zero_var_features = [(i, cont_idx[i], variances[i]) for i in zero_var_idx]
        logging.error(f"发现零方差或接近零方差的特征! 特征索引 (原始,连续) 和方差: {zero_var_features}")
    
    # 5. 计算并检查协方差矩阵的条件数
    try:
        cov_matrix = np.cov(X_cont, rowvar=False)
        if cov_matrix.shape[0] > 5:
            logging.info(f"协方差矩阵 (前 5x5 部分):\n{cov_matrix[:5, :5]}")
        else:
            logging.info(f"协方差矩阵:\n{cov_matrix}")
        
        # 计算条件数
        eigvals = np.linalg.eigvals(cov_matrix)
        cond_num = np.max(np.abs(eigvals)) / np.min(np.abs(eigvals))
        logging.info(f"协方差矩阵条件数: {cond_num:.2e}")
        
        if cond_num > 1e10:
            logging.error(f"协方差矩阵病态! 条件数 = {cond_num:.2e}")
            
        # 检查特征值
        min_eigval = np.min(np.abs(eigvals))
        if min_eigval < 1e-10:
            logging.error(f"协方差矩阵接近奇异! 最小特征值 = {min_eigval:.2e}")
            
    except np.linalg.LinAlgError as e:
        logging.error(f"计算协方差矩阵或其特征值时发生错误: {str(e)}")
    except Exception as e:
        logging.error(f"计算协方差矩阵属性时发生未预期的错误: {str(e)}")
        
    # 6. 检查特征之间的相关性
    try:
        corr_matrix = np.corrcoef(X_cont, rowvar=False)
        high_corr_pairs = []
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                if abs(corr_matrix[i,j]) > 0.95:
                    feat_i = selected_features_names[cont_idx[i]] if selected_features_names else f"特征_{cont_idx[i]}"
                    feat_j = selected_features_names[cont_idx[j]] if selected_features_names else f"特征_{cont_idx[j]}"
                    high_corr_pairs.append((feat_i, feat_j, corr_matrix[i,j]))
        
        if high_corr_pairs:
            logging.warning("发现高度相关的特征对:")
            for pair in high_corr_pairs:
                logging.warning(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
                
    except np.linalg.LinAlgError as e:
        logging.error(f"计算相关系数矩阵时发生错误: {str(e)}")
    except Exception as e:
        logging.error(f"计算特征相关性时发生未预期的错误: {str(e)}")
    
    # 7. 检查是否存在无穷值
    inf_mask = np.isinf(X_cont)
    if np.any(inf_mask):
        inf_counts = np.sum(inf_mask, axis=0)
        inf_features = [(i, cont_idx[i], inf_counts[i]) for i in range(len(cont_idx)) if inf_counts[i] > 0]
        logging.error(f"发现无穷值! 特征索引 (原始,连续) 和计数: {inf_features}")
    
    # 8. 检查是否存在 NaN
    nan_mask = np.isnan(X_cont)
    if np.any(nan_mask):
        nan_counts = np.sum(nan_mask, axis=0)
        nan_features = [(i, cont_idx[i], nan_counts[i]) for i in range(len(cont_idx)) if nan_counts[i] > 0]
        logging.error(f"发现 NaN 值! 特征索引 (原始,连续) 和计数: {nan_features}")
    
    # 9. 检查方差
    variances = np.var(X_cont, axis=0)
    zero_var_idx = np.where(variances < 1e-10)[0]
    if len(zero_var_idx) > 0:
        zero_var_features = [(i, cont_idx[i], variances[i]) for i in zero_var_idx]
        logging.error(f"发现零方差或接近零方差的特征! 特征索引 (原始,连续) 和方差: {zero_var_features}")
    
    # 10. 计算并检查协方差矩阵的条件数
    cov_matrix = np.cov(X_cont, rowvar=False)
    X_cont = X_data[:, cont_idx]

    if X_cont.shape[0] < X_cont.shape[1] and X_cont.shape[0] > 0 : # Check if samples < features for continuous part
        logging.warning(f"{data_name} (Continuous Features Part): Number of samples ({X_cont.shape[0]}) is less than number of continuous features ({X_cont.shape[1]}). Covariance matrix will likely be singular or ill-conditioned.")

    logging.info(f"--- Checking Properties for {data_name} (Continuous Features Shape: {X_cont.shape}) ---")

    # Check for zero variance columns
    variances = np.var(X_cont, axis=0)
    # Using a small threshold, verify it's appropriate for scaled data (usually std=1, var=1)
    # If data isn't perfectly scaled, var can be small. 1e-9 is okay.
    zero_var_cols_indices_in_X_cont = np.where(variances < 1e-9)[0] 
    
    if len(zero_var_cols_indices_in_X_cont) > 0:
        if selected_features_names and cont_idx:
            original_feature_names = [selected_features_names[cont_idx[i]] for i in zero_var_cols_indices_in_X_cont]
            logging.warning(f"{data_name} has continuous columns with zero or near-zero variance at continuous_feature_indices: {zero_var_cols_indices_in_X_cont}. Corresponding original feature names: {original_feature_names}")
        else:
            logging.warning(f"{data_name} has continuous columns with zero or near-zero variance at continuous_feature_indices: {zero_var_cols_indices_in_X_cont}.")
    else:
        logging.info(f"{data_name}: No continuous columns with zero or near-zero variance found.")

    # Check rank of the continuous data matrix
    if X_cont.size > 0: # np.linalg.matrix_rank fails on empty array
        rank_X_cont = np.linalg.matrix_rank(X_cont)
        logging.info(f"{data_name} Rank of X_cont (continuous features): {rank_X_cont} (Full rank would be {min(X_cont.shape)})")
    else:
        logging.info(f"{data_name} X_cont is empty, skipping rank check.")


    # Covariance matrix properties for continuous features
    if X_cont.shape[0] > 1 : # Need at least 2 samples to compute covariance
        # Covariance matrix (before regularization)
        Cov_matrix = np.cov(X_cont, rowvar=False)
        
        if np.any(np.isnan(Cov_matrix)) or np.any(np.isinf(Cov_matrix)):
            logging.error(f"{data_name}: Covariance matrix (no regularization) for continuous features contains NaNs or Infs!")
            logging.error(f"Cov_matrix (first 5x5 snippet if large, else full):\n{Cov_matrix[:min(5, Cov_matrix.shape[0]), :min(5, Cov_matrix.shape[1])]}")
            logging.error(f"X_cont (first 5x5 snippet if large, else full):\n{X_cont[:min(5, X_cont.shape[0]), :min(5, X_cont.shape[1])]}")
        else:
            rank_Cov = np.linalg.matrix_rank(Cov_matrix)
            logging.info(f"{data_name} Rank of Covariance Matrix (no regularization, continuous features): {rank_Cov} (Full rank would be {Cov_matrix.shape[0]})")
            
            try:
                cond_Cov = np.linalg.cond(Cov_matrix)
                logging.info(f"{data_name} Condition Number of Covariance Matrix (no regularization, continuous features): {cond_Cov:.2e}")
                if cond_Cov > 1e12: # Threshold for ill-conditioning
                    logging.warning(f"{data_name}: Covariance matrix (no regularization, continuous features) is ill-conditioned or singular (Cond num: {cond_Cov:.2e}).")
            except np.linalg.LinAlgError:
                logging.warning(f"{data_name}: Could not compute condition number for Covariance Matrix (no regularization, continuous features) - likely singular.")

            # Covariance matrix (with regularization, as used in coral_transform for Ct)
            # The regularization in coral_transform is on Ct, which is cov(Xt_cont_centered).
            # Here we check Cov_matrix of X_cont. This is a proxy check.
            Cov_matrix_reg = Cov_matrix + 1e-5 * np.eye(Cov_matrix.shape[0])
            rank_Cov_reg = np.linalg.matrix_rank(Cov_matrix_reg)
            logging.info(f"{data_name} Rank of Covariance Matrix (with 1e-5 regularization, continuous features): {rank_Cov_reg}")
            
            try:
                cond_Cov_reg = np.linalg.cond(Cov_matrix_reg)
                logging.info(f"{data_name} Condition Number of Covariance Matrix (with 1e-5 regularization, continuous features): {cond_Cov_reg:.2e}")
            except np.linalg.LinAlgError:
                logging.warning(f"{data_name}: Could not compute condition number for regularized Covariance Matrix (continuous features).")
            
            try:
                s_vals = scipy.linalg.svdvals(Cov_matrix_reg) 
                logging.info(f"{data_name} Singular values of regularized Cov_matrix (proxy for Ct): min={np.min(s_vals):.2e}, max={np.max(s_vals):.2e}, count_near_zero (<1e-9)={np.sum(s_vals < 1e-9)}")
                if np.any(s_vals < 1e-9):
                    logging.warning(f"{data_name}: Regularized covariance matrix (proxy for Ct) has very small or zero singular values.")
            except Exception as e:
                logging.error(f"{data_name} Error computing singular values for regularized Cov_matrix: {e}")
    elif X_cont.shape[0] <=1 and X_cont.size > 0:
         logging.warning(f"{data_name}: Not enough samples ({X_cont.shape[0]}) in continuous features to compute covariance matrix for detailed check.")
    else:
        logging.info(f"{data_name}: Continuous features part (X_cont) is empty or has zero samples, skipping covariance checks.")
    logging.info(f"--- End of Properties Check for {data_name} ---")


# 定义PKUPH和Mayo模型
class PKUPHModel:
    """
    PKUPH模型的实现
    P(malignant) = e^x / (1+e^x)
    x = -4.496 + (0.07 × Feature2) + (0.676 × Feature48) + (0.736 × Feature49) + 
        (1.267 × Feature4) - (1.615 × Feature50) - (1.408 × Feature53)
    """
    def __init__(self):
        self.intercept_ = -4.496
        self.features = ['Feature2', 'Feature48', 'Feature49', 'Feature4', 'Feature50', 'Feature53']
        self.coefficients = {
            'Feature2': 0.07,
            'Feature48': 0.676,
            'Feature49': 0.736,
            'Feature4': 1.267,
            'Feature50': -1.615,
            'Feature53': -1.408
        }
        
    def fit(self, X, y):
        # 模型已经预定义，不需要训练
        return self
        
    def predict_proba(self, X):
        # 确保X是DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
            
        # 计算线性组合
        x = np.zeros(len(X))
        x += self.intercept_
        
        for feature, coef in self.coefficients.items():
            if feature in X.columns:
                x += coef * X[feature].values
            
        # 计算概率
        p_malignant = 1 / (1 + np.exp(-x))
        
        # 返回两列概率 [P(benign), P(malignant)]
        return np.column_stack((1 - p_malignant, p_malignant))
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class MayoModel:
    """
    Mayo模型的实现
    P(malignant) = e^x / (1+e^x)
    x = -6.8272 + (0.0391 × Feature2) + (0.7917 × Feature3) + (1.3388 × Feature5) + 
        (0.1274 × Feature48) + (1.0407 × Feature49) + (0.7838 × Feature63)
    """
    def __init__(self):
        self.intercept_ = -6.8272
        self.features = ['Feature2', 'Feature3', 'Feature5', 'Feature48', 'Feature49', 'Feature63']
        self.coefficients = {
            'Feature2': 0.0391,
            'Feature3': 0.7917,
            'Feature5': 1.3388,
            'Feature48': 0.1274,
            'Feature49': 1.0407,
            'Feature63': 0.7838
        }
        
    def fit(self, X, y):
        # 模型已经预定义，不需要训练
        return self
        
    def predict_proba(self, X):
        # 确保X是DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
            
        # 计算线性组合
        x = np.zeros(len(X))
        x += self.intercept_
        
        for feature, coef in self.coefficients.items():
            if feature in X.columns:
                x += coef * X[feature].values
            
        # 计算概率
        p_malignant = 1 / (1 + np.exp(-x))
        
        # 返回两列概率 [P(benign), P(malignant)]
        return np.column_stack((1 - p_malignant, p_malignant))
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """计算所有评估指标"""
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'acc_0': conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0,
        'acc_1': conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    }

def print_metrics(dataset_name, metrics):
    """打印评估指标"""
    print(f"{dataset_name}准确率 (Accuracy): {metrics['acc']:.4f}")
    print(f"{dataset_name} AUC: {metrics['auc']:.4f}")
    print(f"{dataset_name} F1分数: {metrics['f1']:.4f}")
    print(f"{dataset_name}类别0准确率: {metrics['acc_0']:.4f}")
    print(f"{dataset_name}类别1准确率: {metrics['acc_1']:.4f}")

# 修改后的CORAL变换函数，区分类别特征和连续特征
def coral_transform(Xs, Xt):
    """
    解析版CORAL变换，直接计算协方差变换矩阵
    将目标域特征对齐到源域协方差框架下
    仅对连续特征进行变换，保留类别特征不变
    
    参数:
    - Xs: 源域特征 [n_samples_source, n_features]
    - Xt: 目标域特征 [n_samples_target, n_features]
    
    返回:
    - Xt_aligned: 对齐后的目标域特征
    """
    # TabPFN自动识别的类别特征索引
    cat_idx = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22]
    all_idx = list(range(Xs.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    
    logging.info(f"类别特征索引: {cat_idx}")
    logging.info(f"连续特征索引: {cont_idx}")
    
    # 提取连续特征
    Xs_cont = Xs[:, cont_idx]
    Xt_cont = Xt[:, cont_idx]
    
    # 对齐均值（可选但推荐）
    mu_s = np.mean(Xs_cont, axis=0)
    mu_t = np.mean(Xt_cont, axis=0)
    Xt_cont_centered = Xt_cont - mu_t + mu_s
    
    # 检查并移除零方差特征
    Xs_cont_var = np.var(Xs_cont, axis=0)
    Xt_cont_var = np.var(Xt_cont_centered, axis=0)
    
    # 找出方差过小的特征（阈值设为1e-8）
    low_var_mask = (Xs_cont_var < 1e-8) | (Xt_cont_var < 1e-8)
    if np.any(low_var_mask):
        low_var_indices = np.where(low_var_mask)[0]
        logging.warning(f"发现低方差连续特征，将被移除: 连续特征索引 {low_var_indices}")
        
        # 移除低方差特征
        valid_mask = ~low_var_mask
        Xs_cont = Xs_cont[:, valid_mask]
        Xt_cont_centered = Xt_cont_centered[:, valid_mask]
        cont_idx_filtered = [cont_idx[i] for i in range(len(cont_idx)) if valid_mask[i]]
        
        logging.info(f"保留的连续特征索引: {cont_idx_filtered}")
    else:
        cont_idx_filtered = cont_idx
    
    # 如果没有足够的连续特征，直接返回原始数据
    if Xs_cont.shape[1] < 2:
        logging.warning("连续特征数量不足，跳过CORAL变换")
        return Xt
    
    # 计算连续特征的协方差矩阵，使用更强的正则化
    regularization = max(1e-3, np.mean(Xs_cont_var) * 0.01)  # 自适应正则化
    Cs = np.cov(Xs_cont, rowvar=False) + regularization*np.eye(Xs_cont.shape[1])
    Ct = np.cov(Xt_cont_centered, rowvar=False) + regularization*np.eye(Xt_cont_centered.shape[1])
    
    # 矩阵平方根 - 目标域到源域的变换
    Ct_inv_sqrt = scipy.linalg.fractional_matrix_power(Ct, -0.5)
    Cs_sqrt = scipy.linalg.fractional_matrix_power(Cs, 0.5)
    
    # 计算转换矩阵 - 先漂白目标域，再上色为源域（仅应用于连续特征）
    A = np.dot(Ct_inv_sqrt, Cs_sqrt)  # 线性映射矩阵
    Xt_cont_aligned = np.dot(Xt_cont_centered, A)
    
    # 将变换后的连续特征与原始类别特征合并
    Xt_aligned = Xt.copy()
    # 只更新那些实际被处理的连续特征
    for i, orig_idx in enumerate(cont_idx_filtered):
        Xt_aligned[:, orig_idx] = Xt_cont_aligned[:, i]
    
    # 记录特征分布变化（只针对处理过的连续特征）
    if len(cont_idx_filtered) > 0:
        # 重新获取原始的连续特征用于比较（仅包含有效特征）
        Xt_cont_original = Xt[:, cont_idx_filtered]
        logging.info(f"连续特征变换前均值差异: {np.mean(np.abs(np.mean(Xs_cont, axis=0) - np.mean(Xt_cont_original, axis=0))):.6f}")
        logging.info(f"连续特征变换后均值差异: {np.mean(np.abs(np.mean(Xs_cont, axis=0) - np.mean(Xt_cont_aligned, axis=0))):.6f}")
        logging.info(f"连续特征变换前标准差差异: {np.mean(np.abs(np.std(Xs_cont, axis=0) - np.std(Xt_cont_original, axis=0))):.6f}")
        logging.info(f"连续特征变换后标准差差异: {np.mean(np.abs(np.std(Xs_cont, axis=0) - np.std(Xt_cont_aligned, axis=0))):.6f}")
    else:
        logging.info("没有有效的连续特征进行CORAL变换")
    
    # 检查类别特征是否保持不变
    if not np.array_equal(Xt[:, cat_idx], Xt_aligned[:, cat_idx]):
        logging.error("错误：类别特征在变换过程中被改变")
    else:
        logging.info("验证成功：类别特征在CORAL变换过程中保持不变")
    
    return Xt_aligned

# 添加阈值优化函数
def optimize_threshold(y_true, y_proba):
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

def run_coral_adaptation_experiment(
    X_source,
    y_source,
    X_target,
    y_target,
    selected_features_names, # Added parameter
    model_name='TabPFN-CORAL',
    tabpfn_params={'device': 'cuda', 'max_time': 60, 'random_state': 42},
    base_path='./results_analytical_coral',
    optimize_decision_threshold=True
):
    """
    运行带有解析版CORAL域适应的TabPFN实验
    
    参数:
    - X_source: 源域特征
    - y_source: 源域标签
    - X_target: 目标域特征
    - y_target: 目标域标签
    - model_name: 模型名称
    - tabpfn_params: TabPFN参数
    - base_path: 结果保存路径
    - optimize_decision_threshold: 是否优化决策阈值
    
    返回:
    - 评估指标
    """
    logging.info(f"\n=== {model_name} Model (Analytical CORAL) ===")
    
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 数据标准化
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # ---- Begin Singularity Check for Target Data ----
    # Define cat_idx as used in coral_transform for consistency in continuous feature identification
    coral_cat_idx = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22] 
    logging.info(f"\nPerforming matrix properties check for target data before CORAL transformation ({model_name})...")
    check_matrix_properties(
        X_target_scaled, 
        data_name=f"Target Data Scaled ({model_name})", 
        cat_idx_list=coral_cat_idx,
        selected_features_names=selected_features_names
    )
    # ---- End Singularity Check ----
    
    # 分析源域和目标域的特征分布差异
    logging.info("Analyzing domain differences before alignment...")
    source_mean = np.mean(X_source_scaled, axis=0)
    target_mean = np.mean(X_target_scaled, axis=0)
    source_std = np.std(X_source_scaled, axis=0)
    target_std = np.std(X_target_scaled, axis=0)
    
    mean_diff = np.mean(np.abs(source_mean - target_mean))
    std_diff = np.mean(np.abs(source_std - target_std))
    logging.info(f"Initial domain difference: Mean diff={mean_diff:.6f}, Std diff={std_diff:.6f}")
    
    # 在源域内划分训练集和测试集
    logging.info("Splitting source domain into train and validation sets (80/20 split)...")
    X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
        X_source_scaled, y_source, test_size=0.2, random_state=42, stratify=y_source
    )
    logging.info(f"Source domain - Training: {X_source_train.shape[0]} samples, Validation: {X_source_val.shape[0]} samples")
    
    # 初始化TabPFN模型（固定，不训练）
    logging.info("Initializing TabPFN model...")
    tabpfn_model = AutoTabPFNClassifier(**tabpfn_params)
    
    # 在源域训练数据上训练TabPFN
    logging.info("Training TabPFN on source domain training data...")
    start_time = time.time()
    tabpfn_model.fit(X_source_train, y_source_train)
    tabpfn_time = time.time() - start_time
    logging.info(f"TabPFN training completed in {tabpfn_time:.2f} seconds")
    
    # 在源域验证集上评估TabPFN
    logging.info("\nEvaluating TabPFN on source domain validation set...")
    y_source_val_pred = tabpfn_model.predict(X_source_val)
    y_source_val_proba = tabpfn_model.predict_proba(X_source_val)
    
    # 计算源域验证指标
    source_metrics = {
        'acc': accuracy_score(y_source_val, y_source_val_pred),
        'auc': roc_auc_score(y_source_val, y_source_val_proba[:, 1]),
        'f1': f1_score(y_source_val, y_source_val_pred)
    }
    
    # 计算混淆矩阵和每类准确率
    conf_matrix = confusion_matrix(y_source_val, y_source_val_pred)
    source_metrics['acc_0'] = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    source_metrics['acc_1'] = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    
    logging.info(f"Source validation - Accuracy: {source_metrics['acc']:.4f}, AUC: {source_metrics['auc']:.4f}, F1: {source_metrics['f1']:.4f}")
    logging.info(f"Source validation - Class 0 Acc: {source_metrics['acc_0']:.4f}, Class 1 Acc: {source_metrics['acc_1']:.4f}")
    
    # 在目标域上进行直接评估（未对齐）
    logging.info("\nEvaluating TabPFN directly on target domain (without alignment)...")
    y_target_pred_direct = tabpfn_model.predict(X_target_scaled)
    y_target_proba_direct = tabpfn_model.predict_proba(X_target_scaled)
    
    # 分析预测分布
    unique_labels, counts = np.unique(y_target_pred_direct, return_counts=True)
    logging.info(f"Direct prediction distribution: {dict(zip(unique_labels, counts))}")
    
    # 计算原始TabPFN在目标域上的性能
    direct_metrics = {
        'acc': accuracy_score(y_target, y_target_pred_direct),
        'auc': roc_auc_score(y_target, y_target_proba_direct[:, 1]),
        'f1': f1_score(y_target, y_target_pred_direct)
    }
    
    # 计算每类准确率
    conf_matrix = confusion_matrix(y_target, y_target_pred_direct)
    direct_metrics['acc_0'] = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    direct_metrics['acc_1'] = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    
    logging.info(f"Direct prediction - Accuracy: {direct_metrics['acc']:.4f}, AUC: {direct_metrics['auc']:.4f}, F1: {direct_metrics['f1']:.4f}")
    logging.info(f"Direct prediction - Class 0 Acc: {direct_metrics['acc_0']:.4f}, Class 1 Acc: {direct_metrics['acc_1']:.4f}")
    
    # 使用解析版CORAL进行特征对齐
    logging.info("\nApplying analytical CORAL transformation...")
    
    # 定义类别特征和连续特征的索引
    coral_cat_idx = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22]
    cont_idx_list = [i for i in range(X_source.shape[1]) if i not in coral_cat_idx]
    
    logging.info(f'类别特征索引: {coral_cat_idx}')
    logging.info(f'连续特征索引: {cont_idx_list}')
    
    # 在执行 CORAL 变换之前检查矩阵属性
    logging.info('\n检查源域数据属性:')
    check_matrix_properties(X_source_scaled, "源域", coral_cat_idx, selected_features_names)
    
    logging.info('\n检查目标域数据属性:')
    check_matrix_properties(X_target_scaled, "目标域", coral_cat_idx, selected_features_names)
    
    # 执行 CORAL 变换
    start_time = time.time()
    X_target_aligned = coral_transform(X_source_scaled, X_target_scaled)
    align_time = time.time() - start_time
    logging.info(f"CORAL transformation completed in {align_time:.2f} seconds")
    
    # 分析对齐前后的特征差异
    mean_diff_after = np.mean(np.abs(np.mean(X_source_scaled, axis=0) - np.mean(X_target_aligned, axis=0)))
    std_diff_after = np.mean(np.abs(np.std(X_source_scaled, axis=0) - np.std(X_target_aligned, axis=0)))
    logging.info(f"After alignment: Mean diff={mean_diff_after:.6f}, Std diff={std_diff_after:.6f}")
    logging.info(f"Difference reduction: Mean: {(mean_diff-mean_diff_after)/mean_diff:.2%}, Std: {(std_diff-std_diff_after)/std_diff:.2%}")
    
    # 在目标域上进行评估
    logging.info("\nEvaluating model on target domain (with analytical CORAL alignment)...")
    
    # 目标域预测（使用CORAL对齐）
    start_time = time.time()
    y_target_pred = tabpfn_model.predict(X_target_aligned)
    y_target_proba = tabpfn_model.predict_proba(X_target_aligned)
    inference_time = time.time() - start_time
    
    # 分析预测分布
    unique_labels, counts = np.unique(y_target_pred, return_counts=True)
    logging.info(f"CORAL-aligned prediction distribution: {dict(zip(unique_labels, counts))}")
    
    # 计算目标域指标
    target_metrics = {
        'acc': accuracy_score(y_target, y_target_pred),
        'auc': roc_auc_score(y_target, y_target_proba[:, 1]),
        'f1': f1_score(y_target, y_target_pred)
    }
    
    # 计算混淆矩阵和每类准确率
    conf_matrix = confusion_matrix(y_target, y_target_pred)
    acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    
    target_metrics['acc_0'] = acc_0
    target_metrics['acc_1'] = acc_1
    
    # 优化决策阈值（可选）
    if optimize_decision_threshold:
        logging.info("\nOptimizing decision threshold using Youden index...")
        optimal_threshold, optimal_metrics = optimize_threshold(y_target, y_target_proba[:, 1])
        
        logging.info(f"Optimal threshold: {optimal_threshold:.4f} (default: 0.5)")
        logging.info(f"Metrics with optimized threshold:")
        logging.info(f"  Accuracy: {optimal_metrics['acc']:.4f} (original: {target_metrics['acc']:.4f})")
        logging.info(f"  F1 Score: {optimal_metrics['f1']:.4f} (original: {target_metrics['f1']:.4f})")
        logging.info(f"  Class 0 Accuracy: {optimal_metrics['acc_0']:.4f} (original: {target_metrics['acc_0']:.4f})")
        logging.info(f"  Class 1 Accuracy: {optimal_metrics['acc_1']:.4f} (original: {target_metrics['acc_1']:.4f})")
        logging.info(f"  Sensitivity: {optimal_metrics['sensitivity']:.4f}")
        logging.info(f"  Specificity: {optimal_metrics['specificity']:.4f}")
        
        # 更新指标
        target_metrics.update({
            'original_acc': target_metrics['acc'],
            'original_f1': target_metrics['f1'],
            'original_acc_0': target_metrics['acc_0'],
            'original_acc_1': target_metrics['acc_1'],
            'optimal_threshold': optimal_threshold
        })
        target_metrics['acc'] = optimal_metrics['acc']
        target_metrics['f1'] = optimal_metrics['f1']
        target_metrics['acc_0'] = optimal_metrics['acc_0']
        target_metrics['acc_1'] = optimal_metrics['acc_1']
        target_metrics['sensitivity'] = optimal_metrics['sensitivity']
        target_metrics['specificity'] = optimal_metrics['specificity']
    
    # 打印结果
    logging.info("\nSource Domain Validation Results:")
    logging.info(f"Accuracy: {source_metrics['acc']:.4f}")
    logging.info(f"AUC: {source_metrics['auc']:.4f}")
    logging.info(f"F1: {source_metrics['f1']:.4f}")
    
    logging.info("\nTarget Domain Evaluation Results (with Analytical CORAL):")
    logging.info(f"Accuracy: {target_metrics['acc']:.4f}")
    logging.info(f"AUC: {target_metrics['auc']:.4f}")
    logging.info(f"F1: {target_metrics['f1']:.4f}")
    logging.info(f"Class 0 Accuracy: {target_metrics['acc_0']:.4f}")
    logging.info(f"Class 1 Accuracy: {target_metrics['acc_1']:.4f}")
    logging.info(f"Inference Time: {inference_time:.4f} seconds")
    
    # 比较对齐前后的性能
    logging.info("\nPerformance Improvement with Analytical CORAL Alignment:")
    logging.info(f"Accuracy: {direct_metrics['acc']:.4f} -> {target_metrics['acc']:.4f} ({target_metrics['acc']-direct_metrics['acc']:.4f})")
    logging.info(f"AUC: {direct_metrics['auc']:.4f} -> {target_metrics['auc']:.4f} ({target_metrics['auc']-direct_metrics['auc']:.4f})")
    logging.info(f"F1: {direct_metrics['f1']:.4f} -> {target_metrics['f1']:.4f} ({target_metrics['f1']-direct_metrics['f1']:.4f})")
    logging.info(f"Class 0 Accuracy: {direct_metrics['acc_0']:.4f} -> {target_metrics['acc_0']:.4f} ({target_metrics['acc_0']-direct_metrics['acc_0']:.4f})")
    logging.info(f"Class 1 Accuracy: {direct_metrics['acc_1']:.4f} -> {target_metrics['acc_1']:.4f} ({target_metrics['acc_1']-direct_metrics['acc_1']:.4f})")
    
    # 保存对齐后的特征
    aligned_features_path = f"{base_path}/{model_name}_aligned_features.npz"
    np.savez(aligned_features_path, 
             X_source=X_source_scaled, 
             X_target=X_target_scaled,
             X_target_aligned=X_target_aligned)
    logging.info(f"Aligned features saved to: {aligned_features_path}")
    
    # 使用导入的可视化模块进行t-SNE可视化
    logging.info("\n使用t-SNE可视化CORAL对齐前后的分布...")
    tsne_save_path = f"{base_path}/{model_name}_tsne.png"
    visualize_tsne(
        X_source=X_source_scaled, 
        X_target=X_target_scaled, 
        y_source=y_source,
        y_target=y_target,
        X_target_aligned=X_target_aligned, 
        title=f'CORAL Domain Adaptation t-SNE Visualization: {model_name}',
        save_path=tsne_save_path
    )
    
    # 绘制特征分布直方图
    logging.info("\n绘制对齐前后的特征分布直方图...")
    hist_save_path = f"{base_path}/{model_name}_histograms.png"
    visualize_feature_histograms(
        X_source=X_source_scaled,
        X_target=X_target_scaled,
        X_target_aligned=X_target_aligned,
        feature_names=selected_features_names,  # Use passed parameter
        n_features_to_plot=None,  # Plot all features
        title=f'Feature Distribution Before and After CORAL Alignment: {model_name}',
        save_path=hist_save_path
    )
    
    # 其余绘图代码保持不变...
    plt.figure(figsize=(15, 5))
    
    # 绘制特征差异
    plt.subplot(1, 3, 1)
    plt.bar(['Before Alignment', 'After Alignment'], [mean_diff, mean_diff_after], color=['red', 'green'])
    plt.title('Mean Feature Difference')
    plt.ylabel('Mean Absolute Difference')
    plt.grid(True, alpha=0.3)
    
    # 绘制特征标准差差异
    plt.subplot(1, 3, 2)
    plt.bar(['Before Alignment', 'After Alignment'], [std_diff, std_diff_after], color=['red', 'green'])
    plt.title('Std Feature Difference')
    plt.ylabel('Mean Absolute Difference')
    plt.grid(True, alpha=0.3)
    
    # 类别分布对比
    plt.subplot(1, 3, 3)
    labels = ['Direct', 'With CORAL']
    class0_accs = [direct_metrics['acc_0'], target_metrics['acc_0']]
    class1_accs = [direct_metrics['acc_1'], target_metrics['acc_1']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, class0_accs, width, label='Class 0')
    plt.bar(x + width/2, class1_accs, width, label='Class 1')
    
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy Comparison')
    plt.xticks(x, labels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/{model_name}_analysis.png", dpi=300)
    plt.close()
    
    # 如果使用了阈值优化，添加ROC曲线和阈值点
    if optimize_decision_threshold and 'optimal_threshold' in target_metrics: # Ensure optimal_threshold exists
        plt.figure(figsize=(8, 6))
        fpr, tpr, thresholds = roc_curve(y_target, y_target_proba[:, 1])
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {target_metrics["auc"]:.4f})')
        
        # 标出最佳阈值点
        optimal_threshold = target_metrics['optimal_threshold'] # Get it from metrics
        optimal_idx = np.where(thresholds >= optimal_threshold)[0][-1] if len(np.where(thresholds >= optimal_threshold)[0]) > 0 else (np.abs(thresholds - optimal_threshold)).argmin()
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', 
                    label=f'Optimal Threshold = {optimal_threshold:.4f}')
        
        # 标出默认阈值0.5对应的点
        default_idx = None
        for i, t in enumerate(thresholds):
            if t <= 0.5:
                default_idx = i
                break
        if default_idx is not None:
            plt.scatter(fpr[default_idx], tpr[default_idx], color='blue', 
                      label='Default Threshold = 0.5')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve and Optimal Decision Threshold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{base_path}/{model_name}_roc_curve.png", dpi=300)
        plt.close()
    
    # 返回完整指标
    return {
        'source': source_metrics,
        'target': target_metrics,
        'direct': direct_metrics,
        'times': {
            'tabpfn': tabpfn_time,
            'align': align_time,
            'inference': inference_time
        },
        'features': {
            'mean_diff_before': mean_diff,
            'std_diff_before': std_diff,
            'mean_diff_after': mean_diff_after,
            'std_diff_after': std_diff_after
        }
    }

# 添加类条件CORAL变换函数
def class_conditional_coral_transform(Xs, ys, Xt, yt_pseudo=None, cat_idx=None, alpha=0.1):
    """
    类条件CORAL变换，对每个类别分别进行协方差对齐
    
    参数:
    - Xs: 源域特征 [n_samples_source, n_features]
    - ys: 源域标签 [n_samples_source]
    - Xt: 目标域特征 [n_samples_target, n_features]
    - yt_pseudo: 目标域伪标签，如果没有则使用源域模型预测 [n_samples_target]
    - cat_idx: 类别特征的索引，如果为None则自动使用TabPFN默认值
    - alpha: 正则化参数，用于平滑类别协方差矩阵
    
    返回:
    - Xt_aligned: 类条件对齐后的目标域特征
    """
    # 如果没有指定类别特征索引，使用TabPFN默认值
    if cat_idx is None:
        cat_idx = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22]
    
    all_idx = list(range(Xs.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    
    # 获取不同的类别
    classes = np.unique(ys)
    n_classes = len(classes)
    
    logging.info(f"执行类条件CORAL对齐，共有{n_classes}个类别")
    
    # 如果目标域没有伪标签，则使用普通CORAL先对齐，然后用源域模型预测
    if yt_pseudo is None:
        logging.info("目标域没有提供伪标签，先使用普通CORAL进行对齐，再用源域模型预测伪标签")
        Xt_temp = coral_transform(Xs, Xt)  # 使用普通CORAL先对齐
        
        # 需要一个模型来预测伪标签，这里假设我们已经有了一个训练好的TabPFN模型
        # 为简化流程，我们在这里使用scikit-learn的最近邻分类器
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xs, ys)
        yt_pseudo = knn.predict(Xt_temp)
        
        logging.info(f"生成的伪标签分布: {np.bincount(yt_pseudo)}")
    
    # 初始化目标域对齐后的特征矩阵
    Xt_aligned = Xt.copy()
    
    # 对每个类别分别执行CORAL
    for c in classes:
        # 获取源域中属于类别c的样本
        Xs_c = Xs[ys == c]
        
        # 获取目标域中属于类别c的样本（根据伪标签）
        class_mask = (yt_pseudo == c)
        Xt_c = Xt[class_mask]
        
        if len(Xt_c) < 2:  # 需要至少2个样本才能计算协方差
            logging.warning(f"类别{c}在目标域中样本数量过少({len(Xt_c)}个)，无法进行类条件CORAL对齐，跳过")
            continue
            
        if len(Xs_c) < 2:  # 需要至少2个样本才能计算协方差
            logging.warning(f"类别{c}在源域中样本数量过少({len(Xs_c)}个)，无法进行类条件CORAL对齐，跳过")
            continue
        
        logging.info(f"对类别{c}进行CORAL对齐：源域{len(Xs_c)}个样本，目标域{len(Xt_c)}个样本")
        
        # 只对连续特征执行CORAL，类别特征保持不变
        Xs_c_cont = Xs_c[:, cont_idx]
        Xt_c_cont = Xt_c[:, cont_idx]
        
        # 对齐均值
        mu_s_c = np.mean(Xs_c_cont, axis=0)
        mu_t_c = np.mean(Xt_c_cont, axis=0)
        Xt_c_cont_centered = Xt_c_cont - mu_t_c + mu_s_c
        
        # 计算类内协方差矩阵
        # 添加正则化项防止矩阵奇异，alpha控制正则化强度
        Cs_c = np.cov(Xs_c_cont, rowvar=False) + alpha * np.eye(len(cont_idx))
        Ct_c = np.cov(Xt_c_cont_centered, rowvar=False) + alpha * np.eye(len(cont_idx))
        
        # 计算变换矩阵 - 先漂白目标域类内协方差，再上色为源域类内协方差
        Ct_c_inv_sqrt = scipy.linalg.fractional_matrix_power(Ct_c, -0.5)
        Cs_c_sqrt = scipy.linalg.fractional_matrix_power(Cs_c, 0.5)
        A_c = np.dot(Ct_c_inv_sqrt, Cs_c_sqrt)
        
        # 应用变换到目标域的类别c样本的连续特征
        Xt_c_cont_aligned = np.dot(Xt_c_cont_centered, A_c)
        
        # 更新对齐后的目标域特征 - 修复索引错误
        # 原代码: Xt_aligned[yt_pseudo == c, cont_idx] = Xt_c_cont_aligned
        # 修复后的代码:
        for i, feat_idx in enumerate(cont_idx):
            Xt_aligned[class_mask, feat_idx] = Xt_c_cont_aligned[:, i]
    
    # 验证类别特征是否保持不变
    if not np.array_equal(Xt[:, cat_idx], Xt_aligned[:, cat_idx]):
        logging.error("错误：类别特征在类条件CORAL变换过程中被改变")
    else:
        logging.info("验证成功：类别特征在类条件CORAL变换过程中保持不变")
    
    return Xt_aligned

# 添加类条件CORAL域适应实验函数
def run_class_conditional_coral_experiment(
    X_source,
    y_source,
    X_target,
    y_target,
    selected_features_names, # Added parameter
    model_name='TabPFN-ClassCORAL',
    tabpfn_params={'device': 'cuda', 'max_time': 60, 'random_state': 42},
    base_path='./results_class_conditional_coral',
    optimize_decision_threshold=True,
    alpha=0.1,
    use_target_labels=False,  # 是否使用部分真实标签，False则使用伪标签
    target_label_ratio=0.1    # 如果使用真实标签，从目标域取多少比例
):
    """
    运行带有类条件CORAL域适应的TabPFN实验
    
    参数:
    - X_source: 源域特征
    - y_source: 源域标签
    - X_target: 目标域特征
    - y_target: 目标域标签
    - model_name: 模型名称
    - tabpfn_params: TabPFN参数
    - base_path: 结果保存路径
    - optimize_decision_threshold: 是否优化决策阈值
    - alpha: 类条件CORAL正则化参数
    - use_target_labels: 是否使用部分目标域真实标签（而非伪标签）
    - target_label_ratio: 使用多少比例的目标域标签
    
    返回:
    - 评估指标
    """
    logging.info(f"\n=== {model_name} Model (Class-Conditional CORAL) ===")
    
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 数据标准化
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # 在源域内划分训练集和测试集
    logging.info("Splitting source domain into train and validation sets (80/20 split)...")
    X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
        X_source_scaled, y_source, test_size=0.2, random_state=42, stratify=y_source
    )
    logging.info(f"Source domain - Training: {X_source_train.shape[0]} samples, Validation: {X_source_val.shape[0]} samples")
    
    # 初始化TabPFN模型
    logging.info("Initializing TabPFN model...")
    tabpfn_model = AutoTabPFNClassifier(**tabpfn_params)
    
    # 在源域训练数据上训练TabPFN
    logging.info("Training TabPFN on source domain training data...")
    start_time = time.time()
    tabpfn_model.fit(X_source_train, y_source_train)
    tabpfn_time = time.time() - start_time
    logging.info(f"TabPFN training completed in {tabpfn_time:.2f} seconds")
    
    # 在源域验证集上评估TabPFN
    logging.info("\nEvaluating TabPFN on source domain validation set...")
    y_source_val_pred = tabpfn_model.predict(X_source_val)
    y_source_val_proba = tabpfn_model.predict_proba(X_source_val)
    
    # 计算源域验证指标
    source_metrics = {
        'acc': accuracy_score(y_source_val, y_source_val_pred),
        'auc': roc_auc_score(y_source_val, y_source_val_proba[:, 1]),
        'f1': f1_score(y_source_val, y_source_val_pred)
    }
    
    # 计算混淆矩阵和每类准确率
    conf_matrix = confusion_matrix(y_source_val, y_source_val_pred)
    source_metrics['acc_0'] = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    source_metrics['acc_1'] = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    
    logging.info(f"Source validation - Accuracy: {source_metrics['acc']:.4f}, AUC: {source_metrics['auc']:.4f}, F1: {source_metrics['f1']:.4f}")
    logging.info(f"Source validation - Class 0 Acc: {source_metrics['acc_0']:.4f}, Class 1 Acc: {source_metrics['acc_1']:.4f}")
    
    # 在目标域上进行直接评估（未对齐）
    logging.info("\nEvaluating TabPFN directly on target domain (without alignment)...")
    y_target_pred_direct = tabpfn_model.predict(X_target_scaled)
    y_target_proba_direct = tabpfn_model.predict_proba(X_target_scaled)
    
    # 分析预测分布
    unique_labels, counts = np.unique(y_target_pred_direct, return_counts=True)
    logging.info(f"Direct prediction distribution: {dict(zip(unique_labels, counts))}")
    
    # 计算直接预测指标
    direct_metrics = evaluate_metrics(y_target, y_target_pred_direct, y_target_proba_direct[:, 1])
    logging.info(f"Direct prediction - Accuracy: {direct_metrics['acc']:.4f}, AUC: {direct_metrics['auc']:.4f}, F1: {direct_metrics['f1']:.4f}")
    logging.info(f"Direct prediction - Class 0 Acc: {direct_metrics['acc_0']:.4f}, Class 1 Acc: {direct_metrics['acc_1']:.4f}")
    
    # 准备类条件CORAL目标域标签/伪标签
    if use_target_labels:
        # 如果使用部分真实标签
        logging.info(f"\nUsing {target_label_ratio:.1%} of target domain true labels for class-conditional CORAL alignment...")
        n_labeled = int(len(y_target) * target_label_ratio)
        
        # 进行分层抽样，确保每个类别都有样本
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-target_label_ratio, random_state=42)
        for labeled_idx, _ in sss.split(X_target_scaled, y_target):
            pass
        
        # 创建伪标签（部分是真实标签，部分是普通CORAL预测的）
        yt_pseudo = np.zeros_like(y_target) - 1  # 初始化为-1表示未知
        yt_pseudo[labeled_idx] = y_target[labeled_idx]  # 填入已知标签
        
        # 对未标记部分使用普通CORAL预测
        # 先使用普通CORAL对齐未标记部分
        X_target_unlabeled = X_target_scaled[yt_pseudo == -1]
        X_target_unlabeled_aligned = coral_transform(X_source_scaled, X_target_unlabeled)
        
        # 对未标记部分进行预测
        yt_pseudo_unlabeled = tabpfn_model.predict(X_target_unlabeled_aligned)
        yt_pseudo[yt_pseudo == -1] = yt_pseudo_unlabeled
        
        logging.info(f"Partial true labels + partial pseudo-labels distribution: {np.bincount(yt_pseudo)}")
    else:
        # 使用完全伪标签
        logging.info("\nGenerating pseudo-labels using standard CORAL for class-conditional CORAL alignment...")
        # 先使用普通CORAL对齐
        X_target_aligned_temp = coral_transform(X_source_scaled, X_target_scaled)
        yt_pseudo = tabpfn_model.predict(X_target_aligned_temp)
        logging.info(f"Generated pseudo-label distribution: {np.bincount(yt_pseudo)}")
    
    # 使用类条件CORAL进行特征对齐
    logging.info("\nApplying class-conditional CORAL transformation...")
    start_time = time.time()
    # TabPFN自动识别的类别特征索引
    cat_idx = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22]
    X_target_aligned = class_conditional_coral_transform(
        X_source_scaled, y_source, X_target_scaled, yt_pseudo, cat_idx, alpha=alpha
    )
    align_time = time.time() - start_time
    logging.info(f"Class-conditional CORAL transformation completed in {align_time:.2f} seconds")
    
    # 在目标域上进行评估
    logging.info("\nEvaluating model on target domain (with class-conditional CORAL alignment)...")
    
    # 目标域预测（使用类条件CORAL对齐）
    start_time = time.time()
    y_target_pred = tabpfn_model.predict(X_target_aligned)
    y_target_proba = tabpfn_model.predict_proba(X_target_aligned)
    inference_time = time.time() - start_time
    
    # 分析类条件CORAL对齐后的预测分布
    unique_labels, counts = np.unique(y_target_pred, return_counts=True)
    logging.info(f"Class-conditional CORAL aligned prediction distribution: {dict(zip(unique_labels, counts))}")
    
    # 计算目标域指标
    target_metrics = evaluate_metrics(y_target, y_target_pred, y_target_proba[:, 1])
    
    # 优化决策阈值（可选）
    if optimize_decision_threshold:
        logging.info("\nOptimizing decision threshold using Youden index...")
        optimal_threshold, optimal_metrics = optimize_threshold(y_target, y_target_proba[:, 1])
        
        logging.info(f"Optimal threshold: {optimal_threshold:.4f} (default: 0.5)")
        logging.info(f"Metrics with optimized threshold:")
        logging.info(f"  Accuracy: {optimal_metrics['acc']:.4f} (original: {target_metrics['acc']:.4f})")
        logging.info(f"  F1 Score: {optimal_metrics['f1']:.4f} (original: {target_metrics['f1']:.4f})")
        logging.info(f"  Class 0 Accuracy: {optimal_metrics['acc_0']:.4f} (original: {target_metrics['acc_0']:.4f})")
        logging.info(f"  Class 1 Accuracy: {optimal_metrics['acc_1']:.4f} (original: {target_metrics['acc_1']:.4f})")
        
        # 更新指标
        target_metrics.update({
            'original_acc': target_metrics['acc'],
            'original_f1': target_metrics['f1'],
            'original_acc_0': target_metrics['acc_0'],
            'original_acc_1': target_metrics['acc_1'],
            'optimal_threshold': optimal_threshold
        })
        target_metrics['acc'] = optimal_metrics['acc']
        target_metrics['f1'] = optimal_metrics['f1']
        target_metrics['acc_0'] = optimal_metrics['acc_0']
        target_metrics['acc_1'] = optimal_metrics['acc_1']
    
    # 打印结果
    logging.info("\nSource Domain Validation Results:")
    logging.info(f"Accuracy: {source_metrics['acc']:.4f}")
    logging.info(f"AUC: {source_metrics['auc']:.4f}")
    logging.info(f"F1: {source_metrics['f1']:.4f}")
    logging.info(f"Class 0 Accuracy: {source_metrics['acc_0']:.4f}")
    logging.info(f"Class 1 Accuracy: {source_metrics['acc_1']:.4f}")
    
    logging.info("\nTarget Domain Evaluation Results (with Class-Conditional CORAL):")
    logging.info(f"Accuracy: {target_metrics['acc']:.4f}")
    logging.info(f"AUC: {target_metrics['auc']:.4f}")
    logging.info(f"F1: {target_metrics['f1']:.4f}")
    logging.info(f"Class 0 Accuracy: {target_metrics['acc_0']:.4f}")
    logging.info(f"Class 1 Accuracy: {target_metrics['acc_1']:.4f}")
    
    # 比较对齐前后的性能
    logging.info("\nPerformance Improvement with Class-Conditional CORAL Alignment:")
    logging.info(f"Accuracy: {direct_metrics['acc']:.4f} -> {target_metrics['acc']:.4f} ({target_metrics['acc']-direct_metrics['acc']:.4f})")
    logging.info(f"AUC: {direct_metrics['auc']:.4f} -> {target_metrics['auc']:.4f} ({target_metrics['auc']-direct_metrics['auc']:.4f})")
    logging.info(f"F1: {direct_metrics['f1']:.4f} -> {target_metrics['f1']:.4f} ({target_metrics['f1']-direct_metrics['f1']:.4f})")
    logging.info(f"Class 0 Accuracy: {direct_metrics['acc_0']:.4f} -> {target_metrics['acc_0']:.4f} ({target_metrics['acc_0']-direct_metrics['acc_0']:.4f})")
    logging.info(f"Class 1 Accuracy: {direct_metrics['acc_1']:.4f} -> {target_metrics['acc_1']:.4f} ({target_metrics['acc_1']-direct_metrics['acc_1']:.4f})")
    
    # 保存对齐后的特征
    aligned_features_path = f"{base_path}/{model_name}_aligned_features.npz"
    np.savez(aligned_features_path, 
             X_source=X_source_scaled, 
             X_target=X_target_scaled,
             X_target_aligned=X_target_aligned,
             yt_pseudo=yt_pseudo)
    logging.info(f"Aligned features saved to: {aligned_features_path}")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 绘制整体指标比较
    plt.subplot(1, 3, 1)
    metrics_labels = ['Accuracy', 'AUC', 'F1']
    metrics_values_direct = [direct_metrics['acc'], direct_metrics['auc'], direct_metrics['f1']]
    metrics_values_coral = [target_metrics['acc'], target_metrics['auc'], target_metrics['f1']]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    plt.bar(x - width/2, metrics_values_direct, width, label='Direct Prediction')
    plt.bar(x + width/2, metrics_values_coral, width, label='Class-Conditional CORAL')
    
    plt.ylabel('Score')
    plt.title('Overall Performance Metrics')
    plt.xticks(x, metrics_labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 绘制每类准确率
    plt.subplot(1, 3, 2)
    class_metrics = ['Class 0 Accuracy', 'Class 1 Accuracy']
    class_values_direct = [direct_metrics['acc_0'], direct_metrics['acc_1']]
    class_values_coral = [target_metrics['acc_0'], target_metrics['acc_1']]
    
    x = np.arange(len(class_metrics))
    
    plt.bar(x - width/2, class_values_direct, width, label='Direct Prediction')
    plt.bar(x + width/2, class_values_coral, width, label='Class-Conditional CORAL')
    
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(x, class_metrics)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 绘制特征差异
    plt.subplot(1, 3, 3)
    
    labels = ['Direct', 'Class-CORAL']
    pred_dist_direct = np.bincount(y_target_pred_direct)
    pred_dist_coral = np.bincount(y_target_pred)
    true_dist = np.bincount(y_target)
    
    # 确保所有柱状图有相同的类别数
    max_classes = max(len(pred_dist_direct), len(pred_dist_coral), len(true_dist))
    if len(pred_dist_direct) < max_classes:
        pred_dist_direct = np.pad(pred_dist_direct, (0, max_classes - len(pred_dist_direct)))
    if len(pred_dist_coral) < max_classes:
        pred_dist_coral = np.pad(pred_dist_coral, (0, max_classes - len(pred_dist_coral)))
    if len(true_dist) < max_classes:
        true_dist = np.pad(true_dist, (0, max_classes - len(true_dist)))
    
    x = np.arange(max_classes)
    width = 0.25
    
    plt.bar(x - width, pred_dist_direct, width, label='Direct Prediction')
    plt.bar(x, pred_dist_coral, width, label='Class-Conditional CORAL')
    plt.bar(x + width, true_dist, width, label='True Distribution')
    
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Prediction Distribution Comparison')
    plt.xticks(x)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/{model_name}_analysis.png", dpi=300)
    plt.close()
    
    # 如果使用了阈值优化，添加ROC曲线
    if optimize_decision_threshold and 'optimal_threshold' in target_metrics: # Ensure optimal_threshold exists
        plt.figure(figsize=(8, 6))
        fpr, tpr, thresholds = roc_curve(y_target, y_target_proba[:, 1])
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {target_metrics["auc"]:.4f})')
        
        # 标出最佳阈值点
        optimal_threshold = target_metrics['optimal_threshold'] # Get it from metrics
        optimal_idx = np.where(thresholds >= optimal_threshold)[0][-1] if len(np.where(thresholds >= optimal_threshold)[0]) > 0 else (np.abs(thresholds - optimal_threshold)).argmin()
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', 
                    label=f'Optimal Threshold = {optimal_threshold:.4f}')
        
        # 标出默认阈值0.5对应的点
        default_idx = None
        for i, t in enumerate(thresholds):
            if t <= 0.5:
                default_idx = i
                break
        if default_idx is not None:
            plt.scatter(fpr[default_idx], tpr[default_idx], color='blue', 
                      label='Default Threshold = 0.5')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curve and Optimal Decision Threshold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{base_path}/{model_name}_roc_curve.png", dpi=300)
        plt.close()
    
    # 特征直方图可视化
    # 定义CORAL类型和域名称用于可视化
    coral_type = "Class-Conditional CORAL with Labels" if use_target_labels else "Class-Conditional CORAL with Pseudo-Labels"
    source_name = "Source Domain"
    target_name = "Target Domain"
    save_base = f"{base_path}/{model_name}"
    
    hist_save_path = f"{base_path}/{model_name}_histograms_visual.png"
    visualize_feature_histograms(
        X_source=X_source_scaled,
        X_target=X_target_scaled,
        X_target_aligned=X_target_aligned,
        feature_names=selected_features_names,  # Ensure this uses the passed parameter
        n_features_to_plot=None,  # Plot all features
        title=f"{coral_type} Feature Distribution: {source_name} → {target_name}",
        save_path=hist_save_path
    )
    
    # 如果有伪标签，创建额外的类条件可视化
    if yt_pseudo is not None:
        # 为每个类别创建特定的可视化（可选）
        classes = np.unique(yt_pseudo)
        logging.info(f"创建{len(classes)}个类别的详细可视化")
        
        for cls in classes:
            # 获取每个类别的索引
            source_idx = y_source == cls
            target_idx = yt_pseudo == cls
            
            if np.sum(source_idx) < 5 or np.sum(target_idx) < 5:
                logging.warning(f"类别{cls}样本数太少，跳过类别特定可视化")
                continue
            
            # 创建类别特定的可视化标题和保存路径
            cls_title = f"{coral_type} Class {cls} Alignment: {source_name} → {target_name}"
            cls_save_path = f"{save_base}_class{cls}_histograms.png"
            
            # 仅对该类别的样本创建直方图
            visualize_feature_histograms(
                X_source=X_source_scaled[source_idx],
                X_target=X_target_scaled[target_idx],
                X_target_aligned=X_target_aligned[target_idx],
                feature_names=selected_features_names, # Use passed parameter
                n_features_to_plot=10,  # 仅显示前10个特征以保持可读性
                title=cls_title,
                save_path=cls_save_path
            )
    
    return {
        'source': source_metrics,
        'target': target_metrics,
        'direct': direct_metrics,
        'times': {
            'tabpfn': tabpfn_time,
            'align': align_time,
            'inference': inference_time
        },
        'other': {
            'yt_pseudo_dist': np.bincount(yt_pseudo).tolist(),
            'use_target_labels': use_target_labels,
            'target_label_ratio': target_label_ratio if use_target_labels else None,
            'alpha': alpha
        }
    }

def calculate_kl_divergence(X_source, X_target, bins=20, epsilon=1e-10):
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
        hist_s, bin_edges = np.histogram(x_s, bins=bins, range=bin_range, density=True)
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
    kl_div = np.mean(list(kl_per_feature.values()))
    
    return kl_div, kl_per_feature

def calculate_wasserstein_distances(X_source, X_target):
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
    avg_wasserstein = np.mean(list(wasserstein_per_feature.values()))
    
    return avg_wasserstein, wasserstein_per_feature

def compute_mmd_kernel(X, Y, gamma=1.0):
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
    
    # 计算核矩阵
    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)
    
    # 计算MMD
    mmd_squared = (np.sum(K_xx) - np.trace(K_xx)) / (n_x * (n_x - 1))
    mmd_squared += (np.sum(K_yy) - np.trace(K_yy)) / (n_y * (n_y - 1))
    mmd_squared -= 2 * np.mean(K_xy)
    
    return np.sqrt(max(mmd_squared, 0))

def compute_domain_discrepancy(X_source, X_target):
    """
    计算源域和目标域之间的分布差异度量
    
    参数:
    - X_source: 源域特征
    - X_target: 目标域特征
    
    返回:
    - discrepancy_metrics: 包含多种差异度量的字典
    """
    # 1. 平均距离
    mean_dist = np.mean(cdist(X_source, X_target))
    
    # 2. 均值差异
    mean_diff = np.linalg.norm(np.mean(X_source, axis=0) - np.mean(X_target, axis=0))
    
    # 3. 协方差矩阵距离
    cov_source = np.cov(X_source, rowvar=False)
    cov_target = np.cov(X_target, rowvar=False)
    cov_diff = np.linalg.norm(cov_source - cov_target, 'fro')
    
    # 4. 最大平均差异(简化版)
    X_s_mean = np.mean(X_source, axis=0, keepdims=True)
    X_t_mean = np.mean(X_target, axis=0, keepdims=True)
    kernel_mean_diff = np.exp(-0.5 * np.sum((X_s_mean - X_t_mean)**2))
    
    # 5. Maximum Mean Discrepancy (MMD)
    mmd_value = compute_mmd_kernel(X_source, X_target)
    
    # 6. KL散度 (平均每个特征的)
    kl_div, kl_per_feature = calculate_kl_divergence(X_source, X_target)
    
    # 7. Wasserstein距离 (平均每个特征的)
    wasserstein_dist, wasserstein_per_feature = calculate_wasserstein_distances(X_source, X_target)
    
    return {
        'mean_distance': mean_dist,
        'mean_difference': mean_diff,
        'covariance_difference': cov_diff,
        'kernel_mean_difference': kernel_mean_diff,
        'mmd': mmd_value,
        'kl_divergence': kl_div,
        'wasserstein_distance': wasserstein_dist
    }

def detect_outliers(X_source, X_target, percentile=95):
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
    # 计算每个源域样本到最近目标域样本的距离
    min_dist_source = np.min(cdist(X_source, X_target), axis=1)
    
    # 计算每个目标域样本到最近源域样本的距离
    min_dist_target = np.min(cdist(X_target, X_source), axis=1)
    
    # 根据百分位数确定阈值
    source_threshold = np.percentile(min_dist_source, percentile)
    target_threshold = np.percentile(min_dist_target, percentile)
    
    # 找出超过阈值的样本索引
    source_outliers = np.where(min_dist_source > source_threshold)[0]
    target_outliers = np.where(min_dist_target > target_threshold)[0]
    
    logging.info(f"源域异常点数量: {len(source_outliers)}/{len(X_source)} ({len(source_outliers)/len(X_source):.2%})")
    logging.info(f"目标域异常点数量: {len(target_outliers)}/{len(X_target)} ({len(target_outliers)/len(X_target):.2%})")
    
    return source_outliers, target_outliers, min_dist_source, min_dist_target


# 主函数
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    
    # 指定设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # 加载所有数据集
    logging.info("\nLoading datasets...")
    logging.info("1. Loading AI4healthcare.xlsx (A)...")
    df_ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    
    logging.info("2. Loading GuangzhouMedicalHospital_features23_no_nan_new_fixed.xlsx (C)...")
    df_guangzhou = pd.read_excel("data/GuangzhouMedicalHospital_features23_no_nan_new_fixed.xlsx")

    # 使用指定的23个特征
    selected_features = [
        'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5',
        'Feature14', 'Feature15', 'Feature17', 'Feature22',
        'Feature39', 'Feature42', 'Feature43', 'Feature45',
        'Feature46', 'Feature47', 'Feature48', 'Feature49',
        'Feature50', 'Feature52', 'Feature53', 'Feature56',
        'Feature57', 'Feature63'
    ]

    logging.info("\n=== Feature Information ===")
    logging.info(f"Number of selected features: {len(selected_features)}")
    logging.info(f"Selected features list: {selected_features}")

    # 检查每个数据集中是否有所有选定的特征
    for dataset_name, dataset in [
        ("AI4health", df_ai4health), 
        ("Guangzhou", df_guangzhou)
    ]:
        missing_features = [f for f in selected_features if f not in dataset.columns]
        if missing_features:
            logging.warning(f"Warning: {dataset_name} missing the following features: {missing_features}")
        else:
            logging.info(f"{dataset_name} contains all selected features")

    # 使用共同特征准备数据
    X_ai4health = df_ai4health[selected_features].copy()
    y_ai4health = df_ai4health["Label"].copy()
    
    X_guangzhou = df_guangzhou[selected_features].copy()
    y_guangzhou = df_guangzhou["Label"].copy()

    # Impute NaNs with column means, then 0 for any columns that were all NaN
    datasets_X = {"AI4health": X_ai4health, "Guangzhou": X_guangzhou}
    logging.info("\\nImputing NaNs in feature matrices (X data)...")
    for name, X_df in datasets_X.items():
        if X_df.isnull().any().any(): # Check if there are any NaNs at all
            logging.info(f"NaNs found in {name}. Performing imputation...")
            # Calculate means for imputation
            means = X_df.mean()
            # First, fill NaNs with column means
            X_df.fillna(means, inplace=True)
            # Then, if any column was all NaN (so its mean was NaN, and fillna(means) didn't change it),
            # fill these remaining NaNs with 0.
            if X_df.isnull().any().any():
                 logging.info(f"Further imputation for {name} (likely all-NaN columns filled with 0).")
                 X_df.fillna(0, inplace=True)
            logging.info(f"NaN imputation complete for {name}.")
        else:
            logging.info(f"No NaNs found in {name}. No imputation needed.")

    # 创建结果目录
    os.makedirs('./results_analytical_coral_A2C', exist_ok=True)
    
    # 是否仅运行TSNE可视化
    only_visualize = False
    
    if only_visualize:
        logging.info("\n\n=== 仅进行CORAL t-SNE可视化 ===")
        
        # 定义域适应配置 (从A到C)
        configs = [
            {
                'name': 'A_to_C',
                'source_name': 'A_AI4health',
                'target_name': 'C_Guangzhou',
                'X_source': X_ai4health,
                'y_source': y_ai4health,
                'X_target': X_guangzhou,
                'y_target': y_guangzhou,
                'npz_file': './results_analytical_coral_A2C/TabPFN-Analytical-CORAL_A_to_C_aligned_features.npz'
            }
        ]
        
        # 为CORAL t-SNE可视化创建数据
        for config in configs:
            logging.info(f"\n\n{'='*50}")
            logging.info(f"CORAL域适应可视化: {config['source_name']} → {config['target_name']}")
            logging.info(f"{'='*50}")
            
            # 加载保存的特征
            npz_data = np.load(config['npz_file'])
            X_source = npz_data['X_source']
            X_target = npz_data['X_target']
            X_target_aligned = npz_data['X_target_aligned']
            
            # 使用t-SNE可视化对齐前后的分布
            tsne_save_path = f"./results_analytical_coral_A2C/TabPFN-Analytical-CORAL_{config['name']}_tsne_detailed.png"
            visualize_tsne(
                X_source=X_source,
                X_target=X_target,
                y_source=config['y_source'],
                y_target=config['y_target'],
                X_target_aligned=X_target_aligned,
                title=f"CORAL Domain Adaptation t-SNE Visualization: {config['source_name']} → {config['target_name']}",
                save_path=tsne_save_path,
                detect_anomalies=True
            )
            
            # 为前5个特征绘制分布直方图
            hist_save_path = f"./results_analytical_coral_A2C/TabPFN-Analytical-CORAL_{config['name']}_histograms.png"
            visualize_feature_histograms(
                X_source=X_source,
                X_target=X_target,
                X_target_aligned=X_target_aligned,
                feature_names=selected_features,  # 这里应该传递特征名称列表
                n_features_to_plot=None,  # Plot all features
                title=f"Feature Distribution Before and After CORAL Alignment: {config['source_name']} → {config['target_name']}",
                save_path=hist_save_path
            )
        
        logging.info("\nCORAL t-SNE可视化完成!")
    else:
        # 运行解析版CORAL域适应实验
        logging.info("\n\n=== Running TabPFN with Analytical CORAL Domain Adaptation ===")
        
        # 定义域适应配置 (从A到C)
        coral_configs = [
            {
                'name': 'A_to_C',
                'source_name': 'A_AI4health',
                'target_name': 'C_Guangzhou',
                'X_source': X_ai4health,
                'y_source': y_ai4health,
                'X_target': X_guangzhou,
                'y_target': y_guangzhou
            }
        ]
        
        # 存储所有实验结果
        all_results = []
        
        # 运行实验
        for config in coral_configs:
            logging.info(f"\n\n{'='*50}")
            logging.info(f"Domain Adaptation: {config['source_name']} → {config['target_name']} (Analytical CORAL)")
            logging.info(f"{'='*50}")
            
            # 运行解析版CORAL域适应实验
            metrics = run_coral_adaptation_experiment(
                X_source=config['X_source'],
                y_source=config['y_source'],
                X_target=config['X_target'],
                y_target=config['y_target'],
                selected_features_names=selected_features, # Pass selected_features
                model_name=f"TabPFN-Analytical-CORAL_{config['name']}",
                base_path='./results_analytical_coral_A2C'
            )
            
            # 保存结果
            result = {
                'source': config['source_name'],
                'target': config['target_name'],
                'source_acc': metrics['source']['acc'],
                'source_auc': metrics['source']['auc'],
                'source_f1': metrics['source']['f1'],
                'target_acc': metrics['target']['acc'],
                'target_auc': metrics['target']['auc'],
                'target_f1': metrics['target']['f1'],
                'target_acc_0': metrics['target']['acc_0'],
                'target_acc_1': metrics['target']['acc_1'],
                'direct_acc': metrics['direct']['acc'],
                'direct_auc': metrics['direct']['auc'], 
                'direct_f1': metrics['direct']['f1'],
                'direct_acc_0': metrics['direct']['acc_0'],
                'direct_acc_1': metrics['direct']['acc_1'],
                'tabpfn_time': metrics['times']['tabpfn'],
                'align_time': metrics['times']['align'],
                'inference_time': metrics['times']['inference'],
                'mean_diff_before': metrics['features']['mean_diff_before'],
                'std_diff_before': metrics['features']['std_diff_before'],
                'mean_diff_after': metrics['features']['mean_diff_after'],
                'std_diff_after': metrics['features']['std_diff_after']
            }
            all_results.append(result)
        
        # 创建结果表格
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('./results_analytical_coral_A2C/all_results.csv', index=False)
        logging.info("Results saved to ./results_analytical_coral_A2C/all_results.csv")
        
        # 可视化比较
        logging.info("\nGenerating visualization for method comparison...")
        for idx, config in enumerate(coral_configs):
            plt.figure(figsize=(12, 8))
            
            # 获取结果
            row = results_df.iloc[idx]
            target_name = config['target_name']
            
            # 绘制准确率和AUC比较
            plt.subplot(2, 2, 1)
            metrics = ['acc', 'auc', 'f1']
            direct_values = [row['direct_acc'], row['direct_auc'], row['direct_f1']]
            coral_values = [row['target_acc'], row['target_auc'], row['target_f1']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            plt.bar(x - width/2, direct_values, width, label='Direct TabPFN')
            plt.bar(x + width/2, coral_values, width, label='With Analytical CORAL')
            
            # 添加数据标签
            for i, v in enumerate(direct_values):
                plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
            for i, v in enumerate(coral_values):
                plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
                
            plt.ylabel('Score')
            plt.title('Performance Metrics')
            plt.xticks(x, ['Accuracy', 'AUC', 'F1'])
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # 绘制每类准确率
            plt.subplot(2, 2, 2)
            class_metrics = ['Class 0 Acc', 'Class 1 Acc']
            direct_values = [row['direct_acc_0'], row['direct_acc_1']]
            coral_values = [row['target_acc_0'], row['target_acc_1']]
            
            x = np.arange(len(class_metrics))
            
            plt.bar(x - width/2, direct_values, width, label='Direct TabPFN')
            plt.bar(x + width/2, coral_values, width, label='With Analytical CORAL')
            
            # 添加数据标签
            for i, v in enumerate(direct_values):
                plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
            for i, v in enumerate(coral_values):
                plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
                
            plt.ylabel('Accuracy')
            plt.title('Per-Class Accuracy')
            plt.xticks(x, class_metrics)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # 绘制特征差异
            plt.subplot(2, 2, 3)
            diff_metrics = ['Mean Diff', 'Std Diff']
            before_values = [row['mean_diff_before'], row['std_diff_before']]
            after_values = [row['mean_diff_after'], row['std_diff_after']]
            
            x = np.arange(len(diff_metrics))
            
            plt.bar(x - width/2, before_values, width, label='Before Alignment')
            plt.bar(x + width/2, after_values, width, label='After Alignment')
            
            # 添加数据标签和减少百分比
            for i, (before, after) in enumerate(zip(before_values, after_values)):
                plt.text(i - width/2, before + 0.01, f'{before:.3f}', ha='center')
                plt.text(i + width/2, after + 0.01, f'{after:.3f}', ha='center')
                reduction = (before - after) / before * 100
                plt.text(i, after/2, f'-{reduction:.1f}%', ha='center', color='black', fontweight='bold')
                
            plt.ylabel('Difference')
            plt.title('Feature Distribution Difference')
            plt.xticks(x, diff_metrics)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # 绘制计算时间
            plt.subplot(2, 2, 4)
            plt.bar(['TabPFN Training', 'CORAL Alignment', 'Inference'], 
                    [row['tabpfn_time'], row['align_time'], row['inference_time']])
            plt.ylabel('Time (seconds)')
            plt.title('Computation Time')
            plt.grid(axis='y', alpha=0.3)
            
            plt.suptitle(f'Analytical CORAL Results: {config["source_name"]} → {target_name}', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(f'./results_analytical_coral_A2C/comparison_{config["name"]}.png', dpi=300)
            plt.close()
            
        logging.info("\nAll results saved to ./results_analytical_coral/ directory")

        # 创建类条件CORAL结果目录
        os.makedirs('./results_class_conditional_coral_A2C', exist_ok=True)
        
        logging.info("\n\n=== 运行带有类条件CORAL域适应的TabPFN ===")
        
        # 存储类条件CORAL实验结果
        class_coral_results = []
        
        for config in coral_configs:
            logging.info(f"\n\n{'='*50}")
            logging.info(f"域适应: {config['source_name']} → {config['target_name']} (类条件CORAL)")
            logging.info(f"{'='*50}")
            
            # 运行类条件CORAL域适应实验
            class_metrics = run_class_conditional_coral_experiment(
                X_source=config['X_source'],
                y_source=config['y_source'],
                X_target=config['X_target'],
                y_target=config['y_target'],
                selected_features_names=selected_features, # Pass selected_features
                model_name=f"TabPFN-ClassCORAL_{config['name']}",
                base_path='./results_class_conditional_coral_A2C',
                optimize_decision_threshold=True,
                alpha=0.1,  # 类条件CORAL正则化参数
                use_target_labels=False  # 不使用真实标签，完全依赖伪标签
            )
            
            # 同样实验但使用10%真实标签
            class_metrics_with_labels = run_class_conditional_coral_experiment(
                X_source=config['X_source'],
                y_source=config['y_source'],
                X_target=config['X_target'],
                y_target=config['y_target'],
                selected_features_names=selected_features, # Pass selected_features
                model_name=f"TabPFN-ClassCORAL_WithLabels_{config['name']}",
                base_path='./results_class_conditional_coral_A2C',
                optimize_decision_threshold=True,
                alpha=0.1,  # 类条件CORAL正则化参数
                use_target_labels=True,  # 使用部分真实标签
                target_label_ratio=0.1   # 使用10%的真实标签
            )
            
            # 保存结果
            class_coral_result = {
                'source': config['source_name'],
                'target': config['target_name'],
                'pseudo_acc': class_metrics['target']['acc'],
                'pseudo_auc': class_metrics['target']['auc'],
                'pseudo_f1': class_metrics['target']['f1'],
                'pseudo_acc_0': class_metrics['target']['acc_0'],
                'pseudo_acc_1': class_metrics['target']['acc_1'],
                'withlabels_acc': class_metrics_with_labels['target']['acc'],
                'withlabels_auc': class_metrics_with_labels['target']['auc'],
                'withlabels_f1': class_metrics_with_labels['target']['f1'], 
                'withlabels_acc_0': class_metrics_with_labels['target']['acc_0'],
                'withlabels_acc_1': class_metrics_with_labels['target']['acc_1'],
                'direct_acc': class_metrics['direct']['acc'],
                'direct_auc': class_metrics['direct']['auc'],
                'direct_f1': class_metrics['direct']['f1']
            }
            class_coral_results.append(class_coral_result)
        
        # 创建结果表格
        class_coral_df = pd.DataFrame(class_coral_results)
        class_coral_df.to_csv('./results_class_conditional_coral_A2C/class_coral_results.csv', index=False)
        logging.info("类条件CORAL结果保存至 ./results_class_conditional_coral_A2C/class_coral_results.csv")
        
        # 进行不同方法的比较
        for idx, config in enumerate(coral_configs):
            plt.figure(figsize=(15, 8))
            
            target_name = config['target_name']
            
            # 获取不同方法的结果
            plain_coral_row = results_df.iloc[idx]
            class_coral_row = class_coral_df.iloc[idx]
            
            # 比较不同方法的准确率
            plt.subplot(2, 2, 1)
            methods = ['Direct', 'Standard CORAL', 'Class-CORAL (Pseudo)', 'Class-CORAL (10% Labels)']
            acc_values = [
                class_coral_row['direct_acc'], 
                plain_coral_row['target_acc'],
                class_coral_row['pseudo_acc'],
                class_coral_row['withlabels_acc']
            ]
            
            plt.bar(methods, acc_values)
            plt.ylabel('Accuracy')
            plt.title('Accuracy Comparison Across Methods')
            plt.xticks(rotation=15)
            plt.grid(axis='y', alpha=0.3)
            
            # 添加数据标签
            for i, v in enumerate(acc_values):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            # 比较不同方法的每类准确率
            plt.subplot(2, 2, 2)
            methods = ['Direct', 'Standard CORAL', 'Class-CORAL (Pseudo)', 'Class-CORAL (10% Labels)']
            
            # 安全获取直接预测的每类准确率
            direct_acc_0 = class_coral_row.get('direct_acc_0', class_metrics['direct'].get('acc_0', 0))
            direct_acc_1 = class_coral_row.get('direct_acc_1', class_metrics['direct'].get('acc_1', 0))
            
            acc0_values = [
                direct_acc_0, 
                plain_coral_row['target_acc_0'],
                class_coral_row['pseudo_acc_0'],
                class_coral_row['withlabels_acc_0']
            ]
            acc1_values = [
                direct_acc_1, 
                plain_coral_row['target_acc_1'],
                class_coral_row['pseudo_acc_1'],
                class_coral_row['withlabels_acc_1']
            ]
            
            x = np.arange(len(methods))
            width = 0.35
            
            plt.bar(x - width/2, acc0_values, width, label='Class 0 Accuracy')
            plt.bar(x + width/2, acc1_values, width, label='Class 1 Accuracy')
            
            plt.ylabel('Accuracy')
            plt.title('Per-Class Accuracy Comparison')
            plt.xticks(x, methods, rotation=15)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # 比较不同方法的AUC
            plt.subplot(2, 2, 3)
            auc_values = [
                class_coral_row['direct_auc'], 
                plain_coral_row['target_auc'],
                class_coral_row['pseudo_auc'],
                class_coral_row['withlabels_auc']
            ]
            
            plt.bar(methods, auc_values)
            plt.ylabel('AUC')
            plt.title('AUC Comparison Across Methods')
            plt.xticks(rotation=15)
            plt.grid(axis='y', alpha=0.3)
            
            # 添加数据标签
            for i, v in enumerate(auc_values):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            # 比较不同方法的F1分数
            plt.subplot(2, 2, 4)
            f1_values = [
                class_coral_row['direct_f1'], 
                plain_coral_row['target_f1'],
                class_coral_row['pseudo_f1'],
                class_coral_row['withlabels_f1']
            ]
            
            plt.bar(methods, f1_values)
            plt.ylabel('F1 Score')
            plt.title('F1 Score Comparison Across Methods')
            plt.xticks(rotation=15)
            plt.grid(axis='y', alpha=0.3)
            
            # 添加数据标签
            for i, v in enumerate(f1_values):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
            
            plt.suptitle(f'CORAL Methods Comparison: {config["source_name"]} → {target_name}', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(f'./results_class_conditional_coral_A2C/methods_comparison_{config["name"]}.png', dpi=300)
            plt.close()
        
        logging.info("\n所有类条件CORAL结果保存至 ./results_class_conditional_coral_A2C/ 目录")
            
    # 运行后续可视化分析
    logging.info("\n\n=== 运行后续的可视化和分析 ===")
    
    # 对所有现有的NPZ文件进行t-SNE可视化
    import glob
    
    # 处理解析版CORAL的NPZ文件
    npz_files = glob.glob('./results_analytical_coral_A2C/*_aligned_features.npz')
    logging.info(f"找到 {len(npz_files)} 个解析版CORAL特征对齐文件用于可视化")
    
    for npz_file in npz_files:
        logging.info(f"处理文件: {npz_file}")
        file_name = os.path.basename(npz_file)
        
        # 从文件名中提取配置信息
        if 'A_to_C' in file_name:
            source_name = 'A_AI4health'
            target_name = 'C_Guangzhou'
            y_source = y_ai4health
            y_target = y_guangzhou
        else:
            logging.warning(f"无法从文件名确定配置: {file_name}, 跳过")
            continue
            
        # 加载特征数据
        data = np.load(npz_file)
        X_source = data['X_source']
        X_target = data['X_target']
        X_target_aligned = data['X_target_aligned']
        
        # 创建保存路径
        save_base = npz_file.replace('_aligned_features.npz', '')
        
        # 使用新模块进行可视化
        # t-SNE可视化
        tsne_save_path = f"{save_base}_tsne_visual.png"
        visualize_tsne(
            X_source=X_source,
            X_target=X_target,
            y_source=y_source,
            y_target=y_target,
            X_target_aligned=X_target_aligned,
            title=f"CORAL Domain Adaptation t-SNE Visualization: {source_name} → {target_name}",
            save_path=tsne_save_path
        )
        
        # 特征直方图可视化
        hist_save_path = f"{save_base}_histograms_visual.png"
        visualize_feature_histograms(
            X_source=X_source,
            X_target=X_target,
            X_target_aligned=X_target_aligned,
            feature_names=selected_features,  # 需要使用特征名称列表
            n_features_to_plot=None,  # Plot all features
            title=f"Feature Distribution Before and After CORAL Alignment: {source_name} → {target_name}",
            save_path=hist_save_path
        )
    
    # 处理类条件CORAL的NPZ文件
    class_coral_npz_files = glob.glob('./results_class_conditional_coral_A2C/*_aligned_features.npz')
    logging.info(f"找到 {len(class_coral_npz_files)} 个类条件CORAL特征对齐文件用于可视化")
    
    for npz_file in class_coral_npz_files:
        logging.info(f"处理文件: {npz_file}")
        file_name = os.path.basename(npz_file)
        
        # 从文件名中提取配置信息
        if 'A_to_C' in file_name:
            source_name = 'A_AI4health'
            target_name = 'C_Guangzhou'
            y_source = y_ai4health
            y_target = y_guangzhou
        else:
            logging.warning(f"无法从文件名确定配置: {file_name}, 跳过")
            continue
        
        # 判断是否为带标签的类条件CORAL
        with_labels = "WithLabels" in file_name
        coral_type = "Class-CORAL with 10% Labels" if with_labels else "Class-CORAL with Pseudo-Labels"
        
        # 加载特征数据
        data = np.load(npz_file)
        X_source = data['X_source']
        X_target = data['X_target']
        X_target_aligned = data['X_target_aligned']
        
        # 如果存在伪标签，则加载
        yt_pseudo = None
        if 'yt_pseudo' in data:
            yt_pseudo = data['yt_pseudo']
            logging.info(f"伪标签分布: {np.bincount(yt_pseudo)}")
        
        # 创建保存路径
        save_base = npz_file.replace('_aligned_features.npz', '')
        
        # t-SNE可视化
        tsne_save_path = f"{save_base}_tsne_visual.png"
        visualize_tsne(
            X_source=X_source,
            X_target=X_target,
            y_source=y_source,
            y_target=y_target,
            X_target_aligned=X_target_aligned,
            title=f"{coral_type} t-SNE Visualization: {source_name} → {target_name}",
            save_path=tsne_save_path
        )
        
        # 特征直方图可视化
        hist_save_path = f"{save_base}_histograms_visual.png"
        visualize_feature_histograms(
            X_source=X_source,
            X_target=X_target,
            X_target_aligned=X_target_aligned,
            feature_names=selected_features,  # 需要使用特征名称列表
            n_features_to_plot=None,  # Plot all features
            title=f"{coral_type} Feature Distribution: {source_name} → {target_name}",
            save_path=hist_save_path
        )
        
        # 如果有伪标签，创建额外的类条件可视化
        if yt_pseudo is not None:
            # 为每个类别创建特定的可视化（可选）
            classes = np.unique(yt_pseudo)
            logging.info(f"创建{len(classes)}个类别的详细可视化")
            
            for cls in classes:
                # 获取每个类别的索引
                source_idx = y_source == cls
                target_idx = yt_pseudo == cls
                
                if np.sum(source_idx) < 5 or np.sum(target_idx) < 5:
                    logging.warning(f"类别{cls}样本数太少，跳过类别特定可视化")
                    continue
                
                # 创建类别特定的可视化标题和保存路径
                cls_title = f"{coral_type} Class {cls} Alignment: {source_name} → {target_name}"
                cls_save_path = f"{save_base}_class{cls}_histograms.png"
                
                # 仅对该类别的样本创建直方图
                visualize_feature_histograms(
                    X_source=X_source[source_idx],
                    X_target=X_target[target_idx],
                    X_target_aligned=X_target_aligned[target_idx],
                    feature_names=selected_features, # 使用全局的selected_features变量
                    n_features_to_plot=10,  # 仅显示前10个特征以保持可读性
                    title=cls_title,
                    save_path=cls_save_path
                )
    
    logging.info("所有可视化完成！")