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
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
# 添加MMD计算所需的导入
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.decomposition import KernelPCA
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 导入可视化模块
from visualize_mmd_tsne import (
    visualize_tsne, 
    visualize_feature_histograms,
    histograms_stats_table, 
    compute_domain_discrepancy, 
    compare_before_after_adaptation,
    visualize_mmd_adaptation_results
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

def evaluate_metrics(y_true, y_pred, y_score=None):
    """Calculate all evaluation metrics"""
    metrics = {}
    metrics['acc'] = accuracy_score(y_true, y_pred)
    
    # Class-specific accuracy
    class_0_mask = (y_true == 0)
    class_1_mask = (y_true == 1)
    
    if np.any(class_0_mask):
        metrics['acc_0'] = accuracy_score(y_true[class_0_mask], y_pred[class_0_mask])
    else:
        metrics['acc_0'] = 0.0
    
    if np.any(class_1_mask):
        metrics['acc_1'] = accuracy_score(y_true[class_1_mask], y_pred[class_1_mask])
    else:
        metrics['acc_1'] = 0.0
    
    # F1-Score
    metrics['f1'] = f1_score(y_true, y_pred, average='binary')
    
    # AUC (if probabilities are available)
    if y_score is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_score)
        except:
            metrics['auc'] = 0.5  # Default AUC for random predictions
    else:
        metrics['auc'] = 0.5
    
    return metrics

def print_metrics(prefix, metrics):
    """Print evaluation metrics"""
    print(f"{prefix} - Accuracy: {metrics['acc']:.4f}")
    if 'auc' in metrics:
        print(f"{prefix} - AUC: {metrics['auc']:.4f}")
    print(f"{prefix} - F1 Score: {metrics['f1']:.4f}")
    print(f"{prefix} - Class 0 Accuracy: {metrics['acc_0']:.4f}")
    print(f"{prefix} - Class 1 Accuracy: {metrics['acc_1']:.4f}")
    if 'optimal_threshold' in metrics:
        print(f"{prefix} - Optimal threshold: {metrics['optimal_threshold']:.4f}")

# 添加MMD计算函数
def compute_mmd(X_s, X_t, kernel='rbf', gamma=1.0):
    """
    计算Maximum Mean Discrepancy (MMD)
    
    参数:
    - X_s: 源域特征 [n_samples_source, n_features]
    - X_t: 目标域特征 [n_samples_target, n_features]
    - kernel: 核函数类型，默认'rbf'
    - gamma: 核函数参数，默认1.0
    
    返回:
    - mmd_value: MMD距离值
    """
    XX = pairwise_kernels(X_s, X_s, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(X_t, X_t, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X_s, X_t, metric=kernel, gamma=gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# 添加不同内核的MMD计算
def compute_multiple_kernels_mmd(X_s, X_t, kernels=None, gammas=None):
    """
    使用多个内核计算MMD
    
    参数:
    - X_s: 源域特征
    - X_t: 目标域特征
    - kernels: 核函数列表，如果为None则使用['rbf', 'laplacian', 'polynomial']
    - gammas: gamma参数列表，如果为None则使用[0.1, 1.0, 10.0]
    
    返回:
    - mmd_values: 各个内核下的MMD值字典
    """
    if kernels is None:
        kernels = ['rbf', 'laplacian', 'polynomial']
    if gammas is None:
        gammas = [0.1, 1.0, 10.0]
    
    mmd_values = {}
    
    for kernel in kernels:
        for gamma in gammas:
            key = f"{kernel}_gamma{gamma}"
            try:
                mmd_values[key] = compute_mmd(X_s, X_t, kernel=kernel, gamma=gamma)
            except Exception as e:
                logging.warning(f"无法计算内核 {key} 的MMD: {str(e)}")
                mmd_values[key] = float('nan')
    
    # 找出最小MMD值及其对应的内核
    valid_mmds = {k: v for k, v in mmd_values.items() if not np.isnan(v)}
    if valid_mmds:
        best_kernel = min(valid_mmds.items(), key=lambda x: x[1])[0]
        mmd_values['best_kernel'] = best_kernel
        mmd_values['min_mmd'] = valid_mmds[best_kernel]
    
    return mmd_values

# 基于线性变换的MMD最小化
class MMDLinearTransform:
    """使用线性变换最小化MMD的特征对齐器"""
    
    def __init__(self, input_dim, gamma=1.0, lr=0.01, n_epochs=100, batch_size=64, lambda_reg=0.01):
        self.input_dim = input_dim
        self.gamma = gamma
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = nn.Linear(input_dim, input_dim, bias=True).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.best_model_state = None
        self.best_mmd = float('inf')
        logging.info(f"Initialized MMDLinearTransform with input_dim={input_dim}")
        
    def rbf_kernel(self, X, Y):
        """计算RBF内核矩阵"""
        X_norm = torch.sum(X ** 2, dim=1, keepdim=True)
        Y_norm = torch.sum(Y ** 2, dim=1, keepdim=True)
        XY = torch.mm(X, Y.t())
        dist = X_norm + Y_norm.t() - 2 * XY
        return torch.exp(-self.gamma * dist)
    
    def compute_mmd_torch(self, X_s, X_t):
        """计算PyTorch版本的MMD"""
        K_XX = self.rbf_kernel(X_s, X_s)
        K_YY = self.rbf_kernel(X_t, X_t)
        K_XY = self.rbf_kernel(X_s, X_t)
        
        n_s = X_s.size(0)
        n_t = X_t.size(0)
        
        mmd = K_XX.sum() / (n_s * n_s) + K_YY.sum() / (n_t * n_t) - 2 * K_XY.sum() / (n_s * n_t)
        return mmd
    
    def fit(self, X_s, X_t, cat_idx=None):
        """训练线性变换以最小化MMD"""
        # 确保输入是连续特征
        if X_s.shape[1] != self.input_dim:
            logging.warning(f"输入维度不匹配: X_s.shape[1]={X_s.shape[1]}, self.input_dim={self.input_dim}")
            # 如果输入维度与初始化时不同，重新创建模型以匹配输入维度
            self.input_dim = X_s.shape[1]
            self.model = nn.Linear(self.input_dim, self.input_dim, bias=True).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
            logging.info(f"已调整模型输入维度为 {self.input_dim}")
        
        # 记录原始MMD
        self.original_mmd = compute_mmd(X_s, X_t, kernel='rbf', gamma=self.gamma)
        logging.info(f"原始MMD: {self.original_mmd:.6f}")
        
        # 转换为PyTorch张量
        X_s_tensor = torch.FloatTensor(X_s).to(self.device)
        X_t_tensor = torch.FloatTensor(X_t).to(self.device)
        
        # 创建数据加载器
        source_dataset = TensorDataset(X_s_tensor)
        target_dataset = TensorDataset(X_t_tensor)
        source_loader = DataLoader(source_dataset, batch_size=self.batch_size, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=self.batch_size, shuffle=True)
        
        # 训练循环
        for epoch in range(self.n_epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0
            
            # 重新创建源数据和目标数据的迭代器
            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            
            while True:
                try:
                    source_batch = next(source_iter)[0]
                except StopIteration:
                    break
                    
                try:
                    target_batch = next(target_iter)[0]
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)[0]
                
                self.optimizer.zero_grad()
                
                # 对目标域应用变换
                target_transformed = self.model(target_batch)
                
                # 计算MMD损失
                mmd_loss = self.compute_mmd_torch(source_batch, target_transformed)
                
                # 添加正则化项以保持变换接近单位矩阵
                weights = next(self.model.parameters())
                identity_reg = self.lambda_reg * torch.norm(weights - torch.eye(weights.shape[0], weights.shape[1]).to(self.device))
                
                loss = mmd_loss + identity_reg
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # 计算整个数据集上的MMD
            self.model.eval()
            with torch.no_grad():
                X_t_transformed = self.model(X_t_tensor).cpu().numpy()
                current_mmd = compute_mmd(X_s, X_t_transformed, kernel='rbf', gamma=self.gamma)
            
            # 保存最佳模型
            if current_mmd < self.best_mmd:
                self.best_mmd = current_mmd
                self.best_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
                logging.info(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.6f}, MMD: {current_mmd:.6f}")
        
        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        logging.info(f"训练完成。原始MMD: {self.original_mmd:.6f}, 最终MMD: {self.best_mmd:.6f}")
        return self
    
    def transform(self, X, cat_idx=None):
        """
        对输入数据应用学习到的变换
        仅转换连续特征，保持分类特征不变
        """
        # 分离类别特征和连续特征
        if cat_idx is None:
            cat_idx = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22]
        all_idx = list(range(X.shape[1]))
        cont_idx = [i for i in all_idx if i not in cat_idx]
        
        # 初始化结果数组，拷贝原始数据
        X_transformed = X.copy()
        
        # 提取连续特征
        X_cont = X[:, cont_idx]
        
        # 确保特征维度匹配
        if X_cont.shape[1] != self.input_dim:
            logging.error(f"变换时特征维度不匹配: 输入={X_cont.shape[1]}, 模型={self.input_dim}")
            # 如果维度不匹配，我们需要调整模型维度或者返回原始数据
            if X_cont.shape[1] > 0:  # 如果有连续特征
                logging.warning("尝试重新初始化模型以匹配输入维度")
                try:
                    # 保存当前学习率等参数
                    old_params = {
                        'gamma': self.gamma,
                        'lr': self.lr,
                        'lambda_reg': self.lambda_reg
                    }
                    # 重新创建模型
                    self.input_dim = X_cont.shape[1]
                    self.model = nn.Linear(self.input_dim, self.input_dim, bias=True).to(self.device)
                    self.optimizer = optim.Adam(self.model.parameters(), lr=old_params['lr'], weight_decay=1e-5)
                    logging.info(f"已调整模型输入维度为 {self.input_dim}")
                    # 我们这里不重新训练模型，只是应用一个简单的恒等变换
                    # 如有需要，可以在这里重新训练模型
                except Exception as e:
                    logging.error(f"重新初始化模型失败: {str(e)}")
                    return X_transformed
            else:
                return X_transformed
        
        # 转换为PyTorch张量
        X_cont_tensor = torch.FloatTensor(X_cont).to(self.device)
        
        # 应用变换
        self.model.eval()
        with torch.no_grad():
            X_cont_transformed = self.model(X_cont_tensor).cpu().numpy()
        
        # 更新连续特征
        X_transformed[:, cont_idx] = X_cont_transformed
        
        return X_transformed

# 基于核PCA的MMD最小化
def mmd_kernel_pca_transform(X_s, X_t, cat_idx=None, n_components=None, kernel='rbf', gamma=1.0):
    """
    使用核PCA进行MMD最小化的特征变换
    
    参数:
    - X_s: 源域特征
    - X_t: 目标域特征
    - cat_idx: 类别特征索引，默认使用TabPFN默认值
    - n_components: PCA组件数，默认None（使用所有组件）
    - kernel: 核函数类型
    - gamma: 核函数参数
    
    返回:
    - X_t_aligned: 对齐后的目标域特征
    """
    # 分离类别特征和连续特征
    if cat_idx is None:
        cat_idx = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22]
    all_idx = list(range(X_s.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    
    # 仅对连续特征应用变换
    X_s_cont = X_s[:, cont_idx]
    X_t_cont = X_t[:, cont_idx]
    
    # 计算初始MMD
    initial_mmd = compute_mmd(X_s_cont, X_t_cont, kernel=kernel, gamma=gamma)
    logging.info(f"初始MMD (核PCA前): {initial_mmd:.6f}")
    
    # 设置组件数
    if n_components is None:
        n_components = min(X_s_cont.shape[1], X_s_cont.shape[0], X_t_cont.shape[0])
    
    # 在合并数据上训练核PCA
    combined_data = np.vstack((X_s_cont, X_t_cont))
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma)
    kpca.fit(combined_data)
    
    # 变换数据到核PCA空间
    X_s_kpca = kpca.transform(X_s_cont)
    X_t_kpca = kpca.transform(X_t_cont)
    
    # 在核PCA空间计算MMD
    kpca_mmd = compute_mmd(X_s_kpca, X_t_kpca, kernel=kernel, gamma=gamma)
    logging.info(f"核PCA空间中的MMD: {kpca_mmd:.6f}")
    
    # 将目标域数据对齐到源域分布
    # 简单地计算均值和协方差调整
    X_s_kpca_mean = np.mean(X_s_kpca, axis=0)
    X_t_kpca_mean = np.mean(X_t_kpca, axis=0)
    
    X_s_kpca_std = np.std(X_s_kpca, axis=0) + 1e-6  # 防止除零
    X_t_kpca_std = np.std(X_t_kpca, axis=0) + 1e-6
    
    # 标准化目标域数据，然后重新缩放为源域分布
    X_t_kpca_aligned = ((X_t_kpca - X_t_kpca_mean) / X_t_kpca_std) * X_s_kpca_std + X_s_kpca_mean
    
    # 在对齐后的核PCA空间计算MMD
    aligned_kpca_mmd = compute_mmd(X_s_kpca, X_t_kpca_aligned, kernel=kernel, gamma=gamma)
    logging.info(f"对齐后在核PCA空间中的MMD: {aligned_kpca_mmd:.6f}")
    
    # 逆变换回原始特征空间 (近似)
    # 对于核PCA，逆变换是一个近似，通常性能有限
    # 我们将使用原始连续特征，但调整它们的分布
    X_t_cont_mean = np.mean(X_t_cont, axis=0)
    X_s_cont_mean = np.mean(X_s_cont, axis=0)
    
    X_t_cont_std = np.std(X_t_cont, axis=0) + 1e-6
    X_s_cont_std = np.std(X_s_cont, axis=0) + 1e-6
    
    X_t_cont_aligned = ((X_t_cont - X_t_cont_mean) / X_t_cont_std) * X_s_cont_std + X_s_cont_mean
    
    # 在原始空间计算最终MMD
    final_mmd = compute_mmd(X_s_cont, X_t_cont_aligned, kernel=kernel, gamma=gamma)
    logging.info(f"核PCA对齐后在原始空间的MMD: {final_mmd:.6f}")
    
    # 构建最终对齐的目标域特征
    X_t_aligned = X_t.copy()
    X_t_aligned[:, cont_idx] = X_t_cont_aligned
    
    # 验证类别特征是否保持不变
    if not np.array_equal(X_t[:, cat_idx], X_t_aligned[:, cat_idx]):
        logging.error("错误：类别特征在核PCA变换过程中被改变")
    else:
        logging.info("验证成功：类别特征在核PCA变换过程中保持不变")
    
    return X_t_aligned, {"initial_mmd": initial_mmd, "kpca_mmd": kpca_mmd, 
                         "aligned_kpca_mmd": aligned_kpca_mmd, "final_mmd": final_mmd}

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



# 添加类条件CORAL域适应实验函数

# 使用MMD变换对齐特征
def mmd_transform(X_s, X_t, method='linear', cat_idx=None, **kwargs):
    """
    Main function for feature alignment using MMD
    
    Parameters:
    - X_s: Source domain features [n_samples_source, n_features]
    - X_t: Target domain features [n_samples_target, n_features]
    - method: Alignment method, options: 'linear', 'kpca', 'mean_std'
    - cat_idx: Index of categorical features
    - **kwargs: Other parameters passed to specific alignment methods
    
    Returns:
    - X_t_aligned: Aligned target domain features
    - mmd_info: MMD related information
    """
    if cat_idx is None:
        cat_idx = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22]
    
    all_idx = list(range(X_s.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    
    # Calculate initial MMD (using continuous features only)
    X_s_cont = X_s[:, cont_idx]
    X_t_cont = X_t[:, cont_idx]
    
    initial_mmd = compute_mmd(X_s_cont, X_t_cont, kernel='rbf', gamma=1.0)
    logging.info(f"Initial MMD: {initial_mmd:.6f}")
    
    # Initialize variables
    final_mmd = None
    mmd_reduction = None
    
    # Choose alignment method
    if method == 'linear':
        # Use linear transformation for alignment
        gamma = kwargs.get('gamma', 1.0)
        lr = kwargs.get('lr', 0.01)
        n_epochs = kwargs.get('n_epochs', 100)
        batch_size = kwargs.get('batch_size', 64)
        lambda_reg = kwargs.get('lambda_reg', 0.01)
        
        # 初始化MMDLinearTransform，使用连续特征的维度
        mmd_linear = MMDLinearTransform(
            input_dim=len(cont_idx),
            gamma=gamma,
            lr=lr,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lambda_reg=lambda_reg
        )
        
        # 训练时只使用连续特征
        mmd_linear.fit(X_s_cont, X_t_cont)
        
        # 创建结果数组
        X_t_aligned = X_t.copy()
        
        # 提取连续特征，变换后再放回
        X_t_cont_aligned = mmd_linear.model(torch.FloatTensor(X_t_cont).to(mmd_linear.device)).detach().cpu().numpy()
        X_t_aligned[:, cont_idx] = X_t_cont_aligned
        
        align_info = {
            'method': 'linear',
            'n_iter': n_epochs,
            'lambda_reg': lambda_reg
        }
        
    elif method == 'kpca':
        # Use kernel PCA alignment
        kernel = kwargs.get('kernel', 'rbf')
        gamma = kwargs.get('gamma', 1.0)
        n_components = kwargs.get('n_components', None)
        
        X_t_aligned, kpca_info = mmd_kernel_pca_transform(
            X_s, X_t, cat_idx, n_components, kernel, gamma
        )
        
        align_info = {
            'method': 'kpca',
            'kernel': kernel,
            'gamma': gamma,
            'n_components': n_components
        }
        
        # Return early since we already have the full aligned X_t
        return X_t_aligned, {
            'method': method,
            'initial_mmd': kpca_info['initial_mmd'],
            'final_mmd': kpca_info['final_mmd'],
            'reduction': (kpca_info['initial_mmd'] - kpca_info['final_mmd']) / kpca_info['initial_mmd'] * 100,
            'align_info': align_info
        }
        
    elif method == 'mean_std':
        # Use simple mean-std alignment (similar to CORAL but without covariance alignment)
        X_t_cont_aligned = np.copy(X_t_cont)
        
        # Only apply mean and std alignment to continuous features
        X_s_mean = np.mean(X_s_cont, axis=0)
        X_s_std = np.std(X_s_cont, axis=0)
        X_t_mean = np.mean(X_t_cont, axis=0)
        X_t_std = np.std(X_t_cont, axis=0)
        
        # Align mean and std
        X_t_cont_aligned = (X_t_cont - X_t_mean) / (X_t_std + 1e-8) * X_s_std + X_s_mean
        
        align_info = {
            'method': 'mean_std',
            'n_iter': 1
        }
        
        # Reconstruct the full feature matrix
        X_t_aligned = np.copy(X_t)
        X_t_aligned[:, cont_idx] = X_t_cont_aligned
        
    else:
        raise ValueError(f"Unknown alignment method: {method}")
    
    # For linear and mean_std methods, compute final MMD
    if method != 'kpca':  # kpca already has final_mmd
        # Extract aligned continuous features
        X_t_aligned_cont = X_t_aligned[:, cont_idx]
        
        # Compute final MMD after alignment
        final_mmd = compute_mmd(X_s_cont, X_t_aligned_cont, kernel='rbf', gamma=1.0)
        logging.info(f"Final MMD after {method} alignment: {final_mmd:.6f}")
        
        # Analyze MMD reduction percentage
        mmd_reduction = (initial_mmd - final_mmd) / initial_mmd * 100
        logging.info(f"MMD reduction: {mmd_reduction:.2f}%")
    
    # Return aligned features and MMD info
    mmd_info = {
        'method': method,
        'initial_mmd': initial_mmd,
        'final_mmd': final_mmd,
        'reduction': mmd_reduction,
        'align_info': align_info
    }
    
    return X_t_aligned, mmd_info

# 运行基于MMD的域适应实验
def run_mmd_adaptation_experiment(
    X_source,
    y_source,
    X_target,
    y_target,
    model_name='TabPFN-MMD',
    tabpfn_params={'device': 'cuda', 'max_time': 60, 'random_state': 42},
    base_path='./results_mmd',
    optimize_decision_threshold=True,
    mmd_method='linear',
    mmd_params=None
):
    """
    运行带有MMD域适应的TabPFN实验
    
    参数:
    - X_source: 源域特征
    - y_source: 源域标签
    - X_target: 目标域特征
    - y_target: 目标域标签
    - model_name: 模型名称
    - tabpfn_params: TabPFN参数
    - base_path: 结果保存路径
    - optimize_decision_threshold: 是否优化决策阈值
    - mmd_method: MMD对齐方法，可选 'linear', 'kpca', 'mean_std'
    - mmd_params: 传递给MMD对齐方法的参数
    
    返回:
    - 评估指标
    """
    logging.info(f"\n=== {model_name} Model (MMD - {mmd_method}) ===")
    
    if mmd_params is None:
        mmd_params = {}
    
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 数据标准化
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # 分析源域和目标域的特征分布差异
    logging.info("Analyzing domain differences (before alignment)...")
    
    # 计算MMD
    initial_mmd = compute_mmd(X_source_scaled, X_target_scaled, kernel='rbf', gamma=1.0)
    logging.info(f"Initial MMD distance: {initial_mmd:.6f}")
    
    # 多核MMD评估
    multi_kernel_mmd = compute_multiple_kernels_mmd(X_source_scaled, X_target_scaled)
    logging.info(f"Best kernel: {multi_kernel_mmd.get('best_kernel', 'unknown')}, MMD value: {multi_kernel_mmd.get('min_mmd', float('inf')):.6f}")
    
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
    source_metrics = evaluate_metrics(y_source_val, y_source_val_pred, y_source_val_proba[:, 1])
    
    logging.info(f"Source validation - Accuracy: {source_metrics['acc']:.4f}, AUC: {source_metrics['auc']:.4f}, F1: {source_metrics['f1']:.4f}")
    logging.info(f"Source validation - Class 0 Accuracy: {source_metrics['acc_0']:.4f}, Class 1 Accuracy: {source_metrics['acc_1']:.4f}")
    
    # 在目标域上直接评估（未对齐）
    logging.info("\nEvaluating TabPFN directly on target domain (without alignment)...")
    y_target_pred_direct = tabpfn_model.predict(X_target_scaled)
    y_target_proba_direct = tabpfn_model.predict_proba(X_target_scaled)
    
    # 分析预测分布
    unique_labels, counts = np.unique(y_target_pred_direct, return_counts=True)
    logging.info(f"Direct prediction distribution: {dict(zip(unique_labels, counts))}")
    
    # 计算直接预测指标
    direct_metrics = evaluate_metrics(y_target, y_target_pred_direct, y_target_proba_direct[:, 1])
    logging.info(f"Direct prediction - Accuracy: {direct_metrics['acc']:.4f}, AUC: {direct_metrics['auc']:.4f}, F1: {direct_metrics['f1']:.4f}")
    logging.info(f"Direct prediction - Class 0 Accuracy: {direct_metrics['acc_0']:.4f}, Class 1 Accuracy: {direct_metrics['acc_1']:.4f}")
    
    # 使用MMD进行特征对齐
    logging.info(f"\nApplying MMD transformation (method: {mmd_method})...")
    start_time = time.time()
    X_target_aligned, mmd_info = mmd_transform(X_source_scaled, X_target_scaled, method=mmd_method, **mmd_params)
    align_time = time.time() - start_time
    logging.info(f"MMD transformation completed in {align_time:.2f} seconds")
    
    # 分析MMD减少情况
    logging.info(f"MMD reduction: {mmd_info['initial_mmd']:.6f} -> {mmd_info['final_mmd']:.6f} ({mmd_info['reduction']:.2f}%)")
    
    # 对齐后重新计算多核MMD
    multi_kernel_mmd_after = compute_multiple_kernels_mmd(X_source_scaled, X_target_aligned)
    logging.info(f"Best kernel after alignment: {multi_kernel_mmd_after.get('best_kernel', 'unknown')}, MMD value: {multi_kernel_mmd_after.get('min_mmd', float('inf')):.6f}")
    
    # 在目标域上进行评估
    logging.info("\nEvaluating model on target domain (with MMD alignment)...")
    
    # 目标域预测（使用MMD对齐）
    start_time = time.time()
    y_target_pred = tabpfn_model.predict(X_target_aligned)
    y_target_proba = tabpfn_model.predict_proba(X_target_aligned)
    inference_time = time.time() - start_time
    
    # 分析预测分布
    unique_labels, counts = np.unique(y_target_pred, return_counts=True)
    logging.info(f"Prediction distribution after MMD alignment: {dict(zip(unique_labels, counts))}")
    
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
    
    logging.info("\nTarget Domain Evaluation Results (with MMD):")
    logging.info(f"Accuracy: {target_metrics['acc']:.4f}")
    logging.info(f"AUC: {target_metrics['auc']:.4f}")
    logging.info(f"F1: {target_metrics['f1']:.4f}")
    logging.info(f"Class 0 Accuracy: {target_metrics['acc_0']:.4f}")
    logging.info(f"Class 1 Accuracy: {target_metrics['acc_1']:.4f}")
    logging.info(f"Inference Time: {inference_time:.4f} seconds")
    
    # 比较对齐前后的性能
    logging.info("\nPerformance Improvement with MMD Alignment:")
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
             mmd_info=mmd_info)
    logging.info(f"Aligned features saved to: {aligned_features_path}")
    
    # 使用可视化模块进行分析
    vis_path = f"{base_path}/{model_name}_visualizations"
    os.makedirs(vis_path, exist_ok=True)
    
    try:
        # 使用可视化模块的函数进行可视化
        visualize_mmd_adaptation_results(
            X_source_scaled, 
            X_target_scaled, 
            X_target_aligned,
            source_labels=y_source,
            target_labels=y_target,
            output_dir=vis_path,
            feature_names=selected_features,  # 添加特征名称列表参数
            method_name=f"MMD-{mmd_method}"
        )
        logging.info(f"Visualization results saved to: {vis_path}")
    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")
        
        # 如果可视化模块失败，使用内置绘图功能进行简单可视化
        plt.figure(figsize=(15, 5))
        
        # 绘制MMD差异
        plt.subplot(1, 3, 1)
        plt.bar(['Before Alignment', 'After Alignment'], [mmd_info['initial_mmd'], mmd_info['final_mmd']], color=['red', 'green'])
        plt.title('MMD Distance')
        plt.ylabel('MMD Value')
        plt.grid(True, alpha=0.3)
        
        # 绘制预测准确率
        plt.subplot(1, 3, 2)
        methods = ['Direct Prediction', f'MMD Alignment\n({mmd_method})']
        acc_values = [direct_metrics['acc'], target_metrics['acc']]
        colors = ['red', 'green']
        plt.bar(methods, acc_values, color=colors)
        plt.title('Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        
        # 类别预测准确率对比
        plt.subplot(1, 3, 3)
        labels = ['Direct Prediction', f'MMD Alignment\n({mmd_method})']
        class0_accs = [direct_metrics['acc_0'], target_metrics['acc_0']]
        class1_accs = [direct_metrics['acc_1'], target_metrics['acc_1']]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, class0_accs, width, label='Class 0')
        plt.bar(x + width/2, class1_accs, width, label='Class 1')
        
        plt.xlabel('Method')
        plt.ylabel('Accuracy')
        plt.title('Class-wise Accuracy Comparison')
        plt.xticks(x, labels)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{base_path}/{model_name}_analysis.png", dpi=300)
        plt.close()
        
        # 如果使用了阈值优化，添加ROC曲线和阈值点
        if optimize_decision_threshold:
            plt.figure(figsize=(8, 6))
            fpr, tpr, thresholds = roc_curve(y_target, y_target_proba[:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {target_metrics["auc"]:.4f})')
            
            # 标出最佳阈值点
            optimal_idx = np.where(thresholds >= optimal_threshold)[0][-1]
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
        'mmd': {
            'method': mmd_method,
            'initial_mmd': mmd_info['initial_mmd'],
            'final_mmd': mmd_info['final_mmd'],
            'reduction': mmd_info['reduction'],
            'multi_kernel_before': multi_kernel_mmd,
            'multi_kernel_after': multi_kernel_mmd_after
        }
    }

# 添加类条件MMD变换函数
def class_conditional_mmd_transform(X_s, y_s, X_t, yt_pseudo=None, method='linear', cat_idx=None, **kwargs):
    """
    Class-conditional MMD transformation for feature alignment
    
    Parameters:
    - X_s: Source domain features [n_samples_source, n_features]
    - y_s: Source domain labels [n_samples_source]
    - X_t: Target domain features [n_samples_target, n_features]
    - yt_pseudo: Target domain pseudo-labels, if not provided, use source domain model to predict [n_samples_target]
    - method: Alignment method, one of 'linear', 'kpca', 'mean_std'
    - cat_idx: Index of categorical features, if None use TabPFN defaults
    
    Returns:
    - X_t_aligned: Class-conditionally aligned target domain features
    - mmd_info: MMD related information
    """
    if cat_idx is None:
        cat_idx = [0, 2, 3, 4, 12, 13, 16, 17, 18, 19, 22]
    
    # Calculate overall initial MMD
    all_idx = list(range(X_s.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    
    # Continuous features
    X_s_cont = X_s[:, cont_idx]
    X_t_cont = X_t[:, cont_idx]
    
    # Overall MMD before alignment
    overall_initial_mmd = compute_mmd(X_s_cont, X_t_cont, kernel='rbf', gamma=1.0)
    logging.info(f"Overall initial MMD: {overall_initial_mmd:.6f}")
    
    # Create a copy of X_t for temporary alignment
    X_t_temp = np.copy(X_t)
    
    # If target domain doesn't have pseudo-labels, use normal MMD first to align, then predict with source model
    if yt_pseudo is None:
        logging.info("No pseudo-labels provided for target domain. Performing standard MMD alignment first, then predicting pseudo-labels.")
        X_t_temp, _ = mmd_transform(X_s, X_t, method=method, cat_idx=cat_idx, **kwargs)
        
        # Use scikit-learn's nearest neighbors classifier to predict pseudo-labels
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_s, y_s)
        yt_pseudo = knn.predict(X_t_temp)
        
        logging.info(f"Generated pseudo-label distribution: {np.bincount(yt_pseudo)}")
    
    # Initialize aligned target domain feature matrix
    X_t_aligned = np.copy(X_t)
    
    # Get unique classes
    classes = np.unique(y_s)
    n_classes = len(classes)
    
    # Class-specific MMD information
    class_specific_info = {}
    
    # For each class, align separately
    for c in classes:
        # Get source domain samples belonging to class c
        X_s_c = X_s[y_s == c]
        
        # Get target domain samples belonging to class c (based on pseudo-labels)
        X_t_c = X_t[yt_pseudo == c]
        
        if len(X_t_c) == 0:
            logging.warning(f"No samples in target domain with pseudo-label {c}, skipping this class")
            class_specific_info[c] = {
                'initial_mmd': None,
                'final_mmd': None,
                'reduction': None,
                'count': 0
            }
            continue
        
        logging.info(f"Class {c}: Source samples = {len(X_s_c)}, Target samples = {len(X_t_c)}")
        
        # Align using MMD for this class only
        X_t_c_aligned, mmd_c_info = mmd_transform(X_s_c, X_t_c, method=method, cat_idx=cat_idx, **kwargs)
        
        # Put transformed samples back into the aligned target feature matrix
        X_t_aligned[yt_pseudo == c] = X_t_c_aligned
        
        # Save class-specific info
        class_specific_info[c] = {
            'initial_mmd': mmd_c_info['initial_mmd'],
            'final_mmd': mmd_c_info['final_mmd'],
            'reduction': mmd_c_info['reduction'],
            'count': len(X_t_c)
        }
    
    # Calculate overall final MMD
    overall_final_mmd = compute_mmd(X_s_cont, X_t_aligned[:, cont_idx], kernel='rbf', gamma=1.0)
    overall_reduction = (overall_initial_mmd - overall_final_mmd) / overall_initial_mmd * 100
    
    logging.info(f"Overall final MMD after class-conditional alignment: {overall_final_mmd:.6f}")
    logging.info(f"Overall MMD reduction: {overall_reduction:.2f}%")
    
    # Compile MMD information
    mmd_info = {
        'method': method,
        'overall_initial_mmd': overall_initial_mmd,
        'overall_final_mmd': overall_final_mmd,
        'overall_reduction': overall_reduction,
        'class_specific': class_specific_info
    }
    
    return X_t_aligned, mmd_info

# Run class-conditional MMD domain adaptation experiment
def run_class_conditional_mmd_experiment(
    X_source,
    y_source,
    X_target,
    y_target,
    model_name='TabPFN-ClassMMD',
    tabpfn_params={'device': 'cuda', 'max_time': 60, 'random_state': 42},
    base_path='./results_mmd',
    optimize_decision_threshold=True,
    mmd_method='linear',
    mmd_params=None,
    use_target_labels=False,
    target_label_ratio=0.1
):
    """
    Run TabPFN experiment with class-conditional MMD domain adaptation
    
    Parameters:
    - X_source: Source domain features
    - y_source: Source domain labels
    - X_target: Target domain features
    - y_target: Target domain labels
    - model_name: Model name
    - tabpfn_params: TabPFN parameters
    - base_path: Results save path
    - optimize_decision_threshold: Whether to optimize decision threshold
    - mmd_method: MMD alignment method, options: 'linear', 'kpca', 'mean_std'
    - mmd_params: Parameters to pass to the MMD alignment method
    - use_target_labels: Whether to use partial target domain real labels (instead of pseudo-labels)
    - target_label_ratio: Proportion of target domain labels to use
    
    Returns:
    - Evaluation metrics
    """
    logging.info(f"\n=== {model_name} Model (Class-Conditional MMD - {mmd_method}) ===")
    
    if mmd_params is None:
        mmd_params = {}
    
    # Create results directory
    os.makedirs(base_path, exist_ok=True)
    
    # Data standardization
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # Split source domain into train and test sets
    logging.info("Splitting source domain into train and validation sets (80/20 split)...")
    X_source_train, X_source_val, y_source_train, y_source_val = train_test_split(
        X_source_scaled, y_source, test_size=0.2, random_state=42, stratify=y_source
    )
    logging.info(f"Source domain - Training: {X_source_train.shape[0]} samples, Validation: {X_source_val.shape[0]} samples")
    
    # Initialize TabPFN model
    logging.info("Initializing TabPFN model...")
    tabpfn_model = AutoTabPFNClassifier(**tabpfn_params)
    
    # Train TabPFN on source domain training data
    logging.info("Training TabPFN on source domain training data...")
    start_time = time.time()
    tabpfn_model.fit(X_source_train, y_source_train)
    tabpfn_time = time.time() - start_time
    logging.info(f"TabPFN training completed in {tabpfn_time:.2f} seconds")
    
    # Evaluate TabPFN on source domain validation set
    logging.info("\nEvaluating TabPFN on source domain validation set...")
    y_source_val_pred = tabpfn_model.predict(X_source_val)
    y_source_val_proba = tabpfn_model.predict_proba(X_source_val)
    
    # Calculate source domain validation metrics
    source_metrics = evaluate_metrics(y_source_val, y_source_val_pred, y_source_val_proba[:, 1])
    
    logging.info(f"Source validation - Accuracy: {source_metrics['acc']:.4f}, AUC: {source_metrics['auc']:.4f}, F1: {source_metrics['f1']:.4f}")
    logging.info(f"Source validation - Class 0 Accuracy: {source_metrics['acc_0']:.4f}, Class 1 Accuracy: {source_metrics['acc_1']:.4f}")
    
    # Direct evaluation on target domain (without alignment)
    logging.info("\nEvaluating TabPFN directly on target domain (without alignment)...")
    y_target_pred_direct = tabpfn_model.predict(X_target_scaled)
    y_target_proba_direct = tabpfn_model.predict_proba(X_target_scaled)
    
    # Analyze prediction distribution
    unique_labels, counts = np.unique(y_target_pred_direct, return_counts=True)
    logging.info(f"Direct prediction distribution: {dict(zip(unique_labels, counts))}")
    
    # Calculate direct prediction metrics
    direct_metrics = evaluate_metrics(y_target, y_target_pred_direct, y_target_proba_direct[:, 1])
    logging.info(f"Direct prediction - Accuracy: {direct_metrics['acc']:.4f}, AUC: {direct_metrics['auc']:.4f}, F1: {direct_metrics['f1']:.4f}")
    logging.info(f"Direct prediction - Class 0 Accuracy: {direct_metrics['acc_0']:.4f}, Class 1 Accuracy: {direct_metrics['acc_1']:.4f}")
    
    # Prepare target domain labels/pseudo-labels for class-conditional MMD
    yt_pseudo = None
    if use_target_labels:
        # If using partial real labels
        logging.info(f"\nUsing {target_label_ratio:.1%} of target domain true labels for class-conditional MMD alignment...")
        n_labeled = int(len(y_target) * target_label_ratio)
        
        # Stratified sampling to ensure samples from each class
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1-target_label_ratio, random_state=42)
        for labeled_idx, _ in sss.split(X_target_scaled, y_target):
            pass
        
        # Create pseudo-labels (some are real labels, some are predicted with standard MMD)
        yt_pseudo = np.zeros_like(y_target) - 1  # Initialize as -1 for unknown
        yt_pseudo[labeled_idx] = y_target[labeled_idx]  # Fill in known labels
        
        # For unlabeled portion, use standard MMD for prediction
        # First align unlabeled portion with standard MMD
        X_target_unlabeled = X_target_scaled[yt_pseudo == -1]
        X_target_unlabeled_aligned, _ = mmd_transform(X_source_scaled, X_target_unlabeled, method=mmd_method, **mmd_params)
        
        # Predict the unlabeled portion
        yt_pseudo_unlabeled = tabpfn_model.predict(X_target_unlabeled_aligned)
        yt_pseudo[yt_pseudo == -1] = yt_pseudo_unlabeled
        
        logging.info(f"Partial true labels + partial pseudo-labels distribution: {np.bincount(yt_pseudo)}")
    
    # Apply class-conditional MMD for feature alignment
    logging.info(f"\nApplying class-conditional MMD transformation (method: {mmd_method})...")
    start_time = time.time()
    X_target_aligned, mmd_info = class_conditional_mmd_transform(
        X_source_scaled, y_source, X_target_scaled, yt_pseudo, 
        method=mmd_method, **mmd_params
    )
    align_time = time.time() - start_time
    logging.info(f"Class-conditional MMD transformation completed in {align_time:.2f} seconds")
    
    # Analyze overall MMD reduction
    logging.info(f"Overall MMD reduction: {mmd_info['overall_initial_mmd']:.6f} -> {mmd_info['overall_final_mmd']:.6f} ({mmd_info['overall_reduction']:.2f}%)")
    
    # Evaluate on target domain
    logging.info("\nEvaluating model on target domain (with class-conditional MMD alignment)...")
    
    # Target domain prediction (using class-conditional MMD alignment)
    start_time = time.time()
    y_target_pred = tabpfn_model.predict(X_target_aligned)
    y_target_proba = tabpfn_model.predict_proba(X_target_aligned)
    inference_time = time.time() - start_time
    
    # Analyze prediction distribution
    unique_labels, counts = np.unique(y_target_pred, return_counts=True)
    logging.info(f"Prediction distribution after class-conditional MMD alignment: {dict(zip(unique_labels, counts))}")
    
    # Calculate target domain metrics
    target_metrics = evaluate_metrics(y_target, y_target_pred, y_target_proba[:, 1])
    
    # Optimize decision threshold (optional)
    if optimize_decision_threshold:
        logging.info("\nOptimizing decision threshold using Youden index...")
        optimal_threshold, optimal_metrics = optimize_threshold(y_target, y_target_proba[:, 1])
        
        logging.info(f"Optimal threshold: {optimal_threshold:.4f} (default: 0.5)")
        logging.info(f"Metrics with optimized threshold:")
        logging.info(f"  Accuracy: {optimal_metrics['acc']:.4f} (original: {target_metrics['acc']:.4f})")
        logging.info(f"  F1 Score: {optimal_metrics['f1']:.4f} (original: {target_metrics['f1']:.4f})")
        logging.info(f"  Class 0 Accuracy: {optimal_metrics['acc_0']:.4f} (original: {target_metrics['acc_0']:.4f})")
        logging.info(f"  Class 1 Accuracy: {optimal_metrics['acc_1']:.4f} (original: {target_metrics['acc_1']:.4f})")
        
        # Update metrics
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
    
    # Print results
    logging.info("\nSource Domain Validation Results:")
    logging.info(f"Accuracy: {source_metrics['acc']:.4f}")
    logging.info(f"AUC: {source_metrics['auc']:.4f}")
    logging.info(f"F1: {source_metrics['f1']:.4f}")
    
    logging.info("\nTarget Domain Evaluation Results (with Class-Conditional MMD):")
    logging.info(f"Accuracy: {target_metrics['acc']:.4f}")
    logging.info(f"AUC: {target_metrics['auc']:.4f}")
    logging.info(f"F1: {target_metrics['f1']:.4f}")
    logging.info(f"Class 0 Accuracy: {target_metrics['acc_0']:.4f}")
    logging.info(f"Class 1 Accuracy: {target_metrics['acc_1']:.4f}")
    
    # Compare performance before and after alignment
    logging.info("\nPerformance Improvement with Class-Conditional MMD Alignment:")
    logging.info(f"Accuracy: {direct_metrics['acc']:.4f} -> {target_metrics['acc']:.4f} ({target_metrics['acc']-direct_metrics['acc']:.4f})")
    logging.info(f"AUC: {direct_metrics['auc']:.4f} -> {target_metrics['auc']:.4f} ({target_metrics['auc']-direct_metrics['auc']:.4f})")
    logging.info(f"F1: {direct_metrics['f1']:.4f} -> {target_metrics['f1']:.4f} ({target_metrics['f1']-direct_metrics['f1']:.4f})")
    logging.info(f"Class 0 Accuracy: {direct_metrics['acc_0']:.4f} -> {target_metrics['acc_0']:.4f} ({target_metrics['acc_0']-direct_metrics['acc_0']:.4f})")
    logging.info(f"Class 1 Accuracy: {direct_metrics['acc_1']:.4f} -> {target_metrics['acc_1']:.4f} ({target_metrics['acc_1']-direct_metrics['acc_1']:.4f})")
    
    # Save aligned features
    aligned_features_path = f"{base_path}/{model_name}_aligned_features.npz"
    np.savez(aligned_features_path, 
             X_source=X_source_scaled, 
             X_target=X_target_scaled,
             X_target_aligned=X_target_aligned,
             yt_pseudo=yt_pseudo,
             mmd_info=mmd_info)
    logging.info(f"Aligned features saved to: {aligned_features_path}")
    
    # 使用可视化模块进行分析
    vis_path = f"{base_path}/{model_name}_visualizations"
    os.makedirs(vis_path, exist_ok=True)
    
    try:
        # 使用可视化模块的函数进行可视化
        visualize_mmd_adaptation_results(
            X_source_scaled, 
            X_target_scaled, 
            X_target_aligned,
            source_labels=y_source,
            target_labels=y_target,
            output_dir=vis_path,
            feature_names=selected_features,  # 添加特征名称列表参数
            method_name=f"ClassMMD-{mmd_method}"
        )
        logging.info(f"Visualization results saved to: {vis_path}")
    except Exception as e:
        logging.error(f"Visualization failed: {str(e)}")
        
        # 使用内置绘图功能进行简单可视化
        plt.figure(figsize=(15, 5))
        
        # Plot overall metrics comparison
        plt.subplot(1, 3, 1)
        metrics_labels = ['Accuracy', 'AUC', 'F1']
        metrics_values_direct = [direct_metrics['acc'], direct_metrics['auc'], direct_metrics['f1']]
        metrics_values_mmd = [target_metrics['acc'], target_metrics['auc'], target_metrics['f1']]
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        plt.bar(x - width/2, metrics_values_direct, width, label='Direct Prediction')
        plt.bar(x + width/2, metrics_values_mmd, width, label=f'Class-Conditional MMD ({mmd_method})')
        
        plt.ylabel('Score')
        plt.title('Overall Performance Metrics')
        plt.xticks(x, metrics_labels)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Plot per-class accuracy
        plt.subplot(1, 3, 2)
        class_metrics = ['Class 0 Accuracy', 'Class 1 Accuracy']
        class_values_direct = [direct_metrics['acc_0'], direct_metrics['acc_1']]
        class_values_mmd = [target_metrics['acc_0'], target_metrics['acc_1']]
        
        x = np.arange(len(class_metrics))
        
        plt.bar(x - width/2, class_values_direct, width, label='Direct Prediction')
        plt.bar(x + width/2, class_values_mmd, width, label=f'Class-Conditional MMD ({mmd_method})')
        
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(x, class_metrics)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # Plot prediction distribution
        plt.subplot(1, 3, 3)
        
        labels = ['Direct Prediction', f'Class-Conditional MMD\n({mmd_method})']
        pred_dist_direct = np.bincount(y_target_pred_direct)
        pred_dist_mmd = np.bincount(y_target_pred)
        true_dist = np.bincount(y_target)
        
        # Ensure all bar charts have the same number of categories
        max_classes = max(len(pred_dist_direct), len(pred_dist_mmd), len(true_dist))
        if len(pred_dist_direct) < max_classes:
            pred_dist_direct = np.pad(pred_dist_direct, (0, max_classes - len(pred_dist_direct)))
        if len(pred_dist_mmd) < max_classes:
            pred_dist_mmd = np.pad(pred_dist_mmd, (0, max_classes - len(pred_dist_mmd)))
        if len(true_dist) < max_classes:
            true_dist = np.pad(true_dist, (0, max_classes - len(true_dist)))
        
        x = np.arange(max_classes)
        width = 0.25
        
        plt.bar(x - width, pred_dist_direct, width, label='Direct Prediction')
        plt.bar(x, pred_dist_mmd, width, label=f'Class-Conditional MMD ({mmd_method})')
        plt.bar(x + width, true_dist, width, label='True Distribution')
        
        plt.xlabel('Class')
        plt.ylabel('Sample Count')
        plt.title('Prediction Distribution Comparison')
        plt.xticks(x)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{base_path}/{model_name}_analysis.png", dpi=300)
        plt.close()
        
        # If decision threshold optimization is used, add ROC curve
        if optimize_decision_threshold:
            plt.figure(figsize=(8, 6))
            fpr, tpr, thresholds = roc_curve(y_target, y_target_proba[:, 1])
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {target_metrics["auc"]:.4f})')
            
            # Mark the optimal threshold point
            optimal_idx = np.where(thresholds >= optimal_threshold)[0][-1]
            plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', 
                        label=f'Optimal Threshold = {optimal_threshold:.4f}')
            
            # Mark the point corresponding to default threshold 0.5
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
    
    return {
        'source': source_metrics,
        'target': target_metrics,
        'direct': direct_metrics,
        'times': {
            'tabpfn': tabpfn_time,
            'align': align_time,
            'inference': inference_time
        },
        'mmd': {
            'method': mmd_method,
            'use_target_labels': use_target_labels,
            'target_label_ratio': target_label_ratio if use_target_labels else None,
            'overall_initial_mmd': mmd_info['overall_initial_mmd'],
            'overall_final_mmd': mmd_info['overall_final_mmd'],
            'overall_reduction': mmd_info['overall_reduction'],
            'class_specific': mmd_info['class_specific']
        }
    }

# 主函数
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    
    # 命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description="TabPFN MMD域适应实验")
    parser.add_argument("--disable-visualization", action="store_true", help="禁用可视化功能")
    parser.add_argument("--output-dir", type=str, default="./results_mmd", help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备 (cuda 或 cpu)")
    args = parser.parse_args()
    
    # 如果命令行禁用了可视化
    if args.disable_visualization:
        has_visualization = False  # 这里直接修改本地变量
    
    # 指定设备
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    logging.info(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载所有数据集
    logging.info("\nLoading datasets...")
    logging.info("1. Loading AI4healthcare.xlsx (A)...")
    df_ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    
    logging.info("2. Loading HenanCancerHospital_features63_58.xlsx (B)...")
    df_henan = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    logging.info("3. Loading GuangzhouMedicalHospital_features23_no_nan.xlsx (C)...")
    df_guangzhou = pd.read_excel("data/GuangzhouMedicalHospital_features23_no_nan.xlsx")

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
    logging.info(f"Selected feature count: {len(selected_features)}")
    logging.info(f"Selected feature list: {selected_features}")

    # 检查每个数据集中是否有所有选定的特征
    for dataset_name, dataset in [
        ("AI4health", df_ai4health), 
        ("Henan", df_henan), 
        ("Guangzhou", df_guangzhou)
    ]:
        missing_features = [f for f in selected_features if f not in dataset.columns]
        if missing_features:
            logging.warning(f"Warning: {dataset_name} is missing the following features: {missing_features}")
        else:
            logging.info(f"{dataset_name} contains all selected features")

    # 使用共同特征准备数据
    X_ai4health = df_ai4health[selected_features].copy()
    y_ai4health = df_ai4health["Label"].copy()
    
    X_henan = df_henan[selected_features].copy()
    y_henan = df_henan["Label"].copy()
    
    X_guangzhou = df_guangzhou[selected_features].copy()
    y_guangzhou = df_guangzhou["Label"].copy()

    # 创建结果目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 运行MMD域适应实验
    logging.info("\n\n=== Running TabPFN with MMD-based Domain Adaptation ===")
    
    # 定义三种MMD方法
    mmd_methods = ['linear', 'kpca', 'mean_std']
    
    # 为每种方法设置参数
    mmd_params = {
        'linear': {
            'gamma': 1.0,
            'lr': 0.01,
            'n_epochs': 100,
            'batch_size': 64,
            'lambda_reg': 0.01
        },
        'kpca': {
            'kernel': 'rbf',
            'gamma': 1.0,
            'n_components': None
        },
        'mean_std': {}
    }
    
    # 定义域适应配置 (从A到B/C)
    mmd_configs = [
        {
            'name': 'A_to_B',
            'source_name': 'A_AI4health',
            'target_name': 'B_Henan',
            'X_source': X_ai4health,
            'y_source': y_ai4health,
            'X_target': X_henan,
            'y_target': y_henan
        },
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
    
    # TabPFN参数设置
    tabpfn_params = {
        'device': device,
        'max_time': 60,
        'random_state': 42
    }
    
    # 运行实验 - 标准MMD
    for config in mmd_configs:
        for method in mmd_methods:
            logging.info(f"\n\n{'='*50}")
            logging.info(f"Domain Adaptation: {config['source_name']} → {config['target_name']} (MMD Method: {method})")
            logging.info(f"{'='*50}")
            
            # 使用特定MMD方法和参数运行实验
            metrics = run_mmd_adaptation_experiment(
                X_source=config['X_source'],
                y_source=config['y_source'],
                X_target=config['X_target'],
                y_target=config['y_target'],
                model_name=f"TabPFN-MMD-{method}_{config['name']}",
                base_path=args.output_dir,
                tabpfn_params=tabpfn_params,
                mmd_method=method,
                mmd_params=mmd_params[method]
            )
            
            # 保存结果
            result = {
                'name': f"TabPFN-MMD-{method}_{config['name']}",
                'method': f"MMD-{method}",
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
                'initial_mmd': metrics['mmd']['initial_mmd'],
                'final_mmd': metrics['mmd']['final_mmd'],
                'mmd_reduction': metrics['mmd']['reduction']
            }
            all_results.append(result)
    
    # 运行实验 - 类条件MMD (仅使用性能最好的方法)
    # 找出每个域适应任务在标准MMD中表现最好的方法
    best_methods = {}
    for config in mmd_configs:
        relevant_results = [r for r in all_results 
                           if r['source'] == config['source_name'] and r['target'] == config['target_name']]
        if relevant_results:
            best_result = max(relevant_results, key=lambda x: x['target_acc'])
            best_method = best_result['method'].split('-')[1]
            best_methods[config['name']] = best_method
    
    # 使用最好的方法运行类条件MMD实验    
    for config in mmd_configs:
        if config['name'] in best_methods:
            best_method = best_methods[config['name']]
            logging.info(f"\n\n{'='*50}")
            logging.info(f"Class Conditional Domain Adaptation: {config['source_name']} → {config['target_name']} (Best MMD Method: {best_method})")
            logging.info(f"{'='*50}")
            
            # 使用伪标签运行类条件实验
            class_metrics = run_class_conditional_mmd_experiment(
                X_source=config['X_source'],
                y_source=config['y_source'],
                X_target=config['X_target'],
                y_target=config['y_target'],
                model_name=f"TabPFN-ClassMMD-{best_method}_{config['name']}",
                base_path=args.output_dir,
                tabpfn_params=tabpfn_params,
                mmd_method=best_method,
                mmd_params=mmd_params[best_method],
                use_target_labels=False
            )
            
            # 使用10%真实标签运行类条件实验
            class_metrics_with_labels = run_class_conditional_mmd_experiment(
                X_source=config['X_source'],
                y_source=config['y_source'],
                X_target=config['X_target'],
                y_target=config['y_target'],
                model_name=f"TabPFN-ClassMMD-WithLabels-{best_method}_{config['name']}",
                base_path=args.output_dir,
                tabpfn_params=tabpfn_params,
                mmd_method=best_method,
                mmd_params=mmd_params[best_method],
                use_target_labels=True,
                target_label_ratio=0.1
            )
            
            # 保存类条件结果
            result = {
                'method': f"ClassMMD-{best_method}",
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
                'direct_f1': class_metrics['direct']['f1'],
                'initial_mmd': class_metrics['mmd']['overall_initial_mmd'],
                'final_mmd': class_metrics['mmd']['overall_final_mmd'],
                'mmd_reduction': class_metrics['mmd']['overall_reduction']
            }
            all_results.append(result)
    
    # 创建结果表格
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{args.output_dir}/all_results.csv', index=False)
    logging.info(f"所有结果已保存至 {args.output_dir}/all_results.csv")
    
    # 可视化比较
    logging.info("\nGenerating comparison visualizations for different MMD methods...")
    
    # 创建结果汇总图表
    plt.figure(figsize=(12, 10))
    
    # 仅使用基本MMD方法（非类条件的）
    domain_list = []
    method_list = []
    acc_list = []
    auc_list = []
    f1_list = []
    pseudo_acc_0_list = []
    pseudo_acc_1_list = []
    
    for domain in ['A_to_B', 'A_to_C']:
        domain_results = []
        for result in all_results:
            if 'name' in result and result['name'].endswith(domain) and 'Class' not in result['name']:
                domain_results.append({
                    'method': result['name'].split('_')[1],
                    'acc': result['target_acc'],
                    'auc': result['target_auc'],
                    'f1': result['target_f1'],
                    'pseudo_acc_0': result['target_acc_0'],
                    'pseudo_acc_1': result['target_acc_1']
                })
        
        # 添加到整体列表
        for dr in domain_results:
            domain_list.append(domain)
            method_list.append(dr['method'])
            acc_list.append(dr['acc'])
            auc_list.append(dr['auc'])
            f1_list.append(dr['f1'])
            pseudo_acc_0_list.append(dr['pseudo_acc_0'])
            pseudo_acc_1_list.append(dr['pseudo_acc_1'])
    
    # 创建对比数据框
    comparison_df = pd.DataFrame({
        'Domain': domain_list,
        'Method': method_list,
        'Accuracy': acc_list,
        'AUC': auc_list,
        'F1': f1_list,
        'Class 0 Acc': pseudo_acc_0_list,
        'Class 1 Acc': pseudo_acc_1_list
    })
    
    # 找出是否有类条件方法
    has_class_conditional = any('ClassMMD' in r['name'] for r in all_results if 'name' in r)
    
    if has_class_conditional:
        # 比较类条件与非类条件
        plt.subplot(2, 2, 1)
        
        # 获取最佳基本方法和对应的类条件方法
        best_basic_method = comparison_df.groupby('Method')['Accuracy'].mean().idxmax()
        
        class_results = []
        for domain in ['A_to_B', 'A_to_C']:
            # 基本方法结果
            basic_results = next(r for r in all_results if r['name'] == f'TabPFN-MMD-{best_basic_method}_{domain}')
            class_results.append({
                'domain': domain,
                'method': 'Basic MMD',
                'acc': basic_results['target_acc']
            })
            
            # 类条件方法结果
            if any(r['name'] == f'TabPFN-ClassMMD-{best_basic_method}_{domain}' for r in all_results):
                class_method = next(r for r in all_results if r['name'] == f'TabPFN-ClassMMD-{best_basic_method}_{domain}')
                class_results.append({
                    'domain': domain,
                    'method': 'Class MMD',
                    'acc': class_method['pseudo_acc']
                })
            
            # 添加类条件+标签方法
            if any(r['name'] == f'TabPFN-ClassMMD-WithLabels-{best_basic_method}_{domain}' for r in all_results):
                labels_method = next(r for r in all_results if r['name'] == f'TabPFN-ClassMMD-WithLabels-{best_basic_method}_{domain}')
                class_results.append({
                    'domain': domain,
                    'method': 'Class MMD+Labels',
                    'acc': labels_method['withlabels_acc']
                })
        
        # 为每个域创建分组柱状图
        df_class = pd.DataFrame(class_results)
        domains = df_class['domain'].unique()
        methods = df_class['method'].unique()
        
        x = np.arange(len(domains))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            domain_values = []
            for domain in domains:
                domain_results = df_class[df_class['domain'] == domain]
                if method in domain_results['method'].values:
                    domain_values.append(domain_results[domain_results['method'] == method]['acc'].values[0])
                else:
                    domain_values.append(0)
            
            plt.bar(x + width * (i - len(methods)/2 + 0.5), domain_values, width, label=method)
        
        plt.xlabel('Domain')
        plt.ylabel('Accuracy')
        plt.title('Comparison of Basic vs Class-Conditional MMD')
        plt.xticks(x, domains)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 类条件MMD方法的类别精度比较
        plt.subplot(2, 2, 2)
        
        class0_values = []
        class1_values = []
        domain_names = []
        
        for domain in ['A_to_B', 'A_to_C']:
            domain_names.append(domain)
            
            # 基本MMD
            domain_results = comparison_df[comparison_df['Method'] == best_basic_method]
            domain_results = domain_results[domain_results['Domain'] == domain]
            class0_values.append(domain_results['Class 0 Acc'].values[0])
            class1_values.append(domain_results['Class 1 Acc'].values[0])
            
            # 类条件MMD
            if any(r['name'] == f'TabPFN-ClassMMD-{best_basic_method}_{domain}' for r in all_results):
                class_result = next(r for r in all_results if r['name'] == f'TabPFN-ClassMMD-{best_basic_method}_{domain}')
                domain_names.append(f"{domain} (Class)")
                class0_values.append(class_result['pseudo_acc_0'])
                class1_values.append(class_result['pseudo_acc_1'])
            
            # 类条件MMD+标签
            if any(r['name'] == f'TabPFN-ClassMMD-WithLabels-{best_basic_method}_{domain}' for r in all_results):
                labels_result = next(r for r in all_results if r['name'] == f'TabPFN-ClassMMD-WithLabels-{best_basic_method}_{domain}')
                domain_names.append(f"{domain} (Class+Labels)")
                class0_values.append(labels_result['withlabels_acc_0'])
                class1_values.append(labels_result['withlabels_acc_1'])
        
        x = np.arange(len(domain_names))
        width = 0.35
        
        plt.bar(x - width/2, class0_values, width, label='Class 0 Accuracy')
        plt.bar(x + width/2, class1_values, width, label='Class 1 Accuracy')
        
        plt.xlabel('Domain and Method')
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy Comparison')
        plt.xticks(x, domain_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 绘制不同MMD方法的主要指标比较
    # 修复subplot参数数量错误
    if has_class_conditional:
        plt.subplot(2, 2, 3)
    else:
        plt.subplot(2, 1, 1)
    
    # 分域绘制
    for domain in ['A_to_B', 'A_to_C']:
        domain_data = comparison_df[comparison_df['Domain'] == domain]
        plt.plot(domain_data['Method'], domain_data['Accuracy'], 'o-', label=f'{domain} Accuracy')
        plt.plot(domain_data['Method'], domain_data['AUC'], 's--', label=f'{domain} AUC')
    
    plt.xlabel('MMD Method')
    plt.ylabel('Score')
    plt.title('MMD Method Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 平均性能比较
    # 修复subplot参数数量错误
    if has_class_conditional:
        plt.subplot(2, 2, 4)
    else:
        plt.subplot(2, 1, 2)
    
    avg_by_method = comparison_df.groupby('Method')[['Accuracy', 'AUC', 'F1', 'Class 0 Acc', 'Class 1 Acc']].mean()
    
    avg_by_method.plot(kind='bar', rot=0)
    plt.title('Average Performance Across Domains')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/mmd_methods_comparison.png', dpi=300)
    plt.close()
    
    # 保存方法比较的表格
    comparison_df.to_csv(f'{args.output_dir}/mmd_methods_comparison.csv', index=False)
    
    # 输出最终结果
    logging.info("\n=== Final Results ===")
    logging.info(f"Best MMD method (by accuracy): {comparison_df.groupby('Method')['Accuracy'].mean().idxmax()}")
    
    avg_results = comparison_df.groupby('Method')[['Accuracy', 'AUC', 'F1']].mean()
    logging.info("\nAverage performance across domains:")
    logging.info(avg_results)
    
    logging.info(f"\nResults saved to {args.output_dir}/")
    logging.info("Done!")