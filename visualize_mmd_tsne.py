#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMD域适应可视化分析工具
包含t-SNE可视化、特征直方图以及域差异统计指标计算
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import sys
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy, wasserstein_distance
from scipy.spatial.distance import cdist
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 配置matplotlib，增加最大图形警告阈值，并启用自动关闭
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 50  # 增加警告阈值
plt.rcParams['figure.figsize'] = (10, 6)  # 设置默认图形大小
plt.rcParams['figure.dpi'] = 100  # 设置默认DPI
plt.ion()  # 启用交互模式，有助于自动回收图形资源

# 设置日志
def setup_logger():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    stdout_handler = logging.StreamHandler(sys.stdout)
    stderr_handler = logging.StreamHandler(sys.stderr)
    
    stdout_handler.setLevel(logging.INFO)
    stderr_handler.setLevel(logging.WARNING)
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stdout_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)
    
    logging.root.addHandler(stdout_handler)
    logging.root.addHandler(stderr_handler)
    logging.root.setLevel(logging.INFO)

# 辅助函数：安全关闭matplotlib图形
def close_figures(fig=None):
    """
    安全关闭matplotlib图形
    
    参数:
    - fig: 特定图形对象。如果为None，关闭所有图形
    """
    try:
        if fig is None:
            plt.close('all')  # 关闭所有图形
            logging.info("已关闭所有图形")
        else:
            plt.close(fig)  # 关闭指定图形
            logging.info(f"已关闭特定图形")
    except Exception as e:
        logging.warning(f"关闭图形时出错: {str(e)}")
        # 尝试强制关闭所有图形
        try:
            plt.close('all')
        except:
            pass

# 基础MMD计算函数
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

# 多核MMD计算
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

# 分布差异度量计算函数
def calculate_kl_divergence(X_source, X_target, bins=20, epsilon=1e-10):
    """计算KL散度"""
    n_features = X_source.shape[1]
    kl_per_feature = {}
    
    for i in range(n_features):
        x_s = X_source[:, i]
        x_t = X_target[:, i]
        
        min_val = min(np.min(x_s), np.min(x_t))
        max_val = max(np.max(x_s), np.max(x_t))
        bin_range = (min_val, max_val)
        
        hist_s, bin_edges = np.histogram(x_s, bins=bins, range=bin_range, density=True)
        hist_t, _ = np.histogram(x_t, bins=bins, range=bin_range, density=True)
        
        hist_s = hist_s + epsilon
        hist_t = hist_t + epsilon
        
        hist_s = hist_s / np.sum(hist_s)
        hist_t = hist_t / np.sum(hist_t)
        
        kl_s_t = entropy(hist_s, hist_t)
        kl_t_s = entropy(hist_t, hist_s)
        
        kl_per_feature[f'feature_{i}'] = (kl_s_t + kl_t_s) / 2
    
    kl_div = np.mean(list(kl_per_feature.values()))
    return kl_div, kl_per_feature

def calculate_wasserstein_distances(X_source, X_target):
    """计算Wasserstein距离"""
    n_features = X_source.shape[1]
    wasserstein_per_feature = {}
    
    for i in range(n_features):
        x_s = X_source[:, i]
        x_t = X_target[:, i]
        
        w_dist = wasserstein_distance(x_s, x_t)
        wasserstein_per_feature[f'feature_{i}'] = w_dist
    
    avg_wasserstein = np.mean(list(wasserstein_per_feature.values()))
    return avg_wasserstein, wasserstein_per_feature

def compute_mmd_kernel(X, Y, gamma=1.0):
    """计算MMD (使用RBF核)"""
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    K_xx = rbf_kernel(X, X, gamma=gamma)
    K_yy = rbf_kernel(Y, Y, gamma=gamma)
    K_xy = rbf_kernel(X, Y, gamma=gamma)
    
    mmd_squared = (np.sum(K_xx) - np.trace(K_xx)) / (n_x * (n_x - 1))
    mmd_squared += (np.sum(K_yy) - np.trace(K_yy)) / (n_y * (n_y - 1))
    mmd_squared -= 2 * np.mean(K_xy)
    
    return np.sqrt(max(mmd_squared, 0))

def compute_domain_discrepancy(X_source, X_target):
    """计算域间差异指标"""
    # 1. 平均距离
    mean_dist = np.mean(cdist(X_source, X_target))
    
    # 2. 均值差异
    mean_diff = np.linalg.norm(np.mean(X_source, axis=0) - np.mean(X_target, axis=0))
    
    # 3. 协方差矩阵距离
    cov_source = np.cov(X_source, rowvar=False)
    cov_target = np.cov(X_target, rowvar=False)
    cov_diff = np.linalg.norm(cov_source - cov_target, 'fro')
    
    # 4. 核均值差异
    X_s_mean = np.mean(X_source, axis=0, keepdims=True)
    X_t_mean = np.mean(X_target, axis=0, keepdims=True)
    kernel_mean_diff = np.exp(-0.5 * np.sum((X_s_mean - X_t_mean)**2))
    
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

def histograms_stats_table(X_source, X_target, X_target_aligned=None, feature_names=None, save_path=None, method_name="MMD"):
    """生成特征分布统计表，用于可视化域适应前后的特征分布变化
    
    参数:
    - X_source: 源域特征矩阵
    - X_target: 目标域特征矩阵
    - X_target_aligned: 对齐后的目标域特征矩阵（如果为None，则仅生成原始差异统计）
    - feature_names: 特征名称列表
    - save_path: 保存路径
    - method_name: 域适应方法名称，用于标题显示
    
    返回:
    - 保存的图像路径
    """
    # 先关闭所有已存在的图形
    close_figures()
    
    try:
        # 为空时生成默认特征名，修改为不带空格的格式
        if feature_names is None:
            feature_names = [f'Feature{i+1}' for i in range(X_source.shape[1])]
        
        # 确保特征名列表长度与数据维度匹配
        if len(feature_names) != X_source.shape[1]:
            logging.warning(f"警告: 特征名称数量({len(feature_names)})与特征维度({X_source.shape[1]})不匹配")
            # 截断或扩展特征名列表以匹配特征维度
            if len(feature_names) > X_source.shape[1]:
                feature_names = feature_names[:X_source.shape[1]]
                logging.warning(f"已截断特征名称列表至{len(feature_names)}个")
            else:
                # 扩展特征名列表，确保使用正确的格式
                feature_names = feature_names + [f'Feature{i+1+len(feature_names)}' for i in range(X_source.shape[1] - len(feature_names))]
                logging.warning(f"已扩展特征名称列表至{len(feature_names)}个")
        
        # 定义阈值
        KL_SEVERE_THRESHOLD = 0.5
        KL_MODERATE_THRESHOLD = 0.2
        WASS_SEVERE_THRESHOLD = 0.5
        WASS_MODERATE_THRESHOLD = 0.2
        
        # 定义不同严重程度的颜色
        SEVERE_COLOR = '#ff6b6b'  # Red
        MODERATE_COLOR = '#feca57'  # Yellow/Orange
        LOW_COLOR = '#1dd1a1'  # Green
        
        # 计算KL散度和Wasserstein距离
        _, kl_div_before = calculate_kl_divergence(X_source, X_target)
        _, wasserstein_before = calculate_wasserstein_distances(X_source, X_target)
        
        if X_target_aligned is not None:
            _, kl_div_after = calculate_kl_divergence(X_source, X_target_aligned)
            _, wasserstein_after = calculate_wasserstein_distances(X_source, X_target_aligned)
        
        # 创建综合的特征统计表
        stats_fig = plt.figure(figsize=(16, 8))
        ax_table = stats_fig.add_subplot(111)
        ax_table.axis('off')
        
        table_data = []
        
        if X_target_aligned is not None:
            table_columns = ['Feature', 'KL Before', 'KL After', 'KL Imp. %', 
                             'Wass. Before', 'Wass. After', 'Wass. Imp. %', 
                             'Initial Shift', 'Aligned Shift']
            
            for i in range(X_source.shape[1]):
                feature_key = f'feature_{i}'
                feature_name = feature_names[i]
                
                kl_before = kl_div_before[feature_key]
                kl_after = kl_div_after[feature_key]
                wass_before = wasserstein_before[feature_key]
                wass_after = wasserstein_after[feature_key]
                
                kl_imp = (kl_before - kl_after) / kl_before * 100 if kl_before > 0 else 0
                wass_imp = (wass_before - wass_after) / wass_before * 100 if wass_before > 0 else 0
                
                # 根据阈值确定初始严重程度
                if kl_before > KL_SEVERE_THRESHOLD or wass_before > WASS_SEVERE_THRESHOLD:
                    initial_severity = "HIGH"
                elif kl_before > KL_MODERATE_THRESHOLD or wass_before > WASS_MODERATE_THRESHOLD:
                    initial_severity = "MEDIUM"
                else:
                    initial_severity = "LOW"
                
                # 根据对齐后的指标确定对齐后的严重程度
                if kl_after > KL_SEVERE_THRESHOLD or wass_after > WASS_SEVERE_THRESHOLD:
                    aligned_severity = "HIGH"
                elif kl_after > KL_MODERATE_THRESHOLD or wass_after > WASS_MODERATE_THRESHOLD:
                    aligned_severity = "MEDIUM"
                else:
                    aligned_severity = "LOW"
                
                # 添加特征统计行
                table_data.append([
                    feature_name,
                    f"{kl_before:.4f}",
                    f"{kl_after:.4f}",
                    f"{kl_imp:.1f}%",
                    f"{wass_before:.4f}",
                    f"{wass_after:.4f}",
                    f"{wass_imp:.1f}%",
                    initial_severity,
                    aligned_severity
                ])
        else:
            # 如果没有对齐后的数据，则只显示原始差异
            table_columns = ['Feature', 'KL Divergence', 'Wasserstein Dist.', 'Shift Severity']
            
            for i in range(X_source.shape[1]):
                feature_key = f'feature_{i}'
                feature_name = feature_names[i]
                
                kl_val = kl_div_before[feature_key]
                wass_val = wasserstein_before[feature_key]
                
                # 根据阈值确定严重程度
                if kl_val > KL_SEVERE_THRESHOLD or wass_val > WASS_SEVERE_THRESHOLD:
                    severity = "HIGH"
                elif kl_val > KL_MODERATE_THRESHOLD or wass_val > WASS_MODERATE_THRESHOLD:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"
                
                # 添加特征统计行
                table_data.append([
                    feature_name,
                    f"{kl_val:.4f}",
                    f"{wass_val:.4f}",
                    severity
                ])
        
        # 创建并设置表格样式
        colWidths = [0.18] + [0.09] * (len(table_columns) - 1)
        table = ax_table.table(
            cellText=table_data,
            colLabels=table_columns,
            loc='center',
            cellLoc='center',
            colWidths=colWidths
        )
        
        # 记录表格中包含的特征
        logging.info(f"特征统计表包含了{len(table_data)}个特征：前5个为 {[row[0] for row in table_data[:5]]}...")
        
        # 改进表格格式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)  # 调整表格比例
        
        # 调整标题行样式
        for i, col_label in enumerate(table_columns):
            table[0, i].set_facecolor('#4a69bd')  # 标题背景色
            table[0, i].set_text_props(color='white', fontweight='bold')
        
        # 根据改进和严重程度设置颜色
        for i, row in enumerate(table_data):
            # 设置特征名列的样式
            table[(i+1, 0)].set_text_props(ha='left', fontweight='bold')
            table[(i+1, 0)].set_facecolor('#f7f1e3')  # 特征名的浅色背景
            
            if X_target_aligned is not None:
                kl_imp = float(row[3].strip('%'))
                wass_imp = float(row[6].strip('%'))
                initial_severity = row[7]
                aligned_severity = row[8]
                
                kl_before_val = float(row[1])
                kl_after_val = float(row[2])
                wass_before_val = float(row[4])
                wass_after_val = float(row[5])
                
                # 设置初始严重程度单元格颜色
                if initial_severity == "HIGH":
                    table[(i+1, 7)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 7)].set_text_props(weight='bold', color='white')
                elif initial_severity == "MEDIUM":
                    table[(i+1, 7)].set_facecolor(MODERATE_COLOR)
                    table[(i+1, 7)].set_text_props(weight='bold')
                else:
                    table[(i+1, 7)].set_facecolor(LOW_COLOR)
                    table[(i+1, 7)].set_text_props(weight='bold', color='white')
                
                # 设置对齐后严重程度单元格颜色
                if aligned_severity == "HIGH":
                    table[(i+1, 8)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 8)].set_text_props(weight='bold', color='white')
                elif aligned_severity == "MEDIUM":
                    table[(i+1, 8)].set_facecolor(MODERATE_COLOR)
                    table[(i+1, 8)].set_text_props(weight='bold')
                else:
                    table[(i+1, 8)].set_facecolor(LOW_COLOR)
                    table[(i+1, 8)].set_text_props(weight='bold', color='white')
                
                # 设置KL改进单元格颜色
                if kl_imp > 50:
                    table[(i+1, 3)].set_facecolor('#26de81')  # 明亮的绿色，表示显著改进
                    table[(i+1, 3)].set_text_props(weight='bold')
                elif kl_imp > 20:
                    table[(i+1, 3)].set_facecolor('#c6efce')  # 浅绿色，表示良好改进
                elif kl_imp < 0:
                    table[(i+1, 3)].set_facecolor('#ffc7ce')  # 浅红色，表示恶化
                
                # 设置Wasserstein改进单元格颜色
                if wass_imp > 50:
                    table[(i+1, 6)].set_facecolor('#26de81')  # 明亮的绿色
                    table[(i+1, 6)].set_text_props(weight='bold')
                elif wass_imp > 20:
                    table[(i+1, 6)].set_facecolor('#c6efce')  # 浅绿色
                elif wass_imp < 0:
                    table[(i+1, 6)].set_facecolor('#ffc7ce')  # 浅红色
                
                # 根据阈值为初始KL和Wasserstein值设置颜色
                if kl_before_val > KL_SEVERE_THRESHOLD:
                    table[(i+1, 1)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 1)].set_text_props(weight='bold', color='white')
                elif kl_before_val > KL_MODERATE_THRESHOLD:
                    table[(i+1, 1)].set_facecolor(MODERATE_COLOR)
                
                if wass_before_val > WASS_SEVERE_THRESHOLD:
                    table[(i+1, 4)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 4)].set_text_props(weight='bold', color='white')
                elif wass_before_val > WASS_MODERATE_THRESHOLD:
                    table[(i+1, 4)].set_facecolor(MODERATE_COLOR)
                
                # 根据阈值为对齐后的KL和Wasserstein值设置颜色
                if kl_after_val > KL_SEVERE_THRESHOLD:
                    table[(i+1, 2)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 2)].set_text_props(weight='bold', color='white')
                elif kl_after_val > KL_MODERATE_THRESHOLD:
                    table[(i+1, 2)].set_facecolor(MODERATE_COLOR)
                else:
                    table[(i+1, 2)].set_facecolor(LOW_COLOR)
                    table[(i+1, 2)].set_text_props(color='white')
                
                if wass_after_val > WASS_SEVERE_THRESHOLD:
                    table[(i+1, 5)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 5)].set_text_props(weight='bold', color='white')
                elif wass_after_val > WASS_MODERATE_THRESHOLD:
                    table[(i+1, 5)].set_facecolor(MODERATE_COLOR)
                else:
                    table[(i+1, 5)].set_facecolor(LOW_COLOR)
                    table[(i+1, 5)].set_text_props(color='white')
            else:
                # 为简单情况设置颜色
                severity = row[3]
                kl_val = float(row[1])
                wass_val = float(row[2])
                
                # 设置严重程度单元格颜色
                if severity == "HIGH":
                    table[(i+1, 3)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 3)].set_text_props(weight='bold', color='white')
                elif severity == "MEDIUM":
                    table[(i+1, 3)].set_facecolor(MODERATE_COLOR)
                    table[(i+1, 3)].set_text_props(weight='bold')
                else:
                    table[(i+1, 3)].set_facecolor(LOW_COLOR)
                    table[(i+1, 3)].set_text_props(weight='bold', color='white')
                
                # 根据阈值为KL和Wasserstein值设置颜色
                if kl_val > KL_SEVERE_THRESHOLD:
                    table[(i+1, 1)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 1)].set_text_props(weight='bold', color='white')
                elif kl_val > KL_MODERATE_THRESHOLD:
                    table[(i+1, 1)].set_facecolor(MODERATE_COLOR)
                
                if wass_val > WASS_SEVERE_THRESHOLD:
                    table[(i+1, 2)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 2)].set_text_props(weight='bold', color='white')
                elif wass_val > WASS_MODERATE_THRESHOLD:
                    table[(i+1, 2)].set_facecolor(MODERATE_COLOR)
        
        # 添加表格标题
        if X_target_aligned is not None:
            plt.title(f'Feature Distribution Shift and {method_name} Alignment Metrics', fontsize=14, fontweight='bold')
        else:
            plt.title('Feature Distribution Shift Metrics', fontsize=14, fontweight='bold')
        
        # 添加阈值说明
        legend_text = (
            f"Distribution Shift Severity Thresholds:\n"
            f"• HIGH: KL > {KL_SEVERE_THRESHOLD} or Wasserstein > {WASS_SEVERE_THRESHOLD}\n"
            f"• MEDIUM: KL > {KL_MODERATE_THRESHOLD} or Wasserstein > {WASS_MODERATE_THRESHOLD}\n"
            f"• LOW: Otherwise"
        )
        plt.figtext(0.5, 0.03, legend_text, ha="center", fontsize=10, 
                  bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Feature statistics table saved to: {save_path}")
            return save_path
        
    except Exception as e:
        logging.error(f"生成特征统计表时出错: {str(e)}")
        raise
    finally:
        plt.close()
        close_figures()

def detect_outliers(X_source, X_target, percentile=95):
    """检测异常点"""
    min_dist_source = np.min(cdist(X_source, X_target), axis=1)
    min_dist_target = np.min(cdist(X_target, X_source), axis=1)
    
    source_threshold = np.percentile(min_dist_source, percentile)
    target_threshold = np.percentile(min_dist_target, percentile)
    
    source_outliers = np.where(min_dist_source > source_threshold)[0]
    target_outliers = np.where(min_dist_target > target_threshold)[0]
    
    return source_outliers, target_outliers, min_dist_source, min_dist_target

# 可视化函数
def visualize_tsne(X_source, X_target, y_source=None, y_target=None, 
                   X_target_aligned=None, title='t-SNE Visualization', 
                   save_path=None, detect_anomalies=True, method_name="MMD"):
    """t-SNE可视化源域和目标域特征分布"""
    # 先关闭所有已存在的图形
    close_figures()
    
    # 使用变量跟踪创建的图形，确保正确关闭
    fig1 = None
    fig2 = None
    
    try:
        if X_target_aligned is not None:
            # 检查适应前后目标域数据是否确实不同
            data_diff = np.abs(X_target - X_target_aligned).mean()
            logging.info(f"Target data before and after adaptation difference: {data_diff:.6f}")
            
            if data_diff < 1e-6:
                logging.warning("WARNING: Target data before and after adaptation is almost identical!")
                logging.warning("This may indicate an issue with the domain adaptation process.")
            
            # 计算对齐前后的域差异
            before_metrics = compute_domain_discrepancy(X_source, X_target)
            after_metrics = compute_domain_discrepancy(X_source, X_target_aligned)
            
            # 计算改进百分比
            improvement = {}
            for k in before_metrics:
                if k in ['kernel_mean_difference', 'kl_per_feature', 'wasserstein_per_feature']:
                    continue
                elif k == 'kernel_mean_difference':  # 这个指标越大越好
                    improvement[k] = (after_metrics[k] - before_metrics[k]) / before_metrics[k] * 100
                else:  # 其他指标越小越好
                    improvement[k] = (before_metrics[k] - after_metrics[k]) / before_metrics[k] * 100
            
            logging.info("Domain discrepancy improvement rates:")
            for k, v in improvement.items():
                logging.info(f"  {k}: {v:.2f}%")
            
            # 创建1行3列的布局
            fig1 = plt.figure(figsize=(24, 8))
            gs = fig1.add_gridspec(1, 3)
            
            # 标准化数据
            scaler = StandardScaler()
            X_source_scaled = scaler.fit_transform(X_source.copy())
            X_target_scaled = scaler.transform(X_target.copy())
            X_target_aligned_scaled = scaler.transform(X_target_aligned.copy())

            # 对齐前 - 单独计算t-SNE，确保前后图不同
            combined_before = np.vstack([X_source_scaled, X_target_scaled])
            tsne_before = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_before)-1))
            X_tsne_before = tsne_before.fit_transform(combined_before)
            X_source_tsne_before = X_tsne_before[:len(X_source)]
            X_target_tsne_before = X_tsne_before[len(X_source):]
            
            # 对齐后 - 单独计算t-SNE
            combined_after = np.vstack([X_source_scaled, X_target_aligned_scaled])
            tsne_after = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_after)-1))
            X_tsne_after = tsne_after.fit_transform(combined_after)
            X_source_tsne_after = X_tsne_after[:len(X_source)]
            X_target_aligned_tsne = X_tsne_after[len(X_source):]
            
            # 对齐前可视化
            ax1 = fig1.add_subplot(gs[0, 0])
            ax1.set_title('Before Adaptation')
            ax1.scatter(X_source_tsne_before[:, 0], X_source_tsne_before[:, 1], alpha=0.7, label='Source')
            ax1.scatter(X_target_tsne_before[:, 0], X_target_tsne_before[:, 1], alpha=0.7, label='Target')
            
            # 如果有标签，用颜色表示类别
            if y_source is not None and y_target is not None:
                ax1.clear()  # 清除之前的绘图
                for cls in np.unique(np.concatenate([y_source, y_target])):
                    ax1.scatter(X_source_tsne_before[y_source == cls, 0], X_source_tsne_before[y_source == cls, 1], 
                                alpha=0.7, label=f'Source-Class{cls}')
                    ax1.scatter(X_target_tsne_before[y_target == cls, 0], X_target_tsne_before[y_target == cls, 1], 
                                alpha=0.7, marker='x', label=f'Target-Class{cls}')
            
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 对齐后可视化
            ax2 = fig1.add_subplot(gs[0, 1])
            ax2.set_title(f'After {method_name} Adaptation')
            ax2.scatter(X_source_tsne_after[:, 0], X_source_tsne_after[:, 1], alpha=0.7, label='Source')
            ax2.scatter(X_target_aligned_tsne[:, 0], X_target_aligned_tsne[:, 1], alpha=0.7, label='Target (Aligned)')
            
            if y_source is not None and y_target is not None:
                ax2.clear()  # 清除之前的绘图
                for cls in np.unique(np.concatenate([y_source, y_target])):
                    ax2.scatter(X_source_tsne_after[y_source == cls, 0], X_source_tsne_after[y_source == cls, 1], 
                                alpha=0.7, label=f'Source-Class{cls}')
                    ax2.scatter(X_target_aligned_tsne[y_target == cls, 0], X_target_aligned_tsne[y_target == cls, 1], 
                                alpha=0.7, marker='x', label=f'Target-Class{cls}')
            
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 指标对比
            ax3 = fig1.add_subplot(gs[0, 2])
            ax3.set_title('Domain Discrepancy Metrics')
            
            metrics_to_show = ['mmd', 'kl_divergence', 'wasserstein_distance', 'covariance_difference']
            metrics_values_before = [before_metrics.get(m, 0) for m in metrics_to_show]
            metrics_values_after = [after_metrics.get(m, 0) for m in metrics_to_show]
            
            x = np.arange(len(metrics_to_show))
            width = 0.35
            
            ax3.bar(x - width/2, metrics_values_before, width, label='Before')
            ax3.bar(x + width/2, metrics_values_after, width, label=f'After {method_name}')
            
            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Value')
            ax3.set_xticks(x)
            ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_show], rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(metrics_values_before):
                ax3.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            
            for i, v in enumerate(metrics_values_after):
                ax3.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
                
            # 添加改进率
            for i, m in enumerate(metrics_to_show):
                if m in improvement:
                    y_pos = max(metrics_values_before[i], metrics_values_after[i]) + 0.05
                    ax3.text(i, y_pos, f'↓{improvement[m]:.1f}%', ha='center', va='bottom', color='green' if improvement[m] > 0 else 'red')
        else:
            # 只有源域和目标域
            fig1 = plt.figure(figsize=(18, 10))
            plt.scatter(X_source_tsne_before[:, 0], X_source_tsne_before[:, 1], alpha=0.7, label='Source')
            plt.scatter(X_target_tsne_before[:, 0], X_target_tsne_before[:, 1], alpha=0.7, label='Target')
            
            # 类别标签
            if y_source is not None and y_target is not None:
                fig2 = plt.figure(figsize=(15, 10))
                
                for cls in np.unique(np.concatenate([y_source, y_target])):
                    plt.scatter(X_source_tsne_before[y_source == cls, 0], X_source_tsne_before[y_source == cls, 1], 
                                alpha=0.7, label=f'Source-Class{cls}')
                    plt.scatter(X_target_tsne_before[y_target == cls, 0], X_target_tsne_before[y_target == cls, 1], 
                                alpha=0.7, marker='x', label=f'Target-Class{cls}')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            if fig1:
                fig1.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"t-SNE plot saved to: {save_path}")
            
        return X_source_tsne_before, X_target_tsne_before, X_target_aligned_tsne if X_target_aligned is not None else None
    
    except Exception as e:
        logging.error(f"t-SNE可视化时出错: {str(e)}")
        raise
    
    finally:
        # 确保所有图形都被关闭
        if fig1:
            plt.close(fig1)
        if fig2:
            plt.close(fig2)
        # 为安全起见，再关闭所有图形
        close_figures()

def visualize_feature_histograms(X_source, X_target, X_target_aligned=None, feature_names=None, 
                                n_features_to_plot=None, title='Feature Distribution Comparison', save_path=None, method_name="MMD"):
    """绘制特征直方图比较分布"""
    try:
        # 为空时生成默认特征名
        if feature_names is None:
            feature_names = [f'Feature{i+1}' for i in range(X_source.shape[1])]
        
        # 添加日志，记录传入的特征名列表
        logging.info(f"特征直方图可视化: 收到特征名称列表: {feature_names[:5]}... 等{len(feature_names)}个特征")
        logging.info(f"源域X_source数据形状: {X_source.shape}, 目标域X_target数据形状: {X_target.shape}")
        
        # 确保特征名列表长度与数据维度匹配
        if len(feature_names) != X_source.shape[1]:
            logging.warning(f"警告: 特征名称数量({len(feature_names)})与特征维度({X_source.shape[1]})不匹配")
            # 截断或扩展特征名列表以匹配特征维度
            if len(feature_names) > X_source.shape[1]:
                feature_names = feature_names[:X_source.shape[1]]
                logging.warning(f"已截断特征名称列表至{len(feature_names)}个")
            else:
                # 扩展特征名列表，确保使用正确的格式
                feature_names = feature_names + [f'Feature{i+1+len(feature_names)}' for i in range(X_source.shape[1] - len(feature_names))]
                logging.warning(f"已扩展特征名称列表至{len(feature_names)}个")
        
        # If n_features_to_plot is None or exceeds the number of features, plot all features
        n_features = X_source.shape[1] if n_features_to_plot is None else min(n_features_to_plot, X_source.shape[1])
        
        # Define thresholds for distribution shift severity
        KL_SEVERE_THRESHOLD = 0.5
        KL_MODERATE_THRESHOLD = 0.2
        WASS_SEVERE_THRESHOLD = 0.5
        WASS_MODERATE_THRESHOLD = 0.2
        
        # Define colors for different severity levels
        SEVERE_COLOR = '#ff6b6b'  # Red
        MODERATE_COLOR = '#feca57'  # Yellow/Orange
        LOW_COLOR = '#1dd1a1'  # Green
        
        # Create and save histogram statistics table (all features)
        _, kl_div = calculate_kl_divergence(X_source, X_target)
        _, wasserstein_dist = calculate_wasserstein_distances(X_source, X_target)
        stats_list = []
        for i in range(X_source.shape[1]):
            stats_list.append({
                'Feature': feature_names[i],
                'KL Before': kl_div[f'feature_{i}'],
                'Wasserstein Before': wasserstein_dist[f'feature_{i}']
            })
        stats_df = pd.DataFrame(stats_list)
        if save_path:
            stats_csv = save_path.replace('.png', '_stats.csv')
            stats_df.to_csv(stats_csv, index=False)
            logging.info(f"Histogram stats table saved to: {stats_csv}")
            # Save stats table as figure
            fig_table = plt.figure(figsize=(12, 6))
            ax_table = fig_table.add_subplot(111)
            ax_table.axis('off')
            table = ax_table.table(cellText=stats_df.values, colLabels=stats_df.columns, loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            stats_table_path = save_path.replace('.png', '_stats_table.png')
            fig_table.savefig(stats_table_path, dpi=300, bbox_inches='tight')
            logging.info(f"Histogram stats table figure saved to: {stats_table_path}")
            plt.close(fig_table)
        
        if X_target_aligned is not None:
            # 计算每个特征的KL散度和Wasserstein距离
            _, kl_div_before = calculate_kl_divergence(X_source, X_target)
            _, kl_div_after = calculate_kl_divergence(X_source, X_target_aligned)
            
            _, wasserstein_before = calculate_wasserstein_distances(X_source, X_target)
            _, wasserstein_after = calculate_wasserstein_distances(X_source, X_target_aligned)
            
            # 根据对齐改进排序特征
            kl_improvement = {}
            for i in range(X_source.shape[1]):
                feature_key = f'feature_{i}'
                if feature_key in kl_div_before and feature_key in kl_div_after:
                    kl_improvement[i] = kl_div_before[feature_key] - kl_div_after[feature_key]
            
            # 使用所有特征而不是只选择5个
            selected_features = list(range(X_source.shape[1]))
            
            # 直方图可视化
            fig, axes = plt.subplots(n_features, 2, figsize=(18, 4*n_features))
            
            # 处理单行的情况
            if len(selected_features) == 1:
                if X_target_aligned is not None:
                    axes = [axes]
                else:
                    axes = [[axes]]
            elif X_target_aligned is None:
                # 如果没有对齐后的数据，将axes转为二维数组形式
                axes = [[ax] for ax in axes]
            
            # 绘制每个选定特征的直方图
            for i, feature_idx in enumerate(selected_features):
                feature_name = feature_names[feature_idx]
                feature_key = f'feature_{feature_idx}'
                
                # 对齐前的直方图
                ax1 = axes[i][0]
                sns.histplot(X_source[:, feature_idx], kde=True, ax=ax1, color='blue', alpha=0.5, label='Source')
                sns.histplot(X_target[:, feature_idx], kde=True, ax=ax1, color='red', alpha=0.5, label='Target')
                
                # 计算原始分布指标
                kl_before = kl_div_before[feature_key]
                wass_before = wasserstein_before[feature_key]
                
                # 确定严重程度
                if kl_before > KL_SEVERE_THRESHOLD or wass_before > WASS_SEVERE_THRESHOLD:
                    severity_level = "HIGH SHIFT"
                    box_color = SEVERE_COLOR
                    text_color = 'white'
                elif kl_before > KL_MODERATE_THRESHOLD or wass_before > WASS_MODERATE_THRESHOLD:
                    severity_level = "MEDIUM SHIFT"
                    box_color = MODERATE_COLOR
                    text_color = 'black'
                else:
                    severity_level = "LOW SHIFT"
                    box_color = LOW_COLOR
                    text_color = 'white'
                
                # 设置标题
                ax1.set_title(f'Before {method_name} - {feature_name} ({severity_level})', 
                             color=text_color, fontweight='bold', 
                             bbox=dict(facecolor=box_color, edgecolor='none', pad=3))
                
                ax1.legend()
                
                # 添加指标文本
                ax1.text(0.05, 0.95, 
                        f'KL: {kl_before:.4f}\nWass: {wass_before:.4f}', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # 如果有对齐后的数据，绘制对齐后的直方图
                if X_target_aligned is not None:
                    ax2 = axes[i][1]
                    sns.histplot(X_source[:, feature_idx], kde=True, ax=ax2, color='blue', alpha=0.5, label='Source')
                    sns.histplot(X_target_aligned[:, feature_idx], kde=True, ax=ax2, color='red', alpha=0.5, label=f'Target ({method_name})')
                    
                    # 计算对齐后指标
                    kl_after = kl_div_after[feature_key]
                    wass_after = wasserstein_after[feature_key]
                    
                    # 计算改进百分比
                    kl_imp_pct = (kl_before - kl_after) / kl_before * 100 if kl_before > 0 else 0
                    wass_imp_pct = (wass_before - wass_after) / wass_before * 100 if wass_before > 0 else 0
                    
                    # 设置改进文本框颜色
                    if kl_imp_pct > 50 or wass_imp_pct > 50:
                        imp_box_color = 'green'
                        imp_text = "MAJOR IMPROVEMENT"
                    elif kl_imp_pct > 20 or wass_imp_pct > 20:
                        imp_box_color = 'lightgreen'
                        imp_text = "GOOD IMPROVEMENT"
                    elif kl_imp_pct < 0 or wass_imp_pct < 0:
                        imp_box_color = 'lightcoral'
                        imp_text = "DETERIORATED"
                    else:
                        imp_box_color = 'lightyellow'
                        imp_text = "MINOR IMPROVEMENT"
                    
                    # 设置标题
                    ax2.set_title(f'After {method_name} - {feature_name}')
                    ax2.legend()
                    
                    # 添加改进文本
                    ax2.text(0.05, 0.95, 
                            f'KL: {kl_after:.4f} (↓{kl_imp_pct:.1f}%)\nWass: {wass_after:.4f} (↓{wass_imp_pct:.1f}%)\n{imp_text}', 
                            transform=ax2.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor=imp_box_color, alpha=0.7))
        
        # 设置整体标题
        plt.suptitle(f'Feature Distribution Visualization (Top {len(selected_features)} Features)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 添加说明文本
        legend_text = (
            f"Distribution Shift Thresholds - HIGH: KL > {KL_SEVERE_THRESHOLD} or Wass > {WASS_SEVERE_THRESHOLD} | "
            f"MEDIUM: KL > {KL_MODERATE_THRESHOLD} or Wass > {WASS_MODERATE_THRESHOLD} | LOW: Otherwise"
        )
        plt.figtext(0.5, 0.01, legend_text, ha="center", fontsize=9, 
                  bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # 保存图像
        if save_path:
            visual_stats_path = save_path if '.png' in save_path else save_path + '_visual_stats.png'
            plt.savefig(visual_stats_path, dpi=300, bbox_inches='tight')
            logging.info(f"Visual stats histogram saved to: {visual_stats_path}")
            return visual_stats_path
        
        plt.show()
        return None
    
    except Exception as e:
        logging.error(f"绘制特征直方图时出错: {str(e)}")
        raise
    finally:
        # 清理
        plt.close()

def histograms_visual_stats_table(X_source, X_target, X_target_aligned=None, feature_names=None, 
                                 save_path=None, method_name="MMD"):
    """生成可视化特征分布统计表，用于详细展示域适应前后的特征分布变化
    
    参数:
    - X_source: 源域特征矩阵
    - X_target: 目标域特征矩阵
    - X_target_aligned: 对齐后的目标域特征矩阵（如果为None，则仅生成原始差异统计）
    - feature_names: 特征名称列表
    - save_path: 保存路径
    - method_name: 域适应方法名称，用于标题显示
    
    返回:
    - 保存的图像路径
    """
    # 先关闭所有已存在的图形
    close_figures()
    
    try:
        # 为空时生成默认特征名，修改为不带空格的格式
        if feature_names is None:
            feature_names = [f'Feature{i+1}' for i in range(X_source.shape[1])]
        
        # 确保特征名列表长度与数据维度匹配
        if len(feature_names) != X_source.shape[1]:
            logging.warning(f"警告: 特征名称数量({len(feature_names)})与特征维度({X_source.shape[1]})不匹配")
            if len(feature_names) > X_source.shape[1]:
                feature_names = feature_names[:X_source.shape[1]]
                logging.warning(f"已截断特征名称列表至{len(feature_names)}个")
            else:
                feature_names = feature_names + [f'Feature{i+1+len(feature_names)}' for i in range(X_source.shape[1] - len(feature_names))]
                logging.warning(f"已扩展特征名称列表至{len(feature_names)}个")
        
        # 定义阈值
        KL_SEVERE_THRESHOLD = 0.5
        KL_MODERATE_THRESHOLD = 0.2
        WASS_SEVERE_THRESHOLD = 0.5
        WASS_MODERATE_THRESHOLD = 0.2
        
        # 定义不同严重程度的颜色
        SEVERE_COLOR = '#ff6b6b'  # Red
        MODERATE_COLOR = '#feca57'  # Yellow/Orange
        LOW_COLOR = '#1dd1a1'  # Green
        
        # 计算特征的最大数量和显示数量
        n_features = X_source.shape[1]
        n_features_to_plot = min(10, n_features) if X_target_aligned is None else min(5, n_features)
        
        # 计算KL散度和Wasserstein距离
        _, kl_div_before = calculate_kl_divergence(X_source, X_target)
        _, wasserstein_before = calculate_wasserstein_distances(X_source, X_target)
        
        if X_target_aligned is not None:
            _, kl_div_after = calculate_kl_divergence(X_source, X_target_aligned)
            _, wasserstein_after = calculate_wasserstein_distances(X_source, X_target_aligned)
            
            # 计算改进率
            kl_improvement = {}
            for i in range(n_features):
                feature_key = f'feature_{i}'
                kl_before_val = kl_div_before[feature_key]
                kl_after_val = kl_div_after[feature_key]
                kl_improvement[i] = kl_before_val - kl_after_val
            
            # 按改进率排序选择特征
            top_features = sorted(kl_improvement.items(), key=lambda x: x[1], reverse=True)[:n_features_to_plot]
            selected_features = [idx for idx, _ in top_features]
        else:
            # 按原始KL散度排序选择特征
            kl_values = [(i, kl_div_before[f'feature_{i}']) for i in range(n_features)]
            selected_features = [idx for idx, _ in sorted(kl_values, key=lambda x: x[1], reverse=True)[:n_features_to_plot]]
        
        # 创建图形
        fig, axes = plt.subplots(len(selected_features), 2 if X_target_aligned is not None else 1, 
                                figsize=(18 if X_target_aligned is not None else 10, 4*len(selected_features)))
        
        # 处理单行的情况
        if len(selected_features) == 1:
            if X_target_aligned is not None:
                axes = [axes]
            else:
                axes = [[axes]]
        elif X_target_aligned is None:
            # 如果没有对齐后的数据，将axes转为二维数组形式
            axes = [[ax] for ax in axes]
        
        # 绘制每个选定特征的直方图
        for i, feature_idx in enumerate(selected_features):
            feature_name = feature_names[feature_idx]
            feature_key = f'feature_{feature_idx}'
            
            # 原始分布直方图
            ax1 = axes[i][0]
            sns.histplot(X_source[:, feature_idx], kde=True, ax=ax1, color='blue', alpha=0.5, label='Source')
            sns.histplot(X_target[:, feature_idx], kde=True, ax=ax1, color='red', alpha=0.5, label='Target')
            
            # 计算原始分布指标
            kl_before = kl_div_before[feature_key]
            wass_before = wasserstein_before[feature_key]
            
            # 确定严重程度
            if kl_before > KL_SEVERE_THRESHOLD or wass_before > WASS_SEVERE_THRESHOLD:
                severity_level = "HIGH SHIFT"
                box_color = SEVERE_COLOR
                text_color = 'white'
            elif kl_before > KL_MODERATE_THRESHOLD or wass_before > WASS_MODERATE_THRESHOLD:
                severity_level = "MEDIUM SHIFT"
                box_color = MODERATE_COLOR
                text_color = 'black'
            else:
                severity_level = "LOW SHIFT"
                box_color = LOW_COLOR
                text_color = 'white'
            
            # 设置标题
            ax1.set_title(f'Before {method_name} - {feature_name} ({severity_level})', 
                         color=text_color, fontweight='bold', 
                         bbox=dict(facecolor=box_color, edgecolor='none', pad=3))
            
            ax1.legend()
            
            # 添加指标文本
            ax1.text(0.05, 0.95, 
                    f'KL: {kl_before:.4f}\nWass: {wass_before:.4f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # 如果有对齐后的数据，绘制对齐后的直方图
            if X_target_aligned is not None:
                ax2 = axes[i][1]
                sns.histplot(X_source[:, feature_idx], kde=True, ax=ax2, color='blue', alpha=0.5, label='Source')
                sns.histplot(X_target_aligned[:, feature_idx], kde=True, ax=ax2, color='red', alpha=0.5, label=f'Target ({method_name})')
                
                # 计算对齐后指标
                kl_after = kl_div_after[feature_key]
                wass_after = wasserstein_after[feature_key]
                
                # 计算改进百分比
                kl_imp_pct = (kl_before - kl_after) / kl_before * 100 if kl_before > 0 else 0
                wass_imp_pct = (wass_before - wass_after) / wass_before * 100 if wass_before > 0 else 0
                
                # 设置改进文本框颜色
                if kl_imp_pct > 50 or wass_imp_pct > 50:
                    imp_box_color = 'green'
                    imp_text = "MAJOR IMPROVEMENT"
                elif kl_imp_pct > 20 or wass_imp_pct > 20:
                    imp_box_color = 'lightgreen'
                    imp_text = "GOOD IMPROVEMENT"
                elif kl_imp_pct < 0 or wass_imp_pct < 0:
                    imp_box_color = 'lightcoral'
                    imp_text = "DETERIORATED"
                else:
                    imp_box_color = 'lightyellow'
                    imp_text = "MINOR IMPROVEMENT"
                
                # 设置标题
                ax2.set_title(f'After {method_name} - {feature_name}')
                ax2.legend()
                
                # 添加改进文本
                ax2.text(0.05, 0.95, 
                        f'KL: {kl_after:.4f} (↓{kl_imp_pct:.1f}%)\nWass: {wass_after:.4f} (↓{wass_imp_pct:.1f}%)\n{imp_text}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=imp_box_color, alpha=0.7))
        
        # 设置整体标题
        plt.suptitle(f'Feature Distribution Visualization (Top {len(selected_features)} Features)', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # 添加说明文本
        legend_text = (
            f"Distribution Shift Thresholds - HIGH: KL > {KL_SEVERE_THRESHOLD} or Wass > {WASS_SEVERE_THRESHOLD} | "
            f"MEDIUM: KL > {KL_MODERATE_THRESHOLD} or Wass > {WASS_MODERATE_THRESHOLD} | LOW: Otherwise"
        )
        plt.figtext(0.5, 0.01, legend_text, ha="center", fontsize=9, 
                  bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # 保存图像
        if save_path:
            visual_stats_path = save_path if '.png' in save_path else save_path + '_visual_stats.png'
            plt.savefig(visual_stats_path, dpi=300, bbox_inches='tight')
            logging.info(f"Visual stats histogram saved to: {visual_stats_path}")
            return visual_stats_path
        
        plt.show()
        return None
    
    except Exception as e:
        logging.error(f"生成可视化特征分布统计时出错: {str(e)}")
        raise
    finally:
        plt.close()

def compare_before_after_adaptation(source_features, target_features, adapted_target_features, 
                                   source_labels=None, target_labels=None, save_dir=None, method_name="MMD", feature_names=None):
    """比较域适应前后的特征分布差异"""
    # 先关闭所有已存在的图形
    close_figures()
    
    # 用于跟踪创建的图形对象
    created_figures = []
    
    try:
        # 计算域差异指标
        metrics_before = compute_domain_discrepancy(source_features, target_features)
        metrics_after = compute_domain_discrepancy(source_features, adapted_target_features)
        
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            tsne_path = os.path.join(save_dir, "tsne_comparison.png")
            hist_path = os.path.join(save_dir, "histogram_comparison.png")
            metrics_path = os.path.join(save_dir, "metrics_comparison.png")
            stats_table_path = os.path.join(save_dir, "stats_table.png")
            visual_stats_path = os.path.join(save_dir, "visual_stats_table.png")
            histograms_visual_path = os.path.join(save_dir, "histograms_visual.png")
        else:
            tsne_path = hist_path = metrics_path = stats_table_path = visual_stats_path = histograms_visual_path = None
        
        # 1. t-SNE可视化 - 同时显示适应前和适应后
        visualize_tsne(
            X_source=source_features, 
            X_target=target_features,
            y_source=source_labels,
            y_target=target_labels,
            X_target_aligned=adapted_target_features,
            title=f"{method_name} Domain Adaptation Before and After",
            save_path=tsne_path,
            method_name=method_name
        )
        
        # 2. 特征直方图可视化 - 选择最具代表性的特征
        # 计算KL散度改进最大的前5个特征
        _, kl_before = calculate_kl_divergence(source_features, target_features)
        _, kl_after = calculate_kl_divergence(source_features, adapted_target_features)
        
        # 计算改进率
        kl_improvement = {}
        for i in range(source_features.shape[1]):
            feature_key = f'feature_{i}'
            kl_before_val = kl_before[feature_key]
            kl_after_val = kl_after[feature_key]
            kl_improvement[i] = kl_before_val - kl_after_val
        
        # 使用所有特征而不是只选择5个
        selected_features = list(range(source_features.shape[1]))
        
        # 如果没有提供特征名称，则生成默认特征名称列表
        if feature_names is None:
            feature_names = [f'Feature{i+1}' for i in range(source_features.shape[1])]
            logging.info(f"使用默认特征名称: Feature1 - Feature{source_features.shape[1]}")
        
        # 3. 生成统计表格
        histograms_stats_table(
            X_source=source_features,
            X_target=target_features,
            X_target_aligned=adapted_target_features,
            feature_names=feature_names,
            save_path=stats_table_path,
            method_name=method_name
        )
        
        # 4. 生成可视化统计表格 (新增的函数调用)
        histograms_visual_stats_table(
            X_source=source_features,
            X_target=target_features,
            X_target_aligned=adapted_target_features,
            feature_names=feature_names,
            save_path=visual_stats_path,
            method_name=method_name
        )
        
        # 5. 直方图可视化
        visualize_feature_histograms(
            X_source=source_features,
            X_target=target_features,
            X_target_aligned=adapted_target_features,
            feature_names=feature_names,
            n_features_to_plot=None,  # 设为None以显示所有特征
            title="All Features Distribution Comparison",
            save_path=hist_path,
            method_name=method_name
        )
        
        # 6. 改善前后的域差异指标图
        fig_metrics = plt.figure(figsize=(12, 6))
        created_figures.append(fig_metrics)
        
        # 选择要绘制的指标
        metrics_to_plot = ['mmd', 'kl_divergence', 'wasserstein_distance', 'covariance_difference']
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        before_values = [metrics_before[metric] for metric in metrics_to_plot]
        after_values = [metrics_after[metric] for metric in metrics_to_plot]
        
        ax_metrics = fig_metrics.add_subplot(111)
        rects1 = ax_metrics.bar(x - width/2, before_values, width, label='Before Adaptation')
        rects2 = ax_metrics.bar(x + width/2, after_values, width, label='After Adaptation')
        
        # 添加改进百分比标签
        for i, (before, after) in enumerate(zip(before_values, after_values)):
            if before > 0:
                reduction = (before - after) / before * 100
                ax_metrics.text(i, max(before, after) + 0.05, f"{reduction:.1f}% ↓", 
                              ha='center', va='bottom', color='green' if reduction > 0 else 'red',
                              fontweight='bold')
        
        # 添加具体数值标签到每个柱状图
        for i, v in enumerate(before_values):
            ax_metrics.text(i - width/2, v * 0.5, f"{v:.4f}", ha='center', va='center', 
                          color='white', fontweight='bold', fontsize=9)
        
        for i, v in enumerate(after_values):
            ax_metrics.text(i + width/2, v * 0.5, f"{v:.4f}", ha='center', va='center', 
                          color='white', fontweight='bold', fontsize=9)
        
        ax_metrics.set_ylabel('Value')
        ax_metrics.set_title(f'Domain Discrepancy Metrics Before and After {method_name}')
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
        ax_metrics.legend()
        
        plt.tight_layout()
        
        if metrics_path:
            plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
            logging.info(f"Metrics comparison saved to: {metrics_path}")
        
        # 7. 关闭所有图形
        for fig in created_figures:
            plt.close(fig)
        
        # 返回各种可视化的路径
        return {
            'tsne': tsne_path,
            'histograms': hist_path,
            'metrics': metrics_path,
            'histograms_stats_table': stats_table_path,
            'histograms_visual_stats_table': visual_stats_path,
            'histograms_visual': histograms_visual_path
        }
    except Exception as e:
        logging.error(f"比较域适应前后差异时出错: {str(e)}")
        raise
    finally:
        # 确保所有图形都被关闭
        for fig in created_figures:
            plt.close(fig)
        # 为安全起见，再关闭所有图形
        close_figures()

def load_and_preprocess_data(source_features_path, target_features_path, source_labels_path=None, target_labels_path=None):
    """加载并预处理特征和标签数据"""
    source_features = np.load(source_features_path)
    target_features = np.load(target_features_path)
    
    source_labels = np.load(source_labels_path) if source_labels_path else None
    target_labels = np.load(target_labels_path) if target_labels_path else None
    
    return source_features, target_features, source_labels, target_labels

def visualize_mmd_adaptation_results(source_features, target_features, adapted_features, 
                                    source_labels=None, target_labels=None, output_dir='./mmd_visualizations',
                                    feature_names=None, method_name="MMD"):
    """
    生成MMD域适应结果的可视化，包括t-SNE、特征直方图等
    
    参数:
    - source_features: 源域特征
    - target_features: 目标域特征
    - adapted_features: 经过MMD适应后的目标域特征
    - source_labels: 源域标签（可选）
    - target_labels: 目标域标签（可选）
    - output_dir: 输出目录
    - feature_names: 特征名称列表（可选）
    - method_name: 域适应方法名称，默认为"MMD"
    
    返回:
    - 生成的可视化路径字典
    """
    # 先关闭所有已存在的图形
    close_figures()
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"创建输出目录: {output_dir}")
        
        # 如果没有提供特征名称，使用默认命名
        if feature_names is None:
            feature_names = [f'Feature{i+1}' for i in range(source_features.shape[1])]
            logging.info(f"使用默认特征名称: Feature1 - Feature{source_features.shape[1]}")
        
        # 确保特征名列表长度与数据维度匹配
        if len(feature_names) != source_features.shape[1]:
            logging.warning(f"警告: 特征名称数量({len(feature_names)})与特征维度({source_features.shape[1]})不匹配")
            if len(feature_names) > source_features.shape[1]:
                feature_names = feature_names[:source_features.shape[1]]
                logging.warning(f"已截断特征名称列表至{len(feature_names)}个")
            else:
                # 扩展特征名列表，保持连续命名，且使用正确格式
                feature_names = feature_names + [f'Feature{i+1+len(feature_names)}' for i in range(source_features.shape[1] - len(feature_names))]
                logging.warning(f"已扩展特征名称列表至{len(feature_names)}个")
        
        # 设置各个可视化的输出路径
        tsne_path = os.path.join(output_dir, "tsne_visualization.png")
        histograms_path = os.path.join(output_dir, "feature_histograms.png")
        stats_table_path = os.path.join(output_dir, "feature_stats_table.png")
        comparison_dir = os.path.join(output_dir, "before_after_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 1. t-SNE可视化
        tsne_result = visualize_tsne(
            X_source=source_features,
            X_target=target_features,
            X_target_aligned=adapted_features,
            y_source=source_labels,
            y_target=target_labels,
            title=f"{method_name} Domain Adaptation Visualization",
            save_path=tsne_path,
            method_name=method_name
        )
        logging.info(f"t-SNE可视化完成，保存至: {tsne_path}")
        
        # 2. 计算域间差异指标
        metrics_before = compute_domain_discrepancy(source_features, target_features)
        metrics_after = compute_domain_discrepancy(source_features, adapted_features)
        
        # 3. 生成并保存特征直方图
        histograms_result = visualize_feature_histograms(
            X_source=source_features,
            X_target=target_features,
            X_target_aligned=adapted_features,
            feature_names=feature_names,
            n_features_to_plot=None,  # 显示所有特征
            title=f"Feature Distribution with {method_name} Adaptation",
            save_path=histograms_path,
            method_name=method_name
        )
        logging.info(f"特征直方图可视化完成，保存至: {histograms_path}")
        
        # 4. 生成特征统计表
        stats_table_result = histograms_stats_table(
            X_source=source_features,
            X_target=target_features,
            X_target_aligned=adapted_features,
            feature_names=feature_names,
            save_path=stats_table_path,
            method_name=method_name
        )
        logging.info(f"特征统计表可视化完成，保存至: {stats_table_path}")
        
        # 5. 生成适应前后的对比可视化
        comparison_result = compare_before_after_adaptation(
            source_features=source_features,
            target_features=target_features,
            adapted_target_features=adapted_features,
            source_labels=source_labels,
            target_labels=target_labels,
            save_dir=comparison_dir,
            method_name=method_name,
            feature_names=feature_names
        )
        logging.info(f"适应前后对比可视化完成，保存至: {comparison_dir}")
        
        # 6. 打印域差异指标
        print(f"\n====== {method_name} 域适应前后差异指标 ======")
        for metric_name in ['mmd', 'kl_divergence', 'wasserstein_distance', 'covariance_difference']:
            before_val = metrics_before[metric_name]
            after_val = metrics_after[metric_name]
            if before_val > 0:
                reduction = (before_val - after_val) / before_val * 100
                print(f"{metric_name}: {before_val:.4f} -> {after_val:.4f} (减少 {reduction:.2f}%)")
            else:
                print(f"{metric_name}: {before_val:.4f} -> {after_val:.4f}")
        
        # 7. 返回所有生成的可视化路径
        result_paths = {
            'tsne': tsne_path,
            'histograms': histograms_path,
            'stats_table': stats_table_path,
            'comparison_dir': comparison_dir
        }
        
        # 添加从compare_before_after_adaptation返回的路径
        if comparison_result:
            for key, path in comparison_result.items():
                result_paths[f'comparison_{key}'] = path
        
        return result_paths
        
    except Exception as e:
        logging.error(f"可视化MMD适应结果时出错: {str(e)}")
        raise
    
    finally:
        # 确保所有图形都被关闭
        close_figures()

# 主函数
if __name__ == "__main__":
    setup_logger()
    
    try:
        # 解析命令行参数
        import argparse
        
        parser = argparse.ArgumentParser(description="MMD域适应可视化分析工具")
        parser.add_argument("--source", required=True, help="源域特征NPZ文件路径")
        parser.add_argument("--target", required=True, help="目标域特征NPZ文件路径")
        parser.add_argument("--aligned", required=True, help="对齐后的特征NPZ文件路径")
        parser.add_argument("--source-labels", help="源域标签NPZ文件路径")
        parser.add_argument("--target-labels", help="目标域标签NPZ文件路径")
        parser.add_argument("--output", help="输出目录", default="./mmd_visualizations")
        parser.add_argument("--method", help="域适应方法名称", default="MMD")
        
        args = parser.parse_args()
        
        # 加载数据
        source_features, target_features, source_labels, target_labels = load_and_preprocess_data(
            args.source, args.target, args.source_labels, args.target_labels
        )
        
        # 加载对齐后特征
        adapted_features = np.load(args.aligned)
        
        # 可视化分析结果
        results = visualize_mmd_adaptation_results(
            source_features, 
            target_features, 
            adapted_features,
            source_labels,
            target_labels,
            args.output,
            method_name=args.method
        )
        
        # 打印完整结果路径
        print("\n生成的可视化结果路径:")
        for name, path in results.items():
            if path:
                print(f"- {name}: {path}")
    
    except Exception as e:
        logging.error(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 确保所有图形都被关闭
        plt.close('all')
        logging.info("脚本执行完毕，所有图形已关闭") 