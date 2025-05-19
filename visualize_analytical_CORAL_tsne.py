#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CORAL域适应可视化分析工具
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
from sklearn.metrics.pairwise import rbf_kernel
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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
    """计算MMD"""
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
                   save_path=None, detect_anomalies=True):
    """t-SNE visualization of source and target domain feature distributions"""
    plt.figure(figsize=(18, 10))
    
    # For better visualization, we'll perform different t-SNE transformations:
    # 1. First for source vs target (before alignment)
    # 2. Then for source vs aligned target (after alignment)
    
    # Standardize data - each feature separately
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    if X_target_aligned is not None:
        X_target_aligned_scaled = scaler.transform(X_target_aligned)
        
        # Calculate domain discrepancy before and after alignment
        before_metrics = compute_domain_discrepancy(X_source_scaled, X_target_scaled)
        after_metrics = compute_domain_discrepancy(X_source_scaled, X_target_aligned_scaled)
        
        # Calculate improvement percentage
        improvement = {}
        for k in before_metrics:
            if k in ['kernel_mean_difference', 'kl_per_feature', 'wasserstein_per_feature']:
                continue
            elif k == 'kernel_mean_difference':  # Higher is better for this metric
                improvement[k] = (after_metrics[k] - before_metrics[k]) / before_metrics[k] * 100
            else:  # Lower is better for other metrics
                improvement[k] = (before_metrics[k] - after_metrics[k]) / before_metrics[k] * 100
        
        logging.info("Domain discrepancy improvement rates:")
        for k, v in improvement.items():
            logging.info(f"  {k}: {v:.2f}%")
        
        # Create 1 row 3 column layout
        fig = plt.figure(figsize=(24, 8))
        gs = fig.add_gridspec(1, 3)
        
        # Before alignment - separate t-SNE
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Before Alignment')
        
        # Combine source and target for t-SNE transformation
        X_combined_before = np.vstack([X_source_scaled, X_target_scaled])
        # Use perplexity appropriate for dataset size
        perplexity = min(30, len(X_combined_before) - 1)
        
        # Apply t-SNE for before alignment visualization
        tsne_before = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_before_tsne = tsne_before.fit_transform(X_combined_before)
        
        # Split results
        X_source_tsne_before = X_before_tsne[:len(X_source)]
        X_target_tsne_before = X_before_tsne[len(X_source):]
        
        # Plot before alignment
        ax1.scatter(X_source_tsne_before[:, 0], X_source_tsne_before[:, 1], 
                   alpha=0.7, label='Source', color='blue')
        ax1.scatter(X_target_tsne_before[:, 0], X_target_tsne_before[:, 1], 
                   alpha=0.7, label='Target', color='red')
        
        # If labels are provided, use colors to represent classes
        if y_source is not None and y_target is not None:
            # Close previous plot and create new one with class coloring
            X_source_by_class = {}
            X_target_by_class = {}
            
            classes = np.unique(np.concatenate([y_source, y_target]))
            # 使用不同的颜色映射方案，确保source和target的class颜色有明显区分
            source_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(classes)))
            target_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(classes)))
            
            for i, cls in enumerate(classes):
                source_idx = y_source == cls
                target_idx = y_target == cls
                
                if np.any(source_idx):
                    X_source_by_class[cls] = X_source_tsne_before[source_idx]
                if np.any(target_idx):
                    X_target_by_class[cls] = X_target_tsne_before[target_idx]
                
                # Plot source points for this class
                if cls in X_source_by_class:
                    ax1.scatter(X_source_by_class[cls][:, 0], X_source_by_class[cls][:, 1],
                               color=source_colors[i], marker='o', alpha=0.7,
                               label=f'Source-Class{cls}')
                
                # Plot target points for this class
                if cls in X_target_by_class:
                    ax1.scatter(X_target_by_class[cls][:, 0], X_target_by_class[cls][:, 1],
                               color=target_colors[i], marker='x', alpha=0.7,
                               label=f'Target-Class{cls}')
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # After alignment - separate t-SNE
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('After CORAL Alignment')
        
        # Combine source and aligned target for t-SNE transformation
        X_combined_after = np.vstack([X_source_scaled, X_target_aligned_scaled])
        perplexity = min(30, len(X_combined_after) - 1)
        
        # Apply t-SNE for after alignment visualization
        tsne_after = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_after_tsne = tsne_after.fit_transform(X_combined_after)
        
        # Split results
        X_source_tsne_after = X_after_tsne[:len(X_source)]
        X_target_aligned_tsne = X_after_tsne[len(X_source):]
        
        # Plot after alignment
        ax2.scatter(X_source_tsne_after[:, 0], X_source_tsne_after[:, 1], 
                   alpha=0.7, label='Source', color='blue')
        ax2.scatter(X_target_aligned_tsne[:, 0], X_target_aligned_tsne[:, 1], 
                   alpha=0.7, label='Target (Aligned)', color='red')
        
        # If labels are provided, use colors to represent classes
        if y_source is not None and y_target is not None:
            X_source_by_class = {}
            X_target_by_class = {}
            
            for i, cls in enumerate(classes):
                source_idx = y_source == cls
                target_idx = y_target == cls
                
                if np.any(source_idx):
                    X_source_by_class[cls] = X_source_tsne_after[source_idx]
                if np.any(target_idx):
                    X_target_by_class[cls] = X_target_aligned_tsne[target_idx]
                
                # Plot source points for this class
                if cls in X_source_by_class:
                    ax2.scatter(X_source_by_class[cls][:, 0], X_source_by_class[cls][:, 1],
                               color=source_colors[i], marker='o', alpha=0.7,
                               label=f'Source-Class{cls}')
                
                # Plot target points for this class
                if cls in X_target_by_class:
                    ax2.scatter(X_target_by_class[cls][:, 0], X_target_by_class[cls][:, 1],
                               color=target_colors[i], marker='x', alpha=0.7,
                               label=f'Target-Class{cls}')
        
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Metrics comparison
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title('Domain Discrepancy Metrics')
        
        metrics_to_show = ['mmd', 'kl_divergence', 'wasserstein_distance', 'covariance_difference']
        metrics_values_before = [before_metrics.get(m, 0) for m in metrics_to_show]
        metrics_values_after = [after_metrics.get(m, 0) for m in metrics_to_show]
        
        x = np.arange(len(metrics_to_show))
        width = 0.35
        
        ax3.bar(x - width/2, metrics_values_before, width, label='Before Alignment')
        ax3.bar(x + width/2, metrics_values_after, width, label='After Alignment')
        
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Value')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_show], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(metrics_values_before):
            ax3.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        
        for i, v in enumerate(metrics_values_after):
            ax3.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            
        # Add improvement rate
        for i, m in enumerate(metrics_to_show):
            if m in improvement:
                y_pos = max(metrics_values_before[i], metrics_values_after[i]) + 0.05
                ax3.text(i, y_pos, f'↓{improvement[m]:.1f}%', ha='center', va='bottom', color='green')
    else:
        # Only source and target domains
        # Combine data for single t-SNE transformation
        X_combined = np.vstack([X_source_scaled, X_target_scaled])
        
        # Use perplexity appropriate for dataset size
        perplexity = min(30, len(X_combined) - 1)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(X_combined)
        
        # Split results
        X_source_tsne = X_tsne[:len(X_source)]
        X_target_tsne = X_tsne[len(X_source):]
        
        plt.scatter(X_source_tsne[:, 0], X_source_tsne[:, 1], alpha=0.7, label='Source', color='blue')
        plt.scatter(X_target_tsne[:, 0], X_target_tsne[:, 1], alpha=0.7, label='Target', color='red')
        
        # Class labels
        if y_source is not None and y_target is not None:
            plt.figure(figsize=(15, 10))
            
            classes = np.unique(np.concatenate([y_source, y_target]))
            # 使用不同的颜色映射方案，确保source和target的class颜色有明显区分
            source_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(classes)))
            target_colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(classes)))
            
            for i, cls in enumerate(classes):
                source_idx = y_source == cls
                target_idx = y_target == cls
                
                if np.any(source_idx):
                    plt.scatter(X_source_tsne[source_idx, 0], X_source_tsne[source_idx, 1], 
                              color=source_colors[i], marker='o', alpha=0.7, label=f'Source-Class{cls}')
                
                if np.any(target_idx):
                    plt.scatter(X_target_tsne[target_idx, 0], X_target_tsne[target_idx, 1], 
                              color=target_colors[i], marker='x', alpha=0.7, label=f'Target-Class{cls}')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"t-SNE plot saved to: {save_path}")
    
    plt.show()
    plt.close()

def visualize_feature_histograms(X_source, X_target, X_target_aligned=None, feature_names=None, 
                                n_features_to_plot=None, title='Feature Distribution Comparison', save_path=None):
    """Plot histograms for each feature to compare distributions"""
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(X_source.shape[1])]
    
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
            # 扩展特征名列表
            feature_names = feature_names + [f'Feature {i+1+len(feature_names)}' for i in range(X_source.shape[1] - len(feature_names))]
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
    
    if X_target_aligned is not None:
        # Calculate KL divergence and Wasserstein distance for each feature
        _, kl_div_before = calculate_kl_divergence(X_source, X_target)
        _, kl_div_after = calculate_kl_divergence(X_source, X_target_aligned)
        
        _, wasserstein_before = calculate_wasserstein_distances(X_source, X_target)
        _, wasserstein_after = calculate_wasserstein_distances(X_source, X_target_aligned)
        
        # Sort features by improvement in KL divergence
        kl_improvement = {}
        improvement_percentages = {}
        for i in range(X_source.shape[1]):
            feature_key = f'feature_{i}'
            if feature_key in kl_div_before and feature_key in kl_div_after:
                kl_improvement[i] = kl_div_before[feature_key] - kl_div_after[feature_key]
                if kl_div_before[feature_key] > 0:
                    improvement_percentages[i] = (kl_div_before[feature_key] - kl_div_after[feature_key]) / kl_div_before[feature_key] * 100
                else:
                    improvement_percentages[i] = 0
        
        # If plotting all features, adjust the layout
        if n_features == X_source.shape[1]:
            # For all features, sort by initial discrepancy (descending)
            kl_values = [(i, kl_div_before[f'feature_{i}']) for i in range(X_source.shape[1])]
            selected_features = [idx for idx, _ in sorted(kl_values, key=lambda x: x[1], reverse=True)]
            
            # Create a comprehensive feature statistics table
            stats_fig = plt.figure(figsize=(16, 8))
            ax_table = stats_fig.add_subplot(111)
            ax_table.axis('off')
            
            table_data = []
            table_columns = ['Feature', 'KL Before', 'KL After', 'KL Imp. %', 
                            'Wass. Before', 'Wass. After', 'Wass. Imp. %', 
                            'Initial Shift', 'Aligned Shift']
            
            for i in range(X_source.shape[1]):
                feature_key = f'feature_{i}'
                # 直接使用传入的特征名称，确保特征名称正确显示
                feature_name = feature_names[i]
                
                kl_before = kl_div_before[feature_key]
                kl_after = kl_div_after[feature_key]
                wass_before = wasserstein_before[feature_key]
                wass_after = wasserstein_after[feature_key]
                
                kl_imp = (kl_before - kl_after) / kl_before * 100 if kl_before > 0 else 0
                wass_imp = (wass_before - wass_after) / wass_before * 100 if wass_before > 0 else 0
                
                # Determine initial severity level based on thresholds
                if kl_before > KL_SEVERE_THRESHOLD or wass_before > WASS_SEVERE_THRESHOLD:
                    initial_severity = "HIGH"
                elif kl_before > KL_MODERATE_THRESHOLD or wass_before > WASS_MODERATE_THRESHOLD:
                    initial_severity = "MEDIUM"
                else:
                    initial_severity = "LOW"
                
                # Determine aligned severity level based on after-alignment metrics
                if kl_after > KL_SEVERE_THRESHOLD or wass_after > WASS_SEVERE_THRESHOLD:
                    aligned_severity = "HIGH"
                elif kl_after > KL_MODERATE_THRESHOLD or wass_after > WASS_MODERATE_THRESHOLD:
                    aligned_severity = "MEDIUM"
                else:
                    aligned_severity = "LOW"
                
                # Add row with feature statistics
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
            
            # Create and style the table with improved formatting
            colWidths = [0.18, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.11, 0.11]
            table = ax_table.table(
                cellText=table_data,
                colLabels=table_columns,
                loc='center',
                cellLoc='center',
                colWidths=colWidths
            )
            
            # 添加日志，记录表格中包含的特征信息
            logging.info(f"特征统计表包含了{len(table_data)}个特征：前5个为 {[row[0] for row in table_data[:5]]}...")
            
            # Improve table formatting
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.0, 1.5)  # Adjust table scale for better proportion
            
            # Adjust header row style
            for i, col_label in enumerate(table_columns):
                table[0, i].set_facecolor('#4a69bd')  # Header background color
                table[0, i].set_text_props(color='white', fontweight='bold')
            
            # Set colors based on improvement and severity
            for i, row in enumerate(table_data):
                kl_imp = float(row[3].strip('%'))
                wass_imp = float(row[6].strip('%'))
                initial_severity = row[7]
                aligned_severity = row[8]
                
                kl_before_val = float(row[1])
                kl_after_val = float(row[2])
                wass_before_val = float(row[4])
                wass_after_val = float(row[5])
                
                # Color the initial severity cell
                if initial_severity == "HIGH":
                    table[(i+1, 7)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 7)].set_text_props(weight='bold', color='white')
                elif initial_severity == "MEDIUM":
                    table[(i+1, 7)].set_facecolor(MODERATE_COLOR)
                    table[(i+1, 7)].set_text_props(weight='bold')
                else:
                    table[(i+1, 7)].set_facecolor(LOW_COLOR)
                    table[(i+1, 7)].set_text_props(weight='bold', color='white')
                
                # Color the aligned severity cell
                if aligned_severity == "HIGH":
                    table[(i+1, 8)].set_facecolor(SEVERE_COLOR)
                    table[(i+1, 8)].set_text_props(weight='bold', color='white')
                elif aligned_severity == "MEDIUM":
                    table[(i+1, 8)].set_facecolor(MODERATE_COLOR)
                    table[(i+1, 8)].set_text_props(weight='bold')
                else:
                    table[(i+1, 8)].set_facecolor(LOW_COLOR)
                    table[(i+1, 8)].set_text_props(weight='bold', color='white')
                
                # Style the feature name column
                table[(i+1, 0)].set_text_props(ha='left', fontweight='bold')
                table[(i+1, 0)].set_facecolor('#f7f1e3')  # Light background for feature names
                
                # Set KL improvement cell color
                if kl_imp > 50:
                    table[(i+1, 3)].set_facecolor('#26de81')  # Bright green for major improvement
                    table[(i+1, 3)].set_text_props(weight='bold')
                elif kl_imp > 20:
                    table[(i+1, 3)].set_facecolor('#c6efce')  # Light green for good improvement
                elif kl_imp < 0:
                    table[(i+1, 3)].set_facecolor('#ffc7ce')  # Light red for deterioration
                
                # Set Wasserstein improvement cell color
                if wass_imp > 50:
                    table[(i+1, 6)].set_facecolor('#26de81')  # Bright green
                    table[(i+1, 6)].set_text_props(weight='bold')
                elif wass_imp > 20:
                    table[(i+1, 6)].set_facecolor('#c6efce')  # Light green
                elif wass_imp < 0:
                    table[(i+1, 6)].set_facecolor('#ffc7ce')  # Light red
                
                # Color code initial KL and Wasserstein values based on thresholds
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
                
                # Color code aligned KL and Wasserstein values based on thresholds
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
            
            # Add table title
            plt.title('Feature Distribution Shift and Alignment Metrics', fontsize=14, fontweight='bold')
            
            # Add a legend/explanation of thresholds
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
                stats_path = save_path.replace('.png', '_stats_table.png')
                plt.savefig(stats_path, dpi=300, bbox_inches='tight')
                logging.info(f"Feature statistics table saved to: {stats_path}")
            
            plt.close()
        
        # Select top N features by improvement or initial discrepancy
        if n_features < X_source.shape[1]:
            # Top features by improvement
            top_features = sorted(kl_improvement.items(), key=lambda x: x[1], reverse=True)[:n_features]
            selected_features = [idx for idx, _ in top_features]
        else:
            # All features sorted by initial KL divergence
            kl_values = [(i, kl_div_before[f'feature_{i}']) for i in range(X_source.shape[1])]
            selected_features = [idx for idx, _ in sorted(kl_values, key=lambda x: x[1], reverse=True)][:n_features]
        
        # Create histograms for selected features
        fig, axes = plt.subplots(n_features, 2, figsize=(18, 4*n_features))
        
        for i, feature_idx in enumerate(selected_features):
            # 直接使用传入的特征名称
            feature_name = feature_names[feature_idx]
            
            # Before alignment histogram
            ax1 = axes[i, 0] if n_features > 1 else axes[0]
            sns.histplot(X_source[:, feature_idx], kde=True, ax=ax1, color='blue', alpha=0.5, label='Source')
            sns.histplot(X_target[:, feature_idx], kde=True, ax=ax1, color='red', alpha=0.5, label='Target')
            
            # Add KL and Wasserstein statistics
            feature_key = f'feature_{feature_idx}'
            kl_before = kl_div_before[feature_key]
            wass_before = wasserstein_before[feature_key]
            
            # Determine severity level and box color
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
            
            # Set title with severity indicator
            ax1.set_title(f'Before Alignment - {feature_name} ({severity_level})', 
                         color=text_color, fontweight='bold', 
                         bbox=dict(facecolor=box_color, edgecolor='none', pad=3))
            
            ax1.legend()
            
            # Add distribution divergence metrics
            ax1.text(0.05, 0.95, 
                    f'KL: {kl_before:.4f}\nWass: {wass_before:.4f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # After alignment histogram
            ax2 = axes[i, 1] if n_features > 1 else axes[1]
            sns.histplot(X_source[:, feature_idx], kde=True, ax=ax2, color='blue', alpha=0.5, label='Source')
            sns.histplot(X_target_aligned[:, feature_idx], kde=True, ax=ax2, color='red', alpha=0.5, label='Target (Aligned)')
            ax2.set_title(f'After CORAL Alignment - {feature_name}')
            ax2.legend()
            
            # Add KL and Wasserstein statistics for after alignment
            kl_after = kl_div_after[feature_key]
            wass_after = wasserstein_after[feature_key]
            
            # Calculate improvement percentage
            kl_imp_pct = (kl_before - kl_after) / kl_before * 100 if kl_before > 0 else 0
            wass_imp_pct = (wass_before - wass_after) / wass_before * 100 if wass_before > 0 else 0
            
            # Set box color based on improvement
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
            
            ax2.text(0.05, 0.95, 
                    f'KL: {kl_after:.4f} (↓{kl_imp_pct:.1f}%)\nWass: {wass_after:.4f} (↓{wass_imp_pct:.1f}%)\n{imp_text}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=imp_box_color, alpha=0.7))
    else:
        # Only source and target domains without alignment
        _, kl_div = calculate_kl_divergence(X_source, X_target)
        _, wasserstein_dist = calculate_wasserstein_distances(X_source, X_target)
        
        # Sort features by KL divergence
        kl_values = [(i, kl_div[f'feature_{i}']) for i in range(X_source.shape[1])]
        selected_features = [idx for idx, _ in sorted(kl_values, key=lambda x: x[1], reverse=True)][:n_features]
        
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
        if n_features == 1:
            axes = [axes]
        
        for i, feature_idx in enumerate(selected_features):
            feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f'Feature {feature_idx+1}'
            
            ax = axes[i]
            sns.histplot(X_source[:, feature_idx], kde=True, ax=ax, color='blue', alpha=0.5, label='Source')
            sns.histplot(X_target[:, feature_idx], kde=True, ax=ax, color='red', alpha=0.5, label='Target')
            
            # Add distribution divergence metrics
            feature_key = f'feature_{feature_idx}'
            kl_val = kl_div[feature_key]
            wass_val = wasserstein_dist[feature_key]
            
            # Classify distribution mismatch severity
            if kl_val > KL_SEVERE_THRESHOLD or wass_val > WASS_SEVERE_THRESHOLD:
                severity_text = "HIGH Distribution Shift"
                box_color = SEVERE_COLOR
                text_color = 'white'
            elif kl_val > KL_MODERATE_THRESHOLD or wass_val > WASS_MODERATE_THRESHOLD:
                severity_text = "MEDIUM Distribution Shift"
                box_color = MODERATE_COLOR
                text_color = 'black'
            else:
                severity_text = "LOW Distribution Shift"
                box_color = LOW_COLOR
                text_color = 'white'
            
            # Set title with severity indicator
            ax.set_title(f'{feature_name} ({severity_text})', 
                       color=text_color, fontweight='bold', 
                       bbox=dict(facecolor=box_color, edgecolor='none', pad=3))
            
            ax.legend()
            
            ax.text(0.05, 0.95, 
                   f'KL: {kl_val:.4f}\nWass: {wass_val:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Add legend for distribution shift thresholds
    legend_text = (
        f"Distribution Shift Thresholds - HIGH: KL > {KL_SEVERE_THRESHOLD} or Wass > {WASS_SEVERE_THRESHOLD} | "
        f"MEDIUM: KL > {KL_MODERATE_THRESHOLD} or Wass > {WASS_MODERATE_THRESHOLD} | LOW: Otherwise"
    )
    plt.figtext(0.5, 0.01, legend_text, ha="center", fontsize=9, 
              bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Feature histograms saved to: {save_path}")
    
    plt.close()
    
    # Create a summary visualization of all features' distribution shifts
    if X_source.shape[1] > n_features:
        all_features_fig = plt.figure(figsize=(14, 8))
        
        # Create a heatmap of all features' KL and Wasserstein distances
        metrics_data = []
        feature_labels = []
        
        for i in range(X_source.shape[1]):
            feature_key = f'feature_{i}'
            feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i+1}'
            feature_labels.append(feature_name)
            
            if X_target_aligned is not None:
                kl_before = kl_div_before[feature_key]
                kl_after = kl_div_after[feature_key]
                wass_before = wasserstein_before[feature_key]
                wass_after = wasserstein_after[feature_key]
                metrics_data.append([kl_before, kl_after, wass_before, wass_after])
            else:
                kl_val = kl_div[feature_key]
                wass_val = wasserstein_dist[feature_key]
                metrics_data.append([kl_val, wass_val])
        
        # Convert to DataFrame for easier manipulation
        if X_target_aligned is not None:
            df_metrics = pd.DataFrame(metrics_data, 
                              columns=['KL Before', 'KL After', 'Wasserstein Before', 'Wasserstein After'], 
                              index=feature_labels)
            
            # Create a subplot for KL divergence before/after
            plt.subplot(1, 2, 1)
            kl_data = df_metrics[['KL Before', 'KL After']].sort_values('KL Before', ascending=False)
            
            # Bar plot of KL divergence
            ax1 = kl_data.plot(kind='bar', figsize=(10, 6), ax=plt.gca())
            plt.title('KL Divergence Before vs After Alignment')
            plt.ylabel('KL Divergence')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            
            # Add threshold lines
            ax1.axhline(y=KL_SEVERE_THRESHOLD, color='red', linestyle='--', alpha=0.7)
            ax1.axhline(y=KL_MODERATE_THRESHOLD, color='orange', linestyle='--', alpha=0.7)
            ax1.text(0, KL_SEVERE_THRESHOLD, f'High Shift Threshold ({KL_SEVERE_THRESHOLD})', 
                   verticalalignment='bottom', color='red')
            ax1.text(0, KL_MODERATE_THRESHOLD, f'Medium Shift Threshold ({KL_MODERATE_THRESHOLD})', 
                   verticalalignment='bottom', color='orange')
            
            # Create a subplot for Wasserstein distance before/after
            plt.subplot(1, 2, 2)
            wass_data = df_metrics[['Wasserstein Before', 'Wasserstein After']].sort_values('Wasserstein Before', ascending=False)
            
            # Bar plot of Wasserstein distance
            ax2 = wass_data.plot(kind='bar', figsize=(10, 6), ax=plt.gca())
            plt.title('Wasserstein Distance Before vs After Alignment')
            plt.ylabel('Wasserstein Distance')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            
            # Add threshold lines
            ax2.axhline(y=WASS_SEVERE_THRESHOLD, color='red', linestyle='--', alpha=0.7)
            ax2.axhline(y=WASS_MODERATE_THRESHOLD, color='orange', linestyle='--', alpha=0.7)
            ax2.text(0, WASS_SEVERE_THRESHOLD, f'High Shift Threshold ({WASS_SEVERE_THRESHOLD})', 
                   verticalalignment='bottom', color='red')
            ax2.text(0, WASS_MODERATE_THRESHOLD, f'Medium Shift Threshold ({WASS_MODERATE_THRESHOLD})', 
                   verticalalignment='bottom', color='orange')
            
        else:
            df_metrics = pd.DataFrame(metrics_data, 
                              columns=['KL Divergence', 'Wasserstein Distance'], 
                              index=feature_labels)
            
            # Sort by KL divergence
            df_metrics = df_metrics.sort_values('KL Divergence', ascending=False)
            
            # Create a bar plot
            ax = df_metrics.plot(kind='bar', figsize=(12, 6), ax=plt.gca())
            plt.title('Distribution Metrics Across All Features')
            plt.ylabel('Metric Value')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            
            # Add threshold lines
            ax.axhline(y=KL_SEVERE_THRESHOLD, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=KL_MODERATE_THRESHOLD, color='orange', linestyle='--', alpha=0.7)
            ax.text(0, KL_SEVERE_THRESHOLD, f'High Shift Threshold (KL={KL_SEVERE_THRESHOLD})', 
                   verticalalignment='bottom', color='red')
            ax.text(0, KL_MODERATE_THRESHOLD, f'Medium Shift Threshold (KL={KL_MODERATE_THRESHOLD})', 
                   verticalalignment='bottom', color='orange')
            
            # Add Wasserstein thresholds on the right y-axis
            ax2 = ax.twinx()
            ax2.set_ylabel('Wasserstein Distance Thresholds')
            ax2.axhline(y=WASS_SEVERE_THRESHOLD, color='red', linestyle=':', alpha=0.7)
            ax2.axhline(y=WASS_MODERATE_THRESHOLD, color='orange', linestyle=':', alpha=0.7)
            ax2.text(len(df_metrics)-1, WASS_SEVERE_THRESHOLD, f'High (Wass={WASS_SEVERE_THRESHOLD})', 
                    verticalalignment='bottom', horizontalalignment='right', color='red')
            ax2.text(len(df_metrics)-1, WASS_MODERATE_THRESHOLD, f'Medium (Wass={WASS_MODERATE_THRESHOLD})', 
                    verticalalignment='bottom', horizontalalignment='right', color='orange')
        
        plt.suptitle('All Features Distribution Shift Summary', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            summary_path = save_path.replace('.png', '_all_features_summary.png')
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            logging.info(f"All features summary saved to: {summary_path}")
        
        plt.close()

def compare_before_after_adaptation(source_features, target_features, adapted_target_features, 
                                   source_labels=None, target_labels=None, save_dir=None):
    """Compare feature distributions before and after domain adaptation"""
    # Calculate domain discrepancy metrics
    metrics_before = compute_domain_discrepancy(source_features, target_features)
    metrics_after = compute_domain_discrepancy(source_features, adapted_target_features)
    
    # Visualization
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        tsne_path = os.path.join(save_dir, "tsne_comparison.png")
        hist_path = os.path.join(save_dir, "histogram_comparison.png")
    else:
        tsne_path = hist_path = None
    
    # t-SNE visualization
    visualize_tsne(
        X_source=source_features, 
        X_target=target_features,
        y_source=source_labels,
        y_target=target_labels,
        X_target_aligned=adapted_target_features,
        title="CORAL Domain Adaptation Comparison",
        save_path=tsne_path
    )
    
    # Feature histograms visualization
    visualize_feature_histograms(
        X_source=source_features,
        X_target=target_features,
        X_target_aligned=adapted_target_features,
        n_features_to_plot=None,  # Plot all features
        title="Feature Distribution Comparison",
        save_path=hist_path
    )
    
    # Print domain discrepancy metrics
    print("====== Domain Discrepancy Metrics Comparison ======")
    for metric in ['mmd', 'wasserstein_distance', 'kl_divergence', 'covariance_difference']:
        before_val = metrics_before[metric]
        after_val = metrics_after[metric]
        reduction = (before_val - after_val) / before_val * 100 if before_val > 0 else 0
        
        print(f"{metric}: {before_val:.4f} -> {after_val:.4f} (reduction {reduction:.2f}%)")
    
    return {
        'before': metrics_before,
        'after': metrics_after
    }

def load_and_preprocess_data(source_features_path, target_features_path, source_labels_path=None, target_labels_path=None):
    """加载并预处理特征和标签数据"""
    source_features = np.load(source_features_path)
    target_features = np.load(target_features_path)
    
    source_labels = np.load(source_labels_path) if source_labels_path else None
    target_labels = np.load(target_labels_path) if target_labels_path else None
    
    return source_features, target_features, source_labels, target_labels

# 主函数
if __name__ == "__main__":
    setup_logger()
    
    # 示例代码
    import argparse
    
    parser = argparse.ArgumentParser(description="CORAL域适应可视化分析工具")
    parser.add_argument("--source", required=True, help="源域特征NPZ文件路径")
    parser.add_argument("--target", required=True, help="目标域特征NPZ文件路径")
    parser.add_argument("--aligned", required=True, help="对齐后的特征NPZ文件路径")
    parser.add_argument("--source-labels", help="源域标签NPZ文件路径")
    parser.add_argument("--target-labels", help="目标域标签NPZ文件路径")
    parser.add_argument("--output", help="输出目录", default="./visualization_output")
    
    args = parser.parse_args()
    
    # 加载数据
    source_features, target_features, source_labels, target_labels = load_and_preprocess_data(
        args.source, args.target, args.source_labels, args.target_labels
    )
    
    # 加载对齐后特征
    adapted_features = np.load(args.aligned)
    
    # 比较域适应前后的结果
    compare_before_after_adaptation(
        source_features, 
        target_features, 
        adapted_features,
        source_labels,
        target_labels,
        args.output
    ) 