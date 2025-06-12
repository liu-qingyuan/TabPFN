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
from scipy.stats import entropy # Added for KL divergence
from scipy.spatial.distance import cdist # Added for distance calculations
from scipy.stats import wasserstein_distance # Added for Wasserstein distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from torch import cuda  # 假设已安装
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

# 导入可视化模块
from visualize_analytical_CORAL_tsne import (
    visualize_tsne,
    visualize_feature_histograms
    # compute_domain_discrepancy, # Removed as per Linter (unused)
    # compare_before_after_adaptation # Removed as per Linter (unused)
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
    print(f"{dataset_name}准确率 (Accuracy): {metrics['acc']:.4f}")
    print(f"{dataset_name} AUC: {metrics['auc']:.4f}")
    print(f"{dataset_name} F1分数: {metrics['f1']:.4f}")
    print(f"{dataset_name}类别0准确率: {metrics['acc_0']:.4f}")
    print(f"{dataset_name}类别1准确率: {metrics['acc_1']:.4f}")

# 修改后的CORAL变换函数，区分类别特征和连续特征
def coral_transform(Xs: np.ndarray, Xt: np.ndarray, cat_idx: list) -> np.ndarray:
    """
    解析版CORAL变换，直接计算协方差变换矩阵
    将目标域特征对齐到源域协方差框架下
    仅对连续特征进行变换，保留类别特征不变
    
    参数:
    - Xs: 源域特征 [n_samples_source, n_features]
    - Xt: 目标域特征 [n_samples_target, n_features]
    - cat_idx: 类别特征的索引列表
    
    返回:
    - Xt_aligned: 对齐后的目标域特征
    """
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
    
    # 计算连续特征的协方差矩阵，添加小的对角线正则化以确保矩阵可逆
    Cs = np.cov(Xs_cont, rowvar=False) + 1e-5*np.eye(len(cont_idx))
    Ct = np.cov(Xt_cont_centered, rowvar=False) + 1e-5*np.eye(len(cont_idx))
    
    # 矩阵平方根 - 目标域到源域的变换
    Ct_inv_sqrt = scipy.linalg.fractional_matrix_power(Ct, -0.5)
    Cs_sqrt = scipy.linalg.fractional_matrix_power(Cs, 0.5)
    
    # 计算转换矩阵 - 先漂白目标域，再上色为源域（仅应用于连续特征）
    A = np.dot(Ct_inv_sqrt, Cs_sqrt)  # 线性映射矩阵
    Xt_cont_aligned = np.dot(Xt_cont_centered, A)
    
    # 将变换后的连续特征与原始类别特征合并
    Xt_aligned = Xt.copy()
    Xt_aligned[:, cont_idx] = Xt_cont_aligned
    
    # 记录特征分布变化
    logging.info(f"连续特征变换前均值差异: {np.mean(np.abs(np.mean(Xs_cont, axis=0) - np.mean(Xt_cont, axis=0))):.6f}")
    logging.info(f"连续特征变换后均值差异: {np.mean(np.abs(np.mean(Xs_cont, axis=0) - np.mean(Xt_cont_aligned, axis=0))):.6f}")
    logging.info(f"连续特征变换前标准差差异: {np.mean(np.abs(np.std(Xs_cont, axis=0) - np.std(Xt_cont, axis=0))):.6f}")
    logging.info(f"连续特征变换后标准差差异: {np.mean(np.abs(np.std(Xs_cont, axis=0) - np.std(Xt_cont_aligned, axis=0))):.6f}")
    
    # 检查类别特征是否保持不变
    if not np.array_equal(Xt[:, cat_idx], Xt_aligned[:, cat_idx]):
        logging.error("错误：类别特征在变换过程中被改变")
    else:
        logging.info("验证成功：类别特征在CORAL变换过程中保持不变")
    
    return Xt_aligned

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

def run_coral_adaptation_experiment(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    cat_idx: list, # Added cat_idx parameter
    model_name: str = 'TabPFN-CORAL',
    tabpfn_params: dict = None, # Modified default
    base_path: str = './results_analytical_coral_A2B', # Updated base_path
    optimize_decision_threshold: bool = True,
    feature_names_for_plot: list = None # Added for histogram feature names
):
    """
    运行带有解析版CORAL域适应的TabPFN实验
    
    参数:
    - X_source: 源域特征
    - y_source: 源域标签
    - X_target: 目标域特征
    - y_target: 目标域标签
    - cat_idx: 类别特征索引列表
    - model_name: 模型名称
    - tabpfn_params: TabPFN参数
    - base_path: 结果保存路径
    - optimize_decision_threshold: 是否优化决策阈值
    - feature_names_for_plot: 特征名称列表，用于绘图
    
    返回:
    - 评估指标
    """
    if tabpfn_params is None: # Initialize default here
        tabpfn_params = {'device': 'cuda', 'max_time': 60, 'random_state': 42}

    logging.info(f"\\n=== {model_name} Model (Analytical CORAL) ===")
    
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 数据标准化
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # 分析源域和目标域的特征分布差异
    logging.info("Analyzing domain differences before alignment...")
    source_mean = np.mean(X_source_scaled, axis=0)
    target_mean = np.mean(X_target_scaled.astype(float), axis=0) # Ensure float for mean
    source_std = np.std(X_source_scaled, axis=0)
    target_std = np.std(X_target_scaled.astype(float), axis=0) # Ensure float for std
    
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
    logging.info("\\nEvaluating TabPFN on source domain validation set...")
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
    logging.info("\\nEvaluating TabPFN directly on target domain (without alignment)...")
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
    logging.info("\\nApplying analytical CORAL transformation...")
    start_time = time.time()
    X_target_aligned = coral_transform(X_source_scaled, X_target_scaled, cat_idx) # Pass cat_idx
    align_time = time.time() - start_time
    logging.info(f"CORAL transformation completed in {align_time:.2f} seconds")
    
    # 分析对齐前后的特征差异
    mean_diff_after = np.mean(np.abs(np.mean(X_source_scaled, axis=0) - np.mean(X_target_aligned, axis=0)))
    std_diff_after = np.mean(np.abs(np.std(X_source_scaled, axis=0) - np.std(X_target_aligned, axis=0)))
    logging.info(f"After alignment: Mean diff={mean_diff_after:.6f}, Std diff={std_diff_after:.6f}")
    if mean_diff > 0: # Avoid division by zero
        logging.info(f"Difference reduction: Mean: {(mean_diff-mean_diff_after)/mean_diff:.2%}, Std: {(std_diff-std_diff_after)/std_diff:.2%}")
    
    # 在目标域上进行评估
    logging.info("\\nEvaluating model on target domain (with analytical CORAL alignment)...")
    
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
    
    optimal_threshold = 0.5 # Default in case not optimized
    # 优化决策阈值（可选）
    if optimize_decision_threshold:
        logging.info("\\nOptimizing decision threshold using Youden index...")
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
    logging.info("\\nSource Domain Validation Results:")
    logging.info(f"Accuracy: {source_metrics['acc']:.4f}")
    logging.info(f"AUC: {source_metrics['auc']:.4f}")
    logging.info(f"F1: {source_metrics['f1']:.4f}")
    
    logging.info("\\nTarget Domain Evaluation Results (with Analytical CORAL):")
    logging.info(f"Accuracy: {target_metrics['acc']:.4f}")
    logging.info(f"AUC: {target_metrics['auc']:.4f}")
    logging.info(f"F1: {target_metrics['f1']:.4f}")
    logging.info(f"Class 0 Accuracy: {target_metrics['acc_0']:.4f}")
    logging.info(f"Class 1 Accuracy: {target_metrics['acc_1']:.4f}")
    logging.info(f"Inference Time: {inference_time:.4f} seconds")
    
    # 比较对齐前后的性能
    logging.info("\\nPerformance Improvement with Analytical CORAL Alignment:")
    logging.info(f"Accuracy: {direct_metrics['acc']:.4f} -> {target_metrics['acc']:.4f} ({target_metrics['acc']-direct_metrics['acc']:.4f})")
    logging.info(f"AUC: {direct_metrics['auc']:.4f} -> {target_metrics['auc']:.4f} ({target_metrics['auc']-direct_metrics['auc']:.4f})")
    logging.info(f"F1: {direct_metrics['f1']:.4f} -> {target_metrics['f1']:.4f} ({target_metrics['f1']-direct_metrics['f1']:.4f})")
    logging.info(f"Class 0 Accuracy: {direct_metrics['acc_0']:.4f} -> {target_metrics['acc_0']:.4f} ({target_metrics['acc_0']-direct_metrics['acc_0']:.4f})")
    logging.info(f"Class 1 Accuracy: {direct_metrics['acc_1']:.4f} -> {target_metrics['acc_1']:.4f} ({target_metrics['acc_1']-direct_metrics['acc_1']:.4f})")
    
    # 保存对齐后的特征
    aligned_features_path = f"{base_path}/{model_name}_aligned_features.npz"
    np.savez(aligned_features_path, 
             X_source=X_source_scaled, 
             X_target=X_target_scaled.astype(float), # Ensure float for savez
             X_target_aligned=X_target_aligned)
    logging.info(f"Aligned features saved to: {aligned_features_path}")
    
    # 使用导入的可视化模块进行t-SNE可视化
    logging.info("\\n使用t-SNE可视化CORAL对齐前后的分布...")
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
    logging.info("\\n绘制对齐前后的特征分布直方图...")
    hist_save_path = f"{base_path}/{model_name}_histograms.png"
    visualize_feature_histograms(
        X_source=X_source_scaled,
        X_target=X_target_scaled,
        X_target_aligned=X_target_aligned,
        feature_names=feature_names_for_plot,  # Use passed feature names
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
    labels_plot = ['Direct', 'With CORAL'] # Renamed to avoid conflict
    class0_accs = [direct_metrics['acc_0'], target_metrics['acc_0']]
    class1_accs = [direct_metrics['acc_1'], target_metrics['acc_1']]
    
    x_plot = np.arange(len(labels_plot)) # Renamed to avoid conflict
    width = 0.35
    
    plt.bar(x_plot - width/2, class0_accs, width, label='Class 0')
    plt.bar(x_plot + width/2, class1_accs, width, label='Class 1')
    
    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy Comparison')
    plt.xticks(x_plot, labels_plot)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/{model_name}_analysis.png", dpi=300)
    plt.close()
    
    # 如果使用了阈值优化，添加ROC曲线和阈值点
    if optimize_decision_threshold:
        plt.figure(figsize=(8, 6))
        fpr, tpr, thresholds_roc = roc_curve(y_target, y_target_proba[:, 1]) # Renamed thresholds
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {target_metrics["auc"]:.4f})')
        
        # 标出最佳阈值点
        # Ensure optimal_threshold is defined from optimize_threshold or has a default
        optimal_idx_roc = np.where(thresholds_roc >= optimal_threshold)[0]
        if len(optimal_idx_roc) > 0:
             optimal_idx_roc = optimal_idx_roc[-1]
             plt.scatter(fpr[optimal_idx_roc], tpr[optimal_idx_roc], color='red', 
                        label=f'Optimal Threshold = {optimal_threshold:.4f}')
        
        # 标出默认阈值0.5对应的点
        default_idx_roc = None # Renamed
        for i, t_roc in enumerate(thresholds_roc): # Renamed loop variable
            if t_roc <= 0.5:
                default_idx_roc = i
                break
        if default_idx_roc is not None:
            plt.scatter(fpr[default_idx_roc], tpr[default_idx_roc], color='blue', 
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
def class_conditional_coral_transform(Xs: np.ndarray, ys: np.ndarray, Xt: np.ndarray, yt_pseudo: np.ndarray = None, cat_idx: list = None, alpha: float = 0.1) -> np.ndarray:
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
    # 如果没有指定类别特征索引, cat_idx must be provided.
    if cat_idx is None:
        # This was a default, but now it must be passed.
        # Consider raising an error or using a globally defined one if that's the design.
        raise ValueError("cat_idx must be provided for class_conditional_coral_transform")

    all_idx = list(range(Xs.shape[1]))
    cont_idx = [i for i in all_idx if i not in cat_idx]
    
    # 获取不同的类别
    classes = np.unique(ys)
    n_classes = len(classes)
    
    logging.info(f"执行类条件CORAL对齐，共有{n_classes}个类别")
    
    # 如果目标域没有伪标签，则使用普通CORAL先对齐，然后用源域模型预测
    if yt_pseudo is None:
        logging.info("目标域没有提供伪标签，先使用普通CORAL进行对齐，再用源域模型预测伪标签")
        Xt_temp = coral_transform(Xs, Xt, cat_idx)  # Pass cat_idx
        
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
            # Xt_aligned[class_mask] = Xt_c # Keep original if not enough samples for this class
            continue
            
        if len(Xs_c) < 2:  # 需要至少2个样本才能计算协方差
            logging.warning(f"类别{c}在源域中样本数量过少({len(Xs_c)}个)，无法进行类条件CORAL对齐，跳过")
            # For target samples of this class, we can't align. What to do?
            # One option: use the overall CORAL transformation for these.
            # Another: leave them as is. For now, leave as is (by continuing).
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
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    cat_idx: list, # Added cat_idx
    model_name: str = 'TabPFN-ClassCORAL',
    tabpfn_params: dict = None, # Modified default
    base_path: str = './results_class_conditional_coral_A2B', # Updated base_path
    optimize_decision_threshold: bool = True,
    alpha: float = 0.1,
    use_target_labels: bool = False,  # 是否使用部分真实标签，False则使用伪标签
    target_label_ratio: float = 0.1,    # 如果使用真实标签，从目标域取多少比例
    feature_names_for_plot: list = None # Added for histogram feature names
):
    """
    运行带有类条件CORAL域适应的TabPFN实验
    
    参数:
    - X_source: 源域特征
    - y_source: 源域标签
    - X_target: 目标域特征
    - y_target: 目标域标签
    - cat_idx: 类别特征索引
    - model_name: 模型名称
    - tabpfn_params: TabPFN参数
    - base_path: 结果保存路径
    - optimize_decision_threshold: 是否优化决策阈值
    - alpha: 类条件CORAL正则化参数
    - use_target_labels: 是否使用部分目标域真实标签（而非伪标签）
    - target_label_ratio: 使用多少比例的目标域标签
    - feature_names_for_plot: 特征名称列表，用于绘图
    
    返回:
    - 评估指标
    """
    if tabpfn_params is None: # Initialize default here
        tabpfn_params = {'device': 'cuda', 'max_time': 60, 'random_state': 42}

    logging.info(f"\\n=== {model_name} Model (Class-Conditional CORAL) ===")
    
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
    logging.info("\\nEvaluating TabPFN on source domain validation set...")
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
    logging.info("\\nEvaluating TabPFN directly on target domain (without alignment)...")
    y_target_pred_direct = tabpfn_model.predict(X_target_scaled)
    y_target_proba_direct = tabpfn_model.predict_proba(X_target_scaled)
    
    # 分析预测分布
    unique_labels, counts = np.unique(y_target_pred_direct, return_counts=True)
    logging.info(f"Direct prediction distribution: {dict(zip(unique_labels, counts))}")
    
    # 计算直接预测指标
    direct_metrics = evaluate_metrics(y_target, y_target_pred_direct, y_target_proba_direct[:, 1])
    logging.info(f"Direct prediction - Accuracy: {direct_metrics['acc']:.4f}, AUC: {direct_metrics['auc']:.4f}, F1: {direct_metrics['f1']:.4f}")
    logging.info(f"Direct prediction - Class 0 Acc: {direct_metrics['acc_0']:.4f}, Class 1 Acc: {direct_metrics['acc_1']:.4f}")
    
    labeled_idx = np.array([]) # Initialize to handle unbound case
    # 准备类条件CORAL目标域标签/伪标签
    if use_target_labels:
        # 如果使用部分真实标签
        logging.info(f"\\nUsing {target_label_ratio:.1%} of target domain true labels for class-conditional CORAL alignment...")
        # n_labeled = int(len(y_target) * target_label_ratio) # Unused variable
        
        # 进行分层抽样，确保每个类别都有样本
        # Ensure there are enough samples for split, otherwise this might fail or give unexpected results
        if len(np.unique(y_target)) > 1 and len(y_target) * (1-target_label_ratio) >= len(np.unique(y_target)):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1-target_label_ratio, random_state=42)
            for train_idx, _ in sss.split(X_target_scaled, y_target): # train_idx here is the labeled_idx
                labeled_idx = train_idx
        else: # Fallback if stratification is not possible (e.g. too few samples or single class)
            logging.warning("StratifiedShuffleSplit not possible, using simple random sampling for labeled set.")
            n_labeled_samples = int(len(y_target) * target_label_ratio)
            labeled_idx = np.random.choice(len(y_target), n_labeled_samples, replace=False)


        # 创建伪标签（部分是真实标签，部分是普通CORAL预测的）
        yt_pseudo = np.zeros_like(y_target) - 1  # 初始化为-1表示未知
        if len(labeled_idx) > 0:
            yt_pseudo[labeled_idx] = y_target[labeled_idx]  # 填入已知标签
        
        # 对未标记部分使用普通CORAL预测
        # 先使用普通CORAL对齐未标记部分
        unlabeled_mask = (yt_pseudo == -1)
        if np.any(unlabeled_mask): # only if there are unlabeled samples
            X_target_unlabeled = X_target_scaled[unlabeled_mask]
            if X_target_unlabeled.shape[0] > 0: # Check if there are actually unlabeled samples
                 X_target_unlabeled_aligned = coral_transform(X_source_scaled, X_target_unlabeled, cat_idx) # Pass cat_idx
                 # 对未标记部分进行预测
                 yt_pseudo_unlabeled = tabpfn_model.predict(X_target_unlabeled_aligned)
                 yt_pseudo[unlabeled_mask] = yt_pseudo_unlabeled
            else:
                logging.info("No unlabeled samples to predict after selecting labeled ones.")
        else:
            logging.info("All target samples used as labeled, no pseudo-labeling needed for remaining.")

        
        logging.info(f"Partial true labels + partial pseudo-labels distribution: {np.bincount(yt_pseudo[yt_pseudo != -1]) if np.any(yt_pseudo != -1) else 'No labels/pseudo-labels generated'}")
    else:
        # 使用完全伪标签
        logging.info("\\nGenerating pseudo-labels using standard CORAL for class-conditional CORAL alignment...")
        # 先使用普通CORAL对齐
        X_target_aligned_temp = coral_transform(X_source_scaled, X_target_scaled, cat_idx) # Pass cat_idx
        yt_pseudo = tabpfn_model.predict(X_target_aligned_temp)
        logging.info(f"Generated pseudo-label distribution: {np.bincount(yt_pseudo)}")
    
    # 使用类条件CORAL进行特征对齐
    logging.info("\\nApplying class-conditional CORAL transformation...")
    start_time = time.time()
    X_target_aligned = class_conditional_coral_transform(
        X_source_scaled, y_source, X_target_scaled, yt_pseudo, cat_idx, alpha=alpha
    )
    align_time = time.time() - start_time
    logging.info(f"Class-conditional CORAL transformation completed in {align_time:.2f} seconds")
    
    # 在目标域上进行评估
    logging.info("\\nEvaluating model on target domain (with class-conditional CORAL alignment)...")
    
    # 目标域预测（使用类条件CORAL对齐）
    start_time = time.time()
    y_target_pred = tabpfn_model.predict(X_target_aligned)
    y_target_proba = tabpfn_model.predict_proba(X_target_aligned)
    inference_time = time.time() - start_time
    
    # 分析类条件CORAL对齐后的预测分布
    unique_labels_cc, counts_cc = np.unique(y_target_pred, return_counts=True) # Renamed
    logging.info(f"Class-conditional CORAL aligned prediction distribution: {dict(zip(unique_labels_cc, counts_cc))}")
    
    # 计算目标域指标
    target_metrics = evaluate_metrics(y_target, y_target_pred, y_target_proba[:, 1])
    
    optimal_threshold_cc = 0.5 # Default
    # 优化决策阈值（可选）
    if optimize_decision_threshold:
        logging.info("\\nOptimizing decision threshold using Youden index...")
        optimal_threshold_cc, optimal_metrics_cc = optimize_threshold(y_target, y_target_proba[:, 1]) # Renamed
        
        logging.info(f"Optimal threshold: {optimal_threshold_cc:.4f} (default: 0.5)")
        logging.info(f"Metrics with optimized threshold:")
        logging.info(f"  Accuracy: {optimal_metrics_cc['acc']:.4f} (original: {target_metrics['acc']:.4f})")
        logging.info(f"  F1 Score: {optimal_metrics_cc['f1']:.4f} (original: {target_metrics['f1']:.4f})")
        logging.info(f"  Class 0 Accuracy: {optimal_metrics_cc['acc_0']:.4f} (original: {target_metrics['acc_0']:.4f})")
        logging.info(f"  Class 1 Accuracy: {optimal_metrics_cc['acc_1']:.4f} (original: {target_metrics['acc_1']:.4f})")
        
        # 更新指标
        target_metrics.update({
            'original_acc': target_metrics['acc'],
            'original_f1': target_metrics['f1'],
            'original_acc_0': target_metrics['acc_0'],
            'original_acc_1': target_metrics['acc_1'],
            'optimal_threshold': optimal_threshold_cc
        })
        target_metrics['acc'] = optimal_metrics_cc['acc']
        target_metrics['f1'] = optimal_metrics_cc['f1']
        target_metrics['acc_0'] = optimal_metrics_cc['acc_0']
        target_metrics['acc_1'] = optimal_metrics_cc['acc_1']
    
    # 打印结果
    logging.info("\\nSource Domain Validation Results:")
    logging.info(f"Accuracy: {source_metrics['acc']:.4f}")
    logging.info(f"AUC: {source_metrics['auc']:.4f}")
    logging.info(f"F1: {source_metrics['f1']:.4f}")
    logging.info(f"Class 0 Accuracy: {source_metrics['acc_0']:.4f}")
    logging.info(f"Class 1 Accuracy: {source_metrics['acc_1']:.4f}")
    
    logging.info("\\nTarget Domain Evaluation Results (with Class-Conditional CORAL):")
    logging.info(f"Accuracy: {target_metrics['acc']:.4f}")
    logging.info(f"AUC: {target_metrics['auc']:.4f}")
    logging.info(f"F1: {target_metrics['f1']:.4f}")
    logging.info(f"Class 0 Accuracy: {target_metrics['acc_0']:.4f}")
    logging.info(f"Class 1 Accuracy: {target_metrics['acc_1']:.4f}")
    
    # 比较对齐前后的性能
    logging.info("\\nPerformance Improvement with Class-Conditional CORAL Alignment:")
    logging.info(f"Accuracy: {direct_metrics['acc']:.4f} -> {target_metrics['acc']:.4f} ({target_metrics['acc']-direct_metrics['acc']:.4f})")
    logging.info(f"AUC: {direct_metrics['auc']:.4f} -> {target_metrics['auc']:.4f} ({target_metrics['auc']-direct_metrics['auc']:.4f})")
    logging.info(f"F1: {direct_metrics['f1']:.4f} -> {target_metrics['f1']:.4f} ({target_metrics['f1']-direct_metrics['f1']:.4f})")
    logging.info(f"Class 0 Accuracy: {direct_metrics['acc_0']:.4f} -> {target_metrics['acc_0']:.4f} ({target_metrics['acc_0']-direct_metrics['acc_0']:.4f})")
    logging.info(f"Class 1 Accuracy: {direct_metrics['acc_1']:.4f} -> {target_metrics['acc_1']:.4f} ({target_metrics['acc_1']-direct_metrics['acc_1']:.4f})")
    
    # 保存对齐后的特征
    aligned_features_path_cc = f"{base_path}/{model_name}_aligned_features.npz" # Renamed
    np.savez(aligned_features_path_cc, 
             X_source=X_source_scaled, 
             X_target=X_target_scaled.astype(float), # Ensure float
             X_target_aligned=X_target_aligned,
             yt_pseudo=yt_pseudo)
    logging.info(f"Aligned features saved to: {aligned_features_path_cc}")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    # 绘制整体指标比较
    plt.subplot(1, 3, 1)
    metrics_labels = ['Accuracy', 'AUC', 'F1']
    metrics_values_direct = [direct_metrics['acc'], direct_metrics['auc'], direct_metrics['f1']]
    metrics_values_coral = [target_metrics['acc'], target_metrics['auc'], target_metrics['f1']]
    
    x_plot_cc1 = np.arange(len(metrics_labels)) # Renamed
    width_cc = 0.35 # Renamed
    
    plt.bar(x_plot_cc1 - width_cc/2, metrics_values_direct, width_cc, label='Direct Prediction')
    plt.bar(x_plot_cc1 + width_cc/2, metrics_values_coral, width_cc, label='Class-Conditional CORAL')
    
    plt.ylabel('Score')
    plt.title('Overall Performance Metrics')
    plt.xticks(x_plot_cc1, metrics_labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 绘制每类准确率
    plt.subplot(1, 3, 2)
    class_metrics_names = ['Class 0 Accuracy', 'Class 1 Accuracy'] # Renamed
    class_values_direct = [direct_metrics['acc_0'], direct_metrics['acc_1']]
    class_values_coral = [target_metrics['acc_0'], target_metrics['acc_1']]
    
    x_plot_cc2 = np.arange(len(class_metrics_names)) # Renamed
    
    plt.bar(x_plot_cc2 - width_cc/2, class_values_direct, width_cc, label='Direct Prediction')
    plt.bar(x_plot_cc2 + width_cc/2, class_values_coral, width_cc, label='Class-Conditional CORAL')
    
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(x_plot_cc2, class_metrics_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 绘制预测分布
    plt.subplot(1, 3, 3)
    
    # labels_plot_cc = ['Direct', 'Class-CORAL'] # Unused variable
    pred_dist_direct_cc = np.bincount(y_target_pred_direct) # Renamed
    pred_dist_coral_cc = np.bincount(y_target_pred) # Renamed
    true_dist_cc = np.bincount(y_target) # Renamed
    
    # 确保所有柱状图有相同的类别数
    max_classes_cc = max(len(pred_dist_direct_cc), len(pred_dist_coral_cc), len(true_dist_cc)) # Renamed
    if len(pred_dist_direct_cc) < max_classes_cc:
        pred_dist_direct_cc = np.pad(pred_dist_direct_cc, (0, max_classes_cc - len(pred_dist_direct_cc)))
    if len(pred_dist_coral_cc) < max_classes_cc:
        pred_dist_coral_cc = np.pad(pred_dist_coral_cc, (0, max_classes_cc - len(pred_dist_coral_cc)))
    if len(true_dist_cc) < max_classes_cc:
        true_dist_cc = np.pad(true_dist_cc, (0, max_classes_cc - len(true_dist_cc)))
    
    x_plot_cc3 = np.arange(max_classes_cc) # Renamed
    width_dist_cc = 0.25 # Renamed
    
    plt.bar(x_plot_cc3 - width_dist_cc, pred_dist_direct_cc, width_dist_cc, label='Direct Prediction')
    plt.bar(x_plot_cc3, pred_dist_coral_cc, width_dist_cc, label='Class-Conditional CORAL')
    plt.bar(x_plot_cc3 + width_dist_cc, true_dist_cc, width_dist_cc, label='True Distribution')
    
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Prediction Distribution Comparison')
    plt.xticks(x_plot_cc3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{base_path}/{model_name}_analysis.png", dpi=300)
    plt.close()
    
    # 如果使用了阈值优化，添加ROC曲线
    if optimize_decision_threshold:
        plt.figure(figsize=(8, 6))
        fpr_cc, tpr_cc, thresholds_roc_cc = roc_curve(y_target, y_target_proba[:, 1]) # Renamed
        plt.plot(fpr_cc, tpr_cc, label=f'ROC Curve (AUC = {target_metrics["auc"]:.4f})')
        
        # 标出最佳阈值点
        optimal_idx_roc_cc_list = np.where(thresholds_roc_cc >= optimal_threshold_cc)[0]
        if len(optimal_idx_roc_cc_list) > 0:
            optimal_idx_roc_cc = optimal_idx_roc_cc_list[-1]
            plt.scatter(fpr_cc[optimal_idx_roc_cc], tpr_cc[optimal_idx_roc_cc], color='red', 
                        label=f'Optimal Threshold = {optimal_threshold_cc:.4f}')
        
        # 标出默认阈值0.5对应的点
        default_idx_roc_cc = None # Renamed
        for i, t_roc in enumerate(thresholds_roc_cc): # Renamed loop variable
            if t_roc <= 0.5:
                default_idx_roc_cc = i
                break
        if default_idx_roc_cc is not None:
            plt.scatter(fpr_cc[default_idx_roc_cc], tpr_cc[default_idx_roc_cc], color='blue', 
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
        'other': {
            'yt_pseudo_dist': np.bincount(yt_pseudo[yt_pseudo !=-1]).tolist() if np.any(yt_pseudo != -1) else [], # Handle empty case
            'use_target_labels': use_target_labels,
            'target_label_ratio': target_label_ratio if use_target_labels else None,
            'alpha': alpha
        }
    }

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


# 主函数
if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    if cuda.is_available():
        cuda.manual_seed(42)
    cuda.manual_seed(42)
    
    # 指定设备
    device = 'cuda' if cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # 加载所有数据集
    logging.info("\\nLoading datasets...")
    logging.info("1. Loading AI4healthcare.xlsx (A)...")
    df_ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    
    logging.info("2. Loading HenanCancerHospital_features63_58.xlsx (B)...")
    df_henan = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    # logging.info("3. Loading GuangzhouMedicalHospital_features23_no_nan.xlsx (C)...") # Removed Dataset C
    # df_guangzhou = pd.read_excel("data/GuangzhouMedicalHospital_features23_no_nan.xlsx") # Removed Dataset C

    # 使用指定的58个特征 (as per user request)
    selected_features = [
        'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10',
        'Feature11', 'Feature13', 'Feature14', 'Feature15', 'Feature16', 'Feature17', 'Feature18', 'Feature19', 'Feature20', 'Feature21',
        'Feature22', 'Feature23', 'Feature24', 'Feature25', 'Feature26', 'Feature27', 'Feature28', 'Feature29', 'Feature30', 'Feature31',
        'Feature32', 'Feature35', 'Feature37', 'Feature38', 'Feature39', 'Feature41', 'Feature42', 'Feature43', 'Feature44', 'Feature45',
        'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 'Feature55',
        'Feature56', 'Feature57', 'Feature58', 'Feature59', 'Feature60', 'Feature61', 'Feature62', 'Feature63'
    ]
    
    # 类别特征名称 (根据提供的索引从 selected_features 中提取0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 39, 40, 43, 44, 45, 46, 47, 48, 49, 57对应加1)
    cat_feature_names = [
        'Feature1', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10', 'Feature11',
        'Feature40', 'Feature41', 'Feature44', 'Feature45', 'Feature46', 'Feature47', 'Feature48', 'Feature49', 'Feature50', 'Feature58'
    ]

    # 计算新的 cat_idx 基于 selected_features (58 features)
    # This assumes all cat_feature_names are present in selected_features
    cat_idx_updated = [selected_features.index(f) for f in cat_feature_names if f in selected_features]


    logging.info("\\n=== Feature Information ===")
    logging.info(f"Number of selected features: {len(selected_features)}")
    logging.info(f"Selected features list: {selected_features}")
    logging.info(f"Categorical feature indices (updated for 58 features): {cat_idx_updated}")


    # 检查每个数据集中是否有所有选定的特征
    for dataset_name, dataset in [
        ("AI4health", df_ai4health), 
        ("Henan", df_henan)
        # ("Guangzhou", df_guangzhou) # Removed Dataset C
    ]:
        missing_features = [f for f in selected_features if f not in dataset.columns]
        if missing_features:
            logging.warning(f"Warning: {dataset_name} missing the following features: {missing_features}")
        else:
            logging.info(f"{dataset_name} contains all selected features")

    # 使用共同特征准备数据
    X_ai4health = df_ai4health[selected_features].copy()
    y_ai4health = df_ai4health["Label"].copy()
    
    X_henan = df_henan[selected_features].copy()
    y_henan = df_henan["Label"].copy()
    
    # X_guangzhou = df_guangzhou[selected_features].copy() # Removed Dataset C
    # y_guangzhou = df_guangzhou["Label"].copy() # Removed Dataset C

    # 创建结果目录
    base_results_path = './results_analytical_coral_A2B' # Updated base path
    os.makedirs(base_results_path, exist_ok=True)
    
    # 是否仅运行TSNE可视化
    only_visualize = False
    
    if only_visualize:
        logging.info("\\n\\n=== 仅进行CORAL t-SNE可视化 ===")
        
        # 定义域适应配置 (从A到B)
        configs_viz = [ # Renamed
            {
                'name': 'A_to_B',
                'source_name': 'A_AI4health',
                'target_name': 'B_Henan',
                'X_source_df': X_ai4health, # Pass DataFrames for y extraction later if needed by visualize
                'y_source': y_ai4health,
                'X_target_df': X_henan,
                'y_target': y_henan,
                'npz_file': f'{base_results_path}/TabPFN-Analytical-CORAL_A_to_B_aligned_features.npz'
            }
            # Removed A_to_C config
        ]
        
        # 为CORAL t-SNE可视化创建数据
        for config_v in configs_viz: # Renamed
            logging.info(f"\\n\\n{'='*50}")
            logging.info(f"CORAL域适应可视化: {config_v['source_name']} → {config_v['target_name']}")
            logging.info(f"{'='*50}")
            
            if not os.path.exists(config_v['npz_file']):
                logging.warning(f"NPZ file not found: {config_v['npz_file']}, skipping visualization.")
                continue

            # 加载保存的特征
            npz_data = np.load(config_v['npz_file'])
            X_source_viz = npz_data['X_source'] # Renamed
            X_target_viz = npz_data['X_target'] # Renamed
            X_target_aligned_viz = npz_data['X_target_aligned'] # Renamed
            
            # 使用t-SNE可视化对齐前后的分布
            tsne_save_path_viz = f"{base_results_path}/TabPFN-Analytical-CORAL_{config_v['name']}_tsne_detailed.png" # Renamed
            visualize_tsne(
                X_source=X_source_viz,
                X_target=X_target_viz,
                y_source=config_v['y_source'],
                y_target=config_v['y_target'],
                X_target_aligned=X_target_aligned_viz,
                title=f"CORAL Domain Adaptation t-SNE Visualization: {config_v['source_name']} → {config_v['target_name']}",
                save_path=tsne_save_path_viz,
                detect_anomalies=True # This was in original, keeping it.
            )
            
            # 为前5个特征绘制分布直方图
            hist_save_path_viz = f"{base_results_path}/TabPFN-Analytical-CORAL_{config_v['name']}_histograms.png" # Renamed
            visualize_feature_histograms(
                X_source=X_source_viz,
                X_target=X_target_viz,
                X_target_aligned=X_target_aligned_viz,
                feature_names=selected_features, 
                n_features_to_plot=None,
                title=f"Feature Distribution Before and After CORAL Alignment: {config_v['source_name']} → {config_v['target_name']}",
                save_path=hist_save_path_viz
            )
        
        logging.info("\\nCORAL t-SNE可视化完成!")
    else:
        # 运行解析版CORAL域适应实验
        logging.info("\\n\\n=== Running TabPFN with Analytical CORAL Domain Adaptation ===")
        
        # 定义域适应配置 (从A到B)
        coral_configs_main = [ # Renamed
            {
                'name': 'A_to_B',
                'source_name': 'A_AI4health',
                'target_name': 'B_Henan',
                'X_source': X_ai4health.values, # Pass numpy arrays to experiment function
                'y_source': y_ai4health.values,
                'X_target': X_henan.values,
                'y_target': y_henan.values
            }
            # Removed A_to_C config
        ]
        
        # 存储所有实验结果
        all_results = []
        
        # 运行实验
        for config_m in coral_configs_main: # Renamed
            logging.info(f"\\n\\n{'='*50}")
            logging.info(f"Domain Adaptation: {config_m['source_name']} → {config_m['target_name']} (Analytical CORAL)")
            logging.info(f"{'='*50}")
            
            # 运行解析版CORAL域适应实验
            metrics_res = run_coral_adaptation_experiment( # Renamed
                X_source=config_m['X_source'],
                y_source=config_m['y_source'],
                X_target=config_m['X_target'],
                y_target=config_m['y_target'],
                cat_idx=cat_idx_updated, # Pass updated cat_idx
                model_name=f"TabPFN-Analytical-CORAL_{config_m['name']}",
                base_path=base_results_path, # Use updated base_path
                feature_names_for_plot=selected_features # Pass feature names for plotting
            )
            
            # 保存结果
            result = {
                'source': config_m['source_name'],
                'target': config_m['target_name'],
                'source_acc': metrics_res['source']['acc'],
                'source_auc': metrics_res['source']['auc'],
                'source_f1': metrics_res['source']['f1'],
                'target_acc': metrics_res['target']['acc'],
                'target_auc': metrics_res['target']['auc'],
                'target_f1': metrics_res['target']['f1'],
                'target_acc_0': metrics_res['target']['acc_0'],
                'target_acc_1': metrics_res['target']['acc_1'],
                'direct_acc': metrics_res['direct']['acc'],
                'direct_auc': metrics_res['direct']['auc'], 
                'direct_f1': metrics_res['direct']['f1'],
                'direct_acc_0': metrics_res['direct']['acc_0'],
                'direct_acc_1': metrics_res['direct']['acc_1'],
                'tabpfn_time': metrics_res['times']['tabpfn'],
                'align_time': metrics_res['times']['align'],
                'inference_time': metrics_res['times']['inference'],
                'mean_diff_before': metrics_res['features']['mean_diff_before'],
                'std_diff_before': metrics_res['features']['std_diff_before'],
                'mean_diff_after': metrics_res['features']['mean_diff_after'],
                'std_diff_after': metrics_res['features']['std_diff_after']
            }
            all_results.append(result)
        
        # 创建结果表格
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f'{base_results_path}/all_results_analytical_coral.csv', index=False) # Updated path
        logging.info(f"Results saved to {base_results_path}/all_results_analytical_coral.csv")
        
        # 可视化比较
        logging.info("\\nGenerating visualization for method comparison...")
        for idx, config_plot in enumerate(coral_configs_main): # Renamed
            plt.figure(figsize=(12, 8))
            
            # 获取结果
            row_plot = results_df.iloc[idx] # Renamed
            target_name_plot = config_plot['target_name'] # Renamed
            
            # 绘制准确率和AUC比较
            plt.subplot(2, 2, 1)
            metrics_names_plot = ['acc', 'auc', 'f1'] # Renamed
            direct_values_plot = [row_plot['direct_acc'], row_plot['direct_auc'], row_plot['direct_f1']] # Renamed
            coral_values_plot = [row_plot['target_acc'], row_plot['target_auc'], row_plot['target_f1']] # Renamed
            
            x_metrics_plot = np.arange(len(metrics_names_plot)) # Renamed
            width_plot = 0.35 # Renamed
            
            plt.bar(x_metrics_plot - width_plot/2, direct_values_plot, width_plot, label='Direct TabPFN')
            plt.bar(x_metrics_plot + width_plot/2, coral_values_plot, width_plot, label='With Analytical CORAL')
            
            # 添加数据标签
            for i, v_plot in enumerate(direct_values_plot): # Renamed
                plt.text(i - width_plot/2, v_plot + 0.01, f'{v_plot:.3f}', ha='center')
            for i, v_plot in enumerate(coral_values_plot): # Renamed
                plt.text(i + width_plot/2, v_plot + 0.01, f'{v_plot:.3f}', ha='center')
                
            plt.ylabel('Score')
            plt.title('Performance Metrics')
            plt.xticks(x_metrics_plot, ['Accuracy', 'AUC', 'F1'])
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # 绘制每类准确率
            plt.subplot(2, 2, 2)
            class_metrics_names_plot = ['Class 0 Acc', 'Class 1 Acc'] # Renamed
            direct_class_values_plot = [row_plot['direct_acc_0'], row_plot['direct_acc_1']] # Renamed
            coral_class_values_plot = [row_plot['target_acc_0'], row_plot['target_acc_1']] # Renamed
            
            x_class_plot = np.arange(len(class_metrics_names_plot)) # Renamed
            
            plt.bar(x_class_plot - width_plot/2, direct_class_values_plot, width_plot, label='Direct TabPFN')
            plt.bar(x_class_plot + width_plot/2, coral_class_values_plot, width_plot, label='With Analytical CORAL')
            
            # 添加数据标签
            for i, v_plot in enumerate(direct_class_values_plot): # Renamed
                plt.text(i - width_plot/2, v_plot + 0.01, f'{v_plot:.3f}', ha='center')
            for i, v_plot in enumerate(coral_class_values_plot): # Renamed
                plt.text(i + width_plot/2, v_plot + 0.01, f'{v_plot:.3f}', ha='center')
                
            plt.ylabel('Accuracy')
            plt.title('Per-Class Accuracy')
            plt.xticks(x_class_plot, class_metrics_names_plot)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # 绘制特征差异
            plt.subplot(2, 2, 3)
            diff_metrics_names_plot = ['Mean Diff', 'Std Diff'] # Renamed
            before_values_plot = [row_plot['mean_diff_before'], row_plot['std_diff_before']] # Renamed
            after_values_plot = [row_plot['mean_diff_after'], row_plot['std_diff_after']] # Renamed
            
            x_diff_plot = np.arange(len(diff_metrics_names_plot)) # Renamed
            
            plt.bar(x_diff_plot - width_plot/2, before_values_plot, width_plot, label='Before Alignment')
            plt.bar(x_diff_plot + width_plot/2, after_values_plot, width_plot, label='After Alignment')
            
            # 添加数据标签和减少百分比
            for i, (before_p, after_p) in enumerate(zip(before_values_plot, after_values_plot)): # Renamed
                plt.text(i - width_plot/2, before_p + 0.01, f'{before_p:.3f}', ha='center')
                plt.text(i + width_plot/2, after_p + 0.01, f'{after_p:.3f}', ha='center')
                reduction = ((before_p - after_p) / before_p * 100) if before_p != 0 else 0
                plt.text(i, after_p/2 if after_p !=0 else 0.01 , f'-{reduction:.1f}%', ha='center', color='black', fontweight='bold')
                
            plt.ylabel('Difference')
            plt.title('Feature Distribution Difference')
            plt.xticks(x_diff_plot, diff_metrics_names_plot)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # 绘制计算时间
            plt.subplot(2, 2, 4)
            plt.bar(['TabPFN Training', 'CORAL Alignment', 'Inference'], 
                    [row_plot['tabpfn_time'], row_plot['align_time'], row_plot['inference_time']])
            plt.ylabel('Time (seconds)')
            plt.title('Computation Time')
            plt.grid(axis='y', alpha=0.3)
            
            plt.suptitle(f'Analytical CORAL Results: {config_plot["source_name"]} → {target_name_plot}', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(f'{base_results_path}/comparison_analytical_coral_{config_plot["name"]}.png', dpi=300) # Updated path
            plt.close()
            
        logging.info(f"\\nAll analytical CORAL results saved to {base_results_path}/ directory")

        # 创建类条件CORAL结果目录
        class_coral_base_path = f'{base_results_path}/class_conditional_coral' # Path based on new base
        os.makedirs(class_coral_base_path, exist_ok=True)
        
        logging.info("\\n\\n=== 运行带有类条件CORAL域适应的TabPFN ===")
        
        # 存储类条件CORAL实验结果
        class_coral_results = []
        
        # class_metrics is used later for direct_acc_0 and direct_acc_1, initialize it
        class_metrics_res_pseudo = {} 

        for config_cc in coral_configs_main: # Use the A_to_B config
            logging.info(f"\\n\\n{'='*50}")
            logging.info(f"域适应: {config_cc['source_name']} → {config_cc['target_name']} (类条件CORAL)")
            logging.info(f"{'='*50}")
            
            # 运行类条件CORAL域适应实验 (Pseudo labels)
            class_metrics_res_pseudo = run_class_conditional_coral_experiment( # Renamed
                X_source=config_cc['X_source'],
                y_source=config_cc['y_source'],
                X_target=config_cc['X_target'],
                y_target=config_cc['y_target'],
                cat_idx=cat_idx_updated, # Pass updated cat_idx
                model_name=f"TabPFN-ClassCORAL_Pseudo_{config_cc['name']}", # Added Pseudo to name
                base_path=class_coral_base_path, # Use class_coral_base_path
                optimize_decision_threshold=True,
                alpha=0.1,
                use_target_labels=False,
                feature_names_for_plot=selected_features
            )
            
            # 同样实验但使用10%真实标签
            class_metrics_res_labels = run_class_conditional_coral_experiment( # Renamed
                X_source=config_cc['X_source'],
                y_source=config_cc['y_source'],
                X_target=config_cc['X_target'],
                y_target=config_cc['y_target'],
                cat_idx=cat_idx_updated, # Pass updated cat_idx
                model_name=f"TabPFN-ClassCORAL_WithLabels_{config_cc['name']}",
                base_path=class_coral_base_path, # Use class_coral_base_path
                optimize_decision_threshold=True,
                alpha=0.1,
                use_target_labels=True,
                target_label_ratio=0.1,
                feature_names_for_plot=selected_features
            )
            
            # 保存结果
            class_coral_result = {
                'source': config_cc['source_name'],
                'target': config_cc['target_name'],
                'pseudo_acc': class_metrics_res_pseudo['target']['acc'],
                'pseudo_auc': class_metrics_res_pseudo['target']['auc'],
                'pseudo_f1': class_metrics_res_pseudo['target']['f1'],
                'pseudo_acc_0': class_metrics_res_pseudo['target']['acc_0'],
                'pseudo_acc_1': class_metrics_res_pseudo['target']['acc_1'],
                'withlabels_acc': class_metrics_res_labels['target']['acc'],
                'withlabels_auc': class_metrics_res_labels['target']['auc'],
                'withlabels_f1': class_metrics_res_labels['target']['f1'], 
                'withlabels_acc_0': class_metrics_res_labels['target']['acc_0'],
                'withlabels_acc_1': class_metrics_res_labels['target']['acc_1'],
                'direct_acc': class_metrics_res_pseudo['direct']['acc'], # From pseudo run, direct is same
                'direct_auc': class_metrics_res_pseudo['direct']['auc'],
                'direct_f1': class_metrics_res_pseudo['direct']['f1']
            }
            class_coral_results.append(class_coral_result)
        
        # 创建结果表格
        class_coral_df = pd.DataFrame(class_coral_results)
        class_coral_df.to_csv(f'{class_coral_base_path}/class_coral_results_summary.csv', index=False) # Updated path
        logging.info(f"类条件CORAL结果保存至 {class_coral_base_path}/class_coral_results_summary.csv")
        
        # 进行不同方法的比较
        for idx_cc_plot, config_cc_plot in enumerate(coral_configs_main): # Renamed
            plt.figure(figsize=(15, 8))
            
            target_name_cc_plot = config_cc_plot['target_name'] # Renamed
            
            # 获取不同方法的结果
            plain_coral_row = results_df.iloc[idx_cc_plot] # Analytical CORAL results
            class_coral_row_plot = class_coral_df.iloc[idx_cc_plot] # Class CORAL summary results
            
            # 比较不同方法的准确率
            plt.subplot(2, 2, 1)
            methods_plot = ['Direct', 'Standard CORAL', 'Class-CORAL (Pseudo)', 'Class-CORAL (10% Labels)'] # Renamed
            acc_values_plot = [ # Renamed
                class_coral_row_plot['direct_acc'], 
                plain_coral_row['target_acc'], # From analytical coral results
                class_coral_row_plot['pseudo_acc'],
                class_coral_row_plot['withlabels_acc']
            ]
            
            plt.bar(methods_plot, acc_values_plot)
            plt.ylabel('Accuracy')
            plt.title('Accuracy Comparison Across Methods')
            plt.xticks(rotation=15)
            plt.grid(axis='y', alpha=0.3)
            
            # 添加数据标签
            for i, v_cc_plot in enumerate(acc_values_plot): # Renamed
                plt.text(i, v_cc_plot + 0.01, f'{v_cc_plot:.3f}', ha='center')
            
            # 比较不同方法的每类准确率
            plt.subplot(2, 2, 2)
            # methods_plot is same as above
            
            # Ensure class_metrics_res_pseudo['direct'] exists and has acc_0, acc_1
            direct_acc_0_val = class_metrics_res_pseudo.get('direct', {}).get('acc_0', 0) if class_metrics_res_pseudo else 0
            direct_acc_1_val = class_metrics_res_pseudo.get('direct', {}).get('acc_1', 0) if class_metrics_res_pseudo else 0

            acc0_values_plot = [ # Renamed
                direct_acc_0_val, 
                plain_coral_row['target_acc_0'],
                class_coral_row_plot['pseudo_acc_0'],
                class_coral_row_plot['withlabels_acc_0']
            ]
            acc1_values_plot = [ # Renamed
                direct_acc_1_val,
                plain_coral_row['target_acc_1'],
                class_coral_row_plot['pseudo_acc_1'],
                class_coral_row_plot['withlabels_acc_1']
            ]
            
            x_methods_plot = np.arange(len(methods_plot)) # Renamed
            width_methods_plot = 0.35 # Renamed
            
            plt.bar(x_methods_plot - width_methods_plot/2, acc0_values_plot, width_methods_plot, label='Class 0 Accuracy')
            plt.bar(x_methods_plot + width_methods_plot/2, acc1_values_plot, width_methods_plot, label='Class 1 Accuracy')
            
            plt.ylabel('Accuracy')
            plt.title('Per-Class Accuracy Comparison')
            plt.xticks(x_methods_plot, methods_plot, rotation=15)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            # 比较不同方法的AUC
            plt.subplot(2, 2, 3)
            auc_values_plot = [ # Renamed
                class_coral_row_plot['direct_auc'], 
                plain_coral_row['target_auc'],
                class_coral_row_plot['pseudo_auc'],
                class_coral_row_plot['withlabels_auc']
            ]
            
            plt.bar(methods_plot, auc_values_plot)
            plt.ylabel('AUC')
            plt.title('AUC Comparison Across Methods')
            plt.xticks(rotation=15)
            plt.grid(axis='y', alpha=0.3)
            
            # 添加数据标签
            for i, v_cc_plot in enumerate(auc_values_plot): # Renamed
                plt.text(i, v_cc_plot + 0.01, f'{v_cc_plot:.3f}', ha='center')
            
            # 比较不同方法的F1分数
            plt.subplot(2, 2, 4)
            f1_values_plot = [ # Renamed
                class_coral_row_plot['direct_f1'], 
                plain_coral_row['target_f1'],
                class_coral_row_plot['pseudo_f1'],
                class_coral_row_plot['withlabels_f1']
            ]
            
            plt.bar(methods_plot, f1_values_plot)
            plt.ylabel('F1 Score')
            plt.title('F1 Score Comparison Across Methods')
            plt.xticks(rotation=15)
            plt.grid(axis='y', alpha=0.3)
            
            # 添加数据标签
            for i, v_cc_plot in enumerate(f1_values_plot): # Renamed
                plt.text(i, v_cc_plot + 0.01, f'{v_cc_plot:.3f}', ha='center')
            
            plt.suptitle(f'CORAL Methods Comparison: {config_cc_plot["source_name"]} → {target_name_cc_plot}', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(f'{class_coral_base_path}/methods_comparison_{config_cc_plot["name"]}.png', dpi=300) # Updated path
            plt.close()
        
        logging.info(f"\\n所有类条件CORAL结果保存至 {class_coral_base_path}/ 目录")
            
    # 运行后续可视化分析
    logging.info("\\n\\n=== 运行后续的可视化和分析 ===")
    
    # 对所有现有的NPZ文件进行t-SNE可视化
    import glob
    
    # 处理解析版CORAL的NPZ文件
    npz_files_analytical = glob.glob(f'{base_results_path}/TabPFN-Analytical-CORAL_*_aligned_features.npz') # Updated path
    logging.info(f"找到 {len(npz_files_analytical)} 个解析版CORAL特征对齐文件用于可视化 from {base_results_path}")
    
    for npz_file_an in npz_files_analytical: # Renamed
        logging.info(f"处理文件: {npz_file_an}")
        file_name_an = os.path.basename(npz_file_an) # Renamed
        
        source_name_an, target_name_an, y_source_an, y_target_an = (None, None, None, None) # Renamed vars
        # 从文件名中提取配置信息 (Only A_to_B is expected now)
        if 'A_to_B' in file_name_an:
            source_name_an = 'A_AI4health'
            target_name_an = 'B_Henan'
            y_source_an = y_ai4health.values # Use .values for numpy array
            y_target_an = y_henan.values
        else:
            logging.warning(f"无法从文件名确定配置 (Analytical CORAL): {file_name_an}, 跳过")
            continue
            
        # 加载特征数据
        data_an = np.load(npz_file_an) # Renamed
        X_source_an_data = data_an['X_source'] # Renamed
        X_target_an_data = data_an['X_target'] # Renamed
        X_target_aligned_an_data = data_an['X_target_aligned'] # Renamed
        
        # 创建保存路径
        save_base_an = npz_file_an.replace('_aligned_features.npz', '') # Renamed
        
        # 使用新模块进行可视化
        # t-SNE可视化
        tsne_save_path_an = f"{save_base_an}_tsne_visual.png" # Renamed
        visualize_tsne(
            X_source=X_source_an_data,
            X_target=X_target_an_data,
            y_source=y_source_an,
            y_target=y_target_an,
            X_target_aligned=X_target_aligned_an_data,
            title=f"CORAL Domain Adaptation t-SNE Visualization: {source_name_an} → {target_name_an}",
            save_path=tsne_save_path_an
        )
        
        # 特征直方图可视化
        hist_save_path_an = f"{save_base_an}_histograms_visual.png" # Renamed
        visualize_feature_histograms(
            X_source=X_source_an_data,
            X_target=X_target_an_data,
            X_target_aligned=X_target_aligned_an_data,
            feature_names=selected_features,
            n_features_to_plot=None,
            title=f"Feature Distribution Before and After CORAL Alignment: {source_name_an} → {target_name_an}",
            save_path=hist_save_path_an
        )
    
    # 处理类条件CORAL的NPZ文件
    class_coral_npz_files_viz = glob.glob(f'{class_coral_base_path}/TabPFN-ClassCORAL_*_aligned_features.npz') # Updated path
    logging.info(f"找到 {len(class_coral_npz_files_viz)} 个类条件CORAL特征对齐文件用于可视化 from {class_coral_base_path}")
    
    for npz_file_cc_viz in class_coral_npz_files_viz: # Renamed
        logging.info(f"处理文件: {npz_file_cc_viz}")
        file_name_cc_viz = os.path.basename(npz_file_cc_viz) # Renamed
        
        source_name_cc_viz, target_name_cc_viz, y_source_cc_viz, y_target_cc_viz = (None, None, None, None) # Renamed
        # 从文件名中提取配置信息 (Only A_to_B is expected)
        if 'A_to_B' in file_name_cc_viz:
            source_name_cc_viz = 'A_AI4health'
            target_name_cc_viz = 'B_Henan'
            y_source_cc_viz = y_ai4health.values
            y_target_cc_viz = y_henan.values
        else:
            logging.warning(f"无法从文件名确定配置 (Class CORAL): {file_name_cc_viz}, 跳过")
            continue
        
        # 判断是否为带标签的类条件CORAL
        with_labels_viz = "WithLabels" in file_name_cc_viz # Renamed
        coral_type_viz = "Class-CORAL with 10% Labels" if with_labels_viz else "Class-CORAL with Pseudo-Labels" # Renamed
        
        # 加载特征数据
        data_cc_viz = np.load(npz_file_cc_viz) # Renamed
        X_source_cc_data = data_cc_viz['X_source'] # Renamed
        X_target_cc_data = data_cc_viz['X_target'] # Renamed
        X_target_aligned_cc_data = data_cc_viz['X_target_aligned'] # Renamed
        
        # 如果存在伪标签，则加载
        yt_pseudo_viz = None # Renamed
        if 'yt_pseudo' in data_cc_viz:
            yt_pseudo_viz = data_cc_viz['yt_pseudo']
            if len(yt_pseudo_viz) > 0: # Check if not empty
                 logging.info(f"伪标签分布: {np.bincount(yt_pseudo_viz[yt_pseudo_viz != -1]) if np.any(yt_pseudo_viz !=-1) else 'Empty or all -1'}")
        
        # 创建保存路径
        save_base_cc_viz = npz_file_cc_viz.replace('_aligned_features.npz', '') # Renamed
        
        # t-SNE可视化
        tsne_save_path_cc_viz = f"{save_base_cc_viz}_tsne_visual.png" # Renamed
        visualize_tsne(
            X_source=X_source_cc_data,
            X_target=X_target_cc_data,
            y_source=y_source_cc_viz,
            y_target=y_target_cc_viz,
            X_target_aligned=X_target_aligned_cc_data,
            title=f"{coral_type_viz} t-SNE Visualization: {source_name_cc_viz} → {target_name_cc_viz}",
            save_path=tsne_save_path_cc_viz
        )
        
        # 特征直方图可视化
        hist_save_path_cc_viz = f"{save_base_cc_viz}_histograms_visual.png" # Renamed
        visualize_feature_histograms(
            X_source=X_source_cc_data,
            X_target=X_target_cc_data,
            X_target_aligned=X_target_aligned_cc_data,
            feature_names=selected_features,
            n_features_to_plot=None,
            title=f"{coral_type_viz} Feature Distribution: {source_name_cc_viz} → {target_name_cc_viz}",
            save_path=hist_save_path_cc_viz
        )
        
        # 如果有伪标签，创建额外的类条件可视化
        if yt_pseudo_viz is not None and len(yt_pseudo_viz) > 0 and np.any(yt_pseudo_viz != -1) :
            # 为每个类别创建特定的可视化（可选）
            classes_viz = np.unique(yt_pseudo_viz[yt_pseudo_viz != -1]) # Use only valid class labels
            logging.info(f"创建{len(classes_viz)}个类别的详细可视化 for {file_name_cc_viz}")
            
            for cls_viz in classes_viz: # Renamed
                # 获取每个类别的索引
                source_idx_viz = (y_source_cc_viz == cls_viz) # Renamed
                # For target, use pseudo labels for consistency with how class-conditional alignment was done for this class
                target_idx_viz = (yt_pseudo_viz == cls_viz) # Renamed
                
                if np.sum(source_idx_viz) < 5 or np.sum(target_idx_viz) < 5:
                    logging.warning(f"类别{cls_viz}样本数太少 (Source: {np.sum(source_idx_viz)}, Target Pseudo: {np.sum(target_idx_viz)})，跳过类别特定可视化 for {file_name_cc_viz}")
                    continue
                
                # 创建类别特定的可视化标题和保存路径
                cls_title_viz = f"{coral_type_viz} Class {cls_viz} Alignment: {source_name_cc_viz} → {target_name_cc_viz}" # Renamed
                cls_save_path_viz = f"{save_base_cc_viz}_class{cls_viz}_histograms.png" # Renamed
                
                # 仅对该类别的样本创建直方图
                visualize_feature_histograms(
                    X_source=X_source_cc_data[source_idx_viz],
                    X_target=X_target_cc_data[target_idx_viz], # Use original target data for this class
                    X_target_aligned=X_target_aligned_cc_data[target_idx_viz], # Use aligned target data for this class
                    feature_names=selected_features,
                    n_features_to_plot=10,
                    title=cls_title_viz,
                    save_path=cls_save_path_viz
                )
    
    logging.info("所有可视化完成！")