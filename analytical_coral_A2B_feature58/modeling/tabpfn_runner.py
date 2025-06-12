import numpy as np
import time
import os
import logging
# import pandas as pd # Removed as not directly used in this module for now
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

# Using relative imports assuming 'modeling' is a submodule of 'analytical_coral_A2B_feature58'
# from ..preprocessing.scaler import fit_apply_scaler # Removed unused import
from ..preprocessing.coral import coral_transform, class_conditional_coral_transform # Removed unused generate_pseudo_labels_for_coral
from ..metrics.evaluation import evaluate_metrics, optimize_threshold # Removed unused print_metrics
from ..visualization import ( # Updated import statement
    visualize_tsne,
    visualize_feature_histograms,
    plot_roc_curve # Updated function name
)
from typing import Optional, List, Dict, Any # Added Optional, List, Dict, Any for improved type hinting

from sklearn.preprocessing import StandardScaler # Added
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve # Added
import matplotlib.pyplot as plt # Added
from scipy.sparse import spmatrix # Added for explicit type checking

def run_coral_adaptation_experiment(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    cat_idx: list, # Added cat_idx parameter
    model_name: str = 'TabPFN-CORAL',
    tabpfn_params: Optional[dict] = None, # Modified default with Optional
    base_path: str = './results_analytical_coral_A2B', # Updated base_path
    optimize_decision_threshold: bool = True,
    feature_names_for_plot: Optional[list] = None # Added for histogram feature names, with Optional
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

    logging.info(f"\n=== {model_name} Model (Analytical CORAL) ===")
    
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 数据标准化
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled_transformed = scaler.transform(X_target)
    try:
        # Attempt to convert to dense if it's a sparse matrix
        X_target_scaled_dense = X_target_scaled_transformed.toarray()
    except AttributeError:
        # If not sparse or no toarray method, assume it's already dense or ndarray-like
        X_target_scaled_dense = X_target_scaled_transformed
    
    # 分析源域和目标域的特征分布差异
    logging.info("Analyzing domain differences before alignment...")
    source_mean = np.mean(X_source_scaled, axis=0)
    target_mean = np.mean(X_target_scaled_dense.astype(float), axis=0) 
    source_std = np.std(X_source_scaled, axis=0)
    target_std = np.std(X_target_scaled_dense.astype(float), axis=0) 
    
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
    y_target_pred_direct = tabpfn_model.predict(X_target_scaled_dense) # Use dense array
    y_target_proba_direct = tabpfn_model.predict_proba(X_target_scaled_dense) # Use dense array
    
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
    start_time = time.time()
    X_target_aligned = coral_transform(X_source_scaled, X_target_scaled_dense, cat_idx) # Pass dense array
    align_time = time.time() - start_time
    logging.info(f"CORAL transformation completed in {align_time:.2f} seconds")
    
    # 分析对齐前后的特征差异
    mean_diff_after = np.mean(np.abs(np.mean(X_source_scaled, axis=0) - np.mean(X_target_aligned, axis=0)))
    std_diff_after = np.mean(np.abs(np.std(X_source_scaled, axis=0) - np.std(X_target_aligned, axis=0)))
    logging.info(f"After alignment: Mean diff={mean_diff_after:.6f}, Std diff={std_diff_after:.6f}")
    if mean_diff > 0: # Avoid division by zero
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
    
    optimal_threshold = 0.5 # Default in case not optimized
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
             X_target=X_target_scaled_dense.astype(float), # Use dense array, ensure float for savez
             X_target_aligned=X_target_aligned)
    logging.info(f"Aligned features saved to: {aligned_features_path}")
    
    # 使用导入的可视化模块进行t-SNE可视化
    logging.info("\n使用t-SNE可视化CORAL对齐前后的分布...")
    tsne_save_path = f"{base_path}/{model_name}_tsne.png"
    visualize_tsne(
        X_source=X_source_scaled, 
        X_target=X_target_scaled_dense, # Use dense array
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
        X_target=X_target_scaled_dense, # Use dense array
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
        logging.info("\n绘制ROC曲线...")
        roc_save_path = f"{base_path}/{model_name}_roc_curve.png"
        plot_roc_curve(
            y_true=y_target,
            y_prob=y_target_proba[:, 1],
            save_path=roc_save_path,
            title=f'ROC Curve: {model_name}',
            optimal_threshold=optimal_threshold # Pass optimal threshold
        )
    
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
        },
        'data': {
            'X_source_scaled': X_source_scaled,
            'X_target_scaled': X_target_scaled_dense,
            'X_target_aligned': X_target_aligned
        }
    }

# 添加类条件CORAL域适应实验函数
def run_class_conditional_coral_experiment(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    cat_idx: list, # Added cat_idx
    model_name: str = 'TabPFN-ClassCORAL',
    tabpfn_params: Optional[dict] = None, # Modified default with Optional
    base_path: str = './results_class_conditional_coral_A2B', # Updated base_path
    optimize_decision_threshold: bool = True,
    alpha: float = 0.1,
    use_target_labels: bool = False,  # 是否使用部分真实标签，False则使用伪标签
    target_label_ratio: float = 0.1,    # 如果使用真实标签，从目标域取多少比例
    feature_names_for_plot: Optional[list] = None # Added for histogram feature names, with Optional
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

    logging.info(f"\n=== {model_name} Model (Class-Conditional CORAL) ===")
    
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 数据标准化
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled_transformed = scaler.transform(X_target)
    try:
        # Attempt to convert to dense if it's a sparse matrix
        X_target_scaled_dense = X_target_scaled_transformed.toarray()
    except AttributeError:
        # If not sparse or no toarray method, assume it's already dense or ndarray-like
        X_target_scaled_dense = X_target_scaled_transformed
    
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
    y_target_pred_direct = tabpfn_model.predict(X_target_scaled_dense) # Use dense array
    y_target_proba_direct = tabpfn_model.predict_proba(X_target_scaled_dense) # Use dense array
    
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
        logging.info(f"\nUsing {target_label_ratio:.1%} of target domain true labels for class-conditional CORAL alignment...")
        # n_labeled = int(len(y_target) * target_label_ratio) # Unused variable
        
        # 进行分层抽样，确保每个类别都有样本
        # Ensure there are enough samples for split, otherwise this might fail or give unexpected results
        if len(np.unique(y_target)) > 1 and len(y_target) * (1-target_label_ratio) >= len(np.unique(y_target)):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1-target_label_ratio, random_state=42)
            for train_idx, _ in sss.split(X_target_scaled_dense, y_target): # train_idx here is the labeled_idx
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
            X_target_unlabeled = X_target_scaled_dense[unlabeled_mask]
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
        logging.info("\nGenerating pseudo-labels using standard CORAL for class-conditional CORAL alignment...")
        # 先使用普通CORAL对齐
        X_target_aligned_temp = coral_transform(X_source_scaled, X_target_scaled_dense, cat_idx) # Pass dense array
        yt_pseudo = tabpfn_model.predict(X_target_aligned_temp)
        logging.info(f"Generated pseudo-label distribution: {np.bincount(yt_pseudo)}")
    
    # 使用类条件CORAL进行特征对齐
    logging.info("\nApplying class-conditional CORAL transformation...")
    start_time = time.time()
    X_target_aligned = class_conditional_coral_transform(
        X_source_scaled, y_source, X_target_scaled_dense, yt_pseudo, cat_idx, alpha=alpha # Use dense array
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
    unique_labels_cc, counts_cc = np.unique(y_target_pred, return_counts=True) # Renamed
    logging.info(f"Class-conditional CORAL aligned prediction distribution: {dict(zip(unique_labels_cc, counts_cc))}")
    
    # 计算目标域指标
    target_metrics = evaluate_metrics(y_target, y_target_pred, y_target_proba[:, 1])
    
    optimal_threshold_cc = 0.5 # Default
    # 优化决策阈值（可选）
    if optimize_decision_threshold:
        logging.info("\nOptimizing decision threshold using Youden index...")
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
    aligned_features_path_cc = f"{base_path}/{model_name}_aligned_features.npz" # Renamed
    np.savez(aligned_features_path_cc, 
             X_source=X_source_scaled, 
             X_target=X_target_scaled_dense.astype(float), # Use dense array, ensure float
             X_target_aligned=X_target_aligned,
             yt_pseudo=yt_pseudo)
    logging.info(f"Aligned features saved to: {aligned_features_path_cc}")
    
    # 绘制特征分布直方图
    logging.info("\n绘制对齐前后的特征分布直方图...")
    hist_save_path = f"{base_path}/{model_name}_histograms.png"
    visualize_feature_histograms(
        X_source=X_source_scaled,
        X_target=X_target_scaled_dense, # Use dense array
        X_target_aligned=X_target_aligned,
        feature_names=feature_names_for_plot,  # Use passed feature names
        n_features_to_plot=None,  # Plot all features
        title=f'Feature Distribution Before and After Class-Conditional CORAL Alignment: {model_name}',
        save_path=hist_save_path
    )
    
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
        logging.info("\n绘制ROC曲线...")
        roc_save_path_cc = f"{base_path}/{model_name}_roc_curve.png" # Renamed save path
        plot_roc_curve(
            y_true=y_target,
            y_prob=y_target_proba[:, 1],
            save_path=roc_save_path_cc,
            title=f'ROC Curve: {model_name}',
            optimal_threshold=optimal_threshold_cc # Pass optimal threshold
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
            'yt_pseudo_dist': np.bincount(yt_pseudo[yt_pseudo !=-1]).tolist() if np.any(yt_pseudo != -1) else [], # Handle empty case
            'use_target_labels': use_target_labels,
            'target_label_ratio': target_label_ratio if use_target_labels else None,
            'alpha': alpha
        },
        'data': {
            'X_source_scaled': X_source_scaled,
            'X_target_scaled': X_target_scaled_dense,
            'X_target_aligned': X_target_aligned
        }
    }