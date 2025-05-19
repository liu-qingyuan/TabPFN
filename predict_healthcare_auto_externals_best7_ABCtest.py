import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import argparse
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from typing import List, Dict, Any, Tuple
from functools import partial

import logging
import sys

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

# 测试日志输出
logging.debug("This is a debug message")     # 不会输出
logging.info("This is an info message")      # 只进入 output.log
logging.warning("This is a warning message")  # 进入output.log以及error.log
logging.error("This is an error message")    # 进入output.log以及error.log
logging.critical("This is a critical message")  # 进入output.log以及error.log





# 尝试导入贝叶斯优化库
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    logging.warning("警告: scikit-optimize未安装，将使用随机搜索。安装方法: pip install scikit-optimize")
    SKOPT_AVAILABLE = False

# Create results directory
os.makedirs('./results/best_7_features_ABC', exist_ok=True)

# Define the best 7 features
best_features = [
    'Feature63', 'Feature2', 'Feature46', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]

# 最佳模型参数 (根据之前的调优结果)
best_model_params = {
    'max_time': 30,
    'preset': 'default',
    'ges_scoring': 'f1',
    'max_models': 10, 
    'validation_method': 'holdout',
    'n_repeats': 150,
    'n_folds': 10,
    'holdout_fraction': 0.5,
    'ges_n_iterations': 20,
    'ignore_limits': False
}

# 是否将特征视为分类特征 (Feature63, Feature46 是分类特征)
use_categorical_features = True
categorical_features = ['Feature63', 'Feature46']

# 创建模型参数配置
phe_init_args = {}
if best_model_params.get('max_models') is not None:
    phe_init_args['max_models'] = best_model_params['max_models']
if best_model_params.get('validation_method'):
    phe_init_args['validation_method'] = best_model_params['validation_method']
if best_model_params.get('n_repeats'):
    phe_init_args['n_repeats'] = best_model_params['n_repeats']
if best_model_params.get('n_folds'):
    phe_init_args['n_folds'] = best_model_params['n_folds']
if best_model_params.get('holdout_fraction'):
    phe_init_args['holdout_fraction'] = best_model_params['holdout_fraction']
if best_model_params.get('ges_n_iterations'):
    phe_init_args['ges_n_iterations'] = best_model_params['ges_n_iterations']

logging.info("\nEvaluating Best 7 Features:")
logging.info(best_features)

# Load data
logging.info("\nLoading datasets...")
train_df = pd.read_excel("data/AI4healthcare.xlsx")
external_B_df = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
external_C_df = pd.read_excel("data/GuangzhouMedicalHospital_features23_no_nan.xlsx")

# 确保数据集C中包含所有需要的特征
missing_features = [feature for feature in best_features if feature not in external_C_df.columns]
if missing_features:
    logging.warning(f"数据集C中缺少以下特征: {missing_features}")
    logging.warning("将跳过数据集C的评估")
    evaluate_dataset_c = False
    # 为了避免未定义错误，创建空的变量
    X_external_C = None
    y_external_C = None
    X_external_C_scaled = None
else:
    evaluate_dataset_c = True
    X_external_C = external_C_df[best_features].copy()
    y_external_C = external_C_df["Label"].copy()

# Prepare data
X_train = train_df[best_features].copy()
y_train = train_df["Label"].copy()
X_external_B = external_B_df[best_features].copy()
y_external_B = external_B_df["Label"].copy()

logging.info("\nData Information:")
logging.info(f"Training Data Shape (A): {X_train.shape}")
logging.info(f"External Data B Shape: {X_external_B.shape}")
logging.info(f"\nTraining Label Distribution (A):\n {y_train.value_counts()}")
logging.info(f"External Data B Label Distribution:\n {y_external_B.value_counts()}")

if evaluate_dataset_c and X_external_C is not None:
    logging.info(f"External Data C Shape: {X_external_C.shape}")
    logging.info(f"External Data C Label Distribution:\n {y_external_C.value_counts()}")

# Apply standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_external_B_scaled = scaler.transform(X_external_B)
if evaluate_dataset_c and X_external_C is not None:
    X_external_C_scaled = scaler.transform(X_external_C)

# ==============================
# 1. 10-fold Cross Validation on Training Data
# ==============================
logging.info("\n10-fold Cross Validation Results (Dataset A):")
logging.info("=" * 50)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # 获取分类特征索引
    categorical_indices = [best_features.index(feature) for feature in categorical_features] if use_categorical_features else None
    
    # Train model with best parameters
    model = AutoTabPFNClassifier(
        max_time=best_model_params.get('max_time', 30),
        preset=best_model_params.get('preset', 'default'),
        ges_scoring_string=best_model_params.get('ges_scoring', 'f1'),
        device='cuda', 
        random_state=42,
        ignore_pretraining_limits=best_model_params.get('ignore_limits', False),
        categorical_feature_indices=categorical_indices,
        phe_init_args=phe_init_args
    )
    
    model.fit(X_fold_train, y_fold_train)
    
    # Evaluate
    y_val_pred = model.predict(X_fold_val)
    y_val_proba = model.predict_proba(X_fold_val)
    
    # Calculate metrics
    fold_acc = accuracy_score(y_fold_val, y_val_pred)
    fold_auc = roc_auc_score(y_fold_val, y_val_proba[:, 1])
    fold_f1 = f1_score(y_fold_val, y_val_pred)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_fold_val, y_val_pred)
    fold_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    fold_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    
    cv_scores.append({
        'fold': fold,
        'accuracy': fold_acc,
        'auc': fold_auc,
        'f1': fold_f1,
        'acc_0': fold_acc_0,
        'acc_1': fold_acc_1
    })
    
    logging.info(f"\nFold {fold}:")
    logging.info(f"Accuracy: {fold_acc:.4f}")
    logging.info(f"AUC: {fold_auc:.4f}")
    logging.info(f"F1: {fold_f1:.4f}")
    logging.info(f"Class 0 Accuracy: {fold_acc_0:.4f}")
    logging.info(f"Class 1 Accuracy: {fold_acc_1:.4f}")

# Calculate and print average cross-validation scores
cv_df = pd.DataFrame(cv_scores)
logging.info("\nAverage Cross-validation Results (Dataset A):")
logging.info("=" * 50)
logging.info(f"Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
logging.info(f"AUC: {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
logging.info(f"F1: {cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}")
logging.info(f"Class 0 Accuracy: {cv_df['acc_0'].mean():.4f} ± {cv_df['acc_0'].std():.4f}")
logging.info(f"Class 1 Accuracy: {cv_df['acc_1'].mean():.4f} ± {cv_df['acc_1'].std():.4f}")

# ==============================
# 2. External Validation on Dataset B
# ==============================
logging.info("\nExternal Validation Results (Dataset B - Henan):")
logging.info("=" * 50)

# 获取分类特征索引
categorical_indices = [best_features.index(feature) for feature in categorical_features] if use_categorical_features else None

# Train final model on full training data with best parameters
final_model_B = AutoTabPFNClassifier(
    max_time=best_model_params.get('max_time', 30),
    preset=best_model_params.get('preset', 'default'),
    ges_scoring_string=best_model_params.get('ges_scoring', 'f1'),
    device='cuda', 
    random_state=42,
    ignore_pretraining_limits=best_model_params.get('ignore_limits', False),
    categorical_feature_indices=categorical_indices,
    phe_init_args=phe_init_args
)

final_model_B.fit(X_train_scaled, y_train)

# Evaluate on external data B
y_external_B_pred = final_model_B.predict(X_external_B_scaled)
y_external_B_proba = final_model_B.predict_proba(X_external_B_scaled)

# Calculate metrics for dataset B
external_B_acc = accuracy_score(y_external_B, y_external_B_pred)
external_B_auc = roc_auc_score(y_external_B, y_external_B_proba[:, 1])
external_B_f1 = f1_score(y_external_B, y_external_B_pred)

# Calculate confusion matrix for dataset B
conf_matrix_B = confusion_matrix(y_external_B, y_external_B_pred)
external_B_acc_0 = conf_matrix_B[0, 0] / (conf_matrix_B[0, 0] + conf_matrix_B[0, 1])
external_B_acc_1 = conf_matrix_B[1, 1] / (conf_matrix_B[1, 0] + conf_matrix_B[1, 1])

logging.info("\nExternal Validation Metrics (Dataset B):")
logging.info(f"Accuracy: {external_B_acc:.4f}")
logging.info(f"AUC: {external_B_auc:.4f}")
logging.info(f"F1: {external_B_f1:.4f}")
logging.info(f"Class 0 Accuracy: {external_B_acc_0:.4f}")
logging.info(f"Class 1 Accuracy: {external_B_acc_1:.4f}")
logging.info(f"\nConfusion Matrix (Dataset B):\n{conf_matrix_B}")

# ==============================
# 3. External Validation on Dataset C (if available)
# ==============================
if evaluate_dataset_c:
    logging.info("\nExternal Validation Results (Dataset C - Guangzhou):")
    logging.info("=" * 50)
    
    # 使用与数据集B相同的模型参数，在全部A数据集上训练
    final_model_C = AutoTabPFNClassifier(
        max_time=best_model_params.get('max_time', 30),
        preset=best_model_params.get('preset', 'default'),
        ges_scoring_string=best_model_params.get('ges_scoring', 'f1'),
        device='cuda', 
        random_state=42,
        ignore_pretraining_limits=best_model_params.get('ignore_limits', False),
        categorical_feature_indices=categorical_indices,
        phe_init_args=phe_init_args
    )
    
    final_model_C.fit(X_train_scaled, y_train)
    
    # Evaluate on external data C
    y_external_C_pred = final_model_C.predict(X_external_C_scaled)
    y_external_C_proba = final_model_C.predict_proba(X_external_C_scaled)
    
    # Calculate metrics for dataset C
    external_C_acc = accuracy_score(y_external_C, y_external_C_pred)
    external_C_auc = roc_auc_score(y_external_C, y_external_C_proba[:, 1])
    external_C_f1 = f1_score(y_external_C, y_external_C_pred)
    
    # Calculate confusion matrix for dataset C
    conf_matrix_C = confusion_matrix(y_external_C, y_external_C_pred)
    external_C_acc_0 = conf_matrix_C[0, 0] / (conf_matrix_C[0, 0] + conf_matrix_C[0, 1])
    external_C_acc_1 = conf_matrix_C[1, 1] / (conf_matrix_C[1, 0] + conf_matrix_C[1, 1])
    
    logging.info("\nExternal Validation Metrics (Dataset C):")
    logging.info(f"Accuracy: {external_C_acc:.4f}")
    logging.info(f"AUC: {external_C_auc:.4f}")
    logging.info(f"F1: {external_C_f1:.4f}")
    logging.info(f"Class 0 Accuracy: {external_C_acc_0:.4f}")
    logging.info(f"Class 1 Accuracy: {external_C_acc_1:.4f}")
    logging.info(f"\nConfusion Matrix (Dataset C):\n{conf_matrix_C}")

# Save results
results = {
    'cross_validation': {
        'accuracy': f"{cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}",
        'auc': f"{cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}",
        'f1': f"{cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}",
        'acc_0': f"{cv_df['acc_0'].mean():.4f} ± {cv_df['acc_0'].std():.4f}",
        'acc_1': f"{cv_df['acc_1'].mean():.4f} ± {cv_df['acc_1'].std():.4f}"
    },
    'external_validation_B': {
        'accuracy': f"{external_B_acc:.4f}",
        'auc': f"{external_B_auc:.4f}",
        'f1': f"{external_B_f1:.4f}",
        'acc_0': f"{external_B_acc_0:.4f}",
        'acc_1': f"{external_B_acc_1:.4f}",
        'confusion_matrix': conf_matrix_B.tolist()
    }
}

if evaluate_dataset_c:
    results['external_validation_C'] = {
        'accuracy': f"{external_C_acc:.4f}",
        'auc': f"{external_C_auc:.4f}",
        'f1': f"{external_C_f1:.4f}",
        'acc_0': f"{external_C_acc_0:.4f}",
        'acc_1': f"{external_C_acc_1:.4f}",
        'confusion_matrix': conf_matrix_C.tolist()
}

# Save to file
with open('./results/best_7_features_ABC/results.txt', 'w') as f:
    f.write("Best 7 Features Evaluation Results (Datasets A, B, C)\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("Features:\n")
    for i, feature in enumerate(best_features, 1):
        cat_suffix = " (categorical)" if use_categorical_features and feature in categorical_features else ""
        f.write(f"{i}. {feature}{cat_suffix}\n")
    
    f.write("\nBest Model Parameters:\n")
    f.write("-" * 30 + "\n")
    for param, value in best_model_params.items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nCross-validation Results (Dataset A):\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['cross_validation'].items():
        f.write(f"{metric}: {value}\n")
    
    f.write("\nExternal Validation Results (Dataset B - Henan):\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['external_validation_B'].items():
        if metric != 'confusion_matrix':
            f.write(f"{metric}: {value}\n")
    f.write(f"\nConfusion Matrix (Dataset B):\n{conf_matrix_B}")
    
    if evaluate_dataset_c:
        f.write("\n\nExternal Validation Results (Dataset C - Guangzhou):\n")
        f.write("-" * 30 + "\n")
        for metric, value in results['external_validation_C'].items():
            if metric != 'confusion_matrix':
                f.write(f"{metric}: {value}\n")
        f.write(f"\nConfusion Matrix (Dataset C):\n{conf_matrix_C}")

logging.info("\nResults have been saved to: ./results/best_7_features_ABC/results.txt")

# 绘制对比图表
plt.figure(figsize=(10, 6))
metrics = ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']
labels = ['Accuracy', 'AUC', 'F1', 'Class 0 Acc', 'Class 1 Acc']
x = np.arange(len(metrics))
width = 0.3

# 数据集 A 平均结果
a_values = [cv_df[metric].mean() for metric in metrics]
plt.bar(x - width, a_values, width, label='Dataset A (Training)', alpha=0.8)

# 数据集 B 结果
b_values = [external_B_acc, external_B_auc, external_B_f1, external_B_acc_0, external_B_acc_1]
plt.bar(x, b_values, width, label='Dataset B (Henan)', alpha=0.8)

# 数据集 C 结果 (如果可用)
if evaluate_dataset_c:
    c_values = [external_C_acc, external_C_auc, external_C_f1, external_C_acc_0, external_C_acc_1]
    plt.bar(x + width, c_values, width, label='Dataset C (Guangzhou)', alpha=0.8)

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Comparison Across Datasets')
plt.xticks(x, labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
plt.savefig('./results/best_7_features_ABC/metrics_comparison.png', dpi=300)

# 如果使用参数优化，替换main函数逻辑
def main():
    """参数解析的main函数被替换为直接运行的逻辑，因为已经在脚本中硬编码了最佳参数"""
    logging.info("\n运行主要评估逻辑已经在脚本主体中完成")
    logging.info("若需要使用参数优化，请使用原始代码中的参数解析部分")

if __name__ == "__main__":
    # 此处已经在脚本主体中完成了数据加载、模型训练和评估
    pass 