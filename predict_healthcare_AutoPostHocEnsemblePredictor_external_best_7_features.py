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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tabpfn_extensions.post_hoc_ensembles.pfn_phe import AutoPostHocEnsemblePredictor, TaskType
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

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

# 设置第三方库的日志级别
logging.getLogger("AutoPostHocEnsemble").setLevel(logging.WARNING)

# 测试日志输出
logging.debug("This is a debug message")     # 不会输出
logging.info("This is an info message")      # 只进入 stdout
logging.warning("This is a warning message")  # 进入stdout以及stderr
logging.error("This is an error message")    # 进入stdout以及stderr
logging.critical("This is a critical message")  # 进入stdout以及stderr

# Create results directory
os.makedirs('./results/best_7_features_results_AutoPostHocEnsemblePredictor', exist_ok=True)

# Define the best 7 features
best_features = [
    'Feature63', 'Feature2', 'Feature46', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]

logging.info("Evaluating Best 7 Features:")
logging.info(f"Features: {best_features}")

# Load data
logging.info("Loading datasets...")
train_df = pd.read_excel("data/AI4healthcare.xlsx")
external_df = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")

# Prepare data
X_train = train_df[best_features].copy()
y_train = train_df["Label"].copy()
X_external = external_df[best_features].copy()
y_external = external_df["Label"].copy()

logging.info("Data Information:")
logging.info(f"Training Data Shape: {X_train.shape}")
logging.info(f"External Data Shape: {X_external.shape}")
logging.info(f"Training Label Distribution:\n{y_train.value_counts()}")
logging.info(f"External Label Distribution:\n{y_external.value_counts()}")

# Apply standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_external_scaled = scaler.transform(X_external)

# ==============================
# 1. 10-fold Cross Validation on Training Data
# ==============================
logging.info("10-fold Cross Validation Results:")
logging.info("=" * 50)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train model with AutoPostHocEnsemblePredictor
    model = AutoPostHocEnsemblePredictor(
        preset="default",
        max_time=30,  # 30 seconds per fold
        task_type=TaskType.BINARY,
        ges_scoring_string="auroc",
        device="cuda",
        bm_random_state=42,
        ges_random_state=42,
        tabpfn_base_model_source="random_portfolio",
        max_models=10,  # 使用10个基础模型
        validation_method="cv",
        n_repeats=1,
        n_folds=5,
        ges_n_iterations=20,
        ignore_pretraining_limits=False
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
    
    logging.info(f"Fold {fold}:")
    logging.info(f"Accuracy: {fold_acc:.4f}")
    logging.info(f"AUC: {fold_auc:.4f}")
    logging.info(f"F1: {fold_f1:.4f}")
    logging.info(f"Class 0 Accuracy: {fold_acc_0:.4f}")
    logging.info(f"Class 1 Accuracy: {fold_acc_1:.4f}")

# Calculate and print average cross-validation scores
cv_df = pd.DataFrame(cv_scores)
logging.info("Average Cross-validation Results:")
logging.info("=" * 50)
logging.info(f"Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
logging.info(f"AUC: {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
logging.info(f"F1: {cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}")
logging.info(f"Class 0 Accuracy: {cv_df['acc_0'].mean():.4f} ± {cv_df['acc_0'].std():.4f}")
logging.info(f"Class 1 Accuracy: {cv_df['acc_1'].mean():.4f} ± {cv_df['acc_1'].std():.4f}")

# ==============================
# 2. External Validation
# ==============================
logging.info("External Validation Results:")
logging.info("=" * 50)

# Train final model on full training data
final_model = AutoPostHocEnsemblePredictor(
    preset="default",
    max_time=30,  # 使用统一的时间配置
    task_type=TaskType.BINARY,
    ges_scoring_string="auroc",
    device="cuda",
    bm_random_state=42,
    ges_random_state=42,
    tabpfn_base_model_source="random_portfolio",
    max_models=10,  # 最终模型使用10个基础模型
    validation_method="cv",
    n_repeats=1,
    n_folds=5,
    ges_n_iterations=20,
    ignore_pretraining_limits=False
)
final_model.fit(X_train_scaled, y_train)

# Evaluate on external data
y_external_pred = final_model.predict(X_external_scaled)
y_external_proba = final_model.predict_proba(X_external_scaled)

# Calculate metrics
external_acc = accuracy_score(y_external, y_external_pred)
external_auc = roc_auc_score(y_external, y_external_proba[:, 1])
external_f1 = f1_score(y_external, y_external_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_external, y_external_pred)
external_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
external_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])

logging.info("External Validation Metrics:")
logging.info(f"Accuracy: {external_acc:.4f}")
logging.info(f"AUC: {external_auc:.4f}")
logging.info(f"F1: {external_f1:.4f}")
logging.info(f"Class 0 Accuracy: {external_acc_0:.4f}")
logging.info(f"Class 1 Accuracy: {external_acc_1:.4f}")
logging.info(f"Confusion Matrix:\n{conf_matrix}")

# Save results
results = {
    'cross_validation': {
        'accuracy': f"{cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}",
        'auc': f"{cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}",
        'f1': f"{cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}",
        'acc_0': f"{cv_df['acc_0'].mean():.4f} ± {cv_df['acc_0'].std():.4f}",
        'acc_1': f"{cv_df['acc_1'].mean():.4f} ± {cv_df['acc_1'].std():.4f}"
    },
    'external_validation': {
        'accuracy': f"{external_acc:.4f}",
        'auc': f"{external_auc:.4f}",
        'f1': f"{external_f1:.4f}",
        'acc_0': f"{external_acc_0:.4f}",
        'acc_1': f"{external_acc_1:.4f}",
        'confusion_matrix': conf_matrix.tolist()
    },
    'model_config': {
        'model_type': 'AutoPostHocEnsemblePredictor',
        'preset': 'default',
        'task_type': 'BINARY',
        'max_time': 30,
        'ges_scoring': 'auroc',
        'max_models': 10,
        'validation_method': 'cv',
        'n_repeats': 1,
        'n_folds': 5,
        'ges_n_iterations': 20,
        'ignore_limits': False
    }
}

# Save to file
with open('./results/best_7_features_results_AutoPostHocEnsemblePredictor/results_auroc.txt', 'w') as f:
    f.write("Best 7 Features Evaluation Results with AutoPostHocEnsemblePredictor\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Features:\n")
    for i, feature in enumerate(best_features, 1):
        f.write(f"{i}. {feature}\n")
    
    f.write("\nModel Configuration:\n")
    f.write("-" * 30 + "\n")
    for param, value in results['model_config'].items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nCross-validation Results:\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['cross_validation'].items():
        f.write(f"{metric}: {value}\n")
    
    f.write("\nExternal Validation Results:\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['external_validation'].items():
        if metric != 'confusion_matrix':
            f.write(f"{metric}: {value}\n")
    f.write(f"\nConfusion Matrix:\n{conf_matrix}")

logging.info("Results have been saved to: ./results/best_7_features_results_AutoPostHocEnsemblePredictor/results_auroc.txt") 