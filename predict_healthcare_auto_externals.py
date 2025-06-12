import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import logging

# logging.getLogger("AutoPostHocEnsemble").setLevel(logging.WARNING)
logging.disable(logging.INFO)      # Completely disable INFO and below logs
# logging.disable(logging.WARNING)   # If you don't even want the Warning level

# Create results directory
os.makedirs('./results/best_7_features_optimized_cv10', exist_ok=True)

# Define the best 7 features
best_features = [
    'Feature63', 'Feature2', 'Feature46', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]

print("\nEvaluating Best 7 Features:")
print(best_features)

# Load data
print("\nLoading datasets...")
train_df = pd.read_excel("data/AI4healthcare.xlsx")
external_df = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")

# Prepare data
X_train = train_df[best_features].copy()
y_train = train_df["Label"].copy()
X_external = external_df[best_features].copy()
y_external = external_df["Label"].copy()

print("\nData Information:")
print("Training Data Shape:", X_train.shape)
print("External Data Shape:", X_external.shape)
print("\nTraining Label Distribution:\n", y_train.value_counts())
print("External Label Distribution:\n", y_external.value_counts())

# Apply standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_external_scaled = scaler.transform(X_external)

# ==============================
# 1. 10-fold Cross Validation on Training Data (Dataset A)
# ==============================
print("\n10-fold Cross Validation Results on Dataset A:")
print("=" * 50)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores_A = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train model with optimized parameters
    model = AutoTabPFNClassifier(
        device='cuda', 
        max_time=30,
        preset='default',
        ges_scoring_string='f1',
        random_state=42,
        ignore_pretraining_limits=False,
        phe_init_args={
            'max_models': 10,
            'validation_method': 'cv',
            'n_repeats': 150,
            'n_folds': 5,
            'ges_n_iterations': 20
        }
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
    
    cv_scores_A.append({
        'fold': fold,
        'accuracy': fold_acc,
        'auc': fold_auc,
        'f1': fold_f1,
        'acc_0': fold_acc_0,
        'acc_1': fold_acc_1
    })
    
    print(f"\nFold {fold}:")
    print(f"Accuracy: {fold_acc:.4f}")
    print(f"AUC: {fold_auc:.4f}")
    print(f"F1: {fold_f1:.4f}")
    print(f"Class 0 Accuracy: {fold_acc_0:.4f}")
    print(f"Class 1 Accuracy: {fold_acc_1:.4f}")

# Calculate and print average cross-validation scores for Dataset A
cv_df_A = pd.DataFrame(cv_scores_A)
print("\nAverage Cross-validation Results on Dataset A:")
print("=" * 50)
print(f"Accuracy: {cv_df_A['accuracy'].mean():.4f} ± {cv_df_A['accuracy'].std():.4f}")
print(f"AUC: {cv_df_A['auc'].mean():.4f} ± {cv_df_A['auc'].std():.4f}")
print(f"F1: {cv_df_A['f1'].mean():.4f} ± {cv_df_A['f1'].std():.4f}")
print(f"Class 0 Accuracy: {cv_df_A['acc_0'].mean():.4f} ± {cv_df_A['acc_0'].std():.4f}")
print(f"Class 1 Accuracy: {cv_df_A['acc_1'].mean():.4f} ± {cv_df_A['acc_1'].std():.4f}")

# ==============================
# 2. 10-fold Cross Validation on External Data (Dataset B)
# ==============================
print("\n10-fold Cross Validation Results on Dataset B:")
print("=" * 50)

kf_ext = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores_B = []

for fold, (train_idx, val_idx) in enumerate(kf_ext.split(X_external_scaled), 1):
    X_fold_train, X_fold_val = X_external_scaled[train_idx], X_external_scaled[val_idx]
    y_fold_train, y_fold_val = y_external.iloc[train_idx], y_external.iloc[val_idx]
    
    # Train model with optimized parameters
    model = AutoTabPFNClassifier(
        device='cuda', 
        max_time=30,
        preset='default',
        ges_scoring_string='f1',
        random_state=42,
        ignore_pretraining_limits=False,
        phe_init_args={
            'max_models': 10,
            'validation_method': 'cv',
            'n_repeats': 150,
            'n_folds': 5,
            'ges_n_iterations': 20
        }
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
    
    cv_scores_B.append({
        'fold': fold,
        'accuracy': fold_acc,
        'auc': fold_auc,
        'f1': fold_f1,
        'acc_0': fold_acc_0,
        'acc_1': fold_acc_1
    })
    
    print(f"\nFold {fold}:")
    print(f"Accuracy: {fold_acc:.4f}")
    print(f"AUC: {fold_auc:.4f}")
    print(f"F1: {fold_f1:.4f}")
    print(f"Class 0 Accuracy: {fold_acc_0:.4f}")
    print(f"Class 1 Accuracy: {fold_acc_1:.4f}")

# Calculate and print average cross-validation scores for Dataset B
cv_df_B = pd.DataFrame(cv_scores_B)
print("\nAverage Cross-validation Results on Dataset B:")
print("=" * 50)
print(f"Accuracy: {cv_df_B['accuracy'].mean():.4f} ± {cv_df_B['accuracy'].std():.4f}")
print(f"AUC: {cv_df_B['auc'].mean():.4f} ± {cv_df_B['auc'].std():.4f}")
print(f"F1: {cv_df_B['f1'].mean():.4f} ± {cv_df_B['f1'].std():.4f}")
print(f"Class 0 Accuracy: {cv_df_B['acc_0'].mean():.4f} ± {cv_df_B['acc_0'].std():.4f}")
print(f"Class 1 Accuracy: {cv_df_B['acc_1'].mean():.4f} ± {cv_df_B['acc_1'].std():.4f}")

# ==============================
# 3. Cross-domain Validation: Train on A, Test on B
# ==============================
print("\nCross-domain Validation: Train on A, Test on B:")
print("=" * 50)

# Train final model on full training data (Dataset A)
final_model = AutoTabPFNClassifier(
    device='cuda', 
    max_time=30,
    preset='default',
    ges_scoring_string='f1',
    random_state=42,
    ignore_pretraining_limits=False,
    phe_init_args={
        'max_models': 10,
        'validation_method': 'cv',
        'n_repeats': 150,
        'n_folds': 5,
        'ges_n_iterations': 20
    }
)
final_model.fit(X_train_scaled, y_train)

# Evaluate on external data (Dataset B)
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

print("\nCross-domain Validation Metrics:")
print(f"Accuracy: {external_acc:.4f}")
print(f"AUC: {external_auc:.4f}")
print(f"F1: {external_f1:.4f}")
print(f"Class 0 Accuracy: {external_acc_0:.4f}")
print(f"Class 1 Accuracy: {external_acc_1:.4f}")
print(f"\nConfusion Matrix:\n{conf_matrix}")

# Save results
results = {
    'model_parameters': {
        'max_time': 30,
        'preset': 'default',
        'ges_scoring': 'f1',
        'max_models': 10,
        'validation_method': 'cv',
        'n_repeats': 150,
        'n_folds': 5,
        'ges_n_iterations': 20,
        'ignore_limits': False
    },
    'dataset_A_cv': {
        'accuracy': f"{cv_df_A['accuracy'].mean():.4f} ± {cv_df_A['accuracy'].std():.4f}",
        'auc': f"{cv_df_A['auc'].mean():.4f} ± {cv_df_A['auc'].std():.4f}",
        'f1': f"{cv_df_A['f1'].mean():.4f} ± {cv_df_A['f1'].std():.4f}",
        'acc_0': f"{cv_df_A['acc_0'].mean():.4f} ± {cv_df_A['acc_0'].std():.4f}",
        'acc_1': f"{cv_df_A['acc_1'].mean():.4f} ± {cv_df_A['acc_1'].std():.4f}"
    },
    'dataset_B_cv': {
        'accuracy': f"{cv_df_B['accuracy'].mean():.4f} ± {cv_df_B['accuracy'].std():.4f}",
        'auc': f"{cv_df_B['auc'].mean():.4f} ± {cv_df_B['auc'].std():.4f}",
        'f1': f"{cv_df_B['f1'].mean():.4f} ± {cv_df_B['f1'].std():.4f}",
        'acc_0': f"{cv_df_B['acc_0'].mean():.4f} ± {cv_df_B['acc_0'].std():.4f}",
        'acc_1': f"{cv_df_B['acc_1'].mean():.4f} ± {cv_df_B['acc_1'].std():.4f}"
    },
    'cross_domain_validation': {
        'accuracy': f"{external_acc:.4f}",
        'auc': f"{external_auc:.4f}",
        'f1': f"{external_f1:.4f}",
        'acc_0': f"{external_acc_0:.4f}",
        'acc_1': f"{external_acc_1:.4f}",
        'confusion_matrix': conf_matrix.tolist()
    }
}

# Save to file
with open('./results/best_7_features_optimized_cv10/results.txt', 'w') as f:
    f.write("Best 7 Features Evaluation Results (Optimized Parameters)\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("Model Parameters:\n")
    f.write("-" * 30 + "\n")
    for param, value in results['model_parameters'].items():
        f.write(f"{param}: {value}\n")
    
    f.write("\nFeatures:\n")
    for i, feature in enumerate(best_features, 1):
        f.write(f"{i}. {feature}\n")
    
    f.write("\nDataset A (AI4health) - 10-fold CV Results:\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['dataset_A_cv'].items():
        f.write(f"{metric}: {value}\n")
    
    f.write("\nDataset B (HenanCancerHospital) - 10-fold CV Results:\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['dataset_B_cv'].items():
        f.write(f"{metric}: {value}\n")
    
    f.write("\nCross-domain Validation (Train A → Test B):\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['cross_domain_validation'].items():
        if metric != 'confusion_matrix':
            f.write(f"{metric}: {value}\n")
    f.write(f"\nConfusion Matrix:\n{conf_matrix}")

print("\nResults have been saved to: ./results/best_7_features_optimized_cv10/results.txt")

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='AutoTabPFN Healthcare Prediction with Optimized Parameters')
    
    # 基本参数
    parser.add_argument('--output_dir', type=str, default='./results/best_7_features_optimized_cv10',
                        help='Output directory for results')
    parser.add_argument('--features', type=str, nargs='+', 
                        default=['Feature63', 'Feature2', 'Feature46', 
                                'Feature56', 'Feature42', 'Feature39', 'Feature43'],
                        help='List of features to use for prediction')
    parser.add_argument('--categorical_features', type=str, nargs='+',
                        default=['Feature63', 'Feature46'],
                        help='List of categorical features among the selected features')
    
    # AutoTabPFN 参数 - 使用优化过的默认值
    parser.add_argument('--max_time', type=int, default=30,
                        help='Maximum time (in seconds) for fitting the post hoc ensemble')
    parser.add_argument('--preset', type=str, default='default', 
                        choices=['default', 'custom_hps', 'avoid_overfitting'],
                        help='The preset to use for the post hoc ensemble')
    parser.add_argument('--ges_scoring', type=str, default='f1',
                        choices=['accuracy', 'roc', 'auroc', 'f1', 'log_loss'],
                        help='Scoring metric for ensemble search')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cpu', 'cuda'],
                        help='Device to use for computation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    parser.add_argument('--ignore_limits', action='store_true',
                        help='Whether to ignore pretraining limits')
    
    # 额外的PHE参数 - 使用优化过的默认值
    parser.add_argument('--max_models', type=int, default=10,
                        help='Maximum number of base models to use')
    parser.add_argument('--validation_method', type=str, default='cv',
                        choices=['holdout', 'cv'],
                        help='Validation method to use')
    parser.add_argument('--n_repeats', type=int, default=150,
                        help='Number of repeats for validation')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--holdout_fraction', type=float, default=0.33,
                        help='Fraction of data to use for holdout validation')
    parser.add_argument('--ges_n_iterations', type=int, default=20,
                        help='Number of iterations for greedy ensemble search')
    
    return parser.parse_args()

def get_categorical_indices(all_features, categorical_features):
    """获取分类特征在特征列表中的索引位置
    
    Args:
        all_features: 所有特征的列表
        categorical_features: 分类特征的列表
        
    Returns:
        分类特征的索引列表
    """
    indices = []
    for cat_feature in categorical_features:
        if cat_feature in all_features:
            indices.append(all_features.index(cat_feature))
    return indices

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建结果目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取分类特征的索引
    categorical_indices = get_categorical_indices(args.features, args.categorical_features)
    
    # 打印特征信息
    print("\nEvaluating features:")
    print(f"All features: {args.features}")
    print(f"Categorical features (indices): {args.categorical_features} {categorical_indices}")
    
    # 加载数据
    print("\nLoading datasets...")
    train_df = pd.read_excel("data/AI4healthcare.xlsx")
    external_df = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    # 准备数据
    X_train = train_df[args.features].copy()
    y_train = train_df["Label"].copy()
    X_external = external_df[args.features].copy()
    y_external = external_df["Label"].copy()
    
    # 转换分类特征为整数类型
    for idx, feature in enumerate(args.features):
        if idx in categorical_indices:
            print(f"Converting {feature} to categorical...")
            X_train[feature] = X_train[feature].astype('category').cat.codes
            X_external[feature] = X_external[feature].astype('category').cat.codes
    
    print("\nData Information:")
    print("Training Data Shape:", X_train.shape)
    print("External Data Shape:", X_external.shape)
    print("\nTraining Label Distribution:\n", y_train.value_counts())
    print("External Label Distribution:\n", y_external.value_counts())
    
    # 应用标准缩放 - 只对非分类特征进行缩放
    numeric_indices = [i for i in range(len(args.features)) if i not in categorical_indices]
    numeric_features = [args.features[i] for i in numeric_indices]
    
    if numeric_features:
        print(f"Scaling numeric features: {numeric_features}")
        scaler = StandardScaler()
        X_train_numeric = X_train[numeric_features].copy()
        X_train_numeric_scaled = scaler.fit_transform(X_train_numeric)
        
        X_external_numeric = X_external[numeric_features].copy()
        X_external_numeric_scaled = scaler.transform(X_external_numeric)
        
        # 将缩放后的数值特征放回数据框
        for i, feature in enumerate(numeric_features):
            X_train[feature] = X_train_numeric_scaled[:, i]
            X_external[feature] = X_external_numeric_scaled[:, i]
    
    # 将DataFrame转换为NumPy数组
    X_train_scaled = X_train.values
    X_external_scaled = X_external.values
    
    # ==============================
    # 1. 10-fold 交叉验证
    # ==============================
    print("\n10-fold Cross Validation Results:")
    print("=" * 50)
    
    kf = KFold(n_splits=10, shuffle=True, random_state=args.random_state)
    cv_scores = []
    
    # 构建附加PHE参数
    phe_init_args = {}
    if args.max_models is not None:
        phe_init_args['max_models'] = args.max_models
    if args.validation_method:
        phe_init_args['validation_method'] = args.validation_method
    if args.n_repeats:
        phe_init_args['n_repeats'] = args.n_repeats
    if args.n_folds:
        phe_init_args['n_folds'] = args.n_folds
    if args.holdout_fraction:
        phe_init_args['holdout_fraction'] = args.holdout_fraction
    if args.ges_n_iterations:
        phe_init_args['ges_n_iterations'] = args.ges_n_iterations
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 使用增强版的模型初始化，传入分类特征索引
        model = AutoTabPFNClassifier(
            max_time=args.max_time,
            preset=args.preset,
            ges_scoring_string=args.ges_scoring,
            device=args.device,
            random_state=args.random_state + fold,  # 每个折使用不同随机种子
            ignore_pretraining_limits=args.ignore_limits,
            categorical_feature_indices=categorical_indices,  # 添加分类特征索引
            phe_init_args=phe_init_args
        )
        
        # 训练模型
        start_time = time.time()
        model.fit(X_fold_train, y_fold_train)
        train_time = time.time() - start_time
        
        # 评估
        y_val_pred = model.predict(X_fold_val)
        y_val_proba = model.predict_proba(X_fold_val)
        
        # 计算指标
        fold_acc = accuracy_score(y_fold_val, y_val_pred)
        fold_auc = roc_auc_score(y_fold_val, y_val_proba[:, 1])
        fold_f1 = f1_score(y_fold_val, y_val_pred)
        
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_fold_val, y_val_pred)
        fold_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        fold_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        
        cv_scores.append({
            'fold': fold,
            'accuracy': fold_acc,
            'auc': fold_auc,
            'f1': fold_f1,
            'acc_0': fold_acc_0,
            'acc_1': fold_acc_1,
            'train_time': train_time
        })
        
        print(f"\nFold {fold}:")
        print(f"Accuracy: {fold_acc:.4f}")
        print(f"AUC: {fold_auc:.4f}")
        print(f"F1: {fold_f1:.4f}")
        print(f"Class 0 Accuracy: {fold_acc_0:.4f}")
        print(f"Class 1 Accuracy: {fold_acc_1:.4f}")
        print(f"Training Time: {train_time:.2f} seconds")
    
    # 计算并打印平均交叉验证得分
    cv_df = pd.DataFrame(cv_scores)
    print("\nAverage Cross-validation Results:")
    print("=" * 50)
    print(f"Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
    print(f"AUC: {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
    print(f"F1: {cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}")
    print(f"Class 0 Accuracy: {cv_df['acc_0'].mean():.4f} ± {cv_df['acc_0'].std():.4f}")
    print(f"Class 1 Accuracy: {cv_df['acc_1'].mean():.4f} ± {cv_df['acc_1'].std():.4f}")
    print(f"Average Training Time: {cv_df['train_time'].mean():.2f} ± {cv_df['train_time'].std():.2f} seconds")
    
    # ==============================
    # 2. 外部验证
    # ==============================
    print("\nExternal Validation Results:")
    print("=" * 50)
    
    # 对最终模型使用更多时间
    final_phe_init_args = phe_init_args.copy()
    if args.max_models is not None:
        final_phe_init_args['max_models'] = min(10, args.max_models * 2)
    
    # 在完整训练数据上训练最终模型
    final_model = AutoTabPFNClassifier(
        max_time=args.max_time * 2,  # 给最终模型更多时间
        preset=args.preset,
        ges_scoring_string=args.ges_scoring,
        device=args.device,
        random_state=args.random_state,
        ignore_pretraining_limits=args.ignore_limits,
        categorical_feature_indices=categorical_indices,  # 添加分类特征索引
        phe_init_args=final_phe_init_args
    )
    
    # 训练和记录时间
    start_time = time.time()
    final_model.fit(X_train_scaled, y_train)
    final_train_time = time.time() - start_time
    print(f"Final model training time: {final_train_time:.2f} seconds")
    
    # 在外部数据上评估
    y_external_pred = final_model.predict(X_external_scaled)
    y_external_proba = final_model.predict_proba(X_external_scaled)
    
    # 计算指标
    external_acc = accuracy_score(y_external, y_external_pred)
    external_auc = roc_auc_score(y_external, y_external_proba[:, 1])
    external_f1 = f1_score(y_external, y_external_pred)
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_external, y_external_pred)
    external_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    external_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    
    print("\nExternal Validation Metrics:")
    print(f"Accuracy: {external_acc:.4f}")
    print(f"AUC: {external_auc:.4f}")
    print(f"F1: {external_f1:.4f}")
    print(f"Class 0 Accuracy: {external_acc_0:.4f}")
    print(f"Class 1 Accuracy: {external_acc_1:.4f}")
    print(f"\nConfusion Matrix:\n{conf_matrix}")
    
    # 保存结果
    results = {
        'parameters': {
            'features': args.features,
            'categorical_features': args.categorical_features,
            'max_time': args.max_time,
            'preset': args.preset,
            'ges_scoring': args.ges_scoring,
            'device': args.device,
            'random_state': args.random_state,
            'ignore_limits': args.ignore_limits,
            'max_models': args.max_models,
            'validation_method': args.validation_method,
            'n_repeats': args.n_repeats,
            'n_folds': args.n_folds,
            'holdout_fraction': args.holdout_fraction,
            'ges_n_iterations': args.ges_n_iterations
        },
        'cross_validation': {
            'accuracy': f"{cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}",
            'auc': f"{cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}",
            'f1': f"{cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}",
            'acc_0': f"{cv_df['acc_0'].mean():.4f} ± {cv_df['acc_0'].std():.4f}",
            'acc_1': f"{cv_df['acc_1'].mean():.4f} ± {cv_df['acc_1'].std():.4f}",
            'train_time': f"{cv_df['train_time'].mean():.2f} ± {cv_df['train_time'].std():.2f} seconds"
        },
        'external_validation': {
            'accuracy': f"{external_acc:.4f}",
            'auc': f"{external_auc:.4f}",
            'f1': f"{external_f1:.4f}",
            'acc_0': f"{external_acc_0:.4f}",
            'acc_1': f"{external_acc_1:.4f}",
            'confusion_matrix': conf_matrix.tolist(),
            'train_time': f"{final_train_time:.2f} seconds"
        }
    }
    
    # 保存结果到文件
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Best 7 Features Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Parameters:\n")
        f.write("-" * 30 + "\n")
        for param, value in results['parameters'].items():
            f.write(f"{param}: {value}\n")
            
        f.write("\nFeatures:\n")
        for i, feature in enumerate(args.features, 1):
            cat_suffix = " (categorical)" if feature in args.categorical_features else ""
            f.write(f"{i}. {feature}{cat_suffix}\n")
        
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
    
    # 也保存CSV格式的结果
    cv_df.to_csv(os.path.join(args.output_dir, 'cv_scores.csv'), index=False)
    
    # 创建可视化
    plt.figure(figsize=(15, 10))
    
    # 绘制折线图（交叉验证得分）
    plt.subplot(2, 2, 1)
    plt.plot(cv_df['fold'], cv_df['accuracy'], 'o-', label='Accuracy')
    plt.plot(cv_df['fold'], cv_df['auc'], 's-', label='AUC')
    plt.plot(cv_df['fold'], cv_df['f1'], '^-', label='F1')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title('Cross-validation Scores by Fold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制类别准确率
    plt.subplot(2, 2, 2)
    plt.plot(cv_df['fold'], cv_df['acc_0'], 'o-', label='Class 0')
    plt.plot(cv_df['fold'], cv_df['acc_1'], 's-', label='Class 1')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy by Fold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制训练时间
    plt.subplot(2, 2, 3)
    plt.bar(cv_df['fold'], cv_df['train_time'])
    plt.xlabel('Fold')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time by Fold')
    plt.grid(True, alpha=0.3)
    
    # 绘制外部验证的混淆矩阵
    plt.subplot(2, 2, 4)
    plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
    plt.title('External Validation Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    
    # 在混淆矩阵中显示数字
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'results_visualization.png'), dpi=300)
    
    print(f"\nResults have been saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 