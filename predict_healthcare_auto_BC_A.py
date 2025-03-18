import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

# 设置日志
logging.disable(logging.INFO)

def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """计算所有评估指标"""
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'acc_0': conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]),
        'acc_1': conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    }

def print_metrics(dataset_name, metrics):
    """打印评估指标"""
    print(f"{dataset_name}准确率 (Accuracy): {metrics['acc']:.4f}")
    print(f"{dataset_name} AUC: {metrics['auc']:.4f}")
    print(f"{dataset_name} F1分数: {metrics['f1']:.4f}")
    print(f"{dataset_name}类别0准确率: {metrics['acc_0']:.4f}")
    print(f"{dataset_name}类别1准确率: {metrics['acc_1']:.4f}")

def run_experiment(
    X,
    y,
    device='cuda',
    max_time=15,
    random_state=42,
    base_path='./results'
):
    """
    Run AutoTabPFN experiment with given parameters
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    device : str
        Device to use for computation ('cuda' or 'cpu')
    max_time : int
        Maximum optimization time in seconds
    random_state : int
        Random state for reproducibility
    base_path : str
        Base path for saving results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with cross-validation scores
    """
    # Create results directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate experiment name based on parameters
    exp_name = f"AutoTabPFN-Health-T{max_time}-R{random_state}"
    
    print("Data Shape:", X.shape)
    print("Label Distribution:\n", y.value_counts())
    
    # Convert data to numpy arrays
    X_values = X.values.astype(np.float32)
    y_values = y.values.astype(np.int32)
    
    # ==============================
    # Cross Validation
    # ==============================
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_values), 1):
        X_train, X_test = X_values[train_idx], X_values[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        
        print(f"\nFold {fold}")
        print("-" * 50)
        
        # Initialize and train model
        start_time = time.time()
        clf = AutoTabPFNClassifier(
            device=device,
            max_time=max_time,
            random_state=random_state
        )
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred_proba = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        fold_time = time.time() - start_time
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        f1 = f1_score(y_test, y_pred)
        
        # Calculate per-class accuracy
        conf_matrix = confusion_matrix(y_test, y_pred)
        acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Class 0 Accuracy: {acc_0:.4f}")
        print(f"Class 1 Accuracy: {acc_1:.4f}")
        print(f"Time: {fold_time:.4f}s")
        
        fold_scores.append({
            'fold': fold,
            'accuracy': acc,
            'auc': auc,
            'f1': f1,
            'acc_0': acc_0,
            'acc_1': acc_1,
            'time': fold_time
        })
    
    # ==============================
    # Summary Results
    # ==============================
    scores_df = pd.DataFrame(fold_scores)
    
    # Save results with experiment name
    scores_df.to_csv(f'{base_path}/{exp_name}.csv', index=False)
    
    # Calculate and save final results
    final_results = pd.DataFrame({
        'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1', 'Time'],
        'Mean': [
            scores_df['auc'].mean(),
            scores_df['f1'].mean(),
            scores_df['accuracy'].mean(),
            scores_df['acc_0'].mean(),
            scores_df['acc_1'].mean(),
            scores_df['time'].mean()
        ],
        'Std': [
            scores_df['auc'].std(),
            scores_df['f1'].std(),
            scores_df['accuracy'].std(),
            scores_df['acc_0'].std(),
            scores_df['acc_1'].std(),
            scores_df['time'].std()
        ]
    })
    final_results.to_csv(f'{base_path}/{exp_name}-Final.csv', index=False)
    
    # Print results
    print("\nFinal Results:")
    print(f"Average Test AUC: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f}")
    print(f"Average Test F1: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f}")
    print(f"Average Test ACC: {scores_df['accuracy'].mean():.4f} ± {scores_df['accuracy'].std():.4f}")
    print(f"Average Test ACC_0: {scores_df['acc_0'].mean():.4f} ± {scores_df['acc_0'].std():.4f}")
    print(f"Average Test ACC_1: {scores_df['acc_1'].mean():.4f} ± {scores_df['acc_1'].std():.4f}")
    print(f"Average Time: {scores_df['time'].mean():.4f} ± {scores_df['time'].std():.4f}")
    
    # ==============================
    # Visualize Results
    # ==============================
    plt.figure(figsize=(15, 5))
    
    # Plot metrics
    plt.subplot(1, 3, 1)
    metrics = ['accuracy', 'auc', 'f1']
    for metric in metrics:
        plt.plot(scores_df['fold'], scores_df[metric], 'o-', label=metric.upper())
        plt.axhline(y=scores_df[metric].mean(), linestyle='--', alpha=0.3)
    plt.title('Performance Metrics across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.legend()
    
    # Plot class-wise accuracy
    plt.subplot(1, 3, 2)
    plt.plot(scores_df['fold'], scores_df['acc_0'], 'bo-', label='Class 0')
    plt.plot(scores_df['fold'], scores_df['acc_1'], 'ro-', label='Class 1')
    plt.axhline(y=scores_df['acc_0'].mean(), color='b', linestyle='--', alpha=0.3)
    plt.axhline(y=scores_df['acc_1'].mean(), color='r', linestyle='--', alpha=0.3)
    plt.title('Class-wise Accuracy across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot time
    plt.subplot(1, 3, 3)
    plt.plot(scores_df['fold'], scores_df['time'], 'go-', label='Time')
    plt.axhline(y=scores_df['time'].mean(), color='g', linestyle='--', alpha=0.3)
    plt.title('Computation Time across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Time (seconds)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{base_path}/{exp_name}-Plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return scores_df

def train_and_evaluate_on_external(
    X_train, 
    y_train, 
    X_test, 
    y_test, 
    device='cuda',
    max_time=15,
    random_state=42,
    base_path='./results'
):
    """
    训练模型并在外部数据集上评估
    """
    # Create results directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate experiment name based on parameters
    exp_name = f"AutoTabPFN-Health-External-T{max_time}-R{random_state}"
    
    print("Training Data Shape:", X_train.shape)
    print("Testing Data Shape:", X_test.shape)
    print("Training Label Distribution:\n", y_train.value_counts())
    print("Testing Label Distribution:\n", y_test.value_counts())
    
    # Convert data to numpy arrays
    X_train_values = X_train.values.astype(np.float32)
    y_train_values = y_train.values.astype(np.int32)
    X_test_values = X_test.values.astype(np.float32)
    y_test_values = y_test.values.astype(np.int32)
    
    # Train model on combined data
    print("\n训练模型...")
    start_time = time.time()
    clf = AutoTabPFNClassifier(
        device=device,
        max_time=max_time,
        random_state=random_state
    )
    clf.fit(X_train_values, y_train_values)
    train_time = time.time() - start_time
    
    # Evaluate on external test set
    print("\n在外部测试集上评估...")
    y_pred_proba = clf.predict_proba(X_test_values)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    metrics = evaluate_metrics(y_test_values, y_pred, y_pred_proba[:, 1])
    metrics['time'] = train_time
    
    # Print results
    print("\n=== 外部测试集评估结果 ===")
    print(f"训练时间: {train_time:.4f}秒")
    print_metrics("外部测试集", metrics)
    
    # Save results
    results = pd.DataFrame({
        'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1', 'Time'],
        'Value': [
            metrics['auc'],
            metrics['f1'],
            metrics['acc'],
            metrics['acc_0'],
            metrics['acc_1'],
            metrics['time']
        ]
    })
    
    results.to_csv(f'{base_path}/{exp_name}-Results.csv', index=False)
    
    # Create visualization
    plt.figure(figsize=(10, 5))
    
    # Plot confusion matrix
    conf_matrix = confusion_matrix(y_test_values, y_pred)
    plt.subplot(1, 2, 1)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Class 0', 'Class 1'])
    plt.yticks(tick_marks, ['Class 0', 'Class 1'])
    
    # Add text annotations to confusion matrix
    thresh = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Plot ROC curve
    plt.subplot(1, 2, 2)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test_values, y_pred_proba[:, 1])
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {metrics["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(f'{base_path}/{exp_name}-Analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return metrics, clf

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    # 加载所有数据集
    print("\n加载数据集...")
    print("1. 加载 AI4healthcare.xlsx (A)...")
    df_ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    
    print("2. 加载 HenanCancerHospital_features63_58.xlsx (B)...")
    df_henan = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    print("3. 加载 GuangzhouMedicalHospital_features22_no_nan.xlsx (C)...")
    df_guangzhou = pd.read_excel("data/GuangzhouMedicalHospital_features22_no_nan.xlsx")

    # 获取每个数据集的所有特征（除了Label）
    features_ai4health = [col for col in df_ai4health.columns if col != 'Label']
    features_henan = [col for col in df_henan.columns if col != 'Label']
    features_guangzhou = [col for col in df_guangzhou.columns if col != 'Label']

    # 找出三个数据集的共同特征
    common_features = set(features_ai4health)
    common_features = common_features.intersection(set(features_henan))
    common_features = common_features.intersection(set(features_guangzhou))
    common_features = list(common_features)

    print("\n=== 特征信息 ===")
    print(f"AI4health 特征数量: {len(features_ai4health)}")
    print(f"河南数据集特征数量: {len(features_henan)}")
    print(f"广州数据集特征数量: {len(features_guangzhou)}")
    print(f"共同特征数量: {len(common_features)}")
    print("共同特征列表:", common_features)

    # 使用共同特征准备数据
    X_ai4health = df_ai4health[common_features].copy()
    y_ai4health = df_ai4health["Label"].copy()
    
    X_henan = df_henan[common_features].copy()
    y_henan = df_henan["Label"].copy()
    
    X_guangzhou = df_guangzhou[common_features].copy()
    y_guangzhou = df_guangzhou["Label"].copy()

    # 合并 B 和 C 数据集
    print("\n合并 B (河南) 和 C (广州) 数据集...")
    X_combined = pd.concat([X_henan, X_guangzhou], axis=0)
    y_combined = pd.concat([y_henan, y_guangzhou], axis=0)
    
    print("合并后数据集形状:", X_combined.shape)
    print("合并后标签分布:\n", y_combined.value_counts())

    # 在合并数据集上进行交叉验证
    print("\n\n=== 在合并数据集 (B+C) 上进行交叉验证 ===")
    scores_df_combined = run_experiment(
        X_combined, 
        y_combined, 
        device='cuda', 
        max_time=180, 
        random_state=42,
        base_path='./results_BC_combined'
    )

    # 在合并数据集上训练，在 A 上测试
    print("\n\n=== 在合并数据集 (B+C) 上训练，在 A (AI4health) 上测试 ===")
    external_metrics, trained_model = train_and_evaluate_on_external(
        X_combined, 
        y_combined, 
        X_ai4health, 
        y_ai4health, 
        device='cuda', 
        max_time=180, 
        random_state=42,
        base_path='./results_BC_train_A_test'
    )

    # 保存数据集信息
    print("\n=== 数据集信息对比 ===")
    datasets_info = pd.DataFrame({
        'Dataset': ['A (AI4health)', 'B (Henan)', 'C (Guangzhou)', 'B+C Combined'],
        'Samples': [len(df_ai4health), len(df_henan), len(df_guangzhou), len(X_combined)],
        'Features': [len(common_features), len(common_features), len(common_features), len(common_features)],
        'Positive_Samples': [sum(y_ai4health), sum(y_henan), sum(y_guangzhou), sum(y_combined)],
        'Negative_Samples': [len(y_ai4health)-sum(y_ai4health), 
                           len(y_henan)-sum(y_henan), 
                           len(y_guangzhou)-sum(y_guangzhou),
                           len(y_combined)-sum(y_combined)]
    })
    print("\n数据集基本信息:")
    print(datasets_info.to_string(index=False))

    # 保存数据集信息
    os.makedirs('./results_BC_A_comparison', exist_ok=True)
    datasets_info.to_csv('./results_BC_A_comparison/datasets_info.csv', index=False)
    print("\n数据集信息已保存至: ./results_BC_A_comparison/datasets_info.csv")

    # 汇总所有结果
    summary_results = pd.DataFrame({
        'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1'],
        'B+C CV': [scores_df_combined['auc'].mean(), 
                  scores_df_combined['f1'].mean(),
                  scores_df_combined['accuracy'].mean(),
                  scores_df_combined['acc_0'].mean(),
                  scores_df_combined['acc_1'].mean()],
        'A (External)': [external_metrics['auc'],
                        external_metrics['f1'],
                        external_metrics['acc'],
                        external_metrics['acc_0'],
                        external_metrics['acc_1']]
    })

    # 保存汇总结果
    summary_results.to_csv('./results_BC_A_comparison/summary_results.csv', index=False)
    print("\n汇总结果已保存至: ./results_BC_A_comparison/summary_results.csv")
    print("\n=== 汇总结果 ===")
    print(summary_results.to_string(index=False))