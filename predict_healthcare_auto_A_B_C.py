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

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    # 加载所有数据集
    print("\n加载数据集...")
    print("1. 加载 AI4healthcare.xlsx...")
    df_ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    
    print("2. 加载 HenanCancerHospital_features63_58.xlsx...")
    df_henan = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    print("3. 加载 GuangzhouMedicalHospital_features22_no_nan.xlsx...")
    df_guangzhou = pd.read_excel("data/GuangzhouMedicalHospital_features22_no_nan.xlsx")

    # 获取每个数据集的所有特征（除了Label）
    features_ai4health = [col for col in df_ai4health.columns if col != 'Label']
    features_henan = [col for col in df_henan.columns if col != 'Label']
    features_guangzhou = [col for col in df_guangzhou.columns if col != 'Label']

    print("\n=== 特征信息 ===")
    print(f"AI4health 特征数量: {len(features_ai4health)}")
    print(f"河南数据集特征数量: {len(features_henan)}")
    print(f"广州数据集特征数量: {len(features_guangzhou)}")

    # 对每个数据集进行实验
    # 1. AI4health 数据集
    print("\n\n=== AI4health 数据集分析 ===")
    X_ai4health = df_ai4health[features_ai4health].copy()
    y_ai4health = df_ai4health["Label"].copy()
    
    scores_df_ai4health = run_experiment(
        X_ai4health, 
        y_ai4health, 
        device='cuda', 
        max_time=180, 
        random_state=42,
        base_path='./results_AI4health'
    )

    # 2. 河南数据集
    print("\n\n=== 河南癌症医院数据集分析 ===")
    X_henan = df_henan[features_henan].copy()
    y_henan = df_henan["Label"].copy()
    
    scores_df_henan = run_experiment(
        X_henan, 
        y_henan, 
        device='cuda', 
        max_time=180, 
        random_state=42,
        base_path='./results_HenanCancerHospital'
    )

    # 3. 广州数据集
    print("\n\n=== 广州医疗医院数据集分析 ===")
    X_guangzhou = df_guangzhou[features_guangzhou].copy()
    y_guangzhou = df_guangzhou["Label"].copy()
    
    scores_df_guangzhou = run_experiment(
        X_guangzhou, 
        y_guangzhou, 
        device='cuda', 
        max_time=180, 
        random_state=42,
        base_path='./results_GuangzhouMedicalHospital'
    )

    # 打印数据集信息对比
    print("\n=== 数据集信息对比 ===")
    datasets_info = pd.DataFrame({
        'Dataset': ['AI4health', 'Henan', 'Guangzhou'],
        'Samples': [len(df_ai4health), len(df_henan), len(df_guangzhou)],
        'Features': [len(features_ai4health), len(features_henan), len(features_guangzhou)],
        'Positive_Samples': [sum(y_ai4health), sum(y_henan), sum(y_guangzhou)],
        'Negative_Samples': [len(y_ai4health)-sum(y_ai4health), 
                           len(y_henan)-sum(y_henan), 
                           len(y_guangzhou)-sum(y_guangzhou)]
    })
    print("\n数据集基本信息:")
    print(datasets_info.to_string(index=False))

    # 保存数据集信息
    os.makedirs('./results_comparison', exist_ok=True)
    datasets_info.to_csv('./results_comparison/datasets_info.csv', index=False)
    print("\n数据集信息已保存至: ./results_comparison/datasets_info.csv")

    # 汇总所有结果
    summary_results = pd.DataFrame({
        'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1'],
        'AI4health': [scores_df_ai4health['auc'].mean(), 
                     scores_df_ai4health['f1'].mean(),
                     scores_df_ai4health['accuracy'].mean(),
                     scores_df_ai4health['acc_0'].mean(),
                     scores_df_ai4health['acc_1'].mean()],
        'Henan': [scores_df_henan['auc'].mean(),
                 scores_df_henan['f1'].mean(),
                 scores_df_henan['accuracy'].mean(),
                 scores_df_henan['acc_0'].mean(),
                 scores_df_henan['acc_1'].mean()],
        'Guangzhou': [scores_df_guangzhou['auc'].mean(),
                     scores_df_guangzhou['f1'].mean(),
                     scores_df_guangzhou['accuracy'].mean(),
                     scores_df_guangzhou['acc_0'].mean(),
                     scores_df_guangzhou['acc_1'].mean()]
    })

    # 保存汇总结果
    summary_results.to_csv('./results_comparison/summary_results.csv', index=False)
    print("\n汇总结果已保存至: ./results_comparison/summary_results.csv")
    print("\n=== 汇总结果 ===")
    print(summary_results.to_string(index=False))