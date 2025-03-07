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
from sklearn.preprocessing import StandardScaler

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
    max_time=15,  # Changed from 180 to 15 seconds
    random_state=42,
    base_path='./results/autoTabPFN'
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
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)
    
    # ==============================
    # Cross Validation
    # ==============================
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
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
    # 定义数据集路径
    datasets = {
        'ai4health': "data/AI4healthcare.xlsx",  # 添加原始训练数据集
        'guangzhou_no_nan': "data/GuangzhouMedicalHospital_features22_no_nan.xlsx",
        'guangzhou': "data/GuangzhouMedicalHospital_features23.xlsx",
        'henan': "data/HenanCancerHospital_features63_58.xlsx"
    }
    
    # 加载训练数据
    print("\n加载训练数据 (AI4healthcare.xlsx)...")
    train_df = pd.read_excel("data/AI4healthcare.xlsx")

    # 定义选定的特征
    selected_features = [
        'Feature2', 'Feature3', 'Feature4', 'Feature5',
        'Feature14', 'Feature15', 'Feature17', 'Feature22',
        'Feature39', 'Feature42', 'Feature43', 'Feature45',
        'Feature46', 'Feature47', 'Feature48', 'Feature49',
        'Feature50', 'Feature52', 'Feature53', 'Feature56',
        'Feature57', 'Feature63'
    ]

    # 准备训练数据
    X_train = train_df[selected_features].copy()
    y_train = train_df["Label"].copy()

    # 训练模型
    print("\n=== 训练模型 ===")
    clf = AutoTabPFNClassifier(
        device='cuda',
        max_time=30,
        random_state=42
    )
    clf.fit(X_train, y_train)

    # 为每个数据集进行预测（包括训练集）
    for dataset_name, dataset_path in datasets.items():
        print(f"\n=== 处理数据集: {dataset_name} ===")
        
        # 加载数据集
        test_df = pd.read_excel(dataset_path)
        
        # 确保数据集包含所有需要的特征
        if not all(feature in test_df.columns for feature in selected_features):
            print(f"警告：{dataset_name} 缺少某些特征，跳过处理")
            continue
        
        # 准备特征
        X_test = test_df[selected_features].copy()
        
        # 预测
        y_pred_proba = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # 创建结果DataFrame
        results_df = test_df.copy()  # 保留原始所有特征
        
        # 添加预测结果
        results_df['Predicted_Label'] = y_pred
        results_df['Probability'] = y_pred_proba[:, 1]
        results_df['Risk_Score'] = np.log(y_pred_proba[:, 1] / (1 - y_pred_proba[:, 1]))
        
        # 保存结果
        results_dir = './results/autoTabPFN'
        dataset_dir = f'{results_dir}/{dataset_name}'
        os.makedirs(dataset_dir, exist_ok=True)
        
        output_path = f'{dataset_dir}/predictions_with_features.xlsx'
        results_df.to_excel(output_path, index=False)
        print(f"预测结果已保存至: {output_path}")
        
        # 如果数据集包含标签，计算并保存评估指标
        if 'Label' in test_df.columns:
            y_test = test_df['Label']
            
            # 计算评估指标
            metrics = evaluate_metrics(y_test, y_pred, y_pred_proba[:, 1])
            
            # 创建评估指标DataFrame
            metrics_df = pd.DataFrame({
                'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1'],
                'Value': [metrics['auc'], metrics['f1'], 
                         metrics['acc'], metrics['acc_0'], 
                         metrics['acc_1']]
            })
            
            # 保存评估指标
            metrics_path = f'{dataset_dir}/evaluation_metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            print(f"评估指标已保存至: {metrics_path}")
            
            # 打印评估结果
            print("\n评估结果:")
            print_metrics(dataset_name, metrics)

            # 如果是训练集，额外打印
            if dataset_name == 'ai4health':
                print("\n这是训练集的预测结果，可用于评估模型的拟合情况")

    print("\n所有数据集处理完成！") 