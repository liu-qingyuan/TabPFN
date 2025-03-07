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
    # 加载训练数据
    print("\n加载训练数据 (AI4healthcare.xlsx)...")
    train_df = pd.read_excel("data/AI4healthcare.xlsx")

    # 加载预测数据
    print("\n加载预测数据 (GuangzhouMedicalHospital_features22_no_nan.xlsx)...")
    test_df = pd.read_excel("data/GuangzhouMedicalHospital_features22_no_nan.xlsx")

    # 定义选定的特征
    selected_features = [
        'Feature2', 'Feature3', 'Feature4', 'Feature5',
        'Feature14', 'Feature15', 'Feature17', 'Feature22',
        'Feature39', 'Feature42', 'Feature43', 'Feature45',
        'Feature46', 'Feature47', 'Feature48', 'Feature49',
        'Feature50', 'Feature52', 'Feature53', 'Feature56',
        'Feature57', 'Feature63'
    ]

    # 分析数据分布
    print("\n=== 数据分布分析 ===")
    print("\n训练集描述性统计：")
    print(train_df[selected_features].describe())
    
    print("\n测试集描述性统计：")
    print(test_df[selected_features].describe())
    
    # 计算每个特征的分布差异
    print("\n特征分布差异（均值差异）：")
    for feature in selected_features:
        train_mean = train_df[feature].mean()
        test_mean = test_df[feature].mean()
        diff_percent = ((test_mean - train_mean) / train_mean) * 100
        print(f"{feature}: {diff_percent:.2f}%")

    print("\n选定的特征:", selected_features)
    print("特征数量:", len(selected_features))

    # 准备训练数据
    X_train = train_df[selected_features].copy()
    y_train = train_df["Label"].copy()

    # 准备预测数据
    X_test = test_df[selected_features].copy()
    y_test = test_df["Label"].copy()

    print("\n=== 使用标准化 ===")
    # 数据标准化版本
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练标准化版本的模型
    print("\n开始训练标准化版本的 AutoTabPFNClassifier...")
    clf_scaled = AutoTabPFNClassifier(
        device='cuda',
        max_time=30,
        random_state=42
    )
    
    start_time = time.time()
    clf_scaled.fit(X_train_scaled, y_train)
    train_time_scaled = time.time() - start_time

    # 评估标准化版本的训练集性能
    print("\n标准化版本 - 训练集性能评估:")
    y_train_pred_proba_scaled = clf_scaled.predict_proba(X_train_scaled)
    y_train_pred_scaled = np.argmax(y_train_pred_proba_scaled, axis=1)
    
    train_metrics_scaled = evaluate_metrics(y_train, y_train_pred_scaled, y_train_pred_proba_scaled[:, 1])
    print_metrics("训练集", train_metrics_scaled)

    # 评估标准化版本的测试集性能
    print("\n标准化版本 - 测试集性能评估:")
    y_pred_proba_scaled = clf_scaled.predict_proba(X_test_scaled)
    y_pred_scaled = np.argmax(y_pred_proba_scaled, axis=1)
    
    test_metrics_scaled = evaluate_metrics(y_test, y_pred_scaled, y_pred_proba_scaled[:, 1])
    print_metrics("测试集", test_metrics_scaled)
    print(f"训练时间: {train_time_scaled:.2f}秒")

    print("\n=== 不使用标准化 ===")
    # 训练未标准化版本的模型
    print("\n开始训练未标准化版本的 AutoTabPFNClassifier...")
    clf_unscaled = AutoTabPFNClassifier(
        device='cuda',
        max_time=10,
        random_state=42
    )
    
    start_time = time.time()
    clf_unscaled.fit(X_train, y_train)
    train_time_unscaled = time.time() - start_time

    # 评估未标准化版本的训练集性能
    print("\n未标准化版本 - 训练集性能评估:")
    y_train_pred_proba_unscaled = clf_unscaled.predict_proba(X_train)
    y_train_pred_unscaled = np.argmax(y_train_pred_proba_unscaled, axis=1)
    
    train_metrics_unscaled = evaluate_metrics(y_train, y_train_pred_unscaled, y_train_pred_proba_unscaled[:, 1])
    print_metrics("训练集", train_metrics_unscaled)

    # 评估未标准化版本的测试集性能
    print("\n未标准化版本 - 测试集性能评估:")
    y_pred_proba_unscaled = clf_unscaled.predict_proba(X_test)
    y_pred_unscaled = np.argmax(y_pred_proba_unscaled, axis=1)
    
    test_metrics_unscaled = evaluate_metrics(y_test, y_pred_unscaled, y_pred_proba_unscaled[:, 1])
    print_metrics("测试集", test_metrics_unscaled)
    print(f"训练时间: {train_time_unscaled:.2f}秒")

    # 保存结果
    results_dir = './results/autoTabPFN'
    features22_dir = f'{results_dir}/features22_no_nan'
    os.makedirs(features22_dir, exist_ok=True)
    
    # 保存两个版本的结果
    results = pd.DataFrame({
        'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1', 'Time'],
        'Train_Scaled': [train_metrics_scaled['auc'], train_metrics_scaled['f1'], 
                        train_metrics_scaled['acc'], train_metrics_scaled['acc_0'], 
                        train_metrics_scaled['acc_1'], train_time_scaled],
        'Test_Scaled': [test_metrics_scaled['auc'], test_metrics_scaled['f1'], 
                       test_metrics_scaled['acc'], test_metrics_scaled['acc_0'], 
                       test_metrics_scaled['acc_1'], train_time_scaled],
        'Train_Unscaled': [train_metrics_unscaled['auc'], train_metrics_unscaled['f1'], 
                          train_metrics_unscaled['acc'], train_metrics_unscaled['acc_0'], 
                          train_metrics_unscaled['acc_1'], train_time_unscaled],
        'Test_Unscaled': [test_metrics_unscaled['auc'], test_metrics_unscaled['f1'], 
                         test_metrics_unscaled['acc'], test_metrics_unscaled['acc_0'], 
                         test_metrics_unscaled['acc_1'], train_time_unscaled]
    })
    
    results.to_csv(f'{features22_dir}/comparison_results.csv', index=False)
    print(f"\n对比结果已保存至: {features22_dir}/comparison_results.csv")

    # 打印对比结果
    print("\n=== 标准化和未标准化版本性能对比 ===")
    print(results.to_string())

    # 保存两个版本的预测结果
    predictions_df = pd.DataFrame({
        'True_Label': test_df["Label"],
        'Scaled_Predicted_Label': y_pred_scaled,
        'Scaled_Probability': y_pred_proba_scaled[:, 1],
        'Scaled_Risk_Score': np.log(y_pred_proba_scaled[:, 1] / (1 - y_pred_proba_scaled[:, 1])),
        'Unscaled_Predicted_Label': y_pred_unscaled,
        'Unscaled_Probability': y_pred_proba_unscaled[:, 1],
        'Unscaled_Risk_Score': np.log(y_pred_proba_unscaled[:, 1] / (1 - y_pred_proba_unscaled[:, 1]))
    })
    predictions_df.to_csv(f'{features22_dir}/comparison_predictions.csv', index=False)
    print(f"\n预测结果对比已保存至: {features22_dir}/comparison_predictions.csv") 