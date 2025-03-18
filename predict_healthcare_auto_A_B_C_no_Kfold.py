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
    使用全部数据进行训练和评估
    """
    # Create results directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate experiment name based on parameters
    exp_name = f"AutoTabPFN-Health-Full-T{max_time}-R{random_state}"
    
    print("Data Shape:", X.shape)
    print("Label Distribution:\n", y.value_counts())
    
    # Convert data to numpy arrays
    X_values = X.values.astype(np.float32)
    y_values = y.values.astype(np.int32)
    
    # Initialize and train model
    print("\n训练模型...")
    start_time = time.time()
    clf = AutoTabPFNClassifier(
        device=device,
        max_time=max_time,
        random_state=random_state
    )
    clf.fit(X_values, y_values)
    
    # Make predictions on training data
    y_pred_proba = clf.predict_proba(X_values)
    y_pred = np.argmax(y_pred_proba, axis=1)
    train_time = time.time() - start_time
    
    # Calculate metrics
    metrics = evaluate_metrics(y_values, y_pred, y_pred_proba[:, 1])
    metrics['time'] = train_time
    
    # Print results
    print("\n=== 训练结果 ===")
    print(f"训练时间: {train_time:.4f}秒")
    print_metrics("训练集", metrics)
    
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
    conf_matrix = confusion_matrix(y_values, y_pred)
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
    fpr, tpr, _ = roc_curve(y_values, y_pred_proba[:, 1])
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
    
    return metrics

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
    results = {}
    
    # 1. AI4health 数据集
    print("\n\n=== AI4health 数据集分析 ===")
    X_ai4health = df_ai4health[features_ai4health].copy()
    y_ai4health = df_ai4health["Label"].copy()
    results['AI4health'] = run_experiment(
        X_ai4health, 
        y_ai4health, 
        device='cuda', 
        max_time=180, 
        random_state=42,
        base_path='./results_AI4health_full'
    )

    # 2. 河南数据集
    print("\n\n=== 河南癌症医院数据集分析 ===")
    X_henan = df_henan[features_henan].copy()
    y_henan = df_henan["Label"].copy()
    results['Henan'] = run_experiment(
        X_henan, 
        y_henan, 
        device='cuda', 
        max_time=180, 
        random_state=42,
        base_path='./results_HenanCancerHospital_full'
    )

    # 3. 广州数据集
    print("\n\n=== 广州医疗医院数据集分析 ===")
    X_guangzhou = df_guangzhou[features_guangzhou].copy()
    y_guangzhou = df_guangzhou["Label"].copy()
    results['Guangzhou'] = run_experiment(
        X_guangzhou, 
        y_guangzhou, 
        device='cuda', 
        max_time=180, 
        random_state=42,
        base_path='./results_GuangzhouMedicalHospital_full'
    )

    # 汇总所有结果
    summary_results = pd.DataFrame({
        'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1', 'Time'],
        'AI4health': [results['AI4health']['auc'], 
                     results['AI4health']['f1'],
                     results['AI4health']['acc'],
                     results['AI4health']['acc_0'],
                     results['AI4health']['acc_1'],
                     results['AI4health']['time']],
        'Henan': [results['Henan']['auc'],
                 results['Henan']['f1'],
                 results['Henan']['acc'],
                 results['Henan']['acc_0'],
                 results['Henan']['acc_1'],
                 results['Henan']['time']],
        'Guangzhou': [results['Guangzhou']['auc'],
                     results['Guangzhou']['f1'],
                     results['Guangzhou']['acc'],
                     results['Guangzhou']['acc_0'],
                     results['Guangzhou']['acc_1'],
                     results['Guangzhou']['time']]
    })

    # 保存汇总结果
    os.makedirs('./results_comparison_full', exist_ok=True)
    summary_results.to_csv('./results_comparison_full/summary_results.csv', index=False)
    print("\n汇总结果已保存至: ./results_comparison_full/summary_results.csv")
    print("\n=== 汇总结果 ===")
    print(summary_results.to_string(index=False))