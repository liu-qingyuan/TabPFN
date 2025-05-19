import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# 导入基础TabPFN分类器
from tabpfn_extensions import TabPFNClassifier
# 导入SHAP-IQ分析模块
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq

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
    print(f"{dataset_name} Accuracy: {metrics['acc']:.4f}")
    print(f"{dataset_name} AUC: {metrics['auc']:.4f}")
    print(f"{dataset_name} F1 Score: {metrics['f1']:.4f}")
    print(f"{dataset_name} Class 0 Accuracy: {metrics['acc_0']:.4f}")
    print(f"{dataset_name} Class 1 Accuracy: {metrics['acc_1']:.4f}")

def analyze_with_shapiq(
    clf,
    X_train,
    y_train,
    X_test,
    feature_names,
    dataset_name,
    n_model_evals=100,
    base_path='./results'
):
    """
    使用SHAP-IQ分析模型
    
    Parameters:
    -----------
    clf : TabPFNClassifier
        训练好的分类器模型
    X_train : numpy.ndarray
        训练数据特征
    y_train : numpy.ndarray
        训练数据标签
    X_test : numpy.ndarray
        测试数据特征
    feature_names : list
        特征名称列表
    dataset_name : str
        数据集名称
    n_model_evals : int
        用于解释的模型评估次数
    base_path : str
        保存结果的基础路径
    """
    print(f"\n===== {dataset_name} Dataset SHAP-IQ Analysis =====")
    
    # 创建保存目录
    shapiq_path = f"{base_path}/shapiq_analysis"
    os.makedirs(shapiq_path, exist_ok=True)
    
    # 1. Shapley Values (SV) 分析
    print("\nCalculating Shapley Values (SV)...")
    sv_explainer = tabpfn_shapiq.get_tabpfn_explainer(
        model=clf,
        data=X_train,
        labels=y_train,
        index="SV",
        verbose=True
    )
    
    # 对每个测试样本计算SHAP值
    n_samples = min(5, len(X_test))  # 选择前5个样本进行分析
    print(f"Analyzing {n_samples} test samples...")
    
    for i in range(n_samples):
        x_explain = X_test[i]
        shapley_values = sv_explainer.explain(x=x_explain, budget=n_model_evals)
        
        # 保存force plot
        plt.figure(figsize=(15, 5))
        shapley_values.plot_force(feature_names=feature_names)
        plt.title(f"{dataset_name} - Sample {i+1} Shapley Values")
        plt.tight_layout()
        plt.savefig(f"{shapiq_path}/{dataset_name}_sv_force_plot_sample_{i+1}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Faithful Shapley Interaction Index (FSII) 分析
    print("\nCalculating Faithful Shapley Interaction Index (FSII)...")
    fsii_explainer = tabpfn_shapiq.get_tabpfn_explainer(
        model=clf,
        data=X_train,
        labels=y_train,
        index="FSII",
        max_order=2,  # 分析特征对之间的交互
        verbose=True
    )
    
    # 对同样的样本计算交互值
    for i in range(n_samples):
        x_explain = X_test[i]
        shapley_interaction_values = fsii_explainer.explain(x=x_explain, budget=n_model_evals)
        
        # 保存upset plot
        plt.figure(figsize=(15, 10))
        shapley_interaction_values.plot_upset(feature_names=feature_names)
        plt.title(f"{dataset_name} - Sample {i+1} Feature Interactions")
        plt.tight_layout()
        plt.savefig(f"{shapiq_path}/{dataset_name}_fsii_upset_plot_sample_{i+1}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\n{dataset_name} Dataset SHAP-IQ Analysis Completed. Results saved in: {shapiq_path}/")

def run_experiment(
    X,
    y,
    dataset_name,
    device='cuda',
    random_state=42,
    base_path='./results',
    test_size=0.2
):
    """运行实验"""
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 生成实验名称
    exp_name = f"TabPFN-{dataset_name}-R{random_state}"
    
    print("Data Shape:", X.shape)
    print("Label Distribution:\n", y.value_counts())
    
    # 特征名称
    feature_names = X.columns.tolist()
    
    # 使用8/2划分数据
    print(f"\nUsing {(1-test_size)*100:.0f}/{test_size*100:.0f} train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # 转换为NumPy数组
    X_train_values = X_train.values.astype(np.float32)
    X_test_values = X_test.values.astype(np.float32)
    y_train_values = y_train.values.astype(np.int32)
    y_test_values = y_test.values.astype(np.int32)
    
    # 初始化并训练模型
    print("\nStarting model training...")
    start_time = time.time()
    clf = TabPFNClassifier(device=device)
    clf.fit(X_train_values, y_train_values)
    
    # 预测
    y_pred_proba = clf.predict_proba(X_test_values)
    y_pred = np.argmax(y_pred_proba, axis=1)
    train_time = time.time() - start_time
    print(f"Training and prediction completed, time: {train_time:.2f} seconds")
    
    # 计算指标
    metrics = evaluate_metrics(y_test_values, y_pred, y_pred_proba[:, 1])
    metrics['time'] = train_time
    
    # 打印结果
    print_metrics(dataset_name, metrics)
    
    # 保存结果
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(f'{base_path}/{exp_name}-results.csv', index=False)
    
    # 混淆矩阵可视化
    plt.figure(figsize=(8, 6))
    conf_matrix = confusion_matrix(y_test_values, y_pred)
    plt.imshow(conf_matrix, cmap='Blues')
    plt.colorbar()
    plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
    plt.yticks([0, 1], ['Actual 0', 'Actual 1'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(conf_matrix[i, j]), 
                    ha="center", va="center", color="red")
    
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{base_path}/{exp_name}-confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # SHAP-IQ分析
    print("\nStarting SHAP-IQ Analysis...")
    analyze_with_shapiq(
        clf=clf,
        X_train=X_train_values,
        y_train=y_train_values,
        X_test=X_test_values,
        feature_names=feature_names,
        dataset_name=dataset_name,
        base_path=base_path
    )
    
    return metrics

if __name__ == "__main__":
    # 加载所有数据集
    print("\nLoading datasets...")
    print("1. Loading AI4healthcare.xlsx...")
    df_ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    
    print("2. Loading HenanCancerHospital_features63_58.xlsx...")
    df_henan = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    print("3. Loading GuangzhouMedicalHospital_features23_no_nan.xlsx...")
    df_guangzhou = pd.read_excel("data/GuangzhouMedicalHospital_features23_no_nan.xlsx")

    # 获取每个数据集的所有特征（除了Label）
    features_ai4health = [col for col in df_ai4health.columns if col != 'Label']
    features_henan = [col for col in df_henan.columns if col != 'Label']
    features_guangzhou = [col for col in df_guangzhou.columns if col != 'Label']

    print("\n=== Feature Information ===")
    print(f"AI4health Features Count: {len(features_ai4health)}")
    print(f"Henan Dataset Features Count: {len(features_henan)}")
    print(f"Guangzhou Dataset Features Count: {len(features_guangzhou)}")

    # 设置结果目录
    os.makedirs('./results_comparison', exist_ok=True)
    
    # 存储所有数据集的指标
    all_metrics = {}

    # 对每个数据集进行实验
    # 1. AI4health 数据集
    print("\n\n=== AI4health Dataset Analysis ===")
    X_ai4health = df_ai4health[features_ai4health].copy()
    y_ai4health = df_ai4health["Label"].copy()
    
    metrics_ai4health = run_experiment(
        X_ai4health, 
        y_ai4health,
        dataset_name="AI4health",
        device='cuda', 
        random_state=42,
        base_path='./results_AI4health',
        test_size=0.2
    )
    all_metrics["AI4health"] = metrics_ai4health

    # 2. 河南数据集
    print("\n\n=== Henan Cancer Hospital Dataset Analysis ===")
    X_henan = df_henan[features_henan].copy()
    y_henan = df_henan["Label"].copy()
    
    metrics_henan = run_experiment(
        X_henan, 
        y_henan,
        dataset_name="Henan",
        device='cuda', 
        random_state=42,
        base_path='./results_HenanCancerHospital',
        test_size=0.2
    )
    all_metrics["Henan"] = metrics_henan

    # 3. 广州数据集
    print("\n\n=== Guangzhou Medical Hospital Dataset Analysis ===")
    X_guangzhou = df_guangzhou[features_guangzhou].copy()
    y_guangzhou = df_guangzhou["Label"].copy()
    
    metrics_guangzhou = run_experiment(
        X_guangzhou, 
        y_guangzhou,
        dataset_name="Guangzhou",
        device='cuda', 
        random_state=42,
        base_path='./results_GuangzhouMedicalHospital',
        test_size=0.2
    )
    all_metrics["Guangzhou"] = metrics_guangzhou

    # 打印数据集信息对比
    print("\n=== Dataset Information Comparison ===")
    datasets_info = pd.DataFrame({
        'Dataset': ['AI4health', 'Henan', 'Guangzhou'],
        'Samples': [len(df_ai4health), len(df_henan), len(df_guangzhou)],
        'Features': [len(features_ai4health), len(features_henan), len(features_guangzhou)],
        'Positive_Samples': [sum(y_ai4health), sum(y_henan), sum(y_guangzhou)],
        'Negative_Samples': [len(y_ai4health)-sum(y_ai4health), 
                           len(y_henan)-sum(y_henan), 
                           len(y_guangzhou)-sum(y_guangzhou)]
    })
    print("\nDataset Basic Information:")
    print(datasets_info.to_string(index=False))

    # 保存数据集信息
    datasets_info.to_csv('./results_comparison/datasets_info.csv', index=False)
    print("\nDataset Information saved to: ./results_comparison/datasets_info.csv")

    # 汇总所有结果
    summary_results = pd.DataFrame({
        'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1'],
        'AI4health': [all_metrics['AI4health']['auc'], 
                     all_metrics['AI4health']['f1'],
                     all_metrics['AI4health']['acc'],
                     all_metrics['AI4health']['acc_0'],
                     all_metrics['AI4health']['acc_1']],
        'Henan': [all_metrics['Henan']['auc'],
                 all_metrics['Henan']['f1'],
                 all_metrics['Henan']['acc'],
                 all_metrics['Henan']['acc_0'],
                 all_metrics['Henan']['acc_1']],
        'Guangzhou': [all_metrics['Guangzhou']['auc'],
                     all_metrics['Guangzhou']['f1'],
                     all_metrics['Guangzhou']['acc'],
                     all_metrics['Guangzhou']['acc_0'],
                     all_metrics['Guangzhou']['acc_1']]
    })

    # 保存汇总结果
    summary_results.to_csv('./results_comparison/summary_results.csv', index=False)
    print("\nSummary Results saved to: ./results_comparison/summary_results.csv")
    print("\n=== Summary Results ===")
    print(summary_results.to_string(index=False))