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
    y_test,
    y_pred,
    feature_names,
    dataset_name,
    n_model_evals=100,
    base_path='./results'
):
    """
    使用SHAP-IQ分析模型
    """
    print(f"\n===== {dataset_name} Dataset SHAP-IQ Analysis =====")
    
    # 创建保存目录
    shapiq_path = f"{base_path}/shapiq_analysis"
    os.makedirs(shapiq_path, exist_ok=True)
    
    # 获取四种预测情况的样本索引
    tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]  # True Positive
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]  # False Negative
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]  # False Positive
    tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]  # True Negative
    
    # 选择每种情况的前2个样本进行分析
    sample_indices = {
        'TP': tp_indices[:2] if len(tp_indices) >= 2 else tp_indices,
        'FN': fn_indices[:2] if len(fn_indices) >= 2 else fn_indices,
        'FP': fp_indices[:2] if len(fp_indices) >= 2 else fp_indices,
        'TN': tn_indices[:2] if len(tn_indices) >= 2 else tn_indices
    }
    
    print("\nSelected samples for analysis:")
    for case, indices in sample_indices.items():
        print(f"{case}: {len(indices)} samples")
    
    # 1. Shapley Values (SV) 分析
    print("\nCalculating Shapley Values (SV)...")
    sv_explainer = tabpfn_shapiq.get_tabpfn_explainer(
        model=clf,
        data=X_train,
        labels=y_train,
        index="SV",
        verbose=True
    )
    
    # 分析每种情况的样本
    for case, indices in sample_indices.items():
        for i, idx in enumerate(indices):
            x_explain = X_test[idx]
            shapley_values = sv_explainer.explain(x=x_explain, budget=n_model_evals)
            
            # 获取基准值和特征重要性
            base_value = shapley_values.values[0]  # 第一个值是基准值
            feature_importance = shapley_values.values[1:]  # 其余值是特征重要性
            
            # 保存force plot
            plt.figure(figsize=(15, 5))
            shapley_values.plot_force(feature_names=feature_names)
            plt.title(f"{dataset_name} - {case} Sample {i+1} Shapley Values")
            plt.tight_layout()
            plt.savefig(f"{shapiq_path}/{dataset_name}_sv_force_plot_{case.lower()}_{i+1}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 打印SHAP值信息
            print(f"\n{case} Sample {i+1} SHAP values:")
            print(f"Base value: {base_value:.6f}")
            print("Feature importance:")
            for j, (name, value) in enumerate(zip(feature_names, feature_importance)):
                print(f"{name}: {value:.6f}")
    
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
    
    # 分析每种情况的样本交互
    for case, indices in sample_indices.items():
        for i, idx in enumerate(indices):
            x_explain = X_test[idx]
            shapley_interaction_values = fsii_explainer.explain(x=x_explain, budget=n_model_evals)
            
            # 保存upset plot
            plt.figure(figsize=(15, 10))
            shapley_interaction_values.plot_upset(feature_names=feature_names)
            plt.title(f"{dataset_name} - {case} Sample {i+1} Feature Interactions")
            plt.tight_layout()
            plt.savefig(f"{shapiq_path}/{dataset_name}_fsii_upset_plot_{case.lower()}_{i+1}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\n{dataset_name} Dataset SHAP-IQ Analysis Completed. Results saved in: {shapiq_path}/")

def run_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    dataset_name,
    device='cuda',
    random_state=42,
    base_path='./results'
):
    """运行实验"""
    # 创建结果目录
    os.makedirs(base_path, exist_ok=True)
    
    # 生成实验名称
    exp_name = f"TabPFN-{dataset_name}-R{random_state}"
    
    print("Training Data Shape:", X_train.shape)
    print("Test Data Shape:", X_test.shape)
    print("Training Label Distribution:\n", pd.Series(y_train).value_counts())
    print("Test Label Distribution:\n", pd.Series(y_test).value_counts())
    
    # 检查特征名称中是否包含标签列
    if 'Label' in feature_names:
        print("Warning: 'Label' found in feature names. Removing it...")
        feature_names = [f for f in feature_names if f != 'Label']
        print(f"Updated feature names: {feature_names}")
    
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
    
    # 打印原始预测概率
    print("\n原始预测概率示例:")
    
    # 获取四种预测情况的样本索引
    tp_indices = np.where((y_test_values == 1) & (y_pred == 1))[0]  # True Positive
    fn_indices = np.where((y_test_values == 1) & (y_pred == 0))[0]  # False Negative
    fp_indices = np.where((y_test_values == 0) & (y_pred == 1))[0]  # False Positive
    tn_indices = np.where((y_test_values == 0) & (y_pred == 0))[0]  # True Negative
    
    # 选择每种情况的前2个样本展示
    sample_indices = {
        'TP': tp_indices[:2] if len(tp_indices) >= 2 else tp_indices,
        'FN': fn_indices[:2] if len(fn_indices) >= 2 else fn_indices,
        'FP': fp_indices[:2] if len(fp_indices) >= 2 else fp_indices,
        'TN': tn_indices[:2] if len(tn_indices) >= 2 else tn_indices
    }
    
    for case, indices in sample_indices.items():
        print(f"\n{case} 样本的预测概率:")
        for i, idx in enumerate(indices):
            pos_prob = y_pred_proba[idx, 1]  # 正类的概率
            neg_prob = y_pred_proba[idx, 0]  # 负类的概率
            print(f"{case} 样本 {i+1} (索引 {idx}):")
            print(f"  预测概率: [负类: {neg_prob:.6f}, 正类: {pos_prob:.6f}]")
            print(f"  预测标签: {y_pred[idx]}")
            print(f"  真实标签: {y_test_values[idx]}")
    
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
        y_test=y_test_values,
        y_pred=y_pred,
        feature_names=feature_names,
        dataset_name=dataset_name,
        base_path=base_path
    )
    
    return metrics

if __name__ == "__main__":
    # 加载数据集
    print("\nLoading datasets...")
    print("1. Loading AI4healthcare.xlsx...")
    df_ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    
    print("2. Loading HenanCancerHospital_features63_58.xlsx...")
    df_henan = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    # 获取两个数据集的特征列表（明确排除Label列）
    features_ai4health = [col for col in df_ai4health.columns if col != 'Label']
    features_henan = [col for col in df_henan.columns if col != 'Label']
    
    # 找出共有的特征
    common_features = list(set(features_ai4health) & set(features_henan))
    print(f"\nNumber of common features: {len(common_features)}")
    print("Common features:", common_features)
    
    # 准备训练数据（B数据集）
    X_train = df_henan[common_features].copy()
    y_train = df_henan["Label"].copy()
    
    # 准备测试数据（A数据集）
    X_test = df_ai4health[common_features].copy()
    y_test = df_ai4health["Label"].copy()
    
    # 打印数据形状以确认
    print("\nData shapes after preparation:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 设置结果目录
    base_path = './results_B_train_A_test'
    os.makedirs(base_path, exist_ok=True)
    
    # 运行实验
    print("\n\n=== Cross-Dataset Analysis (B->A) ===")
    metrics = run_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=common_features,
        dataset_name="B_train_A_test",
        device='cuda',
        random_state=42,
        base_path=base_path
    )
    
    # 保存数据集信息
    datasets_info = pd.DataFrame({
        'Dataset': ['B (Training)', 'A (Testing)'],
        'Samples': [len(X_train), len(X_test)],
        'Features': [len(common_features), len(common_features)],
        'Positive_Samples': [sum(y_train), sum(y_test)],
        'Negative_Samples': [len(y_train)-sum(y_train), len(y_test)-sum(y_test)]
    })
    
    print("\nDataset Information:")
    print(datasets_info.to_string(index=False))
    datasets_info.to_csv(f'{base_path}/datasets_info.csv', index=False)
    
    # 保存结果
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(f'{base_path}/summary_results.csv', index=False)
    print("\nResults saved to:", base_path)