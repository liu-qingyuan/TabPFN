import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import logging
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 设置日志
logging.disable(logging.INFO)

def calculate_risk_score(data):
    """
    计算风险评分
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含所需特征的数据框
    
    Returns:
    --------
    np.array
        风险评分
    """
    risk_score = (-1.137 + 
                  0.036 * data['Feature2'] +
                  0.380 * data['Feature5'] +
                  0.195 * data['Feature48'] +
                  0.016 * data['Feature49'] -
                  0.290 * data['Feature50'] +
                  0.026 * data['Feature52'] -
                  0.168 * data['Feature56'] -
                  0.236 * data['Feature57'] +
                  0.052 * data['Feature61'] +
                  0.018 * data['Feature42'] +
                  0.004 * data['Feature43'])
    return risk_score

def calculate_probability(risk_score):
    """
    根据风险评分计算恶性概率
    
    Parameters:
    -----------
    risk_score : np.array
        风险评分
    
    Returns:
    --------
    np.array
        恶性概率
    """
    return np.exp(risk_score) / (1 + np.exp(risk_score))

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    # 加载训练数据
    print("\n加载训练数据 (AI4healthcare.xlsx)...")
    train_df = pd.read_excel("data/AI4healthcare.xlsx")

    # 加载预测数据
    print("\n加载预测数据 (GuangzhouMedicalHospital_features23.xlsx)...")
    test_df = pd.read_excel("data/GuangzhouMedicalHospital_features23.xlsx")

    # 定义用于风险评分的特征
    risk_features = [
        'Feature2', 'Feature5', 'Feature48', 'Feature49', 
        'Feature50', 'Feature52', 'Feature56', 'Feature57', 
        'Feature61', 'Feature42', 'Feature43'
    ]

    print("\n选定的特征:", risk_features)
    print("特征数量:", len(risk_features))

    # 分析数据分布
    print("\n=== 数据分布分析 ===")
    print("\n训练集描述性统计：")
    print(train_df[risk_features].describe())
    
    print("\n测试集描述性统计：")
    print(test_df[risk_features].describe())
    
    # 计算每个特征的分布差异
    print("\n特征分布差异（均值差异）：")
    for feature in risk_features:
        train_mean = train_df[feature].mean()
        test_mean = test_df[feature].mean()
        diff_percent = ((test_mean - train_mean) / train_mean) * 100
        print(f"{feature}: {diff_percent:.2f}%")

    print("\n训练数据形状:", train_df[risk_features].shape)
    print("测试数据形状:", test_df[risk_features].shape)
    print("\n训练数据标签分布:\n", train_df["Label"].value_counts())
    print("\n测试数据标签分布:\n", test_df["Label"].value_counts())

    # 计算训练集的风险评分和概率
    print("\n计算训练集风险评分和概率...")
    train_risk_scores = calculate_risk_score(train_df)
    train_probabilities = calculate_probability(train_risk_scores)
    train_predictions = (train_probabilities >= 0.5).astype(int)

    # 计算测试集的风险评分和概率
    print("\n计算测试集风险评分和概率...")
    test_risk_scores = calculate_risk_score(test_df)
    test_probabilities = calculate_probability(test_risk_scores)
    test_predictions = (test_probabilities >= 0.5).astype(int)

    # 评估训练集性能
    print("\n训练集性能评估:")
    train_acc = accuracy_score(train_df["Label"], train_predictions)
    train_auc = roc_auc_score(train_df["Label"], train_probabilities)
    train_f1 = f1_score(train_df["Label"], train_predictions)
    
    train_conf_matrix = confusion_matrix(train_df["Label"], train_predictions)
    train_acc_0 = train_conf_matrix[0, 0] / (train_conf_matrix[0, 0] + train_conf_matrix[0, 1])
    train_acc_1 = train_conf_matrix[1, 1] / (train_conf_matrix[1, 0] + train_conf_matrix[1, 1])
    
    print(f"训练集准确率 (Accuracy): {train_acc:.4f}")
    print(f"训练集 AUC: {train_auc:.4f}")
    print(f"训练集 F1分数: {train_f1:.4f}")
    print(f"训练集类别0准确率: {train_acc_0:.4f}")
    print(f"训练集类别1准确率: {train_acc_1:.4f}")

    # 评估测试集性能
    print("\n测试集性能评估:")
    test_acc = accuracy_score(test_df["Label"], test_predictions)
    test_auc = roc_auc_score(test_df["Label"], test_probabilities)
    test_f1 = f1_score(test_df["Label"], test_predictions)
    
    test_conf_matrix = confusion_matrix(test_df["Label"], test_predictions)
    test_acc_0 = test_conf_matrix[0, 0] / (test_conf_matrix[0, 0] + test_conf_matrix[0, 1])
    test_acc_1 = test_conf_matrix[1, 1] / (test_conf_matrix[1, 0] + test_conf_matrix[1, 1])

    print(f"测试集准确率 (Accuracy): {test_acc:.4f}")
    print(f"测试集 AUC: {test_auc:.4f}")
    print(f"测试集 F1分数: {test_f1:.4f}")
    print(f"测试集类别0准确率: {test_acc_0:.4f}")
    print(f"测试集类别1准确率: {test_acc_1:.4f}")

    # 保存结果
    results_dir = './results/LR'
    features23_dir = f'{results_dir}/features23'
    os.makedirs(features23_dir, exist_ok=True)
    
    results = pd.DataFrame({
        'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1'],
        'Train': [train_auc, train_f1, train_acc, train_acc_0, train_acc_1],
        'Test': [test_auc, test_f1, test_acc, test_acc_0, test_acc_1]
    })
    
    results.to_csv(f'{features23_dir}/cross_hospital_prediction_results.csv', index=False)
    print(f"\n结果已保存至: {features23_dir}/cross_hospital_prediction_results.csv")

    # 打印训练集和测试集的性能对比
    print("\n=== 训练集和测试集性能对比 ===")
    print(results.to_string())

    # 保存预测结果
    predictions_df = pd.DataFrame({
        'True_Label': test_df["Label"],
        'Predicted_Label': test_predictions,
        'Probability': test_probabilities,
        'Risk_Score': test_risk_scores
    })
    predictions_df.to_csv(f'{features23_dir}/predictions.csv', index=False)
    print(f"\n预测结果已保存至: {features23_dir}/predictions.csv") 