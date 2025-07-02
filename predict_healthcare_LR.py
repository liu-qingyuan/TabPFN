import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import os
import logging
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# 设置日志
logging.disable(logging.INFO)

def calculate_risk_score(data: pd.DataFrame) -> np.ndarray:
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
    return risk_score.values

def calculate_probability(risk_score: np.ndarray) -> np.ndarray:
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
    # 加载数据集A (AI4health)
    print("\n加载数据集A (AI4healthcare.xlsx)...")
    dataset_A = pd.read_excel("data/AI4healthcare.xlsx")

    # 加载数据集B (河南)
    print("\n加载数据集B (HenanCancerHospital_features63_58.xlsx)...")
    dataset_B = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")

    # 加载数据集C (广州)
    print("\n加载数据集C (GuangzhouMedicalHospital_features24.xlsx)...")
    dataset_C = pd.read_excel("data/GuangzhouMedicalHospital_features24.xlsx")

    # 定义用于风险评分的特征
    risk_features = [
        'Feature2', 'Feature5', 'Feature48', 'Feature49', 
        'Feature50', 'Feature52', 'Feature56', 'Feature57', 
        'Feature61', 'Feature42', 'Feature43'
    ]

    print("\n选定的特征:", risk_features)
    print("特征数量:", len(risk_features))

    # 创建数据集字典
    datasets = {
        'A_AI4health': dataset_A,
        'B_Henan': dataset_B,
        'C_Guangzhou': dataset_C
    }

    # 分析数据分布
    print("\n=== 数据分布分析 ===")
    for name, df in datasets.items():
        print(f"\n{name} 数据集描述性统计：")
        print(df[risk_features].describe())
        print(f"{name} 数据形状:", df[risk_features].shape)
        print(f"{name} 标签分布:\n", df["Label"].value_counts())

    # 保存结果
    results_dir = './results/LR'
    features23_dir = f'{results_dir}/features23'
    os.makedirs(features23_dir, exist_ok=True)

    # 存储所有结果
    all_results = []
    all_predictions = {}

    # 对每个数据集分别进行LR评估
    for dataset_name, df in datasets.items():
        print(f"\n=== 评估数据集 {dataset_name} ===")
        
        # 计算风险评分和概率
        risk_scores = calculate_risk_score(df)
        probabilities = calculate_probability(risk_scores)
        predictions = (probabilities >= 0.5).astype(int)

        # 评估性能
        acc = accuracy_score(df["Label"], predictions)
        auc = roc_auc_score(df["Label"], probabilities)
        f1 = f1_score(df["Label"], predictions)
        
        conf_matrix = confusion_matrix(df["Label"], predictions)
        acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])

        print(f"{dataset_name} 准确率 (Accuracy): {acc:.4f}")
        print(f"{dataset_name} AUC: {auc:.4f}")
        print(f"{dataset_name} F1分数: {f1:.4f}")
        print(f"{dataset_name} 类别0准确率: {acc_0:.4f}")
        print(f"{dataset_name} 类别1准确率: {acc_1:.4f}")

        # 保存单个数据集结果
        dataset_results = pd.DataFrame({
            'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1'],
            dataset_name: [auc, f1, acc, acc_0, acc_1]
        })
        
        all_results.append({
            'name': dataset_name,
            'auc': auc,
            'f1': f1,
            'acc': acc,
            'acc_0': acc_0,
            'acc_1': acc_1
        })
        
        # 保存预测结果
        predictions_df = pd.DataFrame({
            'True_Label': df["Label"],
            'Predicted_Label': predictions,
            'Probability': probabilities,
            'Risk_Score': risk_scores
        })
        
        all_predictions[dataset_name] = predictions_df
        
        # 保存到文件
        dataset_results.to_csv(f'{features23_dir}/{dataset_name}_results.csv', index=False)
        predictions_df.to_csv(f'{features23_dir}/{dataset_name}_predictions.csv', index=False)
        
        print(f"结果已保存至: {features23_dir}/{dataset_name}_results.csv")
        print(f"预测结果已保存至: {features23_dir}/{dataset_name}_predictions.csv")

    # 创建综合结果对比
    print("\n=== 三个数据集性能对比 ===")
    summary_results = pd.DataFrame({
        'Metric': ['AUC', 'F1', 'ACC', 'ACC_0', 'ACC_1'],
        'A_AI4health': [all_results[0]['auc'], all_results[0]['f1'], all_results[0]['acc'], 
                        all_results[0]['acc_0'], all_results[0]['acc_1']],
        'B_Henan': [all_results[1]['auc'], all_results[1]['f1'], all_results[1]['acc'], 
                    all_results[1]['acc_0'], all_results[1]['acc_1']],
        'C_Guangzhou': [all_results[2]['auc'], all_results[2]['f1'], all_results[2]['acc'], 
                        all_results[2]['acc_0'], all_results[2]['acc_1']]
    })
    
    print(summary_results.to_string())
    summary_results.to_csv(f'{features23_dir}/ABC_comparison_results.csv', index=False)
    print(f"\n综合结果已保存至: {features23_dir}/ABC_comparison_results.csv") 