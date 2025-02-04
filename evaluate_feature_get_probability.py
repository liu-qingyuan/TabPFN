import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import time
import os
from tqdm import tqdm
import torch

# 创建保存结果的目录
os.makedirs("results/feature_number_evaluation", exist_ok=True)

# 读取数据
df = pd.read_excel("data/AI4healthcare.xlsx")
features = [c for c in df.columns if c.startswith("Feature")]
X = df[features].copy()
y = df["Label"].copy()

# 读取特征排名
feature_ranking = pd.read_csv("results/RFE_feature_ranking.csv")
ranked_features = feature_ranking['Feature'].tolist()

# 设置随机种子
np.random.seed(42)

# 定义要测试的特征数量
feature_numbers = list(range(6, 15))  # 测试6到14个特征

# 存储所有结果
all_results = []
# 存储预测概率
probability_results = []

# 对每个特征数量进行评估
for n_features in tqdm(feature_numbers, desc=f"Evaluating features (6-14)"):
    # 选择前n个特征
    selected_features = ranked_features[:n_features]
    X_selected = X[selected_features]
    
    # 10折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []
    
    # 存储当前特征数量的所有预测概率
    all_test_indices = []
    all_predictions = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_selected), 1):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 训练模型
        start_time = time.time()
        clf = TabPFNClassifier(
            device='cuda',
            n_estimators=32,
            softmax_temperature=0.9,
            balance_probabilities=False,
            average_before_softmax=False,
            ignore_pretraining_limits=True,
            random_state=42
        )
        
        # 确保每个fold使用相同的随机状态
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
        clf.fit(X_train, y_train)
        
        # 预测
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        fold_time = time.time() - start_time
        
        # 存储测试集索引和预测概率
        all_test_indices.extend(test_idx)
        all_predictions.extend(y_pred_proba[:, 1])  # 只保存正类的概率
        
        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        f1 = f1_score(y_test, y_pred)
        
        # 计算每个类别的准确率
        conf_matrix = confusion_matrix(y_test, y_pred)
        acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        
        fold_scores.append({
            'fold': fold,
            'accuracy': acc,
            'auc': auc,
            'f1': f1,
            'acc_0': acc_0,
            'acc_1': acc_1,
            'time': fold_time
        })
    
    # 将预测结果按原始索引排序
    sorted_indices = np.argsort(all_test_indices)
    sorted_predictions = np.array(all_predictions)[sorted_indices]
    
    # 存储预测概率
    probability_results.append({
        'n_features': n_features,
        'predictions': sorted_predictions
    })
    
    # 计算平均值和标准差
    metrics = ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1', 'time']
    mean_scores = {f'mean_{m}': np.mean([s[m] for s in fold_scores]) for m in metrics}
    std_scores = {f'std_{m}': np.std([s[m] for s in fold_scores]) for m in metrics}
    
    # 合并结果
    result = {
        'n_features': n_features,
        'features': ', '.join(selected_features),
        **mean_scores,
        **std_scores
    }
    all_results.append(result)
    
    # 打印当前结果
    print(f"\nResults for {n_features} features:")
    print(f"Features: {', '.join(selected_features)}")
    print(f"Mean AUC: {mean_scores['mean_auc']:.4f} ± {std_scores['std_auc']:.4f}")
    print(f"Mean ACC: {mean_scores['mean_accuracy']:.4f} ± {std_scores['std_accuracy']:.4f}")
    print(f"Mean F1: {mean_scores['mean_f1']:.4f} ± {std_scores['std_f1']:.4f}")
    print(f"Mean ACC_0: {mean_scores['mean_acc_0']:.4f} ± {std_scores['std_acc_0']:.4f}")
    print(f"Mean ACC_1: {mean_scores['mean_acc_1']:.4f} ± {std_scores['std_acc_1']:.4f}")
    print(f"Mean Time: {mean_scores['mean_time']:.4f}s ± {std_scores['std_time']:.4f}s")

# 将结果保存到CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("results/feature_number_evaluation/feature_number_comparison.csv", index=False)

# 创建预测概率的DataFrame
prob_data = {}
prob_data['Label'] = y  # 添加真实标签

# 为每个特征数量添加预测概率列
for prob_result in probability_results:
    n_features = prob_result['n_features']
    prob_data[f'Features_{n_features}'] = prob_result['predictions']

# 保存预测概率
prob_df = pd.DataFrame(prob_data)
prob_df.to_csv("results/feature_number_evaluation/prediction_probabilities.csv", index=False)

# 创建可视化
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 10))
metrics = ['auc', 'accuracy', 'f1']
colors = ['blue', 'green', 'red']

for metric, color in zip(metrics, colors):
    mean_values = [result[f'mean_{metric}'] for result in all_results]
    std_values = [result[f'std_{metric}'] for result in all_results]
    
    plt.plot(feature_numbers, mean_values, marker='o', color=color, label=metric.upper())
    plt.fill_between(
        feature_numbers,
        [m - s for m, s in zip(mean_values, std_values)],
        [m + s for m, s in zip(mean_values, std_values)],
        color=color,
        alpha=0.2
    )

plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.title('Model Performance vs Number of Features')
plt.legend()
plt.grid(True)
plt.savefig("results/feature_number_evaluation/performance_comparison.png")
plt.close()

print("\nResults have been saved to:")
print("1. results/feature_number_evaluation/feature_number_comparison.csv")
print("2. results/feature_number_evaluation/prediction_probabilities.csv")
print("3. results/feature_number_evaluation/performance_comparison.png") 