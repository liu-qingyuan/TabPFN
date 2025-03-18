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

# 定义PKUPH和Mayo模型
class PKUPHModel:
    """
    PKUPH模型的实现
    P(malignant) = e^x / (1+e^x)
    x = -4.496 + (0.07 × Feature2) + (0.676 × Feature48) + (0.736 × Feature49) + 
        (1.267 × Feature4) - (1.615 × Feature50) - (1.408 × Feature53)
    """
    def __init__(self):
        self.intercept_ = -4.496
        self.features = ['Feature2', 'Feature48', 'Feature49', 'Feature4', 'Feature50', 'Feature53']
        self.coefficients = {
            'Feature2': 0.07,
            'Feature48': 0.676,
            'Feature49': 0.736,
            'Feature4': 1.267,
            'Feature50': -1.615,
            'Feature53': -1.408
        }
        
    def fit(self, X, y):
        # 模型已经预定义，不需要训练
        return self
        
    def predict_proba(self, X):
        # 确保X是DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
            
        # 计算线性组合
        x = np.zeros(len(X))
        x += self.intercept_
        
        for feature, coef in self.coefficients.items():
            if feature in X.columns:
                x += coef * X[feature].values
            
        # 计算概率
        p_malignant = 1 / (1 + np.exp(-x))
        
        # 返回两列概率 [P(benign), P(malignant)]
        return np.column_stack((1 - p_malignant, p_malignant))
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class MayoModel:
    """
    Mayo模型的实现
    P(malignant) = e^x / (1+e^x)
    x = -6.8272 + (0.0391 × Feature2) + (0.7917 × Feature3) + (1.3388 × Feature5) + 
        (0.1274 × Feature48) + (1.0407 × Feature49) + (0.7838 × Feature63)
    """
    def __init__(self):
        self.intercept_ = -6.8272
        self.features = ['Feature2', 'Feature3', 'Feature5', 'Feature48', 'Feature49', 'Feature63']
        self.coefficients = {
            'Feature2': 0.0391,
            'Feature3': 0.7917,
            'Feature5': 1.3388,
            'Feature48': 0.1274,
            'Feature49': 1.0407,
            'Feature63': 0.7838
        }
        
    def fit(self, X, y):
        # 模型已经预定义，不需要训练
        return self
        
    def predict_proba(self, X):
        # 确保X是DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.features)
            
        # 计算线性组合
        x = np.zeros(len(X))
        x += self.intercept_
        
        for feature, coef in self.coefficients.items():
            if feature in X.columns:
                x += coef * X[feature].values
            
        # 计算概率
        p_malignant = 1 / (1 + np.exp(-x))
        
        # 返回两列概率 [P(benign), P(malignant)]
        return np.column_stack((1 - p_malignant, p_malignant))
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

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

def run_experiment_with_model(
    X,
    y,
    model_name,
    model_constructor,
    model_params={},
    base_path='./results'
):
    """
    使用指定模型运行10折交叉验证实验
    
    Parameters:
    -----------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标变量
    model_name : str
        模型名称
    model_constructor : class
        模型构造函数
    model_params : dict
        模型参数
    base_path : str
        结果保存路径
    
    Returns:
    --------
    pd.DataFrame
        包含交叉验证分数的DataFrame
    """
    # 创建结果目录（如果不存在）
    os.makedirs(base_path, exist_ok=True)
    
    # 生成基于模型名称的实验名称
    exp_name = f"{model_name}-Experiment"
    
    print(f"\n=== {model_name} 模型 ===")
    print("数据形状:", X.shape)
    print("标签分布:\n", y.value_counts())
    
    # 转换数据为numpy数组
    X_values = X.copy()
    y_values = y.values.astype(np.int32)
    
    # ==============================
    # 交叉验证
    # ==============================
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_values), 1):
        X_train, X_test = X_values.iloc[train_idx], X_values.iloc[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        
        print(f"\n折 {fold}")
        print("-" * 50)
        
        # 初始化并训练模型
        start_time = time.time()
        model = model_constructor(**model_params)
        model.fit(X_train, y_train)
        
        # 进行预测
        y_pred_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        fold_time = time.time() - start_time
        
        # 计算指标
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        f1 = f1_score(y_test, y_pred)
        
        # 计算每类的准确率
        conf_matrix = confusion_matrix(y_test, y_pred)
        acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        
        print(f"准确率: {acc:.4f}")
        print(f"ROC AUC: {auc:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"类别0准确率: {acc_0:.4f}")
        print(f"类别1准确率: {acc_1:.4f}")
        print(f"时间: {fold_time:.4f}秒")
        
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
    # 汇总结果
    # ==============================
    scores_df = pd.DataFrame(fold_scores)
    
    # 保存结果
    scores_df.to_csv(f'{base_path}/{exp_name}.csv', index=False)
    
    # 计算并保存最终结果
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
    
    # 打印结果
    print("\n最终结果:")
    print(f"平均测试AUC: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f}")
    print(f"平均测试F1: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f}")
    print(f"平均测试准确率: {scores_df['accuracy'].mean():.4f} ± {scores_df['accuracy'].std():.4f}")
    print(f"平均测试类别0准确率: {scores_df['acc_0'].mean():.4f} ± {scores_df['acc_0'].std():.4f}")
    print(f"平均测试类别1准确率: {scores_df['acc_1'].mean():.4f} ± {scores_df['acc_1'].std():.4f}")
    print(f"平均时间: {scores_df['time'].mean():.4f} ± {scores_df['time'].std():.4f}")
    
    # ==============================
    # 可视化结果
    # ==============================
    plt.figure(figsize=(15, 5))
    
    # 绘制指标
    plt.subplot(1, 3, 1)
    metrics = ['accuracy', 'auc', 'f1']
    for metric in metrics:
        plt.plot(scores_df['fold'], scores_df[metric], 'o-', label=metric.upper())
        plt.axhline(y=scores_df[metric].mean(), linestyle='--', alpha=0.3)
    plt.title('不同折的表现指标')
    plt.xlabel('折')
    plt.ylabel('分数')
    plt.legend()
    
    # 绘制每类准确率
    plt.subplot(1, 3, 2)
    plt.plot(scores_df['fold'], scores_df['acc_0'], 'bo-', label='类别0')
    plt.plot(scores_df['fold'], scores_df['acc_1'], 'ro-', label='类别1')
    plt.axhline(y=scores_df['acc_0'].mean(), color='b', linestyle='--', alpha=0.3)
    plt.axhline(y=scores_df['acc_1'].mean(), color='r', linestyle='--', alpha=0.3)
    plt.title('不同折的各类准确率')
    plt.xlabel('折')
    plt.ylabel('准确率')
    plt.legend()
    
    # 绘制时间
    plt.subplot(1, 3, 3)
    plt.plot(scores_df['fold'], scores_df['time'], 'go-', label='时间')
    plt.axhline(y=scores_df['time'].mean(), color='g', linestyle='--', alpha=0.3)
    plt.title('不同折的计算时间')
    plt.xlabel('折')
    plt.ylabel('时间（秒）')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{base_path}/{exp_name}-Plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return scores_df

# ==============================
# 主程序
# ==============================
if __name__ == "__main__":
    # 加载所有数据集
    print("\n加载数据集...")
    print("1. 加载 AI4healthcare.xlsx (A)...")
    df_ai4health = pd.read_excel("data/AI4healthcare.xlsx")
    
    print("2. 加载 HenanCancerHospital_features63_58.xlsx (B)...")
    df_henan = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")
    
    print("3. 加载 GuangzhouMedicalHospital_features23_no_nan.xlsx (C)...")
    df_guangzhou = pd.read_excel("data/GuangzhouMedicalHospital_features23_no_nan.xlsx")

    # 使用指定的23个特征（添加了Feature1）
    selected_features = [
        'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5',
        'Feature14', 'Feature15', 'Feature17', 'Feature22',
        'Feature39', 'Feature42', 'Feature43', 'Feature45',
        'Feature46', 'Feature47', 'Feature48', 'Feature49',
        'Feature50', 'Feature52', 'Feature53', 'Feature56',
        'Feature57', 'Feature63'
    ]

    print("\n=== 特征信息 ===")
    print(f"选择的特征数量: {len(selected_features)}")
    print("选择的特征列表:", selected_features)

    # 检查每个数据集中是否有所有选定的特征
    for dataset_name, dataset in [
        ("AI4health", df_ai4health), 
        ("Henan", df_henan), 
        ("Guangzhou", df_guangzhou)
    ]:
        missing_features = [f for f in selected_features if f not in dataset.columns]
        if missing_features:
            print(f"警告: {dataset_name} 缺少以下特征: {missing_features}")
        else:
            print(f"{dataset_name} 包含所有选定的特征")

    # 使用共同特征准备数据
    X_ai4health = df_ai4health[selected_features].copy()
    y_ai4health = df_ai4health["Label"].copy()
    
    X_henan = df_henan[selected_features].copy()
    y_henan = df_henan["Label"].copy()
    
    X_guangzhou = df_guangzhou[selected_features].copy()
    y_guangzhou = df_guangzhou["Label"].copy()

    # 创建结果目录
    os.makedirs('./results_auto_and_otherbaselines_ABC_features23', exist_ok=True)

    # 使用多个模型并比较结果
    models = [
        {
            'name': 'AutoTabPFN',
            'constructor': AutoTabPFNClassifier,
            'params': {'device': 'cuda', 'max_time': 60, 'random_state': 42}
        },
        {
            'name': 'PKUPH',
            'constructor': PKUPHModel,
            'params': {}
        },
        {
            'name': 'Mayo',
            'constructor': MayoModel,
            'params': {}
        }
    ]

    # 存储所有模型在所有数据集上的结果
    all_results = {}

    # 对每个数据集运行所有模型
    for dataset_name, X, y in [
        ("A_AI4health", X_ai4health, y_ai4health),
        ("B_Henan", X_henan, y_henan),
        ("C_Guangzhou", X_guangzhou, y_guangzhou)
    ]:
        print(f"\n\n{'='*50}")
        print(f"数据集: {dataset_name}")
        print(f"{'='*50}")
        
        dataset_results = {}
        
        for model_info in models:
            result_dir = f"./results_auto_and_otherbaselines_ABC_features23/{dataset_name}"
            scores_df = run_experiment_with_model(
                X, 
                y, 
                model_info['name'],
                model_info['constructor'],
                model_info['params'],
                base_path=result_dir
            )
            
            dataset_results[model_info['name']] = {
                'auc': scores_df['auc'].mean(),
                'auc_std': scores_df['auc'].std(),
                'f1': scores_df['f1'].mean(),
                'f1_std': scores_df['f1'].std(),
                'acc': scores_df['accuracy'].mean(),
                'acc_std': scores_df['accuracy'].std(),
                'acc_0': scores_df['acc_0'].mean(),
                'acc_0_std': scores_df['acc_0'].std(),
                'acc_1': scores_df['acc_1'].mean(),
                'acc_1_std': scores_df['acc_1'].std(),
                'time': scores_df['time'].mean(),
                'time_std': scores_df['time'].std()
            }
        
        all_results[dataset_name] = dataset_results

    # 创建一个汇总表格
    print("\n\n=============== 汇总结果 ===============")
    
    # 每个数据集的汇总表
    for dataset_name, results in all_results.items():
        # AUC比较
        auc_comparison = pd.DataFrame({
            'Model': [model for model in results.keys()],
            'AUC': [results[model]['auc'] for model in results.keys()],
            'AUC_Std': [results[model]['auc_std'] for model in results.keys()],
            'ACC': [results[model]['acc'] for model in results.keys()],
            'ACC_Std': [results[model]['acc_std'] for model in results.keys()],
            'F1': [results[model]['f1'] for model in results.keys()],
            'F1_Std': [results[model]['f1_std'] for model in results.keys()],
            'ACC_0': [results[model]['acc_0'] for model in results.keys()],
            'ACC_1': [results[model]['acc_1'] for model in results.keys()],
            'Time': [results[model]['time'] for model in results.keys()]
        }).sort_values('AUC', ascending=False)
        
        print(f"\n=== {dataset_name} 数据集上的模型比较 ===")
        print(auc_comparison.to_string(index=False))
        
        # 保存结果
        auc_comparison.to_csv(f'./results_auto_and_otherbaselines_ABC_features23/{dataset_name}_models_comparison.csv', index=False)
    
    # 创建跨数据集的模型比较
    model_performance = {}
    for model_name in models[0]['name'], models[1]['name'], models[2]['name']:
        model_performance[model_name] = {
            'A_AI4health': all_results['A_AI4health'][model_name]['acc'],
            'B_Henan': all_results['B_Henan'][model_name]['acc'],
            'C_Guangzhou': all_results['C_Guangzhou'][model_name]['acc'],
            'A_AUC': all_results['A_AI4health'][model_name]['auc'],
            'B_AUC': all_results['B_Henan'][model_name]['auc'],
            'C_AUC': all_results['C_Guangzhou'][model_name]['auc']
        }
    
    model_comparison = pd.DataFrame(model_performance).T
    print("\n=== 不同数据集上的模型准确率比较 ===")
    print(model_comparison.to_string())
    
    # 保存跨数据集的比较结果
    model_comparison.to_csv('./results_auto_and_otherbaselines_ABC_features23/cross_dataset_comparison.csv')
    
    # 绘制比较图
    plt.figure(figsize=(12, 8))
    
    # 准确率比较
    plt.subplot(2, 1, 1)
    for i, model_name in enumerate([m['name'] for m in models]):
        datasets = ['A_AI4health', 'B_Henan', 'C_Guangzhou']
        accuracies = [all_results[d][model_name]['acc'] for d in datasets]
        plt.bar([x + i*0.25 for x in range(len(datasets))], accuracies, width=0.25, label=model_name)
    
    plt.xticks([i + 0.25 for i in range(len(datasets))], datasets)
    plt.ylabel('准确率')
    plt.title('各个数据集上的模型准确率比较')
    plt.legend()
    
    # AUC比较
    plt.subplot(2, 1, 2)
    for i, model_name in enumerate([m['name'] for m in models]):
        datasets = ['A_AI4health', 'B_Henan', 'C_Guangzhou']
        aucs = [all_results[d][model_name]['auc'] for d in datasets]
        plt.bar([x + i*0.25 for x in range(len(datasets))], aucs, width=0.25, label=model_name)
    
    plt.xticks([i + 0.25 for i in range(len(datasets))], datasets)
    plt.ylabel('AUC')
    plt.title('各个数据集上的模型AUC比较')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('./results_auto_and_otherbaselines_ABC_features23/model_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n所有结果已保存至 ./results_auto_and_otherbaselines_ABC_features23/ 目录")