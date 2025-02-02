try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import KFold
import time

# 定义要使用的特征
selected_features = [
    'Feature2', 'Feature5', 'Feature39', 'Feature40', 'Feature42',
    'Feature43', 'Feature46', 'Feature48', 'Feature56', 'Feature61',
    'Feature63'
]

# 加载数据
print("Loading data...")
df = pd.read_excel("data/AI4healthcare.xlsx")
X = df[selected_features].copy()  # 只使用选中的特征
y = df["Label"].copy()

print("\nData Shape:", X.shape)
print("Label Distribution:\n", y.value_counts())

# 加载最佳模型
print("\nLoading best model...")
best_model_info = joblib.load('./results/best_model_rfe/TabPFN-Health-RFE11-Best-Model.joblib')

# 访问组件
model = best_model_info['model']
parameters = best_model_info['parameters']
auc_score = best_model_info['auc_score']

print("\nBest Model Parameters:")
print("=" * 50)
print(f"Best AUC Score: {auc_score:.4f}")
for key, value in parameters.items():
    print(f"{key}: {value}")

print("\nSelected Features:")
for feature in selected_features:
    print(feature)

# 10折交叉验证
print("\nPerforming 10-fold cross validation...")
kf = KFold(n_splits=10, shuffle=True, random_state=42)
results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    start_time = time.time()
    
    # 获取训练集和测试集
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # 计算指标
    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities[:, 1])
    f1 = f1_score(y_test, predictions)
    
    # 计算每个类别的准确率
    class_0_mask = (y_test == 0)
    class_1_mask = (y_test == 1)
    acc_0 = accuracy_score(y_test[class_0_mask], predictions[class_0_mask])
    acc_1 = accuracy_score(y_test[class_1_mask], predictions[class_1_mask])
    
    # 计算时间
    elapsed_time = time.time() - start_time
    
    # 保存结果
    results.append({
        'fold': fold,
        'accuracy': acc,
        'auc': auc,
        'f1': f1,
        'acc_0': acc_0,
        'acc_1': acc_1,
        'time': elapsed_time
    })
    
    # 打印当前折的结果
    print(f"\nFold {fold}")
    print("-" * 50)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Class 0 Accuracy: {acc_0:.4f}")
    print(f"Class 1 Accuracy: {acc_1:.4f}")
    print(f"Time: {elapsed_time:.4f}s")

# 计算平均结果
results_df = pd.DataFrame(results)
print("\nFinal Results:")
print(f"Average Test AUC: {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
print(f"Average Test F1: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
print(f"Average Test ACC: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
print(f"Average Test ACC_0: {results_df['acc_0'].mean():.4f} ± {results_df['acc_0'].std():.4f}")
print(f"Average Test ACC_1: {results_df['acc_1'].mean():.4f} ± {results_df['acc_1'].std():.4f}")
print(f"Average Time: {results_df['time'].mean():.4f} ± {results_df['time'].std():.4f}") 