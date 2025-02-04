import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from tabpfn import TabPFNClassifier
import time
import os

def evaluate_feature_set(X, y, features, n_fold=10):
    """评估一个特征子集的性能"""
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X[features]), 1):
        start_time = time.time()
        
        # 获取训练集和测试集
        X_train = X.iloc[train_idx][features]
        X_test = X.iloc[test_idx][features]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        # 训练模型
        model = TabPFNClassifier(
            device='cuda',
            n_estimators=32,
            softmax_temperature=0.9,
            balance_probabilities=False,
            average_before_softmax=False,
            ignore_pretraining_limits=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # 计算指标
        fold_scores.append({
            'fold': fold,
            'auc': roc_auc_score(y_test, y_proba),
            'acc': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'time': time.time() - start_time
        })
        
        print(f"Fold {fold}: AUC={fold_scores[-1]['auc']:.4f}, "
              f"ACC={fold_scores[-1]['acc']:.4f}, "
              f"F1={fold_scores[-1]['f1']:.4f}, "
              f"Time={fold_scores[-1]['time']:.2f}s")
    
    # 计算平均分数
    scores_df = pd.DataFrame(fold_scores)
    mean_scores = scores_df.mean()
    std_scores = scores_df.std()
    
    return {
        'n_features': len(features),
        'features': features,
        'mean_auc': mean_scores['auc'],
        'std_auc': std_scores['auc'],
        'mean_acc': mean_scores['acc'],
        'std_acc': std_scores['acc'],
        'mean_f1': mean_scores['f1'],
        'std_f1': std_scores['f1'],
        'mean_time': mean_scores['time'],
        'std_time': std_scores['time']
    }

def rfe_with_cv(X, y, min_features=3, step=1):
    """使用交叉验证的RFE过程"""
    current_features = list(X.columns)
    history = []
    
    while len(current_features) >= min_features:
        print(f"\nEvaluating {len(current_features)} features:")
        print(", ".join(current_features))
        
        # 评估当前特征集
        result = evaluate_feature_set(X, y, current_features)
        history.append(result)
        
        # 如果达到最小特征数，停止
        if len(current_features) == min_features:
            break
            
        # 计算特征重要性
        model = TabPFNClassifier(
            device='cuda',
            n_estimators=32,
            random_state=42
        )
        model.fit(X[current_features], y)
        
        # 使用模型的特征重要性
        importances = np.zeros(len(current_features))
        for i in range(len(current_features)):
            X_temp = X[current_features].copy()
            X_temp[current_features[i]] = np.random.permutation(X_temp[current_features[i]])
            y_pred = model.predict_proba(X_temp)[:, 1]
            importances[i] = -roc_auc_score(y, y_pred)  # 负号表示重要性
            
        # 移除最不重要的特征
        worst_idx = np.argsort(importances)[:step]
        for idx in sorted(worst_idx, reverse=True):
            removed_feature = current_features[idx]
            print(f"Removing feature: {removed_feature}")
            del current_features[idx]
    
    return pd.DataFrame(history)

if __name__ == "__main__":
    # 加载数据
    print("Loading data...")
    df = pd.read_excel("data/AI4healthcare.xlsx")
    features = [c for c in df.columns if c.startswith("Feature")]
    X = df[features].copy()
    y = df["Label"].copy()
    
    # 运行RFE
    print("\nStarting RFE with cross-validation...")
    history = rfe_with_cv(X, y, min_features=3, step=1)
    
    # 保存结果
    os.makedirs('./results/rfe_cv', exist_ok=True)
    history.to_csv('./results/rfe_cv/feature_selection_history.csv', index=False)
    
    # 打印最终结果
    print("\nFeature Selection History:")
    print(history[['n_features', 'mean_auc', 'mean_acc', 'mean_f1']]) 