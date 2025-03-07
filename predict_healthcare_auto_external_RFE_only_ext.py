import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
import joblib
from types import SimpleNamespace
from tqdm import tqdm

class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"  # 作为类属性声明这是一个分类器
    
    def __sklearn_tags__(self):
        # 返回一个包含所有 RFE 代码可能访问的字段的对象
        return SimpleNamespace(
            estimator_type="classifier",
            binary_only=True,
            classifier_tags=SimpleNamespace(poor_score=False),
            regressor_tags=SimpleNamespace(poor_score=False),
            input_tags=SimpleNamespace(sparse=False, allow_nan=True),
            target_tags=SimpleNamespace(required=True)
        )

    def __init__(self, device='cuda', random_state=42, max_time=1):  # 简化参数列表
        self.device = device
        self.random_state = random_state
        self.max_time = max_time  # 添加 max_time 属性

    def fit(self, X, y):
        # 先设置类别信息，以便评分函数可以访问
        self.classes_ = np.unique(y)
        
        # 初始化 TabPFN 模型
        self.model_ = AutoTabPFNClassifier(
            device=self.device,
            random_state=self.random_state,
            max_time=self.max_time  # 使用 max_time 参数
        )
        self.model_.fit(X, y)
        
        # 使用置换重要性计算各个特征的重要性
        result = permutation_importance(
            self, X, y, 
            scoring='roc_auc',
            n_repeats=5,  # 使用固定值
            random_state=self.random_state
        )
        self.feature_importances_ = result.importances_mean
        self.feature_importances_std_ = result.importances_std
        
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
    def score(self, X, y):
        y_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_proba)
    
    def get_feature_importance(self):
        """返回特征重要性"""
        return self.feature_importances_

    def get_feature_importance_scores(self):
        """返回特征重要性分数和标准差"""
        return {
            'mean': self.feature_importances_,
            'std': self.feature_importances_std_
        }

def select_features_rfe(X, y, n_features=3):
    """
    使用TabPFN作为基础模型的RFE进行特征选择
    """
    n_features_total = X.shape[1]
    n_iterations = n_features_total - n_features
    
    # 初始化TabPFN包装器
    base_model = TabPFNWrapper(
        device='cuda',
        random_state=42,
        max_time=1
    )
    
    # 初始化RFE
    rfe = RFE(
        estimator=base_model,
        n_features_to_select=n_features,
        step=1,
        verbose=2  # 启用详细输出
    )
    
    # 创建进度条
    print("Fitting RFE with TabPFN as base model...")
    with tqdm(total=n_iterations, desc='Eliminating features') as pbar:
        # 拟合RFE
        rfe.fit(X, y)
        pbar.update(n_iterations)
    
    # 获取选中的特征
    selected_features = X.columns[rfe.support_].tolist()
    
    # 获取特征重要性排名
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Rank': rfe.ranking_
    }).sort_values('Rank')
    
    return selected_features, feature_ranking

def run_experiment(
    X,
    y,
    device='cuda',
    n_estimators=32,
    softmax_temperature=0.9,
    balance_probabilities=False,
    average_before_softmax=False,
    ignore_pretraining_limits=True,
    random_state=42,
    base_path='./results'
):
    """
    Run TabPFN experiment with given parameters
    """
    # Create results directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate experiment name based on parameters
    exp_name = f"TabPFN-Health-RFE3-N{n_estimators}-S{softmax_temperature}-B{balance_probabilities}-A{average_before_softmax}-I{ignore_pretraining_limits}-R{random_state}"
    
    print("Data Shape:", X.shape)
    print("Label Distribution:\n", y.value_counts())
    
    # ==============================
    # Cross Validation
    # ==============================
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        print(f"\nFold {fold}")
        print("-" * 50)
        
        # Initialize and train model with parameters
        start_time = time.time()
        clf = AutoTabPFNClassifier(
            device=device,
            random_state=random_state,
            max_time=1  # 添加 max_time 参数
        )
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
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
    
    return scores_df, clf

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    # Create results directory
    os.makedirs('./results/auto_rfe_analysis_guangzhou_no_nan_only_ext', exist_ok=True)

    # Define selected features
    selected_features = [
        'Feature2', 'Feature3', 'Feature4', 'Feature5',
        'Feature14', 'Feature15', 'Feature17', 'Feature22',
        'Feature39', 'Feature42', 'Feature43', 'Feature45',
        'Feature46', 'Feature47', 'Feature48', 'Feature49',
        'Feature50', 'Feature52', 'Feature53', 'Feature56',
        'Feature57', 'Feature63'
    ]

    print("\nLoading dataset...")
    # 只加载广州数据集
    df = pd.read_excel("data/GuangzhouMedicalHospital_features22_no_nan.xlsx")

    # 准备数据
    X = df[selected_features].copy()
    y = df["Label"].copy()

    # 转换为numpy数组
    X_np = X.values.astype(np.float32)
    y_np = y.values.astype(np.int32)

    print("\nData Information:")
    print("Data Shape:", X_np.shape)
    print("\nLabel Distribution:\n", pd.Series(y_np).value_counts())

    # Get feature ranking using RFE
    print("\nGetting feature ranking using RFE...")
    base_model = TabPFNWrapper(device='cuda', random_state=42, max_time=1)
    rfe = RFE(
        estimator=base_model,
        n_features_to_select=1,
        step=1,
        verbose=1
    )

    # 一次性完成所有特征选择
    rfe.fit(X_np, y_np)

    # Create feature ranking DataFrame
    feature_ranks = pd.DataFrame({
        'Feature': selected_features,
        'Rank': rfe.ranking_
    }).sort_values('Rank')

    print("\nFeature Ranking:")
    print(feature_ranks)

    # Save feature ranking
    feature_ranks.to_csv('./results/auto_rfe_analysis_guangzhou_no_nan_only_ext/feature_ranking.csv', index=False)

    # Initialize variables to store results
    all_results = []

    # Evaluate different feature subsets based on ranking
    print("\nEvaluating feature subsets...")
    for n_features in tqdm(range(1, len(selected_features) + 1), desc='Evaluating feature subsets'):
        # Get top k features
        selected_features_subset = feature_ranks.nsmallest(n_features, 'Rank')['Feature'].tolist()
        selected_indices = [selected_features.index(f) for f in selected_features_subset]
        
        # Get selected features data
        X_selected = X_np[:, selected_indices]
        
        # 使用5折交叉验证评估性能
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_selected), 1):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y_np[train_idx], y_np[test_idx]
            
            # Train and evaluate model
            model = AutoTabPFNClassifier(
                device='cuda',
                random_state=42,
                max_time=1  # 添加 max_time 参数
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            f1 = f1_score(y_test, y_pred)
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
            acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
            
            fold_metrics.append({
                'acc': acc,
                'auc': auc,
                'f1': f1,
                'acc_0': acc_0,
                'acc_1': acc_1
            })
        
        # Calculate average metrics
        avg_metrics = {
            'test_acc': np.mean([m['acc'] for m in fold_metrics]),
            'test_auc': np.mean([m['auc'] for m in fold_metrics]),
            'test_f1': np.mean([m['f1'] for m in fold_metrics]),
            'test_acc_0': np.mean([m['acc_0'] for m in fold_metrics]),
            'test_acc_1': np.mean([m['acc_1'] for m in fold_metrics])
        }
        
        # Store results
        result = {
            'n_features': n_features,
            'features': selected_features_subset,
            **avg_metrics
        }
        all_results.append(result)
        
        # Print current results
        tqdm.write(f"\nFeatures: {n_features}")
        tqdm.write(f"CV AUC: {avg_metrics['test_auc']:.4f}")
        tqdm.write(f"CV Accuracy: {avg_metrics['test_acc']:.4f}")
        tqdm.write(f"CV F1: {avg_metrics['test_f1']:.4f}")
        tqdm.write(f"CV Class 0 Accuracy: {avg_metrics['test_acc_0']:.4f}")
        tqdm.write(f"CV Class 1 Accuracy: {avg_metrics['test_acc_1']:.4f}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Sort results by test AUC
    results_df = results_df.sort_values('test_auc', ascending=False)

    # Save results
    results_df.to_csv('./results/auto_rfe_analysis_guangzhou_no_nan_only_ext/feature_selection_results.csv', index=False)

    # ... 保持可视化代码不变，只修改输出路径 ...
    plt.savefig('./results/auto_rfe_analysis_guangzhou_no_nan_only_ext/performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nResults have been saved to:")
    print("1. ./results/auto_rfe_analysis_guangzhou_no_nan_only_ext/feature_ranking.csv")
    print("2. ./results/auto_rfe_analysis_guangzhou_no_nan_only_ext/feature_selection_results.csv")
    print("3. ./results/auto_rfe_analysis_guangzhou_no_nan_only_ext/performance_analysis.png") 