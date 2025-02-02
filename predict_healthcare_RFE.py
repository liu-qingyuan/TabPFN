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
from tabpfn import TabPFNClassifier
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

    def __init__(self, device='cuda', n_estimators=32, softmax_temperature=0.9,
                 balance_probabilities=False, average_before_softmax=False,
                 ignore_pretraining_limits=True, random_state=42,
                 n_repeats=5):
        self.device = device
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.random_state = random_state
        self.n_repeats = n_repeats

    def fit(self, X, y):
        # 先设置类别信息，以便评分函数可以访问
        self.classes_ = np.unique(y)
        
        # 初始化 TabPFN 模型
        self.model_ = TabPFNClassifier(
            device=self.device,
            n_estimators=self.n_estimators,
            softmax_temperature=self.softmax_temperature,
            balance_probabilities=self.balance_probabilities,
            average_before_softmax=self.average_before_softmax,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        
        # 使用置换重要性计算各个特征的重要性
        result = permutation_importance(
            self, X, y, 
            scoring='roc_auc',
            n_repeats=self.n_repeats,
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

def select_features_rfe(X, y, n_features=11):
    """
    使用TabPFN作为基础模型的RFE进行特征选择
    """
    n_features_total = X.shape[1]
    n_iterations = n_features_total - n_features
    
    # 初始化TabPFN包装器
    base_model = TabPFNWrapper(
        device='cuda',
        n_estimators=32,
        softmax_temperature=0.9,
        balance_probabilities=False,
        average_before_softmax=False,
        ignore_pretraining_limits=True,
        random_state=42
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
    exp_name = f"TabPFN-Health-RFE11-N{n_estimators}-S{softmax_temperature}-B{balance_probabilities}-A{average_before_softmax}-I{ignore_pretraining_limits}-R{random_state}"
    
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
        clf = TabPFNClassifier(
            device=device,
            n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            balance_probabilities=balance_probabilities,
            average_before_softmax=average_before_softmax,
            ignore_pretraining_limits=ignore_pretraining_limits,
            random_state=random_state
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
# 1. Load Data
df = pd.read_excel("data/AI4healthcare.xlsx")
features = [c for c in df.columns if c.startswith("Feature")]
X = df[features].copy()
y = df["Label"].copy()

# 2. Select Features using TabPFN importance
print("Selecting features using TabPFN importance...")
selected_features, feature_ranking = select_features_rfe(X, y, n_features=11)

# Save feature ranking
feature_ranking.to_csv('./results/RFE_feature_ranking.csv', index=False)
print("\nSelected Features:")
print("\n".join(selected_features))

# 3. Use selected features
X_selected = X[selected_features]

# 4. Run experiments with selected features
experiments = [
    {
        'device': 'cuda',
        'n_estimators': 32,
        'softmax_temperature': 0.9,
        'balance_probabilities': False,
        'average_before_softmax': False,
        'ignore_pretraining_limits': True,
        'random_state': 42
    }
]

# Run experiments and track best model
results = []
best_auc = 0
best_model = None
best_params = None

for params in experiments:
    print(f"\nRunning experiment with parameters: {params}")
    print("=" * 50)
    result_df, model = run_experiment(X_selected, y, **params)
    mean_auc = result_df['auc'].mean()
    
    # Track results
    results.append({
        'params': params,
        'mean_auc': mean_auc,
        'mean_acc': result_df['accuracy'].mean(),
        'mean_f1': result_df['f1'].mean(),
        'mean_time': result_df['time'].mean()
    })
    
    # Update best model if current is better
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model = model
        best_params = params

# Save best model and parameters
if best_model is not None:
    best_model_info = {
        'model': best_model,
        'parameters': best_params,
        'selected_features': selected_features,
        'auc_score': best_auc,
        'timestamp': time.strftime("%Y%m%d-%H%M%S")
    }
    
    save_path = './results/best_model_rfe'
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(best_model_info, f'{save_path}/TabPFN-Health-RFE11-Best-Model.joblib')
    
    with open(f'{save_path}/TabPFN-Health-RFE11-Best-Parameters.txt', 'w') as f:
        f.write("Best Model Parameters:\n")
        f.write("=" * 50 + "\n")
        f.write(f"AUC Score: {best_auc:.4f}\n")
        f.write(f"Timestamp: {best_model_info['timestamp']}\n")
        f.write("\nParameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        f.write("\nSelected Features:\n")
        for feature in selected_features:
            f.write(f"{feature}\n")

print("\nFeature Selection Results saved to: ./results/RFE_feature_ranking.csv")
print("Best Model saved to: ./results/best_model_rfe/") 