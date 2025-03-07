import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import logging

# logging.getLogger("AutoPostHocEnsemble").setLevel(logging.WARNING)
logging.disable(logging.INFO)      # Completely disable INFO and below logs
# logging.disable(logging.WARNING)   # If you don't even want the Warning level

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from tqdm import tqdm
import torch
import random

# 动态添加 sklearn 所需的属性
from types import SimpleNamespace

def __sklearn_tags__(self):
    # 如果定义了 _more_tags() 方法，则使用它返回的字典，否则返回空字典
    tags = self._more_tags() if hasattr(self, '_more_tags') else {}
    # 确保至少包含 requires_fit 键，默认值为 True
    tags.setdefault("requires_fit", True)
    # 将字典包装为 SimpleNamespace，这样可以通过属性方式访问
    return SimpleNamespace(**tags)

# 使用猴子补丁覆盖原来的 __sklearn_tags__ 方法
AutoTabPFNClassifier.__sklearn_tags__ = __sklearn_tags__

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(42)
random.seed(42)

class AutoTabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for AutoTabPFNClassifier to use with sklearn's RFE
    """
    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        return SimpleNamespace(
            estimator_type="classifier",
            binary_only=True,
            classifier_tags=SimpleNamespace(poor_score=False),
            regressor_tags=SimpleNamespace(poor_score=False),
            input_tags=SimpleNamespace(sparse=False, allow_nan=True),
            target_tags=SimpleNamespace(required=True)
        )

    def __init__(self,
                 device='cuda',
                 random_state=42,
                 n_repeats=5,
                 max_time=1):
        self.device = device
        self.random_state = random_state
        self.n_repeats = n_repeats
        self.max_time = max_time

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        self.model_ = AutoTabPFNClassifier(
            device=self.device,
            random_state=self.random_state,
            max_time=self.max_time
        )
        self.model_.fit(X, y)
        
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

# Create results directory
os.makedirs('./results/auto_rfe_analysis_guangzhou_no_nan', exist_ok=True)

# Define selected features (使用与 predict_healthcare_auto_predict_all.py 相同的特征列表)
selected_features = [
    'Feature2', 'Feature3', 'Feature4', 'Feature5',
    'Feature14', 'Feature15', 'Feature17', 'Feature22',
    'Feature39', 'Feature42', 'Feature43', 'Feature45',
    'Feature46', 'Feature47', 'Feature48', 'Feature49',
    'Feature50', 'Feature52', 'Feature53', 'Feature56',
    'Feature57', 'Feature63'
]

print("\nLoading datasets...")
# Load training data
train_df = pd.read_excel("data/AI4healthcare.xlsx")
# Load external test data
test_df = pd.read_excel("data/GuangzhouMedicalHospital_features22_no_nan.xlsx")

# 使用预定义的特征列表，而不是动态生成
all_features = selected_features

# Prepare data
X_train = train_df[all_features].copy()
y_train = train_df["Label"].copy()
X_test = test_df[all_features].copy()
y_test = test_df["Label"].copy()

# Convert to numpy arrays and ensure correct data types
X_train_np = X_train.values.astype(np.float32)
y_train_np = y_train.values.astype(np.int32)
X_test_np = X_test.values.astype(np.float32)
y_test_np = y_test.values.astype(np.int32)

print("\nData Information:")
print("Training Data Shape:", X_train_np.shape)
print("Test Data Shape:", X_test_np.shape)
print("\nTraining Label Distribution:\n", pd.Series(y_train_np).value_counts())
print("Test Label Distribution:\n", pd.Series(y_test_np).value_counts())

# Get feature ranking using RFE
print("\nGetting feature ranking using RFE...")
base_model = AutoTabPFNWrapper(device='cuda', random_state=42, max_time=1)
rfe = RFE(
    estimator=base_model,
    n_features_to_select=1,
    step=1,
    verbose=1
)

# 一次性完成所有特征选择
rfe.fit(X_train_np, y_train_np)

# Create feature ranking DataFrame
feature_ranks = pd.DataFrame({
    'Feature': all_features,
    'Rank': rfe.ranking_
}).sort_values('Rank')

print("\nFeature Ranking:")
print(feature_ranks)

# Save feature ranking
feature_ranks.to_csv('./results/auto_rfe_analysis_guangzhou_no_nan/feature_ranking.csv', index=False)

# Initialize variables to store results
all_results = []

# Evaluate different feature subsets based on ranking
print("\nEvaluating feature subsets...")
for n_features in tqdm(range(1, len(all_features) + 1), desc='Evaluating feature subsets'):
    # Get top k features
    selected_features = feature_ranks.nsmallest(n_features, 'Rank')['Feature'].tolist()
    selected_indices = [all_features.index(f) for f in selected_features]
    
    # Get selected features data
    X_train_selected = X_train_np[:, selected_indices]
    X_test_selected = X_test_np[:, selected_indices]
    
    # Train and evaluate model
    model = AutoTabPFNClassifier(
        device='cuda',
        random_state=42,
        max_time=1
    )
    model.fit(X_train_selected, y_train_np)
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test_selected)
    y_test_proba = model.predict_proba(X_test_selected)
    
    # Calculate metrics
    test_acc = accuracy_score(y_test_np, y_test_pred)
    test_auc = roc_auc_score(y_test_np, y_test_proba[:, 1])
    test_f1 = f1_score(y_test_np, y_test_pred)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test_np, y_test_pred)
    test_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    test_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    
    # Store results
    result = {
        'n_features': n_features,
        'features': selected_features,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_acc_0': test_acc_0,
        'test_acc_1': test_acc_1,
        'confusion_matrix': conf_matrix.tolist()
    }
    all_results.append(result)
    
    # Print current results without breaking progress bar
    tqdm.write(f"\nFeatures: {n_features}")
    tqdm.write(f"Test AUC: {test_auc:.4f}")
    tqdm.write(f"Test Accuracy: {test_acc:.4f}")
    tqdm.write(f"Test F1: {test_f1:.4f}")
    tqdm.write(f"Class 0 Accuracy: {test_acc_0:.4f}")
    tqdm.write(f"Class 1 Accuracy: {test_acc_1:.4f}")

# Convert results to DataFrame
results_df = pd.DataFrame(all_results)

# Sort results by test AUC
results_df = results_df.sort_values('test_auc', ascending=False)

# Save results
results_df.to_csv('./results/auto_rfe_analysis_guangzhou_no_nan/feature_selection_results.csv', index=False)

# Print best configuration
print("\nBest Configuration:")
print("=" * 50)
best_result = results_df.iloc[0]
print(f"Number of features: {best_result['n_features']}")
print(f"Test AUC: {best_result['test_auc']:.4f}")
print(f"Test Accuracy: {best_result['test_acc']:.4f}")
print(f"Test F1: {best_result['test_f1']:.4f}")
print(f"Test Class 0 Accuracy: {best_result['test_acc_0']:.4f}")
print(f"Test Class 1 Accuracy: {best_result['test_acc_1']:.4f}")
print("\nBest features:", best_result['features'])

# Create visualization
plt.figure(figsize=(12, 6))

# Plot performance metrics vs number of features
plt.subplot(1, 2, 1)
plt.plot(results_df['n_features'], results_df['test_auc'], 'o-', label='AUC')
plt.plot(results_df['n_features'], results_df['test_acc'], 'o-', label='Accuracy')
plt.plot(results_df['n_features'], results_df['test_f1'], 'o-', label='F1')
plt.title('Performance vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Score')
plt.legend()
plt.grid(True)

# Plot class-wise accuracy vs number of features
plt.subplot(1, 2, 2)
plt.plot(results_df['n_features'], results_df['test_acc_0'], 'o-', label='Class 0')
plt.plot(results_df['n_features'], results_df['test_acc_1'], 'o-', label='Class 1')
plt.title('Class-wise Accuracy vs Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('./results/auto_rfe_analysis_guangzhou_no_nan/performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults have been saved to:")
print("1. ./results/auto_rfe_analysis_guangzhou_no_nan/feature_ranking.csv")
print("2. ./results/auto_rfe_analysis_guangzhou_no_nan/feature_selection_results.csv")
print("3. ./results/auto_rfe_analysis_guangzhou_no_nan/performance_analysis.png") 