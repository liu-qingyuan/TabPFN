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
from tabpfn_extensions.rf_pfn import RandomForestTabPFNClassifier
from tabpfn_extensions import TabPFNClassifier
from tqdm import tqdm
import torch
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(42)
random.seed(42)

class RFTabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for RandomForestTabPFNClassifier to use with sklearn's RFE
    """
    _estimator_type = "classifier"

    def __init__(self,
                 device='cuda',
                 random_state=42,
                 n_repeats=5):
        self.device = device
        self.random_state = random_state
        self.n_repeats = n_repeats

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        base_model = TabPFNClassifier(device=self.device)
        self.model_ = RandomForestTabPFNClassifier(
            tabpfn=base_model,
            random_state=self.random_state
        )
        self.model_.fit(X, y)
        
        # Call permutation_importance on the underlying model
        result = permutation_importance(
            self.model_,  # Use the underlying RandomForestTabPFNClassifier
            X, 
            y,
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
os.makedirs('./results/rf_rfe_analysis', exist_ok=True)

# Define features to exclude
excluded_features = ['Feature12', 'Feature33', 'Feature34', 'Feature36', 'Feature40']

print("\nLoading datasets...")
# Load training data
train_df = pd.read_excel("data/AI4healthcare.xlsx")
# Load external test data
test_df = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")

# Get all features except Label and excluded features
all_features = [col for col in train_df.columns if col != 'Label' and col not in excluded_features]

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

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

print("\nData Information:")
print("Training Data Shape:", X_train_scaled.shape)
print("Test Data Shape:", X_test_scaled.shape)
print("\nTraining Label Distribution:\n", pd.Series(y_train_np).value_counts())
print("Test Label Distribution:\n", pd.Series(y_test_np).value_counts())

# Get feature ranking using RFE
print("\nGetting feature ranking using RFE...")
base_model = RFTabPFNWrapper(device='cuda', random_state=42)
rfe = RFE(
    estimator=base_model,
    n_features_to_select=1,
    step=1,
    verbose=1
)

# 一次性完成所有特征选择
rfe.fit(X_train_scaled, y_train_np)

# Create feature ranking DataFrame
feature_ranks = pd.DataFrame({
    'Feature': all_features,
    'Rank': rfe.ranking_
}).sort_values('Rank')

print("\nFeature Ranking:")
print(feature_ranks)

# Save feature ranking
feature_ranks.to_csv('./results/rf_rfe_analysis/feature_ranking.csv', index=False)

# Initialize variables to store results
all_results = []

# Evaluate different feature subsets based on ranking
print("\nEvaluating feature subsets...")
for n_features in tqdm(range(1, len(all_features) + 1), desc='Evaluating feature subsets'):
    # Get top k features
    selected_features = feature_ranks.nsmallest(n_features, 'Rank')['Feature'].tolist()
    selected_indices = [all_features.index(f) for f in selected_features]
    
    # Get selected features data
    X_train_selected = X_train_scaled[:, selected_indices]
    X_test_selected = X_test_scaled[:, selected_indices]
    
    # Train and evaluate model
    base_model = TabPFNClassifier(device='cuda')
    model = RandomForestTabPFNClassifier(
        tabpfn=base_model,
        random_state=42
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
results_df.to_csv('./results/rf_rfe_analysis/feature_selection_results.csv', index=False)

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
plt.savefig('./results/rf_rfe_analysis/performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nResults have been saved to:")
print("1. ./results/rf_rfe_analysis/feature_ranking.csv")
print("2. ./results/rf_rfe_analysis/feature_selection_results.csv")
print("3. ./results/rf_rfe_analysis/performance_analysis.png") 