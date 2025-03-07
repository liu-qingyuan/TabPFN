import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import logging

# logging.getLogger("AutoPostHocEnsemble").setLevel(logging.WARNING)
logging.disable(logging.INFO)      # Completely disable INFO and below logs
# logging.disable(logging.WARNING)   # If you don't even want the Warning level

# Create results directory
os.makedirs('./results/best_8_features', exist_ok=True)

# Define the best 8 features
best_features = [
    'Feature63', 'Feature2', 'Feature46', 'Feature61',
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]

print("\nEvaluating Best 8 Features:")
print(best_features)

# Load data
print("\nLoading datasets...")
train_df = pd.read_excel("data/AI4healthcare.xlsx")
external_df = pd.read_excel("data/HenanCancerHospital_features63_58.xlsx")

# Prepare data
X_train = train_df[best_features].copy()
y_train = train_df["Label"].copy()
X_external = external_df[best_features].copy()
y_external = external_df["Label"].copy()

print("\nData Information:")
print("Training Data Shape:", X_train.shape)
print("External Data Shape:", X_external.shape)
print("\nTraining Label Distribution:\n", y_train.value_counts())
print("External Label Distribution:\n", y_external.value_counts())

# Apply standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_external_scaled = scaler.transform(X_external)

# ==============================
# 1. 10-fold Cross Validation on Training Data
# ==============================
print("\n10-fold Cross Validation Results:")
print("=" * 50)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
    X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Train model
    model = AutoTabPFNClassifier(device='cuda', max_time=2, random_state=42)
    model.fit(X_fold_train, y_fold_train)
    
    # Evaluate
    y_val_pred = model.predict(X_fold_val)
    y_val_proba = model.predict_proba(X_fold_val)
    
    # Calculate metrics
    fold_acc = accuracy_score(y_fold_val, y_val_pred)
    fold_auc = roc_auc_score(y_fold_val, y_val_proba[:, 1])
    fold_f1 = f1_score(y_fold_val, y_val_pred)
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_fold_val, y_val_pred)
    fold_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    fold_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    
    cv_scores.append({
        'fold': fold,
        'accuracy': fold_acc,
        'auc': fold_auc,
        'f1': fold_f1,
        'acc_0': fold_acc_0,
        'acc_1': fold_acc_1
    })
    
    print(f"\nFold {fold}:")
    print(f"Accuracy: {fold_acc:.4f}")
    print(f"AUC: {fold_auc:.4f}")
    print(f"F1: {fold_f1:.4f}")
    print(f"Class 0 Accuracy: {fold_acc_0:.4f}")
    print(f"Class 1 Accuracy: {fold_acc_1:.4f}")

# Calculate and print average cross-validation scores
cv_df = pd.DataFrame(cv_scores)
print("\nAverage Cross-validation Results:")
print("=" * 50)
print(f"Accuracy: {cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}")
print(f"AUC: {cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}")
print(f"F1: {cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}")
print(f"Class 0 Accuracy: {cv_df['acc_0'].mean():.4f} ± {cv_df['acc_0'].std():.4f}")
print(f"Class 1 Accuracy: {cv_df['acc_1'].mean():.4f} ± {cv_df['acc_1'].std():.4f}")

# ==============================
# 2. External Validation
# ==============================
print("\nExternal Validation Results:")
print("=" * 50)

# Train final model on full training data
final_model = AutoTabPFNClassifier(device='cuda', max_time=2, random_state=42)
final_model.fit(X_train_scaled, y_train)

# Evaluate on external data
y_external_pred = final_model.predict(X_external_scaled)
y_external_proba = final_model.predict_proba(X_external_scaled)

# Calculate metrics
external_acc = accuracy_score(y_external, y_external_pred)
external_auc = roc_auc_score(y_external, y_external_proba[:, 1])
external_f1 = f1_score(y_external, y_external_pred)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_external, y_external_pred)
external_acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
external_acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])

print("\nExternal Validation Metrics:")
print(f"Accuracy: {external_acc:.4f}")
print(f"AUC: {external_auc:.4f}")
print(f"F1: {external_f1:.4f}")
print(f"Class 0 Accuracy: {external_acc_0:.4f}")
print(f"Class 1 Accuracy: {external_acc_1:.4f}")
print(f"\nConfusion Matrix:\n{conf_matrix}")

# Save results
results = {
    'cross_validation': {
        'accuracy': f"{cv_df['accuracy'].mean():.4f} ± {cv_df['accuracy'].std():.4f}",
        'auc': f"{cv_df['auc'].mean():.4f} ± {cv_df['auc'].std():.4f}",
        'f1': f"{cv_df['f1'].mean():.4f} ± {cv_df['f1'].std():.4f}",
        'acc_0': f"{cv_df['acc_0'].mean():.4f} ± {cv_df['acc_0'].std():.4f}",
        'acc_1': f"{cv_df['acc_1'].mean():.4f} ± {cv_df['acc_1'].std():.4f}"
    },
    'external_validation': {
        'accuracy': f"{external_acc:.4f}",
        'auc': f"{external_auc:.4f}",
        'f1': f"{external_f1:.4f}",
        'acc_0': f"{external_acc_0:.4f}",
        'acc_1': f"{external_acc_1:.4f}",
        'confusion_matrix': conf_matrix.tolist()
    }
}

# Save to file
with open('./results/best_8_features/results.txt', 'w') as f:
    f.write("Best 8 Features Evaluation Results\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("Features:\n")
    for i, feature in enumerate(best_features, 1):
        f.write(f"{i}. {feature}\n")
    
    f.write("\nCross-validation Results:\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['cross_validation'].items():
        f.write(f"{metric}: {value}\n")
    
    f.write("\nExternal Validation Results:\n")
    f.write("-" * 30 + "\n")
    for metric, value in results['external_validation'].items():
        if metric != 'confusion_matrix':
            f.write(f"{metric}: {value}\n")
    f.write(f"\nConfusion Matrix:\n{conf_matrix}")

print("\nResults have been saved to: ./results/best_8_features/results.txt") 