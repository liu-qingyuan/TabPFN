import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from tabpfn import TabPFNClassifier

# Create results directory if it doesn't exist
os.makedirs('./results', exist_ok=True)

# ==============================
# 1. Read Data
# ==============================
df = pd.read_excel("data/AI4healthcare.xlsx")
features = [c for c in df.columns if c.startswith("Feature")]
X = df[features].copy()
y = df["Label"].copy()

print("Data Shape:", X.shape)
print("Label Distribution:\n", y.value_counts())

# ==============================
# 2. Cross Validation
# ==============================
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"\nFold {fold}")
    print("-" * 50)
    
    # Initialize and train model
    start_time = time.time()
    clf = TabPFNClassifier(device='cuda')
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
# 3. Summary Results
# ==============================
scores_df = pd.DataFrame(fold_scores)

# Save per-fold results
scores_df.to_csv('./results/TabPFN-Health.csv', index=False)

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
final_results.to_csv('./results/TabPFN-Health-Final.csv', index=False)

# Print results
print("\nFinal Results:")
print(f"Average Test AUC: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f}")
print(f"Average Test F1: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f}")
print(f"Average Test ACC: {scores_df['accuracy'].mean():.4f} ± {scores_df['accuracy'].std():.4f}")
print(f"Average Test ACC_0: {scores_df['acc_0'].mean():.4f} ± {scores_df['acc_0'].std():.4f}")
print(f"Average Test ACC_1: {scores_df['acc_1'].mean():.4f} ± {scores_df['acc_1'].std():.4f}")
print(f"Average Time: {scores_df['time'].mean():.4f} ± {scores_df['time'].std():.4f}")

# ==============================
# 4. Visualize Results
# ==============================
plt.figure(figsize=(15, 5))

# Plot metrics
plt.subplot(1, 3, 1)
metrics = ['accuracy', 'auc', 'f1']
for metric in metrics:
    plt.plot(scores_df['fold'], scores_df[metric], 'o-', label=metric.upper())
    plt.axhline(y=scores_df[metric].mean(), linestyle='--', alpha=0.3)
plt.title('Performance Metrics across Folds')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.legend()

# Plot class-wise accuracy
plt.subplot(1, 3, 2)
plt.plot(scores_df['fold'], scores_df['acc_0'], 'bo-', label='Class 0')
plt.plot(scores_df['fold'], scores_df['acc_1'], 'ro-', label='Class 1')
plt.axhline(y=scores_df['acc_0'].mean(), color='b', linestyle='--', alpha=0.3)
plt.axhline(y=scores_df['acc_1'].mean(), color='r', linestyle='--', alpha=0.3)
plt.title('Class-wise Accuracy across Folds')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()

# Plot time
plt.subplot(1, 3, 3)
plt.plot(scores_df['fold'], scores_df['time'], 'go-', label='Time')
plt.axhline(y=scores_df['time'].mean(), color='g', linestyle='--', alpha=0.3)
plt.title('Computation Time across Folds')
plt.xlabel('Fold')
plt.ylabel('Time (seconds)')
plt.legend()

plt.tight_layout()
# Save the figure before showing it
plt.savefig('./results/TabPFN-Health-Plots.png', dpi=300, bbox_inches='tight')
plt.show() 