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
import joblib  # Add this import for model saving

# Print TabPFNClassifier parameters
print("TabPFNClassifier parameters:")
print(TabPFNClassifier.__init__.__doc__)

def run_experiment(
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
    exp_name = f"TabPFN-Health-N{n_estimators}-S{softmax_temperature}-B{balance_probabilities}-A{average_before_softmax}-I{ignore_pretraining_limits}-R{random_state}"
    
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
    # 3. Summary Results
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
    plt.savefig(f'{base_path}/{exp_name}-Plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return scores_df, clf  # Return both scores and model

# ==============================
# Run Multiple Experiments
# ==============================
# Define parameter combinations to try
experiments = [
    # Test different n_estimators
    {
        'device': 'cuda',
        'n_estimators': 16,
        'softmax_temperature': 0.9,
        'balance_probabilities': False,
        'average_before_softmax': False,
        'ignore_pretraining_limits': True,
        'random_state': 42
    },
    {
        'device': 'cuda',
        'n_estimators': 32,
        'softmax_temperature': 0.9,
        'balance_probabilities': False,
        'average_before_softmax': False,
        'ignore_pretraining_limits': True,
        'random_state': 42
    },
    {
        'device': 'cuda',
        'n_estimators': 64,
        'softmax_temperature': 0.9,
        'balance_probabilities': False,
        'average_before_softmax': False,
        'ignore_pretraining_limits': True,
        'random_state': 42
    },
    # Test different softmax temperatures
    {
        'device': 'cuda',
        'n_estimators': 32,
        'softmax_temperature': 0.5,
        'balance_probabilities': False,
        'average_before_softmax': False,
        'ignore_pretraining_limits': True,
        'random_state': 42
    },
    {
        'device': 'cuda',
        'n_estimators': 32,
        'softmax_temperature': 1.0,
        'balance_probabilities': False,
        'average_before_softmax': False,
        'ignore_pretraining_limits': True,
        'random_state': 42
    },
    # Test probability balancing and averaging
    {
        'device': 'cuda',
        'n_estimators': 32,
        'softmax_temperature': 0.9,
        'balance_probabilities': True,
        'average_before_softmax': False,
        'ignore_pretraining_limits': True,
        'random_state': 42
    },
    {
        'device': 'cuda',
        'n_estimators': 32,
        'softmax_temperature': 0.9,
        'balance_probabilities': False,
        'average_before_softmax': True,
        'ignore_pretraining_limits': True,
        'random_state': 42
    }
]

# Run all experiments and track best model
results = []
best_auc = 0
best_model = None
best_params = None

for params in experiments:
    print(f"\nRunning experiment with parameters: {params}")
    print("=" * 50)
    result_df, model = run_experiment(**params)
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
    # Create a dictionary with all relevant information
    best_model_info = {
        'model': best_model,
        'parameters': best_params,
        'auc_score': best_auc,
        'timestamp': time.strftime("%Y%m%d-%H%M%S")
    }
    
    # Save to file
    save_path = './results/best_model'
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(best_model_info, f'{save_path}/TabPFN-Health-Best-Model.joblib')
    
    # Save parameters to a more readable format
    with open(f'{save_path}/TabPFN-Health-Best-Parameters.txt', 'w') as f:
        f.write("Best Model Parameters:\n")
        f.write("=" * 50 + "\n")
        f.write(f"AUC Score: {best_auc:.4f}\n")
        f.write(f"Timestamp: {best_model_info['timestamp']}\n")
        f.write("\nParameters:\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")

# Save overall comparison
comparison_df = pd.DataFrame([{
    'n_estimators': r['params']['n_estimators'],
    'softmax_temperature': r['params']['softmax_temperature'],
    'balance_probabilities': r['params']['balance_probabilities'],
    'average_before_softmax': r['params']['average_before_softmax'],
    'ignore_pretraining_limits': r['params']['ignore_pretraining_limits'],
    'random_state': r['params']['random_state'],
    'AUC': r['mean_auc'],
    'ACC': r['mean_acc'],
    'F1': r['mean_f1'],
    'Time': r['mean_time']
} for r in results])

comparison_df.to_csv('./results/TabPFN-Health-Comparison.csv', index=False)
print("\nParameter Comparison:")
print(comparison_df)

# Print best model information
print("\nBest Model:")
print("=" * 50)
print(f"AUC Score: {best_auc:.4f}")
for key, value in best_params.items():
    print(f"{key}: {value}") 