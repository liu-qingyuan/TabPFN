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
from tabpfn_extensions.hpo import TunedTabPFNClassifier
import joblib

# 设置日志级别
logging.getLogger("AutoPostHocEnsemble").setLevel(logging.WARNING)

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
TunedTabPFNClassifier.__sklearn_tags__ = __sklearn_tags__

def run_experiment(
    X,
    y,
    device='cuda',
    random_state=42,
    base_path='./results'
):
    """
    Run TunedTabPFN experiment with given parameters
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    device : str
        Device to use for computation ('cuda' or 'cpu')
    random_state : int
        Random state for reproducibility
    base_path : str
        Base path for saving results
    
    Returns:
    --------
    tuple
        (scores_df, clf) - DataFrame with scores and trained classifier
    """
    # Create results directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate experiment name based on parameters
    exp_name = f"TunedTabPFN-Health-R{random_state}"
    
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
        
        # Convert data to numpy arrays
        X_train_np = X_train.values
        y_train_np = y_train.values
        X_test_np = X_test.values
        
        # Initialize and train model
        clf = TunedTabPFNClassifier(
            device=device,
            random_state=random_state,
            n_trials=50  # 设置优化试验次数
        )

        clf.fit(X_train_np, y_train_np)
        y_pred_proba = clf.predict_proba(X_test_np)
        y_pred = np.argmax(y_pred_proba, axis=1)
            
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
    
    if not fold_scores:
        raise RuntimeError("No successful folds completed")
    
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
    
    # Train a final model on the full dataset for saving
    print("\nTraining final model on full dataset...")
    final_clf = TunedTabPFNClassifier(
        device=device,
        random_state=random_state,
        n_trials=50
    )
    final_clf.fit(X.values, y.values)
    
    return scores_df, final_clf

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    # 1. Load Data
    df = pd.read_excel("data/AI4healthcare.xlsx")

    # Define selected features (removing duplicate Feature46)
    selected_features = [
        'Feature63', 'Feature2', 'Feature46', 'Feature61', 'Feature56', 
        'Feature42', 'Feature39', 'Feature43', 'Feature48', 'Feature5', 
        'Feature22'
    ]

    # Print feature information
    print("\nSelected features:", selected_features)
    print("Number of selected features:", len(selected_features))

    # Prepare data with selected features
    X = df[selected_features].copy()
    y = df["Label"].copy()

    # Run experiment with TunedTabPFNClassifier
    print("\nRunning experiment with TunedTabPFNClassifier...")
    result_df, final_model = run_experiment(
        X=X,
        y=y,
        device='cuda',
        random_state=42,
        base_path='./results/tuned_model'
    )

    # Save final model and parameters
    best_model_info = {
        'model': final_model,
        'features': selected_features,
        'cv_auc_score': result_df['auc'].mean(),
        'cv_accuracy': result_df['accuracy'].mean(),
        'cv_f1': result_df['f1'].mean(),
        'timestamp': time.strftime("%Y%m%d-%H%M%S")
    }
    
    # Save to file
    save_path = './results/tuned_model'
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(best_model_info, f'{save_path}/TunedTabPFN-Health-Model.joblib')
    
    # Save parameters in readable format
    with open(f'{save_path}/TunedTabPFN-Health-Parameters.txt', 'w') as f:
        f.write("Model Parameters:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Number of Features: {len(selected_features)}\n")
        f.write(f"Features: {', '.join(selected_features)}\n")
        f.write(f"Cross-validation AUC Score: {result_df['auc'].mean():.4f} ± {result_df['auc'].std():.4f}\n")
        f.write(f"Cross-validation Accuracy: {result_df['accuracy'].mean():.4f} ± {result_df['accuracy'].std():.4f}\n")
        f.write(f"Cross-validation F1 Score: {result_df['f1'].mean():.4f} ± {result_df['f1'].std():.4f}\n")
        f.write(f"Timestamp: {best_model_info['timestamp']}\n")

print("\nResults have been saved to:")
print("1. ./results/tuned_model/") 