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
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.preprocessing import StandardScaler

# Disable logging
logging.disable(logging.INFO)

def evaluate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    conf_matrix = confusion_matrix(y_true, y_pred)
    return {
        'acc': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_pred_proba),
        'f1': f1_score(y_true, y_pred),
        'acc_0': conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]),
        'acc_1': conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    }

def print_metrics(dataset_name, metrics):
    """Print evaluation metrics"""
    print(f"{dataset_name} Accuracy: {metrics['acc']:.4f}")
    print(f"{dataset_name} AUC: {metrics['auc']:.4f}")
    print(f"{dataset_name} F1 Score: {metrics['f1']:.4f}")
    print(f"{dataset_name} Class 0 Accuracy: {metrics['acc_0']:.4f}")
    print(f"{dataset_name} Class 1 Accuracy: {metrics['acc_1']:.4f}")

def run_experiment(
    X,
    y,
    device='cuda',
    max_time=15,
    random_state=42,
    base_path='./results/combined_tabpfn'
):
    """
    Run AutoTabPFN experiment with given parameters
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    device : str
        Device to use for computation ('cuda' or 'cpu')
    max_time : int
        Maximum optimization time in seconds
    random_state : int
        Random state for reproducibility
    base_path : str
        Base path for saving results
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with cross-validation scores
    """
    # Create results directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate experiment name based on parameters
    exp_name = f"CombinedTabPFN-T{max_time}-R{random_state}"
    
    print("Data Shape:", X.shape)
    print("Label Distribution:\n", y.value_counts())
    
    # Convert data to numpy arrays
    X_values = X.values.astype(np.float32)
    y_values = y.values.astype(np.int32)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)
    
    # ==============================
    # Cross Validation
    # ==============================
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        
        print(f"\nFold {fold}")
        print("-" * 50)
        
        # Initialize and train model
        start_time = time.time()
        clf = AutoTabPFNClassifier(
            device=device,
            max_time=max_time,
            random_state=random_state
        )
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred_proba = clf.predict_proba(X_test)
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
    
    # ==============================
    # Visualize Results
    # ==============================
    plt.figure(figsize=(18, 10))
    
    # Plot 1: Performance metrics across folds
    plt.subplot(2, 2, 1)
    metrics = ['accuracy', 'auc', 'f1']
    for metric in metrics:
        plt.plot(scores_df['fold'], scores_df[metric], 'o-', label=metric.upper())
        plt.axhline(y=scores_df[metric].mean(), linestyle='--', alpha=0.3)
    plt.title('Performance Metrics across Folds', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Class-wise accuracy
    plt.subplot(2, 2, 2)
    plt.plot(scores_df['fold'], scores_df['acc_0'], 'bo-', label='Class 0')
    plt.plot(scores_df['fold'], scores_df['acc_1'], 'ro-', label='Class 1')
    plt.axhline(y=scores_df['acc_0'].mean(), color='b', linestyle='--', alpha=0.3)
    plt.axhline(y=scores_df['acc_1'].mean(), color='r', linestyle='--', alpha=0.3)
    plt.title('Class-wise Accuracy across Folds', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Time per fold
    plt.subplot(2, 2, 3)
    plt.plot(scores_df['fold'], scores_df['time'], 'go-', label='Time')
    plt.axhline(y=scores_df['time'].mean(), color='g', linestyle='--', alpha=0.3)
    plt.title('Computation Time across Folds', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Summarized performance metrics
    plt.subplot(2, 2, 4)
    metrics_mean = [scores_df['accuracy'].mean(), scores_df['auc'].mean(), 
                   scores_df['f1'].mean(), scores_df['acc_0'].mean(), 
                   scores_df['acc_1'].mean()]
    metrics_std = [scores_df['accuracy'].std(), scores_df['auc'].std(), 
                  scores_df['f1'].std(), scores_df['acc_0'].std(), 
                  scores_df['acc_1'].std()]
    metric_names = ['Accuracy', 'AUC', 'F1', 'Class 0 Acc', 'Class 1 Acc']
    
    bars = plt.bar(metric_names, metrics_mean, yerr=metrics_std, 
                  capsize=10, color=['blue', 'green', 'orange', 'cyan', 'magenta'])
    plt.title('Average Performance Metrics', fontsize=14)
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{base_path}/{exp_name}-Plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return scores_df

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    # Define dataset paths
    datasets = {
        'ai4health': "data/AI4healthcare.xlsx",
        'guangzhou_no_nan': "data/GuangzhouMedicalHospital_features23_no_nan.xlsx",
        'henan': "data/HenanCancerHospital_features63_58.xlsx"
    }
    
    # Define selected features
    selected_features = [
        'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5',
        'Feature14', 'Feature15', 'Feature17', 'Feature22', 'Feature39', 
        'Feature42', 'Feature43', 'Feature45', 'Feature46', 'Feature47', 
        'Feature48', 'Feature49', 'Feature50', 'Feature52', 'Feature53', 
        'Feature56', 'Feature57', 'Feature63'
    ]
    
    # Load and combine all datasets
    print("\nLoading and combining all datasets...")
    all_data = []
    
    for dataset_name, dataset_path in datasets.items():
        print(f"Loading dataset: {dataset_name}")
        try:
            df = pd.read_excel(dataset_path)
            
            # Check if dataset has required features and label
            if not all(feature in df.columns for feature in selected_features):
                print(f"Warning: {dataset_name} is missing some required features, skipping")
                continue
                
            if 'Label' not in df.columns:
                print(f"Warning: {dataset_name} does not have a Label column, skipping")
                continue
                
            # Select only required columns
            df_selected = df[selected_features + ['Label']].copy()
            
            # Add dataset source column
            df_selected['DataSource'] = dataset_name
            
            # Append to list
            all_data.append(df_selected)
            print(f"Added {len(df_selected)} samples from {dataset_name}")
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
    
    # Combine all datasets
    if not all_data:
        print("No valid datasets found. Exiting.")
        exit(1)
        
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Label distribution:\n{combined_df['Label'].value_counts()}")
    print(f"Data sources:\n{combined_df['DataSource'].value_counts()}")
    
    # Save combined dataset
    os.makedirs('./results/combined_tabpfn', exist_ok=True)
    combined_df.to_excel('./results/combined_tabpfn/combined_dataset.xlsx', index=False)
    
    # Prepare data for modeling
    X = combined_df[selected_features].copy()
    y = combined_df["Label"].copy()

    # Run cross-validation experiment
    print("\n=== Running 10-fold Cross-Validation on Combined Dataset ===")
    
    # Run experiment with combined data
    result_df = run_experiment(
        X=X,
        y=y,
        device='cuda',
        max_time=30,
        random_state=42,
        base_path='./results/combined_tabpfn'
    )
    
    print("\nExperiment completed successfully!") 