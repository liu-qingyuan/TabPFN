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
    dataset_sources,
    device='cuda',
    max_time=15,
    random_state=42,
    base_path='./results/combined_tabpfn_cv10'
):
    """
    Run AutoTabPFN experiment with 10-fold cross-validation
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    dataset_sources : pd.Series
        Source dataset for each sample (A, B, C)
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
    tuple of pd.DataFrame
        DataFrames with cross-validation scores and per-dataset performances
    """
    # Create results directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate experiment name based on parameters
    exp_name = f"CombinedTabPFN-T{max_time}-R{random_state}"
    
    print("Data Shape:", X.shape)
    print("Label Distribution:\n", y.value_counts())
    print("Dataset Distribution:\n", dataset_sources.value_counts())
    
    # Convert data to numpy arrays
    X_values = X.values.astype(np.float32)
    y_values = y.values.astype(np.int32)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)
    
    # Identify unique datasets
    unique_datasets = np.unique(dataset_sources)
    
    # ==============================
    # 10-Fold Cross Validation
    # ==============================
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    fold_scores = []
    dataset_performances = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_values[train_idx], y_values[test_idx]
        ds_train = dataset_sources.iloc[train_idx].reset_index(drop=True)
        ds_test = dataset_sources.iloc[test_idx].reset_index(drop=True)
        
        print(f"\n{'='*50}")
        print(f"Fold {fold} of 10")
        print(f"{'='*50}")
        
        # Print dataset distribution in this fold
        print("\nDataset Distribution in Fold:")
        for ds in unique_datasets:
            train_count = np.sum(ds_train == ds)
            test_count = np.sum(ds_test == ds)
            print(f"Dataset {ds}: {train_count} train samples, {test_count} test samples")
        
        # Initialize and train model
        start_time = time.time()
        clf = AutoTabPFNClassifier(
            device=device,
            max_time=max_time,
            random_state=random_state
        )
        clf.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred_proba = clf.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        fold_time = time.time() - start_time
        
        # Calculate overall metrics for this fold
        overall_metrics = evaluate_metrics(y_test, y_pred, y_pred_proba[:, 1])
        
        print("\nOverall Test Metrics:")
        print_metrics("Test", overall_metrics)
        print(f"Time: {fold_time:.4f}s")
        
        # Calculate per-dataset metrics
        for ds in unique_datasets:
            ds_indices = ds_test == ds
            if np.sum(ds_indices) > 0:
                ds_y_true = y_test[ds_indices]
                ds_y_pred = y_pred[ds_indices]
                ds_y_pred_proba = y_pred_proba[ds_indices, 1]
                
                try:
                    ds_metrics = evaluate_metrics(ds_y_true, ds_y_pred, ds_y_pred_proba)
                    
                    print(f"\nDataset {ds} Test Metrics:")
                    print_metrics(f"Dataset {ds}", ds_metrics)
                    
                    # Store dataset performance for this fold
                    dataset_performances.append({
                        'fold': fold,
                        'dataset': ds,
                        'accuracy': ds_metrics['acc'],
                        'auc': ds_metrics['auc'],
                        'f1': ds_metrics['f1'],
                        'acc_0': ds_metrics['acc_0'],
                        'acc_1': ds_metrics['acc_1'],
                        'sample_count': np.sum(ds_indices)
                    })
                    
                except Exception as e:
                    print(f"Error calculating metrics for dataset {ds}: {e}")
        
        # Store overall fold performance
        fold_scores.append({
            'fold': fold,
            'accuracy': overall_metrics['acc'],
            'auc': overall_metrics['auc'],
            'f1': overall_metrics['f1'],
            'acc_0': overall_metrics['acc_0'],
            'acc_1': overall_metrics['acc_1'],
            'time': fold_time
        })
    
    # ==============================
    # Summary Results
    # ==============================
    # Convert to DataFrames
    scores_df = pd.DataFrame(fold_scores)
    dataset_perf_df = pd.DataFrame(dataset_performances)
    
    # Save results
    scores_df.to_csv(f'{base_path}/{exp_name}-CV-Overall.csv', index=False)
    dataset_perf_df.to_csv(f'{base_path}/{exp_name}-CV-PerDataset.csv', index=False)
    
    # Print overall summary
    print("\n" + "="*50)
    print("Cross-Validation Summary")
    print("="*50)
    
    print("\nOverall Performance Across All Folds:")
    print(f"Average Accuracy: {scores_df['accuracy'].mean():.4f} ± {scores_df['accuracy'].std():.4f}")
    print(f"Average AUC: {scores_df['auc'].mean():.4f} ± {scores_df['auc'].std():.4f}")
    print(f"Average F1: {scores_df['f1'].mean():.4f} ± {scores_df['f1'].std():.4f}")
    print(f"Average Class 0 Accuracy: {scores_df['acc_0'].mean():.4f} ± {scores_df['acc_0'].std():.4f}")
    print(f"Average Class 1 Accuracy: {scores_df['acc_1'].mean():.4f} ± {scores_df['acc_1'].std():.4f}")
    print(f"Average Time: {scores_df['time'].mean():.4f}s")
    
    # Print per-dataset summary
    print("\nPer-Dataset Performance:")
    for ds in unique_datasets:
        ds_data = dataset_perf_df[dataset_perf_df['dataset'] == ds]
        if not ds_data.empty:
            print(f"\nDataset {ds}:")
            print(f"Average Accuracy: {ds_data['accuracy'].mean():.4f} ± {ds_data['accuracy'].std():.4f}")
            print(f"Average AUC: {ds_data['auc'].mean():.4f} ± {ds_data['auc'].std():.4f}")
            print(f"Average F1: {ds_data['f1'].mean():.4f} ± {ds_data['f1'].std():.4f}")
            print(f"Average Class 0 Accuracy: {ds_data['acc_0'].mean():.4f} ± {ds_data['acc_0'].std():.4f}")
            print(f"Average Class 1 Accuracy: {ds_data['acc_1'].mean():.4f} ± {ds_data['acc_1'].std():.4f}")
    
    # ==============================
    # Visualize Results
    # ==============================
    # Figure 1: Overall performance across folds
    plt.figure(figsize=(20, 16))
    
    # Plot 1: Overall performance metrics across folds
    plt.subplot(3, 2, 1)
    metrics = ['accuracy', 'auc', 'f1']
    for metric in metrics:
        plt.plot(scores_df['fold'], scores_df[metric], 'o-', label=metric.upper())
    plt.title('Overall Performance Metrics Across Folds', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Per-dataset AUC across folds
    plt.subplot(3, 2, 2)
    for ds in unique_datasets:
        ds_data = dataset_perf_df[dataset_perf_df['dataset'] == ds]
        if not ds_data.empty:
            plt.plot(ds_data['fold'], ds_data['auc'], 'o-', label=f'Dataset {ds}')
    plt.title('AUC by Dataset Across Folds', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('AUC', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Per-dataset F1 across folds
    plt.subplot(3, 2, 3)
    for ds in unique_datasets:
        ds_data = dataset_perf_df[dataset_perf_df['dataset'] == ds]
        if not ds_data.empty:
            plt.plot(ds_data['fold'], ds_data['f1'], 'o-', label=f'Dataset {ds}')
    plt.title('F1 Score by Dataset Across Folds', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Per-dataset Accuracy across folds
    plt.subplot(3, 2, 4)
    for ds in unique_datasets:
        ds_data = dataset_perf_df[dataset_perf_df['dataset'] == ds]
        if not ds_data.empty:
            plt.plot(ds_data['fold'], ds_data['accuracy'], 'o-', label=f'Dataset {ds}')
    plt.title('Accuracy by Dataset Across Folds', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 5: Per-dataset class 0 accuracy across folds
    plt.subplot(3, 2, 5)
    for ds in unique_datasets:
        ds_data = dataset_perf_df[dataset_perf_df['dataset'] == ds]
        if not ds_data.empty:
            plt.plot(ds_data['fold'], ds_data['acc_0'], 'o-', label=f'Dataset {ds}')
    plt.title('Class 0 Accuracy by Dataset Across Folds', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Class 0 Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 6: Per-dataset class 1 accuracy across folds
    plt.subplot(3, 2, 6)
    for ds in unique_datasets:
        ds_data = dataset_perf_df[dataset_perf_df['dataset'] == ds]
        if not ds_data.empty:
            plt.plot(ds_data['fold'], ds_data['acc_1'], 'o-', label=f'Dataset {ds}')
    plt.title('Class 1 Accuracy by Dataset Across Folds', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Class 1 Accuracy', fontsize=12)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{base_path}/{exp_name}-Fold-Plots.png', dpi=300, bbox_inches='tight')
    
    # Figure 2: Summary comparison across datasets
    plt.figure(figsize=(20, 10))
    
    # Summary statistics per dataset
    ds_summary = dataset_perf_df.groupby('dataset').agg({
        'accuracy': ['mean', 'std'],
        'auc': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'acc_0': ['mean', 'std'],
        'acc_1': ['mean', 'std']
    }).reset_index()
    
    # Add overall performance to the summary
    overall_row = pd.DataFrame({
        'dataset': ['Overall'],
        'accuracy': [scores_df['accuracy'].mean()],
        'accuracy_std': [scores_df['accuracy'].std()],
        'auc': [scores_df['auc'].mean()],
        'auc_std': [scores_df['auc'].std()],
        'f1': [scores_df['f1'].mean()],
        'f1_std': [scores_df['f1'].std()],
        'acc_0': [scores_df['acc_0'].mean()],
        'acc_0_std': [scores_df['acc_0'].std()],
        'acc_1': [scores_df['acc_1'].mean()],
        'acc_1_std': [scores_df['acc_1'].std()]
    })
    
    # Convert multi-index to flat columns for easier plotting
    flat_cols = {}
    for col, values in ds_summary.items():
        if col[0] == 'dataset':
            flat_cols['dataset'] = values
        else:
            if col[1] == 'mean':
                flat_cols[col[0]] = values
            else:
                flat_cols[f'{col[0]}_std'] = values
    
    ds_summary_flat = pd.DataFrame(flat_cols)
    ds_summary_with_overall = pd.concat([overall_row, ds_summary_flat], ignore_index=True)
    
    # Save summary to CSV
    ds_summary_with_overall.to_csv(f'{base_path}/{exp_name}-Summary.csv', index=False)
    
    # Plot 1: Dataset comparison - Accuracy, AUC, F1
    plt.subplot(1, 2, 1)
    datasets = ds_summary_with_overall['dataset'].tolist()
    metrics = ['accuracy', 'auc', 'f1']
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = ds_summary_with_overall[metric].values
        errors = ds_summary_with_overall[f'{metric}_std'].values
        plt.bar(x + (i-1)*width, values, width, yerr=errors, 
               label=metric.upper(), capsize=5)
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Performance Metrics by Dataset', fontsize=14)
    plt.xticks(x, datasets)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Dataset comparison - Class-wise accuracy
    plt.subplot(1, 2, 2)
    metrics = ['acc_0', 'acc_1']
    labels = ['Class 0 Accuracy', 'Class 1 Accuracy']
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        values = ds_summary_with_overall[metric].values
        errors = ds_summary_with_overall[f'{metric}_std'].values
        plt.bar(x + (i-0.5)*width, values, width, yerr=errors, 
               label=label, capsize=5)
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Class-wise Accuracy by Dataset', fontsize=14)
    plt.xticks(x, datasets)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{base_path}/{exp_name}-Summary-Plots.png', dpi=300, bbox_inches='tight')
    
    return scores_df, dataset_perf_df

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    # Define dataset paths
    datasets = {
        'A': "data/AI4healthcare.xlsx",
        'B': "data/GuangzhouMedicalHospital_features23_no_nan.xlsx",
        'C': "data/HenanCancerHospital_features63_58.xlsx"
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
    os.makedirs('./results/combined_tabpfn_cv10', exist_ok=True)
    combined_df.to_excel('./results/combined_tabpfn_cv10/combined_dataset.xlsx', index=False)
    
    # Prepare data for modeling
    X = combined_df[selected_features].copy()
    y = combined_df["Label"].copy()
    dataset_sources = combined_df["DataSource"].copy()

    # Run experiment with combined data
    print("\n=== Running 10-fold Cross-Validation ===")
    overall_cv, dataset_cv = run_experiment(
        X=X,
        y=y,
        dataset_sources=dataset_sources,
        device='cuda',
        max_time=30,
        random_state=42,
        base_path='./results/combined_tabpfn_cv10'
    )
    
    print("\nExperiment completed successfully!") 