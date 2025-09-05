#!/usr/bin/env python3
"""
Unified RFE Feature Selection and Performance Evaluation Script

This script combines the functionality of:
1. predict_healthcare_RFE.py - RFE feature selection using TabPFN
2. evaluate_feature_numbers.py - Performance evaluation across different feature counts

Author: Generated for UDA Medical Imbalance Project
Date: 2024
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from tabpfn import TabPFNClassifier
from types import SimpleNamespace
from tqdm import tqdm
import torch


class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper class to make TabPFN compatible with sklearn's RFE
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
        # Set class information for scoring functions
        self.classes_ = np.unique(y)
        
        # Initialize TabPFN model
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
        
        # Calculate feature importance using permutation importance
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
        """Return feature importance scores"""
        return self.feature_importances_

    def get_feature_importance_scores(self):
        """Return feature importance scores and standard deviations"""
        return {
            'mean': self.feature_importances_,
            'std': self.feature_importances_std_
        }


def select_features_rfe(X, y, n_features=3):
    """
    Use RFE with TabPFN as base estimator for feature selection
    """
    n_features_total = X.shape[1]
    n_iterations = n_features_total - n_features
    
    # Initialize TabPFN wrapper
    base_model = TabPFNWrapper(
        device='cuda',
        n_estimators=32,
        softmax_temperature=0.9,
        balance_probabilities=False,
        average_before_softmax=False,
        ignore_pretraining_limits=True,
        random_state=42
    )
    
    # Initialize RFE
    rfe = RFE(
        estimator=base_model,
        n_features_to_select=n_features,
        step=1,
        verbose=2
    )
    
    # Create progress bar and fit RFE
    print("Fitting RFE with TabPFN as base model...")
    with tqdm(total=n_iterations, desc='Eliminating features') as pbar:
        rfe.fit(X, y)
        pbar.update(n_iterations)
    
    # Get selected features
    selected_features = X.columns[rfe.support_].tolist()
    
    # Get feature importance ranking
    feature_ranking = pd.DataFrame({
        'Feature': X.columns,
        'Rank': rfe.ranking_
    }).sort_values('Rank')
    
    return selected_features, feature_ranking


def evaluate_feature_performance(X, y, feature_ranking, results_dir):
    """
    Evaluate model performance across different numbers of features
    Using the complete RFE ranking from most to least important features
    """
    print("\n" + "="*60)
    print("PHASE 2: Performance Evaluation Across Feature Counts")
    print("="*60)
    
    # Get ranked features (sorted by RFE rank: 1=most important, 63=least important)
    ranked_features = feature_ranking.sort_values('Rank')['Feature'].tolist()
    
    print(f"Using RFE ranking order: {ranked_features[:5]}...{ranked_features[-5:]}")
    print(f"Total features available: {len(ranked_features)}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define feature numbers to test (3 to 63)
    feature_numbers = list(range(3, len(ranked_features) + 1))
    print(f"Will evaluate feature counts: {feature_numbers[0]} to {feature_numbers[-1]} ({len(feature_numbers)} evaluations)")
    
    # Store all results
    all_results = []
    
    # Evaluate each feature count
    for n_features in tqdm(feature_numbers, desc=f"Evaluating features (3-{len(ranked_features)})"):
        # Select top n features according to RFE ranking (rank 1 = most important)
        selected_features = ranked_features[:n_features]
        X_selected = X[selected_features]
        
        # Only print details for first few and key milestones to reduce output
        if n_features <= 10 or n_features % 10 == 0:
            print(f"\n--- Evaluating {n_features} features ---")
            print(f"Top {min(5, len(selected_features))} features: {selected_features[:5]}")
        
        # 10-fold cross validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_selected), 1):
            X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            start_time = time.time()
            clf = TabPFNClassifier(
                device='cuda',
                n_estimators=32,
                softmax_temperature=0.9,
                balance_probabilities=False,
                average_before_softmax=False,
                ignore_pretraining_limits=True,
                random_state=42
            )
            
            # Ensure reproducible results
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
                
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
            
            fold_scores.append({
                'fold': fold,
                'accuracy': acc,
                'auc': auc,
                'f1': f1,
                'acc_0': acc_0,
                'acc_1': acc_1,
                'time': fold_time
            })
        
        # Calculate mean and std
        metrics = ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1', 'time']
        mean_scores = {f'mean_{m}': np.mean([s[m] for s in fold_scores]) for m in metrics}
        std_scores = {f'std_{m}': np.std([s[m] for s in fold_scores]) for m in metrics}
        
        # Store result
        result = {
            'n_features': n_features,
            'features': ', '.join(selected_features),
            **mean_scores,
            **std_scores
        }
        all_results.append(result)
        
        # Print current result (simplified output)
        if n_features <= 10 or n_features % 10 == 0:
            print(f"Results for {n_features} features:")
            print(f"AUC: {mean_scores['mean_auc']:.4f}±{std_scores['std_auc']:.4f}, "
                  f"ACC: {mean_scores['mean_accuracy']:.4f}±{std_scores['std_accuracy']:.4f}, "
                  f"F1: {mean_scores['mean_f1']:.4f}±{std_scores['std_f1']:.4f}")
        else:
            # Quick summary for intermediate results
            print(f"N={n_features}: AUC={mean_scores['mean_auc']:.3f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(results_dir, "feature_number_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Create visualization
    create_performance_visualization(all_results, feature_numbers, results_dir)
    
    return results_df


def create_performance_visualization(all_results, feature_numbers, results_dir):
    """
    Create performance comparison visualization
    """
    plt.figure(figsize=(15, 10))
    metrics = ['auc', 'accuracy', 'f1']
    colors = ['blue', 'green', 'red']
    
    for metric, color in zip(metrics, colors):
        mean_values = [result[f'mean_{metric}'] for result in all_results]
        std_values = [result[f'std_{metric}'] for result in all_results]
        
        plt.plot(feature_numbers, mean_values, marker='o', color=color, label=metric.upper())
        plt.fill_between(
            feature_numbers,
            [m - s for m, s in zip(mean_values, std_values)],
            [m + s for m, s in zip(mean_values, std_values)],
            color=color,
            alpha=0.2
        )
    
    plt.xlabel('Number of Features')
    plt.ylabel('Score')
    plt.title('Model Performance vs Number of Features')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(results_dir, "performance_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {plot_path}")


def main():
    """
    Main execution function
    """
    print("="*60)
    print("UNIFIED RFE FEATURE SELECTION AND PERFORMANCE EVALUATION")
    print("="*60)
    
    # Create results directory
    results_dir = os.path.join("..", "results", "feature_selection_evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data_path = os.path.join("..", "..", "data", "AI4healthcare.xlsx")
    df = pd.read_excel(data_path)
    features = [c for c in df.columns if c.startswith("Feature")]
    X = df[features].copy()
    y = df["Label"].copy()
    
    print(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"All feature columns: {len(features)} features from {features[0]} to {features[-1]}")
    
    print(f"Data Shape: {X.shape}")
    print(f"Label Distribution:\n{y.value_counts()}")
    
    # Phase 1: RFE Feature Selection
    print("\n" + "="*60)
    print("PHASE 1: RFE Feature Selection with TabPFN")
    print("="*60)
    
    print("Performing RFE with TabPFN to rank ALL features from most to least important...")
    print("This will generate a complete ranking of all 63 features.")
    
    # Modified: Use RFE to select 3 features but get complete ranking of all 63 features
    selected_features, feature_ranking = select_features_rfe(X, y, n_features=3)
    
    # Save feature ranking (this contains all 63 features ranked by importance)
    ranking_path = os.path.join("..", "results", "RFE_feature_ranking.csv")
    feature_ranking.to_csv(ranking_path, index=False)
    
    print(f"\nComplete feature ranking (all 63 features) saved to: {ranking_path}")
    print(f"RFE process: Started with {X.shape[1]} features, eliminated down to 3 features")
    print(f"Ranking explanation: Rank 1 = most important (selected first), Rank {X.shape[1]} = least important (eliminated first)")
    
    print("\nTop 10 most important features (Rank 1-10, selected first by RFE):")
    print(feature_ranking.head(10).to_string(index=False))
    print("\nBottom 10 least important features (eliminated first by RFE):")
    print(feature_ranking.tail(10).to_string(index=False))
    
    # Verify RFE logic: the 3 selected features should have ranks 1, 2, 3
    print(f"\nVerification - The 3 selected features and their ranks:")
    selected_feature_ranks = feature_ranking[feature_ranking['Feature'].isin(selected_features)].sort_values('Rank')
    print(selected_feature_ranks.to_string(index=False))
    
    # Phase 2: Performance Evaluation
    print(f"\nNow evaluating performance using features 3-63 in order of RFE ranking...")
    results_df = evaluate_feature_performance(X, y, feature_ranking, results_dir)
    
    # Summary
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    
    print("\nGenerated Files:")
    print(f"1. Feature Rankings: {ranking_path}")
    print(f"2. Performance Results: {os.path.join(results_dir, 'feature_number_comparison.csv')}")
    print(f"3. Performance Plot: {os.path.join(results_dir, 'performance_comparison.png')}")
    
    # Find best performing feature count
    best_auc_idx = results_df['mean_auc'].idxmax()
    best_result = results_df.iloc[best_auc_idx]
    
    print(f"\nBest Performance Summary:")
    print(f"Best Feature Count: {best_result['n_features']}")
    print(f"Best AUC: {best_result['mean_auc']:.4f} ± {best_result['std_auc']:.4f}")
    print(f"Best Accuracy: {best_result['mean_accuracy']:.4f} ± {best_result['std_accuracy']:.4f}")
    print(f"Best F1: {best_result['mean_f1']:.4f} ± {best_result['std_f1']:.4f}")
    
    return results_df, feature_ranking


if __name__ == "__main__":
    results_df, feature_ranking = main()