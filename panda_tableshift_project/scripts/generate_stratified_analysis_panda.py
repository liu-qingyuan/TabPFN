#!/usr/bin/env python3
"""
Generate stratified analysis table for PANDA on TableShift data.

This script reproduces a stratified analysis (Subgroup, n, AUC, Accuracy, Sensitivity)
for the PANDA model, similar to the medical research analysis but applied to
the TableShift project context (defaulting to BRFSS Diabetes).

It generates a table showing model performance across different patient subgroups
(e.g., Age, Sex, BMI).
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from tableshift import get_dataset

# Add project root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

try:
    from experiments.run_complete_analysis import (
        load_tabpfn_params,
        instantiate_model,
        ensure_proba,
    )
    from src.utils import sample_data
except ImportError:
    print("Could not import project modules. Ensure you are in the panda_tableshift_project root or scripts directory.")
    sys.exit(1)


def get_stratification_rules(df):
    """
    Define stratification rules based on available columns.
    Returns a list of tuples: (Group Name, Filter Function)
    """
    rules = []
    columns = df.columns
    
    # 1. Sex/Gender Stratification
    if 'Sex' in columns:
        # Assuming 1=Male, 2=Female or 0/1. We'll use unique values.
        uniques = sorted(df['Sex'].unique())
        if len(uniques) == 2:
            rules.append(('Gender: Male (or 0)', lambda x: x['Sex'] == uniques[0]))
            rules.append(('Gender: Female (or 1)', lambda x: x['Sex'] == uniques[1]))
    elif 'SEX' in columns: # BRFSS often uses caps
         uniques = sorted(df['SEX'].unique())
         if len(uniques) >= 2:
             rules.append(('Gender: Male', lambda x: x['SEX'] == uniques[0]))
             rules.append(('Gender: Female', lambda x: x['SEX'] == uniques[1]))

    # 2. Age Stratification
    if 'Age' in columns:
        median_age = df['Age'].median()
        rules.append((f'Age <= {median_age}', lambda x: x['Age'] <= median_age))
        rules.append((f'Age > {median_age}', lambda x: x['Age'] > median_age))
    elif 'AGE' in columns:
        median_age = df['AGE'].median()
        rules.append((f'Age <= {median_age}', lambda x: x['AGE'] <= median_age))
        rules.append((f'Age > {median_age}', lambda x: x['AGE'] > median_age))
        
    # 3. BMI Stratification (Nodule Size equivalent - continuous physical feature)
    if 'BMI' in columns:
        median_bmi = df['BMI'].median()
        rules.append((f'BMI <= {median_bmi:.1f}', lambda x: x['BMI'] <= median_bmi))
        rules.append((f'BMI > {median_bmi:.1f}', lambda x: x['BMI'] > median_bmi))

    # 4. General Health (Smoker equivalent - categorical health status)
    if 'GenHlth' in columns:
        # 1-5 scale usually. Split by Good vs Poor/Fair
        rules.append(('GenHlth: Excellent/Very Good', lambda x: x['GenHlth'] <= 2))
        rules.append(('GenHlth: Good/Fair/Poor', lambda x: x['GenHlth'] > 2))
        
    return rules

def calculate_metrics(y_true, y_pred_proba, y_pred_class):
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = np.nan
        
    acc = accuracy_score(y_true, y_pred_class)
    sens = recall_score(y_true, y_pred_class, pos_label=1)
    
    return auc, acc, sens

def main():
    parser = argparse.ArgumentParser(description="Generate PANDA Stratified Analysis Table.")
    parser.add_argument("--dataset", type=str, default="brfss_diabetes", help="TableShift dataset name")
    parser.add_argument("--n_train", type=int, default=2048, help="Number of training samples")
    parser.add_argument("--n_test", type=int, default=4096, help="Number of test samples")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    args = parser.parse_args()

    print(f"🚀 Starting PANDA Stratified Analysis on {args.dataset}...")

    # 1. Load Data
    print("Loading dataset...")
    try:
        dataset = get_dataset(args.dataset, cache_dir=str(repo_root / "data" / "raw"), initialize_data=True, use_cached=False)
    except Exception as e:
        print(f"Error loading dataset {args.dataset}: {e}")
        return

    # Use OOD Test for evaluation to match the 'Target Cohort' concept in medical project
    X_train, y_train, _, _ = dataset.get_pandas("train")
    X_test, y_test, _, _ = dataset.get_pandas("ood_test") 
    
    # Sampling for speed if needed
    X_train_sub, y_train_sub = sample_data(X_train, y_train, args.n_train)
    X_test_sub, y_test_sub = sample_data(X_test, y_test, args.n_test)
    
    print(f"Data Loaded: Train n={len(X_train_sub)}, Test n={len(X_test_sub)}")

    # 2. Load and Train PANDA Model
    print("Initializing PANDA model...")
    tuning_csv = repo_root / "results" / "tuning_extended_brfss_diabetes.csv"
    
    try:
        _, panda_cfg = load_tabpfn_params(tuning_csv)
    except Exception:
        print("Warning: Could not load tuning results, using default PANDA config.")
        # Fallback config if file missing
        from experiments.run_complete_analysis import ModelConfig
        panda_cfg = ModelConfig("PANDA", {}, is_adaptation=True)

    model = instantiate_model(panda_cfg, device=args.device)
    
    print("Training PANDA (Fitting)...")
    # Convert to numpy for fitting
    X_train_np = np.asarray(X_train_sub)
    y_train_np = np.asarray(y_train_sub)
    X_test_np = np.asarray(X_test_sub)
    
    # Fit (with adaptation if configured)
    if getattr(panda_cfg, "is_adaptation", False):
        model.fit(X_train_np, y_train_np, X_target=X_test_np)
    else:
        model.fit(X_train_np, y_train_np)

    # 3. Predictions
    print("Running Inference...")
    y_pred_proba_all = ensure_proba(model, X_test_np)
    
    # Extract probability of positive class (index 1) if 2D
    if y_pred_proba_all.ndim == 2 and y_pred_proba_all.shape[1] == 2:
        y_pred_proba = y_pred_proba_all[:, 1]
    else:
        y_pred_proba = y_pred_proba_all

    y_pred_class = (y_pred_proba > 0.5).astype(int)

    # 4. Stratified Analysis
    print("\n📊 Generating Stratified Table...")
    print(f"Available columns: {X_test_sub.columns.tolist()}")
    
    # Define Rules
    rules = get_stratification_rules(X_test_sub)
    
    results = []
    
    # Overall Performance
    auc, acc, sens = calculate_metrics(y_test_sub, y_pred_proba, y_pred_class)
    results.append({
        "Subgroup": "Overall",
        "n": len(y_test_sub),
        "AUC": auc,
        "Accuracy": acc,
        "Sensitivity": sens
    })
    
    # Subgroup Performance
    for name, filter_func in rules:
        mask = filter_func(X_test_sub)
        n_count = mask.sum()
        
        if n_count < 10:
            print(f"Skipping {name}: n={n_count} (too small)")
            continue
            
        y_sub_true = y_test_sub[mask]
        y_sub_proba = y_pred_proba[mask]
        y_sub_class = y_pred_class[mask]
        
        # Check if we have both classes
        if len(np.unique(y_sub_true)) < 2:
            auc = np.nan # AUC undefined for single class
        else:
            auc, acc, sens = calculate_metrics(y_sub_true, y_sub_proba, y_sub_class)
            
        results.append({
            "Subgroup": name,
            "n": n_count,
            "AUC": auc,
            "Accuracy": acc,
            "Sensitivity": sens
        })

    # 5. Output Table
    df_res = pd.DataFrame(results)
    
    # Formatting
    print("\n" + "="*70)
    print(f"{ 'Subgroup':<30} | {'n':<6} | {'AUC':<6} | {'Accuracy':<8} | {'Sensitivity':<11}")
    print("-" * 70)
    
    for _, row in df_res.iterrows():
        auc_str = f"{row['AUC']:.3f}" if not np.isnan(row['AUC']) else "N/A"
        print(f"{row['Subgroup']:<30} | {row['n']:<6} | {auc_str:<6} | {row['Accuracy']:.3f}    | {row['Sensitivity']:.3f}")
        
    print("="*70)
    
    # Save to CSV
    output_path = repo_root / "results" / "stratified_analysis_panda.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(output_path, index=False)
    print(f"\n✅ Results saved to: {output_path}")

if __name__ == "__main__":
    main()
