import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from adapt.feature_based import TCA
from sklearn.metrics import roc_auc_score, accuracy_score
import sys
import os

# Add project root to path
sys.path.append('/Users/lqy/work/TabPFN')
from panda_heart_project.data.loader import load_heart_disease_data

def inspect_data():
    print("🔍 Inspecting Heart Disease Data Distribution...")
    
    # 1. Load Data
    data_dict = load_heart_disease_data()
    
    hungarian = data_dict['Hungarian']
    switzerland = data_dict['Switzerland']
    va = data_dict['VA']
    
    # 2. Check Class Balance
    for name, df in [('Hungarian', hungarian), ('Switzerland', switzerland), ('VA', va)]:
        labels = df['target'].values
        n_total = len(labels)
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        print(f"\n📊 {name}:")
        print(f"   Total: {n_total}")
        print(f"   Sick (1): {n_pos} ({n_pos/n_total:.1%})")
        print(f"   Healthy (0): {n_neg} ({n_neg/n_total:.1%})")
        
        if n_neg < 10:
            print("   ⚠️ WARNING: Less than 10 negative samples! AUC will be unstable.")

    return hungarian, switzerland, va

def check_predictions(source_df, target_df, source_name, target_name):
    print(f"\n🔬 Testing PANDA predictions: {source_name} -> {target_name}")
    
    # Prepare data
    X_source = source_df.drop(columns=['target']).values
    y_source = source_df['target'].values
    X_target = target_df.drop(columns=['target']).values
    y_target = target_df['target'].values
    
    # Run TCA
    print("   Running TCA...")
    tca = TCA(kernel='linear', mu=0.1, n_components=8, verbose=0, random_state=42)
    X_source_tca = tca.fit_transform(X_source, X_target)
    X_target_tca = tca.transform(X_target)
    
    # Run TabPFN
    print("   Running TabPFN (n=32)...")
    model = TabPFNClassifier(device='cpu', n_estimators=32, random_state=42, ignore_pretraining_limits=True)
    model.fit(X_source_tca, y_source)
    y_proba = model.predict_proba(X_target_tca)[:, 1]
    
    # Analyze Probabilities
    print("\n   📉 Probability Distribution Analysis:")
    print(f"   Min Prob: {y_proba.min():.4f}")
    print(f"   Max Prob: {y_proba.max():.4f}")
    print(f"   Mean Prob: {y_proba.mean():.4f}")
    print(f"   Std Dev:  {y_proba.std():.4f}")
    
    # Check separation
    pos_probs = y_proba[y_target == 1]
    neg_probs = y_proba[y_target == 0]
    
    print(f"\n   Sick Samples (n={len(pos_probs)}) Mean Score: {pos_probs.mean():.4f}")
    if len(neg_probs) > 0:
        print(f"   Healthy Samples (n={len(neg_probs)}) Mean Score: {neg_probs.mean():.4f}")
        print(f"   Score Gap: {pos_probs.mean() - neg_probs.mean():.4f}")
    else:
        print("   No Healthy Samples to compare!")

    auc = roc_auc_score(y_target, y_proba) if len(np.unique(y_target)) > 1 else 0.5
    acc = accuracy_score(y_target, y_proba > 0.5)
    print(f"\n   🏆 Results: AUC = {auc:.4f}, Accuracy = {acc:.4f}")

if __name__ == "__main__":
    hungarian, switzerland, va = inspect_data()
    check_predictions(hungarian, switzerland, 'Hungarian', 'Switzerland')
