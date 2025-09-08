#!/usr/bin/env python3
"""
Extract complete feature mappings from best3 to best58
Generate all BEST_N_FEATURES configurations for config/settings.py
"""

import pandas as pd
from pathlib import Path

def extract_complete_feature_mappings():
    """Extract complete feature mappings from the 58-feature CSV file"""
    
    csv_path = Path("uda_medical_imbalance_project/results/feature_selection_evaluation_58features_20250906_003222/feature_number_comparison.csv")
    
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        return
        
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    print("# Complete feature configurations based on 58-feature RFE results")
    print("# From: feature_selection_evaluation_58features_20250906_003222")
    print("# Generated: best3 through best58")
    print("")
    
    # Generate all configurations from 3 to 58
    for index, row in df.iterrows():
        n_features = row['n_features']
        features_str = row['features']
        mean_auc = row['mean_auc']
        
        # Parse feature names
        feature_names = [f.strip() for f in features_str.split(',')]
        
        # Generate the configuration
        print(f"# 最佳{n_features}特征配置 (AUC: {mean_auc:.4f})")
        print(f"BEST_{n_features}_FEATURES = [")
        
        # Format features - 4 per line for readability
        for i, feature in enumerate(feature_names):
            if i == 0:
                print(f"    '{feature}'", end="")
            elif i == len(feature_names) - 1:
                print(f", '{feature}'")
            else:
                print(f", '{feature}'", end="")
                if (i + 1) % 4 == 0:  # New line every 4 features
                    print(",")
                    print("    ", end="")
        
        print("]")
        print("")
    
    print("\n# Summary:")
    print(f"# Total configurations: {len(df)}")
    print(f"# Range: best{df['n_features'].min()} to best{df['n_features'].max()}")
    print(f"# Best AUC: {df['mean_auc'].max():.4f} (best{df.loc[df['mean_auc'].idxmax(), 'n_features']})")

if __name__ == "__main__":
    extract_complete_feature_mappings()