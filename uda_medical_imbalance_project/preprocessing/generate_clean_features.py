#!/usr/bin/env python3
"""
Generate clean feature configurations from best3 to best58
"""

import pandas as pd
from pathlib import Path

def generate_clean_feature_configs():
    """Generate clean feature configurations"""
    
    csv_path = Path("../results/feature_selection_evaluation_58features_20250906_003222/feature_number_comparison.csv")
    df = pd.read_csv(csv_path)
    
    print("# Complete feature configurations based on 58-feature RFE results")
    print("# Generated: best3 through best58 (56 total configurations)")
    print("")
    
    # Generate configurations for missing numbers (13, 14, 16-19, 21-31, 33-57)
    configs_needed = []
    for index, row in df.iterrows():
        n_features = int(row['n_features'])
        features_str = row['features']
        mean_auc = row['mean_auc']
        
        # Parse feature names
        feature_names = [f.strip() for f in features_str.split(',')]
        
        configs_needed.append((n_features, feature_names, mean_auc))
    
    # Generate the missing configurations
    current_configs = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 32, 58}  # Currently defined
    
    for n_features, feature_names, mean_auc in configs_needed:
        if n_features not in current_configs:
            print(f"# 最佳{n_features}特征配置 (AUC: {mean_auc:.4f})")
            print(f"BEST_{n_features}_FEATURES = [")
            
            # Format features cleanly - 4 per line
            lines = []
            line = []
            for feature in feature_names:
                line.append(f"'{feature}'")
                if len(line) == 4:
                    lines.append("    " + ", ".join(line) + ",")
                    line = []
            
            if line:  # Handle remaining features
                if len(lines) > 0:  # Not the first line
                    lines.append("    " + ", ".join(line))
                else:  # Only one line
                    lines.append("    " + ", ".join(line))
            else:  # Remove trailing comma from last line
                lines[-1] = lines[-1].rstrip(',')
            
            for line in lines:
                print(line)
            print("]")
            print("")
    
    # Generate elif conditions for get_features_by_type
    print("\n# Additional elif conditions for get_features_by_type function:")
    for n_features, _, _ in configs_needed:
        if n_features not in current_configs:
            print(f"    elif feature_type == 'best{n_features}':")
            print(f"        return BEST_{n_features}_FEATURES.copy()")
    
    print(f"\n# Total new configurations to add: {len([n for n, _, _ in configs_needed if n not in current_configs])}")

if __name__ == "__main__":
    generate_clean_feature_configs()