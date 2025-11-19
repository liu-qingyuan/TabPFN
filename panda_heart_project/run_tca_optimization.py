import logging
import sys
import pandas as pd
import numpy as np
from adapt.feature_based import TCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Append project root to path
sys.path.append('.')

from panda_heart_project.data.loader import HeartDiseaseDataLoader
from src.tabpfn.classifier import TabPFNClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_optimization():
    # Load Data
    loader = HeartDiseaseDataLoader(data_path="panda_heart_project/data/processed/uci_heart_disease_combined.csv")
    data = loader.load_data()
    
    # Feature Selection
    features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg', 'thalach', 'exang', 'oldpeak']
    target = 'target'
    
    # Define Tasks
    tasks = [
        ('Cleveland', 'Hungarian'),
        ('Hungarian', 'Switzerland')
    ]
    
    # Define Parameter Grid
    param_grid = [
        {'kernel': 'linear', 'mu': 0.1, 'n_components': 5, 'name': 'Linear_Mu0.1_Dim5'},
        {'kernel': 'linear', 'mu': 1.0, 'n_components': 9, 'name': 'Linear_Mu1.0_Dim9'},
        {'kernel': 'rbf', 'mu': 0.1, 'n_components': 9, 'gamma': 0.1, 'name': 'RBF_Mu0.1_G0.1'},
        {'kernel': 'rbf', 'mu': 1.0, 'n_components': 5, 'gamma': 1.0, 'name': 'RBF_Mu1.0_G1.0'},
    ]
    
    results = []
    
    logger.info(f"🚀 Starting TCA Optimization on {len(tasks)} tasks...")
    
    for source_name, target_name in tasks:
        logger.info(f"\nTask: {source_name} -> {target_name}")
        
        # Prepare Data
        source_df = data[data['center'] == source_name]
        target_df = data[data['center'] == target_name]
        
        # Handle NaNs
        X_source_raw = source_df[features].fillna(source_df[features].mean()).values
        y_source = source_df[target].values
        X_target_raw = target_df[features].fillna(target_df[features].mean()).values
        y_target = target_df[target].values
        
        # 1. Run TabPFN Baseline (The score to beat)
        tabpfn = TabPFNClassifier(device='cpu', n_estimators=4)
        tabpfn.fit(X_source_raw, y_source)
        y_pred_base = tabpfn.predict(X_target_raw)
        acc_base = accuracy_score(y_target, y_pred_base)
        logger.info(f"  🏁 TabPFN Baseline: {acc_base:.4f}")
        
        # SCALING IS CRITICAL FOR TCA
        scaler = StandardScaler()
        X_source = scaler.fit_transform(X_source_raw)
        X_target = scaler.transform(X_target_raw)
        
        # 2. Run TCA Configs
        for params in param_grid:
            try:
                # Init TCA
                tca_params = {k: v for k, v in params.items() if k != 'name'}
                tca = TCA(Xt=X_target, verbose=0, **tca_params)
                
                # Fit Adapt
                X_source_adapt = tca.fit_transform(X_source, X_target)
                X_target_adapt = tca.transform(X_target)
                
                # Train TabPFN on Adapted Features
                clf = TabPFNClassifier(device='cpu', n_estimators=4)
                clf.fit(X_source_adapt, y_source)
                y_pred = clf.predict(X_target_adapt)
                
                acc = accuracy_score(y_target, y_pred)
                gap = acc - acc_base
                icon = "✅" if gap > 0 else "❌"
                
                logger.info(f"    {icon} {params['name']}: {acc:.4f} (Gap: {gap:+.4f})")
                
                results.append({
                    'task': f"{source_name}->{target_name}",
                    'config': params['name'],
                    'acc_base': acc_base,
                    'acc_panda': acc,
                    'gap': gap
                })
                
            except Exception as e:
                logger.error(f"    ⚠️ Error with {params['name']}: {e}")

    # Summary
    logger.info("\n🏆 Optimization Summary:")
    if not results:
        logger.error("No results collected due to errors.")
        return

    df_res = pd.DataFrame(results)
    print(df_res.groupby('config')['gap'].mean().sort_values(ascending=False))

if __name__ == "__main__":
    run_optimization()