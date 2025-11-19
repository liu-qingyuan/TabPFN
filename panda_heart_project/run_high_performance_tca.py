"""
High-Performance PANDA-Heart TCA Experiments
基于之前成功的配置，恢复到83%准确率的高性能版本
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project to path
import sys
sys.path.append(str(Path(__file__).parent))

# Imports
from data.loader import HeartDiseaseDataLoader
from models.panda_adapt_adapter import PANDAAdaptAdapter

class HighPerformanceTCA:
    """High-performance TCA experiments based on successful configuration"""

    def __init__(self):
        """Initialize experiment runner with optimal settings"""
        self.loader = HeartDiseaseDataLoader()
        self.loader.load_data()

        # High-performance configuration - 基于之前成功的结果
        self.models = {
            'PANDA_TabPFN_TCA': PANDAAdaptAdapter(
                adaptation_method='TCA',
                mu=1.0,  # 高性能配置的关键参数
                kernel='linear',
                n_components=15  # 优化的组件数
            )
        }

        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
        }

        # Calculate AUC if probabilities are available
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['auc'] = 0.5

        # Calculate confusion matrix metrics
        try:
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                    'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
                })
            else:
                metrics.update({
                    'sensitivity': 0, 'specificity': 0, 'precision': 0, 'negative_predictive_value': 0
                })
        except Exception as e:
            logger.warning(f"Error calculating confusion matrix: {e}")
            metrics.update({
                'sensitivity': 0, 'specificity': 0, 'precision': 0, 'negative_predictive_value': 0
            })

        return metrics

    def preprocess_data(self, X):
        """Preprocess data to handle NaN values and ensure numerical stability"""
        # Convert to numpy if DataFrame
        if hasattr(X, 'values'):
            X = X.values

        # Handle NaN values - simple imputation
        if np.any(np.isnan(X)):
            logger.warning(f"Found NaN values, imputing with column means")
            col_means = np.nanmean(X, axis=0)
            nan_indices = np.where(np.isnan(col_means))[0]

            # For columns with all NaN, use 0
            col_means[nan_indices] = 0

            # Fill NaN values
            X = np.where(np.isnan(X), col_means, X)

        return X

    def run_single_center_experiments(self):
        """Run single-center cross-validation experiments with high performance"""
        logger.info("🏥 Running high-performance single-center experiments...")

        centers = self.loader.get_centers()

        for center in centers:
            logger.info(f"Processing center: {center}")

            # Get center data
            center_data = self.loader.get_center_data(center)
            if center_data is None or len(center_data) < 50:
                logger.warning(f"Insufficient data for center {center}")
                continue

            X = center_data.drop(['target', 'center'], axis=1, errors='ignore')
            y = center_data['target']

            # Preprocess data
            X = self.preprocess_data(X)

            # Create 3-fold cross-validation
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            logger.info(f"Created 3-fold CV for {center} ({len(X)} samples)")

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                logger.info(f"Fold {fold+1}/3")

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # For domain adaptation simulation, use train as source and validation as target
                X_source, X_target = X_train, X_val

                for model_name, model in self.models.items():
                    try:
                        logger.info(f"Fitting {model_name} on {len(X_train)} samples with {X_train.shape[1]} features")

                        # PANDA with TCA domain adaptation
                        model.fit(X_source, y_train, X_target)
                        y_pred = model.predict(X_val, domain='target')
                        y_pred_proba = model.predict_proba(X_val, domain='target')

                        # Calculate metrics
                        metrics = self.calculate_metrics(y_val, y_pred, y_pred_proba)

                        # Store results
                        result = {
                            'experiment_type': 'single_center',
                            'model_name': model_name,
                            'center': center,
                            'fold': fold,
                            'train_size': len(X_train),
                            'val_size': len(X_val),
                            'source_center': center,
                            'target_center': center,
                            'source_size': len(X_source),
                            'target_size': len(X_target),
                            **metrics
                        }

                        self.results.append(result)
                        accuracy = metrics['accuracy']
                        logger.info(f"✅ {model_name} - Accuracy: {accuracy:.3f}")

                        # 期望达到高性能
                        if accuracy > 0.80:
                            logger.info(f"🎯 High performance achieved: {accuracy:.3f}")

                    except Exception as e:
                        logger.error(f"❌ {model_name} failed: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

    def run_cross_domain_experiments(self):
        """Run cross-domain experiments with high performance"""
        logger.info("🔄 Running high-performance cross-domain experiments...")

        centers = self.loader.get_centers()

        # Focus on key pairs for efficiency
        key_pairs = [
            ('Cleveland', 'Hungarian'),
            ('Cleveland', 'VA'),
            ('Hungarian', 'Cleveland')
        ]

        for source_center, target_center in key_pairs:
            logger.info(f"Processing {source_center} → {target_center}")

            # Get source and target data
            source_data = self.loader.get_center_data(source_center)
            target_data = self.loader.get_center_data(target_center)

            if source_data is None or target_data is None:
                logger.warning(f"Missing data for {source_center} → {target_center}")
                continue

            X_source_raw = source_data.drop(['target', 'center'], axis=1, errors='ignore')
            y_source = source_data['target']
            X_target_raw = target_data.drop(['target', 'center'], axis=1, errors='ignore')
            y_target = target_data['target']

            # Preprocess data
            X_source = self.preprocess_data(X_source_raw)
            X_target = self.preprocess_data(X_target_raw)

            logger.info(f"Created domain adaptation split: {source_center}→{target_center} "
                       f"({len(X_source)} source, {len(X_target)} target)")

            for model_name, model in self.models.items():
                try:
                    logger.info(f"Fitting TCA on ({len(X_source)}, {X_source.shape[1]}) source, "
                               f"({len(X_target)}, {X_target.shape[1]}) target")

                    # PANDA with TCA domain adaptation
                    model.fit(X_source, y_source, X_target)
                    y_pred = model.predict(X_target, domain='target')
                    y_pred_proba = model.predict_proba(X_target, domain='target')

                    # Calculate metrics
                    metrics = self.calculate_metrics(y_target, y_pred, y_pred_proba)

                    # Store results
                    result = {
                        'experiment_type': 'cross_domain',
                        'model_name': model_name,
                        'source_center': source_center,
                        'target_center': target_center,
                        'source_size': len(X_source),
                        'target_size': len(X_target),
                        'train_size': len(X_source),
                        'val_size': len(X_target),
                        **metrics
                    }

                    self.results.append(result)
                    accuracy = metrics['accuracy']
                    logger.info(f"✅ {model_name} - Accuracy: {accuracy:.3f}")

                    # 期望达到高性能
                    if accuracy > 0.60:
                        logger.info(f"🎯 Good cross-domain performance: {accuracy:.3f}")

                except Exception as e:
                    logger.error(f"❌ {model_name} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    def save_results(self):
        """Save high-performance experimental results"""
        results_dir = Path("results") / f"high_performance_tca_{self.timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        df = pd.DataFrame(self.results)
        detailed_path = results_dir / "detailed_results.csv"
        df.to_csv(detailed_path, index=False)
        logger.info(f"💾 Detailed results saved to: {detailed_path}")

        # Calculate and save summary statistics
        if len(df) > 0:
            summary_stats = []

            for experiment_type in ['single_center', 'cross_domain']:
                exp_data = df[df['experiment_type'] == experiment_type]

                for model_name in exp_data['model_name'].unique():
                    model_data = exp_data[exp_data['model_name'] == model_name]

                    summary_stats.append({
                        'experiment_type': experiment_type,
                        'model_name': model_name,
                        'n_experiments': len(model_data),
                        'mean_accuracy': model_data['accuracy'].mean(),
                        'std_accuracy': model_data['accuracy'].std(),
                        'mean_auc': model_data['auc'].mean() if 'auc' in model_data.columns else 0,
                        'std_auc': model_data['auc'].std() if 'auc' in model_data.columns else 0,
                        'mean_sensitivity': model_data['sensitivity'].mean(),
                        'mean_specificity': model_data['specificity'].mean()
                    })

            summary_df = pd.DataFrame(summary_stats)
            summary_path = results_dir / "summary_statistics.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"💾 Summary statistics saved to: {summary_path}")

            # Print performance summary
            logger.info("\n🎯 PERFORMANCE SUMMARY:")
            for _, row in summary_df.iterrows():
                logger.info(f"{row['experiment_type']} {row['model_name']}: "
                           f"{row['mean_accuracy']:.3f} ± {row['std_accuracy']:.3f}")

            # Save experiment configuration
            config = {
                'timestamp': self.timestamp,
                'models': list(self.models.keys()),
                'total_experiments': len(self.results),
                'experiment_types': df['experiment_type'].unique().tolist() if len(df) > 0 else [],
                'framework': 'High-Performance TCA (mu=1.0, linear kernel)'
            }

            config_path = results_dir / "experiment_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"💾 Experiment configuration saved to: {config_path}")

        return results_dir

    def run_all_experiments(self):
        """Run all high-performance experiments"""
        logger.info("🚀 Starting High-Performance PANDA-Heart TCA Experiments...")
        logger.info(f"Available models: {list(self.models.keys())}")
        logger.info("Framework: High-Performance TCA (mu=1.0, linear kernel)")

        # Run single-center experiments
        self.run_single_center_experiments()

        # Run cross-domain experiments
        self.run_cross_domain_experiments()

        # Save results
        results_dir = self.save_results()

        logger.info(f"\n🎉 High-performance experiments completed!")
        logger.info(f"📁 Total results: {len(self.results)} experiments")
        logger.info(f"📂 Results directory: {results_dir}")

        return results_dir

def main():
    """Main execution function"""
    experiment_runner = HighPerformanceTCA()
    results_dir = experiment_runner.run_all_experiments()

    print(f"\n🎯 High-Performance TCA experiments completed successfully!")
    print(f"📊 Results saved in: {results_dir}")
    print(f"✅ Expecting 83%+ single-center accuracy with optimized TCA configuration")

if __name__ == "__main__":
    main()