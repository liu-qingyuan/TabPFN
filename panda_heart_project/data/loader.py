"""
Heart Disease Data Loader for PANDA Framework
Handles UCI multi-center heart disease dataset loading, preprocessing, and split generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import logging

logger = logging.getLogger(__name__)

class HeartDiseaseDataLoader:
    """
    Heart Disease Data Loader for PANDA Heart Project
    Loads and preprocesses UCI multi-center heart disease datasets
    """

    # Clinical feature configuration
    # Selected features based on intersection across all 4 centers (Cleveland, Hungarian, Switzerland, VA)
    # Dropped features with high missing rates (>50% in any center) to prevent imputation-induced domain shift:
    # - fbs (61% missing in Switzerland)
    # - slope (65% missing in Hungarian)
    # - ca (99% missing in Hungarian/VA)
    # - thal (90% missing in Hungarian)
    CLINICAL_FEATURES = {
        'age': {'type': 'continuous', 'scaler': 'clinical_normalized'},
        'sex': {'type': 'binary', 'scaler': None},
        'cp': {'type': 'ordinal', 'scaler': None, 'range': (1, 4)},
        'trestbps': {'type': 'continuous', 'scaler': 'clinical_normalized'},
        'chol': {'type': 'continuous', 'scaler': 'clinical_normalized'},
        # 'fbs': {'type': 'binary', 'scaler': None},
        'restecg': {'type': 'ordinal', 'scaler': None, 'range': (0, 2)},
        'thalach': {'type': 'continuous', 'scaler': 'age_adjusted'},
        'exang': {'type': 'binary', 'scaler': None},
        'oldpeak': {'type': 'continuous', 'scaler': 'robust_scaling'},
        # 'slope': {'type': 'ordinal', 'scaler': None, 'range': (1, 3)},
        # 'ca': {'type': 'numeric', 'scaler': 'normalized', 'range': (0, 3)},
        # 'thal': {'type': 'categorical', 'scaler': 'one_hot', 'values': [3, 6, 7]}
    }

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize Heart Disease Data Loader

        Args:
            data_path: Path to the processed UCI heart disease dataset
                      If None, uses default path
        """
        if data_path is None:
            data_path = Path(__file__).parent / "processed" / "uci_heart_disease_combined.csv"
        else:
            data_path = Path(data_path)

        self.data_path = data_path
        self.df = None
        self.processed_data = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_names = list(self.CLINICAL_FEATURES.keys())

        logger.info(f"Initialized HeartDiseaseDataLoader with data path: {data_path}")

    def load_data(self) -> pd.DataFrame:
        """Load the combined heart disease dataset"""
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")

            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset: {len(self.df)} samples from {self.df['center'].nunique()} centers")

            # Validate required columns
            required_cols = self.feature_names + ['center', 'target']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            return self.df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def get_center_data(self, center_name: str) -> pd.DataFrame:
        """Get data for a specific center"""
        if self.df is None:
            self.load_data()

        center_data = self.df[self.df['center'] == center_name].copy()
        if len(center_data) == 0:
            raise ValueError(f"No data found for center: {center_name}")

        logger.info(f"Retrieved {len(center_data)} samples for {center_name}")
        return center_data

    def get_centers(self) -> List[str]:
        """Get list of available centers"""
        if self.df is None:
            self.load_data()
        return self.df['center'].unique().tolist()

    def preprocess_features(self, X: pd.DataFrame, fit_scalers: bool = True, center_name: str = None) -> pd.DataFrame:
        """
        Preprocess features according to clinical requirements

        Args:
            X: Feature DataFrame
            fit_scalers: Whether to fit scalers (True for training, False for testing)
            center_name: Name of the center for center-specific preprocessing

        Returns:
            Preprocessed features DataFrame
        """
        X_processed = X.copy()

        # Explicitly drop target column 'num' if present to prevent data leakage
        if 'num' in X_processed.columns:
            X_processed.drop('num', axis=1, inplace=True)

        for feature_name, config in self.CLINICAL_FEATURES.items():
            if feature_name not in X_processed.columns:
                continue

            feature_data = X_processed[feature_name].values.reshape(-1, 1)

            # Handle missing values
            if np.any(pd.isna(X_processed[feature_name])):
                imputer_key = f"{center_name}_{feature_name}" if center_name else feature_name

                if config['type'] == 'continuous':
                    imputer = KNNImputer(n_neighbors=5) if fit_scalers else self.imputers.get(imputer_key)
                else:
                    imputer = SimpleImputer(strategy='most_frequent') if fit_scalers else self.imputers.get(imputer_key)

                if fit_scalers:
                    imputer.fit(feature_data)
                    self.imputers[imputer_key] = imputer

                if imputer is not None:
                    X_processed[feature_name] = imputer.transform(feature_data).ravel()

            # Apply scaling/transformation
            scaler_key = f"{center_name}_{feature_name}" if center_name else feature_name

            if config['scaler'] is None:
                continue  # No scaling needed

            elif config['scaler'] == 'normalized':
                scaler = StandardScaler() if fit_scalers else self.scalers.get(scaler_key)
                if fit_scalers:
                    scaler.fit(X_processed[[feature_name]])
                    self.scalers[scaler_key] = scaler
                if scaler is not None:
                    X_processed[feature_name] = scaler.transform(X_processed[[feature_name]]).ravel()

            elif config['scaler'] == 'clinical_normalized':
                # Clinical normalization based on medical reference ranges
                if feature_name == 'age':
                    # Age normalized to 0-1 (assuming range 20-80)
                    X_processed[feature_name] = (X_processed[feature_name] - 20) / 60
                elif feature_name == 'trestbps':
                    # BP normalized (90-140 mmHg normal range)
                    X_processed[feature_name] = (X_processed[feature_name] - 90) / 50
                elif feature_name == 'chol':
                    # Cholesterol normalized (150-300 mg/dl)
                    X_processed[feature_name] = (X_processed[feature_name] - 150) / 150

            elif config['scaler'] == 'robust_scaling':
                scaler = RobustScaler() if fit_scalers else self.scalers.get(scaler_key)
                if fit_scalers:
                    scaler.fit(X_processed[[feature_name]])
                    self.scalers[scaler_key] = scaler
                if scaler is not None:
                    X_processed[feature_name] = scaler.transform(X_processed[[feature_name]]).ravel()

            elif config['scaler'] == 'age_adjusted' and feature_name == 'thalach':
                # Maximum heart rate adjusted for age (220 - age is theoretical max)
                if 'age' in X_processed.columns:
                    age_normalized = X_processed['age'] * 60 + 20  # Denormalize age
                    theoretical_max_hr = 220 - age_normalized
                    X_processed[feature_name] = X_processed[feature_name] / theoretical_max_hr

            elif config['scaler'] == 'one_hot' and config['type'] == 'categorical':
                # One-hot encoding for categorical variables
                dummies = pd.get_dummies(X_processed[feature_name], prefix=feature_name)
                X_processed = pd.concat([X_processed, dummies], axis=1)
                X_processed.drop(feature_name, axis=1, inplace=True)

        # Final safety check for NaN values
        if X_processed.isnull().values.any():
            nan_count = X_processed.isnull().sum().sum()
            logger.warning(f"Found {nan_count} NaN values after preprocessing. Applying final mean imputation.")
            
            # Fill remaining NaNs with 0 (safe fallback for scaled/normalized data where mean~0)
            # Using 0 is safer than mean here because scalers might have made mean=0 anyway
            X_processed.fillna(0, inplace=True)
            
        return X_processed

    def create_single_center_split(self, center_name: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Create train/test split for single center experiments

        Args:
            center_name: Name of the center
            test_size: Proportion of test data
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        center_data = self.get_center_data(center_name)

        X = center_data[self.feature_names].copy()
        y = center_data['target'].copy()

        # Preprocess features
        X_processed = self.preprocess_features(X, fit_scalers=True, center_name=center_name)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, random_state=random_state, stratify=y
        )

        logger.info(f"Created {center_name} split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def create_cross_validation_folds(self, center_name: str, n_folds: int = 10, random_state: int = 42) -> List[Tuple]:
        """
        Create stratified k-fold cross-validation splits for single center

        Args:
            center_name: Name of the center
            n_folds: Number of CV folds
            random_state: Random seed

        Returns:
            List of (X_train, X_val, y_train, y_val) tuples
        """
        center_data = self.get_center_data(center_name)

        X = center_data[self.feature_names].copy()
        y = center_data['target'].copy()

        # Preprocess features
        X_processed = self.preprocess_features(X, fit_scalers=True, center_name=center_name)

        # Create CV folds
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        folds = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_processed, y)):
            X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            folds.append((X_train, X_val, y_train, y_val))

        logger.info(f"Created {n_folds}-fold CV for {center_name}")
        return folds

    def create_domain_adaptation_split(self, source_center: str, target_center: str) -> Tuple:
        """
        Create source/target split for domain adaptation experiments

        Args:
            source_center: Source domain center name
            target_center: Target domain center name

        Returns:
            Tuple of (X_source, y_source, X_target, y_target)
        """
        source_data = self.get_center_data(source_center)
        target_data = self.get_center_data(target_center)

        X_source = source_data[self.feature_names].copy()
        y_source = source_data['target'].copy()

        X_target = target_data[self.feature_names].copy()
        y_target = target_data['target'].copy()

        # Preprocess features (fit separately for each domain to preserve domain characteristics)
        X_source_processed = self.preprocess_features(X_source, fit_scalers=True, center_name=source_center)
        X_target_processed = self.preprocess_features(X_target, fit_scalers=True, center_name=target_center)

        logger.info(f"Created domain adaptation split: {source_center}→{target_center} "
                   f"({len(X_source_processed)} source, {len(X_target_processed)} target)")

        return X_source_processed, y_source, X_target_processed, y_target

    def create_loco_splits(self) -> List[Tuple]:
        """
        Create Leave-One-Center-Out (LOCO) cross-validation splits

        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        centers = self.get_centers()
        loco_splits = []

        for test_center in centers:
            train_centers = [c for c in centers if c != test_center]

            # Combine training centers
            train_data_list = [self.get_center_data(center) for center in train_centers]
            train_data = pd.concat(train_data_list, ignore_index=True)
            test_data = self.get_center_data(test_center)

            X_train = train_data[self.feature_names].copy()
            y_train = train_data['target'].copy()

            X_test = test_data[self.feature_names].copy()
            y_test = test_data['target'].copy()

            # Preprocess features
            X_train_processed = self.preprocess_features(X_train, fit_scalers=True, center_name="loco_train")
            X_test_processed = self.preprocess_features(X_test, fit_scalers=False, center_name="loco_test")

            loco_splits.append((X_train_processed, X_test_processed, y_train, y_test))

            logger.info(f"Created LOCO split: test on {test_center}, train on {', '.join(train_centers)}")

        return loco_splits

    def get_data_statistics(self) -> Dict:
        """Get comprehensive data statistics"""
        if self.df is None:
            self.load_data()

        stats = {
            'total_samples': len(self.df),
            'centers': {},
            'overall_stats': {}
        }

        # Overall statistics
        stats['overall_stats'] = {
            'target_distribution': self.df['target'].value_counts().to_dict(),
            'sex_distribution': self.df['sex'].value_counts().to_dict(),
            'age_stats': {
                'mean': float(self.df['age'].mean()),
                'std': float(self.df['age'].std()),
                'min': float(self.df['age'].min()),
                'max': float(self.df['age'].max())
            }
        }

        # Per-center statistics
        for center in self.get_centers():
            center_data = self.df[self.df['center'] == center]
            stats['centers'][center] = {
                'sample_count': len(center_data),
                'target_distribution': center_data['target'].value_counts().to_dict(),
                'missing_values': center_data[self.feature_names].isnull().sum().to_dict(),
                'age_stats': {
                    'mean': float(center_data['age'].mean()),
                    'std': float(center_data['age'].std()),
                    'min': float(center_data['age'].min()),
                    'max': float(center_data['age'].max())
                }
            }

        return stats

    def __repr__(self):
        return f"HeartDiseaseDataLoader(data_path='{self.data_path}')"


# Convenience function for quick loading
def load_heart_disease_data(data_path: Optional[str] = None) -> HeartDiseaseDataLoader:
    """
    Quick load function for heart disease data

    Args:
        data_path: Optional path to data file

    Returns:
        HeartDiseaseDataLoader instance
    """
    loader = HeartDiseaseDataLoader(data_path)
    loader.load_data()
    return loader


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Load data
    loader = load_heart_disease_data()

    # Print statistics
    stats = loader.get_data_statistics()
    print(f"📊 Dataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Centers: {list(stats['centers'].keys())}")

    # Example: Create single center split
    if 'Cleveland' in stats['centers']:
        X_train, X_test, y_train, y_test = loader.create_single_center_split('Cleveland')
        print(f"\n🏥 Cleveland split:")
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Example: Create domain adaptation split
    centers = loader.get_centers()
    if len(centers) >= 2:
        X_source, y_source, X_target, y_target = loader.create_domain_adaptation_split(
            centers[0], centers[1]
        )
        print(f"\n🔄 Domain adaptation {centers[0]} → {centers[1]}:")
        print(f"Source: {X_source.shape}, Target: {X_target.shape}")