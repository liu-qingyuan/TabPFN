"""
Baseline Models for PANDA-Heart Project
Implements 7 baseline models for comparison with PANDA framework
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import logging

# TabPFN imports
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    logging.warning("TabPFN not available. Install with: pip install tabpfn")

# sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve
from sklearn.base import BaseEstimator, ClassifierMixin

# XGBoost imports
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

logger = logging.getLogger(__name__)

class BaselineModel(BaseEstimator, ClassifierMixin):
    """
    Base class for all baseline models
    Provides common functionality for training, prediction, and evaluation
    """

    def __init__(self, model_name: str, random_state: int = 42, **kwargs):
        """
        Initialize baseline model

        Args:
            model_name: Name of the model
            random_state: Random seed for reproducibility
            **kwargs: Model-specific parameters
        """
        self.model_name = model_name
        self.random_state = random_state
        self.kwargs = kwargs
        self.classifier = None
        self.feature_scaler = StandardScaler()
        self.fitted = False

        logger.info(f"Initialized {model_name} baseline model")

    @abstractmethod
    def _create_classifier(self) -> BaseEstimator:
        """Create the underlying classifier"""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaselineModel':
        """
        Fit the baseline model

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            Self for method chaining
        """
        X = np.asarray(X)
        y = np.asarray(y)

        logger.info(f"Fitting {self.model_name} on {len(X)} samples with {X.shape[1]} features")

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)

        # Create and train classifier
        self.classifier = self._create_classifier()
        self.classifier.fit(X_scaled, y)

        self.fitted = True
        logger.info(f"{self.model_name} fitted successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict on test data

        Args:
            X: Test features (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        X_scaled = self.feature_scaler.transform(X)
        predictions = self.classifier.predict(X_scaled)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities on test data

        Args:
            X: Test features (n_samples, n_features)

        Returns:
            Predicted probabilities (n_samples, n_classes)
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        X_scaled = self.feature_scaler.transform(X)

        if hasattr(self.classifier, "predict_proba"):
            probabilities = self.classifier.predict_proba(X_scaled)
        else:
            # Fallback for models without probability prediction
            predictions = self.classifier.predict(X_scaled)
            n_classes = len(np.unique(predictions))
            probabilities = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                probabilities[i, pred] = 1.0

        return probabilities

    def score(self, X: np.ndarray, y: np.ndarray, metric: str = 'accuracy') -> float:
        """
        Score the model on test data

        Args:
            X: Test features (n_samples, n_features)
            y: True labels (n_samples,)
            metric: Scoring metric ('accuracy', 'auc')

        Returns:
            Score value
        """
        y_pred = self.predict(X)

        if metric == 'accuracy':
            return accuracy_score(y, y_pred)
        elif metric == 'auc':
            y_proba = self.predict_proba(X)
            if len(np.unique(y)) == 2:
                return roc_auc_score(y, y_proba[:, 1])
            else:
                return roc_auc_score(y, y_proba, multi_class='ovr')
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available

        Returns:
            Feature importance array or None
        """
        if not self.fitted:
            return None

        if hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        elif hasattr(self.classifier, 'coef_'):
            return np.abs(self.classifier.coef_).flatten()
        else:
            return None

    def __repr__(self):
        return f"{self.__class__.__name__}(model_name='{self.model_name}', fitted={self.fitted})"


class LASSOLRModel(BaselineModel):
    """LASSO Logistic Regression baseline model"""

    def _create_classifier(self) -> LogisticRegression:
        """Create LASSO logistic regression classifier"""
        return LogisticRegression(
            penalty='l1',
            solver='saga',
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000,
            **self.kwargs
        )


class XGBoostModel(BaselineModel):
    """XGBoost baseline model"""

    def _create_classifier(self) -> XGBClassifier:
        """Create XGBoost classifier"""
        if not XGBOOST_AVAILABLE or XGBClassifier is None:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")

        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='logloss',
            **self.kwargs
        )


class RandomForestModel(BaselineModel):
    """Random Forest baseline model"""

    def _create_classifier(self) -> RandomForestClassifier:
        """Create Random Forest classifier"""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            **self.kwargs
        )


class SVMModel(BaselineModel):
    """Support Vector Machine baseline model"""

    def _create_classifier(self) -> SVC:
        """Create SVM classifier"""
        return SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            class_weight='balanced',
            random_state=self.random_state,
            **self.kwargs
        )


class KNNModel(BaselineModel):
    """K-Nearest Neighbors baseline model"""

    def _create_classifier(self) -> KNeighborsClassifier:
        """Create KNN classifier"""
        return KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            algorithm='auto',
            **self.kwargs
        )


class TabPFNOnlyModel(BaselineModel):
    """TabPFN without domain adaptation baseline model"""

    def _create_classifier(self) -> TabPFNClassifier:
        """Create TabPFN classifier"""
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN not available. Install with: pip install tabpfn")

        return TabPFNClassifier(
            device='auto',
            random_state=self.random_state,
            **self.kwargs
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TabPFNOnlyModel':
        """
        Fit TabPFN model (no scaling for TabPFN as it handles normalization internally)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        logger.info(f"Fitting TabPFN on {len(X)} samples with {X.shape[1]} features")

        # TabPFN doesn't require feature scaling
        self.classifier = self._create_classifier()
        self.classifier.fit(X, y)

        # Store feature scaler for consistency with other models
        self.feature_scaler.fit(X)  # Fit but don't transform for TabPFN

        self.fitted = True
        logger.info("TabPFN fitted successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with TabPFN (no scaling)"""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        # TabPFN handles its own preprocessing
        predictions = self.classifier.predict(X)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with TabPFN (no scaling)"""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)
        # TabPFN handles its own preprocessing
        probabilities = self.classifier.predict_proba(X)

        return probabilities


class BaselineModelFactory:
    """Factory for creating baseline models"""

    BASELINE_MODELS = {
        'LASSO_LR': LASSOLRModel,
        'XGBoost': XGBoostModel,
        'Random_Forest': RandomForestModel,
        'SVM': SVMModel,
        'KNN': KNNModel,
        'TabPFN_Only': TabPFNOnlyModel
    }

    @staticmethod
    def create_model(model_name: str, **kwargs) -> BaselineModel:
        """
        Create a baseline model by name

        Args:
            model_name: Name of the model
            **kwargs: Model-specific parameters

        Returns:
            Baseline model instance
        """
        if model_name not in BaselineModelFactory.BASELINE_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(BaselineModelFactory.BASELINE_MODELS.keys())}")

        model_class = BaselineModelFactory.BASELINE_MODELS[model_name]
        return model_class(model_name=model_name, **kwargs)

    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available baseline models"""
        return list(BaselineModelFactory.BASELINE_MODELS.keys())

    @staticmethod
    def create_all_models(**common_kwargs) -> Dict[str, BaselineModel]:
        """
        Create all available baseline models

        Args:
            **common_kwargs: Common parameters for all models

        Returns:
            Dictionary of model_name -> model_instance
        """
        models = {}
        for model_name in BaselineModelFactory.get_available_models():
            try:
                models[model_name] = BaselineModelFactory.create_model(model_name, **common_kwargs)
                logger.info(f"Created {model_name} baseline model")
            except Exception as e:
                logger.warning(f"Failed to create {model_name}: {e}")

        return models


# Convenience functions for creating specific models
def create_lasso_lr(**kwargs) -> LASSOLRModel:
    """Create LASSO Logistic Regression model"""
    return BaselineModelFactory.create_model('LASSO_LR', **kwargs)


def create_xgboost(**kwargs) -> XGBoostModel:
    """Create XGBoost model"""
    return BaselineModelFactory.create_model('XGBoost', **kwargs)


def create_random_forest(**kwargs) -> RandomForestModel:
    """Create Random Forest model"""
    return BaselineModelFactory.create_model('Random_Forest', **kwargs)


def create_svm(**kwargs) -> SVMModel:
    """Create SVM model"""
    return BaselineModelFactory.create_model('SVM', **kwargs)


def create_knn(**kwargs) -> KNNModel:
    """Create KNN model"""
    return BaselineModelFactory.create_model('KNN', **kwargs)


def create_tabpfn_only(**kwargs) -> TabPFNOnlyModel:
    """Create TabPFN only model"""
    return BaselineModelFactory.create_model('TabPFN_Only', **kwargs)


def test_all_baselines(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Test all baseline models on given data

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary of model_name -> {metric_name: score}
    """
    results = {}

    for model_name in BaselineModelFactory.get_available_models():
        logger.info(f"Testing {model_name}")

        try:
            # Create and train model
            model = BaselineModelFactory.create_model(model_name, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            accuracy = model.score(X_test, y_test, metric='accuracy')
            auc_score = model.score(X_test, y_test, metric='auc')

            results[model_name] = {
                'accuracy': accuracy,
                'auc': auc_score
            }

            logger.info(f"{model_name}: Accuracy = {accuracy:.3f}, AUC = {auc_score:.3f}")

        except Exception as e:
            logger.error(f"{model_name} failed: {e}")
            results[model_name] = {
                'accuracy': np.nan,
                'auc': np.nan,
                'error': str(e)
            }

    return results


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data for testing
    np.random.seed(42)
    n_train, n_test = 300, 100
    n_features = 13

    X_train = np.random.randn(n_train, n_features)
    y_train = (X_train[:, 0] + X_train[:, 1] + np.random.randn(n_train) * 0.1 > 0).astype(int)

    X_test = np.random.randn(n_test, n_features)
    y_test = (X_test[:, 0] + X_test[:, 1] + np.random.randn(n_test) * 0.1 > 0).astype(int)

    print("🚀 Testing Baseline Models")
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

    # Test all baseline models
    results = test_all_baselines(X_train, y_train, X_test, y_test)

    print("\n📊 Results Summary:")
    print("-" * 60)
    print(f"{'Model':<15} {'Accuracy':<10} {'AUC':<10} {'Status':<15}")
    print("-" * 60)

    for model_name, metrics in results.items():
        accuracy = metrics.get('accuracy', np.nan)
        auc_score = metrics.get('auc', np.nan)
        status = "✅ Success" if not np.isnan(accuracy) else "❌ Failed"

        print(f"{model_name:<15} {accuracy:<10.3f} {auc_score:<10.3f} {status:<15}")

    print("-" * 60)

    # Test individual model creation
    print(f"\n🔧 Testing Individual Model Creation:")
    for model_name in ['LASSO_LR', 'XGBoost', 'TabPFN_Only']:
        try:
            model = BaselineModelFactory.create_model(model_name, random_state=42)
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test, metric='accuracy')
            print(f"  ✅ {model_name}: {accuracy:.3f}")
        except Exception as e:
            print(f"  ❌ {model_name}: {e}")

    print("\n✨ Baseline Models testing completed!")