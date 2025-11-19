"""
PANDA-Heart Adapter using adapt library
Simplified implementation with TCA domain adaptation method only
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from sklearn.preprocessing import StandardScaler

# TabPFN imports
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    logging.warning("TabPFN not available. Install with: pip install tabpfn")

# Adapt library imports
try:
    from adapt.feature_based import TCA
    ADAPT_AVAILABLE = True
    logging.info("Using adapt library for domain adaptation (TCA only)")
except ImportError:
    ADAPT_AVAILABLE = False
    logging.warning("adapt library not available. Install with: pip install adapt")

logger = logging.getLogger(__name__)

class PANDAAdaptAdapter:
    """
    PANDA-Heart Domain Adapter using adapt library
    Strict implementation for TCA (Transfer Component Analysis) only.
    Includes StandardScaler for robust kernel adaptation.
    """

    def __init__(self, adaptation_method: str = 'TCA', random_state: int = 42, **kwargs):
        """
        Initialize PANDA Adapt Adapter

        Args:
            adaptation_method: Domain adaptation method (strictly 'TCA')
            random_state: Random seed
            **kwargs: Additional parameters for TCA
        """
        if not ADAPT_AVAILABLE:
            raise ImportError("adapt library not available. Install with: pip install adapt")

        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN not available. Install with: pip install tabpfn")

        # Strict enforcement of TCA
        if adaptation_method != 'TCA':
            logger.warning(f"Method '{adaptation_method}' requested but only 'TCA' is supported in this strict mode. Using TCA.")
        
        self.adaptation_method = 'TCA'
        self.random_state = random_state
        self.adapt_params = kwargs
        self.fitted = False
        self.scaler = None

        # Initialize TabPFN
        self.tabpfn_classifier = TabPFNClassifier(
            device='auto',
            n_estimators=32,
            random_state=random_state
        )

        logger.info(f"Initialized PANDAAdaptAdapter with TCA")

    def _create_adapt_method(self, X_source, X_target):
        """Create TCA domain adaptation method with winning parameters"""
        # Winning TCA parameters from optimization
        params = {
            'kernel': 'rbf',
            'mu': 0.1,
            'gamma': 0.1,
            'n_components': min(9, X_source.shape[1]), # Optimized dimension
            'Xt': X_target,  # Target domain needed for TCA
            'verbose': 0
        }
        params.update(self.adapt_params)
        return TCA(**params)

    def fit(self, X_source, y_source, X_target):
        """
        Fit domain adapter on source data with target domain for adaptation

        Args:
            X_source: Source domain features
            y_source: Source domain labels
            X_target: Target domain features (unlabeled)

        Returns:
            Self (fitted adapter)
        """
        # Convert to numpy arrays
        if hasattr(X_source, 'values'):
            X_source = X_source.values
        if hasattr(X_target, 'values'):
            X_target = X_target.values
        if hasattr(y_source, 'values'):
            y_source = y_source.values

        logger.info(f"Fitting TCA on {X_source.shape} source, {X_target.shape} target")

        try:
            # 1. Standardize Features (Critical for RBF Kernel)
            self.scaler = StandardScaler()
            X_source_scaled = self.scaler.fit_transform(X_source)
            X_target_scaled = self.scaler.transform(X_target)

            # 2. Create and fit TCA
            self.adapt_method = self._create_adapt_method(X_source_scaled, X_target_scaled)

            # Fit TCA (Unsupervised alignment)
            # Using fit_transform(Xs, Xt) signature based on optimization results
            self.X_source_transformed = self.adapt_method.fit_transform(X_source_scaled, X_target_scaled)
            # We don't strictly need X_target_transformed for training, but good to have
            self.X_target_transformed = self.adapt_method.transform(X_target_scaled)

            # 3. Fit TabPFN on transformed source data
            self.tabpfn_classifier.fit(self.X_source_transformed, y_source)

            self.fitted = True
            logger.info(f"Successfully fitted TCA adapter with StandardScaler")

        except Exception as e:
            logger.error(f"Error fitting TCA: {e}")
            # Fallback mechanism
            logger.warning("Falling back to raw features due to TCA error.")
            self.scaler = None # Reset scaler
            self.X_source_transformed = X_source
            self.X_target_transformed = X_target
            self.tabpfn_classifier.fit(X_source, y_source)
            self.fitted = True

        return self

    def predict(self, X, domain='target'):
        """
        Make predictions on data
        """
        if not self.fitted:
            raise ValueError("Adapter not fitted. Call fit() first.")

        # Convert to numpy array
        if hasattr(X, 'values'):
            X = X.values

        # 1. Scale
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # 2. Transform (TCA)
        if self.adapt_method is not None and hasattr(self, 'adapt_method'):
            try:
                X_transformed = self.adapt_method.transform(X)
            except Exception:
                 X_transformed = X # Fallback
        else:
            X_transformed = X

        return self.tabpfn_classifier.predict(X_transformed)

    def predict_proba(self, X, domain='target'):
        """
        Make probability predictions on data
        """
        if not self.fitted:
            raise ValueError("Adapter not fitted. Call fit() first.")

        # Convert to numpy array
        if hasattr(X, 'values'):
            X = X.values

        # 1. Scale
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # 2. Transform (TCA)
        if self.adapt_method is not None and hasattr(self, 'adapt_method'):
            try:
                X_transformed = self.adapt_method.transform(X)
            except Exception:
                 X_transformed = X # Fallback
        else:
            X_transformed = X

        return self.tabpfn_classifier.predict_proba(X_transformed)

    def __repr__(self):
        return f"PANDAAdaptAdapter(method='{self.adaptation_method}', fitted={self.fitted})"