import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin

# TabPFN imports
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("TabPFN not available.")

# Adapt library imports
try:
    from adapt.feature_based import TCA, CORAL
    ADAPT_AVAILABLE = True
except ImportError:
    try:
        from adapt.feature_based import TCA
        CORAL = None # CORAL might not be available in older versions or different paths
        ADAPT_AVAILABLE = True
    except ImportError:
        ADAPT_AVAILABLE = False
        print("adapt library not available.")

class PANDAAdapter(BaseEstimator, ClassifierMixin):
    """
    PANDA Adapter (TabPFN + UDA) for TableShift.
    Supports TCA and CORAL.
    """
    def __init__(self, 
                 adaptation_method='TCA', 
                 n_estimators=4, 
                 device='cpu', 
                 n_components=10,
                 kernel='rbf',
                 mu=0.1,
                 lambda_=1.0, # For CORAL
                 random_state=42):
        self.adaptation_method = adaptation_method
        self.n_estimators = n_estimators
        self.device = device
        self.n_components = n_components
        self.kernel = kernel
        self.mu = mu
        self.lambda_ = lambda_
        self.random_state = random_state
        
        self.scaler = None
        self.adapter = None
        self.classifier = None
        self.fitted = False

    def fit(self, X_source, y_source, X_target=None):
        """
        Fit the adapter using Source (labeled) and Target (unlabeled) data.
        """
        if not TABPFN_AVAILABLE or not ADAPT_AVAILABLE:
            raise RuntimeError("Dependencies missing.")
            
        # Input validation
        if isinstance(X_source, pd.DataFrame): X_source = X_source.values
        if isinstance(X_target, pd.DataFrame): X_target = X_target.values
        if isinstance(y_source, pd.Series): y_source = y_source.values
        
        # 1. Preprocessing (Scaling)
        # Critical for both TCA (kernel) and CORAL (covariance alignment)
        self.scaler = StandardScaler()
        X_source_scaled = self.scaler.fit_transform(X_source)
        
        if X_target is not None:
            X_target_scaled = self.scaler.transform(X_target)
        else:
            X_target_scaled = None

        # 2. Domain Adaptation
        if X_target_scaled is not None:
            if self.adaptation_method == 'TCA':
                # Ensure n_components <= n_features
                n_comps = min(self.n_components, X_source.shape[1])
                
                print(f"Fitting TCA with kernel={self.kernel}, n_components={n_comps}, mu={self.mu}...")
                self.adapter = TCA(
                    kernel=self.kernel,
                    mu=self.mu,
                    n_components=n_comps,
                    Xt=X_target_scaled,
                    verbose=0,
                    random_state=self.random_state
                )
                self.adapter.fit(X_source_scaled, y_source)
                X_source_transformed = self.adapter.transform(X_source_scaled)
                
            elif self.adaptation_method == 'CORAL':
                if CORAL is None:
                    raise ValueError("CORAL not available in adapt installation.")
                    
                print(f"Fitting CORAL with lambda={self.lambda_}...")
                self.adapter = CORAL(
                    lambda_=self.lambda_,
                    Xt=X_target_scaled,
                    verbose=0,
                    random_state=self.random_state
                )
                self.adapter.fit(X_source_scaled, y_source)
                X_source_transformed = self.adapter.transform(X_source_scaled)
                
            else:
                print(f"Unknown or No Adapt method {self.adaptation_method}, using raw scaled features.")
                X_source_transformed = X_source_scaled
        else:
            X_source_transformed = X_source_scaled

        # 3. TabPFN Classifier
        print(f"Fitting TabPFN with n_estimators={self.n_estimators}...")
        self.classifier = TabPFNClassifier(
            device=self.device,
            n_estimators=self.n_estimators,
            ignore_pretraining_limits=True,
            random_state=self.random_state
        )
        
        self.classifier.fit(X_source_transformed, y_source)
        self.fitted = True
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        if not self.fitted:
            raise RuntimeError("Model not fitted")
            
        if isinstance(X, pd.DataFrame): X = X.values
        
        # 1. Scale
        X_scaled = self.scaler.transform(X)
        
        # 2. Adapt
        if self.adapter is not None:
            X_transformed = self.adapter.transform(X_scaled)
        else:
            X_transformed = X_scaled
            
        # 3. Predict
        return self.classifier.predict_proba(X_transformed)