import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Optional, Tuple

def fit_scaler(X: np.ndarray) -> StandardScaler:
    """拟合StandardScaler"""
    scaler = StandardScaler()
    scaler.fit(X)
    logging.info("StandardScaler fitted.")
    return scaler

def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """应用已拟合的StandardScaler"""
    X_scaled = scaler.transform(X)
    if not isinstance(X_scaled, np.ndarray):
        logging.warning("Scaler transform did not return a numpy array. Attempting conversion.")
        X_scaled = np.array(X_scaled) 
    logging.info("StandardScaler applied to data.")
    return X_scaled

def fit_apply_scaler(X_source: np.ndarray, X_target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """拟合源数据并应用于源数据和目标数据（如果提供）"""
    scaler = fit_scaler(X_source)
    X_source_scaled = apply_scaler(scaler, X_source)
    X_target_scaled: Optional[np.ndarray] = None
    if X_target is not None:
        X_target_scaled = apply_scaler(scaler, X_target)
    return X_source_scaled, X_target_scaled, scaler 