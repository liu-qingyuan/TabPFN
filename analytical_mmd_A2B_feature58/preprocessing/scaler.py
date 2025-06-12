import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Optional, Tuple, List

def fit_scaler(X: np.ndarray, categorical_indices: Optional[List[int]] = None) -> StandardScaler:
    """拟合StandardScaler，只对非分类特征进行缩放"""
    scaler = StandardScaler()
    
    if categorical_indices is None or len(categorical_indices) == 0:
        # 没有分类特征，对所有特征缩放
        scaler.fit(X)
        logging.info("StandardScaler fitted on all features.")
    else:
        # 只对非分类特征缩放
        numeric_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]
        if len(numeric_indices) > 0:
            X_numeric = X[:, numeric_indices]
            scaler.fit(X_numeric)
            logging.info(f"StandardScaler fitted on {len(numeric_indices)} numeric features (excluding {len(categorical_indices)} categorical features).")
        else:
            logging.warning("所有特征都是分类特征，跳过标准化")
            # 创建一个空的scaler，实际上不会进行任何变换
            scaler = None
    
    return scaler

def apply_scaler(scaler: StandardScaler, X: np.ndarray, categorical_indices: Optional[List[int]] = None) -> np.ndarray:
    """应用已拟合的StandardScaler，只对非分类特征进行缩放"""
    if scaler is None:
        # 没有有效的scaler，返回原始数据
        logging.info("No valid scaler, returning original data.")
        return X.copy()
    
    if categorical_indices is None or len(categorical_indices) == 0:
        # 没有分类特征，对所有特征缩放
        X_scaled = scaler.transform(X)
        if not isinstance(X_scaled, np.ndarray):
            logging.warning("Scaler transform did not return a numpy array. Attempting conversion.")
            X_scaled = np.array(X_scaled) 
        logging.info("StandardScaler applied to all features.")
        return X_scaled
    else:
        # 只对非分类特征缩放
        numeric_indices = [i for i in range(X.shape[1]) if i not in categorical_indices]
        X_result = X.copy()
        
        if len(numeric_indices) > 0:
            X_numeric = X[:, numeric_indices]
            X_numeric_scaled = scaler.transform(X_numeric)
            X_result[:, numeric_indices] = X_numeric_scaled
            logging.info(f"StandardScaler applied to {len(numeric_indices)} numeric features, {len(categorical_indices)} categorical features kept unchanged.")
        else:
            logging.info("所有特征都是分类特征，无需标准化")
        
        return X_result

def fit_apply_scaler(X_source: np.ndarray, X_target: Optional[np.ndarray] = None, 
                    categorical_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
    """拟合源数据并应用于源数据和目标数据（如果提供），只对非分类特征进行缩放"""
    scaler = fit_scaler(X_source, categorical_indices)
    X_source_scaled = apply_scaler(scaler, X_source, categorical_indices)
    X_target_scaled: Optional[np.ndarray] = None
    if X_target is not None:
        X_target_scaled = apply_scaler(scaler, X_target, categorical_indices)
    return X_source_scaled, X_target_scaled, scaler 