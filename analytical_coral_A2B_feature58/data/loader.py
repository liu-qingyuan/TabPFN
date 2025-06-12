import pandas as pd
import numpy as np
import logging

def load_excel(path: str, features: list[str], label_col: str) -> tuple[np.ndarray, np.ndarray]:
    """从Excel加载数据，检查特征，返回X, y (numpy arrays)"""
    logging.info(f"Loading data from: {path}")
    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {path}")
        raise
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        error_msg = f"Dataset at {path} is missing the following required features: {missing_features}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # 确保标签列存在
    if label_col not in df.columns:
        error_msg = f"Dataset at {path} is missing the label column: {label_col}"
        logging.error(error_msg)
        raise ValueError(error_msg)
        
    X = df[features].values
    y = df[label_col].values.astype(np.int64)
    logging.info(f"Successfully loaded {len(df)} records from {path}.")
    return X, y

def load_all_datasets(data_path_a: str, data_path_b: str, features: list[str], label_col: str) -> dict:
    """加载所有需要的数据集 (A 和 B)"""
    datasets = {}
    logging.info("Loading Dataset A (AI4Health)...")
    datasets['X_A'], datasets['y_A'] = load_excel(data_path_a, features, label_col)
    
    logging.info("Loading Dataset B (Henan)...")
    datasets['X_B'], datasets['y_B'] = load_excel(data_path_b, features, label_col)
    
    return datasets 