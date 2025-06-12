import pandas as pd
import numpy as np
import logging
import sys
import os
from typing import List, Tuple, Optional

# 添加项目根目录到路径以支持绝对导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import DATA_PATHS, LABEL_COL

def load_data_with_features(dataset_id: str, features: List[str], categorical_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据数据集ID和特征列表加载数据
    
    参数:
    - dataset_id: 数据集标识符 ('A', 'B', 'C')
    - features: 特征列表
    - categorical_indices: 类别特征索引（暂未使用，保持接口兼容性）
    
    返回:
    - X: 特征矩阵 (numpy array)
    - y: 标签向量 (numpy array)
    """
    if dataset_id not in DATA_PATHS:
        raise ValueError(f"不支持的数据集ID: {dataset_id}. 支持的ID: {list(DATA_PATHS.keys())}")
    
    data_path = DATA_PATHS[dataset_id]
    logging.info(f"从 {data_path} 加载数据集 {dataset_id}")
    
    try:
        # 读取Excel文件
        df = pd.read_excel(data_path)
        
        # 检查特征是否存在
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            error_msg = f"数据集 {dataset_id} 缺少以下特征: {missing_features}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # 检查标签列是否存在
        if LABEL_COL not in df.columns:
            error_msg = f"数据集 {dataset_id} 缺少标签列: {LABEL_COL}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        # 提取特征和标签
        X = df[features].values
        y = df[LABEL_COL].values.astype(np.int64)
        
        logging.info(f"成功加载数据集 {dataset_id}: {X.shape[0]} 样本, {X.shape[1]} 特征")
        logging.info(f"标签分布: {np.bincount(y)}")
        
        return X, y
        
    except FileNotFoundError:
        error_msg = f"数据文件未找到: {data_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"加载数据集 {dataset_id} 时发生错误: {e}"
        logging.error(error_msg)
        raise 