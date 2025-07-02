# -*- coding: utf-8 -*-
"""
标准化处理模块

提供StandardScaler和RobustScaler两种标准化方法，
专门针对医疗数据的特点进行优化处理。

Author: UDA Medical Project Team
Date: 2024
"""

from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


class MedicalDataScaler(BaseEstimator, TransformerMixin):
    """
    医疗数据标准化器
    
    支持StandardScaler和RobustScaler两种标准化方法，
    能够处理混合数据类型（数值+类别特征）
    """
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 categorical_features: Optional[list] = None,
                 handle_missing: bool = True):
        """
        初始化标准化器
        
        Args:
            scaler_type: 标准化类型 ('standard' | 'robust')
            categorical_features: 类别特征列表
            handle_missing: 是否处理缺失值
        """
        self.scaler_type = scaler_type.lower()
        self.categorical_features = categorical_features or []
        self.handle_missing = handle_missing
        
        # 验证标准化类型
        if self.scaler_type not in ['standard', 'robust', 'none']:
            raise ValueError(f"不支持的标准化类型: {scaler_type}. 支持的类型: ['standard', 'robust', 'none']")
        
        # 初始化标准化器
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:  # none
            self.scaler = None
        
        # 存储拟合信息
        self.numerical_features_ = None
        self.feature_names_ = None
        self.is_fitted_ = False
        
        logger.info(f"初始化医疗数据标准化器: {scaler_type}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None):
        """
        拟合标准化器
        
        Args:
            X: 输入特征矩阵
            y: 目标变量（未使用）
            
        Returns:
            self: 返回自身实例
        """
        # 转换为DataFrame便于处理
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        self.feature_names_ = list(X_df.columns)
        
        # 识别数值特征（排除类别特征）
        self.numerical_features_ = [
            col for col in self.feature_names_ 
            if col not in self.categorical_features
        ]
        
        logger.info(f"识别到 {len(self.numerical_features_)} 个数值特征，{len(self.categorical_features)} 个类别特征")
        
        # 仅对数值特征进行标准化拟合
        if self.numerical_features_ and self.scaler_type != 'none':
            X_numerical = X_df[self.numerical_features_]
            
            # 处理缺失值
            if self.handle_missing:
                X_numerical = X_numerical.fillna(X_numerical.median())
            
            # 拟合标准化器
            self.scaler.fit(X_numerical)
            
            logger.info(f"标准化器拟合完成 - 类型: {self.scaler_type}")
            if self.scaler_type == 'standard':
                logger.info(f"均值: {self.scaler.mean_[:3]}...")
                logger.info(f"标准差: {self.scaler.scale_[:3]}...")
            elif self.scaler_type == 'robust':
                logger.info(f"中位数: {self.scaler.center_[:3]}...")
                logger.info(f"IQR: {self.scaler.scale_[:3]}...")
        elif self.scaler_type == 'none':
            logger.info("跳过标准化 - 类型: none")
        
        self.is_fitted_ = True
        return self
    
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        应用标准化变换
        
        Args:
            X: 输入特征矩阵
            
        Returns:
            X_scaled: 标准化后的特征矩阵
        """
        if not self.is_fitted_:
            raise ValueError("标准化器尚未拟合，请先调用 fit() 方法")
        
        # 转换为DataFrame便于处理
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_)
        else:
            X_df = X.copy()
        
        # 创建输出数组
        X_scaled = X_df.values.copy().astype(float)
        
        # 仅对数值特征进行标准化
        if self.numerical_features_ and self.scaler_type != 'none':
            X_numerical = X_df[self.numerical_features_]
            
            # 处理缺失值
            if self.handle_missing:
                X_numerical = X_numerical.fillna(X_numerical.median())
            
            # 应用标准化
            X_numerical_scaled = self.scaler.transform(X_numerical)
            
            # 将标准化后的数值特征放回原位置
            numerical_indices = [self.feature_names_.index(col) for col in self.numerical_features_]
            X_scaled[:, numerical_indices] = X_numerical_scaled
        
        logger.debug(f"标准化变换完成 - 输入形状: {X.shape}, 输出形状: {X_scaled.shape}")
        return X_scaled
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        拟合并应用标准化变换
        
        Args:
            X: 输入特征矩阵
            y: 目标变量（未使用）
            
        Returns:
            X_scaled: 标准化后的特征矩阵
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        逆标准化变换
        
        Args:
            X: 标准化后的特征矩阵
            
        Returns:
            X_original: 逆变换后的特征矩阵
        """
        if not self.is_fitted_:
            raise ValueError("标准化器尚未拟合，请先调用 fit() 方法")
        
        # 转换为DataFrame便于处理
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names_)
        else:
            X_df = X.copy()
        
        # 创建输出数组
        X_original = X_df.values.copy()
        
        # 仅对数值特征进行逆标准化
        if self.numerical_features_ and self.scaler_type != 'none':
            numerical_indices = [self.feature_names_.index(col) for col in self.numerical_features_]
            X_numerical_scaled = X_original[:, numerical_indices]
            
            # 应用逆标准化
            X_numerical_original = self.scaler.inverse_transform(X_numerical_scaled)
            X_original[:, numerical_indices] = X_numerical_original
        
        return X_original
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        获取特征信息
        
        Returns:
            feature_info: 特征信息字典
        """
        if not self.is_fitted_:
            raise ValueError("标准化器尚未拟合，请先调用 fit() 方法")
        
        info = {
            'scaler_type': self.scaler_type,
            'total_features': len(self.feature_names_),
            'numerical_features': len(self.numerical_features_),
            'categorical_features': len(self.categorical_features),
            'numerical_feature_names': self.numerical_features_,
            'categorical_feature_names': self.categorical_features
        }
        
        # 添加标准化参数信息
        if self.numerical_features_ and self.scaler_type != 'none':
            if self.scaler_type == 'standard':
                info['means'] = self.scaler.mean_.tolist()
                info['stds'] = self.scaler.scale_.tolist()
            elif self.scaler_type == 'robust':
                info['medians'] = self.scaler.center_.tolist()
                info['iqrs'] = self.scaler.scale_.tolist()
        elif self.scaler_type == 'none':
            info['note'] = '未进行标准化，保持原始数据'
        
        return info


def create_scaler(scaler_type: str = 'standard', 
                  categorical_features: Optional[list] = None,
                  **kwargs) -> MedicalDataScaler:
    """
    标准化器工厂函数
    
    Args:
        scaler_type: 标准化类型 ('standard' | 'robust')
        categorical_features: 类别特征列表
        **kwargs: 其他参数
        
    Returns:
        scaler: 标准化器实例
    """
    return MedicalDataScaler(
        scaler_type=scaler_type,
        categorical_features=categorical_features,
        **kwargs
    )


def get_available_scalers() -> Dict[str, str]:
    """
    获取可用的标准化器类型
    
    Returns:
        scalers: 可用标准化器字典
    """
    return {
        'standard': 'StandardScaler - 基于均值和标准差的标准化',
        'robust': 'RobustScaler - 基于中位数和IQR的鲁棒标准化',
        'none': 'NoScaler - 不进行标准化（保持原始数据）'
    }


def compare_scalers(X: Union[np.ndarray, pd.DataFrame], 
                   categorical_features: Optional[list] = None) -> Dict[str, Dict[str, float]]:
    """
    比较不同标准化器的效果
    
    Args:
        X: 输入特征矩阵
        categorical_features: 类别特征列表
        
    Returns:
        comparison: 比较结果字典
    """
    results = {}
    
    for scaler_type in ['standard', 'robust', 'none']:
        scaler = create_scaler(scaler_type, categorical_features)
        X_scaled = scaler.fit_transform(X)
        
        # 计算统计信息（仅针对数值特征）
        if isinstance(X, pd.DataFrame):
            numerical_cols = [col for col in X.columns if col not in (categorical_features or [])]
            numerical_indices = [X.columns.get_loc(col) for col in numerical_cols]
        else:
            numerical_indices = list(range(X.shape[1]))
            if categorical_features:
                # 假设类别特征索引已知
                numerical_indices = [i for i in range(X.shape[1]) if i not in categorical_features]
        
        if numerical_indices:
            X_numerical_scaled = X_scaled[:, numerical_indices]
            results[scaler_type] = {
                'mean': float(np.mean(X_numerical_scaled)),
                'std': float(np.std(X_numerical_scaled)),
                'median': float(np.median(X_numerical_scaled)),
                'iqr': float(np.percentile(X_numerical_scaled, 75) - np.percentile(X_numerical_scaled, 25)),
                'min': float(np.min(X_numerical_scaled)),
                'max': float(np.max(X_numerical_scaled))
            }
    
    return results


# 便捷函数
def standard_scale(X: Union[np.ndarray, pd.DataFrame], 
                  categorical_features: Optional[list] = None) -> Tuple[np.ndarray, MedicalDataScaler]:
    """
    标准标准化便捷函数
    
    Args:
        X: 输入特征矩阵
        categorical_features: 类别特征列表
        
    Returns:
        X_scaled: 标准化后的特征矩阵
        scaler: 拟合后的标准化器
    """
    scaler = create_scaler('standard', categorical_features)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def robust_scale(X: Union[np.ndarray, pd.DataFrame], 
                categorical_features: Optional[list] = None) -> Tuple[np.ndarray, MedicalDataScaler]:
    """
    鲁棒标准化便捷函数
    
    Args:
        X: 输入特征矩阵
        categorical_features: 类别特征列表
        
    Returns:
        X_scaled: 标准化后的特征矩阵
        scaler: 拟合后的标准化器
    """
    scaler = create_scaler('robust', categorical_features)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def no_scale(X: Union[np.ndarray, pd.DataFrame], 
            categorical_features: Optional[list] = None) -> Tuple[np.ndarray, MedicalDataScaler]:
    """
    不进行标准化便捷函数
    
    Args:
        X: 输入特征矩阵
        categorical_features: 类别特征列表
        
    Returns:
        X_original: 原始特征矩阵
        scaler: 拟合后的标准化器
    """
    scaler = create_scaler('none', categorical_features)
    X_original = scaler.fit_transform(X)
    return X_original, scaler 