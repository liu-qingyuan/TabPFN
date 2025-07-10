"""
UDA Medical Imbalance Project - 类别不平衡处理模块

本模块提供SMOTE、BorderlineSMOTE、ADASYN等不平衡数据处理方法，
集成了统一的异常处理和数据验证机制。

作者: UDA Medical Team
日期: 2024-01-30
版本: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, Dict, Any, List
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE, SMOTENC, BorderlineSMOTE, ADASYN, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks
import logging

# 导入统一的异常处理和验证系统
from utils.exceptions import PreprocessingError, DataValidationError, handle_exceptions
from utils.validators import DataValidator
from utils.performance import TimerContext, profile_function
from utils.helpers import ensure_array, ensure_dataframe

logger = logging.getLogger(__name__)


class ImbalanceHandler(BaseEstimator, TransformerMixin):
    """
    类别不平衡处理器
    
    支持多种不平衡数据处理方法，包括过采样、欠采样和混合采样策略
    """
    
    SUPPORTED_METHODS = {
        'smote': SMOTE,
        'smotenc': SMOTENC,
        'borderline_smote': BorderlineSMOTE,
        'kmeans_smote': KMeansSMOTE,
        'svm_smote': SVMSMOTE,
        'adasyn': ADASYN,
        'smote_tomek': SMOTETomek,
        'smote_enn': SMOTEENN,
        'random_under': RandomUnderSampler,
        'edited_nn': EditedNearestNeighbours,
        'none': None  # 不进行不平衡处理
    }
    
    def __init__(
        self,
        method: str = 'smote',
        random_state: int = 42,
        k_neighbors: int = 5,
        categorical_features: Optional[Union[List[int], List[str], np.ndarray]] = None,
        feature_type: str = 'best7',
        smote_config: Optional[Dict[str, Any]] = None,
        tomek_config: Optional[Dict[str, Any]] = None,
        enn_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        初始化不平衡处理器
        
        Args:
            method: 不平衡处理方法名称
            random_state: 随机种子
            k_neighbors: SMOTE相关方法的近邻数量
            categorical_features: 类别特征索引（用于SMOTENC）
            smote_config: SMOTE配置（用于组合方法）
            tomek_config: TomekLinks配置（用于SMOTETomek）
            enn_config: EditedNearestNeighbours配置（用于SMOTEENN）
            **kwargs: 其他参数传递给具体的不平衡处理方法
        """
        self.method = method
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.categorical_features = categorical_features
        self.feature_type = feature_type
        self.smote_config = smote_config or {}
        self.tomek_config = tomek_config or {}
        self.enn_config = enn_config or {}
        self.kwargs = kwargs
        self.sampler = None
        self.original_distribution = None
        self.resampled_distribution = None
        
        # 自动设置类别特征（如果未指定）
        if self.categorical_features is None:
            self._set_categorical_features()
        
        self._validate_method()
        self._initialize_sampler()
    
    def _set_categorical_features(self):
        """根据feature_type自动设置类别特征"""
        try:
            from config.settings import get_categorical_indices
            self.categorical_features = get_categorical_indices(self.feature_type)
            logger.info(f"自动设置类别特征索引: {self.categorical_features} (feature_type: {self.feature_type})")
        except ImportError:
            logger.warning("无法导入get_categorical_indices函数，将使用None作为类别特征")
            self.categorical_features = None
        except Exception as e:
            logger.warning(f"设置类别特征失败: {e}，将使用None作为类别特征")
            self.categorical_features = None
    
    def _validate_method(self):
        """验证方法名称是否支持"""
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"不支持的方法: {self.method}. "
                f"支持的方法: {list(self.SUPPORTED_METHODS.keys())}"
            )
    
    def _initialize_sampler(self):
        """初始化采样器"""
        if self.method == 'none':
            self.sampler = None
            logger.info("不进行不平衡处理")
            return
            
        sampler_class = self.SUPPORTED_METHODS[self.method]
        
        # 为不同方法设置特定参数
        params = {
            'random_state': self.random_state,
            **self.kwargs
        }
        
        # SMOTE相关方法需要k_neighbors参数
        if self.method in ['smote', 'borderline_smote', 'kmeans_smote', 'svm_smote']:
            params['k_neighbors'] = self.k_neighbors
        
        # SMOTENC需要categorical_features参数
        if self.method == 'smotenc':
            if self.categorical_features is None or len(self.categorical_features) == 0:
                logger.warning("SMOTENC方法未检测到类别特征，将使用普通SMOTE")
                sampler_class = SMOTE
            else:
                params['categorical_features'] = self.categorical_features
            params['k_neighbors'] = self.k_neighbors
        
        # ADASYN使用n_neighbors参数
        if self.method == 'adasyn':
            params['n_neighbors'] = self.k_neighbors
        
        # 组合方法的参数设置
        if self.method == 'smote_tomek':
            # 配置内部SMOTE - 使用SMOTE作为基础模型
            smote_params = {
                'k_neighbors': self.k_neighbors,
                'random_state': self.random_state,
                **self.smote_config
            }
            
            # SMOTETomek要求smote参数必须是SMOTE的实例
            smote_sampler = SMOTE(**smote_params)
            logger.info("SMOTETomek使用SMOTE作为基础SMOTE模型")
            
            # 配置TomekLinks
            tomek_params = {
                'sampling_strategy': 'all',
                **self.tomek_config
            }
            
            params = {
                'random_state': self.random_state,
                'smote': smote_sampler,
                'tomek': TomekLinks(**tomek_params)
            }
            
        elif self.method == 'smote_enn':
            # 配置内部SMOTE - 使用SMOTE作为基础模型
            smote_params = {
                'k_neighbors': self.k_neighbors,
                'random_state': self.random_state,
                **self.smote_config
            }
            
            # SMOTEENN要求smote参数必须是SMOTE的实例
            smote_sampler = SMOTE(**smote_params)
            logger.info("SMOTEENN使用SMOTE作为基础SMOTE模型")
            
            # 配置EditedNearestNeighbours
            enn_params = {
                'sampling_strategy': 'all',
                **self.enn_config
            }
            
            params = {
                'random_state': self.random_state,
                'smote': smote_sampler,
                'enn': EditedNearestNeighbours(**enn_params)
            }
        
        try:
            self.sampler = sampler_class(**params)
            logger.info(f"已初始化{self.method}采样器，参数: {params}")
        except Exception as e:
            logger.error(f"初始化采样器失败: {e}")
            raise
    
    @handle_exceptions(reraise=True)
    @profile_function(track_memory=True, track_cpu=False)
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        拟合采样器
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            self
            
        Raises:
            PreprocessingError: 数据预处理失败时
            DataValidationError: 数据验证失败时
        """
        with TimerContext(f"ImbalanceHandler.fit({self.method})", auto_log=True):
            try:
                # 验证输入数据
                DataValidator.validate_features(X, min_features=1)
                DataValidator.validate_labels(y, min_samples_per_class=1)
                DataValidator.validate_data_consistency(X, y)
                
                # 转换数据格式
                X_array = ensure_array(X)
                y_array = ensure_array(y)
                
                # 记录原始分布
                self.original_distribution = self._get_class_distribution(y_array)
                logger.info(f"原始类别分布: {self.original_distribution}")
                
                # 检查是否需要处理不平衡
                if not self._needs_resampling(y_array):
                    logger.warning("数据已经平衡，跳过重采样")
                    self.sampler = None
                    return self
                
                # 拟合采样器
                if self.sampler is not None:
                    try:
                        self.sampler.fit(X_array, y_array)
                        logger.info(f"{self.method}采样器拟合完成")
                    except Exception as e:
                        raise PreprocessingError(
                            f"采样器拟合失败: {str(e)}",
                            preprocessing_step="imbalance_fit",
                            input_shape=X_array.shape,
                            cause=e
                        )
                
                return self
                
            except (PreprocessingError, DataValidationError):
                raise
            except Exception as e:
                raise PreprocessingError(
                    f"不平衡处理拟合过程中发生未知错误: {str(e)}",
                    preprocessing_step="imbalance_fit",
                    input_shape=X.shape if hasattr(X, 'shape') else None,
                    cause=e
                )
    
    @handle_exceptions(reraise=True)
    @profile_function(track_memory=True, track_cpu=False)
    def transform(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用不平衡处理变换
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            重采样后的(X, y)
            
        Raises:
            PreprocessingError: 数据预处理失败时
            DataValidationError: 数据验证失败时
        """
        with TimerContext(f"ImbalanceHandler.transform({self.method})", auto_log=True):
            try:
                # 验证输入数据
                DataValidator.validate_features(X, min_features=1)
                DataValidator.validate_labels(y, min_samples_per_class=1)
                DataValidator.validate_data_consistency(X, y)
                
                # 如果采样器为None，直接返回原数据
                if self.sampler is None:
                    logger.info("采样器为空，返回原始数据")
                    return ensure_array(X), ensure_array(y)
                
                # 转换数据格式
                X_array = ensure_array(X)
                y_array = ensure_array(y)
                
                try:
                    # 应用重采样
                    X_resampled, y_resampled = self.sampler.fit_resample(X_array, y_array)
                    
                    # 记录重采样后的分布
                    self.resampled_distribution = self._get_class_distribution(y_resampled)
                    logger.info(f"重采样后类别分布: {self.resampled_distribution}")
                    
                    # 验证输出数据
                    if X_resampled.shape[1] != X_array.shape[1]:
                        raise PreprocessingError(
                            f"重采样后特征数量不一致: 原始={X_array.shape[1]}, 重采样后={X_resampled.shape[1]}",
                            preprocessing_step="imbalance_transform",
                            input_shape=X_array.shape,
                            output_shape=X_resampled.shape
                        )
                    
                    logger.info(f"重采样完成: {X_array.shape} -> {X_resampled.shape}")
                    return X_resampled, y_resampled
                    
                except Exception as e:
                    raise PreprocessingError(
                        f"重采样失败: {str(e)}",
                        preprocessing_step="imbalance_resample",
                        input_shape=X_array.shape,
                        cause=e
                    )
                    
            except (PreprocessingError, DataValidationError):
                raise
            except Exception as e:
                raise PreprocessingError(
                    f"不平衡处理变换过程中发生未知错误: {str(e)}",
                    preprocessing_step="imbalance_transform",
                    input_shape=X.shape if hasattr(X, 'shape') else None,
                    cause=e
                )
    
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        拟合并变换数据
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            重采样后的(X, y)
        """
        return self.fit(X, y).transform(X, y)
    
    def get_sampling_info(self) -> Dict[str, Any]:
        """
        获取采样信息
        
        Returns:
            包含采样前后分布信息的字典
        """
        return {
            'method': self.method,
            'original_distribution': self.original_distribution,
            'resampled_distribution': self.resampled_distribution,
            'sampler_params': self.sampler.get_params() if self.sampler else None
        }
    
    def _convert_to_array(self, data: Union[np.ndarray, pd.DataFrame, pd.Series], is_target: bool = False) -> np.ndarray:
        """转换数据为numpy数组"""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)
    
    def _get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """获取类别分布"""
        unique, counts = np.unique(y, return_counts=True)
        return {str(cls): int(count) for cls, count in zip(unique, counts)}
    
    def _needs_resampling(self, y: np.ndarray, imbalance_threshold: float = 0.5) -> bool:
        """
        检查是否需要重采样
        
        Args:
            y: 标签数据
            imbalance_threshold: 不平衡阈值，当最小类别占比小于此值时认为需要重采样
            
        Returns:
            是否需要重采样
        """
        if self.method == 'none':
            return False
            
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            return False
        
        # 计算最小类别占比
        min_ratio = np.min(counts) / np.sum(counts)
        # 如果最小类别占比小于阈值，则需要重采样
        return min_ratio < imbalance_threshold


class ImbalanceHandlerFactory:
    """不平衡处理器工厂类"""
    
    @staticmethod
    def create_handler(
        method: str,
        random_state: int = 42,
        categorical_features: Optional[Union[List[int], List[str], np.ndarray]] = None,
        **kwargs
    ) -> ImbalanceHandler:
        """
        创建不平衡处理器
        
        Args:
            method: 处理方法名称
            random_state: 随机种子
            categorical_features: 类别特征索引
            **kwargs: 其他参数
            
        Returns:
            ImbalanceHandler实例
        """
        return ImbalanceHandler(
            method=method,
            random_state=random_state,
            categorical_features=categorical_features,
            **kwargs
        )
    
    @staticmethod
    def get_recommended_params(method: str, has_categorical: bool = False) -> Dict[str, Any]:
        """
        获取推荐参数配置
        
        Args:
            method: 方法名称
            has_categorical: 是否包含类别特征
            
        Returns:
            推荐参数字典
        """
        base_params = {
            'random_state': 42,
            'k_neighbors': 5
        }
        
        method_specific_params = {
            'smote': base_params,
            'smotenc': {**base_params, 'categorical_features': 'auto' if has_categorical else None},
            'borderline_smote': {**base_params, 'kind': 'borderline-1'},
            'kmeans_smote': {**base_params, 'cluster_balance_threshold': 0.1},
            'svm_smote': {**base_params, 'svm_estimator': None},
            'adasyn': {**base_params, 'n_neighbors': 5},
            'smote_tomek': {
                **base_params,
                'smote_config': {'k_neighbors': 5},
                'tomek_config': {'sampling_strategy': 'all'}
            },
            'smote_enn': {
                **base_params,
                'smote_config': {'k_neighbors': 5},
                'enn_config': {'sampling_strategy': 'all', 'n_neighbors': 3}
            },
            'random_under': {'random_state': 42, 'sampling_strategy': 'auto'},
            'edited_nn': {'n_neighbors': 3, 'sampling_strategy': 'all'},
            'none': {}
        }
        
        return method_specific_params.get(method, base_params)


# 便捷函数
def create_imbalance_handler(method: str = 'smote', **kwargs) -> ImbalanceHandler:
    """创建不平衡处理器的便捷函数"""
    return ImbalanceHandlerFactory.create_handler(method=method, **kwargs)


def get_available_methods() -> List[str]:
    """获取所有可用的不平衡处理方法"""
    return list(ImbalanceHandler.SUPPORTED_METHODS.keys())


def test_imbalance_handler():
    """测试不平衡处理器功能"""
    from sklearn.datasets import make_classification
    
    # 创建不平衡数据集
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, weights=[0.9, 0.1],
        random_state=42
    )
    
    print(f"原始数据形状: {X.shape}")
    print(f"原始类别分布: {np.bincount(y)}")
    
    # 测试不同方法
    methods = ['smote', 'borderline_smote', 'adasyn', 'smote_tomek', 'smote_enn']
    
    for method in methods:
        try:
            handler = create_imbalance_handler(method=method)
            X_resampled, y_resampled = handler.fit_transform(X, y)
            print(f"{method}: {X_resampled.shape}, 类别分布: {np.bincount(y_resampled)}")
        except Exception as e:
            print(f"{method} 失败: {e}")


if __name__ == "__main__":
    test_imbalance_handler() 