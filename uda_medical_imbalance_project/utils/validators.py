"""
UDA Medical Imbalance Project - 数据验证模块

提供数据格式、形状、内容和配置的验证功能，确保数据质量和一致性。

作者: UDA Medical Team
日期: 2024-01-30
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional, Tuple
import logging
from .exceptions import DataValidationError, ModelConfigurationError, validate_and_raise

logger = logging.getLogger(__name__)

class DataValidator:
    """数据验证器 - 提供全面的数据验证功能"""
    
    @staticmethod
    def validate_features(
        X: Union[np.ndarray, pd.DataFrame],
        required_features: Optional[List[str]] = None,
        min_features: Optional[int] = None,
        max_features: Optional[int] = None
    ) -> bool:
        """
        验证特征数据
        
        Args:
            X: 特征数据
            required_features: 必需的特征名称列表
            min_features: 最小特征数量
            max_features: 最大特征数量
            
        Returns:
            验证是否通过
            
        Raises:
            DataValidationError: 验证失败时
        """
        try:
            # 检查数据是否为空
            if X is None:
                raise DataValidationError("特征数据不能为None")
            
            # 检查数据形状
            if isinstance(X, pd.DataFrame):
                n_samples, n_features = X.shape
                feature_names = list(X.columns)
            elif isinstance(X, np.ndarray):
                if len(X.shape) != 2:
                    raise DataValidationError(
                        f"特征数据必须是2维数组，当前维度: {len(X.shape)}",
                        data_info={'shape': X.shape}
                    )
                n_samples, n_features = X.shape
                feature_names = [f"feature_{i}" for i in range(n_features)]
            else:
                raise DataValidationError(
                    f"不支持的数据类型: {type(X)}",
                    expected_format="pd.DataFrame或np.ndarray",
                    actual_format=str(type(X))
                )
            
            # 检查样本数量
            if n_samples == 0:
                raise DataValidationError("特征数据不能为空")
            
            # 检查特征数量范围
            if min_features is not None and n_features < min_features:
                raise DataValidationError(
                    f"特征数量不足，需要至少{min_features}个特征，当前{n_features}个",
                    data_info={'n_features': n_features, 'min_required': min_features}
                )
            
            if max_features is not None and n_features > max_features:
                raise DataValidationError(
                    f"特征数量过多，最多允许{max_features}个特征，当前{n_features}个",
                    data_info={'n_features': n_features, 'max_allowed': max_features}
                )
            
            # 检查必需特征
            if required_features is not None:
                missing_features = [f for f in required_features if f not in feature_names]
                if missing_features:
                    raise DataValidationError(
                        f"缺少必需特征: {missing_features}",
                        data_info={
                            'missing_features': missing_features,
                            'available_features': feature_names
                        }
                    )
            
            # 检查数据类型和缺失值
            if isinstance(X, pd.DataFrame):
                # 检查数值列
                numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
                non_numeric_columns = [col for col in X.columns if col not in numeric_columns]
                
                if non_numeric_columns:
                    logger.warning(f"发现非数值特征: {non_numeric_columns}")
                
                # 检查缺失值
                missing_counts = X.isnull().sum()
                columns_with_missing = missing_counts[missing_counts > 0]
                if len(columns_with_missing) > 0:
                    logger.warning(f"发现缺失值: {dict(columns_with_missing)}")
            
            elif isinstance(X, np.ndarray):
                # 检查NaN值
                if np.isnan(X).any():
                    nan_count = np.isnan(X).sum()
                    logger.warning(f"发现{nan_count}个NaN值")
                
                # 检查无穷值
                if np.isinf(X).any():
                    inf_count = np.isinf(X).sum()
                    logger.warning(f"发现{inf_count}个无穷值")
            
            logger.info(f"特征验证通过: {n_samples}样本, {n_features}特征")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(
                f"特征验证过程中发生错误: {str(e)}",
                cause=e
            )
    
    @staticmethod
    def validate_labels(
        y: Union[np.ndarray, pd.Series],
        expected_classes: Optional[List] = None,
        min_samples_per_class: Optional[int] = None
    ) -> bool:
        """
        验证标签数据
        
        Args:
            y: 标签数据
            expected_classes: 期望的类别列表
            min_samples_per_class: 每个类别的最小样本数
            
        Returns:
            验证是否通过
            
        Raises:
            DataValidationError: 验证失败时
        """
        try:
            # 检查数据是否为空
            if y is None:
                raise DataValidationError("标签数据不能为None")
            
            # 转换为数组
            if isinstance(y, pd.Series):
                y_array = y.values
            elif isinstance(y, np.ndarray):
                y_array = y
            else:
                y_array = np.array(y)
            
            # 检查维度
            if len(y_array.shape) != 1:
                raise DataValidationError(
                    f"标签数据必须是1维数组，当前维度: {len(y_array.shape)}",
                    data_info={'shape': y_array.shape}
                )
            
            # 检查样本数量
            n_samples = len(y_array)
            if n_samples == 0:
                raise DataValidationError("标签数据不能为空")
            
            # 检查类别
            unique_classes, class_counts = np.unique(y_array, return_counts=True)
            n_classes = len(unique_classes)
            
            # 检查类别数量
            if n_classes < 2:
                raise DataValidationError(
                    f"至少需要2个类别，当前只有{n_classes}个类别",
                    data_info={'unique_classes': unique_classes.tolist()}
                )
            
            # 检查期望类别
            if expected_classes is not None:
                missing_classes = [c for c in expected_classes if c not in unique_classes]
                unexpected_classes = [c for c in unique_classes if c not in expected_classes]
                
                if missing_classes:
                    raise DataValidationError(
                        f"缺少期望的类别: {missing_classes}",
                        data_info={
                            'missing_classes': missing_classes,
                            'available_classes': unique_classes.tolist()
                        }
                    )
                
                if unexpected_classes:
                    logger.warning(f"发现意外的类别: {unexpected_classes}")
            
            # 检查每个类别的样本数量
            if min_samples_per_class is not None:
                insufficient_classes = []
                for cls, count in zip(unique_classes, class_counts):
                    if count < min_samples_per_class:
                        insufficient_classes.append((cls, count))
                
                if insufficient_classes:
                    raise DataValidationError(
                        f"部分类别样本数量不足（需要至少{min_samples_per_class}个）: {insufficient_classes}",
                        data_info={
                            'insufficient_classes': insufficient_classes,
                            'min_required': min_samples_per_class
                        }
                    )
            
            # 计算类别不平衡比例
            max_count = max(class_counts)
            min_count = min(class_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 10:
                logger.warning(f"数据严重不平衡，比例: {imbalance_ratio:.2f}:1")
            elif imbalance_ratio > 5:
                logger.warning(f"数据不平衡，比例: {imbalance_ratio:.2f}:1")
            
            logger.info(f"标签验证通过: {n_samples}样本, {n_classes}类别, 分布: {dict(zip(unique_classes, class_counts))}")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(
                f"标签验证过程中发生错误: {str(e)}",
                cause=e
            )
    
    @staticmethod
    def validate_data_consistency(
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> bool:
        """
        验证特征和标签数据的一致性
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            验证是否通过
            
        Raises:
            DataValidationError: 验证失败时
        """
        try:
            # 检查样本数量一致性
            if isinstance(X, pd.DataFrame):
                n_samples_X = len(X)
            elif isinstance(X, np.ndarray):
                n_samples_X = X.shape[0]
            else:
                raise DataValidationError(f"不支持的特征数据类型: {type(X)}")
            
            if isinstance(y, (pd.Series, np.ndarray)):
                n_samples_y = len(y)
            else:
                n_samples_y = len(np.array(y))
            
            if n_samples_X != n_samples_y:
                raise DataValidationError(
                    f"特征和标签样本数量不一致: X={n_samples_X}, y={n_samples_y}",
                    data_info={
                        'X_samples': n_samples_X,
                        'y_samples': n_samples_y
                    }
                )
            
            # 检查索引一致性（如果都是pandas对象）
            if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
                if not X.index.equals(y.index):
                    logger.warning("特征和标签的索引不完全一致，可能影响数据对应关系")
            
            logger.info(f"数据一致性验证通过: {n_samples_X}样本")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(
                f"数据一致性验证过程中发生错误: {str(e)}",
                cause=e
            )
    
    @staticmethod
    def validate_train_test_split(
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        y_test: Union[np.ndarray, pd.Series]
    ) -> bool:
        """
        验证训练测试集分割的有效性
        
        Args:
            X_train, X_test: 训练和测试特征
            y_train, y_test: 训练和测试标签
            
        Returns:
            验证是否通过
        """
        try:
            # 验证各个数据集
            DataValidator.validate_features(X_train)
            DataValidator.validate_features(X_test)
            DataValidator.validate_labels(y_train)
            DataValidator.validate_labels(y_test)
            
            # 验证一致性
            DataValidator.validate_data_consistency(X_train, y_train)
            DataValidator.validate_data_consistency(X_test, y_test)
            
            # 检查特征一致性
            if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
                if not X_train.columns.equals(X_test.columns):
                    raise DataValidationError(
                        "训练集和测试集的特征列不一致",
                        data_info={
                            'train_features': list(X_train.columns),
                            'test_features': list(X_test.columns)
                        }
                    )
            elif isinstance(X_train, np.ndarray) and isinstance(X_test, np.ndarray):
                if X_train.shape[1] != X_test.shape[1]:
                    raise DataValidationError(
                        f"训练集和测试集的特征数量不一致: train={X_train.shape[1]}, test={X_test.shape[1]}",
                        data_info={
                            'train_features': X_train.shape[1],
                            'test_features': X_test.shape[1]
                        }
                    )
            
            # 检查类别一致性
            train_classes = set(np.unique(y_train))
            test_classes = set(np.unique(y_test))
            
            if train_classes != test_classes:
                missing_in_test = train_classes - test_classes
                missing_in_train = test_classes - train_classes
                
                if missing_in_test:
                    logger.warning(f"测试集中缺少训练集的类别: {missing_in_test}")
                if missing_in_train:
                    logger.warning(f"训练集中缺少测试集的类别: {missing_in_train}")
            
            logger.info("训练测试集验证通过")
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(
                f"训练测试集验证过程中发生错误: {str(e)}",
                cause=e
            )


class ConfigValidator:
    """配置验证器 - 验证模型和实验配置"""
    
    @staticmethod
    def validate_model_config(
        model_type: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        验证模型配置
        
        Args:
            model_type: 模型类型
            config: 模型配置字典
            
        Returns:
            验证是否通过
        """
        try:
            if model_type == 'tabpfn':
                return ConfigValidator._validate_tabpfn_config(config)
            elif model_type in ['pkuph', 'mayo']:
                return ConfigValidator._validate_baseline_config(config)
            elif model_type in ['svm', 'rf', 'xgboost', 'gbdt']:
                return ConfigValidator._validate_ml_config(model_type, config)
            else:
                logger.warning(f"未知的模型类型: {model_type}")
                return True  # 对未知类型给予宽松验证
                
        except Exception as e:
            raise ModelConfigurationError(
                f"模型配置验证失败: {str(e)}",
                model_type=model_type,
                invalid_params=config,
                cause=e
            )
    
    @staticmethod
    def _validate_tabpfn_config(config: Dict[str, Any]) -> bool:
        """验证TabPFN配置"""
        # 检查必需参数
        if 'n_estimators' in config:
            n_est = config['n_estimators']
            if not isinstance(n_est, int) or n_est <= 0:
                raise ModelConfigurationError(
                    f"n_estimators必须是正整数，当前值: {n_est}",
                    model_type='tabpfn',
                    invalid_params={'n_estimators': n_est}
                )
        
        # 检查设备配置
        if 'device' in config:
            device = config['device']
            valid_devices = ['auto', 'cpu', 'cuda', 'mps']
            if device not in valid_devices and not device.startswith('cuda:'):
                logger.warning(f"可能无效的设备配置: {device}")
        
        return True
    
    @staticmethod
    def _validate_baseline_config(config: Dict[str, Any]) -> bool:
        """验证基线模型配置"""
        # 基线模型通常不需要额外配置
        return True
    
    @staticmethod
    def _validate_ml_config(model_type: str, config: Dict[str, Any]) -> bool:
        """验证机器学习模型配置"""
        # 基本的参数验证
        if 'random_state' in config:
            rs = config['random_state']
            if rs is not None and (not isinstance(rs, int) or rs < 0):
                raise ModelConfigurationError(
                    f"random_state必须是非负整数或None，当前值: {rs}",
                    model_type=model_type,
                    invalid_params={'random_state': rs}
                )
        
        return True


# 便捷验证函数
def validate_features(
    X: Union[np.ndarray, pd.DataFrame],
    required_features: Optional[List[str]] = None,
    min_features: Optional[int] = None,
    max_features: Optional[int] = None
) -> bool:
    """验证特征数据的便捷函数"""
    return DataValidator.validate_features(X, required_features, min_features, max_features)


def validate_labels(
    y: Union[np.ndarray, pd.Series],
    expected_classes: Optional[List] = None,
    min_samples_per_class: Optional[int] = None
) -> bool:
    """验证标签数据的便捷函数"""
    return DataValidator.validate_labels(y, expected_classes, min_samples_per_class)


def validate_data_shape(
    X: Union[np.ndarray, pd.DataFrame],
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_samples: Optional[int] = None,
    max_samples: Optional[int] = None
) -> bool:
    """
    验证数据形状
    
    Args:
        X: 数据
        expected_shape: 期望形状
        min_samples: 最小样本数
        max_samples: 最大样本数
        
    Returns:
        验证是否通过
    """
    try:
        if isinstance(X, pd.DataFrame):
            actual_shape = X.shape
        elif isinstance(X, np.ndarray):
            actual_shape = X.shape
        else:
            raise DataValidationError(f"不支持的数据类型: {type(X)}")
        
        # 检查期望形状
        if expected_shape is not None:
            if actual_shape != expected_shape:
                raise DataValidationError(
                    f"数据形状不匹配，期望: {expected_shape}，实际: {actual_shape}",
                    data_info={'expected_shape': expected_shape, 'actual_shape': actual_shape}
                )
        
        # 检查样本数范围
        n_samples = actual_shape[0]
        if min_samples is not None and n_samples < min_samples:
            raise DataValidationError(
                f"样本数不足，需要至少{min_samples}个，当前{n_samples}个",
                data_info={'n_samples': n_samples, 'min_required': min_samples}
            )
        
        if max_samples is not None and n_samples > max_samples:
            raise DataValidationError(
                f"样本数过多，最多允许{max_samples}个，当前{n_samples}个",
                data_info={'n_samples': n_samples, 'max_allowed': max_samples}
            )
        
        return True
        
    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(
            f"数据形状验证过程中发生错误: {str(e)}",
            cause=e
        )


def validate_model_params(
    model_type: str,
    params: Dict[str, Any]
) -> bool:
    """验证模型参数的便捷函数"""
    return ConfigValidator.validate_model_config(model_type, params)


# 导出的验证器和函数
__all__ = [
    'DataValidator',
    'ConfigValidator',
    'validate_features',
    'validate_labels',
    'validate_data_shape',
    'validate_model_params'
]