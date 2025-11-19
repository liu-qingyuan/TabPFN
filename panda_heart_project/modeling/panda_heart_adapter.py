"""
PANDA-Heart模型适配器
包含TabPFN集成和基线模型实现
"""

import numpy as np
import pandas as pd
import torch
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import logging
import time
import joblib
from pathlib import Path

# 导入TabPFN
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    warnings.warn("TabPFN不可用，将跳过TabPFN相关模型")
    TABPFN_AVAILABLE = False

# 导入scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 导入XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    warnings.warn("XGBoost不可用，将跳过XGBoost模型")
    XGBOOST_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class BaseHeartDiseaseModel(ABC):
    """心脏病分类模型基类"""

    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        """
        初始化基础模型

        Args:
            model_name: 模型名称
            config: 模型配置
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        self.training_time = 0.0
        self.feature_names = None
        self.classes_ = None

    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'BaseHeartDiseaseModel':
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别"""
        pass

    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率"""
        pass

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
                   X_test: Union[np.ndarray, pd.DataFrame] = None) -> Dict[str, Any]:
        """
        训练并预测

        Args:
            X: 训练特征
            y: 训练标签
            X_test: 测试特征（可选）

        Returns:
            包含预测结果的字典
        """
        start_time = time.time()
        self.fit(X, y)
        self.training_time = time.time() - start_time

        results = {
            'model_name': self.model_name,
            'training_time': self.training_time,
            'training_samples': len(X),
            'n_features': X.shape[1] if hasattr(X, 'shape') else len(X[0]),
            'feature_names': getattr(X, 'columns', self.feature_names)
        }

        # 训练集预测
        y_train_pred = self.predict(X)
        y_train_proba = self.predict_proba(X)
        results.update({
            'y_train_pred': y_train_pred,
            'y_train_proba': y_train_proba
        })

        # 测试集预测（如果提供）
        if X_test is not None:
            y_test_pred = self.predict(X_test)
            y_test_proba = self.predict_proba(X_test)
            results.update({
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba,
                'test_samples': len(X_test)
            })

        return results

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性（如果支持）"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            return None

    def save_model(self, path: str) -> None:
        """保存模型"""
        model_data = {
            'model_name': self.model_name,
            'config': self.config,
            'model': self.model,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'feature_names': self.feature_names,
            'classes_': self.classes_
        }
        joblib.dump(model_data, path)
        logger.info(f"模型已保存: {path}")

    def load_model(self, path: str) -> None:
        """加载模型"""
        model_data = joblib.load(path)
        self.model_name = model_data['model_name']
        self.config = model_data['config']
        self.model = model_data['model']
        self.is_fitted = model_data['is_fitted']
        self.training_time = model_data['training_time']
        self.feature_names = model_data['feature_names']
        self.classes_ = model_data['classes_']
        logger.info(f"模型已加载: {path}")


class TabPFNHeartDiseaseModel(BaseHeartDiseaseModel):
    """TabPFN心脏病分类模型"""

    def __init__(self, config: Dict[str, Any] = None):
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN不可用，请先安装TabPFN")

        default_config = {
            'n_estimators': 32,
            'device': 'auto',
            'ignore_pretraining_limits': True,
            'softmax_temperature': 0.9,
            'balance_probabilities': True
        }
        config = {**default_config, **(config or {})}
        super().__init__('TabPFN_Only', config)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'TabPFNHeartDiseaseModel':
        """训练TabPFN模型"""
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN不可用")

        # 转换为numpy数组
        X_array = self._convert_to_numpy(X)
        y_array = self._convert_to_numpy(y)

        # 验证数据
        self._validate_data(X_array, y_array)

        # 保存特征名称
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()

        # 创建TabPFN模型
        self.model = TabPFNClassifier(
            n_estimators=self.config['n_estimators'],
            device=self.config['device'],
            ignore_pretraining_limits=self.config['ignore_pretraining_limits'],
            softmax_temperature=self.config['softmax_temperature'],
            balance_probabilities=self.config['balance_probabilities']
        )

        # 训练模型
        logger.info(f"训练TabPFN模型: {X_array.shape[0]} 样本, {X_array.shape[1]} 特征")
        self.model.fit(X_array, y_array)

        self.is_fitted = True
        self.classes_ = np.unique(y_array)

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_array = self._convert_to_numpy(X)
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_array = self._convert_to_numpy(X)
        return self.model.predict_proba(X_array)

    def _convert_to_numpy(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """转换为numpy数组"""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    def _validate_data(self, X: np.ndarray, y: np.ndarray):
        """验证数据有效性"""
        if len(X) != len(y):
            raise ValueError("X和y的长度必须相同")

        if X.shape[0] > 1024:
            logger.warning(f"TabPFN建议样本数不超过1024，当前有{X.shape[0]}个样本")

        if X.shape[1] > 100:
            logger.warning(f"TabPFN建议特征数不超过100，当前有{X.shape[1]}个特征")


class PANDATabPFNModel(BaseHeartDiseaseModel):
    """PANDA_TabPFN模型（TabPFN + 域适应）"""

    def __init__(self, config: Dict[str, Any] = None, domain_adapter=None):
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN不可用，请先安装TabPFN")

        default_config = {
            'n_estimators': 32,
            'device': 'auto',
            'ignore_pretraining_limits': True,
            'softmax_temperature': 0.9,
            'balance_probabilities': True
        }
        config = {**default_config, **(config or {})}
        super().__init__('PANDA_TabPFN', config)
        self.domain_adapter = domain_adapter

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
            X_target: Union[np.ndarray, pd.DataFrame] = None) -> 'PANDATabPFNModel':
        """
        训练PANDA_TabPFN模型

        Args:
            X: 源域特征
            y: 源域标签
            X_target: 目标域特征（用于域适应）
        """
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN不可用")

        # 转换为numpy数组
        X_array = self._convert_to_numpy(X)
        y_array = self._convert_to_numpy(y)

        # 保存特征名称
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()

        # 域适应
        if self.domain_adapter and X_target is not None:
            logger.info("执行域适应...")
            X_target_array = self._convert_to_numpy(X_target)

            # 训练域适应器
            self.domain_adapter.fit(X_array, y_array, X_target_array)

            # 变换源域数据
            X_array_adapted, _ = self.domain_adapter.transform(X_array, X_target_array)
            X_train = X_array_adapted
            logger.info(f"域适应完成，变换后维度: {X_train.shape}")
        else:
            X_train = X_array

        # 创建TabPFN模型
        self.model = TabPFNClassifier(
            n_estimators=self.config['n_estimators'],
            device=self.config['device'],
            ignore_pretraining_limits=self.config['ignore_pretraining_limits'],
            softmax_temperature=self.config['softmax_temperature'],
            balance_probabilities=self.config['balance_probabilities']
        )

        # 训练模型
        logger.info(f"训练PANDA_TabPFN模型: {X_train.shape[0]} 样本, {X_train.shape[1]} 特征")
        self.model.fit(X_train, y_array)

        self.is_fitted = True
        self.classes_ = np.unique(y_array)

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame], apply_adaptation: bool = True) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_array = self._convert_to_numpy(X)

        # 应用域适应
        if apply_adaptation and self.domain_adapter:
            _, X_transformed = self.domain_adapter.transform(X, X)
            X_pred = X_transformed
        else:
            X_pred = X_array

        return self.model.predict(X_pred)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame], apply_adaptation: bool = True) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        X_array = self._convert_to_numpy(X)

        # 应用域适应
        if apply_adaptation and self.domain_adapter:
            _, X_transformed = self.domain_adapter.transform(X, X)
            X_pred = X_transformed
        else:
            X_pred = X_array

        return self.model.predict_proba(X_pred)

    def _convert_to_numpy(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """转换为numpy数组"""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)


class LogisticRegressionHeartDiseaseModel(BaseHeartDiseaseModel):
    """逻辑回归心脏病分类模型（LASSO）"""

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'penalty': 'l1',
            'C': 1.0,
            'solver': 'saga',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42
        }
        config = {**default_config, **(config or {})}
        super().__init__('LASSO_LR', config)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'LogisticRegressionHeartDiseaseModel':
        """训练逻辑回归模型"""
        # 创建特征标准化和逻辑回归的pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**self.config))
        ])

        # 转换数据
        X_array = self._prepare_data(X)
        y_array = self._prepare_data(y)

        # 保存特征名称
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()

        # 训练模型
        logger.info(f"训练逻辑回归模型: {X_array.shape[0]} 样本, {X_array.shape[1]} 特征")
        self.model.fit(X_array, y_array)

        self.is_fitted = True
        self.classes_ = self.model.classes_

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        X_array = self._prepare_data(X)
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        X_array = self._prepare_data(X)
        return self.model.predict_proba(X_array)

    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """准备数据"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return np.array(data)


class XGBoostHeartDiseaseModel(BaseHeartDiseaseModel):
    """XGBoost心脏病分类模型"""

    def __init__(self, config: Dict[str, Any] = None):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost不可用，请先安装XGBoost")

        default_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1
        }
        config = {**default_config, **(config or {})}
        super().__init__('XGBoost', config)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'XGBoostHeartDiseaseModel':
        """训练XGBoost模型"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost不可用")

        # 创建XGBoost模型
        self.model = xgb.XGBClassifier(**self.config)

        # 转换数据
        X_array = self._prepare_data(X)
        y_array = self._prepare_data(y)

        # 保存特征名称
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
            self.model.fit(X, y)  # XGBoost可以直接处理DataFrame
        else:
            self.model.fit(X_array, y_array)

        self.is_fitted = True
        self.classes_ = self.model.classes_

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.model.predict_proba(X)

    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """准备数据"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return np.array(data)


class RandomForestHeartDiseaseModel(BaseHeartDiseaseModel):
    """随机森林心脏病分类模型"""

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'n_estimators': 200,
            'max_depth': None,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        config = {**default_config, **(config or {})}
        super().__init__('Random_Forest', config)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'RandomForestHeartDiseaseModel':
        """训练随机森林模型"""
        self.model = RandomForestClassifier(**self.config)

        # 转换数据
        X_array = self._prepare_data(X)
        y_array = self._prepare_data(y)

        # 保存特征名称
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()

        # 训练模型
        logger.info(f"训练随机森林模型: {X_array.shape[0]} 样本, {X_array.shape[1]} 特征")
        self.model.fit(X_array, y_array)

        self.is_fitted = True
        self.classes_ = self.model.classes_

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        X_array = self._prepare_data(X)
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        X_array = self._prepare_data(X)
        return self.model.predict_proba(X_array)

    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """准备数据"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return np.array(data)


class SVMHeartDiseaseModel(BaseHeartDiseaseModel):
    """支持向量机心脏病分类模型"""

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42
        }
        config = {**default_config, **(config or {})}
        super().__init__('SVM', config)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'SVMHeartDiseaseModel':
        """训练SVM模型"""
        # 创建特征标准化和SVM的pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(**self.config))
        ])

        # 转换数据
        X_array = self._prepare_data(X)
        y_array = self._prepare_data(y)

        # 保存特征名称
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()

        # 训练模型
        logger.info(f"训练SVM模型: {X_array.shape[0]} 样本, {X_array.shape[1]} 特征")
        self.model.fit(X_array, y_array)

        self.is_fitted = True
        self.classes_ = self.model.classes_

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        X_array = self._prepare_data(X)
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        X_array = self._prepare_data(X)
        return self.model.predict_proba(X_array)

    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """准备数据"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return np.array(data)


class KNNHeartDiseaseModel(BaseHeartDiseaseModel):
    """K近邻心脏病分类模型"""

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'euclidean',
            'algorithm': 'auto'
        }
        config = {**default_config, **(config or {})}
        super().__init__('KNN', config)

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'KNNHeartDiseaseModel':
        """训练KNN模型"""
        # 创建特征标准化和KNN的pipeline
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier(**self.config))
        ])

        # 转换数据
        X_array = self._prepare_data(X)
        y_array = self._prepare_data(y)

        # 保存特征名称
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()

        # 训练模型
        logger.info(f"训练KNN模型: {X_array.shape[0]} 样本, {X_array.shape[1]} 特征")
        self.model.fit(X_array, y_array)

        self.is_fitted = True
        self.classes_ = self.model.classes_

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测类别"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        X_array = self._prepare_data(X)
        return self.model.predict(X_array)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        X_array = self._prepare_data(X)
        return self.model.predict_proba(X_array)

    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """准备数据"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        return np.array(data)


class HeartDiseaseModelFactory:
    """心脏病模型工厂"""

    @staticmethod
    def create_model(model_name: str, config: Dict[str, Any] = None,
                    domain_adapter=None) -> BaseHeartDiseaseModel:
        """
        创建模型实例

        Args:
            model_name: 模型名称
            config: 模型配置
            domain_adapter: 域适应器（仅PANDA_TabPFN需要）

        Returns:
            模型实例
        """
        model_name = model_name.upper()

        if model_name == 'TABPFN_ONLY':
            return TabPFNHeartDiseaseModel(config)
        elif model_name == 'PANDA_TABPFN':
            return PANDATabPFNModel(config, domain_adapter)
        elif model_name == 'LASSO_LR':
            return LogisticRegressionHeartDiseaseModel(config)
        elif model_name == 'XGBOOST':
            return XGBoostHeartDiseaseModel(config)
        elif model_name == 'RANDOM_FOREST':
            return RandomForestHeartDiseaseModel(config)
        elif model_name == 'SVM':
            return SVMHeartDiseaseModel(config)
        elif model_name == 'KNN':
            return KNNHeartDiseaseModel(config)
        else:
            raise ValueError(f"未知的模型名称: {model_name}")

    @staticmethod
    def get_available_models() -> List[str]:
        """获取可用模型列表"""
        models = ['LASSO_LR', 'RANDOM_FOREST', 'SVM', 'KNN']

        if TABPFN_AVAILABLE:
            models.extend(['TABPFN_ONLY', 'PANDA_TABPFN'])

        if XGBOOST_AVAILABLE:
            models.append('XGBOOST')

        return models

    @staticmethod
    def get_model_configs() -> Dict[str, Dict[str, Any]]:
        """获取各模型的默认配置"""
        return {
            'TABPFN_ONLY': {
                'n_estimators': 32,
                'device': 'auto',
                'ignore_pretraining_limits': True,
                'softmax_temperature': 0.9,
                'balance_probabilities': True
            },
            'PANDA_TABPFN': {
                'n_estimators': 32,
                'device': 'auto',
                'ignore_pretraining_limits': True,
                'softmax_temperature': 0.9,
                'balance_probabilities': True
            },
            'LASSO_LR': {
                'penalty': 'l1',
                'C': 1.0,
                'solver': 'saga',
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42
            },
            'XGBOOST': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            },
            'RANDOM_FOREST': {
                'n_estimators': 200,
                'max_depth': None,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            },
            'SVM': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'class_weight': 'balanced',
                'random_state': 42
            },
            'KNN': {
                'n_neighbors': 5,
                'weights': 'distance',
                'metric': 'euclidean',
                'algorithm': 'auto'
            }
        }


if __name__ == "__main__":
    # 测试模型创建
    print("=== 心脏病模型测试 ===")

    try:
        # 获取可用模型
        available_models = HeartDiseaseModelFactory.get_available_models()
        print(f"可用模型: {available_models}")

        # 创建测试数据
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)

        # 测试每个模型
        for model_name in available_models:
            try:
                print(f"\n测试模型: {model_name}")
                model = HeartDiseaseModelFactory.create_model(model_name)

                # 训练和预测
                results = model.fit_predict(X_train, y_train, X_test)

                print(f"  训练时间: {results['training_time']:.3f}s")
                print(f"  训练样本: {results['training_samples']}")
                print(f"  特征数: {results['n_features']}")
                print(f"  预测完成")

            except Exception as e:
                print(f"  错误: {e}")

        print("\n✅ 模型测试完成")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()