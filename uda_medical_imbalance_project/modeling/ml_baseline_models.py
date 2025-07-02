"""
UDA Medical Imbalance Project - 机器学习基线模型实现

包含现代机器学习模型，用于与TabPFN进行对比实验。
这些模型使用与TabPFN相同的特征集、标准化和不平衡处理方法。
"""

import numpy as np
import pandas as pd
from typing import Union, Any, List, Optional, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("警告: XGBoost未安装，XGBoost模型将不可用")


class MLBaselineModel(BaseEstimator, ClassifierMixin):
    """
    机器学习基线模型的包装类
    
    统一接口，支持不同的机器学习算法，使用默认配置
    """
    
    def __init__(self, 
                 model_type: str = 'svm',
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: bool = False):
        """
        初始化机器学习基线模型
        
        Args:
            model_type: 模型类型 ('svm', 'dt', 'rf', 'gbdt', 'xgboost')
            random_state: 随机种子
            n_jobs: 并行作业数
            verbose: 是否显示详细信息
        """
        super().__init__()
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # 初始化模型
        self.model = None
        self.is_fitted_ = False
        
        self._setup_model()
    
    def _setup_model(self):
        """设置模型（使用默认配置）"""
        if self.model_type == 'svm':
            self.model = SVC(
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'  # 处理不平衡数据
            )
            
        elif self.model_type == 'dt':
            self.model = DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            )
            
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=self.n_jobs
            )
            
        elif self.model_type == 'gbdt':
            self.model = GradientBoostingClassifier(
                random_state=self.random_state
            )
            
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost未安装，请安装后使用: pip install xgboost")
            
            self.model = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                eval_metric='logloss'
            )
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'MLBaselineModel':
        """
        拟合模型
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            self: 拟合后的模型
        """
        if self.verbose:
            print(f"训练{self.model_type.upper()}模型（使用默认配置）...")
        
        # 转换数据格式
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # 使用默认参数训练模型
        self.model.fit(X_array, y_array)
        self.is_fitted_ = True
        
        if self.verbose:
            print(f"  {self.model_type.upper()}模型训练完成")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征数据
            
        Returns:
            predictions: 预测标签
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        return self.model.predict(X_array)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测类别概率
        
        Args:
            X: 特征数据
            
        Returns:
            proba: 概率数组 [P(class_0), P(class_1)]
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        return self.model.predict_proba(X_array)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        获取特征重要性（如果模型支持）
        
        Returns:
            feature_importance: 特征重要性数组，如果不支持则返回None
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # 对于线性模型，使用系数的绝对值作为重要性
            return np.abs(self.model.coef_[0])
        else:
            return None
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        Returns:
            info: 模型信息字典
        """
        info = {
            'model_type': self.model_type,
            'is_fitted': self.is_fitted_
        }
        
        if self.is_fitted_ and self.model is not None:
            info['model_params'] = self.model.get_params()
            
            # 获取特征重要性
            feature_importance = self.get_feature_importance()
            if feature_importance is not None:
                info['has_feature_importance'] = True
                info['feature_importance'] = feature_importance.tolist()
            else:
                info['has_feature_importance'] = False
        
        return info


def get_ml_baseline_model(model_type: str, **kwargs) -> MLBaselineModel:
    """
    获取机器学习基线模型实例
    
    Args:
        model_type: 模型类型 ('svm', 'dt', 'rf', 'gbdt', 'xgboost')
        **kwargs: 其他参数传递给MLBaselineModel
        
    Returns:
        model: 模型实例
    """
    supported_models = ['svm', 'dt', 'rf', 'gbdt', 'xgboost']
    
    if model_type.lower() not in supported_models:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的模型: {supported_models}")
    
    return MLBaselineModel(model_type=model_type, **kwargs)


def evaluate_ml_baseline_models(X_train: Union[np.ndarray, pd.DataFrame], 
                               y_train: Union[np.ndarray, pd.Series],
                               X_test: Union[np.ndarray, pd.DataFrame], 
                               y_test: Union[np.ndarray, pd.Series],
                               models: Optional[List[str]] = None,
                               random_state: int = 42,
                               verbose: bool = False) -> Dict:
    """
    评估机器学习基线模型
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        models: 要评估的模型列表，默认为所有支持的模型
        random_state: 随机种子
        verbose: 是否显示详细信息
        
    Returns:
        results: 评估结果字典
    """
    if models is None:
        models = ['svm', 'dt', 'rf', 'gbdt']
        if XGBOOST_AVAILABLE:
            models.append('xgboost')
    
    results = {}
    
    for model_type in models:
        if verbose:
            print(f"\n评估{model_type.upper()}模型...")
        
        try:
            # 创建模型
            model = get_ml_baseline_model(
                model_type=model_type,
                random_state=random_state,
                verbose=verbose
            )
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]  # 获取正类概率
            
            # 计算指标
            from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_proba),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0)
            }
            
            # 获取模型信息
            model_info = model.get_model_info()
            
            results[model_type] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_proba,
                'metrics': metrics,
                'model_info': model_info
            }
            
            if verbose:
                print(f"  {model_type.upper()}结果:")
                print(f"    准确率: {metrics['accuracy']:.4f}")
                print(f"    AUC: {metrics['auc']:.4f}")
                print(f"    F1: {metrics['f1']:.4f}")
                
        except Exception as e:
            if verbose:
                print(f"  评估{model_type.upper()}模型时出错: {e}")
            results[model_type] = {'error': str(e)}
    
    return results


def get_model_comparison_summary(results: Dict) -> pd.DataFrame:
    """
    获取模型对比摘要
    
    Args:
        results: 评估结果字典
        
    Returns:
        summary_df: 对比摘要DataFrame
    """
    summary_data = []
    
    for model_type, result in results.items():
        if 'error' not in result and 'metrics' in result:
            metrics = result['metrics']
            model_info = result.get('model_info', {})
            
            summary_data.append({
                'Model': model_type.upper(),
                'Accuracy': metrics['accuracy'],
                'AUC': metrics['auc'],
                'F1': metrics['f1'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Uses_Default_Config': True,
                'Has_Feature_Importance': model_info.get('has_feature_importance', False)
            })
        else:
            summary_data.append({
                'Model': model_type.upper(),
                'Accuracy': 0,
                'AUC': 0,
                'F1': 0,
                'Precision': 0,
                'Recall': 0,
                'Uses_Default_Config': True,
                'Has_Feature_Importance': False,
                'Error': result.get('error', 'Unknown error')
            })
    
    return pd.DataFrame(summary_data) 