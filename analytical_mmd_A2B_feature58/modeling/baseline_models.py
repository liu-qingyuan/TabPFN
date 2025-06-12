"""
基线模型实现

包含医疗领域的传统预测模型，用于对比实验。
"""

import numpy as np
import pandas as pd
from typing import Union, Any, List, Optional


class PKUPHModel:
    """
    PKUPH模型的实现
    
    基于北京大学人民医院的预测模型：
    P(malignant) = e^x / (1+e^x)
    x = -4.496 + (0.07 × Feature2) + (0.676 × Feature48) + (0.736 × Feature49) + 
        (1.267 × Feature4) - (1.615 × Feature50) - (1.408 × Feature53)
    
    参考文献：具体医学研究论文
    """
    
    def __init__(self) -> None:
        self.intercept_ = -4.496
        self.features = ['Feature2', 'Feature48', 'Feature49', 'Feature4', 'Feature50', 'Feature53']
        self.coefficients = {
            'Feature2': 0.07,
            'Feature48': 0.676,
            'Feature49': 0.736,
            'Feature4': 1.267,
            'Feature50': -1.615,
            'Feature53': -1.408
        }
        self.is_fitted_ = False
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'PKUPHModel':
        """
        拟合模型（预定义模型，实际不需要训练）
        
        参数:
        - X: 特征数据
        - y: 标签数据
        
        返回:
        - self: 拟合后的模型
        """
        # 模型已经预定义，不需要训练
        self.is_fitted_ = True
        return self
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测类别概率
        
        参数:
        - X: 特征数据
        
        返回:
        - proba: 概率数组 [P(benign), P(malignant)]
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        # 确保X是DataFrame
        if isinstance(X, np.ndarray):
            # 假设特征顺序与self.features一致
            X_df = pd.DataFrame(X, columns=self.features)
        else:
            X_df = X.copy()
            
        # 计算线性组合
        x = np.full(len(X_df), self.intercept_)
        
        for feature, coef in self.coefficients.items():
            if feature in X_df.columns:
                feature_values = np.array(X_df[feature], dtype=float)
                x += coef * feature_values
            else:
                # 如果特征不存在，使用0值（警告用户）
                print(f"警告: 特征 {feature} 不存在于输入数据中，使用0值替代")
            
        # 计算概率 - 使用数值稳定的sigmoid
        x_clipped = np.clip(x, -500, 500)  # 防止数值溢出
        p_malignant = 1 / (1 + np.exp(-x_clipped))
        
        # 返回两列概率 [P(benign), P(malignant)]
        return np.column_stack((1 - p_malignant, p_malignant))
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测类别
        
        参数:
        - X: 特征数据
        
        返回:
        - predictions: 预测标签 (0: benign, 1: malignant)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def get_feature_names(self) -> List[str]:
        """获取模型使用的特征名称"""
        return self.features.copy()


class MayoModel:
    """
    Mayo模型的实现
    
    基于Mayo Clinic的预测模型：
    P(malignant) = e^x / (1+e^x)
    x = -6.8272 + (0.0391 × Feature2) + (0.7917 × Feature3) + (1.3388 × Feature5) + 
        (0.1274 × Feature48) + (1.0407 × Feature49) + (0.7838 × Feature63)
    
    参考文献：具体医学研究论文
    """
    
    def __init__(self) -> None:
        self.intercept_ = -6.8272
        self.features = ['Feature2', 'Feature3', 'Feature5', 'Feature48', 'Feature49', 'Feature63']
        self.coefficients = {
            'Feature2': 0.0391,
            'Feature3': 0.7917,
            'Feature5': 1.3388,
            'Feature48': 0.1274,
            'Feature49': 1.0407,
            'Feature63': 0.7838
        }
        self.is_fitted_ = False
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'MayoModel':
        """
        拟合模型（预定义模型，实际不需要训练）
        
        参数:
        - X: 特征数据
        - y: 标签数据
        
        返回:
        - self: 拟合后的模型
        """
        # 模型已经预定义，不需要训练
        self.is_fitted_ = True
        return self
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测类别概率
        
        参数:
        - X: 特征数据
        
        返回:
        - proba: 概率数组 [P(benign), P(malignant)]
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        # 确保X是DataFrame
        if isinstance(X, np.ndarray):
            # 假设特征顺序与self.features一致
            X_df = pd.DataFrame(X, columns=self.features)
        else:
            X_df = X.copy()
            
        # 计算线性组合
        x = np.full(len(X_df), self.intercept_)
        
        for feature, coef in self.coefficients.items():
            if feature in X_df.columns:
                feature_values = np.array(X_df[feature], dtype=float)
                x += coef * feature_values
            else:
                # 如果特征不存在，使用0值（警告用户）
                print(f"警告: 特征 {feature} 不存在于输入数据中，使用0值替代")
            
        # 计算概率 - 使用数值稳定的sigmoid
        x_clipped = np.clip(x, -500, 500)  # 防止数值溢出
        p_malignant = 1 / (1 + np.exp(-x_clipped))
        
        # 返回两列概率 [P(benign), P(malignant)]
        return np.column_stack((1 - p_malignant, p_malignant))
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测类别
        
        参数:
        - X: 特征数据
        
        返回:
        - predictions: 预测标签 (0: benign, 1: malignant)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def get_feature_names(self) -> List[str]:
        """获取模型使用的特征名称"""
        return self.features.copy()


def get_baseline_model(model_name: str) -> Any:
    """
    获取基线模型实例
    
    参数:
    - model_name: 模型名称 ('pkuph' 或 'mayo')
    
    返回:
    - model: 模型实例
    """
    models = {
        'pkuph': PKUPHModel,
        'mayo': MayoModel
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"不支持的基线模型: {model_name}. 支持的模型: {list(models.keys())}")
    
    return models[model_name.lower()]()


def evaluate_baseline_models(X_train: Union[np.ndarray, pd.DataFrame], 
                            y_train: Union[np.ndarray, pd.Series],
                            X_test: Union[np.ndarray, pd.DataFrame], 
                            y_test: Union[np.ndarray, pd.Series],
                            models: Optional[List[str]] = None) -> dict:
    """
    评估基线模型
    
    参数:
    - X_train: 训练特征
    - y_train: 训练标签
    - X_test: 测试特征
    - y_test: 测试标签
    - models: 要评估的模型列表，默认为['pkuph', 'mayo']
    
    返回:
    - results: 评估结果字典
    """
    if models is None:
        models = ['pkuph', 'mayo']
    
    results = {}
    
    for model_name in models:
        try:
            # 获取模型
            model = get_baseline_model(model_name)
            
            # 训练（实际上是标记为已拟合）
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
            
            results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_proba,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"评估模型 {model_name} 时出错: {e}")
            results[model_name] = {'error': str(e)}
    
    return results 