"""
UDA Medical Imbalance Project - 论文方法实现

包含基于研究论文的预测模型，这些模型使用特定的特征组合。
"""

import numpy as np
import pandas as pd
from typing import Union, Any, List, Optional
from sklearn.base import BaseEstimator, ClassifierMixin


class PaperLRModel(BaseEstimator, ClassifierMixin):
    """
    论文中的逻辑回归模型实现
    
    基于predict_healthcare_LR.py中的风险评分模型：
    risk_score = -1.137 + 0.036 × Feature2 + 0.380 × Feature5 + 0.195 × Feature48 + 
                 0.016 × Feature49 - 0.290 × Feature50 + 0.026 × Feature52 - 
                 0.168 × Feature56 - 0.236 × Feature57 + 0.052 × Feature61 + 
                 0.018 × Feature42 + 0.004 × Feature43
    
    P(malignant) = e^risk_score / (1 + e^risk_score)
    
    注意：这些特征主要在数据集A中存在，适用于源域训练。
    """
    
    def __init__(self) -> None:
        self.intercept_ = -1.137
        self.features = [
            'Feature2', 'Feature5', 'Feature48', 'Feature49', 'Feature50', 
            'Feature52', 'Feature56', 'Feature57', 'Feature61', 'Feature42', 'Feature43'
        ]
        self.coefficients = {
            'Feature2': 0.036,
            'Feature5': 0.380,
            'Feature48': 0.195,
            'Feature49': 0.016,
            'Feature50': -0.290,
            'Feature52': 0.026,
            'Feature56': -0.168,
            'Feature57': -0.236,
            'Feature61': 0.052,
            'Feature42': 0.018,
            'Feature43': 0.004
        }
        self.is_fitted_ = False
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> 'PaperLRModel':
        """
        拟合模型（预定义模型，实际不需要训练）
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            self: 拟合后的模型
        """
        # 模型已经预定义，不需要训练
        self.is_fitted_ = True
        return self
        
    def calculate_risk_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        计算风险评分
        
        Args:
            X: 特征数据
            
        Returns:
            risk_scores: 风险评分数组
        """
        # 确保X是DataFrame
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.features)
        else:
            X_df = X.copy()
            
        # 计算线性组合（风险评分）
        risk_score = np.full(len(X_df), self.intercept_)
        
        for feature, coef in self.coefficients.items():
            if feature in X_df.columns:
                feature_values = np.array(X_df[feature], dtype=float)
                risk_score += coef * feature_values
            else:
                # 如果特征不存在，使用0值（警告用户）
                print(f"警告: 特征 {feature} 不存在于输入数据中，使用0值替代")
                
        return risk_score
        
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测类别概率
        
        Args:
            X: 特征数据
            
        Returns:
            proba: 概率数组 [P(benign), P(malignant)]
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        # 计算风险评分
        risk_scores = self.calculate_risk_score(X)
        
        # 计算概率 - 使用数值稳定的sigmoid
        risk_scores_clipped = np.clip(risk_scores, -500, 500)  # 防止数值溢出
        p_malignant = np.exp(risk_scores_clipped) / (1 + np.exp(risk_scores_clipped))
        
        # 返回两列概率 [P(benign), P(malignant)]
        return np.column_stack((1 - p_malignant, p_malignant))
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测类别
        
        Args:
            X: 特征数据
            
        Returns:
            predictions: 预测标签 (0: benign, 1: malignant)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def get_feature_names(self) -> List[str]:
        """获取模型使用的特征名称"""
        return self.features.copy()
    
    def get_risk_scores(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        获取风险评分（用于分析）
        
        Args:
            X: 特征数据
            
        Returns:
            risk_scores: 风险评分数组
        """
        if not self.is_fitted_:
            raise ValueError("模型尚未拟合，请先调用fit方法")
        
        return self.calculate_risk_score(X)


def get_paper_method(method_name: str) -> Any:
    """
    获取论文方法实例
    
    Args:
        method_name: 方法名称 ('paper_lr' 或其他论文方法)
        
    Returns:
        model: 模型实例
    """
    methods = {
        'paper_lr': PaperLRModel,
        'lr_paper': PaperLRModel,  # 别名
        'paper_method': PaperLRModel,  # 通用别名
    }
    
    if method_name.lower() not in methods:
        raise ValueError(f"不支持的论文方法: {method_name}. 支持的方法: {list(methods.keys())}")
    
    return methods[method_name.lower()]()


def evaluate_paper_methods(X_train: Union[np.ndarray, pd.DataFrame], 
                          y_train: Union[np.ndarray, pd.Series],
                          X_test: Union[np.ndarray, pd.DataFrame], 
                          y_test: Union[np.ndarray, pd.Series],
                          methods: Optional[List[str]] = None) -> dict:
    """
    评估论文方法
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征
        y_test: 测试标签
        methods: 要评估的方法列表，默认为['paper_lr']
        
    Returns:
        results: 评估结果字典
    """
    if methods is None:
        methods = ['paper_lr']
    
    results = {}
    
    for method_name in methods:
        try:
            # 获取模型
            model = get_paper_method(method_name)
            
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
            
            # 如果是PaperLRModel，还可以获取风险评分
            additional_info = {}
            if hasattr(model, 'get_risk_scores'):
                risk_scores = model.get_risk_scores(X_test)
                additional_info['risk_scores'] = risk_scores
                additional_info['risk_score_stats'] = {
                    'mean': np.mean(risk_scores),
                    'std': np.std(risk_scores),
                    'min': np.min(risk_scores),
                    'max': np.max(risk_scores)
                }
            
            results[method_name] = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_proba,
                'metrics': metrics,
                **additional_info
            }
            
        except Exception as e:
            print(f"评估方法 {method_name} 时出错: {e}")
            results[method_name] = {'error': str(e)}
    
    return results


def compare_with_original_lr_script(X_test: Union[np.ndarray, pd.DataFrame], 
                                   y_test: Union[np.ndarray, pd.Series]) -> dict:
    """
    与原始LR脚本结果进行对比验证
    
    Args:
        X_test: 测试特征数据
        y_test: 测试标签
        
    Returns:
        comparison_results: 对比结果
    """
    # 使用我们的实现
    model = PaperLRModel()
    model.fit(X_test, y_test)  # 预定义模型不需要真正的训练数据
    
    # 预测
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    risk_scores = model.get_risk_scores(X_test)
    
    # 计算指标
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
    
    acc = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    f1 = f1_score(y_test, predictions)
    
    conf_matrix = confusion_matrix(y_test, predictions)
    if conf_matrix.shape == (2, 2):
        acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
        acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    else:
        acc_0 = acc_1 = 0
    
    return {
        'model_implementation': 'PaperLRModel',
        'metrics': {
            'accuracy': acc,
            'auc': auc,
            'f1': f1,
            'acc_0': acc_0,
            'acc_1': acc_1
        },
        'predictions': predictions,
        'probabilities': probabilities,
        'risk_scores': risk_scores,
        'risk_score_stats': {
            'mean': np.mean(risk_scores),
            'std': np.std(risk_scores),
            'min': np.min(risk_scores),
            'max': np.max(risk_scores)
        },
        'feature_names': model.get_feature_names()
    } 