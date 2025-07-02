"""
UDA Medical Imbalance Project - UDA预处理器

统一管理域适应算法的选择、配置和执行，集成到数据预处理流程中。
支持多种UDA方法的自动选择和参数优化。

作者: UDA Medical Team
日期: 2025-06-27
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

# 导入UDA方法
try:
    from uda.adapt_methods import (
        is_adapt_available,
        get_available_adapt_methods,
        create_adapt_method,
        AdaptUDAMethod
    )
    UDA_AVAILABLE = True
except ImportError:
    logging.warning("UDA方法模块不可用")
    UDA_AVAILABLE = False

# 导入评估工具
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


@dataclass
class UDAConfig:
    """UDA配置类"""
    # 基本配置
    method_name: str = 'KMM'
    base_estimator: Optional[Any] = None
    
    # 方法特定参数
    method_params: Dict[str, Any] = None
    
    # 评估配置
    evaluation_metrics: List[str] = None
    cross_validation: bool = True
    cv_folds: int = 5
    
    # 输出配置
    save_results: bool = True
    output_dir: str = "results/uda"
    
    def __post_init__(self):
        if self.method_params is None:
            self.method_params = {}
        if self.evaluation_metrics is None:
            self.evaluation_metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']


class UDAProcessor:
    """UDA预处理器 - 统一管理域适应算法"""
    
    # 推荐的方法配置（基于测试结果）
    RECOMMENDED_METHODS = {
        'best_overall': {
            'method': 'SA',  # Subspace Alignment - 测试中AUC最高
            'description': '子空间对齐 - 在医疗数据上表现最佳',
            'params': {
                'n_components': None,
                'verbose': 0,
                'random_state': 42
            }
        },
        'stable_performance': {
            'method': 'TCA',  # Transfer Component Analysis
            'description': '迁移成分分析 - 性能稳定',
            'params': {
                'n_components': None,
                'mu': 1.0,
                'kernel': 'linear',
                'verbose': 0,
                'random_state': 42
            }
        },
        'instance_weighting': {
            'method': 'NNW',  # Nearest Neighbors Weighting
            'description': '最近邻权重 - 实例重加权方法中最佳',
            'params': {
                'n_neighbors': 5,
                'verbose': 0,
                'random_state': 42
            }
        },
        'fast_execution': {
            'method': 'CORAL',  # Correlation Alignment
            'description': '相关性对齐 - 执行速度快',
            'params': {
                'lambda_': 1.0,
                'verbose': 0,
                'random_state': 42
            }
        }
    }
    
    # 方法类型分组
    METHOD_TYPES = {
        'instance_based': ['KMM', 'KLIEP', 'LDM', 'ULSIF', 'RULSIF', 'NNW', 'IWC', 'IWN'],
        'feature_based': ['CORAL', 'SA', 'TCA', 'FMMD', 'PRED'],
        'deep_learning': ['DANN', 'ADDA', 'WDGRL', 'DEEPCORAL', 'CDAN', 'MCD', 'MDD']
    }
    
    def __init__(self, config: Optional[UDAConfig] = None):
        """
        初始化UDA处理器
        
        参数:
        - config: UDA配置对象
        """
        self.config = config or UDAConfig()
        self.uda_method = None
        self.is_fitted = False
        self.results_history = []
        
        # 检查UDA可用性
        if not UDA_AVAILABLE:
            raise ImportError("UDA方法不可用，请检查uda.adapt_methods模块")
        
        if not is_adapt_available():
            raise ImportError("Adapt库不可用，请安装: pip install adapt-python")
        
        logger.info(f"UDA处理器初始化完成，使用方法: {self.config.method_name}")
    
    def get_available_methods(self) -> Dict[str, Dict[str, str]]:
        """获取可用的UDA方法列表"""
        return get_available_adapt_methods()
    
    def get_recommended_method(self, scenario: str = 'best_overall') -> Dict[str, Any]:
        """
        获取推荐的方法配置
        
        参数:
        - scenario: 使用场景 ('best_overall', 'stable_performance', 'instance_weighting', 'fast_execution')
        
        返回:
        - 推荐方法的配置字典
        """
        if scenario not in self.RECOMMENDED_METHODS:
            logger.warning(f"未知场景: {scenario}，使用默认推荐")
            scenario = 'best_overall'
        
        return self.RECOMMENDED_METHODS[scenario]
    
    def create_uda_method(self, method_name: Optional[str] = None, 
                         estimator: Optional[Any] = None,
                         **params) -> AdaptUDAMethod:
        """
        创建UDA方法实例
        
        参数:
        - method_name: 方法名称，默认使用配置中的方法
        - estimator: 基础估计器
        - **params: 方法特定参数
        
        返回:
        - UDA方法实例
        """
        method_name = method_name or self.config.method_name
        estimator = estimator or self.config.base_estimator
        
        # 合并参数
        merged_params = {**self.config.method_params, **params}
        
        # 创建UDA方法
        uda_method = create_adapt_method(
            method_name=method_name,
            estimator=estimator,
            **merged_params
        )
        
        logger.info(f"创建UDA方法: {method_name}")
        return uda_method
    
    def fit_transform(self, X_source: Union[np.ndarray, pd.DataFrame], 
                     y_source: Union[np.ndarray, pd.Series],
                     X_target: Union[np.ndarray, pd.DataFrame],
                     y_target: Optional[Union[np.ndarray, pd.Series]] = None,
                     method_name: Optional[str] = None) -> Tuple[AdaptUDAMethod, Dict[str, Any]]:
        """
        拟合UDA方法并进行域适应
        
        参数:
        - X_source: 源域特征
        - y_source: 源域标签
        - X_target: 目标域特征
        - y_target: 目标域标签（可选，用于评估）
        - method_name: 方法名称（可选）
        
        返回:
        - (uda_method, results): UDA方法实例和结果字典
        """
        # 创建UDA方法
        self.uda_method = self.create_uda_method(method_name)
        
        # 记录开始时间
        start_time = datetime.now()
        
        # 拟合UDA方法
        logger.info(f"开始拟合UDA方法: {self.uda_method.method_name}")
        logger.info(f"源域数据: {X_source.shape}, 目标域数据: {X_target.shape}")
        
        try:
            self.uda_method.fit(X_source, y_source, X_target, y_target)
            self.is_fitted = True
            
            # 计算拟合时间
            fit_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"UDA方法拟合完成，耗时: {fit_time:.2f}秒")
            
            # 评估结果（如果有目标域标签）
            results = {
                'method_name': self.uda_method.method_name,
                'fit_time': fit_time,
                'source_samples': len(X_source),
                'target_samples': len(X_target),
                'features': X_source.shape[1] if hasattr(X_source, 'shape') else len(X_source[0]),
                'timestamp': datetime.now().isoformat()
            }
            
            if y_target is not None:
                eval_results = self.evaluate_performance(X_target, y_target)
                results.update(eval_results)
            
            # 保存到历史记录
            self.results_history.append(results)
            
            # 保存结果（如果配置要求）
            if self.config.save_results:
                self._save_results(results)
            
            return self.uda_method, results
            
        except Exception as e:
            logger.error(f"UDA方法拟合失败: {e}")
            raise
    
    def evaluate_performance(self, X_target: Union[np.ndarray, pd.DataFrame],
                           y_target: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        评估UDA方法性能
        
        参数:
        - X_target: 目标域特征
        - y_target: 目标域标签
        
        返回:
        - 性能指标字典
        """
        if not self.is_fitted:
            raise ValueError("UDA方法未拟合，请先调用fit_transform")
        
        # 预测
        y_pred = self.uda_method.predict(X_target)
        y_pred_proba = self.uda_method.predict_proba(X_target)
        
        # 计算指标
        metrics = {}
        
        if 'accuracy' in self.config.evaluation_metrics:
            metrics['accuracy'] = float(accuracy_score(y_target, y_pred))
        
        if 'f1' in self.config.evaluation_metrics:
            metrics['f1'] = float(f1_score(y_target, y_pred, average='binary'))
        
        if 'precision' in self.config.evaluation_metrics:
            metrics['precision'] = float(precision_score(y_target, y_pred, average='binary'))
        
        if 'recall' in self.config.evaluation_metrics:
            metrics['recall'] = float(recall_score(y_target, y_pred, average='binary'))
        
        if 'auc' in self.config.evaluation_metrics and y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                metrics['auc'] = float(roc_auc_score(y_target, y_pred_proba[:, 1]))
            else:
                metrics['auc'] = float(roc_auc_score(y_target, y_pred_proba))
        
        logger.info(f"性能评估完成: {metrics}")
        return metrics
    
    def compare_methods(self, X_source: Union[np.ndarray, pd.DataFrame],
                       y_source: Union[np.ndarray, pd.Series],
                       X_target: Union[np.ndarray, pd.DataFrame],
                       y_target: Union[np.ndarray, pd.Series],
                       methods: Optional[List[str]] = None,
                       max_methods: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        比较多种UDA方法的性能
        
        参数:
        - X_source, y_source: 源域数据
        - X_target, y_target: 目标域数据
        - methods: 要比较的方法列表，None则使用推荐方法
        - max_methods: 最大比较方法数
        
        返回:
        - 各方法的结果字典
        """
        if methods is None:
            # 使用推荐方法
            methods = [
                'SA',     # 最佳整体性能
                'TCA',    # 稳定性能
                'NNW',    # 最佳实例重加权
                'CORAL',  # 快速执行
                'FMMD'    # 特征匹配
            ]
        
        methods = methods[:max_methods]  # 限制方法数量
        comparison_results = {}
        
        logger.info(f"开始比较 {len(methods)} 种UDA方法")
        
        for method_name in methods:
            try:
                logger.info(f"测试方法: {method_name}")
                
                # 创建临时配置
                temp_config = UDAConfig(
                    method_name=method_name,
                    base_estimator=self.config.base_estimator,
                    save_results=False  # 比较时不保存单个结果
                )
                
                # 创建临时处理器
                temp_processor = UDAProcessor(temp_config)
                
                # 拟合和评估
                _, results = temp_processor.fit_transform(
                    X_source, y_source, X_target, y_target, method_name
                )
                
                comparison_results[method_name] = results
                
            except Exception as e:
                logger.error(f"方法 {method_name} 测试失败: {e}")
                comparison_results[method_name] = {'error': str(e)}
        
        # 找出最佳方法
        valid_results = {k: v for k, v in comparison_results.items() 
                        if 'error' not in v and 'auc' in v}
        
        if valid_results:
            best_method = max(valid_results.keys(), 
                            key=lambda k: valid_results[k].get('auc', 0))
            logger.info(f"最佳方法: {best_method} (AUC: {valid_results[best_method]['auc']:.4f})")
            
            # 添加比较摘要
            comparison_results['comparison_summary'] = {
                'best_method': best_method,
                'best_auc': valid_results[best_method]['auc'],
                'methods_tested': len(methods),
                'successful_methods': len(valid_results),
                'timestamp': datetime.now().isoformat()
            }
        
        # 保存比较结果
        if self.config.save_results:
            self._save_results(comparison_results, 'method_comparison')
        
        return comparison_results
    
    def get_method_recommendation(self, X_source: Union[np.ndarray, pd.DataFrame],
                                 X_target: Union[np.ndarray, pd.DataFrame],
                                 requirements: Optional[Dict[str, Any]] = None) -> str:
        """
        基于数据特征和需求推荐UDA方法
        
        参数:
        - X_source: 源域特征
        - X_target: 目标域特征
        - requirements: 需求字典 (如 {'speed': 'fast', 'accuracy': 'high'})
        
        返回:
        - 推荐的方法名称
        """
        requirements = requirements or {}
        
        # 分析数据特征
        n_samples_source = len(X_source)
        n_samples_target = len(X_target)
        n_features = X_source.shape[1] if hasattr(X_source, 'shape') else len(X_source[0])
        
        logger.info(f"数据分析: 源域{n_samples_source}样本, 目标域{n_samples_target}样本, {n_features}特征")
        
        # 基于数据规模和需求推荐
        if requirements.get('speed') == 'fast' or n_samples_source > 1000:
            # 大数据集或需要快速执行
            recommended = 'CORAL'
            reason = "快速执行，适合大数据集"
        elif requirements.get('accuracy') == 'high' or n_features > 20:
            # 高精度需求或高维特征
            recommended = 'SA'
            reason = "高精度，适合高维特征"
        elif n_samples_source < 200:
            # 小样本数据集
            recommended = 'NNW'
            reason = "适合小样本数据集"
        else:
            # 默认推荐
            recommended = 'TCA'
            reason = "稳定性能，通用性好"
        
        logger.info(f"推荐方法: {recommended} - {reason}")
        return recommended
    
    def _save_results(self, results: Dict[str, Any], 
                     result_type: str = 'uda_results') -> Path:
        """保存结果到文件"""
        # 创建输出目录
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result_type}_{timestamp}.json"
        filepath = output_dir / filename
        
        # 保存JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"结果已保存到: {filepath}")
        return filepath
    
    def get_results_summary(self) -> Dict[str, Any]:
        """获取历史结果摘要"""
        if not self.results_history:
            return {'message': '暂无历史结果'}
        
        summary = {
            'total_experiments': len(self.results_history),
            'methods_used': list(set(r['method_name'] for r in self.results_history)),
            'average_performance': {},
            'best_result': None
        }
        
        # 计算平均性能
        metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
        for metric in metrics:
            values = [r.get(metric) for r in self.results_history if metric in r]
            if values:
                summary['average_performance'][metric] = np.mean(values)
        
        # 找出最佳结果
        auc_results = [r for r in self.results_history if 'auc' in r]
        if auc_results:
            summary['best_result'] = max(auc_results, key=lambda x: x['auc'])
        
        return summary


def create_uda_processor(method_name: str = 'SA', 
                        base_estimator: Optional[Any] = None,
                        **kwargs) -> UDAProcessor:
    """
    便捷函数：创建UDA处理器
    
    参数:
    - method_name: UDA方法名称
    - base_estimator: 基础估计器
    - **kwargs: 其他配置参数
    
    返回:
    - UDA处理器实例
    """
    config = UDAConfig(
        method_name=method_name,
        base_estimator=base_estimator,
        **kwargs
    )
    return UDAProcessor(config)


def get_uda_recommendations() -> Dict[str, Any]:
    """
    获取UDA方法使用建议
    
    返回:
    - 使用建议字典
    """
    return {
        'medical_data_recommendations': {
            'best_overall': 'SA - 子空间对齐，在医疗数据测试中AUC最高(0.7008)',
            'stable_choice': 'TCA - 迁移成分分析，性能稳定可靠',
            'fast_execution': 'CORAL - 相关性对齐，执行速度快',
            'small_dataset': 'NNW - 最近邻权重，适合小样本数据'
        },
        'method_types': {
            'instance_based': '实例重加权方法，通过调整样本权重减少域差异',
            'feature_based': '特征对齐方法，通过特征变换减少域差异',
            'deep_learning': '深度学习方法，需要大量数据和计算资源'
        },
        'usage_tips': [
            '医疗数据推荐使用SA或TCA方法',
            '小数据集(<200样本)推荐NNW方法',
            '需要快速执行推荐CORAL方法',
            '高维特征(>20维)推荐SA方法',
            '可以使用compare_methods进行多方法对比'
        ]
    }


if __name__ == "__main__":
    # 使用示例
    print("UDA处理器使用示例:")
    
    # 1. 检查可用方法
    if UDA_AVAILABLE and is_adapt_available():
        processor = UDAProcessor()
        methods = processor.get_available_methods()
        print(f"可用方法数量: {len(methods)}")
        
        # 2. 获取推荐
        recommendations = get_uda_recommendations()
        print("医疗数据推荐方法:")
        for key, value in recommendations['medical_data_recommendations'].items():
            print(f"  {key}: {value}")
    else:
        print("UDA方法不可用，请检查环境配置") 