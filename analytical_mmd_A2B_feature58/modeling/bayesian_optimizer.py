"""
贝叶斯优化模块

实现基于目标域验证集的贝叶斯超参数优化，采用三分法数据划分：
1. A域训练集：用于模型训练
2. 目标域验证集：用于贝叶斯优化目标函数评估
3. 目标域保留测试集：用于最终模型泛化能力评估
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import warnings
import time

# 导入 AutoTabPFNClassifier
try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
    AUTO_TABPFN_AVAILABLE = True
except ImportError:
    logging.warning("AutoTabPFN不可用，请安装tabpfn_extensions")
    AUTO_TABPFN_AVAILABLE = False

from ..config.settings import (
    DATA_PATHS, LABEL_COL, get_features_by_type, get_categorical_indices
)
from ..data.loader import load_all_datasets
from ..preprocessing.scaler import fit_apply_scaler
from ..modeling.model_selector import get_model

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)

class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy数据类型"""
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)

class BayesianOptimizer:
    """贝叶斯优化器类"""
    
    def __init__(self, 
                 model_type: str = 'auto',
                 feature_type: str = 'best7',
                 use_categorical: bool = True,
                 validation_split: float = 0.7,
                 n_calls: int = 50,
                 random_state: int = 42,
                 save_path: str = './results_bayesian_optimization',
                 target_domain: str = 'B'):  # 添加目标域参数
        """
        初始化贝叶斯优化器
        
        参数:
        - model_type: 模型类型 ('auto', 'base', 'rf', 'tuned')
        - feature_type: 特征类型 ('all', 'best7')
        - use_categorical: 是否使用类别特征
        - validation_split: 验证集比例 (0.7表示70%用于验证，30%用于holdout)
        - n_calls: 贝叶斯优化迭代次数
        - random_state: 随机种子
        - save_path: 结果保存路径
        - target_domain: 目标域选择 ('B' 或 'C')
        """
        self.model_type = model_type
        self.feature_type = feature_type
        self.use_categorical = use_categorical
        self.validation_split = validation_split
        self.n_calls = n_calls
        self.random_state = random_state
        self.save_path = save_path
        self.target_domain = target_domain  # 添加目标域属性
        
        # 获取特征和类别索引
        self.features = get_features_by_type(feature_type)
        self.categorical_indices = get_categorical_indices(feature_type) if use_categorical else []
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 初始化数据存储
        self.X_train = None
        self.y_train = None
        self.X_ext_val = None
        self.y_ext_val = None
        self.X_ext_holdout = None
        self.y_ext_holdout = None
        self.scaler = None
        
        # 优化结果存储
        self.optimization_results = []
        self.best_params = None
        self.best_score = None
        self.final_model = None
        
        # 添加优秀配置存储
        self.good_configs = []  # 存储测试集AUC > 0.7的配置
        
        logging.info(f"初始化贝叶斯优化器:")
        logging.info(f"  模型类型: {model_type}")
        logging.info(f"  特征类型: {feature_type} ({len(self.features)}个特征)")
        logging.info(f"  使用类别特征: {use_categorical}")
        logging.info(f"  验证集比例: {validation_split}")
        logging.info(f"  优化迭代次数: {n_calls}")
        logging.info(f"  目标域: {target_domain}")  # 添加目标域日志
        logging.info(f"  固定参数: validation_method=cv, n_folds=5, max_time=1")
        
    def load_and_prepare_data(self) -> None:
        """加载并准备数据，实现三分法划分"""
        logging.info("加载和准备数据...")
        
        # 加载数据集A (训练集)
        df_A = pd.read_excel(DATA_PATHS['A'])
        X_A_raw = df_A[self.features].values
        y_A = df_A[LABEL_COL].values
        
        # 加载目标域数据集 (需要划分为验证集和保留测试集)
        df_target = pd.read_excel(DATA_PATHS[self.target_domain])
        X_target_raw = df_target[self.features].values
        y_target = df_target[LABEL_COL].values
        
        # 数据标准化 - 用A数据集拟合scaler
        X_A_scaled, X_target_scaled, self.scaler = fit_apply_scaler(X_A_raw, X_target_raw)
        
        # 设置A域训练数据
        self.X_train = X_A_scaled
        self.y_train = y_A
        
        # 对目标域数据进行三分法划分：验证集 vs 保留测试集
        self.X_ext_val, self.X_ext_holdout, self.y_ext_val, self.y_ext_holdout = train_test_split(
            X_target_scaled, y_target,
            train_size=self.validation_split,
            stratify=y_target,
            random_state=self.random_state
        )
        
        # 打印数据信息
        logging.info(f"数据划分完成:")
        logging.info(f"  A域训练集: {self.X_train.shape[0]} 样本")
        logging.info(f"  {self.target_domain}域验证集: {self.X_ext_val.shape[0]} 样本")
        logging.info(f"  {self.target_domain}域保留测试集: {self.X_ext_holdout.shape[0]} 样本")
        logging.info(f"  A域标签分布: {np.bincount(self.y_train.astype(int))}")
        logging.info(f"  {self.target_domain}域验证集标签分布: {np.bincount(self.y_ext_val.astype(int))}")
        logging.info(f"  {self.target_domain}域保留测试集标签分布: {np.bincount(self.y_ext_holdout.astype(int))}")
        
    def define_search_space(self) -> List:
        """定义超参数搜索空间"""
        if self.model_type == 'auto':
            # 优化的AutoTabPFN参数空间 - 修复max_time支持问题
            search_space = [
                # 训练时间 - 限制为AutoTabPFN支持的范围 (1-180秒)
                # 根据错误日志，大于180的值不被支持
                Categorical([1, 5, 10, 15, 30, 60, 120, 180], name='max_time'),
                
                # 预设配置 - 对泛化能力很重要
                Categorical(['default', 'avoid_overfitting'], name='preset'),
                
                # 评分指标 - 医疗数据通常关注AUC和F1
                Categorical(['accuracy', 'roc', 'f1'], name='ges_scoring'),
                
                # 模型数量 - 适中范围，避免过拟合
                Categorical([10, 15, 20, 25, 30], name='max_models'),
                
                # 验证方法 - CV对小数据集更稳定
                Categorical(['cv'], name='validation_method'),  # 固定为CV，更适合医疗数据
                
                # 重复次数 - 减少范围，提高效率
                Integer(50, 150, name='n_repeats'),
                
                # 折数 - 医疗数据常用5折
                Categorical([5], name='n_folds'),  # 固定为5折
                
                # GES迭代次数 - 重要的超参数
                Integer(20, 40, name='ges_n_iterations'),
                
                # 是否忽略预训练限制 - 对跨域可能重要
                Categorical([True, False], name='ignore_limits'),
            ]
            
            logging.info("定义了9个超参数的搜索空间")
            logging.info("搜索空间优化要点:")
            logging.info("  - 修复max_time范围 (1-180秒)，移除不支持的300秒")
            logging.info("  - 专注于AUC和F1评分指标")
            logging.info("  - 固定使用CV验证方法")
            logging.info("  - 移除holdout_fraction参数")
            logging.info("  - 增加ges_n_iterations搜索范围")
            
        elif self.model_type == 'rf':
            # Random Forest参数空间
            search_space = [
                Integer(50, 500, name='n_estimators'),  # 树的数量
                Integer(1, 20, name='max_depth'),  # 最大深度
                Integer(2, 20, name='min_samples_split'),  # 最小分割样本数
                Integer(1, 10, name='min_samples_leaf'),  # 最小叶子样本数
                Real(0.1, 1.0, name='max_features'),  # 最大特征比例
                Categorical(['auto', 'balanced'], name='class_weight'),  # 类别权重
            ]
        elif self.model_type == 'base':
            # 基础TabPFN参数空间 (参数较少)
            search_space = [
                Integer(1, 5, name='N_ensemble_configurations'),  # 集成配置数
                Categorical(['auto', 'balanced'], name='class_weight'),  # 类别权重
            ]
        else:
            # 默认参数空间
            search_space = [
                Real(0.01, 1.0, name='learning_rate'),
                Integer(10, 200, name='max_iter'),
            ]
        
        return search_space
    
    def objective_function(self, params: List) -> float:
        """
        目标函数：评估给定超参数组合的性能
        
        参数:
        - params: 超参数值列表，顺序对应搜索空间定义
        
        返回:
        - float: 负验证集AUC (因为gp_minimize最小化目标函数)
        """
        try:
            # 将参数列表转换为字典
            search_space = self.define_search_space()
            param_dict = {dim.name: params[i] for i, dim in enumerate(search_space)}
            
            logging.info(f"开始评估参数组合: {param_dict}")
            
            # 根据模型类型创建模型
            if self.model_type == 'auto':
                # 检查AutoTabPFN是否可用
                if not AUTO_TABPFN_AVAILABLE:
                    logging.error("AutoTabPFN不可用，无法创建模型")
                    raise ImportError("AutoTabPFN不可用，请安装tabpfn_extensions")
                
                # 构建phe_init_args参数
                phe_init_args = {
                    'max_models': param_dict.get('max_models', 15),
                    'validation_method': param_dict.get('validation_method', 'cv'),
                    'n_repeats': param_dict.get('n_repeats', 100),
                    'n_folds': param_dict.get('n_folds', 5),
                    'ges_n_iterations': param_dict.get('ges_n_iterations', 20)
                }
                
                logging.info(f"创建AutoTabPFN模型:")
                logging.info(f"  max_time: {param_dict.get('max_time', 30)}")
                logging.info(f"  preset: {param_dict.get('preset', 'default')}")
                logging.info(f"  ges_scoring: {param_dict.get('ges_scoring', 'roc')}")
                logging.info(f"  phe_init_args: {phe_init_args}")
                
                # 创建AutoTabPFN模型
                model = AutoTabPFNClassifier(
                    max_time=param_dict.get('max_time', 30),  # 使用搜索到的max_time
                    preset=param_dict.get('preset', 'default'),
                    ges_scoring_string=param_dict.get('ges_scoring', 'roc'),
                    device='cuda',
                    random_state=self.random_state,
                    ignore_pretraining_limits=param_dict.get('ignore_limits', False),
                    categorical_feature_indices=self.categorical_indices if self.categorical_indices else None,
                    phe_init_args=phe_init_args
                )
                
                logging.info("AutoTabPFN模型创建成功")
                
            else:
                # 其他模型类型的处理保持原有逻辑
                logging.info(f"创建{self.model_type}模型，参数: {param_dict}")
                model_config = {
                    'categorical_feature_indices': self.categorical_indices if self.use_categorical else []
                }
                model_config.update(param_dict)
                model = get_model(self.model_type, **model_config)
                logging.info(f"{self.model_type}模型创建成功")
            
            # 训练模型
            if self.X_train is not None:
                logging.info(f"开始训练模型，训练集大小: {self.X_train.shape}")
                start_time = time.time()
                model.fit(self.X_train, self.y_train)
                training_time = time.time() - start_time
                logging.info(f"模型训练完成，耗时: {training_time:.2f}秒")
            else:
                logging.error("训练数据未加载")
                raise ValueError("训练数据未加载，请先调用load_and_prepare_data()")
            
            # 在验证集上预测
            if self.X_ext_val is not None:
                logging.info(f"在验证集上预测，验证集大小: {self.X_ext_val.shape}")
                y_pred_proba = model.predict_proba(self.X_ext_val)
                if y_pred_proba.ndim > 1:
                    y_pred_proba = y_pred_proba[:, 1]  # 取正类概率
            else:
                logging.error("验证数据未加载")
                raise ValueError("验证数据未加载，请先调用load_and_prepare_data()")
            
            # 计算验证集AUC
            val_auc_score = roc_auc_score(self.y_ext_val, y_pred_proba)
            logging.info(f"验证集AUC: {val_auc_score:.4f}")
            
            # 在测试集上预测和评估
            if self.X_ext_holdout is not None:
                logging.info(f"在测试集上预测，测试集大小: {self.X_ext_holdout.shape}")
                y_test_pred_proba = model.predict_proba(self.X_ext_holdout)
                if y_test_pred_proba.ndim > 1:
                    y_test_pred_proba = y_test_pred_proba[:, 1]
            else:
                logging.error("测试数据未加载")
                raise ValueError("测试数据未加载，请先调用load_and_prepare_data()")
                
            # 计算测试集AUC
            test_auc_score = roc_auc_score(self.y_ext_holdout, y_test_pred_proba)
            logging.info(f"测试集AUC: {test_auc_score:.4f}")
            
            # 记录试验结果
            trial_result = {
                'params': {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                          for k, v in param_dict.items()},  # 转换numpy类型为Python原生类型
                'validation_auc': float(val_auc_score),
                'test_auc': float(test_auc_score),
                'training_time': float(training_time),
                'trial_id': len(self.optimization_results)
            }
            self.optimization_results.append(trial_result)
            
            # 如果测试集AUC > 0.7，保存配置
            if test_auc_score > 0.7:
                good_config = {
                    'trial_id': len(self.optimization_results),
                    'params': {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                              for k, v in param_dict.items()},
                    'validation_auc': float(val_auc_score),
                    'test_auc': float(test_auc_score),
                    'training_time': float(training_time)
                }
                self.good_configs.append(good_config)
                logging.info(f"★ 优秀配置 {len(self.good_configs)}: 测试集AUC={test_auc_score:.4f} > 0.7，已保存配置")
            
            logging.info(f"试验 {len(self.optimization_results)}: 验证集AUC={val_auc_score:.4f}, 测试集AUC={test_auc_score:.4f}, 参数={param_dict}")
            
            # 返回负验证集AUC (因为gp_minimize最小化目标函数)
            return float(-val_auc_score)
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            logging.error("=" * 60)
            logging.error("目标函数评估失败 - 详细错误报告")
            logging.error("=" * 60)
            logging.error(f"错误类型: {error_type}")
            logging.error(f"错误信息: {error_msg}")
            logging.error(f"失败的参数组合: {param_dict}")
            logging.error(f"模型类型: {self.model_type}")
            
            # 详细的错误分析
            if 'max_time' in error_msg and 'not supported' in error_msg:
                logging.error("🔍 错误分析: max_time参数不被支持")
                logging.error(f"   当前max_time值: {param_dict.get('max_time', 'unknown')}")
                logging.error("   可能原因:")
                logging.error("   1. 传递给了普通TabPFN而不是AutoTabPFN")
                logging.error("   2. AutoTabPFN版本不支持该max_time值")
                logging.error("   3. 模型创建过程中参数传递错误")
                
            elif 'memory' in error_msg.lower() or 'cuda' in error_msg.lower():
                logging.error("🔍 错误分析: GPU内存相关问题")
                logging.error("   建议解决方案:")
                logging.error("   1. 减少max_models参数")
                logging.error("   2. 减少n_repeats参数")
                logging.error("   3. 使用CPU设备")
                
            elif 'import' in error_msg.lower() or 'module' in error_msg.lower():
                logging.error("🔍 错误分析: 模块导入问题")
                logging.error("   可能缺少依赖包:")
                logging.error("   1. tabpfn_extensions")
                logging.error("   2. AutoTabPFNClassifier")
                
            elif 'categorical' in error_msg.lower():
                logging.error("🔍 错误分析: 类别特征处理问题")
                logging.error(f"   类别特征索引: {self.categorical_indices}")
                
            else:
                logging.error("🔍 错误分析: 未知错误类型")
                logging.error("   建议:")
                logging.error("   1. 检查数据格式和大小")
                logging.error("   2. 验证模型参数有效性")
                logging.error("   3. 查看完整的错误堆栈")
            
            # 导入traceback以获取详细错误信息
            import traceback
            logging.error("完整错误堆栈:")
            logging.error(traceback.format_exc())
            logging.error("=" * 60)
            
            # 返回一个较大的值表示失败 (因为我们要最小化)
            return 1.0
    
    def run_optimization(self) -> Dict[str, Any]:
        """运行贝叶斯优化"""
        logging.info("开始贝叶斯优化...")
        
        # 确保数据已加载
        if self.X_train is None:
            self.load_and_prepare_data()
        
        # 定义搜索空间
        search_space = self.define_search_space()
        
        # 创建目标函数装饰器
        @use_named_args(search_space)
        def objective(**params):
            # 将命名参数转换为列表形式
            param_values = [params[dim.name] for dim in search_space]
            return self.objective_function(param_values)
        
        # 运行贝叶斯优化
        logging.info(f"开始{self.n_calls}次贝叶斯优化迭代...")
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            acq_func='EI',  # Expected Improvement
            n_initial_points=20  # 20%的初始随机点数，更合理的比例
        )
        
        # 提取最佳参数
        self.best_params = {}
        for i, dim in enumerate(search_space):
            self.best_params[dim.name] = result.x[i]
        
        self.best_score = -result.fun  # 转换回正AUC
        
        logging.info("贝叶斯优化完成!")
        logging.info(f"最佳验证集AUC: {self.best_score:.4f}")
        logging.info(f"最佳参数: {self.best_params}")
        
        # 输出优秀配置汇总
        if self.good_configs:
            logging.info(f"\n发现 {len(self.good_configs)} 个优秀配置 (测试集AUC > 0.7):")
            for i, config in enumerate(self.good_configs):
                logging.info(f"  配置 {i+1}: 验证集AUC={config['validation_auc']:.4f}, 测试集AUC={config['test_auc']:.4f}")
        else:
            logging.info("未发现测试集AUC > 0.7的配置")
        
        return {
            'best_params': self.best_params,
            'best_validation_auc': self.best_score,
            'optimization_history': self.optimization_results,
            'good_configs': self.good_configs,  # 添加优秀配置
            'total_trials': len(self.optimization_results)
        }
    
    def train_final_model(self) -> Dict[str, Any]:
        """使用最佳参数在完整A域数据上训练最终模型"""
        logging.info("使用最佳参数训练最终模型...")
        
        if self.best_params is None:
            raise ValueError("请先运行贝叶斯优化获取最佳参数")
        
        # 根据模型类型创建最终模型
        if self.model_type == 'auto':
            # 创建AutoTabPFN模型
            self.final_model = AutoTabPFNClassifier(
                max_time=self.best_params.get('max_time', 30),  # 使用搜索到的max_time
                preset=self.best_params.get('preset', 'default'),
                ges_scoring_string=self.best_params.get('ges_scoring', 'roc'),
                device='cuda',
                random_state=self.random_state,
                ignore_pretraining_limits=self.best_params.get('ignore_limits', False),
                categorical_feature_indices=self.categorical_indices if self.categorical_indices else None,
                phe_init_args={
                    'max_models': self.best_params.get('max_models', 15),
                    'validation_method': 'cv',
                    'n_folds': 5,
                    'ges_n_iterations': self.best_params.get('ges_n_iterations', 20)
                }
            )
        else:
            # 其他模型类型的处理保持原有逻辑
            final_config = {
                'categorical_feature_indices': self.categorical_indices if self.use_categorical else []
            }
            final_config.update(self.best_params)
            self.final_model = get_model(self.model_type, **final_config)
        
        # 训练最终模型
        self.final_model.fit(self.X_train, self.y_train)
        
        logging.info("最终模型训练完成")
        
        return self.best_params
    
    def evaluate_final_model(self) -> Dict[str, Any]:
        """在保留测试集上评估最终模型"""
        logging.info("在保留测试集上评估最终模型...")
        
        if self.final_model is None:
            raise ValueError("请先训练最终模型")
        
        # 在保留测试集上预测
        y_pred = self.final_model.predict(self.X_ext_holdout)
        y_pred_proba = self.final_model.predict_proba(self.X_ext_holdout)
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 1]
        
        # 计算评估指标
        holdout_results = {
            'auc': roc_auc_score(self.y_ext_holdout, y_pred_proba),
            'f1': f1_score(self.y_ext_holdout, y_pred),
            'accuracy': accuracy_score(self.y_ext_holdout, y_pred),
            'confusion_matrix': confusion_matrix(self.y_ext_holdout, y_pred).tolist()
        }
        
        # 同时在验证集上评估以便对比
        y_val_pred = self.final_model.predict(self.X_ext_val)
        y_val_pred_proba = self.final_model.predict_proba(self.X_ext_val)
        if y_val_pred_proba.ndim > 1:
            y_val_pred_proba = y_val_pred_proba[:, 1]
        
        validation_results = {
            'auc': roc_auc_score(self.y_ext_val, y_val_pred_proba),
            'f1': f1_score(self.y_ext_val, y_val_pred),
            'accuracy': accuracy_score(self.y_ext_val, y_val_pred),
            'confusion_matrix': confusion_matrix(self.y_ext_val, y_val_pred).tolist()
        }
        
        logging.info("最终模型评估完成:")
        logging.info(f"  验证集 - AUC: {validation_results['auc']:.4f}, F1: {validation_results['f1']:.4f}, Acc: {validation_results['accuracy']:.4f}")
        logging.info(f"  保留测试集 - AUC: {holdout_results['auc']:.4f}, F1: {holdout_results['f1']:.4f}, Acc: {holdout_results['accuracy']:.4f}")
        
        return {
            'validation_performance': validation_results,
            'holdout_performance': holdout_results
        }
    
    def plot_confusion_matrix(self, results: Dict[str, Any]) -> None:
        """绘制混淆矩阵"""
        logging.info("绘制混淆矩阵...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 验证集混淆矩阵
        cm_val = np.array(results['validation_performance']['confusion_matrix'])
        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Validation Set Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 保留测试集混淆矩阵
        cm_holdout = np.array(results['holdout_performance']['confusion_matrix'])
        sns.heatmap(cm_holdout, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_title('Holdout Test Set Confusion Matrix')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        
        # 保存图片
        cm_path = os.path.join(self.save_path, 'confusion_matrices.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"混淆矩阵已保存到: {cm_path}")
    
    def save_results(self, optimization_results: Dict[str, Any], 
                    final_results: Dict[str, Any]) -> None:
        """保存所有结果"""
        logging.info("保存优化结果...")
        
        # 保存优化历史
        optimization_path = os.path.join(self.save_path, 'optimization_history.json')
        with open(optimization_path, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # 保存最终评估结果
        final_path = os.path.join(self.save_path, 'final_evaluation.json')
        final_save_data = {
            'best_params': self.best_params,
            'model_config': {
                'model_type': self.model_type,
                'feature_type': self.feature_type,
                'use_categorical': self.use_categorical,
                'features_count': len(self.features),
                'categorical_indices_count': len(self.categorical_indices)
            },
            'data_split': {
                'train_samples': self.X_train.shape[0],
                'validation_samples': self.X_ext_val.shape[0],
                'holdout_samples': self.X_ext_holdout.shape[0],
                'validation_split_ratio': self.validation_split
            },
            'performance': final_results
        }
        
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_save_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        logging.info(f"优化历史已保存到: {optimization_path}")
        logging.info(f"最终评估结果已保存到: {final_path}")
    
    def run_complete_optimization(self) -> Dict[str, Any]:
        """运行完整的贝叶斯优化流程"""
        logging.info("=" * 60)
        logging.info("开始完整贝叶斯优化流程")
        logging.info("=" * 60)
        
        try:
            # 1. 加载和准备数据
            self.load_and_prepare_data()
            
            # 2. 运行贝叶斯优化
            optimization_results = self.run_optimization()
            
            # 3. 训练最终模型
            final_config = self.train_final_model()
            
            # 4. 评估最终模型
            final_results = self.evaluate_final_model()
            
            # 5. 绘制混淆矩阵
            self.plot_confusion_matrix(final_results)
            
            # 6. 保存结果
            self.save_results(optimization_results, final_results)
            
            logging.info("=" * 60)
            logging.info("贝叶斯优化流程完成!")
            logging.info("=" * 60)
            
            return {
                'optimization_results': optimization_results,
                'final_results': final_results,
                'final_config': final_config
            }
            
        except Exception as e:
            logging.error(f"贝叶斯优化流程失败: {e}")
            raise

def run_bayesian_optimization(model_type: str = 'auto',
                            feature_type: str = 'best7',
                            use_categorical: bool = True,
                            validation_split: float = 0.7,
                            n_calls: int = 50,
                            save_path: str = './results_bayesian_optimization',
                            target_domain: str = 'B',  # 添加目标域参数
                            **kwargs) -> Dict[str, Any]:
    """运行完整的贝叶斯优化流程
    
    参数:
        model_type: 模型类型
        feature_type: 特征类型
        use_categorical: 是否使用类别特征
        validation_split: 验证集比例
        n_calls: 贝叶斯优化迭代次数
        save_path: 结果保存路径
        target_domain: 目标域选择 ('B' 或 'C')
    """
    logging.info("=" * 60)
    logging.info("开始完整贝叶斯优化流程")
    logging.info("=" * 60)
    
    # 创建优化器实例
    optimizer = BayesianOptimizer(
        model_type=model_type,
        feature_type=feature_type,
        use_categorical=use_categorical,
        validation_split=validation_split,
        n_calls=n_calls,
        save_path=save_path,
        target_domain=target_domain  # 传递目标域参数
    )
    
    # 运行完整优化流程
    results = optimizer.run_complete_optimization()
    
    return results 