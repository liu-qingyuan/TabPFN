"""
贝叶斯优化与MMD域适应集成模块

实现同时优化模型超参数和MMD域适应参数的贝叶斯优化器，
采用三分法数据划分策略，确保模型选择和最终评估的独立性。

工作流程：
1. A域训练集：用于模型训练
2. B域验证集：用于贝叶斯优化目标函数评估（包含MMD域适应）
3. B域保留测试集：用于最终模型泛化能力评估
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
    DATA_PATHS, LABEL_COL, get_features_by_type, get_categorical_indices, MMD_METHODS
)
from ..data.loader import load_all_datasets
from ..preprocessing.scaler import fit_apply_scaler
from ..preprocessing.mmd import mmd_transform
from ..preprocessing.class_conditional_mmd import class_conditional_mmd_transform
from ..modeling.model_selector import get_model
from ..metrics.evaluation import evaluate_metrics

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

class BayesianMMDOptimizer:
    """贝叶斯优化器类 - 只优化MMD参数，使用固定的最佳模型参数"""
    
    def __init__(self, 
                 model_type: str = 'auto',
                 feature_type: str = 'best7',
                 mmd_method: str = 'linear',
                 use_class_conditional: bool = False,
                 use_categorical: bool = True,
                 validation_split: float = 0.7,
                 n_calls: int = 50,
                 random_state: int = 42,
                 target_domain: str = 'B',
                 save_path: str = './results_bayesian_mmd_optimization',
                 use_fixed_model_params: bool = True,
                 evaluate_source_cv: bool = False):
        """
        初始化贝叶斯MMD优化器
        
        参数:
        - model_type: 模型类型 ('auto', 'base', 'rf', 'tuned')
        - feature_type: 特征类型 ('all', 'best7')
        - mmd_method: MMD方法 ('linear', 'kpca', 'mean_std')
        - use_class_conditional: 是否使用类条件MMD
        - use_categorical: 是否使用类别特征
        - validation_split: 验证集比例 (0.7表示70%用于验证，30%用于holdout)
        - n_calls: 贝叶斯优化迭代次数
        - random_state: 随机种子
        - target_domain: 目标域选择 ('B' 或 'C')
        - save_path: 结果保存路径
        - use_fixed_model_params: 是否使用固定的最佳模型参数（推荐True）
        - evaluate_source_cv: 是否评估A域交叉验证基准（可选）
        """
        self.model_type = model_type
        self.feature_type = feature_type
        self.mmd_method = mmd_method
        self.use_class_conditional = use_class_conditional
        self.use_categorical = use_categorical
        self.validation_split = validation_split
        self.n_calls = n_calls
        self.random_state = random_state
        self.target_domain = target_domain
        self.save_path = save_path
        self.use_fixed_model_params = use_fixed_model_params
        self.evaluate_source_cv = evaluate_source_cv
        
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
        self.X_train_raw = None
        self.X_ext_val_raw = None
        self.X_ext_holdout_raw = None
        self.scaler = None
        
        # 优化结果存储
        self.optimization_results = []
        self.best_params = None
        self.best_score = None
        self.final_model = None
        
        # 添加优秀配置存储
        self.good_configs = []  # 存储测试集AUC > 0.7的配置
        
        # 固定的最佳模型参数（基于贝叶斯优化实验结果）
        self.fixed_model_params = {
            'auto': {
                'max_time': 180,  # 基于贝叶斯优化的最佳结果
                'preset': 'default',
                'ges_scoring': 'roc',
                'max_models': 25,
                'n_repeats': 136,
                'ges_n_iterations': 35,
                'ignore_limits': False,
            },
            'rf': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 0.8,
                'class_weight': 'balanced'
            }
        }
        
        logging.info(f"初始化贝叶斯MMD优化器:")
        logging.info(f"  模型类型: {model_type}")
        logging.info(f"  特征类型: {feature_type} ({len(self.features)}个特征)")
        logging.info(f"  MMD方法: {mmd_method}")
        logging.info(f"  类条件MMD: {use_class_conditional}")
        logging.info(f"  使用类别特征: {use_categorical}")
        logging.info(f"  验证集比例: {validation_split}")
        logging.info(f"  优化迭代次数: {n_calls}")
        logging.info(f"  目标域: {target_domain}")
        logging.info(f"  使用固定模型参数: {use_fixed_model_params}")
        logging.info(f"  评估A域CV基准: {evaluate_source_cv}")
        
        if use_fixed_model_params:
            logging.info(f"  固定模型参数: {self.fixed_model_params.get(model_type, {})}")

    def load_and_prepare_data(self) -> None:
        """加载并准备数据，实现三分法划分"""
        logging.info("加载和准备数据...")
        
        # 加载数据集A (训练集)
        df_A = pd.read_excel(DATA_PATHS['A'])
        X_A_raw = df_A[self.features].values
        y_A = df_A[LABEL_COL].values
        
        # 根据target_domain加载目标域数据集
        if self.target_domain == 'B':
            df_target = pd.read_excel(DATA_PATHS['B'])
        elif self.target_domain == 'C':
            df_target = pd.read_excel(DATA_PATHS['C'])
        else:
            raise ValueError(f"不支持的目标域: {self.target_domain}")
        
        X_target_raw = df_target[self.features].values
        y_target = df_target[LABEL_COL].values
        
        # 数据标准化 - 用A数据集拟合scaler
        X_A_scaled, X_target_scaled, self.scaler = fit_apply_scaler(X_A_raw, X_target_raw)
        
        # 设置A域训练数据
        self.X_train = X_A_scaled
        self.y_train = y_A
        self.X_train_raw = X_A_raw
        
        # 对目标域数据进行三分法划分：验证集 vs 保留测试集
        self.X_ext_val, self.X_ext_holdout, self.y_ext_val, self.y_ext_holdout = train_test_split(
            X_target_scaled, y_target,
            train_size=self.validation_split,
            stratify=y_target,
            random_state=self.random_state
        )
        
        # 同时划分原始数据（用于MMD域适应）
        self.X_ext_val_raw, self.X_ext_holdout_raw, _, _ = train_test_split(
            X_target_raw, y_target,
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
        """定义搜索空间 - 包含MMD方法选择、类条件选择和对应参数"""
        search_space = []
        
        if not self.use_fixed_model_params:
            # 1. 模型参数（如果不使用固定参数）
            if self.model_type == 'auto':
                # AutoTabPFN参数 - 使用更合理的范围
                search_space.extend([
                    Categorical([60, 120, 180, 240, 300], name='max_time'),  # 合理的时间范围
                    Categorical(['default', 'avoid_overfitting'], name='preset'),
                    Categorical(['accuracy', 'roc', 'f1'], name='ges_scoring'),
                    Categorical([15, 20, 25, 30], name='max_models'),
                    Integer(80, 120, name='n_repeats'),
                    Integer(20, 30, name='ges_n_iterations'),
                    Categorical([True, False], name='ignore_limits'),
                ])
            elif self.model_type == 'rf':
                # Random Forest参数
                search_space.extend([
                    Integer(50, 200, name='n_estimators'),
                    Integer(5, 15, name='max_depth'),
                    Integer(2, 10, name='min_samples_split'),
                    Integer(1, 5, name='min_samples_leaf'),
                    Real(0.5, 1.0, name='max_features'),
                    Categorical(['auto', 'balanced'], name='class_weight')
                ])
        
        # 2. MMD方法选择和类条件选择
        search_space.extend([
            Categorical(['linear', 'mean_std'], name='mmd_method_choice'),  # 选择MMD方法
            Categorical([True, False], name='use_class_conditional_choice'),  # 选择是否使用类条件MMD
        ])
        
        # 3. Linear MMD参数（当选择linear时使用）
        search_space.extend([
            Real(1e-5, 1e-2, prior='log-uniform', name='linear_lr'),
            Integer(100, 500, name='linear_n_epochs'),
            Integer(32, 128, name='linear_batch_size'),
            Real(1e-6, 1e-2, prior='log-uniform', name='linear_lambda_reg'),
            Real(0.1, 5.0, name='linear_gamma'),
            Categorical([True, False], name='linear_staged_training'),
            Categorical([True, False], name='linear_dynamic_gamma'),
        ])
        
        # 4. Mean-Std MMD参数（当选择mean_std时使用）
        search_space.extend([
            Real(1e-8, 1e-6, prior='log-uniform', name='mean_std_eps'),
        ])
        
        param_count = len(search_space)
        if self.use_fixed_model_params:
            logging.info(f"定义了{param_count}个超参数的搜索空间（使用固定模型参数）")
            logging.info(f"  MMD方法选择: linear, mean_std")
            logging.info(f"  类条件MMD选择: True/False")
            logging.info(f"  Linear MMD参数: 7个")
            logging.info(f"  Mean-Std MMD参数: 1个")
        else:
            model_param_count = 7 if self.model_type == 'auto' else 6
            mmd_param_count = param_count - model_param_count
            logging.info(f"定义了{param_count}个超参数的联合搜索空间")
            logging.info(f"  模型参数: {model_param_count}")
            logging.info(f"  MMD相关参数: {mmd_param_count}")
        
        return search_space
    
    def objective_function(self, params: List) -> float:
        """
        目标函数：评估给定超参数组合的性能（包含MMD域适应）
        
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
            
            # 分离模型参数和MMD相关参数
            model_params = {}
            mmd_method_choice = param_dict.get('mmd_method_choice', self.mmd_method)
            use_class_conditional_choice = param_dict.get('use_class_conditional_choice', self.use_class_conditional)
            
            # 根据选择的MMD方法提取对应参数
            mmd_params = {}
            if mmd_method_choice == 'linear':
                mmd_params = {
                    'lr': param_dict.get('linear_lr'),
                    'n_epochs': param_dict.get('linear_n_epochs'),
                    'batch_size': param_dict.get('linear_batch_size'),
                    'lambda_reg': param_dict.get('linear_lambda_reg'),
                    'gamma': param_dict.get('linear_gamma'),
                    'staged_training': param_dict.get('linear_staged_training'),
                    'dynamic_gamma': param_dict.get('linear_dynamic_gamma'),
                }
            elif mmd_method_choice == 'mean_std':
                mmd_params = {
                    'eps': param_dict.get('mean_std_eps'),
                }
            
            # 提取模型参数
            for key, value in param_dict.items():
                if not key.startswith(('mmd_method_choice', 'use_class_conditional_choice', 'linear_', 'mean_std_')):
                    model_params[key] = value
            
            # 如果使用固定模型参数，则覆盖搜索到的模型参数
            if self.use_fixed_model_params:
                model_params = self.fixed_model_params.get(self.model_type, {}).copy()
                logging.info(f"使用固定模型参数: {model_params}")
            else:
                logging.info(f"搜索模型参数: {model_params}")
            
            logging.info(f"选择的MMD方法: {mmd_method_choice}")
            logging.info(f"选择的类条件MMD: {use_class_conditional_choice}")
            logging.info(f"MMD参数: {mmd_params}")
            
            # 1. 创建模型
            if self.model_type == 'auto':
                if not AUTO_TABPFN_AVAILABLE:
                    raise ImportError("AutoTabPFN不可用，请安装tabpfn_extensions")
                
                # 构建phe_init_args参数
                phe_init_args = {
                    'max_models': model_params.get('max_models', 20),
                    'validation_method': 'cv',
                    'n_repeats': model_params.get('n_repeats', 100),
                    'n_folds': 5,
                    'ges_n_iterations': model_params.get('ges_n_iterations', 25)
                }
                
                model = AutoTabPFNClassifier(
                    max_time=model_params.get('max_time', 120),
                    preset=model_params.get('preset', 'default'),
                    ges_scoring_string=model_params.get('ges_scoring', 'roc'),
                    device='cuda',
                    random_state=self.random_state,
                    ignore_pretraining_limits=model_params.get('ignore_limits', False),
                    categorical_feature_indices=self.categorical_indices if self.categorical_indices else None,
                    phe_init_args=phe_init_args
                )
            else:
                model_config = {
                    'categorical_feature_indices': self.categorical_indices if self.use_categorical else []
                }
                model_config.update(model_params)
                model = get_model(self.model_type, **model_config)
            
            # 2. 在A域训练集上训练模型
            logging.info(f"开始训练模型，训练集大小: {self.X_train.shape}")
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            logging.info(f"模型训练完成，耗时: {training_time:.2f}秒")
            
            # 3. 进行MMD域适应
            logging.info(f"开始MMD域适应，方法: {mmd_method_choice}")
            
            # 获取MMD方法的默认参数并更新
            default_mmd_params = MMD_METHODS.get(mmd_method_choice, {}).copy()
            default_mmd_params.update(mmd_params)
            
            # 修复numpy类型问题
            for key, value in default_mmd_params.items():
                if isinstance(value, np.integer):
                    default_mmd_params[key] = int(value)
                elif isinstance(value, np.floating):
                    default_mmd_params[key] = float(value)
                elif isinstance(value, np.bool_):
                    default_mmd_params[key] = bool(value)
            
            if use_class_conditional_choice:
                # 使用类条件MMD
                X_ext_val_adapted, adaptation_info = class_conditional_mmd_transform(
                    self.X_train_raw, self.y_train, self.X_ext_val_raw,
                    method=mmd_method_choice,
                    cat_idx=self.categorical_indices,
                    **default_mmd_params
                )
            else:
                # 使用标准MMD
                X_ext_val_adapted, adaptation_info = mmd_transform(
                    self.X_train_raw, self.X_ext_val_raw,
                    method=mmd_method_choice,
                    cat_idx=self.categorical_indices,
                    **default_mmd_params
                )
            
            logging.info(f"MMD域适应完成，MMD减少: {adaptation_info.get('reduction', 0):.2f}%")
            
            # 4. 在域适应后的验证集上预测
            logging.info(f"在域适应后的验证集上预测，验证集大小: {X_ext_val_adapted.shape}")
            y_pred_proba = model.predict_proba(X_ext_val_adapted)
            
            # 修复预测概率维度问题
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]  # 取正类概率
            elif y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 1:
                y_pred_proba = y_pred_proba.ravel()  # 展平为1D
            
            # 5. 计算验证集AUC
            val_auc_score = roc_auc_score(self.y_ext_val, y_pred_proba)
            logging.info(f"验证集AUC (域适应后): {val_auc_score:.4f}")
            
            # 6. 在保留测试集上评估（用于记录，不用于优化）
            if use_class_conditional_choice:
                X_ext_holdout_adapted, _ = class_conditional_mmd_transform(
                    self.X_train_raw, self.y_train, self.X_ext_holdout_raw,
                    method=mmd_method_choice,
                    cat_idx=self.categorical_indices,
                    **default_mmd_params
                )
            else:
                X_ext_holdout_adapted, _ = mmd_transform(
                    self.X_train_raw, self.X_ext_holdout_raw,
                    method=mmd_method_choice,
                    cat_idx=self.categorical_indices,
                    **default_mmd_params
                )
            
            y_test_pred_proba = model.predict_proba(X_ext_holdout_adapted)
            
            # 修复预测概率维度问题
            if y_test_pred_proba.ndim > 1 and y_test_pred_proba.shape[1] > 1:
                y_test_pred_proba = y_test_pred_proba[:, 1]
            elif y_test_pred_proba.ndim > 1 and y_test_pred_proba.shape[1] == 1:
                y_test_pred_proba = y_test_pred_proba.ravel()
            
            test_auc_score = roc_auc_score(self.y_ext_holdout, y_test_pred_proba)
            logging.info(f"保留测试集AUC (域适应后): {test_auc_score:.4f}")
            
            # 7. 记录试验结果
            trial_result = {
                'params': {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                          for k, v in param_dict.items()},
                'model_params': {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                               for k, v in model_params.items()},
                'mmd_method_choice': mmd_method_choice,
                'use_class_conditional_choice': use_class_conditional_choice,
                'mmd_params': {k: (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                             for k, v in mmd_params.items()},
                'validation_auc': float(val_auc_score),
                'test_auc': float(test_auc_score),
                'training_time': float(training_time),
                'mmd_reduction': float(adaptation_info.get('reduction', 0)),
                'trial_id': len(self.optimization_results),
                'use_fixed_model_params': self.use_fixed_model_params
            }
            self.optimization_results.append(trial_result)
            
            # 如果测试集AUC > 0.7，保存配置
            if test_auc_score > 0.7:
                good_config = trial_result.copy()
                self.good_configs.append(good_config)
                logging.info(f"★ 优秀配置 {len(self.good_configs)}: 测试集AUC={test_auc_score:.4f} > 0.7，已保存配置")
            
            logging.info(f"试验 {len(self.optimization_results)}: 验证集AUC={val_auc_score:.4f}, 测试集AUC={test_auc_score:.4f}")
            
            # 返回负验证集AUC (因为gp_minimize最小化目标函数)
            return float(-val_auc_score)
            
        except Exception as e:
            logging.error(f"目标函数评估失败: {e}")
            import traceback
            logging.error(f"详细错误信息: {traceback.format_exc()}")
            # 返回一个较大的值，表示失败
            return 1.0
    
    def run_optimization(self) -> Dict[str, Any]:
        """运行贝叶斯优化"""
        logging.info("开始贝叶斯MMD优化...")
        
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
            n_initial_points=max(10, self.n_calls // 5)  # 20%的初始随机点数
        )
        
        # 提取最佳参数
        self.best_params = {}
        for i, dim in enumerate(search_space):
            self.best_params[dim.name] = result.x[i]
        
        self.best_score = -result.fun  # 转换回正AUC
        
        logging.info("贝叶斯MMD优化完成!")
        logging.info(f"最佳验证集AUC: {self.best_score:.4f}")
        logging.info(f"最佳参数: {self.best_params}")
        
        # 分离最佳参数
        best_model_params = {}
        best_mmd_params = {}
        
        for key, value in self.best_params.items():
            if key.startswith('mmd_'):
                best_mmd_params[key[4:]] = value  # 去掉'mmd_'前缀
            else:
                best_model_params[key] = value
        
        logging.info(f"最佳模型参数: {best_model_params}")
        logging.info(f"最佳MMD参数: {best_mmd_params}")
        
        # 输出优秀配置汇总
        if self.good_configs:
            logging.info(f"\n发现 {len(self.good_configs)} 个优秀配置 (测试集AUC > 0.7):")
            for i, config in enumerate(self.good_configs):
                logging.info(f"  配置 {i+1}: 验证集AUC={config['validation_auc']:.4f}, 测试集AUC={config['test_auc']:.4f}")
        else:
            logging.info("未发现测试集AUC > 0.7的配置")
        
        return {
            'best_params': self.best_params,
            'best_model_params': best_model_params,
            'best_mmd_params': best_mmd_params,
            'best_validation_auc': self.best_score,
            'total_trials': len(self.optimization_results),
            'good_configs': self.good_configs,
            'optimization_history': self.optimization_results
        }
    
    def evaluate_final_model(self) -> Dict[str, Any]:
        """使用最佳参数评估最终模型"""
        logging.info("使用最佳参数评估最终模型...")
        
        if self.best_params is None:
            raise ValueError("请先运行优化")
        
        # 分离参数并获取最佳选择
        model_params = {}
        mmd_method_choice = self.best_params.get('mmd_method_choice', self.mmd_method)
        use_class_conditional_choice = self.best_params.get('use_class_conditional_choice', self.use_class_conditional)
        
        # 根据选择的MMD方法提取对应参数
        mmd_params = {}
        if mmd_method_choice == 'linear':
            mmd_params = {
                'lr': self.best_params.get('linear_lr'),
                'n_epochs': self.best_params.get('linear_n_epochs'),
                'batch_size': self.best_params.get('linear_batch_size'),
                'lambda_reg': self.best_params.get('linear_lambda_reg'),
                'gamma': self.best_params.get('linear_gamma'),
                'staged_training': self.best_params.get('linear_staged_training'),
                'dynamic_gamma': self.best_params.get('linear_dynamic_gamma'),
            }
        elif mmd_method_choice == 'mean_std':
            mmd_params = {
                'eps': self.best_params.get('mean_std_eps'),
            }
        
        # 提取模型参数
        for key, value in self.best_params.items():
            if not key.startswith(('mmd_method_choice', 'use_class_conditional_choice', 'linear_', 'mean_std_')):
                model_params[key] = value
        
        # 如果使用固定模型参数，则覆盖
        if self.use_fixed_model_params:
            model_params = self.fixed_model_params.get(self.model_type, {}).copy()
        
        logging.info(f"最终评估使用的MMD方法: {mmd_method_choice}")
        logging.info(f"最终评估使用的类条件MMD: {use_class_conditional_choice}")
        logging.info(f"最终评估使用的MMD参数: {mmd_params}")
        
        # 创建最终模型
        if self.model_type == 'auto':
            phe_init_args = {
                'max_models': model_params.get('max_models', 20),
                'validation_method': 'cv',
                'n_repeats': model_params.get('n_repeats', 100),
                'n_folds': 5,
                'ges_n_iterations': model_params.get('ges_n_iterations', 25)
            }
            
            self.final_model = AutoTabPFNClassifier(
                max_time=model_params.get('max_time', 120),
                preset=model_params.get('preset', 'default'),
                ges_scoring_string=model_params.get('ges_scoring', 'roc'),
                device='cuda',
                random_state=self.random_state,
                ignore_pretraining_limits=model_params.get('ignore_limits', False),
                categorical_feature_indices=self.categorical_indices if self.categorical_indices else None,
                phe_init_args=phe_init_args
            )
        else:
            model_config = {
                'categorical_feature_indices': self.categorical_indices if self.use_categorical else []
            }
            model_config.update(model_params)
            self.final_model = get_model(self.model_type, **model_config)
        
        # 训练最终模型
        self.final_model.fit(self.X_train, self.y_train)
        
        # 获取MMD参数
        default_mmd_params = MMD_METHODS.get(mmd_method_choice, {}).copy()
        default_mmd_params.update(mmd_params)
        
        # 修复numpy类型问题
        for key, value in default_mmd_params.items():
            if isinstance(value, np.integer):
                default_mmd_params[key] = int(value)
            elif isinstance(value, np.floating):
                default_mmd_params[key] = float(value)
            elif isinstance(value, np.bool_):
                default_mmd_params[key] = bool(value)
        
        # 在验证集上评估
        if use_class_conditional_choice:
            X_ext_val_adapted, _ = class_conditional_mmd_transform(
                self.X_train_raw, self.y_train, self.X_ext_val_raw,
                method=mmd_method_choice,
                cat_idx=self.categorical_indices,
                **default_mmd_params
            )
        else:
            X_ext_val_adapted, _ = mmd_transform(
                self.X_train_raw, self.X_ext_val_raw,
                method=mmd_method_choice,
                cat_idx=self.categorical_indices,
                **default_mmd_params
            )
        
        # 修复预测和概率预测
        y_val_pred = self.final_model.predict(X_ext_val_adapted)
        y_val_pred_proba = self.final_model.predict_proba(X_ext_val_adapted)
        
        # 修复预测概率维度问题
        if y_val_pred_proba.ndim > 1 and y_val_pred_proba.shape[1] > 1:
            y_val_pred_proba_1d = y_val_pred_proba[:, 1]
        elif y_val_pred_proba.ndim > 1 and y_val_pred_proba.shape[1] == 1:
            y_val_pred_proba_1d = y_val_pred_proba.ravel()
        else:
            y_val_pred_proba_1d = y_val_pred_proba
        
        val_metrics = evaluate_metrics(
            self.y_ext_val,
            y_val_pred,
            y_val_pred_proba_1d
        )
        
        # 在保留测试集上评估
        if use_class_conditional_choice:
            X_ext_holdout_adapted, _ = class_conditional_mmd_transform(
                self.X_train_raw, self.y_train, self.X_ext_holdout_raw,
                method=mmd_method_choice,
                cat_idx=self.categorical_indices,
                **default_mmd_params
            )
        else:
            X_ext_holdout_adapted, _ = mmd_transform(
                self.X_train_raw, self.X_ext_holdout_raw,
                method=mmd_method_choice,
                cat_idx=self.categorical_indices,
                **default_mmd_params
            )
        
        # 修复预测和概率预测
        y_holdout_pred = self.final_model.predict(X_ext_holdout_adapted)
        y_holdout_pred_proba = self.final_model.predict_proba(X_ext_holdout_adapted)
        
        # 修复预测概率维度问题
        if y_holdout_pred_proba.ndim > 1 and y_holdout_pred_proba.shape[1] > 1:
            y_holdout_pred_proba_1d = y_holdout_pred_proba[:, 1]
        elif y_holdout_pred_proba.ndim > 1 and y_holdout_pred_proba.shape[1] == 1:
            y_holdout_pred_proba_1d = y_holdout_pred_proba.ravel()
        else:
            y_holdout_pred_proba_1d = y_holdout_pred_proba
        
        holdout_metrics = evaluate_metrics(
            self.y_ext_holdout,
            y_holdout_pred,
            y_holdout_pred_proba_1d
        )
        
        # 计算泛化差距
        generalization_gap = {
            'auc_gap': val_metrics['auc'] - holdout_metrics['auc'],
            'f1_gap': val_metrics['f1'] - holdout_metrics['f1'],
            'acc_gap': val_metrics['acc'] - holdout_metrics['acc']
        }
        
        return {
            'validation_performance': val_metrics,
            'holdout_performance': holdout_metrics,
            'generalization_gap': generalization_gap,
            'best_model_params': model_params,
            'best_mmd_params': mmd_params
        }
    
    def save_results(self, optimization_results: Dict[str, Any], 
                    final_results: Dict[str, Any],
                    source_cv_results: Optional[Dict[str, Any]] = None,
                    cross_domain_performance: Optional[Dict[str, Any]] = None) -> None:
        """保存优化结果"""
        # 保存优化历史
        history_file = os.path.join(self.save_path, 'bayesian_mmd_optimization_history.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(optimization_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # 保存最终评估结果
        final_file = os.path.join(self.save_path, 'final_mmd_evaluation.json')
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # 保存A域CV基准结果（如果有）
        if source_cv_results:
            source_cv_file = os.path.join(self.save_path, 'source_domain_cv_baseline.json')
            with open(source_cv_file, 'w', encoding='utf-8') as f:
                json.dump(source_cv_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # 保存跨域性能比较（如果有）
        if cross_domain_performance:
            cross_domain_file = os.path.join(self.save_path, 'cross_domain_performance_comparison.json')
            with open(cross_domain_file, 'w', encoding='utf-8') as f:
                json.dump(cross_domain_performance, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # 保存实验配置
        config = {
            'model_type': self.model_type,
            'feature_type': self.feature_type,
            'mmd_method': self.mmd_method,
            'use_class_conditional': self.use_class_conditional,
            'use_categorical': self.use_categorical,
            'validation_split': self.validation_split,
            'n_calls': self.n_calls,
            'random_state': self.random_state,
            'target_domain': self.target_domain,
            'evaluate_source_cv': self.evaluate_source_cv,
            'features': self.features,
            'categorical_indices': self.categorical_indices
        }
        
        config_file = os.path.join(self.save_path, 'experiment_config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        logging.info(f"结果已保存到: {self.save_path}")
        if source_cv_results:
            logging.info(f"  - source_domain_cv_baseline.json: A域交叉验证基准")
        if cross_domain_performance:
            logging.info(f"  - cross_domain_performance_comparison.json: 跨域性能比较")
        logging.info(f"  - bayesian_mmd_optimization_history.json: 贝叶斯优化历史")
        logging.info(f"  - final_mmd_evaluation.json: 最终评估结果")
        logging.info(f"  - experiment_config.json: 实验配置")

    def evaluate_source_domain_cv(self, cv_folds: int = 5) -> Dict[str, Any]:
        """
        评估A域内的交叉验证性能作为基准
        
        参数:
        - cv_folds: 交叉验证折数
        
        返回:
        - A域CV性能指标字典
        """
        logging.info(f"在A域进行{cv_folds}折交叉验证...")
        
        # 使用固定的最佳模型参数
        if self.use_fixed_model_params and self.model_type in self.fixed_model_params:
            model_params = self.fixed_model_params[self.model_type].copy()
        else:
            # 如果没有固定参数，使用默认配置
            from ..config.settings import get_model_config
            model_params = get_model_config(self.model_type, categorical_feature_indices=self.categorical_indices)
        
        logging.info(f"使用模型参数: {model_params}")
        
        # 初始化交叉验证
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # 存储每折的结果
        cv_scores = []
        auc_scores = []
        f1_scores = []
        acc_scores = []
        
        # 执行交叉验证
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
            logging.info(f"执行第 {fold_idx + 1}/{cv_folds} 折...")
            
            # 划分训练和验证集
            X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
            y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
            
            # 创建并训练模型
            model = get_model(self.model_type, **model_params)
            
            start_time = time.time()
            model.fit(X_fold_train, y_fold_train)
            train_time = time.time() - start_time
            
            # 预测并评估
            y_pred = model.predict(X_fold_val)
            y_pred_proba = model.predict_proba(X_fold_val)
            
            # 修复预测概率维度问题
            if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]  # 取正类概率
            elif y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 1:
                y_pred_proba = y_pred_proba.ravel()  # 展平为1D
            
            # 计算指标
            auc = roc_auc_score(y_fold_val, y_pred_proba)
            f1 = f1_score(y_fold_val, y_pred)
            acc = accuracy_score(y_fold_val, y_pred)
            
            auc_scores.append(auc)
            f1_scores.append(f1)
            acc_scores.append(acc)
            
            cv_scores.append({
                'fold': fold_idx + 1,
                'auc': auc,
                'f1': f1,
                'accuracy': acc,
                'train_time': train_time,
                'train_size': len(y_fold_train),
                'val_size': len(y_fold_val)
            })
            
            logging.info(f"第 {fold_idx + 1} 折结果: AUC={auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
        
        # 计算总体统计
        results = {
            'cv_folds': cv_folds,
            'individual_folds': cv_scores,
            'mean_scores': {
                'auc': np.mean(auc_scores),
                'f1': np.mean(f1_scores),
                'accuracy': np.mean(acc_scores)
            },
            'std_scores': {
                'auc': np.std(auc_scores),
                'f1': np.std(f1_scores),
                'accuracy': np.std(acc_scores)
            },
            'model_params': model_params,
            'total_samples': len(self.y_train),
            'label_distribution': {
                'class_0': int(np.sum(self.y_train == 0)),
                'class_1': int(np.sum(self.y_train == 1))
            }
        }
        
        # 打印总体结果
        logging.info("=" * 50)
        logging.info(f"A域 {cv_folds}折交叉验证结果:")
        logging.info("=" * 50)
        logging.info(f"AUC: {results['mean_scores']['auc']:.4f} ± {results['std_scores']['auc']:.4f}")
        logging.info(f"F1:  {results['mean_scores']['f1']:.4f} ± {results['std_scores']['f1']:.4f}")
        logging.info(f"Acc: {results['mean_scores']['accuracy']:.4f} ± {results['std_scores']['accuracy']:.4f}")
        logging.info(f"样本数: {results['total_samples']} (类别0: {results['label_distribution']['class_0']}, 类别1: {results['label_distribution']['class_1']})")
        logging.info("=" * 50)
        
        return results
    
    def run_complete_optimization(self) -> Dict[str, Any]:
        """运行完整的贝叶斯MMD优化流程"""
        start_time = time.time()
        
        # 1. 加载数据
        self.load_and_prepare_data()
        
        # 2. 可选：评估A域交叉验证基准
        source_cv_results = None
        cross_domain_performance = None
        
        if self.evaluate_source_cv:
            logging.info("步骤1: 评估A域交叉验证性能（基准）")
            source_cv_results = self.evaluate_source_domain_cv(cv_folds=5)
        
        # 3. 运行贝叶斯优化
        logging.info(f"步骤{'2' if self.evaluate_source_cv else '1'}: 运行贝叶斯MMD优化")
        optimization_results = self.run_optimization()
        
        # 4. 评估最终模型
        logging.info(f"步骤{'3' if self.evaluate_source_cv else '2'}: 使用最佳参数评估最终模型")
        final_results = self.evaluate_final_model()
        
        # 5. 如果评估了A域基准，计算跨域性能比较
        if self.evaluate_source_cv and source_cv_results:
            baseline_auc = source_cv_results['cv_results']['mean_scores']['auc']
            validation_auc = final_results['validation_performance']['auc']
            holdout_auc = final_results['holdout_performance']['auc']
            
            cross_domain_performance = {
                'source_domain_cv_auc': baseline_auc,
                'target_validation_auc': validation_auc,
                'target_holdout_auc': holdout_auc,
                'cross_domain_gap_validation': validation_auc - baseline_auc,
                'cross_domain_gap_holdout': holdout_auc - baseline_auc,
                'cross_domain_improvement_validation': ((validation_auc / baseline_auc) - 1) * 100 if baseline_auc > 0 else 0,
                'cross_domain_improvement_holdout': ((holdout_auc / baseline_auc) - 1) * 100 if baseline_auc > 0 else 0
            }
            
            # 打印跨域性能比较
            logging.info("=" * 60)
            logging.info("跨域性能比较:")
            logging.info("=" * 60)
            logging.info(f"A域CV基准AUC:     {baseline_auc:.4f}")
            logging.info(f"B域验证集AUC:     {validation_auc:.4f} (差距: {cross_domain_performance['cross_domain_gap_validation']:+.4f})")
            logging.info(f"B域测试集AUC:     {holdout_auc:.4f} (差距: {cross_domain_performance['cross_domain_gap_holdout']:+.4f})")
            logging.info(f"跨域改进-验证集:   {cross_domain_performance['cross_domain_improvement_validation']:+.2f}%")
            logging.info(f"跨域改进-测试集:   {cross_domain_performance['cross_domain_improvement_holdout']:+.2f}%")
        
        total_time = time.time() - start_time
        
        # 6. 保存结果
        self.save_results(optimization_results, final_results, source_cv_results, cross_domain_performance)
        
        # 7. 组织完整结果
        complete_results = {
            'optimization_results': optimization_results,
            'final_results': final_results,
            'experiment_config': {
                'model_type': self.model_type,
                'feature_type': self.feature_type,
                'mmd_method': self.mmd_method,
                'use_class_conditional': self.use_class_conditional,
                'target_domain': self.target_domain,
                'evaluate_source_cv': self.evaluate_source_cv
            },
            'experiment_info': {
                'total_time': total_time
            }
        }
        
        # 添加可选的A域基准和跨域比较结果
        if source_cv_results:
            complete_results['source_domain_cv'] = source_cv_results
        if cross_domain_performance:
            complete_results['cross_domain_performance'] = cross_domain_performance
        
        logging.info(f"贝叶斯MMD优化完成，总耗时: {total_time:.2f}秒")
        
        return complete_results

def run_bayesian_mmd_optimization(model_type: str = 'auto',
                                 feature_type: str = 'best7',
                                 mmd_method: str = 'linear',
                                 use_class_conditional: bool = False,
                                 use_categorical: bool = True,
                                 validation_split: float = 0.7,
                                 n_calls: int = 50,
                                 target_domain: str = 'B',
                                 save_path: str = './results_bayesian_mmd_optimization',
                                 use_fixed_model_params: bool = True,
                                 evaluate_source_cv: bool = False,
                                 **kwargs) -> Dict[str, Any]:
    """
    运行贝叶斯MMD优化的便捷函数
    
    参数:
    - model_type: 模型类型
    - feature_type: 特征类型
    - mmd_method: MMD方法
    - use_class_conditional: 是否使用类条件MMD
    - use_categorical: 是否使用类别特征
    - validation_split: 验证集比例 (0.7表示70%用于验证，30%用于holdout)
    - n_calls: 优化迭代次数
    - target_domain: 目标域选择
    - save_path: 保存路径
    - use_fixed_model_params: 是否使用固定的最佳模型参数（推荐True，只优化MMD参数）
    - evaluate_source_cv: 是否评估A域交叉验证基准（可选，默认False）
    - **kwargs: 其他参数
    
    返回:
    - Dict[str, Any]: 完整的优化结果
    """
    
    optimizer = BayesianMMDOptimizer(
        model_type=model_type,
        feature_type=feature_type,
        mmd_method=mmd_method,
        use_class_conditional=use_class_conditional,
        use_categorical=use_categorical,
        validation_split=validation_split,
        n_calls=n_calls,
        target_domain=target_domain,
        save_path=save_path,
        use_fixed_model_params=use_fixed_model_params,
        evaluate_source_cv=evaluate_source_cv,
        **kwargs
    )
    
    return optimizer.run_complete_optimization() 