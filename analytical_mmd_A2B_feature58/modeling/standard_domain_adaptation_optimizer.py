"""
标准域适应优化器

实现符合域适应研究标准的实验设计：
1. 只使用源域数据进行模型训练和参数调优
2. 目标域数据仅用于最终评估
3. 通过源域内交叉验证选择最佳超参数
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
    DATA_PATHS, LABEL_COL, get_features_by_type, get_categorical_indices, MMD_METHODS, get_model_config
)
from ..data.loader import load_all_datasets
from ..preprocessing.scaler import fit_apply_scaler, apply_scaler
from ..preprocessing.mmd import mmd_transform
from ..preprocessing.class_conditional_mmd import class_conditional_mmd_transform
from ..preprocessing.coral import coral_transform, class_conditional_coral_transform, adaptive_coral_transform
from ..preprocessing.mean_variance_alignment import mean_variance_transform
from ..modeling.model_selector import get_model
from ..metrics.evaluation import evaluate_metrics
from ..modeling.baseline_models import get_baseline_model, evaluate_baseline_models

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)

class StandardDomainAdaptationOptimizer:
    """
    标准域适应优化器
    
    严格遵循域适应研究的实验设计原则：
    1. 源域数据用于训练和参数调优
    2. 目标域数据仅用于最终评估
    3. 通过源域内交叉验证选择最佳参数
    """
    
    def __init__(self, 
                 model_type: str = 'auto',
                 feature_type: str = 'best7',
                 mmd_method: str = 'linear',
                 use_class_conditional: bool = False,
                 use_categorical: bool = True,
                 source_val_split: float = 0.2,
                 cv_folds: int = 5,
                 n_calls: int = 50,
                 random_state: int = 42,
                 target_domain: str = 'B',
                 save_path: str = './results_standard_domain_adaptation',
                 use_source_cv_for_mmd_tuning: bool = True):
        """
        初始化标准域适应优化器
        
        参数:
        - model_type: 模型类型 ('auto', 'base', 'rf', 'tuned')
        - feature_type: 特征类型 ('all', 'best7')
        - mmd_method: MMD方法 ('linear', 'kpca', 'mean_std')
        - use_class_conditional: 是否使用类条件MMD
        - use_categorical: 是否使用类别特征
        - source_val_split: 源域验证集比例（用于最终模型选择）
        - cv_folds: 交叉验证折数（用于参数调优）
        - n_calls: 贝叶斯优化迭代次数
        - random_state: 随机种子
        - target_domain: 目标域选择 ('B' 或 'C')
        - save_path: 结果保存路径
        - use_source_cv_for_mmd_tuning: 是否使用源域CV调优MMD参数
        """
        self.model_type = model_type
        self.feature_type = feature_type
        self.mmd_method = mmd_method
        self.use_class_conditional = use_class_conditional
        self.use_categorical = use_categorical
        self.source_val_split = source_val_split
        self.cv_folds = cv_folds
        self.n_calls = n_calls
        self.random_state = random_state
        self.target_domain = target_domain
        self.save_path = save_path
        self.use_source_cv_for_mmd_tuning = use_source_cv_for_mmd_tuning
        
        # 获取特征和类别索引
        self.features = get_features_by_type(feature_type)
        self.categorical_indices = get_categorical_indices(feature_type) if use_categorical else []
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 初始化数据存储
        self.X_source_train = None
        self.y_source_train = None
        self.X_source_val = None
        self.y_source_val = None
        self.X_target_test = None
        self.y_target_test = None
        self.X_source_train_raw = None
        self.X_source_val_raw = None
        self.X_target_test_raw = None
        self.scaler = None
        
        # 优化结果存储
        self.optimization_results = []
        self.best_params = None
        self.best_score = None
        self.final_model = None
        
        logging.info(f"初始化标准域适应优化器:")
        logging.info(f"  模型类型: {model_type}")
        logging.info(f"  特征类型: {feature_type} ({len(self.features)}个特征)")
        logging.info(f"  MMD方法: {mmd_method}")
        logging.info(f"  类条件MMD: {use_class_conditional}")
        logging.info(f"  使用类别特征: {use_categorical}")
        logging.info(f"  源域验证集比例: {source_val_split}")
        logging.info(f"  交叉验证折数: {cv_folds}")
        logging.info(f"  优化迭代次数: {n_calls}")
        logging.info(f"  目标域: {target_domain}")
        logging.info(f"  使用源域CV调优MMD: {use_source_cv_for_mmd_tuning}")

    def _apply_domain_adaptation(self, X_source: np.ndarray, y_source: np.ndarray, 
                                X_target: np.ndarray, adaptation_params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        应用域适应方法的通用函数
        
        参数:
        - X_source: 源域特征
        - y_source: 源域标签
        - X_target: 目标域特征
        - adaptation_params: 域适应参数
        
        返回:
        - X_target_adapted: 适应后的目标域特征
        - adaptation_info: 适应信息
        """
        if self.mmd_method in ['linear', 'kpca', 'mean_std']:
            # MMD方法
            if self.use_class_conditional:
                return class_conditional_mmd_transform(
                    X_source, y_source, X_target,
                    method=self.mmd_method,
                    cat_idx=self.categorical_indices,
                    **adaptation_params
                )
            else:
                return mmd_transform(
                    X_source, X_target,
                    method=self.mmd_method,
                    cat_idx=self.categorical_indices,
                    **adaptation_params
                )
        
        elif self.mmd_method == 'coral':
            # CORAL方法 - 过滤掉不相关的MMD参数
            coral_params = {}
            if 'regularization' in adaptation_params:
                coral_params['regularization'] = adaptation_params['regularization']
            if 'standardize' in adaptation_params:
                coral_params['standardize'] = adaptation_params['standardize']
            
            if self.use_class_conditional:
                # 需要伪标签，这里使用源域模型预测
                temp_model = get_model(self.model_type, categorical_feature_indices=self.categorical_indices)
                temp_model.fit(X_source, y_source)
                y_target_pseudo = temp_model.predict(X_target)
                
                return class_conditional_coral_transform(
                    X_source, y_source, X_target, y_target_pseudo,
                    **coral_params
                )
            else:
                return coral_transform(X_source, X_target, **coral_params)
        
        elif self.mmd_method == 'adaptive_coral':
            # 自适应CORAL方法 - 过滤参数
            coral_params = {}
            if 'regularization_range' in adaptation_params:
                coral_params['regularization_range'] = adaptation_params['regularization_range']
            if 'n_trials' in adaptation_params:
                coral_params['n_trials'] = adaptation_params['n_trials']
            if 'standardize' in adaptation_params:
                coral_params['standardize'] = adaptation_params['standardize']
            return adaptive_coral_transform(X_source, X_target, **coral_params)
        
        elif self.mmd_method in ['mean_variance', 'adaptive_mean_variance']:
            # Mean-Variance Alignment方法 - 过滤参数
            mv_params = {}
            if 'align_mean' in adaptation_params:
                mv_params['align_mean'] = adaptation_params['align_mean']
            if 'align_variance' in adaptation_params:
                mv_params['align_variance'] = adaptation_params['align_variance']
            if 'standardize' in adaptation_params:
                mv_params['standardize'] = adaptation_params['standardize']
            
            if self.mmd_method == 'adaptive_mean_variance':
                from ..preprocessing.mean_variance_alignment import adaptive_mean_variance_transform
                return adaptive_mean_variance_transform(X_source, X_target, **mv_params)
            else:
                return mean_variance_transform(X_source, X_target, **mv_params)
        
        elif self.mmd_method in ['tca', 'adaptive_tca']:
            # TCA方法 - 过滤参数
            from ..preprocessing.tca import tca_transform, adaptive_tca_transform
            
            tca_params = {}
            if 'subspace_dim' in adaptation_params:
                tca_params['subspace_dim'] = adaptation_params['subspace_dim']
            if 'gamma' in adaptation_params:
                tca_params['gamma'] = adaptation_params['gamma']
            if 'regularization' in adaptation_params:
                tca_params['regularization'] = adaptation_params['regularization']
            if 'standardize' in adaptation_params:
                tca_params['standardize'] = adaptation_params['standardize']
            
            if self.mmd_method == 'adaptive_tca':
                # 自适应TCA的特殊参数
                if 'subspace_dim_range' in adaptation_params:
                    tca_params['subspace_dim_range'] = adaptation_params['subspace_dim_range']
                if 'gamma_range' in adaptation_params:
                    tca_params['gamma_range'] = adaptation_params['gamma_range']
                if 'n_trials' in adaptation_params:
                    tca_params['n_trials'] = adaptation_params['n_trials']
                X_s_trans, X_t_trans, info = adaptive_tca_transform(X_source, X_target, **tca_params)
                return X_t_trans, info  # TCA返回变换后的目标域特征
            else:
                X_s_trans, X_t_trans, info = tca_transform(X_source, X_target, **tca_params)
                return X_t_trans, info  # TCA返回变换后的目标域特征
        
        elif self.mmd_method in ['jda', 'adaptive_jda']:
            # JDA方法 - 过滤参数
            from ..preprocessing.jda import jda_transform, adaptive_jda_transform
            
            jda_params = {}
            if 'subspace_dim' in adaptation_params:
                jda_params['subspace_dim'] = adaptation_params['subspace_dim']
            if 'gamma' in adaptation_params:
                jda_params['gamma'] = adaptation_params['gamma']
            if 'mu' in adaptation_params:
                jda_params['mu'] = adaptation_params['mu']
            if 'max_iterations' in adaptation_params:
                jda_params['max_iterations'] = adaptation_params['max_iterations']
            if 'regularization' in adaptation_params:
                jda_params['regularization'] = adaptation_params['regularization']
            if 'standardize' in adaptation_params:
                jda_params['standardize'] = adaptation_params['standardize']
            
            if self.mmd_method == 'adaptive_jda':
                # 自适应JDA的特殊参数
                if 'subspace_dim_range' in adaptation_params:
                    jda_params['subspace_dim_range'] = adaptation_params['subspace_dim_range']
                if 'gamma_range' in adaptation_params:
                    jda_params['gamma_range'] = adaptation_params['gamma_range']
                if 'mu_range' in adaptation_params:
                    jda_params['mu_range'] = adaptation_params['mu_range']
                if 'n_trials' in adaptation_params:
                    jda_params['n_trials'] = adaptation_params['n_trials']
                X_s_trans, X_t_trans, info = adaptive_jda_transform(X_source, y_source, X_target, **jda_params)
                return X_t_trans, info  # JDA返回变换后的目标域特征
            else:
                X_s_trans, X_t_trans, info = jda_transform(X_source, y_source, X_target, **jda_params)
                return X_t_trans, info  # JDA返回变换后的目标域特征
        
        else:
            raise ValueError(f"不支持的域适应方法: {self.mmd_method}")

    def load_and_prepare_data(self) -> None:
        """加载并准备数据，严格分离源域和目标域"""
        logging.info("加载和准备数据...")
        
        # 加载数据集A (源域)
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
        
        # 数据标准化 - 用A数据集拟合scaler，只对非分类特征进行缩放
        X_A_scaled, X_target_scaled, self.scaler = fit_apply_scaler(
            X_A_raw, X_target_raw, categorical_indices=self.categorical_indices
        )
        
        # 源域数据划分：训练集 vs 验证集
        self.X_source_train, self.X_source_val, self.y_source_train, self.y_source_val = train_test_split(
            X_A_scaled, y_A,
            test_size=self.source_val_split,
            stratify=y_A,
            random_state=self.random_state
        )
        
        # 同时划分原始数据（用于MMD域适应）
        self.X_source_train_raw, self.X_source_val_raw, _, _ = train_test_split(
            X_A_raw, y_A,
            test_size=self.source_val_split,
            stratify=y_A,
            random_state=self.random_state
        )
        
        # 目标域数据：仅用于最终测试
        self.X_target_test = X_target_scaled
        self.y_target_test = y_target
        self.X_target_test_raw = X_target_raw
        
        # 打印数据信息
        logging.info(f"数据划分完成:")
        logging.info(f"  源域训练集: {self.X_source_train.shape[0]} 样本")
        logging.info(f"  源域验证集: {self.X_source_val.shape[0]} 样本")
        logging.info(f"  目标域测试集: {self.X_target_test.shape[0]} 样本")
        logging.info(f"  源域训练集标签分布: {np.bincount(self.y_source_train.astype(int))}")
        logging.info(f"  源域验证集标签分布: {np.bincount(self.y_source_val.astype(int))}")
        logging.info(f"  目标域测试集标签分布: {np.bincount(self.y_target_test.astype(int))}")

    def define_search_space(self) -> List:
        """定义超参数搜索空间"""
        search_space = []
        
        # 模型参数空间 - 基于best_model_summary.txt中的最佳参数调整
        if self.model_type == 'auto':
            search_space.extend([
                # 训练时间 - 以30秒为中心扩展范围
                Categorical([15, 30, 45, 60], name='max_time'),
                
                # 预设配置 - 保持default为主，增加avoid_overfitting选项
                Categorical(['default', 'avoid_overfitting'], name='preset'),
                
                # 评分指标 - 以accuracy为主，保留其他选项
                Categorical(['accuracy', 'roc', 'f1'], name='ges_scoring'),
                
                # 模型数量 - 以15为中心调整范围
                Categorical([10, 15, 20, 25], name='max_models'),
                
                # 验证方法 - 固定为CV，与最佳参数一致
                Categorical(['cv'], name='validation_method'),
                
                # 重复次数 - 以100为中心，缩小范围提高稳定性
                Integer(80, 120, name='n_repeats'),
                
                # 折数 - 以10折为主，增加5折选项
                Categorical([5, 10], name='n_folds'),
                
                # holdout_fraction - 虽然使用CV，但保留此参数以备用
                Categorical([0.3, 0.4, 0.5], name='holdout_fraction'),
                
                # GES迭代次数 - 以20为中心调整
                Integer(15, 25, name='ges_n_iterations'),
                
                # 是否忽略预训练限制 - 保持False为主
                Categorical([False, True], name='ignore_limits'),
            ])
        elif self.model_type == 'rf':
            search_space.extend([
                Integer(50, 500, name='n_estimators'),
                Integer(1, 20, name='max_depth'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 10, name='min_samples_leaf'),
                Real(0.1, 1.0, name='max_features'),
                Categorical(['auto', 'balanced'], name='class_weight'),
            ])
        
        # MMD参数空间（如果启用）
        if self.use_source_cv_for_mmd_tuning:
            if self.mmd_method == 'linear':
                search_space.extend([
                    Real(0.001, 0.1, name='mmd_lr'),
                    Integer(50, 500, name='mmd_n_epochs'),
                    Integer(16, 128, name='mmd_batch_size'),
                    Real(0.001, 0.1, name='mmd_lambda_reg'),
                    Real(0.1, 10.0, name='mmd_gamma'),
                    Categorical([True, False], name='mmd_staged_training'),
                    Categorical([True, False], name='mmd_dynamic_gamma'),
                ])
            elif self.mmd_method == 'mean_std':
                search_space.extend([
                    Real(1e-8, 1e-4, name='mmd_eps'),
                ])
        
        return search_space

    def objective_function(self, params: List) -> float:
        """
        目标函数：使用源域交叉验证评估参数组合
        
        关键：完全不使用目标域数据进行参数调优
        """
        try:
            # 将参数列表转换为字典
            search_space = self.define_search_space()
            param_dict = {dim.name: params[i] for i, dim in enumerate(search_space)}
            
            logging.info(f"开始评估参数组合: {param_dict}")
            
            # 分离模型参数和MMD参数
            model_params = {}
            mmd_params = {}
            
            for key, value in param_dict.items():
                if key.startswith('mmd_'):
                    mmd_params[key[4:]] = value  # 去掉'mmd_'前缀
                else:
                    model_params[key] = value
            
            # 使用源域交叉验证评估
            cv_scores = []
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_source_train, self.y_source_train)):
                logging.info(f"  交叉验证 Fold {fold + 1}/{self.cv_folds}")
                
                # 获取当前fold的数据
                X_fold_train = self.X_source_train[train_idx]
                y_fold_train = self.y_source_train[train_idx]
                X_fold_val = self.X_source_train[val_idx]
                y_fold_val = self.y_source_train[val_idx]
                
                X_fold_train_raw = self.X_source_train_raw[train_idx]
                X_fold_val_raw = self.X_source_train_raw[val_idx]
                
                # 创建并训练模型
                if self.model_type == 'auto' and AUTO_TABPFN_AVAILABLE:
                    # 构建phe_init_args参数
                    phe_init_args = {
                        'max_models': model_params.get('max_models', 15),
                        'validation_method': model_params.get('validation_method', 'cv'),
                        'n_repeats': model_params.get('n_repeats', 100),
                        'n_folds': model_params.get('n_folds', 5),
                        'holdout_fraction': model_params.get('holdout_fraction', 0.4),
                        'ges_n_iterations': model_params.get('ges_n_iterations', 20)
                    }
                    
                    # 创建AutoTabPFN模型
                    model = AutoTabPFNClassifier(
                        max_time=model_params.get('max_time', 30),
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
                
                model.fit(X_fold_train, y_fold_train)
                
                # 如果启用MMD调优，在源域内模拟域适应
                if self.use_source_cv_for_mmd_tuning and mmd_params:
                    # 获取MMD方法的默认参数并更新
                    default_mmd_params = MMD_METHODS.get(self.mmd_method, {}).copy()
                    default_mmd_params.update(mmd_params)
                    
                    # 修复numpy类型问题
                    for key, value in default_mmd_params.items():
                        if isinstance(value, np.integer):
                            default_mmd_params[key] = int(value)
                        elif isinstance(value, np.floating):
                            default_mmd_params[key] = float(value)
                        elif isinstance(value, np.bool_):
                            default_mmd_params[key] = bool(value)
                    
                    # 在源域内进行域适应（模拟跨域场景）
                    X_fold_val_adapted, _ = self._apply_domain_adaptation(
                        X_fold_train_raw, y_fold_train, X_fold_val_raw,
                        default_mmd_params
                    )
                    
                    # 重新标准化适应后的数据
                    fold_scaler = self.scaler  # 使用全局scaler保持一致性
                    X_fold_val_final = fold_scaler.transform(X_fold_val_adapted)
                else:
                    X_fold_val_final = X_fold_val
                
                # 预测并计算AUC
                y_pred_proba = model.predict_proba(X_fold_val_final)
                
                # 修复预测概率维度问题
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                elif y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 1:
                    y_pred_proba = y_pred_proba.ravel()
                
                fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
                cv_scores.append(fold_auc)
                logging.info(f"    Fold {fold + 1} AUC: {fold_auc:.4f}")
            
            # 计算平均CV分数
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            
            logging.info(f"  平均CV AUC: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
            
            # 记录试验结果
            trial_result = {
                'params': param_dict,
                'model_params': model_params,
                'mmd_params': mmd_params,
                'cv_scores': cv_scores,
                'mean_cv_score': mean_cv_score,
                'std_cv_score': std_cv_score,
                'timestamp': time.time()
            }
            self.optimization_results.append(trial_result)
            
            return -mean_cv_score  # 返回负值用于最小化
            
        except Exception as e:
            logging.error(f"目标函数评估失败: {e}")
            return 0.0  # 返回最差分数

    def run_optimization(self) -> Dict[str, Any]:
        """运行贝叶斯优化"""
        logging.info("开始贝叶斯优化...")
        
        search_space = self.define_search_space()
        
        @use_named_args(search_space)
        def objective(**params):
            param_values = [params[dim.name] for dim in search_space]
            return self.objective_function(param_values)
        
        # 运行贝叶斯优化
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=self.n_calls,
            random_state=self.random_state,
            acq_func='EI',
            n_initial_points=min(10, self.n_calls // 5)
        )
        
        # 提取最佳参数
        best_params_dict = {dim.name: result.x[i] for i, dim in enumerate(search_space)}
        
        # 分离最佳参数
        best_model_params = {}
        best_mmd_params = {}
        
        for key, value in best_params_dict.items():
            if key.startswith('mmd_'):
                best_mmd_params[key[4:]] = value
            else:
                best_model_params[key] = value
        
        self.best_params = {
            'model_params': best_model_params,
            'mmd_params': best_mmd_params
        }
        self.best_score = -result.fun
        
        logging.info(f"优化完成! 最佳CV AUC: {self.best_score:.4f}")
        logging.info(f"最佳模型参数: {best_model_params}")
        logging.info(f"最佳MMD参数: {best_mmd_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_result': result,
            'all_trials': self.optimization_results
        }

    def train_final_model(self) -> None:
        """使用最佳参数在完整源域数据上训练最终模型"""
        logging.info("使用最佳参数训练最终模型...")
        
        if self.best_params is None:
            raise ValueError("请先运行优化以获得最佳参数")
        
        # 使用完整的源域训练数据
        X_full_source = np.vstack([self.X_source_train, self.X_source_val])
        y_full_source = np.hstack([self.y_source_train, self.y_source_val])
        
        # 创建最终模型
        if self.model_type == 'auto' and AUTO_TABPFN_AVAILABLE:
            # 构建phe_init_args参数
            model_params = self.best_params['model_params']
            phe_init_args = {
                'max_models': model_params.get('max_models', 15),
                'validation_method': model_params.get('validation_method', 'cv'),
                'n_repeats': model_params.get('n_repeats', 100),
                'n_folds': model_params.get('n_folds', 5),
                'holdout_fraction': model_params.get('holdout_fraction', 0.4),
                'ges_n_iterations': model_params.get('ges_n_iterations', 20)
            }
            
            # 创建AutoTabPFN模型
            self.final_model = AutoTabPFNClassifier(
                max_time=model_params.get('max_time', 30),
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
            model_config.update(self.best_params['model_params'])
            self.final_model = get_model(self.model_type, **model_config)
        
        # 训练最终模型
        self.final_model.fit(X_full_source, y_full_source)
        logging.info("最终模型训练完成")

    def evaluate_baseline_models_performance(self) -> Dict[str, Any]:
        """评估基线模型（PKUPH和Mayo）性能"""
        logging.info("评估基线模型性能...")
        
        # 基线模型需要使用所有可用特征（58个特征），而不是受限于best7
        from ..config.settings import SELECTED_FEATURES
        
        # 加载完整特征的数据
        df_A = pd.read_excel(DATA_PATHS['A'])
        X_A_full = df_A[SELECTED_FEATURES].values
        y_A = df_A[LABEL_COL].values
        
        if self.target_domain == 'B':
            df_target = pd.read_excel(DATA_PATHS['B'])
        else:
            df_target = pd.read_excel(DATA_PATHS['C'])
        X_target_full = df_target[SELECTED_FEATURES].values
        y_target = df_target[LABEL_COL].values
        
        # 为基线模型创建完整特征的DataFrame
        X_full_source_df = pd.DataFrame(X_A_full, columns=SELECTED_FEATURES)
        X_target_df = pd.DataFrame(X_target_full, columns=SELECTED_FEATURES)
        
        # 源域验证集划分（使用相同的随机状态保持一致性）
        X_source_train_full, X_source_val_full, y_source_train_full, y_source_val_full = train_test_split(
            X_full_source_df, y_A,
            test_size=self.source_val_split,
            stratify=y_A,
            random_state=self.random_state
        )
        
        baseline_results = {}
        
        # 评估基线模型在源域和目标域上的性能
        for model_name in ['pkuph', 'mayo']:
            try:
                model = get_baseline_model(model_name)
                model.fit(X_full_source_df, y_A)
                
                # 源域性能（使用验证集）
                y_source_pred = model.predict(X_source_val_full)
                y_source_proba = model.predict_proba(X_source_val_full)[:, 1]
                
                source_metrics = evaluate_metrics(
                    y_source_val_full,
                    y_source_pred,
                    y_source_proba
                )
                
                # 目标域性能
                y_target_pred = model.predict(X_target_df)
                y_target_proba = model.predict_proba(X_target_df)[:, 1]
                
                target_metrics = evaluate_metrics(
                    y_target,
                    y_target_pred,
                    y_target_proba
                )
                
                # 源域10折交叉验证
                from sklearn.model_selection import StratifiedKFold
                from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
                
                cv_scores = {'accuracy': [], 'auc': [], 'f1': []}
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_state)
                
                for train_idx, val_idx in skf.split(X_full_source_df, y_A):
                    X_cv_train = X_full_source_df.iloc[train_idx]
                    y_cv_train = y_A[train_idx]
                    X_cv_val = X_full_source_df.iloc[val_idx]
                    y_cv_val = y_A[val_idx]
                    
                    cv_model = get_baseline_model(model_name)
                    cv_model.fit(X_cv_train, y_cv_train)
                    
                    y_cv_pred = cv_model.predict(X_cv_val)
                    y_cv_proba = cv_model.predict_proba(X_cv_val)[:, 1]
                    
                    cv_scores['accuracy'].append(accuracy_score(y_cv_val, y_cv_pred))
                    cv_scores['auc'].append(roc_auc_score(y_cv_val, y_cv_proba))
                    cv_scores['f1'].append(f1_score(y_cv_val, y_cv_pred, zero_division=0))
                
                cv_metrics = {
                    'accuracy': {'mean': np.mean(cv_scores['accuracy']), 'std': np.std(cv_scores['accuracy'])},
                    'auc': {'mean': np.mean(cv_scores['auc']), 'std': np.std(cv_scores['auc'])},
                    'f1': {'mean': np.mean(cv_scores['f1']), 'std': np.std(cv_scores['f1'])}
                }
                
                baseline_results[model_name] = {
                    'source_validation': source_metrics,
                    'target_direct': target_metrics,
                    'source_cv': cv_metrics,
                    'model_features': model.get_feature_names()
                }
                
                logging.info(f"{model_name.upper()} 模型结果:")
                logging.info(f"  源域验证集 AUC: {source_metrics['auc']:.4f}")
                logging.info(f"  目标域测试集 AUC: {target_metrics['auc']:.4f}")
                logging.info(f"  源域10折CV AUC: {cv_metrics['auc']['mean']:.4f} ± {cv_metrics['auc']['std']:.4f}")
                
            except Exception as e:
                logging.error(f"评估基线模型 {model_name} 时出错: {e}")
                baseline_results[model_name] = {'error': str(e)}
        
        return baseline_results

    def evaluate_autotabpfn_source_cv(self, cv_folds: int = 10) -> Dict[str, Any]:
        """评估AutoTabPFN在源域上的性能（交叉验证或8:2划分）"""
        if cv_folds <= 0:
            logging.info("评估AutoTabPFN源域8:2划分性能...")
            logging.info("注意：此评估使用全部数据集A，与域适应实验的数据划分独立")
            return self._evaluate_autotabpfn_source_split()
        else:
            logging.info(f"评估AutoTabPFN源域{cv_folds}折交叉验证性能...")
            logging.info("注意：此评估使用全部数据集A，与域适应实验的数据划分独立")
            return self._evaluate_autotabpfn_source_cv_internal(cv_folds)
        
    def _evaluate_autotabpfn_source_split(self) -> Dict[str, Any]:
        """使用8:2划分评估AutoTabPFN在源域上的性能"""
        # 重新加载完整的源域数据集A
        from ..config.settings import DATA_PATHS, LABEL_COL
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        
        df_A = pd.read_excel(DATA_PATHS['A'])
        X_full_source_raw = df_A[self.features].values
        y_full_source = df_A[LABEL_COL].values
        
        # 使用相同的scaler进行标准化（只对非分类特征）
        X_full_source_scaled = apply_scaler(self.scaler, X_full_source_raw, self.categorical_indices)
        
        logging.info(f"源域完整数据集大小: {X_full_source_scaled.shape[0]} 样本")
        
        # 8:2划分
        X_train_80, X_val_20, y_train_80, y_val_20 = train_test_split(
            X_full_source_scaled, y_full_source,
            test_size=0.2, stratify=y_full_source, random_state=self.random_state
        )
        
        logging.info(f"8:2划分 - 训练集: {X_train_80.shape[0]} 样本, 验证集: {X_val_20.shape[0]} 样本")
        
        # 获取最佳参数
        if not hasattr(self, 'best_params') or self.best_params is None:
            raise ValueError("请先设置best_params")
        
        model_params = self.best_params['model_params']
        
        try:
            from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
            
            # 准备phe_init_args参数
            phe_init_args = {}
            for key in ['max_models', 'validation_method', 'n_repeats', 'n_folds', 'holdout_fraction', 'ges_n_iterations']:
                if key in model_params:
                    phe_init_args[key] = model_params[key]
            
            # 创建模型
            model = AutoTabPFNClassifier(
                device='cuda',
                max_time=model_params.get('max_time', 60),
                preset=model_params.get('preset', 'default'),
                ges_scoring_string=model_params.get('ges_scoring', 'roc'),
                random_state=self.random_state,
                categorical_feature_indices=self.categorical_indices,
                ignore_pretraining_limits=model_params.get('ignore_limits', False),
                phe_init_args=phe_init_args if phe_init_args else None
            )
            
            # 训练和评估
            model.fit(X_train_80, y_train_80)
            
            y_pred = model.predict(X_val_20)
            y_proba = model.predict_proba(X_val_20)
            
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                y_proba_1d = y_proba[:, 1]
            else:
                y_proba_1d = y_proba.ravel()
            
            accuracy = accuracy_score(y_val_20, y_pred)
            auc = roc_auc_score(y_val_20, y_proba_1d)
            f1 = f1_score(y_val_20, y_pred, zero_division=0)
            
            logging.info(f"AutoTabPFN源域8:2划分结果:")
            logging.info(f"  AUC: {auc:.4f}")
            logging.info(f"  ACC: {accuracy:.4f}")
            logging.info(f"  F1:  {f1:.4f}")
            
            # 返回格式与CV结果一致
            return {
                'accuracy': {'mean': accuracy, 'std': 0.0},
                'auc': {'mean': auc, 'std': 0.0},
                'f1': {'mean': f1, 'std': 0.0}
            }
            
        except Exception as e:
            logging.error(f"8:2划分评估失败: {e}")
            return None
    
    def _evaluate_autotabpfn_source_cv_internal(self, cv_folds: int) -> Dict[str, Any]:
        """使用交叉验证评估AutoTabPFN在源域上的性能"""
        # 重新加载完整的源域数据集A（不受训练/验证划分限制）
        from ..config.settings import DATA_PATHS, LABEL_COL
        import pandas as pd
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        
        df_A = pd.read_excel(DATA_PATHS['A'])
        X_full_source_raw = df_A[self.features].values  # 使用所选特征
        y_full_source = df_A[LABEL_COL].values
        
        # 使用相同的scaler进行标准化（只对非分类特征）
        X_full_source_scaled = apply_scaler(self.scaler, X_full_source_raw, self.categorical_indices)
        
        logging.info(f"源域完整数据集大小: {X_full_source_scaled.shape[0]} 样本")
        logging.info(f"域适应使用的训练数据大小: {self.X_source_train_raw.shape[0]} 样本 (80%)")
        logging.info(f"域适应使用的验证数据大小: {self.X_source_val_raw.shape[0]} 样本 (20%)")
        
        cv_scores = {'accuracy': [], 'auc': [], 'f1': []}
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # 获取最佳参数
        if not hasattr(self, 'best_params') or self.best_params is None:
            raise ValueError("请先设置best_params")
        
        model_params = self.best_params['model_params']
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_full_source_scaled, y_full_source), 1):
            logging.info(f"  处理第 {fold}/{cv_folds} 折...")
            
            X_cv_train = X_full_source_scaled[train_idx]
            y_cv_train = y_full_source[train_idx]
            X_cv_val = X_full_source_scaled[val_idx]
            y_cv_val = y_full_source[val_idx]
            
            # 创建和训练AutoTabPFN模型
            try:
                from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
                
                # 使用最佳参数创建模型
                # 准备phe_init_args参数
                phe_init_args = {}
                for key in ['max_models', 'validation_method', 'n_repeats', 'n_folds', 'holdout_fraction', 'ges_n_iterations']:
                    if key in model_params:
                        phe_init_args[key] = model_params[key]
                
                cv_model = AutoTabPFNClassifier(
                    device='cuda',
                    max_time=model_params.get('max_time', 60),
                    preset=model_params.get('preset', 'default'),
                    ges_scoring_string=model_params.get('ges_scoring', 'roc'),
                    random_state=self.random_state,
                    categorical_feature_indices=self.categorical_indices,
                    ignore_pretraining_limits=model_params.get('ignore_limits', False),
                    phe_init_args=phe_init_args if phe_init_args else None
                )
                
                cv_model.fit(X_cv_train, y_cv_train)
                
                y_cv_pred = cv_model.predict(X_cv_val)
                y_cv_proba = cv_model.predict_proba(X_cv_val)
                
                if y_cv_proba.ndim > 1 and y_cv_proba.shape[1] > 1:
                    y_cv_proba_1d = y_cv_proba[:, 1]
                else:
                    y_cv_proba_1d = y_cv_proba.ravel()
                
                cv_scores['accuracy'].append(accuracy_score(y_cv_val, y_cv_pred))
                cv_scores['auc'].append(roc_auc_score(y_cv_val, y_cv_proba_1d))
                cv_scores['f1'].append(f1_score(y_cv_val, y_cv_pred, zero_division=0))
                
                logging.info(f"    Fold {fold} - AUC: {cv_scores['auc'][-1]:.4f}, ACC: {cv_scores['accuracy'][-1]:.4f}, F1: {cv_scores['f1'][-1]:.4f}")
                
            except Exception as e:
                logging.error(f"Fold {fold} 评估失败: {e}")
                continue
        
        if len(cv_scores['auc']) == 0:
            logging.error("所有CV fold都失败了")
            return None
        
        # 计算统计结果
        cv_metrics = {
            'accuracy': {'mean': np.mean(cv_scores['accuracy']), 'std': np.std(cv_scores['accuracy'])},
            'auc': {'mean': np.mean(cv_scores['auc']), 'std': np.std(cv_scores['auc'])},
            'f1': {'mean': np.mean(cv_scores['f1']), 'std': np.std(cv_scores['f1'])}
        }
        
        logging.info(f"AutoTabPFN源域{cv_folds}折CV结果:")
        logging.info(f"  AUC: {cv_metrics['auc']['mean']:.4f} ± {cv_metrics['auc']['std']:.4f}")
        logging.info(f"  ACC: {cv_metrics['accuracy']['mean']:.4f} ± {cv_metrics['accuracy']['std']:.4f}")
        logging.info(f"  F1:  {cv_metrics['f1']['mean']:.4f} ± {cv_metrics['f1']['std']:.4f}")
        
        return cv_metrics

    def evaluate_final_model(self) -> Dict[str, Any]:
        """评估最终模型在目标域上的性能"""
        logging.info("评估最终模型在目标域上的性能...")
        
        if self.final_model is None:
            raise ValueError("请先训练最终模型")
        
        # 导入apply_scaler函数
        from ..preprocessing.scaler import apply_scaler
        
        # 准备完整源域数据用于MMD域适应
        X_full_source_raw = np.vstack([self.X_source_train_raw, self.X_source_val_raw])
        y_full_source = np.hstack([self.y_source_train, self.y_source_val])
        
        # 1. 直接在目标域上评估（无域适应）
        y_target_pred_direct = self.final_model.predict(self.X_target_test)
        y_target_proba_direct = self.final_model.predict_proba(self.X_target_test)
        
        if y_target_proba_direct.ndim > 1 and y_target_proba_direct.shape[1] > 1:
            y_target_proba_direct_1d = y_target_proba_direct[:, 1]
        else:
            y_target_proba_direct_1d = y_target_proba_direct.ravel()
        
        direct_metrics = evaluate_metrics(
            self.y_target_test,
            y_target_pred_direct,
            y_target_proba_direct_1d
        )
        
        # 2. 使用MMD域适应后评估
        adapted_metrics = None
        if self.best_params['mmd_params']:
            logging.info("进行MMD域适应...")
            
            # 获取MMD方法的默认参数并更新
            default_mmd_params = MMD_METHODS.get(self.mmd_method, {}).copy()
            default_mmd_params.update(self.best_params['mmd_params'])
            
            # 修复numpy类型问题
            for key, value in default_mmd_params.items():
                if isinstance(value, np.integer):
                    default_mmd_params[key] = int(value)
                elif isinstance(value, np.floating):
                    default_mmd_params[key] = float(value)
                elif isinstance(value, np.bool_):
                    default_mmd_params[key] = bool(value)
            
            # 进行域适应
            X_target_adapted, adaptation_info = self._apply_domain_adaptation(
                X_full_source_raw, y_full_source, self.X_target_test_raw,
                default_mmd_params
            )
            
            # 标准化适应后的数据（只对非分类特征）
            X_target_adapted_scaled = apply_scaler(self.scaler, X_target_adapted, self.categorical_indices)
            
            # 在适应后的数据上预测
            y_target_pred_adapted = self.final_model.predict(X_target_adapted_scaled)
            y_target_proba_adapted = self.final_model.predict_proba(X_target_adapted_scaled)
            
            if y_target_proba_adapted.ndim > 1 and y_target_proba_adapted.shape[1] > 1:
                y_target_proba_adapted_1d = y_target_proba_adapted[:, 1]
            else:
                y_target_proba_adapted_1d = y_target_proba_adapted.ravel()
            
            adapted_metrics = evaluate_metrics(
                self.y_target_test,
                y_target_pred_adapted,
                y_target_proba_adapted_1d
            )
            
            logging.info(f"域适应完成，MMD减少: {adaptation_info.get('reduction', 0):.2f}%")
        
        # 3. 源域内验证性能（作为基准）
        y_source_val_pred = self.final_model.predict(self.X_source_val)
        y_source_val_proba = self.final_model.predict_proba(self.X_source_val)
        
        if y_source_val_proba.ndim > 1 and y_source_val_proba.shape[1] > 1:
            y_source_val_proba_1d = y_source_val_proba[:, 1]
        else:
            y_source_val_proba_1d = y_source_val_proba.ravel()
        
        source_metrics = evaluate_metrics(
            self.y_source_val,
            y_source_val_pred,
            y_source_val_proba_1d
        )
        
        results = {
            'source_validation': source_metrics,
            'target_direct': direct_metrics,
            'target_adapted': adapted_metrics,
            'adaptation_info': adaptation_info if adapted_metrics else None
        }
        
        # 打印结果
        logging.info("=" * 50)
        logging.info("最终评估结果:")
        logging.info("=" * 50)
        logging.info(f"源域验证集 AUC: {source_metrics['auc']:.4f}")
        logging.info(f"目标域直接预测 AUC: {direct_metrics['auc']:.4f}")
        if adapted_metrics:
            logging.info(f"目标域域适应后 AUC: {adapted_metrics['auc']:.4f}")
            improvement = adapted_metrics['auc'] - direct_metrics['auc']
            logging.info(f"域适应改进: {improvement:.4f} ({improvement/direct_metrics['auc']*100:.1f}%)")
        
        return results

    def save_results(self, optimization_results: Dict[str, Any], 
                    evaluation_results: Dict[str, Any],
                    baseline_results: Dict[str, Any] = None) -> None:
        """保存所有结果"""
        logging.info("保存实验结果...")
        
        # 保存优化结果
        opt_results_path = os.path.join(self.save_path, 'optimization_results.json')
        with open(opt_results_path, 'w', encoding='utf-8') as f:
            # 处理不可序列化的对象
            serializable_results = {
                'best_params': optimization_results['best_params'],
                'best_score': optimization_results['best_score'],
                'total_trials': len(optimization_results['all_trials']),
                'all_trials': []
            }
            
            for trial in optimization_results['all_trials']:
                serializable_trial = {
                    'params': trial['params'],
                    'model_params': trial['model_params'],
                    'mmd_params': trial['mmd_params'],
                    'cv_scores': trial['cv_scores'],
                    'mean_cv_score': trial['mean_cv_score'],
                    'std_cv_score': trial['std_cv_score'],
                    'timestamp': trial['timestamp']
                }
                serializable_results['all_trials'].append(serializable_trial)
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存评估结果
        eval_results_path = os.path.join(self.save_path, 'evaluation_results.json')
        with open(eval_results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        # 保存基线模型结果
        if baseline_results is not None and isinstance(baseline_results, dict):
            baseline_results_path = os.path.join(self.save_path, 'baseline_models_results.json')
            with open(baseline_results_path, 'w', encoding='utf-8') as f:
                # 移除模型对象，只保存结果
                serializable_baseline = {}
                for model_name, results in baseline_results.items():
                    if 'error' not in results:
                        serializable_baseline[model_name] = {
                            'source_validation': results['source_validation'],
                            'target_direct': results['target_direct'],
                            'source_cv': results['source_cv'],
                            'model_features': results['model_features']
                        }
                    else:
                        serializable_baseline[model_name] = results
                json.dump(serializable_baseline, f, indent=2, ensure_ascii=False)
        
        # 保存实验配置
        config_path = os.path.join(self.save_path, 'experiment_config.json')
        config = {
            'model_type': self.model_type,
            'feature_type': self.feature_type,
            'mmd_method': self.mmd_method,
            'use_class_conditional': self.use_class_conditional,
            'use_categorical': self.use_categorical,
            'source_val_split': self.source_val_split,
            'cv_folds': self.cv_folds,
            'n_calls': self.n_calls,
            'random_state': self.random_state,
            'target_domain': self.target_domain,
            'use_source_cv_for_mmd_tuning': self.use_source_cv_for_mmd_tuning,
            'features': self.features,
            'categorical_indices': self.categorical_indices
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logging.info(f"结果已保存到: {self.save_path}")

    def run_complete_experiment(self) -> Dict[str, Any]:
        """运行完整的标准域适应实验"""
        logging.info("开始运行完整的标准域适应实验...")
        
        # 1. 加载和准备数据
        self.load_and_prepare_data()
        
        # 2. 评估基线模型
        baseline_results = self.evaluate_baseline_models_performance()
        
        # 3. 运行贝叶斯优化
        optimization_results = self.run_optimization()
        
        # 4. 训练最终模型
        self.train_final_model()
        
        # 5. 评估最终模型
        evaluation_results = self.evaluate_final_model()
        
        # 6. 保存结果
        self.save_results(optimization_results, evaluation_results, baseline_results)
        
        # 7. 返回完整结果
        complete_results = {
            'optimization': optimization_results,
            'evaluation': evaluation_results,
            'baseline_models': baseline_results,
            'config': {
                'model_type': self.model_type,
                'feature_type': self.feature_type,
                'mmd_method': self.mmd_method,
                'target_domain': self.target_domain,
                'best_params': self.best_params
            }
        }
        
        return complete_results


def run_standard_domain_adaptation(model_type: str = 'auto',
                                 feature_type: str = 'best7',
                                 mmd_method: str = 'linear',
                                 use_class_conditional: bool = False,
                                 use_categorical: bool = True,
                                 source_val_split: float = 0.2,
                                 cv_folds: int = 5,
                                 n_calls: int = 50,
                                 target_domain: str = 'B',
                                 save_path: str = './results_standard_domain_adaptation',
                                 use_source_cv_for_mmd_tuning: bool = True,
                                 **kwargs) -> Dict[str, Any]:
    """
    运行标准域适应实验的便捷函数
    
    参数:
    - model_type: 模型类型
    - feature_type: 特征类型
    - mmd_method: MMD方法
    - use_class_conditional: 是否使用类条件MMD
    - use_categorical: 是否使用类别特征
    - source_val_split: 源域验证集比例
    - cv_folds: 交叉验证折数
    - n_calls: 优化迭代次数
    - target_domain: 目标域选择
    - save_path: 保存路径
    - use_source_cv_for_mmd_tuning: 是否使用源域CV调优MMD参数
    - **kwargs: 其他参数
    
    返回:
    - Dict[str, Any]: 完整的实验结果
    """
    
    optimizer = StandardDomainAdaptationOptimizer(
        model_type=model_type,
        feature_type=feature_type,
        mmd_method=mmd_method,
        use_class_conditional=use_class_conditional,
        use_categorical=use_categorical,
        source_val_split=source_val_split,
        cv_folds=cv_folds,
        n_calls=n_calls,
        target_domain=target_domain,
        save_path=save_path,
        use_source_cv_for_mmd_tuning=use_source_cv_for_mmd_tuning,
        **kwargs
    )
    
    return optimizer.run_complete_experiment()