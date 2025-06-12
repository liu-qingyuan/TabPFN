"""
跨域实验运行器

整合多模型支持、MMD域适应和跨数据集评估功能
支持A→B、A→C的域适应实验，以及ABC三个数据集的综合评估
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import KFold
import json
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

from ..config.settings import (
    DATA_PATHS, LABEL_COL,
    get_features_by_type, get_categorical_indices, get_model_config, MMD_METHODS
)
from ..preprocessing.scaler import fit_apply_scaler
from ..preprocessing.mmd import mmd_transform
from ..preprocessing.class_conditional_mmd import class_conditional_mmd_transform
from .model_selector import get_model
from ..metrics.evaluation import evaluate_metrics
from ..visualization import (
    compare_before_after_adaptation, plot_mmd_methods_comparison,
    visualize_tsne, visualize_feature_histograms, plot_roc_curve,
    generate_performance_comparison_plots
)
from ..metrics.classification import (
    calculate_basic_metrics, calculate_cv_summary, calculate_improvement,
    log_metrics
)
from ..metrics.cross_domain_metrics import (
    evaluate_model_on_external_cv, evaluate_single_external_dataset,
    log_external_cv_results, calculate_cross_domain_improvement,
    summarize_cross_domain_results
)

class CrossDomainExperimentRunner:
    """跨域实验运行器"""
    
    def __init__(self, 
                 model_type: str = 'auto',
                 feature_type: str = 'best7',
                 use_mmd_adaptation: bool = True,
                 mmd_method: str = 'linear',
                 use_class_conditional: bool = False,
                 use_threshold_optimizer: bool = False,
                 save_path: str = './results_cross_domain',
                 skip_cv_on_a: bool = False,
                 evaluation_mode: str = 'cv',  # 'single' 或 'cv'
                 data_split_strategy: str = 'two-way',  # 'two-way' 或 'three-way'
                 validation_split: Optional[float] = None,  # 三分法时的验证集比例
                 target_domain: str = 'B',  # 目标域选择: 'B' 或 'C'
                 **kwargs: Any):
        """
        初始化跨域实验运行器
        
        参数:
        - model_type: 模型类型 ('auto', 'base', 'rf')
        - feature_type: 特征类型 ('all', 'best7')
        - use_mmd_adaptation: 是否使用MMD域适应
        - mmd_method: MMD方法 ('linear', 'kpca', 'mean_std')
        - use_class_conditional: 是否使用类条件MMD
        - use_threshold_optimizer: 是否使用阈值优化
        - save_path: 结果保存路径
        - skip_cv_on_a: 是否跳过数据集A的交叉验证
        - evaluation_mode: 外部评估模式 ('single': 单次评估, 'cv': 10折CV评估, 'proper_cv': 正确的10折CV)
        - data_split_strategy: 数据划分策略 ('two-way': 二分法, 'three-way': 三分法)
        - validation_split: 三分法时验证集比例 (默认0.7)
        - target_domain: 目标域选择: 'B' 或 'C'
        - **kwargs: 其他参数
        """
        self.model_type = model_type
        self.feature_type = feature_type
        self.use_mmd_adaptation = use_mmd_adaptation
        self.mmd_method = mmd_method
        self.use_class_conditional = use_class_conditional
        self.use_threshold_optimizer = use_threshold_optimizer
        self.save_path = save_path
        self.skip_cv_on_a = skip_cv_on_a
        self.evaluation_mode = evaluation_mode
        self.data_split_strategy = data_split_strategy
        self.validation_split = validation_split if validation_split is not None else 0.7
        self.target_domain = target_domain
        self.kwargs = kwargs
        
        # 验证参数
        if data_split_strategy == 'three-way' and (validation_split is None or validation_split <= 0 or validation_split >= 1):
            raise ValueError("三分法模式下，validation_split 必须在 (0, 1) 范围内")
        
        # 获取特征和类别索引
        self.features = get_features_by_type(feature_type)
        self.categorical_indices = get_categorical_indices(feature_type)
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 初始化结果存储
        self.results = {}
        
        logging.info(f"初始化跨域实验运行器:")
        logging.info(f"  模型类型: {model_type}")
        logging.info(f"  特征类型: {feature_type} ({len(self.features)}个特征)")
        logging.info(f"  数据划分策略: {data_split_strategy}")
        if data_split_strategy == 'three-way':
            logging.info(f"  验证集比例: {self.validation_split}")
        logging.info(f"  MMD域适应: {use_mmd_adaptation}")
        if use_mmd_adaptation:
            logging.info(f"  MMD方法: {mmd_method}")
            logging.info(f"  类条件MMD: {use_class_conditional}")
        logging.info(f"  阈值优化: {use_threshold_optimizer}")
        logging.info(f"  目标域: {target_domain}")
        
    def load_datasets(self) -> Dict[str, np.ndarray]:
        """加载所有数据集"""
        logging.info("加载数据集...")
        
        datasets = {}
        
        # 加载数据集A (训练集)
        df_A = pd.read_excel(DATA_PATHS['A'])
        datasets['X_A_raw'] = df_A[self.features].values
        datasets['y_A'] = df_A[LABEL_COL].values
        
        # 根据target_domain参数加载目标域数据集
        if self.target_domain == 'B':
            # 加载数据集B
            df_target = pd.read_excel(DATA_PATHS['B'])
            target_name = 'B'
        elif self.target_domain == 'C':
            # 加载数据集C
            df_target = pd.read_excel(DATA_PATHS['C'])
            target_name = 'C'
        else:
            raise ValueError(f"不支持的目标域: {self.target_domain}")
        
        X_target_raw = df_target[self.features].values
        y_target = df_target[LABEL_COL].values
        
        # 根据数据划分策略处理目标域数据
        if self.data_split_strategy == 'two-way':
            # 二分法：完整目标域用于测试
            datasets[f'X_{target_name}_raw'] = X_target_raw
            datasets[f'y_{target_name}'] = y_target
            datasets[f'X_{target_name}_val_raw'] = None
            datasets[f'y_{target_name}_val'] = None
            datasets[f'X_{target_name}_holdout_raw'] = None
            datasets[f'y_{target_name}_holdout'] = None
            
        elif self.data_split_strategy == 'three-way':
            # 三分法：目标域划分为验证集和保留测试集
            from sklearn.model_selection import train_test_split
            
            # 按照指定比例划分目标域数据
            X_target_val_raw, X_target_holdout_raw, y_target_val, y_target_holdout = train_test_split(
                X_target_raw, y_target,
                train_size=self.validation_split,
                random_state=42,
                stratify=y_target
            )
            
            datasets[f'X_{target_name}_raw'] = X_target_raw  # 完整目标域数据（用于某些分析）
            datasets[f'y_{target_name}'] = y_target
            datasets[f'X_{target_name}_val_raw'] = X_target_val_raw  # 验证集
            datasets[f'y_{target_name}_val'] = y_target_val
            datasets[f'X_{target_name}_holdout_raw'] = X_target_holdout_raw  # 保留测试集
            datasets[f'y_{target_name}_holdout'] = y_target_holdout
            
            logging.info(f"{target_name}域数据划分:")
            logging.info(f"  验证集: {X_target_val_raw.shape[0]} 样本")
            logging.info(f"  保留测试集: {X_target_holdout_raw.shape[0]} 样本")
            logging.info(f"  验证集标签分布: {np.bincount(y_target_val.astype(int))}")
            logging.info(f"  保留测试集标签分布: {np.bincount(y_target_holdout.astype(int))}")
        
        # 为了保持向后兼容性，设置B和C的数据
        if self.target_domain == 'B':
            # 如果目标域是B，设置B的数据，C设为None
            datasets['X_B_raw'] = datasets[f'X_{target_name}_raw']
            datasets['y_B'] = datasets[f'y_{target_name}']
            datasets['X_B_val_raw'] = datasets[f'X_{target_name}_val_raw']
            datasets['y_B_val'] = datasets[f'y_{target_name}_val']
            datasets['X_B_holdout_raw'] = datasets[f'X_{target_name}_holdout_raw']
            datasets['y_B_holdout'] = datasets[f'y_{target_name}_holdout']
            datasets['X_C_raw'] = None
            datasets['y_C'] = None
            datasets['has_C'] = False
        else:
            # 如果目标域是C，设置C的数据，B设为None
            datasets['X_C_raw'] = datasets[f'X_{target_name}_raw']
            datasets['y_C'] = datasets[f'y_{target_name}']
            datasets['X_C_val_raw'] = datasets[f'X_{target_name}_val_raw']
            datasets['y_C_val'] = datasets[f'y_{target_name}_val']
            datasets['X_C_holdout_raw'] = datasets[f'X_{target_name}_holdout_raw']
            datasets['y_C_holdout'] = datasets[f'y_{target_name}_holdout']
            datasets['X_B_raw'] = None
            datasets['y_B'] = None
            datasets['has_C'] = True
        
        # 数据标准化 - 用A数据集拟合scaler
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        datasets['X_A_scaled'] = scaler.fit_transform(datasets['X_A_raw'])
        
        if self.data_split_strategy == 'two-way':
            # 二分法：标准化完整目标域数据
            datasets[f'X_{target_name}_scaled'] = scaler.transform(datasets[f'X_{target_name}_raw'])
            datasets[f'X_{target_name}_val_scaled'] = None
            datasets[f'X_{target_name}_holdout_scaled'] = None
            
        elif self.data_split_strategy == 'three-way':
            # 三分法：分别标准化验证集和保留测试集
            datasets[f'X_{target_name}_scaled'] = scaler.transform(datasets[f'X_{target_name}_raw'])  # 完整目标域
            datasets[f'X_{target_name}_val_scaled'] = scaler.transform(datasets[f'X_{target_name}_val_raw'])  # 验证集
            datasets[f'X_{target_name}_holdout_scaled'] = scaler.transform(datasets[f'X_{target_name}_holdout_raw'])  # 保留测试集
        
        # 为了保持向后兼容性，设置B和C的标准化数据
        if self.target_domain == 'B':
            datasets['X_B_scaled'] = datasets[f'X_{target_name}_scaled']
            datasets['X_B_val_scaled'] = datasets[f'X_{target_name}_val_scaled']
            datasets['X_B_holdout_scaled'] = datasets[f'X_{target_name}_holdout_scaled']
            datasets['X_C_scaled'] = None
        else:
            datasets['X_C_scaled'] = datasets[f'X_{target_name}_scaled']
            datasets['X_C_val_scaled'] = datasets[f'X_{target_name}_val_scaled']
            datasets['X_C_holdout_scaled'] = datasets[f'X_{target_name}_holdout_scaled']
            datasets['X_B_scaled'] = None
            datasets['X_B_val_scaled'] = None
            datasets['X_B_holdout_scaled'] = None
        
        datasets['scaler'] = scaler
        datasets['target_domain'] = self.target_domain
        
        # 打印数据信息
        logging.info(f"数据集A形状: {datasets['X_A_raw'].shape}")
        logging.info(f"数据集{target_name}形状: {datasets[f'X_{target_name}_raw'].shape}")
        logging.info(f"数据集A标签分布: {np.bincount(datasets['y_A'].astype(int))}")
        logging.info(f"数据集{target_name}标签分布: {np.bincount(datasets[f'y_{target_name}'].astype(int))}")
        
        return datasets
    
    def run_cross_validation(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 10) -> Dict[str, Any]:
        """在数据集A上运行交叉验证"""
        logging.info(f"运行{cv_folds}折交叉验证...")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 创建模型
            model_config = get_model_config(self.model_type, categorical_feature_indices=self.categorical_indices)
            model = get_model(
                self.model_type,
                **model_config
            )
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)
            
            # 计算指标
            metrics = calculate_basic_metrics(y_val, y_pred, y_proba)
            metrics['fold'] = fold
            cv_scores.append(metrics)
            
            logging.info(f"Fold {fold}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
        
        # 计算平均结果
        avg_results = calculate_cv_summary(cv_scores)
        
        logging.info("交叉验证平均结果:")
        log_metrics(avg_results, "CV")
        
        return avg_results
    
    def run_domain_adaptation(self, X_source: np.ndarray, y_source: np.ndarray, 
                            X_target: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """运行域适应"""
        if not self.use_mmd_adaptation:
            return X_target, {'method': 'none', 'mmd_reduction': 0}
        
        logging.info(f"运行{self.mmd_method}域适应...")
        
        # 从配置文件获取MMD方法参数
        mmd_params = MMD_METHODS.get(self.mmd_method, {}).copy()
        
        # 合并用户提供的参数
        mmd_params.update(self.kwargs)
        
        if self.use_class_conditional:
            # 使用类条件MMD
            X_target_aligned, adaptation_info = class_conditional_mmd_transform(
                X_source, y_source, X_target,
                method=self.mmd_method,
                cat_idx=self.categorical_indices,
                **mmd_params
            )
        else:
            # 使用标准MMD
            X_target_aligned, adaptation_info = mmd_transform(
                X_source, X_target,
                method=self.mmd_method,
                cat_idx=self.categorical_indices,
                **mmd_params
            )
        
        logging.info(f"域适应完成，MMD减少: {adaptation_info.get('reduction', 0):.2f}%")
        
        return X_target_aligned, adaptation_info
    
    def evaluate_external_dataset_cv(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray,
                                   X_train_raw: np.ndarray, X_test_raw: np.ndarray,
                                   dataset_name: str, cv_folds: int = 10) -> Dict[str, Any]:
        """在外部数据集上进行10折交叉验证评估，包括域适应前后对比"""
        logging.info(f"在数据集{dataset_name}上进行{cv_folds}折交叉验证评估...")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # 存储结果
        cv_scores_without_da = []  # 不使用域适应
        cv_scores_with_da = []     # 使用域适应
        
        for fold, (_, val_idx) in enumerate(kf.split(X_test), 1):
            X_test_val = X_test[val_idx]
            y_test_val = y_test[val_idx]
            
            # 对应的原始数据
            X_test_val_raw = X_test_raw[val_idx]
            
            # 创建模型配置
            model_config = get_model_config(self.model_type, categorical_feature_indices=self.categorical_indices)
            
            # ===== 1. 不使用域适应的评估 =====
            model_no_da = get_model(
                self.model_type,
                **model_config
            )
            
            # 在A数据上训练，在B数据上测试（不做域适应）
            model_no_da.fit(X_train, y_train)
            y_pred_no_da = model_no_da.predict(X_test_val)
            y_proba_no_da = model_no_da.predict_proba(X_test_val)
            
            # 计算指标
            metrics_no_da = calculate_basic_metrics(y_test_val, y_pred_no_da, y_proba_no_da)
            metrics_no_da['fold'] = fold
            cv_scores_without_da.append(metrics_no_da)
            
            # ===== 2. 使用域适应的评估 =====
            if self.use_mmd_adaptation:
                # 域适应：使用原始数据（域适应算法内部会处理标准化）
                X_test_val_adapted, _ = self.run_domain_adaptation(
                    X_train_raw, y_train, X_test_val_raw
                )
                
                model_with_da = get_model(
                    self.model_type,
                    **model_config
                )
                
                # 在A数据上训练，在域适应后的B数据上测试
                model_with_da.fit(X_train, y_train)
                y_pred_with_da = model_with_da.predict(X_test_val_adapted)
                y_proba_with_da = model_with_da.predict_proba(X_test_val_adapted)
                
                # 计算指标
                metrics_with_da = calculate_basic_metrics(y_test_val, y_pred_with_da, y_proba_with_da)
                metrics_with_da['fold'] = fold
                cv_scores_with_da.append(metrics_with_da)
            else:
                # 如果不使用域适应，复制无域适应的结果
                cv_scores_with_da.append(metrics_no_da.copy())
            
            logging.info(f"Fold {fold} - 无域适应: Acc={metrics_no_da['accuracy']:.4f}, AUC={metrics_no_da['auc']:.4f}")
            if self.use_mmd_adaptation:
                metrics_with_da = cv_scores_with_da[-1]  # 获取最后添加的结果
                logging.info(f"Fold {fold} - 有域适应: Acc={metrics_with_da['accuracy']:.4f}, AUC={metrics_with_da['auc']:.4f}")
        
        # 计算平均结果
        results = {
            'without_domain_adaptation': calculate_cv_summary(cv_scores_without_da),
            'with_domain_adaptation': calculate_cv_summary(cv_scores_with_da)
        }
        
        # 计算改进幅度
        if self.use_mmd_adaptation:
            results['improvement'] = calculate_improvement(
                results['without_domain_adaptation'],
                results['with_domain_adaptation']
            )
        
        logging.info(f"数据集{dataset_name} 10折CV结果:")
        log_metrics(results['without_domain_adaptation'], f"无域适应-{dataset_name}")
        if self.use_mmd_adaptation:
            log_metrics(results['with_domain_adaptation'], f"有域适应-{dataset_name}")
            log_metrics(results['improvement'], f"改进幅度-{dataset_name}")
        
        return results
    
    def evaluate_external_dataset_single(self, X_train: np.ndarray, y_train: np.ndarray,
                                        X_test: np.ndarray, y_test: np.ndarray,
                                        X_train_raw: np.ndarray, X_test_raw: np.ndarray,
                                        dataset_name: str) -> Dict[str, Any]:
        """
        按照predict_healthcare_auto_external_adjust_parameter.py的方式进行外部数据集评估
        关键：在全部A数据上训练一个模型，然后在全部B数据上直接评估（不使用交叉验证）
        """
        logging.info(f"在数据集{dataset_name}上进行单次评估（与原始脚本一致）...")
        
        # 使用hyperopt最佳参数配置
        from ..config.settings import get_model_config
        model_config = get_model_config(self.model_type, categorical_feature_indices=self.categorical_indices)
        
        # ===== 1. 不使用域适应的评估 =====
        # 在全部A数据上训练一个模型
        model_no_da = get_model(
            self.model_type,
            **model_config
        )
        
        logging.info("训练无域适应模型（使用hyperopt最佳参数）...")
        model_no_da.fit(X_train, y_train)
        
        # 在全部B数据上直接评估（与原始脚本一致）
        y_pred = model_no_da.predict(X_test)
        y_proba = model_no_da.predict_proba(X_test)
        
        # 计算指标
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba[:, 1])
        f1 = f1_score(y_test, y_pred)
        
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(y_test, y_pred)
        acc_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
        acc_1 = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
        
        results_no_da = {
            'accuracy': acc,
            'auc': auc,
            'f1': f1,
            'acc_0': acc_0,
            'acc_1': acc_1,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        logging.info(f"无域适应结果:")
        logging.info(f"  Accuracy: {acc:.4f}")
        logging.info(f"  AUC: {auc:.4f}")
        logging.info(f"  F1: {f1:.4f}")
        logging.info(f"  Class 0 Accuracy: {acc_0:.4f}")
        logging.info(f"  Class 1 Accuracy: {acc_1:.4f}")
        
        # ===== 2. 使用域适应的评估 =====
        if self.use_mmd_adaptation:
            logging.info("训练有域适应模型...")
            
            # 使用原始数据进行域适应
            X_test_adapted, adaptation_info = self.run_domain_adaptation(
                X_train_raw, y_train, X_test_raw
            )
            
            # 在全部A数据上训练一个模型
            model_with_da = get_model(
                self.model_type,
                **model_config
            )
            
            model_with_da.fit(X_train, y_train)
            
            # 在域适应后的全部B数据上直接评估
            y_pred_da = model_with_da.predict(X_test_adapted)
            y_proba_da = model_with_da.predict_proba(X_test_adapted)
            
            # 计算指标
            acc_da = accuracy_score(y_test, y_pred_da)
            auc_da = roc_auc_score(y_test, y_proba_da[:, 1])
            f1_da = f1_score(y_test, y_pred_da)
            
            # 计算混淆矩阵
            conf_matrix_da = confusion_matrix(y_test, y_pred_da)
            acc_0_da = conf_matrix_da[0, 0] / (conf_matrix_da[0, 0] + conf_matrix_da[0, 1]) if (conf_matrix_da[0, 0] + conf_matrix_da[0, 1]) > 0 else 0
            acc_1_da = conf_matrix_da[1, 1] / (conf_matrix_da[1, 0] + conf_matrix_da[1, 1]) if (conf_matrix_da[1, 0] + conf_matrix_da[1, 1]) > 0 else 0
            
            results_with_da = {
                'accuracy': acc_da,
                'auc': auc_da,
                'f1': f1_da,
                'acc_0': acc_0_da,
                'acc_1': acc_1_da,
                'confusion_matrix': conf_matrix_da.tolist()
            }
            
            logging.info(f"有域适应结果:")
            logging.info(f"  Accuracy: {acc_da:.4f}")
            logging.info(f"  AUC: {auc_da:.4f}")
            logging.info(f"  F1: {f1_da:.4f}")
            logging.info(f"  Class 0 Accuracy: {acc_0_da:.4f}")
            logging.info(f"  Class 1 Accuracy: {acc_1_da:.4f}")
            
            logging.info(f"域适应完成，MMD减少: {adaptation_info.get('reduction', 0):.2f}%")
        else:
            results_with_da = results_no_da.copy()
        
        # 组织结果
        results = {
            'without_domain_adaptation': results_no_da,
            'with_domain_adaptation': results_with_da
        }
        
        # 计算改进幅度
        if self.use_mmd_adaptation and results_no_da != results_with_da:
            auc_improvement = results_with_da['auc'] - results_no_da['auc']
            acc_improvement = results_with_da['accuracy'] - results_no_da['accuracy']
            results['improvement'] = {
                'auc_improvement': auc_improvement,
                'accuracy_improvement': acc_improvement
            }
            logging.info(f"域适应改进: AUC +{auc_improvement:.4f}, Accuracy +{acc_improvement:.4f}")
        
        return results
    
    def evaluate_external_dataset_proper_cv(self, X_train: np.ndarray, y_train: np.ndarray,
                                           X_test: np.ndarray, y_test: np.ndarray,
                                           X_train_raw: np.ndarray, X_test_raw: np.ndarray,
                                           dataset_name: str, cv_folds: int = 10) -> Dict[str, Any]:
        """
        按照predict_healthcare_auto_external_adjust_parameter.py的方式进行外部数据集评估
        关键：在全部A数据上训练一个模型，然后在B数据的不同折上评估
        """
        logging.info(f"在数据集{dataset_name}上进行正确的10折交叉验证评估...")
        
        # 创建模型配置
        model_config = get_model_config(self.model_type, categorical_feature_indices=self.categorical_indices)
        
        # ===== 1. 不使用域适应的评估 =====
        # 在全部A数据上训练一个模型
        model_no_da = get_model(
            self.model_type,
            **model_config
        )
        
        logging.info("训练无域适应模型...")
        model_no_da.fit(X_train, y_train)
        
        # 使用这个模型在B数据的10折上评估
        results_no_da = evaluate_model_on_external_cv(
            model_no_da, X_test, y_test, cv_folds
        )
        
        # ===== 2. 使用域适应的评估 =====
        if self.use_mmd_adaptation:
            logging.info("训练有域适应模型...")
            
            # 使用原始数据进行域适应
            X_test_adapted, adaptation_info = self.run_domain_adaptation(
                X_train_raw, y_train, X_test_raw
            )
            
            # 在全部A数据上训练一个模型
            model_with_da = get_model(
                self.model_type,
                **model_config
            )
            
            model_with_da.fit(X_train, y_train)
            
            # 使用这个模型在域适应后的B数据的10折上评估
            results_with_da = evaluate_model_on_external_cv(
                model_with_da, X_test_adapted, y_test, cv_folds
            )
            
            logging.info(f"域适应完成，MMD减少: {adaptation_info.get('reduction', 0):.2f}%")
        else:
            results_with_da = results_no_da.copy()
        
        # 组织结果
        results = {
            'without_domain_adaptation': results_no_da,
            'with_domain_adaptation': results_with_da
        }
        
        # 计算改进幅度
        if self.use_mmd_adaptation and results_no_da != results_with_da:
            results['improvement'] = calculate_cross_domain_improvement(
                results_no_da, results_with_da
            )
        
        logging.info(f"数据集{dataset_name}正确10折CV结果:")
        log_external_cv_results(results_no_da, f"无域适应-{dataset_name}")
        if self.use_mmd_adaptation and results_no_da != results_with_da:
            log_external_cv_results(results_with_da, f"有域适应-{dataset_name}")
        
        return results
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """运行完整的跨域实验"""
        logging.info("开始运行完整跨域实验...")
        
        # 加载数据
        datasets = self.load_datasets()
        
        # 1. 数据集A上的交叉验证（可选）
        if not self.skip_cv_on_a:
            logging.info("运行数据集A上的10折交叉验证...")
            cv_results = self.run_cross_validation(
                datasets['X_A_scaled'], 
                datasets['y_A']
            )
            self.results['cross_validation_A'] = cv_results
        else:
            logging.info("跳过数据集A的交叉验证，直接进行A→B域适应实验")
            self.results['cross_validation_A'] = None
        
        # 2. A→目标域域适应评估（根据evaluation_mode选择评估方式）
        target_domain_name = self.target_domain
        
        if target_domain_name == 'B':
            X_target_scaled = datasets['X_B_scaled']
            y_target = datasets['y_B']
            X_target_raw = datasets['X_B_raw']
        else:  # target_domain_name == 'C'
            X_target_scaled = datasets['X_C_scaled']
            y_target = datasets['y_C']
            X_target_raw = datasets['X_C_raw']
        
        if self.evaluation_mode == 'single':
            # 单次评估模式（与原始脚本一致）
            target_results = self.evaluate_external_dataset_single(
                datasets['X_A_scaled'], datasets['y_A'],  # 训练数据（标准化后）
                X_target_scaled, y_target,  # 测试数据（标准化后）
                datasets['X_A_raw'], X_target_raw, # 原始数据（供域适应使用）
                target_domain_name
            )
        elif self.evaluation_mode == 'proper_cv':
            # 正确的10折CV评估模式（按照predict_healthcare_auto_external_adjust_parameter.py的方式）
            target_results = self.evaluate_external_dataset_proper_cv(
                datasets['X_A_scaled'], datasets['y_A'],  # 训练数据（标准化后）
                X_target_scaled, y_target,  # 测试数据（标准化后）
                datasets['X_A_raw'], X_target_raw, # 原始数据（供域适应使用）
                target_domain_name
            )
        else:
            # 原来的10折CV评估模式（每折重新训练模型）
            target_results = self.evaluate_external_dataset_cv(
                datasets['X_A_scaled'], datasets['y_A'],  # 训练数据（标准化后）
                X_target_scaled, y_target,  # 测试数据（标准化后）
                datasets['X_A_raw'], X_target_raw, # 原始数据（供域适应使用）
                target_domain_name
            )
        
        # 根据目标域设置结果
        if target_domain_name == 'B':
            self.results['external_validation_B'] = target_results
            # 如果目标域是B，则不评估C
            if datasets['has_C']:
                self.results['external_validation_C'] = None
        else:  # target_domain_name == 'C'
            self.results['external_validation_C'] = target_results
            # 如果目标域是C，则不评估B
            self.results['external_validation_B'] = None
        
        # 保存结果
        self.save_results()
        
        # 生成可视化
        if self.kwargs.get('save_visualizations', True):
            self.generate_visualizations(datasets)
        
        logging.info("跨域实验完成!")
        
        return self.results
    
    def save_results(self):
        """保存实验结果"""
        # 保存JSON格式结果
        results_file = os.path.join(self.save_path, 'experiment_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存文本格式结果
        text_file = os.path.join(self.save_path, 'results_summary.txt')
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("跨域实验结果总结\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"实验配置:\n")
            f.write(f"  模型类型: {self.model_type}\n")
            f.write(f"  特征类型: {self.feature_type}\n")
            f.write(f"  特征列表: {self.features}\n")
            f.write(f"  数据划分策略: {self.data_split_strategy}\n")
            if self.data_split_strategy == 'three-way':
                f.write(f"  验证集比例: {self.validation_split}\n")
            f.write(f"  MMD域适应: {self.use_mmd_adaptation}\n")
            if self.use_mmd_adaptation:
                f.write(f"  MMD方法: {self.mmd_method}\n")
                f.write(f"  类条件MMD: {self.use_class_conditional}\n")
            f.write(f"  阈值优化: {self.use_threshold_optimizer}\n")
            f.write(f"  目标域: {self.target_domain}\n\n")
            
            # 使用新的汇总函数
            summary_text = summarize_cross_domain_results(self.results)
            f.write(summary_text)
        
        logging.info(f"结果已保存到: {self.save_path}")
    
    def generate_visualizations(self, datasets: Dict[str, np.ndarray]):
        """生成可视化图表"""
        logging.info("生成可视化图表...")
        
        viz_dir = os.path.join(self.save_path, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        try:
            # 1. 使用新的性能对比图模块生成完整的性能对比图套件
            self._generate_comprehensive_performance_plots(viz_dir)
            
            # 2. 保留原有的简单性能对比图（向后兼容）
            self._plot_performance_comparison(viz_dir)
            
            # 3. 如果使用了域适应，生成域适应相关图表
            if self.use_mmd_adaptation:
                self._plot_domain_adaptation_results(datasets, viz_dir)
            
            logging.info(f"可视化图表已保存到: {viz_dir}")
            
        except Exception as e:
            logging.warning(f"生成可视化时出错: {e}")
    
    def _generate_comprehensive_performance_plots(self, viz_dir: str):
        """使用新的性能对比图模块生成完整的性能对比图套件"""
        try:
            # 准备结果数据，格式化为新模块期望的格式
            results_for_plots = {}
            
            # 添加当前实验的结果
            experiment_name = f"{self.model_type.upper()}_MMD_{self.mmd_method.upper()}"
            if self.use_class_conditional:
                experiment_name = f"{self.model_type.upper()}_ClassMMD_{self.mmd_method.upper()}"
            
            # 构建实验结果字典
            experiment_results = {}
            
            # 添加B数据集结果
            if 'external_validation_B' in self.results:
                experiment_results = self.results['external_validation_B']
            
            # 如果有C数据集结果，创建多数据集对比
            if 'external_validation_C' in self.results:
                # 为跨数据集对比准备数据
                results_for_plots[f"Dataset_B_{experiment_name}"] = self.results['external_validation_B']
                results_for_plots[f"Dataset_C_{experiment_name}"] = self.results['external_validation_C']
            else:
                # 单数据集结果
                results_for_plots[experiment_name] = experiment_results
            
            # 调用新的性能对比图生成函数
            if results_for_plots:
                generate_performance_comparison_plots(
                    results_dict=results_for_plots,
                    save_dir=viz_dir,
                    experiment_name=f"{self.model_type.upper()}_Model_Performance"
                )
                logging.info("✓ 完整性能对比图套件生成完成")
            else:
                logging.warning("没有足够的结果数据生成性能对比图")
                
        except Exception as e:
            logging.warning(f"生成完整性能对比图时出错: {e}")
            import traceback
            logging.debug(f"详细错误信息: {traceback.format_exc()}")
    
    def _plot_performance_comparison(self, viz_dir: str):
        """绘制性能对比图"""
        plt.figure(figsize=(12, 8))
        
        # 准备数据
        metrics_names = ['Accuracy', 'AUC', 'F1', 'Class 0 Acc', 'Class 1 Acc']
        
        # 提取数据
        cv_results = self.results['cross_validation_A']
        b_results = self.results['external_validation_B']
        
        # 解析交叉验证结果的均值（如果有的话）
        cv_values = []
        if cv_results is not None:
            for metric in ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']:
                value_str = cv_results[metric]
                mean_value = float(value_str.split(' ± ')[0])
                cv_values.append(mean_value)
        else:
            cv_values = [0] * 5  # 如果没有CV结果，使用0填充
        
        # 处理B数据集结果 - 修复结果访问方式
        if 'without_domain_adaptation' in b_results:
            b_no_da = b_results['without_domain_adaptation']
            if 'means' in b_no_da:
                # 如果是CV结果，使用means
                b_values = [b_no_da['means']['accuracy'], b_no_da['means']['auc'], 
                           b_no_da['means']['f1'], b_no_da['means']['acc_0'], b_no_da['means']['acc_1']]
            else:
                # 如果是单次评估结果，直接使用
                b_values = [b_no_da['accuracy'], b_no_da['auc'], b_no_da['f1'], 
                           b_no_da['acc_0'], b_no_da['acc_1']]
        else:
            b_values = [0] * 5  # 如果没有结果，使用0填充
        
        x = np.arange(len(metrics_names))
        width = 0.25
        
        if cv_results is not None:
            plt.bar(x - width, cv_values, width, label='Dataset A (CV)', alpha=0.8)
        plt.bar(x, b_values, width, label='Dataset B', alpha=0.8)
        
        # 如果有数据集C的结果
        if 'external_validation_C' in self.results:
            c_results = self.results['external_validation_C']
            if 'without_domain_adaptation' in c_results:
                c_no_da = c_results['without_domain_adaptation']
                if 'means' in c_no_da:
                    # 如果是CV结果，使用means
                    c_values = [c_no_da['means']['accuracy'], c_no_da['means']['auc'], 
                               c_no_da['means']['f1'], c_no_da['means']['acc_0'], c_no_da['means']['acc_1']]
                else:
                    # 如果是单次评估结果，直接使用
                    c_values = [c_no_da['accuracy'], c_no_da['auc'], c_no_da['f1'],
                               c_no_da['acc_0'], c_no_da['acc_1']]
                plt.bar(x + width, c_values, width, label='Dataset C', alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title(f'Performance Comparison - {self.model_type.upper()} Model')
        plt.xticks(x, metrics_names)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(os.path.join(viz_dir, 'performance_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_domain_adaptation_results(self, datasets: Dict[str, np.ndarray], viz_dir: str):
        """绘制域适应结果图"""
        logging.info("生成域适应相关可视化...")
        
        try:
            # 获取域适应前后的数据
            X_A_scaled = datasets['X_A_scaled']
            X_B_scaled = datasets['X_B_scaled']
            y_A = datasets['y_A']
            y_B = datasets['y_B']
            
            # 如果使用了MMD域适应，需要重新计算适应后的数据
            if self.use_mmd_adaptation:
                # 重新进行域适应以获取适应后的数据用于可视化
                X_B_adapted, adaptation_info = self.run_domain_adaptation(
                    X_A_scaled, y_A, X_B_scaled
                )
                
                # 生成A→B的t-SNE对比可视化
                method_name = f"{self.model_type.upper()}-MMD-{self.mmd_method.upper()}"
                if self.use_class_conditional:
                    method_name = f"{self.model_type.upper()}-ClassMMD-{self.mmd_method.upper()}"
                
                # 调用综合对比可视化函数
                comparison_results = compare_before_after_adaptation(
                    source_features=X_A_scaled,
                    target_features=X_B_scaled,
                    adapted_target_features=X_B_adapted,
                    source_labels=y_A,
                    target_labels=y_B,
                    save_dir=viz_dir,
                    method_name=method_name,
                    feature_names=self.features
                )
                
                logging.info(f"✓ {method_name} 域适应可视化完成")
                
                # 如果有数据集C，也生成A→C的可视化
                if datasets['has_C'] and datasets['X_C_raw'] is not None:
                    X_C_scaled = datasets['X_C_scaled']
                    y_C = datasets['y_C']
                    
                    # A→C域适应
                    X_C_adapted, _ = self.run_domain_adaptation(
                        X_A_scaled, y_A, X_C_scaled
                    )
                    
                    # 创建C数据集的可视化子目录
                    viz_c_dir = os.path.join(viz_dir, 'A_to_C')
                    os.makedirs(viz_c_dir, exist_ok=True)
                    
                    # 生成A→C的可视化
                    compare_before_after_adaptation(
                        source_features=X_A_scaled,
                        target_features=X_C_scaled,
                        adapted_target_features=X_C_adapted,
                        source_labels=y_A,
                        target_labels=y_C,
                        save_dir=viz_c_dir,
                        method_name=f"{method_name}_A_to_C",
                        feature_names=self.features
                    )
                    
                    logging.info(f"✓ {method_name} A→C 域适应可视化完成")
            
            else:
                logging.info("未使用域适应，跳过域适应可视化")
                
        except Exception as e:
            logging.error(f"生成域适应可视化时出错: {e}")
            import traceback
            logging.error(f"详细错误信息: {traceback.format_exc()}")

def run_cross_domain_experiment(model_type: str = 'auto',
                               feature_type: str = 'best7',
                               mmd_method: str = 'linear',
                               use_class_conditional: bool = False,
                               use_threshold_optimizer: bool = False,
                               save_path: str = './results_cross_domain',
                               skip_cv_on_a: bool = False,
                               evaluation_mode: str = 'cv',
                               data_split_strategy: str = 'two-way',
                               validation_split: Optional[float] = None,
                               target_domain: str = 'B',
                               **kwargs: Any) -> Dict[str, Any]:
    """
    运行跨域实验的便捷函数
    
    参数:
    - model_type: 模型类型
    - feature_type: 特征类型
    - mmd_method: MMD方法
    - use_class_conditional: 是否使用类条件MMD
    - use_threshold_optimizer: 是否使用阈值优化
    - save_path: 保存路径
    - skip_cv_on_a: 是否跳过数据集A的交叉验证
    - evaluation_mode: 外部评估模式 ('single': 单次评估, 'cv': 10折CV评估, 'proper_cv': 正确的10折CV)
    - data_split_strategy: 数据划分策略 ('two-way': 二分法, 'three-way': 三分法)
    - validation_split: 三分法时验证集比例
    - target_domain: 目标域选择: 'B' 或 'C'
    - **kwargs: 其他参数
    
    返回:
    - Dict[str, Any]: 实验结果
    """
    
    runner = CrossDomainExperimentRunner(
        model_type=model_type,
        feature_type=feature_type,
        use_mmd_adaptation=True,
        mmd_method=mmd_method,
        use_class_conditional=use_class_conditional,
        use_threshold_optimizer=use_threshold_optimizer,
        save_path=save_path,
        skip_cv_on_a=skip_cv_on_a,
        evaluation_mode=evaluation_mode,
        data_split_strategy=data_split_strategy,
        validation_split=validation_split,
        target_domain=target_domain,
        **kwargs
    )
    
    return runner.run_full_experiment() 