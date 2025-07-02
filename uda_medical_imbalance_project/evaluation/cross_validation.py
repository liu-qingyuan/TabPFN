"""
交叉验证模块
实现10折交叉验证，集成RFE特征选择、标准化、不平衡处理和多种模型对比
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, 
    precision_score, recall_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SMOTENC, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from tabpfn import TabPFNClassifier
except ImportError:
    print("Warning: TabPFN not available. Using RandomForest as fallback.")
    from sklearn.ensemble import RandomForestClassifier as TabPFNClassifier

from config.settings import (
    BEST_7_FEATURES, BEST_8_FEATURES, BEST_9_FEATURES, 
    BEST_10_FEATURES, SELECTED_FEATURES,
    get_categorical_features, get_features_by_type
)
from preprocessing.imbalance_handler import ImbalanceHandler
from preprocessing.scalers import create_scaler
from modeling.baseline_models import get_baseline_model
from modeling.paper_methods import get_paper_method


class CrossValidationEvaluator:
    """
    交叉验证评估器
    支持RFE特征选择、标准化、不平衡处理和多种模型的10折交叉验证
    """
    
    def __init__(
        self,
        model_type: str = 'tabpfn',
        feature_set: str = 'best7',
        scaler_type: str = 'standard',
        imbalance_method: str = 'smote',
        model_params: Optional[Dict] = None,
        cv_folds: int = 10,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        初始化交叉验证评估器
        
        Args:
            model_type: 模型类型 ('tabpfn', 'pkuph', 'mayo', 'paper_lr')
            feature_set: 特征集选择 ('best7', 'best8', 'best9', 'best10', 'all')
            scaler_type: 标准化类型 ('standard', 'robust', 'none')
            imbalance_method: 不平衡处理方法
            model_params: 模型参数
            cv_folds: 交叉验证折数
            random_state: 随机种子
            verbose: 是否输出详细信息
        """
        self.model_type = model_type
        self.feature_set = feature_set
        self.scaler_type = scaler_type
        self.imbalance_method = imbalance_method
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        
        # 获取特征列表
        self.features = self._get_feature_list()
        self.categorical_features = get_categorical_features(feature_set)
        
        # 模型参数
        self.model_params = model_params or {}
        
        # 初始化不平衡处理器
        self.imbalance_handler = ImbalanceHandler()
        
        # 存储结果
        self.fold_results = []
        self.summary_results = {}
        
    def _get_feature_list(self) -> List[str]:
        """获取特征列表"""
        # 基线模型和论文方法使用各自定义的特征
        if self.model_type == 'pkuph':
            # PKUPH模型使用固定的6个特征
            return ['Feature2', 'Feature48', 'Feature49', 'Feature4', 'Feature50', 'Feature53']
        elif self.model_type == 'mayo':
            # Mayo模型使用固定的6个特征
            return ['Feature2', 'Feature3', 'Feature5', 'Feature48', 'Feature49', 'Feature63']
        elif self.model_type == 'paper_lr':
            # 论文LR方法使用固定的11个特征
            return ['Feature2', 'Feature5', 'Feature48', 'Feature49', 'Feature50', 
                   'Feature52', 'Feature56', 'Feature57', 'Feature61', 'Feature42', 'Feature43']
        else:
            # TabPFN等其他模型使用通用特征集
            feature_sets = {
                'best7': BEST_7_FEATURES,
                'best8': BEST_8_FEATURES,
                'best9': BEST_9_FEATURES,
                'best10': BEST_10_FEATURES,
                'all63': get_features_by_type('all63'),
                'selected58': SELECTED_FEATURES
            }
            return feature_sets.get(self.feature_set, BEST_7_FEATURES)
    
    def _should_apply_preprocessing(self) -> bool:
        """判断是否需要应用预处理（标准化和不平衡处理）"""
        # 传统医疗基线模型和论文方法不需要预处理，其他模型（TabPFN和机器学习模型）需要预处理
        return self.model_type not in ['pkuph', 'mayo', 'paper_lr']
    
    def _create_scaler(self):
        """创建标准化器"""
        if not self._should_apply_preprocessing():
            # 基线模型和论文方法不需要标准化
            return create_scaler('none', self.categorical_features)
        else:
            # TabPFN等模型需要标准化
            return create_scaler(self.scaler_type, self.categorical_features)
    
    def _create_imbalance_sampler(self):
        """创建不平衡处理采样器"""
        if self.imbalance_method == 'none' or not self._should_apply_preprocessing():
            return None
            
        # 获取类别特征索引（相对于选定特征的索引）
        categorical_indices = []
        if self.categorical_features:
            for cat_feature in self.categorical_features:
                if cat_feature in self.features:
                    categorical_indices.append(self.features.index(cat_feature))
        
        samplers = {
            'smote': SMOTE(random_state=self.random_state),
            'smotenc': SMOTENC(
                categorical_features=categorical_indices if categorical_indices else [0],
                random_state=self.random_state
            ) if categorical_indices else SMOTE(random_state=self.random_state),
            'borderline_smote': BorderlineSMOTE(random_state=self.random_state),
            'kmeans_smote': KMeansSMOTE(random_state=self.random_state),
            'svm_smote': SVMSMOTE(random_state=self.random_state),
            'adasyn': ADASYN(random_state=self.random_state),
            'random_under': RandomUnderSampler(random_state=self.random_state),
            'edited_nn': EditedNearestNeighbours(sampling_strategy='all'),
        }
        
        # 组合方法需要特殊处理
        if self.imbalance_method == 'smote_tomek':
            base_smote = SMOTE(random_state=self.random_state)
            return SMOTETomek(smote=base_smote, random_state=self.random_state)
            
        elif self.imbalance_method == 'smote_enn':
            base_smote = SMOTE(random_state=self.random_state)
            return SMOTEENN(smote=base_smote, random_state=self.random_state)
        
        return samplers.get(self.imbalance_method, SMOTE(random_state=self.random_state))
    
    def _create_model(self):
        """创建模型"""
        if self.model_type == 'tabpfn':
            return self._create_tabpfn_model()
        elif self.model_type in ['pkuph', 'mayo']:
            return get_baseline_model(self.model_type)
        elif self.model_type == 'paper_lr':
            return get_paper_method('paper_lr')
        elif self.model_type in ['svm', 'dt', 'rf', 'gbdt', 'xgboost']:
            # 机器学习基线模型
            from modeling.ml_baseline_models import get_ml_baseline_model
            return get_ml_baseline_model(
                model_type=self.model_type,
                random_state=self.random_state,
                verbose=self.verbose
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _create_tabpfn_model(self):
        """创建TabPFN模型"""
        try:
            # 获取类别特征索引（相对于选定特征的索引）
            categorical_indices = []
            if self.categorical_features:
                for cat_feature in self.categorical_features:
                    if cat_feature in self.features:
                        categorical_indices.append(self.features.index(cat_feature))
            
            tabpfn_params = {
                'n_estimators': self.model_params.get('n_estimators', 32),
                'categorical_features_indices': categorical_indices if categorical_indices else None,
                'softmax_temperature': self.model_params.get('softmax_temperature', 0.9),
                'balance_probabilities': self.model_params.get('balance_probabilities', False),
                'average_before_softmax': self.model_params.get('average_before_softmax', False),
                'model_path': self.model_params.get('model_path', 'auto'),
                'device': self.model_params.get('device', 'auto'),
                'ignore_pretraining_limits': self.model_params.get('ignore_pretraining_limits', True),
                'inference_precision': self.model_params.get('inference_precision', 'auto'),
                'fit_mode': self.model_params.get('fit_mode', 'fit_preprocessors'),
                'memory_saving_mode': self.model_params.get('memory_saving_mode', 'auto'),
                'random_state': self.model_params.get('random_state', self.random_state),
                'n_jobs': self.model_params.get('n_jobs', -1),
                'inference_config': self.model_params.get('inference_config', None)
            }
            
            if self.verbose:
                print(f"TabPFN参数: n_estimators={tabpfn_params['n_estimators']}, "
                      f"categorical_features={len(categorical_indices) if categorical_indices else 0}, "
                      f"device={tabpfn_params['device']}")
            
            return TabPFNClassifier(**tabpfn_params)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to create TabPFN model ({e}). Using RandomForest fallback.")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: Optional[np.ndarray]) -> Dict:
        """计算评估指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
        }
        
        # 计算AUC（如果有概率预测）
        if y_pred_proba is not None and len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        elif y_pred_proba is not None and len(y_pred_proba.shape) == 1:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics['auc'] = np.nan
            
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def run_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        运行10折交叉验证
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            交叉验证结果字典
        """
        if self.verbose:
            print(f"开始10折交叉验证...")
            print(f"模型类型: {self.model_type}")
            print(f"特征集: {self.feature_set} ({len(self.features)}个特征)")
            print(f"指定特征列表: {self.features}")
            if self._should_apply_preprocessing():
                print(f"标准化: {self.scaler_type}")
                print(f"不平衡处理: {self.imbalance_method}")
            else:
                print(f"标准化: 跳过（{self.model_type}模型不需要预处理）")
                print(f"不平衡处理: 跳过（{self.model_type}模型不需要预处理）")
            print(f"类别特征: {len(self.categorical_features)}个")
            print("-" * 60)
        
        # 确保数据是DataFrame格式
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        
        if isinstance(y, np.ndarray):
            y_selected = pd.Series(y)
        else:
            y_selected = y.copy()
        
        # 根据模型类型选择对应的特征
        required_features = self.features  # 从_get_feature_list()获取的特征列表
        
        # 检查所需特征是否存在于数据中
        missing_features = [f for f in required_features if f not in X_df.columns]
        if missing_features:
            raise ValueError(f"数据中缺少以下特征: {missing_features}")
        
        # 选择所需的特征
        X_selected = X_df[required_features].copy()
        
        if self.verbose:
            print(f"选择的特征: {required_features}")
            print(f"特征数据形状: {X_selected.shape}")
            print(f"✓ 特征选择完成: 从{X_df.shape[1]}个特征中选择了{X_selected.shape[1]}个特征")
            print(f"✓ 使用的特征: {list(X_selected.columns)}")
        
        # 初始化交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # 存储每折结果
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        all_y_pred_proba = []
        
        start_time = time.time()
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_selected, y_selected), 1):
            fold_start_time = time.time()
            
            # 分割数据
            X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
            y_train, y_test = y_selected.iloc[train_idx], y_selected.iloc[test_idx]
            
            if self.verbose:
                print(f"\nFold {fold}/{self.cv_folds}")
                print(f"训练集: {len(X_train)}样本, 测试集: {len(X_test)}样本")
                print(f"训练集类别分布: {dict(pd.Series(y_train).value_counts().sort_index())}")
            
            # 步骤1: 标准化处理
            scaler = self._create_scaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 转换回DataFrame以保持特征名称
            X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            
            if self.verbose and self._should_apply_preprocessing():
                print(f"标准化完成: {self.scaler_type}")
            
            # 步骤2: 不平衡处理（仅对训练集，且仅对需要预处理的模型）
            if self._should_apply_preprocessing() and self.imbalance_method != 'none':
                sampler = self._create_imbalance_sampler()
                if sampler is not None:
                    try:
                        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
                        # 确保返回值是正确的类型
                        if isinstance(X_train_resampled, tuple):
                            X_train_resampled = X_train_resampled[0]
                        if isinstance(y_train_resampled, tuple):
                            y_train_resampled = y_train_resampled[0]
                        # 转换回DataFrame
                        if isinstance(X_train_resampled, np.ndarray):
                            X_train_resampled = pd.DataFrame(
                                X_train_resampled, 
                                columns=X_train_scaled.columns
                            )
                        if self.verbose:
                            print(f"重采样后训练集: {len(X_train_resampled)}样本")
                            print(f"重采样后类别分布: {dict(pd.Series(y_train_resampled).value_counts().sort_index())}")
                    except Exception as e:
                        if self.verbose:
                            print(f"重采样失败，使用原始训练集: {e}")
                        X_train_resampled, y_train_resampled = X_train_scaled, y_train
                else:
                    X_train_resampled, y_train_resampled = X_train_scaled, y_train
            else:
                X_train_resampled, y_train_resampled = X_train_scaled, y_train
            
            # 步骤3: 训练模型
            model = self._create_model()
            try:
                model.fit(X_train_resampled, y_train_resampled)
                
                # 预测
                y_pred = model.predict(X_test_scaled)
                try:
                    y_pred_proba = model.predict_proba(X_test_scaled)
                except:
                    y_pred_proba = None
                
                # 计算指标
                metrics = self._calculate_metrics(y_test.values, y_pred, y_pred_proba)
                metrics['fold'] = fold
                metrics['train_time'] = time.time() - fold_start_time
                
                fold_metrics.append(metrics)
                all_y_true.extend(y_test.tolist())
                all_y_pred.extend(y_pred.tolist())
                if y_pred_proba is not None:
                    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                        all_y_pred_proba.extend(y_pred_proba[:, 1].tolist())
                    else:
                        all_y_pred_proba.extend(y_pred_proba.tolist())
                
                if self.verbose:
                    print(f"AUC: {metrics['auc']:.4f}, "
                          f"Accuracy: {metrics['accuracy']:.4f}, "
                          f"F1: {metrics['f1']:.4f}")
                    print(f"训练时间: {metrics['train_time']:.2f}秒")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Fold {fold} 训练失败: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # 计算总体指标
        if fold_metrics:
            # 计算各指标的均值和标准差
            metric_names = ['accuracy', 'auc', 'f1', 'precision', 'recall']
            summary = {}
            
            for metric in metric_names:
                values = [m[metric] for m in fold_metrics if not np.isnan(m[metric])]
                if values:
                    summary[f'{metric}_mean'] = np.mean(values)
                    summary[f'{metric}_std'] = np.std(values)
                else:
                    summary[f'{metric}_mean'] = np.nan
                    summary[f'{metric}_std'] = np.nan
            
            # 总体混淆矩阵
            if all_y_true and all_y_pred:
                overall_cm = confusion_matrix(all_y_true, all_y_pred)
                summary['overall_confusion_matrix'] = overall_cm
                
                # 总体指标
                summary['overall_accuracy'] = accuracy_score(all_y_true, all_y_pred)
                summary['overall_f1'] = f1_score(all_y_true, all_y_pred, average='binary')
                summary['overall_precision'] = precision_score(all_y_true, all_y_pred, average='binary')
                summary['overall_recall'] = recall_score(all_y_true, all_y_pred, average='binary')
                
                if all_y_pred_proba:
                    summary['overall_auc'] = roc_auc_score(all_y_true, all_y_pred_proba)
            
            summary['total_time'] = total_time
            summary['n_folds'] = len(fold_metrics)
            
            # 存储结果
            self.fold_results = fold_metrics
            self.summary_results = summary
            
            if self.verbose:
                self._print_summary()
            
            return {
                'fold_results': fold_metrics,
                'summary': summary,
                'predictions': {
                    'y_true': all_y_true,
                    'y_pred': all_y_pred,
                    'y_pred_proba': all_y_pred_proba if all_y_pred_proba else None
                }
            }
        else:
            if self.verbose:
                print("所有折都失败了！")
            return {'fold_results': [], 'summary': {}, 'predictions': {}}
    
    def _print_summary(self):
        """打印结果摘要"""
        print("\n" + "="*60)
        print("交叉验证结果摘要")
        print("="*60)
        
        summary = self.summary_results
        
        print(f"模型类型: {self.model_type}")
        print(f"特征集: {self.feature_set} ({len(self.features)}个特征)")
        if self._should_apply_preprocessing():
            print(f"标准化: {self.scaler_type}")
            print(f"不平衡处理: {self.imbalance_method}")
        else:
            print(f"标准化: 跳过（{self.model_type}模型不需要预处理）")
            print(f"不平衡处理: 跳过（{self.model_type}模型不需要预处理）")
        print(f"成功完成折数: {summary.get('n_folds', 0)}/{self.cv_folds}")
        print(f"总耗时: {summary.get('total_time', 0):.2f}秒")
        print()
        
        # 各指标的均值±标准差
        metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
        print("各折指标 (均值 ± 标准差):")
        for metric in metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in summary and not np.isnan(summary[mean_key]):
                print(f"{metric.upper():>10}: {summary[mean_key]:.4f} ± {summary[std_key]:.4f}")
        
        print()
        
        # 总体指标（基于合并预测的重新计算，仅供参考）
        if 'overall_accuracy' in summary:
            print("总体指标（基于合并预测重新计算，仅供参考）:")
            print(f"{'Accuracy':>10}: {summary['overall_accuracy']:.4f}")
            if 'overall_auc' in summary:
                print(f"{'AUC':>10}: {summary['overall_auc']:.4f}")
            print(f"{'F1':>10}: {summary['overall_f1']:.4f}")
            print(f"{'Precision':>10}: {summary['overall_precision']:.4f}")
            print(f"{'Recall':>10}: {summary['overall_recall']:.4f}")
            print("注意：标准做法应使用各折指标的平均值作为最终结果")
        
        # 混淆矩阵
        if 'overall_confusion_matrix' in summary:
            print("\n总体混淆矩阵:")
            cm = summary['overall_confusion_matrix']
            print(f"[[{cm[0,0]:4d}, {cm[0,1]:4d}]")
            print(f" [{cm[1,0]:4d}, {cm[1,1]:4d}]]")
        
        print("="*60)
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Optional[Dict]:
        """
        获取特征重要性（如果模型支持）
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            特征重要性字典
        """
        try:
            # 数据加载器已经选择了正确的特征
            X_selected = X.copy()
            
            # 标准化
            scaler = self._create_scaler()
            X_scaled = scaler.fit_transform(X_selected)
            X_scaled = pd.DataFrame(X_scaled, columns=X_selected.columns)
            
            # 不平衡处理
            if self._should_apply_preprocessing() and self.imbalance_method != 'none':
                sampler = self._create_imbalance_sampler()
                if sampler is not None:
                    X_resampled, y_resampled = sampler.fit_resample(X_scaled, y)
                    # 确保返回值是正确的类型
                    if isinstance(X_resampled, tuple):
                        X_resampled = X_resampled[0]
                    if isinstance(y_resampled, tuple):
                        y_resampled = y_resampled[0]
                    if isinstance(X_resampled, np.ndarray):
                        X_resampled = pd.DataFrame(X_resampled, columns=X_scaled.columns)
                else:
                    X_resampled, y_resampled = X_scaled, y
            else:
                X_resampled, y_resampled = X_scaled, y
            
            # 训练模型
            model = self._create_model()
            model.fit(X_resampled, y_resampled)
            
            # 获取特征重要性
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(self.features, model.feature_importances_))
                return importance_dict
            elif hasattr(model, 'coef_'):
                # 对于线性模型，使用系数的绝对值作为重要性
                importance_dict = dict(zip(self.features, np.abs(model.coef_[0])))
                return importance_dict
            else:
                if self.verbose:
                    print("模型不支持特征重要性分析")
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"特征重要性分析失败: {e}")
            return None


def run_cv_experiment(
    X: pd.DataFrame, 
    y: pd.Series,
    model_types: Optional[List[str]] = None,
    feature_sets: Optional[List[str]] = None,
    scaler_types: Optional[List[str]] = None,
    imbalance_methods: Optional[List[str]] = None,
    cv_folds: int = 10,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    运行多种配置的交叉验证实验
    
    Args:
        X: 特征数据
        y: 标签数据
        model_types: 模型类型列表
        feature_sets: 特征集列表
        scaler_types: 标准化类型列表
        imbalance_methods: 不平衡处理方法列表
        cv_folds: 交叉验证折数
        random_state: 随机种子
        verbose: 是否输出详细信息
        
    Returns:
        实验结果字典
    """
    # 设置默认值
    if model_types is None:
        model_types = ['tabpfn', 'pkuph', 'mayo', 'paper_lr']
    
    if feature_sets is None:
        feature_sets = ['best7', 'best8', 'best9', 'best10']
    
    if scaler_types is None:
        scaler_types = ['standard', 'robust', 'none']
    
    if imbalance_methods is None:
        imbalance_methods = [
            'none', 'smote', 'smotenc', 'borderline_smote', 
            'kmeans_smote', 'svm_smote', 'adasyn', 
            'smote_tomek', 'smote_enn', 'random_under', 'edited_nn'
        ]
    
    results = {}
    
    for model_type in model_types:
        for feature_set in feature_sets:
            # 基线模型和论文方法只测试无预处理的情况
            if model_type in ['pkuph', 'mayo', 'paper_lr']:
                test_scaler_types = ['none']
                test_imbalance_methods = ['none']
            else:
                test_scaler_types = scaler_types
                test_imbalance_methods = imbalance_methods
            
            for scaler_type in test_scaler_types:
                for imbalance_method in test_imbalance_methods:
                    experiment_name = f"{model_type}_{feature_set}_{scaler_type}_{imbalance_method}"
                    
                    if verbose:
                        print(f"\n{'='*80}")
                        print(f"实验: {experiment_name}")
                        print(f"{'='*80}")
                    
                    evaluator = CrossValidationEvaluator(
                        model_type=model_type,
                        feature_set=feature_set,
                        scaler_type=scaler_type,
                        imbalance_method=imbalance_method,
                        cv_folds=cv_folds,
                        random_state=random_state,
                        verbose=verbose
                    )
                    
                    result = evaluator.run_cross_validation(X, y)
                    results[experiment_name] = result
    
    return results


def run_model_comparison_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_set: str = 'best7',
    scaler_type: str = 'standard',
    imbalance_method: str = 'smote',
    cv_folds: int = 10,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    运行模型对比的交叉验证实验
    
    Args:
        X: 特征数据
        y: 标签数据
        feature_set: 特征集选择（对TabPFN和机器学习模型有效，传统基线模型使用各自的特征）
        scaler_type: 标准化类型（对TabPFN和机器学习模型有效）
        imbalance_method: 不平衡处理方法（对TabPFN和机器学习模型有效）
        cv_folds: 交叉验证折数
        random_state: 随机种子
        verbose: 是否输出详细信息
        
    Returns:
        模型对比结果字典
    """
    model_configs = [
        # TabPFN模型（使用指定的特征集和预处理）
        {'model_type': 'tabpfn', 'feature_set': feature_set, 'scaler_type': scaler_type, 'imbalance_method': imbalance_method},
        
        # 机器学习基线模型（使用相同的特征集和预处理）
        {'model_type': 'svm', 'feature_set': feature_set, 'scaler_type': scaler_type, 'imbalance_method': imbalance_method},
        {'model_type': 'dt', 'feature_set': feature_set, 'scaler_type': scaler_type, 'imbalance_method': imbalance_method},
        {'model_type': 'rf', 'feature_set': feature_set, 'scaler_type': scaler_type, 'imbalance_method': imbalance_method},
        {'model_type': 'gbdt', 'feature_set': feature_set, 'scaler_type': scaler_type, 'imbalance_method': imbalance_method},
        {'model_type': 'xgboost', 'feature_set': feature_set, 'scaler_type': scaler_type, 'imbalance_method': imbalance_method},
        
        # 传统医疗基线模型（使用各自的特征集，无预处理）
        {'model_type': 'pkuph', 'feature_set': 'selected58', 'scaler_type': 'none', 'imbalance_method': 'none'},
        {'model_type': 'mayo', 'feature_set': 'selected58', 'scaler_type': 'none', 'imbalance_method': 'none'},
        {'model_type': 'paper_lr', 'feature_set': 'selected58', 'scaler_type': 'none', 'imbalance_method': 'none'}
    ]
    
    results = {}
    
    for config in model_configs:
        model_type = config['model_type']
        model_feature_set = config['feature_set']
        model_scaler_type = config['scaler_type']
        imbalance_method = config['imbalance_method']
        experiment_name = f"{model_type}_{model_feature_set}"
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"模型对比实验: {experiment_name}")
            print(f"{'='*80}")
        
        evaluator = CrossValidationEvaluator(
            model_type=model_type,
            feature_set=model_feature_set,
            scaler_type=model_scaler_type,
            imbalance_method=imbalance_method,
            cv_folds=cv_folds,
            random_state=random_state,
            verbose=verbose
        )
        
        result = evaluator.run_cross_validation(X, y)
        results[experiment_name] = result
    
    # 输出对比摘要
    if verbose:
        print(f"\n{'='*80}")
        print("模型对比结果摘要")
        print(f"{'='*80}")
        print(f"{'模型':<15} {'AUC':<10} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 80)
        
        for experiment_name, result in results.items():
            # 从实验名称中提取模型名称（去掉特征集后缀）
            if '_all' in experiment_name:
                model_name = experiment_name.replace('_all', '')
            elif f'_{feature_set}' in experiment_name:
                model_name = experiment_name.replace(f'_{feature_set}', '')
            else:
                model_name = experiment_name
                
            if 'summary' in result and result['summary']:
                summary = result['summary']
                auc = summary.get('auc_mean', 0)
                acc = summary.get('accuracy_mean', 0)
                f1 = summary.get('f1_mean', 0)
                prec = summary.get('precision_mean', 0)
                rec = summary.get('recall_mean', 0)
                
                print(f"{model_name:<15} {auc:<10.4f} {acc:<10.4f} {f1:<10.4f} {prec:<10.4f} {rec:<10.4f}")
            else:
                print(f"{model_name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
        
        print("="*80)
    
    return results 