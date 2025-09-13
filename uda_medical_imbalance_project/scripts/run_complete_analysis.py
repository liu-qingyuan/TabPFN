#!/usr/bin/env python3
"""
完整的医疗数据UDA分析流程

这个脚本提供完整的分析流程：
1. 源域10折交叉验证对比（TabPFN、论文方法、基线模型）
2. UDA域适应方法对比（基于ADAPT库）
3. 可视化分析和结果对比

运行示例: python scripts/run_complete_analysis.py
"""

# TODO: 优化流程 将项目模块化 提高代码复用性

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.loader import MedicalDataLoader
from uda.adapt_methods import is_adapt_available


class CompleteAnalysisRunner:
    """完整分析流程运行器"""
    
    def __init__(
        self,
        feature_type: str = 'best8',
        scaler_type: str = 'none',  # 不使用标准化
        imbalance_method: str = 'none',  # 不使用不平衡处理
        cv_folds: int = 10,
        random_state: int = 42,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        初始化分析运行器
        
        Args:
            feature_type: 特征集类型 ('all63', 'selected58', 'best3', 'best4', ..., 'best58')
            scaler_type: 标准化方法 ('standard', 'robust', 'none')
            imbalance_method: 不平衡处理方法
            cv_folds: 交叉验证折数
            random_state: 随机种子
            output_dir: 输出目录
            verbose: 是否输出详细信息
        """
        self.feature_type = feature_type
        self.scaler_type = scaler_type
        self.imbalance_method = imbalance_method
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose
        
        # 添加参数确认输出
        if self.verbose:
            print(f"🔧 CompleteAnalysisRunner 初始化确认:")
            print(f"   实际接收的 feature_type: '{self.feature_type}'")
            print(f"   实际接收的 scaler_type: '{self.scaler_type}'")
            print(f"   实际接收的 imbalance_method: '{self.imbalance_method}'")
            print(f"   实际接收的 cv_folds: {self.cv_folds}")
            print(f"   实际接收的 random_state: {self.random_state}")
            print(f"   输出目录: {output_dir}")
            print()
        
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/complete_analysis_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储结果
        self.results = {
            'config': {
                'feature_type': feature_type,
                'scaler_type': scaler_type,
                'imbalance_method': imbalance_method,
                'cv_folds': cv_folds,
                'random_state': random_state
            },
            'source_domain_cv': {},
            'uda_methods': {},
            'visualizations': {}
        }
        
        if self.verbose:
            print(f"🔧 完整分析流程初始化")
            print(f"   特征集: {feature_type}")
            print(f"   标准化: {scaler_type}")
            print(f"   不平衡处理: {imbalance_method}")
            print(f"   交叉验证: {cv_folds}折")
            print(f"   输出目录: {output_dir}")
    
    def load_data_for_cv(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """加载医疗数据"""
        if self.verbose:
            print(f"\n📊 加载医疗数据...")
        
        loader = MedicalDataLoader()
        
        try:
            # 为了支持所有模型，需要加载包含所有特征的数据集
            # TabPFN会从中选择子集特征，基线模型使用全部特征
            # 注意：数据集B只有58个特征，所以使用selected58特征集
            
            # 首先尝试加载selected58特征集（支持所有模型）
            data_A = loader.load_dataset('A', feature_type='selected58')
            data_B = loader.load_dataset('B', feature_type='selected58')
            
            # 提取特征和标签
            X_A = pd.DataFrame(data_A['X'], columns=data_A['feature_names'])
            y_A = pd.Series(data_A['y'])
            X_B = pd.DataFrame(data_B['X'], columns=data_B['feature_names'])
            y_B = pd.Series(data_B['y'])
            
            # 验证特征一致性
            if data_A['feature_names'] != data_B['feature_names']:
                raise ValueError(f"源域和目标域特征不一致:\n源域: {data_A['feature_names']}\n目标域: {data_B['feature_names']}")
            
            # 使用原始特征顺序，保持一致性
            common_features = data_A['feature_names']
            
            if self.verbose:
                print(f"✅ 数据加载完成:")
                print(f"   源域A: {X_A.shape}, 类别分布: {dict(y_A.value_counts().sort_index())}")
                print(f"   目标域B: {X_B.shape}, 类别分布: {dict(y_B.value_counts().sort_index())}")
                print(f"   加载特征集: selected58 (支持所有模型)")
                print(f"   TabPFN将从中选择: {self.feature_type} 特征")
                print(f"   基线模型将使用: selected58 特征")
                print(f"   特征总数: {len(common_features)}")
            
            return X_A.values, y_A.values.astype(int), X_B.values, y_B.values.astype(int), common_features
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 加载指定特征集失败: {e}")
                print(f"   尝试使用all63特征集作为备选...")
            
            # 备选方案：如果selected58不可用，尝试使用best8
            try:
                fallback_feature_type = 'best8'
                print(f"🚨 警告: selected58特征集加载失败!")
                print(f"🔄 自动切换到备选特征集: '{fallback_feature_type}'")
                print(f"💡 这会影响最终结果的准确性!")
                if self.verbose:
                    print(f"   尝试使用{fallback_feature_type}特征集作为备选...")
                
                # 更新配置记录，标记使用了fallback - 但保持原始feature_type不变
                self.results['config']['cv_original_requested'] = 'selected58' 
                self.results['config']['cv_actual_used'] = fallback_feature_type
                self.results['config']['cv_fallback_used'] = True
                # 保持原始feature_type不变，用于UDA分析
                
                data_A = loader.load_dataset('A', feature_type=fallback_feature_type)
                data_B = loader.load_dataset('B', feature_type=fallback_feature_type)
                
                # 提取特征和标签
                X_A = pd.DataFrame(data_A['X'], columns=data_A['feature_names'])
                y_A = pd.Series(data_A['y'])
                X_B = pd.DataFrame(data_B['X'], columns=data_B['feature_names'])
                y_B = pd.Series(data_B['y'])
                
                # 验证特征一致性
                if data_A['feature_names'] != data_B['feature_names']:
                    raise ValueError(f"源域和目标域特征不一致:\n源域: {data_A['feature_names']}\n目标域: {data_B['feature_names']}")
                
                common_features = data_A['feature_names']
                
                if self.verbose:
                    print(f"✅ 使用{fallback_feature_type}特征集加载完成:")
                    print(f"   源域A: {X_A.shape}, 类别分布: {dict(y_A.value_counts().sort_index())}")
                    print(f"   目标域B: {X_B.shape}, 类别分布: {dict(y_B.value_counts().sort_index())}")
                    print(f"   特征数量: {len(common_features)}")
                
                return X_A.values, y_A.values.astype(int), X_B.values, y_B.values.astype(int), common_features
                
            except Exception as e2:
                raise RuntimeError(f"数据加载失败，尝试了selected58和{fallback_feature_type}特征集都失败:\n原始错误: {e}\n备选错误: {e2}")
    
    def load_data_for_uda(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        专门为UDA分析加载和预处理数据
        包括特征筛选、标准化和不平衡处理
        """
        if self.verbose:
            print(f"\n📊 为UDA分析加载和预处理数据")
            print("=" * 50)
        
        from data.loader import MedicalDataLoader
        loader = MedicalDataLoader()
        
        try:
            # 加载指定特征集的数据
            if self.verbose:
                print(f"   加载特征集: {self.feature_type}")
            
            if self.verbose:
                print(f"🔍 load_uda_data: 使用 feature_type='{self.feature_type}' 加载数据...")
            data_A = loader.load_dataset('A', feature_type=self.feature_type)
            data_B = loader.load_dataset('B', feature_type=self.feature_type)
            
            # 提取特征和标签
            X_A = pd.DataFrame(data_A['X'], columns=data_A['feature_names'])
            y_A = pd.Series(data_A['y'])
            X_B = pd.DataFrame(data_B['X'], columns=data_B['feature_names'])
            y_B = pd.Series(data_B['y'])
            
            # 验证特征一致性
            if data_A['feature_names'] != data_B['feature_names']:
                raise ValueError(f"源域和目标域特征不一致:\n源域: {data_A['feature_names']}\n目标域: {data_B['feature_names']}")
            
            # 确保A和B数据集使用相同的特征列（特征对齐）
            common_features = list(set(X_A.columns) & set(X_B.columns))
            if len(common_features) != len(X_A.columns) or len(common_features) != len(X_B.columns):
                if self.verbose:
                    print(f"⚠ 警告: A和B数据集特征不完全一致")
                    print(f"  A特征: {list(X_A.columns)}")
                    print(f"  B特征: {list(X_B.columns)}")
                    print(f"  共同特征: {common_features}")
                # 使用共同特征
                X_A = X_A[common_features]
                X_B = X_B[common_features]
                common_features = list(X_A.columns)
            else:
                common_features = list(X_A.columns)
            
            if self.verbose:
                print(f"✅ 原始数据加载完成:")
                print(f"   源域A: {X_A.shape}, 类别分布: {dict(y_A.value_counts().sort_index())}")
                print(f"   目标域B: {X_B.shape}, 类别分布: {dict(y_B.value_counts().sort_index())}")
                print(f"   特征列表: {common_features}")
                print(f"   特征数量: {len(common_features)}")
            
            # 转换为numpy数组
            X_source = X_A.values
            y_source = y_A.values.astype(int)
            X_target = X_B.values
            y_target = y_B.values.astype(int)
            
            # 应用预处理（标准化和不平衡处理）
            X_source_processed, y_source_processed, X_target_processed = self._preprocess_uda_data(
                X_source, y_source, X_target, common_features
            )
            
            return X_source_processed, y_source_processed, X_target_processed, y_target.astype(int), common_features
            
        except Exception as e:
            # 备选方案：如果指定特征集不可用，尝试使用best8
            try:
                fallback_feature_type = 'best8' if self.feature_type != 'best8' else 'best7'
                print(f"🚨 警告: 原始特征集 '{self.feature_type}' 加载失败!")
                print(f"🔄 自动切换到备选特征集: '{fallback_feature_type}'")
                print(f"💡 这会影响最终结果的准确性!")
                if self.verbose:
                    print(f"   尝试使用{fallback_feature_type}特征集作为备选...")
                
                # 更新配置记录，标记使用了fallback - 但保持原始feature_type不变
                self.results['config']['uda_original_requested'] = self.feature_type
                self.results['config']['uda_actual_used'] = fallback_feature_type
                self.results['config']['uda_fallback_used'] = True
                # 保持原始feature_type不变，用于正确的配置记录
                
                data_A = loader.load_dataset('A', feature_type=fallback_feature_type)
                data_B = loader.load_dataset('B', feature_type=fallback_feature_type)
                
                # 提取特征和标签
                X_A = pd.DataFrame(data_A['X'], columns=data_A['feature_names'])
                y_A = pd.Series(data_A['y'])
                X_B = pd.DataFrame(data_B['X'], columns=data_B['feature_names'])
                y_B = pd.Series(data_B['y'])
                
                # 验证特征一致性
                if data_A['feature_names'] != data_B['feature_names']:
                    raise ValueError(f"源域和目标域特征不一致:\n源域: {data_A['feature_names']}\n目标域: {data_B['feature_names']}")
                
                common_features = data_A['feature_names']
                
                if self.verbose:
                    print(f"✅ 使用{fallback_feature_type}特征集加载完成:")
                    print(f"   源域A: {X_A.shape}, 类别分布: {dict(y_A.value_counts().sort_index())}")
                    print(f"   目标域B: {X_B.shape}, 类别分布: {dict(y_B.value_counts().sort_index())}")
                    print(f"   特征数量: {len(common_features)}")
                
                # 转换为numpy数组
                X_source = X_A.values
                y_source = y_A.values.astype(int)
                X_target = X_B.values
                y_target = y_B.values.astype(int)
                
                # 应用预处理
                X_source_processed, y_source_processed, X_target_processed = self._preprocess_uda_data(
                    X_source, y_source, X_target, common_features
                )
                
                return X_source_processed, y_source_processed, X_target_processed, y_target.astype(int), common_features
                
            except Exception as e2:
                raise RuntimeError(f"UDA数据加载失败，尝试了{self.feature_type}和{fallback_feature_type}特征集都失败:\n原始错误: {e}\n备选错误: {e2}")
    
    def _preprocess_uda_data(self, X_source: np.ndarray, y_source: np.ndarray, X_target: np.ndarray, 
                           feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对UDA数据进行预处理（标准化和不平衡处理）
        参考test_cross_validation.py中CrossValidationEvaluator的做法
        """
        if self.verbose:
            print(f"\n🔧 UDA数据预处理:")
            print(f"   标准化方法: {self.scaler_type}")
            print(f"   不平衡处理: {self.imbalance_method}")
        
        # 直接导入避免yaml依赖
        try:
            from config.settings import get_categorical_features
        except ImportError:
            # 备选方案：直接加载settings.py
            import importlib.util
            from pathlib import Path
            project_root = Path(__file__).parent.parent
            settings_path = project_root / "config" / "settings.py"
            
            spec = importlib.util.spec_from_file_location("settings", settings_path)
            settings = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(settings)
            get_categorical_features = settings.get_categorical_features
        
        # 获取类别特征索引
        categorical_features = get_categorical_features(self.feature_type)
        categorical_indices = [i for i, name in enumerate(feature_names) if name in categorical_features]
        
        # 1. 标准化处理 - UDA数据不使用标准化
        # 对于UDA分析，跳过标准化步骤，使用原始特征
        X_source_scaled = X_source.copy()
        X_target_scaled = X_target.copy()
        if self.verbose:
            print(f"   ⚠ UDA分析跳过标准化，使用原始特征")
        
        # 2. 不平衡处理（只对源域数据）
        if self.imbalance_method != 'none':
            try:
                if self.imbalance_method == 'smote':
                    from imblearn.over_sampling import SMOTE
                    sampler = SMOTE(random_state=self.random_state)
                elif self.imbalance_method == 'smotenc':
                    from imblearn.over_sampling import SMOTENC
                    sampler = SMOTENC(categorical_features=categorical_indices, random_state=self.random_state)
                elif self.imbalance_method == 'borderline_smote':
                    from imblearn.over_sampling import BorderlineSMOTE
                    sampler = BorderlineSMOTE(random_state=self.random_state)
                elif self.imbalance_method == 'kmeans_smote':
                    from imblearn.over_sampling import KMeansSMOTE
                    sampler = KMeansSMOTE(random_state=self.random_state)
                elif self.imbalance_method == 'adasyn':
                    from imblearn.over_sampling import ADASYN
                    sampler = ADASYN(random_state=self.random_state)
                elif self.imbalance_method == 'smote_tomek':
                    from imblearn.combine import SMOTETomek
                    sampler = SMOTETomek(random_state=self.random_state)
                elif self.imbalance_method == 'random_under':
                    from imblearn.under_sampling import RandomUnderSampler
                    sampler = RandomUnderSampler(random_state=self.random_state)
                else:
                    raise ValueError(f"不支持的不平衡处理方法: {self.imbalance_method}")
                
                # 应用不平衡处理
                X_source_resampled, y_source_resampled = sampler.fit_resample(X_source_scaled, y_source)
                
                # 确保返回numpy数组
                X_source_resampled = np.array(X_source_resampled)
                y_source_resampled = np.array(y_source_resampled)
                
                if self.verbose:
                    print(f"   ✅ 不平衡处理完成")
                    print(f"   源域样本数变化: {len(y_source)} -> {len(y_source_resampled)}")
                    print(f"   源域类别分布: {dict(pd.Series(y_source_resampled).value_counts().sort_index())}")
                
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠ 不平衡处理失败: {e}, 使用原始数据")
                X_source_resampled = X_source_scaled
                y_source_resampled = y_source
        else:
            X_source_resampled = X_source_scaled
            y_source_resampled = y_source
            if self.verbose:
                print(f"   ⚠ 跳过不平衡处理")
        
        return X_source_resampled, y_source_resampled, X_target_scaled
    
    def run_source_domain_cv(self, X_source: np.ndarray, y_source: np.ndarray, feature_names: List[str]) -> Dict:
        """运行源域10折交叉验证对比"""
        if self.verbose:
            print(f"\n🔬 源域10折交叉验证对比")
            print("=" * 50)
        
        # 转换为DataFrame，使用正确的特征名称
        X_df = pd.DataFrame(X_source, columns=feature_names)
        y_series = pd.Series(y_source)
        
        if self.verbose:
            print(f"   数据形状: {X_df.shape}")
            print(f"   特征列表: {list(X_df.columns)}")
        
        # 运行模型对比，包含所有基线模型
        # TabPFN使用指定特征集，基线模型使用selected58特征集
        
        if self.verbose:
            print(f"   TabPFN将使用: {self.feature_type} 特征集 + {self.scaler_type} 标准化 + {self.imbalance_method} 不平衡处理")
            print(f"   基线模型将使用: selected58 特征集 + 无预处理")
        
        # 导入并使用run_model_comparison_cv函数
        try:
            from evaluation.cross_validation import run_model_comparison_cv
            
            cv_results = run_model_comparison_cv(
                X_df, y_series,
                feature_type=self.feature_type,  # TabPFN使用指定特征类型，基线模型在内部使用selected58
                scaler_type=self.scaler_type,  # TabPFN使用指定标准化方法
                imbalance_method=self.imbalance_method,  # TabPFN使用指定不平衡处理方法
                cv_folds=self.cv_folds,
                random_state=self.random_state,
                verbose=self.verbose
            )
        except ImportError as e:
            if self.verbose:
                print(f"❌ 无法导入交叉验证模块: {e}")
            return {}
        
        # 保存结果
        self.results['source_domain_cv'] = cv_results
        
        # 保存详细结果到文件
        cv_results_file = self.output_dir / "source_domain_cv_results.json"
        with open(cv_results_file, 'w', encoding='utf-8') as f:
            json.dump(cv_results, f, indent=2, ensure_ascii=False, default=str)
        
        if self.verbose:
            print(f"📁 源域CV结果已保存: {cv_results_file}")
        
        return cv_results
    
    def run_uda_methods(
        self, 
        X_source: np.ndarray, 
        y_source: np.ndarray,
        X_target: np.ndarray, 
        y_target: np.ndarray,
        feature_names: List[str]
    ) -> Dict:
        """运行UDA方法对比"""
        if self.verbose:
            print(f"\n🔄 UDA方法对比分析")
            print("=" * 50)
        
        # 检查ADAPT库可用性
        if not is_adapt_available():
            print("❌ ADAPT库不可用，跳过UDA分析")
            return {}
        
        # 选择要测试的UDA方法
        uda_methods_to_test = ['TCA', 'SA', 'CORAL', 'KMM']
        uda_results = {}
        
        # 创建基础估计器
        try:
            from tabpfn import TabPFNClassifier
            base_estimator = TabPFNClassifier(
                n_estimators=32, 
                random_state=self.random_state,
                ignore_pretraining_limits=True  # 允许超过500个特征
            )
            if self.verbose:
                print("✅ 使用TabPFN作为基础估计器 (ignore_pretraining_limits=True)")
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            base_estimator = LogisticRegression(penalty=None, random_state=self.random_state, max_iter=1000)
            if self.verbose:
                print("⚠ 使用LogisticRegression作为fallback")
        
        # 1. 首先测试无UDA的TabPFN基线（直接在目标域上测试）
        if self.verbose:
            print(f"\n--- 测试基线: 无UDA的TabPFN ---")
        
        try:
            # 直接用TabPFN在源域训练，目标域测试（无域适应）
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
            
            # 训练TabPFN模型
            baseline_model = base_estimator
            baseline_model.fit(X_source, y_source)
            
            # 在目标域上预测
            y_pred = baseline_model.predict(X_target)
            
            # 计算性能指标
            accuracy = accuracy_score(y_target, y_pred)
            f1 = f1_score(y_target, y_pred, average='binary')
            precision = precision_score(y_target, y_pred, average='binary')
            recall = recall_score(y_target, y_pred, average='binary')
            
            try:
                y_proba = baseline_model.predict_proba(X_target)
                if y_proba is not None and len(y_proba.shape) > 1:
                    auc = roc_auc_score(y_target, y_proba[:, 1])
                    # 保存预测数据用于ROC曲线绘制
                    y_proba_for_roc = y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba
                else:
                    auc = np.nan
                    y_proba_for_roc = None
            except:
                auc = np.nan
                y_proba_for_roc = None
            
            # 存储基线结果
            baseline_results = {
                'method_name': 'TabPFN_NoUDA',
                'accuracy': float(accuracy),
                'auc': float(auc) if not np.isnan(auc) else None,
                'f1': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'is_baseline': True,
                'y_true': y_target.tolist() if hasattr(y_target, 'tolist') else list(y_target),
                'y_pred_proba': y_proba_for_roc.tolist() if y_proba_for_roc is not None else None
            }
            
            uda_results['TabPFN_NoUDA'] = baseline_results
            
            if self.verbose:
                print(f"✅ TabPFN基线 完成:")
                print(f"   准确率: {accuracy:.4f}")
                if not np.isnan(auc):
                    print(f"   AUC: {auc:.4f}")
                print(f"   F1: {f1:.4f}")
                
        except Exception as e:
            if self.verbose:
                print(f"❌ TabPFN基线 失败: {e}")
            uda_results['TabPFN_NoUDA'] = {'error': str(e), 'is_baseline': True}
        
        # 2. 测试传统基线模型（PKUPH、Mayo、Paper_LR）- 只在目标域B上测试
        baseline_models = ['PKUPH', 'Mayo', 'Paper_LR']
        
        # 加载目标域B的selected58特征集数据（用于传统基线模型）
        try:
            from data.loader import MedicalDataLoader
            loader = MedicalDataLoader()
            data_B_selected58 = loader.load_dataset('B', feature_type='selected58')
            X_target_selected58 = pd.DataFrame(data_B_selected58['X'], columns=data_B_selected58['feature_names'])
            y_target_selected58 = pd.Series(data_B_selected58['y'])
            
            if self.verbose:
                print(f"\n--- 测试传统基线模型（仅在目标域B上测试）---")
                print(f"   目标域B数据: {X_target_selected58.shape}")
                print(f"   特征集: selected58 ({len(data_B_selected58['feature_names'])}个特征)")
        
        except Exception as e:
            if self.verbose:
                print(f"❌ 无法加载目标域B的selected58数据: {e}")
            X_target_selected58 = None
            y_target_selected58 = None
        
        if X_target_selected58 is not None:
            for model_name in baseline_models:
                if self.verbose:
                    print(f"\n--- 测试基线模型: {model_name} ---")
                
                try:
                    # 基线模型使用10折交叉验证
                    from evaluation.cross_validation import CrossValidationEvaluator
                    
                    # 创建基线模型评估器
                    evaluator = CrossValidationEvaluator(
                        model_type=model_name.lower(),
                        feature_type='selected58',  # 强制使用selected58特征类型
                        scaler_type='none',        # 基线模型不使用标准化
                        imbalance_method='none',   # 基线模型不使用不平衡处理
                        cv_folds=10,
                        random_state=self.random_state,
                        verbose=False
                    )
                    
                    if self.verbose:
                        print(f"   模型配置: {model_name}")
                        print(f"   特征集: selected58")
                        print(f"   实际使用特征数: {len(evaluator.features)}")
                        print(f"   实际使用特征: {evaluator.features}")
                        print(f"   测试方式: 10折交叉验证")
                    
                    # 基线模型在目标域B上进行10折交叉验证
                    cv_result = evaluator.run_cross_validation(X_target_selected58, y_target_selected58)
                    
                    if cv_result['summary'] and 'auc_mean' in cv_result['summary']:
                        summary = cv_result['summary']
                        
                        # 从交叉验证结果中提取预测数据
                        prediction_data = {}
                        if 'predictions' in cv_result and cv_result['predictions']:
                            pred_data = cv_result['predictions']
                            if 'y_true' in pred_data and 'y_pred_proba' in pred_data and pred_data['y_pred_proba']:
                                prediction_data = {
                                    'y_true': pred_data['y_true'],
                                    'y_pred_proba': pred_data['y_pred_proba']
                                }
                        
                        # 存储基线模型结果
                        baseline_model_results = {
                            'method_name': model_name,
                            'accuracy': float(summary.get('accuracy_mean', 0)),
                            'auc': float(summary.get('auc_mean', 0)) if summary.get('auc_mean') is not None else None,
                            'f1': float(summary.get('f1_mean', 0)),
                            'precision': float(summary.get('precision_mean', 0)),
                            'recall': float(summary.get('recall_mean', 0)),
                            'is_baseline': True,
                            'test_type': 'target_domain_cv',  # 标记为目标域交叉验证
                            'baseline_category': 'traditional_baseline',  # 标记为传统基线
                            'feature_set_used': 'selected58',  # 记录使用的特征集
                            'actual_features_count': len(evaluator.features),  # 记录实际特征数
                        }
                        
                        # 添加预测数据
                        baseline_model_results.update(prediction_data)
                        
                        uda_results[model_name] = baseline_model_results
                        
                        if self.verbose:
                            print(f"✅ {model_name}基线 完成:")
                            print(f"   准确率: {baseline_model_results['accuracy']:.4f}")
                            if baseline_model_results['auc'] is not None:
                                print(f"   AUC: {baseline_model_results['auc']:.4f}")
                            print(f"   F1: {baseline_model_results['f1']:.4f}")
                            print(f"   实际特征数: {baseline_model_results['actual_features_count']}")
                            print(f"   测试样本数: {len(y_target_selected58)}")
                    else:
                        raise ValueError(f"{model_name}模型未返回有效的性能指标")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"❌ {model_name}基线 失败: {e}")
                    uda_results[model_name] = {
                        'error': str(e), 
                        'method_name': model_name,
                        'is_baseline': True,
                        'test_type': 'target_domain_cv',
                        'baseline_category': 'traditional_baseline'
                    }
        
        # 3. 测试机器学习基线模型（使用与TabPFN相同的特征集和预处理）- 只在目标域B上测试
        ml_baseline_models = ['SVM', 'DT', 'RF', 'GBDT', 'XGBoost']
        
        # 准备目标域B的预处理数据（与TabPFN使用相同的特征集）
        try:
            # 使用与UDA分析相同的预处理数据
            X_target_df = pd.DataFrame(X_target, columns=feature_names)
            y_target_series = pd.Series(y_target)
            
            if self.verbose:
                print(f"\n--- 测试机器学习基线模型（仅在目标域B上测试）---")
                print(f"   目标域B数据: {X_target_df.shape}")
                print(f"   特征集: {self.feature_type} ({len(feature_names)}个特征)")
                print(f"   预处理: {self.scaler_type} 标准化 + {self.imbalance_method} 不平衡处理")
            
            for model_name in ml_baseline_models:
                if self.verbose:
                    print(f"\n--- 测试机器学习基线模型: {model_name} ---")
                
                try:
                    # 检查XGBoost可用性
                    if model_name.lower() == 'xgboost':
                        try:
                            import xgboost as xgb  # 使用别名避免未使用警告
                            if self.verbose:
                                print(f"   ✅ XGBoost可用，版本: {xgb.__version__}")
                        except ImportError:
                            if self.verbose:
                                print(f"   ⚠ XGBoost不可用，跳过")
                            continue
                    
                    # 使用交叉验证评估器
                    from evaluation.cross_validation import CrossValidationEvaluator
                    
                    # 为机器学习基线模型使用实际可用的特征
                    # 由于特征对齐，需要确保evaluator期望的特征与传入数据匹配
                    if self.verbose:
                        print(f"   实际可用特征: {len(feature_names)}个")
                        print(f"   特征列表: {feature_names[:5]}..." if len(feature_names) > 5 else f"   特征列表: {feature_names}")
                        
                        # 检查原始特征类型是否匹配
                        from config.settings import get_features_by_type
                        expected_features = get_features_by_type(self.feature_type)
                        missing_features = [f for f in expected_features if f not in feature_names]
                        if missing_features:
                            print(f"   ⚠ 缺失特征: {missing_features}")
                            print(f"   💡 将使用实际可用的特征进行测试")
                    
                    # 使用'selected58'作为通用的特征类型，但实际特征由数据决定
                    # 这样避免特征定义不匹配的问题
                    evaluator = CrossValidationEvaluator(
                        model_type=model_name.lower(),
                        feature_type='selected58',      # 使用通用特征类型
                        scaler_type=self.scaler_type,      # 使用与TabPFN相同的标准化
                        imbalance_method=self.imbalance_method,  # 使用与TabPFN相同的不平衡处理
                        cv_folds=10,
                        random_state=self.random_state,
                        verbose=False
                    )
                    
                    if self.verbose:
                        print(f"   模型配置: {model_name}")
                        print(f"   evaluator特征类型: selected58")
                        print(f"   evaluator预期特征数: {len(evaluator.features)}")
                        print(f"   实际传入特征数: {len(feature_names)}")
                        print(f"   预处理: {self.scaler_type} + {self.imbalance_method}")
                    
                    # 确保传递的数据只包含evaluator期望的特征
                    # 这样避免特征不匹配的问题
                    evaluator_features = evaluator.features
                    available_features = [f for f in evaluator_features if f in X_target_df.columns]
                    missing_features = [f for f in evaluator_features if f not in X_target_df.columns]
                    
                    if missing_features:
                        if self.verbose:
                            print(f"   ⚠ 跳过缺失特征: {missing_features}")
                            print(f"   ✅ 使用可用特征: {len(available_features)}/{len(evaluator_features)}")
                        
                        # 使用可用特征的子集
                        X_target_subset = X_target_df[available_features].copy()
                        
                        # 创建一个新的evaluator，仅使用可用特征
                        # 通过动态创建特征配置来实现
                        temp_feature_type = f"temp_{len(available_features)}"
                        
                        # 临时修改evaluator的特征列表
                        evaluator.features = available_features
                        evaluator.categorical_features = [f for f in evaluator.categorical_features if f in available_features]
                        evaluator.categorical_indices = [i for i, f in enumerate(available_features) if f in evaluator.categorical_features]
                        
                        cv_result = evaluator.run_cross_validation(X_target_subset, y_target_series)
                    else:
                        # 运行10折交叉验证（在目标域B上）
                        cv_result = evaluator.run_cross_validation(X_target_df, y_target_series)
                    
                    if cv_result['summary'] and 'auc_mean' in cv_result['summary']:
                        summary = cv_result['summary']
                        
                        # 从交叉验证结果中提取预测数据
                        prediction_data = {}
                        if 'predictions' in cv_result and cv_result['predictions']:
                            pred_data = cv_result['predictions']
                            if 'y_true' in pred_data and 'y_pred_proba' in pred_data and pred_data['y_pred_proba']:
                                prediction_data = {
                                    'y_true': pred_data['y_true'],
                                    'y_pred_proba': pred_data['y_pred_proba']
                                }
                        
                        # 存储机器学习基线模型结果
                        ml_baseline_results = {
                            'method_name': model_name,
                            'accuracy': float(summary.get('accuracy_mean', 0)),
                            'auc': float(summary.get('auc_mean', 0)) if summary.get('auc_mean') is not None else None,
                            'f1': float(summary.get('f1_mean', 0)),
                            'precision': float(summary.get('precision_mean', 0)),
                            'recall': float(summary.get('recall_mean', 0)),
                            'is_baseline': True,
                            'test_type': 'target_domain_cv',  # 标记为目标域交叉验证
                            'baseline_category': 'ml_baseline'  # 标记为机器学习基线
                        }
                        
                        # 添加预测数据
                        ml_baseline_results.update(prediction_data)
                        
                        uda_results[model_name] = ml_baseline_results
                        
                        if self.verbose:
                            print(f"✅ {model_name}基线 完成:")
                            print(f"   准确率: {ml_baseline_results['accuracy']:.4f}")
                            if ml_baseline_results['auc'] is not None:
                                print(f"   AUC: {ml_baseline_results['auc']:.4f}")
                            print(f"   F1: {ml_baseline_results['f1']:.4f}")
                    else:
                        raise ValueError(f"{model_name}模型未返回有效的性能指标")
                        
                except Exception as e:
                    if self.verbose:
                        print(f"❌ {model_name}基线 失败: {e}")
                    uda_results[model_name] = {
                        'error': str(e), 
                        'method_name': model_name,
                        'is_baseline': True,
                        'test_type': 'target_domain_cv',
                        'baseline_category': 'ml_baseline'
                    }
        
        except Exception as e:
            if self.verbose:
                print(f"❌ 无法测试机器学习基线模型: {e}")
        
        # 4. 然后测试各种UDA方法
        for method_name in uda_methods_to_test:
            if self.verbose:
                print(f"\n--- 测试UDA方法: {method_name} ---")
            
            try:
                # 数据已经在load_data_for_uda中预处理完成，直接使用
                if self.verbose:
                    print(f"  使用预处理后的数据: {len(feature_names)}个特征")
                    print(f"  特征列表: {feature_names}")
                
                # 使用create_uda_processor便捷函数创建UDA处理器
                from preprocessing.uda_processor import create_uda_processor
                
                processor = create_uda_processor(
                    method_name=method_name,
                    base_estimator=base_estimator,
                    save_results=False
                )
                
                # 针对不同方法优化参数（参考real_data_visualization.py）
                if method_name == 'TCA':
                    # TCA参数优化：针对医疗数据的小样本、高维特征
                    processor.config.method_params.update({
                        'n_components': None,  
                        'mu': 0.1,  # 较小的mu值，减少正则化，适合小样本
                        'kernel': 'linear'  # 线性核，适合医疗特征
                    })
                    if self.verbose:
                        print(f"  TCA参数优化: n_components=None, mu=1, kernel=linear")
                elif method_name == 'SA':
                    # SA参数优化
                    processor.config.method_params.update({
                        'n_components': None  # 自动选择最佳组件数
                    })
                    if self.verbose:
                        print(f"  SA参数优化: n_components=auto")
                elif method_name == 'CORAL':
                    # CORAL参数优化：相关性对齐
                    processor.config.method_params.update({
                        'lambda_': 1.0  # 平衡参数，适合医疗数据
                    })
                    if self.verbose:
                        print(f"  CORAL参数优化: lambda_=1.0")
                elif method_name == 'KMM':
                    # KMM参数优化：核均值匹配
                    processor.config.method_params.update({
                        'kernel': 'linear',  # 线性核，适合小样本医疗数据
                        'B': 1000,          # 权重边界
                        'eps': None         # 自动计算约束参数
                    })
                    if self.verbose:
                        print(f"  KMM参数优化: kernel=linear, B=1000")
                
                if self.verbose:
                    print(f"  创建{method_name}处理器成功")
                
                # 运行UDA方法（使用预处理后的数据）
                uda_method, method_results = processor.fit_transform(
                    X_source, y_source, X_target, y_target
                )
                
                if self.verbose:
                    print(f"  {method_name}拟合完成")
                    
                    # 显示实际使用的n_components数量（中间结果）
                    if hasattr(uda_method, 'adapt_model'):
                        adapt_model = uda_method.adapt_model
                        if method_name == 'TCA' and hasattr(adapt_model, 'vectors_'):
                            actual_n_components = adapt_model.vectors_.shape[1]
                            print(f"  TCA实际n_components: {actual_n_components} (输入时设为None)")
                        elif method_name == 'SA' and hasattr(adapt_model, 'pca_src_'):
                            actual_n_components = adapt_model.pca_src_.n_components_
                            print(f"  SA实际n_components: {actual_n_components} (输入时设为None)")
                        elif method_name in ['TCA', 'SA'] and hasattr(adapt_model, 'n_components'):
                            print(f"  {method_name}配置n_components: {adapt_model.n_components}")
                    
                    print(f"  性能结果: {method_results}")
                
                # 验证结果有效性
                if not method_results or 'accuracy' not in method_results:
                    raise ValueError(f"{method_name}方法未返回有效的性能指标")
                
                # 性能指标已经从UDAProcessor获取
                
                # 生成可视化分析
                method_output_dir = self.output_dir / f"uda_{method_name}"
                method_output_dir.mkdir(exist_ok=True)
                
                try:
                    from preprocessing.uda_visualizer import create_uda_visualizer
                    visualizer = create_uda_visualizer(
                        save_plots=True,
                        output_dir=str(method_output_dir)
                    )
                    
                    # 转换uda_method类型以匹配visualizer的期望
                    viz_results = visualizer.visualize_domain_adaptation_complete(
                        X_source, y_source,
                        X_target, y_target,
                        uda_method=uda_method,
                        method_name=method_name
                    )
                except Exception as viz_error:
                    if self.verbose:
                        print(f"  ⚠ 可视化生成失败: {viz_error}")
                    viz_results = {'error': str(viz_error)}
                
                # 获取预测数据用于ROC曲线绘制
                try:
                    # 使用UDA方法进行预测
                    y_pred = uda_method.predict(X_target)
                    y_pred_proba = uda_method.predict_proba(X_target)
                    
                    # 如果predict_proba返回二维数组，取正类概率
                    if y_pred_proba is not None and len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                        y_pred_proba_for_roc = y_pred_proba[:, 1]
                    else:
                        y_pred_proba_for_roc = y_pred_proba
                    
                    prediction_data = {
                        'y_true': y_target.tolist() if hasattr(y_target, 'tolist') else list(y_target),
                        'y_pred_proba': y_pred_proba_for_roc.tolist() if y_pred_proba_for_roc is not None else None
                    }
                except Exception as pred_error:
                    if self.verbose:
                        print(f"  ⚠ 无法获取预测数据: {pred_error}")
                    prediction_data = {}
                
                # 存储结果（从UDAProcessor获取的结果）
                final_results = {
                    'method_name': method_name,
                    'accuracy': method_results.get('accuracy', 0),
                    'auc': method_results.get('auc', None),
                    'f1': method_results.get('f1', 0),
                    'precision': method_results.get('precision', 0),
                    'recall': method_results.get('recall', 0),
                    'output_dir': str(method_output_dir),
                    'visualization_results': viz_results,
                    'is_baseline': False  # UDA方法不是基线
                }
                
                # 添加预测数据
                final_results.update(prediction_data)
                
                uda_results[method_name] = final_results
                
                if self.verbose:
                    print(f"✅ {method_name} 完成:")
                    print(f"   准确率: {final_results['accuracy']:.4f}")
                    if final_results['auc'] is not None:
                        print(f"   AUC: {final_results['auc']:.4f}")
                    print(f"   F1: {final_results['f1']:.4f}")
                
            except Exception as e:
                if self.verbose:
                    print(f"❌ {method_name} 失败: {e}")
                    import traceback
                    print(f"  详细错误信息:")
                    traceback.print_exc()
                
                uda_results[method_name] = {
                    'error': str(e),
                    'method_name': method_name,
                    'accuracy': 0,
                    'auc': None,
                    'f1': 0,
                    'precision': 0,
                    'recall': 0
                }
        
        # 保存UDA结果
        self.results['uda_methods'] = uda_results
        
        uda_results_file = self.output_dir / "uda_methods_results.json"
        with open(uda_results_file, 'w', encoding='utf-8') as f:
            json.dump(uda_results, f, indent=2, ensure_ascii=False, default=str)
        
        if self.verbose:
            print(f"📁 UDA结果已保存: {uda_results_file}")
        
        return uda_results
    
    def generate_comparison_visualizations(self) -> Dict:
        """生成对比可视化图表"""
        if self.verbose:
            print(f"\n📊 生成对比可视化图表")
            print("=" * 50)
        
        # 使用新的可视化模块
        try:
            from preprocessing.analysis_visualizer import create_analysis_visualizer
        except ImportError as e:
            if self.verbose:
                print(f"❌ 无法导入可视化模块: {e}")
            return {}
        
        visualizer = create_analysis_visualizer(
            output_dir=str(self.output_dir),
            save_plots=True,
            show_plots=self.verbose  # 只在verbose模式下显示图表
        )
        
        # 收集预测数据（如果可用）
        cv_predictions = {}
        uda_predictions = {}
        
        # 从CV结果中提取预测数据
        if 'source_domain_cv' in self.results:
            for exp_name, result in self.results['source_domain_cv'].items():
                # 优先使用predictions字段中的合并数据
                if 'predictions' in result and result['predictions']:
                    predictions = result['predictions']
                    if 'y_true' in predictions and 'y_pred_proba' in predictions:
                        y_true = predictions['y_true']
                        y_pred_proba = predictions['y_pred_proba']
                        
                        if y_true and y_pred_proba:
                            cv_predictions[exp_name] = {
                                'y_true': y_true,
                                'y_pred_proba': y_pred_proba
                            }
                # 如果没有predictions字段，则从fold_results中提取
                elif 'fold_results' in result:
                    all_y_true = []
                    all_y_pred_proba = []
                    for fold_result in result['fold_results']:
                        if 'y_true' in fold_result and 'y_pred_proba' in fold_result:
                            all_y_true.extend(fold_result['y_true'])
                            all_y_pred_proba.extend(fold_result['y_pred_proba'])
                    
                    if all_y_true and all_y_pred_proba:
                        cv_predictions[exp_name] = {
                            'y_true': all_y_true,
                            'y_pred_proba': all_y_pred_proba
                        }
        
        # 从UDA结果中提取预测数据
        if 'uda_methods' in self.results:
            for method_name, result in self.results['uda_methods'].items():
                if 'y_true' in result and 'y_pred_proba' in result:
                    uda_predictions[method_name] = {
                        'y_true': result['y_true'],
                        'y_pred_proba': result['y_pred_proba'],
                        'is_baseline': result.get('is_baseline', False),
                        'baseline_category': result.get('baseline_category', None)
                    }
        
        # 生成所有可视化图表
        viz_results = visualizer.generate_all_visualizations(
            cv_results=self.results['source_domain_cv'],
            uda_results=self.results['uda_methods'],
            cv_predictions=cv_predictions,
            uda_predictions=uda_predictions
        )
        
        if self.verbose:
            for viz_name, viz_path in viz_results.items():
                if viz_path:
                    print(f"✅ {viz_name} 已保存: {viz_path}")
                else:
                    print(f"⚠ {viz_name} 生成失败")
        
        # 生成Nature标准组合图像 (使用原生matplotlib方法)
        try:
            if self.verbose:
                print(f"\n📊 生成Nature标准组合图像...")
            
            # 直接调用visualizer的组合图像方法
            combined_figure_path = visualizer.plot_combined_analysis_figure(
                cv_results=self.results['source_domain_cv'],
                uda_results=self.results['uda_methods'],
                cv_predictions=cv_predictions,
                uda_predictions=uda_predictions
            )
            
            if combined_figure_path:
                viz_results['combined_analysis_figure'] = combined_figure_path
                if self.verbose:
                    print(f"✅ 组合分析图像生成完成: {combined_figure_path}")
            else:
                viz_results['combined_analysis_figure'] = None
                if self.verbose:
                    print(f"⚠️ 组合分析图像生成失败")
                    
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 组合图像生成出错: {e}")
            viz_results['combined_analysis_figure'] = None
        
        # Nature标准组合热力图已经在generate_all_visualizations中生成
        # 不需要额外的处理步骤
        
        self.results['visualizations'] = viz_results
        return viz_results
    
# 原来的可视化方法已移动到 preprocessing/analysis_visualizer.py 模块中
    
    def generate_final_report(self) -> str:
        """生成最终分析报告"""
        if self.verbose:
            print(f"\n📋 生成最终分析报告")
            print("=" * 50)
        
        report_content = []
        report_content.append("# 完整医疗数据UDA分析报告\n")
        report_content.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 配置信息
        report_content.append("## 分析配置\n")
        config = self.results['config']
        report_content.append(f"- 特征集: {config['feature_type']}")
        report_content.append(f"- 标准化方法: {config['scaler_type']}")
        report_content.append(f"- 不平衡处理: {config['imbalance_method']}")
        report_content.append(f"- 交叉验证折数: {config['cv_folds']}")
        report_content.append(f"- 随机种子: {config['random_state']}\n")
        
        # 源域CV结果
        if self.results['source_domain_cv']:
            report_content.append("## 源域10折交叉验证结果\n")
            cv_results = self.results['source_domain_cv']
            
            report_content.append("| 方法 | AUC | Accuracy | F1 | Precision | Recall |")
            report_content.append("|------|-----|----------|----|-----------| -------|")
            
            for exp_name, result in cv_results.items():
                if 'summary' in result and result['summary']:
                    # 处理方法名称显示
                    raw_method_name = exp_name.split('_')[0].upper()
                    if raw_method_name == 'PAPER':
                        method_name = 'Paper_LR'
                    else:
                        method_name = raw_method_name
                    summary = result['summary']
                    
                    auc = summary.get('auc_mean', 0)
                    acc = summary.get('accuracy_mean', 0)
                    f1 = summary.get('f1_mean', 0)
                    prec = summary.get('precision_mean', 0)
                    rec = summary.get('recall_mean', 0)
                    
                    report_content.append(f"| {method_name} | {auc:.4f} | {acc:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} |")
            
            report_content.append("")
        
        # UDA方法结果
        if self.results['uda_methods']:
            report_content.append("## UDA方法对比结果\n")
            uda_results = self.results['uda_methods']
            successful_methods = {k: v for k, v in uda_results.items() if 'error' not in v}
            
            if successful_methods:
                report_content.append("| 方法 | AUC | Accuracy | F1 | Precision | Recall | 类型 |")
                report_content.append("|------|-----|----------|----|-----------| -------|------|")
                
                # 按类型分组显示结果
                tabpfn_baseline = {}
                traditional_baselines = {}
                ml_baselines = {}
                uda_methods = {}
                
                for method, result in successful_methods.items():
                    if result.get('is_baseline', False):
                        if method == 'TabPFN_NoUDA':
                            tabpfn_baseline[method] = result
                        elif result.get('baseline_category') == 'ml_baseline':
                            ml_baselines[method] = result
                        elif result.get('baseline_category') == 'traditional_baseline':
                            traditional_baselines[method] = result
                        else:
                            # 兼容旧格式，PKUPH、Mayo、Paper_LR等
                            traditional_baselines[method] = result
                    else:
                        uda_methods[method] = result
                
                # 先显示TabPFN基线
                for method, result in tabpfn_baseline.items():
                    auc = result.get('auc', 0) if result.get('auc') is not None else 0
                    acc = result.get('accuracy', 0)
                    f1 = result.get('f1', 0)
                    prec = result.get('precision', 0)
                    rec = result.get('recall', 0)
                    
                    report_content.append(f"| {method} | {auc:.4f} | {acc:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} | TabPFN基线 |")
                
                # 显示传统基线方法
                for method, result in traditional_baselines.items():
                    auc = result.get('auc', 0) if result.get('auc') is not None else 0
                    acc = result.get('accuracy', 0)
                    f1 = result.get('f1', 0)
                    prec = result.get('precision', 0)
                    rec = result.get('recall', 0)
                    
                    report_content.append(f"| {method} | {auc:.4f} | {acc:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} | 传统基线 |")
                
                # 显示机器学习基线方法
                for method, result in ml_baselines.items():
                    auc = result.get('auc', 0) if result.get('auc') is not None else 0
                    acc = result.get('accuracy', 0)
                    f1 = result.get('f1', 0)
                    prec = result.get('precision', 0)
                    rec = result.get('recall', 0)
                    
                    report_content.append(f"| {method} | {auc:.4f} | {acc:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} | 机器学习基线 |")
                
                # 再显示UDA方法
                for method, result in uda_methods.items():
                    auc = result.get('auc', 0) if result.get('auc') is not None else 0
                    acc = result.get('accuracy', 0)
                    f1 = result.get('f1', 0)
                    prec = result.get('precision', 0)
                    rec = result.get('recall', 0)
                    
                    report_content.append(f"| {method} | {auc:.4f} | {acc:.4f} | {f1:.4f} | {prec:.4f} | {rec:.4f} | UDA方法 |")
                
                report_content.append("")
            
            # 失败的方法
            failed_methods = {k: v for k, v in uda_results.items() if 'error' in v}
            if failed_methods:
                report_content.append("### 失败的方法\n")
                for method, result in failed_methods.items():
                    method_type = "基线方法" if result.get('is_baseline', False) else "UDA方法"
                    report_content.append(f"- {method} ({method_type}): {result['error']}")
                report_content.append("")
        
        # 结论和建议
        report_content.append("## 结论和建议\n")
        
        # 找出最佳方法
        best_source_method = ""
        best_source_auc = 0
        
        if self.results['source_domain_cv']:
            for exp_name, result in self.results['source_domain_cv'].items():
                if 'summary' in result and result['summary']:
                    auc = result['summary'].get('auc_mean', 0)
                    if auc > best_source_auc:
                        best_source_auc = auc
                        raw_method_name = exp_name.split('_')[0].upper()
                        if raw_method_name == 'PAPER':
                            best_source_method = 'Paper_LR'
                        else:
                            best_source_method = raw_method_name
        
        # 找出UDA方法中的最佳方法和基线
        best_uda_method = ""
        best_uda_auc = 0
        baseline_auc = 0
        
        if self.results['uda_methods']:
            successful_uda = {k: v for k, v in self.results['uda_methods'].items() if 'error' not in v}
            
            # 获取TabPFN_NoUDA基线结果
            if 'TabPFN_NoUDA' in successful_uda:
                baseline_result = successful_uda['TabPFN_NoUDA']
                baseline_auc = baseline_result.get('auc', 0) if baseline_result.get('auc') is not None else 0
            
            # 找出最佳UDA方法（排除基线）
            for method, result in successful_uda.items():
                if method != 'TabPFN_NoUDA':  # 排除基线
                    auc = result.get('auc', 0) if result.get('auc') is not None else 0
                    if auc > best_uda_auc:
                        best_uda_auc = auc
                        best_uda_method = method
        
        if best_source_method:
            report_content.append(f"- **最佳源域方法**: {best_source_method} (AUC: {best_source_auc:.4f})")
        
        if baseline_auc > 0:
            report_content.append(f"- **TabPFN无UDA基线**: TabPFN_NoUDA (AUC: {baseline_auc:.4f})")
        
        if best_uda_method:
            report_content.append(f"- **最佳UDA方法**: {best_uda_method} (AUC: {best_uda_auc:.4f})")
        
        # 比较最佳UDA方法与TabPFN无UDA基线
        if baseline_auc > 0 and best_uda_auc > 0:
            improvement = best_uda_auc - baseline_auc
            if improvement > 0:
                report_content.append(f"- **域适应效果**: {best_uda_method}相比TabPFN无UDA基线提升了 {improvement:.4f} AUC")
            else:
                report_content.append(f"- **域适应效果**: {best_uda_method}相比TabPFN无UDA基线下降了 {abs(improvement):.4f} AUC")
        
        report_content.append(f"\n详细结果和可视化图表请查看: {self.output_dir}")
        
        # 保存报告
        report_file = self.output_dir / "analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        if self.verbose:
            print(f"📁 分析报告已保存: {report_file}")
            print("\n" + "="*60)
            print("分析报告预览:")
            print("="*60)
            for line in report_content[:20]:  # 显示前20行
                print(line)
            if len(report_content) > 20:
                print("...")
            print("="*60)
        
        return str(report_file)
    
    def run_complete_analysis(self) -> Dict:
        """运行完整分析流程"""
        if self.verbose:
            print(f"🚀 开始完整分析流程")
            print("=" * 60)
        
        try:
            # 1. 加载数据用于源域交叉验证（基线模型需要selected58特征集）
            X_source_cv, y_source_cv, X_target_cv, y_target_cv, feature_names_cv = self.load_data_for_cv()
            
            # 2. 源域10折交叉验证
            self.run_source_domain_cv(X_source_cv, y_source_cv, feature_names_cv)
            
            # 3. 加载数据用于UDA分析（使用指定特征集，包含预处理）
            X_source_uda, y_source_uda, X_target_uda, y_target_uda, feature_names_uda = self.load_data_for_uda()
            
            # 4. UDA方法对比
            self.run_uda_methods(X_source_uda, y_source_uda, X_target_uda, y_target_uda, feature_names_uda)
            
            # 5. 生成对比可视化
            self.generate_comparison_visualizations()
            
            # 6. 生成最终报告
            report_file = self.generate_final_report()
            
            # 7. 保存完整结果
            complete_results_file = self.output_dir / "complete_results.json"
            with open(complete_results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
            
            if self.verbose:
                print(f"\n✅ 完整分析流程完成！")
                print(f"📁 所有结果已保存到: {self.output_dir}")
                print(f"📋 分析报告: {report_file}")
                print(f"📊 完整结果: {complete_results_file}")
            
            return self.results
            
        except Exception as e:
            if self.verbose:
                print(f"❌ 分析流程失败: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='完整医疗数据UDA分析流程')
    parser.add_argument('--feature_type', type=str, default='best8',
                        help='特征类型 (default: best8)')
    parser.add_argument('--scaler_type', type=str, default='none',
                        help='标准化方法 (default: none)')
    parser.add_argument('--imbalance_method', type=str, default='none',
                        help='不平衡处理方法 (default: none)')
    parser.add_argument('--cv_folds', type=int, default=10,
                        help='交叉验证折数 (default: 10)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子 (default: 42)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细信息 (default: True)')
    
    args = parser.parse_args()
    
    print("🏥 完整医疗数据UDA分析流程")
    print("=" * 60)
    print(f"📋 分析配置:")
    print(f"   特征类型: {args.feature_type}")
    print(f"   标准化方法: {args.scaler_type}")
    print(f"   不平衡处理: {args.imbalance_method}")
    print(f"   交叉验证折数: {args.cv_folds}")
    print(f"   随机种子: {args.random_state}")
    print("=" * 60)
    
    # 创建分析运行器
    runner = CompleteAnalysisRunner(
        feature_type=args.feature_type,
        scaler_type=args.scaler_type,
        imbalance_method=args.imbalance_method,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        verbose=args.verbose
    )
    
    # 运行完整分析
    results = runner.run_complete_analysis()
    
    if 'error' not in results:
        print(f"\n🎉 分析成功完成！")
        print(f"📁 查看结果目录: {runner.output_dir}")
    else:
        print(f"\n❌ 分析失败: {results['error']}")


if __name__ == "__main__":
    main() 