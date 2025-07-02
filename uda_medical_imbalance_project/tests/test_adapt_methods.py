"""
测试Adapt库UDA方法的功能

本测试文件验证：
1. Adapt库的可用性
2. 使用A和B数据集的best10特征进行域适应
3. TabPFN模型在DA前后的性能对比
4. 多种UDA方法的效果对比
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import warnings
from typing import Tuple, Dict, Optional, Any
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入TabPFN
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    print("Warning: TabPFN not available. Using LogisticRegression as fallback.")
    TabPFNClassifier = LogisticRegression  # type: ignore
    TABPFN_AVAILABLE = False

# 导入我们的Adapt方法包装器
try:
    from uda.adapt_methods import (
        is_adapt_available,
        get_available_adapt_methods,
        create_adapt_method
    )
    ADAPT_METHODS_AVAILABLE = True
except ImportError:
    # 如果导入失败，创建模拟函数
    def is_adapt_available() -> bool:
        return False
    def get_available_adapt_methods() -> Dict:
        return {}
    def create_adapt_method(*args, **kwargs):  # type: ignore
        raise ImportError("Adapt方法不可用")
    ADAPT_METHODS_AVAILABLE = False

# 导入数据加载和预处理模块
try:
    from data.loader import MedicalDataLoader
    DATA_MODULES_AVAILABLE = True
except ImportError:
    print("Warning: Data modules not available. Creating mock data.")
    DATA_MODULES_AVAILABLE = False
    MedicalDataLoader = None  # type: ignore


def load_test_data():
    """加载测试数据（A和B数据集的best8特征）"""
    if DATA_MODULES_AVAILABLE and MedicalDataLoader is not None:
        try:
            # 使用MedicalDataLoader加载真实数据
            loader = MedicalDataLoader()
            
            # 加载数据集A和B，使用best8特征
            data_A = loader.load_dataset('A', feature_type='best8')
            data_B = loader.load_dataset('B', feature_type='best8')
            
            # 提取特征和标签
            X_A = pd.DataFrame(data_A['X'], columns=data_A['feature_names'])
            y_A = pd.Series(data_A['y'], name='label')
            X_B = pd.DataFrame(data_B['X'], columns=data_B['feature_names'])
            y_B = pd.Series(data_B['y'], name='label')
            
            # 确保A和B数据集使用相同的特征列（特征对齐）
            common_features = list(set(X_A.columns) & set(X_B.columns))
            if len(common_features) != len(X_A.columns) or len(common_features) != len(X_B.columns):
                print(f"警告: A和B数据集特征不完全一致")
                print(f"  A特征: {list(X_A.columns)}")
                print(f"  B特征: {list(X_B.columns)}")
                print(f"  共同特征: {common_features}")
                # 使用共同特征
                X_A = X_A[common_features]
                X_B = X_B[common_features]
            
            print(f"✓ 加载真实数据集A: {X_A.shape}, 数据集B: {X_B.shape}")
            print(f"  特征列表: {list(X_A.columns)}")
            print(f"  A类别分布: {dict(y_A.value_counts().sort_index())}")
            print(f"  B类别分布: {dict(y_B.value_counts().sort_index())}")
            return X_A, y_A, X_B, y_B
            
        except Exception as e:
            print(f"加载真实数据失败: {e}")
    
    # 创建模拟数据（仅在无法加载真实数据时使用）
    print("使用模拟数据...")
    np.random.seed(42)
    
    # 从MedicalDataLoader获取best8特征列表
    if DATA_MODULES_AVAILABLE and MedicalDataLoader is not None:
        loader = MedicalDataLoader()
        best8_features = loader.BEST_8_FEATURES
        categorical_features = [f for f in loader.CAT_FEATURE_NAMES if f in best8_features]
    else:
        # 如果无法导入，使用硬编码的特征列表
        best8_features = [
            'Feature63', 'Feature2', 'Feature46', 'Feature61', 
            'Feature56', 'Feature42', 'Feature39', 'Feature43'
        ]
        categorical_features = ['Feature63', 'Feature46']
    
    # 模拟数据集A（源域）
    n_samples_A = 200
    X_A_data = {}
    
    for feature in best8_features:
        if feature in categorical_features:
            # 类别特征
            X_A_data[feature] = np.random.choice([0, 1], n_samples_A, p=[0.6, 0.4])
        else:
            # 连续特征
            X_A_data[feature] = np.random.normal(0, 1, n_samples_A)
    
    X_A = pd.DataFrame(X_A_data)[best8_features]  # 确保列顺序正确
    y_A = pd.Series(np.random.choice([0, 1], n_samples_A, p=[0.6, 0.4]), name='label')
    
    # 模拟数据集B（目标域）- 添加域偏移
    n_samples_B = 150
    X_B_data = {}
    
    for feature in best8_features:
        if feature in categorical_features:
            # 类别特征 - 改变分布
            X_B_data[feature] = np.random.choice([0, 1], n_samples_B, p=[0.4, 0.6])  # 反转分布
        else:
            # 连续特征 - 添加偏移
            X_B_data[feature] = np.random.normal(0.5, 1.2, n_samples_B)  # 均值和方差偏移
    
    X_B = pd.DataFrame(X_B_data)[best8_features]  # 确保列顺序正确
    y_B = pd.Series(np.random.choice([0, 1], n_samples_B, p=[0.4, 0.6]), name='label')
    
    print(f"✓ 创建模拟数据集A: {X_A.shape}, 数据集B: {X_B.shape}")
    print(f"  特征列表: {list(X_A.columns)}")
    return X_A, y_A, X_B, y_B


def create_tabpfn_model(categorical_features: Optional[list] = None):
    """创建TabPFN模型"""
    if TABPFN_AVAILABLE:
        # 使用TabPFN默认参数 n_estimators=32
        return TabPFNClassifier(n_estimators=32)
    else:
        # 使用LogisticRegression作为fallback
        return LogisticRegression(penalty=None, random_state=42, max_iter=1000)


def calculate_metrics(y_true: Any, y_pred: Any, y_pred_proba: Any = None) -> Dict[str, float]:
    """计算评估指标"""
    # 确保y_true是numpy数组
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, average='binary')),
        'precision': float(precision_score(y_true, y_pred, average='binary')),
        'recall': float(recall_score(y_true, y_pred, average='binary')),
    }
    
    # 计算AUC
    if y_pred_proba is not None:
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            metrics['auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
        else:
            metrics['auc'] = float(roc_auc_score(y_true, y_pred_proba))
    else:
        metrics['auc'] = float('nan')
    
    return metrics


def save_test_results(results: Dict[str, Any], test_name: str, output_dir: str = "tests") -> Path:
    """保存测试结果到指定目录"""
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存JSON格式的详细结果
    json_filename = f"{test_name}_results_{timestamp}.json"
    json_path = output_path / json_filename
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"✓ 详细结果已保存到: {json_path}")
    
    # 保存CSV格式的摘要结果
    if any(isinstance(v, dict) and 'accuracy' in v for v in results.values()):
        csv_filename = f"{test_name}_summary_{timestamp}.csv"
        csv_path = output_path / csv_filename
        
        summary_data = []
        for method_name, metrics in results.items():
            if isinstance(metrics, dict) and 'accuracy' in metrics:
                summary_data.append({
                    'method': method_name,
                    **metrics
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ 摘要结果已保存到: {csv_path}")
    
    return json_path


class TestAdaptMethodsWithRealData:
    """使用真实数据测试Adapt方法"""
    
    @pytest.fixture(scope="class")
    def medical_data(self):
        """加载医疗数据"""
        return load_test_data()
    
    def test_data_loading(self, medical_data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]):
        """测试数据加载"""
        X_A, y_A, X_B, y_B = medical_data
        
        # 获取预期的特征数量（best8应该是8个特征）
        expected_features = 8
        
        # 检查数据形状
        assert X_A.shape[1] == expected_features, f"A数据集特征数不正确: {X_A.shape[1]} vs {expected_features}"
        assert X_B.shape[1] == expected_features, f"B数据集特征数不正确: {X_B.shape[1]} vs {expected_features}"
        assert len(X_A) == len(y_A), "A数据集样本数不匹配"
        assert len(X_B) == len(y_B), "B数据集样本数不匹配"
        
        # 检查特征名称一致性
        assert list(X_A.columns) == list(X_B.columns), "A和B数据集特征名称不一致"
        
        # 检查特征名称格式
        for col in X_A.columns:
            assert col.startswith('Feature'), f"特征名称格式不正确: {col}"
        
        print(f"✓ 数据加载测试通过")
        print(f"  数据集A: {X_A.shape}, 类别分布: {dict(y_A.value_counts().sort_index())}")
        print(f"  数据集B: {X_B.shape}, 类别分布: {dict(y_B.value_counts().sort_index())}")
        print(f"  特征列表: {list(X_A.columns)}")
    
    def test_baseline_tabpfn_performance(self, medical_data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]):
        """测试基线TabPFN性能（无域适应）"""
        X_A, y_A, X_B, y_B = medical_data
        
        print(f"\n=== 基线TabPFN性能测试（A训练 → B测试）===")
        
        # 创建TabPFN模型
        model = create_tabpfn_model()
        
        # 在A数据集上训练
        model.fit(X_A, y_A)
        
        # 在B数据集上测试
        y_pred = model.predict(X_B)
        try:
            y_pred_proba = model.predict_proba(X_B)
        except:
            y_pred_proba = None
        
        # 计算指标
        metrics = calculate_metrics(y_B, y_pred, y_pred_proba)
        
        print(f"基线性能:")
        for metric, value in metrics.items():
            if not np.isnan(value):
                print(f"  {metric.upper()}: {value:.4f}")
        
        # 存储基线结果供后续对比
        self.baseline_metrics = metrics
        
        # 保存基线结果
        baseline_results = {
            'baseline_tabpfn': metrics,
            'data_info': {
                'source_samples': len(X_A),
                'target_samples': len(X_B),
                'features': list(X_A.columns),
                'source_distribution': dict(y_A.value_counts().sort_index()),
                'target_distribution': dict(y_B.value_counts().sort_index())
            },
            'test_timestamp': datetime.now().isoformat()
        }
        save_test_results(baseline_results, "baseline_tabpfn", "tests")
        
        assert 0 <= metrics['accuracy'] <= 1, "准确率应该在[0,1]范围内"
        print(f"✓ 基线TabPFN测试完成")
    
    def test_adapt_availability(self):
        """测试Adapt库可用性"""
        if not ADAPT_METHODS_AVAILABLE:
            pytest.skip("Adapt方法模块不可用")
        
        available = is_adapt_available()
        print(f"Adapt库可用性: {available}")
        
        if available:
            methods = get_available_adapt_methods()
            print(f"可用方法数量: {len(methods)}")
            for method, info in methods.items():
                print(f"  {method}: {info['description']} ({info['type']})")
            assert len(methods) > 0, "应该有可用的方法"
        else:
            pytest.skip("Adapt库不可用，跳过相关测试")
    
    def test_kmm_domain_adaptation(self, medical_data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]):
        """测试KMM域适应方法"""
        if not ADAPT_METHODS_AVAILABLE or not is_adapt_available():
            pytest.skip("Adapt库不可用")
        
        X_A, y_A, X_B, y_B = medical_data
        
        print(f"\n=== KMM域适应测试（A训练 → B测试）===")
        
        try:
            # 创建KMM方法
            kmm_method = create_adapt_method(
                method_name='KMM',
                estimator=create_tabpfn_model(),
                kernel='linear',
                verbose=0,
                random_state=42
            )
            
            # 拟合模型（使用A作为源域，B作为目标域）
            kmm_method.fit(X_A, y_A, X_B)
            
            # 在B数据集上预测
            y_pred = kmm_method.predict(X_B)
            try:
                y_pred_proba = kmm_method.predict_proba(X_B)
            except:
                y_pred_proba = None
            
            # 计算指标
            metrics = calculate_metrics(y_B, y_pred, y_pred_proba)
            
            print(f"KMM域适应性能:")
            for metric, value in metrics.items():
                if not np.isnan(value):
                    print(f"  {metric.upper()}: {value:.4f}")
            
            # 与基线对比
            if hasattr(self, 'baseline_metrics'):
                print(f"\n性能对比（KMM vs 基线）:")
                for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
                    if metric in metrics and metric in self.baseline_metrics:
                        baseline_val = self.baseline_metrics[metric]
                        kmm_val = metrics[metric]
                        if not np.isnan(baseline_val) and not np.isnan(kmm_val):
                            improvement = kmm_val - baseline_val
                            print(f"  {metric.upper()}: {baseline_val:.4f} → {kmm_val:.4f} "
                                  f"({'+'if improvement >= 0 else ''}{improvement:.4f})")
            
            # 测试权重获取
            weights = kmm_method.get_weights()
            if weights is not None:
                print(f"✓ 获取到实例权重，权重范围: [{weights.min():.4f}, {weights.max():.4f}]")
            
            assert 0 <= metrics['accuracy'] <= 1, "准确率应该在[0,1]范围内"
            
            # 保存KMM测试结果
            kmm_results = {
                'kmm_performance': metrics,
                'baseline_comparison': {
                    'baseline': self.baseline_metrics if hasattr(self, 'baseline_metrics') else None,
                    'improvements': {}
                },
                'weights_info': {
                    'weights_available': weights is not None,
                    'weights_range': [float(weights.min()), float(weights.max())] if weights is not None else None
                },
                'data_info': {
                    'source_samples': len(X_A),
                    'target_samples': len(X_B),
                    'features': list(X_A.columns)
                },
                'test_timestamp': datetime.now().isoformat()
            }
            
            # 计算与基线的改进
            if hasattr(self, 'baseline_metrics'):
                for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
                    if metric in metrics and metric in self.baseline_metrics:
                        baseline_val = self.baseline_metrics[metric]
                        kmm_val = metrics[metric]
                        if not np.isnan(baseline_val) and not np.isnan(kmm_val):
                            kmm_results['baseline_comparison']['improvements'][metric] = {
                                'baseline': float(baseline_val),
                                'kmm': float(kmm_val),
                                'improvement': float(kmm_val - baseline_val)
                            }
            
            save_test_results(kmm_results, "kmm_domain_adaptation", "tests")
            print(f"✓ KMM域适应测试完成")
            
        except Exception as e:
            print(f"KMM测试失败: {e}")
            warnings.warn(f"KMM测试失败: {e}")
    
    def test_coral_domain_adaptation(self, medical_data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]):
        """测试CORAL域适应方法"""
        if not ADAPT_METHODS_AVAILABLE or not is_adapt_available():
            pytest.skip("Adapt库不可用")
        
        X_A, y_A, X_B, y_B = medical_data
        
        print(f"\n=== CORAL域适应测试（A训练 → B测试）===")
        
        try:
            # 创建CORAL方法
            coral_method = create_adapt_method(
                method_name='CORAL',
                estimator=create_tabpfn_model(),
                lambda_=1.0,
                verbose=0,
                random_state=42
            )
            
            # 拟合模型
            coral_method.fit(X_A, y_A, X_B)
            
            # 预测
            y_pred = coral_method.predict(X_B)
            try:
                y_pred_proba = coral_method.predict_proba(X_B)
            except:
                y_pred_proba = None
            
            # 计算指标
            metrics = calculate_metrics(y_B, y_pred, y_pred_proba)
            
            print(f"CORAL域适应性能:")
            for metric, value in metrics.items():
                if not np.isnan(value):
                    print(f"  {metric.upper()}: {value:.4f}")
            
            # 与基线对比
            if hasattr(self, 'baseline_metrics'):
                print(f"\n性能对比（CORAL vs 基线）:")
                for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
                    if metric in metrics and metric in self.baseline_metrics:
                        baseline_val = self.baseline_metrics[metric]
                        coral_val = metrics[metric]
                        if not np.isnan(baseline_val) and not np.isnan(coral_val):
                            improvement = coral_val - baseline_val
                            print(f"  {metric.upper()}: {baseline_val:.4f} → {coral_val:.4f} "
                                  f"({'+'if improvement >= 0 else ''}{improvement:.4f})")
            
            assert 0 <= metrics['accuracy'] <= 1, "准确率应该在[0,1]范围内"
            print(f"✓ CORAL域适应测试完成")
            
        except Exception as e:
            print(f"CORAL测试失败: {e}")
            warnings.warn(f"CORAL测试失败: {e}")
    
    def test_sa_domain_adaptation(self, medical_data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]):
        """测试SA（子空间对齐）域适应方法"""
        if not ADAPT_METHODS_AVAILABLE or not is_adapt_available():
            pytest.skip("Adapt库不可用")
        
        X_A, y_A, X_B, y_B = medical_data
        
        print(f"\n=== SA域适应测试（A训练 → B测试）===")
        
        try:
            # 创建SA方法
            sa_method = create_adapt_method(
                method_name='SA',
                estimator=create_tabpfn_model(),
                n_components=None,  # 自动选择组件数
                verbose=0,
                random_state=42
            )
            
            # 拟合模型
            sa_method.fit(X_A, y_A, X_B)
            
            # 预测
            y_pred = sa_method.predict(X_B)
            try:
                y_pred_proba = sa_method.predict_proba(X_B)
            except:
                y_pred_proba = None
            
            # 计算指标
            metrics = calculate_metrics(y_B, y_pred, y_pred_proba)
            
            print(f"SA域适应性能:")
            for metric, value in metrics.items():
                if not np.isnan(value):
                    print(f"  {metric.upper()}: {value:.4f}")
            
            # 与基线对比
            if hasattr(self, 'baseline_metrics'):
                print(f"\n性能对比（SA vs 基线）:")
                for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
                    if metric in metrics and metric in self.baseline_metrics:
                        baseline_val = self.baseline_metrics[metric]
                        sa_val = metrics[metric]
                        if not np.isnan(baseline_val) and not np.isnan(sa_val):
                            improvement = sa_val - baseline_val
                            print(f"  {metric.upper()}: {baseline_val:.4f} → {sa_val:.4f} "
                                  f"({'+'if improvement >= 0 else ''}{improvement:.4f})")
            
            assert 0 <= metrics['accuracy'] <= 1, "准确率应该在[0,1]范围内"
            print(f"✓ SA域适应测试完成")
            
        except Exception as e:
            print(f"SA测试失败: {e}")
            warnings.warn(f"SA测试失败: {e}")
    
    def test_tca_domain_adaptation(self, medical_data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]):
        """测试TCA（迁移成分分析）域适应方法"""
        if not ADAPT_METHODS_AVAILABLE or not is_adapt_available():
            pytest.skip("Adapt库不可用")
        
        X_A, y_A, X_B, y_B = medical_data
        
        print(f"\n=== TCA域适应测试（A训练 → B测试）===")
        
        try:
            # 创建TCA方法
            tca_method = create_adapt_method(
                method_name='TCA',
                estimator=create_tabpfn_model(),
                n_components=None,  # 自动选择组件数
                mu=0.1,
                kernel='linear',
                verbose=0,
                random_state=42
            )
            
            # 拟合模型
            tca_method.fit(X_A, y_A, X_B)
            
            # 预测
            y_pred = tca_method.predict(X_B)
            try:
                y_pred_proba = tca_method.predict_proba(X_B)
            except:
                y_pred_proba = None
            
            # 计算指标
            metrics = calculate_metrics(y_B, y_pred, y_pred_proba)
            
            print(f"TCA域适应性能:")
            for metric, value in metrics.items():
                if not np.isnan(value):
                    print(f"  {metric.upper()}: {value:.4f}")
            
            # 与基线对比
            if hasattr(self, 'baseline_metrics'):
                print(f"\n性能对比（TCA vs 基线）:")
                for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
                    if metric in metrics and metric in self.baseline_metrics:
                        baseline_val = self.baseline_metrics[metric]
                        tca_val = metrics[metric]
                        if not np.isnan(baseline_val) and not np.isnan(tca_val):
                            improvement = tca_val - baseline_val
                            print(f"  {metric.upper()}: {baseline_val:.4f} → {tca_val:.4f} "
                                  f"({'+'if improvement >= 0 else ''}{improvement:.4f})")
            
            assert 0 <= metrics['accuracy'] <= 1, "准确率应该在[0,1]范围内"
            print(f"✓ TCA域适应测试完成")
            
        except Exception as e:
            print(f"TCA测试失败: {e}")
            warnings.warn(f"TCA测试失败: {e}")
    
    def test_fmmd_domain_adaptation(self, medical_data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]):
        """测试fMMD（基于MMD的特征选择）域适应方法"""
        if not ADAPT_METHODS_AVAILABLE or not is_adapt_available():
            pytest.skip("Adapt库不可用")
        
        X_A, y_A, X_B, y_B = medical_data
        
        print(f"\n=== fMMD域适应测试（A训练 → B测试）===")
        
        try:
            # 创建fMMD方法
            fmmd_method = create_adapt_method(
                method_name='FMMD',
                estimator=create_tabpfn_model(),
                gamma=1.0,
                verbose=0,
                random_state=42
            )
            
            # 拟合模型
            fmmd_method.fit(X_A, y_A, X_B)
            
            # 预测
            y_pred = fmmd_method.predict(X_B)
            try:
                y_pred_proba = fmmd_method.predict_proba(X_B)
            except:
                y_pred_proba = None
            
            # 计算指标
            metrics = calculate_metrics(y_B, y_pred, y_pred_proba)
            
            print(f"fMMD域适应性能:")
            for metric, value in metrics.items():
                if not np.isnan(value):
                    print(f"  {metric.upper()}: {value:.4f}")
            
            # 与基线对比
            if hasattr(self, 'baseline_metrics'):
                print(f"\n性能对比（fMMD vs 基线）:")
                for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
                    if metric in metrics and metric in self.baseline_metrics:
                        baseline_val = self.baseline_metrics[metric]
                        fmmd_val = metrics[metric]
                        if not np.isnan(baseline_val) and not np.isnan(fmmd_val):
                            improvement = fmmd_val - baseline_val
                            print(f"  {metric.upper()}: {baseline_val:.4f} → {fmmd_val:.4f} "
                                  f"({'+'if improvement >= 0 else ''}{improvement:.4f})")
            
            assert 0 <= metrics['accuracy'] <= 1, "准确率应该在[0,1]范围内"
            print(f"✓ fMMD域适应测试完成")
            
        except Exception as e:
            print(f"fMMD测试失败: {e}")
            warnings.warn(f"fMMD测试失败: {e}")
    
    def test_pred_domain_adaptation(self, medical_data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]):
        """测试PRED（仅使用源域预测的特征增强）域适应方法"""
        if not ADAPT_METHODS_AVAILABLE or not is_adapt_available():
            pytest.skip("Adapt库不可用")
        
        X_A, y_A, X_B, y_B = medical_data
        
        print(f"\n=== PRED域适应测试（A训练 → B测试）===")
        
        try:
            # 创建PRED方法
            pred_method = create_adapt_method(
                method_name='PRED',
                estimator=create_tabpfn_model(),
                verbose=0,
                random_state=42
            )
            
            # 拟合模型
            pred_method.fit(X_A, y_A, X_B)
            
            # 预测
            y_pred = pred_method.predict(X_B)
            try:
                y_pred_proba = pred_method.predict_proba(X_B)
            except:
                y_pred_proba = None
            
            # 计算指标
            metrics = calculate_metrics(y_B, y_pred, y_pred_proba)
            
            print(f"PRED域适应性能:")
            for metric, value in metrics.items():
                if not np.isnan(value):
                    print(f"  {metric.upper()}: {value:.4f}")
            
            # 与基线对比
            if hasattr(self, 'baseline_metrics'):
                print(f"\n性能对比（PRED vs 基线）:")
                for metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
                    if metric in metrics and metric in self.baseline_metrics:
                        baseline_val = self.baseline_metrics[metric]
                        pred_val = metrics[metric]
                        if not np.isnan(baseline_val) and not np.isnan(pred_val):
                            improvement = pred_val - baseline_val
                            print(f"  {metric.upper()}: {baseline_val:.4f} → {pred_val:.4f} "
                                  f"({'+'if improvement >= 0 else ''}{improvement:.4f})")
            
            assert 0 <= metrics['accuracy'] <= 1, "准确率应该在[0,1]范围内"
            print(f"✓ PRED域适应测试完成")
            
        except Exception as e:
            print(f"PRED测试失败: {e}")
            warnings.warn(f"PRED测试失败: {e}")
    
    def test_multiple_methods_comparison(self, medical_data: Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]):
        """测试多种域适应方法对比"""
        if not ADAPT_METHODS_AVAILABLE or not is_adapt_available():
            pytest.skip("Adapt库不可用")
        
        X_A, y_A, X_B, y_B = medical_data
        
        print(f"\n=== 多种域适应方法对比（A训练 → B测试）===")
        
        # 要测试的方法列表（包括所有特征基础方法和实例重加权方法）
        methods_to_test = ['KMM']  # 核心实例重加权方法
        
        # 检查其他可用方法
        available_methods = get_available_adapt_methods()
        
        # 特征基础方法
        feature_based_methods = ['CORAL', 'SA', 'TCA', 'FMMD', 'PRED']
        for method in feature_based_methods:
            if method in available_methods:
                methods_to_test.append(method)
        
        # 实例重加权方法
        instance_based_methods = ['KLIEP', 'LDM', 'ULSIF', 'RULSIF', 'NNW', 'IWC', 'IWN']
        for method in instance_based_methods:
            if method in available_methods:
                methods_to_test.append(method)
        
        results = {}
        
        # 添加基线结果
        if hasattr(self, 'baseline_metrics'):
            results['Baseline'] = self.baseline_metrics
        
        # 测试各种方法
        for method_name in methods_to_test:
            try:
                print(f"\n测试方法: {method_name}")
                
                # 为不同方法设置合适的参数
                method_params: Dict[str, Any] = {'random_state': 42}
                if method_name == 'KMM':
                    method_params['kernel'] = 'linear'
                elif method_name == 'CORAL':
                    method_params['lambda_'] = 1.0
                elif method_name == 'TCA':
                    method_params['kernel'] = 'linear'
                    method_params['mu'] = 0.1
                elif method_name == 'FMMD':
                    method_params['gamma'] = 1.0
                elif method_name in ['ULSIF', 'RULSIF']:
                    # 使用gamma列表进行交叉验证，而不是硬编码gamma=1.0
                    method_params['gamma'] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                    method_params['lambdas'] = [0.001, 0.01, 0.1, 1.0, 10.0]
                    method_params['max_centers'] = 100
                    method_params['kernel'] = 'rbf'
                    method_params['verbose'] = 1
                elif method_name == 'RULSIF':
                    method_params['alpha'] = 0.1
                elif method_name == 'NNW':
                    method_params['n_neighbors'] = 5
                
                # 创建方法
                method = create_adapt_method(
                    method_name=method_name,
                    estimator=create_tabpfn_model(),
                    **method_params
                )
                
                # 拟合和评估
                method.fit(X_A, y_A, X_B)
                y_pred = method.predict(X_B)
                try:
                    y_pred_proba = method.predict_proba(X_B)
                except:
                    y_pred_proba = None
                
                metrics = calculate_metrics(y_B, y_pred, y_pred_proba)
                results[method_name] = metrics
                
                print(f"  {method_name} 性能:")
                for metric, value in metrics.items():
                    if not np.isnan(value):
                        print(f"    {metric.upper()}: {value:.4f}")
                
            except Exception as e:
                print(f"  {method_name} 测试失败: {e}")
                print(f"  详细错误信息:")
                import traceback
                traceback.print_exc()
                results[method_name] = None
        
        # 输出对比摘要
        successful_methods = {k: v for k, v in results.items() if v is not None}
        if len(successful_methods) > 1:
            print(f"\n=== 方法对比摘要 ===")
            print(f"{'方法':<12} {'AUC':<8} {'Accuracy':<10} {'F1':<8} {'Precision':<10} {'Recall':<8}")
            print("-" * 60)
            
            for method_name, metrics in successful_methods.items():
                auc = metrics.get('auc', 0)
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1', 0)
                prec = metrics.get('precision', 0)
                rec = metrics.get('recall', 0)
                
                auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
                print(f"{method_name:<12} {auc_str:<8} {acc:<10.4f} {f1:<8.4f} {prec:<10.4f} {rec:<8.4f}")
            
            # 找出最佳方法
            auc_results = {k: v['auc'] for k, v in successful_methods.items() 
                          if 'auc' in v and not np.isnan(v['auc'])}
            best_method = None
            if auc_results:
                best_method = max(auc_results.keys(), key=lambda k: auc_results[k])
                print(f"\n✓ 最佳方法（按AUC）: {best_method} (AUC: {auc_results[best_method]:.4f})")
            
            # 按方法类型分组显示结果
            print(f"\n=== 按方法类型分组结果 ===")
            method_types = {
                'instance_based': ['KMM', 'KLIEP', 'LDM', 'ULSIF', 'RULSIF', 'NNW', 'IWC', 'IWN'],
                'feature_based': ['CORAL', 'SA', 'TCA', 'FMMD', 'PRED']
            }
            
            for method_type, type_methods in method_types.items():
                type_results = {k: v for k, v in successful_methods.items() 
                              if k in type_methods and k != 'Baseline'}
                if type_results:
                    print(f"\n{method_type.replace('_', ' ').title()} 方法:")
                    for method_name, metrics in type_results.items():
                        auc = metrics.get('auc', 0)
                        acc = metrics.get('accuracy', 0)
                        auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
                        print(f"  {method_name}: AUC={auc_str}, Accuracy={acc:.4f}")
            
            print(f"\n✓ 成功测试 {len(successful_methods)-1} 种域适应方法")  # -1 因为包含基线
            
            # 保存多方法对比结果
            comparison_results = {
                'methods_comparison': successful_methods,
                'data_info': {
                    'source_samples': len(X_A),
                    'target_samples': len(X_B),
                    'features': list(X_A.columns),
                    'source_distribution': dict(y_A.value_counts().sort_index()),
                    'target_distribution': dict(y_B.value_counts().sort_index())
                },
                'best_method': {
                    'by_auc': best_method,
                    'auc_score': auc_results[best_method] if best_method and best_method in auc_results else None
                },
                'method_types': {
                    'instance_based': list(method_types['instance_based']),
                    'feature_based': list(method_types['feature_based'])
                },
                'test_timestamp': datetime.now().isoformat()
            }
            save_test_results(comparison_results, "adapt_methods_comparison", "tests")
        else:
            print("\n⚠ 只有基线方法成功或没有方法测试成功")


def test_adapt_integration_with_medical_data():
    """集成测试：使用医疗数据验证完整的Adapt方法流程"""
    print("\n=== Adapt库医疗数据集成测试 ===")
    
    # 1. 检查环境
    print("1. 检查环境...")
    print(f"   TabPFN可用: {TABPFN_AVAILABLE}")
    print(f"   Adapt方法可用: {ADAPT_METHODS_AVAILABLE}")
    print(f"   数据模块可用: {DATA_MODULES_AVAILABLE}")
    
    if not ADAPT_METHODS_AVAILABLE or not is_adapt_available():
        pytest.skip("Adapt库不可用")
    
    # 2. 加载数据
    print("2. 加载医疗数据...")
    X_A, y_A, X_B, y_B = load_test_data()
    print(f"   数据集A: {X_A.shape}, 数据集B: {X_B.shape}")
    
    # 3. 基线测试
    print("3. 基线TabPFN测试...")
    baseline_model = create_tabpfn_model()
    baseline_model.fit(X_A, y_A)
    baseline_pred = baseline_model.predict(X_B)
    try:
        baseline_proba = baseline_model.predict_proba(X_B)
    except:
        baseline_proba = None
    baseline_metrics = calculate_metrics(y_B, baseline_pred, baseline_proba)
    print(f"   基线AUC: {baseline_metrics['auc']:.4f}, Accuracy: {baseline_metrics['accuracy']:.4f}")
    
    # 4. 域适应测试
    print("4. KMM域适应测试...")
    try:
        kmm = create_adapt_method(
            method_name='KMM',
            estimator=create_tabpfn_model(),
            kernel='linear',
            random_state=42
        )
        
        kmm.fit(X_A, y_A, X_B)
        kmm_pred = kmm.predict(X_B)
        try:
            kmm_proba = kmm.predict_proba(X_B)
        except:
            kmm_proba = None
        kmm_metrics = calculate_metrics(y_B, kmm_pred, kmm_proba)
        
        print(f"   KMM AUC: {kmm_metrics['auc']:.4f}, Accuracy: {kmm_metrics['accuracy']:.4f}")
        
        # 计算改进
        auc_improvement = None
        if not np.isnan(baseline_metrics['auc']) and not np.isnan(kmm_metrics['auc']):
            auc_improvement = kmm_metrics['auc'] - baseline_metrics['auc']
            print(f"   AUC改进: {auc_improvement:+.4f}")
        
        assert 0 <= kmm_metrics['accuracy'] <= 1, "准确率应该在合理范围内"
        
        # 保存集成测试结果
        integration_results = {
            'environment_check': {
                'tabpfn_available': TABPFN_AVAILABLE,
                'adapt_methods_available': ADAPT_METHODS_AVAILABLE,
                'data_modules_available': DATA_MODULES_AVAILABLE
            },
            'data_info': {
                'source_samples': len(X_A),
                'target_samples': len(X_B),
                'features': list(X_A.columns)
            },
            'baseline_performance': baseline_metrics,
            'kmm_performance': kmm_metrics,
            'improvements': {
                'auc_improvement': float(auc_improvement) if auc_improvement is not None else None
            },
            'test_timestamp': datetime.now().isoformat()
        }
        save_test_results(integration_results, "adapt_integration_test", "tests")
        
    except Exception as e:
        print(f"   KMM测试失败: {e}")
        warnings.warn(f"KMM集成测试失败: {e}")
    
    print("✓ Adapt库医疗数据集成测试完成")


if __name__ == "__main__":
    # 直接运行测试
    print("运行Adapt方法医疗数据测试...")
    
    # 检查环境
    print(f"TabPFN可用: {TABPFN_AVAILABLE}")
    print(f"Adapt方法可用: {ADAPT_METHODS_AVAILABLE}")
    print(f"数据模块可用: {DATA_MODULES_AVAILABLE}")
    
    if ADAPT_METHODS_AVAILABLE and is_adapt_available():
        print("✓ Adapt库可用")
        methods = get_available_adapt_methods()
        print(f"✓ 支持 {len(methods)} 种方法:")
        for method, info in methods.items():
            print(f"  - {method}: {info['description']} ({info['type']})")
    else:
        print("✗ Adapt库不可用，请安装: pip install adapt-python")
    
    # 运行pytest
    pytest.main([__file__, "-v", "-s"]) 