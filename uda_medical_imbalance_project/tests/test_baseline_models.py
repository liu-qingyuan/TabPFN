"""
UDA Medical Imbalance Project - 基线模型测试

测试PKUPH和Mayo模型的功能和性能。
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.baseline_models import PKUPHModel, MayoModel, get_baseline_model, evaluate_baseline_models


class TestPKUPHModel:
    """测试PKUPH模型"""
    
    def setup_method(self):
        """设置测试数据"""
        self.model = PKUPHModel()
        self.n_samples = 100
        np.random.seed(42)
        
        # 创建包含PKUPH所需特征的测试数据
        self.features = ['Feature2', 'Feature48', 'Feature49', 'Feature4', 'Feature50', 'Feature53']
        self.X = pd.DataFrame({
            feature: np.random.normal(0, 1, self.n_samples) for feature in self.features
        })
        self.y = np.random.randint(0, 2, self.n_samples)
    
    def test_model_initialization(self):
        """测试模型初始化"""
        assert self.model.intercept_ == -4.496
        assert len(self.model.features) == 6
        assert len(self.model.coefficients) == 6
        assert not self.model.is_fitted_
    
    def test_model_fit(self):
        """测试模型拟合"""
        fitted_model = self.model.fit(self.X, self.y)
        assert fitted_model is self.model
        assert self.model.is_fitted_
    
    def test_predict_proba_before_fit(self):
        """测试未拟合时预测概率应该报错"""
        with pytest.raises(ValueError, match="模型尚未拟合"):
            self.model.predict_proba(self.X)
    
    def test_predict_proba_after_fit(self):
        """测试拟合后预测概率"""
        self.model.fit(self.X, self.y)
        proba = self.model.predict_proba(self.X)
        
        assert proba.shape == (self.n_samples, 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_predict_after_fit(self):
        """测试拟合后预测类别"""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        
        assert predictions.shape == (self.n_samples,)
        assert np.all(np.isin(predictions, [0, 1]))
    
    def test_get_feature_names(self):
        """测试获取特征名称"""
        feature_names = self.model.get_feature_names()
        assert feature_names == self.features
        assert feature_names is not self.model.features  # 应该返回副本
    
    def test_numpy_input(self):
        """测试numpy数组输入"""
        self.model.fit(self.X, self.y)
        X_numpy = self.X.values
        
        proba = self.model.predict_proba(X_numpy)
        predictions = self.model.predict(X_numpy)
        
        assert proba.shape == (self.n_samples, 2)
        assert predictions.shape == (self.n_samples,)
    
    def test_missing_features_warning(self, capsys):
        """测试缺失特征时的警告"""
        # 创建缺少某些特征的数据
        X_incomplete = self.X[['Feature2', 'Feature48']].copy()
        self.model.fit(X_incomplete, self.y[:len(X_incomplete)])
        
        self.model.predict_proba(X_incomplete)
        captured = capsys.readouterr()
        assert "警告" in captured.out


class TestMayoModel:
    """测试Mayo模型"""
    
    def setup_method(self):
        """设置测试数据"""
        self.model = MayoModel()
        self.n_samples = 100
        np.random.seed(42)
        
        # 创建包含Mayo所需特征的测试数据
        self.features = ['Feature2', 'Feature3', 'Feature5', 'Feature48', 'Feature49', 'Feature63']
        self.X = pd.DataFrame({
            feature: np.random.normal(0, 1, self.n_samples) for feature in self.features
        })
        self.y = np.random.randint(0, 2, self.n_samples)
    
    def test_model_initialization(self):
        """测试模型初始化"""
        assert self.model.intercept_ == -6.8272
        assert len(self.model.features) == 6
        assert len(self.model.coefficients) == 6
        assert not self.model.is_fitted_
    
    def test_model_fit(self):
        """测试模型拟合"""
        fitted_model = self.model.fit(self.X, self.y)
        assert fitted_model is self.model
        assert self.model.is_fitted_
    
    def test_predict_proba_after_fit(self):
        """测试拟合后预测概率"""
        self.model.fit(self.X, self.y)
        proba = self.model.predict_proba(self.X)
        
        assert proba.shape == (self.n_samples, 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_predict_after_fit(self):
        """测试拟合后预测类别"""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        
        assert predictions.shape == (self.n_samples,)
        assert np.all(np.isin(predictions, [0, 1]))
    
    def test_get_feature_names(self):
        """测试获取特征名称"""
        feature_names = self.model.get_feature_names()
        assert feature_names == self.features
        assert feature_names is not self.model.features  # 应该返回副本


class TestBaselineModelFactory:
    """测试基线模型工厂函数"""
    
    def test_get_baseline_model_pkuph(self):
        """测试获取PKUPH模型"""
        model = get_baseline_model('pkuph')
        assert isinstance(model, PKUPHModel)
        
        model = get_baseline_model('PKUPH')  # 测试大小写不敏感
        assert isinstance(model, PKUPHModel)
    
    def test_get_baseline_model_mayo(self):
        """测试获取Mayo模型"""
        model = get_baseline_model('mayo')
        assert isinstance(model, MayoModel)
        
        model = get_baseline_model('MAYO')  # 测试大小写不敏感
        assert isinstance(model, MayoModel)
    
    def test_get_baseline_model_invalid(self):
        """测试获取不支持的模型"""
        with pytest.raises(ValueError, match="不支持的基线模型"):
            get_baseline_model('invalid_model')


class TestBaselineModelEvaluation:
    """测试基线模型评估功能"""
    
    def setup_method(self):
        """设置测试数据"""
        self.n_samples = 200
        np.random.seed(42)
        
        # 创建包含所有基线模型特征的测试数据
        all_features = ['Feature2', 'Feature3', 'Feature4', 'Feature5', 
                       'Feature48', 'Feature49', 'Feature50', 'Feature53', 'Feature63']
        self.X = pd.DataFrame({
            feature: np.random.normal(0, 1, self.n_samples) for feature in all_features
        })
        self.y = np.random.randint(0, 2, self.n_samples)
        
        # 分割训练和测试集
        split_idx = self.n_samples // 2
        self.X_train = self.X[:split_idx]
        self.y_train = self.y[:split_idx]
        self.X_test = self.X[split_idx:]
        self.y_test = self.y[split_idx:]
    
    def test_evaluate_baseline_models_default(self):
        """测试默认评估（PKUPH和Mayo）"""
        results = evaluate_baseline_models(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        assert 'pkuph' in results
        assert 'mayo' in results
        
        for model_name, result in results.items():
            assert 'error' not in result
            assert 'model' in result
            assert 'predictions' in result
            assert 'probabilities' in result
            assert 'metrics' in result
            
            # 检查指标
            metrics = result['metrics']
            assert 'accuracy' in metrics
            assert 'auc' in metrics
            assert 'f1' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            
            # 检查指标范围
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['auc'] <= 1
            assert 0 <= metrics['f1'] <= 1
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
    
    def test_evaluate_baseline_models_specific(self):
        """测试评估特定模型"""
        results = evaluate_baseline_models(
            self.X_train, self.y_train, self.X_test, self.y_test,
            models=['pkuph']
        )
        
        assert len(results) == 1
        assert 'pkuph' in results
        assert 'mayo' not in results
    
    def test_evaluate_baseline_models_with_missing_features(self):
        """测试缺失特征时的评估"""
        # 只保留部分特征
        X_incomplete = self.X[['Feature2', 'Feature48']].copy()
        
        results = evaluate_baseline_models(
            X_incomplete, self.y_train, X_incomplete, self.y_test
        )
        
        # 应该仍然能运行，但可能有警告
        assert 'pkuph' in results
        assert 'mayo' in results


class TestModelPerformanceConsistency:
    """测试模型性能一致性"""
    
    def test_pkuph_model_deterministic(self):
        """测试PKUPH模型的确定性"""
        n_samples = 50
        features = ['Feature2', 'Feature48', 'Feature49', 'Feature4', 'Feature50', 'Feature53']
        
        # 创建固定的测试数据
        np.random.seed(123)
        X = pd.DataFrame({
            feature: np.random.normal(0, 1, n_samples) for feature in features
        })
        y = np.random.randint(0, 2, n_samples)
        
        # 多次运行应该得到相同结果
        model1 = PKUPHModel()
        model1.fit(X, y)
        pred1 = model1.predict_proba(X)
        
        model2 = PKUPHModel()
        model2.fit(X, y)
        pred2 = model2.predict_proba(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
    
    def test_mayo_model_deterministic(self):
        """测试Mayo模型的确定性"""
        n_samples = 50
        features = ['Feature2', 'Feature3', 'Feature5', 'Feature48', 'Feature49', 'Feature63']
        
        # 创建固定的测试数据
        np.random.seed(123)
        X = pd.DataFrame({
            feature: np.random.normal(0, 1, n_samples) for feature in features
        })
        y = np.random.randint(0, 2, n_samples)
        
        # 多次运行应该得到相同结果
        model1 = MayoModel()
        model1.fit(X, y)
        pred1 = model1.predict_proba(X)
        
        model2 = MayoModel()
        model2.fit(X, y)
        pred2 = model2.predict_proba(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2)


def test_baseline_models_integration():
    """集成测试：测试基线模型的完整工作流程"""
    print("\n🧪 运行基线模型集成测试...")
    
    # 创建测试数据
    n_samples = 100
    np.random.seed(42)
    
    # 包含所有基线模型需要的特征
    all_features = ['Feature2', 'Feature3', 'Feature4', 'Feature5', 
                   'Feature48', 'Feature49', 'Feature50', 'Feature53', 'Feature63']
    X = pd.DataFrame({
        feature: np.random.normal(0, 1, n_samples) for feature in all_features
    })
    y = np.random.randint(0, 2, n_samples)
    
    print(f"测试数据: {X.shape}, 标签分布: {np.bincount(y)}")
    
    # 分割数据
    split_idx = n_samples // 2
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 测试每个模型
    models = ['pkuph', 'mayo']
    results = {}
    
    for model_name in models:
        print(f"\n📊 测试 {model_name.upper()} 模型:")
        
        # 创建模型
        model = get_baseline_model(model_name)
        print(f"  特征: {model.get_feature_names()}")
        
        # 训练和预测
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # 计算指标
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        
        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities[:, 1])
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        print(f"  准确率: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  F1分数: {f1:.4f}")
        
        results[model_name] = {
            'accuracy': accuracy,
            'auc': auc,
            'f1': f1
        }
    
    # 使用评估函数进行对比
    print(f"\n🔄 使用评估函数进行对比:")
    eval_results = evaluate_baseline_models(X_train, y_train, X_test, y_test)
    
    for model_name, result in eval_results.items():
        if 'error' not in result:
            metrics = result['metrics']
            print(f"  {model_name.upper()}: AUC={metrics['auc']:.4f}, "
                  f"ACC={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        else:
            print(f"  {model_name.upper()}: 错误 - {result['error']}")
    
    print("\n✅ 基线模型集成测试完成!")
    return results


if __name__ == "__main__":
    # 运行集成测试
    test_baseline_models_integration()
    
    # 运行pytest测试
    print("\n🧪 运行pytest测试...")
    pytest.main([__file__, "-v"]) 