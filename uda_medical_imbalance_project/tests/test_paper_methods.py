"""
UDA Medical Imbalance Project - 论文方法测试

测试基于predict_healthcare_LR.py的论文方法功能和性能。
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.paper_methods import (
    PaperLRModel, get_paper_method, evaluate_paper_methods, 
    compare_with_original_lr_script
)


class TestPaperLRModel:
    """测试论文LR模型"""
    
    def setup_method(self):
        """设置测试数据"""
        self.model = PaperLRModel()
        self.n_samples = 100
        np.random.seed(42)
        
        # 创建包含论文方法所需特征的测试数据
        self.features = [
            'Feature2', 'Feature5', 'Feature48', 'Feature49', 'Feature50', 
            'Feature52', 'Feature56', 'Feature57', 'Feature61', 'Feature42', 'Feature43'
        ]
        self.X = pd.DataFrame({
            feature: np.random.normal(0, 1, self.n_samples) for feature in self.features
        })
        self.y = np.random.randint(0, 2, self.n_samples)
    
    def test_model_initialization(self):
        """测试模型初始化"""
        assert self.model.intercept_ == -1.137
        assert len(self.model.features) == 11
        assert len(self.model.coefficients) == 11
        assert not self.model.is_fitted_
        
        # 检查特征和系数的对应关系
        expected_features = [
            'Feature2', 'Feature5', 'Feature48', 'Feature49', 'Feature50', 
            'Feature52', 'Feature56', 'Feature57', 'Feature61', 'Feature42', 'Feature43'
        ]
        assert self.model.features == expected_features
        
        # 检查系数值
        assert self.model.coefficients['Feature2'] == 0.036
        assert self.model.coefficients['Feature5'] == 0.380
        assert self.model.coefficients['Feature50'] == -0.290
    
    def test_model_fit(self):
        """测试模型拟合"""
        fitted_model = self.model.fit(self.X, self.y)
        assert fitted_model is self.model
        assert self.model.is_fitted_
    
    def test_calculate_risk_score(self):
        """测试风险评分计算"""
        self.model.fit(self.X, self.y)
        risk_scores = self.model.calculate_risk_score(self.X)
        
        assert risk_scores.shape == (self.n_samples,)
        assert isinstance(risk_scores, np.ndarray)
        
        # 手动计算第一个样本的风险评分进行验证
        first_sample = self.X.iloc[0]
        expected_score = self.model.intercept_
        for feature, coef in self.model.coefficients.items():
            expected_score += coef * first_sample[feature]
        
        np.testing.assert_almost_equal(risk_scores[0], expected_score, decimal=6)
    
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
        
        # 验证概率计算公式
        risk_scores = self.model.calculate_risk_score(self.X)
        expected_p_malignant = np.exp(risk_scores) / (1 + np.exp(risk_scores))
        np.testing.assert_array_almost_equal(proba[:, 1], expected_p_malignant)
    
    def test_predict_after_fit(self):
        """测试拟合后预测类别"""
        self.model.fit(self.X, self.y)
        predictions = self.model.predict(self.X)
        
        assert predictions.shape == (self.n_samples,)
        assert np.all(np.isin(predictions, [0, 1]))
        
        # 验证预测与概率的一致性
        proba = self.model.predict_proba(self.X)
        expected_predictions = np.argmax(proba, axis=1)
        np.testing.assert_array_equal(predictions, expected_predictions)
    
    def test_get_feature_names(self):
        """测试获取特征名称"""
        feature_names = self.model.get_feature_names()
        assert feature_names == self.features
        assert feature_names is not self.model.features  # 应该返回副本
    
    def test_get_risk_scores(self):
        """测试获取风险评分"""
        self.model.fit(self.X, self.y)
        risk_scores = self.model.get_risk_scores(self.X)
        
        assert risk_scores.shape == (self.n_samples,)
        
        # 应该与calculate_risk_score返回相同结果
        expected_scores = self.model.calculate_risk_score(self.X)
        np.testing.assert_array_equal(risk_scores, expected_scores)
    
    def test_get_risk_scores_before_fit(self):
        """测试未拟合时获取风险评分应该报错"""
        with pytest.raises(ValueError, match="模型尚未拟合"):
            self.model.get_risk_scores(self.X)
    
    def test_numpy_input(self):
        """测试numpy数组输入"""
        self.model.fit(self.X, self.y)
        X_numpy = self.X.values
        
        proba = self.model.predict_proba(X_numpy)
        predictions = self.model.predict(X_numpy)
        risk_scores = self.model.get_risk_scores(X_numpy)
        
        assert proba.shape == (self.n_samples, 2)
        assert predictions.shape == (self.n_samples,)
        assert risk_scores.shape == (self.n_samples,)
    
    def test_missing_features_warning(self, capsys):
        """测试缺失特征时的警告"""
        # 创建缺少某些特征的数据
        X_incomplete = self.X[['Feature2', 'Feature5', 'Feature48']].copy()
        self.model.fit(X_incomplete, self.y[:len(X_incomplete)])
        
        self.model.predict_proba(X_incomplete)
        captured = capsys.readouterr()
        assert "警告" in captured.out


class TestPaperMethodFactory:
    """测试论文方法工厂函数"""
    
    def test_get_paper_method_paper_lr(self):
        """测试获取论文LR方法"""
        model = get_paper_method('paper_lr')
        assert isinstance(model, PaperLRModel)
        
        # 测试别名
        model = get_paper_method('lr_paper')
        assert isinstance(model, PaperLRModel)
        
        model = get_paper_method('paper_method')
        assert isinstance(model, PaperLRModel)
    
    def test_get_paper_method_case_insensitive(self):
        """测试大小写不敏感"""
        model = get_paper_method('PAPER_LR')
        assert isinstance(model, PaperLRModel)
    
    def test_get_paper_method_invalid(self):
        """测试获取不支持的方法"""
        with pytest.raises(ValueError, match="不支持的论文方法"):
            get_paper_method('invalid_method')


class TestPaperMethodEvaluation:
    """测试论文方法评估功能"""
    
    def setup_method(self):
        """设置测试数据"""
        self.n_samples = 200
        np.random.seed(42)
        
        # 创建包含论文方法所需特征的测试数据
        self.features = [
            'Feature2', 'Feature5', 'Feature48', 'Feature49', 'Feature50', 
            'Feature52', 'Feature56', 'Feature57', 'Feature61', 'Feature42', 'Feature43'
        ]
        self.X = pd.DataFrame({
            feature: np.random.normal(0, 1, self.n_samples) for feature in self.features
        })
        self.y = np.random.randint(0, 2, self.n_samples)
        
        # 分割训练和测试集
        split_idx = self.n_samples // 2
        self.X_train = self.X[:split_idx]
        self.y_train = self.y[:split_idx]
        self.X_test = self.X[split_idx:]
        self.y_test = self.y[split_idx:]
    
    def test_evaluate_paper_methods_default(self):
        """测试默认评估（paper_lr）"""
        results = evaluate_paper_methods(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        
        assert 'paper_lr' in results
        
        result = results['paper_lr']
        assert 'error' not in result
        assert 'model' in result
        assert 'predictions' in result
        assert 'probabilities' in result
        assert 'metrics' in result
        assert 'risk_scores' in result
        assert 'risk_score_stats' in result
        
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
        
        # 检查风险评分统计
        risk_stats = result['risk_score_stats']
        assert 'mean' in risk_stats
        assert 'std' in risk_stats
        assert 'min' in risk_stats
        assert 'max' in risk_stats
    
    def test_evaluate_paper_methods_specific(self):
        """测试评估特定方法"""
        results = evaluate_paper_methods(
            self.X_train, self.y_train, self.X_test, self.y_test,
            methods=['lr_paper']
        )
        
        assert len(results) == 1
        assert 'lr_paper' in results
    
    def test_evaluate_paper_methods_with_missing_features(self):
        """测试缺失特征时的评估"""
        # 只保留部分特征
        X_incomplete = self.X[['Feature2', 'Feature5', 'Feature48']].copy()
        
        results = evaluate_paper_methods(
            X_incomplete, self.y_train, X_incomplete, self.y_test
        )
        
        # 应该仍然能运行，但可能有警告
        assert 'paper_lr' in results


class TestCompareWithOriginalScript:
    """测试与原始脚本的对比功能"""
    
    def setup_method(self):
        """设置测试数据"""
        self.n_samples = 100
        np.random.seed(42)
        
        # 创建包含论文方法所需特征的测试数据
        self.features = [
            'Feature2', 'Feature5', 'Feature48', 'Feature49', 'Feature50', 
            'Feature52', 'Feature56', 'Feature57', 'Feature61', 'Feature42', 'Feature43'
        ]
        self.X = pd.DataFrame({
            feature: np.random.normal(0, 1, self.n_samples) for feature in self.features
        })
        self.y = np.random.randint(0, 2, self.n_samples)
    
    def test_compare_with_original_lr_script(self):
        """测试与原始LR脚本的对比"""
        results = compare_with_original_lr_script(self.X, self.y)
        
        assert 'model_implementation' in results
        assert results['model_implementation'] == 'PaperLRModel'
        
        assert 'metrics' in results
        metrics = results['metrics']
        assert 'accuracy' in metrics
        assert 'auc' in metrics
        assert 'f1' in metrics
        assert 'acc_0' in metrics
        assert 'acc_1' in metrics
        
        assert 'predictions' in results
        assert 'probabilities' in results
        assert 'risk_scores' in results
        assert 'risk_score_stats' in results
        assert 'feature_names' in results
        
        # 检查数据形状
        assert len(results['predictions']) == self.n_samples
        assert len(results['probabilities']) == self.n_samples
        assert len(results['risk_scores']) == self.n_samples
        assert results['feature_names'] == self.features


class TestModelPerformanceConsistency:
    """测试模型性能一致性"""
    
    def test_paper_lr_model_deterministic(self):
        """测试论文LR模型的确定性"""
        n_samples = 50
        features = [
            'Feature2', 'Feature5', 'Feature48', 'Feature49', 'Feature50', 
            'Feature52', 'Feature56', 'Feature57', 'Feature61', 'Feature42', 'Feature43'
        ]
        
        # 创建固定的测试数据
        np.random.seed(123)
        X = pd.DataFrame({
            feature: np.random.normal(0, 1, n_samples) for feature in features
        })
        y = np.random.randint(0, 2, n_samples)
        
        # 多次运行应该得到相同结果
        model1 = PaperLRModel()
        model1.fit(X, y)
        pred1 = model1.predict_proba(X)
        risk1 = model1.get_risk_scores(X)
        
        model2 = PaperLRModel()
        model2.fit(X, y)
        pred2 = model2.predict_proba(X)
        risk2 = model2.get_risk_scores(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
        np.testing.assert_array_almost_equal(risk1, risk2)
    
    def test_risk_score_formula_validation(self):
        """测试风险评分公式的正确性"""
        # 创建简单的测试数据
        X = pd.DataFrame({
            'Feature2': [1.0],
            'Feature5': [2.0],
            'Feature48': [0.5],
            'Feature49': [1.5],
            'Feature50': [0.8],
            'Feature52': [1.2],
            'Feature56': [0.3],
            'Feature57': [0.7],
            'Feature61': [1.1],
            'Feature42': [0.9],
            'Feature43': [1.3]
        })
        y = np.array([1])
        
        model = PaperLRModel()
        model.fit(X, y)
        
        # 手动计算期望的风险评分
        expected_risk = (-1.137 + 
                        0.036 * 1.0 +
                        0.380 * 2.0 +
                        0.195 * 0.5 +
                        0.016 * 1.5 +
                        (-0.290) * 0.8 +
                        0.026 * 1.2 +
                        (-0.168) * 0.3 +
                        (-0.236) * 0.7 +
                        0.052 * 1.1 +
                        0.018 * 0.9 +
                        0.004 * 1.3)
        
        calculated_risk = model.get_risk_scores(X)[0]
        np.testing.assert_almost_equal(calculated_risk, expected_risk, decimal=6)
        
        # 验证概率计算
        expected_prob = np.exp(expected_risk) / (1 + np.exp(expected_risk))
        calculated_prob = model.predict_proba(X)[0, 1]
        np.testing.assert_almost_equal(calculated_prob, expected_prob, decimal=6)


def test_paper_methods_integration():
    """集成测试：测试论文方法的完整工作流程"""
    print("\n🧪 运行论文方法集成测试...")
    
    # 创建测试数据
    n_samples = 100
    np.random.seed(42)
    
    # 包含论文方法需要的特征
    features = [
        'Feature2', 'Feature5', 'Feature48', 'Feature49', 'Feature50', 
        'Feature52', 'Feature56', 'Feature57', 'Feature61', 'Feature42', 'Feature43'
    ]
    X = pd.DataFrame({
        feature: np.random.normal(0, 1, n_samples) for feature in features
    })
    y = np.random.randint(0, 2, n_samples)
    
    print(f"测试数据: {X.shape}, 标签分布: {np.bincount(y)}")
    
    # 分割数据
    split_idx = n_samples // 2
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 测试论文方法
    print(f"\n📊 测试论文LR方法:")
    
    # 创建模型
    model = get_paper_method('paper_lr')
    print(f"  特征: {model.get_feature_names()}")
    
    # 训练和预测
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    risk_scores = model.get_risk_scores(X_test)
    
    # 计算指标
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities[:, 1])
    f1 = f1_score(y_test, predictions, zero_division=0)
    
    print(f"  准确率: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  风险评分范围: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]")
    print(f"  风险评分均值: {risk_scores.mean():.3f}")
    
    print("\n✅ 论文方法集成测试完成!")
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'risk_score_stats': {
            'mean': risk_scores.mean(),
            'std': risk_scores.std(),
            'min': risk_scores.min(),
            'max': risk_scores.max()
        }
    }


if __name__ == "__main__":
    # 运行集成测试
    test_paper_methods_integration()
    
    # 运行pytest测试
    print("\n🧪 运行pytest测试...")
    pytest.main([__file__, "-v"]) 