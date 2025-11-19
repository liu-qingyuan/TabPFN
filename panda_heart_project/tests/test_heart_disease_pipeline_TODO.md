# 测试方案 TODO

## 模块概述
**文件**: `tests/test_heart_disease_pipeline.py`
**功能**: PANDA-Heart项目完整测试框架
**负责人**: [待分配]
**预计工时**: 24小时

---

## 📋 详细任务清单

### TASK-022: 单元测试实现
**优先级**: 🔥 High | **预计工时**: 8小时 | **截止**: Week 5

#### 子任务
- [ ] **TASK-022-1**: 数据处理测试
  - **数据加载**: UCI 4中心数据加载正确性
  - **特征编码**: 14个临床特征编码验证
  - **缺失值处理**: 各中心缺失值策略验证
  - **数据质量**: 数据范围和类型验证

- [ ] **TASK-022-2**: 模型测试
  - **TabPFN模型**: 预训练模型加载和推理
  - **域适应算法**: TCA/CORAL/SA算法正确性
  - **基线模型**: 传统ML模型训练和预测
  - **集成策略**: PANDA_TabPFN集成验证

- [ ] **TASK-022-3**: 评估测试
  - **医学指标**: AUC、敏感性、特异性计算
  - **校准指标**: Brier score、ECE计算验证
  - **跨域指标**: 性能保持率、适应增益计算
  - **统计检验**: 显著性检验和效应量计算

#### 验收标准
- [ ] 单元测试覆盖率>90%
- [ ] 所有关键功能测试通过
- [ ] 边界条件测试完整

#### 技术要求
```python
# 伪代码示例
import pytest
import numpy as np
from unittest.mock import Mock, patch
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

class TestDataProcessing:
    """数据处理模块测试"""

    @pytest.fixture
    def sample_heart_data(self):
        """生成示例心脏病数据"""
        np.random.seed(42)
        n_samples = 200
        n_features = 14

        X = np.random.randn(n_samples, n_features)
        X[:, 0] = np.abs(X[:, 0] * 20 + 50)  # age: 30-90
        X[:, 1] = np.random.randint(0, 2, n_samples)  # sex: 0/1
        X[:, 2] = np.random.randint(1, 5, n_samples)  # cp: 1-4

        y = np.random.randint(0, 2, n_samples)
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']

        return X, y, feature_names

    def test_feature_encoding_correctness(self, sample_heart_data):
        """测试特征编码正确性"""
        X, y, feature_names = sample_heart_data
        encoder = ClinicalFeatureEncoder()

        # 测试编码
        X_encoded = encoder.fit_transform(X, y)

        # 验证特征维度
        assert X_encoded.shape[0] == X.shape[0], "样本数量不应改变"

        # 验证年龄范围
        age_idx = feature_names.index('age')
        assert np.all(X_encoded[:, age_idx] >= -3) and np.all(X_encoded[:, age_idx] <= 3), "年龄应在合理范围内"

        # 验证二进制特征
        sex_idx = feature_names.index('sex')
        assert np.all(np.isin(X_encoded[:, sex_idx], [0, 1])), "性别应为0/1"

    def test_missing_value_handling(self):
        """测试缺失值处理"""
        # 创建带缺失值的数据
        X = np.random.randn(100, 14)
        X[::5, 2] = np.nan  # 20%缺失率
        y = np.random.randint(0, 2, 100)

        processor = DataProcessor()
        X_processed = processor.fit_transform(X, y)

        # 验证无缺失值
        assert not np.isnan(X_processed).any(), "处理后不应有缺失值"

    def test_clinical_feature_validation(self, sample_heart_data):
        """测试临床特征验证"""
        X, y, feature_names = sample_heart_data
        validator = ClinicalFeatureValidator()

        # 测试正常数据
        assert validator.validate_features(X, feature_names), "正常数据应通过验证"

        # 测试异常年龄
        X_invalid = X.copy()
        X_invalid[0, 0] = 150  # 异常年龄
        assert not validator.validate_features(X_invalid, feature_names), "异常年龄应被检测"

class TestDomainAdaptation:
    """域适应模块测试"""

    @pytest.fixture
    def source_target_data(self):
        """生成源域和目标域数据"""
        # 源域数据
        X_source, y_source = make_classification(
            n_samples=300, n_features=14, n_informative=10,
            n_redundant=2, random_state=42
        )

        # 目标域数据（分布偏移）
        X_target, y_target = make_classification(
            n_samples=200, n_features=14, n_informative=10,
            n_redundant=2, shift=0.5, random_state=123
        )

        return X_source, y_source, X_target, y_target

    def test_tca_algorithm(self, source_target_data):
        """测试TCA算法"""
        X_source, y_source, X_target, y_target = source_target_data
        tca = HeartDiseaseTCA(mu=0.1, n_components=10)

        # 测试训练
        tca.fit(X_source, y_source, X_target)
        assert hasattr(tca, 'components_'), "训练后应有components_属性"

        # 测试变换
        X_source_tca, X_target_tca = tca.transform(X_source, X_target)
        assert X_source_tca.shape[1] == 10, "变换后维度应为10"
        assert X_target_tca.shape[1] == 10, "变换后维度应为10"

        # 测试MMD距离减小
        original_mmd = compute_mmd_distance(X_source, X_target)
        adapted_mmd = compute_mmd_distance(X_source_tca, X_target_tca)
        assert adapted_mmd < original_mmd, "域适应应减小MMD距离"

    def test_coral_algorithm(self, source_target_data):
        """测试CORAL算法"""
        X_source, y_source, X_target, y_target = source_target_data
        coral = HeartDiseaseCORAL(reg_param=1e-3)

        # 测试训练
        coral.fit(X_source, y_source, X_target)
        assert hasattr(coral, 'transformation_matrix_'), "训练后应有变换矩阵"

        # 测试变换
        X_source_coral = coral.transform(X_source)
        X_target_coral = coral.transform(X_target)
        assert X_source_coral.shape == X_source.shape, "CORAL不应改变维度"

    def test_sa_algorithm(self, source_target_data):
        """测试SA算法"""
        X_source, y_source, X_target, y_target = source_target_data
        sa = HeartDiseaseSA(n_components=0.9)

        # 测试训练
        sa.fit(X_source, y_source, X_target)
        assert hasattr(sa, 'alignment_matrix_'), "训练后应有对齐矩阵"

        # 测试变换
        X_source_sa = sa.transform(X_source)
        X_target_sa = sa.transform(X_target)
        assert X_source_sa.shape[1] <= X_source.shape[1], "SA应降维"

class TestModelEvaluation:
    """模型评估测试"""

    @pytest.fixture
    def model_predictions(self):
        """生成模型预测结果"""
        n_samples = 200
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.rand(n_samples)
        y_pred = (y_prob > 0.5).astype(int)

        return y_true, y_pred, y_prob

    def test_medical_metrics_computation(self, model_predictions):
        """测试医学指标计算"""
        y_true, y_pred, y_prob = model_predictions
        evaluator = MedicalMetrics()

        metrics = evaluator.compute_core_metrics(y_true, y_pred, y_prob)

        # 验证指标范围
        assert 0 <= metrics['auc_roc'] <= 1, "AUC应在0-1范围内"
        assert 0 <= metrics['sensitivity'] <= 1, "敏感性应在0-1范围内"
        assert 0 <= metrics['specificity'] <= 1, "特异性应在0-1范围内"

    def test_calibration_metrics(self, model_predictions):
        """测试校准指标计算"""
        y_true, _, y_prob = model_predictions
        evaluator = MedicalMetrics()

        calibration_metrics = evaluator.compute_calibration_metrics(y_true, y_prob)

        # 验证校准指标
        assert calibration_metrics['brier_score'] >= 0, "Brier分数应为非负"
        assert 0 <= calibration_metrics['ece'] <= 1, "ECE应在0-1范围内"
```

---

### TASK-023: 集成测试实现
**优先级**: 🔥 High | **预计工时**: 8小时 | **截止**: Week 6

#### 子任务
- [ ] **TASK-023-1**: 端到端流程测试
  - **完整流程**: 数据加载→模型训练→评估→可视化
  - **多模型对比**: 7种模型的完整对比流程
  - **多域适应**: 6种域适应方法的集成测试
  - **错误处理**: 流程中异常情况的处理

- [ ] **TASK-023-2**: 跨中心集成测试
  - **LOCO-CV流程**: Leave-One-Center-Out完整测试
  - **多中心数据**: 4个中心数据的兼容性测试
  - **结果一致性**: 重复实验的结果一致性验证
  - **性能基准**: 与预期性能基准的对比

- [ ] **TASK-023-3**: 系统集成测试
  - **配置管理**: 配置文件和参数管理测试
  - **资源管理**: 内存、计算资源使用测试
  - **并发处理**: 多进程/多线程并发测试
  - **持久化**: 模型保存和加载测试

#### 验收标准
- [ ] 端到端流程测试覆盖所有场景
- [ ] 跨中心集成测试验证泛化能力
- [ ] 系统集成测试确保稳定性和性能

#### 技术要求
```python
# 伪代码示例
class TestEndToEndPipeline:
    """端到端流程测试"""

    @pytest.mark.integration
    def test_complete_single_center_experiment(self):
        """测试完整单中心实验"""
        # 设置实验配置
        config = ExperimentConfig()
        config.experiment_type = 'single_center'
        config.models = ['PANDA_TabPFN', 'LASSO_LR']
        config.uda_methods = ['No_UDA']

        # 运行实验
        runner = ExperimentRunner(config)
        results = runner.run_experiment()

        # 验证结果结构
        assert 'single_center' in results, "应包含单中心结果"
        assert len(results['single_center']) > 0, "应有实验结果"

        # 验证结果完整性
        for center_result in results['single_center'].values():
            assert 'cv_results' in center_result, "应包含交叉验证结果"
            assert 'model_performance' in center_result, "应包含模型性能"

    @pytest.mark.integration
    def test_complete_two_center_experiment(self):
        """测试完整两中心跨域实验"""
        config = ExperimentConfig()
        config.experiment_type = 'two_center'
        config.source_centers = ['Cleveland', 'Hungarian']
        config.target_centers = ['VA', 'Switzerland']
        config.models = ['PANDA_TabPFN', 'TabPFN_Only']
        config.uda_methods = ['TCA', 'No_UDA']

        runner = ExperimentRunner(config)
        results = runner.run_experiment()

        # 验证跨域结果
        assert 'domain_adaptation' in results, "应包含域适应结果"
        assert len(results['domain_adaptation']) > 0, "应有域适应实验结果"

        # 验证适应效果
        for domain_result in results['domain_adaptation']:
            if 'TCA' in domain_result:
                assert 'adaptation_gain' in domain_result['TCA'], "应包含适应增益"

    @pytest.mark.integration
    def test_complete_multi_center_experiment(self):
        """测试完整多中心LOCO-CV实验"""
        config = ExperimentConfig()
        config.experiment_type = 'multi_center'
        config.validation_method = 'loco_cv'
        config.models = ['PANDA_TabPFN', 'LASSO_LR', 'XGBoost']
        config.uda_methods = ['TCA', 'CORAL', 'No_UDA']

        runner = ExperimentRunner(config)
        results = runner.run_experiment()

        # 验证LOCO-CV结果
        assert 'multi_center' in results, "应包含多中心结果"
        assert len(results['multi_center']['loco_results']) == 4, "应有4个LOCO实验"

        # 验证性能对比
        performance_comparison = results['multi_center']['performance_comparison']
        assert len(performance_comparison) > 0, "应有性能对比结果"

class TestSystemIntegration:
    """系统集成测试"""

    @pytest.mark.system
    def test_configuration_management(self):
        """测试配置管理"""
        config_manager = HeartDiseaseConfigManager()

        # 测试配置加载
        config = config_manager.get_config('data')
        assert 'feature_names' in config, "数据配置应包含特征名称"

        # 测试配置更新
        config_manager.update_config('experiment', {'n_repetitions': 10})
        updated_config = config_manager.get_config('experiment')
        assert updated_config['n_repetitions'] == 10, "配置更新应生效"

    @pytest.mark.system
    def test_resource_management(self):
        """测试资源管理"""
        # 监控内存使用
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # 运行大数据集实验
        X_large = np.random.randn(10000, 14)
        y_large = np.random.randint(0, 2, 10000)

        model = PANDATabPFN()
        model.fit(X_large[:8000], y_large[:8000])
        _ = model.predict_proba(X_large[8000:])

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 验证内存使用合理（<2GB）
        assert memory_increase < 2 * 1024**3, "内存增长应控制在2GB以内"

    @pytest.mark.system
    def test_model_persistence(self):
        """测试模型持久化"""
        # 训练模型
        X_train, y_train = make_classification(n_samples=1000, n_features=14, random_state=42)
        model = PANDATabPFN()
        model.fit(X_train, y_train)

        # 保存模型
        save_path = "test_model.pkl"
        model.save_model(save_path)
        assert os.path.exists(save_path), "模型文件应被保存"

        # 加载模型
        loaded_model = PANDATabPFN.load_model(save_path)
        assert isinstance(loaded_model, PANDATabPFN), "加载的应为PANDATabPFN实例"

        # 验证预测一致性
        X_test = np.random.randn(100, 14)
        original_pred = model.predict_proba(X_test)
        loaded_pred = loaded_model.predict_proba(X_test)
        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=6)

        # 清理测试文件
        os.remove(save_path)
```

---

### TASK-024: 性能测试实现
**优先级**: 🔥 Medium | **预计工时**: 8小时 | **截止**: Week 7

#### 子任务
- [ ] **TASK-024-1**: 计算性能测试
  - **训练时间**: 各模型训练时间基准测试
  - **推理时间**: 模型预测时间性能测试
  - **内存使用**: 不同数据规模的内存消耗
  - **并发性能**: 多进程/多线程性能提升

- [ ] **TASK-024-2**: 可扩展性测试
  - **数据规模**: 不同样本数量的性能变化
  - **特征数量**: 不同特征维度的性能影响
  - **模型复杂度**: TabPFN集成大小对性能的影响
  - **域适应复杂度**: 不同域适应方法的计算开销

- [ ] **TASK-024-3**: 稳定性测试
  - **长时间运行**: 连续运行的稳定性验证
  - **内存泄漏**: 长期使用的内存泄漏检测
  - **异常恢复**: 异常情况下的系统恢复
  - **边界条件**: 极端参数下的系统行为

#### 验收标准
- [ ] 性能基准符合预期要求
- [ ] 可扩展性测试覆盖实际使用场景
- [ ] 稳定性测试确保系统可靠性

#### 技术要求
```python
# 伪代码示例
class TestPerformance:
    """性能测试"""

    @pytest.mark.performance
    def test_training_time_benchmark(self):
        """测试训练时间基准"""
        data_sizes = [100, 500, 1000, 2000]
        models = ['PANDA_TabPFN', 'LASSO_LR', 'XGBoost']
        training_times = {}

        for size in data_sizes:
            X, y = make_classification(n_samples=size, n_features=14, random_state=42)
            training_times[size] = {}

            for model_name in models:
                model = create_model(model_name)

                start_time = time.time()
                model.fit(X, y)
                end_time = time.time()

                training_time = end_time - start_time
                training_times[size][model_name] = training_time

                # 验证训练时间合理性
                if model_name == 'PANDA_TabPFN':
                    assert training_time < 300, f"PANDA_TabPFN训练时间应小于5分钟 (实际: {training_time:.2f}s)"
                elif model_name in ['LASSO_LR', 'XGBoost']:
                    assert training_time < 60, f"传统模型训练时间应小于1分钟 (实际: {training_time:.2f}s)"

        # 打印性能基准
        print("训练时间基准 (秒):")
        for size, times in training_times.items():
            print(f"样本数 {size}: {times}")

    @pytest.mark.performance
    def test_inference_time_benchmark(self):
        """测试推理时间基准"""
        X_test = np.random.randn(1000, 14)
        models = ['PANDA_TabPFN', 'LASSO_LR', 'XGBoost']
        inference_times = {}

        for model_name in models:
            # 预训练模型
            X_train, y_train = make_classification(n_samples=1000, n_features=14, random_state=42)
            model = create_model(model_name)
            model.fit(X_train, y_train)

            # 测试推理时间
            start_time = time.time()
            for _ in range(100):  # 重复100次
                _ = model.predict(X_test)
            end_time = time.time()

            avg_inference_time = (end_time - start_time) / 100
            inference_times[model_name] = avg_inference_time

            # 验证推理时间
            assert avg_inference_time < 1.0, f"{model_name}推理时间应小于1秒 (实际: {avg_inference_time:.4f}s)"

        print("推理时间基准 (秒):", inference_times)

    @pytest.mark.performance
    def test_memory_usage_scaling(self):
        """测试内存使用扩展性"""
        import psutil
        process = psutil.Process()

        data_sizes = [1000, 2000, 5000, 10000]
        memory_usage = {}

        for size in data_sizes:
            # 记录初始内存
            initial_memory = process.memory_info().rss

            # 创建和处理数据
            X, y = make_classification(n_samples=size, n_features=14, random_state=42)
            model = PANDATabPFN()
            model.fit(X, y)

            # 记录峰值内存
            peak_memory = process.memory_info().rss
            memory_increase = (peak_memory - initial_memory) / 1024**2  # MB
            memory_usage[size] = memory_increase

            # 清理内存
            del model, X, y
            gc.collect()

        # 验证内存使用线性增长
        sizes = np.array(data_sizes)
        memories = np.array([memory_usage[size] for size in data_sizes])
        correlation = np.corrcoef(sizes, memories)[0, 1]
        assert correlation > 0.8, "内存使用应与数据规模正相关"

        print("内存使用情况 (MB):", memory_usage)

    @pytest.mark.stability
    def test_long_running_stability(self):
        """测试长时间运行稳定性"""
        errors = []
        n_iterations = 50

        for i in range(n_iterations):
            try:
                # 随机数据
                X, y = make_classification(n_samples=200, n_features=14, random_state=i)

                # 训练和预测
                model = PANDATabPFN()
                model.fit(X, y)
                pred = model.predict(X)
                prob = model.predict_proba(X)

                # 验证结果
                assert len(pred) == len(y), "预测长度应匹配"
                assert prob.shape == (len(y), 2), "概率输出形状应正确"
                assert np.all(np.isclose(prob.sum(axis=1), 1.0)), "概率和应为1"

            except Exception as e:
                errors.append(f"Iteration {i}: {str(e)}")

        # 验证错误率
        error_rate = len(errors) / n_iterations
        assert error_rate < 0.1, f"长时间运行错误率应小于10% (实际: {error_rate:.2%})"

        if errors:
            print("发现的错误:", errors[:5])  # 只显示前5个错误

class TestScalability:
    """可扩展性测试"""

    @pytest.mark.scalability
    def test_data_size_scaling(self):
        """测试数据规模扩展性"""
        data_sizes = [500, 1000, 2000, 5000]
        performance_metrics = []

        for size in data_sizes:
            X, y = make_classification(n_samples=size, n_features=14, random_state=42)

            # 训练时间和性能
            start_time = time.time()
            model = PANDATabPFN()
            model.fit(X, y)
            training_time = time.time() - start_time

            # 评估性能
            scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            mean_auc = scores.mean()

            performance_metrics.append({
                'size': size,
                'training_time': training_time,
                'auc': mean_auc,
                'time_per_sample': training_time / size
            })

        # 验证性能扩展性
        for i in range(1, len(performance_metrics)):
            prev_perf = performance_metrics[i-1]
            curr_perf = performance_metrics[i]

            # 训练时间不应指数增长
            time_ratio = curr_perf['training_time'] / prev_perf['training_time']
            size_ratio = curr_perf['size'] / prev_perf['size']
            assert time_ratio < size_ratio ** 1.5, "训练时间增长应慢于数据规模的1.5次方"

        print("扩展性测试结果:")
        for metric in performance_metrics:
            print(f"样本数: {metric['size']}, 训练时间: {metric['training_time']:.2f}s, "
                  f"AUC: {metric['auc']:.3f}, 时间/样本: {metric['time_per_sample']:.4f}s")
```

---

## 🔧 实现细节

### 测试配置
```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: 单元测试
    integration: 集成测试
    system: 系统测试
    performance: 性能测试
    scalability: 可扩展性测试
    stability: 稳定性测试
    slow: 慢速测试（可能需要几分钟）
addopts =
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=panda_heart
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

# conftest.py
import pytest
import numpy as np
from sklearn.datasets import make_classification

@pytest.fixture(scope="session")
def random_seed():
    """全局随机种子"""
    np.random.seed(42)
    return 42

@pytest.fixture
def sample_heart_dataset():
    """示例心脏病数据集"""
    X, y = make_classification(
        n_samples=1000,
        n_features=14,
        n_informative=10,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],  # 模拟类别不平衡
        flip_y=0.01,
        random_state=42
    )
    return X, y

@pytest.fixture
def mock_tabpfn_model():
    """模拟TabPFN模型"""
    class MockTabPFN:
        def __init__(self):
            self.is_fitted = False

        def fit(self, X, y):
            self.is_fitted = True
            return self

        def predict(self, X):
            if not self.is_fitted:
                raise ValueError("Model not fitted")
            return np.random.randint(0, 2, len(X))

        def predict_proba(self, X):
            if not self.is_fitted:
                raise ValueError("Model not fitted")
            prob = np.random.rand(len(X), 2)
            prob = prob / prob.sum(axis=1, keepdims=True)
            return prob

    return MockTabPFN()
```

---

## 🧪 测试执行计划

### 自动化测试流程
```python
# scripts/run_tests.py
#!/usr/bin/env python3
"""自动化测试执行脚本"""

import subprocess
import sys
import argparse

def run_test_suite(test_type="all"):
    """运行测试套件"""

    test_commands = {
        "unit": "pytest tests/ -m unit",
        "integration": "pytest tests/ -m integration",
        "system": "pytest tests/ -m system",
        "performance": "pytest tests/ -m performance",
        "all": "pytest tests/",
        "coverage": "pytest tests/ --cov=panda_heart --cov-report=html",
        "quick": "pytest tests/ -m 'not slow and not performance'"
    }

    if test_type not in test_commands:
        print(f"Unknown test type: {test_type}")
        print(f"Available types: {list(test_commands.keys())}")
        return False

    command = test_commands[test_type]
    print(f"Running: {command}")

    result = subprocess.run(command, shell=True)
    return result.returncode == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PANDA-Heart test suite")
    parser.add_argument("--type", default="all", choices=["unit", "integration", "system", "performance", "all", "coverage", "quick"])
    args = parser.parse_args()

    success = run_test_suite(args.type)
    sys.exit(0 if success else 1)
```

---

## 📊 预期输出

### 测试报告
- `tests/reports/unit_test_report.html` - 单元测试报告
- `tests/reports/integration_test_report.html` - 集成测试报告
- `tests/reports/performance_benchmark.json` - 性能基准报告
- `tests/reports/coverage_report/` - 代码覆盖率报告

### 测试数据
- `tests/data/sample_heart_data.csv` - 测试用心脏病数据
- `tests/data/mock_models/` - 模拟模型文件
- `tests/fixtures/` - 测试夹具和工具函数

---

## 🚨 风险与缓解

### 风险识别
1. **测试覆盖不全** (关键功能遗漏)
2. **测试环境差异** (开发/生产环境不一致)
3. **性能测试不稳定** (环境因素影响)

### 缓解策略
1. **代码覆盖率监控 + 关键路径测试**
2. **容器化测试环境 + 环境标准化**
3. **多次运行 + 统计分析**

---

## 📞 联系信息
**负责人**: [待分配]
**测试工程师**: [QA工程师]
**性能工程师**: [性能优化专家]

*最后更新: 2025-11-18*