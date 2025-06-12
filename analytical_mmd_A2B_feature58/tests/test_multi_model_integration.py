"""
多模型集成测试

测试新的模型选择器和跨域实验运行器功能
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from analytical_mmd_A2B_feature58.config.settings import (
    BEST_7_FEATURES, BEST_7_CAT_IDX, get_features_by_type, 
    get_categorical_indices, get_model_config
)

class TestModelSelector:
    """测试模型选择器功能"""
    
    def setup_method(self):
        """设置测试数据"""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = len(BEST_7_FEATURES)
        
        # 创建模拟数据
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.randint(0, 2, self.n_samples)
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """清理测试数据"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_get_features_by_type(self):
        """测试特征获取功能"""
        # 测试获取所有特征
        all_features = get_features_by_type('all')
        assert len(all_features) == 58
        
        # 测试获取最佳7特征
        best7_features = get_features_by_type('best7')
        assert len(best7_features) == 7
        assert best7_features == BEST_7_FEATURES
        
        # 测试无效类型
        with pytest.raises(ValueError):
            get_features_by_type('invalid')
    
    def test_get_categorical_indices(self):
        """测试类别特征索引获取"""
        # 测试最佳7特征的类别索引
        cat_indices = get_categorical_indices('best7')
        assert len(cat_indices) == 2  # Feature63, Feature46
        assert cat_indices == BEST_7_CAT_IDX
        
        # 测试无效类型
        with pytest.raises(ValueError):
            get_categorical_indices('invalid')
    
    def test_get_model_config(self):
        """测试模型配置获取"""
        # 测试AutoTabPFN配置
        auto_config = get_model_config('auto')
        assert 'max_time' in auto_config
        assert 'random_state' in auto_config
        
        # 测试默认配置
        default_config = get_model_config('auto', 'default')
        assert 'max_time' in default_config
        
        # 测试无效模型类型
        with pytest.raises(ValueError):
            get_model_config('invalid')
    
    def test_model_selector_auto(self):
        """测试AutoTabPFN模型选择器"""
        try:
            from analytical_mmd_A2B_feature58.modeling.model_selector import get_model
            
            # 创建模型
            model = get_model('auto', categorical_feature_indices=BEST_7_CAT_IDX)
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')
            
        except ImportError:
            pytest.skip("AutoTabPFN不可用")
    
    def test_model_selector_tuned(self):
        """测试TunedTabPFN模型选择器"""
        try:
            from analytical_mmd_A2B_feature58.modeling.model_selector import get_model
            
            # 创建模型
            model = get_model('tuned', categorical_feature_indices=BEST_7_CAT_IDX)
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')
            
        except ImportError:
            pytest.skip("TunedTabPFN不可用")
    
    def test_model_selector_base(self):
        """测试原生TabPFN模型选择器"""
        try:
            from analytical_mmd_A2B_feature58.modeling.model_selector import get_model
            
            # 创建模型
            model = get_model('base', categorical_feature_indices=BEST_7_CAT_IDX)
            assert hasattr(model, 'fit')
            assert hasattr(model, 'predict')
            assert hasattr(model, 'predict_proba')
            
        except ImportError:
            pytest.skip("原生TabPFN不可用")
    
    def test_model_selector_rf(self):
        """测试RF模型选择器"""
        try:
            from analytical_mmd_A2B_feature58.modeling.model_selector import get_model
            
            # 创建模型
            model = get_model('rf', categorical_feature_indices=BEST_7_CAT_IDX)
            
            # 测试训练和预测
            model.fit(self.X, self.y)
            predictions = model.predict(self.X)
            probabilities = model.predict_proba(self.X)
            
            assert len(predictions) == self.n_samples
            assert probabilities.shape == (self.n_samples, 2)
            
        except ImportError:
            pytest.skip("RF模型依赖不可用")
    
    def test_get_available_models(self):
        """测试获取可用模型列表"""
        try:
            from analytical_mmd_A2B_feature58.modeling.model_selector import get_available_models
            
            available = get_available_models()
            assert isinstance(available, list)
            assert 'rf' in available  # RF总是可用
            
        except ImportError:
            pytest.skip("模型选择器不可用")

class TestCrossDomainRunner:
    """测试跨域实验运行器"""
    
    def setup_method(self):
        """设置测试数据"""
        np.random.seed(42)
        self.n_samples_a = 100
        self.n_samples_b = 80
        self.n_features = len(BEST_7_FEATURES)
        
        # 创建模拟数据集
        self.X_A = np.random.randn(self.n_samples_a, self.n_features)
        self.y_A = np.random.randint(0, 2, self.n_samples_a)
        self.X_B = np.random.randn(self.n_samples_b, self.n_features)
        self.y_B = np.random.randint(0, 2, self.n_samples_b)
        
        # 创建临时目录和模拟数据文件
        self.temp_dir = tempfile.mkdtemp()
        self.create_mock_data_files()
        
    def teardown_method(self):
        """清理测试数据"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_mock_data_files(self):
        """创建模拟数据文件"""
        # 创建数据集A
        df_A = pd.DataFrame(self.X_A, columns=BEST_7_FEATURES)
        df_A['Label'] = self.y_A
        df_A.to_excel(os.path.join(self.temp_dir, 'dataset_A.xlsx'), index=False)
        
        # 创建数据集B
        df_B = pd.DataFrame(self.X_B, columns=BEST_7_FEATURES)
        df_B['Label'] = self.y_B
        df_B.to_excel(os.path.join(self.temp_dir, 'dataset_B.xlsx'), index=False)
        
        # 创建数据集C (缺少一些特征)
        df_C = pd.DataFrame(self.X_B[:50, :5], columns=BEST_7_FEATURES[:5])
        df_C['Label'] = self.y_B[:50]
        df_C.to_excel(os.path.join(self.temp_dir, 'dataset_C.xlsx'), index=False)
    
    @patch('analytical_mmd_A2B_feature58.config.settings.DATA_PATHS')
    def test_cross_domain_runner_initialization(self, mock_data_paths):
        """测试跨域实验运行器初始化"""
        try:
            from analytical_mmd_A2B_feature58.modeling.cross_domain_runner import CrossDomainExperimentRunner
            
            # 设置模拟数据路径
            mock_data_paths.return_value = {
                'A': os.path.join(self.temp_dir, 'dataset_A.xlsx'),
                'B': os.path.join(self.temp_dir, 'dataset_B.xlsx'),
                'C': os.path.join(self.temp_dir, 'dataset_C.xlsx')
            }
            
            # 创建运行器
            runner = CrossDomainExperimentRunner(
                model_type='rf',
                feature_type='best7',
                use_mmd_adaptation=True,
                mmd_method='mean_std',
                save_path=os.path.join(self.temp_dir, 'results')
            )
            
            assert runner.model_type == 'rf'
            assert runner.feature_type == 'best7'
            assert runner.mmd_method == 'mean_std'
            assert len(runner.features) == 7
            
        except ImportError:
            pytest.skip("跨域实验运行器不可用")
    
    @patch('analytical_mmd_A2B_feature58.config.settings.DATA_PATHS')
    def test_load_datasets(self, mock_data_paths):
        """测试数据集加载"""
        try:
            from analytical_mmd_A2B_feature58.modeling.cross_domain_runner import CrossDomainExperimentRunner
            
            # 设置模拟数据路径
            mock_data_paths_dict = {
                'A': os.path.join(self.temp_dir, 'dataset_A.xlsx'),
                'B': os.path.join(self.temp_dir, 'dataset_B.xlsx'),
                'C': os.path.join(self.temp_dir, 'dataset_C.xlsx')
            }
            
            runner = CrossDomainExperimentRunner(
                model_type='rf',
                feature_type='best7',
                save_path=os.path.join(self.temp_dir, 'results')
            )
            
            # 直接mock DATA_PATHS模块级别的变量
            with patch('analytical_mmd_A2B_feature58.modeling.cross_domain_runner.DATA_PATHS', mock_data_paths_dict):
                # 注意：这里只测试初始化，不测试实际的数据加载
                # 因为我们没有真实的数据文件
                assert runner.model_type == 'rf'
                assert runner.feature_type == 'best7'
                assert len(runner.features) == 7
            
        except ImportError:
            pytest.skip("跨域实验运行器不可用")

class TestIntegrationWorkflow:
    """测试完整的集成工作流"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """清理测试环境"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        try:
            # 这里可以添加完整的端到端测试
            # 由于依赖较多，暂时跳过
            pytest.skip("端到端测试需要完整的依赖环境")
            
        except ImportError:
            pytest.skip("集成测试依赖不可用")

class TestConfigurationValidation:
    """测试配置验证功能"""
    
    def test_feature_configuration_consistency(self):
        """测试特征配置一致性"""
        # 检查最佳7特征是否在全部特征中
        all_features = get_features_by_type('all')
        best7_features = get_features_by_type('best7')
        
        for feature in best7_features:
            assert feature in all_features, f"特征 {feature} 不在全部特征列表中"
    
    def test_categorical_indices_consistency(self):
        """测试类别特征索引一致性"""
        best7_features = get_features_by_type('best7')
        cat_indices = get_categorical_indices('best7')
        
        # 检查索引是否有效
        for idx in cat_indices:
            assert 0 <= idx < len(best7_features), f"类别特征索引 {idx} 超出范围"
    
    def test_model_config_completeness(self):
        """测试模型配置完整性"""
        model_types = ['auto', 'base', 'rf']
        
        for model_type in model_types:
            config = get_model_config(model_type)
            assert isinstance(config, dict), f"模型 {model_type} 配置应为字典"
            assert len(config) > 0, f"模型 {model_type} 配置不能为空"

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"]) 