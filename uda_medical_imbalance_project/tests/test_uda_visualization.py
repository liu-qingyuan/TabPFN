#!/usr/bin/env python3
"""
UDA可视化功能测试

测试UDA可视化器的各个功能：
1. 降维可视化 (PCA, t-SNE)
2. 特征分布对比
3. 域距离度量
4. 性能对比

运行测试: python tests/test_uda_visualization.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_visualizer_basic_functionality():
    """测试可视化器基本功能"""
    print("=== 测试UDA可视化器基本功能 ===")
    
    try:
        from preprocessing.uda_visualizer import create_uda_visualizer
        
        # 创建可视化器
        visualizer = create_uda_visualizer(
            figsize=(10, 8),
            save_plots=False,  # 测试时不保存图片
            output_dir="tests/temp_viz"
        )
        
        print("✓ 可视化器创建成功")
        
        # 创建模拟数据
        np.random.seed(42)
        X_source = np.random.normal(0, 1, (100, 6))
        y_source = np.random.choice([0, 1], 100, p=[0.6, 0.4])
        X_target = np.random.normal(0.5, 1.2, (80, 6))
        y_target = np.random.choice([0, 1], 80, p=[0.4, 0.6])
        
        print("✓ 模拟数据创建成功")
        
        # 测试降维可视化
        print("\n1. 测试降维可视化...")
        dim_results = visualizer.plot_dimensionality_reduction(
            X_source, y_source, X_target, y_target,
            uda_method=None, method_name="Test"
        )
        print("✓ 降维可视化测试通过")
        
        # 测试特征分布
        print("\n2. 测试特征分布...")
        dist_results = visualizer.plot_feature_distributions(
            X_source, X_target, uda_method=None, method_name="Test"
        )
        print("✓ 特征分布测试通过")
        
        # 测试域距离度量
        print("\n3. 测试域距离度量...")
        distance_results = visualizer.calculate_domain_distances(
            X_source, X_target, uda_method=None, method_name="Test"
        )
        
        # 检查距离度量结果
        expected_metrics = ['kl_divergence_before', 'wasserstein_before', 'mmd_before']
        for metric in expected_metrics:
            assert metric in distance_results, f"缺少距离度量: {metric}"
            assert isinstance(distance_results[metric], (int, float)), f"{metric} 不是数值类型"
        
        print("✓ 域距离度量测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 可视化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizer_with_uda_method():
    """测试带UDA方法的可视化"""
    print("\n=== 测试带UDA方法的可视化 ===")
    
    try:
        from preprocessing.uda_processor import create_uda_processor
        from preprocessing.uda_visualizer import create_uda_visualizer
        from uda.adapt_methods import is_adapt_available
        from sklearn.linear_model import LogisticRegression
        
        if not is_adapt_available():
            print("⚠ Adapt库不可用，跳过UDA方法测试")
            return True
        
        # 创建数据
        np.random.seed(42)
        X_source = np.random.normal(0, 1, (150, 6))
        y_source = np.random.choice([0, 1], 150, p=[0.6, 0.4])
        X_target = np.random.normal(0.8, 1.3, (100, 6))
        y_target = np.random.choice([0, 1], 100, p=[0.4, 0.6])
        
        # 创建UDA处理器
        processor = create_uda_processor(
            method_name='CORAL',  # 使用CORAL方法，相对稳定
            base_estimator=LogisticRegression(penalty=None, random_state=42, max_iter=1000),
            save_results=False
        )
        
        # 拟合UDA方法
        uda_method, results = processor.fit_transform(
            X_source, y_source, X_target, y_target
        )
        
        print(f"✓ UDA方法拟合成功: {processor.config.method_name}")
        
        # 创建可视化器
        visualizer = create_uda_visualizer(
            figsize=(12, 9),
            save_plots=False,
            output_dir="tests/temp_viz"
        )
        
        # 完整可视化分析
        viz_results = visualizer.visualize_domain_adaptation_complete(
            X_source, y_source, X_target, y_target,
            uda_method=uda_method,
            method_name=processor.config.method_name
        )
        
        # 检查结果
        expected_keys = ['dimensionality_reduction', 'feature_distributions', 'domain_distances']
        for key in expected_keys:
            assert key in viz_results, f"缺少可视化结果: {key}"
        
        print("✓ 带UDA方法的可视化测试通过")
        return True
        
    except Exception as e:
        print(f"❌ UDA方法可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_distance_metrics():
    """测试距离度量计算"""
    print("\n=== 测试距离度量计算 ===")
    
    try:
        from preprocessing.uda_visualizer import UDAVisualizer
        
        visualizer = UDAVisualizer(save_plots=False)
        
        # 创建测试数据
        np.random.seed(42)
        X1 = np.random.normal(0, 1, (100, 4))
        X2 = np.random.normal(1, 1.5, (80, 4))
        
        # 测试KL散度
        kl_div = visualizer._calculate_kl_divergence(X1, X2)
        assert isinstance(kl_div, (int, float)), "KL散度应该是数值"
        assert kl_div >= 0, "KL散度应该非负"
        print(f"✓ KL散度计算: {kl_div:.4f}")
        
        # 测试Wasserstein距离
        ws_dist = visualizer._calculate_wasserstein_distance(X1, X2)
        assert isinstance(ws_dist, (int, float)), "Wasserstein距离应该是数值"
        assert ws_dist >= 0, "Wasserstein距离应该非负"
        print(f"✓ Wasserstein距离计算: {ws_dist:.4f}")
        
        # 测试MMD
        mmd = visualizer._calculate_mmd(X1, X2)
        assert isinstance(mmd, (int, float)), "MMD应该是数值"
        assert mmd >= 0, "MMD应该非负"
        print(f"✓ MMD计算: {mmd:.4f}")
        
        print("✓ 距离度量计算测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 距离度量测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """测试性能对比功能"""
    print("\n=== 测试性能对比功能 ===")
    
    try:
        from preprocessing.uda_visualizer import create_uda_visualizer
        from sklearn.linear_model import LogisticRegression
        
        # 创建模拟UDA方法
        class MockUDAMethod:
            def __init__(self):
                self.model = LogisticRegression(random_state=42, max_iter=1000, penalty=None)
                self.fitted = False
            
            def fit(self, X_source, y_source, X_target):
                self.model.fit(X_source, y_source)
                self.fitted = True
                return self
            
            def predict(self, X):
                if not self.fitted:
                    raise ValueError("Method not fitted")
                return self.model.predict(X)
            
            def predict_proba(self, X):
                if not self.fitted:
                    raise ValueError("Method not fitted")
                return self.model.predict_proba(X)
        
        # 创建数据
        np.random.seed(42)
        X_source = np.random.normal(0, 1, (100, 4))
        y_source = np.random.choice([0, 1], 100, p=[0.6, 0.4])
        X_target = np.random.normal(0.5, 1.2, (80, 4))
        y_target = np.random.choice([0, 1], 80, p=[0.4, 0.6])
        
        # 创建并拟合模拟UDA方法
        mock_uda = MockUDAMethod()
        mock_uda.fit(X_source, y_source, X_target)
        
        # 创建可视化器
        visualizer = create_uda_visualizer(save_plots=False)
        
        # 测试性能对比
        perf_results = visualizer.plot_performance_comparison(
            X_source, y_source, X_target, y_target,
            uda_method=mock_uda,
            method_name="Mock_UDA"
        )
        
        # 检查结果
        assert 'baseline_scores' in perf_results, "缺少基线分数"
        assert 'uda_scores' in perf_results, "缺少UDA分数"
        assert 'improvements' in perf_results, "缺少改进指标"
        
        print("✓ 性能对比功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 性能对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("UDA可视化功能测试")
    print("=" * 50)
    
    tests = [
        test_visualizer_basic_functionality,
        test_distance_metrics,
        test_performance_comparison,
        test_visualizer_with_uda_method,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ 测试 {test_func.__name__} 异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有可视化功能测试通过！")
        return True
    else:
        print("⚠ 部分测试失败，请检查相关功能")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 