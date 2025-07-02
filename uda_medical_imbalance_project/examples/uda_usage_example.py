#!/usr/bin/env python3
"""
UDA处理器使用示例

展示如何使用preprocessing/uda_processor.py进行统一的域适应处理
包括方法选择、参数配置、性能评估和结果比较。

运行示例:
python examples/uda_usage_example.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from preprocessing.uda_processor import (
    UDAProcessor, UDAConfig, 
    create_uda_processor, get_uda_recommendations
)
from data.loader import MedicalDataLoader

# 导入TabPFN（如果可用）
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: TabPFN不可用，使用LogisticRegression")


def load_example_data():
    """加载示例数据（A和B数据集）"""
    try:
        # 使用真实医疗数据
        loader = MedicalDataLoader()
        
        # 加载数据集A（源域）和B（目标域）
        data_A = loader.load_dataset('A', feature_type='best8')
        data_B = loader.load_dataset('B', feature_type='best8')
        
        # 转换为DataFrame格式
        X_source = pd.DataFrame(data_A['X'], columns=data_A['feature_names'])
        y_source = pd.Series(data_A['y'], name='label')
        X_target = pd.DataFrame(data_B['X'], columns=data_B['feature_names'])
        y_target = pd.Series(data_B['y'], name='label')
        
        print(f"✓ 加载真实数据: 源域{X_source.shape}, 目标域{X_target.shape}")
        return X_source, y_source, X_target, y_target
        
    except Exception as e:
        print(f"加载真实数据失败: {e}")
        print("使用模拟数据...")
        
        # 创建模拟数据
        np.random.seed(42)
        n_samples_source, n_samples_target = 200, 150
        n_features = 8
        
        # 源域数据
        X_source = pd.DataFrame(
            np.random.normal(0, 1, (n_samples_source, n_features)),
            columns=[f'Feature{i}' for i in range(n_features)]
        )
        y_source = pd.Series(np.random.choice([0, 1], n_samples_source, p=[0.6, 0.4]))
        
        # 目标域数据（添加域偏移）
        X_target = pd.DataFrame(
            np.random.normal(0.5, 1.2, (n_samples_target, n_features)),
            columns=[f'Feature{i}' for i in range(n_features)]
        )
        y_target = pd.Series(np.random.choice([0, 1], n_samples_target, p=[0.4, 0.6]))
        
        print(f"✓ 创建模拟数据: 源域{X_source.shape}, 目标域{X_target.shape}")
        return X_source, y_source, X_target, y_target


def example_1_basic_usage():
    """示例1: 基本使用方法"""
    print("\n" + "="*60)
    print("示例1: UDA处理器基本使用")
    print("="*60)
    
    # 加载数据
    X_source, y_source, X_target, y_target = load_example_data()
    
    # 创建基础估计器
    if TABPFN_AVAILABLE:
        estimator = TabPFNClassifier()
    else:
        estimator = LogisticRegression(penalty=None, random_state=42, max_iter=1000)
    
    # 方法1: 使用便捷函数创建处理器
    processor = create_uda_processor(
        method_name='SA',  # 使用SA方法（测试中表现最佳）
        base_estimator=estimator,
        save_results=True,
        output_dir="results/uda_examples"
    )
    
    print(f"使用方法: {processor.config.method_name}")
    print(f"可用方法数量: {len(processor.get_available_methods())}")
    
    # 拟合和评估
    uda_method, results = processor.fit_transform(
        X_source, y_source, X_target, y_target
    )
    
    print(f"\n性能结果:")
    for metric, value in results.items():
        if isinstance(value, float) and metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
            print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"✓ 基本使用示例完成")


def example_2_method_comparison():
    """示例2: 多方法对比"""
    print("\n" + "="*60)
    print("示例2: 多种UDA方法对比")
    print("="*60)
    
    # 加载数据
    X_source, y_source, X_target, y_target = load_example_data()
    
    # 创建处理器
    processor = create_uda_processor(
        base_estimator=LogisticRegression(penalty=None, random_state=42, max_iter=1000),
        save_results=True,
        output_dir="results/uda_examples"
    )
    
    # 比较多种方法
    methods_to_compare = ['SA', 'TCA', 'CORAL', 'NNW', 'KMM']
    
    print(f"比较方法: {methods_to_compare}")
    comparison_results = processor.compare_methods(
        X_source, y_source, X_target, y_target,
        methods=methods_to_compare,
        max_methods=5
    )
    
    # 显示比较结果
    print(f"\n=== 方法对比结果 ===")
    print(f"{'方法':<8} {'AUC':<8} {'Accuracy':<10} {'F1':<8}")
    print("-" * 40)
    
    for method_name, results in comparison_results.items():
        if method_name != 'comparison_summary' and 'error' not in results:
            auc = results.get('auc', 0)
            acc = results.get('accuracy', 0)
            f1 = results.get('f1', 0)
            print(f"{method_name:<8} {auc:<8.4f} {acc:<10.4f} {f1:<8.4f}")
    
    # 显示最佳方法
    if 'comparison_summary' in comparison_results:
        summary = comparison_results['comparison_summary']
        print(f"\n✓ 最佳方法: {summary['best_method']} (AUC: {summary['best_auc']:.4f})")
    
    print(f"✓ 方法对比示例完成")


def example_3_custom_configuration():
    """示例3: 自定义配置"""
    print("\n" + "="*60)
    print("示例3: 自定义UDA配置")
    print("="*60)
    
    # 加载数据
    X_source, y_source, X_target, y_target = load_example_data()
    
    # 创建自定义配置
    custom_config = UDAConfig(
        method_name='TCA',  # 使用TCA方法
        method_params={
            'n_components': 5,  # 指定主成分数量
            'mu': 0.5,          # 调整正则化参数
            'kernel': 'linear', # 使用线性核
            'verbose': 1,       # 开启详细输出
            'random_state': 42
        },
        evaluation_metrics=['accuracy', 'auc', 'f1'],  # 指定评估指标
        save_results=True,
        output_dir="results/uda_examples/custom"
    )
    
    # 创建处理器
    processor = UDAProcessor(custom_config)
    
    print(f"自定义配置:")
    print(f"  方法: {custom_config.method_name}")
    print(f"  参数: {custom_config.method_params}")
    print(f"  评估指标: {custom_config.evaluation_metrics}")
    
    # 拟合和评估
    uda_method, results = processor.fit_transform(
        X_source, y_source, X_target, y_target
    )
    
    print(f"\n自定义配置结果:")
    for metric in custom_config.evaluation_metrics:
        if metric in results:
            print(f"  {metric.upper()}: {results[metric]:.4f}")
    
    print(f"✓ 自定义配置示例完成")


def example_4_method_recommendation():
    """示例4: 方法推荐"""
    print("\n" + "="*60)
    print("示例4: UDA方法推荐系统")
    print("="*60)
    
    # 加载数据
    X_source, y_source, X_target, y_target = load_example_data()
    
    # 创建处理器
    processor = create_uda_processor()
    
    # 获取推荐方法
    print("基于数据特征的方法推荐:")
    
    # 场景1: 需要高精度
    recommendation_1 = processor.get_method_recommendation(
        X_source, X_target, 
        requirements={'accuracy': 'high'}
    )
    print(f"  高精度需求: {recommendation_1}")
    
    # 场景2: 需要快速执行
    recommendation_2 = processor.get_method_recommendation(
        X_source, X_target,
        requirements={'speed': 'fast'}
    )
    print(f"  快速执行需求: {recommendation_2}")
    
    # 获取通用推荐
    print(f"\n医疗数据推荐方法:")
    recommendations = get_uda_recommendations()
    for key, value in recommendations['medical_data_recommendations'].items():
        print(f"  {key}: {value}")
    
    print(f"✓ 方法推荐示例完成")


def example_5_integration_workflow():
    """示例5: 完整的集成工作流"""
    print("\n" + "="*60)
    print("示例5: 完整的UDA集成工作流")
    print("="*60)
    
    # 加载数据
    X_source, y_source, X_target, y_target = load_example_data()
    
    # 步骤1: 获取推荐方法
    processor = create_uda_processor()
    recommended_method = processor.get_method_recommendation(X_source, X_target)
    print(f"步骤1 - 推荐方法: {recommended_method}")
    
    # 步骤2: 使用推荐方法进行域适应
    processor_recommended = create_uda_processor(
        method_name=recommended_method,
        base_estimator=LogisticRegression(penalty=None, random_state=42, max_iter=1000),
        save_results=True,
        output_dir="results/uda_examples/workflow"
    )
    
    uda_method, results = processor_recommended.fit_transform(
        X_source, y_source, X_target, y_target
    )
    
    print(f"步骤2 - 推荐方法性能:")
    print(f"  AUC: {results.get('auc', 0):.4f}")
    print(f"  Accuracy: {results.get('accuracy', 0):.4f}")
    
    # 步骤3: 与基线方法对比
    print(f"\n步骤3 - 与基线对比:")
    baseline_methods = ['SA', 'CORAL']  # 快速对比
    
    comparison_results = processor.compare_methods(
        X_source, y_source, X_target, y_target,
        methods=baseline_methods + [recommended_method],
        max_methods=3
    )
    
    # 显示对比
    print(f"方法对比:")
    for method_name, method_results in comparison_results.items():
        if method_name != 'comparison_summary' and 'error' not in method_results:
            auc = method_results.get('auc', 0)
            print(f"  {method_name}: AUC={auc:.4f}")
    
    # 步骤4: 获取结果摘要
    summary = processor_recommended.get_results_summary()
    print(f"\n步骤4 - 结果摘要:")
    print(f"  实验次数: {summary.get('total_experiments', 0)}")
    if 'best_result' in summary and summary['best_result']:
        best = summary['best_result']
        print(f"  最佳结果: {best['method_name']} (AUC: {best.get('auc', 0):.4f})")
    
    print(f"✓ 完整工作流示例完成")


def main():
    """主函数 - 运行所有示例"""
    print("UDA处理器使用示例")
    print("=" * 60)
    
    try:
        # 检查环境
        from preprocessing.uda_processor import UDAProcessor
        from uda.adapt_methods import is_adapt_available
        
        if not is_adapt_available():
            print("❌ Adapt库不可用，请安装: pip install adapt-python")
            return
        
        print("✅ 环境检查通过")
        
        # 运行示例
        example_1_basic_usage()
        example_2_method_comparison() 
        example_3_custom_configuration()
        example_4_method_recommendation()
        example_5_integration_workflow()
        
        print("\n" + "="*60)
        print("✅ 所有示例运行完成！")
        print("结果已保存到 results/uda_examples/ 目录")
        print("="*60)
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请检查项目环境和依赖安装")
    except Exception as e:
        print(f"❌ 运行错误: {e}")


if __name__ == "__main__":
    main() 