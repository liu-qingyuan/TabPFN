"""
UDA Medical Imbalance Project - 优化后系统使用示例

展示如何使用优化后的系统进行医疗数据处理和分析。
包含配置管理、异常处理、性能监控和数据验证的完整示例。

作者: UDA Medical Team
日期: 2024-01-30
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 导入优化后的模块
from config import get_config_manager, get_project_config
from utils.exceptions import UDAMedicalError, ExceptionContext
from utils.validators import DataValidator
from utils.performance import PerformanceMonitor, monitor_performance
from utils.helpers import ensure_dataframe, save_json, format_duration
from preprocessing.imbalance_handler import ImbalanceHandler

def main():
    """主函数 - 展示优化后系统的完整使用流程"""
    
    print("🏥 UDA Medical Imbalance Project - 优化后系统演示")
    print("=" * 60)
    
    # 1. 配置管理演示
    print("\n📋 1. 配置管理系统")
    print("-" * 30)
    
    # 获取全局配置管理器
    config_manager = get_config_manager()
    project_config = get_project_config()
    
    print(f"项目名称: {project_config.project_name}")
    print(f"版本: {project_config.version}")
    print(f"数据目录: {project_config.data_dir}")
    print(f"结果目录: {project_config.results_dir}")
    
    # 获取模型配置
    tabpfn_config = config_manager.get_model_config("tabpfn")
    print(f"TabPFN配置: {tabpfn_config}")
    
    # 2. 性能监控演示
    print("\n⚡ 2. 性能监控系统")
    print("-" * 30)
    
    with PerformanceMonitor("data_generation", track_memory=True, track_cpu=True) as monitor:
        # 生成模拟医疗数据
        n_samples, n_features = 1000, 20
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        # 添加自定义性能指标
        monitor.add_custom_metric("samples_generated", n_samples)
        monitor.add_custom_metric("features_generated", n_features)
    
    # 3. 数据验证演示
    print("\n🔍 3. 数据验证系统")
    print("-" * 30)
    
    try:
        # 转换为DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        X_df = ensure_dataframe(X, columns=feature_names)
        
        # 数据验证
        print("正在验证特征数据...")
        DataValidator.validate_features(X_df, min_features=5, max_features=50)
        print("✅ 特征验证通过")
        
        print("正在验证标签数据...")
        DataValidator.validate_labels(y, expected_classes=[0, 1], min_samples_per_class=50)
        print("✅ 标签验证通过")
        
        print("正在验证数据一致性...")
        DataValidator.validate_data_consistency(X_df, y)
        print("✅ 一致性验证通过")
        
    except UDAMedicalError as e:
        print(f"❌ 数据验证失败: {e}")
        return
    
    # 4. 异常处理演示
    print("\n🛡️ 4. 异常处理系统")
    print("-" * 30)
    
    # 使用异常上下文管理器
    with ExceptionContext("data_processing_demo", suppress_exceptions=False):
        try:
            # 模拟一个可能失败的操作
            result = demonstrate_safe_operation(X_df, y)
            print(f"✅ 安全操作完成: {result}")
        except UDAMedicalError as e:
            print(f"❌ 捕获到预期异常: {e.error_code}")
            print(f"   错误信息: {e.message}")
    
    # 5. 不平衡数据处理演示
    print("\n⚖️ 5. 优化后的不平衡处理")
    print("-" * 30)
    
    # 创建不平衡数据
    imbalanced_y = create_imbalanced_data(y, ratio=0.1)
    print(f"原始分布: {np.bincount(imbalanced_y)}")
    
    # 使用优化后的不平衡处理器
    with monitor_performance("imbalance_processing", track_memory=True):
        try:
            # 创建处理器
            imbalance_handler = ImbalanceHandler(
                method='smote',
                random_state=42,
                k_neighbors=5
            )
            
            # 拟合和变换（带自动性能监控和异常处理）
            X_resampled, y_resampled = imbalance_handler.fit_transform(X_df, imbalanced_y)
            
            print(f"重采样后分布: {np.bincount(y_resampled)}")
            print(f"数据形状变化: {X_df.shape} -> {X_resampled.shape}")
            
            # 获取采样信息
            sampling_info = imbalance_handler.get_sampling_info()
            print(f"采样方法: {sampling_info['method']}")
            
        except UDAMedicalError as e:
            print(f"❌ 不平衡处理失败: {e}")
            return
    
    # 6. 结果保存演示
    print("\n💾 6. 结果保存系统")  
    print("-" * 30)
    
    # 准备结果数据
    results = {
        "experiment_info": {
            "name": "optimized_system_demo",
            "timestamp": pd.Timestamp.now().isoformat(),
            "version": "2.0.0"
        },
        "data_info": {
            "original_samples": len(y),
            "resampled_samples": len(y_resampled),
            "features": n_features,
            "original_distribution": dict(zip(*np.unique(imbalanced_y, return_counts=True))),
            "resampled_distribution": dict(zip(*np.unique(y_resampled, return_counts=True)))
        },
        "performance_metrics": {
            "memory_usage_mb": monitor.metrics.memory_used if hasattr(monitor, 'metrics') else None,
            "processing_time_seconds": monitor.metrics.duration if hasattr(monitor, 'metrics') else None
        }
    }
    
    # 保存结果
    output_file = project_config.results_dir / "optimized_demo_results.json"
    save_json(results, output_file)
    print(f"✅ 结果已保存到: {output_file}")
    
    # 7. 系统总结
    print("\n🎉 7. 优化效果总结")
    print("-" * 30)
    
    print("✅ 配置管理: 统一、集中、可序列化")
    print("✅ 异常处理: 结构化、可追踪、自动恢复") 
    print("✅ 数据验证: 全面、智能、预防性")
    print("✅ 性能监控: 自动、详细、可分析")
    print("✅ 模块优化: 类型安全、错误容错、高性能")
    
    print("\n🚀 系统优化完成！")
    print("现在您可以更安全、高效地进行医疗数据分析。")


def demonstrate_safe_operation(X: pd.DataFrame, y: np.ndarray) -> str:
    """演示安全操作 - 带有完整的验证和错误处理"""
    
    # 使用数据验证
    DataValidator.validate_features(X, min_features=1)
    DataValidator.validate_labels(y)
    
    # 模拟一些数据处理
    processed_samples = len(X)
    processed_features = X.shape[1]
    
    return f"处理了 {processed_samples} 样本和 {processed_features} 特征"


def create_imbalanced_data(y: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    """创建不平衡数据集用于演示"""
    
    # 获取正类和负类索引
    positive_indices = np.where(y == 1)[0]
    negative_indices = np.where(y == 0)[0]
    
    # 计算需要保留的正类样本数
    n_positive_keep = int(len(negative_indices) * ratio)
    n_positive_keep = min(n_positive_keep, len(positive_indices))
    
    # 随机选择正类样本
    np.random.seed(42)
    selected_positive = np.random.choice(positive_indices, n_positive_keep, replace=False)
    
    # 合并索引并创建新的标签数组
    all_indices = np.concatenate([negative_indices, selected_positive])
    imbalanced_y = y[all_indices]
    
    return imbalanced_y


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()