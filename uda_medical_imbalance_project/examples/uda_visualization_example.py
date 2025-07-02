#!/usr/bin/env python3
"""
UDA可视化完整示例

展示如何使用UDA处理器和可视化器进行完整的域适应分析：
1. 数据加载和预处理
2. UDA方法应用
3. 结果可视化分析
4. 多方法对比可视化

运行示例:
python examples/uda_visualization_example.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
try:
    from preprocessing.uda_processor import create_uda_processor
    from preprocessing.uda_visualizer import create_uda_visualizer
    from data.loader import MedicalDataLoader
except ImportError as e:
    print(f"导入项目模块失败: {e}")
    print("请确保在正确的项目目录中运行此脚本")
    sys.exit(1)

# 导入TabPFN（如果可用）
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("Warning: TabPFN不可用，使用LogisticRegression")


def load_medical_data():
    """加载医疗数据"""
    try:
        # 使用真实医疗数据
        loader = MedicalDataLoader()
        
        # 加载数据集A（源域）和B（目标域）
        data_A = loader.load_dataset('A', feature_type='best8')
        data_B = loader.load_dataset('B', feature_type='best8')
        
        # 转换为numpy数组
        X_source = data_A['X']
        y_source = data_A['y']
        X_target = data_B['X']
        y_target = data_B['y']
        
        print(f"✓ 加载真实医疗数据: 源域{X_source.shape}, 目标域{X_target.shape}")
        print(f"  特征列表: {data_A['feature_names']}")
        print(f"  源域类别分布: {dict(zip(*np.unique(y_source, return_counts=True)))}")
        print(f"  目标域类别分布: {dict(zip(*np.unique(y_target, return_counts=True)))}")
        
        return X_source, y_source, X_target, y_target
        
    except Exception as e:
        print(f"加载真实数据失败: {e}")
        print("使用模拟数据...")
        
        # 创建模拟数据
        np.random.seed(42)
        n_samples_source, n_samples_target = 300, 200
        n_features = 8
        
        # 源域数据
        X_source = np.random.normal(0, 1, (n_samples_source, n_features))
        y_source = np.random.choice([0, 1], n_samples_source, p=[0.6, 0.4])
        
        # 目标域数据（添加域偏移）
        X_target = np.random.normal(0.8, 1.5, (n_samples_target, n_features))
        y_target = np.random.choice([0, 1], n_samples_target, p=[0.4, 0.6])
        
        print(f"✓ 创建模拟数据: 源域{X_source.shape}, 目标域{X_target.shape}")
        return X_source, y_source, X_target, y_target


def example_1_single_method_visualization():
    """示例1: 单个UDA方法的完整可视化分析"""
    print("\n" + "="*70)
    print("示例1: 单个UDA方法的完整可视化分析")
    print("="*70)
    
    # 加载数据
    X_source, y_source, X_target, y_target = load_medical_data()
    
    # 创建基础估计器
    if TABPFN_AVAILABLE:
        estimator = TabPFNClassifier()
    else:
        estimator = LogisticRegression(penalty=None, random_state=42, max_iter=1000)
    
    # 创建UDA处理器（使用SA方法 - 测试中表现最佳）
    processor = create_uda_processor(
        method_name='SA',
        base_estimator=estimator,
        save_results=True,
        output_dir="results/uda_examples/visualization"
    )
    
    print(f"使用方法: {processor.config.method_name}")
    
    # 拟合UDA方法
    uda_method, results = processor.fit_transform(
        X_source, y_source, X_target, y_target
    )
    
    print(f"UDA方法性能:")
    for metric, value in results.items():
        if isinstance(value, float) and metric in ['accuracy', 'auc', 'f1', 'precision', 'recall']:
            print(f"  {metric.upper()}: {value:.4f}")
    
    # 创建可视化器
    visualizer = create_uda_visualizer(
        figsize=(14, 10),
        save_plots=True,
        output_dir="results/uda_examples/visualization"
    )
    
    # 完整可视化分析
    print(f"\n开始可视化分析...")
    viz_results = visualizer.visualize_domain_adaptation_complete(
        X_source, y_source, X_target, y_target,
        uda_method=uda_method,
        method_name=processor.config.method_name
    )
    
    # 输出距离度量结果
    if 'domain_distances' in viz_results:
        distances = viz_results['domain_distances']
        print(f"\n域距离度量结果:")
        # 优先显示标准化指标
        if 'normalized_kl_divergence_improvement' in distances:
            print(f"  标准化KL散度改进: {distances['normalized_kl_divergence_improvement']:.4f}")
        if 'normalized_wasserstein_improvement' in distances:
            print(f"  标准化Wasserstein距离改进: {distances['normalized_wasserstein_improvement']:.4f}")
        
        # 显示原始备用指标
        print(f"  KL散度改进: {distances.get('kl_divergence_improvement', 0):.4f}")
        print(f"  Wasserstein距离改进: {distances.get('wasserstein_improvement', 0):.4f}")
        print(f"  MMD改进: {distances.get('mmd_improvement', 0):.4f}")
    
    print(f"✓ 单个方法可视化分析完成")
    return viz_results


def example_2_multiple_methods_comparison():
    """示例2: 多种UDA方法的可视化对比"""
    print("\n" + "="*70)
    print("示例2: 多种UDA方法的可视化对比")
    print("="*70)
    
    # 加载数据
    X_source, y_source, X_target, y_target = load_medical_data()
    
    # 要对比的方法
    methods_to_compare = ['SA', 'TCA', 'CORAL']
    
    # 创建可视化器
    visualizer = create_uda_visualizer(
        figsize=(16, 12),
        save_plots=True,
        output_dir="results/uda_examples/comparison"
    )
    
    comparison_results = {}
    
    for method_name in methods_to_compare:
        print(f"\n--- 分析方法: {method_name} ---")
        
        try:
            # 创建UDA处理器
            processor = create_uda_processor(
                method_name=method_name,
                base_estimator=LogisticRegression(penalty=None, random_state=42, max_iter=1000),
                save_results=False  # 对比时不保存单个结果
            )
            
            # 拟合UDA方法
            uda_method, results = processor.fit_transform(
                X_source, y_source, X_target, y_target
            )
            
            print(f"  性能: AUC={results.get('auc', 0):.4f}, Accuracy={results.get('accuracy', 0):.4f}")
            
            # 可视化分析
            viz_results = visualizer.visualize_domain_adaptation_complete(
                X_source, y_source, X_target, y_target,
                uda_method=uda_method,
                method_name=method_name
            )
            
            comparison_results[method_name] = {
                'performance': results,
                'visualization': viz_results
            }
            
        except Exception as e:
            print(f"  方法 {method_name} 分析失败: {e}")
            comparison_results[method_name] = {'error': str(e)}
    
    # 生成综合对比图
    generate_methods_comparison_plot(comparison_results, visualizer)
    
    print(f"✓ 多方法对比可视化完成")
    return comparison_results


def generate_methods_comparison_plot(comparison_results: Dict[str, Any], visualizer) -> None:
    """生成方法对比的综合图表"""
    
    # 提取性能数据
    methods = []
    performance_data = {}
    distance_data = {}
    
    for method_name, results in comparison_results.items():
        if 'error' not in results:
            methods.append(method_name)
            
            # 性能指标
            perf = results['performance']
            for metric in ['accuracy', 'auc', 'f1']:
                if metric not in performance_data:
                    performance_data[metric] = []
                performance_data[metric].append(perf.get(metric, 0))
            
            # 距离指标（优先使用标准化版本）
            if 'visualization' in results and 'domain_distances' in results['visualization']:
                distances = results['visualization']['domain_distances']
                
                # 优先使用标准化指标
                priority_metrics = [
                    'normalized_kl_divergence_improvement',
                    'normalized_wasserstein_improvement',
                    'kl_divergence_improvement',
                    'wasserstein_improvement',
                    'mmd_improvement'
                ]
                
                for metric in priority_metrics:
                    if metric in distances:
                        if metric not in distance_data:
                            distance_data[metric] = []
                        distance_data[metric].append(distances.get(metric, 0))
    
    if not methods:
        print("没有成功的方法可供对比")
        return
    
    # 创建综合对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('UDA Methods Comprehensive Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#E63946', '#2A9D8F']
    
    # 1. 性能指标对比
    x = np.arange(len(methods))
    width = 0.25
    
    for i, (metric, values) in enumerate(performance_data.items()):
        ax1.bar(x + i*width, values, width, label=metric.upper(), 
               color=colors[i % len(colors)], alpha=0.8)
    
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. AUC对比（雷达图风格）
    if 'auc' in performance_data:
        auc_values = performance_data['auc']
        bars = ax2.bar(methods, auc_values, color=colors[:len(methods)], alpha=0.7)
        ax2.set_ylabel('AUC Score')
        ax2.set_title('AUC Performance Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, auc_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 3. 距离改进对比
    if distance_data:
        x_dist = np.arange(len(methods))
        width_dist = 0.25
        
        for i, (metric, values) in enumerate(distance_data.items()):
            metric_name = metric.replace('_improvement', '').replace('_', ' ').title()
            ax3.bar(x_dist + i*width_dist, values, width_dist, 
                   label=metric_name, color=colors[i % len(colors)], alpha=0.8)
        
        ax3.set_xlabel('Methods')
        ax3.set_ylabel('Distance Improvement')
        ax3.set_title('Domain Distance Improvement')
        ax3.set_xticks(x_dist + width_dist)
        ax3.set_xticklabels(methods)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 4. 综合评分
    # 计算综合评分（性能 + 距离改进）
    if performance_data and distance_data:
        composite_scores = []
        for i in range(len(methods)):
            perf_score = np.mean([performance_data[metric][i] for metric in performance_data.keys()])
            # 距离改进归一化到[0,1]
            dist_improvements = [distance_data[metric][i] for metric in distance_data.keys()]
            dist_score = np.mean([max(0, min(1, (imp + 1) / 2)) for imp in dist_improvements])
            composite_score = 0.7 * perf_score + 0.3 * dist_score  # 性能权重更高
            composite_scores.append(composite_score)
        
        bars = ax4.bar(methods, composite_scores, color=colors[:len(methods)], alpha=0.7)
        ax4.set_ylabel('Composite Score')
        ax4.set_title('Overall Performance Score')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # 添加数值标签和排名
        for bar, score in zip(bars, composite_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 找出最佳方法
        best_idx = np.argmax(composite_scores)
        best_method = methods[best_idx]
        ax4.text(0.02, 0.98, f'Best: {best_method}', transform=ax4.transAxes,
                fontsize=12, fontweight='bold', va='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    if visualizer.save_plots:
        save_path = visualizer.output_dir / "methods_comprehensive_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  综合对比图已保存: {save_path}")
    
    plt.show()


def example_3_baseline_comparison():
    """示例3: 基线对比可视化"""
    print("\n" + "="*70)
    print("示例3: 基线对比可视化（无域适应 vs 域适应）")
    print("="*70)
    
    # 加载数据
    X_source, y_source, X_target, y_target = load_medical_data()
    
    # 创建可视化器
    visualizer = create_uda_visualizer(
        figsize=(16, 10),
        save_plots=True,
        output_dir="results/uda_examples/baseline_comparison"
    )
    
    # 1. 无域适应的可视化
    print("1. 分析基线（无域适应）...")
    baseline_results = visualizer.visualize_domain_adaptation_complete(
        X_source, y_source, X_target, y_target,
        uda_method=None,
        method_name="Baseline_No_DA"
    )
    
    # 2. 使用最佳UDA方法
    print("2. 分析最佳UDA方法（SA）...")
    processor = create_uda_processor(
        method_name='SA',
        base_estimator=LogisticRegression(penalty=None, random_state=42, max_iter=1000),
        save_results=False
    )
    
    uda_method, _ = processor.fit_transform(
        X_source, y_source, X_target, y_target
    )
    
    uda_results = visualizer.visualize_domain_adaptation_complete(
        X_source, y_source, X_target, y_target,
        uda_method=uda_method,
        method_name="SA_Domain_Adaptation"
    )
    
    # 3. 生成对比总结
    print("\n=== 基线 vs UDA 对比总结 ===")
    
    if 'domain_distances' in baseline_results and 'domain_distances' in uda_results:
        baseline_dist = baseline_results['domain_distances']
        uda_dist = uda_results['domain_distances']
        
        print("域距离改进:")
        
        # 优先显示标准化指标
        priority_metrics = [
            ('normalized_kl_divergence_improvement', '标准化KL散度改进'),
            ('normalized_wasserstein_improvement', '标准化Wasserstein改进'),
            ('kl_divergence_improvement', 'KL散度改进'),
            ('wasserstein_improvement', 'Wasserstein改进'),
            ('mmd_improvement', 'MMD改进')
        ]
        
        for metric_key, metric_name in priority_metrics:
            if metric_key in baseline_dist and metric_key in uda_dist:
                baseline_val = baseline_dist.get(metric_key, 0)
                uda_val = uda_dist.get(metric_key, 0)
                print(f"  {metric_name}: 基线={baseline_val:.4f}, UDA={uda_val:.4f}")
    
    print(f"✓ 基线对比可视化完成")


def main():
    """主函数 - 运行所有可视化示例"""
    print("UDA可视化完整示例")
    print("=" * 70)
    
    try:
        # 检查环境
        from uda.adapt_methods import is_adapt_available
        
        if not is_adapt_available():
            print("❌ Adapt库不可用，请安装: pip install adapt-python")
            return
        
        print("✅ 环境检查通过")
        
        # 运行示例
        print("\n开始运行可视化示例...")
        
        # 示例1: 单个方法完整分析
        example_1_single_method_visualization()
        
        # 示例2: 多方法对比
        example_2_multiple_methods_comparison()
        
        # 示例3: 基线对比
        example_3_baseline_comparison()
        
        print("\n" + "="*70)
        print("✅ 所有可视化示例运行完成！")
        print("可视化结果已保存到 results/uda_examples/ 目录")
        print("="*70)
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请检查项目环境和依赖安装")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 