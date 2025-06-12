import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import Optional, List, Dict, Any
from .tsne_plots import visualize_tsne
from .histogram_plots import visualize_feature_histograms, histograms_stats_table
from .metrics import compute_domain_discrepancy

def compare_before_after_adaptation(source_features: np.ndarray, target_features: np.ndarray,
                                   adapted_target_features: np.ndarray, source_labels: np.ndarray,
                                   target_labels: np.ndarray, save_dir: str, method_name: str = "MMD",
                                   feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    生成适应前后的综合对比可视化
    
    参数:
    - source_features: 源域特征
    - target_features: 目标域特征（适应前）
    - adapted_target_features: 目标域特征（适应后）
    - source_labels: 源域标签
    - target_labels: 目标域标签
    - save_dir: 保存目录
    - method_name: 方法名称（如 "MMD-linear", "MMD-kpca"等）
    - feature_names: 特征名称列表
    
    返回:
    - comparison_results: 包含各种度量指标的字典
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. t-SNE 可视化
    tsne_path = os.path.join(save_dir, f"{method_name}_tsne_comparison.png")
    visualize_tsne(
        X_source=source_features,
        X_target=target_features,
        X_target_aligned=adapted_target_features,
        y_source=source_labels,
        y_target=target_labels,
        title=f'{method_name} Domain Adaptation - t-SNE Visualization',
        save_path=tsne_path,
        method_name=method_name
    )
    
    # 2. 特征分布直方图
    hist_path = os.path.join(save_dir, f"{method_name}_feature_histograms.png")
    visualize_feature_histograms(
        X_source=source_features,
        X_target=target_features,
        X_target_aligned=adapted_target_features,
        feature_names=feature_names,
        title=f'{method_name} Domain Adaptation - Feature Distributions',
        save_path=hist_path,
        method_name=method_name
    )
    
    # 3. 统计表格（只保留一个，移除重复的visual_stats）
    stats_path = os.path.join(save_dir, f"{method_name}_statistics_table.png")
    histograms_stats_table(
        X_source=source_features,
        X_target=target_features,
        X_target_aligned=adapted_target_features,
        feature_names=feature_names,
        save_path=stats_path,
        method_name=method_name
    )
    
    # 4. 计算域差异指标
    before_metrics = compute_domain_discrepancy(source_features, target_features)
    after_metrics = compute_domain_discrepancy(source_features, adapted_target_features)
    
    # 计算改进百分比
    improvement_metrics = {}
    for key in before_metrics:
        if isinstance(before_metrics[key], (int, float)) and isinstance(after_metrics[key], (int, float)):
            before_val = before_metrics[key]
            after_val = after_metrics[key]
            if abs(before_val) > 1e-9:
                if key == 'kernel_mean_difference':  # 越高越好
                    improvement_metrics[key] = (after_val - before_val) / abs(before_val) * 100
                else:  # 越低越好
                    improvement_metrics[key] = (before_val - after_val) / abs(before_val) * 100
            else:
                improvement_metrics[key] = 0.0
    
    comparison_results = {
        'before_metrics': before_metrics,
        'after_metrics': after_metrics,
        'improvement_metrics': improvement_metrics,
        'method_name': method_name,
        'visualizations': {
            'tsne_path': tsne_path,
            'histograms_path': hist_path,
            'stats_table_path': stats_path
        }
    }
    
    logging.info(f"✓ {method_name} 综合对比可视化完成，结果保存至: {save_dir}")
    
    return comparison_results

def visualize_mmd_adaptation_results(source_features: np.ndarray, target_features: np.ndarray,
                                   adapted_features: np.ndarray, source_labels: np.ndarray,
                                   target_labels: np.ndarray, output_dir: str,
                                   feature_names: Optional[List[str]] = None,
                                   method_name: str = "MMD") -> Dict[str, Any]:
    """
    生成完整的MMD适应结果可视化
    
    参数:
    - source_features: 源域特征
    - target_features: 目标域特征（适应前）
    - adapted_features: 适应后的特征
    - source_labels: 源域标签
    - target_labels: 目标域标签
    - output_dir: 输出目录
    - feature_names: 特征名称列表
    - method_name: MMD方法名称
    
    返回:
    - results: 包含所有结果的字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 调用综合对比函数
    comparison_results = compare_before_after_adaptation(
        source_features=source_features,
        target_features=target_features,
        adapted_target_features=adapted_features,
        source_labels=source_labels,
        target_labels=target_labels,
        save_dir=output_dir,
        method_name=method_name,
        feature_names=feature_names
    )
    
    # 生成总结报告
    summary_path = os.path.join(output_dir, f"{method_name}_adaptation_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"{method_name} Domain Adaptation Results Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Domain Discrepancy Metrics:\n")
        f.write("-" * 30 + "\n")
        
        before_metrics = comparison_results['before_metrics']
        after_metrics = comparison_results['after_metrics']
        improvement_metrics = comparison_results['improvement_metrics']
        
        for key in ['mmd', 'kl_divergence', 'wasserstein_distance', 'mean_difference', 'covariance_difference']:
            if key in before_metrics and key in after_metrics:
                f.write(f"{key.replace('_', ' ').title()}:\n")
                f.write(f"  Before: {before_metrics[key]:.6f}\n")
                f.write(f"  After:  {after_metrics[key]:.6f}\n")
                if key in improvement_metrics:
                    f.write(f"  Improvement: {improvement_metrics[key]:.2f}%\n")
                f.write("\n")
        
        f.write("Generated Visualizations:\n")
        f.write("-" * 25 + "\n")
        for viz_name, viz_path in comparison_results['visualizations'].items():
            f.write(f"- {viz_name}: {os.path.basename(viz_path)}\n")
    
    logging.info(f"✓ {method_name} 完整适应结果可视化完成，总结报告: {summary_path}")
    
    return comparison_results

def plot_mmd_methods_comparison(results_dict: Dict[str, Dict[str, Any]], save_path: str) -> None:
    """
    对比不同MMD方法的效果
    
    参数:
    - results_dict: 包含不同MMD方法结果的字典
    - save_path: 保存路径
    """
    methods = list(results_dict.keys())
    metrics_to_compare = ['mmd', 'kl_divergence', 'wasserstein_distance', 'covariance_difference']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics_to_compare):
        ax = axes[i]
        
        before_values = []
        after_values = []
        method_names = []
        
        for method in methods:
            if metric in results_dict[method]['before_metrics'] and metric in results_dict[method]['after_metrics']:
                before_values.append(results_dict[method]['before_metrics'][metric])
                after_values.append(results_dict[method]['after_metrics'][metric])
                method_names.append(method)
        
        if before_values and after_values:
            x = np.arange(len(method_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, before_values, width, label='Before Adaptation', alpha=0.8)
            bars2 = ax.bar(x + width/2, after_values, width, label='After Adaptation', alpha=0.8)
            
            # 添加数值标签
            for bar in bars1:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
            
            for bar in bars2:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('MMD Methods')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(method_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('MMD Methods Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"MMD methods comparison plot saved to: {save_path}")
    
    plt.close(fig) 

def generate_performance_comparison_plots(results_dict: Dict[str, Dict[str, Any]], 
                                        save_dir: str,
                                        experiment_name: str = "Experiment") -> None:
    """
    生成完整的性能对比图表套件
    
    参数:
    - results_dict: 实验结果字典
    - save_dir: 保存目录
    - experiment_name: 实验名称
    """
    from .performance_plots import (
        plot_metrics_comparison, plot_domain_adaptation_improvement,
        plot_cross_dataset_performance, plot_model_comparison,
        plot_metrics_radar_chart, create_performance_summary_table
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 基础性能指标对比图
    metrics_comparison_path = os.path.join(save_dir, f"{experiment_name}_metrics_comparison.png")
    plot_metrics_comparison(
        results_dict=results_dict,
        save_path=metrics_comparison_path,
        title=f"{experiment_name} - Performance Metrics Comparison"
    )
    
    # 2. 域适应改进效果图
    improvement_path = os.path.join(save_dir, f"{experiment_name}_domain_adaptation_improvement.png")
    plot_domain_adaptation_improvement(
        results_dict=results_dict,
        save_path=improvement_path,
        title=f"{experiment_name} - Domain Adaptation Improvement"
    )
    
    # 3. 跨数据集性能对比图
    cross_dataset_path = os.path.join(save_dir, f"{experiment_name}_cross_dataset_performance.png")
    plot_cross_dataset_performance(
        results_dict=results_dict,
        save_path=cross_dataset_path,
        title=f"{experiment_name} - Cross-Dataset Performance"
    )
    
    # 4. 模型性能对比热力图
    model_comparison_path = os.path.join(save_dir, f"{experiment_name}_model_comparison.png")
    plot_model_comparison(
        results_dict=results_dict,
        save_path=model_comparison_path,
        title=f"{experiment_name} - Model Performance Comparison"
    )
    
    # 5. 性能指标雷达图
    radar_chart_path = os.path.join(save_dir, f"{experiment_name}_performance_radar.png")
    plot_metrics_radar_chart(
        results_dict=results_dict,
        save_path=radar_chart_path,
        title=f"{experiment_name} - Performance Radar Chart"
    )
    
    # 6. 性能汇总表格
    summary_table_path = os.path.join(save_dir, f"{experiment_name}_performance_summary.png")
    summary_df = create_performance_summary_table(
        results_dict=results_dict,
        save_path=summary_table_path,
        title=f"{experiment_name} - Performance Summary Table"
    )
    
    logging.info(f"✓ 完整性能对比图表套件已生成，保存至: {save_dir}")
    logging.info(f"  - 指标对比图: {os.path.basename(metrics_comparison_path)}")
    logging.info(f"  - 改进效果图: {os.path.basename(improvement_path)}")
    logging.info(f"  - 跨数据集对比: {os.path.basename(cross_dataset_path)}")
    logging.info(f"  - 模型对比热力图: {os.path.basename(model_comparison_path)}")
    logging.info(f"  - 雷达图: {os.path.basename(radar_chart_path)}")
    logging.info(f"  - 汇总表格: {os.path.basename(summary_table_path)}")
    
    return summary_df 