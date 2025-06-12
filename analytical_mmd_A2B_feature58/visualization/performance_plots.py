"""
性能对比图可视化模块

提供各种性能指标的对比可视化功能，包括：
- 多模型性能对比
- 域适应前后性能对比
- 跨数据集性能对比
- 指标趋势图
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Optional, Any, Tuple
import logging

def plot_metrics_comparison(results_dict: Dict[str, Dict[str, Any]], 
                          save_path: Optional[str] = None,
                          title: str = "Performance Metrics Comparison",
                          figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    绘制多个实验结果的性能指标对比图
    
    参数:
    - results_dict: 实验结果字典，格式为 {experiment_name: {metrics}}
    - save_path: 保存路径
    - title: 图表标题
    - figsize: 图表大小
    """
    # 提取指标数据
    metrics_data = []
    experiment_names = []
    
    for exp_name, results in results_dict.items():
        experiment_names.append(exp_name)
        
        # 处理不同格式的结果
        if 'without_domain_adaptation' in results:
            # 跨域实验结果格式
            no_da = results['without_domain_adaptation']
            with_da = results.get('with_domain_adaptation', no_da)
            
            # 提取指标值
            if 'means' in no_da:
                # CV结果格式
                metrics_data.append({
                    'experiment': f"{exp_name} (No DA)",
                    'accuracy': no_da['means']['accuracy'],
                    'auc': no_da['means']['auc'],
                    'f1': no_da['means']['f1'],
                    'acc_0': no_da['means']['acc_0'],
                    'acc_1': no_da['means']['acc_1']
                })
                metrics_data.append({
                    'experiment': f"{exp_name} (With DA)",
                    'accuracy': with_da['means']['accuracy'],
                    'auc': with_da['means']['auc'],
                    'f1': with_da['means']['f1'],
                    'acc_0': with_da['means']['acc_0'],
                    'acc_1': with_da['means']['acc_1']
                })
            else:
                # 单次评估结果格式
                metrics_data.append({
                    'experiment': f"{exp_name} (No DA)",
                    'accuracy': no_da['accuracy'],
                    'auc': no_da['auc'],
                    'f1': no_da['f1'],
                    'acc_0': no_da['acc_0'],
                    'acc_1': no_da['acc_1']
                })
                metrics_data.append({
                    'experiment': f"{exp_name} (With DA)",
                    'accuracy': with_da['accuracy'],
                    'auc': with_da['auc'],
                    'f1': with_da['f1'],
                    'acc_0': with_da['acc_0'],
                    'acc_1': with_da['acc_1']
                })
        else:
            # 直接的指标格式
            if isinstance(results.get('accuracy'), str):
                # CV格式 "0.85 ± 0.03"
                acc_val = float(results['accuracy'].split(' ± ')[0])
                auc_val = float(results['auc'].split(' ± ')[0])
                f1_val = float(results['f1'].split(' ± ')[0])
                acc_0_val = float(results['acc_0'].split(' ± ')[0])
                acc_1_val = float(results['acc_1'].split(' ± ')[0])
            else:
                # 直接数值格式
                acc_val = results['accuracy']
                auc_val = results['auc']
                f1_val = results['f1']
                acc_0_val = results['acc_0']
                acc_1_val = results['acc_1']
            
            metrics_data.append({
                'experiment': exp_name,
                'accuracy': acc_val,
                'auc': auc_val,
                'f1': f1_val,
                'acc_0': acc_0_val,
                'acc_1': acc_1_val
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(metrics_data)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    metrics = ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']
    metric_names = ['Accuracy', 'AUC', 'F1-Score', 'Class 0 Accuracy', 'Class 1 Accuracy']
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(df)))
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # 绘制柱状图
        bars = ax.bar(range(len(df)), df[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for bar, value in zip(bars, df[metric]):
            height = bar.get_height()
            ax.annotate(f'{value:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['experiment'], rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.1)
    
    # 隐藏最后一个子图
    axes[-1].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    # 调整布局以防止标签重叠
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"性能对比图已保存到: {save_path}")
    
    plt.close(fig)

def plot_domain_adaptation_improvement(results_dict: Dict[str, Dict[str, Any]], 
                                     save_path: Optional[str] = None,
                                     title: str = "Domain Adaptation Improvement",
                                     figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    绘制域适应改进效果图
    
    参数:
    - results_dict: 实验结果字典
    - save_path: 保存路径
    - title: 图表标题
    - figsize: 图表大小
    """
    improvement_data = []
    
    for exp_name, results in results_dict.items():
        if 'improvement' in results:
            improvement = results['improvement']
            if isinstance(improvement, dict):
                # 处理不同格式的改进数据
                if 'auc_improvement' in improvement:
                    # 单次评估格式
                    improvement_data.append({
                        'experiment': exp_name,
                        'auc_improvement': improvement['auc_improvement'],
                        'accuracy_improvement': improvement['accuracy_improvement']
                    })
                else:
                    # CV格式
                    for metric in ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']:
                        if metric in improvement:
                            value = improvement[metric]
                            if isinstance(value, str):
                                value = float(value.split(' ± ')[0])
                            improvement_data.append({
                                'experiment': exp_name,
                                'metric': metric,
                                'improvement': value
                            })
    
    if not improvement_data:
        logging.warning("没有找到改进数据，跳过改进效果图")
        return
    
    df = pd.DataFrame(improvement_data)
    
    if 'metric' in df.columns:
        # CV格式数据
        fig, ax = plt.subplots(figsize=figsize)
        
        # 透视表
        pivot_df = df.pivot(index='experiment', columns='metric', values='improvement')
        
        # 绘制热力图
        sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Improvement'}, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Experiments', fontsize=12)
        
    else:
        # 单次评估格式数据
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        experiments = df['experiment']
        auc_improvements = df['auc_improvement']
        acc_improvements = df['accuracy_improvement']
        
        # AUC改进
        bars1 = ax1.bar(range(len(experiments)), auc_improvements, 
                       color='skyblue', alpha=0.8, edgecolor='black')
        ax1.set_title('AUC Improvement', fontsize=12, fontweight='bold')
        ax1.set_ylabel('AUC Improvement', fontsize=10)
        ax1.set_xticks(range(len(experiments)))
        ax1.set_xticklabels(experiments, rotation=45, ha='right')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        
        # 添加数值标签
        for bar, value in zip(bars1, auc_improvements):
            height = bar.get_height()
            ax1.annotate(f'{value:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15), 
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9, fontweight='bold')
        
        # Accuracy改进
        bars2 = ax2.bar(range(len(experiments)), acc_improvements, 
                       color='lightcoral', alpha=0.8, edgecolor='black')
        ax2.set_title('Accuracy Improvement', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy Improvement', fontsize=10)
        ax2.set_xticks(range(len(experiments)))
        ax2.set_xticklabels(experiments, rotation=45, ha='right')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        
        # 添加数值标签
        for bar, value in zip(bars2, acc_improvements):
            height = bar.get_height()
            ax2.annotate(f'{value:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15), 
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9, fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"域适应改进图已保存到: {save_path}")
    
    plt.close(fig)

def plot_cross_dataset_performance(results_dict: Dict[str, Dict[str, Any]], 
                                 save_path: Optional[str] = None,
                                 title: str = "Cross-Dataset Performance",
                                 figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    绘制跨数据集性能对比图
    
    参数:
    - results_dict: 包含多个数据集结果的字典
    - save_path: 保存路径
    - title: 图表标题
    - figsize: 图表大小
    """
    # 组织数据
    datasets = []
    metrics_data = {
        'accuracy': {'no_da': [], 'with_da': []},
        'auc': {'no_da': [], 'with_da': []},
        'f1': {'no_da': [], 'with_da': []},
        'acc_0': {'no_da': [], 'with_da': []},
        'acc_1': {'no_da': [], 'with_da': []}
    }
    
    for dataset_name, results in results_dict.items():
        # 更合理地简化数据集名称显示
        display_name = dataset_name
        # 移除冗余的前缀，但保留核心信息
        if display_name.startswith('Dataset_'):
            display_name = display_name.replace('Dataset_', '')
        # 简化MMD方法名称
        display_name = display_name.replace('_MMD_Linear', ' (Linear)')
        display_name = display_name.replace('_MMD_KPCA', ' (KPCA)')
        display_name = display_name.replace('_MMD_', ' (MMD)')
        datasets.append(display_name)
        
        if 'without_domain_adaptation' in results:
            no_da = results['without_domain_adaptation']
            with_da = results.get('with_domain_adaptation', no_da)
            
            for metric in metrics_data.keys():
                # 提取数值
                if 'means' in no_da:
                    no_da_val = no_da['means'][metric]
                    with_da_val = with_da['means'][metric]
                else:
                    no_da_val = no_da[metric]
                    with_da_val = with_da[metric]
                
                metrics_data[metric]['no_da'].append(no_da_val)
                metrics_data[metric]['with_da'].append(with_da_val)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    metrics = ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']
    metric_names = ['Accuracy', 'AUC', 'F1-Score', 'Class 0 Accuracy', 'Class 1 Accuracy']
    
    x = np.arange(len(datasets))
    width = 0.35
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        no_da_values = metrics_data[metric]['no_da']
        with_da_values = metrics_data[metric]['with_da']
        
        bars1 = ax.bar(x - width/2, no_da_values, width, label='Without DA', 
                      alpha=0.8, color='lightcoral', edgecolor='black')
        bars2 = ax.bar(x + width/2, with_da_values, width, label='With DA', 
                      alpha=0.8, color='skyblue', edgecolor='black')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xlabel('Dataset', fontsize=10)
        ax.set_xticks(x)
        # 旋转x轴标签并调整对齐方式
        ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, 1.1)
    
    # 隐藏最后一个子图
    axes[-1].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    # 调整布局以防止标签重叠
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"跨数据集性能图已保存到: {save_path}")
    
    plt.close(fig)

def plot_model_comparison(results_dict: Dict[str, Dict[str, Any]], 
                         save_path: Optional[str] = None,
                         title: str = "Model Performance Comparison",
                         figsize: Tuple[int, int] = (16, 10)) -> None:
    """
    绘制不同模型的性能对比图
    
    参数:
    - results_dict: 不同模型的结果字典
    - save_path: 保存路径
    - title: 图表标题
    - figsize: 图表大小
    """
    # 提取数据
    models = list(results_dict.keys())
    metrics = ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']
    metric_names = ['Accuracy', 'AUC', 'F1-Score', 'Class 0 Accuracy', 'Class 1 Accuracy']
    
    # 准备数据矩阵
    data_matrix = np.zeros((len(models), len(metrics)))
    
    for i, model in enumerate(models):
        results = results_dict[model]
        
        # 处理不同格式的结果
        if 'with_domain_adaptation' in results:
            # 使用域适应后的结果
            target_results = results['with_domain_adaptation']
        elif 'without_domain_adaptation' in results:
            # 使用无域适应的结果
            target_results = results['without_domain_adaptation']
        else:
            # 直接使用结果
            target_results = results
        
        for j, metric in enumerate(metrics):
            if 'means' in target_results:
                value = target_results['means'][metric]
            else:
                value = target_results[metric]
            
            if isinstance(value, str):
                value = float(value.split(' ± ')[0])
            
            data_matrix[i, j] = value
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metric_names)
    ax.set_yticklabels(models)
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 添加数值标签
    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                   ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Performance Value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"模型对比图已保存到: {save_path}")
    
    plt.close(fig)

def plot_metrics_radar_chart(results_dict: Dict[str, Dict[str, Any]], 
                           save_path: Optional[str] = None,
                           title: str = "Performance Radar Chart",
                           figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    绘制性能指标雷达图
    
    参数:
    - results_dict: 实验结果字典
    - save_path: 保存路径
    - title: 图表标题
    - figsize: 图表大小
    """
    metrics = ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']
    metric_names = ['Accuracy', 'AUC', 'F1-Score', 'Class 0 Acc', 'Class 1 Acc']
    
    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(results_dict)))
    
    for i, (exp_name, results) in enumerate(results_dict.items()):
        # 提取指标值
        values = []
        
        # 处理不同格式的结果
        if 'with_domain_adaptation' in results:
            target_results = results['with_domain_adaptation']
        elif 'without_domain_adaptation' in results:
            target_results = results['without_domain_adaptation']
        else:
            target_results = results
        
        for metric in metrics:
            if 'means' in target_results:
                value = target_results['means'][metric]
            else:
                value = target_results[metric]
            
            if isinstance(value, str):
                value = float(value.split(' ± ')[0])
            
            values.append(value)
        
        values += values[:1]  # 闭合图形
        
        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2, label=exp_name, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"雷达图已保存到: {save_path}")
    
    plt.close(fig)

def create_performance_summary_table(results_dict: Dict[str, Dict[str, Any]], 
                                    save_path: Optional[str] = None,
                                    title: str = "Performance Summary Table") -> pd.DataFrame:
    """
    创建性能汇总表格
    
    参数:
    - results_dict: 实验结果字典
    - save_path: 保存路径
    - title: 表格标题
    
    返回:
    - pd.DataFrame: 汇总表格
    """
    summary_data = []
    
    for exp_name, results in results_dict.items():
        # 处理不同格式的结果
        if 'without_domain_adaptation' in results and 'with_domain_adaptation' in results:
            # 跨域实验格式
            no_da = results['without_domain_adaptation']
            with_da = results['with_domain_adaptation']
            
            for da_type, da_results in [('Without DA', no_da), ('With DA', with_da)]:
                row_data = {'Experiment': f"{exp_name} ({da_type})"}
                
                for metric in ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']:
                    if 'means' in da_results:
                        value = da_results['means'][metric]
                        std = da_results['stds'][metric]
                        row_data[metric.upper()] = f"{value:.4f} ± {std:.4f}"
                    else:
                        value = da_results[metric]
                        row_data[metric.upper()] = f"{value:.4f}"
                
                summary_data.append(row_data)
        else:
            # 直接结果格式
            row_data = {'Experiment': exp_name}
            
            for metric in ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1']:
                if metric in results:
                    value = results[metric]
                    if isinstance(value, str):
                        row_data[metric.upper()] = value
                    else:
                        row_data[metric.upper()] = f"{value:.4f}"
                else:
                    row_data[metric.upper()] = "N/A"
            
            summary_data.append(row_data)
    
    # 创建DataFrame
    df = pd.DataFrame(summary_data)
    
    if save_path:
        # 保存为CSV
        csv_path = save_path.replace('.png', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 创建表格图像
        fig, ax = plt.subplots(figsize=(16, max(6, len(df) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 设置表格样式
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"性能汇总表格已保存到: {csv_path} 和 {save_path}")
    
    return df 