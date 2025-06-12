import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import pandas as pd # 导入 pandas 用于类型注解
import seaborn as sns

def plot_experiment_comparison(results_df: pd.DataFrame, config_name: str, base_save_path: str):
    """
    生成并保存实验结果的对比图。

    参数:
    - results_df: 包含实验结果的 Pandas DataFrame。
                  期望包含的列: 'direct_acc', 'direct_auc', 'direct_f1',
                               'target_acc', 'target_auc', 'target_f1',
                               'direct_acc_0', 'direct_acc_1', 'target_acc_0', 'target_acc_1',
                               'mean_diff_initial', 'std_diff_initial',
                               'mean_diff_after_align', 'std_diff_after_align',
                               'train_time', 'align_time', 'inference_time'
    - config_name: 实验配置的名称 (例如 'A_to_B')。
    - base_save_path: 保存图表的目录路径。
    """
    try:
        # 确保保存路径存在
        os.makedirs(base_save_path, exist_ok=True)

        # 假设 results_df 只有一行，对应当前的实验配置
        if results_df.empty:
            logging.warning("Results DataFrame is empty. Skipping comparison plot generation.")
            return
        
        row_plot = results_df.iloc[0]

        plt.figure(figsize=(12, 8))
        target_name_plot = config_name # 使用配置名称作为目标名称的一部分

        # 绘制准确率和AUC比较
        plt.subplot(2, 2, 1)
        metrics_names_plot = ['acc', 'auc', 'f1']
        direct_values_plot = [row_plot['direct_acc'], row_plot['direct_auc'], row_plot['direct_f1']]
        coral_values_plot = [row_plot['target_acc'], row_plot['target_auc'], row_plot['target_f1']]
        
        x_metrics_plot = np.arange(len(metrics_names_plot))
        width_plot = 0.35
        
        plt.bar(x_metrics_plot - width_plot/2, direct_values_plot, width_plot, label='Direct TabPFN')
        plt.bar(x_metrics_plot + width_plot/2, coral_values_plot, width_plot, label='With Analytical CORAL')
        
        for i, v_plot in enumerate(direct_values_plot):
            plt.text(i - width_plot/2, v_plot + 0.01, f'{v_plot:.3f}', ha='center')
        for i, v_plot in enumerate(coral_values_plot):
            plt.text(i + width_plot/2, v_plot + 0.01, f'{v_plot:.3f}', ha='center')
            
        plt.ylabel('Score')
        plt.title('Performance Metrics')
        plt.xticks(x_metrics_plot, ['Accuracy', 'AUC', 'F1'])
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # 绘制每类准确率
        plt.subplot(2, 2, 2)
        class_metrics_names_plot = ['Class 0 Acc', 'Class 1 Acc']
        direct_class_values_plot = [row_plot['direct_acc_0'], row_plot['direct_acc_1']]
        coral_class_values_plot = [row_plot['target_acc_0'], row_plot['target_acc_1']]
        
        x_class_plot = np.arange(len(class_metrics_names_plot))
        
        plt.bar(x_class_plot - width_plot/2, direct_class_values_plot, width_plot, label='Direct TabPFN')
        plt.bar(x_class_plot + width_plot/2, coral_class_values_plot, width_plot, label='With Analytical CORAL')
        
        for i, v_plot in enumerate(direct_class_values_plot):
            plt.text(i - width_plot/2, v_plot + 0.01, f'{v_plot:.3f}', ha='center')
        for i, v_plot in enumerate(coral_class_values_plot):
            plt.text(i + width_plot/2, v_plot + 0.01, f'{v_plot:.3f}', ha='center')
            
        plt.ylabel('Accuracy')
        plt.title('Per-Class Accuracy')
        plt.xticks(x_class_plot, class_metrics_names_plot)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # 绘制特征差异
        plt.subplot(2, 2, 3)
        diff_metrics_names_plot = ['Mean Diff', 'Std Diff']
        before_values_plot = [row_plot['mean_diff_initial'], row_plot['std_diff_initial']] # 键名已在脚本中更改
        after_values_plot = [row_plot['mean_diff_after_align'], row_plot['std_diff_after_align']] # 键名已在脚本中更改
        
        x_diff_plot = np.arange(len(diff_metrics_names_plot))
        
        plt.bar(x_diff_plot - width_plot/2, before_values_plot, width_plot, label='Before Alignment')
        plt.bar(x_diff_plot + width_plot/2, after_values_plot, width_plot, label='After Alignment')
        
        for i, (before_p, after_p) in enumerate(zip(before_values_plot, after_values_plot)):
            plt.text(i - width_plot/2, before_p + 0.01, f'{before_p:.3f}', ha='center')
            plt.text(i + width_plot/2, after_p + 0.01, f'{after_p:.3f}', ha='center')
            reduction = ((before_p - after_p) / before_p * 100) if before_p != 0 else 0
            plt.text(i, max(0.01, after_p/2) , f'-{reduction:.1f}%', ha='center', color='black', fontweight='bold') # Ensure text is visible
            
        plt.ylabel('Difference')
        plt.title('Feature Distribution Difference')
        plt.xticks(x_diff_plot, diff_metrics_names_plot)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # 绘制计算时间
        plt.subplot(2, 2, 4)
        # 键名已在脚本中更改
        time_keys = ['train_time', 'align_time', 'inference_time'] 
        time_labels = ['TabPFN Training', 'CORAL Alignment', 'Inference']
        time_values = [row_plot[key] for key in time_keys]

        plt.bar(time_labels, time_values)
        plt.ylabel('Time (seconds)')
        plt.title('Computation Time')
        plt.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Analytical CORAL Results: {target_name_plot}', fontsize=16) # 使用 target_name_plot
        plt.tight_layout()
        plt.subplots_adjust(top=0.9) # 调整标题和子图间距
        
        # 文件名修正
        save_file_name = f"comparison_analytical_coral_{config_name.replace(' -> ', '_to_')}.png"
        full_save_path = os.path.join(base_save_path, save_file_name)
        plt.savefig(full_save_path, dpi=300)
        plt.close()
        logging.info(f"Comparison plot saved to {full_save_path}")

    except Exception as e:
        logging.error(f"Error generating comparison plot for {config_name}: {e}", exc_info=True)

def plot_class_conditional_variants_comparison(
    direct_metrics: pd.Series,
    pseudo_label_metrics: pd.Series,
    true_label_metrics: pd.Series,
    config_name: str,
    base_save_path: str,
    target_name_plot: str = "Target Domain"
):
    """
    Generates and saves a comparison plot for different Class-Conditional CORAL variants
    against direct application.

    Args:
    - direct_metrics: Pandas Series with metrics for direct application (e.g., acc, auc, f1).
                      Expected keys like 'direct_acc', 'direct_auc'.
    - pseudo_label_metrics: Pandas Series with metrics for C-CORAL with pseudo labels.
                            Expected keys like 'pseudo_acc', 'pseudo_auc'.
    - true_label_metrics: Pandas Series with metrics for C-CORAL with true labels.
                          Expected keys like 'withlabels_acc', 'withlabels_auc'.
    - config_name: Name for the plot file (e.g., "A_to_B_ClassCORAL_Comparison").
    - base_save_path: Directory path to save the plot.
    - target_name_plot: Name of the target domain for the plot title.
    """
    try:
        os.makedirs(base_save_path, exist_ok=True)
        
        metrics_to_plot = ['acc', 'auc', 'f1'] # Key metrics to compare
        labels = ['Direct', 'C-CORAL (Pseudo)', 'C-CORAL (True Labels)']
        
        data_for_plot = {
            'Metric': [],
            'Score': [],
            'Method': []
        }

        for metric in metrics_to_plot:
            # Extract scores, ensuring keys match how they are stored in the series
            # Adjust key prefixes if necessary (e.g. if series already have 'direct_', 'pseudo_', etc.)
            # For direct_metrics, we assume keys like 'target_acc', 'target_auc' if they come from a 'plain_coral_df'
            # or 'direct_acc' etc. if directly from direct evaluation. Let's standardize to not require prefix in input series.
            
            # Standardize key access: assume input series have simple keys like 'acc', 'auc'
            direct_score = direct_metrics.get(metric, direct_metrics.get(f'target_{metric}', direct_metrics.get(f'direct_{metric}', np.nan)))
            pseudo_score = pseudo_label_metrics.get(metric, pseudo_label_metrics.get(f'pseudo_{metric}', np.nan))
            true_label_score = true_label_metrics.get(metric, true_label_metrics.get(f'withlabels_{metric}', np.nan))

            scores = [direct_score, pseudo_score, true_label_score]
            
            for i, method_label in enumerate(labels):
                data_for_plot['Metric'].append(metric.upper())
                data_for_plot['Score'].append(scores[i])
                data_for_plot['Method'].append(method_label)

        df_plot = pd.DataFrame(data_for_plot)
        
        if df_plot['Score'].isnull().all():
            logging.warning(f"No valid scores found to plot for {config_name}. Skipping plot generation.")
            return

        plt.figure(figsize=(12, 7))
        sns.barplot(x='Metric', y='Score', hue='Method', data=df_plot, palette='viridis')
        
        plt.title(f'Class-Conditional CORAL Variants Comparison on {target_name_plot}', fontsize=15, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.ylim(0, 1.05) # Assuming scores are between 0 and 1
        plt.legend(title='Adaptation Method', loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        plot_filename = f"{config_name}_comparison.png"
        full_save_path = os.path.join(base_save_path, plot_filename)
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Class-Conditional CORAL variants comparison plot saved to: {full_save_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error generating class-conditional variants comparison plot for {config_name}: {e}", exc_info=True) 