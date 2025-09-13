#!/usr/bin/env python3
"""
特征扫描分析脚本 - 运行从best3到best58的所有特征组合分析

这个脚本自动运行所有可能的特征配置，生成性能比较可视化，
便于找到最优的特征组合进行医疗数据域适应分析。

运行示例: python scripts/run_feature_sweep_analysis.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置matplotlib中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.style.use('seaborn-v0_8-whitegrid')

from scripts.run_complete_analysis import CompleteAnalysisRunner

# Direct settings import to avoid yaml dependency
import importlib.util
def load_settings_direct():
    """Load settings module directly without complex imports"""
    settings_path = project_root / "config" / "settings.py"
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    return settings

def get_features_by_type(feature_type: str):
    """Get features by type without complex imports - support all best3-best58"""
    settings = load_settings_direct()
    
    # 直接使用settings模块的get_features_by_type函数
    try:
        return settings.get_features_by_type(feature_type)
    except Exception:
        # 备选方案：如果settings.py的get_features_by_type函数失败，尝试直接访问属性
        try:
            # 动态获取特征集属性
            if feature_type == 'all63':
                return getattr(settings, 'ALL_63_FEATURES', [])
            elif feature_type == 'selected58':
                return getattr(settings, 'SELECTED_58_FEATURES', [])
            else:
                # 对于bestN特征集，尝试获取对应的属性
                attr_name = f"{feature_type.upper()}_FEATURES"
                return getattr(settings, attr_name, [])
        except Exception:
            # 如果所有方法都失败，返回空列表
            return []


class FeatureSweepAnalyzer:
    """特征扫描分析器"""
    
    def __init__(
        self,
        feature_range: Tuple[int, int] = (3, 58),
        output_dir: Optional[str] = None,
        max_workers: int = None,
        verbose: bool = True
    ):
        """
        初始化特征扫描分析器
        
        Args:
            feature_range: 特征数量范围 (min_features, max_features)
            output_dir: 输出目录
            max_workers: 最大并行工作进程数
            verbose: 是否输出详细信息
        """
        self.feature_range = feature_range
        self.verbose = verbose
        
        # 设置并行工作数
        if max_workers is None:
            self.max_workers = min(4, mp.cpu_count() // 2)  # 保守的并行度
        else:
            self.max_workers = max_workers
            
        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results/feature_sweep_analysis_{timestamp}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.results_dir = self.output_dir / "individual_results"
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 存储结果
        self.sweep_results = {}
        self.performance_summary = {}
        
        if self.verbose:
            print(f"🔧 特征扫描分析器初始化")
            print(f"   特征范围: best{feature_range[0]} ~ best{feature_range[1]}")
            print(f"   输出目录: {output_dir}")
            print(f"   最大并行数: {self.max_workers}")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.output_dir / "feature_sweep.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler() if self.verbose else logging.NullHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_available_feature_sets(self) -> List[str]:
        """获取可用的特征集列表"""
        available_sets = []
        min_features, max_features = self.feature_range
        
        for n_features in range(min_features, max_features + 1):
            feature_type = f"best{n_features}"
            try:
                features = get_features_by_type(feature_type)
                if features:
                    available_sets.append(feature_type)
                    if self.verbose:
                        print(f"✅ {feature_type}: {len(features)}个特征")
                else:
                    if self.verbose:
                        print(f"⚠️ {feature_type}: 未定义")
            except Exception as e:
                if self.verbose:
                    print(f"❌ {feature_type}: {e}")
        
        return available_sets
    
    def run_single_feature_analysis(self, feature_type: str) -> Dict:
        """运行单个特征集的分析"""
        try:
            self.logger.info(f"开始分析特征集: {feature_type}")
            
            # 创建个体结果目录
            individual_output_dir = self.results_dir / feature_type
            
            # 创建分析运行器
            runner = CompleteAnalysisRunner(
                feature_type=feature_type,
                scaler_type='none',  # 使用默认配置
                imbalance_method='none',
                cv_folds=10,
                random_state=42,
                output_dir=str(individual_output_dir),
                verbose=False  # 关闭详细输出避免日志混乱
            )
            
            # 运行完整分析
            results = runner.run_complete_analysis()
            
            if 'error' in results:
                self.logger.error(f"特征集 {feature_type} 分析失败: {results['error']}")
                return {
                    'feature_type': feature_type,
                    'status': 'failed',
                    'error': results['error'],
                    'n_features': int(feature_type.replace('best', ''))
                }
            
            # 提取性能指标
            performance = self.extract_performance_metrics(results, feature_type)
            
            self.logger.info(f"特征集 {feature_type} 分析完成")
            return performance
            
        except Exception as e:
            self.logger.error(f"特征集 {feature_type} 执行异常: {e}")
            return {
                'feature_type': feature_type,
                'status': 'failed',
                'error': str(e),
                'n_features': int(feature_type.replace('best', ''))
            }
    
    def extract_performance_metrics(self, results: Dict, feature_type: str) -> Dict:
        """提取性能指标"""
        performance = {
            'feature_type': feature_type,
            'n_features': int(feature_type.replace('best', '')),
            'status': 'success'
        }
        
        # 1. 提取源域交叉验证结果 (TabPFN)
        if 'source_domain_cv' in results:
            cv_results = results['source_domain_cv']
            
            # 找到TabPFN结果
            tabpfn_key = None
            for key in cv_results.keys():
                if 'tabpfn' in key.lower():
                    tabpfn_key = key
                    break
            
            if tabpfn_key and 'summary' in cv_results[tabpfn_key]:
                summary = cv_results[tabpfn_key]['summary']
                performance.update({
                    'source_auc': summary.get('auc_mean', 0),
                    'source_accuracy': summary.get('accuracy_mean', 0),
                    'source_f1': summary.get('f1_mean', 0),
                    'source_precision': summary.get('precision_mean', 0),
                    'source_recall': summary.get('recall_mean', 0),
                    'source_auc_std': summary.get('auc_std', 0)
                })
        
        # 2. 提取UDA结果
        if 'uda_methods' in results:
            uda_results = results['uda_methods']
            
            # TabPFN无UDA基线
            if 'TabPFN_NoUDA' in uda_results:
                baseline = uda_results['TabPFN_NoUDA']
                if 'error' not in baseline:
                    performance.update({
                        'target_baseline_auc': baseline.get('auc', 0),
                        'target_baseline_accuracy': baseline.get('accuracy', 0),
                        'target_baseline_f1': baseline.get('f1', 0)
                    })
            
            # TCA结果
            if 'TCA' in uda_results:
                tca = uda_results['TCA']
                if 'error' not in tca:
                    performance.update({
                        'target_tca_auc': tca.get('auc', 0),
                        'target_tca_accuracy': tca.get('accuracy', 0),
                        'target_tca_f1': tca.get('f1', 0)
                    })

                    # 计算TCA相对于基线的提升
                    if 'target_baseline_auc' in performance and performance['target_baseline_auc'] > 0:
                        improvement = performance['target_tca_auc'] - performance['target_baseline_auc']
                        performance['tca_auc_improvement'] = improvement

            # SA结果
            if 'SA' in uda_results:
                sa = uda_results['SA']
                if 'error' not in sa:
                    performance.update({
                        'target_sa_auc': sa.get('auc', 0),
                        'target_sa_accuracy': sa.get('accuracy', 0),
                        'target_sa_f1': sa.get('f1', 0)
                    })

                    # 计算SA相对于基线的提升
                    if 'target_baseline_auc' in performance and performance['target_baseline_auc'] > 0:
                        improvement = performance['target_sa_auc'] - performance['target_baseline_auc']
                        performance['sa_auc_improvement'] = improvement

            # CORAL结果
            if 'CORAL' in uda_results:
                coral = uda_results['CORAL']
                if 'error' not in coral:
                    performance.update({
                        'target_coral_auc': coral.get('auc', 0),
                        'target_coral_accuracy': coral.get('accuracy', 0),
                        'target_coral_f1': coral.get('f1', 0)
                    })

                    # 计算CORAL相对于基线的提升
                    if 'target_baseline_auc' in performance and performance['target_baseline_auc'] > 0:
                        improvement = performance['target_coral_auc'] - performance['target_baseline_auc']
                        performance['coral_auc_improvement'] = improvement

            # KMM结果
            if 'KMM' in uda_results:
                kmm = uda_results['KMM']
                if 'error' not in kmm:
                    performance.update({
                        'target_kmm_auc': kmm.get('auc', 0),
                        'target_kmm_accuracy': kmm.get('accuracy', 0),
                        'target_kmm_f1': kmm.get('f1', 0)
                    })

                    # 计算KMM相对于基线的提升
                    if 'target_baseline_auc' in performance and performance['target_baseline_auc'] > 0:
                        improvement = performance['target_kmm_auc'] - performance['target_baseline_auc']
                        performance['kmm_auc_improvement'] = improvement
        
        return performance
    
    def run_parallel_feature_sweep(self) -> Dict:
        """并行运行特征扫描分析"""
        if self.verbose:
            print(f"\n🚀 开始并行特征扫描分析")
            print("=" * 50)
        
        # 获取可用特征集
        available_sets = self.get_available_feature_sets()
        
        if not available_sets:
            raise ValueError("没有找到可用的特征集配置")
        
        if self.verbose:
            print(f"📊 将分析 {len(available_sets)} 个特征集")
            print(f"🔄 使用 {self.max_workers} 个并行进程")
        
        # 并行执行分析
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_feature = {
                executor.submit(self.run_single_feature_analysis, feature_type): feature_type 
                for feature_type in available_sets
            }
            
            # 收集结果
            for future in as_completed(future_to_feature):
                feature_type = future_to_feature[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    status = "✅" if result['status'] == 'success' else "❌"
                    if self.verbose:
                        print(f"{status} {feature_type} 完成")
                        
                except Exception as e:
                    self.logger.error(f"处理 {feature_type} 结果时出错: {e}")
                    results.append({
                        'feature_type': feature_type,
                        'status': 'failed',
                        'error': str(e),
                        'n_features': int(feature_type.replace('best', ''))
                    })
        
        # 整理结果
        self.sweep_results = {r['feature_type']: r for r in results}
        
        # 生成性能总结
        self.performance_summary = self.generate_performance_summary()
        
        return self.sweep_results
    
    def generate_performance_summary(self) -> pd.DataFrame:
        """生成性能总结表"""
        summary_data = []
        
        for feature_type, result in self.sweep_results.items():
            if result['status'] == 'success':
                summary_data.append({
                    'feature_type': feature_type,
                    'n_features': result['n_features'],
                    'source_auc': result.get('source_auc', 0),
                    'source_accuracy': result.get('source_accuracy', 0),
                    'source_f1': result.get('source_f1', 0),
                    'target_baseline_auc': result.get('target_baseline_auc', 0),
                    'target_tca_auc': result.get('target_tca_auc', 0),
                    'target_sa_auc': result.get('target_sa_auc', 0),
                    'target_coral_auc': result.get('target_coral_auc', 0),
                    'target_kmm_auc': result.get('target_kmm_auc', 0),
                    'tca_improvement': result.get('tca_auc_improvement', 0),
                    'sa_improvement': result.get('sa_auc_improvement', 0),
                    'coral_improvement': result.get('coral_auc_improvement', 0),
                    'kmm_improvement': result.get('kmm_auc_improvement', 0)
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data).sort_values('n_features')
            return df
        else:
            return pd.DataFrame()
    
    def plot_performance_comparison(self) -> str:
        """综合性能比较图表生成器 - 生成所有UDA方法的个别图表和汇总图表"""
        if self.verbose:
            print(f"\n📊 开始生成综合性能可视化分析")

        # 准备数据
        df = self.performance_summary

        if df.empty:
            self.logger.warning("没有成功的结果可用于可视化")
            return ""

        generated_plots = []

        # 1. 生成每个UDA方法的单独性能分析图表
        uda_methods = ['TCA', 'SA', 'CORAL', 'KMM']

        if self.verbose:
            print("\n📈 生成个别UDA方法性能分析图表:")

        for method in uda_methods:
            try:
                individual_plot_path = self.plot_individual_uda_performance(method)
                if individual_plot_path:
                    generated_plots.append(individual_plot_path)
                    if self.verbose:
                        print(f"  ✅ {method}方法图表: {Path(individual_plot_path).name}")
            except Exception as e:
                self.logger.warning(f"生成{method}个别图表时出错: {e}")

        # 2. 生成综合汇总性能图表
        if self.verbose:
            print("\n📊 生成综合汇总性能图表:")

        try:
            summary_plot_path = self.plot_comprehensive_summary()
            if summary_plot_path:
                generated_plots.append(summary_plot_path)
                if self.verbose:
                    print(f"  ✅ 综合汇总图表: {Path(summary_plot_path).name}")
        except Exception as e:
            self.logger.warning(f"生成综合汇总图表时出错: {e}")

        # 3. 生成UDA方法排名分析图表
        if self.verbose:
            print("\n🏆 生成UDA方法排名分析图表:")

        try:
            ranking_plot_path = self.plot_uda_ranking_analysis()
            if ranking_plot_path:
                generated_plots.append(ranking_plot_path)
                if self.verbose:
                    print(f"  ✅ 排名分析图表: {Path(ranking_plot_path).name}")
        except Exception as e:
            self.logger.warning(f"生成排名分析图表时出错: {e}")

        # 4. 生成原始版本的性能比较图表（保持向后兼容性）
        if self.verbose:
            print("\n📋 生成原始性能比较图表（向后兼容）:")

        try:
            original_plot_path = self._plot_original_performance_comparison()
            if original_plot_path:
                generated_plots.append(original_plot_path)
                if self.verbose:
                    print(f"  ✅ 原始比较图表: {Path(original_plot_path).name}")
        except Exception as e:
            self.logger.warning(f"生成原始比较图表时出错: {e}")

        # 生成性能分析报告
        if self.verbose:
            print("\n📝 性能分析总结:")
            best_methods = self.identify_best_method_per_feature_set()
            if best_methods:
                print("  各特征集最佳UDA方法:")
                for feature_set, method_info in best_methods.items():
                    print(f"    {feature_set}: {method_info['method']} (AUC: {method_info['auc']:.3f})")

        # 返回主要的综合汇总图表路径
        if generated_plots:
            main_plot = next((path for path in generated_plots if "comprehensive_summary" in path), generated_plots[0])
            if self.verbose:
                print(f"\n🎯 生成了 {len(generated_plots)} 个图表文件")
                print(f"✅ 主要图表路径: {main_plot}")
            return main_plot
        else:
            self.logger.error("未能生成任何图表")
            return ""

    def _plot_original_performance_comparison(self) -> str:
        """生成原始版本的性能比较图表（保持向后兼容性）"""
        df = self.performance_summary

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Selection Performance Analysis (Original)\n(Best3 ~ Best58)',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. AUC性能曲线 (左上)
        ax1 = axes[0, 0]

        # 绘制源域和目标域AUC
        ax1.plot(df['n_features'], df['source_auc'], 'b-o',
                label='Source Domain (10-fold CV)', linewidth=2, markersize=5)
        ax1.plot(df['n_features'], df['target_baseline_auc'], 'r-s',
                label='Target Baseline (No UDA)', linewidth=2, markersize=4)
        ax1.plot(df['n_features'], df['target_tca_auc'], 'g-^',
                label='Target TCA', linewidth=2, markersize=4)
        ax1.plot(df['n_features'], df['target_sa_auc'], 'm-v',
                label='Target SA', linewidth=2, markersize=4)
        ax1.plot(df['n_features'], df['target_coral_auc'], 'c-<',
                label='Target CORAL', linewidth=2, markersize=4)
        ax1.plot(df['n_features'], df['target_kmm_auc'], 'y->',
                label='Target KMM', linewidth=2, markersize=4)

        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('AUC Performance vs Number of Features')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.6, 1.0)

        # 2. UDA方法提升效果 (右上)
        ax2 = axes[0, 1]

        # 绘制所有UDA方法相对于基线的提升
        x = df['n_features']
        width = 0.2  # 柱状图宽度
        x_pos = np.arange(len(x))

        ax2.bar(x_pos - 1.5*width, df['tca_improvement'], width, label='TCA', alpha=0.8)
        ax2.bar(x_pos - 0.5*width, df['sa_improvement'], width, label='SA', alpha=0.8)
        ax2.bar(x_pos + 0.5*width, df['coral_improvement'], width, label='CORAL', alpha=0.8)
        ax2.bar(x_pos + 1.5*width, df['kmm_improvement'], width, label='KMM', alpha=0.8)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('AUC Improvement vs Baseline')
        ax2.set_title('UDA Methods Domain Adaptation Improvement')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x.astype(int), rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 准确率对比 (左下)
        ax3 = axes[1, 0]

        ax3.plot(df['n_features'], df['source_accuracy'], 'b-o',
                label='Source Domain', linewidth=2, markersize=5)
        ax3.plot(df['n_features'], df['target_baseline_auc'], 'r-s',  # 使用AUC作为参考
                label='Target Baseline (AUC)', linewidth=2, markersize=4)

        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Score')
        ax3.set_title('Accuracy Performance Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 性能热力图 (右下)
        ax4 = axes[1, 1]

        # 准备热力图数据
        heatmap_data = df[['source_auc', 'target_baseline_auc', 'target_tca_auc',
                          'target_sa_auc', 'target_coral_auc', 'target_kmm_auc']].T
        heatmap_data.columns = df['feature_type'].values

        # 只显示部分特征集以避免过度拥挤
        step = max(1, len(heatmap_data.columns) // 15)  # 最多显示15个标签
        selected_cols = heatmap_data.columns[::step]
        heatmap_subset = heatmap_data[selected_cols]

        sns.heatmap(heatmap_subset, annot=True, cmap='RdYlGn', center=0.8,
                   fmt='.3f', cbar_kws={'label': 'AUC Score'}, ax=ax4)
        ax4.set_title('Performance Heatmap (Selected Features)')
        ax4.set_ylabel('Method')
        ax4.set_xlabel('Feature Set')

        plt.tight_layout()

        # 保存图表
        plot_path = self.output_dir / "performance_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(plot_path)

    def plot_individual_uda_performance(self, method_name: str) -> str:
        """绘制单个UDA方法的详细性能分析图表"""
        if self.verbose:
            print(f"\n📊 生成{method_name}方法详细性能分析")

        # 准备数据
        df = self.performance_summary

        if df.empty:
            self.logger.warning("没有成功的结果可用于可视化")
            return ""

        # 创建2x2子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{method_name} Method Performance Analysis\n(Feature Selection Sweep)',
                     fontsize=16, fontweight='bold', y=0.98)

        # 数据列名映射
        method_mapping = {
            'TCA': ('target_tca_auc', 'tca_improvement'),
            'SA': ('target_sa_auc', 'sa_improvement'),
            'CORAL': ('target_coral_auc', 'coral_improvement'),
            'KMM': ('target_kmm_auc', 'kmm_improvement')
        }

        if method_name not in method_mapping:
            self.logger.error(f"未知的UDA方法: {method_name}")
            return ""

        auc_col, improvement_col = method_mapping[method_name]

        # 1. AUC性能对比 (左上)
        ax1 = axes[0, 0]
        ax1.plot(df['n_features'], df['source_auc'], 'b-o',
                label='Source Domain (10-fold CV)', linewidth=2, markersize=5)
        ax1.plot(df['n_features'], df['target_baseline_auc'], 'r-s',
                label='Target Baseline (No UDA)', linewidth=2, markersize=4)
        ax1.plot(df['n_features'], df[auc_col], 'g-^',
                label=f'Target {method_name}', linewidth=2, markersize=4)

        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('AUC Score')
        ax1.set_title(f'{method_name} vs Baseline AUC Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.6, 1.0)

        # 2. 提升效果分析 (右上)
        ax2 = axes[0, 1]
        improvement_data = df[improvement_col].values
        colors = ['green' if x > 0 else 'red' for x in improvement_data]

        bars = ax2.bar(df['n_features'], improvement_data,
                      color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('AUC Improvement vs Baseline')
        ax2.set_title(f'{method_name} Domain Adaptation Improvement')
        ax2.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, improvement in zip(bars, improvement_data):
            if abs(improvement) > 0.001:  # 只显示有意义的改进
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                        f'{improvement:.3f}', ha='center', va='bottom', fontsize=8)

        # 3. 性能稳定性分析 (左下)
        ax3 = axes[1, 0]

        # 计算性能变化率
        auc_values = df[auc_col].values
        performance_std = np.std(auc_values)
        performance_mean = np.mean(auc_values)
        cv = performance_std / performance_mean if performance_mean > 0 else 0

        ax3.plot(df['n_features'], df[auc_col], 'g-o', linewidth=2, markersize=5)
        ax3.axhline(y=performance_mean, color='orange', linestyle='--',
                   label=f'Mean AUC: {performance_mean:.3f}')
        ax3.fill_between(df['n_features'],
                        performance_mean - performance_std,
                        performance_mean + performance_std,
                        alpha=0.2, color='orange',
                        label=f'±1 STD: {performance_std:.3f}')

        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('AUC Score')
        ax3.set_title(f'{method_name} Performance Stability\n(CV: {cv:.3f})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 最佳特征数识别 (右下)
        ax4 = axes[1, 1]

        # 找到最佳性能的特征数
        best_idx = np.argmax(df[auc_col].values)
        best_n_features = df['n_features'].iloc[best_idx]
        best_auc = df[auc_col].iloc[best_idx]
        best_improvement = df[improvement_col].iloc[best_idx]

        # 绘制性能曲线并高亮最佳点
        ax4.plot(df['n_features'], df[auc_col], 'g-o', linewidth=2, markersize=4)
        ax4.scatter([best_n_features], [best_auc], color='red', s=100, zorder=5,
                   label=f'Best: {best_n_features} features')

        # 添加最佳点信息
        ax4.annotate(f'Best Performance\n{best_n_features} features\nAUC: {best_auc:.3f}\nImprovement: +{best_improvement:.3f}',
                    xy=(best_n_features, best_auc), xytext=(10, 10),
                    textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('AUC Score')
        ax4.set_title(f'{method_name} Optimal Feature Selection')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图表
        plot_path = self.output_dir / f"performance_comparison_{method_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✅ {method_name}详细性能图表已保存: {plot_path}")

        return str(plot_path)

    def identify_best_method_per_feature_set(self) -> Dict:
        """识别每个特征集下的最佳UDA方法"""
        df = self.performance_summary

        if df.empty:
            return {}

        best_methods = {}
        uda_methods = ['TCA', 'SA', 'CORAL', 'KMM']
        uda_columns = ['target_tca_auc', 'target_sa_auc', 'target_coral_auc', 'target_kmm_auc']

        for _, row in df.iterrows():
            feature_type = row['feature_type']
            n_features = row['n_features']

            # 获取各方法的AUC
            aucs = [row[col] for col in uda_columns]
            best_idx = np.argmax(aucs)
            best_method = uda_methods[best_idx]
            best_auc = aucs[best_idx]

            best_methods[feature_type] = {
                'method': best_method,
                'auc': best_auc,
                'n_features': n_features,
                'improvement': row[f'{best_method.lower()}_improvement']
            }

        return best_methods

    def plot_comprehensive_summary(self) -> str:
        """绘制增强的综合汇总性能图表"""
        if self.verbose:
            print(f"\n📊 生成增强综合汇总性能图表")

        # 准备数据
        df = self.performance_summary

        if df.empty:
            self.logger.warning("没有成功的结果可用于可视化")
            return ""

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive UDA Methods Performance Analysis\n(Feature Selection Sweep)',
                     fontsize=16, fontweight='bold', y=0.98)

        # 1. 所有UDA方法AUC性能曲线对比 (左上)
        ax1 = axes[0, 0]

        # 绘制源域和目标域AUC
        ax1.plot(df['n_features'], df['source_auc'], 'b-o',
                label='Source Domain (10-fold CV)', linewidth=2, markersize=5)
        ax1.plot(df['n_features'], df['target_baseline_auc'], 'r-s',
                label='Target Baseline (No UDA)', linewidth=2, markersize=4)
        ax1.plot(df['n_features'], df['target_tca_auc'], 'g-^',
                label='Target TCA', linewidth=2, markersize=4)
        ax1.plot(df['n_features'], df['target_sa_auc'], 'm-v',
                label='Target SA', linewidth=2, markersize=4)
        ax1.plot(df['n_features'], df['target_coral_auc'], 'c-<',
                label='Target CORAL', linewidth=2, markersize=4)
        ax1.plot(df['n_features'], df['target_kmm_auc'], 'y->',
                label='Target KMM', linewidth=2, markersize=4)

        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('All UDA Methods AUC Performance Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.6, 1.0)

        # 2. 所有方法提升效果并排对比 (右上)
        ax2 = axes[0, 1]

        # 绘制所有UDA方法相对于基线的提升
        x = df['n_features']
        width = 0.2  # 柱状图宽度
        x_pos = np.arange(len(x))

        ax2.bar(x_pos - 1.5*width, df['tca_improvement'], width, label='TCA', alpha=0.8)
        ax2.bar(x_pos - 0.5*width, df['sa_improvement'], width, label='SA', alpha=0.8)
        ax2.bar(x_pos + 0.5*width, df['coral_improvement'], width, label='CORAL', alpha=0.8)
        ax2.bar(x_pos + 1.5*width, df['kmm_improvement'], width, label='KMM', alpha=0.8)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('AUC Improvement vs Baseline')
        ax2.set_title('UDA Methods Improvement Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x.astype(int), rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 最佳方法识别图 (左下)
        ax3 = axes[1, 0]

        # 获取每个特征数下的最佳方法
        best_methods = self.identify_best_method_per_feature_set()
        uda_methods = ['TCA', 'SA', 'CORAL', 'KMM']
        method_colors = {'TCA': 'green', 'SA': 'magenta', 'CORAL': 'cyan', 'KMM': 'orange'}

        # 创建最佳方法的散点图
        for feature_type, info in best_methods.items():
            method = info['method']
            n_features = info['n_features']
            auc = info['auc']

            # 检查是否已经存在该方法的图例
            existing_labels = [item.get_text() for item in ax3.get_legend_handles_labels()[1]] if ax3.get_legend() else []
            label = method if method not in existing_labels else ""
            ax3.scatter([n_features], [auc], color=method_colors[method],
                       s=100, alpha=0.7, label=label)

        # 绘制连接线显示最佳方法的变化
        sorted_features = sorted([(info['n_features'], info['method'], info['auc']) for info in best_methods.values()])
        prev_method = None
        for i, (n_feat, method, auc) in enumerate(sorted_features):
            if i > 0:
                prev_n_feat, prev_method, prev_auc = sorted_features[i-1]
                ax3.plot([prev_n_feat, n_feat], [prev_auc, auc], 'k--', alpha=0.3, linewidth=1)

        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('Best AUC Score')
        ax3.set_title('Best Performing UDA Method per Feature Set')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 方法稳定性分析 (右下)
        ax4 = axes[1, 1]

        # 计算各方法在不同特征数下的方差
        uda_columns = ['target_tca_auc', 'target_sa_auc', 'target_coral_auc', 'target_kmm_auc']
        method_names = ['TCA', 'SA', 'CORAL', 'KMM']

        stability_data = []
        for i, col in enumerate(uda_columns):
            values = df[col].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            cv = std_val / mean_val if mean_val > 0 else 0
            stability_data.append({
                'method': method_names[i],
                'mean': mean_val,
                'std': std_val,
                'cv': cv
            })

        # 绘制稳定性柱状图
        methods = [data['method'] for data in stability_data]
        cvs = [data['cv'] for data in stability_data]
        means = [data['mean'] for data in stability_data]

        bars = ax4.bar(methods, cvs, alpha=0.7, color=['green', 'magenta', 'cyan', 'orange'])
        ax4.set_xlabel('UDA Methods')
        ax4.set_ylabel('Coefficient of Variation (CV)')
        ax4.set_title('Method Performance Stability\n(Lower CV = More Stable)')
        ax4.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, cv, mean in zip(bars, cvs, means):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                    f'CV: {cv:.3f}\nμ: {mean:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()

        # 保存图表
        plot_path = self.output_dir / "comprehensive_performance_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✅ 增强综合汇总性能图表已保存: {plot_path}")

        return str(plot_path)

    def plot_uda_ranking_analysis(self) -> str:
        """绘制UDA方法排名分析图表"""
        if self.verbose:
            print(f"\n📊 生成UDA方法排名分析图表")

        # 准备数据
        df = self.performance_summary

        if df.empty:
            self.logger.warning("没有成功的结果可用于可视化")
            return ""

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('UDA Methods Ranking Analysis\n(Feature Selection Sweep)',
                     fontsize=16, fontweight='bold', y=0.95)

        # 1. 各UDA方法的排名变化趋势 (左)
        uda_columns = ['target_tca_auc', 'target_sa_auc', 'target_coral_auc', 'target_kmm_auc']
        method_names = ['TCA', 'SA', 'CORAL', 'KMM']
        colors = ['green', 'magenta', 'cyan', 'orange']

        rankings = []
        for _, row in df.iterrows():
            aucs = [row[col] for col in uda_columns]
            # 计算排名 (1=最好, 4=最差)
            rank_indices = np.argsort(aucs)[::-1]  # 降序排列的索引
            ranks = [0] * len(aucs)
            for rank, idx in enumerate(rank_indices):
                ranks[idx] = rank + 1
            rankings.append(ranks)

        rankings = np.array(rankings).T  # 转置使每行代表一个方法

        # 绘制排名变化趋势
        for i, (method, color) in enumerate(zip(method_names, colors)):
            ax1.plot(df['n_features'], rankings[i], 'o-', color=color,
                    label=method, linewidth=2, markersize=6)

        ax1.set_xlabel('Number of Features')
        ax1.set_ylabel('Ranking (1=Best, 4=Worst)')
        ax1.set_title('UDA Methods Ranking Trends')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.5, 4.5)
        ax1.invert_yaxis()  # 倒转y轴，使1在顶部

        # 2. 胜率统计 (右)
        # 计算各方法在多少个特征集上表现最佳
        best_methods = self.identify_best_method_per_feature_set()
        win_counts = {method: 0 for method in method_names}

        for info in best_methods.values():
            win_counts[info['method']] += 1

        total_sets = len(best_methods)
        win_rates = [win_counts[method] / total_sets * 100 for method in method_names]

        bars = ax2.bar(method_names, win_rates, alpha=0.7, color=colors)
        ax2.set_xlabel('UDA Methods')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('UDA Methods Win Rate\n(% of Feature Sets Where Method Performs Best)')
        ax2.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, rate, count in zip(bars, win_rates, [win_counts[method] for method in method_names]):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{rate:.1f}%\n({count}/{total_sets})', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        # 保存图表
        plot_path = self.output_dir / "uda_methods_ranking.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        if self.verbose:
            print(f"✅ UDA方法排名分析图表已保存: {plot_path}")

        return str(plot_path)

    def generate_summary_report(self) -> str:
        """生成总结报告"""
        if self.verbose:
            print(f"\n📋 生成总结报告")
        
        report_content = []
        report_content.append("# Feature Sweep Analysis Report\n")
        report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 分析配置
        report_content.append("## Analysis Configuration\n")
        report_content.append(f"- Feature Range: best{self.feature_range[0]} ~ best{self.feature_range[1]}")
        report_content.append(f"- Parallel Workers: {self.max_workers}")
        report_content.append(f"- Total Feature Sets: {len(self.sweep_results)}")
        
        # 成功率统计
        successful = sum(1 for r in self.sweep_results.values() if r['status'] == 'success')
        failed = len(self.sweep_results) - successful
        report_content.append(f"- Successful Analyses: {successful}")
        report_content.append(f"- Failed Analyses: {failed}\n")
        
        # 性能总结
        if not self.performance_summary.empty:
            report_content.append("## Performance Summary\n")
            
            # 最佳性能
            best_source = self.performance_summary.loc[self.performance_summary['source_auc'].idxmax()]
            best_target_baseline = self.performance_summary.loc[self.performance_summary['target_baseline_auc'].idxmax()]
            best_target_tca = self.performance_summary.loc[self.performance_summary['target_tca_auc'].idxmax()]
            best_improvement = self.performance_summary.loc[self.performance_summary['tca_improvement'].idxmax()]
            
            report_content.append(f"### Best Performance Results\n")
            report_content.append(f"- **Best Source Domain**: {best_source['feature_type']} (AUC: {best_source['source_auc']:.4f})")
            report_content.append(f"- **Best Target Baseline**: {best_target_baseline['feature_type']} (AUC: {best_target_baseline['target_baseline_auc']:.4f})")
            report_content.append(f"- **Best Target TCA**: {best_target_tca['feature_type']} (AUC: {best_target_tca['target_tca_auc']:.4f})")
            report_content.append(f"- **Best TCA Improvement**: {best_improvement['feature_type']} (+{best_improvement['tca_improvement']:.4f})")
            
            # 性能趋势
            report_content.append(f"\n### Performance Trends\n")
            report_content.append(f"- Average Source AUC: {self.performance_summary['source_auc'].mean():.4f}")
            report_content.append(f"- Average Target Baseline AUC: {self.performance_summary['target_baseline_auc'].mean():.4f}")
            report_content.append(f"- Average Target TCA AUC: {self.performance_summary['target_tca_auc'].mean():.4f}")
            report_content.append(f"- Average TCA Improvement: {self.performance_summary['tca_improvement'].mean():.4f}")
            
            # 详细结果表
            report_content.append(f"\n### Detailed Results\n")
            report_content.append("| Feature Set | N Features | Source AUC | Target Baseline | Target TCA | TCA Improvement |")
            report_content.append("|-------------|------------|------------|-----------------|------------|-----------------|")
            
            for _, row in self.performance_summary.iterrows():
                report_content.append(f"| {row['feature_type']} | {row['n_features']} | {row['source_auc']:.4f} | {row['target_baseline_auc']:.4f} | {row['target_tca_auc']:.4f} | {row['tca_improvement']:+.4f} |")
        
        # 失败的分析
        failed_analyses = {k: v for k, v in self.sweep_results.items() if v['status'] == 'failed'}
        if failed_analyses:
            report_content.append(f"\n### Failed Analyses\n")
            for feature_type, result in failed_analyses.items():
                report_content.append(f"- **{feature_type}**: {result.get('error', 'Unknown error')}")
        
        # 建议
        if not self.performance_summary.empty:
            report_content.append(f"\n## Recommendations\n")
            
            # 基于结果的建议
            best_overall = self.performance_summary.loc[
                (self.performance_summary['target_tca_auc'] + self.performance_summary['source_auc']).idxmax()
            ]
            
            report_content.append(f"- **Recommended Feature Set**: {best_overall['feature_type']}")
            report_content.append(f"  - Source Performance: {best_overall['source_auc']:.4f} AUC")
            report_content.append(f"  - Target Performance: {best_overall['target_tca_auc']:.4f} AUC")
            report_content.append(f"  - TCA Improvement: {best_overall['tca_improvement']:+.4f}")
            
            # 平衡性建议
            if best_improvement['tca_improvement'] > 0.01:
                report_content.append(f"- **TCA Domain Adaptation** shows positive effects for most feature sets")
            else:
                report_content.append(f"- **Limited TCA Benefits** - consider alternative UDA methods")
        
        report_content.append(f"\nDetailed results and visualizations available in: {self.output_dir}")
        
        # 保存报告
        report_file = self.output_dir / "feature_sweep_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        if self.verbose:
            print(f"✅ 总结报告已保存: {report_file}")
        
        return str(report_file)
    
    def save_results(self):
        """保存所有结果"""
        # 保存原始结果
        results_file = self.output_dir / "sweep_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.sweep_results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存性能总结
        if not self.performance_summary.empty:
            summary_file = self.output_dir / "performance_summary.csv"
            self.performance_summary.to_csv(summary_file, index=False)
            
            summary_json = self.output_dir / "performance_summary.json"
            self.performance_summary.to_json(summary_json, orient='records', indent=2)
        
        if self.verbose:
            print(f"✅ 结果已保存到: {self.output_dir}")
    
    def run_complete_feature_sweep(self) -> Dict:
        """运行完整的特征扫描分析"""
        try:
            if self.verbose:
                print(f"🚀 开始完整特征扫描分析")
                print("=" * 60)
            
            # 1. 并行运行特征扫描
            sweep_results = self.run_parallel_feature_sweep()
            
            # 2. 生成可视化
            plot_path = self.plot_performance_comparison()
            
            # 3. 生成报告
            report_path = self.generate_summary_report()
            
            # 4. 保存结果
            self.save_results()
            
            if self.verbose:
                print(f"\n✅ 特征扫描分析完成！")
                print(f"📁 结果目录: {self.output_dir}")
                print(f"📊 性能图表: {plot_path}")
                print(f"📋 分析报告: {report_path}")
            
            return {
                'status': 'success',
                'results': sweep_results,
                'summary': self.performance_summary.to_dict('records') if not self.performance_summary.empty else [],
                'output_dir': str(self.output_dir),
                'plot_path': plot_path,
                'report_path': report_path
            }
            
        except Exception as e:
            self.logger.error(f"完整特征扫描分析失败: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            
            return {
                'status': 'failed',
                'error': str(e),
                'output_dir': str(self.output_dir)
            }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='特征扫描分析 - 从best3到best58')
    parser.add_argument('--min_features', type=int, default=3,
                        help='最小特征数 (default: 3)')
    parser.add_argument('--max_features', type=int, default=58,
                        help='最大特征数 (default: 58)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='最大并行工作数 (default: auto)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录 (default: auto-generated)')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='静默模式')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🔬 TabPFN医疗数据特征扫描分析")
        print("=" * 60)
        print(f"📋 分析配置:")
        print(f"   特征范围: best{args.min_features} ~ best{args.max_features}")
        print(f"   并行工作数: {args.max_workers or 'auto'}")
        print(f"   输出目录: {args.output_dir or 'auto-generated'}")
        print("=" * 60)
    
    # 创建分析器
    analyzer = FeatureSweepAnalyzer(
        feature_range=(args.min_features, args.max_features),
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        verbose=not args.quiet
    )
    
    # 运行完整分析
    results = analyzer.run_complete_feature_sweep()
    
    if results['status'] == 'success':
        if not args.quiet:
            print(f"\n🎉 分析成功完成！")
            print(f"📁 查看结果: {results['output_dir']}")
            print(f"📊 性能图表: {results['plot_path']}")
            print(f"📋 分析报告: {results['report_path']}")
        exit(0)
    else:
        if not args.quiet:
            print(f"\n❌ 分析失败: {results['error']}")
        exit(1)


if __name__ == "__main__":
    main()