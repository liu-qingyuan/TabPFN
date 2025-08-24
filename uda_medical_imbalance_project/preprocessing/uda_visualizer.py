"""
UDA Medical Imbalance Project - UDA结果可视化器

实现域适应前后的可视化对比分析，包括：
1. 降维可视化 (PCA, t-SNE)
2. 特征分布对比
3. 域距离度量 (KL散度, Wasserstein距离, MMD)
4. 性能对比图表

作者: UDA Medical Team
日期: 2025-06-27
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 科学计算和可视化
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from scipy.stats import entropy

# Wasserstein距离导入（兼容不同scipy版本）
try:
    from scipy.spatial.distance import wasserstein_distance
except ImportError:
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        # 如果都不可用，使用简化实现
        def wasserstein_distance(u_values: np.ndarray, v_values: np.ndarray) -> float:
            """简化的Wasserstein距离实现"""
            return np.mean(np.abs(np.sort(u_values) - np.sort(v_values)))

# 导入项目模块
try:
    from uda.adapt_methods import AdaptUDAMethod
    UDA_AVAILABLE = True
except ImportError:
    UDA_AVAILABLE = False
    # 创建占位符类型
    class AdaptUDAMethod:
        pass

import logging
logger = logging.getLogger(__name__)

# 设置中文字体和样式
# 设置Nature期刊标准字体
plt.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'text.usetex': False,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'patch.linewidth': 0.8
})
sns.set_style("whitegrid")
sns.set_palette("husl")

# 添加ADAPT库的专业指标导入
try:
    from adapt.metrics import (
        cov_distance,
        neg_j_score, 
        linear_discrepancy,
        normalized_linear_discrepancy,
        frechet_distance,
        normalized_frechet_distance
    )
    ADAPT_AVAILABLE = True
    print("✅ ADAPT库指标导入成功")
except ImportError as e:
    print(f"⚠️ ADAPT库导入失败: {e}")
    print("📝 将使用备用指标计算方法")
    ADAPT_AVAILABLE = False

class UDAVisualizer:
    """UDA结果可视化器"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), 
                 save_plots: bool = True, 
                 output_dir: str = "results/uda_visualization"):
        """
        初始化可视化器
        
        参数:
        - figsize: 图片大小
        - save_plots: 是否保存图片
        - output_dir: 输出目录
        """
        super().__init__()
        self.figsize = figsize
        self.save_plots = save_plots
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nature期刊科研配色方案
        self.colors = {
            'source': '#2C91E0',      # 科研三色配色1 - 源域
            'target': '#3ABF99',      # 科研三色配色2 - 目标域
            'adapted': '#F0A73A',     # 科研三色配色3 - 适应后
            'class_0': '#9BDCFC',     # 科研双色配色1 - 类别0
            'class_1': '#F0CFEA'      # 科研双色配色2 - 类别1
        }
        
        logger.info(f"UDA可视化器初始化完成，输出目录: {self.output_dir}")
    
    def visualize_domain_adaptation_complete(self, 
                                           X_source: np.ndarray, 
                                           y_source: np.ndarray,
                                           X_target: np.ndarray, 
                                           y_target: np.ndarray,
                                           uda_method: Optional[AdaptUDAMethod] = None,
                                           method_name: str = "UDA") -> Dict[str, Any]:
        """
        完整的域适应可视化分析
        
        参数:
        - X_source, y_source: 源域数据
        - X_target, y_target: 目标域数据  
        - uda_method: 训练好的UDA方法
        - method_name: 方法名称
        
        返回:
        - 可视化结果字典
        """
        results = {}
        
        print(f"\n=== {method_name} 域适应可视化分析 ===")
        
        # 1. 降维可视化
        print("1. 生成降维可视化...")
        pca_results = self.plot_dimensionality_reduction(
            X_source, y_source, X_target, y_target, 
            uda_method, method_name
        )
        results['dimensionality_reduction'] = pca_results
        
        # 2. 特征分布对比
        print("2. 分析特征分布...")
        dist_results = self.plot_feature_distributions(
            X_source, X_target, uda_method, method_name
        )
        results['feature_distributions'] = dist_results
        
        # 3. 域距离度量
        print("3. 计算域距离度量...")
        distance_results = self.calculate_domain_distances(
            X_source, X_target, uda_method, method_name
        )
        results['domain_distances'] = distance_results
        
        # 4. 性能对比
        if uda_method is not None:
            print("4. 生成性能对比...")
            perf_results = self.plot_performance_comparison(
                X_source, y_source, X_target, y_target,
                uda_method, method_name
            )
            results['performance'] = perf_results
        
        print(f"✓ {method_name} 可视化分析完成")
        return results
    
    def plot_dimensionality_reduction(self, 
                                    X_source: np.ndarray, 
                                    y_source: np.ndarray,
                                    X_target: np.ndarray, 
                                    y_target: np.ndarray,
                                    uda_method: Optional[AdaptUDAMethod] = None,
                                    method_name: str = "UDA") -> Dict[str, Any]:
        """降维可视化 (PCA + t-SNE)"""
        
        # 数据准备 - 获取适应后的特征
        X_source_adapted, X_target_adapted = self._get_adapted_features(
            X_source, X_target, uda_method, method_name
        )
        
        # 直接使用原始特征，不进行标准化
        print("  📝 直接使用原始特征进行可视化，不进行标准化")
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        # 移除主标题以符合Nature期刊要求
        # fig.suptitle(f'{method_name} - Domain Adaptation Visualization', fontsize=16, fontweight='bold')
        
        results = {}
        
        # PCA可视化
        self._plot_pca_comparison(
            X_source, y_source, X_target, y_target,
            X_source_adapted, X_target_adapted,
            axes[0, :], method_name
        )
        
        # t-SNE可视化
        tsne_results = self._plot_tsne_comparison(
            X_source, y_source, X_target, y_target,
            X_source_adapted, X_target_adapted,
            axes[1, :], method_name
        )
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.output_dir / f"{method_name}_dimensionality_reduction.pdf"
            plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"  降维可视化已保存: {save_path}")
        
        plt.show()
        
        results.update(tsne_results)
        return results
    
    def _plot_pca_comparison(self, X_source: np.ndarray, y_source: np.ndarray, 
                           X_target: np.ndarray, y_target: np.ndarray,
                           X_source_adapted: np.ndarray, X_target_adapted: np.ndarray, 
                           axes: np.ndarray, method_name: str) -> None:
        """PCA对比图"""
        
        # 合并数据进行PCA
        X_combined_before = np.vstack([X_source, X_target])
        X_combined_after = np.vstack([X_source_adapted, X_target_adapted])
        
        # PCA降维
        pca = PCA(n_components=2)
        X_pca_before = pca.fit_transform(X_combined_before)
        X_pca_after = pca.fit_transform(X_combined_after)
        
        n_source = len(X_source)
        
        # 域适应前
        ax1 = axes[0]
        ax1.scatter(X_pca_before[:n_source, 0], X_pca_before[:n_source, 1], 
                   c=self.colors['source'], alpha=0.6, label='Source Domain', s=50)
        ax1.scatter(X_pca_before[n_source:, 0], X_pca_before[n_source:, 1], 
                   c=self.colors['target'], alpha=0.6, label='Target Domain', s=50)
        ax1.set_title('a', fontweight='bold', fontsize=24, pad=20, loc='left')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 域适应后
        ax2 = axes[1]
        ax2.scatter(X_pca_after[:n_source, 0], X_pca_after[:n_source, 1], 
                   c=self.colors['source'], alpha=0.6, label='Source Domain', s=50)
        ax2.scatter(X_pca_after[n_source:, 0], X_pca_after[n_source:, 1], 
                   c=self.colors['adapted'], alpha=0.6, label='Target Domain (Adapted)', s=50)
        ax2.set_title('b', fontweight='bold', fontsize=24, pad=20, loc='left')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    def _plot_tsne_comparison(self, X_source: np.ndarray, y_source: np.ndarray, 
                            X_target: np.ndarray, y_target: np.ndarray,
                            X_source_adapted: np.ndarray, X_target_adapted: np.ndarray, 
                            axes: np.ndarray, method_name: str) -> Dict[str, Any]:
        """t-SNE对比图"""
        
        # 为了计算效率，限制样本数量
        max_samples = 500
        if len(X_source) > max_samples:
            idx_source = np.random.choice(len(X_source), max_samples, replace=False)
            X_source = X_source[idx_source]
            y_source = y_source[idx_source]
            X_source_adapted = X_source_adapted[idx_source]
        
        if len(X_target) > max_samples:
            idx_target = np.random.choice(len(X_target), max_samples, replace=False)
            X_target = X_target[idx_target]
            y_target = y_target[idx_target]
            X_target_adapted = X_target_adapted[idx_target]
        
        # 合并数据进行t-SNE
        X_combined_before = np.vstack([X_source, X_target])
        X_combined_after = np.vstack([X_source_adapted, X_target_adapted])
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne_before = tsne.fit_transform(X_combined_before)
        
        tsne_after = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne_after = tsne_after.fit_transform(X_combined_after)
        
        n_source = len(X_source)
        
        # 域适应前
        ax1 = axes[0]
        ax1.scatter(X_tsne_before[:n_source, 0], X_tsne_before[:n_source, 1], 
                   c=self.colors['source'], alpha=0.6, label='Source Domain', s=50)
        ax1.scatter(X_tsne_before[n_source:, 0], X_tsne_before[n_source:, 1], 
                   c=self.colors['target'], alpha=0.6, label='Target Domain', s=50)
        ax1.set_title('c', fontweight='bold', fontsize=24, pad=20, loc='left')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 域适应后
        ax2 = axes[1]
        ax2.scatter(X_tsne_after[:n_source, 0], X_tsne_after[:n_source, 1], 
                   c=self.colors['source'], alpha=0.6, label='Source Domain', s=50)
        ax2.scatter(X_tsne_after[n_source:, 0], X_tsne_after[n_source:, 1], 
                   c=self.colors['adapted'], alpha=0.6, label='Target Domain (Adapted)', s=50)
        ax2.set_title('d', fontweight='bold', fontsize=24, pad=20, loc='left')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        return {
            'tsne_before': X_tsne_before,
            'tsne_after': X_tsne_after,
            'n_source_samples': n_source
        }
    
    def plot_feature_distributions(self, 
                                 X_source: np.ndarray, 
                                 X_target: np.ndarray,
                                 uda_method: Optional[AdaptUDAMethod] = None,
                                 method_name: str = "UDA") -> Dict[str, Any]:
        """特征分布对比可视化 - 由于特征尺度差异大，跳过可视化"""
        
        print(f"  📝 跳过特征分布可视化（特征尺度差异过大，不适合直接可视化）")
        
        # 获取适应后的特征用于返回
        X_source_adapted, X_target_adapted = self._get_adapted_features(
            X_source, X_target, uda_method, method_name
        )
        
        return {
            'source_original': X_source,
            'target_original': X_target,
            'source_adapted': X_source_adapted,
            'target_adapted': X_target_adapted
        }
    
    def calculate_domain_distances(self, 
                                 X_source: np.ndarray, 
                                 X_target: np.ndarray,
                                 uda_method: Optional[AdaptUDAMethod] = None,
                                 method_name: str = "UDA") -> Dict[str, float]:
        """计算域距离度量 - 使用ADAPT库的专业指标"""
        
        # 获取适应后的特征
        X_source_adapted, X_target_adapted = self._get_adapted_features(
            X_source, X_target, uda_method, method_name
        )
        
        distances = {}
        
        if ADAPT_AVAILABLE:
            # 使用ADAPT库的标准化指标（只保留稳定的标准化版本）
            print(f"  📊 使用ADAPT库标准化指标计算域距离...")
            
            # 1. 标准化线性差异 (Normalized Linear Discrepancy)
            try:
                norm_linear_disc_before = normalized_linear_discrepancy(X_source, X_target)
                distances['normalized_linear_discrepancy_before'] = float(norm_linear_disc_before)
                
                if uda_method is not None:
                    norm_linear_disc_after = normalized_linear_discrepancy(X_source_adapted, X_target_adapted)
                    distances['normalized_linear_discrepancy_after'] = float(norm_linear_disc_after)
                    distances['normalized_linear_discrepancy_improvement'] = float(norm_linear_disc_before - norm_linear_disc_after)
                    
                    print(f"  📊 标准化线性差异 (Normalized Linear Discrepancy):")
                    print(f"    适应前: {norm_linear_disc_before:.6f}")
                    print(f"    适应后: {norm_linear_disc_after:.6f}")
                    print(f"    改进: {norm_linear_disc_before - norm_linear_disc_after:.6f}")
                else:
                    distances['normalized_linear_discrepancy_after'] = distances['normalized_linear_discrepancy_before']
                    distances['normalized_linear_discrepancy_improvement'] = 0.0
            except Exception as e:
                print(f"  ⚠️ 标准化线性差异计算失败: {e}")
            
            # 2. 标准化Frechet距离 (Normalized Frechet Distance)
            try:
                norm_frechet_dist_before = normalized_frechet_distance(X_source, X_target)
                distances['normalized_frechet_distance_before'] = float(norm_frechet_dist_before)
                
                if uda_method is not None:
                    norm_frechet_dist_after = normalized_frechet_distance(X_source_adapted, X_target_adapted)
                    distances['normalized_frechet_distance_after'] = float(norm_frechet_dist_after)
                    distances['normalized_frechet_distance_improvement'] = float(norm_frechet_dist_before - norm_frechet_dist_after)
                    
                    print(f"  📊 标准化Frechet距离 (Normalized Frechet Distance):")
                    print(f"    适应前: {norm_frechet_dist_before:.6f}")
                    print(f"    适应后: {norm_frechet_dist_after:.6f}")
                    print(f"    改进: {norm_frechet_dist_before - norm_frechet_dist_after:.6f}")
                else:
                    distances['normalized_frechet_distance_after'] = distances['normalized_frechet_distance_before']
                    distances['normalized_frechet_distance_improvement'] = 0.0
            except Exception as e:
                print(f"  ⚠️ 标准化Frechet距离计算失败: {e}")
            
            # 3. 标准化Wasserstein距离 (我们自定义的实现)
            try:
                norm_ws_before = self._calculate_normalized_wasserstein_distance(X_source, X_target)
                distances['normalized_wasserstein_before'] = norm_ws_before
                
                if uda_method is not None:
                    norm_ws_after = self._calculate_normalized_wasserstein_distance(X_source_adapted, X_target_adapted)
                    distances['normalized_wasserstein_after'] = norm_ws_after
                    distances['normalized_wasserstein_improvement'] = norm_ws_before - norm_ws_after
                    
                    print(f"  📊 标准化Wasserstein距离 (Normalized Wasserstein Distance):")
                    print(f"    适应前: {norm_ws_before:.6f}")
                    print(f"    适应后: {norm_ws_after:.6f}")
                    print(f"    改进: {norm_ws_before - norm_ws_after:.6f}")
                else:
                    distances['normalized_wasserstein_after'] = norm_ws_before
                    distances['normalized_wasserstein_improvement'] = 0.0
            except Exception as e:
                print(f"  ⚠️ 标准化Wasserstein距离计算失败: {e}")
            
            # 4. 标准化KL散度 (我们自定义的实现)
            try:
                norm_kl_before = self._calculate_normalized_kl_divergence(X_source, X_target)
                distances['normalized_kl_divergence_before'] = norm_kl_before
                
                if uda_method is not None:
                    norm_kl_after = self._calculate_normalized_kl_divergence(X_source_adapted, X_target_adapted)
                    distances['normalized_kl_divergence_after'] = norm_kl_after
                    distances['normalized_kl_divergence_improvement'] = norm_kl_before - norm_kl_after
                    
                    print(f"  📊 标准化KL散度 (Normalized KL Divergence):")
                    print(f"    适应前: {norm_kl_before:.6f}")
                    print(f"    适应后: {norm_kl_after:.6f}")
                    print(f"    改进: {norm_kl_before - norm_kl_after:.6f}")
                else:
                    distances['normalized_kl_divergence_after'] = norm_kl_before
                    distances['normalized_kl_divergence_improvement'] = 0.0
            except Exception as e:
                print(f"  ⚠️ 标准化KL散度计算失败: {e}")
        
        else:
            # 备用方案：使用原有的指标和新增的标准化指标
            print(f"  📊 使用备用指标计算域距离...")
            
            # 1. 标准化KL散度计算 - 仿照ADAPT库实现
            try:
                norm_kl_before = self._calculate_normalized_kl_divergence(X_source, X_target)
                distances['normalized_kl_divergence_before'] = norm_kl_before
                
                if uda_method is not None:
                    norm_kl_after = self._calculate_normalized_kl_divergence(X_source_adapted, X_target_adapted)
                    distances['normalized_kl_divergence_after'] = norm_kl_after
                    distances['normalized_kl_divergence_improvement'] = norm_kl_before - norm_kl_after
                    
                    print(f"  📊 标准化KL散度计算:")
                    print(f"    适应前: {norm_kl_before:.6f}")
                    print(f"    适应后: {norm_kl_after:.6f}")
                    print(f"    改进: {norm_kl_before - norm_kl_after:.6f}")
                else:
                    distances['normalized_kl_divergence_after'] = norm_kl_before
                    distances['normalized_kl_divergence_improvement'] = 0.0
            except Exception as e:
                print(f"  ⚠️ 标准化KL散度计算失败: {e}")
            
            # 2. 标准化Wasserstein距离计算 - 仿照ADAPT库实现
            try:
                norm_ws_before = self._calculate_normalized_wasserstein_distance(X_source, X_target)
                distances['normalized_wasserstein_before'] = norm_ws_before
                
                if uda_method is not None:
                    norm_ws_after = self._calculate_normalized_wasserstein_distance(X_source_adapted, X_target_adapted)
                    distances['normalized_wasserstein_after'] = norm_ws_after
                    distances['normalized_wasserstein_improvement'] = norm_ws_before - norm_ws_after
                    
                    print(f"  📊 标准化Wasserstein距离计算:")
                    print(f"    适应前: {norm_ws_before:.6f}")
                    print(f"    适应后: {norm_ws_after:.6f}")
                    print(f"    改进: {norm_ws_before - norm_ws_after:.6f}")
                else:
                    distances['normalized_wasserstein_after'] = norm_ws_before
                    distances['normalized_wasserstein_improvement'] = 0.0
            except Exception as e:
                print(f"  ⚠️ 标准化Wasserstein距离计算失败: {e}")
            
            # 3. 原始KL散度计算 - 使用analytical_CORAL的实现
            try:
                kl_before, kl_per_feature_before = self._calculate_kl_divergence_improved(X_source, X_target)
                distances['kl_divergence_before'] = kl_before
                
                if uda_method is not None:
                    kl_after, kl_per_feature_after = self._calculate_kl_divergence_improved(X_source_adapted, X_target_adapted)
                    distances['kl_divergence_after'] = kl_after
                    distances['kl_divergence_improvement'] = kl_before - kl_after
                    
                    print(f"  📊 原始KL散度计算:")
                    print(f"    适应前: {kl_before:.6f}")
                    print(f"    适应后: {kl_after:.6f}")
                    print(f"    改进: {kl_before - kl_after:.6f}")
                else:
                    distances['kl_divergence_after'] = kl_before
                    distances['kl_divergence_improvement'] = 0.0
            except Exception as e:
                print(f"  ⚠️ 原始KL散度计算失败: {e}")
        
            # 4. 原始Wasserstein距离计算 - 使用analytical_CORAL的实现
            try:
                ws_before, ws_per_feature_before = self._calculate_wasserstein_distance_improved(X_source, X_target)
                distances['wasserstein_before'] = ws_before
                
                if uda_method is not None:
                    ws_after, ws_per_feature_after = self._calculate_wasserstein_distance_improved(X_source_adapted, X_target_adapted)
                    distances['wasserstein_after'] = ws_after
                    distances['wasserstein_improvement'] = ws_before - ws_after
                    
                    print(f"  📊 原始Wasserstein距离计算:")
                    print(f"    适应前: {ws_before:.6f}")
                    print(f"    适应后: {ws_after:.6f}")
                    print(f"    改进: {ws_before - ws_after:.6f}")
                else:
                    distances['wasserstein_after'] = ws_before
                    distances['wasserstein_improvement'] = 0.0
            except Exception as e:
                print(f"  ⚠️ 原始Wasserstein距离计算失败: {e}")
        
            # 5. MMD计算 - 使用analytical_CORAL的实现
            try:
                mmd_before = self._calculate_mmd_improved(X_source, X_target)
                distances['mmd_before'] = mmd_before
                
                if uda_method is not None:
                    mmd_after = self._calculate_mmd_improved(X_source_adapted, X_target_adapted)
                    distances['mmd_after'] = mmd_after
                    distances['mmd_improvement'] = mmd_before - mmd_after
                    
                    print(f"  📊 MMD计算:")
                    print(f"    适应前: {mmd_before:.6f}")
                    print(f"    适应后: {mmd_after:.6f}")
                    print(f"    改进: {mmd_before - mmd_after:.6f}")
                else:
                    distances['mmd_after'] = mmd_before
                    distances['mmd_improvement'] = 0.0
            except Exception as e:
                print(f"  ⚠️ MMD计算失败: {e}")
        
        # 可视化距离度量
        self._plot_distance_metrics(distances, method_name)
        
        return distances
    
    def _get_adapted_features(self, 
                            X_source: np.ndarray, 
                            X_target: np.ndarray,
                            uda_method: Optional[AdaptUDAMethod] = None,
                            method_name: str = "UDA") -> Tuple[np.ndarray, np.ndarray]:
        """获取域适应后的特征 - 改进版本，正确处理不同UDA方法的特征变换"""
        
        if uda_method is None:
            return X_source, X_target
        
        X_source_adapted = X_source.copy()
        X_target_adapted = X_target.copy()
        
        # 调试信息：记录原始特征
        print(f"  🔍 原始特征统计:")
        print(f"    源域均值: {np.mean(X_source, axis=0)[:3]} ...")
        print(f"    目标域均值: {np.mean(X_target, axis=0)[:3]} ...")
        
        try:
            # 测试预测功能
            try:
                test_pred = uda_method.predict(X_target[:5])
                print(f"  ✓ UDA方法预测功能正常，预测结果: {test_pred}")
            except Exception as e:
                print(f"  ⚠ UDA方法预测功能异常: {e}")
            
            # 获取UDA方法的内部模型
            adapt_model = uda_method.adapt_model
            print(f"  📋 内部模型类型: {type(adapt_model).__name__}")
            print(f"  📋 可用属性: {[attr for attr in dir(adapt_model) if not attr.startswith('_')][:10]}...")
            
            # 改进的特征变换获取方法
            transform_success = False
            
            # 特殊处理：对于不同类型的UDA方法使用不同策略
            if method_name in ['TCA', 'SA', 'FMMD', 'PRED']:
                # 这些方法可能改变特征维度，需要特殊处理
                print(f"  📝 {method_name}方法可能改变特征维度，使用特殊处理")
                
                # 方法1: 尝试获取变换后的特征（用于降维可视化）
                if hasattr(adapt_model, 'transform'):
                    try:
                        if method_name == 'SA':
                            # SA方法需要domain参数
                            X_source_transformed = adapt_model.transform(X_source, domain="src")
                            X_target_transformed = adapt_model.transform(X_target, domain="tgt")
                        else:
                            # 其他方法
                            X_source_transformed = adapt_model.transform(X_source)
                            X_target_transformed = adapt_model.transform(X_target)
                        
                        print(f"  ✓ transform成功: {X_source.shape} -> {X_source_transformed.shape}")
                        
                        # 检查维度是否改变
                        if X_source_transformed.shape[1] != X_source.shape[1]:
                            print(f"  ⚠ 特征维度改变: {X_source.shape[1]} -> {X_source_transformed.shape[1]}")
                            print(f"  📝 使用变换后的高维特征进行距离度量计算")
                            # 使用变换后的特征进行距离度量
                            # 这样可以真正反映域适应的效果
                            X_source_adapted = X_source_transformed
                            X_target_adapted = X_target_transformed
                        else:
                            # 维度未改变，可以直接使用变换后的特征
                            X_source_adapted = X_source_transformed
                            X_target_adapted = X_target_transformed
                        
                        transform_success = True
                        
                    except Exception as e:
                        print(f"  ⚠ transform失败: {e}")
                
                # 方法2: 如果transform失败，尝试获取变换矩阵
                if not transform_success and hasattr(adapt_model, 'A_'):
                    try:
                        A = adapt_model.A_
                        print(f"  🔍 变换矩阵A_形状: {A.shape}")
                        
                        # 检查矩阵维度是否匹配
                        if A.shape[0] == X_source.shape[1]:
                            X_source_adapted = X_source @ A
                            X_target_adapted = X_target @ A
                            transform_success = True
                            print(f"  ✓ 使用变换矩阵A_成功: {A.shape}")
                        else:
                            print(f"  ⚠ 变换矩阵维度不匹配: {A.shape[0]} != {X_source.shape[1]}")
                    except Exception as e:
                        print(f"  ⚠ 变换矩阵A_失败: {e}")
                    
            elif method_name == 'CORAL':
                # CORAL方法：协方差对齐，维度不变
                try:
                    X_source_adapted, X_target_adapted = self._manual_coral_alignment(X_source, X_target)
                    transform_success = True
                    print(f"  ✓ CORAL对齐成功")
                except Exception as e:
                    print(f"  ⚠ CORAL对齐失败: {e}")
            
            elif method_name in ['KMM', 'KLIEP', 'LDM', 'ULSIF', 'RULSIF', 'NNW', 'IWC', 'IWN']:
                # 实例重加权方法：不改变特征，只改变样本权重
                print(f"  📝 {method_name}为实例重加权方法，不改变特征空间")
                # 对于实例重加权方法，我们使用原始特征
                # 这些方法的"适应"体现在模型权重上，而不是特征变换上
                X_source_adapted = X_source
                X_target_adapted = X_target
                transform_success = True
            
            else:
                # 其他方法：尝试通用的transform方法
                if hasattr(adapt_model, 'transform'):
                    try:
                        X_source_adapted = adapt_model.transform(X_source)
                        X_target_adapted = adapt_model.transform(X_target)
                        transform_success = True
                        print(f"  ✓ 通用transform成功: {X_source.shape} -> {X_source_adapted.shape}")
                    except Exception as e:
                        print(f"  ⚠ 通用transform失败: {e}")
            
            # 检查变换是否有效
            if transform_success:
                # 检查维度是否匹配，只有在维度相同时才计算特征变化
                if X_source.shape[1] == X_source_adapted.shape[1] and X_target.shape[1] == X_target_adapted.shape[1]:
                    source_diff = np.mean(np.abs(X_source - X_source_adapted))
                    target_diff = np.mean(np.abs(X_target - X_target_adapted))
                    print(f"    源域特征平均变化: {source_diff:.6f}")
                    print(f"    目标域特征平均变化: {target_diff:.6f}")
                    
                    if source_diff < 1e-10 and target_diff < 1e-10:
                        print(f"  ⚠ 警告: 特征变化极小，可能没有真正进行域适应")
                else:
                    # 维度不匹配的情况
                    print(f"    源域特征维度变化: {X_source.shape[1]} -> {X_source_adapted.shape[1]}")
                    print(f"    目标域特征维度变化: {X_target.shape[1]} -> {X_target_adapted.shape[1]}")
                
                # 验证特征维度一致性
                if X_source_adapted.shape[1] != X_target_adapted.shape[1]:
                    print(f"  ⚠ 警告: 源域和目标域适应后特征维度不一致")
                    print(f"    源域: {X_source_adapted.shape}, 目标域: {X_target_adapted.shape}")
                    # 回退到原始特征
                    X_source_adapted = X_source
                    X_target_adapted = X_target
                    print(f"  📝 回退到原始特征进行距离度量计算")
                
                # 显示适应后特征统计
                print(f"    适应后源域均值: {np.mean(X_source_adapted, axis=0)[:3]} ...")
                print(f"    适应后目标域均值: {np.mean(X_target_adapted, axis=0)[:3]} ...")
            else:
                print(f"  ⚠ 未知的UDA方法类型或变换失败: {method_name}")
        
        except Exception as e:
            print(f"  ⚠ 获取适应后特征失败: {e}")
            import traceback
            traceback.print_exc()
        
        return X_source_adapted, X_target_adapted
    
    def _manual_coral_alignment(self, X_source: np.ndarray, X_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """手动实现CORAL协方差对齐"""
        # 计算协方差矩阵
        cov_src = np.cov(X_source.T)
        cov_tar = np.cov(X_target.T)
        
        # 计算变换矩阵 (白化 + 着色)
        U_src, S_src, Vt_src = np.linalg.svd(cov_src)
        U_tar, S_tar, Vt_tar = np.linalg.svd(cov_tar)
        
        # 避免奇异值过小
        S_src = np.maximum(S_src, 1e-8)
        S_tar = np.maximum(S_tar, 1e-8)
        
        # 白化矩阵和着色矩阵
        whiten = U_src @ np.diag(1.0 / np.sqrt(S_src)) @ Vt_src
        color = U_tar @ np.diag(np.sqrt(S_tar)) @ Vt_tar
        
        # 应用变换
        X_source_mean = np.mean(X_source, axis=0)
        X_target_mean = np.mean(X_target, axis=0)
        
        X_source_centered = X_source - X_source_mean
        
        # CORAL变换：源域 -> 白化 -> 目标域着色
        X_source_adapted = (X_source_centered @ whiten @ color) + X_target_mean
        X_target_adapted = X_target  # 目标域保持不变
        
        return X_source_adapted, X_target_adapted
    
    def _calculate_kl_divergence_improved(self, X1: np.ndarray, X2: np.ndarray, bins: int = 20, epsilon: float = 1e-10) -> Tuple[float, Dict[str, float]]:
        """改进的KL散度计算 - 参考analytical_CORAL实现"""
        n_features = X1.shape[1]
        kl_per_feature = {}
        
        for i in range(n_features):
            x_s = X1[:, i]
            x_t = X2[:, i]
            
            # 使用统一的bin范围
            min_val = min(np.min(x_s), np.min(x_t))
            max_val = max(np.max(x_s), np.max(x_t))
            bin_range = (min_val, max_val)
            
            # 计算直方图
            hist_s, _ = np.histogram(x_s, bins=bins, range=bin_range, density=True)
            hist_t, _ = np.histogram(x_t, bins=bins, range=bin_range, density=True)
            
            # 避免零值并归一化
            hist_s = hist_s + epsilon
            hist_t = hist_t + epsilon
            hist_s = hist_s / np.sum(hist_s)
            hist_t = hist_t / np.sum(hist_t)
            
            # 计算对称KL散度
            kl_s_t = entropy(hist_s, hist_t)
            kl_t_s = entropy(hist_t, hist_s)
            kl_per_feature[f'feature_{i}'] = (kl_s_t + kl_t_s) / 2
        
        return float(np.mean(list(kl_per_feature.values()))), kl_per_feature
    
    def _calculate_wasserstein_distance_improved(self, X1: np.ndarray, X2: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """改进的Wasserstein距离计算 - 参考analytical_CORAL实现"""
        n_features = X1.shape[1]
        wasserstein_per_feature = {}
        
        for i in range(n_features):
            x_s = X1[:, i]
            x_t = X2[:, i]
            ws_dist = wasserstein_distance(x_s, x_t)
            wasserstein_per_feature[f'feature_{i}'] = ws_dist
        
        return float(np.mean(list(wasserstein_per_feature.values()))), wasserstein_per_feature
    
    def _calculate_normalized_wasserstein_distance(self, X1: np.ndarray, X2: np.ndarray) -> float:
        """标准化Wasserstein距离计算 - 仿照ADAPT库的normalized_frechet_distance实现"""
        
        # 计算缩放因子和中心点 (参考ADAPT库的实现)
        std_factor = (np.std(X1, axis=0) + np.std(X2, axis=0)) / 2
        mean_center = (np.mean(X1, axis=0) + np.mean(X2, axis=0)) / 2
        
        # 避免零标准差
        std_factor = np.where(std_factor == 0, 1, std_factor)
        
        # 标准化数据
        X1_normalized = (X1 - mean_center) / std_factor
        X2_normalized = (X2 - mean_center) / std_factor
        
        # 计算Wasserstein距离
        n_features = X1_normalized.shape[1]
        wasserstein_sum = 0
        
        for i in range(n_features):
            x_s = X1_normalized[:, i]
            x_t = X2_normalized[:, i]
            ws_dist = wasserstein_distance(x_s, x_t)
            wasserstein_sum += ws_dist
        
        # 除以特征数量进行归一化
        return float(wasserstein_sum / n_features)
    
    def _calculate_normalized_kl_divergence(self, X1: np.ndarray, X2: np.ndarray, bins: int = 20, epsilon: float = 1e-10) -> float:
        """标准化KL散度计算 - 仿照ADAPT库的标准化方法"""
        
        # 计算缩放因子和中心点
        std_factor = (np.std(X1, axis=0) + np.std(X2, axis=0)) / 2
        mean_center = (np.mean(X1, axis=0) + np.mean(X2, axis=0)) / 2
        
        # 避免零标准差
        std_factor = np.where(std_factor == 0, 1, std_factor)
        
        # 标准化数据
        X1_normalized = (X1 - mean_center) / std_factor
        X2_normalized = (X2 - mean_center) / std_factor
        
        # 计算KL散度
        n_features = X1_normalized.shape[1]
        kl_sum = 0
        
        for i in range(n_features):
            x_s = X1_normalized[:, i]
            x_t = X2_normalized[:, i]
            
            # 使用统一的bin范围
            min_val = min(np.min(x_s), np.min(x_t))
            max_val = max(np.max(x_s), np.max(x_t))
            bin_range = (min_val, max_val)
            
            # 计算直方图
            hist_s, _ = np.histogram(x_s, bins=bins, range=bin_range, density=True)
            hist_t, _ = np.histogram(x_t, bins=bins, range=bin_range, density=True)
            
            # 避免零值并归一化
            hist_s = hist_s + epsilon
            hist_t = hist_t + epsilon
            hist_s = hist_s / np.sum(hist_s)
            hist_t = hist_t / np.sum(hist_t)
            
            # 计算对称KL散度
            kl_s_t = entropy(hist_s, hist_t)
            kl_t_s = entropy(hist_t, hist_s)
            kl_sum += (kl_s_t + kl_t_s) / 2
        
        # 除以特征数量进行归一化
        return float(kl_sum / n_features)
    
    def _calculate_mmd_improved(self, X1: np.ndarray, X2: np.ndarray, gamma: float = 1.0) -> float:
        """改进的MMD计算 - 参考analytical_CORAL实现"""
        from sklearn.metrics.pairwise import rbf_kernel
        
        n_x = X1.shape[0]
        n_y = X2.shape[0]
        
        # 限制样本数量以提高计算效率
        max_samples = 200
        if n_x > max_samples:
            idx1 = np.random.choice(n_x, max_samples, replace=False)
            X1 = X1[idx1]
            n_x = max_samples
        if n_y > max_samples:
            idx2 = np.random.choice(n_y, max_samples, replace=False)
            X2 = X2[idx2]
            n_y = max_samples
        
        # 计算核矩阵
        K_xx = rbf_kernel(X1, X1, gamma=gamma)
        K_yy = rbf_kernel(X2, X2, gamma=gamma)
        K_xy = rbf_kernel(X1, X2, gamma=gamma)
        
        # 计算MMD²
        mmd_squared = (np.sum(K_xx) - np.trace(K_xx)) / (n_x * (n_x - 1))
        mmd_squared += (np.sum(K_yy) - np.trace(K_yy)) / (n_y * (n_y - 1))
        mmd_squared -= 2 * np.mean(K_xy)
        
        return float(np.sqrt(max(mmd_squared, 0)))
    
    def _plot_distance_metrics(self, distances: Dict[str, float], method_name: str) -> None:
        """可视化距离度量"""
        
        # 只显示标准化指标（更稳定和可比较）
        available_metrics = []
        metric_labels = []
        
        # 检查标准化指标是否可用
        if 'normalized_linear_discrepancy_before' in distances:
            available_metrics.append('normalized_linear_discrepancy')
            metric_labels.append('Normalized Linear Discrepancy')
        if 'normalized_frechet_distance_before' in distances:
            available_metrics.append('normalized_frechet_distance')
            metric_labels.append('Normalized Frechet Distance')
        if 'normalized_wasserstein_before' in distances:
            available_metrics.append('normalized_wasserstein')
            metric_labels.append('Normalized Wasserstein Distance')
        if 'normalized_kl_divergence_before' in distances:
            available_metrics.append('normalized_kl_divergence')
            metric_labels.append('Normalized KL Divergence')
        
        if not available_metrics:
            print("  ⚠️ 没有可用的距离指标进行可视化")
            return
        
        # 提取数值
        before_values = []
        after_values = []
        improvements = []
        
        for metric in available_metrics:
            if f'{metric}_before' in distances:
                before_values.append(distances[f'{metric}_before'])
                after_values.append(distances[f'{metric}_after'])
                improvements.append(distances[f'{metric}_improvement'])
        
        if not before_values:
            print("  ⚠️ 没有有效的距离度量数据进行可视化")
            return
        
        # 调整图表大小以适应更多指标
        n_metrics = len(available_metrics)
        if n_metrics <= 3:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        fig.suptitle(f'{method_name} - Domain Distance Metrics', fontsize=16, fontweight='bold')
        
        # 距离对比
        x = np.arange(len(available_metrics))
        width = 0.35
        
        ax1.bar(x - width/2, before_values, width, label='Before DA', color=self.colors['target'], alpha=0.8)
        ax1.bar(x + width/2, after_values, width, label='After DA', color=self.colors['adapted'], alpha=0.8)
        
        ax1.set_xlabel('Distance Metrics')
        ax1.set_ylabel('Distance Value')
        ax1.set_title('Domain Distance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metric_labels, rotation=45 if n_metrics > 3 else 0, ha='right' if n_metrics > 3 else 'center')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 改进程度
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax2.bar(available_metrics, improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('Distance Metrics')
        ax2.set_ylabel('Improvement (Before - After)')
        ax2.set_title('Domain Distance Improvement')
        ax2.set_xticklabels(metric_labels, rotation=45 if n_metrics > 3 else 0, ha='right' if n_metrics > 3 else 'center')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.output_dir / f"{method_name}_distance_metrics.pdf"
            plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"  距离度量图已保存: {save_path}")
        
        plt.show()
    
    def plot_performance_comparison(self, 
                                  X_source: np.ndarray, 
                                  y_source: np.ndarray,
                                  X_target: np.ndarray, 
                                  y_target: np.ndarray,
                                  uda_method: AdaptUDAMethod,
                                  method_name: str = "UDA") -> Dict[str, Any]:
        """性能对比可视化"""
        
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
        from sklearn.linear_model import LogisticRegression
        
        # 基线模型（无域适应）
        baseline = LogisticRegression(penalty=None, random_state=42, max_iter=1000)
        baseline.fit(X_source, y_source)
        
        # 预测
        y_pred_baseline = baseline.predict(X_target)
        y_pred_uda = uda_method.predict(X_target)
        
        try:
            y_proba_baseline = baseline.predict_proba(X_target)[:, 1]
        except:
            y_proba_baseline = None
            
        try:
            y_proba_uda = uda_method.predict_proba(X_target)
            if y_proba_uda is not None and len(y_proba_uda.shape) > 1:
                y_proba_uda = y_proba_uda[:, 1]
        except:
            y_proba_uda = None
        
        # 计算指标
        metrics = ['Accuracy', 'F1', 'Precision', 'Recall']
        baseline_scores = [
            accuracy_score(y_target, y_pred_baseline),
            f1_score(y_target, y_pred_baseline, average='binary'),
            precision_score(y_target, y_pred_baseline, average='binary'),
            recall_score(y_target, y_pred_baseline, average='binary')
        ]
        
        uda_scores = [
            accuracy_score(y_target, y_pred_uda),
            f1_score(y_target, y_pred_uda, average='binary'),
            precision_score(y_target, y_pred_uda, average='binary'),
            recall_score(y_target, y_pred_uda, average='binary')
        ]
        
        # 添加AUC（如果可用）
        if y_proba_baseline is not None and y_proba_uda is not None:
            metrics.append('AUC')
            baseline_scores.append(roc_auc_score(y_target, y_proba_baseline))
            uda_scores.append(roc_auc_score(y_target, y_proba_uda))
        
        # 可视化
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_scores, width, 
                      label='Baseline (No DA)', color=self.colors['target'], alpha=0.8)
        bars2 = ax.bar(x + width/2, uda_scores, width, 
                      label=f'{method_name}', color=self.colors['adapted'], alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title(f'Performance Comparison: {method_name} vs Baseline')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.output_dir / f"{method_name}_performance_comparison.pdf"
            plt.savefig(save_path, format='pdf', dpi=1200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"  性能对比图已保存: {save_path}")
        
        plt.show()
        
        return {
            'baseline_scores': dict(zip(metrics, baseline_scores)),
            'uda_scores': dict(zip(metrics, uda_scores)),
            'improvements': {metric: uda - baseline for metric, uda, baseline 
                           in zip(metrics, uda_scores, baseline_scores)}
        }


def create_uda_visualizer(figsize: Tuple[int, int] = (12, 8),
                         save_plots: bool = True,
                         output_dir: str = "results/uda_visualization") -> UDAVisualizer:
    """便捷函数：创建UDA可视化器"""
    return UDAVisualizer(figsize=figsize, save_plots=save_plots, output_dir=output_dir)


if __name__ == "__main__":
    # 使用示例
    print("UDA可视化器使用示例:")
    
    # 创建模拟数据
    np.random.seed(42)
    X_source = np.random.normal(0, 1, (200, 8))
    y_source = np.random.choice([0, 1], 200, p=[0.6, 0.4])
    X_target = np.random.normal(0.5, 1.2, (150, 8))
    y_target = np.random.choice([0, 1], 150, p=[0.4, 0.6])
    
    # 创建可视化器
    visualizer = create_uda_visualizer()
    
    # 生成可视化（无域适应）
    results = visualizer.visualize_domain_adaptation_complete(
        X_source, y_source, X_target, y_target,
        uda_method=None, method_name="No_DA"
    )
    
    print("✓ 可视化示例完成") 