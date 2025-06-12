# MMD visualization module 
try:
    from ..metrics.discrepancy import calculate_kl_divergence, calculate_wasserstein_distances, compute_mmd, compute_domain_discrepancy, detect_outliers
except ImportError:
    try:
        from analytical_mmd_A2B_feature58.metrics.discrepancy import calculate_kl_divergence, calculate_wasserstein_distances, compute_mmd, compute_domain_discrepancy, detect_outliers
    except ImportError:
        # 定义空函数以避免导入错误
        def calculate_kl_divergence(*args, **kwargs): return 0.0, {}
        def calculate_wasserstein_distances(*args, **kwargs): return 0.0, {}
        def compute_mmd(*args, **kwargs): return 0.0
        def compute_domain_discrepancy(*args, **kwargs): return {}
        def detect_outliers(*args, **kwargs): return [], [], [], []

from .tsne_plots import visualize_tsne
from .histogram_plots import visualize_feature_histograms, histograms_stats_table, histograms_visual_stats_table
from .comparison_plots import compare_before_after_adaptation, visualize_mmd_adaptation_results, plot_mmd_methods_comparison, generate_performance_comparison_plots
from .roc_plots import plot_roc_curve
from .utils import close_figures, setup_matplotlib_style
from .performance_plots import (
    plot_metrics_comparison, plot_domain_adaptation_improvement,
    plot_cross_dataset_performance, plot_model_comparison,
    plot_metrics_radar_chart, create_performance_summary_table
)

__all__ = [
    # Metrics
    'calculate_kl_divergence',
    'calculate_wasserstein_distances',
    'compute_mmd',
    'compute_domain_discrepancy',
    'detect_outliers',
    
    # Visualization functions
    'visualize_tsne',
    'visualize_feature_histograms',
    'histograms_stats_table',
    'histograms_visual_stats_table',
    'compare_before_after_adaptation',
    'visualize_mmd_adaptation_results',
    'plot_mmd_methods_comparison',
    'generate_performance_comparison_plots',
    'plot_roc_curve',
    
    # Performance comparison plots
    'plot_metrics_comparison',
    'plot_domain_adaptation_improvement',
    'plot_cross_dataset_performance',
    'plot_model_comparison',
    'plot_metrics_radar_chart',
    'create_performance_summary_table',
    
    # Utilities
    'close_figures',
    'setup_matplotlib_style',
] 