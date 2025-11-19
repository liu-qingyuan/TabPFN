"""
PANDA-Heart项目配置文件
包含数据、模型、实验等所有配置参数
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np

# ==================== 基础路径配置 ====================

@dataclass
class PathConfig:
    """路径配置"""
    # 项目根目录
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 数据路径
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # 模型路径
    models_dir: str = "models"
    checkpoints_dir: str = "models/checkpoints"

    # 结果路径
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"

    # 日志路径
    logs_dir: str = "logs"
    log_file: str = "logs/panda_heart.log"

    def __post_init__(self):
        """初始化后处理路径"""
        # 确保路径是相对于项目根目录的
        if not os.path.isabs(self.data_dir):
            self.data_dir = os.path.join(self.project_root, self.data_dir)
        if not os.path.isabs(self.raw_data_dir):
            self.raw_data_dir = os.path.join(self.project_root, self.raw_data_dir)
        if not os.path.isabs(self.processed_data_dir):
            self.processed_data_dir = os.path.join(self.project_root, self.processed_data_dir)

# ==================== 数据配置 ====================

@dataclass
class DataConfig:
    """UCI心脏病数据配置"""
    # 数据源配置
    cleveland_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    hungarian_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
    va_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"
    switzerland_url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data"

    # 本地存储路径
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # 特征配置
    feature_names: List[str] = field(default_factory=lambda: [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ])

    # 缺失值配置
    missing_values: List[str] = field(default_factory=lambda: ['?', '-9', '-9.0'])
    missing_threshold: float = 0.5  # 缺失率阈值

    # 数据质量配置
    age_range: tuple = (0, 120)
    bp_range: tuple = (50, 250)
    chol_range: tuple = (100, 600)
    heart_rate_range: tuple = (60, 220)
    oldpeak_range: tuple = (0, 10)

    # 数据预处理配置
    binary_labels: bool = True  # 是否使用二分类标签
    quality_check: bool = True  # 是否进行数据质量检查
    standardize_features: bool = True  # 是否标准化特征

    # 可用中心（排除Switzerland，因为数据质量太差）
    available_centers: List[str] = field(default_factory=lambda: ['Cleveland', 'Hungarian', 'VA'])

# ==================== 环境配置 ====================

@dataclass
class EnvironmentConfig:
    """环境配置"""
    # 计算设备
    device: str = "auto"  # auto, cpu, mps, cuda
    num_workers: int = 4
    batch_size: int = 32

    # 内存配置
    max_memory_usage: float = 0.8  # 最大内存使用率
    chunk_size: int = 1000  # 大数据集分块大小

    # 随机种子
    random_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42

    # 并行配置
    use_multiprocessing: bool = True
    n_jobs: int = -1  # -1表示使用所有可用CPU核心

    # GPU配置
    use_gpu: bool = False  # 在Apple Silicon上使用MPS
    precision: str = "32"  # 计算精度: 16, 32

# ==================== TabPFN配置 ====================

@dataclass
class TabPFNConfig:
    """TabPFN模型配置"""
    # 基础配置
    model_name: str = "local"  # 使用本地仓库版本
    ensemble_size: int = 32
    device: str = "auto"

    # 模型参数
    n_layers: int = 10
    d_model: int = 128
    n_heads: int = 4
    max_epochs: int = 1000

    # 推理配置
    batch_size: int = 32
    max_inference_samples: int = 1024

    # 集成多样性
    feature_subsets: bool = True
    input_transformations: bool = True
    seed_variation: bool = True
    model_variants: int = 8

    # 预测配置
    predict_proba: bool = True
    return_logits: bool = False

    # 缓存配置
    cache_predictions: bool = True
    cache_dir: str = "models/cache"

# ==================== 域适应配置 ====================

@dataclass
class DomainAdaptationConfig:
    """域适应配置"""
    # TCA配置
    tca_kernel_types: List[str] = field(default_factory=lambda: ['linear', 'rbf'])
    tca_mu_values: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    tca_n_components: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])
    tca_gamma_values: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0])

    # CORAL配置
    coral_reg_params: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1])
    coral_adaptive_weights: bool = True

    # SA配置
    sa_n_components: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95, 0.99])
    sa_alignment_methods: List[str] = field(default_factory=lambda: ['procrustes', 'least_squares'])

    # PCA配置（基线）
    pca_n_components: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])
    pca_whiten: bool = False

    # 通用配置
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"
    n_jobs: int = -1

    # 可用域适应方法
    available_methods: List[str] = field(default_factory=lambda: [
        'TCA', 'CORAL', 'SA', 'PCA', 'No_UDA'
    ])

# ==================== 基线模型配置 ====================

@dataclass
class BaselineModelsConfig:
    """基线模型配置"""
    # 逻辑回归
    lr_penalty: str = "l1"
    lr_c_values: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    lr_solver: str = "saga"
    lr_class_weight: str = "balanced"
    lr_max_iter: int = 1000

    # XGBoost
    xgb_n_estimators: List[int] = field(default_factory=lambda: [50, 100, 200])
    xgb_max_depth: List[int] = field(default_factory=lambda: [3, 6, 9])
    xgb_learning_rate: List[float] = field(default_factory=lambda: [0.01, 0.1, 0.2])
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0

    # 随机森林
    rf_n_estimators: List[int] = field(default_factory=lambda: [100, 200, 500])
    rf_max_depth: List[Union[int, None]] = field(default_factory=lambda: [None, 10, 20])
    rf_class_weight: str = "balanced"
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1

    # SVM
    svm_c_values: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"
    svm_probability: bool = True
    svm_class_weight: str = "balanced"

    # KNN
    knn_n_neighbors: List[int] = field(default_factory=lambda: [3, 5, 7])
    knn_weights: str = "distance"
    knn_metric: str = "euclidean"
    knn_algorithm: str = "auto"

    # 可用模型
    available_models: List[str] = field(default_factory=lambda: [
        'LASSO_LR', 'XGBoost', 'Random_Forest', 'SVM', 'KNN', 'TabPFN_Only'
    ])

# ==================== 实验配置 ====================

@dataclass
class ExperimentConfig:
    """实验配置"""
    # 实验类型
    experiment_types: List[str] = field(default_factory=lambda: [
        'single_center', 'two_center', 'multi_center'
    ])

    # 模型配置
    models: List[str] = field(default_factory=lambda: [
        'PANDA_TabPFN', 'TabPFN_Only', 'LASSO_LR', 'XGBoost',
        'Random_Forest', 'SVM', 'KNN'
    ])

    # 域适应方法
    uda_methods: List[str] = field(default_factory=lambda: [
        'TCA', 'CORAL', 'SA', 'PCA', 'No_UDA'
    ])

    # 实验重复
    n_repetitions: int = 5
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])

    # 数据中心配置
    centers: List[str] = field(default_factory=lambda: ['Cleveland', 'Hungarian', 'VA'])

    # 交叉验证配置
    cv_folds: int = 10
    stratify: bool = True
    shuffle: bool = True

    # 实验总数计算
    @property
    def total_experiments(self) -> int:
        """计算总实验数量"""
        n_exp_types = len(self.experiment_types)
        n_models = len(self.models)
        n_uda_methods = len(self.uda_methods)
        n_repetitions = self.n_repetitions

        # 基础实验数量
        base_experiments = n_exp_types * n_models * n_uda_methods * n_repetitions

        # 跨域实验的特殊情况
        if 'two_center' in self.experiment_types:
            n_center_pairs = len(self.centers) * (len(self.centers) - 1)  # 有序对
            base_experiments += n_models * n_uda_methods * n_repetitions * n_center_pairs

        return base_experiments

# ==================== 评估配置 ====================

@dataclass
class EvaluationConfig:
    """评估配置"""
    # 主要评估指标
    primary_metrics: List[str] = field(default_factory=lambda: [
        'auc_roc', 'auc_pr', 'sensitivity', 'specificity', 'npv', 'ppv',
        'accuracy', 'f1_score', 'brier_score', 'calibration_slope'
    ])

    # 医学阈值
    medical_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'min_sensitivity': 0.90,  # 筛查要求
        'min_specificity': 0.80,
        'max_brier_score': 0.15,
        'max_ece': 0.05,
        'min_ppv': 0.70,
        'min_npv': 0.90
    })

    # 域适应指标
    domain_metrics: Dict[str, float] = field(default_factory=lambda: {
        'min_performance_retention': 0.85,
        'min_adaptation_gain': 0.03,
        'min_mmd_reduction': 0.20
    })

    # 统计检验
    significance_level: float = 0.05
    multiple_correction: str = "bonferroni"
    effect_size_threshold: float = 0.5  # Cohen's d

    # 校准指标
    calibration_bins: int = 10
    calibration_methods: List[str] = field(default_factory=lambda: [
        'isotonic', 'sigmoid', 'temperature'
    ])

# ==================== 可视化配置 ====================

@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 基础配置
    figure_size: tuple = (12, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8"
    color_palette: str = "Set2"
    font_size: int = 12

    # 医学颜色
    clinical_colors: Dict[str, str] = field(default_factory=lambda: {
        'disease': '#FF6B6B',
        'no_disease': '#4ECDC4',
        'high_risk': '#FF4757',
        'medium_risk': '#FFA502',
        'low_risk': '#26DE81',
        'threshold': '#FFD93D'
    })

    # 输出配置
    save_format: str = "png"
    transparent: bool = False
    bbox_inches: str = "tight"

    # 交互式配置
    interactive_plots: bool = True
    plotly_config: Dict[str, Any] = field(default_factory=lambda: {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d']
    })

    # ROC曲线配置
    roc_colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])

    # 决策曲线配置
    decision_threshold_range: tuple = (0.0, 1.0, 0.01)
    net_benefit_colors: Dict[str, str] = field(default_factory=lambda: {
        'treat_all': '#FF6B6B',
        'treat_none': '#4ECDC4',
        'model': '#2E86AB'
    })

# ==================== 主配置类 ====================

class HeartDiseaseConfig:
    """PANDA-Heart主配置类"""

    def __init__(self):
        # 初始化各模块配置
        self.paths = PathConfig()
        self.data = DataConfig()
        self.environment = EnvironmentConfig()
        self.tabpfn = TabPFNConfig()
        self.domain_adaptation = DomainAdaptationConfig()
        self.baseline_models = BaselineModelsConfig()
        self.experiment = ExperimentConfig()
        self.evaluation = EvaluationConfig()
        self.visualization = VisualizationConfig()

    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 验证路径
            for attr in ['data_dir', 'raw_data_dir', 'processed_data_dir']:
                path = getattr(self.paths, attr, None)
                if path and not os.path.isabs(path):
                    return False

            # 验证模型配置
            assert self.tabpfn.ensemble_size > 0, "TabPFN集成大小必须大于0"
            assert len(self.experiment.models) > 0, "必须至少有一个模型"
            assert len(self.experiment.centers) > 0, "必须至少有一个数据中心"

            # 验证医学阈值
            for metric, threshold in self.evaluation.medical_thresholds.items():
                assert 0 <= threshold <= 1, f"{metric}阈值必须在0-1范围内"

            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

    def create_directories(self):
        """创建必要的目录"""
        directories = [
            self.paths.data_dir,
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.models_dir,
            self.paths.results_dir,
            self.paths.figures_dir,
            self.paths.tables_dir,
            self.paths.logs_dir
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'experiment_count': self.experiment.total_experiments,
            'models_count': len(self.experiment.models),
            'centers_count': len(self.experiment.centers),
            'uda_methods_count': len(self.experiment.uda_methods),
            'repetitions': self.experiment.n_repetitions,
            'primary_metrics': len(self.evaluation.primary_metrics),
            'tabpfn_ensemble_size': self.tabpfn.ensemble_size
        }

# ==================== 全局配置实例 ====================

# 创建全局配置实例
config = HeartDiseaseConfig()

# 便捷函数
def get_config() -> HeartDiseaseConfig:
    """获取全局配置实例"""
    return config

def update_config(updates: Dict[str, Any]):
    """更新配置"""
    for section, values in updates.items():
        if hasattr(config, section):
            section_obj = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)
                else:
                    print(f"警告: {section}.{key} 不存在")
        else:
            print(f"警告: 配置节 {section} 不存在")

if __name__ == "__main__":
    # 测试配置
    print("=== PANDA-Heart 配置测试 ===")

    # 验证配置
    if config.validate_config():
        print("✅ 配置验证通过")
    else:
        print("❌ 配置验证失败")

    # 创建目录
    config.create_directories()
    print("✅ 目录创建完成")

    # 显示配置摘要
    summary = config.get_summary()
    print("\n配置摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"\n总实验数量: {config.experiment.total_experiments}")
    print("✅ 配置模块测试完成")