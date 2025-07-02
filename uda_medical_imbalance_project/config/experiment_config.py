"""
UDA Medical Imbalance Project - 实验配置管理

本模块提供实验的全局配置管理，包括数据处理、模型选择、UDA方法等配置
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class PreprocessingConfig:
    """数据预处理配置"""
    # 特征选择配置
    feature_count: int = 7  # 7-10个特征可选
    categorical_features: List[str] = field(default_factory=list)  # 类别特征列表（默认空）
    
    # 标准化配置
    scaler_type: str = "standard"  # standard | robust
    
    # 类别不平衡处理（可选）
    imbalance_method: Optional[str] = "smote"  # smote | borderline_smote | adasyn | None
    
    # SMOTE相关参数
    smote_k_neighbors: int = 5
    smote_random_state: int = 42


@dataclass 
class SourceDomainConfig:
    """源域方法对比配置"""
    # 交叉验证配置
    cv_folds: int = 10
    cv_random_state: int = 42
    
    # 模型列表
    models: List[str] = field(default_factory=lambda: [
        "tabpfn",
        "paper_method", 
        "pkuph_baseline",
        "mayo_baseline"
    ])


@dataclass
class UDAConfig:
    """UDA算法配置"""
    # 协变量偏移方法
    covariate_shift_methods: List[str] = field(default_factory=lambda: ["DM"])
    
    # 隐藏协变量偏移 - 线性/核方法
    linear_kernel_methods: List[str] = field(default_factory=lambda: [
        "SA", "TCA", "JDA", "CORAL"
    ])
    
    # 深度学习方法
    deep_methods: List[str] = field(default_factory=lambda: [
        "DANN", "ADDA", "WDGRL", "DeepCORAL", "MCD", "MDD", "CDAN"
    ])
    
    # 最优传输方法
    optimal_transport_methods: List[str] = field(default_factory=lambda: ["POT"])
    
    # UDA方法特定参数
    uda_params: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "CORAL": {
            "lambda_value": 1.0
        },
        "TCA": {
            "n_components": 10,
            "kernel": "linear",
            "mu": 1.0
        },
        "JDA": {
            "n_components": 10,
            "kernel": "linear", 
            "mu": 1.0,
            "n_iter": 10
        },
        "DANN": {
            "epochs": 100,
            "batch_size": 32,
            "lr": 0.001,
            "lambda_value": 1.0
        },
        "POT": {
            "reg": 0.1,
            "metric": "sqeuclidean"
        }
    })


@dataclass
class EvaluationConfig:
    """评估配置"""
    # 评估指标
    metrics: List[str] = field(default_factory=lambda: [
        "auc", "accuracy", "f1", "precision", "recall"
    ])
    
    # 混淆矩阵类别标签
    class_labels: List[str] = field(default_factory=lambda: ["阴性", "阳性"])
    
    # ROC曲线配置
    roc_pos_label: int = 1


@dataclass
class VisualizationConfig:
    """可视化配置"""
    # 降维可视化
    enable_pca: bool = True
    enable_tsne: bool = True
    tsne_perplexity: int = 30
    tsne_n_iter: int = 1000
    
    # 分布对比
    enable_distribution_plots: bool = True
    enable_distance_metrics: bool = True
    
    # 距离度量类型
    distance_metrics: List[str] = field(default_factory=lambda: [
        "kl_divergence", "wasserstein", "mmd"
    ])
    
    # 图表样式
    figure_size: tuple = (10, 8)
    dpi: int = 300
    style: str = "seaborn-v0_8"


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    # 实验基本信息
    name: str = "medical_uda_experiment"
    description: str = "UDA方法在医疗数据上的对比实验"
    version: str = "1.0.0"
    
    # 数据路径配置
    data_source_path: str = "data/source_domain.xlsx"
    data_target_path: str = "data/target_domain.xlsx"
    
    # 输出路径配置
    output_dir: str = "experiments"
    
    # 随机种子
    random_state: int = 42
    
    # 子配置
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    source_domain: SourceDomainConfig = field(default_factory=SourceDomainConfig)
    uda: UDAConfig = field(default_factory=UDAConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = ExperimentConfig()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> ExperimentConfig:
        """从YAML文件加载配置"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 更新配置
        self._update_config_from_dict(config_dict)
        return self.config
    
    def save_config(self, save_path: str):
        """保存配置到YAML文件"""
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self._config_to_dict()
        
        with open(save_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def get_uda_methods(self) -> List[str]:
        """获取所有启用的UDA方法"""
        all_methods = []
        all_methods.extend(self.config.uda.covariate_shift_methods)
        all_methods.extend(self.config.uda.linear_kernel_methods)
        all_methods.extend(self.config.uda.deep_methods)
        all_methods.extend(self.config.uda.optimal_transport_methods)
        return all_methods
    
    def get_uda_params(self, method: str) -> Dict[str, Any]:
        """获取特定UDA方法的参数"""
        return self.config.uda.uda_params.get(method, {})
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置"""
        # 递归更新配置对象
        def update_dataclass(obj: Any, data: Dict[str, Any]) -> None:
            for key, value in data.items():
                if hasattr(obj, key):
                    current_value = getattr(obj, key)
                    if hasattr(current_value, '__dict__'):  # 嵌套dataclass
                        update_dataclass(current_value, value)
                    else:
                        setattr(obj, key, value)
        
        update_dataclass(self.config, config_dict)
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典"""
        import dataclasses
        
        def asdict_recursive(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj):
                return {field.name: asdict_recursive(getattr(obj, field.name)) 
                       for field in dataclasses.fields(obj)}
            elif isinstance(obj, list):
                return [asdict_recursive(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: asdict_recursive(value) for key, value in obj.items()}
            else:
                return obj
        
        result = asdict_recursive(self.config)
        # 确保返回类型正确
        return result if isinstance(result, dict) else {}


# 预定义配置模板
def get_minimal_config() -> ExperimentConfig:
    """获取最小配置（快速测试用）"""
    config = ExperimentConfig()
    config.preprocessing.feature_count = 7
    config.preprocessing.imbalance_method = None
    config.uda.linear_kernel_methods = ["CORAL"]
    config.uda.deep_methods = []
    config.uda.optimal_transport_methods = []
    return config


def get_full_config() -> ExperimentConfig:
    """获取完整配置（所有方法）"""
    return ExperimentConfig()


def get_linear_methods_config() -> ExperimentConfig:
    """获取仅线性方法配置"""
    config = ExperimentConfig()
    config.uda.deep_methods = []
    config.uda.optimal_transport_methods = []
    return config


# 使用示例
if __name__ == "__main__":
    # 创建配置管理器
    config_manager = ConfigManager()
    
    # 保存默认配置
    config_manager.save_config("configs/default_config.yaml")
    
    # 保存最小配置
    config_manager.config = get_minimal_config()
    config_manager.save_config("configs/minimal_config.yaml")
    
    print("配置文件已生成!")
    print(f"UDA方法列表: {config_manager.get_uda_methods()}")
    print(f"CORAL参数: {config_manager.get_uda_params('CORAL')}") 