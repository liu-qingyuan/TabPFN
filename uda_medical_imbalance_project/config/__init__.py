"""
UDA Medical Imbalance Project - 统一配置管理模块

提供项目的统一配置管理接口，包括：
- 模型配置 (TabPFN, 基线模型, UDA方法)
- 实验配置 (特征选择, 预处理, 交叉验证)
- 数据配置 (数据集路径, 特征映射)
- 输出配置 (结果保存, 可视化)

作者: UDA Medical Team
日期: 2024-01-30
"""

from typing import Dict, Any, Optional, List
import os
from pathlib import Path
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)

# 导入配置类
try:
    from .model_config import ModelConfig, TabPFNConfig, BaselineConfig
    from .uda_config import UDAConfig, UDAMethodConfig
    from .experiment_config import ExperimentConfig, CrossValidationConfig
    MODEL_CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"部分配置模块不可用: {e}")
    MODEL_CONFIG_AVAILABLE = False

@dataclass
class ProjectConfig:
    """项目主配置类"""
    
    # 项目信息
    project_name: str = "UDA Medical Imbalance Project"
    version: str = "1.0.0"
    
    # 路径配置
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "results")
    config_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "configs")
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_file: Optional[str] = "uda_medical.log"
    
    # 计算资源配置
    n_jobs: int = -1
    random_state: int = 42
    use_gpu: bool = True
    gpu_device: str = "auto"
    
    # 默认实验配置
    default_feature_set: str = "best8"
    default_scaler: str = "standard"
    default_imbalance_method: str = "smote"
    default_cv_folds: int = 10
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保路径是Path对象
        for attr_name in ['project_root', 'data_dir', 'results_dir', 'config_dir']:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str):
                setattr(self, attr_name, Path(attr_value))
        
        # 创建必要的目录
        self.create_directories()
    
    def create_directories(self):
        """创建必要的目录"""
        for dir_path in [self.data_dir, self.results_dir, self.config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    def save_to_file(self, filepath: Optional[Path] = None):
        """保存配置到文件"""
        if filepath is None:
            filepath = self.config_dir / "project_config.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"项目配置已保存到: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'ProjectConfig':
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 转换路径字符串为Path对象
        for key in ['project_root', 'data_dir', 'results_dir', 'config_dir']:
            if key in config_dict:
                config_dict[key] = Path(config_dict[key])
        
        return cls(**config_dict)

class ConfigManager:
    """配置管理器 - 统一管理所有配置"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径，如果为None则使用默认配置
        """
        self.project_config = self._load_project_config(config_file)
        self._setup_logging()
        
        # 初始化其他配置
        if MODEL_CONFIG_AVAILABLE:
            self.model_config = ModelConfig()
            self.uda_config = UDAConfig()
            self.experiment_config = ExperimentConfig()
        else:
            logger.warning("模型配置模块不可用，使用简化配置")
            self.model_config = None
            self.uda_config = None
            self.experiment_config = None
    
    def _load_project_config(self, config_file: Optional[Path]) -> ProjectConfig:
        """加载项目配置"""
        if config_file and config_file.exists():
            try:
                return ProjectConfig.load_from_file(config_file)
            except Exception as e:
                logger.warning(f"加载配置文件失败，使用默认配置: {e}")
        
        return ProjectConfig()
    
    def _setup_logging(self):
        """设置日志系统"""
        # 配置根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.project_config.log_level))
        
        # 清除现有处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 创建格式化器
        formatter = logging.Formatter(self.project_config.log_format)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # 文件处理器（如果启用）
        if self.project_config.log_to_file and self.project_config.log_file:
            log_file_path = self.project_config.results_dir / self.project_config.log_file
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    def get_model_config(self, model_type: str = "tabpfn") -> Dict[str, Any]:
        """获取模型配置"""
        if self.model_config:
            return self.model_config.get_config(model_type)
        else:
            # 返回默认配置
            return {
                'random_state': self.project_config.random_state,
                'n_jobs': self.project_config.n_jobs
            }
    
    def get_uda_config(self, method_name: str = "SA") -> Dict[str, Any]:
        """获取UDA方法配置"""
        if self.uda_config:
            return self.uda_config.get_method_config(method_name)
        else:
            return {
                'random_state': self.project_config.random_state,
                'verbose': 0
            }
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """获取实验配置"""
        if self.experiment_config:
            return self.experiment_config.to_dict()
        else:
            return {
                'feature_set': self.project_config.default_feature_set,
                'scaler': self.project_config.default_scaler,
                'imbalance_method': self.project_config.default_imbalance_method,
                'cv_folds': self.project_config.default_cv_folds,
                'random_state': self.project_config.random_state
            }
    
    def update_config(self, config_type: str, **kwargs):
        """更新配置"""
        if config_type == "project":
            for key, value in kwargs.items():
                if hasattr(self.project_config, key):
                    setattr(self.project_config, key, value)
        elif config_type == "model" and self.model_config:
            # 更新模型配置的逻辑
            pass
        elif config_type == "uda" and self.uda_config:
            # 更新UDA配置的逻辑
            pass
        elif config_type == "experiment" and self.experiment_config:
            # 更新实验配置的逻辑
            pass
        else:
            logger.warning(f"未知的配置类型: {config_type}")
    
    def save_all_configs(self):
        """保存所有配置"""
        self.project_config.save_to_file()
        
        if self.model_config:
            # 保存模型配置
            pass
        if self.uda_config:
            # 保存UDA配置
            pass
        if self.experiment_config:
            # 保存实验配置
            pass
    
    def get_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        summary = {
            'project': {
                'name': self.project_config.project_name,
                'version': self.project_config.version,
                'data_dir': str(self.project_config.data_dir),
                'results_dir': str(self.project_config.results_dir)
            },
            'compute': {
                'n_jobs': self.project_config.n_jobs,
                'random_state': self.project_config.random_state,
                'use_gpu': self.project_config.use_gpu
            },
            'defaults': {
                'feature_set': self.project_config.default_feature_set,
                'scaler': self.project_config.default_scaler,
                'imbalance_method': self.project_config.default_imbalance_method,
                'cv_folds': self.project_config.default_cv_folds
            }
        }
        
        return summary

# 全局配置管理器实例
_config_manager = None

def get_config_manager(config_file: Optional[Path] = None) -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager

def get_project_config() -> ProjectConfig:
    """获取项目配置的便捷函数"""
    return get_config_manager().project_config

def get_model_config(model_type: str = "tabpfn") -> Dict[str, Any]:
    """获取模型配置的便捷函数"""
    return get_config_manager().get_model_config(model_type)

def get_uda_config(method_name: str = "SA") -> Dict[str, Any]:
    """获取UDA配置的便捷函数"""
    return get_config_manager().get_uda_config(method_name)

def get_experiment_config() -> Dict[str, Any]:
    """获取实验配置的便捷函数"""
    return get_config_manager().get_experiment_config()

# 导出主要接口
__all__ = [
    'ProjectConfig',
    'ConfigManager',
    'get_config_manager',
    'get_project_config',
    'get_model_config',
    'get_uda_config',
    'get_experiment_config'
] 