"""
模型选择器模块

支持多种TabPFN模型类型的统一接口，包括：
- AutoTabPFN: 自动化的TabPFN集成模型
- TunedTabPFN: 超参数优化的TabPFN模型
- BaseTabPFN: 原生TabPFN模型
- RFTabPFN: 随机森林风格的TabPFN集成
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestClassifier

# 导入配置
from ..config.settings import MODEL_CONFIGS

# 尝试导入不同的TabPFN实现
try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
    AUTO_TABPFN_AVAILABLE = True
except ImportError:
    logging.warning("AutoTabPFN不可用，请安装tabpfn_extensions")
    AUTO_TABPFN_AVAILABLE = False

try:
    from tabpfn_extensions.hpo import TunedTabPFNClassifier
    TUNED_TABPFN_AVAILABLE = True
except ImportError:
    logging.warning("TunedTabPFN不可用，请安装tabpfn_extensions")
    TUNED_TABPFN_AVAILABLE = False

try:
    from tabpfn_extensions import TabPFNClassifier
    BASE_TABPFN_AVAILABLE = True
except ImportError:
    try:
        from tabpfn import TabPFNClassifier
        BASE_TABPFN_AVAILABLE = True
    except ImportError:
        logging.warning("原生TabPFN不可用，请安装tabpfn或tabpfn_extensions")
        BASE_TABPFN_AVAILABLE = False

try:
    from tabpfn_extensions.rf_pfn import RandomForestTabPFNClassifier
    from tabpfn_extensions import TabPFNClassifier as BaseTabPFN
    RF_TABPFN_AVAILABLE = True
except ImportError:
    logging.warning("RF TabPFN不可用，请安装tabpfn_extensions")
    RF_TABPFN_AVAILABLE = False

def get_model(model_type: str, categorical_feature_indices: Optional[List[int]] = None, **kwargs: Any):
    """
    获取指定类型的模型
    
    参数:
    - model_type: 模型类型 ('auto', 'tuned', 'base', 'rf')
    - categorical_feature_indices: 类别特征索引（某些模型可能需要）
    - **kwargs: 模型参数
    
    返回:
    - 模型实例
    """
    
    # 移除不需要的参数
    clean_kwargs = {k: v for k, v in kwargs.items() 
                   if k not in ['categorical_feature_indices']}
    
    if model_type == "auto":
        if not AUTO_TABPFN_AVAILABLE:
            raise ImportError("AutoTabPFN不可用，请安装tabpfn_extensions")
        
        # 从配置文件获取默认参数
        config_params = MODEL_CONFIGS.get('auto', {}).copy()
        
        # 分离直接参数和phe_init_args参数
        direct_params = {}
        phe_init_args = {}
        
        # 直接传递给AutoTabPFN的参数
        direct_param_names = {
            'max_time', 'preset', 'ges_scoring_string', 'device', 
            'random_state', 'ignore_pretraining_limits', 'categorical_feature_indices'
        }
        
        # 从配置文件中提取参数
        for key, value in config_params.items():
            if key == 'phe_init_args':
                # 如果配置文件中有phe_init_args，直接使用
                phe_init_args.update(value)
            elif key in direct_param_names:
                direct_params[key] = value
        
        # 合并用户传入的参数，但避免重复
        for key, value in clean_kwargs.items():
            if key in direct_param_names:
                direct_params[key] = value
            elif key in {'max_models', 'validation_method', 'n_repeats', 'n_folds', 
                        'holdout_fraction', 'ges_n_iterations'}:
                phe_init_args[key] = value
        
        # 处理categorical_feature_indices参数
        if categorical_feature_indices is not None:
            direct_params['categorical_feature_indices'] = categorical_feature_indices
        elif 'categorical_feature_indices' not in direct_params or direct_params['categorical_feature_indices'] is None:
            direct_params['categorical_feature_indices'] = categorical_feature_indices
        # 添加phe_init_args参数
        if phe_init_args:
            direct_params['phe_init_args'] = phe_init_args
        
        logging.info(f"创建AutoTabPFN模型，参数: {direct_params}")
        return AutoTabPFNClassifier(**direct_params)
        
    elif model_type == "tuned":
        if not TUNED_TABPFN_AVAILABLE:
            raise ImportError("TunedTabPFN不可用，请安装tabpfn_extensions")
        
        # 设置默认参数
        default_params = {
            'random_state': 42
        }
        default_params.update(clean_kwargs)
        
        logging.info(f"创建TunedTabPFN模型，参数: {default_params}")
        return TunedTabPFNClassifier(**default_params)
        
    elif model_type == "base":
        if not BASE_TABPFN_AVAILABLE:
            raise ImportError("原生TabPFN不可用，请安装tabpfn或tabpfn_extensions")
        
        # 设置默认参数
        default_params = {
            'device': 'cuda',
            'random_state': 42
        }
        default_params.update(clean_kwargs)
        
        logging.info(f"创建原生TabPFN模型，参数: {default_params}")
        return TabPFNClassifier(**default_params)
        
    elif model_type == "rf":
        if RF_TABPFN_AVAILABLE:
            # 使用RF TabPFN
            base_tabpfn = BaseTabPFN()
            
            # 设置默认参数
            default_params = {
                'n_estimators': 10,
                'max_depth': 3,
                'random_state': 42
            }
            default_params.update(clean_kwargs)
            
            logging.info(f"创建RF TabPFN模型，参数: {default_params}")
            return RandomForestTabPFNClassifier(tabpfn=base_tabpfn, **default_params)
        else:
            # 降级为随机森林
            logging.warning("RF TabPFN不可用，使用随机森林作为替代")
            default_params = {
                'n_estimators': 10,
                'max_depth': None,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(clean_kwargs)
            
            return RandomForestClassifier(**default_params)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持的类型: 'auto', 'tuned', 'base', 'rf'")

def get_available_models() -> List[str]:
    """获取可用的模型类型列表"""
    available = []
    
    if AUTO_TABPFN_AVAILABLE:
        available.append("auto")
    if TUNED_TABPFN_AVAILABLE:
        available.append("tuned")
    if BASE_TABPFN_AVAILABLE:
        available.append("base")
    
    # RF总是可用（作为fallback）
    available.append("rf")
    
    return available

def validate_model_params(model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证和标准化模型参数
    
    参数:
    - model_type: 模型类型
    - params: 原始参数字典
    
    返回:
    - Dict[str, Any]: 验证后的参数字典
    """
    validated_params = params.copy()
    
    if model_type == "auto":
        # AutoTabPFN参数验证
        if 'max_time' in validated_params and validated_params['max_time'] <= 0:
            logging.warning("max_time必须为正数，设置为默认值30")
            validated_params['max_time'] = 30
            
    elif model_type == "tuned":
        # TunedTabPFN参数验证
        # 大多数参数由模型内部处理
        pass
            
    elif model_type == "base":
        # 原生TabPFN参数验证
        if 'device' not in validated_params:
            validated_params['device'] = 'cuda'
            
    elif model_type == "rf":
        # RF参数验证
        if 'n_estimators' in validated_params and validated_params['n_estimators'] <= 0:
            validated_params['n_estimators'] = 10
            
    return validated_params

# 模型配置预设 - 简化版本
MODEL_PRESETS = {
    'default': {
        'model_type': 'auto'
    }
}

def get_model_preset(preset_name: str, **override_params) -> Dict[str, Any]:
    """
    获取预设的模型配置
    
    参数:
    - preset_name: 预设名称
    - **override_params: 覆盖参数
    
    返回:
    - Dict[str, Any]: 模型配置
    """
    if preset_name not in MODEL_PRESETS:
        raise ValueError(f"不支持的预设: {preset_name}. 可用预设: {list(MODEL_PRESETS.keys())}")
    
    config = MODEL_PRESETS[preset_name].copy()
    config.update(override_params)
    
    return config

def test_model_availability():
    """测试模型可用性"""
    results = {}
    
    for model_type in ['auto', 'tuned', 'base', 'rf']:
        try:
            model = get_model(model_type)
            results[model_type] = True
            logging.info(f"{model_type} 模型可用: {type(model).__name__}")
        except Exception as e:
            results[model_type] = False
            logging.warning(f"{model_type} 模型不可用: {e}")
    
    return results

if __name__ == "__main__":
    # 测试模型可用性
    logging.basicConfig(level=logging.INFO)
    print("测试模型可用性...")
    results = test_model_availability()
    print(f"可用模型: {[k for k, v in results.items() if v]}") 