# config/settings.py 文档

## 概述

`config/settings.py` 是项目的配置管理核心文件，定义了所有重要的常量、参数和配置项。本文档详细介绍该文件的结构、各配置项的含义以及使用方法。

## 文件位置
```
analytical_mmd_A2B_feature58/config/settings.py
```

## 主要功能

### 1. 特征管理
- 定义项目使用的58个核心医疗特征
- 管理最佳特征组合（7个特征）
- 标识类别特征和数值特征

### 2. 模型配置
- AutoTabPFN配置参数
- TunedTabPFN配置参数  
- 原生TabPFN配置参数
- RF集成模型配置参数

### 3. MMD算法配置
- 线性MMD变换参数
- 核PCA MMD参数
- 均值标准差对齐参数
- 类条件MMD参数

### 4. 数据路径管理
- 数据集文件路径
- 结果保存路径
- 可视化输出路径

## 详细配置项

### 特征配置

#### SELECTED_FEATURES
```python
SELECTED_FEATURES = [
    'Feature1', 'Feature2', 'Feature3', ..., 'Feature63'
]
```
- **类型**: `List[str]`
- **长度**: 58个特征
- **作用**: 定义项目使用的核心医疗特征列表
- **来源**: 基于特征选择算法从原始数据集筛选
- **用途**: 数据加载、模型训练、结果分析的基础

#### BEST_7_FEATURES
```python
BEST_7_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]
```
- **类型**: `List[str]`
- **长度**: 7个特征
- **作用**: 经过实验验证的最优特征组合
- **性能**: 在跨域任务中表现最佳
- **推荐场景**: 小样本实验、快速验证、资源受限环境

#### CAT_FEATURE_NAMES
```python
CAT_FEATURE_NAMES = [
    'Feature1', 'Feature3', 'Feature5', ..., 'Feature63'
]
```
- **类型**: `List[str]`
- **作用**: 标识需要特殊处理的类别特征
- **用途**: 模型训练时的categorical_feature_indices参数
- **重要性**: 影响模型对类别特征的处理方式

### 模型配置

#### AutoTabPFN配置 (MODEL_CONFIGS['auto'])
```python
MODEL_CONFIGS['auto'] = {
    'max_time': 30,                    # 最大训练时间(秒)
    'preset': 'default',               # 预设配置
    'ges_scoring_string': 'accuracy',  # 集成评分方法
    'device': 'cuda',                  # 计算设备
    'random_state': 42,                # 随机种子
    'ignore_pretraining_limits': False, # 是否忽略预训练限制
    'phe_init_args': {                 # PostHocEnsemble参数
        'max_models': 15,              # 最大模型数量
        'validation_method': 'cv',     # 验证方法
        'n_repeats': 100,              # 重复次数
        'n_folds': 10,                 # 交叉验证折数
        'holdout_fraction': 0.4,       # 保留集比例
        'ges_n_iterations': 20         # 集成搜索迭代次数
    }
}
```

**关键参数说明**:
- `max_time`: 控制单个模型的最大训练时间，影响性能和速度平衡
- `ges_scoring_string`: 集成评分指标，可选 'accuracy', 'auc', 'f1'
- `phe_init_args`: PostHocEnsemble的详细配置
  - `max_models`: 集成中的最大模型数量，影响性能和计算时间
  - `validation_method`: 'cv'(交叉验证), 'holdout'(保留集), 'none'(无验证)
  - `n_repeats`: 验证重复次数，影响结果稳定性

### MMD方法配置

#### 线性MMD配置 (MMD_METHODS['linear'])
```python
MMD_METHODS['linear'] = {
    'n_epochs': 200,                    # 训练轮数
    'lr': 3e-4,                        # 学习率
    'batch_size': 64,                  # 批大小
    'lambda_reg': 1e-3,                # 正则化系数
    'staged_training': True,           # 分阶段训练
    'dynamic_gamma': True,             # 动态gamma调整
    'gamma_search_values': [0.01, 0.05, 0.1],  # gamma搜索值
    'standardize_features': True,      # 特征标准化
    'use_gradient_clipping': True,     # 梯度裁剪
    'max_grad_norm': 1.0,             # 最大梯度范数
    'monitor_gradients': True         # 梯度监控
}
```

**关键参数详解**:
- `staged_training`: 先用小gamma训练，再用大gamma微调，提高收敛稳定性
- `dynamic_gamma`: 自动搜索最优gamma值，避免手动调参
- `gamma_search_values`: gamma候选值列表，影响核函数的带宽
- `use_gradient_clipping`: 防止梯度爆炸，提高训练稳定性

### 数据路径配置

#### DATA_PATHS
```python
DATA_PATHS = {
    'A': "/home/24052432g/TabPFN/data/AI4healthcare.xlsx",
    'B': "/home/24052432g/TabPFN/data/HenanCancerHospital_features63_58.xlsx"
}
```
- **类型**: `Dict[str, str]`
- **作用**: 定义数据集文件的绝对路径
- **键值说明**:
  - 'A': AI4Health数据集（源域）
  - 'B': 河南癌症医院数据集（目标域）

#### LABEL_COL
```python
LABEL_COL = "Label"
```
- **类型**: `str`
- **作用**: 指定标签列的列名
- **重要性**: 数据加载时用于区分特征和标签

### 实验配置

#### EXPERIMENT_CONFIG
```python
EXPERIMENT_CONFIG = {
    'test_size': 0.2,                   # 验证集比例
    'random_state': 42,                 # 随机种子
    'optimize_threshold': True,         # 阈值优化
    'cross_validation': False,          # 交叉验证
    'save_visualizations': True,        # 保存可视化
    'save_models': False               # 保存模型
}
```

#### CROSS_DOMAIN_CONFIG
```python
CROSS_DOMAIN_CONFIG = {
    'use_best_7_features': True,        # 使用最佳7特征
    'enable_cross_validation': True,    # 启用交叉验证
    'cv_folds': 10,                    # 交叉验证折数
    'evaluate_all_datasets': True,     # 评估所有数据集
    'save_detailed_results': True      # 保存详细结果
}
```

## 配置函数

### get_features_by_type()
```python
def get_features_by_type(feature_type: str = 'all') -> List[str]:
    """
    根据类型获取特征列表
    
    参数:
    - feature_type: 'all'(所有特征), 'best7'(最佳7特征), 'categorical'(类别特征)
    
    返回:
    - List[str]: 特征名称列表
    
    使用示例:
    >>> features = get_features_by_type('best7')
    >>> print(f"最佳7特征: {features}")
    """
```

### get_categorical_indices()
```python
def get_categorical_indices(feature_type: str = 'all') -> List[int]:
    """
    根据特征类型获取类别特征索引
    
    参数:
    - feature_type: 'all'(所有特征), 'best7'(最佳7特征)
    
    返回:
    - List[int]: 类别特征在特征列表中的索引
    
    使用示例:
    >>> cat_indices = get_categorical_indices('best7')
    >>> print(f"类别特征索引: {cat_indices}")
    """
```

### get_model_config()
```python
def get_model_config(model_type: str, preset: str = 'balanced', 
                     categorical_feature_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    获取模型配置
    
    参数:
    - model_type: 'auto', 'base', 'rf'
    - preset: 'fast'(快速), 'balanced'(平衡), 'accurate'(高精度)
    - categorical_feature_indices: 类别特征索引
    
    返回:
    - Dict[str, Any]: 模型配置字典
    
    使用示例:
    >>> config = get_model_config('auto', 'balanced')
    >>> model = get_model('auto', **config)
    """
```

## 使用示例

### 基础配置使用
```python
from analytical_mmd_A2B_feature58.config.settings import (
    SELECTED_FEATURES, BEST_7_FEATURES, MODEL_CONFIGS, MMD_METHODS
)

# 获取特征列表
features = get_features_by_type('best7')
cat_indices = get_categorical_indices('best7')

# 获取模型配置
model_config = get_model_config('auto', 'balanced', cat_indices)

# 获取MMD配置
mmd_config = MMD_METHODS['linear'].copy()
```

### 自定义配置
```python
# 修改模型配置
custom_model_config = MODEL_CONFIGS['auto'].copy()
custom_model_config.update({
    'max_time': 60,  # 增加训练时间
    'phe_init_args': {
        'max_models': 20,    # 增加模型数量
        'n_repeats': 150     # 增加重复次数
    }
})

# 修改MMD配置
custom_mmd_config = MMD_METHODS['linear'].copy()
custom_mmd_config.update({
    'n_epochs': 300,         # 增加训练轮数
    'lr': 1e-4,             # 降低学习率
    'lambda_reg': 1e-2,     # 增加正则化
})
```

### 环境适应配置
```python
import torch

# GPU内存不足时的配置
if torch.cuda.is_available():
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if memory_gb < 8:
        # 内存优化配置
        MMD_METHODS['linear']['batch_size'] = 32
        MODEL_CONFIGS['auto']['max_time'] = 15
        MODEL_CONFIGS['auto']['phe_init_args']['max_models'] = 8

# CPU环境配置
if not torch.cuda.is_available():
    MODEL_CONFIGS['auto']['device'] = 'cpu'
    MODEL_CONFIGS['base']['device'] = 'cpu'
```

## 配置验证

### 参数有效性检查
```python
def validate_config():
    """验证配置的有效性"""
    # 检查特征配置
    assert len(BEST_7_FEATURES) == 7, "BEST_7_FEATURES必须包含7个特征"
    assert all(f in SELECTED_FEATURES for f in BEST_7_FEATURES), "BEST_7_FEATURES必须是SELECTED_FEATURES的子集"
    
    # 检查类别特征
    assert all(f in SELECTED_FEATURES for f in CAT_FEATURE_NAMES), "所有类别特征必须在SELECTED_FEATURES中"
    
    # 检查数据路径
    import os
    for dataset, path in DATA_PATHS.items():
        assert os.path.exists(path), f"数据集{dataset}的路径不存在: {path}"
    
    # 检查模型配置
    for model_type, config in MODEL_CONFIGS.items():
        if 'random_state' in config:
            assert isinstance(config['random_state'], int), f"{model_type}的random_state必须是整数"
    
    print("配置验证通过！")
```

### 配置兼容性检查
```python
def check_compatibility():
    """检查配置的兼容性"""
    # 检查设备兼容性
    import torch
    if not torch.cuda.is_available():
        for model_type in MODEL_CONFIGS:
            if 'device' in MODEL_CONFIGS[model_type]:
                if MODEL_CONFIGS[model_type]['device'] == 'cuda':
                    print(f"警告: {model_type}配置为CUDA但CUDA不可用")
    
    # 检查内存需求
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory_gb < 8:
            print("警告: GPU内存不足8GB，建议使用内存优化配置")
```

## 最佳实践

### 1. 配置管理策略
- **环境分离**: 为不同环境（开发、测试、生产）维护不同的配置
- **版本控制**: 重要配置变更需要记录和版本管理
- **参数验证**: 使用配置验证函数确保参数有效性

### 2. 性能优化配置
```python
# 高性能配置（充足资源）
HIGH_PERFORMANCE_CONFIG = {
    'model': {
        'max_time': 120,
        'phe_init_args': {
            'max_models': 25,
            'n_repeats': 200
        }
    },
    'mmd': {
        'n_epochs': 300,
        'batch_size': 128,
        'gamma_search_values': [0.001, 0.01, 0.05, 0.1, 0.5]
    }
}

# 快速配置（资源受限）
FAST_CONFIG = {
    'model': {
        'max_time': 15,
        'phe_init_args': {
            'max_models': 5,
            'n_repeats': 30
        }
    },
    'mmd': {
        'n_epochs': 100,
        'batch_size': 32,
        'gamma_search_values': [0.01, 0.1]
    }
}
```

### 3. 调试配置
```python
# 调试模式配置
DEBUG_CONFIG = {
    'save_intermediate_results': True,
    'verbose_logging': True,
    'monitor_gradients': True,
    'save_training_history': True
}
```

## 故障排除

### 常见配置问题

1. **特征索引错误**
   ```python
   # 问题: 类别特征索引超出范围
   # 解决: 确保索引在有效范围内
   features = get_features_by_type('best7')
   cat_indices = get_categorical_indices('best7')
   assert all(idx < len(features) for idx in cat_indices)
   ```

2. **设备配置不匹配**
   ```python
   # 问题: 配置CUDA但设备不支持
   # 解决: 动态检测设备能力
   import torch
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   MODEL_CONFIGS['auto']['device'] = device
   ```

3. **内存配置不当**
   ```python
   # 问题: 批大小过大导致内存不足
   # 解决: 根据可用内存动态调整
   if torch.cuda.is_available():
       memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
       if memory_gb < 8:
           MMD_METHODS['linear']['batch_size'] = 32
   ```

### 配置调试工具
```python
def debug_config():
    """配置调试工具"""
    print("=== 配置调试信息 ===")
    print(f"特征数量: {len(SELECTED_FEATURES)}")
    print(f"最佳特征: {BEST_7_FEATURES}")
    print(f"类别特征数量: {len(CAT_FEATURE_NAMES)}")
    
    for model_type, config in MODEL_CONFIGS.items():
        print(f"\n{model_type}模型配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    for method, config in MMD_METHODS.items():
        print(f"\n{method} MMD配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")
```

## 总结

`config/settings.py` 是项目的配置中枢，合理的配置管理可以：
- 提高实验效率和结果可重现性
- 适应不同的硬件环境和资源限制
- 支持灵活的功能组合和参数调优
- 简化部署和维护工作

建议在修改配置前备份原始设置，并通过小规模实验验证配置的有效性。 