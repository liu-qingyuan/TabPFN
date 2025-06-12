# 配置文档 (Configuration Documentation)

## 概述

`config/settings.py` 是项目的配置管理中心，包含所有重要的常量、参数和配置项。本文档详细介绍每个配置组的含义和使用方法。

## 文件结构

```python
config/
└── settings.py    # 全局配置设置
```

## 核心配置组

### 1. 特征配置 (Feature Configuration)

#### 全局特征列表
```python
SELECTED_FEATURES = [
    'Feature1', 'Feature2', ..., 'Feature63'  # 58个精选特征
]
```
- **作用**: 定义项目使用的58个核心医疗特征
- **来源**: 基于特征选择算法从原始数据集中筛选
- **用途**: 数据加载、模型训练、结果分析

#### 最佳特征组合
```python
BEST_7_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]
```
- **作用**: 经过实验验证的最优7特征组合
- **性能**: 在跨域任务中表现最佳
- **推荐**: 小样本或快速实验时使用

#### 类别特征配置
```python
CAT_FEATURE_NAMES = [
    'Feature1', 'Feature3', ..., 'Feature63'  # 类别特征名称
]
BEST_7_CAT_FEATURES = ['Feature63', 'Feature46']  # 最佳7特征中的类别特征
```
- **作用**: 标识需要特殊处理的类别特征
- **用途**: 模型训练时的categorical_feature_indices参数

### 2. 模型配置 (Model Configuration)

#### AutoTabPFN配置
```python
MODEL_CONFIGS = {
    'auto': {
        'max_time': 30,                    # 最大训练时间(秒)
        'preset': 'default',               # 预设配置
        'ges_scoring_string': 'accuracy',  # 集成评分方法
        'device': 'cuda',                  # 计算设备
        'random_state': 42,                # 随机种子
        'ignore_pretraining_limits': False, # 是否忽略预训练限制
        'phe_init_args': {                 # PostHocEnsemble初始化参数
            'max_models': 15,              # 最大模型数量
            'validation_method': 'cv',     # 验证方法
            'n_repeats': 100,              # 重复次数
            'n_folds': 10,                 # 折数
            'holdout_fraction': 0.4,       # 保留集比例
            'ges_n_iterations': 20         # 集成搜索迭代次数
        }
    }
}
```

**参数详解**:
- `max_time`: 控制单个模型的最大训练时间
- `ges_scoring_string`: 可选值 'accuracy', 'auc', 'f1'
- `phe_init_args`: PostHocEnsemble的详细配置
  - `max_models`: 集成中的最大模型数量
  - `validation_method`: 'cv', 'holdout', 'none'
  - `n_repeats`: 交叉验证或holdout的重复次数

#### 其他模型配置
```python
MODEL_CONFIGS = {
    'tuned': {                           # 超参数优化TabPFN
        'random_state': 42
    },
    'base': {                           # 原生TabPFN
        'device': 'cuda',
        'random_state': 42
    },
    'rf': {                             # 随机森林(备用)
        'n_estimators': 10,
        'max_depth': None,
        'random_state': 42,
        'n_jobs': -1
    }
}
```

### 3. MMD方法配置 (MMD Methods Configuration)

#### 线性MMD配置
```python
MMD_METHODS = {
    'linear': {
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
}
```

**关键参数说明**:
- `staged_training`: 先用小gamma训练，再用大gamma微调
- `dynamic_gamma`: 自动搜索最优gamma值
- `gamma_search_values`: gamma候选值列表
- `use_gradient_clipping`: 防止梯度爆炸

#### 核PCA MMD配置
```python
'kpca': {
    'kernel': 'rbf',                       # 核函数类型
    'gamma': 0.05,                        # RBF核参数
    'n_components': 10,                   # 主成分数量
    'use_inverse_transform': False,       # 是否使用逆变换
    'standardize': True                   # 是否标准化
}
```

#### 均值标准差对齐
```python
'mean_std': {
    # 简单的均值标准差对齐，无需额外参数
}
```

### 4. 数据路径配置 (Data Paths)

```python
DATA_PATHS = {
    'A': "/home/24052432g/TabPFN/data/AI4healthcare.xlsx",
    'B': "/home/24052432g/TabPFN/data/HenanCancerHospital_features63_58.xlsx"
}

LABEL_COL = "Label"  # 标签列名
```

### 5. 可视化配置 (Visualization Configuration)

```python
VISUALIZATION_CONFIG = {
    'tsne_perplexity': 30,               # t-SNE困惑度
    'tsne_n_iter': 1000,                # t-SNE迭代次数
    'histogram_bins': 30,               # 直方图箱数
    'figure_dpi': 300,                  # 图片分辨率
    'n_features_to_plot': 12           # 直方图显示特征数
}
```

### 6. 实验配置 (Experiment Configuration)

```python
EXPERIMENT_CONFIG = {
    'test_size': 0.2,                   # 验证集比例
    'random_state': 42,                 # 随机种子
    'optimize_threshold': True,         # 阈值优化
    'cross_validation': False,          # 交叉验证
    'save_visualizations': True,        # 保存可视化
    'save_models': False               # 保存模型
}

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
    - feature_type: 'all', 'best7', 'categorical'
    
    返回:
    - List[str]: 特征名称列表
    """
```

### get_categorical_indices()
```python
def get_categorical_indices(feature_type: str = 'all') -> List[int]:
    """
    根据特征类型获取类别特征索引
    
    参数:
    - feature_type: 'all', 'best7'
    
    返回:
    - List[int]: 类别特征索引列表
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
    - preset: 'fast', 'balanced', 'accurate'
    - categorical_feature_indices: 类别特征索引
    
    返回:
    - Dict[str, Any]: 模型配置字典
    """
```

## 配置自定义指南

### 修改特征配置
```python
# 添加新的特征组合
CUSTOM_FEATURES = ['Feature1', 'Feature5', 'Feature10']

# 修改最佳特征
BEST_7_FEATURES = ['Feature63', 'Feature2', 'Feature46', ...]
```

### 调整模型参数
```python
# 提高AutoTabPFN性能
MODEL_CONFIGS['auto'].update({
    'max_time': 60,  # 增加训练时间
    'phe_init_args': {
        'max_models': 20,    # 增加模型数量
        'n_repeats': 150     # 增加重复次数
    }
})
```

### 优化MMD参数
```python
# 线性MMD调优
MMD_METHODS['linear'].update({
    'n_epochs': 300,         # 增加训练轮数
    'lr': 1e-4,             # 降低学习率
    'lambda_reg': 1e-2,     # 增加正则化
    'batch_size': 32        # 减少批大小(内存不足时)
})
```

### 环境适应配置
```python
# GPU内存不足时
MMD_METHODS['linear']['batch_size'] = 32
MODEL_CONFIGS['auto']['max_time'] = 15

# CPU环境
MODEL_CONFIGS['auto']['device'] = 'cpu'
MODEL_CONFIGS['base']['device'] = 'cpu'
```

## 最佳实践

### 1. 特征选择策略
- **小数据集**: 使用 `feature_type='best7'`
- **大数据集**: 使用 `feature_type='all'`
- **快速验证**: 使用 `feature_type='best7'`

### 2. 模型选择指南
- **高性能需求**: `model_type='auto'`
- **快速实验**: `model_type='base'` 或 `model_type='rf'`
- **参数优化**: `model_type='tuned'`

### 3. MMD方法选择
- **平衡性能**: `method='linear'`
- **非线性关系**: `method='kpca'`
- **快速对齐**: `method='mean_std'`

### 4. 资源管理
```python
# 内存优化配置
MEMORY_OPTIMIZED_CONFIG = {
    'batch_size': 32,
    'n_components': 5,  # KPCA
    'max_models': 8,    # AutoTabPFN
    'save_visualizations': False
}

# 高性能配置
HIGH_PERFORMANCE_CONFIG = {
    'batch_size': 128,
    'n_components': 15,
    'max_models': 25,
    'n_repeats': 200
}
```

## 故障排除

### 常见配置问题

1. **特征索引错误**
   ```python
   # 确保类别特征索引在有效范围内
   assert all(idx < len(features) for idx in cat_indices)
   ```

2. **模型参数冲突**
   ```python
   # 避免设备不匹配
   if not torch.cuda.is_available():
       MODEL_CONFIGS['auto']['device'] = 'cpu'
   ```

3. **内存不足**
   ```python
   # 动态调整批大小
   if torch.cuda.get_device_properties(0).total_memory < 8e9:  # <8GB
       MMD_METHODS['linear']['batch_size'] = 32
   ```

### 配置验证
```python
def validate_config():
    """验证配置的有效性"""
    # 检查特征配置
    assert len(BEST_7_FEATURES) == 7
    assert all(f in SELECTED_FEATURES for f in BEST_7_FEATURES)
    
    # 检查类别特征
    assert all(f in SELECTED_FEATURES for f in CAT_FEATURE_NAMES)
    
    # 检查路径
    assert all(os.path.exists(path) for path in DATA_PATHS.values())
```

## 总结

配置管理是项目的核心，合理的配置可以：
- 提高实验效率
- 确保结果可重现
- 适应不同硬件环境
- 支持灵活的功能组合

建议在修改配置前备份原始设置，并通过小规模实验验证配置的有效性。 