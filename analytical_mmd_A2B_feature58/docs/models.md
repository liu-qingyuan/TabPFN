# 模型文档 (Models Documentation)

## 概述

本项目支持多种TabPFN模型类型，提供统一的接口和灵活的配置选项。本文档详细介绍每种模型的特点、使用方法和最佳实践。

## 文件结构

```python
modeling/
├── model_selector.py           # 统一模型选择器
├── cross_domain_runner.py      # 跨域实验运行器
└── __init__.py                # 模块初始化
```

## 支持的模型类型

### 1. AutoTabPFN (`auto`)

#### 特点
AutoTabPFN是最先进的自动化表格数据预测模型，结合了TabPFN的强大能力和自动化机器学习的便利性。

**核心优势**：
- 🤖 **全自动化**：无需手动调参，自动优化超参数
- 🎯 **高性能**：在大多数表格数据任务上表现优异
- 🔧 **可配置**：支持多种集成策略和验证方法
- ⚡ **GPU加速**：充分利用GPU资源进行训练

#### 配置参数

```python
MODEL_CONFIGS['auto'] = {
    'max_time': 30,                    # 最大训练时间(秒)
    'preset': 'default',               # 预设配置
    'ges_scoring_string': 'accuracy',  # 集成评分方法
    'device': 'cuda',                  # 计算设备
    'random_state': 42,                # 随机种子
    'ignore_pretraining_limits': False, # 是否忽略预训练限制
    'categorical_feature_indices': None, # 类别特征索引
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

#### 关键参数详解

**基础参数**：
- `max_time`: 控制总体训练时间上限
- `preset`: 预设配置（'fast', 'default', 'high_quality'）
- `ges_scoring_string`: 集成评分指标
  - `'accuracy'`: 准确率（分类任务）
  - `'auc'`: ROC-AUC（二分类任务）
  - `'f1'`: F1分数（不平衡数据）

**PostHocEnsemble参数**：
- `max_models`: 集成中包含的最大模型数量
- `validation_method`: 模型验证策略
  - `'cv'`: 交叉验证
  - `'holdout'`: 保留集验证
  - `'none'`: 无验证
- `n_repeats`: 验证重复次数，影响结果稳定性

#### 使用示例

```python
from analytical_mmd_A2B_feature58.modeling.model_selector import get_model

# 基础使用
model = get_model('auto', categorical_feature_indices=[0, 2])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 自定义配置
model = get_model(
    'auto',
    categorical_feature_indices=[0, 2],
    max_time=60,
    phe_init_args={
        'max_models': 20,
        'validation_method': 'cv',
        'n_repeats': 150
    }
)
```

#### 性能调优指南

**高性能配置**：
```python
high_performance_config = {
    'max_time': 120,
    'phe_init_args': {
        'max_models': 25,
        'n_repeats': 200,
        'validation_method': 'cv'
    }
}
```

**快速配置**：
```python
fast_config = {
    'max_time': 15,
    'phe_init_args': {
        'max_models': 8,
        'n_repeats': 50,
        'validation_method': 'holdout'
    }
}
```

**内存优化配置**：
```python
memory_optimized_config = {
    'max_time': 30,
    'phe_init_args': {
        'max_models': 10,
        'n_repeats': 80,
        'holdout_fraction': 0.3
    }
}
```

### 2. TunedTabPFN (`tuned`)

#### 特点
TunedTabPFN通过贝叶斯优化等方法自动搜索最佳超参数配置。

**核心优势**：
- 🎯 **智能调优**：自动超参数搜索
- ⚖️ **平衡性能**：在性能和训练时间间取得平衡
- 📊 **适应性强**：能适应不同规模和类型的数据集
- 🔍 **透明度高**：提供调优过程的详细信息

#### 配置参数

```python
MODEL_CONFIGS['tuned'] = {
    'random_state': 42,                # 随机种子
    'categorical_feature_indices': None # 类别特征索引
}
```

#### 使用示例

```python
# 基础使用
model = get_model('tuned', categorical_feature_indices=[0, 2])
model.fit(X_train, y_train)

# 获取调优历史
if hasattr(model, 'optimization_history_'):
    print("最佳参数:", model.best_params_)
    print("最佳分数:", model.best_score_)
```

#### 适用场景
- 对性能要求较高但计算资源有限
- 需要了解最优参数配置
- 数据集规模中等（100-10000样本）

### 3. 原生TabPFN (`base`)

#### 特点
官方原版TabPFN实现，轻量级且训练速度快。

**核心优势**：
- ⚡ **速度快**：训练和推理速度极快
- 💡 **简单易用**：参数配置简单
- 📏 **内存友好**：内存占用小
- 🎓 **经典可靠**：基于原始论文实现

#### 限制条件
- 样本数量：≤ 1000
- 特征数量：≤ 100
- 仅支持数值和类别特征

#### 配置参数

```python
MODEL_CONFIGS['base'] = {
    'device': 'cuda',                  # 计算设备
    'random_state': 42,                # 随机种子
    'categorical_feature_indices': None # 类别特征索引
}
```

#### 使用示例

```python
# 基础使用
model = get_model('base', categorical_feature_indices=[0, 2])
model.fit(X_train, y_train)

# CPU模式
model = get_model('base', device='cpu')
model.fit(X_train, y_train)
```

#### 适用场景
- 小规模数据集快速验证
- 原型开发和概念验证
- 计算资源受限的环境
- 需要最快速度的场景

### 4. RF风格TabPFN集成 (`rf`)

#### 特点
随机森林风格的TabPFN集成，结合了随机森林的集成思想和TabPFN的预测能力。

**核心优势**：
- 🌳 **集成优势**：多个TabPFN模型集成
- 🎲 **随机性**：Bootstrap采样和特征子集
- 🛡️ **鲁棒性**：对异常值和噪声不敏感
- 🔄 **备用方案**：TabPFN不可用时自动降级为随机森林

#### 配置参数

```python
MODEL_CONFIGS['rf'] = {
    'n_estimators': 10,                # 基础模型数量
    'max_depth': None,                 # 最大深度（随机森林备用）
    'random_state': 42,                # 随机种子
    'n_jobs': -1                       # 并行作业数
}
```

#### 使用示例

```python
# TabPFN集成模式
model = get_model('rf', n_estimators=15)
model.fit(X_train, y_train)

# 随机森林备用模式（当TabPFN不可用时）
model = get_model('rf', 
                  n_estimators=100, 
                  max_depth=10)
model.fit(X_train, y_train)
```

#### 适用场景
- 需要高鲁棒性的任务
- 数据质量不确定的情况
- TabPFN环境配置困难时的备用方案
- 集成学习爱好者

## 模型选择器 (`model_selector.py`)

### 核心函数

#### get_model()
```python
def get_model(model_type: str, categorical_feature_indices: Optional[List[int]] = None, 
              **kwargs: Any):
    """
    获取指定类型的模型
    
    参数:
    - model_type: 模型类型 ('auto', 'tuned', 'base', 'rf')
    - categorical_feature_indices: 类别特征索引
    - **kwargs: 模型特定参数
    
    返回:
    - 模型实例
    """
```

#### get_available_models()
```python
def get_available_models() -> List[str]:
    """
    获取当前环境中可用的模型类型
    
    返回:
    - List[str]: 可用模型类型列表
    """
```

#### validate_model_params()
```python
def validate_model_params(model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证和标准化模型参数
    
    参数:
    - model_type: 模型类型
    - params: 原始参数字典
    
    返回:
    - Dict[str, Any]: 验证后的参数字典
    """
```

### 使用示例

```python
from analytical_mmd_A2B_feature58.modeling.model_selector import (
    get_model, get_available_models, validate_model_params
)

# 检查可用模型
available_models = get_available_models()
print(f"可用模型: {available_models}")

# 参数验证
params = {'max_time': 60, 'device': 'cuda'}
validated_params = validate_model_params('auto', params)

# 创建模型
model = get_model('auto', **validated_params)
```

## 跨域实验运行器 (`cross_domain_runner.py`)

### CrossDomainExperimentRunner类

#### 初始化参数
```python
def __init__(self, 
             model_type: str = 'auto',
             feature_type: str = 'best7',
             use_mmd_adaptation: bool = True,
             mmd_method: str = 'linear',
             use_class_conditional: bool = False,
             use_threshold_optimizer: bool = False,
             save_path: str = './results_cross_domain',
             skip_cv_on_a: bool = False,
             evaluation_mode: str = 'cv',
             **kwargs: Any):
```

#### 核心方法

**数据加载**：
```python
def load_datasets(self) -> Dict[str, np.ndarray]:
    """加载所有数据集并进行预处理"""
```

**交叉验证**：
```python
def run_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                        cv_folds: int = 10) -> Dict[str, Any]:
    """在源域数据上运行交叉验证"""
```

**外部验证**：
```python
def evaluate_external_dataset_cv(self, X_train, y_train, X_test, y_test,
                                X_train_raw, X_test_raw, dataset_name):
    """在目标域数据上进行外部验证"""
```

### 使用示例

```python
from analytical_mmd_A2B_feature58.modeling.cross_domain_runner import CrossDomainExperimentRunner

# 创建实验运行器
runner = CrossDomainExperimentRunner(
    model_type='auto',
    feature_type='best7',
    use_mmd_adaptation=True,
    mmd_method='linear',
    save_path='./my_experiment_results'
)

# 运行完整实验
results = runner.run_full_experiment()

# 查看结果
print(f"源域CV AUC: {results['cross_validation_A']['auc']}")
print(f"目标域AUC: {results['external_validation_B']['without_domain_adaptation']['auc']}")
```

## 模型比较

### 性能对比

| 模型类型 | 训练速度 | 预测准确率 | 内存使用 | 适用数据规模 | 推荐场景 |
|----------|----------|------------|----------|-------------|----------|
| AutoTabPFN | 中等 | 极高 | 高 | 大 | 生产环境 |
| TunedTabPFN | 慢 | 高 | 中等 | 中-大 | 研究开发 |
| 原生TabPFN | 快 | 高 | 低 | 小 | 快速验证 |
| RF集成 | 中等 | 中-高 | 中等 | 任意 | 备用方案 |
