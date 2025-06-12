# modeling/model_selector.py 文档

## 概述

`modeling/model_selector.py` 是项目的模型选择和管理核心模块，提供统一的模型创建接口，支持多种TabPFN变体和配置管理。本模块简化了模型的选择、配置和实例化过程。

## 文件位置
```
analytical_mmd_A2B_feature58/modeling/model_selector.py
```

## 主要功能

### 1. 统一模型接口
- 支持多种TabPFN模型类型
- 统一的创建和配置接口
- 自动参数验证和优化
- 环境适应性检查

### 2. 模型类型管理
- AutoTabPFN（自动化集成）
- TunedTabPFN（超参数优化）
- 原生TabPFN（轻量级）
- RF风格集成（备用方案）

### 3. 配置管理
- 预设配置模板
- 动态参数调整
- 类别特征处理
- 设备兼容性管理

### 4. 错误处理和降级
- 优雅的模型降级机制
- 依赖检查和安装提示
- 兼容性验证
- 详细的错误报告

## 核心函数

### get_model()

```python
def get_model(
    model_type: str,
    categorical_feature_indices: Optional[List[int]] = None,
    preset: str = 'balanced',
    device: Optional[str] = None,
    **kwargs: Any
) -> Any:
    """
    获取指定类型的模型实例
    
    参数:
    - model_type (str): 模型类型，支持的值:
        - 'auto': AutoTabPFN - 自动化TabPFN集成
        - 'tuned': TunedTabPFN - 超参数优化TabPFN
        - 'base': 原生TabPFN - 轻量级实现
        - 'rf': RF风格TabPFN集成 - 随机森林风格集成
    - categorical_feature_indices (List[int], optional): 类别特征索引列表
    - preset (str): 预设配置类型
        - 'fast': 快速配置，优先速度
        - 'balanced': 平衡配置，速度和性能兼顾（默认）
        - 'accurate': 高精度配置，优先性能
    - device (str, optional): 计算设备 ('cuda', 'cpu')，None表示自动检测
    - **kwargs: 模型特定的额外参数
    
    返回:
    - 模型实例: 具有fit/predict/predict_proba方法的模型对象
    
    异常:
    - ValueError: 不支持的模型类型或参数错误
    - ImportError: 所需的包未安装
    - RuntimeError: 模型创建失败
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.modeling.model_selector import get_model

# 基础使用 - AutoTabPFN
model = get_model('auto', categorical_feature_indices=[0, 2])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# 使用预设配置
fast_model = get_model('auto', preset='fast')  # 快速配置
accurate_model = get_model('auto', preset='accurate')  # 高精度配置

# 自定义参数
custom_model = get_model(
    'auto',
    categorical_feature_indices=[0, 2],
    max_time=60,  # 增加训练时间
    phe_init_args={
        'max_models': 20,
        'n_repeats': 150
    }
)

# 指定设备
gpu_model = get_model('auto', device='cuda')
cpu_model = get_model('auto', device='cpu')
```

### get_available_models()

```python
def get_available_models() -> List[str]:
    """
    获取当前环境中可用的模型类型
    
    返回:
    - List[str]: 可用模型类型列表
    
    功能:
    - 检查依赖包的安装状态
    - 验证模型的可用性
    - 返回实际可用的模型列表
    
    注意:
    - 某些模型可能因为依赖缺失而不可用
    - 'rf'模型通常总是可用（使用sklearn作为后备）
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.modeling.model_selector import get_available_models

# 检查可用模型
available = get_available_models()
print(f"可用模型: {available}")

# 根据可用性选择模型
if 'auto' in available:
    model = get_model('auto')
elif 'tuned' in available:
    model = get_model('tuned')
else:
    model = get_model('rf')  # 备用方案
```

### validate_model_params()

```python
def validate_model_params(
    model_type: str, 
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    验证和标准化模型参数
    
    参数:
    - model_type (str): 模型类型
    - params (Dict[str, Any]): 原始参数字典
    
    返回:
    - Dict[str, Any]: 验证和标准化后的参数字典
    
    功能:
    - 参数类型检查和转换
    - 参数范围验证
    - 默认值填充
    - 兼容性检查
    
    异常:
    - ValueError: 参数值无效
    - TypeError: 参数类型错误
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.modeling.model_selector import validate_model_params

# 验证参数
raw_params = {
    'max_time': '30',  # 字符串会被转换为整数
    'device': 'cuda',
    'random_state': 42
}

validated_params = validate_model_params('auto', raw_params)
print(f"验证后参数: {validated_params}")

# 使用验证后的参数创建模型
model = get_model('auto', **validated_params)
```

### get_model_info()

```python
def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    获取模型类型的详细信息
    
    参数:
    - model_type (str): 模型类型
    
    返回:
    - Dict[str, Any]: 模型信息字典，包含:
        - 'name': 模型名称
        - 'description': 模型描述
        - 'supported_features': 支持的特征类型
        - 'requirements': 依赖要求
        - 'performance_characteristics': 性能特征
        - 'recommended_use_cases': 推荐使用场景
    """
```

## 模型类型详解

### AutoTabPFN ('auto')

```python
def create_auto_tabpfn(
    categorical_feature_indices: Optional[List[int]] = None,
    max_time: int = 30,
    preset: str = 'balanced',
    **kwargs
):
    """
    创建AutoTabPFN模型实例
    
    特点:
    - 自动化机器学习
    - 集成多个TabPFN模型
    - 自动超参数优化
    - 高性能表现
    
    适用场景:
    - 生产环境部署
    - 高性能需求
    - 自动化流程
    """
```

#### 配置参数

```python
AUTO_TABPFN_CONFIGS = {
    'fast': {
        'max_time': 15,
        'phe_init_args': {
            'max_models': 5,
            'n_repeats': 30,
            'validation_method': 'holdout'
        }
    },
    'balanced': {
        'max_time': 30,
        'phe_init_args': {
            'max_models': 15,
            'n_repeats': 100,
            'validation_method': 'cv'
        }
    },
    'accurate': {
        'max_time': 120,
        'phe_init_args': {
            'max_models': 25,
            'n_repeats': 200,
            'validation_method': 'cv'
        }
    }
}
```

### TunedTabPFN ('tuned')

```python
def create_tuned_tabpfn(
    categorical_feature_indices: Optional[List[int]] = None,
    optimization_method: str = 'bayesian',
    n_trials: int = 50,
    **kwargs
):
    """
    创建TunedTabPFN模型实例
    
    特点:
    - 贝叶斯优化超参数
    - 平衡性能和训练时间
    - 提供优化历史
    - 适应性强
    
    适用场景:
    - 研究和开发
    - 参数调优
    - 中等规模数据集
    """
```

### 原生TabPFN ('base')

```python
def create_base_tabpfn(
    categorical_feature_indices: Optional[List[int]] = None,
    device: str = 'cuda',
    **kwargs
):
    """
    创建原生TabPFN模型实例
    
    特点:
    - 轻量级实现
    - 快速训练和推理
    - 内存友好
    - 简单易用
    
    限制:
    - 样本数量 ≤ 1000
    - 特征数量 ≤ 100
    
    适用场景:
    - 小数据集
    - 快速验证
    - 原型开发
    """
```

### RF风格集成 ('rf')

```python
def create_rf_tabpfn(
    categorical_feature_indices: Optional[List[int]] = None,
    n_estimators: int = 10,
    use_tabpfn: bool = True,
    **kwargs
):
    """
    创建RF风格TabPFN集成模型
    
    特点:
    - 随机森林风格集成
    - 鲁棒性强
    - 自动降级机制
    - 备用方案
    
    降级策略:
    - TabPFN可用时使用TabPFN集成
    - TabPFN不可用时使用RandomForest
    
    适用场景:
    - 备用方案
    - 鲁棒性需求
    - 环境不确定
    """
```

## 高级功能

### 模型比较和选择

#### compare_models()

```python
def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_types: List[str] = None,
    categorical_feature_indices: List[int] = None,
    cv_folds: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    比较多个模型的性能
    
    参数:
    - X_train, y_train: 训练数据
    - X_test, y_test: 测试数据
    - model_types: 要比较的模型类型列表
    - categorical_feature_indices: 类别特征索引
    - cv_folds: 交叉验证折数
    
    返回:
    - Dict[str, Dict[str, float]]: 各模型的性能指标
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.modeling.model_selector import compare_models

# 比较所有可用模型
results = compare_models(
    X_train, y_train, X_test, y_test,
    model_types=['auto', 'tuned', 'base', 'rf'],
    categorical_feature_indices=[0, 2],
    cv_folds=5
)

# 查看结果
for model_type, metrics in results.items():
    print(f"{model_type}模型:")
    print(f"  测试AUC: {metrics['test_auc']:.4f}")
    print(f"  CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    print(f"  训练时间: {metrics['fit_time']:.2f}秒")
```

### 自动模型选择

#### auto_select_model()

```python
def auto_select_model(
    X: np.ndarray,
    y: np.ndarray,
    categorical_feature_indices: List[int] = None,
    criteria: str = 'balanced',
    time_budget: float = 300.0
) -> Tuple[Any, str, Dict[str, Any]]:
    """
    根据数据特征自动选择最适合的模型
    
    参数:
    - X, y: 训练数据
    - categorical_feature_indices: 类别特征索引
    - criteria: 选择标准 ('speed', 'accuracy', 'balanced')
    - time_budget: 时间预算（秒）
    
    返回:
    - Tuple[模型实例, 模型类型, 选择信息]
    
    选择逻辑:
    - 根据数据规模选择合适的模型
    - 考虑时间预算限制
    - 平衡性能和效率
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.modeling.model_selector import auto_select_model

# 自动选择模型
model, model_type, selection_info = auto_select_model(
    X_train, y_train,
    categorical_feature_indices=[0, 2],
    criteria='balanced',
    time_budget=180.0  # 3分钟预算
)

print(f"选择的模型: {model_type}")
print(f"选择原因: {selection_info['reason']}")
print(f"预期性能: {selection_info['expected_performance']}")

# 使用选择的模型
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 模型配置优化

#### optimize_model_config()

```python
def optimize_model_config(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    categorical_feature_indices: List[int] = None,
    optimization_target: str = 'auc',
    n_trials: int = 20
) -> Dict[str, Any]:
    """
    优化模型配置参数
    
    参数:
    - model_type: 模型类型
    - X, y: 训练数据
    - categorical_feature_indices: 类别特征索引
    - optimization_target: 优化目标 ('auc', 'accuracy', 'f1')
    - n_trials: 优化试验次数
    
    返回:
    - Dict[str, Any]: 优化后的配置参数
    """
```

## 错误处理和降级机制

### 优雅降级

```python
def get_model_with_fallback(
    model_type: str,
    categorical_feature_indices: List[int] = None,
    **kwargs
) -> Tuple[Any, str]:
    """
    带降级机制的模型获取
    
    降级顺序:
    1. 尝试创建指定模型
    2. 如果失败，尝试更简单的模型
    3. 最终降级到RandomForest
    
    返回:
    - Tuple[模型实例, 实际使用的模型类型]
    """
    
    fallback_chain = {
        'auto': ['tuned', 'base', 'rf'],
        'tuned': ['base', 'rf'],
        'base': ['rf'],
        'rf': []
    }
    
    # 尝试创建指定模型
    try:
        model = get_model(model_type, categorical_feature_indices, **kwargs)
        return model, model_type
    except Exception as e:
        print(f"创建{model_type}模型失败: {e}")
        
        # 尝试降级
        for fallback_type in fallback_chain.get(model_type, []):
            try:
                model = get_model(fallback_type, categorical_feature_indices, **kwargs)
                print(f"降级到{fallback_type}模型")
                return model, fallback_type
            except Exception:
                continue
        
        # 最终降级到sklearn RandomForest
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42)
        print("降级到RandomForest")
        return model, 'sklearn_rf'
```

### 依赖检查

```python
def check_dependencies() -> Dict[str, bool]:
    """
    检查各模型类型的依赖状态
    
    返回:
    - Dict[str, bool]: 各依赖的可用状态
    """
    
    dependencies = {}
    
    # 检查AutoTabPFN
    try:
        import tabpfn_extensions
        dependencies['auto'] = True
    except ImportError:
        dependencies['auto'] = False
    
    # 检查TunedTabPFN
    try:
        import optuna
        dependencies['tuned'] = True
    except ImportError:
        dependencies['tuned'] = False
    
    # 检查原生TabPFN
    try:
        import tabpfn
        dependencies['base'] = True
    except ImportError:
        dependencies['base'] = False
    
    # sklearn总是可用
    dependencies['rf'] = True
    
    return dependencies
```

## 性能监控和分析

### 模型性能分析

```python
def analyze_model_performance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str
) -> Dict[str, Any]:
    """
    分析模型性能
    
    返回:
    - Dict[str, Any]: 性能分析结果
    """
    
    import time
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    
    # 预测时间
    start_time = time.time()
    predictions = model.predict(X_test)
    predict_time = time.time() - start_time
    
    # 概率预测时间
    start_time = time.time()
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)
        proba_time = time.time() - start_time
    else:
        probabilities = None
        proba_time = None
    
    # 计算指标
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    
    auc = None
    if probabilities is not None:
        if probabilities.shape[1] == 2:  # 二分类
            auc = roc_auc_score(y_test, probabilities[:, 1])
        else:  # 多分类
            auc = roc_auc_score(y_test, probabilities, multi_class='ovr')
    
    return {
        'model_type': model_type,
        'accuracy': accuracy,
        'f1_score': f1,
        'auc': auc,
        'predict_time': predict_time,
        'proba_time': proba_time,
        'samples_per_second': len(X_test) / predict_time
    }
```

## 使用示例和最佳实践

### 完整的模型选择流程

```python
from analytical_mmd_A2B_feature58.modeling.model_selector import (
    get_available_models, get_model_with_fallback, analyze_model_performance
)

def complete_model_selection_workflow(X_train, y_train, X_test, y_test, 
                                    categorical_feature_indices=None):
    """完整的模型选择工作流程"""
    
    # 1. 检查可用模型
    available_models = get_available_models()
    print(f"可用模型: {available_models}")
    
    # 2. 选择首选模型类型
    if 'auto' in available_models:
        preferred_model = 'auto'
    elif 'tuned' in available_models:
        preferred_model = 'tuned'
    else:
        preferred_model = 'rf'
    
    # 3. 创建模型（带降级机制）
    model, actual_model_type = get_model_with_fallback(
        preferred_model,
        categorical_feature_indices=categorical_feature_indices
    )
    
    print(f"使用模型: {actual_model_type}")
    
    # 4. 训练模型
    print("训练模型...")
    start_time = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_time
    print(f"训练时间: {fit_time:.2f}秒")
    
    # 5. 性能分析
    performance = analyze_model_performance(model, X_test, y_test, actual_model_type)
    
    print("性能分析:")
    print(f"  准确率: {performance['accuracy']:.4f}")
    print(f"  F1分数: {performance['f1_score']:.4f}")
    if performance['auc']:
        print(f"  AUC: {performance['auc']:.4f}")
    print(f"  预测速度: {performance['samples_per_second']:.0f} 样本/秒")
    
    return model, performance

# 使用示例
model, performance = complete_model_selection_workflow(
    X_train, y_train, X_test, y_test,
    categorical_feature_indices=[0, 2]
)
```

### 批量模型实验

```python
def batch_model_experiment(datasets, categorical_indices_list):
    """批量模型实验"""
    
    results = []
    
    for i, (X_train, y_train, X_test, y_test) in enumerate(datasets):
        print(f"\n实验 {i+1}/{len(datasets)}")
        
        cat_indices = categorical_indices_list[i] if i < len(categorical_indices_list) else None
        
        # 测试所有可用模型
        available_models = get_available_models()
        
        experiment_results = {}
        
        for model_type in available_models:
            try:
                print(f"  测试 {model_type} 模型...")
                
                model = get_model(model_type, categorical_feature_indices=cat_indices)
                
                # 训练和评估
                start_time = time.time()
                model.fit(X_train, y_train)
                fit_time = time.time() - start_time
                
                performance = analyze_model_performance(model, X_test, y_test, model_type)
                performance['fit_time'] = fit_time
                
                experiment_results[model_type] = performance
                
            except Exception as e:
                print(f"    {model_type} 模型失败: {e}")
                experiment_results[model_type] = {'error': str(e)}
        
        results.append({
            'experiment_id': i,
            'results': experiment_results
        })
    
    return results
```

## 故障排除

### 常见问题

1. **模型创建失败**
   ```python
   # 问题: 依赖包未安装
   # 解决: 检查依赖并提供安装指导
   try:
       model = get_model('auto')
   except ImportError as e:
       print(f"依赖缺失: {e}")
       print("请安装: pip install tabpfn-extensions")
   ```

2. **内存不足**
   ```python
   # 问题: 模型内存需求过大
   # 解决: 使用更轻量的模型
   try:
       model = get_model('auto', max_time=60)
   except RuntimeError as e:
       if "memory" in str(e).lower():
           print("内存不足，切换到轻量模型")
           model = get_model('base')
   ```

3. **数据规模限制**
   ```python
   # 问题: 数据超出模型限制
   # 解决: 自动选择合适的模型
   n_samples, n_features = X_train.shape
   
   if n_samples > 1000 or n_features > 100:
       if 'auto' in get_available_models():
           model = get_model('auto')
       else:
           model = get_model('rf')
   else:
       model = get_model('base')
   ```

## 总结

`modeling/model_selector.py` 模块提供了完整的模型选择和管理功能，主要特点包括：

- **统一接口**: 简化了多种模型的使用
- **智能选择**: 根据数据特征自动选择合适的模型
- **鲁棒性**: 完善的错误处理和降级机制
- **灵活配置**: 支持多种预设和自定义配置
- **性能监控**: 提供详细的性能分析功能

通过合理使用这些功能，可以大大简化模型选择过程，提高实验效率和结果的可靠性。 