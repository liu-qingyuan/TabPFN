# modeling/cross_domain_runner.py 文档

## 概述

跨域实验运行器，整合多模型支持、MMD域适应和跨数据集评估功能。

## 主要类

### CrossDomainExperimentRunner

跨域实验运行器主类。

```python
class CrossDomainExperimentRunner:
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

**参数:**
- `model_type`: 模型类型 ('auto', 'base', 'rf')
- `feature_type`: 特征类型 ('all', 'best7')
- `use_mmd_adaptation`: 是否使用MMD域适应
- `mmd_method`: MMD方法 ('linear', 'kpca', 'mean_std')
- `use_class_conditional`: 是否使用类条件MMD
- `save_path`: 结果保存路径
- `evaluation_mode`: 评估模式 ('single', 'cv', 'proper_cv')

## 主要方法

### load_datasets()

```python
def load_datasets(self) -> Dict[str, np.ndarray]:
```

加载所有数据集并进行标准化。

**返回:**
- 包含所有数据集的字典

### run_cross_validation()

```python
def run_cross_validation(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 10) -> Dict[str, Any]:
```

在数据集上运行交叉验证。

**参数:**
- `X`: 特征数据
- `y`: 标签数据
- `cv_folds`: 交叉验证折数

**返回:**
- 交叉验证结果

### run_domain_adaptation()

```python
def run_domain_adaptation(self, X_source: np.ndarray, y_source: np.ndarray, 
                        X_target: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
```

运行域适应。

**参数:**
- `X_source`: 源域特征
- `y_source`: 源域标签
- `X_target`: 目标域特征

**返回:**
- `X_target_aligned`: 适应后的目标域特征
- `adaptation_info`: 适应信息

### evaluate_external_dataset_cv()

```python
def evaluate_external_dataset_cv(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               X_train_raw: np.ndarray, X_test_raw: np.ndarray,
                               dataset_name: str, cv_folds: int = 10) -> Dict[str, Any]:
```

使用交叉验证评估外部数据集。

### run_full_experiment()

```python
def run_full_experiment(self) -> Dict[str, Any]:
```

运行完整的跨域实验。

**返回:**
- 完整的实验结果

## 主要函数

### run_cross_domain_experiment()

```python
def run_cross_domain_experiment(model_type: str = 'auto',
                               feature_type: str = 'best7',
                               mmd_method: str = 'linear',
                               use_class_conditional: bool = False,
                               use_threshold_optimizer: bool = False,
                               save_path: str = './results_cross_domain',
                               skip_cv_on_a: bool = False,
                               evaluation_mode: str = 'cv',
                               **kwargs: Any) -> Dict[str, Any]:
```

运行跨域实验的便捷函数。

**使用示例:**
```python
from analytical_mmd_A2B_feature58.modeling.cross_domain_runner import run_cross_domain_experiment

# 运行基础实验
results = run_cross_domain_experiment(
    model_type='auto',
    feature_type='best7',
    mmd_method='linear',
    use_mmd_adaptation=True
)

# 运行类条件MMD实验
results = run_cross_domain_experiment(
    model_type='auto',
    mmd_method='linear',
    use_class_conditional=True,
    save_path='./results_class_conditional'
)
```

## 使用示例

### 完整实验流程

```python
from analytical_mmd_A2B_feature58.modeling.cross_domain_runner import CrossDomainExperimentRunner

# 创建实验运行器
runner = CrossDomainExperimentRunner(
    model_type='auto',
    feature_type='best7',
    use_mmd_adaptation=True,
    mmd_method='linear',
    save_path='./my_results'
)

# 运行完整实验
results = runner.run_full_experiment()

# 保存结果
runner.save_results()
``` 