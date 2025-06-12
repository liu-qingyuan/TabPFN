# modeling/tabpfn_runner.py 文档

## 概述

TabPFN专用运行器，提供TabPFN模型的训练、预测和评估功能。

## 主要类

### TabPFNRunner

```python
class TabPFNRunner:
    def __init__(self, model_type: str = 'auto', **kwargs):
```

TabPFN模型运行器。

**参数:**
- `model_type`: 模型类型 ('auto', 'tuned', 'base')

## 主要方法

### fit()

```python
def fit(self, X: np.ndarray, y: np.ndarray) -> 'TabPFNRunner':
```

训练TabPFN模型。

### predict()

```python
def predict(self, X: np.ndarray) -> np.ndarray:
```

进行预测。

### predict_proba()

```python
def predict_proba(self, X: np.ndarray) -> np.ndarray:
```

预测概率。

### evaluate()

```python
def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
```

评估模型性能。

## 主要函数

### run_tabpfn_experiment()

```python
def run_tabpfn_experiment(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         model_type: str = 'auto',
                         **kwargs) -> Dict[str, Any]:
```

运行TabPFN实验。

**使用示例:**
```python
from analytical_mmd_A2B_feature58.modeling.tabpfn_runner import run_tabpfn_experiment

results = run_tabpfn_experiment(
    X_train, y_train, X_test, y_test,
    model_type='auto'
)
```

### run_cross_validation_tabpfn()

```python
def run_cross_validation_tabpfn(X: np.ndarray, y: np.ndarray,
                                cv_folds: int = 10,
                                model_type: str = 'auto',
                                **kwargs) -> Dict[str, Any]:
```

运行TabPFN交叉验证。 