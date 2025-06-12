# preprocessing/threshold_optimizer.py 文档

## 概述

阈值优化器，用于自动寻找分类任务的最佳决策阈值，提高分类性能。

## 主要函数

### optimize_threshold_youden()

```python
def optimize_threshold_youden(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
```

使用Youden指数寻找最佳决策阈值。

**参数:**
- `y_true`: 真实标签
- `y_proba`: 预测为正类的概率

**返回:**
- `optimal_threshold`: 最佳决策阈值
- `optimal_metrics`: 性能指标字典

**使用示例:**
```python
from analytical_mmd_A2B_feature58.preprocessing.threshold_optimizer import optimize_threshold_youden

threshold, metrics = optimize_threshold_youden(y_true, y_proba)
print(f"最佳阈值: {threshold:.4f}")
print(f"准确率: {metrics['acc']:.4f}")
```

### apply_threshold_optimization()

```python
def apply_threshold_optimization(y_true: np.ndarray, y_pred_original: np.ndarray, 
                                y_proba: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
```

应用阈值优化并返回优化后的预测结果。

**参数:**
- `y_true`: 真实标签
- `y_pred_original`: 原始预测标签
- `y_proba`: 预测概率

**返回:**
- `y_pred_optimized`: 优化后的预测标签
- `optimization_info`: 优化信息

**使用示例:**
```python
y_pred_opt, info = apply_threshold_optimization(y_true, y_pred, y_proba)
print(f"准确率提升: {info['improvements']['acc']:+.4f}")
```

### get_roc_curve_data()

```python
def get_roc_curve_data(y_true: np.ndarray, y_proba: np.ndarray, 
                      optimal_threshold: float) -> Dict[str, Any]:
```

获取ROC曲线数据用于可视化。

**参数:**
- `y_true`: 真实标签
- `y_proba`: 预测概率
- `optimal_threshold`: 最佳阈值

**返回:**
- `roc_data`: ROC曲线相关数据

### get_threshold_optimization_suffix()

```python
def get_threshold_optimization_suffix(use_threshold_optimization: bool) -> str:
```

根据是否使用阈值优化返回路径后缀。

**使用示例:**
```python
suffix = get_threshold_optimization_suffix(True)  # "_threshold_optimized"
``` 