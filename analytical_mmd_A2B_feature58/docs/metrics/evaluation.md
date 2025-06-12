# metrics/evaluation.py 文档

## 概述

分类评估模块，包含分类任务的评估指标计算函数，包括准确率、AUC、F1分数等基础指标和阈值优化功能。

## 主要函数

### evaluate_metrics()

```python
def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
```

计算所有评估指标。

**参数:**
- `y_true`: 真实标签
- `y_pred`: 预测标签
- `y_pred_proba`: 预测概率

**返回:**
- 包含各种评估指标的字典：
  - `acc`: 总体准确率
  - `auc`: AUC值
  - `f1`: F1分数
  - `acc_0`: 类别0的准确率
  - `acc_1`: 类别1的准确率

**使用示例:**
```python
from analytical_mmd_A2B_feature58.metrics.evaluation import evaluate_metrics

metrics = evaluate_metrics(y_true, y_pred, y_pred_proba)
print(f"准确率: {metrics['acc']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")
```

### print_metrics()

```python
def print_metrics(dataset_name: str, metrics: Dict[str, float]):
```

打印评估指标。

**参数:**
- `dataset_name`: 数据集名称
- `metrics`: 评估指标字典

**使用示例:**
```python
print_metrics("测试集", metrics)
```

### optimize_threshold()

```python
def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, float]]:
```

使用Youden指数(敏感性+特异性-1)寻找最佳决策阈值。

**参数:**
- `y_true`: 真实标签
- `y_proba`: 预测为正类的概率

**返回:**
- `optimal_threshold`: 最佳决策阈值
- `optimal_metrics`: 在最佳阈值下的性能指标，包含：
  - `threshold`: 最佳阈值
  - `acc`: 准确率
  - `f1`: F1分数
  - `sensitivity`: 敏感性（TPR）
  - `specificity`: 特异性（TNR）
  - `acc_0`: 类别0准确率
  - `acc_1`: 类别1准确率

**使用示例:**
```python
from analytical_mmd_A2B_feature58.metrics.evaluation import optimize_threshold

optimal_thresh, optimal_metrics = optimize_threshold(y_true, y_proba)
print(f"最佳阈值: {optimal_thresh:.4f}")
print(f"优化后准确率: {optimal_metrics['acc']:.4f}")
```

## 兼容性导入

模块还从 `metrics.discrepancy` 导入以下函数以保持向后兼容：
- `calculate_kl_divergence`
- `calculate_wasserstein_distances`
- `compute_mmd_kernel`
- `compute_domain_discrepancy`
- `detect_outliers` 