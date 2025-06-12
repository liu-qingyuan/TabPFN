# metrics/classification.py 文档

## 概述

分类指标计算模块，提供各种分类任务的评估指标计算功能。

## 主要函数

### calculate_basic_metrics()

```python
def calculate_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                          y_proba: np.ndarray) -> Dict[str, float]:
```

计算基础分类指标。

**参数:**
- `y_true`: 真实标签
- `y_pred`: 预测标签
- `y_proba`: 预测概率 (N, 2)

**返回:**
- 包含accuracy、auc、f1、precision、recall、acc_0、acc_1等指标的字典

**使用示例:**
```python
from analytical_mmd_A2B_feature58.metrics.classification import calculate_basic_metrics

metrics = calculate_basic_metrics(y_true, y_pred, y_proba)
print(f"准确率: {metrics['accuracy']:.4f}")
print(f"AUC: {metrics['auc']:.4f}")
```

### calculate_cv_summary()

```python
def calculate_cv_summary(cv_scores: List[Dict[str, Any]]) -> Dict[str, str]:
```

计算交叉验证结果的汇总统计。

**参数:**
- `cv_scores`: 交叉验证各折的结果列表

**返回:**
- 包含均值±标准差格式的汇总结果

**使用示例:**
```python
cv_scores = [
    {'accuracy': 0.85, 'auc': 0.90},
    {'accuracy': 0.87, 'auc': 0.92},
    # ... 更多折的结果
]
summary = calculate_cv_summary(cv_scores)
print(summary['accuracy'])  # "0.8600 ± 0.0100"
```

### calculate_improvement()

```python
def calculate_improvement(results_before: Dict[str, str], 
                        results_after: Dict[str, str]) -> Dict[str, str]:
```

计算改进幅度。

**参数:**
- `results_before`: 改进前的结果
- `results_after`: 改进后的结果

**返回:**
- 改进幅度字典

### log_metrics()

```python
def log_metrics(metrics: Dict[str, Any], prefix: str = ""):
```

记录指标到日志。

**参数:**
- `metrics`: 指标字典
- `prefix`: 日志前缀

### compare_metrics()

```python
def compare_metrics(metrics1: Dict[str, Any], metrics2: Dict[str, Any], 
                   labels: Optional[List[str]] = None) -> str:
```

比较两组指标。

**参数:**
- `metrics1`: 第一组指标
- `metrics2`: 第二组指标
- `labels`: 标签列表

**返回:**
- 比较结果的格式化字符串

**使用示例:**
```python
comparison = compare_metrics(metrics_before, metrics_after, ["Before MMD", "After MMD"])
print(comparison)
``` 