# visualization/metrics.py 文档

## 概述

可视化度量接口模块，为可视化功能提供度量函数的接口转发。所有度量函数的实际实现已迁移到 `metrics.discrepancy` 模块。

## 导入的函数

本模块从 `metrics.discrepancy` 模块导入以下函数：

### calculate_kl_divergence()

```python
def calculate_kl_divergence(*args, **kwargs):
```

计算KL散度。

### calculate_wasserstein_distances()

```python
def calculate_wasserstein_distances(*args, **kwargs):
```

计算Wasserstein距离。

### compute_mmd_kernel()

```python
def compute_mmd_kernel(*args, **kwargs):
```

计算MMD核函数。

### compute_mmd()

```python
def compute_mmd(*args, **kwargs):
```

计算MMD距离。

### compute_domain_discrepancy()

```python
def compute_domain_discrepancy(*args, **kwargs):
```

计算域差距指标。

### detect_outliers()

```python
def detect_outliers(*args, **kwargs):
```

检测异常值。

## 使用说明

这是一个接口转发模块，建议直接使用 `metrics.discrepancy` 模块中的函数：

```python
# 推荐使用方式
from analytical_mmd_A2B_feature58.metrics.discrepancy import compute_mmd

# 或者通过此接口模块
from analytical_mmd_A2B_feature58.visualization.metrics import compute_mmd
```

## 兼容性处理

模块包含完整的导入错误处理机制，如果导入失败会提供空函数实现以避免导入错误。 