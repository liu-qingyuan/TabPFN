# metrics/discrepancy.py 文档

## 概述

统一的分布差异度量模块，包含所有用于计算分布差异的度量函数，包括KL散度、Wasserstein距离、MMD等。

## 主要函数

### calculate_kl_divergence()

```python
def calculate_kl_divergence(X_source: np.ndarray, X_target: np.ndarray, 
                           bins: int = 20, epsilon: float = 1e-10) -> Tuple[float, Dict[str, float]]:
```

使用直方图计算每个特征的KL散度，并返回平均值。

**参数:**
- `X_source`: 源域特征 [n_samples_source, n_features]
- `X_target`: 目标域特征 [n_samples_target, n_features]
- `bins`: 直方图的箱数，默认20
- `epsilon`: 平滑因子，防止除零错误，默认1e-10

**返回:**
- `kl_div`: 平均KL散度
- `kl_per_feature`: 每个特征的KL散度字典

### calculate_wasserstein_distances()

```python
def calculate_wasserstein_distances(X_source: np.ndarray, X_target: np.ndarray) -> Tuple[float, Dict[str, float]]:
```

计算每个特征的Wasserstein距离（Earth Mover's Distance）。

**参数:**
- `X_source`: 源域特征
- `X_target`: 目标域特征

**返回:**
- `avg_wasserstein`: 平均Wasserstein距离
- `wasserstein_per_feature`: 每个特征的Wasserstein距离字典

### compute_mmd_kernel()

```python
def compute_mmd_kernel(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
```

计算基于RBF核的MMD (Maximum Mean Discrepancy)。

**参数:**
- `X`: 第一个分布的样本 [n_samples_x, n_features]
- `Y`: 第二个分布的样本 [n_samples_y, n_features]
- `gamma`: RBF核参数，默认1.0

**返回:**
- MMD值

### compute_mmd()

```python
def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
```

`compute_mmd_kernel`的别名，为了兼容性。

### compute_domain_discrepancy()

```python
def compute_domain_discrepancy(X_source: np.ndarray, X_target: np.ndarray) -> Dict[str, Any]:
```

计算源域和目标域之间的分布差异度量。

**参数:**
- `X_source`: 源域特征
- `X_target`: 目标域特征

**返回:**
包含多种差异度量的字典：
- `mean_distance`: 平均距离
- `mean_difference`: 均值差异
- `covariance_difference`: 协方差矩阵距离
- `kernel_mean_difference`: 核均值差异
- `mmd`: MMD值
- `kl_divergence`: KL散度
- `kl_per_feature`: 每个特征的KL散度
- `wasserstein_distance`: Wasserstein距离
- `wasserstein_per_feature`: 每个特征的Wasserstein距离

### detect_outliers()

```python
def detect_outliers(X_source: np.ndarray, X_target: np.ndarray, 
                   percentile: int = 95) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

检测异常点。

**参数:**
- `X_source`: 源域特征
- `X_target`: 目标域特征
- `percentile`: 异常点阈值百分位数，默认95

**返回:**
- `source_outliers`: 源域异常点索引
- `target_outliers`: 目标域异常点索引
- `min_dist_source`: 源域样本到目标域的最小距离
- `min_dist_target`: 目标域样本到源域的最小距离

## 使用示例

```python
from analytical_mmd_A2B_feature58.metrics.discrepancy import (
    compute_domain_discrepancy, compute_mmd_kernel, detect_outliers
)

# 计算域差距
discrepancy = compute_domain_discrepancy(X_source, X_target)
print(f"MMD距离: {discrepancy['mmd']:.6f}")
print(f"KL散度: {discrepancy['kl_divergence']:.6f}")

# 计算MMD
mmd_dist = compute_mmd_kernel(X_source, X_target, gamma=1.0)

# 检测异常值
outliers_s, outliers_t, _, _ = detect_outliers(X_source, X_target)
print(f"源域异常点数量: {len(outliers_s)}")
``` 