# preprocessing/scaler.py 文档

## 概述

数据标准化器模块，提供StandardScaler的拟合和应用功能。

## 主要函数

### fit_scaler()

```python
def fit_scaler(X: np.ndarray) -> StandardScaler:
```

拟合StandardScaler。

**参数:**
- `X`: 输入数据

**返回:**
- 拟合好的StandardScaler对象

### apply_scaler()

```python
def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
```

应用已拟合的StandardScaler。

**参数:**
- `scaler`: 已拟合的StandardScaler对象
- `X`: 要标准化的数据

**返回:**
- 标准化后的数据

### fit_apply_scaler()

```python
def fit_apply_scaler(X_source: np.ndarray, X_target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray], StandardScaler]:
```

拟合源数据并应用于源数据和目标数据（如果提供）。

**参数:**
- `X_source`: 源域数据
- `X_target`: 目标域数据（可选）

**返回:**
- `X_source_scaled`: 标准化后的源域数据
- `X_target_scaled`: 标准化后的目标域数据（如果提供）
- `scaler`: 拟合的StandardScaler对象

**使用示例:**
```python
from analytical_mmd_A2B_feature58.preprocessing.scaler import fit_apply_scaler

# 同时处理源域和目标域数据
X_source_scaled, X_target_scaled, scaler = fit_apply_scaler(X_source, X_target)

# 只处理源域数据
X_source_scaled, _, scaler = fit_apply_scaler(X_source)
```

## 支持的标准化器类型

- **standard**: StandardScaler - 零均值单位方差标准化
- **minmax**: MinMaxScaler - 最小-最大归一化到[0,1]
- **robust**: RobustScaler - 基于中位数和四分位数的鲁棒标准化 