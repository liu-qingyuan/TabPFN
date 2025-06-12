# metrics/cross_domain_metrics.py 文档

## 概述

跨域实验指标计算模块，提供跨域实验中的各种指标计算功能，包括外部数据集的交叉验证评估、单次评估指标计算、结果汇总和比较。

## 主要函数

### evaluate_model_on_external_cv()

```python
def evaluate_model_on_external_cv(model, X_external: np.ndarray, 
                                y_external: np.ndarray, n_folds: int = 10) -> Dict[str, Any]:
```

使用已训练的模型在外部数据集上进行K折交叉验证评估。

**参数:**
- `model`: 已训练的模型
- `X_external`: 外部数据集特征
- `y_external`: 外部数据集标签
- `n_folds`: 交叉验证折数，默认10

**返回:**
包含详细结果的字典：
- `fold_results`: 每折的详细结果
- `overall`: 整体指标
- `means`: 各指标的平均值
- `stds`: 各指标的标准差

### evaluate_single_external_dataset()

```python
def evaluate_single_external_dataset(model, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
```

在外部数据集上进行单次评估。

**参数:**
- `model`: 模型类
- `X_train`: 训练数据特征
- `y_train`: 训练数据标签
- `X_test`: 测试数据特征
- `y_test`: 测试数据标签

**返回:**
- 评估指标字典

### log_external_cv_results()

```python
def log_external_cv_results(results: Dict[str, Any], prefix: str):
```

记录外部CV结果。

**参数:**
- `results`: 评估结果字典
- `prefix`: 日志前缀

### calculate_cross_domain_improvement()

```python
def calculate_cross_domain_improvement(results_before: Dict[str, Any], 
                                     results_after: Dict[str, Any]) -> Dict[str, float]:
```

计算跨域实验中的改进幅度。

**参数:**
- `results_before`: 域适应前的结果
- `results_after`: 域适应后的结果

**返回:**
- 改进幅度字典

### summarize_cross_domain_results()

```python
def summarize_cross_domain_results(results: Dict[str, Any]) -> str:
```

汇总跨域实验结果为可读的字符串。

**参数:**
- `results`: 实验结果字典

**返回:**
- 格式化的结果摘要字符串

## 使用示例

```python
from analytical_mmd_A2B_feature58.metrics.cross_domain_metrics import (
    evaluate_model_on_external_cv, calculate_cross_domain_improvement, 
    summarize_cross_domain_results
)

# 外部数据集交叉验证
cv_results = evaluate_model_on_external_cv(trained_model, X_external, y_external)
print(f"平均AUC: {cv_results['means']['auc']:.4f} ± {cv_results['stds']['auc']:.4f}")

# 计算改进幅度
improvement = calculate_cross_domain_improvement(results_before, results_after)
print(f"AUC改进: {improvement['auc']:+.4f}")

# 生成结果摘要
summary = summarize_cross_domain_results(all_results)
print(summary)
``` 