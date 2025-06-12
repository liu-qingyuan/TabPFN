# visualization/histogram_plots.py 文档

## 概述

特征分布直方图可视化模块，用于比较源域和目标域的特征分布差异。

## 主要函数

### visualize_feature_histograms()

```python
def visualize_feature_histograms(X_source: np.ndarray, X_target: np.ndarray, 
                                 X_target_aligned: Optional[np.ndarray] = None, 
                                 feature_names: Optional[List[str]] = None, 
                                 n_features_to_plot: Optional[int] = None, 
                                 title: str = 'Feature Distribution Comparison', 
                                save_path: Optional[str] = None, method_name: str = "MMD") -> None:
```

绘制特征分布直方图，比较域适应前后的分布变化。

**参数:**
- `X_source`: 源域数据
- `X_target`: 目标域数据
- `X_target_aligned`: 对齐后的目标域数据（可选）
- `feature_names`: 特征名称列表
- `n_features_to_plot`: 要绘制的特征数量
- `title`: 图表标题
- `save_path`: 保存路径
- `method_name`: 方法名称

**功能:**
- 计算KL散度和Wasserstein距离
- 按分布差异排序选择特征
- 生成前后对比直方图
- 显示改进百分比

**使用示例:**
```python
from analytical_mmd_A2B_feature58.visualization.histogram_plots import visualize_feature_histograms

visualize_feature_histograms(
    X_source=X_source,
    X_target=X_target,
    X_target_aligned=X_target_aligned,
    feature_names=feature_names,
    n_features_to_plot=10,
    save_path='./histograms.png',
    method_name='Linear MMD'
)
```

### histograms_stats_table()

```python
def histograms_stats_table(X_source: np.ndarray, X_target: np.ndarray, 
                          X_target_aligned: Optional[np.ndarray] = None,
                          feature_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None, method_name: str = "MMD") -> None:
```

生成特征分布统计表格。

**参数:**
- `X_source`: 源域数据
- `X_target`: 目标域数据
- `X_target_aligned`: 对齐后的目标域数据
- `feature_names`: 特征名称列表
- `save_path`: 保存路径
- `method_name`: 方法名称

**功能:**
- 计算各特征的统计指标
- 生成详细的统计表格
- 显示改进情况

### histograms_visual_stats_table()

```python
def histograms_visual_stats_table(X_source: np.ndarray, X_target: np.ndarray, 
                                 X_target_aligned: Optional[np.ndarray] = None,
                                 feature_names: Optional[List[str]] = None,
                                 save_path: Optional[str] = None, method_name: str = "MMD") -> None:
```

生成可视化统计表格。

**功能:**
- 结合图表和统计数据
- 生成综合分析报告
- 支持多种输出格式 