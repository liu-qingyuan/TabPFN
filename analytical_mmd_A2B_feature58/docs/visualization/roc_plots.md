# visualization/roc_plots.py 文档

## 概述

ROC曲线绘制模块，用于分类模型的性能可视化和分析。

## 主要函数

### plot_roc_curve()

```python
def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                  save_path: Optional[str] = None, 
                  title: str = 'ROC Curve', 
                  optimal_threshold: Optional[float] = None) -> None:
```

绘制ROC曲线并可选地标记最优阈值和默认阈值。

**参数:**
- `y_true`: 真实标签
- `y_prob`: 预测概率
- `save_path`: 保存路径（可选）
- `title`: 图表标题，默认'ROC Curve'
- `optimal_threshold`: 最优阈值（可选），如果提供会在图上标记

**功能:**
- 计算并绘制ROC曲线
- 显示AUC值
- 标记最优阈值点（红色）
- 标记默认阈值0.5（蓝色）
- 添加对角线参考线
- 保存高分辨率图片

**使用示例:**
```python
from analytical_mmd_A2B_feature58.visualization.roc_plots import plot_roc_curve

# 基础ROC曲线
plot_roc_curve(y_true, y_prob, save_path='roc.png')

# 带最优阈值的ROC曲线
plot_roc_curve(y_true, y_prob, 
               save_path='roc_optimized.png',
               title='TabPFN ROC Curve',
               optimal_threshold=0.65)
``` 