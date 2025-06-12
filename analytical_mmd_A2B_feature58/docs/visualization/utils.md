# visualization/utils.py 文档

## 概述

可视化通用工具模块，提供matplotlib图形管理和样式设置功能。

## 主要函数

### close_figures()

```python
def close_figures():
```

关闭所有打开的matplotlib图形，释放内存。

**功能:**
- 调用 `plt.close('all')` 关闭所有图形
- 记录调试日志
- 用于内存管理和防止图形累积

**使用示例:**
```python
from analytical_mmd_A2B_feature58.visualization.utils import close_figures

# 在批量生成图表后释放内存
for i in range(100):
    # 生成图表...
    pass
close_figures()  # 释放内存
```

### setup_matplotlib_style()

```python
def setup_matplotlib_style():
```

设置matplotlib的默认样式。

**配置项:**
- `figure.dpi = 100`: 显示分辨率
- `savefig.dpi = 300`: 保存分辨率
- `font.size = 10`: 默认字体大小
- `axes.titlesize = 12`: 坐标轴标题字体大小
- `axes.labelsize = 10`: 坐标轴标签字体大小
- `xtick.labelsize = 9`: X轴刻度标签字体大小
- `ytick.labelsize = 9`: Y轴刻度标签字体大小
- `legend.fontsize = 9`: 图例字体大小
- `figure.titlesize = 14`: 图形标题字体大小

**使用示例:**
```python
from analytical_mmd_A2B_feature58.visualization.utils import setup_matplotlib_style

# 在脚本开始时设置统一样式
setup_matplotlib_style()

# 后续的所有图表都会使用这些样式设置
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 2])
plt.title("示例图表")
plt.savefig("example.png")  # 将以300 DPI保存
``` 