# Performance Plots 性能对比图模块

## 概述

`performance_plots.py` 模块提供了全面的性能指标可视化功能，专门用于展示和对比机器学习模型在不同条件下的性能表现。该模块支持多种图表类型，包括柱状图、热力图、雷达图和汇总表格，特别适用于域适应实验的结果分析。

## 核心功能

### 1. 多实验性能对比 (`plot_metrics_comparison`)

**功能描述**：
- 生成多个实验结果的性能指标对比图
- 支持域适应前后的对比显示
- 自动处理不同格式的实验结果数据

**主要特性**：
- 支持5个核心指标：Accuracy、AUC、F1-Score、Class 0 Accuracy、Class 1 Accuracy
- 自动识别CV结果格式和单次评估格式
- 生成2×3子图布局，每个指标独立显示
- 添加数值标签，便于精确比较

**使用示例**：
```python
from analytical_mmd_A2B_feature58.visualization.performance_plots import plot_metrics_comparison

# 实验结果字典
results_dict = {
    'Experiment_A': {
        'without_domain_adaptation': {
            'accuracy': 0.85, 'auc': 0.88, 'f1': 0.82,
            'acc_0': 0.87, 'acc_1': 0.83
        },
        'with_domain_adaptation': {
            'accuracy': 0.89, 'auc': 0.92, 'f1': 0.87,
            'acc_0': 0.91, 'acc_1': 0.87
        }
    }
}

plot_metrics_comparison(
    results_dict=results_dict,
    save_path='./performance_comparison.png',
    title='Model Performance Comparison'
)
```

### 2. 域适应改进效果图 (`plot_domain_adaptation_improvement`)

**功能描述**：
- 专门展示域适应技术带来的性能改进
- 支持热力图和柱状图两种显示方式
- 自动计算和显示改进幅度

**主要特性**：
- 自动检测改进数据格式（CV格式或单次评估格式）
- CV格式：生成改进热力图，显示所有指标的改进情况
- 单次评估格式：生成AUC和Accuracy改进的双柱状图
- 支持正负改进值的可视化

**使用示例**：
```python
from analytical_mmd_A2B_feature58.visualization.performance_plots import plot_domain_adaptation_improvement

# 包含改进数据的结果字典
results_with_improvement = {
    'MMD_Linear': {
        'improvement': {
            'auc_improvement': 0.04,
            'accuracy_improvement': 0.03
        }
    }
}

plot_domain_adaptation_improvement(
    results_dict=results_with_improvement,
    save_path='./domain_adaptation_improvement.png',
    title='Domain Adaptation Improvement Analysis'
)
```

### 3. 跨数据集性能对比 (`plot_cross_dataset_performance`)

**功能描述**：
- 展示模型在不同数据集上的性能表现
- 对比域适应前后在各数据集上的效果
- 适用于A→B、A→C等跨域实验分析

**主要特性**：
- 并排柱状图显示域适应前后的性能
- 支持多个数据集的同时比较
- 每个指标独立子图，便于详细分析
- 自动添加图例和数值标签

**使用示例**：
```python
from analytical_mmd_A2B_feature58.visualization.performance_plots import plot_cross_dataset_performance

# 跨数据集结果
cross_dataset_results = {
    'Dataset_B': {
        'without_domain_adaptation': {'accuracy': 0.75, 'auc': 0.78, ...},
        'with_domain_adaptation': {'accuracy': 0.82, 'auc': 0.85, ...}
    },
    'Dataset_C': {
        'without_domain_adaptation': {'accuracy': 0.70, 'auc': 0.73, ...},
        'with_domain_adaptation': {'accuracy': 0.78, 'auc': 0.81, ...}
    }
}

plot_cross_dataset_performance(
    results_dict=cross_dataset_results,
    save_path='./cross_dataset_performance.png',
    title='Cross-Dataset Performance Analysis'
)
```

### 4. 模型性能对比热力图 (`plot_model_comparison`)

**功能描述**：
- 生成模型性能的热力图矩阵
- 直观显示不同模型在各指标上的表现
- 适用于多模型对比分析

**主要特性**：
- 使用颜色深浅表示性能高低
- 自动添加数值标签
- 支持模型名称和指标名称的自定义显示
- 包含颜色条图例

**使用示例**：
```python
from analytical_mmd_A2B_feature58.visualization.performance_plots import plot_model_comparison

# 多模型结果
model_results = {
    'AutoTabPFN': {'accuracy': 0.89, 'auc': 0.92, ...},
    'TunedTabPFN': {'accuracy': 0.87, 'auc': 0.90, ...},
    'BaseTabPFN': {'accuracy': 0.85, 'auc': 0.88, ...}
}

plot_model_comparison(
    results_dict=model_results,
    save_path='./model_comparison_heatmap.png',
    title='Model Performance Heatmap'
)
```

### 5. 性能指标雷达图 (`plot_metrics_radar_chart`)

**功能描述**：
- 生成多维性能指标的雷达图
- 直观展示模型在各个维度的综合表现
- 支持多个实验的叠加比较

**主要特性**：
- 5个维度的雷达图：Accuracy、AUC、F1-Score、Class 0 Acc、Class 1 Acc
- 支持多个实验结果的叠加显示
- 自动颜色分配和图例生成
- 填充区域显示整体性能轮廓

**使用示例**：
```python
from analytical_mmd_A2B_feature58.visualization.performance_plots import plot_metrics_radar_chart

plot_metrics_radar_chart(
    results_dict=results_dict,
    save_path='./performance_radar_chart.png',
    title='Performance Radar Chart Analysis'
)
```

### 6. 性能汇总表格 (`create_performance_summary_table`)

**功能描述**：
- 创建详细的性能汇总表格
- 同时生成CSV文件和可视化表格图像
- 支持标准差显示（CV结果）

**主要特性**：
- 自动格式化数值显示
- 支持CV结果的均值±标准差格式
- 生成美观的表格图像
- 同时输出CSV文件便于进一步分析

**使用示例**：
```python
from analytical_mmd_A2B_feature58.visualization.performance_plots import create_performance_summary_table

summary_df = create_performance_summary_table(
    results_dict=results_dict,
    save_path='./performance_summary_table.png',
    title='Comprehensive Performance Summary'
)

# 返回的DataFrame可用于进一步分析
print(summary_df.head())
```

## 集成使用

### 完整性能分析套件

模块还提供了一个集成函数，可以一次性生成所有类型的性能对比图：

```python
from analytical_mmd_A2B_feature58.visualization.comparison_plots import generate_performance_comparison_plots

# 一次性生成所有性能对比图
generate_performance_comparison_plots(
    results_dict=comprehensive_results,
    save_dir='./performance_analysis',
    experiment_name='MMD_Domain_Adaptation'
)
```

这将生成：
- 基础性能指标对比图
- 域适应改进效果图
- 跨数据集性能对比图
- 模型性能对比热力图
- 性能指标雷达图
- 性能汇总表格

## 数据格式要求

### 支持的结果格式

1. **跨域实验格式**：
```python
{
    'experiment_name': {
        'without_domain_adaptation': {
            'accuracy': 0.85, 'auc': 0.88, 'f1': 0.82,
            'acc_0': 0.87, 'acc_1': 0.83
        },
        'with_domain_adaptation': {
            'accuracy': 0.89, 'auc': 0.92, 'f1': 0.87,
            'acc_0': 0.91, 'acc_1': 0.87
        },
        'improvement': {
            'auc_improvement': 0.04,
            'accuracy_improvement': 0.03
        }
    }
}
```

2. **CV结果格式**：
```python
{
    'experiment_name': {
        'means': {
            'accuracy': 0.85, 'auc': 0.88, 'f1': 0.82,
            'acc_0': 0.87, 'acc_1': 0.83
        },
        'stds': {
            'accuracy': 0.03, 'auc': 0.02, 'f1': 0.04,
            'acc_0': 0.02, 'acc_1': 0.03
        }
    }
}
```

3. **字符串格式（CV）**：
```python
{
    'experiment_name': {
        'accuracy': '0.85 ± 0.03',
        'auc': '0.88 ± 0.02',
        'f1': '0.82 ± 0.04',
        'acc_0': '0.87 ± 0.02',
        'acc_1': '0.83 ± 0.03'
    }
}
```

## 配置选项

### 图表样式配置

所有函数都支持以下通用参数：
- `figsize`: 图表大小，默认根据图表类型优化
- `title`: 图表标题
- `save_path`: 保存路径，支持PNG格式
- `dpi`: 图像分辨率，默认300

### 颜色和样式

- 使用matplotlib的tab10颜色映射
- 支持透明度设置
- 自动网格线和标签旋转
- 统一的字体大小和样式

## 性能优化

### 内存管理
- 每个函数结束时自动关闭图形对象
- 支持大量数据的批量处理
- 优化的颜色映射计算

### 错误处理
- 自动检测和处理缺失数据
- 支持不完整的结果字典
- 提供详细的日志信息

## 扩展功能

### 自定义指标
模块设计支持扩展到其他性能指标，只需修改指标列表：

```python
# 在函数中修改这些列表来支持新指标
metrics = ['accuracy', 'auc', 'f1', 'acc_0', 'acc_1', 'precision', 'recall']
metric_names = ['Accuracy', 'AUC', 'F1-Score', 'Class 0 Acc', 'Class 1 Acc', 'Precision', 'Recall']
```

### 输出格式
- 支持PNG、PDF、SVG等多种输出格式
- 可配置的图像质量和大小
- 支持批量导出

## 最佳实践

1. **数据准备**：确保所有实验结果使用一致的指标名称
2. **命名规范**：使用描述性的实验名称和保存路径
3. **批量处理**：对于多个实验，使用集成函数一次性生成所有图表
4. **结果验证**：检查生成的图表和数值是否符合预期
5. **文档记录**：保存实验配置和结果解释

## 故障排除

### 常见问题

1. **导入错误**：确保所有依赖包已正确安装
2. **数据格式错误**：检查结果字典的键名和数据类型
3. **保存路径问题**：确保目标目录存在且有写入权限
4. **内存不足**：对于大量数据，考虑分批处理

### 调试技巧

- 使用logging模块查看详细的执行信息
- 检查中间结果的数据格式
- 验证输入数据的完整性和一致性

## 版本历史

- v1.0: 初始版本，包含基础性能对比功能
- v1.1: 添加域适应改进效果图
- v1.2: 增加跨数据集性能对比
- v1.3: 完善雷达图和汇总表格功能 