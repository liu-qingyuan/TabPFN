# visualization/comparison_plots.py 文档

## 概述

方法效果比较图表模块，用于可视化不同MMD方法的性能对比。

## 主要函数

### plot_method_comparison()

```python
def plot_method_comparison(results_dict: Dict[str, Dict[str, Any]], 
                          metrics: List[str] = ['accuracy', 'auc', 'f1'],
                          title: str = 'Method Comparison',
                          save_path: Optional[str] = None) -> None:
```

绘制多种方法的性能对比图。

**参数:**
- `results_dict`: 方法结果字典
- `metrics`: 要比较的指标列表
- `title`: 图表标题
- `save_path`: 保存路径

**使用示例:**
```python
from analytical_mmd_A2B_feature58.visualization.comparison_plots import plot_method_comparison

results = {
    'Linear MMD': {'accuracy': 0.85, 'auc': 0.90},
    'KPCA MMD': {'accuracy': 0.87, 'auc': 0.92},
    'Mean-Std': {'accuracy': 0.83, 'auc': 0.88}
}

plot_method_comparison(results, save_path='comparison.png')
```

### plot_before_after_comparison()

```python
def plot_before_after_comparison(metrics_before: Dict[str, float],
                                metrics_after: Dict[str, float],
                                method_name: str = 'MMD',
                                save_path: Optional[str] = None) -> None:
```

绘制域适应前后的性能对比。

### plot_cross_domain_heatmap()

```python
def plot_cross_domain_heatmap(results_matrix: np.ndarray,
                             dataset_names: List[str],
                             metric_name: str = 'AUC',
                             save_path: Optional[str] = None) -> None:
```

绘制跨域性能热力图。

### plot_feature_importance_comparison()

```python
def plot_feature_importance_comparison(importance_dict: Dict[str, np.ndarray],
                                     feature_names: List[str],
                                     save_path: Optional[str] = None) -> None:
```

比较不同方法的特征重要性。 

# Comparison Plots 对比图模块

## 概述

`comparison_plots.py` 模块提供了域适应前后的综合对比可视化功能，以及完整的性能对比图表套件。该模块整合了多种可视化技术，用于全面分析域适应效果和模型性能。

## 核心功能

### 1. 域适应前后综合对比 (`compare_before_after_adaptation`)

**功能描述**：
- 生成域适应前后的全面对比可视化
- 整合t-SNE、特征分布和统计分析
- 计算域差异指标和改进百分比

**主要特性**：
- t-SNE可视化：展示数据分布的变化
- 特征分布直方图：对比各特征的分布差异
- 统计表格：量化分析域适应效果
- 域差异指标：MMD、KL散度、Wasserstein距离等

### 2. MMD适应结果可视化 (`visualize_mmd_adaptation_results`)

**功能描述**：
- 生成完整的MMD域适应结果报告
- 包含可视化图表和文本总结
- 适用于单个MMD方法的详细分析

### 3. MMD方法对比 (`plot_mmd_methods_comparison`)

**功能描述**：
- 对比不同MMD方法的效果
- 生成多指标的柱状图对比
- 支持适应前后的效果对比

## 新增功能：完整性能对比图表套件

### 4. 性能对比图表套件 (`generate_performance_comparison_plots`)

**功能描述**：
这是一个集成函数，可以一次性生成所有类型的性能对比图表，包括：

1. **基础性能指标对比图**：多个实验的核心指标对比
2. **域适应改进效果图**：展示域适应带来的性能提升
3. **跨数据集性能对比图**：不同数据集上的性能表现
4. **模型性能对比热力图**：多模型性能的矩阵对比
5. **性能指标雷达图**：多维性能的综合展示
6. **性能汇总表格**：详细的数值汇总和CSV导出

**使用示例**：
```python
from analytical_mmd_A2B_feature58.visualization.comparison_plots import generate_performance_comparison_plots

# 完整的实验结果字典
comprehensive_results = {
    'Dataset_B': {
        'without_domain_adaptation': {
            'accuracy': 0.75, 'auc': 0.78, 'f1': 0.72,
            'acc_0': 0.77, 'acc_1': 0.73
        },
        'with_domain_adaptation': {
            'accuracy': 0.82, 'auc': 0.85, 'f1': 0.79,
            'acc_0': 0.84, 'acc_1': 0.80
        },
        'improvement': {
            'auc_improvement': 0.07,
            'accuracy_improvement': 0.07
        }
    },
    'Dataset_C': {
        'without_domain_adaptation': {
            'accuracy': 0.70, 'auc': 0.73, 'f1': 0.68,
            'acc_0': 0.72, 'acc_1': 0.68
        },
        'with_domain_adaptation': {
            'accuracy': 0.78, 'auc': 0.81, 'f1': 0.75,
            'acc_0': 0.80, 'acc_1': 0.76
        },
        'improvement': {
            'auc_improvement': 0.08,
            'accuracy_improvement': 0.08
        }
    }
}

# 一次性生成所有性能对比图
summary_df = generate_performance_comparison_plots(
    results_dict=comprehensive_results,
    save_dir='./performance_analysis',
    experiment_name='MMD_Domain_Adaptation'
)
```

**生成的文件**：
- `MMD_Domain_Adaptation_metrics_comparison.png`: 基础指标对比图
- `MMD_Domain_Adaptation_domain_adaptation_improvement.png`: 改进效果图
- `MMD_Domain_Adaptation_cross_dataset_performance.png`: 跨数据集对比图
- `MMD_Domain_Adaptation_model_comparison.png`: 模型对比热力图
- `MMD_Domain_Adaptation_performance_radar.png`: 雷达图
- `MMD_Domain_Adaptation_performance_summary.png`: 汇总表格图像
- `MMD_Domain_Adaptation_performance_summary.csv`: 汇总表格CSV文件

## 支持的图表类型

### 1. 基础性能指标对比图
- **格式**：2×3子图布局
- **指标**：Accuracy、AUC、F1-Score、Class 0 Accuracy、Class 1 Accuracy
- **特点**：柱状图显示，自动添加数值标签

### 2. 域适应改进效果图
- **CV格式**：热力图显示所有指标的改进情况
- **单次评估格式**：AUC和Accuracy改进的双柱状图
- **特点**：支持正负改进值，零线参考

### 3. 跨数据集性能对比图
- **格式**：并排柱状图
- **对比**：域适应前后在各数据集上的表现
- **特点**：每个指标独立子图，图例清晰

### 4. 模型性能对比热力图
- **格式**：矩阵热力图
- **用途**：多模型在各指标上的表现对比
- **特点**：颜色深浅表示性能高低，包含数值标签

### 5. 性能指标雷达图
- **格式**：5维雷达图
- **维度**：Accuracy、AUC、F1-Score、Class 0 Acc、Class 1 Acc
- **特点**：支持多实验叠加，填充区域显示

### 6. 性能汇总表格
- **格式**：结构化表格
- **输出**：PNG图像 + CSV文件
- **特点**：支持CV结果的均值±标准差格式

## 数据格式兼容性

模块支持多种实验结果格式：

1. **跨域实验格式**：包含域适应前后和改进数据
2. **CV结果格式**：包含均值和标准差
3. **字符串格式**：`"0.85 ± 0.03"` 形式的CV结果
4. **直接数值格式**：简单的数值字典

## 集成优势

### 一站式解决方案
- 单个函数调用生成所有图表
- 统一的命名规范和保存路径
- 自动创建目录结构

### 灵活性
- 支持部分数据的可视化
- 自动处理缺失数据
- 可配置的图表标题和样式

### 可扩展性
- 易于添加新的图表类型
- 支持自定义指标
- 模块化设计便于维护

## 使用建议

1. **完整分析**：使用 `generate_performance_comparison_plots` 进行全面分析
2. **特定需求**：直接调用 `performance_plots` 模块中的特定函数
3. **批量处理**：对多个实验使用循环调用
4. **结果验证**：检查生成的图表和CSV文件

## 性能优化

- 自动内存管理，避免内存泄漏
- 批量处理优化
- 错误处理和日志记录
- 支持大规模数据集

## 版本更新

- v1.0: 基础域适应对比功能
- v1.1: 添加MMD方法对比
- v1.2: 集成性能对比图表套件
- v1.3: 完善数据格式兼容性和错误处理 