# visualization/tsne_plots.py 文档

## 概述

`visualization/tsne_plots.py` 是项目的t-SNE可视化核心模块，专门用于生成高质量的降维可视化图表，展示域适应前后的数据分布变化。本模块提供了丰富的可视化选项和自动化的图表生成功能。

## 文件位置
```
analytical_mmd_A2B_feature58/visualization/tsne_plots.py
```

## 主要功能

### 1. t-SNE降维可视化
- 高维数据的二维投影
- 多种距离度量支持
- 参数自动优化
- 批量数据处理

### 2. 域适应对比可视化
- 适应前后数据分布对比
- 源域和目标域的分布展示
- MMD距离变化可视化
- 多方法效果比较

### 3. 交互式图表
- 可缩放的散点图
- 数据点标注和悬停信息
- 图例和颜色映射
- 高分辨率输出

### 4. 批量可视化
- 多数据集批量处理
- 统一的图表风格
- 自动文件命名和保存
- 进度监控和错误处理

## 核心函数

### plot_tsne_comparison()

```python
def plot_tsne_comparison(
    X_source: np.ndarray,
    X_target_before: np.ndarray,
    X_target_after: np.ndarray,
    method_name: str = "MMD Transform",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    alpha: float = 0.7,
    s: int = 20,
    show_legend: bool = True,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    dpi: int = 300
) -> plt.Figure:
    """
    生成域适应前后的t-SNE对比可视化
    
    参数:
    - X_source (np.ndarray): 源域数据，形状为 (n_samples, n_features)
    - X_target_before (np.ndarray): 适应前的目标域数据
    - X_target_after (np.ndarray): 适应后的目标域数据
    - method_name (str): 方法名称，用于图表标题
    - save_path (str, optional): 保存路径，None表示不保存
    - figsize (Tuple[int, int]): 图表尺寸 (宽度, 高度)
    - perplexity (int): t-SNE困惑度参数，控制局部和全局结构的平衡
    - n_iter (int): t-SNE迭代次数，影响收敛质量
    - random_state (int): 随机种子，确保结果可重现
    - alpha (float): 散点透明度 (0-1)
    - s (int): 散点大小
    - show_legend (bool): 是否显示图例
    - title_fontsize (int): 标题字体大小
    - label_fontsize (int): 标签字体大小
    - dpi (int): 图像分辨率
    
    返回:
    - plt.Figure: matplotlib图形对象
    
    功能:
    - 生成三个子图：源域分布、适应前目标域、适应后目标域
    - 使用一致的颜色方案和布局
    - 自动计算和显示MMD距离
    - 支持高质量图像输出
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.visualization.tsne_plots import plot_tsne_comparison
import numpy as np

# 准备数据
X_source = np.random.randn(500, 10)
X_target_before = np.random.randn(400, 10) + 1.5  # 有分布偏移
X_target_after = np.random.randn(400, 10) + 0.3   # 适应后偏移减小

# 生成对比图
fig = plot_tsne_comparison(
    X_source=X_source,
    X_target_before=X_target_before,
    X_target_after=X_target_after,
    method_name="Linear MMD Transform",
    save_path="./results/tsne_comparison.png",
    figsize=(18, 6),
    perplexity=50,
    dpi=300
)

# 显示图表
plt.show()
```

### plot_single_tsne()

```python
def plot_single_tsne(
    X_source: np.ndarray,
    X_target: np.ndarray,
    title: str = "t-SNE Visualization",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42,
    alpha: float = 0.7,
    s: int = 30,
    colors: Tuple[str, str] = ('#1f77b4', '#ff7f0e'),
    show_mmd: bool = True,
    **kwargs
) -> plt.Figure:
    """
    生成单个t-SNE可视化图
    
    参数:
    - X_source (np.ndarray): 源域数据
    - X_target (np.ndarray): 目标域数据
    - title (str): 图表标题
    - save_path (str, optional): 保存路径
    - figsize (Tuple[int, int]): 图表尺寸
    - perplexity (int): t-SNE困惑度参数
    - n_iter (int): 迭代次数
    - random_state (int): 随机种子
    - alpha (float): 透明度
    - s (int): 散点大小
    - colors (Tuple[str, str]): 源域和目标域的颜色
    - show_mmd (bool): 是否显示MMD距离
    - **kwargs: 传递给t-SNE的额外参数
    
    返回:
    - plt.Figure: matplotlib图形对象
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.visualization.tsne_plots import plot_single_tsne

# 生成单个t-SNE图
fig = plot_single_tsne(
    X_source=X_source,
    X_target=X_target_after,
    title="Domain Adaptation Result",
    save_path="./results/single_tsne.png",
    figsize=(12, 10),
    colors=('#2E8B57', '#DC143C'),  # 自定义颜色
    show_mmd=True
)
```

### create_tsne_grid()

```python
def create_tsne_grid(
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    titles: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 15),
    ncols: int = 3,
    perplexity: int = 30,
    **kwargs
) -> plt.Figure:
    """
    创建多个t-SNE图的网格布局
    
    参数:
    - datasets (Dict[str, Tuple[np.ndarray, np.ndarray]]): 数据集字典
        键为数据集名称，值为(源域数据, 目标域数据)元组
    - titles (Dict[str, str], optional): 自定义标题字典
    - save_path (str, optional): 保存路径
    - figsize (Tuple[int, int]): 整体图表尺寸
    - ncols (int): 网格列数
    - perplexity (int): t-SNE困惑度参数
    - **kwargs: 传递给单个t-SNE图的参数
    
    返回:
    - plt.Figure: matplotlib图形对象
    
    用途:
    - 比较多种方法的效果
    - 展示不同参数设置的结果
    - 生成论文或报告用的综合图表
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.visualization.tsne_plots import create_tsne_grid

# 准备多个数据集
datasets = {
    'Original': (X_source, X_target_before),
    'Linear MMD': (X_source, X_target_linear),
    'Kernel PCA MMD': (X_source, X_target_kpca),
    'Mean-Std Align': (X_source, X_target_mean_std)
}

# 自定义标题
titles = {
    'Original': 'Before Adaptation',
    'Linear MMD': 'Linear MMD Transform',
    'Kernel PCA MMD': 'Kernel PCA MMD Transform',
    'Mean-Std Align': 'Mean-Std Alignment'
}

# 创建网格图
fig = create_tsne_grid(
    datasets=datasets,
    titles=titles,
    save_path="./results/tsne_grid.png",
    figsize=(24, 18),
    ncols=2,
    perplexity=40,
    alpha=0.6,
    s=25
)
```

### plot_tsne_with_labels()

```python
def plot_tsne_with_labels(
    X_source: np.ndarray,
    y_source: np.ndarray,
    X_target: np.ndarray,
    y_target: np.ndarray,
    title: str = "t-SNE with Class Labels",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    perplexity: int = 30,
    class_names: Optional[List[str]] = None,
    marker_styles: Optional[Dict[str, str]] = None,
    **kwargs
) -> plt.Figure:
    """
    生成带类别标签的t-SNE可视化
    
    参数:
    - X_source, y_source: 源域数据和标签
    - X_target, y_target: 目标域数据和标签
    - title (str): 图表标题
    - save_path (str, optional): 保存路径
    - figsize (Tuple[int, int]): 图表尺寸
    - perplexity (int): t-SNE困惑度参数
    - class_names (List[str], optional): 类别名称列表
    - marker_styles (Dict[str, str], optional): 标记样式字典
    - **kwargs: 额外参数
    
    返回:
    - plt.Figure: matplotlib图形对象
    
    特点:
    - 使用不同颜色表示不同类别
    - 使用不同标记区分源域和目标域
    - 自动生成详细的图例
    - 支持多类别分类问题
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.visualization.tsne_plots import plot_tsne_with_labels

# 生成带标签的t-SNE图
fig = plot_tsne_with_labels(
    X_source=X_source,
    y_source=y_source,
    X_target=X_target_after,
    y_target=y_target,
    title="Domain Adaptation with Class Labels",
    save_path="./results/tsne_with_labels.png",
    class_names=['Negative', 'Positive'],
    marker_styles={'source': 'o', 'target': '^'},
    figsize=(14, 12)
)
```

## 高级功能

### 动态t-SNE参数优化

#### optimize_tsne_params()

```python
def optimize_tsne_params(
    X: np.ndarray,
    perplexity_range: Tuple[int, int] = (5, 100),
    n_trials: int = 10,
    quality_metric: str = 'kl_divergence'
) -> Dict[str, Any]:
    """
    自动优化t-SNE参数
    
    参数:
    - X (np.ndarray): 输入数据
    - perplexity_range (Tuple[int, int]): 困惑度搜索范围
    - n_trials (int): 试验次数
    - quality_metric (str): 质量评估指标
    
    返回:
    - Dict[str, Any]: 最优参数和质量评估结果
    
    功能:
    - 自动搜索最优困惑度值
    - 评估降维质量
    - 提供参数选择建议
    """
```

### 交互式可视化

#### create_interactive_tsne()

```python
def create_interactive_tsne(
    X_source: np.ndarray,
    X_target: np.ndarray,
    labels_source: Optional[np.ndarray] = None,
    labels_target: Optional[np.ndarray] = None,
    save_html: Optional[str] = None
) -> Any:
    """
    创建交互式t-SNE可视化（使用plotly）
    
    参数:
    - X_source, X_target: 源域和目标域数据
    - labels_source, labels_target: 可选的标签信息
    - save_html (str, optional): 保存HTML文件路径
    
    返回:
    - plotly图形对象
    
    特点:
    - 支持缩放和平移
    - 悬停显示详细信息
    - 可选择性显示/隐藏数据点
    - 导出为HTML格式
    """
```

### 批量可视化处理

#### batch_tsne_visualization()

```python
def batch_tsne_visualization(
    experiments: List[Dict[str, Any]],
    output_dir: str,
    config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    批量生成t-SNE可视化
    
    参数:
    - experiments (List[Dict[str, Any]]): 实验配置列表
    - output_dir (str): 输出目录
    - config (Dict[str, Any], optional): 全局配置
    
    返回:
    - List[str]: 生成的文件路径列表
    
    功能:
    - 批量处理多个实验结果
    - 统一的命名和保存策略
    - 进度监控和错误处理
    - 生成汇总报告
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.visualization.tsne_plots import batch_tsne_visualization

# 定义实验配置
experiments = [
    {
        'name': 'linear_mmd',
        'X_source': X_source,
        'X_target_before': X_target_before,
        'X_target_after': X_target_linear,
        'method_name': 'Linear MMD'
    },
    {
        'name': 'kernel_pca_mmd',
        'X_source': X_source,
        'X_target_before': X_target_before,
        'X_target_after': X_target_kpca,
        'method_name': 'Kernel PCA MMD'
    }
]

# 批量生成可视化
generated_files = batch_tsne_visualization(
    experiments=experiments,
    output_dir="./results/visualizations/",
    config={
        'figsize': (15, 5),
        'dpi': 300,
        'perplexity': 40
    }
)

print(f"生成了 {len(generated_files)} 个可视化文件")
```

## 可视化样式和主题

### 预定义样式

```python
# 学术论文风格
ACADEMIC_STYLE = {
    'figsize': (12, 8),
    'dpi': 300,
    'alpha': 0.7,
    's': 25,
    'colors': ('#1f77b4', '#ff7f0e'),
    'title_fontsize': 16,
    'label_fontsize': 14,
    'legend_fontsize': 12,
    'grid': True,
    'grid_alpha': 0.3
}

# 演示报告风格
PRESENTATION_STYLE = {
    'figsize': (16, 10),
    'dpi': 150,
    'alpha': 0.8,
    's': 40,
    'colors': ('#2E8B57', '#DC143C'),
    'title_fontsize': 20,
    'label_fontsize': 16,
    'legend_fontsize': 14,
    'grid': False
}

# 高对比度风格
HIGH_CONTRAST_STYLE = {
    'figsize': (12, 8),
    'dpi': 300,
    'alpha': 0.9,
    's': 30,
    'colors': ('#000000', '#FF0000'),
    'title_fontsize': 16,
    'label_fontsize': 14,
    'legend_fontsize': 12,
    'grid': True,
    'grid_alpha': 0.5
}
```

### 应用样式

```python
def apply_style(style_name: str = 'academic') -> Dict[str, Any]:
    """
    应用预定义样式
    
    参数:
    - style_name (str): 样式名称 ('academic', 'presentation', 'high_contrast')
    
    返回:
    - Dict[str, Any]: 样式配置字典
    """
    
    styles = {
        'academic': ACADEMIC_STYLE,
        'presentation': PRESENTATION_STYLE,
        'high_contrast': HIGH_CONTRAST_STYLE
    }
    
    return styles.get(style_name, ACADEMIC_STYLE)

# 使用示例
style_config = apply_style('presentation')
fig = plot_tsne_comparison(
    X_source, X_target_before, X_target_after,
    **style_config
)
```

## 性能优化

### 大数据集处理

```python
def plot_tsne_large_dataset(
    X_source: np.ndarray,
    X_target: np.ndarray,
    max_samples: int = 2000,
    sampling_method: str = 'random',
    **kwargs
) -> plt.Figure:
    """
    处理大数据集的t-SNE可视化
    
    参数:
    - X_source, X_target: 源域和目标域数据
    - max_samples (int): 最大样本数量
    - sampling_method (str): 采样方法 ('random', 'stratified', 'kmeans')
    - **kwargs: 传递给plot_tsne_comparison的参数
    
    返回:
    - plt.Figure: matplotlib图形对象
    
    优化策略:
    - 智能采样减少计算量
    - 保持数据分布特征
    - 加速t-SNE计算
    """
    
    # 采样逻辑
    if len(X_source) > max_samples:
        if sampling_method == 'random':
            indices = np.random.choice(len(X_source), max_samples, replace=False)
            X_source_sampled = X_source[indices]
        elif sampling_method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=max_samples, random_state=42)
            X_source_sampled = kmeans.fit(X_source).cluster_centers_
        else:
            X_source_sampled = X_source[:max_samples]
    else:
        X_source_sampled = X_source
    
    # 类似地处理目标域数据
    # ...
    
    return plot_tsne_comparison(X_source_sampled, X_target_sampled, **kwargs)
```

### 并行处理

```python
def parallel_tsne_computation(
    datasets: List[Tuple[np.ndarray, np.ndarray]],
    n_jobs: int = -1,
    **tsne_params
) -> List[np.ndarray]:
    """
    并行计算多个数据集的t-SNE嵌入
    
    参数:
    - datasets: 数据集列表
    - n_jobs: 并行作业数量
    - **tsne_params: t-SNE参数
    
    返回:
    - List[np.ndarray]: t-SNE嵌入结果列表
    """
    
    from joblib import Parallel, delayed
    from sklearn.manifold import TSNE
    
    def compute_single_tsne(data):
        X_combined = np.vstack(data)
        tsne = TSNE(**tsne_params)
        return tsne.fit_transform(X_combined)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_tsne)(dataset) for dataset in datasets
    )
    
    return results
```

## 质量评估和验证

### t-SNE质量评估

```python
def evaluate_tsne_quality(
    X_original: np.ndarray,
    X_embedded: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    评估t-SNE降维质量
    
    参数:
    - X_original: 原始高维数据
    - X_embedded: t-SNE嵌入结果
    - k: 近邻数量
    
    返回:
    - Dict[str, float]: 质量指标字典
        - 'trustworthiness': 可信度
        - 'continuity': 连续性
        - 'neighborhood_preservation': 邻域保持度
    """
    
    from sklearn.metrics import trustworthiness
    from sklearn.neighbors import NearestNeighbors
    
    # 计算可信度
    trust = trustworthiness(X_original, X_embedded, n_neighbors=k)
    
    # 计算连续性
    nbrs_orig = NearestNeighbors(n_neighbors=k+1).fit(X_original)
    nbrs_emb = NearestNeighbors(n_neighbors=k+1).fit(X_embedded)
    
    # 详细计算逻辑...
    
    return {
        'trustworthiness': trust,
        'continuity': continuity,
        'neighborhood_preservation': neighborhood_preservation
    }
```

### 可视化效果验证

```python
def validate_visualization_effect(
    X_source: np.ndarray,
    X_target_before: np.ndarray,
    X_target_after: np.ndarray,
    method_name: str
) -> Dict[str, Any]:
    """
    验证可视化效果
    
    参数:
    - X_source: 源域数据
    - X_target_before: 适应前目标域数据
    - X_target_after: 适应后目标域数据
    - method_name: 方法名称
    
    返回:
    - Dict[str, Any]: 验证结果
    """
    
    from analytical_mmd_A2B_feature58.preprocessing.mmd import compute_mmd
    
    # 计算MMD距离
    mmd_before = compute_mmd(X_source, X_target_before)
    mmd_after = compute_mmd(X_source, X_target_after)
    improvement = (mmd_before - mmd_after) / mmd_before * 100
    
    # t-SNE质量评估
    X_combined_before = np.vstack([X_source, X_target_before])
    X_combined_after = np.vstack([X_source, X_target_after])
    
    tsne = TSNE(random_state=42)
    X_tsne_before = tsne.fit_transform(X_combined_before)
    X_tsne_after = tsne.fit_transform(X_combined_after)
    
    quality_before = evaluate_tsne_quality(X_combined_before, X_tsne_before)
    quality_after = evaluate_tsne_quality(X_combined_after, X_tsne_after)
    
    return {
        'method_name': method_name,
        'mmd_improvement': improvement,
        'mmd_before': mmd_before,
        'mmd_after': mmd_after,
        'tsne_quality_before': quality_before,
        'tsne_quality_after': quality_after,
        'visualization_improvement': quality_after['trustworthiness'] - quality_before['trustworthiness']
    }
```

## 使用示例和最佳实践

### 完整的可视化工作流程

```python
from analytical_mmd_A2B_feature58.visualization.tsne_plots import *
from analytical_mmd_A2B_feature58.preprocessing.mmd import mmd_transform

def complete_visualization_workflow(X_source, X_target, output_dir="./results/"):
    """完整的可视化工作流程"""
    
    # 1. 应用不同的域适应方法
    methods = ['linear', 'kpca', 'mean_std']
    results = {}
    
    for method in methods:
        X_adapted, info = mmd_transform(X_source, X_target, method=method)
        results[method] = {
            'X_adapted': X_adapted,
            'info': info
        }
    
    # 2. 生成单独的对比图
    for method, result in results.items():
        fig = plot_tsne_comparison(
            X_source=X_source,
            X_target_before=X_target,
            X_target_after=result['X_adapted'],
            method_name=f"{method.upper()} Transform",
            save_path=f"{output_dir}/tsne_comparison_{method}.png",
            **apply_style('academic')
        )
        plt.close(fig)
    
    # 3. 生成综合网格图
    datasets = {
        'Original': (X_source, X_target),
        **{method: (X_source, result['X_adapted']) 
           for method, result in results.items()}
    }
    
    fig = create_tsne_grid(
        datasets=datasets,
        save_path=f"{output_dir}/tsne_grid_comparison.png",
        figsize=(20, 15),
        ncols=2,
        **apply_style('presentation')
    )
    plt.close(fig)
    
    # 4. 生成质量评估报告
    quality_report = {}
    for method, result in results.items():
        quality_report[method] = validate_visualization_effect(
            X_source, X_target, result['X_adapted'], method
        )
    
    # 5. 保存质量报告
    import json
    with open(f"{output_dir}/quality_report.json", 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    print(f"可视化完成，结果保存在 {output_dir}")
    return quality_report

# 使用示例
quality_report = complete_visualization_workflow(X_source, X_target)
```

## 故障排除

### 常见问题

1. **t-SNE收敛问题**
   ```python
   # 问题: t-SNE不收敛或结果不稳定
   # 解决: 调整参数和增加迭代次数
   def robust_tsne(X, max_attempts=3):
       for attempt in range(max_attempts):
           try:
               tsne = TSNE(
                   perplexity=min(30, len(X)//4),
                   n_iter=2000,
                   random_state=42 + attempt,
                   early_exaggeration=12.0,
                   learning_rate=200.0
               )
               return tsne.fit_transform(X)
           except Exception as e:
               print(f"尝试 {attempt+1} 失败: {e}")
               if attempt == max_attempts - 1:
                   raise
   ```

2. **内存不足**
   ```python
   # 问题: 大数据集导致内存不足
   # 解决: 使用采样或分批处理
   if len(X_combined) > 5000:
       print("数据集过大，使用采样...")
       X_combined = plot_tsne_large_dataset(
           X_source, X_target, max_samples=3000
       )
   ```

3. **可视化质量差**
   ```python
   # 问题: t-SNE结果聚类效果差
   # 解决: 优化参数或使用其他降维方法
   quality = evaluate_tsne_quality(X_original, X_embedded)
   if quality['trustworthiness'] < 0.8:
       print("t-SNE质量较差，尝试调整参数...")
       # 尝试不同的困惑度值
       best_params = optimize_tsne_params(X_original)
   ```

## 总结

`visualization/tsne_plots.py` 模块提供了完整的t-SNE可视化功能，主要特点包括：

- **专业性**: 高质量的学术级可视化输出
- **灵活性**: 丰富的参数配置和样式选项
- **效率**: 针对大数据集的优化处理
- **可靠性**: 完善的质量评估和验证机制
- **易用性**: 简洁的API和批量处理功能

通过合理使用这些功能，可以生成高质量的域适应效果可视化，有效展示算法的性能和改进效果。 