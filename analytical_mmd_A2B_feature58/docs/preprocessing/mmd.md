# preprocessing/mmd.py 文档

## 概述

`preprocessing/mmd.py` 是项目的MMD算法核心实现文件，包含Maximum Mean Discrepancy的各种计算方法和域适应变换算法。本模块提供了完整的MMD理论实现和实用的域适应工具。

## 文件位置
```
analytical_mmd_A2B_feature58/preprocessing/mmd.py
```

## 主要功能

### 1. MMD距离计算
- 无偏MMD估计
- 多种核函数支持
- 高效的矩阵运算实现
- 数值稳定性保证

### 2. 线性MMD变换
- 可学习的线性变换矩阵
- 分阶段训练策略
- 动态gamma搜索
- 梯度裁剪和监控

### 3. 核PCA MMD变换
- 非线性域适应
- 核主成分分析
- 降维和去噪
- 多种核函数支持

### 4. 统一变换接口
- 方法无关的调用接口
- 自动参数优化
- 结果验证和报告
- 性能监控

## 核心函数

### compute_mmd()

```python
def compute_mmd(
    X: np.ndarray, 
    Y: np.ndarray, 
    kernel: str = 'rbf',
    gamma: float = 1.0, 
    degree: int = 3, 
    coef0: float = 1.0,
    unbiased: bool = True
) -> float:
    """
    计算两个数据集之间的MMD距离
    
    参数:
    - X (np.ndarray): 源域数据，形状为 (n_samples_1, n_features)
    - Y (np.ndarray): 目标域数据，形状为 (n_samples_2, n_features)
    - kernel (str): 核函数类型
        - 'rbf': 径向基函数核 (默认)
        - 'linear': 线性核
        - 'poly': 多项式核
        - 'sigmoid': Sigmoid核
    - gamma (float): RBF/poly/sigmoid核参数，默认1.0
    - degree (int): 多项式核的度数，默认3
    - coef0 (float): 多项式/sigmoid核的独立项，默认1.0
    - unbiased (bool): 是否使用无偏估计，默认True
    
    返回:
    - float: MMD距离值，非负数
    
    数学原理:
    MMD²(P, Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    其中 x,x' ~ P, y,y' ~ Q
    
    无偏估计:
    MMD²(X, Y) = 1/(n(n-1)) ∑∑k(xᵢ,xⱼ) - 2/(nm) ∑∑k(xᵢ,yⱼ) + 1/(m(m-1)) ∑∑k(yᵢ,yⱼ)
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.preprocessing.mmd import compute_mmd
import numpy as np

# 生成示例数据
X_source = np.random.randn(100, 10)
X_target = np.random.randn(80, 10) + 0.5  # 轻微分布偏移

# 计算MMD距离
mmd_distance = compute_mmd(X_source, X_target, kernel='rbf', gamma=0.1)
print(f"MMD距离: {mmd_distance:.6f}")

# 比较不同核函数
kernels = ['rbf', 'linear', 'poly']
for kernel in kernels:
    mmd = compute_mmd(X_source, X_target, kernel=kernel)
    print(f"{kernel}核MMD: {mmd:.6f}")
```

### median_heuristic_gamma()

```python
def median_heuristic_gamma(
    X: np.ndarray, 
    Y: np.ndarray = None,
    subsample: int = 1000
) -> float:
    """
    使用中值启发式计算RBF核的gamma参数
    
    参数:
    - X (np.ndarray): 第一个数据集
    - Y (np.ndarray, optional): 第二个数据集，如果为None则仅使用X
    - subsample (int): 子采样大小，用于大数据集加速计算
    
    返回:
    - float: 推荐的gamma值
    
    原理:
    1. 计算所有样本对之间的欧氏距离
    2. 取距离的中值作为带宽参数
    3. gamma = 1 / (2 * median_distance²)
    
    这种启发式方法通常能为RBF核提供合理的gamma值
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.preprocessing.mmd import median_heuristic_gamma

# 自动计算gamma值
gamma = median_heuristic_gamma(X_source, X_target)
print(f"推荐gamma值: {gamma:.6f}")

# 使用推荐gamma计算MMD
mmd_optimized = compute_mmd(X_source, X_target, kernel='rbf', gamma=gamma)
print(f"优化后MMD: {mmd_optimized:.6f}")
```

### MMDLinearTransform类

```python
class MMDLinearTransform:
    """
    线性MMD变换器
    
    学习一个线性变换矩阵W，使得变换后的目标域数据与源域数据的MMD距离最小：
    min_W MMD²(X_source, X_target @ W) + λ||W||²
    """
    
    def __init__(
        self,
        input_dim: int,
        n_epochs: int = 200,
        lr: float = 3e-4,
        batch_size: int = 64,
        lambda_reg: float = 1e-3,
        gamma: float = 1.0,
        device: str = 'cuda',
        staged_training: bool = True,
        dynamic_gamma: bool = True,
        gamma_search_values: List[float] = [0.01, 0.05, 0.1],
        standardize_features: bool = True,
        use_gradient_clipping: bool = True,
        max_grad_norm: float = 1.0,
        monitor_gradients: bool = True,
        early_stopping: bool = True,
        patience: int = 20,
        min_delta: float = 1e-6
    ):
        """
        初始化线性MMD变换器
        
        参数:
        - input_dim (int): 输入特征维度
        - n_epochs (int): 训练轮数，默认200
        - lr (float): 学习率，默认3e-4
        - batch_size (int): 批大小，默认64
        - lambda_reg (float): 正则化系数，默认1e-3
        - gamma (float): RBF核参数，默认1.0
        - device (str): 计算设备 ('cuda' 或 'cpu')
        - staged_training (bool): 是否使用分阶段训练
        - dynamic_gamma (bool): 是否动态搜索gamma
        - gamma_search_values (List[float]): gamma候选值
        - standardize_features (bool): 是否标准化特征
        - use_gradient_clipping (bool): 是否使用梯度裁剪
        - max_grad_norm (float): 最大梯度范数
        - monitor_gradients (bool): 是否监控梯度
        - early_stopping (bool): 是否使用早停
        - patience (int): 早停耐心值
        - min_delta (float): 早停最小改进量
        """
```

#### 核心方法

##### fit()
```python
def fit(
    self, 
    X_source: np.ndarray, 
    X_target: np.ndarray
) -> 'MMDLinearTransform':
    """
    训练变换矩阵
    
    参数:
    - X_source (np.ndarray): 源域数据
    - X_target (np.ndarray): 目标域数据
    
    返回:
    - MMDLinearTransform: 训练后的变换器
    
    训练流程:
    1. 数据预处理和标准化
    2. 初始化变换矩阵W
    3. 如果启用分阶段训练：
       - 第一阶段：使用小gamma进行粗对齐
       - 第二阶段：使用大gamma进行精细对齐
    4. 如果启用动态gamma：
       - 搜索最优gamma值
    5. 优化MMD损失函数
    6. 记录训练历史和统计信息
    """
```

##### transform()
```python
def transform(self, X: np.ndarray) -> np.ndarray:
    """
    应用学习到的变换
    
    参数:
    - X (np.ndarray): 待变换的数据
    
    返回:
    - np.ndarray: 变换后的数据
    
    注意:
    - 变换器必须先调用fit()方法训练
    - 输入数据会自动应用训练时的标准化
    """
```

##### fit_transform()
```python
def fit_transform(
    self, 
    X_source: np.ndarray, 
    X_target: np.ndarray
) -> np.ndarray:
    """
    训练并应用变换（便捷方法）
    
    参数:
    - X_source (np.ndarray): 源域数据
    - X_target (np.ndarray): 目标域数据
    
    返回:
    - np.ndarray: 变换后的目标域数据
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.preprocessing.mmd import MMDLinearTransform

# 创建变换器
transformer = MMDLinearTransform(
    input_dim=X_target.shape[1],
    n_epochs=200,
    lr=3e-4,
    staged_training=True,
    dynamic_gamma=True
)

# 训练变换器
transformer.fit(X_source, X_target)

# 应用变换
X_target_aligned = transformer.transform(X_target)

# 验证效果
mmd_before = compute_mmd(X_source, X_target)
mmd_after = compute_mmd(X_source, X_target_aligned)
improvement = (mmd_before - mmd_after) / mmd_before * 100

print(f"MMD变换前: {mmd_before:.6f}")
print(f"MMD变换后: {mmd_after:.6f}")
print(f"改进百分比: {improvement:.2f}%")

# 查看训练历史
if hasattr(transformer, 'training_history_'):
    import matplotlib.pyplot as plt
    plt.plot(transformer.training_history_['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MMD Loss')
    plt.show()
```

### 核PCA MMD变换

#### mmd_kernel_pca_transform()

```python
def mmd_kernel_pca_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    kernel: str = 'rbf',
    gamma: float = 1.0,
    n_components: int = 10,
    use_inverse_transform: bool = False,
    standardize: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    使用核PCA进行MMD域适应
    
    参数:
    - X_source (np.ndarray): 源域数据
    - X_target (np.ndarray): 目标域数据
    - kernel (str): 核函数类型 ('rbf', 'linear', 'poly')
    - gamma (float): 核函数参数
    - n_components (int): 保留的主成分数量
    - use_inverse_transform (bool): 是否使用逆变换回原空间
    - standardize (bool): 是否标准化数据
    
    返回:
    - Tuple[np.ndarray, Dict[str, Any]]: (变换后的目标域数据, 变换信息)
    
    原理:
    1. 合并源域和目标域数据
    2. 计算核矩阵K
    3. 对核矩阵进行中心化
    4. 执行特征值分解
    5. 选择前n_components个主成分
    6. 将目标域数据投影到主成分空间
    7. 可选：逆变换回原始特征空间
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.preprocessing.mmd import mmd_kernel_pca_transform

# 核PCA变换
X_target_kpca, kpca_info = mmd_kernel_pca_transform(
    X_source=X_source,
    X_target=X_target,
    kernel='rbf',
    gamma=0.1,
    n_components=15,
    standardize=True
)

print(f"核PCA变换信息:")
print(f"  解释方差比: {kpca_info['explained_variance_ratio'][:5]}")
print(f"  累积解释方差: {kpca_info['cumulative_variance']:.3f}")
print(f"  变换前MMD: {kpca_info['mmd_before']:.6f}")
print(f"  变换后MMD: {kpca_info['mmd_after']:.6f}")
```

### 均值标准差对齐

#### mmd_mean_std_transform()

```python
def mmd_mean_std_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    feature_wise: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    简单的均值标准差对齐
    
    参数:
    - X_source (np.ndarray): 源域数据
    - X_target (np.ndarray): 目标域数据
    - feature_wise (bool): 是否按特征进行对齐
    
    返回:
    - Tuple[np.ndarray, Dict[str, Any]]: (对齐后的目标域数据, 对齐信息)
    
    原理:
    对每个特征（或整体）：
    1. 计算源域的均值μ_s和标准差σ_s
    2. 计算目标域的均值μ_t和标准差σ_t
    3. 变换：X_target_new = (X_target - μ_t) / σ_t * σ_s + μ_s
    
    这种方法简单快速，适合作为基线方法
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.preprocessing.mmd import mmd_mean_std_transform

# 均值标准差对齐
X_target_aligned, align_info = mmd_mean_std_transform(
    X_source=X_source,
    X_target=X_target,
    feature_wise=True
)

print(f"对齐信息:")
print(f"  变换前MMD: {align_info['mmd_before']:.6f}")
print(f"  变换后MMD: {align_info['mmd_after']:.6f}")
print(f"  改进百分比: {align_info['improvement_percent']:.2f}%")
```

### 统一变换接口

#### mmd_transform()

```python
def mmd_transform(
    X_source: np.ndarray,
    X_target: np.ndarray,
    method: str = 'linear',
    cat_idx: Optional[List[int]] = None,
    **kwargs: Any
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    统一的MMD域适应变换接口
    
    参数:
    - X_source (np.ndarray): 源域数据
    - X_target (np.ndarray): 目标域数据
    - method (str): MMD方法
        - 'linear': 线性MMD变换
        - 'kpca': 核PCA MMD变换
        - 'mean_std': 均值标准差对齐
    - cat_idx (List[int], optional): 类别特征索引
    - **kwargs: 方法特定参数
    
    返回:
    - Tuple[np.ndarray, Dict[str, Any]]: (对齐后的目标域数据, 适应信息)
    
    适应信息字典包含:
    - 'method': 使用的方法
    - 'mmd_before': 适应前的MMD距离
    - 'mmd_after': 适应后的MMD距离
    - 'reduction': MMD减少百分比
    - 'parameters': 使用的参数
    - 'training_time': 训练时间（如适用）
    - 'convergence_info': 收敛信息（如适用）
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.preprocessing.mmd import mmd_transform

# 线性MMD变换
X_aligned_linear, info_linear = mmd_transform(
    X_source, X_target, 
    method='linear',
    n_epochs=200,
    lr=3e-4,
    staged_training=True
)

# 核PCA MMD变换
X_aligned_kpca, info_kpca = mmd_transform(
    X_source, X_target,
    method='kpca',
    kernel='rbf',
    gamma=0.1,
    n_components=15
)

# 均值标准差对齐
X_aligned_simple, info_simple = mmd_transform(
    X_source, X_target,
    method='mean_std'
)

# 比较不同方法
methods = ['linear', 'kpca', 'mean_std']
results = [info_linear, info_kpca, info_simple]

for method, result in zip(methods, results):
    print(f"{method}方法:")
    print(f"  MMD减少: {result['reduction']:.2f}%")
    print(f"  训练时间: {result.get('training_time', 'N/A')}")
```

## 高级功能

### 批处理MMD计算

#### compute_mmd_batch()

```python
def compute_mmd_batch(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 1000,
    kernel: str = 'rbf',
    gamma: float = 1.0
) -> float:
    """
    批处理计算MMD，适用于大数据集
    
    参数:
    - X, Y: 数据集
    - batch_size: 批大小
    - kernel, gamma: 核函数参数
    
    返回:
    - float: MMD距离
    
    优势:
    - 内存友好
    - 支持大数据集
    - 保持计算精度
    """
```

### MMD梯度计算

#### compute_mmd_gradient()

```python
def compute_mmd_gradient(
    X_source: np.ndarray,
    X_target: np.ndarray,
    W: np.ndarray,
    gamma: float = 1.0
) -> np.ndarray:
    """
    计算MMD损失相对于变换矩阵W的梯度
    
    参数:
    - X_source: 源域数据
    - X_target: 目标域数据
    - W: 当前变换矩阵
    - gamma: RBF核参数
    
    返回:
    - np.ndarray: 梯度矩阵
    
    用途:
    - 自定义优化算法
    - 梯度分析
    - 调试训练过程
    """
```

### 多核MMD

#### compute_multi_kernel_mmd()

```python
def compute_multi_kernel_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernels: List[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    使用多个核函数计算MMD
    
    参数:
    - X, Y: 数据集
    - kernels: 核函数配置列表
    
    返回:
    - Dict[str, float]: 各核函数的MMD值
    
    默认核函数:
    - RBF核（多个gamma值）
    - 线性核
    - 多项式核
    """
```

## 性能优化

### GPU加速

```python
# 使用PyTorch进行GPU加速
import torch

def compute_mmd_gpu(X, Y, gamma=1.0):
    """GPU加速的MMD计算"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)
    
    # 计算核矩阵
    XX = torch.cdist(X_tensor, X_tensor) ** 2
    YY = torch.cdist(Y_tensor, Y_tensor) ** 2
    XY = torch.cdist(X_tensor, Y_tensor) ** 2
    
    # RBF核
    K_XX = torch.exp(-gamma * XX)
    K_YY = torch.exp(-gamma * YY)
    K_XY = torch.exp(-gamma * XY)
    
    # MMD计算
    mmd = K_XX.mean() - 2 * K_XY.mean() + K_YY.mean()
    
    return mmd.item()
```

### 内存优化

```python
def compute_mmd_memory_efficient(X, Y, chunk_size=1000):
    """内存高效的MMD计算"""
    n, m = len(X), len(Y)
    
    # 分块计算避免大矩阵
    mmd_xx = 0
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            X_chunk_i = X[i:i+chunk_size]
            X_chunk_j = X[j:j+chunk_size]
            mmd_xx += compute_kernel_chunk(X_chunk_i, X_chunk_j).sum()
    
    # 类似地计算其他项
    # ...
    
    return mmd_value
```

## 验证和测试

### MMD计算验证

```python
def validate_mmd_computation():
    """验证MMD计算的正确性"""
    
    # 测试1: 相同分布的MMD应该接近0
    X = np.random.randn(100, 5)
    Y = np.random.randn(100, 5)
    mmd_same = compute_mmd(X, X)
    assert mmd_same < 1e-10, f"相同数据的MMD应该为0，实际值: {mmd_same}"
    
    # 测试2: 不同分布的MMD应该大于0
    Y_shifted = Y + 2.0
    mmd_diff = compute_mmd(X, Y_shifted)
    assert mmd_diff > 0, f"不同分布的MMD应该大于0，实际值: {mmd_diff}"
    
    # 测试3: MMD的对称性
    mmd_xy = compute_mmd(X, Y)
    mmd_yx = compute_mmd(Y, X)
    assert abs(mmd_xy - mmd_yx) < 1e-10, "MMD应该满足对称性"
    
    print("✅ MMD计算验证通过")
```

### 变换效果验证

```python
def validate_transformation_effect(X_source, X_target, X_target_transformed):
    """验证变换效果"""
    
    mmd_before = compute_mmd(X_source, X_target)
    mmd_after = compute_mmd(X_source, X_target_transformed)
    
    improvement = (mmd_before - mmd_after) / mmd_before * 100
    
    validation_result = {
        'mmd_before': mmd_before,
        'mmd_after': mmd_after,
        'improvement_percent': improvement,
        'is_improved': mmd_after < mmd_before,
        'significant_improvement': improvement > 5.0  # 5%阈值
    }
    
    return validation_result
```

## 故障排除

### 常见问题

1. **数值不稳定**
   ```python
   # 问题: 核矩阵计算中出现数值溢出
   # 解决: 使用数值稳定的实现
   def stable_rbf_kernel(X, Y, gamma):
       # 避免直接计算exp(大数)
       distances = cdist(X, Y, metric='sqeuclidean')
       max_dist = np.max(distances)
       normalized_distances = distances - max_dist
       return np.exp(-gamma * normalized_distances) * np.exp(-gamma * max_dist)
   ```

2. **内存不足**
   ```python
   # 问题: 大数据集导致内存溢出
   # 解决: 使用分块计算
   if X.shape[0] * Y.shape[0] > 1e6:  # 大于100万个样本对
       mmd = compute_mmd_batch(X, Y, batch_size=1000)
   else:
       mmd = compute_mmd(X, Y)
   ```

3. **收敛问题**
   ```python
   # 问题: 线性变换训练不收敛
   # 解决: 调整学习率和正则化
   if not transformer.converged_:
       print("训练未收敛，尝试调整参数:")
       print("- 降低学习率")
       print("- 增加正则化")
       print("- 启用分阶段训练")
   ```

4. **GPU内存不足**
   ```python
   # 问题: GPU内存不足
   # 解决: 动态调整批大小
   try:
       result = compute_mmd_gpu(X, Y)
   except RuntimeError as e:
       if "out of memory" in str(e):
           print("GPU内存不足，切换到CPU计算")
           result = compute_mmd(X, Y)
   ```

## 最佳实践

### 1. 参数选择指南

```python
def get_recommended_parameters(X_source, X_target):
    """获取推荐的参数配置"""
    
    n_samples = min(len(X_source), len(X_target))
    n_features = X_source.shape[1]
    
    # 根据数据规模调整参数
    if n_samples < 1000:
        config = {
            'method': 'mean_std',  # 小数据集使用简单方法
            'n_epochs': 100
        }
    elif n_samples < 5000:
        config = {
            'method': 'linear',
            'n_epochs': 200,
            'lr': 3e-4,
            'batch_size': 64
        }
    else:
        config = {
            'method': 'linear',
            'n_epochs': 300,
            'lr': 1e-4,
            'batch_size': 128,
            'staged_training': True
        }
    
    # 根据特征数量调整gamma
    gamma = median_heuristic_gamma(X_source, X_target)
    config['gamma'] = gamma
    
    return config
```

### 2. 实验设计

```python
def comprehensive_mmd_experiment(X_source, X_target):
    """全面的MMD实验"""
    
    results = {}
    
    # 测试所有方法
    methods = ['mean_std', 'linear', 'kpca']
    
    for method in methods:
        print(f"测试{method}方法...")
        
        start_time = time.time()
        X_aligned, info = mmd_transform(
            X_source, X_target, method=method
        )
        end_time = time.time()
        
        results[method] = {
            'mmd_reduction': info['reduction'],
            'training_time': end_time - start_time,
            'final_mmd': info['mmd_after']
        }
    
    # 选择最佳方法
    best_method = max(results.keys(), 
                     key=lambda x: results[x]['mmd_reduction'])
    
    print(f"最佳方法: {best_method}")
    print(f"MMD减少: {results[best_method]['mmd_reduction']:.2f}%")
    
    return results, best_method
```

## 总结

`preprocessing/mmd.py` 模块提供了完整的MMD算法实现，主要特点包括：

- **理论完备**: 实现了MMD的各种变体和计算方法
- **实用性强**: 提供了多种域适应变换算法
- **性能优化**: 支持GPU加速和内存优化
- **易于使用**: 统一的接口和丰富的配置选项
- **可靠性高**: 完善的验证和错误处理机制

通过合理使用这些功能，可以有效地进行跨域数据适应，提高模型在目标域上的性能。 