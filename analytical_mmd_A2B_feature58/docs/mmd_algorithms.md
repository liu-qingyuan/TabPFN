# MMD算法文档 (MMD Algorithms Documentation)

## 概述

Maximum Mean Discrepancy (MMD) 是一种强大的统计度量方法，用于衡量两个概率分布之间的差异。本文档详细介绍项目中实现的各种MMD算法，包括理论基础、实现细节和使用指南。

## 文件结构

```python
preprocessing/
├── mmd.py                      # 核心MMD算法实现
├── class_conditional_mmd.py    # 类条件MMD实现
└── threshold_optimizer.py      # 阈值优化器
```

## 理论基础

### MMD定义

给定两个概率分布 P 和 Q，以及映射函数 φ: X → H（将输入空间映射到再生核希尔伯特空间 H），MMD定义为：

```
MMD²(P, Q) = ||μ_P - μ_Q||²_H
```

其中：
- μ_P = E_{x~P}[φ(x)] 是分布P在H中的均值嵌入
- μ_Q = E_{y~Q}[φ(y)] 是分布Q在H中的均值嵌入

### 经验估计

在实际应用中，我们使用有限样本来估计MMD：

```
MMD²(X, Y) = 1/n² ∑∑k(xᵢ,xⱼ) - 2/nm ∑∑k(xᵢ,yⱼ) + 1/m² ∑∑k(yᵢ,yⱼ)
```

其中：
- X = {x₁, ..., xₙ} 是源域样本
- Y = {y₁, ..., yₘ} 是目标域样本  
- k(·,·) 是核函数

## 核心算法实现

### 1. 基础MMD计算 (`mmd.py`)

#### compute_mmd()
```python
def compute_mmd(X: np.ndarray, Y: np.ndarray, kernel: str = 'rbf', 
                gamma: float = 1.0, degree: int = 3, coef0: float = 1.0) -> float:
    """
    计算两个数据集之间的MMD距离（无偏估计）
    
    参数:
    - X: 源域数据 (n_samples_1, n_features)
    - Y: 目标域数据 (n_samples_2, n_features) 
    - kernel: 核函数类型 ('rbf', 'linear', 'poly')
    - gamma: RBF/poly核参数
    - degree: 多项式核度数
    - coef0: 多项式/sigmoid核参数
    
    返回:
    - float: MMD距离值
    """
```

**实现特点**：
- 使用无偏估计避免过度拟合
- 支持多种核函数类型
- 自动处理不同样本大小
- 数值稳定的计算方法

#### median_heuristic_gamma()
```python
def median_heuristic_gamma(X: np.ndarray, Y: np.ndarray = None) -> float:
    """
    使用中值启发式计算RBF核的gamma参数
    
    参数:
    - X: 第一个数据集
    - Y: 第二个数据集（可选）
    
    返回:
    - float: 推荐的gamma值
    """
```

**原理**：
- 计算所有样本对之间的欧氏距离
- 使用距离的中值作为带宽参数
- gamma = 1 / (2 * median_distance²)

### 2. 线性MMD变换 (`MMDLinearTransform`)

#### 核心思想
学习一个线性变换矩阵 W，使得变换后的目标域数据与源域数据的MMD距离最小：

```
min_W MMD²(X_source, X_target @ W)
```

#### 关键特性

**分阶段训练** (`staged_training=True`):
```python
# 第一阶段：使用小gamma进行粗对齐
gamma_initial = min(gamma_search_values)
# 第二阶段：使用大gamma进行精细对齐  
gamma_final = max(gamma_search_values)
```

**动态gamma搜索** (`dynamic_gamma=True`):
```python
# 自动搜索最优gamma值
best_gamma = None
best_loss = float('inf')
for gamma in gamma_search_values:
    loss = compute_mmd_loss(X_source, X_target, gamma)
    if loss < best_loss:
        best_loss = loss
        best_gamma = gamma
```

**梯度裁剪** (`use_gradient_clipping=True`):
```python
# 防止梯度爆炸
if use_gradient_clipping:
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
```

#### 训练过程
```python
class MMDLinearTransform:
    def fit(self, X_source, X_target):
        """
        训练线性变换矩阵
        
        训练流程:
        1. 初始化变换矩阵W
        2. 如果启用分阶段训练：
           - 第一阶段：小gamma训练
           - 第二阶段：大gamma微调
        3. 如果启用动态gamma：
           - 搜索最优gamma值
        4. 优化MMD损失函数
        5. 记录训练历史
        """
```

### 3. 核PCA MMD变换

#### 原理
使用核主成分分析将两个域的数据映射到公共的核诱导特征空间：

```python
def mmd_kernel_pca_transform(X_source, X_target, kernel='rbf', 
                           gamma=1.0, n_components=10):
    """
    使用核PCA进行MMD域适应
    
    步骤:
    1. 合并源域和目标域数据
    2. 计算核矩阵K
    3. 执行核PCA降维
    4. 分别提取源域和目标域的降维结果
    """
```

**优势**：
- 处理非线性分布差异
- 降维去噪效果
- 计算相对稳定

**参数调优**：
- `n_components`: 保留的主成分数量
- `gamma`: RBF核参数
- `kernel`: 核函数类型

### 4. 均值标准差对齐

#### 简单而有效的方法
```python
def mmd_mean_std_transform(X_source, X_target):
    """
    简单的均值标准差对齐
    
    对每个特征：
    1. 计算源域的均值μ_s和标准差σ_s
    2. 计算目标域的均值μ_t和标准差σ_t  
    3. 变换：X_target_new = (X_target - μ_t) / σ_t * σ_s + μ_s
    """
```

**特点**：
- 计算速度极快
- 内存占用小
- 适合快速验证和基线对比
- 对特征规模敏感

### 5. 类条件MMD (`class_conditional_mmd.py`)

#### 核心思想
为每个类别分别进行域适应，保持类别特有的分布特征：

```python
def class_conditional_mmd_transform(X_source, y_source, X_target, 
                                  method='linear', use_pseudo_labels=True):
    """
    类条件MMD域适应
    
    流程:
    1. 为目标域生成伪标签（如果需要）
    2. 按类别分组处理
    3. 为每个类别独立进行MMD变换
    4. 合并结果
    """
```

#### 伪标签生成
```python
def generate_pseudo_labels(X_source, y_source, X_target, k=5):
    """
    使用k-NN生成伪标签
    
    参数:
    - k: 近邻数量
    
    返回:
    - y_pseudo: 伪标签
    - confidence: 置信度分数
    """
```

#### 优势与挑战
**优势**：
- 保持类别间的差异
- 提高类别不平衡数据的性能
- 支持部分标注的目标域数据

**挑战**：
- 伪标签质量影响效果
- 计算复杂度增加
- 需要合理的类别划分

### 6. 阈值优化 (`threshold_optimizer.py`)

#### Youden指数优化
```python
def find_optimal_threshold(y_true, y_scores):
    """
    使用Youden指数寻找最优分类阈值
    
    Youden指数 = Sensitivity + Specificity - 1
              = TPR + TNR - 1
    
    返回:
    - optimal_threshold: 最优阈值
    - best_youden: 最佳Youden指数值
    """
```

#### ROC曲线分析
```python
def get_roc_curve_data(y_true, y_scores):
    """
    计算ROC曲线数据
    
    返回:
    - fpr: 假阳性率
    - tpr: 真阳性率  
    - thresholds: 阈值数组
    - auc: AUC值
    """
```

## 统一接口函数

### mmd_transform()
```python
def mmd_transform(X_source: np.ndarray, X_target: np.ndarray,
                 method: str = 'linear', cat_idx: List[int] = None,
                 **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    统一的MMD变换接口
    
    参数:
    - X_source: 源域数据
    - X_target: 目标域数据
    - method: MMD方法 ('linear', 'kpca', 'mean_std')
    - cat_idx: 类别特征索引
    - **kwargs: 方法特定参数
    
    返回:
    - X_target_aligned: 对齐后的目标域数据
    - adaptation_info: 适应信息字典
    """
```

## 使用示例

### 基础使用
```python
from analytical_mmd_A2B_feature58.preprocessing.mmd import mmd_transform

# 线性MMD变换
X_target_aligned, info = mmd_transform(
    X_source=X_train,
    X_target=X_test, 
    method='linear',
    n_epochs=200,
    lr=3e-4,
    staged_training=True
)

print(f"MMD减少: {info['reduction']:.2f}%")
```

### 高级配置
```python
# 自定义参数的线性MMD
X_aligned, info = mmd_transform(
    X_source=X_train,
    X_target=X_test,
    method='linear',
    n_epochs=300,
    lr=1e-4,
    batch_size=32,
    lambda_reg=1e-2,
    staged_training=True,
    dynamic_gamma=True,
    gamma_search_values=[0.01, 0.05, 0.1, 0.5],
    use_gradient_clipping=True,
    max_grad_norm=0.5
)
```

### 核PCA MMD
```python
X_aligned, info = mmd_transform(
    X_source=X_train,
    X_target=X_test,
    method='kpca',
    kernel='rbf',
    gamma=0.1,
    n_components=15
)
```

### 类条件MMD
```python
from analytical_mmd_A2B_feature58.preprocessing.class_conditional_mmd import class_conditional_mmd_transform

X_aligned, info = class_conditional_mmd_transform(
    X_source=X_train,
    y_source=y_train,
    X_target=X_test,
    method='linear',
    use_pseudo_labels=True,
    k_neighbors=5
)
```

## 性能优化指南

### 内存优化
```python
# 大数据集的批处理策略
mmd_config = {
    'batch_size': 32,           # 减少批大小
    'n_components': 8,          # KPCA: 减少主成分
    'use_gradient_clipping': True,
    'standardize_features': True
}
```

### 计算加速
```python
# GPU加速设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 并行处理类条件MMD
class_conditional_config = {
    'n_jobs': -1,              # 并行处理类别
    'use_cache': True          # 缓存中间结果
}
```

### 参数调优策略

#### 学习率调度
```python
# 动态学习率
lr_schedule = {
    'initial_lr': 3e-4,
    'decay_factor': 0.9,
    'decay_steps': 50
}
```

#### Gamma值选择
```python
# 根据数据规模调整gamma候选值
if n_features < 10:
    gamma_values = [0.1, 0.5, 1.0]    # 小特征数
elif n_features < 50:  
    gamma_values = [0.01, 0.05, 0.1]  # 中等特征数
else:
    gamma_values = [0.001, 0.01, 0.05] # 大特征数
```

## 算法比较

| 方法 | 计算复杂度 | 内存需求 | 非线性能力 | 解释性 | 推荐场景 |
|------|-----------|----------|------------|--------|----------|
| 线性MMD | O(n²d) | 中等 | 无 | 高 | 平衡性能需求 |
| 核PCA MMD | O(n³) | 高 | 强 | 中等 | 非线性分布 |
| 均值标准差 | O(nd) | 低 | 无 | 高 | 快速基线 |
| 类条件MMD | O(Cn²d) | 高 | 中等 | 中等 | 类别不平衡 |

其中：
- n: 样本数量
- d: 特征维度  
- C: 类别数量

## 故障排除

### 常见问题

1. **收敛缓慢**
   ```python
   # 解决方案
   config = {
       'lr': 1e-3,              # 增加学习率
       'n_epochs': 300,         # 增加训练轮数
       'staged_training': True  # 启用分阶段训练
   }
   ```

2. **内存不足**
   ```python
   # 解决方案
   config = {
       'batch_size': 16,        # 减少批大小
       'method': 'mean_std'     # 使用简单方法
   }
   ```

3. **数值不稳定**
   ```python
   # 解决方案
   config = {
       'lambda_reg': 1e-2,      # 增加正则化
       'standardize_features': True,
       'use_gradient_clipping': True
   }
   ```

### 调试工具

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 监控训练过程
config = {
    'monitor_gradients': True,
    'save_history': True
}

# 验证结果
def validate_mmd_result(X_original, X_aligned, X_source):
    """验证MMD变换效果"""
    mmd_before = compute_mmd(X_source, X_original)
    mmd_after = compute_mmd(X_source, X_aligned)
    improvement = (mmd_before - mmd_after) / mmd_before * 100
    print(f"MMD改进: {improvement:.2f}%")
```

## 最佳实践

1. **方法选择**：
   - 首选线性MMD进行初步尝试
   - 数据有明显非线性关系时使用核PCA
   - 快速验证使用均值标准差对齐

2. **参数调优**：
   - 从默认参数开始
   - 使用网格搜索优化关键参数
   - 监控训练过程避免过拟合

3. **效果评估**：
   - 计算MMD减少百分比
   - 评估下游任务性能
   - 可视化分布对齐效果

4. **计算资源管理**：
   - 根据硬件条件选择合适的批大小
   - 大数据集考虑使用简化方法
   - 启用梯度裁剪提高稳定性 