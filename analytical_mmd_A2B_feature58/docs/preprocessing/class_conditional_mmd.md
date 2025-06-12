# preprocessing/class_conditional_mmd.py 文档

## 概述

类条件MMD模块，按类别分别进行域适应，提高跨域性能。

## 主要函数

### generate_pseudo_labels()

```python
def generate_pseudo_labels(X_source: np.ndarray, y_source: np.ndarray, 
                          X_target: np.ndarray, method: str = 'knn', 
                          **kwargs) -> np.ndarray:
```

为目标域生成伪标签。

**参数:**
- `X_source`: 源域特征
- `y_source`: 源域标签
- `X_target`: 目标域特征
- `method`: 伪标签生成方法，默认'knn'

**返回:**
- `pseudo_labels`: 目标域伪标签

**使用示例:**
```python
from analytical_mmd_A2B_feature58.preprocessing.class_conditional_mmd import generate_pseudo_labels

pseudo_labels = generate_pseudo_labels(X_source, y_source, X_target, method='knn', n_neighbors=5)
```

### create_partial_labels()

```python
def create_partial_labels(y_target: np.ndarray, label_ratio: float = 0.1, 
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
```

创建部分标签（分层采样）。

**参数:**
- `y_target`: 目标域真实标签
- `label_ratio`: 使用的标签比例
- `random_state`: 随机种子

**返回:**
- `partial_labels`: 部分标签数组（-1表示未标记）
- `labeled_indices`: 已标记样本的索引

### class_conditional_mmd_transform()

```python
def class_conditional_mmd_transform(X_source: np.ndarray, y_source: np.ndarray, 
                                   X_target: np.ndarray, 
                                   target_labels: Optional[np.ndarray] = None,
                                   use_partial_labels: bool = False,
                                   label_ratio: float = 0.1,
                                   method: str = 'linear', 
                                   cat_idx: Optional[list] = None,
                                   random_state: int = 42,
                                   **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
```

类条件MMD变换，为每个类别分别进行域适应。

**参数:**
- `X_source`: 源域特征
- `y_source`: 源域标签
- `X_target`: 目标域特征
- `target_labels`: 目标域标签（可选）
- `use_partial_labels`: 是否使用部分真实标签
- `method`: MMD对齐方法 ('linear', 'kpca', 'mean_std')
- `cat_idx`: 类别特征索引

**返回:**
- `X_target_aligned`: 类条件对齐后的目标域特征
- `mmd_info`: MMD相关信息

**使用示例:**
```python
from analytical_mmd_A2B_feature58.preprocessing.class_conditional_mmd import class_conditional_mmd_transform

X_aligned, info = class_conditional_mmd_transform(
    X_source, y_source, X_target,
    method='linear',
    cat_idx=[0, 2, 4]
)
```

### run_class_conditional_mmd_experiment()

```python
def run_class_conditional_mmd_experiment(X_source: np.ndarray, y_source: np.ndarray,
                                        X_target: np.ndarray, y_target: np.ndarray,
                                        tabpfn_model,
                                        method: str = 'linear',
                                        use_partial_labels: bool = False,
                                        label_ratio: float = 0.1,
                                        cat_idx: Optional[list] = None,
                                        random_state: int = 42,
                                        **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
```

运行完整的类条件MMD实验。

**参数:**
- `X_source`, `y_source`: 源域数据和标签
- `X_target`, `y_target`: 目标域数据和标签
- `tabpfn_model`: TabPFN模型
- `method`: MMD方法
- `use_partial_labels`: 是否使用部分标签
- `cat_idx`: 类别特征索引

**返回:**
- `y_pred`: 预测结果
- `y_proba`: 预测概率
- `experiment_info`: 实验信息 