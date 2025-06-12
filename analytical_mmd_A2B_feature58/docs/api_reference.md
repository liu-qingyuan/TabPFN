# API参考文档 (API Reference)

## 概述

本文档详细介绍项目中所有公共API的使用方法、参数说明和返回值。所有API按模块组织，提供完整的函数签名和使用示例。

## 模块导入

```python
# 数据加载
from analytical_mmd_A2B_feature58.data.loader import load_excel, load_all_datasets

# MMD算法
from analytical_mmd_A2B_feature58.preprocessing.mmd import (
    compute_mmd, mmd_transform, MMDLinearTransform
)

# 模型选择
from analytical_mmd_A2B_feature58.modeling.model_selector import get_model

# 跨域实验
from analytical_mmd_A2B_feature58.modeling.cross_domain_runner import (
    CrossDomainExperimentRunner, run_cross_domain_experiment
)

# 配置管理
from analytical_mmd_A2B_feature58.config.settings import (
    get_features_by_type, get_model_config
)
```

## 数据加载模块 (data.loader)

### load_excel()

```python
def load_excel(
    file_path: str, 
    features: List[str], 
    label_col: str,
    sheet_name: Union[str, int] = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从Excel文件加载数据
    
    参数:
    - file_path (str): Excel文件路径
    - features (List[str]): 要加载的特征列名列表
    - label_col (str): 标签列名
    - sheet_name (Union[str, int]): 工作表名或索引，默认为0
    
    返回:
    - Tuple[np.ndarray, np.ndarray]: (特征矩阵, 标签向量)
    
    异常:
    - FileNotFoundError: 文件不存在
    - KeyError: 指定的列不存在
    - ValueError: 数据格式错误
    
    示例:
    >>> X, y = load_excel(
    ...     'data.xlsx', 
    ...     ['Feature1', 'Feature2'], 
    ...     'Label'
    ... )
    >>> print(f"数据形状: {X.shape}, 标签数量: {len(y)}")
    """
```

### load_all_datasets()

```python
def load_all_datasets(
    data_path_a: str,
    data_path_b: str, 
    features: List[str],
    label_col: str,
    validate_features: bool = True
) -> Dict[str, np.ndarray]:
    """
    加载所有数据集
    
    参数:
    - data_path_a (str): 数据集A的文件路径
    - data_path_b (str): 数据集B的文件路径
    - features (List[str]): 特征列表
    - label_col (str): 标签列名
    - validate_features (bool): 是否验证特征完整性
    
    返回:
    - Dict[str, np.ndarray]: 包含以下键的字典:
        - 'X_A': 数据集A的特征
        - 'y_A': 数据集A的标签
        - 'X_B': 数据集B的特征  
        - 'y_B': 数据集B的标签
    
    示例:
    >>> datasets = load_all_datasets(
    ...     'dataset_A.xlsx',
    ...     'dataset_B.xlsx', 
    ...     SELECTED_FEATURES,
    ...     'Label'
    ... )
    >>> print(f"数据集A: {datasets['X_A'].shape}")
    >>> print(f"数据集B: {datasets['X_B'].shape}")
    """
```

## MMD算法模块 (preprocessing.mmd)

### compute_mmd()

```python
def compute_mmd(
    X: np.ndarray, 
    Y: np.ndarray, 
    kernel: str = 'rbf',
    gamma: float = 1.0, 
    degree: int = 3, 
    coef0: float = 1.0
) -> float:
    """
    计算两个数据集之间的MMD距离
    
    参数:
    - X (np.ndarray): 源域数据，形状为 (n_samples_1, n_features)
    - Y (np.ndarray): 目标域数据，形状为 (n_samples_2, n_features)
    - kernel (str): 核函数类型，可选值:
        - 'rbf': 径向基函数核 (默认)
        - 'linear': 线性核
        - 'poly': 多项式核
    - gamma (float): RBF/poly核参数，默认1.0
    - degree (int): 多项式核的度数，默认3
    - coef0 (float): 多项式/sigmoid核的独立项，默认1.0
    
    返回:
    - float: MMD距离值，非负数
    
    注意:
    - 使用无偏估计，适合小样本情况
    - 返回值越小表示两个分布越相似
    
    示例:
    >>> mmd_distance = compute_mmd(X_source, X_target, kernel='rbf', gamma=0.1)
    >>> print(f"MMD距离: {mmd_distance:.6f}")
    """
```

### median_heuristic_gamma()

```python
def median_heuristic_gamma(
    X: np.ndarray, 
    Y: np.ndarray = None
) -> float:
    """
    使用中值启发式计算RBF核的gamma参数
    
    参数:
    - X (np.ndarray): 第一个数据集
    - Y (np.ndarray, optional): 第二个数据集，如果为None则仅使用X
    
    返回:
    - float: 推荐的gamma值
    
    原理:
    - 计算所有样本对之间的欧氏距离
    - gamma = 1 / (2 * median_distance²)
    
    示例:
    >>> gamma = median_heuristic_gamma(X_source, X_target)
    >>> print(f"推荐gamma值: {gamma:.6f}")
    """
```

### mmd_transform()

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
    - method (str): MMD方法，可选值:
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
    
    示例:
    >>> X_aligned, info = mmd_transform(
    ...     X_source, X_target, 
    ...     method='linear',
    ...     n_epochs=200,
    ...     lr=3e-4
    ... )
    >>> print(f"MMD减少: {info['reduction']:.2f}%")
    """
```

### MMDLinearTransform类

```python
class MMDLinearTransform:
    """线性MMD变换器"""
    
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
        monitor_gradients: bool = True
    ):
        """
        初始化线性MMD变换器
        
        参数:
        - input_dim (int): 输入特征维度
        - n_epochs (int): 训练轮数
        - lr (float): 学习率
        - batch_size (int): 批大小
        - lambda_reg (float): 正则化系数
        - gamma (float): RBF核参数
        - device (str): 计算设备 ('cuda' 或 'cpu')
        - staged_training (bool): 是否使用分阶段训练
        - dynamic_gamma (bool): 是否动态搜索gamma
        - gamma_search_values (List[float]): gamma候选值
        - standardize_features (bool): 是否标准化特征
        - use_gradient_clipping (bool): 是否使用梯度裁剪
        - max_grad_norm (float): 最大梯度范数
        - monitor_gradients (bool): 是否监控梯度
        """
    
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
        """
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        应用变换
        
        参数:
        - X (np.ndarray): 待变换的数据
        
        返回:
        - np.ndarray: 变换后的数据
        """
    
    def fit_transform(
        self, 
        X_source: np.ndarray, 
        X_target: np.ndarray
    ) -> np.ndarray:
        """
        训练并应用变换
        
        参数:
        - X_source (np.ndarray): 源域数据
        - X_target (np.ndarray): 目标域数据
        
        返回:
        - np.ndarray: 变换后的目标域数据
        """
```

## 模型选择模块 (modeling.model_selector)

### get_model()

```python
def get_model(
    model_type: str,
    categorical_feature_indices: Optional[List[int]] = None,
    **kwargs: Any
):
    """
    获取指定类型的模型实例
    
    参数:
    - model_type (str): 模型类型，可选值:
        - 'auto': AutoTabPFN
        - 'tuned': TunedTabPFN
        - 'base': 原生TabPFN
        - 'rf': RF风格TabPFN集成
    - categorical_feature_indices (List[int], optional): 类别特征索引
    - **kwargs: 模型特定参数
    
    返回:
    - 模型实例: 具有fit/predict/predict_proba方法的模型对象
    
    异常:
    - ValueError: 不支持的模型类型
    - ImportError: 所需的包未安装
    
    示例:
    >>> model = get_model('auto', categorical_feature_indices=[0, 2])
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """
```

### get_available_models()

```python
def get_available_models() -> List[str]:
    """
    获取当前环境中可用的模型类型
    
    返回:
    - List[str]: 可用模型类型列表
    
    示例:
    >>> available = get_available_models()
    >>> print(f"可用模型: {available}")
    """
```

## 跨域实验模块 (modeling.cross_domain_runner)

### CrossDomainExperimentRunner类

```python
class CrossDomainExperimentRunner:
    """跨域实验运行器"""
    
    def __init__(
        self,
        model_type: str = 'auto',
        feature_type: str = 'best7',
        use_mmd_adaptation: bool = True,
        mmd_method: str = 'linear',
        use_class_conditional: bool = False,
        use_threshold_optimizer: bool = False,
        save_path: str = './results_cross_domain',
        skip_cv_on_a: bool = False,
        evaluation_mode: str = 'cv',
        **kwargs: Any
    ):
        """
        初始化跨域实验运行器
        
        参数:
        - model_type (str): 模型类型
        - feature_type (str): 特征类型 ('all' 或 'best7')
        - use_mmd_adaptation (bool): 是否使用MMD域适应
        - mmd_method (str): MMD方法
        - use_class_conditional (bool): 是否使用类条件MMD
        - use_threshold_optimizer (bool): 是否使用阈值优化
        - save_path (str): 结果保存路径
        - skip_cv_on_a (bool): 是否跳过源域交叉验证
        - evaluation_mode (str): 评估模式 ('cv' 或 'single')
        - **kwargs: 其他参数
        """
    
    def load_datasets(self) -> Dict[str, np.ndarray]:
        """
        加载所有数据集
        
        返回:
        - Dict[str, np.ndarray]: 包含所有数据集的字典
        """
    
    def run_cross_validation(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        cv_folds: int = 10
    ) -> Dict[str, Any]:
        """
        在源域数据上运行交叉验证
        
        参数:
        - X (np.ndarray): 特征数据
        - y (np.ndarray): 标签数据
        - cv_folds (int): 交叉验证折数
        
        返回:
        - Dict[str, Any]: 交叉验证结果
        """
    
    def evaluate_external_dataset_cv(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray, 
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_train_raw: np.ndarray,
        X_test_raw: np.ndarray,
        dataset_name: str
    ) -> Dict[str, Any]:
        """
        在目标域数据上进行外部验证
        
        参数:
        - X_train (np.ndarray): 训练集特征
        - y_train (np.ndarray): 训练集标签
        - X_test (np.ndarray): 测试集特征
        - y_test (np.ndarray): 测试集标签
        - X_train_raw (np.ndarray): 原始训练集特征（用于域适应）
        - X_test_raw (np.ndarray): 原始测试集特征（用于域适应）
        - dataset_name (str): 数据集名称
        
        返回:
        - Dict[str, Any]: 外部验证结果
        """
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """
        运行完整的跨域实验
        
        返回:
        - Dict[str, Any]: 完整实验结果
        """
```

### run_cross_domain_experiment()

```python
def run_cross_domain_experiment(
    model_type: str = 'auto',
    feature_type: str = 'best7', 
    mmd_method: str = 'linear',
    use_class_conditional: bool = False,
    use_threshold_optimizer: bool = False,
    save_path: str = './results_cross_domain',
    **kwargs: Any
) -> Dict[str, Any]:
    """
    运行跨域实验的便捷函数
    
    参数:
    - model_type (str): 模型类型
    - feature_type (str): 特征类型
    - mmd_method (str): MMD方法
    - use_class_conditional (bool): 是否使用类条件MMD
    - use_threshold_optimizer (bool): 是否使用阈值优化
    - save_path (str): 结果保存路径
    - **kwargs: 其他参数
    
    返回:
    - Dict[str, Any]: 实验结果字典
    
    示例:
    >>> results = run_cross_domain_experiment(
    ...     model_type='auto',
    ...     feature_type='best7',
    ...     mmd_method='linear'
    ... )
    >>> print(f"源域AUC: {results['cross_validation_A']['auc']}")
    """
```

## 配置管理模块 (config.settings)

### get_features_by_type()

```python
def get_features_by_type(feature_type: str = 'all') -> List[str]:
    """
    根据类型获取特征列表
    
    参数:
    - feature_type (str): 特征类型，可选值:
        - 'all': 所有58个特征
        - 'best7': 最佳7个特征
        - 'categorical': 仅类别特征
    
    返回:
    - List[str]: 特征名称列表
    
    异常:
    - ValueError: 不支持的特征类型
    
    示例:
    >>> features = get_features_by_type('best7')
    >>> print(f"最佳7特征: {features}")
    """
```

### get_categorical_indices()

```python
def get_categorical_indices(feature_type: str = 'all') -> List[int]:
    """
    根据特征类型获取类别特征索引
    
    参数:
    - feature_type (str): 特征类型 ('all' 或 'best7')
    
    返回:
    - List[int]: 类别特征在特征列表中的索引
    
    示例:
    >>> cat_indices = get_categorical_indices('best7')
    >>> print(f"类别特征索引: {cat_indices}")
    """
```

### get_model_config()

```python
def get_model_config(
    model_type: str,
    preset: str = 'balanced',
    categorical_feature_indices: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    获取模型配置
    
    参数:
    - model_type (str): 模型类型 ('auto', 'base', 'rf')
    - preset (str): 预设类型，可选值:
        - 'fast': 快速配置
        - 'balanced': 平衡配置 (默认)
        - 'accurate': 高精度配置
    - categorical_feature_indices (List[int], optional): 类别特征索引
    
    返回:
    - Dict[str, Any]: 模型配置字典
    
    示例:
    >>> config = get_model_config('auto', 'balanced')
    >>> print(f"模型配置: {config}")
    """
```

## 评估指标模块 (metrics)

### evaluate_model_on_external_cv()

```python
def evaluate_model_on_external_cv(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 10
) -> Dict[str, Any]:
    """
    在外部数据集上进行交叉验证评估
    
    参数:
    - model: 已训练的模型
    - X_test (np.ndarray): 测试集特征
    - y_test (np.ndarray): 测试集标签
    - cv_folds (int): 交叉验证折数
    
    返回:
    - Dict[str, Any]: 评估结果字典，包含:
        - 'means': 各指标的均值
        - 'stds': 各指标的标准差
        - 'individual_scores': 每折的详细分数
    
    示例:
    >>> results = evaluate_model_on_external_cv(model, X_test, y_test)
    >>> print(f"平均AUC: {results['means']['auc']:.4f}")
    """
```

## 可视化模块 (visualization)

### visualize_tsne()

```python
def visualize_tsne(
    X_before: np.ndarray,
    X_after: np.ndarray,
    y_source: np.ndarray,
    y_target: np.ndarray,
    save_path: str,
    title: str = "t-SNE Visualization",
    perplexity: int = 30,
    n_iter: int = 1000
) -> None:
    """
    生成域适应前后的t-SNE可视化对比
    
    参数:
    - X_before (np.ndarray): 适应前的目标域数据
    - X_after (np.ndarray): 适应后的目标域数据
    - y_source (np.ndarray): 源域标签
    - y_target (np.ndarray): 目标域标签
    - save_path (str): 图片保存路径
    - title (str): 图片标题
    - perplexity (int): t-SNE困惑度参数
    - n_iter (int): t-SNE迭代次数
    
    示例:
    >>> visualize_tsne(
    ...     X_target_original, X_target_aligned,
    ...     y_source, y_target,
    ...     'tsne_comparison.png'
    ... )
    """
```

## 工具函数

### 数据验证

```python
def validate_data_compatibility(
    X_source: np.ndarray,
    X_target: np.ndarray
) -> bool:
    """
    验证源域和目标域数据的兼容性
    
    参数:
    - X_source (np.ndarray): 源域数据
    - X_target (np.ndarray): 目标域数据
    
    返回:
    - bool: 数据是否兼容
    
    检查项目:
    - 特征维度是否匹配
    - 数据类型是否一致
    - 是否包含NaN值
    """
```

### 结果保存

```python
def save_experiment_results(
    results: Dict[str, Any],
    save_path: str,
    format: str = 'json'
) -> None:
    """
    保存实验结果
    
    参数:
    - results (Dict[str, Any]): 实验结果字典
    - save_path (str): 保存路径
    - format (str): 保存格式 ('json', 'pickle', 'txt')
    
    示例:
    >>> save_experiment_results(results, './results.json')
    """
```

## 错误处理

### 自定义异常

```python
class MMDConvergenceError(Exception):
    """MMD训练收敛错误"""
    pass

class IncompatibleDataError(Exception):
    """数据不兼容错误"""
    pass

class ModelNotAvailableError(Exception):
    """模型不可用错误"""
    pass
```

## 类型定义

```python
from typing import Union, List, Dict, Any, Optional, Tuple

# 数据类型
DataMatrix = np.ndarray
Labels = np.ndarray
FeatureList = List[str]
CategoricalIndices = List[int]

# 结果类型
ExperimentResults = Dict[str, Any]
ModelConfig = Dict[str, Any]
AdaptationInfo = Dict[str, Any]

# 模型类型
ModelType = Union[str, object]
```

## 使用示例汇总

### 完整工作流程

```python
from analytical_mmd_A2B_feature58 import *

# 1. 加载数据
datasets = load_all_datasets(
    'dataset_A.xlsx',
    'dataset_B.xlsx',
    get_features_by_type('best7'),
    'Label'
)

# 2. 进行MMD域适应
X_aligned, info = mmd_transform(
    datasets['X_A'],
    datasets['X_B'],
    method='linear',
    cat_idx=get_categorical_indices('best7')
)

# 3. 训练模型
model = get_model('auto', categorical_feature_indices=[0, 2])
model.fit(datasets['X_A'], datasets['y_A'])

# 4. 评估性能
results = evaluate_model_on_external_cv(
    model, X_aligned, datasets['y_B']
)

# 5. 运行完整跨域实验
experiment_results = run_cross_domain_experiment(
    model_type='auto',
    feature_type='best7',
    mmd_method='linear'
)

print(f"实验完成，AUC改进: {experiment_results['improvement']:.3f}")
```

## 版本兼容性

| API函数 | 最低版本 | 变更说明 |
|---------|----------|----------|
| `get_model()` | v1.0.0 | 初始版本 |
| `mmd_transform()` | v1.0.0 | 初始版本 |
| `run_cross_domain_experiment()` | v1.1.0 | 新增函数 |
| `CrossDomainExperimentRunner` | v1.1.0 | 新增类 |

## 性能注意事项

1. **内存使用**：大数据集时建议使用批处理
2. **计算时间**：复杂模型可能需要较长训练时间
3. **GPU支持**：确保CUDA环境正确配置
4. **并行处理**：部分函数支持多进程加速

## 最佳实践

1. **参数验证**：使用 `validate_model_params()` 验证参数
2. **错误处理**：捕获并处理自定义异常
3. **资源管理**：及时释放大对象的内存
4. **结果保存**：使用标准化的结果保存格式
5. **日志记录**：启用详细日志便于调试
``` 