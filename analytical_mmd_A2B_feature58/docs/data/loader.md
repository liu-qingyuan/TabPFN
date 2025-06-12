# data/loader.py 文档

## 概述

`data/loader.py` 是项目的数据加载核心模块，负责从Excel文件中加载医疗数据集，进行数据预处理和验证。本模块提供了统一的数据加载接口，支持多种数据格式和验证机制。

## 文件位置
```
analytical_mmd_A2B_feature58/data/loader.py
```

## 主要功能

### 1. Excel数据加载
- 支持多种Excel格式（.xlsx, .xls）
- 灵活的工作表选择
- 自动数据类型推断
- 缺失值处理

### 2. 数据验证
- 特征完整性检查
- 数据类型验证
- 缺失值统计
- 数据分布分析

### 3. 多数据集管理
- 同时加载多个数据集
- 数据集间特征对齐
- 标签一致性检查
- 数据集元信息管理

## 核心函数

### load_excel()

```python
def load_excel(
    file_path: str, 
    features: List[str], 
    label_col: str,
    sheet_name: Union[str, int] = 0,
    validate_data: bool = True,
    handle_missing: str = 'raise'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从Excel文件加载数据
    
    参数:
    - file_path (str): Excel文件的绝对或相对路径
    - features (List[str]): 要加载的特征列名列表
    - label_col (str): 标签列名
    - sheet_name (Union[str, int]): 工作表名称或索引，默认为0（第一个工作表）
    - validate_data (bool): 是否进行数据验证，默认True
    - handle_missing (str): 缺失值处理策略
        - 'raise': 发现缺失值时抛出异常（默认）
        - 'drop': 删除包含缺失值的行
        - 'fill': 使用默认值填充缺失值
    
    返回:
    - Tuple[np.ndarray, np.ndarray]: (特征矩阵, 标签向量)
        - 特征矩阵: 形状为 (n_samples, n_features)
        - 标签向量: 形状为 (n_samples,)
    
    异常:
    - FileNotFoundError: 文件不存在
    - KeyError: 指定的列不存在
    - ValueError: 数据格式错误或包含缺失值
    - pd.errors.ExcelFileError: Excel文件格式错误
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.data.loader import load_excel
from analytical_mmd_A2B_feature58.config.settings import BEST_7_FEATURES, LABEL_COL

# 基础使用
X, y = load_excel(
    file_path='data/AI4healthcare.xlsx',
    features=BEST_7_FEATURES,
    label_col=LABEL_COL
)

# 指定工作表
X, y = load_excel(
    file_path='data/medical_data.xlsx',
    features=BEST_7_FEATURES,
    label_col=LABEL_COL,
    sheet_name='Sheet2'  # 或者使用索引: sheet_name=1
)

# 处理缺失值
X, y = load_excel(
    file_path='data/incomplete_data.xlsx',
    features=BEST_7_FEATURES,
    label_col=LABEL_COL,
    handle_missing='drop'  # 删除包含缺失值的行
)
```

### load_all_datasets()

```python
def load_all_datasets(
    data_path_a: str,
    data_path_b: str,
    features: List[str],
    label_col: str,
    validate_features: bool = True,
    ensure_compatibility: bool = True,
    return_metadata: bool = False
) -> Union[Dict[str, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    """
    加载所有数据集并进行兼容性检查
    
    参数:
    - data_path_a (str): 数据集A（源域）的文件路径
    - data_path_b (str): 数据集B（目标域）的文件路径
    - features (List[str]): 要加载的特征列表
    - label_col (str): 标签列名
    - validate_features (bool): 是否验证特征完整性，默认True
    - ensure_compatibility (bool): 是否确保数据集间兼容性，默认True
    - return_metadata (bool): 是否返回数据集元信息，默认False
    
    返回:
    - Dict[str, np.ndarray]: 包含以下键的数据字典:
        - 'X_A': 数据集A的特征矩阵
        - 'y_A': 数据集A的标签向量
        - 'X_B': 数据集B的特征矩阵
        - 'y_B': 数据集B的标签向量
    - 如果return_metadata=True，还返回元信息字典
    
    异常:
    - ValueError: 数据集不兼容或特征不匹配
    - FileNotFoundError: 数据文件不存在
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.data.loader import load_all_datasets
from analytical_mmd_A2B_feature58.config.settings import (
    DATA_PATHS, BEST_7_FEATURES, LABEL_COL
)

# 基础使用
datasets = load_all_datasets(
    data_path_a=DATA_PATHS['A'],
    data_path_b=DATA_PATHS['B'],
    features=BEST_7_FEATURES,
    label_col=LABEL_COL
)

print(f"数据集A形状: {datasets['X_A'].shape}")
print(f"数据集B形状: {datasets['X_B'].shape}")

# 获取元信息
datasets, metadata = load_all_datasets(
    data_path_a=DATA_PATHS['A'],
    data_path_b=DATA_PATHS['B'],
    features=BEST_7_FEATURES,
    label_col=LABEL_COL,
    return_metadata=True
)

print(f"数据集A样本数: {metadata['dataset_A']['n_samples']}")
print(f"数据集B类别分布: {metadata['dataset_B']['class_distribution']}")
```

### validate_dataset()

```python
def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    features: List[str],
    dataset_name: str = "Unknown"
) -> Dict[str, Any]:
    """
    验证数据集的完整性和质量
    
    参数:
    - X (np.ndarray): 特征矩阵
    - y (np.ndarray): 标签向量
    - features (List[str]): 特征名称列表
    - dataset_name (str): 数据集名称，用于错误报告
    
    返回:
    - Dict[str, Any]: 验证结果字典，包含:
        - 'is_valid': 是否通过验证
        - 'n_samples': 样本数量
        - 'n_features': 特征数量
        - 'missing_values': 缺失值统计
        - 'class_distribution': 类别分布
        - 'feature_types': 特征类型信息
        - 'warnings': 警告信息列表
        - 'errors': 错误信息列表
    """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.data.loader import validate_dataset

# 验证数据集
validation_result = validate_dataset(
    X=datasets['X_A'],
    y=datasets['y_A'],
    features=BEST_7_FEATURES,
    dataset_name="AI4Health"
)

if validation_result['is_valid']:
    print("数据集验证通过")
    print(f"样本数: {validation_result['n_samples']}")
    print(f"特征数: {validation_result['n_features']}")
    print(f"类别分布: {validation_result['class_distribution']}")
else:
    print("数据集验证失败:")
    for error in validation_result['errors']:
        print(f"  错误: {error}")
    for warning in validation_result['warnings']:
        print(f"  警告: {warning}")
```

### check_dataset_compatibility()

```python
def check_dataset_compatibility(
    X_a: np.ndarray,
    X_b: np.ndarray,
    features: List[str],
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    检查两个数据集的兼容性
    
    参数:
    - X_a (np.ndarray): 第一个数据集的特征矩阵
    - X_b (np.ndarray): 第二个数据集的特征矩阵
    - features (List[str]): 特征名称列表
    - tolerance (float): 数值比较的容差，默认1e-6
    
    返回:
    - Dict[str, Any]: 兼容性检查结果，包含:
        - 'compatible': 是否兼容
        - 'feature_dimension_match': 特征维度是否匹配
        - 'feature_ranges': 各特征的值域比较
        - 'distribution_similarity': 分布相似性分析
        - 'recommendations': 改进建议
    """
```

## 数据预处理功能

### preprocess_features()

```python
def preprocess_features(
    X: np.ndarray,
    features: List[str],
    categorical_indices: List[int] = None,
    normalize: bool = True,
    handle_outliers: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    对特征进行预处理
    
    参数:
    - X (np.ndarray): 原始特征矩阵
    - features (List[str]): 特征名称列表
    - categorical_indices (List[int]): 类别特征索引
    - normalize (bool): 是否标准化数值特征
    - handle_outliers (bool): 是否处理异常值
    
    返回:
    - Tuple[np.ndarray, Dict[str, Any]]: (预处理后的特征矩阵, 预处理信息)
    """
```

### encode_categorical_features()

```python
def encode_categorical_features(
    X: np.ndarray,
    categorical_indices: List[int],
    method: str = 'label'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    编码类别特征
    
    参数:
    - X (np.ndarray): 特征矩阵
    - categorical_indices (List[int]): 类别特征索引
    - method (str): 编码方法
        - 'label': 标签编码
        - 'onehot': 独热编码
        - 'target': 目标编码
    
    返回:
    - Tuple[np.ndarray, Dict[str, Any]]: (编码后的特征矩阵, 编码器信息)
    """
```

## 数据统计和分析

### get_dataset_statistics()

```python
def get_dataset_statistics(
    X: np.ndarray,
    y: np.ndarray,
    features: List[str]
) -> Dict[str, Any]:
    """
    获取数据集的详细统计信息
    
    参数:
    - X (np.ndarray): 特征矩阵
    - y (np.ndarray): 标签向量
    - features (List[str]): 特征名称列表
    
    返回:
    - Dict[str, Any]: 统计信息字典，包含:
        - 'basic_stats': 基础统计（均值、标准差、最值等）
        - 'class_distribution': 类别分布
        - 'feature_correlations': 特征相关性矩阵
        - 'missing_patterns': 缺失值模式
        - 'outlier_detection': 异常值检测结果
    """
```

### compare_datasets()

```python
def compare_datasets(
    X_a: np.ndarray,
    y_a: np.ndarray,
    X_b: np.ndarray,
    y_b: np.ndarray,
    features: List[str],
    dataset_names: Tuple[str, str] = ("Dataset A", "Dataset B")
) -> Dict[str, Any]:
    """
    比较两个数据集的差异
    
    参数:
    - X_a, y_a: 第一个数据集的特征和标签
    - X_b, y_b: 第二个数据集的特征和标签
    - features: 特征名称列表
    - dataset_names: 数据集名称元组
    
    返回:
    - Dict[str, Any]: 比较结果，包含:
        - 'size_comparison': 样本数量比较
        - 'distribution_differences': 分布差异分析
        - 'statistical_tests': 统计检验结果
        - 'domain_gap_metrics': 域差距度量
    """
```

## 缓存和性能优化

### DataLoader类

```python
class DataLoader:
    """
    高级数据加载器，支持缓存和批处理
    """
    
    def __init__(
        self,
        cache_dir: str = './cache',
        enable_cache: bool = True,
        cache_format: str = 'pickle'
    ):
        """
        初始化数据加载器
        
        参数:
        - cache_dir: 缓存目录
        - enable_cache: 是否启用缓存
        - cache_format: 缓存格式 ('pickle', 'hdf5', 'parquet')
        """
    
    def load_with_cache(
        self,
        file_path: str,
        features: List[str],
        label_col: str,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        带缓存的数据加载
        
        如果缓存存在且有效，直接从缓存加载；
        否则从原始文件加载并创建缓存。
        """
    
    def clear_cache(self, file_path: str = None):
        """
        清理缓存
        
        参数:
        - file_path: 特定文件的缓存，None表示清理所有缓存
        """
```

#### 使用示例

```python
from analytical_mmd_A2B_feature58.data.loader import DataLoader

# 创建数据加载器
loader = DataLoader(cache_dir='./data_cache', enable_cache=True)

# 首次加载（从Excel文件）
X, y = loader.load_with_cache(
    file_path=DATA_PATHS['A'],
    features=BEST_7_FEATURES,
    label_col=LABEL_COL
)

# 再次加载（从缓存）
X, y = loader.load_with_cache(
    file_path=DATA_PATHS['A'],
    features=BEST_7_FEATURES,
    label_col=LABEL_COL
)  # 这次会更快，因为使用了缓存
```

## 错误处理和日志

### 常见异常类型

```python
class DataLoadingError(Exception):
    """数据加载错误基类"""
    pass

class FeatureMismatchError(DataLoadingError):
    """特征不匹配错误"""
    pass

class DataValidationError(DataLoadingError):
    """数据验证错误"""
    pass

class IncompatibleDatasetError(DataLoadingError):
    """数据集不兼容错误"""
    pass
```

### 日志配置

```python
import logging

# 配置数据加载器日志
logger = logging.getLogger('data_loader')
logger.setLevel(logging.INFO)

# 使用示例
def load_excel_with_logging(file_path, features, label_col):
    """带日志的数据加载"""
    logger.info(f"开始加载数据: {file_path}")
    logger.info(f"特征数量: {len(features)}")
    
    try:
        X, y = load_excel(file_path, features, label_col)
        logger.info(f"数据加载成功: {X.shape[0]}样本, {X.shape[1]}特征")
        return X, y
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise
```

## 使用示例和最佳实践

### 完整的数据加载流程

```python
from analytical_mmd_A2B_feature58.data.loader import (
    load_all_datasets, validate_dataset, check_dataset_compatibility
)
from analytical_mmd_A2B_feature58.config.settings import (
    DATA_PATHS, BEST_7_FEATURES, LABEL_COL
)

def load_and_validate_data():
    """完整的数据加载和验证流程"""
    
    # 1. 加载所有数据集
    print("正在加载数据集...")
    datasets = load_all_datasets(
        data_path_a=DATA_PATHS['A'],
        data_path_b=DATA_PATHS['B'],
        features=BEST_7_FEATURES,
        label_col=LABEL_COL,
        validate_features=True
    )
    
    # 2. 验证各个数据集
    print("验证数据集A...")
    validation_a = validate_dataset(
        datasets['X_A'], datasets['y_A'], 
        BEST_7_FEATURES, "AI4Health"
    )
    
    print("验证数据集B...")
    validation_b = validate_dataset(
        datasets['X_B'], datasets['y_B'], 
        BEST_7_FEATURES, "HenanCancer"
    )
    
    # 3. 检查数据集兼容性
    print("检查数据集兼容性...")
    compatibility = check_dataset_compatibility(
        datasets['X_A'], datasets['X_B'], BEST_7_FEATURES
    )
    
    # 4. 报告结果
    if validation_a['is_valid'] and validation_b['is_valid'] and compatibility['compatible']:
        print("✅ 所有数据集验证通过")
        return datasets
    else:
        print("❌ 数据集验证失败")
        return None

# 使用
datasets = load_and_validate_data()
```

### 处理大数据集

```python
def load_large_dataset_in_chunks(
    file_path: str,
    features: List[str],
    label_col: str,
    chunk_size: int = 10000
):
    """分块加载大数据集"""
    
    chunks_X = []
    chunks_y = []
    
    # 使用pandas分块读取
    for chunk in pd.read_excel(file_path, chunksize=chunk_size):
        X_chunk = chunk[features].values
        y_chunk = chunk[label_col].values
        
        chunks_X.append(X_chunk)
        chunks_y.append(y_chunk)
    
    # 合并所有块
    X = np.vstack(chunks_X)
    y = np.concatenate(chunks_y)
    
    return X, y
```

### 数据质量检查

```python
def comprehensive_data_quality_check(X, y, features):
    """全面的数据质量检查"""
    
    quality_report = {
        'missing_values': {},
        'outliers': {},
        'duplicates': 0,
        'class_imbalance': {},
        'feature_variance': {}
    }
    
    # 检查缺失值
    for i, feature in enumerate(features):
        missing_count = np.isnan(X[:, i]).sum()
        quality_report['missing_values'][feature] = missing_count
    
    # 检查异常值（使用IQR方法）
    for i, feature in enumerate(features):
        Q1 = np.percentile(X[:, i], 25)
        Q3 = np.percentile(X[:, i], 75)
        IQR = Q3 - Q1
        outlier_count = np.sum((X[:, i] < Q1 - 1.5*IQR) | (X[:, i] > Q3 + 1.5*IQR))
        quality_report['outliers'][feature] = outlier_count
    
    # 检查重复样本
    unique_rows = np.unique(X, axis=0)
    quality_report['duplicates'] = X.shape[0] - unique_rows.shape[0]
    
    # 检查类别不平衡
    unique, counts = np.unique(y, return_counts=True)
    quality_report['class_imbalance'] = dict(zip(unique, counts))
    
    # 检查特征方差
    for i, feature in enumerate(features):
        quality_report['feature_variance'][feature] = np.var(X[:, i])
    
    return quality_report
```

## 性能优化建议

### 1. 内存优化
```python
# 使用适当的数据类型
def optimize_dtypes(df):
    """优化DataFrame的数据类型以节省内存"""
    for col in df.columns:
        if df[col].dtype == 'int64':
            if df[col].min() >= 0 and df[col].max() <= 255:
                df[col] = df[col].astype('uint8')
            elif df[col].min() >= -128 and df[col].max() <= 127:
                df[col] = df[col].astype('int8')
        elif df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df
```

### 2. 并行加载
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_load_datasets(file_paths, features, label_col):
    """并行加载多个数据集"""
    
    def load_single(path):
        return load_excel(path, features, label_col)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(load_single, file_paths))
    
    return results
```

### 3. 缓存策略
```python
import hashlib
import pickle

def get_cache_key(file_path, features, label_col):
    """生成缓存键"""
    content = f"{file_path}_{features}_{label_col}"
    return hashlib.md5(content.encode()).hexdigest()

def save_to_cache(data, cache_key, cache_dir):
    """保存到缓存"""
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)

def load_from_cache(cache_key, cache_dir):
    """从缓存加载"""
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None
```

## 故障排除

### 常见问题和解决方案

1. **Excel文件读取失败**
   ```python
   # 问题: 文件格式不支持或文件损坏
   # 解决: 检查文件格式和完整性
   try:
       df = pd.read_excel(file_path)
   except pd.errors.ExcelFileError as e:
       print(f"Excel文件错误: {e}")
       # 尝试其他格式
       df = pd.read_csv(file_path.replace('.xlsx', '.csv'))
   ```

2. **特征列不存在**
   ```python
   # 问题: 指定的特征列在文件中不存在
   # 解决: 检查列名并提供建议
   available_columns = df.columns.tolist()
   missing_features = [f for f in features if f not in available_columns]
   if missing_features:
       print(f"缺失特征: {missing_features}")
       print(f"可用列: {available_columns}")
   ```

3. **内存不足**
   ```python
   # 问题: 数据集太大导致内存不足
   # 解决: 使用分块加载或数据类型优化
   try:
       df = pd.read_excel(file_path)
   except MemoryError:
       print("内存不足，尝试分块加载...")
       df = load_large_dataset_in_chunks(file_path, features, label_col)
   ```

4. **数据类型不匹配**
   ```python
   # 问题: 数据类型与预期不符
   # 解决: 自动类型转换
   def safe_type_conversion(df, features):
       for feature in features:
           try:
               df[feature] = pd.to_numeric(df[feature], errors='coerce')
           except:
               print(f"警告: 特征 {feature} 无法转换为数值类型")
       return df
   ```

## 总结

`data/loader.py` 模块提供了完整的数据加载和验证功能，主要特点包括：

- **灵活性**: 支持多种数据格式和加载选项
- **可靠性**: 完善的数据验证和错误处理机制
- **性能**: 缓存机制和并行加载支持
- **易用性**: 简洁的API和丰富的使用示例

通过合理使用这些功能，可以确保数据加载过程的稳定性和效率，为后续的模型训练和分析提供高质量的数据基础。 