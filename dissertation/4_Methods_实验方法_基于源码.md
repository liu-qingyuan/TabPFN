# 4. Methods（方法）

## 4.1 数据集与样本

### 4.1.1 数据集描述

本研究基于真实的医疗数据集，使用项目中 `MedicalDataLoader`模块实现数据加载：

**数据集A（源域）**: AI4healthcare.xlsx

- 用途：源域训练数据
- 样本数量：n = 295（基于实际代码分析）
- 变量名：`AI4health`数据集

**数据集B（目标域）**: HenanCancerHospital_features63_58.xlsx

- 用途：目标域测试数据
- 样本数量：n = 190（基于实际代码分析）
- 变量名：`HenanCancerHospital`数据集

**数据集C（验证域）**: GuangzhouMedicalHospital_features23_no_nan_new_fixed.xlsx

- 用途：额外验证数据

### 4.1.2 特征工程

基于源码 `MedicalDataLoader`类中定义的特征集：

**原始特征数**：63个特征（Feature1-Feature63）

**RFE筛选特征集**：

- **ALL_63_FEATURES**: 全部63个特征
- **SELECTED_58_FEATURES**: 数据集B的58个特征
- **BEST_7_FEATURES**: ['Feature63', 'Feature2', 'Feature46', 'Feature56', 'Feature42', 'Feature39', 'Feature43']
- **BEST_8_FEATURES**: ['Feature63', 'Feature2', 'Feature46', 'Feature61', 'Feature56', 'Feature42', 'Feature39', 'Feature43']
- **BEST_9_FEATURES**: 添加'Feature48'
- **BEST_10_FEATURES**: 添加'Feature5'

**类别特征**：20个类别特征已预定义

```python
CAT_FEATURE_NAMES = [
    'Feature1', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 
    'Feature8', 'Feature9', 'Feature10', 'Feature11', 'Feature45', 'Feature46', 
    'Feature49', 'Feature50', 'Feature51', 'Feature52', 'Feature53', 'Feature54', 
    'Feature55', 'Feature63'
]
```

### 4.1.3 标签定义

- **目标变量**: Label列（二分类：0/1）
- **类别含义**: 异常/正常分类任务

### 4.1.4 伦理声明

所有医疗数据已完全匿名化处理，特征名称使用Feature1-Feature63的通用标识符，无法追溯到具体患者信息。

## 4.2 TabPFN方法

### 4.2.1 数据加载流程

基于 `MedicalDataLoader`类的实际实现：

```python
def load_dataset(self, dataset_id: str, feature_type: str = 'best8'):
    # 1. 加载Excel数据文件
    data = pd.read_excel(data_path)
  
    # 2. 选择预定义特征集
    selected_features = self._get_features_by_type(feature_type)
  
    # 3. 提取特征和标签
    X = data[selected_features].values
    y = data[self.target_column].values  # 'Label'列
  
    # 4. 处理缺失值（均值填充）
    if np.isnan(X).any():
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
  
    # 5. 确保二分类标签
    if len(np.unique(y)) != 2:
        y = (y > np.median(y)).astype(int)
```

### 4.2.2 TabPFN预处理机制详细实现

**TabPFN内置预处理流水线**（基于源码分析）：

#### 阶段1：数据类型修复和类别特征识别

```python
# src/tabpfn/classifier.py:462-482
# 1. 数据类型修复：确保类别特征为正确的数据类型
X = _fix_dtypes(X, cat_indices=self.categorical_features_indices)

# 2. 类别特征处理：序数编码+NaN处理
ord_encoder = _get_ordinal_encoder()
X = _process_text_na_dataframe(X, ord_encoder=ord_encoder, fit_encoder=True)

# 3. 智能类别特征推断
self.inferred_categorical_indices_ = infer_categorical_features(
    X=X, provided=self.categorical_features_indices,
    min_samples_for_inference=MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
    max_unique_for_category=MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
    min_unique_for_numerical=MIN_UNIQUE_FOR_NUMERICAL_FEATURES
)
```

#### 阶段2：集成预处理配置生成

**每个集成成员都有独立的预处理流水线**（基于源码 `src/tabpfn/model/preprocessing.py`）：

#### 4.2.2.1 完整预处理流程图

```mermaid
graph TD
    A[原始数据 X, y] --> B[阶段1 基础预处理流程]

    B --> B1[数据类型修复 fix_dtypes]
    B1 --> B2[类别特征编码 process_text_na_dataframe]
    B2 --> B3[智能类别特征推断 infer_categorical_features]
    B3 --> C[阶段2 集成配置生成]

    C --> C1[设定 n_estimators 为 32]
    C1 --> C2[生成 32 个 EnsembleConfig 实例]
    C2 --> C3[为每个成员分配预处理配置]

    C3 --> D1[配置1 quantile_uni_coarse 成员数 16]
    C3 --> D2[配置2 none 成员数 16]

    D1 --> E1[阶段3 配置1 预处理流水线]
    D2 --> E2[阶段3 配置2 预处理流水线]

    %% 配置1流程
    E1 --> P1_1[步骤1 移除常数特征 RemoveConstantFeaturesStep]
    P1_1 --> P1_2[步骤2 特征顺序重排 ShuffleFeaturesStep rotate方式]
    P1_2 --> P1_3[步骤3 分布变换 ReshapeFeatureDistributionsStep 使用分位数变换和SVD]
    P1_3 --> P1_4[步骤4 类别特征编码 EncodeCategoricalFeaturesStep 使用ordinal编码]
    P1_4 --> P1_5[步骤5 添加指纹特征 AddFingerprintFeaturesStep 使用哈希增强]
    P1_5 --> P1_6[步骤6 多项式特征生成 NanHandlingPolynomialFeaturesStep --默认不进行]

    %% 配置2流程
    E2 --> P2_1[步骤1 移除常数特征 RemoveConstantFeaturesStep]
    P2_1 --> P2_2[步骤2 特征顺序重排 ShuffleFeaturesStep rotate方式]
    P2_2 --> P2_3[步骤3 分布变换 ReshapeFeatureDistributionsStep transform为none]
    P2_3 --> P2_4[步骤4 类别特征处理 EncodeCategoricalFeaturesStep --numeric视为数值]
    P2_4 --> P2_5[步骤5 添加指纹特征 AddFingerprintFeaturesStep 使用哈希增强]
    P2_5 --> P2_6[步骤6 多项式特征生成 NanHandlingPolynomialFeaturesStep --默认不进行]

    %% 阶段4：并行推理
    P1_6 --> F1[阶段4 配置1 并行推理]
    P2_6 --> F2[阶段4 配置2 并行推理]

    F1 --> F1_1[配置1 成员1 推理]
    F1 --> F1_2[配置1 成员2 推理]
    F1 --> F1_3[配置1 成员3 到 15 推理]
    F1 --> F1_4[配置1 成员16 推理]

    F2 --> F2_1[配置2 成员1 推理]
    F2 --> F2_2[配置2 成员2 推理]
    F2 --> F2_3[配置2 成员3 到 15 推理]
    F2 --> F2_4[配置2 成员16 推理]

    %% 阶段5：集成输出聚合
    F1_1 --> G[阶段5 集成聚合]
    F1_2 --> G
    F1_3 --> G
    F1_4 --> G

    F2_1 --> G
    F2_2 --> G
    F2_3 --> G
    F2_4 --> G

    G --> G1[Softmax 激活 + 概率平均]
    G1 --> G2[类别分布平衡调整 balance_probabilities=True 逆频率加权]
    G2 --> H[最终分类预测输出]

```

**预处理流水线详细组成**：

```python
class SequentialFeatureTransformer:
    steps = [
        RemoveConstantFeaturesStep(),           # 移除常数特征
        ShuffleFeaturesStep(),                  # 特征重排（集成多样性）
        ReshapeFeatureDistributionsStep(),      # 特征分布变换
        EncodeCategoricalFeaturesStep(),        # 类别特征编码
        AddFingerprintFeaturesStep(),           # 添加指纹特征
        NanHandlingPolynomialFeaturesStep()     # 多项式特征生成
    ]
```

#### 4.2.2.2 六步预处理流水线详细分析

##### 4.2.2.2.1 Step 1: RemoveConstantFeaturesStep - 移除常数特征

**目的**：移除所有样本中取值完全相同的特征，避免无信息特征影响模型性能。

**默认参数**：无参数

**实现原理**（基于源码 `src/tabpfn/model/preprocessing.py:452-461`）：

```python
def _fit(self, X, categorical_features):
    # 检测常数特征：第一行与所有行比较，平均相等率=1.0表示常数特征
    sel_ = ((X[0:1, :] == X).mean(axis=0) < 1.0).tolist()
  
    if not any(sel_):
        raise ValueError("所有特征都是常数特征，无法预测！")
  
    # 更新类别特征索引：移除常数特征后重新编号
    return [new_idx for new_idx, idx in enumerate(np.where(sel_)[0]) 
            if idx in categorical_features]
```

**作用机制**：

- 自动识别方差为0的特征
- 保护机制：如果所有特征都是常数则报错
- 智能索引更新：移除特征后自动调整类别特征索引

##### 4.2.2.2.2 Step 2: ShuffleFeaturesStep - 特征重排（集成多样性）

**目的**：通过改变特征顺序增加集成成员的多样性，补偿Transformer对位置敏感性。

**默认参数**（基于源码 `src/tabpfn/model/preprocessing.py:529-538`）：

- `shuffle_method="rotate"`：旋转方法（默认）
- `shuffle_index=0`：旋转偏移量（由集成配置自动分配）
- `random_state=None`：随机种子

**实现原理**：

```python
def _fit(self, X, categorical_features):
    if self.shuffle_method == "rotate":
        # 旋转模式：特征索引环形移位
        index_permutation = np.roll(
            np.arange(X.shape[1]), 
            self.shuffle_index  # 每个集成成员不同的偏移量
        ).tolist()
    elif self.shuffle_method == "shuffle":
        # 随机模式：完全随机打乱特征顺序
        index_permutation = rng.permutation(X.shape[1]).tolist()
  
    return [new_idx for new_idx, idx in enumerate(index_permutation) 
            if idx in categorical_features]
```

**集成多样性策略**：

- **rotate模式**：32个成员使用不同的 `feature_shift_count`值
- 成员1：特征顺序 [F0,F1,F2,F3] → [F0,F1,F2,F3]
- 成员2：特征顺序 [F0,F1,F2,F3] → [F3,F0,F1,F2]
- 成员3：特征顺序 [F0,F1,F2,F3] → [F2,F3,F0,F1]

##### 4.2.2.2.3 Step 3: ReshapeFeatureDistributionsStep - 特征分布变换

**目的**：根据数据分布特性自适应地选择最优的数值变换方法。

**默认参数**（基于源码 `src/tabpfn/preprocessing.py:188-199`）：

**配置1 (16个成员)**: `quantile_uni_coarse`

- `transform_name="quantile_uni_coarse"`：粗粒度分位数变换
- `append_original=True`：保留原始特征
- `apply_to_categorical=False`：不应用于类别特征
- `subsample_features=-1`：不进行特征子采样
- `global_transformer_name="svd"`：SVD全局降维

**配置2 (16个成员)**: `none`

- `transform_name="none"`：不进行数值变换
- `append_original=False`：不保留原始特征
- `apply_to_categorical=False`：不应用于类别特征

**智能分布检测与变换选择**：

```python
# 配置1: quantile_uni_coarse 变换
QuantileTransformer(
    output_distribution="uniform",     # 输出均匀分布
    n_quantiles=max(n_samples // 10, 2), # 分位数数量：样本数/10
    random_state=random_state
)

# 配置2: none 变换
FunctionTransformer(func=_identity)  # 恒等变换，保持原值
```

##### 4.2.2.2.4 Step 3 深度解析：关键参数详细实现

###### 4.2.2.2.4.1 分位数变换（QuantileTransformer）的数学原理

**分位数变换**是一种强大的非参数统计变换，将任意分布的数据映射到指定的目标分布。

**实现机制**（基于源码 `sklearn.preprocessing.QuantileTransformer`）：

```python
# 配置1使用的分位数变换器
QuantileTransformer(
    output_distribution="uniform",  # 输出均匀分布 U(0,1)
    n_quantiles=max(n_samples // 10, 2),  # 分位数数量
    random_state=random_state,
    subsample=1e6  # 大数据集子采样限制
)
```

####### 4.2.2.2.4.1.1 分位数计算过程

**Step 1: 分位数点计算**

```python
# 医疗数据示例：295个样本
n_samples = 295
n_quantiles = max(295 // 10, 2) = max(29, 2) = 29

# 29个分位数点：0%, 3.57%, 7.14%, ..., 96.43%, 100%
quantile_points = np.linspace(0, 100, n_quantiles)
# [0.0, 3.57, 7.14, 10.71, ..., 96.43, 100.0]
```

**Step 2: 每个特征的分位数统计**

```python
def fit_transform_quantile(X_feature):
    # 1. 计算每个分位数点对应的实际数值
    # 例如Feature63（年龄特征）：
    # 0%分位数（最小值）: 23岁
    # 50%分位数（中位数）: 45岁  
    # 100%分位数（最大值）: 78岁
    quantiles_values = np.percentile(X_feature, quantile_points)
  
    # 2. 建立映射表：原始值 → 均匀分布值
    # 23岁 → 0.0, 45岁 → 0.5, 78岁 → 1.0, 等等
    return quantiles_values
```

**Step 3: 均匀分布映射**

```python
def transform(X_feature):
    # 对每个样本值进行插值映射
    for value in X_feature:
        # 1. 找到value在分位数表中的位置
        # 2. 线性插值计算对应的[0,1]均匀分布值
  
        # 示例：如果患者年龄=52岁
        # 52介于45岁(50%分位数)和60岁(75%分位数)之间
        # 插值得到：0.5 + (52-45)/(60-45) * (0.75-0.5) = 0.617
        uniform_value = interpolate(value, quantiles_values, [0, 1])
  
    return uniform_values  # 所有值都在[0,1]区间内
```

####### 4.2.2.2.4.1.2 分位数变换的核心优势

**1. 分布标准化**

```python
# 变换前：任意分布（偏态、异常值、多峰等）
X_original = [23, 25, 28, 33, 45, 52, 65, 78, 82, 95]  # 年龄分布

# 变换后：严格的[0,1]均匀分布
X_uniform = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 0.95, 1.0]
```

**2. 异常值鲁棒性**

```python
# 极端异常值的处理
X_with_outlier = [23, 25, 28, 33, 45, 52, 65, 78, 82, 999]  # 999岁异常值

# 分位数变换自动压缩到[0,1]
X_robust = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 0.95, 1.0]
# 异常值999被映射到1.0，不会破坏整体分布
```

**3. 非线性关系保持**

分位数变换是**单调变换**，保持数据的相对排序关系：

```python
# 原始排序：A < B < C
# 变换后排序：transform(A) < transform(B) < transform(C)
# 保持所有的大小关系和排序信息
```

####### 4.2.2.2.4.1.3 医疗数据中的实际应用

**特征分布差异处理**：

```python
# 医疗特征的典型分布问题：
Feature_Age = [18, 25, 35, 45, 55, 65, 75, 85]     # 正态分布
Feature_BMI = [18.5, 22.1, 25.0, 30.5, 35.2, 40.8] # 右偏分布  
Feature_Income = [2w, 5w, 8w, 12w, 50w, 100w]      # 极度右偏+异常值

# 分位数变换后，所有特征都变成[0,1]均匀分布
# 消除了不同特征间的量纲和分布差异
```

####### 4.2.2.2.4.1.4 TabPFN Transformer的受益机制

**1. 注意力权重稳定**

```python
# Transformer注意力计算：Attention = softmax(QK^T / √d_k)
# 
# 变换前：特征值差异巨大
# Feature1: [0.1, 0.2, 0.3]  (BMI)
# Feature2: [50000, 80000, 120000]  (收入) 
# → 注意力权重被大数值特征主导
#
# 变换后：所有特征在[0,1]均匀分布
# Feature1: [0.2, 0.4, 0.6]
# Feature2: [0.3, 0.7, 0.9]
# → 注意力权重更加平衡和稳定
```

**2. 梯度流稳定**

```python
# 均匀分布的数值范围稳定，避免梯度爆炸/消失
# 所有特征的激活值都在合理范围内[0,1]
# 提升了预训练Transformer的泛化能力
```

####### 4.2.2.2.4.1.5 与其他变换方法的对比

```python
# StandardScaler: X' = (X - μ) / σ
# 优势：简单快速
# 缺点：假设正态分布，对异常值敏感

# RobustScaler: X' = (X - median) / IQR  
# 优势：对异常值鲁棒
# 缺点：仍然保留原始分布形状

# QuantileTransformer: X' = F^(-1)(F(X))
# 优势：任意分布→目标分布，极强的鲁棒性
# 缺点：计算复杂度较高，需要足够样本数
```

####### 4.2.2.2.4.1.6 特征维度对比总结

| 配置  | 原始特征 | 变换特征  | SVD特征 | 最终维度       | 计算复杂度 |
| ----- | -------- | --------- | ------- | -------------- | ---------- |
| 配置1 | 8维保留  | 8维分位数 | 4维SVD  | **20维** | 高         |
| 配置2 | 8维不变  | 无变换    | 无SVD   | **8维**  | 低         |

**分位数变换的实际收益**：

- **数值稳定性**：所有特征标准化到[0,1]区间，避免数值溢出
- **分布一致性**：消除不同医疗指标的分布差异（年龄vs收入vs BMI）
- **异常值鲁棒性**：自动处理医疗数据中的测量误差和极端值
- **Transformer兼容性**：均匀分布最适合预训练Transformer的注意力机制

**集成效果**：

- **16个配置1成员**：每个处理20维特征，提供丰富的特征表示
- **16个配置2成员**：每个处理8维特征，保持简洁高效
- **互补性**：复杂变换 + 简单保持 = 平衡的集成策略

##### 4.2.2.2.5 SVD全局降维 (`global_transformer_name="svd"`)

**目的**：对所有特征进行全局降维，减少特征维度并提取主要信息。

**实现原理**（基于源码 `src/tabpfn/model/preprocessing.py:790-822`）：

```python
def get_all_global_transformers(self, num_examples, num_features, random_state):
    return {
        "svd": FeatureUnion([
            # 保留原始特征的通道
            ("passthrough", FunctionTransformer(func=_identity)),
  
            # SVD降维通道
            ("svd", Pipeline([
                # Step 1: 安全标准化（不减均值，保持稀疏性）
                ("save_standard", make_standard_scaler_safe(
                    ("standard", StandardScaler(with_mean=False))
                )),
      
                # Step 2: 截断SVD降维
                ("svd", TruncatedSVD(
                    algorithm="arpack",
                    n_components=max(1, min(
                        num_examples // 10 + 1,    # 样本数/10+1
                        num_features // 2,         # 特征数/2
                    )),
                    random_state=random_state,
                ))
            ]))
        ])
    }
```

**SVD组件数计算**：

- **公式**：`n_components = max(1, min(n_samples//10+1, n_features//2))`
- **医疗数据示例**（n_samples=295, n_features=8）：
  - `n_samples//10+1 = 295//10+1 = 30`
  - `n_features//2 = 8//2 = 4`
  - `n_components = max(1, min(30, 4)) = 4`

**SVD降维效果**：

- **输入特征**：8维原始特征
- **SVD降维**：4维主成分
- **最终特征**：8维原始 + 4维SVD = **12维总特征**

##### 4.2.2.2.6 保留原始特征 (`append_original=True`)

**目的**：在应用变换的同时保留原始特征，提供多层次的特征表示。

**实现机制**（基于源码 `src/tabpfn/model/preprocessing.py:894-906`）：

```python
def _set_transformer_and_cat_ix(self, n_samples, n_features, categorical_features):
    # -------- Append to original 策略 ------
    if self.append_to_original and not self.apply_to_categorical:
        trans_ixs = numerical_ix  # 仅对数值特征应用变换
  
        # 关键：添加passthrough变换器保留ALL原始特征
        transformers.append(("original", "passthrough", all_feats_ix))
        cat_ix = categorical_features  # 类别特征索引保持不变
  
    # 构建ColumnTransformer
    transformers.append(("feat_transform", _transformer, trans_ixs))
  
    transformer = ColumnTransformer(
        transformers,
        remainder="drop",      # 丢弃未处理的特征
        sparse_threshold=0.0   # 强制密集矩阵输出
    )
```

**特征组合策略**：

```python
# 配置1 (append_original=True) 的特征构成：
# 1. "original" passthrough: 保留8个原始特征 [F1, F2, ..., F8]
# 2. "feat_transform": 数值特征quantile变换 [Q1, Q2, ..., Q8] 
# 3. "global_transformer" SVD: 4维主成分 [SVD1, SVD2, SVD3, SVD4]
# 最终特征矩阵: [F1,F2,...,F8, Q1,Q2,...,Q8, SVD1,SVD2,SVD3,SVD4] = 20维

# 配置2 (append_original=False) 的特征构成：
# 1. "feat_transform": none变换(恒等) [F1, F2, ..., F8]
# 最终特征矩阵: [F1, F2, ..., F8] = 8维
```

##### 4.2.2.2.7 特征子采样 (`subsample_features`)

**目的**：随机选择特征子集进行处理，增加集成多样性并降低计算复杂度。

**默认设置**：`subsample_features=-1`（不进行特征子采样）

**实现原理**（基于源码 `src/tabpfn/model/preprocessing.py:871-887`）：

```python
def _set_transformer_and_cat_ix(self, n_samples, n_features, categorical_features):
    if self.subsample_features > 0:
        # 1. 计算子采样特征数量
        subsample_features = int(self.subsample_features * n_features) + 1
  
        # 2. 处理超采样情况（允许重复采样）
        replace = subsample_features > n_features
  
        # 3. 随机选择特征索引
        self.subsampled_features_ = rng.choice(
            list(range(n_features)),    # 从所有特征中选择
            subsample_features,         # 选择的特征数量
            replace=replace,           # 是否允许重复
        )
  
        # 4. 更新类别特征索引（重新映射到子采样后的索引）
        categorical_features = [
            new_idx for new_idx, idx in enumerate(self.subsampled_features_)
            if idx in categorical_features
        ]
        n_features = subsample_features
    else:
        # 默认使用全部特征
        self.subsampled_features_ = np.arange(n_features)

# 在实际变换时应用特征子采样
def fit_transform(self, X, categorical_features):
    transformer.fit_transform(X[:, self.subsampled_features_])  # 仅使用选定特征
```

**特征子采样示例**：

```python
# 假设 subsample_features=0.75, n_features=8
subsample_count = int(0.75 * 8) + 1 = 7
# 随机选择7个特征，例如: [0, 2, 3, 4, 5, 6, 7]
# 类别特征重新映射: 原索引[1,3,5] → 新索引[1,2,4]（如果这些索引被选中）
```

##### 4.2.2.2.8 配置对比：详细特征变换流程

**配置1 (quantile_uni_coarse + SVD + append_original)**：

**Step-by-step变换流程**：

```python
# 输入：X shape=(295, 8)，8个医疗特征
X_input = np.array([[feat1, feat2, ..., feat8], ...])  # shape: (295, 8)

# Step 1: 保留原始特征
X_original = X_input.copy()  # shape: (295, 8)

# Step 2: 分位数变换（仅数值特征）
quantile_transformer = QuantileTransformer(
    output_distribution="uniform", 
    n_quantiles=max(295//10, 2) = 29
)
X_quantile = quantile_transformer.fit_transform(X_numerical)  # shape: (295, 8)

# Step 3: 特征拼接（原始 + 变换）
X_combined = np.concatenate([X_original, X_quantile], axis=1)  # shape: (295, 16)

# Step 4: SVD全局降维
svd = TruncatedSVD(n_components=4)
X_svd = svd.fit_transform(StandardScaler(with_mean=False).fit_transform(X_combined))  # shape: (295, 4)

# Step 5: 最终特征拼接
X_final = np.concatenate([X_combined, X_svd], axis=1)  # shape: (295, 20)
# 最终：8原始 + 8分位数变换 + 4SVD主成分 = 20维特征
```

**配置2 (none + numeric)**：

**Step-by-step变换流程**：

```python
# 输入：X shape=(295, 8)
X_input = np.array([[feat1, feat2, ..., feat8], ...])  # shape: (295, 8)

# Step 1: 恒等变换（不改变数值）
identity_transformer = FunctionTransformer(func=lambda x: x)
X_transformed = identity_transformer.fit_transform(X_input)  # shape: (295, 8)

# Step 2: 类别特征数值化处理（直接当作数值）
# categorical_transform_name="numeric" → 跳过类别编码

# 最终特征
X_final = X_transformed  # shape: (295, 8)
# 最终：8个原始特征（未变换）= 8维特征
```

##### 4.2.2.2.9 Step 4: EncodeCategoricalFeaturesStep - 类别特征编码

**目的**：将类别特征转换为数值形式，支持多种编码策略。

**默认参数**（基于源码 `src/tabpfn/preprocessing.py`）：

**配置1**: `ordinal_very_common_categories_shuffled`

- `categorical_transform_name="ordinal_very_common_categories_shuffled"`
- 仅编码非常常见的类别（最少类别计数≥10且类别数<样本数/10）
- 支持类别排列混洗增强集成多样性

**配置2**: `numeric`

- `categorical_transform_name="numeric"`
- 将类别特征直接当作数值特征处理

**实现机制**：

```python
def _get_transformer(self, X, categorical_features):
    if self.categorical_transform_name.startswith("ordinal"):
        # 序数编码：类别 → 0,1,2,3...
        ct = ColumnTransformer([
            ("ordinal_encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=np.nan  # 未知类别用NaN表示
            ), categorical_features)
        ])
  
        # 智能类别过滤
        if name.startswith("_very_common_categories"):
            categorical_features = [
                i for i, col in enumerate(X.T)
                if i in categorical_features
                and self.get_least_common_category_count(col) >= 10
                and len(np.unique(col)) < (len(X) // 10)
            ]
  
        # 类别排列混洗（集成多样性）
        if name.endswith("_shuffled"):
            for col_ix in categorical_features:
                col_cats = len(encoder.categories_[col_ix])
                perm = rng.permutation(col_cats)  # 随机排列类别编码
                X_col[not_nan_mask] = perm[X_col[not_nan_mask].astype(int)]
    
    elif self.categorical_transform_name == "numeric":
        return None, categorical_features  # 不进行编码转换
```

##### 4.2.2.2.10 Step 5: AddFingerprintFeaturesStep - 添加指纹特征

**目的**：为每个样本生成唯一的哈希指纹，帮助模型区分重复样本，提升稳定性。

**默认参数**：

- `random_state=None`：随机种子（由集成配置提供）
- `_CONSTANT=10**12`：哈希常数

**实现原理**（基于源码 `src/tabpfn/model/preprocessing.py:477-523`）：

```python
def float_hash_arr(arr):
    # SHA256哈希 + 模运算生成0-1范围的浮点哈希值
    _hash = int(hashlib.sha256(arr.tobytes()).hexdigest(), 16)
    return _hash % _CONSTANT / _CONSTANT

def _transform(self, X, is_test=False):
    X_h = np.zeros(X.shape[0])
    salted_X = X + self.rnd_salt_  # 添加随机盐值
  
    if is_test:
        # 测试时：允许哈希碰撞，保持一致性
        for i, row in enumerate(salted_X):
            h = float_hash_arr(row + self.rnd_salt_)
            X_h[i] = h
    else:
        # 训练时：处理哈希碰撞，确保唯一性
        seen_hashes = set()
        for i, row in enumerate(salted_X):
            h = float_hash_arr(row)
            add_to_hash = 0
            while h in seen_hashes:  # 碰撞检测
                add_to_hash += 1
                h = float_hash_arr(row + add_to_hash)
            seen_hashes.add(h)
            X_h[i] = h
  
    # 将哈希特征作为新列添加到原特征矩阵
    return np.concatenate([X, X_h.reshape(-1, 1)], axis=1)
```

**指纹特征优势**：

- **去重能力**：自动识别完全相同的样本
- **注意力增强**：帮助Transformer注意力机制区分相似样本
- **稳定性提升**：减少重复样本带来的预测不确定性

##### 4.2.2.2.11 Step 6: NanHandlingPolynomialFeaturesStep - 多项式特征生成

**目的**：生成二阶交互特征（特征乘积），捕捉特征间的非线性关系。

**默认参数**：

- `max_features=None`：多项式特征数量限制（默认：`POLYNOMIAL_FEATURES="no"`）
- `random_state=None`：随机种子

**实现逻辑**（基于源码 `src/tabpfn/model/preprocessing.py:1220-1278`）：

```python
def _fit(self, X, categorical_features):
    # 计算可生成的多项式特征数量
    n_polynomials = (X.shape[1] * (X.shape[1] - 1)) // 2 + X.shape[1]
    n_polynomials = min(self.max_poly_features, n_polynomials) \
                    if self.max_poly_features else n_polynomials
  
    # 标准化输入特征（不减均值，避免破坏稀疏性）
    X = self.standardizer.fit_transform(X)
  
    # 随机选择特征对进行交互
    self.poly_factor_1_idx = rng.choice(np.arange(0, X.shape[1]), 
                                       size=n_polynomials, replace=True)
  
    # 避免重复的特征对组合
    for i in range(len(self.poly_factor_1_idx)):
        poly_factor_1_ = self.poly_factor_1_idx[i]
        used_indices = self.poly_factor_2_idx[self.poly_factor_1_idx == poly_factor_1_]
        indices_ = set(range(poly_factor_1_, X.shape[1])) - set(used_indices.tolist())
        if len(indices_) > 0:
            self.poly_factor_2_idx[i] = rng.choice(list(indices_), size=1)

def _transform(self, X, is_test=False):
    X = self.standardizer.transform(X)
    # 生成多项式特征：X[:, i] * X[:, j]
    poly_features_xs = X[:, self.poly_factor_1_idx] * X[:, self.poly_factor_2_idx]
    return np.hstack((X, poly_features_xs))  # 拼接原特征和多项式特征
```

**注意**：在默认配置中，`POLYNOMIAL_FEATURES="no"`，因此该步骤通常不生成多项式特征。

#### 4.2.2.3 n_estimators=32 的完整处理流程

当您设置 `n_estimators=32`时，TabPFN执行以下完整流程：

##### 4.2.2.3.1. 集成配置分配

```python
# 默认2个预处理配置
preprocessor_configs = default_classifier_preprocessor_configs()  # 2个配置
balance_count = 32 // 2  # 16

# 均衡分配给32个集成成员
configs_ = [config1] * 16 + [config2] * 16
# 配置1: quantile_uni_coarse + SVD (16个成员)
# 配置2: none + numeric (16个成员)
```

##### 4.2.2.3.2. 多样性参数生成

```python
# 特征偏移：32个不同的偏移值
featshifts = np.arange(start, start + 32)
# 例如：[15, 16, 17, ..., 46]

# 类别排列：32种不同的类别顺序混洗
class_permutations = generate_32_different_permutations()

# 样本子采样：32个不同的样本子集（如果启用）
subsamples = generate_32_subsamples() if subsample_size else [None] * 32
```

##### 4.2.2.3.3. 并行预处理执行

**32个集成成员同时执行各自的6步预处理流水线**，每个成员使用：

- 独特的特征偏移值
- 独特的类别排列
- 独特的随机种子
- 预设的预处理配置（16个用配置1，16个用配置2）

##### 4.2.2.3.4. 计算复杂度分析

- **时间复杂度**：O(32 × n_samples × n_features × preprocessing_steps)
- **空间复杂度**：O(32 × n_samples × processed_features)
- **并行加速**：通过 `n_jobs=-1`实现CPU并行预处理

##### 4.2.2.3.5. 预测性能提升

- **方差减少**：32个模型平均后预测方差显著降低
- **偏差校正**：不同预处理策略互相补偿各自的偏差
- **鲁棒性增强**：对数据质量问题和异常值更加鲁棒

## 4.3 UDA方法

### 4.3.1 UDA预处理

基于 `UDAProcessor`和 `UDAVisualizer`的实现：

**标准化选项**：

```python
scaler_options = {
    'standard': StandardScaler(),  # Z-score标准化
    'robust': RobustScaler(),      # 鲁棒标准化  
    'none': None                   # 不标准化（默认）
}
```

**不平衡处理**：

```python
imbalance_methods = {
    'smote': SMOTE(),
    'borderline_smote': BorderlineSMOTE(), 
    'adasyn': ADASYN(),
    'none': None  # 默认不处理
}
```

## 4.4 模型架构

### 4.4.1 TabPFN架构（预训练模型）

**基础架构**：PerFeatureTransformer

```python
class PerFeatureTransformer(nn.Module):
    def __init__(self, ninp=128, nhead=4, nhid=512, nlayers=10):
        # 每个特征独立处理的Transformer
        self.transformer_encoder = LayerStack(
            layer_creator=lambda: PerFeatureEncoderLayer(
                d_model=ninp,      # 128维嵌入
                nhead=nhead,       # 4个注意力头
                dim_feedforward=nhid,  # 512维前馈网络
            ),
            num_layers=nlayers     # 10层Transformer
        )
```

**预训练模型使用**：

```python
class TabPFNClassifier:
    def __init__(self, n_estimators=4, model_path="auto"):
        # 加载预训练模型权重
        self.model_, self.config_, _ = initialize_tabpfn_model(
            model_path=model_path,
            which="classifier"
        )
```

**关键特性详细实现**：

#### 1. 上下文学习（In-Context Learning）机制

**核心原理**：TabPFN将训练数据直接作为Transformer的输入上下文，无需传统的梯度更新训练过程。

**实际实现**（基于源码 `src/tabpfn/inference.py:170-198`）：

```python
def iter_outputs(self, X, device, autocast):
    # 1. 预处理训练数据和测试数据
    X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)
  
    # 2. 拼接为完整上下文：[训练数据, 测试数据]
    X_full = torch.cat([X_train, X_test], dim=0).unsqueeze(1)
  
    # 3. 直接前向传播，无需梯度更新
    with torch.inference_mode():
        output = self.model(
            style, X_full, y_train,
            categorical_inds=cat_ix,
            single_eval_pos=len(y_train)  # 指定训练/测试边界
        )
```

**上下文结构**：

- `X_full`: 维度为 `[n_train + n_test, 1, n_features]`
- `y_train`: 仅包含训练样本标签，测试样本标签为空
- `single_eval_pos`: 标记训练数据结束位置（第len(y_train)个位置）

#### 2. 集成预测（Ensemble Prediction）详细机制

**集成配置生成**（基于源码 `src/tabpfn/preprocessing.py:263-369`）：

```python
def generate_for_classification(n=4, preprocessor_configs, random_state):
    # 1. 特征偏移生成：每个集成成员使用不同的特征排列
    start = rng.integers(0, MAXIMUM_FEATURE_SHIFT)
    featshifts = np.arange(start, start + n)
  
    # 2. 类别排列生成：防止模型过拟合特定类别顺序
    if class_shift_method == "rotate":
        arange = np.arange(0, n_classes)
        shifts = rng.permutation(n_classes).tolist()
        class_permutations = [np.roll(arange, s) for s in shifts]
  
    # 3. 数据子采样：每个成员使用不同的样本子集
    subsamples = generate_index_permutations(
        n=n, max_index=max_index, 
        subsample=subsample_size, random_state=static_seed
    )
  
    # 4. 预处理多样性：分配不同的预处理配置
    balance_count = n // len(preprocessor_configs)
    configs = balance(preprocessor_configs, balance_count)
  
    return [ClassifierEnsembleConfig(...) for each ensemble member]
```

**集成成员差异化策略**：

- **特征层面**：`feature_shift_count` - 特征排列偏移
- **样本层面**：`subsample_ix` - 样本子集选择
- **类别层面**：`class_permutation` - 类别标签排列
- **预处理层面**：`preprocess_config` - 预处理方法配置

**预测聚合过程**（基于源码 `src/tabpfn/classifier.py:577-583`）：

```python
# 收集所有集成成员的输出
outputs = []
for output, config in self.executor_.iter_outputs(X, device, autocast):
    # 应用温度缩放
    if self.softmax_temperature != 1:
        output = output[:, :self.n_classes_].float() / self.softmax_temperature
  
    # 恢复原始类别顺序（逆转class_permutation）
    if config.class_permutation is not None:
        output = output[..., config.class_permutation]
  
    outputs.append(output)

# 集成策略选择
if self.average_before_softmax:
    # 策略1：先平均logits再softmax（更保守）
    output = torch.stack(outputs).mean(dim=0)
    output = torch.nn.functional.softmax(output, dim=1)
else:
    # 策略2：先softmax再平均概率（默认，更鲁棒）
    outputs = [torch.nn.functional.softmax(o, dim=1) for o in outputs]
    output = torch.stack(outputs).mean(dim=0)
```

#### 3. 内置类别平衡（Built-in Class Balancing）

**实现机制**（基于源码 `src/tabpfn/classifier.py:585-588`）：

```python
if self.balance_probabilities:
    # 1. 计算训练集中的类别分布
    class_prob_in_train = self.class_counts_ / self.class_counts_.sum()
  
    # 2. 逆频率重加权：降低多数类概率，提升少数类概率
    output = output / torch.Tensor(class_prob_in_train).to(self.device_)
  
    # 3. 重新归一化确保概率和为1
    output = output / output.sum(dim=-1, keepdim=True)
```

**平衡原理**：

- 通过**逆频率加权**自动调整类别概率
- 多数类概率被除以较大的 `class_prob_in_train`值，降低其影响
- 少数类概率被除以较小的 `class_prob_in_train`值，提升其影响
- 特别适用于医疗数据中的类别不平衡场景

### 4.4.2 UDA方法集成

#### 4.4.2.1 TCA（Transfer Component Analysis）详细实现

**TCA核心原理与数学基础**

TCA是一种基于核方法的域适应技术，旨在通过学习一个公共特征子空间来最小化源域和目标域之间的分布差异。

##### 4.4.2.1.1 数学原理

TCA算法的核心思想是寻找一个变换矩阵，使得源域和目标域在变换后的特征空间中具有相似的分布。

**目标函数**：

```
min tr(W^T * K * L * K^T * W) + mu * tr(W^T * K * H * K^T * W)
W
```

其中：

- `W`: 变换矩阵
- `K`: 核矩阵（源域和目标域的联合核矩阵）
- `L`: 拉普拉斯矩阵，用于衡量域间差异
- `H`: 正则化矩阵
- `mu`: 正则化参数

##### 4.4.2.1.2 核矩阵构建

**联合核矩阵计算**（基于ADAPT库源码 `adapt/feature_based/_tca.py`）：

```python
def _get_kernel(self, X, Y=None, kernel_params=None):
    # 1. 源域核矩阵 Kss
    Kss = pairwise_kernels(self.Xs_, self.Xs_, 
                          metric=self.kernel, 
                          **kernel_params)
  
    # 2. 目标域核矩阵 Ktt  
    Ktt = pairwise_kernels(self.Xt_, self.Xt_,
                          metric=self.kernel,
                          **kernel_params)
  
    # 3. 源-目标域交叉核矩阵 Kst
    Kst = pairwise_kernels(self.Xs_, self.Xt_,
                          metric=self.kernel, 
                          **kernel_params)
  
    # 4. 构建联合核矩阵 K
    K = np.block([[Kss, Kst],
                  [Kst.T, Ktt]])
  
    return K
```

##### 4.4.2.1.3 拉普拉斯矩阵构建

**域差异度量矩阵L**：

```python
def _compute_laplacian_matrix(self, ns, nt):
    # ns: 源域样本数, nt: 目标域样本数
    n = ns + nt
    L = np.zeros((n, n))
  
    # 源域内部相似性: +1/(ns*ns)
    L[:ns, :ns] = 1.0 / (ns * ns)
  
    # 目标域内部相似性: +1/(nt*nt)  
    L[ns:, ns:] = 1.0 / (nt * nt)
  
    # 源-目标域间差异: -1/(ns*nt)
    L[:ns, ns:] = -1.0 / (ns * nt)
    L[ns:, :ns] = -1.0 / (ns * nt)
  
    return L
```

##### 4.4.2.1.4 正则化矩阵构建

**中心化矩阵H**：

```python
def _compute_centering_matrix(self, n):
    # H = I - (1/n) * 1 * 1^T
    # 用于核矩阵中心化，消除均值偏移
    H = np.eye(n) - np.ones((n, n)) / n
    return H
```

##### 4.4.2.1.5 广义特征值分解

**关键优化步骤**（基于源码实现）：

```python
def fit(self, X, y, Xt):
    # 1. 存储源域和目标域数据
    self.Xs_ = X
    self.Xt_ = Xt
    ns, nt = len(X), len(Xt)
  
    # 2. 计算核矩阵
    K = self._get_kernel(X, Xt)
  
    # 3. 构建拉普拉斯矩阵和中心化矩阵
    L = self._compute_laplacian_matrix(ns, nt)
    H = self._compute_centering_matrix(ns + nt)
  
    # 4. 构建目标矩阵
    # A = K * L * K (要最小化的项)
    # B = K * H * K (正则化项)
    A = K.dot(L).dot(K)
    B = K.dot(H).dot(K) + self.mu * np.eye(ns + nt)
  
    # 5. 求解线性系统（ADAPT实际实现）
    a = np.eye(n+m) + self.mu * K.dot(L.dot(K))
    b = K.dot(H.dot(K))
    sol = linalg.lstsq(a, b)[0]  # 最小二乘求解，而非广义特征值分解
  
    # 6. 特征值分解获得变换向量  
    values, vectors = linalg.eigh(sol)
    args = np.argsort(np.abs(values))[::-1][:self.n_components]
    self.vectors_ = np.real(vectors[:, args])
  
    # 7. 返回变换后的源域特征（fit_transform模式）
    X_transformed = K.dot(self.vectors_)[:ns]
    return X_transformed
```

##### 4.4.2.1.6 特征变换实现

**变换过程**：

```python
def _transform_features(self, X):
    # 1. 计算X与训练数据的核矩阵
    if hasattr(self, 'Xs_') and hasattr(self, 'Xt_'):
        # 训练阶段：X来自源域或目标域
        K_new = pairwise_kernels(X, 
                               np.vstack([self.Xs_, self.Xt_]),
                               metric=self.kernel)
  
    # 2. 应用学习到的变换矩阵
    X_transformed = K_new.dot(self.W_)
  
    return X_transformed

def transform(self, X):
    """对新数据进行域适应变换"""
    return self._transform_features(X)

def predict(self, X):
    """预测新样本标签"""
    X_transformed = self.transform(X)
    return self.estimator_.predict(X_transformed)
```

##### 4.4.2.1.7 参数详解与默认值

基于ADAPT库的完整参数配置：

```python
class TCA(BaseAdaptEstimator):
    def __init__(self, 
                 estimator=None,           # 默认：None，自动选择LogisticRegression
                 Xt=None,                  # 目标域数据（必须提供）
                 n_components=20,          # 默认：20个主成分
                 mu=0.1,                   # 默认：0.1正则化强度
                 kernel="rbf",             # 默认：RBF核函数
                 gamma=None,               # RBF核参数，默认：1/n_features
                 degree=3,                 # 多项式核度数，默认：3
                 coef0=1,                  # 多项式/sigmoid核参数，默认：1
                 kernel_params=None,       # 其他核参数
                 random_state=None):       # 随机种子
```

**参数作用详解**：

###### 4.4.2.1.7.1 n_components（主成分数量）

**作用机制**：

- 控制降维后的特征维度
- 较大值：保留更多信息，但计算复杂度增加
- 较小值：降低维度，可能丢失重要信息

**推荐设置**：

```python
# ADAPT库默认值
n_components = 20  # TCA类的默认参数

# 项目实际配置（run_complete_analysis.py）
n_components = None  # 自动选择最佳组件数！
# 自动选择逻辑：根据数据特性确定最优组件数量
```

**实际效果**：

- `n_components=None`：自动选择避免手动调参
- 医疗数据（295+190样本）：ADAPT自动确定最优组件数
- 平衡信息保留与过拟合风险的最佳策略

###### 4.4.2.1.7.2 mu（正则化参数）

**数学作用**：

```
目标函数 = 域差异项 + mu × 正则化项
Domain_Loss + mu × Regularization_Loss
```

**取值范围与效果**：

- `mu = 0.0`：纯域适应，最大化减少域间差异
- `mu = 0.1`（默认）：平衡域适应和数据结构保持
- `mu = 1.0`：更强正则化，保持原始特征结构
- `mu > 1.0`：主要保持数据结构，弱化域适应效果

**项目中的实际参数配置**（基于源码 `run_complete_analysis.py`）：

```python
# TCA参数优化：针对医疗数据的小样本、高维特征
processor.config.method_params.update({
    'n_components': None,     # 自动选择最佳组件数
    'mu': 0.1,               # 较小的mu值，减少正则化，适合小样本
    'kernel': 'linear'       # 线性核，适合医疗特征
})
```

**n_components=None的特殊机制**：

根据ADAPT库源码分析（`/adapt/feature_based/_tca.py`和`_sa.py`），当`n_components=None`时的处理机制存在差异：

**TCA方法**：
- **源码实际**：TCA类的`__init__`方法中`n_components=20`（第60行）
- **文档说明**：官方文档声称默认为`None`，但实际代码中为固定值20
- **当前实现**：项目中设置`n_components=None`时，ADAPT库使用sklearn默认行为

**SA方法**：
- **源码实际**：SA类的`__init__`方法中`n_components=None`（第59行）  
- **自动机制**：当`n_components=None`时，PCA自动使用输入特征维度作为组件数
- **具体逻辑**：第94-95行创建`PCA(self.n_components)`，None传递给sklearn.decomposition.PCA
- **sklearn行为**：`PCA(n_components=None)`等价于`min(n_samples, n_features)`

**实际自动确定逻辑**：
```python
# SA源码第94行：self.pca_src_ = PCA(self.n_components)
# 当n_components=None时：
actual_components = min(n_samples, n_features)
# 医疗数据示例：
# 源域：min(295, 8) = 8个组件
# 目标域：min(190, 8) = 8个组件
```

**项目中的验证方法**：
```python
# run_complete_analysis.py 新增的调试输出
if method_name == 'TCA' and hasattr(adapt_model, 'vectors_'):
    actual_n_components = adapt_model.vectors_.shape[1]
    print(f"TCA实际n_components: {actual_n_components}")
elif method_name == 'SA' and hasattr(adapt_model, 'pca_src_'):
    actual_n_components = adapt_model.pca_src_.n_components_
    print(f"SA实际n_components: {actual_n_components}")
```

###### 4.4.2.1.7.3 kernel（核函数类型）

**支持的核函数**：

**Linear核（线性核）**：

```python
kernel = "linear"
# K(x,y) = x^T * y
# 优势：计算简单，可解释性强
# 适用：线性可分的域适应问题
```

**RBF核（径向基函数核）**：

```python
kernel = "rbf"
gamma = 1.0 / n_features  # 默认gamma值
# K(x,y) = exp(-gamma * ||x-y||^2)  
# 优势：非线性映射能力强，适用性广
# 适用：复杂的非线性域差异
```

**Polynomial核（多项式核）**：

```python
kernel = "poly" 
degree = 3  # 默认3次多项式
coef0 = 1   # 常数项
# K(x,y) = (gamma * x^T * y + coef0)^degree
# 优势：介于线性和RBF之间的复杂度
```

**项目中的选择**：

```python
# 医疗数据使用线性核的原因：
kernel = "linear"
# 1. 医疗特征多为线性相关（年龄、BMI、检验指标等）
# 2. 线性核计算效率高，适合小样本场景
# 3. 结果可解释性强，符合医疗应用需求
```

##### 4.4.2.1.8 医疗数据中的TCA工作流程

**完整实现流程**（基于项目源码）：

```python
class AdaptUDAMethod:
    def _create_adapt_model(self):
        if self.method_name == 'TCA':
            self.adapt_model = TCA(
                estimator=LogisticRegression(penalty=None),  # 无正则化的逻辑回归
                n_components=self.params.get('n_components', 6),  # 6维特征子空间
                mu=self.params.get('mu', 0.1),              # 轻度正则化
                kernel=self.params.get('kernel', 'linear')   # 线性核函数
            )

    def fit_and_predict(self, X_source, y_source, X_target):
        # 1. TCA域适应训练
        self.adapt_model.fit(X_source, y_source, X_target)
  
        # 2. 目标域预测
        y_pred = self.adapt_model.predict(X_target)
        y_pred_proba = self.adapt_model.predict_proba(X_target)[:, 1]
  
        return y_pred, y_pred_proba
```

**域适应过程可视化**（基于ADAPT库实际实现）：

```mermaid
graph TD
    A[源域 AI4health 295样本] --> C[TCA核矩阵计算]
    B[目标域 HenanCancer 190样本] --> C
  
    C --> D[联合核空间 K 485 485]
    D --> E[拉普拉斯矩阵L 域差异度量]
    D --> F[中心化矩阵H 正则化]
  
    E --> G[线性系统求解 lstsq]
    F --> G
    G --> H[特征值分解获得变换向量]
    H --> I[选择n_components个主成分 自动确定]
  
    I --> J[源域特征变换 295 n_components]
    I --> K[目标域特征变换 190 n_components]
  
    J --> L[LogisticRegression训练]
    K --> L
    L --> M[目标域预测结果]
```

**与理论的关键区别**：

- **求解方法**：ADAPT使用 `linalg.lstsq()` 最小二乘法，不是广义特征值分解
- **组件数量**：`n_components=None` 时自动确定，不固定为6个
- **实现简化**：直接在 `fit_transform()` 中完成训练，无需单独的 `fit()` 步骤

##### 4.4.2.1.9 TCA与其他UDA方法的比较

基于 `adapt_methods.py`中的ADAPT库集成：

**TCA vs CORAL（Correlation Alignment）**：

```python
# TCA: 基于核方法的子空间学习
TCA(estimator=LogisticRegression(penalty=None),
    n_components=6, mu=0.1, kernel='linear')

# CORAL: 协方差矩阵对齐
CORAL(estimator=LogisticRegression(penalty=None),
      lambda_=1.0)  # 协方差对齐强度
```

**比较维度**：

| 方法  | 理论基础          | 计算复杂度 | 适用场景     | 参数调优难度          |
| ----- | ----------------- | ---------- | ------------ | --------------------- |
| TCA   | 核方法+特征值分解 | O(n³)     | 非线性域差异 | 中等（3个主参数）     |
| CORAL | 协方差矩阵对齐    | O(d³)     | 线性域差异   | 简单（1个参数）       |
| SA    | 子空间对齐        | O(d²n)    | 中等域差异   | 简单（维度选择）      |
| KMM   | 核均值匹配        | O(n²)     | 样本权重调整 | 复杂（核参数+正则化） |

**医疗应用中的TCA优势**：

1. **理论保证**：基于核方法的严格数学理论
2. **灵活性**：支持多种核函数适应不同数据特性
3. **降维能力**：从原始特征（8维）到TCA子空间（6维），提取最重要的域不变特征
4. **平衡性**：通过mu参数平衡域适应和特征保持

#### 4.4.2.2 其他UDA方法简介

基于ADAPT库的其他域适应方法：

**CORAL（Correlation Alignment）**：

```python
from adapt.feature_based import CORAL

CORAL(estimator=LogisticRegression(penalty=None),
      lambda_=1.0)  # 协方差对齐强度
```

- **原理**：通过对齐源域和目标域的二阶统计量（协方差矩阵）来减少域差异
- **优势**：计算简单，无需超参数调优
- **适用场景**：线性域差异，特征维度不太高的情况

**SA（Subspace Alignment）**：

```python
from adapt.feature_based import SA

SA(estimator=LogisticRegression(penalty=None),
   n_components=6)  # 子空间维度
```

- **原理**：通过子空间对齐减少源域和目标域的分布差异
- **优势**：简单有效，计算复杂度低
- **适用场景**：中等程度的域差异

**KMM（Kernel Mean Matching）**：

```python
from adapt.instance_based import KMM

KMM(estimator=LogisticRegression(penalty=None),
    kernel='rbf', gamma=1.0, B=1000)
```

- **原理**：通过重新加权源域样本来匹配目标域的分布
- **优势**：理论基础扎实，适用性广
- **缺点**：参数调优复杂，计算开销大

**KLIEP（Kullback-Leibler Importance Estimation Procedure）**：

```python
from adapt.instance_based import KLIEP

KLIEP(estimator=LogisticRegression(penalty=None),
      kernel='rbf', gamma=1.0)
```

- **原理**：通过KL散度最小化估计重要性权重
- **优势**：统计学理论保证
- **适用场景**：需要精确密度估计的场景

### 4.4.3 集成架构流程

**实际实现流程**（基于CompleteAnalysisRunner）：

```mermaid
graph LR
    A[原始特征] --> B[TabPFN预处理]
    B --> C[TabPFN特征提取128维]
    C --> D[UDA域对齐ADAPT库]
    D --> E[逻辑回归分类]
    E --> F[预测结果]
```

## 4.5 训练策略

### 4.5.1 TabPFN配置

**预训练模型配置**：

```python
TabPFNClassifier(
    n_estimators=4,           # 集成数量
    balance_probabilities=True, # 类别平衡
    device="auto",            # 自动选择GPU/CPU
    fit_mode="fit_preprocessors", # 预处理缓存模式
    random_state=42
)
```

### 4.5.2 UDA训练策略

**ADAPT库训练**：

```python
# TCA方法训练
uda_method = create_adapt_method('TCA', 
    estimator=LogisticRegression(penalty="none"),
    n_components=20, mu=0.1, kernel='linear'
)
uda_method.fit(X_source, y_source, X_target)
```

**超参数配置**（基于源码）：

-     TCA components: 20
- TCA mu: 0.1
- TCA kernel: 'linear'
- 基础分类器: LogisticRegression(penalty="none")

## 4.6 验证设计

### 4.6.1 CompleteAnalysisRunner分析框架

**双重数据加载策略**：

```python
def run_complete_analysis(self):
    # 1. CV分析数据加载（兼容性）
    cv_data = self.load_data_for_cv()  # 使用selected58特征
  
    # 2. UDA分析数据加载（优化的）
    uda_data = self.load_data_for_uda()  # 使用best8特征+预处理
```

### 4.6.2 验证流程

**源域10折交叉验证**：

```python
def run_source_domain_cv(self):
    # TabPFN vs 机器学习基线 vs 传统方法
    methods = ['TABPFN', 'SVM', 'RF', 'GBDT', 'XGBoost', 'PKUPH', 'Mayo', 'Paper_LR']
    # 10折分层交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

**外部验证（跨域）**：

```python
def run_uda_methods_comparison(self):
    # 在源域A训练，目标域B测试
    # 方法对比：TabPFN_NoUDA vs UDA方法 vs 基线
    uda_methods = ['TCA', 'SA', 'CORAL', 'KMM']
```

## 4.7 评估指标

### 4.7.1 性能指标

基于实际代码实现的指标计算：

```python
def calculate_metrics(y_true, y_pred, y_pred_proba):
    return {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }
```

### 4.7.2 可视化评估

**UDAVisualizer功能**：

```python
def visualize_domain_adaptation_complete(self):
    # 1. PCA/t-SNE降维可视化
    # 2. 标准化距离度量计算
    # 3. 性能对比图表生成
```

**距离度量**：

- normalized_linear_discrepancy
- normalized_frechet_distance
- normalized_wasserstein_distance
- normalized_kl_divergence

### 4.7.3 结果保存

**自动化结果管理**：

```python
# 时间戳目录结构
output_dir = f"results/complete_analysis_{timestamp}"
# 保存内容：
# - complete_results.json：完整数值结果
# - analysis_report.md：分析报告  
# - *.png：可视化图表
# - uda_*/：各UDA方法详细分析
```

## 4.8 实现框架

### 4.8.1 技术栈

**核心依赖**：

- Python 3.9+
- TabPFN 2.0+
- adapt-python 0.4+（ADAPT库）
- PyTorch 2.0+
- scikit-learn 1.3+

### 4.8.2 关键模块

**数据层**：`MedicalDataLoader` - 医疗数据加载和预处理
**UDA层**：`adapt_methods.py` - ADAPT库方法封装
**可视化层**：`UDAVisualizer` - 域适应效果可视化
**执行层**：`CompleteAnalysisRunner` - 自动化分析引擎

### 4.8.3 实验配置

**默认实验配置**：

```python
config = {
    'feature_set': 'best8',      # 8个最优特征
    'scaler_type': 'none',       # 不使用标准化
    'imbalance_method': 'none',  # 不处理类别不平衡
    'cv_folds': 10,              # 10折交叉验证
    'random_state': 42           # 固定随机种子
}
```

这份基于实际源码的Methods文档准确反映了TabPFN+UDA项目的真实实现，包括预训练模型的使用方式、ADAPT库的集成、数据集的实际结构等关键技术细节。
