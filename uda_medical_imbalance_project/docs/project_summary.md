# UDA Medical Imbalance Project - 项目总结

## 📋 项目概述

本项目是一个专注于医疗数据不平衡问题与无监督域适应（UDA）的综合性机器学习实验框架。基于您提供的需求规范，我们构建了一个结构化、可复现的实验平台。

## 🎯 核心功能

### 1. 数据预处理流程

- **RFE预筛选特征集**：基于递归特征消除预筛选的最优特征组合
  - **BEST7特征集**：7个特征中包含2个类别特征（Feature63, Feature46）
  - **BEST8特征集**：8个特征中包含2个类别特征（Feature63, Feature46）
  - **BEST9特征集**：9个特征中包含2个类别特征（Feature63, Feature46）
  - **BEST10特征集**：10个特征中包含3个类别特征（Feature63, Feature46, Feature5）
  - **ALL63特征集**：63个特征中包含多个类别特征
- **标准化处理**：提供StandardScaler、RobustScaler和NoScaler（不标准化）三种选择
- **类别不平衡处理**：集成10种不平衡处理方法，包括SMOTE系列、ADASYN、组合方法和欠采样
- **类别特征处理**：自动识别和处理类别特征，支持混合数据类型

### 2. 源域方法对比

- **10折交叉验证**：确保结果的稳定性和可靠性
- **多模型支持**：
  - TabPFN模型（自动表格深度学习）
  - 论文方法（领域特定方法）
  - PKUPH基线模型（北京大学人民医院标准）
  - Mayo基线模型（梅奥诊所标准）

### 3. UDA算法体系

#### 3.1 协变量偏移方法
- **DM (Discriminative Method)**：实例重加权方法

#### 3.2 隐藏协变量偏移方法

**线性和核方法**：
- **SA (Subspace Alignment)**：子空间对齐
- **TCA (Transfer Component Analysis)**：迁移成分分析
- **JDA (Joint Distribution Adaptation)**：联合分布适应
- **CORAL (CORrelation ALignment)**：相关性对齐

**深度学习方法**：
- **DANN (Domain-Adversarial Neural Networks)**
- **ADDA (Adversarial Discriminative Domain Adaptation)**
- **WDGRL (Wasserstein Distance Guided Representation Learning)**
- **DeepCORAL (Deep CORAL)**
- **MCD (Maximum Classifier Discrepancy)**
- **MDD (Margin Disparity Discrepancy)**
- **CDAN (Conditional Domain Adversarial Network)**

#### 3.3 最优传输方法
- **POT (Python Optimal Transport)**

### 4. 可视化分析

#### 4.1 域适应前后对比
- **PCA降维可视化**：观察域间分布变化
- **t-SNE可视化**：非线性降维展示聚类效果
- **特征分布对比**：
  - KL散度计算和可视化
  - Wasserstein距离度量
  - MMD距离分析

#### 4.2 性能指标可视化
- **ROC-AUC曲线**：多方法ROC对比
- **混淆矩阵**：分类结果详细分析
- **性能雷达图**：多指标综合展示
- **方法对比柱状图**：直观性能对比

### 5. 性能评估体系

- **主要指标**：AUC、准确率、F1、精确率、召回率
- **目标域评估**：同时提供加DA和不加DA的结果对比
- **统计分析**：均值、标准差、显著性检验

## 🏗️ 项目架构

```
uda_medical_imbalance_project/
├── 📁 config/                    # 配置管理
│   ├── experiment_config.py      # 实验配置类
│   └── __init__.py              # 模块初始化
├── 📁 data/                     # 数据处理
│   └── loader.py               # 医疗数据加载器
├── 📁 preprocessing/            # 数据预处理
│   ├── scalers.py              # 标准化（Standard/Robust）
│   ├── imbalance_handler.py    # 不平衡处理（SMOTE等）
│   └── categorical_encoder.py  # 类别特征编码
├── 📁 uda/                      # 无监督域适应
│   ├── covariate_shift/        # 协变量偏移方法
│   ├── hidden_shift/           # 隐藏协变量偏移方法
│   │   ├── linear_kernel/      # 线性和核方法
│   │   ├── deep/              # 深度学习方法
│   │   └── optimal_transport/  # 最优传输方法
│   └── uda_factory.py          # UDA方法工厂
├── 📁 modeling/                 # 机器学习模型
│   ├── baseline_models.py      # 基线模型（PKUPH、Mayo）
│   └── paper_methods.py        # 论文方法实现
├── 📁 evaluation/              # 评估模块
│   ├── metrics.py              # 评估指标计算
│   ├── cross_validation.py     # 交叉验证
│   ├── performance_analyzer.py # 性能分析
│   └── comparator.py           # 方法对比
├── 📁 visualization/           # 可视化
│   ├── dimensionality_plots.py # PCA、t-SNE可视化
│   ├── distribution_plots.py   # 特征分布对比
│   ├── distance_plots.py       # 距离度量可视化
│   ├── performance_plots.py    # 性能指标图表
│   └── comparison_plots.py     # 方法对比图表
├── 📁 scripts/                 # 执行脚本
│   ├── main_experiment.py      # 主实验脚本
│   └── run_full_uda_experiment.py # 完整实验流程
├── 📁 examples/                # 示例代码
│   └── quick_start_example.py  # 快速开始示例
├── 📁 tests/                   # 测试套件
│   ├── test_categorical_features.py    # 类别特征测试
│   ├── test_real_data_scalers.py       # 标准化器测试
│   ├── test_imbalance_handler.py       # 不平衡处理器测试
│   ├── test_imbalance_comprehensive.py # 全面不平衡处理测试
│   ├── imgs/                           # 测试生成的图像
│   │   ├── scalers/                    # 标准化器测试图像
│   │   └── imbalance_handler/          # 不平衡处理器测试图像
│   ├── conftest.py                     # 测试配置
│   └── __init__.py                     # 测试模块初始化
├── 📁 utils/                   # 工具函数
├── 📁 docs/                    # 详细文档
├── 📁 configs/                 # 配置文件
│   └── default_config.yaml    # 默认配置
├── README.md                   # 项目说明
├── requirements.txt            # 依赖列表
└── pytest.ini                 # 测试配置
```

## 📊 数据使用规范

### 数据域定义
- **源域 (X_source, Y_source)**：全部数据用于训练和UDA特征对齐
- **目标域 (X_target)**：全部无标签数据用于UDA特征对齐
- **目标域标签 (Y_target)**：仅用于模型最终性能评估

### 数据流程
```
原始数据 → RFE预筛选特征集 → 标准化 → 不平衡处理 → UDA对齐 → 模型训练 → 评估
```

## 🚀 快速开始

### 1. 安装依赖
```bash
cd uda_medical_imbalance_project
pip install -r requirements.txt
```

### 2. 运行示例
```bash
# 快速开始示例
python examples/quick_start_example.py

# 完整实验流程
python scripts/main_experiment.py --log-level INFO
```

### 3. 自定义配置
```bash
# 使用自定义配置运行
python scripts/run_full_uda_experiment.py --config configs/custom_config.yaml
```

## 📈 实验配置示例

```yaml
# 数据预处理配置
preprocessing:
  feature_set: "best7"         # 特征集选择 (best7|best8|best9|best10|all63)
  scaler: "standard"           # 标准化方法 (standard|robust|none)
  imbalance_method: "smote"    # 不平衡处理方法 (none|smote|smotenc|borderline_smote|kmeans_smote|svm_smote|adasyn|smote_tomek|smote_enn|random_under)
  categorical_features: []      # 类别特征列表（自动从settings.py获取）

# UDA方法配置
uda:
  linear_kernel_methods:
    - "SA"
    - "TCA" 
    - "JDA"
    - "CORAL"
  deep_methods:
    - "DANN"
    - "DeepCORAL"
  optimal_transport_methods:
    - "POT"

# 评估配置
evaluation:
  metrics:
    - "auc"
    - "accuracy"
    - "f1"
    - "precision"
    - "recall"
```

## 🎯 实验输出

### 结果目录结构
```
experiments/experiment_YYYYMMDD_HHMMSS/
├── config/                 # 实验配置备份
├── preprocessed_data/      # 预处理后的数据
├── uda_results/           # UDA变换结果
├── model_predictions/     # 模型预测结果
├── evaluation_metrics/    # 评估指标
├── visualizations/        # 可视化图表
│   ├── pca_plots/
│   ├── tsne_plots/
│   ├── distribution_plots/
│   └── performance_plots/
└── experiment_report.html # 实验报告
```

## 📝 关键特性

### 1. 模块化设计
- 每个组件独立可测试、可配置、易扩展
- 清晰的接口定义和工厂模式

### 2. 灵活配置
- YAML配置文件支持
- 命令行参数覆盖
- 多种预定义配置模板

### 3. 全面测试
- 单元测试覆盖
- 集成测试支持
- 性能测试框架

### 4. 丰富可视化
- 英文图像标签（符合科技写作标准）
- 多维度对比分析
- 高质量图表输出

### 5. 实验跟踪
- 完整的实验记录
- 可重现的实验配置
- 详细的性能报告

## 🧪 测试验证

### 标准化器测试结果

通过运行 `tests/test_real_data_scalers.py`，我们验证了不同标准化器的效果：

#### 标准化器类型
- **StandardScaler**：基于均值和标准差的标准化（均值0，标准差1）
- **RobustScaler**：基于中位数和IQR的鲁棒标准化（中位数0，IQR标准化）
- **NoScaler**：不进行标准化，保持原始数据不变

### 类别特征测试结果

通过运行 `tests/test_categorical_features.py`，我们验证了不同特征集的类别特征配置：

#### BEST7特征集分析（RFE预筛选）
- **总特征数**：7个
- **类别特征**：2个（Feature63, Feature46）
- **数值特征**：5个（Feature2, Feature39, Feature42, Feature43, Feature56）
- **类别特征比例**：28.6%

#### BEST8特征集分析（RFE预筛选）
- **总特征数**：8个
- **类别特征**：2个（Feature63, Feature46）
- **数值特征**：6个（Feature2, Feature61, Feature56, Feature42, Feature39, Feature43）
- **类别特征比例**：25.0%

#### BEST9特征集分析（RFE预筛选）
- **总特征数**：9个
- **类别特征**：2个（Feature63, Feature46）
- **数值特征**：7个（Feature2, Feature61, Feature56, Feature42, Feature39, Feature43, Feature48）
- **类别特征比例**：22.2%

#### BEST10特征集分析（RFE预筛选）
- **总特征数**：10个
- **类别特征**：3个（Feature63, Feature46, Feature5）
- **数值特征**：7个（Feature2, Feature61, Feature56, Feature42, Feature39, Feature43, Feature48）
- **类别特征比例**：30.0%

#### ALL特征集分析
- **总特征数**：58个
- **类别特征**：20个
- **数值特征**：38个
- **类别特征比例**：34.5%

### 不平衡处理器测试结果

通过运行 `tests/test_imbalance_comprehensive.py`，我们验证了10种不平衡处理方法的效果：

#### 支持的不平衡处理方法
- **none**：不进行重采样，保持原始数据分布
- **smote**：标准SMOTE合成少数类样本
- **smotenc**：处理类别特征的SMOTE变体
- **borderline_smote**：边界线SMOTE，关注边界区域样本
- **kmeans_smote**：基于K-means聚类的SMOTE
- **svm_smote**：基于SVM的SMOTE变体
- **adasyn**：自适应合成采样，根据密度生成样本
- **smote_tomek**：SMOTE + Tomek Links清理组合方法
- **smote_enn**：SMOTE + Edited Nearest Neighbours清理组合方法
- **random_under**：随机欠采样多数类

#### 测试数据集特征
- **原始数据**：295个样本（106正例，189负例），不平衡比例约1:1.78
- **特征集**：使用BEST10特征集（10个特征，包含3个类别特征）
- **类别特征**：Feature63, Feature46, Feature5

#### 处理效果摘要
- **过采样方法**：生成377-378个样本，实现完全或接近完全平衡
- **组合方法**：在过采样基础上清理噪声，样本数略少于纯过采样
- **欠采样方法**：减少到212个样本，实现完全平衡但信息损失较大

#### 可视化输出
测试生成的可视化图像保存在 `tests/imgs/imbalance_handler/` 目录：
- **综合对比图**：6个子图展示不同方法的效果对比
- **PCA降维可视化**：展示各方法处理前后的2D数据分布
- **t-SNE降维可视化**：非线性降维展示聚类效果
- **详细统计表格**：各方法的样本数量和平衡改善评分

### 测试运行方式

```bash
# 运行标准化器测试（包含可视化）
cd uda_medical_imbalance_project
python tests/test_real_data_scalers.py

# 运行类别特征测试
python tests/test_categorical_features.py

# 运行不平衡处理器测试
python tests/test_imbalance_handler.py

# 运行全面不平衡处理测试（包含可视化）
python tests/test_imbalance_comprehensive.py

# 运行所有测试
pytest tests/ -v
```

## 🔬 技术实现细节

### 不平衡处理器实现特点

#### 1. 智能类别特征处理
- **自动检测**：通过`settings.py`中的`get_categorical_features_for_feature_set()`函数自动获取类别特征索引
- **混合数据支持**：所有方法都能正确处理包含类别特征的混合数据类型
- **SMOTENC集成**：组合方法（SMOTETomek、SMOTEENN）自动使用SMOTENC作为基础SMOTE方法

#### 2. 组合方法配置策略
```python
# SMOTETomek配置：SMOTENC + Tomek Links清理
if categorical_indices:
    smote_sampler = SMOTENC(categorical_features=categorical_indices, random_state=42)
else:
    smote_sampler = SMOTE(random_state=42)
sampler = SMOTETomek(smote=smote_sampler, random_state=42)

# SMOTEENN配置：SMOTENC + ENN清理  
if categorical_indices:
    smote_sampler = SMOTENC(categorical_features=categorical_indices, random_state=42)
else:
    smote_sampler = SMOTE(random_state=42)
sampler = SMOTEENN(smote=smote_sampler, random_state=42)
```

#### 3. 平衡判断逻辑
- **不平衡阈值**：当少数类比例 < 0.5时判断为不平衡
- **智能跳过**：对于已经平衡的数据集，某些方法会智能跳过重采样
- **强制执行模式**：测试模式下可强制执行所有方法以便对比分析

#### 4. 可视化集成
- **PCA降维**：2D可视化展示重采样前后的数据分布变化
- **t-SNE降维**：非线性降维揭示数据的聚类结构
- **统计对比**：详细的样本数量、平衡比例、改善评分统计
- **方法分组**：按过采样、组合方法、欠采样分组展示效果

## 🔧 技术栈

- **核心框架**：Python 3.8+, NumPy, Pandas, Scikit-learn
- **深度学习**：PyTorch，TF
- **域适应**：Adapt-python
- **不平衡处理**：Imbalanced-learn
- **可视化**：Matplotlib, Seaborn, Plotly
- **配置管理**：PyYAML, Hydra
- **测试框架**：Pytest
- **实验跟踪**：MLflow, WandB

## 🎓 符合项目规则

本项目严格遵循TabPFN项目规则：

1. **医疗数据处理模式**：标准化处理流程，跨数据集统一性
2. **跨数据集实验命名**：按A/B/C标识不同数据集
3. **特征选择策略**：RFE为主，关注6-8个特征的性能
4. **结果目录组织**：独立results目录，可解释性分析子目录
5. **模型评估标准**：AUC为主要指标，考虑多指标稳定性

## 🤝 使用建议

1. **开始实验前**：仔细阅读配置文件，根据数据集特点调整参数
2. **特征选择**：建议从7个特征开始，逐步增加到10个进行对比
3. **UDA方法选择**：先测试简单方法（CORAL、SA），再尝试复杂方法
4. **性能评估**：重点关注AUC和F1分数，考虑医疗场景的特殊需求
5. **结果解释**：结合可视化分析理解域适应的效果

## 📚 进一步扩展

- 添加更多医疗数据集支持
- 集成更多先进的域适应方法
- 添加自动超参数优化
- 支持多GPU并行训练
- 集成模型解释性分析工具

---

**注意**：本项目遵循中文回复 + 英文图像标签的统一语言输出风格，确保科研标准的图表质量。 