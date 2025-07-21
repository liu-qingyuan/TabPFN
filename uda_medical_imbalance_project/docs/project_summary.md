# UDA Medical Imbalance Project - 项目总结

## 📋 项目概述

本项目是一个专注于医疗数据不平衡问题与无监督域适应（UDA）的综合性机器学习实验项目，基于ADAPT库实现多种域适应算法，提供完整的可视化分析和性能评估框架。

## 🌟 项目特色

- 🏥 **医疗数据专用**：针对多医院医疗数据集的跨域适应和不平衡处理
- ⚖️ **不平衡数据处理**：集成SMOTE、BorderlineSMOTE、ADASYN等先进方法
- 🔄 **基于ADAPT库的UDA算法**：使用成熟的adapt-python库实现多种域适应方法
- 🤖 **多模型对比**：TabPFN、经典基线模型、论文方法全面对比
- 📊 **专业UDA可视化**：PCA、t-SNE、标准化距离度量、性能对比等多维度可视化
- 🔧 **模块化设计**：每个组件独立可测试、可配置、易扩展
- 📈 **全面评估**：ROC-AUC、准确率、F1、精确率召回率等多指标评估
- 🎯 **灵活配置**：支持多种特征选择、标准化和不平衡处理策略
- 🚀 **一键式完整分析**：通过CompleteAnalysisRunner实现端到端自动化分析流程

## 🏗️ 项目架构

### 目录结构

```
uda_medical_imbalance_project/
├── 📁 config/                    # 配置管理
│   ├── model_config.py          # 模型配置（TabPFN、基线、论文方法）
│   ├── uda_config.py            # UDA算法配置
│   └── experiment_config.py     # 实验全局配置
├── 📁 data/                     # 数据处理
│   ├── loader.py               # 医疗数据加载器
│   └── validator.py            # 数据验证器
├── 📁 preprocessing/            # 数据预处理与UDA可视化
│   ├── scalers.py              # 标准化（Standard/Robust）
│   ├── imbalance_handler.py    # 不平衡处理（SMOTE等）
│   ├── uda_processor.py        # UDA数据处理器
│   └── uda_visualizer.py       # UDA专业可视化分析器
├── 📁 uda/                      # 基于ADAPT库的域适应
│   └── adapt_methods.py        # ADAPT库UDA方法包装器
├── 📁 modeling/                 # 机器学习模型
│   ├── baseline_models.py      # 基线模型（PKUPH、Mayo）
│   └── paper_methods.py        # 论文方法实现
├── 📁 evaluation/              # 评估模块
│   ├── metrics.py              # 评估指标计算
│   ├── cross_validation.py     # 交叉验证
│   ├── performance_analyzer.py # 性能分析
│   └── comparator.py           # 方法对比
├── 📁 examples/                # 使用示例
│   ├── quick_start_example.py  # 快速开始示例
│   ├── uda_usage_example.py    # UDA方法使用示例
│   ├── real_data_visualization.py # 真实数据可视化
│   └── uda_visualization_example.py # UDA可视化示例
├── 📁 scripts/                 # 执行脚本
│   ├── run_complete_analysis.py     # 完整分析流程（新增核心引擎）
│   ├── run_source_domain_comparison.py # 源域方法对比
│   ├── run_uda_methods.py            # UDA方法运行
│   ├── run_preprocessing.py          # 预处理流程
│   └── visualize_results.py          # 结果可视化
├── 📁 tests/                   # 测试套件
│   ├── test_categorical_features.py  # 类别特征测试
│   ├── test_real_data_scalers.py     # 标准化器测试
│   ├── test_imbalance_handler.py     # 不平衡处理器测试
│   ├── test_imbalance_comprehensive.py # 全面不平衡处理测试
│   └── test_adapt_methods.py         # ADAPT方法测试
├── 📁 results/                 # 实验结果输出
├── 📁 docs/                    # 详细文档
└── 📁 configs/                 # 配置文件
```

## 🚀 CompleteAnalysisRunner - 核心分析引擎

### 完整分析流程

**CompleteAnalysisRunner** 是项目的核心分析引擎，提供端到端的自动化分析流程，包含六个主要步骤：

1. **数据加载与预处理** - 双重加载策略确保兼容性
2. **源域交叉验证** - TabPFN vs 传统基线 vs 机器学习基线
3. **UDA方法对比** - 无UDA基线 vs UDA方法 vs 其他基线
4. **可视化生成** - ROC曲线、校准曲线、决策曲线、雷达图
5. **报告生成** - 自动识别最佳方法和域适应效果
6. **结果保存** - 结构化输出所有结果和可视化

### 技术架构层次

项目采用分层技术架构，**CompleteAnalysisRunner** 作为核心引擎协调各层组件：

- **数据层**：管理多医院医疗数据集和特征集
- **预处理层**：数据加载、特征选择、标准化、不平衡处理
- **模型层**：TabPFN、传统基线、机器学习基线、UDA方法
- **评估层**：交叉验证、性能指标、域适应评估、预测数据收集
- **可视化层**：ROC曲线、校准曲线、决策曲线、雷达图、UDA专业可视化
- **输出层**：结构化保存JSON结果、Markdown报告、PNG图表

### 双重数据加载策略

CompleteAnalysisRunner采用智能的双重数据加载策略：

1. **CV分析数据加载** (`load_data_for_cv`)
   - 使用 `selected58` 特征集（兼容所有基线模型）
   - TabPFN从中选择指定的特征子集
   - 传统基线（PKUPH、Mayo、Paper_LR）使用全部58个特征
   - 机器学习基线使用指定特征集配置

2. **UDA分析数据加载** (`load_data_for_uda`)
   - 使用指定的特征集（如best8）
   - 应用完整的预处理流程（标准化 + 不平衡处理）
   - 确保UDA方法获得最优的数据质量

## 🔬 核心功能详解

### 1. 数据预处理流程

#### 1.1 预筛选特征集
- **特征来源**：基于RFE（递归特征消除）预筛选的最优特征
- **可选特征集**：
  - **best7**：7个最优特征
  - **best8**：8个最优特征  
  - **best9**：9个最优特征
  - **best10**：10个最优特征
  - **all63**：全部63个选定特征
- **类别特征处理**：自动识别和处理混合数据类型

#### 1.2 标准化方法
```python
# 可选的标准化方法
scalers = {
    'standard': StandardScaler(),
    'robust': RobustScaler(),
    'none': NoScaler()  # 不进行标准化
}
```

#### 1.3 类别不平衡处理
```python
# 支持的不平衡处理方法
imbalance_methods = {
    'none': None,                           # 不进行重采样
    'smote': SMOTE(),                       # 标准SMOTE
    'smotenc': SMOTENC(),                   # 处理类别特征的SMOTE
    'borderline_smote': BorderlineSMOTE(),  # 边界线SMOTE
    'kmeans_smote': KMeansSMOTE(),         # K-means聚类SMOTE
    'svm_smote': SVMSMOTE(),               # SVM-SMOTE
    'adasyn': ADASYN(),                    # 自适应合成采样
    'smote_tomek': SMOTETomek(),           # SMOTE + Tomek清理
    'smote_enn': SMOTEENN(),               # SMOTE + ENN清理
    'random_under': RandomUnderSampler()   # 随机欠采样
}
```

### 2. 基于ADAPT库的UDA方法

#### 2.1 实例重加权方法 (Instance-Based)

**KMM (Kernel Mean Matching)**
```python
kmm_method = create_adapt_method(
    method_name='KMM',
    estimator=LogisticRegression(penalty="none"),
    kernel='rbf',        # 核函数类型
    gamma=1.0,          # 核函数带宽
    verbose=0,
    random_state=42
)
```

**KLIEP (Kullback-Leibler Importance Estimation Procedure)**
```python
kliep_method = create_adapt_method(
    method_name='KLIEP',
    estimator=LogisticRegression(penalty="none"),
    gamma=1.0,
    verbose=0,
    random_state=42
)
```

#### 2.2 特征对齐方法 (Feature-Based)

**CORAL (CORrelation ALignment)**
```python
coral_method = create_adapt_method(
    method_name='CORAL',
    estimator=LogisticRegression(penalty="none"),
    lambda_=1.0,        # 正则化参数
    verbose=0,
    random_state=42
)
```

**SA (Subspace Alignment)**
```python
sa_method = create_adapt_method(
    method_name='SA',
    estimator=LogisticRegression(penalty="none"),
    n_components=None,   # 主成分数量
    verbose=0,
    random_state=42
)
```

**TCA (Transfer Component Analysis)**
```python
tca_method = create_adapt_method(
    method_name='TCA',
    estimator=LogisticRegression(penalty="none"),
    n_components=6,      # 传输成分数量
    mu=0.1,             # 正则化参数
    kernel='linear',     # 核函数类型
    verbose=0,
    random_state=42
)
```

**fMMD (feature-based Maximum Mean Discrepancy)**
```python
fmmd_method = create_adapt_method(
    method_name='FMMD',
    estimator=LogisticRegression(penalty="none"),
    gamma=1.0,
    verbose=0,
    random_state=42
)
```

#### 2.3 深度学习方法 (Deep Learning)

**DANN (Domain-Adversarial Neural Networks)**
```python
dann_method = create_adapt_method(
    method_name='DANN',
    lambda_=1.0,        # 域适应损失权重
    lr=0.001,           # 学习率
    epochs=100,         # 训练轮数
    batch_size=32,      # 批次大小
    verbose=0,
    random_state=42
)
```

## 🎨 UDA专业可视化分析

### UDAVisualizer功能特点

**UDAVisualizer** 提供完整的域适应效果可视化分析：

```python
from preprocessing.uda_visualizer import UDAVisualizer

# 创建可视化器
visualizer = UDAVisualizer(
    figsize=(12, 8),
    save_plots=True,
    output_dir="results/uda_visualization"
)

# 完整可视化分析
results = visualizer.visualize_domain_adaptation_complete(
    X_source, y_source, X_target, y_target,
    uda_method=uda_method,
    method_name="TCA"
)
```

### 主要可视化功能

#### 1. 降维可视化
- **PCA可视化**：主成分分析展示域间分布
- **t-SNE可视化**：非线性降维展示聚类效果
- **域适应前后对比**：直观展示UDA效果

#### 2. 标准化距离度量
```python
# 支持的标准化距离指标
distance_metrics = {
    'normalized_linear_discrepancy': '标准化线性差异',
    'normalized_frechet_distance': '标准化Frechet距离', 
    'normalized_wasserstein_distance': '标准化Wasserstein距离',
    'normalized_kl_divergence': '标准化KL散度'
}
```

**特点**：
- 只显示标准化版本的距离指标，确保跨不同UDA方法的可比性
- 避免维度变化导致的数值异常问题
- 提供稳定可靠的域适应效果评估

#### 3. 智能特征处理
- **维度兼容性检查**：自动处理特征维度变化的UDA方法
- **特征变换策略**：
  - TCA/SA/FMMD：可能改变特征维度，特殊处理
  - CORAL：协方差对齐，维度不变
  - KMM/KLIEP：实例重加权，不改变特征
- **回退机制**：维度不匹配时自动使用原始特征空间进行距离计算

#### 4. 性能对比可视化
- **基线对比**：UDA方法 vs 无域适应基线
- **多指标展示**：准确率、AUC、F1、精确率、召回率
- **改进程度量化**：域适应带来的性能提升

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
CUDA >= 11.0 (推荐GPU加速)
```

### 安装依赖

```bash
cd uda_medical_imbalance_project
pip install -r requirements.txt
```

### 一键运行完整分析

```bash
# 运行完整分析流程（推荐）
python scripts/run_complete_analysis.py

# 这将自动执行：
# 1. 源域10折交叉验证对比（TabPFN vs 基线模型）
# 2. UDA域适应方法对比（TCA、SA、CORAL、KMM等）
# 3. 生成完整可视化分析报告
# 4. 输出性能对比和改进建议
```

### 运行示例脚本

```bash
# 完整分析流程（推荐）- 一键运行所有分析
python scripts/run_complete_analysis.py

# 独立模块示例
python examples/quick_start_example.py          # 快速开始示例
python examples/uda_usage_example.py            # UDA方法使用示例
python examples/real_data_visualization.py      # 真实数据可视化分析
python examples/uda_visualization_example.py    # UDA可视化示例

# 分步骤运行（高级用户）
python scripts/run_source_domain_comparison.py  # 仅源域方法对比
python scripts/run_uda_methods.py              # 仅UDA方法分析
python scripts/visualize_results.py            # 仅结果可视化
```

## 📊 CompleteAnalysisRunner使用指南

### 基本配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `feature_set` | str | 'best8' | 特征集选择，影响TabPFN和机器学习基线的特征数量 |
| `scaler_type` | str | 'none' | 标准化方法，只影响TabPFN和UDA方法 |
| `imbalance_method` | str | 'none' | 不平衡处理方法，只影响TabPFN和UDA方法 |
| `cv_folds` | int | 10 | 交叉验证折数，影响所有CV分析 |
| `random_state` | int | 42 | 随机种子，确保结果可重复性 |
| `output_dir` | Optional[str] | None | 输出目录，None时自动生成 |
| `verbose` | bool | True | 是否显示详细执行信息 |

### 分析流程详解

#### 1. 数据加载与预处理
- **双重数据加载策略**
- **智能特征对齐**
- **自动回退机制**

#### 2. 源域10折交叉验证
- **TabPFN方法**：使用指定特征集 + 预处理配置
- **传统基线**：PKUPH、Mayo、Paper_LR（使用selected58特征集）
- **机器学习基线**：SVM、DT、RF、GBDT、XGBoost（使用相同配置）

#### 3. UDA方法对比分析
- **基线对比**：TabPFN无UDA vs 传统基线 vs 机器学习基线
- **UDA方法**：TCA、SA、CORAL、KMM等ADAPT库方法
- **目标域测试**：所有方法在目标域B上评估

#### 4. 专业可视化生成
- **ROC曲线对比**：源域CV vs UDA方法性能对比
- **校准曲线分析**：模型预测概率校准效果
- **决策曲线分析**：临床决策价值评估
- **性能雷达图**：多维度性能指标可视化

#### 5. 分析报告生成
- **Markdown格式报告**：包含所有性能指标和结论
- **最佳方法识别**：自动识别各类别中的最佳方法
- **域适应效果评估**：量化UDA方法的改进程度

#### 6. 结果保存与输出
- **JSON结果文件**：完整的实验结果和配置
- **可视化图表**：PNG格式的专业图表
- **分析报告**：详细的Markdown分析报告

## 📈 实验结果输出

### CompleteAnalysisRunner输出结构

```
results/complete_analysis_YYYYMMDD_HHMMSS/
├── 📋 analysis_report.md                    # 完整分析报告
├── 📊 complete_results.json                 # 完整实验结果
├── 📈 source_domain_cv_results.json         # 源域CV详细结果
├── 🔄 uda_methods_results.json              # UDA方法详细结果
├── 📊 performance_comparison.png            # 性能对比图
├── 📈 roc_curves_comparison.png             # ROC曲线对比
├── 📉 calibration_curves.png                # 校准曲线分析
├── 🎯 decision_curve_analysis.png           # 决策曲线分析
├── 🕸️ performance_radar_chart.png           # 性能雷达图
└── 📁 uda_[method_name]/                    # 各UDA方法详细分析
    ├── TCA_RealData_dimensionality_reduction.png
    ├── TCA_RealData_distance_metrics.png
    └── TCA_RealData_performance_comparison.png
```

### 预期输出示例

运行 `python scripts/run_complete_analysis.py` 后，您将看到：

```bash
🏥 完整医疗数据UDA分析流程
============================================================
🔧 完整分析流程初始化
   特征集: best8
   标准化: none
   不平衡处理: none
   交叉验证: 10折
   输出目录: results/complete_analysis_20241230_143025

📊 加载医疗数据...
✅ 数据加载完成:
   源域A: (200, 58), 类别分布: {0: 120, 1: 80}
   目标域B: (180, 58), 类别分布: {0: 95, 1: 85}

🔬 源域10折交叉验证对比
✅ TabPFN 完成: AUC: 0.8456, Accuracy: 0.7892
✅ Paper_LR 完成: AUC: 0.8234, Accuracy: 0.7654
✅ PKUPH 完成: AUC: 0.8012, Accuracy: 0.7423

🔄 UDA方法对比分析
✅ TabPFN_NoUDA 完成: AUC: 0.7892, Accuracy: 0.7234
✅ TCA 完成: AUC: 0.8123, Accuracy: 0.7456
✅ CORAL 完成: AUC: 0.8045, Accuracy: 0.7389

📊 生成对比可视化图表
✅ ROC曲线对比 已保存
✅ 校准曲线分析 已保存
✅ 决策曲线分析 已保存
✅ 性能雷达图 已保存

📋 生成最终分析报告
✅ 完整分析流程完成！
📁 所有结果已保存到: results/complete_analysis_20241230_143025
📋 分析报告: results/complete_analysis_20241230_143025/analysis_report.md
```

## 🧪 测试验证体系

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_adapt_methods.py -v

# 运行不平衡处理器测试
python tests/test_imbalance_handler.py

# 运行全面不平衡处理测试（包含可视化）
python tests/test_imbalance_comprehensive.py
```

### 类别特征测试结果

#### BEST特征集分析（RFE预筛选）

- **BEST7特征集**：7个特征中包含2个类别特征（Feature63, Feature46）
- **BEST8特征集**：8个特征中包含2个类别特征（Feature63, Feature46）
- **BEST9特征集**：9个特征中包含2个类别特征（Feature63, Feature46）
- **BEST10特征集**：10个特征中包含3个类别特征（Feature63, Feature46, Feature5）
- **ALL63特征集**：63个特征中包含多个类别特征

### 不平衡处理器测试结果

#### 支持的10种不平衡处理方法

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

#### 处理效果特点

- **过采样方法**：生成377-378个样本，实现完全或接近完全平衡
- **组合方法**：在过采样基础上清理噪声，样本数略少于纯过采样
- **欠采样方法**：减少到212个样本，实现完全平衡但信息损失较大

## 📝 最新更新 (v2.0)

### 🎨 UDA可视化分析器重大改进
- **修复距离度量计算**：解决了维度变化导致的指标异常问题
- **标准化指标优先**：只显示标准化版本的距离度量，确保跨方法可比性
- **智能特征处理**：自动处理不同UDA方法的特征变换策略
- **跳过特征分布可视化**：避免特征尺度差异导致的可视化问题

### 🔧 技术改进
- **维度兼容性检查**：自动检测并处理特征维度变化
- **回退机制**：维度不匹配时使用原始特征空间计算距离
- **稳定性提升**：移除不稳定的非标准化指标显示

### 📊 距离度量优化
- **标准化线性差异**：基于ADAPT库的专业实现
- **标准化Frechet距离**：提供稳定的分布距离度量
- **标准化Wasserstein距离**：自定义实现，遵循ADAPT库标准
- **标准化KL散度**：改进的散度计算方法

## 🎯 符合项目规则

本项目严格遵循TabPFN项目规则：

1. **医疗数据处理模式**：标准化处理流程，跨数据集统一性
2. **跨数据集实验命名**：按A/B/C标识不同数据集
3. **特征选择策略**：RFE为主，关注6-8个特征的性能
4. **结果目录组织**：独立results目录，可解释性分析子目录
5. **模型评估标准**：AUC为主要指标，考虑多指标稳定性

## 🔧 技术栈

- **核心框架**：Python 3.8+, NumPy, Pandas, Scikit-learn
- **域适应**：Adapt-python（核心库）
- **不平衡处理**：Imbalanced-learn
- **深度学习**：PyTorch（可选）
- **可视化**：Matplotlib, Seaborn
- **配置管理**：PyYAML
- **测试框架**：Pytest

## 🤝 使用建议

1. **开始实验前**：运行`python scripts/run_complete_analysis.py`获得完整分析
2. **特征选择**：建议从best8特征开始，这是经过RFE优化的配置
3. **UDA方法选择**：先测试简单方法（CORAL、SA），再尝试复杂方法（TCA、KMM）
4. **性能评估**：重点关注AUC指标，结合可视化分析理解域适应效果
5. **结果解释**：查看生成的analysis_report.md获得详细分析结论

## 📚 进一步扩展

- 添加更多医疗数据集支持
- 集成更多ADAPT库中的先进域适应方法
- 添加自动超参数优化
- 支持多GPU并行训练
- 集成模型解释性分析工具（SHAP等）

---

**注意**：本项目遵循中文回复 + 英文图像标签的统一语言输出风格，确保科研标准的图表质量。所有可视化图表中的标签、图例、标题、坐标轴名称均使用英文，符合国际科技写作标准。 