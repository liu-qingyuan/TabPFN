# UDA Medical Imbalance Analysis Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 专注于医疗数据不平衡问题与无监督域适应（UDA）的综合性机器学习实验项目，基于ADAPT库实现多种域适应算法，提供完整的可视化分析和性能评估框架

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

## 📋 目录

- [快速开始](#快速开始)
- [项目架构](#项目架构)
- [完整分析流程](#完整分析流程)
- [核心功能](#核心功能)
- [UDA可视化分析](#uda可视化分析)
- [ADAPT库UDA方法详解](#adapt库uda方法详解)
- [使用指南](#使用指南)
- [实验配置](#实验配置)
- [开发指南](#开发指南)

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

### 快速使用示例

```python
# 基础UDA方法使用
from uda.adapt_methods import create_adapt_method
from sklearn.linear_model import LogisticRegression
import numpy as np

# 创建KMM域适应方法
kmm_method = create_adapt_method(
    method_name='KMM',
    estimator=LogisticRegression(penalty="none"),
    kernel='rbf',
    gamma=1.0,
    verbose=0,
    random_state=42
)

# 拟合模型
kmm_method.fit(X_source, y_source, X_target)

# 预测目标域
y_pred = kmm_method.predict(X_target)
y_proba = kmm_method.predict_proba(X_target)

# 评估性能
accuracy = kmm_method.score(X_target, y_target)
```

```python
# UDA完整可视化分析
from preprocessing.uda_visualizer import UDAVisualizer
from preprocessing.uda_processor import UDAProcessor

# 创建UDA处理器和可视化器
processor = UDAProcessor()
visualizer = UDAVisualizer(save_plots=True, output_dir="results/uda_analysis")

# 预处理数据
X_source_processed, y_source, X_target_processed, y_target = processor.process_datasets(
    X_source, y_source, X_target, y_target,
    feature_count=8,
    scaler_type='standard',
    imbalance_method='smote'
)

# 创建并拟合UDA方法
uda_method = create_adapt_method('TCA', estimator=LogisticRegression(penalty="none"))
uda_method.fit(X_source_processed, y_source, X_target_processed)

# 生成完整可视化分析
results = visualizer.visualize_domain_adaptation_complete(
    X_source_processed, y_source,
    X_target_processed, y_target,
    uda_method=uda_method,
    method_name="TCA"
)
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

### 预期输出

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

## 🚀 完整分析流程

### CompleteAnalysisRunner 执行流程图

上图展示了 **CompleteAnalysisRunner** 的完整执行流程，包含六个主要步骤：

1. **数据加载与预处理** - 双重加载策略确保兼容性
2. **源域交叉验证** - TabPFN vs 传统基线 vs 机器学习基线
3. **UDA方法对比** - 无UDA基线 vs UDA方法 vs 其他基线
4. **可视化生成** - ROC曲线、校准曲线、决策曲线、雷达图
5. **报告生成** - 自动识别最佳方法和域适应效果
6. **结果保存** - 结构化输出所有结果和可视化

上图展示了项目的分层技术架构，**CompleteAnalysisRunner** 作为核心引擎协调各层组件：

- **数据层**：管理多医院医疗数据集和特征集
- **预处理层**：数据加载、特征选择、标准化、不平衡处理
- **模型层**：TabPFN、传统基线、机器学习基线、UDA方法
- **评估层**：交叉验证、性能指标、域适应评估、预测数据收集
- **可视化层**：ROC曲线、校准曲线、决策曲线、雷达图、UDA专业可视化
- **输出层**：结构化保存JSON结果、Markdown报告、PNG图表

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
│   ├── run_full_uda_experiment.py    # 完整实验流程
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

## 🔬 核心功能

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

## 🎨 UDA可视化分析

### 2.1 专业UDA可视化器

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

### 2.2 可视化功能详解

#### 2.2.1 降维可视化
- **PCA可视化**：主成分分析展示域间分布
- **t-SNE可视化**：非线性降维展示聚类效果
- **域适应前后对比**：直观展示UDA效果

#### 2.2.2 标准化距离度量
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

#### 2.2.3 智能特征处理
- **维度兼容性检查**：自动处理特征维度变化的UDA方法
- **特征变换策略**：
  - TCA/SA/FMMD：可能改变特征维度，特殊处理
  - CORAL：协方差对齐，维度不变
  - KMM/KLIEP：实例重加权，不改变特征
- **回退机制**：维度不匹配时自动使用原始特征空间进行距离计算

#### 2.2.4 性能对比可视化
- **基线对比**：UDA方法 vs 无域适应基线
- **多指标展示**：准确率、AUC、F1、精确率、召回率
- **改进程度量化**：域适应带来的性能提升

### 2.3 可视化输出

每次分析生成以下可视化文件：
```
results/method_name_YYYYMMDD_HHMMSS/
├── Method_RealData_dimensionality_reduction.png  # 降维可视化
├── Method_RealData_distance_metrics.png          # 距离度量对比
└── Method_RealData_performance_comparison.png    # 性能对比
```

## 🔄 ADAPT库UDA方法详解

### 3.1 实例重加权方法 (Instance-Based)

**KMM (Kernel Mean Matching)**
```python
# 使用示例
from uda.adapt_methods import create_adapt_method

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

### 3.2 特征对齐方法 (Feature-Based)

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

### 3.3 深度学习方法 (Deep Learning)

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

## 🔧 使用指南

### CompleteAnalysisRunner - 核心分析引擎

**CompleteAnalysisRunner** 是项目的核心分析引擎，提供端到端的自动化分析流程：

```python
from scripts.run_complete_analysis import CompleteAnalysisRunner

# 创建分析运行器
runner = CompleteAnalysisRunner(
    feature_set='best8',           # 特征集选择：'best7'|'best8'|'best9'|'best10'|'all'
    scaler_type='none',            # 标准化方法：'standard'|'robust'|'none'
    imbalance_method='none',       # 不平衡处理：'smote'|'borderline_smote'|'adasyn'|'none'
    cv_folds=10,                   # 交叉验证折数：推荐10折
    random_state=42,               # 随机种子：确保结果可重复
    output_dir=None,               # 输出目录：None时自动生成时间戳目录
    verbose=True                   # 详细输出：显示执行过程
)

# 运行完整分析
results = runner.run_complete_analysis()

# 查看结果
print(f"分析完成！结果保存在: {runner.output_dir}")
```

### 配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `feature_set` | str | 'best8' | 特征集选择，影响TabPFN和机器学习基线的特征数量 |
| `scaler_type` | str | 'none' | 标准化方法，只影响TabPFN和UDA方法 |
| `imbalance_method` | str | 'none' | 不平衡处理方法，只影响TabPFN和UDA方法 |
| `cv_folds` | int | 10 | 交叉验证折数，影响所有CV分析 |
| `random_state` | int | 42 | 随机种子，确保结果可重复性 |
| `output_dir` | Optional[str] | None | 输出目录，None时自动生成 |
| `verbose` | bool | True | 是否显示详细执行信息 |

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

### 分析流程详解

CompleteAnalysisRunner执行以下六个主要步骤：

#### 1. 数据加载与预处理
- **双重数据加载策略**：
  - CV分析：使用selected58特征集（兼容所有基线模型）
  - UDA分析：使用指定特征集（best8等）+ 预处理
- **智能特征对齐**：确保源域和目标域特征一致性
- **自动回退机制**：数据加载失败时自动尝试备选特征集

#### 2. 源域10折交叉验证
- **TabPFN方法**：使用指定特征集 + 预处理配置
- **传统基线**：PKUPH、Mayo、Paper_LR（使用selected58特征集）
- **机器学习基线**：SVM、DT、RF、GBDT、XGBoost（使用相同配置）
- **性能指标**：AUC、准确率、F1、精确率、召回率

#### 3. UDA方法对比分析
- **基线对比**：TabPFN无UDA vs 传统基线 vs 机器学习基线
- **UDA方法**：TCA、SA、CORAL、KMM等ADAPT库方法
- **目标域测试**：所有方法在目标域B上评估
- **预测数据收集**：保存用于ROC曲线和校准曲线分析

#### 4. 专业可视化生成
- **ROC曲线对比**：源域CV vs UDA方法性能对比
- **校准曲线分析**：模型预测概率校准效果
- **决策曲线分析**：临床决策价值评估
- **性能雷达图**：多维度性能指标可视化

#### 5. 分析报告生成
- **Markdown格式报告**：包含所有性能指标和结论
- **最佳方法识别**：自动识别各类别中的最佳方法
- **域适应效果评估**：量化UDA方法的改进程度
- **失败方法记录**：记录失败方法及错误原因

#### 6. 结果保存与输出
- **JSON结果文件**：完整的实验结果和配置
- **可视化图表**：PNG格式的专业图表
- **分析报告**：详细的Markdown分析报告
- **目录结构化**：按时间戳组织的结果目录

### 完整UDA实验流程

```python
from preprocessing.uda_processor import UDAProcessor
from preprocessing.uda_visualizer import UDAVisualizer
from uda.adapt_methods import create_adapt_method
from sklearn.linear_model import LogisticRegression

# 1. 数据预处理
processor = UDAProcessor()
X_source_processed, y_source, X_target_processed, y_target = processor.process_datasets(
    X_source, y_source, X_target, y_target,
    feature_count=8,
    scaler_type='standard',
    imbalance_method='smote'
)

# 2. 创建UDA方法
uda_methods = ['TCA', 'SA', 'CORAL', 'KMM']
results = {}

for method_name in uda_methods:
    print(f"\n--- 测试方法: {method_name} ---")
    
    # 创建方法
    uda_method = create_adapt_method(
        method_name=method_name,
        estimator=LogisticRegression(penalty="none"),
        random_state=42
    )
    
    # 拟合方法
    uda_method.fit(X_source_processed, y_source, X_target_processed)
    
    # 评估性能
    accuracy = uda_method.score(X_target_processed, y_target)
    results[method_name] = accuracy
    
    # 生成可视化分析
    visualizer = UDAVisualizer(save_plots=True, output_dir=f"results/{method_name}_analysis")
    visualization_results = visualizer.visualize_domain_adaptation_complete(
        X_source_processed, y_source,
        X_target_processed, y_target,
        uda_method=uda_method,
        method_name=method_name
    )

# 3. 输出对比结果
print("\n📊 UDA方法性能对比:")
for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {method}: {acc:.4f}")
```

### 自定义距离度量分析

```python
from preprocessing.uda_visualizer import UDAVisualizer

# 创建可视化器
visualizer = UDAVisualizer(save_plots=True)

# 计算域距离度量
distances = visualizer.calculate_domain_distances(
    X_source, X_target,
    uda_method=uda_method,
    method_name="TCA"
)

# 查看标准化距离指标
print("标准化距离指标:")
for metric in ['normalized_linear_discrepancy', 'normalized_frechet_distance', 
               'normalized_wasserstein', 'normalized_kl_divergence']:
    if f'{metric}_improvement' in distances:
        improvement = distances[f'{metric}_improvement']
        print(f"  {metric}: {improvement:.6f}")
```

## 📊 实验配置

### 配置文件示例

```yaml
# configs/experiment_config.yaml
experiment:
  name: "medical_uda_experiment"
  description: "基于ADAPT库的UDA方法在医疗数据上的对比实验"
  
preprocessing:
  feature_set: "best8"         # 特征集选择: best7|best8|best9|best10|all
  scaler: "standard"           # 标准化方法: standard|robust|none
  imbalance_method: "smote"    # 不平衡处理: none|smote|smotenc|borderline_smote|kmeans_smote|svm_smote|adasyn|smote_tomek|smote_enn|random_under
  force_resampling: false      # 是否强制执行重采样（测试模式）

source_domain:
  cv_folds: 10
  models:
    - "tabpfn"
    - "paper_method"
    - "pkuph_baseline"
    - "mayo_baseline"

uda_methods:
  instance_based:
    - method: "KMM"
      params:
        kernel: "rbf"
        gamma: 1.0
    - method: "KLIEP"
      params:
        gamma: 1.0
        
  feature_based:
    - method: "CORAL"
      params:
        lambda_: 1.0
    - method: "SA"
      params:
        n_components: null
    - method: "TCA"
      params:
        n_components: 6
        mu: 0.1
        kernel: "linear"
    - method: "FMMD"
      params:
        gamma: 1.0

evaluation:
  metrics:
    - "auc"          # 主要指标
    - "accuracy"
    - "f1"
    - "precision"
    - "recall"
  comparison_baseline: true    # 是否包含无DA的基线对比
  
visualization:
  enable_pca: true
  enable_tsne: true
  enable_distance_metrics: true
  skip_feature_distributions: true  # 跳过特征分布可视化
  save_individual_plots: true
  plot_format: "png"
  
output:
  results_dir: "results"
  save_preprocessed_data: true
  save_model_predictions: true
  generate_html_report: true
```

## 🧪 测试和验证

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

### 测试ADAPT方法功能

```bash
# 测试ADAPT库方法的可用性和基本功能
cd uda_medical_imbalance_project
python -c "
from uda.adapt_methods import is_adapt_available, get_available_adapt_methods
print('ADAPT库可用:', is_adapt_available())
if is_adapt_available():
    methods = get_available_adapt_methods()
    print('支持的方法数量:', len(methods))
    for method, info in methods.items():
        print(f'  {method}: {info['description']}')
"
```

## 📝 最新更新

### v2.0 主要更新 (2024-12)

#### 🎨 UDA可视化分析器重大改进
- **修复距离度量计算**：解决了维度变化导致的指标异常问题
- **标准化指标优先**：只显示标准化版本的距离度量，确保跨方法可比性
- **智能特征处理**：自动处理不同UDA方法的特征变换策略
- **跳过特征分布可视化**：避免特征尺度差异导致的可视化问题

#### 🔧 技术改进
- **维度兼容性检查**：自动检测并处理特征维度变化
- **回退机制**：维度不匹配时使用原始特征空间计算距离
- **稳定性提升**：移除不稳定的非标准化指标显示

#### 📊 距离度量优化
- **标准化线性差异**：基于ADAPT库的专业实现
- **标准化Frechet距离**：提供稳定的分布距离度量
- **标准化Wasserstein距离**：自定义实现，遵循ADAPT库标准
- **标准化KL散度**：改进的散度计算方法

## 📄 实验结果输出

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

### 分析报告示例

```markdown
# 完整医疗数据UDA分析报告

## 源域10折交叉验证结果
| 方法 | AUC | Accuracy | F1 | Precision | Recall |
|------|-----|----------|----|-----------| -------|
| TabPFN | 0.8456 | 0.7892 | 0.7654 | 0.7834 | 0.7481 |
| Paper_LR | 0.8234 | 0.7654 | 0.7321 | 0.7456 | 0.7189 |

## UDA方法对比结果
| 方法 | AUC | Accuracy | F1 | Precision | Recall | 类型 |
|------|-----|----------|----|-----------| -------|------|
| TabPFN_NoUDA | 0.7892 | 0.7234 | 0.6987 | 0.7123 | 0.6854 | TabPFN基线 |
| TCA | 0.8123 | 0.7456 | 0.7234 | 0.7345 | 0.7125 | UDA方法 |
| CORAL | 0.8045 | 0.7389 | 0.7156 | 0.7267 | 0.7048 | UDA方法 |

## 结论和建议
- **最佳源域方法**: TabPFN (AUC: 0.8456)
- **TabPFN无UDA基线**: TabPFN_NoUDA (AUC: 0.7892)
- **最佳UDA方法**: TCA (AUC: 0.8123)
- **域适应效果**: TCA相比TabPFN无UDA基线提升了 0.0231 AUC
```

## 🤝 贡献指南

1. Fork 本项目
2. 创建功能分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢 [ADAPT库](https://adapt-python.github.io/adapt/) 提供的完整域适应算法实现
- 感谢 TabPFN 团队提供的自动表格学习框架
- 感谢医疗数据提供方的数据支持 