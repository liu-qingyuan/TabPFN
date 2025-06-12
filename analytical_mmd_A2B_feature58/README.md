# Analytical MMD Domain Adaptation for Healthcare Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> 基于最大均值差异(MMD)的医疗数据跨域适应分析包，专注于AI4Health到河南癌症医院数据集的域转移学习

## 🌟 项目特色

- 🏥 **医疗数据专用**：针对医疗数据集的跨医院、跨设备域适应
- 🔬 **多种MMD方法**：线性变换、核PCA、均值标准差对齐等
- 🤖 **AutoTabPFN集成**：结合最新的表格数据自动机器学习
- 📊 **丰富可视化**：t-SNE、特征分布、性能对比图表
- 🔧 **模块化设计**：每个组件独立可测试、可配置
- 📈 **完整评估**：交叉验证、外部验证、改进度量
- 🎯 **灵活数据划分**：支持二分法和三分法数据划分策略
- 🚀 **贝叶斯优化**：集成超参数优化和模型选择功能

## 📋 目录

- [快速开始](#快速开始)
- [项目架构](#项目架构)
- [核心算法](#核心算法)
- [数据划分策略](#数据划分策略)
- [使用指南](#使用指南)
- [API文档](#api文档)
- [实验结果](#实验结果)
- [开发指南](#开发指南)
- [问题排查](#问题排查)

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
CUDA >= 11.0 (推荐，用于GPU加速)
```

### 安装依赖

```bash
cd analytical_mmd_A2B_feature58
pip install -r requirements.txt
```

### 基础使用

```python
# 运行完整的跨域实验（二分法）
python scripts/run_analytical_mmd.py --mode cross-domain --model-type auto

# 使用三分法数据划分
python scripts/run_analytical_mmd.py --mode cross-domain --model-type auto --data-split-strategy three-way

# 启用贝叶斯优化
python scripts/run_analytical_mmd.py --mode cross-domain --model-type auto --data-split-strategy three-way --use-bayesian-optimization

# 比较所有MMD方法
python scripts/run_analytical_mmd.py --compare-all

# 使用类条件MMD
python scripts/run_analytical_mmd.py --mode cross-domain --use-class-conditional
```

## 🏗️ 项目架构

### 目录结构

```
analytical_mmd_A2B_feature58/
├── 📁 config/                    # 配置管理
│   └── settings.py              # 全局设置：特征、模型、MMD参数
├── 📁 data/                     # 数据处理
│   └── loader.py               # Excel数据加载器
├── 📁 modeling/                 # 机器学习模型
│   ├── cross_domain_runner.py  # 跨域实验运行器
│   └── model_selector.py       # 多模型选择器
├── 📁 preprocessing/            # 数据预处理
│   ├── mmd.py                  # 核心MMD算法
│   ├── class_conditional_mmd.py # 类条件MMD
│   ├── scaler.py               # 数据标准化
│   └── threshold_optimizer.py  # 决策阈值优化
├── 📁 metrics/                  # 评估指标
│   ├── classification.py       # 分类指标
│   ├── cross_domain_metrics.py # 跨域评估
│   ├── discrepancy.py          # 域差异度量
│   └── evaluation.py           # 通用评估
├── 📁 visualization/            # 可视化
│   ├── tsne_plots.py           # t-SNE降维图
│   ├── histogram_plots.py      # 特征分布图
│   ├── comparison_plots.py     # 性能对比图
│   └── roc_plots.py            # ROC曲线
├── 📁 scripts/                 # 执行脚本
│   └── run_analytical_mmd.py   # 主执行脚本
├── 📁 tests/                   # 测试套件
├── 📁 utils/                   # 工具函数
└── 📁 doc/                     # 详细文档
```

### 核心组件

| 模块 | 功能 | 主要类/函数 |
|------|------|------------|
| **MMD算法** | 域适应核心 | `MMDLinearTransform`, `mmd_transform()` |
| **模型选择** | 多模型支持 | `get_model()`, `AutoTabPFN`, `TunedTabPFN` |
| **跨域运行器** | 实验管理 | `CrossDomainExperimentRunner` |
| **评估指标** | 性能度量 | `evaluate_model_on_external_cv()` |
| **可视化** | 结果展示 | `visualize_tsne()`, `plot_performance_comparison()` |

## 🔬 核心算法

### Maximum Mean Discrepancy (MMD)

MMD是衡量两个概率分布差异的统计度量：

```math
MMD²(P, Q) = ||μ_P - μ_Q||²_H
```

其中 μ_P 和 μ_Q 是分布P和Q在再生核希尔伯特空间H中的均值嵌入。

### 支持的MMD变体

#### 1. 线性MMD变换
- **原理**：学习线性变换矩阵最小化源域和目标域的MMD距离
- **优点**：计算效率高，解释性强
- **配置**：支持分阶段训练、动态gamma、梯度裁剪

```python
# 线性MMD配置示例
mmd_config = {
    'method': 'linear',
    'n_epochs': 200,
    'lr': 3e-4,
    'staged_training': True,
    'dynamic_gamma': True,
    'gamma_search_values': [0.01, 0.05, 0.1]
}
```

#### 2. 核主成分分析MMD
- **原理**：使用核PCA将数据映射到公共特征空间
- **优点**：处理非线性关系
- **参数**：核函数、gamma值、主成分数量

#### 3. 类条件MMD
- **原理**：为每个类别分别进行域适应
- **优点**：保持类别特有的分布特征
- **支持**：伪标签生成、部分监督学习

## 🎯 数据划分策略

### 概述

本项目支持灵活的数据划分策略和贝叶斯优化功能，允许用户根据实验需求选择不同的数据划分方式。

### 支持的划分策略

#### 二分法 (Two-way Split) - 默认
```
A域数据 (训练集) → 完整用于训练
B域数据 (测试集) → 完整用于测试
```
- **适用场景**: 标准域适应评估，与原始方法保持一致
- **优点**: 最大化利用测试数据，结果稳定
- **缺点**: 无法进行模型选择和超参数优化

#### 三分法 (Three-way Split) - 新增
```
A域数据 (训练集) → 完整用于训练
B域数据 → 验证集 (80%) + 保留测试集 (20%)
```
- **适用场景**: 需要超参数优化或模型选择
- **优点**: 支持严格的模型评估，避免数据泄露
- **缺点**: 减少了可用的测试数据

### 贝叶斯优化集成

当使用三分法时，可启用贝叶斯优化：
- **目标**: 在验证集上最大化AUC
- **评估**: 最终在保留测试集上评估泛化能力
- **分析**: 自动计算泛化差距，判断过拟合风险

### 数据划分参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data-split-strategy` | str | `two-way` | 数据划分策略 (`two-way` \| `three-way`) |
| `--validation-split` | float | `0.8` | 三分法时验证集比例 |
| `--use-bayesian-optimization` | flag | `False` | 启用贝叶斯优化 |
| `--bo-n-calls` | int | `50` | 优化迭代次数 |
| `--bo-random-state` | int | `42` | 随机种子 |

### 输出结果结构

**二分法输出**:
```
results_cross_domain_{model_type}_{method}_{feature_type}_two-way/
├── results.json                 # 实验结果
├── experiment_config.txt        # 实验配置
└── visualizations/             # 可视化结果
```

**三分法输出**:
```
results_cross_domain_{model_type}_{method}_{feature_type}_three-way_val80/
├── results.json                 # 包含验证集和保留测试集结果
├── experiment_config.txt        # 实验配置
└── visualizations/             # 可视化结果
```

**贝叶斯优化输出**:
```
results_bayesian_optimization_{model_type}_{feature_type}/
├── optimization_history.json   # 优化历史
├── final_evaluation.json       # 最终评估结果
├── confusion_matrices.png      # 混淆矩阵对比
└── bayesian_optimization.log   # 详细日志
```

## 📖 使用指南

### 数据划分策略使用

#### 基本命令

```bash
# 二分法（默认）
python scripts/run_analytical_mmd.py --model-type auto --method linear

# 三分法
python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way

# 三分法 + 贝叶斯优化
python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way --use-bayesian-optimization
```

#### 高级配置

```bash
# 自定义验证集比例
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --data-split-strategy three-way \
    --validation-split 0.7

# 贝叶斯优化参数调整
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --data-split-strategy three-way \
    --use-bayesian-optimization \
    --bo-n-calls 100 \
    --bo-random-state 42

# 完整配置示例
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --feature-type best7 \
    --method linear \
    --data-split-strategy three-way \
    --validation-split 0.8 \
    --use-bayesian-optimization \
    --bo-n-calls 50 \
    --use-class-conditional \
    --skip-cv-on-a
```

### 传统命令行界面

```bash
# 基础跨域实验
python scripts/run_analytical_mmd.py \
    --mode cross-domain \
    --model-type auto \
    --method linear \
    --feature-type best7

# 比较所有方法
python scripts/run_analytical_mmd.py --compare-all

# 使用类条件MMD
python scripts/run_analytical_mmd.py \
    --mode cross-domain \
    --use-class-conditional \
    --method kpca

# 阈值优化
python scripts/run_analytical_mmd.py \
    --mode cross-domain \
    --use-threshold-optimizer

# 自定义输出目录
python scripts/run_analytical_mmd.py \
    --mode cross-domain \
    --output-dir ./my_results
```

### 最佳实践

#### 选择数据划分策略
- **研究对比**: 使用二分法与文献结果对比
- **模型开发**: 使用三分法进行超参数优化
- **生产部署**: 使用三分法评估泛化能力

#### 验证集比例设置
- **0.8 (推荐)**: 平衡验证集大小和测试集代表性
- **0.7**: 更大的保留测试集，更可靠的泛化评估
- **0.9**: 更大的验证集，更稳定的优化过程

#### 贝叶斯优化配置
- **快速测试**: `--bo-n-calls 20`
- **标准优化**: `--bo-n-calls 50`
- **深度优化**: `--bo-n-calls 100`

### 性能分析

#### 泛化能力评估
三分法模式下自动计算：
- **验证集AUC**: 模型选择依据
- **保留测试集AUC**: 泛化能力指标
- **泛化差距**: 验证集AUC - 保留测试集AUC

**判断标准**:
- 差距 < 0.05: ✅ 泛化能力良好
- 差距 ≥ 0.05: ⚠️ 可能存在过拟合

#### 优化收敛分析
贝叶斯优化提供：
- 优化历史曲线
- 参数重要性分析
- 收敛诊断信息

### 编程接口

```python
from analytical_mmd_A2B_feature58.modeling.cross_domain_runner import run_cross_domain_experiment

# 运行跨域实验（二分法）
results = run_cross_domain_experiment(
    model_type='auto',
    feature_type='best7',
    mmd_method='linear',
    use_class_conditional=False,
    save_path='./results_custom'
)

# 运行跨域实验（三分法）
results = run_cross_domain_experiment(
    model_type='auto',
    feature_type='best7',
    mmd_method='linear',
    data_split_strategy='three-way',
    validation_split=0.8,
    save_path='./results_custom'
)

# 访问结果
print(f"交叉验证AUC: {results['cross_validation_A']['auc']}")
print(f"外部验证AUC: {results['external_validation_B']['without_domain_adaptation']['auc']}")
```

### 自定义配置

```python
# 修改MMD参数
from analytical_mmd_A2B_feature58.config.settings import MMD_METHODS

MMD_METHODS['linear'].update({
    'n_epochs': 300,
    'lr': 1e-4,
    'lambda_reg': 1e-2
})

# 修改模型参数
from analytical_mmd_A2B_feature58.config.settings import MODEL_CONFIGS

MODEL_CONFIGS['auto'].update({
    'max_time': 60,
    'phe_init_args': {
        'max_models': 20,
        'n_repeats': 150
    }
})
```

## 📊 实验结果

### 典型性能表现

| 数据集转换 | 无域适应AUC | 线性MMD AUC | 改进幅度 |
|-----------|------------|-------------|---------|
| A→B       | 0.660      | 0.692       | +3.2%   |
| A→C       | 0.634      | 0.671       | +3.7%   |

### 结果解释

- **交叉验证结果**：在源域(数据集A)上的10折交叉验证性能
- **外部验证结果**：在目标域(数据集B/C)上的评估性能
- **域适应改进**：使用MMD后的性能提升

## 🧪 测试套件

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_mmd_basic.py
pytest tests/test_statistics_consistency.py

# 测试数据划分策略功能
python tests/test_data_split_strategies.py

# 生成覆盖率报告
pytest --cov=analytical_mmd_A2B_feature58
```

### 测试类别

- **基础功能测试**：MMD计算、数据加载
- **统计一致性测试**：确保不同实现的结果一致
- **数据划分策略测试**：验证二分法、三分法和贝叶斯优化功能
- **可视化测试**：验证图表生成功能
- **集成测试**：端到端实验流程

## 🔧 开发指南

### 添加新的MMD方法

1. 在 `preprocessing/mmd.py` 中实现新方法
2. 在 `config/settings.py` 中添加配置
3. 编写对应的测试用例
4. 更新文档

### 代码风格

```bash
# 代码格式化
black analytical_mmd_A2B_feature58/

# 类型检查
mypy analytical_mmd_A2B_feature58/

# 代码质量检查
flake8 analytical_mmd_A2B_feature58/
```

### 版本控制

遵循语义化版本控制：
- 主版本号：不兼容的API更改
- 次版本号：向后兼容的功能增加
- 修订号：向后兼容的bug修复

## 🐛 问题排查

### 常见问题

**Q: ModuleNotFoundError: No module named 'tabpfn_extensions'**
```bash
pip install tabpfn-extensions
```

**Q: CUDA out of memory**
```python
# 减少批次大小
MMD_METHODS['linear']['batch_size'] = 32
```

**Q: 收敛问题**
```python
# 调整学习率和正则化
MMD_METHODS['linear'].update({
    'lr': 1e-4,
    'lambda_reg': 1e-2
})
```

### 调试模式

   ```bash
# 启用详细日志
python scripts/run_analytical_mmd.py --mode cross-domain --debug

# 保存中间结果
python scripts/run_analytical_mmd.py --mode cross-domain --save-intermediate
```

## 📚 文档索引

- [配置文档](doc/config.md) - 详细配置参数说明
- [MMD算法文档](doc/mmd_algorithms.md) - MMD方法技术细节  
- [模型文档](doc/models.md) - 支持的机器学习模型
- [API参考](doc/api_reference.md) - 完整API文档
- [实验指南](doc/experiments.md) - 实验设计和执行
- [可视化指南](doc/visualization.md) - 图表生成和解释

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献指南

欢迎提交Issue和Pull Request！请遵循以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📞 联系方式

- **项目维护者**：TabPFN团队
- **邮箱**：support@tabpfn.com
- **文档**：[在线文档](https://docs.tabpfn.com)

---

*最后更新：2025年5月29日* 

## ⚠️ 注意事项

### 数据划分策略约束
1. 贝叶斯优化只能在三分法模式下使用
2. 验证集比例必须在 (0, 1) 范围内
3. B域数据必须足够进行有意义的划分

### 常见错误
   ```bash
# ❌ 错误：在二分法下使用贝叶斯优化
python scripts/run_analytical_mmd.py --use-bayesian-optimization

# ✅ 正确：在三分法下使用贝叶斯优化
python scripts/run_analytical_mmd.py --data-split-strategy three-way --use-bayesian-optimization
```

### 其他常见问题

**Q: ModuleNotFoundError: No module named 'tabpfn_extensions'**
   ```bash
pip install tabpfn-extensions
```

**Q: CUDA out of memory**
```python
# 减少批次大小
MMD_METHODS['linear']['batch_size'] = 32
```

**Q: 收敛问题**
```python
# 调整学习率和正则化
MMD_METHODS['linear'].update({
    'lr': 1e-4,
    'lambda_reg': 1e-2
})
```

### 调试模式

```bash
# 启用详细日志
python scripts/run_analytical_mmd.py --mode cross-domain --debug

# 保存中间结果
python scripts/run_analytical_mmd.py --mode cross-domain --save-intermediate
```