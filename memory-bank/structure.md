# TabPFN 项目结构

## 目前目录结构概览

TabPFN 项目目前有以下主要目录和文件结构：

```
TabPFN
├── .github/                    # GitHub 配置文件
├── data/                       # 数据目录（不包含实际数据）
├── docs/                       # 项目文档
├── examples/                   # 使用示例
├── results/                    # 主要实验结果
│   ├── auto_rfe_analysis/      # 自动化递归特征消除分析
│   ├── autoTabPFN/             # 自动化 TabPFN 实验结果
│   ├── feature_selection/      # 特征选择结果
│   └── ...                     # 其他实验结果
├── results_*/                  # 不同实验设置下的结果目录
│   ├── shap_analysis/          # SHAP 解释性分析
│   └── shapiq_analysis/        # ShapIQ 解释性分析
├── scripts/                    # 实用脚本
├── src/                        # 源代码
│   └── tabpfn/                 # TabPFN 核心库
│       ├── misc/               # 辅助函数和工具
│       └── model/              # 模型定义和实现
└── tests/                      # 测试代码
```

## 标准化后目录结构

为提高项目可维护性，项目已开始重构为以下更标准化的目录结构：

```
TabPFN/
├── jobs/                       # 作业配置和脚本
│   ├── templates/              # 作业模板
│   │   ├── single_dataset.sh   # 单数据集作业模板
│   │   ├── cross_dataset.sh    # 跨数据集作业模板
│   │   └── feature_selection.sh # 特征选择作业模板
│   ├── configs/                # 作业配置
│   │   ├── dataset_A/          # A数据集(AI4health)配置
│   │   ├── dataset_B/          # B数据集(HenanCancerHospital)配置
│   │   ├── dataset_C/          # C数据集(GuangzhouMedicalHospital)配置
│   │   └── cross/             # 跨数据集配置
│   └── submit.py              # 统一作业提交工具
├── logs/                       # 组织化日志存储
│   ├── dataset_A/              # A数据集日志
│   ├── dataset_B/              # B数据集日志
│   ├── dataset_C/              # C数据集日志
│   └── experiments/            # 按实验类型分类日志
├── scripts/                    # 执行脚本
│   ├── predict/                # 预测脚本
│   │   └── predict_cross_dataset.py # 跨数据集预测脚本
│   ├── analyze/                # 分析脚本
│   └── experiment/             # 实验脚本
│       └── run_feature_selection.py # 特征选择脚本
├── src/                        # 源代码
│   ├── tabpfn/                 # 原始TabPFN核心实现
│   │   ├── misc/               # 辅助函数和工具
│   │   └── model/              # 模型定义和实现
│   └── healthcare/             # 医疗模块
│       ├── __init__.py         # 医疗模块初始化
│       ├── data/               # 数据处理模块
│       │   └── __init__.py     # 数据处理初始化
│       ├── models/             # 模型实现
│       │   └── __init__.py     # 模型模块初始化
│       ├── feature_selection/  # 特征选择实现
│       │   └── __init__.py     # 特征选择初始化
│       ├── domain_adaptation/  # 域适应方法
│       │   └── __init__.py     # 域适应模块初始化
│       └── evaluation/         # 评估和可视化
│           └── __init__.py     # 评估模块初始化
├── experiments/                # 实验配置
│   ├── single_dataset/         # 单数据集实验
│   ├── cross_dataset/          # 跨数据集实验
│   └── domain_adaptation/      # 域适应实验
├── results/                    # 统一结果存储（将逐步整合现有results_*）
│   ├── dataset_A/              # A数据集结果
│   ├── dataset_B/              # B数据集结果
│   ├── dataset_C/              # C数据集结果
│   ├── cross_AB_C/             # AB训练C测试结果
│   └── analysis/               # 分析结果
├── memory-bank/                # 项目Memory Bank
└── tests/                      # 测试代码
```

## 关键文件说明

### 核心实现文件

- `src/tabpfn/`: TabPFN 核心实现，包含模型定义和训练逻辑
- `src/healthcare/`: 医疗数据分析模块，包含数据处理、特征选择等功能
- `pyproject.toml`: 项目依赖和配置信息

### 作业提交系统

- `jobs/submit.py`: 统一作业提交工具
- `jobs/templates/`: 作业模板目录，包含不同类型实验的作业脚本
- `jobs/configs/`: 作业配置目录，按数据集和实验类型组织配置文件

### 主要脚本文件

1. **预测和分析**
   - `scripts/predict/predict_cross_dataset.py`: 跨数据集预测脚本
   - `scripts/experiment/run_feature_selection.py`: 特征选择实验脚本

2. **传统脚本文件**（尚未重构）
   - `predict_healthcare.py`: 基础医疗数据分类脚本
   - `predict_healthcare_auto.py`: 自动化医疗数据分类
   - `analyze_healthcare_data.py`: 医疗数据分析工具

3. **自动化特征选择**（尚未重构）
   - `predict_healthcare_auto_external_RFE.py`: 自动化递归特征消除
   - `evaluate_feature_numbers.py`: 评估特征数量对性能的影响

### 结果目录

- `results/`: 标准化结果存储目录（逐步整合）
- `results_*/`: 传统结果目录（将逐步迁移到标准结构） 