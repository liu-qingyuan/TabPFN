# 项目文档目录

此目录包含项目的相关文档和分析报告。

## 目录内容

- [CORAL域适应分析结果说明](./CORAL域适应分析结果说明.md) - 详细介绍了解析版CORAL和类条件CORAL域适应方法的实现及相关实验结果分析。

## 数据集说明

项目使用了三个不同的数据集：

- **Dataset A**: AI4health数据集
- **Dataset B**: HenanCancerHospital数据集
- **Dataset C**: GuangzhouMedicalHospital数据集

## 相关代码文件

主要代码文件包括：

- `predict_healthcare_auto_and_otherbaselines_ABC_features23_analytical_CORAL.py` - 解析版CORAL方法实现
- `visualize_analytical_CORAL_tsne.py` - CORAL可视化工具

# 域适应方法文档

本目录包含项目中实现的各种域适应方法的详细文档：

## 可用文档

1. [CORAL域适应分析结果说明](CORAL域适应分析结果说明.md) - 包含解析版CORAL和类条件CORAL方法的实现原理与结果分析
2. [MMD域适应分析结果说明](MMD域适应分析结果说明.md) - 包含各种MMD变体(线性MMD、核PCA MMD、均值标准差MMD)及类条件MMD的实现原理与结果分析

## 域适应方法概述

域适应是迁移学习的一个重要分支，用于解决源域和目标域之间分布差异的问题。在本项目中，我们实现了几种主要的域适应方法：

1. **CORAL (CORrelation ALignment)** - 通过对齐协方差矩阵减少域差异
2. **MMD (Maximum Mean Discrepancy)** - 通过最小化域之间的分布差异来进行域适应
3. **类条件变体** - 考虑类别信息的域适应方法，对每个类别分别进行域适应

每个文档详细介绍了对应方法的工作原理、实现细节、实验结果分析和性能比较。 