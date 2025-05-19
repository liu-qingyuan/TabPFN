# TabPFN 技术上下文

## 技术栈

TabPFN 项目采用 Python 作为主要开发语言，结合多个数据科学和机器学习库构建完整的表格数据分析解决方案：

- **核心编程语言**: Python
- **数据处理**: NumPy, Pandas
- **机器学习框架**: PyTorch (TabPFN 基础), scikit-learn
- **特征选择**: scikit-learn (RFE)
- **域适应**: 自定义实现的 CORAL, MMD 方法
- **可视化**: Matplotlib, Seaborn
- **可解释性**: SHAP, ShapIQ
- **优化**: Hyperopt

## 依赖项

项目主要依赖以下库和框架：

```
numpy
pandas
torch
scikit-learn
matplotlib
seaborn
shap
hyperopt
```

核心依赖包含在项目根目录的 `pyproject.toml` 文件中，可通过标准 Python 包管理工具安装。

## 开发环境

项目开发环境为 Linux 系统 (Ubuntu 6.5.0-28-generic)，采用以下工具链：

- Python 环境管理: conda
- 版本控制: Git
- 测试框架: pytest
- 代码质量: pre-commit hooks

## 文件结构组织

TabPFN 采用模块化结构组织代码，主要包括：

1. `src/tabpfn/`: 核心模型实现
   - `model/`: 模型定义和实现
   - `misc/`: 辅助工具和功能

2. 数据处理脚本：根目录包含多种医疗数据集处理脚本
   - `predict_healthcare*.py`: 医疗数据分析脚本
   - `analyze_healthcare_data.py`: 数据分析工具

3. 结果输出：项目中包含丰富的结果目录结构
   - `results/`: 主要实验结果
   - `results_*/`: 特定实验配置的结果

4. 测试框架：
   - `tests/`: 单元测试和集成测试
   - `tests/reference_predictions/`: 参考预测结果

## 技术约束

1. **计算资源**：
   - TabPFN 模型需要足够的内存加载预训练模型
   - 预测过程计算密集，但不需要 GPU 加速

2. **数据格式**：
   - 输入数据需要是标准表格格式（CSV 或类似）
   - 特征需要进行预处理（标准化、缺失值处理等）

3. **模型限制**：
   - 默认配置支持最多 1000 个样本和 100 个特征
   - 适用于二分类问题，多分类需要特殊处理

4. **域适应约束**：
   - 不同数据集间需要有共享的特征空间
   - 特征分布差异不应过大，否则需要更强的域适应方法 