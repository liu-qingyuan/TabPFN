# 完整医疗数据UDA分析流程 - 快速开始指南

## 🚀 快速开始

### 1. 基本使用

```bash
# 使用默认配置运行完整分析
python scripts/run_complete_analysis.py
```

### 2. 配置文件使用

```bash
# 使用配置文件运行分析
python scripts/run_configurable_analysis.py --config configs/complete_analysis_config.yaml

# 使用命令行参数覆盖配置
python scripts/run_configurable_analysis.py --feature-set best7 --imbalance-method borderline_smote
```

### 3. 测试流程

```bash
# 测试完整分析流程
python scripts/test_complete_flow.py

# 运行使用示例
python examples/complete_analysis_example.py
```

## 📋 分析流程说明

### 完整分析包含以下步骤：

1. **数据加载**
   - 加载医疗数据集A (源域) 和B (目标域)
   - 特征对齐和基本统计

2. **源域10折交叉验证**
   - TabPFN模型
   - 论文方法 (Paper LR)
   - 基线模型 (PKUPH, Mayo)
   - 性能指标对比

3. **UDA域适应方法**
   - 实例重加权方法: KMM, KLIEP
   - 特征对齐方法: CORAL, SA, TCA, FMMD
   - 基于ADAPT库的专业实现

4. **可视化分析**
   - 源域CV结果对比图
   - UDA方法性能对比图
   - 综合对比图 (最佳源域 vs 最佳UDA)

5. **结果报告**
   - Markdown格式分析报告
   - JSON格式详细结果
   - 配置文件备份

## ⚙️ 配置参数

### 主要配置项：

- **feature_set**: 特征集选择
  - `best7`: 7个最佳特征
  - `best8`: 8个最佳特征 (推荐)
  - `best9`: 9个最佳特征
  - `best10`: 10个最佳特征

- **scaler_type**: 标准化方法
  - `standard`: 标准化 (推荐)
  - `robust`: 鲁棒标准化
  - `none`: 无标准化

- **imbalance_method**: 不平衡处理
  - `smote`: SMOTE (推荐)
  - `borderline_smote`: BorderlineSMOTE
  - `adasyn`: ADASYN
  - `none`: 无处理

- **cv_folds**: 交叉验证折数
  - 推荐值: 10 (完整分析), 3-5 (快速测试)

## 📊 输出结果

### 生成的文件结构：
```
results/complete_analysis_YYYYMMDD_HHMMSS/
├── analysis_report.md              # 分析报告
├── complete_results.json           # 完整结果
├── source_domain_cv_results.json   # 源域CV结果
├── uda_methods_results.json        # UDA方法结果
├── used_config.yaml                # 使用的配置
├── source_cv_comparison.png        # 源域对比图
├── uda_methods_comparison.png      # UDA方法对比图
├── overall_comparison.png          # 综合对比图
└── uda_*/                          # 各UDA方法详细结果
    ├── *_distance_metrics.png      # 距离度量图
    ├── *_dimensionality_reduction.png  # 降维可视化
    └── visualization_results.json  # 可视化结果
```

## 🔧 自定义配置

### 创建自定义配置文件：

```yaml
# configs/my_config.yaml
experiment:
  name: "my_medical_uda_analysis"
  random_state: 42
  verbose: true

preprocessing:
  feature_set: "best8"
  scaler: "standard"
  imbalance_method: "smote"

source_domain:
  cv_folds: 10
  models:
    - "tabpfn"
    - "paper_method"
    - "pkuph_baseline"
    - "mayo_baseline"

uda_methods:
  feature_based:
    - method: "TCA"
      params:
        n_components: 6
        mu: 0.1
        kernel: "linear"
    - method: "CORAL"
      params:
        lambda_: 1.0
```

### 使用自定义配置：

```bash
python scripts/run_configurable_analysis.py --config configs/my_config.yaml
```

## 🧪 测试和验证

### 1. 环境测试
```bash
# 测试基本功能
python scripts/test_complete_flow.py
```

### 2. 单步测试
```python
from scripts.run_complete_analysis import CompleteAnalysisRunner

# 创建运行器
runner = CompleteAnalysisRunner(
    feature_set='best8',
    cv_folds=3,  # 快速测试
    verbose=True
)

# 分步运行
X_source, y_source, X_target, y_target, features = runner.load_data()
cv_results = runner.run_source_domain_cv(X_source, y_source)
```

### 3. 示例运行
```bash
# 运行所有示例
python examples/complete_analysis_example.py
```

## 📈 性能优化建议

### 快速测试配置：
- `cv_folds: 3` (减少交叉验证折数)
- `feature_set: "best7"` (减少特征数量)
- 选择部分UDA方法测试

### 完整分析配置：
- `cv_folds: 10` (完整交叉验证)
- `feature_set: "best8"` (平衡性能和计算量)
- 测试所有可用的UDA方法

## ❓ 常见问题

### Q1: ADAPT库不可用怎么办？
```bash
# 安装ADAPT库
pip install adapt-python
```

### Q2: 内存不足怎么办？
- 减少 `cv_folds` 数量
- 使用较少的特征集
- 选择部分UDA方法测试

### Q3: 如何只运行源域分析？
```python
runner = CompleteAnalysisRunner(...)
cv_results = runner.run_source_domain_cv(X_source, y_source)
```

### Q4: 如何添加新的UDA方法？
修改配置文件中的 `uda_methods` 部分，或在代码中扩展 `run_uda_methods` 方法。

## 📚 更多信息

- 详细文档: `README.md`
- 代码示例: `examples/`
- 配置模板: `configs/`
- 测试脚本: `scripts/test_*.py` 