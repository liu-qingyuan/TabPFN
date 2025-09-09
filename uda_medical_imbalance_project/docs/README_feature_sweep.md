# Feature Sweep Analysis Script

## 概述

`run_feature_sweep_analysis.py` 是一个自动化脚本，用于评估从 best3 到 best58 的所有特征组合的性能表现。该脚本能够：

1. **并行执行分析** - 支持多进程并行处理，大幅提升分析效率
2. **生成性能对比图表** - 类似现有的 `performance_comparison.png` 格式
3. **自动结果汇总** - 生成详细的性能报告和建议
4. **结构化输出** - 所有结果保存到统一目录便于查看

## 功能特性

### 核心功能
- **特征范围扫描**: 支持从 best3 到 best58 的任意范围
- **并行处理**: 自动检测系统资源，支持多进程并行执行
- **完整分析流程**: 包含源域交叉验证、UDA方法对比、可视化生成
- **智能容错**: 单个特征集失败不影响整体分析进度

### 可视化输出
1. **性能对比曲线图** - AUC、准确率随特征数量变化趋势
2. **TCA改进效果图** - 域适应相对于基线的提升效果
3. **性能热力图** - 不同方法在不同特征集上的表现
4. **组合分析图表** - 多子图综合展示所有结果

### 报告功能
- **Markdown格式报告** - 包含最佳性能、趋势分析、详细结果表格
- **CSV/JSON数据导出** - 便于进一步分析和处理
- **失败分析** - 记录失败的特征集和原因
- **性能建议** - 基于结果自动生成优化建议

## 使用方法

### 基本用法

```bash
# 默认运行 (best3 到 best58)
python scripts/run_feature_sweep_analysis.py

# 指定特征范围
python scripts/run_feature_sweep_analysis.py --min_features 3 --max_features 20

# 设置并行工作数
python scripts/run_feature_sweep_analysis.py --max_workers 4

# 静默模式
python scripts/run_feature_sweep_analysis.py --quiet

# 指定输出目录
python scripts/run_feature_sweep_analysis.py --output_dir results/my_sweep_analysis
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--min_features` | 3 | 最小特征数量 |
| `--max_features` | 58 | 最大特征数量 |
| `--max_workers` | auto | 最大并行工作进程数 |
| `--output_dir` | auto-generated | 输出目录路径 |
| `--quiet` | False | 静默模式，减少输出信息 |

### 高级用法

```bash
# 快速测试 (只分析少数几个特征集)
python scripts/run_feature_sweep_analysis.py --min_features 3 --max_features 8 --max_workers 2

# 完整分析 (推荐用于正式实验)
python scripts/run_feature_sweep_analysis.py --min_features 3 --max_features 32 --max_workers 4

# 单线程调试模式
python scripts/run_feature_sweep_analysis.py --min_features 5 --max_features 10 --max_workers 1
```

## 输出结构

```
results/feature_sweep_analysis_YYYYMMDD_HHMMSS/
├── individual_results/           # 每个特征集的详细结果
│   ├── best3/
│   │   ├── complete_results.json
│   │   ├── analysis_report.md
│   │   └── uda_TCA/             # UDA方法可视化
│   ├── best4/
│   └── ...
├── performance_comparison.png    # 主要性能对比图表
├── feature_sweep_report.md      # 总结报告
├── performance_summary.csv      # 性能数据表格
├── performance_summary.json     # 性能数据JSON格式
├── sweep_results.json          # 原始结果数据
└── feature_sweep.log           # 执行日志
```

## 结果解读

### 性能对比图表

生成的 `performance_comparison.png` 包含四个子图：

1. **AUC Performance vs Number of Features** (左上)
   - 蓝线：源域10折交叉验证结果
   - 红线：目标域基线 (无UDA)
   - 绿线：目标域TCA结果 (有UDA)

2. **TCA Domain Adaptation Improvement** (右上)
   - 条形图显示TCA相对于基线的AUC提升
   - 绿色：正向提升，红色：负向影响

3. **Accuracy Performance Comparison** (左下)
   - 准确率对比分析

4. **Performance Heatmap** (右下)
   - 热力图显示不同方法的性能分布

### 报告内容

`feature_sweep_report.md` 包含：

- **Best Performance Results**: 各项指标的最佳特征集
- **Performance Trends**: 性能趋势分析
- **Detailed Results**: 完整的结果对比表格
- **Recommendations**: 基于实验结果的建议

### 关键指标说明

| 指标 | 说明 |
|------|------|
| Source AUC | 源域10折交叉验证AUC |
| Target Baseline AUC | 目标域无UDA基线AUC |
| Target TCA AUC | 目标域使用TCA的AUC |
| TCA Improvement | TCA相对于基线的提升 |

## 性能和资源

### 执行时间估算

- **单个特征集**: 约5-10分钟 (取决于硬件配置)
- **完整扫描 (best3-best32)**: 约2-4小时 (4并发)
- **大范围扫描 (best3-best58)**: 约4-8小时 (4并发)

### 资源要求

- **内存**: 建议8GB以上
- **CPU**: 支持多核并行，建议4核以上
- **存储**: 每次完整扫描约1-2GB输出文件
- **GPU**: 可选，TabPFN支持GPU加速

### 优化建议

1. **合理设置并行数**: 通常设为CPU核心数的一半
2. **分段执行**: 对于大范围扫描，可以分段执行避免资源耗尽
3. **磁盘空间**: 确保有足够空间保存所有结果文件

## 故障排除

### 常见问题

1. **导入错误**: 确保在项目根目录运行脚本
2. **内存不足**: 减少 `--max_workers` 参数值
3. **TabPFN限制**: 某些特征集可能超出TabPFN预训练限制
4. **依赖缺失**: 确保安装了所有必需的Python包

### 调试模式

```bash
# 单线程调试模式
python scripts/run_feature_sweep_analysis.py --max_workers 1 --min_features 3 --max_features 5

# 检查日志
tail -f results/feature_sweep_analysis_*/feature_sweep.log
```

## 示例结果

基于历史数据的典型结果模式：

- **最佳源域性能**: best8-best10 特征集通常表现最佳 (AUC ~0.83)
- **目标域适应**: TCA通常能带来0.01-0.03的AUC提升
- **性能平衡点**: best8 特征集往往在复杂性和性能间达到最佳平衡

## 扩展使用

### 集成到实验流程

```bash
# 1. 特征扫描分析
python scripts/run_feature_sweep_analysis.py --min_features 3 --max_features 20

# 2. 根据结果选择最佳特征集 (例如 best8)
python scripts/run_complete_analysis.py --feature_type best8

# 3. 进行深度分析和论文图表生成
python examples/real_data_visualization.py --feature_set best8
```

### 自动化脚本

可以创建shell脚本自动化整个流程：

```bash
#!/bin/bash
# auto_analysis.sh

echo "Starting feature sweep analysis..."
python scripts/run_feature_sweep_analysis.py --min_features 3 --max_features 20

echo "Running detailed analysis on best feature set..."
# 这里可以添加自动选择最佳特征集的逻辑
python scripts/run_complete_analysis.py --feature_type best8

echo "Analysis complete!"
```

## 更新日志

- **v1.0** (2025-01-08): 初始版本，支持并行特征扫描和可视化
- 功能包括：性能对比、TCA分析、自动报告生成
- 支持best3到best58的完整特征范围扫描