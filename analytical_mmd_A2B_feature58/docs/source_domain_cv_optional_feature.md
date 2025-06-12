# A域交叉验证基准评估 - 可选功能说明

## 概述

A域交叉验证基准评估现在作为贝叶斯MMD优化的可选功能提供。用户可以根据需要选择是否评估A域内的模型性能作为跨域比较的基准。

## 功能特点

### 1. 可选性设计
- **默认禁用**：`--evaluate-source-cv` 默认为 `False`
- **按需启用**：只有在需要跨域性能比较时才启用
- **性能考虑**：避免不必要的计算开销（通常增加15-20%运行时间）

### 2. 两种使用模式

#### 基本模式（默认）
```bash
python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method linear
```
- 专注于贝叶斯MMD优化
- 输出：优化历史、最终评估结果、实验配置
- 运行时间：标准

#### 完整分析模式
```bash
python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method linear --evaluate-source-cv
```
- 包含A域CV基准评估
- 提供跨域性能比较分析
- 输出：基本文件 + A域基准 + 跨域比较
- 运行时间：增加15-20%

## 输出文件对比

### 基本模式输出
```
results_bayesian_mmd_optimization/
├── bayesian_mmd_optimization_history.json  # 优化历史
├── final_mmd_evaluation.json               # 最终评估结果
└── experiment_config.json                  # 实验配置
```

### 完整分析模式输出
```
results_bayesian_mmd_optimization/
├── bayesian_mmd_optimization_history.json          # 优化历史
├── final_mmd_evaluation.json                       # 最终评估结果
├── experiment_config.json                          # 实验配置
├── source_domain_cv_baseline.json                  # A域CV基准 ⭐ 新增
└── cross_domain_performance_comparison.json        # 跨域性能比较 ⭐ 新增
```

## 使用场景建议

### 何时启用A域CV评估

1. **研究目的**：需要发表论文或详细分析跨域性能时
2. **方法比较**：与其他域适应方法进行严格比较时  
3. **性能诊断**：怀疑域适应效果不理想，需要基准参考时
4. **完整实验**：进行系统性的实验评估时

### 何时保持禁用

1. **快速原型**：快速测试不同参数配置时
2. **资源受限**：计算资源紧张或时间紧迫时
3. **重复实验**：已有A域基准，只需调整MMD参数时
4. **生产环境**：实际部署时不需要基准比较

## 结果解读

### 启用A域CV评估时的额外输出

```
A域交叉验证基准:
  AUC: 0.8234 ± 0.0156
  F1:  0.7845 ± 0.0234
  Acc: 0.7756 ± 0.0198

跨域性能比较:
  A域CV基准 → B域验证集: +2.45%
  A域CV基准 → B域测试集: +1.78%
  性能差距-验证集: +0.0201
  性能差距-测试集: +0.0147
```

### 性能指标含义

- **跨域改进百分比**：目标域相对于源域的性能提升幅度
- **性能差距**：目标域与源域的绝对AUC差异
- **正值**：表示域适应带来性能提升
- **负值**：表示存在负迁移效应

## 参数配置

### 命令行参数
- `--evaluate-source-cv`：启用A域CV评估（布尔标志）
- 无需额外参数，自动进行5折交叉验证

### 程序化调用
```python
from analytical_mmd_A2B_feature58.modeling.bayesian_mmd_optimizer import run_bayesian_mmd_optimization

# 基本模式
results = run_bayesian_mmd_optimization(
    model_type='auto',
    mmd_method='linear',
    evaluate_source_cv=False  # 默认值
)

# 完整分析模式
results = run_bayesian_mmd_optimization(
    model_type='auto', 
    mmd_method='linear',
    evaluate_source_cv=True   # 启用A域CV评估
)
```

## 性能开销

| 模式 | A域CV评估 | 额外时间开销 | 额外文件输出 |
|------|-----------|--------------|--------------|
| 基本模式 | ❌ | 0% | 3个文件 |
| 完整模式 | ✅ | 15-20% | 5个文件 |

## 最佳实践

1. **首次运行**：建议启用 `--evaluate-source-cv` 获得完整分析
2. **参数调优**：后续调优过程可禁用以提高效率
3. **结果报告**：最终结果报告时应包含A域基准比较
4. **文档记录**：在实验日志中记录是否启用了A域CV评估

## 与其他工具的兼容性

此功能完全兼容现有的分析工具：
- 可视化脚本会自动检测并使用A域基准数据
- 比较分析工具可处理包含或不包含基准的结果
- 现有的结果文件格式保持不变

## 小结

A域CV评估作为可选功能提供了灵活性：
- 🎯 **目标导向**：根据具体需求选择使用
- ⚡ **性能优化**：避免不必要的计算开销
- 📊 **完整分析**：需要时提供详细的跨域比较
- 🔄 **向后兼容**：不影响现有功能和脚本 