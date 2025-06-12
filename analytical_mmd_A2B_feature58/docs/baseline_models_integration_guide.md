# 基线模型集成指南 📊

## 概述

我们为固定参数域适应实验新增了两个传统医疗预测模型作为基线对比：
- **PKUPH模型**：基于北京大学人民医院的预测模型
- **Mayo模型**：基于Mayo Clinic的预测模型

这些基线模型将与AutoTabPFN进行全面的性能对比，包括源域和目标域的表现。

## 🔧 新增功能

### 1. 基线模型实现

#### PKUPH模型
```python
# 模型公式
P(malignant) = e^x / (1+e^x)
x = -4.496 + (0.07 × Feature2) + (0.676 × Feature48) + (0.736 × Feature49) + 
    (1.267 × Feature4) - (1.615 × Feature50) - (1.408 × Feature53)
```

#### Mayo模型
```python
# 模型公式
P(malignant) = e^x / (1+e^x)
x = -6.8272 + (0.0391 × Feature2) + (0.7917 × Feature3) + (1.3388 × Feature5) + 
    (0.1274 × Feature48) + (1.0407 × Feature49) + (0.7838 × Feature63)
```

### 2. 评估指标

每个模型将在以下三个场景中评估：

#### 📈 源域10折交叉验证
- **AUC**: Area Under ROC Curve
- **ACC**: Accuracy 
- **F1**: F1-Score
- 输出格式：`mean ± std`

#### 🔍 源域验证集评估
- 使用20%的源域数据作为验证集
- 评估指标：AUC, ACC, F1, Precision, Recall

#### 🎯 目标域测试集评估
- 直接在目标域（B或C）上测试
- 评估指标：AUC, ACC, F1, Precision, Recall

## 🚀 使用方法

### 基本使用

```bash
# 包含基线模型对比的A→B实验
python scripts/run_fixed_params_domain_adaptation.py \
    --target-domain B \
    --include-baselines

# 包含基线模型对比的A→C实验  
python scripts/run_fixed_params_domain_adaptation.py \
    --target-domain C \
    --include-baselines
```

### 高级配置

```bash
# 完整配置示例
python scripts/run_fixed_params_domain_adaptation.py \
    --target-domain B \
    --include-baselines \
    --source-cv-folds 10 \
    --domain-adapt-method coral \
    --use-class-conditional \
    --feature-type best7 \
    --output-dir ./results_with_baselines
```

### 新增命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--include-baselines` | flag | False | 启用基线模型评估 |
| `--source-cv-folds` | int | 10 | 源域交叉验证折数 |

## 📊 输出结果

### 控制台输出示例

```
🔍 基线模型性能对比:
--------------------------------------------------

PKUPH 模型:
  源域10折CV - AUC: 0.8234 ± 0.0456
  源域10折CV - ACC: 0.7891 ± 0.0389
  源域10折CV - F1:  0.7654 ± 0.0421
  源域验证集 - AUC: 0.8156
  目标域测试 - AUC: 0.7523
  域差距: 0.0633 (7.8%)

MAYO 模型:
  源域10折CV - AUC: 0.8456 ± 0.0398
  源域10折CV - ACC: 0.8023 ± 0.0356
  源域10折CV - F1:  0.7834 ± 0.0412
  源域验证集 - AUC: 0.8389
  目标域测试 - AUC: 0.7745
  域差距: 0.0644 (7.7%)

🚀 AutoTabPFN模型性能:
--------------------------------------------------
最终模型性能:
  源域验证集 AUC: 0.8912
  源域验证集 ACC: 0.8456
  源域验证集 F1:  0.8234
  目标域直接预测 AUC: 0.8234
  目标域域适应后 AUC: 0.8456
  域适应改进: 0.0222 (2.7%)

📊 模型性能对比分析:
--------------------------------------------------
AutoTabPFN 域差距: 0.0678 (7.6%)

基线模型 vs AutoTabPFN (目标域AUC对比):
  AutoTabPFN vs PKUPH: +0.0933 (+12.4%)
    ✓ AutoTabPFN优于PKUPH模型
  AutoTabPFN vs MAYO: +0.0711 (+9.2%)
    ✓ AutoTabPFN优于MAYO模型
```

### 保存文件

实验结果会保存以下文件：
- `optimization_results.json`: 实验配置和参数
- `evaluation_results.json`: AutoTabPFN模型评估结果
- `baseline_models_results.json`: 基线模型评估结果 ⭐ **新增**
- `experiment_config.json`: 实验配置信息
- `experiment.log`: 完整实验日志

### baseline_models_results.json 结构

```json
{
  "pkuph": {
    "source_validation": {
      "auc": 0.8156,
      "accuracy": 0.7891,
      "f1": 0.7654,
      "precision": 0.7543,
      "recall": 0.7789
    },
    "target_direct": {
      "auc": 0.7523,
      "accuracy": 0.7234,
      "f1": 0.7012,
      "precision": 0.6945,
      "recall": 0.7089
    },
    "source_cv": {
      "auc": {"mean": 0.8234, "std": 0.0456},
      "accuracy": {"mean": 0.7891, "std": 0.0389},
      "f1": {"mean": 0.7654, "std": 0.0421}
    },
    "model_features": ["Feature2", "Feature48", "Feature49", "Feature4", "Feature50", "Feature53"]
  },
  "mayo": {
    // 类似结构
  }
}
```

## 🎯 实验场景

### 场景1：快速基线对比
```bash
# 只评估基线模型，不进行域适应
python scripts/run_fixed_params_domain_adaptation.py \
    --target-domain B \
    --include-baselines \
    --no-mmd
```

### 场景2：完整性能对比
```bash
# 包含域适应的完整对比
python scripts/run_fixed_params_domain_adaptation.py \
    --target-domain B \
    --include-baselines \
    --domain-adapt-method coral \
    --use-class-conditional
```

### 场景3：不同目标域对比
```bash
# A→B实验
python scripts/run_fixed_params_domain_adaptation.py \
    --target-domain B --include-baselines

# A→C实验  
python scripts/run_fixed_params_domain_adaptation.py \
    --target-domain C --include-baselines
```

## 📈 性能分析指标

### 域差距分析
- **计算公式**: `源域AUC - 目标域AUC`
- **评判标准**:
  - `> 0.1`: 显著域差距，域适应很有必要
  - `0.05-0.1`: 中等域差距，域适应可能有帮助
  - `< 0.05`: 域差距较小，良好泛化能力

### 模型对比分析
- **AutoTabPFN vs 基线模型**: 目标域AUC改进
- **域适应效果**: 域适应前后的性能提升
- **一致性分析**: 所有模型的域差距对比

## 🔧 技术实现

### 核心组件
1. **`baseline_models.py`**: PKUPH和Mayo模型实现
2. **`standard_domain_adaptation_optimizer.py`**: 增强的优化器
3. **`run_fixed_params_domain_adaptation.py`**: 更新的实验脚本

### 新增方法
- `evaluate_baseline_models_performance()`: 基线模型评估
- `PKUPHModel` & `MayoModel`: 预定义医疗模型
- `evaluate_baseline_models()`: 通用评估函数

## 🎯 预期结果

通过这个增强的实验框架，您将获得：

1. **完整的模型对比**: AutoTabPFN vs 传统医疗模型
2. **多维性能评估**: 源域CV、源域验证、目标域测试
3. **域适应效果量化**: 明确的改进幅度和统计意义
4. **详细的分析报告**: 包含所有关键指标和对比分析

这将为医疗AI模型的性能评估提供更加全面和可信的实验证据。

## 📝 注意事项

1. **特征依赖**: 确保数据集包含基线模型所需的特征
2. **数据质量**: 基线模型对特征值的范围和分布敏感
3. **对比公平性**: 所有模型使用相同的数据划分和评估标准
4. **结果解释**: 考虑模型复杂度差异对性能比较的影响 