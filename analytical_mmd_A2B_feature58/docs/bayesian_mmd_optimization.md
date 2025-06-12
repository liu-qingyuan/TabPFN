# 贝叶斯MMD优化文档

## 概述

贝叶斯MMD优化是本项目的核心创新功能，实现了**模型超参数**和**MMD域适应参数**的联合优化。与传统的分步优化相比，这种端到端的优化方法能够获得更好的域适应效果。

## 核心优势

### 🎯 端到端优化
- **传统方法**：先优化模型参数，再调整MMD参数
- **我们的方法**：同时优化两类参数，避免局部最优

### 🔬 科学的评估策略
- **三分法数据划分**：训练集、验证集、保留测试集
- **无偏评估**：验证集用于优化，保留测试集用于最终评估
- **泛化能力分析**：自动计算并报告泛化差距

### 🚀 高效的搜索策略
- **贝叶斯优化**：基于高斯过程的智能搜索
- **Expected Improvement**：平衡探索与利用
- **自适应搜索空间**：根据MMD方法动态调整参数范围

## 技术实现

### 搜索空间设计

#### AutoTabPFN模型参数
```python
search_space = [
    Categorical([1, 5, 10, 15, 30, 60, 120, 180], name='max_time'),
    Categorical(['default', 'avoid_overfitting'], name='preset'),
    Categorical(['accuracy', 'roc', 'f1'], name='ges_scoring'),
    Categorical([10, 15, 20, 25, 30], name='max_models'),
    Integer(50, 150, name='n_repeats'),
    Integer(20, 40, name='ges_n_iterations'),
    Categorical([True, False], name='ignore_limits'),
]
```

#### Linear MMD参数
```python
mmd_search_space = [
    Real(1e-5, 1e-1, prior='log-uniform', name='mmd_lr'),
    Integer(100, 1000, name='mmd_n_epochs'),
    Integer(32, 256, name='mmd_batch_size'),
    Real(1e-5, 1e-1, prior='log-uniform', name='mmd_lambda_reg'),
    Real(0.1, 10.0, name='mmd_gamma'),
    Categorical([True, False], name='mmd_staged_training'),
    Categorical([True, False], name='mmd_dynamic_gamma'),
]
```

#### KPCA MMD参数
```python
kpca_search_space = [
    Integer(10, min(50, n_features), name='mmd_n_components'),
    Real(0.1, 10.0, name='mmd_gamma'),
    Categorical([True, False], name='mmd_standardize'),
]
```

### 目标函数设计

```python
def objective_function(params):
    """
    目标函数：评估给定超参数组合的性能
    
    流程：
    1. 分离模型参数和MMD参数
    2. 创建并训练模型（A域训练集）
    3. 进行MMD域适应（A域→目标域验证集）
    4. 在域适应后的验证集上评估
    5. 返回负AUC（因为gp_minimize最小化目标函数）
    """
    
    # 1. 参数分离
    model_params = {k: v for k, v in params.items() if not k.startswith('mmd_')}
    mmd_params = {k[4:]: v for k, v in params.items() if k.startswith('mmd_')}
    
    # 2. 模型训练
    model = create_model(model_params)
    model.fit(X_train, y_train)
    
    # 3. MMD域适应
    X_val_adapted, adaptation_info = mmd_transform(
        X_train_raw, X_val_raw, 
        method=mmd_method, 
        **mmd_params
    )
    
    # 4. 验证集评估
    y_pred_proba = model.predict_proba(X_val_adapted)[:, 1]
    val_auc = roc_auc_score(y_val, y_pred_proba)
    
    # 5. 保留测试集评估（仅记录，不用于优化）
    X_test_adapted, _ = mmd_transform(
        X_train_raw, X_test_raw, 
        method=mmd_method, 
        **mmd_params
    )
    y_test_pred_proba = model.predict_proba(X_test_adapted)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    # 记录试验结果
    record_trial(model_params, mmd_params, val_auc, test_auc)
    
    return -val_auc  # 返回负值用于最小化
```

### 数据划分策略

```python
def load_and_prepare_data():
    """三分法数据划分"""
    
    # 加载数据
    df_A = pd.read_excel(DATA_PATHS['A'])  # 训练域
    df_target = pd.read_excel(DATA_PATHS[target_domain])  # 目标域
    
    # 数据预处理
    X_A_scaled, X_target_scaled, scaler = fit_apply_scaler(
        X_A_raw, X_target_raw, categorical_indices
    )
    
    # 目标域三分法划分
    X_val, X_holdout, y_val, y_holdout = train_test_split(
        X_target_scaled, y_target,
        train_size=validation_split,  # 默认0.7
        stratify=y_target,
        random_state=random_state
    )
    
    return {
        'X_train': X_A_scaled, 'y_train': y_A,
        'X_val': X_val, 'y_val': y_val,
        'X_holdout': X_holdout, 'y_holdout': y_holdout,
        'X_train_raw': X_A_raw,
        'X_val_raw': X_val_raw,
        'X_holdout_raw': X_holdout_raw
    }
```

## 使用指南

### 基本用法

```bash
# 最简单的用法
python scripts/run_bayesian_mmd_optimization.py

# 指定模型和MMD方法
python scripts/run_bayesian_mmd_optimization.py \
    --model-type auto \
    --mmd-method linear

# 选择目标域
python scripts/run_bayesian_mmd_optimization.py \
    --model-type auto \
    --mmd-method linear \
    --target-domain C
```

### 高级配置

```bash
# 完整配置示例
python scripts/run_bayesian_mmd_optimization.py \
    --model-type auto \
    --feature-type best7 \
    --mmd-method linear \
    --use-class-conditional \
    --target-domain B \
    --validation-split 0.7 \
    --n-calls 50 \
    --random-state 42 \
    --auto-run-mmd-after-bo \
    --output-dir ./my_optimization_results
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-type` | str | 'auto' | 模型类型：auto, base, rf, tuned |
| `--feature-type` | str | 'best7' | 特征类型：all, best7 |
| `--mmd-method` | str | 'linear' | MMD方法：linear, kpca, mean_std |
| `--use-class-conditional` | flag | False | 是否使用类条件MMD |
| `--target-domain` | str | 'B' | 目标域：B, C |
| `--validation-split` | float | 0.7 | 验证集比例 |
| `--n-calls` | int | 50 | 优化迭代次数 |
| `--random-state` | int | 42 | 随机种子 |
| `--auto-run-mmd-after-bo` | flag | False | 优化后自动运行完整实验 |

## 输出结果

### 文件结构
```
results_bayesian_mmd_optimization_auto_linear_best7/
├── bayesian_mmd_optimization_history.json    # 优化历史
├── final_mmd_evaluation.json                 # 最终评估结果
├── experiment_config.json                    # 实验配置
└── bayesian_mmd_optimization.log            # 详细日志
```

### 结果解读

#### 优化历史 (bayesian_mmd_optimization_history.json)
```json
{
  "best_params": {
    "max_time": 30,
    "preset": "avoid_overfitting",
    "mmd_lr": 0.001,
    "mmd_gamma": 1.5
  },
  "best_validation_auc": 0.8234,
  "total_trials": 50,
  "good_configs": [
    {
      "validation_auc": 0.8234,
      "test_auc": 0.7891,
      "model_params": {...},
      "mmd_params": {...}
    }
  ],
  "optimization_history": [...]
}
```

#### 最终评估 (final_mmd_evaluation.json)
```json
{
  "validation_performance": {
    "auc": 0.8234,
    "f1": 0.7456,
    "accuracy": 0.8012
  },
  "holdout_performance": {
    "auc": 0.7891,
    "f1": 0.7123,
    "accuracy": 0.7789
  },
  "generalization_gap": {
    "auc_gap": 0.0343,
    "f1_gap": 0.0333,
    "accuracy_gap": 0.0223
  }
}
```

### 性能分析

#### 泛化能力评估
- **AUC差距 < 0.05**：泛化能力良好 ✅
- **AUC差距 ≥ 0.05**：可能过拟合 ⚠️

#### 优化效果评估
- **验证集AUC > 0.8**：优化效果优秀
- **测试集AUC > 0.75**：实际性能良好
- **发现优秀配置数量**：搜索空间覆盖度

## 最佳实践

### 参数选择建议

#### 验证集比例
- **0.7**：平衡优化稳定性和测试可靠性（推荐）
- **0.9**：更大的验证集，更稳定的优化过程

#### 优化迭代次数
- **快速测试**：20-30次
- **标准优化**：50次（推荐）
- **深度优化**：100次以上

#### MMD方法选择
- **Linear**：最灵活，参数最多，适合深度优化
- **KPCA**：中等复杂度，适合中等规模优化
- **Mean-Std**：最简单，适合快速测试

### 实验设计建议

#### 对比实验
```bash
# 1. 基线：无域适应
python scripts/run_analytical_mmd.py --model-type auto --skip-cv-on-a

# 2. 标准MMD：手动参数
python scripts/run_analytical_mmd.py --model-type auto --method linear

# 3. 贝叶斯MMD：优化参数
python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method linear
```

#### 消融实验
```bash
# 仅优化模型参数
python scripts/run_bayesian_optimization.py --model-type auto

# 同时优化模型和MMD参数
python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method linear
```

## 故障排除

### 常见问题

#### 1. 优化收敛慢
**原因**：搜索空间过大或目标函数噪声大
**解决**：
- 减少搜索空间维度
- 增加初始随机点数量
- 使用更稳定的评估指标

#### 2. 泛化差距大
**原因**：验证集过拟合
**解决**：
- 增加验证集大小（减少validation_split）
- 使用更保守的模型参数
- 增加正则化强度

#### 3. 优化结果不稳定
**原因**：随机性影响
**解决**：
- 固定随机种子
- 增加优化迭代次数
- 使用多次运行的平均结果

### 调试技巧

#### 启用详细日志
```bash
python scripts/run_bayesian_mmd_optimization.py \
    --model-type auto \
    --mmd-method linear \
    --n-calls 10 \
    --log-file debug.log
```

#### 快速验证
```bash
# 最小配置测试
python scripts/run_bayesian_mmd_optimization.py \
    --model-type auto \
    --mmd-method mean_std \
    --n-calls 5
```

## 扩展开发

### 添加新的搜索空间
```python
# 在 define_search_space() 中添加
if self.model_type == 'new_model':
    search_space.extend([
        Real(0.1, 1.0, name='new_param'),
        Integer(1, 10, name='another_param'),
    ])
```

### 自定义目标函数
```python
def custom_objective_function(self, params):
    """自定义目标函数，可以优化多个指标"""
    
    # 获取基本性能
    val_auc = self.evaluate_performance(params)
    
    # 添加自定义约束
    if some_constraint_violated(params):
        return 1.0  # 惩罚违反约束的参数
    
    # 多目标优化
    val_f1 = self.evaluate_f1(params)
    combined_score = 0.7 * val_auc + 0.3 * val_f1
    
    return -combined_score
```

### 集成新的MMD方法
```python
# 在搜索空间定义中添加
elif self.mmd_method == 'new_mmd_method':
    search_space.extend([
        Real(0.01, 1.0, name='mmd_new_param1'),
        Integer(5, 50, name='mmd_new_param2'),
    ])
```

## 理论背景

### 贝叶斯优化原理
贝叶斯优化使用高斯过程（Gaussian Process）建模目标函数，通过获取函数（Acquisition Function）指导下一步搜索方向。

#### Expected Improvement (EI)
```
EI(x) = E[max(f(x) - f(x*), 0)]
```
其中 f(x*) 是当前最佳值。

### MMD域适应理论
Maximum Mean Discrepancy (MMD) 衡量两个分布之间的差异：
```
MMD²(P, Q) = ||μ_P - μ_Q||²_H
```
其中 H 是再生核希尔伯特空间。

### 联合优化的理论优势
传统的分步优化可能陷入局部最优：
```
θ* = argmin L(θ_model, θ_mmd_fixed)
φ* = argmin L(θ_model_fixed, θ_mmd)
```

而联合优化能找到全局最优：
```
(θ*, φ*) = argmin L(θ_model, θ_mmd)
```

## 参考文献

1. Snoek, J., et al. "Practical Bayesian optimization of machine learning algorithms." NIPS 2012.
2. Gretton, A., et al. "A kernel two-sample test." JMLR 2012.
3. Long, M., et al. "Learning transferable features with deep adaptation networks." ICML 2015.

---

*本文档持续更新中，如有问题请提交Issue或联系开发团队。* 