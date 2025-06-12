# 标准域适应实验运行指南

## 1. 如何运行完整流程

### 环境准备

确保你在正确的目录下：
```bash
cd analytical_mmd_A2B_feature58
```

### 基本运行命令

#### 1.1 最简单的运行方式（A→B）
```bash
python scripts/run_standard_domain_adaptation.py \
    --model-type auto \
    --mmd-method linear \
    --target-domain B
```

#### 1.2 指定目标域为C（A→C）
```bash
python scripts/run_standard_domain_adaptation.py \
    --model-type auto \
    --mmd-method linear \
    --target-domain C
```

#### 1.3 使用类条件MMD
```bash
python scripts/run_standard_domain_adaptation.py \
    --model-type auto \
    --mmd-method linear \
    --use-class-conditional \
    --target-domain B
```

#### 1.4 快速测试（减少优化次数）
```bash
python scripts/run_standard_domain_adaptation.py \
    --model-type auto \
    --mmd-method linear \
    --target-domain B \
    --n-calls 20 \
    --cv-folds 3
```

#### 1.5 完整配置示例
```bash
python scripts/run_standard_domain_adaptation.py \
    --model-type auto \
    --feature-type best7 \
    --mmd-method linear \
    --target-domain B \
    --cv-folds 5 \
    --source-val-split 0.2 \
    --n-calls 50 \
    --random-state 42 \
    --output-dir ./results_my_experiment
```

### 参数说明

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| `--model-type` | 模型类型 | auto, rf, base | auto |
| `--feature-type` | 特征类型 | best7, all | best7 |
| `--mmd-method` | MMD方法 | linear, mean_std | linear |
| `--target-domain` | 目标域 | B, C | B |
| `--cv-folds` | 交叉验证折数 | 整数 ≥ 2 | 5 |
| `--source-val-split` | 源域验证集比例 | 0-1之间的浮点数 | 0.2 |
| `--n-calls` | 贝叶斯优化迭代次数 | 正整数 | 50 |
| `--random-state` | 随机种子 | 整数 | 42 |
| `--use-class-conditional` | 使用类条件MMD | 标志参数 | False |
| `--no-mmd-tuning` | 禁用MMD参数调优 | 标志参数 | False |
| `--no-categorical` | 禁用类别特征 | 标志参数 | False |

### 运行时间估算

- **快速测试**（n-calls=20, cv-folds=3）：约 10-20 分钟
- **标准配置**（n-calls=50, cv-folds=5）：约 30-60 分钟
- **完整优化**（n-calls=100, cv-folds=5）：约 1-2 小时

## 2. 调参指标详解

### 2.1 核心调参指标：源域交叉验证AUC

**关键原则**：完全基于源域数据进行参数调优，不使用目标域数据。

```python
# 调参的核心逻辑
def objective_function(params):
    cv_scores = []
    
    # 在源域训练集内进行5折交叉验证
    for fold in range(5):
        X_fold_train, X_fold_val = split_source_data(fold)
        
        # 1. 训练模型
        model.fit(X_fold_train, y_fold_train)
        
        # 2. 如果启用MMD调优，在源域内模拟域适应
        if use_mmd_tuning:
            X_fold_val_adapted = mmd_transform(
                X_fold_train, X_fold_val, **mmd_params
            )
            y_pred_proba = model.predict_proba(X_fold_val_adapted)
        else:
            y_pred_proba = model.predict_proba(X_fold_val)
        
        # 3. 计算AUC作为评估指标
        fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(fold_auc)
    
    # 4. 返回平均AUC（贝叶斯优化会最大化这个值）
    return np.mean(cv_scores)
```

### 2.2 为什么选择AUC作为调参指标？

1. **类别不平衡友好**：医疗数据通常存在类别不平衡，AUC比准确率更可靠
2. **概率排序能力**：AUC衡量模型区分正负样本的能力
3. **阈值无关**：不依赖于特定的分类阈值
4. **域适应友好**：在跨域场景下，AUC通常比其他指标更稳定

### 2.3 调参的两个层面

#### 层面1：模型参数调优
```python
# AutoTabPFN参数
model_params = {
    'max_time': 60-300,           # 训练时间限制
    'preset': ['default', 'best_quality'],  # 预设配置
    'ges_scoring': ['roc', 'f1'], # 评分方法
    'max_models': 10-50,          # 最大模型数
    'n_repeats': 50-200,          # 重复次数
    # ...
}

# 随机森林参数
rf_params = {
    'n_estimators': 50-500,       # 树的数量
    'max_depth': 1-20,            # 最大深度
    'min_samples_split': 2-20,    # 最小分割样本数
    # ...
}
```

#### 层面2：MMD域适应参数调优（如果启用）
```python
# Linear MMD参数
mmd_params = {
    'lr': 0.001-0.1,              # 学习率
    'n_epochs': 50-500,           # 训练轮数
    'batch_size': 16-128,         # 批大小
    'lambda_reg': 0.001-0.1,      # 正则化系数
    'gamma': 0.1-10.0,            # MMD权重
    'staged_training': [True, False],     # 分阶段训练
    'dynamic_gamma': [True, False],       # 动态gamma
}

# Mean-Std MMD参数
mean_std_params = {
    'eps': 1e-8 to 1e-4,          # 数值稳定性参数
}
```

### 2.4 调参策略对比

#### 策略1：联合调优（默认）
- **同时调优**：模型参数 + MMD参数
- **优点**：可能找到最优参数组合
- **缺点**：搜索空间大，耗时较长
- **适用**：有充足时间，追求最佳性能

#### 策略2：分离调优（保守）
```bash
# 只调优模型参数，固定MMD参数
python scripts/run_standard_domain_adaptation.py \
    --model-type auto \
    --no-mmd-tuning \
    --target-domain B
```
- **只调优**：模型参数
- **优点**：搜索空间小，速度快，更稳定
- **缺点**：可能错过最优MMD参数
- **适用**：快速实验，或MMD参数已知较好值

### 2.5 调参过程监控

实验运行时会输出详细日志：

```
2024-01-15 10:30:15 - INFO - 开始评估参数组合: {'max_time': 120, 'mmd_lr': 0.01, ...}
2024-01-15 10:30:20 - INFO -   交叉验证 Fold 1/5
2024-01-15 10:30:25 - INFO -     Fold 1 AUC: 0.8234
2024-01-15 10:30:30 - INFO -   交叉验证 Fold 2/5
2024-01-15 10:30:35 - INFO -     Fold 2 AUC: 0.8156
...
2024-01-15 10:31:00 - INFO -   平均CV AUC: 0.8195 ± 0.0089
```

### 2.6 最终评估指标

调参完成后，会在多个层面评估模型：

1. **源域验证集AUC**：模型在源域的性能上界
2. **目标域直接预测AUC**：无域适应的跨域基线性能
3. **目标域域适应后AUC**：MMD域适应后的跨域性能
4. **域适应改进**：域适应带来的性能提升

```
最终评估结果:
源域验证集 AUC: 0.8456
目标域直接预测 AUC: 0.7234
目标域域适应后 AUC: 0.7589
域适应改进: 0.0355 (4.9%)
```

## 3. 结果文件说明

实验完成后会生成：

```
results_standard_domain_adaptation_auto_linear_best7_target_B/
├── optimization_results.json      # 贝叶斯优化的所有试验结果
├── evaluation_results.json        # 最终模型在各个数据集上的性能
├── experiment_config.json         # 实验配置参数
└── experiment.log                 # 完整的实验日志
```

## 4. 常见问题

### Q1: 如果实验中断了怎么办？
A: 目前不支持断点续传，需要重新运行。建议先用小的n-calls测试。

### Q2: 如何选择合适的n-calls？
A: 
- 快速测试：20-30
- 正常实验：50-100  
- 充分搜索：100-200

### Q3: CV折数如何选择？
A: 
- 数据量小：3-5折
- 数据量中等：5折（推荐）
- 数据量大：5-10折

### Q4: 如何判断实验是否成功？
A: 查看日志中的关键指标：
- 最佳CV AUC > 0.7（说明模型有效）
- 域适应改进 > 0（说明MMD有效）
- 无异常错误信息 