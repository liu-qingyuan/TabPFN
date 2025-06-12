# Analytical MMD A2B Feature58 工作流程文档

## 概述

本项目实现了基于MMD（Maximum Mean Discrepancy）的域适应方法，用于医疗数据的跨域预测。支持多种数据划分策略、目标域选择和贝叶斯优化功能。

## 核心特性

### 1. 目标域选择

项目现在支持灵活的目标域选择：

#### 目标域B（河南癌症医院）
- **默认选项**：与原有A2B实验保持一致
- **数据特点**：河南癌症医院的医疗数据
- **使用场景**：标准的A→B域适应实验

#### 目标域C（广州医科大学）
- **新增选项**：支持A→C域适应实验
- **数据特点**：广州医科大学的医疗数据
- **使用场景**：探索不同医院间的域适应效果

### 2. 数据划分策略

项目支持两种数据划分策略：

#### 二分法 (Two-way Split)
- **A域数据**：完整用作训练集
- **B域数据**：完整用作测试集
- **适用场景**：标准的域适应评估，与原始MMD方法保持一致

#### 三分法 (Three-way Split)
- **A域数据**：完整用作训练集
- **B域数据**：划分为验证集(默认80%)和保留测试集(20%)
- **验证集**：用于模型选择和超参数优化
- **保留测试集**：用于最终泛化能力评估
- **适用场景**：需要进行超参数优化或模型选择的场景

### 3. 贝叶斯优化集成

项目现在支持两种贝叶斯优化模式：

#### 标准贝叶斯优化
- **功能**：仅优化模型超参数
- **适用场景**：快速模型调优，不涉及域适应参数
- **脚本**：`run_bayesian_optimization.py`

#### 贝叶斯MMD优化（推荐）
- **功能**：同时优化模型超参数和MMD域适应参数
- **优势**：端到端优化，获得最佳的域适应效果
- **脚本**：`run_bayesian_mmd_optimization.py`
- **搜索空间**：
  - 模型参数：max_time, preset, ges_scoring, max_models等
  - MMD参数：lr, n_epochs, gamma, lambda_reg等（根据MMD方法而定）

### 4. 数据划分策略

### 🚀 主函数执行流程

### 1. 程序入口 - main()

```python
def main():
    """主函数 - 程序执行的起点"""
```

**执行步骤:**

#### 1.1 参数解析
```python
args = parse_arguments()
```
- 解析命令行参数
- 支持的主要参数：
  - `--model-type`: 模型类型 (auto/tuned/base/rf)
  - `--method`: MMD方法 (linear/kpca/mean_std)
  - `--feature-type`: 特征类型 (all/best7)
  - `--use-class-conditional`: 是否使用类条件MMD
  - `--use-threshold-optimizer`: 是否使用阈值优化
  - `--skip-cv-on-a`: 是否跳过数据集A的交叉验证
  - `--evaluation-mode`: 评估模式 (cv/proper_cv/single)
  - `--data-split-strategy`: 数据划分策略 (`two-way` | `three-way`)
  - `--validation-split`: 三分法时验证集比例 (默认: 0.7)
  - `--target-domain`: 目标域选择 (`B` | `C`) (默认: B)

#### 1.2 日志系统初始化
```python
logger = setup_experiment_logging(args.log_file)
```
- 创建带时间戳的日志文件
- 设置控制台和文件双重输出
- 日志格式：`时间戳 - 模块名 - 级别 - 消息`

#### 1.3 实验配置打印
```python
logger.info("实验配置:")
logger.info(f"  模型类型: {args.model_type}")
# ... 其他配置信息
```

#### 1.4 模型可用性检查
```python
available_models = get_available_models()
if args.model_type not in available_models:
    logger.error(f"模型类型 {args.model_type} 不可用")
    return
```

#### 1.5 实验模式分发
```python
if args.compare_all:
    run_comparison_experiment(args, logger)
else:
    # 直接运行跨域实验
    run_cross_domain_experiment_mode(args, logger)
```

## 🔄 实验模式详细流程

### 主要模式: 跨域实验

#### 函数: `run_cross_domain_experiment_mode()`

**步骤1: 数据验证**
```python
if not validate_data_paths():
    logger.error("数据文件验证失败")
    return
```
- 检查 AI4Health、河南癌症医院、广州医科大学数据文件是否存在
- 验证数据路径的完整性

**步骤2: 参数准备**
```python
model_kwargs = prepare_model_kwargs(args)
mmd_kwargs = prepare_mmd_kwargs(args, args.method)
```

**prepare_model_kwargs() 详细流程:**
```python
# 获取基础配置
if args.model_preset:
    base_config = get_model_config(args.model_type, args.model_preset)
else:
    base_config = get_model_config(args.model_type)

# 覆盖特定参数
if args.max_time is not None:
    model_kwargs['max_time'] = args.max_time
# ... 其他参数覆盖

# 验证参数
model_kwargs = validate_model_params(args.model_type, model_kwargs)
```

**prepare_mmd_kwargs() 详细流程:**
```python
# 获取MMD方法基础配置
mmd_kwargs = MMD_METHODS.get(method, {}).copy()

# Linear方法特定参数
if method == 'linear':
    if args.lr is not None:
        mmd_kwargs['lr'] = args.lr
    # ... 其他linear参数
    
    # 预设配置处理
    if args.use_preset == 'conservative':
        mmd_kwargs.update({
            'lr': 1e-4,
            'lambda_reg': 1e-2,
            'staged_training': True,
            # ...
        })
```

**步骤3: 保存路径生成**
```python
if args.output_dir:
    save_path = args.output_dir
else:
    suffix = generate_experiment_suffix(args)
    save_path = f"./results_cross_domain_{args.model_type}_{args.method}_{args.feature_type}{suffix}"
```

**步骤4: 核心实验执行**
```python
results = run_cross_domain_experiment(
    model_type=args.model_type,
    feature_type=args.feature_type,
    mmd_method=args.method,
    use_class_conditional=args.use_class_conditional,
    use_threshold_optimizer=args.use_threshold_optimizer,
    save_path=save_path,
    skip_cv_on_a=args.skip_cv_on_a,
    evaluation_mode=args.evaluation_mode,
    save_visualizations=not args.no_visualizations,
    **{**model_kwargs, **mmd_kwargs}
)
```

**步骤5: 结果输出**
```python
# 打印数据集A交叉验证结果
if 'cross_validation_A' in results:
    cv_results = results['cross_validation_A']
    logger.info(f"数据集A交叉验证 - 准确率: {cv_results['accuracy']}")
    logger.info(f"数据集A交叉验证 - AUC: {cv_results['auc']}")

# 打印数据集B外部验证结果
if 'external_validation_B' in results:
    # 无域适应结果
    # 有域适应结果
    
# 打印数据集C外部验证结果
if 'external_validation_C' in results:
    # 无域适应结果
    # 有域适应结果
```

### 模式2: 方法比较模式 (compare-all)

#### 函数: `run_comparison_experiment()`

**执行流程:**
```python
methods = ['linear', 'kpca', 'mean_std']
all_results = {}

for method in methods:
    logger.info(f"运行{method}方法...")
    
    # 更新参数
    args.method = method
    
    try:
        # 直接运行跨域实验
        run_cross_domain_experiment_mode(args, logger)
        
        logger.info(f"{method}方法完成")
        
    except Exception as e:
        logger.error(f"{method}方法失败: {e}")
        continue

logger.info("所有方法比较完成!")
```

## 🔧 核心跨域实验流程详解

### run_cross_domain_experiment() 函数流程

这是整个系统的核心函数，位于 `modeling/cross_domain_runner.py` 中：

#### 阶段1: 数据准备
1. **数据加载**
   ```python
   datasets = load_all_datasets()
   # 加载 AI4Health (A), 河南癌症医院 (B), 广州医科大学 (C)
   ```

2. **特征选择**
   ```python
   features = get_features_by_type(feature_type)
   categorical_indices = get_categorical_indices(feature_type)
   ```

3. **数据预处理**
   ```python
   # 数据标准化
   X_A_scaled, scaler = fit_apply_scaler(X_A, categorical_indices)
   X_B_scaled = apply_scaler(X_B, scaler, categorical_indices)
   X_C_scaled = apply_scaler(X_C, scaler, categorical_indices)
   ```

#### 阶段2: 基线评估 (可选)
```python
if not skip_cv_on_a:
    # 数据集A上的10折交叉验证
    cv_results = evaluate_model_on_external_cv(model, X_A_scaled, y_A, n_folds=10)
```

#### 阶段3: 无域适应评估
```python
# 在数据集A上训练，在B和C上测试
model.fit(X_A_scaled, y_A)

# 数据集B评估
y_B_pred = model.predict(X_B_scaled)
y_B_proba = model.predict_proba(X_B_scaled)[:, 1]
b_results_no_da = evaluate_metrics(y_B, y_B_pred, y_B_proba)

# 数据集C评估
y_C_pred = model.predict(X_C_scaled)
y_C_proba = model.predict_proba(X_C_scaled)[:, 1]
c_results_no_da = evaluate_metrics(y_C, y_C_pred, y_C_proba)
```

#### 阶段4: MMD域适应
```python
if use_class_conditional:
    # 类条件MMD变换
    X_B_adapted = class_conditional_mmd_transform(
        X_A_scaled, y_A, X_B_scaled, method=mmd_method, **mmd_kwargs
    )
    X_C_adapted = class_conditional_mmd_transform(
        X_A_scaled, y_A, X_C_scaled, method=mmd_method, **mmd_kwargs
    )
else:
    # 标准MMD变换
    X_B_adapted = mmd_transform(
        X_A_scaled, X_B_scaled, method=mmd_method, **mmd_kwargs
    )
    X_C_adapted = mmd_transform(
        X_A_scaled, X_C_scaled, method=mmd_method, **mmd_kwargs
    )
```

#### 阶段5: 域适应后评估
```python
# 重新训练模型
model.fit(X_A_scaled, y_A)

# 在适应后的数据上评估
y_B_pred_adapted = model.predict(X_B_adapted)
y_B_proba_adapted = model.predict_proba(X_B_adapted)[:, 1]
b_results_with_da = evaluate_metrics(y_B, y_B_pred_adapted, y_B_proba_adapted)

y_C_pred_adapted = model.predict(X_C_adapted)
y_C_proba_adapted = model.predict_proba(X_C_adapted)[:, 1]
c_results_with_da = evaluate_metrics(y_C, y_C_pred_adapted, y_C_proba_adapted)
```

#### 阶段6: 阈值优化 (可选)
```python
if use_threshold_optimizer:
    # 在数据集B上优化阈值
    optimal_threshold_B, optimized_metrics_B = optimize_threshold(y_B, y_B_proba_adapted)
    
    # 在数据集C上优化阈值
    optimal_threshold_C, optimized_metrics_C = optimize_threshold(y_C, y_C_proba_adapted)
```

#### 阶段7: 可视化生成
```python
if save_visualizations:
    # t-SNE对比可视化
    compare_before_after_adaptation(
        X_A_scaled, X_B_scaled, X_B_adapted,
        y_A, y_B, y_B,
        save_path=os.path.join(save_path, 'visualizations')
    )
    
    # ROC曲线
    plot_roc_curve(y_B, y_B_proba_adapted, 
                   save_path=os.path.join(save_path, 'roc_curve_B.png'))
```

#### 阶段8: 结果保存
```python
# 保存详细结果
results = {
    'cross_validation_A': cv_results,
    'external_validation_B': {
        'without_domain_adaptation': b_results_no_da,
        'with_domain_adaptation': b_results_with_da
    },
    'external_validation_C': {
        'without_domain_adaptation': c_results_no_da,
        'with_domain_adaptation': c_results_with_da
    },
    'experiment_config': {
        'model_type': model_type,
        'feature_type': feature_type,
        'mmd_method': mmd_method,
        'use_class_conditional': use_class_conditional,
        'use_threshold_optimizer': use_threshold_optimizer
    }
}

# 保存到JSON文件
with open(os.path.join(save_path, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)
```

## 📊 MMD方法具体实现流程

### Linear MMD 变换流程

#### 1. 初始化
```python
transformer = MMDLinearTransform(
    gamma=gamma,
    lr=lr,
    n_epochs=n_epochs,
    batch_size=batch_size,
    lambda_reg=lambda_reg,
    staged_training=True,
    dynamic_gamma=True
)
```

#### 2. 分阶段训练
```python
if staged_training:
    # 阶段1: 低学习率预训练
    transformer.fit(X_source, X_target, lr=lr*0.1, n_epochs=n_epochs//3)
    
    # 阶段2: 正常学习率训练
    transformer.fit(X_source, X_target, lr=lr, n_epochs=n_epochs//3)
    
    # 阶段3: 低学习率精调
    transformer.fit(X_source, X_target, lr=lr*0.1, n_epochs=n_epochs//3)
```

#### 3. 动态Gamma搜索
```python
if dynamic_gamma:
    best_gamma = None
    best_mmd = float('inf')
    
    for gamma_candidate in gamma_search_values:
        transformer.gamma = gamma_candidate
        X_target_transformed = transformer.transform(X_target)
        mmd_dist = compute_mmd_kernel(X_source, X_target_transformed, gamma_candidate)
        
        if mmd_dist < best_mmd:
            best_mmd = mmd_dist
            best_gamma = gamma_candidate
    
    transformer.gamma = best_gamma
```

#### 4. 变换应用
```python
X_target_transformed = transformer.transform(X_target)
```

### Kernel PCA MMD 变换流程

#### 1. 核PCA拟合
```python
# 合并源域和目标域数据
X_combined = np.vstack([X_source, X_target])

# 拟合核PCA
kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma)
X_combined_transformed = kpca.fit_transform(X_combined)
```

#### 2. 域分离
```python
n_source = X_source.shape[0]
X_source_kpca = X_combined_transformed[:n_source]
X_target_kpca = X_combined_transformed[n_source:]
```

#### 3. 均值对齐
```python
source_mean = np.mean(X_source_kpca, axis=0)
target_mean = np.mean(X_target_kpca, axis=0)
X_target_aligned = X_target_kpca + (source_mean - target_mean)
```

### Mean-Std 对齐流程

#### 1. 统计量计算
```python
source_mean = np.mean(X_source, axis=0)
source_std = np.std(X_source, axis=0)
target_mean = np.mean(X_target, axis=0)
target_std = np.std(X_target, axis=0)
```

#### 2. 标准化和重新缩放
```python
# 标准化目标域
X_target_normalized = (X_target - target_mean) / (target_std + 1e-8)

# 重新缩放到源域分布
X_target_aligned = X_target_normalized * source_std + source_mean
```

## 🎯 使用示例和典型命令

### 基本用法
```bash
# 二分法（默认目标域B）
python scripts/run_analytical_mmd.py --model-type auto --method linear

# 二分法（目标域C）
python scripts/run_analytical_mmd.py --model-type auto --method linear --target-domain C

# 三分法（默认目标域B）
python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way

# 三分法（目标域C）
python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way --target-domain C

# 三分法 + 贝叶斯优化（目标域B）
python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way --use-bayesian-optimization

# 三分法 + 贝叶斯优化（目标域C）
python scripts/run_analytical_mmd.py --model-type auto --method linear --data-split-strategy three-way --use-bayesian-optimization --target-domain C
```

### 高级配置
```bash
# 自定义验证集比例（目标域B）
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --data-split-strategy three-way \
    --validation-split 0.7

# 自定义验证集比例（目标域C）
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --data-split-strategy three-way \
    --validation-split 0.7 \
    --target-domain C

# 贝叶斯优化参数调整（目标域B）
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --data-split-strategy three-way \
    --use-bayesian-optimization \
    --bo-n-calls 100 \
    --bo-random-state 42

# 贝叶斯优化参数调整（目标域C）
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --data-split-strategy three-way \
    --use-bayesian-optimization \
    --bo-n-calls 100 \
    --bo-random-state 42 \
    --target-domain C

# 完整配置示例（目标域B）
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --feature-type best7 \
    --method linear \
    --data-split-strategy three-way \
    --validation-split 0.7 \
    --use-bayesian-optimization \
    --bo-n-calls 50 \
    --use-class-conditional \
    --skip-cv-on-a

# 完整配置示例（目标域C）
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --feature-type best7 \
    --method linear \
    --data-split-strategy three-way \
    --validation-split 0.7 \
    --use-bayesian-optimization \
    --bo-n-calls 50 \
    --use-class-conditional \
    --skip-cv-on-a \
    --target-domain C
```

### 参数调优
```bash
# Linear方法参数调优
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --lr 0.001 \
    --n-epochs 500 \
    --lambda-reg 1e-4 \
    --use-gradient-clipping

# 使用保守预设
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --use-preset conservative
```

### 贝叶斯MMD优化用法
```bash
# 基本贝叶斯MMD优化（目标域B）
python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method linear

# 贝叶斯MMD优化（目标域C）
python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method linear --target-domain C

# 使用类条件MMD + 最佳7特征
python scripts/run_bayesian_mmd_optimization.py --model-type auto --feature-type best7 --mmd-method linear --use-class-conditional

# 核PCA MMD优化
python scripts/run_bayesian_mmd_optimization.py --model-type auto --mmd-method kpca --target-domain B

# 自定义优化参数
python scripts/run_bayesian_mmd_optimization.py \
    --model-type auto \
    --feature-type best7 \
    --mmd-method linear \
    --use-class-conditional \
    --target-domain C \
    --validation-split 0.7 \
    --n-calls 100 \
    --auto-run-mmd-after-bo

# Random Forest + Mean-Std MMD
python scripts/run_bayesian_mmd_optimization.py --model-type rf --mmd-method mean_std --n-calls 30
```

### 集成到主脚本的用法
```bash
# 在主脚本中使用贝叶斯MMD优化
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --feature-type best7 \
    --data-split-strategy three-way \
    --use-bayesian-mmd-optimization \
    --bo-n-calls 50 \
    --target-domain B

# 优化后自动运行完整实验
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --feature-type best7 \
    --data-split-strategy three-way \
    --use-bayesian-mmd-optimization \
    --auto-run-mmd-after-bo \
    --target-domain C
```

## 📁 输出结果结构

### 目录结构
```
# 默认目标域B的结果
results_cross_domain_auto_linear_all/
├── results.json                    # 主要结果文件
├── experiment_log.txt              # 实验日志
├── visualizations/                 # 可视化结果
│   ├── tsne_comparison_A_to_B.png  # t-SNE对比图
│   ├── roc_curve_B.png             # ROC曲线
│   └── ...
├── metrics/                        # 详细指标
└── models/                         # 保存的模型

# 目标域C的结果
results_cross_domain_auto_linear_all_target_C/
├── results.json                    # 主要结果文件
├── experiment_log.txt              # 实验日志
├── visualizations/                 # 可视化结果
│   ├── tsne_comparison_A_to_C.png  # t-SNE对比图
│   ├── roc_curve_C.png             # ROC曲线
│   └── ...
├── metrics/                        # 详细指标
└── models/                         # 保存的模型
```

### results.json 结构
```json
{
  "cross_validation_A": {
    "accuracy": 0.8234,
    "auc": 0.8756,
    "f1": 0.8123,
    "fold_results": [...]
  },
  "external_validation_B": {
    "without_domain_adaptation": {
      "accuracy": 0.7123,
      "auc": 0.7456,
      "f1": 0.6987
    },
    "with_domain_adaptation": {
      "accuracy": 0.7834,
      "auc": 0.8123,
      "f1": 0.7654
    }
  },
  "external_validation_C": {
    "without_domain_adaptation": {
      "accuracy": 0.6987,
      "auc": 0.7234,
      "f1": 0.6543
    },
    "with_domain_adaptation": {
      "accuracy": 0.7456,
      "auc": 0.7789,
      "f1": 0.7123
    }
  },
  "experiment_config": {
    "model_type": "auto",
    "feature_type": "all",
    "mmd_method": "linear",
    "use_class_conditional": false,
    "use_threshold_optimizer": false
  }
}
```

## 🔍 故障排除

### 常见问题

#### 1. 数据文件不存在
```
错误: 数据文件验证失败
解决: 检查 config/settings.py 中的 DATA_PATHS 配置
```

#### 2. 模型类型不可用
```
错误: 模型类型 auto 不可用
解决: 安装 autotabpfn 包或选择其他模型类型
```

#### 3. MMD计算失败
```
错误: MMD变换过程中出现数值不稳定
解决: 使用 --use-preset conservative 或调整学习率
```

#### 4. 内存不足
```
错误: CUDA out of memory
解决: 减少批大小 --batch-size 32 或使用CPU --device cpu
```

### 调试技巧

#### 1. 启用详细日志
```bash
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --log-file debug.log
```

#### 2. 监控梯度
```bash
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --monitor-gradients
```

#### 3. 快速验证
```bash
python scripts/run_analytical_mmd.py \
    --model-type auto \
    --method linear \
    --skip-cv-on-a \
    --no-visualizations
```

## 📈 性能优化建议

### 1. 计算资源优化
- 使用GPU加速: `--device cuda`
- 调整批大小: `--batch-size 128`
- 并行处理: 设置环境变量 `OMP_NUM_THREADS=4`

### 2. 实验效率优化
- 跳过交叉验证: `--skip-cv-on-a`
- 禁用可视化: `--no-visualizations`
- 使用快速预设: `--model-preset fast`

### 3. 算法收敛优化
- 使用分阶段训练: 默认启用
- 启用梯度裁剪: `--use-gradient-clipping`
- 调整学习率: `--lr 0.001`

## ✅ 修复后的可视化功能

### 现在会生成的可视化文件
```
results_cross_domain_auto_linear_best7/
├── visualizations/
│   ├── performance_comparison.png           # 性能对比图
│   ├── AUTO-MMD-LINEAR_tsne_comparison.png  # t-SNE对比图
│   ├── AUTO-MMD-LINEAR_feature_histograms.png # 特征分布直方图
│   ├── AUTO-MMD-LINEAR_statistics_table.png   # 统计表格
│   └── A_to_C/                              # 如果有数据集C
│       ├── AUTO-MMD-LINEAR_A_to_C_tsne_comparison.png
│       └── ...
```

### 可视化内容说明
1. **t-SNE对比图**: 显示域适应前后的数据分布变化
2. **特征分布直方图**: 对比源域、目标域和适应后的特征分布
3. **统计表格**: 详细的统计指标对比
4. **性能对比图**: 不同数据集上的模型性能对比

---

**文档版本**: v1.1  
**最后更新**: 2024年12月  
**维护者**: Analytical MMD A2B Feature58 项目团队

## 🔧 贝叶斯优化模块集成

### 新增功能：贝叶斯超参数优化

#### 概述
贝叶斯优化模块实现了基于目标域验证集的超参数优化，采用三分法数据划分策略：

1. **A域训练集**: 用于模型训练
2. **B域验证集**: 用于贝叶斯优化目标函数评估 (80%)
3. **B域保留测试集**: 用于最终模型泛化能力评估 (20%)

#### 使用方法

##### 1. 独立运行贝叶斯优化
```bash
# 基本用法
python scripts/run_bayesian_optimization.py --model-type auto --feature-type best7

# 高级配置
python scripts/run_bayesian_optimization.py \
    --model-type auto \
    --feature-type best7 \
    --validation-split 0.7 \
    --n-calls 50 \
    --no-categorical
```

##### 2. 集成到主工作流程
在 `run_analytical_mmd.py` 中添加贝叶斯优化选项：

```python
# 添加命令行参数
parser.add_argument('--use-bayesian-optimization', action='store_true',
                   help='使用贝叶斯优化进行超参数调优')
parser.add_argument('--bo-n-calls', type=int, default=50,
                   help='贝叶斯优化迭代次数')

# 在主函数中集成
if args.use_bayesian_optimization:
    from analytical_mmd_A2B_feature58.modeling.bayesian_optimizer import run_bayesian_optimization
    
    # 运行贝叶斯优化
    bo_results = run_bayesian_optimization(
        model_type=args.model_type,
        feature_type=args.feature_type,
        n_calls=args.bo_n_calls,
        save_path=os.path.join(save_path, 'bayesian_optimization')
    )
    
    # 使用优化后的参数
    optimized_params = bo_results['optimization_results']['best_params']
    model_kwargs.update(optimized_params)
```