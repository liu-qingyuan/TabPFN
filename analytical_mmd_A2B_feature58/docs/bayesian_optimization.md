# 贝叶斯优化模块文档

## 概述

贝叶斯优化模块实现了基于目标域验证集的超参数优化，采用三分法数据划分策略，确保模型选择和最终评估的独立性。

## 🎯 核心特性

### 1. 三分法数据划分
- **A域训练集**: 用于模型训练
- **B域验证集**: 用于贝叶斯优化目标函数评估 (80%)
- **B域保留测试集**: 用于最终模型泛化能力评估 (20%)

### 2. 支持的模型类型
- **AutoTabPFN** (`auto`): 自动化TabPFN，支持时间限制和模型数量优化
- **基础TabPFN** (`base`): 原生TabPFN，参数较少
- **Random Forest** (`rf`): 传统机器学习模型，参数丰富
- **TunedTabPFN** (`tuned`): 调优版TabPFN

### 3. 智能超参数空间
根据不同模型类型自动定义合适的超参数搜索空间：

#### AutoTabPFN 参数空间
```python
search_space = [
    Integer(30, 300, name='max_time'),        # 最大训练时间(秒)
    Integer(1, 10, name='max_models'),        # 最大模型数
    Real(0.1, 1.0, name='ensemble_size'),     # 集成大小
    Categorical(['auto', 'balanced'], name='class_weight')  # 类别权重
]
```

#### Random Forest 参数空间
```python
search_space = [
    Integer(50, 500, name='n_estimators'),    # 树的数量
    Integer(1, 20, name='max_depth'),         # 最大深度
    Integer(2, 20, name='min_samples_split'), # 最小分割样本数
    Integer(1, 10, name='min_samples_leaf'),  # 最小叶子样本数
    Real(0.1, 1.0, name='max_features'),      # 最大特征比例
    Categorical(['auto', 'balanced'], name='class_weight')
]
```

## 📋 使用方法

### 命令行使用

#### 基本用法
```bash
# AutoTabPFN模型，使用最佳7特征
python scripts/run_bayesian_optimization.py --model-type auto --feature-type best7

# Random Forest模型，30次优化迭代
python scripts/run_bayesian_optimization.py --model-type rf --n-calls 30

# 不使用类别特征
python scripts/run_bayesian_optimization.py --model-type auto --no-categorical
```

#### 高级配置
```bash
# 自定义验证集比例和优化次数
python scripts/run_bayesian_optimization.py \
    --model-type auto \
    --validation-split 0.7 \
    --n-calls 100 \
    --random-state 123

# 指定输出目录
python scripts/run_bayesian_optimization.py \
    --model-type auto \
    --output-dir ./my_optimization_results
```

### Python API 使用

#### 简单调用
```python
from analytical_mmd_A2B_feature58.modeling.bayesian_optimizer import run_bayesian_optimization

# 运行贝叶斯优化
results = run_bayesian_optimization(
    model_type='auto',
    feature_type='best7',
    use_categorical=True,
    n_calls=50
)

print(f"最佳AUC: {results['optimization_results']['best_validation_auc']:.4f}")
print(f"最佳参数: {results['optimization_results']['best_params']}")
```

#### 高级使用
```python
from analytical_mmd_A2B_feature58.modeling.bayesian_optimizer import BayesianOptimizer

# 创建优化器实例
optimizer = BayesianOptimizer(
    model_type='auto',
    feature_type='best7',
    use_categorical=True,
    validation_split=0.8,
    n_calls=50,
    save_path='./my_results'
)

# 运行完整优化流程
results = optimizer.run_complete_optimization()

# 访问详细结果
print("优化历史:", len(optimizer.optimization_results))
print("最佳参数:", optimizer.best_params)
print("最终模型:", optimizer.final_model)
```

## 📊 输出结果

### 目录结构
```
results_bayesian_optimization_auto_best7_with_categorical/
├── bayesian_optimization.log           # 详细日志
├── optimization_history.json           # 优化历史记录
├── final_evaluation.json              # 最终评估结果
└── confusion_matrices.png             # 混淆矩阵图
```

### optimization_history.json 结构
```json
{
  "best_params": {
    "max_time": 180,
    "max_models": 5,
    "ensemble_size": 0.8,
    "class_weight": "balanced"
  },
  "best_validation_auc": 0.8234,
  "optimization_history": [
    {
      "trial_id": 0,
      "params": {...},
      "validation_auc": 0.7856
    },
    ...
  ],
  "total_trials": 50
}
```

### final_evaluation.json 结构
```json
{
  "best_params": {...},
  "model_config": {
    "model_type": "auto",
    "feature_type": "best7",
    "use_categorical": true,
    "features_count": 7,
    "categorical_indices_count": 3
  },
  "data_split": {
    "train_samples": 1234,
    "validation_samples": 456,
    "holdout_samples": 114,
    "validation_split_ratio": 0.8
  },
  "performance": {
    "validation_performance": {
      "auc": 0.8234,
      "f1": 0.7856,
      "accuracy": 0.8012,
      "confusion_matrix": [[45, 8], [12, 49]]
    },
    "holdout_performance": {
      "auc": 0.8156,
      "f1": 0.7723,
      "accuracy": 0.7895,
      "confusion_matrix": [[23, 4], [6, 25]]
    }
  }
}
```

## 🔧 核心算法流程

### 1. 数据准备阶段
```python
def load_and_prepare_data(self):
    # 1. 加载A域和B域数据
    df_A = pd.read_excel(DATA_PATHS['A'])
    df_B = pd.read_excel(DATA_PATHS['B'])
    
    # 2. 特征提取
    X_A = df_A[self.features].values
    X_B = df_B[self.features].values
    
    # 3. 数据标准化 (用A域拟合scaler)
    X_A_scaled, X_B_scaled, scaler = fit_apply_scaler(X_A, X_B)
    
    # 4. B域三分法划分
    X_val, X_holdout, y_val, y_holdout = train_test_split(
        X_B_scaled, y_B, 
        train_size=0.8, 
        stratify=y_B
    )
```

### 2. 贝叶斯优化阶段
```python
def objective_function(self, params):
    # 1. 创建模型配置
    model_config = {'categorical_feature_indices': self.categorical_indices}
    model_config.update(params)
    
    # 2. 训练模型 (在A域训练集上)
    model = get_model(self.model_type, **model_config)
    model.fit(self.X_train, self.y_train)
    
    # 3. 验证集评估 (在B域验证集上)
    y_pred_proba = model.predict_proba(self.X_ext_val)[:, 1]
    auc_score = roc_auc_score(self.y_ext_val, y_pred_proba)
    
    # 4. 返回负AUC (gp_minimize最小化目标)
    return -auc_score
```

### 3. 最终评估阶段
```python
def evaluate_final_model(self):
    # 1. 使用最佳参数训练最终模型
    final_model = get_model(self.model_type, **self.best_params)
    final_model.fit(self.X_train, self.y_train)
    
    # 2. 验证集性能 (用于对比)
    val_metrics = evaluate_on_dataset(final_model, self.X_ext_val, self.y_ext_val)
    
    # 3. 保留测试集性能 (真实泛化能力)
    holdout_metrics = evaluate_on_dataset(final_model, self.X_ext_holdout, self.y_ext_holdout)
    
    return {'validation_performance': val_metrics, 'holdout_performance': holdout_metrics}
```

## 📈 性能分析

### 泛化能力评估
系统自动计算验证集和保留测试集之间的性能差距：

```python
# 计算泛化差距
auc_gap = validation_auc - holdout_auc
f1_gap = validation_f1 - holdout_f1
accuracy_gap = validation_accuracy - holdout_accuracy

# 泛化能力判断
if abs(auc_gap) < 0.05:
    print("✓ 模型泛化能力良好")
else:
    print("⚠ 模型可能存在过拟合")
```

### 优化收敛分析
通过 `optimization_history.json` 可以分析优化过程：

```python
import json
import matplotlib.pyplot as plt

# 加载优化历史
with open('optimization_history.json', 'r') as f:
    history = json.load(f)

# 绘制收敛曲线
aucs = [trial['validation_auc'] for trial in history['optimization_history']]
plt.plot(aucs)
plt.xlabel('Trial')
plt.ylabel('Validation AUC')
plt.title('Bayesian Optimization Convergence')
plt.show()
```

## ⚙️ 配置选项

### 模型特定配置

#### AutoTabPFN 推荐配置
```python
# 快速模式 (适合初步探索)
run_bayesian_optimization(
    model_type='auto',
    n_calls=30,
    validation_split=0.8
)

# 精确模式 (适合最终优化)
run_bayesian_optimization(
    model_type='auto',
    n_calls=100,
    validation_split=0.7
)
```

#### Random Forest 推荐配置
```python
# 平衡模式
run_bayesian_optimization(
    model_type='rf',
    n_calls=50,
    feature_type='all'  # RF通常能处理更多特征
)
```

### 验证集比例选择指南

| 数据集大小 | 推荐验证集比例 | 说明 |
|-----------|---------------|------|
| < 500样本 | 0.7 | 保留更多数据用于测试 |
| 500-1000样本 | 0.8 | 平衡验证和测试 |
| > 1000样本 | 0.8-0.9 | 可以用更多数据进行验证 |

## 🚀 集成到现有工作流程

### 1. 添加到主脚本
在 `run_analytical_mmd.py` 中添加贝叶斯优化选项：

```python
# 在parse_arguments()中添加
parser.add_argument('--use-bayesian-optimization', action='store_true',
                   help='使用贝叶斯优化进行超参数调优')
parser.add_argument('--bo-n-calls', type=int, default=50,
                   help='贝叶斯优化迭代次数')

# 在主函数中添加
if args.use_bayesian_optimization:
    from .modeling.bayesian_optimizer import run_bayesian_optimization
    
    bo_results = run_bayesian_optimization(
        model_type=args.model_type,
        feature_type=args.feature_type,
        n_calls=args.bo_n_calls,
        save_path=os.path.join(save_path, 'bayesian_optimization')
    )
    
    # 使用优化后的参数继续实验
    optimized_params = bo_results['optimization_results']['best_params']
    model_kwargs.update(optimized_params)
```

### 2. 与MMD域适应结合
```python
# 先进行贝叶斯优化
bo_results = run_bayesian_optimization(model_type='auto', feature_type='best7')
best_params = bo_results['optimization_results']['best_params']

# 然后使用优化参数进行MMD域适应实验
results = run_cross_domain_experiment(
    model_type='auto',
    feature_type='best7',
    mmd_method='linear',
    **best_params  # 使用优化后的参数
)
```

## 🔍 故障排除

### 常见问题

#### 1. 优化收敛缓慢
```python
# 解决方案：增加初始随机点
result = gp_minimize(
    func=objective,
    dimensions=search_space,
    n_calls=50,
    n_initial_points=15  # 增加到15个初始点
)
```

#### 2. 内存不足
```python
# 解决方案：减少并行度或使用更小的模型
optimizer = BayesianOptimizer(
    model_type='base',  # 使用更轻量的模型
    n_calls=30         # 减少迭代次数
)
```

#### 3. 验证集性能不稳定
```python
# 解决方案：使用更大的验证集比例
optimizer = BayesianOptimizer(
    validation_split=0.9,  # 增加验证集比例
    random_state=42        # 固定随机种子
)
```

## 📚 参考资料

### 相关论文
1. Mockus, J. (1974). On Bayesian methods for seeking the extremum
2. Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical bayesian optimization of machine learning algorithms

### 相关包文档
- [scikit-optimize](https://scikit-optimize.github.io/stable/)
- [AutoTabPFN](https://github.com/automl/autotabpfn)

### 项目内相关文档
- [workflow.md](./workflow.md) - 主要工作流程
- [models.md](./models.md) - 模型配置说明
- [config.md](./config.md) - 配置文件说明 