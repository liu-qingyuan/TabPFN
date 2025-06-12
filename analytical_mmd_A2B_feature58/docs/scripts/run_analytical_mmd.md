# scripts/run_analytical_mmd.py 文档

## 概述

主要实验脚本，用于运行基于MMD的域适应实验。支持多种模型类型和MMD方法。

## 命令行参数

### 实验模式
- `--mode`: 实验模式 ('standard', 'cross-domain')

### 模型参数
- `--model-type`: 模型类型 ('auto', 'tuned', 'base', 'rf')
- `--model-preset`: 模型预设 ('fast', 'balanced', 'accurate')

### 特征参数
- `--feature-type`: 特征类型 ('all', 'best7')

### MMD方法
- `--method`: MMD方法 ('linear', 'kpca', 'mean_std')
- `--compare-all`: 比较所有MMD方法

### MMD参数
- `--gamma`: RBF核参数
- `--lr`: 学习率
- `--n-epochs`: 训练轮数
- `--batch-size`: 批大小

## 主要函数

### parse_arguments()

```python
def parse_arguments():
```

解析命令行参数。

### setup_experiment_logging()

```python
def setup_experiment_logging(log_file: Optional[str] = None):
```

设置实验日志。

### prepare_model_kwargs()

```python
def prepare_model_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
```

准备模型参数。

### prepare_mmd_kwargs()

```python
def prepare_mmd_kwargs(args: argparse.Namespace, method: str) -> Dict[str, Any]:
```

准备MMD参数。

### run_standard_experiment()

```python
def run_standard_experiment(args: argparse.Namespace, logger: logging.Logger):
```

运行标准MMD实验。

### run_cross_domain_experiment_mode()

```python
def run_cross_domain_experiment_mode(args: argparse.Namespace, logger: logging.Logger):
```

运行跨域实验模式。

## 使用示例

### 基础用法

```bash
# AutoTabPFN + Linear MMD
python scripts/run_analytical_mmd.py --model-type auto --method linear

# 使用最佳7特征
python scripts/run_analytical_mmd.py --model-type auto --method linear --feature-type best7

# 跨域实验模式
python scripts/run_analytical_mmd.py --mode cross-domain --model-type auto --method linear
```

### 高级用法

```bash
# 类条件MMD + 阈值优化
python scripts/run_analytical_mmd.py --model-type auto --method linear --use-class-conditional --use-threshold-optimizer

# 比较所有方法
python scripts/run_analytical_mmd.py --compare-all --model-type auto

# 自定义参数
python scripts/run_analytical_mmd.py --model-type auto --method linear --gamma 0.5 --lr 0.001 --n-epochs 500
``` 