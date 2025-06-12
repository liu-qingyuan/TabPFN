# utils/logging_setup.py 文档

## 概述

统一日志配置系统，提供标准化的日志格式和级别管理。

## 主要函数

### setup_logger()

```python
def setup_logger(log_level: int = logging.INFO, log_file: Optional[str] = None, console_output: bool = True) -> logging.Logger:
```

设置日志配置。

**参数:**
- `log_level`: 日志级别，默认`logging.INFO`
- `log_file`: 日志文件路径，如果为None则不保存到文件
- `console_output`: 是否输出到控制台，默认True

**返回:**
- 配置好的logger对象

**功能:**
- 清除现有的处理器
- 设置统一的日志格式：`"%(asctime)s - %(name)s - %(levelname)s - %(message)s"`
- 支持控制台和文件双重输出
- 自动创建日志目录

**使用示例:**
```python
from analytical_mmd_A2B_feature58.utils.logging_setup import setup_logger
import logging

# 基础配置（仅控制台输出）
logger = setup_logger(log_level=logging.INFO)
logger.info("开始实验")

# 保存到文件
logger = setup_logger(
    log_level=logging.DEBUG, 
    log_file='./logs/experiment.log',
    console_output=True
)
logger.debug("调试信息")
```

### get_experiment_logger()

```python
def get_experiment_logger(experiment_name: str, base_path: str = "./logs") -> logging.Logger:
```

为特定实验创建日志器。

**参数:**
- `experiment_name`: 实验名称
- `base_path`: 日志文件基础路径，默认"./logs"

**返回:**
- 配置好的logger对象

**功能:**
- 自动生成带时间戳的日志文件名
- 格式：`{experiment_name}_{timestamp}.log`
- 同时输出到控制台和文件

**使用示例:**
```python
from analytical_mmd_A2B_feature58.utils.logging_setup import get_experiment_logger

# 创建实验专用日志器
logger = get_experiment_logger('mmd_experiment', './logs')
logger.info("实验开始")
# 日志将保存到 ./logs/mmd_experiment_20241201_143022.log

# 自定义日志目录
logger = get_experiment_logger('cross_domain_test', './results/logs')
```

## 日志格式

所有日志使用统一格式：
```
2024-12-01 14:30:22 - root - INFO - 实验开始
2024-12-01 14:30:23 - analytical_mmd_A2B_feature58.modeling.model_selector - DEBUG - 创建AutoTabPFN模型
```

格式说明：
- `%(asctime)s`: 时间戳（YYYY-MM-DD HH:MM:SS）
- `%(name)s`: 日志器名称
- `%(levelname)s`: 日志级别
- `%(message)s`: 日志消息 