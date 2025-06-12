# pytest.ini 文档

## 概述

pytest测试框架配置文件，定义测试发现规则和运行选项。

## 配置项说明

### 测试发现配置

- **testpaths = tests**: 测试文件搜索路径
- **python_files = test_*.py**: 测试文件命名模式
- **python_classes = Test***: 测试类命名模式
- **python_functions = test_***: 测试函数命名模式

### 运行选项

- **addopts = -v --tb=short**: 
  - `-v`: 详细输出模式
  - `--tb=short`: 简短的错误回溯信息

### 警告过滤

- **ignore::DeprecationWarning**: 忽略弃用警告
- **ignore::PendingDeprecationWarning**: 忽略待弃用警告

## 使用方法

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_mmd_basic.py

# 运行特定测试函数
pytest tests/test_mmd_basic.py::test_compute_mmd

# 显示详细输出
pytest -v

# 显示测试覆盖率
pytest --cov=analytical_mmd_A2B_feature58
``` 