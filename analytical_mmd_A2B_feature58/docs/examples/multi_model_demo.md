# examples/multi_model_demo.py 文档

## 概述

多模型功能演示脚本，展示如何使用新的多模型支持和跨域实验功能。

## 主要函数

### setup_logging()

```python
def setup_logging():
```

设置日志配置。

**返回:**
- logger对象

### demo_model_selector()

```python
def demo_model_selector():
```

演示模型选择器功能。

**功能:**
- 检查可用模型类型
- 演示特征配置
- 测试每种可用的模型类型
- 进行RF模型的完整训练和预测测试

### demo_config_management()

```python
def demo_config_management():
```

演示配置管理功能。

**功能:**
- 展示特征配置（all, best7）
- 展示模型配置（auto, base, rf）
- 显示可用预设配置
- 展示最佳7特征详情

### demo_cross_domain_runner()

```python
def demo_cross_domain_runner():
```

演示跨域实验运行器功能。

**功能:**
- 创建CrossDomainExperimentRunner实例
- 展示配置参数
- 创建模拟数据集进行测试

### create_mock_datasets()

```python
def create_mock_datasets(temp_dir, logger):
```

创建模拟数据集文件。

**参数:**
- `temp_dir`: 临时目录路径
- `logger`: 日志对象

### demo_command_line_usage()

```python
def demo_command_line_usage():
```

演示命令行使用方法。

### main()

```python
def main():
```

主函数，运行所有演示。

## 使用示例

```bash
# 运行完整演示
python examples/multi_model_demo.py

# 查看演示输出
python examples/multi_model_demo.py 2>&1 | tee demo_output.log
```

## 演示内容

1. **模型选择器演示**: 展示如何使用不同类型的模型
2. **配置管理演示**: 展示特征和模型配置的使用
3. **跨域实验演示**: 展示跨域实验运行器的基本用法
4. **命令行使用演示**: 展示脚本的命令行接口 