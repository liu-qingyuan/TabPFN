# tests/test_skip_cv.py 文档

## 概述

测试跳过数据集A交叉验证功能的简单脚本。

## 主要函数

### test_skip_cv()

```python
def test_skip_cv():
```

测试跳过交叉验证功能。

**功能:**
- 测试1: 正常运行（包含数据集A的交叉验证）
- 测试2: 跳过数据集A的交叉验证

**测试内容:**
- 使用RF模型进行快速测试
- 使用best7特征集
- 使用mean_std MMD方法
- 比较skip_cv_on_a参数的效果

## 使用示例

```bash
# 运行测试
python tests/test_skip_cv.py
```

## 测试验证

脚本会验证以下内容：
1. 正常模式下数据集A的交叉验证结果存在
2. 跳过模式下数据集A的交叉验证被跳过
3. 数据集B的外部验证结果正常生成 