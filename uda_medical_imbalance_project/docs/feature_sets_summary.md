# 特征集总结文档

## 📋 RFE预筛选特征集说明

本项目使用的特征集都是基于**递归特征消除（RFE）**预筛选的最优特征组合，无需额外的特征选择步骤。

## 🎯 可用特征集

### BEST7特征集
```python
BEST_7_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature56', 
    'Feature42', 'Feature39', 'Feature43'
]
```
- **总特征数**：7个
- **类别特征**：2个（Feature63, Feature46）
- **数值特征**：5个
- **类别特征比例**：28.6%

### BEST8特征集
```python
BEST_8_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature61',
    'Feature56', 'Feature42', 'Feature39', 'Feature43'
]
```
- **总特征数**：8个
- **类别特征**：2个（Feature63, Feature46）
- **数值特征**：6个
- **类别特征比例**：25.0%

### BEST9特征集
```python
BEST_9_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature61',
    'Feature56', 'Feature42', 'Feature39', 'Feature43', 'Feature48'
]
```
- **总特征数**：9个
- **类别特征**：2个（Feature63, Feature46）
- **数值特征**：7个
- **类别特征比例**：22.2%

### BEST10特征集
```python
BEST_10_FEATURES = [
    'Feature63', 'Feature2', 'Feature46', 'Feature61', 
    'Feature56', 'Feature42', 'Feature39', 'Feature43', 'Feature48', 'Feature5'
]
```
- **总特征数**：10个
- **类别特征**：3个（Feature63, Feature46, Feature5）
- **数值特征**：7个
- **类别特征比例**：30.0%

### ALL特征集
- **总特征数**：58个（全部选定特征）
- **类别特征**：20个
- **数值特征**：38个
- **类别特征比例**：34.5%

## 🔍 特征集构建逻辑

特征集按照RFE重要性递增构建：
1. **BEST7** → **BEST8**：添加 Feature61
2. **BEST8** → **BEST9**：添加 Feature48
3. **BEST9** → **BEST10**：添加 Feature5（类别特征）

## 📊 类别特征分布

### 核心类别特征
- **Feature63**：在所有特征集中都存在的核心类别特征
- **Feature46**：在所有特征集中都存在的核心类别特征

### 扩展类别特征
- **Feature5**：仅在BEST10特征集中出现的额外类别特征

## 🧪 验证方法

运行以下命令验证特征集配置：
```bash
cd uda_medical_imbalance_project
python tests/test_categorical_features.py
```

## 💡 使用建议

1. **医疗数据分析**：推荐从BEST7开始，逐步增加到BEST10进行对比
2. **计算资源有限**：使用BEST7或BEST8特征集
3. **追求最佳性能**：使用BEST10特征集
4. **研究特征重要性**：使用ALL特征集进行全面分析

## 🔧 技术实现

- 特征集在 `data/loader.py` 中预定义
- 自动识别类别特征索引
- 支持混合数据类型处理
- 无需额外的特征选择步骤 