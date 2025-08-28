# Feature Performance PDF Generation Guide

## 概述
该脚本用于根据特征数量评估结果生成高质量的学术论文级PDF图表。

## 功能特性
- 📊 生成标准性能对比图 (DPI: 1200)
- 🎨 生成综合分析图 (DPI: 900) 
- 📈 多种性能指标可视化 (Accuracy, AUC, F1-Score)
- ⚡ 训练时间复杂度分析
- 🎯 最佳特征数量推荐
- 📋 详细统计总结

## 环境要求

### Python版本
- Python 3.7+

### 必要依赖包
```bash
pip install pandas matplotlib numpy seaborn
```

或使用conda:
```bash
conda install pandas matplotlib numpy seaborn
```

## 使用方法

### 基本用法
```bash
cd uda_medical_imbalance_project/scripts
python generate_feature_performance_pdf.py
```

### 输入文件
脚本会自动读取:
```
results/feature_number_evaluation/feature_number_comparison.csv
```

### 输出文件
生成的文件保存在:
```
results/feature_number_evaluation/
├── performance_comparison.pdf          # 标准版 (DPI: 1200)
├── performance_comparison.png          # PNG预览
├── performance_comparison_comprehensive.pdf  # 综合版 (DPI: 900)
└── performance_comparison_comprehensive.png  # PNG预览
```

## CSV文件格式要求

输入的CSV文件应包含以下列:
- `n_features`: 特征数量
- `mean_accuracy`: 平均准确率
- `std_accuracy`: 准确率标准差  
- `mean_auc`: 平均AUC
- `std_auc`: AUC标准差
- `mean_f1`: 平均F1分数
- `std_f1`: F1分数标准差
- `mean_time`: 平均训练时间
- `std_time`: 训练时间标准差
- `mean_acc_0`: 类别0平均准确率
- `mean_acc_1`: 类别1平均准确率

## 输出图表说明

### 标准版图表 (4子图)
1. **(A) Accuracy vs. Number of Features** - 准确率随特征数变化
2. **(B) AUC vs. Number of Features** - AUC随特征数变化  
3. **(C) F1-Score vs. Number of Features** - F1分数随特征数变化
4. **(D) Training Time vs. Number of Features** - 训练时间随特征数变化

### 综合版图表 (6子图)
1. **Performance Metrics vs. Number of Features** - 主要性能指标综合趋势
2. **Class-Specific Accuracy** - 类别特异性准确率
3. **Training Time Complexity** - 训练时间复杂度
4. **Performance Stability** - 性能稳定性分析 (变异系数)
5. **Optimal Feature Number Distribution** - 最佳特征数量分布

## 学术用途

### 论文应用建议
- 标准版适合插入论文正文
- 综合版适合补充材料或详细分析
- 所有图表DPI≥900，符合主流学术期刊要求

### 图表特性
- ✅ 高分辨率 (900-1200 DPI)
- ✅ 学术期刊友好的色彩方案
- ✅ 清晰的标注和图例
- ✅ 专业的排版和字体
- ✅ PDF矢量格式，支持无损缩放

## 故障排除

### 常见问题
1. **ModuleNotFoundError**: 安装缺少的依赖包
2. **FileNotFoundError**: 检查CSV文件路径是否正确
3. **Empty DataFrame**: 检查CSV文件格式和数据完整性

### 自定义选项
可以修改脚本中的DPI设置和颜色方案来满足特定需求。

## 版本信息
- 版本: 1.0
- 作者: TabPFN+TCA Medical Research Team
- 日期: 2025-08-22