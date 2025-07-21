# 基线特征表格生成报告

生成时间: 2025-07-07 10:18:11

## 配置信息

- 特征集: best8
- 输出目录: /home/24052432g/TabPFN/uda_medical_imbalance_project/results/baseline_characteristics_20250707_101808

## 数据集信息

- Train Cohort (A): 295 样本
- Test Cohort (B): 190 样本
- 总特征数: 10
- 连续特征数: 6
- 类别特征数: 3

## 输出文件

- CSV 表格: /home/24052432g/TabPFN/uda_medical_imbalance_project/results/baseline_characteristics_20250707_101808/baseline_characteristics_best8.csv
- XLSX 表格: /home/24052432g/TabPFN/uda_medical_imbalance_project/results/baseline_characteristics_20250707_101808/baseline_characteristics_best8.xlsx
- 可视化图表: /home/24052432g/TabPFN/uda_medical_imbalance_project/results/baseline_characteristics_20250707_101808/baseline_characteristics_all_features_best8.png

## 使用说明

### 表格格式说明
- **连续变量**: 显示为 `均值 ± 标准差`
- **类别变量**: 显示为 `频数 (百分比%)`
- **Train Cohort (A)**: 源域数据集（AI4health）
- **Test Cohort (B)**: 目标域数据集（HenanCancerHospital）

### 特征映射
特征代码到原始名称的映射基于 `data/Feature_Ranking_with_Original_Names.csv`

### 引用格式
此表格适用于临床研究论文，格式参考《Lancet》《JAMA》《NEJM》等期刊的基线特征表格标准。