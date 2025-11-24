# PANDA-TableShift: 泛医疗跨域基准测试项目 PRD

## 🎯 项目概述

### 1.1 项目背景

TableShift (NeurIPS 2023) 是专门针对表格数据分布偏移（Distribution Shift）构建的基准测试套件。本项目旨在利用 TableShift 中定义的标准化医疗健康相关任务，进一步验证 **PANDA (TabPFN + TCA)** 框架在公开、大规模、定义明确的跨域场景下的泛化能力。

### 1.2 研究目标

- **主要目标**: 验证 PANDA 框架在 TableShift 定义的 "ID vs OOD" 严格偏移场景下的有效性。
- **具体场景**:
  1. **Diabetes (BRFSS)**: 验证在种族（Race）偏移下的公平性和鲁棒性。
  2. **Hospital Readmission**: 验证在不同入院来源（Admission Source）下的跨机构泛化能力。
- **学术目标**: 将 PANDA 的验证范围从私有小样本医疗数据（肺结节、心脏病）扩展到大规模公共基准，增强论文的说服力。

### 1.3 任务定义

| 任务名称                       | 任务类型              | 数据来源                | Shift 定义 (Source → Target)                                    | 样本量 (Est.) |
| :----------------------------- | :-------------------- | :---------------------- | :--------------------------------------------------------------- | :------------ |
| **Diabetes**             | 二分类 (是否糖尿病)   | BRFSS 调查数据          | **种族偏移**: White (Non-Hispanic) → Other Race/Ethnicity | ~250k (Total) |
| **Hospital Readmission** | 二分类 (30天内再入院) | UCI Diabetes (130 医院) | **机构偏移**: Admission Source A → Admission Source B     | ~100k (Total) |

---

## 📊 数据与偏移分析

### 2.1 Diabetes Prediction (BRFSS)

- **背景**: 基于 CDC 的行为风险因素监测系统 (BRFSS)。
- **输入特征**: 20+ 个特征，包括生活方式（吸烟、BMI）、既往病史、人口学特征。
- **Shift 挑战**: 训练集为白人数据，测试集为非白人数据。模型往往在多数群体（白人）上表现好，在少数群体上性能下降。PANDA 需要缩小这种 Performance Gap。

### 2.2 Hospital Readmission (UCI)

- **背景**: 覆盖 1999-2008 年 130 家美国医院的糖尿病患者临床护理数据。
- **输入特征**: 40+ 个特征，包括药物使用、化验结果、诊断代码等。
- **Shift 挑战**: 按照“入院来源”划分域（例如：急诊转入 vs 门诊转入 vs 其他医院转入）。这模拟了模型在不同类型医疗流程或机构间的迁移。

---

## 🏗️ PANDA-TableShift 技术架构

### 3.1 核心流程

```mermaid
graph LR
    A[TableShift Data API] --> B[数据预处理 & 采样]
    B --> C{实验分组}
    C -->|Source Domain| D[TabPFN Encoder]
    C -->|Target Domain| E[Unsupervised Features]
    D & E --> F[TCA 域适配层]
    F --> G[TabPFN / Classifier]
    G --> H[OOD 预测评估]
```

### 3.2 适配策略

由于 TableShift 数据集规模可能较大（>10k），而 TabPFN 原生针对小样本（<10k）：

1. **采样策略 (Subsampling)**: 从 Source 和 Target 中构建多个 "Support Set" (e.g., size=1024, 2048) 进行 TabPFN 推理，验证 PANDA 在**小样本跨域**场景下的优势（这是 TabPFN 的甜点区）。
2. **全量对比**: 使用 XGBoost/LightGBM 在全量数据上训练作为 "Skyline" 或强基线，对比 PANDA 在小样本下是否能逼近全量传统模型的性能。

---

## 🧪 实验设计

### 4.1 对比模型（固定集合）

- **PANDA (TabPFN + TCA)**: 适配版，`n_estimators=32`（与既有 TCA 实验一致；
  图表不单独展示 `32 vs 1` 的差异，仅作为内部配置记录）。
- **TabPFN (No TCA)**: 普通版，`n_estimators=1`。
- **传统模型基线**（参数均复用历史调参）：
  - **SVM**
  - **Decision Tree (DT)**
  - **Random Forest (RF)**
  - **GBDT**
  - **XGBoost**

### 4.2 参数与可复现性约束

- 模型超参严格复用
  `panda_tableshift_project/results/tuning_extended_brfss_diabetes.csv` 的
  最佳/已用配置；运行脚本需显式复刻读取该表，避免参数漂移。
- PANDA(TCA) 与 TabPFN(No TCA) 的 `n_estimators` 分别锁定为 32 与 1，只在
  结果元数据/表格中记录，不在图表中单独强调。
- 固定随机种子、数据拆分与预处理流程，确保与现有实验可重复对比。

### 4.3 评估指标

- **AUC (Area Under ROC)**: 主要性能指标。
- **Accuracy**: 辅助指标。
- **OOD Performance Drop**: `Source_Metric - Target_Metric` (越小越好)。
- **Adaptation Gain**: `PANDA_Metric - Baseline_Metric` (验证 TCA 的有效性)。

### 4.4 可视化与结果结构

- 目标产物：`combined_analysis_figure.pdf`、`combined_heatmaps_nature.pdf`，
  路径/命名仿照
  `uda_medical_imbalance_project/results/complete_analysis_20251118_165736/`。
- 代码复用：
  - 参考 `uda_medical_imbalance_project/scripts/run_complete_analysis.py`
    的可视化调用链。
  - 参考 `uda_medical_imbalance_project/preprocessing/analysis_visualizer.py`
    的绘图实现与版式，迁移/改写到 `panda_tableshift_project`。
- 结果结构：在 `panda_tableshift_project/results/<timestamp_run>/` 下保存
  指标表（结构化 CSV/JSON）、配置、以及组合图 PDF，与参考目录一致。

---

## 📅 实施计划 (Todo List)

### Phase 1: 环境与数据准备

- [X] **S1. 环境配置**:
  - [X] 创建 `panda_tableshift_project` 目录结构。
  - [X] 安装 `tableshift` 库 (`pip install tableshift`) 及依赖。
  - [X] 确认 TabPFN 和 Adapt 库在当前环境中可用。
- [X] **S2. 数据探索**:
  - [X] 编写脚本下载并加载 `Diabetes` 数据集，查看特征分布和 Shift 定义。
  - [X] 编写脚本下载并加载 `Hospital Readmission` 数据集。
  - [X] 确认 Source/Target 的划分逻辑。

### Phase 2: 基线实验 (Baseline)

- [X] **S3. Diabetes 基线**:
  - [X] 运行 TabPFN (No TCA, `n_estimators=1`) 在 Diabetes 任务上的评估。
  - [X] 运行传统模型基线：SVM、DT、RF、GBDT、XGBoost，参数取自
    `results/tuning_extended_brfss_diabetes.csv`（直接读取以确保一致）。
- [X] **S4. Readmission 基线**:
  - [X] 运行 TabPFN (No TCA, `n_estimators=1`) 在 Readmission 任务上的评估。
  - [X] 运行 SVM、DT、RF、GBDT、XGBoost，参数沿用同一表或同样的读取逻辑。

### Phase 3: PANDA 适配实验 (Adaptation)

- [X] **S5. PANDA 实现**:
  - [X] 将 `panda_heart_project` 中的 `PANDA_Adapter` 逻辑迁移到本项目。
  - [X] 针对 TableShift 的数据格式（Pandas/Numpy）进行接口适配。
- [X] **S6. 跨域验证 (Linear TCA)**:
  - [X] **Exp 1 (Race Shift)**: 在 Diabetes 上应用 TCA 版 TabPFN
    (`n_estimators=32`)，与 TabPFN 无 TCA (`n_estimators=1`) 及传统模型基线
    一并写入同一指标表和可视化（图中不单列 `32 vs 1`）。
  - [X] **Exp 2 (Institution Shift)**: 在 Readmission 上按相同方式记录和绘
    制。
  - [x] **结论**: Linear TCA 已完成对比（参数取自 `tuning_extended_brfss_diabetes.csv`），
    结果落盘于 `results/complete_analysis_brfss_diabetes_20251121_142307/`，当前版本不再追加调参。

### Phase 3.5: 可视化与结果固化（无额外调参）

- [x] **S7. 可视化复用与落盘**:
  - [x] 直接复用 `uda_medical_imbalance_project/scripts/run_complete_analysis.py`
    的调用链和 `preprocessing/analysis_visualizer.py` 的绘图实现，不新增调参。
  - [x] 在 `panda_tableshift_project` 内包装/调用生成同款版式的
    `combined_analysis_figure.pdf`、`combined_heatmaps_nature.pdf`，存放于
    `results/complete_analysis_brfss_diabetes_20251121_142307/`。
  - [x] 指标表（含模型、超参、配置）结构化落盘，与图像一并输出。

### Phase 4: 报告与整合

- [x] **S9. 结果汇总**:
  - [x] 生成对比表格：PANDA(TCA,32)、TabPFN(No TCA,1)、SVM/DT/RF/GBDT/XGBoost。
  - [x] 绘制参考可视化：沿用 `uda_medical_imbalance_project/scripts/run_complete_analysis.py`
    + `preprocessing/analysis_visualizer.py` 的组合图，输出
      `combined_analysis_figure.pdf` 与 `combined_heatmaps_nature.pdf`，路径为
      `panda_tableshift_project/results/complete_analysis_brfss_diabetes_20251121_142307/`。
- [ ] **S10. 文档输出**:
  - [ ] 更新论文，添加 "Experiment on Public Benchmarks" 章节。
  - [ ] 撰写 `results/tableshift_analysis_report_final.md`。

---

## 📁 目录结构规划

```text
panda_tableshift_project/
├── docs/
│   └── PANDA_TableShift_PRD.md         # 本文件
├── data/
│   └── download_tableshift.py          # 数据下载与加载脚本
├── experiments/
│   ├── run_baseline.py                 # 基线实验
│   ├── run_panda.py                    # PANDA 实验 (Linear/RBF)
│   └── tuning_panda.py                 # [New] 参数搜索脚本
├── src/
│   ├── utils.py                        # 通用工具
│   └── adapter.py                      # PANDA 适配器逻辑 (复用)
├── results/                            # 结果输出
└── requirements.txt
```
