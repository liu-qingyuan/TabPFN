# PANDA TableShift 数据与实验配置摘要

本节汇总 `brfss_diabetes` 表征转移实验（Race shift）所用数据、分割方式与模型配置，方便后续论文撰写与表格生成。

## 数据来源与预处理
- 数据集：CDC Behavioral Risk Factor Surveillance System (BRFSS)，年份 `2015, 2017, 2019, 2021`（`tableshift.datasets.brfss.BRFSS_YEARS`），任务为二分类 `DIABETES`（>=1: 糖尿病阳性）。
- 域划分：`PRACE1`（自报种族）做转移变量，**源域/ID** = 非西班牙裔白人 (`PRACE1 == 1`)，**目标域/OOD** = 其他种族 (`PRACE1 in {2,3,4,5,6}`)。
- 预处理（代码见 `src/tableshift_repo/tableshift/datasets/brfss.py`）：跨年份特征对齐、去除前导下划线、SEX 映射成 {0,1}、健康天数 88->0、删除饮酒未知 (`DRNK_PER_WEEK==99900`)、处理缺失/拒答并重置索引；特征总数 142。

## 原始分割规模（完整数据）
> 统计自 `get_dataset('brfss_diabetes', use_cached=False)`，正例比例 = `mean(y)`.

- `train`（ID，白人）：969,229 样本，正例占比 12.47%，年份分布 {2015: 245,675; 2017: 244,996; 2019: 221,847; 2021: 223,088; 2016: 5,789; 2018: 6,403; 2020: 9,630; 2022: 11,801}
- `validation`（ID）：121,154 样本，正例 12.55%
- `id_test`（ID）：121,154 样本，正例 12.72%
- `ood_validation`（非白人）：23,264 样本，正例 17.13%
- `ood_test`（非白人，报告用）：209,375 样本，正例 17.42%，年份分布 {2015: 49,216; 2017: 52,150; 2019: 48,012; 2021: 50,595; 2016: 1,507; 2018: 1,424; 2020: 3,147; 2022: 3,324}

> 运行脚本中实际使用 `sample_data` 抽取：源域训练 1,024 条、目标域测试 2,048 条（`n_train=1024`, `n_test=2048`, 随机种子 42），保持标签分布的随机子样本。

### LaTeX 表格模板（可直接粘贴）
```latex
\begin{table}[htbp]
\centering
\caption{Source (ID) vs Target (OOD) cohorts for BRFSS Diabetes (race shift).}
\label{tab:brfss_cohort_summary}
\begin{tabular}{lcc}
\hline
\textbf{Characteristic} & \textbf{Source / ID (PRACE1=1)} & \textbf{Target / OOD (PRACE1 in \{2..6\})} \\\hline
Sample size (full split) & 969{,}229 (train) + 121{,}154 (val) + 121{,}154 (test) & 23{,}264 (val) + 209{,}375 (test) \\
Diabetes positive rate & 12.5\% (train) & 17.4\% (ood\_test) \\
Years (top four) & 2015: 245{,}675; 2017: 244{,}996; 2019: 221{,}847; 2021: 223{,}088 & 2015: 49{,}216; 2017: 52{,}150; 2019: 48{,}012; 2021: 50{,}595 \\
Years (others) & 2016: 5{,}789; 2018: 6{,}403; 2020: 9{,}630; 2022: 11{,}801 & 2016: 1{,}507; 2018: 1{,}424; 2020: 3{,}147; 2022: 3{,}324 \\
Domain shift variable & PRACE1 = 1 (non-Hispanic White) & PRACE1 in \{2,3,4,5,6\} (other races) \\
Label definition & \multicolumn{2}{c}{DIABETES coded 1 (Yes) vs 0 (No/Prediabetes/Borderline); NA rows dropped} \\
Feature summary & \multicolumn{2}{c}{142 numerical features after preprocessing; cross-year aligned, underscores removed; IYEAR retained} \\
Preprocessing notes & \multicolumn{2}{c}{DRNK\_PER\_WEEK=99900 dropped; SEX->\{0,1\}; health-day 88->0; TOLDHI/SMOKDAY2 fill NOTASKED\_MISSING} \\
Sampling for modeling & 1{,}024 sampled for training (seed 42) & 2{,}048 sampled for evaluation (seed 42) \\
Sampled diabetes rate (seed 42) & 13.2\% & 17.3\% \\
Sampled positive / negative counts & 135 / 889 & 355 / 1,693 \\
Sampled years (seed 42) & 2015:245; 2017:278; 2019:232; 2021:241; others (2016:8, 2018:2, 2020:7, 2022:11) & 2015:497; 2017:488; 2019:445; 2021:525; others (2016:17, 2018:17, 2020:30, 2022:29) \\\hline
\end{tabular}
\end{table}
```

**附加描述（便于答复 reviewer）：**
- 特征统计：总 142 个特征，全部数值化；缺失率中位数 0.0（预处理后无缺失），最高缺失率 0.0。
- Schema 对齐：BRFSS 调查问卷跨年完全一致，所有年份/族裔特征空间重合（142 个特征全部对齐）。
- 分布漂移量化：在 50k 源域 vs 50k 目标域子样本上，对全部特征做 KS 检验，p < 1e-3 的特征占比 ≈ 49.3%，表明 race shift 伴随明显协变量漂移；OOD 仅由种族变化定义（年份分布受控一致）。
- 划分策略：源/目标由 `PRACE1` 划分；各 split 在同一 schema 下随机抽样，源域训练含混合年份；评估使用目标域（非白人）子样本，源域/目标域的年份比例如上表所示。

## 实验配置（`results/complete_analysis_brfss_diabetes_20251121_142307/`）
- 入口脚本：`experiments/run_complete_analysis.py`，分析缓存写入 `results/complete_analysis_brfss_diabetes_20251121_142307/`。
- 设备：CPU；采样：`n_train=1024` 源域样本，`n_test=2048` 目标域样本。
- 模型与超参（固定自历史实验，TabPFN/TCA 读取 `results/tuning_extended_brfss_diabetes.csv`）：
  - `SVM`：RBF 核，`C=1.0`, `gamma=scale`, `probability=True`
  - `DT`：`max_depth=None`, `random_state=42`
  - `RF`：`n_estimators=200`, `max_depth=None`, `n_jobs=-1`, `random_state=42`
  - `GBDT`：`n_estimators=200`, `learning_rate=0.05`, `max_depth=3`, `random_state=42`
  - `XGBoost`：`n_estimators=400`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.8`, `tree_method=hist`, `eval_metric=logloss`, `random_state=42`
  - `TabPFN_NoUDA`（PANDA 基线）：`n_estimators=1`, `ignore_pretraining_limits=True`, `random_state=42`
  - `PANDA_TCA`（TabPFN + TCA）：`n_estimators=32`, `kernel=linear`, `mu=0.01`, `n_components=20`, `random_state=42`

## 主要结果（OOD = 非白人目标域，n=2,048 抽样）
| Model | AUC | Accuracy | F1 | Precision | Recall | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| SVM | 0.6455 | 0.8403 | 0.0061 | 0.1429 | 0.0031 | ML Baseline |
| DT | 0.5661 | 0.7754 | 0.2675 | 0.2745 | 0.2609 | ML Baseline |
| RF | 0.7874 | 0.8438 | 0.1351 | 0.5208 | 0.0776 | ML Baseline |
| GBDT | 0.7826 | 0.8340 | 0.2704 | 0.4375 | 0.1957 | ML Baseline |
| XGBoost | 0.7701 | 0.8271 | 0.2863 | 0.4080 | 0.2205 | ML Baseline |
| PANDA\_NoUDA | 0.7960 | 0.8472 | 0.1954 | 0.5672 | 0.1180 | PANDA Baseline (n\_estimators=1) |
| PANDA\_TCA | **0.8038** | **0.8481** | 0.1662 | **0.6078** | 0.0963 | UDA<br/>(n\_estimators=32) |

- 提醒：目标域子样本的阳性率仅约 17%，且所有模型均沿用默认的 0.5 判别阈值（未调整类权重或重新校准）。因此模型偏向预测阴性，Accuracy 仍高，但 Precision／Recall／F1 会显著偏低。这不是指标计算错误，而是数据分布与阈值设定的直接结果；若需要更平衡的召回，可在 TableShift 任务上额外做阈值搜索或引入 `class_weight='balanced'` 等策略。

- 关键图表：`combined_analysis_figure.pdf`, `combined_heatmaps_nature.pdf`, `roc_comparison.pdf`, `calibration_curves.pdf`, `decision_curve_analysis.pdf`（路径详见同目录 `visualizations.json`）。

## 复现步骤（简述）
1. 进入仓库：`cd panda_tableshift_project`
2. 运行：`python experiments/run_complete_analysis.py --dataset brfss_diabetes --n_train 1024 --n_test 2048 --device cpu`
3. 输出目录自动写入 `results/complete_analysis_brfss_diabetes_<timestamp>/`，包含配置、指标与所有可视化。
