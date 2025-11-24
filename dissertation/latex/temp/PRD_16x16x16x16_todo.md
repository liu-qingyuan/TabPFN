# 16:16:16:16 数据补充 PRD（临时版）

## 背景
当前 Beamer 文件 `dissertation/latex/AI4Medicine_PolyU_Beamer_Slides/main.tex` 主要展示 8:8:8:8 集成架构的数据。为便于比较，需要同步展示 16:16:16:16 架构（旧配置）的细节，包括性能指标与可视化素材。

## 目标
在 PPT 中新增 / 完善 16:16:16:16 相关内容，确保与 8:8:8:8 数据同等清晰，方便读者对比两套架构的优劣。

## todo
- [ ] **确认数据来源**：
  - `uda_medical_imbalance_project/results/feature_sweep_analysis_58features_16_16_16_16_UDA_20250914_002222/`
    - `performance_summary.csv` 中 `best8` 行（源域 AUC 0.8346、TCA AUC 0.7024、Baseline 0.6565）。
    - `individual_results/best8/analysis_report.md`（含详细表格与 UDA 对比结果）。
  - 若需要完整流程背景，可引用 `analysis_report.md` 里的步长说明。
- [ ] **整理关键指标**：
  - 源域 AUC、无 UDA 基线、TCA AUC、提升幅度等数据，整理成表格条目。
  - 对齐 8:8:8:8 的指标口径（小数位、一致的术语）。
- [ ] **准备可视化资源**：
  - 从 `individual_results/best8/` 拷贝所需图片（如 `combined_analysis_figure.pdf`、`combined_heatmaps_nature.pdf`、`uda_methods_comparison.pdf`）。
  - 命名统一（参考 8:8:8:8 的 `best11_*` 命名，如改为 `best8_analysis_old.pdf`）。确认文件已放置在 `AI4Medicine_PolyU_Beamer_Slides/`。
- [ ] **更新 main.tex**：
  - 在对比章节补充一段描述 16:16:16:16 数据来源及其表现（可在现有“旧架构”描述中补充更完整的数值说明）。
  - 视需要新增单独的 frame 展示旧架构的可视化（若已有 `best8_*` 插图，可检查引用是否正确）。
  - 确保表格中“旧架构(16:16)”行与 CSV 数据一致；若增加更多指标（例如召回率、F1），需同步更新表头。
- [ ] **校对与验证**：
  - 编译 LaTeX（`pdflatex main.tex`）确认新增内容版式正常。
  - 复核数值、单位和颜色标记（绿色/红色高亮需与结论一致）。
  - 若图像更新，检查生成时间与版本标识以便后续追踪。

## 交付物
- 更新后的 `main.tex`（含 16:16:16:16 指标描述与表格）。
- 对应的图像文件（命名与引用一致）。
- 若有需要，补充一段 footnote 或 slide 注释说明数据来源。

## 待确认事项
- 是否需要在总结页新增对 16:16:16:16 的定量评价。
- 是否补充更多特征集（best7/best9）用于敏感性分析。
- 是否需要附带训练时间、模型复杂度等额外维度。

2025-09-14
