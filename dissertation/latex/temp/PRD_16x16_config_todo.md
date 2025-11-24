# 16:16:16:16（best7）数据补充 PRD（临时版）

## 背景
目前幻灯片重点呈现 8:8:8:8 集成架构。旧架构 16:16:16:16（两种基础配置×16 模型）在 best7 特征集下表现更佳，需要在 `dissertation/latex/AI4Medicine_PolyU_Beamer_Slides/main.tex` 中同步展示，方便与新架构对比。

## 目标
在 PPT 中补充 16:16:16:16 + best7 结果的数值与图像，使观众清楚旧架构的性能、优势与不足。

## todo
- [ ] **确认数据来源**：
  - 主目录：`uda_medical_imbalance_project/results/feature_sweep_analysis_58features_16_16_16_16_UDA_20250914_002222/`
  - 关键文件：
    - `performance_summary.csv` 中 `best7` 行（源域 AUC 0.8285、无 UDA 基线 0.6948、TCA AUC 0.7013、提升 +0.0065）。
    - `individual_results/best7/analysis_report.md`（含详细表格与 UDA 对比结果）。
    - `individual_results/best7/` 下的图像输出（`combined_analysis_figure.pdf/png`、`combined_heatmaps_nature.*`、`uda_methods_comparison.*` 等）。
- [ ] **整理呈现内容**：
  - 提炼源域 AUC、目标域 AUC、无 UDA 基线、TCA 提升倍数/数值等指标。
  - 若需要扩展表格，可加入 Accuracy、F1 以保持与 8:8:8:8 对齐。
- [ ] **准备图像资源**：
  - 将 `best7` 相关的 PDF/PNG 拷贝到 `AI4Medicine_PolyU_Beamer_Slides/`，命名与 LaTeX 一致（如 `best7_analysis_old.pdf`、`best7_heatmaps_old.pdf`）。
  - 确认图像尺寸与现有幻灯片版式匹配。
- [ ] **更新 main.tex**：
  - 在“集成配置架构对比”或新增 frame 中引用 best7 数据及图像。
  - 调整表格条目，使 “旧架构(16:16)” 行引用 best7 数值；必要时说明该行的特征集选择。
  - 在说明文本中增加 best7 表现总结（例如：TCA 提升有限、但 baseline 较高等）。
- [ ] **校对与编译**：
  - 运行 `pdflatex main.tex` 检查版式。
  - 核对数值与 CSV/Markdown 一致，检查颜色标注（green/red）。
  - 确认新图像正确显示，文件路径无误。

## 交付物
- 更新后的 `main.tex`，展示 16:16:16:16 best7 指标与图像。
- 对应的图像文件放置于 Beamer 目录，命名统一。
- 视需要补充脚注或引用，注明数据来源目录。

## 待确认事项
- 是否在总结页补充一句关于 16:16 架构在 best7 下的结论。
- 是否需要对比其他特征集（best6/best8）以说明趋势。
- 是否加入训练耗时或模型数量等额外信息。

2025-09-14
