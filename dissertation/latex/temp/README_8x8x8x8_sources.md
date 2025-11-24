# PANDA 8:8:8:8 Slide Data Notes

This memo captures how the 8:8:8:8 architecture results in `AI4Medicine_PolyU_Beamer_Slides/main.tex` were produced and which artifacts feed each slide.

## 1. Source runs
- Feature sweep: `uda_medical_imbalance_project/results/feature_sweep_analysis_58features_8_8_8_8_UDA_20250913_192120/`
  - `performance_summary.csv` → Best11 row shows source AUC 0.8444, baseline AUC 0.6554, TCA AUC 0.7056, improvement +0.0502.
  - `individual_results/best11/analysis_report.md` tabulates the same metrics and provides UDA comparisons used in slide tables.
  - Figures under `individual_results/best11/` (`combined_analysis_figure.pdf`, `combined_heatmaps_nature.pdf`, etc.) were copied into the Beamer `source` directory and renamed (`best11_*`).
- Full analysis bundle: `uda_medical_imbalance_project/results/complete_analysis_20250913_185344_8:8:8:8_UDA/`
  - `analysis_report.md` confirms the end-to-end run (best10 features) and supplies overall context for pipeline diagrams and comparison panels.

## 2. Slide integration
- `main.tex` section “集成配置架构对比：16:16 vs 8:8:8:8” cites the Best11 metrics pulled from the feature sweep run (lines 170–205).
- The comparative table references the old 16:16 results (from `feature_sweep_analysis_58features_16_16_16_16_UDA_20250914_002222/individual_results/best8/analysis_report.md`) and the new 8:8:8:8 sweep.
- Visual slides (`best11_analysis_new.pdf`, `best11_heatmaps_new.pdf`, `best11_uda_comparison.png`, etc.) originate from the `individual_results/best11` export on 2025‑09‑13.

## 3. Reproduction steps
1. Run the feature sweep script for the 8:8:8:8 architecture (see project notebooks) → produces the `feature_sweep_analysis_58features_8_8_8_8_UDA_20250913_192120` directory.
2. Optionally execute a full pipeline run via `python uda_medical_imbalance_project/scripts/run_complete_analysis.py --feature-set best11` (config tracked in `complete_analysis_20250913_185344_8:8:8:8_UDA`).
3. Copy the relevant PNG/PDF exports from `individual_results/best11/` into `dissertation/latex/AI4Medicine_PolyU_Beamer_Slides/` and update `main.tex` to reference them.
4. Quote the metrics directly from `performance_summary.csv` and `analysis_report.md` when editing the LaTeX tables.

## 4. Validation
- Cross-check numbers by opening the CSV/Markdown reports in each result folder.
- Confirm slide assets align with source exports (dimensions and legends) before running `pdflatex`.

## 5. Change history
- Created: 2025-09-14
