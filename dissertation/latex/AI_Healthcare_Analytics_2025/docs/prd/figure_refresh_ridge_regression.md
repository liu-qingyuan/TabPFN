# PRD: Figure refresh with LASSO Regression naming

## Goal

Update the visualization code to replace "Ridge LR" labels with "LASSO LR" throughout all PANDA evaluation figures, then regenerate fresh experimental results. The updated code will generate new figures with correct LASSO regression naming while maintaining the Nature-style layout used in the dissertation.

## Approach

Instead of regenerating from frozen results, we will:
1. Update the visualization code to use correct LASSO labeling
2. Run a fresh complete analysis experiment
3. Generate new results with proper LASSO regression naming throughout

## Data Source

- Fresh experimental run using: `python scripts/run_complete_analysis.py --feature-type best8`
- Results will be generated in: `results/complete_analysis_YYYYMMDD_HHMMSS/`
- All JSON results and visualization files will be created with updated labeling

## Target Figures

The experiment will generate:

1. `combined_heatmaps_nature.pdf` (and .png counterpart)
2. `decision_curve_analysis.pdf` (and .png counterpart)
3. `calibration_curves.pdf` (and .png counterpart)
4. `roc_curves/roc_comparison.pdf` (and .png counterpart)
5. Any additional performance comparison figures

Figures will be generated in the new timestamped results directory: `results/complete_analysis_YYYYMMDD_HHMMSS/`

## Functional Requirements

- Update visualization code to display "LASSO LR" instead of "Ridge LR" in all legends, axis labels, and figure annotations
- Run complete analysis experiment to generate fresh results with updated labeling
- Maintain existing visualization pipeline styles (color maps, font sizes, DPI, panel layout) from `scripts/run_complete_analysis.py` + `preprocessing/analysis_visualizer.py`
- Generate results that can be used directly in LaTeX `Section/Results.tex` and other dissertation components
- Experimental run requires GPU access for TabPFN training and evaluation

## Acceptance Criteria

- All generated figures display "LASSO LR" instead of "Ridge LR" in legends, labels, and annotations
- Experimental run completes successfully with updated visualization code
- Generated results directory contains all expected files: JSON results, PDF/PNG figures, and analysis report
- Performance metrics remain consistent with previous runs (only labeling changes)
- Results can be directly used in dissertation LaTeX compilation

## Risks & Mitigations

- *Risk*: Visualization code hardcodes "Ridge LR" naming. *Mitigation*: systematically replace all RIDGE_LABEL references with LASSO_LABEL in analysis_visualizer.py.
- *Risk*: GPU access required for TabPFN training. *Mitigation*: ensure proper GPU environment or use TabPFN Client for hosted inference.
- *Risk*: Experimental run time may be lengthy. *Mitigation*: monitor execution and verify intermediate outputs are generated correctly.
- *Risk*: Results may vary slightly due to random factors. *Mitigation*: use fixed random seeds for reproducible comparisons.

## Implementation Plan

### Phase 1: Code Updates
1. **Modify `analysis_visualizer.py`**:
   - Change `RIDGE_LABEL = "Ridge LR"` to `LASSO_LABEL = "LASSO LR"`
   - Update `_normalize_baseline_label()` function to handle Ridge→LASSO mapping
   - Replace all `RIDGE_LABEL` references with `LASSO_LABEL` throughout the file

2. **Target locations in `analysis_visualizer.py`**:
   - Line 21: `RIDGE_LABEL = "Ridge LR"` → `LASSO_LABEL = "LASSO LR"`
   - Line 25: Function documentation update
   - Line 30: `return RIDGE_LABEL` → `return LASSO_LABEL`
   - All subsequent RIDGE_LABEL references throughout the file

### Phase 2: Fresh Experimental Run
1. **Execution Environment**: Navigate to `/Users/lqy/work/TabPFN/uda_medical_imbalance_project`
2. **Run Complete Analysis**:
   ```bash
   python scripts/run_complete_analysis.py --feature-type best8 --cv-folds 10 --random-state 42
   ```
3. **Generated Results**: New timestamped directory with all JSON results and figures

### Phase 3: Validation and Integration
1. **Label Verification**: Confirm all figures display "LASSO LR" instead of "Ridge LR"
2. **Result Integrity**: Verify performance metrics are consistent with expectations
3. **LaTeX Integration**: Update dissertation figure paths to point to new results directory

## Updated TODOs

### Phase 1: Code Updates ✅ COMPLETED
- [x] Update figure_refresh_ridge_regression.md to reflect current experimental approach
- [x] Modify `analysis_visualizer.py` - change Paper_LR display names to LASSO_LABEL
- [x] Update `run_complete_analysis.py` - change Paper_LR display names to LASSO LR
- [x] Update `QUICK_START.md` documentation to reflect LASSO LR naming
- [x] Replace all Paper_LR/Paper method references with LASSO LR throughout visualization code

### Phase 2: Fresh Experimental Run ⏳ PENDING
- [ ] Run complete analysis experiment to generate fresh results with LASSO labels
  ```bash
  cd uda_medical_imbalance_project
  python scripts/run_complete_analysis.py --feature-type best8 --cv-folds 10 --random-state 42
  ```

### Phase 3: Validation and Integration ⏳ PENDING
- [ ] Verify all generated figures show "LASSO LR" instead of "Paper LR" in legends and labels
- [ ] Check that performance metrics remain consistent (only labeling should change)
- [ ] Copy regenerated figures to dissertation directory if needed
- [ ] Update dissertation LaTeX figure paths to point to new results directory
- [ ] Final verification that dissertation compiles correctly with updated figures

### Implementation Notes:
- **Files Modified**: `analysis_visualizer.py` (6 display name mappings), `run_complete_analysis.py` (2 display name mappings), `QUICK_START.md` (1 documentation reference)
- **Internal Names Preserved**: All internal method names (`'paper_lr'`, `'Paper_LR'`, `'paper_method'`) remain unchanged for compatibility
- **Display Names Updated**: All user-facing labels now show "LASSO LR" instead of "Paper LR" or "Paper method"
