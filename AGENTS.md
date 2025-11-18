# Repository Guidelines

## Project Structure & Module Organization
- `src/tabpfn/` holds estimators (`classifier.py`, `regressor.py`), helpers (`utils.py`), and transformer blocks under `model/`; `tests/` mirrors the layout and `tests/reference_predictions/` stores platform-specific goldens—refresh only when behavior changes intentionally.
- `examples/` provides runnable recipes and `scripts/` centralizes maintenance helpers (dependency bounds, weight downloads); reuse these patterns instead of adding ad-hoc tools.
- `uda_medical_imbalance_project/` hosts the research stack (`scripts/`, `tests/`, `results/`); keep generated figures inside its `results/`.

## Build, Test, and Development Commands
- `python -m pip install -e ".[dev]"` then `pre-commit install` / `pre-commit run --all-files` set up the editable env and trigger ruff/mypy checks.
- `ruff check src tests` plus `ruff format src tests` enforce style; `pytest` (e.g., `pytest tests/test_classifier_interface.py -k cpu`) runs the suite; run `pyright` for stricter typing and `python scripts/download_all_models.py` to fetch weights.

## Coding Style & Naming Conventions
- Use four-space indentation, lines ≤88 chars, and include `from __future__ import annotations` at the top of new modules.
- Follow `snake_case` for functions/variables, `PascalCase` for classes, and align estimator names with sklearn counterparts (`TabPFNClassifier`).
- Provide explicit type hints—`mypy` runs with `disallow_untyped_defs`; keep docstrings concise and reserve inline comments for non-obvious tensor work.

## Testing Guidelines
- Place new tests under `tests/`, reusing fixtures instead of downloading datasets, and mark GPU-only cases so CPU runs can skip them.
- Update `tests/reference_predictions/` via `python tests/test_consistency.py --update-reference`; log seed/hardware and run `FORCE_CONSISTENCY_TESTS=1 pytest tests/test_consistency.py` when validating portability.

## UDA Medical Project Notes
- Install research extras with `pip install -r uda_medical_imbalance_project/requirements.txt`; run the full pipeline using `python uda_medical_imbalance_project/scripts/run_complete_analysis.py` or schedule `sbatch job_script.sh`.
- Feature selection lives in `predict_healthcare_RFE.py` with sets (`best7`, `best8`, etc.); TabPFN embeddings feed TCA for cross-hospital evaluation, so store outputs in timestamped `results/` folders.

## Dissertation & Publication Assets
- `dissertation/` houses Nature-style manuscripts and figures; the LaTeX source is `dissertation/latex/2025_AI4Medicine/main.tex` with references in `refs.bib` and curated figures such as `combined_heatmaps_nature.pdf`.
- Build the paper from that directory with `pdflatex main.tex && bibtex main && pdflatex main.tex`; keep PDFs in place and exclude auxiliary files from git.
- Follow Nature guidance: concise titles, ≤150-word abstract, and cite sources via BibTeX keys.

## Commit & Pull Request Guidelines
- Use commitizen (`cz commit`) so messages follow conventional commits (`feat(model): add encoder tweak`) and keep each commit scoped to one logical change.
- Before opening a PR, run `pre-commit run --all-files`, `pytest`, and integration scripts; summarize results and link issues.
