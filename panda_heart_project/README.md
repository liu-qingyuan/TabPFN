# PANDA-Heart: Heart Disease Cross-Center Diagnosis with TabPFN + TCA

🫀 **PANDA-Heart** is a framework for cross-center heart disease diagnosis integrating TabPFN with **Transfer Component Analysis (TCA)**.

## 🎯 Project Status
**Current Phase**: ✅ Phase 2 Completed (TCA Experiments)

- **Models**: PANDA-TCA vs 6 Baselines (TabPFN_Only, LR, XGBoost, RF, SVM, KNN)
- **Evaluation**: Single-center & Cross-domain (6 pairs)
- **Metric**: **AUC** (Medical Standard)
- **Result**: PANDA-TCA achieved **99.0% performance retention** across domains.

## 📂 Simplified Structure

```
panda_heart_project/
├── analyze_results.py   # 📊 Generate analysis charts & reports (PDF/MD)
├── run_experiments.py   # 🧪 Run TCA experiments (Single-center & Cross-domain)
├── data/                # 💾 Data storage and loader
│   ├── loader.py        # Data loader class
│   ├── download_data.py # Data downloader script
│   └── processed/       # Processed UCI datasets
├── models/              # 🧠 Model definitions
│   ├── panda_adapt_adapter.py # PANDA-TCA Adapter
│   └── baseline_models.py     # Baseline models factory
├── results/             # 📈 Output directory (PDFs, CSVs, Reports)
├── docs/                # 📚 Documentation & PRD
└── tests/               # ✅ Unit tests
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Data** (If not already present)
   ```bash
   python panda_heart_project/data/download_data.py
   ```

3. **Run Experiments**
   ```bash
   python panda_heart_project/run_experiments.py
   ```
   *Outputs results to `results/tca_only_results_[timestamp]/`*

4. **Generate Analysis**
   ```bash
   python panda_heart_project/analyze_results.py
   ```
   *Generates `results/panda_heart_tca_analysis.pdf` and `results/tca_only_analysis_report.md`*

## 📊 Key Results

- **Single-Center AUC**: ~0.996
- **Cross-Domain AUC**: ~0.986
- **Retention Rate**: 99.0%

*For detailed specifications, see `docs/PANDA_Heart_Final_PRD.md`.*