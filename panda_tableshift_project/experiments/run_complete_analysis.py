"""End-to-end analysis runner for PANDA TableShift project.

Replicates the visualization pipeline used in
`uda_medical_imbalance_project` by reusing its `analysis_visualizer`
module. Generates structured metrics for the fixed model set:
PANDA(TabPFN+TCA, n_estimators=32), TabPFN(no TCA, n_estimators=1),
SVM, Decision Tree, Random Forest, GBDT, and XGBoost, then produces
`combined_analysis_figure.pdf` and `combined_heatmaps_nature.pdf` under
`results/<timestamped_run>/`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tableshift import get_dataset
from xgboost import XGBClassifier

# Local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "src"))
from src.adapter import PANDAAdapter
from src.utils import calculate_detailed_metrics, sample_data


# ------------------------------ Configs ------------------------------ #


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any]
    is_adaptation: bool = False


def load_tabpfn_params(tuning_path: Path) -> Tuple[ModelConfig, ModelConfig]:
    """Load TabPFN baseline and PANDA(TCA) params from tuning CSV.

    Baseline: method is NaN/None, n_estimators=1.
    PANDA(TCA): method == 'TCA', n_estimators=32.
    """

    if not tuning_path.exists():
        baseline = ModelConfig(
            name="tabpfn_notca",
            params={"n_estimators": 1, "random_state": 42},
            is_adaptation=False,
        )
        panda = ModelConfig(
            name="panda_tca",
            params={
                "n_estimators": 32,
                "kernel": "linear",
                "mu": 0.01,
                "n_components": 20,
                "random_state": 42,
            },
            is_adaptation=True,
        )
        return baseline, panda

    df = pd.read_csv(tuning_path)
    baselines = df[(df["method"].isna()) & (df["n_estimators"] == 1)]
    panda_rows = df[(df["method"].fillna("").str.upper() == "TCA") & (df["n_estimators"] == 32)]

    if baselines.empty:
        raise ValueError("No TabPFN baseline row with n_estimators=1 found in tuning CSV")
    if panda_rows.empty:
        raise ValueError("No TCA row with n_estimators=32 found in tuning CSV")

    best_base = baselines.sort_values(by="AUC", ascending=False).iloc[0]
    best_panda = panda_rows.sort_values(by="AUC", ascending=False).iloc[0]

    baseline = ModelConfig(
        name="tabpfn_notca",
        params={
            "n_estimators": int(best_base["n_estimators"]),
            "random_state": 42,
        },
        is_adaptation=False,
    )

    panda = ModelConfig(
        name="panda_tca",
        params={
            "n_estimators": int(best_panda["n_estimators"]),
            "kernel": best_panda.get("kernel", "linear"),
            "mu": float(best_panda.get("mu", 0.01)),
            "n_components": int(best_panda.get("n_components", 20)),
            "random_state": 42,
        },
        is_adaptation=True,
    )

    return baseline, panda


def get_classical_model_configs(random_state: int = 42) -> List[ModelConfig]:
    """Fixed classical baselines.

    These are kept stable to mirror prior experiments; adjust only if a
    documented tuning file is provided.
    """

    return [
        ModelConfig("svm", {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True}),
        ModelConfig("dt", {"max_depth": None, "random_state": random_state}),
        ModelConfig("rf", {"n_estimators": 200, "max_depth": None, "n_jobs": -1, "random_state": random_state}),
        ModelConfig("gbdt", {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": random_state}),
        ModelConfig("xgboost", {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8, "tree_method": "hist", "eval_metric": "logloss", "random_state": random_state}),
    ]


# ------------------------------ Modeling ----------------------------- #


def instantiate_model(cfg: ModelConfig, device: str = "cpu"):
    name = cfg.name.lower()
    p = cfg.params

    if name.startswith("svm"):
        return SVC(**p)
    if name == "dt":
        return DecisionTreeClassifier(**p)
    if name == "rf":
        return RandomForestClassifier(**p)
    if name == "gbdt":
        return GradientBoostingClassifier(**p)
    if name == "xgboost":
        return XGBClassifier(**p)
    if name.startswith("tabpfn"):
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(
            device=device,
            n_estimators=int(p.get("n_estimators", 1)),
            ignore_pretraining_limits=True,
            random_state=p.get("random_state", 42),
        )
    if name.startswith("panda"):
        return PANDAAdapter(
            adaptation_method="TCA",
            device=device,
            n_estimators=int(p.get("n_estimators", 32)),
            kernel=p.get("kernel", "linear"),
            mu=p.get("mu", 0.01),
            n_components=p.get("n_components", 20),
            random_state=p.get("random_state", 42),
        )

    raise ValueError(f"Unknown model config: {cfg.name}")


def ensure_proba(model, X):
    """Return probability estimates for binary classification."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1:
            return np.vstack([1 - proba, proba]).T
        return proba
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        probs = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - probs, probs]).T
    raise ValueError("Model does not support probability estimates")


def canonical_name(name: str) -> str:
    lname = name.lower()
    if lname.startswith("svm"):
        return "SVM"
    if lname == "dt":
        return "DT"
    if lname == "rf":
        return "RF"
    if lname == "gbdt":
        return "GBDT"
    if lname == "xgboost":
        return "XGBoost"
    if lname.startswith("tabpfn"):
        return "PANDA_NoUDA"
    if lname.startswith("panda"):
        return "PANDA_TCA"
    return name


def evaluate_model(cfg: ModelConfig, model, X_train, y_train, X_test, y_test, X_target=None):
    """Fit model and compute metrics + predictions."""
    X_train_np = np.asarray(X_train)
    y_train_np = np.asarray(y_train)
    X_test_np = np.asarray(X_test)
    X_target_np = np.asarray(X_target) if X_target is not None else None

    if cfg.is_adaptation and hasattr(model, "fit"):
        model.fit(X_train_np, y_train_np, X_target=X_target_np)
    else:
        model.fit(X_train_np, y_train_np)

    proba = ensure_proba(model, X_test_np)
    metrics = calculate_detailed_metrics(np.asarray(y_test), proba)
    return metrics, proba


def cross_val_evaluate(cfg: ModelConfig, X, y, n_splits: int = 5, device: str = "cpu"):
    """Perform stratified CV on source domain to mimic source CV curves."""
    X_np = np.asarray(X)
    y_np = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    y_true_all: List[float] = []
    y_proba_all: List[float] = []
    fold_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in skf.split(X_np, y_np):
        model = instantiate_model(cfg, device=device)
        model.fit(X_np[train_idx], y_np[train_idx])
        proba = ensure_proba(model, X_np[val_idx])
        metrics = calculate_detailed_metrics(y_np[val_idx], proba)
        fold_metrics.append(metrics)

        probs_1d = proba[:, 1] if proba.ndim > 1 else proba
        y_true_all.extend(y_np[val_idx].tolist())
        y_proba_all.extend(probs_1d.tolist())

    summary = build_summary_from_list(fold_metrics)
    predictions = {"y_true": y_true_all, "y_pred_proba": y_proba_all}
    return summary, predictions


# --------------------------- Visualization -------------------------- #


def load_visualizer_factory() -> Any:
    """Dynamically load create_analysis_visualizer from UDA project."""

    repo_root = Path(__file__).resolve().parent.parent.parent
    viz_path = repo_root / "uda_medical_imbalance_project" / "preprocessing" / "analysis_visualizer.py"
    if not viz_path.exists():
        raise FileNotFoundError(f"Visualizer module not found at {viz_path}")

    spec = importlib.util.spec_from_file_location("analysis_visualizer", viz_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.create_analysis_visualizer


def build_summary(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "auc_mean": metrics["AUC"],
        "auc_std": 0.0,
        "accuracy_mean": metrics["Accuracy"],
        "accuracy_std": 0.0,
        "f1_mean": metrics["F1"],
        "f1_std": 0.0,
        "precision_mean": metrics["Precision"],
        "precision_std": 0.0,
        "recall_mean": metrics["Recall"],
        "recall_std": 0.0,
    }


def build_summary_from_list(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["AUC", "Accuracy", "F1", "Precision", "Recall"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in metrics_list]
        summary[f"{k.lower()}_mean"] = float(np.mean(vals))
        summary[f"{k.lower()}_std"] = float(np.std(vals))
    return summary


def save_metrics_table(rows: List[Dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def save_tableshift_results(
    cv_results: Dict[str, Dict[str, Any]],
    uda_results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Persist the summary table used in the manuscript (ID vs OOD)."""

    label_map = {
        "PANDA_TCA": "PANDA + TCA",
        "PANDA_NoUDA": "PANDA (No UDA)",
        "RF": "Random Forest",
        "GBDT": "Gradient Boosting (GBDT)",
        "XGBoost": "XGBoost",
        "SVM": "SVM",
        "DT": "Decision Tree",
    }

    rows: List[Dict[str, Any]] = []
    for key in ["PANDA_TCA", "PANDA_NoUDA", "RF", "GBDT", "XGBoost", "SVM", "DT"]:
        ood = uda_results.get(key)
        if ood is None:
            continue
        if key == "PANDA_TCA":
            source = cv_results.get("PANDA_NoUDA")
        else:
            source = cv_results.get(key)
        if source is None:
            continue

        id_auc = source["summary"]["auc_mean"]
        ood_auc = ood["auc"]
        rows.append(
            {
                "Model": label_map.get(key, key),
                "ID_AUC": round(id_auc, 3),
                "OOD_AUC": round(ood_auc, 3),
                "OOD_Accuracy": round(ood["accuracy"], 3),
                "Gap": round(id_auc - ood_auc, 3),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


# ------------------------------ Runner ------------------------------ #


def run_analysis(dataset: str, n_train: int, n_test: int, device: str, output_root: Path) -> Path:
    cache_dir = output_root.parent / "data" / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Data
    dset = get_dataset(dataset, cache_dir=str(cache_dir), initialize_data=True, use_cached=False)
    X_train, y_train, _, _ = dset.get_pandas("train")
    X_test, y_test, _, _ = dset.get_pandas("ood_test")

    X_train_sub, y_train_sub = sample_data(X_train, y_train, n_train)
    if n_test and n_test < len(X_test):
        X_test_sub, y_test_sub = sample_data(X_test, y_test, n_test)
    else:
        X_test_sub, y_test_sub = X_test, y_test

    tuning_csv = output_root.parent / "results" / "tuning_extended_brfss_diabetes.csv"
    tabpfn_baseline_cfg, panda_cfg = load_tabpfn_params(tuning_csv)
    classical_cfgs = get_classical_model_configs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"complete_analysis_{dataset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, Any]] = []
    cv_results: Dict[str, Dict[str, Any]] = {}
    uda_results: Dict[str, Dict[str, Any]] = {}
    cv_predictions: Dict[str, Dict[str, Any]] = {}
    uda_predictions: Dict[str, Dict[str, Any]] = {}
    source_rows: List[Dict[str, Any]] = []

    create_visualizer = load_visualizer_factory()
    visualizer = create_visualizer(output_dir=str(run_dir), save_plots=True, show_plots=False)

    # Evaluate classical baselines
    for cfg in classical_cfgs + [tabpfn_baseline_cfg]:
        method_key = canonical_name(cfg.name)

        # 1) Source CV for source panel
        cv_summary, cv_pred = cross_val_evaluate(cfg, X_train_sub, y_train_sub, device=device)
        cv_results[method_key] = {"summary": cv_summary, "config": cfg.params}
        cv_predictions[method_key] = cv_pred

        source_rows.append(
            {
                "dataset": dataset,
                "split": "source_cv",
                "model": method_key,
                "AUC": cv_summary["auc_mean"],
                "Accuracy": cv_summary["accuracy_mean"],
                "F1": cv_summary["f1_mean"],
                "Precision": cv_summary["precision_mean"],
                "Recall": cv_summary["recall_mean"],
                "params": json.dumps(cfg.params),
            }
        )

        # 2) Target (OOD) evaluation for target/UDA panel
        model = instantiate_model(cfg, device=device)
        metrics, proba = evaluate_model(cfg, model, X_train_sub, y_train_sub, X_test_sub, y_test_sub)

        metrics_rows.append(
            {
                "dataset": dataset,
                "split": "ood_test",
                "model": method_key,
                **metrics,
                "params": json.dumps(cfg.params),
            }
        )

        baseline_category = "ml_baseline"
        if method_key == "PANDA_NoUDA":
            baseline_category = "panda_baseline"

        target_y_true = y_test_sub.tolist()
        target_y_pred_proba = proba[:, 1].tolist() if proba.ndim > 1 else proba.tolist()

        uda_results[method_key] = {
            "summary": build_summary(metrics),
            "config": cfg.params,
            "is_baseline": True,
            "baseline_category": baseline_category,
            "y_true": target_y_true,
            "y_pred_proba": target_y_pred_proba,
            "auc": metrics["AUC"],
            "accuracy": metrics["Accuracy"],
            "f1": metrics["F1"],
            "precision": metrics["Precision"],
            "recall": metrics["Recall"],
        }
        uda_predictions[method_key] = {
            "y_true": target_y_true,
            "y_pred_proba": target_y_pred_proba,
            "is_baseline": True,
            "baseline_category": baseline_category,
        }

    # Evaluate PANDA(TCA)
    panda_model = instantiate_model(panda_cfg, device=device)
    panda_metrics, panda_proba = evaluate_model(
        panda_cfg, panda_model, X_train_sub, y_train_sub, X_test_sub, y_test_sub, X_target=X_test_sub
    )

    panda_key = canonical_name(panda_cfg.name)
    uda_results[panda_key] = {
        "summary": build_summary(panda_metrics),
        "config": panda_cfg.params,
        "is_baseline": False,
        "baseline_category": "UDA",
        "y_true": y_test_sub.tolist(),
        "y_pred_proba": panda_proba[:, 1].tolist() if panda_proba.ndim > 1 else panda_proba.tolist(),
        "auc": panda_metrics["AUC"],
        "accuracy": panda_metrics["Accuracy"],
        "f1": panda_metrics["F1"],
        "precision": panda_metrics["Precision"],
        "recall": panda_metrics["Recall"],
    }
    uda_predictions[panda_key] = {
        "y_true": y_test_sub.tolist(),
        "y_pred_proba": panda_proba[:, 1].tolist() if panda_proba.ndim > 1 else panda_proba.tolist(),
    }

    metrics_rows.append(
        {
            "dataset": dataset,
            "split": "ood_test",
            "model": panda_key,
            **panda_metrics,
            "params": json.dumps(panda_cfg.params),
        }
    )

    # Save metrics table
    combined_metrics = metrics_rows + source_rows
    save_metrics_table(combined_metrics, run_dir / "metrics_summary.csv")
    save_metrics_table(source_rows, run_dir / "source_cv_summary.csv")
    save_tableshift_results(cv_results, uda_results, run_dir / "tableshift_results.csv")

    # Save config snapshot
    config_snapshot = {
        "dataset": dataset,
        "n_train": n_train,
        "n_test": n_test,
        "device": device,
        "timestamp": timestamp,
        "models": [canonical_name(cfg.name) for cfg in classical_cfgs]
        + [canonical_name(tabpfn_baseline_cfg.name), canonical_name(panda_cfg.name)],
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    # Visualizations
    viz_results = visualizer.generate_all_visualizations(
        cv_results=cv_results,
        uda_results=uda_results,
        cv_predictions=cv_predictions,
        uda_predictions=uda_predictions,
    )

    # Persist visualization paths for reference
    (run_dir / "visualizations.json").write_text(json.dumps(viz_results, indent=2), encoding="utf-8")

    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete analysis and generate combined visualizations.")
    parser.add_argument("--dataset", type=str, default="brfss_diabetes", help="TableShift dataset name")
    parser.add_argument("--n_train", type=int, default=1024, help="Number of training samples (source domain)")
    parser.add_argument("--n_test", type=int, default=2048, help="Number of test samples for evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device for TabPFN (cpu/cuda)")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "results"),
        help="Root directory for saving analysis outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    start = time.time()
    run_dir = run_analysis(
        dataset=args.dataset,
        n_train=args.n_train,
        n_test=args.n_test,
        device=args.device,
        output_root=output_root,
    )
    elapsed = time.time() - start
    print(f"✅ Analysis finished in {elapsed:.1f}s. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
