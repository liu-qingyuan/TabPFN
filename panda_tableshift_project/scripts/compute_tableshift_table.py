from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from tableshift import get_dataset

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from experiments.run_complete_analysis import (
    canonical_name,
    get_classical_model_configs,
    instantiate_model,
    load_tabpfn_params,
    ensure_proba,
)
from src.utils import calculate_detailed_metrics, sample_data


def evaluate_on_split(
    cfg,
    model,
    X_train,
    y_train,
    X_id,
    y_id,
    X_ood,
    y_ood,
):
    X_train_np = np.asarray(X_train)
    y_train_np = np.asarray(y_train)
    X_id_np = np.asarray(X_id)
    y_id_np = np.asarray(y_id)
    X_ood_np = np.asarray(X_ood)
    y_ood_np = np.asarray(y_ood)

    if getattr(cfg, "is_adaptation", False):
        model.fit(X_train_np, y_train_np, X_target=X_ood_np)
    else:
        model.fit(X_train_np, y_train_np)

    id_proba = ensure_proba(model, X_id_np)
    ood_proba = ensure_proba(model, X_ood_np)

    id_metrics = calculate_detailed_metrics(y_id_np, id_proba)
    ood_metrics = calculate_detailed_metrics(y_ood_np, ood_proba)
    return id_metrics, ood_metrics


def build_label_map():
    return {
        "PANDA_TCA": "PANDA + TCA",
        "PANDA_NoUDA": "PANDA (No UDA)",
        "RF": "Random Forest",
        "GBDT": "Gradient Boosting (GBDT)",
        "XGBoost": "XGBoost",
        "SVM": "SVM",
        "DT": "Decision Tree",
    }


def main():
    parser = argparse.ArgumentParser(description="Compute TableShift summary metrics for Table 16.")
    parser.add_argument("--dataset", type=str, default="brfss_diabetes")
    parser.add_argument("--n_train", type=int, default=1024)
    parser.add_argument("--n_test", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="results/tableshift_summary.csv")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(args.dataset, cache_dir=str(Path(__file__).resolve().parent.parent / "data" / "raw"), initialize_data=True, use_cached=False)
    X_train, y_train, _, _ = dataset.get_pandas("train")
    X_id, y_id, _, _ = dataset.get_pandas("id_test")
    X_ood, y_ood, _, _ = dataset.get_pandas("ood_test")

    X_train_sub, y_train_sub = sample_data(X_train, y_train, args.n_train)
    X_id_sub, y_id_sub = sample_data(X_id, y_id, args.n_test)
    X_ood_sub, y_ood_sub = sample_data(X_ood, y_ood, args.n_test)

    tuning_csv = Path(__file__).resolve().parent.parent / "results" / "tuning_extended_brfss_diabetes.csv"
    tabpfn_baseline_cfg, panda_cfg = load_tabpfn_params(tuning_csv)
    classical_cfgs = get_classical_model_configs()

    label_map = build_label_map()
    rows = []

    all_cfgs = classical_cfgs + [tabpfn_baseline_cfg, panda_cfg]
    for cfg in all_cfgs:
        method_key = canonical_name(cfg.name)
        model = instantiate_model(cfg, device=args.device)
        id_metrics, ood_metrics = evaluate_on_split(
            cfg,
            model,
            X_train_sub,
            y_train_sub,
            X_id_sub,
            y_id_sub,
            X_ood_sub,
            y_ood_sub,
        )

        id_auc = id_metrics["AUC"]
        ood_auc = ood_metrics["AUC"]
        rows.append(
            {
                "Model": label_map.get(method_key, method_key),
                "ID_AUC": round(id_auc, 3),
                "OOD_AUC": round(ood_auc, 3),
                "OOD_Accuracy": round(ood_metrics["Accuracy"], 3),
                "Gap": round(id_auc - ood_auc, 3),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved TableShift summary to {output_path}")


if __name__ == "__main__":
    main()
