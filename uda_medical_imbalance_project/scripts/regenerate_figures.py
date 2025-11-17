#!/usr/bin/env python3
"""Regenerate visualization assets for a completed PANDA analysis run.

This script reuses the serialized metrics and predictions written by
``run_complete_analysis.py`` to rebuild all PDF/PNG figures. It is useful
when minor cosmetic tweaks (for example, relabeling baselines) require a
fresh export without重新训练模型.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.analysis_visualizer import create_analysis_visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate visualization figures from an existing results folder."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Path to a results directory produced by run_complete_analysis.py. "
            "If omitted, the most recent 'complete_analysis_*' folder under "
            "uda_medical_imbalance_project/results will be used."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages (only errors will be printed).",
    )
    return parser.parse_args()


def find_latest_results_dir(base_dir: Path) -> Path | None:
    candidates = [p for p in base_dir.glob("complete_analysis_*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def load_predictions(complete_results_path: Path) -> Tuple[Dict, Dict]:
    if not complete_results_path.exists():
        return {}, {}

    payload = load_json(complete_results_path)
    source_payload = payload.get("source_domain_cv", {})
    uda_payload = payload.get("uda_methods", {})

    cv_predictions = {
        method: data.get("predictions")
        for method, data in source_payload.items()
        if isinstance(data, dict) and "predictions" in data
    }

    uda_predictions = {
        method: data.get("predictions")
        for method, data in uda_payload.items()
        if isinstance(data, dict) and "predictions" in data
    }

    return cv_predictions, uda_predictions


def regenerate_visualizations(results_dir: Path, quiet: bool) -> None:
    source_cv_file = results_dir / "source_domain_cv_results.json"
    uda_methods_file = results_dir / "uda_methods_results.json"

    if not source_cv_file.exists():
        raise FileNotFoundError(f"Missing source CV results: {source_cv_file}")
    if not uda_methods_file.exists():
        raise FileNotFoundError(f"Missing UDA methods results: {uda_methods_file}")

    if not quiet:
        print(f"📂 使用结果目录: {results_dir}")
        print("📊 加载结果数据…")

    source_cv_results = load_json(source_cv_file)
    uda_methods_results = load_json(uda_methods_file)

    cv_predictions, uda_predictions = load_predictions(
        results_dir / "complete_results.json"
    )

    if not quiet:
        print(
            f"✅ 载入 {len(cv_predictions)} 个CV预测、"
            f"{len(uda_predictions)} 个UDA预测"
        )
        print("🎨 创建可视化器并重新导出全部图像…")

    visualizer = create_analysis_visualizer(
        output_dir=str(results_dir),
        save_plots=True,
        show_plots=False,
    )

    viz_results = visualizer.generate_all_visualizations(
        cv_results=source_cv_results,
        uda_results=uda_methods_results,
        cv_predictions=cv_predictions,
        uda_predictions=uda_predictions,
    )

    if quiet:
        return

    generated = [path for path in (viz_results or {}).values() if path]
    print(f"✅ 图片重新生成完成，输出 {len(generated)} 个文件")

    key_figures = [
        "combined_heatmaps_nature",
        "decision_curve_analysis",
        "calibration_curves",
        "roc_comparison",
    ]
    print("\n🎯 关键图片检查：")
    for name in key_figures:
        for suffix in (".pdf", ".png"):
            path = results_dir / f"{name}{suffix}"
            status = "✅" if path.exists() else "❌"
            print(f"  {status} {name}{suffix}")


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    if results_dir is None:
        default_base = PROJECT_ROOT / "results"
        results_dir = find_latest_results_dir(default_base)
        if results_dir is None:
            raise SystemExit(
                "未找到任何 complete_analysis_* 结果，请先运行 run_complete_analysis.py"
            )
        if not args.quiet:
            print(f"ℹ️ 未指定结果目录，自动使用最新: {results_dir}")

    results_dir = results_dir.resolve()
    regenerate_visualizations(results_dir, quiet=args.quiet)


if __name__ == "__main__":
    main()
