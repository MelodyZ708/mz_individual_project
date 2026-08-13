#!/usr/bin/env python3
"""Aggregate five-repeat Top-7 evaluations on three lightswitch datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "channel_selection_results/step_f_multi_dataset_evaluation/"
    "lightswitch_5x_evaluation"
)
DEFAULT_PLAN = SCRIPT_DIR / "lightswitch_5x_plan.json"
DEFAULT_CANDIDATES = SCRIPT_DIR / "top7_candidate_plan.json"
BASELINE_KEY = "5,29,40,52"
METRICS = {
    "historical_evo_ape_mean_m": ("historical_ate_mean_cm", 100.0),
    "historical_evo_ape_rmse_m": ("historical_ate_rmse_cm", 100.0),
    "historical_evo_rpe_rmse_m": ("historical_rpe_rmse_cm", 100.0),
    "se3_ate_rmse_m": ("se3_ate_rmse_cm", 100.0),
    "se3_ate_mean_m": ("se3_ate_mean_cm", 100.0),
    "translation_rpe_rmse_m": ("translation_rpe_rmse_cm", 100.0),
    "translation_rpe_max_m": ("translation_rpe_max_cm", 100.0),
    "rotation_rpe_rmse_deg": ("rotation_rpe_rmse_deg", 1.0),
    "rotation_rpe_max_deg": ("rotation_rpe_max_deg", 1.0),
    "coverage_ratio": ("coverage_ratio", 1.0),
    "elapsed_seconds": ("runtime_seconds", 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--candidate-plan", type=Path, default=DEFAULT_CANDIDATES)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_database(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], []
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        connection.close()
        raise RuntimeError(f"SQLite integrity check failed: {path}: {integrity}")
    columns = [row[1] for row in connection.execute("PRAGMA table_info(evaluations)")]
    rows = [dict(row) for row in connection.execute("SELECT * FROM evaluations ORDER BY id")]
    connection.close()
    return rows, columns


def blank(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: blank(row.get(key)) for key in fields} for row in rows)


def metric_statistics(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def geometric_mean(values: list[float]) -> float | None:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(statistics.fmean(math.log(value) for value in values))


def validate(plan: dict[str, Any], candidate_plan: dict[str, Any]):
    datasets = plan.get("datasets", [])
    candidates = candidate_plan.get("candidates", [])
    if plan.get("protocol") != "top7_lightswitch_5x_v1":
        raise ValueError("Unexpected experiment protocol")
    if len(datasets) != 3 or plan.get("replicates_per_candidate") != 5:
        raise ValueError("Expected three datasets and five replicates")
    if len(candidates) != 7:
        raise ValueError("Expected seven candidates")
    if plan.get("planned_runs") != 105:
        raise ValueError("Expected 105 planned runs")
    return datasets, candidates


def collect(
    output_dir: Path,
    datasets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    expected = {item["candidate_key"] for item in candidates}
    candidate_by_key = {item["candidate_key"]: item for item in candidates}
    raw: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    raw_columns: list[str] = []

    for dataset in datasets:
        database = output_dir / "per_dataset" / dataset["key"] / "evaluations.sqlite3"
        rows, columns = read_database(database)
        for column in columns:
            if column not in raw_columns:
                raw_columns.append(column)
        grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in expected}
        seen: set[tuple[str, int]] = set()
        for row in rows:
            key = row["candidate_key"]
            replicate = int(row["replicate"])
            if key not in expected or not 0 <= replicate < 5:
                raise ValueError(f"Unexpected row in {dataset['key']}: {key} rep={replicate}")
            identity = (key, replicate)
            if identity in seen:
                raise ValueError(f"Duplicate row in {dataset['key']}: {identity}")
            seen.add(identity)
            grouped[key].append(row)
            enriched = {
                "dataset_order": dataset["order"],
                "dataset_key": dataset["key"],
                "family": dataset["family"],
                "source_rank": candidate_by_key[key]["source_rank"],
                "is_baseline": key == BASELINE_KEY,
            }
            enriched.update(row)
            raw.append(enriched)

        for candidate in candidates:
            key = candidate["candidate_key"]
            candidate_rows = sorted(grouped[key], key=lambda row: row["replicate"])
            passes = [row for row in candidate_rows if row["status"] == "PASS"]
            status_counts = Counter(row["status"] for row in candidate_rows)
            summary: dict[str, Any] = {
                "dataset_order": dataset["order"],
                "dataset_key": dataset["key"],
                "family": dataset["family"],
                "source_rank": candidate["source_rank"],
                "label": candidate["label"],
                "candidate_key": key,
                "channels": f"[{key}]",
                "is_baseline": key == BASELINE_KEY,
                "planned_replicates": 5,
                "completed_replicates": len(candidate_rows),
                "pass_count": len(passes),
                "pass_rate": len(passes) / 5.0,
                "failure_count": len(candidate_rows) - len(passes),
                "status_counts": json.dumps(dict(sorted(status_counts.items()))),
                "failure_frames": json.dumps(
                    [row["failure_frame_index"] for row in candidate_rows if row["status"] != "PASS"]
                ),
                "failure_timestamps": json.dumps(
                    [row["failure_timestamp"] for row in candidate_rows if row["status"] != "PASS"]
                ),
            }
            for source, (name, scale) in METRICS.items():
                values = [
                    float(row[source]) * scale
                    for row in passes
                    if row.get(source) is not None and math.isfinite(float(row[source]))
                ]
                for statistic_name, value in metric_statistics(values).items():
                    summary[f"{name}_{statistic_name}"] = value
            summaries.append(summary)
    return raw, summaries, raw_columns


def rank_and_overall(
    summaries: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for dataset in datasets:
        cells = [row for row in summaries if row["dataset_key"] == dataset["key"]]
        ordered = sorted(
            cells,
            key=lambda row: (
                -row["pass_count"],
                row["historical_ate_mean_cm_mean"]
                if row["historical_ate_mean_cm_mean"] is not None
                else math.inf,
            ),
        )
        for rank, row in enumerate(ordered, 1):
            row["reliability_aware_dataset_rank"] = rank
        baseline = next(row for row in cells if row["candidate_key"] == BASELINE_KEY)
        baseline_ate = baseline["historical_ate_mean_cm_mean"]
        for row in cells:
            candidate_ate = row["historical_ate_mean_cm_mean"]
            row["mean_ate_ratio_to_baseline"] = (
                candidate_ate / baseline_ate
                if candidate_ate is not None and baseline_ate is not None and baseline_ate > 0
                else None
            )

    overall: list[dict[str, Any]] = []
    for candidate in candidates:
        key = candidate["candidate_key"]
        cells = [row for row in summaries if row["candidate_key"] == key]
        ratios = [
            row["mean_ate_ratio_to_baseline"]
            for row in cells
            if row["mean_ate_ratio_to_baseline"] is not None
        ]
        ranks = [float(row["reliability_aware_dataset_rank"]) for row in cells]
        overall.append(
            {
                "source_rank": candidate["source_rank"],
                "label": candidate["label"],
                "candidate_key": key,
                "channels": f"[{key}]",
                "is_baseline": key == BASELINE_KEY,
                "planned_runs": 15,
                "completed_runs": sum(row["completed_replicates"] for row in cells),
                "total_pass_count": sum(row["pass_count"] for row in cells),
                "overall_pass_rate": sum(row["pass_count"] for row in cells) / 15.0,
                "datasets_with_5_of_5_pass": sum(row["pass_count"] == 5 for row in cells),
                "datasets_with_any_pass": sum(row["pass_count"] > 0 for row in cells),
                "baseline_comparable_datasets": len(ratios),
                "beats_baseline_dataset_means": sum(ratio < 1 for ratio in ratios),
                "geomean_mean_ate_ratio_to_baseline": geometric_mean(ratios),
                "mean_reliability_aware_dataset_rank": statistics.fmean(ranks),
                "fr1_pass_count": next(row["pass_count"] for row in cells if row["dataset_key"].startswith("fr1")),
                "fr2_pass_count": next(row["pass_count"] for row in cells if row["dataset_key"].startswith("fr2")),
                "fr3_pass_count": next(row["pass_count"] for row in cells if row["dataset_key"].startswith("fr3")),
                "fr1_ate_mean_cm": next(row["historical_ate_mean_cm_mean"] for row in cells if row["dataset_key"].startswith("fr1")),
                "fr2_ate_mean_cm": next(row["historical_ate_mean_cm_mean"] for row in cells if row["dataset_key"].startswith("fr2")),
                "fr3_ate_mean_cm": next(row["historical_ate_mean_cm_mean"] for row in cells if row["dataset_key"].startswith("fr3")),
            }
        )
    overall.sort(
        key=lambda row: (
            -row["total_pass_count"],
            -row["datasets_with_5_of_5_pass"],
            row["geomean_mean_ate_ratio_to_baseline"]
            if row["geomean_mean_ate_ratio_to_baseline"] is not None
            else math.inf,
            row["mean_reliability_aware_dataset_rank"],
        )
    )
    for rank, row in enumerate(overall, 1):
        row["aggregate_rank"] = rank
    return overall


def heatmap(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    field: str,
    title: str,
    filename: str,
    value_format: str,
) -> None:
    lookup = {(row["candidate_key"], row["dataset_key"]): row for row in summaries}
    matrix = np.asarray(
        [
            [lookup[(candidate["candidate_key"], dataset["key"])].get(field) for dataset in datasets]
            for candidate in candidates
        ],
        dtype=float,
    )
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(datasets)), [item["key"] for item in datasets], rotation=20, ha="right")
    axis.set_yticks(range(len(candidates)), [f"[{item['candidate_key']}]" for item in candidates])
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, "—" if np.isnan(value) else format(value, value_format), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / filename, dpi=180)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    raw: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    overall: list[dict[str, Any]],
) -> None:
    passes = sum(row.get("status") == "PASS" for row in raw)
    lines = [
        "# Three-lightswitch Top-7 × 5-repeat evaluation",
        "",
        f"- Completed: {len(raw)}/105",
        f"- PASS: {passes}; non-PASS: {len(raw) - passes}",
        "- Cell means/std/medians use PASS observations only; PASS count always uses all five planned runs.",
        "- Primary metric: keyframe evo_ape ATE mean with --align --correct_scale.",
        "",
        "## Overall reliability-first ranking",
        "",
        "| rank | channels | total PASS/15 | 5/5 datasets | fr1 | fr2 | fr3 | ATE ratio geomean |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        ratio = row["geomean_mean_ate_ratio_to_baseline"]
        lines.append(
            f"| {row['aggregate_rank']} | `{row['channels']}` | {row['total_pass_count']}/15 | "
            f"{row['datasets_with_5_of_5_pass']}/3 | {row['fr1_pass_count']}/5 | "
            f"{row['fr2_pass_count']}/5 | {row['fr3_pass_count']}/5 | "
            f"{ratio:.4f} |" if ratio is not None else
            f"| {row['aggregate_rank']} | `{row['channels']}` | {row['total_pass_count']}/15 | "
            f"{row['datasets_with_5_of_5_pass']}/3 | {row['fr1_pass_count']}/5 | "
            f"{row['fr2_pass_count']}/5 | {row['fr3_pass_count']}/5 | — |"
        )
    lines.extend([
        "",
        "Detailed per-dataset means, standard deviations, medians and failure locations are in `per_dataset_candidate_summary.csv`.",
    ])
    (output_dir / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    plan = load_json(args.experiment_plan.resolve())
    candidate_plan = load_json(args.candidate_plan.resolve())
    datasets, candidates = validate(plan, candidate_plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw, summaries, raw_columns = collect(args.output_dir, datasets, candidates)
    overall = rank_and_overall(summaries, datasets, candidates)
    metadata = ["dataset_order", "dataset_key", "family", "source_rank", "is_baseline"]
    write_csv(args.output_dir / "all_runs_raw.csv", raw, metadata + raw_columns)
    write_csv(args.output_dir / "per_dataset_candidate_summary.csv", summaries, list(summaries[0]))
    write_csv(
        args.output_dir / "candidate_overall_summary.csv",
        overall,
        ["aggregate_rank"] + [key for key in overall[0] if key != "aggregate_rank"],
    )
    heatmap(args.output_dir, summaries, datasets, candidates, "pass_count", "PASS count out of five", "pass_count_heatmap.png", ".0f")
    heatmap(args.output_dir, summaries, datasets, candidates, "historical_ate_mean_cm_mean", "Mean historical ATE over PASS runs (cm)", "ate_mean_heatmap.png", ".2f")
    heatmap(args.output_dir, summaries, datasets, candidates, "historical_ate_mean_cm_std", "Historical ATE standard deviation over PASS runs (cm)", "ate_std_heatmap.png", ".2f")
    write_summary(args.output_dir, raw, summaries, overall)
    protocol = {
        "protocol": "top7_lightswitch_5x_aggregate_v1",
        "planned_runs": 105,
        "completed_runs": len(raw),
        "pass_runs": sum(row.get("status") == "PASS" for row in raw),
        "mean_policy": "PASS observations only",
        "ranking": "total PASS desc, datasets with 5/5 PASS desc, baseline-normalized mean ATE geomean asc",
    }
    (args.output_dir / "aggregate_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"[AGGREGATE] completed={len(raw)}/105 PASS={protocol['pass_runs']} output={args.output_dir}")


if __name__ == "__main__":
    main()
