#!/usr/bin/env python3
"""Aggregate 7 QueensCAMP degradations × 8 configurations × 3 repeats."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

_CACHE = Path(tempfile.gettempdir()) / "mz_queenscamp_aggregate_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "channel_selection_results/step_h_queenscamp_degradation_evaluation/"
    "three_repeats"
)
DEFAULT_PLAN = SCRIPT_DIR / "queenscamp_3x_plan.json"
DEFAULT_CANDIDATES = SCRIPT_DIR / "top7_plus_gray_candidate_plan.json"
HISTORICAL_BASELINE_KEY = "5,29,40,52"
GRAY_KEY = "gray"
REPLICATES = 3
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
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--candidate-plan", type=Path, default=DEFAULT_CANDIDATES)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate(
    plan: dict[str, Any], candidate_plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = plan.get("datasets", [])
    candidates = candidate_plan.get("candidates", [])
    if plan.get("protocol") != "top7_plus_gray_queenscamp_3x_v1":
        raise ValueError("Unexpected experiment protocol")
    if candidate_plan.get("protocol") != "top7_plus_gray_candidate_plan_v1":
        raise ValueError("Unexpected candidate protocol")
    if len(datasets) != 7 or plan.get("dataset_count") != 7:
        raise ValueError("Expected exactly seven degradation datasets")
    if len(candidates) != 8 or plan.get("configuration_count") != 8:
        raise ValueError("Expected exactly eight configurations")
    if plan.get("replicates_per_configuration") != REPLICATES:
        raise ValueError("Expected three replicates per configuration")
    if plan.get("planned_runs") != 168:
        raise ValueError("Expected 168 planned runs")
    if plan.get("timeout_seconds_per_run") != 500:
        raise ValueError("Expected a 500-second timeout")
    dataset_keys = [item["key"] for item in datasets]
    candidate_keys = [item["candidate_key"] for item in candidates]
    if len(set(dataset_keys)) != len(dataset_keys):
        raise ValueError("Duplicate dataset key")
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("Duplicate candidate key")
    if HISTORICAL_BASELINE_KEY not in candidate_keys or GRAY_KEY not in candidate_keys:
        raise ValueError("Historical CNN baseline or gray control is missing")
    return datasets, candidates


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
    rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM evaluations ORDER BY replicate, id"
        )
    ]
    connection.close()
    return rows, columns


def blank(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {key: blank(row.get(key)) for key in fields} for row in rows
        )


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


def display_key(key: str) -> str:
    return "gray" if key == GRAY_KEY else f"[{key}]"


def collect(
    output_dir: Path,
    datasets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    expected = {item["candidate_key"] for item in candidates}
    candidate_by_key = {item["candidate_key"]: item for item in candidates}
    raw: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    authoritative_columns: list[str] | None = None

    for dataset in datasets:
        database = output_dir / "per_dataset" / dataset["key"] / "evaluations.sqlite3"
        rows, columns = read_database(database)
        if columns:
            if authoritative_columns is None:
                authoritative_columns = columns
            elif columns != authoritative_columns:
                raise ValueError(f"Database schema differs for {dataset['key']}")
        grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in expected}
        seen: set[tuple[str, int]] = set()
        for row in rows:
            key = str(row["candidate_key"])
            replicate = int(row["replicate"])
            if key not in expected or not 0 <= replicate < REPLICATES:
                raise ValueError(
                    f"Unexpected row in {dataset['key']}: {key} rep={replicate}"
                )
            identity = (key, replicate)
            if identity in seen:
                raise ValueError(f"Duplicate row in {dataset['key']}: {identity}")
            seen.add(identity)
            grouped[key].append(row)
            enriched = {
                "dataset_order": dataset["order"],
                "dataset_key": dataset["key"],
                "degradation": dataset["degradation"],
                "source_rank": candidate_by_key[key]["source_rank"],
                "is_historical_cnn_baseline": key == HISTORICAL_BASELINE_KEY,
                "is_gray_control": key == GRAY_KEY,
            }
            enriched.update(row)
            raw.append(enriched)

        for candidate in candidates:
            key = candidate["candidate_key"]
            candidate_rows = sorted(grouped[key], key=lambda row: row["replicate"])
            passes = [row for row in candidate_rows if row["status"] == "PASS"]
            status_counts = Counter(str(row["status"]) for row in candidate_rows)
            summary: dict[str, Any] = {
                "dataset_order": dataset["order"],
                "dataset_key": dataset["key"],
                "degradation": dataset["degradation"],
                "source_rank": candidate["source_rank"],
                "label": candidate["label"],
                "candidate_key": key,
                "configuration": display_key(key),
                "is_historical_cnn_baseline": key == HISTORICAL_BASELINE_KEY,
                "is_gray_control": key == GRAY_KEY,
                "planned_replicates": REPLICATES,
                "completed_replicates": len(candidate_rows),
                "pass_count": len(passes),
                "pass_rate_planned": len(passes) / REPLICATES,
                "failure_count_completed": len(candidate_rows) - len(passes),
                "missing_replicates": REPLICATES - len(candidate_rows),
                "status_counts": json.dumps(dict(sorted(status_counts.items()))),
                "failure_frames": json.dumps(
                    [
                        row["failure_frame_index"]
                        for row in candidate_rows
                        if row["status"] != "PASS"
                    ]
                ),
                "failure_timestamps": json.dumps(
                    [
                        row["failure_timestamp"]
                        for row in candidate_rows
                        if row["status"] != "PASS"
                    ]
                ),
            }
            for source, (name, scale) in METRICS.items():
                values = [
                    float(row[source]) * scale
                    for row in passes
                    if row.get(source) is not None
                    and math.isfinite(float(row[source]))
                ]
                for statistic_name, value in metric_statistics(values).items():
                    summary[f"{name}_{statistic_name}"] = value
            summaries.append(summary)
    return raw, summaries, authoritative_columns or []


def rank_and_summarise(
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
                row["source_rank"],
            ),
        )
        for rank, row in enumerate(ordered, start=1):
            row["reliability_aware_dataset_rank"] = rank
        baseline = next(
            row for row in cells if row["candidate_key"] == HISTORICAL_BASELINE_KEY
        )
        baseline_ate = baseline["historical_ate_mean_cm_mean"]
        for row in cells:
            candidate_ate = row["historical_ate_mean_cm_mean"]
            row["mean_ate_ratio_to_historical_cnn_baseline"] = (
                candidate_ate / baseline_ate
                if candidate_ate is not None
                and baseline_ate is not None
                and baseline_ate > 0
                else None
            )

    overall: list[dict[str, Any]] = []
    planned_per_configuration = len(datasets) * REPLICATES
    for candidate in candidates:
        key = candidate["candidate_key"]
        cells = [row for row in summaries if row["candidate_key"] == key]
        ratios = [
            row["mean_ate_ratio_to_historical_cnn_baseline"]
            for row in cells
            if row["mean_ate_ratio_to_historical_cnn_baseline"] is not None
        ]
        ranks = [float(row["reliability_aware_dataset_rank"]) for row in cells]
        completed = sum(row["completed_replicates"] for row in cells)
        passed = sum(row["pass_count"] for row in cells)
        overall.append(
            {
                "source_rank": candidate["source_rank"],
                "label": candidate["label"],
                "candidate_key": key,
                "configuration": display_key(key),
                "is_historical_cnn_baseline": key == HISTORICAL_BASELINE_KEY,
                "is_gray_control": key == GRAY_KEY,
                "planned_runs": planned_per_configuration,
                "completed_runs": completed,
                "total_pass_count": passed,
                "overall_pass_rate_planned": passed / planned_per_configuration,
                "datasets_with_3_of_3_pass": sum(row["pass_count"] == 3 for row in cells),
                "datasets_with_any_pass": sum(row["pass_count"] > 0 for row in cells),
                "baseline_comparable_datasets": len(ratios),
                "beats_historical_baseline_dataset_means": sum(
                    ratio < 1.0 for ratio in ratios
                ),
                "geomean_mean_ate_ratio_to_historical_baseline": geometric_mean(ratios),
                "mean_reliability_aware_dataset_rank": statistics.fmean(ranks),
            }
        )
    overall.sort(
        key=lambda row: (
            -row["total_pass_count"],
            -row["datasets_with_3_of_3_pass"],
            row["geomean_mean_ate_ratio_to_historical_baseline"]
            if row["geomean_mean_ate_ratio_to_historical_baseline"] is not None
            else math.inf,
            row["mean_reliability_aware_dataset_rank"],
        )
    )
    for rank, row in enumerate(overall, start=1):
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
            [
                lookup[(candidate["candidate_key"], dataset["key"])].get(field)
                for dataset in datasets
            ]
            for candidate in candidates
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(12.0, 6.5))
    masked = np.ma.masked_invalid(matrix)
    image = axis.imshow(masked, aspect="auto", cmap="viridis_r")
    axis.set_xticks(
        range(len(datasets)),
        [item["degradation"] for item in datasets],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(
        range(len(candidates)),
        [display_key(item["candidate_key"]) for item in candidates],
    )
    axis.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                "—" if np.isnan(value) else format(value, value_format),
                ha="center",
                va="center",
                fontsize=8,
            )
    if np.any(np.isfinite(matrix)):
        figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def format_optional(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def write_summary(
    output_dir: Path,
    raw: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
) -> None:
    passes = sum(row.get("status") == "PASS" for row in raw)
    lines = [
        "# QueensCAMP degradation robustness: Top-7 + gray, three repeats",
        "",
        f"- Completed: {len(raw)}/168",
        f"- PASS: {passes}; completed non-PASS: {len(raw) - passes}",
        "- Every dataset/configuration cell has three planned independent runs.",
        "- Metric means/std/medians use PASS observations only; PASS count uses all planned runs.",
        "- Primary accuracy metric: keyframe evo_ape ATE mean with `--align --correct_scale`.",
        "- Cross-degradation accuracy is compared by the per-dataset ATE ratio to `[5,29,40,52]`; raw ATE scales are not pooled.",
        "",
        "## Reliability-first aggregate ranking",
        "",
        "| rank | configuration | PASS/21 | datasets 3/3 | datasets any PASS | beats CNN baseline | ATE ratio geomean |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['aggregate_rank']} | `{row['configuration']}` | "
            f"{row['total_pass_count']}/21 | {row['datasets_with_3_of_3_pass']}/7 | "
            f"{row['datasets_with_any_pass']}/7 | "
            f"{row['beats_historical_baseline_dataset_means']}/"
            f"{row['baseline_comparable_datasets']} | "
            f"{format_optional(row['geomean_mean_ate_ratio_to_historical_baseline'], 4)} |"
        )
    lines.extend(["", "## Best configuration within each degradation", ""])
    lines.extend(
        [
            "| degradation | configuration | PASS | mean ATE/cm | reliability-aware rank |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for dataset in datasets:
        cells = [row for row in summaries if row["dataset_key"] == dataset["key"]]
        best = min(cells, key=lambda row: row["reliability_aware_dataset_rank"])
        lines.append(
            f"| {dataset['degradation']} | `{best['configuration']}` | "
            f"{best['pass_count']}/3 | "
            f"{format_optional(best['historical_ate_mean_cm_mean'])} | "
            f"{best['reliability_aware_dataset_rank']} |"
        )
    lines.extend(
        [
            "",
            "Detailed means, standard deviations, medians, RPE metrics and failure locations are stored in `per_dataset_configuration_summary.csv`.",
        ]
    )
    (output_dir / "aggregate_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    plan = load_json(args.experiment_plan.resolve())
    candidate_plan = load_json(args.candidate_plan.resolve())
    datasets, candidates = validate(plan, candidate_plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, summaries, raw_columns = collect(output_dir, datasets, candidates)
    overall = rank_and_summarise(summaries, datasets, candidates)
    metadata = [
        "dataset_order",
        "dataset_key",
        "degradation",
        "source_rank",
        "is_historical_cnn_baseline",
        "is_gray_control",
    ]
    write_csv(output_dir / "all_runs_raw.csv", raw, metadata + raw_columns)
    write_csv(
        output_dir / "per_dataset_configuration_summary.csv",
        summaries,
        list(summaries[0]),
    )
    write_csv(
        output_dir / "configuration_overall_summary.csv",
        overall,
        ["aggregate_rank"] + [key for key in overall[0] if key != "aggregate_rank"],
    )
    heatmap(
        output_dir,
        summaries,
        datasets,
        candidates,
        "pass_count",
        "PASS count out of three",
        "pass_count_heatmap.png",
        ".0f",
    )
    heatmap(
        output_dir,
        summaries,
        datasets,
        candidates,
        "historical_ate_mean_cm_mean",
        "Mean historical keyframe ATE over PASS runs (cm)",
        "ate_mean_heatmap.png",
        ".2f",
    )
    heatmap(
        output_dir,
        summaries,
        datasets,
        candidates,
        "historical_ate_mean_cm_std",
        "Historical keyframe ATE standard deviation over PASS runs (cm)",
        "ate_std_heatmap.png",
        ".2f",
    )
    write_summary(output_dir, raw, summaries, overall, datasets)
    protocol = {
        "protocol": "top7_plus_gray_queenscamp_3x_aggregate_v1",
        "planned_runs": 168,
        "completed_runs": len(raw),
        "pass_runs": passes if (passes := sum(row.get("status") == "PASS" for row in raw)) else 0,
        "mean_policy": "PASS observations only",
        "primary_metric": "keyframe evo_ape ATE mean with --align --correct_scale",
        "ranking": "total PASS desc, datasets with 3/3 PASS desc, baseline-normalized ATE geomean asc",
    }
    (output_dir / "aggregate_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[AGGREGATE] completed={len(raw)}/168 PASS={protocol['pass_runs']} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
