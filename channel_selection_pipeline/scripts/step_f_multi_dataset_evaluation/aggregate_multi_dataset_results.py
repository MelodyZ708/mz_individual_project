#!/usr/bin/env python3
"""Aggregate the eight-dataset Top-7 evaluation without mixing raw ATE scales.

Each dataset has an independent Step-E-compatible SQLite database.  This tool
preserves every raw field, creates a 56-cell scorecard, computes per-dataset
ranks and baseline-relative ratios, and produces a failure-aware cross-dataset
summary.
"""

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

_CACHE = Path(tempfile.gettempdir()) / "mz_step_f_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "channel_selection_results/step_f_multi_dataset_evaluation"
DEFAULT_DATASET_PLAN = SCRIPT_DIR / "dataset_plan.json"
DEFAULT_CANDIDATE_PLAN = SCRIPT_DIR / "top7_candidate_plan.json"
BASELINE_KEY = "5,29,40,52"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Aggregate resumable Top-7 multi-dataset evaluation outputs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-plan", type=Path, default=DEFAULT_DATASET_PLAN)
    parser.add_argument("--candidate-plan", type=Path, default=DEFAULT_CANDIDATE_PLAN)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_plans(
    dataset_payload: dict[str, Any], candidate_payload: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = dataset_payload.get("datasets", [])
    candidates = candidate_payload.get("candidates", [])
    if dataset_payload.get("protocol") != "top7_multi_dataset_generalisation_v1":
        raise ValueError("Unexpected dataset protocol")
    if candidate_payload.get("protocol") != "top7_multi_dataset_candidate_plan_v1":
        raise ValueError("Unexpected candidate protocol")
    if len(datasets) != 8 or len(candidates) != 7:
        raise ValueError("Expected eight datasets and seven candidates")
    if dataset_payload.get("timeout_seconds_per_run") != 500:
        raise ValueError("Dataset plan timeout must be exactly 500 seconds")
    if len({item["key"] for item in datasets}) != 8:
        raise ValueError("Duplicate dataset key")
    if len({item["candidate_key"] for item in candidates}) != 7:
        raise ValueError("Duplicate candidate key")
    if BASELINE_KEY not in {item["candidate_key"] for item in candidates}:
        raise ValueError("Historical baseline is absent")
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
    rows = [dict(row) for row in connection.execute("SELECT * FROM evaluations ORDER BY id")]
    connection.close()
    return rows, columns


def cm(value: Any) -> float | None:
    return None if value is None else float(value) * 100.0


def safe_mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def geometric_mean(values: list[float]) -> float | None:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(statistics.fmean(math.log(value) for value in values))


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


def collect(
    output_dir: Path,
    datasets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    candidate_by_key = {item["candidate_key"]: item for item in candidates}
    expected_keys = set(candidate_by_key)
    all_raw: list[dict[str, Any]] = []
    scorecard: list[dict[str, Any]] = []
    raw_columns: list[str] = []
    for dataset in datasets:
        database = output_dir / "per_dataset" / dataset["key"] / "evaluations.sqlite3"
        rows, columns = read_database(database)
        if columns and not raw_columns:
            raw_columns = columns
        elif columns and set(columns) != set(raw_columns):
            missing = sorted(set(raw_columns) - set(columns))
            extra = sorted(set(columns) - set(raw_columns))
            raise ValueError(
                f"Database schema fields differ for {dataset['key']}: "
                f"missing={missing}, extra={extra}"
            )
        by_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            if row["replicate"] != 0:
                raise ValueError(f"Unexpected replicate in {dataset['key']}: {row['replicate']}")
            key = row["candidate_key"]
            if key not in expected_keys or key in by_key:
                raise ValueError(f"Unexpected/duplicate candidate in {dataset['key']}: {key}")
            by_key[key] = row
            enriched = {
                "dataset_order": dataset["order"],
                "dataset_key": dataset["key"],
                "dataset_family": dataset["family"],
                "dataset_condition": dataset["condition"],
                "dataset_directory": dataset["directory_name"],
                "source_rank": candidate_by_key[key]["source_rank"],
                "is_baseline": key == BASELINE_KEY,
            }
            enriched.update(row)
            all_raw.append(enriched)

        valid_passes = [
            row
            for row in rows
            if row["status"] == "PASS" and row["historical_evo_ape_mean_m"] is not None
        ]
        valid_passes.sort(key=lambda row: row["historical_evo_ape_mean_m"])
        rank_by_key = {row["candidate_key"]: rank for rank, row in enumerate(valid_passes, 1)}
        baseline = by_key.get(BASELINE_KEY)
        baseline_ate = (
            baseline["historical_evo_ape_mean_m"]
            if baseline is not None
            and baseline["status"] == "PASS"
            and baseline["historical_evo_ape_mean_m"] is not None
            else None
        )
        baseline_se3 = (
            baseline["se3_ate_rmse_m"]
            if baseline is not None
            and baseline["status"] == "PASS"
            and baseline["se3_ate_rmse_m"] is not None
            else None
        )
        for candidate in candidates:
            key = candidate["candidate_key"]
            row = by_key.get(key)
            status = row["status"] if row is not None else "NOT_RUN"
            historical_mean = (
                row["historical_evo_ape_mean_m"]
                if row is not None and row["status"] == "PASS"
                else None
            )
            se3_rmse = (
                row["se3_ate_rmse_m"]
                if row is not None and row["status"] == "PASS"
                else None
            )
            ratio = (
                historical_mean / baseline_ate
                if historical_mean is not None and baseline_ate is not None and baseline_ate > 0
                else None
            )
            se3_ratio = (
                se3_rmse / baseline_se3
                if se3_rmse is not None and baseline_se3 is not None and baseline_se3 > 0
                else None
            )
            scorecard.append(
                {
                    "dataset_order": dataset["order"],
                    "dataset_key": dataset["key"],
                    "dataset_family": dataset["family"],
                    "dataset_condition": dataset["condition"],
                    "expected_matched_frames": dataset["expected_matched_frames"],
                    "source_rank": candidate["source_rank"],
                    "label": candidate["label"],
                    "candidate_key": key,
                    "channels": f"[{key}]",
                    "is_baseline": key == BASELINE_KEY,
                    "status": status,
                    "reason": row["reason"] if row is not None else "",
                    "dataset_rank": rank_by_key.get(key),
                    "coverage_ratio": row["coverage_ratio"] if row is not None else None,
                    "associated_poses": row["associated_poses"] if row is not None else None,
                    "elapsed_seconds": row["elapsed_seconds"] if row is not None else None,
                    "historical_evo_ape_mean_cm": cm(historical_mean),
                    "historical_evo_ape_rmse_cm": (
                        cm(row["historical_evo_ape_rmse_m"])
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "historical_evo_rpe_rmse_cm": (
                        cm(row.get("historical_evo_rpe_rmse_m"))
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "ate_ratio_to_baseline": ratio,
                    "ate_improvement_vs_baseline_percent": (
                        (1.0 - ratio) * 100.0 if ratio is not None else None
                    ),
                    "se3_ate_rmse_cm": cm(se3_rmse),
                    "se3_ate_mean_cm": (
                        cm(row["se3_ate_mean_m"])
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "se3_ate_max_cm": (
                        cm(row["se3_ate_max_m"])
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "se3_rmse_ratio_to_baseline": se3_ratio,
                    "translation_rpe_rmse_cm": (
                        cm(row["translation_rpe_rmse_m"])
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "translation_rpe_max_cm": (
                        cm(row["translation_rpe_max_m"])
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "rotation_rpe_rmse_deg": (
                        row["rotation_rpe_rmse_deg"]
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "rotation_rpe_max_deg": (
                        row["rotation_rpe_max_deg"]
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "rotation_ape_rmse_deg": (
                        row["rotation_ape_rmse_deg"]
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "allframe_sim3_mean_cm": (
                        cm(row["allframe_sim3_mean_m"])
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "allframe_sim3_scale": (
                        row["allframe_sim3_scale"]
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "keyframe_sim3_mean_cm": (
                        cm(row["keyframe_sim3_mean_m"])
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "keyframe_sim3_scale": (
                        row["keyframe_sim3_scale"]
                        if row is not None and row["status"] == "PASS"
                        else None
                    ),
                    "photo_mse_median": row["photo_mse_median"] if row is not None else None,
                    "photo_mse_p95": row["photo_mse_p95"] if row is not None else None,
                    "photo_mse_nonfinite_count": (
                        row["photo_mse_nonfinite_count"] if row is not None else None
                    ),
                    "valid_ratio_min": row["valid_ratio_min"] if row is not None else None,
                    "h_cond_max": row["h_cond_max"] if row is not None else None,
                    "failure_frame_index": row["failure_frame_index"] if row is not None else None,
                    "failure_timestamp": row["failure_timestamp"] if row is not None else None,
                    "log_path": row["log_path"] if row is not None else None,
                    "trajectory_path": row["trajectory_path"] if row is not None else None,
                    "keyframe_trajectory_path": (
                        row["keyframe_trajectory_path"] if row is not None else None
                    ),
                }
            )
    return all_raw, scorecard, raw_columns


def candidate_summary(
    scorecard: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    dataset_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        key = candidate["candidate_key"]
        cells = [row for row in scorecard if row["candidate_key"] == key]
        completed = [row for row in cells if row["status"] != "NOT_RUN"]
        passes = [row for row in cells if row["status"] == "PASS"]
        ratios = [row["ate_ratio_to_baseline"] for row in passes if row["ate_ratio_to_baseline"] is not None]
        ranks = [float(row["dataset_rank"]) for row in passes if row["dataset_rank"] is not None]
        status_counts = Counter(row["status"] for row in completed)
        rows.append(
            {
                "source_rank": candidate["source_rank"],
                "label": candidate["label"],
                "candidate_key": key,
                "channels": f"[{key}]",
                "is_baseline": key == BASELINE_KEY,
                "planned_datasets": dataset_count,
                "completed_datasets": len(completed),
                "pass_datasets": len(passes),
                "failed_datasets": len(completed) - len(passes),
                "timeout_datasets": status_counts.get("TIMEOUT", 0),
                "baseline_paired_datasets": len(ratios),
                "beats_baseline_count": sum(ratio < 1.0 for ratio in ratios),
                "ties_baseline_count": sum(math.isclose(ratio, 1.0) for ratio in ratios),
                "geomean_ate_ratio_to_baseline": geometric_mean(ratios),
                "mean_ate_improvement_vs_baseline_percent": (
                    (1.0 - safe_mean(ratios)) * 100.0 if ratios else None
                ),
                "median_ate_improvement_vs_baseline_percent": (
                    (1.0 - safe_median(ratios)) * 100.0 if ratios else None
                ),
                "mean_dataset_rank": safe_mean(ranks),
                "median_dataset_rank": safe_median(ranks),
                "dataset_wins": sum(rank == 1 for rank in ranks),
                "dataset_top3": sum(rank <= 3 for rank in ranks),
                "mean_translation_rpe_max_cm": safe_mean(
                    [row["translation_rpe_max_cm"] for row in passes if row["translation_rpe_max_cm"] is not None]
                ),
                "mean_rotation_rpe_max_deg": safe_mean(
                    [row["rotation_rpe_max_deg"] for row in passes if row["rotation_rpe_max_deg"] is not None]
                ),
                "status_counts": json.dumps(status_counts, sort_keys=True),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["pass_datasets"],
            -row["baseline_paired_datasets"],
            row["geomean_ate_ratio_to_baseline"]
            if row["geomean_ate_ratio_to_baseline"] is not None
            else math.inf,
            row["mean_dataset_rank"] if row["mean_dataset_rank"] is not None else math.inf,
            row["source_rank"],
        )
    )
    for rank, row in enumerate(rows, 1):
        row["aggregate_rank"] = rank
    return rows


def condition_summary(
    scorecard: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {(row["candidate_key"], row["dataset_family"], row["dataset_condition"]): row for row in scorecard}
    rows: list[dict[str, Any]] = []
    family_conditions: dict[str, list[str]] = {}
    for dataset in datasets:
        family_conditions.setdefault(dataset["family"], []).append(dataset["condition"])
    for candidate in candidates:
        key = candidate["candidate_key"]
        for family, conditions in family_conditions.items():
            if "clean" not in conditions:
                continue
            clean = lookup[(key, family, "clean")]
            for condition in (value for value in conditions if value != "clean"):
                altered = lookup[(key, family, condition)]
                clean_ate = clean["historical_evo_ape_mean_cm"]
                altered_ate = altered["historical_evo_ape_mean_cm"]
                ratio = (
                    altered_ate / clean_ate
                    if clean_ate is not None and altered_ate is not None and clean_ate > 0
                    else None
                )
                rows.append(
                    {
                        "source_rank": candidate["source_rank"],
                        "candidate_key": key,
                        "channels": f"[{key}]",
                        "family": family,
                        "condition": condition,
                        "clean_status": clean["status"],
                        "condition_status": altered["status"],
                        "clean_ate_mean_cm": clean_ate,
                        "condition_ate_mean_cm": altered_ate,
                        "condition_to_clean_ate_ratio": ratio,
                        "condition_vs_clean_change_percent": (
                            (ratio - 1.0) * 100.0 if ratio is not None else None
                        ),
                    }
                )
    return rows


def heatmaps(
    output_dir: Path,
    datasets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    scorecard: list[dict[str, Any]],
) -> None:
    lookup = {(row["candidate_key"], row["dataset_key"]): row for row in scorecard}
    row_labels = [f"#{item['source_rank']} [{item['candidate_key']}]" for item in candidates]
    column_labels = [item["key"].replace("_", "\n", 2) for item in datasets]
    specifications = (
        ("historical_evo_ape_mean_cm", "Historical keyframe ATE mean (cm)", "viridis_r", "ate_mean_heatmap.png"),
        ("ate_improvement_vs_baseline_percent", "ATE improvement vs dataset baseline (%)", "RdBu", "baseline_relative_heatmap.png"),
        ("dataset_rank", "Within-dataset rank (PASS only)", "viridis_r", "dataset_rank_heatmap.png"),
    )
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for field, title, cmap, filename in specifications:
        matrix = np.full((len(candidates), len(datasets)), np.nan)
        statuses: list[list[str]] = []
        for row_index, candidate in enumerate(candidates):
            status_row: list[str] = []
            for column_index, dataset in enumerate(datasets):
                cell = lookup[(candidate["candidate_key"], dataset["key"])]
                value = cell[field]
                if value is not None:
                    matrix[row_index, column_index] = float(value)
                status_row.append(cell["status"])
            statuses.append(status_row)
        if np.all(np.isnan(matrix)):
            continue
        figure, axis = plt.subplots(figsize=(12.5, 6.3))
        if field == "ate_improvement_vs_baseline_percent":
            limit = max(5.0, float(np.nanmax(np.abs(matrix))))
            image = axis.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit)
        else:
            image = axis.imshow(matrix, cmap=cmap)
        for row_index in range(len(candidates)):
            for column_index in range(len(datasets)):
                value = matrix[row_index, column_index]
                text = f"{value:.2f}" if math.isfinite(value) else statuses[row_index][column_index]
                axis.text(column_index, row_index, text, ha="center", va="center", fontsize=8)
        axis.set_xticks(range(len(datasets)))
        axis.set_xticklabels(column_labels, fontsize=8)
        axis.set_yticks(range(len(candidates)))
        axis.set_yticklabels(row_labels, fontsize=8.5)
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.82)
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_summary(
    output_dir: Path,
    datasets: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    scorecard: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    completed = sum(row["status"] != "NOT_RUN" for row in scorecard)
    passes = sum(row["status"] == "PASS" for row in scorecard)
    status_counts = Counter(row["status"] for row in scorecard if row["status"] != "NOT_RUN")
    lines = [
        "# Step F：Top-7多数据集综合评估",
        "",
        "## 固定协议",
        "",
        "- 8个数据集 × 7个配置 = 56次，每个配置每个数据集运行1次。",
        "- 单次timeout：500秒。",
        "- 主指标：historical keyframe `evo_ape --align --correct_scale` ATE mean。",
        "- 同时保存all-frame SE(3) ATE、translation/rotation RPE、coverage、photometric/numerical diagnostics和失败位置。",
        "- 综合排序优先保证PASS数据集数量，再比较dataset-baseline配对覆盖数、ATE ratio几何均值和dataset内平均排名。",
        "",
        "## 数据集",
        "",
        markdown_table(
            ["order", "key", "family", "condition", "matched frames"],
            [[str(item["order"]), item["key"], item["family"], item["condition"], f"{item['expected_matched_frames']:,}"] for item in datasets],
        ),
        "",
        "## 当前进度",
        "",
        f"- 已完成：{completed}/56；PASS：{passes}。",
        "- 状态：" + (", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())) if status_counts else "尚未运行"),
        "",
        "## 综合排名",
        "",
    ]
    summary_rows: list[list[str]] = []
    for row in summaries:
        ratio = row["geomean_ate_ratio_to_baseline"]
        mean_rank = row["mean_dataset_rank"]
        summary_rows.append(
            [
                str(row["aggregate_rank"]),
                str(row["source_rank"]),
                f"`[{row['candidate_key']}]`",
                f"{row['pass_datasets']}/8",
                f"{row['beats_baseline_count']}/{row['baseline_paired_datasets']}",
                f"{ratio:.4f}" if ratio is not None else "—",
                f"{mean_rank:.2f}" if mean_rank is not None else "—",
                str(row["dataset_wins"]),
            ]
        )
    lines.extend(
        [
            markdown_table(
                ["aggregate rank", "fr1 source rank", "channels", "PASS", "beats baseline", "ATE ratio geomean", "mean dataset rank", "dataset wins"],
                summary_rows,
            ),
            "",
            "ATE ratio < 1表示优于同一数据集上的baseline。不同序列长度和运动尺度不同，因此绝对ATE不直接跨数据集求平均。",
            "",
            "## 输出说明",
            "",
            "- `per_dataset/<dataset>/evaluations.sqlite3`：各数据集权威、可恢复数据库。",
            "- `all_runs_raw.csv`：保留评估器全部原始字段。",
            "- `dataset_scorecard.csv`：56个计划单元及主要cm/degree指标。",
            "- `candidate_aggregate_summary.csv`：failure-aware综合统计。",
            "- `condition_robustness.csv`：同一配置的flashlight/lightswitch相对clean变化。",
            "- `plots/`：绝对ATE、baseline-relative improvement和dataset rank热力图。",
        ]
    )
    (output_dir / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    dataset_payload = load_json(args.dataset_plan.resolve())
    candidate_payload = load_json(args.candidate_plan.resolve())
    datasets, candidates = validate_plans(dataset_payload, candidate_payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw, scorecard, raw_columns = collect(args.output_dir, datasets, candidates)
    summaries = candidate_summary(scorecard, candidates, len(datasets))
    robustness = condition_summary(scorecard, candidates, datasets)

    metadata = [
        "dataset_order", "dataset_key", "dataset_family", "dataset_condition",
        "dataset_directory", "source_rank", "is_baseline",
    ]
    write_csv(args.output_dir / "all_runs_raw.csv", raw, metadata + raw_columns)
    write_csv(args.output_dir / "dataset_scorecard.csv", scorecard, list(scorecard[0].keys()))
    write_csv(
        args.output_dir / "candidate_aggregate_summary.csv",
        summaries,
        ["aggregate_rank"] + [key for key in summaries[0] if key != "aggregate_rank"],
    )
    write_csv(
        args.output_dir / "condition_robustness.csv",
        robustness,
        list(robustness[0].keys()),
    )
    heatmaps(args.output_dir, datasets, candidates, scorecard)
    write_summary(args.output_dir, datasets, candidates, scorecard, summaries)
    protocol = {
        "protocol": "top7_multi_dataset_aggregate_v1",
        "dataset_plan": str(args.dataset_plan.resolve()),
        "candidate_plan": str(args.candidate_plan.resolve()),
        "planned_runs": 56,
        "completed_runs": len(raw),
        "primary_metric": "keyframe evo_ape --align --correct_scale ATE mean",
        "historical_rpe_metric": (
            "keyframe evo_rpe --align --correct_scale translation RMSE"
        ),
        "aggregate_rule": [
            "pass_datasets descending",
            "baseline_paired_datasets descending",
            "geometric mean of per-dataset ATE/baseline ATE ascending",
            "mean within-dataset rank ascending",
        ],
    }
    (args.output_dir / "aggregate_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(f"[AGGREGATE] completed={len(raw)}/56; outputs={args.output_dir}")


if __name__ == "__main__":
    main()
