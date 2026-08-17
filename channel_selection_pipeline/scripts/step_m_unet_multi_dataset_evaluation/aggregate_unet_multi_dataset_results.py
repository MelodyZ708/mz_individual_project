#!/usr/bin/env python3
"""Aggregate the U-Net Enc0/Enc1 3×3 multi-dataset evaluation.

Raw trajectories and all evaluator diagnostics remain in each per-dataset
SQLite database.  This tool creates a long-form table, a failure-aware ATE
scorecard, a wide ATE matrix and a compact Markdown hand-off.  It deliberately
does not average raw ATE across datasets of different length/scale.
"""

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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "channel_selection_results/step_m_unet_multi_dataset_evaluation"
DEFAULT_DATASET_PLAN = SCRIPT_DIR / "unet_dataset_plan.json"
DEFAULT_CANDIDATE_PLAN = SCRIPT_DIR / "unet_candidate_plan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
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
    dataset_plan: dict[str, Any], candidate_plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if dataset_plan.get("protocol") != "unet_enc0_enc1_three_dataset_three_lighting_conditions_v1":
        raise ValueError("Unexpected U-Net dataset plan")
    if candidate_plan.get("protocol") != "unet_enc0_enc1_multi_dataset_candidate_plan_v1":
        raise ValueError("Unexpected U-Net candidate plan")
    datasets = dataset_plan.get("datasets")
    candidates = candidate_plan.get("candidates")
    if not isinstance(datasets, list) or len(datasets) != 9:
        raise ValueError("Expected exactly nine datasets")
    if not isinstance(candidates, list) or len(candidates) != 13:
        raise ValueError("Expected exactly thirteen U-Net candidates")
    if dataset_plan.get("timeout_seconds_per_run") != 500:
        raise ValueError("Expected 500-second timeout")
    if len({item["key"] for item in datasets}) != len(datasets):
        raise ValueError("Duplicate dataset key")
    if len({item["candidate_key"] for item in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate key")
    if {item["enc_level"] for item in candidates} != {0, 1}:
        raise ValueError("Candidate plan must contain both Enc0 and Enc1")
    candidate_keys = {item["candidate_key"] for item in candidates}
    for dataset in datasets:
        excluded = dataset.get("excluded_candidate_keys", [])
        if not isinstance(excluded, list) or len(excluded) != len(set(excluded)):
            raise ValueError(f"Invalid/duplicate exclusions for {dataset['key']}")
        unknown = set(excluded) - candidate_keys
        if unknown:
            raise ValueError(f"Unknown exclusions for {dataset['key']}: {sorted(unknown)}")
        if excluded and not str(dataset.get("exclusion_reason", "")).strip():
            raise ValueError(f"Missing exclusion_reason for {dataset['key']}")
    return datasets, candidates


def read_database(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], []
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        connection.close()
        raise RuntimeError(f"SQLite integrity check failed for {path}: {integrity}")
    columns = [row[1] for row in connection.execute("PRAGMA table_info(evaluations)")]
    rows = [dict(row) for row in connection.execute("SELECT * FROM evaluations ORDER BY id")]
    connection.close()
    return rows, columns


def cm(value: Any) -> float | None:
    return None if value is None else float(value) * 100.0


def blank(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: blank(row.get(field)) for field in fields})


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    datasets, candidates = validate_plans(
        load_json(args.dataset_plan.resolve()), load_json(args.candidate_plan.resolve())
    )
    candidate_by_key = {item["candidate_key"]: item for item in candidates}
    raw_rows: list[dict[str, Any]] = []
    scorecard: list[dict[str, Any]] = []
    raw_schema: list[str] = []
    winners: list[dict[str, Any]] = []

    for dataset in datasets:
        key = dataset["key"]
        excluded_keys = set(dataset.get("excluded_candidate_keys", []))
        exclusion_reason = str(dataset.get("exclusion_reason", "")).strip()
        database = args.output_dir / "per_dataset" / key / "evaluations.sqlite3"
        rows, schema = read_database(database)
        if schema and not raw_schema:
            raw_schema = schema
        elif schema and set(schema) != set(raw_schema):
            raise ValueError(f"Database schema differs for {key}")
        by_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            candidate_key = str(row["candidate_key"])
            if row["replicate"] != 0:
                raise ValueError(f"Unexpected replicate in {key}: {row['replicate']}")
            if candidate_key not in candidate_by_key:
                raise ValueError(f"Unexpected candidate in {key}: {candidate_key}")
            if candidate_key in by_key:
                raise ValueError(f"Duplicate candidate in {key}: {candidate_key}")
            by_key[candidate_key] = row
            enriched = {
                "dataset_order": dataset["order"],
                "dataset_key": key,
                "dataset_family": dataset["family"],
                "dataset_condition": dataset["condition"],
                "dataset_directory": dataset["directory_name"],
                "enc_level": candidate_by_key[candidate_key]["enc_level"],
                "channel_count": len(candidate_by_key[candidate_key]["channels"]),
            }
            enriched.update(row)
            raw_rows.append(enriched)

        valid = [
            row
            for row in rows
            if row["status"] == "PASS" and row["historical_evo_ape_mean_m"] is not None
        ]
        valid.sort(key=lambda row: float(row["historical_evo_ape_mean_m"]))
        rank_by_key = {str(row["candidate_key"]): rank for rank, row in enumerate(valid, start=1)}
        if valid:
            winner = valid[0]
            winners.append(
                {
                    "dataset_key": key,
                    "winner_label": winner["label"],
                    "winner_key": winner["candidate_key"],
                    "winner_ate_mean_cm": cm(winner["historical_evo_ape_mean_m"]),
                }
            )
        for candidate in candidates:
            candidate_key = candidate["candidate_key"]
            row = by_key.get(candidate_key)
            passed = row is not None and row["status"] == "PASS"
            scorecard.append(
                {
                    "dataset_order": dataset["order"],
                    "dataset_key": key,
                    "dataset_family": dataset["family"],
                    "dataset_condition": dataset["condition"],
                    "expected_matched_frames": dataset["expected_matched_frames"],
                    "label": candidate["label"],
                    "candidate_key": candidate_key,
                    "role": candidate["role"],
                    "enc_level": candidate["enc_level"],
                    "channels": "[" + ",".join(map(str, candidate["channels"])) + "]",
                    "channel_count": len(candidate["channels"]),
                    "source_fr1_lightswitch_ate_mean_cm": candidate["source_fr1_lightswitch_ate_mean_cm"],
                    "status": (
                        row["status"]
                        if row is not None
                        else "SKIPPED_BY_SAFETY"
                        if candidate_key in excluded_keys
                        else "NOT_RUN"
                    ),
                    "reason": row["reason"] if row is not None else exclusion_reason if candidate_key in excluded_keys else "",
                    "dataset_rank": rank_by_key.get(candidate_key),
                    "coverage_ratio": row["coverage_ratio"] if row is not None else None,
                    "trajectory_poses": row["trajectory_poses"] if row is not None else None,
                    "associated_poses": row["associated_poses"] if row is not None else None,
                    "elapsed_seconds": row["elapsed_seconds"] if row is not None else None,
                    "historical_evo_ape_mean_cm": cm(row["historical_evo_ape_mean_m"]) if passed else None,
                    "historical_evo_ape_rmse_cm": cm(row["historical_evo_ape_rmse_m"]) if passed else None,
                    "historical_evo_rpe_rmse_cm": cm(row["historical_evo_rpe_rmse_m"]) if passed else None,
                    "se3_ate_rmse_cm": cm(row["se3_ate_rmse_m"]) if passed else None,
                    "se3_ate_mean_cm": cm(row["se3_ate_mean_m"]) if passed else None,
                    "se3_ate_max_cm": cm(row["se3_ate_max_m"]) if passed else None,
                    "translation_rpe_rmse_cm": cm(row["translation_rpe_rmse_m"]) if passed else None,
                    "translation_rpe_max_cm": cm(row["translation_rpe_max_m"]) if passed else None,
                    "rotation_rpe_rmse_deg": row["rotation_rpe_rmse_deg"] if passed else None,
                    "rotation_rpe_max_deg": row["rotation_rpe_max_deg"] if passed else None,
                    "rotation_ape_rmse_deg": row["rotation_ape_rmse_deg"] if passed else None,
                    "photo_mse_median": row["photo_mse_median"] if row is not None else None,
                    "photo_mse_p95": row["photo_mse_p95"] if row is not None else None,
                    "photo_mse_nonfinite_count": row["photo_mse_nonfinite_count"] if row is not None else None,
                    "valid_ratio_min": row["valid_ratio_min"] if row is not None else None,
                    "h_cond_max": row["h_cond_max"] if row is not None else None,
                    "crazy_affine_count": row["crazy_affine_count"] if row is not None else None,
                }
            )

    score_fields = list(scorecard[0]) if scorecard else []
    write_csv(args.output_dir / "all_runs_long.csv", raw_rows, [
        "dataset_order", "dataset_key", "dataset_family", "dataset_condition", "dataset_directory",
        "enc_level", "channel_count", *raw_schema,
    ])
    write_csv(args.output_dir / "dataset_scorecard.csv", scorecard, score_fields)

    matrix_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        item: dict[str, Any] = {
            "label": candidate["label"],
            "candidate_key": candidate["candidate_key"],
            "enc_level": candidate["enc_level"],
            "channels": "[" + ",".join(map(str, candidate["channels"])) + "]",
            "channel_count": len(candidate["channels"]),
            "source_fr1_lightswitch_ate_mean_cm": candidate["source_fr1_lightswitch_ate_mean_cm"],
        }
        for dataset in datasets:
            match = next(
                row for row in scorecard
                if row["dataset_key"] == dataset["key"] and row["candidate_key"] == candidate["candidate_key"]
            )
            item[f"{dataset['key']}__status"] = match["status"]
            item[f"{dataset['key']}__ate_mean_cm"] = match["historical_evo_ape_mean_cm"]
            item[f"{dataset['key']}__rank"] = match["dataset_rank"]
        matrix_rows.append(item)
    matrix_fields = list(matrix_rows[0]) if matrix_rows else []
    write_csv(args.output_dir / "ate_mean_matrix.csv", matrix_rows, matrix_fields)

    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        rows = [row for row in scorecard if row["candidate_key"] == candidate["candidate_key"]]
        passes = [row for row in rows if row["status"] == "PASS" and row["dataset_rank"] is not None]
        summaries.append(
            {
                "label": candidate["label"],
                "candidate_key": candidate["candidate_key"],
                "enc_level": candidate["enc_level"],
                "channels": "[" + ",".join(map(str, candidate["channels"])) + "]",
                "channel_count": len(candidate["channels"]),
                "pass_count": len(passes),
                "failure_or_pending_count": sum(
                    row["status"] not in {"PASS", "SKIPPED_BY_SAFETY"} for row in rows
                ),
                "safety_skipped_count": sum(row["status"] == "SKIPPED_BY_SAFETY" for row in rows),
                "mean_dataset_rank_on_passes": (
                    statistics.fmean(float(row["dataset_rank"]) for row in passes) if passes else None
                ),
                "median_dataset_rank_on_passes": (
                    statistics.median(float(row["dataset_rank"]) for row in passes) if passes else None
                ),
                "datasets_won": sum(row["dataset_rank"] == 1 for row in passes),
            }
        )
    summaries.sort(key=lambda row: (-row["pass_count"], row["mean_dataset_rank_on_passes"] or math.inf))
    write_csv(args.output_dir / "candidate_robustness_summary.csv", summaries, list(summaries[0]) if summaries else [])

    counts = Counter(row["status"] for row in scorecard)
    lines = [
        "# U-Net Enc0/Enc1 multi-dataset evaluation",
        "",
        "- Scope: 3 TUM dataset families × clean/lightswitch/flashlight = 9 full sequences.",
        "- Configurations: 13 selected U-Net candidates; one run per non-excluded cell; nominal total = 117.",
        "- Safety scope change: Enc0-all16 and Enc1-all32 are explicitly omitted only on fr3 lightswitch after the Enc0-all16 run coincided with NVIDIA Xid 79 / PCIe receiver error.",
        "- Tracking: U-Net Enc0 or Enc1 selected channels. Mapping remains gray with sensor depth.",
        "- Primary metric: historical keyframe `evo_ape --align --correct_scale` ATE mean.",
        "- All raw trajectory diagnostics are retained in each `per_dataset/*/evaluations.sqlite3` database.",
        f"- Current cell statuses: {dict(sorted(counts.items()))}.",
        "- Do not average raw ATE values across sequences; use within-dataset ranks and pass counts for cross-sequence comparison.",
        "",
        "## Current dataset winners",
        "",
        "| Dataset | Winner | Historical ATE mean (cm) |",
        "|---|---|---:|",
    ]
    if winners:
        for winner in winners:
            lines.append(
                f"| {winner['dataset_key']} | {winner['winner_label']} ({winner['winner_key']}) | {winner['winner_ate_mean_cm']:.4f} |"
            )
    else:
        lines.append("| No completed PASS rows yet |  |  |")
    lines.extend([
        "",
        "## Candidate robustness summary",
        "",
        "| Candidate | Enc | Channels | PASS / 9 | Mean rank on PASS | Datasets won |",
        "|---|---:|---|---:|---:|---:|",
    ])
    for row in summaries:
        mean_rank = "" if row["mean_dataset_rank_on_passes"] is None else f"{row['mean_dataset_rank_on_passes']:.3f}"
        lines.append(
            f"| {row['label']} | {row['enc_level']} | {row['channels']} | {row['pass_count']} | {mean_rank} | {row['datasets_won']} |"
        )
    (args.output_dir / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    active_cells = len(datasets) * len(candidates) - sum(
        len(dataset.get("excluded_candidate_keys", [])) for dataset in datasets
    )
    print(f"[AGGREGATE] rows={len(raw_rows)}/{active_cells} active cells; status_counts={dict(sorted(counts.items()))}")
    print(f"[WRITE] {args.output_dir / 'dataset_scorecard.csv'}")
    print(f"[WRITE] {args.output_dir / 'ate_mean_matrix.csv'}")
    print(f"[WRITE] {args.output_dir / 'candidate_robustness_summary.csv'}")


if __name__ == "__main__":
    main()
