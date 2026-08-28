#!/usr/bin/env python3
"""Aggregate the multi-sequence direct-parent versus C2F comparison.

Raw ATE is sequence dependent, so this tool never treats an arithmetic mean of
ATE across fr1/fr2/fr3 as the C2F conclusion.  Instead it computes every C2F
gain/loss against its own direct parents on the *same* sequence, then reports
the number of within-sequence wins, paired availability, and per-dataset
absolute/percentage deltas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "channel_selection_results/step_q_c2f_multi_dataset_evaluation"
DEFAULT_DATASET_PLAN = SCRIPT_DIR / "c2f_multi_dataset_plan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument("--architecture", choices=("resnet", "unet"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-plan", type=Path, default=DEFAULT_DATASET_PLAN)
    parser.add_argument("--candidate-plan", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def default_candidate_plan(architecture: str) -> Path:
    return SCRIPT_DIR / f"{architecture}_c2f_parent_comparison_plan.json"


def validate_plans(dataset_doc: dict[str, Any], candidate_doc: dict[str, Any], architecture: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if dataset_doc.get("protocol") != "c2f_parent_comparative_three_dataset_three_lighting_conditions_v1":
        raise ValueError("Unexpected multi-dataset plan protocol")
    if candidate_doc.get("protocol") != "c2f_parent_comparative_candidate_plan_v1":
        raise ValueError("Unexpected C2F parent-comparison candidate plan protocol")
    if candidate_doc.get("architecture") != architecture:
        raise ValueError("Candidate-plan architecture does not match --architecture")
    datasets = dataset_doc.get("datasets")
    candidates = candidate_doc.get("candidates")
    if not isinstance(datasets, list) or len(datasets) != 9 or dataset_doc.get("dataset_count") != 9:
        raise ValueError("Expected the frozen nine-sequence plan")
    if dataset_doc.get("timeout_seconds_per_run") != 500 or dataset_doc.get("replicates_per_candidate") != 1:
        raise ValueError("Expected 500 seconds and one replicate per configuration")
    if not isinstance(candidates, list) or candidate_doc.get("selection", {}).get("selected_count") != len(candidates):
        raise ValueError("Candidate selection count does not match candidate list")
    if len({item.get("key") for item in datasets}) != len(datasets):
        raise ValueError("Duplicate dataset key")
    labels = [str(item.get("label", "")) for item in candidates]
    keys = [str(item.get("candidate_key", "")) for item in candidates]
    if not all(labels) or len(set(labels)) != len(labels) or not all(keys) or len(set(keys)) != len(keys):
        raise ValueError("Candidate labels/keys must be non-empty and unique")
    if sum(item.get("mode") == "gray" for item in candidates) != 1:
        raise ValueError("Plan must contain exactly one gray baseline")
    label_set = set(labels)
    for expected_order, dataset in enumerate(datasets, start=1):
        if dataset.get("order") != expected_order:
            raise ValueError("Dataset order is not frozen")
    for candidate in candidates:
        if candidate.get("mode") == "c2f":
            parents = candidate.get("parent_labels")
            if not isinstance(parents, list) or len(parents) != 2 or any(parent not in label_set for parent in parents):
                raise ValueError(f"Invalid parent_labels for {candidate.get('label')}")
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


def configuration_display(candidate: dict[str, Any]) -> str:
    def branch_text(branch: dict[str, Any]) -> str:
        return f"{branch['layer']}[" + ",".join(f"d{item}" for item in branch["channels"]) + "]"
    if candidate["mode"] == "gray":
        return "gray"
    if candidate["mode"] == "direct":
        return "direct " + branch_text(candidate)
    return f"C2F-{candidate['variant']}; fine {branch_text(candidate['fine'])}; coarse {branch_text(candidate['coarse'])}"


def pass_ate(row: dict[str, Any] | None) -> float | None:
    if row is None or row.get("status") != "PASS" or row.get("historical_evo_ape_mean_m") is None:
        return None
    return cm(row["historical_evo_ape_mean_m"])


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / args.architecture
    if args.candidate_plan is None:
        args.candidate_plan = default_candidate_plan(args.architecture)
    args.output_dir = args.output_dir.resolve()
    args.dataset_plan = args.dataset_plan.resolve()
    args.candidate_plan = args.candidate_plan.resolve()
    dataset_doc = load_json(args.dataset_plan)
    candidate_doc = load_json(args.candidate_plan)
    datasets, candidates = validate_plans(dataset_doc, candidate_doc, args.architecture)
    candidate_by_key = {str(item["candidate_key"]): item for item in candidates}
    candidate_by_label = {str(item["label"]): item for item in candidates}

    raw_rows: list[dict[str, Any]] = []
    scorecard: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    winners: list[dict[str, Any]] = []
    raw_schema: list[str] = []
    scorecard_lookup: dict[tuple[str, str], dict[str, Any]] = {}

    for dataset in datasets:
        dataset_key = str(dataset["key"])
        db_path = args.output_dir / "per_dataset" / dataset_key / "evaluations.sqlite3"
        rows, schema = read_database(db_path)
        if schema and not raw_schema:
            raw_schema = schema
        elif schema and set(schema) != set(raw_schema):
            raise ValueError(f"Database schema differs for {dataset_key}")
        by_key: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row["candidate_key"])
            if row["replicate"] != 0:
                raise ValueError(f"Unexpected replicate {row['replicate']} in {dataset_key}")
            if key not in candidate_by_key:
                raise ValueError(f"Unexpected candidate key {key!r} in {dataset_key}")
            if key in by_key:
                raise ValueError(f"Duplicate candidate key {key!r} in {dataset_key}")
            by_key[key] = row
            enriched = {
                "dataset_order": dataset["order"],
                "dataset_key": dataset_key,
                "dataset_family": dataset["family"],
                "dataset_condition": dataset["condition"],
                "dataset_directory": dataset["directory_name"],
                "candidate_mode": candidate_by_key[key]["mode"],
            }
            enriched.update(row)
            raw_rows.append(enriched)
        valid = [row for row in rows if pass_ate(row) is not None]
        valid.sort(key=lambda row: float(row["historical_evo_ape_mean_m"]))
        rank_by_key = {str(row["candidate_key"]): index for index, row in enumerate(valid, start=1)}
        if valid:
            winner = valid[0]
            winners.append({
                "dataset_key": dataset_key,
                "winner_label": winner["label"],
                "winner_ate_mean_cm": pass_ate(winner),
            })
        for candidate in candidates:
            key = str(candidate["candidate_key"])
            row = by_key.get(key)
            passed = pass_ate(row) is not None
            long_row = {
                "dataset_order": dataset["order"],
                "dataset_key": dataset_key,
                "dataset_family": dataset["family"],
                "dataset_condition": dataset["condition"],
                "expected_matched_frames": dataset["expected_matched_frames"],
                "label": candidate["label"],
                "candidate_key": key,
                "mode": candidate["mode"],
                "variant": candidate.get("variant", ""),
                "configuration": configuration_display(candidate),
                "role": candidate["role"],
                "parent_labels": "|".join(candidate.get("parent_labels", [])),
                "source_fr1_lightswitch_ate_mean_cm": candidate.get("source_fr1_lightswitch_ate_mean_cm"),
                "status": row["status"] if row is not None else "NOT_RUN",
                "reason": row["reason"] if row is not None else "",
                "dataset_rank": rank_by_key.get(key),
                "coverage_ratio": row["coverage_ratio"] if row is not None else None,
                "trajectory_poses": row["trajectory_poses"] if row is not None else None,
                "associated_poses": row["associated_poses"] if row is not None else None,
                "elapsed_seconds": row["elapsed_seconds"] if row is not None else None,
                "historical_evo_ape_mean_cm": pass_ate(row),
                "historical_evo_ape_rmse_cm": cm(row["historical_evo_ape_rmse_m"]) if passed else None,
                "historical_evo_rpe_rmse_cm": cm(row["historical_evo_rpe_rmse_m"]) if passed else None,
                "se3_ate_rmse_cm": cm(row["se3_ate_rmse_m"]) if passed else None,
                "se3_ate_mean_cm": cm(row["se3_ate_mean_m"]) if passed else None,
                "translation_rpe_max_cm": cm(row["translation_rpe_max_m"]) if passed else None,
                "rotation_rpe_max_deg": row["rotation_rpe_max_deg"] if passed else None,
                "photo_mse_nonfinite_count": row["photo_mse_nonfinite_count"] if row is not None else None,
                "crazy_affine_count": row["crazy_affine_count"] if row is not None else None,
            }
            scorecard.append(long_row)
            scorecard_lookup[(dataset_key, str(candidate["label"]))] = long_row

    for dataset in datasets:
        dataset_key = str(dataset["key"])
        for candidate in candidates:
            if candidate["mode"] != "c2f":
                continue
            c2f = scorecard_lookup[(dataset_key, str(candidate["label"]))]
            fine_label, coarse_label = candidate["parent_labels"]
            fine = scorecard_lookup[(dataset_key, str(fine_label))]
            coarse = scorecard_lookup[(dataset_key, str(coarse_label))]
            c2f_ate = c2f["historical_evo_ape_mean_cm"]
            fine_ate = fine["historical_evo_ape_mean_cm"]
            coarse_ate = coarse["historical_evo_ape_mean_cm"]
            both_parents_pass = fine_ate is not None and coarse_ate is not None
            comparable = c2f_ate is not None and both_parents_pass
            best_parent_label = ""
            best_parent_ate: float | None = None
            if both_parents_pass:
                if fine_ate <= coarse_ate:
                    best_parent_label, best_parent_ate = fine_label, fine_ate
                else:
                    best_parent_label, best_parent_ate = coarse_label, coarse_ate
            if comparable:
                comparison_status = "COMPARABLE"
                delta_fine = c2f_ate - fine_ate
                pct_fine = 100.0 * delta_fine / fine_ate
                assert best_parent_ate is not None
                delta_best = c2f_ate - best_parent_ate
                pct_best = 100.0 * delta_best / best_parent_ate
            elif c2f["status"] == "NOT_RUN" or fine["status"] == "NOT_RUN" or coarse["status"] == "NOT_RUN":
                comparison_status = "NOT_RUN"
                delta_fine = pct_fine = delta_best = pct_best = None
            elif c2f_ate is None:
                comparison_status = "C2F_NONPASS"
                delta_fine = pct_fine = delta_best = pct_best = None
            else:
                comparison_status = "PARENT_NONPASS"
                delta_fine = pct_fine = delta_best = pct_best = None
            comparisons.append({
                "dataset_order": dataset["order"],
                "dataset_key": dataset_key,
                "dataset_family": dataset["family"],
                "dataset_condition": dataset["condition"],
                "c2f_label": candidate["label"],
                "c2f_variant": candidate["variant"],
                "c2f_configuration": configuration_display(candidate),
                "c2f_status": c2f["status"],
                "c2f_ate_mean_cm": c2f_ate,
                "fine_parent_label": fine_label,
                "fine_parent_status": fine["status"],
                "fine_parent_ate_mean_cm": fine_ate,
                "coarse_parent_label": coarse_label,
                "coarse_parent_status": coarse["status"],
                "coarse_parent_ate_mean_cm": coarse_ate,
                "best_direct_parent_label": best_parent_label,
                "best_direct_parent_ate_mean_cm": best_parent_ate,
                "comparison_status": comparison_status,
                "delta_vs_fine_cm": delta_fine,
                "percent_delta_vs_fine": pct_fine,
                "beats_fine_parent": comparable and delta_fine < 0.0,
                "delta_vs_better_direct_parent_cm": delta_best,
                "percent_delta_vs_better_direct_parent": pct_best,
                "beats_better_direct_parent": comparable and delta_best < 0.0,
            })

    score_fields = list(scorecard[0]) if scorecard else []
    write_csv(args.output_dir / "all_runs_long.csv", raw_rows, [
        "dataset_order", "dataset_key", "dataset_family", "dataset_condition", "dataset_directory", "candidate_mode", *raw_schema,
    ])
    write_csv(args.output_dir / "dataset_scorecard.csv", scorecard, score_fields)
    comparison_fields = list(comparisons[0]) if comparisons else []
    write_csv(args.output_dir / "c2f_pairwise_comparison.csv", comparisons, comparison_fields)

    matrix_rows: list[dict[str, Any]] = []
    comparison_by_key = {(row["dataset_key"], row["c2f_label"]): row for row in comparisons}
    for candidate in candidates:
        item: dict[str, Any] = {
            "label": candidate["label"],
            "candidate_key": candidate["candidate_key"],
            "mode": candidate["mode"],
            "variant": candidate.get("variant", ""),
            "configuration": configuration_display(candidate),
            "role": candidate["role"],
            "source_fr1_lightswitch_ate_mean_cm": candidate.get("source_fr1_lightswitch_ate_mean_cm"),
        }
        for dataset in datasets:
            dataset_key = str(dataset["key"])
            row = scorecard_lookup[(dataset_key, str(candidate["label"]))]
            item[f"{dataset_key}__status"] = row["status"]
            item[f"{dataset_key}__ate_mean_cm"] = row["historical_evo_ape_mean_cm"]
            item[f"{dataset_key}__rank"] = row["dataset_rank"]
            if candidate["mode"] == "c2f":
                comparison = comparison_by_key[(dataset_key, str(candidate["label"]))]
                item[f"{dataset_key}__delta_vs_fine_cm"] = comparison["delta_vs_fine_cm"]
                item[f"{dataset_key}__delta_vs_better_parent_cm"] = comparison["delta_vs_better_direct_parent_cm"]
        matrix_rows.append(item)
    write_csv(args.output_dir / "ate_mean_matrix.csv", matrix_rows, list(matrix_rows[0]) if matrix_rows else [])

    effect_summary: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["mode"] != "c2f":
            continue
        rows = [row for row in comparisons if row["c2f_label"] == candidate["label"]]
        comparable_rows = [row for row in rows if row["comparison_status"] == "COMPARABLE"]
        fine_deltas = [float(row["delta_vs_fine_cm"]) for row in comparable_rows]
        best_deltas = [float(row["delta_vs_better_direct_parent_cm"]) for row in comparable_rows]
        fine_pcts = [float(row["percent_delta_vs_fine"]) for row in comparable_rows]
        best_pcts = [float(row["percent_delta_vs_better_direct_parent"]) for row in comparable_rows]
        effect_summary.append({
            "c2f_label": candidate["label"],
            "variant": candidate["variant"],
            "configuration": configuration_display(candidate),
            "c2f_pass_count": sum(row["c2f_status"] == "PASS" for row in rows),
            "comparable_parent_pairs": len(comparable_rows),
            "beats_fine_parent_count": sum(bool(row["beats_fine_parent"]) for row in comparable_rows),
            "beats_better_direct_parent_count": sum(bool(row["beats_better_direct_parent"]) for row in comparable_rows),
            "median_delta_vs_fine_cm": median_or_none(fine_deltas),
            "median_percent_delta_vs_fine": median_or_none(fine_pcts),
            "median_delta_vs_better_parent_cm": median_or_none(best_deltas),
            "median_percent_delta_vs_better_parent": median_or_none(best_pcts),
            "noncomparable_c2f_nonpass_count": sum(row["comparison_status"] == "C2F_NONPASS" for row in rows),
            "noncomparable_parent_nonpass_count": sum(row["comparison_status"] == "PARENT_NONPASS" for row in rows),
            "not_run_count": sum(row["comparison_status"] == "NOT_RUN" for row in rows),
        })
    effect_summary.sort(key=lambda row: (-row["beats_better_direct_parent_count"], -(row["beats_fine_parent_count"]), row["median_percent_delta_vs_better_parent"] if row["median_percent_delta_vs_better_parent"] is not None else math.inf))
    write_csv(args.output_dir / "c2f_effect_summary.csv", effect_summary, list(effect_summary[0]) if effect_summary else [])

    variant_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparisons:
        variant_groups[str(row["c2f_variant"])].append(row)
    variant_summary: list[dict[str, Any]] = []
    for variant, rows in sorted(variant_groups.items()):
        comparable_rows = [row for row in rows if row["comparison_status"] == "COMPARABLE"]
        variant_summary.append({
            "variant": variant,
            "c2f_cells": len(rows),
            "c2f_pass_count": sum(row["c2f_status"] == "PASS" for row in rows),
            "comparable_parent_pairs": len(comparable_rows),
            "beats_fine_parent_count": sum(bool(row["beats_fine_parent"]) for row in comparable_rows),
            "beats_better_direct_parent_count": sum(bool(row["beats_better_direct_parent"]) for row in comparable_rows),
            "median_percent_delta_vs_fine": median_or_none([float(row["percent_delta_vs_fine"]) for row in comparable_rows]),
            "median_percent_delta_vs_better_parent": median_or_none([float(row["percent_delta_vs_better_direct_parent"]) for row in comparable_rows]),
        })
    write_csv(args.output_dir / "c2f_variant_summary.csv", variant_summary, list(variant_summary[0]) if variant_summary else [])

    counts = Counter(row["status"] for row in scorecard)
    lines = [
        f"# {args.architecture.upper()} multi-sequence C2F parent comparison",
        "",
        "- Scope: fr1/fr2/fr3 × clean/lightswitch/flashlight = nine full sequences.",
        f"- Configurations: {len(candidates)} per sequence ({sum(item['mode'] == 'direct' for item in candidates)} direct parents, {sum(item['mode'] == 'c2f' for item in candidates)} C2F cells, one gray baseline).",
        "- Mapping remains gray with sensor depth; only tracking features change.",
        "- Primary metric: historical keyframe `evo_ape --align --correct_scale` translation ATE mean.",
        "- Interpretation rule: assess C2F using within-dataset deltas against its direct fine parent and the better of both direct parents. Do not average raw ATE across fr1/fr2/fr3.",
        f"- Current cell statuses: {dict(sorted(counts.items()))}.",
        "",
        "## Per-dataset winner among the focused comparison set",
        "",
        "| Dataset | Winner | Historical ATE mean (cm) |",
        "|---|---|---:|",
    ]
    if winners:
        for winner in winners:
            lines.append(f"| {winner['dataset_key']} | {winner['winner_label']} | {winner['winner_ate_mean_cm']:.4f} |")
    else:
        lines.append("| No completed PASS rows yet |  |  |")
    lines.extend([
        "",
        "## C2F effect summary",
        "",
        "| C2F configuration | Variant | Comparable pairs | Beats fine | Beats better direct parent | Median Δ vs fine (%) | Median Δ vs better parent (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in effect_summary:
        fine = "" if row["median_percent_delta_vs_fine"] is None else f"{row['median_percent_delta_vs_fine']:.2f}"
        best = "" if row["median_percent_delta_vs_better_parent"] is None else f"{row['median_percent_delta_vs_better_parent']:.2f}"
        lines.append(
            f"| {row['c2f_label']} | {row['variant']} | {row['comparable_parent_pairs']} | "
            f"{row['beats_fine_parent_count']} | {row['beats_better_direct_parent_count']} | {fine} | {best} |"
        )
    lines.extend([
        "",
        "Negative Δ means C2F is lower/better than the referenced direct parent on that dataset.",
        "",
        "## Files",
        "",
        "- `dataset_scorecard.csv`: one row per configuration × dataset, with complete diagnostics.",
        "- `c2f_pairwise_comparison.csv`: the primary evidence table: C2F, both parents, absolute and percentage deltas.",
        "- `c2f_effect_summary.csv`: win/loss counts and median within-sequence deltas for each focused C2F configuration.",
        "- `ate_mean_matrix.csv`: presentation-oriented wide table; C2F rows also include their per-dataset parent deltas.",
    ])
    (args.output_dir / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    nominal_cells = len(datasets) * len(candidates)
    print(f"[AGGREGATE] rows={len(raw_rows)}/{nominal_cells}; status_counts={dict(sorted(counts.items()))}")
    print(f"[WRITE] {args.output_dir / 'c2f_pairwise_comparison.csv'}")
    print(f"[WRITE] {args.output_dir / 'c2f_effect_summary.csv'}")
    print(f"[WRITE] {args.output_dir / 'ate_mean_matrix.csv'}")


if __name__ == "__main__":
    main()
