#!/usr/bin/env python3
"""Merge externally reused Step-Q rows with new Step-R C2F grid cells.

This tool is intentionally read-only with respect to Step-Q.  The merged
scorecard retains the provenance of every cell so the full selected grid can be
analysed without rerunning, copying, or silently replacing prior experiments.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RUNNER_PATH = SCRIPT_DIR / "run_c2f_grid_completion.py"
DATASET_PLAN = PROJECT_ROOT / "channel_selection_pipeline/scripts/step_q_c2f_multi_dataset_evaluation/c2f_multi_dataset_plan.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "channel_selection_results/step_r_c2f_grid_completion"


def load_runner():
    spec = importlib.util.spec_from_file_location("c2f_completion_runner_for_aggregate", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import completion runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument("--architecture", choices=("resnet", "unet"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--completion-plan", type=Path)
    return parser.parse_args()


def read_database(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], []
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed for {path}: {integrity}")
        columns = [row[1] for row in connection.execute("PRAGMA table_info(evaluations)")]
        rows = [dict(row) for row in connection.execute("SELECT * FROM evaluations WHERE replicate=0 ORDER BY id")]
    finally:
        connection.close()
    return rows, columns


def cm(value: Any) -> float | None:
    return None if value is None else float(value) * 100.0


def blank(value: Any) -> Any:
    return "" if value is None or isinstance(value, float) and not math.isfinite(value) else value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: blank(row.get(field)) for field in fields})


def history_ate(row: dict[str, Any] | None) -> float | None:
    if row is None or row.get("status") != "PASS" or row.get("historical_evo_ape_mean_m") is None:
        return None
    return cm(row["historical_evo_ape_mean_m"])


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def source_description(item) -> str:
    return f"Step-Q:{item.reused_step_q_label}" if item.is_reused else "Step-R:new"


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / args.architecture
    args.output_dir = args.output_dir.resolve()
    if args.completion_plan is None:
        args.completion_plan = runner.default_plan_path(args.architecture)
    args.completion_plan = args.completion_plan.resolve()
    plan, items, reuse_output = runner.load_grid(SimpleNamespace(architecture=args.architecture, completion_plan=args.completion_plan))
    datasets_doc = json.loads(DATASET_PLAN.read_text(encoding="utf-8"))
    datasets = datasets_doc["datasets"]
    if len(datasets) != 9:
        raise ValueError("Expected frozen nine-dataset plan")
    new_labels = {item.label for item in items if not item.is_reused}
    item_by_label = {item.label: item for item in items}
    direct_by_kind_rank = {(item.kind, item.fine_rank if item.kind == "direct_fine" else item.coarse_rank): item for item in items if item.kind.startswith("direct_")}

    long_rows: list[dict[str, Any]] = []
    scorecard: list[dict[str, Any]] = []
    score_by_dataset_label: dict[tuple[str, str], dict[str, Any]] = {}
    schema: list[str] = []
    for dataset in datasets:
        key = dataset["key"]
        q_rows, q_schema = read_database(reuse_output / "per_dataset" / key / "evaluations.sqlite3")
        r_rows, r_schema = read_database(args.output_dir / "per_dataset" / key / "evaluations.sqlite3")
        if q_schema and not schema:
            schema = q_schema
        for candidate_schema in (q_schema, r_schema):
            if candidate_schema and set(candidate_schema) != set(schema):
                raise ValueError(f"Database schema differs for {key}")
        q_by_label = {str(row["label"]): row for row in q_rows}
        r_by_label = {str(row["label"]): row for row in r_rows}
        unexpected = set(r_by_label) - new_labels
        if unexpected:
            raise ValueError(f"Step-R database has unexpected label(s) in {key}: {sorted(unexpected)}")
        valid_entries: list[tuple[Any, dict[str, Any]]] = []
        for item in items:
            row = q_by_label.get(item.reused_step_q_label) if item.is_reused else r_by_label.get(item.label)
            if row is not None:
                valid_entries.append((item, row))
                enriched = {"dataset_order": dataset["order"], "dataset_key": key, "dataset_family": dataset["family"], "dataset_condition": dataset["condition"], "source": source_description(item), "standard_label": item.label, "kind": item.kind}
                enriched.update(row)
                long_rows.append(enriched)
            ate = history_ate(row)
            entry = {
                "dataset_order": dataset["order"], "dataset_key": key, "dataset_family": dataset["family"], "dataset_condition": dataset["condition"],
                "standard_label": item.label, "source_label": item.reused_step_q_label if item.is_reused else item.label,
                "source": source_description(item), "kind": item.kind, "variant": item.variant or "", "fine_rank": item.fine_rank or "", "coarse_rank": item.coarse_rank or "",
                "configuration": item.spec.display, "status": row["status"] if row is not None else "NOT_RUN", "reason": row["reason"] if row is not None else "",
                "historical_evo_ape_mean_cm": ate, "historical_evo_ape_rmse_cm": cm(row["historical_evo_ape_rmse_m"]) if ate is not None else None,
                "historical_evo_rpe_rmse_cm": cm(row["historical_evo_rpe_rmse_m"]) if ate is not None else None,
                "se3_ate_rmse_cm": cm(row["se3_ate_rmse_m"]) if ate is not None else None,
                "se3_ate_mean_cm": cm(row["se3_ate_mean_m"]) if ate is not None else None,
                "translation_rpe_max_cm": cm(row["translation_rpe_max_m"]) if ate is not None else None,
                "rotation_rpe_max_deg": row["rotation_rpe_max_deg"] if ate is not None else None,
                "coverage_ratio": row["coverage_ratio"] if row is not None else None, "elapsed_seconds": row["elapsed_seconds"] if row is not None else None,
            }
            scorecard.append(entry)
            score_by_dataset_label[(key, item.label)] = entry
        valid_entries.sort(key=lambda pair: history_ate(pair[1]) if history_ate(pair[1]) is not None else math.inf)
        for rank, (item, row) in enumerate((pair for pair in valid_entries if history_ate(pair[1]) is not None), start=1):
            score_by_dataset_label[(key, item.label)]["dataset_rank"] = rank
        for item in items:
            score_by_dataset_label[(key, item.label)].setdefault("dataset_rank", None)

    pairwise: list[dict[str, Any]] = []
    for dataset in datasets:
        key = dataset["key"]
        for item in items:
            if item.kind != "c2f":
                continue
            fine = direct_by_kind_rank[("direct_fine", item.fine_rank)]
            coarse = direct_by_kind_rank[("direct_coarse", item.coarse_rank)]
            c2f_row = score_by_dataset_label[(key, item.label)]
            fine_row = score_by_dataset_label[(key, fine.label)]
            coarse_row = score_by_dataset_label[(key, coarse.label)]
            c2f_ate, fine_ate, coarse_ate = (c2f_row["historical_evo_ape_mean_cm"], fine_row["historical_evo_ape_mean_cm"], coarse_row["historical_evo_ape_mean_cm"])
            comparable = c2f_ate is not None and fine_ate is not None and coarse_ate is not None
            if fine_ate is not None and coarse_ate is not None:
                best_label, best_ate = (fine.label, fine_ate) if fine_ate <= coarse_ate else (coarse.label, coarse_ate)
            else:
                best_label, best_ate = "", None
            if comparable:
                delta_fine = c2f_ate - fine_ate
                delta_best = c2f_ate - best_ate
                percent_fine = 100 * delta_fine / fine_ate
                percent_best = 100 * delta_best / best_ate
                comparison_status = "COMPARABLE"
            elif c2f_row["status"] == "NOT_RUN" or fine_row["status"] == "NOT_RUN" or coarse_row["status"] == "NOT_RUN":
                comparison_status = "NOT_RUN"
                delta_fine = delta_best = percent_fine = percent_best = None
            elif c2f_ate is None:
                comparison_status = "C2F_NONPASS"
                delta_fine = delta_best = percent_fine = percent_best = None
            else:
                comparison_status = "PARENT_NONPASS"
                delta_fine = delta_best = percent_fine = percent_best = None
            pairwise.append({
                "dataset_order": dataset["order"], "dataset_key": key, "dataset_family": dataset["family"], "dataset_condition": dataset["condition"],
                "standard_label": item.label, "source": c2f_row["source"], "variant": item.variant, "fine_rank": item.fine_rank, "coarse_rank": item.coarse_rank,
                "c2f_status": c2f_row["status"], "c2f_ate_mean_cm": c2f_ate,
                "fine_parent": fine.label, "fine_status": fine_row["status"], "fine_ate_mean_cm": fine_ate,
                "coarse_parent": coarse.label, "coarse_status": coarse_row["status"], "coarse_ate_mean_cm": coarse_ate,
                "better_parent": best_label, "better_parent_ate_mean_cm": best_ate, "comparison_status": comparison_status,
                "delta_vs_fine_cm": delta_fine, "percent_delta_vs_fine": percent_fine, "beats_fine": comparable and delta_fine < 0,
                "delta_vs_better_parent_cm": delta_best, "percent_delta_vs_better_parent": percent_best, "beats_better_parent": comparable and delta_best < 0,
            })

    effects: list[dict[str, Any]] = []
    for item in items:
        if item.kind != "c2f":
            continue
        rows = [row for row in pairwise if row["standard_label"] == item.label]
        comparable = [row for row in rows if row["comparison_status"] == "COMPARABLE"]
        effects.append({
            "standard_label": item.label, "variant": item.variant, "fine_rank": item.fine_rank, "coarse_rank": item.coarse_rank, "source": source_description(item),
            "c2f_pass_count": sum(row["c2f_status"] == "PASS" for row in rows), "comparable_pairs": len(comparable),
            "beats_fine_count": sum(bool(row["beats_fine"]) for row in comparable), "beats_better_parent_count": sum(bool(row["beats_better_parent"]) for row in comparable),
            "median_percent_delta_vs_fine": median([float(row["percent_delta_vs_fine"]) for row in comparable]),
            "median_percent_delta_vs_better_parent": median([float(row["percent_delta_vs_better_parent"]) for row in comparable]),
            "c2f_nonpass_count": sum(row["comparison_status"] == "C2F_NONPASS" for row in rows), "parent_nonpass_count": sum(row["comparison_status"] == "PARENT_NONPASS" for row in rows),
        })
    effects.sort(key=lambda row: (-row["beats_better_parent_count"], row["median_percent_delta_vs_better_parent"] if row["median_percent_delta_vs_better_parent"] is not None else math.inf))

    fields = list(scorecard[0]) if scorecard else []
    write_csv(args.output_dir / "merged_scorecard.csv", scorecard, fields)
    write_csv(args.output_dir / "merged_pairwise_comparison.csv", pairwise, list(pairwise[0]) if pairwise else [])
    write_csv(args.output_dir / "c2f_effect_summary.csv", effects, list(effects[0]) if effects else [])
    write_csv(args.output_dir / "all_runs_long.csv", long_rows, ["dataset_order", "dataset_key", "dataset_family", "dataset_condition", "source", "standard_label", "kind", *schema])
    counts = Counter(row["status"] for row in scorecard)
    lines = [
        f"# {args.architecture.upper()} Step-R reduced complete C2F grid",
        "",
        f"- Selected full grid: {sum(item.kind == 'c2f' for item in items)} C2F pairs + {sum(item.kind != 'c2f' for item in items)} direct parents; no gray baseline.",
        f"- Reused from Step-Q without rerun: {sum(item.is_reused for item in items)} configurations per dataset.",
        f"- New Step-R cells per dataset: {sum(not item.is_reused for item in items)}.",
        "- C2F effects are computed only against direct parents on the same sequence; negative delta means lower/better C2F ATE.",
        f"- Current merged status counts: {dict(sorted(counts.items()))}.",
        "",
        "| C2F | Variant | F rank | C rank | PASS | Comparable | Beats better parent | Median Δ (%) | Source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in effects:
        delta = "" if row["median_percent_delta_vs_better_parent"] is None else f"{row['median_percent_delta_vs_better_parent']:+.2f}"
        lines.append(f"| {row['standard_label']} | {row['variant']} | {row['fine_rank']} | {row['coarse_rank']} | {row['c2f_pass_count']}/9 | {row['comparable_pairs']} | {row['beats_better_parent_count']}/{row['comparable_pairs']} | {delta} | {row['source']} |")
    (args.output_dir / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    total_cells = len(items) * len(datasets)
    print(f"[AGGREGATE] merged_rows={len(long_rows)}/{total_cells}; status_counts={dict(sorted(counts.items()))}")
    print(f"[WRITE] {args.output_dir / 'merged_pairwise_comparison.csv'}")
    print(f"[WRITE] {args.output_dir / 'c2f_effect_summary.csv'}")


if __name__ == "__main__":
    main()
