#!/usr/bin/env python3
"""Aggregate fr2/fr3 QueensCAMP: 14 degradations × 8 configurations × 3 runs.

The primary ATE exactly follows the earlier full-sequence experiments:
keyframe evo_ape mean after alignment and scale correction.  Raw ATE is never
pooled across the two base sequences; cross-condition comparisons use each
dataset's ratio to the historical four-channel CNN baseline.
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

_CACHE = Path(tempfile.gettempdir()) / "queenscamp_fr2_fr3_aggregate_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "channel_selection_results/step_i_queenscamp_fr2_fr3_evaluation/three_repeats"
DEFAULT_PLAN = SCRIPT_DIR / "queenscamp_fr2_fr3_3x_plan.json"
DEFAULT_CANDIDATES = SCRIPT_DIR / "top7_plus_gray_candidate_plan.json"
BASELINE = "5,29,40,52"
GRAY = "gray"
REPEATS = 3
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


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--candidate-plan", type=Path, default=DEFAULT_CANDIDATES)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(plan: dict[str, Any], candidate_plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets, candidates = plan.get("datasets"), candidate_plan.get("candidates")
    if plan.get("protocol") != "top7_plus_gray_queenscamp_fr2_fr3_3x_v1":
        raise ValueError("Unexpected experiment protocol")
    if candidate_plan.get("protocol") != "top7_plus_gray_candidate_plan_v1":
        raise ValueError("Unexpected candidate protocol")
    if not isinstance(datasets, list) or len(datasets) != 14 or plan.get("dataset_count") != 14:
        raise ValueError("Expected 14 datasets")
    if {item.get("family") for item in datasets} != {"fr2_desk", "fr3_long_office_household"}:
        raise ValueError("Expected seven fr2 and seven fr3 degradation datasets")
    if any(sum(item.get("family") == family for item in datasets) != 7 for family in {"fr2_desk", "fr3_long_office_household"}):
        raise ValueError("Each base sequence must have seven degradations")
    if not isinstance(candidates, list) or len(candidates) != 8 or plan.get("configuration_count") != 8:
        raise ValueError("Expected Top-7 plus gray")
    if plan.get("replicates_per_configuration") != REPEATS or plan.get("timeout_seconds_per_run") != 500:
        raise ValueError("Expected three repeats and 500-second timeout")
    if plan.get("planned_runs") != 336:
        raise ValueError("Expected 336 planned runs")
    if [x.get("order") for x in datasets] != list(range(1, 15)):
        raise ValueError("Dataset order must be 1..14")
    expected = ["5,6,24,29", "1,26,30,40", "15,17,52,59", "1,5,24,29", "5,6,15,35", "6,10,34,41", BASELINE, GRAY]
    if [x.get("candidate_key") for x in candidates] != expected or candidates[-1].get("channels") is not None:
        raise ValueError("Candidate plan does not match frozen Top-7 + gray protocol")
    if len({x["key"] for x in datasets}) != 14:
        raise ValueError("Duplicate dataset key")
    return datasets, candidates


def read_db(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], []
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {path}: {integrity}")
    columns = [row[1] for row in conn.execute("PRAGMA table_info(evaluations)")]
    rows = [dict(row) for row in conn.execute("SELECT * FROM evaluations ORDER BY replicate, id")]
    conn.close()
    return rows, columns


def stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("mean", "std", "median", "min", "max")}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values), "min": min(values), "max": max(values),
    }


def gmean(values: list[float]) -> float | None:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(statistics.fmean(math.log(value) for value in values))


def config_name(key: str) -> str:
    return "gray" if key == GRAY else f"[{key}]"


SUMMARY_FIELDS = [
    "dataset_order", "family", "dataset_key", "degradation", "source_rank", "label", "candidate_key", "configuration",
    "is_historical_cnn_baseline", "is_gray_control", "planned_replicates", "completed_replicates", "pass_count",
    "pass_rate_planned", "failure_count_completed", "missing_replicates", "status_counts", "failure_frames", "failure_timestamps",
]
for _source, (_name, _scale) in METRICS.items():
    SUMMARY_FIELDS += [f"{_name}_{suffix}" for suffix in ("mean", "std", "median", "min", "max")]
SUMMARY_FIELDS += ["reliability_aware_dataset_rank", "mean_ate_ratio_to_historical_cnn_baseline"]


def collect(output: Path, datasets: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    allowed = {candidate["candidate_key"] for candidate in candidates}
    by_key = {candidate["candidate_key"]: candidate for candidate in candidates}
    raw, summaries = [], []
    authoritative_columns: list[str] | None = None
    for dataset in datasets:
        rows, columns = read_db(output / "per_dataset" / dataset["key"] / "evaluations.sqlite3")
        if columns:
            if authoritative_columns is None:
                authoritative_columns = columns
            elif columns != authoritative_columns:
                raise ValueError(f"Database schema differs for {dataset['key']}")
        grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in allowed}
        seen: set[tuple[str, int]] = set()
        for row in rows:
            key, repeat = str(row["candidate_key"]), int(row["replicate"])
            if key not in allowed or not 0 <= repeat < REPEATS:
                raise ValueError(f"Unexpected row in {dataset['key']}: {key}, repeat={repeat}")
            if (key, repeat) in seen:
                raise ValueError(f"Duplicate row in {dataset['key']}: {(key, repeat)}")
            seen.add((key, repeat)); grouped[key].append(row)
            raw.append({"dataset_order": dataset["order"], "family": dataset["family"], "dataset_key": dataset["key"], "degradation": dataset["degradation"], "source_rank": by_key[key]["source_rank"], "is_historical_cnn_baseline": key == BASELINE, "is_gray_control": key == GRAY, **row})
        for candidate in candidates:
            key, cell = candidate["candidate_key"], sorted(grouped[candidate["candidate_key"]], key=lambda x: x["replicate"])
            passed = [row for row in cell if row["status"] == "PASS"]
            counts = Counter(str(row["status"]) for row in cell)
            summary: dict[str, Any] = {
                "dataset_order": dataset["order"], "family": dataset["family"], "dataset_key": dataset["key"], "degradation": dataset["degradation"],
                "source_rank": candidate["source_rank"], "label": candidate["label"], "candidate_key": key, "configuration": config_name(key),
                "is_historical_cnn_baseline": key == BASELINE, "is_gray_control": key == GRAY, "planned_replicates": REPEATS,
                "completed_replicates": len(cell), "pass_count": len(passed), "pass_rate_planned": len(passed) / REPEATS,
                "failure_count_completed": len(cell) - len(passed), "missing_replicates": REPEATS - len(cell), "status_counts": json.dumps(dict(sorted(counts.items()))),
                "failure_frames": json.dumps([row["failure_frame_index"] for row in cell if row["status"] != "PASS"]),
                "failure_timestamps": json.dumps([row["failure_timestamp"] for row in cell if row["status"] != "PASS"]),
            }
            for source, (name, scale) in METRICS.items():
                vals = [float(row[source]) * scale for row in passed if row.get(source) is not None and math.isfinite(float(row[source]))]
                for suffix, value in stats(vals).items(): summary[f"{name}_{suffix}"] = value
            summaries.append(summary)
    return raw, summaries, authoritative_columns or []


def rank_cells(summaries: list[dict[str, Any]], datasets: list[dict[str, Any]]) -> None:
    for dataset in datasets:
        cells = [row for row in summaries if row["dataset_key"] == dataset["key"]]
        for rank, row in enumerate(sorted(cells, key=lambda x: (-x["pass_count"], x["historical_ate_mean_cm_mean"] if x["historical_ate_mean_cm_mean"] is not None else math.inf, x["source_rank"])), 1):
            row["reliability_aware_dataset_rank"] = rank
        baseline = next(row for row in cells if row["candidate_key"] == BASELINE)["historical_ate_mean_cm_mean"]
        for row in cells:
            ate = row["historical_ate_mean_cm_mean"]
            row["mean_ate_ratio_to_historical_cnn_baseline"] = ate / baseline if ate is not None and baseline is not None and baseline > 0 else None


def aggregate_rows(summaries: list[dict[str, Any]], datasets: list[dict[str, Any]], candidates: list[dict[str, Any]], family: str | None) -> list[dict[str, Any]]:
    scoped = [d for d in datasets if family is None or d["family"] == family]
    total = len(scoped) * REPEATS
    rows = []
    for candidate in candidates:
        key = candidate["candidate_key"]
        cells = [r for r in summaries if r["candidate_key"] == key and r["dataset_key"] in {d["key"] for d in scoped}]
        ratios = [r["mean_ate_ratio_to_historical_cnn_baseline"] for r in cells if r["mean_ate_ratio_to_historical_cnn_baseline"] is not None]
        rows.append({
            "family": family or "all_14_datasets", "source_rank": candidate["source_rank"], "label": candidate["label"], "candidate_key": key, "configuration": config_name(key),
            "planned_runs": total, "completed_runs": sum(r["completed_replicates"] for r in cells), "total_pass_count": sum(r["pass_count"] for r in cells),
            "overall_pass_rate_planned": sum(r["pass_count"] for r in cells) / total, "datasets_with_3_of_3_pass": sum(r["pass_count"] == 3 for r in cells),
            "datasets_with_any_pass": sum(r["pass_count"] > 0 for r in cells), "baseline_comparable_datasets": len(ratios),
            "beats_historical_baseline_dataset_means": sum(value < 1 for value in ratios), "geomean_mean_ate_ratio_to_historical_baseline": gmean(ratios),
            "mean_reliability_aware_dataset_rank": statistics.fmean(float(r["reliability_aware_dataset_rank"]) for r in cells),
        })
    rows.sort(key=lambda r: (-r["total_pass_count"], -r["datasets_with_3_of_3_pass"], r["geomean_mean_ate_ratio_to_historical_baseline"] if r["geomean_mean_ate_ratio_to_historical_baseline"] is not None else math.inf, r["mean_reliability_aware_dataset_rank"]))
    for rank, row in enumerate(rows, 1): row["aggregate_rank"] = rank
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else value for key, value in row.items()})


def heatmap(output: Path, summaries: list[dict[str, Any]], datasets: list[dict[str, Any]], candidates: list[dict[str, Any]], family: str, field: str, title: str, filename: str, fmt: str) -> None:
    scoped = [d for d in datasets if d["family"] == family]
    lookup = {(r["candidate_key"], r["dataset_key"]): r for r in summaries}
    matrix = np.asarray([[lookup[(c["candidate_key"], d["key"])].get(field) for d in scoped] for c in candidates], dtype=float)
    fig, ax = plt.subplots(figsize=(12, 6.3)); image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(7), [d["degradation"] for d in scoped], rotation=25, ha="right")
    ax.set_yticks(range(8), [config_name(c["candidate_key"]) for c in candidates]); ax.set_title(title)
    for i in range(8):
        for j in range(7): ax.text(j, i, "—" if np.isnan(matrix[i,j]) else format(matrix[i,j], fmt), ha="center", va="center", fontsize=8)
    if np.any(np.isfinite(matrix)): fig.colorbar(image, ax=ax, shrink=.8)
    (output / "plots").mkdir(parents=True, exist_ok=True); fig.tight_layout(); fig.savefig(output / "plots" / filename, dpi=180, bbox_inches="tight"); plt.close(fig)


def optional(value: float | None, digits: int = 3) -> str: return "—" if value is None else f"{value:.{digits}f}"


def report(output: Path, raw: list[dict[str, Any]], summaries: list[dict[str, Any]], overall: list[dict[str, Any]], datasets: list[dict[str, Any]]) -> None:
    passing = sum(r["status"] == "PASS" for r in raw)
    lines = ["# QueensCAMP fr2/fr3 robustness: Top-7 + gray, three repeats", "", f"- Completed: {len(raw)}/336; PASS: {passing}; non-PASS: {len(raw)-passing}.", "- Primary accuracy metric: historical keyframe `evo_ape` ATE mean with `--align --correct_scale`.", "- Mean/std/median metrics use PASS runs only. Reliability uses all three planned repetitions.", "- ATE is compared within each base sequence/degradation. Aggregate accuracy uses the geometric mean of the ratio to CNN baseline `[5,29,40,52]`, never pooled raw ATE.", "", "## Overall reliability-first ranking", "", "|rank|configuration|PASS/42|datasets 3/3|beats CNN baseline|ATE ratio geomean|", "|---:|---|---:|---:|---:|---:|"]
    for r in overall:
        lines.append(f"|{r['aggregate_rank']}|`{r['configuration']}`|{r['total_pass_count']}/42|{r['datasets_with_3_of_3_pass']}/14|{r['beats_historical_baseline_dataset_means']}/{r['baseline_comparable_datasets']}|{optional(r['geomean_mean_ate_ratio_to_historical_baseline'],4)}|")
    lines += ["", "## Best configuration per degradation", "", "|family|degradation|configuration|PASS|mean ATE/cm|rank|", "|---|---|---|---:|---:|---:|"]
    for d in datasets:
        cells = [r for r in summaries if r["dataset_key"] == d["key"]]; best = min(cells, key=lambda r: r["reliability_aware_dataset_rank"])
        lines.append(f"|{d['family']}|{d['degradation']}|`{best['configuration']}`|{best['pass_count']}/3|{optional(best['historical_ate_mean_cm_mean'])}|{best['reliability_aware_dataset_rank']}|")
    lines += ["", "Detailed per-dataset means, variation, RPE diagnostics and failure locations: `per_dataset_configuration_summary.csv`."]
    (output / "aggregate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    a = args(); output = a.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    datasets, candidates = validate(load(a.experiment_plan.resolve()), load(a.candidate_plan.resolve()))
    raw, summaries, raw_columns = collect(output, datasets, candidates); rank_cells(summaries, datasets)
    family_rows = [row for family in ("fr2_desk", "fr3_long_office_household") for row in aggregate_rows(summaries, datasets, candidates, family)]
    overall = aggregate_rows(summaries, datasets, candidates, None)
    metadata = ["dataset_order", "family", "dataset_key", "degradation", "source_rank", "is_historical_cnn_baseline", "is_gray_control"]
    write_csv(output / "all_runs_raw.csv", raw, metadata + raw_columns)
    write_csv(output / "per_dataset_configuration_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(output / "per_family_configuration_summary.csv", family_rows, list(family_rows[0]))
    write_csv(output / "configuration_overall_summary.csv", overall, list(overall[0]))
    for family, short in (("fr2_desk", "fr2_desk"), ("fr3_long_office_household", "fr3_long_office_household")):
        heatmap(output, summaries, datasets, candidates, family, "pass_count", f"{short}: PASS count out of three", f"{short}_pass_count_heatmap.png", ".0f")
        heatmap(output, summaries, datasets, candidates, family, "historical_ate_mean_cm_mean", f"{short}: mean historical keyframe ATE (cm)", f"{short}_ate_mean_heatmap.png", ".2f")
        heatmap(output, summaries, datasets, candidates, family, "historical_ate_mean_cm_std", f"{short}: ATE standard deviation (cm)", f"{short}_ate_std_heatmap.png", ".2f")
    report(output, raw, summaries, overall, datasets)
    protocol = {"protocol": "top7_plus_gray_queenscamp_fr2_fr3_3x_aggregate_v1", "planned_runs": 336, "completed_runs": len(raw), "pass_runs": sum(r["status"] == "PASS" for r in raw), "mean_policy": "PASS observations only", "primary_metric": "keyframe evo_ape ATE mean with --align --correct_scale", "ranking": "PASS count desc, 3/3 datasets desc, baseline-normalized ATE geomean asc"}
    (output / "aggregate_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"[AGGREGATE] completed={len(raw)}/336 PASS={protocol['pass_runs']} output={output}")


if __name__ == "__main__": main()
