#!/usr/bin/env python3
"""Repair post-processing errors from saved trajectories without rerunning COMO."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
EVALUATOR_PATH = (
    SCRIPT_DIR.parent
    / "step_e_full_sequence_evaluation/run_full_sequence_evaluation.py"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "channel_selection_results/step_f_multi_dataset_evaluation"
)
DEFAULT_DATASET_PLAN = SCRIPT_DIR / "dataset_plan.json"
REPAIRABLE_REASON = "Trajectory timestamps are not strictly increasing:"


def load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "step_f_repair_evaluator", EVALUATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {EVALUATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = load_evaluator()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalculate saved ERROR_TRAJECTORY_EVALUATION rows affected by "
            "duplicate TUM ground-truth timestamps. COMO is never launched."
        )
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-plan", type=Path, default=DEFAULT_DATASET_PLAN)
    parser.add_argument(
        "--evo-ape", type=Path, default=evaluator.DEFAULT_EVO_APE
    )
    parser.add_argument(
        "--evo-rpe", type=Path, default=evaluator.DEFAULT_EVO_RPE
    )
    return parser.parse_args()


def sqlite_backup(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(source_path)
    destination = sqlite3.connect(destination_path)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()


def restore_result(row: sqlite3.Row):
    values = {
        field.name: row[field.name]
        for field in dataclasses.fields(evaluator.FullSequenceResult)
    }
    return evaluator.FullSequenceResult(**values)


def score_saved_result(
    result,
    dataset_dir: Path,
    matched_timestamps: list[float],
    minimum_coverage: float,
    completion_tolerance_seconds: float,
    evo_ape: Path,
    evo_rpe: Path,
) -> None:
    trajectory = Path(result.trajectory_path)
    keyframes = (
        Path(result.keyframe_trajectory_path)
        if result.keyframe_trajectory_path
        else None
    )
    metrics, trajectory_poses, associated, last_timestamp, frozen = (
        evaluator.evaluate_all_frames(trajectory, dataset_dir / "groundtruth.txt")
    )
    result.trajectory_poses = trajectory_poses
    result.associated_poses = associated
    result.last_timestamp = last_timestamp
    result.expected_matched_frames = len(matched_timestamps)
    result.coverage_ratio = trajectory_poses / len(matched_timestamps)

    if frozen:
        result.status = "FAIL_POSE_FROZEN"
        result.reason = "Final 30 poses froze while ground truth kept moving"
        return
    if result.coverage_ratio < minimum_coverage:
        result.status = "FAIL_INCOMPLETE"
        result.reason = (
            f"Coverage {result.coverage_ratio:.3f} is below "
            f"{minimum_coverage:.3f}"
        )
        return
    if last_timestamp < matched_timestamps[-1] - completion_tolerance_seconds:
        result.status = "FAIL_INCOMPLETE"
        result.reason = (
            f"Trajectory ended at {last_timestamp:.6f}, before final dataset "
            f"timestamp {matched_timestamps[-1]:.6f}"
        )
        return

    result.status = "PASS"
    result.reason = "Recovered from saved trajectory after duplicate-GT fix"
    result.se3_ate_rmse_m = metrics["rmse"]
    result.se3_ate_mean_m = metrics["mean"]
    result.se3_ate_median_m = metrics["median"]
    result.se3_ate_max_m = metrics["max"]
    result.se3_ate_std_m = metrics["std"]
    result.rotation_ape_rmse_deg = metrics["rotation_ape_rmse_deg"]
    result.rotation_ape_mean_deg = metrics["rotation_ape_mean_deg"]
    result.rotation_ape_max_deg = metrics["rotation_ape_max_deg"]
    result.translation_rpe_rmse_m = metrics["translation_rpe_rmse"]
    result.translation_rpe_max_m = metrics["translation_rpe_max"]
    result.rotation_rpe_rmse_deg = metrics["rotation_rpe_rmse_deg"]
    result.rotation_rpe_max_deg = metrics["rotation_rpe_max_deg"]
    result.allframe_sim3_rmse_m = metrics["legacy_sim3_rmse"]
    result.allframe_sim3_mean_m = metrics["legacy_sim3_mean"]
    result.allframe_sim3_scale = metrics["legacy_sim3_scale"]

    if keyframes is None or not keyframes.is_file():
        return
    legacy, legacy_poses = evaluator.core.evaluate_legacy_keyframe_sim3(
        keyframes,
        dataset_dir / "groundtruth.txt",
        matched_timestamps[0],
        matched_timestamps[-1],
    )
    result.keyframe_associated_poses = legacy_poses
    if legacy is not None:
        result.keyframe_sim3_rmse_m = legacy["legacy_sim3_rmse"]
        result.keyframe_sim3_mean_m = legacy["legacy_sim3_mean"]
        result.keyframe_sim3_scale = legacy["legacy_sim3_scale"]
    historical = evaluator.evaluate_historical_evo_ape(
        evo_ape, dataset_dir / "groundtruth.txt", keyframes
    )
    result.historical_evo_ape_rmse_m = historical["rmse"]
    result.historical_evo_ape_mean_m = historical["mean"]
    result.historical_evo_rpe_rmse_m = evaluator.evaluate_historical_evo_rpe(
        evo_rpe, dataset_dir / "groundtruth.txt", keyframes
    )


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.dataset_plan = args.dataset_plan.resolve()
    args.evo_ape = args.evo_ape.resolve()
    args.evo_rpe = args.evo_rpe.resolve()
    plan = json.loads(args.dataset_plan.read_text(encoding="utf-8"))
    dataset_root = Path(plan["dataset_root"])
    found: list[tuple[dict, Path, list[sqlite3.Row]]] = []

    for dataset in plan["datasets"]:
        database = (
            args.output_dir
            / "per_dataset"
            / dataset["key"]
            / "evaluations.sqlite3"
        )
        if not database.is_file():
            continue
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            connection.close()
            raise RuntimeError(f"SQLite integrity check failed: {database}: {integrity}")
        rows = connection.execute(
            """
            SELECT * FROM evaluations
            WHERE status='ERROR_TRAJECTORY_EVALUATION' AND reason LIKE ?
            ORDER BY id
            """,
            (REPAIRABLE_REASON + "%",),
        ).fetchall()
        connection.close()
        if rows:
            found.append((dataset, database, rows))

    total = sum(len(rows) for _, _, rows in found)
    print(f"[REPAIR PLAN] matching saved rows={total}; COMO will not be launched")
    for dataset, _, rows in found:
        print(f"  {dataset['key']}: {len(rows)}")
    if not args.execute:
        print("[DRY RUN] Add --execute to back up databases and repair these rows")
        return
    if total == 0:
        print("[DONE] No matching rows require repair")
        return

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = args.output_dir / "repair_backups" / stamp
    repaired = 0
    failed = 0
    for dataset, database, rows in found:
        sqlite_backup(database, backup_dir / f"{dataset['key']}.sqlite3")
        dataset_dir = dataset_root / dataset["directory_name"]
        matched = evaluator.read_timestamp_index(dataset_dir / "matched_rgb.txt")
        evaluation_plan_path = database.parent / "evaluation_plan.json"
        evaluation_plan = json.loads(evaluation_plan_path.read_text(encoding="utf-8"))
        minimum_coverage = float(evaluation_plan.get("minimum_coverage", 0.90))
        completion_tolerance = 0.10
        store = evaluator.ResultStore(database)
        try:
            for original in rows:
                result = restore_result(original)
                if not result.trajectory_path or not Path(result.trajectory_path).is_file():
                    print(f"[SKIP] {dataset['key']} {result.label}: missing trajectory")
                    failed += 1
                    continue
                try:
                    score_saved_result(
                        result,
                        dataset_dir,
                        matched,
                        minimum_coverage,
                        completion_tolerance,
                        args.evo_ape,
                        args.evo_rpe,
                    )
                except (FloatingPointError, OSError, RuntimeError, ValueError) as error:
                    print(f"[UNRECOVERED] {dataset['key']} {result.label}: {error}")
                    failed += 1
                    continue
                store.add(result)
                repaired += 1
                ate = (
                    f"{result.historical_evo_ape_mean_m * 100:.4f}cm"
                    if result.historical_evo_ape_mean_m is not None
                    else "n/a"
                )
                print(
                    f"[REPAIRED] {dataset['key']} {result.label}: "
                    f"status={result.status} historical_ATE_mean={ate}"
                )
            evaluator.export_results(store, database.parent)
        finally:
            store.close()

    print(f"[BACKUP] {backup_dir}")
    print(f"[DONE] repaired={repaired} unrecovered={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
