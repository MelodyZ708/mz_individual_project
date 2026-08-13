#!/usr/bin/env python3
"""Freeze the baseline+2%, RPE-safe MVS survivors for full-sequence evaluation."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_DB = (
    PROJECT_ROOT
    / "channel_selection_results/step_d_fail_fast_evaluation/"
    "r070_bruteforce_v2/evaluations.sqlite3"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_e_full_sequence_evaluation/"
    "second_round_baseline_plus2_rpe_safe"
)
BASELINE_KEY = "5,29,40,52"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create the frozen second-round full-sequence candidate plan.",
    )
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ate-margin-percent", type=float, default=2.0)
    parser.add_argument("--translation-rpe-max-cm", type=float, default=6.0)
    parser.add_argument("--rotation-rpe-max-deg", type=float, default=5.0)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=3713,
        help="Abort rather than silently changing the agreed candidate population.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_db = args.source_db.resolve()
    output_dir = args.output_dir.resolve()
    if not source_db.is_file():
        raise FileNotFoundError(source_db)
    if args.ate_margin_percent < 0:
        raise ValueError("--ate-margin-percent cannot be negative")
    if args.translation_rpe_max_cm <= 0 or args.rotation_rpe_max_deg <= 0:
        raise ValueError("RPE thresholds must be positive")

    connection = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        raise RuntimeError(f"Source database integrity check failed: {integrity}")
    baseline = connection.execute(
        """
        SELECT * FROM evaluations
        WHERE stage='bruteforce' AND replicate=0 AND status='PASS'
          AND candidate_key=?
        """,
        (BASELINE_KEY,),
    ).fetchone()
    if baseline is None:
        raise RuntimeError(f"Baseline {BASELINE_KEY} is missing from brute-force PASS rows")

    cutoff_m = float(baseline["ate_rmse_m"]) * (
        1.0 + args.ate_margin_percent / 100.0
    )
    translation_cutoff_m = args.translation_rpe_max_cm / 100.0
    rows = connection.execute(
        """
        SELECT candidate_key, channels_json, associated_poses,
               ate_rmse_m, ate_mean_m,
               translation_rpe_max_m, rotation_rpe_max_deg
        FROM evaluations
        WHERE stage='bruteforce'
          AND replicate=0
          AND status='PASS'
          AND associated_poses=40
          AND ate_rmse_m<=?
          AND translation_rpe_max_m<=?
          AND rotation_rpe_max_deg<=?
        ORDER BY ate_rmse_m, translation_rpe_max_m,
                 rotation_rpe_max_deg, candidate_key
        """,
        (cutoff_m, translation_cutoff_m, args.rotation_rpe_max_deg),
    ).fetchall()
    connection.close()

    if len(rows) != args.expected_count:
        raise RuntimeError(
            f"Selection produced {len(rows):,} candidates, expected "
            f"{args.expected_count:,}; no plan was written"
        )
    if BASELINE_KEY not in {row["candidate_key"] for row in rows}:
        raise RuntimeError("The known baseline did not survive its own qualification gate")

    candidates = []
    for rank, row in enumerate(rows, start=1):
        channels = [int(value) for value in row["candidate_key"].split(",")]
        label = f"candidate_{rank:04d}_ch_" + "_".join(map(str, channels))
        candidates.append(
            {
                "selection_rank": rank,
                "label": label,
                "channels": channels,
                "candidate_key": row["candidate_key"],
                "role": "MVS PASS; baseline+2% ATE and RPE-safe second-round candidate",
                "mvs": {
                    "associated_poses": row["associated_poses"],
                    "se3_ate_rmse_cm": row["ate_rmse_m"] * 100.0,
                    "se3_ate_mean_cm": row["ate_mean_m"] * 100.0,
                    "translation_rpe_max_cm": row["translation_rpe_max_m"] * 100.0,
                    "rotation_rpe_max_deg": row["rotation_rpe_max_deg"],
                },
            }
        )

    payload = {
        "protocol": "second_round_full_sequence_baseline_plus2_rpe_safe_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_database": str(source_db),
        "source_stage": "bruteforce replicate=0",
        "selection": {
            "required_status": "PASS",
            "required_associated_poses": 40,
            "baseline_candidate_key": BASELINE_KEY,
            "baseline_mvs_se3_ate_rmse_cm": baseline["ate_rmse_m"] * 100.0,
            "ate_margin_percent": args.ate_margin_percent,
            "mvs_se3_ate_rmse_cutoff_cm": cutoff_m * 100.0,
            "translation_rpe_max_cutoff_cm": args.translation_rpe_max_cm,
            "rotation_rpe_max_cutoff_deg": args.rotation_rpe_max_deg,
            "selected_count": len(candidates),
        },
        "full_sequence_primary_metric": (
            "keyframe evo_ape ATE mean from: evo_ape tum groundtruth.txt "
            "data_tum.txt --align --correct_scale"
        ),
        "timeout_seconds_per_run": 300,
        "candidates": candidates,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "candidate_plan.json"
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = output_dir / "candidate_plan.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "selection_rank",
                "label",
                "channels",
                "mvs_se3_ate_rmse_cm",
                "mvs_se3_ate_mean_cm",
                "translation_rpe_max_cm",
                "rotation_rpe_max_deg",
            ]
        )
        for item in candidates:
            writer.writerow(
                [
                    item["selection_rank"],
                    item["label"],
                    item["candidate_key"],
                    item["mvs"]["se3_ate_rmse_cm"],
                    item["mvs"]["se3_ate_mean_cm"],
                    item["mvs"]["translation_rpe_max_cm"],
                    item["mvs"]["rotation_rpe_max_deg"],
                ]
            )

    print("=" * 78)
    print("SECOND-ROUND FULL-SEQUENCE CANDIDATE PLAN")
    print("=" * 78)
    print(f"Source DB: {source_db}")
    print(f"Baseline MVS SE(3) ATE RMSE: {baseline['ate_rmse_m'] * 100:.4f} cm")
    print(f"ATE cutoff (+{args.ate_margin_percent:.1f}%): {cutoff_m * 100:.4f} cm")
    print(
        f"RPE cutoffs: translation <= {args.translation_rpe_max_cm:.1f} cm; "
        f"rotation <= {args.rotation_rpe_max_deg:.1f} deg"
    )
    print(f"Frozen candidates: {len(candidates):,}")
    print(f"Primary full-sequence metric: historical evo_ape ATE mean")
    print(f"Plan: {plan_path}")
    print(f"Readable table: {csv_path}")
    print("No COMO evaluation was launched.")


if __name__ == "__main__":
    main()
