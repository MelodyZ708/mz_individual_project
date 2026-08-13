#!/usr/bin/env python3
"""Compare the three fr2/lightswitch failure reruns with their original rows."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = PROJECT_ROOT / "channel_selection_results/step_f_multi_dataset_evaluation"
ORIGINAL_DB = RESULT_ROOT / "per_dataset/fr2_desk_lightswitch/evaluations.sqlite3"
RERUN_ROOT = RESULT_ROOT / "repeat_checks/fr2_desk_lightswitch_failed_rerun"
RERUN_DB = RERUN_ROOT / "evaluations.sqlite3"
EXPECTED = (
    "1,26,30,40",
    "1,5,24,29",
    "5,29,40,52",
)


def read_rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        connection.close()
        raise RuntimeError(f"SQLite integrity check failed: {path}")
    rows = {
        row["candidate_key"]: dict(row)
        for row in connection.execute("SELECT * FROM evaluations ORDER BY id")
    }
    connection.close()
    return rows


def cm(row: dict | None, field: str) -> float | None:
    value = row.get(field) if row else None
    return None if value is None else float(value) * 100.0


def blank(value):
    return "" if value is None else value


def main() -> None:
    original = read_rows(ORIGINAL_DB)
    rerun = read_rows(RERUN_DB)
    RERUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in EXPECTED:
        first = original.get(key)
        second = rerun.get(key)
        rows.append(
            {
                "candidate_key": key,
                "original_status": first["status"] if first else "MISSING",
                "original_failure_frame": first.get("failure_frame_index") if first else None,
                "original_failure_timestamp": first.get("failure_timestamp") if first else None,
                "rerun_status": second["status"] if second else "NOT_RUN",
                "rerun_failure_frame": second.get("failure_frame_index") if second else None,
                "rerun_failure_timestamp": second.get("failure_timestamp") if second else None,
                "rerun_historical_ate_mean_cm": cm(second, "historical_evo_ape_mean_m"),
                "rerun_historical_ate_rmse_cm": cm(second, "historical_evo_ape_rmse_m"),
                "rerun_historical_rpe_rmse_cm": cm(second, "historical_evo_rpe_rmse_m"),
                "rerun_se3_ate_rmse_cm": cm(second, "se3_ate_rmse_m"),
                "rerun_translation_rpe_max_cm": cm(second, "translation_rpe_max_m"),
                "rerun_rotation_rpe_max_deg": (
                    second.get("rotation_rpe_max_deg") if second else None
                ),
                "rerun_coverage_ratio": second.get("coverage_ratio") if second else None,
            }
        )
    csv_path = RERUN_ROOT / "comparison_with_original.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {field: blank(value) for field, value in row.items()} for row in rows
        )

    lines = [
        "# fr2/desk/lightswitch failed-configuration rerun",
        "",
        "The original rows remain unchanged in the primary Step-F database.",
        "",
        "| channels | original | original failure frame | rerun | rerun ATE mean (cm) | rerun failure frame |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                str(blank(row[field]))
                for field in (
                    "candidate_key",
                    "original_status",
                    "original_failure_frame",
                    "rerun_status",
                    "rerun_historical_ate_mean_cm",
                    "rerun_failure_frame",
                )
            )
            + " |"
        )
    (RERUN_ROOT / "comparison_with_original.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[COMPARISON] {csv_path}")


if __name__ == "__main__":
    main()
