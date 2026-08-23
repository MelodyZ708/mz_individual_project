#!/usr/bin/env python3
"""Generate data figures for the Chinese MVS stage-one advisor report."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
MANIFEST = Path(
    "/home/melody/data/tum/"
    "rgbd_dataset_freiburg1_desk_lightswitch_mvs_failure_anchor_"
    "idx248_brighten_dim_50f/frame_manifest.csv"
)
MVS_DB = (
    ROOT
    / "channel_selection_results/step_d_fail_fast_evaluation/"
    "r070_bruteforce_v2/evaluations.sqlite3"
)
FULL_DB = (
    ROOT
    / "channel_selection_results/step_e_full_sequence_evaluation/"
    "fr1_desk_lightswitch/evaluations.sqlite3"
)


def brightness_figure() -> None:
    with MANIFEST.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    x = np.array([int(row["mvs_index"]) for row in rows])
    mean = np.array([float(row["mean_luminance"]) for row in rows])
    median = np.array([float(row["median_luminance"]) for row in rows])

    fig, axis = plt.subplots(figsize=(10.8, 5.4))
    axis.axvspan(-0.5, 9.5, color="#b8c8dc", alpha=0.35, label="Warm-up (0-9)")
    axis.axvspan(9.5, 49.5, color="#f3d49b", alpha=0.20, label="Scored (10-49)")
    axis.plot(x, mean, color="#165a9f", linewidth=2.2, label="Mean luminance")
    axis.plot(x, median, color="#5b2a86", linewidth=1.8, label="Median luminance")
    events = [
        (5, "Brightening\nonset", "#1f8a70", 5.0, 111),
        (13, "Failure\nanchor", "#c0392b", 11.7, 123),
        (14, "First printed\nNaN", "#922b21", 15.6, 142),
        (43, "Dimming\nonset", "#d35400", 43.0, 145),
        (49, "Strong\ndimming", "#7d6608", 49.0, 145),
    ]
    for frame, label, color, text_x, text_y in events:
        axis.axvline(frame, color=color, linestyle="--", linewidth=1.4)
        axis.annotate(
            label,
            xy=(frame, mean[frame]),
            xytext=(text_x, text_y),
            ha="center",
            fontsize=8.5,
            color=color,
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
        )
    axis.set_xlim(-0.5, 49.5)
    axis.set_ylim(100, 265)
    axis.set_xlabel("MVS frame index")
    axis.set_ylabel("Luminance (0-255)")
    axis.set_title("FAIL50 MVS: brightening, saturation and subsequent dimming")
    axis.grid(alpha=0.22)
    axis.legend(loc="lower center", ncol=4, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "mvs_brightness_events.png", dpi=220)
    plt.close(fig)


def generalisation_figure() -> None:
    full = sqlite3.connect(FULL_DB)
    full.row_factory = sqlite3.Row
    mvs = sqlite3.connect(MVS_DB)
    mvs.row_factory = sqlite3.Row
    full_rows = full.execute(
        """
        SELECT label,candidate_key,se3_ate_rmse_m
        FROM evaluations
        WHERE status='PASS' AND candidate_key!='gray'
        """
    ).fetchall()
    points = []
    for row in full_rows:
        short = mvs.execute(
            """
            SELECT ate_rmse_m FROM evaluations
            WHERE candidate_key=? AND status='PASS' AND replicate=0
            ORDER BY CASE stage WHEN 'bruteforce' THEN 1 WHEN 'rescue' THEN 2 ELSE 3 END
            LIMIT 1
            """,
            (row["candidate_key"],),
        ).fetchone()
        if short is not None:
            points.append(
                (
                    row["label"],
                    row["candidate_key"],
                    short["ate_rmse_m"] * 100,
                    row["se3_ate_rmse_m"] * 100,
                )
            )
    full.close()
    mvs.close()

    fig, axis = plt.subplots(figsize=(9.5, 5.8))
    for label, channels, x_value, y_value in points:
        baseline = label == "known_cnn_baseline"
        axis.scatter(
            x_value,
            y_value,
            s=105 if baseline else 70,
            color="#c0392b" if baseline else "#1769aa",
            marker="*" if baseline else "o",
            zorder=3,
        )
        annotation = "baseline" if baseline else channels
        offset = (6, -13) if label == "mvs_best_ate" else (6, 5)
        axis.annotate(
            annotation,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.2,
        )
    axis.set_xlabel("MVS all-frame SE(3) ATE RMSE (cm)")
    axis.set_ylabel("Full-sequence all-frame SE(3) ATE RMSE (cm)")
    axis.set_title("Low MVS ATE did not imply low full-sequence ATE")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "mvs_vs_full_initial_validation.png", dpi=220)
    plt.close(fig)


def search_outcome_figure() -> None:
    labels = ["Legal\ncombinations", "MVS PASS", "MVS FAIL", "RPE-safe\nPASS", "Finalists"]
    values = [55554, 25003, 30551, 7335, 20]
    colors = ["#566573", "#1e8449", "#b03a2e", "#2874a6", "#7d3c98"]
    fig, axis = plt.subplots(figsize=(10.2, 5.1))
    bars = axis.bar(labels, values, color=colors, width=0.66)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 900,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axis.set_ylabel("Number of four-channel combinations")
    axis.set_title("MVS search funnel")
    axis.set_ylim(0, 61000)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "mvs_search_funnel.png", dpi=220)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    brightness_figure()
    generalisation_figure()
    search_outcome_figure()
    print(f"Figures written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
