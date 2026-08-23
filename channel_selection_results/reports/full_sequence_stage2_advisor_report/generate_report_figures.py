#!/usr/bin/env python3
"""Generate figures for the Chinese stage-two full-sequence report."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
FULL_DB = (
    ROOT
    / "channel_selection_results/step_e_full_sequence_evaluation/"
    "second_round_baseline_plus2_rpe_safe/evaluations.sqlite3"
)
MVS_DB = (
    ROOT
    / "channel_selection_results/step_d_fail_fast_evaluation/"
    "r070_bruteforce_v2/evaluations.sqlite3"
)


def selection_funnel() -> None:
    labels = [
        "MVS PASS",
        "Within baseline\n+2% ATE",
        "+ MVS RPE-safe",
        "Full-sequence\nPASS",
        "Beat baseline",
    ]
    values = [25003, 14492, 3713, 2835, 6]
    colors = ["#566573", "#2874a6", "#7d3c98", "#1e8449", "#d35400"]
    fig, axis = plt.subplots(figsize=(10.6, 5.2))
    bars = axis.bar(labels, values, color=colors, width=0.66)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 480,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    axis.set_ylabel("Number of four-channel combinations")
    axis.set_title("Stage-two selection and full-sequence evaluation funnel")
    axis.set_ylim(0, 28000)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "stage2_selection_funnel.png", dpi=220)
    plt.close(fig)


def top_ranking() -> None:
    connection = sqlite3.connect(FULL_DB)
    rows = connection.execute(
        """
        SELECT candidate_key,historical_evo_ape_mean_m
        FROM evaluations WHERE status='PASS'
        ORDER BY historical_evo_ape_mean_m LIMIT 12
        """
    ).fetchall()
    connection.close()
    labels = [row[0] for row in rows][::-1]
    values = np.array([row[1] * 100 for row in rows][::-1])
    baseline_value = next(value for label, value in zip(labels, values) if label == "5,29,40,52")
    colors = ["#c0392b" if label == "5,29,40,52" else "#2471a3" for label in labels]
    fig, axis = plt.subplots(figsize=(10.2, 6.7))
    y_positions = np.arange(len(labels))
    axis.hlines(
        y_positions,
        np.minimum(values, baseline_value),
        np.maximum(values, baseline_value),
        color="#aab7b8",
        linewidth=1.5,
        zorder=1,
    )
    axis.scatter(values, y_positions, color=colors, s=72, zorder=3)
    axis.axvline(
        baseline_value,
        color="#c0392b",
        linestyle="--",
        linewidth=1.5,
        label=f"baseline = {baseline_value:.4f} cm",
    )
    for y_position, value in zip(y_positions, values):
        axis.text(
            value + 0.035,
            y_position,
            f"{value:.4f}",
            va="center",
            fontsize=8.5,
        )
    axis.set_yticks(y_positions, labels)
    axis.set_xlabel("Historical keyframe evo_ape ATE mean (cm; lower is better)")
    axis.set_title("Top full-sequence combinations and known baseline")
    axis.set_xlim(min(values) - 0.18, max(values) + 0.35)
    axis.grid(axis="x", alpha=0.22)
    axis.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "stage2_full_top_ranking.png", dpi=220)
    plt.close(fig)


def failure_distribution() -> None:
    connection = sqlite3.connect(FULL_DB)
    frames = np.array(
        [
            row[0]
            for row in connection.execute(
                "SELECT failure_frame_index FROM evaluations "
                "WHERE status='FAIL_TRACKING_NAN'"
            )
        ],
        dtype=int,
    )
    connection.close()
    bins = np.arange(0, 576, 8)
    fig, axis = plt.subplots(figsize=(10.6, 5.2))
    axis.hist(frames, bins=bins, color="#b03a2e", edgecolor="white", linewidth=0.6)
    axis.axvspan(249, 284, color="#f4d03f", alpha=0.28, label="Original challenge region (249-284)")
    axis.axvline(249, color="#7b241c", linestyle="--", linewidth=1.2)
    axis.set_xlabel("Full-sequence failure frame index")
    axis.set_ylabel("FAIL_TRACKING_NAN count")
    axis.set_title("Full-sequence failure localisation (878 failures)")
    axis.set_xlim(0, 573)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "stage2_failure_frames.png", dpi=220)
    plt.close(fig)


def mvs_full_relationship() -> None:
    connection = sqlite3.connect(FULL_DB)
    connection.execute("ATTACH ? AS mvs", (str(MVS_DB),))
    rows = connection.execute(
        """
        SELECT m.ate_rmse_m*100,e.historical_evo_ape_mean_m*100,
               e.candidate_key
        FROM evaluations e
        JOIN mvs.evaluations m
          ON m.stage='bruteforce' AND m.replicate=0
         AND m.candidate_key=e.candidate_key
        WHERE e.status='PASS' AND e.historical_evo_ape_mean_m IS NOT NULL
        """
    ).fetchall()
    connection.close()
    mvs_values = np.array([row[0] for row in rows])
    full_values = np.array([row[1] for row in rows])
    rho, p_value = spearmanr(mvs_values, full_values)
    baseline_index = next(index for index, row in enumerate(rows) if row[2] == "5,29,40,52")

    fig, axis = plt.subplots(figsize=(9.6, 6.0))
    axis.scatter(mvs_values, full_values, s=12, alpha=0.25, color="#2471a3", edgecolors="none")
    axis.scatter(
        mvs_values[baseline_index],
        full_values[baseline_index],
        s=130,
        marker="*",
        color="#c0392b",
        label="known baseline",
        zorder=4,
    )
    axis.text(
        0.03,
        0.95,
        f"n = {len(rows):,}\nSpearman rho = {rho:.3f}\np = {p_value:.3f}",
        transform=axis.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
    )
    axis.set_xlabel("MVS all-frame SE(3) ATE RMSE (cm)")
    axis.set_ylabel("Full-sequence historical evo_ape ATE mean (cm)")
    axis.set_title("MVS local ATE had almost no ranking relationship with full-sequence ATE")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "stage2_mvs_full_relationship.png", dpi=220)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    selection_funnel()
    top_ranking()
    failure_distribution()
    mvs_full_relationship()
    print(f"Figures written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
