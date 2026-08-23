#!/usr/bin/env python3
"""Generate compact figures for the Top-7 validation interpretation report."""

from __future__ import annotations

import csv
import itertools
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

_CACHE = Path(tempfile.gettempdir()) / "mz_top7_report_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parents[2]
RESULT_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_e_full_sequence_evaluation/"
    "top7_repeat_feature_cluster_analysis"
)
SECOND_STAGE_DB = (
    PROJECT_ROOT
    / "channel_selection_results/step_e_full_sequence_evaluation/"
    "second_round_baseline_plus2_rpe_safe/evaluations.sqlite3"
)
CLUSTER_FILE = (
    PROJECT_ROOT
    / "channel_selection_results/step_b_correlation_clustering/"
    "threshold_r070/clusters/clusters_conv1.json"
)


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULT_DIR / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_figure(results: list[dict[str, str]]) -> None:
    labels = [f"#{row['original_full_rank']} [{row['candidate_key']}]" for row in results]
    y = np.arange(len(results))
    metrics = (
        ("original_historical_ate_mean_cm", "Historical keyframe ATE mean (cm)"),
        ("repeat_allframe_se3_ate_rmse_cm", "All-frame SE(3) ATE RMSE (cm)"),
        ("repeat_translation_rpe_max_cm", "Translation RPE max (cm)"),
        ("repeat_rotation_rpe_max_deg", "Rotation RPE max (deg)"),
    )
    colors = []
    for row in results:
        rank = int(row["original_full_rank"])
        colors.append(
            "#c0392b"
            if rank == 7
            else "#1b9e77"
            if rank == 2
            else "#d95f02"
            if rank == 4
            else "#2c7fb8"
            if rank == 1
            else "#8c96a0"
        )
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), sharey=True)
    for axis, (field, title) in zip(axes.flat, metrics):
        values = np.array([float(row[field]) for row in results])
        baseline = values[-1]
        axis.axvline(baseline, color="#c0392b", linestyle="--", lw=1.5, alpha=0.8)
        axis.scatter(values, y, c=colors, s=70, zorder=3)
        for value, row_y in zip(values, y):
            axis.text(value, row_y - 0.18, f"{value:.2f}", fontsize=8, ha="center")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        axis.invert_yaxis()
        axis.set_yticks(y)
        axis.set_yticklabels(labels)
        axis.set_xlabel("lower is better")
    figure.suptitle(
        "Top-7 accuracy and local-stability diagnostics\n"
        "Dashed red line: historical baseline [5,29,40,52]",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(REPORT_DIR / "top7_metric_tradeoffs.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def frequency_figure(
    results: list[dict[str, str]], membership: list[dict[str, str]]
) -> None:
    winners = {row["candidate_key"] for row in results if int(row["original_full_rank"]) <= 6}
    baseline = next(row["candidate_key"] for row in results if row["is_baseline"] == "True")
    winner_rows = [row for row in membership if row["candidate_key"] in winners]
    baseline_rows = [row for row in membership if row["candidate_key"] == baseline]
    cluster_counts = Counter(int(row["cluster_id"]) for row in winner_rows)
    channel_counts = Counter(int(row["channel"]) for row in winner_rows)
    baseline_clusters = {int(row["cluster_id"]) for row in baseline_rows}
    baseline_channels = {int(row["channel"]) for row in baseline_rows}

    clusters = sorted(cluster_counts, key=lambda value: (cluster_counts[value], -value))
    channels = sorted(channel_counts, key=lambda value: (channel_counts[value], -value))
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 6.8))
    for axis, items, counts, baseline_items, prefix, title in (
        (axes[0], clusters, cluster_counts, baseline_clusters, "C", "Cluster frequency among six winners"),
        (axes[1], channels, channel_counts, baseline_channels, "ch", "Channel frequency among six winners"),
    ):
        positions = np.arange(len(items))
        values = [counts[item] for item in items]
        axis.barh(positions, values, color="#4c78a8")
        axis.set_yticks(positions)
        axis.set_yticklabels([f"{prefix}{item}" for item in items])
        axis.set_xticks(range(0, max(values) + 1))
        axis.set_xlabel("number of winner combinations (out of 6)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        for pos, item, value in zip(positions, items, values):
            axis.text(value + 0.06, pos, str(value), va="center", fontsize=9)
            if item in baseline_items:
                axis.text(0.05, pos, "B", va="center", ha="left", color="#c0392b", fontweight="bold")
    figure.suptitle(
        "Recurring components in the six combinations that beat baseline\n"
        "Red B marks a component also present in baseline",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(REPORT_DIR / "winner_channel_cluster_frequency.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def pair_figure(results: list[dict[str, str]], membership: list[dict[str, str]]) -> None:
    by_combo: dict[str, list[int]] = defaultdict(list)
    for row in membership:
        by_combo[row["candidate_key"]].append(int(row["cluster_id"]))
    winner_keys = [
        row["candidate_key"] for row in results if int(row["original_full_rank"]) <= 6
    ]
    frequency = Counter(
        cluster for key in winner_keys for cluster in set(by_combo[key])
    )
    selected = sorted(cluster for cluster, count in frequency.items() if count >= 2)
    pair_counts: Counter[tuple[int, int]] = Counter()
    for key in winner_keys:
        pair_counts.update(itertools.combinations(sorted(set(by_combo[key])), 2))
    matrix = np.zeros((len(selected), len(selected)), dtype=int)
    for row_index, first in enumerate(selected):
        matrix[row_index, row_index] = frequency[first]
        for column_index, second in enumerate(selected):
            if row_index != column_index:
                matrix[row_index, column_index] = pair_counts[tuple(sorted((first, second)))]
    figure, axis = plt.subplots(figsize=(7.8, 6.8))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(3, matrix.max()))
    for row_index in range(len(selected)):
        for column_index in range(len(selected)):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value >= 3 else "black",
            )
    axis.set_xticks(range(len(selected)))
    axis.set_xticklabels([f"C{value}" for value in selected])
    axis.set_yticks(range(len(selected)))
    axis.set_yticklabels([f"C{value}" for value in selected])
    axis.set_title(
        "Exploratory cluster co-occurrence among six winners\n"
        "Diagonal = cluster frequency; off-diagonal = pair frequency"
    )
    figure.colorbar(image, ax=axis, shrink=0.82, label="count")
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "winner_cluster_pair_cooccurrence.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def enrichment_figure(
    results: list[dict[str, str]], membership: list[dict[str, str]]
) -> None:
    cluster_payload = json.loads(CLUSTER_FILE.read_text(encoding="utf-8"))
    channel_to_cluster = {
        int(member): int(cluster["cluster_id"])
        for cluster in cluster_payload["clusters"]
        for member in cluster["members"]
    }
    connection = sqlite3.connect(f"file:{SECOND_STAGE_DB}?mode=ro", uri=True)
    pass_keys = [
        row[0]
        for row in connection.execute(
            "SELECT candidate_key FROM evaluations "
            "WHERE status='PASS' AND replicate=0"
        )
    ]
    connection.close()
    winner_keys = {
        row["candidate_key"]
        for row in results
        if int(row["original_full_rank"]) <= 6
    }
    all_channels = Counter()
    all_clusters = Counter()
    winner_channels = Counter()
    winner_clusters = Counter()
    for key in pass_keys:
        channels = {int(value) for value in key.split(",")}
        clusters = {channel_to_cluster[channel] for channel in channels}
        all_channels.update(channels)
        all_clusters.update(clusters)
        if key in winner_keys:
            winner_channels.update(channels)
            winner_clusters.update(clusters)
    if len(winner_keys) != 6 or len(pass_keys) != 2835:
        raise ValueError("Unexpected winner/background population for enrichment")

    recurrent_clusters = [item for item, count in winner_clusters.items() if count >= 2]
    recurrent_channels = [item for item, count in winner_channels.items() if count >= 2]
    recurrent_clusters.sort(
        key=lambda item: winner_clusters[item] / (all_clusters[item] / len(pass_keys))
    )
    recurrent_channels.sort(
        key=lambda item: winner_channels[item] / (all_channels[item] / len(pass_keys))
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))
    for axis, items, win_counts, bg_counts, prefix, title in (
        (
            axes[0], recurrent_clusters, winner_clusters, all_clusters, "C",
            "Recurrent clusters: winners vs all 2,835 full PASS",
        ),
        (
            axes[1], recurrent_channels, winner_channels, all_channels, "ch",
            "Recurrent channels: winners vs all 2,835 full PASS",
        ),
    ):
        y = np.arange(len(items))
        winner_rate = np.array([win_counts[item] / 6 for item in items])
        background_rate = np.array([bg_counts[item] / len(pass_keys) for item in items])
        axis.scatter(background_rate * 100, y, marker="o", s=70, color="#8c96a0", label="all PASS")
        axis.scatter(winner_rate * 100, y, marker="D", s=70, color="#2c7fb8", label="six winners")
        for row_y, left, right in zip(y, background_rate, winner_rate):
            axis.plot([left * 100, right * 100], [row_y, row_y], color="#b7c1c6", zorder=0)
            axis.text(
                right * 100 + 1.2,
                row_y,
                f"{right / left:.2f}×",
                va="center",
                fontsize=8.5,
            )
        axis.set_yticks(y)
        axis.set_yticklabels([f"{prefix}{item}" for item in items])
        axis.set_xlabel("combination prevalence (%)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    axes[1].legend(loc="lower right")
    figure.suptitle(
        "Descriptive enrichment corrects for unequal component availability\n"
        "Ratios are exploratory because only six winners are observed",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(REPORT_DIR / "winner_component_enrichment.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    results = read_csv("repeat_comparison.csv")
    membership = read_csv("cluster_membership.csv")
    if len(results) != 7 or len(membership) != 28:
        raise ValueError("Expected seven combinations and 28 membership rows")
    metric_figure(results)
    frequency_figure(results, membership)
    pair_figure(results, membership)
    enrichment_figure(results, membership)
    print(f"Report figures written to {REPORT_DIR}")


if __name__ == "__main__":
    main()
