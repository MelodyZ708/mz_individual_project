#!/usr/bin/env python3
"""Summarise the independent Top-7 repeat and visualise Conv1 responses.

The script never launches COMO.  It reads the new repeat database, the
authoritative second-stage database, the original post-ReLU feature archives,
and the final r=0.70 Conv1 clusters.  All derived tables and plots are written
below the dedicated Top-7 result directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_CACHE = Path(tempfile.gettempdir()) / "mz_top7_repeat_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_PLAN = SCRIPT_DIR / "top7_candidate_plan.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "channel_selection_results/step_e_full_sequence_evaluation/"
    "top7_repeat_feature_cluster_analysis"
)
DEFAULT_REFERENCE_DB = (
    PROJECT_ROOT
    / "channel_selection_results/step_e_full_sequence_evaluation/"
    "second_round_baseline_plus2_rpe_safe/evaluations.sqlite3"
)
DEFAULT_FEATURE_MANIFEST = (
    PROJECT_ROOT
    / "channel_selection_results/step_0_data_preparation/"
    "features_post_relu/feature_manifest.csv"
)
DEFAULT_CLUSTERS = (
    PROJECT_ROOT
    / "channel_selection_results/step_b_correlation_clustering/"
    "threshold_r070/clusters/clusters_conv1.json"
)
SELECTED_FRAME_INDICES = (246, 250, 254)
BASELINE_KEY = "5,29,40,52"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Analyse the independent full-sequence repeat of the six "
            "second-stage winners plus the historical baseline."
        ),
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-db", type=Path, default=DEFAULT_REFERENCE_DB)
    parser.add_argument(
        "--repeat-db",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_DIR/evaluations.sqlite3.",
    )
    parser.add_argument(
        "--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST
    )
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument(
        "--skip-feature-plots",
        action="store_true",
        help="Generate only metric and cluster summaries.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def load_plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 7:
        raise ValueError("Top-7 plan must contain exactly seven candidates")
    expected_keys = set()
    for expected_rank, item in enumerate(candidates, start=1):
        channels = tuple(sorted(int(value) for value in item["channels"]))
        key = ",".join(str(value) for value in channels)
        if item.get("full_sequence_rank") != expected_rank:
            raise ValueError(f"Plan rank mismatch at row {expected_rank}")
        if item.get("candidate_key") != key or key in expected_keys:
            raise ValueError(f"Invalid or duplicate candidate key: {key}")
        item["channels"] = list(channels)
        expected_keys.add(key)
    if BASELINE_KEY not in expected_keys:
        raise ValueError(f"Historical baseline {BASELINE_KEY} is absent from plan")
    return candidates


def database_rows(path: Path) -> dict[str, sqlite3.Row]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    if integrity != "ok":
        connection.close()
        raise RuntimeError(f"SQLite integrity check failed for {path}: {integrity}")
    rows = connection.execute(
        "SELECT * FROM evaluations WHERE replicate=0 ORDER BY id"
    ).fetchall()
    connection.close()
    result: dict[str, sqlite3.Row] = {}
    for row in rows:
        key = str(row["candidate_key"])
        if key in result:
            raise ValueError(f"Duplicate replicate-0 candidate in {path}: {key}")
        result[key] = row
    return result


def cm(value: Any) -> float | None:
    return None if value is None else float(value) * 100.0


def finite_or_blank(value: float | None) -> float | str:
    return "" if value is None or not math.isfinite(value) else value


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_repeat_table(
    candidates: list[dict[str, Any]],
    reference: dict[str, sqlite3.Row],
    repeat: dict[str, sqlite3.Row],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in candidates:
        key = item["candidate_key"]
        if key not in reference:
            raise KeyError(f"Top-7 candidate absent from reference database: {key}")
        original = reference[key]
        if original["status"] != "PASS" or original["historical_evo_ape_mean_m"] is None:
            raise ValueError(f"Reference row is not a valid PASS: {key}")
        rerun = repeat.get(key)
        original_mean = cm(original["historical_evo_ape_mean_m"])
        repeat_mean = (
            cm(rerun["historical_evo_ape_mean_m"])
            if rerun is not None and rerun["status"] == "PASS"
            else None
        )
        delta = (
            repeat_mean - original_mean
            if repeat_mean is not None and original_mean is not None
            else None
        )
        two_run_mean = (
            (original_mean + repeat_mean) / 2.0
            if repeat_mean is not None and original_mean is not None
            else None
        )
        rows.append(
            {
                "original_full_rank": item["full_sequence_rank"],
                "label": item["label"],
                "candidate_key": key,
                "channels": f"[{key}]",
                "is_baseline": key == BASELINE_KEY,
                "original_status": original["status"],
                "repeat_status": rerun["status"] if rerun is not None else "NOT_RUN",
                "original_historical_ate_mean_cm": original_mean,
                "repeat_historical_ate_mean_cm": repeat_mean,
                "repeat_minus_original_cm": delta,
                "absolute_repeat_delta_cm": abs(delta) if delta is not None else None,
                "two_run_ate_mean_cm": two_run_mean,
                "original_historical_ate_rmse_cm": cm(
                    original["historical_evo_ape_rmse_m"]
                ),
                "repeat_historical_ate_rmse_cm": (
                    cm(rerun["historical_evo_ape_rmse_m"])
                    if rerun is not None and rerun["status"] == "PASS"
                    else None
                ),
                "repeat_allframe_se3_ate_rmse_cm": (
                    cm(rerun["se3_ate_rmse_m"])
                    if rerun is not None and rerun["status"] == "PASS"
                    else None
                ),
                "repeat_translation_rpe_max_cm": (
                    cm(rerun["translation_rpe_max_m"])
                    if rerun is not None and rerun["status"] == "PASS"
                    else None
                ),
                "repeat_rotation_rpe_max_deg": (
                    float(rerun["rotation_rpe_max_deg"])
                    if rerun is not None
                    and rerun["status"] == "PASS"
                    and rerun["rotation_rpe_max_deg"] is not None
                    else None
                ),
                "repeat_coverage": (
                    float(rerun["coverage_ratio"])
                    if rerun is not None and rerun["coverage_ratio"] is not None
                    else None
                ),
                "repeat_elapsed_seconds": (
                    float(rerun["elapsed_seconds"]) if rerun is not None else None
                ),
                "repeat_reason": rerun["reason"] if rerun is not None else "",
            }
        )

    repeat_passes = sorted(
        (row for row in rows if row["repeat_historical_ate_mean_cm"] is not None),
        key=lambda row: row["repeat_historical_ate_mean_cm"],
    )
    for rank, row in enumerate(repeat_passes, start=1):
        row["repeat_rank"] = rank
    two_run_passes = sorted(
        (row for row in rows if row["two_run_ate_mean_cm"] is not None),
        key=lambda row: row["two_run_ate_mean_cm"],
    )
    for rank, row in enumerate(two_run_passes, start=1):
        row["two_run_rank"] = rank
    for row in rows:
        row.setdefault("repeat_rank", None)
        row.setdefault("two_run_rank", None)
    return rows


def load_cluster_lookup(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if float(payload.get("primary_correlation_threshold", -1)) != 0.7:
        raise ValueError(f"Expected r=0.70 Conv1 clusters, got {path}")
    lookup: dict[int, dict[str, Any]] = {}
    for cluster in payload["clusters"]:
        for member in cluster["members"]:
            channel = int(member)
            if channel in lookup:
                raise ValueError(f"Channel {channel} appears in multiple clusters")
            lookup[channel] = cluster
    return lookup, payload


def build_cluster_tables(
    candidates: list[dict[str, Any]], lookup: dict[int, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    membership: list[dict[str, Any]] = []
    channel_combinations: dict[int, list[str]] = defaultdict(list)
    for item in candidates:
        cluster_ids: list[int] = []
        for position, channel in enumerate(item["channels"], start=1):
            if channel not in lookup:
                raise KeyError(
                    f"Top-7 channel {channel} is absent from eligible r=0.70 clusters"
                )
            cluster = lookup[channel]
            cluster_id = int(cluster["cluster_id"])
            cluster_ids.append(cluster_id)
            if channel == int(cluster["medoid"]):
                role = "medoid"
            elif cluster.get("second_representative") == channel:
                role = "second_representative"
            else:
                role = "member"
            metric = cluster.get("member_metrics", {}).get(str(channel), {})
            membership.append(
                {
                    "original_full_rank": item["full_sequence_rank"],
                    "candidate_key": item["candidate_key"],
                    "channels": f"[{item['candidate_key']}]",
                    "channel_position": position,
                    "channel": channel,
                    "cluster_id": cluster_id,
                    "cluster_size": int(cluster["size"]),
                    "cluster_members": ",".join(
                        str(value) for value in cluster["members"]
                    ),
                    "medoid": int(cluster["medoid"]),
                    "second_representative": (
                        cluster.get("second_representative")
                        if cluster.get("second_representative") is not None
                        else ""
                    ),
                    "channel_role": role,
                    "cross_light_ncc": metric.get("cross_light_ncc", ""),
                    "robust_gradient_energy": metric.get(
                        "robust_gradient_energy", ""
                    ),
                }
            )
            channel_combinations[channel].append(item["candidate_key"])
        if len(set(cluster_ids)) != 4:
            raise ValueError(
                f"Candidate {item['candidate_key']} contains repeated r=0.70 clusters"
            )

    channel_counts = Counter(row["channel"] for row in membership)
    frequency: list[dict[str, Any]] = []
    for channel, count in sorted(
        channel_counts.items(), key=lambda pair: (-pair[1], pair[0])
    ):
        cluster = lookup[channel]
        frequency.append(
            {
                "channel": channel,
                "selection_count": count,
                "selection_fraction": count / len(candidates),
                "cluster_id": int(cluster["cluster_id"]),
                "cluster_size": int(cluster["size"]),
                "channel_role": next(
                    row["channel_role"]
                    for row in membership
                    if row["channel"] == channel
                ),
                "combinations": ";".join(channel_combinations[channel]),
            }
        )

    patterns: list[dict[str, Any]] = []
    by_combo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in membership:
        by_combo[row["candidate_key"]].append(row)
    for item in candidates:
        rows = sorted(by_combo[item["candidate_key"]], key=lambda row: row["channel_position"])
        patterns.append(
            {
                "original_full_rank": item["full_sequence_rank"],
                "candidate_key": item["candidate_key"],
                "channels": f"[{item['candidate_key']}]",
                "cluster_ids": ",".join(str(row["cluster_id"]) for row in rows),
                "cluster_pattern": " + ".join(
                    f"ch{row['channel']}→C{row['cluster_id']}" for row in rows
                ),
                "singleton_cluster_channels": sum(
                    int(row["cluster_size"] == 1) for row in rows
                ),
                "medoid_channels": sum(row["channel_role"] == "medoid" for row in rows),
                "second_representatives": sum(
                    row["channel_role"] == "second_representative" for row in rows
                ),
            }
        )
    return membership, frequency, patterns


def load_selected_feature_frames(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    by_index = {int(row["frame_index"]): row for row in manifest}
    missing = [index for index in SELECTED_FRAME_INDICES if index not in by_index]
    if missing:
        raise KeyError(f"Feature manifest lacks selected frames: {missing}")
    rows = [by_index[index] for index in SELECTED_FRAME_INDICES]
    expected_phases = ("before", "peak", "after")
    for row, phase in zip(rows, expected_phases):
        if row["selection_source"] != "turn_on" or row["phase"] != phase:
            raise ValueError(
                f"Frame {row['frame_index']} is not expected turn_on/{phase}"
            )
        for field in ("feature_file", "clean_path", "lightswitch_path"):
            require_file(Path(row[field]), f"Feature-frame {field}")
    return rows


def percentile_limit(array: np.ndarray, percentile: float = 99.5) -> float:
    finite = np.asarray(array, dtype=np.float32)
    value = float(np.nanpercentile(finite, percentile))
    return value if math.isfinite(value) and value > 0 else 1.0


def save_input_frames(frames: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(len(frames), 2, figsize=(10.5, 9.2))
    for row_index, row in enumerate(frames):
        for column, field in enumerate(("clean_path", "lightswitch_path")):
            bgr = cv2.imread(row[field], cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not read input image: {row[field]}")
            axes[row_index, column].imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            axes[row_index, column].axis("off")
            axes[row_index, column].set_title(
                f"frame {row['frame_index']} · {row['phase']} · "
                f"{'clean' if column == 0 else 'lightswitch'}",
                fontsize=9.5,
            )
    figure.suptitle("Selected matched frames around the principal turn-on event")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def load_combo_features(
    frames: list[dict[str, Any]], channels: list[int]
) -> list[dict[str, np.ndarray]]:
    result: list[dict[str, np.ndarray]] = []
    for row in frames:
        with np.load(row["feature_file"], allow_pickle=False) as archive:
            clean = np.asarray(archive["conv1_clean"][channels], dtype=np.float32)
            light = np.asarray(archive["conv1_light"][channels], dtype=np.float32)
        result.append({"clean": clean, "light": light, "diff": np.abs(light - clean)})
    return result


def save_lightswitch_overview(
    item: dict[str, Any],
    frames: list[dict[str, Any]],
    features: list[dict[str, np.ndarray]],
    cluster_lookup: dict[int, dict[str, Any]],
    output: Path,
) -> None:
    channels = item["channels"]
    limits = [
        percentile_limit(np.stack([sample["light"][column] for sample in features]))
        for column in range(4)
    ]
    figure, axes = plt.subplots(3, 4, figsize=(13.5, 9.1))
    for row_index, (frame, sample) in enumerate(zip(frames, features)):
        for column, channel in enumerate(channels):
            axis = axes[row_index, column]
            axis.imshow(
                sample["light"][column], cmap="viridis", vmin=0, vmax=limits[column]
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                cluster_id = cluster_lookup[channel]["cluster_id"]
                axis.set_title(f"channel {channel} · cluster C{cluster_id}")
            if column == 0:
                axis.set_ylabel(
                    f"frame {frame['frame_index']}\n{frame['phase']}", fontsize=10
                )
    figure.suptitle(
        f"[{item['candidate_key']}] lightswitch Conv1 post-ReLU maps\n"
        "Each channel uses one fixed 0–99.5th percentile scale across the three frames",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_clean_light_details(
    item: dict[str, Any],
    frame: dict[str, Any],
    sample: dict[str, np.ndarray],
    cluster_lookup: dict[int, dict[str, Any]],
    output: Path,
) -> None:
    channels = item["channels"]
    figure, axes = plt.subplots(4, 3, figsize=(10.8, 13.2))
    for row_index, channel in enumerate(channels):
        common_limit = percentile_limit(
            np.stack([sample["clean"][row_index], sample["light"][row_index]])
        )
        diff_limit = percentile_limit(sample["diff"][row_index])
        for column, (kind, cmap, limit) in enumerate(
            (
                ("clean", "viridis", common_limit),
                ("light", "viridis", common_limit),
                ("diff", "magma", diff_limit),
            )
        ):
            axis = axes[row_index, column]
            axis.imshow(sample[kind][row_index], cmap=cmap, vmin=0, vmax=limit)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(
                    {"clean": "Clean", "light": "Lightswitch", "diff": "|Light − clean|"}[kind]
                )
            if column == 0:
                cluster_id = cluster_lookup[channel]["cluster_id"]
                axis.set_ylabel(f"channel {channel}\ncluster C{cluster_id}")
    figure.suptitle(
        f"[{item['candidate_key']}] · frame {frame['frame_index']} · {frame['phase']}\n"
        "Conv1 post-ReLU; clean/light share a per-channel scale",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_feature_plots(
    candidates: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    cluster_lookup: dict[int, dict[str, Any]],
    output_dir: Path,
) -> None:
    plot_root = output_dir / "feature_maps"
    save_input_frames(frames, plot_root / "selected_input_frames.png")
    for item in candidates:
        features = load_combo_features(frames, item["channels"])
        slug = f"rank_{item['full_sequence_rank']:02d}_ch_{item['candidate_key'].replace(',', '_')}"
        save_lightswitch_overview(
            item,
            frames,
            features,
            cluster_lookup,
            plot_root / "lightswitch_overviews" / f"{slug}.png",
        )
        for frame, sample in zip(frames, features):
            save_clean_light_details(
                item,
                frame,
                sample,
                cluster_lookup,
                plot_root / "clean_light_difference" / slug
                / f"frame_{int(frame['frame_index']):03d}_{frame['phase']}.png",
            )


def save_repeat_plot(rows: list[dict[str, Any]], output: Path) -> None:
    valid = [row for row in rows if row["repeat_historical_ate_mean_cm"] is not None]
    if not valid:
        return
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    positions = np.arange(len(valid))
    original = np.array([row["original_historical_ate_mean_cm"] for row in valid])
    repeated = np.array([row["repeat_historical_ate_mean_cm"] for row in valid])
    for index, row in enumerate(valid):
        color = "#c0392b" if row["is_baseline"] else "#2c7fb8"
        axis.plot([index, index], [original[index], repeated[index]], color="#9aa7ad", lw=2)
        axis.scatter(index, original[index], marker="o", s=65, color=color, zorder=3)
        axis.scatter(index, repeated[index], marker="D", s=60, color=color, zorder=3)
    axis.scatter([], [], marker="o", color="#555555", label="original second-stage run")
    axis.scatter([], [], marker="D", color="#555555", label="independent repeat")
    axis.set_xticks(positions)
    axis.set_xticklabels([f"[{row['candidate_key']}]" for row in valid], rotation=28, ha="right")
    axis.set_ylabel("Historical keyframe evo_ape ATE mean (cm)")
    axis.set_title("Top-7 full-sequence repeatability")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_frequency_plot(rows: list[dict[str, Any]], output: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row["selection_count"], -row["channel"]))
    figure, axis = plt.subplots(figsize=(9.2, max(4.8, 0.35 * len(ordered))))
    y = np.arange(len(ordered))
    values = [row["selection_count"] for row in ordered]
    axis.barh(y, values, color="#4c78a8")
    axis.set_yticks(y)
    axis.set_yticklabels(
        [f"ch {row['channel']}  (C{row['cluster_id']})" for row in ordered]
    )
    axis.set_xlabel("Number of Top-7 combinations containing the channel")
    axis.set_title("Channel recurrence across the six winners and baseline")
    axis.set_xticks(range(0, max(values) + 1))
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_summary(
    output: Path,
    result_rows: list[dict[str, Any]],
    frequency: list[dict[str, Any]],
    patterns: list[dict[str, Any]],
    feature_frames: list[dict[str, Any]],
) -> None:
    passed = [row for row in result_rows if row["repeat_status"] == "PASS"]
    completed = [row for row in result_rows if row["repeat_status"] != "NOT_RUN"]
    lines = [
        "# Top-7独立重复、Feature Map与Cluster归属汇总",
        "",
        "## 协议",
        "",
        "- 对象：第二阶段full-sequence主指标优于baseline的6组，加baseline `[5,29,40,52]`，共7组。",
        "- 重复：每组新运行1次完整 `fr1/desk_lightswitch`，不复用原轨迹。",
        "- 主指标：与历史脚本一致的keyframe `evo_ape --align --correct_scale` ATE mean。",
        "- Feature map：ResNet-18 Conv1 post-ReLU native-resolution结果。",
        "- 帧：246/250/254，分别对应主要turn-on事件的before/peak/after。",
        "- 聚类：最终搜索采用的全局Conv1 correlation clustering，`r=0.70`稳定化簇。",
        "",
        "## 重复运行结果",
        "",
        f"当前已完成 {len(completed)}/7，其中PASS {len(passed)}。",
        "",
    ]
    metric_table: list[list[str]] = []
    for row in result_rows:
        repeat_value = row["repeat_historical_ate_mean_cm"]
        delta = row["repeat_minus_original_cm"]
        mean_value = row["two_run_ate_mean_cm"]
        metric_table.append(
            [
                str(row["original_full_rank"]),
                f"`[{row['candidate_key']}]`",
                f"{row['original_historical_ate_mean_cm']:.4f}",
                f"{repeat_value:.4f}" if repeat_value is not None else row["repeat_status"],
                f"{delta:+.4f}" if delta is not None else "—",
                f"{mean_value:.4f}" if mean_value is not None else "—",
                str(row["two_run_rank"] or "—"),
            ]
        )
    lines.extend(
        [
            markdown_table(
                ["原rank", "channels", "原ATE mean/cm", "重复ATE mean/cm", "差值/cm", "两次均值/cm", "两次均值rank"],
                metric_table,
            ),
            "",
        ]
    )
    if len(passed) == 7:
        original_ranks = np.array([row["original_full_rank"] for row in passed], dtype=float)
        repeat_ranks = np.array([row["repeat_rank"] for row in passed], dtype=float)
        rho = float(np.corrcoef(original_ranks, repeat_ranks)[0, 1])
        deltas = sorted(passed, key=lambda row: row["absolute_repeat_delta_cm"])
        two_run = sorted(passed, key=lambda row: row["two_run_ate_mean_cm"])
        baseline = next(row for row in passed if row["is_baseline"])
        winner = two_run[0]
        lines.extend(
            [
                "## 重复性初步解读",
                "",
                f"- 原排名与本次重复排名的Spearman相关（7组、无ties）为 **{rho:.3f}**。",
                f"- 两次ATE均值最低的是 `[{winner['candidate_key']}]`：**{winner['two_run_ate_mean_cm']:.4f} cm**。",
                f"- baseline两次ATE均值为 **{baseline['two_run_ate_mean_cm']:.4f} cm**。",
                f"- 最小单次差值：`[{deltas[0]['candidate_key']}]`，|Δ|={deltas[0]['absolute_repeat_delta_cm']:.4f} cm。",
                f"- 最大单次差值：`[{deltas[-1]['candidate_key']}]`，|Δ|={deltas[-1]['absolute_repeat_delta_cm']:.4f} cm。",
                "- 每组目前只有两次观测；两次均值和差值用于检查稳定性，不应当作充分的方差估计。",
                "",
            ]
        )
    lines.extend(["## Cluster归属", ""])
    pattern_rows = [
        [
            str(row["original_full_rank"]),
            f"`[{row['candidate_key']}]`",
            row["cluster_pattern"],
            str(row["singleton_cluster_channels"]),
            str(row["medoid_channels"]),
        ]
        for row in patterns
    ]
    lines.extend(
        [
            markdown_table(
                ["原rank", "channels", "r=0.70 cluster pattern", "singleton数", "medoid数"],
                pattern_rows,
            ),
            "",
            "所有组合都来自4个不同的r=0.70簇，这是第一阶段合法组合约束的直接结果；因此这里能分析的是哪些簇/代表反复出现，而不是验证簇内共选。",
            "",
            "### 高频channel/cluster",
            "",
        ]
    )
    frequency_rows = [
        [
            str(row["channel"]),
            f"C{row['cluster_id']}",
            str(row["cluster_size"]),
            row["channel_role"],
            f"{row['selection_count']}/7",
        ]
        for row in frequency
    ]
    lines.extend(
        [
            markdown_table(
                ["channel", "cluster", "cluster size", "代表角色", "出现次数"],
                frequency_rows,
            ),
            "",
            "注意：Top-7并非7个独立随机样本，且包含baseline；高频只能作为后续功能解释线索，不能直接证明某channel具有因果优势。",
            "",
            "## Feature map输出",
            "",
            "选择帧："
            + ", ".join(
                f"frame {row['frame_index']} ({row['phase']})" for row in feature_frames
            )
            + "。",
            "",
            "- `feature_maps/lightswitch_overviews/`：每组一张3×4总览，纵向比较before/peak/after。",
            "- `feature_maps/clean_light_difference/`：每组每帧一张详细图，展示clean、lightswitch和绝对差。",
            "- clean/light使用同一channel尺度；不同channel仍各自缩放，因此颜色强度不能跨channel直接比较。",
            "",
            "## 可审计文件",
            "",
            "- `evaluations.sqlite3`：本次独立重复的权威记录。",
            "- `repeat_comparison.csv`：原运行、本次重复、差值和两次均值。",
            "- `cluster_membership.csv`：28个channel选择的逐项cluster归属。",
            "- `channel_frequency.csv`：channel在Top-7中的复现频率。",
            "- `combination_cluster_patterns.csv`：每组的cluster pattern。",
            "- `selected_feature_frames.csv`：可视化帧与原始特征文件路径。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.repeat_db = (
        args.repeat_db.resolve()
        if args.repeat_db is not None
        else args.output_dir / "evaluations.sqlite3"
    )
    plan_path = require_file(args.plan, "Top-7 candidate plan")
    reference_path = require_file(args.reference_db, "Second-stage reference DB")
    repeat_path = require_file(args.repeat_db, "Top-7 repeat DB")
    manifest_path = require_file(args.feature_manifest, "Feature manifest")
    cluster_path = require_file(args.clusters, "r=0.70 Conv1 clusters")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_plan(plan_path)
    reference = database_rows(reference_path)
    repeated = database_rows(repeat_path)
    unknown = set(repeated).difference(item["candidate_key"] for item in candidates)
    if unknown:
        raise ValueError(f"Repeat DB contains candidates outside frozen Top-7: {sorted(unknown)}")
    result_rows = build_repeat_table(candidates, reference, repeated)
    cluster_lookup, cluster_payload = load_cluster_lookup(cluster_path)
    membership, frequency, patterns = build_cluster_tables(candidates, cluster_lookup)
    feature_frames = load_selected_feature_frames(manifest_path)

    repeat_fields = list(result_rows[0].keys())
    serialised_results = [
        {key: finite_or_blank(value) if isinstance(value, float) else value for key, value in row.items()}
        for row in result_rows
    ]
    write_csv(args.output_dir / "repeat_comparison.csv", serialised_results, repeat_fields)
    write_csv(
        args.output_dir / "cluster_membership.csv",
        membership,
        list(membership[0].keys()),
    )
    write_csv(
        args.output_dir / "channel_frequency.csv",
        frequency,
        list(frequency[0].keys()),
    )
    write_csv(
        args.output_dir / "combination_cluster_patterns.csv",
        patterns,
        list(patterns[0].keys()),
    )
    write_csv(
        args.output_dir / "selected_feature_frames.csv",
        feature_frames,
        list(feature_frames[0].keys()),
    )
    protocol = {
        "protocol": "full_sequence_top7_independent_repeat_feature_cluster_analysis_v1",
        "candidate_plan": str(plan_path),
        "reference_database": str(reference_path),
        "repeat_database": str(repeat_path),
        "feature_manifest": str(manifest_path),
        "selected_frame_indices": list(SELECTED_FRAME_INDICES),
        "feature_definition": "ResNet-18 Conv1 post-ReLU, native resolution",
        "cluster_file": str(cluster_path),
        "cluster_threshold": cluster_payload["primary_correlation_threshold"],
        "primary_metric": "keyframe evo_ape --align --correct_scale ATE mean",
    }
    (args.output_dir / "analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )

    save_repeat_plot(result_rows, args.output_dir / "plots/repeat_ate_comparison.png")
    save_frequency_plot(frequency, args.output_dir / "plots/channel_frequency.png")
    if not args.skip_feature_plots:
        make_feature_plots(candidates, feature_frames, cluster_lookup, args.output_dir)
    write_summary(
        args.output_dir / "summary.md",
        result_rows,
        frequency,
        patterns,
        feature_frames,
    )

    status_counts = Counter(row["repeat_status"] for row in result_rows)
    print("[ANALYSIS] Repeat statuses: " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())))
    print(f"[ANALYSIS] Tables, plots and summary written to: {args.output_dir}")


if __name__ == "__main__":
    main()
