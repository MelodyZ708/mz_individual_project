#!/usr/bin/env python3
"""Visualise and quantify complementarity in two Conv1 channel combinations.

This is a read-only analysis of the 30 paired post-ReLU feature archives.  It
does not launch COMO or modify any evaluation database.  The reported metrics
are descriptive evidence of diversity/complementarity; they are not a causal
substitute for channel-masking ablations in the tracker.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

_CACHE = Path(tempfile.gettempdir()) / "mz_channel_complementarity_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "channel_selection_results/step_0_data_preparation/"
    "features_post_relu/feature_manifest.csv"
)
DEFAULT_CLUSTERS = (
    PROJECT_ROOT
    / "channel_selection_results/step_b_correlation_clustering/"
    "threshold_r070/clusters/clusters_conv1.json"
)
DEFAULT_DIAGNOSTICS = (
    PROJECT_ROOT
    / "channel_selection_results/step_b_correlation_clustering/"
    "threshold_r070/diagnostics/channel_numeric_diagnostics_conv1.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "channel_selection_results/step_g_channel_complementarity/"
    "two_combination_analysis"
)

COMBINATIONS = {
    "combo_A": (15, 17, 52, 59),
    "combo_B": (1, 5, 24, 29),
}
SELECTED_FRAMES = (246, 250, 254)
FRAME_PHASES = {246: "before", 250: "peak", 254: "after"}
SUPPORT_QUANTILE = 0.85
EPS = 1e-12
CHANNEL_COLORS = np.asarray(
    [
        [0.121, 0.466, 0.705],
        [1.000, 0.498, 0.055],
        [0.172, 0.627, 0.172],
        [0.839, 0.153, 0.157],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Analyse feature-level complementarity in two Conv1 combinations.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path)
    if len(rows) != 30:
        raise ValueError(f"Expected 30 paired timestamps, found {len(rows)}")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        frame_index = int(row["frame_index"])
        feature_file = require_file(Path(row["feature_file"]), "Feature archive")
        light_path = require_file(Path(row["lightswitch_path"]), "Lightswitch RGB")
        if frame_index in seen:
            raise ValueError(f"Duplicate frame index in manifest: {frame_index}")
        seen.add(frame_index)
        result.append(
            {
                **row,
                "frame_index": frame_index,
                "feature_file": feature_file,
                "lightswitch_path": light_path,
            }
        )
    missing = set(SELECTED_FRAMES).difference(seen)
    if missing:
        raise ValueError(f"Selected visualisation frames are absent: {sorted(missing)}")
    return result


def load_cluster_lookup(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not math.isclose(float(payload["primary_correlation_threshold"]), 0.70):
        raise ValueError("Expected the final r=0.70 Conv1 clustering")
    lookup: dict[int, dict[str, Any]] = {}
    for cluster in payload["clusters"]:
        for channel in cluster["members"]:
            lookup[int(channel)] = cluster
    for channel in {value for combo in COMBINATIONS.values() for value in combo}:
        if channel not in lookup:
            raise KeyError(f"Channel {channel} is absent from the r=0.70 clusters")
    return lookup


def load_diagnostics(path: Path) -> dict[int, dict[str, str]]:
    rows = read_csv(path)
    return {int(row["channel"]): row for row in rows}


def zscore_flat(array: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(array, dtype=np.float64).reshape(-1)
    std = float(vector.std())
    if not math.isfinite(std) or std <= EPS:
        return None
    return (vector - float(vector.mean())) / std


def gradient_data(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    gy, gx = np.gradient(np.asarray(array, dtype=np.float32))
    magnitude = np.hypot(gx, gy)
    energy = float(np.mean(np.square(magnitude, dtype=np.float64)))
    threshold = float(np.quantile(magnitude, SUPPORT_QUANTILE))
    support = (magnitude >= threshold) & (magnitude > EPS)
    return magnitude, support, energy


def effective_rank(correlation: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(correlation.astype(np.float64))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= EPS:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -float(np.sum(probabilities * np.log(probabilities + EPS)))
    return float(np.exp(entropy))


def safe_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def analyse_combo(
    label: str,
    channels: tuple[int, ...],
    manifest: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[int, Any]]:
    pair_accumulator: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    channel_accumulator: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    union_coverages: list[float] = []
    overlap_coverages: list[float] = []
    ranks: list[float] = []
    visual_data: dict[int, Any] = {}

    for row in manifest:
        with np.load(row["feature_file"], allow_pickle=False) as archive:
            condition_maps = {
                "clean": np.asarray(archive["conv1_clean"][list(channels)], dtype=np.float32),
                "light": np.asarray(archive["conv1_light"][list(channels)], dtype=np.float32),
            }

        if row["frame_index"] in SELECTED_FRAMES:
            bgr = cv2.imread(str(row["lightswitch_path"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Could not read {row['lightswitch_path']}")
            visual_data[row["frame_index"]] = {
                "rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                "maps": condition_maps["light"].copy(),
            }

        for condition, maps in condition_maps.items():
            zscores = [zscore_flat(feature_map) for feature_map in maps]
            if all(vector is not None for vector in zscores):
                matrix = np.stack(zscores)
                signed_corr = (matrix @ matrix.T) / matrix.shape[1]
                signed_corr = np.clip(signed_corr, -1.0, 1.0)
                ranks.append(effective_rank(signed_corr))
            supports: list[np.ndarray] = []
            for position, channel in enumerate(channels):
                _magnitude, support, energy = gradient_data(maps[position])
                supports.append(support)
                channel_accumulator[channel][f"gradient_energy_{condition}"].append(energy)
                channel_accumulator[channel][f"active_ratio_{condition}"].append(
                    float(np.count_nonzero(maps[position] > 0)) / maps[position].size
                )
            support_count = np.sum(np.stack(supports), axis=0)
            union = support_count > 0
            union_count = int(np.count_nonzero(union))
            union_coverages.append(union_count / union.size)
            overlap_coverages.append(
                float(np.count_nonzero(support_count >= 2)) / max(union_count, 1)
            )
            for position, channel in enumerate(channels):
                own = supports[position]
                own_count = int(np.count_nonzero(own))
                exclusive = own & (support_count == 1)
                channel_accumulator[channel]["exclusive_support_fraction"].append(
                    float(np.count_nonzero(exclusive)) / max(own_count, 1)
                )
            for left in range(4):
                for right in range(left + 1, 4):
                    key = (channels[left], channels[right])
                    if zscores[left] is not None and zscores[right] is not None:
                        corr = abs(
                            float(np.dot(zscores[left], zscores[right]) / zscores[left].size)
                        )
                        pair_accumulator[key][f"abs_corr_{condition}"].append(corr)
                    intersection = int(np.count_nonzero(supports[left] & supports[right]))
                    pair_union = int(np.count_nonzero(supports[left] | supports[right]))
                    pair_accumulator[key][f"gradient_jaccard_{condition}"].append(
                        intersection / max(pair_union, 1)
                    )

        for position, channel in enumerate(channels):
            clean_z = zscore_flat(condition_maps["clean"][position])
            light_z = zscore_flat(condition_maps["light"][position])
            if clean_z is not None and light_z is not None:
                channel_accumulator[channel]["paired_structural_ncc"].append(
                    float(np.dot(clean_z, light_z) / clean_z.size)
                )
            _, _, clean_energy = gradient_data(condition_maps["clean"][position])
            _, _, light_energy = gradient_data(condition_maps["light"][position])
            channel_accumulator[channel]["gradient_retention"].append(
                light_energy / max(clean_energy, EPS)
            )
            if row["selection_source"] in {"turn_on", "turn_off"}:
                channel_accumulator[channel]["gradient_retention_event"].append(
                    light_energy / max(clean_energy, EPS)
                )

    pair_rows: list[dict[str, Any]] = []
    for (left, right), metrics in sorted(pair_accumulator.items()):
        corr_clean = float(np.median(metrics["abs_corr_clean"]))
        corr_light = float(np.median(metrics["abs_corr_light"]))
        jaccard_clean = float(np.median(metrics["gradient_jaccard_clean"]))
        jaccard_light = float(np.median(metrics["gradient_jaccard_light"]))
        pair_rows.append(
            {
                "combination": label,
                "channels": f"[{','.join(map(str, channels))}]",
                "channel_i": left,
                "channel_j": right,
                "median_abs_corr_clean": corr_clean,
                "median_abs_corr_light": corr_light,
                "robust_abs_corr_min": min(corr_clean, corr_light),
                "median_gradient_jaccard_clean": jaccard_clean,
                "median_gradient_jaccard_light": jaccard_light,
                "mean_gradient_jaccard": (jaccard_clean + jaccard_light) / 2.0,
            }
        )

    channel_rows: list[dict[str, Any]] = []
    for channel in channels:
        metrics = channel_accumulator[channel]
        channel_rows.append(
            {
                "combination": label,
                "channels": f"[{','.join(map(str, channels))}]",
                "channel": channel,
                "paired_structural_ncc_median": float(
                    np.median(metrics["paired_structural_ncc"])
                ),
                "gradient_energy_clean_median": float(
                    np.median(metrics["gradient_energy_clean"])
                ),
                "gradient_energy_light_median": float(
                    np.median(metrics["gradient_energy_light"])
                ),
                "gradient_retention_median": float(
                    np.median(metrics["gradient_retention"])
                ),
                "event_gradient_retention_median": float(
                    np.median(metrics["gradient_retention_event"])
                ),
                "active_ratio_clean_median": float(
                    np.median(metrics["active_ratio_clean"])
                ),
                "active_ratio_light_median": float(
                    np.median(metrics["active_ratio_light"])
                ),
                "exclusive_gradient_support_fraction_mean": safe_mean(
                    metrics["exclusive_support_fraction"]
                ),
            }
        )

    robust_corrs = [row["robust_abs_corr_min"] for row in pair_rows]
    gradient_jaccards = [row["mean_gradient_jaccard"] for row in pair_rows]
    combo_row = {
        "combination": label,
        "channels": f"[{','.join(map(str, channels))}]",
        "mean_pair_robust_abs_corr": float(np.mean(robust_corrs)),
        "max_pair_robust_abs_corr": float(np.max(robust_corrs)),
        "mean_pair_gradient_jaccard": float(np.mean(gradient_jaccards)),
        "mean_gradient_union_image_fraction": float(np.mean(union_coverages)),
        "mean_overlap_fraction_within_union": float(np.mean(overlap_coverages)),
        "mean_effective_rank_out_of_4": float(np.mean(ranks)),
    }
    return combo_row, channel_rows, pair_rows, visual_data


def add_cluster_fields(
    channel_rows: list[dict[str, Any]],
    clusters: dict[int, dict[str, Any]],
    diagnostics: dict[int, dict[str, str]],
) -> None:
    for row in channel_rows:
        channel = int(row["channel"])
        cluster = clusters[channel]
        if channel == int(cluster["medoid"]):
            role = "medoid"
        elif cluster.get("second_representative") == channel:
            role = "second_representative"
        else:
            role = "member"
        row["r070_cluster"] = int(cluster["cluster_id"])
        row["cluster_size"] = int(cluster["size"])
        row["cluster_role"] = role
        row["clustering_cross_light_ncc"] = float(
            diagnostics[channel]["cross_light_ncc"]
        )
        row["clustering_robust_gradient_energy"] = float(
            diagnostics[channel]["robust_gradient_energy"]
        )


def save_feature_overview(
    label: str,
    channels: tuple[int, ...],
    clusters: dict[int, dict[str, Any]],
    visual_data: dict[int, Any],
    output: Path,
) -> None:
    limits = []
    for position in range(4):
        values = np.stack([visual_data[index]["maps"][position] for index in SELECTED_FRAMES])
        limit = float(np.percentile(values, 99.5))
        limits.append(limit if limit > 0 else 1.0)
    figure, axes = plt.subplots(3, 4, figsize=(13.6, 9.2))
    for row_index, frame_index in enumerate(SELECTED_FRAMES):
        for position, channel in enumerate(channels):
            axis = axes[row_index, position]
            axis.imshow(
                visual_data[frame_index]["maps"][position],
                cmap="viridis",
                vmin=0,
                vmax=limits[position],
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"channel {channel} · C{clusters[channel]['cluster_id']}")
            if position == 0:
                axis.set_ylabel(f"frame {frame_index}\n{FRAME_PHASES[frame_index]}")
    figure.suptitle(
        f"{label}: [{','.join(map(str, channels))}] lightswitch Conv1 post-ReLU\n"
        "Each channel has one fixed scale across before/peak/after",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def ownership_rgb(supports: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(supports)
    count = stack.sum(axis=0)
    result = np.zeros((*count.shape, 3), dtype=np.float32)
    for position, color in enumerate(CHANNEL_COLORS):
        result[stack[position] & (count == 1)] = color
    result[count >= 2] = 1.0
    return result


def save_gradient_coverage(
    label: str,
    channels: tuple[int, ...],
    visual_data: dict[int, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(3, 6, figsize=(17.0, 8.9))
    for row_index, frame_index in enumerate(SELECTED_FRAMES):
        data = visual_data[frame_index]
        supports = [gradient_data(feature_map)[1] for feature_map in data["maps"]]
        axes[row_index, 0].imshow(data["rgb"])
        axes[row_index, 0].set_ylabel(f"frame {frame_index}\n{FRAME_PHASES[frame_index]}")
        for position, channel in enumerate(channels):
            axes[row_index, position + 1].imshow(
                supports[position], cmap="gray", vmin=0, vmax=1
            )
        axes[row_index, 5].imshow(ownership_rgb(supports))
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
    titles = ["RGB"] + [f"ch {channel}\ntop-15% gradient" for channel in channels] + [
        "support ownership\ncolor=exclusive, white=overlap"
    ]
    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontsize=9.5)
    figure.suptitle(
        f"{label}: salient-gradient coverage for [{','.join(map(str, channels))}]",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def matrix_from_pairs(
    channels: tuple[int, ...], pair_rows: list[dict[str, Any]], field: str
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    position = {channel: index for index, channel in enumerate(channels)}
    for row in pair_rows:
        i = position[int(row["channel_i"])]
        j = position[int(row["channel_j"])]
        matrix[i, j] = matrix[j, i] = float(row[field])
    return matrix


def annotate_heatmap(axis: Any, matrix: np.ndarray, diagonal_text: str = "—") -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            text = diagonal_text if row == column else f"{matrix[row, column]:.3f}"
            axis.text(column, row, text, ha="center", va="center", fontsize=9)


def save_pairwise_heatmaps(
    all_pairs: dict[str, list[dict[str, Any]]], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 9.0))
    for row_index, (label, channels) in enumerate(COMBINATIONS.items()):
        corr = matrix_from_pairs(channels, all_pairs[label], "robust_abs_corr_min")
        jaccard = matrix_from_pairs(channels, all_pairs[label], "mean_gradient_jaccard")
        for column, (matrix, title, vmax) in enumerate(
            (
                (corr, "robust |correlation| (lower = less redundant)", 1.0),
                (jaccard, "gradient-support Jaccard (lower = less overlap)", 0.5),
            )
        ):
            axis = axes[row_index, column]
            shown = matrix.copy()
            np.fill_diagonal(shown, np.nan)
            image = axis.imshow(shown, cmap="magma_r", vmin=0, vmax=vmax)
            annotate_heatmap(axis, matrix)
            axis.set_xticks(range(4), labels=[f"ch {x}" for x in channels])
            axis.set_yticks(range(4), labels=[f"ch {x}" for x in channels])
            axis.set_title(f"{label} {title}", fontsize=10)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Within-combination diversity across all 30 paired timestamps")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def save_channel_roles(channel_rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.4))
    labels = [f"{row['combination']}\nch {row['channel']}" for row in channel_rows]
    x = np.arange(len(channel_rows))
    values_and_titles = (
        ("paired_structural_ncc_median", "Clean↔light structural NCC\n(higher = more stable)"),
        (
            "event_gradient_retention_median",
            "Light/clean gradient-energy ratio at events\n(1 = retained)",
        ),
        (
            "exclusive_gradient_support_fraction_mean",
            "Exclusive share of own salient gradients\n(higher = more unique coverage)",
        ),
    )
    colors = [CHANNEL_COLORS[index % 4] for index in range(len(channel_rows))]
    for axis, (field, title) in zip(axes, values_and_titles):
        values = [float(row[field]) for row in channel_rows]
        axis.bar(x, values, color=colors)
        axis.set_xticks(x, labels=labels, rotation=45, ha="right", fontsize=8)
        axis.set_title(title, fontsize=10.5)
        axis.grid(axis="y", alpha=0.25)
        if field != "event_gradient_retention_median":
            axis.set_ylim(0, 1.05)
        else:
            axis.axhline(1.0, color="#555555", ls="--", lw=1)
    figure.suptitle("Per-channel roles across all 30 paired timestamps")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def write_summary(
    output: Path,
    combo_rows: list[dict[str, Any]],
    channel_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    by_label_channels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_label_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in channel_rows:
        by_label_channels[row["combination"]].append(row)
    for row in pair_rows:
        by_label_pairs[row["combination"]].append(row)

    lines = [
        "# Two-combination Conv1 complementarity analysis",
        "",
        "## Scope and interpretation",
        "",
        "- Inputs: all 30 paired clean/lightswitch timestamps; native Conv1 post-ReLU maps.",
        "- Salient gradient support: top 15% spatial gradient magnitude within each channel/map.",
        "- Robust correlation: min of the median clean and median lightswitch absolute correlations.",
        "- These are descriptive feature-level indicators. They do not prove a causal tracking benefit.",
        "",
        "## Combination-level results",
        "",
        "| combination | channels | mean pair |r| | max pair |r| | mean gradient Jaccard | union coverage | overlap within union | effective rank / 4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combo_rows:
        lines.append(
            f"| {row['combination']} | `{row['channels']}` | "
            f"{row['mean_pair_robust_abs_corr']:.3f} | "
            f"{row['max_pair_robust_abs_corr']:.3f} | "
            f"{row['mean_pair_gradient_jaccard']:.3f} | "
            f"{row['mean_gradient_union_image_fraction']:.3f} | "
            f"{row['mean_overlap_fraction_within_union']:.3f} | "
            f"{row['mean_effective_rank_out_of_4']:.3f} |"
        )
    lines.extend(["", "## Channel roles", ""])
    for label, channels in COMBINATIONS.items():
        lines.extend(
            [
                f"### {label} `[{','.join(map(str, channels))}]`",
                "",
                "| channel | r=.70 cluster/role | clean↔light NCC | gradient retention | exclusive salient-gradient share |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for row in by_label_channels[label]:
            lines.append(
                f"| {row['channel']} | C{row['r070_cluster']} / {row['cluster_role']} | "
                f"{row['paired_structural_ncc_median']:.3f} | "
                f"{row['event_gradient_retention_median']:.3f} | "
                f"{row['exclusive_gradient_support_fraction_mean']:.3f} |"
            )
        most_stable = max(
            by_label_channels[label], key=lambda row: row["paired_structural_ncc_median"]
        )
        most_sensitive = min(
            by_label_channels[label], key=lambda row: row["paired_structural_ncc_median"]
        )
        most_unique = max(
            by_label_channels[label],
            key=lambda row: row["exclusive_gradient_support_fraction_mean"],
        )
        least_redundant_pair = min(
            by_label_pairs[label], key=lambda row: row["robust_abs_corr_min"]
        )
        lines.extend(
            [
                "",
                f"- Most illumination-stable: ch{most_stable['channel']} (NCC {most_stable['paired_structural_ncc_median']:.3f}).",
                f"- Most illumination-sensitive: ch{most_sensitive['channel']} (NCC {most_sensitive['paired_structural_ncc_median']:.3f}).",
                f"- Largest exclusive gradient-support share: ch{most_unique['channel']} ({most_unique['exclusive_gradient_support_fraction_mean']:.3f}).",
                f"- Least-correlated pair: ch{least_redundant_pair['channel_i']}+ch{least_redundant_pair['channel_j']} (robust |r| {least_redundant_pair['robust_abs_corr_min']:.3f}).",
                "",
            ]
        )
    lines.extend(
        [
            "## What can establish cooperation causally",
            "",
            "1. Instrument the tracker with per-channel residual/Jacobian/Hessian logging.",
            "2. Mask one channel at a time while retaining the four-channel tensor shape and rerun the same sequence.",
            "3. Mask channel pairs and compare the observed degradation with the sum of their single-mask degradations; the non-additive term is the pair interaction.",
            "4. Repeat the full, single-mask and pair-mask conditions across several sequences/runs before calling an interaction general.",
            "",
            "Recommended mechanism quantities: per-channel residual reduction, Hessian log-determinant/condition-number contribution, failure frame, ATE/RPE, and pair interaction effect.",
            "",
            "## Outputs",
            "",
            "- `feature_maps_*.png`: before/peak/after feature maps.",
            "- `gradient_coverage_*.png`: salient-gradient ownership; channel colors are exclusive support and white is overlap.",
            "- `pairwise_diversity.png`: correlation and gradient-overlap matrices.",
            "- `channel_roles.png`: illumination stability, gradient retention and unique coverage.",
            "- CSV files contain the exact values used in the plots.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = require_file(args.manifest, "Feature manifest")
    cluster_path = require_file(args.clusters, "Cluster JSON")
    diagnostics_path = require_file(args.diagnostics, "Numeric diagnostics")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    clusters = load_cluster_lookup(cluster_path)
    diagnostics = load_diagnostics(diagnostics_path)
    combo_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    all_pairs: dict[str, list[dict[str, Any]]] = {}

    for label, channels in COMBINATIONS.items():
        combo_row, combo_channels, combo_pairs, visual_data = analyse_combo(
            label, channels, manifest
        )
        add_cluster_fields(combo_channels, clusters, diagnostics)
        combo_rows.append(combo_row)
        channel_rows.extend(combo_channels)
        pair_rows.extend(combo_pairs)
        all_pairs[label] = combo_pairs
        suffix = "_".join(map(str, channels))
        save_feature_overview(
            label,
            channels,
            clusters,
            visual_data,
            output_dir / f"feature_maps_{suffix}.png",
        )
        save_gradient_coverage(
            label,
            channels,
            visual_data,
            output_dir / f"gradient_coverage_{suffix}.png",
        )

    save_pairwise_heatmaps(all_pairs, output_dir / "pairwise_diversity.png")
    save_channel_roles(channel_rows, output_dir / "channel_roles.png")
    write_csv(output_dir / "combination_metrics.csv", combo_rows)
    write_csv(output_dir / "channel_metrics.csv", channel_rows)
    write_csv(output_dir / "pair_metrics.csv", pair_rows)
    write_summary(output_dir / "findings.md", combo_rows, channel_rows, pair_rows)
    protocol = {
        "combinations": {key: list(value) for key, value in COMBINATIONS.items()},
        "feature_layer": "conv1_post_relu_native_resolution",
        "paired_timestamp_count": len(manifest),
        "selected_visualisation_frames": list(SELECTED_FRAMES),
        "salient_gradient_support_quantile": SUPPORT_QUANTILE,
        "correlation_definition": "min(median_clean_abs_corr, median_light_abs_corr)",
        "causal_claim": False,
        "manifest": str(manifest_path),
        "clusters": str(cluster_path),
    }
    (output_dir / "analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] Wrote two-combination analysis to {output_dir}")


if __name__ == "__main__":
    main()
