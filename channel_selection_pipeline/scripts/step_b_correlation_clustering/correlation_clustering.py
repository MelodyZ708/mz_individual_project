#!/usr/bin/env python3
"""
Step B: robust correlation clustering for ResNet-18 feature channels.

This stage is deliberately independent of Step A functional profiling. It:

* computes per-frame spatial Pearson correlations after per-channel z-scoring;
* aggregates median absolute correlations separately for clean/light inputs;
* uses D = 1 - min(R_clean, R_light) for average-linkage HCA;
* fixes |r| >= 0.90 (distance <= 0.10) as the primary partition;
* reports threshold sensitivity and stratified 20-of-30 subsampling stability;
* selects a structural medoid and, when needed, one quality-aware alternative.

Constant feature maps cannot have a Pearson correlation. They are reported as
numerically ineligible rather than imported from a Step A "dead channel" list.
NPZ matrices are retained for later computation; every two-dimensional matrix
also receives a human-readable PNG visualisation and CSV export where useful.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "mz_channel_selection_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score, silhouette_score


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_0_data_preparation"
    / "features_post_relu"
)
DEFAULT_MANIFEST = DEFAULT_FEATURE_DIR / "feature_manifest.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "channel_selection_results/step_b_correlation_clustering"
)
LAYERS = ("conv1", "layer1", "layer2")
EXPECTED_CHANNELS = {"conv1": 64, "layer1": 64, "layer2": 128}
CONDITIONS = ("clean", "light")
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step B robust channel correlation clustering."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.90,
        help="Primary robust absolute-correlation threshold (default: 0.90).",
    )
    parser.add_argument(
        "--minimum-valid-frames",
        type=int,
        default=20,
        help="Minimum valid frames required in each condition.",
    )
    parser.add_argument(
        "--stability-repeats",
        type=int,
        default=20,
        help="Number of repeated frame subsamples.",
    )
    parser.add_argument(
        "--stability-frames",
        type=int,
        default=20,
        help="Frames per stability subsample; 20 enables protocol stratification.",
    )
    parser.add_argument(
        "--ncc-spread-threshold",
        type=float,
        default=0.15,
        help="Cluster cross-light NCC spread that triggers a second representative.",
    )
    parser.add_argument(
        "--gradient-ratio-threshold",
        type=float,
        default=2.0,
        help="Cluster robust-gradient ratio that triggers a second representative.",
    )
    parser.add_argument(
        "--minimum-cluster-stability",
        type=float,
        default=0.80,
        help=(
            "Minimum within-cluster pair co-clustering probability. Unstable "
            "clusters are refined by complete-linkage consensus clustering."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Step B output files.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature manifest not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Feature manifest is empty: {path}")
    rows.sort(key=lambda row: int(row["sample_id"]))
    for row_index, row in enumerate(rows, start=1):
        feature_path = Path(row["feature_file"])
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Manifest row {row_index} references missing file: {feature_path}"
            )
    return rows


def prepare_output_directory(path: Path, overwrite: bool) -> Path:
    output_dir = path.resolve()
    marker = output_dir / "correlation_clustering_summary.json"
    if marker.exists() and not overwrite:
        raise FileExistsError(
            f"Step B outputs already exist in {output_dir}. Use --overwrite "
            "only when replacement is intended."
        )
    for child in (
        "matrices",
        "clusters",
        "diagnostics",
        "plots",
        "npz_matrix_png",
    ):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_args(args: argparse.Namespace, frame_count: int) -> None:
    if not 0.0 < args.correlation_threshold < 1.0:
        raise ValueError("--correlation-threshold must lie in (0,1)")
    if not 1 <= args.minimum_valid_frames <= frame_count:
        raise ValueError("--minimum-valid-frames must lie within the frame count")
    if args.stability_repeats < 1:
        raise ValueError("--stability-repeats must be positive")
    if not 2 <= args.stability_frames <= frame_count:
        raise ValueError("--stability-frames must lie in [2, frame_count]")
    if args.ncc_spread_threshold < 0:
        raise ValueError("--ncc-spread-threshold must be non-negative")
    if args.gradient_ratio_threshold < 1:
        raise ValueError("--gradient-ratio-threshold must be at least 1")
    if not 0.0 <= args.minimum_cluster_stability <= 1.0:
        raise ValueError("--minimum-cluster-stability must lie in [0,1]")


def spatial_correlation(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return signed channel correlation and a per-channel validity mask."""
    if feature.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got {feature.shape}")
    channel_count = feature.shape[0]
    flat = feature.astype(np.float32, copy=False).reshape(channel_count, -1)
    means = np.mean(flat, axis=1, dtype=np.float64).astype(np.float32)
    centred = flat - means[:, None]
    std = np.sqrt(
        np.mean(centred * centred, axis=1, dtype=np.float64)
    ).astype(np.float32)
    valid = np.isfinite(std) & (std > EPS)
    normalised = np.zeros_like(centred, dtype=np.float32)
    normalised[valid] = centred[valid] / std[valid, None]
    correlation = (normalised @ normalised.T) / flat.shape[1]
    correlation = np.clip(correlation, -1.0, 1.0).astype(np.float32)
    correlation[~valid, :] = np.nan
    correlation[:, ~valid] = np.nan
    valid_ids = np.flatnonzero(valid)
    correlation[valid_ids, valid_ids] = 1.0
    return correlation, valid


def gradient_energy(feature: np.ndarray) -> np.ndarray:
    """Mean squared horizontal plus vertical finite-difference energy."""
    array = feature.astype(np.float32, copy=False)
    dx = np.diff(array, axis=2)
    dy = np.diff(array, axis=1)
    return (
        np.mean(dx * dx, axis=(1, 2), dtype=np.float64)
        + np.mean(dy * dy, axis=(1, 2), dtype=np.float64)
    )


def paired_channel_ncc(
    clean: np.ndarray, light: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Signed spatial NCC between paired clean/light maps per channel."""
    channel_count = clean.shape[0]
    x = clean.astype(np.float32, copy=False).reshape(channel_count, -1)
    y = light.astype(np.float32, copy=False).reshape(channel_count, -1)
    x = x - np.mean(x, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    y = y - np.mean(y, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    sx = np.sqrt(np.mean(x * x, axis=1, dtype=np.float64))
    sy = np.sqrt(np.mean(y * y, axis=1, dtype=np.float64))
    valid = np.isfinite(sx) & np.isfinite(sy) & (sx > EPS) & (sy > EPS)
    result = np.full(channel_count, np.nan, dtype=np.float64)
    numerator = np.mean(x * y, axis=1, dtype=np.float64)
    result[valid] = numerator[valid] / (sx[valid] * sy[valid])
    return np.clip(result, -1.0, 1.0), valid


def aggregate_layer(
    manifest: list[dict[str, str]], layer: str
) -> dict[str, np.ndarray]:
    frame_count = len(manifest)
    channel_count = EXPECTED_CHANNELS[layer]
    signed = {
        condition: np.full(
            (frame_count, channel_count, channel_count), np.nan, dtype=np.float32
        )
        for condition in CONDITIONS
    }
    valid = {
        condition: np.zeros((frame_count, channel_count), dtype=bool)
        for condition in CONDITIONS
    }
    gradients = {
        condition: np.full((frame_count, channel_count), np.nan, dtype=np.float64)
        for condition in CONDITIONS
    }
    cross_ncc = np.full((frame_count, channel_count), np.nan, dtype=np.float64)

    for frame_index, row in enumerate(manifest):
        with np.load(row["feature_file"]) as archive:
            clean = archive[f"{layer}_clean"]
            light = archive[f"{layer}_light"]
            expected_shape = EXPECTED_CHANNELS[layer]
            if clean.shape[0] != expected_shape or light.shape != clean.shape:
                raise ValueError(
                    f"Unexpected {layer} arrays in {row['feature_file']}: "
                    f"clean={clean.shape}, light={light.shape}"
                )
            for condition, feature in (("clean", clean), ("light", light)):
                corr, valid_mask = spatial_correlation(feature)
                signed[condition][frame_index] = corr
                valid[condition][frame_index] = valid_mask
                gradients[condition][frame_index] = gradient_energy(feature)
            cross_ncc[frame_index], _ = paired_channel_ncc(clean, light)
        print(f"[Correlation] {layer} {frame_index + 1:02d}/{frame_count:02d}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        signed_median = {
            condition: np.nanmedian(signed[condition], axis=0)
            for condition in CONDITIONS
        }
        absolute_median = {
            condition: np.nanmedian(np.abs(signed[condition]), axis=0)
            for condition in CONDITIONS
        }
        median_cross_ncc = np.nanmedian(cross_ncc, axis=0)
        median_gradient = {
            condition: np.nanmedian(gradients[condition], axis=0)
            for condition in CONDITIONS
        }

    valid_counts = {
        condition: np.sum(valid[condition], axis=0).astype(np.int32)
        for condition in CONDITIONS
    }
    pair_valid_counts = {
        condition: (
            valid[condition].astype(np.int16).T
            @ valid[condition].astype(np.int16)
        ).astype(np.int16)
        for condition in CONDITIONS
    }
    robust_gradient = np.minimum(
        median_gradient["clean"], median_gradient["light"]
    )
    return {
        "frame_signed_clean": signed["clean"],
        "frame_signed_light": signed["light"],
        "valid_clean": valid["clean"],
        "valid_light": valid["light"],
        "valid_frame_count_clean": valid_counts["clean"],
        "valid_frame_count_light": valid_counts["light"],
        "pair_valid_count_clean": pair_valid_counts["clean"],
        "pair_valid_count_light": pair_valid_counts["light"],
        "corr_signed_clean": signed_median["clean"],
        "corr_signed_light": signed_median["light"],
        "corr_abs_clean": absolute_median["clean"],
        "corr_abs_light": absolute_median["light"],
        "cross_light_ncc": median_cross_ncc,
        "gradient_energy_clean": median_gradient["clean"],
        "gradient_energy_light": median_gradient["light"],
        "robust_gradient_energy": robust_gradient,
    }


def build_distance(
    data: dict[str, np.ndarray], minimum_valid_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eligible = (
        (data["valid_frame_count_clean"] >= minimum_valid_frames)
        & (data["valid_frame_count_light"] >= minimum_valid_frames)
    )
    eligible_ids = np.flatnonzero(eligible)
    if eligible_ids.size < 2:
        raise ValueError("Fewer than two numerically eligible channels")

    robust_full = np.minimum(data["corr_abs_clean"], data["corr_abs_light"])
    robust = robust_full[np.ix_(eligible, eligible)].astype(np.float64)
    robust = np.nan_to_num(robust, nan=0.0, posinf=1.0, neginf=0.0)
    robust = np.clip((robust + robust.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(robust, 1.0)
    distance = 1.0 - robust
    distance = np.clip((distance + distance.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)

    robust_output = np.full_like(robust_full, np.nan, dtype=np.float64)
    distance_output = np.full_like(robust_full, np.nan, dtype=np.float64)
    robust_output[np.ix_(eligible, eligible)] = robust
    distance_output[np.ix_(eligible, eligible)] = distance
    return eligible, eligible_ids, robust_output, distance_output


def relabel_by_leaf_order(labels: np.ndarray, leaf_order: np.ndarray) -> np.ndarray:
    mapping: dict[int, int] = {}
    next_label = 1
    for position in leaf_order:
        old = int(labels[position])
        if old not in mapping:
            mapping[old] = next_label
            next_label += 1
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int32)


def partition(
    linkage_matrix: np.ndarray,
    distance_threshold: float,
    leaf_order: np.ndarray,
) -> np.ndarray:
    raw = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")
    return relabel_by_leaf_order(raw, leaf_order)


def threshold_sensitivity(
    distance: np.ndarray,
    linkage_matrix: np.ndarray,
    leaf_order: np.ndarray,
) -> list[dict[str, float | int | None]]:
    channel_count = distance.shape[0]
    records: list[dict[str, float | int | None]] = []
    for threshold in np.round(np.arange(0.05, 0.701, 0.05), 2):
        labels = partition(linkage_matrix, float(threshold), leaf_order)
        counts = np.bincount(labels)[1:]
        cluster_count = len(counts)
        silhouette: float | None = None
        if 1 < cluster_count < channel_count:
            silhouette = float(
                silhouette_score(distance, labels, metric="precomputed")
            )
        records.append(
            {
                "distance_threshold": float(threshold),
                "correlation_threshold": float(1.0 - threshold),
                "cluster_count": int(cluster_count),
                "silhouette": silhouette,
                "singleton_proportion": float(np.sum(counts == 1) / channel_count),
                "compression_ratio": float((channel_count - cluster_count) / channel_count),
            }
        )
    return records


def stratified_subsample(
    manifest: list[dict[str, str]], sample_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, str]:
    """Use 8 uniform + two frames from each of six events for the 20-frame protocol."""
    if sample_size == 20:
        groups: list[tuple[np.ndarray, int]] = []
        uniform = np.asarray(
            [i for i, row in enumerate(manifest) if row["selection_source"] == "uniform"]
        )
        if uniform.size >= 8:
            groups.append((uniform, 8))
            valid_protocol = True
            for source in ("turn_on", "turn_off"):
                event_ranks = sorted(
                    {
                        row["event_rank"]
                        for row in manifest
                        if row["selection_source"] == source
                    }
                )
                if len(event_ranks) != 3:
                    valid_protocol = False
                    break
                for event_rank in event_ranks:
                    indices = np.asarray(
                        [
                            i
                            for i, row in enumerate(manifest)
                            if row["selection_source"] == source
                            and row["event_rank"] == event_rank
                        ]
                    )
                    if indices.size < 2:
                        valid_protocol = False
                        break
                    groups.append((indices, 2))
                if not valid_protocol:
                    break
            if valid_protocol and sum(count for _, count in groups) == 20:
                selected = np.concatenate(
                    [rng.choice(indices, size=count, replace=False) for indices, count in groups]
                )
                return np.sort(selected), "stratified_8_uniform_2_per_event"

    selected = rng.choice(len(manifest), size=sample_size, replace=False)
    return np.sort(selected), "unstratified_without_replacement"


def subset_distance(
    data: dict[str, np.ndarray], subset: np.ndarray, eligible: np.ndarray
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clean = np.nanmedian(
            np.abs(data["frame_signed_clean"][subset]), axis=0
        )
        light = np.nanmedian(
            np.abs(data["frame_signed_light"][subset]), axis=0
        )
    robust = np.minimum(clean, light)[np.ix_(eligible, eligible)]
    robust = np.nan_to_num(robust, nan=0.0, posinf=1.0, neginf=0.0)
    robust = np.clip((robust + robust.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(robust, 1.0)
    distance = 1.0 - robust
    np.fill_diagonal(distance, 0.0)
    return distance


def medoid_local(members: np.ndarray, distance: np.ndarray) -> int:
    within = distance[np.ix_(members, members)]
    sums = np.sum(within, axis=1)
    return int(members[int(np.argmin(sums))])


def bootstrap_stability(
    manifest: list[dict[str, str]],
    data: dict[str, np.ndarray],
    eligible: np.ndarray,
    full_labels: np.ndarray,
    repeats: int,
    sample_size: int,
    distance_threshold: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    channel_count = len(full_labels)
    cocluster_sum = np.zeros((channel_count, channel_count), dtype=np.float64)
    distance_matrices: list[np.ndarray] = []
    ari_values: list[float] = []
    retention_values: list[float] = []
    sampling_modes: Counter[str] = Counter()
    medoid_counts: dict[int, Counter[int]] = {
        int(cluster): Counter() for cluster in np.unique(full_labels)
    }
    upper = np.triu(np.ones((channel_count, channel_count), dtype=bool), k=1)
    full_cocluster = full_labels[:, None] == full_labels[None, :]
    retained_pairs = upper & full_cocluster

    for _ in range(repeats):
        subset, mode = stratified_subsample(manifest, sample_size, rng)
        sampling_modes[mode] += 1
        distance = subset_distance(data, subset, eligible)
        distance_matrices.append(distance)
        z = linkage(squareform(distance, checks=False), method="average")
        order = leaves_list(z)
        labels = partition(z, distance_threshold, order)
        current_cocluster = labels[:, None] == labels[None, :]
        cocluster_sum += current_cocluster
        # sklearn warns that partitions with mostly singleton labels resemble a
        # regression target. These are valid clustering labels here.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The number of unique classes is greater than 50%",
                category=UserWarning,
            )
            ari_values.append(float(adjusted_rand_score(full_labels, labels)))
        if np.any(retained_pairs):
            retention_values.append(float(np.mean(current_cocluster[retained_pairs])))
        else:
            retention_values.append(1.0)
        for cluster in np.unique(full_labels):
            members = np.flatnonzero(full_labels == cluster)
            medoid_counts[int(cluster)][medoid_local(members, distance)] += 1

    probability = cocluster_sum / repeats
    cluster_stability: dict[int, float] = {}
    cluster_stability_minimum: dict[int, float] = {}
    for cluster in np.unique(full_labels):
        members = np.flatnonzero(full_labels == cluster)
        if len(members) == 1:
            cluster_stability[int(cluster)] = 1.0
            cluster_stability_minimum[int(cluster)] = 1.0
        else:
            within_upper = np.triu_indices(len(members), k=1)
            within = probability[np.ix_(members, members)][within_upper]
            cluster_stability[int(cluster)] = float(np.mean(within))
            cluster_stability_minimum[int(cluster)] = float(np.min(within))

    return {
        "cocluster_probability": probability,
        "ari_values": np.asarray(ari_values),
        "retention_values": np.asarray(retention_values),
        "sampling_modes": dict(sampling_modes),
        "cluster_stability": cluster_stability,
        "cluster_stability_minimum": cluster_stability_minimum,
        "medoid_counts": medoid_counts,
        "distance_matrices": distance_matrices,
    }


def enforce_bootstrap_stability(
    raw_labels: np.ndarray,
    leaf_order: np.ndarray,
    stability: dict[str, Any],
    minimum_stability: float,
) -> tuple[np.ndarray, dict[str, Any], list[int]]:
    """Refine unstable raw clusters into complete-linkage consensus subclusters."""
    provisional = np.zeros_like(raw_labels, dtype=np.int32)
    rejected_raw_clusters: list[int] = []
    next_provisional = 1
    for raw_cluster in np.unique(raw_labels):
        members = np.flatnonzero(raw_labels == raw_cluster)
        if (
            len(members) == 1
            or stability["cluster_stability_minimum"][int(raw_cluster)]
            >= minimum_stability
        ):
            provisional[members] = next_provisional
            next_provisional += 1
        else:
            rejected_raw_clusters.append(int(raw_cluster))
            probability = stability["cocluster_probability"][np.ix_(members, members)]
            consensus_distance = np.clip(1.0 - probability, 0.0, 1.0)
            consensus_distance = (
                consensus_distance + consensus_distance.T
            ) / 2.0
            np.fill_diagonal(consensus_distance, 0.0)
            consensus_linkage = linkage(
                squareform(consensus_distance, checks=False), method="complete"
            )
            sublabels = fcluster(
                consensus_linkage,
                t=round(1.0 - minimum_stability, 10),
                criterion="distance",
            )
            for sublabel in np.unique(sublabels):
                provisional[members[sublabels == sublabel]] = next_provisional
                next_provisional += 1

    final_labels = relabel_by_leaf_order(provisional, leaf_order)

    final_mean: dict[int, float] = {}
    final_minimum: dict[int, float] = {}
    final_medoid_counts: dict[int, Counter[int]] = {}
    for final_cluster in np.unique(final_labels):
        members = np.flatnonzero(final_labels == final_cluster)
        if len(members) == 1:
            local = int(members[0])
            final_mean[int(final_cluster)] = 1.0
            final_minimum[int(final_cluster)] = 1.0
            final_medoid_counts[int(final_cluster)] = Counter(
                {local: len(stability["distance_matrices"])}
            )
        else:
            within_upper = np.triu_indices(len(members), k=1)
            within = stability["cocluster_probability"][np.ix_(members, members)][
                within_upper
            ]
            final_mean[int(final_cluster)] = float(np.mean(within))
            final_minimum[int(final_cluster)] = float(np.min(within))
            counts: Counter[int] = Counter()
            for distance in stability["distance_matrices"]:
                counts[medoid_local(members, distance)] += 1
            final_medoid_counts[int(final_cluster)] = counts

    final_stability = dict(stability)
    final_stability["cluster_stability"] = final_mean
    final_stability["cluster_stability_minimum"] = final_minimum
    final_stability["medoid_counts"] = final_medoid_counts
    return final_labels, final_stability, rejected_raw_clusters


def quality_percentiles(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    finite = eligible & np.isfinite(values)
    if np.any(finite):
        ranks = rankdata(values[finite], method="average")
        output[finite] = ranks / len(ranks)
    return output


def select_representatives(
    layer: str,
    labels: np.ndarray,
    eligible_ids: np.ndarray,
    distance: np.ndarray,
    data: dict[str, np.ndarray],
    stability: dict[str, Any],
    ncc_spread_threshold: float,
    gradient_ratio_threshold: float,
) -> list[dict[str, Any]]:
    ncc = data["cross_light_ncc"]
    gradient = data["robust_gradient_energy"]
    eligible_mask = np.zeros_like(ncc, dtype=bool)
    eligible_mask[eligible_ids] = True
    ncc_quality = quality_percentiles(ncc, eligible_mask)
    gradient_quality = quality_percentiles(np.log10(gradient + EPS), eligible_mask)
    quality = (ncc_quality + gradient_quality) / 2.0

    clusters: list[dict[str, Any]] = []
    for cluster_id in np.unique(labels):
        local_members = np.flatnonzero(labels == cluster_id)
        members = eligible_ids[local_members]
        medoid_position = medoid_local(local_members, distance)
        medoid = int(eligible_ids[medoid_position])
        ncc_values = ncc[members]
        grad_values = gradient[members]
        finite_ncc = ncc_values[np.isfinite(ncc_values)]
        finite_grad = grad_values[np.isfinite(grad_values)]
        ncc_spread = (
            float(np.max(finite_ncc) - np.min(finite_ncc))
            if finite_ncc.size
            else 0.0
        )
        if finite_grad.size:
            minimum_gradient = float(np.min(finite_grad))
            maximum_gradient = float(np.max(finite_grad))
            gradient_ratio = (
                maximum_gradient / minimum_gradient
                if minimum_gradient > EPS
                else (float("inf") if maximum_gradient > EPS else 1.0)
            )
        else:
            gradient_ratio = 1.0

        second: int | None = None
        trigger_reasons: list[str] = []
        if len(members) > 1 and ncc_spread >= ncc_spread_threshold:
            trigger_reasons.append("cross_light_ncc_spread")
        if len(members) > 1 and gradient_ratio >= gradient_ratio_threshold:
            trigger_reasons.append("robust_gradient_ratio")
        if trigger_reasons:
            alternatives = members[members != medoid]
            second = int(alternatives[np.argmax(quality[alternatives])])

        medoid_frequency = stability["medoid_counts"][int(cluster_id)]
        frequency_payload = {
            str(int(eligible_ids[local])): int(count)
            for local, count in sorted(medoid_frequency.items())
        }
        clusters.append(
            {
                "layer": layer,
                "cluster_id": int(cluster_id),
                "size": int(len(members)),
                "members": [int(channel) for channel in members],
                "medoid": medoid,
                "second_representative": second,
                "second_representative_trigger": trigger_reasons,
                "cross_light_ncc_spread": ncc_spread,
                "robust_gradient_ratio": gradient_ratio,
                "bootstrap_cluster_stability": stability["cluster_stability"][
                    int(cluster_id)
                ],
                "bootstrap_cluster_minimum_pair_stability": stability[
                    "cluster_stability_minimum"
                ][int(cluster_id)],
                "bootstrap_medoid_frequency": frequency_payload,
                "member_metrics": {
                    str(int(channel)): {
                        "cross_light_ncc": float(ncc[channel]),
                        "robust_gradient_energy": float(gradient[channel]),
                        "quality_percentile_score": float(quality[channel]),
                    }
                    for channel in members
                },
            }
        )
    return clusters


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["channel"] + [f"ch_{i}" for i in range(matrix.shape[1])])
        for channel, row in enumerate(matrix):
            writer.writerow(
                [channel]
                + ["" if not np.isfinite(value) else f"{value:.8f}" for value in row]
            )


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def plot_matrix(
    matrix: np.ndarray,
    path: Path,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    size = matrix.shape[0]
    figure_size = 9 if size <= 64 else 11
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    masked = np.ma.masked_invalid(matrix)
    colourmap = matplotlib.colormaps[cmap].copy()
    colourmap.set_bad("#d9d9d9")
    image = ax.imshow(masked, cmap=colourmap, vmin=vmin, vmax=vmax, interpolation="none")
    tick_step = 4 if size <= 64 else 8
    ticks = np.arange(0, size, tick_step)
    ax.set_xticks(ticks, labels=ticks, rotation=90, fontsize=7)
    ax.set_yticks(ticks, labels=ticks, fontsize=7)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sorted_heatmap(
    layer: str,
    robust: np.ndarray,
    eligible_ids: np.ndarray,
    leaf_order: np.ndarray,
    labels: np.ndarray,
    path: Path,
) -> None:
    ordered = robust[np.ix_(leaf_order, leaf_order)]
    ordered_ids = eligible_ids[leaf_order]
    ordered_labels = labels[leaf_order]
    fig_size = 10 if len(eligible_ids) <= 64 else 12
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(ordered, cmap="viridis", vmin=0, vmax=1, interpolation="none")
    step = 4 if len(eligible_ids) <= 64 else 8
    ticks = np.arange(0, len(eligible_ids), step)
    ax.set_xticks(ticks, labels=ordered_ids[ticks], rotation=90, fontsize=7)
    ax.set_yticks(ticks, labels=ordered_ids[ticks], fontsize=7)
    boundaries = np.flatnonzero(np.diff(ordered_labels)) + 0.5
    for boundary in boundaries:
        ax.axhline(boundary, color="white", linewidth=0.35, alpha=0.7)
        ax.axvline(boundary, color="white", linewidth=0.35, alpha=0.7)
    ax.set_title(f"{layer}: robust |r|, HCA leaf order")
    ax.set_xlabel("Original channel ID")
    ax.set_ylabel("Original channel ID")
    fig.colorbar(image, ax=ax, label="min(median |r| clean, median |r| light)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dendrogram(
    layer: str,
    linkage_matrix: np.ndarray,
    eligible_ids: np.ndarray,
    distance_threshold: float,
    path: Path,
) -> None:
    width = 15 if len(eligible_ids) <= 64 else 22
    fig, ax = plt.subplots(figsize=(width, 7))
    dendrogram(
        linkage_matrix,
        labels=[str(channel) for channel in eligible_ids],
        leaf_rotation=90,
        leaf_font_size=7,
        color_threshold=distance_threshold,
        above_threshold_color="#777777",
        ax=ax,
    )
    ax.axhline(
        distance_threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"primary distance={distance_threshold:.2f}",
    )
    ax.set_title(f"{layer}: average-linkage dendrogram")
    ax.set_xlabel("Original channel ID")
    ax.set_ylabel("Distance = 1 - robust |r|")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sensitivity(
    layer: str,
    records: list[dict[str, Any]],
    primary_distance: float,
    path: Path,
) -> None:
    x = np.asarray([row["distance_threshold"] for row in records])
    silhouette = np.asarray(
        [np.nan if row["silhouette"] is None else row["silhouette"] for row in records]
    )
    clusters = np.asarray([row["cluster_count"] for row in records])
    singleton = np.asarray([row["singleton_proportion"] for row in records])
    compression = np.asarray([row["compression_ratio"] for row in records])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    panels = (
        (silhouette, "Silhouette", "tab:blue"),
        (clusters, "Number of clusters", "tab:orange"),
        (singleton, "Singleton proportion", "tab:green"),
        (compression, "Compression ratio", "tab:purple"),
    )
    for ax, (values, label, colour) in zip(axes.flat, panels):
        ax.plot(x, values, marker="o", color=colour)
        ax.axvline(primary_distance, color="red", linestyle="--", linewidth=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("HCA distance threshold (1 - |r|)")
    axes[1, 1].set_xlabel("HCA distance threshold (1 - |r|)")
    fig.suptitle(
        f"{layer}: threshold sensitivity (red = pre-registered primary threshold)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_stability(layer: str, stability: dict[str, Any], path: Path) -> None:
    ari = stability["ari_values"]
    retention = stability["retention_values"]
    matrix = stability["cocluster_probability"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(np.arange(1, len(ari) + 1), ari, marker="o")
    axes[0].axhline(np.mean(ari), color="red", linestyle="--", label=f"mean={np.mean(ari):.3f}")
    axes[0].set_title("Adjusted Rand Index")
    axes[0].set_xlabel("Subsample repeat")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[1].plot(np.arange(1, len(retention) + 1), retention, marker="o", color="tab:green")
    axes[1].axhline(
        np.mean(retention),
        color="red",
        linestyle="--",
        label=f"mean={np.mean(retention):.3f}",
    )
    axes[1].set_title("Full-cluster pair retention")
    axes[1].set_xlabel("Subsample repeat")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()
    image = axes[2].imshow(matrix, cmap="viridis", vmin=0, vmax=1, interpolation="none")
    axes[2].set_title("Co-clustering probability")
    axes[2].set_xlabel("Eligible-channel position")
    axes[2].set_ylabel("Eligible-channel position")
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(f"{layer}: stratified subsampling stability")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def process_layer(
    layer: str,
    manifest: list[dict[str, str]],
    output_dir: Path,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict[str, Any]:
    data = aggregate_layer(manifest, layer)
    eligible, eligible_ids, robust_full, distance_full = build_distance(
        data, args.minimum_valid_frames
    )
    robust = robust_full[np.ix_(eligible, eligible)]
    distance = distance_full[np.ix_(eligible, eligible)]
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    leaf_order = leaves_list(linkage_matrix)
    primary_distance = round(1.0 - args.correlation_threshold, 10)
    raw_labels = partition(linkage_matrix, primary_distance, leaf_order)
    sensitivity = threshold_sensitivity(distance, linkage_matrix, leaf_order)
    raw_stability = bootstrap_stability(
        manifest,
        data,
        eligible,
        raw_labels,
        args.stability_repeats,
        args.stability_frames,
        primary_distance,
        rng,
    )
    labels, stability, rejected_raw_clusters = enforce_bootstrap_stability(
        raw_labels,
        leaf_order,
        raw_stability,
        args.minimum_cluster_stability,
    )
    clusters = select_representatives(
        layer,
        labels,
        eligible_ids,
        distance,
        data,
        stability,
        args.ncc_spread_threshold,
        args.gradient_ratio_threshold,
    )

    full_cocluster = np.full((len(eligible), len(eligible)), np.nan, dtype=np.float64)
    full_cocluster[np.ix_(eligible, eligible)] = stability["cocluster_probability"]
    matrices = {
        "corr_signed_clean": data["corr_signed_clean"].astype(np.float32),
        "corr_signed_light": data["corr_signed_light"].astype(np.float32),
        "corr_abs_clean": data["corr_abs_clean"].astype(np.float32),
        "corr_abs_light": data["corr_abs_light"].astype(np.float32),
        "corr_robust": robust_full.astype(np.float32),
        "distance": distance_full.astype(np.float32),
        "bootstrap_cocluster_probability": full_cocluster.astype(np.float32),
    }
    matrix_path = output_dir / "matrices" / f"correlation_matrices_{layer}.npz"
    np.savez_compressed(
        matrix_path,
        channels=np.arange(len(eligible), dtype=np.int32),
        eligible_mask=eligible,
        valid_frame_count_clean=data["valid_frame_count_clean"],
        valid_frame_count_light=data["valid_frame_count_light"],
        pair_valid_count_clean=data["pair_valid_count_clean"],
        pair_valid_count_light=data["pair_valid_count_light"],
        cross_light_ncc=data["cross_light_ncc"].astype(np.float32),
        gradient_energy_clean=data["gradient_energy_clean"].astype(np.float32),
        gradient_energy_light=data["gradient_energy_light"].astype(np.float32),
        robust_gradient_energy=data["robust_gradient_energy"].astype(np.float32),
        cluster_labels_full=np.asarray(
            [labels[np.flatnonzero(eligible_ids == channel)[0]] if eligible[channel] else -1 for channel in range(len(eligible))],
            dtype=np.int32,
        ),
        **matrices,
    )

    for name in (
        "corr_abs_clean",
        "corr_abs_light",
        "corr_robust",
        "distance",
    ):
        write_matrix_csv(
            output_dir / "matrices" / f"{name}_{layer}.csv", matrices[name]
        )
    matrix_plot_settings = {
        "corr_signed_clean": ("coolwarm", -1.0, 1.0),
        "corr_signed_light": ("coolwarm", -1.0, 1.0),
        "corr_abs_clean": ("viridis", 0.0, 1.0),
        "corr_abs_light": ("viridis", 0.0, 1.0),
        "corr_robust": ("viridis", 0.0, 1.0),
        "distance": ("magma", 0.0, 1.0),
        "bootstrap_cocluster_probability": ("viridis", 0.0, 1.0),
    }
    for name, matrix in matrices.items():
        cmap, vmin, vmax = matrix_plot_settings[name]
        plot_matrix(
            matrix,
            output_dir / "npz_matrix_png" / f"{layer}_{name}.png",
            f"{layer}: {name} (visual copy of NPZ matrix)",
            cmap,
            vmin,
            vmax,
        )

    write_records_csv(
        output_dir / "diagnostics" / f"threshold_sensitivity_{layer}.csv",
        sensitivity,
    )
    ineligible = np.flatnonzero(~eligible).tolist()
    numeric_records = [
        {
            "channel": channel,
            "eligible": bool(eligible[channel]),
            "valid_frames_clean": int(data["valid_frame_count_clean"][channel]),
            "valid_frames_light": int(data["valid_frame_count_light"][channel]),
            "cross_light_ncc": float(data["cross_light_ncc"][channel]),
            "gradient_energy_clean": float(data["gradient_energy_clean"][channel]),
            "gradient_energy_light": float(data["gradient_energy_light"][channel]),
            "robust_gradient_energy": float(data["robust_gradient_energy"][channel]),
        }
        for channel in range(len(eligible))
    ]
    write_records_csv(
        output_dir / "diagnostics" / f"channel_numeric_diagnostics_{layer}.csv",
        numeric_records,
    )
    representative_records: list[dict[str, Any]] = []
    for cluster in clusters:
        representative_records.append(
            {
                "cluster_id": cluster["cluster_id"],
                "cluster_size": cluster["size"],
                "representative_role": "medoid",
                "channel": cluster["medoid"],
                "trigger": "structural_centrality",
                "cross_light_ncc": data["cross_light_ncc"][cluster["medoid"]],
                "robust_gradient_energy": data["robust_gradient_energy"][cluster["medoid"]],
            }
        )
        if cluster["second_representative"] is not None:
            second = cluster["second_representative"]
            representative_records.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "cluster_size": cluster["size"],
                    "representative_role": "second",
                    "channel": second,
                    "trigger": ";".join(cluster["second_representative_trigger"]),
                    "cross_light_ncc": data["cross_light_ncc"][second],
                    "robust_gradient_energy": data["robust_gradient_energy"][second],
                }
            )
    write_records_csv(
        output_dir / "clusters" / f"representatives_{layer}.csv",
        representative_records,
    )

    cluster_payload = {
        "layer": layer,
        "primary_correlation_threshold": args.correlation_threshold,
        "primary_distance_threshold": primary_distance,
        "linkage": "average",
        "bootstrap_stability_rule": {
            "metric": "minimum within-cluster pair co-clustering probability",
            "minimum": args.minimum_cluster_stability,
            "action_when_below_minimum": (
                "complete-linkage consensus split within the raw cluster"
            ),
            "raw_cluster_count": int(len(np.unique(raw_labels))),
            "rejected_raw_cluster_ids": rejected_raw_clusters,
        },
        "eligible_channels": eligible_ids.tolist(),
        "numerically_ineligible_channels": ineligible,
        "clusters": clusters,
        "representative_channels": sorted(
            {
                channel
                for cluster in clusters
                for channel in (cluster["medoid"], cluster["second_representative"])
                if channel is not None
            }
        ),
    }
    cluster_path = output_dir / "clusters" / f"clusters_{layer}.json"
    cluster_path.write_text(
        json.dumps(json_safe(cluster_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stability_payload = {
        "layer": layer,
        "repeats": args.stability_repeats,
        "frames_per_repeat": args.stability_frames,
        "sampling_modes": stability["sampling_modes"],
        "adjusted_rand_index": stability["ari_values"],
        "full_cluster_pair_retention": stability["retention_values"],
        "mean_adjusted_rand_index": float(np.mean(stability["ari_values"])),
        "mean_full_cluster_pair_retention": float(
            np.mean(stability["retention_values"])
        ),
        "raw_cluster_stability_mean": raw_stability["cluster_stability"],
        "raw_cluster_stability_minimum_pair": raw_stability[
            "cluster_stability_minimum"
        ],
        "rejected_raw_cluster_ids": rejected_raw_clusters,
        "final_cluster_stability_mean": stability["cluster_stability"],
        "final_cluster_stability_minimum_pair": stability[
            "cluster_stability_minimum"
        ],
    }
    (output_dir / "diagnostics" / f"bootstrap_stability_{layer}.json").write_text(
        json.dumps(json_safe(stability_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_sorted_heatmap(
        layer,
        robust,
        eligible_ids,
        leaf_order,
        labels,
        output_dir / "plots" / f"correlation_heatmap_{layer}.png",
    )
    plot_dendrogram(
        layer,
        linkage_matrix,
        eligible_ids,
        primary_distance,
        output_dir / "plots" / f"dendrogram_{layer}.png",
    )
    plot_sensitivity(
        layer,
        sensitivity,
        primary_distance,
        output_dir / "plots" / f"threshold_sensitivity_{layer}.png",
    )
    plot_stability(
        layer,
        stability,
        output_dir / "plots" / f"bootstrap_stability_{layer}.png",
    )

    cluster_sizes = [cluster["size"] for cluster in clusters]
    representative_count = len(cluster_payload["representative_channels"])
    result = {
        "channels_total": len(eligible),
        "channels_eligible": int(np.sum(eligible)),
        "numerically_ineligible_channels": ineligible,
        "cluster_count": len(clusters),
        "raw_cluster_count_before_stability_filter": int(
            len(np.unique(raw_labels))
        ),
        "bootstrap_rejected_raw_cluster_count": len(rejected_raw_clusters),
        "cluster_sizes_descending": sorted(cluster_sizes, reverse=True),
        "singleton_clusters": int(np.sum(np.asarray(cluster_sizes) == 1)),
        "medoid_count": len(clusters),
        "second_representative_count": int(
            sum(cluster["second_representative"] is not None for cluster in clusters)
        ),
        "representative_count": representative_count,
        "mean_bootstrap_ari": float(np.mean(stability["ari_values"])),
        "mean_pair_retention": float(np.mean(stability["retention_values"])),
        "matrix_npz": str(matrix_path),
        "clusters_json": str(cluster_path),
    }
    print(
        f"[Layer] {layer}: eligible={result['channels_eligible']}/{result['channels_total']}, "
        f"clusters={result['cluster_count']}, representatives={representative_count}"
    )
    return result


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest.resolve())
    validate_args(args, len(manifest))
    output_dir = prepare_output_directory(args.output_dir, args.overwrite)
    rng = np.random.default_rng(args.seed)
    print(f"[Input] {len(manifest)} paired feature archives")
    print("[Independence] Step A functional-profile files are not read")
    print(
        f"[Primary] |r| >= {args.correlation_threshold:.2f}; "
        f"average linkage; minimum valid frames={args.minimum_valid_frames}"
    )

    layer_results = {
        layer: process_layer(layer, manifest, output_dir, args, rng)
        for layer in LAYERS
    }
    ineligible_payload = {
        "status": "numerical eligibility for Pearson correlation only",
        "not_a_functional_profile": True,
        "standard_deviation_epsilon": EPS,
        "minimum_valid_frames_per_condition": args.minimum_valid_frames,
        "layers": {
            layer: {
                "channels": result["numerically_ineligible_channels"],
                "count": len(result["numerically_ineligible_channels"]),
            }
            for layer, result in layer_results.items()
        },
    }
    (output_dir / "diagnostics" / "numerically_ineligible_channels.json").write_text(
        json.dumps(ineligible_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "step": "B - Robust Correlation Clustering",
        "principle": (
            "Redundancy reduction only. No functional labels are imported, and "
            "ATE remains the final channel-combination criterion."
        ),
        "input": {
            "manifest": str(args.manifest.resolve()),
            "paired_timestamps": len(manifest),
            "activation_position": "post-ReLU",
            "feature_resolution": "native",
        },
        "method": {
            "per_frame_normalisation": "spatial zero mean / unit variance per channel",
            "per_condition_aggregation": "median(abs(Pearson r)) across valid frames",
            "robust_similarity": "min(R_clean, R_light)",
            "distance": "1 - robust_similarity",
            "linkage": "average",
            "primary_correlation_threshold": args.correlation_threshold,
            "primary_distance_threshold": round(
                1.0 - args.correlation_threshold, 10
            ),
            "threshold_sensitivity_distance_range": [0.05, 0.70, 0.05],
            "stability": {
                "repeats": args.stability_repeats,
                "frames_per_repeat": args.stability_frames,
                "sampling": "stratified when the 30-frame protocol is present",
                "seed": args.seed,
            },
            "second_representative": {
                "ncc_spread_threshold": args.ncc_spread_threshold,
                "robust_gradient_ratio_threshold": args.gradient_ratio_threshold,
            },
            "cluster_stability_acceptance": {
                "metric": "minimum within-cluster pair co-clustering probability",
                "minimum": args.minimum_cluster_stability,
                "unstable_cluster_action": (
                    "complete-linkage consensus split within each raw cluster"
                ),
            },
        },
        "storage": {
            "numeric": "compressed NPZ plus CSV matrices",
            "visual": "one PNG per two-dimensional NPZ matrix plus analysis plots",
        },
        "layers": layer_results,
    }
    summary_path = output_dir / "correlation_clustering_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Output] {output_dir}")
    print("[Done] Step B correlation clustering complete")


if __name__ == "__main__":
    main()
