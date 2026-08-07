#!/usr/bin/env python3
"""
Step A: functional profiling for post-ReLU ResNet-18 channels.

This script consumes the per-timestamp NPZ archives from Step 0 and produces:

  * functional_profile_conv1.csv
  * functional_profile_layer1.csv
  * functional_profile_layer2.csv
  * one robust-z descriptor heatmap per layer
  * confirmed_dead_candidates.json
  * one manual-review image per layer
  * functional_profiling_summary.json

Functional labels are soft, multi-axis descriptions. They do not remove
channels or constrain later combinations. "confirmed_dead_candidate" also
remains a candidate until its review image has been inspected manually.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(tempfile.gettempdir()) / "mz_channel_selection_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
import seaborn as sns
import torch
import torchvision
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_0_data_preparation"
    / "features_post_relu"
)
DEFAULT_MANIFEST = DEFAULT_FEATURE_DIR / "feature_manifest.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_a_functional_profiling"
)
LAYERS = ("conv1", "layer1", "layer2")
CHANNEL_COUNTS = {"conv1": 64, "layer1": 64, "layer2": 128}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Step A functional channel descriptors."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=20,
        help="Number of frame-subset repeats for dead-candidate stability.",
    )
    parser.add_argument(
        "--bootstrap-frames",
        type=int,
        default=20,
        help="Frames sampled without replacement in each bootstrap repeat.",
    )
    parser.add_argument(
        "--dead-median-fraction",
        type=float,
        default=0.01,
        help="Dead threshold as a fraction of the layer median.",
    )
    parser.add_argument(
        "--dead-stability-threshold",
        type=float,
        default=0.90,
        help="Minimum bootstrap selection frequency for a dead candidate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing Step A reports.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature manifest not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Feature manifest is empty: {path}")
    for row_index, row in enumerate(rows, start=1):
        feature_path = Path(row["feature_file"])
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"Manifest row {row_index} references missing file: "
                f"{feature_path}"
            )
    rows.sort(key=lambda row: int(row["sample_id"]))
    return rows


def prepare_output_directory(path: Path, overwrite: bool) -> Path:
    output_dir = path.resolve()
    known = [
        output_dir / f"functional_profile_{layer}.csv" for layer in LAYERS
    ] + [
        output_dir / "confirmed_dead_candidates.json",
        output_dir / "functional_profiling_summary.json",
    ]
    if any(item.exists() for item in known) and not overwrite:
        raise FileExistsError(
            f"Step A outputs already exist in {output_dir}. Use --overwrite "
            "only when replacement is intended."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "heatmaps").mkdir(exist_ok=True)
    (output_dir / "dead_channel_review").mkdir(exist_ok=True)
    return output_dir


def spatial_gradient_metrics(
    feature: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return energy, orientation coherence, and dominant angle per channel."""
    array = feature.astype(np.float32, copy=False)
    gy, gx = np.gradient(array, axis=(1, 2))
    gx2 = np.sum(gx * gx, axis=(1, 2), dtype=np.float64)
    gy2 = np.sum(gy * gy, axis=(1, 2), dtype=np.float64)
    gxy = np.sum(gx * gy, axis=(1, 2), dtype=np.float64)
    pixel_count = feature.shape[1] * feature.shape[2]
    energy = (gx2 + gy2) / pixel_count
    coherence = np.sqrt((gx2 - gy2) ** 2 + 4.0 * gxy**2) / (
        gx2 + gy2 + EPS
    )
    angle = np.mod(
        0.5 * np.degrees(np.arctan2(2.0 * gxy, gx2 - gy2)), 180.0
    )
    return energy, coherence, angle


def spectral_centroid(feature: np.ndarray) -> np.ndarray:
    """Normalised radial spectral centroid, 0=DC and 1=Nyquist corner."""
    array = feature.astype(np.float32, copy=False)
    array = array - np.mean(array, axis=(1, 2), keepdims=True)
    spectrum = scipy.fft.rfft2(array, axes=(-2, -1), workers=1)
    power = np.abs(spectrum) ** 2
    height, width = feature.shape[1:]
    fy = np.fft.fftfreq(height).astype(np.float32)
    fx = np.fft.rfftfreq(width).astype(np.float32)
    radius = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    radius /= np.sqrt(0.5**2 + 0.5**2)
    numerator = np.sum(
        power * radius[None, :, :], axis=(1, 2), dtype=np.float64
    )
    denominator = np.sum(power, axis=(1, 2), dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPS,
    )


def spatial_entropy(feature: np.ndarray) -> np.ndarray:
    """Entropy of non-negative activation mass, normalised to [0,1]."""
    array = np.maximum(feature.astype(np.float64, copy=False), 0.0)
    total = np.sum(array, axis=(1, 2), keepdims=True)
    probability = np.divide(
        array, total, out=np.zeros_like(array), where=total > EPS
    )
    log_probability = np.zeros_like(probability)
    positive = probability > 0
    log_probability[positive] = np.log(probability[positive])
    entropy = -np.sum(probability * log_probability, axis=(1, 2))
    return entropy / np.log(feature.shape[1] * feature.shape[2])


def component_metrics(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Count salient connected regions and return their mean fractional area.

    Saliency is defined per map as activation > mean(positive) + std(positive).
    Components smaller than 0.05% of the native map (minimum 2 pixels) are
    ignored so isolated interpolation/noise pixels do not dominate the count.
    """
    channel_count, height, width = feature.shape
    counts = np.zeros(channel_count, dtype=np.float64)
    scales = np.zeros(channel_count, dtype=np.float64)
    minimum_area = max(2, int(round(0.0005 * height * width)))
    for channel in range(channel_count):
        fmap = feature[channel].astype(np.float32, copy=False)
        positive = fmap[fmap > 0]
        if positive.size == 0:
            continue
        threshold = float(positive.mean() + positive.std())
        mask = (fmap > threshold).astype(np.uint8)
        number, _, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if number <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        areas = areas[areas >= minimum_area]
        if areas.size:
            counts[channel] = float(areas.size)
            scales[channel] = float(np.mean(areas) / (height * width))
    return counts, scales


def paired_metrics(
    clean: np.ndarray, light: np.ndarray
) -> dict[str, np.ndarray]:
    channel_count = clean.shape[0]
    clean_flat = clean.reshape(channel_count, -1).astype(np.float64)
    light_flat = light.reshape(channel_count, -1).astype(np.float64)
    clean_mean = clean_flat.mean(axis=1)
    light_mean = light_flat.mean(axis=1)
    clean_centered = clean_flat - clean_mean[:, None]
    light_centered = light_flat - light_mean[:, None]
    numerator = np.sum(clean_centered * light_centered, axis=1)
    denominator = np.sqrt(
        np.sum(clean_centered**2, axis=1)
        * np.sum(light_centered**2, axis=1)
    )
    ncc = np.divide(
        numerator,
        denominator,
        out=np.full(channel_count, np.nan, dtype=np.float64),
        where=denominator > EPS,
    )
    clean_abs_mean = np.mean(np.abs(clean_flat), axis=1)
    mad = np.mean(np.abs(light_flat - clean_flat), axis=1)
    relative_mad = mad / (clean_abs_mean + EPS)
    mean_shift_absolute = light_mean - clean_mean
    mean_shift_relative = mean_shift_absolute / (clean_abs_mean + EPS)
    return {
        "cross_light_ncc": ncc,
        "relative_mad": relative_mad,
        "mean_shift_absolute": mean_shift_absolute,
        "mean_shift_relative": mean_shift_relative,
    }


def initialise_frame_metrics(
    frame_count: int, channel_count: int, include_region_metrics: bool
) -> dict:
    names = [
        "activation_variance_clean",
        "activation_variance_light",
        "active_ratio_clean",
        "active_ratio_light",
        "gradient_energy_clean",
        "gradient_energy_light",
        "orientation_coherence_clean",
        "orientation_coherence_light",
        "dominant_angle_clean",
        "dominant_angle_light",
        "spectral_centroid_clean",
        "spectral_centroid_light",
        "cross_light_ncc",
        "relative_mad",
        "mean_shift_absolute",
        "mean_shift_relative",
        "gradient_retention",
    ]
    result = {
        name: np.full((frame_count, channel_count), np.nan, dtype=np.float64)
        for name in names
    }
    if include_region_metrics:
        for condition in ("clean", "light"):
            for name in (
                "spatial_entropy",
                "component_count",
                "component_scale",
            ):
                result[f"{name}_{condition}"] = np.full(
                    (frame_count, channel_count), np.nan, dtype=np.float64
                )
    return result


def fill_condition_metrics(
    storage: dict[str, np.ndarray],
    frame_index: int,
    condition: str,
    feature: np.ndarray,
    include_region_metrics: bool,
) -> None:
    storage[f"activation_variance_{condition}"][frame_index] = np.var(
        feature, axis=(1, 2), dtype=np.float64
    )
    storage[f"active_ratio_{condition}"][frame_index] = np.mean(
        feature > 0.0, axis=(1, 2)
    )
    energy, coherence, angle = spatial_gradient_metrics(feature)
    storage[f"gradient_energy_{condition}"][frame_index] = energy
    storage[f"orientation_coherence_{condition}"][frame_index] = coherence
    storage[f"dominant_angle_{condition}"][frame_index] = angle
    storage[f"spectral_centroid_{condition}"][frame_index] = spectral_centroid(
        feature
    )
    if include_region_metrics:
        storage[f"spatial_entropy_{condition}"][frame_index] = spatial_entropy(
            feature
        )
        count, scale = component_metrics(feature)
        storage[f"component_count_{condition}"][frame_index] = count
        storage[f"component_scale_{condition}"][frame_index] = scale


def safe_nanmedian(values: np.ndarray, axis: int = 0) -> np.ndarray:
    if axis != 0 or values.ndim != 2:
        raise ValueError("safe_nanmedian currently expects a 2-D array, axis=0")
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column in range(values.shape[1]):
        finite = values[:, column][np.isfinite(values[:, column])]
        if finite.size:
            result[column] = float(np.median(finite))
    return result


def safe_iqr(values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column in range(values.shape[1]):
        finite = values[:, column][np.isfinite(values[:, column])]
        if finite.size:
            q25, q75 = np.percentile(finite, [25, 75])
            result[column] = float(q75 - q25)
    return result


def aggregate_frame_metrics(frame_metrics: dict[str, np.ndarray]) -> dict:
    aggregated: dict[str, np.ndarray] = {}
    for name, values in frame_metrics.items():
        aggregated[name] = safe_nanmedian(values, axis=0)
        aggregated[f"{name}_iqr"] = safe_iqr(values)
    aggregated["gradient_energy"] = safe_nanmedian(
        np.stack(
            (
                aggregated["gradient_energy_clean"],
                aggregated["gradient_energy_light"],
            )
        ),
        axis=0,
    )
    aggregated["orientation_coherence"] = safe_nanmedian(
        np.stack(
            (
                aggregated["orientation_coherence_clean"],
                aggregated["orientation_coherence_light"],
            )
        ),
        axis=0,
    )
    aggregated["spectral_centroid"] = safe_nanmedian(
        np.stack(
            (
                aggregated["spectral_centroid_clean"],
                aggregated["spectral_centroid_light"],
            )
        ),
        axis=0,
    )
    return aggregated


def pearson_safe(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float64) - np.mean(first)
    second = second.astype(np.float64) - np.mean(second)
    denominator = np.sqrt(np.sum(first**2) * np.sum(second**2))
    return float(np.sum(first * second) / denominator) if denominator > EPS else 0.0


def conv1_weight_metrics() -> dict[str, np.ndarray]:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    weights = model.conv1.weight.detach().cpu().numpy().astype(np.float32)
    channel_count = weights.shape[0]
    result = {
        name: np.zeros(channel_count, dtype=np.float64)
        for name in (
            "weight_corr_rg",
            "weight_corr_rb",
            "weight_corr_gb",
            "weight_corr_mean",
            "weight_corr_min",
            "weight_energy_r",
            "weight_energy_g",
            "weight_energy_b",
            "weight_pca_pc1_r",
            "weight_pca_pc1_g",
            "weight_pca_pc1_b",
            "weight_pca_explained_ratio",
            "weight_orientation_coherence",
            "weight_dominant_angle",
            "weight_spectral_centroid",
            "weight_dc_power_ratio",
        )
    }
    for channel in range(channel_count):
        kernel = weights[channel]
        correlations = (
            pearson_safe(kernel[0].ravel(), kernel[1].ravel()),
            pearson_safe(kernel[0].ravel(), kernel[2].ravel()),
            pearson_safe(kernel[1].ravel(), kernel[2].ravel()),
        )
        result["weight_corr_rg"][channel] = correlations[0]
        result["weight_corr_rb"][channel] = correlations[1]
        result["weight_corr_gb"][channel] = correlations[2]
        result["weight_corr_mean"][channel] = np.mean(correlations)
        result["weight_corr_min"][channel] = np.min(correlations)

        energy = np.sum(kernel.astype(np.float64) ** 2, axis=(1, 2))
        energy_share = energy / (energy.sum() + EPS)
        for colour, value in zip(("r", "g", "b"), energy_share):
            result[f"weight_energy_{colour}"][channel] = value

        colour_vectors = kernel.reshape(3, -1).astype(np.float64)
        colour_vectors -= colour_vectors.mean(axis=1, keepdims=True)
        covariance = colour_vectors @ colour_vectors.T
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        pc1 = eigenvectors[:, order[0]]
        if pc1[np.argmax(np.abs(pc1))] < 0:
            pc1 = -pc1
        result["weight_pca_pc1_r"][channel] = pc1[0]
        result["weight_pca_pc1_g"][channel] = pc1[1]
        result["weight_pca_pc1_b"][channel] = pc1[2]
        result["weight_pca_explained_ratio"][channel] = (
            eigenvalues[order[0]] / (np.maximum(eigenvalues, 0).sum() + EPS)
        )

        gy, gx = np.gradient(kernel, axis=(1, 2))
        gx2 = float(np.sum(gx * gx))
        gy2 = float(np.sum(gy * gy))
        gxy = float(np.sum(gx * gy))
        result["weight_orientation_coherence"][channel] = np.sqrt(
            (gx2 - gy2) ** 2 + 4.0 * gxy**2
        ) / (gx2 + gy2 + EPS)
        result["weight_dominant_angle"][channel] = np.mod(
            0.5 * np.degrees(np.arctan2(2.0 * gxy, gx2 - gy2)), 180.0
        )

        spectrum = np.fft.fftshift(
            np.fft.fft2(kernel, s=(32, 32), axes=(-2, -1)), axes=(-2, -1)
        )
        power = np.sum(np.abs(spectrum) ** 2, axis=0)
        frequencies = np.fft.fftshift(np.fft.fftfreq(32))
        radius = np.sqrt(
            frequencies[:, None] ** 2 + frequencies[None, :] ** 2
        )
        radius /= np.sqrt(0.5**2 + 0.5**2)
        total_power = float(power.sum())
        result["weight_spectral_centroid"][channel] = float(
            np.sum(power * radius) / (total_power + EPS)
        )
        result["weight_dc_power_ratio"][channel] = float(
            power[16, 16] / (total_power + EPS)
        )
    return result


def dead_candidate_analysis(
    frame_metrics: dict[str, np.ndarray],
    repeats: int,
    subset_size: int,
    median_fraction: float,
    stability_threshold: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    variance_frame = np.maximum(
        frame_metrics["activation_variance_clean"],
        frame_metrics["activation_variance_light"],
    )
    gradient_frame = np.maximum(
        frame_metrics["gradient_energy_clean"],
        frame_metrics["gradient_energy_light"],
    )
    variance_aggregate = np.median(variance_frame, axis=0)
    gradient_aggregate = np.median(gradient_frame, axis=0)
    variance_threshold = median_fraction * float(
        np.median(variance_aggregate)
    )
    gradient_threshold = median_fraction * float(
        np.median(gradient_aggregate)
    )
    base_rule = (variance_aggregate < variance_threshold) & (
        gradient_aggregate < gradient_threshold
    )

    frame_count = variance_frame.shape[0]
    subset_size = min(subset_size, frame_count)
    selections = np.zeros((repeats, variance_frame.shape[1]), dtype=bool)
    for repeat in range(repeats):
        subset = rng.choice(frame_count, size=subset_size, replace=False)
        subset_variance = np.median(variance_frame[subset], axis=0)
        subset_gradient = np.median(gradient_frame[subset], axis=0)
        repeat_variance_threshold = median_fraction * float(
            np.median(subset_variance)
        )
        repeat_gradient_threshold = median_fraction * float(
            np.median(subset_gradient)
        )
        selections[repeat] = (
            subset_variance < repeat_variance_threshold
        ) & (subset_gradient < repeat_gradient_threshold)
    stability = selections.mean(axis=0)
    candidate = base_rule & (stability >= stability_threshold)
    return {
        "variance_aggregate": variance_aggregate,
        "gradient_aggregate": gradient_aggregate,
        "variance_threshold": variance_threshold,
        "gradient_threshold": gradient_threshold,
        "base_rule": base_rule,
        "bootstrap_stability": stability,
        "confirmed_dead_candidate": candidate,
    }


def quantile_finite(values: np.ndarray, quantile: float) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, quantile)) if finite.size else 0.0


def add_soft_labels(
    layer: str, descriptors: dict[str, np.ndarray]
) -> dict[str, list[str]]:
    ncc = descriptors["cross_light_ncc"]
    relative_mad = descriptors["relative_mad"]
    ncc_low, ncc_high = (
        quantile_finite(ncc, 1 / 3),
        quantile_finite(ncc, 2 / 3),
    )
    mad_low, mad_high = (
        quantile_finite(relative_mad, 1 / 3),
        quantile_finite(relative_mad, 2 / 3),
    )
    spectral = descriptors["spectral_centroid"]
    spectral_high = quantile_finite(spectral, 2 / 3)

    labels = {
        "density_label": [],
        "structure_label": [],
        "illumination_label": [],
        "colour_label": [],
        "frequency_label": [],
        "soft_label": [],
    }
    for channel in range(len(ncc)):
        active = descriptors["active_ratio_clean"][channel]
        if active < 0.01:
            density = "Ultra-sparse"
        elif active < 0.05:
            density = "Sparse-localized"
        elif active > 0.50:
            density = "Dense-global"
        else:
            density = "Moderate-region"

        coherence = descriptors["orientation_coherence"][channel]
        if coherence >= 0.55:
            structure = "Oriented-edge"
        elif layer != "conv1" and descriptors[
            "component_scale_clean"
        ][channel] >= quantile_finite(
            descriptors["component_scale_clean"], 2 / 3
        ):
            structure = "Region-blob"
        elif spectral[channel] >= spectral_high:
            structure = "Texture/high-frequency"
        else:
            structure = "Mixed/isotropic"

        if (
            np.isfinite(ncc[channel])
            and ncc[channel] >= ncc_high
            and relative_mad[channel] <= mad_low
        ):
            illumination = "Stable"
        elif (
            not np.isfinite(ncc[channel])
            or ncc[channel] <= ncc_low
            or relative_mad[channel] >= mad_high
        ):
            illumination = "Sensitive"
        else:
            illumination = "Intermediate"

        if layer == "conv1":
            mean_corr = descriptors["weight_corr_mean"][channel]
            min_corr = descriptors["weight_corr_min"][channel]
            max_energy = max(
                descriptors["weight_energy_r"][channel],
                descriptors["weight_energy_g"][channel],
                descriptors["weight_energy_b"][channel],
            )
            if mean_corr >= 0.50:
                colour = "Achromatic"
            elif min_corr <= -0.25:
                colour = "Opponent"
            elif max_energy >= 0.60:
                colour = "Colour-selective"
            else:
                colour = "Mixed-colour"
            weight_spectral = descriptors["weight_spectral_centroid"][channel]
            if descriptors["weight_dc_power_ratio"][channel] >= 0.25:
                frequency = "Low-frequency/DC"
            elif weight_spectral >= quantile_finite(
                descriptors["weight_spectral_centroid"], 2 / 3
            ):
                frequency = "High-frequency"
            else:
                frequency = "Mid-frequency"
        else:
            colour = "N/A"
            if spectral[channel] >= spectral_high:
                frequency = "High-frequency"
            elif spectral[channel] <= quantile_finite(spectral, 1 / 3):
                frequency = "Low-frequency"
            else:
                frequency = "Mid-frequency"

        labels["density_label"].append(density)
        labels["structure_label"].append(structure)
        labels["illumination_label"].append(illumination)
        labels["colour_label"].append(colour)
        labels["frequency_label"].append(frequency)
        labels["soft_label"].append(
            " + ".join(
                value
                for value in (
                    colour if layer == "conv1" else density,
                    structure,
                    frequency,
                    illumination,
                )
                if value != "N/A"
            )
        )
    return labels


def descriptor_rows(
    layer: str,
    descriptors: dict[str, np.ndarray],
    labels: dict[str, list[str]],
    dead: dict[str, Any],
) -> list[dict[str, object]]:
    channel_count = CHANNEL_COUNTS[layer]
    rows: list[dict[str, object]] = []
    for channel in range(channel_count):
        row: dict[str, object] = {"channel": channel}
        for name, values in descriptors.items():
            value = values[channel]
            row[name] = float(value) if np.isscalar(value) else value
        for name, values in labels.items():
            row[name] = values[channel]
        row["dead_variance_max_condition"] = float(
            dead["variance_aggregate"][channel]
        )
        row["dead_gradient_max_condition"] = float(
            dead["gradient_aggregate"][channel]
        )
        row["dead_base_rule"] = bool(dead["base_rule"][channel])
        row["dead_bootstrap_stability"] = float(
            dead["bootstrap_stability"][channel]
        )
        row["confirmed_dead_candidate"] = bool(
            dead["confirmed_dead_candidate"][channel]
        )
        row["manual_review_status"] = (
            "pending" if row["confirmed_dead_candidate"] else "not_required"
        )
        rows.append(row)
    return rows


def write_profile_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def robust_zscore_matrix(matrix: np.ndarray) -> np.ndarray:
    median = np.nanmedian(matrix, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(matrix - median), axis=0, keepdims=True)
    scale = 1.4826 * mad
    fallback = np.nanstd(matrix, axis=0, keepdims=True)
    scale = np.where(scale > EPS, scale, fallback)
    scale = np.where(scale > EPS, scale, 1.0)
    zscore = (matrix - median) / scale
    zscore = np.where(np.isfinite(zscore), zscore, 0.0)
    return np.clip(zscore, -3.0, 3.0)


def plot_descriptor_heatmap(
    layer: str,
    descriptors: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    metrics = [
        "activation_variance_clean",
        "activation_variance_light",
        "active_ratio_clean",
        "active_ratio_light",
        "gradient_energy_clean",
        "gradient_energy_light",
        "orientation_coherence",
        "spectral_centroid",
        "cross_light_ncc",
        "relative_mad",
        "mean_shift_relative",
        "gradient_retention",
    ]
    if layer != "conv1":
        metrics[6:6] = [
            "spatial_entropy_clean",
            "component_count_clean",
            "component_scale_clean",
        ]
    else:
        metrics.extend(
            [
                "weight_corr_mean",
                "weight_corr_min",
                "weight_orientation_coherence",
                "weight_spectral_centroid",
                "weight_dc_power_ratio",
            ]
        )
    matrix = np.column_stack([descriptors[name] for name in metrics])
    zscore = robust_zscore_matrix(matrix)
    height = 10 if layer != "layer2" else 16
    tick_step = 4 if CHANNEL_COUNTS[layer] == 64 else 8
    channel_labels = [
        str(channel) if channel % tick_step == 0 else ""
        for channel in range(CHANNEL_COUNTS[layer])
    ]
    fig, ax = plt.subplots(figsize=(15, height))
    sns.heatmap(
        zscore,
        cmap="vlag",
        vmin=-3,
        vmax=3,
        center=0,
        xticklabels=metrics,
        yticklabels=channel_labels,
        cbar_kws={"label": "Robust z-score (clipped ±3)"},
        ax=ax,
    )
    ax.set_title(
        f"{layer} functional descriptors · post-ReLU · 30 paired timestamps"
    )
    ax.set_xlabel("Metric")
    ax.set_ylabel("Channel (0-based)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def normalise_map(feature_map: np.ndarray) -> np.ndarray:
    high = float(np.percentile(feature_map, 99))
    if high <= EPS:
        return np.zeros_like(feature_map, dtype=np.float32)
    return np.clip(feature_map / high, 0.0, 1.0)


def plot_dead_review(
    layer: str,
    candidates: list[int],
    clean_means: dict[int, np.ndarray],
    light_means: dict[int, np.ndarray],
    output_path: Path,
) -> None:
    if not candidates:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.text(
            0.5,
            0.5,
            f"{layer}: no confirmed-dead candidates under the current rule",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    fig, axes = plt.subplots(
        len(candidates),
        3,
        figsize=(10, max(3, 2.8 * len(candidates))),
        squeeze=False,
    )
    for row, channel in enumerate(candidates):
        clean = clean_means[channel]
        light = light_means[channel]
        difference = np.abs(light - clean)
        for column, (title, fmap) in enumerate(
            (("clean mean", clean), ("light mean", light), ("abs diff", difference))
        ):
            axes[row, column].imshow(normalise_map(fmap), cmap="viridis")
            axes[row, column].set_title(f"ch {channel:03d} · {title}")
            axes[row, column].axis("off")
    fig.suptitle(
        f"{layer} confirmed-dead candidates · manual review required",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def accumulate_candidate_means(
    manifest: list[dict[str, str]],
    candidates_by_layer: dict[str, list[int]],
) -> dict[str, dict[str, dict[int, np.ndarray]]]:
    accumulators: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for layer, candidates in candidates_by_layer.items():
        accumulators[layer] = {
            "clean": {channel: None for channel in candidates},
            "light": {channel: None for channel in candidates},
        }
    if not any(candidates_by_layer.values()):
        return accumulators

    for row in manifest:
        with np.load(row["feature_file"]) as archive:
            for layer, candidates in candidates_by_layer.items():
                for condition in ("clean", "light"):
                    feature = archive[f"{layer}_{condition}"]
                    for channel in candidates:
                        fmap = feature[channel].astype(np.float64)
                        current = accumulators[layer][condition][channel]
                        accumulators[layer][condition][channel] = (
                            fmap.copy() if current is None else current + fmap
                        )
    count = len(manifest)
    for layer, candidates in candidates_by_layer.items():
        for condition in ("clean", "light"):
            for channel in candidates:
                accumulators[layer][condition][channel] /= count
    return accumulators


def main() -> None:
    args = parse_args()
    if args.bootstrap_repeats < 1 or args.bootstrap_frames < 1:
        raise ValueError("Bootstrap counts must be positive")
    if not 0 < args.dead_median_fraction < 1:
        raise ValueError("--dead-median-fraction must lie in (0,1)")
    if not 0 <= args.dead_stability_threshold <= 1:
        raise ValueError("--dead-stability-threshold must lie in [0,1]")

    manifest = read_manifest(args.manifest.resolve())
    output_dir = prepare_output_directory(args.output_dir, args.overwrite)
    frame_count = len(manifest)
    rng = np.random.default_rng(args.seed)

    frame_metrics = {
        layer: initialise_frame_metrics(
            frame_count,
            CHANNEL_COUNTS[layer],
            include_region_metrics=(layer != "conv1"),
        )
        for layer in LAYERS
    }
    print(f"[Input] {frame_count} paired feature archives")
    print("[Mode] post-ReLU, native resolution, robust median aggregation")

    for frame_index, row in enumerate(manifest):
        with np.load(row["feature_file"]) as archive:
            for layer in LAYERS:
                clean = archive[f"{layer}_clean"].astype(np.float32, copy=False)
                light = archive[f"{layer}_light"].astype(np.float32, copy=False)
                include_region = layer != "conv1"
                fill_condition_metrics(
                    frame_metrics[layer],
                    frame_index,
                    "clean",
                    clean,
                    include_region,
                )
                fill_condition_metrics(
                    frame_metrics[layer],
                    frame_index,
                    "light",
                    light,
                    include_region,
                )
                paired = paired_metrics(clean, light)
                for name, values in paired.items():
                    frame_metrics[layer][name][frame_index] = values
                clean_energy = frame_metrics[layer][
                    "gradient_energy_clean"
                ][frame_index]
                light_energy = frame_metrics[layer][
                    "gradient_energy_light"
                ][frame_index]
                frame_metrics[layer]["gradient_retention"][frame_index] = (
                    light_energy / (clean_energy + EPS)
                )
        print(f"[Profile] {frame_index + 1:02d}/{frame_count:02d}")

    weight_descriptors = conv1_weight_metrics()
    all_descriptors: dict[str, dict[str, np.ndarray]] = {}
    dead_results: dict[str, dict[str, Any]] = {}
    profiles: dict[str, list[dict[str, object]]] = {}
    candidates_by_layer: dict[str, list[int]] = {}

    for layer in LAYERS:
        descriptors = aggregate_frame_metrics(frame_metrics[layer])
        if layer == "conv1":
            descriptors.update(weight_descriptors)
        dead = dead_candidate_analysis(
            frame_metrics[layer],
            repeats=args.bootstrap_repeats,
            subset_size=args.bootstrap_frames,
            median_fraction=args.dead_median_fraction,
            stability_threshold=args.dead_stability_threshold,
            rng=rng,
        )
        labels = add_soft_labels(layer, descriptors)
        rows = descriptor_rows(layer, descriptors, labels, dead)
        profile_path = output_dir / f"functional_profile_{layer}.csv"
        write_profile_csv(rows, profile_path)
        plot_descriptor_heatmap(
            layer,
            descriptors,
            output_dir / "heatmaps" / f"functional_descriptor_{layer}.png",
        )
        candidates = np.flatnonzero(
            dead["confirmed_dead_candidate"]
        ).tolist()
        candidates_by_layer[layer] = candidates
        all_descriptors[layer] = descriptors
        dead_results[layer] = dead
        profiles[layer] = rows
        print(
            f"[Layer] {layer}: {CHANNEL_COUNTS[layer]} channels, "
            f"{len(candidates)} confirmed-dead candidates"
        )

    review_means = accumulate_candidate_means(manifest, candidates_by_layer)
    for layer in LAYERS:
        plot_dead_review(
            layer,
            candidates_by_layer[layer],
            review_means[layer]["clean"],
            review_means[layer]["light"],
            output_dir
            / "dead_channel_review"
            / f"dead_candidate_review_{layer}.png",
        )

    dead_payload = {
        "status": (
            "candidate_only; manual review required before channel exclusion"
        ),
        "rule": {
            "per_frame_condition_combination": (
                "max(clean, light) for variance and gradient energy"
            ),
            "temporal_aggregation": "median across paired timestamps",
            "layer_threshold": (
                f"{args.dead_median_fraction} × layer median, both metrics"
            ),
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_subset_frames": min(args.bootstrap_frames, frame_count),
            "minimum_stability": args.dead_stability_threshold,
            "seed": args.seed,
        },
        "layers": {},
    }
    for layer in LAYERS:
        dead = dead_results[layer]
        dead_payload["layers"][layer] = {
            "variance_threshold": dead["variance_threshold"],
            "gradient_energy_threshold": dead["gradient_threshold"],
            "base_rule_channels": np.flatnonzero(dead["base_rule"]).tolist(),
            "confirmed_dead_candidates": candidates_by_layer[layer],
            "candidate_bootstrap_stability": {
                str(channel): float(dead["bootstrap_stability"][channel])
                for channel in candidates_by_layer[layer]
            },
        }
    (output_dir / "confirmed_dead_candidates.json").write_text(
        json.dumps(dead_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = {
        "step": "A - Functional Profiling",
        "principle": (
            "Descriptors and labels are explanatory only; no active channel "
            "is removed or used as a hard functional constraint."
        ),
        "input": {
            "manifest": str(args.manifest.resolve()),
            "paired_timestamps": frame_count,
            "activation_position": "post-ReLU",
            "native_resolution": True,
        },
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "opencv": cv2.__version__,
        },
        "aggregation": {
            "primary": "median across 30 frames",
            "dispersion": "interquartile range",
        },
        "metric_definitions": {
            "gradient_energy": "mean(gx^2 + gy^2), numpy spatial gradient",
            "orientation_coherence": (
                "global structure-tensor coherence in [0,1]"
            ),
            "spectral_centroid": (
                "zero-mean radial FFT power centroid / Nyquist-corner radius"
            ),
            "spatial_entropy": (
                "entropy of non-negative activation mass / log(HW)"
            ),
            "components": (
                "8-connected activation > positive mean + positive std; "
                "minimum component area=max(2 pixels, 0.05% map)"
            ),
            "cross_light_ncc": "Pearson NCC of paired native feature maps",
            "relative_mad": "mean(abs(light-clean)) / mean(abs(clean))",
            "gradient_retention": (
                "light gradient energy / clean gradient energy"
            ),
        },
        "outputs": {
            layer: {
                "profile_csv": str(
                    output_dir / f"functional_profile_{layer}.csv"
                ),
                "channels": CHANNEL_COUNTS[layer],
                "confirmed_dead_candidates": candidates_by_layer[layer],
            }
            for layer in LAYERS
        },
    }
    (output_dir / "functional_profiling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Output] {output_dir}")
    print("[Done] Step A functional profiling complete")


if __name__ == "__main__":
    main()
