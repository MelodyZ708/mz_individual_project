#!/usr/bin/env python3
"""Visualize selected U-Net Enc0/Enc1 tracking feature maps.

This intentionally follows ``UNetFeatureExtractor.extract`` in COMO exactly:
input RGB is ImageNet-normalized, Enc0 is ``unet.base`` and Enc1 is
``unet.down_convs[0](Enc0)``.  It never changes COMO's shared YAML config.

The three default frames are the before / peak / after samples of the MVS
turn-on challenge (indices 246, 250, 254).  It writes both a temporal tracking
view and a ResNet-style ``Clean | Lightswitch | |Light-clean|`` view.

The U-Net's ResidualConv ends in LeakyReLU, rather than standard ReLU.  The
Clean/Lightswitch view therefore displays the *actual post-LeakyReLU tracking
activation* (no additional ReLU clipping is applied).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep Matplotlib's transient cache out of the user's home directory before it
# is imported.  This does not affect the generated experiment artefacts.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_unet_feature_maps")

import cv2
import matplotlib
import numpy as np
import torch


# __file__ is inside ``channel_selection_pipeline/scripts/step_l...``.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMO_DIR = PROJECT_ROOT / "como"
DEFAULT_DATASET = Path("/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch")
DEFAULT_CLEAN_DATASET = Path("/home/melody/data/tum/rgbd_dataset_freiburg1_desk")
DEFAULT_CHECKPOINT = COMO_DIR / "models/scannet.ckpt"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "channel_selection_results/step_l_unet_feature_map_visualization"
)

# These are the requested candidates, grouped by the feature level used by
# tracking.  Labels are kept stable so result paths can be cited in reports.
REQUESTED_GROUPS = {
    "enc0": {
        "channels": 16,
        "groups": {
            "enc0_ch_03": (3,),
            "enc0_ch_02_14": (2, 14),
            "enc0_ch_03_07_12": (3, 7, 12),
            "enc0_ch_02_03_07_12_13_14": (2, 3, 7, 12, 13, 14),
        },
    },
    "enc1": {
        "channels": 32,
        "groups": {
            "enc1_ch_00_05": (0, 5),
            "enc1_ch_05_06_17_18_28_30": (5, 6, 17, 18, 28, 30),
        },
    },
}

FRAME_SPECS = (
    (246, "before"),
    (250, "peak"),
    (254, "after"),
)


def parse_matches(dataset: Path) -> list[tuple[str, Path]]:
    matches: list[tuple[str, Path]] = []
    for line in (dataset / "matched_rgb.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        timestamp, relative = line.split(maxsplit=1)
        path = dataset / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing RGB frame referenced by matched_rgb.txt: {path}")
        matches.append((timestamp, path))
    return matches


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_encoder(checkpoint: Path):
    sys.path.insert(0, str(COMO_DIR))
    from como.depth_cov.core.DepthCovModule import DepthCovModule

    # This is the same checkpoint loading call used by Mapping.load_model(),
    # but visualisation uses CPU to avoid perturbing an active tracking job.
    module = DepthCovModule.load_from_checkpoint(
        str(checkpoint), train_size=torch.tensor([192, 256]), map_location="cpu"
    )
    module.eval().to("cpu", dtype=torch.float32)
    return module.gaussian_cov_net


def extract_native_features(unet, rgb: np.ndarray) -> dict[str, np.ndarray]:
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        enc0 = unet.base(unet.normalize(tensor))
        enc1 = unet.down_convs[0](enc0)
    return {
        "enc0": enc0.squeeze(0).cpu().numpy().astype(np.float32, copy=False),
        "enc1": enc1.squeeze(0).cpu().numpy().astype(np.float32, copy=False),
    }


def robust_limits(maps: list[np.ndarray]) -> tuple[float, float]:
    """Return a signed, shared colour range for one channel across frames."""

    values = np.concatenate([feature.reshape(-1) for feature in maps])
    low, high = np.percentile(values, [1.0, 99.0])
    bound = max(abs(float(low)), abs(float(high)), 1e-8)
    return -bound, bound


def save_rgb_strip(rgb_frames: list[np.ndarray], labels: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(rgb_frames), figsize=(4.7 * len(rgb_frames), 3.8))
    for axis, image, label in zip(axes, rgb_frames, labels):
        axis.imshow(image)
        axis.set_title(label, fontsize=11)
        axis.axis("off")
    fig.suptitle("Input RGB frames: MVS turn-on challenge", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_channel_strip(
    maps: list[np.ndarray], labels: list[str], level: str, channel: int, path: Path
) -> tuple[float, float]:
    import matplotlib.pyplot as plt

    vmin, vmax = robust_limits(maps)
    fig, axes = plt.subplots(
        1,
        len(maps) + 1,
        figsize=(4.7 * len(maps) + 0.7, 3.7),
        gridspec_kw={"width_ratios": [1] * len(maps) + [0.055]},
        constrained_layout=True,
    )
    image_axes = axes[: len(maps)]
    colorbar_axis = axes[-1]
    image_artist = None
    for axis, feature, label in zip(image_axes, maps, labels):
        image_artist = axis.imshow(
            feature, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="none"
        )
        axis.set_title(label, fontsize=10)
        axis.axis("off")
    fig.colorbar(image_artist, cax=colorbar_axis, label="activation")
    fig.suptitle(
        f"{level.upper()} d{channel:02d}: shared robust scale [{vmin:.3g}, {vmax:.3g}]",
        fontsize=12,
        fontweight="bold",
    )
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return vmin, vmax


def save_group_montage(
    rgb_frames: list[np.ndarray],
    labels: list[str],
    level: str,
    group_name: str,
    channels: tuple[int, ...],
    feature_frames: list[np.ndarray],
    path: Path,
) -> dict[str, list[float]]:
    """Rows are input RGB then requested channels; columns are timepoints."""

    import matplotlib.pyplot as plt

    rows = len(channels) + 1
    fig, axes = plt.subplots(
        rows,
        len(rgb_frames) + 1,
        figsize=(4.35 * len(rgb_frames) + 0.75, 2.85 * rows),
        gridspec_kw={"width_ratios": [1] * len(rgb_frames) + [0.055]},
        constrained_layout=True,
    )
    if rows == 1:
        axes = np.asarray([axes])
    display_ranges: dict[str, list[float]] = {}

    for column, (axis, image, label) in enumerate(zip(axes[0, : len(rgb_frames)], rgb_frames, labels)):
        axis.imshow(image)
        axis.set_title(label, fontsize=11)
        axis.axis("off")
        if column == 0:
            axis.set_ylabel("input RGB", fontsize=11, fontweight="bold")
    axes[0, -1].axis("off")

    for row, channel in enumerate(channels, start=1):
        maps = [features[channel] for features in feature_frames]
        vmin, vmax = robust_limits(maps)
        display_ranges[f"d{channel}"] = [vmin, vmax]
        image_artist = None
        for column, (axis, feature) in enumerate(zip(axes[row, : len(rgb_frames)], maps)):
            image_artist = axis.imshow(
                feature, cmap="coolwarm", vmin=vmin, vmax=vmax, interpolation="none"
            )
            axis.axis("off")
            if column == 0:
                native_shape = f"{feature.shape[0]}×{feature.shape[1]}"
                axis.set_ylabel(f"{level} d{channel}\n({native_shape})", fontsize=10, fontweight="bold")
        cbar = fig.colorbar(image_artist, cax=axes[row, -1])
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"{level.upper()} {group_name}: selected tracking feature maps\n"
        "Each channel uses one shared 1st--99th percentile signed scale across the three frames.",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return display_ranges


def clean_light_limits(clean: np.ndarray, light: np.ndarray) -> tuple[float, float]:
    """One robust linear scale shared by a clean/lightswitch channel pair."""

    values = np.concatenate((clean.reshape(-1), light.reshape(-1)))
    low, high = np.percentile(values, [1.0, 99.0])
    if float(high - low) < 1e-8:
        midpoint = float((high + low) / 2.0)
        return midpoint - 1e-6, midpoint + 1e-6
    return float(low), float(high)


def difference_limit(difference: np.ndarray) -> float:
    return max(float(np.percentile(difference, 99.0)), 1e-8)


def save_clean_light_details(
    level: str,
    group_name: str,
    channels: tuple[int, ...],
    record: dict[str, object],
    clean_features: np.ndarray,
    light_features: np.ndarray,
    path: Path,
) -> dict[str, dict[str, float]]:
    """Write the same Clean | Light | absolute-difference layout as the ResNet plots."""

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(channels), 3, figsize=(11.2, 3.0 * len(channels) + 0.85)
    )
    if len(channels) == 1:
        axes = np.asarray([axes])
    ranges: dict[str, dict[str, float]] = {}

    for row, channel in enumerate(channels):
        clean = clean_features[channel]
        light = light_features[channel]
        difference = np.abs(light - clean)
        low, high = clean_light_limits(clean, light)
        diff_high = difference_limit(difference)
        ranges[f"d{channel}"] = {
            "clean_lightswitch_low": low,
            "clean_lightswitch_high": high,
            "absolute_difference_high": diff_high,
        }
        for column, (name, feature, cmap, vmin, vmax) in enumerate(
            (
                ("Clean", clean, "viridis", low, high),
                ("Lightswitch", light, "viridis", low, high),
                ("|Light − clean|", difference, "magma", 0.0, diff_high),
            )
        ):
            axis = axes[row, column]
            axis.imshow(feature, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(name, fontsize=11)
            if column == 0:
                axis.set_ylabel(
                    f"Enc{level[-1]} d{channel}\n"
                    f"{feature.shape[0]}×{feature.shape[1]}",
                    fontsize=10,
                )

    figure.suptitle(
        f"{level.upper()} {group_name} · frame {record['matched_rgb_index']} · {record['phase']}\n"
        "Actual post-LeakyReLU tracking activations; clean/lightswitch share a per-channel scale",
        fontsize=12.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return ranges


def save_clean_lightswitch_inputs(
    clean_frames: list[np.ndarray],
    light_frames: list[np.ndarray],
    records: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(records), 2, figsize=(9.4, 3.35 * len(records)))
    for row, (clean, light, record) in enumerate(zip(clean_frames, light_frames, records)):
        for column, (name, image) in enumerate((("Clean", clean), ("Lightswitch", light))):
            axis = axes[row, column]
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(
                f"frame {record['matched_rgb_index']} · {record['phase']} · {name}",
                fontsize=10.5,
            )
    figure.suptitle("Matched Clean / Lightswitch input pairs around the MVS turn-on event", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Lightswitch TUM dataset used by the U-Net tracking search.",
    )
    parser.add_argument(
        "--clean-dataset",
        type=Path,
        default=DEFAULT_CLEAN_DATASET,
        help="Matched clean TUM dataset for Clean/Lightswitch comparison panels.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    matplotlib.use("Agg")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    matches = parse_matches(args.dataset)
    clean_by_timestamp = dict(parse_matches(args.clean_dataset))
    requested_indices = [index for index, _ in FRAME_SPECS]
    if max(requested_indices) >= len(matches):
        raise ValueError(f"Requested frame is beyond matched_rgb length {len(matches)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_records: list[dict[str, object]] = []
    rgb_frames: list[np.ndarray] = []
    clean_rgb_frames: list[np.ndarray] = []
    for index, phase in FRAME_SPECS:
        timestamp, lightswitch_path = matches[index]
        clean_path = clean_by_timestamp.get(timestamp)
        if clean_path is None:
            raise KeyError(
                f"Clean dataset has no exact matched RGB timestamp for lightswitch frame {index}: {timestamp}"
            )
        rgb = read_rgb(lightswitch_path)
        clean_rgb = read_rgb(clean_path)
        if rgb.shape != clean_rgb.shape:
            raise ValueError(
                f"Clean/lightswitch RGB shapes differ at frame {index}: {clean_rgb.shape} vs {rgb.shape}"
            )
        rgb_frames.append(rgb)
        clean_rgb_frames.append(clean_rgb)
        frame_records.append(
            {
                "matched_rgb_index": index,
                "timestamp": timestamp,
                "phase": phase,
                "clean_rgb_path": str(clean_path),
                "lightswitch_rgb_path": str(lightswitch_path),
                "rgb_shape_hwc": list(rgb.shape),
            }
        )

    print("[LOAD] Loading COMO ScanNet U-Net checkpoint on CPU")
    unet = load_encoder(args.checkpoint)
    print("[EXTRACT] Extracting native-resolution Enc0 and Enc1 maps for matched clean/lightswitch pairs")
    extracted = [extract_native_features(unet, rgb) for rgb in rgb_frames]
    extracted_clean = [extract_native_features(unet, rgb) for rgb in clean_rgb_frames]

    labels = [
        f"frame {record['matched_rgb_index']} ({record['phase']})\n{record['timestamp']}"
        for record in frame_records
    ]
    common_dir = args.output_dir / "frames"
    common_dir.mkdir(exist_ok=True)
    save_rgb_strip(rgb_frames, labels, common_dir / "input_rgb_frames246_250_254.png")
    save_clean_lightswitch_inputs(
        clean_rgb_frames,
        rgb_frames,
        frame_records,
        common_dir / "clean_lightswitch_input_pairs_frames246_250_254.png",
    )

    manifest: dict[str, object] = {
        "lightswitch_dataset": str(args.dataset),
        "clean_dataset": str(args.clean_dataset),
        "checkpoint": str(args.checkpoint),
        "extraction": {
            "same_as_tracking": "normalize -> unet.base (Enc0) -> unet.down_convs[0] (Enc1)",
            "feature_resolution": {"enc0": "native full RGB resolution", "enc1": "native H/2 x W/2"},
            "tracking_activation": "ResidualConv output after LeakyReLU; no additional standard-ReLU clipping",
            "temporal_display": "per-channel shared signed 1st--99th percentile scale across lightswitch frames; coolwarm; interpolation=none",
            "clean_lightswitch_display": "per-channel clean/lightswitch shared p01-p99 linear scale (viridis); absolute difference p99 scale (magma); interpolation=none",
        },
        "frames": frame_records,
        "groups": {},
        "clean_lightswitch_details": {},
    }

    for level, level_spec in REQUESTED_GROUPS.items():
        level_dir = args.output_dir / level
        level_dir.mkdir(exist_ok=True)
        level_features = [frame_features[level] for frame_features in extracted]
        level_clean_features = [frame_features[level] for frame_features in extracted_clean]
        expected_channels = int(level_spec["channels"])
        if any(features.shape[0] != expected_channels for features in level_features):
            raise RuntimeError(f"Unexpected {level} channel count in extracted feature maps")

        for group_name, channels in level_spec["groups"].items():
            group_dir = level_dir / group_name
            channel_dir = group_dir / "per_channel"
            channel_dir.mkdir(parents=True, exist_ok=True)
            montage_path = group_dir / "feature_map_montage_frames246_250_254.png"
            ranges = save_group_montage(
                rgb_frames,
                labels,
                level,
                group_name,
                channels,
                level_features,
                montage_path,
            )
            channel_files: list[str] = []
            native_arrays: dict[str, np.ndarray] = {}
            for channel in channels:
                maps = [features[channel] for features in level_features]
                strip_path = channel_dir / f"{level}_d{channel:02d}_frames246_250_254.png"
                save_channel_strip(maps, labels, level, channel, strip_path)
                channel_files.append(str(strip_path.relative_to(args.output_dir)))
                for record, feature in zip(frame_records, maps):
                    native_arrays[
                        f"{level}_d{channel:02d}_frame{record['matched_rgb_index']:03d}"
                    ] = feature
            npz_path = group_dir / "native_feature_maps_frames246_250_254.npz"
            np.savez_compressed(npz_path, **native_arrays)

            manifest["groups"][group_name] = {
                "level": level,
                "channels": list(channels),
                "native_shape": list(level_features[0].shape[1:]),
                "montage_png": str(montage_path.relative_to(args.output_dir)),
                "per_channel_pngs": channel_files,
                "native_feature_maps_npz": str(npz_path.relative_to(args.output_dir)),
                "display_ranges": ranges,
            }
            print(f"[WRITE] {montage_path}")

            detail_dir = (
                args.output_dir / "clean_lightswitch_post_activation" / level / group_name
            )
            clean_light_arrays: dict[str, np.ndarray] = {}
            detail_paths: list[str] = []
            detail_ranges: dict[str, dict[str, dict[str, float]]] = {}
            for record, clean_feature, light_feature in zip(
                frame_records, level_clean_features, level_features
            ):
                frame_index = int(record["matched_rgb_index"])
                detail_path = detail_dir / f"frame_{frame_index:03d}_{record['phase']}.png"
                ranges_for_frame = save_clean_light_details(
                    level,
                    group_name,
                    channels,
                    record,
                    clean_feature,
                    light_feature,
                    detail_path,
                )
                detail_paths.append(str(detail_path.relative_to(args.output_dir)))
                detail_ranges[f"frame_{frame_index:03d}"] = ranges_for_frame
                for channel in channels:
                    clean_light_arrays[f"{level}_d{channel:02d}_clean_frame{frame_index:03d}"] = clean_feature[channel]
                    clean_light_arrays[f"{level}_d{channel:02d}_lightswitch_frame{frame_index:03d}"] = light_feature[channel]
            clean_light_npz = detail_dir / "native_clean_lightswitch_feature_maps_frames246_250_254.npz"
            np.savez_compressed(clean_light_npz, **clean_light_arrays)
            manifest["clean_lightswitch_details"][group_name] = {
                "level": level,
                "channels": list(channels),
                "detail_pngs": detail_paths,
                "native_feature_maps_npz": str(clean_light_npz.relative_to(args.output_dir)),
                "display_ranges": detail_ranges,
            }
            print(f"[WRITE] {detail_dir}")

    (args.output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "README.md").write_text(
        "# Selected U-Net tracking feature-map visualisations\n\n"
        "- Frames 246/250/254 are the before/peak/after samples of the selected MVS turn-on challenge.\n"
        "- `enc0/` contains [3], [2,14], [3,7,12], and [2,3,7,12,13,14].\n"
        "- `enc1/` contains [0,5] and [5,6,17,18,28,30].\n"
        "- `clean_lightswitch_post_activation/` is the primary ResNet-style view: one PNG per group per frame, with Clean / Lightswitch / |Light-clean| columns.\n"
        "- U-Net ResidualConv uses LeakyReLU. The maps are actual post-LeakyReLU tracking features, not standard-ReLU-clipped surrogates.\n"
        "- Clean/lightswitch share a scale within each channel; different channels retain independent scales.\n"
        "- `.npz` files preserve native maps for later quantitative analysis.\n",
        encoding="utf-8",
    )
    print(f"[DONE] Wrote visualisations to {args.output_dir}")


if __name__ == "__main__":
    main()
