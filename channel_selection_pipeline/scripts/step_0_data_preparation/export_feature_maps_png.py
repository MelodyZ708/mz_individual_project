#!/usr/bin/env python3
"""
Export every per-frame NPZ feature archive as readable PNG contact sheets.

Each NPZ contains six arrays. The exporter writes six PNG files per timestamp:

    conv1_clean.png, conv1_light.png
    layer1_clean.png, layer1_light.png
    layer2_clean.png, layer2_light.png

All channels are included. Conv1/layer1 use an 8x8 grid and layer2 uses a
16x8 grid. For a given timestamp, layer, and channel, clean/light use the same
1st--99th percentile display range so that their appearance is comparable.

The numerical NPZ files are deliberately retained: PNG is a visualisation,
not a lossless replacement for Step A/B computations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEATURE_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_0_data_preparation"
    / "features_post_relu"
)
DEFAULT_MANIFEST = DEFAULT_FEATURE_DIR / "feature_manifest.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_0_data_preparation"
    / "features_post_relu_png"
)
EXPECTED_KEYS = (
    "conv1_clean",
    "conv1_light",
    "layer1_clean",
    "layer1_light",
    "layer2_clean",
    "layer2_light",
)
GRID_COLUMNS = {"conv1": 8, "layer1": 8, "layer2": 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all feature NPZ archives into PNG contact sheets."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--tile-width",
        type=int,
        default=160,
        help="Width of each channel tile; height follows the feature aspect ratio.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace PNG files that already exist.",
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
        feature_file = Path(row["feature_file"])
        if not feature_file.is_file():
            raise FileNotFoundError(
                f"Manifest row {row_index} references a missing NPZ: "
                f"{feature_file}"
            )
    return rows


def shared_channel_ranges(
    clean: np.ndarray, light: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if clean.shape != light.shape:
        raise ValueError(
            f"Clean/light feature shapes differ: {clean.shape} vs {light.shape}"
        )
    channel_count = clean.shape[0]
    clean_flat = clean.reshape(channel_count, -1)
    light_flat = light.reshape(channel_count, -1)
    combined = np.concatenate((clean_flat, light_flat), axis=1)
    low = np.percentile(combined, 1.0, axis=1)
    high = np.percentile(combined, 99.0, axis=1)
    return low.astype(np.float32), high.astype(np.float32)


def colourise_channel(
    feature_map: np.ndarray,
    low: float,
    high: float,
    tile_width: int,
    channel: int,
) -> np.ndarray:
    if high - low < 1e-12:
        normalised = np.zeros_like(feature_map, dtype=np.uint8)
    else:
        normalised = np.clip(
            (feature_map.astype(np.float32) - low) / (high - low), 0.0, 1.0
        )
        normalised = np.rint(normalised * 255.0).astype(np.uint8)

    height, width = feature_map.shape
    tile_height = max(1, int(round(tile_width * height / width)))
    resized = cv2.resize(
        normalised,
        (tile_width, tile_height),
        interpolation=cv2.INTER_AREA,
    )
    colour = cv2.applyColorMap(resized, cv2.COLORMAP_VIRIDIS)

    label = f"ch {channel:03d}"
    cv2.rectangle(colour, (0, 0), (66, 22), (0, 0, 0), thickness=-1)
    cv2.putText(
        colour,
        label,
        (4, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    return colour


def make_contact_sheet(
    feature: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    columns: int,
    tile_width: int,
    title: str,
) -> np.ndarray:
    if feature.ndim != 3:
        raise ValueError(f"Expected [C,H,W], got shape {feature.shape}")
    channel_count = feature.shape[0]
    rows = int(np.ceil(channel_count / columns))
    tiles = [
        colourise_channel(
            feature[channel],
            float(low[channel]),
            float(high[channel]),
            tile_width,
            channel,
        )
        for channel in range(channel_count)
    ]
    tile_height = tiles[0].shape[0]
    gap = 3
    header_height = 62
    sheet_height = header_height + rows * tile_height + (rows - 1) * gap
    sheet_width = columns * tile_width + (columns - 1) * gap
    sheet = np.full((sheet_height, sheet_width, 3), 245, dtype=np.uint8)

    cv2.putText(
        sheet,
        title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    subtitle = (
        f"shape={tuple(feature.shape)} | post-ReLU | native resolution | "
        "shared clean/light scale per channel (p01-p99)"
    )
    cv2.putText(
        sheet,
        subtitle,
        (12, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 70, 70),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    for channel, tile in enumerate(tiles):
        row, column = divmod(channel, columns)
        y = header_height + row * (tile_height + gap)
        x = column * (tile_width + gap)
        sheet[y : y + tile_height, x : x + tile_width] = tile
    return sheet


def write_index(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "timestamp",
        "selection_source",
        "phase",
        "event_rank",
        "layer",
        "condition",
        "png_path",
        "source_npz",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.tile_width < 40:
        raise ValueError("--tile-width must be at least 40 pixels")

    manifest_rows = read_manifest(args.manifest.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_index_rows: list[dict[str, object]] = []

    print(f"[Input] {len(manifest_rows)} NPZ archives")
    print(f"[Output] {output_dir}")
    print("[Layout] conv1/layer1=8x8, layer2=16x8; all channels included")

    for item_index, manifest_row in enumerate(manifest_rows, start=1):
        sample_id = int(manifest_row["sample_id"])
        timestamp = manifest_row["timestamp"]
        source_npz = Path(manifest_row["feature_file"])

        with np.load(source_npz) as archive:
            missing = set(EXPECTED_KEYS).difference(archive.files)
            if missing:
                raise ValueError(
                    f"{source_npz} is missing arrays: {sorted(missing)}"
                )
            for layer in ("conv1", "layer1", "layer2"):
                clean = archive[f"{layer}_clean"]
                light = archive[f"{layer}_light"]
                low, high = shared_channel_ranges(clean, light)
                for condition, feature in (("clean", clean), ("light", light)):
                    png_path = output_dir / (
                        f"sample_{sample_id:02d}_{timestamp}_"
                        f"{layer}_{condition}.png"
                    )
                    if png_path.exists() and not args.overwrite:
                        raise FileExistsError(
                            f"PNG already exists: {png_path}. Use --overwrite "
                            "to replace existing exports."
                        )
                    title = (
                        f"sample {sample_id:02d} | timestamp {timestamp} | "
                        f"{layer} {condition}"
                    )
                    sheet = make_contact_sheet(
                        feature,
                        low,
                        high,
                        columns=GRID_COLUMNS[layer],
                        tile_width=args.tile_width,
                        title=title,
                    )
                    if not cv2.imwrite(str(png_path), sheet):
                        raise OSError(f"OpenCV failed to write: {png_path}")
                    png_index_rows.append(
                        {
                            "sample_id": sample_id,
                            "timestamp": timestamp,
                            "selection_source": manifest_row["selection_source"],
                            "phase": manifest_row["phase"],
                            "event_rank": manifest_row["event_rank"],
                            "layer": layer,
                            "condition": condition,
                            "png_path": str(png_path),
                            "source_npz": str(source_npz),
                        }
                    )
        print(
            f"[Export] {item_index:02d}/{len(manifest_rows):02d} "
            f"sample_{sample_id:02d}_{timestamp}"
        )

    index_path = output_dir / "png_manifest.csv"
    write_index(png_index_rows, index_path)
    total_bytes = sum(path.stat().st_size for path in output_dir.rglob("*.png"))
    print(f"[Index] {index_path}")
    print(
        f"[Done] {len(png_index_rows)} PNG files, "
        f"{total_bytes / (1024 ** 2):.1f} MiB"
    )


if __name__ == "__main__":
    main()
