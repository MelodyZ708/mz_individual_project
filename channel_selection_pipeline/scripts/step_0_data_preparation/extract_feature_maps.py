#!/usr/bin/env python3
"""
Extract native-resolution post-ReLU ResNet-18 feature maps for paired frames.

Storage layout
--------------
One compressed NPZ archive is written per paired timestamp. Each archive has:

    conv1_clean, conv1_light   [64, H/2, W/2]
    layer1_clean, layer1_light [64, H/4, W/4]
    layer2_clean, layer2_light [128, H/8, W/8]

Keeping one timestamp per file makes individual samples easy to inspect and
prevents downstream Step A/B scripts from loading the entire feature corpus
into RAM. Human-readable CSV/JSON metadata and a PNG preview are also emitted.

The extraction points intentionally match COMO tracking:

    conv1: conv1 -> bn1 -> relu
    layer1: conv1 -> bn1 -> relu -> maxpool -> layer1
    layer2: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2

All outputs are at native feature resolution; no interpolation is performed.
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
# XDG_CACHE_HOME above keeps plotting/font caches writable, but PyTorch should
# still reuse the already-downloaded ImageNet weights in the user's torch cache.
os.environ.setdefault("TORCH_HOME", str(Path.home() / ".cache" / "torch"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights, resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAIRED_JSON = (
    PROJECT_ROOT
    / "channel_selection_results/step_0_data_preparation/paired_frames"
    / "paired_frames.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_0_data_preparation"
    / "features_post_relu"
)
LAYERS = ("conv1", "layer1", "layer2")
CONDITIONS = ("clean", "light")


class NativePostReLUResNet18(nn.Module):
    """Return native-resolution outputs matching COMO's ResNet-18 paths."""

    def __init__(self) -> None:
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, rgb_01: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (rgb_01 - self.image_mean) / self.image_std
        x = self.conv1(x)
        x = self.bn1(x)
        conv1 = self.relu(x)
        x = self.maxpool(conv1)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        return {
            "conv1": conv1,
            "layer1": layer1,
            "layer2": layer2,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract post-ReLU conv1/layer1/layer2 ResNet-18 features for "
            "the selected paired timestamps."
        )
    )
    parser.add_argument(
        "--paired-json", type=Path, default=DEFAULT_PAIRED_JSON
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or a concrete device such as 'cuda:0'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of paired timestamps per forward batch.",
    )
    parser.add_argument(
        "--storage-dtype",
        choices=("float32", "float16"),
        default="float32",
        help=(
            "On-disk feature dtype. float32 is the default to avoid changing "
            "dead-channel and correlation diagnostics."
        ),
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Use uncompressed NPZ files for faster writes but larger output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing feature output directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug only: process the first N selected timestamps.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but CUDA is unavailable"
        )
    return device


def load_selected_frames(paired_json: Path) -> tuple[dict[str, Any], list[dict]]:
    if not paired_json.is_file():
        raise FileNotFoundError(
            f"Paired-frame JSON not found: {paired_json}. Run "
            "sample_paired_frames.py first."
        )
    payload = json.loads(paired_json.read_text(encoding="utf-8"))
    frames = payload.get("selected_frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No selected_frames list found in {paired_json}")

    required = {
        "sample_id",
        "frame_index",
        "timestamp",
        "clean_path",
        "lightswitch_path",
        "selection_source",
        "phase",
        "event_rank",
    }
    for index, frame in enumerate(frames):
        missing = required.difference(frame)
        if missing:
            raise ValueError(
                f"selected_frames[{index}] is missing fields: {sorted(missing)}"
            )
        for field in ("clean_path", "lightswitch_path"):
            if not Path(frame[field]).is_file():
                raise FileNotFoundError(
                    f"Image in selected_frames[{index}] does not exist: "
                    f"{frame[field]}"
                )
    frames.sort(key=lambda frame: int(frame["sample_id"]))
    return payload, frames


def load_rgb_tensor(path: Path) -> torch.Tensor:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    array = np.ascontiguousarray(image_rgb.transpose(2, 0, 1))
    return torch.from_numpy(array).to(torch.float32).div_(255.0)


def validate_batch_shapes(
    clean_tensors: list[torch.Tensor],
    light_tensors: list[torch.Tensor],
    batch_start: int,
) -> tuple[int, int]:
    shapes = {
        tuple(tensor.shape)
        for tensor in clean_tensors + light_tensors
    }
    if len(shapes) != 1:
        raise ValueError(
            f"Images in batch starting at sample {batch_start + 1} have "
            f"different shapes: {sorted(shapes)}"
        )
    channels, height, width = next(iter(shapes))
    if channels != 3:
        raise ValueError(f"Expected RGB input, got shape {(channels, height, width)}")
    return height, width


def feature_statistics(
    feature: np.ndarray,
    frame: dict,
    layer: str,
    condition: str,
) -> list[dict[str, object]]:
    flat = feature.reshape(feature.shape[0], -1).astype(np.float64)
    minimum = flat.min(axis=1)
    maximum = flat.max(axis=1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    active_ratio = np.mean(flat > 0.0, axis=1)
    rows = []
    for channel in range(feature.shape[0]):
        rows.append(
            {
                "sample_id": int(frame["sample_id"]),
                "timestamp": frame["timestamp"],
                "selection_source": frame["selection_source"],
                "phase": frame["phase"],
                "event_rank": frame["event_rank"],
                "layer": layer,
                "condition": condition,
                "channel": channel,
                "minimum": float(minimum[channel]),
                "maximum": float(maximum[channel]),
                "mean": float(mean[channel]),
                "std": float(std[channel]),
                "active_ratio": float(active_ratio[channel]),
            }
        )
    return rows


def save_npz_atomic(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    compressed: bool,
) -> None:
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")
    save_function = np.savez_compressed if compressed else np.savez
    with temporary_path.open("wb") as handle:
        save_function(handle, **arrays)
    temporary_path.replace(output_path)


def write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_preview_frame(frames: list[dict]) -> int:
    preferred = [
        (
            index,
            frame,
        )
        for index, frame in enumerate(frames)
        if frame["selection_source"] == "turn_on"
        and frame["phase"] == "peak"
        and str(frame["event_rank"]) == "1"
    ]
    return preferred[0][0] if preferred else 0


def normalise_for_display(feature_map: np.ndarray) -> np.ndarray:
    low, high = np.percentile(feature_map, [1, 99])
    if high - low < 1e-12:
        return np.zeros_like(feature_map, dtype=np.float32)
    return np.clip((feature_map - low) / (high - low), 0.0, 1.0)


def write_preview(
    preview_arrays: dict[str, np.ndarray],
    preview_frame: dict,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for row, condition in enumerate(CONDITIONS):
        for column, layer in enumerate(LAYERS):
            feature = preview_arrays[f"{layer}_{condition}"]
            mean_activation = feature.astype(np.float32).mean(axis=0)
            axes[row, column].imshow(
                normalise_for_display(mean_activation),
                cmap="viridis",
                vmin=0,
                vmax=1,
            )
            if row == 0:
                axes[row, column].set_title(
                    f"{layer}\nnative shape={tuple(feature.shape)}",
                    fontsize=11,
                )
            if column == 0:
                row_label = "Clean" if condition == "clean" else "Lightswitch"
                axes[row, column].text(
                    0.02,
                    0.97,
                    row_label,
                    transform=axes[row, column].transAxes,
                    va="top",
                    ha="left",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    bbox={
                        "facecolor": "black",
                        "alpha": 0.65,
                        "edgecolor": "none",
                        "pad": 3,
                    },
                )
            axes[row, column].axis("off")
    fig.suptitle(
        "Post-ReLU mean activation preview\n"
        f"sample={preview_frame['sample_id']}, "
        f"timestamp={preview_frame['timestamp']}, "
        f"source={preview_frame['selection_source']} "
        f"{preview_frame['phase']} rank={preview_frame['event_rank']}",
        fontsize=12,
    )
    fig.subplots_adjust(
        left=0.04, right=0.99, bottom=0.04, top=0.86, wspace=0.08, hspace=0.15
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def prepare_output_directory(output_dir: Path, overwrite: bool) -> Path:
    output_dir = output_dir.resolve()
    feature_dir = output_dir / "frames"
    known_outputs = [
        output_dir / "feature_manifest.csv",
        output_dir / "feature_channel_statistics.csv",
        output_dir / "feature_store.json",
        output_dir / "feature_preview.png",
    ]
    existing_features = list(feature_dir.glob("*.npz")) if feature_dir.is_dir() else []
    if (existing_features or any(path.exists() for path in known_outputs)) and not overwrite:
        raise FileExistsError(
            f"Feature outputs already exist in {output_dir}. Use --overwrite "
            "only if you intend to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    paired_payload, frames = load_selected_frames(args.paired_json.resolve())
    if args.limit is not None:
        frames = frames[: args.limit]
    output_dir = prepare_output_directory(args.output_dir, args.overwrite)
    feature_dir = output_dir / "frames"
    device = resolve_device(args.device)
    storage_dtype = np.float32 if args.storage_dtype == "float32" else np.float16
    compressed = not args.no_compression

    print(f"[Input] paired frames: {len(frames)} from {args.paired_json.resolve()}")
    print(f"[Model] ResNet-18 ImageNet1K V1, post-ReLU, device={device}")
    print(f"[Store] per-frame NPZ, dtype={args.storage_dtype}, compressed={compressed}")
    print("[Resolution] native feature maps; no upsampling")

    model = NativePostReLUResNet18().to(device)
    manifest_rows: list[dict[str, object]] = []
    statistics_rows: list[dict[str, object]] = []
    feature_shapes: dict[str, list[int]] = {}
    input_shape: list[int] | None = None
    preview_index = choose_preview_frame(frames)
    preview_arrays: dict[str, np.ndarray] | None = None

    for batch_start in range(0, len(frames), args.batch_size):
        batch_frames = frames[batch_start : batch_start + args.batch_size]
        clean_tensors = [
            load_rgb_tensor(Path(frame["clean_path"])) for frame in batch_frames
        ]
        light_tensors = [
            load_rgb_tensor(Path(frame["lightswitch_path"]))
            for frame in batch_frames
        ]
        height, width = validate_batch_shapes(
            clean_tensors, light_tensors, batch_start
        )
        current_input_shape = [3, height, width]
        if input_shape is None:
            input_shape = current_input_shape
        elif input_shape != current_input_shape:
            raise ValueError(
                f"Input resolution changed from {input_shape} to "
                f"{current_input_shape}"
            )

        batch = torch.stack(clean_tensors + light_tensors).to(
            device, non_blocking=True
        )
        extracted = model(batch)
        pair_count = len(batch_frames)

        for local_index, frame in enumerate(batch_frames):
            arrays: dict[str, np.ndarray] = {}
            for layer in LAYERS:
                for condition, tensor_index in (
                    ("clean", local_index),
                    ("light", pair_count + local_index),
                ):
                    key = f"{layer}_{condition}"
                    array = (
                        extracted[layer][tensor_index]
                        .detach()
                        .to("cpu")
                        .numpy()
                        .astype(storage_dtype, copy=False)
                    )
                    arrays[key] = array
                    feature_shapes[key] = list(array.shape)
                    statistics_rows.extend(
                        feature_statistics(array, frame, layer, condition)
                    )

            sample_id = int(frame["sample_id"])
            timestamp = str(frame["timestamp"])
            filename = f"sample_{sample_id:02d}_{timestamp}.npz"
            feature_path = feature_dir / filename
            save_npz_atomic(feature_path, arrays, compressed=compressed)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "frame_index": int(frame["frame_index"]),
                    "timestamp": timestamp,
                    "selection_source": frame["selection_source"],
                    "phase": frame["phase"],
                    "event_rank": frame["event_rank"],
                    "clean_path": frame["clean_path"],
                    "lightswitch_path": frame["lightswitch_path"],
                    "feature_file": str(feature_path),
                    "feature_keys": ",".join(arrays),
                    "storage_dtype": args.storage_dtype,
                }
            )
            global_index = batch_start + local_index
            if global_index == preview_index:
                preview_arrays = {key: value.copy() for key, value in arrays.items()}

        del batch, extracted
        if device.type == "cuda":
            torch.cuda.empty_cache()
        completed = min(batch_start + len(batch_frames), len(frames))
        print(f"[Extract] {completed:02d}/{len(frames):02d} paired timestamps")

    manifest_path = output_dir / "feature_manifest.csv"
    statistics_path = output_dir / "feature_channel_statistics.csv"
    metadata_path = output_dir / "feature_store.json"
    preview_path = output_dir / "feature_preview.png"
    write_csv(
        manifest_path,
        manifest_rows,
        [
            "sample_id",
            "frame_index",
            "timestamp",
            "selection_source",
            "phase",
            "event_rank",
            "clean_path",
            "lightswitch_path",
            "feature_file",
            "feature_keys",
            "storage_dtype",
        ],
    )
    write_csv(
        statistics_path,
        statistics_rows,
        [
            "sample_id",
            "timestamp",
            "selection_source",
            "phase",
            "event_rank",
            "layer",
            "condition",
            "channel",
            "minimum",
            "maximum",
            "mean",
            "std",
            "active_ratio",
        ],
    )

    metadata = {
        "format_version": 1,
        "storage": {
            "format": "per-frame NPZ",
            "compressed": compressed,
            "dtype": args.storage_dtype,
            "frame_directory": str(feature_dir),
            "manifest": str(manifest_path),
            "array_keys": [
                f"{layer}_{condition}"
                for layer in LAYERS
                for condition in CONDITIONS
            ],
        },
        "model": {
            "architecture": "torchvision ResNet-18",
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "device": str(device),
            "eval_mode": True,
        },
        "preprocessing": {
            "rgb_range": [0.0, 1.0],
            "channel_order": "RGB",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        },
        "extraction": {
            "activation_position": "post-ReLU",
            "native_resolution": True,
            "upsampling": None,
            "paths": {
                "conv1": "conv1 -> bn1 -> relu",
                "layer1": (
                    "conv1 -> bn1 -> relu -> maxpool -> layer1"
                ),
                "layer2": (
                    "conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2"
                ),
            },
            "input_shape_chw": input_shape,
            "feature_shapes_chw": feature_shapes,
        },
        "source": {
            "paired_json": str(args.paired_json.resolve()),
            "paired_protocol": paired_payload.get("protocol", {}),
            "processed_frame_count": len(frames),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if preview_arrays is None:
        raise AssertionError("Preview frame was not captured")
    write_preview(
        preview_arrays, frames[preview_index], preview_path
    )

    total_bytes = sum(path.stat().st_size for path in feature_dir.glob("*.npz"))
    print(f"[Output] {metadata_path}")
    print(f"[Output] {manifest_path}")
    print(f"[Output] {statistics_path}")
    print(f"[Output] {preview_path}")
    print(
        f"[Done] {len(manifest_rows)} NPZ files, "
        f"{total_bytes / (1024 ** 3):.2f} GiB"
    )


if __name__ == "__main__":
    main()
