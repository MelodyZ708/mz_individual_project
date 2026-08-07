#!/usr/bin/env python3
"""Build four diverse 30-frame MVS clips from desk_lightswitch.

Selection is based only on adjacent-frame luminance changes in the lightswitch
sequence. The four non-overlapping clips cover early/middle/late viewpoints and
balance positive and negative illumination transitions. Each clip contains 10
warm-up frames followed by a 20-frame scored window; the event anchor occurs
five frames into the scored window, so scoring contains pre-change context and
the subsequent response rather than only the strongest frame pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_mvs_sequences import (
    DEFAULT_OUTPUT_PARENT,
    DEFAULT_SOURCE,
    estimate_fps,
    read_groundtruth,
    read_tum_index,
    validate_source_pairs,
    write_index,
    write_video,
)


CLIPS = {
    "A": {
        "start": 82,
        "end": 111,
        "anchor": 97,
        "direction": "positive",
        "interpretation": "early sequence; strong brightening/recovery",
    },
    "B": {
        "start": 173,
        "end": 202,
        "anchor": 188,
        "direction": "negative",
        "interpretation": "middle-early sequence; strong turn-off",
    },
    "C": {
        "start": 368,
        "end": 397,
        "anchor": 383,
        "direction": "positive",
        "interpretation": "middle-late sequence; strong turn-on",
    },
    "D": {
        "start": 450,
        "end": 479,
        "anchor": 465,
        "direction": "negative",
        "interpretation": "late sequence; strong turn-off",
    },
    "C50": {
        "start": 368,
        "end": 417,
        "anchor": 383,
        "direction": "positive_then_negative",
        "interpretation": (
            "challenging continuous segment; sharp brightening, bright plateau, "
            "then strong dimming"
        ),
        "warmup_frames": 10,
        "event_markers": {
            383: "turn_on_anchor",
            416: "dimming_anchor",
        },
        "output_stem": (
            "rgbd_dataset_freiburg1_desk_lightswitch_mvs_challenging_"
            "c_brighten_dim_idx368_417_50f"
        ),
    },
    "FAIL30": {
        "start": 224,
        "end": 253,
        "anchor": 248,
        "direction": "brightening_to_tracking_failure",
        "interpretation": (
            "failure-centred segment from the full gray-baseline run; rapid "
            "brightening drives the image into saturation immediately before "
            "the first inferred non-finite tracked pose"
        ),
        "warmup_frames": 10,
        "event_markers": {
            240: "rapid_brightening_onset",
            248: "first_invalid_pose_inferred",
            249: "first_confirmed_nan_affine",
        },
        "output_stem": (
            "rgbd_dataset_freiburg1_desk_lightswitch_mvs_failure_anchor_"
            "idx248_30f"
        ),
        "direction_balance": (
            "not applicable; this clip targets the full-run brightening failure"
        ),
        "viewpoint_diversity": (
            "30 consecutive frames retain motion and scene variation around failure"
        ),
    },
    "FAIL50": {
        "start": 235,
        "end": 284,
        "anchor": 248,
        "direction": "brightening_failure_saturation_then_dimming",
        "interpretation": (
            "failure-centred segment covering pre-failure tracking, rapid "
            "brightening into saturation, the gray-baseline hard failure, "
            "the saturated plateau, and subsequent strong dimming"
        ),
        "warmup_frames": 10,
        "event_markers": {
            240: "rapid_brightening_onset",
            248: "first_invalid_pose_inferred",
            249: "full_run_first_confirmed_nan_affine",
            278: "clear_dimming_onset",
            284: "strong_dimming_reached",
        },
        "output_stem": (
            "rgbd_dataset_freiburg1_desk_lightswitch_mvs_failure_anchor_"
            "idx248_brighten_dim_50f"
        ),
        "direction_balance": (
            "one continuous segment containing failure-inducing brightening and dimming"
        ),
        "viewpoint_diversity": (
            "50 consecutive frames retain motion and scene variation across both transitions"
        ),
    },
}
DIVERSE_CLIP_IDS = ("A", "B", "C", "D")
WARMUP_FRAMES = 10
SCORED_FRAMES = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build four diverse 30-frame MVS clips."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument(
        "--clips", nargs="+", choices=tuple(CLIPS), default=list(DIVERSE_CLIP_IDS)
    )
    parser.add_argument("--video-fps", type=float, default=0.0)
    return parser.parse_args()


def luminance_diagnostics(
    source_dir: Path, rgb_entries: list[tuple[str, str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.empty(len(rgb_entries), dtype=np.float64)
    medians = np.empty(len(rgb_entries), dtype=np.float64)
    for index, (_, relative_path) in enumerate(rgb_entries):
        image = cv2.imread(str(source_dir / relative_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode RGB image: {source_dir / relative_path}")
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        means[index] = float(np.mean(luminance))
        medians[index] = float(np.median(luminance))
    delta_mean = np.full(len(rgb_entries), np.nan, dtype=np.float64)
    delta_median = np.full(len(rgb_entries), np.nan, dtype=np.float64)
    delta_mean[1:] = np.diff(np.log(means + 1e-6))
    delta_median[1:] = np.diff(np.log(medians + 1e-6))
    return means, medians, delta_mean, delta_median


def copy_clip(
    source_dir: Path,
    output_parent: Path,
    clip_id: str,
    specification: dict[str, Any],
    rgb_all: list[tuple[str, str]],
    depth_all: list[tuple[str, str]],
    gt_headers: list[str],
    gt_all: list[tuple[float, str]],
    means: np.ndarray,
    medians: np.ndarray,
    delta_mean: np.ndarray,
    delta_median: np.ndarray,
    video_fps: float,
) -> Path:
    start = int(specification["start"])
    end = int(specification["end"])
    anchor = int(specification["anchor"])
    frame_count = end - start + 1
    warmup_frames = int(specification.get("warmup_frames", WARMUP_FRAMES))
    scored_frames = frame_count - warmup_frames
    if frame_count < 20 or scored_frames < 10:
        raise ValueError(f"Clip {clip_id} has an invalid warm-up/scored layout")
    if not start + warmup_frames <= anchor <= end:
        raise ValueError(f"Clip {clip_id} anchor is outside the scored window")

    name = specification.get(
        "output_stem",
        (
            f"rgbd_dataset_freiburg1_desk_lightswitch_mvs_diverse_"
            f"{clip_id.lower()}_{specification['direction']}_idx{anchor:03d}_"
            f"{frame_count}f"
        ),
    )
    target = output_parent / name
    if target.exists():
        raise FileExistsError(
            f"Target already exists: {target}. Move or remove it explicitly before rebuilding."
        )
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}_building_", dir=output_parent))
    (temporary / "rgb").mkdir()
    (temporary / "depth").mkdir()

    rgb_entries: list[tuple[str, str]] = []
    depth_entries: list[tuple[str, str]] = []
    rgb_paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    for mvs_index, source_index in enumerate(range(start, end + 1)):
        rgb_timestamp, rgb_source_rel = rgb_all[source_index]
        depth_timestamp, depth_source_rel = depth_all[source_index]
        rgb_target_rel = f"rgb/{Path(rgb_source_rel).name}"
        depth_target_rel = f"depth/{Path(depth_source_rel).name}"
        shutil.copy2(source_dir / rgb_source_rel, temporary / rgb_target_rel)
        shutil.copy2(source_dir / depth_source_rel, temporary / depth_target_rel)
        phase = "warmup" if mvs_index < warmup_frames else "scored"
        event_markers = specification.get(
            "event_markers", {anchor: "event_anchor"}
        )
        event_role = str(event_markers.get(source_index, ""))
        rgb_entries.append((rgb_timestamp, rgb_target_rel))
        depth_entries.append((depth_timestamp, depth_target_rel))
        rgb_paths.append(temporary / rgb_target_rel)
        manifest_rows.append(
            {
                "mvs_index": mvs_index,
                "source_paired_index": source_index,
                "phase": phase,
                "event_role": event_role,
                "rgb_timestamp": rgb_timestamp,
                "rgb_path": rgb_target_rel,
                "depth_timestamp": depth_timestamp,
                "depth_path": depth_target_rel,
                "mean_luminance": float(means[source_index]),
                "median_luminance": float(medians[source_index]),
                "adjacent_log_mean_change": float(delta_mean[source_index]),
                "adjacent_log_median_change": float(delta_median[source_index]),
            }
        )

    for filename in ("rgb.txt", "matched_rgb.txt"):
        write_index(
            temporary / filename,
            rgb_entries,
            f"Diverse MVS {clip_id} RGB index (30 frames)",
            source_dir,
        )
    for filename in ("depth.txt", "matched_depth.txt"):
        write_index(
            temporary / filename,
            depth_entries,
            f"Diverse MVS {clip_id} depth index (30 frames)",
            source_dir,
        )

    first_time = float(rgb_entries[0][0]) - 0.05
    last_time = float(rgb_entries[-1][0]) + 0.05
    selected_gt = [line for timestamp, line in gt_all if first_time <= timestamp <= last_time]
    if not selected_gt:
        raise ValueError(f"No ground truth overlaps clip {clip_id}")
    (temporary / "groundtruth.txt").write_text(
        "\n".join((gt_headers or ["# ground truth trajectory"]) + selected_gt) + "\n",
        encoding="utf-8",
    )
    with (temporary / "frame_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    fps = estimate_fps(rgb_entries, video_fps)
    video_name = f"{name}.mp4"
    video_info = write_video(temporary / video_name, rgb_paths, manifest_rows, fps)
    anchor_local = anchor - start
    marker_payload = [
        {
            "source_paired_index": int(marker_index),
            "mvs_index": int(marker_index - start),
            "role": str(role),
            "timestamp": rgb_all[int(marker_index)][0],
            "adjacent_log_mean_change": float(delta_mean[int(marker_index)]),
            "adjacent_log_median_change": float(delta_median[int(marker_index)]),
        }
        for marker_index, role in specification.get(
            "event_markers", {anchor: "event_anchor"}
        ).items()
    ]
    metadata = {
        "format_version": 1,
        "purpose": "short MVS clip for channel-combination ATE",
        "source_sequence": str(source_dir),
        "sequence_name": name,
        "clip_id": clip_id,
        "frame_count": frame_count,
        "source_paired_index_range_inclusive": [start, end],
        "warmup_mvs_indices_inclusive": [0, warmup_frames - 1],
        "scored_mvs_indices_inclusive": [warmup_frames, frame_count - 1],
        "event_anchor": {
            "source_paired_index": anchor,
            "mvs_index": anchor_local,
            "timestamp": rgb_all[anchor][0],
            "direction": specification["direction"],
            "interpretation": specification["interpretation"],
            "adjacent_log_mean_change": float(delta_mean[anchor]),
            "adjacent_log_median_change": float(delta_median[anchor]),
            "mean_luminance_before": float(means[anchor - 1]),
            "mean_luminance_after": float(means[anchor]),
        },
        "event_markers": marker_payload,
        "selection_policy": {
            "signal": "adjacent log mean luminance, checked against log median luminance",
            "segment_not_pair": True,
            "temporal_non_overlap": True,
            "direction_balance": (
                specification.get("direction_balance")
                or (
                    "two positive and two negative clips"
                    if clip_id in DIVERSE_CLIP_IDS
                    else "one continuous segment containing brightening and dimming"
                )
            ),
            "viewpoint_diversity": (
                specification.get("viewpoint_diversity")
                or (
                    "clips distributed over early/middle/late sequence"
                    if clip_id in DIVERSE_CLIP_IDS
                    else "temporal-phase diversity within one continuous viewpoint segment"
                )
            ),
        },
        "groundtruth_entry_count": len(selected_gt),
        "video": {"file": video_name, **video_info},
        "caution": (
            f"{warmup_frames} warm-up frames are a short-sequence pilot assumption and must be "
            "validated with known tracking configurations."
        ),
    }
    (temporary / "mvs_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (temporary / "README.md").write_text(
        f"# MVS {clip_id}\n\n"
        f"- Source indices: {start}--{end}\n"
        f"- Warm-up: MVS 0--{warmup_frames - 1}\n"
        f"- Scored: MVS {warmup_frames}--{frame_count - 1}\n"
        f"- Event anchor: source {anchor}, MVS {anchor_local}\n"
        f"- Direction: {specification['direction']}\n"
        f"- Preview: `{video_name}`\n",
        encoding="utf-8",
    )
    temporary.rename(target)
    return target


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_parent = args.output_parent.resolve()
    rgb_all = read_tum_index(source_dir / "matched_rgb.txt")
    depth_all = read_tum_index(source_dir / "matched_depth.txt")
    validate_source_pairs(source_dir, rgb_all, depth_all)
    gt_headers, gt_all = read_groundtruth(source_dir / "groundtruth.txt")
    means, medians, delta_mean, delta_median = luminance_diagnostics(
        source_dir, rgb_all
    )
    print(f"[Source] {source_dir} ({len(rgb_all)} paired frames)")
    outputs = []
    for clip_id in dict.fromkeys(args.clips):
        specification = CLIPS[clip_id]
        anchor = specification["anchor"]
        output = copy_clip(
            source_dir,
            output_parent,
            clip_id,
            specification,
            rgb_all,
            depth_all,
            gt_headers,
            gt_all,
            means,
            medians,
            delta_mean,
            delta_median,
            args.video_fps,
        )
        outputs.append(output)
        print(
            f"[Built] {clip_id}: {output} "
            f"delta={delta_mean[anchor]:+.4f}"
        )
    print(f"[Done] Built {len(outputs)} diverse MVS sequences")


if __name__ == "__main__":
    main()
