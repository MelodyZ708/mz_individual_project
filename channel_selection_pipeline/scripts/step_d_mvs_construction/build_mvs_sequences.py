#!/usr/bin/env python3
"""Build 40/60/80-frame TUM-format MVS pilot sequences and MP4 previews.

The default anchor is the strongest selected turn-on event detected during
Step 0 (rank 1). RGB/depth pairs are sliced from matched_rgb.txt and
matched_depth.txt, while ground truth is restricted to the selected time span.

Each derived sequence is self-contained and includes:

* rgb/ and depth/ images;
* rgb.txt, depth.txt, matched_rgb.txt, and matched_depth.txt;
* groundtruth.txt;
* frame_manifest.csv and mvs_metadata.json;
* a labelled MP4 containing every RGB frame in sequence order.
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = Path(
    "/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch"
)
DEFAULT_EVENTS = (
    PROJECT_ROOT
    / "channel_selection_results/step_0_data_preparation/paired_frames"
    / "paired_frames.json"
)
DEFAULT_OUTPUT_PARENT = Path("/home/melody/data/tum")
PILOT_LAYOUTS = {
    40: {"warmup": 15, "scored": 15, "recovery": 10, "trailing": 0},
    60: {"warmup": 20, "scored": 20, "recovery": 20, "trailing": 0},
    80: {"warmup": 30, "scored": 20, "recovery": 20, "trailing": 10},
}
PHASE_COLOURS = {
    "warmup": (220, 180, 40),
    "scored": (45, 45, 235),
    "recovery": (40, 190, 70),
    "trailing": (180, 100, 190),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TUM-format MVS pilot sequences and MP4 previews."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--events-json", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument(
        "--event-kind", choices=("turn_on", "turn_off"), default="turn_on"
    )
    parser.add_argument(
        "--event-rank",
        type=int,
        default=1,
        help="Rank within the selected events of the requested kind.",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[40, 60, 80],
        choices=sorted(PILOT_LAYOUTS),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=0.0,
        help="Override preview FPS; 0 estimates it from RGB timestamps.",
    )
    return parser.parse_args()


def read_tum_index(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"TUM index not found: {path}")
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 2:
            raise ValueError(f"Invalid TUM index line in {path}: {line}")
        entries.append((fields[0], fields[1]))
    if not entries:
        raise ValueError(f"No data entries in {path}")
    return entries


def read_groundtruth(path: Path) -> tuple[list[str], list[tuple[float, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    headers: list[str] = []
    entries: list[tuple[float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            headers.append(stripped)
            continue
        fields = stripped.split()
        if len(fields) < 8:
            raise ValueError(f"Invalid ground-truth line: {line}")
        entries.append((float(fields[0]), stripped))
    if not entries:
        raise ValueError(f"No ground-truth entries in {path}")
    return headers, entries


def find_event(path: Path, kind: str, rank: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        event
        for event in payload["detected_events"]
        if event["selected"]
        and event["kind"] == kind
        and event["rank_within_kind"] == rank
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one selected {kind} event with rank {rank}, got {len(matches)}"
        )
    return matches[0]


def validate_source_pairs(
    source_dir: Path,
    rgb_entries: list[tuple[str, str]],
    depth_entries: list[tuple[str, str]],
) -> None:
    if len(rgb_entries) != len(depth_entries):
        raise ValueError(
            f"Matched RGB/depth counts differ: {len(rgb_entries)} vs {len(depth_entries)}"
        )
    for index, ((_, rgb_rel), (_, depth_rel)) in enumerate(
        zip(rgb_entries, depth_entries)
    ):
        rgb_path = source_dir / rgb_rel
        depth_path = source_dir / depth_rel
        if not rgb_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(
                f"Missing source image at pair {index}: {rgb_path} / {depth_path}"
            )


def phase_ranges(transition_index: int, length: int) -> dict[str, tuple[int, int]]:
    layout = PILOT_LAYOUTS[length]
    start = transition_index - layout["warmup"]
    cursor = start
    ranges: dict[str, tuple[int, int]] = {}
    for phase in ("warmup", "scored", "recovery", "trailing"):
        count = layout[phase]
        ranges[phase] = (cursor, cursor + count - 1) if count else (-1, -1)
        cursor += count
    if cursor - start != length:
        raise AssertionError(f"Pilot layout does not sum to {length}")
    return ranges


def phase_at(source_index: int, ranges: dict[str, tuple[int, int]]) -> str:
    for phase, (start, end) in ranges.items():
        if start <= source_index <= end:
            return phase
    raise ValueError(f"Source index {source_index} lies outside the phase ranges")


def write_index(
    path: Path,
    entries: list[tuple[str, str]],
    title: str,
    source: Path,
) -> None:
    lines = [
        f"# {title}",
        "# timestamp filename",
        f"# derived from {source}",
    ]
    lines.extend(f"{timestamp} {relative_path}" for timestamp, relative_path in entries)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def estimate_fps(entries: list[tuple[str, str]], override: float) -> float:
    if override > 0:
        return override
    timestamps = np.asarray([float(timestamp) for timestamp, _ in entries])
    differences = np.diff(timestamps)
    differences = differences[differences > 0]
    if differences.size == 0:
        return 30.0
    return float(np.clip(1.0 / np.median(differences), 1.0, 120.0))


def labelled_frame(
    image: np.ndarray,
    mvs_index: int,
    total: int,
    source_index: int,
    timestamp: str,
    phase: str,
    event_role: str = "",
) -> np.ndarray:
    output = image.copy()
    colour = PHASE_COLOURS[phase]
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (output.shape[1], 72), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, output, 0.32, 0, output)
    cv2.rectangle(output, (0, 0), (output.shape[1] - 1, output.shape[0] - 1), colour, 5)
    cv2.putText(
        output,
        f"MVS {mvs_index + 1:02d}/{total:02d}  source index {source_index:03d}",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"{phase.upper()}  timestamp {timestamp}",
        (14, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        colour,
        2,
        cv2.LINE_AA,
    )
    if event_role:
        label = event_role.upper().replace("_", " ")
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
        )[0]
        x = output.shape[1] - text_size[0] - 16
        cv2.rectangle(
            output,
            (x - 8, output.shape[0] - 43),
            (output.shape[1] - 8, output.shape[0] - 8),
            colour,
            -1,
        )
        cv2.putText(
            output,
            label,
            (x, output.shape[0] - 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def write_video(
    path: Path,
    rgb_paths: list[Path],
    manifest_rows: list[dict[str, Any]],
    fps: float,
) -> dict[str, Any]:
    first = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Could not decode first RGB frame: {rgb_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialise the mp4v MP4 writer")
    try:
        for mvs_index, (rgb_path, row) in enumerate(zip(rgb_paths, manifest_rows)):
            image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (height, width):
                raise ValueError(f"Invalid RGB frame for video: {rgb_path}")
            writer.write(
                labelled_frame(
                    image,
                    mvs_index,
                    len(rgb_paths),
                    int(row["source_paired_index"]),
                    str(row["rgb_timestamp"]),
                    str(row["phase"]),
                    str(row.get("event_role", "")),
                )
            )
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(path))
    decoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    decoded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if decoded_frames != len(rgb_paths):
        raise RuntimeError(
            f"MP4 verification failed: expected {len(rgb_paths)} frames, got {decoded_frames}"
        )
    return {
        "codec": "mp4v",
        "fps": fps,
        "frames": decoded_frames,
        "width": decoded_width,
        "height": decoded_height,
    }


def build_sequence(
    source_dir: Path,
    output_parent: Path,
    event: dict[str, Any],
    length: int,
    rgb_all: list[tuple[str, str]],
    depth_all: list[tuple[str, str]],
    gt_headers: list[str],
    gt_all: list[tuple[float, str]],
    video_fps: float,
) -> Path:
    transition = int(event["transition_index"])
    ranges = phase_ranges(transition, length)
    sequence_start = ranges["warmup"][0]
    sequence_end = max(end for _, end in ranges.values())
    if sequence_start < 0 or sequence_end >= len(rgb_all):
        raise ValueError(
            f"MVS {length} window [{sequence_start},{sequence_end}] lies outside "
            f"the {len(rgb_all)} paired frames"
        )
    event_label = event["kind"]
    rank = int(event["rank_within_kind"])
    name = (
        f"rgbd_dataset_freiburg1_desk_lightswitch_mvs_"
        f"{event_label}_rank{rank}_{length}f"
    )
    target = output_parent / name
    if target.exists():
        raise FileExistsError(
            f"Target MVS sequence already exists: {target}. Move or remove it "
            "explicitly before rebuilding."
        )
    output_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}_building_", dir=output_parent))
    try:
        (temporary / "rgb").mkdir()
        (temporary / "depth").mkdir()
        rgb_entries: list[tuple[str, str]] = []
        depth_entries: list[tuple[str, str]] = []
        rgb_paths: list[Path] = []
        manifest_rows: list[dict[str, Any]] = []

        for mvs_index, source_index in enumerate(range(sequence_start, sequence_end + 1)):
            rgb_timestamp, rgb_source_rel = rgb_all[source_index]
            depth_timestamp, depth_source_rel = depth_all[source_index]
            rgb_name = Path(rgb_source_rel).name
            depth_name = Path(depth_source_rel).name
            rgb_target_rel = f"rgb/{rgb_name}"
            depth_target_rel = f"depth/{depth_name}"
            shutil.copy2(source_dir / rgb_source_rel, temporary / rgb_target_rel)
            shutil.copy2(source_dir / depth_source_rel, temporary / depth_target_rel)
            rgb_entries.append((rgb_timestamp, rgb_target_rel))
            depth_entries.append((depth_timestamp, depth_target_rel))
            rgb_paths.append(temporary / rgb_target_rel)
            manifest_rows.append(
                {
                    "mvs_index": mvs_index,
                    "source_paired_index": source_index,
                    "phase": phase_at(source_index, ranges),
                    "rgb_timestamp": rgb_timestamp,
                    "rgb_path": rgb_target_rel,
                    "depth_timestamp": depth_timestamp,
                    "depth_path": depth_target_rel,
                }
            )

        for filename in ("rgb.txt", "matched_rgb.txt"):
            write_index(
                temporary / filename,
                rgb_entries,
                f"MVS RGB index ({length} frames)",
                source_dir,
            )
        for filename in ("depth.txt", "matched_depth.txt"):
            write_index(
                temporary / filename,
                depth_entries,
                f"MVS depth index ({length} frames)",
                source_dir,
            )

        start_time = float(rgb_entries[0][0]) - 0.05
        end_time = float(rgb_entries[-1][0]) + 0.05
        selected_gt = [line for timestamp, line in gt_all if start_time <= timestamp <= end_time]
        if not selected_gt:
            raise ValueError("No ground-truth entries overlap the MVS time range")
        groundtruth_lines = gt_headers or ["# ground truth trajectory"]
        (temporary / "groundtruth.txt").write_text(
            "\n".join(groundtruth_lines + selected_gt) + "\n", encoding="utf-8"
        )

        with (temporary / "frame_manifest.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

        fps = estimate_fps(rgb_entries, video_fps)
        video_name = f"{name}.mp4"
        video_info = write_video(
            temporary / video_name, rgb_paths, manifest_rows, fps
        )
        metadata = {
            "format_version": 1,
            "purpose": "MVS pilot for channel-combination ATE evaluation",
            "source_sequence": str(source_dir),
            "sequence_name": name,
            "frame_count": length,
            "source_paired_index_range_inclusive": [sequence_start, sequence_end],
            "event": event,
            "phase_ranges_source_index_inclusive": {
                phase: list(bounds) for phase, bounds in ranges.items()
            },
            "phase_counts": PILOT_LAYOUTS[length],
            "scored_mvs_indices_inclusive": [
                ranges["scored"][0] - sequence_start,
                ranges["scored"][1] - sequence_start,
            ],
            "rgb_timestamp_range": [rgb_entries[0][0], rgb_entries[-1][0]],
            "groundtruth_entry_count": len(selected_gt),
            "video": {"file": video_name, **video_info},
            "notes": [
                "The 40-frame sequence is intentionally the shortest pilot.",
                "Only the scored phase should contribute to the primary MVS score.",
                "Warm-up/recovery/trailing frames remain available for tracker-state checks.",
            ],
        }
        (temporary / "mvs_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (temporary / "README.md").write_text(
            f"# {name}\n\n"
            f"Derived from `{source_dir}` around {event_label} rank {rank}.\n\n"
            f"- Frames: {length}\n"
            f"- Source paired indices: {sequence_start}--{sequence_end}\n"
            f"- Scored source indices: {ranges['scored'][0]}--{ranges['scored'][1]}\n"
            f"- Preview: `{video_name}`\n\n"
            "See `mvs_metadata.json` and `frame_manifest.csv` for exact boundaries.\n",
            encoding="utf-8",
        )
        temporary.rename(target)
    except Exception:
        # Keep the temporary directory for forensic inspection; it is hidden
        # and never mistaken for a completed sequence.
        raise
    return target


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_parent = args.output_parent.resolve()
    event = find_event(args.events_json.resolve(), args.event_kind, args.event_rank)
    rgb_all = read_tum_index(source_dir / "matched_rgb.txt")
    depth_all = read_tum_index(source_dir / "matched_depth.txt")
    validate_source_pairs(source_dir, rgb_all, depth_all)
    gt_headers, gt_all = read_groundtruth(source_dir / "groundtruth.txt")

    print(f"[Source] {source_dir}")
    print(
        f"[Event] {event['kind']} rank={event['rank_within_kind']} "
        f"transition={event['transition_index']} extremum={event['extremum_index']}"
    )
    outputs = []
    for length in dict.fromkeys(args.lengths):
        output = build_sequence(
            source_dir,
            output_parent,
            event,
            length,
            rgb_all,
            depth_all,
            gt_headers,
            gt_all,
            args.video_fps,
        )
        outputs.append(output)
        print(f"[Built] {output}")
    print(f"[Done] Built {len(outputs)} MVS pilot sequences")


if __name__ == "__main__":
    main()
