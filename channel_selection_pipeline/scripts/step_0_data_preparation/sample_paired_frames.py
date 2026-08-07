#!/usr/bin/env python3
"""
Select 30 paired clean/lightswitch timestamps for channel analysis.

Protocol:
  * 12 approximately uniform timestamps over the full matched sequence.
  * 3 turn-on events, with one frame before/peak/after each (9 frames).
  * 3 turn-off events, with one frame before/peak/after each (9 frames).

The event detector uses the clean-normalised brightness residual

    log((mean_light + eps) / (mean_clean + eps))

rather than the lightswitch brightness alone. This removes most brightness
changes caused by camera motion or scene content. Positive sustained
excursions are turn-on events; negative excursions are turn-off events. Within
each excursion, the "peak" frame is the largest signed first-difference during
the event onset, matching the agreed brightness-jump protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Headless cluster jobs often have a read-only home directory. Keep plotting
# caches in the writable system temporary directory unless the caller already
# configured them.
_CACHE_ROOT = Path(tempfile.gettempdir()) / "mz_channel_selection_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CLEAN_DIR = Path(
    "/home/melody/data/tum/rgbd_dataset_freiburg1_desk"
)
DEFAULT_LIGHT_DIR = Path(
    "/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_0_data_preparation/paired_frames"
)


@dataclass(frozen=True)
class IndexEntry:
    timestamp_text: str
    timestamp: float
    relative_path: str
    absolute_path: Path


@dataclass(frozen=True)
class PairedFrame:
    timestamp_text: str
    timestamp: float
    clean_path: Path
    light_path: Path


@dataclass
class Event:
    kind: str
    start_index: int
    end_index: int
    transition_index: int
    extremum_index: int
    score: float
    selected: bool = False
    rank: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select 30 paired clean/lightswitch frames using uniform sampling "
            "and automatically detected turn-on/turn-off events."
        )
    )
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--light-dir", type=Path, default=DEFAULT_LIGHT_DIR)
    parser.add_argument(
        "--index-file",
        default="matched_rgb.txt",
        help="Index filename used independently inside both sequence folders.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--uniform-count", type=int, default=12)
    parser.add_argument("--events-per-kind", type=int, default=3)
    parser.add_argument(
        "--event-offset",
        type=int,
        default=4,
        help="N in the event samples [transition-N, transition, transition+N].",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Odd running-median window used before first differencing.",
    )
    parser.add_argument(
        "--min-log-ratio",
        type=float,
        default=0.05,
        help="Minimum absolute log brightness ratio for an event excursion.",
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=6.0,
        help="Robust-noise multiplier used to derive the event threshold.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Merge event-mask gaps no longer than this many frames.",
    )
    parser.add_argument(
        "--min-event-frames",
        type=int,
        default=5,
        help="Discard sustained brightness excursions shorter than this.",
    )
    return parser.parse_args()


def read_index(sequence_dir: Path, index_name: str) -> dict[str, IndexEntry]:
    index_path = sequence_dir / index_name
    if not index_path.is_file():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    entries: dict[str, IndexEntry] = {}
    for line_number, raw_line in enumerate(
        index_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(
                f"Malformed index line {index_path}:{line_number}: {raw_line!r}"
            )
        timestamp_text, relative_path = fields[0], fields[1]
        if timestamp_text in entries:
            raise ValueError(
                f"Duplicate timestamp {timestamp_text} in {index_path}"
            )
        absolute_path = (sequence_dir / relative_path).resolve()
        if not absolute_path.is_file():
            raise FileNotFoundError(
                f"Image referenced by {index_path}:{line_number} does not exist: "
                f"{absolute_path}"
            )
        entries[timestamp_text] = IndexEntry(
            timestamp_text=timestamp_text,
            timestamp=float(timestamp_text),
            relative_path=relative_path,
            absolute_path=absolute_path,
        )
    if not entries:
        raise ValueError(f"No data entries found in {index_path}")
    return entries


def pair_entries(
    clean_entries: dict[str, IndexEntry],
    light_entries: dict[str, IndexEntry],
) -> list[PairedFrame]:
    common = set(clean_entries).intersection(light_entries)
    if not common:
        raise ValueError("The clean and lightswitch indices share no timestamps")

    pairs = [
        PairedFrame(
            timestamp_text=timestamp_text,
            timestamp=clean_entries[timestamp_text].timestamp,
            clean_path=clean_entries[timestamp_text].absolute_path,
            light_path=light_entries[timestamp_text].absolute_path,
        )
        for timestamp_text in common
    ]
    pairs.sort(key=lambda pair: pair.timestamp)
    return pairs


def mean_luminance(image_path: Path) -> float:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    # OpenCV stores colour as BGR. Y is a closer brightness measure than a
    # simple three-channel average while retaining the requested frame mean.
    luminance = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[..., 0]
    return float(np.mean(luminance, dtype=np.float64))


def compute_brightness_curves(
    pairs: Iterable[PairedFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_values = []
    light_values = []
    for pair in pairs:
        clean_values.append(mean_luminance(pair.clean_path))
        light_values.append(mean_luminance(pair.light_path))

    clean = np.asarray(clean_values, dtype=np.float64)
    light = np.asarray(light_values, dtype=np.float64)
    eps = 1e-6
    residual = np.log((light + eps) / (clean + eps))
    return clean, light, residual


def running_median(values: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd integer")
    if window == 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray(
        [np.median(padded[i : i + window]) for i in range(len(values))],
        dtype=np.float64,
    )


def fill_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    result = mask.copy()
    if max_gap <= 0:
        return result
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and not result[index]:
            index += 1
        end = index
        if (
            start > 0
            and end < len(result)
            and result[start - 1]
            and result[end]
            and end - start <= max_gap
        ):
            result[start:end] = True
    return result


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def robust_event_threshold(
    residual: np.ndarray, minimum: float, noise_multiplier: float
) -> float:
    # Estimate noise from the half of frames closest to the paired baseline.
    # This remains conservative even when illumination events occupy much of
    # the sequence.
    centered = residual - np.median(residual)
    quiet_count = max(3, len(centered) // 2)
    quiet = np.partition(np.abs(centered), quiet_count - 1)[:quiet_count]
    quiet_median = np.median(quiet)
    quiet_mad = 1.4826 * np.median(np.abs(quiet - quiet_median))
    return float(max(minimum, noise_multiplier * quiet_mad))


def detect_events(
    smoothed_residual: np.ndarray,
    threshold: float,
    max_gap: int,
    min_event_frames: int,
) -> tuple[list[Event], np.ndarray]:
    derivative = np.diff(smoothed_residual, prepend=smoothed_residual[0])
    active_mask = fill_short_false_gaps(
        np.abs(smoothed_residual) >= threshold, max_gap=max_gap
    )

    events: list[Event] = []
    for start, end in true_runs(active_mask):
        if end - start + 1 < min_event_frames:
            continue
        event_values = smoothed_residual[start : end + 1]
        median_value = float(np.median(event_values))
        if median_value == 0:
            continue
        kind = "turn_on" if median_value > 0 else "turn_off"
        if kind == "turn_on":
            extremum = start + int(np.argmax(event_values))
            onset_end = extremum
            transition = start + int(np.argmax(derivative[start : onset_end + 1]))
            score = float(derivative[transition])
        else:
            extremum = start + int(np.argmin(event_values))
            onset_end = extremum
            transition = start + int(np.argmin(derivative[start : onset_end + 1]))
            score = float(-derivative[transition])
        events.append(
            Event(
                kind=kind,
                start_index=start,
                end_index=end,
                transition_index=transition,
                extremum_index=extremum,
                score=score,
            )
        )
    return events, derivative


def choose_events(
    events: list[Event],
    events_per_kind: int,
    offset: int,
    frame_count: int,
) -> list[Event]:
    chosen: list[Event] = []
    for kind in ("turn_on", "turn_off"):
        eligible = [
            event
            for event in events
            if event.kind == kind
            and event.transition_index - offset >= 0
            and event.transition_index + offset < frame_count
        ]
        eligible.sort(key=lambda event: event.score, reverse=True)
        if len(eligible) < events_per_kind:
            raise RuntimeError(
                f"Detected only {len(eligible)} usable {kind} events; "
                f"{events_per_kind} are required. Try lowering --min-log-ratio "
                "or --min-event-frames."
            )
        for rank, event in enumerate(eligible[:events_per_kind], start=1):
            event.selected = True
            event.rank = rank
            chosen.append(event)
    return chosen


def event_selections(
    chosen_events: list[Event], offset: int
) -> dict[int, list[dict[str, object]]]:
    selected: dict[int, list[dict[str, object]]] = {}
    for event in chosen_events:
        assert event.rank is not None
        for phase, index in (
            ("before", event.transition_index - offset),
            ("peak", event.transition_index),
            ("after", event.transition_index + offset),
        ):
            selected.setdefault(index, []).append(
                {
                    "source": event.kind,
                    "event_rank": event.rank,
                    "phase": phase,
                    "transition_index": event.transition_index,
                }
            )
    expected = len(chosen_events) * 3
    if len(selected) != expected:
        raise RuntimeError(
            "Event windows overlap and do not yield unique frames. Increase "
            "event separation or reduce --event-offset."
        )
    return selected


def nearest_unused_index(
    target: int, frame_count: int, used: set[int]
) -> int:
    for distance in range(frame_count):
        candidates = (target - distance, target + distance)
        for candidate in candidates:
            if 0 <= candidate < frame_count and candidate not in used:
                return candidate
    raise RuntimeError("No unused frame remains for uniform sampling")


def add_uniform_selections(
    selected: dict[int, list[dict[str, object]]],
    frame_count: int,
    uniform_count: int,
) -> None:
    if uniform_count < 1:
        raise ValueError("--uniform-count must be positive")
    targets = np.linspace(0, frame_count - 1, uniform_count)
    used = set(selected)
    for uniform_rank, target_float in enumerate(targets, start=1):
        target = int(round(float(target_float)))
        index = nearest_unused_index(target, frame_count, used)
        selected[index] = [
            {
                "source": "uniform",
                "uniform_rank": uniform_rank,
                "target_index": target,
            }
        ]
        used.add(index)


def serialise_event(event: Event, pairs: list[PairedFrame]) -> dict[str, object]:
    return {
        "kind": event.kind,
        "start_index": event.start_index,
        "end_index": event.end_index,
        "transition_index": event.transition_index,
        "extremum_index": event.extremum_index,
        "start_timestamp": pairs[event.start_index].timestamp_text,
        "end_timestamp": pairs[event.end_index].timestamp_text,
        "transition_timestamp": pairs[event.transition_index].timestamp_text,
        "extremum_timestamp": pairs[event.extremum_index].timestamp_text,
        "jump_score": event.score,
        "selected": event.selected,
        "rank_within_kind": event.rank,
    }


def build_rows(
    selected: dict[int, list[dict[str, object]]],
    pairs: list[PairedFrame],
    clean: np.ndarray,
    light: np.ndarray,
    residual: np.ndarray,
    smoothed: np.ndarray,
    derivative: np.ndarray,
) -> list[dict[str, object]]:
    rows = []
    for sample_id, index in enumerate(sorted(selected), start=1):
        pair = pairs[index]
        reasons = selected[index]
        rows.append(
            {
                "sample_id": sample_id,
                "frame_index": index,
                "timestamp": pair.timestamp_text,
                "clean_path": str(pair.clean_path),
                "lightswitch_path": str(pair.light_path),
                "selection_source": "+".join(
                    str(reason["source"]) for reason in reasons
                ),
                "phase": "+".join(
                    str(reason.get("phase", "")) for reason in reasons
                ).strip("+"),
                "event_rank": "+".join(
                    str(reason.get("event_rank", "")) for reason in reasons
                ).strip("+"),
                "clean_mean_luminance": float(clean[index]),
                "lightswitch_mean_luminance": float(light[index]),
                "log_brightness_ratio": float(residual[index]),
                "smoothed_log_brightness_ratio": float(smoothed[index]),
                "first_difference": float(derivative[index]),
                "selection_details": reasons,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "frame_index",
        "timestamp",
        "clean_path",
        "lightswitch_path",
        "selection_source",
        "phase",
        "event_rank",
        "clean_mean_luminance",
        "lightswitch_mean_luminance",
        "log_brightness_ratio",
        "smoothed_log_brightness_ratio",
        "first_difference",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def plot_diagnostics(
    pairs: list[PairedFrame],
    clean: np.ndarray,
    light: np.ndarray,
    residual: np.ndarray,
    smoothed: np.ndarray,
    derivative: np.ndarray,
    threshold: float,
    events: list[Event],
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    time_seconds = np.asarray(
        [pair.timestamp - pairs[0].timestamp for pair in pairs]
    )
    fig, axes = plt.subplots(
        3, 1, figsize=(15, 11), sharex=True, constrained_layout=True
    )

    axes[0].plot(time_seconds, clean, label="clean", linewidth=1.2)
    axes[0].plot(time_seconds, light, label="lightswitch", linewidth=1.2)
    axes[0].set_ylabel("Mean luminance (Y)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        time_seconds, residual, color="0.65", linewidth=0.8, label="raw"
    )
    axes[1].plot(
        time_seconds,
        smoothed,
        color="black",
        linewidth=1.3,
        label="running median",
    )
    axes[1].axhline(threshold, color="tab:red", linestyle="--", linewidth=0.9)
    axes[1].axhline(-threshold, color="tab:blue", linestyle="--", linewidth=0.9)
    for event in events:
        colour = "tab:red" if event.kind == "turn_on" else "tab:blue"
        alpha = 0.16 if event.selected else 0.05
        axes[1].axvspan(
            time_seconds[event.start_index],
            time_seconds[event.end_index],
            color=colour,
            alpha=alpha,
        )
    axes[1].set_ylabel("log(light / clean)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    axes[2].plot(time_seconds, derivative, color="0.35", linewidth=0.9)
    source_style = {
        "uniform": ("tab:green", "o"),
        "turn_on": ("tab:red", "^"),
        "turn_off": ("tab:blue", "v"),
    }
    labelled: set[str] = set()
    for row in rows:
        source = str(row["selection_source"]).split("+")[0]
        colour, marker = source_style[source]
        index = int(row["frame_index"])
        label = source if source not in labelled else None
        axes[2].scatter(
            time_seconds[index],
            derivative[index],
            color=colour,
            marker=marker,
            s=42,
            zorder=3,
            label=label,
        )
        labelled.add(source)
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set_xlabel("Time from first paired frame (s)")
    axes[2].set_ylabel("First difference")
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.2)

    fig.suptitle(
        "Paired-frame sampling: 12 uniform + "
        "3×(before/peak/after) turn-on + "
        "3×(before/peak/after) turn-off",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.event_offset < 1:
        raise ValueError("--event-offset must be at least 1")
    if args.events_per_kind < 1:
        raise ValueError("--events-per-kind must be positive")
    if args.min_event_frames < 1:
        raise ValueError("--min-event-frames must be positive")

    clean_entries = read_index(args.clean_dir, args.index_file)
    light_entries = read_index(args.light_dir, args.index_file)
    pairs = pair_entries(clean_entries, light_entries)
    required_count = args.uniform_count + 6 * args.events_per_kind
    if len(pairs) < required_count:
        raise RuntimeError(
            f"Only {len(pairs)} paired frames are available, but "
            f"{required_count} unique samples are required."
        )

    print(f"[Input] clean:       {args.clean_dir}")
    print(f"[Input] lightswitch: {args.light_dir}")
    print(f"[Input] paired timestamps from {args.index_file}: {len(pairs)}")

    clean, light, residual = compute_brightness_curves(pairs)
    smoothed = running_median(residual, args.smooth_window)
    threshold = robust_event_threshold(
        smoothed, args.min_log_ratio, args.noise_multiplier
    )
    events, derivative = detect_events(
        smoothed,
        threshold=threshold,
        max_gap=args.max_gap,
        min_event_frames=args.min_event_frames,
    )
    chosen_events = choose_events(
        events,
        events_per_kind=args.events_per_kind,
        offset=args.event_offset,
        frame_count=len(pairs),
    )

    selected = event_selections(chosen_events, args.event_offset)
    add_uniform_selections(selected, len(pairs), args.uniform_count)
    if len(selected) != required_count:
        raise AssertionError(
            f"Internal error: selected {len(selected)} frames, expected "
            f"{required_count}"
        )

    rows = build_rows(
        selected, pairs, clean, light, residual, smoothed, derivative
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paired_frames.csv"
    json_path = output_dir / "paired_frames.json"
    plot_path = output_dir / "brightness_event_sampling.png"

    write_csv(rows, csv_path)
    payload = {
        "protocol": {
            "total_samples": required_count,
            "uniform_samples": args.uniform_count,
            "events_per_kind": args.events_per_kind,
            "samples_per_event": ["before", "peak", "after"],
            "event_offset_frames": args.event_offset,
            "pairing_index": args.index_file,
            "event_signal": "log(mean_luminance_light / mean_luminance_clean)",
            "peak_definition": "largest signed first difference during event onset",
        },
        "inputs": {
            "clean_dir": str(args.clean_dir.resolve()),
            "lightswitch_dir": str(args.light_dir.resolve()),
            "paired_frame_count": len(pairs),
        },
        "detector": {
            "smooth_window": args.smooth_window,
            "event_threshold_log_ratio": threshold,
            "minimum_log_ratio": args.min_log_ratio,
            "noise_multiplier": args.noise_multiplier,
            "max_gap_frames": args.max_gap,
            "minimum_event_frames": args.min_event_frames,
        },
        "detected_events": [
            serialise_event(event, pairs) for event in events
        ],
        "selected_frames": rows,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_diagnostics(
        pairs,
        clean,
        light,
        residual,
        smoothed,
        derivative,
        threshold,
        events,
        rows,
        plot_path,
    )

    print(
        f"[Detect] threshold={threshold:.5f}, "
        f"turn-on={sum(event.kind == 'turn_on' for event in events)}, "
        f"turn-off={sum(event.kind == 'turn_off' for event in events)}"
    )
    for event in sorted(
        chosen_events, key=lambda item: (item.kind, item.rank or math.inf)
    ):
        pair = pairs[event.transition_index]
        print(
            f"[Select] {event.kind} #{event.rank}: "
            f"index={event.transition_index}, timestamp={pair.timestamp_text}, "
            f"jump={event.score:.5f}"
        )
    print(f"[Output] {json_path}")
    print(f"[Output] {csv_path}")
    print(f"[Output] {plot_path}")
    print(f"[Done] selected {len(rows)} unique paired timestamps")


if __name__ == "__main__":
    main()
