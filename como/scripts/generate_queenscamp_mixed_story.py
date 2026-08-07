#!/usr/bin/env python3
"""Generate the non-stacked QueensCAMP Mixed Story V1 for TUM fr1/desk.

The schedule is defined over ``matched_rgb.txt`` because those are the frames
that COMO actually consumes. Each frame has exactly one active state: clean or
one degradation. Intensities ramp within a state, but effects are never
composited with one another.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

import generate_queenscamp_tum_variants as common


STORY_NAME = "queenscamp_mixed_story_v1"
EXPECTED_FRAMES = 572
STORY_SEGMENTS = (
    {"frames": [0, 59], "effect": "clean", "story": "normal initialization"},
    {
        "frames": [60, 129],
        "effect": "underexposure",
        "story": "illumination progressively dims",
    },
    {
        "frames": [130, 184],
        "effect": "overexposure",
        "story": "power returns and auto-exposure temporarily overshoots",
    },
    {"frames": [185, 309], "effect": "clean", "story": "exposure recovery"},
    {
        "frames": [310, 369],
        "effect": "blur",
        "story": "camera accelerates; blur peaks near the GT high-speed window",
    },
    {"frames": [370, 439], "effect": "wet", "story": "liquid splashes the lens"},
    {
        "frames": [440, 509],
        "effect": "condensation",
        "story": "remaining moisture develops into lens haze",
    },
    {
        "frames": [510, 571],
        "effect": "dirt",
        "story": "moisture recedes and leaves residue",
    },
)


def linear(frame: int, start: int, end: int, first: float, last: float) -> float:
    if start == end:
        return first
    fraction = (frame - start) / (end - start)
    return first + fraction * (last - first)


def odd_kernel(value: float) -> int:
    rounded = int(round(value))
    rounded = max(1, rounded)
    return rounded if rounded % 2 else rounded + 1


def state_for_frame(frame: int) -> tuple[str, str, float]:
    """Return (effect, parameter name, value) for one matched-frame index."""
    if 0 <= frame <= 59 or 185 <= frame <= 309:
        return "clean", "identity", 1.0
    if 60 <= frame <= 109:
        return "underexposure", "gamma", linear(frame, 60, 109, 1.15, 2.40)
    if 110 <= frame <= 129:
        return "underexposure", "gamma", linear(frame, 110, 129, 2.40, 1.60)
    if 130 <= frame <= 144:
        return "overexposure", "gamma", 0.45
    if 145 <= frame <= 184:
        # Stop just short of identity so every frame labelled overexposure is
        # measurably degraded while still appearing visually recovered.
        return "overexposure", "gamma", linear(frame, 145, 184, 0.45, 0.98)
    if 310 <= frame <= 339:
        return "blur", "kernel_size", float(
            odd_kernel(linear(frame, 310, 339, 7, 45))
        )
    if 340 <= frame <= 349:
        return "blur", "kernel_size", 45.0
    if 350 <= frame <= 369:
        return "blur", "kernel_size", float(
            odd_kernel(linear(frame, 350, 369, 45, 7))
        )
    if 370 <= frame <= 389:
        return "wet", "alpha", linear(frame, 370, 389, 0.25, 0.85)
    if 390 <= frame <= 419:
        return "wet", "alpha", 0.85
    if 420 <= frame <= 439:
        return "wet", "alpha", linear(frame, 420, 439, 0.85, 0.20)
    if 440 <= frame <= 469:
        return "condensation", "alpha", linear(frame, 440, 469, 0.25, 1.00)
    if 470 <= frame <= 489:
        return "condensation", "alpha", 1.00
    if 490 <= frame <= 509:
        return "condensation", "alpha", linear(frame, 490, 509, 1.00, 0.30)
    if 510 <= frame <= 539:
        return "dirt", "alpha", linear(frame, 510, 539, 0.25, 0.75)
    if 540 <= frame <= 571:
        return "dirt", "alpha", 0.75
    raise IndexError(f"Mixed Story V1 has no schedule for frame {frame}")


def load_official_module(repo: Path) -> ModuleType:
    script = repo / "insert_failures.py"
    spec = importlib.util.spec_from_file_location("queenscamp_insert_failures", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official QueensCAMP injector: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_templates(repo: Path) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for effect in ("wet", "condensation", "dirt"):
        candidates = sorted((repo / "failures" / effect).glob("*.png"))
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one pinned {effect} template, found {len(candidates)}"
            )
        image = cv2.imread(str(candidates[0]), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] != 4:
            raise RuntimeError(f"Invalid QueensCAMP template: {candidates[0]}")
        templates[effect] = image
    return templates


def transform(
    image: np.ndarray,
    effect: str,
    value: float,
    official: ModuleType,
    templates: dict[str, np.ndarray],
) -> np.ndarray:
    if effect == "clean":
        return image
    if effect in ("underexposure", "overexposure"):
        return official.gamma_correction(image, value)
    if effect == "blur":
        return official.blur_image(image, int(value))
    if effect in templates:
        bgra = official.overlay_images(image, templates[effect], value)
        # The alpha plane is fully opaque after the official operation. Dropping
        # it preserves official BGR pixel values and keeps every TUM RGB PNG 3-ch.
        return bgra[:, :, :3]
    raise ValueError(f"Unsupported effect: {effect}")


def write_schedule(path: Path, entries: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("frame\ttimestamp\teffect\tparameter\tvalue\trgb_path\n")
        for entry in entries:
            stream.write(
                f"{entry['frame']}\t{entry['timestamp']}\t{entry['effect']}\t"
                f"{entry['parameter']}\t{entry['value']:.8g}\t{entry['rgb_path']}\n"
            )


def read_como_index(sequence: Path, filename: str) -> list[tuple[str, Path]]:
    """Match TumOdometryDataset's current physical-line indexing exactly."""
    index_path = sequence / filename
    lines = index_path.read_text(encoding="utf-8").splitlines()
    entries: list[tuple[str, Path]] = []
    for line in lines[3:]:
        fields = line.split()
        if len(fields) < 2:
            raise ValueError(f"Malformed COMO index line in {index_path}: {line!r}")
        entries.append((fields[0], sequence / fields[1]))
    return entries


def validate_story(
    source: Path, output: Path, schedule: list[dict[str, object]]
) -> dict[str, object]:
    for filename in common.METADATA_FILES:
        if common.file_sha256(source / filename) != common.file_sha256(output / filename):
            raise RuntimeError(f"Metadata changed unexpectedly: {filename}")

    effect_counts = Counter()
    changed_counts = Counter()
    difference_sums: dict[str, list[float]] = defaultdict(list)
    channels: set[int] = set()

    for entry in schedule:
        relative = Path(str(entry["rgb_path"]))
        clean = cv2.imread(str(source / relative), cv2.IMREAD_COLOR)
        degraded = cv2.imread(str(output / relative), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(output / relative), cv2.IMREAD_UNCHANGED)
        if clean is None or degraded is None or raw is None:
            raise RuntimeError(f"Unreadable frame: {relative}")
        if clean.shape != degraded.shape:
            raise RuntimeError(f"Shape changed for {relative}")
        channels.add(1 if raw.ndim == 2 else raw.shape[2])
        effect = str(entry["effect"])
        effect_counts[effect] += 1
        difference = np.abs(clean.astype(np.int16) - degraded.astype(np.int16))
        is_changed = bool(np.any(difference))
        changed_counts[effect] += int(is_changed)
        difference_sums[effect].append(float(difference.mean()))
        if effect == "clean" and is_changed:
            raise RuntimeError(f"Clean frame changed unexpectedly: {relative}")
        if effect != "clean" and not is_changed:
            raise RuntimeError(f"Degraded frame did not change: {relative}")

    if channels != {3}:
        raise RuntimeError(f"Expected uniform 3-channel output, found {sorted(channels)}")

    return {
        "matched_frames": len(schedule),
        "effect_frame_counts": dict(effect_counts),
        "changed_frame_counts": dict(changed_counts),
        "stored_png_channels": sorted(channels),
        "mean_absolute_pixel_difference_by_effect": {
            effect: float(np.mean(values))
            for effect, values in difference_sums.items()
        },
        "metadata_sha256": {
            filename: common.file_sha256(output / filename)
            for filename in common.METADATA_FILES
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate non-stacked QueensCAMP Mixed Story V1 for TUM fr1/desk."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <source-parent>/<source-name>_queenscamp_mixed_story_v1",
    )
    parser.add_argument(
        "--queenscamp-repo",
        type=Path,
        default=Path.home() / ".cache" / "como" / "queenscamp-dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source.parent / f"{source.name}_{STORY_NAME}"
    )
    repo = args.queenscamp_repo.expanduser().resolve()

    common.validate_source(source)
    matched_entries = read_como_index(source, "matched_rgb.txt")
    if len(matched_entries) != EXPECTED_FRAMES:
        raise ValueError(
            f"Mixed Story V1 requires {EXPECTED_FRAMES} matched frames, "
            f"but {source} has {len(matched_entries)}"
        )

    schedule: list[dict[str, object]] = []
    for frame, (timestamp, image_path) in enumerate(matched_entries):
        effect, parameter, value = state_for_frame(frame)
        schedule.append(
            {
                "frame": frame,
                "timestamp": timestamp,
                "effect": effect,
                "parameter": parameter,
                "value": value,
                "rgb_path": str(image_path.relative_to(source)),
            }
        )

    print("Source:", source)
    print("Output:", output)
    print("Matched frames:", len(schedule))
    print("Effects:", dict(Counter(str(x["effect"]) for x in schedule)))
    if output.exists():
        if args.dry_run:
            print("[dry-run] Output exists; a real run would refuse to overwrite it.")
            return
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.dry_run:
        print("[dry-run] No files written and no repository cloned.")
        return

    common.prepare_queenscamp(repo, dry_run=False)
    official = load_official_module(repo)
    templates = load_templates(repo)
    shutil.copytree(source, output, copy_function=shutil.copy2)

    try:
        for entry in schedule:
            if entry["effect"] == "clean":
                continue
            relative = Path(str(entry["rgb_path"]))
            image = cv2.imread(str(source / relative), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot read source RGB: {source / relative}")
            result = transform(
                image,
                str(entry["effect"]),
                float(entry["value"]),
                official,
                templates,
            )
            if not cv2.imwrite(str(output / relative), result):
                raise RuntimeError(f"Cannot write output RGB: {output / relative}")

        schedule_path = output / "mixed_story_schedule.tsv"
        write_schedule(schedule_path, schedule)
        validation = validate_story(source, output, schedule)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "story_name": STORY_NAME,
            "story_summary": (
                "normal initialization; illumination dims; exposure overshoots "
                "on recovery; GT-aligned fast-motion blur; liquid splash; "
                "condensation; drying residue"
            ),
            "non_stacking": True,
            "schedule_index": "matched_rgb.txt order (COMO-consumed frames)",
            "segments": STORY_SEGMENTS,
            "source_sequence": str(source),
            "output_sequence": str(output),
            "queenscamp_repository": common.QUEENSCAMP_URL,
            "queenscamp_commit": common.QUEENSCAMP_COMMIT,
            "generator": str(Path(__file__).resolve()),
            "schedule_file": schedule_path.name,
            "validation": validation,
        }
        (output / "queenscamp_generation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        print("Mixed Story V1 generated and validated:", output)
    except Exception:
        print("Incomplete output retained for inspection:", output)
        raise


if __name__ == "__main__":
    main()
