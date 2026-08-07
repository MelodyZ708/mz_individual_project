#!/usr/bin/env python3
"""Generate COMO-ready TUM variants with QueensCAMP camera failures.

The script deliberately invokes the official QueensCAMP implementation at a
pinned Git commit instead of duplicating its image-processing code here.
Every output sequence is copied independently from the clean source sequence,
then only its ``rgb/`` images are overwritten. Depth, timestamps, associations,
intrinsics (when present), and ground truth therefore remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


QUEENSCAMP_URL = "https://github.com/larocs/queenscamp-dataset.git"
QUEENSCAMP_COMMIT = "26df800ffcad93a0a373937410bf9a4073083c20"
FAILURES = {
    "underexposure": {"gamma": 2},
    "overexposure": {"gamma": 0.5},
    "blur": {"kernel_size": 35},
    "condensation": {"alpha": 1.0},
    "dirt": {"alpha": 0.65},
    "wet": {"alpha": 0.75},
}
METADATA_FILES = (
    "rgb.txt",
    "depth.txt",
    "matched_rgb.txt",
    "matched_depth.txt",
    "groundtruth.txt",
)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_index(sequence: Path, filename: str) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    index_path = sequence / filename
    with index_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed line in {index_path}: {stripped!r}")
            entries.append((fields[0], sequence / fields[1]))
    return entries


def validate_source(source: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source sequence does not exist: {source}")

    missing = [name for name in METADATA_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Source is not directly COMO-ready; missing: " + ", ".join(missing)
        )

    for directory in ("rgb", "depth"):
        if not (source / directory).is_dir():
            raise FileNotFoundError(f"Missing source directory: {source / directory}")

    matched_rgb = read_index(source, "matched_rgb.txt")
    matched_depth = read_index(source, "matched_depth.txt")
    if not matched_rgb or len(matched_rgb) != len(matched_depth):
        raise ValueError(
            "matched_rgb.txt and matched_depth.txt must be non-empty and have "
            f"equal lengths; got {len(matched_rgb)} and {len(matched_depth)}"
        )

    for _, path in matched_rgb + matched_depth:
        if not path.is_file():
            raise FileNotFoundError(f"Indexed source frame does not exist: {path}")


def prepare_queenscamp(repo: Path, *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] QueensCAMP cache: {repo}")
        print(f"[dry-run] Required commit: {QUEENSCAMP_COMMIT}")
        return

    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", QUEENSCAMP_URL, str(repo)])
    elif not (repo / ".git").is_dir():
        raise ValueError(f"QueensCAMP path is not a Git repository: {repo}")

    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if actual != QUEENSCAMP_COMMIT:
        # Fetching the exact object makes an old or shallow cache usable.
        run(["git", "fetch", "origin", QUEENSCAMP_COMMIT], cwd=repo)
        run(["git", "checkout", "--detach", QUEENSCAMP_COMMIT], cwd=repo)
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    if actual != QUEENSCAMP_COMMIT:
        raise RuntimeError(f"Expected QueensCAMP {QUEENSCAMP_COMMIT}, got {actual}")

    if not (repo / "insert_failures.py").is_file():
        raise FileNotFoundError(f"Missing QueensCAMP injector: {repo / 'insert_failures.py'}")


def validate_output(source: Path, output: Path) -> dict[str, object]:
    for filename in METADATA_FILES:
        source_hash = file_sha256(source / filename)
        output_hash = file_sha256(output / filename)
        if source_hash != output_hash:
            raise RuntimeError(f"Metadata changed unexpectedly: {output / filename}")

    rgb_entries = read_index(source, "rgb.txt")
    changed = 0
    channels: set[int] = set()
    mean_differences: list[float] = []

    for _, source_image_path in rgb_entries:
        relative = source_image_path.relative_to(source)
        output_image_path = output / relative
        clean = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
        degraded = cv2.imread(str(output_image_path), cv2.IMREAD_COLOR)
        unchanged = cv2.imread(str(output_image_path), cv2.IMREAD_UNCHANGED)

        if clean is None or degraded is None or unchanged is None:
            raise RuntimeError(f"Unreadable source/output image: {relative}")
        if clean.shape != degraded.shape:
            raise RuntimeError(
                f"Image shape changed for {relative}: {clean.shape} -> {degraded.shape}"
            )

        channel_count = 1 if unchanged.ndim == 2 else unchanged.shape[2]
        channels.add(channel_count)
        difference = np.abs(clean.astype(np.int16) - degraded.astype(np.int16))
        if np.any(difference):
            changed += 1
        mean_differences.append(float(difference.mean()))

    if changed != len(rgb_entries):
        raise RuntimeError(
            f"Only {changed}/{len(rgb_entries)} indexed RGB frames changed in {output}"
        )

    matched_rgb = read_index(output, "matched_rgb.txt")
    matched_depth = read_index(output, "matched_depth.txt")
    for _, path in matched_rgb + matched_depth:
        if not path.is_file():
            raise FileNotFoundError(f"COMO-indexed output frame is missing: {path}")

    return {
        "rgb_index_entries": len(rgb_entries),
        "changed_rgb_index_entries": changed,
        "matched_frames": len(matched_rgb),
        "stored_png_channels": sorted(channels),
        "mean_absolute_pixel_difference": float(np.mean(mean_differences)),
        "metadata_sha256": {
            name: file_sha256(output / name) for name in METADATA_FILES
        },
    }


def generate_variant(
    source: Path,
    output_root: Path,
    queenscamp_repo: Path,
    failure: str,
    *,
    dry_run: bool,
) -> None:
    output = output_root / f"{source.name}_queenscamp_{failure}"
    print(f"\n[{failure}] {output}")
    if output.exists():
        if dry_run:
            print("[dry-run] Output already exists; a real run would refuse to overwrite it.")
            return
        raise FileExistsError(
            f"Output already exists (refusing to overwrite): {output}"
        )
    if dry_run:
        return

    shutil.copytree(source, output, copy_function=shutil.copy2)
    try:
        run(
            [
                sys.executable,
                "insert_failures.py",
                "--sequence_path",
                str(source / "rgb"),
                "--failure_type",
                failure,
                "--output_path",
                str(output / "rgb"),
            ],
            cwd=queenscamp_repo,
        )
        validation = validate_output(source, output)
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_sequence": str(source),
            "output_sequence": str(output),
            "failure": failure,
            "official_parameters": FAILURES[failure],
            "queenscamp_repository": QUEENSCAMP_URL,
            "queenscamp_commit": QUEENSCAMP_COMMIT,
            "generator": str(Path(__file__).resolve()),
            "validation": validation,
        }
        (output / "queenscamp_generation_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[OK] {failure}: {validation['changed_rgb_index_entries']} RGB "
            f"frames changed; {validation['matched_frames']} COMO matched frames"
        )
    except Exception:
        print(
            f"[ERROR] Incomplete output retained for inspection: {output}",
            file=sys.stderr,
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate six QueensCAMP-degraded, directly COMO-compatible copies "
            "of a TUM RGB-D sequence. Existing outputs are never overwritten."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Clean TUM sequence containing matched_rgb/depth and groundtruth files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Parent directory for outputs (default: source parent).",
    )
    parser.add_argument(
        "--queenscamp-repo",
        type=Path,
        default=Path.home() / ".cache" / "como" / "queenscamp-dataset",
        help="Clone/cache location for the pinned official QueensCAMP repository.",
    )
    parser.add_argument(
        "--failures",
        nargs="+",
        choices=tuple(FAILURES),
        default=list(FAILURES),
        help="Subset to generate (default: all six).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the source and print the plan without writing or cloning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else source.parent
    )
    queenscamp_repo = args.queenscamp_repo.expanduser().resolve()

    validate_source(source)
    prepare_queenscamp(queenscamp_repo, dry_run=args.dry_run)
    print("Source:", source)
    print("Output root:", output_root)
    print("Failures:", ", ".join(args.failures))
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    for failure in args.failures:
        generate_variant(
            source,
            output_root,
            queenscamp_repo,
            failure,
            dry_run=args.dry_run,
        )

    print("\nDry run complete." if args.dry_run else "\nAll variants generated and validated.")


if __name__ == "__main__":
    main()
