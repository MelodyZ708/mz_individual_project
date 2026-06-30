#!/usr/bin/env python3
"""
One-click preprocessing for an iPhone LiDAR TUM-like sequence so it becomes
usable by the current COMO fork.

What this script does:
1. Optionally backs up the original sequence.
2. Rebuilds rgb.txt from actual rgb image files.
3. Rebuilds depth.txt by nearest-timestamp matching to rgb.txt.
4. Detects and fixes 16-bit depth endianness problems (byteswap) if needed.
5. Optionally resizes RGB images to COMO's default 256x192.
6. Updates intrinsics.txt to the resized RGB resolution.
7. Optionally deletes unmatched extra RGB/depth image files.

Recommended usage:
    python3 preprocess_iphone_sequence_for_como.py \
      "/path/to/sequence_YYYYMMDD_HHMMSS" \
      --resize-rgb --delete-unmatched
"""

import argparse
import glob
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


TARGET_W = 256
TARGET_H = 192
RGB_EXTS = ("*.png", "*.jpg", "*.jpeg")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess iPhone LiDAR sequence for COMO.")
    parser.add_argument("sequence_dir", type=Path, help="Path to sequence_* directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional output directory. If omitted, process in place.",
    )
    parser.add_argument(
        "--resize-rgb",
        action="store_true",
        help="Resize RGB images to 256x192 and update intrinsics.txt",
    )
    parser.add_argument(
        "--delete-unmatched",
        action="store_true",
        help="Delete extra RGB/depth files that are not used by the rebuilt indices",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a full backup copy before in-place processing.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=8,
        help="How many depth images to sample when detecting byte-order issues.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")


def list_rgb_files(rgb_dir: Path):
    files = []
    for pattern in RGB_EXTS:
        files.extend(glob.glob(str(rgb_dir / pattern)))
    return sorted(Path(p) for p in files)


def list_depth_files(depth_dir: Path):
    return sorted(Path(p) for p in glob.glob(str(depth_dir / "*.png")))


def timestamp_from_name(path: Path):
    return float(path.stem)


def detect_depth_byteswap(depth_files, sample_count):
    if not depth_files:
        return False, None

    idxs = np.linspace(0, len(depth_files) - 1, min(sample_count, len(depth_files)), dtype=int)
    original_q95 = []
    swapped_q95 = []

    for idx in idxs:
        d = cv2.imread(str(depth_files[idx]), cv2.IMREAD_UNCHANGED)
        if d is None or d.dtype != np.uint16:
            continue

        nz = d[d > 0]
        if nz.size:
            original_q95.append(float(np.quantile(nz.astype(np.float32) / 5000.0, 0.95)))

        d_swap = d.byteswap()
        nz2 = d_swap[d_swap > 0]
        if nz2.size:
            swapped_q95.append(float(np.quantile(nz2.astype(np.float32) / 5000.0, 0.95)))

    if not original_q95 or not swapped_q95:
        return False, None

    orig_med = float(np.median(original_q95))
    swap_med = float(np.median(swapped_q95))

    should_swap = orig_med > 4.0 and swap_med < orig_med
    return should_swap, {"orig_q95_median": orig_med, "swap_q95_median": swap_med}


def fix_depth_endianness(depth_files):
    for path in depth_files:
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        d_fix = d.byteswap()
        cv2.imwrite(str(path), d_fix)


def write_rgb_index(seq_dir: Path, rgb_files):
    rgb_txt = seq_dir / "rgb.txt"
    with open(rgb_txt, "w", encoding="utf-8") as f:
        f.write("# RGB images\n")
        f.write("# timestamp filename\n")
        for path in rgb_files:
            ts = path.stem
            f.write(f"{ts} rgb/{path.name}\n")
    return rgb_txt


def read_index(index_path: Path):
    entries = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((float(parts[0]), parts[1]))
    return entries


def write_depth_index_from_matching(seq_dir: Path, rgb_entries, depth_files):
    depth_entries = [(timestamp_from_name(p), f"depth/{p.name}") for p in depth_files]
    depth_entries.sort(key=lambda x: x[0])

    matched = []
    j = 0
    for rgb_ts, _ in rgb_entries:
        while (
            j + 1 < len(depth_entries)
            and abs(depth_entries[j + 1][0] - rgb_ts) <= abs(depth_entries[j][0] - rgb_ts)
        ):
            j += 1
        matched.append((rgb_ts, depth_entries[j][1], abs(depth_entries[j][0] - rgb_ts)))

    depth_txt = seq_dir / "depth.txt"
    with open(depth_txt, "w", encoding="utf-8") as f:
        f.write("# depth maps\n")
        f.write("# timestamp filename\n")
        for rgb_ts, rel, _ in matched:
            f.write(f"{rgb_ts:.6f} {rel}\n")

    max_dt = max((x[2] for x in matched), default=0.0)
    mean_dt = sum((x[2] for x in matched), 0.0) / len(matched) if matched else 0.0
    return depth_txt, max_dt, mean_dt


def resize_rgb_in_place(rgb_files):
    for path in rgb_files:
        img = cv2.imread(str(path))
        if img is None:
            continue
        resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(path), resized)


def update_intrinsics_for_resized_rgb(intrinsics_path: Path):
    if not intrinsics_path.exists():
        return False

    lines = intrinsics_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    updated = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            new_lines.append(line)
            continue

        parts = stripped.split()
        try:
            values = [float(x) for x in parts]
        except ValueError:
            new_lines.append(line)
            continue

        if len(values) >= 6 and not updated:
            orig_w, orig_h = values[0], values[1]
            fx, fy, cx, cy = values[2], values[3], values[4], values[5]
            scale_x = TARGET_W / orig_w
            scale_y = TARGET_H / orig_h
            new_fx = fx * scale_x
            new_fy = fy * scale_y
            new_cx = cx * scale_x
            new_cy = cy * scale_y
            new_lines.append(
                f"{TARGET_W} {TARGET_H} {new_fx:.4f} {new_fy:.4f} {new_cx:.4f} {new_cy:.4f}"
            )
            updated = True
        else:
            new_lines.append(line)

    intrinsics_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return updated


def delete_unmatched_files(seq_dir: Path, used_rgb_names, used_depth_names):
    rgb_dir = seq_dir / "rgb"
    depth_dir = seq_dir / "depth"
    deleted_rgb = 0
    deleted_depth = 0

    for p in list_rgb_files(rgb_dir):
        if p.name not in used_rgb_names:
            p.unlink()
            deleted_rgb += 1

    for p in list_depth_files(depth_dir):
        if p.name not in used_depth_names:
            p.unlink()
            deleted_depth += 1

    return deleted_rgb, deleted_depth


def backup_sequence(seq_dir: Path):
    backup_dir = seq_dir.parent / f"{seq_dir.name}_backup_before_preprocess"
    if backup_dir.exists():
        return backup_dir
    shutil.copytree(seq_dir, backup_dir)
    return backup_dir


def prepare_output_dir(seq_dir: Path, out_dir):
    if out_dir is None:
        return seq_dir

    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    shutil.copytree(seq_dir, out_dir)
    return out_dir


def main():
    args = parse_args()
    seq_src = args.sequence_dir.resolve()
    ensure_exists(seq_src, "sequence directory")
    ensure_exists(seq_src / "rgb", "rgb directory")
    ensure_exists(seq_src / "depth", "depth directory")
    ensure_exists(seq_src / "intrinsics.txt", "intrinsics.txt")

    if args.backup and args.out_dir is None:
        backup_dir = backup_sequence(seq_src)
        print(f"[backup] {backup_dir}")

    out_dir = args.out_dir.resolve() if args.out_dir else None
    seq = prepare_output_dir(seq_src, out_dir)
    rgb_dir = seq / "rgb"
    depth_dir = seq / "depth"
    intrinsics_path = seq / "intrinsics.txt"

    rgb_files = list_rgb_files(rgb_dir)
    depth_files = list_depth_files(depth_dir)

    print(f"[sequence] {seq}")
    print(f"[rgb files]   {len(rgb_files)}")
    print(f"[depth files] {len(depth_files)}")

    should_swap, stats = detect_depth_byteswap(depth_files, args.sample_count)
    if stats:
        print(
            f"[depth check] q95 median original={stats['orig_q95_median']:.3f} m, "
            f"byteswapped={stats['swap_q95_median']:.3f} m"
        )
    if should_swap:
        print("[depth fix] Detected likely endianness problem. Applying byteswap to all depth PNGs.")
        fix_depth_endianness(depth_files)
    else:
        print("[depth fix] No byteswap applied.")

    if args.resize_rgb:
        print(f"[rgb resize] Resizing RGB images to {TARGET_W}x{TARGET_H}")
        resize_rgb_in_place(rgb_files)
        updated = update_intrinsics_for_resized_rgb(intrinsics_path)
        print(f"[intrinsics] updated={updated}")
    else:
        print("[rgb resize] skipped")

    rgb_files = list_rgb_files(rgb_dir)
    depth_files = list_depth_files(depth_dir)

    rgb_txt = write_rgb_index(seq, rgb_files)
    rgb_entries = read_index(rgb_txt)
    depth_txt, max_dt, mean_dt = write_depth_index_from_matching(seq, rgb_entries, depth_files)

    print(f"[rgb.txt]   {rgb_txt}")
    print(f"[depth.txt] {depth_txt}")
    print(f"[matching]  rgb={len(rgb_entries)} depth={len(depth_files)} max_dt={max_dt:.6f}s mean_dt={mean_dt:.6f}s")

    if args.delete_unmatched:
        used_rgb_names = {Path(rel).name for _, rel in rgb_entries}
        used_depth_names = {Path(rel).name for _, rel in read_index(depth_txt)}
        deleted_rgb, deleted_depth = delete_unmatched_files(seq, used_rgb_names, used_depth_names)
        print(f"[cleanup] deleted extra rgb={deleted_rgb}, depth={deleted_depth}")
    else:
        print("[cleanup] skipped")

    print()
    print("Done. This sequence is now much closer to what your COMO fork expects:")
    print("- rgb.txt rebuilt from actual files")
    print("- depth.txt matched to rgb timestamps")
    print("- depth format corrected if byte-order problem was detected")
    if args.resize_rgb:
        print(f"- RGB resized to {TARGET_W}x{TARGET_H} and intrinsics updated")
    print()
    print("Recommended next step:")
    print(f'python3 verify_tum_like_depth.py "{seq}"')


if __name__ == "__main__":
    main()