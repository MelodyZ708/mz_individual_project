"""
iPhone sequence RGB pre-processing script
==========================================
Resizes all RGB images from 1920x1440 → 256x192 in-place (or to a new directory).
Also updates intrinsics.txt to match the new resolution.

Usage:
    python preprocess_rgb.py --seq_dir /home/melody/data/tum/sequence_20260623_093547
    python preprocess_rgb.py --seq_dir /home/melody/data/tum/sequence_20260623_093547 --out_dir /home/melody/data/tum/sequence_20260623_093547_small
"""

import argparse
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

TARGET_W = 256
TARGET_H = 192


def resize_rgb_sequence(seq_dir: Path, out_dir: Path):
    rgb_dir = seq_dir / "rgb"
    out_rgb_dir = out_dir / "rgb"
    out_rgb_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
    if not png_files:
        print(f"[ERROR] No images found in {rgb_dir}")
        return

    print(f"Found {len(png_files)} RGB images. Resizing {png_files[0].name} as sample check...")

    # Sanity check: read first image
    sample = cv2.imread(str(png_files[0]))
    if sample is None:
        print(f"[ERROR] Cannot read {png_files[0]}")
        return
    orig_h, orig_w = sample.shape[:2]
    print(f"  Original size : {orig_w} x {orig_h}")
    print(f"  Target size   : {TARGET_W} x {TARGET_H}")

    if orig_w == TARGET_W and orig_h == TARGET_H:
        print("[INFO] Images are already at target resolution. Nothing to do.")
        return

    for img_path in tqdm(png_files, desc="Resizing RGB"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] Skipping unreadable file: {img_path.name}")
            continue
        resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        out_path = out_rgb_dir / img_path.name
        cv2.imwrite(str(out_path), resized)

    print(f"[DONE] Resized images saved to: {out_rgb_dir}")


def update_intrinsics(seq_dir: Path, out_dir: Path):
    src = seq_dir / "intrinsics.txt"
    dst = out_dir / "intrinsics.txt"

    if not src.exists():
        print("[WARN] intrinsics.txt not found, skipping.")
        return

    lines = src.read_text().splitlines()
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
            # Line format: width height fx fy cx cy
            orig_w, orig_h = values[0], values[1]
            fx, fy, cx, cy = values[2], values[3], values[4], values[5]

            scale_x = TARGET_W / orig_w
            scale_y = TARGET_H / orig_h

            new_fx = fx * scale_x
            new_fy = fy * scale_y
            new_cx = cx * scale_x
            new_cy = cy * scale_y

            new_line = f"{TARGET_W} {TARGET_H} {new_fx:.4f} {new_fy:.4f} {new_cx:.4f} {new_cy:.4f}"
            new_lines.append(new_line)
            updated = True

            print(f"[intrinsics] Original : {int(orig_w)}x{int(orig_h)}  fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}")
            print(f"[intrinsics] Updated  : {TARGET_W}x{TARGET_H}  fx={new_fx:.4f} fy={new_fy:.4f} cx={new_cx:.4f} cy={new_cy:.4f}")
        else:
            new_lines.append(line)

    dst.write_text("\n".join(new_lines) + "\n")
    print(f"[DONE] intrinsics.txt saved to: {dst}")


def copy_other_files(seq_dir: Path, out_dir: Path):
    """Copy depth/, depth.txt, rgb.txt, arkit_pose.txt, metadata.txt as-is."""
    for item in seq_dir.iterdir():
        if item.name == "rgb":
            continue  # already handled
        if item.name == "intrinsics.txt":
            continue  # already handled
        dst = out_dir / item.name
        if item.is_dir():
            if not dst.exists():
                shutil.copytree(str(item), str(dst))
                print(f"[COPY] {item.name}/ → {dst}")
        else:
            shutil.copy2(str(item), str(dst))
            print(f"[COPY] {item.name} → {dst}")


def main():
    parser = argparse.ArgumentParser(description="Resize iPhone RGB sequence to 192x256")
    parser.add_argument("--seq_dir", type=str, required=True,
                        help="Path to the sequence directory (contains rgb/, depth/, intrinsics.txt, ...)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory. If not specified, overwrites in-place (original RGB backed up to rgb_orig/).")
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    assert seq_dir.exists(), f"seq_dir does not exist: {seq_dir}"

    if args.out_dir is None:
        # In-place mode: backup original rgb/ to rgb_orig/
        rgb_orig = seq_dir / "rgb_orig"
        if not rgb_orig.exists():
            print(f"[BACKUP] Backing up original rgb/ → rgb_orig/")
            shutil.copytree(str(seq_dir / "rgb"), str(rgb_orig))
        out_dir = seq_dir
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        copy_other_files(seq_dir, out_dir)

    resize_rgb_sequence(seq_dir if args.out_dir is None else seq_dir, out_dir)
    update_intrinsics(seq_dir, out_dir)

    print()
    print("=" * 50)
    print("Pre-processing complete.")
    print(f"Output directory : {out_dir}")
    print()
    print("Next steps:")
    print("  1. Check intrinsics.txt in the output directory.")
    print("  2. Make sure depth_scale in intrinsics.txt is 1000.0 (iPhone LiDAR = mm units).")
    print("  3. Run COMO with --dataset_dir pointing to the output directory.")


if __name__ == "__main__":
    main()