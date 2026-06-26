"""
convert_iphone_depth.py
=======================
Converts iPhone LiDAR depth images to TUM-compatible format.

iPhone LiDAR stores depth as uint16 with scale=13107:
    depth_meters = pixel_value / 13107   (range: 0 ~ 5m)

TUM format uses scale=5000:
    depth_meters = pixel_value / 5000    (range: 0 ~ 13m)

This script re-encodes the depth so that the physical depth in metres
is preserved, but the integer encoding matches TUM convention.

    new_pixel = old_pixel * (5000 / 13107)

The converted depth/ folder replaces the original in-place after
backing up the original to depth_orig/.

Usage:
    python convert_iphone_depth.py --seq_dir /home/melody/data/tum/sequence_20260623_093547_small
    python convert_iphone_depth.py --seq_dir /path/to/seq --iphone_scale 13107 --tum_scale 5000
"""

import argparse
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def convert_depth_dir(depth_dir: Path, out_depth_dir: Path,
                      iphone_scale: float, tum_scale: float):
    out_depth_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([
        f for f in depth_dir.iterdir()
        if f.suffix == ".png" and not f.name.startswith("._")
    ])

    if not files:
        print(f"[ERROR] No depth PNG files found in {depth_dir}")
        return

    print(f"Found {len(files)} depth images.")
    print(f"Scale conversion: {iphone_scale} → {tum_scale}  "
          f"(multiply by {tum_scale / iphone_scale:.6f})")

    # Sanity check on first file
    sample = cv2.imread(str(files[0]), cv2.IMREAD_ANYDEPTH)
    print(f"Sample before: max={sample.max()}, mean={sample.mean():.1f}  "
          f"→ {sample.mean() / iphone_scale:.3f} m")

    ratio = tum_scale / iphone_scale  # ≈ 0.3815

    for f in tqdm(files, desc="Converting depth"):
        d = cv2.imread(str(f), cv2.IMREAD_ANYDEPTH)
        if d is None:
            print(f"[WARN] Cannot read {f.name}, skipping.")
            continue
        d_float = d.astype(np.float32) * ratio
        d_new = np.clip(np.round(d_float), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out_depth_dir / f.name), d_new)

    # Verify one converted file
    sample_out = cv2.imread(str(out_depth_dir / files[0].name), cv2.IMREAD_ANYDEPTH)
    print(f"Sample after : max={sample_out.max()}, mean={sample_out.mean():.1f}  "
          f"→ {sample_out.mean() / tum_scale:.3f} m")
    print(f"[DONE] Converted depth saved to: {out_depth_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert iPhone LiDAR depth to TUM scale format"
    )
    parser.add_argument(
        "--seq_dir", type=str, required=True,
        help="Path to the sequence directory (contains depth/ folder)"
    )
    parser.add_argument(
        "--iphone_scale", type=float, default=13107.0,
        help="Scale factor of iPhone depth (default: 13107, i.e. 65535/5m)"
    )
    parser.add_argument(
        "--tum_scale", type=float, default=5000.0,
        help="Target TUM scale factor (default: 5000)"
    )
    parser.add_argument(
        "--no_backup", action="store_true",
        help="Skip backing up original depth/ to depth_orig/"
    )
    args = parser.parse_args()

    seq_dir = Path(args.seq_dir)
    assert seq_dir.exists(), f"seq_dir does not exist: {seq_dir}"

    depth_dir = seq_dir / "depth"
    assert depth_dir.exists(), f"depth/ folder not found in {seq_dir}"

    depth_orig_dir = seq_dir / "depth_orig"
    depth_converted_dir = seq_dir / "depth_converted"

    # Step 1: Convert to a temporary folder first
    print("=" * 55)
    print("Step 1: Converting depth images...")
    convert_depth_dir(depth_dir, depth_converted_dir,
                      args.iphone_scale, args.tum_scale)

    # Step 2: Backup original depth/
    if not args.no_backup:
        if not depth_orig_dir.exists():
            print(f"\nStep 2: Backing up original depth/ → depth_orig/")
            shutil.copytree(str(depth_dir), str(depth_orig_dir))
            print(f"[BACKUP] {depth_orig_dir}")
        else:
            print(f"\nStep 2: depth_orig/ already exists, skipping backup.")

    # Step 3: Replace depth/ with converted
    print(f"\nStep 3: Replacing depth/ with converted images...")
    shutil.rmtree(str(depth_dir))
    shutil.move(str(depth_converted_dir), str(depth_dir))
    print(f"[DONE] depth/ now contains TUM-scale depth images.")

    print()
    print("=" * 55)
    print("Conversion complete!")
    print(f"  Sequence dir : {seq_dir}")
    print(f"  Original backup : {depth_orig_dir}")
    print(f"  depth/ is now TUM-compatible (scale=5000)")
    print()
    print("Verify with:")
    print(f"  python3 -c \"import cv2; d=cv2.imread('{depth_dir}/206138.879785.png', "
          f"cv2.IMREAD_ANYDEPTH); print(d.mean()/5000, 'm')\"")


if __name__ == "__main__":
    main()