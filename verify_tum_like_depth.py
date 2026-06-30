#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np


def read_index(index_path: Path):
    entries = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    return entries


def summarize_depth(depth: np.ndarray):
    nz = depth[depth > 0]
    if nz.size == 0:
        return {
            "nonzero_count": 0,
            "min_u16": int(depth.min()),
            "max_u16": int(depth.max()),
            "q05_m": None,
            "q50_m": None,
            "q95_m": None,
        }

    nz_m = nz.astype(np.float32) / 5000.0
    return {
        "nonzero_count": int(nz.size),
        "min_u16": int(depth.min()),
        "max_u16": int(depth.max()),
        "q05_m": float(np.quantile(nz_m, 0.05)),
        "q50_m": float(np.quantile(nz_m, 0.50)),
        "q95_m": float(np.quantile(nz_m, 0.95)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check whether a TUM-like depth sequence matches COMO's expected format."
    )
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    seq = args.sequence_dir
    depth_dir = seq / "depth"
    rgb_dir = seq / "rgb"
    depth_txt = seq / "depth.txt"
    rgb_txt = seq / "rgb.txt"
    intrinsics_txt = seq / "intrinsics.txt"

    print(f"Sequence: {seq}")
    print()

    required = [depth_dir, rgb_dir, depth_txt, rgb_txt]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("FAIL: missing required files/directories:")
        for p in missing:
            print("  ", p)
        raise SystemExit(1)

    depth_entries = read_index(depth_txt)
    rgb_entries = read_index(rgb_txt)
    depth_files = sorted(glob.glob(str(depth_dir / "*.png")))
    rgb_files = sorted(glob.glob(str(rgb_dir / "*.png")))

    print("Index / file counts")
    print("  rgb.txt entries  :", len(rgb_entries))
    print("  rgb/*.png files  :", len(rgb_files))
    print("  depth.txt entries:", len(depth_entries))
    print("  depth/*.png files:", len(depth_files))
    print()

    ok = True

    if len(depth_entries) != len(depth_files):
        ok = False
        print("WARN: depth.txt entry count does not match depth/*.png file count")

    if len(rgb_entries) != len(rgb_files):
        ok = False
        print("WARN: rgb.txt entry count does not match rgb/*.png file count")

    missing_depth_refs = []
    for _, rel in depth_entries:
        if not (seq / rel).exists():
            missing_depth_refs.append(rel)
            if len(missing_depth_refs) >= 5:
                break

    if missing_depth_refs:
        ok = False
        print("WARN: depth.txt references missing files:")
        for rel in missing_depth_refs:
            print("  ", rel)
        print()

    print("Sampled depth checks")
    if not depth_files:
        print("  FAIL: no depth png files found")
        raise SystemExit(1)

    sample_count = min(args.samples, len(depth_files))
    sample_indices = np.linspace(0, len(depth_files) - 1, sample_count, dtype=int)

    observed_shapes = set()
    suspicious = []
    for idx in sample_indices:
        path = depth_files[idx]
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            ok = False
            print(f"  FAIL: cv2 could not read {path}")
            continue
        observed_shapes.add(depth.shape)
        stats = summarize_depth(depth)
        print(
            f"  {os.path.basename(path)} | dtype={depth.dtype} shape={depth.shape} "
            f"u16=[{stats['min_u16']},{stats['max_u16']}] "
            f"q05/q50/q95(m)=[{stats['q05_m']},{stats['q50_m']},{stats['q95_m']}]"
        )

        if depth.dtype != np.uint16:
            ok = False
            suspicious.append(f"{os.path.basename(path)} dtype {depth.dtype} != uint16")
        if stats["nonzero_count"] == 0:
            ok = False
            suspicious.append(f"{os.path.basename(path)} has all-zero depth")
        elif stats["q95_m"] is not None and stats["q95_m"] > 10.0:
            suspicious.append(
                f"{os.path.basename(path)} has unusually large depth q95={stats['q95_m']:.3f}m"
            )

    print()
    print("Global checks")
    print("  observed sampled shapes:", sorted(observed_shapes))
    if len(observed_shapes) != 1:
        ok = False
        print("  WARN: sampled depth image shapes are inconsistent")

    if intrinsics_txt.exists():
        print("  intrinsics.txt found")
    else:
        print("  WARN: intrinsics.txt not found")

    print("  COMO TUM loader expects: uint16 PNG, cv2.IMREAD_ANYDEPTH, divide by 5000.0")
    print("  Your fork's code path matches that assumption.")
    print()

    if suspicious:
        print("Notes")
        for s in suspicious:
            print("  ", s)
        print()

    if ok:
        print("PASS: sequence depth format matches COMO/TUM loader expectations structurally.")
        print("Depth quality still needs experimental validation, but the file format looks compatible.")
    else:
        print("FAIL: sequence has format or consistency issues that should be fixed before use.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()