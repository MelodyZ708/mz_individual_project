#!/usr/bin/env python3
"""
Convert an MP4 video to TUM-like sequence format (same as iPhone ARKit recordings).

Output structure:
    output_dir/
        rgb/              # PNG frames, named by timestamp
        rgb.txt           # timestamp path list
        intrinsics.txt    # scaled camera intrinsics
        metadata.txt      # dataset metadata

Usage:
    python mp4_to_tum_sequence.py --input video.mp4 --output /path/to/sequence_dir

Notes:
    - No depth or arkit_pose (monocular mode, no LiDAR)
    - Intrinsics are scaled from original iPhone 1920x1440 to target resolution
    - Default target resolution: 640x480
"""

import os
import cv2
import argparse
import numpy as np
from datetime import datetime


# ── Original iPhone intrinsics (from intrinsics.txt, at 256x192) ──────────────
# The ARKit app recorded at 256x192 with these values:
#   width=256, height=192, fx=181.9548, fy=181.9548, cx=127.3493, cy=97.1560
# But the actual sensor full resolution is 1920x1440 (from metadata.txt).
# Scale factor from 256x192 → 1920x1440:
#   x_scale = 1920/256 = 7.5,  y_scale = 1440/192 = 7.5
# So full-res intrinsics:
#   fx_full = 181.9548 * 7.5 = 1364.661
#   fy_full = 181.9548 * 7.5 = 1364.661
#   cx_full = 127.3493 * 7.5 = 955.120
#   cy_full =  97.1560 * 7.5 = 728.670
IPHONE_FULL_RES_W = 1920
IPHONE_FULL_RES_H = 1440
IPHONE_FX = 181.9548 * (1920 / 256)   # 1364.661
IPHONE_FY = 181.9548 * (1440 / 192)   # 1364.661
IPHONE_CX = 127.3493 * (1920 / 256)   # 955.120
IPHONE_CY =  97.1560 * (1440 / 192)   # 728.670


def scale_intrinsics(fx, fy, cx, cy, src_w, src_h, dst_w, dst_h):
    """Scale intrinsics from src resolution to dst resolution."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    return fx * sx, fy * sy, cx * sx, cy * sy


def mp4_to_sequence(input_mp4, output_dir, target_w=640, target_h=480, fps=None):
    os.makedirs(output_dir, exist_ok=True)
    rgb_dir = os.path.join(output_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_mp4)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_mp4}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Info] Input: {src_w}x{src_h} @ {src_fps:.2f}fps, {total_frames} frames")
    print(f"[Info] Output: {target_w}x{target_h}, saving to {output_dir}")

    # ── Scale intrinsics from iPhone full-res to target resolution ──────────
    # The mp4 was recorded at src_w x src_h (cropped/scaled from full sensor)
    # We treat the mp4 as coming from the full-res sensor, scaled to src resolution
    fx, fy, cx, cy = scale_intrinsics(
        IPHONE_FX, IPHONE_FY, IPHONE_CX, IPHONE_CY,
        IPHONE_FULL_RES_W, IPHONE_FULL_RES_H,
        src_w, src_h
    )
    # Then scale again from src to target
    fx, fy, cx, cy = scale_intrinsics(fx, fy, cx, cy, src_w, src_h, target_w, target_h)
    print(f"[Info] Scaled intrinsics ({target_w}x{target_h}): "
          f"fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")

    # ── Extract frames ───────────────────────────────────────────────────────
    rgb_entries = []
    frame_idx = 0
    start_time = 0.0  # base timestamp in seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute timestamp (seconds, 6 decimal places like TUM)
        timestamp = start_time + frame_idx / src_fps

        # Resize to target resolution
        if src_w != target_w or src_h != target_h:
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Save as PNG named by timestamp
        ts_str = f"{timestamp:.6f}"
        filename = f"{ts_str}.png"
        filepath = os.path.join(rgb_dir, filename)
        cv2.imwrite(filepath, frame)

        rgb_entries.append((ts_str, f"rgb/{filename}"))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    print(f"[Info] Extracted {frame_idx} frames total.")

    # ── Write rgb.txt ────────────────────────────────────────────────────────
    rgb_txt_path = os.path.join(output_dir, "rgb.txt")
    with open(rgb_txt_path, "w") as f:
        f.write("# RGB frames\n")
        f.write("# timestamp filename\n")
        for ts, path in rgb_entries:
            f.write(f"{ts} {path}\n")
    print(f"[Info] Written rgb.txt ({len(rgb_entries)} entries)")

    # ── Write intrinsics.txt ─────────────────────────────────────────────────
    intrinsics_path = os.path.join(output_dir, "intrinsics.txt")
    with open(intrinsics_path, "w") as f:
        f.write("# color camera intrinsics\n")
        f.write("# width height fx fy cx cy\n")
        f.write(f"{target_w} {target_h} {fx:.4f} {fy:.4f} {cx:.4f} {cy:.4f}\n")
        f.write("# depth png scale factor\n")
        f.write("5000.0\n")
    print(f"[Info] Written intrinsics.txt")

    # ── Write metadata.txt ───────────────────────────────────────────────────
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("device=iPhone\n")
        f.write("recorder=mp4_video\n")
        f.write("depth_source=none\n")
        f.write(f"rgb_width={target_w}\n")
        f.write(f"rgb_height={target_h}\n")
        f.write("depth_scale_factor=5000.0\n")
        f.write(f"source_video={os.path.basename(input_mp4)}\n")
        f.write(f"source_fps={src_fps:.4f}\n")
        f.write(f"total_frames={frame_idx}\n")
        f.write("note=converted from mp4, no depth, monocular mode\n")
    print(f"[Info] Written metadata.txt")

    print(f"\n[Done] Sequence saved to: {output_dir}")
    print(f"       {frame_idx} RGB frames, {target_w}x{target_h}")
    print(f"       intrinsics: fx={fx:.4f} fy={fy:.4f} cx={cx:.4f} cy={cy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Convert MP4 to TUM-like sequence format")
    parser.add_argument("--input", "-i", required=True, help="Input MP4 file path")
    parser.add_argument("--output", "-o", required=True, help="Output sequence directory")
    parser.add_argument("--width", type=int, default=640, help="Target width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Target height (default: 480)")
    args = parser.parse_args()

    mp4_to_sequence(
        input_mp4=args.input,
        output_dir=args.output,
        target_w=args.width,
        target_h=args.height,
    )


if __name__ == "__main__":
    main()
