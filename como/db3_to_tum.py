#!/usr/bin/env python3
"""
db3_to_tum.py — Convert RealSense .db3 recording to TUM RGB-D format
=====================================================================
Reads a .db3 file recorded by rs-record (librealsense 2.55+) and extracts
RGB + Depth frames into TUM format:

    <output_dir>/
    ├── rgb/           ← color frames as PNG  (uint8 RGB)
    ├── depth/         ← depth frames as PNG  (uint16, millimetres)
    ├── rgb.txt        ← "# timestamp filename\n<ts> rgb/<ts>.png\n..."
    ├── depth.txt      ← "# timestamp filename\n<ts> depth/<ts>.png\n..."
    └── intrinsics.txt ← camera intrinsics (fx fy cx cy)

Usage:
    python db3_to_tum.py --input recording2.db3 --output_dir recordings/my_seq
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    parser = argparse.ArgumentParser(description="Convert .db3 to TUM RGB-D format")
    parser.add_argument("--input",      type=str, required=True,
                        help="Path to input .db3 file")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same name as .db3 file)")
    parser.add_argument("--align_depth", action="store_true", default=True,
                        help="Align depth to color frame (default: True)")
    parser.add_argument("--no_align",   dest="align_depth", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Output directory
    if args.output_dir is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output_dir = os.path.join(os.path.dirname(args.input), base + "_tum")

    rgb_dir   = os.path.join(args.output_dir, "rgb")
    depth_dir = os.path.join(args.output_dir, "depth")
    os.makedirs(rgb_dir,   exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    rgb_txt_path   = os.path.join(args.output_dir, "rgb.txt")
    depth_txt_path = os.path.join(args.output_dir, "depth.txt")

    print(f"[db3_to_tum] Input  : {args.input}")
    print(f"[db3_to_tum] Output : {args.output_dir}")

    # ── Setup pipeline with playback ───────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_device_from_file(args.input, repeat_playback=False)

    profile = pipeline.start(config)

    # Set playback to non-realtime (process as fast as possible)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # Get intrinsics from the recording
    try:
        color_stream = profile.get_stream(rs.stream.color)
        intr = color_stream.as_video_stream_profile().get_intrinsics()
        print(f"[Intrinsics] fx={intr.fx:.4f}, fy={intr.fy:.4f}, "
              f"cx={intr.ppx:.4f}, cy={intr.ppy:.4f}, "
              f"w={intr.width}, h={intr.height}")
        with open(os.path.join(args.output_dir, "intrinsics.txt"), "w") as f:
            f.write("# RealSense camera intrinsics from recording\n")
            f.write("# width height fx fy cx cy\n")
            f.write(f"{intr.width} {intr.height} "
                    f"{intr.fx:.6f} {intr.fy:.6f} "
                    f"{intr.ppx:.6f} {intr.ppy:.6f}\n")
            f.write("# distortion coefficients (k1 k2 p1 p2 k3)\n")
            f.write(" ".join(f"{c:.8f}" for c in intr.coeffs) + "\n")
    except Exception as e:
        print(f"[WARN] Could not read intrinsics: {e}")

    align = rs.align(rs.stream.color) if args.align_depth else None

    # ── Extract frames ─────────────────────────────────────────────────────
    frame_count = 0
    rgb_entries   = []
    depth_entries = []

    print("Extracting frames...")

    try:
        while True:
            try:
                frameset = pipeline.wait_for_frames(timeout_ms=3000)
            except RuntimeError:
                # End of file
                break

            if align is not None:
                frameset = align.process(frameset)

            ts = frameset.get_timestamp() / 1000.0  # ms → s
            ts_str = f"{ts:.6f}"

            # RGB
            color_frame = frameset.get_color_frame()
            if color_frame:
                color_np = np.asanyarray(color_frame.get_data())  # BGR uint8
                rgb_fname = f"rgb/{ts_str}.png"
                cv2.imwrite(os.path.join(args.output_dir, rgb_fname), color_np)
                rgb_entries.append(f"{ts_str} {rgb_fname}")

            # Depth
            depth_frame = frameset.get_depth_frame()
            if depth_frame:
                depth_np = np.asanyarray(depth_frame.get_data())  # uint16 mm
                depth_fname = f"depth/{ts_str}.png"
                cv2.imwrite(os.path.join(args.output_dir, depth_fname), depth_np)
                depth_entries.append(f"{ts_str} {depth_fname}")

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"\r  Extracted {frame_count} frames...", end="", flush=True)

    except Exception as e:
        print(f"\n[WARN] Stopped early: {e}")
    finally:
        pipeline.stop()

    # Write txt files
    with open(rgb_txt_path, "w") as f:
        f.write("# timestamp filename\n")
        f.write("\n".join(rgb_entries) + "\n")

    with open(depth_txt_path, "w") as f:
        f.write("# timestamp filename\n")
        f.write("\n".join(depth_entries) + "\n")

    print(f"\n\n[Done] Extracted {frame_count} frames")
    print(f"[Done] RGB   : {len(rgb_entries)} frames → {rgb_txt_path}")
    print(f"[Done] Depth : {len(depth_entries)} frames → {depth_txt_path}")
    print(f"\nTo run COMO on this data:")
    print(f"  python como/como_dataset.py --dataset_type=tum \\")
    print(f"    --dataset_dir={args.output_dir}/")


if __name__ == "__main__":
    main()