#!/usr/bin/env python3
"""
record_realsense.py — Standalone headless RealSense recorder (TUM format)
=========================================================================
Records RGB + Depth frames from a RealSense camera and saves them in
TUM RGB-D format:

    <save_dir>/<YYYYMMDD-HHMMSS>/
    ├── rgb/           ← color frames as PNG  (uint8, BGR→RGB saved as RGB)
    ├── depth/         ← depth frames as PNG  (uint16, millimetres)
    ├── rgb.txt        ← "# timestamp filename\n<ts> rgb/<ts>.png\n..."
    └── depth.txt      ← "# timestamp filename\n<ts> depth/<ts>.png\n..."

No GUI, no DISPLAY, no COMO dependency required.
Press Ctrl+C to stop recording.

Usage:
    python record_realsense.py [--save_dir recordings] [--width 640] \
                               [--height 480] [--fps 30] [--depth]
"""

import argparse
import os
import sys
import time
import signal

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args():
    parser = argparse.ArgumentParser(description="Headless RealSense TUM recorder")
    parser.add_argument("--save_dir", type=str, default="recordings",
                        help="Root directory to save recordings (default: recordings/)")
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps",    type=int, default=30)
    parser.add_argument("--depth",  action="store_true", default=True,
                        help="Also record depth stream (default: True)")
    parser.add_argument("--no_depth", dest="depth", action="store_false",
                        help="Disable depth recording")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Create output directories ──────────────────────────────────────────
    timestr = time.strftime("%Y%m%d-%H%M%S")
    record_dir  = os.path.join(args.save_dir, timestr)
    rgb_dir     = os.path.join(record_dir, "rgb")
    depth_dir   = os.path.join(record_dir, "depth")
    os.makedirs(rgb_dir,   exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    rgb_txt_path   = os.path.join(record_dir, "rgb.txt")
    depth_txt_path = os.path.join(record_dir, "depth.txt")

    rgb_txt   = open(rgb_txt_path,   "w")
    depth_txt = open(depth_txt_path, "w")
    rgb_txt.write("# timestamp filename\n")
    depth_txt.write("# timestamp filename\n")

    print(f"[RealSense Recorder] Saving to: {record_dir}")
    print(f"  Resolution : {args.width}x{args.height} @ {args.fps} fps")
    print(f"  Depth      : {'enabled' if args.depth else 'disabled'}")
    print("Press Ctrl+C to stop.\n")

    # ── Configure RealSense pipeline ───────────────────────────────────────
    pipeline = rs.pipeline()
    config   = rs.config()

    config.enable_stream(rs.stream.color, args.width, args.height,
                         rs.format.bgr8, args.fps)
    if args.depth:
        config.enable_stream(rs.stream.depth, args.width, args.height,
                             rs.format.z16, args.fps)

    profile = pipeline.start(config)

    # Auto-exposure / auto-white-balance
    try:
        sensors = profile.get_device().query_sensors()
        for sensor in sensors:
            if sensor.supports(rs.option.enable_auto_exposure):
                sensor.set_option(rs.option.enable_auto_exposure, True)
            if sensor.supports(rs.option.enable_auto_white_balance):
                sensor.set_option(rs.option.enable_auto_white_balance, True)
    except Exception as e:
        print(f"[WARN] Could not set auto-exposure/WB: {e}")

    # Print intrinsics for reference
    try:
        rgb_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        intr = rgb_profile.get_intrinsics()
        print(f"[Camera Intrinsics] fx={intr.fx:.4f}, fy={intr.fy:.4f}, "
              f"cx={intr.ppx:.4f}, cy={intr.ppy:.4f}, "
              f"w={intr.width}, h={intr.height}")
        # Save intrinsics to file for later use
        with open(os.path.join(record_dir, "intrinsics.txt"), "w") as f:
            f.write(f"# RealSense camera intrinsics\n")
            f.write(f"# width height fx fy cx cy\n")
            f.write(f"{intr.width} {intr.height} "
                    f"{intr.fx:.6f} {intr.fy:.6f} "
                    f"{intr.ppx:.6f} {intr.ppy:.6f}\n")
            f.write(f"# distortion coefficients (k1 k2 p1 p2 k3)\n")
            f.write(" ".join(f"{c:.8f}" for c in intr.coeffs) + "\n")
        print(f"[Camera Intrinsics] Saved to {record_dir}/intrinsics.txt")
    except Exception as e:
        print(f"[WARN] Could not read intrinsics: {e}")

    # ── Align depth to color ───────────────────────────────────────────────
    align = rs.align(rs.stream.color) if args.depth else None

    # ── Graceful shutdown on Ctrl+C ────────────────────────────────────────
    running = [True]

    def _stop(sig, frame):
        running[0] = False

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── Main recording loop ────────────────────────────────────────────────
    frame_count = 0
    t_start = time.time()

    try:
        while running[0]:
            frameset = pipeline.wait_for_frames(timeout_ms=5000)

            if align is not None:
                frameset = align.process(frameset)

            # Timestamp in seconds (RealSense reports in ms)
            ts = frameset.get_timestamp() / 1000.0
            ts_str = f"{ts:.6f}"

            # ── RGB ────────────────────────────────────────────────────────
            color_frame = frameset.get_color_frame()
            if not color_frame:
                continue
            color_np = np.asanyarray(color_frame.get_data())  # BGR uint8

            rgb_fname   = f"rgb/{ts_str}.png"
            rgb_path    = os.path.join(record_dir, rgb_fname)
            cv2.imwrite(rgb_path, color_np)
            rgb_txt.write(f"{ts_str} {rgb_fname}\n")

            # ── Depth ──────────────────────────────────────────────────────
            if args.depth:
                depth_frame = frameset.get_depth_frame()
                if depth_frame:
                    depth_np = np.asanyarray(depth_frame.get_data())  # uint16 mm
                    depth_fname = f"depth/{ts_str}.png"
                    depth_path  = os.path.join(record_dir, depth_fname)
                    cv2.imwrite(depth_path, depth_np)
                    depth_txt.write(f"{ts_str} {depth_fname}\n")

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - t_start
                fps_actual = frame_count / elapsed
                print(f"\r  Recorded {frame_count} frames | "
                      f"{fps_actual:.1f} fps | {elapsed:.0f}s elapsed",
                      end="", flush=True)

    finally:
        pipeline.stop()
        rgb_txt.flush();   rgb_txt.close()
        depth_txt.flush(); depth_txt.close()
        elapsed = time.time() - t_start
        print(f"\n\n[Done] Recorded {frame_count} frames in {elapsed:.1f}s")
        print(f"[Done] Data saved to: {record_dir}")
        if frame_count > 0:
            print(f"       rgb/    : {frame_count} frames")
            print(f"       depth/  : {frame_count} frames")
            print(f"       rgb.txt, depth.txt, intrinsics.txt")


if __name__ == "__main__":
    main()