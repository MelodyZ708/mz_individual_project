#!/usr/bin/env python3
from pathlib import Path

SEQ_DIR = Path("/home/melody/data/tum/sequence_20260630_233103/processed")
RGB_TXT = SEQ_DIR / "rgb.txt"
ARKIT_POSE = SEQ_DIR / "arkit_pose.txt"
OUT_FILE = SEQ_DIR / "arkit_pose_filtered.txt"


def read_rgb_timestamps(path):
    ts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            t = line.split()[0]
            ts.append(t)
    return ts


def read_pose_lines(path):
    pose_map = {}
    header_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_lines.append(line.rstrip("\n"))
                continue
            parts = stripped.split()
            ts = parts[0]
            pose_map[ts] = stripped
    return header_lines, pose_map


def main():
    rgb_ts = read_rgb_timestamps(RGB_TXT)
    header_lines, pose_map = read_pose_lines(ARKIT_POSE)

    kept = 0
    missing = []

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        if header_lines:
            for h in header_lines:
                f.write(h + "\n")
        else:
            f.write("# timestamp tx ty tz qx qy qz qw\n")

        for ts in rgb_ts:
            if ts in pose_map:
                f.write(pose_map[ts] + "\n")
                kept += 1
            else:
                missing.append(ts)

    print("rgb frames:", len(rgb_ts))
    print("poses written:", kept)
    print("missing pose timestamps:", len(missing))
    if missing:
        print("first few missing:", missing[:10])
    print("output:", OUT_FILE)


if __name__ == "__main__":
    main()