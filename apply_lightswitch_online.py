"""
apply_lightswitch_online.py
============================
Online lightswitch augmentation with intensity_scale.

Reads the original TUM dataset, applies a scaled version of the
mode_lightswitch effect from lighting_synthesis_v2.py, and writes
augmented RGB frames to --out-dir/rgb/.  Also writes a patched rgb.txt.

All other files (depth, groundtruth, timestamps) are untouched and
should be symlinked into --out-dir by the calling script.

intensity_scale definition
--------------------------
  scale = 0.0  →  clean (identity, no augmentation)
  scale = 1.0  →  default lightswitch strength (same as lighting_synthesis_v2)
  scale > 1.0  →  stronger than default

The scaling is applied in log-space on the gamma_map offset and the
color_gain offset, so:
  - scale=0 always gives gamma_map=1 (identity) and color_gain=[1,1,1]
  - scale=1 reproduces the original mode_lightswitch exactly
  - the effect is always non-linear and spatially varying (not affine)

Usage
-----
  cd /home/melody/code/individual_project
  python apply_lightswitch_online.py \
      --dataset-dir /path/to/fr1_desk \
      --out-dir     /tmp/como_aug_scale_0.75 \
      --intensity-scale 0.75 \
      --seed 42
"""

import argparse
import os
import sys
import numpy as np
import cv2

# ── import helpers from lighting_synthesis_v2 ────────────────────────────────
# Both scripts must live in the same directory.
# Default location: /home/melody/code/individual_project/
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from lighting_synthesis_v2 import (
    read_index,
    associate,
    load_rgb,
    load_depth_m,
    compute_guidance_maps,
    make_event_schedule,
    event_envelope,
    active_event_index,
    event_progress,
)

DEPTH_FACTOR = 5000.0


def mode_lightswitch_scaled(rgb, depth_m, t, events, intensity_scale):
    """
    Scaled version of mode_lightswitch.

    The gamma_map offset and color_gain offset are scaled in log-space:
      gamma_offset_scaled = exp(intensity_scale * log(gamma_offset_default))

    This ensures:
      scale=0  → no effect (gamma=1, gains=[1,1,1])
      scale=1  → identical to original mode_lightswitch
      scale>1  → stronger effect
    """
    h, w = rgb.shape[:2]
    env = event_envelope(t, events)
    if env < 1e-4 or intensity_scale < 1e-6:
        return rgb.copy()

    guides = compute_guidance_maps(rgb, depth_m)
    idx = active_event_index(t, events)
    event = events[max(idx, 0)]
    prog = event_progress(t, event)

    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    # Spatial illumination field (identical to original)
    top_bias = 1.0 - np.clip(vs / max(1.0, h - 1.0), 0.0, 1.0)
    center_x = 0.55 * w + 0.12 * w * np.sin(1.4 * np.pi * prog + 0.5 * idx)
    center_y = 0.20 * h
    radial = np.exp(-(((us - center_x) / (0.75 * w)) ** 2 +
                      ((vs - center_y) / (0.55 * h)) ** 2))
    illum_field = cv2.GaussianBlur(0.50 * top_bias + 0.50 * radial, (0, 0), 11.0)
    illum_field = np.clip(0.65 + 0.70 * illum_field + 0.22 * guides['smooth'], 0.4, 1.65)

    rgb01 = np.clip(rgb, 0, 255) / 255.0
    event_peak = event[2]

    if idx % 2 == 0:
        # Light on: bright, washed, spatially uneven
        # Default gamma offset: env * event_peak * (0.72 + 0.32 * illum_field)
        # Scale in log-space: offset_scaled = exp(s * log(offset_default))
        gamma_offset_default = env * event_peak * (0.72 + 0.32 * illum_field)
        # log-space scale: for offset near 0, use linear approximation
        gamma_offset_scaled = np.expm1(
            intensity_scale * np.log1p(gamma_offset_default)
        )
        gamma_map = np.clip(1.0 - gamma_offset_scaled, 0.15, 1.0)
        out = rgb01 ** gamma_map[..., None]

        brightness_boost_default = env * 0.32 * illum_field
        brightness_boost_scaled = np.expm1(
            intensity_scale * np.log1p(brightness_boost_default)
        )
        out *= (1.0 + brightness_boost_scaled[..., None])

        # Color gain: scale offset in log-space
        # default gains: [1.05, 1.02, 0.96] → offsets: [+0.05, +0.02, -0.04]
        gain_offsets_default = np.array([0.05, 0.02, -0.04], dtype=np.float32)
        # For positive offsets: exp(s * log(1 + offset)) - 1
        # For negative offsets: -(exp(s * log(1 + |offset|)) - 1)
        gain_offsets_scaled = np.sign(gain_offsets_default) * np.expm1(
            intensity_scale * np.log1p(np.abs(gain_offsets_default))
        )
        color_gain = 1.0 + gain_offsets_scaled

    else:
        # Light off: dark, shadow crushing
        gamma_offset_default = env * event_peak * (1.20 + 0.55 * (1.35 - illum_field))
        gamma_offset_scaled = np.expm1(
            intensity_scale * np.log1p(gamma_offset_default)
        )
        gamma_map = np.clip(1.0 + gamma_offset_scaled, 1.0, 3.5)
        out = rgb01 ** gamma_map[..., None]

        dark_offset_default = env * 0.42 * (1.25 - 0.55 * illum_field)
        dark_offset_scaled = np.expm1(
            intensity_scale * np.log1p(dark_offset_default)
        )
        out *= np.clip(1.0 - dark_offset_scaled[..., None], 0.0, 1.0)

        gain_offsets_default = np.array([-0.08, -0.03, 0.06], dtype=np.float32)
        gain_offsets_scaled = np.sign(gain_offsets_default) * np.expm1(
            intensity_scale * np.log1p(np.abs(gain_offsets_default))
        )
        color_gain = 1.0 + gain_offsets_scaled

    out = np.clip(out * color_gain[None, None, :], 0.0, 1.0)
    return (255.0 * out).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Online lightswitch augmentation with intensity_scale')
    parser.add_argument('--dataset-dir', required=True, help='Original TUM dataset directory')
    parser.add_argument('--out-dir', required=True, help='Output directory for augmented RGB + patched rgb.txt')
    parser.add_argument('--intensity-scale', type=float, default=1.0,
                        help='Intensity scale: 0=clean, 1=default lightswitch, >1=stronger')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for event schedule')
    parser.add_argument('--fps', type=float, default=30.0, help='Sequence frame rate')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    out_dir = args.out_dir
    scale = args.intensity_scale

    # ── Load index files ──────────────────────────────────────────────────────
    # Prefer matched_rgb.txt if it exists (pre-processed)
    rgb_index_path = os.path.join(dataset_dir, 'matched_rgb.txt')
    if not os.path.exists(rgb_index_path):
        rgb_index_path = os.path.join(dataset_dir, 'rgb.txt')

    depth_index_path = os.path.join(dataset_dir, 'matched_depth.txt')
    if not os.path.exists(depth_index_path):
        depth_index_path = os.path.join(dataset_dir, 'depth.txt')
        if not os.path.exists(depth_index_path):
            depth_index_path = None

    rgb_entries = read_index(rgb_index_path)
    depth_entries = read_index(depth_index_path) if depth_index_path else []
    pairs = associate(rgb_entries, depth_entries) if depth_entries else \
            [(ts, fname, None) for ts, fname in rgb_entries]

    timestamps = np.array([ts for ts, _, _ in pairs], dtype=np.float64)
    t0 = pairs[0][0]
    total_t = pairs[-1][0] - t0

    rng = np.random.default_rng(args.seed)
    events = make_event_schedule(total_t, rng, 'lightswitch', args.fps)

    print(f"[LightswitchAug] scale={scale:.2f}, {len(pairs)} frames, {total_t:.1f}s, {len(events)} events", file=sys.stderr, flush=True)

    # ── Process frames ────────────────────────────────────────────────────────
    rgb_out_dir = os.path.join(out_dir, 'rgb')
    os.makedirs(rgb_out_dir, exist_ok=True)

    index_lines = [
        f'# augmented lightswitch RGB (scale={scale:.2f}, seed={args.seed})\n',
        '# timestamp filename\n',
        '#\n',
    ]

    for i, (ts, rgb_fname, depth_fname) in enumerate(pairs):
        t = ts - t0
        rgb = load_rgb(dataset_dir, rgb_fname)
        depth_m = load_depth_m(dataset_dir, depth_fname) if depth_fname else None

        if scale < 1e-6:
            out = rgb.copy()
        else:
            out = mode_lightswitch_scaled(rgb, depth_m, t, events, scale)

        base_name = os.path.basename(rgb_fname)
        out_path = os.path.join(rgb_out_dir, base_name)
        cv2.imwrite(out_path, out.astype(np.uint8))

        rel_path = f'rgb/{base_name}'
        index_lines.append(f'{ts:.6f} {rel_path}\n')

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(pairs)}] processed", file=sys.stderr, flush=True)

    # ── Write patched rgb.txt ─────────────────────────────────────────────────
    rgb_txt_out = os.path.join(out_dir, 'rgb.txt')
    with open(rgb_txt_out, 'w') as f:
        f.writelines(index_lines)

    print(f"[LightswitchAug] Done. rgb.txt written to {rgb_txt_out}", file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
