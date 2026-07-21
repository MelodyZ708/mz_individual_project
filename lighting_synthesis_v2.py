"""
lighting_synthesis_v2.py
========================
Realistic lighting-change synthesis for TUM RGB-D sequences (e.g. fr1/desk).

Usage
-----
  python lighting_synthesis_v2.py \
      --root /path/to/fr1_desk \
      --out-root /path/to/output \
      --modes lightswitch flashlight gamma specular \
      --fps 30 \
      --seed 42
"""

import argparse
import os
import numpy as np
import cv2

# ── camera intrinsics (TUM freiburg1) ─────────────────────────────────────────
FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3
DEPTH_FACTOR = 5000.0          # uint16 raw / 5000 = metres

# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_index(path):
    entries = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            entries.append((float(parts[0]), parts[1]))
    return entries


def associate(rgb_entries, depth_entries, max_diff=0.02):
    depth_ts = np.array([d[0] for d in depth_entries])
    pairs = []
    for ts, fname in rgb_entries:
        idx = np.argmin(np.abs(depth_ts - ts))
        if abs(depth_ts[idx] - ts) <= max_diff:
            pairs.append((ts, fname, depth_entries[idx][1]))
        else:
            pairs.append((ts, fname, None))
    return pairs


def load_rgb(root, fname):
    im = cv2.imread(os.path.join(root, fname), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Cannot read {os.path.join(root, fname)}")
    return im.astype(np.float32)


def load_depth_m(root, fname):
    if fname is None:
        return None
    d = cv2.imread(os.path.join(root, fname), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    return d.astype(np.float32) / DEPTH_FACTOR


# ── noise helpers ─────────────────────────────────────────────────────────────

def fractal_noise(h, w, scale=4, octaves=3, persistence=0.5, seed=0):
    rng = np.random.default_rng(seed)
    total = np.zeros((h, w), dtype=np.float32)
    amp, amp_sum = 1.0, 0.0
    freq = scale
    for _ in range(octaves):
        gh = max(2, h // freq + 2)
        gw = max(2, w // freq + 2)
        grid = rng.random((gh, gw)).astype(np.float32)
        grid_up = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
        total += grid_up * amp
        amp_sum += amp
        amp *= persistence
        freq = max(1, freq // 2)
    total /= amp_sum
    total -= total.min()
    total /= (total.max() + 1e-8)
    return total


def smoothstep(edge0, edge1, x):
    x = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def compute_guidance_maps(rgb, depth_m):
    gray = rgb.mean(axis=-1).astype(np.float32)
    gray01 = np.clip(gray / 255.0, 0.0, 1.0)

    local_mean = cv2.GaussianBlur(gray01, (0, 0), 9.0)
    local_sq_mean = cv2.GaussianBlur(gray01 * gray01, (0, 0), 9.0)
    local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0.0)
    var_scale = np.percentile(local_var, 95) + 1e-6
    texture_map = np.clip(local_var / var_scale, 0.0, 1.0)
    texture_map = cv2.GaussianBlur(texture_map, (0, 0), 3.0)
    smooth_map = 1.0 - texture_map

    grad_x = cv2.Sobel(gray01, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray01, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    grad_scale = np.percentile(grad, 95) + 1e-6
    edge_map = np.clip(grad / grad_scale, 0.0, 1.0)

    if depth_m is not None:
        d = depth_m.copy()
        valid = d > 0.05
        if np.any(valid):
            fill = np.median(d[valid])
            d[~valid] = fill
        else:
            d[:] = 1.5
        dx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        depth_grad = np.sqrt(dx * dx + dy * dy)
        dg_scale = np.percentile(depth_grad[valid], 90) + 1e-6 if np.any(valid) else 1.0
        planar_map = 1.0 - np.clip(depth_grad / dg_scale, 0.0, 1.0)
        planar_map *= valid.astype(np.float32)
    else:
        planar_map = np.ones_like(gray01, dtype=np.float32)

    return {
        'gray': gray,
        'gray01': gray01,
        'texture': texture_map,
        'smooth': smooth_map,
        'edge': edge_map,
        'planar': cv2.GaussianBlur(planar_map, (0, 0), 3.0),
    }


def active_event_index(t, events):
    for idx, (t_start, t_end, _) in enumerate(events):
        if t_start <= t <= t_end:
            return idx
    return -1


def event_progress(t, event):
    t_start, t_end, _ = event
    dur = max(t_end - t_start, 1e-6)
    return np.clip((t - t_start) / dur, 0.0, 1.0)


def mode_schedule_params(mode):
    presets = {
        'flashlight': {
            'first_quiet': (0.4, 1.0),
            'duration': (1.2, 2.6),
            'gap': (0.5, 1.2),
            'intensity': (0.80, 1.15),
        },
        'gamma': {
            'first_quiet': (0.5, 1.2),
            'duration': (1.0, 2.0),
            'gap': (0.8, 1.8),
            'intensity': (0.75, 1.00),
        },
        'specular': {
            'first_quiet': (0.3, 0.8),
            'duration': (0.9, 2.0),
            'gap': (0.4, 1.0),
            'intensity': (0.75, 1.05),
        },
        'lightswitch': {
            'first_quiet': (0.5, 1.2),
            'duration': (0.8, 1.8),
            'gap': (0.4, 1.1),
            'intensity': (0.80, 1.05),
        },
    }
    return presets.get(mode, {
        'first_quiet': (0.8, 1.5),
        'duration': (1.0, 2.2),
        'gap': (1.0, 2.0),
        'intensity': (0.60, 0.95),
    })


# ── event schedule ────────────────────────────────────────────────────────────

def make_event_schedule(total_t, rng, mode, fps=30):
    params = mode_schedule_params(mode)
    first_quiet_lo, first_quiet_hi = params['first_quiet']
    dur_lo, dur_hi = params['duration']
    gap_lo, gap_hi = params['gap']
    inten_lo, inten_hi = params['intensity']

    events = []
    t = rng.uniform(first_quiet_lo, first_quiet_hi)
    min_tail = max(0.8, dur_lo * 0.5)
    while t < total_t - min_tail:
        duration = rng.uniform(dur_lo, dur_hi)
        intensity = rng.uniform(inten_lo, inten_hi)
        t_end = min(t + duration, total_t - 0.2)
        if t_end - t < 0.3:
            break
        events.append((t, t_end, float(intensity)))
        gap = rng.uniform(gap_lo, gap_hi)
        t = t_end + gap
    return events


def event_envelope(t, events, ramp=0.4):
    val = 0.0
    for t_start, t_end, peak in events:
        if t < t_start or t > t_end:
            continue
        dur = t_end - t_start
        if dur <= 0:
            continue
        r = min(ramp, dur * 0.3)
        if t - t_start < r:
            alpha = (t - t_start) / r
        elif t_end - t < r:
            alpha = (t_end - t) / r
        else:
            alpha = 1.0
        val = max(val, alpha * peak)
    return val


def event_frame_span(t0, timestamps, t_start, t_end):
    rel_ts = timestamps - t0
    start_idx = int(np.searchsorted(rel_ts, t_start, side='left'))
    end_idx = int(np.searchsorted(rel_ts, t_end, side='right') - 1)
    start_idx = int(np.clip(start_idx, 0, len(timestamps) - 1))
    end_idx = int(np.clip(end_idx, start_idx, len(timestamps) - 1))
    return start_idx, end_idx


def write_event_schedule(out_root, events_by_mode, timestamps, t0, filename='event_schedule.txt'):
    schedule_rows = []
    for mode_name, evs in events_by_mode.items():
        for idx, (t_start, t_end, peak) in enumerate(evs):
            start_frame, end_frame = event_frame_span(t0, timestamps, t_start, t_end)
            schedule_rows.append({
                'mode': mode_name,
                'event_id': idx,
                't_start': t_start,
                't_end': t_end,
                'duration': t_end - t_start,
                'peak': peak,
                'start_frame': start_frame,
                'end_frame': end_frame,
            })

    schedule_rows.sort(key=lambda row: (row['t_start'], row['t_end'], row['mode'], row['event_id']))

    lines = ['# lighting event schedule sorted by start time\n']
    lines.append(
        '# start_frame end_frame mode event_id start_s end_s duration_s peak_intensity\n'
    )
    for row in schedule_rows:
        lines.append(
            f"{row['start_frame']} {row['end_frame']} {row['mode']} {row['event_id']} "
            f"{row['t_start']:.3f} {row['t_end']:.3f} {row['duration']:.3f} {row['peak']:.3f}\n"
        )

    with open(os.path.join(out_root, filename), 'w') as f:
        f.writelines(lines)


def write_video(video_path, frames, fps):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f'Failed to open video writer for {video_path}')
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


# ── mode implementations ──────────────────────────────────────────────────────

def mode_flashlight(rgb, depth_m, t, events, rng_state):
    h, w = rgb.shape[:2]
    env = event_envelope(t, events)
    if env < 1e-4:
        return rgb.copy()

    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    if depth_m is not None:
        d = depth_m.copy()
        invalid = d <= 0.05
        d[invalid] = 1.5
        d = np.clip(d, 0.3, 6.0)
        atten = 1.0 / (d ** 2)
        atten = atten / (atten.max() + 1e-8)
    else:
        atten = np.ones((h, w), dtype=np.float32)

    idx = active_event_index(t, events)
    event = events[max(idx, 0)]
    prog = event_progress(t, event)

    n_sources = 2 + (idx % 2)
    burst = np.zeros((h, w), dtype=np.float32)
    for j in range(n_sources):
        phase = prog + 0.23 * j
        a = 0.8 + 0.25 * j
        b = 1.1 + 0.20 * j
        xL = w * (0.5 + (0.24 + 0.05 * j) * np.sin(2 * np.pi * phase * a + 0.8 * j))
        yL = h * (0.5 + (0.18 + 0.04 * j) * np.sin(2 * np.pi * phase * b + 1.2 + 0.6 * j))
        sigma = (0.08 + 0.02 * j) * min(h, w)
        gauss = np.exp(-((us - xL) ** 2 + (vs - yL) ** 2) / (2 * sigma ** 2))
        burst += (1.0 - 0.18 * j) * gauss

    burst = np.clip(burst, 0.0, 1.75)
    alpha = 1.45 * env
    factor = 1.0 + alpha * burst * atten
    out = np.clip(rgb * factor[..., None], 0, 255)
    return out.astype(np.float32)


def mode_shadows(rgb, base_noise, t, events, params=None):
    if params is None:
        params = {}
    h, w = rgb.shape[:2]
    ch, cw = base_noise.shape
    env = event_envelope(t, events)
    if env < 1e-4:
        return rgb.copy()

    speed_x = params.get('speed_x', 60.0)
    speed_y = params.get('speed_y', 40.0)
    ox = int(speed_x * t) % max(1, cw - w)
    oy = int(speed_y * t) % max(1, ch - h)
    n = base_noise[oy:oy + h, ox:ox + w]

    threshold = params.get('threshold', 0.62)
    softness  = params.get('softness',  0.15)
    m = np.clip((threshold - n) / (softness + 1e-8), 0, 1)
    m = m * m * (3 - 2 * m)

    ksize = max(3, int(0.05 * min(h, w)) | 1)
    m = cv2.GaussianBlur(m, (ksize, ksize), 0)

    beta = 0.60 * env
    out = rgb * (1.0 - beta * m[..., None])
    return np.clip(out, 0, 255).astype(np.float32)


def mode_gamma(rgb, t, events):
    env = event_envelope(t, events)
    if env < 1e-4:
        return rgb.copy()

    active_event = None
    for ev in events:
        t_start, t_end, peak = ev
        if t_start <= t <= t_end:
            active_event = ev
            break

    if active_event is None:
        return rgb.copy()

    idx = events.index(active_event)
    env_peak = active_event[2]
    if idx % 2 == 0:
        g_target = 0.28
    else:
        g_target = 2.45

    g = 1.0 + (g_target - 1.0) * env
    g = np.clip(g, 0.3, 2.5)
    rgb01 = np.clip(rgb, 0, 255) / 255.0

    out = rgb01 ** g

    if idx % 2 == 0:
        out = np.sqrt(np.clip(out, 0.0, 1.0))
    else:
        out = np.clip(1.15 * out - 0.15 * out * out, 0.0, 1.0) ** (1.0 + 0.35 * env_peak)

    out = 255.0 * np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def mode_specular(rgb, depth_m, t, events):
    h, w = rgb.shape[:2]
    env = event_envelope(t, events)
    if env < 1e-4:
        return rgb.copy()

    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

    guides = compute_guidance_maps(rgb, depth_m)
    gray = guides['gray']
    reflective_mask = (gray > 70).astype(np.float32)
    reflective_mask *= 0.55 * guides['smooth'] + 0.45 * guides['planar']
    reflective_mask = cv2.GaussianBlur(reflective_mask, (0, 0), 7.0)

    idx = active_event_index(t, events)
    event = events[max(idx, 0)]
    prog = event_progress(t, event)
    n_streaks = 2 + (idx % 3 == 0)
    blob = np.zeros((h, w), dtype=np.float32)
    for j in range(n_streaks):
        x_frac = 0.12 + 0.76 * ((prog * (1.1 + 0.25 * j) + 0.17 * j) % 1.0)
        y_frac = 0.48 + 0.18 * np.sin(2.2 * np.pi * prog + 0.8 * j)
        xS = x_frac * w
        yS = y_frac * h
        sx = (0.12 + 0.03 * j) * w
        sy = (0.025 + 0.01 * j) * h
        streak = np.exp(-((us - xS) ** 2 / (2 * sx ** 2) +
                          (vs - yS) ** 2 / (2 * sy ** 2)))
        blob += streak

    blob = np.clip(blob, 0.0, 1.8)
    highlight = blob * reflective_mask
    lam = 235.0 * env
    out = np.clip(rgb + lam * highlight[..., None], 0, 255)
    return out.astype(np.float32)


def mode_lightswitch(rgb, depth_m, t, events):
    h, w = rgb.shape[:2]
    env = event_envelope(t, events)
    if env < 1e-4:
        return rgb.copy()

    guides = compute_guidance_maps(rgb, depth_m)
    idx = active_event_index(t, events)
    event = events[max(idx, 0)]
    prog = event_progress(t, event)
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))

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
        gamma_map = 1.0 - env * event_peak * (0.72 + 0.32 * illum_field)
        gamma_map = np.clip(gamma_map, 0.22, 1.0)
        out = rgb01 ** gamma_map[..., None]
        out *= (1.0 + env * 0.32 * illum_field[..., None])
        color_gain = np.array([1.05, 1.02, 0.96], dtype=np.float32)
    else:
        gamma_map = 1.0 + env * event_peak * (1.20 + 0.55 * (1.35 - illum_field))
        gamma_map = np.clip(gamma_map, 1.0, 2.8)
        out = rgb01 ** gamma_map[..., None]
        out *= (1.0 - env * 0.42 * (1.25 - 0.55 * illum_field[..., None]))
        color_gain = np.array([0.92, 0.97, 1.06], dtype=np.float32)

    out = np.clip(out * color_gain[None, None, :], 0.0, 1.0)
    return (255.0 * out).astype(np.float32)


# ── sequence runner ───────────────────────────────────────────────────────────

MODES = ['flashlight', 'gamma', 'specular', 'lightswitch']


def run(root, out_root, modes, test_n=None, fps=30, seed=0):
    rgb_entries   = read_index(os.path.join(root, 'rgb.txt'))
    depth_entries = read_index(os.path.join(root, 'depth.txt'))
    pairs = associate(rgb_entries, depth_entries)

    if test_n:
        idxs = np.linspace(0, len(pairs) - 1, test_n).astype(int)
        pairs = [pairs[i] for i in idxs]

    timestamps = np.array([ts for ts, _, _ in pairs], dtype=np.float64)
    t0       = pairs[0][0]
    total_t  = pairs[-1][0] - t0
    rng      = np.random.default_rng(seed)

    events = {mode: make_event_schedule(total_t, rng, mode, fps) for mode in modes}

    print(f"Sequence: {len(pairs)} frames, {total_t:.1f} s")
    for mode_name, evs in events.items():
        print(f"  events[{mode_name}]: {len(evs)} events")

    original_frames = []
    for _, rgb_fname, _ in pairs:
        original_frames.append(load_rgb(root, rgb_fname).astype(np.uint8))
    original_video_path = os.path.join(out_root, 'rgb_original.mp4')
    write_video(original_video_path, original_frames, fps)
    print(f'  wrote original video -> {original_video_path}')

    for mode in modes:
        out_dir = os.path.join(out_root, f'rgb_{mode}')
        os.makedirs(out_dir, exist_ok=True)
        write_event_schedule(
            out_dir,
            {mode: events[mode]},
            timestamps,
            t0,
            filename='event.txt',
        )
        index_lines = [
            f'# synthesized {mode} lighting (derived from {root})\n',
            '# timestamp filename\n',
        ]
        video_frames = []

        for ts, rgb_fname, depth_fname in pairs:
            t = ts - t0
            rgb     = load_rgb(root, rgb_fname)
            depth_m = load_depth_m(root, depth_fname) if depth_fname else None

            if mode == 'flashlight':
                out = mode_flashlight(rgb, depth_m, t, events[mode], rng)
            elif mode == 'gamma':
                out = mode_gamma(rgb, t, events[mode])
            elif mode == 'specular':
                out = mode_specular(rgb, depth_m, t, events[mode])
            elif mode == 'lightswitch':
                out = mode_lightswitch(rgb, depth_m, t, events[mode])
            else:
                raise ValueError(f"Unknown mode: {mode}")

            base_name = os.path.basename(rgb_fname)
            out_rel   = f'rgb_{mode}/{base_name}'
            out_u8 = out.astype(np.uint8)
            cv2.imwrite(os.path.join(out_root, out_rel), out_u8)
            video_frames.append(out_u8)
            index_lines.append(f'{ts:.6f} {out_rel}\n')

        idx_path = os.path.join(out_root, f'rgb_{mode}.txt')
        with open(idx_path, 'w') as f:
            f.writelines(index_lines)
        video_path = os.path.join(out_root, f'rgb_{mode}.mp4')
        write_video(video_path, video_frames, fps)
        print(f'[{mode}] wrote {len(pairs)} frames -> {out_dir}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Synthesise realistic lighting changes on TUM RGB-D sequences.')
    ap.add_argument('--root',     required=True,
                    help='Path to sequence root (contains rgb/, depth/, rgb.txt, depth.txt)')
    ap.add_argument('--out-root', default=None,
                    help='Output directory. Defaults to <root>/synth_lighting_output')
    ap.add_argument('--modes',    nargs='+', default=MODES,
                    choices=MODES, help='Which modes to generate')
    ap.add_argument('--test-n',   type=int, default=None,
                    help='Process only N evenly-spaced frames (for quick testing)')
    ap.add_argument('--fps',      type=float, default=30.0,
                    help='Nominal frame rate of the sequence (used for event timing)')
    ap.add_argument('--seed',     type=int, default=42)
    args = ap.parse_args()

    if args.out_root is None:
        args.out_root = os.path.join(args.root, 'synth_lighting_output')

    os.makedirs(args.out_root, exist_ok=True)
    run(args.root, args.out_root, args.modes,
        test_n=args.test_n, fps=args.fps, seed=args.seed)
