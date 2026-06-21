"""
compute_bqs_unet.py
===================
BQS (Basin Quality Score) evaluation for U-Net encoder channels.
Scores each channel individually — NO greedy search.

BQS formula (identical to analyze_basin_metrics.py v5.7):
  basin_quality = 0.75 * LQBS + 0.25 * Width
  retention     = shape_similarity × sharpness_retention × minpos_consistency × bright_symmetry
  BQS           = basin_quality × weighted_average_retention
    where weighted_average_retention = 0.4 * retention_30 + 0.6 * retention_50

Outputs:
  vis_results/bqs_unet_enc{N}/bqs_scores.csv        -- per-channel BQS + sub-metrics
  vis_results/bqs_unet_enc{N}/bqs_ranking.png        -- bar chart sorted by BQS
  vis_results/bqs_unet_enc{N}/bqs_vs_resnet.png      -- comparison with ResNet Conv1 (if csv provided)

Usage:
  cd /vol/bitbucket/mz325/individual_project/como
  python ../compute_bqs_unet.py --enc_level 1 [--frame 306]
  python ../compute_bqs_unet.py --enc_level 0 [--frame 306]

  # Optional: compare with ResNet Conv1 BQS results
  python ../compute_bqs_unet.py --enc_level 1 \
      --resnet_csv ../vis_results/basin_metrics/channel_ranking.csv
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import glob
import argparse
import csv
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

from como.depth_cov.core.DepthCovModule import DepthCovModule

# ============================================================
# BQS Configuration (identical to analyze_basin_metrics.py v5.7)
# ============================================================
BQS_CONFIG = {
    'grid_range':               30,
    'grid_step':                1,
    'local_radius':             10,
    'retention_local_radius':   10,
    'width_threshold':          0.20,
    'basin_weights': {
        'local_quadratic_bowl': 0.75,
        'basin_width':          0.25,
    },
    'convexity_scale':          0.002,
    'dead_threshold':           1e-8,
    # Normalisation bounds from ResNet Conv1 single-channel analysis
    # (reuse same bounds for fair comparison)
    'MAX_LQBS':  0.55,
    'MAX_WIDTH': 10.0,
    # Retention weights
    'ret_w30': 0.4,
    'ret_w50': 0.6,
}

CONDITIONS = [
    ('clean',    0.0),
    ('bright30', 0.3),
    ('bright50', 0.5),
]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.titlesize': 11,
    'figure.dpi': 120,
    'savefig.dpi': 200,
})


# ============================================================
# U-Net Feature Extractor (same as visualize_basin_unet.py)
# ============================================================
class UNetEncoderExtractor:
    def __init__(self, model_path: str, enc_level: int = 1, device: str = 'cuda:0'):
        self.device    = device
        self.enc_level = enc_level

        print(f'[INFO] Loading DepthCovModule from {model_path} ...')
        module = DepthCovModule.load_from_checkpoint(model_path, map_location=device)
        module.eval()
        module.to(device)
        for p in module.parameters():
            p.requires_grad = False

        self.unet = module.gaussian_cov_net
        self.mean = IMAGENET_MEAN.to(device)
        self.std  = IMAGENET_STD.to(device)
        print(f'[INFO] U-Net loaded. enc_level={enc_level}')

    @torch.no_grad()
    def extract_all_channels(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        orig_size = rgb_tensor.shape[-2:]
        x = (rgb_tensor - self.mean) / self.std
        x = self.unet.base(x)
        if self.enc_level == 0:
            return x
        x = self.unet.down_convs[0](x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x


# ============================================================
# Image Utilities
# ============================================================
def load_image_numpy(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0

def numpy_to_tensor(img, device):
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

def apply_brightness(img, factor):
    return img.copy() if factor == 0.0 else np.clip(img + factor, 0.0, 1.0)


# ============================================================
# Cost Landscape (GPU-accelerated via grid_sample)
# ============================================================
@torch.no_grad()
def compute_cost_landscape_gpu(feat_ref_np, feat_tgt_np, dx_vals, dy_vals, device):
    """
    feat_ref_np, feat_tgt_np: [H, W] single channel numpy arrays
    Returns cost_grid: [N_dy, N_dx] numpy array
    """
    H, W = feat_ref_np.shape
    ref = torch.from_numpy(feat_ref_np).float().to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    tgt = torch.from_numpy(feat_tgt_np).float().to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    step_x = 2.0 / (W - 1)
    step_y = 2.0 / (H - 1)

    N_dy, N_dx = len(dy_vals), len(dx_vals)
    cost_grid = np.zeros((N_dy, N_dx), dtype=np.float64)

    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

    for i, dy in enumerate(dy_vals):
        for j, dx_val in enumerate(dx_vals):
            g = base.clone()
            g[..., 0] += dx_val * step_x
            g[..., 1] += dy * step_y
            g = g.unsqueeze(0)  # [1, H, W, 2]
            tgt_shifted = F.grid_sample(tgt, g, mode='bilinear',
                                        padding_mode='border', align_corners=True)
            diff = tgt_shifted - ref
            cost_grid[i, j] = float(diff.pow(2).mean().cpu())

    return cost_grid


# ============================================================
# BQS Sub-metric Functions (identical to analyze_basin_metrics.py)
# ============================================================
def clamp01(x):
    return float(max(0.0, min(1.0, x)))

def normalize_grid(grid):
    gmin, gmax = float(np.min(grid)), float(np.max(grid))
    if gmax - gmin < BQS_CONFIG['dead_threshold']:
        return np.zeros_like(grid, dtype=np.float64), False
    return (grid.astype(np.float64) - gmin) / (gmax - gmin), True

def extract_local_patch(grid, radius):
    h, w = grid.shape
    cy, cx = h // 2, w // 2
    si = slice(max(0, cy - radius), min(h, cy + radius + 1))
    sj = slice(max(0, cx - radius), min(w, cx + radius + 1))
    return grid[si, sj]

def safe_corr(a, b):
    a = a.ravel().astype(np.float64); a -= a.mean()
    b = b.ravel().astype(np.float64); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return clamp01(np.dot(a, b) / denom) if denom > 1e-12 else 0.0

def compute_quadratic_fit(cost_grid, dx, dy):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    DX, DY = np.meshgrid(dx, dy)
    X = np.column_stack([DX.ravel(), DY.ravel()])
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    y = cost_grid.ravel()
    if np.std(y) < 1e-12:
        return 0.0, 0.0
    reg = LinearRegression().fit(X_poly, y)
    y_pred = reg.predict(X_poly)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = max(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0, 0.0)
    coef = reg.coef_
    H_mat = np.array([[2 * coef[3], coef[4]], [coef[4], 2 * coef[5]]])
    return r2, float(np.min(np.linalg.eigvalsh(H_mat)))

def compute_symmetry_score(cost_grid, radius):
    patch = extract_local_patch(cost_grid, radius)
    if patch.size == 0: return 0.0
    dyn = float(np.max(patch) - np.min(patch))
    if dyn < 1e-12: return 0.0
    return clamp01(1.0 - float(np.mean(np.abs(patch - np.rot90(patch, 2)))) / dyn)

def compute_lqbs(grid_norm, dx, dy):
    radius = BQS_CONFIG['local_radius']
    ci, cj = len(dy) // 2, len(dx) // 2
    si = slice(max(0, ci - radius), min(len(dy), ci + radius + 1))
    sj = slice(max(0, cj - radius), min(len(dx), cj + radius + 1))
    local_grid = grid_norm[si, sj]
    local_dx   = dx[sj]
    local_dy   = dy[si]
    r2, min_eigval = compute_quadratic_fit(local_grid, local_dx, local_dy)
    sym   = compute_symmetry_score(grid_norm, radius)
    scale = BQS_CONFIG['convexity_scale']
    conv  = min_eigval / (min_eigval + scale) if min_eigval > 0 else 0.0
    return r2 * conv * sym

def compute_basin_width(grid_norm, threshold):
    h, w = grid_norm.shape
    cy, cx = h // 2, w // 2
    if np.std(grid_norm) < 1e-12: return 0.0
    mask = grid_norm <= threshold
    if not mask[cy, cx]: return 0.0
    visited = np.zeros_like(mask, dtype=bool)
    q = deque([(cy, cx)]); visited[cy, cx] = True
    component = []
    for dy0, dx0 in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        pass  # init
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while q:
        y, x = q.popleft(); component.append((y, x))
        for dy0, dx0 in neighbors:
            ny, nx = y + dy0, x + dx0
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True; q.append((ny, nx))
    dists = [np.sqrt((y - cy) ** 2 + (x - cx) ** 2) for y, x in component]
    return float(max(dists)) if dists else 0.0

def compute_sharpness(grid_norm, dx, dy):
    h, w = grid_norm.shape
    cy, cx = h // 2, w // 2
    if cx <= 0 or cx >= w - 1 or cy <= 0 or cy >= h - 1: return 0.0
    sx = float(dx[1] - dx[0]) if len(dx) > 1 else 1.0
    sy = float(dy[1] - dy[0]) if len(dy) > 1 else 1.0
    fxx = (grid_norm[cy, cx+1] - 2*grid_norm[cy, cx] + grid_norm[cy, cx-1]) / (sx**2)
    fyy = (grid_norm[cy+1, cx] - 2*grid_norm[cy, cx] + grid_norm[cy-1, cx]) / (sy**2)
    return float(np.sqrt(fxx * fyy)) if fxx > 0 and fyy > 0 else 0.0

def compute_shape_sim(clean_norm, bright_norm, radius):
    if np.std(clean_norm) < 1e-12 or np.std(bright_norm) < 1e-12: return 0.0
    g_corr = safe_corr(clean_norm, bright_norm)
    c_local = extract_local_patch(clean_norm, radius)
    b_local = extract_local_patch(bright_norm, radius)
    return 0.4 * g_corr + 0.6 * safe_corr(c_local, b_local)

def compute_minpos_consistency(grid_norm, radius):
    h, w = grid_norm.shape
    cy, cx = h // 2, w // 2
    patch = extract_local_patch(grid_norm, radius)
    if patch.size == 0: return 0.0
    py, px = np.unravel_index(np.argmin(patch), patch.shape)
    lcy, lcx = patch.shape[0] // 2, patch.shape[1] // 2
    max_dist = np.sqrt(lcy**2 + lcx**2)
    if max_dist < 1e-12: return 1.0
    return clamp01(1.0 - np.sqrt((py - lcy)**2 + (px - lcx)**2) / max_dist)


# ============================================================
# Full BQS computation for one channel
# ============================================================
def compute_channel_bqs(grids_by_cond, dx, dy):
    """
    grids_by_cond: dict {'clean': ndarray, 'bright30': ndarray, 'bright50': ndarray}
    Returns dict with BQS and all sub-metrics.
    """
    cfg = BQS_CONFIG
    norm_grids = {}
    for cond, grid in grids_by_cond.items():
        ng, valid = normalize_grid(grid)
        if not valid:
            return {
                'bqs_score': 0.0, 'basin_quality': 0.0,
                'lqbs': 0.0, 'basin_width': 0.0,
                'retention_30': 0.0, 'retention_50': 0.0,
                'shape_sim_30': 0.0, 'shape_sim_50': 0.0,
                'sharp_ret_30': 0.0, 'sharp_ret_50': 0.0,
                'minpos_30': 0.0, 'minpos_50': 0.0,
                'bright_sym_30': 0.0, 'bright_sym_50': 0.0,
                'dead': True,
            }
        norm_grids[cond] = ng

    clean = norm_grids['clean']
    lqbs  = compute_lqbs(clean, dx, dy)
    width = compute_basin_width(clean, cfg['width_threshold'])
    sharp_clean = compute_sharpness(clean, dx, dy)

    results = {'dead': False}
    ret_avg = 0.0
    for w_ret, cond in [(cfg['ret_w30'], 'bright30'), (cfg['ret_w50'], 'bright50')]:
        b_grid    = norm_grids[cond]
        shape_sim = compute_shape_sim(clean, b_grid, cfg['retention_local_radius'])
        sharp_b   = compute_sharpness(b_grid, dx, dy)
        sharp_ret = clamp01(sharp_b / sharp_clean) if sharp_clean > 1e-12 else 0.0
        minpos    = compute_minpos_consistency(b_grid, cfg['retention_local_radius'])
        sym       = compute_symmetry_score(b_grid, cfg['local_radius'])
        ret       = shape_sim * sharp_ret * minpos * sym
        ret_avg  += w_ret * ret
        suffix = '30' if cond == 'bright30' else '50'
        results[f'shape_sim_{suffix}']  = round(shape_sim, 6)
        results[f'sharp_ret_{suffix}']  = round(sharp_ret, 6)
        results[f'minpos_{suffix}']     = round(minpos, 6)
        results[f'bright_sym_{suffix}'] = round(sym, 6)
        results[f'retention_{suffix}']  = round(ret, 6)

    lqbs_norm  = lqbs  / cfg['MAX_LQBS']  if cfg['MAX_LQBS']  > 0 else lqbs
    width_norm = width / cfg['MAX_WIDTH'] if cfg['MAX_WIDTH'] > 0 else width
    bq  = (cfg['basin_weights']['local_quadratic_bowl'] * lqbs_norm +
           cfg['basin_weights']['basin_width']          * width_norm)
    bqs = bq * ret_avg

    results.update({
        'lqbs':          round(lqbs, 6),
        'basin_width':   round(width, 6),
        'basin_quality': round(bq, 6),
        'bqs_score':     round(bqs, 6),
    })
    return results


# ============================================================
# Plotting
# ============================================================
def plot_bqs_ranking(rows, output_dir, enc_level):
    rows_sorted = sorted(rows, key=lambda r: r['bqs_score'], reverse=True)
    channels = [f"UCh{r['channel']:02d}" for r in rows_sorted]
    scores   = [r['bqs_score'] for r in rows_sorted]
    dead     = [r['dead'] for r in rows_sorted]

    colors = ['#d62728' if d else '#1f77b4' for d in dead]

    fig, ax = plt.subplots(figsize=(max(10, len(channels) * 0.5), 5))
    bars = ax.bar(channels, scores, color=colors)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=1.2, label='0.05 threshold')
    ax.set_xlabel('U-Net Channel')
    ax.set_ylabel('BQS Score')
    ax.set_title(f'BQS Ranking — U-Net enc{enc_level}\n'
                 f'(formula: BQS = basin_quality × retention, red = dead channel)')
    ax.legend()
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.tight_layout()
    out = os.path.join(output_dir, 'bqs_ranking.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[INFO] BQS ranking chart → {out}')


def plot_bqs_vs_resnet(unet_rows, resnet_csv_path, output_dir, enc_level):
    """Compare U-Net BQS scores with ResNet Conv1 BQS scores."""
    import csv as csv_mod
    try:
        resnet_rows = []
        with open(resnet_csv_path, newline='') as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                resnet_rows.append({
                    'channel': int(row['channel']),
                    'bqs_score': float(row['bqs_score']),
                })
    except Exception as e:
        print(f'[WARN] Could not load ResNet CSV: {e}')
        return

    unet_mean   = np.mean([r['bqs_score'] for r in unet_rows])
    resnet_mean = np.mean([r['bqs_score'] for r in resnet_rows])
    unet_max    = max(r['bqs_score'] for r in unet_rows)
    resnet_max  = max(r['bqs_score'] for r in resnet_rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: distribution comparison
    ax = axes[0]
    unet_scores   = [r['bqs_score'] for r in unet_rows]
    resnet_scores = [r['bqs_score'] for r in resnet_rows]
    ax.hist(unet_scores,   bins=15, alpha=0.7, label=f'U-Net enc{enc_level} (n={len(unet_scores)})',   color='#1f77b4')
    ax.hist(resnet_scores, bins=15, alpha=0.7, label=f'ResNet Conv1 (n={len(resnet_scores)})', color='#ff7f0e')
    ax.axvline(unet_mean,   color='#1f77b4', linestyle='--', linewidth=1.5, label=f'U-Net mean={unet_mean:.4f}')
    ax.axvline(resnet_mean, color='#ff7f0e', linestyle='--', linewidth=1.5, label=f'ResNet mean={resnet_mean:.4f}')
    ax.set_xlabel('BQS Score')
    ax.set_ylabel('Count')
    ax.set_title('BQS Distribution: U-Net vs ResNet')
    ax.legend(fontsize=8)

    # Right: summary bar
    ax2 = axes[1]
    labels   = [f'U-Net enc{enc_level}\n(mean)', f'ResNet Conv1\n(mean)',
                f'U-Net enc{enc_level}\n(max)',  f'ResNet Conv1\n(max)']
    values   = [unet_mean, resnet_mean, unet_max, resnet_max]
    bar_colors = ['#1f77b4', '#ff7f0e', '#aec7e8', '#ffbb78']
    ax2.bar(labels, values, color=bar_colors)
    for i, v in enumerate(values):
        ax2.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9)
    ax2.set_ylabel('BQS Score')
    ax2.set_title('Summary: U-Net vs ResNet BQS')

    plt.suptitle(f'U-Net enc{enc_level} vs ResNet-18 Conv1 — BQS Comparison', fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, 'bqs_vs_resnet.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'[INFO] BQS vs ResNet comparison → {out}')


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enc_level',   type=int, default=1, choices=[0, 1])
    parser.add_argument('--frame',       type=int, default=306)
    parser.add_argument('--rgb_dir',     type=str,
                        default='/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/')
    parser.add_argument('--model_path',  type=str, default='models/scannet.ckpt')
    parser.add_argument('--device',      type=str, default='cuda:0')
    parser.add_argument('--resnet_csv',  type=str, default=None,
                        help='Path to ResNet Conv1 channel_ranking.csv for comparison plot')
    args = parser.parse_args()

    enc_level  = args.enc_level
    output_dir = f'vis_results/bqs_unet_enc{enc_level}'
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print(f'  BQS Evaluation — U-Net enc{enc_level}')
    print('=' * 60)

    # ── Load extractor ──
    extractor = UNetEncoderExtractor(
        model_path=args.model_path,
        enc_level=enc_level,
        device=args.device,
    )

    # ── Load images ──
    all_images = sorted(glob.glob(os.path.join(args.rgb_dir, '*.png')))
    if not all_images:
        print(f'[ERROR] No images in {args.rgb_dir}'); sys.exit(1)

    frame_idx  = args.frame
    rgb_np     = load_image_numpy(all_images[frame_idx])
    rgb_tensor = numpy_to_tensor(rgb_np, args.device)

    # ── Extract features for all conditions ──
    print(f'[INFO] Extracting features for frame {frame_idx}...')
    feats = {}
    for cond_name, factor in CONDITIONS:
        tgt_np     = apply_brightness(rgb_np, factor)
        tgt_tensor = numpy_to_tensor(tgt_np, args.device)
        with torch.no_grad():
            feat = extractor.extract_all_channels(tgt_tensor)
        feats[cond_name] = feat[0].cpu().numpy()  # [C, H, W]
        del tgt_tensor

    del rgb_tensor
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    num_channels = feats['clean'].shape[0]
    print(f'[INFO] Channels: {num_channels}')

    # ── Grid ──
    cfg     = BQS_CONFIG
    dx_vals = np.arange(-cfg['grid_range'], cfg['grid_range'] + 1, cfg['grid_step'])
    dy_vals = np.arange(-cfg['grid_range'], cfg['grid_range'] + 1, cfg['grid_step'])

    # ── Evaluate each channel ──
    all_rows = []
    for ch in range(num_channels):
        print(f'  Ch{ch:02d}/{num_channels-1} ...', end=' ', flush=True)
        grids = {}
        for cond_name, _ in CONDITIONS:
            feat_ref = feats['clean'][ch]
            feat_tgt = feats[cond_name][ch]
            grids[cond_name] = compute_cost_landscape_gpu(
                feat_ref, feat_tgt, dx_vals, dy_vals, args.device
            )

        metrics = compute_channel_bqs(grids, dx_vals, dy_vals)
        metrics['channel'] = ch
        all_rows.append(metrics)
        print(f"BQS={metrics['bqs_score']:.4f}  "
              f"LQBS={metrics['lqbs']:.4f}  "
              f"Width={metrics['basin_width']:.2f}  "
              f"{'[DEAD]' if metrics['dead'] else ''}")

    # ── Save CSV ──
    fieldnames = ['channel', 'bqs_score', 'basin_quality', 'lqbs', 'basin_width',
                  'retention_30', 'retention_50',
                  'shape_sim_30', 'shape_sim_50',
                  'sharp_ret_30', 'sharp_ret_50',
                  'minpos_30', 'minpos_50',
                  'bright_sym_30', 'bright_sym_50',
                  'dead']
    csv_path = os.path.join(output_dir, 'bqs_scores.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\n[INFO] BQS scores → {csv_path}')

    # ── Print ranking ──
    sorted_rows = sorted(all_rows, key=lambda r: r['bqs_score'], reverse=True)
    print('\n=== BQS Ranking ===')
    for rank, r in enumerate(sorted_rows, 1):
        dead_tag = ' [DEAD]' if r['dead'] else ''
        print(f"  #{rank:2d}  Ch{r['channel']:02d}  BQS={r['bqs_score']:.4f}  "
              f"LQBS={r['lqbs']:.4f}  Width={r['basin_width']:.2f}"
              f"  Ret30={r.get('retention_30', 0):.4f}  Ret50={r.get('retention_50', 0):.4f}"
              f"{dead_tag}")

    # ── Plots ──
    plot_bqs_ranking(all_rows, output_dir, enc_level)

    if args.resnet_csv:
        plot_bqs_vs_resnet(all_rows, args.resnet_csv, output_dir, enc_level)

    print(f'\n[DONE] All outputs in {output_dir}/')


if __name__ == '__main__':
    main()