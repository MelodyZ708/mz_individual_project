"""
visualize_basin_layer1_64ch.py
==============================
Stage 2: Convergence Basin — All 64 ResNet-18 Layer1 Channels (Individual, 3D Surface)

For each of the 64 channels from ResNet18 Layer1 (conv1->bn1->relu->maxpool->layer1),
generates a single figure with 3 columns (3D surface): Clean / +30% / +50%.

Mirrors the exact format of visualize_basin_all64_channels.py (Stage 1 / Conv1+BN+ReLU),
but uses Layer1 features.

Output:
  vis_results/convergence_basin_layer1/channel_XX.png  (XX = 00..63)
  vis_results/convergence_basin_layer1/bqs_summary.csv
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as mplNormalize
import numpy as np
from PIL import Image
import os
import sys
import glob
import cv2
import csv
from typing import Tuple, Dict

from torchvision.models import resnet18, ResNet18_Weights

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/convergence_basin_layer1',
    'device': 'cuda:0',
    'frame_index': 306,
    'max_shift_px': 30,
    'grid_size': 61,
    'sharpness_radius': 5,
}

BRIGHTNESS_CONDITIONS = [
    {'factor': 0.0, 'label': 'Clean'},
    {'factor': 0.3, 'label': '+30%'},
    {'factor': 0.5, 'label': '+50%'},
]

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'mathtext.fontset': 'cm',
})


# ============================================================
# Feature Extractor: conv1 -> bn1 -> relu -> maxpool -> layer1
# ============================================================
class Layer1Extractor:
    """Extract all 64 Layer1 features at once, upsampled to original resolution."""

    def __init__(self, device='cuda:0'):
        self.device = device
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        resnet.eval()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1

        for p in list(self.conv1.parameters()) + list(self.bn1.parameters()) + \
                 list(self.layer1.parameters()):
            p.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract_all_channels(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input:  [1, 3, H, W] in [0, 1]
        Output: [1, 64, H, W] upsampled to original resolution
        """
        orig_size = rgb_tensor.shape[-2:]
        x = (rgb_tensor - self.mean) / self.std
        x = self.conv1(x)     # [1, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # [1, 64, H/4, W/4]
        x = self.layer1(x)    # [1, 64, H/4, W/4]  (Layer1 keeps 64ch)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x              # [1, 64, H, W]


# ============================================================
# Core helpers (identical to visualize_basin_all64_channels.py)
# ============================================================
def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = image.shape[:2]
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 0.0:
        return image.copy()
    return np.clip(image + factor, 0.0, 1.0)


def load_image_numpy(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_cost_landscape(
    feat_ref: np.ndarray,
    feat_target: np.ndarray,
    max_shift: float,
    grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx_vals = np.linspace(-max_shift, max_shift, grid_size)
    dy_vals = np.linspace(-max_shift, max_shift, grid_size)
    cost_grid = np.zeros((grid_size, grid_size))
    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            shifted = shift_image(feat_target, dx, dy)
            residual = shifted.astype(np.float64) - feat_ref.astype(np.float64)
            cost_grid[i, j] = np.mean(residual ** 2)
    return dx_vals, dy_vals, cost_grid


def compute_sharpness(cost_grid, dx_vals, dy_vals, radius=5):
    grid_size = cost_grid.shape[0]
    center = grid_size // 2
    step_x = dx_vals[1] - dx_vals[0]
    step_y = dy_vals[1] - dy_vals[0]
    c_min, c_max = cost_grid.min(), cost_grid.max()
    if c_max - c_min > 1e-10:
        cost_norm = (cost_grid - c_min) / (c_max - c_min)
    else:
        cost_norm = np.zeros_like(cost_grid)

    def _sharpness(arr, step, c_idx, r):
        lo = max(c_idx - r, 0)
        hi = min(c_idx + r, len(arr) - 1)
        seg = arr[lo:hi+1]
        if len(seg) < 2:
            return 0.0
        return float(np.mean(np.abs(seg[1:] - seg[:-1]) / step))

    full_r = grid_size // 2
    x_local  = _sharpness(cost_norm[center, :], step_x, center, radius)
    y_local  = _sharpness(cost_norm[:, center], step_y, center, radius)
    x_global = _sharpness(cost_norm[center, :], step_x, center, full_r)
    y_global = _sharpness(cost_norm[:, center], step_y, center, full_r)
    return {
        'local':    (x_local + y_local) / 2.0,
        'global':   (x_global + y_global) / 2.0,
        'x_local':  x_local, 'y_local':  y_local,
        'x_global': x_global, 'y_global': y_global,
    }


# ============================================================
# BQS helpers (matching analyze_basin_metrics.py)
# ============================================================
def normalize_grid(cost_grid):
    c_min, c_max = cost_grid.min(), cost_grid.max()
    if c_max - c_min < 1e-10:
        return np.zeros_like(cost_grid), False
    return (cost_grid - c_min) / (c_max - c_min), True


def compute_lqbs(cost_grid):
    norm, valid = normalize_grid(cost_grid)
    if not valid:
        return 0.0
    grid_size = norm.shape[0]
    center = grid_size // 2
    half = center
    y_idx, x_idx = np.mgrid[0:grid_size, 0:grid_size]
    dist2 = ((y_idx - center) / half) ** 2 + ((x_idx - center) / half) ** 2
    ideal = np.clip(dist2, 0, 1)
    corr = np.corrcoef(norm.ravel(), ideal.ravel())[0, 1]
    return float(max(corr, 0.0))


def compute_basin_width(cost_grid, threshold=0.3):
    norm, valid = normalize_grid(cost_grid)
    if not valid:
        return 0.0
    basin_mask = norm < threshold
    return float(np.sum(basin_mask) / norm.size)


def compute_shape_similarity(clean_grid, perturbed_grid):
    c_norm, cv = normalize_grid(clean_grid)
    p_norm, pv = normalize_grid(perturbed_grid)
    if not cv or not pv:
        return 0.0
    corr = np.corrcoef(c_norm.ravel(), p_norm.ravel())[0, 1]
    return float(max(corr, 0.0))


def compute_sharpness_retention(clean_grid, perturbed_grid):
    def _sharp(g):
        n, v = normalize_grid(g)
        if not v:
            return 0.0
        center = n.shape[0] // 2
        gx = np.gradient(n[center, :])
        gy = np.gradient(n[:, center])
        return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
    s_clean = _sharp(clean_grid)
    s_pert  = _sharp(perturbed_grid)
    if s_clean < 1e-10:
        return 1.0
    return float(min(s_pert / s_clean, 1.0))


def compute_minpos_consistency(clean_grid, perturbed_grid):
    cy, cx = np.unravel_index(np.argmin(clean_grid), clean_grid.shape)
    py, px = np.unravel_index(np.argmin(perturbed_grid), perturbed_grid.shape)
    dist = np.sqrt((cy - py) ** 2 + (cx - px) ** 2)
    max_dist = np.sqrt(2) * clean_grid.shape[0]
    return float(max(1.0 - dist / max_dist, 0.0))


def compute_symmetry(cost_grid):
    n, v = normalize_grid(cost_grid)
    if not v:
        return 0.0
    sym_h = 1.0 - np.mean(np.abs(n - np.fliplr(n)))
    sym_v = 1.0 - np.mean(np.abs(n - np.flipud(n)))
    return float((sym_h + sym_v) / 2.0)


def compute_bqs(clean_grid, bright30_grid, bright50_grid):
    lqbs  = compute_lqbs(clean_grid)
    width = compute_basin_width(clean_grid)
    bq    = 0.75 * lqbs + 0.25 * width

    shape30 = compute_shape_similarity(clean_grid, bright30_grid)
    shape50 = compute_shape_similarity(clean_grid, bright50_grid)
    shape_sim = (shape30 + shape50) / 2.0

    ret30 = compute_sharpness_retention(clean_grid, bright30_grid)
    ret50 = compute_sharpness_retention(clean_grid, bright50_grid)
    sharp_ret = (ret30 + ret50) / 2.0

    minpos30 = compute_minpos_consistency(clean_grid, bright30_grid)
    minpos50 = compute_minpos_consistency(clean_grid, bright50_grid)
    minpos = (minpos30 + minpos50) / 2.0

    sym30 = compute_symmetry(bright30_grid)
    sym50 = compute_symmetry(bright50_grid)
    sym = (sym30 + sym50) / 2.0

    retention = shape_sim * sharp_ret * minpos * sym
    bqs = bq * retention
    return {
        'BQS': round(bqs, 6),
        'BQ': round(bq, 6),
        'LQBS': round(lqbs, 6),
        'Width': round(width, 6),
        'ShapeSim': round(shape_sim, 6),
        'SharpRet': round(sharp_ret, 6),
        'MinPos': round(minpos, 6),
        'Symmetry': round(sym, 6),
        'Retention': round(retention, 6),
    }


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    cfg = CONFIG
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    all_images = sorted(glob.glob(os.path.join(cfg['rgb_dir'], "*.png")))
    if not all_images:
        print(f"[ERROR] No images found in {cfg['rgb_dir']}")
        sys.exit(1)

    frame_idx = cfg['frame_index']
    if frame_idx >= len(all_images):
        print(f"[ERROR] Frame index {frame_idx} out of range")
        sys.exit(1)

    print("=" * 60)
    print("  Convergence Basin — All 64 Layer1 Channels (Stage 2)")
    print("=" * 60)
    print(f"  Extraction:  conv1->bn1->relu->maxpool->layer1")
    print(f"  Dataset:     {cfg['rgb_dir']}")
    print(f"  Frame:       {frame_idx}")
    print(f"  Grid:        {cfg['grid_size']}x{cfg['grid_size']}")
    print(f"  Shift range: +/-{cfg['max_shift_px']} px")
    print(f"  Output:      {output_dir}")
    print("=" * 60)

    device = cfg['device']
    extractor = Layer1Extractor(device=device)

    rgb_np = load_image_numpy(all_images[frame_idx])
    rgb_tensor = numpy_to_tensor(rgb_np, device)

    print("\n  Extracting 64 Layer1 features for all brightness conditions...")
    target_feats = {}
    for cond in BRIGHTNESS_CONDITIONS:
        factor = cond['factor']
        label  = cond['label']
        rgb_t  = apply_brightness(rgb_np, factor)
        t_tensor = numpy_to_tensor(rgb_t, device)
        feat = extractor.extract_all_channels(t_tensor)
        target_feats[label] = feat[0].cpu().numpy()  # [64, H, W]
        del t_tensor
        print(f"  [{label}] done")

    del rgb_tensor
    torch.cuda.empty_cache()

    # CSV
    csv_rows = []

    print(f"\n  Processing 64 channels...")
    for ch in range(64):
        print(f"  Channel {ch:02d}/63 ...", end=" ", flush=True)

        costs = {}
        sharpness_per_cond = {}
        for cond in BRIGHTNESS_CONDITIONS:
            label = cond['label']
            feat_ref_ch = target_feats['Clean'][ch]
            feat_tgt_ch = target_feats[label][ch]
            dx, dy, cost = compute_cost_landscape(
                feat_ref_ch, feat_tgt_ch,
                cfg['max_shift_px'], cfg['grid_size']
            )
            costs[label] = cost
            sharpness_per_cond[label] = compute_sharpness(
                cost, dx, dy, radius=cfg['sharpness_radius']
            )

        # BQS
        bqs_metrics = compute_bqs(costs['Clean'], costs['+30%'], costs['+50%'])

        # Kill%
        kill_clean = float(np.mean(target_feats['Clean'][ch] == 0)) * 100
        kill_50    = float(np.mean(target_feats['+50%'][ch]  == 0)) * 100

        # CSV row
        row = {'channel': ch, 'kill_clean': f"{kill_clean:.1f}", 'kill_50': f"{kill_50:.1f}"}
        row.update(bqs_metrics)
        csv_rows.append(row)

        # ── Plot ──
        fig = plt.figure(figsize=(18, 6))
        DX_grid, DY_grid = np.meshgrid(dx, dy, indexing='ij')

        for idx, cond in enumerate(BRIGHTNESS_CONDITIONS):
            label     = cond['label']
            cost_data = costs[label]

            c_min, c_max = cost_data.min(), cost_data.max()
            cost_norm = (cost_data - c_min) / (c_max - c_min) \
                        if c_max - c_min > 1e-10 else np.zeros_like(cost_data)

            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            cmap = plt.get_cmap('YlOrRd')
            norm = mplNormalize(vmin=0, vmax=1)
            fc   = cmap(norm(cost_norm))

            ax.plot_surface(DY_grid, DX_grid, cost_norm,
                            facecolors=fc, edgecolor='k',
                            linewidth=0.12, alpha=0.92,
                            shade=True, rcount=40, ccount=40,
                            antialiased=True)

            off = -0.05
            ax.contourf(DY_grid, DX_grid, cost_norm, zdir='z', offset=off,
                        levels=20, cmap='gray_r', alpha=0.7)
            ax.contour(DY_grid, DX_grid, cost_norm, zdir='z', offset=off,
                       levels=10, colors='k', linewidths=0.4, alpha=0.5)

            ax.set_xlabel(r'$\Delta x$ [px]', labelpad=8)
            ax.set_ylabel(r'$\Delta y$ [px]', labelpad=8)
            ax.set_zlabel('Norm. Cost', labelpad=6)
            ax.set_zlim(off, 1.05)
            ax.view_init(elev=32, azim=-50)
            ax.tick_params(labelsize=8)

            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
                pane.set_edgecolor('lightgray')

            s = sharpness_per_cond[label]
            ax.set_title(f"{label}\nLocal={s['local']:.4f}  Global={s['global']:.4f}",
                         fontsize=11, pad=10)

        fig.suptitle(
            f"Channel {ch:02d} — Layer1 (conv1→bn1→relu→maxpool→layer1) — Frame {frame_idx}\n"
            f"BQS={bqs_metrics['BQS']:.4f}  LQBS={bqs_metrics['LQBS']:.4f}  "
            f"Width={bqs_metrics['Width']:.4f}  Retention={bqs_metrics['Retention']:.4f}  "
            f"Kill%(Clean/+50%)={kill_clean:.1f}%/{kill_50:.1f}%",
            fontsize=11, y=1.02
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"channel_{ch:02d}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f"saved → {os.path.basename(out_path)}")

    # Write CSV
    csv_path = os.path.join(output_dir, "bqs_summary.csv")
    fieldnames = ['channel', 'BQS', 'BQ', 'LQBS', 'Width',
                  'ShapeSim', 'SharpRet', 'MinPos', 'Symmetry', 'Retention',
                  'kill_clean', 'kill_50']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # Sorted ranking
    sorted_rows = sorted(csv_rows, key=lambda r: float(r['BQS']), reverse=True)
    rank_path = os.path.join(output_dir, "channel_ranking.csv")
    with open(rank_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rank'] + fieldnames)
        writer.writeheader()
        for rank, row in enumerate(sorted_rows, 1):
            writer.writerow({'rank': rank, **row})

    print(f"\n[DONE] All outputs saved to: {output_dir}/")
    print(f"  - 64 channel PNG files")
    print(f"  - bqs_summary.csv  (all channels)")
    print(f"  - channel_ranking.csv  (sorted by BQS)")