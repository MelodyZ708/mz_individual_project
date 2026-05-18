"""
Multi-Frame Channel Validation — True Top-K Discovery
=======================================================
Re-evaluate the top 8-channel combinations (from the initial random search)
across 20 uniformly-sampled frames to filter out "lucky" combinations and
identify the truly robust ones.

For every (combination × frame × brightness condition) we record:
  - Local Sharpness (combined, x-direction, y-direction)
  - Global Sharpness
  - Basin Width (50% threshold)
  - Hessian Condition Number (ratio of x vs y local sharpness)
  - Retention vs clean condition

Outputs:
  - Per-combination CSV with per-frame detail
  - Summary CSV with mean / std / worst-case statistics
  - Boxplot of Retention_50 across 20 frames for each combination
  - Boxplot of Hessian Condition Number

Usage:
  python validate_channels_multiframe.py [--n_frames 20] [--quick]

Author: mz325
Date: 2026-05
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import os
import glob
import csv
import cv2
import argparse
import json
from typing import Tuple, List, Dict, Optional
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/multiframe_validation',
    'device': 'cuda:0',

    # Perturbation range (pixels)
    'max_shift_px': 30,
    'grid_size': 61,

    # Sharpness config
    'sharpness_radius': 5,
}

# ── Top 10 combinations from the initial 8-channel random search ──
# Extracted from summary.csv, ranked by Avg_Retention_50%
TOP_COMBINATIONS = [
    {'name': 'Rank01_Run05',  'channels': [6, 7, 12, 15, 36, 45, 58, 62]},
    {'name': 'Rank02_Run01',  'channels': [8, 22, 23, 27, 28, 42, 48, 60]},
    {'name': 'Rank03_Run16',  'channels': [1, 4, 7, 13, 22, 34, 41, 54]},
    {'name': 'Rank04_Run12',  'channels': [8, 9, 17, 19, 37, 41, 51, 52]},
    {'name': 'Rank05_Run20',  'channels': [2, 9, 19, 20, 28, 49, 55, 58]},
    {'name': 'Rank06_Run19',  'channels': [17, 19, 27, 38, 42, 52, 61, 62]},
    {'name': 'Rank07_Run14',  'channels': [1, 2, 22, 36, 41, 46, 48, 51]},
    {'name': 'Rank08_Run17',  'channels': [1, 4, 6, 15, 19, 57, 58, 61]},
    {'name': 'Rank09_Run06',  'channels': [6, 7, 16, 38, 45, 51, 52, 57]},
    {'name': 'Rank10_Run23',  'channels': [13, 15, 25, 28, 33, 34, 40, 45, 49, 52, 55, 62]},
]

# Also include a grayscale baseline for reference
INCLUDE_GRAY_BASELINE = True

BRIGHTNESS_CONDITIONS = [
    {'key': 'clean',    'factor': 0.0, 'label': 'Clean',           'suffix': '_clean'},
    {'key': 'bright30', 'factor': 0.3, 'label': 'Brightness +30%', 'suffix': '_bright30'},
    {'key': 'bright50', 'factor': 0.5, 'label': 'Brightness +50%', 'suffix': '_bright50'},
]

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 250,
    'mathtext.fontset': 'cm',
})


# ============================================================
# Lightweight Feature Extractor
# ============================================================
class DirectConv1Extractor(nn.Module):
    """
    Directly extract arbitrary conv1 channels from ResNet18.
    Uses conv1 + bn1 + relu to match the random_channel_search.py convention.
    """
    def __init__(self, channel_indices: List[int], device: str = "cuda:0"):
        super().__init__()
        self.device = device
        self.channel_indices = channel_indices

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        resnet.eval()

        self.feature_extractor = nn.Sequential(
            resnet.conv1,   # 3 → 64, stride=2, H/2 × W/2
            resnet.bn1,
            resnet.relu,
        )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.indices_tensor = torch.tensor(channel_indices, device=device, dtype=torch.long)

    def forward(self, rgb_img: torch.Tensor) -> torch.Tensor:
        x = rgb_img.float()
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        with torch.no_grad():
            features = self.feature_extractor(x)

        selected = features[:, self.indices_tensor, :, :]

        upsampled = F_torch.interpolate(
            selected, size=rgb_img.shape[-2:],
            mode='bilinear', align_corners=False
        )
        return upsampled


# ============================================================
# Core Functions
# ============================================================

def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = image.shape[:2]
    orig_shape = image.shape
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(image, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    if warped.shape != orig_shape:
        warped = warped.reshape(orig_shape)
    return warped


def apply_brightness_perturbation(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 0.0:
        return image.copy()
    return np.clip(image + factor, 0.0, 1.0)


def load_image_numpy(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_tensor(image_np: np.ndarray, device: str = "cuda:0") -> torch.Tensor:
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_photometric_cost(img_ref: np.ndarray, img_warped: np.ndarray) -> float:
    residual = img_warped.astype(np.float64) - img_ref.astype(np.float64)
    return np.mean(residual ** 2)


def extract_features_numpy(rgb_tensor: torch.Tensor, extractor, mode: str) -> np.ndarray:
    device = rgb_tensor.device
    with torch.no_grad():
        if mode == 'gray':
            weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
            feat = (rgb_tensor * weights).sum(dim=1, keepdim=True)
        elif mode == 'cnn':
            feat = extractor(rgb_tensor)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return feat[0].permute(1, 2, 0).cpu().numpy()


def compute_2d_cost_landscape(
    feat_ref: np.ndarray, feat_target: np.ndarray,
    max_shift: float, grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx_values = np.linspace(-max_shift, max_shift, grid_size)
    dy_values = np.linspace(-max_shift, max_shift, grid_size)
    cost_grid = np.zeros((grid_size, grid_size))
    for i, dy in enumerate(dy_values):
        for j, dx in enumerate(dx_values):
            shifted = shift_image(feat_target, dx, dy)
            cost_grid[i, j] = compute_photometric_cost(feat_ref, shifted)
    return dx_values, dy_values, cost_grid


# ============================================================
# Sharpness & Basin Width & Condition Number
# ============================================================

def compute_sharpness(
    cost_grid: np.ndarray, dx_values: np.ndarray, dy_values: np.ndarray,
    radius_px: int = 5
) -> Dict[str, float]:
    grid_size = cost_grid.shape[0]
    center = grid_size // 2
    step_x = dx_values[1] - dx_values[0]
    step_y = dy_values[1] - dy_values[0]

    min_flat = np.argmin(cost_grid)
    min_iy, min_ix = np.unravel_index(min_flat, cost_grid.shape)
    min_dx = dx_values[min_ix]
    min_dy = dy_values[min_iy]
    min_cost = cost_grid[min_iy, min_ix]

    c_min, c_max = cost_grid.min(), cost_grid.max()
    if c_max - c_min > 1e-10:
        cost_norm = (cost_grid - c_min) / (c_max - c_min)
    else:
        cost_norm = np.zeros_like(cost_grid)

    def _slice_sharpness(arr, step, center_idx, r):
        lo = max(center_idx - r, 0)
        hi = min(center_idx + r, len(arr) - 1)
        segment = arr[lo:hi+1]
        if len(segment) < 2:
            return 0.0
        grad = np.abs(segment[1:] - segment[:-1]) / step
        return float(np.mean(grad))

    full_r = grid_size // 2

    x_local = _slice_sharpness(cost_norm[center, :], step_x, center, radius_px)
    y_local = _slice_sharpness(cost_norm[:, center], step_y, center, radius_px)
    local_combined = (x_local + y_local) / 2.0

    x_global = _slice_sharpness(cost_norm[center, :], step_x, center, full_r)
    y_global = _slice_sharpness(cost_norm[:, center], step_y, center, full_r)
    global_combined = (x_global + y_global) / 2.0

    # Basin Width (50%) on raw cost
    max_shift = dx_values[-1]
    step = dx_values[1] - dx_values[0]
    raw_max = cost_grid.max()
    if raw_max > 1e-10:
        cost_row = cost_grid[center, :]
        basin_mask = cost_row < (0.5 * raw_max)
        basin_width = np.sum(basin_mask) * step
    else:
        basin_width = 2 * max_shift

    # Hessian Condition Number: ratio of x vs y local sharpness
    # A perfectly isotropic basin has condition_number = 1.0
    # Higher values indicate anisotropy (elongated basin)
    eps = 1e-8
    sharpness_max = max(x_local, y_local)
    sharpness_min = min(x_local, y_local)
    condition_number = sharpness_max / (sharpness_min + eps)

    return {
        'x_local': x_local, 'y_local': y_local, 'local': local_combined,
        'x_global': x_global, 'y_global': y_global, 'global': global_combined,
        'basin_width': basin_width,
        'min_cost': min_cost,
        'min_location': (min_dx, min_dy),
        'condition_number': condition_number,
    }


# ============================================================
# Frame Sampling
# ============================================================

def sample_frame_indices(n_total: int, n_frames: int) -> List[int]:
    """
    Uniformly sample n_frames indices from [0, n_total-1].
    Avoids the very first and very last frames (boundary artifacts).
    """
    margin = max(10, n_total // 50)  # skip first/last ~2% of sequence
    usable_start = margin
    usable_end = n_total - margin
    usable_range = usable_end - usable_start

    if n_frames >= usable_range:
        # If we want more frames than available, just take all usable
        return list(range(usable_start, usable_end))

    step = usable_range / n_frames
    indices = [int(usable_start + i * step) for i in range(n_frames)]
    return indices


# ============================================================
# Single Combination Evaluation (across all frames)
# ============================================================

def evaluate_combination(
    combo: Dict,
    frame_indices: List[int],
    all_images: List[str],
    device: str,
    cfg: dict,
    output_dir: str,
    include_gray: bool = True,
) -> Dict:
    """
    Evaluate one channel combination across all sampled frames.
    Returns a dict with per-frame metrics and aggregate statistics.
    """
    name = combo['name']
    channels = combo['channels']
    n_ch = len(channels)
    ch_str = ','.join(str(c) for c in channels)

    combo_dir = os.path.join(output_dir, name)
    os.makedirs(combo_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  {name}: Channels = [{ch_str}]  ({n_ch} ch)")
    print(f"  Evaluating across {len(frame_indices)} frames")
    print(f"{'='*70}")

    extractor = DirectConv1Extractor(channels, device)

    # Storage for per-frame metrics
    per_frame_records = []

    for fi, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(all_images):
            continue

        img_path = all_images[frame_idx]
        frame_name = os.path.basename(img_path)
        print(f"  [{fi+1:2d}/{len(frame_indices)}] Frame {frame_idx} ({frame_name})", flush=True)

        rgb_np = load_image_numpy(img_path)
        rgb_tensor = numpy_to_tensor(rgb_np, device)

        # Reference features (always clean)
        feat_ref_cnn = extract_features_numpy(rgb_tensor, extractor, 'cnn')
        feat_ref_gray = extract_features_numpy(rgb_tensor, None, 'gray') if include_gray else None

        # Store clean sharpness for retention calculation
        clean_local_cnn = None
        clean_local_gray = None

        for cond in BRIGHTNESS_CONDITIONS:
            cond_key = cond['key']
            bright_factor = cond['factor']

            rgb_target_np = apply_brightness_perturbation(rgb_np, bright_factor)
            rgb_target_tensor = numpy_to_tensor(rgb_target_np, device)

            feat_target_cnn = extract_features_numpy(rgb_target_tensor, extractor, 'cnn')
            del rgb_target_tensor

            # CNN cost landscape
            dx, dy, cost_cnn = compute_2d_cost_landscape(
                feat_ref_cnn, feat_target_cnn,
                cfg['max_shift_px'], cfg['grid_size']
            )
            s_cnn = compute_sharpness(cost_cnn, dx, dy, cfg['sharpness_radius'])

            if cond_key == 'clean':
                clean_local_cnn = s_cnn['local']

            # Compute retention
            if clean_local_cnn and clean_local_cnn > 1e-10:
                retention_cnn = (s_cnn['local'] / clean_local_cnn) * 100.0
            else:
                retention_cnn = 0.0

            record = {
                'combo_name': name,
                'channels': ch_str,
                'n_channels': n_ch,
                'frame_idx': frame_idx,
                'frame_name': frame_name,
                'condition': cond_key,
                'cnn_local': s_cnn['local'],
                'cnn_x_local': s_cnn['x_local'],
                'cnn_y_local': s_cnn['y_local'],
                'cnn_global': s_cnn['global'],
                'cnn_basin_width': s_cnn['basin_width'],
                'cnn_min_cost': s_cnn['min_cost'],
                'cnn_condition_number': s_cnn['condition_number'],
                'cnn_retention': retention_cnn,
            }

            # Gray baseline
            if include_gray and feat_ref_gray is not None:
                feat_target_gray = extract_features_numpy(
                    numpy_to_tensor(rgb_target_np, device), None, 'gray'
                )
                _, _, cost_gray = compute_2d_cost_landscape(
                    feat_ref_gray, feat_target_gray,
                    cfg['max_shift_px'], cfg['grid_size']
                )
                s_gray = compute_sharpness(cost_gray, dx, dy, cfg['sharpness_radius'])

                if cond_key == 'clean':
                    clean_local_gray = s_gray['local']

                if clean_local_gray and clean_local_gray > 1e-10:
                    retention_gray = (s_gray['local'] / clean_local_gray) * 100.0
                else:
                    retention_gray = 0.0

                record.update({
                    'gray_local': s_gray['local'],
                    'gray_x_local': s_gray['x_local'],
                    'gray_y_local': s_gray['y_local'],
                    'gray_global': s_gray['global'],
                    'gray_basin_width': s_gray['basin_width'],
                    'gray_condition_number': s_gray['condition_number'],
                    'gray_retention': retention_gray,
                })

            per_frame_records.append(record)

            # Compact print
            print(f"    {cond['label']:<18} CNN: L={s_cnn['local']:.4f} "
                  f"(x={s_cnn['x_local']:.4f} y={s_cnn['y_local']:.4f}) "
                  f"CN={s_cnn['condition_number']:.2f} "
                  f"BW={s_cnn['basin_width']:.0f} "
                  f"Ret={retention_cnn:.1f}%")

        del rgb_tensor, rgb_np
        torch.cuda.empty_cache()

    # ── Save per-frame CSV ──
    csv_path = os.path.join(combo_dir, 'per_frame_metrics.csv')
    if per_frame_records:
        keys = per_frame_records[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rec in per_frame_records:
                row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in rec.items()}
                writer.writerow(row)

    # ── Compute aggregate statistics ──
    agg = _compute_aggregates(name, ch_str, n_ch, per_frame_records)

    # Save aggregate JSON
    agg_path = os.path.join(combo_dir, 'aggregate_metrics.json')
    with open(agg_path, 'w') as f:
        json.dump(agg, f, indent=2)

    return {
        'aggregate': agg,
        'per_frame': per_frame_records,
    }


def _compute_aggregates(name, ch_str, n_ch, records):
    """Compute mean, std, min, 5th percentile for key metrics."""
    agg = {
        'combo_name': name,
        'channels': ch_str,
        'n_channels': n_ch,
    }

    for cond_key in ['clean', 'bright30', 'bright50']:
        cond_records = [r for r in records if r['condition'] == cond_key]
        if not cond_records:
            continue

        for metric in ['cnn_local', 'cnn_x_local', 'cnn_y_local', 'cnn_global',
                        'cnn_basin_width', 'cnn_condition_number', 'cnn_retention']:
            vals = [r[metric] for r in cond_records if metric in r]
            if vals:
                arr = np.array(vals)
                agg[f'{cond_key}_{metric}_mean'] = float(np.mean(arr))
                agg[f'{cond_key}_{metric}_std'] = float(np.std(arr))
                agg[f'{cond_key}_{metric}_min'] = float(np.min(arr))
                agg[f'{cond_key}_{metric}_p5'] = float(np.percentile(arr, 5))
                agg[f'{cond_key}_{metric}_median'] = float(np.median(arr))

        # Gray baseline
        for metric in ['gray_local', 'gray_retention', 'gray_condition_number']:
            vals = [r[metric] for r in cond_records if metric in r]
            if vals:
                arr = np.array(vals)
                agg[f'{cond_key}_{metric}_mean'] = float(np.mean(arr))
                agg[f'{cond_key}_{metric}_std'] = float(np.std(arr))

    return agg


# ============================================================
# Visualization
# ============================================================

def plot_retention_boxplot(all_results: List[Dict], output_dir: str):
    """
    Boxplot of Retention_50 across all frames for each combination.
    Also shows gray baseline as a horizontal band.
    """
    fig, ax = plt.subplots(figsize=(max(14, len(all_results) * 1.2), 7))

    combo_names = []
    combo_data = []
    gray_data_all = []

    for res in all_results:
        name = res['aggregate']['combo_name']
        records_50 = [r for r in res['per_frame']
                      if r['condition'] == 'bright50']
        retentions = [r['cnn_retention'] for r in records_50]

        if retentions:
            combo_names.append(name.replace('Rank', 'R').replace('_Run', '\nRun'))
            combo_data.append(retentions)

        # Collect gray retention
        gray_rets = [r.get('gray_retention', None) for r in records_50]
        gray_rets = [g for g in gray_rets if g is not None]
        gray_data_all.extend(gray_rets)

    if not combo_data:
        print("  [Warning] No data for retention boxplot.")
        return

    bp = ax.boxplot(combo_data, labels=combo_names, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=5),
                    medianprops=dict(color='black', linewidth=1.5))

    # Color boxes by median retention
    medians = [np.median(d) for d in combo_data]
    max_med = max(medians) if medians else 1
    min_med = min(medians) if medians else 0
    cmap = plt.get_cmap('RdYlGn')

    for i, (patch, med) in enumerate(zip(bp['boxes'], medians)):
        if max_med > min_med:
            norm_val = (med - min_med) / (max_med - min_med)
        else:
            norm_val = 0.5
        patch.set_facecolor(cmap(norm_val))
        patch.set_alpha(0.7)

    # Gray baseline band
    if gray_data_all:
        gray_mean = np.mean(gray_data_all)
        gray_std = np.std(gray_data_all)
        ax.axhspan(gray_mean - gray_std, gray_mean + gray_std,
                    color='gray', alpha=0.15, label=f'Gray baseline (mean={gray_mean:.1f}%)')
        ax.axhline(gray_mean, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.axhline(100, color='green', linestyle=':', linewidth=1, alpha=0.5, label='100% (no degradation)')

    ax.set_ylabel('Retention_50 (%)\n(Local Sharpness at +50% / Clean × 100)')
    ax.set_title('Multi-Frame Validation: Retention under +50% Brightness\n'
                 f'({len(combo_data[0]) if combo_data else 0} frames per combination)',
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=0, labelsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'boxplot_retention50.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_condition_number_boxplot(all_results: List[Dict], output_dir: str):
    """
    Boxplot of Hessian Condition Number under +50% brightness.
    Lower = more isotropic basin = better.
    """
    fig, ax = plt.subplots(figsize=(max(14, len(all_results) * 1.2), 7))

    combo_names = []
    combo_data = []

    for res in all_results:
        name = res['aggregate']['combo_name']
        records_50 = [r for r in res['per_frame']
                      if r['condition'] == 'bright50']
        cns = [r['cnn_condition_number'] for r in records_50]

        if cns:
            combo_names.append(name.replace('Rank', 'R').replace('_Run', '\nRun'))
            combo_data.append(cns)

    if not combo_data:
        return

    bp = ax.boxplot(combo_data, labels=combo_names, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='blue', markersize=5),
                    medianprops=dict(color='black', linewidth=1.5))

    for patch in bp['boxes']:
        patch.set_facecolor('#AEC6CF')
        patch.set_alpha(0.7)

    ax.axhline(1.0, color='green', linestyle=':', linewidth=1.5, alpha=0.7,
               label='Perfect isotropy (CN=1)')
    ax.set_ylabel('Condition Number (max(Sx,Sy) / min(Sx,Sy))')
    ax.set_title('Multi-Frame Validation: Basin Isotropy under +50% Brightness\n'
                 '(Lower = more isotropic = better constrained)',
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'boxplot_condition_number.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_xy_sharpness_scatter(all_results: List[Dict], output_dir: str):
    """
    Scatter plot: x-direction vs y-direction local sharpness under +50%.
    Points on the diagonal = isotropic. Off-diagonal = anisotropic.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for idx, res in enumerate(all_results):
        name = res['aggregate']['combo_name']
        records_50 = [r for r in res['per_frame']
                      if r['condition'] == 'bright50']
        xs = [r['cnn_x_local'] for r in records_50]
        ys = [r['cnn_y_local'] for r in records_50]

        short_name = name.replace('Rank', 'R').replace('_Run', ' R')
        ax.scatter(xs, ys, c=[colors[idx]], label=short_name, alpha=0.6, s=30, edgecolors='k', linewidths=0.3)

    # Diagonal line
    lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.3, label='Isotropic line')

    ax.set_xlabel('X-direction Local Sharpness')
    ax.set_ylabel('Y-direction Local Sharpness')
    ax.set_title('X vs Y Sharpness under +50% Brightness\n(On diagonal = isotropic basin)',
                 fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'scatter_xy_sharpness.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_per_frame_retention_heatmap(all_results: List[Dict], output_dir: str):
    """
    Heatmap: rows = combinations, columns = frame indices.
    Cell color = Retention_50 for that (combo, frame).
    Reveals which combos are universally good vs frame-dependent.
    """
    combo_names = []
    frame_indices = None
    data_matrix = []

    for res in all_results:
        name = res['aggregate']['combo_name']
        records_50 = sorted(
            [r for r in res['per_frame'] if r['condition'] == 'bright50'],
            key=lambda r: r['frame_idx']
        )
        if not records_50:
            continue

        if frame_indices is None:
            frame_indices = [r['frame_idx'] for r in records_50]

        retentions = [r['cnn_retention'] for r in records_50]
        combo_names.append(name.replace('Rank', 'R').replace('_Run', ' R'))
        data_matrix.append(retentions)

    if not data_matrix:
        return

    data_matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(max(14, len(frame_indices) * 0.6), max(6, len(combo_names) * 0.5)))

    im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlGn', vmin=40, vmax=140)

    ax.set_xticks(range(len(frame_indices)))
    ax.set_xticklabels([str(fi) for fi in frame_indices], fontsize=7, rotation=45)
    ax.set_yticks(range(len(combo_names)))
    ax.set_yticklabels(combo_names, fontsize=9)

    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Channel Combination')
    ax.set_title('Per-Frame Retention_50 Heatmap\n(Green=good, Red=poor)', fontweight='bold')

    # Annotate cells
    for i in range(data_matrix.shape[0]):
        for j in range(data_matrix.shape[1]):
            val = data_matrix[i, j]
            color = 'white' if val < 60 or val > 120 else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=6, color=color)

    plt.colorbar(im, ax=ax, label='Retention_50 (%)', shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, 'heatmap_retention50.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Frame Channel Validation')
    parser.add_argument('--n_frames', type=int, default=20,
                        help='Number of frames to sample uniformly (default: 20)')
    parser.add_argument('--quick', action='store_true',
                        help='Skip gray baseline computation for faster runs')
    parser.add_argument('--combos', type=str, default='all',
                        help='Comma-separated combo indices (1-based) to evaluate, or "all"')
    args = parser.parse_args()

    cfg = CONFIG
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    device = cfg['device']

    include_gray = not args.quick

    # Load image list
    all_images = sorted(glob.glob(os.path.join(cfg['rgb_dir'], "*.png")))
    n_images = len(all_images)

    if n_images == 0:
        print(f"[ERROR] No images found in {cfg['rgb_dir']}")
        return

    # Sample frames
    frame_indices = sample_frame_indices(n_images, args.n_frames)

    # Select combinations
    if args.combos == 'all':
        combos = TOP_COMBINATIONS
    else:
        indices = [int(x.strip()) - 1 for x in args.combos.split(',')]
        combos = [TOP_COMBINATIONS[i] for i in indices if 0 <= i < len(TOP_COMBINATIONS)]

    print(f"\n{'='*70}")
    print(f"  MULTI-FRAME CHANNEL VALIDATION")
    print(f"{'='*70}")
    print(f"  Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total images:     {n_images}")
    print(f"  Sampled frames:   {len(frame_indices)}")
    print(f"  Frame indices:    {frame_indices}")
    print(f"  Combinations:     {len(combos)}")
    print(f"  Gray baseline:    {include_gray}")
    print(f"  Grid:             {cfg['grid_size']}x{cfg['grid_size']}")
    print(f"  Output:           {output_dir}")
    print(f"{'='*70}")

    # ── Evaluate all combinations ──
    all_results = []
    for combo in combos:
        result = evaluate_combination(
            combo=combo,
            frame_indices=frame_indices,
            all_images=all_images,
            device=device,
            cfg=cfg,
            output_dir=output_dir,
            include_gray=include_gray,
        )
        all_results.append(result)

    # ============================================================
    # Summary & Ranking
    # ============================================================
    print(f"\n\n{'='*100}")
    print(f"  MULTI-FRAME VALIDATION — FINAL RANKING")
    print(f"  Sorted by: Mean Retention_50 (primary), then Worst-case Retention_50 (secondary)")
    print(f"{'='*100}")

    # Sort by mean retention_50, then by worst-case
    all_results.sort(key=lambda r: (
        -r['aggregate'].get('bright50_cnn_retention_mean', 0),
        -r['aggregate'].get('bright50_cnn_retention_min', 0),
    ))

    header = (f"  {'Rank':<5} {'Name':<18} {'#Ch':<4} "
              f"{'Ret50_Mean':<12} {'Ret50_Std':<11} {'Ret50_Min':<11} {'Ret50_P5':<11} "
              f"{'CN50_Mean':<10} {'Local50_Mean':<13}")
    print(header)
    print(f"  {'-'*110}")

    for rank, res in enumerate(all_results, 1):
        a = res['aggregate']
        marker = " ***" if rank <= 5 else ""
        print(f"  {rank:<5} {a['combo_name']:<18} {a['n_channels']:<4} "
              f"{a.get('bright50_cnn_retention_mean', 0):<12.1f} "
              f"{a.get('bright50_cnn_retention_std', 0):<11.1f} "
              f"{a.get('bright50_cnn_retention_min', 0):<11.1f} "
              f"{a.get('bright50_cnn_retention_p5', 0):<11.1f} "
              f"{a.get('bright50_cnn_condition_number_mean', 0):<10.2f} "
              f"{a.get('bright50_cnn_local_mean', 0):<13.4f}{marker}")

    # ── Generate plots ──
    print(f"\n  Generating visualizations...")
    plot_retention_boxplot(all_results, output_dir)
    plot_condition_number_boxplot(all_results, output_dir)
    plot_xy_sharpness_scatter(all_results, output_dir)
    plot_per_frame_retention_heatmap(all_results, output_dir)

    # ── Save summary CSV ──
    summary_path = os.path.join(output_dir, 'summary_multiframe.csv')
    if all_results:
        agg_list = [r['aggregate'] for r in all_results]
        all_keys = set()
        for a in agg_list:
            all_keys.update(a.keys())
        all_keys = sorted(all_keys)

        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for a in agg_list:
                row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in a.items()}
                writer.writerow(row)
        print(f"  Summary CSV saved: {summary_path}")

    # ── Save all per-frame data to a single CSV ──
    all_frames_path = os.path.join(output_dir, 'all_per_frame_metrics.csv')
    all_frame_records = []
    for res in all_results:
        all_frame_records.extend(res['per_frame'])
    if all_frame_records:
        keys = all_frame_records[0].keys()
        with open(all_frames_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rec in all_frame_records:
                row = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                       for k, v in rec.items()}
                writer.writerow(row)
        print(f"  All per-frame CSV saved: {all_frames_path}")

    print(f"\n{'='*70}")
    print(f"  MULTI-FRAME VALIDATION — COMPLETE")
    print(f"  Total combinations evaluated: {len(all_results)}")
    print(f"  Frames per combination: {len(frame_indices)}")
    print(f"  Output: {output_dir}/")
    if all_results:
        best = all_results[0]['aggregate']
        print(f"  True Top 1: {best['combo_name']} — [{best['channels']}]")
        print(f"              Mean Ret50={best.get('bright50_cnn_retention_mean', 0):.1f}% "
              f"± {best.get('bright50_cnn_retention_std', 0):.1f}% "
              f"(worst={best.get('bright50_cnn_retention_min', 0):.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()