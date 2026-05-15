"""
Random Channel Search (5-Channel) — Convergence Basin Analysis
================================================================
Following the finding that Top 1 (8ch) actually only uses 6 active channels,
this script tests whether fewer (5) well-chosen channels can achieve similar
or better performance.

Design:
  - Fixed runs: Top 1 and Top 2 combinations with dead channels removed
    (selecting 5 from the active ones).
  - Random runs: 15 random combinations of 5 channels from 56 active channels.
  - Same evaluation protocol: Clean / +30% / +50% brightness, 3 frames.

Scoring (same as before):
  Primary:   Avg Local Sharpness Retention (+50% L / Clean L * 100%)
  Secondary: Avg Local Sharpness under +50% brightness

Usage:
  cd /vol/bitbucket/mz325/individual_project
  python random_channel_search_5ch.py [--quick]

Author: mz325
Date: 2026-05
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, ResNet18_Weights
import os
import sys
import glob
import csv
import cv2
import argparse
import json
from collections import defaultdict
from typing import Tuple, List, Dict, Optional
from datetime import datetime

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/channel_search_5ch',
    'device': 'cuda:0',

    # Perturbation range (pixels)
    'max_shift_px': 30,
    'grid_size': 61,

    # Sharpness config
    'sharpness_radius': 5,
}

# The 7 dead channels (BatchNorm gamma ≈ 0, output ≈ 1e-7)
# Note: Using 0-indexed channel IDs
DEAD_CHANNELS = [2, 4, 7, 9, 13, 36, 48]
ALL_CHANNELS = list(range(64))
ACTIVE_CHANNELS = sorted([ch for ch in ALL_CHANNELS if ch not in DEAD_CHANNELS])
# 57 active channels

# ── Fixed combinations (Top 1 & Top 2 with dead channels removed) ──
# Top 1 original: [6,7,12,15,36,45,58,62] → dead: 7, 36 → active: [6,12,15,45,58,62]
# We need 5 channels, so we create two variants:
# Top 2 original: [8,22,23,27,28,42,48,60] → dead: 48 → active: [8,22,23,27,28,42,60]
# We pick the 5 most promising (based on Color Opponent + Edge diversity)

FIXED_COMBINATIONS = [
    # Top 1 variant A: drop Ch 62 (Grayscale Texture, redundant with Ch 58)
    {'name': 'Top1_nodead_v1', 'channels': [6, 12, 15, 45, 58]},
    # Top 1 variant B: drop Ch 58 (keep Ch 62 instead)
    {'name': 'Top1_nodead_v2', 'channels': [6, 12, 15, 45, 62]},
    # Top 2 variant A: keep 2 edges + 3 color opponents
    {'name': 'Top2_nodead_v1', 'channels': [8, 22, 23, 27, 28]},
    # Top 2 variant B: keep edges + color with more direction diversity
    {'name': 'Top2_nodead_v2', 'channels': [8, 23, 27, 42, 60]},
]

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
# Lightweight Feature Extractor (bypass CNNFeatureExtractor)
# ============================================================
class DirectConv1Extractor(nn.Module):
    """
    Directly extract arbitrary conv1 channels from ResNet18.
    No dependency on CNNFeatureExtractor — allows any channel index in [0, 63].
    Uses conv1 + bn1 (NO ReLU) to match the actual COMO system.
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
            # NO ReLU — matches COMO's actual feature extraction
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
# Core Functions (same as original random_channel_search.py)
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
# Sharpness & Basin Width Computation
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

    return {
        'x_local': x_local, 'y_local': y_local, 'local': local_combined,
        'x_global': x_global, 'y_global': y_global, 'global': global_combined,
        'basin_width': basin_width,
        'min_cost': min_cost,
        'min_location': (min_dx, min_dy),
    }


# ============================================================
# Visualization (simplified for batch runs)
# ============================================================

def plot_3d_comparison(
    costs: Dict[str, np.ndarray], dx: np.ndarray, dy: np.ndarray,
    title: str, output_path: str,
    sharpness_all: Optional[Dict[str, Dict]] = None
):
    """Side-by-side 3D: Gray vs CNN (this combination)."""
    modalities = list(costs.keys())
    n_mod = len(modalities)
    fig = plt.figure(figsize=(7 * n_mod, 7))
    DX, DY = np.meshgrid(dx, dy, indexing='ij')

    colormaps = {'gray': 'GnBu_r', 'cnn': 'YlOrRd'}

    for idx, mod in enumerate(modalities):
        ax = fig.add_subplot(1, n_mod, idx + 1, projection='3d')
        cost_data = costs[mod]
        c_min, c_max = cost_data.min(), cost_data.max()
        if c_max - c_min > 1e-10:
            cost_norm = (cost_data - c_min) / (c_max - c_min)
        else:
            cost_norm = np.zeros_like(cost_data)

        cmap = plt.get_cmap(colormaps.get(mod, 'viridis'))
        norm = Normalize(vmin=0, vmax=1)
        facecolors = cmap(norm(cost_norm))

        ax.plot_surface(DY, DX, cost_norm, facecolors=facecolors,
                        edgecolor='k', linewidth=0.12, alpha=0.92,
                        shade=True, rcount=40, ccount=40, antialiased=True)

        contour_offset = -0.05
        ax.contourf(DY, DX, cost_norm, zdir='z', offset=contour_offset,
                    levels=20, cmap='gray_r', alpha=0.7)
        ax.contour(DY, DX, cost_norm, zdir='z', offset=contour_offset,
                   levels=10, colors='k', linewidths=0.4, alpha=0.5)

        ax.set_xlabel(r'$\Delta x$ [px]', labelpad=8)
        ax.set_ylabel(r'$\Delta y$ [px]', labelpad=8)
        ax.set_zlabel('Norm. Cost', labelpad=6)

        if mod == 'gray':
            subtitle = 'Gray (1ch)'
        else:
            subtitle = 'CNN (5ch)'

        if sharpness_all and mod in sharpness_all:
            s = sharpness_all[mod]
            subtitle += (f"\nLocal={s['local']:.4f}  Global={s['global']:.4f}"
                         f"\nBW={s['basin_width']:.0f}px  MinC={s['min_cost']:.4f}")

        ax.set_title(subtitle, fontweight='bold', fontsize=11, pad=12)
        ax.set_zlim(contour_offset, 1.05)
        ax.view_init(elev=32, azim=-50)
        ax.tick_params(labelsize=8)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('lightgray')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


# ============================================================
# Main
# ============================================================

def run_single_combination(
    run_id: int,
    channel_indices: List[int],
    n_channels: int,
    frame_configs: List[Tuple[int, str]],
    all_images: List[str],
    device: str,
    output_base: str,
    cfg: dict,
    quick: bool = False,
    run_name: str = None,
) -> Dict:
    """
    Run convergence basin analysis for one channel combination.
    Returns a dict with all metrics for this run.
    """
    ch_str = ','.join(str(c) for c in channel_indices)
    if run_name:
        folder_name = f"run_{run_id:02d}_{run_name}_ch[{ch_str}]"
    else:
        folder_name = f"run_{run_id:02d}_ch[{ch_str}]"
    run_dir = os.path.join(output_base, folder_name)
    os.makedirs(run_dir, exist_ok=True)

    label = run_name if run_name else f"Random"
    print(f"\n{'='*70}")
    print(f"  Run {run_id} ({label}): Channels = [{ch_str}]  ({n_channels} ch)")
    print(f"  Output: {run_dir}")
    print(f"{'='*70}")

    # Build extractor for this combination
    extractor = DirectConv1Extractor(channel_indices, device)

    run_metrics = {
        'run_id': run_id,
        'run_name': run_name if run_name else 'random',
        'channels': ch_str,
        'n_channels': n_channels,
    }

    # Collect per-condition metrics for scoring
    local_sharpness_clean = []
    local_sharpness_30 = []
    local_sharpness_50 = []

    for frame_i, (ref_idx, frame_label) in enumerate(frame_configs):
        if ref_idx >= len(all_images):
            continue

        print(f"\n  [{frame_i+1}/{len(frame_configs)}] {frame_label}")

        rgb_np = load_image_numpy(all_images[ref_idx])
        rgb_tensor = numpy_to_tensor(rgb_np, device)

        # Reference features (always clean)
        feat_ref_gray = extract_features_numpy(rgb_tensor, None, 'gray')
        feat_ref_cnn = extract_features_numpy(rgb_tensor, extractor, 'cnn')

        for cond in BRIGHTNESS_CONDITIONS:
            cond_key = cond['key']
            cond_label = cond['label']
            cond_suffix = cond['suffix']
            bright_factor = cond['factor']
            suffix = f"_frame{frame_i+1}{cond_suffix}"

            print(f"    {cond_label}...", end=" ", flush=True)

            rgb_target_np = apply_brightness_perturbation(rgb_np, bright_factor)
            rgb_target_tensor = numpy_to_tensor(rgb_target_np, device)

            feat_target_gray = extract_features_numpy(rgb_target_tensor, None, 'gray')
            feat_target_cnn = extract_features_numpy(rgb_target_tensor, extractor, 'cnn')
            del rgb_target_tensor

            # Compute cost landscapes
            dx, dy, cost_gray = compute_2d_cost_landscape(
                feat_ref_gray, feat_target_gray,
                cfg['max_shift_px'], cfg['grid_size']
            )
            dx, dy, cost_cnn = compute_2d_cost_landscape(
                feat_ref_cnn, feat_target_cnn,
                cfg['max_shift_px'], cfg['grid_size']
            )

            # Compute sharpness
            s_gray = compute_sharpness(cost_gray, dx, dy, cfg['sharpness_radius'])
            s_cnn = compute_sharpness(cost_cnn, dx, dy, cfg['sharpness_radius'])

            # Store metrics
            frame_tag = f"f{frame_i+1}"
            for metric_key in ['local', 'global', 'x_local', 'y_local',
                               'x_global', 'y_global', 'basin_width', 'min_cost']:
                run_metrics[f'gray_{frame_tag}_{cond_key}_{metric_key}'] = s_gray[metric_key]
                run_metrics[f'cnn_{frame_tag}_{cond_key}_{metric_key}'] = s_cnn[metric_key]

            min_loc_gray = s_gray['min_location']
            min_loc_cnn = s_cnn['min_location']
            run_metrics[f'gray_{frame_tag}_{cond_key}_min_loc'] = f"({min_loc_gray[0]:+.1f},{min_loc_gray[1]:+.1f})"
            run_metrics[f'cnn_{frame_tag}_{cond_key}_min_loc'] = f"({min_loc_cnn[0]:+.1f},{min_loc_cnn[1]:+.1f})"

            # Collect scoring metrics
            if cond_key == 'clean':
                local_sharpness_clean.append(s_cnn['local'])
            elif cond_key == 'bright30':
                local_sharpness_30.append(s_cnn['local'])
            elif cond_key == 'bright50':
                local_sharpness_50.append(s_cnn['local'])

            # Print compact summary
            print(f"Gray: L={s_gray['local']:.4f} G={s_gray['global']:.4f} BW={s_gray['basin_width']:.0f} | "
                  f"CNN: L={s_cnn['local']:.4f} G={s_cnn['global']:.4f} BW={s_cnn['basin_width']:.0f}")

            # Generate plots (only comparison plots to save time)
            if not quick:
                costs = {'gray': cost_gray, 'cnn': cost_cnn}
                sharpness_all = {'gray': s_gray, 'cnn': s_cnn}
                plot_title = f"Run {run_id} — [{ch_str}] — {frame_label} — {cond_label}"
                plot_path = os.path.join(run_dir, f"basin_3d{suffix}.png")
                plot_3d_comparison(costs, dx, dy, plot_title, plot_path, sharpness_all)

        del rgb_tensor, rgb_np
        torch.cuda.empty_cache()

    # Compute scoring metrics
    mean_clean = np.mean(local_sharpness_clean) if local_sharpness_clean else 0.0
    mean_50 = np.mean(local_sharpness_50) if local_sharpness_50 else 0.0
    mean_30 = np.mean(local_sharpness_30) if local_sharpness_30 else 0.0

    # Primary: Retention = B50_L / Clean_L * 100%
    if mean_clean > 1e-10:
        retention_50 = (mean_50 / mean_clean) * 100.0
        retention_30 = (mean_30 / mean_clean) * 100.0
    else:
        retention_50 = 0.0
        retention_30 = 0.0

    run_metrics['score_retention_50'] = retention_50
    run_metrics['score_retention_30'] = retention_30
    run_metrics['score_local_clean_mean'] = mean_clean
    run_metrics['score_local_50_mean'] = mean_50
    run_metrics['score_local_30_mean'] = mean_30

    print(f"\n  >>> SCORES: Retention_50={retention_50:.1f}%  |  "
          f"Local_Clean={mean_clean:.4f}  |  Local_50={mean_50:.4f}")

    # Save per-run metrics
    metrics_path = os.path.join(run_dir, 'metrics.json')
    metrics_json = {}
    for k, v in run_metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            metrics_json[k] = float(v)
        else:
            metrics_json[k] = v
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)

    return run_metrics


def main():
    parser = argparse.ArgumentParser(description='5-Channel Search for Convergence Basin')
    parser.add_argument('--n_random_runs', type=int, default=15,
                        help='Number of random 5-channel combinations to test')
    parser.add_argument('--quick', action='store_true',
                        help='Skip plot generation for faster runs')
    args = parser.parse_args()

    cfg = CONFIG
    output_base = cfg['output_dir']
    os.makedirs(output_base, exist_ok=True)
    device = cfg['device']

    # Load image list
    all_images = sorted(glob.glob(os.path.join(cfg['rgb_dir'], "*.png")))
    n_images = len(all_images)

    frame_configs = [
        (50, "Early (frame 50)"),
        (n_images // 2, f"Middle (frame {n_images // 2})"),
        (n_images - 50, f"Late (frame {n_images - 50})"),
    ]

    n_fixed = len(FIXED_COMBINATIONS)
    n_total = n_fixed + args.n_random_runs

    print(f"{'='*70}")
    print(f"  5-CHANNEL SEARCH — Convergence Basin Analysis")
    print(f"{'='*70}")
    print(f"  Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Active channels:  {len(ACTIVE_CHANNELS)} / 64")
    print(f"  Dead channels:    {DEAD_CHANNELS}")
    print(f"  Fixed runs:       {n_fixed} (Top1/Top2 de-dead variants)")
    print(f"  Random runs:      {args.n_random_runs} x 5-ch")
    print(f"  Total runs:       {n_total}")
    print(f"  Conditions:       Clean / +30% / +50%")
    print(f"  Frames:           {len(frame_configs)}")
    print(f"  Grid:             {cfg['grid_size']}x{cfg['grid_size']}")
    print(f"  Quick mode:       {args.quick}")
    print(f"  Output:           {output_base}")
    print(f"{'='*70}")

    all_results = []
    run_counter = 1

    # ── Fixed runs: Top 1 & Top 2 de-dead variants ──
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Fixed Combinations (Top1/Top2 without dead channels)")
    print(f"{'='*70}")

    for combo in FIXED_COMBINATIONS:
        result = run_single_combination(
            run_id=run_counter,
            channel_indices=combo['channels'],
            n_channels=5,
            frame_configs=frame_configs,
            all_images=all_images,
            device=device,
            output_base=output_base,
            cfg=cfg,
            quick=args.quick,
            run_name=combo['name'],
        )
        all_results.append(result)
        run_counter += 1

    # ── Random runs: 15 x 5-ch ──
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Random Combinations ({args.n_random_runs} x 5-ch)")
    print(f"{'='*70}")

    for i in range(args.n_random_runs):
        indices = sorted(np.random.choice(ACTIVE_CHANNELS, size=5, replace=False).tolist())

        result = run_single_combination(
            run_id=run_counter,
            channel_indices=indices,
            n_channels=5,
            frame_configs=frame_configs,
            all_images=all_images,
            device=device,
            output_base=output_base,
            cfg=cfg,
            quick=args.quick,
        )
        all_results.append(result)
        run_counter += 1

    # ============================================================
    # Summary & Ranking
    # ============================================================
    print(f"\n\n{'='*100}")
    print(f"  FINAL RANKING — Sorted by Retention_50 (Primary) then Local_50 (Secondary)")
    print(f"{'='*100}")

    # Sort by primary (Retention_50) then secondary (Local_50)
    all_results.sort(key=lambda r: (-r['score_retention_50'], -r['score_local_50_mean']))

    print(f"  {'Rank':<5} {'Run':<5} {'Name':<20} {'Channels':<30} "
          f"{'Ret_50%':<10} {'Ret_30%':<10} {'L_Clean':<10} {'L_B50':<10}")
    print(f"  {'-'*110}")

    for rank, r in enumerate(all_results, 1):
        marker = " ***" if rank <= 3 else ""
        name = r.get('run_name', 'random')[:18]
        print(f"  {rank:<5} {r['run_id']:<5} {name:<20} "
              f"{r['channels']:<30} "
              f"{r['score_retention_50']:<10.1f} {r['score_retention_30']:<10.1f} "
              f"{r['score_local_clean_mean']:<10.4f} {r['score_local_50_mean']:<10.4f}{marker}")

    print(f"  {'-'*110}")
    print(f"  *** = Top 3 candidates")

    # ── Detailed view of Top 3 ──
    print(f"\n{'='*100}")
    print(f"  TOP 3 — Detailed Metrics")
    print(f"{'='*100}")

    for rank, r in enumerate(all_results[:3], 1):
        name = r.get('run_name', 'random')
        print(f"\n  #{rank}: Run {r['run_id']} ({name}) — Channels [{r['channels']}]")
        print(f"       Retention_50={r['score_retention_50']:.1f}%  "
              f"Retention_30={r['score_retention_30']:.1f}%  "
              f"L_Clean={r['score_local_clean_mean']:.4f}  "
              f"L_B50={r['score_local_50_mean']:.4f}")
        print(f"  {'Frame':<22} {'Cond':<18} {'Mod':<6} "
              f"{'Local':<9} {'Global':<9} {'BW':<8} {'MinLoc'}")
        print(f"  {'-'*90}")

        for fi, (_, frame_label) in enumerate(frame_configs, 1):
            for cond in BRIGHTNESS_CONDITIONS:
                ck = cond['key']
                cl = cond['label']
                for mod_prefix, mod_name in [('gray', 'Gray'), ('cnn', 'CNN')]:
                    loc = r.get(f'{mod_prefix}_f{fi}_{ck}_local', 0)
                    glo = r.get(f'{mod_prefix}_f{fi}_{ck}_global', 0)
                    bw = r.get(f'{mod_prefix}_f{fi}_{ck}_basin_width', 0)
                    ml = r.get(f'{mod_prefix}_f{fi}_{ck}_min_loc', '(?,?)')
                    print(f"  {frame_label:<22} {cl:<18} {mod_name:<6} "
                          f"{loc:<9.4f} {glo:<9.4f} {bw:<8.0f} {ml}")

    # ── Save summary CSV ──
    summary_path = os.path.join(output_base, 'summary_5ch.csv')
    if all_results:
        keys = all_results[0].keys()
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_results:
                row = {}
                for k, v in r.items():
                    if isinstance(v, (np.floating, np.integer)):
                        row[k] = float(v)
                    else:
                        row[k] = v
                writer.writerow(row)
        print(f"\n  Summary CSV saved: {summary_path}")

    print(f"\n{'='*70}")
    print(f"  5-CHANNEL SEARCH — COMPLETE")
    print(f"  Total runs: {len(all_results)}")
    print(f"  Output: {output_base}/")
    print(f"  Best combination: Run {all_results[0]['run_id']} — [{all_results[0]['channels']}]")
    print(f"  Best Retention_50: {all_results[0]['score_retention_50']:.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()