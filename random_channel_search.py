"""
Random Channel Search — Convergence Basin Analysis
====================================================
Randomly sample different combinations of conv1 channels (from the 56 active
channels out of 64 total) and evaluate their convergence basin quality under
brightness changes.

Design (following supervisor guidance):
  - Each run: randomly pick N channels (default 8) from the 56 active conv1
    channels (bypassing CNNFeatureExtractor to allow arbitrary channel indices).
  - For each combination, compute the convergence basin (translation-based)
    under Clean / +30% / +50% brightness conditions.
  - Record all metrics (Local/Global Sharpness, Basin Width, MinCost) to CSV.
  - Generate basin plots for each combination in its own subfolder.
  - At the end, rank all combinations and identify the best ones.

Scoring:
  Primary:   Mean Basin Width under +50% brightness (across 3 frames)
  Secondary: Mean Local Sharpness under +50% brightness (across 3 frames)

Usage:
  cd /vol/bitbucket/mz325/individual_project
  python random_channel_search.py [--n_runs 20] [--n_channels 8] [--quick]

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
    'output_dir': 'vis_results/channel_search',
    'device': 'cuda:0',

    # Perturbation range (pixels)
    'max_shift_px': 30,
    'grid_size': 61,

    # Sharpness config
    'sharpness_radius': 5,
}

# The 8 dead channels (BatchNorm gamma/beta/mean/var all zero)
# Identified from conv1 → bn1 → relu analysis
DEAD_CHANNELS = [0, 11, 21, 29, 35, 39, 43, 53]
ALL_CHANNELS = list(range(64))
ACTIVE_CHANNELS = sorted([ch for ch in ALL_CHANNELS if ch not in DEAD_CHANNELS])
# 56 active channels

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
# Core Functions (same as visualize_convergence_basin_translation.py)
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
            subtitle = f'CNN ({costs[mod].shape[-1] if hasattr(costs[mod], "shape") else "?"}ch)'
            # Get actual channel count from sharpness metadata if available
            subtitle = f'CNN'

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
) -> Dict:
    """
    Run convergence basin analysis for one channel combination.
    Returns a dict with all metrics for this run.
    """
    ch_str = ','.join(str(c) for c in channel_indices)
    folder_name = f"run_{run_id:02d}_ch[{ch_str}]"
    run_dir = os.path.join(output_base, folder_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Run {run_id}: Channels = [{ch_str}]  ({n_channels} ch)")
    print(f"  Output: {run_dir}")
    print(f"{'='*70}")

    # Build extractor for this combination
    extractor = DirectConv1Extractor(channel_indices, device)

    run_metrics = {
        'run_id': run_id,
        'channels': ch_str,
        'n_channels': n_channels,
    }

    # Collect per-condition metrics for scoring
    basin_widths_50 = []
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
            _, _, cost_cnn = compute_2d_cost_landscape(
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

            # Collect scoring metrics (only +50%)
            if cond_key == 'bright50':
                basin_widths_50.append(s_cnn['basin_width'])
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
    run_metrics['score_bw50_mean'] = np.mean(basin_widths_50) if basin_widths_50 else 0.0
    run_metrics['score_local50_mean'] = np.mean(local_sharpness_50) if local_sharpness_50 else 0.0

    # Save per-run metrics
    metrics_path = os.path.join(run_dir, 'metrics.json')
    # Convert numpy types for JSON serialization
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
    parser = argparse.ArgumentParser(description='Random Channel Search for Convergence Basin')
    parser.add_argument('--n_runs', type=int, default=20,
                        help='Number of random channel combinations to test')
    parser.add_argument('--n_channels', type=int, default=8,
                        help='Number of channels per combination')
    parser.add_argument('--quick', action='store_true',
                        help='Skip plot generation for faster runs')
    parser.add_argument('--extra_sizes', type=str, default='',
                        help='Comma-separated extra channel counts to try, e.g. "12,16"')
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

    print(f"{'='*70}")
    print(f"  RANDOM CHANNEL SEARCH — Convergence Basin Analysis")
    print(f"{'='*70}")
    print(f"  Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Active channels:  {len(ACTIVE_CHANNELS)} / 64")
    print(f"  Dead channels:    {DEAD_CHANNELS}")
    print(f"  Runs:             {args.n_runs} x {args.n_channels}-ch")
    if args.extra_sizes:
        print(f"  Extra sizes:      {args.extra_sizes}")
    print(f"  Conditions:       Clean / +30% / +50%")
    print(f"  Frames:           {len(frame_configs)}")
    print(f"  Grid:             {cfg['grid_size']}x{cfg['grid_size']}")
    print(f"  Quick mode:       {args.quick}")
    print(f"  Output:           {output_base}")
    print(f"{'='*70}")

    all_results = []
    run_counter = 1

    # ── Main runs: n_runs x n_channels ──
    for i in range(args.n_runs):
        # Random sample from active channels (no fixed seed!)
        indices = sorted(np.random.choice(ACTIVE_CHANNELS, size=args.n_channels, replace=False).tolist())

        result = run_single_combination(
            run_id=run_counter,
            channel_indices=indices,
            n_channels=args.n_channels,
            frame_configs=frame_configs,
            all_images=all_images,
            device=device,
            output_base=output_base,
            cfg=cfg,
            quick=args.quick,
        )
        all_results.append(result)
        run_counter += 1

    # ── Extra sizes (e.g. 12, 16 channels) ──
    if args.extra_sizes:
        for size_str in args.extra_sizes.split(','):
            extra_n = int(size_str.strip())
            # Run 5 random combinations for each extra size
            n_extra_runs = 5
            print(f"\n{'='*70}")
            print(f"  EXTRA: {n_extra_runs} runs with {extra_n} channels")
            print(f"{'='*70}")
            for i in range(n_extra_runs):
                indices = sorted(np.random.choice(ACTIVE_CHANNELS, size=extra_n, replace=False).tolist())
                result = run_single_combination(
                    run_id=run_counter,
                    channel_indices=indices,
                    n_channels=extra_n,
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
    print(f"  FINAL RANKING — Sorted by Mean Basin Width under +50% Brightness")
    print(f"{'='*100}")

    # Sort by primary (BW50) then secondary (Local50)
    all_results.sort(key=lambda r: (-r['score_bw50_mean'], -r['score_local50_mean']))

    print(f"  {'Rank':<5} {'Run':<5} {'#Ch':<4} {'Channels':<45} "
          f"{'BW50 Mean':<12} {'Local50 Mean':<14} {'BW30 Mean':<12}")
    print(f"  {'-'*110}")

    for rank, r in enumerate(all_results, 1):
        # Also compute BW30 mean for reference
        bw30_vals = []
        for fi in range(1, 4):
            key = f"cnn_f{fi}_bright30_basin_width"
            if key in r:
                bw30_vals.append(r[key])
        bw30_mean = np.mean(bw30_vals) if bw30_vals else 0.0

        marker = " ***" if rank <= 3 else ""
        print(f"  {rank:<5} {r['run_id']:<5} {r['n_channels']:<4} "
              f"{r['channels']:<45} "
              f"{r['score_bw50_mean']:<12.1f} {r['score_local50_mean']:<14.4f} "
              f"{bw30_mean:<12.1f}{marker}")

    print(f"  {'-'*110}")
    print(f"  *** = Top 3 candidates for further evaluation")

    # ── Detailed view of Top 3 ──
    print(f"\n{'='*100}")
    print(f"  TOP 3 — Detailed Metrics")
    print(f"{'='*100}")

    for rank, r in enumerate(all_results[:3], 1):
        print(f"\n  #{rank}: Run {r['run_id']} — Channels [{r['channels']}] ({r['n_channels']} ch)")
        print(f"  {'Frame':<22} {'Cond':<18} {'Mod':<6} "
              f"{'Local':<9} {'Global':<9} {'X-Loc':<9} {'Y-Loc':<9} "
              f"{'BW':<8} {'MinCost':<10} {'MinLoc'}")
        print(f"  {'-'*110}")

        for fi, (_, frame_label) in enumerate(frame_configs, 1):
            for cond in BRIGHTNESS_CONDITIONS:
                ck = cond['key']
                cl = cond['label']
                for mod_prefix, mod_name in [('gray', 'Gray'), ('cnn', 'CNN')]:
                    loc = r.get(f'{mod_prefix}_f{fi}_{ck}_local', 0)
                    glo = r.get(f'{mod_prefix}_f{fi}_{ck}_global', 0)
                    xl = r.get(f'{mod_prefix}_f{fi}_{ck}_x_local', 0)
                    yl = r.get(f'{mod_prefix}_f{fi}_{ck}_y_local', 0)
                    bw = r.get(f'{mod_prefix}_f{fi}_{ck}_basin_width', 0)
                    mc = r.get(f'{mod_prefix}_f{fi}_{ck}_min_cost', 0)
                    ml = r.get(f'{mod_prefix}_f{fi}_{ck}_min_loc', '(?,?)')
                    print(f"  {frame_label:<22} {cl:<18} {mod_name:<6} "
                          f"{loc:<9.4f} {glo:<9.4f} {xl:<9.4f} {yl:<9.4f} "
                          f"{bw:<8.0f} {mc:<10.6f} {ml}")

    # ── Post-hoc channel frequency analysis ──
    print(f"\n{'='*100}")
    print(f"  POST-HOC ANALYSIS — Channel Frequency in Top 5 vs Bottom 5")
    print(f"{'='*100}")

    top5 = all_results[:5]
    bottom5 = all_results[-5:]

    top_channels = defaultdict(int)
    bottom_channels = defaultdict(int)

    for r in top5:
        for ch in r['channels'].split(','):
            top_channels[int(ch)] += 1
    for r in bottom5:
        for ch in r['channels'].split(','):
            bottom_channels[int(ch)] += 1

    # Find channels that appear much more in top than bottom
    all_seen = sorted(set(list(top_channels.keys()) + list(bottom_channels.keys())))
    print(f"  {'Channel':<10} {'Top5 Count':<12} {'Bot5 Count':<12} {'Diff':<8} {'Note'}")
    print(f"  {'-'*60}")
    for ch in all_seen:
        tc = top_channels.get(ch, 0)
        bc = bottom_channels.get(ch, 0)
        diff = tc - bc
        note = ""
        if tc >= 3 and bc == 0:
            note = "<-- Strong winner"
        elif bc >= 3 and tc == 0:
            note = "<-- Avoid"
        elif diff >= 2:
            note = "<-- Promising"
        print(f"  {ch:<10} {tc:<12} {bc:<12} {diff:<+8} {note}")

    # ── Save summary CSV ──
    summary_path = os.path.join(output_base, 'summary.csv')
    if all_results:
        keys = all_results[0].keys()
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in all_results:
                # Convert numpy types
                row = {}
                for k, v in r.items():
                    if isinstance(v, (np.floating, np.integer)):
                        row[k] = float(v)
                    else:
                        row[k] = v
                writer.writerow(row)
        print(f"\n  Summary CSV saved: {summary_path}")

    print(f"\n{'='*70}")
    print(f"  RANDOM CHANNEL SEARCH — COMPLETE")
    print(f"  Total runs: {len(all_results)}")
    print(f"  Output: {output_base}/")
    print(f"  Best combination: Run {all_results[0]['run_id']} — [{all_results[0]['channels']}]")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()