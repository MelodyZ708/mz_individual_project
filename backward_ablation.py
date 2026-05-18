"""
Backward Ablation — Channel Importance Discovery
===================================================
Starting from a validated 8-channel combination, iteratively remove the
least-important channel (Leave-One-Out) to discover:
  1. Which individual channels are critical (core features)
  2. Which are redundant (can be removed with minimal loss)
  3. The minimum viable channel set

Protocol:
  - Start with an 8-channel combination
  - At each step, try removing each remaining channel one at a time
  - Evaluate each (N-1)-channel subset across all sampled frames
  - Remove the channel whose removal causes the LEAST performance drop
  - Repeat: 8 → 7 → 6 → 5 → 4 → 3 → 2 → 1

Evaluation uses the same multi-frame protocol as validate_channels_multiframe.py:
  - 20 uniformly sampled frames
  - Clean / +30% / +50% brightness conditions
  - Primary metric: Mean Retention_50 across all frames
  - Secondary: Worst-case Retention_50

Outputs:
  - Ablation path CSV (performance at each step)
  - Performance drop curve plot
  - Channel importance ranking

Usage:
  python backward_ablation.py [--n_frames 20] [--top_k 5]
  python backward_ablation.py --frame_indices 41 306 512 --top_k 2

  By default, reads True Top 5 from Step 1 output.
  Can also specify combinations manually with --channels.

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
    'output_dir': 'vis_results/backward_ablation',
    'multiframe_summary': 'vis_results/multiframe_validation/summary_multiframe.csv',
    'device': 'cuda:0',

    # Perturbation range (pixels)
    'max_shift_px': 30,
    'grid_size': 61,

    # Sharpness config
    'sharpness_radius': 5,
}

BRIGHTNESS_CONDITIONS = [
    {'key': 'clean',    'factor': 0.0, 'label': 'Clean'},
    {'key': 'bright30', 'factor': 0.3, 'label': 'Brightness +30%'},
    {'key': 'bright50', 'factor': 0.5, 'label': 'Brightness +50%'},
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
# Lightweight Feature Extractor (same as validation script)
# ============================================================
class DirectConv1Extractor(nn.Module):
    def __init__(self, channel_indices: List[int], device: str = "cuda:0"):
        super().__init__()
        self.device = device
        self.channel_indices = channel_indices

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        resnet.eval()

        self.feature_extractor = nn.Sequential(
            resnet.conv1,
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


def extract_features_numpy(rgb_tensor: torch.Tensor, extractor) -> np.ndarray:
    with torch.no_grad():
        feat = extractor(rgb_tensor)
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


def compute_sharpness(
    cost_grid: np.ndarray, dx_values: np.ndarray, dy_values: np.ndarray,
    radius_px: int = 5
) -> Dict[str, float]:
    grid_size = cost_grid.shape[0]
    center = grid_size // 2
    step_x = dx_values[1] - dx_values[0]
    step_y = dy_values[1] - dy_values[0]

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

    # Basin Width
    raw_max = cost_grid.max()
    step = dx_values[1] - dx_values[0]
    if raw_max > 1e-10:
        cost_row = cost_grid[center, :]
        basin_mask = cost_row < (0.5 * raw_max)
        basin_width = np.sum(basin_mask) * step
    else:
        basin_width = 2 * dx_values[-1]

    eps = 1e-8
    sharpness_max = max(x_local, y_local)
    sharpness_min = min(x_local, y_local)
    condition_number = sharpness_max / (sharpness_min + eps)

    return {
        'x_local': x_local, 'y_local': y_local, 'local': local_combined,
        'x_global': x_global, 'y_global': y_global, 'global': global_combined,
        'basin_width': basin_width,
        'condition_number': condition_number,
    }


def sample_frame_indices(n_total: int, n_frames: int) -> List[int]:
    margin = max(10, n_total // 50)
    usable_start = margin
    usable_end = n_total - margin
    usable_range = usable_end - usable_start

    if n_frames >= usable_range:
        return list(range(usable_start, usable_end))

    step = usable_range / n_frames
    indices = [int(usable_start + i * step) for i in range(n_frames)]
    return indices


# ============================================================
# Evaluate a channel subset across multiple frames
# ============================================================

def evaluate_channel_subset(
    channels: List[int],
    frame_indices: List[int],
    all_images: List[str],
    device: str,
    cfg: dict,
    preloaded_images: Optional[Dict[int, np.ndarray]] = None,
) -> Dict:
    """
    Quick evaluation of a channel subset. Returns aggregate metrics.
    Uses preloaded images if available to avoid redundant disk I/O.
    """
    extractor = DirectConv1Extractor(channels, device)

    retentions_50 = []
    retentions_30 = []
    locals_clean = []
    locals_50 = []
    condition_numbers_50 = []

    for frame_idx in frame_indices:
        if frame_idx >= len(all_images):
            continue

        if preloaded_images and frame_idx in preloaded_images:
            rgb_np = preloaded_images[frame_idx]
        else:
            rgb_np = load_image_numpy(all_images[frame_idx])

        rgb_tensor = numpy_to_tensor(rgb_np, device)
        feat_ref = extract_features_numpy(rgb_tensor, extractor)

        clean_local = None

        for cond in BRIGHTNESS_CONDITIONS:
            cond_key = cond['key']
            bright_factor = cond['factor']

            rgb_target_np = apply_brightness_perturbation(rgb_np, bright_factor)
            rgb_target_tensor = numpy_to_tensor(rgb_target_np, device)
            feat_target = extract_features_numpy(rgb_target_tensor, extractor)
            del rgb_target_tensor

            dx, dy, cost = compute_2d_cost_landscape(
                feat_ref, feat_target,
                cfg['max_shift_px'], cfg['grid_size']
            )
            s = compute_sharpness(cost, dx, dy, cfg['sharpness_radius'])

            if cond_key == 'clean':
                clean_local = s['local']
                locals_clean.append(s['local'])

            if clean_local and clean_local > 1e-10:
                retention = (s['local'] / clean_local) * 100.0
            else:
                retention = 0.0

            if cond_key == 'bright50':
                retentions_50.append(retention)
                locals_50.append(s['local'])
                condition_numbers_50.append(s['condition_number'])
            elif cond_key == 'bright30':
                retentions_30.append(retention)

        del rgb_tensor
        torch.cuda.empty_cache()

    return {
        'channels': channels,
        'n_channels': len(channels),
        'ret50_mean': float(np.mean(retentions_50)) if retentions_50 else 0.0,
        'ret50_std': float(np.std(retentions_50)) if retentions_50 else 0.0,
        'ret50_min': float(np.min(retentions_50)) if retentions_50 else 0.0,
        'ret30_mean': float(np.mean(retentions_30)) if retentions_30 else 0.0,
        'local_clean_mean': float(np.mean(locals_clean)) if locals_clean else 0.0,
        'local_50_mean': float(np.mean(locals_50)) if locals_50 else 0.0,
        'cn50_mean': float(np.mean(condition_numbers_50)) if condition_numbers_50 else 0.0,
    }


# ============================================================
# Backward Ablation for One Starting Combination
# ============================================================

def load_ablation_checkpoint(combo_dir: str) -> Optional[Dict]:
    """
    Load a previously saved ablation_path.csv to resume from.
    Returns dict with 'ablation_path', 'current_channels', 'removal_order', 'last_step'
    or None if no checkpoint found.
    """
    csv_path = os.path.join(combo_dir, 'ablation_path.csv')
    importance_path = os.path.join(combo_dir, 'channel_importance.json')

    # If importance.json exists, this combo is fully completed
    if os.path.exists(importance_path):
        return {'completed': True, 'csv_path': csv_path}

    if not os.path.exists(csv_path):
        return None

    # Load partial results
    ablation_path = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = {
                'step': int(row['step']),
                'n_channels': int(row['n_channels']),
                'channel_str': row['channel_str'],
                'removed_channel': int(row['removed_channel']) if row['removed_channel'] else None,
                'channels': [int(c.strip()) for c in row['channel_str'].split(',')],
                'ret50_mean': float(row['ret50_mean']),
                'ret50_std': float(row['ret50_std']),
                'ret50_min': float(row['ret50_min']),
                'ret30_mean': float(row['ret30_mean']),
                'local_clean_mean': float(row['local_clean_mean']),
                'local_50_mean': float(row['local_50_mean']),
                'cn50_mean': float(row['cn50_mean']),
            }
            ablation_path.append(rec)

    if not ablation_path:
        return None

    last_rec = ablation_path[-1]
    current_channels = [int(c.strip()) for c in last_rec['channel_str'].split(',')]
    removal_order = [rec['removed_channel'] for rec in ablation_path if rec['removed_channel'] is not None]
    last_step = last_rec['step']

    return {
        'completed': False,
        'ablation_path': ablation_path,
        'current_channels': current_channels,
        'removal_order': removal_order,
        'last_step': last_step,
    }


def run_backward_ablation(
    combo_name: str,
    starting_channels: List[int],
    frame_indices: List[int],
    all_images: List[str],
    device: str,
    cfg: dict,
    output_dir: str,
    resume: bool = False,
) -> List[Dict]:
    """
    Perform backward elimination from starting_channels down to 1 channel.
    Returns the full ablation path.
    If resume=True, will try to load checkpoint and continue from where it stopped.
    """
    combo_dir = os.path.join(output_dir, combo_name)
    os.makedirs(combo_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  BACKWARD ABLATION: {combo_name}")
    print(f"  Starting channels: {starting_channels}")
    print(f"  Frames: {len(frame_indices)}")
    print(f"{'='*70}")

    # ── Check for resume ──
    if resume:
        checkpoint = load_ablation_checkpoint(combo_dir)
        if checkpoint and checkpoint.get('completed'):
            print(f"  [RESUME] This combo is already COMPLETE. Skipping.")
            # Reload the full path from CSV for plotting
            reloaded = load_ablation_checkpoint(combo_dir)
            # Re-read the CSV to get the full path
            csv_path = os.path.join(combo_dir, 'ablation_path.csv')
            ablation_path = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rec = {
                        'step': int(row['step']),
                        'n_channels': int(row['n_channels']),
                        'channel_str': row['channel_str'],
                        'removed_channel': int(row['removed_channel']) if row['removed_channel'] else None,
                        'channels': [int(c.strip()) for c in row['channel_str'].split(',')],
                        'ret50_mean': float(row['ret50_mean']),
                        'ret50_std': float(row['ret50_std']),
                        'ret50_min': float(row['ret50_min']),
                        'ret30_mean': float(row['ret30_mean']),
                        'local_clean_mean': float(row['local_clean_mean']),
                        'local_50_mean': float(row['local_50_mean']),
                        'cn50_mean': float(row['cn50_mean']),
                    }
                    ablation_path.append(rec)
            return ablation_path

        elif checkpoint and not checkpoint.get('completed'):
            print(f"  [RESUME] Found checkpoint at Step {checkpoint['last_step']} "
                  f"({checkpoint['current_channels']})")
            print(f"  [RESUME] Continuing from {len(checkpoint['current_channels'])} channels...")
            ablation_path = checkpoint['ablation_path']
            current_channels = checkpoint['current_channels']
            removal_order = checkpoint['removal_order']
            step = checkpoint['last_step']

            # Preload images
            print(f"  Preloading {len(frame_indices)} images...", flush=True)
            preloaded = {}
            for fi in frame_indices:
                if fi < len(all_images):
                    preloaded[fi] = load_image_numpy(all_images[fi])
            print(f"  Done. ({len(preloaded)} images loaded)")

            # Jump to the iterative removal loop below
            # (current_channels, ablation_path, removal_order, step are all set)
            return _continue_ablation(
                combo_name, starting_channels, current_channels, ablation_path,
                removal_order, step, frame_indices, all_images, device, cfg,
                output_dir, preloaded
            )
        else:
            print(f"  [RESUME] No checkpoint found. Starting fresh.")

    # Preload images for efficiency
    print(f"  Preloading {len(frame_indices)} images...", flush=True)
    preloaded = {}
    for fi in frame_indices:
        if fi < len(all_images):
            preloaded[fi] = load_image_numpy(all_images[fi])
    print(f"  Done. ({len(preloaded)} images loaded)")

    current_channels = list(starting_channels)
    ablation_path = []
    removal_order = []  # Track which channel was removed at each step

    # ── Step 0: Evaluate the full starting set ──
    print(f"\n  Step 0: Evaluate full set ({len(current_channels)} ch) [{','.join(str(c) for c in current_channels)}]")
    result_full = evaluate_channel_subset(
        current_channels, frame_indices, all_images, device, cfg, preloaded
    )
    result_full['step'] = 0
    result_full['removed_channel'] = None
    result_full['channel_str'] = ','.join(str(c) for c in current_channels)
    ablation_path.append(result_full)

    print(f"    Ret50={result_full['ret50_mean']:.1f}% ± {result_full['ret50_std']:.1f}% "
          f"(min={result_full['ret50_min']:.1f}%) CN={result_full['cn50_mean']:.2f}")

    return _continue_ablation(
        combo_name, starting_channels, current_channels, ablation_path,
        removal_order, 0, frame_indices, all_images, device, cfg,
        output_dir, preloaded
    )


def _continue_ablation(
    combo_name: str,
    starting_channels: List[int],
    current_channels: List[int],
    ablation_path: List[Dict],
    removal_order: List[int],
    step: int,
    frame_indices: List[int],
    all_images: List[str],
    device: str,
    cfg: dict,
    output_dir: str,
    preloaded: Dict,
) -> List[Dict]:
    """
    Continue the iterative removal loop. Shared by both fresh and resumed runs.
    """
    combo_dir = os.path.join(output_dir, combo_name)

    # ── Iterative removal ──
    while len(current_channels) > 1:
        step += 1
        n_remaining = len(current_channels)
        print(f"\n  Step {step}: Leave-One-Out from {n_remaining} channels "
              f"[{','.join(str(c) for c in current_channels)}]")

        best_subset_result = None
        best_removed = None
        best_ret50 = -float('inf')

        for ch_to_remove in current_channels:
            subset = [c for c in current_channels if c != ch_to_remove]
            result = evaluate_channel_subset(
                subset, frame_indices, all_images, device, cfg, preloaded
            )

            # Print compact result
            print(f"    Remove Ch{ch_to_remove:2d} → [{','.join(str(c) for c in subset)}] "
                  f"Ret50={result['ret50_mean']:.1f}% (min={result['ret50_min']:.1f}%)")

            # Primary: highest mean retention; Secondary: highest worst-case
            if (result['ret50_mean'] > best_ret50 + 0.1 or
                (abs(result['ret50_mean'] - best_ret50) <= 0.1 and
                 result['ret50_min'] > (best_subset_result['ret50_min'] if best_subset_result else -999))):
                best_ret50 = result['ret50_mean']
                best_subset_result = result
                best_removed = ch_to_remove

        # Record the removal
        current_channels = [c for c in current_channels if c != best_removed]
        removal_order.append(best_removed)

        best_subset_result['step'] = step
        best_subset_result['removed_channel'] = best_removed
        best_subset_result['channel_str'] = ','.join(str(c) for c in current_channels)
        ablation_path.append(best_subset_result)

        print(f"  → Removed Ch{best_removed} (least important at this step)")
        print(f"    New set: [{','.join(str(c) for c in current_channels)}] "
              f"Ret50={best_subset_result['ret50_mean']:.1f}% ± {best_subset_result['ret50_std']:.1f}%")

        # ── Save intermediate checkpoint after each step ──
        csv_path_tmp = os.path.join(combo_dir, 'ablation_path.csv')
        fieldnames_tmp = ['step', 'n_channels', 'channel_str', 'removed_channel',
                          'ret50_mean', 'ret50_std', 'ret50_min', 'ret30_mean',
                          'local_clean_mean', 'local_50_mean', 'cn50_mean']
        with open(csv_path_tmp, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_tmp)
            writer.writeheader()
            for rec in ablation_path:
                row = {k: rec.get(k, '') for k in fieldnames_tmp}
                writer.writerow(row)
        print(f"    [Checkpoint saved: step {step}]")


    # ── Save ablation path (final) ──
    csv_path = os.path.join(combo_dir, 'ablation_path.csv')
    fieldnames = ['step', 'n_channels', 'channel_str', 'removed_channel',
                  'ret50_mean', 'ret50_std', 'ret50_min', 'ret30_mean',
                  'local_clean_mean', 'local_50_mean', 'cn50_mean']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in ablation_path:
            row = {k: rec.get(k, '') for k in fieldnames}
            writer.writerow(row)
    print(f"\n  Ablation path saved: {csv_path}")

    # ── Channel importance ranking ──
    # Channels removed LAST are the most important (hardest to remove)
    importance = list(reversed(removal_order))
    importance.append(current_channels[0])  # The last surviving channel
    importance = list(reversed(importance))  # Now: last_survivor first, then removed-last, ..., removed-first

    # Actually, let's think about this more carefully:
    # removal_order = [first_removed, second_removed, ..., second_to_last_removed]
    # current_channels = [last_survivor]
    # Importance: last_survivor > second_to_last_removed > ... > first_removed
    importance_ranking = [current_channels[0]] + list(reversed(removal_order))

    print(f"\n  Channel Importance Ranking (most → least important):")
    for rank, ch in enumerate(importance_ranking, 1):
        marker = " ← CORE" if rank <= 3 else (" ← useful" if rank <= 5 else " ← redundant")
        print(f"    #{rank}: Ch{ch}{marker}")

    importance_path = os.path.join(combo_dir, 'channel_importance.json')
    with open(importance_path, 'w') as f:
        json.dump({
            'combo_name': combo_name,
            'starting_channels': starting_channels,
            'importance_ranking': importance_ranking,
            'removal_order': removal_order,
            'last_survivor': current_channels[0],
        }, f, indent=2)

    return ablation_path


# ============================================================
# Visualization
# ============================================================

def plot_ablation_curve(all_ablation_paths: Dict[str, List[Dict]], output_dir: str):
    """
    Plot performance drop curve for all combinations.
    X-axis: number of channels remaining.
    Y-axis: Mean Retention_50.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_ablation_paths)))

    # ── Left: Mean Retention_50 ──
    ax = axes[0]
    for idx, (name, path) in enumerate(all_ablation_paths.items()):
        n_chs = [p['n_channels'] for p in path]
        ret50s = [p['ret50_mean'] for p in path]
        short_name = name.replace('Rank', 'R').replace('_Run', ' R')
        ax.plot(n_chs, ret50s, 'o-', color=colors[idx], label=short_name,
                linewidth=2, markersize=6)

        # Annotate removed channels
        for p in path:
            if p['removed_channel'] is not None:
                ax.annotate(f"-Ch{p['removed_channel']}",
                           (p['n_channels'], p['ret50_mean']),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=6, color=colors[idx], alpha=0.7)

    ax.axhline(100, color='green', linestyle=':', alpha=0.5, label='100% baseline')
    ax.set_xlabel('Number of Channels')
    ax.set_ylabel('Mean Retention_50 (%)')
    ax.set_title('Backward Ablation: Performance vs Channel Count', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    ax.invert_xaxis()  # 8 on left, 1 on right
    ax.set_xticks(range(1, 9))

    # ── Right: Worst-case Retention_50 ──
    ax = axes[1]
    for idx, (name, path) in enumerate(all_ablation_paths.items()):
        n_chs = [p['n_channels'] for p in path]
        ret50_mins = [p['ret50_min'] for p in path]
        short_name = name.replace('Rank', 'R').replace('_Run', ' R')
        ax.plot(n_chs, ret50_mins, 's--', color=colors[idx], label=short_name,
                linewidth=1.5, markersize=5)

    ax.axhline(100, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Channels')
    ax.set_ylabel('Worst-case Retention_50 (%)')
    ax.set_title('Backward Ablation: Robustness (Worst Frame)', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    ax.invert_xaxis()
    ax.set_xticks(range(1, 9))

    plt.tight_layout()
    path = os.path.join(output_dir, 'ablation_curves.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_channel_importance_comparison(all_ablation_paths: Dict[str, List[Dict]], output_dir: str):
    """
    Bar chart showing how often each channel appears in the top-3 most important
    across all tested combinations.
    """
    importance_counts = {}  # ch -> count of times in top-3 importance

    for name, path in all_ablation_paths.items():
        # The last 3 channels to survive are the most important
        if len(path) >= 3:
            # path[-1] has 1 channel, path[-2] has 2, path[-3] has 3
            for p in path[-3:]:
                for ch in p['channels']:
                    importance_counts[ch] = importance_counts.get(ch, 0) + 1

    if not importance_counts:
        return

    # Sort by count
    sorted_chs = sorted(importance_counts.items(), key=lambda x: -x[1])
    channels = [str(c[0]) for c in sorted_chs]
    counts = [c[1] for c in sorted_chs]

    fig, ax = plt.subplots(figsize=(max(10, len(channels) * 0.5), 5))
    bars = ax.bar(channels, counts, color='steelblue', edgecolor='black', linewidth=0.5)

    # Highlight channels that appear in ALL combinations' top-3
    max_count = max(counts)
    for bar, count in zip(bars, counts):
        if count == max_count:
            bar.set_color('darkred')
            bar.set_alpha(0.8)

    ax.set_xlabel('Channel Index')
    ax.set_ylabel(f'Times in Top-3 Importance (out of {len(all_ablation_paths)} combos)')
    ax.set_title('Cross-Combination Channel Importance\n'
                 '(Red = appears in ALL combinations\' core set)',
                 fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path_out = os.path.join(output_dir, 'channel_importance_frequency.png')
    plt.savefig(path_out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path_out}")


# ============================================================
# Load True Top-K from Step 1
# ============================================================

def load_true_top_k(summary_path: str, top_k: int = 5) -> List[Dict]:
    """
    Read the multi-frame validation summary and return the top-K combinations.
    """
    if not os.path.exists(summary_path):
        print(f"  [Warning] Summary file not found: {summary_path}")
        print(f"  Using hardcoded Top 10 from initial search instead.")
        return None

    combos = []
    with open(summary_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            combos.append(row)

    # Sort by bright50_cnn_retention_mean (descending)
    combos.sort(key=lambda r: -float(r.get('bright50_cnn_retention_mean', 0)))

    result = []
    for i, c in enumerate(combos[:top_k]):
        channels = [int(ch.strip()) for ch in c['channels'].split(',')]
        result.append({
            'name': c.get('combo_name', f'Top{i+1}'),
            'channels': channels,
        })

    return result


# ============================================================
# Fallback: Hardcoded Top 5 from initial search
# ============================================================
FALLBACK_TOP5 = [
    {'name': 'Rank01_Run05',  'channels': [6, 7, 12, 15, 36, 45, 58, 62]},
    {'name': 'Rank02_Run01',  'channels': [8, 22, 23, 27, 28, 42, 48, 60]},
    {'name': 'Rank03_Run16',  'channels': [1, 4, 7, 13, 22, 34, 41, 54]},
    {'name': 'Rank04_Run12',  'channels': [8, 9, 17, 19, 37, 41, 51, 52]},
    {'name': 'Rank05_Run20',  'channels': [2, 9, 19, 20, 28, 49, 55, 58]},
]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Backward Ablation — Channel Importance Discovery')
    parser.add_argument('--n_frames', type=int, default=20,
                        help='Number of frames to sample uniformly (default: 20). '
                             'Ignored if --frame_indices is specified.')
    parser.add_argument('--frame_indices', type=int, nargs='+', default=None,
                        help='Manually specify frame indices, e.g. --frame_indices 41 306 512')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top combinations to ablate (default: 5)')
    parser.add_argument('--channels', type=str, default='',
                        help='Manually specify a single combination, e.g. "6,7,12,15,36,45,58,62"')
    parser.add_argument('--use_fallback', action='store_true',
                        help='Skip loading from Step 1 and use hardcoded Top 5')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run: skip completed combos, '
                             'continue interrupted combos from last checkpoint')
    args = parser.parse_args()

    cfg = CONFIG
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    device = cfg['device']

    # Load image list
    all_images = sorted(glob.glob(os.path.join(cfg['rgb_dir'], "*.png")))
    n_images = len(all_images)

    if n_images == 0:
        print(f"[ERROR] No images found in {cfg['rgb_dir']}")
        return

    # Sample frames
    if args.frame_indices is not None:
        # Use manually specified frame indices
        frame_indices = [fi for fi in args.frame_indices if fi < n_images]
        if len(frame_indices) == 0:
            print(f"[ERROR] All specified frame indices are out of range (total images: {n_images})")
            return
        if len(frame_indices) < len(args.frame_indices):
            print(f"  [Warning] Some frame indices were out of range and removed.")
        print(f"  Using manually specified frame indices: {frame_indices}")
    else:
        frame_indices = sample_frame_indices(n_images, args.n_frames)

    # Determine which combinations to ablate
    if args.channels:
        # Manual specification
        channels = [int(ch.strip()) for ch in args.channels.split(',')]
        combos = [{'name': f'Manual_{"_".join(str(c) for c in channels)}', 'channels': channels}]
    elif args.use_fallback:
        combos = FALLBACK_TOP5[:args.top_k]
    else:
        # Try loading from Step 1 output
        loaded = load_true_top_k(cfg['multiframe_summary'], args.top_k)
        if loaded:
            combos = loaded
        else:
            combos = FALLBACK_TOP5[:args.top_k]

    print(f"\n{'='*70}")
    print(f"  BACKWARD ABLATION — Channel Importance Discovery")
    print(f"{'='*70}")
    print(f"  Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total images:     {n_images}")
    print(f"  Sampled frames:   {len(frame_indices)}")
    print(f"  Frame indices:    {frame_indices}")
    print(f"  Combinations:     {len(combos)}")
    for c in combos:
        print(f"    {c['name']}: {c['channels']}")
    print(f"  Output:           {output_dir}")
    print(f"{'='*70}")

    # ── Run ablation for each combination ──
    all_ablation_paths = {}

    for combo in combos:
        path = run_backward_ablation(
            combo_name=combo['name'],
            starting_channels=combo['channels'],
            frame_indices=frame_indices,
            all_images=all_images,
            device=device,
            cfg=cfg,
            output_dir=output_dir,
            resume=args.resume,
        )
        all_ablation_paths[combo['name']] = path

    # ── Generate comparison plots ──
    print(f"\n  Generating comparison visualizations...")
    plot_ablation_curve(all_ablation_paths, output_dir)
    plot_channel_importance_comparison(all_ablation_paths, output_dir)

    # ── Save combined summary ──
    summary_path = os.path.join(output_dir, 'ablation_summary.csv')
    all_records = []
    for name, path in all_ablation_paths.items():
        for rec in path:
            rec_out = {
                'combo_name': name,
                'step': rec['step'],
                'n_channels': rec['n_channels'],
                'channel_str': rec['channel_str'],
                'removed_channel': rec['removed_channel'],
                'ret50_mean': rec['ret50_mean'],
                'ret50_std': rec['ret50_std'],
                'ret50_min': rec['ret50_min'],
                'ret30_mean': rec['ret30_mean'],
                'local_clean_mean': rec['local_clean_mean'],
                'local_50_mean': rec['local_50_mean'],
                'cn50_mean': rec['cn50_mean'],
            }
            all_records.append(rec_out)

    if all_records:
        fieldnames = all_records[0].keys()
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in all_records:
                writer.writerow(rec)
        print(f"  Combined summary saved: {summary_path}")

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  BACKWARD ABLATION — COMPLETE")
    print(f"{'='*70}")

    for name, path in all_ablation_paths.items():
        print(f"\n  {name}:")
        importance_path = os.path.join(output_dir, name, 'channel_importance.json')
        if os.path.exists(importance_path):
            with open(importance_path, 'r') as f:
                imp = json.load(f)
            print(f"    Importance: {' > '.join(f'Ch{c}' for c in imp['importance_ranking'])}")
            print(f"    Core (top 3): {imp['importance_ranking'][:3]}")

        # Find the "knee" — where performance drops sharply
        for i in range(1, len(path)):
            drop = path[i-1]['ret50_mean'] - path[i]['ret50_mean']
            if drop > 10:  # >10% drop = significant
                print(f"    ⚠ Sharp drop at {path[i-1]['n_channels']}→{path[i]['n_channels']} ch "
                      f"(Ret50: {path[i-1]['ret50_mean']:.1f}%→{path[i]['ret50_mean']:.1f}%, "
                      f"removed Ch{path[i]['removed_channel']})")

    print(f"\n  Output: {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()