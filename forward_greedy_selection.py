"""
Forward Greedy Selection — Channel Combination Discovery (v3)
===============================================================
Starting from individual channels within a validated combination, greedily
build up the optimal subset by adding the channel that maximizes a composite
objective function at each step.

This is the complement to Backward Ablation:
  - Backward: 8->1, removes least important channel each step
  - Forward:  1->N, adds most beneficial channel each step

Key improvements in v3 (over v2):
  1. Exhaustive search for small pools: when alive channels <= 5, enumerate
     all 2^N - 1 non-empty subsets instead of greedy. Guarantees global optimum.
  2. Look-ahead deduplication: sorted([ch1,ch2]) used as key to avoid
     evaluating the same pair twice (e.g. +6+45 and +45+6).
  3. Path pruning: seeds whose Score < 50% of the best seed are skipped,
     since they cannot realistically overtake the best path.
  4. Resume support: --resume flag reads existing selection_summary.json
     and single_channel_rankings.csv to skip completed phases and reuse
     single-channel evaluations.

Protocol:
  Phase 1: Rank01 internal forward selection
    - Filter dead channels -> if alive <= 5: exhaustive, else: greedy
  Phase 2: Rank02 internal forward selection (same)
  Phase 3: Cross-combination forward selection
    - Seed = top channels from Phase 1 + Phase 2
    - Candidate pool = union of alive channels from both combinations

Objective Function (Composite Score, alpha=0.7):
  Score = 0.7 * Sharpness_+50% + 0.3 * Sharpness_Clean

Usage:
  python forward_greedy_selection.py
  python forward_greedy_selection.py --frame_indices 41 306 512
  python forward_greedy_selection.py --resume
  python forward_greedy_selection.py --alpha 0.7 --max_paths 3

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
from itertools import combinations
from typing import Tuple, List, Dict, Optional
from datetime import datetime


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/home/melody/data/tum/rgbd_dataset_freiburg2_desk/rgb/',
    'output_dir': 'vis_results/forward_greedy_selection',
    'device': 'cuda:0',

    # Perturbation range (pixels)
    'max_shift_px': 30,
    'grid_size': 61,

    # Sharpness config
    'sharpness_radius': 5,

    # Composite score weight (alpha * Sharp_50 + (1-alpha) * Sharp_clean)
    'alpha': 0.7,

    # Exhaustive search threshold: if alive pool <= this, enumerate all subsets
    'exhaustive_threshold': 5,

    # Path pruning: skip seeds whose Score < this fraction of the best seed
    'seed_prune_ratio': 0.5,
}

# ── Channel combinations from previous experiments ──
RANK01_CHANNELS = [6, 7, 12, 15, 36, 45, 58, 62]
RANK02_CHANNELS = [8, 22, 23, 27, 28, 42, 48, 60]

# ── Dead channel lists (from activation analysis) ──
# Always dead: kill% = 100% under BOTH Clean and +50% (BN gamma ~ 0)
ALWAYS_DEAD_CHANNELS = [7, 36, 48]

# Dead at +50%: alive under Clean but kill% -> 100% at +50% brightness
DEAD_AT_50_CHANNELS = [2, 3, 4, 9, 12, 13, 29, 35, 38, 54, 58, 62]

# Combined: all channels that should be excluded from forward selection
ALL_DEAD_CHANNELS = sorted(list(set(ALWAYS_DEAD_CHANNELS + DEAD_AT_50_CHANNELS)))

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
# Lightweight Feature Extractor (same as backward_ablation.py)
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
# Core Functions (same as backward_ablation.py)
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


def filter_dead_channels(candidate_pool: List[int]) -> Tuple[List[int], List[int]]:
    """Remove dead channels from candidate pool."""
    alive = [ch for ch in candidate_pool if ch not in ALL_DEAD_CHANNELS]
    dead = [ch for ch in candidate_pool if ch in ALL_DEAD_CHANNELS]
    return alive, dead


# ============================================================
# Evaluate a channel subset
# ============================================================

def evaluate_channel_subset(
    channels: List[int],
    frame_indices: List[int],
    all_images: List[str],
    device: str,
    cfg: dict,
    preloaded_images: Optional[Dict[int, np.ndarray]] = None,
) -> Dict:
    """Evaluate a channel subset. Returns aggregate metrics including composite score."""
    alpha = cfg.get('alpha', 0.7)
    extractor = DirectConv1Extractor(channels, device)

    retentions_50 = []
    retentions_30 = []
    locals_clean = []
    locals_50 = []
    condition_numbers_clean = []
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
                condition_numbers_clean.append(s['condition_number'])

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

    # Compute composite score
    mean_sharp_clean = float(np.mean(locals_clean)) if locals_clean else 0.0
    mean_sharp_50 = float(np.mean(locals_50)) if locals_50 else 0.0
    composite_score = alpha * mean_sharp_50 + (1 - alpha) * mean_sharp_clean

    n_survive_50 = sum(1 for ch in channels if ch not in ALL_DEAD_CHANNELS)
    survival_rate = n_survive_50 / len(channels) * 100.0 if channels else 0.0

    return {
        'channels': channels,
        'n_channels': len(channels),
        'composite_score': composite_score,
        'ret50_mean': float(np.mean(retentions_50)) if retentions_50 else 0.0,
        'ret50_std': float(np.std(retentions_50)) if retentions_50 else 0.0,
        'ret50_min': float(np.min(retentions_50)) if retentions_50 else 0.0,
        'ret30_mean': float(np.mean(retentions_30)) if retentions_30 else 0.0,
        'local_clean_mean': mean_sharp_clean,
        'local_50_mean': mean_sharp_50,
        'cn_clean_mean': float(np.mean(condition_numbers_clean)) if condition_numbers_clean else 0.0,
        'cn50_mean': float(np.mean(condition_numbers_50)) if condition_numbers_50 else 0.0,
        'n_survive_50': n_survive_50,
        'survival_rate': survival_rate,
    }


# ============================================================
# Resume helpers
# ============================================================

def _load_phase_summary(phase_dir: str) -> Optional[Dict]:
    """Try to load a completed phase summary."""
    summary_path = os.path.join(phase_dir, 'selection_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def _load_single_rankings_csv(phase_dir: str) -> Optional[List[Dict]]:
    """Try to load single-channel rankings from CSV."""
    csv_path = os.path.join(phase_dir, 'single_channel_rankings.csv')
    if not os.path.exists(csv_path):
        return None
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'seed_channel': int(row['channel']),
                'composite_score': float(row['composite_score']),
                'local_clean_mean': float(row['local_clean_mean']),
                'local_50_mean': float(row['local_50_mean']),
                'ret50_mean': float(row['ret50_mean']),
                'cn_clean_mean': float(row['cn_clean_mean']),
                'cn50_mean': float(row['cn50_mean']),
                'survival_rate': float(row['survival_rate']),
            })
    return results if results else None


def _reconstruct_phase_result(summary: Dict) -> Dict:
    """Reconstruct a phase result dict from a loaded summary (for Phase 3 seed)."""
    best_info = None
    for p in summary.get('all_paths_summary', []):
        if p['path_idx'] == summary.get('best_path_idx'):
            best_info = p
            break
    if best_info is None and summary.get('all_paths_summary'):
        best_info = summary['all_paths_summary'][0]

    if best_info is None:
        return {'best_path': None, 'all_paths': [], 'single_results': None}

    return {
        'best_path': {
            'path_idx': best_info['path_idx'],
            'seed': best_info['seed'],
            'final_channels': best_info['final_channels'],
            'final_score': best_info['final_score'],
            'selection_path': [],  # not needed for Phase 3 seed extraction
        },
        'all_paths': [],
        'single_results': None,
    }


# ============================================================
# Exhaustive Search (for small pools)
# ============================================================

def run_exhaustive_search(
    phase_name: str,
    alive_pool: List[int],
    dead_removed: List[int],
    candidate_pool: List[int],
    frame_indices: List[int],
    all_images: List[str],
    device: str,
    cfg: dict,
    output_dir: str,
) -> Dict:
    """
    Enumerate all 2^N - 1 non-empty subsets of alive_pool.
    Returns the same structure as run_forward_selection for compatibility.
    """
    phase_dir = os.path.join(output_dir, phase_name)
    os.makedirs(phase_dir, exist_ok=True)

    n = len(alive_pool)
    total_subsets = 2**n - 1

    print(f"\n{'='*70}")
    print(f"  EXHAUSTIVE SEARCH: {phase_name}")
    print(f"  Original pool:  {candidate_pool}")
    print(f"  Dead (removed): {dead_removed}")
    print(f"  Alive pool:     {alive_pool} ({n} channels)")
    print(f"  Total subsets:  {total_subsets}")
    print(f"  Frame indices:  {frame_indices}")
    print(f"  Alpha:          {cfg['alpha']}")
    print(f"{'='*70}")

    # Preload images
    preloaded = {}
    for fi in frame_indices:
        if fi < len(all_images):
            preloaded[fi] = load_image_numpy(all_images[fi])
    print(f"  Preloaded {len(preloaded)} frames into memory.")

    # Evaluate all subsets
    all_results = []
    for size in range(1, n + 1):
        for combo in combinations(alive_pool, size):
            channels = sorted(list(combo))
            result = evaluate_channel_subset(
                channels, frame_indices, all_images, device, cfg, preloaded
            )
            result['channel_str'] = ','.join(str(c) for c in channels)
            all_results.append(result)

            print(f"    [{result['channel_str']:>15s}] ({size}ch): "
                  f"Score={result['composite_score']:.5f}, "
                  f"Sharp_50={result['local_50_mean']:.4f}, "
                  f"Sharp_C={result['local_clean_mean']:.4f}, "
                  f"Ret50={result['ret50_mean']:.1f}%")

    # Sort by composite score
    all_results.sort(key=lambda r: -r['composite_score'])

    # Print ranking
    print(f"\n  {'─'*60}")
    print(f"  EXHAUSTIVE RANKING (top 10 / {total_subsets}):")
    print(f"  {'─'*60}")
    for rank, r in enumerate(all_results[:10], 1):
        print(f"    #{rank:2d}: [{r['channel_str']:>15s}] ({r['n_channels']}ch) "
              f"Score={r['composite_score']:.5f}, "
              f"Ret50={r['ret50_mean']:.1f}%, "
              f"CN={r['cn50_mean']:.2f}")

    # Also print best per channel count
    print(f"\n  Best per channel count:")
    for size in range(1, n + 1):
        size_results = [r for r in all_results if r['n_channels'] == size]
        if size_results:
            best = size_results[0]  # already sorted
            print(f"    {size}ch: [{best['channel_str']:>15s}] "
                  f"Score={best['composite_score']:.5f}")

    # Save all results as CSV
    csv_path = os.path.join(phase_dir, 'exhaustive_all_subsets.csv')
    fieldnames = ['rank', 'n_channels', 'channel_str', 'composite_score',
                  'local_clean_mean', 'local_50_mean',
                  'ret50_mean', 'ret50_std', 'ret50_min', 'ret30_mean',
                  'cn_clean_mean', 'cn50_mean', 'n_survive_50', 'survival_rate']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, r in enumerate(all_results, 1):
            row = {k: r.get(k, '') for k in fieldnames}
            row['rank'] = rank
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    # Build compatible result structure
    best = all_results[0]
    best_channels = best['channels']

    # Build a "path" from 1ch to best for visualization compatibility
    # (just the global optimum as a single-step path)
    selection_path = [{
        'step': 0,
        'added_channel': None,
        'channel_str': best['channel_str'],
        **best,
    }]

    best_path = {
        'path_idx': 1,
        'seed': best_channels[:1],  # first channel as nominal seed
        'final_channels': best_channels,
        'final_score': best['composite_score'],
        'n_steps': 1,
        'selection_path': selection_path,
    }

    # Also save single-channel rankings for resume compatibility
    single_results = [r for r in all_results if r['n_channels'] == 1]
    single_results.sort(key=lambda r: -r['composite_score'])
    for r in single_results:
        r['seed_channel'] = r['channels'][0]

    single_csv = os.path.join(phase_dir, 'single_channel_rankings.csv')
    fieldnames_single = ['rank', 'channel', 'composite_score', 'local_clean_mean',
                        'local_50_mean', 'ret50_mean', 'cn_clean_mean', 'cn50_mean',
                        'survival_rate']
    with open(single_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_single)
        writer.writeheader()
        for rank, r in enumerate(single_results, 1):
            writer.writerow({
                'rank': rank,
                'channel': r['seed_channel'],
                'composite_score': f"{r['composite_score']:.6f}",
                'local_clean_mean': f"{r['local_clean_mean']:.6f}",
                'local_50_mean': f"{r['local_50_mean']:.6f}",
                'ret50_mean': f"{r['ret50_mean']:.1f}",
                'cn_clean_mean': f"{r['cn_clean_mean']:.2f}",
                'cn50_mean': f"{r['cn50_mean']:.2f}",
                'survival_rate': f"{r['survival_rate']:.1f}",
            })
    print(f"  Saved: {single_csv}")

    # Save summary
    summary = {
        'phase_name': phase_name,
        'search_mode': 'exhaustive',
        'original_pool': candidate_pool,
        'dead_removed': dead_removed,
        'alive_pool': alive_pool,
        'total_subsets_evaluated': total_subsets,
        'alpha': cfg['alpha'],
        'frame_indices': frame_indices,
        'best_path_idx': 1,
        'best_final_channels': best_channels,
        'best_final_score': best['composite_score'],
        'all_paths_summary': [{
            'path_idx': 1,
            'seed': best_channels[:1],
            'final_channels': best_channels,
            'final_score': best['composite_score'],
            'n_steps': 1,
        }],
        'top_10': [
            {
                'rank': i + 1,
                'channels': r['channels'],
                'n_channels': r['n_channels'],
                'composite_score': r['composite_score'],
                'ret50_mean': r['ret50_mean'],
                'cn50_mean': r['cn50_mean'],
            }
            for i, r in enumerate(all_results[:10])
        ],
    }

    summary_path = os.path.join(phase_dir, 'selection_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    print(f"\n  {'='*60}")
    print(f"  BEST for {phase_name} (EXHAUSTIVE):")
    print(f"    Channels: [{best['channel_str']}] ({best['n_channels']}ch)")
    print(f"    Score:    {best['composite_score']:.5f}")
    print(f"  {'='*60}")

    return {
        'best_path': best_path,
        'all_paths': [best_path],
        'single_results': single_results,
    }


# ============================================================
# Forward Greedy Selection with Look-ahead (deduped)
# ============================================================

def run_forward_selection(
    phase_name: str,
    candidate_pool: List[int],
    frame_indices: List[int],
    all_images: List[str],
    device: str,
    cfg: dict,
    output_dir: str,
    max_paths: int = 3,
    seed_channels: Optional[List[int]] = None,
    cached_single_results: Optional[List[Dict]] = None,
) -> Dict:
    """
    Forward greedy selection with:
      - Dead channel filtering
      - Look-ahead with deduplication
      - Path pruning (skip weak seeds)
      - Optional cached single-channel results (from resume)
    """
    phase_dir = os.path.join(output_dir, phase_name)
    os.makedirs(phase_dir, exist_ok=True)

    # ── Filter dead channels ──
    alive_pool, dead_removed = filter_dead_channels(candidate_pool)

    print(f"\n{'='*70}")
    print(f"  FORWARD GREEDY SELECTION: {phase_name}")
    print(f"  Original pool:  {candidate_pool}")
    print(f"  Dead (removed): {dead_removed}")
    print(f"  Alive pool:     {alive_pool}")
    if seed_channels:
        print(f"  Seed channels:  {seed_channels}")
    print(f"  Frame indices:  {frame_indices}")
    print(f"  Alpha:          {cfg['alpha']}")
    print(f"  Look-ahead:     enabled (1-step, deduped)")
    print(f"  Path pruning:   seed Score < {cfg['seed_prune_ratio']*100:.0f}% of best -> skip")
    print(f"{'='*70}")

    if not alive_pool:
        print(f"  [WARNING] No alive channels in pool! Skipping {phase_name}.")
        return {'best_path': None, 'all_paths': [], 'single_results': None}

    # Preload images
    preloaded = {}
    for fi in frame_indices:
        if fi < len(all_images):
            preloaded[fi] = load_image_numpy(all_images[fi])
    print(f"  Preloaded {len(preloaded)} frames into memory.")

    # ── Step 1: Evaluate all alive single channels (or use cached/seed) ──
    if seed_channels is None:
        if cached_single_results is not None:
            # Resume: reuse cached single-channel rankings
            single_results = cached_single_results
            print(f"\n  Step 1: Loaded {len(single_results)} cached single-channel rankings.")
            for rank, r in enumerate(single_results, 1):
                print(f"    #{rank}: Ch{r['seed_channel']:2d} "
                      f"Score={r['composite_score']:.5f} [CACHED]")
        else:
            print(f"\n  Step 1: Evaluating all {len(alive_pool)} alive single channels...")
            single_results = []

            for ch in alive_pool:
                result = evaluate_channel_subset(
                    [ch], frame_indices, all_images, device, cfg, preloaded
                )
                result['seed_channel'] = ch
                single_results.append(result)

                print(f"    Ch{ch:2d}: Score={result['composite_score']:.5f} "
                      f"(Sharp_50={result['local_50_mean']:.4f}, "
                      f"Sharp_C={result['local_clean_mean']:.4f}, "
                      f"Ret50={result['ret50_mean']:.1f}%)")

            # Sort by composite score (descending)
            single_results.sort(key=lambda r: -r['composite_score'])

        print(f"\n  Single-channel ranking by Composite Score:")
        best_single_score = single_results[0]['composite_score'] if single_results else 0
        prune_threshold = best_single_score * cfg['seed_prune_ratio']

        viable_seeds = []
        for rank, r in enumerate(single_results, 1):
            is_viable = r['composite_score'] >= prune_threshold
            if is_viable and len(viable_seeds) < max_paths:
                viable_seeds.append(r)
                marker = " <-- SEED"
            elif not is_viable:
                marker = f" [PRUNED: < {cfg['seed_prune_ratio']*100:.0f}% of best]"
            else:
                marker = ""
            print(f"    #{rank}: Ch{r['seed_channel']:2d} "
                  f"Score={r['composite_score']:.5f}{marker}")

        # Save single-channel rankings
        single_csv = os.path.join(phase_dir, 'single_channel_rankings.csv')
        fieldnames_single = ['rank', 'channel', 'composite_score', 'local_clean_mean',
                            'local_50_mean', 'ret50_mean', 'cn_clean_mean', 'cn50_mean',
                            'survival_rate']
        with open(single_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_single)
            writer.writeheader()
            for rank, r in enumerate(single_results, 1):
                writer.writerow({
                    'rank': rank,
                    'channel': r['seed_channel'],
                    'composite_score': f"{r['composite_score']:.6f}",
                    'local_clean_mean': f"{r['local_clean_mean']:.6f}",
                    'local_50_mean': f"{r['local_50_mean']:.6f}",
                    'ret50_mean': f"{r['ret50_mean']:.1f}",
                    'cn_clean_mean': f"{r['cn_clean_mean']:.2f}",
                    'cn50_mean': f"{r['cn50_mean']:.2f}",
                    'survival_rate': f"{r['survival_rate']:.1f}",
                })
        print(f"  Saved: {single_csv}")

        # Determine seed channels for each path (only viable ones)
        path_seeds = [[r['seed_channel']] for r in viable_seeds]
        print(f"\n  Viable seeds after pruning: {len(path_seeds)} paths")
    else:
        # Phase 3: use provided seed, filter dead from seed too
        seed_channels = [ch for ch in seed_channels if ch not in ALL_DEAD_CHANNELS]
        single_results = None
        path_seeds = [seed_channels]
        max_paths = 1

    # ── Step 2: Greedy forward selection with deduped look-ahead ──
    all_paths = []

    for path_idx, seed in enumerate(path_seeds):
        print(f"\n  {'─'*60}")
        print(f"  Path {path_idx + 1}/{len(path_seeds)}: Seed = {seed}")
        print(f"  {'─'*60}")

        current_channels = list(seed)
        remaining_pool = [ch for ch in alive_pool if ch not in current_channels]
        selection_path = []

        # Evaluate seed (for multi-channel seeds like Phase 3)
        if len(seed) == 1 and single_results is not None:
            # Reuse single-channel result
            seed_result = None
            for r in single_results:
                if r.get('seed_channel') == seed[0]:
                    seed_result = dict(r)
                    break
            if seed_result is None:
                seed_result = evaluate_channel_subset(
                    current_channels, frame_indices, all_images, device, cfg, preloaded
                )
        else:
            seed_result = evaluate_channel_subset(
                current_channels, frame_indices, all_images, device, cfg, preloaded
            )

        seed_result['step'] = 0
        seed_result['added_channel'] = None
        seed_result['channel_str'] = ','.join(str(c) for c in current_channels)
        selection_path.append(seed_result)

        current_score = seed_result['composite_score']
        print(f"    Seed [{','.join(str(c) for c in current_channels)}]: "
              f"Score={current_score:.5f}")

        # Greedy loop with deduped look-ahead
        step = 0
        while remaining_pool:
            step += 1
            n_remaining = len(remaining_pool)
            print(f"\n    Step {step}: Trying to add 1 of {n_remaining} candidates "
                  f"to [{','.join(str(c) for c in current_channels)}]")

            # ── Evaluate all single-step additions ──
            candidates = []
            for ch_to_add in remaining_pool:
                trial_channels = sorted(current_channels + [ch_to_add])
                result = evaluate_channel_subset(
                    trial_channels, frame_indices, all_images, device, cfg, preloaded
                )
                improvement = result['composite_score'] - current_score
                imp_pct = (improvement / current_score * 100) if current_score > 1e-10 else 0.0

                print(f"      +Ch{ch_to_add:2d}: Score={result['composite_score']:.5f} "
                      f"(delta={improvement:+.5f}, {imp_pct:+.1f}%)")

                candidates.append({
                    'channel': ch_to_add,
                    'result': result,
                    'score': result['composite_score'],
                    'improvement': improvement,
                })

            # Sort candidates by score (descending)
            candidates.sort(key=lambda c: -c['score'])
            best = candidates[0]

            # ── Case 1: Best candidate improves score -> accept ──
            if best['improvement'] > 0:
                current_channels = sorted(current_channels + [best['channel']])
                remaining_pool = [ch for ch in remaining_pool if ch != best['channel']]
                current_score = best['score']

                best['result']['step'] = step
                best['result']['added_channel'] = best['channel']
                best['result']['channel_str'] = ','.join(str(c) for c in current_channels)
                selection_path.append(best['result'])

                imp_pct = (best['improvement'] / (current_score - best['improvement']) * 100) \
                    if (current_score - best['improvement']) > 1e-10 else 0.0
                print(f"\n    --> Added Ch{best['channel']} (delta={best['improvement']:+.5f})")
                print(f"        New set: [{','.join(str(c) for c in current_channels)}]")
                print(f"        Score={current_score:.5f}, "
                      f"Ret50={best['result']['ret50_mean']:.1f}%, "
                      f"CN={best['result']['cn50_mean']:.2f}, "
                      f"Survival={best['result']['survival_rate']:.0f}%")

                # Save checkpoint
                _save_path_csv(selection_path, phase_dir, f'path_{path_idx+1}')
                continue

            # ── Case 2: No single-step improvement -> TOP-3 DEDUPED LOOK-AHEAD ──
            # Only look ahead from the top-3 scoring candidates (not all).
            # For each top-3 candidate as forced first step, try pairing with
            # every remaining channel. Pairs are deduped via sorted tuple set.
            la_top_k = min(3, len(candidates))
            la_candidates = [c['channel'] for c in candidates[:la_top_k]]
            print(f"\n    No single-step improvement. Entering look-ahead (top-{la_top_k}, deduped)...")
            print(f"    Top-{la_top_k} candidates for LA: {la_candidates}")
            print(f"    (Checking if adding 2 channels can beat current Score={current_score:.5f})")

            look_ahead_found = False
            best_la_score = current_score  # must beat this
            best_la_ch1 = None
            best_la_ch2 = None
            best_la_result = None

            # Generate unique pairs: each top-3 candidate paired with every
            # remaining channel, deduplicated via sorted tuple.
            seen_pairs = set()
            la_pairs = []
            for ch1 in la_candidates:
                for ch2 in remaining_pool:
                    if ch2 == ch1:
                        continue
                    pair_key = tuple(sorted([ch1, ch2]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        la_pairs.append(pair_key)

            n_pairs = len(la_pairs)
            print(f"    Evaluating {n_pairs} unique pairs...")

            for pair_idx, (ch1, ch2) in enumerate(la_pairs):
                trial_2step = sorted(current_channels + [ch1, ch2])
                result_2step = evaluate_channel_subset(
                    trial_2step, frame_indices, all_images, device, cfg, preloaded
                )

                la_improvement = result_2step['composite_score'] - current_score
                la_pct = (la_improvement / current_score * 100) if current_score > 1e-10 else 0.0

                print(f"      LA [{pair_idx+1}/{n_pairs}]: "
                      f"+Ch{ch1:2d}+Ch{ch2:2d}: "
                      f"Score={result_2step['composite_score']:.5f} "
                      f"(delta={la_improvement:+.5f}, {la_pct:+.1f}%)")

                if result_2step['composite_score'] > best_la_score:
                    best_la_score = result_2step['composite_score']
                    best_la_ch1 = ch1
                    best_la_ch2 = ch2
                    best_la_result = result_2step
                    look_ahead_found = True

            if look_ahead_found:
                # Accept both channels in one jump
                current_channels = sorted(current_channels + [best_la_ch1, best_la_ch2])
                remaining_pool = [ch for ch in remaining_pool
                                  if ch != best_la_ch1 and ch != best_la_ch2]
                current_score = best_la_score

                best_la_result['step'] = step
                best_la_result['added_channel'] = f"{best_la_ch1}+{best_la_ch2}"
                best_la_result['channel_str'] = ','.join(str(c) for c in current_channels)
                selection_path.append(best_la_result)

                print(f"\n    --> LOOK-AHEAD SUCCESS: Added Ch{best_la_ch1} + Ch{best_la_ch2}")
                print(f"        New set: [{','.join(str(c) for c in current_channels)}]")
                print(f"        Score={current_score:.5f}, "
                      f"Ret50={best_la_result['ret50_mean']:.1f}%, "
                      f"CN={best_la_result['cn50_mean']:.2f}, "
                      f"Survival={best_la_result['survival_rate']:.0f}%")

                # Save checkpoint
                _save_path_csv(selection_path, phase_dir, f'path_{path_idx+1}')
            else:
                # Truly no improvement even with look-ahead
                print(f"\n    STOP: Look-ahead found no improvement either.")
                print(f"    Optimal subset for this path: "
                      f"[{','.join(str(c) for c in current_channels)}] "
                      f"(Score={current_score:.5f})")
                break

        # Record this path
        all_paths.append({
            'path_idx': path_idx + 1,
            'seed': seed,
            'final_channels': current_channels,
            'final_score': current_score,
            'n_steps': len(selection_path),
            'selection_path': selection_path,
        })

        # Save final path
        _save_path_csv(selection_path, phase_dir, f'path_{path_idx+1}')

    # ── Determine best path ──
    if not all_paths:
        print(f"  [WARNING] No paths completed for {phase_name}.")
        return {'best_path': None, 'all_paths': [], 'single_results': single_results}

    best_path = max(all_paths, key=lambda p: p['final_score'])

    print(f"\n  {'='*60}")
    print(f"  BEST PATH for {phase_name}:")
    print(f"    Seed: {best_path['seed']}")
    print(f"    Final: [{','.join(str(c) for c in best_path['final_channels'])}]")
    print(f"    Score: {best_path['final_score']:.5f}")
    print(f"  {'='*60}")

    # Save summary
    summary = {
        'phase_name': phase_name,
        'search_mode': 'greedy_with_lookahead',
        'original_pool': candidate_pool,
        'dead_removed': dead_removed,
        'alive_pool': alive_pool,
        'alpha': cfg['alpha'],
        'frame_indices': frame_indices,
        'best_path_idx': best_path['path_idx'],
        'best_final_channels': best_path['final_channels'],
        'best_final_score': best_path['final_score'],
        'all_paths_summary': [
            {
                'path_idx': p['path_idx'],
                'seed': p['seed'],
                'final_channels': p['final_channels'],
                'final_score': p['final_score'],
                'n_steps': p['n_steps'],
            }
            for p in all_paths
        ],
    }

    summary_path = os.path.join(phase_dir, 'selection_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    return {
        'best_path': best_path,
        'all_paths': all_paths,
        'single_results': single_results,
    }


def _save_path_csv(selection_path: List[Dict], phase_dir: str, path_name: str):
    """Save a selection path as CSV."""
    csv_path = os.path.join(phase_dir, f'{path_name}_selection.csv')
    fieldnames = ['step', 'n_channels', 'channel_str', 'added_channel',
                  'composite_score', 'local_clean_mean', 'local_50_mean',
                  'ret50_mean', 'ret50_std', 'ret50_min', 'ret30_mean',
                  'cn_clean_mean', 'cn50_mean', 'n_survive_50', 'survival_rate']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in selection_path:
            row = {k: rec.get(k, '') for k in fieldnames}
            writer.writerow(row)


# ============================================================
# Visualization
# ============================================================

def plot_forward_selection_curves(all_phase_results: Dict, output_dir: str):
    """Plot score progression for all phases and paths."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ── Left: Composite Score vs N channels ──
    ax = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 1, 10))
    color_idx = 0

    for phase_name, result in all_phase_results.items():
        if result['best_path'] is None:
            continue
        for path_info in result['all_paths']:
            path = path_info['selection_path']
            if not path:
                continue
            n_chs = [p['n_channels'] for p in path]
            scores = [p['composite_score'] for p in path]
            label = f"{phase_name} P{path_info['path_idx']} (seed={path_info['seed']})"
            is_best = (path_info['path_idx'] == result['best_path']['path_idx'])
            lw = 2.5 if is_best else 1.2
            al = 1.0 if is_best else 0.5
            ax.plot(n_chs, scores, 'o-', color=colors[color_idx % 10],
                   label=label, linewidth=lw, markersize=5, alpha=al)

            for p in path:
                if p.get('added_channel') is not None:
                    ax.annotate(f"+Ch{p['added_channel']}",
                               (p['n_channels'], p['composite_score']),
                               textcoords="offset points", xytext=(5, 5),
                               fontsize=6, color=colors[color_idx % 10], alpha=0.7)
            color_idx += 1

    ax.set_xlabel('Number of Channels')
    ax.set_ylabel('Composite Score')
    ax.set_title('Forward Selection: Score vs Channel Count', fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)

    # ── Middle: Absolute Sharpness (+50%) ──
    ax = axes[1]
    color_idx = 0
    for phase_name, result in all_phase_results.items():
        if result['best_path'] is None:
            continue
        best_path = result['best_path']
        path = best_path['selection_path']
        if not path:
            continue
        n_chs = [p['n_channels'] for p in path]
        sharp_50 = [p['local_50_mean'] for p in path]
        sharp_clean = [p['local_clean_mean'] for p in path]
        ax.plot(n_chs, sharp_50, 'o-', color=colors[color_idx % 10],
               label=f"{phase_name} Sharp_50", linewidth=2, markersize=5)
        ax.plot(n_chs, sharp_clean, 's--', color=colors[color_idx % 10],
               label=f"{phase_name} Sharp_Clean", linewidth=1.5, markersize=4, alpha=0.6)
        color_idx += 1

    ax.set_xlabel('Number of Channels')
    ax.set_ylabel('Absolute Local Sharpness')
    ax.set_title('Best Paths: Sharpness Progression', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Right: Retention & Survival Rate ──
    ax = axes[2]
    ax2 = ax.twinx()
    color_idx = 0
    for phase_name, result in all_phase_results.items():
        if result['best_path'] is None:
            continue
        best_path = result['best_path']
        path = best_path['selection_path']
        if not path:
            continue
        n_chs = [p['n_channels'] for p in path]
        ret50 = [p['ret50_mean'] for p in path]
        survival = [p['survival_rate'] for p in path]
        ax.plot(n_chs, ret50, 'o-', color=colors[color_idx % 10],
               label=f"{phase_name} Ret50", linewidth=2, markersize=5)
        ax2.plot(n_chs, survival, 's:', color=colors[color_idx % 10],
                label=f"{phase_name} Survival%", linewidth=1.5, markersize=4, alpha=0.6)
        color_idx += 1

    ax.axhline(100, color='green', linestyle=':', alpha=0.5)
    ax.set_xlabel('Number of Channels')
    ax.set_ylabel('Retention_50 (%)')
    ax2.set_ylabel('Survival Rate (%)')
    ax.set_title('Best Paths: Retention & Survival', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax2.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path_out = os.path.join(output_dir, 'forward_selection_curves.png')
    plt.savefig(path_out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path_out}")


def plot_comparison_table(all_phase_results: Dict, output_dir: str):
    """Generate a comparison summary table as CSV."""
    rows = []
    for phase_name, result in all_phase_results.items():
        if result['best_path'] is None:
            continue
        best = result['best_path']
        last_step = best['selection_path'][-1] if best['selection_path'] else None
        if last_step is None:
            continue
        rows.append({
            'Phase': phase_name,
            'Optimal Channels': str(best['final_channels']),
            'N': len(best['final_channels']),
            'Score': f"{best['final_score']:.5f}",
            'Sharp_50': f"{last_step['local_50_mean']:.4f}",
            'Sharp_Clean': f"{last_step['local_clean_mean']:.4f}",
            'Ret50': f"{last_step['ret50_mean']:.1f}%",
            'CN_50': f"{last_step['cn50_mean']:.2f}",
            'Survival': f"{last_step['survival_rate']:.0f}%",
        })

    csv_path = os.path.join(output_dir, 'optimal_combinations_comparison.csv')
    if rows:
        fieldnames = rows[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"  Saved: {csv_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Forward Greedy Selection v3 — Channel Combination Discovery')
    parser.add_argument('--frame_indices', type=int, nargs='+', default=None,
                        help='Manually specify frame indices (default: 41 306 512)')
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='Weight for +50%% Sharpness in composite score (default: 0.7)')
    parser.add_argument('--max_paths', type=int, default=3,
                        help='Number of seed paths to try per phase (default: 3)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume: skip completed phases, reuse single-channel rankings')
    parser.add_argument('--skip_phase1', action='store_true',
                        help='Skip Phase 1 (Rank01 internal)')
    parser.add_argument('--skip_phase2', action='store_true',
                        help='Skip Phase 2 (Rank02 internal)')
    parser.add_argument('--skip_phase3', action='store_true',
                        help='Skip Phase 3 (Cross-combination)')
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg['alpha'] = args.alpha
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    device = cfg['device']

    # Load image list
    all_images = sorted(glob.glob(os.path.join(cfg['rgb_dir'], "*.png")))
    n_images = len(all_images)

    if n_images == 0:
        print(f"[ERROR] No images found in {cfg['rgb_dir']}")
        return

    # Frame indices
    if args.frame_indices is not None:
        frame_indices = [fi for fi in args.frame_indices if fi < n_images]
    else:
        frame_indices = [41, 306, 512]

    # ── Filter dead channels from each combination ──
    rank01_alive, rank01_dead = filter_dead_channels(RANK01_CHANNELS)
    rank02_alive, rank02_dead = filter_dead_channels(RANK02_CHANNELS)

    print(f"\n{'='*70}")
    print(f"  FORWARD GREEDY SELECTION v3 — Channel Combination Discovery")
    print(f"{'='*70}")
    print(f"  Date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total images:     {n_images}")
    print(f"  Frame indices:    {frame_indices}")
    print(f"  Alpha:            {args.alpha}")
    print(f"  Max paths/phase:  {args.max_paths}")
    print(f"  Resume mode:      {'ON' if args.resume else 'OFF'}")
    print(f"  Exhaustive if:    alive <= {cfg['exhaustive_threshold']} channels")
    print(f"  Seed pruning:     < {cfg['seed_prune_ratio']*100:.0f}% of best")
    print(f"  Look-ahead:       deduped (combinations, not permutations)")
    print(f"  Output:           {output_dir}")
    print(f"  Objective:        Score = {args.alpha}*Sharp_50 + {1-args.alpha:.1f}*Sharp_Clean")
    print(f"{'='*70}")
    print(f"\n  Dead channels (excluded from all searches):")
    print(f"    Always dead (BN gamma~0): {ALWAYS_DEAD_CHANNELS}")
    print(f"    Dead at +50% brightness:  {DEAD_AT_50_CHANNELS}")
    print(f"    Combined exclusion list:   {ALL_DEAD_CHANNELS}")
    print(f"\n  Rank01: {RANK01_CHANNELS}")
    print(f"    Alive: {rank01_alive}  ({len(rank01_alive)} channels)")
    print(f"    Dead:  {rank01_dead}")
    print(f"    Mode:  {'EXHAUSTIVE' if len(rank01_alive) <= cfg['exhaustive_threshold'] else 'GREEDY'}")
    print(f"  Rank02: {RANK02_CHANNELS}")
    print(f"    Alive: {rank02_alive}  ({len(rank02_alive)} channels)")
    print(f"    Dead:  {rank02_dead}")
    print(f"    Mode:  {'EXHAUSTIVE' if len(rank02_alive) <= cfg['exhaustive_threshold'] else 'GREEDY'}")

    all_phase_results = {}

    # ── Phase 1: Rank01 Internal ──
    if not args.skip_phase1:
        phase1_dir = os.path.join(output_dir, 'Phase1_Rank01')

        # Check resume
        if args.resume:
            existing_summary = _load_phase_summary(phase1_dir)
            if existing_summary:
                print(f"\n  [RESUME] Phase 1 already completed. Loading from summary.")
                print(f"    Best: {existing_summary['best_final_channels']} "
                      f"Score={existing_summary['best_final_score']:.5f}")
                all_phase_results['Phase1_Rank01'] = _reconstruct_phase_result(existing_summary)
            else:
                existing_summary = None

        if 'Phase1_Rank01' not in all_phase_results:
            if len(rank01_alive) <= cfg['exhaustive_threshold']:
                # Exhaustive search
                result_p1 = run_exhaustive_search(
                    phase_name='Phase1_Rank01',
                    alive_pool=rank01_alive,
                    dead_removed=rank01_dead,
                    candidate_pool=RANK01_CHANNELS,
                    frame_indices=frame_indices,
                    all_images=all_images,
                    device=device,
                    cfg=cfg,
                    output_dir=output_dir,
                )
            else:
                # Greedy with look-ahead
                cached_singles = None
                if args.resume:
                    cached_singles = _load_single_rankings_csv(phase1_dir)
                    if cached_singles:
                        print(f"  [RESUME] Loaded {len(cached_singles)} single-channel rankings for Phase 1.")

                result_p1 = run_forward_selection(
                    phase_name='Phase1_Rank01',
                    candidate_pool=RANK01_CHANNELS,
                    frame_indices=frame_indices,
                    all_images=all_images,
                    device=device,
                    cfg=cfg,
                    output_dir=output_dir,
                    max_paths=args.max_paths,
                    cached_single_results=cached_singles,
                )
            all_phase_results['Phase1_Rank01'] = result_p1

    # ── Phase 2: Rank02 Internal ──
    if not args.skip_phase2:
        phase2_dir = os.path.join(output_dir, 'Phase2_Rank02')

        # Check resume
        if args.resume:
            existing_summary = _load_phase_summary(phase2_dir)
            if existing_summary:
                print(f"\n  [RESUME] Phase 2 already completed. Loading from summary.")
                print(f"    Best: {existing_summary['best_final_channels']} "
                      f"Score={existing_summary['best_final_score']:.5f}")
                all_phase_results['Phase2_Rank02'] = _reconstruct_phase_result(existing_summary)
            else:
                existing_summary = None

        if 'Phase2_Rank02' not in all_phase_results:
            if len(rank02_alive) <= cfg['exhaustive_threshold']:
                result_p2 = run_exhaustive_search(
                    phase_name='Phase2_Rank02',
                    alive_pool=rank02_alive,
                    dead_removed=rank02_dead,
                    candidate_pool=RANK02_CHANNELS,
                    frame_indices=frame_indices,
                    all_images=all_images,
                    device=device,
                    cfg=cfg,
                    output_dir=output_dir,
                )
            else:
                cached_singles = None
                if args.resume:
                    cached_singles = _load_single_rankings_csv(phase2_dir)
                    if cached_singles:
                        print(f"  [RESUME] Loaded {len(cached_singles)} single-channel rankings for Phase 2.")

                result_p2 = run_forward_selection(
                    phase_name='Phase2_Rank02',
                    candidate_pool=RANK02_CHANNELS,
                    frame_indices=frame_indices,
                    all_images=all_images,
                    device=device,
                    cfg=cfg,
                    output_dir=output_dir,
                    max_paths=args.max_paths,
                    cached_single_results=cached_singles,
                )
            all_phase_results['Phase2_Rank02'] = result_p2

    # ── Phase 3: Cross-Combination Forward Selection ──
    if not args.skip_phase3:
        p1_best = all_phase_results.get('Phase1_Rank01', {}).get('best_path')
        p2_best = all_phase_results.get('Phase2_Rank02', {}).get('best_path')

        if p1_best and p2_best:
            p1_seed = p1_best['seed']
            p2_seed = p2_best['seed']
            cross_seed = sorted(list(set(p1_seed + p2_seed)))

            # Candidate pool: union of all channels from both combinations
            cross_pool_all = sorted(list(set(RANK01_CHANNELS + RANK02_CHANNELS)))

            # Determine search mode for cross-combination
            cross_alive, cross_dead = filter_dead_channels(cross_pool_all)

            print(f"\n  Phase 3 setup:")
            print(f"    Seed from Phase 1: {p1_seed}")
            print(f"    Seed from Phase 2: {p2_seed}")
            print(f"    Cross-seed: {cross_seed}")
            print(f"    Cross alive pool: {cross_alive} ({len(cross_alive)} channels)")
            print(f"    Mode: GREEDY (pool > {cfg['exhaustive_threshold']})")

            result_p3 = run_forward_selection(
                phase_name='Phase3_Cross',
                candidate_pool=cross_pool_all,
                frame_indices=frame_indices,
                all_images=all_images,
                device=device,
                cfg=cfg,
                output_dir=output_dir,
                max_paths=1,
                seed_channels=cross_seed,
            )
            all_phase_results['Phase3_Cross'] = result_p3
        else:
            print(f"\n  [WARNING] Cannot run Phase 3: Phase 1 or Phase 2 missing.")

    # ── Generate visualizations ──
    if all_phase_results:
        print(f"\n  Generating comparison visualizations...")
        plot_forward_selection_curves(all_phase_results, output_dir)
        plot_comparison_table(all_phase_results, output_dir)

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"  FORWARD GREEDY SELECTION v3 — COMPLETE")
    print(f"{'='*70}")

    for phase_name, result in all_phase_results.items():
        if result['best_path'] is None:
            print(f"\n  {phase_name}: No valid path found.")
            continue
        best = result['best_path']
        last_step = best['selection_path'][-1] if best['selection_path'] else None
        print(f"\n  {phase_name}:")
        print(f"    Optimal: [{','.join(str(c) for c in best['final_channels'])}] "
              f"({len(best['final_channels'])} channels)")
        print(f"    Score:   {best['final_score']:.5f}")
        if last_step:
            print(f"    Sharp_50={last_step['local_50_mean']:.4f}, "
                  f"Sharp_C={last_step['local_clean_mean']:.4f}, "
                  f"Ret50={last_step['ret50_mean']:.1f}%, "
                  f"CN={last_step['cn50_mean']:.2f}, "
                  f"Survival={last_step['survival_rate']:.0f}%")

    print(f"\n  Output: {output_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()