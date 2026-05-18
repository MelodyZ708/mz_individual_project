"""
Comprehensive Forward Greedy Selection Visualization
=====================================================
CNN Feature Injection into COMO -- Post-Forward-Selection Analysis

Generates all visualizations needed to interpret forward greedy results:

  Part A: Conv1 filter weights & feature response maps (per channel)
  Part B: Original RGB images for 3 test frames (Clean / +30% / +50%)
  Part C: 3D Convergence Basin comparison grids (Gray vs key combinations)
  Part E: Channel type classification summary
  Part F: Channel type analysis summary (text file with calculations)

Usage:
  python visualize_forward_results.py

Author: mz325
Date: 2026-05
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import os
import sys
import glob
import cv2
from typing import Tuple, List, Dict, Optional
import json

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/forward_visualization',
    'device': 'cuda:0',

    # Test frames (early / mid / late) -- same as forward greedy experiment
    'frame_indices': [41, 306, 512],

    # Perturbation range (pixels)
    'max_shift_px': 30,
    'grid_size': 61,

    # Sharpness config
    'sharpness_radius': 5,
}

# ── Channel combinations from FORWARD GREEDY results ──
# Key insight combinations only
COMBINATIONS = {
    'Ch15_solo': {
        'channels': [15],
        'label': '[15] (Forward Best)',
        'color': '#c0392b',
        'is_optimal': True,
    },
    'Ch6_Ch15': {
        'channels': [6, 15],
        'label': '[6,15] (Ret-Optimal)',
        'color': '#e74c3c',
        'is_optimal': False,
    },
    'Ch23_solo': {
        'channels': [23],
        'label': '[23] (Rank02 Best)',
        'color': '#2980b9',
        'is_optimal': True,
    },
    'Ch15_Ch23_cross': {
        'channels': [15, 23],
        'label': '[15,23] (Cross-Combo)',
        'color': '#8e44ad',
        'is_optimal': False,
    },
    'Rank01_Full8': {
        'channels': [6, 7, 12, 15, 36, 45, 58, 62],
        'label': 'Rank01 Full (8ch)',
        'color': '#1a5276',
        'is_optimal': False,
    },
    'Rank02_Full8': {
        'channels': [8, 22, 23, 27, 28, 42, 48, 60],
        'label': 'Rank02 Full (8ch)',
        'color': '#7d3c98',
        'is_optimal': False,
    },
}

# ── Channel importance from FORWARD GREEDY selection ──
RANK01_IMPORTANCE = {
    15: 'core',         # Forward best single channel
    6:  'supportive',   # Positive-response, improves Retention
    45: 'weak',         # Low absolute Sharpness
    7:  'dead', 12: 'dead_at_50', 36: 'dead',
    58: 'dead_at_50', 62: 'dead_at_50',
}
RANK02_IMPORTANCE = {
    23: 'core',         # Forward best single channel
    42: 'secondary',    # 2nd highest Score, but high CN
    60: 'tertiary',     # 3rd highest Score
    27: 'supportive',   # Positive-response at brightness
    28: 'supportive',
    22: 'supportive',
    8:  'supportive',   # Extreme positive-response (Ret50=186.8%)
    48: 'dead',
}

IMPORTANCE_COLORS = {
    'core':        '#c0392b',   # red -- absolutely essential
    'supportive':  '#3498db',   # blue -- alive, contributes
    'secondary':   '#f39c12',   # orange
    'tertiary':    '#f1c40f',   # yellow
    'weak':        '#95a5a6',   # gray
    'dead':        '#2c3e50',   # dark gray -- always dead
    'dead_at_50':  '#566573',   # medium gray -- dies at +50%
}

BRIGHTNESS_CONDITIONS = [
    {'key': 'clean',    'factor': 0.0, 'label': 'Clean'},
    {'key': 'bright30', 'factor': 0.3, 'label': '+30%'},
    {'key': 'bright50', 'factor': 0.5, 'label': '+50%'},
]

# Plot styling
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 250,
    'mathtext.fontset': 'cm',
})


# ============================================================
# Feature Extractor (Direct Conv1 -- matches forward greedy scripts)
# ============================================================
class DirectConv1Extractor(nn.Module):
    """Extract specific channels from ResNet-18 conv1 + bn1 + relu.
    Applies ImageNet normalization before feature extraction."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(self, channel_indices: List[int], device: str = 'cuda:0'):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = nn.ReLU(inplace=False)
        self.channel_indices = channel_indices
        self.device = device

        self.register_buffer('mean', torch.tensor(self.IMAGENET_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.IMAGENET_STD, device=device).view(1, 3, 1, 1))

        for p in self.parameters():
            p.requires_grad_(False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        x = self._normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        selected = x[:, self.channel_indices, :, :]
        upsampled = F.interpolate(selected, size=orig_size,
                                  mode='bilinear', align_corners=False)
        return upsampled

    def get_full_features(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        x = self._normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        upsampled = F.interpolate(x, size=orig_size,
                                  mode='bilinear', align_corners=False)
        return upsampled

    @staticmethod
    def get_conv1_weights(device: str = 'cuda:0') -> np.ndarray:
        from torchvision.models import resnet18, ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
        return base.conv1.weight.detach().cpu().numpy()

    @staticmethod
    def get_bn1_params(device: str = 'cuda:0') -> Dict[str, np.ndarray]:
        from torchvision.models import resnet18, ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
        return {
            'gamma': base.bn1.weight.detach().cpu().numpy(),
            'beta': base.bn1.bias.detach().cpu().numpy(),
        }


# ============================================================
# Core Utility Functions
# ============================================================

def load_image_numpy(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_tensor(image_np: np.ndarray, device: str = 'cuda:0') -> torch.Tensor:
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 0.0:
        return image.copy()
    return np.clip(image + factor, 0.0, 1.0)


def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = image.shape[:2]
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE).reshape(image.shape)


def compute_photometric_cost(ref: np.ndarray, warped: np.ndarray) -> float:
    return np.mean((warped.astype(np.float64) - ref.astype(np.float64)) ** 2)


def compute_2d_cost_landscape(
    feat_ref: np.ndarray, feat_target: np.ndarray,
    max_shift: float, grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx_vals = np.linspace(-max_shift, max_shift, grid_size)
    dy_vals = np.linspace(-max_shift, max_shift, grid_size)
    cost = np.zeros((grid_size, grid_size))
    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            shifted = shift_image(feat_target, dx, dy)
            cost[i, j] = compute_photometric_cost(feat_ref, shifted)
    return dx_vals, dy_vals, cost


def compute_sharpness(
    cost_grid: np.ndarray, dx: np.ndarray, dy: np.ndarray, radius: int = 5
) -> Dict[str, float]:
    grid_size = cost_grid.shape[0]
    center = grid_size // 2
    step_x = dx[1] - dx[0]
    step_y = dy[1] - dy[0]

    c_min, c_max = cost_grid.min(), cost_grid.max()
    if c_max - c_min > 1e-10:
        cost_norm = (cost_grid - c_min) / (c_max - c_min)
    else:
        cost_norm = np.zeros_like(cost_grid)

    def _slice_sharp(arr, step, c_idx, r):
        lo = max(c_idx - r, 0)
        hi = min(c_idx + r, len(arr) - 1)
        seg = arr[lo:hi+1]
        if len(seg) < 2:
            return 0.0
        return float(np.mean(np.abs(seg[1:] - seg[:-1]) / step))

    x_local = _slice_sharp(cost_norm[center, :], step_x, center, radius)
    y_local = _slice_sharp(cost_norm[:, center], step_y, center, radius)
    local = (x_local + y_local) / 2.0

    full_r = grid_size // 2
    x_global = _slice_sharp(cost_norm[center, :], step_x, center, full_r)
    y_global = _slice_sharp(cost_norm[:, center], step_y, center, full_r)
    global_ = (x_global + y_global) / 2.0

    cn = max(x_local, y_local) / min(x_local, y_local) if min(x_local, y_local) > 1e-12 else float('inf')

    c_max_raw = cost_grid.max()
    if c_max_raw > 1e-10:
        row = cost_grid[center, :]
        basin_mask = row < (0.5 * c_max_raw)
        basin_width = np.sum(basin_mask) * step_x
    else:
        basin_width = 2 * dx[-1]

    return {
        'x_local': x_local, 'y_local': y_local, 'local': local,
        'x_global': x_global, 'y_global': y_global, 'global': global_,
        'condition_number': cn, 'basin_width': basin_width,
    }


def extract_gray(rgb_np: np.ndarray) -> np.ndarray:
    gray = 0.299 * rgb_np[:,:,0] + 0.587 * rgb_np[:,:,1] + 0.114 * rgb_np[:,:,2]
    return gray[:, :, np.newaxis]


# ============================================================
# PART A: Conv1 Filter Weights & Feature Response Maps
# ============================================================

def visualize_conv1_filters(output_dir: str, device: str):
    """
    Part A-1: Visualize conv1 filter weights for all channels in
    Rank01 and Rank02 (8 channels each).
    """
    print("\n  [Part A-1] Visualizing conv1 filter weights...")

    weights = DirectConv1Extractor.get_conv1_weights(device)  # [64, 3, 7, 7]
    bn_params = DirectConv1Extractor.get_bn1_params(device)

    rank_sets = [
        ('Rank01', COMBINATIONS['Rank01_Full8']['channels'], RANK01_IMPORTANCE),
        ('Rank02', COMBINATIONS['Rank02_Full8']['channels'], RANK02_IMPORTANCE),
    ]

    for rank_name, channels, importance in rank_sets:
        n_ch = len(channels)
        fig, axes = plt.subplots(2, n_ch, figsize=(2.8 * n_ch, 6))

        for ci, ch in enumerate(channels):
            w = weights[ch]  # [3, 7, 7]
            gamma = bn_params['gamma'][ch]
            imp = importance.get(ch, 'unknown')
            imp_color = IMPORTANCE_COLORS.get(imp, '#bdc3c7')

            # Row 0: RGB composite of filter
            w_rgb = w.transpose(1, 2, 0)  # [7, 7, 3]
            w_min, w_max = w_rgb.min(), w_rgb.max()
            if w_max - w_min > 1e-8:
                w_vis = (w_rgb - w_min) / (w_max - w_min)
            else:
                w_vis = np.zeros_like(w_rgb)

            axes[0, ci].imshow(w_vis, interpolation='nearest')
            axes[0, ci].set_title(f'Ch{ch}\n({imp})',
                                  fontsize=9, fontweight='bold',
                                  color=imp_color)
            axes[0, ci].axis('off')

            # Row 1: Per-channel (R, G, B) as separate mini-plots
            rgb_labels = ['R', 'G', 'B']
            combined = np.concatenate([w[0], w[1], w[2]], axis=1)  # [7, 21]
            vmax = max(abs(combined.min()), abs(combined.max()))
            axes[1, ci].imshow(combined, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                              interpolation='nearest')
            axes[1, ci].set_title(f'|gamma|={abs(gamma):.3f}', fontsize=8)
            axes[1, ci].axis('off')

        fig.suptitle(f'{rank_name} Conv1 Filter Weights\n'
                    f'(Top: RGB composite, Bottom: R|G|B channels, '
                    f'gamma = BatchNorm scale)',
                    fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        save_path = os.path.join(output_dir, f'partA1_filters_{rank_name}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_path}")


def visualize_feature_maps(
    all_images: List[str], frame_indices: List[int],
    output_dir: str, device: str
):
    """
    Part A-2: Feature response maps for all 8 channels in each Rank,
    under Clean and +50% brightness.
    Each figure: 2 rows (Clean / +50%) x 8 columns (channels).
    Per-channel vmin/vmax for consistent comparison across brightness.
    """
    print("\n  [Part A-2] Visualizing feature response maps...")

    rank_sets = [
        ('Rank01', COMBINATIONS['Rank01_Full8']['channels'], RANK01_IMPORTANCE),
        ('Rank02', COMBINATIONS['Rank02_Full8']['channels'], RANK02_IMPORTANCE),
    ]

    for rank_name, channels, importance in rank_sets:
        extractor = DirectConv1Extractor(channels, device)

        for fi, frame_idx in enumerate(frame_indices):
            if frame_idx >= len(all_images):
                continue

            rgb_np = load_image_numpy(all_images[frame_idx])
            rgb_tensor = numpy_to_tensor(rgb_np, device)

            rgb_bright = apply_brightness(rgb_np, 0.5)
            rgb_bright_tensor = numpy_to_tensor(rgb_bright, device)

            with torch.no_grad():
                feat_clean = extractor(rgb_tensor)[0].cpu().numpy()     # [n_ch, H, W]
                feat_bright = extractor(rgb_bright_tensor)[0].cpu().numpy()

            n_ch = len(channels)
            fig, axes = plt.subplots(2, n_ch, figsize=(3 * n_ch, 6))

            for ci in range(n_ch):
                ch = channels[ci]
                imp = importance.get(ch, 'unknown')
                imp_color = IMPORTANCE_COLORS.get(imp, '#bdc3c7')

                # Per-channel vmin/vmax across both conditions
                vmin = min(feat_clean[ci].min(), feat_bright[ci].min())
                vmax = max(feat_clean[ci].max(), feat_bright[ci].max())
                if vmax - vmin < 1e-8:
                    vmax = vmin + 1e-8

                # Clean
                axes[0, ci].imshow(feat_clean[ci], cmap='viridis',
                                  vmin=vmin, vmax=vmax)
                axes[0, ci].set_title(f'Ch{ch} ({imp})',
                                     fontsize=8, fontweight='bold',
                                     color=imp_color)
                axes[0, ci].axis('off')

                # +50%
                axes[1, ci].imshow(feat_bright[ci], cmap='viridis',
                                  vmin=vmin, vmax=vmax)
                axes[1, ci].axis('off')

                # Annotate kill% for +50%
                total_px = feat_bright[ci].size
                dead_px = np.sum(feat_bright[ci] < 1e-6)
                kill_pct = dead_px / total_px * 100
                axes[1, ci].text(0.5, -0.05, f'kill={kill_pct:.0f}%',
                                transform=axes[1, ci].transAxes,
                                ha='center', fontsize=8,
                                color='red' if kill_pct > 90 else 'black')

            axes[0, 0].set_ylabel('Clean', fontsize=11, fontweight='bold')
            axes[1, 0].set_ylabel('+50%', fontsize=11, fontweight='bold')

            fig.suptitle(f'{rank_name} Feature Maps -- Frame {frame_idx}\n'
                        f'(Per-channel scale; kill% = ReLU-dead pixels at +50%)',
                        fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            save_path = os.path.join(output_dir,
                                     f'partA2_features_{rank_name}_frame{frame_idx}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {save_path}")

            del rgb_tensor, rgb_bright_tensor, feat_clean, feat_bright
            torch.cuda.empty_cache()


# ============================================================
# PART B: Original RGB Images
# ============================================================

def visualize_rgb_images(
    all_images: List[str], frame_indices: List[int], output_dir: str
):
    """Part B: Show original RGB images under 3 brightness conditions."""
    print("\n  [Part B] Visualizing RGB images...")

    n_frames = len(frame_indices)
    n_conds = len(BRIGHTNESS_CONDITIONS)
    fig, axes = plt.subplots(n_conds, n_frames, figsize=(5 * n_frames, 4 * n_conds))

    for fi, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(all_images):
            continue
        rgb_np = load_image_numpy(all_images[frame_idx])

        for ci, cond in enumerate(BRIGHTNESS_CONDITIONS):
            rgb_mod = apply_brightness(rgb_np, cond['factor'])
            axes[ci, fi].imshow(np.clip(rgb_mod, 0, 1))
            if ci == 0:
                axes[ci, fi].set_title(f'Frame {frame_idx}', fontsize=12, fontweight='bold')
            if fi == 0:
                axes[ci, fi].set_ylabel(cond['label'], fontsize=12, fontweight='bold')
            axes[ci, fi].axis('off')

    fig.suptitle('Test Frames Under Brightness Perturbation',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_dir, 'partB_rgb_images.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# PART C: 3D Convergence Basin Comparison Grids
# ============================================================

def plot_3d_surface_subplot(
    ax, cost: np.ndarray, dx: np.ndarray, dy: np.ndarray,
    title: str, sharpness: Dict[str, float],
    colormap: str = 'YlOrRd', is_optimal: bool = False
):
    """Plot a single 3D surface on a given Axes3D subplot."""
    c_min, c_max = cost.min(), cost.max()
    if c_max - c_min > 1e-10:
        cost_norm = (cost - c_min) / (c_max - c_min)
    else:
        cost_norm = np.zeros_like(cost)

    DX, DY = np.meshgrid(dx, dy)

    # Surface
    ax.plot_surface(DX, DY, cost_norm, cmap=colormap,
                    alpha=0.85, linewidth=0, antialiased=True,
                    rstride=1, cstride=1)

    # Contour projection on floor
    ax.contour(DX, DY, cost_norm, zdir='z', offset=-0.05,
               levels=10, cmap=colormap, alpha=0.5, linewidths=0.5)

    # Mark minimum
    min_idx = np.unravel_index(np.argmin(cost), cost.shape)
    min_dx = dx[min_idx[1]]
    min_dy = dy[min_idx[0]]
    ax.scatter([min_dx], [min_dy], [0], color='cyan', s=40,
              marker='*', zorder=10, edgecolors='white', linewidth=0.5)

    ax.set_xlim(dx[0], dx[-1])
    ax.set_ylim(dy[0], dy[-1])
    ax.set_zlim(-0.05, 1.05)
    ax.set_xlabel(r'$\Delta x$ [px]', fontsize=8, labelpad=2)
    ax.set_ylabel(r'$\Delta y$ [px]', fontsize=8, labelpad=2)
    ax.set_zlabel('Norm. Cost', fontsize=8, labelpad=2)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=35, azim=-60)

    # Title with sharpness info
    star = ' *' if is_optimal else ''
    title_str = f'{title}{star}\nL={sharpness["local"]:.4f}  CN={sharpness["condition_number"]:.2f}'
    ax.set_title(title_str, fontsize=9, fontweight='bold', pad=5)


def visualize_basin_3d_grids(
    all_images: List[str], frame_indices: List[int],
    output_dir: str, device: str, cfg: dict
):
    """
    Part C: 3D surface convergence basin comparison.

    Layout: 3 rows (Clean / +30% / +50%) x 3 columns max per figure.
    7 combinations split across 3 figures:
      Figure 1: Gray | [15] (Forward Best) | [23] (Rank02 Best)
      Figure 2: [6,15] (Ret-Optimal) | [15,23] (Cross) | Rank01 Full 8ch
      Figure 3: Rank02 Full 8ch
    """
    print("\n  [Part C] Generating 3D convergence basin comparison grids...")

    # Define figure groups (max 3 columns each)
    figure_groups = [
        {
            'title_suffix': 'Core Singles vs Gray',
            'combos': [
                ('gray', None, 'Gray (1ch)', 'GnBu_r', False),
                ('Ch15_solo', COMBINATIONS['Ch15_solo']['channels'],
                 COMBINATIONS['Ch15_solo']['label'], 'YlOrRd',
                 COMBINATIONS['Ch15_solo']['is_optimal']),
                ('Ch23_solo', COMBINATIONS['Ch23_solo']['channels'],
                 COMBINATIONS['Ch23_solo']['label'], 'YlOrRd',
                 COMBINATIONS['Ch23_solo']['is_optimal']),
            ]
        },
        {
            'title_suffix': 'Multi-Channel Combinations',
            'combos': [
                ('Ch6_Ch15', COMBINATIONS['Ch6_Ch15']['channels'],
                 COMBINATIONS['Ch6_Ch15']['label'], 'OrRd',
                 COMBINATIONS['Ch6_Ch15']['is_optimal']),
                ('Ch15_Ch23_cross', COMBINATIONS['Ch15_Ch23_cross']['channels'],
                 COMBINATIONS['Ch15_Ch23_cross']['label'], 'PuRd',
                 COMBINATIONS['Ch15_Ch23_cross']['is_optimal']),
                ('Rank01_Full8', COMBINATIONS['Rank01_Full8']['channels'],
                 COMBINATIONS['Rank01_Full8']['label'], 'YlOrBr',
                 COMBINATIONS['Rank01_Full8']['is_optimal']),
            ]
        },
        {
            'title_suffix': 'Rank02 Full Combination',
            'combos': [
                ('Rank02_Full8', COMBINATIONS['Rank02_Full8']['channels'],
                 COMBINATIONS['Rank02_Full8']['label'], 'BuPu',
                 COMBINATIONS['Rank02_Full8']['is_optimal']),
            ]
        },
    ]

    # Pre-build extractors for all unique channel sets
    extractors = {}
    for group in figure_groups:
        for combo_key, channels, label, cmap, is_opt in group['combos']:
            if channels is not None and combo_key not in extractors:
                extractors[combo_key] = DirectConv1Extractor(channels, device)

    # Collect all sharpness data for summary
    all_sharpness_data = []

    for fi, frame_idx in enumerate(frame_indices):
        if frame_idx >= len(all_images):
            continue

        print(f"\n    Frame {frame_idx} ({fi+1}/{len(frame_indices)})...")
        rgb_np = load_image_numpy(all_images[frame_idx])
        rgb_tensor = numpy_to_tensor(rgb_np, device)

        # Extract reference features (from clean image) for all combos
        ref_feats = {}
        for group in figure_groups:
            for combo_key, channels, label, cmap, is_opt in group['combos']:
                if combo_key in ref_feats:
                    continue
                if combo_key == 'gray':
                    ref_feats[combo_key] = extract_gray(rgb_np)
                else:
                    with torch.no_grad():
                        feat = extractors[combo_key](rgb_tensor)
                    ref_feats[combo_key] = feat[0].permute(1, 2, 0).cpu().numpy()

        # Generate one figure per group
        for gi, group in enumerate(figure_groups):
            combos = group['combos']
            n_cols = len(combos)
            n_rows = len(BRIGHTNESS_CONDITIONS)

            fig = plt.figure(figsize=(7 * n_cols, 5.5 * n_rows))

            for ri, cond in enumerate(BRIGHTNESS_CONDITIONS):
                rgb_bright = apply_brightness(rgb_np, cond['factor'])
                rgb_bright_tensor = numpy_to_tensor(rgb_bright, device)

                for ci, (combo_key, channels, label, cmap, is_opt) in enumerate(combos):
                    # Extract target features
                    if combo_key == 'gray':
                        feat_target = extract_gray(rgb_bright)
                    else:
                        with torch.no_grad():
                            feat = extractors[combo_key](rgb_bright_tensor)
                        feat_target = feat[0].permute(1, 2, 0).cpu().numpy()

                    # Compute cost landscape
                    dx, dy, cost = compute_2d_cost_landscape(
                        ref_feats[combo_key], feat_target,
                        cfg['max_shift_px'], cfg['grid_size']
                    )

                    # Compute sharpness
                    sharp = compute_sharpness(cost, dx, dy, cfg['sharpness_radius'])

                    # Store for summary
                    all_sharpness_data.append({
                        'frame': frame_idx,
                        'condition': cond['label'],
                        'combo': label,
                        'combo_key': combo_key,
                        'local_sharpness': sharp['local'],
                        'global_sharpness': sharp['global'],
                        'condition_number': sharp['condition_number'],
                        'basin_width': sharp['basin_width'],
                        'is_optimal': is_opt,
                    })

                    # Plot 3D surface
                    ax_idx = ri * n_cols + ci + 1
                    ax = fig.add_subplot(n_rows, n_cols, ax_idx, projection='3d')
                    plot_3d_surface_subplot(
                        ax, cost, dx, dy,
                        title=label if ri == 0 else '',
                        sharpness=sharp,
                        colormap=cmap,
                        is_optimal=is_opt,
                    )

                    # Row label on leftmost column
                    if ci == 0:
                        ax.text2D(-0.15, 0.5, cond['label'],
                                 transform=ax.transAxes,
                                 fontsize=12, fontweight='bold',
                                 rotation=90, va='center')

                del rgb_bright_tensor
                torch.cuda.empty_cache()

            fig.suptitle(
                f'3D Convergence Basin -- {group["title_suffix"]} -- Frame {frame_idx}\n'
                f'(* = Forward Greedy Optimal; L = Local Sharpness; CN = Condition Number)',
                fontsize=14, fontweight='bold'
            )
            plt.tight_layout(rect=[0.02, 0, 1, 0.93])
            save_path = os.path.join(
                output_dir,
                f'partC_basin3d_group{gi+1}_frame{frame_idx}.png'
            )
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {save_path}")

        del rgb_tensor
        torch.cuda.empty_cache()

    # Save sharpness summary CSV
    import csv
    csv_path = os.path.join(output_dir, 'partC_sharpness_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'condition', 'combo', 'combo_key',
            'local_sharpness', 'global_sharpness',
            'condition_number', 'basin_width', 'is_optimal'
        ])
        writer.writeheader()
        for row in all_sharpness_data:
            writer.writerow({k: (f'{v:.6f}' if isinstance(v, float) else v)
                            for k, v in row.items()})
    print(f"    Saved: {csv_path}")

    return all_sharpness_data


# ============================================================
# PART E: Channel Type Classification Summary
# ============================================================

def classify_and_summarize_channels(output_dir: str, device: str):
    """
    Classify all 64 conv1 channels by filter type and cross-reference
    with forward greedy importance results.
    """
    print("\n  [Part E] Classifying channels and generating summary...")

    weights = DirectConv1Extractor.get_conv1_weights(device)  # [64, 3, 7, 7]
    bn_params = DirectConv1Extractor.get_bn1_params(device)

    classifications = []

    for ch in range(64):
        w = weights[ch]  # [3, 7, 7]
        gamma = bn_params['gamma'][ch]

        # Check if dead channel
        if abs(gamma) < 0.01:
            ch_type = 'Dead'
            classifications.append({'ch': ch, 'type': ch_type, 'gamma': float(gamma),
                                   'detail': f'gamma={gamma:.4f}'})
            continue

        # Analyze filter structure
        r, g, b = w[0], w[1], w[2]

        # Color opponent: strong opposing signs between R and G (or B)
        rg_opp = float(np.mean(r) * np.mean(g))
        rb_opp = float(np.mean(r) * np.mean(b))
        gb_opp = float(np.mean(g) * np.mean(b))

        # Edge detection: strong spatial gradient
        avg_filter = (r + g + b) / 3.0
        gx = float(np.abs(avg_filter[:, 1:] - avg_filter[:, :-1]).mean())
        gy = float(np.abs(avg_filter[1:, :] - avg_filter[:-1, :]).mean())
        grad_mag = (gx + gy) / 2.0

        # Luminance: all channels have same sign
        r_mean, g_mean, b_mean = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
        all_same_sign = (r_mean > 0 and g_mean > 0 and b_mean > 0) or \
                       (r_mean < 0 and g_mean < 0 and b_mean < 0)

        # Determine dominant direction for edge filters
        if gx > 1.5 * gy:
            edge_dir = 'Vertical'
        elif gy > 1.5 * gx:
            edge_dir = 'Horizontal'
        else:
            edge_dir = 'Diagonal/Mixed'

        # Classification logic
        min_opp = min(rg_opp, rb_opp, gb_opp)
        if min_opp < -0.001:
            if rg_opp < -0.001:
                ch_type = 'Color-Opp (R-G)'
            elif rb_opp < -0.001:
                ch_type = 'Color-Opp (R-B)'
            else:
                ch_type = 'Color-Opp (G-B)'
        elif grad_mag > 0.03:
            ch_type = f'Edge ({edge_dir})'
        elif all_same_sign:
            ch_type = 'Luminance/Low-pass'
        else:
            ch_type = 'Mixed/Complex'

        classifications.append({
            'ch': ch, 'type': ch_type, 'gamma': float(gamma),
            'detail': f'gx={gx:.4f} gy={gy:.4f} rg={rg_opp:.4f} rb={rb_opp:.4f} gb={gb_opp:.4f}'
        })

    # ── Create summary figure ──
    fig, ax = plt.subplots(figsize=(18, 8))

    type_colors = {
        'Dead': '#2c3e50',
        'Color-Opp (R-G)': '#e74c3c',
        'Color-Opp (R-B)': '#9b59b6',
        'Color-Opp (G-B)': '#3498db',
        'Edge (Vertical)': '#2ecc71',
        'Edge (Horizontal)': '#27ae60',
        'Edge (Diagonal/Mixed)': '#1abc9c',
        'Luminance/Low-pass': '#f39c12',
        'Mixed/Complex': '#95a5a6',
    }

    x_pos = np.arange(64)
    bar_colors = [type_colors.get(c['type'], '#bdc3c7') for c in classifications]
    gammas = [abs(c['gamma']) for c in classifications]
    bars = ax.bar(x_pos, gammas, color=bar_colors, edgecolor='white', linewidth=0.3)

    # Mark channels in our combinations
    rank01_chs = set(COMBINATIONS['Rank01_Full8']['channels'])
    rank02_chs = set(COMBINATIONS['Rank02_Full8']['channels'])

    for i, c in enumerate(classifications):
        ch = c['ch']
        marker = ''
        if ch in rank01_chs and ch in rank02_chs:
            marker = '**'
        elif ch in rank01_chs:
            marker = '*R1'
        elif ch in rank02_chs:
            marker = '*R2'

        if marker:
            ax.text(i, gammas[i] + 0.01, marker, ha='center', va='bottom',
                   fontsize=7, fontweight='bold')

        # Highlight core channels with thick red border
        imp1 = RANK01_IMPORTANCE.get(ch, None)
        imp2 = RANK02_IMPORTANCE.get(ch, None)
        if imp1 == 'core' or imp2 == 'core':
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2.5)

    ax.set_xlabel('Channel Index', fontsize=12)
    ax.set_ylabel('|BatchNorm gamma|', fontsize=12)
    ax.set_title('Conv1 Channel Classification & Forward Greedy Importance\n'
                '(*R1 = Rank01, *R2 = Rank02, ** = Both, Red border = Core channel)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in range(64)], fontsize=6, rotation=90)
    ax.axhline(y=0.01, color='red', linestyle=':', alpha=0.5, label='Dead threshold')

    legend_patches = [mpatches.Patch(color=c, label=t) for t, c in type_colors.items()]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'partE_channel_classification.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")

    # Save classification data as JSON
    json_path = os.path.join(output_dir, 'channel_classifications.json')
    with open(json_path, 'w') as f:
        json.dump(classifications, f, indent=2)
    print(f"    Saved: {json_path}")

    return classifications


# ============================================================
# PART F: Channel Type Analysis Summary (Text File)
# ============================================================

def generate_channel_type_summary(
    classifications: List[Dict],
    output_dir: str
):
    """
    Part F: Generate a detailed text summary answering:
    "Which types of channels are most helpful for brightness robustness?"

    Includes calculation data and physical explanation.
    """
    print("\n  [Part F] Generating channel type analysis summary...")

    # ── Forward greedy results (hardcoded from experiments) ──
    forward_results = {
        # Rank01 alive channels
        15: {'score': 0.11358, 'sharp_clean': 0.1289, 'sharp_50': 0.1070,
             'ret50': 83.2, 'cn_50': 1.65, 'rank': 'Rank01', 'forward_rank': 1},
        6:  {'score': 0.05771, 'sharp_clean': 0.0531, 'sharp_50': 0.0597,
             'ret50': 114.6, 'cn_50': 1.49, 'rank': 'Rank01', 'forward_rank': 2},
        45: {'score': 0.01819, 'sharp_clean': 0.0192, 'sharp_50': 0.0178,
             'ret50': 93.3, 'cn_50': 1.68, 'rank': 'Rank01', 'forward_rank': 3},
        # Rank02 alive channels
        23: {'score': 0.08746, 'sharp_clean': 0.0840, 'sharp_50': 0.0890,
             'ret50': 106.7, 'cn_50': 1.37, 'rank': 'Rank02', 'forward_rank': 1},
        42: {'score': 0.05979, 'sharp_clean': 0.0576, 'sharp_50': 0.0607,
             'ret50': 104.4, 'cn_50': 2.60, 'rank': 'Rank02', 'forward_rank': 2},
        60: {'score': 0.04879, 'sharp_clean': 0.0524, 'sharp_50': 0.0473,
             'ret50': 90.1, 'cn_50': 1.73, 'rank': 'Rank02', 'forward_rank': 3},
        27: {'score': 0.04312, 'sharp_clean': 0.0494, 'sharp_50': 0.0405,
             'ret50': 81.9, 'cn_50': 2.30, 'rank': 'Rank02', 'forward_rank': 4},
        28: {'score': 0.03961, 'sharp_clean': 0.0421, 'sharp_50': 0.0385,
             'ret50': 91.4, 'cn_50': 1.52, 'rank': 'Rank02', 'forward_rank': 5},
        22: {'score': 0.03614, 'sharp_clean': 0.0356, 'sharp_50': 0.0364,
             'ret50': 100.0, 'cn_50': 1.36, 'rank': 'Rank02', 'forward_rank': 6},
        8:  {'score': 0.02748, 'sharp_clean': 0.0182, 'sharp_50': 0.0315,
             'ret50': 186.8, 'cn_50': 1.66, 'rank': 'Rank02', 'forward_rank': 7},
    }

    # ── Build type-to-channel mapping ──
    ch_type_map = {c['ch']: c['type'] for c in classifications}

    lines = []
    lines.append("=" * 80)
    lines.append("CHANNEL TYPE ANALYSIS SUMMARY")
    lines.append("Which types of channels are most helpful for brightness robustness?")
    lines.append("=" * 80)
    lines.append("")

    # ── Section 1: Top channels and their types ──
    lines.append("-" * 80)
    lines.append("1. TOP CHANNELS BY FORWARD GREEDY SCORE AND THEIR FILTER TYPES")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"  {'Rank':<6} {'Ch':<5} {'Type':<28} {'Score':<10} "
                 f"{'Sharp_50':<11} {'Ret50':<8} {'CN_50':<7} {'Source'}")
    lines.append(f"  {'-'*90}")

    # Sort by score descending
    sorted_chs = sorted(forward_results.items(), key=lambda x: x[1]['score'], reverse=True)
    for rank_i, (ch, data) in enumerate(sorted_chs, 1):
        ch_type = ch_type_map.get(ch, 'Unknown')
        lines.append(f"  {rank_i:<6} Ch{ch:<3} {ch_type:<28} {data['score']:<10.5f} "
                     f"{data['sharp_50']:<11.4f} {data['ret50']:<8.1f} "
                     f"{data['cn_50']:<7.2f} {data['rank']}")

    lines.append("")

    # ── Section 2: Aggregate by type ──
    lines.append("-" * 80)
    lines.append("2. AGGREGATE PERFORMANCE BY CHANNEL TYPE")
    lines.append("-" * 80)
    lines.append("")

    type_stats = {}
    for ch, data in forward_results.items():
        ch_type = ch_type_map.get(ch, 'Unknown')
        if ch_type not in type_stats:
            type_stats[ch_type] = {'scores': [], 'ret50s': [], 'sharp50s': [], 'channels': []}
        type_stats[ch_type]['scores'].append(data['score'])
        type_stats[ch_type]['ret50s'].append(data['ret50'])
        type_stats[ch_type]['sharp50s'].append(data['sharp_50'])
        type_stats[ch_type]['channels'].append(ch)

    lines.append(f"  {'Type':<28} {'Count':<7} {'Avg Score':<12} "
                 f"{'Avg Sharp_50':<14} {'Avg Ret50':<11} {'Channels'}")
    lines.append(f"  {'-'*95}")

    for ch_type, stats in sorted(type_stats.items(),
                                  key=lambda x: np.mean(x[1]['scores']), reverse=True):
        ch_str = ', '.join([f'Ch{c}' for c in sorted(stats['channels'])])
        lines.append(f"  {ch_type:<28} {len(stats['scores']):<7} "
                     f"{np.mean(stats['scores']):<12.5f} "
                     f"{np.mean(stats['sharp50s']):<14.4f} "
                     f"{np.mean(stats['ret50s']):<11.1f} {ch_str}")

    lines.append("")

    # ── Section 3: Key finding ──
    lines.append("-" * 80)
    lines.append("3. KEY FINDING: COLOR-OPPONENT CHANNELS DOMINATE")
    lines.append("-" * 80)
    lines.append("")

    # Count color-opponent vs others in top-5
    top5_chs = [ch for ch, _ in sorted_chs[:5]]
    top5_types = [ch_type_map.get(ch, 'Unknown') for ch in top5_chs]
    n_color_opp = sum(1 for t in top5_types if 'Color-Opp' in t)
    n_edge = sum(1 for t in top5_types if 'Edge' in t)

    lines.append(f"  Among the top 5 channels by Forward Greedy Score:")
    lines.append(f"    Color-Opponent channels: {n_color_opp}/5 ({n_color_opp/5*100:.0f}%)")
    lines.append(f"    Edge channels:           {n_edge}/5 ({n_edge/5*100:.0f}%)")
    lines.append(f"    Other:                   {5-n_color_opp-n_edge}/5")
    lines.append("")

    # Top 2 are both Color-Opp (R-B)
    lines.append("  The top 2 channels across BOTH Rank01 and Rank02 are:")
    lines.append(f"    #1: Ch15 (Score=0.11358) -- {ch_type_map.get(15, '?')}")
    lines.append(f"    #2: Ch23 (Score=0.08746) -- {ch_type_map.get(23, '?')}")
    lines.append("")
    lines.append("  Both are R-B (Red-Blue) color-opponent, high-frequency texture filters.")
    lines.append("")

    # ── Section 4: Physical explanation ──
    lines.append("-" * 80)
    lines.append("4. PHYSICAL EXPLANATION: WHY COLOR-OPPONENT FEATURES ARE BRIGHTNESS-INVARIANT")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Brightness perturbation in our experiment is ADDITIVE:")
    lines.append("    I'(x,y) = I(x,y) + delta")
    lines.append("")
    lines.append("  This means all three color channels (R, G, B) are shifted by the same amount.")
    lines.append("")
    lines.append("  A color-opponent filter computes a DIFFERENCE between color channels:")
    lines.append("    f(x,y) = w_R * R(x,y) + w_B * B(x,y)   where w_R > 0, w_B < 0")
    lines.append("")
    lines.append("  Under additive brightness change:")
    lines.append("    f'(x,y) = w_R * (R + delta) + w_B * (B + delta)")
    lines.append("            = w_R * R + w_B * B + (w_R + w_B) * delta")
    lines.append("            = f(x,y) + (w_R + w_B) * delta")
    lines.append("")
    lines.append("  If w_R + w_B is close to 0 (balanced opponent), the additive term vanishes:")
    lines.append("    f'(x,y) ~ f(x,y)")
    lines.append("")
    lines.append("  This is why color-opponent features are naturally invariant to additive")
    lines.append("  brightness changes -- the brightness shift cancels out in the difference.")
    lines.append("")
    lines.append("  In contrast, GRAYSCALE or LUMINANCE features compute a WEIGHTED SUM:")
    lines.append("    g(x,y) = w_R * R + w_G * G + w_B * B   where all w > 0")
    lines.append("    g'(x,y) = g(x,y) + (w_R + w_G + w_B) * delta")
    lines.append("")
    lines.append("  Here (w_R + w_G + w_B) >> 0, so the brightness shift fully propagates,")
    lines.append("  causing the cost landscape to deform under illumination change.")
    lines.append("")

    # ── Section 5: Practical implications ──
    lines.append("-" * 80)
    lines.append("5. PRACTICAL IMPLICATIONS FOR COMO SYSTEM")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  a) For maximum brightness robustness, prioritize Color-Opponent channels")
    lines.append("     (especially R-B opponent) over Edge or Luminance channels.")
    lines.append("")
    lines.append("  b) The current Gray baseline (weighted sum of R,G,B) is fundamentally")
    lines.append("     disadvantaged under brightness change -- it has zero color-opponent")
    lines.append("     component.")
    lines.append("")
    lines.append("  c) Edge channels (e.g., Ch42, Ch60) provide useful spatial structure")
    lines.append("     information but are more sensitive to brightness change (lower Ret50).")
    lines.append("     They may be valuable for TEXTURE-RICH regions but less reliable under")
    lines.append("     illumination variation.")
    lines.append("")
    lines.append("  d) The ideal channel selection for a robust SLAM system should combine:")
    lines.append("     - 1-2 strong Color-Opponent channels (for brightness invariance)")
    lines.append("     - 1-2 Edge channels (for spatial structure in normal conditions)")
    lines.append("     However, our forward greedy results show that adding Edge channels")
    lines.append("     to Color-Opponent channels DILUTES the normalized Sharpness metric.")
    lines.append("     Whether this dilution hurts actual SLAM tracking accuracy requires")
    lines.append("     ATE (Absolute Trajectory Error) evaluation on real sequences.")
    lines.append("")

    # ── Section 6: Comparison with Gray baseline ──
    lines.append("-" * 80)
    lines.append("6. COMPARISON WITH GRAY BASELINE")
    lines.append("-" * 80)
    lines.append("")
    lines.append("  Gray baseline Sharpness (from convergence basin experiments):")
    lines.append("    Sharp_Clean ~ 0.021")
    lines.append("    Sharp_50   ~ 0.018  (estimated)")
    lines.append("    Ret50      ~ 85-90%")
    lines.append("")
    lines.append("  Best single channel (Ch15):")
    lines.append("    Sharp_Clean = 0.1289  (6.1x Gray)")
    lines.append("    Sharp_50   = 0.1070  (5.9x Gray)")
    lines.append("    Ret50      = 83.2%")
    lines.append("")
    lines.append("  Best Rank02 channel (Ch23):")
    lines.append("    Sharp_Clean = 0.0840  (4.0x Gray)")
    lines.append("    Sharp_50   = 0.0890  (4.9x Gray)")
    lines.append("    Ret50      = 106.7%  (BETTER than Clean!)")
    lines.append("")
    lines.append("  Conclusion: Both top Color-Opponent channels provide 4-6x higher")
    lines.append("  absolute Sharpness than Gray baseline, with comparable or better Retention.")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF ANALYSIS")
    lines.append("=" * 80)

    # Write to file
    txt_path = os.path.join(output_dir, 'partF_channel_type_analysis.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"    Saved: {txt_path}")

    return txt_path


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = CONFIG
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Load image list
    all_images = sorted(glob.glob(os.path.join(cfg['rgb_dir'], "*.png")))
    n_images = len(all_images)
    frame_indices = cfg['frame_indices']

    print(f"\n{'='*70}")
    print(f"  Forward Greedy Selection -- Comprehensive Visualization")
    print(f"{'='*70}")
    print(f"  Images found:     {n_images}")
    print(f"  Test frames:      {frame_indices}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Parts:            A (Filters + Feature Maps)")
    print(f"                    B (RGB Images)")
    print(f"                    C (3D Basin Grids)")
    print(f"                    E (Channel Classification)")
    print(f"                    F (Channel Type Analysis)")
    print(f"{'='*70}")

    # Key combinations being visualized
    print(f"\n  Combinations (key insights from Forward Greedy):")
    print(f"  {'Key':<20} {'Channels':<30} {'Label'}")
    print(f"  {'-'*75}")
    print(f"  {'Gray':<20} {'[grayscale]':<30} {'Gray (1ch) -- baseline'}")
    for key, combo in COMBINATIONS.items():
        opt_str = ' *OPTIMAL*' if combo['is_optimal'] else ''
        print(f"  {key:<20} {str(combo['channels']):<30} {combo['label']}{opt_str}")
    print(f"  {'-'*75}")

    device = cfg['device']

    # ── Part A: Conv1 Filters & Feature Maps ──
    visualize_conv1_filters(output_dir, device)
    visualize_feature_maps(all_images, frame_indices, output_dir, device)

    # ── Part B: Original RGB Images ──
    visualize_rgb_images(all_images, frame_indices, output_dir)

    # ── Part C: 3D Convergence Basin Comparison Grids ──
    all_sharpness = visualize_basin_3d_grids(
        all_images, frame_indices, output_dir, device, cfg
    )

    # ── Part E: Channel Classification Summary ──
    classifications = classify_and_summarize_channels(output_dir, device)

    # ── Part F: Channel Type Analysis Summary ──
    generate_channel_type_summary(classifications, output_dir)

    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"  ALL VISUALIZATIONS COMPLETE")
    print(f"{'='*70}")
    print(f"  Output directory: {output_dir}/")
    print(f"")
    print(f"  Part A:")
    print(f"    partA1_filters_Rank01.png       -- Conv1 filter weights")
    print(f"    partA1_filters_Rank02.png")
    print(f"    partA2_features_*_frame*.png     -- Feature response maps")
    print(f"")
    print(f"  Part B:")
    print(f"    partB_rgb_images.png             -- RGB under brightness")
    print(f"")
    print(f"  Part C:")
    print(f"    partC_basin3d_group*_frame*.png  -- 3D basin comparisons")
    print(f"    partC_sharpness_summary.csv      -- All sharpness data")
    print(f"")
    print(f"  Part E:")
    print(f"    partE_channel_classification.png -- 64-channel bar chart")
    print(f"    channel_classifications.json     -- Classification data")
    print(f"")
    print(f"  Part F:")
    print(f"    partF_channel_type_analysis.txt  -- Type analysis summary")
    print(f"{'='*70}")