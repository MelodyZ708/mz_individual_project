"""
Comprehensive Ablation Visualization
=====================================
CNN Feature Injection into COMO — Post-Ablation Analysis

Generates all visualizations needed to interpret backward ablation results:

  Part A: Conv1 filter weights & feature response maps (per channel)
  Part B: Original RGB images for 3 test frames (Clean / +30% / +50%)
  Part C: Convergence basin comparison grid (Gray vs 8ch vs Optimal sub-combo)
  Part D: 1D cross-section overlays (Clean vs +50% basin shape comparison)
  Part E: Channel type classification summary

Usage:
  python visualize_ablation_results.py

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
    'output_dir': 'vis_results/ablation_visualization',
    'device': 'cuda:0',

    # Test frames (early / mid / late) — same as ablation experiment
    'frame_indices': [41, 306, 512],

    # Perturbation range (pixels)
    'max_shift_px': 30,
    'grid_size': 61,

    # Sharpness config
    'sharpness_radius': 5,
}

# ── Channel combinations from ablation results ──
COMBINATIONS = {
    'Rank01_Full8': {
        'channels': [6, 7, 12, 15, 36, 45, 58, 62],
        'label': 'Rank01 Full (8ch)',
        'color': '#1a5276',
    },
    'Rank01_Optimal4': {
        'channels': [12, 15, 58, 62],
        'label': 'Rank01 Optimal (4ch)',
        'color': '#2e86c1',
    },
    'Rank02_Full8': {
        'channels': [8, 22, 23, 27, 28, 42, 48, 60],
        'label': 'Rank02 Full (8ch)',
        'color': '#7d3c98',
    },
    'Rank02_Optimal3': {
        'channels': [8, 27, 48],
        'label': 'Rank02 Optimal (3ch)',
        'color': '#a569bd',
    },
}

# ── Channel importance from ablation (removal order = least to most important) ──
# Rank01: removed 45 → 6 → 7 → 36 → 62 → 12 → 58 → [15 survives]
RANK01_IMPORTANCE = {
    45: 'redundant', 6: 'redundant', 7: 'redundant', 36: 'dead',
    62: 'moderate', 12: 'important', 58: 'critical', 15: 'core',
}
# Rank02: removed 60 → 28 → 22 → 42 → 23 → 48 → 27 → [8 survives]
RANK02_IMPORTANCE = {
    60: 'redundant', 28: 'redundant', 22: 'redundant', 42: 'moderate',
    23: 'moderate', 48: 'dead', 27: 'important', 8: 'core',
}

IMPORTANCE_COLORS = {
    'core':      '#c0392b',   # red — absolutely essential
    'critical':  '#e74c3c',   # lighter red
    'important': '#f39c12',   # orange
    'moderate':  '#f1c40f',   # yellow
    'redundant': '#95a5a6',   # gray
    'dead':      '#2c3e50',   # dark gray
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
# Feature Extractor (Direct Conv1 — matches ablation scripts)
# ============================================================
class DirectConv1Extractor(nn.Module):
    """Extract specific channels from ResNet-18 conv1 + bn1 + relu.
    Applies ImageNet normalization before feature extraction."""

    # ImageNet normalization constants
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

        # Register ImageNet normalization as buffers
        self.register_buffer('mean', torch.tensor(self.IMAGENET_MEAN, device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(self.IMAGENET_STD, device=device).view(1, 3, 1, 1))

        for p in self.parameters():
            p.requires_grad_(False)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet normalization."""
        return (x.float() - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]  # (H, W)
        x = self._normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        selected = x[:, self.channel_indices, :, :]
        # Upsample 2× back to original resolution (conv1 stride=2)
        upsampled = F.interpolate(selected, size=orig_size,
                                  mode='bilinear', align_corners=False)
        return upsampled

    def get_full_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get ALL 64 channels (before channel selection)."""
        orig_size = x.shape[-2:]  # (H, W)
        x = self._normalize(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Upsample 2× back to original resolution
        upsampled = F.interpolate(x, size=orig_size,
                                  mode='bilinear', align_corners=False)
        return upsampled

    @staticmethod
    def get_conv1_weights(device: str = 'cuda:0') -> np.ndarray:
        """Get conv1 filter weights as numpy [64, 3, 7, 7]."""
        from torchvision.models import resnet18, ResNet18_Weights
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
        return base.conv1.weight.detach().cpu().numpy()

    @staticmethod
    def get_bn1_params(device: str = 'cuda:0') -> Dict[str, np.ndarray]:
        """Get bn1 gamma and beta parameters."""
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
    """Load image as float32 numpy [H, W, 3] in [0, 1]."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_tensor(image_np: np.ndarray, device: str = 'cuda:0') -> torch.Tensor:
    """Convert numpy [H, W, 3] to tensor [1, 3, H, W]."""
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    """Apply additive brightness change."""
    if factor == 0.0:
        return image.copy()
    return np.clip(image + factor, 0.0, 1.0)


def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift image by (dx, dy) pixels using affine transformation."""
    h, w = image.shape[:2]
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE).reshape(image.shape)


def compute_photometric_cost(ref: np.ndarray, warped: np.ndarray) -> float:
    """MSE photometric cost."""
    return np.mean((warped.astype(np.float64) - ref.astype(np.float64)) ** 2)


def compute_2d_cost_landscape(
    feat_ref: np.ndarray, feat_target: np.ndarray,
    max_shift: float, grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cost over 2D translation grid."""
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
    """Compute sharpness metrics on normalized cost surface."""
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

    # Condition number
    cn = max(x_local, y_local) / min(x_local, y_local) if min(x_local, y_local) > 1e-12 else float('inf')

    # Basin width (50%)
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
    """Convert RGB to grayscale feature [H, W, 1]."""
    gray = 0.299 * rgb_np[:,:,0] + 0.587 * rgb_np[:,:,1] + 0.114 * rgb_np[:,:,2]
    return gray[:, :, np.newaxis]


# ============================================================
# PART A: Conv1 Filter Weights & Feature Response Maps
# ============================================================

def visualize_conv1_filters(output_dir: str, device: str):
    """
    Visualize conv1 filter weights for Rank01 and Rank02 channels,
    with importance color-coding from ablation results.
    """
    print("\n  [Part A-1] Visualizing Conv1 filter weights...")
    weights = DirectConv1Extractor.get_conv1_weights(device)  # [64, 3, 7, 7]
    bn_params = DirectConv1Extractor.get_bn1_params(device)

    for combo_name, channels, importance_map in [
        ('Rank01', COMBINATIONS['Rank01_Full8']['channels'], RANK01_IMPORTANCE),
        ('Rank02', COMBINATIONS['Rank02_Full8']['channels'], RANK02_IMPORTANCE),
    ]:
        n_ch = len(channels)
        fig, axes = plt.subplots(2, n_ch, figsize=(2.5 * n_ch, 6))

        for idx, ch in enumerate(channels):
            # Row 1: RGB composite of the 7x7x3 filter
            w = weights[ch]  # [3, 7, 7]
            # Normalize each filter to [0, 1] for display
            w_rgb = np.transpose(w, (1, 2, 0))  # [7, 7, 3]
            w_min, w_max = w_rgb.min(), w_rgb.max()
            if w_max - w_min > 1e-8:
                w_disp = (w_rgb - w_min) / (w_max - w_min)
            else:
                w_disp = np.zeros_like(w_rgb)

            ax = axes[0, idx]
            ax.imshow(w_disp, interpolation='nearest')
            ax.set_title(f'Ch {ch}', fontsize=10, fontweight='bold')
            ax.axis('off')

            # Color border based on importance
            imp = importance_map.get(ch, 'redundant')
            border_color = IMPORTANCE_COLORS[imp]
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4)
                spine.set_visible(True)
            ax.set_frame_on(True)

            # Row 2: Per-channel weights (R, G, B stacked vertically)
            ax2 = axes[1, idx]
            # Show the 3 input channels side by side
            r_ch = w[0]  # [7, 7]
            g_ch = w[1]
            b_ch = w[2]
            combined = np.concatenate([r_ch, g_ch, b_ch], axis=0)  # [21, 7]
            im = ax2.imshow(combined, cmap='RdBu_r', interpolation='nearest',
                           vmin=-combined.max(), vmax=combined.max())
            ax2.set_yticks([3, 10, 17])
            ax2.set_yticklabels(['R', 'G', 'B'], fontsize=9)
            ax2.set_xticks([])

            # Annotate importance and BN gamma
            gamma = bn_params['gamma'][ch]
            imp_label = imp.upper()
            ax2.set_xlabel(f'{imp_label}\n' + r'$\gamma$' + f'={gamma:.3f}', fontsize=8)

        # Add legend
        legend_patches = [mpatches.Patch(color=c, label=l.capitalize())
                         for l, c in IMPORTANCE_COLORS.items()]
        fig.legend(handles=legend_patches, loc='lower center', ncol=6,
                  fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(f'{combo_name} — Conv1 Filter Weights (7×7×3)\n'
                    f'Top: RGB composite | Bottom: R/G/B channels (RdBu colormap)',
                    fontsize=13, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        save_path = os.path.join(output_dir, f'partA_filters_{combo_name}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_path}")


def visualize_feature_maps(
    all_images: List[str], frame_indices: List[int],
    output_dir: str, device: str
):
    """
    Visualize feature response maps for each channel under Clean and +50%.
    """
    print("\n  [Part A-2] Visualizing feature response maps...")

    for combo_name, channels, importance_map in [
        ('Rank01', COMBINATIONS['Rank01_Full8']['channels'], RANK01_IMPORTANCE),
        ('Rank02', COMBINATIONS['Rank02_Full8']['channels'], RANK02_IMPORTANCE),
    ]:
        extractor = DirectConv1Extractor(list(range(64)), device)  # extract all 64

        for fi, frame_idx in enumerate(frame_indices):
            rgb_np = load_image_numpy(all_images[frame_idx])
            rgb_tensor = numpy_to_tensor(rgb_np, device)
            rgb_bright = apply_brightness(rgb_np, 0.5)
            rgb_bright_tensor = numpy_to_tensor(rgb_bright, device)

            with torch.no_grad():
                feats_clean = extractor.get_full_features(rgb_tensor)[0].cpu().numpy()  # [64, H, W]
                feats_bright = extractor.get_full_features(rgb_bright_tensor)[0].cpu().numpy()

            n_ch = len(channels)
            fig, axes = plt.subplots(2, n_ch, figsize=(2.8 * n_ch, 6))

            for idx, ch in enumerate(channels):
                imp = importance_map.get(ch, 'redundant')
                border_color = IMPORTANCE_COLORS[imp]

                fm_clean = feats_clean[ch]
                fm_bright = feats_bright[ch]

                # Use joint min/max across Clean and +50% for THIS channel
                # so both rows share the same color scale per channel
                joint_min = min(fm_clean.min(), fm_bright.min())
                joint_max = max(fm_clean.max(), fm_bright.max())

                # Row 1: Clean
                ax = axes[0, idx]
                ax.imshow(fm_clean, cmap='viridis', interpolation='bilinear',
                          vmin=joint_min, vmax=joint_max)
                if idx == 0:
                    ax.set_ylabel('Clean', fontsize=11, fontweight='bold')
                ax.set_title(f'Ch {ch} ({imp})', fontsize=9, fontweight='bold',
                           color=border_color)
                ax.axis('off')

                # Row 2: +50% brightness
                ax2 = axes[1, idx]
                ax2.imshow(fm_bright, cmap='viridis', interpolation='bilinear',
                          vmin=joint_min, vmax=joint_max)  # same per-channel scale
                if idx == 0:
                    ax2.set_ylabel('+50%', fontsize=11, fontweight='bold')
                ax2.axis('off')

                # Compute correlation between clean and bright
                if fm_clean.std() > 1e-8 and fm_bright.std() > 1e-8:
                    corr = np.corrcoef(fm_clean.flatten(), fm_bright.flatten())[0, 1]
                else:
                    corr = 0.0
                ax2.set_xlabel(f'corr={corr:.3f}', fontsize=8)

            fig.suptitle(f'{combo_name} Feature Maps — Frame {frame_idx}\n'
                        f'Clean vs +50% Brightness (same color scale)',
                        fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            save_path = os.path.join(output_dir,
                                     f'partA_featuremaps_{combo_name}_frame{frame_idx}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"    Saved: {save_path}")

            del rgb_tensor, rgb_bright_tensor
            torch.cuda.empty_cache()


# ============================================================
# PART B: Original RGB Images
# ============================================================

def visualize_rgb_images(
    all_images: List[str], frame_indices: List[int], output_dir: str
):
    """Show the 3 test frames under Clean, +30%, +50% brightness."""
    print("\n  [Part B] Visualizing original RGB images...")

    n_frames = len(frame_indices)
    fig, axes = plt.subplots(3, n_frames, figsize=(5 * n_frames, 12))

    for fi, frame_idx in enumerate(frame_indices):
        rgb_np = load_image_numpy(all_images[frame_idx])

        for ri, cond in enumerate(BRIGHTNESS_CONDITIONS):
            img = apply_brightness(rgb_np, cond['factor'])
            ax = axes[ri, fi]
            ax.imshow(np.clip(img, 0, 1))
            if fi == 0:
                ax.set_ylabel(cond['label'], fontsize=13, fontweight='bold')
            if ri == 0:
                ax.set_title(f'Frame {frame_idx}', fontsize=13, fontweight='bold')
            ax.axis('off')

    fig.suptitle('Test Frames — Brightness Conditions', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(output_dir, 'partB_rgb_images.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


# ============================================================
# PART C: Convergence Basin Comparison Grid
# ============================================================

def visualize_basin_comparison(
    all_images: List[str], frame_indices: List[int],
    output_dir: str, device: str, cfg: dict
):
    """
    For each frame: 4-column x 3-row grid of 2D heatmaps.
    Columns: Gray | Rank01 Full 8ch | Rank01 Optimal 4ch | Rank02 Optimal 3ch
    Rows: Clean | +30% | +50%
    """
    print("\n  [Part C] Generating convergence basin comparison grids...")

    modalities = [
        ('gray', None, 'Gray (1ch)'),
        ('Rank01_Full8', COMBINATIONS['Rank01_Full8']['channels'], COMBINATIONS['Rank01_Full8']['label']),
        ('Rank01_Optimal4', COMBINATIONS['Rank01_Optimal4']['channels'], COMBINATIONS['Rank01_Optimal4']['label']),
        ('Rank02_Optimal3', COMBINATIONS['Rank02_Optimal3']['channels'], COMBINATIONS['Rank02_Optimal3']['label']),
    ]

    # Pre-build extractors
    extractors = {}
    for mod_key, channels, _ in modalities:
        if channels is not None:
            extractors[mod_key] = DirectConv1Extractor(channels, device)

    for fi, frame_idx in enumerate(frame_indices):
        print(f"\n    Frame {frame_idx} ({fi+1}/{len(frame_indices)})...")
        rgb_np = load_image_numpy(all_images[frame_idx])
        rgb_tensor = numpy_to_tensor(rgb_np, device)

        # Extract reference features (always from clean image)
        ref_feats = {}
        for mod_key, channels, _ in modalities:
            if mod_key == 'gray':
                ref_feats[mod_key] = extract_gray(rgb_np)
            else:
                with torch.no_grad():
                    feat = extractors[mod_key](rgb_tensor)
                ref_feats[mod_key] = feat[0].permute(1, 2, 0).cpu().numpy()

        # Create figure: 4 cols x 3 rows
        fig, axes = plt.subplots(3, 4, figsize=(22, 14))

        for ri, cond in enumerate(BRIGHTNESS_CONDITIONS):
            rgb_bright = apply_brightness(rgb_np, cond['factor'])
            rgb_bright_tensor = numpy_to_tensor(rgb_bright, device)

            for ci, (mod_key, channels, mod_label) in enumerate(modalities):
                # Extract target features
                if mod_key == 'gray':
                    feat_target = extract_gray(rgb_bright)
                else:
                    with torch.no_grad():
                        feat = extractors[mod_key](rgb_bright_tensor)
                    feat_target = feat[0].permute(1, 2, 0).cpu().numpy()

                # Compute cost landscape
                dx, dy, cost = compute_2d_cost_landscape(
                    ref_feats[mod_key], feat_target,
                    cfg['max_shift_px'], cfg['grid_size']
                )

                # Compute sharpness
                sharp = compute_sharpness(cost, dx, dy, cfg['sharpness_radius'])

                # Normalize cost
                c_min, c_max = cost.min(), cost.max()
                if c_max - c_min > 1e-10:
                    cost_norm = (cost - c_min) / (c_max - c_min)
                else:
                    cost_norm = np.zeros_like(cost)

                # Plot
                ax = axes[ri, ci]
                extent = [dx[0], dx[-1], dy[-1], dy[0]]
                im = ax.imshow(cost_norm, extent=extent, cmap='inferno',
                              aspect='equal', interpolation='bilinear',
                              vmin=0, vmax=1)

                DX_grid, DY_grid = np.meshgrid(dx, dy)
                ax.contour(DX_grid, DY_grid, cost_norm,
                          levels=8, colors='white', linewidths=0.4, alpha=0.5)

                # Mark minimum
                min_idx = np.unravel_index(np.argmin(cost), cost.shape)
                min_dx = dx[min_idx[1]]
                min_dy = dy[min_idx[0]]
                ax.plot(min_dx, min_dy, 'c*', markersize=10,
                       markeredgecolor='white', markeredgewidth=0.6)

                # Annotations
                ret_str = ''
                if cond['key'] != 'clean':
                    # We need clean sharpness for retention
                    # Compute it inline
                    _, _, cost_clean = compute_2d_cost_landscape(
                        ref_feats[mod_key], ref_feats[mod_key],
                        cfg['max_shift_px'], cfg['grid_size']
                    ) if ri == 0 else (None, None, None)

                info_text = (f"L={sharp['local']:.4f}\n"
                            f"BW={sharp['basin_width']:.1f}px\n"
                            f"CN={sharp['condition_number']:.2f}")
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6),
                       color='white', family='monospace')

                if ri == 0:
                    ax.set_title(mod_label, fontsize=12, fontweight='bold')
                if ci == 0:
                    ax.set_ylabel(cond['label'], fontsize=12, fontweight='bold')

                ax.axhline(y=0, color='white', linestyle='--', alpha=0.2, linewidth=0.5)
                ax.axvline(x=0, color='white', linestyle='--', alpha=0.2, linewidth=0.5)

                if ri < 2:
                    ax.set_xticklabels([])
                if ci > 0:
                    ax.set_yticklabels([])

            del rgb_bright_tensor
            torch.cuda.empty_cache()

        fig.suptitle(f'Convergence Basin Comparison — Frame {frame_idx}\n'
                    f'(2D Heatmap, Normalized Cost, +/-{cfg["max_shift_px"]}px)',
                    fontsize=15, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = os.path.join(output_dir, f'partC_basin_grid_frame{frame_idx}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_path}")

        del rgb_tensor
        torch.cuda.empty_cache()


# ============================================================
# PART D: 1D Cross-Section Overlays
# ============================================================

def visualize_1d_cross_sections(
    all_images: List[str], frame_indices: List[int],
    output_dir: str, device: str, cfg: dict
):
    """
    For each frame: overlay Clean vs +50% 1D cross-sections (X and Y)
    for Gray, Rank01 Full, Rank01 Optimal, Rank02 Optimal.
    This is the most direct way to see if basin shape is preserved.
    """
    print("\n  [Part D] Generating 1D cross-section overlays...")

    modalities = [
        ('gray', None, 'Gray (1ch)', '#7f8c8d'),
        ('Rank01_Full8', COMBINATIONS['Rank01_Full8']['channels'],
         'Rank01 Full 8ch', COMBINATIONS['Rank01_Full8']['color']),
        ('Rank01_Optimal4', COMBINATIONS['Rank01_Optimal4']['channels'],
         'Rank01 Opt 4ch', COMBINATIONS['Rank01_Optimal4']['color']),
        ('Rank02_Optimal3', COMBINATIONS['Rank02_Optimal3']['channels'],
         'Rank02 Opt 3ch', COMBINATIONS['Rank02_Optimal3']['color']),
    ]

    extractors = {}
    for mod_key, channels, _, _ in modalities:
        if channels is not None:
            extractors[mod_key] = DirectConv1Extractor(channels, device)

    for fi, frame_idx in enumerate(frame_indices):
        print(f"\n    Frame {frame_idx} ({fi+1}/{len(frame_indices)})...")
        rgb_np = load_image_numpy(all_images[frame_idx])
        rgb_tensor = numpy_to_tensor(rgb_np, device)
        rgb_bright = apply_brightness(rgb_np, 0.5)
        rgb_bright_tensor = numpy_to_tensor(rgb_bright, device)

        fig, axes = plt.subplots(len(modalities), 2, figsize=(14, 3.5 * len(modalities)))

        for mi, (mod_key, channels, mod_label, color) in enumerate(modalities):
            # Extract features
            if mod_key == 'gray':
                feat_ref = extract_gray(rgb_np)
                feat_clean = extract_gray(rgb_np)
                feat_bright = extract_gray(rgb_bright)
            else:
                with torch.no_grad():
                    feat_ref_t = extractors[mod_key](rgb_tensor)
                    feat_bright_t = extractors[mod_key](rgb_bright_tensor)
                feat_ref = feat_ref_t[0].permute(1, 2, 0).cpu().numpy()
                feat_clean = feat_ref.copy()
                feat_bright = feat_bright_t[0].permute(1, 2, 0).cpu().numpy()

            # Compute cost landscapes
            dx, dy, cost_clean = compute_2d_cost_landscape(
                feat_ref, feat_clean, cfg['max_shift_px'], cfg['grid_size'])
            _, _, cost_bright = compute_2d_cost_landscape(
                feat_ref, feat_bright, cfg['max_shift_px'], cfg['grid_size'])

            center = cfg['grid_size'] // 2

            # Normalize both to [0, 1] using clean's range for fair comparison
            c_min = min(cost_clean.min(), cost_bright.min())
            c_max = max(cost_clean.max(), cost_bright.max())
            if c_max - c_min > 1e-10:
                clean_norm = (cost_clean - c_min) / (c_max - c_min)
                bright_norm = (cost_bright - c_min) / (c_max - c_min)
            else:
                clean_norm = np.zeros_like(cost_clean)
                bright_norm = np.zeros_like(cost_bright)

            # X-direction (row at center)
            ax_x = axes[mi, 0]
            ax_x.plot(dx, clean_norm[center, :], color=color, linewidth=2,
                     label='Clean', linestyle='-')
            ax_x.plot(dx, bright_norm[center, :], color=color, linewidth=2,
                     label='+50%', linestyle='--', alpha=0.7)
            ax_x.fill_between(dx, clean_norm[center, :], bright_norm[center, :],
                            alpha=0.15, color=color)
            ax_x.set_ylabel('Norm. Cost', fontsize=10)
            ax_x.set_title(f'{mod_label} — X direction', fontsize=11, fontweight='bold')
            ax_x.legend(fontsize=9)
            ax_x.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax_x.set_xlim(-cfg['max_shift_px'], cfg['max_shift_px'])
            ax_x.set_ylim(-0.05, 1.05)
            if mi < len(modalities) - 1:
                ax_x.set_xticklabels([])
            else:
                ax_x.set_xlabel(r'$\Delta x$ [px]')

            # Y-direction (column at center)
            ax_y = axes[mi, 1]
            ax_y.plot(dy, clean_norm[:, center], color=color, linewidth=2,
                     label='Clean', linestyle='-')
            ax_y.plot(dy, bright_norm[:, center], color=color, linewidth=2,
                     label='+50%', linestyle='--', alpha=0.7)
            ax_y.fill_between(dy, clean_norm[:, center], bright_norm[:, center],
                            alpha=0.15, color=color)
            ax_y.set_title(f'{mod_label} — Y direction', fontsize=11, fontweight='bold')
            ax_y.legend(fontsize=9)
            ax_y.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax_y.set_xlim(-cfg['max_shift_px'], cfg['max_shift_px'])
            ax_y.set_ylim(-0.05, 1.05)
            if mi < len(modalities) - 1:
                ax_y.set_xticklabels([])
            else:
                ax_y.set_xlabel(r'$\Delta y$ [px]')

        fig.suptitle(f'1D Cross-Section: Clean vs +50% Brightness — Frame {frame_idx}\n'
                    f'(Shaded area = shape difference; ideal = no shading)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        save_path = os.path.join(output_dir, f'partD_cross_section_frame{frame_idx}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {save_path}")

        del rgb_tensor, rgb_bright_tensor
        torch.cuda.empty_cache()


# ============================================================
# PART E: Channel Type Classification Summary
# ============================================================

def classify_and_summarize_channels(output_dir: str, device: str):
    """
    Classify all 64 conv1 channels by filter type and cross-reference
    with ablation importance results.
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
            classifications.append({'ch': ch, 'type': ch_type, 'gamma': gamma,
                                   'detail': f'gamma={gamma:.4f}'})
            continue

        # Analyze filter structure
        r, g, b = w[0], w[1], w[2]

        # Color opponent: strong opposing signs between R and G (or B)
        rg_opp = np.mean(r) * np.mean(g)
        rb_opp = np.mean(r) * np.mean(b)
        gb_opp = np.mean(g) * np.mean(b)

        # Edge detection: strong spatial gradient
        # Compute gradient magnitude of the average across RGB
        avg_filter = (r + g + b) / 3.0
        gx = np.abs(avg_filter[:, 1:] - avg_filter[:, :-1]).mean()
        gy = np.abs(avg_filter[1:, :] - avg_filter[:-1, :]).mean()
        grad_mag = (gx + gy) / 2.0

        # Luminance: all channels have same sign
        r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
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
            # Strong color opponent
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
            'ch': ch, 'type': ch_type, 'gamma': gamma,
            'detail': f'gx={gx:.4f} gy={gy:.4f} rg={rg_opp:.4f}'
        })

    # ── Create summary figure ──
    fig, ax = plt.subplots(figsize=(18, 8))

    # Color map for types
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

    # Bar chart: one bar per channel, colored by type
    x_pos = np.arange(64)
    bar_colors = [type_colors.get(c['type'], '#bdc3c7') for c in classifications]
    gammas = [abs(c['gamma']) for c in classifications]
    bars = ax.bar(x_pos, gammas, color=bar_colors, edgecolor='white', linewidth=0.3)

    # Mark channels that appear in our combinations
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
                   fontsize=8, fontweight='bold')

        # Highlight core channels
        imp1 = RANK01_IMPORTANCE.get(ch, None)
        imp2 = RANK02_IMPORTANCE.get(ch, None)
        if imp1 == 'core' or imp2 == 'core':
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2.5)

    ax.set_xlabel('Channel Index', fontsize=12)
    ax.set_ylabel('|BatchNorm γ|', fontsize=12)
    ax.set_title('Conv1 Channel Classification & Ablation Importance\n'
                '(*R1 = Rank01, *R2 = Rank02, ** = Both, Red border = Core channel)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in range(64)], fontsize=6, rotation=90)
    ax.axhline(y=0.01, color='red', linestyle=':', alpha=0.5, label='Dead threshold')

    # Legend for types
    legend_patches = [mpatches.Patch(color=c, label=t) for t, c in type_colors.items()]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=2)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'partE_channel_classification.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")

    # Save classification data as JSON (convert numpy types to native Python)
    json_path = os.path.join(output_dir, 'channel_classifications.json')
    classifications_clean = []
    for c in classifications:
        classifications_clean.append({
            k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else
                int(v) if isinstance(v, (np.integer, np.int32, np.int64)) else v)
            for k, v in c.items()
        })
    with open(json_path, 'w') as f:
        json.dump(classifications_clean, f, indent=2)
    print(f"    Saved: {json_path}")


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
    print(f"  Comprehensive Ablation Visualization")
    print(f"{'='*70}")
    print(f"  Images found:     {n_images}")
    print(f"  Test frames:      {frame_indices}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Parts:            A (Filters+Features), B (RGB), C (Basin Grid),")
    print(f"                    D (1D Cross-Sections), E (Classification)")
    print(f"{'='*70}")

    device = cfg['device']

    # ── Part A: Conv1 Filters & Feature Maps ──
    visualize_conv1_filters(output_dir, device)
    visualize_feature_maps(all_images, frame_indices, output_dir, device)

    # ── Part B: Original RGB Images ──
    visualize_rgb_images(all_images, frame_indices, output_dir)

    # ── Part C: Convergence Basin Comparison Grid ──
    visualize_basin_comparison(all_images, frame_indices, output_dir, device, cfg)

    # ── Part D: 1D Cross-Section Overlays ──
    visualize_1d_cross_sections(all_images, frame_indices, output_dir, device, cfg)

    # ── Part E: Channel Classification Summary ──
    classify_and_summarize_channels(output_dir, device)

    print(f"\n{'='*70}")
    print(f"  ALL VISUALIZATIONS COMPLETE")
    print(f"  Output: {output_dir}/")
    print(f"{'='*70}")