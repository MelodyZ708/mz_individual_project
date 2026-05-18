"""
Convergence Basin 1x3 Visualization — Per-Combination Figures
=============================================================
CNN Feature Injection into COMO — Forward Greedy Selection Results

For each channel combination, generates a single figure with 1x3 layout:
  [Clean] | [Brightness +30%] | [Brightness +50%]

Uses Frame 306 (middle of TUM freiburg1_desk sequence).
Style matches the original publication-quality 3D surface plots with
contour projection on the floor.

Combinations (9 total):
  1. Gray (1ch)                — baseline
  2. Rank01 Full 8ch           — [6,7,12,15,36,45,58,62]
  3. Rank02 Full 8ch           — [8,22,23,27,28,42,48,60]
  4. [Ch15] (Forward Best *)   — single channel, Rank01 core
  5. [Ch23] (Forward Best *)   — single channel, Rank02 core
  6. [Ch6]                     — positive-response channel
  7. [Ch6, Ch15]               — Retention-optimal pair
  8. [Ch8]                     — extreme positive-response
  9. [Ch23, Ch42]              — CN-correcting pair

Usage:
  python visualize_basin_1x3.py

Author: mz325
Date: 2026-05
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
import os
import sys
import glob
import cv2
from typing import Tuple, List, Dict, Optional

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/basin_1x3',
    'device': 'cuda:0',

    # Test frame: middle of sequence
    'frame_index': 306,

    # Perturbation range (pixels)
    'max_shift_px': 30,
    'grid_size': 61,

    # Sharpness config
    'sharpness_radius': 5,
}

# All brightness conditions
BRIGHTNESS_CONDITIONS = [
    {'key': 'clean',    'factor': 0.0, 'label': 'Clean'},
    {'key': 'bright30', 'factor': 0.3, 'label': 'Brightness +30%'},
    {'key': 'bright50', 'factor': 0.5, 'label': 'Brightness +50%'},
]

# ── Channel combinations to visualize ──
COMBINATIONS = [
    {
        'name': 'Gray (1ch)',
        'channels': None,  # special: grayscale
        'colormap': 'GnBu_r',
        'is_forward_best': False,
        'filename': '01_gray',
    },
    {
        'name': 'Rank01 Full (8ch)',
        'channels': [6, 7, 12, 15, 36, 45, 58, 62],
        'colormap': 'YlOrRd',
        'is_forward_best': False,
        'filename': '02_rank01_full8',
    },
    {
        'name': 'Rank02 Full (8ch)',
        'channels': [8, 22, 23, 27, 28, 42, 48, 60],
        'colormap': 'YlOrRd',
        'is_forward_best': False,
        'filename': '03_rank02_full8',
    },
    {
        'name': '[Ch15] (Forward Best *)',
        'channels': [15],
        'colormap': 'YlOrRd',
        'is_forward_best': True,
        'filename': '04_ch15_forward_best',
    },
    {
        'name': '[Ch23] (Forward Best *)',
        'channels': [23],
        'colormap': 'YlOrRd',
        'is_forward_best': True,
        'filename': '05_ch23_forward_best',
    },
    {
        'name': '[Ch6]',
        'channels': [6],
        'colormap': 'OrRd',
        'is_forward_best': False,
        'filename': '06_ch6',
    },
    {
        'name': '[Ch6, Ch15]',
        'channels': [6, 15],
        'colormap': 'OrRd',
        'is_forward_best': False,
        'filename': '07_ch6_ch15',
    },
    {
        'name': '[Ch8]',
        'channels': [8],
        'colormap': 'PuRd',
        'is_forward_best': False,
        'filename': '08_ch8',
    },
    {
        'name': '[Ch23, Ch42]',
        'channels': [23, 42],
        'colormap': 'BuPu',
        'is_forward_best': False,
        'filename': '09_ch23_ch42',
    },
]

# Plot styling — publication quality
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
# Feature Extractor (Direct Conv1)
# ============================================================
class DirectConv1Extractor(nn.Module):
    """Extract specific channels from ResNet-18 conv1 + bn1 + relu."""

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


def extract_gray(rgb_np: np.ndarray) -> np.ndarray:
    gray = 0.299 * rgb_np[:, :, 0] + 0.587 * rgb_np[:, :, 1] + 0.114 * rgb_np[:, :, 2]
    return gray[:, :, np.newaxis]


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

    return {
        'x_local': x_local, 'y_local': y_local, 'local': local,
        'x_global': x_global, 'y_global': y_global, 'global': global_,
        'condition_number': cn,
    }


# ============================================================
# 3D Surface Plotting (Publication Style, matching original)
# ============================================================

def plot_single_3d_surface(
    ax, cost: np.ndarray, dx: np.ndarray, dy: np.ndarray,
    subtitle: str, sharpness: Dict[str, float],
    colormap: str = 'YlOrRd'
):
    """
    Plot a single 3D surface on a given Axes3D subplot.
    Style: surface with wireframe edges + gray contour projection on floor.
    """
    c_min, c_max = cost.min(), cost.max()
    if c_max - c_min > 1e-10:
        cost_norm = (cost - c_min) / (c_max - c_min)
    else:
        cost_norm = np.zeros_like(cost)

    DX, DY = np.meshgrid(dx, dy, indexing='ij')

    # Surface with facecolors
    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=1)
    facecolors = cmap(norm(cost_norm))

    ax.plot_surface(
        DY, DX, cost_norm,
        facecolors=facecolors,
        edgecolor='k', linewidth=0.12,
        alpha=0.92, shade=True,
        rcount=40, ccount=40, antialiased=True
    )

    # Contour projection on floor
    contour_offset = -0.05
    ax.contourf(DY, DX, cost_norm, zdir='z', offset=contour_offset,
                levels=20, cmap='gray_r', alpha=0.7)
    ax.contour(DY, DX, cost_norm, zdir='z', offset=contour_offset,
               levels=10, colors='k', linewidths=0.4, alpha=0.5)

    # Axes
    ax.set_xlabel(r'$\Delta x$ [px]', labelpad=8)
    ax.set_ylabel(r'$\Delta y$ [px]', labelpad=8)
    ax.set_zlabel('Norm. Cost', labelpad=6)
    ax.set_zlim(contour_offset, 1.05)
    ax.view_init(elev=32, azim=-50)
    ax.tick_params(labelsize=8)

    # Transparent panes
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('lightgray')

    # Subtitle with metrics
    title_str = (f"{subtitle}\n"
                 f"L={sharpness['local']:.4f}  CN={sharpness['condition_number']:.2f}")
    ax.set_title(title_str, fontweight='bold', fontsize=11, pad=10)


def generate_1x3_figure(
    combo: dict,
    rgb_np: np.ndarray,
    device: str,
    cfg: dict,
    output_dir: str
):
    """
    Generate a single 1x3 figure for one channel combination.
    Layout: [Clean] | [+30%] | [+50%]
    """
    combo_name = combo['name']
    channels = combo['channels']
    colormap = combo['colormap']
    is_best = combo['is_forward_best']
    filename = combo['filename']

    print(f"\n  Processing: {combo_name}")

    # Build extractor
    if channels is not None:
        extractor = DirectConv1Extractor(channels, device)
    else:
        extractor = None

    # Extract reference features (from clean image)
    if channels is not None:
        rgb_tensor = numpy_to_tensor(rgb_np, device)
        with torch.no_grad():
            feat_ref = extractor(rgb_tensor)[0].permute(1, 2, 0).cpu().numpy()
        del rgb_tensor
    else:
        feat_ref = extract_gray(rgb_np)

    # Create figure: 1 row x 3 columns
    fig = plt.figure(figsize=(21, 7))

    for ci, cond in enumerate(BRIGHTNESS_CONDITIONS):
        # Prepare target image
        rgb_target = apply_brightness(rgb_np, cond['factor'])

        # Extract target features
        if channels is not None:
            rgb_target_tensor = numpy_to_tensor(rgb_target, device)
            with torch.no_grad():
                feat_target = extractor(rgb_target_tensor)[0].permute(1, 2, 0).cpu().numpy()
            del rgb_target_tensor
        else:
            feat_target = extract_gray(rgb_target)

        # Compute cost landscape
        dx, dy, cost = compute_2d_cost_landscape(
            feat_ref, feat_target,
            cfg['max_shift_px'], cfg['grid_size']
        )

        # Compute sharpness
        sharp = compute_sharpness(cost, dx, dy, cfg['sharpness_radius'])

        # Plot
        ax = fig.add_subplot(1, 3, ci + 1, projection='3d')
        plot_single_3d_surface(
            ax, cost, dx, dy,
            subtitle=cond['label'],
            sharpness=sharp,
            colormap=colormap
        )

        # Print metrics
        print(f"    {cond['label']:>15s}: L={sharp['local']:.4f}  "
              f"G={sharp['global']:.4f}  CN={sharp['condition_number']:.2f}")

    # Suptitle
    suptitle = f"Convergence Basin - Middle (frame 306) - {combo_name}"
    fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"basin_1x3_{filename}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")

    # Cleanup
    if extractor is not None:
        del extractor
    torch.cuda.empty_cache()


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

    print(f"{'='*60}")
    print(f"  Convergence Basin — 1x3 Per-Combination Figures")
    print(f"  Frame: 306 (Middle)")
    print(f"  Conditions: Clean / +30% / +50%")
    print(f"{'='*60}")
    print(f"  Images found:     {n_images}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Grid size:        {cfg['grid_size']}x{cfg['grid_size']}")
    print(f"  Shift range:      +/-{cfg['max_shift_px']} pixels")
    print(f"  Combinations:     {len(COMBINATIONS)}")
    print(f"{'='*60}")

    # Load the reference frame
    frame_idx = cfg['frame_index']
    if frame_idx >= n_images:
        print(f"  [ERROR] Frame index {frame_idx} out of range (total: {n_images})")
        sys.exit(1)

    rgb_np = load_image_numpy(all_images[frame_idx])
    print(f"\n  Loaded frame {frame_idx}: {all_images[frame_idx]}")
    print(f"  Image shape: {rgb_np.shape}")

    # Generate one figure per combination
    for combo in COMBINATIONS:
        generate_1x3_figure(combo, rgb_np, cfg['device'], cfg, output_dir)

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"  ALL DONE — {len(COMBINATIONS)} figures generated")
    print(f"  Output directory: {output_dir}/")
    print(f"{'='*60}")