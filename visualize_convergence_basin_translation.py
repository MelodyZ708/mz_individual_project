"""
Convergence Basin Visualization — Translation-Based (X/Y Pixel Shift)
=====================================================================
CNN Feature Injection into COMO — Photometric Cost Landscape Analysis

Experiment Design (following supervisor guidance):
  - Take a single reference image, extract features (Gray / CNN / CNN+RGB).
  - Apply systematic 2D translation perturbations of +/-30 pixels in X and Y.
  - Compute MSE photometric cost between the shifted features and the original.
  - Ground truth = identity (zero shift) -> minimum cost at center (0,0).

  This directly corresponds to COMO's dense photometric tracking: if the pose
  estimate has a small error, the reprojected model image is shifted relative
  to the live image. The cost landscape reveals how strongly each feature
  representation guides the Gauss-Newton optimizer back to the correct alignment.

  Additionally, we test robustness by applying progressive brightness
  perturbation (+30%, +50%) to the target image before feature extraction,
  simulating real-world illumination changes between keyframe and live frame.

Conditions (all run in a single execution):
  A. Clean (no perturbation) — baseline cost landscape shape
  B. Brightness +30% — moderate illumination change
  C. Brightness +50% — extreme illumination change

Outputs:
  1. 3D surface plots with contour projection on the floor (publication style)
  2. 2D heatmaps with contour lines
  3. Quantitative metrics: raw steepness, local curvature, basin width
  4. Sharpness metrics: X-Sharpness and Y-Sharpness (mean absolute gradient
     within +/-5 px of the minimum), printed to console and annotated on plots
  5. Cross-condition sharpness degradation summary (Clean -> +30% -> +50%)

Reference: Czarnowski et al., "Semantic Texture for Robust Dense Tracking",
ICCV 2017 Workshop — convergence basin analysis for CNN vs RGB features.

Usage:
  cd /vol/bitbucket/mz325/individual_project
  python visualize_convergence_basin_translation.py

Author: mz325
Date: 2026-05
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import glob
import cv2
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

sys.path.append('/vol/bitbucket/mz325/individual_project')
from como.utils.image_processing import CNNFeatureExtractor

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/convergence_basin_translation',
    'device': 'cuda:0',

    # Camera intrinsics — TUM freiburg1 (Kinect v1, 640x480)
    'fx': 517.3,
    'fy': 516.5,
    'cx': 318.6,
    'cy': 255.3,

    # Perturbation range (pixels)
    'max_shift_px': 30,        # +/-30 pixels
    'grid_size': 61,           # 61x61 = 3721 evaluations per modality

    # CNN config
    'cnn_channels': 8,

    # Sharpness config
    'sharpness_radius': 5,     # +/-5 grid points around the minimum
}

# All brightness conditions to test (run sequentially in one execution)
BRIGHTNESS_CONDITIONS = [
    {'key': 'clean',     'factor': 0.0,  'label': 'Clean',             'suffix': '_clean'},
    {'key': 'bright30',  'factor': 0.3,  'label': 'Brightness +30%',   'suffix': '_bright30'},
    {'key': 'bright50',  'factor': 0.5,  'label': 'Brightness +50%',   'suffix': '_bright50'},
]

# Plot styling — publication quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 250,
    'mathtext.fontset': 'cm',
})

MODALITY_CONFIG = {
    'gray': {'label': 'Gray (1ch)', 'color': '#1f77b4'},
    'cnn_only': {'label': 'CNN-only (8ch)', 'color': '#d62728'},
    'cnn_rgb': {'label': 'CNN+RGB (11ch)', 'color': '#ff7f0e'},
}


# ============================================================
# Core Functions
# ============================================================

def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Shift an image by (dx, dy) pixels using affine transformation.

    Uses sub-pixel interpolation for smooth cost landscapes.
    Border pixels are replicated (BORDER_REPLICATE) to avoid
    artificial cost from black borders.
    """
    h, w = image.shape[:2]
    orig_shape = image.shape

    M = np.float64([[1, 0, dx],
                     [0, 1, dy]])

    warped = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    if warped.shape != orig_shape:
        warped = warped.reshape(orig_shape)

    return warped


def apply_brightness_perturbation(image: np.ndarray, factor: float) -> np.ndarray:
    """Apply additive brightness change. factor=0.3 means +30%."""
    if factor == 0.0:
        return image.copy()
    return np.clip(image + factor, 0.0, 1.0)


def load_image_tensor(image_path: str, device: str = "cuda:0") -> torch.Tensor:
    """Load an image as a normalized [0,1] tensor of shape [1, 3, H, W]."""
    img = Image.open(image_path).convert('RGB')
    tensor = ToTensor()(img).unsqueeze(0).to(device)
    return tensor


def load_image_numpy(image_path: str) -> np.ndarray:
    """Load an image as a float32 numpy array [H, W, 3] in [0, 1]."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_tensor(image_np: np.ndarray, device: str = "cuda:0") -> torch.Tensor:
    """Convert numpy [H, W, 3] float32 image to tensor [1, 3, H, W]."""
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def compute_photometric_cost(img_ref: np.ndarray, img_warped: np.ndarray) -> float:
    """Compute per-pixel MSE photometric cost."""
    residual = img_warped.astype(np.float64) - img_ref.astype(np.float64)
    return np.mean(residual ** 2)


def extract_features_numpy(rgb_tensor: torch.Tensor, extractor, mode: str) -> np.ndarray:
    """
    Extract features and convert to numpy [H, W, C].
    mode: 'gray', 'cnn_only', 'cnn_rgb'
    """
    device = rgb_tensor.device
    with torch.no_grad():
        if mode == 'gray':
            weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
            feat = (rgb_tensor * weights).sum(dim=1, keepdim=True)
        elif mode in ('cnn_only', 'cnn_rgb'):
            feat = extractor(rgb_tensor)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    return feat[0].permute(1, 2, 0).cpu().numpy()


def compute_2d_cost_landscape_translation(
    feat_ref: np.ndarray,
    feat_target: np.ndarray,
    max_shift: float,
    grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute photometric cost over a 2D grid of (dx, dy) translation perturbations.

    Returns:
        dx_values, dy_values: 1D arrays of shifts (pixels)
        cost_grid: 2D array [grid_size, grid_size]
    """
    dx_values = np.linspace(-max_shift, max_shift, grid_size)
    dy_values = np.linspace(-max_shift, max_shift, grid_size)
    cost_grid = np.zeros((grid_size, grid_size))

    for i, dy in enumerate(dy_values):
        for j, dx in enumerate(dx_values):
            shifted = shift_image(feat_target, dx, dy)
            cost_grid[i, j] = compute_photometric_cost(feat_ref, shifted)

    return dx_values, dy_values, cost_grid


# ============================================================
# Sharpness Computation
# ============================================================

def compute_sharpness(
    cost_grid: np.ndarray,
    dx_values: np.ndarray,
    dy_values: np.ndarray,
    radius_px: int = 5
) -> Dict[str, float]:
    """
    Compute directional sharpness of the convergence basin.

    Sharpness is computed on the **normalized** cost surface (min-max scaled
    to [0, 1]) so that it reflects the *shape* of the basin rather than the
    absolute cost magnitude.  This is critical because different modalities
    have very different raw cost scales (e.g. Gray max ~0.088 vs CNN max
    ~0.039), yet the optimizer sees the normalized gradient direction.

    Two sharpness scales are computed (both on normalized [0,1] cost):
      - Local  (+/-radius_px around center): precision convergence near optimum
      - Global (full +/-max_shift range):    overall basin guidance strength

    Metric: mean absolute gradient (first-order finite difference).
    Higher value = steeper basin = stronger optimizer guidance.

    Returns dict with:
        x_local, y_local, local       (normalized, +/-radius_px)
        x_global, y_global, global_    (normalized, full range)
        x_raw, y_raw                   (raw cost, +/-radius_px, for ref)
        min_location, min_cost
    """
    grid_size = cost_grid.shape[0]
    center = grid_size // 2
    step_x = dx_values[1] - dx_values[0]
    step_y = dy_values[1] - dy_values[0]

    # Find actual minimum location
    min_flat = np.argmin(cost_grid)
    min_iy, min_ix = np.unravel_index(min_flat, cost_grid.shape)
    min_dx = dx_values[min_ix]
    min_dy = dy_values[min_iy]
    min_cost = cost_grid[min_iy, min_ix]

    # ── Normalize cost to [0, 1] ──
    c_min, c_max = cost_grid.min(), cost_grid.max()
    if c_max - c_min > 1e-10:
        cost_norm = (cost_grid - c_min) / (c_max - c_min)
    else:
        cost_norm = np.zeros_like(cost_grid)

    # ── Helper: mean |gradient| over a slice [lo, hi] ──
    def _slice_sharpness(arr: np.ndarray, step: float,
                         center_idx: int, r: int) -> float:
        lo = max(center_idx - r, 0)
        hi = min(center_idx + r, len(arr) - 1)
        segment = arr[lo:hi+1]
        if len(segment) < 2:
            return 0.0
        grad = np.abs(segment[1:] - segment[:-1]) / step
        return float(np.mean(grad))

    full_r = grid_size // 2  # full range = entire slice

    # ── Local sharpness (normalized, +/-radius_px) ──
    x_local = _slice_sharpness(cost_norm[center, :], step_x, center, radius_px)
    y_local = _slice_sharpness(cost_norm[:, center], step_y, center, radius_px)
    local_combined = (x_local + y_local) / 2.0

    # ── Global sharpness (normalized, full range) ──
    x_global = _slice_sharpness(cost_norm[center, :], step_x, center, full_r)
    y_global = _slice_sharpness(cost_norm[:, center], step_y, center, full_r)
    global_combined = (x_global + y_global) / 2.0

    # ── Raw local sharpness (for reference) ──
    x_raw = _slice_sharpness(cost_grid[center, :], step_x, center, radius_px)
    y_raw = _slice_sharpness(cost_grid[:, center], step_y, center, radius_px)

    return {
        'x_local': x_local,
        'y_local': y_local,
        'local': local_combined,
        'x_global': x_global,
        'y_global': y_global,
        'global': global_combined,
        'x_raw': x_raw,
        'y_raw': y_raw,
        'min_location': (min_dx, min_dy),
        'min_cost': min_cost,
    }


def print_sharpness_table(
    sharpness_results: Dict[str, Dict[str, float]],
    condition_label: str,
    radius_px: int = 5
):
    """Print a formatted sharpness comparison table for all modalities."""
    print(f"\n    {'='*115}")
    print(f"    Sharpness Analysis — {condition_label}")
    print(f"    (All on NORMALIZED [0,1] cost.  Local = +/-{radius_px} px near minimum.  Global = full +/-30 px range.)")
    print(f"    {'─'*115}")
    print(f"    {'Modality':<18} {'X-Loc':<9} {'Y-Loc':<9} {'Local':<9} "
          f"{'X-Glo':<9} {'Y-Glo':<9} {'Global':<9} "
          f"{'X-Raw':<10} {'Y-Raw':<10} {'Min Loc':<14} {'MinCost'}")
    print(f"    {'─'*115}")

    for mod, s in sharpness_results.items():
        label = MODALITY_CONFIG[mod]['label']
        loc_str = f"({s['min_location'][0]:+.1f},{s['min_location'][1]:+.1f})"
        print(f"    {label:<18} "
              f"{s['x_local']:<9.4f} {s['y_local']:<9.4f} {s['local']:<9.4f} "
              f"{s['x_global']:<9.4f} {s['y_global']:<9.4f} {s['global']:<9.4f} "
              f"{s['x_raw']:<10.6f} {s['y_raw']:<10.6f} "
              f"{loc_str:<14} {s['min_cost']:.6f}")

    print(f"    {'─'*115}")

    # CNN/Gray ratios
    def _ratio(a, b):
        return a / b if b > 1e-12 else float('inf')

    if 'gray' in sharpness_results and 'cnn_only' in sharpness_results:
        g = sharpness_results['gray']
        c = sharpness_results['cnn_only']
        print(f"    CNN/Gray:  Local={_ratio(c['local'], g['local']):.2f}x  "
              f"Global={_ratio(c['global'], g['global']):.2f}x")

    if 'gray' in sharpness_results and 'cnn_rgb' in sharpness_results:
        g = sharpness_results['gray']
        h = sharpness_results['cnn_rgb']
        print(f"    Hyb/Gray:  Local={_ratio(h['local'], g['local']):.2f}x  "
              f"Global={_ratio(h['global'], g['global']):.2f}x")

    print(f"    {'='*115}")


# ============================================================
# Visualization Functions
# ============================================================

def plot_3d_surface_publication(
    cost_data: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    title: str,
    output_path: str,
    colormap: str = 'GnBu_r',
    sharpness: Optional[Dict[str, float]] = None
):
    """Plot a single 3D surface with contour projection on the floor."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    DX, DY = np.meshgrid(dx, dy, indexing='ij')

    c_min, c_max = cost_data.min(), cost_data.max()
    if c_max - c_min > 1e-10:
        cost_norm = (cost_data - c_min) / (c_max - c_min)
    else:
        cost_norm = np.zeros_like(cost_data)

    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=1)
    facecolors = cmap(norm(cost_norm))

    ax.plot_surface(
        DY, DX, cost_norm,
        facecolors=facecolors,
        edgecolor='k', linewidth=0.15, alpha=0.92,
        shade=True, rcount=40, ccount=40, antialiased=True
    )

    contour_offset = -0.05
    ax.contourf(DY, DX, cost_norm, zdir='z', offset=contour_offset,
                levels=20, cmap='gray_r', alpha=0.7)
    ax.contour(DY, DX, cost_norm, zdir='z', offset=contour_offset,
               levels=10, colors='k', linewidths=0.5, alpha=0.5)

    ax.set_xlabel(r'$\Delta x$ [px]', labelpad=10)
    ax.set_ylabel(r'$\Delta y$ [px]', labelpad=10)
    ax.set_zlabel('Norm. Cost', labelpad=8)

    if sharpness is not None:
        title_full = (f"{title}\n"
                      f"Loc={sharpness['local']:.4f}  "
                      f"Glo={sharpness['global']:.4f}")
        ax.set_title(title_full, fontweight='bold', pad=15, fontsize=12)
    else:
        ax.set_title(title, fontweight='bold', pad=15, fontsize=14)

    ax.set_zlim(contour_offset, 1.05)
    ax.view_init(elev=32, azim=-50)
    ax.tick_params(labelsize=9)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('lightgray')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_3d_comparison(
    costs: Dict[str, np.ndarray],
    dx: np.ndarray,
    dy: np.ndarray,
    frame_label: str,
    condition_label: str,
    output_dir: str,
    suffix: str = "",
    sharpness_all: Optional[Dict[str, Dict[str, float]]] = None
):
    """Plot side-by-side 3D surfaces for all modalities."""
    modalities = list(costs.keys())
    n_mod = len(modalities)

    fig = plt.figure(figsize=(7 * n_mod, 7))
    DX, DY = np.meshgrid(dx, dy, indexing='ij')

    colormaps = {'gray': 'GnBu_r', 'cnn_only': 'YlOrRd', 'cnn_rgb': 'YlOrRd'}

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

        ax.plot_surface(
            DY, DX, cost_norm,
            facecolors=facecolors, edgecolor='k', linewidth=0.12,
            alpha=0.92, shade=True, rcount=40, ccount=40, antialiased=True
        )

        contour_offset = -0.05
        ax.contourf(DY, DX, cost_norm, zdir='z', offset=contour_offset,
                    levels=20, cmap='gray_r', alpha=0.7)
        ax.contour(DY, DX, cost_norm, zdir='z', offset=contour_offset,
                   levels=10, colors='k', linewidths=0.4, alpha=0.5)

        ax.set_xlabel(r'$\Delta x$ [px]', labelpad=8)
        ax.set_ylabel(r'$\Delta y$ [px]', labelpad=8)
        ax.set_zlabel('Norm. Cost', labelpad=6)

        subtitle = MODALITY_CONFIG[mod]['label']
        if sharpness_all is not None and mod in sharpness_all:
            s = sharpness_all[mod]
            subtitle += (f"\nLocal={s['local']:.4f}  "
                         f"Global={s['global']:.4f}")
        ax.set_title(subtitle, fontweight='bold', fontsize=12, pad=12)

        ax.set_zlim(contour_offset, 1.05)
        ax.view_init(elev=32, azim=-50)
        ax.tick_params(labelsize=8)

        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('lightgray')

    fig.suptitle(
        f"Convergence Basin — Translation — {frame_label} — {condition_label}",
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"convergence_basin_3d_translation{suffix}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_2d_heatmaps(
    costs: Dict[str, np.ndarray],
    dx: np.ndarray,
    dy: np.ndarray,
    frame_label: str,
    condition_label: str,
    output_dir: str,
    suffix: str = "",
    sharpness_all: Optional[Dict[str, Dict[str, float]]] = None
):
    """Plot 2D heatmaps with contour lines for all modalities."""
    modalities = list(costs.keys())
    n_mod = len(modalities)

    fig, axes = plt.subplots(1, n_mod, figsize=(6 * n_mod, 5))
    if n_mod == 1:
        axes = [axes]

    extent = [dx[0], dx[-1], dy[-1], dy[0]]

    for idx, mod in enumerate(modalities):
        cost_data = costs[mod]

        c_min, c_max = cost_data.min(), cost_data.max()
        if c_max - c_min > 1e-10:
            cost_norm = (cost_data - c_min) / (c_max - c_min)
        else:
            cost_norm = np.zeros_like(cost_data)

        im = axes[idx].imshow(
            cost_norm, extent=extent,
            cmap='inferno', aspect='equal',
            interpolation='bilinear', vmin=0, vmax=1
        )

        DX_grid, DY_grid = np.meshgrid(dx, dy, indexing='ij')
        axes[idx].contour(
            DX_grid, DY_grid, cost_norm,
            levels=10, colors='white', linewidths=0.5, alpha=0.6
        )

        axes[idx].set_xlabel(r"$\Delta x$ [px]")
        axes[idx].set_ylabel(r"$\Delta y$ [px]")

        subtitle = MODALITY_CONFIG[mod]['label']
        if sharpness_all is not None and mod in sharpness_all:
            s = sharpness_all[mod]
            subtitle += (f"\nLocal={s['local']:.4f}  "
                         f"Global={s['global']:.4f}")
        axes[idx].set_title(subtitle, fontweight='bold', fontsize=11)

        axes[idx].axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
        axes[idx].axvline(x=0, color='white', linestyle='--', alpha=0.3, linewidth=0.8)

        # Mark minimum
        min_idx = np.unravel_index(np.argmin(cost_norm), cost_norm.shape)
        min_dx = dx[min_idx[1]] if cost_norm.shape[1] == len(dx) else 0
        min_dy = dy[min_idx[0]] if cost_norm.shape[0] == len(dy) else 0
        axes[idx].plot(min_dx, min_dy, 'c*', markersize=12,
                       markeredgecolor='white', markeredgewidth=0.8)

        plt.colorbar(im, ax=axes[idx], shrink=0.82, label='Normalized Cost')

    fig.suptitle(
        f"Convergence Basin Heatmap — Translation — {frame_label} — {condition_label}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"convergence_basin_2d_translation{suffix}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def print_basin_metrics(
    costs: Dict[str, np.ndarray],
    dx: np.ndarray,
    dy: np.ndarray,
    max_shift: float,
    condition_label: str
):
    """
    Print quantitative basin metrics for each modality.

    Metrics (all computed on RAW, un-normalized cost):
      1. Raw Steepness: (mean_edge_cost - center_cost) / max_shift
      2. Local Curvature: average 2nd-order finite difference at center
      3. Basin Width (50%): width where cost < 50% of max
      4. Max Cost: absolute maximum cost value
    """
    center_idx = len(dx) // 2
    step = (2 * max_shift) / (len(dx) - 1)

    print(f"\n    {'='*80}")
    print(f"    Basin Metrics — {condition_label}")
    print(f"    {'Modality':<18} {'Raw Steep.':<12} {'Local Curv.':<14} "
          f"{'Basin W. (50%)':<16} {'Max Cost'}")
    print(f"    {'-'*80}")

    results = {}
    for mod, cost_data in costs.items():
        label = MODALITY_CONFIG[mod]['label']

        center_cost = cost_data[center_idx, center_idx]
        edge_costs = [
            cost_data[0, center_idx], cost_data[-1, center_idx],
            cost_data[center_idx, 0], cost_data[center_idx, -1]
        ]
        raw_steepness = (np.mean(edge_costs) - center_cost) / max_shift

        curv_dy = (cost_data[center_idx - 1, center_idx] +
                   cost_data[center_idx + 1, center_idx] -
                   2 * center_cost) / (step ** 2)
        curv_dx = (cost_data[center_idx, center_idx - 1] +
                   cost_data[center_idx, center_idx + 1] -
                   2 * center_cost) / (step ** 2)
        local_curvature = (curv_dx + curv_dy) / 2.0

        c_max = cost_data.max()
        if c_max > 1e-10:
            cost_row = cost_data[center_idx, :]
            basin_mask = cost_row < (0.5 * c_max)
            basin_width_px = np.sum(basin_mask) * step
        else:
            basin_width_px = 2 * max_shift

        print(f"    {label:<18} {raw_steepness:<12.6f} {local_curvature:<14.6f} "
              f"{basin_width_px:<16.2f} {c_max:.6f}")

        results[mod] = {
            'raw_steepness': raw_steepness,
            'local_curvature': local_curvature,
            'basin_width': basin_width_px,
            'max_cost': c_max,
        }

    print(f"    {'-'*80}")

    if 'gray' in results and 'cnn_only' in results:
        g, c = results['gray'], results['cnn_only']
        steep_r = c['raw_steepness'] / g['raw_steepness'] if g['raw_steepness'] > 0 else float('inf')
        curv_r = c['local_curvature'] / g['local_curvature'] if g['local_curvature'] > 0 else float('inf')
        maxc_r = c['max_cost'] / g['max_cost'] if g['max_cost'] > 0 else float('inf')
        bw_r = c['basin_width'] / g['basin_width'] if g['basin_width'] > 0 else float('inf')
        print(f"    CNN/Gray Ratios:  Steep={steep_r:.2f}x  Curv={curv_r:.2f}x  "
              f"MaxCost={maxc_r:.2f}x  BasinW={bw_r:.2f}x")

    if 'gray' in results and 'cnn_rgb' in results:
        g, h = results['gray'], results['cnn_rgb']
        steep_r = h['raw_steepness'] / g['raw_steepness'] if g['raw_steepness'] > 0 else float('inf')
        curv_r = h['local_curvature'] / g['local_curvature'] if g['local_curvature'] > 0 else float('inf')
        maxc_r = h['max_cost'] / g['max_cost'] if g['max_cost'] > 0 else float('inf')
        bw_r = h['basin_width'] / g['basin_width'] if g['basin_width'] > 0 else float('inf')
        print(f"    Hyb/Gray Ratios:  Steep={steep_r:.2f}x  Curv={curv_r:.2f}x  "
              f"MaxCost={maxc_r:.2f}x  BasinW={bw_r:.2f}x")

    print(f"    {'='*80}")
    return results


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

    cond_labels = [c['label'] for c in BRIGHTNESS_CONDITIONS]

    print(f"{'='*60}")
    print(f"  Convergence Basin — Translation (X/Y Pixel Shift)")
    print(f"  One-shot: Clean / +30% / +50% brightness")
    print(f"{'='*60}")
    print(f"  Images found:     {n_images}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Grid size:        {cfg['grid_size']}x{cfg['grid_size']} = {cfg['grid_size']**2} evaluations/modality")
    print(f"  Shift range:      +/-{cfg['max_shift_px']} pixels")
    print(f"  Sharpness radius: +/-{cfg['sharpness_radius']} grid points")
    print(f"  Conditions:       {', '.join(cond_labels)}")
    print(f"  Modalities:       {', '.join(MODALITY_CONFIG[m]['label'] for m in MODALITY_CONFIG)}")
    n_total = len(BRIGHTNESS_CONDITIONS) * 3 * 3  # conditions x frames x modalities
    est_minutes = n_total * 10 / 60  # rough: ~10s per cost landscape
    print(f"  Estimated runs:   {n_total} cost landscapes")
    print(f"{'='*60}")

    # Select 3 representative frames (early / middle / late)
    frame_configs = [
        (50, "Early (frame 50)"),
        (n_images // 2, f"Middle (frame {n_images // 2})"),
        (n_images - 50, f"Late (frame {n_images - 50})"),
    ]

    # Initialize CNN feature extractors
    device = cfg['device']

    extractor_cnn_only = CNNFeatureExtractor(
        target_channels=cfg['cnn_channels'],
        device=device,
        mode="cnn_only"
    )

    extractor_cnn_rgb = CNNFeatureExtractor(
        target_channels=cfg['cnn_channels'],
        device=device,
        mode="rgb_cnn"
    )

    # Collect all sharpness results for cross-condition summary
    all_sharpness_summary = []

    for frame_idx_i, (ref_idx, frame_label) in enumerate(frame_configs):
        if ref_idx >= n_images:
            print(f"\n  [!] Skipping frame {frame_idx_i + 1}: index out of range")
            continue

        print(f"\n  [{frame_idx_i + 1}/{len(frame_configs)}] {frame_label}")
        print(f"  {'─' * 50}")

        # Load reference image
        rgb_np = load_image_numpy(all_images[ref_idx])
        rgb_tensor = load_image_tensor(all_images[ref_idx], device)

        # Extract reference features (always from clean image)
        print(f"    Extracting reference features...")
        feat_ref = {}
        feat_ref['gray'] = extract_features_numpy(rgb_tensor, None, 'gray')
        feat_ref['cnn_only'] = extract_features_numpy(rgb_tensor, extractor_cnn_only, 'cnn_only')
        feat_ref['cnn_rgb'] = extract_features_numpy(rgb_tensor, extractor_cnn_rgb, 'cnn_rgb')

        # ── Loop over all brightness conditions ──
        for cond in BRIGHTNESS_CONDITIONS:
            cond_key = cond['key']
            cond_label = cond['label']
            cond_suffix = cond['suffix']
            bright_factor = cond['factor']
            suffix = f"_frame{frame_idx_i + 1}{cond_suffix}"

            print(f"\n    --- Condition: {cond_label} (factor={bright_factor}) ---")

            # Prepare target image with brightness perturbation
            rgb_target_np = apply_brightness_perturbation(rgb_np, bright_factor)

            # Extract target features
            rgb_target_tensor = numpy_to_tensor(rgb_target_np, device)
            feat_target = {}
            feat_target['gray'] = extract_features_numpy(rgb_target_tensor, None, 'gray')
            feat_target['cnn_only'] = extract_features_numpy(rgb_target_tensor, extractor_cnn_only, 'cnn_only')
            feat_target['cnn_rgb'] = extract_features_numpy(rgb_target_tensor, extractor_cnn_rgb, 'cnn_rgb')
            del rgb_target_tensor

            # Compute cost landscapes
            modalities_data = {}
            for mod in ['gray', 'cnn_only', 'cnn_rgb']:
                mod_label = MODALITY_CONFIG[mod]['label']
                print(f"    Computing cost landscape: {mod_label} ({cfg['grid_size']}x{cfg['grid_size']})...")
                dx, dy, cost = compute_2d_cost_landscape_translation(
                    feat_ref[mod], feat_target[mod],
                    cfg['max_shift_px'], cfg['grid_size']
                )
                modalities_data[mod] = cost

            # ── Compute sharpness ──
            sharpness_results = {}
            for mod, cost in modalities_data.items():
                sharpness_results[mod] = compute_sharpness(
                    cost, dx, dy, radius_px=cfg['sharpness_radius']
                )

            # Print sharpness table
            print_sharpness_table(sharpness_results, cond_label, cfg['sharpness_radius'])

            # Store for cross-condition summary
            for mod, s in sharpness_results.items():
                all_sharpness_summary.append({
                    'frame': frame_label,
                    'condition': cond_label,
                    'modality': MODALITY_CONFIG[mod]['label'],
                    **s,
                })

            # ── Generate visualizations ──
            print(f"    Generating plots...")

            # 1. Individual 3D surfaces
            for mod, cost in modalities_data.items():
                mod_label = MODALITY_CONFIG[mod]['label']
                cmap = 'GnBu_r' if mod == 'gray' else 'YlOrRd'
                plot_3d_surface_publication(
                    cost, dx, dy,
                    title=f"{mod_label} — {frame_label} — {cond_label}",
                    output_path=os.path.join(output_dir, f"basin_3d_{mod}{suffix}.png"),
                    colormap=cmap,
                    sharpness=sharpness_results[mod]
                )

            # 2. Side-by-side 3D comparison
            plot_3d_comparison(
                modalities_data, dx, dy, frame_label, cond_label,
                output_dir, suffix,
                sharpness_all=sharpness_results
            )

            # 3. 2D heatmaps
            plot_2d_heatmaps(
                modalities_data, dx, dy, frame_label, cond_label,
                output_dir, suffix,
                sharpness_all=sharpness_results
            )

            # 4. Basin metrics
            print_basin_metrics(
                modalities_data, dx, dy,
                cfg['max_shift_px'], cond_label
            )

        # Cleanup per frame
        del rgb_tensor, rgb_np, feat_ref
        torch.cuda.empty_cache()

    # ============================================================
    # Cross-Condition Sharpness Summary
    # ============================================================
    print(f"\n{'='*110}")
    print(f"  SHARPNESS SUMMARY — All Frames x All Conditions")
    print(f"{'='*110}")
    print(f"  {'Frame':<25} {'Condition':<20} {'Modality':<18} "
          f"{'Local':<10} {'Global':<10} {'X-Loc':<9} {'Y-Loc':<9} "
          f"{'X-Glo':<9} {'Y-Glo':<9} {'Min Loc'}")
    print(f"  {'-'*120}")
    for entry in all_sharpness_summary:
        loc_str = f"({entry['min_location'][0]:+.1f}, {entry['min_location'][1]:+.1f})"
        print(f"  {entry['frame']:<25} {entry['condition']:<20} {entry['modality']:<18} "
              f"{entry['local']:<10.4f} {entry['global']:<10.4f} "
              f"{entry['x_local']:<9.4f} {entry['y_local']:<9.4f} "
              f"{entry['x_global']:<9.4f} {entry['y_global']:<9.4f} {loc_str}")
    print(f"  {'-'*120}")

    # ── Progressive degradation: Clean -> +30% -> +50% ──
    print(f"\n{'='*110}")
    print(f"  SHARPNESS DEGRADATION — Progressive Brightness Change")
    print(f"  (Retention % = sharpness under brightness / sharpness under Clean)")
    print(f"{'='*110}")
    grouped = defaultdict(dict)
    for entry in all_sharpness_summary:
        key = (entry['frame'], entry['modality'])
        grouped[key][entry['condition']] = (entry['local'], entry['global'])

    print(f"  {'Frame':<25} {'Modality':<18} {'Metric':<8} "
          f"{'Clean':<10} {'+30%':<10} {'Ret.30%':<9} {'+50%':<10} {'Ret.50%':<9}")
    print(f"  {'-'*120}")

    for (frame, mod), conds in sorted(grouped.items()):
        clean_loc, clean_glo = conds.get('Clean', (0.0, 0.0))
        b30_loc, b30_glo = conds.get('Brightness +30%', (0.0, 0.0))
        b50_loc, b50_glo = conds.get('Brightness +50%', (0.0, 0.0))

        ret30_loc = (b30_loc / clean_loc * 100) if clean_loc > 1e-12 else 0.0
        ret50_loc = (b50_loc / clean_loc * 100) if clean_loc > 1e-12 else 0.0
        ret30_glo = (b30_glo / clean_glo * 100) if clean_glo > 1e-12 else 0.0
        ret50_glo = (b50_glo / clean_glo * 100) if clean_glo > 1e-12 else 0.0

        print(f"  {frame:<25} {mod:<18} {'Local':<8} "
              f"{clean_loc:<10.4f} {b30_loc:<10.4f} {ret30_loc:<8.1f}% {b50_loc:<10.4f} {ret50_loc:<8.1f}%")
        print(f"  {'':<25} {'':<18} {'Global':<8} "
              f"{clean_glo:<10.4f} {b30_glo:<10.4f} {ret30_glo:<8.1f}% {b50_glo:<10.4f} {ret50_glo:<8.1f}%")

    print(f"  {'='*110}")

    # ============================================================
    # Final Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Convergence Basin — Translation — COMPLETE")
    print(f"  Output files in: {output_dir}/")
    print(f"    basin_3d_<mod>_frame*_<cond>.png  — Individual 3D")
    print(f"    convergence_basin_3d_translation_*.png — Side-by-side")
    print(f"    convergence_basin_2d_translation_*.png — 2D heatmaps")
    print(f"  Conditions: {', '.join(cond_labels)}")
    print(f"  Sharpness annotated on all plots + summary above.")
    print(f"{'='*60}")