"""
Convergence Basin Visualization — Translation-Based (X/Y Pixel Shift)
=====================================================================
CNN Feature Injection into COMO — Photometric Cost Landscape Analysis

Experiment Design (following supervisor guidance):
  - Take a single reference image, extract features (Gray / CNN / CNN+RGB).
  - Apply systematic 2D translation perturbations of ±30 pixels in X and Y.
  - Compute MSE photometric cost between the shifted features and the original.
  - Ground truth = identity (zero shift) → minimum cost at center (0,0).

  This directly corresponds to COMO's dense photometric tracking: if the pose
  estimate has a small error, the reprojected model image is shifted relative
  to the live image. The cost landscape reveals how strongly each feature
  representation guides the Gauss-Newton optimizer back to the correct alignment.

  Additionally, we test robustness by applying brightness perturbation or
  Gaussian noise to the shifted image before computing cost, simulating
  real-world appearance changes between keyframe and live frame.

Conditions:
  A. Clean (no perturbation) — baseline cost landscape shape
  B. Brightness change (+30% intensity) — tests illumination robustness
  C. Gaussian noise (σ=0.05) — tests noise robustness

Outputs:
  1. 3D surface plots with contour projection on the floor (publication style)
  2. 2D heatmaps with contour lines
  3. Quantitative metrics: raw steepness, local curvature, basin width

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
    'max_shift_px': 30,        # ±30 pixels
    'grid_size': 61,           # 61x61 = 3721 evaluations per modality

    # CNN config
    'cnn_channels': 8,

    # Robustness perturbations
    'brightness_factor': 0.5,  # +30% brightness
    'noise_sigma': 0.05,       # Gaussian noise σ
}

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

CONDITION_CONFIG = {
    'clean': {'label': 'Clean', 'suffix': '_clean'},
    'bright': {'label': 'Brightness +30%', 'suffix': '_bright'},
    'noise': {'label': 'Noise σ=0.05', 'suffix': '_noise'},
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

    Args:
        image: [H, W] or [H, W, C] numpy array
        dx: horizontal shift (positive = right)
        dy: vertical shift (positive = down)

    Returns:
        Shifted image with same shape as input.
    """
    h, w = image.shape[:2]
    orig_shape = image.shape

    # Affine matrix for pure translation
    M = np.float64([[1, 0, dx],
                     [0, 1, dy]])

    warped = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # cv2.warpAffine may squeeze single-channel [H,W,1] -> [H,W]
    if warped.shape != orig_shape:
        warped = warped.reshape(orig_shape)

    return warped


def apply_brightness_perturbation(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Apply additive brightness change to an image.

    Args:
        image: [H, W, C] float array in [0, 1]
        factor: brightness change (e.g., 0.3 = +30%)

    Returns:
        Perturbed image clipped to [0, 1].
    """
    return np.clip(image + factor, 0.0, 1.0)


def apply_noise_perturbation(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian noise to an image.

    Args:
        image: [H, W, C] float array in [0, 1]
        sigma: noise standard deviation

    Returns:
        Noisy image clipped to [0, 1].
    """
    noise = np.random.RandomState(42).randn(*image.shape).astype(np.float32) * sigma
    return np.clip(image + noise, 0.0, 1.0)


def load_image_tensor(image_path: str, device: str = "cuda:0") -> torch.Tensor:
    """Load an image as a normalized [0,1] tensor of shape [1, 3, H, W]."""
    img = Image.open(image_path).convert('RGB')
    tensor = ToTensor()(img).unsqueeze(0).to(device)
    return tensor


def load_image_numpy(image_path: str) -> np.ndarray:
    """Load an image as a float32 numpy array [H, W, 3] in [0, 1]."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def rgb_to_gray_np(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB numpy array to grayscale [H, W, 1]."""
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return gray[:, :, np.newaxis]


def numpy_to_tensor(image_np: np.ndarray, device: str = "cuda:0") -> torch.Tensor:
    """
    Convert numpy [H, W, 3] float32 image to tensor [1, 3, H, W].
    """
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
        elif mode == 'cnn_only':
            feat = extractor(rgb_tensor)
        elif mode == 'cnn_rgb':
            feat = extractor(rgb_tensor)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # [1, C, H, W] -> [H, W, C]
    return feat[0].permute(1, 2, 0).cpu().numpy()


def extract_features_from_numpy_image(
    rgb_np: np.ndarray,
    extractor_cnn_only,
    extractor_cnn_rgb,
    device: str = "cuda:0"
) -> Dict[str, np.ndarray]:
    """
    Extract all three feature modalities from a numpy RGB image.

    Returns dict: {'gray': [H,W,1], 'cnn_only': [H,W,8], 'cnn_rgb': [H,W,11]}
    """
    rgb_tensor = numpy_to_tensor(rgb_np, device)

    feats = {}
    feats['gray'] = extract_features_numpy(rgb_tensor, None, 'gray')
    feats['cnn_only'] = extract_features_numpy(rgb_tensor, extractor_cnn_only, 'cnn_only')
    feats['cnn_rgb'] = extract_features_numpy(rgb_tensor, extractor_cnn_rgb, 'cnn_rgb')

    del rgb_tensor
    return feats


def compute_2d_cost_landscape_translation(
    feat_ref: np.ndarray,
    feat_target: np.ndarray,
    max_shift: float,
    grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute photometric cost over a 2D grid of (dx, dy) translation perturbations.

    For each grid point (dx, dy):
      - Shift feat_target by (dx, dy) pixels
      - Compute MSE against feat_ref

    When feat_target == feat_ref (clean condition), ground truth is at (0,0).
    When feat_target has brightness/noise perturbation, the minimum may still
    be at (0,0) but the basin shape changes.

    Args:
        feat_ref: Reference features [H, W, C] — the "model" image
        feat_target: Target features [H, W, C] — the "live" image (may be perturbed)
        max_shift: Maximum shift in pixels (±max_shift)
        grid_size: Number of grid points per axis

    Returns:
        dx_values: 1D array of X shifts (pixels)
        dy_values: 1D array of Y shifts (pixels)
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
# Visualization Functions
# ============================================================

def plot_3d_surface_publication(
    cost_data: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    title: str,
    output_path: str,
    colormap: str = 'GnBu_r'
):
    """
    Plot a single 3D surface with contour projection on the floor.
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    DX, DY = np.meshgrid(dx, dy, indexing='ij')

    # Normalize cost to [0, 1] for visualization
    c_min, c_max = cost_data.min(), cost_data.max()
    if c_max - c_min > 1e-10:
        cost_norm = (cost_data - c_min) / (c_max - c_min)
    else:
        cost_norm = np.zeros_like(cost_data)

    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=1)
    facecolors = cmap(norm(cost_norm))

    surf = ax.plot_surface(
        DY, DX, cost_norm,
        facecolors=facecolors,
        edgecolor='k',
        linewidth=0.15,
        alpha=0.92,
        shade=True,
        rcount=40, ccount=40,
        antialiased=True
    )

    # Floor contour
    contour_offset = -0.05
    ax.contourf(
        DY, DX, cost_norm,
        zdir='z', offset=contour_offset,
        levels=20, cmap='gray_r', alpha=0.7
    )
    ax.contour(
        DY, DX, cost_norm,
        zdir='z', offset=contour_offset,
        levels=10, colors='k', linewidths=0.5, alpha=0.5
    )

    ax.set_xlabel('Δx [px]', labelpad=10)
    ax.set_ylabel('Δy [px]', labelpad=10)
    ax.set_zlabel('Norm. Cost', labelpad=8)
    ax.set_title(title, fontweight='bold', pad=15, fontsize=14)

    ax.set_zlim(contour_offset, 1.05)
    ax.view_init(elev=32, azim=-50)
    ax.tick_params(labelsize=9)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

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
    suffix: str = ""
):
    """
    Plot side-by-side 3D surfaces for all modalities (1 row, N columns).
    """
    modalities = list(costs.keys())
    n_mod = len(modalities)

    fig = plt.figure(figsize=(7 * n_mod, 7))
    DX, DY = np.meshgrid(dx, dy, indexing='ij')

    colormaps = {
        'gray': 'GnBu_r',
        'cnn_only': 'YlOrRd',
        'cnn_rgb': 'YlOrRd',
    }

    for idx, mod in enumerate(modalities):
        ax = fig.add_subplot(1, n_mod, idx + 1, projection='3d')
        cost_data = costs[mod]

        c_min, c_max = cost_data.min(), cost_data.max()
        if c_max - c_min > 1e-10:
            cost_norm = (cost_data - c_min) / (c_max - c_min)
        else:
            cost_norm = np.zeros_like(cost_data)

        cmap_name = colormaps.get(mod, 'viridis')
        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=0, vmax=1)
        facecolors = cmap(norm(cost_norm))

        ax.plot_surface(
            DY, DX, cost_norm,
            facecolors=facecolors,
            edgecolor='k',
            linewidth=0.12,
            alpha=0.92,
            shade=True,
            rcount=40, ccount=40,
            antialiased=True
        )

        contour_offset = -0.05
        ax.contourf(
            DY, DX, cost_norm,
            zdir='z', offset=contour_offset,
            levels=20, cmap='gray_r', alpha=0.7
        )
        ax.contour(
            DY, DX, cost_norm,
            zdir='z', offset=contour_offset,
            levels=10, colors='k', linewidths=0.4, alpha=0.5
        )

        ax.set_xlabel('Δx [px]', labelpad=8)
        ax.set_ylabel('Δy [px]', labelpad=8)
        ax.set_zlabel('Norm. Cost', labelpad=6)
        ax.set_title(MODALITY_CONFIG[mod]['label'], fontweight='bold', fontsize=13, pad=12)
        ax.set_zlim(contour_offset, 1.05)
        ax.view_init(elev=32, azim=-50)
        ax.tick_params(labelsize=8)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')

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
    suffix: str = ""
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
            interpolation='bilinear',
            vmin=0, vmax=1
        )

        DX_grid, DY_grid = np.meshgrid(dx, dy, indexing='ij')
        axes[idx].contour(
            DX_grid, DY_grid, cost_norm,
            levels=10, colors='white', linewidths=0.5, alpha=0.6
        )

        axes[idx].set_xlabel("Δx [px]")
        axes[idx].set_ylabel("Δy [px]")
        axes[idx].set_title(MODALITY_CONFIG[mod]['label'], fontweight='bold')
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
    step = (2 * max_shift) / (len(dx) - 1)  # grid step in pixels

    print(f"\n    {'='*80}")
    print(f"    Condition: {condition_label}")
    print(f"    {'Modality':<18} {'Raw Steep.':<12} {'Local Curv.':<14} "
          f"{'Basin W. (50%)':<16} {'Max Cost'}")
    print(f"    {'-'*80}")

    results = {}
    for mod, cost_data in costs.items():
        label = MODALITY_CONFIG[mod]['label']

        # --- Raw Steepness ---
        center_cost = cost_data[center_idx, center_idx]
        edge_costs = [
            cost_data[0, center_idx], cost_data[-1, center_idx],
            cost_data[center_idx, 0], cost_data[center_idx, -1]
        ]
        raw_steepness = (np.mean(edge_costs) - center_cost) / max_shift

        # --- Local Curvature at center ---
        curv_dy = (cost_data[center_idx - 1, center_idx] +
                   cost_data[center_idx + 1, center_idx] -
                   2 * center_cost) / (step ** 2)
        curv_dx = (cost_data[center_idx, center_idx - 1] +
                   cost_data[center_idx, center_idx + 1] -
                   2 * center_cost) / (step ** 2)
        local_curvature = (curv_dx + curv_dy) / 2.0

        # --- Basin Width at 50% of max cost ---
        c_max = cost_data.max()
        if c_max > 1e-10:
            cost_row = cost_data[center_idx, :]
            basin_mask = cost_row < (0.5 * c_max)
            basin_width_px = np.sum(basin_mask) * step
        else:
            basin_width_px = 2 * max_shift

        max_cost = cost_data.max()

        print(f"    {label:<18} {raw_steepness:<12.6f} {local_curvature:<14.6f} "
              f"{basin_width_px:<16.2f} {max_cost:.6f}")

        results[mod] = {
            'raw_steepness': raw_steepness,
            'local_curvature': local_curvature,
            'basin_width': basin_width_px,
            'max_cost': max_cost,
        }

    print(f"    {'-'*80}")

    # --- Print ratios ---
    if 'gray' in results and 'cnn_only' in results:
        gray = results['gray']
        cnn = results['cnn_only']
        steep_r = cnn['raw_steepness'] / gray['raw_steepness'] if gray['raw_steepness'] > 0 else float('inf')
        curv_r = cnn['local_curvature'] / gray['local_curvature'] if gray['local_curvature'] > 0 else float('inf')
        maxc_r = cnn['max_cost'] / gray['max_cost'] if gray['max_cost'] > 0 else float('inf')
        bw_r = cnn['basin_width'] / gray['basin_width'] if gray['basin_width'] > 0 else float('inf')
        print(f"    CNN/Gray Ratios:  Steep={steep_r:.2f}×  Curv={curv_r:.2f}×  "
              f"MaxCost={maxc_r:.2f}×  BasinW={bw_r:.2f}×")

    if 'gray' in results and 'cnn_rgb' in results:
        gray = results['gray']
        hybrid = results['cnn_rgb']
        steep_r = hybrid['raw_steepness'] / gray['raw_steepness'] if gray['raw_steepness'] > 0 else float('inf')
        curv_r = hybrid['local_curvature'] / gray['local_curvature'] if gray['local_curvature'] > 0 else float('inf')
        maxc_r = hybrid['max_cost'] / gray['max_cost'] if gray['max_cost'] > 0 else float('inf')
        bw_r = hybrid['basin_width'] / gray['basin_width'] if gray['basin_width'] > 0 else float('inf')
        print(f"    RGB/Gray Ratios:  Steep={steep_r:.2f}×  Curv={curv_r:.2f}×  "
              f"MaxCost={maxc_r:.2f}×  BasinW={bw_r:.2f}×")

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

    print(f"{'='*60}")
    print(f"  Convergence Basin — Translation (X/Y Pixel Shift)")
    print(f"  (Shift features and measure photometric cost)")
    print(f"{'='*60}")
    print(f"  Images found:     {n_images}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Grid size:        {cfg['grid_size']}x{cfg['grid_size']} = {cfg['grid_size']**2} evaluations/modality")
    print(f"  Shift range:      ±{cfg['max_shift_px']} pixels")
    print(f"  Camera intrinsics: fx={cfg['fx']}, fy={cfg['fy']}, cx={cfg['cx']}, cy={cfg['cy']}")
    print(f"  Conditions:       Clean, Brightness +{cfg['brightness_factor']*100:.0f}%, Noise σ={cfg['noise_sigma']}")
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

    # Define conditions
    conditions = ['clean', 'bright', 'noise']

    for frame_idx_i, (ref_idx, frame_label) in enumerate(frame_configs):
        if ref_idx >= n_images:
            print(f"\n  [!] Skipping frame {frame_idx_i + 1}: index out of range")
            continue

        print(f"\n  [{frame_idx_i + 1}/{len(frame_configs)}] {frame_label}")
        print(f"  {'─' * 50}")

        # Load reference image
        rgb_np = load_image_numpy(all_images[ref_idx])
        rgb_tensor = load_image_tensor(all_images[ref_idx], device)

        # Extract reference features (from clean image)
        print(f"    Extracting reference features...")
        feat_ref = {}
        feat_ref['gray'] = extract_features_numpy(rgb_tensor, None, 'gray')
        feat_ref['cnn_only'] = extract_features_numpy(rgb_tensor, extractor_cnn_only, 'cnn_only')
        feat_ref['cnn_rgb'] = extract_features_numpy(rgb_tensor, extractor_cnn_rgb, 'cnn_rgb')

        for cond in conditions:
            cond_label = CONDITION_CONFIG[cond]['label']
            cond_suffix = CONDITION_CONFIG[cond]['suffix']
            suffix = f"_frame{frame_idx_i + 1}{cond_suffix}"

            print(f"\n    --- Condition: {cond_label} ---")

            # Prepare target image (with perturbation applied BEFORE feature extraction)
            if cond == 'clean':
                rgb_target_np = rgb_np.copy()
            elif cond == 'bright':
                rgb_target_np = apply_brightness_perturbation(rgb_np, cfg['brightness_factor'])
            elif cond == 'noise':
                rgb_target_np = apply_noise_perturbation(rgb_np, cfg['noise_sigma'])

            # Extract target features from (possibly perturbed) image
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

            # Generate visualizations
            print(f"    Generating plots...")

            # 1. Individual 3D surfaces
            for mod, cost in modalities_data.items():
                mod_label = MODALITY_CONFIG[mod]['label']
                cmap = 'GnBu_r' if mod == 'gray' else 'YlOrRd'
                plot_3d_surface_publication(
                    cost, dx, dy,
                    title=f"{mod_label} — {frame_label} — {cond_label}",
                    output_path=os.path.join(output_dir, f"basin_3d_{mod}{suffix}.png"),
                    colormap=cmap
                )

            # 2. Side-by-side 3D comparison
            plot_3d_comparison(
                modalities_data, dx, dy, frame_label, cond_label,
                output_dir, suffix
            )

            # 3. 2D heatmaps
            plot_2d_heatmaps(
                modalities_data, dx, dy, frame_label, cond_label,
                output_dir, suffix
            )

            # 4. Quantitative metrics
            print_basin_metrics(
                modalities_data, dx, dy,
                cfg['max_shift_px'], cond_label
            )

        # Cleanup
        del rgb_tensor, rgb_np
        del feat_ref
        torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Convergence Basin — Translation")
    print(f"  Output files in: {output_dir}/")
    print(f"    basin_3d_<modality>_frame*_<cond>.png  — Individual 3D surfaces")
    print(f"    convergence_basin_3d_translation_*.png — Side-by-side comparison")
    print(f"    convergence_basin_2d_translation_*.png — 2D heatmaps")
    print(f"  Conditions: Clean, Brightness, Noise")
    print(f"{'='*60}")