"""
Convergence Basin Visualization — RGB + Channel 6 vs Gray
==========================================================
Compares the photometric cost landscape of:
  1. Gray (1ch) — baseline
  2. RGB + CNN Channel 6 (4ch) — rgb_cnn mode with direct channel 6 from layer1

Translation-based: applies +/-30 pixel shifts in X/Y, computes MSE cost.

Usage:
  cd ~/code/individual_project/como
  python visualize_basin_rgb_ch6.py

Author: mz325
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
from typing import Tuple, Dict, Optional

sys.path.insert(0, '/home/melody/code/individual_project/como')
sys.path.insert(0, '/home/melody/code/individual_project')
from como.utils.image_processing import CNNFeatureExtractor

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/home/melody/data/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/convergence_basin_rgb_ch6',
    'device': 'cuda:0',

    # Perturbation range (pixels)
    'max_shift_px': 30,
    'grid_size': 61,           # 61x61 grid

    # CNN config
    'cnn_channels': 1,         # single channel (ch6)
    'channel_select': 'd6',    # direct absolute index 6
    'cnn_layer': 'layer1',

    # Sharpness config
    'sharpness_radius': 5,
}

# Brightness conditions
BRIGHTNESS_CONDITIONS = [
    {'key': 'clean',     'factor': 0.0,  'label': 'Clean',             'suffix': '_clean'},
    {'key': 'bright30',  'factor': 0.3,  'label': 'Brightness +30%',   'suffix': '_bright30'},
    {'key': 'bright50',  'factor': 0.5,  'label': 'Brightness +50%',   'suffix': '_bright50'},
]

# Plot styling
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
    'rgb_ch6': {'label': 'RGB+Ch6 (4ch)', 'color': '#ff7f0e'},
}

# ============================================================
# Core Functions
# ============================================================

def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift image by (dx, dy) pixels using affine transformation."""
    h, w = image.shape[:2]
    orig_shape = image.shape
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    if warped.shape != orig_shape:
        warped = warped.reshape(orig_shape)
    return warped


def apply_brightness_perturbation(image: np.ndarray, factor: float) -> np.ndarray:
    """Apply additive brightness change."""
    if factor == 0.0:
        return image.copy()
    return np.clip(image + factor, 0.0, 1.0)


def load_image_numpy(image_path: str) -> np.ndarray:
    """Load image as float32 numpy [H, W, 3] in [0, 1]."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_tensor(image_np: np.ndarray, device: str = "cuda:0") -> torch.Tensor:
    """Convert numpy [H, W, 3] to tensor [1, 3, H, W]."""
    return torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_photometric_cost(img_ref: np.ndarray, img_warped: np.ndarray) -> float:
    """Compute per-pixel MSE."""
    residual = img_warped.astype(np.float64) - img_ref.astype(np.float64)
    return np.mean(residual ** 2)


def extract_features_numpy(rgb_tensor: torch.Tensor, extractor, mode: str) -> np.ndarray:
    """Extract features and convert to numpy [H, W, C]."""
    device = rgb_tensor.device
    with torch.no_grad():
        if mode == 'gray':
            weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
            feat = (rgb_tensor * weights).sum(dim=1, keepdim=True)
        elif mode == 'rgb_ch6':
            feat = extractor(rgb_tensor)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return feat[0].permute(1, 2, 0).cpu().numpy()


def compute_2d_cost_landscape(
    feat_ref: np.ndarray,
    feat_target: np.ndarray,
    max_shift: float,
    grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute photometric cost over a 2D grid of (dx, dy) shifts."""
    dx_values = np.linspace(-max_shift, max_shift, grid_size)
    dy_values = np.linspace(-max_shift, max_shift, grid_size)
    cost_grid = np.zeros((grid_size, grid_size))

    for i, dy in enumerate(dy_values):
        for j, dx in enumerate(dx_values):
            shifted = shift_image(feat_target, dx, dy)
            cost_grid[i, j] = compute_photometric_cost(feat_ref, shifted)

    return dx_values, dy_values, cost_grid


# ============================================================
# Sharpness
# ============================================================

def compute_sharpness(
    cost_grid: np.ndarray,
    dx_values: np.ndarray,
    dy_values: np.ndarray,
    radius_px: int = 5
) -> Dict[str, float]:
    """Compute directional sharpness on normalized [0,1] cost."""
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

    return {
        'x_local': x_local, 'y_local': y_local, 'local': local_combined,
        'x_global': x_global, 'y_global': y_global, 'global': global_combined,
        'min_location': (min_dx, min_dy), 'min_cost': min_cost,
    }


# ============================================================
# Visualization
# ============================================================

def plot_comparison(
    costs: Dict[str, np.ndarray],
    dx: np.ndarray,
    dy: np.ndarray,
    frame_label: str,
    condition_label: str,
    output_dir: str,
    suffix: str = "",
    sharpness_all: Optional[Dict[str, Dict[str, float]]] = None
):
    """Plot side-by-side 3D surfaces and 2D heatmaps."""
    modalities = list(costs.keys())
    n_mod = len(modalities)

    # ── 3D Surface Comparison ──
    fig = plt.figure(figsize=(7 * n_mod, 6))
    for idx, mod in enumerate(modalities):
        ax = fig.add_subplot(1, n_mod, idx + 1, projection='3d')
        cost_data = costs[mod]

        c_min, c_max = cost_data.min(), cost_data.max()
        if c_max - c_min > 1e-10:
            cost_norm = (cost_data - c_min) / (c_max - c_min)
        else:
            cost_norm = np.zeros_like(cost_data)

        DX, DY = np.meshgrid(dx, dy)
        cmap = 'GnBu_r' if mod == 'gray' else 'YlOrRd'
        ax.plot_surface(DX, DY, cost_norm, cmap=cmap,
                        edgecolor='k', linewidth=0.1, alpha=0.9,
                        rcount=40, ccount=40, antialiased=True)

        ax.contourf(DX, DY, cost_norm, zdir='z', offset=-0.05,
                    levels=20, cmap='gray_r', alpha=0.6)

        ax.set_xlabel(r"$\Delta x$ [px]")
        ax.set_ylabel(r"$\Delta y$ [px]")
        ax.set_zlabel("Normalized Cost")
        ax.set_zlim(-0.05, 1.05)

        title = MODALITY_CONFIG[mod]['label']
        if sharpness_all and mod in sharpness_all:
            s = sharpness_all[mod]
            title += f"\nLocal={s['local']:.4f}  Global={s['global']:.4f}"
        ax.set_title(title, fontweight='bold', fontsize=11)

    fig.suptitle(
        f"Convergence Basin — {frame_label} — {condition_label}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"basin_3d{suffix}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")

    # ── 2D Heatmap Comparison ──
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

        im = axes[idx].imshow(cost_norm, extent=extent, cmap='inferno',
                              aspect='equal', interpolation='bilinear', vmin=0, vmax=1)

        DX_grid, DY_grid = np.meshgrid(dx, dy)
        axes[idx].contour(DX_grid, DY_grid, cost_norm,
                          levels=10, colors='white', linewidths=0.5, alpha=0.6)

        axes[idx].set_xlabel(r"$\Delta x$ [px]")
        axes[idx].set_ylabel(r"$\Delta y$ [px]")
        axes[idx].axhline(y=0, color='white', linestyle='--', alpha=0.3)
        axes[idx].axvline(x=0, color='white', linestyle='--', alpha=0.3)

        title = MODALITY_CONFIG[mod]['label']
        if sharpness_all and mod in sharpness_all:
            s = sharpness_all[mod]
            title += f"\nLocal={s['local']:.4f}  Global={s['global']:.4f}"
        axes[idx].set_title(title, fontweight='bold', fontsize=11)

        plt.colorbar(im, ax=axes[idx], shrink=0.82, label='Normalized Cost')

    fig.suptitle(
        f"Convergence Basin Heatmap — {frame_label} — {condition_label}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"basin_2d{suffix}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


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

    if n_images == 0:
        print(f"[ERROR] No images found in {cfg['rgb_dir']}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  Convergence Basin — Gray vs RGB+Ch6 (layer1)")
    print(f"{'='*60}")
    print(f"  Images found:     {n_images}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Grid size:        {cfg['grid_size']}x{cfg['grid_size']}")
    print(f"  Shift range:      +/-{cfg['max_shift_px']} pixels")
    print(f"  CNN config:       layer1, channel_select=d6, mode=rgb_cnn")
    print(f"{'='*60}")

    # Select 3 representative frames
    frame_configs = [
        (50, "Early (frame 50)"),
        (n_images // 2, f"Middle (frame {n_images // 2})"),
        (n_images - 50, f"Late (frame {n_images - 50})"),
    ]

    # Initialize CNN feature extractor: RGB + Channel 6
    device = cfg['device']
    extractor_rgb_ch6 = CNNFeatureExtractor(
        target_channels=cfg['cnn_channels'],
        device=device,
        mode="rgb_cnn",
        channel_select=cfg['channel_select'],
        cnn_layer=cfg['cnn_layer']
    )

    # Collect sharpness for summary
    all_sharpness = []

    for frame_idx_i, (ref_idx, frame_label) in enumerate(frame_configs):
        if ref_idx >= n_images:
            print(f"\n  [!] Skipping: index {ref_idx} out of range")
            continue

        print(f"\n  [{frame_idx_i + 1}/{len(frame_configs)}] {frame_label}")
        print(f"  {'─' * 50}")

        # Load reference image
        rgb_np = load_image_numpy(all_images[ref_idx])
        rgb_tensor = numpy_to_tensor(rgb_np, device)

        # Extract reference features (clean)
        print(f"    Extracting reference features...")
        feat_ref = {}
        feat_ref['gray'] = extract_features_numpy(rgb_tensor, None, 'gray')
        feat_ref['rgb_ch6'] = extract_features_numpy(rgb_tensor, extractor_rgb_ch6, 'rgb_ch6')

        # Loop over brightness conditions
        for cond in BRIGHTNESS_CONDITIONS:
            cond_label = cond['label']
            cond_suffix = cond['suffix']
            bright_factor = cond['factor']
            suffix = f"_frame{frame_idx_i + 1}{cond_suffix}"

            print(f"\n    --- {cond_label} ---")

            # Prepare target with brightness perturbation
            rgb_target_np = apply_brightness_perturbation(rgb_np, bright_factor)
            rgb_target_tensor = numpy_to_tensor(rgb_target_np, device)

            feat_target = {}
            feat_target['gray'] = extract_features_numpy(rgb_target_tensor, None, 'gray')
            feat_target['rgb_ch6'] = extract_features_numpy(rgb_target_tensor, extractor_rgb_ch6, 'rgb_ch6')
            del rgb_target_tensor

            # Compute cost landscapes
            modalities_data = {}
            for mod in ['gray', 'rgb_ch6']:
                mod_label = MODALITY_CONFIG[mod]['label']
                print(f"    Computing: {mod_label} ({cfg['grid_size']}x{cfg['grid_size']})...")
                dx, dy, cost = compute_2d_cost_landscape(
                    feat_ref[mod], feat_target[mod],
                    cfg['max_shift_px'], cfg['grid_size']
                )
                modalities_data[mod] = cost

            # Sharpness
            sharpness_results = {}
            for mod, cost in modalities_data.items():
                sharpness_results[mod] = compute_sharpness(
                    cost, dx, dy, radius_px=cfg['sharpness_radius']
                )

            # Print results
            print(f"\n    {'─'*70}")
            print(f"    Sharpness — {cond_label}")
            print(f"    {'Modality':<18} {'Local':<10} {'Global':<10} {'Min Location'}")
            print(f"    {'─'*70}")
            for mod, s in sharpness_results.items():
                label = MODALITY_CONFIG[mod]['label']
                loc_str = f"({s['min_location'][0]:+.1f}, {s['min_location'][1]:+.1f})"
                print(f"    {label:<18} {s['local']:<10.4f} {s['global']:<10.4f} {loc_str}")

            if 'gray' in sharpness_results and 'rgb_ch6' in sharpness_results:
                g = sharpness_results['gray']
                h = sharpness_results['rgb_ch6']
                ratio_l = h['local'] / g['local'] if g['local'] > 1e-12 else float('inf')
                ratio_g = h['global'] / g['global'] if g['global'] > 1e-12 else float('inf')
                print(f"    RGB+Ch6/Gray:  Local={ratio_l:.2f}x  Global={ratio_g:.2f}x")
            print(f"    {'─'*70}")

            # Store for summary
            for mod, s in sharpness_results.items():
                all_sharpness.append({
                    'frame': frame_label, 'condition': cond_label,
                    'modality': MODALITY_CONFIG[mod]['label'], **s
                })

            # Generate plots
            print(f"    Generating plots...")
            plot_comparison(
                modalities_data, dx, dy, frame_label, cond_label,
                output_dir, suffix, sharpness_all=sharpness_results
            )

        del rgb_tensor, rgb_np, feat_ref
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"  SUMMARY — All Frames x All Conditions")
    print(f"{'='*80}")
    print(f"  {'Frame':<25} {'Condition':<20} {'Modality':<18} {'Local':<10} {'Global':<10}")
    print(f"  {'-'*80}")
    for entry in all_sharpness:
        print(f"  {entry['frame']:<25} {entry['condition']:<20} {entry['modality']:<18} "
              f"{entry['local']:<10.4f} {entry['global']:<10.4f}")
    print(f"  {'='*80}")

    print(f"\n  DONE. Results in: {output_dir}/")
