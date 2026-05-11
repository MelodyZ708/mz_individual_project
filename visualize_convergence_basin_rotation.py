"""
Convergence Basin Visualization — Rotation-Based (Pitch/Yaw)
=============================================================
CNN Feature Injection into COMO — Photometric Cost Landscape Analysis

**Method A**: Warp same image against itself.
  - Load ONE reference image, extract features.
  - For each (pitch, yaw) perturbation: warp the features by that rotation,
    compute MSE against the original (unwarped) features.
  - Ground truth = identity (zero perturbation) → minimum cost at center (0,0).
  - This cleanly measures "how sensitive is each representation to rotation
    perturbations" and reveals the convergence basin shape.

Outputs:
  1. 3D surface plots with contour projection on the floor (publication style)
  2. 2D heatmaps with contour lines
  3. Quantitative metrics: raw steepness, local curvature, basin width

Reference style: Alismail et al., "Photometric Bundle Adjustment for
Vision-Based SLAM", Fig. 1 — 3D bowl with floor contour projection.

Usage:
  cd /vol/bitbucket/mz325/individual_project
  python visualize_convergence_basin_rotation.py

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
from typing import Tuple, List, Dict

sys.path.append('/vol/bitbucket/mz325/individual_project')
from como.utils.image_processing import CNNFeatureExtractor

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/convergence_basin_rotation_methodA',
    'device': 'cuda:0',

    # Camera intrinsics — TUM freiburg1 (Kinect v1, 640x480)
    'fx': 517.3,
    'fy': 516.5,
    'cx': 318.6,
    'cy': 255.3,

    # Perturbation range
    'max_angle_rad': 0.15,     # ±0.15 rad ≈ ±8.6 degrees
    'grid_size': 41,           # 41x41 = 1681 evaluations per modality

    # CNN config
    'cnn_channels': 8,
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


# ============================================================
# Core Functions
# ============================================================

def get_intrinsic_matrix(cfg: dict) -> np.ndarray:
    """Build 3x3 camera intrinsic matrix K."""
    K = np.array([
        [cfg['fx'], 0,         cfg['cx']],
        [0,         cfg['fy'], cfg['cy']],
        [0,         0,         1.0      ]
    ], dtype=np.float64)
    return K


def rotation_matrix_pitch_yaw(pitch: float, yaw: float) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from pitch (rotation around X-axis)
    and yaw (rotation around Y-axis).

    R = Ry(yaw) @ Rx(pitch)
    """
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1,  0,   0 ],
        [0,  cp, -sp],
        [0,  sp,  cp]
    ], dtype=np.float64)

    Ry = np.array([
        [ cy, 0, sy],
        [ 0,  1, 0 ],
        [-sy, 0, cy]
    ], dtype=np.float64)

    return Ry @ Rx


def warp_image_rotation(image: np.ndarray, K: np.ndarray,
                        pitch: float, yaw: float) -> np.ndarray:
    """
    Warp an image by a pure rotation (pitch, yaw) using homography.

    For a pure rotation R, the induced homography is:
        H = K @ R @ K^{-1}

    This is the standard formulation used in photometric SLAM for
    evaluating the cost landscape w.r.t. rotational perturbations.
    """
    R = rotation_matrix_pitch_yaw(pitch, yaw)
    K_inv = np.linalg.inv(K)
    H = K @ R @ K_inv

    h, w = image.shape[:2]
    orig_shape = image.shape
    warped = cv2.warpPerspective(
        image, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    # cv2.warpPerspective squeezes single-channel [H,W,1] -> [H,W]
    if warped.shape != orig_shape:
        warped = warped.reshape(orig_shape)
    return warped


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


def compute_2d_cost_landscape_rotation(
    feat_ref: np.ndarray,
    K: np.ndarray,
    max_angle: float,
    grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute photometric cost over a 2D grid of (pitch, yaw) perturbations.

    **Method A**: Warp feat_ref by each (pitch, yaw) rotation and compute
    MSE against the original (unwarped) feat_ref.

    Ground truth is at (0, 0) where the warped image equals the original,
    giving zero cost. This cleanly reveals the convergence basin shape.

    Returns:
        pitch_values: 1D array of pitch values (rad)
        yaw_values: 1D array of yaw values (rad)
        cost_grid: 2D array [grid_size, grid_size]
    """
    pitch_values = np.linspace(-max_angle, max_angle, grid_size)
    yaw_values = np.linspace(-max_angle, max_angle, grid_size)
    cost_grid = np.zeros((grid_size, grid_size))

    for i, pitch in enumerate(pitch_values):
        for j, yaw in enumerate(yaw_values):
            warped = warp_image_rotation(feat_ref, K, pitch, yaw)
            cost_grid[i, j] = compute_photometric_cost(feat_ref, warped)

    return pitch_values, yaw_values, cost_grid


# ============================================================
# Visualization Functions
# ============================================================

def plot_3d_surface_publication(
    cost_data: np.ndarray,
    pitch: np.ndarray,
    yaw: np.ndarray,
    title: str,
    output_path: str,
    colormap: str = 'GnBu_r'
):
    """
    Plot a single 3D surface with contour projection on the floor,
    matching the publication style from the reference figures.

    Features:
    - Wireframe mesh surface with color mapping
    - Contour projection on the bottom (z=0 plane)
    - Clean axis labels with radian units
    """
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    PITCH, YAW = np.meshgrid(pitch, yaw, indexing='ij')

    # Normalize cost to [0, 1] for consistent visualization
    c_min, c_max = cost_data.min(), cost_data.max()
    if c_max - c_min > 1e-10:
        cost_norm = (cost_data - c_min) / (c_max - c_min)
    else:
        cost_norm = np.zeros_like(cost_data)

    # --- 3D Surface ---
    cmap = plt.get_cmap(colormap)
    norm = Normalize(vmin=0, vmax=1)
    facecolors = cmap(norm(cost_norm))

    surf = ax.plot_surface(
        PITCH, YAW, cost_norm,
        facecolors=facecolors,
        edgecolor='k',
        linewidth=0.15,
        alpha=0.92,
        shade=True,
        rcount=40, ccount=40,
        antialiased=True
    )

    # --- Contour projection on the floor (z=0 plane) ---
    # Use gray colormap for the floor contour
    contour_offset = -0.05  # Slightly below z=0
    ax.contourf(
        PITCH, YAW, cost_norm,
        zdir='z', offset=contour_offset,
        levels=20,
        cmap='gray_r',
        alpha=0.7
    )
    ax.contour(
        PITCH, YAW, cost_norm,
        zdir='z', offset=contour_offset,
        levels=10,
        colors='k',
        linewidths=0.5,
        alpha=0.5
    )

    # --- Axis formatting ---
    ax.set_xlabel('pitch [rad]', labelpad=10)
    ax.set_ylabel('yaw [rad]', labelpad=10)
    ax.set_zlabel('Normalized Cost', labelpad=8)
    ax.set_title(title, fontweight='bold', pad=15, fontsize=14)

    ax.set_zlim(contour_offset, 1.05)
    ax.view_init(elev=32, azim=-50)
    ax.tick_params(labelsize=9)

    # Clean up grid
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
    pitch: np.ndarray,
    yaw: np.ndarray,
    frame_label: str,
    output_dir: str,
    suffix: str = ""
):
    """
    Plot side-by-side 3D surfaces for all modalities (1 row, N columns).
    Each subplot has its own contour projection on the floor.
    """
    modalities = list(costs.keys())
    n_mod = len(modalities)

    fig = plt.figure(figsize=(7 * n_mod, 7))
    PITCH, YAW = np.meshgrid(pitch, yaw, indexing='ij')

    colormaps = {
        'gray': 'GnBu_r',
        'cnn_only': 'YlOrRd',
        'cnn_rgb': 'YlOrRd',
    }

    for idx, mod in enumerate(modalities):
        ax = fig.add_subplot(1, n_mod, idx + 1, projection='3d')
        cost_data = costs[mod]

        # Normalize
        c_min, c_max = cost_data.min(), cost_data.max()
        if c_max - c_min > 1e-10:
            cost_norm = (cost_data - c_min) / (c_max - c_min)
        else:
            cost_norm = np.zeros_like(cost_data)

        cmap_name = colormaps.get(mod, 'viridis')
        cmap = plt.get_cmap(cmap_name)
        norm = Normalize(vmin=0, vmax=1)
        facecolors = cmap(norm(cost_norm))

        # Surface
        ax.plot_surface(
            PITCH, YAW, cost_norm,
            facecolors=facecolors,
            edgecolor='k',
            linewidth=0.12,
            alpha=0.92,
            shade=True,
            rcount=40, ccount=40,
            antialiased=True
        )

        # Floor contour
        contour_offset = -0.05
        ax.contourf(
            PITCH, YAW, cost_norm,
            zdir='z', offset=contour_offset,
            levels=20, cmap='gray_r', alpha=0.7
        )
        ax.contour(
            PITCH, YAW, cost_norm,
            zdir='z', offset=contour_offset,
            levels=10, colors='k', linewidths=0.4, alpha=0.5
        )

        ax.set_xlabel('pitch [rad]', labelpad=8)
        ax.set_ylabel('yaw [rad]', labelpad=8)
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
        f"Convergence Basin — Method A (Self-Warp) — {frame_label}",
        fontsize=15, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"convergence_basin_3d_rotation{suffix}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def plot_2d_heatmaps(
    costs: Dict[str, np.ndarray],
    pitch: np.ndarray,
    yaw: np.ndarray,
    frame_label: str,
    output_dir: str,
    suffix: str = ""
):
    """Plot 2D heatmaps with contour lines for all modalities."""
    modalities = list(costs.keys())
    n_mod = len(modalities)

    fig, axes = plt.subplots(1, n_mod, figsize=(6 * n_mod, 5))
    if n_mod == 1:
        axes = [axes]

    extent = [yaw[0], yaw[-1], pitch[-1], pitch[0]]

    for idx, mod in enumerate(modalities):
        cost_data = costs[mod]

        # Normalize
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

        # Contour lines
        PITCH, YAW_grid = np.meshgrid(pitch, yaw, indexing='ij')
        axes[idx].contour(
            YAW_grid, PITCH, cost_norm,
            levels=10, colors='white', linewidths=0.5, alpha=0.6
        )

        axes[idx].set_xlabel("yaw [rad]")
        axes[idx].set_ylabel("pitch [rad]")
        axes[idx].set_title(MODALITY_CONFIG[mod]['label'], fontweight='bold')
        axes[idx].axhline(y=0, color='white', linestyle='--', alpha=0.3, linewidth=0.8)
        axes[idx].axvline(x=0, color='white', linestyle='--', alpha=0.3, linewidth=0.8)

        # Mark minimum — should be at (0, 0) for Method A
        min_idx = np.unravel_index(np.argmin(cost_norm), cost_norm.shape)
        min_yaw = yaw[min_idx[1]] if cost_norm.shape[1] == len(yaw) else 0
        min_pitch = pitch[min_idx[0]] if cost_norm.shape[0] == len(pitch) else 0
        axes[idx].plot(min_yaw, min_pitch, 'c*', markersize=12,
                       markeredgecolor='white', markeredgewidth=0.8)

        plt.colorbar(im, ax=axes[idx], shrink=0.82, label='Normalized Cost')

    fig.suptitle(
        f"Convergence Basin Heatmap — Method A (Self-Warp) — {frame_label}",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"convergence_basin_2d_rotation{suffix}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_path}")


def print_basin_metrics(
    costs: Dict[str, np.ndarray],
    pitch: np.ndarray,
    yaw: np.ndarray,
    max_angle: float
):
    """
    Print quantitative basin metrics for each modality.

    Metrics (all computed on RAW, un-normalized cost):
      1. Raw Steepness: (mean_edge_cost - center_cost) / max_angle
         → Higher = cost rises more per radian = steeper bowl
      2. Local Curvature: average 2nd-order finite difference at center
         along pitch and yaw axes (Hessian diagonal approximation)
         → Higher = sharper minimum = stronger gradient near optimum
      3. Basin Width (50%): width of region where normalized cost < 0.5
         → Narrower = sharper/more peaked basin
      4. CNN/Gray Ratio: for quick comparison
    """
    center_idx = len(pitch) // 2
    step = (2 * max_angle) / (len(pitch) - 1)  # grid step in radians

    print(f"\n    {'='*80}")
    print(f"    {'Modality':<18} {'Raw Steep.':<12} {'Local Curv.':<14} "
          f"{'Basin W. (50%)':<16} {'Max Cost'}")
    print(f"    {'-'*80}")

    results = {}
    for mod, cost_data in costs.items():
        label = MODALITY_CONFIG[mod]['label']

        # --- Raw Steepness (un-normalized) ---
        center_cost = cost_data[center_idx, center_idx]
        edge_costs = [
            cost_data[0, center_idx], cost_data[-1, center_idx],
            cost_data[center_idx, 0], cost_data[center_idx, -1]
        ]
        raw_steepness = (np.mean(edge_costs) - center_cost) / max_angle

        # --- Local Curvature at center (2nd-order finite difference) ---
        # d²C/dpitch² ≈ (C[c-1,c] + C[c+1,c] - 2*C[c,c]) / step²
        # d²C/dyaw²   ≈ (C[c,c-1] + C[c,c+1] - 2*C[c,c]) / step²
        curv_pitch = (cost_data[center_idx - 1, center_idx] +
                      cost_data[center_idx + 1, center_idx] -
                      2 * center_cost) / (step ** 2)
        curv_yaw = (cost_data[center_idx, center_idx - 1] +
                    cost_data[center_idx, center_idx + 1] -
                    2 * center_cost) / (step ** 2)
        local_curvature = (curv_pitch + curv_yaw) / 2.0

        # --- Basin Width at 50% of max cost (un-normalized) ---
        c_max = cost_data.max()
        if c_max > 1e-10:
            # Measure along yaw axis through center
            cost_row = cost_data[center_idx, :]
            basin_mask = cost_row < (0.5 * c_max)
            basin_width_rad = np.sum(basin_mask) * step
        else:
            basin_width_rad = 2 * max_angle  # flat = full width

        max_cost = cost_data.max()

        print(f"    {label:<18} {raw_steepness:<12.4f} {local_curvature:<14.2f} "
              f"{basin_width_rad:<16.4f} {max_cost:.6f}")

        results[mod] = {
            'raw_steepness': raw_steepness,
            'local_curvature': local_curvature,
            'basin_width': basin_width_rad,
            'max_cost': max_cost,
        }

    print(f"    {'-'*80}")

    # --- Print ratios (CNN vs Gray) ---
    if 'gray' in results and 'cnn_only' in results:
        gray = results['gray']
        cnn = results['cnn_only']
        steep_ratio = cnn['raw_steepness'] / gray['raw_steepness'] if gray['raw_steepness'] > 0 else float('inf')
        curv_ratio = cnn['local_curvature'] / gray['local_curvature'] if gray['local_curvature'] > 0 else float('inf')
        print(f"    CNN/Gray Ratios:  Steepness={steep_ratio:.2f}×  "
              f"Curvature={curv_ratio:.2f}×  "
              f"MaxCost={cnn['max_cost']/gray['max_cost']:.2f}×")

    if 'gray' in results and 'cnn_rgb' in results:
        gray = results['gray']
        hybrid = results['cnn_rgb']
        steep_ratio = hybrid['raw_steepness'] / gray['raw_steepness'] if gray['raw_steepness'] > 0 else float('inf')
        curv_ratio = hybrid['local_curvature'] / gray['local_curvature'] if gray['local_curvature'] > 0 else float('inf')
        print(f"    RGB/Gray Ratios:  Steepness={steep_ratio:.2f}×  "
              f"Curvature={curv_ratio:.2f}×  "
              f"MaxCost={hybrid['max_cost']/gray['max_cost']:.2f}×")

    print(f"    {'='*80}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = CONFIG
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    K = get_intrinsic_matrix(cfg)

    # Load image list
    all_images = sorted(glob.glob(os.path.join(cfg['rgb_dir'], "*.png")))
    n_images = len(all_images)

    print(f"{'='*60}")
    print(f"  Convergence Basin — Rotation (Pitch/Yaw) — Method A")
    print(f"  (Warp same image against itself)")
    print(f"{'='*60}")
    print(f"  Images found:     {n_images}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Grid size:        {cfg['grid_size']}x{cfg['grid_size']} = {cfg['grid_size']**2} evaluations/modality")
    print(f"  Angle range:      ±{cfg['max_angle_rad']} rad (±{np.degrees(cfg['max_angle_rad']):.1f}°)")
    print(f"  Camera intrinsics: fx={cfg['fx']}, fy={cfg['fy']}, cx={cfg['cx']}, cy={cfg['cy']}")
    print(f"{'='*60}")

    # Select 3 representative frames (early / middle / late)
    frame_configs = [
        (50, "Early (frame 50)"),
        (n_images // 2, f"Middle (frame {n_images // 2})"),
        (n_images - 50, f"Late (frame {n_images - 50})"),
    ]

    # Initialize CNN feature extractors
    device = cfg['device']

    # CNN-only extractor (8ch, conv1)
    extractor_cnn_only = CNNFeatureExtractor(
        target_channels=cfg['cnn_channels'],
        device=device,
        mode="cnn_only"
    )

    # CNN+RGB extractor (11ch, conv1)
    extractor_cnn_rgb = CNNFeatureExtractor(
        target_channels=cfg['cnn_channels'],
        device=device,
        mode="rgb_cnn"
    )

    for frame_idx_i, (ref_idx, label) in enumerate(frame_configs):
        if ref_idx >= n_images:
            print(f"\n  [!] Skipping frame {frame_idx_i + 1}: index out of range")
            continue

        suffix = f"_frame{frame_idx_i + 1}"
        print(f"\n  [{frame_idx_i + 1}/{len(frame_configs)}] {label}")
        print(f"  {'─' * 50}")

        # Load ONE image only (Method A: self-warp)
        rgb_ref_tensor = load_image_tensor(all_images[ref_idx], device)

        # Extract features as numpy [H, W, C]
        print(f"    Extracting features from single reference image...")
        feat_gray = extract_features_numpy(rgb_ref_tensor, None, 'gray')
        feat_cnn_only = extract_features_numpy(rgb_ref_tensor, extractor_cnn_only, 'cnn_only')
        feat_cnn_rgb = extract_features_numpy(rgb_ref_tensor, extractor_cnn_rgb, 'cnn_rgb')

        # Compute cost landscapes (Method A: warp feat against itself)
        modalities_data = {}

        print(f"    Computing cost landscape: Gray ({cfg['grid_size']}x{cfg['grid_size']})...")
        pitch, yaw, cost_gray = compute_2d_cost_landscape_rotation(
            feat_gray, K,
            cfg['max_angle_rad'], cfg['grid_size']
        )
        modalities_data['gray'] = cost_gray

        print(f"    Computing cost landscape: CNN-only ({cfg['grid_size']}x{cfg['grid_size']})...")
        _, _, cost_cnn_only = compute_2d_cost_landscape_rotation(
            feat_cnn_only, K,
            cfg['max_angle_rad'], cfg['grid_size']
        )
        modalities_data['cnn_only'] = cost_cnn_only

        print(f"    Computing cost landscape: CNN+RGB ({cfg['grid_size']}x{cfg['grid_size']})...")
        _, _, cost_cnn_rgb = compute_2d_cost_landscape_rotation(
            feat_cnn_rgb, K,
            cfg['max_angle_rad'], cfg['grid_size']
        )
        modalities_data['cnn_rgb'] = cost_cnn_rgb

        # Generate visualizations
        print(f"    Generating plots...")

        # 1. Individual 3D surfaces (publication style)
        for mod, cost in modalities_data.items():
            mod_label = MODALITY_CONFIG[mod]['label']
            cmap = 'GnBu_r' if mod == 'gray' else 'YlOrRd'
            plot_3d_surface_publication(
                cost, pitch, yaw,
                title=f"{mod_label} — {label}",
                output_path=os.path.join(output_dir, f"basin_3d_{mod}{suffix}.png"),
                colormap=cmap
            )

        # 2. Side-by-side 3D comparison
        plot_3d_comparison(modalities_data, pitch, yaw, label, output_dir, suffix)

        # 3. 2D heatmaps with contour lines
        plot_2d_heatmaps(modalities_data, pitch, yaw, label, output_dir, suffix)

        # 4. Quantitative metrics (RAW, un-normalized)
        print_basin_metrics(modalities_data, pitch, yaw, cfg['max_angle_rad'])

        # Cleanup
        del rgb_ref_tensor
        del feat_gray, feat_cnn_only, feat_cnn_rgb
        torch.cuda.empty_cache()

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Method A: Self-Warp Convergence Basin")
    print(f"  Output files in: {output_dir}/")
    print(f"    basin_3d_<modality>_frame*.png   — Individual 3D surfaces")
    print(f"    convergence_basin_3d_rotation_frame*.png — Side-by-side comparison")
    print(f"    convergence_basin_2d_rotation_frame*.png — 2D heatmaps")
    print(f"{'='*60}")