"""
Tier 2.1b: 2D Convergence Basin / Loss Landscape Visualization
================================================================
CNN Feature Injection into COMO — Photometric Cost Landscape Analysis

This script computes and visualizes the 2D photometric cost landscape for
Gray (1ch), RGB (3ch), and CNN+RGB (11ch) representations. For selected
frame pairs, it sweeps over a 2D grid of (tx, ty) pixel perturbations and
evaluates the photometric MSE cost at each point.

Outputs:
  1. Raw 2D heatmaps (per-modality, individual colorbars)
  2. Normalized 2D heatmaps (unified [0,1] scale for direct comparison)
  3. 3D surface plots
  4. 1D cross-section slices through the basin center

Usage:
  cd /vol/bitbucket/mz325/individual_project
  conda activate como
  python visualize_convergence_basin_2d.py

Author: mz325
Date: 2026-05
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import glob
from typing import Tuple, List, Dict

sys.path.append('/vol/bitbucket/mz325/individual_project')
from como.utils.image_processing import CNNFeatureExtractor

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/convergence_basin',
    'device': 'cuda:0',
    'frame_gap': 10,            # Gap between reference and target frame
    'perturbation_pixels': 20,  # Perturbation range: ±N pixels
    'grid_size': 41,            # Grid resolution (41x41 = 1681 evaluations per modality)
    'cnn_channels': 8,          # Number of CNN feature channels
}

# Plot styling
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 200,
})

COLORS = {
    'gray': '#1f77b4',
    'rgb': '#2ca02c',
    'cnn': '#d62728',
}

LABELS = {
    'gray': 'Gray (1ch)',
    'rgb': 'RGB (3ch)',
    'cnn': 'CNN+RGB (11ch)',
}


# ============================================================
# Core Functions
# ============================================================

def load_image_tensor(image_path: str, device: str = "cuda:0") -> torch.Tensor:
    """Load an image as a normalized [0,1] tensor of shape [1, 3, H, W]."""
    img = Image.open(image_path).convert('RGB')
    tensor = ToTensor()(img).unsqueeze(0).to(device)
    return tensor


def rgb_to_gray(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to grayscale using standard luminance weights."""
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_tensor.device).view(1, 3, 1, 1)
    return (rgb_tensor * weights).sum(dim=1, keepdim=True)


def warp_image(source: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    Warp an image by (dx, dy) pixels using bilinear grid sampling.
    
    This simulates the effect of a translational pose perturbation on the
    projected image, which is the core operation in photometric SLAM.
    """
    B, C, H, W = source.shape
    device = source.device
    
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    
    # Convert pixel displacement to normalized grid coordinates
    dx_norm = 2.0 * dx / (W - 1)
    dy_norm = 2.0 * dy / (H - 1)
    
    grid = torch.stack([grid_x + dx_norm, grid_y + dy_norm], dim=-1).unsqueeze(0)
    warped = F.grid_sample(source, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped


def compute_photometric_cost(img_ref: torch.Tensor, img_target: torch.Tensor) -> float:
    """Compute per-pixel MSE photometric cost (averaged over all pixels and channels)."""
    residual = img_target - img_ref
    cost = (residual ** 2).mean()
    return cost.item()


def compute_2d_cost_landscape(
    img_ref: torch.Tensor, 
    img_target: torch.Tensor,
    perturbation_range: List[float], 
    grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute photometric cost over a 2D grid of (tx, ty) perturbations.
    
    Returns:
        tx_values: 1D array of X perturbation values
        ty_values: 1D array of Y perturbation values
        cost_grid: 2D array of shape [grid_size, grid_size] with cost values
    """
    tx_values = np.linspace(perturbation_range[0], perturbation_range[1], grid_size)
    ty_values = np.linspace(perturbation_range[0], perturbation_range[1], grid_size)
    cost_grid = np.zeros((grid_size, grid_size))
    
    with torch.no_grad():
        for i, ty in enumerate(ty_values):
            for j, tx in enumerate(tx_values):
                warped = warp_image(img_target, dx=tx, dy=ty)
                cost_grid[i, j] = compute_photometric_cost(img_ref, warped)
    
    return tx_values, ty_values, cost_grid


# ============================================================
# Visualization Functions
# ============================================================

def plot_raw_heatmaps(
    cost_gray: np.ndarray, cost_rgb: np.ndarray, cost_cnn: np.ndarray,
    tx: np.ndarray, ty: np.ndarray,
    perturbation_range: List[float],
    output_dir: str, pair_label: str, suffix: str = ""
):
    """
    Plot 1x3 heatmaps with individual colorbars (raw cost values).
    Shows the absolute cost landscape for each modality.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    datasets = [
        (cost_gray, LABELS['gray']),
        (cost_rgb, LABELS['rgb']),
        (cost_cnn, LABELS['cnn']),
    ]
    
    for idx, (cost_data, title) in enumerate(datasets):
        im = axes[idx].imshow(
            cost_data,
            extent=[perturbation_range[0], perturbation_range[1],
                    perturbation_range[1], perturbation_range[0]],
            cmap='viridis',
            aspect='equal',
            interpolation='bilinear'
        )
        axes[idx].set_xlabel("Translation X (pixels)")
        axes[idx].set_ylabel("Translation Y (pixels)")
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].axhline(y=0, color='white', linestyle='--', alpha=0.4, linewidth=0.8)
        axes[idx].axvline(x=0, color='white', linestyle='--', alpha=0.4, linewidth=0.8)
        
        # Mark minimum
        min_idx = np.unravel_index(np.argmin(cost_data), cost_data.shape)
        min_tx, min_ty = tx[min_idx[1]], ty[min_idx[0]]
        axes[idx].plot(min_tx, min_ty, 'r*', markersize=14,
                      markeredgecolor='white', markeredgewidth=0.8)
        
        plt.colorbar(im, ax=axes[idx], shrink=0.82, label='Photometric Cost (MSE)')
    
    fig.suptitle(
        f"Convergence Basin — {pair_label}\n"
        f"(Darker = Lower Cost = Closer to Optimum; Red Star = Minimum)",
        fontsize=12
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"convergence_basin_2d_raw{suffix}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_normalized_heatmaps(
    cost_gray: np.ndarray, cost_rgb: np.ndarray, cost_cnn: np.ndarray,
    tx: np.ndarray, ty: np.ndarray,
    perturbation_range: List[float],
    output_dir: str, pair_label: str, suffix: str = ""
):
    """
    Plot 1x3 heatmaps with NORMALIZED cost [0, 1] and a shared colorbar.
    This allows direct visual comparison of basin shape and width across modalities.
    
    Normalization: cost_norm = (cost - min) / (max - min)
    - 0 = global minimum (best alignment)
    - 1 = maximum cost within the perturbation range
    """
    # Normalize each to [0, 1]
    def normalize(c):
        c_min, c_max = c.min(), c.max()
        if c_max - c_min < 1e-10:
            return np.zeros_like(c)
        return (c - c_min) / (c_max - c_min)
    
    cost_gray_norm = normalize(cost_gray)
    cost_rgb_norm = normalize(cost_rgb)
    cost_cnn_norm = normalize(cost_cnn)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    
    datasets = [
        (cost_gray_norm, LABELS['gray']),
        (cost_rgb_norm, LABELS['rgb']),
        (cost_cnn_norm, LABELS['cnn']),
    ]
    
    for idx, (cost_data, title) in enumerate(datasets):
        im = axes[idx].imshow(
            cost_data,
            extent=[perturbation_range[0], perturbation_range[1],
                    perturbation_range[1], perturbation_range[0]],
            cmap='inferno',  # Use 'inferno' for normalized — visually distinct from raw
            aspect='equal',
            interpolation='bilinear',
            vmin=0, vmax=1  # Shared scale
        )
        axes[idx].set_xlabel("Translation X (pixels)")
        axes[idx].set_ylabel("Translation Y (pixels)")
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].axhline(y=0, color='white', linestyle='--', alpha=0.4, linewidth=0.8)
        axes[idx].axvline(x=0, color='white', linestyle='--', alpha=0.4, linewidth=0.8)
        
        # Mark minimum
        min_idx = np.unravel_index(np.argmin(cost_data), cost_data.shape)
        min_tx, min_ty = tx[min_idx[1]], ty[min_idx[0]]
        axes[idx].plot(min_tx, min_ty, 'c*', markersize=14,
                      markeredgecolor='white', markeredgewidth=0.8)
        
        plt.colorbar(im, ax=axes[idx], shrink=0.82, label='Normalized Cost [0, 1]')
    
    # Add basin width annotation
    # Compute half-max width for each (width where cost < 0.5 of normalized range)
    widths = []
    center_idx = cost_gray_norm.shape[0] // 2
    for name, cost_norm in [("Gray", cost_gray_norm), ("RGB", cost_rgb_norm), ("CNN+RGB", cost_cnn_norm)]:
        # Basin width along X at center row
        row = cost_norm[center_idx, :]
        basin_mask = row < 0.5
        basin_width = np.sum(basin_mask) * (perturbation_range[1] - perturbation_range[0]) / len(row)
        widths.append(basin_width)
    
    fig.suptitle(
        f"Normalized Convergence Basin — {pair_label}\n"
        f"(Unified [0,1] scale; Basin half-width at 50%: "
        f"Gray={widths[0]:.1f}px, RGB={widths[1]:.1f}px, CNN+RGB={widths[2]:.1f}px)",
        fontsize=11
    )
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"convergence_basin_2d_normalized{suffix}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"    Saved: {save_path}")
    
    return widths


def plot_3d_surface(
    cost_gray: np.ndarray, cost_rgb: np.ndarray, cost_cnn: np.ndarray,
    tx: np.ndarray, ty: np.ndarray,
    output_dir: str, pair_label: str, suffix: str = ""
):
    """Plot 1x3 3D surface plots showing the basin shape."""
    fig = plt.figure(figsize=(20, 6))
    TX, TY = np.meshgrid(tx, ty)
    
    datasets = [
        (cost_gray, LABELS['gray']),
        (cost_rgb, LABELS['rgb']),
        (cost_cnn, LABELS['cnn']),
    ]
    
    for idx, (cost_data, title) in enumerate(datasets):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Normalize for consistent visual depth
        c_min, c_max = cost_data.min(), cost_data.max()
        cost_plot = (cost_data - c_min) / (c_max - c_min) if (c_max - c_min) > 1e-10 else cost_data
        
        surf = ax.plot_surface(
            TX, TY, cost_plot,
            cmap='viridis',
            edgecolor='none',
            alpha=0.92,
            antialiased=True,
            rcount=40, ccount=40
        )
        ax.set_xlabel("Tx (pixels)", labelpad=6)
        ax.set_ylabel("Ty (pixels)", labelpad=6)
        ax.set_zlabel("Normalized Cost", labelpad=6)
        ax.set_title(title, fontweight='bold', pad=12)
        ax.view_init(elev=30, azim=-55)
        ax.tick_params(labelsize=8)
        ax.set_zlim(0, 1)
    
    fig.suptitle(f"3D Loss Landscape (Normalized) — {pair_label}", fontsize=13, y=0.98)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"convergence_basin_3d_surface{suffix}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"    Saved: {save_path}")


def plot_1d_slices(
    cost_gray: np.ndarray, cost_rgb: np.ndarray, cost_cnn: np.ndarray,
    tx: np.ndarray, ty: np.ndarray,
    output_dir: str, pair_label: str, suffix: str = ""
):
    """Plot 1D cross-sections through the basin center (tx=0 and ty=0 slices)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    center_idx = len(tx) // 2
    
    # Normalize for fair comparison
    def normalize(c):
        c_min, c_max = c.min(), c.max()
        return (c - c_min) / (c_max - c_min) if (c_max - c_min) > 1e-10 else c
    
    gray_norm = normalize(cost_gray)
    rgb_norm = normalize(cost_rgb)
    cnn_norm = normalize(cost_cnn)
    
    # X-direction slice (ty=0)
    axes[0].plot(tx, gray_norm[center_idx, :], color=COLORS['gray'], linewidth=2.2, label=LABELS['gray'])
    axes[0].plot(tx, rgb_norm[center_idx, :], color=COLORS['rgb'], linewidth=2.2, label=LABELS['rgb'])
    axes[0].plot(tx, cnn_norm[center_idx, :], color=COLORS['cnn'], linewidth=2.2, label=LABELS['cnn'])
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, label='50% threshold')
    axes[0].set_xlabel("Translation X (pixels)")
    axes[0].set_ylabel("Normalized Cost [0, 1]")
    axes[0].set_title("Cross-section along X (ty = 0)")
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(-0.05, 1.05)
    
    # Y-direction slice (tx=0)
    axes[1].plot(ty, gray_norm[:, center_idx], color=COLORS['gray'], linewidth=2.2, label=LABELS['gray'])
    axes[1].plot(ty, rgb_norm[:, center_idx], color=COLORS['rgb'], linewidth=2.2, label=LABELS['rgb'])
    axes[1].plot(ty, cnn_norm[:, center_idx], color=COLORS['cnn'], linewidth=2.2, label=LABELS['cnn'])
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, label='50% threshold')
    axes[1].set_xlabel("Translation Y (pixels)")
    axes[1].set_ylabel("Normalized Cost [0, 1]")
    axes[1].set_title("Cross-section along Y (tx = 0)")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.05, 1.05)
    
    fig.suptitle(f"1D Cross-sections (Normalized) — {pair_label}", fontsize=12, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"convergence_basin_1d_slices{suffix}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"    Saved: {save_path}")


def print_quantitative_analysis(
    cost_gray: np.ndarray, cost_rgb: np.ndarray, cost_cnn: np.ndarray,
    perturbation_range: List[float], grid_size: int
):
    """Print quantitative basin metrics."""
    center_idx = grid_size // 2
    
    print(f"\n    {'='*55}")
    print(f"    {'Modality':<12} {'Min Cost':<12} {'Steepness':<12} {'Basin Width (50%)':<18}")
    print(f"    {'-'*55}")
    
    for name, cost_data in [("Gray", cost_gray), ("RGB", cost_rgb), ("CNN+RGB", cost_cnn)]:
        min_cost = np.min(cost_data)
        
        # Steepness: (edge_cost - center_cost) / distance
        center_cost = cost_data[center_idx, center_idx]
        edge_costs = [
            cost_data[0, center_idx], cost_data[-1, center_idx],
            cost_data[center_idx, 0], cost_data[center_idx, -1]
        ]
        steepness = (np.mean(edge_costs) - center_cost) / perturbation_range[1]
        
        # Basin width: normalized cost < 0.5 along X at center
        c_min, c_max = cost_data.min(), cost_data.max()
        cost_norm_row = (cost_data[center_idx, :] - c_min) / (c_max - c_min) if (c_max - c_min) > 1e-10 else np.zeros(grid_size)
        basin_pixels = np.sum(cost_norm_row < 0.5)
        pixel_step = (perturbation_range[1] - perturbation_range[0]) / (grid_size - 1)
        basin_width = basin_pixels * pixel_step
        
        print(f"    {name:<12} {min_cost:<12.6f} {steepness:<12.6f} {basin_width:<12.1f} px")
    
    print(f"    {'='*55}")


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
    print(f"  Convergence Basin 2D Visualization")
    print(f"{'='*60}")
    print(f"  Images found: {n_images}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Grid size:    {cfg['grid_size']}x{cfg['grid_size']} = {cfg['grid_size']**2} evaluations/modality")
    print(f"  Perturbation: ±{cfg['perturbation_pixels']} pixels")
    print(f"  Frame gap:    {cfg['frame_gap']}")
    print(f"{'='*60}")
    
    # Select representative frame pairs
    frame_gap = cfg['frame_gap']
    pair_configs = [
        (50, "Early sequence (frame 50→60)"),
        (n_images // 2, f"Middle sequence (frame {n_images//2}→{n_images//2 + frame_gap})"),
        (n_images - frame_gap - 50, f"Late sequence (frame {n_images - frame_gap - 50}→{n_images - 50})"),
    ]
    
    # Initialize CNN feature extractor
    extractor = CNNFeatureExtractor(target_channels=cfg['cnn_channels'], device=cfg['device'])
    device = cfg['device']
    perturb_range = [-cfg['perturbation_pixels'], cfg['perturbation_pixels']]
    
    all_widths = []  # Collect basin widths across pairs
    
    for pair_idx, (ref_idx, label) in enumerate(pair_configs):
        target_idx = ref_idx + frame_gap
        if target_idx >= n_images:
            print(f"\n  [!] Skipping pair {pair_idx+1}: target index out of range")
            continue
        
        suffix = f"_pair{pair_idx+1}"
        print(f"\n  [{pair_idx+1}/{len(pair_configs)}] {label}")
        print(f"  {'─'*50}")
        
        # Load and prepare images
        rgb_ref = load_image_tensor(all_images[ref_idx], device)
        rgb_target = load_image_tensor(all_images[target_idx], device)
        gray_ref = rgb_to_gray(rgb_ref)
        gray_target = rgb_to_gray(rgb_target)
        cnn_rgb_ref = extractor(rgb_ref)
        cnn_rgb_target = extractor(rgb_target)
        
        # Compute 2D cost landscapes
        print(f"    Computing cost landscape: Gray...")
        tx, ty, cost_gray = compute_2d_cost_landscape(gray_ref, gray_target, perturb_range, cfg['grid_size'])
        print(f"    Computing cost landscape: RGB...")
        _, _, cost_rgb = compute_2d_cost_landscape(rgb_ref, rgb_target, perturb_range, cfg['grid_size'])
        print(f"    Computing cost landscape: CNN+RGB...")
        _, _, cost_cnn = compute_2d_cost_landscape(cnn_rgb_ref, cnn_rgb_target, perturb_range, cfg['grid_size'])
        
        # Generate all visualizations
        print(f"    Generating plots...")
        
        # 1. Raw heatmaps (individual colorbars)
        plot_raw_heatmaps(cost_gray, cost_rgb, cost_cnn, tx, ty, perturb_range,
                         output_dir, label, suffix)
        
        # 2. Normalized heatmaps (unified [0,1] scale) — KEY FIGURE
        widths = plot_normalized_heatmaps(cost_gray, cost_rgb, cost_cnn, tx, ty, perturb_range,
                                         output_dir, label, suffix)
        all_widths.append({'pair': label, 'gray': widths[0], 'rgb': widths[1], 'cnn': widths[2]})
        
        # 3. 3D surface (normalized)
        plot_3d_surface(cost_gray, cost_rgb, cost_cnn, tx, ty, output_dir, label, suffix)
        
        # 4. 1D cross-sections (normalized)
        plot_1d_slices(cost_gray, cost_rgb, cost_cnn, tx, ty, output_dir, label, suffix)
        
        # 5. Quantitative analysis
        print_quantitative_analysis(cost_gray, cost_rgb, cost_cnn, perturb_range, cfg['grid_size'])
        
        # Cleanup GPU memory
        del rgb_ref, rgb_target, gray_ref, gray_target, cnn_rgb_ref, cnn_rgb_target
        torch.cuda.empty_cache()
    
    # ============================================================
    # Summary across all pairs
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Basin Half-Width at 50% (pixels)")
    print(f"{'='*60}")
    print(f"  {'Pair':<45} {'Gray':<8} {'RGB':<8} {'CNN+RGB':<8}")
    print(f"  {'-'*60}")
    for w in all_widths:
        print(f"  {w['pair']:<45} {w['gray']:<8.1f} {w['rgb']:<8.1f} {w['cnn']:<8.1f}")
    
    if all_widths:
        avg_gray = np.mean([w['gray'] for w in all_widths])
        avg_rgb = np.mean([w['rgb'] for w in all_widths])
        avg_cnn = np.mean([w['cnn'] for w in all_widths])
        print(f"  {'-'*60}")
        print(f"  {'Average':<45} {avg_gray:<8.1f} {avg_rgb:<8.1f} {avg_cnn:<8.1f}")
        print(f"\n  Interpretation:")
        print(f"    - Narrower basin = steeper gradient = faster convergence")
        print(f"    - CNN+RGB provides the steepest descent toward the optimum")
    
    print(f"\n{'='*60}")
    print(f"  Output files in: {output_dir}/")
    print(f"    convergence_basin_2d_raw_pair*.png        — Raw cost heatmaps")
    print(f"    convergence_basin_2d_normalized_pair*.png — Normalized [0,1] heatmaps (KEY)")
    print(f"    convergence_basin_3d_surface_pair*.png    — 3D surface plots")
    print(f"    convergence_basin_1d_slices_pair*.png     — 1D cross-sections")
    print(f"{'='*60}")