"""
Convergence Basin — All 64 Conv1 Channels (Individual, 3D Surface)
===================================================================
For each of the 64 channels from ResNet18 conv1 + bn1 + relu, generates a
single figure with 3 columns (3D surface): Clean / Brightness+30% / Brightness+50%.

This matches the feature extraction used in forward_greedy_selection.py and
visualize_forward_results.py (conv1 -> BN -> ReLU, NOT layer1).

Fixed frame: frame 306 from the dataset.

Usage:
  cd /vol/bitbucket/mz325/individual_project
  python visualize_basin_all64_channels.py

Output:
  vis_results/convergence_basin_64ch/channel_XX.png  (XX = 00..63)
  vis_results/convergence_basin_64ch/sharpness_summary.csv
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as mplNormalize
import numpy as np
from PIL import Image
import os
import sys
import glob
import cv2
import csv
from typing import Tuple, Dict

sys.path.insert(0, '/vol/bitbucket/mz325/individual_project/como')
sys.path.insert(0, '/vol/bitbucket/mz325/individual_project')

from torchvision.models import resnet18, ResNet18_Weights

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/convergence_basin_64ch',
    'device': 'cuda:0',
    'frame_index': 306,

    # Perturbation range
    'max_shift_px': 30,
    'grid_size': 61,

    # Sharpness
    'sharpness_radius': 5,
}

BRIGHTNESS_CONDITIONS = [
    {'factor': 0.0, 'label': 'Clean'},
    {'factor': 0.3, 'label': '+30%'},
    {'factor': 0.5, 'label': '+50%'},
]

plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'mathtext.fontset': 'cm',
})


# ============================================================
# Feature Extractor: conv1 + bn1 + relu (matches forward_greedy)
# ============================================================

class Conv1BNReLUExtractor:
    """Extract all 64 conv1+bn1+relu features at once.
    This matches DirectConv1Extractor in forward_greedy_selection.py."""

    def __init__(self, device='cuda:0'):
        self.device = device
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        resnet.eval()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU(inplace=False)

        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def extract_all_channels(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input: [1, 3, H, W] in [0, 1]
        Output: [1, 64, H, W] upsampled to original resolution
        """
        orig_size = rgb_tensor.shape[-2:]
        x = (rgb_tensor - self.mean) / self.std
        x = self.conv1(x)       # [1, 64, H/2, W/2]
        x = self.bn1(x)
        x = self.relu(x)        # After BN + ReLU
        upsampled = F.interpolate(
            x, size=orig_size,
            mode='bilinear', align_corners=False
        )
        return upsampled  # [1, 64, H, W]


# ============================================================
# Core Functions
# ============================================================

def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = image.shape[:2]
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(image, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
    return warped


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 0.0:
        return image.copy()
    return np.clip(image + factor, 0.0, 1.0)


def load_image_numpy(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def compute_cost_landscape(
    feat_ref: np.ndarray,
    feat_target: np.ndarray,
    max_shift: float,
    grid_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """feat_ref, feat_target: [H, W] single channel."""
    dx_vals = np.linspace(-max_shift, max_shift, grid_size)
    dy_vals = np.linspace(-max_shift, max_shift, grid_size)
    cost_grid = np.zeros((grid_size, grid_size))

    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            shifted = shift_image(feat_target, dx, dy)
            residual = shifted.astype(np.float64) - feat_ref.astype(np.float64)
            cost_grid[i, j] = np.mean(residual ** 2)

    return dx_vals, dy_vals, cost_grid


def compute_sharpness(cost_grid, dx_vals, dy_vals, radius=5):
    grid_size = cost_grid.shape[0]
    center = grid_size // 2
    step_x = dx_vals[1] - dx_vals[0]
    step_y = dy_vals[1] - dy_vals[0]

    c_min, c_max = cost_grid.min(), cost_grid.max()
    if c_max - c_min > 1e-10:
        cost_norm = (cost_grid - c_min) / (c_max - c_min)
    else:
        cost_norm = np.zeros_like(cost_grid)

    def _sharpness(arr, step, c_idx, r):
        lo = max(c_idx - r, 0)
        hi = min(c_idx + r, len(arr) - 1)
        seg = arr[lo:hi+1]
        if len(seg) < 2:
            return 0.0
        return float(np.mean(np.abs(seg[1:] - seg[:-1]) / step))

    full_r = grid_size // 2
    x_local = _sharpness(cost_norm[center, :], step_x, center, radius)
    y_local = _sharpness(cost_norm[:, center], step_y, center, radius)
    x_global = _sharpness(cost_norm[center, :], step_x, center, full_r)
    y_global = _sharpness(cost_norm[:, center], step_y, center, full_r)

    return {
        'local': (x_local + y_local) / 2.0,
        'global': (x_global + y_global) / 2.0,
        'x_local': x_local, 'y_local': y_local,
        'x_global': x_global, 'y_global': y_global,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = CONFIG
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    all_images = sorted(glob.glob(os.path.join(cfg['rgb_dir'], "*.png")))
    n_images = len(all_images)
    if n_images == 0:
        print(f"[ERROR] No images found in {cfg['rgb_dir']}")
        sys.exit(1)

    frame_idx = cfg['frame_index']
    if frame_idx >= n_images:
        print(f"[ERROR] Frame index {frame_idx} out of range (total {n_images})")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"  Convergence Basin — All 64 Conv1+BN+ReLU Channels")
    print(f"{'='*60}")
    print(f"  Extraction:   conv1 -> bn1 -> relu (matches forward_greedy)")
    print(f"  Dataset:      {cfg['rgb_dir']}")
    print(f"  Frame:        {frame_idx}")
    print(f"  Grid:         {cfg['grid_size']}x{cfg['grid_size']}")
    print(f"  Shift range:  +/-{cfg['max_shift_px']} px")
    print(f"  Output:       {output_dir}")
    print(f"  Total jobs:   64 channels x 3 conditions = 192 cost landscapes")
    print(f"{'='*60}")

    device = cfg['device']
    extractor = Conv1BNReLUExtractor(device=device)

    # Load reference image
    rgb_np = load_image_numpy(all_images[frame_idx])
    rgb_tensor = numpy_to_tensor(rgb_np, device)

    # Extract all 64 channels for reference (clean)
    print(f"\n  Extracting 64 conv1+bn+relu features for reference frame...")
    all_feat_ref = extractor.extract_all_channels(rgb_tensor)  # [1, 64, H, W]
    all_feat_ref_np = all_feat_ref[0].cpu().numpy()  # [64, H, W]

    # Prepare target features for each brightness condition
    target_feats = {}
    for cond in BRIGHTNESS_CONDITIONS:
        factor = cond['factor']
        label = cond['label']
        print(f"  Extracting features for condition: {label}")
        rgb_target_np = apply_brightness(rgb_np, factor)
        rgb_target_tensor = numpy_to_tensor(rgb_target_np, device)
        feat_target = extractor.extract_all_channels(rgb_target_tensor)
        target_feats[label] = feat_target[0].cpu().numpy()  # [64, H, W]
        del rgb_target_tensor
    
    del rgb_tensor
    torch.cuda.empty_cache()

    # CSV for sharpness summary
    csv_path = os.path.join(output_dir, "sharpness_summary.csv")
    csv_rows = []

    # Process each channel
    print(f"\n  Processing 64 channels...")
    for ch in range(64):
        print(f"  Channel {ch:02d}/63 ...", end=" ", flush=True)

        feat_ref_ch = all_feat_ref_np[ch]  # [H, W]

        # Compute cost landscape for each condition
        costs = {}
        sharpness_per_cond = {}
        for cond in BRIGHTNESS_CONDITIONS:
            label = cond['label']
            feat_target_ch = target_feats[label][ch]  # [H, W]
            dx, dy, cost = compute_cost_landscape(
                feat_ref_ch, feat_target_ch,
                cfg['max_shift_px'], cfg['grid_size']
            )
            costs[label] = cost
            sharpness_per_cond[label] = compute_sharpness(
                cost, dx, dy, radius=cfg['sharpness_radius']
            )

        # Save sharpness to CSV
        for cond_label, s in sharpness_per_cond.items():
            csv_rows.append({
                'channel': ch,
                'condition': cond_label,
                'local_sharpness': f"{s['local']:.6f}",
                'global_sharpness': f"{s['global']:.6f}",
                'x_local': f"{s['x_local']:.6f}",
                'y_local': f"{s['y_local']:.6f}",
            })

        # ── Plot: 1 row x 3 columns, 3D surface (Clean / +30% / +50%) ──
        fig = plt.figure(figsize=(18, 6))
        DX_grid, DY_grid = np.meshgrid(dx, dy, indexing='ij')

        for idx, cond in enumerate(BRIGHTNESS_CONDITIONS):
            label = cond['label']
            cost_data = costs[label]

            c_min, c_max = cost_data.min(), cost_data.max()
            if c_max - c_min > 1e-10:
                cost_norm = (cost_data - c_min) / (c_max - c_min)
            else:
                cost_norm = np.zeros_like(cost_data)

            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

            cmap = plt.get_cmap('YlOrRd')
            norm = mplNormalize(vmin=0, vmax=1)
            facecolors = cmap(norm(cost_norm))

            ax.plot_surface(
                DY_grid, DX_grid, cost_norm,
                facecolors=facecolors,
                edgecolor='k', linewidth=0.12, alpha=0.92,
                shade=True, rcount=40, ccount=40, antialiased=True
            )

            contour_offset = -0.05
            ax.contourf(DY_grid, DX_grid, cost_norm, zdir='z', offset=contour_offset,
                        levels=20, cmap='gray_r', alpha=0.7)
            ax.contour(DY_grid, DX_grid, cost_norm, zdir='z', offset=contour_offset,
                       levels=10, colors='k', linewidths=0.4, alpha=0.5)

            ax.set_xlabel(r'$\Delta x$ [px]', labelpad=8)
            ax.set_ylabel(r'$\Delta y$ [px]', labelpad=8)
            ax.set_zlabel('Norm. Cost', labelpad=6)
            ax.set_zlim(contour_offset, 1.05)
            ax.view_init(elev=32, azim=-50)
            ax.tick_params(labelsize=8)

            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.fill = False
                pane.set_edgecolor('lightgray')

            s = sharpness_per_cond[label]
            ax.set_title(f"{label}\nLocal={s['local']:.4f}  Global={s['global']:.4f}",
                         fontweight='bold', fontsize=10, pad=12)

        fig.suptitle(f"Channel {ch:02d} — Conv1+BN+ReLU — Frame {frame_idx}",
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"channel_{ch:02d}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"saved")

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['channel', 'condition', 'local_sharpness',
                                               'global_sharpness', 'x_local', 'y_local'])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'='*60}")
    print(f"  DONE — 64 channel plots saved to: {output_dir}/")
    print(f"  Sharpness CSV: {csv_path}")
    print(f"{'='*60}")