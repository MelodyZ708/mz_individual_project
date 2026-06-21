"""
visualize_basin_unet.py
=======================
Convergence Basin — U-Net Encoder Channels (Individual, 3D Surface)
====================================================================
For each channel from COMO DepthCovModule's U-Net encoder (enc0: 16ch or enc1: 32ch),
generates a single figure with 3 columns (3D surface): Clean / Brightness+30% / Brightness+50%.

This is the P3 direction: Zero-cost feature reuse from the U-Net already present
in COMO's Mapping module, rather than loading an external ResNet-18.

Extraction path:
  enc0: gaussian_cov_net.base         -> 16ch, H×W
  enc1: gaussian_cov_net.down_convs[0]-> 32ch, H/2×W/2 (upsampled back to H×W)

Usage:
  cd /vol/bitbucket/mz325/individual_project/como
  python ../visualize_basin_unet.py [--enc_level 0|1] [--frame 306]

Output:
  vis_results/convergence_basin_unet_enc{N}/channel_XX.png
  vis_results/convergence_basin_unet_enc{N}/sharpness_summary.csv
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import glob
import csv
import argparse
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as mplNormalize
from PIL import Image
import cv2

# ── Must run from como/ subdirectory ──
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

from como.depth_cov.core.DepthCovModule import DepthCovModule

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir':    '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'model_path': 'models/scannet.ckpt',
    'device':     'cuda:0',
    'frame_index': 306,
    'max_shift_px': 30,
    'grid_size':    61,
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

# ImageNet normalisation (same as ResNet, used by DepthCovModule)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================
# U-Net Feature Extractor
# ============================================================
class UNetEncoderExtractor:
    """
    Extracts shallow encoder features from COMO's DepthCovModule U-Net.

    enc_level=0: self.base output        -> 16ch, H×W   (receptive field ~5×5)
    enc_level=1: self.down_convs[0] output -> 32ch, H/2×W/2 (receptive field ~11×11)
                 upsampled back to H×W for fair comparison with ResNet.
    """

    def __init__(self, model_path: str, enc_level: int = 1, device: str = 'cuda:0'):
        self.device = device
        self.enc_level = enc_level

        # Load DepthCovModule (U-Net lives inside as gaussian_cov_net)
        print(f"[INFO] Loading DepthCovModule from {model_path} ...")
        self.module = DepthCovModule.load_from_checkpoint(
            model_path,
            map_location=device,
        )
        self.module.eval()
        self.module.to(device)
        for p in self.module.parameters():
            p.requires_grad = False

        self.unet = self.module.gaussian_cov_net
        print(f"[INFO] U-Net loaded. enc_level={enc_level}")

        self.mean = IMAGENET_MEAN.to(device)
        self.std  = IMAGENET_STD.to(device)

    @torch.no_grad()
    def extract_all_channels(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input:  [1, 3, H, W] in [0, 1]
        Output: [1, C, H, W] — all channels at original resolution
        """
        orig_size = rgb_tensor.shape[-2:]
        x = (rgb_tensor - self.mean) / self.std

        # Encoder forward (partial)
        x = self.unet.base(x)           # enc0: [1, 16, H, W]
        if self.enc_level == 0:
            return x                    # 16ch, already H×W

        x = self.unet.down_convs[0](x)  # enc1: [1, 32, H/2, W/2]
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x                        # 32ch, upsampled to H×W


# ============================================================
# Image Utilities
# ============================================================
def load_image_numpy(path: str) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_tensor(img: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def apply_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    if factor == 0.0:
        return image.copy()
    return np.clip(image + factor, 0.0, 1.0)


def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = image.shape[:2]
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


# ============================================================
# Cost Landscape
# ============================================================
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


# ============================================================
# Sharpness
# ============================================================
def compute_sharpness(cost_grid, dx_vals, dy_vals, radius=5) -> Dict:
    grid_size = cost_grid.shape[0]
    center = grid_size // 2
    step_x = dx_vals[1] - dx_vals[0]
    step_y = dy_vals[1] - dy_vals[0]

    c_min, c_max = cost_grid.min(), cost_grid.max()
    cost_norm = (cost_grid - c_min) / (c_max - c_min) if c_max - c_min > 1e-10 else np.zeros_like(cost_grid)

    def _sharpness(arr, step, c_idx, r):
        lo = max(c_idx - r, 0)
        hi = min(c_idx + r, len(arr) - 1)
        seg = arr[lo:hi + 1]
        if len(seg) < 2:
            return 0.0
        return float(np.mean(np.abs(seg[1:] - seg[:-1]) / step))

    full_r = grid_size // 2
    x_local  = _sharpness(cost_norm[center, :], step_x, center, radius)
    y_local  = _sharpness(cost_norm[:, center], step_y, center, radius)
    x_global = _sharpness(cost_norm[center, :], step_x, center, full_r)
    y_global = _sharpness(cost_norm[:, center], step_y, center, full_r)

    return {
        'local':    (x_local  + y_local)  / 2.0,
        'global':   (x_global + y_global) / 2.0,
        'x_local':  x_local,
        'y_local':  y_local,
        'x_global': x_global,
        'y_global': y_global,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='U-Net Convergence Basin Visualization')
    parser.add_argument('--enc_level', type=int, default=1, choices=[0, 1],
                        help='Encoder level: 0=16ch H×W, 1=32ch H/2 (default: 1)')
    parser.add_argument('--frame', type=int, default=306,
                        help='Frame index in dataset (default: 306)')
    parser.add_argument('--rgb_dir', type=str, default=CONFIG['rgb_dir'])
    parser.add_argument('--model_path', type=str, default=CONFIG['model_path'])
    parser.add_argument('--device', type=str, default=CONFIG['device'])
    args = parser.parse_args()

    enc_level  = args.enc_level
    frame_idx  = args.frame
    output_dir = f'vis_results/convergence_basin_unet_enc{enc_level}'
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 60)
    print(f'  Convergence Basin — U-Net Encoder enc{enc_level}')
    print('=' * 60)
    print(f'  Model:      {args.model_path}')
    print(f'  Dataset:    {args.rgb_dir}')
    print(f'  Frame:      {frame_idx}')
    print(f'  Grid:       {CONFIG["grid_size"]}×{CONFIG["grid_size"]}')
    print(f'  Shift:      ±{CONFIG["max_shift_px"]} px')
    print(f'  Output:     {output_dir}')
    print('=' * 60)

    # ── Load extractor ──
    extractor = UNetEncoderExtractor(
        model_path=args.model_path,
        enc_level=enc_level,
        device=args.device,
    )

    # ── Load images ──
    all_images = sorted(glob.glob(os.path.join(args.rgb_dir, '*.png')))
    if not all_images:
        print(f'[ERROR] No images found in {args.rgb_dir}')
        sys.exit(1)
    if frame_idx >= len(all_images):
        print(f'[ERROR] Frame {frame_idx} out of range (total {len(all_images)})')
        sys.exit(1)

    rgb_np     = load_image_numpy(all_images[frame_idx])
    rgb_tensor = numpy_to_tensor(rgb_np, args.device)

    # ── Extract reference features ──
    print(f'\n[INFO] Extracting reference features (enc{enc_level})...')
    all_feat_ref = extractor.extract_all_channels(rgb_tensor)  # [1, C, H, W]
    all_feat_ref_np = all_feat_ref[0].cpu().numpy()            # [C, H, W]
    num_channels = all_feat_ref_np.shape[0]
    print(f'[INFO] Channels: {num_channels}  Shape: {all_feat_ref_np.shape}')

    # ── Extract target features for each brightness condition ──
    target_feats = {}
    for cond in BRIGHTNESS_CONDITIONS:
        factor = cond['factor']
        label  = cond['label']
        print(f'[INFO] Extracting features for condition: {label}')
        rgb_target_np     = apply_brightness(rgb_np, factor)
        rgb_target_tensor = numpy_to_tensor(rgb_target_np, args.device)
        feat_target = extractor.extract_all_channels(rgb_target_tensor)
        target_feats[label] = feat_target[0].cpu().numpy()  # [C, H, W]
        del rgb_target_tensor

    del rgb_tensor
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    # ── CSV for sharpness summary ──
    csv_path = os.path.join(output_dir, 'sharpness_summary.csv')
    csv_rows  = []

    # ── Process each channel ──
    print(f'\n[INFO] Processing {num_channels} channels...')
    for ch in range(num_channels):
        print(f'  Channel {ch:02d}/{num_channels - 1} ...', end=' ', flush=True)

        feat_ref_ch = all_feat_ref_np[ch]  # [H, W]

        # Compute cost landscape for each condition
        costs = {}
        sharpness_per_cond = {}
        for cond in BRIGHTNESS_CONDITIONS:
            label = cond['label']
            feat_target_ch = target_feats[label][ch]
            dx, dy, cost = compute_cost_landscape(
                feat_ref_ch, feat_target_ch,
                CONFIG['max_shift_px'], CONFIG['grid_size']
            )
            costs[label] = cost
            sharpness_per_cond[label] = compute_sharpness(
                cost, dx, dy, radius=CONFIG['sharpness_radius']
            )

        # Save sharpness to CSV
        for cond_label, s in sharpness_per_cond.items():
            csv_rows.append({
                'channel':          ch,
                'condition':        cond_label,
                'local_sharpness':  f"{s['local']:.6f}",
                'global_sharpness': f"{s['global']:.6f}",
                'x_local':          f"{s['x_local']:.6f}",
                'y_local':          f"{s['y_local']:.6f}",
            })

        # ── Plot: 1 row × 3 columns, 3D surface ──
        fig = plt.figure(figsize=(18, 6))
        DX_grid, DY_grid = np.meshgrid(dx, dy, indexing='ij')

        for idx, cond in enumerate(BRIGHTNESS_CONDITIONS):
            label     = cond['label']
            cost_data = costs[label]

            c_min, c_max = cost_data.min(), cost_data.max()
            cost_norm = (cost_data - c_min) / (c_max - c_min) if c_max - c_min > 1e-10 else np.zeros_like(cost_data)

            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

            cmap      = plt.get_cmap('YlOrRd')
            norm      = mplNormalize(vmin=0, vmax=1)
            facecolors = cmap(norm(cost_norm))

            ax.plot_surface(
                DY_grid, DX_grid, cost_norm,
                facecolors=facecolors,
                edgecolor='k', linewidth=0.12, alpha=0.92,
                shade=True, rcount=40, ccount=40, antialiased=True,
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
            ax.set_title(
                f'{label}\nLocal={s["local"]:.4f}  Global={s["global"]:.4f}',
                fontsize=10,
            )

        # ── Super title ──
        enc_path = (
            f'gaussian_cov_net.base (16ch, H×W)'
            if enc_level == 0
            else f'gaussian_cov_net.down_convs[0] (32ch, H/2×W/2)'
        )
        fig.suptitle(
            f'Channel {ch:02d} — U-Net enc{enc_level} — Frame {frame_idx}\n'
            f'Extraction: COMO DepthCovModule → {enc_path}',
            fontsize=11, y=1.02,
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, f'channel_{ch:02d}.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f'saved → {out_path}')

    # ── Write CSV ──
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'channel', 'condition',
            'local_sharpness', 'global_sharpness',
            'x_local', 'y_local',
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f'\n[DONE] Sharpness summary → {csv_path}')
    print(f'[DONE] All channel plots → {output_dir}/')
    print(f'       channel_00.png ~ channel_{num_channels - 1:02d}.png')


if __name__ == '__main__':
    main()