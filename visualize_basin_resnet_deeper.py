"""
visualize_basin_resnet_deeper.py
=================================
Convergence Basin — ResNet-18 Deeper Layers (layer3 / layer4)
==============================================================
For each channel extracted from ResNet-18 layer3 (256ch, H/16×W/16) or
layer4 (512ch, H/32×W/32), generates a single figure with 3 columns
(3D surface): Clean / Brightness+30% / Brightness+50%.

This mirrors the existing UNet basin visualizer but uses the ResNet-18
backbone already integrated into CNNFeatureExtractor, targeting the
deeper layers to test illumination robustness.

Extraction paths:
  layer3: conv1→bn1→relu→maxpool→layer1→layer2→layer3  (256ch, H/16×W/16, upsample 16×)
  layer4: conv1→bn1→relu→maxpool→layer1→layer2→layer3→layer4 (512ch, H/32×W/32, upsample 32×)

Usage (run from como/ subdirectory):
  python ../visualize_basin_resnet_deeper.py --layer layer3 --frame 306
  python ../visualize_basin_resnet_deeper.py --layer layer4 --frame 306

Output:
  vis_results/convergence_basin_resnet_layer3/channel_XXX.png
  vis_results/convergence_basin_resnet_layer3/sharpness_summary.csv
  vis_results/convergence_basin_resnet_layer4/channel_XXX.png
  vis_results/convergence_basin_resnet_layer4/sharpness_summary.csv
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
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as mplNormalize
from PIL import Image
import cv2

# ── Must run from como/ subdirectory ──
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir':          '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'device':           'cuda:0',
    'frame_index':      306,
    'max_shift_px':     30,
    'grid_size':        61,
    'sharpness_radius': 5,
}

BRIGHTNESS_CONDITIONS = [
    {'factor': 0.0, 'label': 'Clean'},
    {'factor': 0.3, 'label': '+30%'},
    {'factor': 0.5, 'label': '+50%'},
]

# Layer metadata: name → (total_channels, upsample_factor, description)
LAYER_META = {
    'layer3': (256, 16,  'conv1→bn1→relu→maxpool→layer1→layer2→layer3'),
    'layer4': (512, 32,  'conv1→bn1→relu→maxpool→layer1→layer2→layer3→layer4'),
}

plt.rcParams.update({
    'font.size':        10,
    'font.family':      'serif',
    'axes.titlesize':   11,
    'axes.labelsize':   10,
    'figure.dpi':       120,
    'savefig.dpi':      200,
    'mathtext.fontset': 'cm',
})

# ImageNet normalisation
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ============================================================
# ResNet-18 Deeper Layer Feature Extractor
# ============================================================
class ResNetDeeperExtractor:
    """
    Extracts all channels from ResNet-18 layer3 or layer4.

    layer3: 256ch, H/16×W/16 → bilinear upsample 16× back to H×W
    layer4: 512ch, H/32×W/32 → bilinear upsample 32× back to H×W

    Uses ImageNet-pretrained weights (ResNet18_Weights.IMAGENET1K_V1).
    All parameters are frozen (eval mode, no_grad).
    """

    def __init__(self, layer: str = 'layer3', device: str = 'cuda:0'):
        assert layer in LAYER_META, f"layer must be one of {list(LAYER_META.keys())}"
        self.layer   = layer
        self.device  = device
        total_ch, self.upsample_factor, path_desc = LAYER_META[layer]
        self.total_channels = total_ch

        print(f"[INFO] Loading ResNet-18 (ImageNet pretrained) ...")
        from torchvision.models import resnet18, ResNet18_Weights
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        resnet.eval()

        if layer == 'layer3':
            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
            )
        else:  # layer4
            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4,
            )

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.mean = IMAGENET_MEAN.to(device)
        self.std  = IMAGENET_STD.to(device)

        print(f"[INFO] Extractor ready: {layer} | {total_ch}ch | "
              f"upsample={self.upsample_factor}× | path: {path_desc}")

    @torch.no_grad()
    def extract_all_channels(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input:  [1, 3, H, W] float32 in [0, 1]
        Output: [1, C, H, W] — all channels upsampled to original resolution
        """
        orig_size = rgb_tensor.shape[-2:]
        x = (rgb_tensor - self.mean) / self.std
        features = self.backbone(x)   # [1, C, H/k, W/k]
        upsampled = F.interpolate(
            features, size=orig_size,
            mode='bilinear', align_corners=False
        )
        return upsampled              # [1, C, H, W]


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
    grid_size: int,
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
# Sharpness Metrics
# ============================================================
def compute_sharpness(cost_grid, dx_vals, dy_vals, radius=5) -> Dict:
    grid_size = cost_grid.shape[0]
    center    = grid_size // 2
    step_x    = dx_vals[1] - dx_vals[0]
    step_y    = dy_vals[1] - dy_vals[0]

    c_min, c_max = cost_grid.min(), cost_grid.max()
    cost_norm = (
        (cost_grid - c_min) / (c_max - c_min)
        if c_max - c_min > 1e-10
        else np.zeros_like(cost_grid)
    )

    def _sharpness(arr, step, c_idx, r):
        lo  = max(c_idx - r, 0)
        hi  = min(c_idx + r, len(arr) - 1)
        seg = arr[lo:hi + 1]
        if len(seg) < 2:
            return 0.0
        return float(np.mean(np.abs(seg[1:] - seg[:-1]) / step))

    full_r   = grid_size // 2
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
    parser = argparse.ArgumentParser(
        description='ResNet-18 Deeper Layer Convergence Basin Visualization'
    )
    parser.add_argument(
        '--layer', type=str, default='layer3', choices=['layer3', 'layer4'],
        help='Which ResNet-18 layer to visualize (default: layer3)',
    )
    parser.add_argument(
        '--frame', type=int, default=CONFIG['frame_index'],
        help='Frame index in dataset (default: 306)',
    )
    parser.add_argument('--rgb_dir',    type=str, default=CONFIG['rgb_dir'])
    parser.add_argument('--device',     type=str, default=CONFIG['device'])
    parser.add_argument(
        '--out_dir', type=str, default=None,
        help='Output root directory. Defaults to vis_results/convergence_basin_resnet_<layer>',
    )
    parser.add_argument(
        '--max_channels', type=int, default=None,
        help='Limit number of channels to visualize (useful for quick tests). '
             'Default: all channels.',
    )
    args = parser.parse_args()

    layer     = args.layer
    frame_idx = args.frame
    total_ch, upsample_factor, _ = LAYER_META[layer]

    if args.out_dir is None:
        output_dir = f'vis_results/convergence_basin_resnet_{layer}'
    else:
        output_dir = args.out_dir
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 65)
    print(f'  Convergence Basin — ResNet-18 {layer}')
    print('=' * 65)
    print(f'  Dataset:      {args.rgb_dir}')
    print(f'  Frame:        {frame_idx}')
    print(f'  Layer:        {layer}  ({total_ch}ch, upsample {upsample_factor}×)')
    print(f'  Grid:         {CONFIG["grid_size"]}×{CONFIG["grid_size"]}')
    print(f'  Shift:        ±{CONFIG["max_shift_px"]} px')
    print(f'  Output:       {output_dir}')
    print('=' * 65)

    # ── Load extractor ──
    extractor = ResNetDeeperExtractor(layer=layer, device=args.device)

    # ── Load images ──
    all_images = sorted(glob.glob(os.path.join(args.rgb_dir, '*.png')))
    if not all_images:
        print(f'[ERROR] No PNG images found in {args.rgb_dir}')
        sys.exit(1)
    if frame_idx >= len(all_images):
        print(f'[ERROR] Frame {frame_idx} out of range (total {len(all_images)})')
        sys.exit(1)

    rgb_np     = load_image_numpy(all_images[frame_idx])
    rgb_tensor = numpy_to_tensor(rgb_np, args.device)

    # ── Extract reference features ──
    print(f'\n[INFO] Extracting reference features ({layer}) ...')
    all_feat_ref    = extractor.extract_all_channels(rgb_tensor)   # [1, C, H, W]
    all_feat_ref_np = all_feat_ref[0].cpu().numpy()                # [C, H, W]
    num_channels    = all_feat_ref_np.shape[0]

    if args.max_channels is not None:
        num_channels = min(num_channels, args.max_channels)
        print(f'[INFO] Limiting to first {num_channels} channels (--max_channels)')

    print(f'[INFO] Channels to process: {num_channels}  '
          f'Feature shape: {all_feat_ref_np.shape}')

    # ── Extract target features for each brightness condition ──
    target_feats = {}
    for cond in BRIGHTNESS_CONDITIONS:
        factor = cond['factor']
        label  = cond['label']
        print(f'[INFO] Extracting features for condition: {label}')
        rgb_target_np     = apply_brightness(rgb_np, factor)
        rgb_target_tensor = numpy_to_tensor(rgb_target_np, args.device)
        feat_target       = extractor.extract_all_channels(rgb_target_tensor)
        target_feats[label] = feat_target[0].cpu().numpy()   # [C, H, W]
        del rgb_target_tensor

    del rgb_tensor
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    # ── CSV for sharpness summary ──
    csv_path = os.path.join(output_dir, 'sharpness_summary.csv')
    csv_rows = []

    # ── Process each channel ──
    print(f'\n[INFO] Processing {num_channels} channels ...')
    for ch in range(num_channels):
        print(f'  Channel {ch:03d}/{num_channels - 1:03d} ...', end=' ', flush=True)

        feat_ref_ch = all_feat_ref_np[ch]   # [H, W]

        # Compute cost landscape for each condition
        costs               = {}
        sharpness_per_cond  = {}
        for cond in BRIGHTNESS_CONDITIONS:
            label          = cond['label']
            feat_target_ch = target_feats[label][ch]
            dx, dy, cost   = compute_cost_landscape(
                feat_ref_ch, feat_target_ch,
                CONFIG['max_shift_px'], CONFIG['grid_size'],
            )
            costs[label]              = cost
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
            cost_norm = (
                (cost_data - c_min) / (c_max - c_min)
                if c_max - c_min > 1e-10
                else np.zeros_like(cost_data)
            )

            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

            cmap       = plt.get_cmap('YlOrRd')
            norm       = mplNormalize(vmin=0, vmax=1)
            facecolors = cmap(norm(cost_norm))

            ax.plot_surface(
                DY_grid, DX_grid, cost_norm,
                facecolors=facecolors,
                edgecolor='k', linewidth=0.12, alpha=0.92,
                shade=True, rcount=40, ccount=40, antialiased=True,
            )

            contour_offset = -0.05
            ax.contourf(DY_grid, DX_grid, cost_norm, zdir='z',
                        offset=contour_offset, levels=20, cmap='gray_r', alpha=0.7)
            ax.contour(DY_grid, DX_grid, cost_norm, zdir='z',
                       offset=contour_offset, levels=10,
                       colors='k', linewidths=0.4, alpha=0.5)

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
        total_ch_meta, up_meta, path_desc = LAYER_META[layer]
        fig.suptitle(
            f'Channel {ch:03d} — ResNet-18 {layer} '
            f'(conv1→bn1→relu→maxpool→layer1→{"→".join(["layer" + str(i) for i in range(2, int(layer[-1]) + 1)])}) '
            f'— Frame {frame_idx}\n'
            f'{total_ch_meta}ch, resolution=H/{upsample_factor}×W/{upsample_factor}, '
            f'upsample={upsample_factor}×',
            fontsize=10, y=1.02,
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, f'channel_{ch:03d}.png')
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
    print(f'[DONE] All channel plots  → {output_dir}/')
    print(f'       channel_000.png ~ channel_{num_channels - 1:03d}.png')


if __name__ == '__main__':
    main()