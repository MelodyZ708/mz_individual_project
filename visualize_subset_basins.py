"""
Visualize Convergence Basins for Greedy-Selected Channel Subsets
================================================================
Generates 3D surface plots (Clean / +30% / +50%) for each subset,
matching the style of the single-channel basin plots.

Usage:
    python visualize_subset_basins.py

Outputs saved to: vis_results/forward_greedy_bqs/
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

# ============================================================
# Configuration
# ============================================================
RGB_DIR    = '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/'
OUTPUT_DIR = 'vis_results/forward_greedy_bqs'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

FRAME_IDX  = 306
GRID_RANGE = 30
GRID_STEP  = 1
CHUNK_SIZE = 256

CONDITIONS = [
    ('Clean',  0.0),
    ('+30%',   0.3),
    ('+50%',   0.5),
]

# The two best subsets from beam search
SUBSETS = [
    {
        'label':   'Beam1/3 Best',
        'channels': [6, 28, 34, 62, 12, 54, 3],
        'bqs':      0.7402,
    },
    {
        'label':   'Beam2 Best',
        'channels': [19, 6, 28, 62, 52, 54],
        'bqs':      0.6990,
    },
]

# ============================================================
# Feature Extractor
# ============================================================
class Conv1BNReLUExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = base.conv1
        self.bn1   = base.bn1
        self.relu  = nn.ReLU(inplace=False)
        self.device = device
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    @torch.no_grad()
    def forward(self, img_tensor):
        orig_size = img_tensor.shape[-2:]
        x = (img_tensor - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x  # (1, 64, H, W)

# ============================================================
# Image Utils
# ============================================================
def load_image_np(path):
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0

def numpy_to_tensor(img_np, device):
    return torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)

def apply_brightness(img_np, factor):
    return np.clip(img_np + factor, 0.0, 1.0) if factor != 0.0 else img_np.copy()

# ============================================================
# GPU Shift Grid
# ============================================================
def build_shift_grid(H, W, dx_vals, dy_vals, device):
    step_x = 2.0 / (W - 1)
    step_y = 2.0 / (H - 1)
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([grid_x, grid_y], dim=-1)
    dy_norm = torch.tensor(dy_vals, dtype=torch.float32, device=device) * step_y
    dx_norm = torch.tensor(dx_vals, dtype=torch.float32, device=device) * step_x
    N_dy, N_dx = len(dy_vals), len(dx_vals)
    N = N_dy * N_dx
    grids = base.unsqueeze(0).expand(N, -1, -1, -1).clone()
    idx = 0
    for i in range(N_dy):
        for j in range(N_dx):
            grids[idx, :, :, 0] += dx_norm[j]
            grids[idx, :, :, 1] += dy_norm[i]
            idx += 1
    return grids

@torch.no_grad()
def compute_cost_landscape_gpu(feat_ref, feat_tgt, channels, shift_grids,
                                N_dy, N_dx, chunk_size=CHUNK_SIZE):
    N = N_dy * N_dx
    ref_sub = feat_ref[channels]
    tgt_sub = feat_tgt[channels]
    cost_flat = torch.zeros(N, dtype=torch.float32, device=feat_ref.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = end - start
        gc = shift_grids[start:end]
        tc = tgt_sub.unsqueeze(0).expand(chunk, -1, -1, -1)
        ts = F.grid_sample(tc, gc, mode='bilinear', padding_mode='border', align_corners=True)
        rc = ref_sub.unsqueeze(0).expand(chunk, -1, -1, -1)
        cost_flat[start:end] = (ts - rc).pow(2).mean(dim=(1, 2, 3))
    return cost_flat.cpu().numpy().reshape(N_dy, N_dx)

# ============================================================
# Sharpness (local curvature at minimum)
# ============================================================
def compute_sharpness(grid_norm, dx, dy):
    h, w = grid_norm.shape
    cy, cx = h // 2, w // 2
    if cx <= 0 or cx >= w-1 or cy <= 0 or cy >= h-1:
        return 0.0
    sx = float(dx[1] - dx[0]) if len(dx) > 1 else 1.0
    sy = float(dy[1] - dy[0]) if len(dy) > 1 else 1.0
    fxx = (grid_norm[cy, cx+1] - 2*grid_norm[cy, cx] + grid_norm[cy, cx-1]) / (sx**2)
    fyy = (grid_norm[cy+1, cx] - 2*grid_norm[cy, cx] + grid_norm[cy-1, cx]) / (sy**2)
    return float(np.sqrt(fxx * fyy)) if fxx > 0 and fyy > 0 else 0.0

# ============================================================
# 3D Plot (matching the reference style)
# ============================================================
def plot_basin_3d(ax, DX, DY, cost_norm, title_top, title_bot, cmap='RdYlBu_r'):
    """Draw a single 3D surface panel."""
    surf = ax.plot_surface(DX, DY, cost_norm,
                           cmap=cmap, linewidth=0.3,
                           edgecolor='white', alpha=0.95,
                           rstride=1, cstride=1)

    # Shadow contour on the floor
    ax.contourf(DX, DY, cost_norm, zdir='z', offset=-0.05,
                cmap='gray', alpha=0.4, levels=20)

    ax.set_zlim(-0.05, 1.05)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('Δx [px]', fontsize=8, labelpad=4)
    ax.set_ylabel('Δy [px]', fontsize=8, labelpad=4)
    ax.set_zlabel('Norm. Cost', fontsize=8, labelpad=4)
    ax.tick_params(axis='both', labelsize=7)
    ax.view_init(elev=28, azim=-55)

    ax.set_title(title_top, fontsize=9, fontweight='bold', pad=2)
    ax.text2D(0.5, 0.97, title_bot, transform=ax.transAxes,
              ha='center', va='top', fontsize=8)

# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_images = sorted(glob.glob(os.path.join(RGB_DIR, '*.png')))
    if not all_images:
        raise FileNotFoundError(f'No images found in {RGB_DIR}')

    extractor = Conv1BNReLUExtractor(device=DEVICE)

    dx = np.arange(-GRID_RANGE, GRID_RANGE + 1, GRID_STEP)
    dy = np.arange(-GRID_RANGE, GRID_RANGE + 1, GRID_STEP)
    N_dx, N_dy = len(dx), len(dy)

    print('Building shift grids on GPU ...', flush=True)
    sample_np = load_image_np(all_images[FRAME_IDX])
    H, W = sample_np.shape[:2]
    shift_grids = build_shift_grid(H, W, dx, dy, DEVICE)
    print(f'Shift grids: {shift_grids.shape}', flush=True)

    DX, DY = np.meshgrid(dx, dy)

    for subset_info in SUBSETS:
        channels = subset_info['channels']
        label    = subset_info['label']
        bqs_val  = subset_info['bqs']
        ch_str   = str(channels)

        print(f'\nProcessing subset: {channels}  ({label})', flush=True)

        rgb_np     = load_image_np(all_images[FRAME_IDX])
        rgb_tensor = numpy_to_tensor(rgb_np, DEVICE)
        with torch.no_grad():
            feat_ref = extractor(rgb_tensor)[0]

        cost_grids = {}
        sharp_vals = {}

        for cond_name, factor in CONDITIONS:
            tgt_np     = apply_brightness(rgb_np, factor)
            tgt_tensor = numpy_to_tensor(tgt_np, DEVICE)
            with torch.no_grad():
                feat_tgt = extractor(tgt_tensor)[0]

            raw = compute_cost_landscape_gpu(feat_ref, feat_tgt, channels,
                                             shift_grids, N_dy, N_dx)
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

            gmin, gmax = raw.min(), raw.max()
            if gmax - gmin < 1e-8:
                norm = np.zeros_like(raw)
            else:
                norm = (raw - gmin) / (gmax - gmin)

            cost_grids[cond_name] = norm
            sharp_vals[cond_name] = compute_sharpness(norm, dx, dy)
            print(f'  {cond_name:8s}  sharpness={sharp_vals[cond_name]:.4f}', flush=True)

        # ── Build figure ──
        fig = plt.figure(figsize=(15, 5))
        fig.patch.set_facecolor('white')

        # Super-title
        n_ch = len(channels)
        suptitle = (f'Convergence Basin — {n_ch}-Channel Subset — Frame {FRAME_IDX}\n'
                    f'Channels: {channels}   BQS={bqs_val:.4f}')
        fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=1.02)

        for col, (cond_name, _) in enumerate(CONDITIONS):
            ax = fig.add_subplot(1, 3, col + 1, projection='3d')
            norm_grid = cost_grids[cond_name]
            sh = sharp_vals[cond_name]

            top_title  = cond_name
            bot_title  = f'Sharpness={sh:.4f}'

            plot_basin_3d(ax, DX, DY, norm_grid, top_title, bot_title)

        plt.tight_layout()

        # Safe filename
        safe_label = label.replace('/', '_').replace(' ', '_')
        out_path = os.path.join(OUTPUT_DIR,
                                f'basin_subset_{safe_label}_frame{FRAME_IDX}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f'  Saved: {out_path}', flush=True)

    print('\nAll done.', flush=True)


if __name__ == '__main__':
    main()