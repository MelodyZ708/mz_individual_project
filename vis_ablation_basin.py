"""
vis_ablation_basin.py
=====================
Ablation analysis of convergence basin for the optimal 4-channel subset.
Shows how each additional complementary channel widens the basin.

Subsets evaluated (Clean condition, Frame 306):
  Step 1: [06]
  Step 2: [06, 28]
  Step 3: [06, 28, 34]
  Step 4: [06, 28, 34, 62]  ← optimal

Also shows individual basins for Ch 34 and Ch 62 alone (to contrast with combined).

Output: vis_results/forward_greedy_bqs/analysis/ablation_basin_optimal4_frame306.png
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
RGB_DIR    = '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/'
FRAME_IDX  = 306
OUT_DIR    = 'vis_results/forward_greedy_bqs/analysis'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
GRID_RANGE = 30
GRID_STEP  = 1
CHUNK_SIZE = 512

# ── Model ────────────────────────────────────────────────────────────────────
class Conv1BNReLUExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = base.conv1
        self.bn1   = base.bn1
        self.relu  = nn.ReLU(inplace=False)
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
        return x

# ── GPU cost landscape ───────────────────────────────────────────────────────
def build_shift_grids(H, W, dx_vals, dy_vals, device):
    """Pre-build all shift sampling grids on GPU. Returns [N, H, W, 2]."""
    grids = []
    base_x = torch.linspace(-1, 1, W, device=device)
    base_y = torch.linspace(-1, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing='ij')
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

    for dy in dy_vals:
        for dx in dx_vals:
            shift_x = 2.0 * dx / W
            shift_y = 2.0 * dy / H
            g = base_grid.clone()
            g[..., 0] -= shift_x
            g[..., 1] -= shift_y
            grids.append(g)
    return torch.stack(grids, dim=0)  # [N, H, W, 2]

@torch.no_grad()
def compute_cost_landscape_gpu(feat_ref, feat_tgt, subset, device,
                                grid_range=GRID_RANGE, grid_step=GRID_STEP,
                                chunk_size=CHUNK_SIZE):
    """
    feat_ref, feat_tgt: [1, 64, H, W] tensors on device.
    subset: list of channel indices.
    Returns: [len_dy, len_dx] numpy cost grid.
    """
    dx_vals = list(range(-grid_range, grid_range + 1, grid_step))
    dy_vals = list(range(-grid_range, grid_range + 1, grid_step))
    N  = len(dx_vals) * len(dy_vals)
    H, W = feat_ref.shape[-2:]

    # Select channels
    ref_ch = feat_ref[0, subset, :, :]  # [C, H, W]
    tgt_ch = feat_tgt[0, subset, :, :]  # [C, H, W]

    # Build shift grids
    shift_grids = build_shift_grids(H, W, dx_vals, dy_vals, device)  # [N, H, W, 2]

    costs = torch.zeros(N, device=device)
    tgt_4d = tgt_ch.unsqueeze(0).expand(chunk_size, -1, -1, -1)  # will be re-sliced

    for start in range(0, N, chunk_size):
        end  = min(start + chunk_size, N)
        bs   = end - start
        g    = shift_grids[start:end]                          # [bs, H, W, 2]
        tgt_b = tgt_ch.unsqueeze(0).expand(bs, -1, -1, -1)    # [bs, C, H, W]
        shifted = F.grid_sample(tgt_b, g, mode='bilinear',
                                padding_mode='border', align_corners=True)  # [bs, C, H, W]
        ref_b = ref_ch.unsqueeze(0).expand(bs, -1, -1, -1)
        diff  = shifted - ref_b
        costs[start:end] = diff.pow(2).mean(dim=(1, 2, 3))
        del shifted, ref_b, diff, tgt_b, g
        torch.cuda.empty_cache()

    cost_np = costs.cpu().numpy().reshape(len(dy_vals), len(dx_vals))
    return cost_np

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_image_tensor(path, device):
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)

def plot_3d_basin(ax, cost_grid, title, title_color='black'):
    H, W = cost_grid.shape
    dx_vals = np.arange(W) - W // 2
    dy_vals = np.arange(H) - H // 2
    X, Y = np.meshgrid(dx_vals, dy_vals)
    norm = (cost_grid - cost_grid.min()) / (cost_grid.max() - cost_grid.min() + 1e-8)

    ax.plot_surface(X, Y, norm, cmap='YlOrRd', edgecolor='none', alpha=0.9)
    ax.contourf(X, Y, norm, zdir='z', offset=0, cmap='gray', alpha=0.4)
    ax.set_zlim(0, 1.05)
    ax.set_xlabel('$\\Delta x$ [px]', fontsize=8)
    ax.set_ylabel('$\\Delta y$ [px]', fontsize=8)
    ax.set_zlabel('Norm. Cost', fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold', color=title_color, pad=6)
    ax.view_init(elev=30, azim=-45)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = DEVICE
    print(f"Device: {device}")

    all_imgs = sorted(glob.glob(os.path.join(RGB_DIR, '*.png')))
    img_path = all_imgs[FRAME_IDX]
    print(f"Frame {FRAME_IDX}: {os.path.basename(img_path)}")

    clean_t = load_image_tensor(img_path, device)
    extractor = Conv1BNReLUExtractor(device=device)
    with torch.no_grad():
        feat = extractor(clean_t)  # [1, 64, H, W]

    # Ablation subsets + individual "weak" channels for contrast
    subsets = [
        ([6],             "Step 1: [06]\n(Core only)",              'red'),
        ([6, 28],         "Step 2: [06, 28]\n(+Robust)",            'darkorange'),
        ([6, 28, 34],     "Step 3: [06, 28, 34]\n(+Comp. 34)",      'green'),
        ([6, 28, 34, 62], "Step 4: [06, 28, 34, 62]\n(Optimal 4ch)","blue"),
        ([34],            "Individual Ch 34\n(alone)",               'gray'),
        ([62],            "Individual Ch 62\n(alone)",               'gray'),
    ]

    fig = plt.figure(figsize=(24, 5))
    fig.suptitle(
        f"Ablation Analysis — Convergence Basin (Clean) — Frame {FRAME_IDX}\n"
        f"How complementary channels widen the basin",
        fontsize=15, fontweight='bold'
    )

    for i, (subset, title, color) in enumerate(subsets):
        print(f"  Computing basin for subset {subset} ...", flush=True)
        cost_grid = compute_cost_landscape_gpu(feat, feat, subset, device)
        ax = fig.add_subplot(1, len(subsets), i+1, projection='3d')
        plot_3d_basin(ax, cost_grid, title, title_color=color)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'ablation_basin_optimal4_frame{FRAME_IDX}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()