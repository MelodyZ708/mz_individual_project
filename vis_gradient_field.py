"""
vis_gradient_field.py
=====================
Visualize gradient vector fields of the optimal 4-channel subset [06, 28, 34, 62]
under Clean conditions, to demonstrate complementary gradient directions.

Layout: 1 row x 5 columns
  Columns 1-4: individual channels (gradient magnitude + quiver)
  Column 5:    combined (sum of all 4 channels' gradients)

Output: vis_results/forward_greedy_bqs/analysis/gradient_field_optimal4_frame306.png
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# ── Config ──────────────────────────────────────────────────────────────────
RGB_DIR   = '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/'
FRAME_IDX = 306
OUT_DIR   = 'vis_results/forward_greedy_bqs/analysis'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

CHANNELS = [6, 28, 34, 62]
TITLES   = ["Ch 06 (Core)", "Ch 28 (Robust)", "Ch 34 (Comp.)", "Ch 62 (Comp.)"]
COLORS   = ["red", "darkorange", "green", "gray"]

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

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_image_tensor(path, device):
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)

def compute_gradient_field(feat_map):
    """feat_map: [H, W] float32 numpy. Returns (gx, gy, magnitude)."""
    f32 = feat_map.astype(np.float32)
    gx  = cv2.Sobel(f32, cv2.CV_64F, 1, 0, ksize=3)
    gy  = cv2.Sobel(f32, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, mag

def plot_channel_grad(ax, feat_map, title, color, quiver_step=20):
    gx, gy, mag = compute_gradient_field(feat_map)
    ax.imshow(mag, cmap='inferno')
    H, W = feat_map.shape
    Yq, Xq = np.mgrid[quiver_step//2:H:quiver_step, quiver_step//2:W:quiver_step]
    # Normalise arrows for display
    gxq = gx[Yq, Xq]
    gyq = gy[Yq, Xq]
    norm = np.sqrt(gxq**2 + gyq**2 + 1e-8)
    ax.quiver(Xq, Yq, gxq/norm, gyq/norm,
              color='white', alpha=0.7, scale=30, width=0.003)
    ax.set_title(title, color=color, fontsize=13, fontweight='bold', pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    return gx, gy

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
        clean_feat = extractor(clean_t)[0].cpu().numpy()  # [64, H, W]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(
        f"Gradient Vector Field (Clean) — Frame {FRAME_IDX}\n"
        f"Complementary gradient directions → isotropic combined field",
        fontsize=15, fontweight='bold'
    )

    total_gx = np.zeros_like(clean_feat[0], dtype=np.float64)
    total_gy = np.zeros_like(clean_feat[0], dtype=np.float64)

    for i, ch in enumerate(CHANNELS):
        gx, gy = plot_channel_grad(axes[i], clean_feat[ch], TITLES[i], COLORS[i])
        total_gx += gx
        total_gy += gy

    # Combined panel
    ax = axes[4]
    total_mag = np.sqrt(total_gx**2 + total_gy**2)
    ax.imshow(total_mag, cmap='inferno')
    H, W = total_mag.shape
    step = 20
    Yq, Xq = np.mgrid[step//2:H:step, step//2:W:step]
    gxq = total_gx[Yq, Xq]; gyq = total_gy[Yq, Xq]
    norm = np.sqrt(gxq**2 + gyq**2 + 1e-8)
    ax.quiver(Xq, Yq, gxq/norm, gyq/norm,
              color='white', alpha=0.7, scale=30, width=0.003)
    ax.set_title("Combined (All 4 Chs)", color='blue', fontsize=13, fontweight='bold', pad=8)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'gradient_field_optimal4_frame{FRAME_IDX}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()