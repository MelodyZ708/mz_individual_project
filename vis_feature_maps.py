"""
vis_feature_maps.py
===================
Visualize feature maps of the optimal 4-channel subset [06, 28, 34, 62]
under Clean and +50% brightness conditions.

Layout: 2 rows x 4 columns
  Row 1: Clean
  Row 2: +50% Brightness
  Each column: one channel

Output: vis_results/forward_greedy_bqs/analysis/feature_maps_optimal4_frame306.png
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

# ── Config ──────────────────────────────────────────────────────────────────
RGB_DIR   = '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/'
FRAME_IDX = 306
OUT_DIR   = 'vis_results/forward_greedy_bqs/analysis'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

# Best 4-channel subset from greedy search
CHANNELS = [6, 28, 34, 62]
LABELS   = [
    "Ch 06\n(Core Anchor)",
    "Ch 28\n(Robust Anchor)",
    "Ch 34\n(Complementary)",
    "Ch 62\n(Dead at +50%)"
]
LABEL_COLORS = ["red", "darkorange", "green", "gray"]

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
        return x  # [1, 64, H, W]

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_image_tensor(path, device):
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)

def apply_brightness(tensor, factor):
    return torch.clamp(tensor + factor, 0.0, 1.0)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = DEVICE
    print(f"Device: {device}")

    # Load image
    all_imgs = sorted(glob.glob(os.path.join(RGB_DIR, '*.png')))
    img_path = all_imgs[FRAME_IDX]
    print(f"Frame {FRAME_IDX}: {os.path.basename(img_path)}")

    clean_t  = load_image_tensor(img_path, device)
    bright_t = apply_brightness(clean_t, 0.5)

    # Extract features
    extractor = Conv1BNReLUExtractor(device=device)
    with torch.no_grad():
        clean_feat  = extractor(clean_t)[0].cpu().numpy()   # [64, H, W]
        bright_feat = extractor(bright_t)[0].cpu().numpy()  # [64, H, W]

    # Compute Kill% for each channel
    def kill_pct(feat_map):
        return 100.0 * np.mean(feat_map == 0)

    # Plot
    fig, axes = plt.subplots(2, len(CHANNELS), figsize=(5 * len(CHANNELS), 9))
    fig.suptitle(
        f"Optimal 4-Channel Subset — Feature Maps — Frame {FRAME_IDX}\n"
        f"Clean (top) vs +50% Brightness (bottom) · same color scale per channel",
        fontsize=15, fontweight='bold', y=1.01
    )

    for i, ch in enumerate(CHANNELS):
        f_clean  = clean_feat[ch]
        f_bright = bright_feat[ch]

        # Shared scale per channel
        vmax = max(f_clean.max(), f_bright.max())
        vmin = 0.0

        # Clean row
        ax = axes[0, i]
        ax.imshow(f_clean, cmap='viridis', vmin=vmin, vmax=vmax)
        kp = kill_pct(f_clean)
        ax.set_title(LABELS[i], color=LABEL_COLORS[i], fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel(f"Kill%={kp:.1f}%", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # Bright row
        ax = axes[1, i]
        ax.imshow(f_bright, cmap='viridis', vmin=vmin, vmax=vmax)
        kp_b = kill_pct(f_bright)
        ax.set_xlabel(f"Kill%={kp_b:.1f}%", fontsize=10, color='red' if kp_b > 80 else 'black')
        ax.set_xticks([]); ax.set_yticks([])

    # Row labels on the left
    fig.text(0.01, 0.72, "Clean",          va='center', rotation='vertical', fontsize=13, fontweight='bold')
    fig.text(0.01, 0.28, "+50% Brightness", va='center', rotation='vertical', fontsize=13, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f'feature_maps_optimal4_frame{FRAME_IDX}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()