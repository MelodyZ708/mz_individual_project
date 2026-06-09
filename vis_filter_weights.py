"""
vis_filter_weights.py
=====================
Visualise Conv1 filter weights (7×7×3) for the optimal 4-channel subset
[06, 28, 34, 62], following the style of the existing Rank01 filter-weight plot.

Layout per channel (one column):
  Row 1  : RGB composite thumbnail  (7×7, colour)
  Row 2-4: R / G / B weight maps    (7×7, RdBu diverging colourmap)
  Footer : role label + Kill% stats

Output: vis_results/forward_greedy_bqs/analysis/filter_weights_optimal4.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import glob
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────────
RGB_DIR   = '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/'
FRAME_IDX = 306
OUT_DIR   = 'vis_results/forward_greedy_bqs/analysis'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

CHANNELS = [6, 28, 34, 62]
ROLES    = [
    ("CORE ANCHOR",    "red",        "Brightness-robust, low-freq structure"),
    ("ROBUST ANCHOR",  "darkorange", "Brightness-robust, reinforces Ch 06"),
    ("COMPLEMENTARY",  "green",      "Edge detector, widens basin Width"),
    ("COMPLEMENTARY",  "gray",       "Colour-contrast, dead at +50% brightness"),
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def load_conv1_weights():
    """Return Conv1 weight tensor [64, 3, 7, 7] from pretrained ResNet18."""
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    w = base.conv1.weight.detach().cpu().numpy()   # [64, 3, 7, 7]
    return w

def make_rgb_composite(w_ch):
    """
    w_ch: [3, 7, 7] raw filter weights.
    Returns a [7, 7, 3] uint8 RGB image for display.
    """
    rgb = w_ch.transpose(1, 2, 0)   # [7, 7, 3]
    lo, hi = rgb.min(), rgb.max()
    rgb_norm = (rgb - lo) / (hi - lo + 1e-8)
    return rgb_norm   # float [0,1]

def compute_kill_pct(feat_map):
    return 100.0 * np.mean(feat_map == 0)

def load_feats(device):
    """Extract feature maps for frame FRAME_IDX under Clean and +50%."""
    import torch.nn.functional as F

    class Conv1BNReLUExtractor(nn.Module):
        def __init__(self, device):
            super().__init__()
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.conv1 = base.conv1
            self.bn1   = base.bn1
            self.relu  = nn.ReLU(inplace=False)
            self.to(device); self.eval()
            for p in self.parameters(): p.requires_grad = False
            self.mean = torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
            self.std  = torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)

        @torch.no_grad()
        def forward(self, x):
            sz = x.shape[-2:]
            x = (x - self.mean) / self.std
            x = self.relu(self.bn1(self.conv1(x)))
            return F.interpolate(x, size=sz, mode='bilinear', align_corners=False)

    all_imgs = sorted(glob.glob(os.path.join(RGB_DIR, '*.png')))
    img = Image.open(all_imgs[FRAME_IDX]).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)

    ext = Conv1BNReLUExtractor(device)
    with torch.no_grad():
        f_clean  = ext(t)[0].cpu().numpy()
        f_bright = ext(torch.clamp(t + 0.5, 0, 1))[0].cpu().numpy()
    return f_clean, f_bright

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    weights  = load_conv1_weights()          # [64, 3, 7, 7]
    print("Loading feature maps …")
    f_clean, f_bright = load_feats(DEVICE)

    n_ch = len(CHANNELS)
    # Each column: 1 RGB composite + 3 R/G/B maps + 1 footer text
    # We use GridSpec: 4 image rows + 1 text row
    fig = plt.figure(figsize=(5 * n_ch, 14))
    fig.patch.set_facecolor('#1a1a2e')

    outer = gridspec.GridSpec(1, n_ch, figure=fig, hspace=0.05, wspace=0.08)

    channel_labels = ['R', 'G', 'B']
    rdbu = plt.cm.RdBu

    for col_idx, ch in enumerate(CHANNELS):
        role_label, role_color, role_desc = ROLES[col_idx]
        w_ch = weights[ch]          # [3, 7, 7]

        inner = gridspec.GridSpecFromSubplotSpec(
            5, 1, subplot_spec=outer[col_idx],
            height_ratios=[3, 1, 1, 1, 0.7],
            hspace=0.08
        )

        # ── Row 0: RGB composite ──────────────────────────────────────────
        ax_rgb = fig.add_subplot(inner[0])
        rgb_img = make_rgb_composite(w_ch)
        ax_rgb.imshow(rgb_img, interpolation='nearest', aspect='equal')
        ax_rgb.set_xticks([]); ax_rgb.set_yticks([])
        for spine in ax_rgb.spines.values():
            spine.set_edgecolor(role_color); spine.set_linewidth(3)

        # Column title
        kp_clean  = compute_kill_pct(f_clean[ch])
        kp_bright = compute_kill_pct(f_bright[ch])
        ax_rgb.set_title(
            f"Ch {ch:02d}",
            color=role_color, fontsize=18, fontweight='bold', pad=10
        )

        # ── Rows 1-3: R / G / B weight maps ──────────────────────────────
        for row_idx, c_name in enumerate(channel_labels):
            ax = fig.add_subplot(inner[row_idx + 1])
            w_slice = w_ch[row_idx]   # [7, 7]
            vabs = np.abs(w_slice).max()
            ax.imshow(w_slice, cmap='RdBu', vmin=-vabs, vmax=vabs,
                      interpolation='nearest', aspect='equal')
            ax.set_ylabel(c_name, color='white', fontsize=11,
                          fontweight='bold', rotation=0, labelpad=12, va='center')
            ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#444466')

        # ── Row 4: footer text ────────────────────────────────────────────
        ax_txt = fig.add_subplot(inner[4])
        ax_txt.axis('off')
        ax_txt.text(0.5, 0.75, role_label,
                    ha='center', va='center', fontsize=12,
                    fontweight='bold', color=role_color,
                    transform=ax_txt.transAxes)
        ax_txt.text(0.5, 0.35,
                    f"Kill% Clean={kp_clean:.1f}%  |  +50%={kp_bright:.1f}%",
                    ha='center', va='center', fontsize=9,
                    color='white' if kp_bright < 90 else 'tomato',
                    transform=ax_txt.transAxes)
        ax_txt.text(0.5, 0.05, role_desc,
                    ha='center', va='center', fontsize=8,
                    color='#aaaacc', style='italic',
                    transform=ax_txt.transAxes)

    # ── Global title ──────────────────────────────────────────────────────
    fig.text(0.5, 0.97,
             "Conv1 Filter Weights (7×7×3) — Optimal 4-Channel Subset [06, 28, 34, 62]",
             ha='center', va='top', fontsize=16, fontweight='bold', color='white')
    fig.text(0.5, 0.945,
             "Top: RGB composite  |  Middle: R/G/B weight maps (RdBu)  |  Bottom: Role & Kill%",
             ha='center', va='top', fontsize=11, color='#aaaacc')

    # ── Legend ────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor='red',        label='Core Anchor — brightness-robust, low-freq'),
        Patch(facecolor='darkorange', label='Robust Anchor — reinforces core'),
        Patch(facecolor='green',      label='Complementary — edge detector, widens Width'),
        Patch(facecolor='gray',       label='Complementary — colour-contrast, dead at +50%'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, framealpha=0.3,
               facecolor='#1a1a2e', edgecolor='#444466',
               labelcolor='white', bbox_to_anchor=(0.5, 0.0))

    plt.savefig(
        os.path.join(OUT_DIR, 'filter_weights_optimal4.png'),
        dpi=150, bbox_inches='tight',
        facecolor=fig.get_facecolor()
    )
    print(f"Saved: {OUT_DIR}/filter_weights_optimal4.png")

if __name__ == '__main__':
    main()