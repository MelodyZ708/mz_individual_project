"""
vis_unet_feature_maps.py
========================
Visualise U-Net Encoder shallow feature maps (P3 Zero-cost Feature Injection)
and compare them with ResNet-18 Conv1 features.

Three output figures
--------------------
1. feature_maps_unet_enc1_all32_frame{N}.png
   3 rows (Clean / +30% / +50%) × 32 columns (all U-Net enc1 channels)
   Each cell: heat-map + Kill% annotation

2. feature_maps_unet_vs_resnet_frame{N}.png
   Side-by-side comparison of U-Net enc1 (top 10 by variance) vs
   ResNet-18 Conv1 (top 10 by variance) under Clean / +30% / +50%
   Layout: 6 rows × 10 cols (3 conds × 2 networks)

3. feature_maps_unet_diff_frame{N}.png
   Difference maps (+50% − Clean) for all 32 U-Net enc1 channels

U-Net architecture recap (COMO DepthCovModule, base_feature_channels=16):
  x_enc[0]: self.base  → 16ch, H×W    (full resolution)
  x_enc[1]: down_convs[0] → 32ch, H/2×W/2  ← we visualise this

Requirements
------------
  - COMO project must be importable (run from dentro como/ directory)
  - models/scannet.ckpt must exist
  - TUM fr1/desk RGB frames at RGB_DIR

Usage
-----
  cd /path/to/como
  python ../../vis_unet_feature_maps.py
  # or override paths:
  python ../../vis_unet_feature_maps.py --rgb_dir /data/tum/fr1_desk/rgb \
      --ckpt models/scannet.ckpt --frame 306 --enc_level 1
"""

import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ── Default paths (edit to match your environment) ───────────────────────────
DEFAULT_RGB_DIR  = '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/'
DEFAULT_CKPT     = 'models/scannet.ckpt'
DEFAULT_FRAME    = 306
DEFAULT_ENC_LEVEL = 1   # 0=16ch H×W, 1=32ch H/2×W/2
DEFAULT_OUT_DIR  = 'vis_results/unet_feature_maps'

# ── Plot style (matches existing scripts) ────────────────────────────────────
plt.rcParams.update({
    'font.size':        9,
    'font.family':      'serif',
    'axes.titlesize':   9,
    'figure.dpi':       120,
    'savefig.dpi':      200,
    'mathtext.fontset': 'cm',
})

BRIGHTNESS_CONDITIONS = [
    {'factor': 0.0, 'label': 'Clean',           'row_color': '#1a1a1a'},
    {'factor': 0.3, 'label': '+30% Brightness',  'row_color': '#b35900'},
    {'factor': 0.5, 'label': '+50% Brightness',  'row_color': '#cc0000'},
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractors
# ─────────────────────────────────────────────────────────────────────────────

class UNetEncoderExtractor(nn.Module):
    """
    Loads COMO's DepthCovModule and extracts Encoder shallow features.

    enc_level=0 → x_enc[0]: self.base output, 16ch, H×W
    enc_level=1 → x_enc[1]: down_convs[0] output, 32ch, H/2×W/2
    Output is bilinearly upsampled to original resolution.
    """

    def __init__(self, ckpt_path, enc_level=1, device='cpu'):
        super().__init__()
        self.enc_level = enc_level
        self.device = device

        # Import COMO modules (must run from inside como/ project)
        try:
            from como.depth_cov.core.DepthCovModule import DepthCovModule
        except ImportError:
            raise ImportError(
                "Cannot import COMO modules. "
                "Make sure to run this script from the como/ project directory "
                "with the conda environment activated."
            )

        network_size = torch.tensor([192, 256])
        dcm = DepthCovModule.load_from_checkpoint(
            ckpt_path, train_size=network_size
        )
        dcm.eval()
        dcm.to(device)
        dcm.to(torch.float)

        self.unet = dcm.gaussian_cov_net
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False

        # Channel count for this enc_level: 16 * 2^enc_level
        self.out_channels = 16 * (2 ** enc_level)
        print(f"[UNetEncoderExtractor] enc_level={enc_level}, "
              f"out_channels={self.out_channels}, ckpt={ckpt_path}")

    @torch.no_grad()
    def forward(self, img_tensor):
        """
        img_tensor: [1, 3, H, W], float32, values in [0, 1]
        Returns: [1, C, H, W] upsampled to original resolution
        """
        orig_size = img_tensor.shape[-2:]

        # U-Net expects [192, 256] input (resize like run_model does)
        x_r = F.interpolate(img_tensor.float(), size=[192, 256],
                            mode='bilinear', align_corners=False)

        # Run Encoder only (manually, to avoid running full Decoder)
        x_norm = self.unet.normalize(x_r)

        x_enc = []
        x_enc.append(self.unet.base(x_norm))          # enc[0]: 16ch, 192×256
        for i in range(self.unet.num_levels):
            x_enc.append(self.unet.down_convs[i](x_enc[-1]))

        feat = x_enc[self.enc_level]                   # select enc level

        # Upsample back to original resolution
        feat_up = F.interpolate(feat.float(), size=orig_size,
                                mode='bilinear', align_corners=False)
        return feat_up  # [1, C, H, W]


class ResNetConv1Extractor(nn.Module):
    """ResNet-18 conv1 → bn1 → relu, upsampled to original resolution."""

    def __init__(self, device='cpu'):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = base.conv1
        self.bn1   = base.bn1
        self.relu  = nn.ReLU(inplace=False)
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.out_channels = 64
        print(f"[ResNetConv1Extractor] ResNet-18 conv1, out_channels=64")

    @torch.no_grad()
    def forward(self, img_tensor):
        orig_size = img_tensor.shape[-2:]
        x = (img_tensor.float() - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x  # [1, 64, H, W]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image_tensor(path, device):
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

def apply_brightness(tensor, factor):
    return torch.clamp(tensor + factor, 0.0, 1.0)

def kill_pct(feat_map):
    """Fraction of zero activations (dead pixels)."""
    return 100.0 * float(np.mean(feat_map == 0))

def top_n_by_variance(feat_np, n=10):
    """Return indices of top-n channels by spatial variance."""
    variances = [float(np.var(feat_np[c])) for c in range(feat_np.shape[0])]
    return np.argsort(variances)[::-1][:n].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: All 32 U-Net enc1 channels (3 rows × 32 cols)
# ─────────────────────────────────────────────────────────────────────────────

def plot_unet_all_channels(feats_dict, enc_level, frame_idx, out_dir):
    """
    3 rows (Clean / +30% / +50%) × N columns (all channels).
    Channels sorted by Clean variance (highest first).
    """
    clean_feat = feats_dict['Clean']
    n_ch = clean_feat.shape[0]
    sorted_chs = np.argsort([np.var(clean_feat[c]) for c in range(n_ch)])[::-1].tolist()

    n_cond = len(BRIGHTNESS_CONDITIONS)
    ncols = n_ch  # all channels

    fig, axes = plt.subplots(
        n_cond, ncols,
        figsize=(2.2 * ncols, 3.0 * n_cond),
        gridspec_kw={'hspace': 0.35, 'wspace': 0.06}
    )

    enc_ch_label = f"{n_ch}ch, H/{2**enc_level}×W/{2**enc_level}"
    fig.suptitle(
        f"U-Net Encoder enc{enc_level} — All {n_ch} Channels — Frame {frame_idx}\n"
        f"Extraction: COMO DepthCovModule → gaussian_cov_net.down_convs[{enc_level-1}]  "
        f"({enc_ch_label})  |  Columns sorted by Clean variance (↓)",
        fontsize=12, fontweight='bold', y=1.01
    )

    for col_i, ch in enumerate(sorted_chs):
        all_vals = np.concatenate([feats_dict[c['label']][ch].ravel()
                                   for c in BRIGHTNESS_CONDITIONS])
        vmax = float(np.percentile(all_vals, 99)) + 1e-8
        vmin = 0.0

        for row_i, cond in enumerate(BRIGHTNESS_CONDITIONS):
            ax = axes[row_i, col_i]
            fmap = feats_dict[cond['label']][ch]
            kp = kill_pct(fmap)

            ax.imshow(fmap, cmap='viridis', vmin=vmin, vmax=vmax,
                      interpolation='nearest', aspect='auto')

            if row_i == 0:
                var_val = float(np.var(clean_feat[ch]))
                ax.set_title(
                    f"Ch{ch:02d} [#{col_i+1}]\nVar={var_val:.3f}",
                    fontsize=7, color='#1a1a1a', fontweight='bold', pad=3
                )

            kp_color = 'red' if kp > 15 else ('darkorange' if kp > 5 else 'white')
            ax.text(0.03, 0.04, f"Kill%={kp:.1f}%",
                    transform=ax.transAxes, fontsize=6.5,
                    color=kp_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55))
            ax.set_xticks([])
            ax.set_yticks([])

            if col_i == 0:
                ax.set_ylabel(cond['label'], fontsize=9,
                              color=cond['row_color'], fontweight='bold', labelpad=5)

    plt.subplots_adjust(hspace=0.35, wspace=0.06, top=0.92)
    out_path = os.path.join(out_dir,
                            f'feature_maps_unet_enc{enc_level}_all{n_ch}_frame{frame_idx}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: U-Net Top-10 vs ResNet Conv1 Top-10 side-by-side
# ─────────────────────────────────────────────────────────────────────────────

def plot_unet_vs_resnet(unet_feats_dict, resnet_feats_dict,
                        unet_top10, resnet_top10,
                        enc_level, frame_idx, out_dir):
    """
    6 rows × 10 cols layout:
      Row 0: U-Net  Clean
      Row 1: U-Net  +30%
      Row 2: U-Net  +50%
      Row 3: ResNet Clean
      Row 4: ResNet +30%
      Row 5: ResNet +50%
    """
    n_top = len(unet_top10)
    assert len(resnet_top10) == n_top

    n_rows = 6  # 3 conds × 2 networks
    fig, axes = plt.subplots(
        n_rows, n_top,
        figsize=(2.6 * n_top, 3.0 * n_rows),
        gridspec_kw={'hspace': 0.38, 'wspace': 0.07}
    )

    fig.suptitle(
        f"U-Net enc{enc_level} (Top-10 by Var)  vs  ResNet-18 Conv1 (Top-10 by Var)\n"
        f"Frame {frame_idx}  |  Clean / +30% / +50% Brightness",
        fontsize=13, fontweight='bold', y=1.01
    )

    # Row metadata
    row_meta = []
    for cond in BRIGHTNESS_CONDITIONS:
        row_meta.append({'net': 'UNet',   'cond': cond})
    for cond in BRIGHTNESS_CONDITIONS:
        row_meta.append({'net': 'ResNet', 'cond': cond})

    row_labels = [
        ('U-Net\nClean',    '#1565C0'),
        ('U-Net\n+30%',     '#1976D2'),
        ('U-Net\n+50%',     '#42A5F5'),
        ('ResNet\nClean',   '#B71C1C'),
        ('ResNet\n+30%',    '#E53935'),
        ('ResNet\n+50%',    '#EF9A9A'),
    ]

    for row_i, meta in enumerate(row_meta):
        net  = meta['net']
        cond = meta['cond']
        chs  = unet_top10 if net == 'UNet' else resnet_top10
        fd   = unet_feats_dict if net == 'UNet' else resnet_feats_dict

        for col_i, ch in enumerate(chs):
            ax = axes[row_i, col_i]
            fmap = fd[cond['label']][ch]
            kp = kill_pct(fmap)

            # Shared colour scale per (net, channel) across conditions
            all_vals = np.concatenate([
                fd[c['label']][ch].ravel() for c in BRIGHTNESS_CONDITIONS
            ])
            vmax = float(np.percentile(all_vals, 99)) + 1e-8

            cmap = 'plasma' if net == 'UNet' else 'viridis'
            ax.imshow(fmap, cmap=cmap, vmin=0, vmax=vmax,
                      interpolation='nearest', aspect='auto')

            # Column header (top row of each network block)
            if row_i == 0 and net == 'UNet':
                var_val = float(np.var(unet_feats_dict['Clean'][ch]))
                ax.set_title(f"UCh{ch:02d}\nVar={var_val:.3f}",
                             fontsize=8, color='#1565C0', fontweight='bold', pad=3)
            elif row_i == 3 and net == 'ResNet':
                var_val = float(np.var(resnet_feats_dict['Clean'][ch]))
                ax.set_title(f"RCh{ch:02d}\nVar={var_val:.4f}",
                             fontsize=8, color='#B71C1C', fontweight='bold', pad=3)

            kp_color = 'red' if kp > 15 else ('darkorange' if kp > 5 else 'white')
            ax.text(0.03, 0.04, f"Kill%={kp:.1f}%",
                    transform=ax.transAxes, fontsize=6.5,
                    color=kp_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55))
            ax.set_xticks([])
            ax.set_yticks([])

            if col_i == 0:
                label, color = row_labels[row_i]
                ax.set_ylabel(label, fontsize=9, color=color,
                              fontweight='bold', labelpad=5)

    # Horizontal separator line between U-Net and ResNet blocks
    fig.add_artist(plt.Line2D(
        [0.01, 0.99], [0.505, 0.505],
        transform=fig.transFigure,
        color='gray', linewidth=1.5, linestyle='--'
    ))

    plt.subplots_adjust(hspace=0.38, wspace=0.07, top=0.93)
    out_path = os.path.join(out_dir,
                            f'feature_maps_unet_vs_resnet_frame{frame_idx}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Difference maps (+50% − Clean) for all U-Net channels
# ─────────────────────────────────────────────────────────────────────────────

def plot_unet_diff_maps(feats_dict, enc_level, frame_idx, out_dir):
    clean_feat = feats_dict['Clean']
    n_ch = clean_feat.shape[0]
    sorted_chs = np.argsort([np.var(clean_feat[c]) for c in range(n_ch)])[::-1].tolist()

    fig, axes = plt.subplots(
        2, n_ch,
        figsize=(2.2 * n_ch, 4.8),
        gridspec_kw={'hspace': 0.35, 'wspace': 0.06}
    )

    fig.suptitle(
        f"U-Net enc{enc_level} — Difference Maps (+50% − Clean) — Frame {frame_idx}\n"
        f"Row 1: |Δ| (absolute)   Row 2: Δ (signed, red=increase, blue=decrease)",
        fontsize=12, fontweight='bold', y=1.01
    )

    for col_i, ch in enumerate(sorted_chs):
        f_clean  = feats_dict['Clean'][ch].astype(np.float64)
        f_bright = feats_dict['+50% Brightness'][ch].astype(np.float64)
        diff     = f_bright - f_clean
        abs_diff = np.abs(diff)

        # Row 1: absolute difference
        ax1 = axes[0, col_i]
        vmax_abs = float(np.percentile(abs_diff, 99)) + 1e-8
        ax1.imshow(abs_diff, cmap='hot', vmin=0, vmax=vmax_abs,
                   interpolation='nearest', aspect='auto')
        mean_abs = float(np.mean(abs_diff))
        ax1.text(0.03, 0.04, f"μ|Δ|={mean_abs:.3f}",
                 transform=ax1.transAxes, fontsize=6.5, color='white',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.55))
        ax1.set_title(f"Ch{ch:02d}", fontsize=7.5, fontweight='bold', pad=3)
        ax1.set_xticks([]); ax1.set_yticks([])
        if col_i == 0:
            ax1.set_ylabel('|+50%−Clean|', fontsize=9, fontweight='bold',
                           color='#8B0000', labelpad=5)

        # Row 2: signed difference
        ax2 = axes[1, col_i]
        vmax_s = float(max(np.percentile(np.abs(diff), 99), 1e-8))
        ax2.imshow(diff, cmap='RdBu_r', vmin=-vmax_s, vmax=vmax_s,
                   interpolation='nearest', aspect='auto')
        mean_s = float(np.mean(diff))
        ax2.text(0.03, 0.04, f"μΔ={mean_s:+.3f}",
                 transform=ax2.transAxes, fontsize=6.5, color='black',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.6))
        ax2.set_xticks([]); ax2.set_yticks([])
        if col_i == 0:
            ax2.set_ylabel('+50%−Clean\n(signed)', fontsize=9,
                           fontweight='bold', color='#00008B', labelpad=5)

    plt.subplots_adjust(hspace=0.35, wspace=0.06, top=0.90)
    out_path = os.path.join(out_dir,
                            f'feature_maps_unet_enc{enc_level}_diff_frame{frame_idx}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Per-channel Kill% bar chart (U-Net vs ResNet comparison)
# ─────────────────────────────────────────────────────────────────────────────

def plot_kill_comparison(unet_feats_dict, resnet_feats_dict,
                         unet_top10, resnet_top10,
                         enc_level, frame_idx, out_dir):
    """
    Bar chart comparing Kill% under +50% brightness for
    U-Net top-10 channels vs ResNet top-10 channels.
    """
    cond_label = '+50% Brightness'
    unet_kills  = [kill_pct(unet_feats_dict[cond_label][ch])  for ch in unet_top10]
    resnet_kills = [kill_pct(resnet_feats_dict[cond_label][ch]) for ch in resnet_top10]

    x = np.arange(10)
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    bars1 = ax.bar(x - width/2, unet_kills,  width,
                   label=f'U-Net enc{enc_level} (Top-10 by Var)', color='#1565C0', alpha=0.85)
    bars2 = ax.bar(x + width/2, resnet_kills, width,
                   label='ResNet-18 Conv1 (Top-10 by Var)', color='#C62828', alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=8, color='#1565C0')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%', ha='center', va='bottom',
                fontsize=8, color='#C62828')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"U:{uc:02d}\nR:{rc:02d}" for uc, rc in zip(unet_top10, resnet_top10)],
        fontsize=9
    )
    ax.set_ylabel('Kill% at +50% Brightness', fontsize=11)
    ax.set_title(
        f'Kill% Comparison: U-Net enc{enc_level} vs ResNet-18 Conv1 — Frame {frame_idx}',
        fontsize=12, fontweight='bold'
    )
    ax.axhline(y=15, color='red', linestyle='--', linewidth=1.2, label='15% threshold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(unet_kills + resnet_kills) * 1.2, 20))

    plt.subplots_adjust(top=0.90)
    out_path = os.path.join(out_dir,
                            f'kill_comparison_unet_vs_resnet_frame{frame_idx}.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualise U-Net encoder feature maps')
    parser.add_argument('--rgb_dir',   default=DEFAULT_RGB_DIR)
    parser.add_argument('--ckpt',      default=DEFAULT_CKPT)
    parser.add_argument('--frame',     type=int, default=DEFAULT_FRAME)
    parser.add_argument('--enc_level', type=int, default=DEFAULT_ENC_LEVEL,
                        help='0=16ch H×W, 1=32ch H/2×W/2')
    parser.add_argument('--out_dir',   default=DEFAULT_OUT_DIR)
    parser.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {args.device}")
    print(f"Frame:  {args.frame}")
    print(f"enc_level: {args.enc_level}")
    print(f"Output: {args.out_dir}")

    # ── Load image ────────────────────────────────────────────────────────────
    all_imgs = sorted(glob.glob(os.path.join(args.rgb_dir, '*.png')))
    if not all_imgs:
        # Also try .jpg
        all_imgs = sorted(glob.glob(os.path.join(args.rgb_dir, '*.jpg')))
    if not all_imgs:
        raise FileNotFoundError(f"No images found in {args.rgb_dir}")
    if args.frame >= len(all_imgs):
        print(f"[WARN] Frame {args.frame} out of range ({len(all_imgs)} total), using last frame.")
        args.frame = len(all_imgs) - 1
    img_path = all_imgs[args.frame]
    print(f"Image: {os.path.basename(img_path)}")

    # ── Build extractors ──────────────────────────────────────────────────────
    print("\nLoading U-Net extractor (COMO DepthCovModule)...")
    unet_extractor = UNetEncoderExtractor(
        ckpt_path=args.ckpt,
        enc_level=args.enc_level,
        device=args.device
    )

    print("Loading ResNet-18 Conv1 extractor...")
    resnet_extractor = ResNetConv1Extractor(device=args.device)

    # ── Extract features for all brightness conditions ────────────────────────
    print("\nExtracting features...")
    unet_feats_dict   = {}
    resnet_feats_dict = {}

    for cond in BRIGHTNESS_CONDITIONS:
        t = load_image_tensor(img_path, args.device)
        t = apply_brightness(t, cond['factor'])

        unet_feat   = unet_extractor(t)[0].cpu().numpy()    # [C_u, H, W]
        resnet_feat = resnet_extractor(t)[0].cpu().numpy()  # [64, H, W]

        # Apply ReLU-like clamp (U-Net uses LeakyReLU, so values can be slightly negative)
        # For Kill% consistency, clamp to 0 (same convention as ResNet ReLU output)
        unet_feat   = np.maximum(unet_feat, 0)

        unet_feats_dict[cond['label']]   = unet_feat
        resnet_feats_dict[cond['label']] = resnet_feat
        print(f"  [{cond['label']}] U-Net shape={unet_feat.shape}, "
              f"ResNet shape={resnet_feat.shape}")

    # ── Select top-10 channels by Clean variance ─────────────────────────────
    unet_top10   = top_n_by_variance(unet_feats_dict['Clean'],   n=10)
    resnet_top10 = top_n_by_variance(resnet_feats_dict['Clean'], n=10)
    print(f"\nU-Net   top-10 channels (by var): {unet_top10}")
    print(f"ResNet  top-10 channels (by var): {resnet_top10}")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # Plot 1: All U-Net channels
    plot_unet_all_channels(unet_feats_dict, args.enc_level, args.frame, args.out_dir)

    # Plot 2: U-Net vs ResNet side-by-side
    plot_unet_vs_resnet(unet_feats_dict, resnet_feats_dict,
                        unet_top10, resnet_top10,
                        args.enc_level, args.frame, args.out_dir)

    # Plot 3: Difference maps
    plot_unet_diff_maps(unet_feats_dict, args.enc_level, args.frame, args.out_dir)

    # Plot 4: Kill% comparison bar chart
    plot_kill_comparison(unet_feats_dict, resnet_feats_dict,
                         unet_top10, resnet_top10,
                         args.enc_level, args.frame, args.out_dir)

    print(f"\nAll outputs saved to: {args.out_dir}/")
    print(f"  feature_maps_unet_enc{args.enc_level}_all{unet_extractor.out_channels}_frame{args.frame}.png")
    print(f"  feature_maps_unet_vs_resnet_frame{args.frame}.png")
    print(f"  feature_maps_unet_enc{args.enc_level}_diff_frame{args.frame}.png")
    print(f"  kill_comparison_unet_vs_resnet_frame{args.frame}.png")


if __name__ == '__main__':
    main()