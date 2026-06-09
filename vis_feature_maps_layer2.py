"""
vis_feature_maps_layer2.py
==========================
Stage 3: Visualise feature maps for the Top-10 Layer2 channels under
Clean / +30% / +50% brightness conditions.

Layout: 3 rows × 10 columns
  Row 1: Clean
  Row 2: +30% Brightness
  Row 3: +50% Brightness
  Each column: one Top-10 channel (ranked by BQS, left = #1)

Additional outputs:
  - Difference maps (+50% − Clean) for each channel
  - Per-channel activation statistics panel

Extraction path: conv1 → bn1 → relu → maxpool → layer1 → layer2
Output channel count: 128

Stage 3 Top-10 channels (by BQS, from channel_ranking.csv):
  Rank  Ch   BQS    Kill%@+50%
    1   39  0.6745   1.7%
    2   66  0.6417   0.1%
    3  120  0.6384  11.5%
    4   58  0.6322   1.4%
    5   81  0.6320   6.7%
    6   18  0.6240  62.7%  ← high Kill%
    7  106  0.6056   4.2%
    8   43  0.6016   0.5%
    9  123  0.5990   2.4%
   10   40  0.5987   7.2%

Output:
  vis_results/feature_maps_layer2/feature_maps_layer2_top10_frame306.png
  vis_results/feature_maps_layer2/feature_maps_layer2_top10_diff_frame306.png
  vis_results/feature_maps_layer2/feature_maps_layer2_top10_stats_frame306.png
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

# ── Config ───────────────────────────────────────────────────────────────────
RGB_DIR   = '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/'
FRAME_IDX = 306
OUT_DIR   = 'vis_results/feature_maps_layer2'
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

# Stage 3 Top-10 channels (BQS rank order)
TOP10_CHANNELS = [39, 66, 120, 58, 81, 18, 106, 43, 123, 40]
TOP10_BQS      = [0.6745, 0.6417, 0.6384, 0.6322, 0.6320,
                  0.6240, 0.6056, 0.6016, 0.5990, 0.5987]
TOP10_KILL50   = [1.7, 0.1, 11.5, 1.4, 6.7, 62.7, 4.2, 0.5, 2.4, 7.2]

BRIGHTNESS_CONDITIONS = [
    {'factor': 0.0, 'label': 'Clean',           'row_color': '#1a1a1a'},
    {'factor': 0.3, 'label': '+30% Brightness',  'row_color': '#b35900'},
    {'factor': 0.5, 'label': '+50% Brightness',  'row_color': '#cc0000'},
]

plt.rcParams.update({
    'font.size':        9,
    'font.family':      'serif',
    'axes.titlesize':   9,
    'figure.dpi':       120,
    'savefig.dpi':      200,
    'mathtext.fontset': 'cm',
})


# ── Feature Extractor ────────────────────────────────────────────────────────
class Layer2Extractor(nn.Module):
    """conv1 → bn1 → relu → maxpool → layer1 → layer2, upsampled to original resolution."""

    def __init__(self, device='cuda'):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = nn.ReLU(inplace=False)
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2   # 64ch → 128ch, stride=2
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, img_tensor):
        orig_size = img_tensor.shape[-2:]
        x = (img_tensor - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x  # [1, 128, H, W]


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_image_tensor(path, device):
    img = Image.open(path).convert('RGB')
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

def apply_brightness(tensor, factor):
    return torch.clamp(tensor + factor, 0.0, 1.0)

def kill_pct(feat_map):
    return 100.0 * float(np.mean(feat_map == 0))


# ── Plot 1: Feature Maps (3 rows × 10 cols) ──────────────────────────────────
def plot_feature_maps(feats_dict, channels, bqs_list, kill50_list,
                      frame_idx, out_dir):
    n_ch   = len(channels)
    n_cond = len(BRIGHTNESS_CONDITIONS)

    fig, axes = plt.subplots(
        n_cond, n_ch,
        figsize=(2.8 * n_ch, 3.2 * n_cond),
        gridspec_kw={'hspace': 0.35, 'wspace': 0.08}
    )

    fig.suptitle(
        f"Stage 3 (Layer2) — Top-10 Channel Feature Maps — Frame {frame_idx}\n"
        f"Extraction: conv1→bn1→relu→maxpool→layer1→layer2  |  Columns sorted by BQS (↓)",
        fontsize=13, fontweight='bold', y=1.01
    )

    for col_i, (ch, bqs, k50) in enumerate(zip(channels, bqs_list, kill50_list)):
        # Shared colour scale across all conditions for this channel
        all_vals = np.concatenate([feats_dict[c['label']][ch].ravel()
                                   for c in BRIGHTNESS_CONDITIONS])
        vmax = float(np.percentile(all_vals, 99))
        vmin = 0.0

        for row_i, cond in enumerate(BRIGHTNESS_CONDITIONS):
            ax   = axes[row_i, col_i]
            fmap = feats_dict[cond['label']][ch]
            kp   = kill_pct(fmap)

            ax.imshow(fmap, cmap='viridis', vmin=vmin, vmax=vmax,
                      interpolation='nearest', aspect='auto')

            # Column header (only top row)
            if row_i == 0:
                kill_flag = ' [!]' if k50 > 15 else ''
                ax.set_title(
                    f"Ch {ch:03d}  [#{col_i+1}]\n"
                    f"BQS={bqs:.4f}{kill_flag}",
                    fontsize=8.5,
                    color='#cc0000' if k50 > 15 else '#1a1a1a',
                    fontweight='bold', pad=4
                )

            # Kill% annotation
            kp_color = 'red' if kp > 15 else ('darkorange' if kp > 5 else 'white')
            ax.text(0.03, 0.04, f"Kill%={kp:.1f}%",
                    transform=ax.transAxes, fontsize=7.5,
                    color=kp_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55))

            ax.set_xticks([])
            ax.set_yticks([])

            # Row label (only leftmost column)
            if col_i == 0:
                ax.set_ylabel(cond['label'], fontsize=10,
                              color=cond['row_color'], fontweight='bold',
                              labelpad=6)

    plt.subplots_adjust(hspace=0.35, wspace=0.08, top=0.92)
    out_path = os.path.join(out_dir, f'feature_maps_layer2_top10_frame{frame_idx}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ── Plot 2: Difference Maps (+50% − Clean) ───────────────────────────────────
def plot_diff_maps(feats_dict, channels, bqs_list, kill50_list,
                   frame_idx, out_dir):
    n_ch = len(channels)

    fig, axes = plt.subplots(
        2, n_ch,
        figsize=(2.8 * n_ch, 5.5),
        gridspec_kw={'hspace': 0.35, 'wspace': 0.08}
    )

    fig.suptitle(
        f"Stage 3 (Layer2) — Feature Map Difference (+50% − Clean) — Frame {frame_idx}\n"
        f"Row 1: |Δ| (absolute)   Row 2: Δ (signed, red=increase, blue=decrease)",
        fontsize=12, fontweight='bold', y=1.01
    )

    for col_i, (ch, bqs, k50) in enumerate(zip(channels, bqs_list, kill50_list)):
        f_clean  = feats_dict['Clean'][ch].astype(np.float64)
        f_bright = feats_dict['+50%'][ch].astype(np.float64)
        diff     = f_bright - f_clean
        abs_diff = np.abs(diff)

        # Row 1: absolute difference
        ax1 = axes[0, col_i]
        vmax_abs = float(np.percentile(abs_diff, 99)) + 1e-8
        ax1.imshow(abs_diff, cmap='hot', vmin=0, vmax=vmax_abs,
                   interpolation='nearest', aspect='auto')
        mean_abs = float(np.mean(abs_diff))
        ax1.text(0.03, 0.04, f"μ|Δ|={mean_abs:.3f}",
                 transform=ax1.transAxes, fontsize=7.5, color='white',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.55))
        if col_i == 0:
            ax1.set_ylabel('|+50% − Clean|', fontsize=10, fontweight='bold',
                           color='#8B0000', labelpad=6)
        kill_flag = ' [!]' if k50 > 15 else ''
        ax1.set_title(
            f"Ch {ch:03d}  [#{col_i+1}]\nBQS={bqs:.4f}{kill_flag}",
            fontsize=8.5,
            color='#cc0000' if k50 > 15 else '#1a1a1a',
            fontweight='bold', pad=4
        )
        ax1.set_xticks([]); ax1.set_yticks([])

        # Row 2: signed difference
        ax2 = axes[1, col_i]
        vmax_signed = float(max(np.percentile(np.abs(diff), 99), 1e-8))
        ax2.imshow(diff, cmap='RdBu_r', vmin=-vmax_signed, vmax=vmax_signed,
                   interpolation='nearest', aspect='auto')
        mean_signed = float(np.mean(diff))
        ax2.text(0.03, 0.04, f"μΔ={mean_signed:+.3f}",
                 transform=ax2.transAxes, fontsize=7.5, color='black',
                 fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))
        if col_i == 0:
            ax2.set_ylabel('+50% − Clean\n(signed)', fontsize=10,
                           fontweight='bold', color='#00008B', labelpad=6)
        ax2.set_xticks([]); ax2.set_yticks([])

    plt.subplots_adjust(hspace=0.35, wspace=0.08, top=0.90)
    out_path = os.path.join(out_dir,
                            f'feature_maps_layer2_top10_diff_frame{frame_idx}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ── Plot 3: Activation Statistics Panel ──────────────────────────────────────
def plot_stats_panel(feats_dict, channels, bqs_list, kill50_list,
                     frame_idx, out_dir):
    n_ch = len(channels)
    cond_labels = [c['label'] for c in BRIGHTNESS_CONDITIONS]
    cond_colors = ['#2196F3', '#FF9800', '#F44336']
    bar_width   = 0.25
    x = np.arange(n_ch)

    fig, (ax_mean, ax_std, ax_kill) = plt.subplots(3, 1, figsize=(14, 9),
                                                    gridspec_kw={'hspace': 0.5})
    fig.suptitle(
        f"Stage 3 (Layer2) — Top-10 Channel Activation Statistics — Frame {frame_idx}",
        fontsize=13, fontweight='bold'
    )

    for ci, (cond, color) in enumerate(zip(cond_labels, cond_colors)):
        means = [float(np.mean(feats_dict[cond][ch])) for ch in channels]
        stds  = [float(np.std(feats_dict[cond][ch]))  for ch in channels]
        kills = [kill_pct(feats_dict[cond][ch])        for ch in channels]

        offset = (ci - 1) * bar_width
        ax_mean.bar(x + offset, means, bar_width, label=cond, color=color, alpha=0.85)
        ax_std.bar( x + offset, stds,  bar_width, label=cond, color=color, alpha=0.85)
        ax_kill.bar(x + offset, kills, bar_width, label=cond, color=color, alpha=0.85)

    ch_labels = [f"Ch{ch:03d}\n#{i+1}" for i, ch in enumerate(channels)]

    for ax, ylabel, title in [
        (ax_mean, 'Mean Activation',  'Mean Activation per Channel'),
        (ax_std,  'Std Activation',   'Activation Std per Channel'),
        (ax_kill, 'Kill% (dead px)',  'Kill% per Channel'),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(ch_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    ax_kill.axhline(y=15, color='red', linestyle='--', linewidth=1.2,
                    label='15% threshold')
    ax_kill.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(out_dir,
                            f'feature_maps_layer2_top10_stats_frame{frame_idx}.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ── Plot 4: Side-by-side Stage 2 vs Stage 3 BQS comparison ──────────────────
def plot_cross_stage_bqs(bqs_s2, bqs_s3, channels_s2, channels_s3, out_dir):
    """
    Simple grouped bar chart comparing Stage 2 and Stage 3 Top-10 BQS values.
    """
    x = np.arange(10)
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    bars1 = ax.bar(x - width/2, bqs_s2, width, label='Stage 2 (Layer1)',
                   color='#1565C0', alpha=0.85)
    bars2 = ax.bar(x + width/2, bqs_s3, width, label='Stage 3 (Layer2)',
                   color='#C62828', alpha=0.85)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', va='bottom',
                fontsize=7, color='#1565C0')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', va='bottom',
                fontsize=7, color='#C62828')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"S2:Ch{c2:02d}\nS3:Ch{c3:03d}" for c2, c3 in zip(channels_s2, channels_s3)],
        fontsize=8.5
    )
    ax.set_ylabel('BQS', fontsize=11)
    ax.set_title('Stage 2 (Layer1) vs Stage 3 (Layer2) — Top-10 BQS Comparison',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(bqs_s2), max(bqs_s3)) * 1.15)
    ax.axhline(y=np.mean(bqs_s2), color='#1565C0', linestyle='--',
               linewidth=1, alpha=0.6, label=f'S2 mean={np.mean(bqs_s2):.4f}')
    ax.axhline(y=np.mean(bqs_s3), color='#C62828', linestyle='--',
               linewidth=1, alpha=0.6, label=f'S3 mean={np.mean(bqs_s3):.4f}')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'stage2_vs_stage3_bqs_top10.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Saved] {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = DEVICE
    print(f"Device: {device}")
    print(f"Extraction: conv1→bn1→relu→maxpool→layer1→layer2  (Stage 3 / Layer2)")
    print(f"Top-10 channels: {TOP10_CHANNELS}")

    # Load frame
    all_imgs = sorted(glob.glob(os.path.join(RGB_DIR, '*.png')))
    if FRAME_IDX >= len(all_imgs):
        print(f"[ERROR] Frame index {FRAME_IDX} out of range (total: {len(all_imgs)})")
        return
    img_path = all_imgs[FRAME_IDX]
    print(f"Frame {FRAME_IDX}: {os.path.basename(img_path)}")

    # Extract features for all conditions
    extractor = Layer2Extractor(device=device)
    feats_dict = {}
    for cond in BRIGHTNESS_CONDITIONS:
        t = load_image_tensor(img_path, device)
        t = apply_brightness(t, cond['factor'])
        feat = extractor(t)[0].cpu().numpy()   # [128, H, W]
        feats_dict[cond['label']] = feat
        print(f"  [{cond['label']}] extracted, shape={feat.shape}")

    print("\n  Generating plots...")

    # Plot 1: Feature maps (3 rows × 10 cols)
    plot_feature_maps(feats_dict, TOP10_CHANNELS, TOP10_BQS, TOP10_KILL50,
                      FRAME_IDX, OUT_DIR)

    # Plot 2: Difference maps
    plot_diff_maps(feats_dict, TOP10_CHANNELS, TOP10_BQS, TOP10_KILL50,
                   FRAME_IDX, OUT_DIR)

    # Plot 3: Statistics panel
    plot_stats_panel(feats_dict, TOP10_CHANNELS, TOP10_BQS, TOP10_KILL50,
                     FRAME_IDX, OUT_DIR)

    # Plot 4: Cross-stage BQS comparison
    S2_BQS      = [0.5387, 0.5333, 0.5101, 0.5033, 0.4779,
                   0.4688, 0.4663, 0.4489, 0.4482, 0.4416]
    S2_CHANNELS = [60, 2, 55, 61, 53, 41, 7, 8, 46, 47]
    plot_cross_stage_bqs(S2_BQS, TOP10_BQS, S2_CHANNELS, TOP10_CHANNELS, OUT_DIR)

    print(f"\n  All outputs saved to: {OUT_DIR}/")
    print(f"    feature_maps_layer2_top10_frame{FRAME_IDX}.png")
    print(f"    feature_maps_layer2_top10_diff_frame{FRAME_IDX}.png")
    print(f"    feature_maps_layer2_top10_stats_frame{FRAME_IDX}.png")
    print(f"    stage2_vs_stage3_bqs_top10.png")


if __name__ == '__main__':
    main()