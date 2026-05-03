"""
Tier 1.3: Photometric Residual Map Visualization
Show how CNN features provide additional constraint signals beyond grayscale.

Layout (for each selected frame):
  Row 1: Original RGB image (reference frame)
  Row 2: Gray system residual (1ch) — all the info Gray optimizer can see
  Row 3: CNN+RGB system — CNN feature residual (ch 3-10, 8 channels RMSE)
         — the EXTRA constraint signal CNN provides
  Row 4: CNN+RGB system — full 11ch RMSE — total information

Narrative: Gray has only 1 weak channel of signal. CNN+RGB adds 8 more
channels of rich gradient information, enabling better convergence.

Data format:
  - r: [1, 49152, C] where C=1 (Gray) or C=11 (CNN+RGB)
  - Channel order for CNN+RGB: [RGB(3), CNN_features(8)]
  - valid_mask: [1, 49152] boolean
  - 49152 = 192 x 256
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import os
import glob

# ============================================================
# Configuration
# ============================================================
BASE_DIR = "como/vis_results"
GRAY_DIR = os.path.join(BASE_DIR, "residuals_gray")
CNN_DIR = os.path.join(BASE_DIR, "residuals_cnn8")
IMG_DIR = "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb"
OUT_DIR = "vis_results"
os.makedirs(OUT_DIR, exist_ok=True)

H, W = 192, 256  # pyramid highest resolution level

# Frames to display
SHOW_FRAMES = [1, 100, 300, 500]


def load_residual(path):
    """Load a residual .pt file and reshape to 2D."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    r = data["r"].squeeze(0)          # [49152, C]
    mask = data["valid_mask"].squeeze(0)  # [49152]
    C = r.shape[1]
    r_2d = r.reshape(H, W, C).numpy()
    mask_2d = mask.reshape(H, W).numpy()
    timestamp = data["timestamp"]
    return r_2d, mask_2d, C, timestamp


def find_rgb_image(timestamp):
    """Find the closest RGB image by timestamp."""
    target = float(timestamp)
    images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    best = None
    best_diff = float("inf")
    for img_path in images:
        ts = float(os.path.basename(img_path).replace(".png", ""))
        diff = abs(ts - target)
        if diff < best_diff:
            best_diff = diff
            best = img_path
    return best


def get_frame_idx(path):
    fname = os.path.basename(path)
    return int(fname.replace("residual_frame", "").replace(".pt", ""))


# ============================================================
# Main Figure: 4-row comparison
# ============================================================
def plot_main_figure():
    gray_files = sorted(glob.glob(os.path.join(GRAY_DIR, "residual_frame*.pt")))
    cnn_files = sorted(glob.glob(os.path.join(CNN_DIR, "residual_frame*.pt")))

    gray_map = {get_frame_idx(f): f for f in gray_files}
    cnn_map = {get_frame_idx(f): f for f in cnn_files}
    available = sorted(set(gray_map.keys()) & set(cnn_map.keys()))
    show = [f for f in SHOW_FRAMES if f in available]
    if len(show) < 2:
        show = available[:4]

    n = len(show)

    # Collect all data first to determine color scales
    all_data = []
    for fidx in show:
        g_r, g_mask, g_C, g_ts = load_residual(gray_map[fidx])
        c_r, c_mask, c_C, c_ts = load_residual(cnn_map[fidx])

        # Gray: |residual| of single channel
        g_abs = np.abs(g_r[:, :, 0])
        g_abs[~g_mask] = np.nan

        # CNN+RGB: CNN feature channels (ch 3-10) RMSE
        cnn_feat_r = c_r[:, :, 3:]  # [H, W, 8] — CNN feature channels only
        cnn_feat_rmse = np.sqrt(np.mean(cnn_feat_r ** 2, axis=2))
        cnn_feat_rmse[~c_mask] = np.nan

        # CNN+RGB: full 11ch RMSE
        full_rmse = np.sqrt(np.mean(c_r ** 2, axis=2))
        full_rmse[~c_mask] = np.nan

        # Find original RGB image
        rgb_path = find_rgb_image(g_ts)
        if rgb_path:
            rgb_img = np.array(Image.open(rgb_path).resize((W, H), Image.BILINEAR))
        else:
            rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

        all_data.append({
            "fidx": fidx,
            "rgb_img": rgb_img,
            "g_abs": g_abs,
            "cnn_feat_rmse": cnn_feat_rmse,
            "full_rmse": full_rmse,
            "g_mask": g_mask,
            "c_mask": c_mask,
        })

    # Determine color scales
    # Row 2 (Gray): use its own scale
    gray_vals = np.concatenate([np.ravel(d["g_abs"][~np.isnan(d["g_abs"])]) for d in all_data])
    gray_vmax = np.percentile(gray_vals, 97)

    # Row 3 (CNN feat): use its own scale
    cnn_vals = np.concatenate([np.ravel(d["cnn_feat_rmse"][~np.isnan(d["cnn_feat_rmse"])]) for d in all_data])
    cnn_vmax = np.percentile(cnn_vals, 97)

    # Row 4 (Full 11ch): use its own scale
    full_vals = np.concatenate([np.ravel(d["full_rmse"][~np.isnan(d["full_rmse"])]) for d in all_data])
    full_vmax = np.percentile(full_vals, 97)

    # Create figure
    fig, axes = plt.subplots(4, n, figsize=(4.5 * n, 13))
    if n == 1:
        axes = axes[:, np.newaxis]

    for i, d in enumerate(all_data):
        fidx = d["fidx"]

        # Row 1: Original RGB
        axes[0, i].imshow(d["rgb_img"])
        axes[0, i].set_title(f"Frame {fidx}", fontsize=12, fontweight="bold")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])

        # Row 2: Gray residual
        g_mean = np.nanmean(d["g_abs"])
        im2 = axes[1, i].imshow(d["g_abs"], cmap="inferno", vmin=0, vmax=gray_vmax,
                                 aspect="auto", interpolation="nearest")
        axes[1, i].set_title(f"mean = {g_mean:.4f}", fontsize=10)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

        # Row 3: CNN feature residual (extra signal)
        c_mean = np.nanmean(d["cnn_feat_rmse"])
        im3 = axes[2, i].imshow(d["cnn_feat_rmse"], cmap="inferno", vmin=0, vmax=cnn_vmax,
                                 aspect="auto", interpolation="nearest")
        axes[2, i].set_title(f"mean = {c_mean:.4f}", fontsize=10)
        axes[2, i].set_xticks([])
        axes[2, i].set_yticks([])

        # Row 4: Full 11ch RMSE
        f_mean = np.nanmean(d["full_rmse"])
        im4 = axes[3, i].imshow(d["full_rmse"], cmap="inferno", vmin=0, vmax=full_vmax,
                                 aspect="auto", interpolation="nearest")
        axes[3, i].set_title(f"mean = {f_mean:.4f}", fontsize=10)
        axes[3, i].set_xticks([])
        axes[3, i].set_yticks([])

    # Row labels
    axes[0, 0].set_ylabel("Original RGB\n(reference frame)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Gray System\n|residual| (1ch)", fontsize=11, fontweight="bold")
    axes[2, 0].set_ylabel("CNN+RGB System\nCNN features RMSE\n(8ch, extra signal)", fontsize=11, fontweight="bold")
    axes[3, 0].set_ylabel("CNN+RGB System\nFull RMSE (11ch)", fontsize=11, fontweight="bold")

    # Colorbars
    fig.colorbar(im2, ax=axes[1, :].tolist(), shrink=0.8, pad=0.02, label="|residual|")
    fig.colorbar(im3, ax=axes[2, :].tolist(), shrink=0.8, pad=0.02, label="RMSE")
    fig.colorbar(im4, ax=axes[3, :].tolist(), shrink=0.8, pad=0.02, label="RMSE")

    fig.suptitle(
        "Photometric Residual Maps: Gray (1ch) vs CNN+RGB (11ch)\n"
        "CNN features provide 8 additional channels of constraint signal for pose optimization",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 0.93, 0.96])

    out_path = os.path.join(OUT_DIR, "residual_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================
# Figure 2: Per-channel CNN feature residual for one frame
# ============================================================
def plot_per_channel_detail(frame_idx=300):
    """Show residual of each of the 8 CNN channels for a single frame."""
    cnn_files = sorted(glob.glob(os.path.join(CNN_DIR, "residual_frame*.pt")))
    cnn_map = {get_frame_idx(f): f for f in cnn_files}

    if frame_idx not in cnn_map:
        # Fall back to first available
        frame_idx = sorted(cnn_map.keys())[0]

    c_r, c_mask, c_C, c_ts = load_residual(cnn_map[frame_idx])

    # CNN feature channels: ch 3-10 (0-indexed)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    channel_names = [f"CNN Ch{i+1}" for i in range(8)]

    # Determine common scale for CNN channels
    cnn_feat = c_r[:, :, 3:]  # [H, W, 8]
    abs_feat = np.abs(cnn_feat)
    abs_feat_masked = abs_feat.copy()
    for ch in range(8):
        abs_feat_masked[:, :, ch][~c_mask] = np.nan
    vmax = np.nanpercentile(abs_feat_masked, 97)

    for ch in range(8):
        row, col = ch // 4, ch % 4
        ax = axes[row, col]

        ch_abs = np.abs(cnn_feat[:, :, ch])
        ch_abs[~c_mask] = np.nan
        ch_mean = np.nanmean(ch_abs)
        ch_std = np.nanstd(ch_abs)

        im = ax.imshow(ch_abs, cmap="inferno", vmin=0, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_title(f"{channel_names[ch]}\nmean={ch_mean:.4f}, std={ch_std:.4f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Mark dead channels
        if ch_mean < 0.001:
            ax.text(W // 2, H // 2, "DEAD", ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02, label="|residual|")
    fig.suptitle(
        f"Per-Channel CNN Feature Residuals — Frame {frame_idx}\n"
        f"Active channels (Ch2/4/5/6/8) show spatially varying residuals → rich gradient signal\n"
        f"Dead channels (Ch1/3/7) show near-zero residuals → no contribution",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "residual_per_channel.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ============================================================
# Figure 3: Statistics summary
# ============================================================
def plot_statistics():
    """Quantitative comparison across all frames."""
    gray_files = sorted(glob.glob(os.path.join(GRAY_DIR, "residual_frame*.pt")))
    cnn_files = sorted(glob.glob(os.path.join(CNN_DIR, "residual_frame*.pt")))

    gray_map = {get_frame_idx(f): f for f in gray_files}
    cnn_map = {get_frame_idx(f): f for f in cnn_files}
    available = sorted(set(gray_map.keys()) & set(cnn_map.keys()))

    frames = []
    gray_means = []
    cnn_feat_means = []
    full_means = []
    gray_coverage = []  # % of pixels with |grad| > threshold
    cnn_coverage = []

    for fidx in available:
        g_r, g_mask, _, _ = load_residual(gray_map[fidx])
        c_r, c_mask, _, _ = load_residual(cnn_map[fidx])

        g_abs = np.abs(g_r[:, :, 0])
        g_abs[~g_mask] = np.nan

        cnn_feat = c_r[:, :, 3:]
        cnn_feat_rmse = np.sqrt(np.mean(cnn_feat ** 2, axis=2))
        cnn_feat_rmse[~c_mask] = np.nan

        full_rmse = np.sqrt(np.mean(c_r ** 2, axis=2))
        full_rmse[~c_mask] = np.nan

        frames.append(fidx)
        gray_means.append(np.nanmean(g_abs))
        cnn_feat_means.append(np.nanmean(cnn_feat_rmse))
        full_means.append(np.nanmean(full_rmse))

    n = len(frames)
    x = np.arange(n)
    bar_w = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    bars1 = ax.bar(x - bar_w, gray_means, bar_w, label="Gray (1ch |residual|)",
                   color="#4A90D9", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x, cnn_feat_means, bar_w, label="CNN features (8ch RMSE)",
                   color="#E74C3C", edgecolor="black", linewidth=0.5)
    bars3 = ax.bar(x + bar_w, full_means, bar_w, label="CNN+RGB full (11ch RMSE)",
                   color="#F39C12", edgecolor="black", linewidth=0.5)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Mean Residual Magnitude", fontsize=12)
    ax.set_title("Residual Magnitude Across Frames\n"
                 "CNN features provide substantial additional signal beyond grayscale",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Frame\n{f}" for f in frames], fontsize=10)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "residual_statistics.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Print summary table
    print("\n" + "=" * 70)
    print("RESIDUAL STATISTICS SUMMARY")
    print("=" * 70)
    print(f"\n{'Frame':<10} {'Gray (1ch)':>12} {'CNN feat (8ch)':>16} {'Full (11ch)':>14} {'CNN/Gray ratio':>16}")
    print("-" * 68)
    for i in range(n):
        ratio = cnn_feat_means[i] / gray_means[i] if gray_means[i] > 0 else float("inf")
        print(f"Frame {frames[i]:<4} {gray_means[i]:>12.4f} {cnn_feat_means[i]:>16.4f} "
              f"{full_means[i]:>14.4f} {ratio:>15.1f}x")

    avg_g = np.mean(gray_means)
    avg_c = np.mean(cnn_feat_means)
    avg_f = np.mean(full_means)
    avg_ratio = avg_c / avg_g if avg_g > 0 else float("inf")
    print("-" * 68)
    print(f"{'Average':<10} {avg_g:>12.4f} {avg_c:>16.4f} {avg_f:>14.4f} {avg_ratio:>15.1f}x")
    print("=" * 70)
    print(f"\nKey insight: CNN features provide {avg_ratio:.1f}x more residual signal than Gray,")
    print(f"giving the optimizer significantly more information for pose estimation.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Tier 1.3: Photometric Residual Visualization")
    print("=" * 60)

    print("\n[1/3] Generating main comparison figure...")
    plot_main_figure()

    print("\n[2/3] Generating per-channel detail figure...")
    plot_per_channel_detail(frame_idx=300)

    print("\n[3/3] Generating statistics figure...")
    plot_statistics()

    print(f"\nAll done! Check {OUT_DIR}/ for:")
    print(f"  - residual_comparison.png   (main: 4-row comparison across frames)")
    print(f"  - residual_per_channel.png  (detail: 8 CNN channels for one frame)")
    print(f"  - residual_statistics.png   (statistics: bar chart across all frames)")