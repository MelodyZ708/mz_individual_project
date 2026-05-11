"""
Layer1 Feature Map Visualization
- Generates 8-channel heatmaps for layer1 features (same format as conv1 version)
- Also generates a side-by-side conv1 vs layer1 comparison figure
- Output: vis_results/layer1/feature_maps/

Usage (on cluster):
  python visualize_features_layer1.py
"""
import matplotlib
matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import os
import sys
import glob

sys.path.append('/vol/bitbucket/mz325/individual_project')
from como.utils.image_processing import CNNFeatureExtractor


def extract_features(img_tensor, cnn_layer="conv1"):
    """Extract 8-channel CNN features from a given layer."""
    extractor = CNNFeatureExtractor(
        target_channels=8, device="cuda:0",
        mode="cnn_only", channel_select="all",
        cnn_layer=cnn_layer
    )
    with torch.no_grad():
        features = extractor(img_tensor)
    # cnn_only mode: output is (1, 8, H, W) directly
    return features[0].cpu().numpy()  # (8, H, W)


def plot_8ch_heatmap(cnn_features, title, save_path):
    """Plot 8-channel feature heatmaps in 2x4 grid (same format as conv1 version)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(title, fontsize=16)

    for i in range(8):
        ax = axes[i // 4, i % 4]
        ch = cnn_features[i]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-6:
            ch_norm = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch_norm = np.zeros_like(ch)

        im = ax.imshow(ch_norm, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f"Ch {i+1} [{ch_min:.2f}, {ch_max:.2f}]", fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_activation_overlay(cnn_features, img_pil, save_path):
    """Plot summed activation + overlay (same format as conv1 version)."""
    img_np = np.array(img_pil) / 255.0

    summed_activation = np.mean(cnn_features, axis=0)
    act_min, act_max = summed_activation.min(), summed_activation.max()
    if act_max - act_min > 1e-6:
        summed_norm = (summed_activation - act_min) / (act_max - act_min)
    else:
        summed_norm = np.zeros_like(summed_activation)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_pil)
    axes[0].set_title("Original RGB Image", fontsize=12)
    axes[0].axis('off')

    im = axes[1].imshow(summed_norm, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Summed Feature Activation (layer1)", fontsize=12)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    cmap = plt.cm.jet
    heatmap_rgba = cmap(summed_norm)[:, :, :3]
    overlay = img_np * 0.5 + heatmap_rgba * 0.5
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (RGB + Activation)", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_side_by_side(conv1_features, layer1_features, img_idx, save_path):
    """
    Side-by-side comparison: conv1 (top row) vs layer1 (bottom row), 8 channels each.
    This is the key comparison figure.
    """
    fig, axes = plt.subplots(2, 8, figsize=(28, 7))
    fig.suptitle(f"conv1 vs layer1 Feature Channels — Image {img_idx+1}", fontsize=18, y=1.02)

    for i in range(8):
        # Top row: conv1
        ch = conv1_features[i]
        ch_min, ch_max = ch.min(), ch.max()
        ch_norm = (ch - ch_min) / (ch_max - ch_min) if ch_max - ch_min > 1e-6 else np.zeros_like(ch)
        axes[0, i].imshow(ch_norm, cmap='viridis', vmin=0, vmax=1)
        axes[0, i].set_title(f"Ch{i+1} [{ch_min:.1f},{ch_max:.1f}]", fontsize=8)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel("conv1", fontsize=14, fontweight='bold', rotation=0, labelpad=50)

        # Bottom row: layer1
        ch = layer1_features[i]
        ch_min, ch_max = ch.min(), ch.max()
        ch_norm = (ch - ch_min) / (ch_max - ch_min) if ch_max - ch_min > 1e-6 else np.zeros_like(ch)
        axes[1, i].imshow(ch_norm, cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f"Ch{i+1} [{ch_min:.1f},{ch_max:.1f}]", fontsize=8)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel("layer1", fontsize=14, fontweight='bold', rotation=0, labelpad=50)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    # --- Output directories ---
    layer1_dir = "vis_results/layer1/feature_maps"
    comparison_dir = "vis_results/comparison"
    os.makedirs(layer1_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # --- Select images (same 5 as conv1 version) ---
    rgb_dir = "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/"
    all_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))

    total = len(all_images)
    indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
    selected_images = [all_images[i] for i in indices]

    print(f"Found {total} images in {rgb_dir}")
    print(f"Selected 5 images at indices: {indices}")
    print(f"Layer1 output: {layer1_dir}")
    print(f"Comparison output: {comparison_dir}")
    print("=" * 60)

    for idx, img_path in enumerate(selected_images):
        print(f"\nProcessing image {idx+1}/5: {os.path.basename(img_path)}")

        img = Image.open(img_path).convert('RGB')
        img_tensor = ToTensor()(img).unsqueeze(0).cuda()

        # --- Layer1 feature maps (standalone) ---
        print("  Extracting layer1 features...")
        layer1_feats = extract_features(img_tensor, cnn_layer="layer1")

        plot_8ch_heatmap(
            layer1_feats,
            f"Layer1 Feature Channels — Image {idx+1}",
            os.path.join(layer1_dir, f"cnn_channels_img{idx+1}.png")
        )
        plot_activation_overlay(
            layer1_feats, img,
            os.path.join(layer1_dir, f"cnn_activation_img{idx+1}.png")
        )

        # --- Conv1 features (for comparison) ---
        print("  Extracting conv1 features...")
        conv1_feats = extract_features(img_tensor, cnn_layer="conv1")

        # --- Side-by-side comparison ---
        plot_side_by_side(
            conv1_feats, layer1_feats, idx,
            os.path.join(comparison_dir, f"conv1_vs_layer1_img{idx+1}.png")
        )

    # --- Summary: count dead channels ---
    print("\n" + "=" * 60)
    print("Dead channel analysis (layer1):")
    print("Checking last image's layer1 features...")
    for i in range(8):
        ch = layer1_feats[i]
        ch_range = ch.max() - ch.min()
        status = "DEAD" if ch_range < 1e-6 else f"ACTIVE (range={ch_range:.3f})"
        print(f"  Ch {i+1}: {status}")

    print("\n" + "=" * 60)
    print("All done!")
    print(f"  Layer1 heatmaps:  {layer1_dir}/")
    print(f"  Comparisons:      {comparison_dir}/")