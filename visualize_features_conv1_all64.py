"""
Visualize ALL 64 channels of ResNet-18 conv1 (conv1 + bn1 + relu) feature maps.

This script directly loads a pretrained ResNet-18 and extracts the full 64-channel
output from conv1+bn1+relu (resolution H/2 × W/2), then upsamples back to original
resolution for visualization.

Output:
  - 8×8 grid of all 64 channels per image
  - Summed activation overlay per image
  - Summary statistics (dead vs active channels)

Output directory: vis_results/feature_maps_conv1_all64/
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, ResNet18_Weights
import os
import glob


def build_conv1_extractor(device="cuda:0"):
    """Build a feature extractor that outputs all 64 channels from conv1+bn1+relu."""
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    resnet.eval()
    
    # conv1 + bn1 + relu only (NOT layer1)
    # Output: 64 channels, H/2 × W/2
    feature_extractor = nn.Sequential(
        resnet.conv1,   # 3 → 64, stride=2, output H/2 × W/2
        resnet.bn1,
        resnet.relu
    )
    
    # Freeze all parameters
    for param in feature_extractor.parameters():
        param.requires_grad = False
    
    print("[Conv1 Extractor] Using conv1+bn1+relu: 64 channels, resolution H/2×W/2, upsample 2×")
    return feature_extractor


def extract_all64_features(img_tensor, extractor, device="cuda:0"):
    """
    Extract all 64 conv1 feature channels from an image tensor.
    
    Args:
        img_tensor: (1, 3, H, W) RGB tensor in [0, 1]
        extractor: nn.Sequential (conv1+bn1+relu)
        device: cuda device
    
    Returns:
        features_np: (64, H, W) numpy array of feature maps at original resolution
    """
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x = (img_tensor.float() - mean) / std
    
    # Extract features
    with torch.no_grad():
        features = extractor(x)  # (1, 64, H/2, W/2)
    
    # Upsample 2× back to original resolution
    features_upsampled = F.interpolate(
        features,
        size=img_tensor.shape[-2:],  # (H, W)
        mode='bilinear',
        align_corners=False
    )
    
    # Convert to numpy: (64, H, W)
    features_np = features_upsampled[0].cpu().numpy()
    return features_np


def visualize_all64_channels(features_np, output_dir, img_idx, img_name):
    """
    Visualize all 64 channels in an 8×8 grid.
    
    Args:
        features_np: (64, H, W) numpy array
        output_dir: output directory path
        img_idx: image index (0-based)
        img_name: image filename for title
    """
    fig, axes = plt.subplots(8, 8, figsize=(24, 24))
    fig.suptitle(f"ResNet-18 Conv1 — All 64 Channels\n{img_name}", fontsize=18, y=0.995)
    
    # Track channel statistics
    channel_stats = []
    
    for ch_idx in range(64):
        row = ch_idx // 8
        col = ch_idx % 8
        ax = axes[row, col]
        
        ch = features_np[ch_idx]
        ch_min, ch_max = ch.min(), ch.max()
        ch_range = ch_max - ch_min
        channel_stats.append({
            'idx': ch_idx,
            'min': ch_min,
            'max': ch_max,
            'range': ch_range,
            'std': ch.std(),
            'is_dead': ch_range < 1e-4
        })
        
        # Normalize each channel individually to [0, 1]
        if ch_range > 1e-4:
            ch_norm = (ch - ch_min) / ch_range
        else:
            ch_norm = np.zeros_like(ch)
        
        im = ax.imshow(ch_norm, cmap='viridis', vmin=0, vmax=1)
        
        # Color-code title: red for dead channels, black for active
        title_color = 'red' if ch_range < 1e-4 else 'black'
        ax.set_title(f"Ch {ch_idx+1}\n[{ch_min:.2f}, {ch_max:.2f}]",
                     fontsize=7, color=title_color, pad=2)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(output_dir, f"conv1_all64_img{img_idx+1}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved 8×8 grid: {save_path}")
    
    return channel_stats


def visualize_activation_overlay(features_np, img_np, output_dir, img_idx, img_name):
    """
    Generate summed activation map and overlay on original image.
    
    Args:
        features_np: (64, H, W) numpy array
        img_np: (H, W, 3) numpy array in [0, 1]
        output_dir: output directory path
        img_idx: image index
        img_name: image filename
    """
    # Mean activation across all 64 channels
    mean_activation = np.mean(features_np, axis=0)
    
    # Normalize to [0, 1]
    act_min, act_max = mean_activation.min(), mean_activation.max()
    if act_max - act_min > 1e-6:
        act_norm = (mean_activation - act_min) / (act_max - act_min)
    else:
        act_norm = np.zeros_like(mean_activation)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Left: Original RGB
    axes[0].imshow(img_np)
    axes[0].set_title("Original RGB Image", fontsize=13)
    axes[0].axis('off')
    
    # Middle: Mean activation heatmap
    im = axes[1].imshow(act_norm, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Mean Activation (64 channels)", fontsize=13)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Right: Overlay
    cmap = plt.cm.jet
    heatmap_rgba = cmap(act_norm)[:, :, :3]
    overlay = img_np * 0.5 + heatmap_rgba * 0.5
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (RGB + Activation)", fontsize=13)
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"conv1_activation_img{img_idx+1}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved activation overlay: {save_path}")


def print_channel_summary(all_stats):
    """Print summary statistics about dead vs active channels."""
    print("\n" + "=" * 60)
    print("CHANNEL SUMMARY ACROSS ALL IMAGES")
    print("=" * 60)
    
    # Aggregate: a channel is "dead" if it's dead in ALL images
    n_channels = 64
    n_images = len(all_stats)
    
    dead_counts = np.zeros(n_channels)
    for img_stats in all_stats:
        for s in img_stats:
            if s['is_dead']:
                dead_counts[s['idx']] += 1
    
    always_dead = [i+1 for i in range(n_channels) if dead_counts[i] == n_images]
    sometimes_dead = [i+1 for i in range(n_channels) if 0 < dead_counts[i] < n_images]
    always_active = [i+1 for i in range(n_channels) if dead_counts[i] == 0]
    
    print(f"\nAlways dead (range < 1e-4 in all images): {len(always_dead)} channels")
    if always_dead:
        print(f"  Channels: {always_dead}")
    
    print(f"\nSometimes dead: {len(sometimes_dead)} channels")
    if sometimes_dead:
        print(f"  Channels: {sometimes_dead}")
    
    print(f"\nAlways active: {len(always_active)} channels")
    print(f"\n  → {len(always_active)}/64 channels carry useful information for conv1")
    
    # Print per-channel std (averaged across images) for ranking
    print("\n" + "-" * 60)
    print("TOP 10 most informative channels (by avg std across images):")
    print("-" * 60)
    
    avg_stds = np.zeros(n_channels)
    for img_stats in all_stats:
        for s in img_stats:
            avg_stds[s['idx']] += s['std']
    avg_stds /= n_images
    
    ranked = np.argsort(avg_stds)[::-1]
    print(f"{'Rank':<6}{'Channel':<10}{'Avg Std':<12}")
    for rank, ch_idx in enumerate(ranked[:10]):
        print(f"{rank+1:<6}Ch {ch_idx+1:<7}{avg_stds[ch_idx]:.4f}")
    
    print("\nBOTTOM 10 (least informative / dead):")
    print(f"{'Rank':<6}{'Channel':<10}{'Avg Std':<12}")
    for rank, ch_idx in enumerate(ranked[-10:]):
        print(f"{64-9+rank:<6}Ch {ch_idx+1:<7}{avg_stds[ch_idx]:.6f}")


def main():
    # Configuration
    output_dir = "vis_results/feature_maps_conv1_all64"
    rgb_dir = "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select 3 representative images (beginning, middle, end)
    all_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    total = len(all_images)
    indices = [0, total // 2, total - 1]
    selected_images = [all_images[i] for i in indices]
    
    print(f"Dataset: {rgb_dir}")
    print(f"Found {total} images, selected 3 at indices: {indices}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Build conv1 extractor (NOT layer1)
    extractor = build_conv1_extractor(device=device)
    
    all_channel_stats = []
    
    for idx, img_path in enumerate(selected_images):
        img_name = os.path.basename(img_path)
        print(f"\n[{idx+1}/3] Processing: {img_name}")
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img) / 255.0  # (H, W, 3) in [0, 1]
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)  # (1, 3, H, W)
        
        # Extract all 64 conv1 features
        features_np = extract_all64_features(img_tensor, extractor, device=device)
        print(f"  Feature shape: {features_np.shape}")  # (64, H, W)
        
        # Visualize 8×8 grid
        stats = visualize_all64_channels(features_np, output_dir, idx, img_name)
        all_channel_stats.append(stats)
        
        # Visualize activation overlay
        visualize_activation_overlay(features_np, img_np, output_dir, idx, img_name)
    
    # Print channel summary
    print_channel_summary(all_channel_stats)
    
    # Save summary to text file
    summary_path = os.path.join(output_dir, "channel_summary.txt")
    import sys
    # Re-run summary with output to file
    original_stdout = sys.stdout
    with open(summary_path, 'w') as f:
        sys.stdout = f
        print_channel_summary(all_channel_stats)
        sys.stdout = original_stdout
    print(f"\nSummary saved to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("All done! Check output directory for results.")
    print(f"  {output_dir}/conv1_all64_img[1-3].png  — 8×8 channel grids")
    print(f"  {output_dir}/conv1_activation_img[1-3].png  — activation overlays")
    print(f"  {output_dir}/channel_summary.txt  — dead/active channel analysis")


if __name__ == "__main__":
    main()