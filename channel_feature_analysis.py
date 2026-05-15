"""
Channel Feature Analysis for Top 5 Combinations from Random Channel Search.

This script provides:
1. Quantitative filter weight analysis (direction, color, frequency)
2. Top-activating patches from Early/Middle/Late frames
3. Intuitive feature map visualization

Visualization modes:
  --use_relu_for_vis (default: True)
    Uses ReLU + viridis colormap for feature maps.
    Black = no activation, bright = strong activation.
    More intuitive and publication-friendly.
    NOTE: The actual COMO system uses pre-ReLU features.
    
  --no_relu_for_vis
    Uses raw pre-ReLU features with gray symmetric colormap.
    Positive = white, Negative = black, Zero = mid-gray.

Usage:
  python channel_feature_analysis.py --output_dir ./channel_analysis_output
  python channel_feature_analysis.py --no_relu_for_vis --output_dir ./channel_analysis_output

Requires: torch, torchvision, matplotlib, numpy, PIL
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, ResNet18_Weights
import glob


# ============================================================
# PART 1: Filter Weight Analysis (Quantitative Classification)
# ============================================================

def analyze_filter_structure(weight_np):
    """
    Analyze a single conv1 filter (3, 7, 7) and return structured classification.
    
    Returns dict with:
      - direction_deg: dominant orientation in degrees (0=horiz, 90=vert)
      - color_type: 'grayscale' | 'color_opponent' | 'color_selective'
      - color_detail: e.g., 'R-G opponent', 'Blue-dominant'
      - frequency: 'low' | 'medium' | 'high'
      - spatial_type: 'edge' | 'blob' | 'texture'
      - summary: one-line human-readable description
    """
    w = weight_np  # (3, 7, 7)
    
    # --- Color Analysis ---
    r, g, b = w[0], w[1], w[2]
    r_energy = np.sum(r**2)
    g_energy = np.sum(g**2)
    b_energy = np.sum(b**2)
    total_energy = r_energy + g_energy + b_energy + 1e-10
    
    # Correlation between channels
    def corr(a, b):
        a_flat = a.flatten() - a.mean()
        b_flat = b.flatten() - b.mean()
        denom = np.sqrt(np.sum(a_flat**2) * np.sum(b_flat**2))
        if denom < 1e-10:
            return 0.0
        return np.sum(a_flat * b_flat) / denom
    
    rg_corr = corr(r, g)
    rb_corr = corr(r, b)
    gb_corr = corr(g, b)
    
    # Determine color type
    avg_corr = (rg_corr + rb_corr + gb_corr) / 3
    
    if avg_corr > 0.7:
        color_type = 'grayscale'
        color_detail = 'Achromatic (all RGB similar)'
    elif rg_corr < -0.3 or rb_corr < -0.3 or gb_corr < -0.3:
        color_type = 'color_opponent'
        # Determine which opponent
        if rg_corr < -0.3 and rg_corr <= rb_corr and rg_corr <= gb_corr:
            color_detail = 'R-G opponent'
        elif rb_corr < -0.3 and rb_corr <= rg_corr and rb_corr <= gb_corr:
            color_detail = 'R-B opponent'
        elif gb_corr < -0.3:
            color_detail = 'G-B (Blue-Yellow) opponent'
        else:
            color_detail = 'Mixed color opponent'
    else:
        color_type = 'color_selective'
        # Find dominant color
        fracs = {'Red': r_energy/total_energy, 'Green': g_energy/total_energy, 'Blue': b_energy/total_energy}
        dominant = max(fracs, key=fracs.get)
        color_detail = f'{dominant}-dominant ({fracs[dominant]*100:.0f}%)'
    
    # --- Spatial/Direction Analysis ---
    # Use mean across RGB for spatial structure
    w_gray = np.mean(w, axis=0)  # (7, 7)
    
    # Compute gradient to find dominant direction
    gy, gx = np.gradient(w_gray)
    
    # Dominant direction via structure tensor
    Ixx = np.sum(gx * gx)
    Iyy = np.sum(gy * gy)
    Ixy = np.sum(gx * gy)
    
    # Eigenvalues of structure tensor
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    discriminant = max(0, trace**2 / 4 - det)
    lambda1 = trace / 2 + np.sqrt(discriminant)
    lambda2 = trace / 2 - np.sqrt(discriminant)
    
    # Orientation coherence (how directional is the filter?)
    coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
    
    # Dominant angle (perpendicular to gradient direction = edge direction)
    angle_rad = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
    # This gives gradient direction; edge direction is perpendicular
    edge_angle_deg = (np.degrees(angle_rad) + 90) % 180
    
    # --- Frequency Analysis ---
    # Use FFT to determine frequency content
    fft2d = np.fft.fft2(w_gray)
    fft_mag = np.abs(np.fft.fftshift(fft2d))
    
    # Radial frequency profile
    cy, cx = 3, 3  # center of 7x7
    total_power = 0
    weighted_freq = 0
    for y in range(7):
        for x in range(7):
            r_dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            power = fft_mag[y, x]**2
            total_power += power
            weighted_freq += r_dist * power
    
    avg_freq = weighted_freq / (total_power + 1e-10)
    
    # DC component ratio
    dc_power = fft_mag[cy, cx]**2
    dc_ratio = dc_power / (total_power + 1e-10)
    
    if dc_ratio > 0.4:
        frequency = 'low'
    elif avg_freq > 2.0:
        frequency = 'high'
    else:
        frequency = 'medium'
    
    # --- Spatial Type ---
    if coherence > 0.5 and frequency != 'low':
        spatial_type = 'edge'
    elif frequency == 'low' and dc_ratio > 0.3:
        spatial_type = 'blob'
    else:
        spatial_type = 'texture'
    
    # --- Generate Summary ---
    if spatial_type == 'edge':
        direction_str = f'{edge_angle_deg:.0f}\u00b0'
        if 0 <= edge_angle_deg < 22.5 or edge_angle_deg >= 157.5:
            dir_name = 'Horizontal'
        elif 22.5 <= edge_angle_deg < 67.5:
            dir_name = 'Diagonal (\u2197)'
        elif 67.5 <= edge_angle_deg < 112.5:
            dir_name = 'Vertical'
        else:
            dir_name = 'Diagonal (\u2198)'
        
        if color_type == 'grayscale':
            summary = f'{dir_name} Edge ({direction_str}), Grayscale'
        elif color_type == 'color_opponent':
            summary = f'{dir_name} Edge ({direction_str}), {color_detail}'
        else:
            summary = f'{dir_name} Edge ({direction_str}), {color_detail}'
    elif spatial_type == 'blob':
        if color_type == 'grayscale':
            summary = f'Low-freq Luminance Blob'
        elif color_type == 'color_opponent':
            summary = f'Color Blob ({color_detail})'
        else:
            summary = f'Color Blob ({color_detail})'
    else:  # texture
        if color_type == 'grayscale':
            summary = f'Grayscale Texture ({frequency}-freq)'
        elif color_type == 'color_opponent':
            summary = f'Color Texture ({color_detail}, {frequency}-freq)'
        else:
            summary = f'Color Texture ({color_detail}, {frequency}-freq)'
    
    return {
        'direction_deg': edge_angle_deg,
        'coherence': coherence,
        'color_type': color_type,
        'color_detail': color_detail,
        'frequency': frequency,
        'dc_ratio': dc_ratio,
        'spatial_type': spatial_type,
        'summary': summary,
    }


# ============================================================
# PART 2: Feature Map Visualization
# ============================================================

def is_dead_channel(features_dict, ch, threshold=1e-4):
    """
    Check if a channel is dead (BN gamma ≈ 0) by checking if its output range
    is negligible across all frames.
    """
    for frame_name, feats in features_dict.items():
        feat = feats[ch]
        if (feat.max() - feat.min()) > threshold:
            return False
    return True


def visualize_feature_maps_intuitive(features_dict, channels, frame_names, output_path, title, 
                                     filter_info, original_images, use_relu=True):
    """
    Visualize feature maps in an intuitive style.
    
    If use_relu=True:
      - Apply ReLU to features (clip negatives to 0)
      - Use 'viridis' colormap: black = no activation, bright yellow/green = strong
      - Much more intuitive for understanding "what does this channel detect?"
      - NOTE: actual COMO system uses pre-ReLU features
    
    If use_relu=False:
      - Show raw pre-ReLU features
      - Use 'gray' symmetric colormap: white = positive, black = negative, gray = zero
      - More faithful to actual system behavior
    
    Dead channels: shown as flat with red "DEAD" label regardless of mode.
    
    Layout: rows = frames (Early/Middle/Late), cols = channels
    Plus a top row showing the filter weight structure.
    """
    n_ch = len(channels)
    n_frames = len(frame_names)
    
    # Detect dead channels
    dead_channels = set()
    for ch in channels:
        if is_dead_channel(features_dict, ch):
            dead_channels.add(ch)
    
    if dead_channels:
        print(f"    Dead channels detected: {sorted(dead_channels)} (will be marked)")
    
    # Layout: filter row + frame rows
    n_rows = 1 + n_frames
    
    fig, axes = plt.subplots(n_rows, n_ch + 1, 
                             figsize=(3.2 * (n_ch + 1), 3.2 * n_rows),
                             gridspec_kw={'height_ratios': [0.6] + [1.0] * n_frames})
    
    mode_str = "ReLU + viridis" if use_relu else "Pre-ReLU + gray"
    fig.suptitle(f'{title}\n[Vis mode: {mode_str}]', fontsize=13, fontweight='bold', y=0.995)
    
    # --- Row 0: Filter structure ---
    axes[0, 0].set_title('Filter\nStructure', fontsize=9, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Get conv1 weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    conv1_weights = model.conv1.weight.data.numpy()
    
    for col, ch in enumerate(channels):
        ax = axes[0, col + 1]
        w = conv1_weights[ch]  # (3, 7, 7)
        
        # Show filter as RGB image (normalized for visibility)
        # Transpose to (7, 7, 3) and normalize to [0, 1]
        w_rgb = w.transpose(1, 2, 0)  # (7, 7, 3)
        w_rgb_norm = (w_rgb - w_rgb.min()) / (w_rgb.max() - w_rgb.min() + 1e-8)
        
        ax.imshow(w_rgb_norm, interpolation='nearest')
        
        # Add classification label
        info = filter_info[ch]
        if ch in dead_channels:
            label = 'DEAD\n(no contribution)'
            title_color = 'red'
        else:
            label = info['summary']
            if len(label) > 25:
                parts = label.split(', ')
                label = '\n'.join(parts)
            title_color = 'black'
        ax.set_title(f'Ch {ch}\n{label}', fontsize=7, pad=2, color=title_color)
        ax.axis('off')
    
    # --- Rows 1-3: Feature maps per frame ---
    for row_idx, frame_name in enumerate(frame_names):
        actual_row = row_idx + 1
        
        # First column: original image
        ax = axes[actual_row, 0]
        ax.imshow(original_images[frame_name])
        ax.set_title(f'{frame_name}', fontsize=9, fontweight='bold')
        ax.axis('off')
        
        features = features_dict[frame_name]
        
        for col, ch in enumerate(channels):
            ax = axes[actual_row, col + 1]
            feat = features[ch]
            
            if ch in dead_channels:
                # Dead channel: show flat with "DEAD" text
                if use_relu:
                    ax.imshow(np.zeros_like(feat), cmap='viridis', vmin=0, vmax=1)
                else:
                    ax.imshow(np.full_like(feat, 0.5), cmap='gray', vmin=0, vmax=1)
                ax.text(0.5, 0.5, 'DEAD', fontsize=12, color='red', fontweight='bold',
                       ha='center', va='center', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            else:
                if use_relu:
                    # ReLU mode: clip negatives, use viridis
                    feat_relu = np.maximum(feat, 0)
                    # Normalize by 99th percentile for contrast
                    p99 = np.percentile(feat_relu, 99)
                    if p99 < 1e-6:
                        p99 = 1.0
                    feat_norm = np.clip(feat_relu / p99, 0, 1)
                    ax.imshow(feat_norm, cmap='viridis', vmin=0, vmax=1)
                else:
                    # Pre-ReLU mode: symmetric gray
                    p1 = np.percentile(feat, 1)
                    p99 = np.percentile(feat, 99)
                    vmax = max(abs(p1), abs(p99))
                    if vmax < 1e-6:
                        vmax = 1.0
                    feat_norm = np.clip(feat / vmax, -1, 1)
                    ax.imshow(feat_norm, cmap='gray', vmin=-1, vmax=1)
            
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# PART 3: Top-Activating Patches
# ============================================================

def find_top_patches(features_np, original_img, channel, n_patches=6, patch_size=32):
    """
    Find the top-N patches (in original image space) that maximally activate a channel.
    
    Args:
        features_np: (64, H_feat, W_feat) feature maps
        original_img: (H, W, 3) original image
        channel: channel index
        n_patches: number of top patches to return
        patch_size: size of patch in original image pixels
    
    Returns:
        list of (patch_img, activation_value, (y, x) center in original image)
    """
    feat = features_np[channel]  # (H_feat, W_feat)
    H_feat, W_feat = feat.shape
    H_orig, W_orig = original_img.shape[:2]
    
    # Scale factor (conv1 stride=2)
    scale_y = H_orig / H_feat
    scale_x = W_orig / W_feat
    
    # Use absolute value (both strong positive and negative responses are interesting)
    feat_abs = np.abs(feat)
    
    # Find top activation locations (with non-maximum suppression)
    patches = []
    used_mask = np.zeros_like(feat_abs, dtype=bool)
    
    half_patch_feat = max(1, int(patch_size / scale_x / 2))
    
    for _ in range(n_patches):
        # Mask out already-used regions
        feat_masked = feat_abs.copy()
        feat_masked[used_mask] = 0
        
        # Find max
        idx = np.unravel_index(np.argmax(feat_masked), feat_masked.shape)
        fy, fx = idx
        
        if feat_masked[fy, fx] < 1e-6:
            break
        
        # Mark region as used (NMS)
        y_start = max(0, fy - half_patch_feat)
        y_end = min(H_feat, fy + half_patch_feat)
        x_start = max(0, fx - half_patch_feat)
        x_end = min(W_feat, fx + half_patch_feat)
        used_mask[y_start:y_end, x_start:x_end] = True
        
        # Map back to original image coordinates
        cy = int(fy * scale_y)
        cx = int(fx * scale_x)
        
        # Extract patch from original image
        half = patch_size // 2
        py_start = max(0, cy - half)
        py_end = min(H_orig, cy + half)
        px_start = max(0, cx - half)
        px_end = min(W_orig, cx + half)
        
        patch = original_img[py_start:py_end, px_start:px_end]
        activation_val = feat[fy, fx]
        
        patches.append((patch, activation_val, (cy, cx)))
    
    return patches


def visualize_top_patches(features_dict, original_images, channels, frame_names,
                         output_path, title, filter_info):
    """
    For each channel, show top-activating patches across all 3 frames.
    
    Layout: rows = channels, cols = [filter | classification | patch1 | patch2 | ... | patch6]
    """
    n_ch = len(channels)
    n_patches_per_ch = 7  # show top 7 patches
    n_cols = 2 + n_patches_per_ch  # filter + class + patches
    
    fig, axes = plt.subplots(n_ch, n_cols, figsize=(2.5 * n_cols, 2.8 * n_ch))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    if n_ch == 1:
        axes = axes[np.newaxis, :]
    
    # Get conv1 weights for filter display
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    conv1_weights = model.conv1.weight.data.numpy()
    
    for row, ch in enumerate(channels):
        # Col 0: Filter weight (RGB)
        ax = axes[row, 0]
        w = conv1_weights[ch]  # (3, 7, 7)
        w_rgb = w.transpose(1, 2, 0)
        w_rgb_norm = (w_rgb - w_rgb.min()) / (w_rgb.max() - w_rgb.min() + 1e-8)
        ax.imshow(w_rgb_norm, interpolation='nearest')
        info = filter_info[ch]
        ax.set_title(f'Ch {ch}\nFilter', fontsize=8)
        ax.axis('off')
        
        # Check if dead
        ch_is_dead = is_dead_channel(features_dict, ch)
        
        # Col 1: Classification text
        ax = axes[row, 1]
        ax.axis('off')
        if ch_is_dead:
            ax.text(0.5, 0.5, 'DEAD CHANNEL\n(BN gamma\u22480)\nNo contribution\nto cost', 
                    fontsize=8, ha='center', va='center', color='red',
                    transform=ax.transAxes, wrap=True,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            ax.text(0.5, 0.5, info['summary'], fontsize=8, ha='center', va='center',
                    transform=ax.transAxes, wrap=True,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Cols 2+: Top activating patches (from all frames combined)
        if ch_is_dead:
            # Dead channel: show "N/A" in all patch slots
            for col in range(n_patches_per_ch):
                ax = axes[row, col + 2]
                ax.axis('off')
                ax.text(0.5, 0.5, 'N/A', fontsize=10, color='gray',
                       ha='center', va='center', transform=ax.transAxes)
        else:
            all_patches = []
            for frame_name in frame_names:
                patches = find_top_patches(
                    features_dict[frame_name], 
                    original_images[frame_name],
                    ch, n_patches=3, patch_size=48
                )
                for p in patches:
                    all_patches.append((p[0], abs(p[1]), p[2], frame_name))
            
            # Sort by activation strength
            all_patches.sort(key=lambda x: x[1], reverse=True)
            
            for col, (patch_img, act_val, center, fname) in enumerate(all_patches[:n_patches_per_ch]):
                ax = axes[row, col + 2]
                if patch_img.size > 0:
                    ax.imshow(patch_img)
                ax.set_title(f'{fname}\n|act|={act_val:.2f}', fontsize=7)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# PART 4: Brightness Comparison
# ============================================================

def visualize_brightness_intuitive(feature_extractor, images_dict, channels, output_path, title,
                                   filter_info, device, use_relu=True):
    """
    Show how feature maps change under brightness perturbation.
    Side-by-side Clean vs +50%.
    """
    n_ch = len(channels)
    
    # Use Middle frame
    frame_name = 'Middle'
    img_np = images_dict[frame_name]  # (H, W, 3) in [0,1]
    
    # Create brightness variants
    img_clean = img_np
    img_bright = np.clip(img_np * 1.5, 0, 1)
    
    # Extract features
    transform = ToTensor()
    
    def get_features(img_array):
        img_pil = Image.fromarray((img_array * 255).astype(np.uint8))
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        # ImageNet normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x = (img_tensor.float() - mean) / std
        with torch.no_grad():
            feats = feature_extractor(x)
        return feats[0].cpu().numpy()
    
    feats_clean = get_features(img_clean)
    feats_bright = get_features(img_bright)
    
    # Layout: 2 rows (Clean, +50%), n_ch+1 cols (image + channels)
    fig, axes = plt.subplots(2, n_ch + 1, figsize=(3 * (n_ch + 1), 6))
    mode_str = "ReLU + viridis" if use_relu else "Pre-ReLU + gray"
    fig.suptitle(f'{title}\n[Vis mode: {mode_str}]', fontsize=12, fontweight='bold')
    
    # Detect dead channels
    dead_chs = set()
    for ch in channels:
        feat_range = feats_clean[ch].max() - feats_clean[ch].min()
        if feat_range < 1e-4:
            dead_chs.add(ch)
    
    conditions = [('Clean', img_clean, feats_clean), ('+50% Brightness', img_bright, feats_bright)]
    
    for row, (cond_name, img, feats) in enumerate(conditions):
        # Image column
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(cond_name, fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        
        for col, ch in enumerate(channels):
            ax = axes[row, col + 1]
            feat = feats[ch]
            
            if ch in dead_chs:
                # Dead channel
                if use_relu:
                    ax.imshow(np.zeros_like(feat), cmap='viridis', vmin=0, vmax=1)
                else:
                    ax.imshow(np.full_like(feat, 0.5), cmap='gray', vmin=0, vmax=1)
                ax.text(0.5, 0.5, 'DEAD', fontsize=10, color='red', fontweight='bold',
                       ha='center', va='center', transform=ax.transAxes)
            else:
                if use_relu:
                    # ReLU mode
                    feat_relu = np.maximum(feat, 0)
                    # Use clean's 99th percentile for consistent scale across conditions
                    feat_clean_relu = np.maximum(feats_clean[ch], 0)
                    p99_clean = np.percentile(feat_clean_relu, 99)
                    if p99_clean < 1e-6:
                        p99_clean = 1.0
                    feat_norm = np.clip(feat_relu / p99_clean, 0, 1.5)
                    ax.imshow(feat_norm, cmap='viridis', vmin=0, vmax=1.5)
                else:
                    # Pre-ReLU mode
                    p1 = np.percentile(feats_clean[ch], 1)
                    p99 = np.percentile(feats_clean[ch], 99)
                    vmax_clean = max(abs(p1), abs(p99), 1e-6)
                    feat_norm = np.clip(feat / vmax_clean, -1.5, 1.5)
                    ax.imshow(feat_norm, cmap='gray', vmin=-1.5, vmax=1.5)
            
            if row == 0:
                info = filter_info[ch]
                if ch in dead_chs:
                    ax.set_title(f'Ch {ch}\nDEAD', fontsize=8, color='red')
                else:
                    short_label = info['summary'][:20]
                    ax.set_title(f'Ch {ch}\n{short_label}', fontsize=8)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Channel Feature Analysis')
    parser.add_argument('--dataset_dir', type=str,
                       default='/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
                       help='Path to TUM RGB directory')
    parser.add_argument('--output_dir', type=str, default='./channel_analysis_output',
                       help='Output directory')
    parser.add_argument('--use_relu_for_vis', action='store_true', default=True,
                       help='Use ReLU + viridis for feature map visualization (default: True)')
    parser.add_argument('--no_relu_for_vis', action='store_true', default=False,
                       help='Use pre-ReLU + gray symmetric colormap (overrides --use_relu_for_vis)')
    args = parser.parse_args()
    
    # Determine visualization mode
    use_relu = True  # default
    if args.no_relu_for_vis:
        use_relu = False
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Visualization mode: {'ReLU + viridis' if use_relu else 'Pre-ReLU + gray'}")
    if use_relu:
        print(f"  NOTE: ReLU is applied ONLY for visualization clarity.")
        print(f"  The actual COMO system uses pre-ReLU features to preserve gradient info.")
    
    # --- Load model ---
    print("\nLoading ResNet18...")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    
    # Feature extractor: conv1 + bn1 (NO ReLU, matching COMO)
    # ReLU is applied only during visualization if use_relu=True
    feature_extractor = nn.Sequential(model.conv1, model.bn1).to(device)
    feature_extractor.eval()
    
    conv1_weights = model.conv1.weight.data.cpu().numpy()  # (64, 3, 7, 7)
    
    # --- Select 3 frames: Early, Middle, Late ---
    all_images = sorted(glob.glob(os.path.join(args.dataset_dir, "*.png")))
    total = len(all_images)
    
    if total == 0:
        print(f"ERROR: No images found in {args.dataset_dir}")
        return
    
    # Frame indices matching the basin experiment
    early_idx = min(49, total - 1)    # frame ~50
    middle_idx = min(305, total - 1)  # frame ~306
    late_idx = min(562, total - 1)    # frame ~563
    
    frame_paths = {
        'Early': all_images[early_idx],
        'Middle': all_images[middle_idx],
        'Late': all_images[late_idx],
    }
    
    print(f"Dataset: {args.dataset_dir} ({total} images)")
    for name, path in frame_paths.items():
        print(f"  {name}: {os.path.basename(path)}")
    
    # --- Extract features for all 3 frames ---
    print("\nExtracting features...")
    features_dict = {}  # {frame_name: (64, H_feat, W_feat)}
    original_images = {}  # {frame_name: (H, W, 3) in [0,1]}
    
    transform = ToTensor()
    
    for frame_name, img_path in frame_paths.items():
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img) / 255.0
        original_images[frame_name] = img_np
        
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # ImageNet normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x = (img_tensor.float() - mean) / std
        
        with torch.no_grad():
            feats = feature_extractor(x)  # (1, 64, H/2, W/2)
        
        features_dict[frame_name] = feats[0].cpu().numpy()
        print(f"  {frame_name}: feature shape = {features_dict[frame_name].shape}")
    
    # --- Define Top 5 combinations ---
    top5 = [
        ('Top1_Run5', [6, 7, 12, 15, 36, 45, 58, 62], '115.1%'),
        ('Top2_Run1', [8, 22, 23, 27, 28, 42, 48, 60], '109.1%'),
        ('Top3_Run16', [1, 4, 7, 13, 22, 34, 41, 54], '105.6%'),
        ('Top4_Run12', [8, 9, 17, 19, 37, 41, 51, 52], '94.5%'),
        ('Top5_Run20', [2, 9, 19, 20, 28, 49, 55, 58], '94.1%'),
    ]
    
    # --- Analyze ALL unique channels across Top 5 ---
    all_channels = sorted(set(ch for _, chs, _ in top5 for ch in chs))
    print(f"\nAll unique channels in Top 5: {all_channels} ({len(all_channels)} total)")
    
    # --- PART 1: Quantitative Classification ---
    print("\n" + "=" * 70)
    print("  QUANTITATIVE FILTER CLASSIFICATION")
    print("=" * 70)
    
    filter_info = {}
    for ch in range(64):
        filter_info[ch] = analyze_filter_structure(conv1_weights[ch])
    
    # Detect dead channels
    dead_channels_set = set()
    for ch in all_channels:
        if is_dead_channel(features_dict, ch):
            dead_channels_set.add(ch)
            filter_info[ch]['summary'] = 'DEAD (BN gamma\u22480, no output)'
            filter_info[ch]['spatial_type'] = 'dead'
    
    if dead_channels_set:
        print(f"\n  Dead channels in Top 5 combinations: {sorted(dead_channels_set)}")
        print(f"  (These channels have near-zero output and do NOT contribute to cost.)")
    
    # Print table for all channels in Top 5
    print(f"\n{'Ch':>4} {'Spatial':>8} {'Freq':>6} {'Color':>16} {'Coherence':>10} {'Summary'}")
    print(f"{'---':>4} {'-------':>8} {'----':>6} {'-----':>16} {'---------':>10} {'-------'}")
    for ch in all_channels:
        info = filter_info[ch]
        if ch in dead_channels_set:
            print(f"{ch:>4} {'DEAD':>8} {'---':>6} {'---':>16} {'---':>10}  DEAD (BN gamma\u22480)")
        else:
            print(f"{ch:>4} {info['spatial_type']:>8} {info['frequency']:>6} "
                  f"{info['color_type']:>16} {info['coherence']:>9.3f}  {info['summary']}")
    
    # Save classification to text file
    class_path = os.path.join(args.output_dir, 'channel_classification.txt')
    with open(class_path, 'w') as f:
        f.write("CHANNEL CLASSIFICATION FOR TOP 5 COMBINATIONS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Visualization mode: {'ReLU + viridis' if use_relu else 'Pre-ReLU + gray'}\n")
        if use_relu:
            f.write("NOTE: ReLU is applied ONLY for visualization clarity.\n")
            f.write("The actual COMO system uses pre-ReLU features.\n\n")
        
        if dead_channels_set:
            f.write(f"DEAD CHANNELS (BN gamma\u22480, no contribution to cost):\n")
            f.write(f"  {sorted(dead_channels_set)}\n")
            f.write(f"  These channels output near-constant values (~1e-7) regardless of input.\n")
            f.write(f"  They are effectively 'free slots' that neither help nor hurt.\n\n")
        
        for run_name, channels, retention in top5:
            n_active = sum(1 for ch in channels if ch not in dead_channels_set)
            f.write(f"\n{run_name} (Retention={retention}): {channels}\n")
            f.write(f"  Active: {n_active}/{len(channels)} channels\n")
            f.write(f"{'Ch':>4} {'Type':<12} {'Summary'}\n")
            f.write(f"{'-'*4} {'-'*12} {'-'*40}\n")
            for ch in channels:
                info = filter_info[ch]
                if ch in dead_channels_set:
                    f.write(f"{ch:>4} {'DEAD':<12} (no output)\n")
                else:
                    f.write(f"{ch:>4} {info['spatial_type']:<12} {info['summary']}\n")
            f.write("\n")
    
    print(f"\n  Classification saved: {class_path}")
    
    # --- PART 2: Feature Map Visualization ---
    print("\n" + "=" * 70)
    print("  GENERATING FEATURE MAP VISUALIZATIONS")
    print("=" * 70)
    
    frame_names = ['Early', 'Middle', 'Late']
    
    for run_name, channels, retention in top5:
        print(f"\n  [{run_name}] Channels={channels}, Retention={retention}")
        
        # Feature maps
        visualize_feature_maps_intuitive(
            features_dict, channels, frame_names,
            os.path.join(args.output_dir, f'{run_name}_feature_maps.png'),
            f'{run_name}: {channels} (Retention={retention})',
            filter_info, original_images,
            use_relu=use_relu
        )
        
        # Top-activating patches
        visualize_top_patches(
            features_dict, original_images, channels, frame_names,
            os.path.join(args.output_dir, f'{run_name}_top_patches.png'),
            f'{run_name}: Top-Activating Patches',
            filter_info
        )
    
    # --- PART 3: Brightness Comparison ---
    print("\n" + "=" * 70)
    print("  GENERATING BRIGHTNESS COMPARISON")
    print("=" * 70)
    
    # Top 1 vs AVOID comparison
    visualize_brightness_intuitive(
        feature_extractor, original_images,
        [6, 7, 12, 15, 36, 45, 58, 62],
        os.path.join(args.output_dir, 'brightness_top1.png'),
        'Top 1 [6,7,12,15,36,45,58,62]: Clean vs +50% Brightness',
        filter_info, device,
        use_relu=use_relu
    )
    
    avoid_channels = [30, 44, 56, 14, 48]  # AVOID + some bottom performers
    visualize_brightness_intuitive(
        feature_extractor, original_images,
        avoid_channels,
        os.path.join(args.output_dir, 'brightness_avoid.png'),
        'AVOID Channels [30,44,56,14,48]: Clean vs +50% Brightness',
        filter_info, device,
        use_relu=use_relu
    )
    
    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Output directory: {args.output_dir}")
    print(f"  Visualization mode: {'ReLU + viridis' if use_relu else 'Pre-ReLU + gray'}")
    print(f"  Files generated:")
    print(f"    - channel_classification.txt (quantitative analysis)")
    for run_name, _, _ in top5:
        print(f"    - {run_name}_feature_maps.png")
        print(f"    - {run_name}_top_patches.png")
    print(f"    - brightness_top1.png (Top1 under brightness change)")
    print(f"    - brightness_avoid.png (AVOID channels under brightness change)")


if __name__ == '__main__':
    main()