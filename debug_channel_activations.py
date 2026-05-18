"""
Quick Debug: Print per-channel activation statistics under Clean and +50% brightness.
Checks whether channels are truly zeroed out by ReLU under brightness change.

Usage:
  python debug_channel_activations.py

Author: mz325
Date: 2026-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import glob
import os

# ── Config ──
RGB_DIR = '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
FRAME_INDICES = [41, 306, 512]

RANK01_CHANNELS = [6, 7, 12, 15, 36, 45, 58, 62]
RANK02_CHANNELS = [8, 22, 23, 27, 28, 42, 48, 60]

BRIGHTNESS_FACTORS = [0.0, 0.3, 0.5]

# ── Setup model ──
print(f"Device: {DEVICE}")
base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(DEVICE).eval()
mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

all_images = sorted(glob.glob(os.path.join(RGB_DIR, "*.png")))
print(f"Total images: {len(all_images)}")

def extract_features(rgb_np):
    """Extract all 64 channels: conv1 -> bn1 -> relu, with ImageNet normalization."""
    t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    t = (t.float() - mean) / std
    with torch.no_grad():
        x = base.conv1(t)
        x_bn = base.bn1(x)
        x_relu = base.relu(x_bn)
    return x_bn[0].cpu().numpy(), x_relu[0].cpu().numpy()  # [64, H, W] each

def extract_features_no_relu(rgb_np):
    """Extract all 64 channels: conv1 -> bn1 (NO relu), with ImageNet normalization."""
    t = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    t = (t.float() - mean) / std
    with torch.no_grad():
        x = base.conv1(t)
        x_bn = base.bn1(x)
    return x_bn[0].cpu().numpy()  # [64, H, W]


print("\n" + "=" * 100)
print("  CHANNEL ACTIVATION STATISTICS: Before and After ReLU")
print("  Clean vs +30% vs +50% Brightness")
print("=" * 100)

for frame_idx in FRAME_INDICES:
    img_path = all_images[frame_idx]
    rgb_np = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.0

    print(f"\n{'─' * 100}")
    print(f"  Frame {frame_idx}: {os.path.basename(img_path)}")
    print(f"{'─' * 100}")

    for bf in BRIGHTNESS_FACTORS:
        label = 'Clean' if bf == 0.0 else f'+{int(bf*100)}%'
        rgb_perturbed = np.clip(rgb_np + bf, 0.0, 1.0)

        feats_bn, feats_relu = extract_features(rgb_perturbed)

        print(f"\n  Brightness: {label}")
        print(f"  {'Ch':>4s}  {'[After BN]':^40s}  {'[After ReLU]':^40s}  {'ReLU':>8s}")
        print(f"  {'':>4s}  {'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s}  {'min':>10s} {'max':>10s} {'mean':>10s} {'nonzero':>10s}  {'kill%':>8s}")
        print(f"  {'─' * 96}")

        for combo_name, channels in [('R01', RANK01_CHANNELS), ('R02', RANK02_CHANNELS)]:
            for ch in channels:
                bn_fm = feats_bn[ch]
                relu_fm = feats_relu[ch]
                total_pixels = relu_fm.size
                nonzero = np.count_nonzero(relu_fm)
                kill_pct = (1.0 - nonzero / total_pixels) * 100.0

                print(f"  {combo_name} Ch{ch:>2d}  "
                      f"{bn_fm.min():>10.4f} {bn_fm.max():>10.4f} {bn_fm.mean():>10.4f} {bn_fm.std():>10.4f}  "
                      f"{relu_fm.min():>10.4f} {relu_fm.max():>10.4f} {relu_fm.mean():>10.4f} {nonzero:>10d}  "
                      f"{kill_pct:>7.1f}%")

            if combo_name == 'R01':
                print(f"  {'─' * 96}")

print(f"\n{'=' * 100}")
print("  SUMMARY: Channels with >95% pixels killed by ReLU under +50%")
print("=" * 100)

# Check all 3 frames under +50%
for frame_idx in FRAME_INDICES:
    rgb_np = np.array(Image.open(all_images[frame_idx]).convert('RGB'), dtype=np.float32) / 255.0
    rgb_bright = np.clip(rgb_np + 0.5, 0.0, 1.0)
    _, feats_relu = extract_features(rgb_bright)

    killed_channels = []
    for ch in range(64):
        fm = feats_relu[ch]
        kill_pct = (1.0 - np.count_nonzero(fm) / fm.size) * 100.0
        if kill_pct > 95.0:
            killed_channels.append((ch, kill_pct))

    print(f"\n  Frame {frame_idx}: {len(killed_channels)} channels >95% killed")
    for ch, pct in killed_channels:
        marker = ""
        if ch in RANK01_CHANNELS:
            marker += " [RANK01]"
        if ch in RANK02_CHANNELS:
            marker += " [RANK02]"
        print(f"    Ch{ch:>2d}: {pct:.1f}% killed{marker}")

print("\nDone.")