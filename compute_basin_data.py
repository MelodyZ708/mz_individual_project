"""
compute_basin_data.py
=====================
Stage 1: Compute 61x61 cost landscape grids for:
  1. all 64 Conv1+BN+ReLU channels
  2. one gray baseline
under 3 brightness conditions, and save raw data as .npy files.

Protocol:
  - SAME FRAME + BRIGHTNESS PERTURBATION ONLY
  - directly comparable to previous visualization setting

Outputs:
  vis_results/basin_metrics/raw_data/channel_XX_clean.npy
  vis_results/basin_metrics/raw_data/channel_XX_bright30.npy
  vis_results/basin_metrics/raw_data/channel_XX_bright50.npy
  vis_results/basin_metrics/raw_data/gray_clean.npy
  vis_results/basin_metrics/raw_data/gray_bright30.npy
  vis_results/basin_metrics/raw_data/gray_bright50.npy
  vis_results/basin_metrics/raw_data/meta.npz
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
import cv2

CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'frame_index': 306,
    'output_dir': 'vis_results/basin_metrics/raw_data',
    'grid_range': 30,
    'grid_step': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

CONDITIONS = [
    ('clean', 0.0),
    ('bright30', 0.3),
    ('bright50', 0.5),
]


class Conv1BNReLUExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = nn.ReLU(inplace=False)
        self.to(device)
        self.eval()

        for p in self.conv1.parameters():
            p.requires_grad = False
        for p in self.bn1.parameters():
            p.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, img_tensor):
        orig_size = img_tensor.shape[-2:]
        x = (img_tensor - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x


def load_image_np(img_path):
    img = Image.open(img_path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_tensor(img_np, device):
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)


def apply_brightness(img_np, factor):
    if factor == 0.0:
        return img_np.copy()
    return np.clip(img_np + factor, 0.0, 1.0)


def rgb_to_gray(img_np):
    return 0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]


def shift_image(image, dx, dy):
    h, w = image.shape[:2]
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped


def compute_cost_landscape(feat_ref, feat_tgt, dx_vals, dy_vals):
    cost_grid = np.zeros((len(dy_vals), len(dx_vals)), dtype=np.float64)
    ref64 = feat_ref.astype(np.float64)

    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            shifted = shift_image(feat_tgt, dx, dy).astype(np.float64)
            residual = shifted - ref64
            cost_grid[i, j] = np.mean(residual ** 2)

    return cost_grid


def main():
    cfg = CONFIG
    device = cfg['device']
    print(f"[INFO] Device: {device}")
    print(f"[INFO] RGB dir: {cfg['rgb_dir']}")

    os.makedirs(cfg['output_dir'], exist_ok=True)

    all_images = sorted(glob.glob(os.path.join(cfg['rgb_dir'], "*.png")))
    if not all_images:
        print(f"[ERROR] No images found in {cfg['rgb_dir']}")
        sys.exit(1)

    frame_idx = cfg['frame_index']
    if frame_idx >= len(all_images):
        print(f"[ERROR] Frame index {frame_idx} out of range (total: {len(all_images)})")
        sys.exit(1)

    ref_path = all_images[frame_idx]
    tgt_path = all_images[frame_idx]

    print(f"[INFO] Reference: {os.path.basename(ref_path)}")
    print(f"[INFO] Target:    {os.path.basename(tgt_path)}")
    print(f"[INFO] Frame index: {frame_idx}")
    print("[INFO] Protocol: SAME FRAME + BRIGHTNESS PERTURBATION ONLY")
    print()

    rgb_np = load_image_np(ref_path)

    extractor = Conv1BNReLUExtractor(device=device)

    ref_tensor = numpy_to_tensor(rgb_np, device)
    with torch.no_grad():
        feat_ref = extractor(ref_tensor)
    feat_ref_np = feat_ref[0].cpu().numpy()

    gray_ref = rgb_to_gray(rgb_np)

    grid_range = cfg['grid_range']
    grid_step = cfg['grid_step']
    dx = np.arange(-grid_range, grid_range + 1, grid_step)
    dy = np.arange(-grid_range, grid_range + 1, grid_step)

    np.savez(
        os.path.join(cfg['output_dir'], 'meta.npz'),
        dx=dx,
        dy=dy,
        frame_index=frame_idx,
        ref_image=os.path.basename(ref_path),
        tgt_image=os.path.basename(tgt_path),
        protocol='same_frame_brightness_only',
        includes_gray_baseline=True,
    )

    print(f"[INFO] Grid: {len(dx)}x{len(dy)} (±{grid_range}px)")
    print(f"[INFO] Conditions: {[c[0] for c in CONDITIONS]}")
    print(f"[INFO] Starting computation...")
    print()

    for cond_name, brightness_factor in CONDITIONS:
        print(f"=== Condition: {cond_name} (brightness +{brightness_factor:.0%}) ===")

        rgb_target_np = apply_brightness(rgb_np, brightness_factor)
        tgt_tensor = numpy_to_tensor(rgb_target_np, device)

        with torch.no_grad():
            feat_tgt = extractor(tgt_tensor)
        feat_tgt_np = feat_tgt[0].cpu().numpy()

        for ch in range(64):
            cost_grid = compute_cost_landscape(
                feat_ref_np[ch],
                feat_tgt_np[ch],
                dx,
                dy
            )
            filename = f"channel_{ch:02d}_{cond_name}.npy"
            np.save(os.path.join(cfg['output_dir'], filename), cost_grid)

            if (ch + 1) % 16 == 0:
                print(f"  Channels 0-{ch} done")

        gray_tgt = rgb_to_gray(rgb_target_np)
        gray_grid = compute_cost_landscape(gray_ref, gray_tgt, dx, dy)
        np.save(os.path.join(cfg['output_dir'], f'gray_{cond_name}.npy'), gray_grid)
        print(f"  Gray baseline saved for '{cond_name}'")
        print()

    print(f"[DONE] Raw data saved to: {cfg['output_dir']}/")
    print("  Channel files: channel_XX_clean.npy / bright30 / bright50")
    print("  Gray files: gray_clean.npy / gray_bright30.npy / gray_bright50.npy")
    print("  Meta: meta.npz")


if __name__ == '__main__':
    main()