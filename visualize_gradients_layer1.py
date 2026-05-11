"""
Conv1 vs Layer1 Gradient Magnitude Comparison
- Computes spatial gradient magnitude for conv1-only and layer1-only features
- Calculates gradient coverage (% pixels with significant gradients)
- Generates side-by-side gradient heatmaps + binary masks
- Output: vis_results/comparison/

Usage (on cluster):
  python visualize_gradients_layer1.py
"""
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn.functional as F
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
    return features  # (1, 8, H, W) on GPU


def compute_gradient_magnitude(features_tensor):
    """
    Compute per-pixel gradient magnitude across all channels.
    features_tensor: (1, C, H, W) on GPU
    Returns: (H, W) numpy array of gradient magnitudes
    """
    # Sobel-like gradient: simple finite difference
    # dx = features[:, :, :, 1:] - features[:, :, :, :-1]
    # dy = features[:, :, 1:, :] - features[:, :, :-1, :]
    dx = features_tensor[:, :, :, 1:] - features_tensor[:, :, :, :-1]
    dy = features_tensor[:, :, 1:, :] - features_tensor[:, :, :-1, :]

    # Pad to original size
    dx = F.pad(dx, (0, 1, 0, 0))  # pad right
    dy = F.pad(dy, (0, 0, 0, 1))  # pad bottom

    # Gradient magnitude per channel, then max across channels
    grad_mag_per_ch = torch.sqrt(dx**2 + dy**2)  # (1, C, H, W)
    grad_mag = grad_mag_per_ch.max(dim=1)[0]  # (1, H, W) — max across channels
    
    return grad_mag[0].cpu().numpy()  # (H, W)


def compute_coverage(grad_mag, threshold_percentile=75):
    """Compute gradient coverage: % of pixels above threshold."""
    threshold = np.percentile(grad_mag[grad_mag > 0], threshold_percentile) if (grad_mag > 0).any() else 0
    coverage = (grad_mag > threshold).mean() * 100
    return coverage, threshold


def plot_gradient_comparison(img_pil, grad_conv1, grad_layer1, grad_gray,
                              cov_conv1, cov_layer1, cov_gray,
                              img_idx, save_path):
    """
    3-column comparison: Gray vs Conv1 vs Layer1
    Top row: gradient magnitude heatmap
    Bottom row: binary gradient mask (above threshold)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Gradient Magnitude Comparison — Image {img_idx+1}", fontsize=16, y=0.98)

    # Shared colorbar range (use conv1 max for fair comparison)
    vmax = max(grad_conv1.max(), grad_layer1.max(), grad_gray.max())

    # --- Top row: Gradient magnitude heatmaps ---
    configs = [
        ("Gray (1ch)", grad_gray, cov_gray),
        ("Conv1 CNN-only (8ch)", grad_conv1, cov_conv1),
        ("Layer1 CNN-only (8ch)", grad_layer1, cov_layer1),
    ]

    for col, (label, grad, cov) in enumerate(configs):
        im = axes[0, col].imshow(grad, cmap='hot', vmin=0, vmax=vmax)
        axes[0, col].set_title(f"{label}\nCoverage: {cov:.1f}%", fontsize=12, fontweight='bold')
        axes[0, col].axis('off')
    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.02, pad=0.02, label='Gradient Magnitude')

    # --- Bottom row: Binary gradient masks ---
    for col, (label, grad, cov) in enumerate(configs):
        _, thresh = compute_coverage(grad)
        mask = (grad > thresh).astype(float)
        axes[1, col].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f"{label}\nPixels with gradient > threshold", fontsize=11)
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    output_dir = "vis_results/comparison"
    os.makedirs(output_dir, exist_ok=True)

    # --- Select images (same 5 as feature map script) ---
    rgb_dir = "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/"
    all_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))

    total = len(all_images)
    indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
    selected_images = [all_images[i] for i in indices]

    print(f"Found {total} images in {rgb_dir}")
    print(f"Selected 5 images at indices: {indices}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Collect coverage stats for summary
    all_cov_gray = []
    all_cov_conv1 = []
    all_cov_layer1 = []

    for idx, img_path in enumerate(selected_images):
        print(f"\nProcessing image {idx+1}/5: {os.path.basename(img_path)}")

        img = Image.open(img_path).convert('RGB')
        img_tensor = ToTensor()(img).unsqueeze(0).cuda()

        # --- Gray gradient (single channel) ---
        gray = img_tensor.mean(dim=1, keepdim=True)  # (1, 1, H, W)
        grad_gray = compute_gradient_magnitude(gray)
        cov_gray, _ = compute_coverage(grad_gray)

        # --- Conv1 gradient ---
        print("  Computing conv1 gradients...")
        conv1_feats = extract_features(img_tensor, cnn_layer="conv1")
        grad_conv1 = compute_gradient_magnitude(conv1_feats)
        cov_conv1, _ = compute_coverage(grad_conv1)

        # --- Layer1 gradient ---
        print("  Computing layer1 gradients...")
        layer1_feats = extract_features(img_tensor, cnn_layer="layer1")
        grad_layer1 = compute_gradient_magnitude(layer1_feats)
        cov_layer1, _ = compute_coverage(grad_layer1)

        print(f"  Coverage — Gray: {cov_gray:.1f}%, Conv1: {cov_conv1:.1f}%, Layer1: {cov_layer1:.1f}%")

        all_cov_gray.append(cov_gray)
        all_cov_conv1.append(cov_conv1)
        all_cov_layer1.append(cov_layer1)

        # --- Plot comparison ---
        plot_gradient_comparison(
            img, grad_conv1, grad_layer1, grad_gray,
            cov_conv1, cov_layer1, cov_gray,
            idx,
            os.path.join(output_dir, f"gradient_conv1_vs_layer1_img{idx+1}.png")
        )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Gradient Coverage Summary (5-image average):")
    print(f"  Gray (1ch):          {np.mean(all_cov_gray):.1f}% ± {np.std(all_cov_gray):.1f}%")
    print(f"  Conv1 CNN-only (8ch): {np.mean(all_cov_conv1):.1f}% ± {np.std(all_cov_conv1):.1f}%")
    print(f"  Layer1 CNN-only (8ch):{np.mean(all_cov_layer1):.1f}% ± {np.std(all_cov_layer1):.1f}%")
    print(f"\n  Conv1/Gray ratio:  {np.mean(all_cov_conv1)/np.mean(all_cov_gray):.1f}×")
    print(f"  Layer1/Gray ratio: {np.mean(all_cov_layer1)/np.mean(all_cov_gray):.1f}×")
    print(f"  Layer1/Conv1 ratio:{np.mean(all_cov_layer1)/np.mean(all_cov_conv1):.2f}×")
    print("\n" + "=" * 60)
    print("All done!")