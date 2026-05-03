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

# 将你的项目路径加入 sys.path
sys.path.append('/vol/bitbucket/mz325/individual_project')
from como.utils.image_processing import CNNFeatureExtractor


def compute_gradient_magnitude(tensor_2d):
    """计算 2D 张量的梯度幅值 (中心差分)"""
    grad_x = np.zeros_like(tensor_2d)
    grad_y = np.zeros_like(tensor_2d)
    grad_x[:, 1:-1] = (tensor_2d[:, 2:] - tensor_2d[:, :-2]) / 2.0
    grad_y[1:-1, :] = (tensor_2d[2:, :] - tensor_2d[:-2, :]) / 2.0
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude


def compute_multichannel_gradient_magnitude(features):
    """计算多通道特征的总梯度幅值
    features: (C, H, W) numpy array
    返回: (H, W) 各通道梯度幅值的平方和再开方
    """
    total_grad_sq = np.zeros(features.shape[1:])
    for c in range(features.shape[0]):
        grad_mag = compute_gradient_magnitude(features[c])
        total_grad_sq += grad_mag**2
    return np.sqrt(total_grad_sq)


def visualize_gradients(image_path, output_dir="vis_results", img_idx=0):
    """对单张图片生成梯度幅值对比可视化"""
    
    # 1. 加载图片
    img = Image.open(image_path).convert('RGB')
    img_tensor = ToTensor()(img).unsqueeze(0).cuda()
    img_np = np.array(img) / 255.0
    
    # 2. Gray 图像
    gray = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
    
    # 3. RGB 图像
    rgb = img_np.transpose(2, 0, 1)  # (3, H, W)
    
    # 4. CNN+RGB 特征 (11通道)
    extractor = CNNFeatureExtractor(target_channels=8, device="cuda:0")
    combined_features = extractor(img_tensor)
    cnn_rgb = combined_features[0].cpu().numpy()  # (11, H, W)
    
    # 5. 计算各模式的梯度幅值
    grad_gray = compute_gradient_magnitude(gray)
    grad_rgb = compute_multichannel_gradient_magnitude(rgb)
    grad_cnn_rgb = compute_multichannel_gradient_magnitude(cnn_rgb)
    
    # 6. 统一色标（用99百分位避免极端值）
    vmax = np.percentile(grad_cnn_rgb, 99)
    
    # ========== 可视化 ==========
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：梯度幅值图（统一色标，方便直观对比）
    axes[0, 0].imshow(grad_gray, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 0].set_title("Gray Gradient Magnitude\n(1 channel)", fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(grad_rgb, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 1].set_title("RGB Gradient Magnitude\n(3 channels)", fontsize=12)
    axes[0, 1].axis('off')
    
    im = axes[0, 2].imshow(grad_cnn_rgb, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 2].set_title("CNN+RGB Gradient Magnitude\n(11 channels)", fontsize=12)
    axes[0, 2].axis('off')
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 第二行：梯度密度（二值化 - 有效梯度区域）
    threshold = vmax * 0.1
    
    mask_gray = grad_gray > threshold
    mask_rgb = grad_rgb > threshold
    mask_cnn_rgb = grad_cnn_rgb > threshold
    
    total_pixels = gray.shape[0] * gray.shape[1]
    pct_gray = 100.0 * mask_gray.sum() / total_pixels
    pct_rgb = 100.0 * mask_rgb.sum() / total_pixels
    pct_cnn_rgb = 100.0 * mask_cnn_rgb.sum() / total_pixels
    
    axes[1, 0].imshow(mask_gray, cmap='gray')
    axes[1, 0].set_title(f"Gray: {pct_gray:.1f}% pixels with gradient", fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_rgb, cmap='gray')
    axes[1, 1].set_title(f"RGB: {pct_rgb:.1f}% pixels with gradient", fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(mask_cnn_rgb, cmap='gray')
    axes[1, 2].set_title(f"CNN+RGB: {pct_cnn_rgb:.1f}% pixels with gradient", fontsize=11)
    axes[1, 2].axis('off')
    
    fig.suptitle(f"Gradient Magnitude Comparison - Image {img_idx+1}", fontsize=14, y=0.98)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"gradient_comparison_img{img_idx+1}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")
    print(f"  Gradient coverage: Gray={pct_gray:.1f}%, RGB={pct_rgb:.1f}%, CNN+RGB={pct_cnn_rgb:.1f}%")


if __name__ == "__main__":
    output_dir = "vis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    rgb_dir = "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/"
    all_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    
    total = len(all_images)
    indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
    selected_images = [all_images[i] for i in indices]
    
    print(f"Found {total} images in {rgb_dir}")
    print(f"Selected 5 images at indices: {indices}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)
    
    for idx, img_path in enumerate(selected_images):
        print(f"\nProcessing image {idx+1}/5: {os.path.basename(img_path)}")
        visualize_gradients(img_path, output_dir=output_dir, img_idx=idx)
    
    print("\n" + "=" * 50)
    print("All done! Check vis_results/ for gradient comparison images.")
