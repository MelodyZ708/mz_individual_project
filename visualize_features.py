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


def visualize_features(image_path, output_dir="vis_results", img_idx=0):
    """对单张图片生成可视化"""
    
    # 1. 加载图片并转换为 Tensor
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img) / 255.0  # 归一化到 [0,1] 用于叠加
    img_tensor = ToTensor()(img).unsqueeze(0).cuda()  # (1, 3, H, W)
    
    # 2. 初始化特征提取器
    extractor = CNNFeatureExtractor(target_channels=8, device="cuda:0")
    
    # 3. 提取特征
    combined_features = extractor(img_tensor)
    cnn_features = combined_features[0, 3:, :, :].cpu().numpy()  # (8, H, W)
    
    # ========== 图1: 8个单通道特征图（归一化到[0,1]） ==========
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Individual CNN Feature Channels - Image {img_idx+1}", fontsize=16)
    
    for i in range(8):
        ax = axes[i // 4, i % 4]
        ch = cnn_features[i]
        # 每个通道单独归一化到 [0, 1]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-6:
            ch_norm = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch_norm = np.zeros_like(ch)
        
        im = ax.imshow(ch_norm, cmap='viridis', vmin=0, vmax=1)
        # 标注原始值范围，方便区分"死通道"和"活通道"
        ax.set_title(f"Ch {i+1} [{ch_min:.2f}, {ch_max:.2f}]", fontsize=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"cnn_channels_img{img_idx+1}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")
    
    # ========== 图2: 总和激活图 + 半透明叠加 ==========
    summed_activation = np.mean(cnn_features, axis=0)
    # 归一化激活图到 [0, 1]
    act_min, act_max = summed_activation.min(), summed_activation.max()
    if act_max - act_min > 1e-6:
        summed_norm = (summed_activation - act_min) / (act_max - act_min)
    else:
        summed_norm = np.zeros_like(summed_activation)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 左: 原图
    axes[0].imshow(img)
    axes[0].set_title("Original RGB Image", fontsize=12)
    axes[0].axis('off')
    
    # 中: 纯激活图
    im = axes[1].imshow(summed_norm, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Summed Feature Activation", fontsize=12)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 右: 半透明叠加（原图 + 激活热力图）
    cmap = plt.cm.jet
    heatmap_rgba = cmap(summed_norm)[:, :, :3]  # (H, W, 3)
    # 叠加：原图 * 0.5 + 热力图 * 0.5
    overlay = img_np * 0.5 + heatmap_rgba * 0.5
    overlay = np.clip(overlay, 0, 1)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (RGB + Activation)", fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"cnn_activation_img{img_idx+1}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    output_dir = "vis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 从 fr1/desk 中均匀选取 5 张图片
    rgb_dir = "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/"
    all_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    
    total = len(all_images)
    # 均匀选取 5 张：开头、1/4、中间、3/4、结尾
    indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
    selected_images = [all_images[i] for i in indices]
    
    print(f"Found {total} images in {rgb_dir}")
    print(f"Selected 5 images at indices: {indices}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)
    
    for idx, img_path in enumerate(selected_images):
        print(f"\nProcessing image {idx+1}/5: {os.path.basename(img_path)}")
        visualize_features(img_path, output_dir=output_dir, img_idx=idx)
    
    print("\n" + "=" * 50)
    print("All done! Check vis_results/ for output images.")


