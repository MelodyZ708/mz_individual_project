import matplotlib
matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from sklearn.decomposition import PCA
import os
import sys
import glob

# 将你的项目路径加入 sys.path
sys.path.append('/vol/bitbucket/mz325/individual_project')
from como.utils.image_processing import CNNFeatureExtractor


def visualize_pca(image_path, output_dir="vis_results", img_idx=0, extractor=None):
    """对单张图片生成 CNN 特征的 PCA 伪彩色可视化"""

    # 1. 加载图片
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img) / 255.0  # (H, W, 3)
    img_tensor = ToTensor()(img).unsqueeze(0).cuda()  # (1, 3, H, W)

    # 2. 提取 CNN 特征
    combined_features = extractor(img_tensor)
    cnn_features = combined_features[0, 3:, :, :].cpu().numpy()  # (8, H, W)
    C, H, W = cnn_features.shape

    # 3. 展平空间维度: (8, H, W) -> (H*W, 8)
    features_flat = cnn_features.reshape(C, H * W).T  # (H*W, 8)

    # 4. PCA 降维: 8 -> 3
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_flat)  # (H*W, 3)

    # 5. 每个主成分归一化到 [0, 1]
    pca_img = pca_result.reshape(H, W, 3)
    for c in range(3):
        ch = pca_img[:, :, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-6:
            pca_img[:, :, c] = (ch - ch_min) / (ch_max - ch_min)
        else:
            pca_img[:, :, c] = 0.0

    # 6. 计算方差解释比
    var_ratio = pca.explained_variance_ratio_
    total_var = var_ratio.sum() * 100

    # ========== 可视化 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 左: 原始 RGB 图像
    axes[0].imshow(img)
    axes[0].set_title("Original RGB Image", fontsize=13)
    axes[0].axis('off')

    # 中: PCA 伪彩色图
    axes[1].imshow(pca_img)
    axes[1].set_title(
        f"CNN Feature PCA (8ch → 3 PCs)\n"
        f"Variance explained: {var_ratio[0]*100:.1f}% + {var_ratio[1]*100:.1f}% + {var_ratio[2]*100:.1f}% = {total_var:.1f}%",
        fontsize=11
    )
    axes[1].axis('off')

    # 右: 半透明叠加 (原图 + PCA)
    overlay = img_np * 0.4 + pca_img * 0.6
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (RGB + PCA)", fontsize=13)
    axes[2].axis('off')

    fig.suptitle(f"CNN Feature PCA Visualization - Image {img_idx+1}", fontsize=15, y=0.98)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"pca_visualization_img{img_idx+1}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")
    print(f"  Variance explained: PC1={var_ratio[0]*100:.1f}%, PC2={var_ratio[1]*100:.1f}%, PC3={var_ratio[2]*100:.1f}%, Total={total_var:.1f}%")


if __name__ == "__main__":
    output_dir = "vis_results"
    os.makedirs(output_dir, exist_ok=True)

    # 从 fr1/desk 中均匀选取 5 张图片
    rgb_dir = "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/"
    all_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))

    total = len(all_images)
    indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
    selected_images = [all_images[i] for i in indices]

    print(f"Found {total} images in {rgb_dir}")
    print(f"Selected 5 images at indices: {indices}")
    print(f"Output directory: {output_dir}")
    print("=" * 50)

    # 初始化一次 extractor，复用
    extractor = CNNFeatureExtractor(target_channels=8, device="cuda:0")

    for idx, img_path in enumerate(selected_images):
        print(f"\nProcessing image {idx+1}/5: {os.path.basename(img_path)}")
        visualize_pca(img_path, output_dir=output_dir, img_idx=idx, extractor=extractor)

    print("\n" + "=" * 50)
    print("All done! Check vis_results/ for PCA visualization images.")