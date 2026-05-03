"""
Tier 2.1: Convergence Basin / Loss Landscape Visualization (v3)

科学采样版本:
- 从 613 帧中均匀采样 20 对帧 (间隔 10 帧)，覆盖整个序列
- 对每对帧计算 Gray / RGB / CNN+RGB 的梯度强度
- 输出 1: 汇总柱状图 (均值 ± 标准差，带误差棒)
- 输出 2: 2 对示例帧的详细曲线图

不需要跑完整的 COMO 程序，只需要 GPU 做图像 warp 和特征提取。
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
import os
import sys
import glob

sys.path.append('/vol/bitbucket/mz325/individual_project')
from como.utils.image_processing import CNNFeatureExtractor


def load_image_tensor(image_path, device="cuda:0"):
    img = Image.open(image_path).convert('RGB')
    tensor = ToTensor()(img).unsqueeze(0).to(device)
    return tensor


def rgb_to_gray(rgb_tensor):
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb_tensor.device).view(1, 3, 1, 1)
    return (rgb_tensor * weights).sum(dim=1, keepdim=True)


def warp_image(source, dx, dy):
    B, C, H, W = source.shape
    device = source.device
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    dx_norm = 2.0 * dx / (W - 1)
    dy_norm = 2.0 * dy / (H - 1)
    grid_x = grid_x + dx_norm
    grid_y = grid_y + dy_norm
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    warped = F.grid_sample(source, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped


def compute_photometric_cost(img_ref, img_target):
    residual = img_target - img_ref
    cost = (residual ** 2).sum(dim=1).mean()
    return cost.item()


def compute_loss_landscape_1d(img_ref, img_target, perturbation_range, num_steps, axis='x'):
    perturbations = np.linspace(perturbation_range[0], perturbation_range[1], num_steps)
    costs = []
    for p in perturbations:
        if axis == 'x':
            warped = warp_image(img_target, dx=p, dy=0)
        else:
            warped = warp_image(img_target, dx=0, dy=p)
        cost = compute_photometric_cost(img_ref, warped)
        costs.append(cost)
    return perturbations, np.array(costs)


def compute_gradient(perturbations, costs):
    dp = perturbations[1] - perturbations[0]
    gradient = np.gradient(costs, dp)
    return perturbations, np.abs(gradient)


def compute_avg_gradient_for_pair(rgb_ref, rgb_target, gray_ref, gray_target,
                                   cnn_rgb_ref, cnn_rgb_target,
                                   perturb_range, num_steps):
    """对一对帧计算 X 和 Y 方向的平均梯度强度，返回 dict"""
    results = {}
    for axis in ['x', 'y']:
        _, cost_gray = compute_loss_landscape_1d(gray_ref, gray_target, perturb_range, num_steps, axis)
        _, cost_rgb = compute_loss_landscape_1d(rgb_ref, rgb_target, perturb_range, num_steps, axis)
        _, cost_cnn = compute_loss_landscape_1d(cnn_rgb_ref, cnn_rgb_target, perturb_range, num_steps, axis)

        perturbs = np.linspace(perturb_range[0], perturb_range[1], num_steps)
        _, grad_gray = compute_gradient(perturbs, cost_gray)
        _, grad_rgb = compute_gradient(perturbs, cost_rgb)
        _, grad_cnn = compute_gradient(perturbs, cost_cnn)

        results[f'gray_{axis}'] = np.mean(grad_gray)
        results[f'rgb_{axis}'] = np.mean(grad_rgb)
        results[f'cnn_{axis}'] = np.mean(grad_cnn)

    return results


def plot_detailed_pair(image_path_ref, image_path_target, extractor, output_dir,
                       pair_idx, pair_label, perturbation_pixels=30, num_steps=121):
    """对一对帧画详细的 2 行 x 2 列图 (Raw + Gradient, X + Y)"""
    device = "cuda:0"
    rgb_ref = load_image_tensor(image_path_ref, device)
    rgb_target = load_image_tensor(image_path_target, device)
    gray_ref = rgb_to_gray(rgb_ref)
    gray_target = rgb_to_gray(rgb_target)
    cnn_rgb_ref = extractor(rgb_ref)
    cnn_rgb_target = extractor(rgb_target)

    perturb_range = [-perturbation_pixels, perturbation_pixels]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'gray': '#1f77b4', 'rgb': '#2ca02c', 'cnn': '#d62728'}

    for col, axis in enumerate(['x', 'y']):
        perturbs, cost_gray = compute_loss_landscape_1d(gray_ref, gray_target, perturb_range, num_steps, axis)
        _, cost_rgb = compute_loss_landscape_1d(rgb_ref, rgb_target, perturb_range, num_steps, axis)
        _, cost_cnn = compute_loss_landscape_1d(cnn_rgb_ref, cnn_rgb_target, perturb_range, num_steps, axis)

        _, grad_gray = compute_gradient(perturbs, cost_gray)
        _, grad_rgb = compute_gradient(perturbs, cost_rgb)
        _, grad_cnn = compute_gradient(perturbs, cost_cnn)

        axis_label = axis.upper()

        # Row 0: Raw cost
        axes[0, col].plot(perturbs, cost_gray, color=colors['gray'], linewidth=1.8, label='Gray (1ch)')
        axes[0, col].plot(perturbs, cost_rgb, color=colors['rgb'], linewidth=1.8, label='RGB (3ch)')
        axes[0, col].plot(perturbs, cost_cnn, color=colors['cnn'], linewidth=1.8, label='CNN+RGB (11ch)')
        axes[0, col].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, col].set_xlabel(f"Perturbation along {axis_label} (pixels)", fontsize=11)
        axes[0, col].set_ylabel("Photometric Cost (MSE)", fontsize=11)
        axes[0, col].set_title(f"Loss Landscape - {axis_label} Translation", fontsize=12)
        axes[0, col].legend(fontsize=9)
        axes[0, col].grid(True, alpha=0.3)

        # Row 1: Gradient magnitude
        axes[1, col].plot(perturbs, grad_gray, color=colors['gray'], linewidth=1.5, alpha=0.8, label='Gray (1ch)')
        axes[1, col].plot(perturbs, grad_rgb, color=colors['rgb'], linewidth=1.5, alpha=0.8, label='RGB (3ch)')
        axes[1, col].plot(perturbs, grad_cnn, color=colors['cnn'], linewidth=1.5, alpha=0.8, label='CNN+RGB (11ch)')
        axes[1, col].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, col].set_xlabel(f"Perturbation along {axis_label} (pixels)", fontsize=11)
        axes[1, col].set_ylabel("|dCost/dp| (gradient magnitude)", fontsize=11)
        axes[1, col].set_title(f"Gradient Magnitude - {axis_label} Translation", fontsize=12)
        axes[1, col].legend(fontsize=9)
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle(f"Loss Landscape Detail - {pair_label}", fontsize=14, y=0.99)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"convergence_detail_{pair_idx+1}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved detail plot: {save_path}")


def plot_summary(all_results, output_dir):
    """画汇总柱状图: 20 对帧的平均梯度强度 (均值 ± 标准差)"""

    # 收集数据
    gray_x = [r['gray_x'] for r in all_results]
    rgb_x = [r['rgb_x'] for r in all_results]
    cnn_x = [r['cnn_x'] for r in all_results]
    gray_y = [r['gray_y'] for r in all_results]
    rgb_y = [r['rgb_y'] for r in all_results]
    cnn_y = [r['cnn_y'] for r in all_results]

    means_x = [np.mean(gray_x), np.mean(rgb_x), np.mean(cnn_x)]
    stds_x = [np.std(gray_x), np.std(rgb_x), np.std(cnn_x)]
    means_y = [np.mean(gray_y), np.mean(rgb_y), np.mean(cnn_y)]
    stds_y = [np.std(gray_y), np.std(rgb_y), np.std(cnn_y)]

    # 合并 X+Y 的平均
    gray_all = gray_x + gray_y
    rgb_all = rgb_x + rgb_y
    cnn_all = cnn_x + cnn_y
    means_all = [np.mean(gray_all), np.mean(rgb_all), np.mean(cnn_all)]
    stds_all = [np.std(gray_all), np.std(rgb_all), np.std(cnn_all)]

    labels = ['Gray (1ch)', 'RGB (3ch)', 'CNN+RGB (11ch)']
    colors = ['#1f77b4', '#2ca02c', '#d62728']
    x_pos = np.arange(3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # X 方向
    bars = axes[0].bar(x_pos, means_x, yerr=stds_x, color=colors, width=0.6,
                       edgecolor='black', linewidth=0.5, capsize=8, error_kw={'linewidth': 1.5})
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylabel("Mean |dCost/dp|", fontsize=12)
    axes[0].set_title("Gradient Strength - X Direction", fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, m, s in zip(bars, means_x, stds_x):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.0001,
                     f'{m:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Y 方向
    bars = axes[1].bar(x_pos, means_y, yerr=stds_y, color=colors, width=0.6,
                       edgecolor='black', linewidth=0.5, capsize=8, error_kw={'linewidth': 1.5})
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylabel("Mean |dCost/dp|", fontsize=12)
    axes[1].set_title("Gradient Strength - Y Direction", fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, m, s in zip(bars, means_y, stds_y):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.0001,
                     f'{m:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 合并 (X+Y 平均)
    bars = axes[2].bar(x_pos, means_all, yerr=stds_all, color=colors, width=0.6,
                       edgecolor='black', linewidth=0.5, capsize=8, error_kw={'linewidth': 1.5})
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(labels, fontsize=10)
    axes[2].set_ylabel("Mean |dCost/dp|", fontsize=12)
    axes[2].set_title("Gradient Strength - Combined (X+Y)", fontsize=13)
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, m, s in zip(bars, means_all, stds_all):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.0001,
                     f'{m:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    n_pairs = len(all_results)
    fig.suptitle(f"Average Gradient Strength Across {n_pairs} Frame Pairs (Mean ± Std)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "convergence_basin_summary.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary plot: {save_path}")

    # 打印统计
    print(f"\n{'='*60}")
    print(f"SUMMARY: Gradient Strength over {n_pairs} frame pairs")
    print(f"{'='*60}")

    print(f"\nX direction:")
    print(f"  Gray:     {means_x[0]:.5f} ± {stds_x[0]:.5f}")
    print(f"  RGB:      {means_x[1]:.5f} ± {stds_x[1]:.5f}")
    print(f"  CNN+RGB:  {means_x[2]:.5f} ± {stds_x[2]:.5f}")
    print(f"  CNN+RGB / Gray: {means_x[2]/max(means_x[0],1e-10):.1f}x")
    print(f"  CNN+RGB / RGB:  {means_x[2]/max(means_x[1],1e-10):.1f}x")

    print(f"\nY direction:")
    print(f"  Gray:     {means_y[0]:.5f} ± {stds_y[0]:.5f}")
    print(f"  RGB:      {means_y[1]:.5f} ± {stds_y[1]:.5f}")
    print(f"  CNN+RGB:  {means_y[2]:.5f} ± {stds_y[2]:.5f}")
    print(f"  CNN+RGB / Gray: {means_y[2]/max(means_y[0],1e-10):.1f}x")
    print(f"  CNN+RGB / RGB:  {means_y[2]/max(means_y[1],1e-10):.1f}x")

    print(f"\nCombined (X+Y):")
    print(f"  Gray:     {means_all[0]:.5f} ± {stds_all[0]:.5f}")
    print(f"  RGB:      {means_all[1]:.5f} ± {stds_all[1]:.5f}")
    print(f"  CNN+RGB:  {means_all[2]:.5f} ± {stds_all[2]:.5f}")
    print(f"  CNN+RGB / Gray: {means_all[2]/max(means_all[0],1e-10):.1f}x")
    print(f"  CNN+RGB / RGB:  {means_all[2]/max(means_all[1],1e-10):.1f}x")


if __name__ == "__main__":
    output_dir = "vis_results"
    os.makedirs(output_dir, exist_ok=True)

    rgb_dir = "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/"
    all_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    n_images = len(all_images)

    print(f"Found {n_images} images in {rgb_dir}")
    print(f"Output directory: {output_dir}")

    # ============================================================
    # 科学采样: 从序列中均匀选取 20 对帧，每对间隔 10 帧
    # ============================================================
    frame_gap = 10          # 每对帧之间的间隔
    n_pairs = 20            # 总共采样多少对
    perturbation_pixels = 30
    num_steps = 121

    # 可用的参考帧范围: [0, n_images - frame_gap - 1]
    max_ref_idx = n_images - frame_gap - 1
    # 均匀采样 n_pairs 个参考帧索引
    ref_indices = np.linspace(0, max_ref_idx, n_pairs, dtype=int)

    print(f"Sampling {n_pairs} frame pairs with gap={frame_gap} frames")
    print(f"Reference frame indices: {ref_indices.tolist()}")
    print("=" * 60)

    # 初始化 CNN 特征提取器
    extractor = CNNFeatureExtractor(target_channels=8, device="cuda:0")
    device = "cuda:0"

    perturb_range = [-perturbation_pixels, perturbation_pixels]
    all_results = []

    for i, ref_idx in enumerate(ref_indices):
        target_idx = ref_idx + frame_gap
        print(f"\n[{i+1}/{n_pairs}] Frame {ref_idx} vs Frame {target_idx} "
              f"({os.path.basename(all_images[ref_idx])} -> {os.path.basename(all_images[target_idx])})")

        rgb_ref = load_image_tensor(all_images[ref_idx], device)
        rgb_target = load_image_tensor(all_images[target_idx], device)
        gray_ref = rgb_to_gray(rgb_ref)
        gray_target = rgb_to_gray(rgb_target)
        cnn_rgb_ref = extractor(rgb_ref)
        cnn_rgb_target = extractor(rgb_target)

        result = compute_avg_gradient_for_pair(
            rgb_ref, rgb_target, gray_ref, gray_target,
            cnn_rgb_ref, cnn_rgb_target,
            perturb_range, num_steps
        )
        all_results.append(result)

        print(f"  Gray  avg grad: X={result['gray_x']:.5f}, Y={result['gray_y']:.5f}")
        print(f"  RGB   avg grad: X={result['rgb_x']:.5f}, Y={result['rgb_y']:.5f}")
        print(f"  CNN   avg grad: X={result['cnn_x']:.5f}, Y={result['cnn_y']:.5f}")

    # ============================================================
    # 输出 1: 汇总柱状图
    # ============================================================
    plot_summary(all_results, output_dir)

    # ============================================================
    # 输出 2: 2 对示例帧的详细曲线图 (序列开头 + 中间)
    # ============================================================
    detail_pairs = [
        (ref_indices[0], "Start of sequence"),
        (ref_indices[n_pairs // 2], "Middle of sequence"),
    ]

    print(f"\n{'='*60}")
    print("Generating detail plots for 2 example pairs...")
    for idx, (ref_idx, label) in enumerate(detail_pairs):
        target_idx = ref_idx + frame_gap
        pair_label = f"Frame {ref_idx} vs {target_idx} ({label})"
        print(f"\nDetail {idx+1}: {pair_label}")
        plot_detailed_pair(
            all_images[ref_idx], all_images[target_idx],
            extractor, output_dir,
            pair_idx=idx, pair_label=pair_label,
            perturbation_pixels=perturbation_pixels,
            num_steps=num_steps
        )

    print(f"\n{'='*60}")
    print("All done! Check vis_results/ for:")
    print("  - convergence_basin_summary.png  (main result: mean ± std over 20 pairs)")
    print("  - convergence_detail_1.png       (example: start of sequence)")
    print("  - convergence_detail_2.png       (example: middle of sequence)")