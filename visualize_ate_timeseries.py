import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation


def read_tum_trajectory(filepath):
    """读取 TUM 格式轨迹文件
    返回: timestamps (N,), positions (N, 3), quaternions (N, 4) [qx, qy, qz, qw]
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            parts = line.split()
            if len(parts) >= 8:
                data.append([float(x) for x in parts[:8]])
    data = np.array(data)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    quaternions = data[:, 4:8]  # qx, qy, qz, qw
    return timestamps, positions, quaternions


def associate_trajectories(ts_gt, ts_est, max_diff=0.02):
    """时间戳关联：为每个估计帧找到最近的 GT 帧
    max_diff: 最大允许时间差 (秒)
    返回: matched_indices_gt, matched_indices_est
    """
    matched_gt = []
    matched_est = []
    for i, t_est in enumerate(ts_est):
        diffs = np.abs(ts_gt - t_est)
        j = np.argmin(diffs)
        if diffs[j] < max_diff:
            matched_gt.append(j)
            matched_est.append(i)
    return np.array(matched_gt), np.array(matched_est)


def align_trajectories_umeyama(pos_gt, pos_est):
    """Umeyama 对齐 (带尺度): 求 s, R, t 使得 pos_gt ≈ s * R @ pos_est + t
    返回对齐后的 pos_est_aligned
    """
    n = pos_gt.shape[0]
    mu_gt = pos_gt.mean(axis=0)
    mu_est = pos_est.mean(axis=0)

    gt_centered = pos_gt - mu_gt
    est_centered = pos_est - mu_est

    # 协方差矩阵
    W = (gt_centered.T @ est_centered) / n

    U, D, Vt = np.linalg.svd(W)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    # 尺度
    var_est = np.sum(est_centered ** 2) / n
    s = np.trace(np.diag(D) @ S) / var_est

    t = mu_gt - s * R @ mu_est

    pos_est_aligned = (s * (R @ pos_est.T)).T + t
    return pos_est_aligned


def compute_per_frame_ate(pos_gt, pos_est_aligned):
    """计算逐帧绝对轨迹误差 (ATE)"""
    errors = np.linalg.norm(pos_gt - pos_est_aligned, axis=1)
    return errors


def visualize_ate_timeseries(gt_file, est_files, labels, colors, output_path,
                             title="Per-frame ATE Time Series"):
    """
    绘制多条轨迹的逐帧 ATE 时间序列

    参数:
        gt_file: groundtruth 文件路径
        est_files: 估计轨迹文件路径列表
        labels: 每条轨迹的标签
        colors: 每条轨迹的颜色
        output_path: 输出图片路径
        title: 图标题
    """
    ts_gt, pos_gt, _ = read_tum_trajectory(gt_file)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    stats_text = []

    for est_file, label, color in zip(est_files, labels, colors):
        if not os.path.exists(est_file):
            print(f"  WARNING: {est_file} not found, skipping.")
            continue

        ts_est, pos_est, _ = read_tum_trajectory(est_file)

        # 时间戳关联
        idx_gt, idx_est = associate_trajectories(ts_gt, ts_est)
        if len(idx_gt) < 10:
            print(f"  WARNING: Only {len(idx_gt)} matched frames for {label}, skipping.")
            continue

        pos_gt_matched = pos_gt[idx_gt]
        pos_est_matched = pos_est[idx_est]

        # Umeyama 对齐 (带尺度校正)
        pos_est_aligned = align_trajectories_umeyama(pos_gt_matched, pos_est_matched)

        # 逐帧 ATE
        errors = compute_per_frame_ate(pos_gt_matched, pos_est_aligned)

        # 时间轴: 相对于起始时间 (秒)
        time_relative = ts_gt[idx_gt] - ts_gt[idx_gt[0]]

        # 上图: 逐帧 ATE 曲线
        axes[0].plot(time_relative, errors * 100, color=color, alpha=0.8,
                     linewidth=1.2, label=label)

        # 统计信息
        rmse = np.sqrt(np.mean(errors ** 2)) * 100
        mean_err = np.mean(errors) * 100
        max_err = np.max(errors) * 100
        stats_text.append(f"{label}: RMSE={rmse:.2f}cm, Mean={mean_err:.2f}cm, Max={max_err:.2f}cm, Frames={len(errors)}")

        print(f"  {label}: RMSE={rmse:.2f}cm, Mean={mean_err:.2f}cm, Max={max_err:.2f}cm")

    # 上图设置
    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel("ATE (cm)", fontsize=12)
    axes[0].set_title(title, fontsize=14)
    axes[0].legend(fontsize=11, loc='upper left')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    # 下图: 统计信息表格
    axes[1].axis('off')
    stats_str = "\n".join(stats_text)
    axes[1].text(0.05, 0.9, stats_str, transform=axes[1].transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


def visualize_ate_multi_run(gt_file, est_files_dict, output_path, title):
    """
    绘制多次运行的 ATE 均值 + 标准差带

    参数:
        gt_file: groundtruth 文件路径
        est_files_dict: {label: [run1.txt, run2.txt, ...], ...}
        output_path: 输出图片路径
    """
    ts_gt, pos_gt, _ = read_tum_trajectory(gt_file)
    color_map = {"Gray": "#1f77b4", "RGB": "#2ca02c", "CNN+RGB-8": "#d62728"}

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    for label, est_files in est_files_dict.items():
        all_errors = []
        common_time = None

        for est_file in est_files:
            if not os.path.exists(est_file):
                print(f"  WARNING: {est_file} not found, skipping.")
                continue

            ts_est, pos_est, _ = read_tum_trajectory(est_file)
            idx_gt, idx_est = associate_trajectories(ts_gt, ts_est)
            if len(idx_gt) < 10:
                continue

            pos_gt_matched = pos_gt[idx_gt]
            pos_est_matched = pos_est[idx_est]
            pos_est_aligned = align_trajectories_umeyama(pos_gt_matched, pos_est_matched)
            errors = compute_per_frame_ate(pos_gt_matched, pos_est_aligned) * 100  # cm
            time_rel = ts_gt[idx_gt] - ts_gt[idx_gt[0]]

            all_errors.append((time_rel, errors))

        if len(all_errors) == 0:
            continue

        # 用最短的时间序列作为公共时间轴，对其他 run 做插值
        min_len = min(len(e[0]) for e in all_errors)
        ref_time = all_errors[0][0][:min_len]

        interpolated = []
        for time_rel, errors in all_errors:
            interp_errors = np.interp(ref_time, time_rel, errors)
            interpolated.append(interp_errors)

        interpolated = np.array(interpolated)
        mean_errors = np.mean(interpolated, axis=0)
        std_errors = np.std(interpolated, axis=0)

        color = color_map.get(label, "#333333")
        ax.plot(ref_time, mean_errors, color=color, linewidth=1.5, label=f"{label} (mean of {len(interpolated)} runs)")
        ax.fill_between(ref_time, mean_errors - std_errors, mean_errors + std_errors,
                        color=color, alpha=0.15)

        rmse = np.sqrt(np.mean(mean_errors ** 2))
        print(f"  {label}: Mean RMSE={rmse:.2f}cm (over {len(interpolated)} runs)")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("ATE (cm)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    output_dir = "vis_results"
    os.makedirs(output_dir, exist_ok=True)

    results_dir = "/vol/bitbucket/mz325/individual_project/como/results"
    gt_fr1 = "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt"

    # ========== 图1: 单次运行对比 (run1) ==========
    print("=" * 50)
    print("Plot 1: Single run comparison (run1)")
    print("=" * 50)

    visualize_ate_timeseries(
        gt_file=gt_fr1,
        est_files=[
            os.path.join(results_dir, "fr1desk_gray_run1.txt"),
            os.path.join(results_dir, "fr1desk_rgb_run1.txt"),
            os.path.join(results_dir, "fr1desk_cnn8_run1.txt"),
        ],
        labels=["Gray", "RGB", "CNN+RGB-8"],
        colors=["#1f77b4", "#2ca02c", "#d62728"],
        output_path=os.path.join(output_dir, "ate_timeseries_fr1desk_run1.png"),
        title="Per-frame ATE Time Series - fr1/desk (Run 1)"
    )

    # ========== 图2: 多次运行均值 + 标准差带 ==========
    print("\n" + "=" * 50)
    print("Plot 2: Multi-run mean + std band (5 runs)")
    print("=" * 50)

    visualize_ate_multi_run(
        gt_file=gt_fr1,
        est_files_dict={
            "Gray": [os.path.join(results_dir, f"fr1desk_gray_run{i}.txt") for i in range(1, 6)],
            "RGB": [os.path.join(results_dir, f"fr1desk_rgb_run{i}.txt") for i in range(1, 6)],
            "CNN+RGB-8": [os.path.join(results_dir, f"fr1desk_cnn8_run{i}.txt") for i in range(1, 6)],
        },
        output_path=os.path.join(output_dir, "ate_timeseries_fr1desk_multi_run.png"),
        title="Per-frame ATE Time Series - fr1/desk (Mean ± Std of 5 Runs)"
    )

    print("\n" + "=" * 50)
    print("All done! Check vis_results/ for ATE time series plots.")