"""
Visualize the Best Channel Combination for a given configuration.
Generates:
1. Metric Evolution Table (CSV) with BQS, LQBS, Width, Retention.
2. 3D Cost Basin plots (Clean, +30%, +50%).
3. Ablation 3D Cost Basin plots (step-by-step addition of channels).
4. Feature Map plots (Clean vs +50% for each channel in the combination).

Usage:
    python visualize_best_combination.py --layer layer2 --relu_mode pre --channels 112 75 40 43
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from forward_greedy_bqs_unified import (
    UnifiedExtractor, load_image_np, numpy_to_tensor, apply_brightness,
    build_shift_grids_gpu,
    compute_cost_landscape_gpu,
    normalize_grid_old, compute_lqbs_old, compute_basin_width_old,
    compute_sharpness_old, compute_shape_sim_old,
    compute_minpos_consistency_old, compute_symmetry_score_old,
    compute_bqs_old,
    normalize_grid_new, compute_lqbs_new, compute_basin_width_new,
    compute_bqs_new,
    compute_shape_similarity_new, compute_sharpness_retention_new,
    compute_minpos_consistency_new, compute_symmetry_new,
    clamp01, CONFIG,
)


def evaluate_subset_detailed(layer_name, channels, frame_idx, all_images,
                              extractor, dx, dy, shift_grids, N_dy, N_dx):
    device = CONFIG['device']
    rgb_np = load_image_np(all_images[frame_idx])

    feats_gpu = {}
    for cond_name, factor in CONFIG['conditions']:
        tgt_np = apply_brightness(rgb_np, factor)
        tgt_tensor = numpy_to_tensor(tgt_np, device)
        with torch.no_grad():
            feat = extractor(tgt_tensor)[0]
        feats_gpu[cond_name] = feat
        del tgt_tensor

    grids = {}
    for cond_name, _ in CONFIG['conditions']:
        grids[cond_name] = compute_cost_landscape_gpu(
            feats_gpu['clean'], feats_gpu[cond_name], channels,
            shift_grids, N_dy, N_dx,
            chunk_size=CONFIG['conv1']['chunk_size'])

    if device == 'cuda':
        torch.cuda.empty_cache()

    if layer_name == 'conv1':
        cfg = CONFIG['conv1']
        norm_clean, valid = normalize_grid_old(grids['clean'])
        if not valid:
            norm_clean = np.zeros((N_dy, N_dx))

        lqbs  = compute_lqbs_old(norm_clean, dx, dy)
        width = compute_basin_width_old(norm_clean, cfg['width_threshold'])
        sharp_clean = compute_sharpness_old(norm_clean, dx, dy)

        norm_grids = {'clean': norm_clean}
        for cond in ('bright30', 'bright50'):
            ng, _ = normalize_grid_old(grids[cond])
            norm_grids[cond] = ng

        ret_avg = 0.0
        for w_ret, cond in [(0.4, 'bright30'), (0.6, 'bright50')]:
            b_grid    = norm_grids[cond]
            shape_sim = compute_shape_sim_old(norm_clean, b_grid)
            sharp_b   = compute_sharpness_old(b_grid, dx, dy)
            sharp_ret = clamp01(sharp_b / sharp_clean) if sharp_clean > 1e-12 else 0.0
            minpos    = compute_minpos_consistency_old(b_grid, cfg['retention_local_radius'])
            sym       = compute_symmetry_score_old(b_grid, cfg['local_radius'])
            ret_avg  += w_ret * (shape_sim * sharp_ret * minpos * sym)

        lqbs_norm  = lqbs  / cfg['MAX_LQBS']  if cfg['MAX_LQBS']  > 0 else lqbs
        width_norm = width / cfg['MAX_WIDTH'] if cfg['MAX_WIDTH'] > 0 else width
        bq  = 0.75 * lqbs_norm + 0.25 * width_norm
        bqs = bq * ret_avg

        return {
            'bqs': bqs, 'lqbs': lqbs, 'width': width, 'retention': ret_avg,
            'grids': grids, 'norm_grids': norm_grids,
            'sharp_clean': sharp_clean,
            'sharp_30': compute_sharpness_old(norm_grids['bright30'], dx, dy),
            'sharp_50': compute_sharpness_old(norm_grids['bright50'], dx, dy),
        }

    else:
        cfg = CONFIG['layer12']
        norm_clean, valid = normalize_grid_new(grids['clean'])
        if not valid:
            norm_clean = np.zeros((N_dy, N_dx))

        lqbs  = compute_lqbs_new(grids['clean'])
        width = compute_basin_width_new(grids['clean'], cfg['width_threshold'])
        bqs   = compute_bqs_new(grids)

        norm_grids = {'clean': norm_clean}
        for cond in ('bright30', 'bright50'):
            ng, _ = normalize_grid_new(grids[cond])
            norm_grids[cond] = ng

        shape_sim = (compute_shape_similarity_new(grids['clean'], grids['bright30']) +
                     compute_shape_similarity_new(grids['clean'], grids['bright50'])) / 2.0
        sharp_ret = (compute_sharpness_retention_new(grids['clean'], grids['bright30']) +
                     compute_sharpness_retention_new(grids['clean'], grids['bright50'])) / 2.0
        minpos    = (compute_minpos_consistency_new(grids['clean'], grids['bright30']) +
                     compute_minpos_consistency_new(grids['clean'], grids['bright50'])) / 2.0
        sym       = (compute_symmetry_new(grids['bright30']) +
                     compute_symmetry_new(grids['bright50'])) / 2.0
        retention = shape_sim * sharp_ret * minpos * sym

        def _sharp_new(ng):
            center = ng.shape[0] // 2
            gx = np.gradient(ng[center, :])
            gy = np.gradient(ng[:, center])
            return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))

        return {
            'bqs': bqs, 'lqbs': lqbs, 'width': width, 'retention': retention,
            'grids': grids, 'norm_grids': norm_grids,
            'sharp_clean': _sharp_new(norm_clean),
            'sharp_30':    _sharp_new(norm_grids['bright30']),
            'sharp_50':    _sharp_new(norm_grids['bright50']),
        }


def plot_3d_basin(ax, grid, dx, dy, title, zlabel='Norm. Cost'):
    DX, DY = np.meshgrid(dx, dy)
    ax.plot_surface(DX, DY, grid, cmap='jet', edgecolor='none', alpha=0.9)
    ax.contour(DX, DY, grid, zdir='z', offset=0, cmap='gray', alpha=0.5)
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel('Δx [px]')
    ax.set_ylabel('Δy [px]')
    ax.set_zlabel(zlabel)
    ax.set_zlim(0, 1.0)
    ax.view_init(elev=30, azim=-45)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer',     type=str, required=True, choices=['conv1', 'layer1', 'layer2'])
    parser.add_argument('--relu_mode', type=str, required=True, choices=['pre', 'post'])
    parser.add_argument('--channels',  type=int, nargs='+', required=True)
    args = parser.parse_args()

    out_dir = f"vis_results/best_combination_{args.layer}_{args.relu_mode}_relu"
    os.makedirs(out_dir, exist_ok=True)

    rgb_dir    = CONFIG['rgb_dir']
    all_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    frame_idx  = CONFIG['target_frames'][0]
    device     = CONFIG['device']

    use_relu  = (args.relu_mode == 'post')
    extractor = UnifiedExtractor(target_layer=args.layer, use_relu=use_relu, device=device)

    max_shift = CONFIG['max_shift_px']
    grid_size = CONFIG['grid_size']
    dx = np.linspace(-max_shift, max_shift, grid_size)
    dy = np.linspace(-max_shift, max_shift, grid_size)
    N_dy, N_dx = len(dy), len(dx)

    test_np     = load_image_np(all_images[frame_idx])
    test_tensor = numpy_to_tensor(test_np, device)
    with torch.no_grad():
        test_feat = extractor(test_tensor)
    H, W = test_feat.shape[-2], test_feat.shape[-1]
    del test_tensor, test_feat
    if device == 'cuda':
        torch.cuda.empty_cache()

    shift_grids = build_shift_grids_gpu(H, W, dx, dy, device)
    rgb_np = load_image_np(all_images[frame_idx])

    # ---------------------------------------------------------
    # 1. Metric Evolution & Ablation Basins
    # ---------------------------------------------------------
    print(f"Evaluating Metric Evolution for {args.layer} {args.relu_mode}-ReLU...")
    metrics_log      = []
    ablation_results = []
    current_subset   = []
    prev_bqs         = 0.0

    for step_idx, ch in enumerate(args.channels):
        current_subset.append(ch)
        res = evaluate_subset_detailed(
            args.layer, current_subset, frame_idx, all_images,
            extractor, dx, dy, shift_grids, N_dy, N_dx)

        marginal = res['bqs'] - prev_bqs if step_idx > 0 else 0.0
        prev_bqs = res['bqs']

        metrics_log.append({
            'Step':                step_idx + 1,
            'Added Channel':       f"{ch:02d}",
            'Current Combination': str(current_subset),
            'BQS':                 round(res['bqs'],       4),
            'Marginal BQS Gain':   f"+{marginal:.4f}" if step_idx > 0 else "-",
            'LQBS':                round(res['lqbs'],      4),
            'Width':               round(res['width'],     4),
            'Retention':           round(res['retention'], 4),
        })

        ablation_results.append({
            'subset':     list(current_subset),
            'clean_norm': res['norm_grids']['clean'],
        })

        if step_idx == len(args.channels) - 1:
            final_res = res

    df_metrics = pd.DataFrame(metrics_log)
    csv_path   = os.path.join(out_dir, 'metric_evolution.csv')
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # ---------------------------------------------------------
    # 2. 3D Cost Basin (Clean, +30%, +50%) for final subset
    # ---------------------------------------------------------
    print("Generating Convergence Basin plot...")
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f'Convergence Basin — {len(args.channels)}-Channel Subset — Frame {frame_idx}\n'
        f'Channels: {args.channels}  BQS={final_res["bqs"]:.4f}',
        fontsize=14, fontweight='bold')

    for i, (cond_label, cond_key, sharp) in enumerate([
        ('Clean',  'clean',    final_res['sharp_clean']),
        ('+30%',   'bright30', final_res['sharp_30']),
        ('+50%',   'bright50', final_res['sharp_50']),
    ]):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        plot_3d_basin(ax, final_res['norm_grids'][cond_key], dx, dy,
                      f"{cond_label}\nSharpness={sharp:.4f}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    basin_path = os.path.join(out_dir, 'convergence_basin_3cond.png')
    plt.savefig(basin_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {basin_path}")

    # ---------------------------------------------------------
    # 3. Ablation Analysis Plot
    # ---------------------------------------------------------
    print("Generating Ablation Analysis plot...")
    n_ablation   = min(4, len(ablation_results))
    n_individual = min(2, len(args.channels) - 1) if len(args.channels) > 1 else 0
    total_plots  = n_ablation + n_individual

    fig = plt.figure(figsize=(4.5 * total_plots, 5))
    fig.suptitle(
        f'Ablation Analysis — Convergence Basin (Clean) — Frame {frame_idx}',
        fontsize=14, fontweight='bold')

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    labels = ["(Core only)", "(+Robust)", "(+Complementary)", "(Optimal)"]
    for i in range(n_ablation):
        ax  = fig.add_subplot(1, total_plots, i + 1, projection='3d')
        res = ablation_results[i]
        plot_3d_basin(ax, res['clean_norm'], dx, dy,
                      f"Step {i+1}: {res['subset']}\n{labels[i] if i < 4 else ''}")
        ax.title.set_color(colors[i])

    if n_individual > 0:
        for i in range(n_individual):
            ch      = args.channels[i + 1]
            res_ind = evaluate_subset_detailed(
                args.layer, [ch], frame_idx, all_images,
                extractor, dx, dy, shift_grids, N_dy, N_dx)
            ax = fig.add_subplot(1, total_plots, n_ablation + i + 1, projection='3d')
            plot_3d_basin(ax, res_ind['norm_grids']['clean'], dx, dy,
                          f"Individual Ch {ch}\n(alone)")
            ax.title.set_color('gray')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    ablation_path = os.path.join(out_dir, 'ablation_basin.png')
    plt.savefig(ablation_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {ablation_path}")

    # ---------------------------------------------------------
    # 4. Feature Maps Plot (Clean vs +50%)
    # ---------------------------------------------------------
    print("Generating Feature Maps plot...")
    rgb_tensor = numpy_to_tensor(rgb_np, device)
    tgt_np     = apply_brightness(rgb_np, 0.5)
    tgt_tensor = numpy_to_tensor(tgt_np, device)

    with torch.no_grad():
        feat_clean  = extractor(rgb_tensor)[0].cpu().numpy()
        feat_bright = extractor(tgt_tensor)[0].cpu().numpy()

    n_feat = min(4, len(args.channels))

    if n_feat == 1:
        fig, axes_raw = plt.subplots(2, 1, figsize=(4, 8))
        axes = np.array([[axes_raw[0]], [axes_raw[1]]])
    else:
        fig, axes = plt.subplots(2, n_feat, figsize=(4 * n_feat, 8))

    fig.suptitle(
        f'Optimal {len(args.channels)}-Channel Subset — Feature Maps — Frame {frame_idx}\n'
        f'Clean (top) vs +50% Brightness (bottom)',
        fontsize=14, fontweight='bold')

    roles       = ["(Core Anchor)", "(Robust Anchor)", "(Complementary)", "(Dead at +50%)"]
    role_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#7f7f7f']

    for i in range(n_feat):
        ch    = args.channels[i]
        c_map = feat_clean[ch]
        b_map = feat_bright[ch]

        kill_pct_clean  = np.mean(c_map == 0) * 100
        kill_pct_bright = np.mean(b_map == 0) * 100

        vmin = min(c_map.min(), b_map.min())
        vmax = max(c_map.max(), b_map.max())
        if vmax - vmin < 1e-6:
            vmax = vmin + 1e-6

        ax = axes[0, i]
        ax.imshow(c_map, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(
            f"Ch {ch:02d}\n{roles[i] if i < len(roles) else ''}",
            color=role_colors[i] if i < len(role_colors) else 'black',
            fontweight='bold')
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, 'Clean', transform=ax.transAxes,
                    rotation=90, va='center', fontweight='bold')
        ax.text(0.5, -0.05, f"Zero%= {kill_pct_clean:.1f}%",
                transform=ax.transAxes, ha='center', fontsize=9)

        ax = axes[1, i]
        ax.imshow(b_map, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if i == 0:
            ax.text(-0.1, 0.5, '+50% Brightness', transform=ax.transAxes,
                    rotation=90, va='center', fontweight='bold')
        kill_color = 'red' if kill_pct_bright > 80 else 'black'
        ax.text(0.5, -0.05, f"Zero%= {kill_pct_bright:.1f}%",
                transform=ax.transAxes, ha='center', fontsize=9, color=kill_color)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.1, wspace=0.05)
    feat_path = os.path.join(out_dir, 'feature_maps.png')
    plt.savefig(feat_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {feat_path}")
    print(f"All visualizations for {args.layer} {args.relu_mode}-ReLU completed in {out_dir}/\n")


if __name__ == '__main__':
    main()