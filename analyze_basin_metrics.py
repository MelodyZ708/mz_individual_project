"""
analyze_basin_metrics.py  (v5.7 — 2026-05-24)
==============================================
Combined basin metric + per-metric ranking tables + gray baseline metrics.

Final combined metric:
  basin_quality = 0.75*LQBS + 0.25*Width
  retention = shape_similarity * sharpness_retention * minpos_consistency * bright_symmetry
  BQS = basin_quality × weighted_average_retention

Also computes the same metric family for a gray baseline:
  gray_clean.npy / gray_bright30.npy / gray_bright50.npy
"""

import os
from collections import deque

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

CONFIG = {
    'raw_data_dir': 'vis_results/basin_metrics/raw_data',
    'output_dir': 'vis_results/basin_metrics',
    'plots_dir': 'vis_results/basin_metrics/plots',
    'rankings_dir': 'vis_results/basin_metrics/metric_rankings',
    'num_channels': 64,
    'conditions': ['clean', 'bright30', 'bright50'],
    'condition_labels': {'clean': 'Clean', 'bright30': '+30%', 'bright50': '+50%'},
    'local_radius': 10,
    'retention_local_radius': 10,
    'width_threshold': 0.20,
    'basin_weights': {
        'local_quadratic_bowl': 0.75,
        'basin_width': 0.25,
    },
    'convexity_scale': 0.002,
    'dead_threshold': 1e-8,
}


def clamp01(x):
    return float(max(0.0, min(1.0, x)))


def normalize_grid(grid):
    gmin = float(np.min(grid))
    gmax = float(np.max(grid))
    if gmax - gmin < CONFIG['dead_threshold']:
        return np.zeros_like(grid, dtype=np.float64), False
    norm = (grid.astype(np.float64) - gmin) / (gmax - gmin)
    return norm, True


def get_local_slice(arr, center, radius):
    lo = max(0, center - radius)
    hi = min(len(arr), center + radius + 1)
    return slice(lo, hi)


def extract_local_patch(grid, dx, dy, radius):
    ci = len(dy) // 2
    cj = len(dx) // 2
    si = get_local_slice(dy, ci, radius)
    sj = get_local_slice(dx, cj, radius)
    return grid[si, sj], dx[sj], dy[si]


def safe_corr(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    corr = np.dot(a, b) / denom
    return clamp01(corr)


def compute_quadratic_fit(cost_grid, dx, dy):
    DX, DY = np.meshgrid(dx, dy)
    X = np.column_stack([DX.ravel(), DY.ravel()])
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    y = cost_grid.ravel()

    if np.std(y) < 1e-12:
        return 0.0, {'xx': 0, 'yy': 0, 'xy': 0, 'x': 0, 'y': 0, 'const': 0}, 0.0

    reg = LinearRegression().fit(X_poly, y)
    y_pred = reg.predict(X_poly)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    r2 = max(r2, 0.0)

    coef = reg.coef_
    coeffs = {
        'const': reg.intercept_,
        'x': coef[1],
        'y': coef[2],
        'xx': coef[3],
        'xy': coef[4],
        'yy': coef[5],
    }

    H = np.array([
        [2 * coeffs['xx'], coeffs['xy']],
        [coeffs['xy'], 2 * coeffs['yy']]
    ])
    eigvals = np.linalg.eigvalsh(H)
    min_eigval = float(np.min(eigvals))
    return r2, coeffs, min_eigval


def compute_symmetry_score(cost_grid, radius):
    h, w = cost_grid.shape
    cy = h // 2
    cx = w // 2

    ys = slice(max(0, cy - radius), min(h, cy + radius + 1))
    xs = slice(max(0, cx - radius), min(w, cx + radius + 1))
    patch = cost_grid[ys, xs]

    if patch.size == 0:
        return 0.0

    patch_rot = np.rot90(patch, 2)
    dyn = float(np.max(patch) - np.min(patch))
    if dyn < 1e-12:
        return 0.0

    mad = float(np.mean(np.abs(patch - patch_rot))) / dyn
    return clamp01(1.0 - mad)


def compute_local_quadratic_bowl_score(cost_grid, dx, dy):
    radius = CONFIG['local_radius']
    local_grid, local_dx, local_dy = extract_local_patch(cost_grid, dx, dy, radius)

    r2, coeffs, min_eigval = compute_quadratic_fit(local_grid, local_dx, local_dy)
    symmetry = compute_symmetry_score(cost_grid, radius)

    scale = CONFIG['convexity_scale']
    if min_eigval > 0:
        convexity_factor = min_eigval / (min_eigval + scale)
    else:
        convexity_factor = 0.0

    lqbs = r2 * convexity_factor * symmetry
    return lqbs, r2, min_eigval, coeffs, symmetry


def compute_connected_basin_width(cost_grid, dx, dy, threshold=0.20):
    h, w = cost_grid.shape
    cy = h // 2
    cx = w // 2

    if np.std(cost_grid) < 1e-12:
        return 0.0

    mask = cost_grid <= threshold
    if not mask[cy, cx]:
        return 0.0

    visited = np.zeros_like(mask, dtype=bool)
    q = deque([(cy, cx)])
    visited[cy, cx] = True
    component = []

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while q:
        y, x = q.popleft()
        component.append((y, x))
        for dy0, dx0 in neighbors:
            ny, nx = y + dy0, x + dx0
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    dists = [np.sqrt((y - cy) ** 2 + (x - cx) ** 2) for y, x in component]
    return float(max(dists)) if dists else 0.0


def compute_local_convergence_success_rate(cost_grid, dx, dy):
    if np.std(cost_grid) < 1e-12:
        return 0.0

    h, w = cost_grid.shape
    cy = h // 2
    cx = w // 2
    radius = CONFIG['local_radius']

    grad_y, grad_x = np.gradient(cost_grid)

    success = 0
    total = 0
    for i in range(max(0, cy - radius), min(h, cy + radius + 1)):
        for j in range(max(0, cx - radius), min(w, cx + radius + 1)):
            if i == cy and j == cx:
                continue

            neg_gx = -grad_x[i, j]
            neg_gy = -grad_y[i, j]
            to_center_x = cx - j
            to_center_y = cy - i
            dot = neg_gx * to_center_x + neg_gy * to_center_y

            if dot > 0:
                success += 1
            total += 1

    return success / total if total > 0 else 0.0


def compute_local_sharpness(cost_grid, dx, dy):
    h, w = cost_grid.shape
    cy = h // 2
    cx = w // 2

    if cx <= 0 or cx >= w - 1 or cy <= 0 or cy >= h - 1:
        return 0.0

    step_x = dx[1] - dx[0] if len(dx) > 1 else 1.0
    step_y = dy[1] - dy[0] if len(dy) > 1 else 1.0

    fxx = (cost_grid[cy, cx + 1] - 2 * cost_grid[cy, cx] + cost_grid[cy, cx - 1]) / (step_x ** 2)
    fyy = (cost_grid[cy + 1, cx] - 2 * cost_grid[cy, cx] + cost_grid[cy - 1, cx]) / (step_y ** 2)

    if fxx > 0 and fyy > 0:
        return float(np.sqrt(fxx * fyy))
    return 0.0


def compute_condition_number(cost_grid, dx, dy):
    h, w = cost_grid.shape
    cy = h // 2
    cx = w // 2

    if cx <= 0 or cx >= w - 1 or cy <= 0 or cy >= h - 1:
        return float('inf')

    step_x = dx[1] - dx[0] if len(dx) > 1 else 1.0
    step_y = dy[1] - dy[0] if len(dy) > 1 else 1.0

    fxx = (cost_grid[cy, cx + 1] - 2 * cost_grid[cy, cx] + cost_grid[cy, cx - 1]) / (step_x ** 2)
    fyy = (cost_grid[cy + 1, cx] - 2 * cost_grid[cy, cx] + cost_grid[cy - 1, cx]) / (step_y ** 2)
    fxy = (
        cost_grid[cy + 1, cx + 1] - cost_grid[cy + 1, cx - 1]
        - cost_grid[cy - 1, cx + 1] + cost_grid[cy - 1, cx - 1]
    ) / (4 * step_x * step_y)

    H = np.array([[fxx, fxy], [fxy, fyy]])
    eigvals = np.linalg.eigvalsh(H)
    if np.min(np.abs(eigvals)) < 1e-12:
        return float('inf')
    return float(np.max(np.abs(eigvals)) / np.min(np.abs(eigvals)))


def compute_shape_similarity(grid_clean_norm, grid_bright_norm):
    if grid_clean_norm is None or grid_bright_norm is None:
        return 0.0
    if np.std(grid_clean_norm) < 1e-12 or np.std(grid_bright_norm) < 1e-12:
        return 0.0

    global_corr = safe_corr(grid_clean_norm, grid_bright_norm)

    radius = CONFIG['retention_local_radius']
    clean_local, _, _ = extract_local_patch(
        grid_clean_norm,
        np.arange(grid_clean_norm.shape[1]),
        np.arange(grid_clean_norm.shape[0]),
        radius
    )
    bright_local, _, _ = extract_local_patch(
        grid_bright_norm,
        np.arange(grid_bright_norm.shape[1]),
        np.arange(grid_bright_norm.shape[0]),
        radius
    )
    local_corr = safe_corr(clean_local, bright_local)

    return 0.4 * global_corr + 0.6 * local_corr


def compute_sharpness_retention(sharp_clean, sharp_bright):
    if sharp_clean <= 1e-12:
        return 0.0
    ratio = sharp_bright / sharp_clean
    return clamp01(min(ratio, 1.0))


def compute_min_position_consistency(cost_grid, radius):
    h, w = cost_grid.shape
    cy = h // 2
    cx = w // 2

    ys = slice(max(0, cy - radius), min(h, cy + radius + 1))
    xs = slice(max(0, cx - radius), min(w, cx + radius + 1))
    patch = cost_grid[ys, xs]

    if patch.size == 0:
        return 0.0

    min_idx = np.unravel_index(np.argmin(patch), patch.shape)
    py, px = min_idx

    local_cy = patch.shape[0] // 2
    local_cx = patch.shape[1] // 2

    dist = np.sqrt((py - local_cy) ** 2 + (px - local_cx) ** 2)
    max_dist = np.sqrt(local_cy ** 2 + local_cx ** 2)
    if max_dist < 1e-12:
        return 1.0

    score = 1.0 - dist / max_dist
    return clamp01(score)


def save_single_metric_ranking(df, metric_name, output_path):
    rank_df = df[['channel', metric_name]].copy()
    rank_df = rank_df.sort_values(metric_name, ascending=False).reset_index(drop=True)
    rank_df['rank'] = range(1, len(rank_df) + 1)
    rank_df = rank_df[['rank', 'channel', metric_name]]
    rank_df.to_csv(output_path, index=False, float_format='%.6f')
    return rank_df


def save_and_print_metric_rankings(ranking_full, rankings_dir):
    os.makedirs(rankings_dir, exist_ok=True)

    metric_specs = [
        ('local_quadratic_bowl_score', 'lqbs'),
        ('basin_width', 'basin_width'),
        ('local_sharpness', 'local_sharpness'),
        ('local_convergence_sr', 'local_convergence_sr'),
        ('shape_sim_30', 'shape_sim_30'),
        ('shape_sim_50', 'shape_sim_50'),
        ('sharp_ret_30', 'sharp_ret_30'),
        ('sharp_ret_50', 'sharp_ret_50'),
        ('minpos_30', 'minpos_30'),
        ('minpos_50', 'minpos_50'),
        ('bright_sym_30', 'bright_sym_30'),
        ('bright_sym_50', 'bright_sym_50'),
        ('retention_30', 'retention_30'),
        ('retention_50', 'retention_50'),
        ('shape_retention_avg', 'retention_avg'),
        ('basin_quality', 'basin_quality'),
        ('bqs_score', 'bqs_score'),
    ]

    print("\n=== Per-Metric Rankings ===")
    for col, file_stub in metric_specs:
        out_csv = os.path.join(rankings_dir, f'ranking_{file_stub}.csv')
        rank_df = save_single_metric_ranking(ranking_full, col, out_csv)
        print(f"[INFO] Saved: {out_csv}")
        print(f"Top 10 by {col}:")
        for _, row in rank_df.head(10).iterrows():
            print(f"  rank={int(row['rank']):<2} Ch{int(row['channel']):<2} {col}={row[col]:.4f}")
        print()


def compute_entity_metrics(name, grids_by_condition, dx, dy):
    """
    Compute the full metric family for a single entity:
    e.g. one channel or gray baseline.
    """
    per_cond = {}
    norm_grids = {}

    for cond, grid in grids_by_condition.items():
        grid_norm, valid = normalize_grid(grid)
        norm_grids[cond] = grid_norm

        if not valid:
            per_cond[cond] = {
                'local_quadratic_bowl_score': 0.0,
                'r2': 0.0,
                'min_eigval': 0.0,
                'symmetry_score': 0.0,
                'basin_width': 0.0,
                'local_convergence_sr': 0.0,
                'local_sharpness': 0.0,
                'condition_number': float('inf'),
            }
            continue

        lqbs, r2, min_eigval, _, sym = compute_local_quadratic_bowl_score(grid_norm, dx, dy)
        width = compute_connected_basin_width(grid_norm, dx, dy, threshold=CONFIG['width_threshold'])
        csr = compute_local_convergence_success_rate(grid_norm, dx, dy)
        sharpness = compute_local_sharpness(grid_norm, dx, dy)
        cond_num = compute_condition_number(grid_norm, dx, dy)

        per_cond[cond] = {
            'local_quadratic_bowl_score': lqbs,
            'r2': r2,
            'min_eigval': min_eigval,
            'symmetry_score': sym,
            'basin_width': width,
            'local_convergence_sr': csr,
            'local_sharpness': sharpness,
            'condition_number': cond_num,
        }

    clean = per_cond['clean']
    shape_sim_30 = compute_shape_similarity(norm_grids['clean'], norm_grids['bright30'])
    shape_sim_50 = compute_shape_similarity(norm_grids['clean'], norm_grids['bright50'])

    sharp_ret_30 = compute_sharpness_retention(clean['local_sharpness'], per_cond['bright30']['local_sharpness'])
    sharp_ret_50 = compute_sharpness_retention(clean['local_sharpness'], per_cond['bright50']['local_sharpness'])

    minpos_30 = compute_min_position_consistency(norm_grids['bright30'], CONFIG['retention_local_radius'])
    minpos_50 = compute_min_position_consistency(norm_grids['bright50'], CONFIG['retention_local_radius'])

    bright_sym_30 = per_cond['bright30']['symmetry_score']
    bright_sym_50 = per_cond['bright50']['symmetry_score']

    ret_30 = shape_sim_30 * sharp_ret_30 * minpos_30 * bright_sym_30
    ret_50 = shape_sim_50 * sharp_ret_50 * minpos_50 * bright_sym_50
    ret_avg = 0.4 * ret_30 + 0.6 * ret_50

    return {
        'name': name,
        'clean_lqbs': clean['local_quadratic_bowl_score'],
        'clean_r2': clean['r2'],
        'clean_min_eigval': clean['min_eigval'],
        'clean_symmetry': clean['symmetry_score'],
        'clean_width': clean['basin_width'],
        'clean_csr': clean['local_convergence_sr'],
        'clean_sharpness': clean['local_sharpness'],
        'shape_sim_30': shape_sim_30,
        'shape_sim_50': shape_sim_50,
        'sharp_ret_30': sharp_ret_30,
        'sharp_ret_50': sharp_ret_50,
        'minpos_30': minpos_30,
        'minpos_50': minpos_50,
        'bright_sym_30': bright_sym_30,
        'bright_sym_50': bright_sym_50,
        'retention_30': ret_30,
        'retention_50': ret_50,
        'retention_avg': ret_avg,
    }


def compute_gray_baseline(raw_dir, dx, dy, output_dir):
    gray_paths = {
        'clean': os.path.join(raw_dir, 'gray_clean.npy'),
        'bright30': os.path.join(raw_dir, 'gray_bright30.npy'),
        'bright50': os.path.join(raw_dir, 'gray_bright50.npy'),
    }

    missing = [k for k, p in gray_paths.items() if not os.path.exists(p)]
    if missing:
        print(f"[WARN] Gray baseline files missing: {missing}")
        print("[WARN] Re-run compute_basin_data.py to generate gray baseline raw data.")
        return

    grids = {cond: np.load(path) for cond, path in gray_paths.items()}
    gray = compute_entity_metrics('gray_baseline', grids, dx, dy)

    clean_df = pd.read_csv(os.path.join(output_dir, 'channel_ranking.csv'))
    max_lqbs = max(float(clean_df['local_quadratic_bowl_score'].max()), 1e-12)
    max_width = max(float(clean_df['basin_width'].max()), 1e-12)

    lqbs_norm = gray['clean_lqbs'] / max_lqbs
    width_norm = gray['clean_width'] / max_width
    basin_quality = (
        CONFIG['basin_weights']['local_quadratic_bowl'] * lqbs_norm +
        CONFIG['basin_weights']['basin_width'] * width_norm
    )
    bqs = basin_quality * gray['retention_avg']

    gray_record = {
        'entity': 'gray_baseline',
        'bqs_score': bqs,
        'basin_quality': basin_quality,
        'clean_lqbs': gray['clean_lqbs'],
        'clean_r2': gray['clean_r2'],
        'clean_min_eigval': gray['clean_min_eigval'],
        'clean_symmetry': gray['clean_symmetry'],
        'clean_width': gray['clean_width'],
        'clean_csr': gray['clean_csr'],
        'clean_sharpness': gray['clean_sharpness'],
        'shape_sim_30': gray['shape_sim_30'],
        'shape_sim_50': gray['shape_sim_50'],
        'sharp_ret_30': gray['sharp_ret_30'],
        'sharp_ret_50': gray['sharp_ret_50'],
        'minpos_30': gray['minpos_30'],
        'minpos_50': gray['minpos_50'],
        'bright_sym_30': gray['bright_sym_30'],
        'bright_sym_50': gray['bright_sym_50'],
        'retention_30': gray['retention_30'],
        'retention_50': gray['retention_50'],
        'retention_avg': gray['retention_avg'],
    }

    gray_df = pd.DataFrame([gray_record])
    gray_csv = os.path.join(output_dir, 'gray_baseline_metrics.csv')
    gray_df.to_csv(gray_csv, index=False, float_format='%.6f')

    print("\n=== Gray Baseline ===")
    print(f"[INFO] Saved: {gray_csv}")
    print(f"Gray BQS:          {gray_record['bqs_score']:.4f}")
    print(f"Gray BasinQuality: {gray_record['basin_quality']:.4f}")
    print(f"Gray LQBS:         {gray_record['clean_lqbs']:.4f}")
    print(f"Gray Width:        {gray_record['clean_width']:.2f}")
    print(f"Gray Ret50:        {gray_record['retention_50']:.4f}")
    print(f"Gray RetAvg:       {gray_record['retention_avg']:.4f}")


def plot_ranking_bar_chart(ranking_df, output_path):
    fig, ax = plt.subplots(figsize=(16, 6))
    max_score = max(float(ranking_df['bqs_score'].max()), 1e-6)
    colors = plt.cm.RdYlGn(ranking_df['bqs_score'].values / max_score)
    ax.bar(range(len(ranking_df)), ranking_df['bqs_score'].values, color=colors, edgecolor='none')

    for idx in range(min(10, len(ranking_df))):
        ch = int(ranking_df.iloc[idx]['channel'])
        score = float(ranking_df.iloc[idx]['bqs_score'])
        ax.annotate(f'Ch{ch}', xy=(idx, score), xytext=(idx, score + 0.005),
                    ha='center', fontsize=7, fontweight='bold')

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Basin Quality Score (BQS)', fontsize=12)
    ax.set_title(
        'Channel Ranking by Basin Quality Score (v5.7)\n'
        'BQS = (0.75×LQBS + 0.25×Width) × Retention',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlim(-0.5, len(ranking_df) - 0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_r2_vs_convexity(metrics_df, output_path):
    clean = metrics_df[metrics_df['condition'] == 'clean'].copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    valid = clean[clean['min_eigval'] > -0.02].copy()
    sc = ax.scatter(
        valid['r2'], valid['min_eigval'],
        c=valid['local_quadratic_bowl_score'],
        cmap='plasma', s=70, edgecolors='k', linewidths=0.4
    )
    ax.set_xlabel('Local Quadratic Fit R²', fontsize=12)
    ax.set_ylabel('Min Eigenvalue of Local Fitted Hessian', fontsize=12)
    ax.set_title('Local R² vs Convexity\n(color = LQBS)', fontsize=11, fontweight='bold')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label='LQBS')

    ax = axes[1]
    sc = ax.scatter(
        clean['local_quadratic_bowl_score'], clean['basin_width'],
        c=clean['local_sharpness'],
        cmap='viridis', s=70, edgecolors='k', linewidths=0.4
    )
    ax.set_xlabel('Local Quadratic Bowl Score', fontsize=12)
    ax.set_ylabel('Connected Basin Width', fontsize=12)
    ax.set_title('LQBS vs Basin Width\n(color = Local Sharpness)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label='Local Sharpness')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shape_retention(ranking_df, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    channels = ranking_df['channel'].values.astype(int)

    ax = axes[0]
    ret30 = ranking_df['retention_30'].values
    colors = plt.cm.RdYlGn(ret30 / max(float(ret30.max()), 1e-6))
    ax.bar(range(len(channels)), ret30, color=colors, edgecolor='none')
    ax.set_ylabel('Retention (+30%)', fontsize=11)
    ax.set_title(
        'Retention = Shape Similarity × Sharpness Retention × MinPos × Bright Symmetry',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlim(-0.5, len(channels) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    ret50 = ranking_df['retention_50'].values
    colors = plt.cm.RdYlGn(ret50 / max(float(ret50.max()), 1e-6))
    ax.bar(range(len(channels)), ret50, color=colors, edgecolor='none')
    ax.set_xlabel('Channel (sorted by BQS rank)', fontsize=11)
    ax.set_ylabel('Retention (+50%)', fontsize=11)
    ax.set_xlim(-0.5, len(channels) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_metric_correlation(metrics_df, output_path):
    clean = metrics_df[metrics_df['condition'] == 'clean'].copy()
    metric_cols = [
        'local_quadratic_bowl_score',
        'r2',
        'min_eigval',
        'symmetry_score',
        'basin_width',
        'local_convergence_sr',
        'local_sharpness',
    ]
    labels = ['LQBS', 'R²', 'Min Eigval', 'Symmetry', 'Width', 'Local CSR', 'Sharpness']

    corr_data = clean[metric_cols].replace([np.inf, -np.inf], np.nan).dropna()
    corr_matrix = corr_data.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, vmin=-1, vmax=1, square=True,
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_title('Metric Correlation Matrix (Clean, v5.7)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_top10_1d_slices(ranking_df, raw_data_dir, dx, dy, output_path):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    top10 = ranking_df.head(10)

    for idx, (_, row) in enumerate(top10.iterrows()):
        ch = int(row['channel'])
        ax = axes[idx]
        npy_path = os.path.join(raw_data_dir, f'channel_{ch:02d}_clean.npy')
        if not os.path.exists(npy_path):
            ax.set_title(f'Ch{ch} (missing)', fontsize=10)
            continue

        grid = np.load(npy_path)
        grid_norm, valid = normalize_grid(grid)
        if not valid:
            ax.set_title(f'Ch{ch} (dead)', fontsize=10)
            continue

        cy = grid.shape[0] // 2
        cx = grid.shape[1] // 2
        x_slice = grid_norm[cy, :]
        y_slice = grid_norm[:, cx]

        ax.plot(dx, x_slice, 'b-', linewidth=1.5, label='x-slice')
        ax.plot(dy, y_slice, 'r--', linewidth=1.5, label='y-slice')

        radius = CONFIG['local_radius']
        x_local = dx[max(0, cx - radius): min(len(dx), cx + radius + 1)]
        x_slice_local = x_slice[max(0, cx - radius): min(len(dx), cx + radius + 1)]

        if len(x_local) >= 5:
            X_fit = x_local.reshape(-1, 1)
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X_fit)
            reg_x = LinearRegression().fit(X_poly, x_slice_local)
            ax.plot(x_local, reg_x.predict(X_poly), 'g:', linewidth=1.2, alpha=0.8, label='local quad fit')

        ax.set_title(f"Ch{ch} (BQS={row['bqs_score']:.3f})", fontsize=10, fontweight='bold')
        ax.set_xlabel(r'$\Delta$ (px)', fontsize=8)
        ax.set_ylabel('Norm. Cost', fontsize=8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.suptitle('Top 10 Channels — 1D Cost Slices (Clean)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_channel_comparison(raw_data_dir, channels, dx, dy, output_path):
    conditions = CONFIG['conditions']
    cond_labels = CONFIG['condition_labels']
    n_ch = len(channels)
    fig, axes = plt.subplots(n_ch, 3, figsize=(15, 4 * n_ch))
    if n_ch == 1:
        axes = axes.reshape(1, -1)

    for row_idx, ch in enumerate(channels):
        for col_idx, cond in enumerate(conditions):
            ax = axes[row_idx, col_idx]
            npy_path = os.path.join(raw_data_dir, f'channel_{ch:02d}_{cond}.npy')
            if not os.path.exists(npy_path):
                ax.set_title(f'Ch{ch} {cond_labels[cond]} (missing)')
                continue

            grid = np.load(npy_path)
            grid_norm, valid = normalize_grid(grid)
            if not valid:
                ax.set_title(f'Ch{ch} {cond_labels[cond]} (dead)')
                continue

            cy = grid.shape[0] // 2
            cx = grid.shape[1] // 2
            x_slice = grid_norm[cy, :]
            y_slice = grid_norm[:, cx]

            ax.plot(dx, x_slice, 'b-', linewidth=1.5, label='x-slice')
            ax.plot(dy, y_slice, 'r--', linewidth=1.5, label='y-slice')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
            ax.set_title(f'Ch{ch} — {cond_labels[cond]}', fontsize=10, fontweight='bold')

            if col_idx == 0:
                ax.set_ylabel('Norm. Cost', fontsize=9)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    plt.suptitle('Channel Comparison Across Brightness Conditions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    cfg = CONFIG
    raw_dir = cfg['raw_data_dir']
    output_dir = cfg['output_dir']
    plots_dir = cfg['plots_dir']
    rankings_dir = cfg['rankings_dir']

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(rankings_dir, exist_ok=True)

    meta_path = os.path.join(raw_dir, 'meta.npz')
    if not os.path.exists(meta_path):
        print(f"[ERROR] meta.npz not found in {raw_dir}. Run compute_basin_data.py first.")
        return

    meta = np.load(meta_path, allow_pickle=True)
    dx = meta['dx']
    dy = meta['dy']

    print(f"[INFO] Grid: {len(dx)}x{len(dy)}, range: [{dx[0]}, {dx[-1]}]")
    print(f"[INFO] Frame: {meta['frame_index']}")
    if 'protocol' in meta:
        print(f"[INFO] Protocol: {meta['protocol']}")
    print(f"[INFO] Basin weights: {cfg['basin_weights']}")
    print(f"[INFO] Formula: BQS = basin_quality × retention")
    print()

    print("=== Step 1: Computing Per-Condition Metrics ===")
    print("  LQBS, Connected Width, Local CSR (diagnostic), Local Sharpness")
    print()

    results = []
    norm_grids = {}

    for ch in range(cfg['num_channels']):
        for cond in cfg['conditions']:
            npy_path = os.path.join(raw_dir, f'channel_{ch:02d}_{cond}.npy')
            if not os.path.exists(npy_path):
                continue

            grid = np.load(npy_path)
            grid_norm, valid = normalize_grid(grid)
            norm_grids[(ch, cond)] = grid_norm

            if not valid:
                results.append({
                    'channel': ch,
                    'condition': cond,
                    'local_quadratic_bowl_score': 0.0,
                    'r2': 0.0,
                    'min_eigval': 0.0,
                    'symmetry_score': 0.0,
                    'basin_width': 0.0,
                    'local_convergence_sr': 0.0,
                    'local_sharpness': 0.0,
                    'condition_number': float('inf'),
                    'coeff_xx': 0.0,
                    'coeff_yy': 0.0,
                })
                continue

            lqbs, r2, min_eigval, coeffs, sym = compute_local_quadratic_bowl_score(grid_norm, dx, dy)
            width = compute_connected_basin_width(grid_norm, dx, dy, threshold=cfg['width_threshold'])
            csr = compute_local_convergence_success_rate(grid_norm, dx, dy)
            sharpness = compute_local_sharpness(grid_norm, dx, dy)
            cond_num = compute_condition_number(grid_norm, dx, dy)

            results.append({
                'channel': ch,
                'condition': cond,
                'local_quadratic_bowl_score': lqbs,
                'r2': r2,
                'min_eigval': min_eigval,
                'symmetry_score': sym,
                'basin_width': width,
                'local_convergence_sr': csr,
                'local_sharpness': sharpness,
                'condition_number': cond_num,
                'coeff_xx': coeffs['xx'],
                'coeff_yy': coeffs['yy'],
            })

        if (ch + 1) % 16 == 0:
            print(f"  Channels 0-{ch} done")

    metrics_df = pd.DataFrame(results)
    metrics_csv_path = os.path.join(output_dir, 'metrics_summary.csv')
    metrics_df.to_csv(metrics_csv_path, index=False, float_format='%.6f')
    print(f"\n[INFO] Metrics saved: {metrics_csv_path}")

    print("\n=== Step 2: Computing Retention ===")
    print("  retention = shape_similarity × sharpness_retention × minpos_consistency × bright_symmetry")

    sharpness_map = {
        (int(row['channel']), row['condition']): float(row['local_sharpness'])
        for _, row in metrics_df.iterrows()
    }
    symmetry_map = {
        (int(row['channel']), row['condition']): float(row['symmetry_score'])
        for _, row in metrics_df.iterrows()
    }

    retention_scores = {}

    for ch in range(cfg['num_channels']):
        clean_grid = norm_grids.get((ch, 'clean'))
        bright30_grid = norm_grids.get((ch, 'bright30'))
        bright50_grid = norm_grids.get((ch, 'bright50'))

        sharp_clean = sharpness_map.get((ch, 'clean'), 0.0)
        sharp_30 = sharpness_map.get((ch, 'bright30'), 0.0)
        sharp_50 = sharpness_map.get((ch, 'bright50'), 0.0)

        sym_30 = symmetry_map.get((ch, 'bright30'), 0.0)
        sym_50 = symmetry_map.get((ch, 'bright50'), 0.0)

        if clean_grid is None or np.std(clean_grid) < 1e-12:
            retention_scores[ch] = {
                'ret_30': 0.0, 'ret_50': 0.0, 'ret_avg': 0.0,
                'shape_sim_30': 0.0, 'shape_sim_50': 0.0,
                'sharp_ret_30': 0.0, 'sharp_ret_50': 0.0,
                'minpos_30': 0.0, 'minpos_50': 0.0,
                'bright_sym_30': 0.0, 'bright_sym_50': 0.0,
            }
            continue

        shape_sim_30 = compute_shape_similarity(clean_grid, bright30_grid) if bright30_grid is not None else 0.0
        shape_sim_50 = compute_shape_similarity(clean_grid, bright50_grid) if bright50_grid is not None else 0.0

        sharp_ret_30 = compute_sharpness_retention(sharp_clean, sharp_30)
        sharp_ret_50 = compute_sharpness_retention(sharp_clean, sharp_50)

        minpos_30 = compute_min_position_consistency(bright30_grid, cfg['retention_local_radius']) if bright30_grid is not None else 0.0
        minpos_50 = compute_min_position_consistency(bright50_grid, cfg['retention_local_radius']) if bright50_grid is not None else 0.0

        bright_sym_30 = sym_30
        bright_sym_50 = sym_50

        ret_30 = shape_sim_30 * sharp_ret_30 * minpos_30 * bright_sym_30
        ret_50 = shape_sim_50 * sharp_ret_50 * minpos_50 * bright_sym_50
        ret_avg = 0.4 * ret_30 + 0.6 * ret_50

        retention_scores[ch] = {
            'ret_30': ret_30,
            'ret_50': ret_50,
            'ret_avg': ret_avg,
            'shape_sim_30': shape_sim_30,
            'shape_sim_50': shape_sim_50,
            'sharp_ret_30': sharp_ret_30,
            'sharp_ret_50': sharp_ret_50,
            'minpos_30': minpos_30,
            'minpos_50': minpos_50,
            'bright_sym_30': bright_sym_30,
            'bright_sym_50': bright_sym_50,
        }

    print("  Done.")

    print("\n=== Step 3: Computing Basin Quality Score (v5.7) ===")
    print("  basin_quality = 0.75*LQBS + 0.25*Width")
    print("  retention = shape_similarity * sharpness_retention * minpos_consistency * bright_symmetry")
    print("  BQS = basin_quality × weighted_average_retention")
    print("  Local CSR is diagnostic only and excluded from final ranking")

    clean_df = metrics_df[metrics_df['condition'] == 'clean'].copy()
    weights = cfg['basin_weights']

    max_lqbs = clean_df['local_quadratic_bowl_score'].max()
    max_lqbs = max_lqbs if max_lqbs > 0 else 1.0

    max_width = clean_df['basin_width'].max()
    max_width = max_width if max_width > 0 else 1.0

    bqs_records = []
    for _, row in clean_df.iterrows():
        ch = int(row['channel'])
        ret = retention_scores.get(ch, {
            'ret_30': 0.0, 'ret_50': 0.0, 'ret_avg': 0.0,
            'shape_sim_30': 0.0, 'shape_sim_50': 0.0,
            'sharp_ret_30': 0.0, 'sharp_ret_50': 0.0,
            'minpos_30': 0.0, 'minpos_50': 0.0,
            'bright_sym_30': 0.0, 'bright_sym_50': 0.0,
        })

        lqbs_norm = float(row['local_quadratic_bowl_score']) / max_lqbs
        width_norm = float(row['basin_width']) / max_width

        basin_quality = (
            weights['local_quadratic_bowl'] * lqbs_norm +
            weights['basin_width'] * width_norm
        )

        bqs = basin_quality * float(ret['ret_avg'])

        bqs_records.append({
            'channel': ch,
            'bqs_score': max(bqs, 0.0),
            'basin_quality': basin_quality,
            'retention_30': ret['ret_30'],
            'retention_50': ret['ret_50'],
            'shape_retention_avg': ret['ret_avg'],
            'shape_sim_30': ret['shape_sim_30'],
            'shape_sim_50': ret['shape_sim_50'],
            'sharp_ret_30': ret['sharp_ret_30'],
            'sharp_ret_50': ret['sharp_ret_50'],
            'minpos_30': ret['minpos_30'],
            'minpos_50': ret['minpos_50'],
            'bright_sym_30': ret['bright_sym_30'],
            'bright_sym_50': ret['bright_sym_50'],
        })

    ranking_df = pd.DataFrame(bqs_records).sort_values('bqs_score', ascending=False).reset_index(drop=True)
    ranking_df['rank'] = range(1, len(ranking_df) + 1)

    ranking_full = ranking_df.merge(
        clean_df[['channel', 'local_quadratic_bowl_score', 'r2', 'min_eigval',
                  'symmetry_score', 'basin_width', 'local_convergence_sr',
                  'local_sharpness', 'condition_number', 'coeff_xx', 'coeff_yy']],
        on='channel', how='left'
    )

    ranking_csv_path = os.path.join(output_dir, 'channel_ranking.csv')
    ranking_full.to_csv(ranking_csv_path, index=False, float_format='%.6f')
    print(f"[INFO] Ranking saved: {ranking_csv_path}")

    print(f"\n=== Top 15 Channels by Basin Quality Score (v5.7) ===")
    print(f"{'Rank':<5} {'Ch':<5} {'BQS':<8} {'BQ':<8} {'LQBS':<8} {'Width':<6} "
          f"{'Ret50':<7} {'ShR50':<7} {'MP50':<7} {'BS50':<7} {'RetAvg':<7}")
    print("-" * 110)
    for _, row in ranking_full.head(15).iterrows():
        print(f"{int(row['rank']):<5} {int(row['channel']):<5} {row['bqs_score']:<8.4f} "
              f"{row['basin_quality']:<8.4f} {row['local_quadratic_bowl_score']:<8.4f} "
              f"{row['basin_width']:<6.2f} {row['retention_50']:<7.3f} "
              f"{row['sharp_ret_50']:<7.3f} {row['minpos_50']:<7.3f} "
              f"{row['bright_sym_50']:<7.3f} {row['shape_retention_avg']:<7.3f}")

    print("\n=== Step 4: Saving Per-Metric Rankings ===")
    save_and_print_metric_rankings(ranking_full, rankings_dir)

    print("\n=== Step 5: Gray Baseline ===")
    compute_gray_baseline(raw_dir, dx, dy, output_dir)

    print("\n=== Generating Plots ===")
    plot_ranking_bar_chart(ranking_full, os.path.join(plots_dir, 'ranking_bar_chart.png'))
    plot_r2_vs_convexity(metrics_df, os.path.join(plots_dir, 'r2_vs_convexity.png'))
    plot_metric_correlation(metrics_df, os.path.join(plots_dir, 'metric_correlation.png'))
    plot_top10_1d_slices(ranking_full, raw_dir, dx, dy, os.path.join(plots_dir, 'top10_1d_slices.png'))
    plot_shape_retention(ranking_full, os.path.join(plots_dir, 'shape_retention.png'))
    plot_channel_comparison(raw_dir, [21, 32, 45, 44, 28, 6, 52, 60],
                            dx, dy, os.path.join(plots_dir, 'channel_comparison_key.png'))

    print(f"\n[DONE] All results in: {output_dir}/")
    print("  v5.7 key design:")
    print("    BQS = basin_quality × retention")
    print("    retention = shape_similarity × sharpness_retention × minpos_consistency × bright_symmetry")
    print("    per-metric ranking CSVs saved in metric_rankings/")
    print("    gray baseline metrics saved in gray_baseline_metrics.csv")


if __name__ == '__main__':
    main()