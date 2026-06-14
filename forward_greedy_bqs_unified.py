"""
forward_greedy_bqs_unified.py
======================================================================
Forward Greedy Selection based on Basin Quality Score (BQS)
Unified version with Beam Search.
Supports target layer selection (conv1 / layer1 / layer2) and
ReLU mode (pre / post).

IMPORTANT — Two formula sets, matching last week's scripts exactly:

  conv1  → "old formula"  (same as forward_greedy_bqs.py last week)
    - Cost landscape : GPU grid_sample + padding_mode='border'
    - LQBS           : polynomial fit R² × convexity × symmetry (local patch)
    - Width          : BFS connected-component radius, threshold=0.20
    - BQ             : 0.75*(lqbs/MAX_LQBS) + 0.25*(width/MAX_WIDTH)
                       MAX_LQBS=0.55, MAX_WIDTH=10.0
    - Retention      : 0.4*ret30 + 0.6*ret50, shape_sim=0.4*global+0.6*local

  layer1 / layer2  → "new formula"  (same as visualize_basin_layer1/2_64/128ch.py)
    - Cost landscape : cv2.warpAffine + BORDER_REPLICATE (CPU)
    - LQBS           : corrcoef(normalised_grid, ideal_paraboloid)
    - Width          : fraction of pixels below threshold=0.30
    - BQ             : 0.75*lqbs + 0.25*width
    - Retention      : shape_sim * sharp_ret * minpos * sym  (equal weight)

This ensures each layer's results are directly comparable to last week's
single-channel analysis for that layer.
"""

import os
import glob
import json
import argparse
import time
import csv
from collections import deque
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir':   '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'device':    'cuda' if torch.cuda.is_available() else 'cpu',

    'max_shift_px': 30,
    'grid_size':    61,
    'dead_threshold': 1e-8,

    'conditions': [
        ('clean',    0.0),
        ('bright30', 0.3),
        ('bright50', 0.5),
    ],

    'target_frames': [306],

    # ── conv1 (old formula) specific ──
    'conv1': {
        'local_radius':           10,
        'retention_local_radius': 10,
        'width_threshold':        0.20,
        'convexity_scale':        0.002,
        'MAX_LQBS':               0.55,
        'MAX_WIDTH':              10.0,
        'chunk_size':             256,   # GPU chunk for grid_sample
    },

    # ── layer1 / layer2 (new formula) specific ──
    'layer12': {
        'width_threshold': 0.30,
    },
}

# ============================================================
# Feature Extractor (Unified)
# ============================================================
class UnifiedExtractor(nn.Module):
    """
    Extracts features from conv1, layer1, or layer2 of ResNet-18.

    target_layer : 'conv1' | 'layer1' | 'layer2'
    use_relu     : True  → post-ReLU
                   False → pre-ReLU (removes the final activation)
    """
    def __init__(self, target_layer='conv1', use_relu=True, device='cuda'):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.target_layer = target_layer
        self.use_relu     = use_relu

        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = nn.ReLU(inplace=False)
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2

        if not use_relu:
            if target_layer == 'conv1':
                self.relu = nn.Identity()
            elif target_layer == 'layer1':
                self.layer1[-1].relu = nn.Identity()
            elif target_layer == 'layer2':
                self.layer2[-1].relu = nn.Identity()

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        orig_size = x.shape[-2:]
        x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.target_layer != 'conv1':
            x = self.maxpool(x)
            x = self.layer1(x)
            if self.target_layer == 'layer2':
                x = self.layer2(x)

        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x  # [1, C, H, W]


# ============================================================
# Image helpers
# ============================================================
def load_image_np(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0

def numpy_to_tensor(img_np, device):
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

def apply_brightness(img_np, factor):
    if factor == 0.0:
        return img_np.copy()
    return np.clip(img_np + factor, 0.0, 1.0)


# ============================================================
# ── OLD FORMULA (conv1) ──
# Cost landscape: GPU grid_sample + padding_mode='border'
# ============================================================
def build_shift_grids_gpu(H, W, dx_vals, dy_vals, device):
    step_x = 2.0 / (W - 1)
    step_y = 2.0 / (H - 1)
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)

    dy_norm = torch.tensor(dy_vals, dtype=torch.float32, device=device) * step_y
    dx_norm = torch.tensor(dx_vals, dtype=torch.float32, device=device) * step_x

    N_dy, N_dx = len(dy_vals), len(dx_vals)
    N = N_dy * N_dx
    grids = base.unsqueeze(0).expand(N, -1, -1, -1).clone()
    idx = 0
    for i in range(N_dy):
        for j in range(N_dx):
            grids[idx, :, :, 0] += dx_norm[j]
            grids[idx, :, :, 1] += dy_norm[i]
            idx += 1
    return grids  # (N, H, W, 2)


@torch.no_grad()
def compute_cost_landscape_gpu(feat_ref_gpu, feat_tgt_gpu, channels,
                                shift_grids, N_dy, N_dx, chunk_size=256):
    """GPU cost landscape for conv1 (old formula). feat_*_gpu: [C, H, W] tensors."""
    N = N_dy * N_dx
    ref_sub = feat_ref_gpu[channels].unsqueeze(0)  # [1, K, H, W]
    tgt_sub = feat_tgt_gpu[channels].unsqueeze(0)  # [1, K, H, W]
    cost_flat = torch.zeros(N, dtype=torch.float32, device=feat_ref_gpu.device)

    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        chunk = end - start
        grids_chunk = shift_grids[start:end]                          # [chunk, H, W, 2]
        tgt_chunk   = tgt_sub.expand(chunk, -1, -1, -1)               # [chunk, K, H, W]
        tgt_shifted = F.grid_sample(tgt_chunk, grids_chunk,
                                    mode='bilinear', padding_mode='border',
                                    align_corners=True)
        ref_chunk   = ref_sub.expand(chunk, -1, -1, -1)
        diff        = tgt_shifted - ref_chunk
        cost_flat[start:end] = diff.pow(2).mean(dim=(1, 2, 3))

    return cost_flat.cpu().numpy().reshape(N_dy, N_dx)


def normalize_grid_old(grid):
    gmin, gmax = float(np.min(grid)), float(np.max(grid))
    if gmax - gmin < CONFIG['dead_threshold']:
        return np.zeros_like(grid, dtype=np.float64), False
    return (grid.astype(np.float64) - gmin) / (gmax - gmin), True

def clamp01(x):
    return float(max(0.0, min(1.0, x)))

def safe_corr(a, b):
    a = a.ravel().astype(np.float64); a -= a.mean()
    b = b.ravel().astype(np.float64); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return clamp01(np.dot(a, b) / denom) if denom > 1e-12 else 0.0

def get_local_slice(n, center, radius):
    return slice(max(0, center - radius), min(n, center + radius + 1))

def extract_local_patch(grid, dx, dy, radius):
    ci, cj = len(dy) // 2, len(dx) // 2
    si = get_local_slice(grid.shape[0], ci, radius)
    sj = get_local_slice(grid.shape[1], cj, radius)
    return grid[si, sj], dx[sj], dy[si]

def compute_quadratic_fit(cost_grid, dx, dy):
    DX, DY = np.meshgrid(dx, dy)
    X = np.column_stack([DX.ravel(), DY.ravel()])
    poly   = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    y = cost_grid.ravel()
    if np.std(y) < 1e-12:
        return 0.0, 0.0
    reg    = LinearRegression().fit(X_poly, y)
    y_pred = reg.predict(X_poly)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2     = max(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0, 0.0)
    coef   = reg.coef_
    H_mat  = np.array([[2*coef[3], coef[4]], [coef[4], 2*coef[5]]])
    return r2, float(np.min(np.linalg.eigvalsh(H_mat)))

def compute_symmetry_score_old(cost_grid, radius):
    h, w = cost_grid.shape
    cy, cx = h // 2, w // 2
    patch = cost_grid[max(0,cy-radius):min(h,cy+radius+1),
                      max(0,cx-radius):min(w,cx+radius+1)]
    if patch.size == 0:
        return 0.0
    dyn = float(np.max(patch) - np.min(patch))
    if dyn < 1e-12:
        return 0.0
    return clamp01(1.0 - float(np.mean(np.abs(patch - np.rot90(patch, 2)))) / dyn)

def compute_lqbs_old(grid_norm, dx, dy):
    radius     = CONFIG['conv1']['local_radius']
    local_grid, local_dx, local_dy = extract_local_patch(grid_norm, dx, dy, radius)
    r2, min_eigval = compute_quadratic_fit(local_grid, local_dx, local_dy)
    sym   = compute_symmetry_score_old(grid_norm, radius)
    scale = CONFIG['conv1']['convexity_scale']
    conv  = min_eigval / (min_eigval + scale) if min_eigval > 0 else 0.0
    return r2 * conv * sym

def compute_basin_width_old(grid_norm, threshold):
    h, w = grid_norm.shape
    cy, cx = h // 2, w // 2
    if np.std(grid_norm) < 1e-12:
        return 0.0
    mask = grid_norm <= threshold
    if not mask[cy, cx]:
        return 0.0
    visited = np.zeros_like(mask, dtype=bool)
    q = deque([(cy, cx)]); visited[cy, cx] = True
    component = []
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while q:
        y, x = q.popleft(); component.append((y, x))
        for dy0, dx0 in neighbors:
            ny, nx = y+dy0, x+dx0
            if 0<=ny<h and 0<=nx<w and mask[ny,nx] and not visited[ny,nx]:
                visited[ny,nx] = True; q.append((ny,nx))
    dists = [np.sqrt((y-cy)**2+(x-cx)**2) for y,x in component]
    return float(max(dists)) if dists else 0.0

def compute_sharpness_old(grid_norm, dx, dy):
    h, w = grid_norm.shape
    cy, cx = h//2, w//2
    if cx<=0 or cx>=w-1 or cy<=0 or cy>=h-1:
        return 0.0
    sx = float(dx[1]-dx[0]) if len(dx)>1 else 1.0
    sy = float(dy[1]-dy[0]) if len(dy)>1 else 1.0
    fxx = (grid_norm[cy,cx+1]-2*grid_norm[cy,cx]+grid_norm[cy,cx-1])/(sx**2)
    fyy = (grid_norm[cy+1,cx]-2*grid_norm[cy,cx]+grid_norm[cy-1,cx])/(sy**2)
    return float(np.sqrt(fxx*fyy)) if fxx>0 and fyy>0 else 0.0

def compute_shape_sim_old(clean_norm, bright_norm):
    if np.std(clean_norm)<1e-12 or np.std(bright_norm)<1e-12:
        return 0.0
    g_corr = safe_corr(clean_norm, bright_norm)
    r = CONFIG['conv1']['retention_local_radius']
    c_local,_,_ = extract_local_patch(clean_norm,
                                       np.arange(clean_norm.shape[1]),
                                       np.arange(clean_norm.shape[0]), r)
    b_local,_,_ = extract_local_patch(bright_norm,
                                       np.arange(bright_norm.shape[1]),
                                       np.arange(bright_norm.shape[0]), r)
    return 0.4*g_corr + 0.6*safe_corr(c_local, b_local)

def compute_minpos_consistency_old(grid_norm, radius):
    h, w = grid_norm.shape
    cy, cx = h//2, w//2
    patch = grid_norm[max(0,cy-radius):min(h,cy+radius+1),
                      max(0,cx-radius):min(w,cx+radius+1)]
    if patch.size == 0:
        return 0.0
    py, px = np.unravel_index(np.argmin(patch), patch.shape)
    lcy, lcx = patch.shape[0]//2, patch.shape[1]//2
    max_dist = np.sqrt(lcy**2+lcx**2)
    if max_dist < 1e-12:
        return 1.0
    return clamp01(1.0 - np.sqrt((py-lcy)**2+(px-lcx)**2)/max_dist)

def compute_bqs_old(grids, dx, dy):
    """
    Old BQS formula for conv1.
    grids: dict with keys 'clean', 'bright30', 'bright50' → 2D numpy arrays
    Returns scalar BQS.
    """
    cfg = CONFIG['conv1']
    clean_raw = grids['clean']
    norm_clean, valid = normalize_grid_old(clean_raw)
    if not valid:
        return 0.0

    lqbs  = compute_lqbs_old(norm_clean, dx, dy)
    width = compute_basin_width_old(norm_clean, cfg['width_threshold'])

    sharp_clean = compute_sharpness_old(norm_clean, dx, dy)

    ret_avg = 0.0
    for w_ret, cond in [(0.4, 'bright30'), (0.6, 'bright50')]:
        bright_raw  = grids[cond]
        norm_bright, bv = normalize_grid_old(bright_raw)
        if not bv:
            continue
        shape_sim = compute_shape_sim_old(norm_clean, norm_bright)
        sharp_b   = compute_sharpness_old(norm_bright, dx, dy)
        sharp_ret = clamp01(sharp_b / sharp_clean) if sharp_clean > 1e-12 else 0.0
        minpos    = compute_minpos_consistency_old(norm_bright,
                                                   cfg['retention_local_radius'])
        sym       = compute_symmetry_score_old(norm_bright, cfg['local_radius'])
        ret_avg  += w_ret * (shape_sim * sharp_ret * minpos * sym)

    lqbs_norm  = lqbs  / cfg['MAX_LQBS']  if cfg['MAX_LQBS']  > 0 else lqbs
    width_norm = width / cfg['MAX_WIDTH'] if cfg['MAX_WIDTH'] > 0 else width
    bq  = 0.75 * lqbs_norm + 0.25 * width_norm
    return float(bq * ret_avg)


# ============================================================
# ── NEW FORMULA (layer1 / layer2) ──
# Cost landscape: cv2.warpAffine + BORDER_REPLICATE (CPU)
# ============================================================
def compute_cost_landscape_multichannel_cpu(feat_ref, feat_tgt, channels,
                                             max_shift, grid_size):
    """
    feat_ref, feat_tgt : numpy [C, H, W]
    channels           : list of channel indices
    Returns: dx_vals, dy_vals, cost_grid
    """
    dx_vals   = np.linspace(-max_shift, max_shift, grid_size)
    dy_vals   = np.linspace(-max_shift, max_shift, grid_size)
    h, w      = feat_ref.shape[1], feat_ref.shape[2]
    cost_grid = np.zeros((grid_size, grid_size), dtype=np.float64)

    for ch in channels:
        ref_ch = feat_ref[ch].astype(np.float32)
        tgt_ch = feat_tgt[ch].astype(np.float32)
        for i, dy in enumerate(dy_vals):
            for j, dx in enumerate(dx_vals):
                M       = np.float64([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(tgt_ch, M, (w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
                residual        = shifted.astype(np.float64) - ref_ch.astype(np.float64)
                cost_grid[i, j] += np.mean(residual ** 2)

    cost_grid /= len(channels)
    return dx_vals, dy_vals, cost_grid


def normalize_grid_new(cost_grid):
    c_min, c_max = cost_grid.min(), cost_grid.max()
    if c_max - c_min < CONFIG['dead_threshold']:
        return np.zeros_like(cost_grid), False
    return (cost_grid - c_min) / (c_max - c_min), True

def compute_lqbs_new(cost_grid):
    norm, valid = normalize_grid_new(cost_grid)
    if not valid:
        return 0.0
    grid_size = norm.shape[0]
    center    = grid_size // 2
    half      = center if center > 0 else 1
    y_idx, x_idx = np.mgrid[0:grid_size, 0:grid_size]
    dist2 = ((y_idx - center) / half) ** 2 + ((x_idx - center) / half) ** 2
    ideal = np.clip(dist2, 0, 1)
    corr  = np.corrcoef(norm.ravel(), ideal.ravel())[0, 1]
    return float(max(corr, 0.0))

def compute_basin_width_new(cost_grid, threshold=0.3):
    norm, valid = normalize_grid_new(cost_grid)
    if not valid:
        return 0.0
    return float(np.sum(norm < threshold) / norm.size)

def compute_shape_similarity_new(clean_grid, perturbed_grid):
    c_norm, cv = normalize_grid_new(clean_grid)
    p_norm, pv = normalize_grid_new(perturbed_grid)
    if not cv or not pv:
        return 0.0
    corr = np.corrcoef(c_norm.ravel(), p_norm.ravel())[0, 1]
    return float(max(corr, 0.0))

def compute_sharpness_retention_new(clean_grid, perturbed_grid):
    def _sharp(g):
        n, v = normalize_grid_new(g)
        if not v:
            return 0.0
        center = n.shape[0] // 2
        gx = np.gradient(n[center, :])
        gy = np.gradient(n[:, center])
        return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
    s_clean = _sharp(clean_grid)
    s_pert  = _sharp(perturbed_grid)
    if s_clean < 1e-10:
        return 1.0
    return float(min(s_pert / s_clean, 1.0))

def compute_minpos_consistency_new(clean_grid, perturbed_grid):
    cy, cx = np.unravel_index(np.argmin(clean_grid), clean_grid.shape)
    py, px = np.unravel_index(np.argmin(perturbed_grid), perturbed_grid.shape)
    dist     = np.sqrt((cy - py) ** 2 + (cx - px) ** 2)
    max_dist = np.sqrt(2) * clean_grid.shape[0]
    return float(max(1.0 - dist / max_dist, 0.0))

def compute_symmetry_new(cost_grid):
    n, v = normalize_grid_new(cost_grid)
    if not v:
        return 0.0
    sym_h = 1.0 - np.mean(np.abs(n - np.fliplr(n)))
    sym_v = 1.0 - np.mean(np.abs(n - np.flipud(n)))
    return float((sym_h + sym_v) / 2.0)

def compute_bqs_new(grids):
    """
    New BQS formula for layer1/layer2.
    grids: dict with keys 'clean', 'bright30', 'bright50' → 2D numpy arrays
    Returns scalar BQS.
    """
    cfg   = CONFIG['layer12']
    clean = grids['clean']
    lqbs  = compute_lqbs_new(clean)
    width = compute_basin_width_new(clean, cfg['width_threshold'])
    bq    = 0.75 * lqbs + 0.25 * width

    shape30   = compute_shape_similarity_new(clean, grids['bright30'])
    shape50   = compute_shape_similarity_new(clean, grids['bright50'])
    shape_sim = (shape30 + shape50) / 2.0

    ret30     = compute_sharpness_retention_new(clean, grids['bright30'])
    ret50     = compute_sharpness_retention_new(clean, grids['bright50'])
    sharp_ret = (ret30 + ret50) / 2.0

    minpos30  = compute_minpos_consistency_new(clean, grids['bright30'])
    minpos50  = compute_minpos_consistency_new(clean, grids['bright50'])
    minpos    = (minpos30 + minpos50) / 2.0

    sym30 = compute_symmetry_new(grids['bright30'])
    sym50 = compute_symmetry_new(grids['bright50'])
    sym   = (sym30 + sym50) / 2.0

    retention = shape_sim * sharp_ret * minpos * sym
    return float(bq * retention)


# ============================================================
# BQS Evaluation for a channel subset (dispatches by layer)
# ============================================================
def evaluate_subset_bqs_conv1(channels, frame_indices, all_images, extractor,
                               shift_grids, N_dy, N_dx, dx, dy):
    """Old formula: GPU grid_sample cost landscape."""
    device     = CONFIG['device']
    bqs_scores = []

    for frame_idx in frame_indices:
        rgb_np = load_image_np(all_images[frame_idx])

        feats_gpu = {}
        for cond_name, factor in CONFIG['conditions']:
            tgt_np     = apply_brightness(rgb_np, factor)
            tgt_tensor = numpy_to_tensor(tgt_np, device)
            with torch.no_grad():
                feat = extractor(tgt_tensor)[0]  # [C, H, W] on GPU
            feats_gpu[cond_name] = feat
            del tgt_tensor

        grids = {}
        for cond_name, _ in CONFIG['conditions']:
            grids[cond_name] = compute_cost_landscape_gpu(
                feats_gpu['clean'], feats_gpu[cond_name], channels,
                shift_grids, N_dy, N_dx, CONFIG['conv1']['chunk_size'])

        bqs_scores.append(compute_bqs_old(grids, dx, dy))

        if device == 'cuda':
            torch.cuda.empty_cache()

    return float(np.mean(bqs_scores))


def evaluate_subset_bqs_layer12(channels, frame_indices, all_images, extractor,
                                shift_grids, N_dy, N_dx):
    """New formula (corrcoef LQBS + pixel-fraction width): GPU grid_sample cost landscape."""
    device     = CONFIG['device']
    bqs_scores = []

    for frame_idx in frame_indices:
        rgb_np = load_image_np(all_images[frame_idx])

        feats_gpu = {}
        for cond_name, factor in CONFIG['conditions']:
            tgt_np     = apply_brightness(rgb_np, factor)
            tgt_tensor = numpy_to_tensor(tgt_np, device)
            with torch.no_grad():
                feat = extractor(tgt_tensor)[0]  # [C, H, W] on GPU
            feats_gpu[cond_name] = feat
            del tgt_tensor

        grids = {}
        for cond_name, _ in CONFIG['conditions']:
            cost = compute_cost_landscape_gpu(
                feats_gpu['clean'], feats_gpu[cond_name], channels,
                shift_grids, N_dy, N_dx,
                chunk_size=CONFIG['conv1']['chunk_size'])
            grids[cond_name] = cost

        bqs_scores.append(compute_bqs_new(grids))

        if device == 'cuda':
            torch.cuda.empty_cache()

    return float(np.mean(bqs_scores))


# ============================================================
# Beam Search Greedy (layer-agnostic wrapper)
# ============================================================
def beam_search_greedy(candidate_pool, frame_indices, all_images, extractor,
                       layer_name,
                       shift_grids=None, N_dy=None, N_dx=None, dx=None, dy=None,
                       beam_size=3, min_improvement=1e-4, max_channels=64):

    def eval_fn(channels):
        if layer_name == 'conv1':
            return evaluate_subset_bqs_conv1(
                channels, frame_indices, all_images, extractor,
                shift_grids, N_dy, N_dx, dx, dy)
        else:
            return evaluate_subset_bqs_layer12(
                channels, frame_indices, all_images, extractor,
                shift_grids, N_dy, N_dx)

    # ── Step 1: seed beams ──
    print(f"\n--- Step 1 (seeding {beam_size} beams) ---", flush=True)
    single_scores = {}
    for ch in candidate_pool:
        t0  = time.time()
        bqs = eval_fn([ch])
        elapsed = time.time() - t0
        single_scores[ch] = bqs
        print(f"  ch={ch:3d}  BQS={bqs:.4f}  ({elapsed:.1f}s)", flush=True)

    sorted_singles = sorted(single_scores.items(), key=lambda x: x[1], reverse=True)
    seeds = sorted_singles[:beam_size]
    print(f"\nTop-{beam_size} seeds: {[(ch, f'{s:.4f}') for ch, s in seeds]}", flush=True)

    beams = [([ch], [bqs], bqs) for ch, bqs in seeds]

    step_logs = []
    for ch, bqs in single_scores.items():
        step_logs.append({'step': 1, 'beam': 'seed', 'channel': ch,
                          'subset': str([ch]), 'bqs': bqs, 'marginal': bqs})

    step = 2
    while step <= max_channels:
        print(f"\n--- Step {step} ---", flush=True)
        new_beams    = []
        any_improved = False

        for beam_idx, (subset, bqs_hist, prev_bqs) in enumerate(beams):
            best_bqs_this_beam = prev_bqs
            best_ch_this_beam  = None
            beam_tag = f"beam{beam_idx+1}(seed=ch{subset[0]})"

            for ch in candidate_pool:
                if ch in subset:
                    continue
                test_subset = subset + [ch]
                t0  = time.time()
                bqs = eval_fn(test_subset)
                elapsed = time.time() - t0
                marginal = bqs - prev_bqs

                print(f"  [{beam_tag}] ch={ch:3d}  subset={test_subset}"
                      f"  BQS={bqs:.4f}  Δ={marginal:+.4f}  ({elapsed:.1f}s)", flush=True)

                step_logs.append({'step': step, 'beam': beam_tag, 'channel': ch,
                                  'subset': str(test_subset), 'bqs': bqs,
                                  'marginal': round(marginal, 6)})

                if bqs > best_bqs_this_beam:
                    best_bqs_this_beam = bqs
                    best_ch_this_beam  = ch

            if best_ch_this_beam is not None:
                marginal_gain = best_bqs_this_beam - prev_bqs
                if marginal_gain >= min_improvement:
                    new_subset   = subset + [best_ch_this_beam]
                    new_bqs_hist = bqs_hist + [best_bqs_this_beam]
                    new_beams.append((new_subset, new_bqs_hist, best_bqs_this_beam))
                    any_improved = True
                    print(f"  >>> [{beam_tag}] selected ch={best_ch_this_beam}"
                          f"  subset={new_subset}"
                          f"  BQS={best_bqs_this_beam:.4f}"
                          f"  Δ={marginal_gain:+.4f}", flush=True)
                else:
                    new_beams.append((subset, bqs_hist, prev_bqs))
                    print(f"  >>> [{beam_tag}] FROZEN (Δ={marginal_gain:+.6f} < {min_improvement})"
                          f"  final subset={subset}  BQS={prev_bqs:.4f}", flush=True)
            else:
                new_beams.append((subset, bqs_hist, prev_bqs))
                print(f"  >>> [{beam_tag}] no candidate improved BQS. FROZEN.", flush=True)

        beams = new_beams
        if not any_improved:
            print(f"\n>>> All beams frozen at step {step}. Stopping.", flush=True)
            break
        step += 1

    best_beam   = max(beams, key=lambda b: b[2])
    best_subset, best_bqs_hist, best_bqs = best_beam
    all_paths   = [(b[0], b[1]) for b in beams]
    return best_subset, best_bqs, all_paths, step_logs


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Forward Greedy BQS Channel Selection (Unified)')
    parser.add_argument('--layer',           type=str,   choices=['conv1', 'layer1', 'layer2'],
                        required=True)
    parser.add_argument('--relu_mode',       type=str,   choices=['pre', 'post'],
                        required=True)
    parser.add_argument('--beam_size',       type=int,   default=3)
    parser.add_argument('--min_improvement', type=float, default=1e-4)
    parser.add_argument('--max_channels',    type=int,   default=None)
    parser.add_argument('--frame_indices',   type=int,   nargs='+', default=None)
    parser.add_argument('--output_dir',      type=str,   default=None)
    args = parser.parse_args()

    if args.frame_indices is not None:
        CONFIG['target_frames'] = args.frame_indices

    total_channels = 128 if args.layer == 'layer2' else 64
    if args.max_channels is None:
        args.max_channels = total_channels
    else:
        args.max_channels = min(args.max_channels, total_channels)

    if args.output_dir is None:
        out_dir = f"vis_results/forward_greedy_bqs_{args.layer}_{args.relu_mode}_relu"
    else:
        out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    use_relu = (args.relu_mode == 'post')

    formula_note = ("old formula: GPU grid_sample + poly-fit LQBS + BFS width"
                    if args.layer == 'conv1' else
                    "new formula: GPU grid_sample + corrcoef LQBS + pixel-fraction width")

    print(f"=== Forward Greedy BQS Selection ===")
    print(f"Target Layer    : {args.layer} ({total_channels}ch)")
    print(f"ReLU Mode       : {args.relu_mode}-ReLU")
    print(f"Formula         : {formula_note}")
    print(f"Device          : {CONFIG['device']}")
    print(f"Target frames   : {CONFIG['target_frames']}")
    print(f"Beam size       : {args.beam_size}")
    print(f"Min improvement : {args.min_improvement}")
    print(f"Max channels    : {args.max_channels}")
    print(f"Output Dir      : {out_dir}")

    all_images = sorted(glob.glob(os.path.join(CONFIG['rgb_dir'], "*.png")))
    if not all_images:
        raise FileNotFoundError(f"No images found in {CONFIG['rgb_dir']}")

    device    = CONFIG['device']
    extractor = UnifiedExtractor(target_layer=args.layer, use_relu=use_relu, device=device)

    # Pre-build shift grids (GPU) — used by all layers
    max_shift = CONFIG['max_shift_px']
    grid_size = CONFIG['grid_size']
    dx_vals   = np.linspace(-max_shift, max_shift, grid_size)
    dy_vals   = np.linspace(-max_shift, max_shift, grid_size)
    N_dy, N_dx = len(dy_vals), len(dx_vals)

    # Infer H, W from a test image
    test_img    = load_image_np(all_images[CONFIG['target_frames'][0]])
    test_tensor = numpy_to_tensor(test_img, device)
    with torch.no_grad():
        test_feat = extractor(test_tensor)
    H, W = test_feat.shape[-2], test_feat.shape[-1]
    del test_tensor, test_feat
    if device == 'cuda':
        torch.cuda.empty_cache()

    print(f"\nBuilding shift grids on GPU ... (H={H}, W={W})", flush=True)
    shift_grids = build_shift_grids_gpu(H, W, dx_vals, dy_vals, device)
    print(f"Shift grids built: {shift_grids.shape}", flush=True)

    candidate_pool = list(range(total_channels))
    print(f"Candidate pool  : {len(candidate_pool)} channels", flush=True)

    total_start = time.time()

    best_subset, best_bqs, all_paths, step_logs = beam_search_greedy(
        candidate_pool,
        CONFIG['target_frames'],
        all_images,
        extractor,
        layer_name=args.layer,
        shift_grids=shift_grids,
        N_dy=N_dy, N_dx=N_dx,
        dx=dx_vals, dy=dy_vals,
        beam_size=args.beam_size,
        min_improvement=args.min_improvement,
        max_channels=args.max_channels,
    )

    total_elapsed = time.time() - total_start

    print(f"\n=== Finished in {total_elapsed/60:.1f} min ===")
    print(f"Best subset : {best_subset}")
    print(f"Best BQS    : {best_bqs:.4f}")
    print("\nAll beam paths:")
    for i, (path_subset, path_hist) in enumerate(all_paths):
        print(f"  Beam {i+1} (seed ch={path_subset[0]}): "
              f"subset={path_subset}  BQS={path_hist[-1]:.4f}")

    summary = {
        'layer':           args.layer,
        'relu_mode':       args.relu_mode,
        'formula':         'old (poly-fit)' if args.layer == 'conv1' else 'new (corrcoef, GPU)',
        'best_subset':     best_subset,
        'best_bqs':        round(best_bqs, 6),
        'frames':          CONFIG['target_frames'],
        'beam_size':       args.beam_size,
        'min_improvement': args.min_improvement,
        'total_time_min':  round(total_elapsed / 60, 2),
        'all_beam_paths':  [
            {'seed_channel': path[0][0], 'subset': path[0], 'bqs_history': path[1]}
            for path in all_paths
        ],
    }
    with open(os.path.join(out_dir, 'greedy_selection_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    csv_path = os.path.join(out_dir, 'step_by_step_log.csv')
    if step_logs:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f,
                fieldnames=['step', 'beam', 'channel', 'subset', 'bqs', 'marginal'])
            writer.writeheader()
            writer.writerows(step_logs)

    print(f"\nOutputs saved to: {out_dir}/")
    print(f"  greedy_selection_summary.json")
    print(f"  step_by_step_log.csv")


if __name__ == '__main__':
    main()