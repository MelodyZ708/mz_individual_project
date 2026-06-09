"""
Forward Greedy Selection based on Basin Quality Score (BQS)
===========================================================
GPU-accelerated version with Beam Search to reduce local optima risk.

Key features:
- All 3721 (dx, dy) shifts computed in chunked GPU batches via grid_sample.
- Beam Search: maintains Top-K candidate subsets at each step.
- Early stopping: halts when marginal BQS gain < min_improvement.
- No hard upper limit on subset size (runs until early stopping).
- Dead channel pruning (BQS=0 from channel_ranking.csv).

Usage:
    python forward_greedy_bqs.py [--beam_size 3] [--min_improvement 1e-4]
                                  [--max_channels 60] [--frame_indices 306]

Outputs saved to: vis_results/forward_greedy_bqs/
"""

import os
import glob
import json
import argparse
import time
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'rgb_dir': '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/',
    'output_dir': 'vis_results/forward_greedy_bqs',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'grid_range': 30,
    'grid_step': 1,

    'local_radius': 10,
    'retention_local_radius': 10,
    'width_threshold': 0.20,
    'convexity_scale': 0.002,
    'dead_threshold': 1e-8,

    'basin_weights': {
        'local_quadratic_bowl': 0.75,
        'basin_width': 0.25,
    },

    'conditions': [
        ('clean',    0.0),
        ('bright30', 0.3),
        ('bright50', 0.5),
    ],

    'target_frames': [306],
}

# Dead channels to exclude (BQS = 0 from channel_ranking.csv)
DEAD_CHANNELS = [7, 9, 36, 48]

# Normalisation bounds from single-channel analysis
MAX_LQBS  = 0.55
MAX_WIDTH = 10.0

# ============================================================
# Feature Extractor
# ============================================================
class Conv1BNReLUExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = base.conv1
        self.bn1   = base.bn1
        self.relu  = nn.ReLU(inplace=False)
        self.device = device
        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    @torch.no_grad()
    def forward(self, img_tensor):
        orig_size = img_tensor.shape[-2:]
        x = (img_tensor - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x  # (1, 64, H, W)

# ============================================================
# Image Utils
# ============================================================
def load_image_np(img_path):
    img = Image.open(img_path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0

def numpy_to_tensor(img_np, device):
    return torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)

def apply_brightness(img_np, factor):
    if factor == 0.0:
        return img_np.copy()
    return np.clip(img_np + factor, 0.0, 1.0)

# ============================================================
# GPU-Accelerated Cost Landscape (chunked to avoid OOM)
# ============================================================
def build_shift_grid(H, W, dx_vals, dy_vals, device):
    """Pre-build sampling grids for all (dx, dy) offsets on GPU."""
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
    """Compute cost landscape on GPU in chunks to avoid OOM."""
    N = N_dy * N_dx
    ref_sub = feat_ref_gpu[channels]
    tgt_sub = feat_tgt_gpu[channels]

    cost_flat = torch.zeros(N, dtype=torch.float32, device=feat_ref_gpu.device)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = end - start
        grids_chunk = shift_grids[start:end]

        tgt_chunk   = tgt_sub.unsqueeze(0).expand(chunk, -1, -1, -1)
        tgt_shifted = F.grid_sample(tgt_chunk, grids_chunk,
                                    mode='bilinear', padding_mode='border',
                                    align_corners=True)
        ref_chunk = ref_sub.unsqueeze(0).expand(chunk, -1, -1, -1)
        diff = tgt_shifted - ref_chunk
        cost_flat[start:end] = diff.pow(2).mean(dim=(1, 2, 3))

    return cost_flat.cpu().numpy().reshape(N_dy, N_dx)

# ============================================================
# BQS Metric Computation
# ============================================================
def clamp01(x): return float(max(0.0, min(1.0, x)))

def normalize_grid(grid):
    gmin, gmax = float(np.min(grid)), float(np.max(grid))
    if gmax - gmin < CONFIG['dead_threshold']:
        return np.zeros_like(grid, dtype=np.float64), False
    return (grid.astype(np.float64) - gmin) / (gmax - gmin), True

def get_local_slice(n, center, radius):
    return slice(max(0, center - radius), min(n, center + radius + 1))

def extract_local_patch(grid, dx, dy, radius):
    ci, cj = len(dy) // 2, len(dx) // 2
    si = get_local_slice(grid.shape[0], ci, radius)
    sj = get_local_slice(grid.shape[1], cj, radius)
    return grid[si, sj], dx[sj], dy[si]

def safe_corr(a, b):
    a = a.ravel().astype(np.float64); a -= a.mean()
    b = b.ravel().astype(np.float64); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return clamp01(np.dot(a, b) / denom) if denom > 1e-12 else 0.0

def compute_quadratic_fit(cost_grid, dx, dy):
    DX, DY = np.meshgrid(dx, dy)
    X = np.column_stack([DX.ravel(), DY.ravel()])
    poly  = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)
    y = cost_grid.ravel()
    if np.std(y) < 1e-12: return 0.0, 0.0
    reg   = LinearRegression().fit(X_poly, y)
    y_pred = reg.predict(X_poly)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = max(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0, 0.0)
    coef  = reg.coef_
    H_mat = np.array([[2*coef[3], coef[4]], [coef[4], 2*coef[5]]])
    return r2, float(np.min(np.linalg.eigvalsh(H_mat)))

def compute_symmetry_score(cost_grid, radius):
    h, w = cost_grid.shape
    cy, cx = h // 2, w // 2
    patch = cost_grid[max(0,cy-radius):min(h,cy+radius+1),
                      max(0,cx-radius):min(w,cx+radius+1)]
    if patch.size == 0: return 0.0
    dyn = float(np.max(patch) - np.min(patch))
    if dyn < 1e-12: return 0.0
    return clamp01(1.0 - float(np.mean(np.abs(patch - np.rot90(patch, 2)))) / dyn)

def compute_lqbs(grid_norm, dx, dy):
    radius = CONFIG['local_radius']
    local_grid, local_dx, local_dy = extract_local_patch(grid_norm, dx, dy, radius)
    r2, min_eigval = compute_quadratic_fit(local_grid, local_dx, local_dy)
    sym   = compute_symmetry_score(grid_norm, radius)
    scale = CONFIG['convexity_scale']
    conv  = min_eigval / (min_eigval + scale) if min_eigval > 0 else 0.0
    return r2 * conv * sym

def compute_basin_width(grid_norm, threshold):
    h, w = grid_norm.shape
    cy, cx = h // 2, w // 2
    if np.std(grid_norm) < 1e-12: return 0.0
    mask = grid_norm <= threshold
    if not mask[cy, cx]: return 0.0
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

def compute_sharpness(grid_norm, dx, dy):
    h, w = grid_norm.shape
    cy, cx = h//2, w//2
    if cx<=0 or cx>=w-1 or cy<=0 or cy>=h-1: return 0.0
    sx = float(dx[1]-dx[0]) if len(dx)>1 else 1.0
    sy = float(dy[1]-dy[0]) if len(dy)>1 else 1.0
    fxx = (grid_norm[cy,cx+1]-2*grid_norm[cy,cx]+grid_norm[cy,cx-1])/(sx**2)
    fyy = (grid_norm[cy+1,cx]-2*grid_norm[cy,cx]+grid_norm[cy-1,cx])/(sy**2)
    return float(np.sqrt(fxx*fyy)) if fxx>0 and fyy>0 else 0.0

def compute_shape_sim(clean_norm, bright_norm):
    if np.std(clean_norm)<1e-12 or np.std(bright_norm)<1e-12: return 0.0
    g_corr = safe_corr(clean_norm, bright_norm)
    r = CONFIG['retention_local_radius']
    c_local,_,_ = extract_local_patch(clean_norm,
                                       np.arange(clean_norm.shape[1]),
                                       np.arange(clean_norm.shape[0]), r)
    b_local,_,_ = extract_local_patch(bright_norm,
                                       np.arange(bright_norm.shape[1]),
                                       np.arange(bright_norm.shape[0]), r)
    return 0.4*g_corr + 0.6*safe_corr(c_local, b_local)

def compute_minpos_consistency(grid_norm, radius):
    h, w = grid_norm.shape
    cy, cx = h//2, w//2
    patch = grid_norm[max(0,cy-radius):min(h,cy+radius+1),
                      max(0,cx-radius):min(w,cx+radius+1)]
    if patch.size == 0: return 0.0
    py, px = np.unravel_index(np.argmin(patch), patch.shape)
    lcy, lcx = patch.shape[0]//2, patch.shape[1]//2
    max_dist = np.sqrt(lcy**2+lcx**2)
    if max_dist < 1e-12: return 1.0
    return clamp01(1.0 - np.sqrt((py-lcy)**2+(px-lcx)**2)/max_dist)

# ============================================================
# BQS Evaluation
# ============================================================
def evaluate_subset_bqs(channels, frame_indices, all_images, extractor,
                        dx, dy, shift_grids, N_dy, N_dx):
    device = CONFIG['device']
    bqs_scores = []

    for frame_idx in frame_indices:
        rgb_np     = load_image_np(all_images[frame_idx])
        rgb_tensor = numpy_to_tensor(rgb_np, device)

        with torch.no_grad():
            feat_ref_gpu = extractor(rgb_tensor)[0]

        grids = {}
        for cond_name, factor in CONFIG['conditions']:
            tgt_np     = apply_brightness(rgb_np, factor)
            tgt_tensor = numpy_to_tensor(tgt_np, device)
            with torch.no_grad():
                feat_tgt_gpu = extractor(tgt_tensor)[0]
            grids[cond_name] = compute_cost_landscape_gpu(
                feat_ref_gpu, feat_tgt_gpu, channels,
                shift_grids, N_dy, N_dx)

        norm_grids, valid = {}, True
        for cond in ['clean', 'bright30', 'bright50']:
            ng, ok = normalize_grid(grids[cond])
            norm_grids[cond] = ng
            if not ok: valid = False

        if not valid:
            bqs_scores.append(0.0)
            continue

        clean       = norm_grids['clean']
        lqbs        = compute_lqbs(clean, dx, dy)
        width       = compute_basin_width(clean, CONFIG['width_threshold'])
        sharp_clean = compute_sharpness(clean, dx, dy)

        ret_avg = 0.0
        for w_ret, cond in [(0.4, 'bright30'), (0.6, 'bright50')]:
            b_grid    = norm_grids[cond]
            shape_sim = compute_shape_sim(clean, b_grid)
            sharp_b   = compute_sharpness(b_grid, dx, dy)
            sharp_ret = clamp01(sharp_b / sharp_clean) if sharp_clean > 1e-12 else 0.0
            minpos    = compute_minpos_consistency(b_grid, CONFIG['retention_local_radius'])
            sym       = compute_symmetry_score(b_grid, CONFIG['local_radius'])
            ret_avg  += w_ret * (shape_sim * sharp_ret * minpos * sym)

        lqbs_norm  = lqbs  / MAX_LQBS  if MAX_LQBS  > 0 else lqbs
        width_norm = width / MAX_WIDTH if MAX_WIDTH > 0 else width
        bq = (CONFIG['basin_weights']['local_quadratic_bowl'] * lqbs_norm +
              CONFIG['basin_weights']['basin_width']          * width_norm)
        bqs_scores.append(bq * ret_avg)

    return float(np.mean(bqs_scores))

# ============================================================
# Beam Search Greedy
# ============================================================
def beam_search_greedy(candidate_pool, frame_indices, all_images, extractor,
                       dx, dy, shift_grids, N_dy, N_dx,
                       beam_size=3, min_improvement=1e-4, max_channels=60):
    """
    Forward greedy selection with beam search.

    Maintains `beam_size` candidate subsets at each step.
    Stops when the best marginal BQS gain across all beams < min_improvement,
    or when max_channels is reached.

    Returns:
        best_subset  : list of channel indices
        best_bqs     : final BQS of best_subset
        all_paths    : list of (subset, bqs_history) for every beam path
        step_logs    : list of dicts for CSV logging
    """
    device = CONFIG['device']

    # ── Step 1: evaluate all single channels, seed beams ──
    print(f"\n--- Step 1 (seeding {beam_size} beams) ---", flush=True)
    single_scores = {}
    for ch in candidate_pool:
        t0  = time.time()
        bqs = evaluate_subset_bqs([ch], frame_indices, all_images, extractor,
                                   dx, dy, shift_grids, N_dy, N_dx)
        if device == 'cuda': torch.cuda.empty_cache()
        elapsed = time.time() - t0
        single_scores[ch] = bqs
        print(f"  ch={ch:2d}  BQS={bqs:.4f}  ({elapsed:.1f}s)", flush=True)

    # Sort and pick top beam_size seeds
    sorted_singles = sorted(single_scores.items(), key=lambda x: x[1], reverse=True)
    seeds = sorted_singles[:beam_size]
    print(f"\nTop-{beam_size} seeds: {[(ch, f'{s:.4f}') for ch, s in seeds]}", flush=True)

    # Each beam: (subset, bqs_history, current_bqs)
    beams = [([ch], [bqs], bqs) for ch, bqs in seeds]

    step_logs = []
    # Log step 1
    for ch, bqs in single_scores.items():
        step_logs.append({'step': 1, 'beam': 'seed', 'channel': ch,
                          'subset': str([ch]), 'bqs': bqs, 'marginal': bqs})

    # ── Steps 2+: expand each beam ──
    step = 2
    while step <= max_channels:
        print(f"\n--- Step {step} ---", flush=True)
        new_beams = []
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
                bqs = evaluate_subset_bqs(test_subset, frame_indices, all_images, extractor,
                                           dx, dy, shift_grids, N_dy, N_dx)
                if device == 'cuda': torch.cuda.empty_cache()
                elapsed = time.time() - t0

                marginal = bqs - prev_bqs
                print(f"  [{beam_tag}] ch={ch:2d}  subset={test_subset}"
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
                          f"  subset={subset + [best_ch_this_beam]}"
                          f"  BQS={best_bqs_this_beam:.4f}"
                          f"  Δ={marginal_gain:+.4f}", flush=True)
                else:
                    # Improvement below threshold — freeze this beam
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

    # ── Pick best beam ──
    best_beam = max(beams, key=lambda b: b[2])
    best_subset, best_bqs_hist, best_bqs = best_beam

    all_paths = [(b[0], b[1]) for b in beams]
    return best_subset, best_bqs, all_paths, step_logs

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam_size',       type=int,   default=3,
                        help='Number of beams (default: 3)')
    parser.add_argument('--min_improvement', type=float, default=1e-4,
                        help='Minimum BQS gain to continue adding a channel (default: 1e-4)')
    parser.add_argument('--max_channels',    type=int,   default=60,
                        help='Hard upper limit on subset size (default: 60, i.e. no limit)')
    parser.add_argument('--frame_indices',   type=int,   nargs='+', default=None,
                        help='Frame indices to evaluate (default: [306])')
    args = parser.parse_args()

    if args.frame_indices is not None:
        CONFIG['target_frames'] = args.frame_indices

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("=== Forward Greedy Selection (BQS — Beam Search + GPU) ===")
    print(f"Device          : {CONFIG['device']}")
    print(f"Target frames   : {CONFIG['target_frames']}")
    print(f"Beam size       : {args.beam_size}")
    print(f"Min improvement : {args.min_improvement}")
    print(f"Max channels    : {args.max_channels}")

    all_images = sorted(glob.glob(os.path.join(CONFIG['rgb_dir'], "*.png")))
    if not all_images:
        raise FileNotFoundError(f"No images found in {CONFIG['rgb_dir']}")

    extractor = Conv1BNReLUExtractor(device=CONFIG['device'])

    grid_range = CONFIG['grid_range']
    dx = np.arange(-grid_range, grid_range + 1, CONFIG['grid_step'])
    dy = np.arange(-grid_range, grid_range + 1, CONFIG['grid_step'])
    N_dx, N_dy = len(dx), len(dy)

    print("\nBuilding shift grids on GPU ...", flush=True)
    sample_img = load_image_np(all_images[CONFIG['target_frames'][0]])
    H, W = sample_img.shape[:2]
    shift_grids = build_shift_grid(H, W, dx, dy, CONFIG['device'])
    print(f"Shift grids built: {shift_grids.shape}  (H={H}, W={W})", flush=True)

    candidate_pool = [ch for ch in range(64) if ch not in DEAD_CHANNELS]
    print(f"Candidate pool  : {len(candidate_pool)} channels  "
          f"(dead excluded: {DEAD_CHANNELS})", flush=True)

    total_start = time.time()

    best_subset, best_bqs, all_paths, step_logs = beam_search_greedy(
        candidate_pool, CONFIG['target_frames'], all_images, extractor,
        dx, dy, shift_grids, N_dy, N_dx,
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

    # ── Save outputs ──
    out = CONFIG['output_dir']

    # 1. Summary JSON
    summary = {
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
    with open(os.path.join(out, 'greedy_selection_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

    # 2. Step-by-step log CSV
    import csv
    csv_path = os.path.join(out, 'step_by_step_log.csv')
    if step_logs:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step','beam','channel','subset','bqs','marginal'])
            writer.writeheader()
            writer.writerows(step_logs)

    print(f"\nOutputs saved to: {out}/")
    print(f"  greedy_selection_summary.json")
    print(f"  step_by_step_log.csv")


if __name__ == '__main__':
    main()