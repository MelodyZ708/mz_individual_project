"""
Visualize Greedy Selection Steps — Per-Step Basin + BQS Component Breakdown
============================================================================
For each step in Beam 1 (seed=ch6) and Beam 2 (seed=ch19), generates one figure:
  - Top row   : 3D convergence basin under Clean / +30% / +50%
  - Bottom row: BQS component trend lines from Step 1 to current step

Usage:
    python visualize_greedy_steps.py

Outputs saved to:
    vis_results/forward_greedy_bqs/step_vis/beam1/  (Beam 1, 7 figures)
    vis_results/forward_greedy_bqs/step_vis/beam2/  (Beam 2, 6 figures)
"""

import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import deque

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
RGB_DIR    = '/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/'
OUTPUT_DIR = 'vis_results/forward_greedy_bqs/step_vis'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

FRAME_IDX   = 306
GRID_RANGE  = 30
GRID_STEP   = 1
CHUNK_SIZE  = 256

LOCAL_RADIUS      = 10
WIDTH_THRESHOLD   = 0.20
CONVEXITY_SCALE   = 0.002
DEAD_THRESHOLD    = 1e-8
MAX_LQBS          = 0.55
MAX_WIDTH         = 10.0

CONDITIONS = [('Clean', 0.0), ('+30%', 0.3), ('+50%', 0.5)]

# Beam paths from greedy_selection_summary.json
BEAMS = [
    {
        'name': 'beam1',
        'label': 'Beam 1 (seed=ch6)',
        'steps': [
            [6],
            [6, 28],
            [6, 28, 34],
            [6, 28, 34, 62],
            [6, 28, 34, 62, 12],
            [6, 28, 34, 62, 12, 54],
            [6, 28, 34, 62, 12, 54, 3],
        ],
        'bqs_history': [
            0.6797, 0.7157, 0.7262, 0.7382, 0.7397, 0.7401, 0.7402
        ],
    },
    {
        'name': 'beam2',
        'label': 'Beam 2 (seed=ch19)',
        'steps': [
            [19],
            [19, 6],
            [19, 6, 28],
            [19, 6, 28, 62],
            [19, 6, 28, 62, 52],
            [19, 6, 28, 62, 52, 54],
        ],
        'bqs_history': [
            0.5650, 0.6563, 0.6908, 0.6963, 0.6986, 0.6990
        ],
    },
]

# BQS component names and display colours
COMPONENT_NAMES = ['LQBS', 'Width', 'ShapeSim', 'SharpRet', 'MinPos', 'Symmetry', 'BQS']
COMPONENT_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#000000']

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
        return x

# ============================================================
# Image Utils
# ============================================================
def load_image_np(path):
    return np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0

def numpy_to_tensor(img_np, device):
    return torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(device)

def apply_brightness(img_np, factor):
    return np.clip(img_np + factor, 0.0, 1.0) if factor != 0.0 else img_np.copy()

# ============================================================
# GPU Shift Grid + Cost Landscape
# ============================================================
def build_shift_grid(H, W, dx_vals, dy_vals, device):
    step_x = 2.0 / (W - 1)
    step_y = 2.0 / (H - 1)
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    base = torch.stack([gx, gy], dim=-1)
    dy_n = torch.tensor(dy_vals, dtype=torch.float32, device=device) * step_y
    dx_n = torch.tensor(dx_vals, dtype=torch.float32, device=device) * step_x
    N = len(dy_vals) * len(dx_vals)
    grids = base.unsqueeze(0).expand(N, -1, -1, -1).clone()
    idx = 0
    for i in range(len(dy_vals)):
        for j in range(len(dx_vals)):
            grids[idx, :, :, 0] += dx_n[j]
            grids[idx, :, :, 1] += dy_n[i]
            idx += 1
    return grids

@torch.no_grad()
def compute_cost_landscape_gpu(feat_ref, feat_tgt, channels, shift_grids,
                                N_dy, N_dx, chunk_size=CHUNK_SIZE):
    N = N_dy * N_dx
    ref_sub = feat_ref[channels]
    tgt_sub = feat_tgt[channels]
    cost_flat = torch.zeros(N, dtype=torch.float32, device=feat_ref.device)
    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        c = e - s
        gc = shift_grids[s:e]
        tc = tgt_sub.unsqueeze(0).expand(c, -1, -1, -1)
        ts = F.grid_sample(tc, gc, mode='bilinear', padding_mode='border', align_corners=True)
        rc = ref_sub.unsqueeze(0).expand(c, -1, -1, -1)
        cost_flat[s:e] = (ts - rc).pow(2).mean(dim=(1, 2, 3))
    return cost_flat.cpu().numpy().reshape(N_dy, N_dx)

# ============================================================
# BQS Component Computation
# ============================================================
def clamp01(x): return float(max(0.0, min(1.0, x)))

def normalize_grid(grid):
    gmin, gmax = float(np.min(grid)), float(np.max(grid))
    if gmax - gmin < DEAD_THRESHOLD:
        return np.zeros_like(grid, dtype=np.float64), False
    return (grid.astype(np.float64) - gmin) / (gmax - gmin), True

def safe_corr(a, b):
    a = a.ravel().astype(np.float64); a -= a.mean()
    b = b.ravel().astype(np.float64); b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return clamp01(np.dot(a, b) / d) if d > 1e-12 else 0.0

def get_local_patch(grid, radius):
    h, w = grid.shape
    cy, cx = h // 2, w // 2
    si = slice(max(0, cy-radius), min(h, cy+radius+1))
    sj = slice(max(0, cx-radius), min(w, cx+radius+1))
    return grid[si, sj]

def compute_lqbs(grid_norm, dx, dy):
    r = LOCAL_RADIUS
    h, w = grid_norm.shape
    cy, cx = h//2, w//2
    si = slice(max(0,cy-r), min(h,cy+r+1))
    sj = slice(max(0,cx-r), min(w,cx+r+1))
    local = grid_norm[si, sj]
    ldx = dx[sj]
    ldy = dy[si]
    DX, DY = np.meshgrid(ldx, ldy)
    X = np.column_stack([DX.ravel(), DY.ravel()])
    poly = PolynomialFeatures(degree=2, include_bias=True)
    Xp = poly.fit_transform(X)
    y = local.ravel()
    if np.std(y) < 1e-12:
        return 0.0
    reg = LinearRegression().fit(Xp, y)
    yp = reg.predict(Xp)
    ss_res = np.sum((y - yp)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = max(1.0 - ss_res/ss_tot if ss_tot > 1e-12 else 0.0, 0.0)
    coef = reg.coef_
    H_mat = np.array([[2*coef[3], coef[4]], [coef[4], 2*coef[5]]])
    min_eig = float(np.min(np.linalg.eigvalsh(H_mat)))
    conv = min_eig / (min_eig + CONVEXITY_SCALE) if min_eig > 0 else 0.0
    # symmetry
    patch = get_local_patch(grid_norm, r)
    dyn = float(np.max(patch) - np.min(patch))
    sym = clamp01(1.0 - float(np.mean(np.abs(patch - np.rot90(patch,2))))/dyn) if dyn > 1e-12 else 0.0
    return r2 * conv * sym

def compute_basin_width(grid_norm):
    h, w = grid_norm.shape
    cy, cx = h//2, w//2
    if np.std(grid_norm) < 1e-12: return 0.0
    mask = grid_norm <= WIDTH_THRESHOLD
    if not mask[cy, cx]: return 0.0
    visited = np.zeros_like(mask, dtype=bool)
    q = deque([(cy, cx)]); visited[cy, cx] = True
    component = []
    for dy0, dx0 in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        pass
    while q:
        y, x = q.popleft(); component.append((y, x))
        for dy0, dx0 in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
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

def compute_symmetry(grid_norm):
    patch = get_local_patch(grid_norm, LOCAL_RADIUS)
    dyn = float(np.max(patch) - np.min(patch))
    if dyn < 1e-12: return 0.0
    return clamp01(1.0 - float(np.mean(np.abs(patch - np.rot90(patch,2))))/dyn)

def compute_minpos(grid_norm):
    r = LOCAL_RADIUS
    h, w = grid_norm.shape
    cy, cx = h//2, w//2
    patch = grid_norm[max(0,cy-r):min(h,cy+r+1), max(0,cx-r):min(w,cx+r+1)]
    if patch.size == 0: return 0.0
    py, px = np.unravel_index(np.argmin(patch), patch.shape)
    lcy, lcx = patch.shape[0]//2, patch.shape[1]//2
    md = np.sqrt(lcy**2+lcx**2)
    return clamp01(1.0 - np.sqrt((py-lcy)**2+(px-lcx)**2)/md) if md > 1e-12 else 1.0

def compute_all_components(norm_grids, dx, dy):
    """
    Returns dict with keys: LQBS, Width, ShapeSim, SharpRet, MinPos, Symmetry, BQS
    All values are in [0,1].
    """
    clean = norm_grids['Clean']

    lqbs_raw  = compute_lqbs(clean, dx, dy)
    width_raw = compute_basin_width(clean)
    sharp_c   = compute_sharpness(clean, dx, dy)

    lqbs_n  = clamp01(lqbs_raw  / MAX_LQBS)
    width_n = clamp01(width_raw / MAX_WIDTH)

    ret_parts = {}
    for cond in ['+30%', '+50%']:
        b = norm_grids[cond]
        shape_sim = compute_shape_sim_full(clean, b)
        sharp_b   = compute_sharpness(b, dx, dy)
        sharp_ret = clamp01(sharp_b / sharp_c) if sharp_c > 1e-12 else 0.0
        minpos    = compute_minpos(b)
        sym       = compute_symmetry(b)
        ret_parts[cond] = {
            'ShapeSim': shape_sim,
            'SharpRet': sharp_ret,
            'MinPos':   minpos,
            'Symmetry': sym,
        }

    # Weighted average across brightness conditions (0.4 for +30%, 0.6 for +50%)
    shape_sim_avg = 0.4*ret_parts['+30%']['ShapeSim'] + 0.6*ret_parts['+50%']['ShapeSim']
    sharp_ret_avg = 0.4*ret_parts['+30%']['SharpRet'] + 0.6*ret_parts['+50%']['SharpRet']
    minpos_avg    = 0.4*ret_parts['+30%']['MinPos']   + 0.6*ret_parts['+50%']['MinPos']
    sym_avg       = 0.4*ret_parts['+30%']['Symmetry'] + 0.6*ret_parts['+50%']['Symmetry']

    retention = shape_sim_avg * sharp_ret_avg * minpos_avg * sym_avg
    bq = 0.75 * lqbs_n + 0.25 * width_n
    bqs = bq * retention

    return {
        'LQBS':      round(lqbs_n, 4),
        'Width':     round(width_n, 4),
        'ShapeSim':  round(shape_sim_avg, 4),
        'SharpRet':  round(sharp_ret_avg, 4),
        'MinPos':    round(minpos_avg, 4),
        'Symmetry':  round(sym_avg, 4),
        'BQS':       round(bqs, 4),
    }

def compute_shape_sim_full(clean_norm, bright_norm):
    if np.std(clean_norm)<1e-12 or np.std(bright_norm)<1e-12: return 0.0
    g_corr = safe_corr(clean_norm, bright_norm)
    r = LOCAL_RADIUS
    c_local = get_local_patch(clean_norm, r)
    b_local = get_local_patch(bright_norm, r)
    return 0.4*g_corr + 0.6*safe_corr(c_local, b_local)

# ============================================================
# 3D Plot
# ============================================================
def plot_basin_3d(ax, DX, DY, cost_norm, cond_label, sharp_val):
    ax.plot_surface(DX, DY, cost_norm, cmap='RdYlBu_r',
                    linewidth=0.3, edgecolor='white', alpha=0.95,
                    rstride=1, cstride=1)
    ax.contourf(DX, DY, cost_norm, zdir='z', offset=-0.05,
                cmap='gray', alpha=0.4, levels=20)
    ax.set_zlim(-0.05, 1.05)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('Δx [px]', fontsize=7, labelpad=3)
    ax.set_ylabel('Δy [px]', fontsize=7, labelpad=3)
    ax.set_zlabel('Norm. Cost', fontsize=7, labelpad=3)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=28, azim=-55)
    ax.set_title(cond_label, fontsize=9, fontweight='bold', pad=2)
    ax.text2D(0.5, 0.97, f'Sharpness={sharp_val:.4f}',
              transform=ax.transAxes, ha='center', va='top', fontsize=7.5)

# ============================================================
# Main
# ============================================================
def main():
    all_images = sorted(glob.glob(os.path.join(RGB_DIR, '*.png')))
    if not all_images:
        raise FileNotFoundError(f'No images in {RGB_DIR}')

    extractor = Conv1BNReLUExtractor(device=DEVICE)

    dx = np.arange(-GRID_RANGE, GRID_RANGE + 1, GRID_STEP)
    dy = np.arange(-GRID_RANGE, GRID_RANGE + 1, GRID_STEP)
    N_dx, N_dy = len(dx), len(dy)
    DX_mesh, DY_mesh = np.meshgrid(dx, dy)

    print('Building shift grids ...', flush=True)
    sample_np = load_image_np(all_images[FRAME_IDX])
    H, W = sample_np.shape[:2]
    shift_grids = build_shift_grid(H, W, dx, dy, DEVICE)
    print(f'Shift grids: {shift_grids.shape}', flush=True)

    rgb_np     = load_image_np(all_images[FRAME_IDX])
    rgb_tensor = numpy_to_tensor(rgb_np, DEVICE)
    with torch.no_grad():
        feat_ref = extractor(rgb_tensor)[0]

    # Master list for the combined CSV (all beams)
    all_rows = []

    for beam in BEAMS:
        beam_name  = beam['name']
        beam_label = beam['label']
        beam_steps = beam['steps']
        bqs_hist   = beam['bqs_history']

        out_dir = os.path.join(OUTPUT_DIR, beam_name)
        os.makedirs(out_dir, exist_ok=True)

        # Accumulate component history across steps
        comp_history = []   # list of dicts, one per step
        beam_rows    = []   # rows for this beam's CSV

        print(f'\n=== {beam_label} ===', flush=True)

        for step_idx, channels in enumerate(beam_steps):
            step_num = step_idx + 1
            print(f'  Step {step_num}: channels={channels}', flush=True)

            # ── Compute cost landscapes ──
            norm_grids = {}
            sharp_vals = {}
            for cond_name, factor in CONDITIONS:
                tgt_np     = apply_brightness(rgb_np, factor)
                tgt_tensor = numpy_to_tensor(tgt_np, DEVICE)
                with torch.no_grad():
                    feat_tgt = extractor(tgt_tensor)[0]
                raw = compute_cost_landscape_gpu(feat_ref, feat_tgt, channels,
                                                 shift_grids, N_dy, N_dx)
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                ng, ok = normalize_grid(raw)
                norm_grids[cond_name] = ng if ok else np.zeros_like(raw)
                sharp_vals[cond_name] = compute_sharpness(ng, dx, dy) if ok else 0.0

            # ── Compute BQS components ──
            comps = compute_all_components(norm_grids, dx, dy)
            comp_history.append(comps)
            print(f'    BQS={comps["BQS"]:.4f}  LQBS={comps["LQBS"]:.4f}  '
                  f'Width={comps["Width"]:.4f}  ShapeSim={comps["ShapeSim"]:.4f}  '
                  f'SharpRet={comps["SharpRet"]:.4f}  MinPos={comps["MinPos"]:.4f}  '
                  f'Sym={comps["Symmetry"]:.4f}', flush=True)

            # ── Accumulate CSV row ──
            added_ch_csv = channels[-1] if len(channels) > 0 else channels[0]
            row = {
                'beam':          beam_name,
                'beam_label':    beam_label,
                'step':          step_num,
                'added_channel': added_ch_csv,
                'subset':        str(channels),
                'n_channels':    len(channels),
                'BQS':           comps['BQS'],
                'LQBS':          comps['LQBS'],
                'Width':         comps['Width'],
                'ShapeSim':      comps['ShapeSim'],
                'SharpRet':      comps['SharpRet'],
                'MinPos':        comps['MinPos'],
                'Symmetry':      comps['Symmetry'],
                'BQ':            round(0.75*comps['LQBS'] + 0.25*comps['Width'], 4),
                'Retention':     round(comps['ShapeSim']*comps['SharpRet']*comps['MinPos']*comps['Symmetry'], 4),
                'marginal_BQS':  round(comps['BQS'] - (comp_history[-2]['BQS'] if len(comp_history) >= 2 else 0.0), 4),
            }
            beam_rows.append(row)
            all_rows.append(row)

            # ── Build figure ──
            # Layout: 2 rows
            #   Row 1: 3 x 3D basin plots
            #   Row 2: 1 x component trend line chart (full width)
            fig = plt.figure(figsize=(15, 10))
            fig.patch.set_facecolor('white')

            gs = GridSpec(2, 3, figure=fig,
                          height_ratios=[1.6, 1.0],
                          hspace=0.45, wspace=0.3)

            # ── Super title ──
            added_ch = channels[-1] if len(channels) > 0 else channels[0]
            suptitle = (f'{beam_label}  —  Step {step_num}  —  Frame {FRAME_IDX}\n'
                        f'Subset: {channels}   Added: ch={added_ch}   BQS={bqs_hist[step_idx]:.4f}')
            fig.suptitle(suptitle, fontsize=11, fontweight='bold', y=1.01)

            # ── Row 1: 3D basins ──
            for col, (cond_name, _) in enumerate(CONDITIONS):
                ax3d = fig.add_subplot(gs[0, col], projection='3d')
                plot_basin_3d(ax3d, DX_mesh, DY_mesh,
                              norm_grids[cond_name],
                              cond_name, sharp_vals[cond_name])

            # ── Row 2: Component trend lines ──
            ax2d = fig.add_subplot(gs[1, :])

            steps_so_far = list(range(1, step_num + 1))
            for comp_name, color in zip(COMPONENT_NAMES, COMPONENT_COLORS):
                vals = [h[comp_name] for h in comp_history]
                lw   = 2.5 if comp_name == 'BQS' else 1.5
                ls   = '-'  if comp_name == 'BQS' else '--'
                ms   = 7    if comp_name == 'BQS' else 5
                ax2d.plot(steps_so_far, vals,
                          color=color, linewidth=lw, linestyle=ls,
                          marker='o', markersize=ms, label=comp_name)
                # Annotate last point
                ax2d.annotate(f'{vals[-1]:.3f}',
                              xy=(steps_so_far[-1], vals[-1]),
                              xytext=(4, 0), textcoords='offset points',
                              fontsize=7, color=color, va='center')

            ax2d.set_xlim(0.5, len(beam_steps) + 0.8)
            ax2d.set_ylim(-0.05, 1.10)
            ax2d.set_xticks(range(1, len(beam_steps) + 1))
            ax2d.set_xticklabels(
                [f'Step {i}\n+ch{beam_steps[i-1][-1]}' for i in range(1, len(beam_steps)+1)],
                fontsize=8)
            ax2d.set_xlabel('Greedy Step', fontsize=9)
            ax2d.set_ylabel('Score (normalised)', fontsize=9)
            ax2d.set_title('BQS Component Trend', fontsize=10, fontweight='bold')
            ax2d.legend(loc='upper left', fontsize=8, ncol=4,
                        framealpha=0.8, edgecolor='gray')
            ax2d.grid(True, alpha=0.3)

            # Highlight current step with vertical line
            ax2d.axvline(x=step_num, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)

            plt.tight_layout()

            out_path = os.path.join(out_dir, f'step{step_num:02d}_ch{added_ch}.png')
            fig.savefig(out_path, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close(fig)
            print(f'    Saved: {out_path}', flush=True)

        # ── Write per-beam CSV ──
        csv_fields = ['beam', 'beam_label', 'step', 'added_channel', 'subset',
                      'n_channels', 'BQS', 'marginal_BQS', 'LQBS', 'Width',
                      'BQ', 'ShapeSim', 'SharpRet', 'MinPos', 'Symmetry', 'Retention']
        beam_csv_path = os.path.join(out_dir, f'{beam_name}_metrics_summary.csv')
        with open(beam_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(beam_rows)
        print(f'  CSV saved: {beam_csv_path}', flush=True)

    # ── Write combined CSV (all beams) ──
    combined_csv_path = os.path.join(OUTPUT_DIR, 'all_beams_metrics_summary.csv')
    csv_fields = ['beam', 'beam_label', 'step', 'added_channel', 'subset',
                  'n_channels', 'BQS', 'marginal_BQS', 'LQBS', 'Width',
                  'BQ', 'ShapeSim', 'SharpRet', 'MinPos', 'Symmetry', 'Retention']
    with open(combined_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'\nCombined CSV saved: {combined_csv_path}', flush=True)

    print('\nAll done.', flush=True)


if __name__ == '__main__':
    main()