"""
Step 3: Convergence Basin Analysis for Selected Candidates

This script calculates and visualises the convergence basin (cost landscape)
for both individual candidate channels and their combinations across
conv1, layer1, and layer2.

Illumination perturbation: ×0.4 global brightness.
Metrics: Basin Width (area < 0.3) and Minimum Drift (px).
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as mplNormalize
from torchvision import models, transforms
from PIL import Image

# ==============================================================================
# 1. Configuration
# ==============================================================================
CANDIDATES_JSON = "step1_candidates.json"
OUT_DIR_SINGLE = "basin_output_single_channels"
OUT_DIR_COMBO = "basin_output_combinations"

CLEAN_IMG_PATH = "/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch/rgb/1305031458.391626.png"
BRIGHTNESS_SCALE = 0.4

MAX_SHIFT_PX = 30
GRID_SIZE = 61

# ==============================================================================
# 2. Model & Feature Extraction Setup
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = model.to(device)
model.eval()

features = {}

def get_hook(name):
    def hook(module, input, output):
        features[name] = output.detach()
    return hook

model.conv1.register_forward_hook(get_hook("conv1"))
model.layer1.register_forward_hook(get_hook("layer1"))
model.layer2.register_forward_hook(get_hook("layer2"))

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def extract_features(pil_img):
    """Extract features and upsample them back to original image size."""
    orig_size = (pil_img.height, pil_img.width)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(tensor)
    
    upsampled = {}
    for k, v in features.items():
        v_up = F.interpolate(v, size=orig_size, mode='bilinear', align_corners=False)
        upsampled[k] = v_up[0].cpu().numpy()
    return upsampled

# ==============================================================================
# 3. Core Basin Helpers
# ==============================================================================
def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    h, w = image.shape[:2]
    M = np.float64([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)

def compute_cost_landscape(feat_ref: np.ndarray, feat_target: np.ndarray) -> np.ndarray:
    dx_vals = np.linspace(-MAX_SHIFT_PX, MAX_SHIFT_PX, GRID_SIZE)
    dy_vals = np.linspace(-MAX_SHIFT_PX, MAX_SHIFT_PX, GRID_SIZE)
    cost_grid = np.zeros((GRID_SIZE, GRID_SIZE))
    
    for i, dy in enumerate(dy_vals):
        for j, dx in enumerate(dx_vals):
            shifted = shift_image(feat_target, dx, dy)
            residual = shifted.astype(np.float64) - feat_ref.astype(np.float64)
            cost_grid[i, j] = np.mean(residual ** 2)
    return cost_grid

def normalize_grid(cost_grid):
    c_min, c_max = cost_grid.min(), cost_grid.max()
    if c_max - c_min < 1e-10:
        return np.zeros_like(cost_grid)
    return (cost_grid - c_min) / (c_max - c_min)

def compute_basin_width(cost_grid, threshold=0.3):
    norm = normalize_grid(cost_grid)
    basin_mask = norm < threshold
    return float(np.sum(basin_mask) / norm.size)

def compute_min_drift(clean_grid, perturbed_grid):
    cy, cx = np.unravel_index(np.argmin(clean_grid), clean_grid.shape)
    py, px = np.unravel_index(np.argmin(perturbed_grid), perturbed_grid.shape)
    
    # Convert grid indices to pixels
    dx_vals = np.linspace(-MAX_SHIFT_PX, MAX_SHIFT_PX, GRID_SIZE)
    dy_vals = np.linspace(-MAX_SHIFT_PX, MAX_SHIFT_PX, GRID_SIZE)
    
    clean_x, clean_y = dx_vals[cx], dy_vals[cy]
    pert_x, pert_y = dx_vals[px], dy_vals[py]
    
    drift = np.sqrt((clean_x - pert_x)**2 + (clean_y - pert_y)**2)
    return float(drift)

# ==============================================================================
# 4. Plotting
# ==============================================================================
def plot_basin(clean_grid, light_grid, title, out_path):
    fig = plt.figure(figsize=(12, 6))
    
    dx_vals = np.linspace(-MAX_SHIFT_PX, MAX_SHIFT_PX, GRID_SIZE)
    dy_vals = np.linspace(-MAX_SHIFT_PX, MAX_SHIFT_PX, GRID_SIZE)
    DX_grid, DY_grid = np.meshgrid(dx_vals, dy_vals, indexing='ij')
    
    grids = [("Clean", clean_grid), (f"Lightswitch (×{BRIGHTNESS_SCALE})", light_grid)]
    
    # Compute metrics
    w_clean = compute_basin_width(clean_grid)
    w_light = compute_basin_width(light_grid)
    drift = compute_min_drift(clean_grid, light_grid)
    
    for idx, (label, cost_data) in enumerate(grids):
        cost_norm = normalize_grid(cost_data)
        
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        cmap = plt.get_cmap('YlOrRd')
        norm_color = mplNormalize(vmin=0, vmax=1)
        
        ax.plot_surface(DY_grid, DX_grid, cost_norm,
                        facecolors=cmap(norm_color(cost_norm)),
                        linewidth=0.12, alpha=0.92, shade=True, rcount=40, ccount=40)
        
        off = -0.05
        ax.contourf(DY_grid, DX_grid, cost_norm, zdir='z', offset=off,
                    levels=20, cmap='gray_r', alpha=0.7)
        ax.contour(DY_grid, DX_grid, cost_norm, zdir='z', offset=off,
                   levels=10, colors='k', linewidths=0.4, alpha=0.5)
        
        ax.set_xlabel(r'$\Delta x$ [px]', labelpad=8)
        ax.set_ylabel(r'$\Delta y$ [px]', labelpad=8)
        ax.set_zlabel('Norm. Cost', labelpad=6)
        ax.set_zlim(off, 1.05)
        ax.view_init(elev=32, azim=-50)
        
        width_val = w_clean if idx == 0 else w_light
        ax.set_title(f"{label}\nBasin Width (<0.3) = {width_val:.3f}", fontsize=11, pad=10)
    
    fig.suptitle(f"{title}\nMin Drift = {drift:.2f} px", fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ==============================================================================
# 5. Main
# ==============================================================================
def main():
    if not os.path.exists(CLEAN_IMG_PATH):
        print(f"ERROR: Image not found:\n  {CLEAN_IMG_PATH}")
        return

    print(f"Loading image: {CLEAN_IMG_PATH}")
    clean_pil = Image.open(CLEAN_IMG_PATH).convert("RGB")
    
    clean_arr = np.array(clean_pil)
    light_arr = np.clip(clean_arr * BRIGHTNESS_SCALE, 0, 255).astype(np.uint8)
    light_pil = Image.fromarray(light_arr)

    print("Extracting features...")
    feat_clean = extract_features(clean_pil)
    feat_light = extract_features(light_pil)

    with open(CANDIDATES_JSON, "r") as f:
        candidates = json.load(f)

    # Gather unique channels across all candidates per layer
    unique_channels = {"conv1": set(), "layer1": set(), "layer2": set()}
    
    for layer, data in candidates.items():
        for combo in data["top3_combinations"]:
            unique_channels[layer].update([int(ch.replace("d", "")) for ch in combo["channels"]])
        unique_channels[layer].update([int(ch.replace("d", "")) for ch in data["frequency_derived"]["channels"]])
        if data["bottom3_combinations"]:
            unique_channels[layer].update([int(ch.replace("d", "")) for ch in data["bottom3_combinations"][0]["channels"]])

    # --------------------------------------------------------------------------
    # Part A: Per-Channel Basins
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("Part A: Per-Channel Convergence Basins")
    print("="*50)
    for layer, channels in unique_channels.items():
        print(f"\nProcessing {len(channels)} unique channels for {layer}...")
        for ch in sorted(channels):
            map_c = feat_clean[layer][ch]
            map_l = feat_light[layer][ch]
            
            # Check if suppressed
            if np.std(map_c) < 0.5:
                print(f"  {layer} d{ch}: Suppressed (std < 0.5), skipping basin plot.")
                continue
                
            cost_c = compute_cost_landscape(map_c, map_c)
            cost_l = compute_cost_landscape(map_c, map_l)
            
            title = f"Channel d{ch} — {layer}"
            out_path = os.path.join(OUT_DIR_SINGLE, layer, f"d{ch}.png")
            plot_basin(cost_c, cost_l, title, out_path)

    # --------------------------------------------------------------------------
    # Part B: Combination-Level Basins
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("Part B: Combination-Level Convergence Basins")
    print("="*50)
    for layer, data in candidates.items():
        print(f"\nProcessing combinations for {layer}...")
        
        # Collect all combos
        combos_to_process = []
        for i, combo in enumerate(data["top3_combinations"]):
            combos_to_process.append((f"Top{i+1}_ATE{combo['ate_cm']:.1f}", combo["channels"]))
        combos_to_process.append(("FreqDerived", data["frequency_derived"]["channels"]))
        if data["bottom3_combinations"]:
            bot = data["bottom3_combinations"][0]
            combos_to_process.append((f"Bottom_ATE{bot['ate_cm']:.1f}", bot["channels"]))
            
        for name, channels in combos_to_process:
            ch_indices = [int(ch.replace("d", "")) for ch in channels]
            
            # Compute average cost across all channels in the combo
            avg_cost_c = np.zeros((GRID_SIZE, GRID_SIZE))
            avg_cost_l = np.zeros((GRID_SIZE, GRID_SIZE))
            
            valid_channels = 0
            for ch in ch_indices:
                map_c = feat_clean[layer][ch]
                map_l = feat_light[layer][ch]
                
                # We include all channels in the combo, even suppressed ones, 
                # as this mimics the actual photometric cost function.
                cost_c = compute_cost_landscape(map_c, map_c)
                cost_l = compute_cost_landscape(map_c, map_l)
                
                avg_cost_c += cost_c
                avg_cost_l += cost_l
                valid_channels += 1
                
            avg_cost_c /= valid_channels
            avg_cost_l /= valid_channels
            
            title = f"Combo: {name} ({','.join(channels)}) — {layer}"
            out_path = os.path.join(OUT_DIR_COMBO, layer, f"{name}.png")
            plot_basin(avg_cost_c, avg_cost_l, title, out_path)

    print(f"\nDone. All figures saved to {OUT_DIR_SINGLE}/ and {OUT_DIR_COMBO}/")

if __name__ == "__main__":
    main()