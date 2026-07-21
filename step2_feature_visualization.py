"""
Step 2: Feature Map Visualization for Selected Candidates (v2)

Changes from v1:
  - "Lightswitch" is now simulated by applying a global brightness reduction (×0.4)
    to the SAME frame as Clean, so Abs Diff reflects only illumination change,
    not camera motion.
  - Near-zero / suppressed channels (std < SUPPRESSED_THRESHOLD) are flagged
    instead of reporting a meaningless NCC value.

Usage:
    python step2_feature_visualization_v2.py

Dependencies:
    pip install torch torchvision numpy matplotlib opencv-python Pillow
"""

import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

# ==============================================================================
# 1. Configuration — only CLEAN_IMG_PATH needs to be set
# ==============================================================================
CANDIDATES_JSON = "step1_candidates.json"
OUT_DIR = "feature_maps_output_v2"

# Path to ONE reference frame from fr1/desk_lightswitch (bright frame)
CLEAN_IMG_PATH = "/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch/rgb/1305031458.391626.png"

# Brightness scale factor to simulate illumination drop (0.4 = 60% reduction)
BRIGHTNESS_SCALE = 0.4

# Channels with feature map std below this threshold are considered suppressed
SUPPRESSED_THRESHOLD = 0.5

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
    """Extract conv1/layer1/layer2 features from a PIL image."""
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(tensor)
    return {k: v[0].cpu().numpy() for k, v in features.items()}

# ==============================================================================
# 3. Helpers
# ==============================================================================
def compute_ncc(map1, map2):
    """Normalized Cross-Correlation between two 2D maps."""
    m1 = map1 - np.mean(map1)
    m2 = map2 - np.mean(map2)
    denom = np.sqrt(np.sum(m1**2) * np.sum(m2**2))
    if denom < 1e-8:
        return None  # suppressed channel
    return float(np.sum(m1 * m2) / denom)

def is_suppressed(feat_map, threshold=SUPPRESSED_THRESHOLD):
    return float(np.std(feat_map)) < threshold

def visualize_channels(layer_name, combo_name, channels,
                       feat_clean, feat_light, out_dir):
    """
    For each channel in the combination, plot:
      Col 0: Clean feature map
      Col 1: Simulated lightswitch feature map (NCC or 'suppressed')
      Col 2: Absolute difference
    """
    n_ch = len(channels)
    fig, axes = plt.subplots(n_ch, 3, figsize=(13, 3.5 * n_ch))
    if n_ch == 1:
        axes = [axes]

    for i, ch_str in enumerate(channels):
        ch_idx = int(ch_str.replace("d", ""))

        map_c = feat_clean[layer_name][ch_idx]
        map_l = feat_light[layer_name][ch_idx]
        diff  = np.abs(map_c - map_l)

        suppressed = is_suppressed(map_c)
        ncc = compute_ncc(map_c, map_l)

        # Use clean map's range for both clean and lightswitch columns
        vmin, vmax = map_c.min(), map_c.max()

        # ── Column 0: Clean ──────────────────────────────────────────────────
        ax_c = axes[i][0]
        im_c = ax_c.imshow(map_c, cmap="viridis", vmin=vmin, vmax=vmax)
        ax_c.set_title(f"{ch_str} — Clean", fontsize=10)
        ax_c.axis("off")
        fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)

        # ── Column 1: Lightswitch ────────────────────────────────────────────
        ax_l = axes[i][1]
        im_l = ax_l.imshow(map_l, cmap="viridis", vmin=vmin, vmax=vmax)
        if suppressed:
            ncc_label = "suppressed (std<0.5)"
        else:
            ncc_label = f"NCC: {ncc:.3f}"
        ax_l.set_title(f"{ch_str} — Lightswitch ({ncc_label})", fontsize=10)
        ax_l.axis("off")
        fig.colorbar(im_l, ax=ax_l, fraction=0.046, pad=0.04)

        # ── Column 2: Abs Diff ───────────────────────────────────────────────
        ax_d = axes[i][2]
        im_d = ax_d.imshow(diff, cmap="magma")
        mean_diff = float(np.mean(diff))
        ax_d.set_title(f"{ch_str} — Abs Diff (mean={mean_diff:.3f})", fontsize=10)
        ax_d.axis("off")
        fig.colorbar(im_d, ax=ax_d, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Layer: {layer_name}  |  Combo: {combo_name}\n"
        f"(Lightswitch simulated by brightness ×{BRIGHTNESS_SCALE})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{layer_name}_{combo_name}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

# ==============================================================================
# 4. Main
# ==============================================================================
def main():
    if not os.path.exists(CLEAN_IMG_PATH):
        print(f"ERROR: Image not found:\n  {CLEAN_IMG_PATH}")
        return

    # Load clean image and create simulated lightswitch version
    print(f"Loading image: {CLEAN_IMG_PATH}")
    clean_pil = Image.open(CLEAN_IMG_PATH).convert("RGB")

    # Simulate illumination drop: scale pixel values, keep as uint8
    clean_arr = np.array(clean_pil)
    light_arr = np.clip(clean_arr * BRIGHTNESS_SCALE, 0, 255).astype(np.uint8)
    light_pil = Image.fromarray(light_arr)

    print("Extracting features for clean and simulated lightswitch images...")
    feat_clean = extract_features(clean_pil)
    feat_light = extract_features(light_pil)

    print(f"\nLoading candidates from {CANDIDATES_JSON}...")
    with open(CANDIDATES_JSON, "r") as f:
        candidates = json.load(f)

    for layer, data in candidates.items():
        print(f"\n{'='*50}")
        print(f"  Layer: {layer}")
        print(f"{'='*50}")

        # Top 3
        for i, combo in enumerate(data["top3_combinations"]):
            print(f"\n  Top{i+1} (ATE={combo['ate_cm']:.1f} cm):")
            visualize_channels(
                layer_name=layer,
                combo_name=f"Top{i+1}_ATE{combo['ate_cm']:.1f}",
                channels=combo["channels"],
                feat_clean=feat_clean,
                feat_light=feat_light,
                out_dir=OUT_DIR,
            )

        # Frequency-derived
        print(f"\n  FreqDerived:")
        visualize_channels(
            layer_name=layer,
            combo_name="FreqDerived",
            channels=data["frequency_derived"]["channels"],
            feat_clean=feat_clean,
            feat_light=feat_light,
            out_dir=OUT_DIR,
        )

        # Bottom (first one only)
        if data["bottom3_combinations"]:
            bot = data["bottom3_combinations"][0]
            print(f"\n  Bottom (ATE={bot['ate_cm']:.1f} cm):")
            visualize_channels(
                layer_name=layer,
                combo_name=f"Bottom_ATE{bot['ate_cm']:.1f}",
                channels=bot["channels"],
                feat_clean=feat_clean,
                feat_light=feat_light,
                out_dir=OUT_DIR,
            )

    print(f"\nDone. All figures saved to: {OUT_DIR}/")

if __name__ == "__main__":
    main()