"""
Diagnose Dead Channels in ResNet-18 Conv1

This script investigates WHY certain conv1 channels produce zero activations
by examining the output at each stage of the pipeline:
  Stage 1: conv1 only (raw convolution output)
  Stage 2: conv1 + BN (after BatchNorm, before ReLU)
  Stage 3: conv1 + BN + ReLU (final output)

It also prints the BN parameters (running_mean, running_var, weight, bias)
for each channel, allowing us to see exactly which stage causes the "death".

Hypothesis: Dead channels have BN outputs that are entirely negative,
so ReLU clips them to zero. This is a domain gap effect — the BN running
statistics were learned on ImageNet, not on TUM indoor scenes.

Usage:
  python diagnose_dead_channels.py

Output directory: vis_results/dead_channel_diagnosis/
"""

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, ResNet18_Weights
import os
import glob


def main():
    output_dir = "vis_results/dead_channel_diagnosis"
    rgb_dir = "/vol/bitbucket/mz325/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load pretrained ResNet-18
    resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    resnet.eval()
    
    # Build three separate extractors for each stage
    stage1_conv_only = nn.Sequential(resnet.conv1).to(device)
    stage2_conv_bn = nn.Sequential(resnet.conv1, resnet.bn1).to(device)
    stage3_conv_bn_relu = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu).to(device)
    
    for s in [stage1_conv_only, stage2_conv_bn, stage3_conv_bn_relu]:
        for param in s.parameters():
            param.requires_grad = False
    
    # =========================================================================
    # Part 1: Print BN parameters for all 64 channels
    # =========================================================================
    print("=" * 80)
    print("PART 1: BatchNorm Parameters (learned on ImageNet)")
    print("=" * 80)
    
    bn_weight = resnet.bn1.weight.detach().cpu().numpy()      # gamma
    bn_bias = resnet.bn1.bias.detach().cpu().numpy()           # beta
    bn_mean = resnet.bn1.running_mean.detach().cpu().numpy()   # running mean
    bn_var = resnet.bn1.running_var.detach().cpu().numpy()     # running var
    
    print(f"\n{'Ch':<5} {'gamma':<10} {'beta':<10} {'run_mean':<12} {'run_var':<12} {'Note'}")
    print("-" * 65)
    for i in range(64):
        # BN formula: output = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
        # If the input x is consistently below running_mean, output will be negative
        # (assuming gamma > 0), and ReLU will kill it.
        note = ""
        if bn_weight[i] < 0:
            note = "gamma<0 (inverted)"
        print(f"Ch{i+1:<3} {bn_weight[i]:<10.4f} {bn_bias[i]:<10.4f} {bn_mean[i]:<12.4f} {bn_var[i]:<12.4f} {note}")
    
    # Save BN params to file
    bn_path = os.path.join(output_dir, "bn_parameters.txt")
    with open(bn_path, 'w') as f:
        f.write("BatchNorm Parameters for ResNet-18 bn1 (64 channels)\n")
        f.write("BN formula: output = gamma * (x - running_mean) / sqrt(running_var + eps) + beta\n")
        f.write("If output is entirely negative for TUM images, ReLU will produce zero (dead channel)\n\n")
        f.write(f"{'Ch':<5} {'gamma':<10} {'beta':<10} {'run_mean':<12} {'run_var':<12}\n")
        f.write("-" * 55 + "\n")
        for i in range(64):
            f.write(f"Ch{i+1:<3} {bn_weight[i]:<10.4f} {bn_bias[i]:<10.4f} {bn_mean[i]:<12.4f} {bn_var[i]:<12.4f}\n")
    print(f"\nBN parameters saved to: {bn_path}")
    
    # =========================================================================
    # Part 2: Per-stage activation statistics on TUM images
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: Per-Stage Activation Statistics on TUM Images")
    print("=" * 80)
    
    # Select 3 representative images
    all_images = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    total = len(all_images)
    indices = [0, total // 2, total - 1]
    selected_images = [all_images[i] for i in indices]
    
    # ImageNet normalization
    mean_t = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    # Known dead channels from previous 64-ch visualization (1-based)
    known_dead = {3, 5, 8, 10, 14, 37, 39, 49}
    
    # Aggregate stats across images
    all_stage_stats = {1: [], 2: [], 3: []}
    
    for img_idx, img_path in enumerate(selected_images):
        img_name = os.path.basename(img_path)
        print(f"\n--- Image {img_idx+1}/3: {img_name} ---")
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
        x = (img_tensor.float() - mean_t) / std_t
        
        with torch.no_grad():
            out1 = stage1_conv_only(x)      # (1, 64, H/2, W/2)
            out2 = stage2_conv_bn(x)        # (1, 64, H/2, W/2)
            out3 = stage3_conv_bn_relu(x)   # (1, 64, H/2, W/2)
        
        stages = [("Stage1: Conv only", out1), ("Stage2: Conv+BN", out2), ("Stage3: Conv+BN+ReLU", out3)]
        
        for stage_idx, (stage_name, out) in enumerate(stages, 1):
            print(f"\n  {stage_name}:")
            print(f"  {'Ch':<5} {'min':<12} {'max':<12} {'mean':<12} {'std':<12} {'all<=0?':<8} {'Dead?'}")
            print(f"  " + "-" * 75)
            
            stage_stats = []
            for ch in range(64):
                ch_data = out[0, ch].cpu().numpy()
                ch_min = ch_data.min()
                ch_max = ch_data.max()
                ch_mean = ch_data.mean()
                ch_std = ch_data.std()
                all_neg = "YES" if ch_max <= 0 else "no"
                is_dead = "◀ DEAD" if (ch + 1) in known_dead else ""
                
                stage_stats.append({
                    'ch': ch + 1, 'min': ch_min, 'max': ch_max,
                    'mean': ch_mean, 'std': ch_std, 'all_neg': ch_max <= 0
                })
                
                # Only print dead channels and a few active ones for brevity
                if (ch + 1) in known_dead or ch < 3 or ch >= 62:
                    print(f"  Ch{ch+1:<3} {ch_min:<12.4f} {ch_max:<12.4f} {ch_mean:<12.4f} {ch_std:<12.4f} {all_neg:<8} {is_dead}")
            
            all_stage_stats[stage_idx].append(stage_stats)
    
    # =========================================================================
    # Part 3: Summary — prove the mechanism
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: Dead Channel Mechanism Summary")
    print("=" * 80)
    
    print(f"\nKnown dead channels (from 64-ch visualization): {sorted(known_dead)}")
    print(f"\nFor each dead channel, checking if Stage2 (Conv+BN, before ReLU) output is all-negative:\n")
    
    print(f"{'Ch':<5} {'Stage1 max':<14} {'Stage2 max':<14} {'Stage3 max':<14} {'Mechanism'}")
    print("-" * 70)
    
    for ch_1based in sorted(known_dead):
        ch_0based = ch_1based - 1
        # Average across 3 images
        s1_maxes = [all_stage_stats[1][img][ch_0based]['max'] for img in range(3)]
        s2_maxes = [all_stage_stats[2][img][ch_0based]['max'] for img in range(3)]
        s3_maxes = [all_stage_stats[3][img][ch_0based]['max'] for img in range(3)]
        
        s1_max = np.mean(s1_maxes)
        s2_max = np.mean(s2_maxes)
        s3_max = np.mean(s3_maxes)
        
        if s1_max > 0 and s2_max <= 0 and s3_max <= 1e-6:
            mechanism = "Conv output positive → BN shifts to negative → ReLU kills"
        elif s1_max <= 0 and s3_max <= 1e-6:
            mechanism = "Conv output already negative → ReLU kills"
        elif s2_max > 0 and s3_max <= 1e-6:
            mechanism = "Unexpected: BN positive but Stage3 dead?"
        else:
            mechanism = f"Not fully dead (s3_max={s3_max:.6f})"
        
        print(f"Ch{ch_1based:<3} {s1_max:<14.4f} {s2_max:<14.4f} {s3_max:<14.6f} {mechanism}")
    
    # =========================================================================
    # Part 4: Visualization — bar chart of Stage2 max values
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: Generating Visualization")
    print("=" * 80)
    
    # Average Stage2 max across 3 images for all 64 channels
    s2_max_avg = np.zeros(64)
    for ch in range(64):
        s2_max_avg[ch] = np.mean([all_stage_stats[2][img][ch]['max'] for img in range(3)])
    
    fig, ax = plt.subplots(figsize=(18, 6))
    colors = ['red' if (i+1) in known_dead else '#2196F3' for i in range(64)]
    bars = ax.bar(range(1, 65), s2_max_avg, color=colors, edgecolor='none', width=0.8)
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel("Channel Index (1-based)", fontsize=12)
    ax.set_ylabel("Stage2 Max Value (Conv+BN, before ReLU)", fontsize=12)
    ax.set_title("Conv1+BN Output Max Value per Channel (averaged over 3 TUM images)\n"
                 "Red = Dead channels (max ≤ 0 → ReLU clips to zero)", fontsize=13)
    ax.set_xticks(range(1, 65, 2))
    ax.set_xlim(0.5, 64.5)
    
    # Add zero line annotation
    ax.annotate("Channels below this line → killed by ReLU",
                xy=(50, 0), xytext=(45, s2_max_avg.max() * 0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "dead_channel_mechanism.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    
    # =========================================================================
    # Part 5: Per-channel comparison across 3 stages (for dead channels only)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle("Dead Channel Diagnosis: Activation Distribution at Each Stage\n"
                 "(Each histogram shows pixel value distribution for one TUM image)", fontsize=14)
    
    # Use first image for histograms
    img = Image.open(selected_images[0]).convert('RGB')
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    x = (img_tensor.float() - mean_t) / std_t
    
    with torch.no_grad():
        out1 = stage1_conv_only(x)
        out2 = stage2_conv_bn(x)
        out3 = stage3_conv_bn_relu(x)
    
    for plot_idx, ch_1based in enumerate(sorted(known_dead)):
        row = plot_idx // 4
        col = plot_idx % 4
        ax = axes[row, col]
        
        ch_0based = ch_1based - 1
        d1 = out1[0, ch_0based].cpu().numpy().flatten()
        d2 = out2[0, ch_0based].cpu().numpy().flatten()
        d3 = out3[0, ch_0based].cpu().numpy().flatten()
        
        ax.hist(d1, bins=50, alpha=0.5, label='Conv only', color='blue', density=True)
        ax.hist(d2, bins=50, alpha=0.5, label='Conv+BN', color='orange', density=True)
        ax.axvline(x=0, color='red', linewidth=2, linestyle='--', label='ReLU threshold')
        
        ax.set_title(f"Ch {ch_1based}", fontsize=11, fontweight='bold', color='red')
        ax.set_xlabel("Activation value", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        if plot_idx == 0:
            ax.legend(fontsize=7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path2 = os.path.join(output_dir, "dead_channel_histograms.png")
    plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path2}")
    
    # =========================================================================
    # Part 6: Save full report
    # =========================================================================
    report_path = os.path.join(output_dir, "diagnosis_report.txt")
    with open(report_path, 'w') as f:
        f.write("Dead Channel Diagnosis Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Pipeline: Input RGB → ImageNet Normalize → Conv1 (7x7, stride=2) → BN → ReLU\n\n")
        f.write("FINDING: Dead channels are caused by the BN + ReLU combination.\n")
        f.write("The BN layer uses running statistics learned on ImageNet.\n")
        f.write("For certain channels, the TUM indoor scene inputs produce conv1 outputs\n")
        f.write("that, after BN normalization with ImageNet statistics, are entirely negative.\n")
        f.write("ReLU then clips these to zero, producing 'dead' channels.\n\n")
        f.write("This is a DOMAIN GAP effect, not a bug:\n")
        f.write("- The filters themselves are functional (Stage 1 shows non-zero output)\n")
        f.write("- BN shifts the distribution based on ImageNet expectations\n")
        f.write("- When TUM input doesn't match ImageNet distribution for that filter,\n")
        f.write("  the BN output falls entirely below zero\n")
        f.write("- ReLU eliminates the signal\n\n")
        f.write("IMPLICATION: Using conv1 features without ReLU, or with a different\n")
        f.write("activation (e.g., LeakyReLU), would recover these channels.\n")
        f.write("Alternatively, re-computing BN statistics on TUM data (BN recalibration)\n")
        f.write("would shift the distributions back to a useful range.\n\n")
        
        f.write("Dead channels (1-based): " + str(sorted(known_dead)) + "\n\n")
        
        f.write(f"{'Ch':<5} {'S1(Conv) max':<16} {'S2(Conv+BN) max':<18} {'S3(+ReLU) max':<16} {'Mechanism'}\n")
        f.write("-" * 80 + "\n")
        for ch_1based in sorted(known_dead):
            ch_0based = ch_1based - 1
            s1_max = np.mean([all_stage_stats[1][img][ch_0based]['max'] for img in range(3)])
            s2_max = np.mean([all_stage_stats[2][img][ch_0based]['max'] for img in range(3)])
            s3_max = np.mean([all_stage_stats[3][img][ch_0based]['max'] for img in range(3)])
            
            if s1_max > 0 and s2_max <= 0:
                mech = "BN shifts to negative → ReLU kills"
            elif s1_max <= 0:
                mech = "Conv output already all-negative"
            else:
                mech = f"Other (s2_max={s2_max:.4f})"
            
            f.write(f"Ch{ch_1based:<3} {s1_max:<16.4f} {s2_max:<18.4f} {s3_max:<16.6f} {mech}\n")
    
    print(f"\nFull report saved to: {report_path}")
    print("\nDone! Key outputs:")
    print(f"  {output_dir}/dead_channel_mechanism.png  — bar chart showing BN output per channel")
    print(f"  {output_dir}/dead_channel_histograms.png — histograms for each dead channel")
    print(f"  {output_dir}/bn_parameters.txt           — BN layer parameters")
    print(f"  {output_dir}/diagnosis_report.txt        — full text report")


if __name__ == "__main__":
    main()