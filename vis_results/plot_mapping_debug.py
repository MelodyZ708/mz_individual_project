# 保存为 vis_results/plot_mapping_debug.py，然后 python3 vis_results/plot_mapping_debug.py

import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("vis_results/mapping_photo_err.txt", delimiter=",")
iters      = data[:, 0]
photo_err  = data[:, 1]
aff_a_max  = data[:, 2]
aff_b_max  = data[:, 3]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(iters, photo_err, color='steelblue', linewidth=0.8)
ax1.set_ylabel("Mapping Photo Error (MSE)")
ax1.set_title("Mapping Photo Error over Iterations")
ax1.grid(True, alpha=0.3)

ax2.plot(iters, aff_a_max, label='max|a| (scale)', color='orange', linewidth=0.8)
ax2.plot(iters, aff_b_max, label='max|b| (bias)',  color='red',    linewidth=0.8)
ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='|a| threshold')
ax2.axhline(y=0.3, color='red',    linestyle='--', alpha=0.5, label='|b| threshold')
ax2.set_ylabel("Affine Param Magnitude")
ax2.set_xlabel("Mapping Iteration")
ax2.set_title("Affine Parameters in Mapping (should stay near 0 in CNN mode)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("vis_results/mapping_debug_plot.png", dpi=150)
print("Saved to vis_results/mapping_debug_plot.png")
