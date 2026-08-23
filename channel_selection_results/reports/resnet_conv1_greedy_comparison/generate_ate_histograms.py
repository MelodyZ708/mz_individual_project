#!/usr/bin/env python3
"""Render auditable ATE histograms for the completed ResNet/UNet greedy runs."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

_CACHE = Path(tempfile.gettempdir()) / "resnet_unet_histogram_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parents[2]
SOURCES = {
    "ResNet Conv1 greedy": (
        PROJECT_ROOT
        / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/evaluations.sqlite3",
        "#3b6fb6",
    ),
    "UNet Enc1 greedy": (
        PROJECT_ROOT
        / "channel_selection_results/step_j_unet_direct_fullseq_greedy/evaluations.sqlite3",
        "#dc8f31",
    ),
    "UNet Enc0 greedy": (
        PROJECT_ROOT
        / "channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy/evaluations.sqlite3",
        "#4f9b72",
    ),
}


def values(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as database:
        rows = database.execute(
            """
            SELECT historical_evo_ape_mean_m
            FROM evaluations
            WHERE replicate=0
              AND status='PASS'
              AND historical_evo_ape_mean_m IS NOT NULL
            """
        ).fetchall()
    return np.asarray([row[0] * 100.0 for row in rows], dtype=float)


def main() -> None:
    data = {label: (values(path), color) for label, (path, color) in SOURCES.items()}
    # Shared 1-cm bins show the operational range. The final overflow bin keeps
    # high-ATE outliers visible without stretching the low-error region.
    bins = np.r_[np.arange(0, 56, 1), np.inf]

    figure, axis = plt.subplots(figsize=(10.2, 5.8))
    for label, (ate, color) in data.items():
        axis.hist(
            ate,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.4,
            color=color,
            label=f"{label} (n={len(ate)}, median={np.median(ate):.2f} cm)",
        )
    axis.set_xlim(4, 55)
    axis.set_xlabel("Historical keyframe ATE mean (cm; lower is better)")
    axis.set_ylabel("Probability density (PASS candidates only)")
    axis.set_title("ATE distributions of full-sequence greedy-search candidates")
    axis.grid(alpha=0.25)
    axis.legend(frameon=True, fontsize=10)
    figure.tight_layout()
    figure.savefig(
        REPORT_DIR / "ATE_Histogram_Combined_Normalized_ResNet_UNet.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharex=True, sharey=False)
    finite_bins = np.arange(4, 56, 1)
    for axis, (label, (ate, color)) in zip(axes, data.items()):
        in_range = ate[ate <= 55]
        overflow = int(np.sum(ate > 55))
        axis.hist(in_range, bins=finite_bins, color=color, alpha=0.82, edgecolor="white", linewidth=0.25)
        axis.axvline(np.median(ate), color="#1f1f1f", linestyle="--", linewidth=1.5, label=f"median {np.median(ate):.2f}")
        axis.axvline(np.min(ate), color="#a71919", linestyle=":", linewidth=1.8, label=f"best {np.min(ate):.2f}")
        axis.set_title(label)
        axis.set_xlabel("ATE mean (cm)")
        axis.set_xlim(4, 55)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=9)
        axis.text(
            0.98,
            0.96,
            f"PASS n={len(ate)}\n>55 cm: {overflow}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#bdbdbd", "alpha": 0.9},
        )
    axes[0].set_ylabel("Number of PASS candidates")
    figure.suptitle("Per-architecture ATE histograms (unique replicate-0 candidates)", y=1.02, fontsize=15)
    figure.tight_layout()
    figure.savefig(
        REPORT_DIR / "ATE_Histogram_Faceted_Counts_ResNet_UNet.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("[DONE] PASS-only, replicate-0 histograms written to:")
    for path in sorted(REPORT_DIR.glob("ATE_Histogram_*_ResNet_UNet.png")):
        print(path)


if __name__ == "__main__":
    main()
