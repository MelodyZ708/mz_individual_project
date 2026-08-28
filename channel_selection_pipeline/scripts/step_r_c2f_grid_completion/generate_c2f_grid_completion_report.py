#!/usr/bin/env python3
"""Generate the Chinese Word report for the completed Step-R C2F grid.

The report deliberately ranks C2F candidates by their paired gain over the
better direct parent on the *same* full sequence.  It therefore separates
the best absolute ATE from the question this experiment answers: whether a
given coarse-to-fine routing adds robust value beyond its constituent direct
branches.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_c2f_grid_completion_report")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
STEP_R_ROOT = PROJECT_ROOT / "channel_selection_results/step_r_c2f_grid_completion"
STEP_Q_PLAN = PROJECT_ROOT / "channel_selection_pipeline/scripts/step_q_c2f_multi_dataset_evaluation/c2f_multi_dataset_plan.json"
STEP_P_DIR = PROJECT_ROOT / "channel_selection_pipeline/scripts/step_p_c2f_best_channels_evaluation"
RESULT_ROOT = PROJECT_ROOT / "channel_selection_results/reports/c2f_complete_grid_multi_dataset"
DOCX_NAME = "C2F_完整网格九数据集评估_中文.docx"

# Follow the ordering used in the requested presentation table.
TABLE_DATASET_KEYS = (
    "fr1_desk_clean", "fr1_desk_flashlight", "fr1_desk_lightswitch",
    "fr2_desk_clean", "fr2_desk_flashlight", "fr2_desk_lightswitch",
    "fr3_long_office_household_clean", "fr3_long_office_household_flashlight", "fr3_long_office_household_lightswitch",
)
FAMILY_KEYS = ("fr1_desk", "fr2_desk", "fr3_long_office_household")
FAMILY_SHORT = {"fr1_desk": "fr1", "fr2_desk": "fr2", "fr3_long_office_household": "fr3"}
CONDITION_SHORT = {"clean": "clean", "flashlight": "flash", "lightswitch": "light\nswitch"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | float | None) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


def signed_percent(value: float | None, digits: int = 1) -> str:
    return "" if value is None else f"{value:+.{digits}f}%"


def cm(value: float | None, digits: int = 2) -> str:
    return "FAIL" if value is None else f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def read_grid_candidates(architecture: str) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    plan = json.loads((STEP_P_DIR / f"{architecture}_c2f_candidate_plan.json").read_text(encoding="utf-8"))
    fine = {int(item["rank"]): tuple(int(value) for value in item["channels"]) for item in plan["fine_branch"]["candidates"]}
    coarse = {int(item["rank"]): tuple(int(value) for value in item["channels"]) for item in plan["coarse_branch"]["candidates"]}
    return fine, coarse


def short_label(architecture: str, label: str) -> str:
    if "_c2f_" not in label:
        return label
    tokens = label.split("_")
    variant = tokens[2].upper()
    fine = int(tokens[3].replace("fine", ""))
    coarse = int(tokens[4].replace("coarse", ""))
    return f"{'U' if architecture == 'unet' else 'R'}-{variant} F{fine}+C{coarse}"


def architecture_name(architecture: str) -> str:
    return "U-Net" if architecture == "unet" else "ResNet"


def component_text(architecture: str, fine_rank: int, coarse_rank: int, fine: dict[int, tuple[int, ...]], coarse: dict[int, tuple[int, ...]]) -> str:
    if architecture == "unet":
        fine_layer, coarse_layer = "Enc0", "Enc1"
    else:
        fine_layer, coarse_layer = "Conv1", "Layer2"
    return (
        f"{fine_layer} [" + ",".join(map(str, fine[fine_rank])) + "] + "
        f"{coarse_layer} [" + ",".join(map(str, coarse[coarse_rank])) + "]"
    )


def is_true(value: str | bool) -> bool:
    return value is True or str(value).lower() == "true"


def add_family_triple_statistics(data: dict[str, Any], effects: dict[str, dict[str, Any]]) -> None:
    pairwise = data["pairwise"]
    labels = sorted(effects)
    for label in labels:
        rows = [row for row in pairwise if row["standard_label"] == label]
        family_wins: list[str] = []
        for family in FAMILY_KEYS:
            group = [row for row in rows if row["dataset_family"] == family]
            win = len(group) == 3 and all(
                row["comparison_status"] == "COMPARABLE" and is_true(row["beats_better_parent"])
                for row in group
            )
            if win:
                family_wins.append(family)
        effects[label]["family_triple_wins"] = tuple(family_wins)
        effects[label]["family_triple_count"] = len(family_wins)


def load_architecture(architecture: str) -> dict[str, Any]:
    root = STEP_R_ROOT / architecture
    pairwise = [row for row in read_csv(root / "merged_pairwise_comparison.csv") if row["variant"] in ("A", "B")]
    scorecard = read_csv(root / "merged_scorecard.csv")
    effects_rows = read_csv(root / "c2f_effect_summary.csv")
    effects: dict[str, dict[str, Any]] = {}
    for row in effects_rows:
        row["fine_rank"] = int(row["fine_rank"])
        row["coarse_rank"] = int(row["coarse_rank"])
        row["c2f_pass_count"] = int(row["c2f_pass_count"])
        row["comparable_pairs"] = int(row["comparable_pairs"])
        row["beats_better_parent_count"] = int(row["beats_better_parent_count"])
        row["median_percent_delta_vs_better_parent"] = as_float(row["median_percent_delta_vs_better_parent"])
        effects[row["standard_label"]] = row
    fine, coarse = read_grid_candidates(architecture)
    data: dict[str, Any] = {
        "architecture": architecture,
        "root": root,
        "pairwise": pairwise,
        "scorecard": scorecard,
        "effects": effects,
        "fine": fine,
        "coarse": coarse,
        "score_by_label_dataset": {(row["standard_label"], row["dataset_key"]): row for row in scorecard},
        "pair_by_label_dataset": {(row["standard_label"], row["dataset_key"]): row for row in pairwise},
    }
    add_family_triple_statistics(data, effects)
    return data


def variant_summary(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for variant in ("A", "B"):
        rows = [row for row in data["pairwise"] if row["variant"] == variant]
        comparable = [row for row in rows if row["comparison_status"] == "COMPARABLE"]
        labels = [label for label, effect in data["effects"].items() if effect["variant"] == variant]
        family_counts = {
            family: sum(family in data["effects"][label]["family_triple_wins"] for label in labels)
            for family in FAMILY_KEYS
        }
        out[variant] = {
            "pairs": len(labels),
            "c2f_pass": sum(row["c2f_status"] == "PASS" for row in rows),
            "c2f_total": len(rows),
            "comparable": len(comparable),
            "wins": sum(is_true(row["beats_better_parent"]) for row in comparable),
            "family_counts": family_counts,
            "any_family": sum(bool(data["effects"][label]["family_triple_wins"]) for label in labels),
            "all_families": sum(len(data["effects"][label]["family_triple_wins"]) == len(FAMILY_KEYS) for label in labels),
        }
    return out


def variant_head_to_head(data: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[int, int, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in data["pairwise"]:
        groups[(int(row["fine_rank"]), int(row["coarse_rank"]), row["dataset_key"])][row["variant"]] = row
    cells: list[tuple[float, float, tuple[int, int, str]]] = []
    for key, pair in groups.items():
        if set(pair) != {"A", "B"} or pair["A"]["c2f_status"] != "PASS" or pair["B"]["c2f_status"] != "PASS":
            continue
        cells.append((float(pair["A"]["c2f_ate_mean_cm"]), float(pair["B"]["c2f_ate_mean_cm"]), key))
    pairs: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for a, b, (fine_rank, coarse_rank, _) in cells:
        pairs[(fine_rank, coarse_rank)].append((a, b))
    medians = [(median(a for a, _ in values), median(b for _, b in values)) for values in pairs.values()]
    return {
        "cells": len(cells),
        "a_lower_cells": sum(a < b for a, b, _ in cells),
        "b_lower_cells": sum(b < a for a, b, _ in cells),
        "median_b_minus_a_cm": median([b - a for a, b, _ in cells]),
        "pairs": len(medians),
        "a_lower_pair_medians": sum(a < b for a, b in medians),
        "b_lower_pair_medians": sum(b < a for a, b in medians),
    }


def sorted_top_candidates(unet: dict[str, Any], resnet: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for data in (unet, resnet):
        for label, effect in data["effects"].items():
            record = dict(effect)
            record["architecture"] = data["architecture"]
            record["label"] = label
            records.append(record)
    # Robust added value first: within-sequence wins, then three-condition
    # family wins, then magnitude of the median paired percentage gain.
    records.sort(
        key=lambda row: (
            -int(row["beats_better_parent_count"]),
            -int(row["family_triple_count"]),
            float(row["median_percent_delta_vs_better_parent"]) if row["median_percent_delta_vs_better_parent"] is not None else math.inf,
            row["architecture"],
            row["label"],
        )
    )
    return records


def make_triple_conditions_plot(unet: dict[str, Any], resnet: dict[str, Any], output: Path) -> None:
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.1), constrained_layout=True, sharey=False)
    for axis, data in zip(axes, (unet, resnet)):
        summary = variant_summary(data)
        x = np.arange(len(FAMILY_KEYS))
        width = 0.34
        a = [summary["A"]["family_counts"][family] for family in FAMILY_KEYS]
        b = [summary["B"]["family_counts"][family] for family in FAMILY_KEYS]
        bars_a = axis.bar(x - width / 2, a, width, label="Variant A", color="#2E75B6")
        bars_b = axis.bar(x + width / 2, b, width, label="Variant B", color="#ED7D31")
        denom = summary["A"]["pairs"]
        axis.set_title(f"{architecture_name(data['architecture'])}: all three lighting conditions")
        axis.set_xticks(x, [FAMILY_SHORT[family] for family in FAMILY_KEYS])
        axis.set_ylabel("# C2F pairs beating better parent in clean + flash + lightswitch")
        axis.set_ylim(0, max(max(a), max(b), 1) + 2)
        axis.grid(axis="y", alpha=0.24)
        for bars in (bars_a, bars_b):
            for bar in bars:
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{int(bar.get_height())}/{denom}", ha="center", va="bottom", fontsize=9)
        axis.legend(loc="upper right")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_variant_effect_plot(unet: dict[str, Any], resnet: dict[str, Any], output: Path) -> None:
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
    datasets = (unet, resnet)
    summaries = [variant_summary(data) for data in datasets]
    heads = [variant_head_to_head(data) for data in datasets]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.35), constrained_layout=True)
    x = np.arange(2)
    width = 0.32
    for index, variant in enumerate(("A", "B")):
        values = [100 * item[variant]["wins"] / item[variant]["comparable"] for item in summaries]
        bars = axes[0].bar(x + (index - 0.5) * width, values, width, label=f"Variant {variant}", color=("#2E75B6", "#ED7D31")[index])
        for bar, value, summary in zip(bars, values, summaries):
            v = summary[variant]
            axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1.6, f"{v['wins']}/{v['comparable']}", ha="center", fontsize=9)
    axes[0].set_xticks(x, [architecture_name(data["architecture"]) for data in datasets])
    axes[0].set_ylim(0, 75)
    axes[0].set_ylabel("Win rate vs same-sequence better parent (%)")
    axes[0].set_title("Paired C2F gain by pyramid routing variant")
    axes[0].grid(axis="y", alpha=0.24)
    axes[0].legend(loc="upper right")

    a_lower = [head["a_lower_cells"] for head in heads]
    b_lower = [head["b_lower_cells"] for head in heads]
    total = [head["cells"] for head in heads]
    bars_a = axes[1].bar(x - width / 2, a_lower, width, label="A lower C2F ATE", color="#2E75B6")
    bars_b = axes[1].bar(x + width / 2, b_lower, width, label="B lower C2F ATE", color="#ED7D31")
    axes[1].set_xticks(x, [architecture_name(data["architecture"]) for data in datasets])
    axes[1].set_ylabel("# same F×C×sequence comparisons")
    axes[1].set_title("Direct A-vs-B head-to-head (both C2F PASS)")
    axes[1].grid(axis="y", alpha=0.24)
    for bars, values in ((bars_a, a_lower), (bars_b, b_lower)):
        for bar, value, n in zip(bars, values, total):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + max(total) * 0.018, f"{value}/{n}", ha="center", fontsize=9)
    # Keep the very tall ResNet-B annotation unobstructed.
    axes[1].legend(loc="upper left")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_top10_table(top: list[dict[str, Any]], data_by_arch: dict[str, dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
    headers = ["Configuration", "wins\n/9", "median Δ\nvs parent", *[key.replace("fr1_desk_", "fr1\n").replace("fr2_desk_", "fr2\n").replace("fr3_long_office_household_", "fr3\n").replace("flashlight", "flash").replace("lightswitch", "light\nswitch") for key in TABLE_DATASET_KEYS]]
    rows: list[list[str]] = []
    cell_wins: list[list[bool]] = []
    csv_rows: list[dict[str, Any]] = []
    for position, candidate in enumerate(top, start=1):
        data = data_by_arch[candidate["architecture"]]
        label = candidate["label"]
        config = short_label(candidate["architecture"], label)
        row = [f"{position}. {config}", f"{candidate['beats_better_parent_count']}/9", signed_percent(candidate["median_percent_delta_vs_better_parent"])]
        wins: list[bool] = []
        csv_row = {
            "rank": position,
            "architecture": architecture_name(candidate["architecture"]),
            "label": label,
            "configuration": config,
            "variant": candidate["variant"],
            "fine_rank": candidate["fine_rank"],
            "coarse_rank": candidate["coarse_rank"],
            "wins_vs_better_parent": candidate["beats_better_parent_count"],
            "family_triple_wins": ";".join(candidate["family_triple_wins"]),
            "median_percent_delta_vs_better_parent": candidate["median_percent_delta_vs_better_parent"],
            "components": component_text(candidate["architecture"], candidate["fine_rank"], candidate["coarse_rank"], data["fine"], data["coarse"]),
        }
        for dataset_key in TABLE_DATASET_KEYS:
            score = data["score_by_label_dataset"][(label, dataset_key)]
            pair = data["pair_by_label_dataset"][(label, dataset_key)]
            value = as_float(score["historical_evo_ape_mean_cm"]) if score["status"] == "PASS" else None
            row.append(cm(value))
            wins.append(pair["comparison_status"] == "COMPARABLE" and is_true(pair["beats_better_parent"]))
            csv_row[dataset_key + "_ate_mean_cm"] = value
            csv_row[dataset_key + "_beats_better_parent"] = wins[-1]
        rows.append(row)
        cell_wins.append(wins)
        csv_rows.append(csv_row)

    fig, axis = plt.subplots(figsize=(19.0, 6.7), constrained_layout=True)
    axis.axis("off")
    widths = [0.17, 0.055, 0.075] + [0.078] * 9
    table = axis.table(cellText=rows, colLabels=headers, cellLoc="center", colLoc="center", colWidths=widths, loc="upper left", bbox=[0.0, 0.06, 1.0, 0.92])
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#1F1F1F")
        cell.set_linewidth(0.9)
        if row_index == 0:
            cell.set_facecolor("#234F84")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(8.8)
        else:
            cell.set_facecolor("#EDF2FA" if row_index % 2 else "#FFFFFF")
            if col_index == 0:
                cell.get_text().set_fontweight("bold")
            if col_index >= 3:
                win = cell_wins[row_index - 1][col_index - 3]
                if win:
                    cell.get_text().set_color("#00843D")
                    cell.get_text().set_fontweight("bold")
                elif cell.get_text().get_text() == "FAIL":
                    cell.get_text().set_color("#B00020")
                    cell.get_text().set_fontweight("bold")
    axis.text(0.0, 0.012, "Primary metric: historical keyframe evo_ape ATE mean (cm). Green = C2F lower than its same-sequence better direct parent; black = not lower.", fontsize=10, transform=axis.transAxes)
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return csv_rows


def grouped_parent_win_rates(data: dict[str, Any], group_key: str, groups: tuple[str, ...]) -> dict[str, dict[str, tuple[int, int]]]:
    """Return wins/comparable cells for A/B, grouped by scene or condition."""
    result: dict[str, dict[str, tuple[int, int]]] = {variant: {} for variant in ("A", "B")}
    for variant in ("A", "B"):
        variant_rows = [row for row in data["pairwise"] if row["variant"] == variant]
        for group in groups:
            comparable = [
                row for row in variant_rows
                if row[group_key] == group and row["comparison_status"] == "COMPARABLE"
            ]
            result[variant][group] = (
                sum(is_true(row["beats_better_parent"]) for row in comparable),
                len(comparable),
            )
    return result


def make_scene_win_rate_plot(unet: dict[str, Any], resnet: dict[str, Any], output: Path) -> None:
    """Make a readable per-scene summary across the three lighting conditions."""
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.55), constrained_layout=True, sharey=True)
    for axis, data in zip(axes, (unet, resnet)):
        summary = grouped_parent_win_rates(data, "dataset_family", FAMILY_KEYS)
        x = np.arange(len(FAMILY_KEYS))
        width = 0.34
        for variant_index, (offset, variant, color) in enumerate(((-width / 2, "A", "#2E75B6"), (width / 2, "B", "#ED7D31"))):
            numerators = [summary[variant][family][0] for family in FAMILY_KEYS]
            denominators = [summary[variant][family][1] for family in FAMILY_KEYS]
            rates = [100 * numerator / denominator if denominator else 0 for numerator, denominator in zip(numerators, denominators)]
            bars = axis.bar(x + offset, rates, width, label=f"Variant {variant}", color=color)
            for bar, numerator, denominator, rate in zip(bars, numerators, denominators, rates):
                axis.text(bar.get_x() + bar.get_width() / 2, rate + 1.4 + 2.2 * variant_index, f"{numerator}/{denominator}", ha="center", va="bottom", fontsize=8.1)
        axis.set_title(f"{architecture_name(data['architecture'])}: grouped by scene")
        axis.set_xticks(x, ["fr1", "fr2", "fr3"])
        axis.set_ylim(0, 86)
        axis.grid(axis="y", alpha=0.24)
        axis.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("C2F win rate vs same-sequence better parent (%)\n(across clean + flashlight + lightswitch)")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def make_lighting_win_rate_plot(unet: dict[str, Any], resnet: dict[str, Any], output: Path) -> None:
    """Make a readable per-lighting-condition summary across the three scenes."""
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.55), constrained_layout=True, sharey=True)
    conditions = ("clean", "flashlight", "lightswitch")
    labels = ("clean", "flash", "light\nswitch")
    for axis, data in zip(axes, (unet, resnet)):
        summary = grouped_parent_win_rates(data, "dataset_condition", conditions)
        x = np.arange(len(conditions))
        width = 0.34
        for variant_index, (offset, variant, color) in enumerate(((-width / 2, "A", "#2E75B6"), (width / 2, "B", "#ED7D31"))):
            numerators = [summary[variant][condition][0] for condition in conditions]
            denominators = [summary[variant][condition][1] for condition in conditions]
            rates = [100 * numerator / denominator if denominator else 0 for numerator, denominator in zip(numerators, denominators)]
            bars = axis.bar(x + offset, rates, width, label=f"Variant {variant}", color=color)
            for bar, numerator, denominator, rate in zip(bars, numerators, denominators, rates):
                axis.text(bar.get_x() + bar.get_width() / 2, rate + 1.4 + 2.2 * variant_index, f"{numerator}/{denominator}", ha="center", va="bottom", fontsize=8.1)
        axis.set_title(f"{architecture_name(data['architecture'])}: grouped by lighting condition")
        axis.set_xticks(x, labels)
        axis.set_ylim(0, 86)
        axis.grid(axis="y", alpha=0.24)
        axis.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("C2F win rate vs same-sequence better parent (%)\n(across fr1 + fr2 + fr3)")
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def ranked_unique_pairs(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Rank unique F×C pairs using their stronger A/B routing result.

    A detailed result table must contain both variants, so a pair may appear
    only once in the shortlist even if both A and B individually rank highly.
    """
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for effect in data["effects"].values():
        groups[(effect["fine_rank"], effect["coarse_rank"])].append(effect)
    ranked: list[dict[str, Any]] = []
    for (fine_rank, coarse_rank), variants in groups.items():
        variants.sort(
            key=lambda row: (
                -int(row["beats_better_parent_count"]),
                -int(row["family_triple_count"]),
                float(row["median_percent_delta_vs_better_parent"]) if row["median_percent_delta_vs_better_parent"] is not None else math.inf,
                row["variant"],
            )
        )
        winner = dict(variants[0])
        winner["architecture"] = data["architecture"]
        winner["best_variant"] = winner["variant"]
        winner["variant_effects"] = {row["variant"]: row for row in variants}
        ranked.append(winner)
    ranked.sort(
        key=lambda row: (
            -int(row["beats_better_parent_count"]),
            -int(row["family_triple_count"]),
            float(row["median_percent_delta_vs_better_parent"]) if row["median_percent_delta_vs_better_parent"] is not None else math.inf,
            row["fine_rank"], row["coarse_rank"],
        )
    )
    return ranked


def standard_label(architecture: str, kind: str, fine_rank: int, coarse_rank: int | None = None, variant: str | None = None) -> str:
    if kind == "fine":
        return f"{architecture}_direct_fine{fine_rank:02d}"
    if kind == "coarse":
        assert coarse_rank is not None
        return f"{architecture}_direct_coarse{coarse_rank:02d}"
    assert coarse_rank is not None and variant in ("A", "B")
    return f"{architecture}_c2f_{variant.lower()}_fine{fine_rank:02d}_coarse{coarse_rank:02d}"


def make_pair_overview_table(top_pairs: list[dict[str, Any]], data: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    """Produce the first of six presentation tables for one architecture."""
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
    headers = ["Rank", "F + C pair", "Best\nvariant", "Best wins\n/9", "3-light\nfamily", "Best median Δ", "Other variant\nwins /9"]
    values: list[list[str]] = []
    csv_rows: list[dict[str, Any]] = []
    for rank, record in enumerate(top_pairs, start=1):
        alternate = "B" if record["best_variant"] == "A" else "A"
        other = record["variant_effects"][alternate]
        short = f"F{record['fine_rank']} + C{record['coarse_rank']}"
        values.append([
            str(rank), short, record["best_variant"], f"{record['beats_better_parent_count']}/9",
            ", ".join(FAMILY_SHORT[item] for item in record["family_triple_wins"]) or "—",
            signed_percent(record["median_percent_delta_vs_better_parent"]),
            f"{other['beats_better_parent_count']}/9",
        ])
        csv_rows.append({
            "rank": rank,
            "architecture": architecture_name(data["architecture"]),
            "fine_rank": record["fine_rank"],
            "coarse_rank": record["coarse_rank"],
            "best_variant": record["best_variant"],
            "best_wins_vs_better_parent": record["beats_better_parent_count"],
            "best_family_triple_wins": ";".join(record["family_triple_wins"]),
            "best_median_percent_delta": record["median_percent_delta_vs_better_parent"],
            "other_variant": alternate,
            "other_wins_vs_better_parent": other["beats_better_parent_count"],
            "components": component_text(data["architecture"], record["fine_rank"], record["coarse_rank"], data["fine"], data["coarse"]),
        })
    fig, axis = plt.subplots(figsize=(11.4, 3.6), constrained_layout=True)
    axis.axis("off")
    table = axis.table(cellText=values, colLabels=headers, cellLoc="center", colLoc="center", colWidths=[0.08, 0.19, 0.13, 0.16, 0.18, 0.15, 0.17], bbox=[0.02, 0.10, 0.96, 0.84])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#1F1F1F")
        cell.set_linewidth(0.9)
        if row == 0:
            cell.set_facecolor("#234F84")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#EAF0F9" if row % 2 else "#FFFFFF")
            if col in (0, 1):
                cell.get_text().set_fontweight("bold")
    axis.text(0.02, 0.015, "Pair ranking: wins vs better parent → number of all-lighting scene families → median paired delta. Detailed A/B parent tables follow.", fontsize=9.2, transform=axis.transAxes)
    fig.savefig(output, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return csv_rows


def make_detailed_pair_table(record: dict[str, Any], data: dict[str, Any], output: Path, rank: int) -> None:
    """Create one sample-style parent/A/B table for a selected F×C pair."""
    plt.rcParams.update({"font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False})
    architecture = data["architecture"]
    fine_rank, coarse_rank = int(record["fine_rank"]), int(record["coarse_rank"])
    row_specs = [
        ("fine-only", standard_label(architecture, "fine", fine_rank)),
        ("coarse-only", standard_label(architecture, "coarse", fine_rank, coarse_rank)),
        ("C2F-A", standard_label(architecture, "c2f", fine_rank, coarse_rank, "A")),
        ("C2F-B", standard_label(architecture, "c2f", fine_rank, coarse_rank, "B")),
    ]
    headers = ["Configuration", *[key.replace("fr1_desk_", "fr1\n").replace("fr2_desk_", "fr2\n").replace("fr3_long_office_household_", "fr3\n").replace("flashlight", "flash").replace("lightswitch", "light\nswitch") for key in TABLE_DATASET_KEYS]]
    cell_values: list[list[str]] = []
    cell_c2f_wins: list[list[bool]] = []
    cell_best: list[list[bool]] = []
    all_values: dict[str, list[float | None]] = {}
    for row_name, label in row_specs:
        all_values[label] = [
            as_float(data["score_by_label_dataset"][(label, dataset_key)]["historical_evo_ape_mean_cm"])
            if data["score_by_label_dataset"][(label, dataset_key)]["status"] == "PASS" else None
            for dataset_key in TABLE_DATASET_KEYS
        ]
        cell_values.append([row_name, *[cm(value) for value in all_values[label]]])
        cell_c2f_wins.append([
            row_name.startswith("C2F")
            and data["pair_by_label_dataset"][(label, dataset_key)]["comparison_status"] == "COMPARABLE"
            and is_true(data["pair_by_label_dataset"][(label, dataset_key)]["beats_better_parent"])
            for dataset_key in TABLE_DATASET_KEYS
        ])
    for row_name, label in row_specs:
        bests: list[bool] = []
        for index, _dataset_key in enumerate(TABLE_DATASET_KEYS):
            candidates = [values[index] for values in all_values.values() if values[index] is not None]
            bests.append(all_values[label][index] is not None and all_values[label][index] == min(candidates))
        cell_best.append(bests)

    fine_text = "[" + ",".join(map(str, data["fine"][fine_rank])) + "]"
    coarse_text = "[" + ",".join(map(str, data["coarse"][coarse_rank])) + "]"
    fig, axis = plt.subplots(figsize=(15.4, 4.0), constrained_layout=True)
    axis.axis("off")
    widths = [0.19] + [0.09] * 9
    table = axis.table(cellText=cell_values, colLabels=headers, cellLoc="center", colLoc="center", colWidths=widths, bbox=[0.0, 0.09, 1.0, 0.77])
    table.auto_set_font_size(False)
    table.set_fontsize(10.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#1F1F1F")
        cell.set_linewidth(1.0)
        if row == 0:
            cell.set_facecolor("#234F84")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#EAF0F9" if row % 2 else "#FFFFFF")
            if col == 0:
                cell.get_text().set_fontweight("bold")
            if col >= 1:
                index = col - 1
                if cell_c2f_wins[row - 1][index]:
                    cell.get_text().set_color("#00843D")
                    cell.get_text().set_fontweight("bold")
                if cell_best[row - 1][index]:
                    cell.set_facecolor("#FFF200")
                if cell.get_text().get_text() == "FAIL":
                    cell.get_text().set_color("#B00020")
                    cell.get_text().set_fontweight("bold")
    arch_title = architecture_name(architecture)
    axis.set_title(f"{arch_title} Top-{rank} pair: F{fine_rank} + C{coarse_rank}  |  fine {fine_text}; coarse {coarse_text}", fontsize=14, fontweight="bold", pad=10)
    axis.text(0.0, 0.012, "Metric: historical keyframe evo_ape ATE mean (cm). Green = C2F beats its better direct parent; yellow = best ATE among the four rows.", fontsize=9.3, transform=axis.transAxes)
    fig.savefig(output, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    unet = load_architecture("unet")
    resnet = load_architecture("resnet")
    data_by_arch = {"unet": unet, "resnet": resnet}
    datasets = {item["key"]: item for item in json.loads(STEP_Q_PLAN.read_text(encoding="utf-8"))["datasets"]}
    if set(TABLE_DATASET_KEYS) != set(datasets):
        raise ValueError("The fixed report ordering does not match the frozen nine-dataset plan")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {"unet": variant_summary(unet), "resnet": variant_summary(resnet)}
    heads = {"unet": variant_head_to_head(unet), "resnet": variant_head_to_head(resnet)}
    scene_plot = RESULT_ROOT / "c2f_parent_win_rate_by_scene.png"
    lighting_plot = RESULT_ROOT / "c2f_parent_win_rate_by_lighting.png"
    variant_plot = RESULT_ROOT / "variant_effect_summary.png"
    make_scene_win_rate_plot(unet, resnet, scene_plot)
    make_lighting_win_rate_plot(unet, resnet, lighting_plot)
    make_variant_effect_plot(unet, resnet, variant_plot)

    top_pairs = {architecture: ranked_unique_pairs(data)[:5] for architecture, data in data_by_arch.items()}
    table_images: dict[str, list[Path]] = {"unet": [], "resnet": []}
    ranked_pair_csv_rows: list[dict[str, Any]] = []
    for architecture, data in data_by_arch.items():
        overview_path = RESULT_ROOT / f"{architecture}_top5_pair_overview.png"
        ranked_pair_csv_rows.extend(make_pair_overview_table(top_pairs[architecture], data, overview_path))
        table_images[architecture].append(overview_path)
        for rank, record in enumerate(top_pairs[architecture], start=1):
            detail_path = RESULT_ROOT / f"{architecture}_top{rank}_pair_f{record['fine_rank']}_c{record['coarse_rank']}.png"
            make_detailed_pair_table(record, data, detail_path, rank)
            table_images[architecture].append(detail_path)
    write_csv(RESULT_ROOT / "top5_c2f_pairs_by_architecture.csv", ranked_pair_csv_rows, list(ranked_pair_csv_rows[0]))

    effect_overview: list[dict[str, Any]] = []
    for architecture, data in data_by_arch.items():
        for label, effect in data["effects"].items():
            effect_overview.append({
                "architecture": architecture_name(architecture),
                "label": label,
                "configuration": short_label(architecture, label),
                "variant": effect["variant"],
                "fine_rank": effect["fine_rank"],
                "coarse_rank": effect["coarse_rank"],
                "c2f_pass_count": effect["c2f_pass_count"],
                "comparable_pairs": effect["comparable_pairs"],
                "wins_vs_better_parent": effect["beats_better_parent_count"],
                "family_triple_wins": ";".join(effect["family_triple_wins"]),
                "median_percent_delta_vs_better_parent": effect["median_percent_delta_vs_better_parent"],
            })
    effect_overview.sort(key=lambda row: (-int(row["wins_vs_better_parent"]), float(row["median_percent_delta_vs_better_parent"]) if row["median_percent_delta_vs_better_parent"] is not None else math.inf))
    write_csv(RESULT_ROOT / "all_c2f_effect_overview.csv", effect_overview, list(effect_overview[0]))

    unet_pass = sum(row["status"] == "PASS" for row in unet["scorecard"])
    resnet_pass = sum(row["status"] == "PASS" for row in resnet["scorecard"])
    c2f_fail_resnet = sum(row["c2f_status"] != "PASS" for row in resnet["pairwise"])

    triple_rows: list[list[object]] = []
    for architecture in ("unet", "resnet"):
        summary = summaries[architecture]
        for variant in ("A", "B"):
            value = summary[variant]
            triple_rows.append([
                architecture_name(architecture), variant, value["pairs"],
                f"{value['family_counts']['fr1_desk']}/{value['pairs']}",
                f"{value['family_counts']['fr2_desk']}/{value['pairs']}",
                f"{value['family_counts']['fr3_long_office_household']}/{value['pairs']}",
                f"{value['any_family']}/{value['pairs']}",
                f"{value['all_families']}/{value['pairs']}",
            ])

    variant_rows: list[list[object]] = []
    for architecture in ("unet", "resnet"):
        summary = summaries[architecture]
        head = heads[architecture]
        for variant in ("A", "B"):
            value = summary[variant]
            rate = 100 * value["wins"] / value["comparable"]
            variant_rows.append([
                architecture_name(architecture), variant,
                f"{value['c2f_pass']}/{value['c2f_total']}",
                f"{value['wins']}/{value['comparable']} ({rate:.1f}%)",
                "A lower" if variant == "A" else "B lower",
                f"{head['a_lower_cells'] if variant == 'A' else head['b_lower_cells']}/{head['cells']}",
                f"{head['a_lower_pair_medians'] if variant == 'A' else head['b_lower_pair_medians']}/{head['pairs']}",
            ])

    lines: list[str] = []
    add = lines.append
    add("---")
    add("title: \"C2F 完整网格九数据集评估（U-Net 与 ResNet）\"")
    add("subtitle: \"Step-R · fine/coarse promising subsets 的系统配对验证\"")
    add("date: \"2026-08-27\"")
    add("---")
    add("")
    add("# 1. 结论摘要")
    add("")
    add(
        "本报告完成了在 fr1 / fr2 / fr3 三个场景、clean / flashlight / lightswitch 三种光照条件上的完整 C2F 配对验证。"
        "核心判断不是跨数据集直接平均原始 ATE，而是每个 C2F 在**同一完整序列**上是否低于其 fine 与 coarse direct parent 中表现较好的一个（better parent）。"
    )
    add("")
    add(
        f"- **U-Net：C2F 在所选网格中表现出可复现但非普遍的增益。** 全部 {unet_pass}/441 个合并 cells PASS；"
        f"A/B 两个 variant 分别有 {summaries['unet']['A']['wins']}/{summaries['unet']['A']['comparable']}（62.2%）和 "
        f"{summaries['unet']['B']['wins']}/{summaries['unet']['B']['comparable']}（61.1%）次优于 better parent。"
        "严格要求 clean、flashlight、lightswitch 三者都胜出的组合，在 fr1/fr2/fr3 分别为 2/21/14 个。"
    )
    add(
        f"- **ResNet：C2F-B 明显优于 C2F-A，但跨 fr3 三光照的稳健收益不足。** {resnet_pass}/639 个合并 cells PASS，"
        f"其中 C2F 本身有 {c2f_fail_resnet} 个 tracking-NaN failure；A/B 的 better-parent 胜率仅为 "
        f"{summaries['resnet']['A']['wins']}/{summaries['resnet']['A']['comparable']}（18.6%）和 "
        f"{summaries['resnet']['B']['wins']}/{summaries['resnet']['B']['comparable']}（40.0%）。"
        "在 fr3 没有任何 ResNet C2F 组合能同时胜出三种光照。"
    )
    add(
        "- **没有一个 C2F 组合能在 fr1、fr2、fr3 的九个完整序列上全部优于其 better parent。** 因而 C2F 的正确结论是“特定通道组合与金字塔 routing 可带来条件性互补”，而不是保证性的架构升级。"
    )
    add("")
    add("# 2. 实验设置与比较规则")
    add("")
    add(markdown_table(
        ["项目", "固定设置"],
        [
            ["U-Net grid", "Enc0 fine Top-5 × Enc1 coarse Top-4 × C2F-A/B = 40 C2F pairs；另含 9 个 direct parents"],
            ["ResNet grid", "Conv1 fine Top-6 × Layer2 coarse Top-5 × C2F-A/B = 60 C2F pairs；另含 11 个 direct parents"],
            ["数据", "fr1/desk、fr2/desk、fr3/long_office_household × clean / flashlight / lightswitch = 9 条完整序列"],
            ["Mapping", "固定 gray + sensor depth；仅 tracking feature mode / selected channels 改变"],
            ["C2F-A", "coarse 使用较浅 pyramid levels；fine 使用更深 level"],
            ["C2F-B", "coarse 使用最浅 level；fine 使用其余两个 level"],
            ["主指标", "historical keyframe evo_ape translation ATE mean（--align --correct_scale），单位 cm"],
            ["可比较定义", "C2F、direct fine、direct coarse 都 PASS，且 C2F ATE < min(fine ATE, coarse ATE)"],
            ["运行规范", "每个 cell 1 次；timeout = 500 s；Step-Q 既有记录只读复用、不重跑"],
        ],
    ))
    add("")
    add("# 3. 总体统计：按 scene 与按 lighting condition")
    add("")
    add(
        "下列两图都使用同一个配对判定：只有 C2F、fine direct、coarse direct 都 PASS，且 C2F ATE 低于两个 direct parent 中较好的一个，才记作一次胜出。"
        "柱顶的 `胜出/可比较` 明确显示 ResNet tracking failure 所造成的有效分母变化。"
    )
    add("")
    add("## 3.1 按 scene 汇总")
    add("")
    add(f"![按 scene 的 C2F 胜率]({scene_plot.name}){{ width=95% }}")
    add("")
    add("**读图。** U-Net 的明显优势集中在 fr2 与 fr3；U-Net 的 fr2 最强，fr1 最弱。fr1 上 U-Net-A 仍高于两个 ResNet variant，但 U-Net-B（21/60）略低于 ResNet-B（35/88），因此不能把“U-Net 更好”理解成每个 routing、每个 scene 都绝对占优。ResNet-B 虽优于 A，但 fr3 的胜率仍低，显示其 C2F 增益没有稳定迁移到 long-office household 场景。")
    add("")
    add("## 3.2 按 lighting condition 汇总")
    add("")
    add(f"![按光照条件的 C2F 胜率]({lighting_plot.name}){{ width=95% }}")
    add("")
    add("**读图。** U-Net 的两种 routing 在 clean / flashlight / lightswitch 下均维持约六成的配对胜率；ResNet B 在 lightswitch 上相对更有利，但 clean 与 flashlight 中仍缺乏一致收益。")
    add("")
    add("## 3.3 严格的三光照稳健性")
    add("")
    add("若要求同一 F×C×variant 在某个 scene family 的 clean、flashlight、lightswitch 三条完整序列都优于 better parent，统计如下。这是最直接的“跨三种光照是否稳定有效”答案。")
    add("")
    add(markdown_table(["架构", "variant", "候选 pairs", "fr1 三光照", "fr2 三光照", "fr3 三光照", "至少一个 family", "三个 family 都满足"], triple_rows))
    add("")
    add("没有一个配置达到 9/9（即三个 family 的三种光照都胜出）。U-Net 的严格收益集中在 fr2 和 fr3；ResNet 在 fr3 为零。")
    add("")
    add("# 4. C2F variant 的影响")
    add("")
    add(markdown_table(
        ["架构", "variant", "C2F PASS", "优于 better parent", "同 F×C 原始 ATE 对决", "胜出 cells", "胜出 pair-median"],
        variant_rows,
    ))
    add("")
    add("“同 F×C 原始 ATE 对决”只比较 A/B 都 PASS 的相同 fine subset、coarse subset、同一序列；此时 direct parents 完全相同，因此可直接比较两种 routing。pair-median 是每个 F×C 跨可用序列的中位 ATE 后再作比较。")
    add("")
    add(f"![Variant 影响]({variant_plot.name}){{ width=95% }}")
    add("")
    add(
        f"- **U-Net：A/B 的 parent 胜率接近**（62.2% vs 61.1%），但在 {heads['unet']['cells']} 个直接 A/B cell 对决中 A 以 {heads['unet']['a_lower_cells']}:{heads['unet']['b_lower_cells']} 更常得到较低 ATE，"
        f"在 20 个 F×C 的 sequence-median 比较中也以 {heads['unet']['a_lower_pair_medians']}:{heads['unet']['b_lower_pair_medians']} 占优。"
        "因此 U-Net 的默认优先级应为 C2F-A，但 B 仍有少数高质量组合（例如 F5+C4），不应被整体平均掩盖。"
    )
    add(
        f"- **ResNet：B 是明显较可靠的 routing。** 在 {heads['resnet']['cells']} 个双方 PASS 的 A/B cell 对决中，B 以 {heads['resnet']['b_lower_cells']}:{heads['resnet']['a_lower_cells']} 占优，"
        f"并在 30 个 F×C 的 median 比较中以 {heads['resnet']['b_lower_pair_medians']}:{heads['resnet']['a_lower_pair_medians']} 占优。"
        "但 B 的 40.0% better-parent 胜率和 fr3 的零个三光照解表明：routing 修正了 A 的问题，却没有使 ResNet C2F 成为跨场景默认策略。"
    )
    add("")
    add("# 5. 具体结果：每个架构的 Top-5 F×C pairs")
    add("")
    add(
        "每个架构均提供 **6 张表**：第 1 张是 Top-5 unique F×C pair 的排名概览；随后 5 张逐 pair 结果表分别列出 fine-only、coarse-only、C2F-A 与 C2F-B。"
        "pair 排名按：九序列中优于 better parent 的次数 → 满足三光照的 scene family 数 → median paired delta。"
    )
    add("")
    for architecture in ("unet", "resnet"):
        arch_title = architecture_name(architecture)
        add(f"## 5.{1 if architecture == 'unet' else 2} {arch_title}")
        add("")
        add(f"![{arch_title} Top-5 pair overview]({table_images[architecture][0].name}){{ width=96% }}")
        add("")
        for rank, (record, image_path) in enumerate(zip(top_pairs[architecture], table_images[architecture][1:]), start=1):
            add(f"### Top-{rank}: F{record['fine_rank']} + C{record['coarse_rank']}（best routing: {record['best_variant']}；{record['beats_better_parent_count']}/9）")
            add("")
            add(f"![{arch_title} Top-{rank} detailed table]({image_path.name}){{ width=100% }}")
            add("")
    add("绿色数值表示该 C2F variant 优于此列的 better direct parent；黄色底色表示四行中 ATE 最低。这样可以同时看出：C2F 是否超过 parent，以及若两种 C2F 都成功，哪一种 routing 的绝对 ATE 更低。")
    add("")
    add("# 6. 关键发现")
    add("")
    add("1. **U-Net 形成了最清晰的 C2F 互补证据。** U-A F4+C2 为总体最强 added-value 配置（8/9、median Δ −13.8%），U-A F5+C4、U-A F1+C2、U-B F5+C4 与 U-A F2+C4 均为 8/9。它们说明浅层 Enc0 的 selected local structures 在加入 Enc1 coarse context 后，可在多个光照变化下获得稳定的配对改善。")
    add("2. **不是所有 direct-best parent 都需要 C2F。** 所有 Top-5 pair 仍存在黑色 cell，且 0 个组合达到 9/9。因此更合理的实际策略是：把 C2F 视为根据 scene/condition 选择的候选 tracking representation，而不是取代最强 direct parent 的固定默认。")
    add("3. **ResNet 的 C2F-B 比 A 更好，但仍不具跨场景稳健性。** B 的 parent win-rate 为 40.0%，超过 A 的 18.6%；R-B F3+C5 在 fr1、fr2 的三光照均有正结果。然而 ResNet 在 fr3 不存在三光照稳健组合，显示 Conv1+Layer2 的 current fusion/routing 仍容易受到 scene geometry 与 illumination distribution 改变的影响。")
    add("4. **必须保留负例。** ResNet A 的低胜率、ResNet fr3 的严格胜出为零、以及 U-Net 的 0 个 9/9 pair 都应与最佳单元格同时报告；它们使结论从“发现了一个低 ATE”提升为关于 C2F 条件性有效范围的可检验结论。")
    add("")
    add("# 7. 局限性与下一步")
    add("")
    add("- 每个 configuration×sequence 为单次完整运行，因此当前表展示的是跨序列趋势，不能估计每一格的运行间方差；最终 shortlist 应对关键 lightswitch sequences 做重复运行。")
    add("- 候选 fine/coarse subsets 来自 fr1/desk_lightswitch 的前序 direct search；该训练式选择会使 fr1 结果带有选择偏差。真正的外部证据主要来自余下 fr2/fr3 与另外两种光照。")
    add("- 主指标遵循项目一贯的 keyframe Sim(3)-aligned ATE mean。所有 all-frame metric-scale SE(3) ATE/RPE、coverage 和 diagnostic logs 仍保存在 Step-R 原始 SQLite/CSV 中，最终固定配置前应一并审阅。")
    add("- 下一步建议：以 U-A F4+C2、U-A F5+C4、U-A F2+C4 为优先 shortlist，保留 R-B F3+C5 作为 ResNet 的正对照；再在新增序列/退化条件上验证，而不是继续在同九序列上扩张 grid。")
    add("")
    add("# 附：可复核数据")
    add("")
    add(f"- 原始合并 pairwise 表：`{STEP_R_ROOT}/{{unet,resnet}}/merged_pairwise_comparison.csv`")
    add(f"- 本报告的全量 C2F effect 概览：`{RESULT_ROOT}/all_c2f_effect_overview.csv`")
    add(f"- 本报告每个架构 Top-5 pair 的机器可读表：`{RESULT_ROOT}/top5_c2f_pairs_by_architecture.csv`")
    add("")
    markdown_path = RESULT_ROOT / "C2F_完整网格九数据集评估_中文.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pandoc = Path("/home/melody/anaconda3/bin/pandoc")
    if not pandoc.is_file():
        located = shutil.which("pandoc")
        if located is None:
            raise RuntimeError("pandoc is required to create the Word report")
        pandoc = Path(located)
    docx_path = RESULT_ROOT / DOCX_NAME
    subprocess.run([
        str(pandoc), "--from", "markdown", "--to", "docx", "--standalone",
        "--resource-path", str(RESULT_ROOT), "--output", str(docx_path), str(markdown_path),
    ], check=True)
    print(f"[WRITE] {scene_plot}")
    print(f"[WRITE] {lighting_plot}")
    print(f"[WRITE] {variant_plot}")
    for architecture in ("unet", "resnet"):
        for image_path in table_images[architecture]:
            print(f"[WRITE] {image_path}")
    print(f"[WRITE] {RESULT_ROOT / 'all_c2f_effect_overview.csv'}")
    print(f"[WRITE] {RESULT_ROOT / 'top5_c2f_pairs_by_architecture.csv'}")
    print(f"[WRITE] {markdown_path}")
    print(f"[WRITE] {docx_path}")


if __name__ == "__main__":
    main()
