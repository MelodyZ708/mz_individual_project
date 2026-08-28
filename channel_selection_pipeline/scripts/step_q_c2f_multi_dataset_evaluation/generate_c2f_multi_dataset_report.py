#!/usr/bin/env python3
"""Create the Chinese Word report for the focused multi-sequence C2F study."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_c2f_multi_dataset_report")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
STAGE_ROOT = PROJECT_ROOT / "channel_selection_results/step_q_c2f_multi_dataset_evaluation"
RESULT_ROOT = PROJECT_ROOT / "channel_selection_results/reports/c2f_multi_dataset_parent_comparison"
DATASET_PLAN = SCRIPT_DIR / "c2f_multi_dataset_plan.json"
DOCX_NAME = "C2F_九数据集Direct_Parent对照评估_中文.docx"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cm(value: str | float | None) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.2f}"


def pct(value: str | float | None) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):+.1f}%"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def compact_label(label: str) -> str:
    aliases = {
        "gray_baseline": "Gray",
        "unet_enc1_direct_c4_d00_d05_d06_d18_d30": "U-C4 direct",
        "unet_enc1_direct_c2_d00_d05_d06_d17_d18_d30": "U-C2 direct",
        "unet_enc0_direct_f1_d02_d03_d07_d12_d13_d14": "U-F1 direct",
        "unet_enc0_direct_f2_d03_d07_d12": "U-F2 direct",
        "unet_enc0_direct_f5_d02_d03_d07_d12_d13": "U-F5 direct",
        "unet_c2f_a_f1_c4_negative_high_parent": "U-A F1+C4",
        "unet_c2f_a_f2_c4_global_best": "U-A F2+C4",
        "unet_c2f_a_f5_c4_positive_synergy": "U-A F5+C4",
        "unet_c2f_b_f2_c2_variant_contrast": "U-B F2+C2",
        "resnet_layer2_direct_c4_d67_d108_d121": "R-C4 direct",
        "resnet_layer2_direct_c5_d108_d121": "R-C5 direct",
        "resnet_conv1_direct_f1_d15_d20_d26_d34": "R-F1 direct",
        "resnet_conv1_direct_f2_d23_d24_d26_d51_d63": "R-F2 direct",
        "resnet_conv1_direct_f6_d29_d33_d52": "R-F6 direct",
        "resnet_c2f_b_f1_c4_negative_high_parent": "R-B F1+C4",
        "resnet_c2f_a_f2_c4_variant_negative": "R-A F2+C4",
        "resnet_c2f_b_f2_c4_global_best": "R-B F2+C4",
        "resnet_c2f_a_f6_c5_positive_synergy": "R-A F6+C5",
    }
    return aliases[label]


def config_text(candidate: dict[str, Any]) -> str:
    if candidate["mode"] == "gray":
        return "gray photometric baseline"
    if candidate["mode"] == "direct":
        return f"{candidate['layer']} [" + ",".join(str(value) for value in candidate["channels"]) + "]"
    fine = candidate["fine"]
    coarse = candidate["coarse"]
    return (
        f"C2F-{candidate['variant']}: fine {fine['layer']} [" + ",".join(str(value) for value in fine["channels"]) +
        f"]; coarse {coarse['layer']} [" + ",".join(str(value) for value in coarse["channels"]) + "]"
    )


def load_architecture(architecture: str) -> dict[str, Any]:
    root = STAGE_ROOT / architecture
    plan_path = SCRIPT_DIR / f"{architecture}_c2f_parent_comparison_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    candidates = plan["candidates"]
    scorecard = read_csv(root / "dataset_scorecard.csv")
    pairwise = read_csv(root / "c2f_pairwise_comparison.csv")
    effects = read_csv(root / "c2f_effect_summary.csv")
    variants = read_csv(root / "c2f_variant_summary.csv")
    return {
        "architecture": architecture,
        "root": root,
        "plan": plan,
        "candidates": candidates,
        "candidate_by_label": {item["label"]: item for item in candidates},
        "scorecard": scorecard,
        "pairwise": pairwise,
        "effects": effects,
        "variants": variants,
        "score_by_label_dataset": {(row["label"], row["dataset_key"]): row for row in scorecard},
        "pair_by_label_dataset": {(row["c2f_label"], row["dataset_key"]): row for row in pairwise},
    }


def make_effect_heatmap(data: dict[str, Any], output: Path) -> None:
    c2f = [item for item in data["candidates"] if item["mode"] == "c2f"]
    datasets = sorted({row["dataset_key"] for row in data["scorecard"]}, key=lambda key: next(int(row["dataset_order"]) for row in data["scorecard"] if row["dataset_key"] == key))
    matrix = np.full((len(c2f), len(datasets)), np.nan)
    status: list[list[str]] = [["NOT_RUN" for _ in datasets] for _ in c2f]
    for i, candidate in enumerate(c2f):
        for j, dataset in enumerate(datasets):
            row = data["pair_by_label_dataset"][(candidate["label"], dataset)]
            status[i][j] = row["comparison_status"]
            if row["comparison_status"] == "COMPARABLE":
                matrix[i, j] = float(row["percent_delta_vs_better_direct_parent"])
    finite = matrix[np.isfinite(matrix)]
    raw_bound = max(10.0, float(np.max(np.abs(finite)))) if finite.size else 10.0
    # One ResNet outlier is +211.2%.  Clip only the colour scale, not the
    # annotation, so the rest of the matrix remains visually informative.
    bound = min(raw_bound, 50.0)
    fig, ax = plt.subplots(figsize=(13.0, 3.9), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdYlGn_r", vmin=-bound, vmax=bound, aspect="auto")
    clip_note = " (colour clipped at ±50%; labels exact)" if raw_bound > bound else ""
    ax.set_title(f"{data['architecture'].upper()}: C2F delta relative to the better direct parent{clip_note}")
    ax.set_xlabel("Full sequence")
    ax.set_ylabel("Focused C2F configuration")
    ax.set_xticks(range(len(datasets)), [item.replace("_long_office_household", "") for item in datasets], rotation=27, ha="right", fontsize=8)
    ax.set_yticks(range(len(c2f)), [compact_label(item["label"]) for item in c2f], fontsize=9)
    for i in range(len(c2f)):
        for j in range(len(datasets)):
            if not math.isfinite(matrix[i, j]):
                text, color = "FAIL" if status[i][j] == "C2F_NONPASS" else "N/A", "#9B1C31"
            else:
                value = matrix[i, j]
                text = f"{value:+.1f}%"
                color = "white" if abs(value) > bound * 0.60 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.1, color=color, fontweight="bold" if text == "FAIL" else "normal")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.90)
    colorbar.set_label("Percent delta: negative = C2F lower/better")
    fig.savefig(output, dpi=230)
    plt.close(fig)


def make_win_rate_plot(unet: dict[str, Any], resnet: dict[str, Any], output: Path) -> None:
    records: list[tuple[str, float, int, int, str]] = []
    for data, color in ((unet, "#3274A1"), (resnet, "#E1812C")):
        for row in data["effects"]:
            wins = int(row["beats_better_direct_parent_count"])
            comparable = int(row["comparable_parent_pairs"])
            records.append((compact_label(row["c2f_label"]), 100.0 * wins / comparable, wins, comparable, color))
    fig, ax = plt.subplots(figsize=(10.8, 4.6), constrained_layout=True)
    y = np.arange(len(records))
    bars = ax.barh(y, [item[1] for item in records], color=[item[4] for item in records])
    ax.set_yticks(y, [item[0] for item in records])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Win rate against the better direct parent (%)")
    ax.set_title("C2F robustness across comparable full-sequence parent pairs")
    ax.grid(axis="x", alpha=0.24)
    for bar, item in zip(bars, records):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2, f"{item[2]}/{item[3]}", va="center", fontsize=9)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#3274A1", label="U-Net"), plt.Rectangle((0, 0), 1, 1, color="#E1812C", label="ResNet")], loc="lower right")
    fig.savefig(output, dpi=230)
    plt.close(fig)


def score_cell(row: dict[str, str]) -> str:
    return cm(row["historical_evo_ape_mean_cm"]) if row["status"] == "PASS" else "FAIL (NaN)"


def architecture_completion(data: dict[str, Any]) -> dict[str, int]:
    return Counter(row["status"] for row in data["scorecard"])


def best_row(data: dict[str, Any], dataset: str) -> dict[str, str] | None:
    rows = [row for row in data["scorecard"] if row["dataset_key"] == dataset and row["status"] == "PASS" and row["historical_evo_ape_mean_cm"]]
    return min(rows, key=lambda row: float(row["historical_evo_ape_mean_cm"])) if rows else None


def main() -> None:
    unet = load_architecture("unet")
    resnet = load_architecture("resnet")
    datasets = json.loads(DATASET_PLAN.read_text(encoding="utf-8"))["datasets"]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    unet_heatmap = RESULT_ROOT / "unet_c2f_parent_delta_heatmap.png"
    resnet_heatmap = RESULT_ROOT / "resnet_c2f_parent_delta_heatmap.png"
    win_rate_plot = RESULT_ROOT / "c2f_parent_win_rates.png"
    make_effect_heatmap(unet, unet_heatmap)
    make_effect_heatmap(resnet, resnet_heatmap)
    make_win_rate_plot(unet, resnet, win_rate_plot)

    unet_counts = architecture_completion(unet)
    resnet_counts = architecture_completion(resnet)
    u_effect = {row["c2f_label"]: row for row in unet["effects"]}
    r_effect = {row["c2f_label"]: row for row in resnet["effects"]}
    u_a_wins = sum(int(row["beats_better_direct_parent_count"]) for row in unet["effects"] if row["variant"] == "A")
    u_a_pairs = sum(int(row["comparable_parent_pairs"]) for row in unet["effects"] if row["variant"] == "A")
    u_b_wins = sum(int(row["beats_better_direct_parent_count"]) for row in unet["effects"] if row["variant"] == "B")
    u_b_pairs = sum(int(row["comparable_parent_pairs"]) for row in unet["effects"] if row["variant"] == "B")
    r_wins = sum(int(row["beats_better_direct_parent_count"]) for row in resnet["effects"])
    r_pairs = sum(int(row["comparable_parent_pairs"]) for row in resnet["effects"])

    lines: list[str] = []
    add = lines.append
    add("# C2F 九数据集 Direct-parent 对照评估（U-Net 与 ResNet）")
    add("")
    add("## 1. 结论摘要")
    add("")
    add(
        "这不是新的 channel search，而是对已经在 fr1/desk_lightswitch 上选出的 direct subsets 与重点 C2F pair 的外部验证。"
        "每个 C2F 结果只和**同一数据集**上的两个 direct parents 比较；因此结论来自配对增益/退化，"
        "不把 fr1、fr2、fr3 的原始 ATE 直接平均。"
    )
    add("")
    add(
        f"**U-Net 的 C2F-A 显示出稳定的正贡献。** 36 个 U-Net C2F cell 全部完成；三个 C2F-A pair 在 27 个可比较 parent pairs 中有 **{u_a_wins}/27** 次优于更强 direct parent（{100*u_a_wins/u_a_pairs:.1f}%）。"
        f"其中 U-A F2+C4（fr1 lightswitch C2F 全局最优）和 U-A F5+C4（原先的强 synergy case）均为 **8/9**，而 U-A F1+C4 也为 **7/9**。"
    )
    add("")
    add(
        f"**U-Net 的 variant B 不具同等稳健性。** U-B F2+C2 只有 **{u_b_wins}/{u_b_pairs}** 次优于更强 parent，"
        "说明 multi-scale channel complementarity 不仅取决于通道，还取决于 coarse/fine 在 pyramid 中的分配方式。"
    )
    add("")
    add(
        f"**ResNet 的 fr1-lightswitch C2F 收益没有稳定迁移。** ResNet 仅有 **{r_wins}/{r_pairs}** 个可比较 C2F cell 优于更强 parent。"
        "最初在 fr1 lightswitch 表现最好的 B F2+C4 虽在 fr1/fr2 lightswitch 仍优于 direct F2，但跨九序列只有 3/9 胜出，"
        "在 clean、flashlight 和 fr3 条件中多次退化。"
    )
    add("")
    add(
        "在本次各自 10 个重点配置的比较集合中，U-Net 的最优值在 9 个序列中有 8 个低于 ResNet 的最优值；"
        "唯一例外为 fr1 flashlight。该比较支持 U-Net C2F-A 作为当前更有希望进行后续多序列扩展验证的路线，"
        "但不等同于所有 U-Net/ResNet 配置之间的穷尽架构比较。"
    )
    add("")
    add("## 2. 实验目的与固定协议")
    add("")
    add(markdown_table(
        ["项目", "固定设置"],
        [
            ["问题", "C2F 是否能在同一完整序列上优于构成它的 direct fine/coarse parents？"],
            ["数据", "fr1/desk、fr2/desk、fr3/long_office_household × clean / lightswitch / flashlight = 9 条完整 TUM 序列"],
            ["每架构配置", "10：gray baseline + 5 个 direct parents + 4 个重点 C2F cases；每格 1 次"],
            ["Mapping", "固定 gray + sensor/GT depth；只改变 tracking feature configuration"],
            ["C2F-A", "coarse at L0/L1，fine at L2"],
            ["C2F-B", "coarse at L0，fine at L1/L2"],
            ["主指标", "historical keyframe evo_ape translation ATE mean（--align --correct_scale），单位 cm"],
            ["完整性", "500 s timeout；coverage >= 90% 且轨迹达到结尾；保留 all-frame SE(3)、RPE、诊断日志"],
        ],
    ))
    add("")
    add("## 3. 完成度与失败")
    add("")
    add(markdown_table(
        ["架构", "计划 cells", "PASS", "FAIL_TRACKING_NAN", "C2F PASS", "Direct PASS", "Gray PASS"],
        [
            ["U-Net", 90, unet_counts["PASS"], unet_counts["FAIL_TRACKING_NAN"], "36/36", "45/45", "7/9"],
            ["ResNet", 90, resnet_counts["PASS"], resnet_counts["FAIL_TRACKING_NAN"], "33/36", "43/45", "7/9"],
        ],
    ))
    add("")
    add(
        "两架构中 gray 均在 fr1 lightswitch 与 fr3 lightswitch 出现非有限 affine/pose diagnostics，因此没有作为这些序列的竞争性可行方案。"
        "ResNet 另有 2 个 direct F1 和 3 个 C2F 运行失败，均发生在 lightswitch 条件；U-Net 所有 selected direct/C2F 组合均完成。"
    )
    add("")
    add("## 4. C2F 的配对效应：核心证据")
    add("")
    add("下图显示每个 C2F 相对同一数据集上**更强的 direct parent**的 ATE 百分比变化：负值（绿色）表示 C2F 更好；红色表示退化。")
    add("")
    add(f"![U-Net C2F parent delta heatmap]({unet_heatmap.name})")
    add("")
    add(f"![ResNet C2F parent delta heatmap]({resnet_heatmap.name})")
    add("")
    add("ResNet 热图的颜色在 ±50% 处截断以保证中等退化仍可辨认；单元格数字始终为未截断的精确百分比。")
    add("")
    add(f"![C2F parent win rates]({win_rate_plot.name})")
    add("")
    effect_rows: list[list[object]] = []
    for data in (unet, resnet):
        for row in data["effects"]:
            effect_rows.append([
                compact_label(row["c2f_label"]),
                row["variant"],
                f"{row['c2f_pass_count']}/9",
                row["comparable_parent_pairs"],
                f"{row['beats_fine_parent_count']}/{row['comparable_parent_pairs']}",
                f"{row['beats_better_direct_parent_count']}/{row['comparable_parent_pairs']}",
                f"{float(row['median_percent_delta_vs_better_parent']):+.2f}%" if row["median_percent_delta_vs_better_parent"] else "",
            ])
    add(markdown_table(
        ["C2F", "Variant", "C2F PASS", "可比较 pairs", "优于 fine", "优于更强 parent", "median Δ vs stronger parent"],
        effect_rows,
    ))
    add("")
    add("### 4.1 U-Net：C2F-A 的增益模式")
    add("")
    add(
        "U-A F2+C4 是最均衡的配置：在 fr1 lightswitch 保留 5.8765 cm 的原始最优结果，并在 clean/flashlight/fr2/fr3 中保持 8/9 次配对胜出。"
        "U-A F5+C4 同样 8/9 胜出，尤其在 fr2 lightswitch 从 direct F5 的 11.8873 cm 降至 3.9342 cm（相对更强 parent 改善 30.8%）。"
        "U-A F1+C4 的命名来自 fr1 lightswitch 上曾劣于 global direct F1 的负对照，但跨九序列反而达到 7/9 胜出，说明该单点负结果并非普遍规律。"
    )
    add("")
    add("### 4.2 ResNet：局部 lightswitch 收益与跨分布退化")
    add("")
    add(
        "R-B F2+C4 在 fr1 和 fr2 lightswitch 分别比 direct F2 低 10.8% 和 11.6%，但在 fr1 clean、fr1 flashlight、fr3 clean、fr3 lightswitch、fr3 flashlight 都更差；"
        "其 median Δ 为 +4.30%，因此不能作为跨条件的默认 C2F 配置。"
        "R-A F6+C5 曾在 fr1 lightswitch 改善 32.9%，但只在 4/8 可比较序列胜出，并在 fr3 flash/clean 显著退化；它更适合作为“C2F 可产生强局部协同”的机制案例，而不是稳健推荐。"
    )
    add("")
    add("## 5. 每个数据集的最优结果（重点比较集合内）")
    add("")
    winner_rows: list[list[object]] = []
    for dataset in datasets:
        key = dataset["key"]
        u = best_row(unet, key)
        r = best_row(resnet, key)
        assert u is not None and r is not None
        lower = "U-Net" if float(u["historical_evo_ape_mean_cm"]) < float(r["historical_evo_ape_mean_cm"]) else "ResNet"
        winner_rows.append([
            key,
            compact_label(u["label"]), cm(u["historical_evo_ape_mean_cm"]),
            compact_label(r["label"]), cm(r["historical_evo_ape_mean_cm"]), lower,
        ])
    add(markdown_table(["数据集", "U-Net 最优", "ATE", "ResNet 最优", "ATE", "较低者"], winner_rows))
    add("")
    add("## 6. 完整 ATE 结果")
    add("")
    add("单位：cm；为主指标 historical keyframe ATE mean。每张表只比较一个 dataset family 下的三种光照条件，避免九列表在 Word 中不可读。")
    add("")
    for title, data in (("U-Net", unet), ("ResNet", resnet)):
        add(f"### {title}")
        add("")
        for family in ("fr1_desk", "fr2_desk", "fr3_long_office_household"):
            family_items = [item for item in datasets if item["family"] == family]
            rows: list[list[object]] = []
            for candidate in data["candidates"]:
                values = [score_cell(data["score_by_label_dataset"][(candidate["label"], item["key"])]) for item in family_items]
                rows.append([compact_label(candidate["label"]), config_text(candidate), *values])
            add(f"#### {family}")
            add("")
            add(markdown_table(["配置", "Tracking feature", *[item["condition"] for item in family_items]], rows))
            add("")
    add("## 7. 解释与下一步")
    add("")
    add("1. **C2F 的作用是条件性的 feature complementarity，而非 guaranteed improvement。** U-Net 证明 C2F-A 能在多场景条件下稳定地给 shallow Enc0 增添 coarse context；ResNet 则说明在某个 MVS/lighting episode 上的最优配对并不足以保证跨分布迁移。")
    add("2. **Variant selection 是模型设计的一部分。** 同为 U-Net F2，A+C4 为 8/9，B+C2 为 4/9；ResNet F2+C4 中 B 在 fr1 lightswitch 上显著优于 A，但在多序列总体仍不足。故以后不能只报告“用了 C2F”，必须固定并报告 pyramid routing variant。")
    add("3. **后续推荐。** 若目标是可泛化 C2F，优先以 U-Net C2F-A（特别是 F2+C4 和 F5+C4）进入新的序列/退化验证；ResNet B F2+C4 可保留为 lightswitch-specialist 对照，R-A F6+C5 可作为机制案例，不应直接作为 default configuration。")
    add("4. **报告呈现。** 最终主表应同时列出 direct fine parent、direct coarse parent、C2F A/B 和它们的 sequence-wise delta；不要只展示 fr1 lightswitch 的 global best，必须保留 ResNet 的 clean/flashlight 退化与 U-Net B 的负/近零案例。")
    add("")
    add("## 8. 局限性")
    add("")
    add("- 每个 configuration×sequence 只运行 1 次；虽然之前多次重复显示较稳定，这里不能估计运行间方差。")
    add("- 重点集合是由 fr1/desk_lightswitch 的 direct greedy 与 C2F grid 预先筛选出来的，因此该序列上的结果不能视为完全独立 test。真正的泛化证据来自余下 8 条序列及其配对趋势。")
    add("- 本文的跨架构‘较低者’仅在已选的 10 个重点配置集合内成立；它不是全通道模型或所有候选配置的穷尽比较。")
    add("- 主指标含 trajectory alignment 与 scale correction，以保持与现有项目评估一致；all-frame metric-scale 指标和 diagnostics 已保留在原始 SQLite/CSV 中，应在选定最终配置后一并复核。")

    markdown_path = RESULT_ROOT / "C2F_九数据集Direct_Parent对照评估_中文.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("pandoc is required to create the Word report")
    docx_path = RESULT_ROOT / DOCX_NAME
    subprocess.run(
        [
            pandoc, "--from", "markdown", "--to", "docx", "--standalone",
            "--resource-path", str(RESULT_ROOT), "--output", str(docx_path), str(markdown_path),
        ],
        check=True,
    )
    print(f"[WRITE] {unet_heatmap}")
    print(f"[WRITE] {resnet_heatmap}")
    print(f"[WRITE] {win_rate_plot}")
    print(f"[WRITE] {markdown_path}")
    print(f"[WRITE] {docx_path}")


if __name__ == "__main__":
    main()
