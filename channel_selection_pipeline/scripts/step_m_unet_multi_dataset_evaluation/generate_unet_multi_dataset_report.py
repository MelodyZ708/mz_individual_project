#!/usr/bin/env python3
"""Create the Chinese Word report for the U-Net 3×3 evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_unet_multi_dataset_report")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESULT_ROOT = PROJECT_ROOT / "channel_selection_results/step_m_unet_multi_dataset_evaluation"
REPORT_ROOT = PROJECT_ROOT / "channel_selection_results/reports/unet_multi_dataset_evaluation"
DATASET_PLAN = SCRIPT_DIR / "unet_dataset_plan.json"
CANDIDATE_PLAN = SCRIPT_DIR / "unet_candidate_plan.json"
DOCX_NAME = "UNet_Enc0_Enc1_三数据集三光照条件验证结果_中文.docx"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def cm(value: str | None) -> str:
    return "" if value in (None, "") else f"{float(value):.2f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def short_id(label: str) -> str:
    mapping = {
        "enc0_k01_single_d03": "E0-1",
        "enc0_k02_d02_d14": "E0-2",
        "enc0_global_rank02_k03_d03_d07_d12": "E0-3 (R2)",
        "enc0_global_rank03_k04_d02_d03_d12_d14": "E0-4 (R3)",
        "enc0_global_rank01_k06_d02_d03_d07_d12_d13_d14": "E0-6 (R1)",
        "enc0_all16": "E0-All16",
        "enc0_bqs_top5_d00_d03_d10_d14_d15": "E0-BQS5",
        "enc1_k02_d00_d05": "E1-2",
        "enc1_global_rank03_k04_d00_d05_d18_d30": "E1-4 (R3)",
        "enc1_global_rank01_k06_d05_d06_d17_d18_d28_d30": "E1-6 (R1)",
        "enc1_global_rank02_k06_d00_d05_d06_d17_d18_d30": "E1-6 (R2)",
        "enc1_all32": "E1-All32",
        "enc1_bqs_top5_d04_d09_d10_d15_d30": "E1-BQS5",
    }
    return mapping[label]


def channel_text(candidate: dict[str, object]) -> str:
    return "[" + ", ".join(str(item) for item in candidate["channels"]) + "]"


def score_cell(row: dict[str, str]) -> str:
    if row["status"] == "PASS":
        return cm(row["historical_evo_ape_mean_cm"])
    if row["status"] == "SKIPPED_BY_SAFETY":
        return "安全跳过"
    return "FAIL (NaN)"


def make_figures(
    candidates: list[dict[str, object]],
    scorecard: list[dict[str, str]],
    summaries: list[dict[str, str]],
) -> tuple[Path, Path]:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_by_key = {row["candidate_key"]: row for row in summaries}
    ranking = sorted(
        candidates,
        key=lambda item: float(summary_by_key[str(item["candidate_key"])]["mean_dataset_rank_on_passes"]),
    )
    labels = [short_id(str(item["label"])) for item in ranking]
    scores = [float(summary_by_key[str(item["candidate_key"])]["mean_dataset_rank_on_passes"]) for item in ranking]
    colors = ["#4c78a8" if int(item["enc_level"]) == 0 else "#f58518" for item in ranking]
    fig, ax = plt.subplots(figsize=(9.2, 6.1))
    bars = ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel("Mean within-dataset rank (lower is better)")
    ax.set_title("U-Net multi-sequence robustness ranking")
    ax.set_xlim(0, max(scores) + 1.4)
    for bar, item in zip(bars, ranking[::-1]):
        record = summary_by_key[str(item["candidate_key"])]
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2, f"wins={record['datasets_won']}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    bar_path = REPORT_ROOT / "unet_mean_rank_bar.png"
    fig.savefig(bar_path, dpi=180)
    plt.close(fig)

    dataset_keys: list[str] = []
    for row in scorecard:
        if row["dataset_key"] not in dataset_keys:
            dataset_keys.append(row["dataset_key"])
    matrix = np.full((len(candidates), len(dataset_keys)), np.nan)
    text = np.full(matrix.shape, "", dtype=object)
    candidate_index = {str(item["candidate_key"]): index for index, item in enumerate(candidates)}
    dataset_index = {key: index for index, key in enumerate(dataset_keys)}
    for row in scorecard:
        y = candidate_index[row["candidate_key"]]
        x = dataset_index[row["dataset_key"]]
        if row["status"] == "PASS":
            matrix[y, x] = float(row["dataset_rank"])
            text[y, x] = str(row["dataset_rank"])
        elif row["status"] == "SKIPPED_BY_SAFETY":
            text[y, x] = "S"
        else:
            text[y, x] = "F"
    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    cmap = plt.cm.YlGn_r.copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(matrix, cmap=cmap, vmin=1, vmax=13, aspect="auto")
    ax.set_xticks(range(9), ["F1-C", "F1-L", "F1-F", "F2-C", "F2-L", "F2-F", "F3-C", "F3-L", "F3-F"])
    ax.set_yticks(range(len(candidates)), [short_id(str(item["label"])) for item in candidates])
    ax.set_title("Within-dataset rank (1=best; F=failure; S=safety skip)")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, text[y, x], ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.88, label="Rank")
    fig.tight_layout()
    heatmap_path = REPORT_ROOT / "unet_within_dataset_rank_heatmap.png"
    fig.savefig(heatmap_path, dpi=180)
    plt.close(fig)
    return bar_path, heatmap_path


def main() -> None:
    datasets = json.loads(DATASET_PLAN.read_text(encoding="utf-8"))["datasets"]
    candidates = json.loads(CANDIDATE_PLAN.read_text(encoding="utf-8"))["candidates"]
    scorecard = read_csv(RESULT_ROOT / "dataset_scorecard.csv")
    summaries = read_csv(RESULT_ROOT / "candidate_robustness_summary.csv")
    status_counts = Counter(row["status"] for row in scorecard)
    if len(scorecard) != 117 or status_counts["PASS"] != 110 or status_counts["FAIL_TRACKING_NAN"] != 5 or status_counts["SKIPPED_BY_SAFETY"] != 2:
        raise ValueError(f"Unexpected incomplete result set: {dict(status_counts)}")
    candidate_by_key = {str(item["candidate_key"]): item for item in candidates}
    result = {(row["dataset_key"], row["candidate_key"]): row for row in scorecard}
    bar_path, heatmap_path = make_figures(candidates, scorecard, summaries)

    lines: list[str] = []
    add = lines.append
    add("# U-Net Enc0/Enc1 三数据集 × 三光照条件验证结果")
    add("")
    add("## 1. 实验目的")
    add("")
    add("本实验检验基于 fr1/desk lightswitch 直接 greedy 搜索得到的 U-Net Enc0 与 Enc1 通道组合，是否能在不同场景长度、环境与光照条件下保持有效。跨序列比较采用序列内排名与成功率；不同序列之间的原始 ATE 不直接平均。")
    add("")
    add("## 2. 实验设置与完成情况")
    add("")
    lines.extend([
        "- 数据：fr1/desk、fr2/desk、fr3/long_office_household，各自包含 clean、lightswitch、flashlight，共 9 个全序列。",
        "- 配置：13 个 U-Net 通道配置（Enc0 7 个、Enc1 6 个），每个 active cell 运行 1 次。",
        "- Tracking：所选 Enc0/Enc1 post-LeakyReLU activation channels；Mapping：gray + sensor depth，保持不变。",
        "- 主指标：historical keyframe `evo_ape --align --correct_scale` translation ATE mean（cm）。同时保留 RPE、全帧 metric-scale SE(3) ATE/RPE、coverage 与数值诊断。",
        "- 单次 timeout：500 s；完成性阈值：coverage ≥ 90%。",
        "- 安全调整：fr3 lightswitch 的 Enc0-all16 启动后发生 NVIDIA Xid 79 / PCIe receiver error。因此仅在该条件跳过 Enc0-all16 与 Enc1-all32；其他 8 个条件仍已评估。安全跳过不计为算法失败。",
        "- 完成统计：**110 PASS / 115 active cells（95.7%）**，5 个 `FAIL_TRACKING_NAN`，2 个 `SKIPPED_BY_SAFETY`。",
    ])
    add("")
    add("## 3. 被评估配置")
    add("")
    config_rows = [[short_id(str(item["label"])), f"Enc{item['enc_level']}", channel_text(item), str(item["role"])] for item in candidates]
    add(markdown_table(["简称", "层", "通道", "在 fr1 lightswitch 搜索中的角色"], config_rows))
    add("")
    add("R1/R2/R3 指同层 direct greedy 最终重复评估的第 1/2/3 名；All 与 BQS 是对照。")
    add("")
    add("## 4. 各数据集最优配置")
    add("")
    winner_rows = []
    for dataset in datasets:
        group = [row for row in scorecard if row["dataset_key"] == dataset["key"] and row["status"] == "PASS"]
        winner = min(group, key=lambda row: float(row["historical_evo_ape_mean_cm"]))
        candidate = candidate_by_key[winner["candidate_key"]]
        winner_rows.append([dataset["key"], dataset["condition"], short_id(str(candidate["label"])), channel_text(candidate), cm(winner["historical_evo_ape_mean_cm"])])
    add(markdown_table(["数据集", "条件", "最优配置", "通道", "ATE mean (cm)"], winner_rows))
    add("")
    add("Enc0 配置赢得 9 个数据集中的 8 个。唯一例外是 fr3 lightswitch：Enc1 的 E1-6 (R2) 获胜，说明深一层特征对长序列光照切换具有互补价值。")
    add("")
    add("## 5. 完整 ATE 结果")
    add("")
    add("单位：cm；数值为 historical keyframe evo ATE mean，越低越好。`FAIL (NaN)` 为非有限 affine/pose diagnostics；`安全跳过` 是前述 Xid 79 相关的明确安全排除。")
    add("")
    for family in ("fr1_desk", "fr2_desk", "fr3_long_office_household"):
        family_datasets = {item["condition"]: item["key"] for item in datasets if item["family"] == family}
        table_rows = []
        for candidate in candidates:
            key = str(candidate["candidate_key"])
            table_rows.append([
                short_id(str(candidate["label"])),
                channel_text(candidate),
                score_cell(result[(family_datasets["clean"], key)]),
                score_cell(result[(family_datasets["lightswitch"], key)]),
                score_cell(result[(family_datasets["flashlight"], key)]),
            ])
        add(f"### {family}")
        add("")
        add(markdown_table(["配置", "通道", "Clean", "Lightswitch", "Flashlight"], table_rows))
        add("")
    add("## 6. 跨序列稳健性：序列内排名")
    add("")
    add("下图及表格将每个数据集内的 ATE 排名作为单位（1 = 最优），然后对成功的序列取平均。该处理避免了不同场景尺度和轨迹长度使绝对 ATE 不可直接比较的问题。")
    add("")
    add(f"![平均序列内排名]({bar_path.name})")
    add("")
    add(f"![各序列内排名矩阵]({heatmap_path.name})")
    add("")
    robustness_rows = []
    for row in summaries:
        candidate = candidate_by_key[row["candidate_key"]]
        robustness_rows.append([
            short_id(str(candidate["label"])), channel_text(candidate), f"{row['pass_count']}/9",
            row["safety_skipped_count"], f"{float(row['mean_dataset_rank_on_passes']):.2f}", row["datasets_won"],
        ])
    add(markdown_table(["配置", "通道", "PASS", "安全跳过", "平均 rank", "获胜数"], robustness_rows))
    add("")
    add("## 7. 失败与最小可行配置")
    add("")
    failures = []
    for row in scorecard:
        if row["status"] == "FAIL_TRACKING_NAN":
            candidate = candidate_by_key[row["candidate_key"]]
            failures.append([row["dataset_key"], short_id(str(candidate["label"])), channel_text(candidate), "非有限 affine/pose diagnostics"])
    add(markdown_table(["数据集", "配置", "通道", "原因"], failures))
    add("")
    add("E0-1 `[3]` 仅 5/9 通过，因此单通道配置不具跨光照鲁棒性。E0-2 `[2,14]` 虽在 fr1 clean 获胜（5.27 cm），但在 fr2 lightswitch 失效。相对地，E1-2 `[0,5]` 以仅两个通道实现 9/9 通过，是最小但稳定的 Enc1 参考点；其精度仍低于最佳组合。")
    add("")
    add("## 8. 主要结论与解读")
    add("")
    conclusions = [
        "**E0-4 (R3) `[2,3,12,14]` 是最稳健的总体选择。** 它 9/9 通过，平均 rank **2.56**（所有配置最低），并在 fr1 flashlight、fr2 lightswitch 获胜。它比 E0-6 (R1) 更紧凑，且跨条件波动更小。",
        "**E0-6 (R1) `[2,3,7,12,13,14]` 是 accuracy-first 的强候选。** 它赢得 fr1 lightswitch、fr3 clean 和 fr3 flashlight 共 3 个数据集，平均 rank 为 3.33；但并非所有条件最优。",
        "**E0-3 (R2) `[3,7,12]` 在 fr2 上尤其有效。** 它在 fr2 clean 与 flashlight 获胜（3.03、3.07 cm），表明三通道已可覆盖部分场景的有效信息；但 fr1 clean 的表现较弱，泛化不如 E0-4。",
        "**Enc1 的跨序列最佳并非原始搜索的 R1，而是 R2。** E1-6 (R2) `[0,5,6,17,18,30]` 的平均 rank **6.89**，优于 Enc1 R1 的 8.11，且在 fr3 lightswitch 获胜（11.30 cm）。这显示 Enc1 更受序列分布影响。",
        "**全通道与 BQS 对照均不占优。** E0-All16 的平均 rank 为 6.00（8 个可比较条件），E1-All32 为 8.38；两层 BQS5 分别为 6.56 与 11.11。直接 greedy 选择显著优于“更多通道”或历史 BQS 选择。",
        "**建议。** 以 E0-4 作为通用、紧凑的主推荐；E0-6 (R1) 作为 accuracy-first 备选；E1-6 (R2) 作为长序列 lightswitch 的互补配置。",
    ]
    lines.extend(f"{index}. {text}" for index, text in enumerate(conclusions, start=1))
    add("")
    add("## 9. 局限性")
    add("")
    lines.extend([
        "- 每格只运行 1 次；本实验依赖先前观察到的运行稳定性，不能量化 run-to-run 方差。",
        "- 主指标使用 `--align --correct_scale` 的 historical keyframe ATE，以保持与既有实验一致；它不能替代未对齐或 metric-scale 的绝对部署误差。",
        "- fr3 lightswitch 的两个 all-channel controls 因 GPU/PCIe Xid 79 关联事件安全跳过，故其 8/9 覆盖不应与完整 9/9 的配置作严格同等比较。",
        "- 本实验只覆盖三类 TUM 场景族与三种光照条件；结论支持这些已评估条件下的稳健性，不足以证明对所有退化或新场景的普遍最优性。",
    ])
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    markdown_path = REPORT_ROOT / "UNet_Enc0_Enc1_三数据集三光照条件验证结果_中文.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("pandoc is required to create the Word report")
    docx_path = REPORT_ROOT / DOCX_NAME
    subprocess.run(
        [
            pandoc,
            "--from",
            "markdown",
            "--to",
            "docx",
            "--standalone",
            "--resource-path",
            str(REPORT_ROOT),
            "--output",
            str(docx_path),
            str(markdown_path),
        ],
        check=True,
    )
    print(f"[WRITE] {markdown_path}")
    print(f"[WRITE] {bar_path}")
    print(f"[WRITE] {heatmap_path}")
    print(f"[WRITE] {docx_path}")


if __name__ == "__main__":
    main()
