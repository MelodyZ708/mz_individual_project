#!/usr/bin/env python3
"""Generate the Chinese QueensCAMP degradation robustness report assets.

The authoritative inputs are the completed Step-H aggregate CSV files. This
script only creates a report-local Markdown document and figures; it never
touches evaluator databases or any COMO configuration.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

_CACHE = Path(tempfile.gettempdir()) / "mz_queenscamp_report_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parents[2]
RESULT_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_h_queenscamp_degradation_evaluation/"
    "three_repeats"
)
CANDIDATE_PLAN = (
    PROJECT_ROOT
    / "channel_selection_pipeline/scripts/step_h_queenscamp_degradation_evaluation/"
    "top7_plus_gray_candidate_plan.json"
)
SUMMARY_CSV = RESULT_DIR / "per_dataset_configuration_summary.csv"
OVERALL_CSV = RESULT_DIR / "configuration_overall_summary.csv"
PROTOCOL_JSON = RESULT_DIR / "aggregate_protocol.json"
OUTPUT_MD = REPORT_DIR / "QueensCAMP_七种退化下Top7与Gray鲁棒性评估_中文.md"
BASELINE = "5,29,40,52"
GRAY = "gray"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def display_configuration(key: str) -> str:
    return "gray" if key == GRAY else f"[{key}]"


def percent_improvement(ratio: float | None) -> str:
    if ratio is None:
        return "—"
    direction = "改善" if ratio <= 1.0 else "劣于"
    return f"{direction}{abs(1.0 - ratio) * 100.0:.1f}%"


def safe_min(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(
        rows,
        key=lambda row: (
            -int(row["pass_count"]),
            as_float(row["historical_ate_mean_cm_mean"])
            if as_float(row["historical_ate_mean_cm_mean"]) is not None
            else math.inf,
            int(row["source_rank"]),
        ),
    )


def save_ate_heatmap(
    candidates: list[dict[str, Any]],
    datasets: list[str],
    lookup: dict[tuple[str, str], dict[str, str]],
    winners: dict[str, str],
) -> None:
    matrix = np.asarray(
        [
            [
                as_float(lookup[(candidate["candidate_key"], degradation)]["historical_ate_mean_cm_mean"])
                for degradation in datasets
            ]
            for candidate in candidates
        ],
        dtype=float,
    )
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#d9d9d9")
    figure, axis = plt.subplots(figsize=(13.4, 6.4))
    image = axis.imshow(matrix, aspect="auto", cmap=cmap)
    axis.set_xticks(range(len(datasets)), datasets, rotation=24, ha="right")
    axis.set_yticks(
        range(len(candidates)),
        [display_configuration(item["candidate_key"]) for item in candidates],
    )
    axis.set_title("QueensCAMP seven-degradation historical keyframe ATE mean (cm)")
    for row, candidate in enumerate(candidates):
        key = candidate["candidate_key"]
        for column, degradation in enumerate(datasets):
            value = matrix[row, column]
            is_winner = winners[degradation] == key
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold" if is_winner else "normal",
                color="white" if value > np.nanmedian(matrix) else "black",
            )
            if is_winner:
                axis.add_patch(
                    plt.Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="white" if value > np.nanmedian(matrix) else "black",
                        linewidth=2.0,
                    )
                )
    figure.colorbar(image, ax=axis, shrink=0.82, label="ATE mean (cm; lower is better)")
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "queenscamp_ate_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_relative_ratio_heatmap(
    candidates: list[dict[str, Any]],
    datasets: list[str],
    lookup: dict[tuple[str, str], dict[str, str]],
) -> None:
    matrix = np.asarray(
        [
            [
                as_float(
                    lookup[(candidate["candidate_key"], degradation)][
                        "mean_ate_ratio_to_historical_cnn_baseline"
                    ]
                )
                for degradation in datasets
            ]
            for candidate in candidates
        ],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(13.4, 6.4))
    image = axis.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.3, vmax=1.6)
    axis.set_xticks(range(len(datasets)), datasets, rotation=24, ha="right")
    axis.set_yticks(
        range(len(candidates)),
        [display_configuration(item["candidate_key"]) for item in candidates],
    )
    axis.set_title("ATE ratio to historical CNN baseline [5,29,40,52] (lower is better)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold" if value < 1.0 else "normal",
            )
    figure.colorbar(image, ax=axis, shrink=0.82, label="candidate ATE / historical-CNN ATE")
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "queenscamp_relative_ate_ratio.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def table_line(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    for path in (SUMMARY_CSV, OVERALL_CSV, PROTOCOL_JSON, CANDIDATE_PLAN):
        if not path.is_file():
            raise FileNotFoundError(path)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_JSON.read_text(encoding="utf-8"))
    if protocol.get("completed_runs") != 168 or protocol.get("pass_runs") != 168:
        raise ValueError("The Step-H result set is not a complete 168/168 PASS run")
    candidates = json.loads(CANDIDATE_PLAN.read_text(encoding="utf-8"))["candidates"]
    summary_rows = read_csv(SUMMARY_CSV)
    overall_rows = read_csv(OVERALL_CSV)
    if len(candidates) != 8 or len(summary_rows) != 56 or len(overall_rows) != 8:
        raise ValueError("Unexpected QueensCAMP report input dimensions")
    if any(int(row["pass_count"]) != 3 for row in summary_rows):
        raise ValueError("Expected every dataset/configuration cell to be 3/3 PASS")
    if any(as_float(row["historical_ate_mean_cm_std"]) != 0.0 for row in summary_rows):
        raise ValueError("Expected deterministic zero ATE standard deviations")

    datasets: list[str] = []
    for row in summary_rows:
        if row["degradation"] not in datasets:
            datasets.append(row["degradation"])
    if datasets != [
        "blur",
        "condensation",
        "dirt",
        "mixed_story_v1",
        "overexposure",
        "wet",
        "underexposure",
    ]:
        raise ValueError(f"Unexpected degradation ordering: {datasets}")
    lookup = {(row["candidate_key"], row["degradation"]): row for row in summary_rows}
    if len(lookup) != 56:
        raise ValueError("Duplicate or missing dataset/configuration summary cell")
    winners = {
        degradation: safe_min(
            [row for row in summary_rows if row["degradation"] == degradation]
        )["candidate_key"]
        for degradation in datasets
    }
    save_ate_heatmap(candidates, datasets, lookup, winners)
    save_relative_ratio_heatmap(candidates, datasets, lookup)

    best_cnn = next(
        row for row in overall_rows if row["candidate_key"] == "5,6,15,35"
    )
    gray = next(row for row in overall_rows if row["candidate_key"] == GRAY)
    diverse = next(row for row in overall_rows if row["candidate_key"] == "15,17,52,59")
    historical = next(row for row in overall_rows if row["candidate_key"] == BASELINE)
    wet_winner = lookup[(winners["wet"], "wet")]
    wet_baseline = lookup[(BASELINE, "wet")]
    exposure_winner = lookup[(winners["overexposure"], "overexposure")]
    exposure_baseline = lookup[(BASELINE, "overexposure")]
    condensation_gray = lookup[(GRAY, "condensation")]
    condensation_baseline = lookup[(BASELINE, "condensation")]
    under_gray = lookup[(GRAY, "underexposure")]
    under_baseline = lookup[(BASELINE, "underexposure")]

    lines = [
        "---",
        'title: "QueensCAMP七种退化下的Top-7与Gray Baseline鲁棒性评估"',
        'subtitle: "7种退化 × 8个配置 × 3次重复（168次完整运行）"',
        'author: "MSc Project 阶段性汇报材料"',
        'date: "2026年8月12日"',
        "lang: zh-CN",
        "---",
        "",
        "# QueensCAMP七种退化下的Top-7与Gray Baseline鲁棒性评估",
        "",
        "*7种退化 × 8个配置 × 3次重复（168次完整运行）*  ",
        "MSc Project 阶段性汇报材料 · 2026年8月12日",
        "",
        "# 执行摘要",
        "",
        "本实验将fr1/desk的同一运动轨迹构造成7种QueensCAMP风格图像退化，并评估前期筛选出的6个通道组合、历史四通道CNN baseline `[5,29,40,52]`，以及gray photometric control。每个数据集/配置运行3次，共168次。Mapping端保持gray并使用配对的RGB-D sensor depth；ground-truth pose仅在运行后用于轨迹指标计算。主精度指标为与历史脚本一致的keyframe `evo_ape tum --align --correct_scale` ATE mean。",
        "",
        "核心结果如下：",
        "",
        f"1. 168/168次运行均PASS，所有56个数据集×配置单元均为3/3 PASS。因此，本批退化均未触发跟踪崩溃，可靠性只能作为共同前提，无法用于区分配置。",
        f"2. 三次重复在每个单元的主ATE完全相同（ATE std = 0.0000 cm）。这说明当前固定输入、软件与硬件路径下执行是确定性的；重复确认了结果可复现，而非估计真实随机方差。",
        f"3. gray control在7种退化上有6种优于历史CNN baseline，baseline-normalized ATE几何均值为 {float(gray['geomean_mean_ate_ratio_to_historical_baseline']):.4f}。但它是对照而非通道选择方案，不能据此直接宣称gray普遍优于CNN。",
        f"4. 仅比较CNN配置时，`[5,6,15,35]`的跨退化ATE比值几何均值最低（{float(best_cnn['geomean_mean_ate_ratio_to_historical_baseline']):.4f}），是本批次最平衡的CNN候选；`[15,17,52,59]`次之（{float(diverse['geomean_mean_ate_ratio_to_historical_baseline']):.4f}）。",
        "5. 最优CNN随退化类型变化：blur为`[1,5,24,29]`，overexposure为`[15,17,52,59]`，wet为`[5,6,15,35]`，而dirt仍由历史CNN baseline获胜。因此没有单一CNN组合在7种退化上均为最低ATE。",
        "",
        "# 1. 实验目的与设置",
        "",
        "## 1.1 目的",
        "",
        "检验从fr1/desk_lightswitch筛选出的通道组合，是否能在不同类型的图像退化下维持轨迹精度；同时以gray和历史四通道CNN baseline作为两个不同性质的对照。",
        "",
        "## 1.2 运行协议",
        "",
        table_line(["项目", "设置"]),
        table_line(["---", "---"]),
        table_line(["基础序列", "TUM fr1/desk；七个QueensCAMP风格退化版本共享同一相机运动与ground truth"]),
        table_line(["退化", "blur、condensation、dirt、mixed_story_v1、overexposure、wet、underexposure"]),
        table_line(["配置", "6个Top-7候选 + 历史CNN baseline `[5,29,40,52]` + gray control，共8个"]),
        table_line(["重复", "每个数据集/配置3次；7 × 8 × 3 = 168次"]),
        table_line(["Mapping", "固定为gray；`use_sensor_depth=true`，使用matched sensor-depth；不使用ground-truth pose建图"]),
        table_line(["Tracking", "gray control或Conv1四通道CNN；其余COMO配置固定"]),
        table_line(["主指标", "keyframe `evo_ape --align --correct_scale` ATE mean（cm，越低越好）"]),
        table_line(["诊断指标", "历史keyframe RPE、全帧SE(3) ATE/RPE、coverage、运行时间"]),
        table_line(["完成门槛", "coverage ≥90%，末帧时间间隔≤0.10 s；timeout=500 s"]),
        "",
        "不同退化的绝对ATE尺度不应直接求平均；跨退化汇总因此以每个数据集内相对于历史CNN baseline的ATE比值几何均值表示。比值小于1表示相对baseline更低的ATE。",
        "",
        "# 2. 每种退化上的最佳配置",
        "",
        "最优规则为：先比较PASS次数，再在同一PASS次数下比较ATE mean。本批所有单元均3/3 PASS，因此由ATE mean决定。",
        "",
        table_line(["退化", "最优配置", "ATE mean/cm", "相对历史CNN", "历史RPE RMSE/cm", "Trans RPE max/cm", "Rot RPE max/deg"]),
        table_line(["---", "---", "---:", "---", "---:", "---:", "---:"]),
    ]
    for degradation in datasets:
        key = winners[degradation]
        row = lookup[(key, degradation)]
        ratio = as_float(row["mean_ate_ratio_to_historical_cnn_baseline"])
        lines.append(
            table_line(
                [
                    degradation,
                    f"`{display_configuration(key)}`",
                    f"{as_float(row['historical_ate_mean_cm_mean']):.3f}",
                    percent_improvement(ratio),
                    f"{as_float(row['historical_rpe_rmse_cm_mean']):.3f}",
                    f"{as_float(row['translation_rpe_max_cm_mean']):.3f}",
                    f"{as_float(row['rotation_rpe_max_deg_mean']):.3f}",
                ]
            )
        )
    lines.extend(
        [
            "",
            "注意：ATE经全局Sim(3) alignment与scale correction后计算；较低ATE不保证没有局部跳变。因此表中同时保留RPE诊断。例如overexposure的最优`[15,17,52,59]`有最低ATE，但Trans RPE max为21.57 cm，说明局部不连续仍应在轨迹可视化中复核。",
            "",
            "# 3. 各配置在七种退化下的表现",
            "",
            "下表为每个单元的3次PASS-run ATE均值（cm）。本批每个单元均为3/3 PASS，且3次ATE相同，所以表中不再重复写`3/3`；粗体表示该退化下的最低ATE。",
            "",
            table_line(["配置"] + datasets),
            table_line(["---"] + ["---:"] * len(datasets)),
        ]
    )
    for candidate in candidates:
        key = candidate["candidate_key"]
        values = []
        for degradation in datasets:
            ate = as_float(lookup[(key, degradation)]["historical_ate_mean_cm_mean"])
            text = f"{ate:.3f}"
            if winners[degradation] == key:
                text = f"**{text}**"
            values.append(text)
        lines.append(table_line([f"`{display_configuration(key)}`"] + values))
    lines.extend(
        [
            "",
            "![图1　七种退化上的ATE均值热力图；方框/粗体表示该列最佳。](queenscamp_ate_heatmap.png){width=97%}",
            "",
            "![图2　各配置相对历史CNN baseline的ATE比值；小于1（绿色）表示改善。](queenscamp_relative_ate_ratio.png){width=97%}",
            "",
            "# 4. 配置级综合结果",
            "",
            table_line(["综合rank", "配置", "PASS/21", "3/3数据集", "优于历史CNN/7", "ATE比值几何均值", "平均数据集rank"]),
            table_line(["---:", "---", "---:", "---:", "---:", "---:", "---:"]),
        ]
    )
    for row in overall_rows:
        lines.append(
            table_line(
                [
                    row["aggregate_rank"],
                    f"`{display_configuration(row['candidate_key'])}`",
                    f"{row['total_pass_count']}/21",
                    f"{row['datasets_with_3_of_3_pass']}/7",
                    f"{row['beats_historical_baseline_dataset_means']}/7",
                    f"{float(row['geomean_mean_ate_ratio_to_historical_baseline']):.4f}",
                    f"{float(row['mean_reliability_aware_dataset_rank']):.2f}",
                ]
            )
        )
    lines.extend(
        [
            "",
            "# 5. 结果解读与insights",
            "",
            "## 5.1 所有配置均完成：本批不能用failure rate区分鲁棒性",
            "",
            "在这些七种退化强度和当前的RGB-D decoupled mapping设置下，gray和所有CNN配置均完成21/21次运行。与此前lightswitch实验不同，本批没有跟踪NaN或coverage failure。因此这里的‘鲁棒性’应理解为**在全部完成的前提下维持较低误差**，而不是failure-avoidance能力。",
            "",
            "## 5.2 gray control表现强，但不应替代通道选择结论",
            "",
            f"gray在condensation（{as_float(condensation_gray['historical_ate_mean_cm_mean']):.3f} cm，相对历史CNN {percent_improvement(as_float(condensation_gray['mean_ate_ratio_to_historical_cnn_baseline']))}）、mixed_story_v1（7.890 cm）和underexposure（{as_float(under_gray['historical_ate_mean_cm_mean']):.3f} cm，相对历史CNN {percent_improvement(as_float(under_gray['mean_ate_ratio_to_historical_cnn_baseline']))}）取得全体最佳，并在6/7种退化上优于历史CNN。它说明当前固定映射/深度条件下，某些合成外观变化并不必然使gray tracking失效。它不是channel selection候选，且结果只来自同一基础运动轨迹的合成版本，不能外推为“gray通常优于CNN”。",
            "",
            "## 5.3 CNN之间存在明确的退化类型偏好",
            "",
            f"- **Blur：** `[1,5,24,29]`为最佳（9.123 cm），比历史CNN低23.6%；gray非常接近（9.215 cm）。",
            f"- **Overexposure：** `[15,17,52,59]`为最佳（{as_float(exposure_winner['historical_ate_mean_cm_mean']):.3f} cm），比历史CNN低{(1-as_float(exposure_winner['mean_ate_ratio_to_historical_cnn_baseline']))*100:.1f}%。这与其前期呈现的高通道多样性相容，但仍只是相关性证据。",
            f"- **Wet：** `[5,6,15,35]`为最佳（{as_float(wet_winner['historical_ate_mean_cm_mean']):.3f} cm），相对历史CNN {percent_improvement(as_float(wet_winner['mean_ate_ratio_to_historical_cnn_baseline']))}，是最强的CNN特异性收益。",
            f"- **Dirt：** 历史CNN baseline自身为最佳（9.935 cm）；所有替代配置均更高。这是当前Top-7对该类局部遮挡/污染迁移不足的直接反例。",
            "",
            "## 5.4 推荐不应压缩成单一‘全局最佳’CNN",
            "",
            f"如果需要一个仅由CNN构成的通用候选，`[5,6,15,35]`最合适：7种退化均3/3 PASS，跨退化ATE比值几何均值为{float(best_cnn['geomean_mean_ate_ratio_to_historical_baseline']):.4f}，并在wet中显著领先。若研究问题强调过曝/illumination sensitivity，则`[15,17,52,59]`是更有针对性的候选（几何均值{float(diverse['geomean_mean_ate_ratio_to_historical_baseline']):.4f}、5/7优于历史CNN）。",
            "",
            "# 6. 局限性与下一步",
            "",
            "1. 七个数据集是同一fr1/desk轨迹的退化版本，而不是七条独立真实轨迹；因此它们适合做配对外观退化比较，但不能代表跨场景泛化。",
            "2. 三次运行的ATE完全相同，表明当前流程确定性很高；它不提供硬件、随机性或新场景下的置信区间。",
            "3. 主ATE采用Sim(3) alignment与scale correction。论文级结论还应同时检查全帧SE(3)指标、RPE和轨迹图，尤其是出现低ATE但高Trans-RPE-max的案例。",
            "4. Mapping固定使用sensor depth，且映射端优化被decouple；结论聚焦于tracking的图像输入/通道选择，不应外推到自由深度优化或纯单目设置。",
            "5. Top-7来源于早期fr1 lightswitch筛选，存在selection bias。QueensCAMP结果是外部验证，而不是对全部64通道组合的重新搜索。",
            "",
            "# 附录：权威数据与可审计文件",
            "",
            "- 56个数据集×配置汇总：`channel_selection_results/step_h_queenscamp_degradation_evaluation/three_repeats/per_dataset_configuration_summary.csv`",
            "- 8个配置综合排序：`channel_selection_results/step_h_queenscamp_degradation_evaluation/three_repeats/configuration_overall_summary.csv`",
            "- 168次原始记录：`channel_selection_results/step_h_queenscamp_degradation_evaluation/three_repeats/all_runs_raw.csv`",
            "- 每个数据集的SQLite记录与trajectory artifacts：`channel_selection_results/step_h_queenscamp_degradation_evaluation/three_repeats/per_dataset/`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] Markdown: {OUTPUT_MD}")
    print(f"[DONE] Figures: {REPORT_DIR / 'queenscamp_ate_heatmap.png'}")


if __name__ == "__main__":
    main()
