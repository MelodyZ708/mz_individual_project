#!/usr/bin/env python3
"""Generate the combined nine-dataset Chinese report and supporting figures."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = Path(__file__).resolve().parent
STEP_F = PROJECT_ROOT / "channel_selection_results/step_f_multi_dataset_evaluation"
OLD_SCORECARD = STEP_F / "dataset_scorecard.csv"
NEW_CELLS = STEP_F / "lightswitch_5x_evaluation/per_dataset_candidate_summary.csv"
CANDIDATE_PLAN = (
    PROJECT_ROOT
    / "channel_selection_pipeline/scripts/step_f_multi_dataset_evaluation/"
    "top7_candidate_plan.json"
)
OUTPUT_MD = REPORT_DIR / "top7_nine_dataset_lightswitch_5x_report_zh.md"
BASELINE = "5,29,40,52"
OLD_DATASETS = (
    ("fr1_desk_clean", "fr1 desk clean", "F1-C"),
    ("fr1_desk_flashlight", "fr1 desk flashlight", "F1-F"),
    ("fr2_desk_clean", "fr2 desk clean", "F2-C"),
    ("fr2_desk_flashlight", "fr2 desk flashlight", "F2-F"),
    ("fr3_office_clean", "fr3 office clean", "F3-C"),
    ("fr3_office_flashlight", "fr3 office flashlight", "F3-F"),
)
NEW_DATASETS = (
    ("fr1_desk_lightswitch", "fr1 desk lightswitch", "F1-L×5"),
    ("fr2_desk_lightswitch", "fr2 desk lightswitch", "F2-L×5"),
    ("fr3_office_lightswitch", "fr3 office lightswitch", "F3-L×5"),
)
DISPLAY_DATASETS = (
    OLD_DATASETS[0], OLD_DATASETS[1], NEW_DATASETS[0],
    OLD_DATASETS[2], OLD_DATASETS[3], NEW_DATASETS[1],
    OLD_DATASETS[4], OLD_DATASETS[5], NEW_DATASETS[2],
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def failure_frame_text(raw: str) -> str:
    values = [value for value in json.loads(raw) if value is not None]
    if not values:
        return "—"
    unique = sorted(set(values))
    if len(unique) == 1:
        return f"{unique[0]} × {len(values)}"
    return ", ".join(map(str, values))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(CANDIDATE_PLAN.read_text(encoding="utf-8"))["candidates"]
    candidate_keys = [item["candidate_key"] for item in candidates]
    source_rank = {item["candidate_key"]: item["source_rank"] for item in candidates}
    old_rows = read_csv(OLD_SCORECARD)
    new_rows = read_csv(NEW_CELLS)
    old_lookup = {
        (row["candidate_key"], row["dataset_key"]): row
        for row in old_rows
        if row["dataset_key"] in {item[0] for item in OLD_DATASETS}
    }
    new_lookup = {
        (row["candidate_key"], row["dataset_key"]): row for row in new_rows
    }

    cells: dict[tuple[str, str], dict] = {}
    for key in candidate_keys:
        for dataset, _, _ in OLD_DATASETS:
            row = old_lookup[(key, dataset)]
            cells[(key, dataset)] = {
                "ate": number(row["historical_evo_ape_mean_cm"]),
                "std": None,
                "passes": 1 if row["status"] == "PASS" else 0,
                "planned": 1,
                "status": row["status"],
            }
        for dataset, _, _ in NEW_DATASETS:
            row = new_lookup[(key, dataset)]
            passes = int(row["pass_count"])
            cells[(key, dataset)] = {
                "ate": number(row["historical_ate_mean_cm_mean"]),
                "std": number(row["historical_ate_mean_cm_std"]),
                "passes": passes,
                "planned": 5,
                "status": "PASS" if passes else "FAIL_TRACKING_NAN",
            }

    winners: dict[str, str] = {}
    for dataset, _, _ in DISPLAY_DATASETS:
        winners[dataset] = min(
            candidate_keys,
            key=lambda key: (
                -cells[(key, dataset)]["passes"],
                cells[(key, dataset)]["ate"]
                if cells[(key, dataset)]["ate"] is not None
                else math.inf,
            ),
        )

    baseline_ates = {
        dataset: cells[(BASELINE, dataset)]["ate"] for dataset, _, _ in DISPLAY_DATASETS
    }
    composite_rows = []
    for key in candidate_keys:
        ratios = []
        beats = 0
        for dataset, _, _ in DISPLAY_DATASETS:
            ate = cells[(key, dataset)]["ate"]
            baseline_ate = baseline_ates[dataset]
            if ate is not None and baseline_ate is not None:
                ratio = ate / baseline_ate
                ratios.append(ratio)
                beats += ate < baseline_ate
        composite_rows.append(
            {
                "key": key,
                "pass_trials": sum(cells[(key, dataset)]["passes"] for dataset, _, _ in DISPLAY_DATASETS),
                "dataset_coverage": sum(cells[(key, dataset)]["passes"] > 0 for dataset, _, _ in DISPLAY_DATASETS),
                "paired": len(ratios),
                "beats": beats,
                "ratio": geometric_mean(ratios),
            }
        )

    # Combined ATE figure.
    matrix = np.asarray(
        [[cells[(key, dataset)]["ate"] for dataset, _, _ in DISPLAY_DATASETS] for key in candidate_keys],
        dtype=float,
    )
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#d9d9d9")
    fig, axis = plt.subplots(figsize=(13.5, 6.0))
    image = axis.imshow(matrix, aspect="auto", cmap=cmap)
    axis.set_xticks(range(9), [item[2] for item in DISPLAY_DATASETS])
    axis.set_yticks(range(7), [f"[{key}]" for key in candidate_keys])
    axis.set_title("Historical keyframe ATE mean (cm): single-run C/F and 5-run mean L")
    for row in range(7):
        for column in range(9):
            value = matrix[row, column]
            axis.text(column, row, "FAIL" if np.isnan(value) else f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, shrink=0.82, label="ATE mean (cm)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "nine_dataset_ate_heatmap.png", dpi=190)
    plt.close(fig)

    # Five-repeat PASS-count figure.
    pass_matrix = np.asarray(
        [[cells[(key, dataset)]["passes"] for dataset, _, _ in NEW_DATASETS] for key in candidate_keys],
        dtype=float,
    )
    fig, axis = plt.subplots(figsize=(8.2, 5.8))
    image = axis.imshow(pass_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=5)
    axis.set_xticks(range(3), [item[2] for item in NEW_DATASETS])
    axis.set_yticks(range(7), [f"[{key}]" for key in candidate_keys])
    axis.set_title("Lightswitch reproducibility: PASS count out of five")
    for row in range(7):
        for column in range(3):
            axis.text(column, row, f"{int(pass_matrix[row, column])}/5", ha="center", va="center", fontsize=10)
    fig.colorbar(image, ax=axis, shrink=0.82, ticks=range(6))
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "lightswitch_pass_count_heatmap.png", dpi=190)
    plt.close(fig)

    lines = [
        "---",
        'title: "Top-7通道配置九数据集综合评估"',
        'subtitle: "Clean/Flashlight单次结果与Lightswitch五次重复结果的统一整理"',
        'author: "MSc Project 阶段性汇报材料"',
        'date: "2026年8月11日"',
        "lang: zh-CN",
        "---",
        "",
        '::: {custom-style="Title"}',
        "Top-7通道配置九数据集综合评估",
        ":::",
        "",
        '::: {custom-style="Subtitle"}',
        "Clean/Flashlight单次结果与Lightswitch五次重复结果的统一整理  ",
        "MSc Project 阶段性汇报材料 · 2026年8月11日",
        ":::",
        "",
        "# 执行摘要",
        "",
        "本文统一整理7个Conv1四通道配置在9个TUM派生数据集上的表现。fr1/fr2/fr3的clean与flashlight采用较早的Step-F单次完整序列结果；三个lightswitch采用最新的每配置5次独立重复结果。所有结果均以COMO keyframe trajectory上的`evo_ape tum --align --correct_scale` ATE mean为主精度指标。",
        "",
        "核心结论如下：",
        "",
        "1. 三个lightswitch共105次运行全部完成，其中85次PASS、20次`FAIL_TRACKING_NAN`。5次重复的PASS/FAIL和ATE逐次完全一致，所有可计算ATE的标准差均为0.0000 cm，表明当前固定数据与配置下结果具有确定性。",
        "2. `[15,17,52,59]`、`[6,10,34,41]`、`[5,6,24,29]`和`[5,6,15,35]`在9个数据集的全部21次计划观测中均PASS；其中`[15,17,52,59]`是lightswitch可靠性优先的第一名。",
        "3. `[1,5,24,29]`在所有8个可与baseline比较且自身有ATE的数据集上均优于baseline，baseline-normalized ATE几何均值改善约13.2%，但在fr2 lightswitch中0/5 PASS，因此是精度型而非可靠性型冠军。",
        "4. Baseline `[5,29,40,52]`在fr1与fr3 lightswitch均为5/5 PASS，但在fr2 lightswitch为0/5，并且5次均在frame 1737失败。此前记忆中的成功未在当前固定协议下复现。",
        "5. clean/flashlight的最佳配置随数据集变化：fr1 clean由`[1,5,24,29]`获胜；fr1 flashlight与fr2 clean/flashlight由`[5,6,15,35]`获胜；fr3 clean/flashlight由`[1,26,30,40]`获胜。不存在单一配置在所有场景中同时达到最低ATE。",
        "",
        "# 1. 数据来源与统一口径",
        "",
        "| 数据组 | 序列 | 统计口径 | 主表显示 |",
        "|---|---|---|---|",
        "| 较早Step-F | fr1 desk clean / flashlight | 每配置1次 | ATE mean + PASS 1/1 |",
        "| 较早Step-F | fr2 desk clean / flashlight | 每配置1次 | ATE mean + PASS 1/1 |",
        "| 较早Step-F | fr3 office clean / flashlight | 每配置1次 | ATE mean + PASS 1/1 |",
        "| 最新重复实验 | fr1 desk lightswitch | 每配置5次 | PASS次数与PASS-run ATE均值±标准差 |",
        "| 最新重复实验 | fr2 desk lightswitch | 每配置5次 | PASS次数与PASS-run ATE均值±标准差 |",
        "| 最新重复实验 | fr3 office lightswitch | 每配置5次 | PASS次数与PASS-run ATE均值±标准差 |",
        "",
        "对于lightswitch，平均ATE仅使用PASS运行；PASS次数始终以5次为分母。若0/5 PASS，则不报告ATE。不同序列的绝对ATE不可直接相加，因此跨数据集比较同时报告完成率、dataset coverage及相对同数据集baseline的ATE比值。",
        "",
        "# 2. 每个数据集上的最优配置",
        "",
        "最佳配置采用可靠性优先规则：先最大化PASS次数，再在相同PASS次数下最小化历史ATE mean。",
        "",
        "| 数据集 | 证据 | 最优配置 | ATE mean | PASS | 相对baseline |",
        "|---|---|---|---:|---:|---:|",
    ]
    for dataset, display, _ in DISPLAY_DATASETS:
        key = winners[dataset]
        cell = cells[(key, dataset)]
        baseline_ate = baseline_ates[dataset]
        improvement = (
            (1.0 - cell["ate"] / baseline_ate) * 100.0
            if cell["ate"] is not None and baseline_ate is not None
            else None
        )
        evidence = "5次均值" if cell["planned"] == 5 else "较早单次"
        relative = f"改善{improvement:.2f}%" if improvement is not None else "baseline 0/5，无ATE"
        lines.append(
            f"| {display} | {evidence} | `[{key}]` | {cell['ate']:.4f} cm | "
            f"{cell['passes']}/{cell['planned']} | {relative} |"
        )

    lines.extend([
        "",
        "# 3. 七个配置在九个数据集上的表现",
        "",
        "表内数字为历史keyframe ATE mean（cm），括号内为PASS次数。L×5列为最新5次平均；C/F列为较早单次结果。粗体表示该数据集按可靠性优先规则选出的最佳配置。",
        "",
        "| 配置 | F1-C | F1-F | F1-L×5 | F2-C | F2-F | F2-L×5 | F3-C | F3-F | F3-L×5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key in candidate_keys:
        values = []
        for dataset, _, _ in DISPLAY_DATASETS:
            cell = cells[(key, dataset)]
            if cell["ate"] is None:
                text = f"FAIL ({cell['passes']}/{cell['planned']})"
            else:
                text = f"{cell['ate']:.3f} ({cell['passes']}/{cell['planned']})"
            if winners[dataset] == key:
                text = f"**{text}**"
            values.append(text)
        lines.append(f"| `[{key}]` | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "![图1　九数据集历史ATE mean热力图；灰色FAIL表示该lightswitch配置5次均未产生有效轨迹。](nine_dataset_ate_heatmap.png){width=98%}",
        "",
        "# 4. Lightswitch五次重复的详细结果",
        "",
        "| 数据集 | 配置 | PASS | ATE mean ± std (cm) | RPE RMSE mean (cm) | 重复失败帧 |",
        "|---|---|---:|---:|---:|---|",
    ])
    for dataset, display, _ in NEW_DATASETS:
        dataset_rows = [row for row in new_rows if row["dataset_key"] == dataset]
        dataset_rows.sort(key=lambda row: source_rank[row["candidate_key"]])
        for row in dataset_rows:
            passes = int(row["pass_count"])
            ate = number(row["historical_ate_mean_cm_mean"])
            std = number(row["historical_ate_mean_cm_std"])
            rpe = number(row["historical_rpe_rmse_cm_mean"])
            ate_text = f"{ate:.4f} ± {std:.4f}" if ate is not None else "—"
            rpe_text = f"{rpe:.4f}" if rpe is not None else "—"
            lines.append(
                f"| {display} | `[{row['candidate_key']}]` | {passes}/5 | "
                f"{ate_text} | {rpe_text} | {failure_frame_text(row['failure_frames'])} |"
            )

    lines.extend([
        "",
        "![图2　三个lightswitch序列的5次PASS计数。](lightswitch_pass_count_heatmap.png){width=78%}",
        "",
        "## 4.1 重复性与失败机制",
        "",
        "- fr1 lightswitch：全部7个配置均5/5 PASS，且每个配置5次ATE完全一致。",
        "- fr2 lightswitch：四个配置均5/5 PASS；`[1,26,30,40]`在frame 1028重复失败5次，`[1,5,24,29]`在frame 1027重复失败5次，baseline在frame 1737重复失败5次。",
        "- fr3 lightswitch：六个配置均5/5 PASS；仅`[1,26,30,40]`在frame 1796重复失败5次。",
        "- 这些结果说明当前COMO执行路径基本确定性；重复实验的主要价值是确认失败是否稳定复现，而不是估计随机方差。",
        "",
        "# 5. 配置级跨数据集综合比较",
        "",
        "这里将6个单次C/F观测与15个lightswitch重复观测合并，因此每配置共有21次计划运行、最多覆盖9个数据集。ATE比值只在candidate与baseline均有有效ATE的数据集上计算。",
        "",
        "| 配置 | PASS/21 | 有PASS的数据集/9 | 可比数据集 | 优于baseline | ATE比值几何均值 | 解释 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    interpretations = {
        "5,6,24,29": "全部完成，但跨序列ATE泛化偏弱",
        "1,26,30,40": "C/F精度强；两类lightswitch稳定失败",
        "15,17,52,59": "lightswitch可靠性冠军；fr2-L最优",
        "1,5,24,29": "baseline-relative精度最强；fr2-L脆弱",
        "5,6,15,35": "全部完成；C/F与总体精度最平衡",
        "6,10,34,41": "全部完成；精度中等但可靠",
        "5,29,40,52": "历史baseline；fr2-L稳定失败",
    }
    for row in composite_rows:
        lines.append(
            f"| `[{row['key']}]` | {row['pass_trials']}/21 | {row['dataset_coverage']}/9 | "
            f"{row['paired']} | {row['beats']} | {row['ratio']:.4f} | {interpretations[row['key']]} |"
        )

    lines.extend([
        "",
        "# 6. 结果解读与推荐",
        "",
        "## 6.1 没有单一无条件冠军",
        "",
        "若将tracking failure视为硬约束，四个21/21 PASS配置构成可靠集合：`[15,17,52,59]`、`[6,10,34,41]`、`[5,6,24,29]`和`[5,6,15,35]`。其中`[5,6,15,35]`在合并的8个baseline可比较数据集上ATE比值几何均值为1.001，几乎与baseline持平，同时避免了baseline的fr2 lightswitch失败，因此是最平衡的general-purpose候选。",
        "",
        "## 6.2 Illumination-switch robustness",
        "",
        "`[15,17,52,59]`在三个lightswitch上15/15 PASS，并以6.8620 cm取得fr2 lightswitch最低ATE；它在baseline 0/5的fr2序列上保持稳定。因此若研究问题强调突发光照变化下的生存能力，该配置是最有解释力的主推荐。",
        "",
        "## 6.3 Accuracy–reliability trade-off",
        "",
        "`[1,5,24,29]`在8个与baseline可比的数据集上全部取得更低ATE，ATE比值几何均值为0.8677，即条件于成功时约改善13.2%。然而它在fr2 lightswitch中0/5 PASS，不能作为单一稳健方案。`[1,26,30,40]`也表现出类似模式：clean/flashlight精度突出，但fr2和fr3 lightswitch均0/5。包含channel 1的两个组合在fr2相邻frame 1027/1028稳定失败，是值得后续消融验证的结构线索，但目前不能直接证明channel 1是因果来源。",
        "",
        "## 6.4 Baseline重新评价",
        "",
        "Baseline在fr1和fr3 lightswitch各5/5 PASS，但在fr2 lightswitch 0/5，并且全部在frame 1737失败。因此此前观察到的baseline成功不能代表当前固定fr2协议下的稳定行为。最新版重复实验支持将baseline记为“跨数据集8/9有成功、fr2 lightswitch确定性失败”，而不是偶发一次失败。",
        "",
        "# 7. 局限性",
        "",
        "1. clean与flashlight仍是单次结果，而lightswitch是5次重复；两类证据的统计强度不同。",
        "2. 5次重复使用完全相同的输入、配置与执行路径，结果确定性一致；它验证实现重复性，但不代表跨硬件、随机初始化或不同真实光照事件的方差为零。",
        "3. 同一family的clean/flashlight/lightswitch共享相机运动与ground truth，有利于配对比较，但不能替代更多独立真实序列。",
        "4. fr2 lightswitch中baseline没有有效ATE，因此该序列无法参与baseline-normalized精度比值；必须同时查看PASS次数与绝对ATE。",
        "5. Top-7来自fr1 lightswitch上的前期筛选，仍存在selection bias；多序列结果用于验证而不是重新穷举全部通道组合。",
        "",
        "# 8. 建议的最终汇报口径",
        "",
        "建议向导师同时报告三种角色，而不是压缩成一个best：",
        "",
        "- **突变光照可靠性主候选：** `[15,17,52,59]`（lightswitch 15/15 PASS，fr2-L最低ATE）。",
        "- **跨条件平衡候选：** `[5,6,15,35]`（九数据集21/21 PASS，C/F表现突出，整体ATE与baseline近似持平）。",
        "- **条件成功时的精度候选：** `[1,5,24,29]`（8/8可比数据集ATE均优于baseline，但fr2-L 0/5）。",
        "",
        "最终论文表格应将PASS rate放在ATE之前：失败配置不能因为其成功子集ATE较低而被错误排到稳定配置之前。",
        "",
        "# 附录：权威结果文件",
        "",
        "- 较早单次结果：`channel_selection_results/step_f_multi_dataset_evaluation/dataset_scorecard.csv`",
        "- Lightswitch五次单元统计：`lightswitch_5x_evaluation/per_dataset_candidate_summary.csv`",
        "- Lightswitch综合统计：`lightswitch_5x_evaluation/candidate_overall_summary.csv`",
        "- 105次原始记录：`lightswitch_5x_evaluation/all_runs_raw.csv`",
    ])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_MD)


if __name__ == "__main__":
    main()
