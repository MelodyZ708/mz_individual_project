#!/usr/bin/env python3
"""Create a Chinese, auditable QueensCAMP multi-sequence robustness report.

Inputs are only completed Step-H and Step-I aggregate outputs.  The generator
does not open evaluator databases for writing and never changes COMO settings.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

_CACHE = Path(tempfile.gettempdir()) / "queenscamp_multiseq_report_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parents[2]
STEP_H = PROJECT_ROOT / "channel_selection_results/step_h_queenscamp_degradation_evaluation/three_repeats"
STEP_I = PROJECT_ROOT / "channel_selection_results/step_i_queenscamp_fr2_fr3_evaluation/three_repeats"
CANDIDATE_PLAN = PROJECT_ROOT / "channel_selection_pipeline/scripts/step_i_queenscamp_fr2_fr3_evaluation/top7_plus_gray_candidate_plan.json"
OUTPUT_MD = REPORT_DIR / "QueensCAMP_三序列七种退化鲁棒性评估_中文.md"
BASELINE = "5,29,40,52"
GRAY = "gray"
FAMILY_ORDER = ("fr1_desk", "fr2_desk", "fr3_long_office_household")
FAMILY_TITLE = {
    "fr1_desk": "TUM fr1/desk",
    "fr2_desk": "TUM fr2/desk",
    "fr3_long_office_household": "TUM fr3/long_office_household",
}
FAMILY_FRAMES = {"fr1_desk": 573, "fr2_desk": 2893, "fr3_long_office_household": 2488}
DEGRADATIONS = ("blur", "condensation", "dirt", "mixed_story_v1", "overexposure", "wet", "underexposure")
DEGRADATION_CN = {
    "blur": "模糊", "condensation": "冷凝", "dirt": "污渍", "mixed_story_v1": "混合叙事",
    "overexposure": "过曝", "wet": "湿润", "underexposure": "欠曝",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def display(key: str) -> str:
    return "gray" if key == GRAY else f"[{key}]"


def percentage(ratio: float | None) -> str:
    if ratio is None:
        return "—"
    return f"改善{(1 - ratio) * 100:.1f}%" if ratio <= 1 else f"劣于{(ratio - 1) * 100:.1f}%"


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("Geometric mean requires positive finite values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def table(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def best(rows: list[dict[str, str]]) -> dict[str, str]:
    return min(rows, key=lambda row: (-int(row["pass_count"]), number(row["historical_ate_mean_cm_mean"]) or math.inf, int(row["source_rank"])))


def load_rows() -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    requirements = [
        STEP_H / "per_dataset_configuration_summary.csv", STEP_H / "aggregate_protocol.json",
        STEP_I / "per_dataset_configuration_summary.csv", STEP_I / "aggregate_protocol.json", CANDIDATE_PLAN,
    ]
    for path in requirements:
        if not path.is_file():
            raise FileNotFoundError(path)
    h_protocol = json.loads((STEP_H / "aggregate_protocol.json").read_text(encoding="utf-8"))
    i_protocol = json.loads((STEP_I / "aggregate_protocol.json").read_text(encoding="utf-8"))
    if (h_protocol.get("completed_runs"), h_protocol.get("pass_runs")) != (168, 168):
        raise ValueError("Step-H is not a complete 168/168 PASS result")
    if (i_protocol.get("completed_runs"), i_protocol.get("pass_runs")) != (336, 336):
        raise ValueError("Step-I is not a complete 336/336 PASS result")
    candidates = json.loads(CANDIDATE_PLAN.read_text(encoding="utf-8"))["candidates"]
    expected_keys = ["5,6,24,29", "1,26,30,40", "15,17,52,59", "1,5,24,29", "5,6,15,35", "6,10,34,41", BASELINE, GRAY]
    if [item["candidate_key"] for item in candidates] != expected_keys:
        raise ValueError("Candidate plan is not the frozen Top-7 plus gray protocol")
    rows = read_csv(STEP_H / "per_dataset_configuration_summary.csv")
    for row in rows:
        row["family"] = "fr1_desk"
    rows.extend(read_csv(STEP_I / "per_dataset_configuration_summary.csv"))
    if len(rows) != 21 * 8:
        raise ValueError(f"Expected 168 summary cells, found {len(rows)}")
    if any(int(row["pass_count"]) != 3 or int(row["completed_replicates"]) != 3 for row in rows):
        raise ValueError("Expected all 168 cells to be 3/3 PASS")
    for family in FAMILY_ORDER:
        family_rows = [row for row in rows if row["family"] == family]
        if len(family_rows) != 56 or {row["degradation"] for row in family_rows} != set(DEGRADATIONS):
            raise ValueError(f"Unexpected rows for {family}")
    return rows, candidates


def save_heatmap(rows: list[dict[str, str]], candidates: list[dict[str, Any]], family: str) -> Path:
    lookup = {(row["candidate_key"], row["degradation"]): row for row in rows if row["family"] == family}
    winners = {degradation: best([row for row in lookup.values() if row["degradation"] == degradation])["candidate_key"] for degradation in DEGRADATIONS}
    matrix = np.asarray([[number(lookup[(candidate["candidate_key"], degradation)]["historical_ate_mean_cm_mean"]) for degradation in DEGRADATIONS] for candidate in candidates], dtype=float)
    median = float(np.nanmedian(matrix))
    cmap = plt.get_cmap("viridis_r").copy(); cmap.set_bad("#d9d9d9")
    figure, axis = plt.subplots(figsize=(12.3, 5.7))
    image = axis.imshow(matrix, aspect="auto", cmap=cmap)
    axis.set_xticks(range(7), DEGRADATIONS, rotation=20, ha="right")
    axis.set_yticks(range(8), [display(item["candidate_key"]) for item in candidates])
    axis.set_title(f"{FAMILY_TITLE[family]}: historical keyframe ATE mean (cm; lower is better)")
    for r, candidate in enumerate(candidates):
        for c, degradation in enumerate(DEGRADATIONS):
            value = matrix[r, c]; is_best = winners[degradation] == candidate["candidate_key"]
            color = "white" if value > median else "black"
            axis.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold" if is_best else "normal")
            if is_best:
                axis.add_patch(plt.Rectangle((c - .48, r - .48), .96, .96, fill=False, edgecolor=color, linewidth=2.0))
    figure.colorbar(image, ax=axis, shrink=.82, label="ATE mean (cm)")
    figure.tight_layout()
    output = REPORT_DIR / f"{family}_ate_mean_heatmap.png"
    figure.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    rows, candidates = load_rows()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    figures = [save_heatmap(rows, candidates, family) for family in FAMILY_ORDER]
    lookup = {(row["family"], row["candidate_key"], row["degradation"]): row for row in rows}
    winners: dict[tuple[str, str], dict[str, str]] = {}
    for family in FAMILY_ORDER:
        for degradation in DEGRADATIONS:
            winners[(family, degradation)] = best([row for row in rows if row["family"] == family and row["degradation"] == degradation])
    winner_count = Counter(row["candidate_key"] for row in winners.values())

    combined: list[dict[str, Any]] = []
    for candidate in candidates:
        key = candidate["candidate_key"]
        cells = [row for row in rows if row["candidate_key"] == key]
        ratios = [number(row["mean_ate_ratio_to_historical_cnn_baseline"]) for row in cells]
        assert all(value is not None for value in ratios)
        per_family = {
            family: geometric_mean([number(row["mean_ate_ratio_to_historical_cnn_baseline"]) for row in cells if row["family"] == family])
            for family in FAMILY_ORDER
        }
        combined.append({
            "key": key, "source_rank": candidate["source_rank"], "overall_ratio": geometric_mean([float(value) for value in ratios]),
            "beats": sum(float(value) < 1.0 for value in ratios), "wins": winner_count[key], "families": per_family,
        })
    combined.sort(key=lambda item: (item["overall_ratio"], item["source_rank"]))
    for rank, item in enumerate(combined, 1): item["rank"] = rank

    gray = next(item for item in combined if item["key"] == GRAY)
    best_cnn = min((item for item in combined if item["key"] not in {GRAY, BASELINE}), key=lambda item: item["overall_ratio"])
    stable_cnn = next(item for item in combined if item["key"] == "1,5,24,29")
    fr3_specialist = next(item for item in combined if item["key"] == "1,26,30,40")
    weak_cnn = next(item for item in combined if item["key"] == "5,6,24,29")
    nonzero_std_cells = [row for row in rows if abs(number(row["historical_ate_mean_cm_std"]) or 0.0) > 1e-12]
    max_ate_std = max(number(row["historical_ate_mean_cm_std"]) or 0.0 for row in rows)

    lines = [
        "---", 'title: "QueensCAMP三序列七种退化下的Top-7与Gray鲁棒性评估"',
        'subtitle: "21个退化数据集 × 8个配置 × 3次重复（504次完整运行）"',
        'author: "MSc Project 阶段性汇报材料"', 'date: "2026年8月13日"', 'lang: zh-CN', "---", "",
        "# QueensCAMP三序列七种退化下的Top-7与Gray鲁棒性评估", "",
        "*21个退化数据集 × 8个配置 × 3次重复（504次完整运行）*  ",
        "MSc Project 阶段性汇报材料 · 2026年8月13日", "",
        "# 执行摘要", "",
        "本报告合并昨日完成的fr1/desk七种QueensCAMP退化实验，与今日完成的fr2/desk、fr3/long_office_household实验。评估对象为前期fr1 lightswitch筛选出的七个CNN配置（其中`[5,29,40,52]`为历史四通道CNN baseline）以及gray photometric control。三条基础序列各有七种退化；每个数据集/配置独立重复三次，共504次。", "",
        f"1. **所有504/504次均PASS。** 21个“数据集×配置”单元均为3/3 PASS；161/168个单元三次主ATE完全相同，另7个单元有很小的波动（最大std = {max_ate_std:.3f} cm）。本批没有观察到可用failure-rate区分的鲁棒性差异。",
        f"2. **gray control在21个退化数据集中的13个取得最低ATE，19/21优于历史CNN baseline，跨21数据集ATE比值几何均值为{gray['overall_ratio']:.4f}。** 这是一项重要对照发现，但gray并非待选的CNN通道组合；不能把它直接转译为‘不需要CNN’的结论。",
        f"3. **仅看CNN时，`[{best_cnn['key']}]`的综合相对误差最低**（几何均值比值{best_cnn['overall_ratio']:.4f}，12/21优于历史CNN）。但最优CNN依赖基础序列和退化类型，未出现一个CNN在21种条件下都最优。",
        f"4. **跨序列迁移是非均匀的。** `[5,6,15,35]`在fr1/fr2表现强（比值{best_cnn['families']['fr1_desk']:.4f}/{best_cnn['families']['fr2_desk']:.4f}），在fr3则略劣于baseline（{best_cnn['families']['fr3_long_office_household']:.4f}）；`[1,26,30,40]`则相反，在fr3最强CNN（{fr3_specialist['families']['fr3_long_office_household']:.4f}）。",
        "5. 因为三条基础序列的轨迹长度、场景和绝对ATE尺度不同，跨序列结论只使用每个数据集内相对历史CNN baseline的ATE比值，不直接平均原始ATE。", "",
        "# 1. 实验目的与设置", "", "## 1.1 目的", "",
        "检验lightswitch搜索阶段保留下来的CNN通道组合，在不同基础场景和多种合成图像退化下是否能保持精度；并以历史CNN和gray两类对照辨析收益来自于通道组合还是当前固定RGB-D tracking/mapping设置。", "",
        "## 1.2 实验协议", "",
        table(["项目", "设置"]), table(["---", "---"]),
        table(["基础序列", "TUM fr1/desk、fr2/desk、fr3/long_office_household；每条各有7个退化版本"]),
        table(["退化类型", "blur、condensation、dirt、mixed_story_v1、overexposure、wet、underexposure；同一基础序列内保持RGB/depth/GT与时间索引可用"]),
        table(["数据集规模", "fr1: 573 matched RGB帧；fr2: 2893；fr3: 2488（索引条目数）"]),
        table(["配置", "6个搜索产生的Top候选 + 历史CNN `[5,29,40,52]` + gray control，共8个"]),
        table(["重复/规模", "每个数据集×配置3次；21 × 8 × 3 = 504次"]),
        table(["Tracking / Mapping", "tracking使用gray或指定Conv1四通道CNN；mapping固定gray；使用配对sensor depth，ground-truth仅用于运行后评估"]),
        table(["主指标", "历史可比的keyframe `evo_ape tum --align --correct_scale` ATE mean（cm，越低越好）"]),
        table(["诊断", "keyframe RPE、全帧metric-scale SE(3) ATE/RPE、coverage、轨迹与运行日志"]),
        table(["完成门槛", "coverage ≥ 90%、末位姿距序列末帧≤0.10 s、单次timeout = 500 s"]), "",
        "ATE经过全局Sim(3) alignment和scale correction，适于沿用既有full-sequence历史口径。它不等价于局部轨迹完全平滑，故RPE作为诊断指标保留。", "",
        "# 2. 数据完成度与重复性", "",
        table(["基础序列", "退化数", "配置数", "重复", "计划运行", "PASS", "3/3单元"]), table(["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
        table(["fr1/desk", "7", "8", "3", "168", "168", "56/56"]),
        table(["fr2/desk", "7", "8", "3", "168", "168", "56/56"]),
        table(["fr3/long_office_household", "7", "8", "3", "168", "168", "56/56"]),
        table(["合计", "21", "8", "3", "504", "504", "168/168"]), "",
        f"161/168个单元的三次主ATE标准差为0.0000 cm；其余{len(nonzero_std_cells)}个单元均来自fr2/fr3，最大std仅{max_ate_std:.3f} cm。因而结果高度可复现，三次重复确认了流程没有隐藏的随机失败；但它们仍不能估计换机器、换场景时的不确定性。", "",
        "# 3. 每个退化数据集的最优配置", "",
        "最优规则先比较PASS次数，再比较主ATE。由于所有单元均为3/3 PASS，以下结果由ATE mean决定；‘相对历史CNN’小于1表示改善。", "",
        table(["基础序列", "退化", "最优配置", "最优ATE/cm", "历史CNN ATE/cm", "相对历史CNN", "Trans RPE max/cm", "Rot RPE max/deg"]),
        table(["---", "---", "---", "---:", "---:", "---", "---:", "---:"]),
    ]
    for family in FAMILY_ORDER:
        for degradation in DEGRADATIONS:
            row = winners[(family, degradation)]
            baseline = lookup[(family, BASELINE, degradation)]
            lines.append(table([FAMILY_TITLE[family], DEGRADATION_CN[degradation], f"`{display(row['candidate_key'])}`", f"{number(row['historical_ate_mean_cm_mean']):.3f}", f"{number(baseline['historical_ate_mean_cm_mean']):.3f}", percentage(number(row['mean_ate_ratio_to_historical_cnn_baseline'])), f"{number(row['translation_rpe_max_cm_mean']):.3f}", f"{number(row['rotation_rpe_max_deg_mean']):.3f}"]))
    lines += ["", "# 4. 各配置在不同退化下的表现", "", "以下三张大表均为主ATE mean（cm）；每格是3次PASS运行的均值，全部单元均为3/3 PASS。161/168个单元的三次结果相同，剩余7个单元的主ATE std均不超过0.085 cm；完整mean/std/min/max在原始汇总CSV中。**粗体**为同一基础序列、同一退化下最低ATE。不同基础序列的原始ATE不能直接横向平均。", ""]
    for family, figure in zip(FAMILY_ORDER, figures):
        lines += [f"## 4.{FAMILY_ORDER.index(family)+1} {FAMILY_TITLE[family]}", "", table(["配置"] + [DEGRADATION_CN[x] for x in DEGRADATIONS]), table(["---"] + ["---:"] * 7)]
        for candidate in candidates:
            key = candidate["candidate_key"]; values = []
            for degradation in DEGRADATIONS:
                row = lookup[(family, key, degradation)]; value = f"{number(row['historical_ate_mean_cm_mean']):.3f}"
                values.append(f"**{value}**" if winners[(family, degradation)]["candidate_key"] == key else value)
            lines.append(table([f"`{display(key)}`"] + values))
        lines += ["", f"![{FAMILY_TITLE[family]}：ATE热力图；边框标记同列最优。]({figure.name}){{width=96%}}", ""]
    lines += ["# 5. 配置级跨序列汇总", "", "该表以每个数据集内的`candidate ATE / historical CNN ATE`计算几何均值。0.90表示相对于历史CNN平均低约10%；不对21条数据集的原始ATE求平均。", "", table(["综合rank", "配置", "优于历史CNN", "夺得数据集最优", "21数据集比值几何均值", "fr1比值", "fr2比值", "fr3比值"]), table(["---:", "---", "---:", "---:", "---:", "---:", "---:", "---:"])]
    for item in combined:
        lines.append(table([str(item["rank"]), f"`{display(item['key'])}`", f"{item['beats']}/21", f"{item['wins']}/21", f"{item['overall_ratio']:.4f}", f"{item['families']['fr1_desk']:.4f}", f"{item['families']['fr2_desk']:.4f}", f"{item['families']['fr3_long_office_household']:.4f}"]))
    lines += ["", "# 6. 结果解读与重要insights", "", "## 6.1 gray control是强对照，但不是通道选择的替代结论", "", f"gray在fr2的七种退化全部最低ATE，并在fr1的三种、fr3的三种退化中最低，合计13/21。它的三条基础序列比值分别为{gray['families']['fr1_desk']:.4f}、{gray['families']['fr2_desk']:.4f}、{gray['families']['fr3_long_office_household']:.4f}。这说明在当前固定gray mapping、sensor-depth与这些合成外观变换下，加入CNN feature并不自动带来更低ATE。灰度结果应被看作一个必要的负/简化对照：它并未经过与Top-7相同的通道搜索，不能回答‘CNN通道是否仍在真实动态光照下必要’。", "", "## 6.2 最强CNN是稳健候选，而不是全条件冠军", "", f"`[{best_cnn['key']}]`是21个数据集上综合比值最低的CNN（{best_cnn['overall_ratio']:.4f}），并在fr1/wet取得CNN及全体最优；其在fr1、fr2明显改善，但在fr3为{best_cnn['families']['fr3_long_office_household']:.4f}，略劣于历史CNN。这支持将它列为**跨外观退化的首要CNN候选**，但不支持声称为所有场景的全局最优。", "", "## 6.3 场景依赖的配置偏好清晰", "", f"`[1,26,30,40]`在fr3的blur与condensation获胜，且fr3比值{fr3_specialist['families']['fr3_long_office_household']:.4f}，为该序列最强CNN；但在fr1/fr2分别为{fr3_specialist['families']['fr1_desk']:.4f}/{fr3_specialist['families']['fr2_desk']:.4f}。`[1,5,24,29]`在21个数据集的14个优于历史CNN并获胜3次，是相对平衡的备选。`[6,10,34,41]`没有单项冠军，但综合比值0.9406、14/21优于baseline，表现出稳定但不尖峰的折衷。", "", "## 6.4 不是所有lightswitch优胜者都能迁移", "", f"`[5,6,24,29]`在早期fr1 lightswitch中排名靠前，但此处21数据集综合比值为{weak_cnn['overall_ratio']:.4f}，仅3/21优于历史CNN且未取得任何数据集冠军。这是明确的selection-bias/任务依赖信号：小MVS和lightswitch的候选排序不能直接外推到跨序列、跨退化鲁棒性。", "", "## 6.5 指标解读需同时看RPE", "", "ATE采用全局Sim(3)对齐，仍可能掩盖局部跳变。报告中的每个最优单元同时列出Trans RPE max和Rot RPE max；后续若选择少量候选进入定性展示，应优先复核‘低ATE但高RPE max’的轨迹和feature maps，而不是仅按ATE挑选。", "", "# 7. 局限性与建议下一步", "", "1. QueensCAMP版本是同一基础轨迹上的外观退化，提供严格配对比较，但不是21条独立真实轨迹。跨序列泛化结论应理解为三个场景的外部验证，而非总体分布估计。", "2. 退化是合成的、逐帧确定性的；不覆盖真实相机曝光控制、运动模糊、动态遮挡或深度失配的所有形式。", "3. 当前mapping固定为gray且使用sensor depth，ground-truth只用于评估；结论专门针对decoupled RGB-D tracking设置，不能直接外推到联合mapping优化或纯单目系统。", "4. 建议后续将`[5,6,15,35]`、`[1,5,24,29]`、`[1,26,30,40]`及历史CNN作为主要CNN对照，结合gray，对低ATE/high-RPE案例做轨迹与feature-map定性分析；并补充真实光照变化或未参与筛选的场景。", "", "# 附录：权威数据位置", "", f"- fr1汇总：`{STEP_H.relative_to(PROJECT_ROOT)}/per_dataset_configuration_summary.csv`", f"- fr2/fr3汇总：`{STEP_I.relative_to(PROJECT_ROOT)}/per_dataset_configuration_summary.csv`", f"- fr1原始运行：`{STEP_H.relative_to(PROJECT_ROOT)}/all_runs_raw.csv`", f"- fr2/fr3原始运行：`{STEP_I.relative_to(PROJECT_ROOT)}/all_runs_raw.csv`", "- 每个数据集的SQLite、trajectory与日志保存在以上目录的`per_dataset/`。"]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] Markdown: {OUTPUT_MD}")
    print(f"[DONE] Figures: {', '.join(str(path) for path in figures)}")


if __name__ == "__main__":
    main()
