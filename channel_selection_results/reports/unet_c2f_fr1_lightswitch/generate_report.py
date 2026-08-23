#!/usr/bin/env python3
"""Generate the U-Net C2F fr1/desk_lightswitch analysis report and figures."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = (
    PROJECT_ROOT
    / "channel_selection_pipeline/scripts/step_p_c2f_best_channels_evaluation/"
    "unet_c2f_candidate_plan.json"
)
RESULT_DIR = (
    PROJECT_ROOT
    / "channel_selection_results/step_p_c2f_best_channels_evaluation/"
    "unet_fr1_desk_lightswitch"
)
ENC0_RECOMMENDATION = (
    PROJECT_ROOT / "channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy/recommendation.json"
)
ENC1_RECOMMENDATION = (
    PROJECT_ROOT / "channel_selection_results/step_j_unet_direct_fullseq_greedy/recommendation.json"
)
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"


def format_channels(channels: list[int]) -> str:
    return "[" + ",".join(f"d{value}" for value in channels) + "]"


def load_rows():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    fine = {int(item["rank"]): item for item in plan["fine_branch"]["candidates"]}
    coarse = {int(item["rank"]): item for item in plan["coarse_branch"]["candidates"]}
    pattern = re.compile(r"unet_c2f_([ab])_fine(\d+)_coarse(\d+)$")
    rows = []
    with (RESULT_DIR / "all_evaluations.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            match = pattern.fullmatch(raw["label"])
            if match is None:
                raise ValueError(f"Unexpected label: {raw['label']}")
            variant = match.group(1).upper()
            fine_rank, coarse_rank = int(match.group(2)), int(match.group(3))
            rows.append(
                {
                    "label": raw["label"],
                    "variant": variant,
                    "fine_rank": fine_rank,
                    "coarse_rank": coarse_rank,
                    "fine": fine[fine_rank],
                    "coarse": coarse[coarse_rank],
                    "status": raw["status"],
                    "ate": None
                    if not raw["historical_evo_ape_mean_m"]
                    else float(raw["historical_evo_ape_mean_m"]) * 100,
                    "ate_rmse": None
                    if not raw["historical_evo_ape_rmse_m"]
                    else float(raw["historical_evo_ape_rmse_m"]) * 100,
                    "hist_rpe": None
                    if not raw["historical_evo_rpe_rmse_m"]
                    else float(raw["historical_evo_rpe_rmse_m"]) * 100,
                    "se3_rmse": None
                    if not raw["se3_ate_rmse_m"]
                    else float(raw["se3_ate_rmse_m"]) * 100,
                    "se3_mean": None
                    if not raw["se3_ate_mean_m"]
                    else float(raw["se3_ate_mean_m"]) * 100,
                    "translation_rpe_max": None
                    if not raw["translation_rpe_max_m"]
                    else float(raw["translation_rpe_max_m"]) * 100,
                    "rotation_rpe_max": None
                    if not raw["rotation_rpe_max_deg"]
                    else float(raw["rotation_rpe_max_deg"]),
                    "coverage": None if not raw["coverage_ratio"] else float(raw["coverage_ratio"]),
                    "runtime": float(raw["elapsed_seconds"]),
                }
            )
    if len(rows) != 72:
        raise ValueError(f"Expected 72 C2F rows, found {len(rows)}")
    if {row["status"] for row in rows} != {"PASS"}:
        raise ValueError(f"Expected all C2F rows to PASS, got {Counter(row['status'] for row in rows)}")
    for row in rows:
        row["better_parent"] = min(
            float(row["fine"]["source_ate_mean_cm"]),
            float(row["coarse"]["source_ate_mean_cm"]),
        )
        row["gain_vs_better_parent"] = row["better_parent"] - row["ate"]
        row["gain_vs_fine"] = float(row["fine"]["source_ate_mean_cm"]) - row["ate"]
        row["gain_vs_coarse"] = float(row["coarse"]["source_ate_mean_cm"]) - row["ate"]
    return plan, fine, coarse, rows


def direct_candidates(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["candidate_key"]): item
        for item in payload["candidates"]
        if item.get("historical_ate_mean_cm") is not None
    }


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "best": float(min(values)),
        "worst": float(max(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def plot_variant_distribution(rows, direct_best):
    values = [[row["ate"] for row in rows if row["variant"] == variant] for variant in ("A", "B")]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    box = ax.boxplot(values, labels=["C2F-A", "C2F-B"], patch_artist=True, widths=0.52)
    for patch, color in zip(box["boxes"], ("#5B8FF9", "#F6BD16")):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    rng = np.random.default_rng(20260823)
    for index, vals in enumerate(values, start=1):
        ax.scatter(index + rng.uniform(-0.11, 0.11, len(vals)), vals, color="#273142", alpha=0.72, s=22)
    ax.axhline(direct_best, color="#D62728", linestyle="--", linewidth=1.5,
               label=f"best direct Enc0 = {direct_best:.4f} cm")
    ax.set_ylabel("Historical keyframe ATE mean (cm)")
    ax.set_title("U-Net C2F ATE distribution: 36 pairings per variant")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    fig.savefig(FIG / "variant_ate_distribution.png", dpi=220)
    plt.close(fig)


def matrix(rows, variant, field):
    data = np.full((6, 6), np.nan)
    for row in rows:
        if row["variant"] == variant:
            data[row["fine_rank"] - 1, row["coarse_rank"] - 1] = row[field]
    return data


def plot_heatmaps(rows, field, filename, title, cmap, center=None, fmt=".2f"):
    matrices = [matrix(rows, variant, field) for variant in ("A", "B")]
    if center is None:
        low = min(float(np.nanmin(data)) for data in matrices)
        high = max(float(np.nanmax(data)) for data in matrices)
        norm = None
    else:
        magnitude = max(abs(float(np.nanmin(data))) for data in matrices)
        magnitude = max(magnitude, max(abs(float(np.nanmax(data))) for data in matrices))
        low, high, norm = -magnitude, magnitude, None
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.3), constrained_layout=True)
    image = None
    for ax, data, variant in zip(axes, matrices, ("A", "B")):
        image = ax.imshow(data, cmap=cmap, vmin=low, vmax=high, norm=norm, aspect="equal")
        ax.set_title(f"C2F-{variant}")
        ax.set_xlabel("Coarse subset rank")
        ax.set_ylabel("Fine subset rank")
        ax.set_xticks(range(6), [str(value) for value in range(1, 7)])
        ax.set_yticks(range(6), [str(value) for value in range(1, 7)])
        for i in range(6):
            for j in range(6):
                value = data[i, j]
                color = "white" if (center is None and value > (low + high) / 2) else "black"
                if center is not None and abs(value) > magnitude * 0.58:
                    color = "white"
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8.5, color=color)
    label = "ATE mean (cm)" if field == "ate" else "gain over better single-layer parent (cm; positive is better)"
    fig.colorbar(image, ax=axes, shrink=0.86, label=label)
    fig.suptitle(title, fontsize=13)
    fig.savefig(FIG / filename, dpi=220)
    plt.close(fig)


def plot_best_by_fine(rows, fine):
    direct = [float(fine[rank]["source_ate_mean_cm"]) for rank in range(1, 7)]
    best_rows = [min((row for row in rows if row["fine_rank"] == rank), key=lambda row: row["ate"]) for rank in range(1, 7)]
    c2f = [row["ate"] for row in best_rows]
    labels = [f"F{rank}\n{format_channels(fine[rank]['channels'])}" for rank in range(1, 7)]
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(9.5, 4.9), constrained_layout=True)
    width = 0.38
    ax.bar(x - width / 2, direct, width, label="direct Enc0", color="#B9C6E4")
    bars = ax.bar(x + width / 2, c2f, width, label="best C2F partner", color="#4E79A7")
    for index, (source, best, bar) in enumerate(zip(direct, c2f, bars)):
        delta = source - best
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                f"{delta:+.2f}", ha="center", va="bottom", fontsize=8,
                color="#148A3D" if delta > 0 else "#B22222")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("Historical keyframe ATE mean (cm; lower is better)")
    ax.set_title("Best C2F partner available for each Enc0 fine subset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(FIG / "best_c2f_by_fine_subset.png", dpi=220)
    plt.close(fig)


def markdown_table(headers, rows):
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plan, fine, coarse, rows = load_rows()
    enc0 = direct_candidates(ENC0_RECOMMENDATION)
    enc1 = direct_candidates(ENC1_RECOMMENDATION)
    best_direct_enc0 = min(float(item["historical_ate_mean_cm"]) for item in enc0.values())
    best_direct_enc1 = min(float(item["historical_ate_mean_cm"]) for item in enc1.values())
    ranked = sorted(rows, key=lambda row: row["ate"])
    best = ranked[0]
    positive = [row for row in rows if row["gain_vs_better_parent"] > 0]
    by_variant = {variant: [row for row in rows if row["variant"] == variant] for variant in ("A", "B")}
    pair_a = {(row["fine_rank"], row["coarse_rank"]): row for row in by_variant["A"]}
    pair_b = {(row["fine_rank"], row["coarse_rank"]): row for row in by_variant["B"]}
    deltas_b_minus_a = [pair_b[key]["ate"] - pair_a[key]["ate"] for key in pair_a]

    plot_variant_distribution(rows, best_direct_enc0)
    plot_heatmaps(rows, "ate", "ate_grid_heatmaps.png", "Full 6×6 C2F grid: primary ATE", "YlGnBu")
    plot_heatmaps(
        rows,
        "gain_vs_better_parent",
        "gain_vs_better_parent_heatmaps.png",
        "C2F gain relative to the better direct parent",
        "RdYlGn",
        center=0,
        fmt="+.2f",
    )
    plot_best_by_fine(rows, fine)

    per_fine_rows = []
    for rank in range(1, 7):
        subset_rows = [row for row in rows if row["fine_rank"] == rank]
        best_row = min(subset_rows, key=lambda row: row["ate"])
        per_fine_rows.append(
            [
                rank,
                format_channels(fine[rank]["channels"]),
                f"{fine[rank]['source_ate_mean_cm']:.4f}",
                f"{best_row['ate']:.4f}",
                best_row["variant"],
                best_row["coarse_rank"],
                f"{best_row['gain_vs_fine']:+.4f}",
            ]
        )
    per_coarse_rows = []
    for rank in range(1, 7):
        subset_rows = [row for row in rows if row["coarse_rank"] == rank]
        best_row = min(subset_rows, key=lambda row: row["ate"])
        per_coarse_rows.append(
            [
                rank,
                format_channels(coarse[rank]["channels"]),
                f"{coarse[rank]['source_ate_mean_cm']:.4f}",
                f"{best_row['ate']:.4f}",
                best_row["variant"],
                best_row["fine_rank"],
                f"{best_row['gain_vs_coarse']:+.4f}",
            ]
        )

    summary = {
        "dataset": str(RESULT_DIR),
        "completed": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "coverage_values": sorted(set(row["coverage"] for row in rows)),
        "runtime_seconds": stats([row["runtime"] for row in rows]),
        "variants": {variant: stats([row["ate"] for row in by_variant[variant]]) for variant in ("A", "B")},
        "best_direct_enc0_ate_cm": best_direct_enc0,
        "best_direct_enc1_ate_cm": best_direct_enc1,
        "best_c2f": best,
        "positive_synergy_count": len(positive),
        "positive_synergy_fraction": len(positive) / len(rows),
        "positive_synergy_rows": positive,
        "variant_pair_comparison": {
            "a_lower_count": sum(delta > 0 for delta in deltas_b_minus_a),
            "b_lower_count": sum(delta < 0 for delta in deltas_b_minus_a),
            "mean_b_minus_a_cm": statistics.mean(deltas_b_minus_a),
            "median_b_minus_a_cm": statistics.median(deltas_b_minus_a),
        },
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    variant_table = []
    for variant in ("A", "B"):
        values = [row["ate"] for row in by_variant[variant]]
        direct_wins = sum(row["gain_vs_better_parent"] > 0 for row in by_variant[variant])
        variant_table.append(
            [
                f"C2F-{variant}",
                len(values),
                f"{statistics.mean(values):.4f}",
                f"{statistics.median(values):.4f}",
                f"{min(values):.4f}",
                f"{max(values):.4f}",
                f"{direct_wins}/36",
            ]
        )
    positive_table = [
        [
            row["variant"],
            row["fine_rank"],
            format_channels(row["fine"]["channels"]),
            row["coarse_rank"],
            format_channels(row["coarse"]["channels"]),
            f"{row['ate']:.4f}",
            f"{row['better_parent']:.4f}",
            f"+{row['gain_vs_better_parent']:.4f}",
            f"{row['gain_vs_better_parent'] / row['better_parent'] * 100:.1f}%",
        ]
        for row in sorted(positive, key=lambda row: row["ate"])
    ]
    top_table = [
        [
            rank,
            row["variant"],
            row["fine_rank"],
            row["coarse_rank"],
            f"{row['ate']:.4f}",
            f"{row['ate_rmse']:.4f}",
            f"{row['hist_rpe']:.4f}",
            f"{row['se3_rmse']:.4f}",
            f"{row['gain_vs_better_parent']:+.4f}",
        ]
        for rank, row in enumerate(ranked[:10], start=1)
    ]
    direct_g3 = enc0["3,7,12"]
    direct_g6 = enc0["2,3,7,12,13,14"]
    report = f"""# U-Net Enc0/Enc1 C2F：fr1/desk_lightswitch 全序列验证

## 1. 结论摘要

本次 72 个 U-Net C2F cells 全部通过轨迹完整性门槛（**72/72 PASS**，每次均为 571/573 tracking poses，coverage = 99.65%）。因此 C2F 的价值不在于把原先失败的 U-Net 配置救活，而在于检验 Enc1 coarse 表征是否能给 Enc0 fine 表征带来额外精度。

最优 C2F 为 **C2F-A / Fine rank 2 `[d3,d7,d12]` / Coarse rank 4 `[d0,d5,d6,d18,d30]`**，primary ATE mean = **{best['ate']:.4f} cm**。它比自身 fine parent `{best['fine']['source_ate_mean_cm']:.4f} cm` 低 **{best['gain_vs_fine']:.4f} cm ({best['gain_vs_fine'] / best['fine']['source_ate_mean_cm'] * 100:.2f}%)**，也比所有 direct Enc0 结果的全局最优 `{best_direct_enc0:.4f} cm` 低 **{best_direct_enc0 - best['ate']:.4f} cm ({(best_direct_enc0 - best['ate']) / best_direct_enc0 * 100:.2f}%)**。

但这个全局最优改善很小，且 C2F-best 的 all-frame metric-scale SE(3) RMSE = **{best['se3_rmse']:.4f} cm**，高于 direct `[d3,d7,d12]` 的 **{direct_g3['allframe_se3_ate_rmse_mean_cm']:.4f} cm**；它的 historical RPE 则较低（{best['hist_rpe']:.4f} vs {direct_g3['historical_rpe_rmse_mean_cm']:.4f} cm）。因此它是主指标上的轻微胜出，而不是在所有指标上无条件支配 direct Enc0。

真正较明确的互补收益出现在非全局最优的 fine subset：Fine rank 5 `[d2,d3,d7,d12,d13]` 与 Coarse rank 4 的 C2F-A 从 **7.0959 cm** 降至 **6.1874 cm**，改善 **0.9085 cm / 12.8%**。这说明 coarse branch 更像有选择地帮助“尚未完全优化的”细粒度表征，而不会自动提升已经很强的 Enc0 G6。

## 2. 实验设置与可比性

- 数据：TUM `fr1/desk_lightswitch`，573 个 matched RGB-D timestamps。
- Tracking：新的 `unet_c2f`；对每帧共享一次 U-Net encoder forward，Enc1 作 coarse（H/2，32-channel universe），Enc0 作 fine（full resolution，16-channel universe）。
- Mapping：始终 gray，并保持 sensor/ground-truth depth；没有重新启用 mapping optimisation。因此此实验只改变 tracking feature allocation。
- Variant A：L0/L1 coarse + L2 fine；Variant B：L0 coarse + L1/L2 fine。
- Candidate grid：每个分支取已完成 direct greedy 中 6 个 repeated-PASS promising subsets，做完整 `2 × 6 × 6 = 72` cross-product；每 cell 一次运行。
- 主排序：historical keyframe `evo_ape tum --align --correct_scale` translation ATE mean，与原 full-sequence greedy 完全一致。保留 historical RPE、all-frame metric-scale SE(3) ATE/RPE、coverage 和 log 作为诊断。
- 注意：direct parent 的引用值为先前各 3 次重复的确定性结果；本次 C2F grid 为每 cell 一次。因此 <0.1 cm 的差异应被视为候选信号，后续应对 top C2F 和 top direct 做重复/跨序列确认。

## 3. 完成度与 variant 总览

{markdown_table(["Variant", "PASS", "ATE mean", "ATE median", "Best", "Worst", "beat better parent"], variant_table)}

![图1：两个 C2F variant 的 ATE 分布](figures/variant_ate_distribution.png)

**C2F-A 整体更合适。** 在相同的 36 个 fine/coarse pair 中，A 的 ATE 更低 **22/36** 次；B 更低 14/36 次。`B − A` 的平均差为 **{statistics.mean(deltas_b_minus_a):.4f} cm**、中位数 **{statistics.median(deltas_b_minus_a):.4f} cm**，均偏向 A。A 的均值/中位数也低于 B（8.5149/8.2973 vs 9.2549/9.1128 cm）。

![图2：完整 6×6 grid 的 primary ATE](figures/ate_grid_heatmaps.png)

## 4. C2F 在多少配置上带来真实提升？

为避免把“比很弱的另一分支好”误当作互补，我把每个 C2F cell 与其两个 direct parent 中 ATE 更低的那个比较。只有 C2F 的 ATE 同时低于 fine 和 coarse parent，才算真实的 pair-level improvement。

- **5/72（6.9%）** cells 超过其更强 direct parent；C2F-A 有 4 个，C2F-B 只有 1 个。
- C2F 相对 fine parent 有 **7/72** 次改善；相对 coarse parent 有 **14/72** 次改善。这再次说明 Enc1 coarse 本身通常较弱，而 C2F 并不会普遍改善当前最强 Enc0 subset。
- 其余 **67/72** cells 都不如其较强 direct parent：C2F-A 的中位数损失为 **{abs(statistics.median([row['gain_vs_better_parent'] for row in by_variant['A']])):.4f} cm**，C2F-B 为 **{abs(statistics.median([row['gain_vs_better_parent'] for row in by_variant['B']])):.4f} cm**。

{markdown_table(["Var.", "Fine rank", "Fine subset", "Coarse rank", "Coarse subset", "C2F ATE", "better parent", "gain", "relative"], positive_table)}

![图3：相对较强 direct parent 的增益；绿色为真正改善](figures/gain_vs_better_parent_heatmaps.png)

## 5. 哪些组合类型有效？

### 5.1 Coarse rank 4 是最稳定的互补 coarse branch

Coarse rank 4 = `[d0,d5,d6,d18,d30]`（单层 Enc1 ATE 7.6408 cm）出现在全局前十中的 **6/10**，并产生 5 个真正 pair-level improvements 中的 **4 个**。它不像 Enc1 rank 1/2 那样单层 ATE 最低，却最适合在 C2F 中提供 early-level guidance；这说明“单层最好”与“最适合作为 coarse initialiser”不是同一准则。

### 5.2 Fine rank 2 与 rank 5 最值得继续保留

Fine rank 2 `[d3,d7,d12]` 配 coarse rank 4 给出全局最佳 5.8765 cm。Fine rank 5 `[d2,d3,d7,d12,d13]` 的 direct ATE 原本较高，但与同一 coarse rank 4 配合后达到 6.1874 cm，是最大的绝对/相对互补收益。反过来，direct 全局最佳 Fine rank 1 `[d2,d3,d7,d12,d13,d14]` 的所有 12 个 C2F pairing 都变差，最佳仅为 6.5324 cm（比 direct 差 0.6116 cm）。

![图4：每个 fine subset 所能找到的最佳 C2F partner](figures/best_c2f_by_fine_subset.png)

### 5.3 C2F-A 优先，但不应将 B 完全丢弃

A 使用两个 coarse levels，整体和多数 matched pair 均优于 B。B 的最佳是 Fine rank 2 + Coarse rank 2，ATE 6.3454 cm；仍落后于 A-best 0.4689 cm。B 在少数 pair 上明显更好（例如 Fine rank 2 + Coarse rank 5），说明 switching point 与具体通道组合存在交互，而不是一个 universal rule；在下一阶段只需保留 A-best 为主、B-best 作为结构性对照。

## 6. 前十 C2F cells 与诊断

{markdown_table(["Rank", "Var.", "Fine", "Coarse", "ATE mean", "ATE RMSE", "hist RPE", "all-frame SE3 RMSE", "gain vs parent"], top_table)}

## 7. 建议与限制

1. **推荐的下一步候选**：优先保留 C2F-A Fine rank 2 + Coarse rank 4；同时保留 C2F-A Fine rank 5 + Coarse rank 4，因为它提供最清晰的互补机制证据。B 的 Fine rank 2 + Coarse rank 2 可作切换策略对照。
2. **不要宣称 C2F 已全面优于 U-Net direct tracking。** 最优主 ATE 的提升只有 0.75%，且 best C2F 并未在 all-frame SE(3) RMSE 上超过 direct Fine rank 2；72-cell grid 的大多数成员更差。
3. **最有价值的发现是结构选择性。** C2F 的 effect 依赖于 subset pairing：coarse rank 4 更适合 early-level initialization，而已有最优的 fine G6 反而不宜加入 coarse stages。这比“深层越多越好”更具体，也更可解释。
4. **确认实验**：在把结果推广到其他 sequence 前，应对上述三项候选至少各重复 3 次，并在 clean/flashlight/其他 lightswitch sequence 上比较；同一 full sequence 的确定性单次网格不构成泛化证据。

## 8. 可审计文件

- 本次 SQLite（权威记录）：`channel_selection_results/step_p_c2f_best_channels_evaluation/unet_fr1_desk_lightswitch/evaluations.sqlite3`
- 本次 console log：`.../unet_fr1_desk_lightswitch/console.log`
- 本次所有 rows：`.../unet_fr1_desk_lightswitch/all_evaluations.csv`
- 本次排名：`.../unet_fr1_desk_lightswitch/pass_ranking.csv`
- 冻结 grid：`channel_selection_pipeline/scripts/step_p_c2f_best_channels_evaluation/unet_c2f_candidate_plan.json`
- Direct Enc0/Enc1 source recommendations：`step_k_unet_enc0_direct_fullseq_greedy/recommendation.json` 与 `step_j_unet_direct_fullseq_greedy/recommendation.json`
"""
    (OUT / "UNet_C2F_fr1_desk_lightswitch_结果分析_中文.md").write_text(report, encoding="utf-8")
    print(f"Wrote report assets to {OUT}")


if __name__ == "__main__":
    main()
