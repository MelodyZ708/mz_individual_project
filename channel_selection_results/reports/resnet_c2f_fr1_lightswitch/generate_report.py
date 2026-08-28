#!/usr/bin/env python3
"""Generate the ResNet Conv1/Layer2 C2F fr1/desk_lightswitch report."""

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
PLAN_PATH = PROJECT_ROOT / "channel_selection_pipeline/scripts/step_p_c2f_best_channels_evaluation/resnet_c2f_candidate_plan.json"
RESULT_DIR = PROJECT_ROOT / "channel_selection_results/step_p_c2f_best_channels_evaluation/resnet_fr1_desk_lightswitch"
CONV1_REC = PROJECT_ROOT / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/recommendation.json"
LAYER2_REC = PROJECT_ROOT / "channel_selection_results/step_o_resnet_layer2_direct_fullseq_greedy/recommendation.json"
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"


def display(channels):
    return "[" + ",".join(f"d{value}" for value in channels) + "]"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def get_recommendation(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["candidate_key"]: item
        for item in payload["candidates"]
        if item.get("historical_ate_mean_cm") is not None
    }


def load_data():
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    fine = {int(item["rank"]): item for item in plan["fine_branch"]["candidates"]}
    coarse = {int(item["rank"]): item for item in plan["coarse_branch"]["candidates"]}
    pattern = re.compile(r"resnet_c2f_([ab])_fine(\d+)_coarse(\d+)$")
    rows = []
    with (RESULT_DIR / "all_evaluations.csv").open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            parsed = pattern.fullmatch(raw["label"])
            if parsed is None:
                raise ValueError(f"Unexpected C2F label: {raw['label']}")
            variant, fr, cr = parsed.group(1).upper(), int(parsed.group(2)), int(parsed.group(3))
            row = {
                "label": raw["label"], "variant": variant, "fr": fr, "cr": cr,
                "fine": fine[fr], "coarse": coarse[cr], "status": raw["status"],
                "reason": raw["reason"], "runtime": float(raw["elapsed_seconds"]),
            }
            if raw["status"] == "PASS":
                row.update(
                    ate=float(raw["historical_evo_ape_mean_m"]) * 100,
                    ate_rmse=float(raw["historical_evo_ape_rmse_m"]) * 100,
                    hist_rpe=float(raw["historical_evo_rpe_rmse_m"]) * 100,
                    se3_rmse=float(raw["se3_ate_rmse_m"]) * 100,
                    se3_mean=float(raw["se3_ate_mean_m"]) * 100,
                    trans_rpe_max=float(raw["translation_rpe_max_m"]) * 100,
                    rot_rpe_max=float(raw["rotation_rpe_max_deg"]),
                    coverage=float(raw["coverage_ratio"]),
                )
                row["better_parent"] = min(float(fine[fr]["source_ate_mean_cm"]), float(coarse[cr]["source_ate_mean_cm"]))
                row["gain"] = row["better_parent"] - row["ate"]
                row["gain_fine"] = float(fine[fr]["source_ate_mean_cm"]) - row["ate"]
                row["gain_coarse"] = float(coarse[cr]["source_ate_mean_cm"]) - row["ate"]
            rows.append(row)
    if len(rows) != 72:
        raise ValueError(f"Expected 72 C2F cells, got {len(rows)}")
    return plan, fine, coarse, rows


def cell_matrix(rows, variant, field):
    data = np.full((6, 6), np.nan)
    for row in rows:
        if row["variant"] == variant and field in row:
            data[row["fr"] - 1, row["cr"] - 1] = row[field]
    return data


def plot_distribution(passes, direct_best):
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    values = [[row["ate"] for row in passes if row["variant"] == name] for name in ("A", "B")]
    box = ax.boxplot(values, tick_labels=["C2F-A", "C2F-B"], patch_artist=True)
    for patch, color in zip(box["boxes"], ("#59A14F", "#4E79A7")):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    rng = np.random.default_rng(20260823)
    for index, values_for_variant in enumerate(values, 1):
        ax.scatter(index + rng.uniform(-0.10, 0.10, len(values_for_variant)), values_for_variant,
                   s=21, alpha=0.70, color="#202A35")
    ax.axhline(direct_best, color="#D62728", linestyle="--", linewidth=1.5,
               label=f"best direct Conv1 = {direct_best:.4f} cm")
    ax.set_title("ResNet C2F ATE distribution (PASS cells only)")
    ax.set_ylabel("Historical keyframe ATE mean (cm)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.savefig(FIG / "variant_ate_distribution.png", dpi=220)
    plt.close(fig)


def plot_grid(rows, field, filename, title, gain=False):
    matrices = [cell_matrix(rows, variant, field) for variant in ("A", "B")]
    if gain:
        bound = max(abs(float(np.nanmin(data))) for data in matrices)
        bound = max(bound, max(abs(float(np.nanmax(data))) for data in matrices))
        vmin, vmax, cmap = -bound, bound, "RdYlGn"
    else:
        vmin = min(float(np.nanmin(data)) for data in matrices)
        vmax = max(float(np.nanmax(data)) for data in matrices)
        cmap = "YlGnBu"
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 5.35), constrained_layout=True)
    image = None
    for ax, data, variant in zip(axes, matrices, ("A", "B")):
        image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"C2F-{variant}")
        ax.set_xlabel("Layer2 coarse subset rank")
        ax.set_ylabel("Conv1 fine subset rank")
        ax.set_xticks(range(6), range(1, 7))
        ax.set_yticks(range(6), range(1, 7))
        for i in range(6):
            for j in range(6):
                value = data[i, j]
                if np.isnan(value):
                    ax.text(j, i, "NaN\nFAIL", ha="center", va="center", color="#D62728", fontsize=8, fontweight="bold")
                    continue
                if gain:
                    text = f"{value:+.2f}"
                    text_color = "white" if abs(value) > bound * 0.58 else "black"
                else:
                    text = f"{value:.2f}"
                    text_color = "white" if value > (vmin + vmax) / 2 else "black"
                ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=8.4)
    label = "ATE mean (cm)" if not gain else "gain over better direct parent (cm; positive = C2F helps)"
    fig.colorbar(image, ax=axes, shrink=0.86, label=label)
    fig.suptitle(title, fontsize=13)
    fig.savefig(FIG / filename, dpi=220)
    plt.close(fig)


def plot_best_by_fine(passes, fine):
    direct = [float(fine[index]["source_ate_mean_cm"]) for index in range(1, 7)]
    best_rows = [min((row for row in passes if row["fr"] == index), key=lambda row: row["ate"]) for index in range(1, 7)]
    c2f = [row["ate"] for row in best_rows]
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(9.7, 4.9), constrained_layout=True)
    width = 0.38
    ax.bar(x - width / 2, direct, width, label="direct Conv1", color="#B9C6E4")
    bars = ax.bar(x + width / 2, c2f, width, label="best C2F partner", color="#4E79A7")
    for source, best, bar in zip(direct, c2f, bars):
        gain = source - best
        ax.text(bar.get_x() + bar.get_width() / 2, best + 0.20, f"{gain:+.2f}", ha="center", va="bottom",
                fontsize=8, color="#148A3D" if gain > 0 else "#B22222")
    ax.set_xticks(x, [f"F{i}\n{display(fine[i]['channels'])}" for i in range(1, 7)], fontsize=8)
    ax.set_ylabel("Historical keyframe ATE mean (cm; lower is better)")
    ax.set_title("Best C2F partner found for every Conv1 fine subset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(FIG / "best_c2f_by_fine.png", dpi=220)
    plt.close(fig)


def summary_stats(values):
    return [f"{statistics.mean(values):.4f}", f"{statistics.median(values):.4f}", f"{min(values):.4f}", f"{max(values):.4f}"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plan, fine, coarse, rows = load_data()
    conv1 = get_recommendation(CONV1_REC)
    layer2 = get_recommendation(LAYER2_REC)
    passes = [row for row in rows if row["status"] == "PASS"]
    failures = [row for row in rows if row["status"] != "PASS"]
    by_variant = {variant: [row for row in passes if row["variant"] == variant] for variant in ("A", "B")}
    direct_best_conv1 = min(float(item["historical_ate_mean_cm"]) for item in conv1.values())
    direct_best_layer2 = min(float(item["historical_ate_mean_cm"]) for item in layer2.values())
    best = min(passes, key=lambda row: row["ate"])
    strict_wins = [row for row in passes if row["gain"] > 0]
    ranked = sorted(passes, key=lambda row: row["ate"])
    matched_a = {(row["fr"], row["cr"]): row for row in by_variant["A"]}
    matched_b = {(row["fr"], row["cr"]): row for row in by_variant["B"]}
    common = sorted(set(matched_a) & set(matched_b))
    delta_b_minus_a = [matched_b[key]["ate"] - matched_a[key]["ate"] for key in common]
    legacy_gate_safe = [row for row in passes if row["trans_rpe_max"] <= 6.0 and row["rot_rpe_max"] <= 5.0]

    plot_distribution(passes, direct_best_conv1)
    plot_grid(rows, "ate", "ate_grid_heatmaps.png", "Full 6×6 ResNet C2F grid: primary ATE")
    plot_grid(rows, "gain", "gain_vs_better_parent_heatmaps.png", "ResNet C2F gain relative to better direct parent", gain=True)
    plot_best_by_fine(passes, fine)

    variant_rows = []
    for variant in ("A", "B"):
        items = by_variant[variant]
        values = [row["ate"] for row in items]
        variant_rows.append([
            f"C2F-{variant}", f"{len(items)}/36", *summary_stats(values),
            f"{sum(row['gain'] > 0 for row in items)}/{len(items)}",
        ])
    positive_rows = [
        [
            row["variant"], row["fr"], display(row["fine"]["channels"]), row["cr"], display(row["coarse"]["channels"]),
            f"{row['ate']:.4f}", f"{row['better_parent']:.4f}", f"+{row['gain']:.4f}",
            f"{row['gain'] / row['better_parent'] * 100:.1f}%",
        ]
        for row in sorted(strict_wins, key=lambda row: row["ate"])
    ]
    top_rows = [
        [
            rank, row["variant"], row["fr"], row["cr"], f"{row['ate']:.4f}", f"{row['hist_rpe']:.4f}",
            f"{row['se3_rmse']:.4f}", f"{row['trans_rpe_max']:.2f}", f"{row['rot_rpe_max']:.2f}", f"{row['gain']:+.4f}",
        ]
        for rank, row in enumerate(ranked[:10], 1)
    ]
    failure_rows = [
        [row["variant"], row["fr"], display(row["fine"]["channels"]), row["cr"], display(row["coarse"]["channels"]), f"{row['runtime']:.1f}"]
        for row in failures
    ]
    per_fine_rows = []
    for rank in range(1, 7):
        items = [row for row in passes if row["fr"] == rank]
        best_row = min(items, key=lambda row: row["ate"])
        expected = 12
        per_fine_rows.append([
            rank, display(fine[rank]["channels"]), f"{fine[rank]['source_ate_mean_cm']:.4f}",
            f"{best_row['ate']:.4f}", best_row["variant"], best_row["cr"], f"{best_row['gain_fine']:+.4f}",
            f"{len(items)}/{expected}",
        ])

    direct_fine2 = conv1["23,24,26,51,63"]
    direct_global = conv1["15,20,26,34"]
    report = f"""# ResNet Conv1/Layer2 C2F：fr1/desk_lightswitch 全序列验证

## 1. 结论摘要

ResNet C2F 的效果比 U-Net 更明显，但不是无条件稳健：72 个 cells 中 **66 PASS、6 个 `FAIL_TRACKING_NAN`**。在通过的 C2F pair 中，有 **20 个** 同时优于其 Conv1 fine 与 Layer2 coarse direct parent；按全部 72 个预注册 cells 计为 **20/72 = 27.8%**，明显高于 U-Net 的 5/72。

最优主指标配置为 **C2F-B / Fine rank 2 `[d23,d24,d26,d51,d63]` / Coarse rank 4 `[d67,d108,d121]`**：ATE mean = **{best['ate']:.4f} cm**。这比自身 fine direct parent `{best['fine']['source_ate_mean_cm']:.4f} cm` 低 **{best['gain_fine']:.4f} cm ({best['gain_fine'] / best['fine']['source_ate_mean_cm'] * 100:.2f}%)**；也比现有 direct Conv1 全局最优 `[d15,d20,d26,d34]` 的 **{direct_best_conv1:.4f} cm** 低 **{direct_best_conv1 - best['ate']:.4f} cm ({(direct_best_conv1 - best['ate']) / direct_best_conv1 * 100:.2f}%)**。

与 Fine rank 2 direct parent 相比，C2F-best 的 secondary diagnostics 也一致改善：historical RPE **{best['hist_rpe']:.4f} vs {direct_fine2['historical_rpe_rmse_mean_cm']:.4f} cm**，all-frame metric-scale SE(3) RMSE **{best['se3_rmse']:.4f} vs {direct_fine2['allframe_se3_ate_rmse_mean_cm']:.4f} cm**。相对 direct Conv1 global best，C2F-best 的 historical RPE 也略低（{best['hist_rpe']:.4f} vs {direct_global['historical_rpe_rmse_mean_cm']:.4f} cm），all-frame SE(3) RMSE 亦较低（{best['se3_rmse']:.4f} vs {direct_global['allframe_se3_ate_rmse_mean_cm']:.4f} cm）。因此这不是只靠一个 ATE 数字得到的优势。

最大的 pair-level gain 则来自 **C2F-A / Fine rank 6 `[d29,d33,d52]` + Coarse rank 5 `[d108,d121]`**：从较强 parent 的 14.6721 cm 降至 9.8490 cm，改善 **4.8231 cm / 32.9%**。这说明 C2F 的主要贡献是将部分中等强度 fine subset 与合适的 coarse initializer 配对，而不是简单为原本最好的 Conv1 set 增加 Layer2。

## 2. 实验设置与可比性

- 数据：TUM `fr1/desk_lightswitch`，573 个 matched RGB-D timestamps。
- Tracking：`cnn_c2f`。Conv1 作 fine branch（64-channel universe）；Layer2 作 coarse branch（128-channel universe）。
- Mapping：固定 gray，固定 sensor/ground-truth depth；只改变 tracking feature allocation。
- C2F-A：L0/L1 coarse + L2 fine；C2F-B：L0 coarse + L1/L2 fine。
- Candidate grid：Conv1 与 Layer2 各取完成 direct greedy 中 ATE 最低的 6 个 repeated-PASS subset，执行完整 `2 × 6 × 6 = 72` pairing；每个 cell 一次。
- 主指标：historical keyframe `evo_ape tum --align --correct_scale` translation ATE mean，与前序 full-sequence greedy 一致。保留 historical RPE、all-frame metric-scale SE(3) ATE/RPE、coverage、NaN fail detection 和原始 log。
- 本次通过的 run 都达到 571/573 poses，coverage = 99.65%。与原 direct parents 相比，C2F grid 是单次运行；因此最好的候选仍需重复和跨序列确认。

## 3. 完成度、variant 分布与稳定性

{table(["Variant", "PASS", "ATE mean", "ATE median", "Best", "Worst", "beat better parent"], variant_rows)}

![图1：A/B 的 PASS ATE 分布](figures/variant_ate_distribution.png)

**B 整体略优于 A。** 在两个 variant 都成功的 32 个 matched pairs 中，B 的 ATE 更低 **19/32** 次，A 更低 13/32 次；`B − A` 的均值为 **{statistics.mean(delta_b_minus_a):.4f} cm**、中位数为 **{statistics.median(delta_b_minus_a):.4f} cm**，均偏向 B。A 的成功率较低（32/36），B 为 34/36。

## 4. C2F 在多少、哪种配置上确实提升？

这里的“提升”采用严格定义：C2F ATE 必须同时低于同一 cell 所含 Conv1 fine 和 Layer2 coarse 两个 direct parent。这样不会把“仅好于较弱的 Layer2 branch”误计为互补。

- **20/72（27.8%）** cells 达到严格改善；若只看完成的轨迹则为 20/66（30.3%）。
- C2F-B 有 12 个严格改善，C2F-A 有 8 个；但最大绝对提升来自 A。
- 相对 fine parent 有 23/66 次改善，相对 coarse parent 有 41/66 次改善。C2F 对 Layer2 的精细化帮助很常见，但真正同时超过强 fine parent 的情况仍只有约三成。

{table(["Var.", "Fine rank", "Fine subset", "Coarse rank", "Coarse subset", "C2F ATE", "better parent", "gain", "relative"], positive_rows)}

![图2：完整 grid 的 primary ATE；红字为 NaN fail](figures/ate_grid_heatmaps.png)

![图3：相对较强 direct parent 的 gain；绿色才是严格 C2F 改善](figures/gain_vs_better_parent_heatmaps.png)

## 5. 有效 pairing 的结构规律

### 5.1 Coarse rank 4/5 是最有价值的切换端 representation

Coarse rank 4 `[d67,d108,d121]` 与 rank 5 `[d108,d121]` 都产生 **12/12 PASS**，且分别在 **12/12** 个 paired cells 中优于它们较弱的 coarse direct baseline。它们没有 Layer2 direct ATE 的前两名那么好，却成为最有效的 C2F initializer：全局最优使用 rank 4，而最大的 32.9% gain 使用 rank 5。相反，Layer2 rank 1/2 虽然 single-layer ATE 最低，却是全部 NaN interaction failures 的唯一 coarse members。

### 5.2 C2F 对中等 fine subset 的增益最大

{table(["Fine rank", "Conv1 subset", "direct ATE", "best C2F", "Var.", "coarse rank", "gain vs fine", "PASS cells"], per_fine_rows)}

Fine rank 6 `[d29,d33,d52]`、rank 5 `[d5,d6,d24,d29]` 和 rank 4 `[d33,d52]` 可获得很大的 C2F improvement；反之 Direct Conv1 global best rank 1 只获得 0.2555 cm 的小幅改善。C2F 的益处因此更像“basin/initialisation rescue”，不是所有 fine feature 的普遍 refinement。

![图4：每个 Conv1 fine subset 的最佳 C2F partner](figures/best_c2f_by_fine.png)

## 6. 失败模式与局部稳定性提醒

6 个失败均为 tracker non-finite diagnostics，而非 timeout 或缺轨迹；它们集中在 **Layer2 coarse rank 1/2 + Fine rank 4/5/6** 的组合。所有 rank 3--6 coarse subsets 均完成。这是 pair interaction，而不是某一个 parent 本身失败：这些 source subsets 均来自 direct greedy 的 repeated PASS results。

{table(["Var.", "Fine rank", "Fine subset", "Coarse rank", "Coarse subset", "runtime (s)"], failure_rows)}

此外，如果机械地把早期 **40-frame MVS** 的 `translation RPE max ≤ 6 cm` 与 `rotation RPE max ≤ 5°` 门槛套到这个 573-frame C2F full-sequence run，只有 2/66 PASS cells 会同时满足。该门槛从未被定义为 full-sequence C2F 的硬筛选规则，因此不应据此推翻 primary ranking；但它提醒我们，best primary ATE candidates 的局部 maximum RPE 仍需在 trajectory visualization 与重复运行中复核。

## 7. 前十 primary ATE 与诊断

{table(["Rank", "Var.", "Fine", "Coarse", "ATE mean", "hist RPE", "all-frame SE3", "T-RPE max", "R-RPE max", "gain"], top_rows)}

## 8. 建议

1. **ResNet C2F 值得继续。** 与 U-Net C2F 的小幅、稀疏收益不同，ResNet 给出了 10.1% 的新 global best，以及 20 个严格 parent-level improvements。
2. **主候选**：C2F-B Fine rank 2 `[d23,d24,d26,d51,d63]` + Coarse rank 4 `[d67,d108,d121]`。它相对自身 parent、direct Conv1 global best、historical RPE 和 all-frame SE(3) RMSE 都更好。
3. **机制候选**：C2F-A Fine rank 6 `[d29,d33,d52]` + Coarse rank 5 `[d108,d121]`。它不是最优 ATE，但 32.9% 改善最能证明 C2F 能帮助较弱 fine subset 进入更好 basin；应与主候选一并保留。
4. **避免区域**：Layer2 coarse rank 1/2 配 Fine rank 4--6。它们产生了所有 NaN failures，不能因 parent 分别优秀而假定可安全结合。
5. **确认实验**：先对主候选、机制候选、direct Conv1 G4 各至少重复 3 次，并保留 full trajectory/RPE visualization；再进入其他 lighting sequences，不能仅用这一条 sequence 宣称普遍优势。

## 9. 可审计文件

- SQLite：`channel_selection_results/step_p_c2f_best_channels_evaluation/resnet_fr1_desk_lightswitch/evaluations.sqlite3`
- Console：`.../resnet_fr1_desk_lightswitch/console.log`
- 全部 rows：`.../resnet_fr1_desk_lightswitch/all_evaluations.csv`
- 排名：`.../resnet_fr1_desk_lightswitch/pass_ranking.csv`
- 冻结 grid：`channel_selection_pipeline/scripts/step_p_c2f_best_channels_evaluation/resnet_c2f_candidate_plan.json`
- Direct sources：`step_n_resnet_conv1_direct_fullseq_greedy/recommendation.json` 与 `step_o_resnet_layer2_direct_fullseq_greedy/recommendation.json`
"""
    (OUT / "ResNet_C2F_fr1_desk_lightswitch_结果分析_中文.md").write_text(report, encoding="utf-8")
    summary = {
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "best_c2f": {key: best[key] for key in ("label", "variant", "fr", "cr", "ate", "gain", "se3_rmse", "hist_rpe")},
        "strict_pair_improvement": {"count": len(strict_wins), "of_planned": len(rows), "of_pass": len(strict_wins) / len(passes)},
        "legacy_mvs_gate_safe_count": len(legacy_gate_safe),
        "matched_variant_comparison": {"pairs": len(common), "a_better": sum(delta > 0 for delta in delta_b_minus_a), "b_better": sum(delta < 0 for delta in delta_b_minus_a)},
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote report assets to {OUT}")


if __name__ == "__main__":
    main()
