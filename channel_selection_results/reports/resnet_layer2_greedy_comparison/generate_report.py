#!/usr/bin/env python3
"""Generate the Layer2-versus-Conv1 greedy report from authoritative SQLite logs."""

from __future__ import annotations

import csv
import json
import math
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

_CACHE = Path(tempfile.gettempdir()) / "resnet_layer2_report_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parents[2]
OUTPUT_MD = REPORT_DIR / "ResNet_Layer2_Greedy与Conv1_Greedy对比_中文.md"
OUTPUT_JSON = REPORT_DIR / "layer2_conv1_distribution_summary.json"
SOURCES = {
    "ResNet Layer2 greedy": {
        "path": PROJECT_ROOT
        / "channel_selection_results/step_o_resnet_layer2_direct_fullseq_greedy/evaluations.sqlite3",
        "paths": PROJECT_ROOT
        / "channel_selection_results/step_o_resnet_layer2_direct_fullseq_greedy/direct_greedy_path.csv",
        "recommendation": PROJECT_ROOT
        / "channel_selection_results/step_o_resnet_layer2_direct_fullseq_greedy/recommendation.json",
        "console": PROJECT_ROOT
        / "channel_selection_results/step_o_resnet_layer2_direct_fullseq_greedy/console.log",
        "short": "ResNet Layer2",
        "channels": 128,
        "color": "#a65b23",
    },
    "ResNet Conv1 greedy": {
        "path": PROJECT_ROOT
        / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/evaluations.sqlite3",
        "paths": PROJECT_ROOT
        / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/direct_greedy_path.csv",
        "recommendation": PROJECT_ROOT
        / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/recommendation.json",
        "console": PROJECT_ROOT
        / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/console.log",
        "short": "ResNet Conv1",
        "channels": 64,
        "color": "#3b6fb6",
    },
}


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def cm(value: float | None, digits: int = 4) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def display(key: str) -> str:
    return "[" + ",".join(f"d{item}" for item in key.split(",")) + "]"


def read_source(name: str, source: dict[str, Any]) -> dict[str, Any]:
    for key in ("path", "paths", "recommendation", "console"):
        require(source[key])
    with sqlite3.connect(source["path"]) as database:
        database.row_factory = sqlite3.Row
        rows = database.execute("SELECT * FROM evaluations WHERE replicate=0 ORDER BY id").fetchall()
        total_saved = database.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
        stage_rows = database.execute(
            "SELECT stage, COUNT(*) AS rows, COUNT(DISTINCT candidate_key) AS unique_candidates "
            "FROM stage_candidates GROUP BY stage ORDER BY stage"
        ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["cardinality"] = (
            len(json.loads(record["channels_json"])) if record["channels_json"] else None
        )
        record["ate_cm"] = (
            float(record["historical_evo_ape_mean_m"]) * 100
            if record["status"] == "PASS" and record["historical_evo_ape_mean_m"] is not None
            else None
        )
        records.append(record)
    values = [record["ate_cm"] for record in records if record["ate_cm"] is not None]
    values_array = np.asarray(values, dtype=float)
    statuses = Counter(record["status"] for record in records)
    by_k: dict[int, dict[str, float | int | None]] = {}
    for cardinality in range(1, 7):
        subset = [record for record in records if record["cardinality"] == cardinality]
        passing = [record["ate_cm"] for record in subset if record["ate_cm"] is not None]
        by_k[cardinality] = {
            "evaluated": len(subset),
            "pass": len(passing),
            "pass_rate": len(passing) / len(subset) if subset else None,
            "best_cm": min(passing) if passing else None,
            "median_cm": float(np.median(passing)) if passing else None,
        }
    return {
        "name": name,
        "short": source["short"],
        "channels": source["channels"],
        "color": source["color"],
        "records": records,
        "values": values,
        "total": len(records),
        "total_saved": total_saved,
        "pass": len(values),
        "fail": sum(count for status, count in statuses.items() if status.startswith("FAIL")),
        "error": sum(count for status, count in statuses.items() if status.startswith("ERROR")),
        "pass_rate": len(values) / len(records),
        "quantiles": {
            label: float(value)
            for label, value in zip(
                ("min", "p05", "p25", "median", "p75", "p95", "max"),
                np.percentile(values_array, (0, 5, 25, 50, 75, 95, 100)),
            )
        },
        "by_k": by_k,
        "stage_counts": [dict(row) for row in stage_rows],
        "recommendation": json.loads(source["recommendation"].read_text(encoding="utf-8")),
    }


def read_paths(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not row["single_run_primary_ate_mean_cm"]:
            continue
        cardinality = int(row["cardinality"])
        current = {
            "ate_cm": float(row["single_run_primary_ate_mean_cm"]),
            "channels": json.loads(row["channels"]),
            "seed": json.loads(row["seed_channels"]),
        }
        if cardinality not in best or current["ate_cm"] < best[cardinality]["ate_cm"]:
            best[cardinality] = current
    return best


def save_histograms(datasets: list[dict[str, Any]]) -> None:
    bins = np.r_[np.arange(0, 81, 2), np.inf]
    figure, axis = plt.subplots(figsize=(10.2, 5.7))
    for dataset in datasets:
        values = np.asarray(dataset["values"], dtype=float)
        axis.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.5,
            color=dataset["color"],
            label=f"{dataset['short']} (n={len(values)}, median={np.median(values):.2f} cm)",
        )
    axis.set_xlim(8, 80)
    axis.set_xlabel("Historical keyframe ATE mean (cm; lower is better)")
    axis.set_ylabel("Probability density (PASS candidates only)")
    axis.set_title("ResNet greedy ATE distributions: Layer2 versus Conv1")
    axis.grid(alpha=0.25)
    axis.legend(frameon=True)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "ate_histogram_normalized_layer2_vs_conv1.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.9), sharex=True)
    for axis, dataset in zip(axes, datasets):
        values = np.asarray(dataset["values"], dtype=float)
        overflow = int(np.sum(values > 80))
        axis.hist(values[values <= 80], bins=np.arange(8, 82, 2), color=dataset["color"], alpha=0.82, edgecolor="white", linewidth=0.3)
        axis.axvline(np.median(values), color="#202020", linestyle="--", linewidth=1.7, label=f"median {np.median(values):.2f}")
        axis.axvline(np.min(values), color="#a71919", linestyle=":", linewidth=2, label=f"best {np.min(values):.2f}")
        axis.set_title(dataset["short"])
        axis.set_xlabel("ATE mean (cm)")
        axis.set_xlim(8, 80)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=9)
        axis.text(0.98, 0.96, f"PASS n={len(values)}\n>80 cm: {overflow}", transform=axis.transAxes, ha="right", va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "#bdbdbd", "alpha": 0.9})
    axes[0].set_ylabel("Number of PASS candidates")
    figure.suptitle("Per-layer ATE histograms (unique replicate-0 candidates)", y=1.02, fontsize=15)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "ate_histogram_faceted_layer2_vs_conv1.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_ecdf(datasets: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(9.3, 5.4))
    for dataset in datasets:
        values = np.sort(np.asarray(dataset["values"], dtype=float))
        axis.step(values, np.arange(1, len(values) + 1) / len(values), where="post", linewidth=2.5, color=dataset["color"], label=dataset["short"])
    axis.set_xlim(8, 80)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("Historical keyframe ATE mean (cm; lower is better)")
    axis.set_ylabel("Empirical CDF of PASS candidates")
    axis.set_title("Layer2 and Conv1 greedy-search ATE ECDF")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "ate_ecdf_layer2_vs_conv1.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_outcomes(datasets: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 4.8))
    positions = np.arange(len(datasets))
    passing = [dataset["pass"] for dataset in datasets]
    failing = [dataset["fail"] + dataset["error"] for dataset in datasets]
    axis.bar(positions, passing, color="#4f9b72", label="PASS")
    axis.bar(positions, failing, bottom=passing, color="#c65d5d", label="FAIL / ERROR")
    for index, dataset in enumerate(datasets):
        axis.text(index, passing[index] + failing[index] + 55, percentage(dataset["pass_rate"]), ha="center", va="bottom", fontsize=10)
    axis.set_xticks(positions, [dataset["short"] for dataset in datasets])
    axis.set_ylabel("Unique replicate-0 configurations")
    axis.set_title("Completion outcome of evaluated greedy candidates")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "pass_failure_layer2_vs_conv1.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_k_profile(datasets: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.9, 4.45))
    for dataset in datasets:
        keys = range(1, 7)
        axes[0].plot(keys, [dataset["by_k"][key]["pass_rate"] * 100 for key in keys], marker="o", linewidth=2.2, color=dataset["color"], label=dataset["short"])
        axes[1].plot(keys, [dataset["by_k"][key]["median_cm"] for key in keys], marker="o", linewidth=2.2, color=dataset["color"], label=dataset["short"])
    axes[0].set_title("PASS rate by selected-channel count")
    axes[0].set_xlabel("K selected channels")
    axes[0].set_ylabel("PASS rate (%)")
    axes[0].set_xticks(range(1, 7))
    axes[0].set_ylim(-2, 102)
    axes[1].set_title("Median PASS ATE by selected-channel count")
    axes[1].set_xlabel("K selected channels")
    axes[1].set_ylabel("Historical keyframe ATE mean (cm)")
    axes[1].set_xticks(range(1, 7))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "cardinality_profile_layer2_vs_conv1.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_paths(paths: dict[str, dict[int, dict[str, Any]]]) -> None:
    figure, axis = plt.subplots(figsize=(8.7, 5.0))
    for name, values in paths.items():
        keys = sorted(values)
        source = SOURCES[name]
        axis.plot(keys, [values[key]["ate_cm"] for key in keys], marker="o", linewidth=2.4, color=source["color"], label=source["short"])
    axis.set_xlabel("K selected channels")
    axis.set_ylabel("Best direct-greedy endpoint ATE mean (cm)")
    axis.set_xticks(range(1, 7))
    axis.set_title("Best direct-greedy endpoint by K")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "greedy_path_layer2_vs_conv1.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def table(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    layer2 = read_source("ResNet Layer2 greedy", SOURCES["ResNet Layer2 greedy"])
    conv1 = read_source("ResNet Conv1 greedy", SOURCES["ResNet Conv1 greedy"])
    datasets = [layer2, conv1]
    paths = {name: read_paths(source["paths"]) for name, source in SOURCES.items()}
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                dataset["name"]: {key: value for key, value in dataset.items() if key not in ("records", "recommendation")}
                for dataset in datasets
            }
            | {"best_paths": paths},
            indent=2,
        ),
        encoding="utf-8",
    )
    save_histograms(datasets)
    save_ecdf(datasets)
    save_outcomes(datasets)
    save_k_profile(datasets)
    save_paths(paths)

    layer2_rec = layer2["recommendation"]
    conv1_rec = conv1["recommendation"]
    layer2_gstar = layer2_rec["adaptive_gstar_after_three_repeats"]
    layer2_g4 = layer2_rec["fixed_g4_after_three_repeats"]
    conv1_gstar = conv1_rec["adaptive_gstar_after_three_repeats"]
    layer2_g6 = next(item for item in layer2_rec["candidates"] if item["tags"] == "G6")
    layer2_all = next(item for item in layer2_rec["candidates"] if item["channels"] == "all128")
    conv1_all = next(item for item in conv1_rec["candidates"] if item["channels"] == "all64")
    g4_candidate = next(item for item in layer2_rec["candidates"] if item["tags"] == "G4")
    g5_candidate = next(item for item in layer2_rec["candidates"] if "G5" in item["tags"])
    g4_vs_g5 = (g4_candidate["historical_ate_mean_cm"] / g5_candidate["historical_ate_mean_cm"]) - 1.0
    layer2_vs_conv1 = (layer2_gstar["historical_ate_mean_cm"] / conv1_gstar["historical_ate_mean_cm"]) - 1.0
    layer2_reduction = 1.0 - layer2_gstar["historical_ate_mean_cm"] / layer2_all["historical_ate_mean_cm"]
    conv1_reduction = 1.0 - conv1_gstar["historical_ate_mean_cm"] / conv1_all["historical_ate_mean_cm"]

    lines = [
        "---",
        'title: "ResNet Layer2 直接全序列 Greedy 搜索结果：与 Conv1 Greedy 的比较"',
        'subtitle: "fr1/desk_lightswitch，573 frames；报告日期：2026-08-21"',
        "lang: zh-CN",
        "---",
        "",
        "# ResNet Layer2 直接全序列 Greedy 搜索结果：与 Conv1 Greedy 的比较",
        "",
        "*完整 fr1/desk_lightswitch（573 matched timestamps）；报告日期：2026-08-21*",
        "",
        "# 执行摘要",
        "",
        f"本次将 ResNet-18 **Layer2 的 128 个 post-ReLU channels** 按与 Conv1/UNet 相同的 direct full-sequence greedy 协议搜索。最终主指标最优为 **G5 `[d41,d60,d67,d108,d121]` = {layer2_gstar['historical_ate_mean_cm']:.4f} cm，3/3 PASS**。固定四通道 G4 `[d60,d67,d108,d121]` 为 {layer2_g4['historical_ate_mean_cm']:.4f} cm，仅比 G5 高 {g4_vs_g5 * 100:.2f}%，但全帧 SE(3) ATE RMSE（{g4_candidate['allframe_se3_ate_rmse_mean_cm']:.2f} vs {g5_candidate['allframe_se3_ate_rmse_mean_cm']:.2f} cm）和历史 RPE RMSE（{g4_candidate['historical_rpe_rmse_mean_cm']:.4f} vs {g5_candidate['historical_rpe_rmse_mean_cm']:.4f} cm）均更低。因此 G5 是按既定 primary metric 的 accuracy-first 选择，G4 是几乎无主指标损失且 secondary diagnostics 更好的 compact/balanced 选择。",
        "",
        f"与已完成的 Conv1 greedy 相比，Layer2 的最佳主 ATE 高 {layer2_vs_conv1 * 100:.1f}%（12.6100 vs {conv1_gstar['historical_ate_mean_cm']:.4f} cm）；Conv1 仍是这个 lightswitch 序列上的 ResNet accuracy-first 层级。Layer2 的失败结构却更友好：15/128 个 singleton 可完成、整体已评估组合 PASS 率为 {percentage(layer2['pass_rate'])}，而 Conv1 singleton 为 0/64、整体 PASS 率为 {percentage(conv1['pass_rate'])}。这说明‘更容易保持 tracking’与‘最终 ATE 更低’并不是同一性质。",
        "",
        "报告包含基于 replicate-0 唯一候选的 ATE 直方图、归一化分布图、ECDF、PASS/FAIL 图、按 K 的剖面和 greedy path。完整 console、SQLite 和 direct path CSV 仍被索引，满足导师对分布而非仅最佳值的审计需求。",
        "",
        "# 1. 共同协议与可比性",
        "",
        table(["项目", "设置"]),
        table(["---", "---"]),
        table(["序列", "完整 `rgbd_dataset_freiburg1_desk_lightswitch`，573 个 matched RGB-D timestamps"]),
        table(["Tracking feature", "ResNet-18 Conv1（64）或 Layer2（128）指定 post-ReLU channels；`cnn_only`"]),
        table(["Layer2 resolution", "原生 H/8 × W/8；按 COMO 既有 Layer2 extractor 进行 x8 tracking upsample"]),
        table(["Mapping", "固定 gray + matched sensor depth；GT pose 仅用于运行后轨迹指标"]),
        table(["主排名", "keyframe `evo_ape --align --correct_scale` translation ATE mean（cm，越低越好）"]),
        table(["终止规则", "NaN/Inf tracking diagnostics 立即 failure；timeout 300 s；coverage ≥90%"]),
        table(["分布统计", "仅 replicate 0、仅 PASS；最终 3 次 repeat 不重复计入候选分布"]),
        "",
        "这两个 ResNet 层使用相同数据、映射、训练权重、COMO运行框架和主评分，故 Layer2/Conv1 的**同序列数值比较有效**。它不能单独分离层深、特征分辨率、通道数量与上采样方式的因果作用；通道 index 也不能跨 layer 解释为同一语义。",
        "",
        "# 2. 搜索完成情况与 ATE 分布",
        "",
        table(["搜索", "可用 channels", "unique replicate-0", "PASS", "FAIL", "PASS rate", "P5 / median / P95 (cm)", "best (cm)"]),
        table(["---", "---:", "---:", "---:", "---:", "---:", "---", "---:"]),
    ]
    for dataset in datasets:
        q = dataset["quantiles"]
        lines.append(table([dataset["short"], str(dataset["channels"]), str(dataset["total"]), str(dataset["pass"]), str(dataset["fail"] + dataset["error"]), percentage(dataset["pass_rate"]), f"{q['p05']:.2f} / {q['median']:.2f} / {q['p95']:.2f}", f"{q['min']:.4f}"]))
    lines.extend(
        [
            "",
            "![图1　归一化直方图：仅PASS、每个组合只保留 replicate 0。它比较分布形状，不受候选数量不同影响。](ate_histogram_normalized_layer2_vs_conv1.png){width=94%}",
            "",
            "![图2　分层计数直方图：黑虚线是中位数，红点线是最佳值；右上角标出超出80 cm显示范围的尾部数量。](ate_histogram_faceted_layer2_vs_conv1.png){width=97%}",
            "",
            "![图3　ATE ECDF：曲线越靠左，代表更多 PASS 候选达到较低ATE。](ate_ecdf_layer2_vs_conv1.png){width=94%}",
            "",
            "![图4　实际评估候选的完成状态。这里的 PASS rate 描述该搜索采样到的候选，不是对全部组合空间的无偏估计。](pass_failure_layer2_vs_conv1.png){width=88%}",
            "",
            f"**分布解读。** Layer2 的整体 PASS rate 高（{percentage(layer2['pass_rate'])} vs Conv1 的 {percentage(conv1['pass_rate'])}），但其 PASS-ATE 中位数更高（{layer2['quantiles']['median']:.2f} vs {conv1['quantiles']['median']:.2f} cm），且 P95 更高（{layer2['quantiles']['p95']:.2f} vs {conv1['quantiles']['p95']:.2f} cm）。相反，Conv1 的低误差端更强，最佳值为 {conv1_gstar['historical_ate_mean_cm']:.4f} cm。由此，Layer2 在本序列上提供较宽的可完成区域，Conv1 则提供更强但较稀疏的低误差 tail；这是对已运行候选的描述，不是对两个无限组合空间的统计显著性结论。",
            "",
            "# 3. Layer2 greedy 搜索路径与通道数行为",
            "",
            f"Layer2 all-128 anchor PASS（{layer2_all['historical_ate_mean_cm']:.4f} cm），gray control 失败。anchor t50=60.7 s，故 auto rule 选择 3 starts。128 个 singleton 中有 15 个 PASS，选取 `[d121]`、`[d67]`、`[d96]` 作为 seed；因此没有触发 C(128,2)=8,128 的 pair rescue。前两个 seed 在 K=3 收敛到相同 backbone `[d67,d108,d121]`，说明这三个 channel 的协同在多条搜索路径中重复出现。",
            "",
            table(["K", "Layer2 best direct endpoint", "ATE mean (cm)", "evaluated / PASS", "PASS ATE median (cm)"]),
            table(["---:", "---", "---:", "---", "---:"]),
        ]
    )
    for cardinality in range(1, 7):
        stats = layer2["by_k"][cardinality]
        endpoint = paths["ResNet Layer2 greedy"].get(cardinality)
        lines.append(table([str(cardinality), "—" if endpoint is None else display(",".join(str(value) for value in endpoint["channels"])), "—" if endpoint is None else f"{endpoint['ate_cm']:.4f}", f"{stats['evaluated']} / {stats['pass']}", cm(stats["median_cm"])]))
    lines.extend(
        [
            "",
            "![图5　Layer2 与 Conv1 在每个 K 的最佳 direct-greedy endpoint；并不要求通道数增加后单调变好。](greedy_path_layer2_vs_conv1.png){width=92%}",
            "",
            "![图6　按K统计的PASS率和 PASS-ATE median；候选数量来自实际 multi-start / random / swap 搜索，而非均匀穷举。](cardinality_profile_layer2_vs_conv1.png){width=97%}",
            "",
            "Layer2 最强路径从 `[d121]` 的 27.5292 cm 出发，+d108 至16.6202，+d67 至15.9524，+d60 至12.6714，+d41 至 **12.6100**；第六个 d95 使主ATE反升至12.7413。其主要收益发生在 K=1→4，K=4→5 的绝对改善仅0.0614 cm。因此不能以‘最多六通道’替代实际的 K 选择。",
            "",
            "# 4. Layer2 与 Conv1 的结果对比",
            "",
            table(["层 / 候选", "configuration", "K", "historical ATE mean/cm", "all-frame SE(3) RMSE/cm", "historical RPE RMSE/cm", "说明"]),
            table(["---", "---", "---:", "---:", "---:", "---:", "---"]),
            table(["Layer2 G5 / Lstar", "[d41,d60,d67,d108,d121]", "5", f"**{g5_candidate['historical_ate_mean_cm']:.4f}**", f"{g5_candidate['allframe_se3_ate_rmse_mean_cm']:.2f}", f"{g5_candidate['historical_rpe_rmse_mean_cm']:.4f}", "primary-metric best"]),
            table(["Layer2 G4", "[d60,d67,d108,d121]", "4", f"{g4_candidate['historical_ate_mean_cm']:.4f}", f"**{g4_candidate['allframe_se3_ate_rmse_mean_cm']:.2f}**", f"**{g4_candidate['historical_rpe_rmse_mean_cm']:.4f}**", "compact / secondary-balanced"]),
            table(["Layer2 G6", "[d41,d60,d67,d95,d108,d121]", "6", f"{layer2_g6['historical_ate_mean_cm']:.4f}", f"{layer2_g6['allframe_se3_ate_rmse_mean_cm']:.2f}", f"{layer2_g6['historical_rpe_rmse_mean_cm']:.4f}", "more channels did not improve primary ATE"]),
            table(["Conv1 G4 / Lstar", "[d15,d20,d26,d34]", "4", f"{conv1_gstar['historical_ate_mean_cm']:.4f}", "14.92", "1.1819", "current ResNet accuracy-first"]),
            "",
            f"Conv1 G4 is lower than Layer2 G5 by {layer2_vs_conv1 * 100:.1f}% on the primary ATE, and also lower on the two listed diagnostic metrics. On this sequence, therefore, **Conv1 remains the preferred ResNet candidate for accuracy-first evaluation**. Layer2 G4 should nevertheless be retained in a later multi-sequence pool because its compact four-channel backbone `[d60,d67,d108,d121]` is independently derived, has 3/3 deterministic PASS, and may respond differently to new appearance changes.",
            "",
            f"Both layers benefit greatly from selection: Layer2 G5 reduces ATE relative to all128 by {percentage(layer2_reduction)} ({layer2_all['historical_ate_mean_cm']:.4f}→{layer2_gstar['historical_ate_mean_cm']:.4f} cm); Conv1 G4 reduces its all64 control by {percentage(conv1_reduction)} (28.3611→{conv1_gstar['historical_ate_mean_cm']:.4f} cm). The numerical reduction should not be compared as a clean architectural effect because the all-channel controls have different channel count/resolution.",
            "",
            "# 5. 可审计日志与搜索日记",
            "",
            table(["实验", "权威 SQLite", "逐步路径", "完整 console 日记", "运行窗口"]),
            table(["---", "---", "---", "---", "---"]),
            table(["Layer2 greedy", "`channel_selection_results/step_o_resnet_layer2_direct_fullseq_greedy/evaluations.sqlite3`（3,632 saved rows）", "`.../direct_greedy_path.csv`", "`.../console.log`", "2026-08-19 14:17 至 2026-08-21 10:26"]),
            table(["Conv1 greedy", "`channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/evaluations.sqlite3`（4,218 saved rows）", "`.../direct_greedy_path.csv`", "`.../console.log`", "2026-08-17 17:52 至 2026-08-19 02:01"]),
            "",
            "Layer2 console 中保存了自动启动数的依据、15个成功 singleton 的排序、每次 forward choice、缓存复用、random control、615个 K=5 one-swap neighbours，以及最终 repeat。本文所有分布图直接由 SQLite 的 replicate-0 原始记录生成，避免从文本日志重新解析数值；SQLite 是权威结果，console 是人类可读的搜索日记。",
            "",
            "# 6. 结论与建议",
            "",
            "1. **Layer2 greedy 成功完成。** 它不需要 pair rescue，发现以 d67/d108/d121 为核心的稳定 backbone，并给出 G5 12.6100 cm / G4 12.6714 cm 的紧凑候选。",
            "2. **当前 ResNet 层级排序：Conv1 优先。** Conv1 G4 = 10.3758 cm 优于 Layer2 G5 = 12.6100 cm，且 secondary diagnostics 同样较低；应作为下一步 accuracy-first 的主要 ResNet 配置。",
            "3. **Layer2 G4 值得保留而非仅保存 G5。** 它只损失0.0614 cm primary ATE，却在全帧SE(3) ATE和历史RPE上更好，且少一个 channel。后续多序列可以同时测 G4 与 G5，而不是假设单一 global optimum 可泛化。",
            "4. **机制上：Layer2 的可行区域更宽，但低误差尾部不如 Conv1。** 这支持把 failure-avoidance、ATE accuracy 和跨序列鲁棒性分开报告。",
            "5. **局限性：**所有结论来自同一条 lightswitch trajectory；3次重复均为确定性复现，不能替代跨数据集与真实外观变化下的外部验证。",
            "",
            "# 附录：本报告图表的数据来源",
            "",
            "- Layer2：`channel_selection_results/step_o_resnet_layer2_direct_fullseq_greedy/`",
            "- Conv1：`channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/`",
            "- 派生的图表摘要：`layer2_conv1_distribution_summary.json`。",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] Markdown: {OUTPUT_MD}")
    for figure in sorted(REPORT_DIR.glob("*.png")):
        print(f"[DONE] Figure: {figure}")


if __name__ == "__main__":
    main()
