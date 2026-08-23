#!/usr/bin/env python3
"""Create an auditable Chinese comparison report for the completed greedy runs.

Inputs are read-only SQLite/CSV results from ResNet Conv1 greedy, UNet Enc0/1
greedy and the earlier ResNet correlation-clustering/MVS-filtered full-sequence
evaluation.  The script never opens a COMO configuration or changes evaluator
databases.
"""

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

_CACHE = Path(tempfile.gettempdir()) / "resnet_conv1_greedy_report_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_DIR.parents[2]
OUTPUT_MD = REPORT_DIR / "ResNet_Conv1_Greedy与UNet及Correlation_BruteForce对比_中文.md"
OUTPUT_JSON = REPORT_DIR / "search_distribution_summary.json"

GREEDY_SOURCES = {
    "ResNet Conv1 greedy": {
        "path": PROJECT_ROOT
        / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/evaluations.sqlite3",
        "paths": PROJECT_ROOT
        / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/direct_greedy_path.csv",
        "recommendation": PROJECT_ROOT
        / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/recommendation.json",
        "console": PROJECT_ROOT
        / "channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/console.log",
        "channel_count": 64,
        "short": "ResNet Conv1",
    },
    "UNet Enc1 greedy": {
        "path": PROJECT_ROOT
        / "channel_selection_results/step_j_unet_direct_fullseq_greedy/evaluations.sqlite3",
        "paths": PROJECT_ROOT
        / "channel_selection_results/step_j_unet_direct_fullseq_greedy/direct_greedy_path.csv",
        "recommendation": PROJECT_ROOT
        / "channel_selection_results/step_j_unet_direct_fullseq_greedy/recommendation.json",
        "console": PROJECT_ROOT
        / "channel_selection_results/step_j_unet_direct_fullseq_greedy/console.log",
        "channel_count": 32,
        "short": "UNet Enc1",
    },
    "UNet Enc0 greedy": {
        "path": PROJECT_ROOT
        / "channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy/evaluations.sqlite3",
        "paths": PROJECT_ROOT
        / "channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy/direct_greedy_path.csv",
        "recommendation": PROJECT_ROOT
        / "channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy/recommendation.json",
        "console": PROJECT_ROOT
        / "channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy/console.log",
        "channel_count": 16,
        "short": "UNet Enc0",
    },
}
CORRELATION_CSV = (
    PROJECT_ROOT
    / "channel_selection_results/step_e_full_sequence_evaluation/"
    "second_round_baseline_plus2_rpe_safe/all_evaluations.csv"
)
CORRELATION_CONSOLE = (
    PROJECT_ROOT
    / "channel_selection_results/step_e_full_sequence_evaluation/"
    "second_round_baseline_plus2_rpe_safe/console.log"
)

COLORS = {
    "ResNet Conv1 greedy": "#3b6fb6",
    "UNet Enc1 greedy": "#dc8f31",
    "UNet Enc0 greedy": "#4f9b72",
    "ResNet correlation brute force": "#8952a1",
}


def require(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def cm(value: float | None) -> str:
    return "—" if value is None or not math.isfinite(value) else f"{value:.4f}"


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def channels_display(key: str) -> str:
    if key == "gray":
        return "gray"
    if key.startswith("all"):
        return key
    return "[" + ",".join(f"d{item}" for item in key.split(",")) + "]"


def read_greedy(name: str, info: dict[str, Any]) -> dict[str, Any]:
    require(info["path"])
    with sqlite3.connect(info["path"]) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT * FROM evaluations WHERE replicate=0 ORDER BY id").fetchall()
        stage_rows = db.execute(
            "SELECT stage, COUNT(*) AS rows, COUNT(DISTINCT candidate_key) AS candidates "
            "FROM stage_candidates GROUP BY stage ORDER BY stage"
        ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        raw = record["channels_json"]
        record["cardinality"] = len(json.loads(raw)) if raw else None
        record["ate_cm"] = (
            float(record["historical_evo_ape_mean_m"]) * 100
            if record["status"] == "PASS"
            and record["historical_evo_ape_mean_m"] is not None
            else None
        )
        records.append(record)
    values = [record["ate_cm"] for record in records if record["ate_cm"] is not None]
    values_array = np.asarray(values, dtype=float)
    status = Counter(record["status"] for record in records)
    counts_by_k: dict[int, dict[str, Any]] = {}
    for cardinality in range(1, 7):
        subset = [record for record in records if record["cardinality"] == cardinality]
        passed = [record["ate_cm"] for record in subset if record["ate_cm"] is not None]
        counts_by_k[cardinality] = {
            "evaluated": len(subset),
            "pass": len(passed),
            "pass_rate": len(passed) / len(subset) if subset else None,
            "best_cm": min(passed) if passed else None,
            "median_cm": float(np.median(passed)) if passed else None,
        }
    return {
        "name": name,
        "short": info["short"],
        "available_channels": info["channel_count"],
        "records": records,
        "values": values,
        "total": len(records),
        "pass": len(values),
        "fail": sum(value for key, value in status.items() if key.startswith("FAIL")),
        "error": sum(value for key, value in status.items() if key.startswith("ERROR")),
        "pass_rate": len(values) / len(records),
        "quantiles": {
            label: float(value)
            for label, value in zip(
                ("min", "p05", "p25", "median", "p75", "p95", "max"),
                np.percentile(values_array, (0, 5, 25, 50, 75, 95, 100)),
            )
        },
        "counts_by_k": counts_by_k,
        "stage_counts": [dict(row) for row in stage_rows],
    }


def read_correlation() -> dict[str, Any]:
    require(CORRELATION_CSV)
    with CORRELATION_CSV.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    records: list[dict[str, Any]] = []
    for row in raw_rows:
        record: dict[str, Any] = dict(row)
        record["cardinality"] = len(json.loads(row["channels_json"])) if row["channels_json"] else None
        record["ate_cm"] = (
            float(row["historical_evo_ape_mean_m"]) * 100 if row["status"] == "PASS" else None
        )
        records.append(record)
    values = [record["ate_cm"] for record in records if record["ate_cm"] is not None]
    values_array = np.asarray(values, dtype=float)
    status = Counter(record["status"] for record in records)
    return {
        "name": "ResNet correlation brute force",
        "short": "Correlation brute force",
        "available_channels": 36,
        "records": records,
        "values": values,
        "total": len(records),
        "pass": len(values),
        "fail": sum(value for key, value in status.items() if key.startswith("FAIL")),
        "error": sum(value for key, value in status.items() if key.startswith("ERROR")),
        "pass_rate": len(values) / len(records),
        "quantiles": {
            label: float(value)
            for label, value in zip(
                ("min", "p05", "p25", "median", "p75", "p95", "max"),
                np.percentile(values_array, (0, 5, 25, 50, 75, 95, 100)),
            )
        },
        "counts_by_k": {
            4: {
                "evaluated": len(records),
                "pass": len(values),
                "pass_rate": len(values) / len(records),
            }
        },
        "stage_counts": [],
    }


def read_paths(path: Path) -> dict[int, dict[str, Any]]:
    require(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        cardinality = int(row["cardinality"])
        value = row["single_run_primary_ate_mean_cm"]
        if not value:
            continue
        candidate = {
            "channels": row["channels"],
            "ate_cm": float(value),
            "seed_index": int(row["seed_index"]),
        }
        if cardinality not in result or candidate["ate_cm"] < result[cardinality]["ate_cm"]:
            result[cardinality] = candidate
    return result


def load_recommendation(path: Path) -> dict[str, Any]:
    require(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_ecdf(datasets: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(9.4, 5.5))
    for dataset in datasets:
        values = np.sort(np.asarray(dataset["values"], dtype=float))
        y = np.arange(1, len(values) + 1) / len(values)
        axis.step(values, y, where="post", linewidth=2.2, label=dataset["short"], color=COLORS[dataset["name"]])
    axis.set_xlim(0, 55)
    axis.set_ylim(0, 1.01)
    axis.set_xlabel("Historical keyframe ATE mean (cm; lower is better)")
    axis.set_ylabel("Empirical CDF of PASS candidates")
    axis.set_title("Full-sequence ATE distribution: evaluated PASS candidates")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=True)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "ate_ecdf_comparison.png", dpi=210, bbox_inches="tight")
    plt.close(figure)


def save_boxplot(datasets: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(9.4, 5.4))
    labels = [dataset["short"] for dataset in datasets]
    boxes = axis.boxplot(
        [dataset["values"] for dataset in datasets],
        tick_labels=labels,
        showfliers=True,
        whis=(5, 95),
        patch_artist=True,
    )
    for patch, dataset in zip(boxes["boxes"], datasets):
        patch.set_facecolor(COLORS[dataset["name"]])
        patch.set_alpha(0.72)
    axis.set_ylabel("Historical keyframe ATE mean (cm; lower is better)")
    axis.set_title("ATE distribution summary (box = 25th–75th percentile; whiskers = 5th–95th)")
    axis.set_ylim(0, 60)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "ate_boxplot_comparison.png", dpi=210, bbox_inches="tight")
    plt.close(figure)


def save_outcome_bars(datasets: list[dict[str, Any]]) -> None:
    names = [dataset["short"] for dataset in datasets]
    passed = np.asarray([dataset["pass"] for dataset in datasets])
    failed = np.asarray([dataset["fail"] + dataset["error"] for dataset in datasets])
    positions = np.arange(len(datasets))
    figure, axis = plt.subplots(figsize=(9.4, 5.2))
    axis.bar(positions, passed, label="PASS", color="#4f9b72")
    axis.bar(positions, failed, bottom=passed, label="FAIL / ERROR", color="#c65d5d")
    for index, dataset in enumerate(datasets):
        axis.text(index, passed[index] + failed[index] + max(35, dataset["total"] * 0.012), percent(dataset["pass_rate"]), ha="center", va="bottom", fontsize=10)
    axis.set_xticks(positions, names)
    axis.set_ylabel("Unique replicate-0 configurations")
    axis.set_title("Completion outcome for the evaluated candidate sets")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "pass_failure_comparison.png", dpi=210, bbox_inches="tight")
    plt.close(figure)


def save_cardinality_profile(greedy: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for dataset in greedy:
        ks = [key for key in range(1, 7) if dataset["counts_by_k"][key]["evaluated"]]
        rates = [dataset["counts_by_k"][key]["pass_rate"] * 100 for key in ks]
        medians = [dataset["counts_by_k"][key]["median_cm"] for key in ks]
        axes[0].plot(ks, rates, marker="o", linewidth=2, label=dataset["short"], color=COLORS[dataset["name"]])
        axes[1].plot(ks, medians, marker="o", linewidth=2, label=dataset["short"], color=COLORS[dataset["name"]])
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
    figure.savefig(REPORT_DIR / "greedy_cardinality_profile.png", dpi=210, bbox_inches="tight")
    plt.close(figure)


def save_best_paths(paths: dict[str, dict[int, dict[str, Any]]]) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    for name, values in paths.items():
        ks = sorted(values)
        axis.plot(
            ks,
            [values[key]["ate_cm"] for key in ks],
            marker="o",
            linewidth=2.4,
            label=GREEDY_SOURCES[name]["short"],
            color=COLORS[name],
        )
    axis.set_xticks(range(1, 7))
    axis.set_xlabel("K selected channels")
    axis.set_ylabel("Best direct-greedy endpoint ATE mean (cm)")
    axis.set_title("Best direct-greedy endpoint by K (not required to be monotonic)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(REPORT_DIR / "greedy_path_by_cardinality.png", dpi=210, bbox_inches="tight")
    plt.close(figure)


def table(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for item in GREEDY_SOURCES.values():
        for key in ("path", "paths", "recommendation", "console"):
            require(item[key])
    require(CORRELATION_CONSOLE)

    greedy = [read_greedy(name, info) for name, info in GREEDY_SOURCES.items()]
    correlation = read_correlation()
    datasets = [greedy[0], greedy[1], greedy[2], correlation]
    paths = {name: read_paths(info["paths"]) for name, info in GREEDY_SOURCES.items()}
    recommendations = {name: load_recommendation(info["recommendation"]) for name, info in GREEDY_SOURCES.items()}
    source_data = {dataset["name"]: {key: value for key, value in dataset.items() if key != "records"} for dataset in datasets}
    source_data["paths"] = paths
    OUTPUT_JSON.write_text(json.dumps(source_data, indent=2), encoding="utf-8")

    save_ecdf(datasets)
    save_boxplot(datasets)
    save_outcome_bars(datasets)
    save_cardinality_profile(greedy)
    save_best_paths(paths)

    resnet, enc1, enc0 = greedy
    corr_best = min(correlation["records"], key=lambda item: item["ate_cm"] if item["ate_cm"] is not None else math.inf)
    resnet_gstar = recommendations["ResNet Conv1 greedy"]["adaptive_gstar_after_three_repeats"]
    enc1_gstar = recommendations["UNet Enc1 greedy"]["adaptive_gstar_after_three_repeats"]
    enc1_lstar = min(
        (item for item in recommendations["UNet Enc1 greedy"]["candidates"] if item["pass_count"] == 3),
        key=lambda item: item["historical_ate_mean_cm"],
    )
    enc0_gstar = recommendations["UNet Enc0 greedy"]["adaptive_gstar_after_three_repeats"]
    enc0_g4 = recommendations["UNet Enc0 greedy"]["fixed_g4_after_three_repeats"]
    enc1_g4 = recommendations["UNet Enc1 greedy"]["fixed_g4_after_three_repeats"]
    resnet_candidates = recommendations["ResNet Conv1 greedy"]["candidates"]
    historical_corr = next(item for item in resnet_candidates if "correlation_search" in item["tags"])
    historical_baseline = next(item for item in resnet_candidates if "cnn_baseline" in item["tags"])
    all64 = next(item for item in resnet_candidates if item["channels"] == "all64")

    improvement_over_corr = 1 - resnet_gstar["historical_ate_mean_cm"] / corr_best["ate_cm"]
    improvement_over_baseline = 1 - resnet_gstar["historical_ate_mean_cm"] / historical_baseline["historical_ate_mean_cm"]
    improvement_over_all64 = 1 - resnet_gstar["historical_ate_mean_cm"] / all64["historical_ate_mean_cm"]
    enc0_over_resnet = 1 - enc0_gstar["historical_ate_mean_cm"] / resnet_gstar["historical_ate_mean_cm"]
    enc1_over_resnet = 1 - enc1_lstar["historical_ate_mean_cm"] / resnet_gstar["historical_ate_mean_cm"]

    lines = [
        "---",
        'title: "ResNet Conv1 直接全序列 Greedy 搜索：与 U-Net Greedy 及 Correlation-Brute-Force 的比较"',
        'subtitle: "fr1/desk_lightswitch，573 frames；报告日期：2026-08-19"',
        "lang: zh-CN",
        "---",
        "",
        "# ResNet Conv1 直接全序列 Greedy 搜索：与 U-Net Greedy 及 Correlation-Brute-Force 的比较",
        "",
        "*fr1/desk_lightswitch，573 个 matched timestamps；报告日期：2026-08-19*",
        "",
        "# 执行摘要",
        "",
        f"本次将 ResNet 最浅层 Conv1（项目内早期常称 conv0）的 64 个 **post-ReLU** channels，按与 U-Net Enc1 相同的 direct full-sequence multi-start greedy 协议重新搜索。最终 **ResNet Conv1 G4 = [d15,d20,d26,d34]，ATE mean = {resnet_gstar['historical_ate_mean_cm']:.4f} cm，3/3 PASS**。它比此前 correlation-clustering based four-channel brute-force 的最佳 `[d5,d6,d24,d29]`（{corr_best['ate_cm']:.4f} cm）低 **{percent(improvement_over_corr)}**，也比历史 CNN baseline `[d5,d29,d40,d52]`（{historical_baseline['historical_ate_mean_cm']:.4f} cm）低 **{percent(improvement_over_baseline)}**。",
        "",
        f"但同一序列上，U-Net 直接 greedy 的最佳尾部仍更低：Enc0 `[d2,d3,d7,d12,d13,d14]` 为 {enc0_gstar['historical_ate_mean_cm']:.4f} cm，Enc1 one-swap 最优 `[d5,d6,d17,d18,d28,d30]` 为 {enc1_lstar['historical_ate_mean_cm']:.4f} cm，分别比 ResNet greedy 最优低 **{percent(enc0_over_resnet)}** 与 **{percent(enc1_over_resnet)}**。这是单一 lightswitch 序列上的条件性排序；它支持后续把这些候选并列带入多序列验证，而不是宣称某一架构已经全局胜出。",
        "",
        "导师所需的分布证据也已保留：本报告直接从三套 greedy SQLite 的 replicate-0 唯一候选记录生成 ATE ECDF、箱线图、PASS/FAIL 统计、按 K 的成功率/ATE曲线和 greedy path 图。完整 console logs 与逐步路径 CSV 仍在原结果目录，可审计每一次 seed、扩展、swap 和最终重复。",
        "",
        "# 1. 实验设置与可比性边界",
        "",
        table(["项目", "共同设置 / 说明"]),
        table(["---", "---"]),
        table(["序列", "完整 `rgbd_dataset_freiburg1_desk_lightswitch`；573 个 matched RGB-D timestamps"]),
        table(["Mapping", "固定 gray，使用 matched sensor depth；ground truth 仅在运行后计算轨迹指标"]),
        table(["主排名指标", "keyframe `evo_ape tum --align --correct_scale` translation ATE mean（cm，越低越好）"]),
        table(["Failure / timeout", "NaN/Inf tracking diagnostics 立即失败；单次上限 300 s；coverage 门槛 90%"]),
        table(["ATE 分布取样", "每个 candidate 只取 replicate 0；仅 PASS 的 ATE 进入分布。最终 repeat 不重复计入分布"]),
        table(["重复", "最终保留候选均补足 3 次；数值标准差为 0，说明固定输入/软件路径下可复现，不等价于独立随机样本的置信区间"]),
        "",
        "三个 direct greedy 搜索在**主数据、映射、主 ATE 口径和 failure rules**上可直接比较。不同架构的 channel index 不具有语义对应关系，`d15` 等只在其网络层内部有意义。",
        "",
        "Correlation-brute-force 是重要的历史对照，但其 full-sequence 表不是无筛选的候选分布：它先从 r=0.70 correlation clustering 的 36 个 representative channels 组成四通道候选，经 MVS PASS、baseline+2% MVS ATE 与 RPE safety gate 后，才将 3,713 个配置送入完整序列。因此其 PASS rate 不能被解释为该方法天生更鲁棒；该分布只适合与新 ResNet greedy 比较**已评估的 full-sequence ATE尾部与最佳值**。",
        "",
        "# 2. 搜索规模、失败结构与ATE分布",
        "",
        table(["搜索", "feature space", "unique replicate-0 configs", "PASS", "FAIL/ERROR", "PASS rate", "ATE P5 / median / P95 (cm)", "best (cm)"]),
        table(["---", "---", "---:", "---:", "---:", "---:", "---", "---:"]),
    ]
    for dataset in datasets:
        q = dataset["quantiles"]
        lines.append(
            table(
                [
                    dataset["short"],
                    (f"{dataset['available_channels']} channels, K=1–6" if "greedy" in dataset["name"] else "36 r=0.70 representatives, K=4 only"),
                    str(dataset["total"]),
                    str(dataset["pass"]),
                    str(dataset["fail"] + dataset["error"]),
                    percent(dataset["pass_rate"]),
                    f"{q['p05']:.2f} / {q['median']:.2f} / {q['p95']:.2f}",
                    f"{q['min']:.4f}",
                ]
            )
        )
    lines.extend(
        [
            "",
            "![图1　各搜索中所有PASS的 replicate-0 候选 ATE ECDF。曲线越靠左，说明低ATE候选比例越高；横轴截在55 cm以便看清主要区域。](ate_ecdf_comparison.png){width=94%}",
            "",
            "![图2　ATE分布的稳健摘要。箱体为25–75百分位，须为5–95百分位；只统计PASS且每个候选仅一次。](ate_boxplot_comparison.png){width=94%}",
            "",
            "![图3　每套实验实际评估的唯一候选完成情况。注意 correlation-brute-force 已经过MVS/RPE预筛，故成功率不是与直接greedy的公平性能比较。](pass_failure_comparison.png){width=94%}",
            "",
            "**分布解读。** ResNet greedy 的 PASS 分布中位数为 24.74 cm，与 correlation-brute-force 的 24.95 cm 接近；但其低误差尾部更强：21 个 PASS 配置低于 15 cm，而 correlation-brute-force 只有 5 个。它说明直接搜索并非简单把整体分布全部左移，而是发现了一个此前 clustering-representative 限制下没有出现的低误差 basin。U-Net Enc0/Enc1 的整体 PASS 分布均明显左移（中位数分别为 12.34/16.24 cm），但候选空间大小、网络架构和采样路径不同，因此这里是表现描述，不是显著性检验。",
            "",
            "# 3. ResNet Conv1 direct greedy 的结果",
            "",
            "## 3.1 关键搜索路径",
            "",
            f"64/64 个 ResNet Conv1 singleton 全部失败，因此算法自动触发全部 C(64,2)=2,016 个 pair rescue sweep；其中 {resnet['counts_by_k'][2]['pass']}/2,016 个 pair PASS。四个 pair seed 为 `[d33,d52]`、`[d26,d51]`、`[d23,d59]`、`[d20,d26]`。这不是 algorithm failure，而是直接证据：在该层与该 illumination challenge 下，可靠 tracking 信号依赖 channel pair 的互补，而非单个通道。",
            "",
            table(["K", "best direct-greedy endpoint", "ATE mean (cm)", "该K已评估 / PASS", "PASS ATE median (cm)"]),
            table(["---:", "---", "---:", "---", "---:"]),
        ]
    )
    for cardinality in range(1, 7):
        stats = resnet["counts_by_k"][cardinality]
        path_info = paths["ResNet Conv1 greedy"].get(cardinality)
        endpoint = "—" if path_info is None else channels_display(",".join(str(item) for item in json.loads(path_info["channels"])))
        endpoint_ate = "—" if path_info is None else f"{path_info['ate_cm']:.4f}"
        lines.append(
            table(
                [
                    str(cardinality),
                    endpoint,
                    endpoint_ate,
                    f"{stats['evaluated']} / {stats['pass']}",
                    cm(stats["median_cm"]),
                ]
            )
        )
    lines.extend(
        [
            "",
            "最强路径由 pair `[d20,d26]` 起步：K=2 为 15.5393 cm，K=3 加 d34 后暂时到 15.7185 cm，K=4 加 d15 后骤降至 **10.3758 cm**；继续加 d45 / d53 并没有维持改善（K=5 11.9254 cm，K=6 11.6218 cm）。因此 K=4 不是任意固定预算，而是由完整 path 中的非单调响应所支持的 optimum；one-channel swap audit 的 240 个邻居也未超过它。",
            "",
            table(["ResNet Conv1 参考 / 结果", "configuration", "K", "ATE mean (cm)", "相对 ResNet greedy G4"]),
            table(["---", "---", "---:", "---:", "---"]),
            table(["Direct greedy G4 / Lstar", "[d15,d20,d26,d34]", "4", f"**{resnet_gstar['historical_ate_mean_cm']:.4f}**", "reference"]),
            table(["Direct greedy G5", "[d23,d24,d26,d51,d63]", "5", "10.4535", "+0.7%"]),
            table(["Direct greedy G6", "[d15,d20,d26,d34,d45,d53]", "6", "11.6218", "+12.0%"]),
            table(["Historical correlation search", "[d5,d6,d24,d29]", "4", f"{historical_corr['historical_ate_mean_cm']:.4f}", f"higher {percent(improvement_over_corr)}"]),
            table(["Historical CNN baseline", "[d5,d29,d40,d52]", "4", f"{historical_baseline['historical_ate_mean_cm']:.4f}", f"higher {percent(improvement_over_baseline)}"]),
            table(["Unselected Conv1", "all64", "64", f"{all64['historical_ate_mean_cm']:.4f}", f"higher {percent(improvement_over_all64)}"]),
            "",
            "## 3.2 与 correlation-based brute force 的含义",
            "",
            f"此前 r=0.70 correlation pipeline 的完整序列最优 `[d5,d6,d24,d29]` 在本次也被作为 historical anchor 重新运行，并得到同一 {historical_corr['historical_ate_mean_cm']:.4f} cm。因而比较不依赖于不同脚本的度量漂移：新的 direct greedy G4 的 {resnet_gstar['historical_ate_mean_cm']:.4f} cm 是相同数据、相同 ATE 口径下 **{percent(improvement_over_corr)} 的降低**。这表明 correlation clustering 对大规模四通道空间的去冗余/压缩有用，但它不应被视为全局最优组合的充分搜索方法；它可能排除了只有在跨cluster/非代表成员组合中才显现的互补。",
            "",
            "与此同时，不能把这项结果误读为 correlation clustering ‘无效’：它在第一阶段将原始四通道组合空间压缩为可计算的候选，并通过MVS/全序列流程给出一个稳定、可重复的 14.0623 cm reference。新 direct greedy 的作用是以更少的结构先验、允许 K=1–6 和 pair rescue 的方式补充该搜索，而不是取代先前的 failure filtering evidence。",
            "",
            "# 4. 与 U-Net greedy 的路径与分布对比",
            "",
            "![图4　每个K的最佳 direct-greedy endpoint；曲线并非要求单调下降。Enc1 的 one-swap 最优 6.7335 cm 不在此图的纯direct G path 中。](greedy_path_by_cardinality.png){width=94%}",
            "",
            "![图5　不同K下实际评估候选的PASS率与PASS-ATE中位数。由于前向路径、pair rescue和random control的候选数不同，这不是均匀枚举的K曲线，而是对本次已评估数据的诊断。](greedy_cardinality_profile.png){width=97%}",
            "",
            table(["Layer / search", "singleton outcome", "pair rescue PASS", "best fixed K=4", "best final configuration", "final ATE (cm)"]),
            table(["---", "---", "---", "---", "---", "---:"]),
            table(["ResNet Conv1", "0/64 PASS", "421/2,016", "[d15,d20,d26,d34] = 10.3758", "[d15,d20,d26,d34]", f"{resnet_gstar['historical_ate_mean_cm']:.4f}"]),
            table(["UNet Enc1", "0/32 PASS", "281/496", f"[d0,d5,d18,d30] = {enc1_g4['historical_ate_mean_cm']:.4f}", "[d5,d6,d17,d18,d28,d30] (swap)", f"{enc1_lstar['historical_ate_mean_cm']:.4f}"]),
            table(["UNet Enc0", "1/16 PASS: d3", "92/120", f"[d2,d3,d12,d14] = {enc0_g4['historical_ate_mean_cm']:.4f}", "[d2,d3,d7,d12,d13,d14]", f"{enc0_gstar['historical_ate_mean_cm']:.4f}"]),
            "",
            "**共同机制证据。** ResNet Conv1 与 UNet Enc1 的所有 singleton 均失败，但很多 pair 成功；这跨越两种架构支持‘有效 tracking cue 是互补集合’而非‘单一万能 channel’的解释。UNet Enc0 有唯一可行 singleton d3，但它的 24.5318 cm 仍很弱，加入 d7/d12 后才达到 6.0004 cm。故 Enc0 的例子同样不支持把单通道可行性当作最优性的证据。",
            "",
            "**架构差异的条件性描述。** 在本序列的已评估 PASS 集合中，Enc0/Enc1 的分布更靠低ATE端，且最佳值分别较 ResNet G4 低 42.9% 与 35.1%。这可能来自特征层级、预训练表示、COMO输入接口或当前任务的共同作用；目前实验并未控制这些因素，不能归因到某一个网络设计细节。后续多序列验证才是检验这一排序能否保持的必要步骤。",
            "",
            "# 5. U-Net 的可审计日志（导师查阅入口）",
            "",
            table(["实验", "SQLite / distribution source", "逐步 greedy path", "完整 console 日记"]),
            table(["---", "---", "---", "---"]),
            table(["UNet Enc1", "`channel_selection_results/step_j_unet_direct_fullseq_greedy/evaluations.sqlite3`（1,596 saved rows）", "`.../direct_greedy_path.csv`", "`.../console.log`；2026-08-14 14:55 至 2026-08-15 13:38"]),
            table(["UNet Enc0", "`channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy/evaluations.sqlite3`（601 saved rows）", "`.../direct_greedy_path.csv`", "`.../console.log`；2026-08-15 15:31 至 23:00"]),
            table(["ResNet Conv1", "`channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/evaluations.sqlite3`（4,218 saved rows）", "`.../direct_greedy_path.csv`", "`.../console.log`；2026-08-17 17:52 至 2026-08-19 02:01"]),
            "",
            "每份 console 逐行记录了 config hash、每次运行的PASS/FAIL、ATE、pair fallback、seed、每一步 greedy choice、swap plan 和最终导出。报告图中的分布数据来自上述 SQLite，而不是从 console 文本二次推断；因此即使日志被截断，数据库仍是权威记录。",
            "",
            "# 6. 结论与后续建议",
            "",
            "1. **ResNet Conv1 direct greedy 已完成且有效。** 它在完整 lightswitch 序列上给出 `[d15,d20,d26,d34]` = 10.3758 cm，并严格优于 correlation-brute-force 历史最优 14.0623 cm。因而 ResNet 与 U-Net 现在都拥有同类 full-sequence greedy 证据。",
            "2. **ResNet 最优仍是四通道。** 本次的 K=5/K=6 并未超过G4；因此后续跨序列池应至少保留 ResNet G4，同时可带 G5 作为接近的不同组成对照，而非默认使用最多六通道。",
            "3. **U-Net 目前有更低的单序列结果，但结论尚未跨序列。** 推荐保留 Enc0 K=3/K=4/K=6，Enc1 K=4 与 swap-K=6，和 ResNet G4 / correlation best / historical baseline 组成后续多序列验证池。",
            "4. **方法学含义。** correlation clustering 适合做去冗余与大空间的计算压缩；direct greedy（含 pair fallback、multi-start 和 swap）更适合在不预先限定代表通道时寻找低误差互补组合。二者可视为互补的筛选层，而非只能二选一。",
            "5. **局限性。** 所有本报告的精度比较均来自同一条 573-frame lightswitch trajectory，且最终重复为确定性复现。它们不能代替不同场景、不同照明叙事和真实退化下的外部验证。",
            "",
            "# 附录：生成本报告的原始数据",
            "",
            "- ResNet greedy：`channel_selection_results/step_n_resnet_conv1_direct_fullseq_greedy/`",
            "- UNet Enc1 greedy：`channel_selection_results/step_j_unet_direct_fullseq_greedy/`",
            "- UNet Enc0 greedy：`channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy/`",
            "- Correlation-brute-force full-sequence round：`channel_selection_results/step_e_full_sequence_evaluation/second_round_baseline_plus2_rpe_safe/all_evaluations.csv`",
            "- 本报告的图表输入摘要：`search_distribution_summary.json`（仅从上述权威记录派生）。",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[DONE] Markdown: {OUTPUT_MD}")
    for figure in sorted(REPORT_DIR.glob("*.png")):
        print(f"[DONE] Figure: {figure}")


if __name__ == "__main__":
    main()
