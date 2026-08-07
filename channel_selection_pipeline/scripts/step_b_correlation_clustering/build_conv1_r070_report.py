#!/usr/bin/env python3
"""Build the Markdown source for the Conv1 r=0.70 clustering report."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_R070 = (
    PROJECT_ROOT
    / "channel_selection_results/step_b_correlation_clustering/threshold_r070"
)
DEFAULT_R080 = (
    PROJECT_ROOT
    / "channel_selection_results/step_b_correlation_clustering/threshold_r080"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r070-dir", type=Path, default=DEFAULT_R070)
    parser.add_argument("--r080-dir", type=Path, default=DEFAULT_R080)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    result = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    result.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(result)


def load_matrix(path: Path) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            left = int(row["channel"])
            for key, raw in row.items():
                if key == "channel" or not raw:
                    continue
                right = int(key.removeprefix("ch_"))
                values[(left, right)] = float(raw)
    return values


def valid_combination_count(clusters: list[dict]) -> int:
    representative_groups = []
    for cluster in clusters:
        group = [cluster["medoid"]]
        if cluster["second_representative"] is not None:
            group.append(cluster["second_representative"])
        representative_groups.append(group)
    return sum(
        math.prod(len(group) for group in selected)
        for selected in itertools.combinations(representative_groups, 4)
    )


def trigger_text(cluster: dict) -> str:
    triggers = cluster["second_representative_trigger"]
    ncc = cluster["cross_light_ncc_spread"]
    ratio = cluster["robust_gradient_ratio"]
    if triggers == ["robust_gradient_ratio"]:
        return (
            f"robust_gradient_ratio = **{ratio:.2f}**（超过 2.0；簇内空间梯度强度差异明显）"
        )
    if set(triggers) == {"cross_light_ncc_spread", "robust_gradient_ratio"}:
        return (
            f"NCC spread = **{ncc:.3f}**（超过 0.15）且 "
            f"gradient ratio = **{ratio:.2f}**（超过 2.0）"
        )
    if triggers == ["cross_light_ncc_spread"]:
        return f"NCC spread = **{ncc:.3f}**（超过 0.15）"
    return (
        f"未触发：NCC spread = {ncc:.3f}（< 0.15），"
        f"gradient ratio = {ratio:.2f}（< 2.0）"
    )


def interpretation(cluster: dict) -> str:
    size = cluster["size"]
    metrics = list(cluster["member_metrics"].values())
    nccs = [item["cross_light_ncc"] for item in metrics]
    ratio = cluster["robust_gradient_ratio"]
    second = cluster["second_representative"]
    if second is None:
        return (
            f"{size} 个成员的 illumination response 与梯度强度差异未达到 second-rep 阈值，"
            f"因此只保留 medoid ch{cluster['medoid']}。"
        )
    if max(nccs) - min(nccs) >= 0.15:
        return (
            f"簇内 NCC 跨度较大（{min(nccs):.3f}–{max(nccs):.3f}），"
            f"同时 gradient energy 相差 {ratio:.1f} 倍。保留 medoid ch{cluster['medoid']} "
            f"与 second representative ch{second}，降低较低阈值错误压缩不同 illumination behavior 的风险。"
        )
    return (
        f"成员的 NCC 较接近，但 gradient energy 相差 {ratio:.1f} 倍；"
        f"保留 ch{cluster['medoid']} 和 ch{second} 以覆盖不同响应强度。"
    )


def member_rows(cluster: dict) -> list[list[str]]:
    rows = []
    for channel in cluster["members"]:
        if channel == cluster["medoid"]:
            role = "**Medoid**"
        elif channel == cluster["second_representative"]:
            role = "Second rep"
        else:
            role = "—"
        metric = cluster["member_metrics"][str(channel)]
        rows.append(
            [
                role,
                f"ch {channel}",
                f"{metric['cross_light_ncc']:.3f}",
                f"{metric['robust_gradient_energy']:.5f}",
                f"{metric['quality_percentile_score']:.3f}",
            ]
        )
    return rows


def pairwise_section(cluster: dict, matrix: dict[tuple[int, int], float]) -> str:
    members = cluster["members"]
    pairs = [(i, j, matrix[(i, j)]) for i, j in itertools.combinations(members, 2)]
    minimum = min(value for _, _, value in pairs)
    maximum = max(value for _, _, value in pairs)
    below = [(i, j, value) for i, j, value in pairs if value < 0.70]
    headers = [""] + [f"ch{channel:02d}" for channel in members]
    rows: list[list[str]] = []
    for row_index, left in enumerate(members):
        row = [f"**ch{left:02d}**" if left == cluster["medoid"] else f"ch{left:02d}"]
        for column_index, right in enumerate(members):
            if column_index < row_index:
                row.append("—")
            elif column_index == row_index:
                row.append("1.000")
            else:
                value = matrix[(left, right)]
                formatted = f"{value:.3f}"
                if abs(value - minimum) < 5e-9 or abs(value - maximum) < 5e-9:
                    formatted = f"**{formatted}**"
                row.append(formatted)
        rows.append(row)
    below_text = (
        "；".join(f"ch{i}—ch{j} = {value:.3f}" for i, j, value in below)
        if below
        else "无"
    )
    return "\n".join(
        [
            f"### Cluster {cluster['cluster_id']}（size = {cluster['size']}）",
            "",
            markdown_table(headers, rows),
            "",
            f"- Pair 数：{len(pairs)}；min = **{minimum:.3f}**，max = **{maximum:.3f}**，mean = **{mean(v for _, _, v in pairs):.3f}**。",
            f"- 低于 0.70 的 pair：{below_text}。",
            "- Average linkage 判断的是簇间平均距离，因此不要求最终簇内每一对都达到 0.70；bootstrap consensus refinement 用于拆分不稳定部分。",
        ]
    )


def main() -> None:
    args = parse_args()
    r070_dir = args.r070_dir.resolve()
    r080_dir = args.r080_dir.resolve()
    cluster_path = r070_dir / "clusters/clusters_conv1.json"
    summary_path = r070_dir / "correlation_clustering_summary.json"
    clusters_document = json.loads(cluster_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["layers"]["conv1"]
    clusters = clusters_document["clusters"]
    representatives = clusters_document["representative_channels"]
    non_singletons = [cluster for cluster in clusters if cluster["size"] > 1]
    matrix = load_matrix(r070_dir / "matrices/corr_robust_conv1.csv")

    r080_document = json.loads(
        (r080_dir / "clusters/clusters_conv1.json").read_text(encoding="utf-8")
    )
    r080_summary = json.loads(
        (r080_dir / "correlation_clustering_summary.json").read_text(encoding="utf-8")
    )["layers"]["conv1"]
    r080_reps = set(r080_document["representative_channels"])
    r070_reps = set(representatives)
    removed = sorted(r080_reps - r070_reps)
    added = sorted(r070_reps - r080_reps)
    constrained_count = valid_combination_count(clusters)
    raw_count = math.comb(len(representatives), 4)

    lines = [
        "# Conv1 Correlation Clustering 结果整理（r = 0.70）",
        "",
        "*threshold = |r| ≥ 0.70（distance ≤ 0.30），average linkage HCA*  ",
        "*日期：2026-07-31*",
        "",
        "## 总览",
        "",
        markdown_table(
            ["项目", "数值"],
            [
                ["Conv1 总 channels", "64"],
                ["Numerically ineligible（post-ReLU 近常量）", "8（ch 2, 4, 7, 9, 13, 36, 38, 48）"],
                ["参与聚类的 eligible channels", "56"],
                ["HCA 原始 cluster 数", str(summary["raw_cluster_count_before_stability_filter"])],
                ["Bootstrap refinement 后 cluster 数", str(summary["cluster_count"])],
                ["其中 singleton cluster", str(summary["singleton_clusters"])],
                ["有 2 个以上成员的 cluster", f"{len(non_singletons)}（" + ", ".join(f"cluster {c['cluster_id']}" for c in non_singletons) + "）"],
                ["Primary representative（medoid）", str(summary["medoid_count"])],
                ["Second representative", f"{summary['second_representative_count']}（" + ", ".join(f"cluster {c['cluster_id']}" for c in non_singletons if c['second_representative'] is not None) + "）"],
                ["**最终 representative channels 总数**", f"**{summary['representative_count']}**"],
                ["Silhouette coefficient（raw HCA）", "0.224"],
                ["Bootstrap ARI 均值", f"{summary['mean_bootstrap_ari']:.3f}"],
                ["Bootstrap pair retention 均值", f"{summary['mean_pair_retention']:.3f}"],
                ["Bootstrap rejected raw clusters", str(summary["bootstrap_rejected_raw_cluster_count"])],
            ],
        ),
        "",
        "r=0.70 的 raw HCA 得到 27 个簇；稳定性规则将不稳定部分拆分后得到 30 个最终簇。后续搜索应使用这 30 个最终簇，而不是未经过 bootstrap refinement 的 27 个 raw clusters。",
        "",
        "## 九个非 Singleton Cluster 详情",
        "",
    ]

    for cluster in non_singletons:
        lines.extend(
            [
                f"### Cluster {cluster['cluster_id']}（size = {cluster['size']}）"
                + (" ⭐ 最大 cluster" if cluster["size"] == max(c["size"] for c in clusters) else ""),
                "",
                markdown_table(
                    ["角色", "Channel", "Cross-light NCC", "Robust Gradient Energy", "Quality Percentile"],
                    member_rows(cluster),
                ),
                "",
                f"- **Second representative 判定**：{trigger_text(cluster)}。",
                f"- **Bootstrap 稳定性**：cluster stability = {cluster['bootstrap_cluster_stability']:.3f}，minimum pair stability = {cluster['bootstrap_cluster_minimum_pair_stability']:.3f}。Medoid frequency："
                + ", ".join(f"ch{channel} {count}/20" for channel, count in cluster["bootstrap_medoid_frequency"].items())
                + "。",
                f"- **解读**：{interpretation(cluster)}",
                "",
            ]
        )

    representative_rows = []
    for cluster in clusters:
        for channel, role in (
            (cluster["medoid"], "medoid"),
            (cluster["second_representative"], "second"),
        ):
            if channel is None:
                continue
            metric = cluster["member_metrics"][str(channel)]
            note = ""
            if cluster["size"] > 1:
                note = f"size={cluster['size']} {role}"
            representative_rows.append(
                (
                    metric["cross_light_ncc"],
                    [
                        str(cluster["cluster_id"]),
                        f"**ch {channel}**" if role == "medoid" and cluster["size"] > 1 else f"ch {channel}" + (" *(2nd)*" if role == "second" else ""),
                        f"{metric['cross_light_ncc']:.3f}",
                        f"{metric['robust_gradient_energy']:.5f}",
                        f"{metric['quality_percentile_score']:.3f}",
                        note,
                    ],
                )
            )
    representative_rows.sort(key=lambda item: item[0], reverse=True)
    lines.extend(
        [
            "## 全部 30 个 Cluster 的代表通道汇总",
            "",
            "按 Cross-light NCC 排序（光照稳定性从高到低；含 6 个 second representatives）。",
            "",
            markdown_table(
                ["Cluster", "代表 Channel", "Cross-light NCC", "Robust Gradient Energy", "Quality Percentile", "备注"],
                [row for _, row in representative_rows],
            ),
            "",
            "## 非 Singleton Cluster 内部 Pairwise Correlation 详表",
            "",
            "数值为 **min(median |r| clean, median |r| light)**，即 robust correlation。主阈值为 0.70。",
            "",
        ]
    )
    for cluster in non_singletons:
        lines.extend([pairwise_section(cluster, matrix), ""])

    all_metrics: dict[int, dict] = {}
    for cluster in clusters:
        all_metrics.update({int(channel): metric for channel, metric in cluster["member_metrics"].items()})
    low_ncc = sorted(all_metrics.items(), key=lambda item: item[1]["cross_light_ncc"])
    low_ncc = [(channel, metric) for channel, metric in low_ncc if metric["cross_light_ncc"] < 0.80]
    high_gradient = sorted(
        all_metrics.items(), key=lambda item: item[1]["robust_gradient_energy"], reverse=True
    )[:6]
    lines.extend(
        [
            "## 需要特别关注的 Channels",
            "",
            "### 低 NCC（illumination response 差异大）",
            "",
            markdown_table(
                ["Channel", "Cross-light NCC", "Gradient Energy", "r=0.70 角色"],
                [
                    [
                        f"ch {channel}",
                        f"{metric['cross_light_ncc']:.3f}",
                        f"{metric['robust_gradient_energy']:.5f}",
                        "representative" if channel in r070_reps else "cluster member；需 swap-back 才能直接评估",
                    ]
                    for channel, metric in low_ncc
                ],
            ),
            "",
            "NCC 低不等于应当删除。它只表示 clean/light 下响应变化明显；最终价值仍由 MVS ATE 决定。特别是 ch28 在 r=0.70 下不再是 representative，这是降低阈值的主要安全风险之一。",
            "",
            "### 高 Gradient Energy（空间结构响应强）",
            "",
            markdown_table(
                ["Channel", "Robust Gradient Energy", "Cross-light NCC", "r=0.70 角色"],
                [
                    [
                        f"ch {channel}",
                        f"{metric['robust_gradient_energy']:.5f}",
                        f"{metric['cross_light_ncc']:.3f}",
                        "representative" if channel in r070_reps else "cluster member",
                    ]
                    for channel, metric in high_gradient
                ],
            ),
            "",
            "### Numerically Ineligible Channels",
            "",
            "**ch 2, 4, 7, 9, 13, 36, 38, 48**",
            "",
            "这些 channel 在 30 帧 post-ReLU feature maps 中空间标准差持续接近零，无法稳定计算 Pearson correlation，因此不参与 Step B。这里称为 numerically ineligible，不额外把 Step A functional label 当作删除依据。",
            "",
            "## 与 r=0.80 的直接比较",
            "",
            markdown_table(
                ["项目", "r=0.80", "r=0.70", "变化"],
                [
                    ["Final clusters", str(r080_summary["cluster_count"]), str(summary["cluster_count"]), "−10"],
                    ["Representatives", str(r080_summary["representative_count"]), str(summary["representative_count"]), "−7"],
                    ["Singleton clusters", str(r080_summary["singleton_clusters"]), str(summary["singleton_clusters"]), f"−{r080_summary['singleton_clusters'] - summary['singleton_clusters']}"],
                    ["Second representatives", str(r080_summary["second_representative_count"]), str(summary["second_representative_count"]), f"+{summary['second_representative_count'] - r080_summary['second_representative_count']}"],
                    ["Silhouette", "0.119", "0.224", "提高"],
                    ["Bootstrap ARI", f"{r080_summary['mean_bootstrap_ari']:.3f}", f"{summary['mean_bootstrap_ari']:.3f}", "略降"],
                    ["Pair retention", f"{r080_summary['mean_pair_retention']:.3f}", f"{summary['mean_pair_retention']:.3f}", "略升"],
                    ["合法 4-channel combinations", f"{valid_combination_count(r080_document['clusters']):,}", f"{constrained_count:,}", f"减少 {(1-constrained_count/valid_combination_count(r080_document['clusters']))*100:.1f}%"],
                ],
            ),
            "",
            f"- r=0.80 representatives 中被 r=0.70 主空间移除：**{', '.join(f'ch {channel}' for channel in removed)}**。",
            f"- r=0.70 新成为 representative：**{', '.join(f'ch {channel}' for channel in added)}**。",
            "- 已知 smoke-test 组合 **[5,29,40,52]** 的四个 channel 在 r=0.70 中仍全部保留，且不违反同簇约束。",
            "- 旧的解释性组合 **[6,28,34,62]** 中 ch28 不再是 representative；需要 cluster swap-back 才能重新评估该原始组合。",
            "",
            "## 对后续搜索的影响",
            "",
            f"- **原始组合数**：C(36,4) = **{raw_count:,}**。",
            f"- **应用同簇不共存约束后的精确组合数**：**{constrained_count:,}**。",
            f"- **相对 r=0.80 的 120,953 个合法组合**：减少 **{(1-constrained_count/120953)*100:.1f}%**。",
            f"- **7.5 秒/run 的串行穷举时间**：约 **{constrained_count*7.5/3600:.1f} 小时（{constrained_count*7.5/86400:.1f} 天）**，尚未计入 fail-fast 节省。",
            "- **硬约束**：同一最终 cluster 的 medoid 与 second representative 不得同时进入一个组合。",
            "- **必要补偿**：如果采用 r=0.70 作为主搜索空间，应把 cluster swap-back 设为必做步骤，尤其检查 ch28、ch42、ch44、ch57、ch58 等在 r=0.80 中仍为代表、但在 r=0.70 中被合并的 channel。",
            "",
            "## 结论与决策提示",
            "",
            "r=0.70 在统计诊断上并非明显失稳：silhouette 提高，bootstrap ARI 仍为 0.877，经过 consensus refinement 后得到 30 个最终簇。但它从“保守去除高度冗余”转向“更积极地合并中等相关 channel”，使 ch28 等具有已知解释价值的 channel 离开主代表空间。",
            "",
            "因此，r=0.70 可以作为降低 exhaustive-search 成本的候选阈值或 sensitivity experiment；若作为主阈值，必须结合 second representatives、Top-K cluster swap-back 和完整 MVS ATE，不能仅凭 correlation clustering 宣称被合并成员可完全互换。r=0.80 则更适合作为保守主实验，配合 budgeted multi-start search 控制运行时间。",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
