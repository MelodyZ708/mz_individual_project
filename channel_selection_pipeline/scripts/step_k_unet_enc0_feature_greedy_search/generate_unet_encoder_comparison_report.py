#!/usr/bin/env python3
"""Build the Markdown source for the U-Net encoder greedy-search report.

The figures are deliberately drawn from the saved, repeated-run summaries.  The
ResNet comparison is copied only from the approved lightswitch Stage-2
full-sequence report.  Earlier non-lightswitch material is intentionally not
used as evidence for the ResNet channel-count discussion.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ENC1 = {
    "best": ("[d5,d6,d17,d18,d28,d30]", 6.7335),
    "g2": ("[d0,d5]", 8.0837),
    "g3": ("[d0,d5,d18]", 8.4908),
    "g4": ("[d0,d5,d18,d30]", 7.1673),
    "g5": ("[d0,d5,d6,d18,d30]", 7.6408),
    "g6": ("[d0,d5,d6,d17,d18,d30]", 6.9553),
    "random": ("[d0,d4,d10,d22,d26]", 8.3566),
    "all": ("all 32 channels", 18.9322),
    "bqs": ("[d4,d9,d10,d15,d30]", 18.4940),
}

ENC0 = {
    "g1": ("[d3]", 24.5318),
    "g2": ("[d2,d14]", 10.3332),
    "g3": ("[d3,d7,d12]", 6.0004),
    "g4": ("[d2,d3,d12,d14]", 6.3905),
    "g5": ("[d2,d3,d7,d12,d13]", 7.0959),
    "best": ("[d2,d3,d7,d12,d13,d14]", 5.9208),
    "random": ("[d2,d4,d6]", 6.6111),
    "all": ("all 16 channels", 12.9502),
    "bqs": ("[d0,d3,d10,d14,d15]", 13.7216),
}

RESNET = [
    ("Accuracy-first best", "[5,6,24,29]", 14.0623, "current 4-channel champion"),
    ("Independent/top-2", "[1,26,30,40]", 14.3273, "Stage-2 rank 2"),
    ("Balanced", "[1,5,24,29]", 14.6058, "best all-metric compromise"),
    ("Historical CNN baseline", "[5,29,40,52]", 15.1682, "comparison baseline"),
]


def reduction(old: float, new: float) -> float:
    return (old - new) / old * 100.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    e0_best = ENC0["best"][1]
    e1_best = ENC1["best"][1]
    resnet_best = RESNET[0][2]
    e0_g4 = ENC0["g4"][1]
    e1_g4 = ENC1["g4"][1]

    md = f"""---
title: "U-Net Enc0 / Enc1 直接全序列 Greedy 搜索结果与 ResNet 对比"
subtitle: "fr1/desk_lightswitch；报告日期：2026-08-16"
lang: zh-CN
---

# 执行摘要

本阶段在完整 **573-frame fr1/desk_lightswitch** 上直接以历史主指标（keyframe `evo_ape --align --correct_scale` 的 translation ATE mean）搜索 U-Net 的两个 encoder level，避免再用局部 MVS/BQS 代理最终性能。两层都形成了可重复的低误差组合：**Enc0 的最终最优六通道 [d2,d3,d7,d12,d13,d14] 为 5.9208 cm，Enc1 的最终最优六通道 [d5,d6,d17,d18,d28,d30] 为 6.7335 cm**；两者均 3/3 次通过且结果一致。

在同一序列、同一主指标下，Enc0 的最佳四通道为 **[d2,d3,d12,d14]，6.3905 cm**，Enc1 的最佳四通道为 **[d0,d5,d18,d30]，7.1673 cm**；二者均明显低于当前 ResNet Conv1 四通道冠军 [5,6,24,29] 的 14.0623 cm。该结论仅说明本 lightswitch 序列上的直接选择结果；跨序列鲁棒性仍需后续验证。

关键的机制结论不是“通道越多越好”：Enc0 的单通道 d3 可以完成跟踪但误差为 24.5318 cm，加入 d7、d12 后三通道降至 6.0004 cm；Enc1 的所有 32 个 singleton 都失败，却可由两通道起获得稳定组合。这表明结果来自通道间互补，而不是某一个单独的“万能 channel”。

# 1. 实验目标、设置与可比性边界

## 1.1 本阶段的问题

目标是在 U-Net Enc0 与 Enc1 的特征中寻找少量、可直接驱动 COMO tracking 的 channel 组合，并同时记录 1--6 通道时的行为。最大通道数为 6；每个 layer 的候选通过全序列直接运行选择，不依赖 MVS/BQS 的排序。

## 1.2 共同实验设置

| 项目 | 设置 |
|---|---|
| 数据 | `rgbd_dataset_freiburg1_desk_lightswitch`，573 个 matched timestamps |
| Tracking feature | U-Net Enc0 或 Enc1 的指定 channel；灰度 mapping 与 sensor depth 保持不变 |
| 主排名指标 | keyframe trajectory 的 Sim(3)-aligned translation ATE mean（`evo_ape --align --correct_scale`） |
| 单次上限 | 300 s；NaN/Inf 或已知 runtime signature 立即记为 tracking failure |
| 重复验证 | 每个最终/对照组合 3 次；本次重复的标准差均为 0.0000 cm |
| Channel 标识 | `d#` 是该 U-Net level 内的局部 channel index，不可跨 level 解释为相同语义 |

## 1.3 哪些结果可以直接比较

| 结果组 | 可否与本报告 U-Net 数值直接排名 | 原因 |
|---|---|---|
| 本报告的 Enc0 与 Enc1 | 可以 | 同一完整序列、同一主指标、同一运行框架 |
| 当前 ResNet Conv1 四通道 full-sequence 结果 | 可以 | 同样为 573-frame lightswitch，且采用相同历史 ATE mean 口径 |
| 早期非-lightswitch ResNet 1/2/3-channel 实验 | 不纳入本报告 | 不是本报告要讨论的 illumination setting，不能作为 lightswitch 证据 |

# 2. 搜索策略与完成情况

Enc1 的 32 个 singleton 全部失败，因此从穷举的二通道组合中挑选多个可行 seed，再以 greedy 扩展、one-channel swap 和最终重复验证。Enc0 有 16 个候选 channel；只有 d3 singletons 可通过，因此保留它，同时以最优 pair seed 补充多个起点，避免被单一初始 channel 锁定。两个 level 都保留每个通道数下的最好结果，而不是假设误差会随 K 单调下降。

| Level | 候选 channel | Singleton 通过情况 | Pair 搜索 | 最终搜索特点 |
|---|---:|---|---|---|
| Enc0 | 16 | 1/16：d3 可通过 | 120 个 pair；92 PASS | singleton + 多个优质 pair seed 的 multi-start greedy |
| Enc1 | 32 | 0/32 | 496 个 pair；281 PASS | 多个 pair seed 的 multi-start greedy，随后 one-swap |

灰度 baseline 在两个 U-Net 实验中均为 0/3，通过失败；因此此处的 improvement 均建立在“能够稳定完成完整序列”的 CNN feature configurations 之间。

# 3. Enc0 结果

## 3.1 各通道数的最好结果

| K | 最好组合 | ATE mean (cm) | 解读 |
|---:|---|---:|---|
| 1 | [d3] | 24.5318 | 不失败，但精度很弱；说明“能跟踪”不等于“足够准确” |
| 2 | [d2,d14] | 10.3332 | 组合已明显优于单 channel |
| 3 | [d3,d7,d12] | 6.0004 | 关键跃迁；相对 d3 降低 {reduction(ENC0['g1'][1], ENC0['g3'][1]):.1f}% |
| 4 | [d2,d3,d12,d14] | 6.3905 | 固定四通道时的最优保留方案 |
| 5 | [d2,d3,d7,d12,d13] | 7.0959 | 并非单调改善，不能把 K=5 自动视为更优 |
| 6 | [d2,d3,d7,d12,d13,d14] | **5.9208** | 最终全局最优；3/3 PASS |

全 16 通道为 12.9502 cm，而最优六通道为 5.9208 cm，误差降低 **{reduction(ENC0['all'][1], e0_best):.1f}%**。早期 BQS top-5 控制组 [d0,d3,d10,d14,d15] 为 13.7216 cm，也说明“按代理挑出的高分 channel”不等同于完整序列最优组合。

## 3.2 结构性观察

- d3 是 Enc0 唯一单独可通过的 channel，却不能独立提供低误差；它与 d7、d12 的组合才产生主要收益。
- d2、d14 在四通道最优中出现；d7、d13 在六通道终点中再次带来收益。这个 pattern 更接近**不同空间/光度线索互补**，而非重复响应。
- K=3 已达到 6.0004 cm，实际上略优于 K=4。因而后续跨序列验证应同时保留 Enc0 K=3、K=4 和 K=6，而不能只带最终六通道。

# 4. Enc1 结果

## 4.1 各通道数的最好结果

| K | 最好组合 | ATE mean (cm) | 解读 |
|---:|---|---:|---|
| 1 | 无（32/32 failure） | -- | 任何单 channel 都不足以维持该序列 tracking |
| 2 | [d0,d5] | 8.0837 | 最小可行组合已经稳定 |
| 3 | [d0,d5,d18] | 8.4908 | 并非每次增加 channel 都获益 |
| 4 | [d0,d5,d18,d30] | 7.1673 | 固定四通道的最优结果 |
| 5 | [d0,d5,d6,d18,d30] | 7.6408 | 暂时恶化，呈明显非单调性 |
| 6 | [d5,d6,d17,d18,d28,d30] | **6.7335** | one-swap 后的最终最优；3/3 PASS |

最终六通道相对于 Enc1 直接 greedy 的 G6 [d0,d5,d6,d17,d18,d30]（6.9553 cm）再降低 **{reduction(ENC1['g6'][1], e1_best):.1f}%**。同时，它相对于 all-32 的 18.9322 cm 降低 **{reduction(ENC1['all'][1], e1_best):.1f}%**，相对于早期 BQS top-5 控制组的 18.4940 cm 降低 **{reduction(ENC1['bqs'][1], e1_best):.1f}%**。

## 4.2 结构性观察

- Enc1 的 singleton 全失败，却有 281/496 个 pair 可通过：该层的有效信息是**至少两种 feature 的协同**，并不集中于一个单独 channel。
- d0、d5、d18、d30 组成很强的四通道基础；最终 swap 将 d0 换为 d28，同时加入 d17，得到更低的六通道误差。这说明局部搜索后的替换仍然有效，而不是只有初始 greedy 路径。
- K=4（7.1673 cm）已经接近 K=6（6.7335 cm），为后续强调速度/通道预算的实验提供了一个强的 compact option。

# 5. Enc0 与 Enc1 的正面对比

| 比较项 | Enc0 | Enc1 | 结论 |
|---|---:|---:|---|
| 最优 4-channel ATE mean | 6.3905 cm | 7.1673 cm | Enc0 低 **{reduction(e1_g4, e0_g4):.1f}%** |
| 最优 6-channel ATE mean | 5.9208 cm | 6.7335 cm | Enc0 低 **{reduction(e1_best, e0_best):.1f}%** |
| 最小可行结构 | 1 channel 能通过，但很弱 | 必须至少 2 channels | Enc0 有一个可运行锚点；Enc1 依赖更强的配对互补 |
| 最早的低误差点 | K=3：6.0004 cm | K=4：7.1673 cm | Enc0 在更小预算下已给出很强结果 |
| 通道数规律 | K=3、K=6 强，K=4/K=5 不单调 | K=2/K=4/K=6 强，K=3/K=5 不单调 | 两层都反驳“更多 channel 必然更好” |

**主要解读：**在这个单一 lightswitch 序列上，Enc0 的最优值与小通道预算都略占优势；Enc1 的“所有 singletons 失败但部分 pairs 成功”则提供了更清晰的互补性证据。两层都值得保留：Enc0 是当前 accuracy-first 候选，Enc1 是不同层级、不同组合逻辑的独立候选，而不是 Enc0 的简单替代。

# 6. 与当前 ResNet Conv1 四通道结果的直接比较（lightswitch）

下表来自既有的 ResNet 第二阶段完整序列报告，使用相同数据、相同 historical keyframe ATE mean 主指标。因此它可以与 U-Net 四通道结果做数值比较。

| 方法 / 候选 | Channels | K | ATE mean (cm) | 相对 ResNet 当前冠军 |
|---|---|---:|---:|---:|
| U-Net Enc0 best-4 | [d2,d3,d12,d14] | 4 | **6.3905** | 低 **{reduction(resnet_best, e0_g4):.1f}%** |
| U-Net Enc1 best-4 | [d0,d5,d18,d30] | 4 | **7.1673** | 低 **{reduction(resnet_best, e1_g4):.1f}%** |
| ResNet accuracy-first best | [5,6,24,29] | 4 | 14.0623 | reference |
| ResNet independent/top-2 | [1,26,30,40] | 4 | 14.3273 | higher 1.9% |
| ResNet rank 3 | [15,17,52,59] | 4 | 14.5576 | higher 3.5% |
| ResNet balanced | [1,5,24,29] | 4 | 14.6058 | higher 3.9% |
| ResNet rank 5 | [5,6,15,35] | 4 | 14.7234 | higher 4.7% |
| ResNet rank 6 | [6,10,34,41] | 4 | 15.1291 | higher 7.6% |
| ResNet historical CNN baseline | [5,29,40,52] | 4 | 15.1682 | higher 7.9% |

若比较不受四通道约束的最终结果，Enc0 K=6 为 5.9208 cm、Enc1 K=6 为 6.7335 cm，分别比 ResNet 当前四通道冠军低 **{reduction(resnet_best, e0_best):.1f}%** 与 **{reduction(resnet_best, e1_best):.1f}%**。这是一项强的同序列结果，但还不能推出 U-Net 在不同照明、模糊、遮挡或跨数据集上都优于 ResNet；后续应以当前 ResNet best-4 和 balanced-4 作为并列对照做多序列验证。

# 7. ResNet 的 lightswitch 结果边界：本次仅有四通道证据

`Full_Sequence_第二阶段结果汇报_中文.docx` 明确记录：该阶段只评估了 **Conv1 四通道**组合。因此，能够在 lightswitch 条件下与 U-Net 进行数值比较的 ResNet 证据是上节的七个四通道配置；该记录中没有可用于本报告的 ResNet 1、2 或 3 通道 full-sequence 结果。

这并不表示 ResNet 的 1/2/3 通道一定无效，而是说明它们在当前 lightswitch 设定下**尚未被本次完整序列协议测量**。之前不含 lightswitch 的早期记录不能替代这一缺口，故本修订版不再引用其数字或以其推断 ResNet 与 U-Net 的层级机制。若后续要做严格的 channel-count 对比，应在同一 573-frame lightswitch sequence、同一 ATE mean 口径下，对 ResNet Conv1/Layer1/Layer2 分别运行 K=1--4（至少包含最优 singleton、pair、triple 和 four-channel）后再讨论。

# 8. 结论与下一步

1. **当前单序列 accuracy-first 选择：**Enc0 [d2,d3,d7,d12,d13,d14]（5.9208 cm）。若固定四通道，选择 Enc0 [d2,d3,d12,d14]（6.3905 cm）。
2. **独立层级候选：**Enc1 [d5,d6,d17,d18,d28,d30]（6.7335 cm）；固定四通道可用 [d0,d5,d18,d30]（7.1673 cm）。
3. **建议后续验证池：**Enc0 K=3/K=4/K=6、Enc1 K=4/K=6、ResNet [5,6,24,29] 与 ResNet balanced [1,5,24,29]。这样可测试通道预算、层级和“准确度 vs 局部稳定性”三种因素。
4. **不要只报告最终六通道：**两个 U-Net layer 的 K-path 都非单调；四通道及 Enc0 三通道是科学上重要的相对照，而不是失败的中间产物。
5. **ResNet 的通道数边界：**目前 lightswitch 下只能公平比较 ResNet 的四通道配置；不将非-lightswitch 的 1/2/3-channel 数值混入本报告。
6. **局限性：**本报告的数值优劣仅在一个 full lightswitch 序列成立；重复运行结果一致证明配置可复现，但不能替代跨序列泛化或统计置信区间。

# 数据来源与可复现文件

- Enc1：`channel_selection_results/step_j_unet_direct_fullseq_greedy/summary.md`、`final_repeats_summary.csv`、`direct_greedy_path.csv`、`evaluations.sqlite3`。
- Enc0：`channel_selection_results/step_k_unet_enc0_direct_fullseq_greedy/summary.md`、`final_repeats_summary.csv`、`direct_greedy_path.csv`、`seed_selection_plan.json`、`evaluations.sqlite3`。
- 可直接比较的 ResNet（仅 lightswitch 的 Conv1 四通道）：`channel_selection_results/reports/full_sequence_stage2_advisor_report/Full_Sequence_第二阶段结果汇报_中文.docx`。
"""
    args.output.write_text(md, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
