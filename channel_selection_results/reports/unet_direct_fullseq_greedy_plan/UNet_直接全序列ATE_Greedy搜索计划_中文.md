---
title: "UNet Encoder 特征的直接全序列 ATE Greedy 搜索计划"
subtitle: "fr1/desk_lightswitch · 两天预算 · 不使用 MVS 或 Convergence Basin proxy"
author: "MSc Project"
date: "2026年8月14日"
lang: zh-CN
---

# UNet Encoder 特征的直接全序列 ATE Greedy 搜索计划

*fr1/desk_lightswitch · 两天预算 · 不使用 MVS 或 Convergence Basin proxy*  
MSc Project · 2026年8月14日

# 1. 结论先行

建议对 **UNet encoder level 1（enc1，32 通道）** 做一次以真实 full-sequence tracking 表现为唯一搜索依据的、**多起点 forward greedy + 等预算随机对照 + 一次 one-channel swap 局部审计**。每条 greedy 路径最多扩展到 **6 通道**；最终卡数在 1–6 间由真实全序列 ATE 的复跑结果选择。4 通道不是硬上限，而是预先指定的可审计比较点：它将与最终 ResNet 的4通道结果同卡数对照；历史对照使用 6 月 BQS-greedy 路径的4通道前缀 `[d4,d15,d9,d10]`，并在本次同一协议下重新运行。

选择 **full sequence**，而不是 MVS。ResNet 已经给出明确的反例：MVS 上 ATE RMSE 为 6.49 cm 的 `[8,40,50,59]`，回到 full sequence 后为 54.35 cm；而历史 CNN baseline `[5,29,40,52]` 在 MVS / full sequence 分别为 14.81 / 19.94 cm。MVS 对“立即失败”很有价值，却不足以决定最终精度排名。因此本实验不再分 MVS 筛选与全序列验证两阶段，也不以 BQS、convergence basin 或其任何派生量决定 channel。

核心产出不是只报一个“最优组合”，而是同时回答四个问题：

1. 同为4通道时，真实 ATE-greedy 的 $G_4$ 相比历史 **BQS-greedy 路径4通道前缀** $B_4$ 是提升还是下降、幅度多少？
2. 允许最多6通道后，最终自适应 UNet 组合 $G^*$ 相比 $G_4$ 的收益或代价多少？
3. greedy 本身是否优于同等 full-sequence 运行预算、且按卡数分层的随机搜索？
4. greedy 终点是否被一个简单的 one-channel swap 改善，从而量化 greedy 的局部贪心损失？
5. $G_4$ 与 $G^*$ 分别相对于 gray、UNet-all 和最终 ResNet full-sequence 基线处于什么位置？

# 2. 已知证据与本次设计的边界

## 2.1 6 月 UNet 初步探索可以复用什么、不能复用什么

0621 文档已经表明，UNet enc1 是比 enc0 更有希望的搜索空间：enc1 有 32 个通道、半分辨率且更大感受野；当时的 BQS Top-5 为 `[d4,d15,d9,d10,d30]`。旧结果如下。

| UNet 特征层 | 旧选择策略 | channels | 旧 fr1/desk ATE RMSE mean/cm | 有效运行 |
|---|---|---|---:|---:|
| enc1 | BQS Top-5 | `[d4,d15,d9,d10,d30]` | 6.891 | 5/5 |
| enc1 | BQS Top-3 | `[d4,d15,d9]` | 7.910 | 3/5 |
| enc1 | BQS 单通道 | `[d4]` | 10.180 | 5/5 |
| enc0 | BQS Top-3 | `[d15,d10,d0]` | 7.755 | 3/5 |
| enc0 | BQS 单通道 | `[d15]` | 17.595 | 2/5 |

这些旧数字说明 enc1 值得优先投入；但它们使用的是当时的 fr1/desk 条件、ATE RMSE 与旧运行环境，**不能**与本次 `fr1/desk_lightswitch` 的全序列结果直接做数值对比。旧 BQS Top-5只保留为背景；本次重新运行的历史 anchor 是同一 BQS greedy 路径的4通道前缀 `[d4,d15,d9,d10]`，绝不参与新 greedy 的打分或候选过滤。

## 2.2 从 ResNet 结果得到的三个设计约束

| ResNet 经验 | 对 UNet 搜索的含义 |
|---|---|
| MVS 可高效淘汰立即 tracking failure，但其 ATE 排名可能与 full sequence 完全不同 | 所有候选直接跑 lightswitch full sequence；不以 MVS 排序。 |
| 相关性聚类对 64 个 Conv1 通道有必要；UNet enc1 只有 32 通道 | 在两天预算内直接探索全部 32 个 enc1 通道，不新增相关性/BQS预过滤。 |
| 单一路径 greedy 会遗漏协同替换；ResNet 采用 brute force、swap-back 等方式补救 | 用多起点降低 seed 偏差，并对最佳 greedy 终点完整枚举一跳 swap 邻域。 |

## 2.3 明确不做的事

- 不跑 MVS，也不把 MVS、BQS、basin width、Hessian 条件数或 feature-map 相似度作为选择 proxy。
- 不做 enc0/enc1 双层大搜索、C2F 或 cross-layer 组合；本次只回答“enc1 的真实 ATE-greedy 是否有价值”。
- 不把 convergence basin 当作独立的目标或“第二阶段”。旧 BQS 组合仅是一个历史对照。
- 不把单次 ATE 的改善直接解释成因果机制；feature map / channel ablation 是后续解释工作。

# 3. 固定实验协议

## 3.1 数据、网络与配置

| 项目 | 固定设置 |
|---|---|
| 数据集 | `/home/melody/data/tum/rgbd_dataset_freiburg1_desk_lightswitch`，完整匹配序列 |
| UNet 特征 | `unet_enc_level: 1`，即 32 个 encoder channels；候选全集为 `d0`–`d31` |
| Tracking | `color: unet`；每个候选只改变 `unet_channel_select` |
| Mapping | 固定 `color: gray`，`use_sensor_depth: true`；不使用 GT depth 建图 |
| 特征计算 | 使用当前实现的 `UNetFeatureExtractor.extract(rgb)`，逐帧直接从同一 mapping U-Net 提取，不读旧 cache |
| 通道数 | greedy 路径完整记录 1–6 通道；最终卡数由复跑后的真实ATE选择，最大为6；4通道结果预注册并单独报告 |
| 单次 timeout | 300 s；正常 full-sequence 运行预计远低于该值 |
| 可恢复性 | 每次运行写入 SQLite（`WAL` + `synchronous=FULL`），共享配置加锁并在每次运行后恢复 |

## 3.2 统一评分和失败定义

主指标必须与历史 full-sequence / `run_random_channel_search.sh` 一致：

$$\text{ATE}_{\text{primary}}=\operatorname{mean}\bigl(\texttt{evo\_ape tum GT trajectory --align --correct\_scale}\bigr).$$

按以下顺序排序：

1. `PASS` 优先于所有失败；
2. `PASS` 内按主 ATE mean 升序；
3. 主 ATE 完全相同才按 keyframe evo RPE RMSE、随后 all-frame SE(3) ATE RMSE 打破平局；
4. 不设 RPE 硬门槛。Trans/Rot RPE max 作为明显局部跳变的**标记**，而不是排除低 ATE 候选的 proxy。

`PASS` 要求：非 NaN 位姿/affine/photometric 诊断、没有已知 runtime exception、没有 timeout、全序列 coverage ≥90%、末位姿接近序列结束，且轨迹尾部不冻结。出现 NaN 或 runtime failure 时立即终止该次运行并记录失败帧；这是节省时间的真实 failure handling，不是 MVS proxy。

# 4. 单一直接搜索流程

以下所有候选都运行**同一条完整 lightswitch 序列**。这里按执行顺序写出，但它们不是“筛选MVS → 验证全序列”的两阶段方案。

## 4.1 仅用于协议检查的四个 anchor

先各运行一次，不参与选 seed：

| label | channels | 目的 |
|---|---|---|
| `gray_current` | gray | 当前环境下的必要对照 |
| `unet_all32` | `all` | 未选择的 U-Net 参考 |
| `bqs_greedy_historical_top4` | `[d4,d15,d9,d10]` | 6 月 BQS-greedy 路径的4通道前缀；同卡数历史 anchor |
| `bqs_greedy_historical_top5` | `[d4,d15,d9,d10,d30]` | 6 月 BQS-greedy 的旧Top-5；只作5通道历史参照 |

这四次同时给出真实中位运行时间 $t_{50}$，用于确认两天预算。它们不是独立搜索阶段。

## 4.2 真实 ATE 的多起点 forward greedy

1. **Singleton sweep：** 直接跑全部 32 个单通道 `[d0]` … `[d31]` 的 full sequence；从通过的 singletons 中按真实主 ATE 取最好的 2--4 个作为 seeds。
   **Fallback：** 若 32 个 singleton 全部失败，则不把“没有单通道可用”误判为 UNet 无效，而是在相同的 full-sequence ATE objective 下完整枚举 $\binom{32}{2}=496$ 个两通道组合；从通过的 pairs 中取最好的 2--4 个作为 two-channel seeds，再从 3 通道继续 greedy 扩展到 6。此时 K=2 已被直接穷尽，因此随机对照不再抽 K=2（没有未见 pair），而继续在 K=3--6 做等预算、无放回随机对照。
2. **多条 greedy 路径：** 对每条 seed 路径，依次把一个尚未选中的候选加到集合中。每一步枚举全部可加 channel，直接运行完整序列；选取主 ATE 最低的增量候选，直到 6 通道。
3. **缓存去重：** 每个 canonical channel set 只运行一次。如果两条路径产生同一候选集合，复用 SQLite 结果，而不浪费预算。
4. **不以第一次变差而提前停止：** 记录每一步的 1/2/3/4/5/6 通道集合与 marginal ATE change。某一步加通道后 ATE 变差，仍继续扩展到6通道，因为非单调协同在旧UNet与ResNet中都已出现。搜索结束后才由复跑选择最终卡数；4通道结果始终保留。

理论上限为：

$$32+S\times(31+30+29+28+27)$$

次 full-sequence evaluator invocation（去重后只会更少），其中 $S$ 是由实际运行时间决定的2–4个起点数。这是正常 singleton 成功时的上限。若触发 all-singleton-fail fallback，上限改为

$$32+496+S\times(30+29+28+27)=528+114S,$$

其中 496 个 pair 是一个完整的直接目标搜索，而非 proxy。这不是 BQS multi-start 或 convergence-basin search；正常路径的所有起点完全由真实 singleton ATE 决定，fallback 路径的所有起点完全由真实 pair ATE 决定。

## 4.3 等预算、按通道数分层的随机对照

为检验 greedy 是否真正值得，不能只将其与历史 BQS 组合比较；也不能只随机抽 4 通道组合，因为主方案允许最终卡数是 1--6。Singleton 已经枚举了全部 32 个通道，因此随机对照只需要覆盖 $k=2,3,4,5,6$。

对每个通道数 $k$，令 $N_{G,k}$ 为 direct-greedy 在该卡数实际运行的、去重后的新候选数。以固定公开 seed（建议 `20260814`），从全部

$$\binom{32}{k}$$

个 $k$ 通道子集中均匀、不放回抽取恰好 $N_{G,k}$ 个随机组合；排除同一 $k$ 下已运行的 greedy 集合与对应 BQS anchor。这样每一个 $k$ 的随机搜索与 greedy 都消耗相同数目的真实 full-sequence evaluator calls，而不是由较容易或较难的卡数获得额外预算。

记：

- $G_k$：direct-greedy 路径中、恰为 $k$ 通道的最佳候选（$k=1,\ldots,6$）；
- $R_k$：同卡数随机样本中主 ATE 最低的候选（$k=2,\ldots,6$）；
- $G^*$：在 $G_1,\ldots,G_6$ 之间、经最终复跑后主 ATE 最低的组合；
- $R^*$：在所有 $R_k$ 之间、经最终复跑后主 ATE 最低的随机组合。

其中 $G_4$ 与 $R_4$ 是预先指定的同卡数比较点；$G^*$ 与 $R^*$ 则回答允许 1--6 通道时的实际最优结果。该随机对照不声称穷尽任一 $\binom{32}{k}$ 空间，只检验在同一 full-sequence 预算下 greedy 是否更有效。

## 4.4 对最佳 greedy 终点的一跳 swap 审计

搜索阶段先以单次结果从 $G_1,\ldots,G_6$ 选出 provisional $G^*_{\mathrm{single}}$，记其卡数为 $K^*$。对它完整枚举所有一换一邻居：移除一个已选 channel，再加入一个未选 channel。

$$K^*\times(32-K^*)\leq 6\times26=156$$

个候选（已在缓存中的组合不重跑）。得到 $L^*_{\mathrm{single}}$，即 best one-swap neighbour 或 $G^*_{\mathrm{single}}$ 本身。最终以复跑均值确定 $G^*$ 和 $L^*$；若复跑使最优通道数改变，仍完整保留原始 swap audit，且清楚标记其中心是哪个 $K^*$。

这一步很重要：它不将另一个 proxy 引入选择，而是在真实 full-sequence objective 下，精确量化“单纯 forward greedy 的一个局部改动还剩多少收益”。若 $L^*=G^*$，至少能说明 $G^*$ 在完整一跳邻域内稳定；若 $L^*$ 更好，则报告 greedy 的可测局部损失，而不夸大为全局最优性证明。

# 5. 最终复跑与要报告的数字

候选搜索阶段的运行默认各一次，避免将两天预算耗在重复上。搜索结束后，对去重后的下列候选补足到 **3 次总观察**：

- gray；
- UNet-all32；
- 两个历史 BQS-greedy anchors：$B_4=[d4,d15,d9,d10]$、$B_5=[d4,d15,d9,d10,d30]$；
- $G_1,\ldots,G_6$：每个 greedy 卡数的 endpoint（其中 $G_4$ 是固定比较点）；
- $R_4$ 与 $R^*$；
- swap 后最优 $L^*$。

若候选重合，合并；最多约 13 个配置，因此最多新增约 26 次运行。这样最终卡数 $G^*$ 由 1--6 个 endpoint 的**三次均值**选择，而不是由单次偶然最小值选择。每个最终配置报告 mean、std、median、min/max ATE，PASS 次数、keyframe RPE、all-frame SE(3) ATE/RPE、coverage、运行时间和失败帧。

核心效应量统一为：

$$\Delta(A\rightarrow C)=\frac{\operatorname{ATE}(A)-\operatorname{ATE}(C)}{\operatorname{ATE}(A)}\times100\%.$$

正值表示 $C$ 改善。至少报告：

| 比较 | 要回答的问题 |
|---|---|
| $B_4 \rightarrow G_4$ | 同为4通道时，真实 ATE-greedy 相对历史 BQS-greedy 路径前缀提升/下降多少？ |
| $G_4 \rightarrow G^*$ | 允许自适应选择1--6通道带来的收益或代价多少？ |
| $R_4 \rightarrow G_4$ | 同为4通道、同预算时，greedy 是否优于随机搜索？ |
| $R^* \rightarrow G^*$ | 允许自适应卡数时，greedy 是否优于等预算分层随机搜索？ |
| $G^* \rightarrow L^*$ | forward greedy 在一跳局部邻域中还留下多少可恢复收益？ |
| gray / UNet-all32 / $B_5$ $\rightarrow L^*$ | channel selection 相对简单、未选择 U-Net 与旧5通道 BQS 对照的净效果是什么？ |

若最终 3 次结果的置信区间重叠，报告“未显示出可分辨优势”，而不是仅依据一次最小值宣称胜出。

# 6. 两天预算与执行保护

默认最大工作量为：

| 项目 | 最多 full-sequence runs |
|---|---:|
| 四个 anchor | 4 |
| 4 / 3 / 2 起点 direct ATE-greedy | 612 / 467 / 322 |
| 对应的等预算分层随机 | 580 / 435 / 290 |
| 最佳 greedy 的一跳 swap | 最多 156 |
| 最终候选补足到 3 次 | 最多约 26 |
| **合计上限** | **约 1,378 / 1,088 / 798** |

以 ResNet full-sequence 的既有经验（约 60–90 s/次）估算，三种规模的纯运行时间约为 23.0 h / 27.2 h / 26.6 h；加上20%安全余量约 27.6 h / 32.6 h / 31.9 h。UNet 实际时间以 anchor 的 $t_{50}$ 为准。

若触发 all-singleton-fail fallback，pair sweep 使上限变为约 **1,626 / 1,398 / 1,170** 次（4/3/2 starts）。其中 K=2 直接穷尽，随机对照只在 K=3--6 抽取；singleton/pair 失败通常会较快退出，因此实际新增墙钟时间需要以该 fallback 的首批运行时间重新估算。

| anchor 测得的 $t_{50}$ | 执行规模 |
|---|---|
| ≤60 s | 默认：4 starts；direct greedy 最多612次、分层随机580次、完整最多156个 swap。 |
| 60–100 s | 3 starts；direct greedy 最多467次、分层随机435次、完整最多156个 swap。 |
| >100 s | 2 starts；direct greedy 最多322次、分层随机290次、完整最多156个 swap；保留至少12小时给最终复跑与汇总。 |

无论哪种规模，启动后在约第36小时停止新增搜索候选，只完成已启动任务与最终复跑。所有队列、抽样 seed、配置 hash、软件版本、每次完整输出和 SQLite 记录必须保存；中断后从数据库中未完成的 `(candidate, replicate)` 恢复。

# 7. 预期结论的解释边界

1. 如果 $G_4$ 显著优于 $B_4$，结论是：**在 lightswitch full-sequence objective 上、同为4通道时，直接 ATE 驱动的选择优于 BQS-greedy 的4通道前缀 anchor**；如果 $G^*$ 进一步更好，则另报告自适应卡数的实际收益。二者都不是“BQS 没有价值”的一般结论。
2. 如果 $R^*$ 与 $G^*$ 接近或更好，结论是：该预算下 greedy 没有显示出明显的搜索效率优势，UNet 通道交互可能很强；不是“UNet feature 无效”。
3. 如果 $L^*$ 明显优于 $G^*$，结论是：forward greedy 有可观的局部最优损失；这正是必须报告 swap audit 的原因。
4. 如果所有 CNN/UNet 配置均不及 gray，结论只能限于当前 decoupled mapping、sensor-depth 与 lightswitch 序列；不能外推到 U-Net 对深度/不确定性建模本身无用。
5. 单一 sequence 的选择仍可能过拟合。两天后最值得做的后续验证，是只取 $B_4,B_5,G_4,G^*,R^*,L^*$ 与 gray 到未参与选择的其他序列/退化上评估，而不是在本轮再扩展搜索空间。

# 8. 推荐实施结果目录

建议创建独立目录，绝不与 ResNet 的 Step-D/E 记录混合：

```text
channel_selection_results/
  step_j_unet_direct_fullseq_greedy/
    evaluations.sqlite3
    candidate_plan.json
    direct_greedy_path.csv
    random_budget_matched_plan.json
    one_swap_audit.csv
    final_repeats_summary.csv
    all_evaluations.csv
    trajectories/
    logs/
    summary.md
```

这使最终文档能够审计每一个 greedy 决策、随机对照的抽样过程，以及“最终组合相对 greedy”的实际提升或下降。

# 9. 最终推荐

**采用 enc1、最多6通道、full-sequence ATE 多起点 greedy；完整记录每条路径的1--6通道结果，并将4通道 $G_4$ 作为预注册同卡数比较点。以按卡数分层的等预算随机和完整 one-swap 审计作为最小但充分的反事实对照。**

它在两天内可执行，并且直接回应导师对 UNet feature 的兴趣：不是再做一次 convergence-basin 解释，而是得到一个在真实 lightswitch tracking 上可量化、可复现、并能说明 greedy 本身有效性与局限性的结果。
