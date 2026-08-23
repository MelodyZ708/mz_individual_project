# Conv1 Correlation Clustering 结果整理（r = 0.70）

*threshold = |r| ≥ 0.70（distance ≤ 0.30），average linkage HCA*  
*日期：2026-07-31*

## 总览

| 项目 | 数值 |
|---|---|
| Conv1 总 channels | 64 |
| Numerically ineligible（post-ReLU 近常量） | 8（ch 2, 4, 7, 9, 13, 36, 38, 48） |
| 参与聚类的 eligible channels | 56 |
| HCA 原始 cluster 数 | 27 |
| Bootstrap refinement 后 cluster 数 | 30 |
| 其中 singleton cluster | 21 |
| 有 2 个以上成员的 cluster | 9（cluster 11, cluster 12, cluster 17, cluster 21, cluster 23, cluster 24, cluster 25, cluster 27, cluster 30） |
| Primary representative（medoid） | 30 |
| Second representative | 6（cluster 11, cluster 12, cluster 21, cluster 23, cluster 25, cluster 30） |
| **最终 representative channels 总数** | **36** |
| Silhouette coefficient（raw HCA） | 0.224 |
| Bootstrap ARI 均值 | 0.877 |
| Bootstrap pair retention 均值 | 0.891 |
| Bootstrap rejected raw clusters | 2 |

r=0.70 的 raw HCA 得到 27 个簇；稳定性规则将不稳定部分拆分后得到 30 个最终簇。后续搜索应使用这 30 个最终簇，而不是未经过 bootstrap refinement 的 27 个 raw clusters。

## 九个非 Singleton Cluster 详情

### Cluster 11（size = 2）

| 角色 | Channel | Cross-light NCC | Robust Gradient Energy | Quality Percentile |
|---|---|---|---|---|
| **Medoid** | ch 6 | 0.901 | 0.01337 | 0.339 |
| Second rep | ch 52 | 0.897 | 0.00607 | 0.259 |

- **Second representative 判定**：robust_gradient_ratio = **2.20**（超过 2.0；簇内空间梯度强度差异明显）。
- **Bootstrap 稳定性**：cluster stability = 1.000，minimum pair stability = 1.000。Medoid frequency：ch6 20/20。
- **解读**：成员的 NCC 较接近，但 gradient energy 相差 2.2 倍；保留 ch6 和 ch52 以覆盖不同响应强度。

### Cluster 12（size = 3）

| 角色 | Channel | Cross-light NCC | Robust Gradient Energy | Quality Percentile |
|---|---|---|---|---|
| **Medoid** | ch 27 | 0.643 | 0.00265 | 0.080 |
| — | ch 28 | 0.767 | 0.00578 | 0.170 |
| Second rep | ch 32 | 0.915 | 0.00271 | 0.241 |

- **Second representative 判定**：NCC spread = **0.272**（超过 0.15）且 gradient ratio = **2.18**（超过 2.0）。
- **Bootstrap 稳定性**：cluster stability = 0.900，minimum pair stability = 0.850。Medoid frequency：ch27 20/20。
- **解读**：簇内 NCC 跨度较大（0.643–0.915），同时 gradient energy 相差 2.2 倍。保留 medoid ch27 与 second representative ch32，降低较低阈值错误压缩不同 illumination behavior 的风险。

### Cluster 17（size = 2）

| 角色 | Channel | Cross-light NCC | Robust Gradient Energy | Quality Percentile |
|---|---|---|---|---|
| **Medoid** | ch 43 | 0.968 | 0.03240 | 0.777 |
| — | ch 57 | 0.968 | 0.01967 | 0.643 |

- **Second representative 判定**：未触发：NCC spread = 0.000（< 0.15），gradient ratio = 1.65（< 2.0）。
- **Bootstrap 稳定性**：cluster stability = 0.850，minimum pair stability = 0.850。Medoid frequency：ch43 20/20。
- **解读**：2 个成员的 illumination response 与梯度强度差异未达到 second-rep 阈值，因此只保留 medoid ch43。

### Cluster 21（size = 2）

| 角色 | Channel | Cross-light NCC | Robust Gradient Energy | Quality Percentile |
|---|---|---|---|---|
| **Medoid** | ch 17 | 0.533 | 0.00040 | 0.027 |
| Second rep | ch 34 | 0.922 | 0.00341 | 0.304 |

- **Second representative 判定**：NCC spread = **0.389**（超过 0.15）且 gradient ratio = **8.59**（超过 2.0）。
- **Bootstrap 稳定性**：cluster stability = 1.000，minimum pair stability = 1.000。Medoid frequency：ch17 20/20。
- **解读**：簇内 NCC 跨度较大（0.533–0.922），同时 gradient energy 相差 8.6 倍。保留 medoid ch17 与 second representative ch34，降低较低阈值错误压缩不同 illumination behavior 的风险。

### Cluster 23（size = 2）

| 角色 | Channel | Cross-light NCC | Robust Gradient Energy | Quality Percentile |
|---|---|---|---|---|
| **Medoid** | ch 1 | 0.956 | 0.02646 | 0.652 |
| Second rep | ch 10 | 0.965 | 0.05538 | 0.821 |

- **Second representative 判定**：robust_gradient_ratio = **2.09**（超过 2.0；簇内空间梯度强度差异明显）。
- **Bootstrap 稳定性**：cluster stability = 0.850，minimum pair stability = 0.850。Medoid frequency：ch1 20/20。
- **解读**：成员的 NCC 较接近，但 gradient energy 相差 2.1 倍；保留 ch1 和 ch10 以覆盖不同响应强度。

### Cluster 24（size = 3）

| 角色 | Channel | Cross-light NCC | Robust Gradient Energy | Quality Percentile |
|---|---|---|---|---|
| **Medoid** | ch 11 | 0.969 | 0.03059 | 0.777 |
| — | ch 42 | 0.972 | 0.03460 | 0.866 |
| — | ch 49 | 0.955 | 0.03448 | 0.714 |

- **Second representative 判定**：未触发：NCC spread = 0.018（< 0.15），gradient ratio = 1.13（< 2.0）。
- **Bootstrap 稳定性**：cluster stability = 0.967，minimum pair stability = 0.950。Medoid frequency：ch11 20/20。
- **解读**：3 个成员的 illumination response 与梯度强度差异未达到 second-rep 阈值，因此只保留 medoid ch11。

### Cluster 25（size = 6）

| 角色 | Channel | Cross-light NCC | Robust Gradient Energy | Quality Percentile |
|---|---|---|---|---|
| — | ch 3 | 0.890 | 0.00155 | 0.125 |
| — | ch 12 | 0.899 | 0.00349 | 0.232 |
| Second rep | ch 29 | 0.924 | 0.01303 | 0.420 |
| — | ch 54 | 0.885 | 0.00043 | 0.098 |
| — | ch 58 | 0.914 | 0.00654 | 0.330 |
| **Medoid** | ch 62 | 0.912 | 0.00214 | 0.196 |

- **Second representative 判定**：robust_gradient_ratio = **30.19**（超过 2.0；簇内空间梯度强度差异明显）。
- **Bootstrap 稳定性**：cluster stability = 1.000，minimum pair stability = 1.000。Medoid frequency：ch62 20/20。
- **解读**：成员的 NCC 较接近，但 gradient energy 相差 30.2 倍；保留 ch62 和 ch29 以覆盖不同响应强度。

### Cluster 27（size = 2）

| 角色 | Channel | Cross-light NCC | Robust Gradient Energy | Quality Percentile |
|---|---|---|---|---|
| **Medoid** | ch 20 | 0.966 | 0.02510 | 0.679 |
| — | ch 44 | 0.961 | 0.02547 | 0.670 |

- **Second representative 判定**：未触发：NCC spread = 0.005（< 0.15），gradient ratio = 1.01（< 2.0）。
- **Bootstrap 稳定性**：cluster stability = 1.000，minimum pair stability = 1.000。Medoid frequency：ch20 20/20。
- **解读**：2 个成员的 illumination response 与梯度强度差异未达到 second-rep 阈值，因此只保留 medoid ch20。

### Cluster 30（size = 13） ⭐ 最大 cluster

| 角色 | Channel | Cross-light NCC | Robust Gradient Energy | Quality Percentile |
|---|---|---|---|---|
| — | ch 16 | 0.983 | 0.00304 | 0.545 |
| — | ch 19 | 0.974 | 0.02191 | 0.750 |
| — | ch 21 | 0.983 | 0.00713 | 0.661 |
| — | ch 22 | 0.972 | 0.01986 | 0.705 |
| — | ch 37 | 0.985 | 0.00275 | 0.562 |
| — | ch 45 | 0.983 | 0.00446 | 0.598 |
| — | ch 46 | 0.982 | 0.00684 | 0.625 |
| — | ch 47 | 0.972 | 0.02855 | 0.786 |
| — | ch 51 | 0.979 | 0.02762 | 0.830 |
| — | ch 53 | 0.975 | 0.01455 | 0.652 |
| — | ch 55 | 0.972 | 0.01295 | 0.589 |
| Second rep | ch 60 | 0.972 | 0.04198 | 0.848 |
| **Medoid** | ch 61 | 0.985 | 0.00353 | 0.616 |

- **Second representative 判定**：robust_gradient_ratio = **15.25**（超过 2.0；簇内空间梯度强度差异明显）。
- **Bootstrap 稳定性**：cluster stability = 1.000，minimum pair stability = 1.000。Medoid frequency：ch61 20/20。
- **解读**：成员的 NCC 较接近，但 gradient energy 相差 15.3 倍；保留 ch61 和 ch60 以覆盖不同响应强度。

## 全部 30 个 Cluster 的代表通道汇总

按 Cross-light NCC 排序（光照稳定性从高到低；含 6 个 second representatives）。

| Cluster | 代表 Channel | Cross-light NCC | Robust Gradient Energy | Quality Percentile | 备注 |
|---|---|---|---|---|---|
| 30 | **ch 61** | 0.985 | 0.00353 | 0.616 | size=13 medoid |
| 30 | ch 60 *(2nd)* | 0.972 | 0.04198 | 0.848 | size=13 second |
| 24 | **ch 11** | 0.969 | 0.03059 | 0.777 | size=3 medoid |
| 26 | ch 41 | 0.968 | 0.01618 | 0.616 |  |
| 17 | **ch 43** | 0.968 | 0.03240 | 0.777 | size=2 medoid |
| 27 | **ch 20** | 0.966 | 0.02510 | 0.679 | size=2 medoid |
| 23 | ch 10 *(2nd)* | 0.965 | 0.05538 | 0.821 | size=2 second |
| 5 | ch 26 | 0.960 | 0.02681 | 0.688 |  |
| 9 | ch 25 | 0.959 | 0.02144 | 0.616 |  |
| 28 | ch 31 | 0.959 | 0.01492 | 0.518 |  |
| 23 | **ch 1** | 0.956 | 0.02646 | 0.652 | size=2 medoid |
| 15 | ch 63 | 0.954 | 0.04497 | 0.750 |  |
| 8 | ch 14 | 0.952 | 0.01651 | 0.518 |  |
| 16 | ch 30 | 0.952 | 0.04002 | 0.705 |  |
| 6 | ch 56 | 0.952 | 0.02129 | 0.545 |  |
| 22 | ch 23 | 0.925 | 0.04214 | 0.705 |  |
| 25 | ch 29 *(2nd)* | 0.924 | 0.01303 | 0.420 | size=6 second |
| 7 | ch 59 | 0.924 | 0.01618 | 0.455 |  |
| 21 | ch 34 *(2nd)* | 0.922 | 0.00341 | 0.304 | size=2 second |
| 18 | ch 39 | 0.918 | 0.01721 | 0.464 |  |
| 10 | ch 50 | 0.916 | 0.02253 | 0.518 |  |
| 12 | ch 32 *(2nd)* | 0.915 | 0.00271 | 0.241 | size=3 second |
| 3 | ch 0 | 0.914 | 0.03169 | 0.580 |  |
| 25 | **ch 62** | 0.912 | 0.00214 | 0.196 | size=6 medoid |
| 14 | ch 15 | 0.903 | 0.05795 | 0.643 |  |
| 11 | **ch 6** | 0.901 | 0.01337 | 0.339 | size=2 medoid |
| 11 | ch 52 *(2nd)* | 0.897 | 0.00607 | 0.259 | size=2 second |
| 29 | ch 35 | 0.896 | 0.00322 | 0.196 |  |
| 2 | ch 40 | 0.889 | 0.01580 | 0.321 |  |
| 1 | ch 5 | 0.874 | 0.02642 | 0.429 |  |
| 4 | ch 24 | 0.873 | 0.03064 | 0.473 |  |
| 19 | ch 33 | 0.868 | 0.00636 | 0.205 |  |
| 13 | ch 8 | 0.867 | 0.01783 | 0.321 |  |
| 12 | **ch 27** | 0.643 | 0.00265 | 0.080 | size=3 medoid |
| 21 | **ch 17** | 0.533 | 0.00040 | 0.027 | size=2 medoid |
| 20 | ch 18 | 0.425 | 0.00200 | 0.045 |  |

## 非 Singleton Cluster 内部 Pairwise Correlation 详表

数值为 **min(median |r| clean, median |r| light)**，即 robust correlation。主阈值为 0.70。

### Cluster 11（size = 2）

|  | ch06 | ch52 |
|---|---|---|
| **ch06** | 1.000 | **0.839** |
| ch52 | — | 1.000 |

- Pair 数：1；min = **0.839**，max = **0.839**，mean = **0.839**。
- 低于 0.70 的 pair：无。
- Average linkage 判断的是簇间平均距离，因此不要求最终簇内每一对都达到 0.70；bootstrap consensus refinement 用于拆分不稳定部分。

### Cluster 12（size = 3）

|  | ch27 | ch28 | ch32 |
|---|---|---|---|
| **ch27** | 1.000 | 0.758 | **0.795** |
| ch28 | — | 1.000 | **0.714** |
| ch32 | — | — | 1.000 |

- Pair 数：3；min = **0.714**，max = **0.795**，mean = **0.756**。
- 低于 0.70 的 pair：无。
- Average linkage 判断的是簇间平均距离，因此不要求最终簇内每一对都达到 0.70；bootstrap consensus refinement 用于拆分不稳定部分。

### Cluster 17（size = 2）

|  | ch43 | ch57 |
|---|---|---|
| **ch43** | 1.000 | **0.724** |
| ch57 | — | 1.000 |

- Pair 数：1；min = **0.724**，max = **0.724**，mean = **0.724**。
- 低于 0.70 的 pair：无。
- Average linkage 判断的是簇间平均距离，因此不要求最终簇内每一对都达到 0.70；bootstrap consensus refinement 用于拆分不稳定部分。

### Cluster 21（size = 2）

|  | ch17 | ch34 |
|---|---|---|
| **ch17** | 1.000 | **0.851** |
| ch34 | — | 1.000 |

- Pair 数：1；min = **0.851**，max = **0.851**，mean = **0.851**。
- 低于 0.70 的 pair：无。
- Average linkage 判断的是簇间平均距离，因此不要求最终簇内每一对都达到 0.70；bootstrap consensus refinement 用于拆分不稳定部分。

### Cluster 23（size = 2）

|  | ch01 | ch10 |
|---|---|---|
| **ch01** | 1.000 | **0.724** |
| ch10 | — | 1.000 |

- Pair 数：1；min = **0.724**，max = **0.724**，mean = **0.724**。
- 低于 0.70 的 pair：无。
- Average linkage 判断的是簇间平均距离，因此不要求最终簇内每一对都达到 0.70；bootstrap consensus refinement 用于拆分不稳定部分。

### Cluster 24（size = 3）

|  | ch11 | ch42 | ch49 |
|---|---|---|---|
| **ch11** | 1.000 | 0.779 | **0.807** |
| ch42 | — | 1.000 | **0.724** |
| ch49 | — | — | 1.000 |

- Pair 数：3；min = **0.724**，max = **0.807**，mean = **0.770**。
- 低于 0.70 的 pair：无。
- Average linkage 判断的是簇间平均距离，因此不要求最终簇内每一对都达到 0.70；bootstrap consensus refinement 用于拆分不稳定部分。

### Cluster 25（size = 6）

|  | ch03 | ch12 | ch29 | ch54 | ch58 | ch62 |
|---|---|---|---|---|---|---|
| ch03 | 1.000 | 0.823 | 0.725 | 0.870 | **0.885** | 0.849 |
| ch12 | — | 1.000 | **0.712** | 0.848 | 0.883 | 0.850 |
| ch29 | — | — | 1.000 | 0.728 | 0.760 | 0.810 |
| ch54 | — | — | — | 1.000 | 0.827 | 0.866 |
| ch58 | — | — | — | — | 1.000 | 0.876 |
| **ch62** | — | — | — | — | — | 1.000 |

- Pair 数：15；min = **0.712**，max = **0.885**，mean = **0.821**。
- 低于 0.70 的 pair：无。
- Average linkage 判断的是簇间平均距离，因此不要求最终簇内每一对都达到 0.70；bootstrap consensus refinement 用于拆分不稳定部分。

### Cluster 27（size = 2）

|  | ch20 | ch44 |
|---|---|---|
| **ch20** | 1.000 | **0.755** |
| ch44 | — | 1.000 |

- Pair 数：1；min = **0.755**，max = **0.755**，mean = **0.755**。
- 低于 0.70 的 pair：无。
- Average linkage 判断的是簇间平均距离，因此不要求最终簇内每一对都达到 0.70；bootstrap consensus refinement 用于拆分不稳定部分。

### Cluster 30（size = 13）

|  | ch16 | ch19 | ch21 | ch22 | ch37 | ch45 | ch46 | ch47 | ch51 | ch53 | ch55 | ch60 | ch61 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ch16 | 1.000 | 0.741 | 0.928 | 0.831 | 0.960 | 0.935 | 0.886 | 0.777 | 0.831 | 0.830 | 0.849 | 0.781 | 0.959 |
| ch19 | — | 1.000 | 0.858 | 0.796 | 0.744 | 0.786 | 0.891 | **0.710** | 0.797 | 0.732 | 0.773 | 0.784 | 0.820 |
| ch21 | — | — | 1.000 | 0.840 | 0.942 | 0.937 | 0.959 | 0.859 | 0.942 | 0.907 | 0.900 | 0.878 | **0.980** |
| ch22 | — | — | — | 1.000 | 0.836 | 0.790 | 0.831 | 0.795 | 0.753 | 0.711 | 0.870 | 0.804 | 0.892 |
| ch37 | — | — | — | — | 1.000 | 0.884 | 0.900 | 0.827 | 0.895 | 0.883 | 0.881 | 0.824 | 0.969 |
| ch45 | — | — | — | — | — | 1.000 | 0.902 | 0.801 | 0.870 | 0.848 | 0.835 | 0.809 | 0.939 |
| ch46 | — | — | — | — | — | — | 1.000 | 0.865 | 0.895 | 0.858 | 0.898 | 0.897 | 0.955 |
| ch47 | — | — | — | — | — | — | — | 1.000 | 0.803 | 0.764 | 0.874 | 0.938 | 0.859 |
| ch51 | — | — | — | — | — | — | — | — | 1.000 | 0.964 | 0.836 | 0.831 | 0.915 |
| ch53 | — | — | — | — | — | — | — | — | — | 1.000 | 0.810 | 0.777 | 0.890 |
| ch55 | — | — | — | — | — | — | — | — | — | — | 1.000 | 0.861 | 0.919 |
| ch60 | — | — | — | — | — | — | — | — | — | — | — | 1.000 | 0.854 |
| **ch61** | — | — | — | — | — | — | — | — | — | — | — | — | 1.000 |

- Pair 数：78；min = **0.710**，max = **0.980**，mean = **0.856**。
- 低于 0.70 的 pair：无。
- Average linkage 判断的是簇间平均距离，因此不要求最终簇内每一对都达到 0.70；bootstrap consensus refinement 用于拆分不稳定部分。

## 需要特别关注的 Channels

### 低 NCC（illumination response 差异大）

| Channel | Cross-light NCC | Gradient Energy | r=0.70 角色 |
|---|---|---|---|
| ch 18 | 0.425 | 0.00200 | representative |
| ch 17 | 0.533 | 0.00040 | representative |
| ch 27 | 0.643 | 0.00265 | representative |
| ch 28 | 0.767 | 0.00578 | cluster member；需 swap-back 才能直接评估 |

NCC 低不等于应当删除。它只表示 clean/light 下响应变化明显；最终价值仍由 MVS ATE 决定。特别是 ch28 在 r=0.70 下不再是 representative，这是降低阈值的主要安全风险之一。

### 高 Gradient Energy（空间结构响应强）

| Channel | Robust Gradient Energy | Cross-light NCC | r=0.70 角色 |
|---|---|---|---|
| ch 15 | 0.05795 | 0.903 | representative |
| ch 10 | 0.05538 | 0.965 | representative |
| ch 63 | 0.04497 | 0.954 | representative |
| ch 23 | 0.04214 | 0.925 | representative |
| ch 60 | 0.04198 | 0.972 | representative |
| ch 30 | 0.04002 | 0.952 | representative |

### Numerically Ineligible Channels

**ch 2, 4, 7, 9, 13, 36, 38, 48**

这些 channel 在 30 帧 post-ReLU feature maps 中空间标准差持续接近零，无法稳定计算 Pearson correlation，因此不参与 Step B。这里称为 numerically ineligible，不额外把 Step A functional label 当作删除依据。

## 与 r=0.80 的直接比较

| 项目 | r=0.80 | r=0.70 | 变化 |
|---|---|---|---|
| Final clusters | 40 | 30 | −10 |
| Representatives | 43 | 36 | −7 |
| Singleton clusters | 36 | 21 | −15 |
| Second representatives | 3 | 6 | +3 |
| Silhouette | 0.119 | 0.224 | 提高 |
| Bootstrap ARI | 0.908 | 0.877 | 略降 |
| Pair retention | 0.868 | 0.891 | 略升 |
| 合法 4-channel combinations | 120,953 | 55,554 | 减少 54.1% |

- r=0.80 representatives 中被 r=0.70 主空间移除：**ch 12, ch 19, ch 22, ch 28, ch 42, ch 44, ch 57, ch 58**。
- r=0.70 新成为 representative：**ch 62**。
- 已知 smoke-test 组合 **[5,29,40,52]** 的四个 channel 在 r=0.70 中仍全部保留，且不违反同簇约束。
- 旧的解释性组合 **[6,28,34,62]** 中 ch28 不再是 representative；需要 cluster swap-back 才能重新评估该原始组合。

## 对后续搜索的影响

- **原始组合数**：C(36,4) = **58,905**。
- **应用同簇不共存约束后的精确组合数**：**55,554**。
- **相对 r=0.80 的 120,953 个合法组合**：减少 **54.1%**。
- **7.5 秒/run 的串行穷举时间**：约 **115.7 小时（4.8 天）**，尚未计入 fail-fast 节省。
- **硬约束**：同一最终 cluster 的 medoid 与 second representative 不得同时进入一个组合。
- **必要补偿**：如果采用 r=0.70 作为主搜索空间，应把 cluster swap-back 设为必做步骤，尤其检查 ch28、ch42、ch44、ch57、ch58 等在 r=0.80 中仍为代表、但在 r=0.70 中被合并的 channel。

## 结论与决策提示

r=0.70 在统计诊断上并非明显失稳：silhouette 提高，bootstrap ARI 仍为 0.877，经过 consensus refinement 后得到 30 个最终簇。但它从“保守去除高度冗余”转向“更积极地合并中等相关 channel”，使 ch28 等具有已知解释价值的 channel 离开主代表空间。

因此，r=0.70 可以作为降低 exhaustive-search 成本的候选阈值或 sensitivity experiment；若作为主阈值，必须结合 second representatives、Top-K cluster swap-back 和完整 MVS ATE，不能仅凭 correlation clustering 宣称被合并成员可完全互换。r=0.80 则更适合作为保守主实验，配合 budgeted multi-start search 控制运行时间。
