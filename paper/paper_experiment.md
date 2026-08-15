# 4. 实验与分析

本节从预测性能、观测泛化、结构必要性、计算效率和理论一致性五个层面评价所提出的模型。主实验分别考察基于已观测历史的一步容量预测，以及不再读取未来观测的递归多步预测；泛化实验进一步将模型置于观测信息更稀疏、健康指标表达能力更弱的数据条件下，以检验整体架构对观测变化的适应能力；随后通过系统消融定位性能与物理一致性的来源，并从模型规模、推理成本和状态空间性质三个方面补充效率与理论证据。

## 4.1 对比方法与统一评测原则

本节所选方法用于回答三个相互关联的问题：本文模型是否超越容量序列本身提供的简单延续规律，显式状态建模是否优于具有较强表达能力的通用时序网络，以及所提出的状态闭环相较已有物理引导方式是否带来额外价值。候选方法依次接受任务相关性、结构代表性和可复现性审查。除无训练基线和具有标准实现的经典网络外，进入数值比较的方法必须具有作者公开实现，并能够在不改变其核心结构的前提下遵循本文的数据划分与信息边界。

### 4.1.1 对比方法的组成与作用

对比方法按照其在证据链中的作用划分为五类。第一类为 Persistence 和局部线性趋势，它们不进行参数学习，用于判断复杂模型是否真正获得了超越容量平滑性和局部斜率的信息。第二类为 GRU、LSTM 及 ModernTCN，代表常用的循环与卷积时序建模方式。第三类为 DLinear、PatchTST、iTransformer 和 TimeMixer，覆盖趋势分解、patch 表示、变量相关性和多尺度混合等近年来具有代表性的通用预测结构。第四类为 PatchFormer、IC2ML、BatteryGPT、TFDM-CR 和 BatteryMFormer，代表针对电池容量、SOH 或退化轨迹设计的专用模型。第五类为 PINN4SOH、PINN-Battery-Prognostics、Hybrid Bayesian PINN 和 HybridPred，用于比较不同物理知识注入方式与本文显式状态建模之间的差异。

上述方法并不共享完全相同的原始任务。为避免将接口改动与方法能力混为一谈，本文将复现分为三个层级：**直接重训**仅接入统一数据加载器并调整预测长度；**任务适配**允许替换输入投影层和输出维度，但保留原模型主体、物理关系、损失函数及训练策略；**协议参照**用于原始任务与本文差异较大的早期全寿命预测或循环内预后方法，此类结果只有在输入时点、预测目标和未来信息边界均可对齐时才参与主排名，否则单独报告。

**表1｜对比方法、任务角色与复现边界。** “一步”和“多步”表示该方法进入相应实验的候选集合，而非直接沿用原论文数值。所有排名结果均应来自本文协议下的本地复现。

| 模型 | 方法类别 | 主要比较作用 | 一步 | 多步 | 复现层级 | 公开实现 |
|---|---|---|:---:|:---:|---|---|
| Persistence / Linear trend | 无训练基线 | 检验容量延续与局部斜率捷径 | ✓ | ✓ | 直接运行 | 本地实现 |
| GRU / LSTM | 循环网络 | 检验经典门控时序编码能力 | ✓ | — | 直接重训 | 标准实现 |
| DLinear [11] | 通用时序模型 | 检验趋势分解与线性预测是否已足够 | ✓ | ✓ | 直接重训 | [GitHub](https://github.com/cure-lab/LTSF-Linear) |
| PatchTST [12] | 通用时序模型 | 比较 patch 化长程依赖建模 | ✓ | ✓ | 直接重训 | [GitHub](https://github.com/yuqinie98/PatchTST) |
| iTransformer [14] | 通用时序模型 | 比较跨变量相关性建模 | ✓ | ✓ | 直接重训 | [GitHub](https://github.com/thuml/iTransformer) |
| TimeMixer [20] | 通用时序模型 | 比较多尺度分解与混合 | ✓ | ✓ | 直接重训 | [GitHub](https://github.com/kwuking/TimeMixer) |
| ModernTCN [21] | 卷积时序模型 | 比较大核卷积时序表示 | ✓ | — | 直接重训 | [GitHub](https://github.com/luodhhh/ModernTCN) |
| PatchFormer [2] | 电池专用模型 | 比较容量恢复感知的双重 patch 建模 | ✓ | — | 任务适配 | [GitHub](https://github.com/USTC-AI4EEE/PatchFormer) |
| PINN4SOH [3] | 物理引导模型 | 比较退化动力学与SOH状态空间约束 | ✓ | — | 任务适配 | [GitHub](https://github.com/wang-fujin/PINN4SOH) |
| PINN-Battery-Prognostics [22] | 物理引导模型 | 比较半经验动力学与多任务物理损失 | ✓ | — | 任务适配 | [GitHub](https://github.com/WenPengfei0823/PINN-Battery-Prognostics) |
| Hybrid Bayesian PINN [23] | 物理–概率模型 | 比较机理方程、模型误差与参数不确定性联合预后 | — | ✓ | 协议参照 | [NASA GitHub](https://github.com/nasa/Li-ion-Battery-Prognosis-Based-on-Hybrid-Bayesian-PINN) |
| HybridPred [24] | 物理–数据混合模型 | 比较物理退化曲线与注意力重建 | — | ✓ | 协议参照 | [GitHub](https://github.com/nathan99sun/HybridPred) |
| IC2ML [25] | 电池健康预测模型 | 比较循环内–循环间联合的SOH与轨迹学习 | ✓ | ✓ | 任务适配 | [GitHub](https://github.com/terencetaothucb/IC2ML-Unified-battery-health-prognostics-via-intra-and-inter-cycle-enhanced-machine-learning) |
| BatteryGPT [16] | 生成式退化预测模型 | 比较早期观测驱动的全寿命轨迹生成 | — | ✓ | 协议参照 | [GitHub](https://github.com/ReparkHjc/BatteryGPT) |
| TFDM-CR [5] | 多步容量预测模型 | 比较时频扩散与容量恢复补偿 | — | ✓ | 任务适配 | [GitHub](https://github.com/sxyyyyyy/TFDM-CR) |
| BatteryMFormer [17] | 退化轨迹预测模型 | 比较多层级退化模式与长跨度预测 | — | ✓ | 协议参照 | [GitHub](https://github.com/Ruifeng-Tan/BatteryMFormer) |
| 本文模型 | 物理状态空间模型 | 观测反演、状态演化与物理容量输出 | ✓ | ✓ | 主模型 | 本文实现 |

### 4.1.2 任务适配边界

通用时序模型仅调整输入通道数、历史长度和预测长度，不改变其主干结构。对于电池专用及物理引导模型，允许的改动限于统一输入接口、目标量纲和输出维度；原模型用于提取退化信息的编码器、动力学方程、物理损失和优化方式均予以保留。任何基线均不得使用本文反演得到的五维有效状态、状态转移形式或物理容量解码器，否则所得结果无法归因于基线自身。

Hybrid Bayesian PINN主要面向循环内放电预后与不确定性传播，HybridPred、BatteryGPT和BatteryMFormer主要从早期观测生成较长或全寿命退化轨迹。它们与本文在线跨循环预测的初始化方式不同，因此只有在使用相同历史窗口、相同目标循环和相同未来信息边界后，才可作为任务适配结果参与排名。无法完成上述对齐时，本文仅将其作为协议参照，不计算相对提升或显著性。

未发现可核验作者实现的 LSTM–GPR [1]、BiGRU-MSTA [15]、PITF [18]、MSTEA-Net [4]、Hybrid Seq2Seq [6]、MAGNet [7] 和 TL-CNN–LSTM–TA [19] 不进入数值主表。这些工作仅用于相关工作中的方法定位；若后续获得作者代码，还需完成输入、目标、预处理和数据划分审计后才能加入比较。

### 4.1.3 公平比较协议

所有可排名方法采用相同的训练、验证和测试电池，相同的目标循环索引、预测起点、历史长度、异常处理与重复实验种子。测试电池仅提供预测时刻之前允许使用的历史观测，不参与特征标准化、超参数选择、早停或参数更新。模型选择和超参数调节仅依据训练与验证电池完成，测试结果不得用于选择配置或随机种子。

考虑到不同方法的原始输入并不完全一致，本文设置两个结果层级。**共同信息结果**仅使用各模型在同一预测时刻均可获得的历史信息，用于主要排名；**原生输入结果**允许方法使用原论文规定的额外信号或工程特征，但必须在表中逐项标注，仅用于展示其完整形态。输入信息不同的结果不进行无条件优劣判断。

一步与多步实验采用不同且明确的信息边界。一步预测在每个目标循环使用此前已完成的真实历史观测，窗口随循环向前移动，但模型参数保持冻结；多步预测仅在起点读取一次真实历史窗口，之后递归回填自身预测，直至达到指定跨度，期间不读取任何真实未来曲线、容量或反演状态。正文和表格分别将二者标记为 **observed-history one-step prediction** 与 **closed-loop recursive rollout**，避免将连续单步评测误写为多步预测。

每个复现模型均保存代码来源、版本或提交号、环境文件、超参数搜索范围和适配补丁。论文中的原文报告值与本文同协议复现值分开呈现；只有后者进入统一排名、统计检验和相对提升计算。

## 4.2 主实验

### 4.2.1 跨电池一步容量预测

一步主实验在 CALCE 和另一套待确定的完整运行曲线数据集上进行。对于每个测试电池，模型以最近64个已观测循环为输入，预测下一循环的绝对容量。实验采用留一电池协议，并报告每块电池及跨电池汇总的 MAE、RMSE 和 $R^2$。考虑到容量轨迹在相邻循环间通常较平滑，Persistence 必须与所有学习模型同时报告；只有在相同目标循环集合上稳定优于 Persistence，才能将较低的一步误差归因于模型学习到了超越最后容量值的退化信息。

一步主表设置12个对比模型。DLinear、PatchTST、iTransformer、TimeMixer和ModernTCN构成具有公开实现的通用时序基线；PatchFormer与IC2ML代表电池专用预测模型；PINN4SOH和PINN-Battery-Prognostics代表通过退化方程或物理损失约束学习过程的方法。所有模型均接收相同的64循环历史并预测下一循环绝对容量，其中任务适配模型必须同时报告改动清单。

**表2｜统一跨电池协议下的一步容量预测。** 所有学习模型报告多次独立运行的 mean ± std；$\Delta_{\mathrm{Per}}<0$ 表示相较 Persistence 降低了 MAE。年份用于显式核验近三年方法的覆盖比例。

| 模型 | 年份 | 类型 | CALCE MAE ↓ | 第二主数据集 MAE ↓ | 平均 RMSE ↓ | 平均 $R^2$ ↑ | $\Delta_{\mathrm{Per}}$ (%) ↓ |
|---|---:|---|---:|---:|---:|---:|---:|
| Persistence | — | 无学习 | [待填] | [待填] | [待填] | [待填] | 0.00 |
| GRU | 2014 | 循环网络 | [待填] | [待填] | [待填] | [待填] | [待填] |
| LSTM | 1997 | 循环网络 | [待填] | [待填] | [待填] | [待填] | [待填] |
| DLinear [11] | 2023 | 长序列预测 | [待填] | [待填] | [待填] | [待填] | [待填] |
| PatchTST [12] | 2023 | 长序列预测 | [待填] | [待填] | [待填] | [待填] | [待填] |
| iTransformer [14] | 2024 | 多变量时序 | [待填] | [待填] | [待填] | [待填] | [待填] |
| TimeMixer [20] | 2024 | 多尺度时序 | [待填] | [待填] | [待填] | [待填] | [待填] |
| ModernTCN [21] | 2024 | 卷积时序 | [待填] | [待填] | [待填] | [待填] | [待填] |
| PINN4SOH [3] | 2024 | 物理引导 | [待填] | [待填] | [待填] | [待填] | [待填] |
| PINN-Battery-Prognostics [22] | 2023/2024 | 物理引导 | [待填] | [待填] | [待填] | [待填] | [待填] |
| PatchFormer [2] | 2025 | 电池专用Transformer | [待填] | [待填] | [待填] | [待填] | [待填] |
| IC2ML [25] | 2026 | 循环内–循环间联合学习 | [待填] | [待填] | [待填] | [待填] | [待填] |
| 本文模型 | — | 状态空间闭环 | [待填] | [待填] | [待填] | [待填] | [待填] |

主表之外，分别绘制真实与预测容量轨迹、逐循环绝对误差以及早期、中期和晚期分段误差。正文先报告跨电池总体结果，再分析最困难电池及误差峰值所在的退化阶段，避免以单个最优电池或最优随机种子代替总体证据。

### 4.2.2 无未来观测的递归多步预测

多步主实验用于检验状态转移在脱离真实未来观测后的稳定性。从预先指定的循环起点取得长度为64的真实历史状态，模型预测下一状态并通过容量解码器得到容量；随后将预测状态回填至历史窗口，重复执行同一过程直至达到目标预测步长。在整个 rollout 中，模型权重保持冻结，且不重新辨识任何真实未来状态。

多步实验至少报告 $h\in\{4,8,16,32\}$ 的误差，并从早期、中期和晚期分别选择多个预注册起点。该表同样设置12个对比模型：DLinear、PatchTST、iTransformer和TimeMixer提供通用多步预测参照；Hybrid Bayesian PINN和HybridPred提供可运行的物理混合参照；BatteryGPT、TFDM-CR、BatteryMFormer和IC2ML覆盖近期电池轨迹预测方法。BatteryGPT和BatteryMFormer的原始任务偏向早期全寿命预测，只有在统一64循环初始化窗口和未来信息边界后才参与主排名，否则单列为“不同任务协议结果”。除平均误差外，还应报告误差随 horizon 的增长率、最差起点以及预测轨迹发生明显漂移的比例。

**表3｜固定模型参数下的递归多步容量预测。** 标记“†”的方法属于协议参照，只有在历史窗口、预测目标和未来信息边界完成对齐后才参与统一排名。

| 模型 | 年份 | 类型 | MAE@$h=4$ ↓ | MAE@$h=8$ ↓ | MAE@$h=16$ ↓ | MAE@$h=32$ ↓ | 最差起点 MAE ↓ |
|---|---:|---|---:|---:|---:|---:|---:|
| Persistence | — | 无学习 | [待填] | [待填] | [待填] | [待填] | [待填] |
| Linear trend | — | 统计外推 | [待填] | [待填] | [待填] | [待填] | [待填] |
| DLinear [11] | 2023 | 长序列预测 | [待填] | [待填] | [待填] | [待填] | [待填] |
| PatchTST [12] | 2023 | 长序列预测 | [待填] | [待填] | [待填] | [待填] | [待填] |
| iTransformer [14] | 2024 | 多变量时序 | [待填] | [待填] | [待填] | [待填] | [待填] |
| TimeMixer [20] | 2024 | 多尺度时序 | [待填] | [待填] | [待填] | [待填] | [待填] |
| Hybrid Bayesian PINN [23]† | 2023 | 物理–概率混合 | [待填] | [待填] | [待填] | [待填] | [待填] |
| HybridPred [24]† | 2025 | 物理–数据混合轨迹 | [待填] | [待填] | [待填] | [待填] | [待填] |
| BatteryGPT [16]† | 2025/2026 | 生成式全寿命轨迹 | [待填] | [待填] | [待填] | [待填] | [待填] |
| TFDM-CR [5] | 2026 | 扩散式容量预测 | [待填] | [待填] | [待填] | [待填] | [待填] |
| BatteryMFormer [17]† | 2026 | 多层级轨迹预测 | [待填] | [待填] | [待填] | [待填] | [待填] |
| IC2ML [25] | 2026 | 联合健康轨迹预测 | [待填] | [待填] | [待填] | [待填] | [待填] |
| 本文模型 | — | 递归状态空间闭环 | [待填] | [待填] | [待填] | [待填] | [待填] |

## 4.3 弱观测条件下的架构泛化

泛化实验不再使用主实验中的完整运行曲线，而选择仅提供循环级宏观健康指标的数据集，例如 TJU。当前 TJU 数据可获得恒流充电时间（CCCT）、恒压充电时间（CVCT）和恒流放电时间（CCDT），但不提供与 CALCE 相同分辨率的循环内放电曲线和逐循环阻抗观测。与完整曲线相比，这些汇总量对局部电压形状、截止点变化和状态间耦合的表达更弱，因此更依赖模型从有限观测中构造稳定状态并学习跨循环演化。

该实验验证的是“架构对观测条件变化的适应能力”，而不是同一组模型权重在不同数据集之间的零样本迁移。为保持论证准确，两个数据集共享观测编码—状态转移—容量输出的功能分解、相同的因果窗口和跨电池评测原则，但观测编码器依据数据集实际提供的指标进行实例化。若输入维度和观察方程发生变化，则必须重新训练相应编码器；除非额外实施源域训练、未知目标域测试，否则不得将其表述为跨数据集 domain generalization。

TJU 实验采用电池级留一测试，并以完全相同的 CCCT、CVCT 和 CCDT 输入重新训练 MLP、GRU、LSTM、Transformer、PINN4SOH、PINN-Battery-Prognostics 和本文模型。两个物理基线均从公开实现出发，只允许调整输入接口和绝对容量输出头，并保留原有退化约束；表中必须标记为任务适配复现，而不能直接沿用原论文指标。这样，泛化实验比较的是不同架构从同一弱观测集合提取退化信息的能力，而不是输入信息量的差异。

**表4｜弱宏观指标条件下的跨电池预测。**

| 数据集 | 模型 | 输入指标 | MAE ↓ | RMSE ↓ | $R^2$ ↑ | 相对 Persistence 改善 (%) ↑ |
|---|---|---|---:|---:|---:|---:|
| TJU | Persistence | 上一循环容量 | [待填] | [待填] | [待填] | 0.00 |
| TJU | MLP | CCCT、CVCT、CCDT | [待填] | [待填] | [待填] | [待填] |
| TJU | GRU | CCCT、CVCT、CCDT | [待填] | [待填] | [待填] | [待填] |
| TJU | LSTM | CCCT、CVCT、CCDT | [待填] | [待填] | [待填] | [待填] |
| TJU | Transformer | CCCT、CVCT、CCDT | [待填] | [待填] | [待填] | [待填] |
| TJU | PINN4SOH（适配） | CCCT、CVCT、CCDT | [待填] | [待填] | [待填] | [待填] |
| TJU | PINN-Battery-Prognostics（适配） | CCCT、CVCT、CCDT | [待填] | [待填] | [待填] | [待填] |
| TJU | 本文模型 | CCCT、CVCT、CCDT | [待填] | [待填] | [待填] | [待填] |

为证明结果来自架构而非容量历史捷径，泛化实验还应增加三项控制：其一，移除上一循环容量，只保留宏观指标；其二，仅使用上一循环容量；其三，使用宏观指标与容量的完整输入。三者的差值能够量化弱指标提供的增量信息，并检验状态表示是否真正吸收了 CCCT、CVCT 和 CCDT 中的退化信号。

## 4.4 消融实验

消融实验围绕完整预测链条展开，而不是只删除神经网络模块。所有变体保持相同的数据划分、随机种子、训练预算、早停准则和测试起点，并至少重复五次；用于支撑主要贡献的变体应与主实验保持相同的重复次数。表5为主文大表，其中每一行只改变一个因素；损失权重扫描、状态维数穷举和更多窗口长度可移至补充材料。

**表5｜观测辨识、时序表示、状态转移、容量解码与训练目标的系统消融。** $\mathrm{MAE}_{1}$ 为一步容量误差，$\mathrm{MAE}_{32}$ 为32步递归误差，$\mathrm{MAE}_{\mathrm{gen}}$ 为弱指标泛化误差；“越界率”统计预测状态超出预设可行域的比例。

| ID | 变体 | 被检验的设计 | $\mathrm{MAE}_{1}$ ↓ | $\mathrm{MAE}_{32}$ ↓ | $\mathrm{MAE}_{\mathrm{gen}}$ ↓ | 曲线 RMSE ↓ | 闭合 MAE ↓ | 状态越界率 ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Full | 完整模型 | 全部设计 | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] |
| A1 | 去除 $dV/dQ$ 形状残差 | 微分曲线是否约束状态形状 | [待填] | [待填] | — | [待填] | [待填] | [待填] |
| A2 | 去除截止点强调 | 截止附近信息是否决定容量闭合 | [待填] | [待填] | — | [待填] | [待填] | [待填] |
| A3 | 去除内阻弱观测 | 阻抗状态是否失稳 | [待填] | [待填] | — | [待填] | [待填] | [待填] |
| A4 | 去除相邻循环弱连续性 | 连续性是否抑制状态跳变 | [待填] | [待填] | — | [待填] | [待填] | [待填] |
| A5 | 单初值替代多初值 | 反演是否受局部最优影响 | [待填] | [待填] | — | [待填] | [待填] | [待填] |
| A6 | L2替代鲁棒损失 | 异常曲线对反演的影响 | [待填] | [待填] | — | [待填] | [待填] | [待填] |
| A7 | 四维状态 | 五维状态是否具有必要信息 | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] |
| A8 | 增加无直接观测支撑的状态 | 额外自由度是否形成不可辨识捷径 | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] |
| B1 | 仅使用 $Z_k$ | 当前状态是否已足够 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| B2 | $Z+\Delta Z$ | 单步变化的增量价值 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| B3 | 去除短期趋势 | 局部变化是否被利用 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| B4 | 去除长期趋势 | 稳定退化方向是否被利用 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| B5 | 历史长度16 | 长历史是否必要 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| B6 | 历史长度32 | 长历史是否必要 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| B7 | 历史长度128 | 更长窗口是否继续受益 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| C1 | 无界加性状态更新 | 正值和变化界约束的作用 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| C2 | 有界加性状态更新 | 乘性更新的作用 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| C3 | 直接预测下一绝对状态 | 相对更新锚点的作用 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| C4 | 逐循环单调约束 | 双向局部更新是否必要 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| C5 | 状态头非零初始化 | 从状态保持开始训练的作用 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| C6 | 放宽/收紧 $\boldsymbol\alpha$ | 单步变化上限敏感性 | [待填] | [待填] | [待填] | — | [待填] | [待填] |
| D1 | 学习型绝对容量头 | 显式物理解码是否必要 | [待填] | [待填] | [待填] | 不适用 | 不适用 | [待填] |
| D2 | $Q_k+\Delta\widehat Q_{k+1}$ | 容量残差捷径的影响 | [待填] | [待填] | [待填] | 不适用 | 不适用 | [待填] |
| D3 | 状态＋自由容量残差校准 | 精度与闭合之间的权衡 | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] |
| E1 | 去除状态损失 | 状态监督的作用 | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] |
| E2 | 去除观测/曲线损失 | 观测一致性监督的作用 | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] |
| E3 | 去除容量损失 | 容量监督是否不可替代 | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] |
| E4 | 去除动力学正则 | 状态可行性约束的作用 | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] |

消融结论按照证据强度进行约束。某一变体只有在目标指标上产生稳定且可重复的退化，才支持相应模块的必要性；若物理解码与学习型容量头精度相当但闭合误差显著更低，则结论应写为“在保持预测精度的同时增加状态–容量可审计性”；若物理解码精度更低但多步更稳定，则应报告为精度与结构约束之间的权衡，而不能笼统称为性能提升。

## 4.5 效率分析

效率分析主要与数据驱动模型比较，用于验证摘要中关于模型紧凑性的表述。比较对象至少包括 GRU、LSTM、TCN、Transformer 和 PatchFormer，并在相同硬件、相同批量大小、相同输入长度和相同计时策略下统计可训练参数量、单样本推理时间、峰值显存和训练时间。由于本文模型还包含状态辨识和容量求解，必须将离线状态缓存生成时间、神经状态转移时间和容量解码时间分别报告，避免用较小的 GRU 参数量掩盖端到端物理计算成本。

**表6｜统一硬件条件下的模型规模与计算开销。**

| 模型 | 可训练参数 | 训练时间/fold | 在线编码时间/循环 | 状态转移时间/循环 | 容量解码时间/循环 | 端到端推理时间/循环 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRU | [待填] | [待填] | [待填] | 不适用 | 不适用 | [待填] | [待填] |
| LSTM | [待填] | [待填] | [待填] | 不适用 | 不适用 | [待填] | [待填] |
| TCN | [待填] | [待填] | [待填] | 不适用 | 不适用 | [待填] | [待填] |
| Transformer | [待填] | [待填] | [待填] | 不适用 | 不适用 | [待填] | [待填] |
| PatchFormer | [待填] | [待填] | [待填] | 不适用 | 不适用 | [待填] | [待填] |
| 本文模型（在线部分） | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] |
| 本文模型（含离线反演摊销） | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] | [待填] |

只有当端到端时间和资源占用同样具有优势时，才将模型描述为“高效”；如果优势仅体现为可训练参数较少，应将摘要和结论限定为“紧凑的时序状态转移器”。推理计时应排除首次 CUDA 初始化，设置预热轮次，并报告至少1000次重复的中位数和四分位范围。

## 4.6 理论与物理一致性分析

本节不将数值可视化包装为严格的理论证明，而是从解析性质、观测闭合、状态敏感性和局部动力学四个层面验证方法部分提出的结构性说法。分析目标是回答：预测状态是否始终位于可行域内，状态是否能够重建观测并生成容量，各状态维度是否真正影响容量输出，以及允许局部双向变化是否具有观测依据。

### 4.6.1 状态转移的解析性质

本文状态转移为

$$
\widehat Z_{k+1}
=
Z_k\odot
\exp\!\left(
\boldsymbol\alpha\odot\tanh r_k
\right).
$$

当 $Z_k>0$ 且 $\boldsymbol\alpha>0$ 时，指数映射直接给出

$$
\widehat Z_{k+1}>0,
$$

并且对第 $i$ 个状态维度有

$$
\exp(-\alpha_i)
\leq
\frac{\widehat z_{i,k+1}}{z_{i,k}}
\leq
\exp(\alpha_i).
$$

因此，该更新在不施加逐循环单调方向的条件下，同时保证状态正值和单步相对变化有界。递归 $h$ 步后还可得到最坏情况下的包络

$$
z_{i,k}\exp(-h\alpha_i)
\leq
\widehat z_{i,k+h}
\leq
z_{i,k}\exp(h\alpha_i),
$$

它说明 $\alpha_i$ 直接控制长期 rollout 的最大漂移速度。理论分析将上述界与实际预测中的单步状态变化分布、越界率和多步误差联合报告；解析有界并不等价于误差不累积，因此不能据此宣称模型具有无条件长期稳定性。

模型的因果性通过输入集合直接验证：$\widehat Z_{k+1}$ 仅依赖 $\mathcal O_{1:k}$ 及训练阶段冻结的参数，目标循环观测 $\mathcal O_{k+1}$ 不进入前向传播。代码级因果审计应与窗口索引测试共同提供，以排除标准化、移动平均和状态缓存中的未来信息泄漏。

### 4.6.2 观测重建与状态–容量闭合

对于由循环观测辨识得到的状态 $Z_k$，首先使用同一观察算子重建运行响应：

$$
\widehat{\mathcal O}_k
=
\mathcal G(Z_k;\boldsymbol\pi_k).
$$

在完整放电曲线数据上，分别报告 $V(Q)$ 的 MAE/RMSE、$dV/dQ$ 误差和截止点局部误差，并覆盖早期、中期、晚期及全部循环分布。随后由容量解码器计算

$$
\widehat Q_k^{\mathrm{closure}}
=
\mathcal D(Z_k;\boldsymbol\pi_k),
\qquad
\epsilon_k^{\mathrm{closure}}
=
\left|
\widehat Q_k^{\mathrm{closure}}-Q_k
\right|.
$$

曲线重建用于验证状态是否保留当前观测中的形状信息，容量闭合用于验证状态、观察方程和容量输出是否位于同一计算链条。二者共同支持“模型内部具有观测–状态–容量一致性”，但不自动证明五个状态参数等于材料级真实退化量，也不证明参数具有唯一解。

**表7｜理论说法、验证量与允许形成的结论。**

| 待验证说法 | 解析或实验量 | 必需对照 | 允许形成的结论 | 不允许形成的结论 |
|---|---|---|---|---|
| 状态保持正值 | $\widehat Z_{k+1}>0$；越界率 | 无界加性更新 | 更新映射保持正值 | 所有预测均物理真实 |
| 单步变化有界 | 状态比值范围；$\alpha$敏感性 | 放宽/取消变化界 | 更新幅度受显式控制 | 长期 rollout 无误差累积 |
| 状态保留观测信息 | $V(Q)$、$dV/dQ$、截止点误差 | 去除各反演残差 | 状态可重建当前观察 | 状态参数被唯一辨识 |
| 状态直接生成容量 | closure MAE；解码消融 | 学习型容量头、残差头 | 状态与容量输出形成闭合 | 闭合必然提高预测精度 |
| 状态维度影响容量 | 无量纲局部灵敏度 | 状态扰动与维度删除 | 各状态对输出具有可计算作用 | 灵敏度等同于因果老化机理 |
| 双向更新适应局部恢复 | 恢复片段误差、方向命中率 | 单调约束更新 | 局部双向变化符合观测 | 老化机理在物理上可逆 |

### 4.6.3 状态–容量敏感性与可辨识性

容量解码器对第 $i$ 个状态的局部灵敏度定义为

$$
S_i(Z)
=
\frac{\partial \mathcal D(Z)}{\partial z_i},
\qquad
\widetilde S_i(Z)
=
\frac{z_i}{\mathcal D(Z)}
\frac{\partial \mathcal D(Z)}{\partial z_i}.
$$

其中 $\widetilde S_i$ 为无量纲灵敏度，用于比较不同量纲状态对容量输出的相对影响。自动微分结果使用中心有限差分复核，并分别统计早期、中期和晚期的灵敏度方向、幅值及电池间差异。若某一状态的灵敏度长期接近零，则其对容量输出的必要性应通过状态维数消融重新审查；若多个状态具有高度共线的响应，则应将其解释限制为组合有效状态。

可辨识性分析由观察算子对状态的 Jacobian

$$
J_k
=
\frac{\partial \mathcal G(Z_k;\boldsymbol\pi_k)}
{\partial Z_k}
$$

出发，计算奇异值谱以及近似 Fisher 信息矩阵 $F_k=J_k^\top WJ_k$ 的条件数。已有研究表明，单一恒流放电电压数据对参数的可辨识信息有限，而 OCV 不同区间对退化模式的敏感性显著不同[8–10]。因此，本文进一步执行多初值反演、观测噪声扰动、OCP 先验扰动和参数 bootstrap/profile likelihood，并比较“相似曲线、不同状态”解出现的频率。分析结果用于界定哪些维度可以稳定辨识，哪些维度只能作为有效代理状态，而不是为所有状态预设唯一物理解。

### 4.6.4 局部恢复、状态轨迹与失败案例

局部动力学分析采用预先定义的片段选择规则，包括 $\Delta Q_k>0$ 的容量恢复、长平台、突降后回归以及反演状态发生显著回调的区间。对所有满足条件的片段统一比较有界双向更新、逐循环单调更新和无约束更新，并分别报告恢复片段与非恢复片段 MAE、状态方向命中率、32步递归误差和长期趋势相关性。只有当双向更新在局部恢复片段取得优势且不显著损害总体退化趋势时，才能认为该设计更符合观测；“reversible”仅指有效状态允许局部回调，不表示材料老化机理可逆。

失败案例按照统一规则选取：最大绝对误差、P95/P99误差、闭合误差最高循环、状态边界命中、截止阈值附近平台和多步漂移最快起点。每个案例同时展示输入观测、状态轨迹、容量预测和对比模型结果。该分析用于明确模型在观测不足、先验失配和长跨度递推条件下的适用边界。

## 参考文献

1. Liu, K. et al. A data-driven approach with uncertainty quantification for predicting future capacities and remaining useful life of lithium-ion battery. *IEEE Transactions on Industrial Electronics* **68**, 3170–3180 (2021). https://doi.org/10.1109/TIE.2020.2973876
2. Liu, L. et al. PatchFormer: A novel patch-based transformer for accurate remaining useful life prediction of lithium-ion batteries. *Journal of Power Sources* **631**, 236187 (2025). https://doi.org/10.1016/j.jpowsour.2025.236187
3. Wang, F. et al. Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis. *Nature Communications* **15**, 4332 (2024). https://doi.org/10.1038/s41467-024-48779-z
4. Cheng, H. & Zhang, L. A physics-informed deep learning framework for remaining useful life prediction of lithium-ion batteries with feature subset construction. *Energy* **346**, 140288 (2026). https://doi.org/10.1016/j.energy.2026.140288
5. TFDM-CR: Time–frequency diffusion modeling for lithium-ion battery capacity prediction incorporating regeneration phenomena. *Energy and AI* **24**, 100703 (2026). https://doi.org/10.1016/j.egyai.2026.100703
6. Xu, L., Deng, Z., Xie, Y., Lin, X. & Hu, X. A novel hybrid physics-based and data-driven approach for degradation trajectory prediction in Li-ion batteries. *IEEE Transactions on Transportation Electrification* **9**, 2628–2644 (2023). https://doi.org/10.1109/TTE.2022.3212024
7. Tan, R. et al. Forecasting battery degradation trajectory under domain shift with domain generalization. *Energy Storage Materials* **72**, 103725 (2024). https://doi.org/10.1016/j.ensm.2024.103725
8. Schmitt, J. et al. Identifiability study of lithium-ion battery capacity fade using degradation mode sensitivity for a minimally and intuitively parametrized electrode-specific cell open-circuit voltage model. *Journal of Power Sources* **605**, 234446 (2024). https://doi.org/10.1016/j.jpowsour.2024.234446
9. Choi, Y. et al. Parameter identification and identifiability analysis of lithium-ion batteries. *Energy Science & Engineering* **10**, 798–814 (2022). https://doi.org/10.1002/ese3.1039
10. Forman, J. C. et al. A computational framework for identifiability and ill-conditioning analysis of lithium-ion battery models. *Industrial & Engineering Chemistry Research* **54**, 12023–12033 (2015). https://doi.org/10.1021/acs.iecr.5b03910
11. Zeng, A., Chen, M., Zhang, L. & Xu, Q. Are Transformers effective for time series forecasting? *Proceedings of the AAAI Conference on Artificial Intelligence* **37**, 11121–11128 (2023). https://doi.org/10.1609/aaai.v37i9.26317
12. Nie, Y. et al. A time series is worth 64 words: Long-term forecasting with Transformers. *International Conference on Learning Representations* (2023). https://openreview.net/forum?id=Jbdc0vTOcol
13. Wu, H. et al. TimesNet: Temporal 2D-variation modeling for general time series analysis. *International Conference on Learning Representations* (2023). https://openreview.net/forum?id=ju_Uqw384Oq
14. Liu, Y. et al. iTransformer: Inverted Transformers are effective for time series forecasting. *International Conference on Learning Representations* (2024). https://openreview.net/forum?id=JePfAI8fah
15. Wang, L. & Wang, S. The application of BiGRU-MSTA based on multi-scale temporal attention mechanism in predicting the remaining life of lithium-ion batteries. *Batteries* **11**, 223 (2025). https://doi.org/10.3390/batteries11060223
16. Hu, J. et al. Early prediction of lithium-ion battery degradation with a generative pre-trained transformer. *Nature Communications* **17**, 126 (2026; online 2025). https://doi.org/10.1038/s41467-025-66819-0
17. Tan, R. et al. BatteryMFormer: Multi-level learning for battery degradation trajectory forecasting. *arXiv* (2026). https://arxiv.org/abs/2605.27044
18. Yang, T., Hu, Y., Ma, X., Cheng, X. & Zhu, Q. Physics-Informed Temporal Former with empirical degradation prior for state-of-health estimation of lithium-ion batteries. *Energy & Fuels* **40**, 10615–10638 (2026). https://doi.org/10.1021/acs.energyfuels.6c00543
19. Yang, X. et al. Early-stage degradation trajectory prediction for lithium-ion batteries: A generalized method across diverse operational conditions. *Journal of Power Sources* **612**, 234808 (2024). https://doi.org/10.1016/j.jpowsour.2024.234808
20. Wang, S. et al. TimeMixer: Decomposable multiscale mixing for time series forecasting. *International Conference on Learning Representations* (2024). https://openreview.net/forum?id=7oLshfEIC2
21. Luo, D. & Wang, X. ModernTCN: A modern pure convolution structure for general time series analysis. *International Conference on Learning Representations* (2024). https://openreview.net/forum?id=vpJMJerXHU
22. Wen, P. et al. Physics-informed neural networks for prognostics and health management of lithium-ion batteries. *IEEE Transactions on Intelligent Vehicles* **9**, 2276–2289 (2024; online 2023). https://doi.org/10.1109/TIV.2023.3315548
23. Nascimento, R. G. et al. A framework for Li-ion battery prognosis based on hybrid Bayesian physics-informed neural networks. *Scientific Reports* **13**, 13856 (2023). https://doi.org/10.1038/s41598-023-33018-0
24. Nicolae, C.-D., Sameer, S., Sun, N. & Yan, K. Optimizing cycle life prediction of lithium-ion batteries via a physics-informed model. *Transactions on Machine Learning Research* (2025). https://openreview.net/forum?id=1weZ9Wsajk
25. Huang, T. et al. IC2ML: Unified battery health prognostics via intra- and inter-cycle enhanced machine learning. *Journal of Power Sources* **666**, 239148 (2026). https://doi.org/10.1016/j.jpowsour.2025.239148

## 尚待实验前冻结的事项

1. 第二个主实验数据集必须提供与 CALCE 相近的循环内运行曲线，且至少包含足够电池支持跨电池隔离；否则它应被归入弱观测泛化实验，而不能与 CALCE共同支撑完整物理闭合。
2. 逐一运行表1链接的官方代码，核对依赖、许可证、输入、目标和预处理；为 PINN4SOH、PINN-Battery-Prognostics、Hybrid Bayesian PINN 与 HybridPred 单独保存适配补丁及改动清单。任何无法在冻结环境中完成训练—推理闭环的方法均自动退出主排名，由同类别可运行模型替代。
3. 冻结单步和多步的预测起点、horizon、历史长度和随机种子，并为所有模型生成同一目标循环索引文件。
4. 核对当前曲线监督是否产生非零梯度；若该项未实际参与优化，则不得在消融和贡献中将其写成已验证设计。
5. 理论部分统一使用“解析性质”“物理一致性”和“局部可辨识性”，不使用“证明模型具有普适泛化能力”或“唯一辨识真实退化参数”等超出证据范围的表述。
