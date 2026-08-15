# BATTER-MoE 原文—源码复现审计

> 审计对象：`BATTER-MoE: A Sparse Mixture-of-Experts Model for Accurate and Efficient Remaining Useful Life Prediction of Lithium-Ion Batteries`，DOI: 10.1109/TTE.2026.3697742。  
> 本文档只把原文明示内容判定为“确定”；图示推断、常见默认值和反向拟合实验均单独标记。

## 1. 结论摘要

当前实现的主体计算图与论文公式高度一致，不能将性能差距简单归因于某一个模块“写错”。多尺度非重叠切片、中心时间戳、CS-SE、Pre-RMSNorm、RoPE、稀疏与共享专家、负载均衡损失以及最新物理时间戳池化均能逐式对应。CT-Reweighting 虽然在论文中没有内部公式，但图 1 的结构与 Efficient Multi-Scale Attention（EMA）的公开实现几乎逐层一致；当前代码也基本遵循这一拓扑。

剩余差距主要来自论文不可复现信息：注意力头数、CS-SE 压缩率、CT 分组数、尺度嵌入是否启用、CT 标量门初始化、Dropout 的具体位置、80/20 的抽样方式、异常点检测伪代码、最大 epoch 和十次实验的 seed 均未披露。当前实现过去把这些假设固化为一套默认值，因此只能称为“论文约束下的重建实现”，不能称为作者源码的完全复现。

## 2. 术语与符号

| 规范名称 | 含义 | 代码对象 |
|---|---|---|
| time-aware multi-scale tokenization | 时间感知多尺度标记化 | `MultiScaleTokenizer` |
| CS-SE | Cross-Scale Squeeze-Excitation attention | `CrossScaleSE` |
| CT-Reweighting | Cross-Time Reweighting | `CrossTimeReweighting` |
| routed expert | Top-k 选中的专门化专家 | `experts` |
| shared expert | 所有 token 均经过的共享专家 | `shared_expert` |
| physical timestamp | patch 中心循环位置 | `positions` |
| normalized capacity | 额定容量归一化值 `C/C0` | `capacity / rated_capacity` |

## 3. 原文证据索引

| 原文位置 | 可确定信息 |
|---|---|
| p.3, Fig. 1, Eq. (1)–(2) | 总体数据流、非重叠多尺度 patch、独立线性投影、可选尺度嵌入、中心时间戳 |
| p.4, Eq. (3)–(4) | 跨尺度均值、求和、共享两层 MLP、Sigmoid 通道回写 |
| p.4, Fig. 1, Eq. (5) | 分组双分支 CT、1×1 与 3×3 卷积、Softmax–MatMul、`softplus(theta)` 标量门 |
| p.4, Eq. (6) | Pre-RMSNorm、RoPE 自注意力残差、MoE 残差、`epsilon=1e-6` |
| p.4, Eq. (7)–(8) | RMSNorm 后 token 级 Softmax Top-k、路由专家宽度 `dff/k`、全宽共享专家、独立 Sigmoid 门 |
| p.4, Eq. (9)–(10) | 最新有效物理时间戳池化、MAE 主损失和路由概率均衡损失 |
| p.5–6, Sec. IV-A/B, Table II | 3σ 孤立异常、线性插值、`C/C0`、ground-truth rolling one-step、完整留出 B0005 |
| p.6, Table III | NASA 的 `P={2,4,8}`、`L=16`、`dmodel=64`、1层、`dff=128`、4专家、Top-2、dropout 0.05、Adam 1e-3、batch 128、patience 10 |
| p.7, Table IV | NASA SP50/SP90；论文报告 Ours MAE 0.0045/0.0046，PatchFormer 0.0073/0.0065 |
| p.8, Table VI | 完整模型及 CS-SE、CT、MoE、patch、RoPE、pooling 消融 |

## 4. 模型逐项核对

| 环节 | 原文要求 | 当前实现 | 判定 |
|---|---|---|---|
| patch 划分 | 每尺度 stride 等于 patch 长度，`Ni=floor(L/pi)` | 完全一致；论文配置均可整除 | 一致 |
| patch 投影 | `vec(S)Wi+bi` | 带 bias 的独立 `Linear` | 一致 |
| 时间戳 | `n*pi+floor((pi-1)/2)` | 完全一致 | 一致 |
| patch mask | patch 内输入 mask 的最小值 | `all(-1)` | 一致 |
| 尺度嵌入 | 原文仅说“if enabled” | 默认启用，正态初始化 `std=0.02` | **开关和初始化未披露** |
| CS-SE context | 各尺度 token 均值后跨尺度求和 | 完全一致 | 一致 |
| CS-SE MLP | `H -> H/r -> K*H`，ReLU、Sigmoid | 完全一致 | `r` 未披露 |
| CT 分组 | 隐藏通道分成 G 组 | 按隐藏维分组 | G 未披露 |
| CT 1×1 分支 | 时间—通道上下文门控 | 与 Fig. 1/EMA 拓扑一致 | 高可信重建 |
| CT 3×3 分支 | 捕获局部时间模式 | `Conv2d(...,3,padding=1)` | 高可信重建 |
| CT 双路融合 | AvgPool、Softmax、MatMul、相加、Sigmoid | 完全覆盖图示操作 | 高可信重建 |
| CT 标量门 | `gamma=softplus(theta)` | 公式一致 | **theta 初始化未披露** |
| Pre-RMSNorm | 两个子层均为前归一化，`epsilon=1e-6` | 完全一致 | 一致 |
| RoPE | 在 MHSA 中编码物理时间位置 | Q/K 旋转，使用 patch 中心位置 | 一致；head 数和 base 未披露 |
| routed experts | token 级 Softmax Top-k，不二次归一化 | 完全一致 | 一致 |
| expert width | 每个路由专家为 `dff/k` | 完全一致 | 一致 |
| shared expert | 全宽，独立 Sigmoid 门 | 完全一致 | 一致 |
| gated FFN | Up/Gate、SiLU、Hadamard、Down | 完全一致 | bias/dropout 位置未披露 |
| auxiliary loss | 各层平均路由概率对均匀分布的平方偏差 | 完全一致 | 一致 |
| pooling | 只平均最大有效物理时间戳处的 token | 完全一致 | 一致 |
| output | 单线性层预测下一循环容量 | 完全一致 | 一致 |

## 5. 论文自身无法消解的歧义

1. 公式 (2) 把每个 patch 展平为一个 `H` 维 token，尺度序列形状为 `Ni × H`；Fig. 1 却标为 `(B,C,Ti,D)`，额外保留了 `C` 维。当前实现优先遵循明确公式。
2. 正文和公式 (6) 明确使用 Pre-RMSNorm；Fig. 1 中央框的“Add + RMSNorm”视觉上更接近后归一化。当前实现优先遵循公式。
3. Fig. 1 的 CT 模块几乎复用了 EMA attention 的图形拓扑，但论文没有引用 EMA，也没有给出 CT 内部方程；因此只能依据图示和公开 EMA 代码重建。
4. 表 IV 本身没有标注 MAE/RMSE 单位，但原文 p.5 明确说明容量按 `C/C0` 归一化；Fig. 4 左侧进一步以 `Error (%)` 展示同一结果，其中 NASA Ours 的约 0.45%/0.88% 正好对应表 IV 的 0.0045/0.0088 乘以 100。因此表 IV 报告的是 `C/C0` 空间中的无量纲 MAE/RMSE，而不是 Ah。Fig. 4 右侧容量轨迹则恢复为物理容量刻度展示。复现代码仍同时报告两套单位，避免横向比较时混淆。
5. “十次独立运行取平均”没有公开 seed；模型在小样本 NASA 上对初始化较敏感，无法逐数复现其均值。

## 6. 数据与训练协议核对

| 项目 | 原文 | 当前复现 | 判定 |
|---|---|---|---|
| 测试电池 | B0005 | B0005 | 一致 |
| 训练/验证电池 | B0006/B0007/B0018 | 相同 | 一致 |
| 测试隔离 | 不参与训练、验证和统计量 | 完全隔离 | 一致 |
| 窗口任务 | 16 个真实历史循环预测下一循环 | 相同 | 一致 |
| 起点 | 50/90 | 相同 | 一致 |
| EOL | 70% `C0`，首次低于阈值 | 相同 | 一致 |
| 训练/验证 | 80/20 | 固定随机窗口拆分 | 拆分方式未披露 |
| 异常处理 | 3σ 孤立异常 + 线性插值 | 满足原则，但局部趋势和 sigma 估计为重建假设 | 伪代码未披露 |
| 优化器 | Adam, lr=1e-3 | 相同 | 一致 |
| batch/patience | 128/10 | 相同 | 一致 |
| 最大 epoch | 未报告 | 300 | 实现假设 |
| checkpoint | validation MAE | validation MAE | 一致 |

## 7. 诊断实验

诊断只改变原文未披露的参数；checkpoint 始终依据验证集 MAE 选择，没有利用 B0005 选择 epoch。完整逐运行结果位于 `outputs/batter_moe_reproduction_audit/`。

### 7.1 当前默认实现的 seed 0–9

- SP50 归一化 MAE：均值 0.005755，标准差 0.001612，单次最优 0.003107。
- SP50 换算为 Ah：平均约 0.011511 Ah。
- 论文表 IV 的 0.0045 是否为归一化误差，原文未明确说明。

### 7.2 未披露结构参数扫描（seed 0/3/8）

| 变体 | SP50 平均归一化 MAE | 结论 |
|---|---:|---|
| 当前默认：8 heads, r=16, G=8, scale embedding on | **0.003778** | 目前最优结构组合 |
| 4 heads | 0.005299 | 下降 |
| 2 heads | 0.007049 | 明显下降 |
| r=4 | 0.005734 | 下降 |
| r=8 | 0.006545 | 下降 |
| G=4 | 0.006564 | 下降 |
| G=16 | 0.006597 | 下降 |
| G=32（EMA 原型默认 factor） | 0.005869 | 下降 |
| scale embedding off | 0.006904 | 明显下降 |

该结果与 NASA 约 94.5K 的论文参数量也相互印证：`G=8` 当前实测 94,054 参数；`G=4` 为 96,006，`G=16` 为 93,558。因而保留 8 heads、`r=16`、`G=8` 和尺度嵌入更合理。

### 7.3 CT 标量门初始化

原文没有报告 `theta` 初始化。旧实现使用 `theta=0`，即初始 `gamma=softplus(0)=0.693`；直接以 `theta=1` 初始化时，seed 0--9 的 SP50 平均归一化 MAE 由 0.005755 降至 **0.005053**，平均停止轮次由 33.3 降至 27.3，更接近论文表 VII 报告的 22。该初始化已作为显式配置写入模型；它仍属于有实验支持的复现假设，而不是原文明示值。

正式十 seed 复现结果如下：

| 起点 | 指标单位 | MAE（均值±标准差） | RMSE（均值±标准差） | RE（均值±标准差） |
|---|---|---:|---:|---:|
| SP50 | `C/C0` | 0.005053±0.001578 | 0.007264±0.001059 | 0.01467±0.02104 |
| SP50 | Ah | 0.010106±0.003157 | 0.014528±0.002117 | 同上 |
| SP90 | `C/C0` | 0.005049±0.001411 | 0.007881±0.000779 | 0.03143±0.04508 |
| SP90 | Ah | 0.010097±0.002822 | 0.015763±0.001558 | 同上 |

按原文确定的归一化口径比较，论文 SP50 的 MAE/RMSE 为 0.0045/0.0088，当前复现为 0.00505/0.00726；SP90 论文为 0.0046/0.0084，当前为 0.00505/0.00788。MAE 尚有约 10%--12% 差距，但 RMSE 已处于相同甚至略优的数量级，说明主体架构已经接近，剩余差距更可能来自未公开的 seed、异常检测、拆分和初始化细节。

### 7.4 原文复现口径与统一比较口径

使用更新后的 `theta=1`，seed 0/3/8 在两种数据入口上的结果为：

| 数据入口 | SP50 平均 MAE (Ah) | 说明 |
|---|---:|---|
| 原始 NASA `.mat` + BATTER-MoE 重建的局部 3σ 清洗 | **0.00741** | 用于逼近原文自身实验 |
| 三模型共享的 BATTER-MoE 对齐后数据 | 0.01259 | 用于统一预处理的公平横向比较 |

两者模型、划分和 seed 相同，差异来自异常点处理后的训练轨迹。NASA 训练样本很少，少数容量点的变化会改变验证早停位置，并通过随机初始化放大。因此，原文复现结果和统一基准结果必须分开保存与表述：前者回答“能否接近论文”，后者回答“同一数据口径下谁更好”。

## 8. 当前复现边界

在没有作者官方源码、补充材料或作者回复的条件下，无法诚实地声称逐数复现。当前可达到的最高标准是：

1. 所有明确公式逐式一致；
2. 图示模块采用可追踪的高可信原型实现；
3. 所有未披露参数显式配置并做敏感性实验；
4. 同时报告归一化与 Ah 指标；
5. 原始结果、seed、配置、预测轨迹和验证选择全部保存。
