# MGI-DSSM 设计逻辑与证据链

> 文档性质：本文档不是可直接投稿的正文，而是文章撰写前的论证蓝图。它规定“提出什么问题、为何需要本文方法、每项设计由什么证据支持、实验结果达到什么条件后才能形成什么结论”。
>
> 文献检索日期：2026-07-30。  
> 检索范围：经典工作至 2026 年 7 月，优先使用原始研究论文、领域综述和出版社页面。  
> 文献统计口径：下文的“32 篇核心证据文献”是围绕本文主张定向筛选的核心样本，不是 PRISMA 系统综述或全领域文献计量结果，不能写成“对全部已有研究的统计”。

# 一、本文要建立的核心论点

## 1. 一句话核心论点

面向跨电池容量预测，本文从完整放电曲线中反演具有观测和方程支撑的低维有效物理状态，学习其跨循环转移，并通过显式物理截止求解器将预测状态直接解码为下一循环绝对容量，从而建立观测曲线、有效状态、状态转移与容量输出之间可验证的闭环。

对应的信息链为：

$$
\mathcal C_j
\xrightarrow{\text{physical inverse}}
Z_j
\xrightarrow{\text{temporal transition}}
\widehat Z_{k+1}
\xrightarrow{\text{physical forward}}
\widehat Q_{k+1}.
$$

其中：

$$
Z_k=
[Q_{\mathrm{Li},k},C_{n,k},C_{p,k},R_{0,k},R_{p,k}].
$$

五个状态是由有限整电池放电观测反演得到的 **有效物理状态**，而不是原位测得或唯一可辨识的材料级真实参数。

## 2. 本文真正要解决的矛盾

本文不是简单比较“数据驱动”和“物理模型”谁更好，而是处理两者之间的结构性矛盾：

- 纯数据驱动模型能够灵活拟合复杂、非线性的容量退化轨迹，但预测精度不自动意味着模型学习到了可验证的退化状态；
- 完整电化学模型具有较强的物理解释能力，但参数多、标定要求高、计算成本较大，且跨循环退化转移通常没有可靠的闭式表达；
- 物理引导神经网络能够融合两者，但“加入物理特征或物理损失”不等于预测状态和容量输出形成了物理闭环；
- 电池容量序列通常变化缓慢，单步预测极易接近 Persistence，因此仅报告较高的 $R^2$ 或较低的 MAE 不能充分证明模型学习了退化动力学。

本文的解决思路是：

1. 已知的曲线—状态—容量关系尽量使用显式方程；
2. 未知的跨循环状态转移使用神经网络学习；
3. 每个状态维度必须具有观测来源或方程作用；
4. 下一循环容量必须由预测状态产生，而不是由上一循环容量加残差得到；
5. 同时用预测误差、物理闭合、多步 rollout、消融和失败案例约束结论。

## 3. 跨数据集模型一致性

NASA、CALCE 与 TJU 均使用 `mgi-physics`：从完整放电曲线反演五维有效物理状态，经统一的状态转移网络预测下一循环状态，再由显式截止电压求解器得到绝对容量。三个数据集仅根据各自额定容量、截止电压、放电电流、OCP先验与训练超参数设置工况，不再设置宏观代理状态或学习型容量解码分支。

因此，跨数据集实验可以用于检验同一“曲线反演—状态转移—物理容量解码”流程在不同电池与工况下的适用性；具体物理参数仍应表述为受有限观测约束的有效整电池状态，而非材料级直接测量真值。

---

# 二、设计逻辑

## 0. 正式采用的任务级问题

> 本节是 Introduction、Abstract 和 Related Work 中应当使用的问题主线。后面的“大类方法问题”只作为背景知识，不得替代本节。

本文的具体任务不是一般 SOH 估计，也不是早期循环寿命分类，而是：

$$
\{\mathcal C_{k-63},\ldots,\mathcal C_k\}
\longrightarrow
\widehat Q_{k+1}
$$

以及在递归条件下预测：

$$
\widehat Q_{k+1:k+H}.
$$

其中输入为历史完整放电曲线，测试电池不参与训练，模型需要同时预测下一循环有效状态和绝对容量。正文将相关困难凝练为两个核心挑战，随后再用 P1–P5 五个证据维度进行内部拆解。

### 0.1 正文最终凝练版：针对两类方法的两个缺口

正式文章的核心问题应当各自对应一类现有方法，并且都能被本文架构直接回答。容量恢复、域偏移和小样本属于实验场景与次级困难，不作为摘要和引言中的一级挑战。

#### 中文推荐版本

可直接用于引言的问题收束与方法引出，严格控制为三句话：

> **代表性数据驱动方法主要从历史容量、健康指标或循环信号中直接学习未来容量、SOH或RUL，但其优化目标通常不要求中间表示对应由放电曲线可辨识、并具有明确方程作用的有效状态，因此难以显式刻画内部状态如何跨循环演化并形成容量衰退 [R2–R6, R19, R22, R25]。现有物理引导方法虽已通过物理健康指标、经验退化方程、状态空间关系或物理正则项增强预测，但当物理知识主要作用于输入、特征或优化目标、而容量仍由学习型输出给出时，预测状态、放电曲线与截止容量并不必然由同一物理前向关系闭合 [R7–R10, R21, R23]。针对这两个缺口，本文从完整放电曲线中反演具有明确方程作用的低维有效状态，学习其未知的跨循环转移，并通过同一物理前向模型和截止电压求解器将预测状态直接解码为下一循环绝对容量。**

以下为便于拆句和修改的展开版本：

> **第一，代表性数据驱动方法主要从历史容量、健康指标或循环信号中学习到未来容量/SOH/RUL的直接映射 [R2–R6, R19, R22, R25]。这类映射能够取得较高预测精度，但其中间表示通常不承担预先定义的物理角色，因而难以从完整放电曲线出发，显式刻画内部有效状态如何跨循环演化并最终形成容量衰退。**
>
> **第二，现有物理引导方法已通过物理健康指标、经验退化方程、状态空间关系或物理正则项增强预测 [R7–R10, R21, R23]，但物理知识往往作用于输入、特征或优化目标，预测容量仍可由学习型输出头直接给出。由此，预测状态、放电曲线与截止容量之间未必受到同一物理前向关系的强制闭合约束。**

紧接着用一句话引出本文：

> **针对上述缺口，本文从完整放电曲线中反演具有明确方程作用的低维有效状态，学习其未知的跨循环转移，并通过同一物理前向模型和截止电压求解器将预测状态直接解码为下一循环绝对容量。**

#### 英文推荐版本

> **First, representative data-driven approaches primarily learn a direct mapping from historical capacity, health indicators or cycling signals to future capacity, SOH or RUL [R2–R6, R19, R22, R25]. Although effective for prediction, their intermediate representations are generally not required to assume predefined physical roles, making it difficult to explicitly describe how curve-observable effective states evolve across cycles and give rise to capacity fade.**
>
> **Second, existing physics-informed approaches incorporate domain knowledge through physical health indicators, empirical degradation equations, state-space relations or physics-based regularization [R7–R10, R21, R23]. However, when physics acts mainly on the inputs, features or training objective while capacity remains a learned output, the predicted state, discharge response and cutoff capacity are not necessarily closed by the same physical forward relation.**
>
> **To bridge these gaps, we invert full discharge curves into low-dimensional effective states with explicit roles in the governing equations, learn their unknown cross-cycle transition, and decode the predicted state into the next-cycle absolute capacity using the same physical forward model and cutoff-voltage solver.**

### 0.2 为什么采用这两个缺口

| 缺口 | 针对的方法 | 文献中可核验的事实 | 本文直接解决方式 |
|---|---|---|---|
| 缺少显式、可验证的有效状态演化链 | 数据驱动容量/SOH/RUL预测 | 代表模型主要以容量、健康指标或循环信号为输入，直接优化未来容量、SOH或RUL [R2–R6, R19, R22, R25] | 曲线反演五维有效状态；在状态空间而非容量标量空间学习跨循环转移 |
| 物理引导未必形成状态—曲线—容量强闭合 | 物理特征、PINN、physics-regularized和混合模型 | 物理信息进入健康指标、经验/状态方程、特征或复合损失 [R7–R10, R21, R23] | 同一物理前向模型用于曲线解释和容量截止求解；容量必须由预测状态产生 |

### 0.3 表述边界

为避免把结构差异夸大成对整个领域的否定，正式写作必须遵循：

- 使用“representative”或“many existing approaches”，不使用“all existing methods”；
- 使用“are not necessarily required to”，不使用“cannot”；
- 不说数据驱动方法没有价值，而是指出其优化目标不要求物理状态闭合；
- 不说物理引导方法约束太弱，而是指出物理知识进入模型的位置不同；
- 不说已有方法完全没有状态空间，而是强调本文状态由曲线反演、进入明确方程并直接生成容量；
- “状态—曲线—容量闭合”是本文提出并需要实验验证的标准，不是已被文献公认的唯一正确标准；
- 五维状态是 effective states，不是材料级真实参数；
- reversible、多尺度趋势、小样本和跨电池差异属于支撑这两个核心缺口的次级设计与实验，不再并列成一级挑战。

### 0.4 两个缺口与全文证据链

| 核心缺口 | 方法证据 | 关键消融 | 结果证据 |
|---|---|---|---|
| 数据驱动直接映射缺少显式有效状态演化 | 曲线反演、五维状态、状态历史与转移 | capacity-only、自由 latent state、去掉 DVA/截止点、仅当前状态、去掉多尺度趋势 | 状态预测误差、轨迹稳定性、单步/多步容量误差 |
| 物理引导未必形成强闭合 | 物理前向模型、容量截止求解、闭合门控 | 学习型容量头、$Q_k+\Delta Q$、去掉曲线监督、去掉闭合门控 | 曲线重建误差、closure MAE、容量误差、状态敏感性 |

### 0.5 两个缺口的“文献事实—本文判断—本文解法—验证”闭环

#### 缺口一：数据驱动方法中的状态演化链

**文献事实。** Roman pipeline [R2] 从充电片段构造工程特征并估计 SOH；LSTM–GPR [R4] 从历史容量预测未来容量及 RUL；PatchFormer [R5] 使用 patch-wise 和 feature-wise attention 建模容量退化序列；BiGRU-MSTA [R22] 使用多尺度时序注意力预测容量/RUL；TFDM-CR [R25] 在时频域建模包含恢复现象的容量序列。这些论文的主要监督目标都是容量、SOH 或 RUL。

**本文判断。** 上述事实不能推出这些模型“没有内部表示”或“不可解释”。能够成立的窄化判断是：

> 它们的预测目标并不要求中间表示同时满足曲线可辨识性、明确方程作用和状态到容量的物理闭合。

这是对优化约束的比较，不是对模型预测能力的否定。

**本文解法。** 本文将学习对象从容量标量转移到：

$$
Z_k=
[Q_{\mathrm{Li},k},C_{n,k},C_{p,k},R_{0,k},R_{p,k}],
$$

并令：

$$
\widehat Z_{k+1}
=
F_\theta(Z_{k-63:k}),
$$

使网络学习“有效状态怎样变化”，而不是直接自由生成容量。

**必须验证。**

1. 五维状态是否可以稳定反演；
2. 状态转移是否比 capacity-only 映射更有利于多步预测；
3. 自由 latent state 是否能取得相同结果；
4. 五维物理命名是否真的对应不同的容量敏感性；
5. 如果状态模型不优于自由 latent state，则不能把显式状态写成性能贡献，只能写成可审计性贡献。

#### 缺口二：物理引导方法中的状态—输出闭合

**文献事实。** PINN4SOH [R7] 将经验退化和状态空间方程与神经网络结合；JES PINN [R8] 将物理先验用于 SOH 估计；领域知识引导模型 [R9] 构造具有物理意义的健康指标后回归容量；MSTEA-Net [R21] 使用数据误差和双物理正则构成复合损失；混合轨迹预测模型 [R23] 使用电化学模型与测量电压构造预测特征。

**本文判断。** 这些方法已经真实地利用了物理知识，不能概括成“物理引导太弱”。本文与它们的窄化差异是：

> 物理知识进入输入、特征、状态方程或损失，并不等价于要求最终容量只能由预测状态经同一物理前向模型产生。

因此，本文关注的是 **constraint placement** 和 **closure path**，而不是简单比较 physics loss 的强弱。

**本文解法。**

$$
\widehat V_{k+1}(q)
=
V(q;\widehat Z_{k+1}),
$$

$$
\widehat Q_{k+1}
=
\inf\{q:\widehat V_{k+1}(q)\le V_{\mathrm{cut}}\}.
$$

预测容量没有独立自由容量头；状态、端电压和截止容量由同一前向模型连接。

**必须验证。**

1. 预测状态能否重建下一循环曲线；
2. 状态能否闭合下一循环容量；
3. 物理解码与学习型容量头相比是否保持预测能力；
4. 闭合是否只是反演时使用容量标签造成的结果；
5. OCP 先验改变时闭合和状态解释是否稳定；
6. 若多个状态组合产生相同容量，必须报告可辨识性边界。

### 0.6 三句话中每句话的职责

| 句子 | 唯一职责 | 不能夹带的内容 |
|---|---|---|
| 第一句 | 数据驱动模型的结构缺口 | 不讨论容量恢复、小样本、效率或跨域 |
| 第二句 | 现有物理引导模型的闭合缺口 | 不笼统否认 PINN、物理特征或混合模型 |
| 第三句 | 本文如何同时连接两个缺口 | 不提前声称精度显著提升或参数真实可辨识 |

### 0.7 推荐的最终措辞强度

建议采用下列动词：

- `primarily learn`：描述代表性数据驱动模型；
- `are not required to`：描述其优化目标没有强制物理状态；
- `incorporate`：承认现有方法真实引入物理知识；
- `do not necessarily enforce`：描述闭合并非必然；
- `we investigate`：将强闭合作为本文要验证的问题；
- `effective states`：限制状态解释强度。

避免使用：

- `ignore physics`；
- `lack any interpretability`；
- `weakly guided`；
- `fail to model degradation`；
- `true microscopic states`；
- `guarantee physical correctness`。

写作限制：

- Abstract：两个缺口各压缩为半句至一句；
- Introduction：用中文推荐版本的两段加一句方法引出；
- Related Work：分别用真实代表模型展开；
- Method：逐一回答“状态从哪里来、如何转移、怎样生成容量”；
- Experiments：证明显式状态链和强闭合确实有用，而不是只展示可视化；
- P1–P3 只作为预测环境和实验压力条件；P4–P5 分别服务于上述两个核心缺口。

### P1. 局部容量恢复、测量噪声与长期衰退趋势共存

这是当前容量轨迹预测方向直接面对的问题，而不是某个模型大类的抽象缺陷。

容量序列同时包含：

- 长期缓慢衰退；
- 局部容量恢复；
- 平台；
- 传感器和环境噪声；
- 内部电化学过程随机性；
- 个别异常跳变。

PatchFormer [R5] 将容量恢复造成的突变和严重波动明确列为 RUL 预测难点。考虑 polarization recovery 的容量预测研究 [R24] 指出，容量波动会显著影响放电容量预测精度。TFDM-CR [R25] 更具体地指出，噪声使序列高度波动，而容量恢复在退化轨迹中引入局部突变，干扰模型捕获长期趋势。两状态混合模型 [R26] 也直接将 capacity recovery 称为容量预测的主要挑战。

可保留的原文短句：

- PatchFormer [R5]： “capacity regeneration phenomena.”
- TFDM-CR [R25]： “introduces local mutations in the degradation trajectory.”
- Two-state hybrid model [R26]： “capacity recovery phenomenon is a major challenge.”

本文对应的设计不是泛泛地“提高特征提取能力”，而是：

1. 通过 $Z$、$\Delta Z$、MA8 和 MA32 分离状态水平、局部变化、短期趋势和长期趋势；
2. 使用 reversible 状态转移允许局部回调；
3. 使用有界乘性更新防止噪声导致无界状态漂移；
4. 用恢复片段专项实验验证，而不是只看全局 MAE。

必须验证：

- 恢复区与非恢复区分段误差；
- direction loss = 0 与单调约束的对比；
- MA8/MA32 消融；
- 多步 rollout 是否平滑掉真实恢复，或错误放大噪声。

### P2. 不同电池与工况形成不同退化域，单一轨迹规律难以直接迁移

跨电池容量轨迹预测的难点不是一般意义上的“泛化不好”，而是训练电池与测试电池的退化路径、速度和局部模式不同。

MAGNet [R27] 将不同运行条件明确建模为 domain shift，并针对 capacity and energy trajectory forecasting 设计域泛化方法。跨工况早期退化预测研究 [R28] 直接指出，现有方法面临数据不足和不同运行条件下泛化困难。多任务容量/功率退化研究 [R29] 将制造差异和耦合非线性老化机制列为预测挑战。EES 的动态工况退化路径研究 [R30] 则指出，在多样使用场景下准确预测未来退化行为仍是重要挑战。

可保留的原文短句：

- MAGNet [R27]： “treats differences in operating conditions as domain shifts.”
- Cross-condition study [R28]： “generalization under different operational conditions.”
- Multi-task study [R29]： “intrinsic manufacturing variances and coupled nonlinear ageing mechanisms.”

本文对应的设计与边界：

1. 采用严格留一电池，而不是同一电池前后段随机切分；
2. 状态标准化仅由训练电池估计；
3. 物理前向关系提供跨电池共享结构；
4. 神经网络只学习状态转移；
5. 当前 CALCE 四块同化学体系电池只能证明 cell-level transfer，不能证明跨化学体系泛化；
6. TJU 分支因输入和状态含义不同，只能作为观测受限适配，不能伪装成相同模型的跨化学验证。

必须验证：

- 每块测试电池单独报告；
- 跨电池 mean ± SD；
- 不同预测起点；
- 不同训练电池组合；
- 状态分布 shift；
- 目标电池不微调与微调结果分开。

### P3. 有限电池样本下，完整寿命轨迹监督稀缺

容量退化实验周期长、成本高，能够覆盖完整寿命且协议一致的电池数量有限。这是容量轨迹预测文献直接讨论的问题。

RCGAN 容量预测研究 [R31] 明确将训练数据有限归因于昂贵实验和数据共享限制。有限数据轨迹预测研究 [R32] 直接指出，小样本限制会影响容量退化预测。混合物理—数据轨迹预测 [R23] 使用物理特征、聚类和数据增强，在有限训练数据下预测未来轨迹。

可保留的原文短句：

- RCGAN [R31]： “limited availability of training data.”
- Hybrid trajectory model [R23]： “using only 20% of training data.”

本文不通过生成数据解决该问题，而是：

1. 用低维五状态压缩每循环完整曲线；
2. 用显式方程承担已知映射，减少自由网络需要学习的关系；
3. 用训练电池共享的物理状态空间提高样本利用效率；
4. 使用小型 GRU 学习剩余的未知转移。

这一设计是否真的更节省数据必须通过训练电池数量和训练窗口比例敏感性实验验证，不能仅凭模型结构宣称。

### P4. 历史观测到未来容量的直接映射，没有显式回答“什么状态发生了怎样的转移”

这一问题必须精确表述为**本文与主流轨迹预测模型的结构差异**，不能写成“现有模型都没有状态”。

PatchFormer [R5]、BiGRU-MSTA [R22]、TFDM-CR [R25] 和 LSTM–GPR [R4] 都能够有效预测容量或 RUL，但其主要预测路径是从容量/健康指标时序中提取模式，再输出未来容量或寿命。它们研究的是如何更好地拟合轨迹。本文进一步提出一个不同问题：

> 能否让下一循环容量必须经过“曲线可辨识有效状态—状态转移—截止容量解码”产生？

该问题是本文提出的研究选择，不是其他论文作者承认的缺陷。因此正确写法是：

> Existing trajectory forecasters primarily optimize the mapping from historical health sequences to future capacity or RUL [R4, R5, R22, R25]. We investigate a complementary formulation in which the forecast is mediated by an explicitly inverted effective state and a physical cutoff solver.

不能写：

> Existing methods ignore battery states or cannot model degradation mechanisms.

本文必须用以下实验证明这种中介状态不是多余包装：

- 仅容量输入模型；
- 自由 latent state 模型；
- 五维反演状态模型；
- 学习型容量头；
- $Q_k+\Delta Q$ 残差头；
- 物理截止解码；
- 单步、多步和闭合三类指标共同比较。

### P5. 曲线拟合、状态预测与容量输出可能分别准确，但三者未必一致

这是本文最核心、也最需要谨慎表述的问题。

已有物理引导模型通过健康指标、经验退化方程、状态空间方程或物理正则增强预测 [R7–R10, R21, R23]。例如：

- PINN4SOH [R7] 使用经验退化和状态空间方程；
- MSTEA-Net [R21] 使用数据误差和双物理正则组成复合损失；
- 混合轨迹模型 [R23] 从电化学模型和测量电压提取混合特征；
- 这些都是真实且有效的物理融合方式。

但本文要研究的是更窄的结构问题：

> 预测状态是否通过同一个显式物理前向模型重建曲线，并由第一次达到截止电压的位置生成容量？

这一“状态—曲线—容量闭合”是本文提出的验证标准，不能说成已有文献已经证明的普遍缺陷。正确写法是：

> Prior physics-informed approaches introduce physical knowledge through features, degradation equations or regularization [R7, R21, R23]. Here, we examine a stricter state-mediated closure criterion: the predicted state must reconstruct the discharge response and generate capacity through the same cutoff-voltage solver.

必须验证：

1. 反演状态重建当前循环曲线；
2. 反演状态闭合当前循环容量；
3. 预测状态重建下一循环目标曲线；
4. 预测状态闭合下一循环容量；
5. 去掉物理解码后，预测精度和闭合如何变化；
6. 不同 OCP 先验下闭合是否稳定；
7. 参数相关性是否允许多组状态产生相似闭合结果。

### 0.1 最终问题链

文章的问题链应当写成：

> Battery capacity forecasting must reconcile slow long-term fade with local regeneration and noisy fluctuations [R5, R24–R26], while transferring degradation dynamics across cells whose trajectories differ because of manufacturing and operating-condition shifts [R27–R30]. These difficulties are compounded by the limited number of consistently cycled full-life cells [R23, R31, R32]. Existing sequence forecasters address the trajectory-mapping problem using recurrent, patch-based, attention or probabilistic models [R4, R5, R22, R25]. We study a complementary state-mediated formulation: historical discharge curves are inverted into bounded effective states, the unknown cross-cycle state transition is learned, and the future capacity is generated by a shared physical cutoff solver.

对应中文逻辑：

> 电池容量预测需要同时处理缓慢长期衰退、局部容量恢复和噪声波动，并将退化动力学迁移到轨迹存在差异的未见电池；完整寿命样本有限又进一步增加了学习难度。现有循环、Patch、注意力和概率模型主要优化历史健康序列到未来容量轨迹的映射。本文研究一种互补的状态中介形式：将历史放电曲线反演为有界有效状态，学习未知的跨循环状态转移，再通过共享物理截止求解器生成未来容量。

### 0.2 不再作为本文核心问题的表述

以下内容只能作为背景，不得放进摘要的问题句或贡献动机：

- 数据驱动方法都是黑箱；
- 数据驱动方法参数量普遍过大；
- 数据驱动方法都缺乏泛化能力；
- 物理引导方法都只是在 loss 中加约束；
- 物理模型都无法在线应用；
- 所有现有方法都忽略内部状态；
- 本文解决了真实材料参数辨识。

## 1. 数据驱动方法大类背景：不作为正式研究缺口

### 1.1 文献统计与方法分布

第一轮筛选的 22 篇文献用于方法类别、可辨识性和评测背景；在任务问题修正后，又增加了 10 篇直接研究容量恢复、退化轨迹、域偏移和有限寿命样本的任务级文献 [R23–R32]。因此，正式研究问题应主要引用 [R23–R32]，而不是依赖大类综述。

前 22 篇的分类为：

- 9 篇以纯数据驱动容量/SOH/RUL 预测或基线评测为主；
- 5 篇以物理引导或领域知识引导学习为主；
- 5 篇聚焦电压曲线诊断、退化模式或参数可辨识性；
- 3 篇为领域综述或方法论总结。

在 6 篇代表性数据驱动研究中，常见输入和输出形式包括：

| 类型 | 代表工作 | 输入 | 输出/任务 | 对本文的启示 |
|---|---|---|---|---|
| 早期寿命统计特征 | Severson et al. [R1] | 早期循环放电曲线差异及统计特征 | 总循环寿命 | 数据驱动方法能从曲线中发现强预测特征 |
| 人工特征＋传统机器学习 | Roman et al. [R2] | 充电曲线片段及 30 个工程特征 | SOH 与置信区间 | 特征工程可实现准确、轻量且带不确定性的预测 |
| 深度网络＋域适配 | Lee et al. [R3] | 多制造商电池运行数据 | 无目标标签条件下的 SOH | 跨电池/跨域预测需要专门的适配机制 |
| 序列模型＋概率模型 | Richardson et al. [R4] | 容量历史 | 未来容量和 RUL | 递归多步与不确定性是独立于单步误差的重要问题 |
| Patch/Transformer | PatchFormer [R5] | 容量及循环级特征序列 | RUL/容量趋势 | 多尺度局部—全局建模可刻画容量恢复与长程依赖 |
| 生成式预训练模型 | BatteryGPT [R6] | 循环内电压、电流、温度序列 | 全寿命 SOH 轨迹 | 大模型可学习丰富时序表示，但仍以数据映射为核心 |

这些文献说明：不能把数据驱动方法简单描述为“效果差”或“无法建模退化”。数据驱动方法已经能够：

- 建模复杂非线性关系；
- 从原始或工程特征中提取退化信息；
- 捕获局部容量恢复和长期依赖；
- 进行跨域适配和不确定性估计；
- 在部分数据集上取得很高的预测精度。

本文应承认这些能力，再从“状态是否可验证、输出是否闭合、评测是否存在捷径”切入。

### 1.2 问题一：预测精度不等于学习到了可验证的退化状态

深度网络可以建立：

$$
\widehat Q_{k+1}=F_\theta(X_{1:k}),
$$

但其中间表示通常由预测损失端到端决定。即使模型获得较低 MAE，中间坐标也不一定对应可观测、可复现或可进入物理方程的退化状态。

这里需要准确区分：

- “黑箱”不等于模型完全没有解释工具；
- attention、SHAP 或特征重要性能够解释输入对输出的影响；
- 但它们通常不能证明隐变量与电池内部状态具有一一对应关系；
- 一个隐变量只有在具有明确观测来源、方程作用和独立验证时，才适合被赋予物理含义。

本文能够回答的部分是：

- 五维状态不是任意神经网络隐变量，而是通过放电电压曲线反演；
- 每一维都进入端电压或化学计量方程；
- 同一物理前向模型同时用于曲线重建和容量解码；
- 状态可接受曲线重建、容量闭合、敏感性和稳定性检查。

本文仍不能声称：

- 五维状态是唯一可辨识的真实材料参数；
- 状态轨迹等价于原位电化学测量；
- 有效状态完全揭示了具体副反应机理。

### 1.3 问题二：单步容量预测容易受 Persistence 捷径影响

对于缓慢变化的容量序列，最简单的基线为：

$$
\widehat Q_{k+1}^{\mathrm{pers}}=Q_k.
$$

如果相邻循环容量变化很小，则 Persistence 本身就可能取得很低 MAE。由此产生三个风险：

1. 一个复杂模型可能主要复制最后容量，而不是真正预测退化状态；
2. 很高的 $R^2$ 可能主要来自容量随循环的整体趋势，而非对局部变化的准确建模；
3. 单步优势不能自动转化为递归多步优势。

本文的结构性应对是：

$$
\widehat Q_{k+1}
=G_{\mathrm{phys}}(\widehat Z_{k+1}),
$$

而不是：

$$
\widehat Q_{k+1}=Q_k+\Delta\widehat Q_{k+1}.
$$

但需要谨慎：状态历史本身由过去曲线反演，其中仍包含历史容量信息。因此本文只能主张：

> 模型不在输出端显式使用上一循环容量作为加性锚点。

不能主张：

> 模型完全不利用历史容量信息。

Persistence 必须作为所有单步和多步主表的显式基线。

### 1.4 问题三：跨电池泛化常被数据划分和协议差异掩盖

近年来的综述指出，实验室标准数据上的高精度不必然转化为真实工况下的鲁棒性，主要问题包括：

- 电池化学体系、制造商和工作条件改变；
- 不完整充电、动态负载和温度扰动；
- 标签稀缺；
- 随机窗口划分造成同一电池轨迹进入训练和测试；
- 不同论文使用不同预测起点、EOL 阈值和清洗规则；
- 只报告最优运行而不报告重复实验分布。[R3, R16–R18]

本文的应对包括：

- 测试电池完全不进入模型拟合；
- 标准化统计量只由训练电池计算；
- 验证和早停不使用测试电池；
- 测试窗口只使用目标循环之前的信息；
- 明确报告每个电池、预测起点和窗口数；
- 十个随机种子全部保留；
- CALCE PatchFormer 协议和 MSTEA 完整寿命协议不能混表；
- TJU 本地清洗版本不能与原始文献版本无条件比较。

### 1.5 问题四：复杂网络结构可能弱化可复现性和部署解释

Transformer、混合注意力、分解模型和集成网络能够提高特征提取能力，但也可能带来：

- 参数量和训练成本增加；
- 模块贡献难以隔离；
- 对预处理和超参数高度敏感；
- 边缘 BMS 部署成本；
- 精度提升来源难以判断。

本文不应笼统宣称所有数据驱动模型“参数量过大”。只有在统一硬件、统一输入和端到端计时下，才能比较效率。本文更适合强调：

- 时序转移网络本身紧凑；
- 已知映射由物理方程承担；
- 网络只学习未知的跨循环转移；
- 端到端成本还必须计入曲线反演和物理求解。

### 1.6 数据驱动问题的最终精炼表述

可用于 Introduction 的核心表述应接近：

> Data-driven models can flexibly capture nonlinear degradation patterns from cycling data, but low one-step error alone does not establish that their latent representations correspond to verifiable degradation states. This issue is particularly important for slowly varying capacity trajectories, for which persistence provides a strong baseline. Moreover, differences in cell-level splitting, prediction origin and preprocessing can obscure genuine cross-cell generalization.

中文逻辑为：

> 数据驱动模型能够灵活刻画复杂退化轨迹，但较低的单步容量误差并不能证明其中间表示对应可验证的退化状态；对于缓慢变化的容量序列，Persistence 又构成了很强的简单基线。此外，不同的数据划分、预测起点和预处理协议容易掩盖真实的跨电池泛化能力。

这一表述避免了对数据驱动方法的全盘否定，并且每个问题都有本文实验可以回答。

---

## 2. 物理引导方法大类背景：不作为正式研究缺口

### 2.1 不能把“物理引导”视为单一方法

现有物理/领域知识引导方法至少可以按照物理信息进入模型的位置分成五类：

| 类型 | 物理知识进入位置 | 优点 | 主要限制 |
|---|---|---|---|
| 物理特征型 | 输入特征，如 ICA、DVA、阻抗、充电时间 | 易与常规模型结合 | 特征与最终输出之间未必形成显式物理链 |
| 物理正则型 | 损失函数，如单调性、经验退化方程、ECM 残差 | 训练方便，能抑制不合理输出 | 软约束可能被数据损失权重抵消 |
| 物理参数型 | 先辨识 ECM/电化学参数，再用于预测 | 参数具有一定物理含义 | 参数可辨识性和标定成本是关键问题 |
| 混合残差型 | 物理模型给出主预测，网络学习残差 | 精度高，能补偿模型失配 | 残差网络可能重新吸收主要预测关系 |
| 物理状态空间型 | 状态、转移或观测方程中显式嵌入物理 | 结构清晰、可做闭环检查 | 需要定义可观测状态并处理模型失配 |

代表性研究显示：

- Nature Communications 2024 的 PINN 使用经验退化和状态空间方程约束 SOH 建模，并在多来源电池上验证稳定性 [R7]；
- Hofmann 等将物理信息融入 SOH 神经网络，但其验证还涉及 P2D 生成数据与多数据集 [R8]；
- 领域知识引导的健康指标可以在缺失部分数据时保持较好的容量估计 [R9]；
- IEEE TIV 的 PINN 工作将经验或物理动力学与数据驱动模型融合 [R10]。

因此，本文不能写“现有物理引导方法都只是加一个 physics loss”。更准确的问题是：不同物理引导层级提供的闭合强度不同。

### 2.2 问题一：物理约束可能停留在损失层，而非输出生成机制

许多 PINN 或 physics-regularized 方法采用：

$$
\mathcal L
=
\mathcal L_{\mathrm{data}}
+\lambda_{\mathrm{phys}}\mathcal L_{\mathrm{phys}}.
$$

这一范式有效，但仍可能存在：

- 物理项只是软惩罚；
- 物理项权重依赖调参；
- 预测输出仍由自由神经网络头直接给出；
- 较低物理残差不一定意味着状态—输出关系闭合；
- 当经验方程失配时，过强约束可能降低局部预测能力。

本文的差异在于：

- 曲线到状态使用带边界的物理反演；
- 状态进入显式端电压方程；
- 下一容量由端电压到达截止电压的位置求得；
- 容量输出不是自由容量头；
- 同一状态同时接受曲线和容量两种闭合检查。

但本文仍包含曲线损失、状态损失和容量损失，因此不应声称“完全不依赖软物理约束”。准确表述应是：

> 本文将物理知识从损失层进一步延伸到状态构造和容量生成路径。

### 2.3 问题二：物理参数具有含义，不代表它们在给定观测下唯一可辨识

电池参数辨识研究表明：

- 不同参数可能对端电压产生相似影响；
- 参数可辨识性依赖激励、SOC 区间、温度和测量噪声；
- OCV 形状、欧姆内阻等参数往往比部分动力学参数更易辨识；
- 电极级容量、锂库存和电极滑移之间可能高度相关；
- 低倍率 OCV/DVA 能提供退化模式信息，但电阻、滞后和模型失配会造成偏差。[R11–R15]

本文通过以下方式降低风险：

- 只保留五个有观测和方程支撑的有效状态；
- 删除缺少独立观测支撑的第六维残差状态；
- 使用电压曲线、$dV/dQ$、截止点和内阻弱观测；
- 使用参数边界、鲁棒损失和多初值；
- 将 $R_p$ 与终端电压关系结合；
- 对相邻循环仅使用弱连续性而非硬锚定。

但这些措施不能证明全局唯一可辨识。因此本文必须补充：

- 参数相关矩阵；
- 多初值稳定性；
- 局部敏感性；
- 参数边界命中率；
- 反演扰动或 bootstrap；
- 对 OCP 先验变化的敏感性。

最终只允许称为：

- effective physical state；
- curve-identifiable effective state；
- low-dimensional whole-cell proxy。

### 2.4 问题三：过强的单调物理先验可能与局部观测冲突

容量衰退的长期趋势通常向下，但逐循环观测可出现：

- 局部容量恢复；
- 平台；
- 测量噪声；
- 温度和静置历史造成的短期回调；
- 反演误差。

PatchFormer 等方法明确将容量恢复视为预测难点 [R5]。如果强制：

$$
Q_{\mathrm{Li},k+1}\le Q_{\mathrm{Li},k},\quad
C_{n,k+1}\le C_{n,k},\quad
C_{p,k+1}\le C_{p,k},
$$

且：

$$
R_{0,k+1}\ge R_{0,k},\quad
R_{p,k+1}\ge R_{p,k},
$$

模型可能无法响应短期恢复和反演波动。

本文的 reversible transition 允许局部状态回调，同时通过：

- 长历史窗口；
- 有界乘性更新；
- 状态监督；
- 曲线监督；
- 容量监督；
- 长期轨迹分析；

保持总体退化结构。

“Reversible”必须解释为：

> effective state estimates are allowed to exhibit local reversals.

不能写成：

> battery degradation is physically reversible.

### 2.5 问题四：高保真物理模型的成本与先验失配

完整 P2D/DFN 模型能够描述丰富的电化学过程，但通常需要：

- 大量材料和动力学参数；
- 专门实验标定；
- 数值偏微分方程求解；
- 对温度、工况和化学体系进行适配。

本文采用简化的电极 OCP、锂平衡、欧姆压降和一阶极化模型，其目标不是替代高保真电化学模型，而是构建：

- 可计算；
- 可反演；
- 可进入神经状态转移；
- 可通过截止电压闭合容量；

的低维有效状态空间。

这同时带来明确边界：

- 固定 OCP 不是 CALCE 电芯材料的专属真值；
- 一阶极化不能覆盖全部动力学；
- 五维状态不能代表全部副反应；
- 物理闭合可能部分来自模型结构和曲线拟合，而不是材料级机理正确性。

### 2.6 物理引导问题的最终精炼表述

可用于 Introduction 的核心表述应接近：

> Physics-informed approaches improve physical consistency by incorporating mechanistic features, empirical degradation laws or equation-based regularization. However, the strength of this guidance depends on where physics enters the model: a soft physical penalty does not necessarily guarantee that the predicted health state explicitly generates the reported capacity. Moreover, physically named parameters may remain correlated or weakly identifiable under limited full-cell observations, and overly strict monotonic priors may conflict with local recovery and measurement variability.

中文逻辑为：

> 物理引导方法通过机理特征、经验退化规律或方程约束增强预测的一致性，但其约束强度取决于物理知识进入模型的位置：损失层的软约束并不能自动保证预测状态显式生成最终容量。同时，具有物理命名的参数在有限整电池观测下仍可能相关或弱可辨识，过强的逐循环单调先验也可能与局部恢复和测量波动冲突。

本文的对应解法是：

1. 曲线反演构造有效状态；
2. 物理方程定义状态作用；
3. 神经网络仅学习未知状态转移；
4. 截止求解器直接生成容量；
5. 通过闭合、敏感性、可辨识性和 reversible 消融约束解释。

---

# 三、方法大类背景的文献证据审计（非核心问题）

> 本节只用于约束 Related Work 中对数据驱动和物理引导方法的准确描述。本文正式研究问题以第二章第 0 节的 P1–P5 为准。本节内容不得替代任务级问题，也不得直接搬入摘要。

## 1. 证据纪律

前文中的“问题”必须区分三种来源，后续写 Introduction 时不能混用：

- **L：文献直接表述。** 原论文明确把该问题写成挑战、限制或实验发现；
- **I：基于多篇文献的保守推论。** 文献提供事实，但总结句是本文作者的综合判断；
- **A：本文自己的评测风险。** 来源是本地模型结构、数据协议或结果审计，不能写成“已有研究普遍认为”。

只有 L 类可以直接写成“existing studies remain limited by…”。I 类必须写成“these observations suggest…”。A 类应写成“to test whether our model…”，而不是用于批评全部已有方法。

## 2. 数据驱动问题的原文依据

| 拟写问题 | 类型 | 文献中的明确表述或发现 | 可写到什么程度 |
|---|---|---|---|
| 跨电池和跨工况泛化困难 | L | Wang et al. [R7] 明确写道，不同电池类型和工况使可靠、稳定的 SOH 估计仍具挑战；其正文进一步指出数据驱动模型的泛化依赖所提取特征，且不同数据集往往需要特定特征。 | 可写“跨电池泛化仍是公开挑战”，不能写“数据驱动模型不能泛化”。 |
| 实验室结果与真实工况存在差距 | L | Chen et al. [R18] 明确指出，多数研究依赖实验室标准数据，真实工况中的不规则驾驶、不完整充电和环境变化会带来泛化与鲁棒性问题。 | 可写“实验室高精度不自动代表真实工况鲁棒性”。 |
| 特征和协议可能引入信息泄漏 | L | Wang et al. [R7] 的 Discussion 明确讨论充放电协议和使用条件造成信息泄漏的风险，并采用 battery A 训练、battery B 测试来规避同电池前后段泄漏。 | 可直接支撑电池级隔离和协议审计。 |
| 数据驱动模型的稳定性依赖特征 | L | Wang et al. [R7] 写明数据驱动模型具有精度和效率优势，但泛化依赖特征，稳定性仍有限。 | 可批评“特征依赖性”，不能笼统称全部方法为黑箱。 |
| 隐变量不等于物理状态 | I | 现有代表模型主要从统计特征、容量序列或循环内信号直接映射到 SOH/RUL [R1–R6]；这些论文并未把任意潜变量都证明为可由物理方程闭合的材料参数。 | 应写成本文的定位推论：“预测潜变量通常不接受与本文相同的曲线—容量闭合检验”。不能写成“所有模型都不可解释”。 |
| Persistence 是必要基线 | L＋A | Bhatt et al. [R19] 在电池容量预测中设置了前一步预测下一步的 Persistence 基线；动态卡车电池 SOH 研究也明确将上一时刻值作为下一时刻预测 [R20]。本地 CALCE 结果又显示 Persistence MAE 低于当前模型。 | 可写“Persistence 是已用于电池容量/SOH 预测的真实基线，且在本文任务上很强”。“所有已有论文忽略 Persistence”没有依据，禁止使用。 |
| 高 $R^2$ 不足以证明优于简单基线 | A | 这是由本文当前 CALCE 结果直接得到：$R^2>0.996$，但 MAE 仍比 Persistence 高 5.2%–6.4%。 | 只能作为本文结果分析，不能冒充外部文献共识。 |
| 复杂网络一定计算量大 | I | 文献确实存在 Transformer、Patch、混合注意力等复杂结构 [R5, R6, R21, R22]，但不同实现规模差别很大。 | 只能要求统一效率评测，不能提前宣称对比模型“参数量过大”。 |

### 2.1 可保留的短原文

以下短句用于保证问题表述可追溯；正式论文应结合上下文释义，不建议大量直接引用。

- Wang et al. [R7]： “reliable and stable battery SOH estimation remains challenging due to diverse battery types and operating conditions.”
- Chen et al. [R18]： “most rely on laboratory-standardized test data.”
- PatchFormer [R5]： “abrupt changes and severe fluctuations ... caused by local capacity regeneration phenomena.”
- Bhatt et al. [R19] 所采用的 Persistence 含义：当前循环容量作为下一循环容量预测。

## 3. 物理引导问题的原文依据

| 拟写问题 | 类型 | 文献中的明确表述或发现 | 可写到什么程度 |
|---|---|---|---|
| 高保真物理模型参数多、计算成本高 | L | Wang et al. [R7] 明确指出，不同化学体系需要不同参数，物理模型计算成本高；同文还指出 P2D 等模型包含大量参数且内部参数难以获得。 | 可直接作为采用低维有效模型的动机。 |
| 物理与机器学习的融合层级不同 | L | Wang et al. [R7] 引用并采用 Aykol 等的五类融合框架，区分 sequential integration 与真正融合方程和网络的 hybrid 方法。 | 可写“物理引导不是单一范式”，不能写“已有方法都只加物理 loss”。 |
| 软物理损失不保证状态生成最终容量 | I | MSTEA-Net [R21] 明确采用预测误差与两个物理正则项组成的复合损失；其容量/RUL 仍由网络预测。由此只能推论它与本文“状态经显式截止求解生成容量”的约束位置不同。 | 可写成结构差异，不能写成 MSTEA-Net 的作者自认缺陷。 |
| 参数物理命名不等于唯一可辨识 | L | EIS 研究 [R11] 明确指出等效电路拟合常常非唯一；Zhou 等 [R14] 专门研究容量衰退参数的可辨识性和敏感性。 | 可直接支撑“必须做相关性、多初值和敏感性分析”。 |
| DVA/ICA 能提供退化模式信息，但有边界 | L | DVA 和 peak-tracking 研究 [R12, R13] 将曲线特征用于 LLI/LAM 等退化模式诊断；相关研究同时表明电阻上升等信息不能仅靠单一微分曲线完整辨识。 | 可支撑使用 DVA，但不能把 DVA 当作材料参数真值。 |
| 逐循环硬单调可能与局部容量恢复冲突 | L＋I | PatchFormer [R5] 明确把局部容量恢复造成的突变和波动视为预测难点；BiGRU-MSTA [R22] 也以短期波动和长期趋势的多尺度建模为动机。硬单调约束会不会降低本文模型性能仍需由本文消融验证。 | 文献支撑“恢复现象存在且需建模”；“方向损失有害”只能由本文实验决定。 |
| 物理闭合优于 loss-level guidance | A | 这是本文提出的结构假设，不是现有文献已经替本文证明的事实。 | 必须通过物理解码消融和闭合实验验证后才能形成贡献。 |

### 3.1 可保留的短原文

- Wang et al. [R7]： “batteries with different chemical compositions require different model parameters, and the models have high computational costs.”
- EIS 研究 [R11]：等效电路拟合 “is often non-unique”。
- MSTEA-Net [R21]： “integrating data-driven prediction errors with dual physics-based regularization terms.”
- BiGRU-MSTA [R22]： “modeling both short-term fluctuations and long-term degradation trends.”

## 4. 经审计后可用于 Related Work 的背景表述

### 4.1 数据驱动方法

> Data-driven approaches have achieved accurate battery health prediction from engineered features and cycling sequences [R1–R6]. Nevertheless, their cross-cell robustness remains sensitive to battery chemistry, operating protocol and feature construction [R7, R18]. Battery-level separation is also essential because protocol- or cell-specific information can leak into the evaluation [R7]. Moreover, persistence has been used as a genuine baseline in battery capacity and SOH forecasting [R19, R20], making it necessary to establish whether a complex model contributes information beyond the most recent observation.

这段中的四个判断分别有对应来源：

1. 已取得准确预测：[R1–R6]；
2. 泛化依赖化学体系、工况和特征：[R7, R18]；
3. 需要电池级隔离：[R7]；
4. Persistence 是真实电池预测基线：[R19, R20]。

### 4.2 物理引导方法

> Physics-informed battery models incorporate domain knowledge through engineered indicators, model parameters, governing equations or regularization losses [R7–R10, R21]. These strategies provide different levels of physical coupling. Full physics-based models can be costly and chemistry-specific [R7], whereas physically named parameters may remain non-unique or weakly identifiable under limited full-cell observations [R11, R14]. This motivates a bounded effective-state formulation whose physical role is tested through curve reconstruction and state-to-capacity closure rather than assumed from parameter names alone.

这段中的三个问题分别有对应来源：

1. 物理融合存在不同层级：[R7–R10, R21]；
2. 高保真模型成本和化学体系依赖：[R7]；
3. 参数非唯一或弱可辨识：[R11, R14]。

最后一句是本文的设计选择，不能伪装成文献结论。

---

# 四、证据链总图

## 1. 主张—机制—实验对应关系

| 编号 | 核心主张 | 设计机制 | 必需实验 | 结论门槛 |
|---|---|---|---|---|
| C1 | 模型实现严格跨电池预测 | 留一电池、训练统计隔离、因果窗口 | 数据划分审计、fold 结果 | 无测试电池参与拟合或早停 |
| C2 | 五维状态可由放电曲线有效反演 | 电压、DVA、截止点、内阻、边界 | 曲线重建、闭合、多初值稳定性 | 误差低且状态不频繁越界 |
| C3 | 下一容量由预测状态显式生成 | 物理截止求解器 | 解码器消融、闭合分析 | 移除物理解码后闭合或多步表现下降 |
| C4 | 模型不依赖输出端容量残差捷径 | 绝对物理容量输出 | 与 residual head、Persistence 对比 | 前向结构审计成立，且多步不快速漂移 |
| C5 | 多尺度状态历史有必要 | 水平、差分、MA8、MA32 | 表示消融、历史长度敏感性 | 多步或跨电池结果稳定改善 |
| C6 | reversible 转移比逐步单调更符合观测 | 有界双向乘性更新 | 局部恢复案例、方向损失消融 | 局部片段和整体误差均有证据 |
| C7 | 方法具有物理可审计性 | 状态—曲线—容量闭环 | 轨迹、敏感性、相关性、失败案例 | 解释与方程和扰动结果一致 |
| C8 | 方法具有预测价值 | 单步、多步、EOL | 主对比、Persistence、重复实验 | 至少在预先指定任务上超过强基线 |
| C9 | 方法紧凑或高效 | 小型 GRU＋显式方程 | 参数量、训练和端到端计时 | 统一硬件下有优势，且计入反演成本 |
| C10 | TJU 验证观测受限适配性 | 代理状态＋直接绝对容量 | TJU 留一电池、Persistence、重复实验 | 只能形成适配性结论，不扩展五维物理解释 |

## 2. 结论优先级

文章的证据优先级应为：

1. **有效性**：是否在公平协议下完成预测；
2. **非捷径性**：是否优于 Persistence 或在多步中显示额外信息；
3. **闭合性**：状态是否能同时解释曲线和容量；
4. **必要性**：关键模块消融是否支持设计；
5. **稳定性**：种子、起点、寿命阶段和多步 horizon 是否稳定；
6. **解释性**：状态趋势、敏感性和相关性是否合理；
7. **边界性**：失败案例和不可辨识风险是否被公开。

只有前面的证据成立，后面的解释才有意义。

---

# 五、主实验证据链

## 1. 研究问题

主实验应回答：

- **RQ1：** 在测试电池完全未参与拟合时，模型的单步容量预测表现如何？
- **RQ2：** 与 Persistence、线性趋势和数据驱动模型相比，模型是否提供额外预测信息？
- **RQ3：** 模型在递归多步预测中是否保持稳定？
- **RQ4：** 物理闭环是否在不牺牲过多精度的情况下提供可审计状态？
- **RQ5：** CALCE 主模型与 TJU 适配分支各自能支持什么结论？

## 2. 主实验协议

### 2.1 CALCE 主协议

推荐以统一的完整寿命留一电池协议作为主协议：

- 电池：CS2_35、CS2_36、CS2_37、CS2_38；
- 每次留一块电池作为测试电池；
- 其余电池用于训练；
- 训练电池尾部固定比例作为验证；
- 历史长度固定为 64；
- 单步目标为 $k+1$；
- 测试从预先冻结的统一起点开始；
- 十个固定随机种子；
- EOL 阈值、异常值处理和循环对齐必须固定。

如果保留 PatchFormer 多起点协议，应作为第二协议单独成表，不能与完整寿命协议混合。

### 2.2 TJU 扩展协议

- 明确本地数据版本和清洗后循环数；
- 测试电池与训练电池隔离；
- 不把代理状态称为五维物理状态；
- 结果主要用于说明宏观观测条件下的适配能力；
- 与文献结果比较时注明数据版本差异。

## 3. 对比方法

### 3.1 必须包含的简单基线

1. Persistence：

$$
\widehat Q_{k+1}=Q_k.
$$

2. 线性趋势外推：

$$
\widehat Q_{k+1}=Q_k+\frac{1}{m}
\sum_{j=0}^{m-1}(Q_{k-j}-Q_{k-j-1}).
$$

3. 简单 GRU/LSTM 容量序列模型；
4. 相同历史特征下的 MLP；
5. 如果计算允许，Gaussian process 或 autoregressive baseline。

简单基线用于判断复杂模型是否真正提供额外信息。

### 3.2 数据驱动代表模型

以下模型均对应真实论文。能否进入数值主表取决于是否能够在本文数据和协议下复现；不能复现或输入条件不同的模型只能进入“文献结果对照表”。

| 模型 | 真实来源 | 原论文任务/结构 | 与本文的可比性 | 处理决定 |
|---|---|---|---|---|
| Persistence | Bhatt et al. [R19]；动态卡车 SOH [R20] | 用当前/上一时刻容量或 SOH 预测下一时刻 | 完全可比，且无需训练 | 必须进入所有单步和多步主表 |
| Linear trend / ARIMA | 动态卡车 SOH [R20] | 基于 SOH 时间序列建立 naïve 和 ARIMA 预测 | 只使用容量/SOH，作为低复杂度趋势基线 | 建议本地重实现 |
| LSTM–GPR | Richardson et al. [R4] | LSTM 拟合容量残差，GPR 提供未来容量和不确定性；论文包含 NASA/CALCE 递归预测 | 数据集部分重合，但输入和预测起点需重新统一 | 能获得实现时本地复现，否则只作方法对照 |
| PatchFormer | [R5] | dual patch-wise attention＋feature-wise attention，预测 RUL 并建模容量恢复 | 与 CALCE/TJU 研究背景接近，但需统一其清洗、起点和输入 | 重要对比；优先使用原作者代码复现 |
| BiGRU-MSTA | [R22] | 多层 BiGRU＋multi-scale temporal attention；在 CALCE CS2_35–38 上实验 | 使用相同 CALCE 电池，但训练比例和任务协议不同 | 可本地复现后进入扩展表，不能直接抄原文数值排名 |
| Hybrid trajectory predictor | [R23] | 电化学模型＋测量电压构造混合特征，聚类和增强后用 seq2seq 预测容量轨迹 | 直接研究容量退化轨迹，但原任务为早期全轨迹预测 | 适合物理—数据混合轨迹对比；需统一任务后复现 |
| Polarization-recovery HELM | [R24] | 显式考虑 polarization recovery 的混合集成容量预测 | 直接研究恢复现象，但模型输入和协议需核对 | 用于恢复片段专项对比 |
| TFDM-CR | [R25] | 时频扩散模型，同时处理噪声、长期趋势和恢复突变 | 直接多步容量预测，论文包含 32 步结果 | 适合作为多步前沿对比；需获得代码和相同数据 |
| Two-state LSTM–GPR | [R26] | 分别建模全局退化区和容量恢复区 | 与本文 reversible 动机直接相关 | 适合作为恢复建模对比 |
| MAGNet | [R27] | 面向 domain shift 的多域电池 capacity/energy trajectory forecasting | 与跨电池/跨工况问题直接相关 | 适合作为域泛化对照，不能与普通留一电池结果混为同一任务 |
| BatteryGPT | [R6] | 生成循环内电压、电流、温度时序，再预测全寿命 SOH | 数据规模和任务不同，模型规模差异大 | 只作前沿背景，除非获得完全一致实现 |
| Roman ML pipeline | [R2] | 充电曲线片段＋30 个工程特征＋传统/非参数模型，并输出置信区间 | 任务为 SOH 估计，数据体系不同 | 只作数据驱动和不确定性背景 |

### 3.2.1 对比模型的短原文记录

为避免后续写作时把模型结构“凭印象改写”，保留以下来自原文摘要或正文的短摘录：

- **PatchFormer [R5]**： “integrating the global correlations and local dependencies of the partitioned time series.”
- **LSTM–GPR [R4]**：论文明确报告 “1-step ahead capacity prediction” 和 recursive prediction tests。
- **BiGRU-MSTA [R22]**： “integrates a BiGRU with an MSTA mechanism.”
- **Roman pipeline [R2]**： “engineers 30 features, performs automatic feature selection and calibrates the algorithms.”
- **BatteryGPT [R6]**： “utilizes a classic data-driven method for SOH estimation.”

以上摘录只用于锁定原模型含义。正式相关工作应释义并引用原文，不应大段复制。

### 3.3 物理引导代表模型

| 模型 | 真实来源 | 物理知识进入位置 | 与本文的核心差异 | 处理决定 |
|---|---|---|---|---|
| PINN4SOH | Wang et al. [R7] | 经验退化和状态空间方程；神经网络近似退化动力学 | 输入为充电前短片段统计特征，SOH 由其 PINN 框架估计；不是本文的曲线反演—截止解码 | 最重要的 physics-informed 背景；若数据/代码兼容则复现 |
| JES PINN | Hofmann et al. [R8] | 物理先验和网络联合用于 SOH；包含 P2D 生成数据验证 | 任务与数据来源不同 | 用于物理引导类别对照，不直接排名 |
| MSTEA-Net | [R21] | 数据预测误差＋两个物理正则项组成 triple-composite loss | 物理主要进入损失；网络以多尺度时序卷积和跨变量注意力预测 | 重要对比，且本项目已有协议命名；必须核对原代码和数据版本后再排名 |
| Domain knowledge-guided ML | Lanubile et al. [R9] | 从真实 EV 工况构造具有物理意义的健康指标，再用线性回归估计 SOH | 物理知识进入特征，不构造本文五维动态状态 | 适合作为“物理特征型”代表 |
| Hybrid Bayesian PINN | Nascimento et al.，见 [R7] 参考文献 | 原理模型与 Bayesian neural network 混合用于 prognosis | 任务和不确定性建模不同 | 相关工作背景；获取代码后才考虑复现 |

### 3.3.1 物理引导模型的短原文记录

- **PINN4SOH [R7]**： “model ... empirical degradation and state space equations.”
- **MSTEA-Net [R21]**： “multi-scale temporal convolutional encoding and a cross-variable attention mechanism.”
- **Domain knowledge-guided ML [R9]**： “five health indicators that can be extracted online from real-world electric vehicle operation.”
- **JES PINN [R8]**：原文将方法命名为 “Physics-Informed Neural Networks for State of Health Estimation”。

### 3.4 对比模型进入主表的准入规则

一个模型只有同时满足以下条件才进入可排名的主表：

1. 使用同一原始数据版本；
2. 使用同一训练/验证/测试电池划分；
3. 使用同一预测起点和测试窗口；
4. 输入信息不超过本文模型在该任务中可获得的信息，或在表中明确标注额外输入；
5. 不使用测试电池标签进行微调；
6. 使用相同 EOL 阈值和异常处理；
7. 至少重复相同种子数，或明确标为单次结果；
8. 代码和配置可核验。

不满足上述条件的真实模型仍可出现在“文献背景对照表”，但其原文数值不得与本地结果直接计算排名或提升百分比。

## 4. 单步主表

建议主表格式：

| Method | Input | Test battery | Start | Windows | MAE | RMSE | $R^2$ | Ecycle | Persistence MAE | $\Delta_{\mathrm{pers}}$ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

其中：

$$
\Delta_{\mathrm{pers}}
=
\mathrm{MAE}_{\mathrm{pers}}
-
\mathrm{MAE}_{\mathrm{model}}.
$$

- $\Delta_{\mathrm{pers}}>0$：模型优于 Persistence；
- $\Delta_{\mathrm{pers}}\le0$：禁止写“提高单步预测精度”；
- 同时报告 mean ± std 和每个电池结果；
- 不得只报告十次中的最佳值。

## 5. 当前本地结果审计

当前 `outputs/final_10runs` 中已有 10 次结果，但其 CALCE 配置为：

- `state_supervision = coordinate`；
- `direction_loss_weight = 0.05`；
- PatchFormer 起点协议；

这与当前文档中拟定的 reversible、curve-supervision 主模型并不一致。因此以下结果只能作为**现状审计**，不能直接当作最终论文主结果。

### 5.1 CALCE 当前结果

| Battery | MAE mean ± SD (Ah) | RMSE mean ± SD (Ah) | $R^2$ mean | Persistence MAE (Ah) | 相对 Persistence |
|---|---:|---:|---:|---:|---:|
| CS2_35 | 0.004817 ± 0.000035 | 0.012197 ± 0.000040 | 0.996588 | 0.004578 | -5.23% |
| CS2_36 | 0.005284 ± 0.000042 | 0.010896 ± 0.000084 | 0.998197 | 0.005016 | -5.34% |
| CS2_37 | 0.004748 ± 0.000042 | 0.009241 ± 0.000005 | 0.997989 | 0.004483 | -5.93% |
| CS2_38 | 0.004735 ± 0.000082 | 0.010277 ± 0.000150 | 0.997406 | 0.004449 | -6.42% |

负号表示模型 MAE 高于 Persistence。当前证据说明：

- 模型预测曲线整体拟合度高；
- 十次随机种子的波动较小；
- 但单步 MAE 没有超过 Persistence；
- 高 $R^2$ 不能单独作为优越性证据；
- CS2_36 的平均 Ecycle 为 124，说明阈值指标存在明显异常；
- 当前不能以“单步精度显著提升”作为 CALCE 主贡献。

### 5.2 TJU 当前结果

| Battery | MAE mean ± SD (Ah) | RMSE mean ± SD (Ah) | $R^2$ mean | Persistence MAE (Ah) | 相对 Persistence 改善 |
|---|---:|---:|---:|---:|---:|
| CY25_1 | 0.000829 ± 0.000012 | 0.001259 ± 0.000008 | 0.999944 | 0.001168 | 28.98% |
| CY25_2 | 0.000982 ± 0.000012 | 0.001468 ± 0.000004 | 0.999927 | 0.001293 | 24.09% |
| CY25_3 | 0.000803 ± 0.000008 | 0.001292 ± 0.000010 | 0.999943 | 0.001122 | 28.42% |

TJU 当前结果稳定优于 Persistence，但必须注意：

- TJU 是宏观代理状态分支；
- 本地数据经过全局异常清洗和重新编号；
- 数据版本与部分文献不一致；
- 不能用 TJU 精度反向证明 CALCE 五维物理状态的有效性。

## 6. 多步证据链

单步预测不足以证明状态转移能力，因此多步实验是本文的关键证据。

### 6.1 多步定义

给定最后已知状态 $Z_k$，递归预测：

$$
\widehat Z_{k+h}
=
F_\theta(
\widehat Z_{k+h-L:k+h-1}
),
$$

并解码：

$$
\widehat Q_{k+h}
=
G_{\mathrm{phys}}(\widehat Z_{k+h}).
$$

必须明确：

- rollout 过程中不能使用真实未来曲线；
- 不能在每一步重新用真实目标循环反演状态；
- 多步 Persistence 对整个未来区间保持 $Q_k$；
- 线性趋势基线在起点冻结斜率；
- 每个方法使用相同起点和 horizon。

### 6.2 推荐 horizon

至少报告：

$$
h\in\{1,4,8,16,32,64\}.
$$

如果完整寿命允许，可增加 100 或直到 EOL。

### 6.3 多步图表

需要：

1. MAE–horizon 曲线；
2. RMSE–horizon 曲线；
3. 每块电池的代表性 rollout；
4. 不同寿命阶段起点的 rollout；
5. 状态越界率；
6. 误差增长率；
7. 与 Persistence 和线性趋势的交叉点。

### 6.4 多步结论门控

- 若模型只在 $h=1$ 有优势，不能宣称学习了稳定退化动力学；
- 若模型在中长 horizon 优于 Persistence，即使单步略差，也可以将贡献转向状态转移与长期预测；
- 若多步同样不优于简单基线，则预测贡献必须降级，重点只能放在物理闭合与可审计性；
- 如果物理闭合好但预测无优势，应明确把文章定位为可解释/可审计框架，而非 SOTA 预测模型。

---

# 六、消融实验证据链

## 1. 消融原则

每次消融只改变一个因素，并保持：

- 相同训练/验证/测试电池；
- 相同测试起点；
- 相同异常处理；
- 相同种子；
- 相同早停规则；
- 相同训练预算；
- 相同指标；
- 至少 5 次，主消融建议 10 次。

不能拿不同数据协议下的历史配置当作单因素消融。

## 2. 贡献一：五维有效状态反演

### 2.1 消融项

| ID | 变化 | 验证问题 |
|---|---|---|
| A1 | 去掉 $dV/dQ$ 残差 | 曲线形状信息是否有贡献 |
| A2 | 去掉截止点强调 | 截止容量是否失去约束 |
| A3 | 去掉内阻弱观测 | 阻抗状态是否更不稳定 |
| A4 | 去掉相邻循环弱连续性 | 状态是否出现跳变 |
| A5 | 普通 L2 替代 soft-L1 | 鲁棒损失是否必要 |
| A6 | 单初值替代多初值 | 局部最优是否显著 |
| A7 | 去掉一个状态维度 | 五维状态的最小性 |
| A8 | 恢复无观测支撑的第六维 | 额外自由度是否形成容量捷径 |

### 2.2 指标

- 电压曲线 RMSE；
- $dV/dQ$ RMSE；
- 截止容量误差；
- closure MAE；
- 参数边界命中率；
- 多初值状态变异系数；
- 下游单步和多步误差。

### 2.3 结论门槛

只有当完整五维反演同时改善曲线重建、闭合或下游预测，才能写“该设计是必要的”。如果某个残差项对所有指标没有影响，应删除或降级为实现细节。

## 3. 贡献二：多尺度状态表示

### 3.1 消融项

| ID | 输入 | 目的 |
|---|---|---|
| B0 | $Z$、$\Delta Z$、MA8、MA32 | 完整模型 |
| B1 | 仅 $Z$ | 检验是否只需当前状态水平 |
| B2 | $Z+\Delta Z$ | 检验瞬时变化的作用 |
| B3 | 去掉 MA8 | 检验短期趋势 |
| B4 | 去掉 MA32 | 检验长期趋势 |
| B5 | 去掉 MA8 和 MA32 | 检验多尺度趋势整体作用 |
| B6 | 序列长度 16/32/64/128 | 历史长度敏感性 |

### 3.2 主要证据

多尺度趋势的价值不应只看单步 MAE，更应看：

- 多步误差增长；
- 局部恢复预测；
- 晚寿命误差；
- 跨电池方差；
- 状态预测稳定性。

## 4. 贡献三：有界 reversible 状态转移

### 4.1 消融项

| ID | 转移形式 |
|---|---|
| C0 | 有界双向乘性更新，方向损失为 0 |
| C1 | 逐步单调方向损失 |
| C2 | 无界加性更新 |
| C3 | 有界加性更新 |
| C4 | 乘性更新但最后一层非零初始化 |
| C5 | 直接预测下一绝对状态 |

### 4.2 评价

- 单步和多步误差；
- 状态越界率；
- 局部恢复片段误差；
- 长期趋势方向；
- 预测状态的单步相对变化分布；
- EOL 误差。

### 4.3 结论门槛

只有在局部恢复片段和总体预测均有改善时，才能称 reversible 设计更合理。如果只改善局部片段但总体退化，应写成权衡；如果仅仅方向损失为零表现更好，应避免将其包装成新的物理定律。

## 5. 贡献四：直接物理容量解码

这是最重要的结构消融。

### 5.1 对比项

| ID | 输出方式 |
|---|---|
| D0 | $G_{\mathrm{phys}}(\widehat Z_{k+1})$ |
| D1 | 学习型绝对容量头 |
| D2 | $Q_k+\Delta \widehat Q_{k+1}$ |
| D3 | 仅使用最后容量的 Persistence |
| D4 | 状态＋自由残差校准器 |

### 5.2 评价

- 单步 MAE/RMSE；
- 多步误差；
- closure MAE；
- 容量输出对状态扰动的响应；
- 输出对 $Q_k$ 的显式依赖；
- 预测曲线与容量的一致性；
- 参数量和推理时间。

### 5.3 可能出现的结果及解释

- **物理解码精度更高且闭合更好**：支持核心贡献；
- **精度相当但闭合更好**：支持“在不牺牲精度的情况下增加可审计性”；
- **精度略低但多步更稳定**：支持长期稳定性权衡；
- **精度和多步均更差**：不能将物理解码表述为性能贡献，只能作为结构约束；
- **自由残差头显著改善精度**：需讨论残差是否破坏闭合，而不能选择性忽略。

## 6. 贡献五：监督信号

### 6.1 消融项

- 去掉曲线损失；
- 去掉弱状态损失；
- 去掉容量损失；
- 去掉晚寿命权重；
- 不同曲线损失权重；
- 不同状态损失权重；
- 不同容量损失权重；
- 方向损失为 0 和非零。

### 6.2 注意

当前部分日志中 `curve=0.000000`，需要确认：

- 是数值非常小被格式化为零；
- 还是该配置实际没有产生有效曲线梯度；
- 或 `coordinate supervision` 与拟定的 `curve supervision` 不一致。

在该问题澄清前，不能将曲线监督写成已经被主结果验证的贡献。

---

# 七、效率分析实验证据链

## 1. 效率主张的边界

本文包含：

1. 离线曲线反演；
2. 状态缓存；
3. GRU 状态转移；
4. 物理容量网格求解。

因此只报告 GRU 参数量不足以证明端到端高效。

## 2. 应报告的成本

| 阶段 | 指标 | 单位 |
|---|---|---|
| 曲线读取与对齐 | 总耗时、每循环耗时 | s、ms/cycle |
| 状态反演 | 总耗时、中位数、P95 | s、ms/cycle |
| 状态缓存 | 文件大小和生成时间 | MB、min |
| 时序训练 | 每 epoch、总训练时间 | s、min |
| 状态预测 | batch=1 和 batch=N 延迟 | ms/sample |
| 物理解码 | 单状态容量求解时间 | ms/sample |
| 端到端推理 | 从可用输入到容量输出 | ms/sample |
| 模型规模 | 可训练参数、checkpoint 大小 | K/M、MB |
| 资源 | CPU、GPU、峰值显存/内存 | 型号、GB |

## 3. 公平对比

所有方法需要：

- 相同硬件；
- 相同 batch size；
- 预热后计时；
- 重复至少 100 次推理；
- 报告中位数和 P95；
- 明确是否包含预处理；
- GPU 计时前后同步；
- 区分离线一次性成本与在线每步成本。

## 4. 推荐效率表

| Method | Input preprocessing | Params | Train time | Offline inversion | Online latency | Peak memory | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|

## 5. 可形成的结论

- 如果 GRU 很小但反演耗时较高：写“紧凑的时序转移网络”，不要写“端到端轻量”；
- 如果反演可离线缓存且在线延迟低：可写“适合离线状态构建后的快速在线预测”；
- 如果端到端成本低于对比模型：可写“在统一硬件下具有更低推理开销”；
- 参数量不是物理模型成本的完整替代指标。

---

# 八、理论、物理一致性与解释性分析证据链

> 建议章节名称使用 `Physical consistency and interpretability analysis`，不要使用“理论证明”。当前模型没有给出泛化界、收敛定理或结构可辨识性定理。

## 1. 曲线重建证据

### 1.1 目的

验证反演状态是否能够通过同一物理前向模型重建观测电压，而不是只拟合容量标签。

### 1.2 图表

- 每块电池选择早期、中期、晚期各 2–3 个循环；
- 真实与重建 $V(Q)$；
- 残差 $V_{\mathrm{pred}}-V_{\mathrm{obs}}$；
- 真实与重建 $dV/dQ$；
- 截止点局部放大；
- 全寿命曲线 RMSE 热图。

### 1.3 统计指标

- 电压 MAE/RMSE；
- DVA MAE/RMSE；
- 截止点电压误差；
- 截止容量误差；
- 不同寿命阶段的误差；
- 各电池分布和 P95。

### 1.4 结论门槛

只在代表曲线上表现好不够，必须同时给出全循环分布。曲线重建好说明状态对当前简化前向模型有效，但不证明参数唯一。

## 2. 容量闭合证据

定义：

$$
\widehat Q_k^{\mathrm{closure}}
=
G_{\mathrm{phys}}(Z_k),
$$

$$
\epsilon_k^{\mathrm{closure}}
=
\left|
\widehat Q_k^{\mathrm{closure}}-Q_k
\right|.
$$

当前日志中的反演闭合 MAE 为：

| Battery | Closure MAE |
|---|---:|
| CS2_35 | 0.000139 Ah |
| CS2_36 | 0.000092 Ah |
| CS2_37 | 0.000061 Ah |
| CS2_38 | 0.000107 Ah |

这些数值说明反演状态能够在当前求解器下高精度闭合容量，但仍需补充：

- 全循环分布；
- 最大值和 P95；
- 闭合误差随寿命变化；
- 不同 OCP 先验下的变化；
- 去掉截止点约束后的变化。

允许的结论是：

> 反演状态与当前物理容量求解器具有较强的内部闭合一致性。

不允许的结论是：

> 五个参数已被唯一、真实地辨识。

## 3. 五维状态轨迹

分别展示：

- $Q_{\mathrm{Li}}$；
- $C_n$；
- $C_p$；
- $R_0$；
- $R_p$。

每个状态分析：

1. 全寿命长期趋势；
2. 局部恢复和平台；
3. 不同电池的一致性；
4. 与容量变化的同步或滞后；
5. 边界命中；
6. 异常跳变；
7. 预测状态与反演目标状态的差异。

不要把“总体下降/上升”直接等同于特定副反应机理。

## 4. 状态—容量敏感性

定义局部灵敏度：

$$
S_i(Z)
=
\frac{\partial G_{\mathrm{phys}}(Z)}
{\partial z_i}.
$$

也可用相对无量纲灵敏度：

$$
\widetilde S_i
=
\frac{z_i}{G_{\mathrm{phys}}(Z)}
\frac{\partial G_{\mathrm{phys}}(Z)}
{\partial z_i}.
$$

推荐使用中心有限差分复核自动微分：

$$
S_i
\approx
\frac{
G_{\mathrm{phys}}(Z+\delta_i e_i)
-
G_{\mathrm{phys}}(Z-\delta_i e_i)
}
{2\delta_i}.
$$

分析：

- 各状态影响方向；
- 早、中、晚寿命的敏感性；
- 电池间差异；
- 灵敏度是否接近零；
- 两个状态是否产生相似响应；
- 截止电压附近的非线性放大。

如果多个状态的敏感性高度共线，应将其作为可辨识性限制报告。

## 5. 状态相关性和可辨识性

### 5.1 必做分析

- Pearson 和 Spearman 相关矩阵；
- Jacobian/Fisher 信息矩阵的条件数；
- 多初值反演结果分布；
- 对电压噪声的扰动实验；
- 对 OCP 先验的敏感性；
- 参数 profile likelihood 或 bootstrap；
- 状态边界命中率。

### 5.2 需要回答

- 是否存在 $Q_{\mathrm{Li}}$、$C_n$、$C_p$ 的互相补偿；
- $R_0$ 和 $R_p$ 是否能由当前放电工况区分；
- 哪些状态相对稳定可辨识；
- 哪些状态只能作为组合有效量；
- 不同初值是否收敛到相似曲线但不同参数。

### 5.3 结论

如果状态相关性较高，应主动将论文表述限定为“effective state representation”，这不会否定模型价值，反而能避免将预测状态夸大成真实材料参数。

## 6. Reversible 局部动力学

### 6.1 片段选择

使用预先定义规则选择：

- $\Delta Q_k>0$ 的局部容量恢复；
- 长平台；
- 容量突降后回归；
- 反演状态发生明显局部回调的循环。

禁止只手工挑选对本文模型最有利的案例。应报告全部满足规则的片段统计。

### 6.2 对比

- direction loss = 0；
- direction loss > 0；
- 硬单调投影；
- 无状态约束转移。

### 6.3 指标

- 恢复片段 MAE；
- 非恢复片段 MAE；
- 状态方向命中率；
- 多步误差；
- 全局趋势相关性。

## 7. 误差和失败案例

### 7.1 必做统计

- 最大绝对误差；
- P95 和 P99 绝对误差；
- 前 1% 样本对 MSE 的贡献；
- 早、中、晚寿命分段误差；
- 容量跳变前后误差；
- EOL 阈值邻域误差；
- 每个电池的最差片段。

### 7.2 CS2_36 专项分析

当前 CS2_36 平均 Ecycle 误差约为 124，而其容量 MAE 和 $R^2$ 并未同步恶化。这说明：

- Ecycle 对 0.77 Ah 附近平台非常敏感；
- 亚毫安时量级偏差可能使首次越阈循环提前或推后很多；
- Ecycle 必须与完整容量曲线、阈值邻域图和 MAE/RMSE 共同解释；
- 不能依据单一 Ecycle 对模型整体做结论。

## 8. 不确定性分析

当前模型尚未显式输出预测区间。建议至少补充：

- 十次随机种子的预测带；
- 状态反演 bootstrap；
- OCP 先验扰动；
- 多步预测随 horizon 的方差；
- calibration 或 coverage，如增加概率输出。

如果不做完整概率建模，至少要避免将点预测写成确定性退化轨迹。

---

# 九、结果到结论的门控规则

## 1. 可以形成强结论的条件

只有满足相应条件后才能使用以下措辞：

| 结论 | 必要条件 |
|---|---|
| “提高预测精度” | 在冻结协议和重复实验下优于主要基线 |
| “优于 Persistence” | 每块或预先定义的汇总指标 $\Delta_{\mathrm{pers}}>0$，并报告不确定性 |
| “改善多步稳定性” | 多个 horizon、多个起点和多个电池上误差增长更慢 |
| “形成物理闭环” | 曲线重建和容量闭合均通过 |
| “状态具有物理意义” | 方程作用、敏感性、稳定性和边界均被分析 |
| “reversible 更合理” | 局部恢复和总体结果共同支持 |
| “轻量高效” | 统一硬件、端到端计时和参数量均有证据 |
| “跨数据集泛化” | 数据版本、化学体系和无目标微调条件明确 |

## 2. 只能形成有限结论的情形

- 单步不优于 Persistence，但多步更好：强调长期状态转移，不强调单步 SOTA；
- 预测不优于基线，但闭合显著：定位为可审计物理状态框架；
- TJU 表现好但数据版本不同：称“在当前处理数据上的适配结果”；
- 状态高度相关：称“有效状态组合”，不称唯一物理参数；
- OCP 先验敏感：将物理解释限定在当前先验条件下；
- 网络参数少但反演慢：称紧凑时序模型，不称端到端轻量。

## 3. 禁止形成的结论

- 首次提出物理引导电池预测；
- 精确揭示内部老化机理；
- 唯一辨识真实锂库存和电极容量；
- 电池老化在物理上可逆；
- 完全不使用历史容量信息；
- 在所有数据集和工况下普适；
- 仅凭 $R^2$ 很高宣称模型优于基线；
- 用 TJU 代理状态证明 CALCE 五维状态；
- 将不同预处理、起点和 EOL 阈值下的文献结果直接排名。

---

# 十、推荐的最终图表与实验清单

## 1. 主文图

1. **Fig. 1**：问题与总体框架；
2. **Fig. 2**：曲线反演和五维状态定义；
3. **Fig. 3**：状态转移和物理容量解码；
4. **Fig. 4**：CALCE 单步完整寿命预测；
5. **Fig. 5**：多步误差—horizon 与代表性 rollout；
6. **Fig. 6**：曲线重建与容量闭合；
7. **Fig. 7**：五维状态轨迹与状态—容量敏感性；
8. **Fig. 8**：reversible 局部恢复与失败案例。

## 2. 主文表

1. **Table 1**：数据集和协议；
2. **Table 2**：单步主结果＋Persistence；
3. **Table 3**：多步结果；
4. **Table 4**：关键结构消融；
5. **Table 5**：效率和端到端成本；
6. **Table 6**：CALCE 与 TJU 证据边界或扩展结果。

## 3. 补充材料

- 全部十次运行；
- 全部电池预测曲线；
- 全部消融；
- 参数边界；
- 多初值结果；
- 相关矩阵；
- OCP 敏感性；
- 全部失败案例；
- 数据对齐审计；
- 配置文件与软件环境；
- 文献对比条件表。

---

# 十一、文献证据表

## 1. 核心文献及其用途

| ID | 文献 | 类别 | 本文使用位置 | 支撑等级 |
|---|---|---|---|---|
| R1 | Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy, 2019 | 数据驱动 | 早期曲线特征可预测寿命 | 强背景支持 |
| R2 | Roman et al., *Machine learning pipeline for battery state-of-health estimation*, Nature Machine Intelligence, 2021 | 数据驱动 | 工程特征、SOH 和置信区间 | 强背景支持 |
| R3 | *Deep learning to estimate lithium-ion battery state of health without additional degradation experiments*, Nature Communications, 2023 | 数据驱动/域适配 | 跨制造商和目标标签稀缺 | 强背景支持 |
| R4 | *A Data-Driven Approach With Uncertainty Quantification for Predicting Future Capacities and Remaining Useful Life of Lithium-ion Battery*, IEEE TIE, 2020 | 数据驱动 | 递归多步和不确定性 | 强背景支持 |
| R5 | *PatchFormer: A novel patch-based transformer for accurate remaining useful life prediction of lithium-ion batteries*, Journal of Power Sources, 2025 | 数据驱动 | 多尺度特征和容量恢复 | 强背景支持 |
| R6 | *Early prediction of lithium-ion battery degradation with a generative pre-trained transformer*, Nature Communications, 2025 | 数据驱动 | 生成式全寿命轨迹 | 背景支持 |
| R7 | *Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis*, Nature Communications, 2024 | 物理引导 | 经验退化和状态空间约束 | 强对比支持 |
| R8 | Hofmann et al., *Physics-Informed Neural Networks for State of Health Estimation in Lithium-Ion Batteries*, Journal of The Electrochemical Society, 2023 | 物理引导 | PINN 与电化学先验 | 强对比支持 |
| R9 | *Domain knowledge-guided machine learning framework for state of health estimation in lithium-ion batteries*, Communications Engineering, 2024 | 领域知识引导 | 可在线提取的物理健康指标 | 强对比支持 |
| R10 | *Physics-Informed Neural Networks for Prognostics and Health Management of Lithium-Ion Batteries*, IEEE TIV, 2023 | 物理引导 | PINN-PHM 方法背景 | 背景支持 |
| R11 | *Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning*, Nature Communications, 2020 | 诊断/可辨识性 | EIS 信息量与 ECM 非唯一拟合风险 | 强支持 |
| R12 | *Differential voltage curve analysis of a lithium-ion battery during discharge*, Journal of Power Sources, 2018 | DVA | $dV/dQ$ 的退化诊断价值 | 强支持 |
| R13 | *Peak-tracking method to quantify degradation modes in lithium-ion batteries via differential voltage and incremental capacity*, Journal of Energy Storage, 2021 | DVA/ICA | LLI/LAM 与微分曲线 | 强支持 |
| R14 | *Identifiability study of lithium-ion battery capacity fade using degradation mode sensitivity...*, Journal of Power Sources, 2024 | 可辨识性 | 状态敏感性和参数相关 | 强支持 |
| R15 | *The importance of degradation mode analysis in parameterising lifetime prediction models...*, Nature Communications, 2025 | 退化模式 | 参数化必须接受退化模式验证 | 强支持 |
| R16 | *Data-Driven Battery Characterization and Prognosis: Recent Progress, Challenges, and Prospects*, Small Methods, 2024 | 综述 | 解释性、泛化和物理学习 | 综述支持 |
| R17 | *Health prognostics for lithium-ion batteries: mechanisms, methods, and prospects*, Energy & Environmental Science, 2023 | 综述 | 数据驱动与物理模型权衡 | 综述支持 |
| R18 | *Towards practical data-driven battery state of health estimation: Advancements and insights targeting real-world data*, Journal of Energy Chemistry, 2025 | 综述 | 实验室到真实工况的泛化问题 | 综述支持 |
| R19 | Bhatt et al., *Machine learning-based approach for useful capacity prediction of second-life batteries...*, IJER, 2021 | 数据驱动/基线 | 电池容量预测与 Persistence 基线 | 直接方法支持 |
| R20 | Huotari et al., *A Dynamic Battery State-of-Health Forecasting Model for Electric Trucks*, ASME IMECE, 2021 | 统计预测/基线 | naïve Persistence、ARIMA 与 bagging | 直接方法支持 |
| R21 | *A physics-informed deep learning framework for RUL prediction...*, Energy, 2026 | 物理引导 | MSTEA-Net 和 triple-composite loss | 强对比支持 |
| R22 | Wang and Wang, *The Application of BiGRU-MSTA...*, Batteries, 2025 | 数据驱动 | CALCE 多尺度时序注意力真实对比模型 | 强对比支持 |
| R23 | *A Novel Hybrid Physics-Based and Data-Driven Approach for Degradation Trajectory Prediction...*, IEEE TTE, 2023 | 混合轨迹预测 | 有限数据、物理特征和未来轨迹 | 任务直接支持 |
| R24 | *Prediction of Li-ion battery capacity degradation considering polarization recovery...* | 容量恢复预测 | 恢复现象影响容量预测 | 任务直接支持 |
| R25 | *TFDM-CR: Time–frequency diffusion modeling... incorporating regeneration phenomena*, Energy and AI, 2026 | 多步容量预测 | 噪声、局部突变、长期趋势 | 任务直接支持 |
| R26 | *A Two-State-Based Hybrid Model... with Capacity Recovery*, Batteries, 2023 | 容量恢复预测 | 全局退化区和恢复区分开建模 | 任务直接支持 |
| R27 | *Forecasting battery degradation trajectory under domain shift with domain generalization*, 2024 | 域泛化轨迹预测 | 工况差异作为 domain shift | 任务直接支持 |
| R28 | *Early-stage degradation trajectory prediction... across diverse operational conditions*, JPS, 2024 | 跨工况轨迹预测 | 数据不足和跨条件泛化 | 任务直接支持 |
| R29 | *Forecasting battery capacity and power degradation with multi-task learning*, ESM, 2022 | 多任务退化预测 | 制造差异和耦合非线性老化 | 任务直接支持 |
| R30 | *Degradation path prediction... under dynamic operating sequences*, EES, 2025 | 动态工况轨迹预测 | 多样使用场景下未来路径 | 任务直接支持 |
| R31 | *Lithium-ion Battery Capacity Prediction via Conditional Recurrent GAN...*, 2025 | 数据增强容量预测 | 完整寿命样本有限 | 任务直接支持，预印本 |
| R32 | *Predicting capacity degradation trajectory... under limited data conditions*, 2024 | 有限数据轨迹预测 | 少量完整退化数据 | 任务直接支持 |

## 2. 参考文献与链接

[R1] K. A. Severson et al. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy* 4, 383–391 (2019). [https://doi.org/10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)

[R2] D. Roman et al. Machine learning pipeline for battery state-of-health estimation. *Nature Machine Intelligence* 3, 447–456 (2021). [https://doi.org/10.1038/s42256-021-00312-3](https://doi.org/10.1038/s42256-021-00312-3)

[R3] Deep learning to estimate lithium-ion battery state of health without additional degradation experiments. *Nature Communications* (2023). [https://doi.org/10.1038/s41467-023-38458-w](https://doi.org/10.1038/s41467-023-38458-w)

[R4] A Data-Driven Approach With Uncertainty Quantification for Predicting Future Capacities and Remaining Useful Life of Lithium-ion Battery. *IEEE Transactions on Industrial Electronics* (2020). [https://doi.org/10.1109/TIE.2020.2973876](https://doi.org/10.1109/TIE.2020.2973876)

[R5] PatchFormer: A novel patch-based transformer for accurate remaining useful life prediction of lithium-ion batteries. *Journal of Power Sources* (2025). [https://doi.org/10.1016/j.jpowsour.2025.236187](https://doi.org/10.1016/j.jpowsour.2025.236187)

[R6] Early prediction of lithium-ion battery degradation with a generative pre-trained transformer. *Nature Communications* (2025). [https://www.nature.com/articles/s41467-025-66819-0](https://www.nature.com/articles/s41467-025-66819-0)

[R7] Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis. *Nature Communications* (2024). [https://doi.org/10.1038/s41467-024-48779-z](https://doi.org/10.1038/s41467-024-48779-z)

[R8] T. Hofmann et al. Physics-Informed Neural Networks for State of Health Estimation in Lithium-Ion Batteries. *Journal of The Electrochemical Society* 170, 090524 (2023). [https://doi.org/10.1149/1945-7111/acf0ef](https://doi.org/10.1149/1945-7111/acf0ef)

[R9] A. Lanubile et al. Domain knowledge-guided machine learning framework for state of health estimation in lithium-ion batteries. *Communications Engineering* 3, 168 (2024). [https://www.nature.com/articles/s44172-024-00304-2](https://www.nature.com/articles/s44172-024-00304-2)

[R10] Physics-Informed Neural Networks for Prognostics and Health Management of Lithium-Ion Batteries. *IEEE Transactions on Intelligent Vehicles* (2023). [https://doi.org/10.1109/TIV.2023.3315548](https://doi.org/10.1109/TIV.2023.3315548)

[R11] Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning. *Nature Communications* (2020). [https://doi.org/10.1038/s41467-020-15235-7](https://doi.org/10.1038/s41467-020-15235-7)

[R12] Differential voltage curve analysis of a lithium-ion battery during discharge. *Journal of Power Sources* (2018). [https://doi.org/10.1016/j.jpowsour.2018.07.043](https://doi.org/10.1016/j.jpowsour.2018.07.043)

[R13] Peak-tracking method to quantify degradation modes in lithium-ion batteries via differential voltage and incremental capacity. *Journal of Energy Storage* (2021). [ScienceDirect article](https://www.sciencedirect.com/science/article/pii/S2352152X2101344X)

[R14] Identifiability study of lithium-ion battery capacity fade using degradation mode sensitivity for a minimally and intuitively parametrized electrode-specific cell open-circuit voltage model. *Journal of Power Sources* (2024). [ScienceDirect article](https://www.sciencedirect.com/science/article/pii/S0378775324003975)

[R15] The importance of degradation mode analysis in parameterising lifetime prediction models of lithium-ion battery degradation. *Nature Communications* (2025). [https://www.nature.com/articles/s41467-025-57968-3](https://www.nature.com/articles/s41467-025-57968-3)

[R16] Data-Driven Battery Characterization and Prognosis: Recent Progress, Challenges, and Prospects. *Small Methods* (2024). [https://doi.org/10.1002/smtd.202301021](https://doi.org/10.1002/smtd.202301021)

[R17] Health prognostics for lithium-ion batteries: mechanisms, methods, and prospects. *Energy & Environmental Science* (2023). [https://doi.org/10.1039/D2EE03019E](https://doi.org/10.1039/D2EE03019E)

[R18] Towards practical data-driven battery state of health estimation: Advancements and insights targeting real-world data. *Journal of Energy Chemistry* 110, 657–680 (2025). [https://doi.org/10.1016/j.jechem.2025.07.022](https://doi.org/10.1016/j.jechem.2025.07.022)

[R19] A. Bhatt, W. Ongsakul, N. M. Manjiparambil & J. G. Singh. Machine learning-based approach for useful capacity prediction of second-life batteries employing appropriate input selection. *International Journal of Energy Research* 45, 21023–21049 (2021). [https://doi.org/10.1002/er.7160](https://doi.org/10.1002/er.7160)

[R20] M. Huotari, S. Arora, A. Malhi & K. Främling. A Dynamic Battery State-of-Health Forecasting Model for Electric Trucks: Li-Ion Batteries Case-Study. *ASME 2020 International Mechanical Engineering Congress and Exposition*, V008T08A021 (2021). [https://doi.org/10.1115/IMECE2020-23949](https://doi.org/10.1115/IMECE2020-23949)

[R21] A physics-informed deep learning framework for remaining useful life prediction of lithium-ion batteries with feature subset construction. *Energy* 346, 140288 (2026). [https://doi.org/10.1016/j.energy.2026.140288](https://doi.org/10.1016/j.energy.2026.140288)

[R22] L. Wang & S. Wang. The Application of BiGRU-MSTA Based on Multi-Scale Temporal Attention Mechanism in Predicting the Remaining Life of Lithium-Ion Batteries. *Batteries* 11, 223 (2025). [https://doi.org/10.3390/batteries11060223](https://doi.org/10.3390/batteries11060223)

[R23] A Novel Hybrid Physics-Based and Data-Driven Approach for Degradation Trajectory Prediction in Li-Ion Batteries. *IEEE Transactions on Transportation Electrification* 9, 2628–2644 (2023). [https://doi.org/10.1109/TTE.2022.3212024](https://doi.org/10.1109/TTE.2022.3212024)

[R24] Prediction of Li-ion battery capacity degradation considering polarization recovery with a hybrid ensemble learning model. [Publisher record](https://www.sciencedirect.com/science/article/pii/S2405829722002732)

[R25] TFDM-CR: Time–frequency diffusion modeling for lithium-ion battery capacity prediction incorporating regeneration phenomena. *Energy and AI* 24, 100703 (2026). [https://doi.org/10.1016/j.egyai.2026.100703](https://doi.org/10.1016/j.egyai.2026.100703)

[R26] A Two-State-Based Hybrid Model for Degradation and Capacity Prediction of Lithium-Ion Batteries with Capacity Recovery. *Batteries* 9, 596 (2023). [https://doi.org/10.3390/batteries9120596](https://doi.org/10.3390/batteries9120596)

[R27] Forecasting battery degradation trajectory under domain shift with domain generalization. (2024). [Publisher record](https://www.sciencedirect.com/science/article/pii/S2405829724005518)

[R28] Early-stage degradation trajectory prediction for lithium-ion batteries: A generalized method across diverse operational conditions. *Journal of Power Sources* (2024). [Publisher record](https://www.sciencedirect.com/science/article/pii/S0378775324007602)

[R29] Forecasting battery capacity and power degradation with multi-task learning. *Energy Storage Materials* 53, 453–466 (2022). [Preprint record](https://arxiv.org/abs/2111.14937)

[R30] Degradation path prediction of lithium-ion batteries under dynamic operating sequences. *Energy & Environmental Science* (2025). [https://doi.org/10.1039/D4EE04787G](https://doi.org/10.1039/D4EE04787G)

[R31] M. A. Chowdhury, G. Modekwe & Q. Lu. Lithium-ion Battery Capacity Prediction via Conditional Recurrent Generative Adversarial Network-based Time-Series Regeneration. (2025). [https://arxiv.org/abs/2503.12258](https://arxiv.org/abs/2503.12258)

[R32] Predicting capacity degradation trajectory for lithium-ion batteries under limited data conditions. (2024). [Journal record](https://esst.cip.com.cn/EN/abstract/abstract2819.shtml)

---

# 十二、正式写作前必须完成的事项

## 1. 版本冻结

- [ ] 冻结 CALCE 主配置；
- [ ] 确认 `state_supervision`；
- [ ] 确认 `direction_loss_weight`；
- [ ] 冻结预测起点；
- [ ] 冻结异常值处理；
- [ ] 冻结状态缓存；
- [ ] 明确主协议和补充协议；
- [ ] 确认主结果目录。

## 2. 实验补全

- [ ] CALCE 最终版本十次重复实验；
- [ ] Persistence 和线性趋势；
- [ ] 多步 rollout；
- [ ] 关键结构消融；
- [ ] 物理解码消融；
- [ ] reversible 对比；
- [ ] 曲线重建分布；
- [ ] 闭合误差分布；
- [ ] 状态敏感性；
- [ ] 可辨识性和相关性；
- [ ] 失败案例；
- [ ] 端到端效率。

## 3. 结论冻结

最终摘要、贡献和标题必须在上述结果完成后确定。当前最稳妥的论文定位是：

> 一个以曲线反演、有效状态转移和物理容量解码构成的可审计跨电池容量预测框架。

当前不适合预设的定位是：

> 一个在所有数据集上显著优于现有方法的高精度通用预测模型。
