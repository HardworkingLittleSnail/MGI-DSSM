## 3. 方法

本节给出宏观观测引导的深度状态空间模型（MGI-DSSM），聚焦于由最近 $L$ 个已完成循环预测下一循环容量的单步预测任务。模型将历史放电观测反演为受电化学关系约束的有效状态，在状态空间中推演下一循环的退化状态，并依据给定放电边界下的截止事件确定目标容量。容量由此成为潜在状态演化在可观测空间中的直接映射。下文依次介绍总体架构、状态反演、状态演化、容量解码与训练目标。

### 3.1 总体架构

MGI-DSSM 围绕历史观测、有效状态与目标容量组织单步预测过程。记第 $k$ 个循环的放电观测为

$$
\mathcal O_k=
\left\{
\mathcal C_k,\boldsymbol\pi_k
\right\},
\qquad
\mathcal C_k=
\left\{
(q_{k,j},V_{k,j})
\right\}_{j=1}^{N_k},
$$

其中，$q_{k,j}$ 和 $V_{k,j}$ 分别表示容量坐标及其对应的端电压；$\boldsymbol\pi_k$ 汇集预测时已知的放电电流、截止电压、极化时间常数和与电池化学体系匹配的电极开路电压先验。该表示将采样频率、放电电流、截止条件与化学体系等数据差异统一归入观测曲线和条件变量，使各数据集共享同一状态反演、状态转移与容量解码流程。

给定截至循环 $k$ 的长度为 $L$ 的历史窗口

$$
\mathcal H_k=
\left\{
\mathcal O_{k-L+1},\ldots,\mathcal O_k
\right\},
$$

MGI-DSSM 的前向过程写为

$$
\mathcal H_k
\xrightarrow{\;\mathcal E\;}
Z_{k-L+1:k}
\xrightarrow{\;\mathcal T_\theta\;}
\widehat Z_{k+1}
\xrightarrow{\;\mathcal D\;}
\widehat Q_{k+1}.
$$

其中，$\mathcal E$ 为基于放电观测的确定性状态反演算子，$\mathcal T_\theta$ 为参数化的跨循环状态转移器，$\mathcal D$ 为固定的物理容量解码器。物理关系承担状态与外部响应之间的映射，时序网络刻画跨循环状态演化，二者通过共享状态变量连接历史观测与目标容量。有效状态定义为

$$
Z_k=
\left[
Q_{\mathrm{Li},k},
C_{n,k},
C_{p,k},
R_{0,k},
R_{p,k}
\right]^{\!\top}\in\Omega_Z,
$$

分别表示有效锂库存、负极有效容量、正极有效容量、欧姆阻抗和极化阻抗。这些变量是在有限整电池观测与简化物理关系下辨识得到的有效代理状态，其物理含义限定于本文的状态空间表征。

因此，本文的预测映射写为 $\mathcal H_k\mapsto\widehat Q_{k+1}$。历史窗口 $\mathcal H_k$ 由循环 $k$ 及其之前已经完成的观测构成；目标循环的曲线 $\mathcal C_{k+1}$、状态 $Z_{k+1}$ 和容量 $Q_{k+1}$ 用于训练监督。沿着这一时间边界，模型首先从历史放电曲线中获得后续演化所需的有效状态。

### 3.2 状态反演

宏观放电曲线记录了内部退化状态在给定运行条件下的外部响应。状态反演以电极平衡和端电压关系为观测模型，将每个已完成循环的曲线投影到统一的五维有效状态空间。由此得到的状态同时承载当前循环的电压–容量响应与跨循环退化信息，并为后续演化建模提供低维、物理一致的坐标。

#### 放电曲线与容量坐标

状态反演首先将离散放电测量组织到统一的容量坐标中。对每个循环提取其主放电阶段，并由采样时间与放电电流构造容量坐标：

$$
q_{k,j}
=
\sum_{l=2}^{j}
\frac{-I_{k,l}\left(t_{k,l}-t_{k,l-1}\right)}{3600},
$$

其中放电电流按负值记录，因而 $q_{k,j}\geq0$。曲线在电压首次达到该电池规定的截止阈值时终止，并在相邻采样点之间进行线性插值。为消除设备容量列与电流积分之间的微小累计偏差，同时保证状态反演与基准容量使用相同的物理坐标，将积分容量轴按当前已完成循环的容量 $Q_k$ 进行端点对齐：

$$
\bar q_{k,j}
=
q_{k,j}
\frac{Q_k}{q_{k,N_k}}.
$$

端点对齐由已完成循环的 $Q_k$ 确定，使积分曲线与该循环的基准容量共享同一坐标。下文仍以 $q$ 表示对齐后的容量坐标。

#### 状态条件电压关系

容量坐标确定后，状态变量通过电极化学计量与端电压建立可计算的观测关系。给定 $Z_k$，放电起点处的正负极化学计量满足

$$
C_{n,k}x_{n,k}^{0}
+
C_{p,k}x_{p,k}^{0}
=
Q_{\mathrm{Li},k}.
$$

共享参考点 $(x_n^{\mathrm{ref}},x_p^{\mathrm{ref}})=(0.90,0.35)$ 经锂平衡投影后确定每个循环的起始化学计量。令

$$
\delta_k=
Q_{\mathrm{Li},k}
-C_{n,k}x_n^{\mathrm{ref}}
-C_{p,k}x_p^{\mathrm{ref}},
$$

则

$$
x_{n,k}^{0}
=
x_n^{\mathrm{ref}}
+
\frac{\delta_k C_{n,k}}
{C_{n,k}^{2}+C_{p,k}^{2}+\epsilon},
$$

$$
x_{p,k}^{0}
=
x_p^{\mathrm{ref}}
+
\frac{\delta_k C_{p,k}}
{C_{n,k}^{2}+C_{p,k}^{2}+\epsilon},
$$

其中 $\epsilon$ 为数值稳定项。放电容量为 $q$ 时，两极化学计量分别演化为

$$
x_{n,k}(q)=x_{n,k}^{0}-\frac{q}{C_{n,k}},
\qquad
x_{p,k}(q)=x_{p,k}^{0}+\frac{q}{C_{p,k}}.
$$

由固定的负极和正极开路电压函数 $U_n(\cdot)$ 与 $U_p(\cdot)$，得到状态条件的端电压

$$
\widehat V_k(q;Z_k,\boldsymbol\pi_k)
=
U_p\!\left[x_{p,k}(q)\right]
-U_n\!\left[x_{n,k}(q)\right]
-I_kR_{0,k}
-I_kR_{p,k}
\left[
1-\exp\!\left(
-\frac{3600q}{I_k\tau_p}
\right)
\right].
$$

其中 $I_k>0$ 表示放电电流幅值；前两项描述平衡电势，后两项分别表示欧姆压降与一阶极化压降。电极开路电压先验依据电池的已知化学体系固定，模型结构与其余平衡关系保持不变。

#### 五维状态反演

上述观测关系将候选状态映射为完整的放电电压响应，据此即可从实测曲线反求有效状态。对于第 $k$ 个循环，令待优化变量为

$$
\boldsymbol\vartheta_k=
[Q_{\mathrm{Li},k},C_{n,k},C_{p,k},R_{0,k}]^{\top}.
$$

候选 $\boldsymbol\vartheta_k$ 对应的末端电压关系给出极化阻抗的条件估计：

$$
R_{p,k}(\boldsymbol\vartheta_k)
=
\operatorname{clip}_{[R_p^{-},R_p^{+}]}
\left\{
\frac{
U_{\mathrm{ocv}}(Q_k;\boldsymbol\vartheta_k)
-I_kR_{0,k}-V_{k,N_k}
}{
I_k\left[1-\exp\!\left(-3600Q_k/(I_k\tau_p)\right)\right]
}
\right\},
$$

其中 $U_{\mathrm{ocv}}=U_p-U_n$，$\operatorname{clip}$ 将解限制在预设可行区间内。于是，五维状态可写为 $Z_k(\boldsymbol\vartheta_k)=[\boldsymbol\vartheta_k^{\top},R_{p,k}(\boldsymbol\vartheta_k)]^{\top}$。这一条件估计减少了高度相关的电阻自由度，并使末端截止响应直接参与状态辨识。其余四个状态量通过同时匹配实测电压、微分电压形状与截止事件进行反演：

$$
\begin{aligned}
\boldsymbol\vartheta_k^{*}
=\arg\min_{\boldsymbol\vartheta\in\Omega_{\vartheta}}
\;&
\sum_j
\rho\!\left(
\widehat V_k(q_j;Z(\boldsymbol\vartheta))-V_{k,j}
\right)
\\
&+\lambda_{\mathrm{DVA}}
\sum_j
\rho\!\left(
\frac{\partial\widehat V_k}{\partial q}(q_j;Z(\boldsymbol\vartheta))
-
\frac{\partial V_k}{\partial q}(q_j)
\right)
\\
&+\lambda_{\mathrm{cut}}
\rho\!\left(
\widehat V_k(Q_k;Z(\boldsymbol\vartheta))-V_{k,N_k}
\right)
\\
&+\lambda_R
\rho\!\left(R_0-R_{0,k}^{\mathrm{obs}}\right)
+\lambda_T
\left\|S_T^{-1}(\boldsymbol\vartheta-\boldsymbol\vartheta_{k-1}^{*})\right\|_2^2,
\end{aligned}
$$

其中，$\rho(\cdot)$ 为鲁棒损失，$S_T$ 用于平衡不同状态量的尺度；数据提供的 $R_{0,k}^{\mathrm{obs}}$ 构成欧姆阻抗的弱观测。原始电压项约束整体响应，微分电压项增强对局部曲线形状与电极区间变化的敏感性，截止项锚定可用容量边界，相邻循环项保持状态轨迹的时间连续性。四类约束共同确定兼具观测一致性、截止一致性与跨循环稳定性的有效状态。

采用有界多初值鲁棒最小二乘求解 $\boldsymbol\vartheta_k^{*}$，并将前一循环解纳入当前循环的候选初值。目标函数最小的可行解与其条件估计的 $R_{p,k}$ 共同构成 $Z_k^{*}$。该过程沿循环顺序推进，形成用于跨循环演化建模的有效状态轨迹 $Z_{1:k}^{*}$。

### 3.3 状态演化

基于反演得到的状态轨迹，MGI-DSSM 学习跨循环转移规律，使当前退化位置、近期变化与累积趋势共同指向下一循环状态。状态演化在统一的有效状态空间中进行，状态与外部响应之间的映射继续由第3.2节的物理关系给出。

首先使用训练电池的统计量标准化状态：

$$
\widetilde Z_t
=
\frac{Z_t-\mu_Z}
{\max(\sigma_Z,10^{-6})},
$$

并计算相邻循环变化

$$
\Delta\widetilde Z_t
=
\widetilde Z_t-\widetilde Z_{t-1}.
$$

绝对状态刻画当前退化位置，一阶差分反映最近变化，两个时间尺度的历史平均进一步提取局部趋势与累积趋势。由此构造状态特征

$$
X_t^{\mathrm{state}}
=
\left[
\widetilde Z_t,
\Delta\widetilde Z_t,
\operatorname{MA}_{8}(\Delta\widetilde Z_t),
\operatorname{MA}_{32}(\Delta\widetilde Z_t)
\right]\in\mathbb R^{20},
$$

其中，$\operatorname{MA}_{w}$ 表示沿历史方向计算的长度为 $w$ 的因果移动平均。首个时间步的差分置零，序列起始处采用首值填充，所有趋势量均由当前及更早状态构造。

长度为 $L$ 的状态特征序列由门控循环单元编码：

$$
h_k
=
\operatorname{GRU}_{\theta}
\left(
X_{k-L+1:k}^{\mathrm{state}}
\right),
\qquad
r_k
=
\operatorname{Head}_{\theta}(h_k),
$$

其中，状态头由层归一化、两层线性映射和 SiLU 激活组成。其输出 $r_k$ 参数化相对于最后已知状态的有界变化：

$$
\widehat Z_{k+1}
=
Z_k\odot
\exp\!\left[
\boldsymbol\alpha\odot\tanh(r_k)
\right],
$$

$$
\boldsymbol\alpha
=
[0.02,0.02,0.02,0.03,0.03]^{\top}.
$$

因此，第 $i$ 个状态维度满足

$$
\exp(-\alpha_i)
\leq
\frac{\widehat z_{i,k+1}}{z_{i,k}}
\leq
\exp(\alpha_i),
$$

该参数化将状态正值性与单循环变化幅度编码进转移结构，并为局部非单调响应保留表达空间。状态头末层采用零初始化，使模型从 $\widehat Z_{k+1}=Z_k$ 的保持映射开始训练，随后逐步学习由历史轨迹支持的状态增量。预测得到的 $\widehat Z_{k+1}$ 随后进入容量解码过程。

### 3.4 容量解码

容量解码接收预测状态 $\widehat Z_{k+1}$，并沿用反演阶段建立的状态–电压关系。给定预测时已知的放电条件 $\boldsymbol\pi_{k+1}$，首先在容量网格 $\{q_j\}_{j=1}^{M}$ 上计算

$$
\widehat V_{k+1}(q_j)
=
\mathcal G
\left(
q_j,\widehat Z_{k+1};\boldsymbol\pi_{k+1}
\right),
$$

其中，$\mathcal G$ 为第3.2节定义的状态条件电压算子。累计最小算子进一步将数值插值引起的局部波动整理为非增电压包络，使容量对应稳定且唯一的首次截止事件：

$$
\widehat V_{k+1}^{\downarrow}(q_j)
=
\min_{l\leq j}
\widehat V_{k+1}(q_l).
$$

令 $j^{*}$ 为电压首次达到截止阈值 $V_{\mathrm{cut},k+1}$ 的网格位置，则下一循环绝对容量定义为

$$
\widehat Q_{k+1}
=
\inf
\left\{
q\geq0:
\widehat V_{k+1}^{\downarrow}(q)
\leq
V_{\mathrm{cut},k+1}
\right\}.
$$

在离散网格上，容量通过截止点两侧的相邻网格线性插值得到：

$$
\widehat Q_{k+1}
=
q_{j^{*}-1}
+
\frac{
V_{\mathrm{cut},k+1}
-\widehat V_{k+1}^{\downarrow}(q_{j^{*}-1})
}{
\widehat V_{k+1}^{\downarrow}(q_{j^{*}})
-\widehat V_{k+1}^{\downarrow}(q_{j^{*}-1})
}
\left(q_{j^{*}}-q_{j^{*}-1}\right).
$$

该固定解码器以预测状态、运行条件与截止规则共同确定下一循环绝对容量。状态变化依次传递为电压曲线变化和截止位置变化，形成“状态—曲线—容量”的闭合推断链。

### 3.5 训练目标

上述闭合推断链同时规定了模型的训练目标：预测状态延续历史轨迹的演化规律，并在统一的物理观察关系下重现下一循环的电压响应与绝对容量。训练过程依次完成状态反演与转移学习。第一阶段按照第3.2节逐循环反演并缓存状态轨迹，随后通过状态–容量闭合误差检验反演器与解码器的一致性：

$$
\varepsilon_{\mathrm{cl}}
=
\frac{1}{N}
\sum_{k=1}^{N}
\left|
\mathcal D(Z_k;\boldsymbol\pi_k)-Q_k
\right|.
$$

训练前固定的容差构成物理模块的质量控制标准。满足 $\varepsilon_{\mathrm{cl}}$ 要求的状态轨迹进入第二阶段，用于学习跨循环状态转移。

第二阶段以 $Z_{k-L+1:k}^{*}$ 为输入，联合约束预测状态、其对应的电压响应以及最终容量：

$$
\mathcal L
=
\lambda_Q\mathcal L_Q
+\lambda_V\mathcal L_V
+\lambda_Z\mathcal L_Z
+\lambda_{\mathrm{dir}}\mathcal L_{\mathrm{dir}}.
$$

容量损失采用加权 Huber 误差：

$$
\mathcal L_Q
=
\frac{1}{B}
\sum_{i=1}^{B}
w_i
\ell_{\mathrm H}
\left(
\widehat Q_{i,k+1},Q_{i,k+1}
\right),
$$

样本权重定义为

$$
w_i=w_i^{\mathrm{life}}w_i^{\mathrm{reg}},
$$

$$
w_i^{\mathrm{life}}
=1+\gamma_{\mathrm{life}}
\operatorname{clip}
\left(1-\frac{Q_{i,k+1}}{Q_{\mathrm{rated}}},0,1\right),
$$

$$
w_i^{\mathrm{reg}}
=1+\gamma_{\mathrm{reg}}
\operatorname{clip}
\left(\frac{Q_{i,k+1}-Q_{i,k}}{0.01},0,5\right).
$$

式中容量均以 Ah 计，因此 $0.01$ 对应 $0.01\,\mathrm{Ah}$ 的回升尺度。$w_i^{\mathrm{life}}$ 提高深度退化样本的贡献，$w_i^{\mathrm{reg}}$ 在启用时增强局部容量回升片段的训练权重，使不同退化阶段得到更均衡的优化。状态坐标损失为

$$
\mathcal L_Z
=
\frac{1}{B}
\sum_{i=1}^{B}
\ell_{\mathrm H}
\left[
S_Z^{-1}
\left(
\widehat Z_{i,k+1}-Z_{i,k+1}^{*}
\right)
\right],
$$

其中，$S_Z$ 平衡不同状态量纲对应的损失尺度。曲线空间监督进一步刻画各状态维度对电压响应的非均匀灵敏度：预测状态与反演目标状态经由同一电压算子生成响应，并在真实可观测容量区间内计算误差：

$$
\mathcal L_V
=
\frac{
\sum_{i,j}M_{i,j}
\ell_{\mathrm H}
\left[
\frac{
\mathcal G(q_j,\widehat Z_{i,k+1};\boldsymbol\pi_{i,k+1})
-\mathcal G(q_j,Z_{i,k+1}^{*};\boldsymbol\pi_{i,k+1})
}{s_V}
\right]
}{
\sum_{i,j}M_{i,j}
},
$$

$$
M_{i,j}=\mathbb I(q_j\leq Q_{i,k+1}),
$$

掩码 $M_{i,j}$ 将监督范围限定在截止容量之前的有效放电区间。由此，$\mathcal L_Z$ 对齐状态坐标，$\mathcal L_V$ 对齐状态对应的可观测响应。方向正则项进一步写为

$$
\mathcal L_{\mathrm{dir}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left[
\sum_{d=1}^{3}
\operatorname{ReLU}
\left(
\widehat z_{i,k+1}^{(d)}-z_{i,k}^{(d)}
\right)
+
\sum_{d=4}^{5}
\operatorname{ReLU}
\left(
z_{i,k}^{(d)}-\widehat z_{i,k+1}^{(d)}
\right)
\right].
$$

该项以软约束表达库存与有效容量的下降趋势以及阻抗的上升趋势。$\lambda_{\mathrm{dir}}$ 控制趋势先验的作用强度，并可设为零以保留局部容量恢复与观测波动。状态正值性和单步相对变化上界由第3.3节的乘性更新持续保证。

训练阶段以 $Z_{k+1}^{*}$、目标电压响应和 $Q_{k+1}$ 构成监督信号，状态转移器的输入始终来自历史窗口。推断阶段从最近 $L$ 个已完成循环的反演状态得到 $\widehat Z_{k+1}$，再由固定解码器生成 $\widehat Q_{k+1}$，由此完成基于历史信息的单步预测。
