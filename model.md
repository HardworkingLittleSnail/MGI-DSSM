# MGI-DSSM完整流程草稿：从放电曲线到状态退化动力学，再到物理容量解码

> **适用范围与配置身份。** 本文档描述的是CALCE数据集上的`mgi-physics`分支，具体对应
> `configs/paper_calce_mstea_reversible_10runs.json`所定义的论文候选配置，而不是
> `configs/final_calce_10runs.json`的统一描述。当前候选配置采用
> 64循环窗口、曲线主监督、21循环局部异常值处理、零方向损失和MSTEA评估协议。

## 0. 总体故事

MGI-DSSM要解决的问题不是“给一串历史容量，外推下一点容量”，而是：

```text
容量为什么会变？
容量变化背后的有效物理状态是什么？
这些状态如何跨循环演化？
预测出的下一状态如何重新生成可观测的端电压和截止容量？
```

因此，模型把容量预测拆成三条连续流程：

```text
流程一：曲线 -> 状态
  从每个循环的完整放电曲线中，反演五维有效物理状态。

流程二：状态历史 -> 下一状态
  在状态空间中构造退化速度和多尺度退化趋势，让神经网络学习跨循环退化动力学。

流程三：下一状态 -> 电压曲线 -> 截止容量
  用与反演器共享物理参数化结构的可微正演器，把预测状态解码成端电压轨迹，
  再由第一次到达截止电压的位置得到绝对容量。
```

整体链路为：

```text
原始放电记录
  -> 容量轴构造
  -> 曲线与官方容量对齐
  -> 物理反演
  -> 五维有效状态 Z_k
  -> 状态标准化
  -> 退化动力学特征构造
  -> GRU状态转移
  -> 下一状态 Z_hat_{k+1}
  -> 端电压重建 V(q; Z_hat)
  -> 截止事件搜索
  -> 下一容量 Q_hat_{k+1}
```

用公式表示：

$$
\mathcal C_j
\xrightarrow{\text{physical inverse}}
Z_j
\xrightarrow{\text{temporal dynamics}}
\widehat Z_{k+1}
\xrightarrow{\text{physical decoder}}
\widehat Q_{k+1}.
$$

这里的核心原则是：

```text
已知关系优先使用方程；
未知退化规律才交给神经网络。
```

固定OCP、锂平衡投影、端电压方程、截止电压求容量，这些不是神经网络创新，也不是网络学习出来的黑箱关系。它们是显式物理结构。神经网络只学习最难写成闭式方程的部分：

$$
F_\theta:
Z_{k-63:k}
\rightarrow
\widehat Z_{k+1}.
$$

<!-- ## 1. 问题定义：为什么不用容量序列直接预测？

### 1.1 传统容量序列预测的问题

如果直接做：

$$
\widehat Q_{k+1}=f(Q_{k-63:k}),
$$

模型看到的是一个压缩后的标量序列。这样做简单，但会丢掉放电曲线里的大量信息，例如：

```text
平台位置如何移动；
曲线斜率如何变化；
末端电压是否提前下坠；
欧姆压降和极化压降是否增强；
相同容量下电压曲线形状是否不同。
```

同样的容量数值，可能对应不同内部状态；同样的容量下降，也可能由不同机制组合造成。只看容量序列，模型很难区分这些情况。

### 1.2 MGI-DSSM的基本选择

MGI-DSSM把容量视为一个结果，而不是直接状态：

```text
内部有效状态发生变化
  -> 放电电压曲线形状变化
  -> 在固定截止电压下形成可观测容量
```

所以模型不直接学习：

$$
\widehat Q_{k+1}=Q_k+\Delta \widehat Q_{k+1}.
$$

而是学习：

$$
\widehat Z_{k+1}=F_\theta(Z_{k-63:k}),
$$

再用物理求解器得到：

$$
\widehat Q_{k+1}
=
G_{\mathrm{phys}}(\widehat Z_{k+1}).
$$

这样做的意义是：

```text
容量预测被约束在一个能重建电压曲线的状态空间里；
预测结果必须经过端电压和截止事件解释；
神经网络不能随意输出一个缺乏物理闭合的容量值。
``` -->

## 2. 流程一：从放电曲线到五维有效状态

流程一的任务是把每个循环的宏观曲线变成五维状态：

$$
\mathcal C_k
\rightarrow
Z_k.
$$

这一流程是整个模型的证据入口。如果这里没有做好，后面的GRU学到的只是噪声状态。因此流程一不仅要拟合曲线，还要保证状态和容量能闭合。

## 2.1 第一步：读取原始放电记录

CALCE原始文件包含多个工作簿、多个循环、多个步骤。当前实现做：

```text
1. 按工作簿第一个Date_Time排序；
2. 按Cycle_Index分组；
3. 选择Step_Index == 7作为放电阶段；
4. 读取time、current、voltage、internal resistance。
```

输入：

```text
CALCE原始XLSX文件
```

输出：

```text
每个循环的原始放电序列:
  t_m
  I_m
  V_m
  R_m
```

为什么这么做：

```text
1. 工作簿顺序必须还原真实测试时间，否则循环顺序会错。
2. 必须按Cycle_Index分组，否则一个循环的曲线会和另一个循环混在一起。
3. 必须固定放电步骤，否则不同工况阶段的数据会进入同一条曲线。
4. 电压和电流是构造容量轴和反演状态的基本观测。
```

## 2.2 第二步：由电流积分构造容量轴

原始记录是时间序列，但电池放电曲线更适合写成容量-电压曲线：

$$
V(q).
$$

容量增量为：

$$
\Delta q_j
=
-\frac{I_j(t_j-t_{j-1})}{3600}.
$$

累计容量为：

$$
q_m
=
\sum_{j=1}^{m}\Delta q_j.
$$

输入：

```text
时间 t_j
电流 I_j
电压 V_j
```

输出：

```text
容量-电压曲线:
  (q_j, V_j)
```

为什么这么做：

```text
1. 端电压方程天然写成 V(q; Z)，因为放电过程中化学计量随容量变化。
2. 不同循环的采样时间点不一定一致，容量轴比时间轴更适合作曲线对齐。
3. 截止容量本身就是电压第一次到达截止值时的q坐标。
```

## 2.3 第三步：曲线容量轴与官方容量标签对齐

原始积分得到的曲线终点容量可能和官方容量摘要不完全一致，原因包括文件缺失、循环计数差异、采样边界不同等。

设原始曲线终点为：

$$
Q_i^{\mathrm{raw}}
=
q_{i,\mathrm{end}}^{\mathrm{raw}},
$$

官方容量为：

$$
Q_j^{\mathrm{official}}.
$$

实现中先用原始官方容量序列完成保序最小成本匹配；在本文锁定的MSTEA配置下，
随后对官方容量序列执行21循环局部均值$\pm3\sigma$检测并对异常点线性插值，
最后把每条已匹配曲线的容量轴缩放到对应的清洗后官方容量：

$$
q_k^{\mathrm{aligned}}
=
q_k^{\mathrm{raw}}
\cdot
\frac{Q_k^{\mathrm{official}}}
{q_{k,\mathrm{end}}^{\mathrm{raw}}}.
$$

输入：

```text
原始曲线列表
官方容量序列
```

输出：

```text
与官方容量标签一致的曲线:
  (q_aligned, V)
```

为什么这么做：

```text
1. 状态反演使用曲线容量轴，最终评估使用官方容量标签，两者必须处在同一坐标。
2. 如果不对齐，反演器可能拟合到一个容量终点，但训练标签是另一个容量终点。
3. 保序对齐避免打乱真实循环顺序。
```

## 2.4 第四步：定义五维有效状态

每个循环的状态定义为：

$$
Z_k=
\begin{bmatrix}
Q_{\mathrm{Li},k}&
C_{n,k}&
C_{p,k}&
R_{0,k}&
R_{p,k}
\end{bmatrix}.
$$

其中：

| 状态 | 含义 | 作用 |
| --- | --- | --- |
| $Q_{\mathrm{Li}}$ | 有效可循环锂库存 | 决定整电池锂平衡和OCV位置 |
| $C_n$ | 负极有效容量尺度 | 决定负极化学计量变化速度 |
| $C_p$ | 正极有效容量尺度 | 决定正极化学计量变化速度 |
| $R_0$ | 等效欧姆阻抗 | 产生瞬时压降 |
| $R_p$ | 等效极化阻抗 | 产生一阶动态极化压降 |

为什么是这五维：

```text
1. 前三维描述热力学/库存相关变化，能影响OCV曲线位置和形状。
2. 后二维描述阻抗/极化相关变化，能影响负载端电压。
3. 这五维都能进入端电压方程，因此能被曲线残差约束。
4. 不保留无观测支撑的“其他因素”维度，避免不可解释容量捷径。
```

边界说明：

```text
这些是有效整电池状态，不是材料级直接测量真值。
这些状态的价值来自曲线重建和容量闭合，而不是唯一可辨识性。
```

## 2.5 第五步：固定OCP先验提供曲线几何

模型使用固定正负极OCP形状：

$$
U_n(x_n),\qquad U_p(x_p).
$$

整电池开路电压为：

$$
U_{\mathrm{ocv}}(q)
=
U_p(x_p(q))-U_n(x_n(q)).
$$

输入：

```text
负极化学计量 x_n
正极化学计量 x_p
```

输出：

```text
开路电压 U_ocv
```

为什么这么做：

```text
1. 放电曲线的平台和斜率主要来自正负极OCP差。
2. 如果不用OCP先验，五维状态很难和曲线形状建立物理联系。
3. 固定OCP能提供合理几何，但不会引入大量不可辨识材料参数。
```

必须注意：

```text
OCP表是固定LCO/石墨形状先验；
不能宣称为CALCE材料专属精确标定曲线。
```

## 2.6 第六步：锂平衡投影确定初始化学计量

状态中有 $Q_{\mathrm{Li}}, C_n, C_p$，但端电压方程需要 $x_{n,0}, x_{p,0}$。模型通过锂平衡把它们联系起来：

$$
C_nx_{n,0}+C_px_{p,0}
=
Q_{\mathrm{Li}}.
$$

参考点为：

$$
x_{n,\mathrm{ref}}=0.90,
\qquad
x_{p,\mathrm{ref}}=0.35.
$$

定义：

$$
\delta
=
Q_{\mathrm{Li}}
-C_nx_{n,\mathrm{ref}}
-C_px_{p,\mathrm{ref}}.
$$

投影得到：

$$
x_{n,0}
=
x_{n,\mathrm{ref}}
+
\delta
\frac{C_n}{C_n^2+C_p^2+\epsilon},
$$

$$
x_{p,0}
=
x_{p,\mathrm{ref}}
+
\delta
\frac{C_p}{C_n^2+C_p^2+\epsilon}.
$$

输入：

```text
Q_Li
C_n
C_p
```

输出：

```text
x_n0
x_p0
```

为什么这么做：

```text
1. 保证正负极初始点和整电池可循环锂库存一致。
2. 避免把x_n0、x_p0也设为自由变量导致反演更不可辨识。
3. 让状态中的前三维真正通过锂平衡进入OCP曲线。
```

## 2.7 第七步：生成放电过程中的化学计量轨迹

放电容量为 $q$ 时：

$$
x_n(q)
=
x_{n,0}
-
\frac{q}{C_n},
$$

$$
x_p(q)
=
x_{p,0}
+
\frac{q}{C_p}.
$$

输入：

```text
q
x_n0, x_p0
C_n, C_p
```

输出：

```text
x_n(q)
x_p(q)
```

为什么这么做：

```text
1. C_n和C_p控制化学计量随容量推进的速度。
2. 容量越小，单位放电量造成的化学计量变化越快。
3. 这使电极容量衰退能体现为电压曲线形状和截止点变化。
```

## 2.8 第八步：端电压方程重建候选曲线

给定候选状态 $Z$，端电压为：

$$
V(q;Z)
=
U_p(x_p(q))
-U_n(x_n(q))
-IR_0
-IR_p
\left(
1-\exp
\left[
-\frac{3600q}{I\tau_p}
\right]
\right).
$$

也就是：

$$
V(q;Z)
=
U_{\mathrm{ocv}}(q)
-IR_0
-\eta_p(q),
$$

其中：

$$
\eta_p(q)
=
IR_p
\left(
1-\exp
\left[
-\frac{3600q}{I\tau_p}
\right]
\right).
$$

输入：

```text
状态 Z
容量网格 q
放电电流 I
极化时间常数 tau_p
```

输出：

```text
重建端电压曲线 V(q; Z)
```

为什么这么做：

```text
1. 这是从状态到可观测电压曲线的前向观测方程。
2. 反演时需要用它比较候选状态和真实曲线。
3. 预测时使用具有相同OCP、锂平衡和一阶极化结构的PyTorch正演器，把预测状态转成容量。
4. 两个实现共享物理参数化结构，但并非逐项完全相同：离线反演使用每循环实测中位电流，
   正演器使用配置中的固定1.1 A电流；正演器还对$C_n,C_p$采用0.2 Ah的分母下限，
   而反演搜索边界允许二者低至0.15 Ah。
```

## 2.9 第九步：构造反演优化问题

对每条曲线，求：

$$
Z_k^\star
=
\arg\min_Z
\mathcal J_k(Z).
$$

当前主要优化：

$$
p=[Q_{\mathrm{Li}},C_n,C_p,R_0].
$$

$R_p$ 由末端电压关系估计：

$$
R_p^{\mathrm{raw}}
=
\frac{
U_{\mathrm{ocv}}(Q)
-IR_0
-V_{\mathrm{obs}}(Q)
}{
I
\left(
1-\exp
\left[
-\frac{3600Q}{I\tau_p}
\right]
\right)
}.
$$

再裁剪：

$$
R_p
=
\operatorname{clip}(R_p^{\mathrm{raw}},0.001,1.20).
$$

为什么 $R_p$ 这样处理：

```text
1. 末端电压对极化压降很敏感，可以给R_p提供直接约束。
2. 少优化一个自由变量可以降低反演不适定性。
3. 裁剪可避免极端曲线导致不可行极化参数。
```

## 2.10 第十步：反演残差为什么这样设计

反演残差不是只有电压误差，而是多项拼接：

```text
r = [
  voltage residual,
  DVA residual,
  cutoff residual,
  resistance residual,
  temporal residual,
  Rp-bound residual
]
```

### 电压残差

$$
r_V(q)
=
V_{\mathrm{pred}}(q)-V_{\mathrm{obs}}(q).
$$

作用：

```text
拟合整体放电曲线。
```

为什么需要：

```text
这是最直接的观测证据。
```

### DVA残差

$$
r_{\mathrm{DVA}}(q)
=
\alpha_{\mathrm{DVA}}
\left(
\frac{dV_{\mathrm{pred}}}{dq}
-
\frac{dV_{\mathrm{obs}}}{dq}
\right).
$$

作用：

```text
约束曲线斜率和形状。
```

为什么需要：

```text
只拟合电压值时，Q_Li、C_n、C_p之间可能互相补偿；
DVA能增加形状证据，帮助区分热力学三维。
```

### 截止点残差

$$
r_{\mathrm{cut}}
=
V_{\mathrm{pred}}(Q_{\mathrm{obs}})
-V_{\mathrm{obs}}(Q_{\mathrm{obs}}).
$$

作用：

```text
强制候选状态在真实容量终点附近也能解释末端电压。
```

为什么需要：

```text
容量是截止事件，如果全局曲线拟合得不错但截止点错了，
后续容量解码会错。
```

### 内阻弱观测残差

$$
r_R
=
\frac{R_0-R_{0,\mathrm{obs}}}{0.2}.
$$

作用：

```text
给R_0一个弱参考。
```

为什么只是弱参考：

```text
设备内阻字段的脉冲时长和测量协议不完整，
不能当作严格欧姆阻抗标签。
```

### 相邻循环弱连续性

$$
r_{\mathrm{temp}}
=
0.015
\frac{p_k-p_{k-1}}{s_p}.
$$

作用：

```text
让相邻循环反演状态不要无意义跳变。
```

为什么只是弱连续：

```text
真实观测可能有局部恢复、平台波动和测量噪声；
如果强行单调或强行贴上一循环，会压制有效状态对局部变化的响应。
```

### $R_p$ 边界残差

$$
r_{R_p}
=
0.1\max(0.001-R_p^{\mathrm{raw}},0)
+
0.1\max(R_p^{\mathrm{raw}}-1.20,0).
$$

作用：

```text
惩罚由末端估计得到的不可行极化阻抗。
```

## 2.11 第十一步：鲁棒多初值最小二乘

使用：

```text
least_squares
bounds:
  lower = [0.20, 0.15, 0.15, 0.001]
  upper = [2.60, 3.50, 3.50, 0.60]
loss = soft_l1
f_scale = 0.01
max_nfev = 80
multi-start = true
```

为什么这么做：

```text
1. 反演是非线性问题，单初值容易陷入局部最优。
2. soft_l1比普通L2更抗异常点。
3. 边界防止状态跑到无物理意义区域。
4. max_nfev控制每条曲线反演成本。
```

流程一最终输出：

```text
每个循环:
  Z_k = [Q_Li, C_n, C_p, R_0, R_p]

每个电池:
  states, shape = [num_cycles, 5]
  capacities, shape = [num_cycles]
```

## 2.12 第十二步：物理闭合检查

反演后的状态必须通过共享同一物理参数化结构的容量解码器还原容量：

$$
\widehat Q_k^{\mathrm{closure}}
=
G_{\mathrm{phys}}(Z_k).
$$

闭合误差：

$$
\mathrm{MAE}_{\mathrm{closure}}
=
\frac1N
\sum_k
|\widehat Q_k^{\mathrm{closure}}-Q_k|.
$$

主配置阈值：

$$
\mathrm{MAE}_{\mathrm{closure}}
\le
0.004\ \mathrm{Ah}.
$$

为什么必须做：

```text
1. 后续训练依赖这些状态作为监督和输入。
2. 如果状态不能被正演器解码回容量，说明反演参数化和容量解码之间没有形成可接受的数值闭合。
3. 闭合失败时继续训练，会让神经网络学习一个不自洽状态空间。
```

需要明确：这里检查的是**反演状态的容量自重构闭合**，不能据此证明五个状态唯一可辨识，
也不能替代原始电压逐点重构误差、独立物理测量或预测状态物理有效性的验证。

到这里，流程一结束。我们已经把宏观曲线变成可用于时序建模的状态轨迹：

$$
\{\mathcal C_1,\ldots,\mathcal C_T\}
\rightarrow
\{Z_1,\ldots,Z_T\}.
$$

## 3. 流程二：从历史状态到下一状态

流程二的任务是学习：

$$
Z_{k-63:k}
\rightarrow
\widehat Z_{k+1}.
$$

这一步才是神经网络真正负责的部分。它不学习OCP，不学习锂平衡，不学习端电压方程，而是学习跨循环退化动力学。

## 3.1 第一步：构造因果历史窗口

对目标循环 $i$，样本为：

```text
states[i-64:i] -> states[i], capacity[i]
```

也就是：

$$
\{Z_{i-64},\ldots,Z_{i-1}\}
\rightarrow
Z_i.
$$

张量形状：

```text
history:
  [B, 64, 5]

target next_state:
  [B, 5]

target capacity:
  [B]
```

为什么这么做：

```text
1. 输入只包含目标循环之前的信息，避免未来泄漏。
2. 64步窗口给模型足够长的局部寿命历史。
3. 目标状态和目标容量只用于监督，不进入输入。
```

## 3.2 第二步：只用训练电池做状态标准化

计算：

$$
\mu_Z,\qquad \sigma_Z.
$$

标准化：

$$
\widetilde Z_t
=
\frac{Z_t-\mu_Z}
{\max(\sigma_Z,10^{-6})}.
$$

输入：

```text
训练电池状态
历史窗口状态
```

输出：

```text
标准化状态 Z_tilde
```

为什么这么做：

```text
1. 五个状态量纲不同，直接输入会让大尺度变量主导训练。
2. 只用训练电池统计量，避免测试电池信息泄漏。
3. 标准化后，差分和趋势特征具有可比尺度。
```

这里“训练电池统计量”指留一验证中除测试电池以外各电池的完整状态序列，
其中也包含随后作为验证尾段的循环。因此该实现没有使用测试电池统计量，
但标准化统计量并非严格只由优化训练窗口拟合。

## 3.3 第三步：构造单步退化速度

状态差分：

$$
\Delta\widetilde Z_t
=
\widetilde Z_t-\widetilde Z_{t-1}.
$$

可解释为状态空间中的退化速度：

$$
v_t
=
\Delta\widetilde Z_t.
$$

输入：

```text
标准化状态序列
```

输出：

```text
delta, shape = [B, 64, 5]
```

为什么这么做：

```text
1. 只看状态水平，不知道状态正在快速退化还是趋于稳定。
2. 差分直接告诉模型最近一循环的变化方向和幅度。
3. 它把“位置预测”变成“位置+速度预测”。
```

## 3.4 第四步：构造短期退化速率

8步因果平均：

$$
T_t^{(8)}
=
\frac18
\sum_{j=0}^{7}
\Delta\widetilde Z_{t-j}.
$$

输入：

```text
delta序列
```

输出：

```text
trend8, shape = [B, 64, 5]
```

为什么这么做：

```text
1. 单步差分可能受反演噪声影响。
2. 8步平均能表达局部退化速率。
3. 对阻抗短期波动、局部恢复、平台扰动更敏感。
```

## 3.5 第五步：构造长期退化速率

32步因果平均：

$$
T_t^{(32)}
=
\frac1{32}
\sum_{j=0}^{31}
\Delta\widetilde Z_{t-j}.
$$

输入：

```text
delta序列
```

输出：

```text
trend32, shape = [B, 64, 5]
```

为什么这么做：

```text
1. 电池退化通常有长期缓慢趋势。
2. 32步平均压制局部噪声，保留长期方向。
3. 它帮助模型区分短期波动和真正退化趋势。
```

对于窗口左边界以前不存在的历史差分，源码不是缩短平均窗口，而是先令窗口首个差分为零，
再使用`replicate`方式向左填充。因此上述MA8和MA32公式适用于历史充分的位置；
窗口开始处的缺失差分按首个零差分补齐。

## 3.6 第六步：拼接动力学输入

最终每个时刻输入为：

$$
X_t
=
[
\widetilde Z_t,
\Delta\widetilde Z_t,
T_t^{(8)},
T_t^{(32)}
].
$$

每项5维，因此：

$$
X_t\in\mathbb R^{20}.
$$

窗口输入：

$$
X_{k-63:k}
\in
\mathbb R^{64\times20}.
$$

输入：

```text
Z_tilde, delta, trend8, trend32
```

输出：

```text
transition_input, shape = [B, 64, 20]
```

为什么这么做：

```text
1. Z_tilde表示当前处在什么状态。
2. delta表示刚刚怎么变。
3. trend8表示近期退化速率。
4. trend32表示长期退化趋势。
5. 四者合起来就是状态空间中的退化动力学表征。
```

可以在论文中表述为：

```text
The temporal input is constructed as a degradation-dynamics descriptor,
including state level, one-cycle velocity, short-term degradation rate,
and long-term degradation rate.
```

## 3.7 第七步：GRU学习未知状态转移

GRU编码历史：

$$
H_{k-63:k}
=
\operatorname{GRU}_{\theta}
(X_{k-63:k}).
$$

取最后上下文：

$$
h_k=H_k.
$$

MLP输出原始转移控制：

$$
r
=
W_2
\operatorname{SiLU}
\left(
W_1\operatorname{LN}(h_k)+b_1
\right)
+b_2.
$$

输入：

```text
transition_input, shape = [B, 64, 20]
```

输出：

```text
raw transition r, shape = [B, 5]
temporal context h_k
```

为什么这么做：

```text
1. 跨循环退化受复杂老化、恢复、测量波动和未建模因素影响，很难写成闭式方程。
2. GRU适合处理因果历史序列。
3. 网络输出的是状态转移控制量，不是容量。
```

## 3.8 第八步：乘性参数化得到下一状态

热力学三维：

$$
\widehat Z_{k+1}^{1:3}
=
Z_k^{1:3}
\odot
\exp
\left(
0.02\tanh r_{1:3}
\right).
$$

阻抗二维：

$$
\widehat Z_{k+1}^{4:5}
=
Z_k^{4:5}
\odot
\exp
\left(
0.03\tanh r_{4:5}
\right).
$$

输入：

```text
last state Z_k
raw transition r
```

输出：

```text
predicted next state Z_hat_{k+1}
```

为什么这么做：

```text
1. exp形式保证状态为正。
2. tanh限制单步变化幅度。
3. 热力学三维和阻抗二维使用不同步长，符合尺度差异。
4. 输出是下一状态绝对水平，而不是容量残差。
```

## 3.9 第九步：为什么当前允许reversible状态转移

早期可以加入方向损失：

$$
\mathcal L_{\mathrm{dir}}
=
\operatorname{ReLU}
(\widehat Z_{k+1}^{1:3}-Z_k^{1:3})
+
\operatorname{ReLU}
(Z_k^{4:5}-\widehat Z_{k+1}^{4:5}).
$$

它假设：

```text
热力学容量相关维度只能下降；
阻抗相关维度只能上升。
```

本文档锁定的MSTEA-reversible候选配置为：

$$
\lambda_{\mathrm{dir}}=0.
$$

为什么这么做：

```text
1. CALCE中存在局部恢复、平台波动和测量噪声。
2. 反演状态是effective state，不是严格不可逆材料变量。
3. 逐循环强单调会错误压制短期回调，使预测变差。
4. 长期退化结构由历史窗口、曲线监督、状态监督和容量监督共同约束。
```

注意表述边界：

```text
reversible不表示电化学老化完全可逆；
它表示有效状态估计允许对局部恢复和反演噪声作出响应。
```

该结论不适用于`configs/final_calce_10runs.json`：后者采用坐标监督且
`direction_loss_weight=0.05`。因此论文结果、消融表和方法描述必须始终标明所对应的配置。

到这里，流程二结束。模型得到：

$$
\widehat Z_{k+1}
=
F_\theta(Z_{k-63:k}).
$$

## 4. 流程三：从下一状态到电压曲线和容量

流程三的任务是把预测状态变成最终容量：

$$
\widehat Z_{k+1}
\rightarrow
\widehat V_{k+1}(q)
\rightarrow
\widehat Q_{k+1}.
$$

这一步让容量预测必须经过物理解释。

## 4.1 第一步：用共享物理参数化结构的正演器重建下一循环电压轨迹

预测状态进入端电压方程：

$$
\widehat V_{k+1}(q)
=
V(q;\widehat Z_{k+1}).
$$

输入：

```text
predicted next state Z_hat_{k+1}
capacity grid q
```

输出：

```text
predicted voltage curve V_hat(q)
```

为什么这么做：

```text
1. 保证预测状态不仅能给容量，还能给出预设容量网格上的候选电压轨迹。
2. 训练时可以用曲线监督约束状态预测。
3. 反演器和解码器共享OCP、锂平衡及一阶极化结构，从而形成方程结构一致的数值闭环。
```

这里的“电压轨迹”需要限定解释范围：真实放电在第一次达到截止电压后已经终止，
因此截止前部分具有观测意义，截止后的数值仅是物理方程在固定容量网格上的延拓，
用于搜索截止事件，不能作为真实可观测曲线解释。

## 4.2 第二步：用截止电压定义容量

容量不是网络头直接输出，而是截止事件：

$$
\widehat Q_{k+1}
=
\inf
\{q:\widehat V_{k+1}(q)\le V_{\mathrm{cut}}\}.
$$

当前：

$$
V_{\mathrm{cut}}=2.7\ \mathrm V.
$$

输入：

```text
predicted voltage curve V_hat(q)
cutoff voltage V_cut
```

输出：

```text
predicted absolute capacity Q_hat
```

为什么这么做：

```text
1. 实验容量本质上就是放电到截止电压时的容量。
2. 这样预测容量和端电压曲线绑定。
3. 避免容量成为一个脱离物理曲线的自由标量。
```

## 4.3 第三步：先做累积最小，避免截止后回升

由于OCP插值裁剪可能造成曲线尾部数值回升，求解器先做：

$$
\widetilde V(q_i)
=
\min_{j\le i}
\widehat V(q_j).
$$

再找第一次：

$$
\widetilde V(q_i)
\le
V_{\mathrm{cut}}.
$$

为什么这么做：

```text
1. 放电应在第一次到达截止电压时结束。
2. 如果数值曲线后面回升，不能继续累加截止后的虚假容量。
3. 累积最小保证容量定义是第一次截止事件。
```

## 4.4 第四步：网格间线性插值

若截止发生在 $q_{i-1}$ 和 $q_i$ 之间：

$$
\widehat Q
=
q_{i-1}
+
\alpha(q_i-q_{i-1}),
$$

其中：

$$
\alpha
=
\frac{
V_{\mathrm{cut}}-\widetilde V(q_{i-1})
}{
\widetilde V(q_i)-\widetilde V(q_{i-1})
}.
$$

为什么这么做：

```text
1. 容量网格是离散的，直接取网格点会有量化误差。
2. 相邻点线性插值能给出更平滑的容量估计。
3. 在首次截止区间固定时，线性插值为相邻网格电压提供局部梯度，
   使容量损失可以反向传播。首次穿越索引仍由离散比较和`argmax`确定，
   因而容量解码器整体是分段可微，而不是全局光滑可微。
```

容量搜索还包含一个上界条件。当前正演网格为400点，范围为
$q\in[0,1.5]$ Ah；如果整个网格内都没有发生2.7 V截止穿越，求解器返回：

$$
\widehat Q=1.5\ \mathrm{Ah}.
$$

因此1.5 Ah既是数值搜索上限，也是无截止穿越样本的截断输出，不能解释为模型识别出的真实容量终点。

## 4.5 第五步：训练时的多空间监督

模型训练不是只看容量误差，而是在状态坐标空间、状态诱导的电压函数空间和截止容量空间联合监督。

### 状态损失

$$
\mathcal L_{\mathrm{state}}
=
\operatorname{SmoothL1}
\left(
\frac{
\widehat Z_{k+1}-Z_{k+1}
}{s_Z}
,
0
\right),
$$

其中：

$$
s_Z=[0.25,0.4,0.4,0.05,0.05].
$$

为什么需要：

```text
直接约束预测状态不要偏离反演目标状态。
```

### 状态诱导的曲线损失

目标曲线不是直接读取原始电压采样点，而是由离线反演得到的目标状态通过PyTorch正演器重建：

$$
V_{\mathrm{target}}(q)
=
V(q;Z_{k+1}).
$$

预测曲线：

$$
\widehat V(q)
=
V(q;\widehat Z_{k+1}).
$$

曲线误差：

$$
e_V(q)
=
\frac{
\widehat V(q)-V_{\mathrm{target}}(q)
}{0.05}.
$$

只在真实截止前监督：

$$
\Omega=\{q:q\le Q_{k+1}\}.
$$

损失：

$$
\mathcal L_{\mathrm{curve}}
=
\frac1{|\Omega|}
\sum_{q\in\Omega}
\operatorname{SmoothL1}(e_V(q);\beta=0.2).
$$

为什么需要：

```text
1. 状态坐标损失逐维约束反演参数，曲线损失则约束这些参数诱导出的函数空间表现。
2. 曲线损失要求预测状态在共享正演器下生成与目标反演状态一致的电压形状。
3. 目标曲线仍由反演状态间接构造，并非独立的原始观测证据；状态损失与曲线损失具有共同来源。
4. 真实循环在截止后没有电压观测，所以只在官方容量定义的截止前区域计算代理曲线损失。
```

### 容量损失

$$
\mathcal L_Q
=
\frac1B
\sum_i
w_i
\operatorname{SmoothL1}
(\widehat Q_i-Q_i;\beta=0.01).
$$

晚寿命权重：

$$
w_i
=
1+\lambda_{\mathrm{late}}
\operatorname{clip}
\left(
1-\frac{Q_i}{Q_{\mathrm{rated}}},
0,1
\right).
$$

为什么需要：

```text
最终任务仍是容量预测，所以容量误差必须主导训练目标。
晚寿命区域对EOL和寿命评估更敏感，因此可适度加权。
```

### 总损失

曲线监督模式下：

$$
\mathcal L
=
\lambda_{\mathrm{curve}}\mathcal L_{\mathrm{curve}}
+
\lambda_{\mathrm{weak}}\mathcal L_{\mathrm{state}}
+
\lambda_Q\mathcal L_Q
+
\lambda_{\mathrm{dir}}\mathcal L_{\mathrm{dir}}
+\cdots
$$

本文档锁定的MSTEA-reversible候选配置：

```text
curve_loss_weight = 0.02
weak_state_loss_weight = 0.02
capacity_loss_weight = 1.0
direction_loss_weight = 0.0
late_life_weight = 0.5
```

## 4.6 第六步：评估为什么必须包含Persistence

Persistence基线为：

$$
\widehat Q_{k+1}^{\mathrm{pers}}
=
Q_k.
$$

为什么必须报告：

```text
1. 电池容量序列具有很强的一阶持续性。
2. 单步预测中，接近上一循环容量本身就是强基线。
3. 如果只报告模型MAE，不和Persistence比较，容易夸大贡献。
4. 若MAE接近Persistence，应强调物理闭合、状态可审计性和跨电池泛化，而不是只强调数值领先。
```

主要指标：

$$
\mathrm{MAE}
=
\frac1N
\sum_i
|\widehat Q_i-Q_i|.
$$

$$
\mathrm{RMSE}
=
\sqrt{
\frac1N
\sum_i
(\widehat Q_i-Q_i)^2
}.
$$

$$
R^2
=
1-
\frac{
\sum_i(\widehat Q_i-Q_i)^2
}{
\sum_i(Q_i-\bar Q)^2
}.
$$

EOL阈值：

$$
Q_{\mathrm{EOL}}
=
0.7Q_{\mathrm{rated}}
=
0.77\ \mathrm{Ah}.
$$

Ecycle：

$$
\mathrm{Ecycle}
=
|k_{\mathrm{EOL}}^{\mathrm{pred}}
-k_{\mathrm{EOL}}^{\mathrm{true}}|.
$$

## 5. 三条流程放在一起看

## 5.1 流程一的输入输出

```text
输入:
  原始放电曲线 C_k

内部步骤:
  读取 -> 积分 -> 对齐 -> OCP/锂平衡/端电压前向 -> 非线性反演 -> 闭合检查

输出:
  Z_k = [Q_Li, C_n, C_p, R_0, R_p]
```

流程一回答的问题：

```text
这一循环的曲线，能被什么样的有效物理状态解释？
```

## 5.2 流程二的输入输出

```text
输入:
  历史状态 Z_{k-63:k}

内部步骤:
  标准化 -> 单步差分 -> 短期速率 -> 长期速率 -> GRU -> 乘性状态更新

输出:
  Z_hat_{k+1}
```

流程二回答的问题：

```text
在过去64个循环里，状态以什么速度和趋势变化？
按照这种退化动力学，下一循环状态应该在哪里？
```

## 5.3 流程三的输入输出

```text
输入:
  预测状态 Z_hat_{k+1}

内部步骤:
  端电压重建 -> 累积最小 -> 第一次截止搜索 -> 线性插值 -> 容量输出

输出:
  Q_hat_{k+1}
```

流程三回答的问题：

```text
如果下一循环处在这个状态，它的放电电压曲线会怎样？
这条曲线第一次到达截止电压时，对应多少容量？
```

## 6. 训练和测试的完整执行顺序

### 6.1 离线阶段

```text
for each battery:
  read raw CALCE files
  extract discharge curves
  integrate current to q axis
  align curves with official capacities
  optionally clean local outliers
  for each cycle:
    run physical inverse
    save Z_k
  cache states and capacities
for each leave-one-battery-out fold:
  load cached states and capacities
  run inverse-state capacity self-reconstruction closure check
  start transition training only if closure passes
```

为什么离线做：

```text
物理反演是非线性优化，成本较高；
缓存后训练阶段可以专注于状态转移学习。
```

### 6.2 训练阶段

```text
choose one held-out test battery
use remaining batteries for training
compute state normalization statistics from training batteries only
construct causal 64-cycle windows
train GRU transition model
decode predicted states through physical solver
optimize curve/state/capacity losses
early stop by validation MAE
```

为什么这样划分：

```text
1. 留一电池测试检验跨电池泛化。
2. 标准化不能使用测试电池，防止数据泄漏。
3. 验证集来自训练电池尾部，早停不看测试电池。
```

### 6.3 测试阶段

```text
for held-out battery:
  build each causal window
  predict next state
  reconstruct voltage curve
  decode cutoff capacity
  compare with true capacity and persistence
  compute MAE, RMSE, R2 and protocol-specific EOL metrics
```

为什么这样测试：

```text
测试窗口只使用目标循环之前的状态；
容量标签只用于评估；
结果能同时反映容量误差和寿命阈值误差。
```

在本文档锁定的MSTEA协议下，代码重新计算绝对EOL循环误差Ecycle/AE，并将RE置为`NaN`；
只有PatchFormer协议分支才保留其AE和RE定义。因此本文候选配置不能同时报告有效的Ecycle和RE。

## 7. 论文写作主线

可以按下面的叙事写论文方法部分。

第一段：说明容量衰退不是孤立标量变化，而是内部有效状态演化后在给定工况和截止电压下形成的宏观结果。

第二段：说明完整放电曲线比容量端点包含更多证据，因此先进行曲线级物理反演，得到五维有效状态。

第三段：说明反演不是任意拟合五个坐标，而是由固定OCP先验、锂平衡投影、端电压方程、DVA残差、截止点残差和弱连续性共同约束。

第四段：说明状态空间已经包含热力学和阻抗两个方面；时序输入进一步构造状态水平、单步退化速度、短期退化速率和长期退化速率，从而学习跨循环退化动力学。

第五段：说明神经网络只负责未知状态转移，不直接输出容量残差。

第六段：说明预测状态经过与反演器共享物理参数化结构的PyTorch正演器生成候选电压轨迹，
并通过第一次到达截止电压的位置得到绝对容量。

第七段：说明训练在状态坐标、状态诱导的曲线函数和截止容量三个空间联合监督；
同时明确曲线目标由反演状态正演生成，评估必须和Persistence比较，并报告容量自重构闭合。

## 8. 可以主张的贡献

```text
1. 将完整放电曲线反演为可审计的五维有效状态。
2. 用固定OCP先验、锂平衡和一阶极化方程构造参数化结构一致的反演-解码闭环。
3. 在状态空间中建模跨循环退化，而不是在容量残差空间外推。
4. 将时序输入显式构造成退化动力学描述，包括速度和多尺度速率。
5. 用物理截止事件直接解码绝对容量。
6. 通过状态坐标、状态诱导曲线和截止容量的多空间监督增强一致性。
7. 通过闭合检查和Persistence对照控制论文叙事边界。
```

## 9. 不能夸大的内容

```text
1. 不能称五维状态为直接测量微观量。
2. 不能称OCP为CALCE材料专属精确曲线。
3. 不能声称反演参数是唯一可辨识材料参数。
4. 不能把固定方程包装成神经网络创新。
5. 不能把TJU代理状态等同于CALCE物理反演状态。
6. 不能隐藏Persistence结果。
7. 不能只选最佳随机种子。
8. 不能把Ecycle改善等同于完整寿命轨迹误差改善。
```

## 10. 最终摘要

本文档所述CALCE `mgi-physics`流程是：先把原始放电记录转换成容量-电压曲线，并与官方容量标签保序对齐和缩放；再用固定OCP先验、锂平衡投影和一阶极化端电压方程，对每个循环执行物理反演，得到五维有效状态；然后在状态序列上构造状态水平、单步退化速度、短期退化速率和长期退化速率，让GRU学习跨循环退化动力学；最后把预测出的下一状态送入共享相同物理参数化结构的PyTorch正演器，生成候选端电压轨迹，并由第一次达到截止电压的位置解码下一循环绝对容量。离线反演与神经正演共享方程结构但不完全等同，曲线监督由目标反演状态间接生成，截止容量解码为分段可微，相关结论仅对应本文开头锁定的MSTEA-reversible候选配置。
