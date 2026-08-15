# MSTEA-Net / Physics Dual Loss 复现说明

论文：Hanlin Cheng and L. Zhang, *A physics-informed deep learning framework for remaining useful life prediction of lithium-ion batteries with feature subset construction*, Energy 346 (2026) 140288, DOI: 10.1016/j.energy.2026.140288。

## 原文机制对应

| 原文 | 实现 |
|---|---|
| 四类健康特征 CCCT、CVCT、CCDT、Resistance | 从 NASA 官方逐周期 charge/discharge/impedance 记录提取 |
| STL 趋势、季节与残差分解，式 (1)-(6) | `features.py::detect_period` 与 `causal_stl` |
| RF top-6、LassoCV、投票、RFECV，式 (7)-(10) | `features.py::select_wrapper_features` |
| 输入投影，式 (11)-(12) | `MSTEANet.input_projection` |
| LSTM 与 k=1/3/7 Conv1D、残差、位置编码，式 (13)-(17) | `MSTEANet.forward` |
| 4 头注意力、两次残差/LayerNorm、FFN，式 (18)-(25) | `MSTEANet.attention`、`ffn` |
| 末时刻 + 两层 GELU 预测头，式 (26)-(28) | `prediction_head` |
| Arrhenius 静态约束与导数约束，式 (29)-(35) | `TripleCompositeLoss` |
| MSE + 1e-4 Larr + 1e-4 Lderiv，式 (36)-(37) | `TripleCompositeLoss.components` |
| hidden=32、heads=4；论文 CALCE/TJU lr=0.01 | `PaperConfig`；NASA 验证集选择 lr=3e-4 |

## NASA 任务适配

统一协议要求 B0005 留出测试，B0006/B0007/B0018 的前 80% 训练、后 20% 验证，16 周期真实历史窗口进行滚动单步容量预测，并在 SP50/SP90 后评估。原文的网络、特征子集构造和物理损失均保留；窗口由 64 改为 16，温度采用 NASA README 所述室温 298.15 K。依据原文第 4.1 节“滑动窗口提取目标电池退化序列作为输入”，已观测 `C/C(0)` 历史作为额外输入通道与 RFECV 健康特征共同进入原始 input projection；预测头仍直接输出容量，绝不执行上一容量加残差。由于 NASA 电芯间的初始容量和充放电时长绝对尺度存在偏置，训练目标采用 `C(N)/C(0)`，四项健康特征在 STL 前同样除以各自首周期值，评估时再恢复为 Ah；容量表示与原文式 (31) 的 `Dmeas=1-C(N)/C(0)` 完全等价。`data/version3` 始终只读，所有变换仅发生在本模型内存中的输入副本。

为满足统一协议的数据隔离，B0005 不参与周期检测、Wrapper 选择或标准化。原论文的 STL 是离线双向分解，直接使用会让测试时刻看到未来；本实现仅用当前及历史点进行 Loess 端点趋势和同相位历史季节项估计。该调整保持“STL 三分量 + Wrapper”思路，同时符合滚动在线任务。

## 论文未披露项

原文没有公开源码，也未给出 Loess span、ACF 经验候选集合、式 (6) 的 alpha/beta 数值、FFN 内部宽度、批量大小与优化器。本实现把这些选择固定为：候选周期 `{5,7,10,14,21,28}` 加 ACF 局部峰、alpha=beta=0.5、Loess span=`2p+1`、FFN 宽度 `2*hidden`、同一电池连续窗口 batch size 16、Adam。连续小批保证式 (35) 不跨电池且只比较相邻周期。结果 JSON 会保存全部配置，避免把未披露选择误称为原文参数。原文 CALCE 使用 500 epoch；当前统一实验最大训练轮数设置为 200。验证总损失连续 20 epoch 未改善至少 `1e-6` 时早停，并恢复验证集最优 checkpoint；测试电池不参与停止判断。

## 运行

快速验证：

```powershell
python Compare-Models/run_physics_dual_loss.py --seeds 7 --max-epochs 2 --device cpu --output-root outputs/physics_dual_loss_smoke
```

NASA 正式十随机种子：

```powershell
python Compare-Models/run_physics_dual_loss.py --device cuda --output-root outputs/comparison_models_10seeds
```

CALCE 与 TJU（默认最大 200 epoch，分别留出 CS2_35 与 CY25-1）：

```powershell
python Compare-Models/run_physics_dual_loss.py --dataset calce --seeds 7 --device cuda --output-root outputs/physics_dual_loss_calce_200ep
python Compare-Models/run_physics_dual_loss.py --dataset tju --seeds 7 --device cuda --output-root outputs/physics_dual_loss_tju_200ep
```

两个数据集都沿用统一 64-cycle 单步任务。CALCE 使用 CCCT/CVCT/CCDT/Resistance；TJU 因无逐周期内阻，按原文对应设置使用 CCCT/CVCT/CCDT。所有输入只在内存中构造。
