# SG-DiTs 原文结构审计

审计来源：`D:/class/相关文献/SG-DiTs.pdf`，Journal of Energy Storage 152 (2026) 120479。审计日期：2026-08-11。

## 逐项对照

| 原文位置 | 原文要求 | 当前代码 | 结论 |
|---|---|---|---|
| pp.5–7, Table 3 | SG 平滑后提取 dQ/dV、dV/dQ 各四项，加 Q_CV_Ah、Tau_exp、SE、MLE，共12项 | `features.py` | 特征类型一致；SG window、entropy bins、MLE 数值算法原文未披露，当前值是显式适配 |
| pp.8–9, Eqs. (5)–(11) | 标准 DDPM 正向加噪和反向去噪 | `DiffusionSchedule` | 正向公式及反向均值一致 |
| pp.9–10, Fig.10, Eqs. (12)–(17) | embedding、堆叠 DiTs block、MHSA、ReLU FFN、AdaLN scale/shift/residual gate、线性 reshape 输出 Noise/Variance | `ConditionalDiT`、`AdaLNZeroBlock` | 主干一致；ReLU、AdaLN-Zero 和 noise/variance 双头均已实现 |
| p.10, Eqs. (18)–(19) | 随机采样 timestep；预测噪声 RMSE；SOH 与 timestep 条件注入 | `fit`、`validation_loss` | 一致 |
| p.10 | `T=1000`，beta 从 `1e-4` 到 `0.02` 线性变化 | `DiffusionSchedule` | 一致 |
| p.18, Table 11 | depth=12、dimension=256、heads=8、patch=4、batch=256、lr=1e-4 | `Config` | 一致 |
| p.17 | 固定窗口 L=40，历史容量窗口作为输入，预测下一容量并递归 | 当前 runner | 历史容量窗口已进入 input/embedding stage；窗口按统一任务改为16，且统一逐窗口评估并非从单个 SP 完全递归 |
| p.10, Eq. (17) | 反向分布包含网络产生的 variance | 当前 sampler | **披露缺口**：双头存在，但正文只给 noise RMSE，未说明 variance 监督；当前按标准 DDPM posterior variance 采样 |
| pp.3,8–10 | 从噪声重建退化轨迹，并由历史物理指标提供条件 | 当前 runner | **原文本身含糊**：Eq. (18) 只把 SOH 写为条件 `c`，没有公开12项指标与 `X0` 的精确张量拼接方式；当前采用“容量轨迹为 X0、12项历史指标为附加 embedding” |
| p.17 | 预测容量追加输入窗口 | 当前 sampler | 默认路径已去掉 inpainting；统一逐窗口评估与论文从单个 SP 完全递归仍是不同任务协议 |

## 审计结论

SG-DiTs 的 DiT block、DDPM 调度、AdaLN 条件注入和公开超参数一致。默认推理已去掉非原文 inpainting；但统一 benchmark 的逐窗口单步评估仍不同于论文从单个 SP 完全递归到 EOL 的协议。论文没有公开12项指标的张量合同和 variance loss，任何具体选择都必须继续标为复现假设，不能伪称作者实现。

代码现已明确禁止将上一周期容量作为输出锚点；反向扩散直接生成容量轨迹。
