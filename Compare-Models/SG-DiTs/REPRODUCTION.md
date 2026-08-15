# SG-DiTs 复现说明

论文：Jing Sun and Jingfan Gao, *SG-DiTs: A physics-informed diffusion-transformer hybrid framework for reliable battery RUL prediction*, Journal of Energy Storage 152 (2026) 120479，DOI `10.1016/j.est.2026.120479`。

## 原文机制

SG-DiTs 的“physics-informed”来自条件输入中的电池机理健康指标，而不是额外物理 loss：

1. 对 CC-CV 充电曲线实施 Savitzky–Golay 平滑。
2. 提取 dQ/dV 与 dV/dQ 的峰数量、主峰位置、主峰高度和积分面积，共 8 项。
3. 提取 CV 容量 `Q_CV_Ah`、电流指数衰减时间常数 `Tau_exp`、Shannon entropy 和 Maximum Lyapunov Exponent，共 12 项。
4. 采用线性 β 调度的 DDPM：`β1=1e-4`、`βT=0.02`、`T=1000`。
5. DiT 使用 patch embedding、12 个 Transformer block、8 头注意力、`d_model=256`；扩散时刻与 SOH 条件通过 AdaLN 的 scale/shift/residual gate 注入。
6. 训练目标为加入噪声与预测噪声之间的 RMSE。

## NASA 统一任务适配

`data/version3` 始终只读。B0006、B0007、B0018 训练，B0005 测试；训练电池按时间 80%/20% 划分训练/验证；窗口从论文的 40 改成统一协议的 16，patch size 4 保持不变；执行真实历史驱动的滚动单步预测，在 SP50/SP90 评估。

干净扩散对象 `X0` 定义为长度 16、末端为下一周期目标的直接容量轨迹，并使用训练电池训练段的全局均值/标准差变换到适合标准高斯扩散的尺度。依据原文第 4.4 节，完整历史容量窗口进入 input/embedding stage；此前 16 周期的 12 项健康特征和最后一个已观测 SOH 同时作为条件。反向过程从纯高斯噪声直接生成标准化容量轨迹，最后仅执行固定的全局仿射逆变换；不使用上一周期容量加残差，也不使用 inpainting。

论文以 80% SOH 定义 EOL；为保证与仓库内现有模型的任务和指标严格一致，本次 NASA 对比实验仅在评估层改用统一的 70% EOL 阈值，模型结构与训练 loss 不因该阈值改变。

## 原文未披露内容

论文未给出 SG window/polyorder、entropy bins、MLE 数值估计算法、优化器、训练 epoch、采样次数、完整输入输出张量定义，以及输出 variance 分支的独立监督方式。本实现固定采用 SG(11,3)、32 bins、Rosenstein 型最近邻轨迹发散斜率、Adam、最大 200 epoch、默认 5 次采样。验证噪声固定；验证 noise RMSE 连续 20 epoch 未改善至少 `1e-6` 时早停，并恢复验证集最优 checkpoint，测试电池不参与停止判断。保留论文图示的 noise/variance 双输出头，但按标准 DDPM posterior variance 采样，只用论文明确给出的 noise RMSE 监督。结果 JSON 逐项保存这些设置和任务适配，不能将它们误称为作者公开参数。

## 运行

```powershell
python Compare-Models/run_sg_dits.py --seeds 7 --device cpu --max-epochs 2 --samples 1 --output-root outputs/sg_dits_smoke
python Compare-Models/run_sg_dits.py --seeds 7 --device cpu --samples 5 --output-root outputs/sg_dits_nasa_200ep
```

CALCE 与 TJU（默认最大 200 epoch）：

```powershell
python Compare-Models/run_sg_dits.py --dataset calce --seeds 7 --device cuda --samples 5 --output-root outputs/sg_dits_calce_200ep
python Compare-Models/run_sg_dits.py --dataset tju --seeds 7 --device cuda --samples 5 --output-root outputs/sg_dits_tju_200ep
```

CALCE 仍从逐采样点 CC-CV 原始曲线计算原文 12 项特征。version3 TJU 仅发布逐循环的 16 项充电统计量而无逐采样曲线，因此选取其中 12 项充电指标作为同宽条件输入；DiT、AdaLN、DDPM、噪声 RMSE 和直接容量生成过程不变。该差异会写入每次运行的 `results.json`，不会把数据适配冒充为原文特征重算。
