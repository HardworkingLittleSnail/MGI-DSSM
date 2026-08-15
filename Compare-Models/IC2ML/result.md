# IC2ML NASA-16 / CALCE-64 实验结果

> 结果状态：**已确认并冻结**（2026-08-03）。本文件记录恢复论文基础输入区间 3.6–3.7 V 后的正式结果、实验协议和复现配置；正式产物目录为 `benchmark_results_etsformer_seq16_64_paper36_37/`。

## 实验配置

- Python 环境：`D:\ChenLi\Anaconda\anaconda\envs\etsformer\python.exe`
- PyTorch：2.11.0+cu128，CUDA 可用
- NASA 输入窗口：16 个历史周期
- CALCE 输入窗口：64 个历史周期
- 每周期输入：论文基础配置 3.6–3.7 V 充电区间的 10 点容量增量
- 任务：下一周期容量预测 + 阈值 RUL + IC2ML direct RUL
- 数据划分：leave-one-battery-out；训练电池原始周期序列最后 15% 用作验证集
- EOL/RUL：与 MGI-DSSM 最终十次实验一致，真实和预测容量曲线均以连续两个点不高于额定容量70%作为EOL
- NASA 额定容量：2.0 Ah；测试起点：50、70、90
- CALCE 额定容量：1.1 Ah；CS2_35/36 从 200，CS2_37/38 从 300 开始
- 优化器：Adam，学习率 1e-4，batch size 128
- 最大训练轮数：500；early-stopping patience：100
- checkpoint：按验证容量 MAE 选择
- 随机种子：`7,17,27,37,47,57,67,77,87,97`
- 轨迹流程：预测下一周期容量变化量，并以已观测的上一周期容量为锚点还原绝对容量
- RUL 损失尺度：NASA 除以 200、CALCE 除以 1000；保存和评估时还原为周期
- 损失：`MSE(SOH) + MSE(capacity delta) + 0.5*MSE(scaled direct RUL)`

## 数据与模型协议

### 数据输入与因果边界

- NASA 电池：B0005、B0006、B0007、B0018；CALCE 电池：CS2_35、CS2_36、CS2_37、CS2_38。
- 每次以一块电池为测试集，其余三块为训练/验证数据；测试电池不参与训练、验证、归一化或 checkpoint 选择。
- 每个周期从论文基础配置 3.6–3.7 V 充电段提取 10 点增量充电容量特征。CALCE 使用由原始 Excel 对齐生成的 `data/CALCE data/CALCE_IC2ML_charge_3.6-3.7.npy`。
- 输入只包含目标周期之前的历史周期。NASA 使用前 16 个周期，CALCE 使用前 64 个周期，不使用未来容量或未来充电曲线。
- 容量只除以数据集额定容量，不做训练集 min-max 或测试电池统计量归一化。
- 训练电池按原始时间顺序切分，尾部 15% 为验证区间；先切分周期，再分别生成滑动窗口，避免窗口级随机切分造成时序泄漏。

### IC2ML 模型配置

| 模块 | 最终配置 |
|---|---|
| Intra-cycle embedding | `10 → 128 → 256`，GELU，LayerNorm |
| 位置编码 | 正弦位置编码，dropout=0.1 |
| Inter-cycle attention | 2-head self-attention，hidden dim=256，dropout=0.1 |
| 时序汇聚 | `context × 256 → 256`，GELU，LayerNorm |
| Inception 分支 | 两级 `1 → 64 → 128`，中间 MaxPool；卷积核包含 1×1、3×3、5×5 |
| Cross-attention | inter-cycle 向量为 query，二维 Inception 网格为 key/value |
| SOH head | `256 → 128 → 1`，逐历史周期输出 |
| Trajectory head | `256 → 256 → 1`，预测下一周期归一化容量变化量 |
| Direct-RUL head | `256 → 128 → 1`，dropout=0.2 |
| 预测 horizon | 1 个周期 |

下一周期容量严格按以下因果关系还原：

```text
q_delta_pred(k+1) = trajectory_head(history[≤k])
Q_pred(k+1) = Q_observed(k) + rated_capacity * q_delta_pred(k+1)
```

### 训练与选模协议

| 配置 | NASA | CALCE |
|---|---:|---:|
| 历史窗口 | 16 | 64 |
| 额定容量 | 2.0 Ah | 1.1 Ah |
| RUL 训练尺度 | 200 cycles | 1000 cycles |
| 测试起点 | 50、70、90 | CS2_35/36：200；CS2_37/38：300 |
| batch size | 128 | 128 |
| optimizer | Adam | Adam |
| learning rate | 1e-4 | 1e-4 |
| 最大 epochs | 500 | 500 |
| early-stopping patience | 100 | 100 |
| checkpoint objective | validation capacity MAE | validation capacity MAE |

- 随机性控制：Python、NumPy、PyTorch CPU/CUDA 使用相同 seed；cuDNN deterministic 开启，benchmark 关闭。
- 十个正式 seed：`7, 17, 27, 37, 47, 57, 67, 77, 87, 97`。
- Direct-RUL 是论文多任务辅助头；公共 RUL 主结果由逐周期容量预测序列与 70% EOL 规则计算。

### 指标定义与汇总规则

设测试容量为 `Q_i`、预测为 `Qhat_i`、样本数为 `N`：

```text
MAE  = mean(|Q_i - Qhat_i|)
RMSE = sqrt(mean((Q_i - Qhat_i)^2))
R²   = 1 - sum((Q_i - Qhat_i)^2) / sum((Q_i - mean(Q))^2)
AE   = |RUL_real - RUL_pred|
RE   = min(AE / max(|true_crossing_index|, 1), 1)
Persistence MAE = mean(|Q_i - Q_(i-1)|)
vs Persistence = (Persistence MAE - Model MAE) / Persistence MAE × 100%
```

- 严格采用 MGI-DSSM `outputs/final_10runs/summary_report.md` 的最终协议：真实和预测容量曲线均必须连续两个点不高于 `0.7 × rated_capacity`，即 symmetric consecutive-threshold EOL rule。
- 每个 NASA seed 先对测试起点 50/70/90 的指标取算术平均；CALCE 每个 seed 只有一个规定起点。
- 随后在十个 seed 上汇总。`mean` 是十 seed 算术平均；`best` 为该指标的十 seed 最优值：MAE/RMSE/AE/RE 取最小，R² 取最大。
- 因此同一行不同指标的 `best` 不保证来自同一个 seed；这不是挑选某个整体最优 checkpoint。每个 seed 的 checkpoint 仅由训练电池验证集容量 MAE 决定。

### 3.6–3.7 V 输入覆盖说明

- NASA 四块电池分别保留 168/168/168/132 个周期，全部输入有限；每块电池仅首周期因无更早可用充电段使用因果零填充。
- CALCE 在论文基础区间的可用性低于原先的 3.9–4.0 V 变体。CS2_35/36/37/38 的缺失充电段分别为 154/219/179/462，占 17.5%/23.4%/18.4%/46.4%。
- 缺失段只使用同一电池更早周期的最后有效曲线前向填充；最前端无历史值时才使用零向量，因此不使用未来周期。
- CALCE 最长连续缺失段分别为 154/182/148/148 个周期。该覆盖限制属于数据条件，可能影响结果，未通过改回 3.9–4.0 V 或使用未来曲线填补来规避。

## 按电池汇总（十种子平均）

NASA 每个 seed 先对起点 50/70/90 的指标取算术平均，再纳入十种子平均；CALCE 使用每块电池规定的单一起点。容量指标单位为 Ah，RUL 指标单位为 cycles。`vs Persistence` 为正表示优于上一周期容量基线，为负表示弱于该基线。

| Dataset | Battery | MAE | RMSE | R² | RUL real | RUL pred | Ecycle/AE | RE | Persistence MAE | vs Persistence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NASA-16 | B0005 | 0.006386 | 0.013379 | 0.9791 | 55.00 | 55.00 | 0.00 | 0.0000 | 0.008323 | +23.27% |
| NASA-16 | B0006 | 0.009319 | 0.020342 | 0.9664 | 39.00 | 39.90 | 0.90 | 0.0297 | 0.011914 | +21.78% |
| NASA-16 | B0007 | 0.005956 | 0.013996 | 0.9642 | 100.00 | 100.00 | 0.00 | 0.0000 | 0.007362 | +19.09% |
| NASA-16 | B0018 | 0.011359 | 0.021904 | 0.6973 | 27.00 | 28.00 | 1.00 | 0.0756 | 0.013646 | +16.76% |
| CALCE-64 | CS2_35 | 0.005464 | 0.012400 | 0.9965 | 441.00 | 441.80 | 1.00 | 0.0023 | 0.004578 | -19.37% |
| CALCE-64 | CS2_36 | 0.005730 | 0.011008 | 0.9982 | 446.00 | 444.30 | 3.10 | 0.0070 | 0.005016 | -14.23% |
| CALCE-64 | CS2_37 | 0.005177 | 0.009342 | 0.9979 | 417.00 | 417.80 | 2.40 | 0.0058 | 0.004483 | -15.49% |
| CALCE-64 | CS2_38 | 0.005287 | 0.010540 | 0.9973 | 458.00 | 459.00 | 1.00 | 0.0022 | 0.004449 | -18.83% |

## NASA：context = 16

| Battery | Runs | MAE best / mean (Ah) | RMSE best / mean (Ah) | R² best / mean | Threshold-RUL AE best / mean (cycles) | Threshold-RUL RE best / mean |
|---|---:|---:|---:|---:|---:|---:|
| B0005 | 10 | 0.006298 / 0.006386 | 0.013291 / 0.013379 | 0.9794 / 0.9791 | 0.00 / 0.00 | 0.0000 / 0.0000 |
| B0006 | 10 | 0.009222 / 0.009319 | 0.020328 / 0.020342 | 0.9664 / 0.9664 | 0.00 / 0.90 | 0.0000 / 0.0297 |
| B0007 | 10 | 0.005944 / 0.005956 | 0.013970 / 0.013996 | 0.9643 / 0.9642 | 0.00 / 0.00 | 0.0000 / 0.0000 |
| B0018 | 10 | 0.011038 / 0.011359 | 0.021767 / 0.021904 | 0.7015 / 0.6973 | 1.00 / 1.00 | 0.0756 / 0.0756 |

## CALCE：context = 64

| Battery | Runs | MAE best / mean (Ah) | RMSE best / mean (Ah) | R² best / mean | Threshold-RUL AE best / mean (cycles) | Threshold-RUL RE best / mean |
|---|---:|---:|---:|---:|---:|---:|
| CS2_35 | 10 | 0.005079 / 0.005464 | 0.012236 / 0.012400 | 0.9966 / 0.9965 | 1.00 / 1.00 | 0.0023 / 0.0023 |
| CS2_36 | 10 | 0.005461 / 0.005730 | 0.010927 / 0.011008 | 0.9982 / 0.9982 | 1.00 / 3.10 | 0.0022 / 0.0070 |
| CS2_37 | 10 | 0.004929 / 0.005177 | 0.009175 / 0.009342 | 0.9980 / 0.9979 | 2.00 / 2.40 | 0.0048 / 0.0058 |
| CS2_38 | 10 | 0.005017 / 0.005287 | 0.010359 / 0.010540 | 0.9974 / 0.9973 | 1.00 / 1.00 | 0.0022 / 0.0022 |

## IC2ML direct RUL 头（辅助结果，不纳入MGI-DSSM统一指标）

| Dataset | Battery | Direct RUL MAE best / mean (cycles) | Direct RUL RMSE best / mean (cycles) |
|---|---|---:|---:|
| NASA-16 | B0005 | 39.82 / 67.19 | 43.26 / 70.07 |
| NASA-16 | B0006 | 26.69 / 47.98 | 28.47 / 49.72 |
| NASA-16 | B0007 | 24.79 / 29.23 | 28.72 / 34.62 |
| NASA-16 | B0018 | 8.68 / 58.68 | 12.60 / 60.39 |
| CALCE-64 | CS2_35 | 59.85 / 93.67 | 87.96 / 128.01 |
| CALCE-64 | CS2_36 | 34.53 / 54.44 | 45.95 / 71.11 |
| CALCE-64 | CS2_37 | 26.67 / 43.84 | 35.55 / 67.43 |
| CALCE-64 | CS2_38 | 45.85 / 57.49 | 62.76 / 80.34 |

## 结果文件与复算验证

正式运行命令（PowerShell）：

```powershell
& 'D:\ChenLi\Anaconda\anaconda\envs\etsformer\python.exe' run_rul_benchmark.py --dataset nasa --seq-len 16 --voltage-start 3.6 --voltage-end 3.7 --output-root benchmark_results_etsformer_seq16_64_paper36_37
& 'D:\ChenLi\Anaconda\anaconda\envs\etsformer\python.exe' run_rul_benchmark.py --dataset calce --seq-len 64 --voltage-start 3.6 --voltage-end 3.7 --output-root benchmark_results_etsformer_seq16_64_paper36_37
```

每个 seed 均保存：

```text
benchmark_results_etsformer_seq16_64_paper36_37/<dataset>/<battery>/seed_<seed>/
├── checkpoint.pth
├── results.json
└── predictions.csv
```

共检查 80 个正式 run，包含 80 个 `checkpoint.pth`、80 个 `results.json` 和 80 个 `predictions.csv`。所有 `results.json` 的解释器路径均包含 `etsformer`，记录输入区间 3.6–3.7 V、NASA `seq_len=16`、CALCE `seq_len=64` 以及 `observed_last_capacity_plus_predicted_delta` 轨迹模式。从逐周期 `predictions.csv` 独立重算 MAE、RMSE、R²、RUL real/pred、Ecycle/AE、RE、Persistence MAE 以及 direct-RUL MAE/RMSE，与 JSON 保存值的误差均不超过 `1e-9`。

## 异常结果根因与修复

旧流程让轨迹头直接回归绝对容量，导致跨电池固定偏置。例如 NASA B0006 的十种子平均偏置约为 −0.099 Ah，而其 persistence MAE 仅约 0.0119 Ah；这说明误差来自容量基准没有对齐，而非真实退化变化无法学习。CALCE 也存在同样问题。

最终流程保留原任务中的历史观测，把上一周期真实容量作为当前 SOH 锚点，模型只预测下一周期容量变化量：

```text
Q_pred(k+1) = Q_observed(k) + Delta_Q_model(k+1)
```

这仍然是相同的历史窗口到下一周期容量任务，没有改变测试起点、数据切分、标签、EOL 或指标。另一个问题是 CALCE 原始 direct-RUL MSE 可达到约 131,991，进入共享编码器的梯度大于容量分支。训练时对 RUL 周期数做固定尺度归一化、评估时还原，避免辅助任务破坏容量表示。
