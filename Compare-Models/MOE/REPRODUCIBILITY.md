# 论文—源码逐项核对

## 已按原文明确定义实现

| 论文位置 | 要求 | 源码 |
|---|---|---|
| 式 1-2 | 非重叠多尺度 patch、独立线性投影、中心物理时间戳、patch 最小有效 mask | `MultiScaleTokenizer` |
| 式 3-4 | 各尺度均值、跨尺度求和、共享两层 MLP、Sigmoid 后逐通道回写 | `CrossScaleSE` |
| 式 5、图 1 | 分组、1x1 上下文门控、3x3 局部分支、Softmax-MatMul 双路融合、`softplus(theta)` 全局门 | `CrossTimeReweighting` |
| 式 6 | Pre-RMSNorm，epsilon 为 1e-6；RoPE 自注意力残差后接 MoE 残差 | `EncoderLayer` |
| 图 1、式 7-8 | Up/Gate、SiLU、Hadamard、Down；token 级 Softmax Top-k；路由专家宽度 `dff/k`；全宽共享专家及独立 Sigmoid 门 | `GatedExpert`、`SparseSharedMoE` |
| 式 8 | Top-k 权重使用原始 Softmax 概率，不做二次归一化 | `SparseSharedMoE.forward` |
| 式 9 | 找到最新有效物理时间戳并平均全部同时间 token | `BATTERMoE.forward` |
| 式 10 | 各层路由概率均值对均匀分布的平方偏差，再跨层平均 | `SparseSharedMoE.forward` |
| 第 IV-A 节 | 3σ 孤立点、线性插值、保留非孤立容量波动、容量 `C/C0`、TJU 其余特征 Min-Max | `data.py` |
| 表 II | 留出完整测试电池；其余电池 80/20 训练验证；测试电池不参与统计量 | `prepare_data` |
| 表 III | Adam、1e-3、batch 128、MAE early stop patience 10、各数据集模型规模 | `config.py`、`train.py` |
| 第 IV-B/C 节 | 真实历史窗口的一步预测；MAE/RMSE/R2；阈值首次穿越换算 RUL 与 RE | `metrics.py`、`run_experiment.py` |

参数量实测为 NASA 94,054、GOTION 1,065,674、TJU 5,399,698；论文报告约 94.5K、1.1M、5.4M。GOTION 和 TJU 与论文舍入值一致，NASA 相差约 0.47%。没有添加无原文依据的参数来强行凑数。

## 原文未充分披露的必要假设

这些选项都集中在配置或注释中，没有冒充论文原始设定：

- 注意力头数未报告：默认 8。
- CS-SE reduction ratio 数值未报告：默认 16。
- CT-Reweighting 分组数未报告：默认 8；内部无公式，按图 1 和段落描述重建。
- 式 (5) 给出了 `gamma=softplus(theta)`，但未报告 `theta` 初始化：默认直接将可学习参数 `theta` 初始化为 1；NASA seed 0--9 的诊断结果优于旧的零初始化，且平均停止轮次更接近论文表 VII。
- 论文只说 scale embedding 可启用，未说明最终实验是否启用：默认启用。
- 训练/验证只说明 80/20，未说明是否打乱：固定随机种子打乱窗口。
- 3σ 孤立点未给检测伪代码：使用五点滚动中值建立局部趋势，截尾残差估计 sigma，仅删除相邻点不同时异常的孤立点。
- 最大 epoch 和十次实验种子值未报告：上限默认 300，种子默认 0-9；patience 10 仍严格按论文。
- 预测未穿过阈值时的处理未报告：在观测终点后一循环右删失。
- 本地 TJU 有三个实验条件、每个三重复。论文只写 CY25-1/2/3；默认采用 `CY25-05_1` 的 #1/#2/#3 三重复。

## 当前材料限制

论文使用 NASA、GOTION、TJU，但当前 `data` 目录没有 GOTION 数据。源码没有用 RWTH 或 CALCE 替代 GOTION，因为这会改变论文任务。NASA 和 TJU 已完成真实数据加载与端到端 smoke test；GOTION 模型配置和通用 CSV 读取器已实现，需补数据后才能验证论文划分和指标。
