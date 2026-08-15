# MSTEA-Net 原文结构审计

审计来源：`D:/class/相关文献/物理双loss.pdf`，Energy 346 (2026) 140288。审计日期：2026-08-11。

## 逐项对照

| 原文位置 | 原文要求 | 当前代码 | 结论 |
|---|---|---|---|
| p.6, Eqs. (11)–(12) | `B×L×D` 健康特征序列经全连接投影到隐藏维 | `input_projection` | 一致 |
| pp.6–7, Eqs. (13)–(17) | LSTM 后并行 `k=1,3,7` Conv1D、GELU、拼接投影、与 LSTM 残差相加、正弦位置编码 | `lstm`、`multiscale`、`multiscale_projection`、`position` | 一致 |
| p.7, Eqs. (18)–(25) | 多头自注意力，随后两次 residual + LayerNorm，含 FFN | `attention`、`attention_norm`、`ffn_norm` | 一致。原文称 cross-variable attention，但公式实际在时间索引 hidden states 上做标准自注意力；代码按公式实现 |
| p.7, Eqs. (26)–(28) | 取最后时间步，`Dhidden→2Dhidden→Dout` 两层预测头，中间 GELU | `prediction_head` | 一致 |
| pp.7–8, Eqs. (29)–(37) | MSE + Arrhenius 一致性 + 相邻周期退化率约束 | `TripleCompositeLoss` | 一致；导数项只对同一电池相邻周期计算 |
| p.8, Table 3 | hidden=32、heads=4、lr=0.01、window=64、CALCE 500 epoch/TJU 300 epoch | `PaperConfig` | hidden、heads 一致；NASA 训练电池验证集选择 lr=3e-4；window 按统一任务改为16；epoch 按用户要求改为80 |
| p.8 | `Ea=0.65 eV`、`b=1.5`、`λarr=λderiv=1e-4` | `TripleCompositeLoss` 默认值 | 一致 |
| pp.4–6 | 四项 CALCE 健康特征→STL→RF/Lasso voting→RFECV | `features.py` | 算法链一致；NASA 的充放电阶段识别和缺失处理属于数据集适配 |

## 审计结论

MSTEA-Net 主干和三项复合 loss 没有结构性错配。当前精度问题主要不是少层、错卷积核或漏掉物理 loss，而是跨数据集适配：NASA 与论文 CALCE/TJU 的健康特征分布不同，同时窗口被强制从 64 缩短到 16、训练预算从 300/500 缩短到 80。

当前将容量写成 `C(N)/C(0)` 再恢复 Ah；这使 Eqs. (31)–(35) 代数形式保持不变，但改变了不同电池在数据 MSE 中的相对权重，属于已披露的任务适配，不是论文原始设置。

代码现已禁止任何上一周期容量锚定或残差还原；预测头按照原文 Eqs. (26)–(28) 直接输出目标容量。

原文第 4.1 节说明滑动窗口从目标电池退化序列提取模型输入。NASA adapter 因此把已观测 `C/C(0)` 作为 input projection 的附加序列通道；这不改变主干或直接预测头，但应与 Table 2 的 RFECV 健康特征列表分开披露。
