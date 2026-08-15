# IC2ML 实现与 RUL 比较任务审计

## 审计基准

- 论文：`IC2ML.pdf`，重点核对公式 (1)–(14)、3.1–3.4 节和敏感性实验。
- 作者源码：公开仓库中的 `models/IC2ML.py`、`dataloader.py`、训练器及随仓库发布的 checkpoint 参数形状。
- 公共任务：上级 `MGI-DSSM` 的 NASA/CALCE 留一电池、测试起点、70% EOL 和指标实现。

## 已发现并修复的不一致

| 层级 | 旧实现问题 | 最终实现 |
|---|---|---|
| 输入窗口 | 曾混用 10/32/64，并把“每周期 10 个充电采样点”误当成历史周期数 | 按公共任务固定 NASA=16 个历史周期、CALCE=64 个历史周期；每周期特征维度仍为论文的 10 |
| CALCE 输入 | 使用了 MGI-DSSM 的放电曲线缓存 | 从原始 Excel 重建论文基础 3.6–3.7 V 充电容量增量 |
| NASA 电压段 | benchmark 曾使用非论文基础区间 | NASA/CALCE 统一恢复论文基础 3.6–3.7 V 充电段 |
| 容量归一化 | 额定容量除法后又做训练集 min-max | 严格只除以额定容量 |
| 嵌入层 | 被简化成 `10→256→256` | 恢复作者 `10→128→256`、GELU 和两级 LayerNorm |
| 自注意力 | 被改成单头且无 dropout | 恢复两头、dropout=0.1 和位置编码 dropout |
| CNN | 被简化为一级 Inception | 恢复 `1→64→128` 两级 Inception 和中间 MaxPool |
| 预测头 | SOH、轨迹、RUL 均被简化成单线性层 | 恢复作者发布的三个 MLP 头 |
| 跨模态注意力 | 全局池化后只有一个 K/V token，softmax 恒为 1 | 按论文式 (2)、(7) 保留二维网格 token，再由 inter-cycle 向量查询 |
| 残差 | 位置编码被同时加入残差支路 | 按论文式 (6) 使用原始 intra embedding 作为残差 |
| 轨迹监督 | 直接跨电池回归下一周期绝对容量，产生明显的电池间系统偏置 | 预测相对最后一个实测容量的单步变化量，再以该实测值重建绝对容量；保持因果性、测试划分和预测目标不变 |
| RUL 标签 | 曾固定除以 100，且相对公共 EOL 规则偏 1 周期 | 与 MGI 连续两点越阈规则对齐；训练时仅做固定尺度数值稳定化（NASA=200、CALCE=1000），评估时还原为周期 |
| 验证划分 | 对生成后的窗口取最后 15% | 先按每块电池原始周期取最后 15%，再生成窗口 |
| 优化器 | AdamW、5e-4 | 恢复作者 Adam、1e-4 |
| 训练上限 | 60 epoch | 使用发布 checkpoint 标注的 500 epoch，patience=100 |
| RUL 评估 | 训练了直接 RUL 头但完全不保存/评估 | 同时保存阈值 RUL 与 direct-RUL MAE/RMSE |

## 最终张量和监督关系

```text
输入 X (NASA):          [batch, 16 cycles, 10 charge samples]
输入 X (CALCE):         [batch, 64 cycles, 10 charge samples]
Intra embedding:        [batch, context, 256]
SOH 输出/标签:          [batch, context]      capacity / rated_capacity
Inter-cycle embedding:  [batch, 256]
轨迹输出/标签:          [batch, 1]            next normalized capacity - last observed normalized capacity
2-D Inception tokens:   NASA [batch, 135, 256]；CALCE [batch, 567, 256]
Cross-attention 输出:   [batch, 256]
Direct RUL 输出/标签:   [batch]               remaining cycles / dataset_fixed_scale
```

训练损失保持论文权重：

```text
Loss = MSE(SOH) + MSE(capacity_delta) + 0.5 * MSE(scaled_direct_RUL)
```

## 公共任务的必要适配

论文基准预测未来 50 个容量点；MGI-DSSM 比较任务要求每个测试周期的一步容量预测，并由整条预测序列计算阈值 RUL。因此最终把可配置的 trajectory horizon 设为 1。绝对容量按 `last_observed_capacity + predicted_delta` 重建。这个适配不改变 IC2ML 编码器、三任务训练、数据划分、测试起点或阈值定义。

公共设置如下：

- 留一电池：其余三块训练，测试电池不参与归一化、训练或选模。
- 历史窗口：NASA=16，CALCE=64；每个周期均为 10 维增量充电特征。
- NASA：额定容量 2.0 Ah，测试起点 50/70/90。
- CALCE：额定容量 1.1 Ah；CS2_35/36 从 200，CS2_37/38 从 300 开始。
- EOL/RUL：严格采用 MGI-DSSM 最终十次实验的对称连续阈值规则；真实和预测容量曲线均要求连续两个点不高于 `0.7 * rated_capacity`。
- 容量指标：MAE、RMSE、R²；阈值 RUL 指标：AE、RE。
- checkpoint 按验证容量 MAE 选择，因为这是公共比较任务的主输出；direct RUL 作为 IC2ML 专有辅助输出另行报告。

## 数据完整性

- NASA 保留 B0005/B0006/B0007/B0018 的 168/168/168/132 个官方放电周期编号。
- CALCE 对齐后保留 CS2_35/36/37/38 的 882/936/972/996 个官方周期。
- CALCE 原始充电曲线与 summary 的顺序对齐 MAE 为 0.00308–0.00337 Ah。
- 3.6–3.7 V 下 CS2_35/36/37/38 分别有 154/219/179/462 个缺失充电段（17.5%/23.4%/18.4%/46.4%），全部只使用同电池更早周期前向填充；无更早值时使用零向量。
- 充电增量与容量的最大绝对相关系数为 0.777/0.449/0.648/0.468；该区间在 CALCE 上的覆盖限制已在结果中披露。

## 最终验证

- 80 次正式运行全部存在 checkpoint、`results.json` 和逐周期 `predictions.csv`。
- 所有正式配置均为 NASA context=16、CALCE context=64、epochs=500、capacity-MAE checkpoint selection。
- 所有结果文件均记录 `trajectory_mode=observed_last_capacity_plus_predicted_delta`，并记录对应数据集的 `rul_scale_cycles`。
- 从 CSV 独立重算的 MAE、RMSE、R²、阈值 RUL AE/RE、direct-RUL MAE/RMSE 与 JSON 在 `1e-9` 内一致。
- cross-attention 单元测试确认：固定二维 tokens、改变 inter-cycle context 后输出发生变化，不再退化为常量权重映射。

完整结果见 `result.md`，逐 seed 产物见 `benchmark_results_etsformer_seq16_64_paper36_37/`。
