# 六个对比模型的统一实验协议

## 1. 公平性边界

统一实验只统一数据与任务，不统一模型本身。四个模型均使用 `data/processed` 下经过同一异常点处理流程得到的容量序列，采用留一电池测试、其余电池按时间顺序划分 80%/20% 训练集与验证集，并执行真实历史窗口驱动的单步容量预测。NASA、CALCE 和 TJU 的窗口分别为 16、64 和 64，早/晚预测起点分别为 50/90、200/400 和 200/400。正式实验使用 seed 7、17、27、37、47、57、67、77、87 和 97。

模型专属的网络结构、输入形式、归一化方式、损失函数和论文给出的超参数保持独立。测试电池不参与训练、验证或归一化统计量估计。一个模型检查点同时评估早、晚两个起点，避免针对测试起点重复训练或选择模型。

## 2. 论文—实现对应关系

| 模型 | 保留的原生输入 | 保留的论文结构与目标 | 统一协议下的改动 |
|---|---|---|---|
| PatchFormer | 单变量历史容量 | 变量嵌入、DPAN、FAN、投影头；SMAPE | 窗口改为 16/64/64；对论文未报告的全部留出电池和数据集组合执行相同架构 |
| RUL-Mamba | 单变量历史容量 | 变量编码、FAN、Mamba 编码器、Mamba-GRN 解码器；SMAPE | 窗口和留出电池服从统一协议；CALCE 使用论文 NASA 单变量配置作结构迁移 |
| BATTER-MoE | NASA/CALCE 为容量；TJU 为 17 个循环级指标 | 多尺度 patch、跨尺度 SE、跨时间重加权、RoPE 注意力、稀疏共享 MoE、最新时刻池化；MAE 与路由均衡损失 | CALCE 使用紧凑的 NASA 主干并将 patch 尺度适配到 64 步；其余数据集使用论文表格配置 |
| IC2ML | 每循环 0.1 V 区间上的 10 点充电容量增量 | 循环内嵌入、循环间自注意力、二维 Inception 表征、跨模态注意力；SOH/轨迹/RUL 三任务 MSE（1:1:0.5） | 轨迹预测长度设为统一的下一步；历史循环数改为 16/64/64；容量标签与统一处理后的循环严格对齐 |
| Autoformer | 单变量历史容量与因果循环时间标记 | 渐进式趋势—季节分解、FFT Auto-Correlation、时间延迟聚合、编码器—解码器 | 预测长度改为 1，窗口改为 16/64/64；短窗口移动平均核为 5，长窗口为 25 |
| iTransformer | 容量、容量一阶差分、3 周期与 7 周期后向趋势 | 每个退化序列作为 variate token，标准自注意力学习变量间关系，投影头逐变量预测 | 四个输入变量均只由当前及历史容量构造；容量为主损失，其余变量仅作 0.2 权重辅助监督 |

## 3. 数据对齐

- 容量标签及保留循环完全取自统一处理后的 `*_Data_batter_moe_preprocessed.npy`，对比模型不再执行自身论文中的异常点删除。
- PatchFormer 与 RUL-Mamba 直接读取统一容量表。
- BATTER-MoE 的 TJU 17 指标从官方逐点记录提取，随后按处理后表中的 `cycle index` 对齐；非容量指标仅使用训练数据拟合 Min–Max 统计量。
- IC2ML 从官方充电记录提取 3.6–3.7 V 区间的 10 点容量增量，再按处理后循环索引或容量序列单调匹配。缺失充电片段只允许使用同一电池的历史值前向填充，序列开头使用零向量，禁止未来信息回填。
- Autoformer 只读取归一化历史容量；iTransformer 的差分与移动趋势均为严格后向计算，不读取预测时点之后的数据。两者的 Min–Max 参数仅由训练电池的训练段估计。
- 所有容量标签、原始曲线、物理缓存及 IC2ML 原生输入统一位于 `data/version3`；缓存只补充模型输入，不改变统一容量序列。

## 4. 论文未完全披露部分

“完全复现论文”与“在新数据集、全部留出电池和统一窗口上比较”并非同一件事。论文没有报告的模型—数据集组合必须进行结构迁移；部分论文也未披露所有内部超参数或特征统计量的精确定义。本实现优先采用作者发布源码中的默认值，并在结果 JSON 的 `native_config` 或 `paper_config` 中逐项保存。BATTER-MoE 的 17 项 TJU 特征按照论文列出的统计量定义显式计算；IC2ML 的充电区间采用论文基准设置 3.6–3.7 V。上述部分均未用测试集调参。

## 5. 输出结构

每次训练保存在：

```text
outputs/comparison_models_10seeds/<model>/<dataset>/<battery>/seed_<seed>/
```

每个目录包含模型检查点、完整训练历史、逐循环预测和结果配置。根目录还会生成：

- `all_results.csv/json`：所有 seed、所有电池、两个起点的完整指标；
- `mean_std_over_seeds.csv/json`：十次实验的均值和标准差；
- `best_run_by_metric.csv/json`：按 MAE、RMSE、R2 和 RE 分别选择的完整最佳行及对应 seed；
- `_logs/`：可实时查看且永久保存的终端训练日志；
- `protocol.json`：统一任务、数据和 seed 清单。

脚本支持断点续跑。只有包含 `"status": "complete"` 的单次结果才会跳过，未完成目录会重新训练。

## 6. Autoformer 与 iTransformer 训练命令

在项目根目录执行以下命令，可按当前对齐协议对两个模型完成 NASA、CALCE、TJU 的十随机种子训练：

```powershell
python run_seven_models_version3_10seeds.py --models autoformer itransformer --datasets nasa calce tju --device cpu --output-root outputs/seven_models_version3_10seeds
```

仅训练 NASA 与 TJU：

```powershell
python run_seven_models_version3_10seeds.py --models autoformer itransformer --datasets nasa tju --device cpu --output-root outputs/seven_models_version3_10seeds
```

若环境中的 PyTorch 支持 CUDA，将 `--device cpu` 改为 `--device cuda`。总调度脚本对三个数据集均显式选择 `version3`，并在已有完整结果时断点跳过。
