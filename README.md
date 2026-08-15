<<<<<<< HEAD
# 七模型电池容量/RUL统一实验（version3）

本项目在同一数据与评估协议下训练七个模型：MGI-DSSM（our_model）、PatchFormer、RUL-Mamba、BATTER-MoE、IC2ML、Autoformer 和 iTransformer。正式入口为 `run_seven_models_version3_10seeds.py`，所有训练数据统一从 `data/version3` 读取。

## 1. version3 数据目录

```text
data/version3/
├── manifest.json
├── NASA data/
│   ├── NASA_Data_minimal_interpolated.npy
│   ├── B0005.mat, B0006.mat, B0007.mat, B0018.mat
│   ├── raw_discharge_curves_batter_moe_v1.npy
│   └── physics_curve_cache_*.npz
├── CALCE data/
│   ├── CALCE_Data.npy
│   ├── CS2_35/, CS2_36/, CS2_37/, CS2_38/
│   ├── raw_discharge_curves_batter_moe_v1.npy
│   └── physics_curve_cache_*.npz
├── TJU data/
│   ├── Dataset_3_NCM_NCA_battery_1C.npy
│   ├── TJU_Data_version2_model_adapter.npy
│   ├── Dataset_3_NCM_NCA_battery/
│   └── physics_curve_cache_*.npz
└── native_inputs/ic2ml/version3/
```

`manifest.json` 记录三个容量主文件的来源、SHA-256、训练/测试电池和固定随机种子。旧的 `data/processed` 与 `data/processed-version2.0` 仅保留用于历史结果复现；version3 正式训练不会在两者之间切换。

## 2. 数据集处理方式

本节描述的是从官方原始文件到 version3 离线数据文件的实际转换。CALCE 和 TJU 严格采用 PatchFormer 发布源码中的处理方式；NASA 先按 PatchFormer 方式提取放电容量，再只修复 7 个事先指定的孤立上冲点。离线处理和训练时归一化是两个不同步骤，不能混写。

### 2.1 CALCE：与 PatchFormer 相同的预处理

官方输入是 CS2_35、CS2_36、CS2_37、CS2_38 四块电池分批保存的 Excel 文件。version3 中分别保留 25、26、27、28 个工作簿，处理代码对应 `Compare-Models/PatchFormer/CALCEDataPreProcess.py`。

1. 对每个工作簿读取第 2 个 worksheet（`sheet_name=1`），取首行 `Date_Time`；同一电池的全部工作簿按该时间升序排列，而不是按文件名排列。
2. 在每个工作簿内按 `Cycle_Index` 分组。
3. 充电阶段定义为 `Step_Index=2`（CC）或 `Step_Index=4`（CV）：
   - `CCCT = max(Test_Time)-min(Test_Time)`，只在 Step 2 内计算；
   - `CVCT = max(Test_Time)-min(Test_Time)`，只在 Step 4 内计算。
4. 放电阶段只取 `Step_Index=7`。若该阶段存在电流记录，则用相邻采样时刻积分容量：

   ```text
   Δt = diff(Test_Time)
   Q_k = Σ(Δt × Current) / 3600
   Capacity = -Q_end
   ```

   电流在原始文件中为负，因此最终乘 `-1` 得到 Ah 为正的放电容量。
5. 同时保存 PatchFormer 的辅助量：
   - `SoH`：放电电压最接近 3.8 V 与 3.4 V 两点之间的容量差；
   - `Resistance`：该放电阶段 `Internal_Resistance(Ohm)` 的均值；
   - `CCCT`、`CVCT`：上述恒流/恒压充电时长。
6. 对放电容量执行 PatchFormer 原代码的分块 2σ 筛选。它不是滑动 3σ：从数组下标 1 开始，每 40 个点构成一个块，在块内计算均值 `μ` 和总体标准差 `σ`，仅保留严格满足 `μ-2σ < Capacity < μ+2σ` 的点。原实现只处理完整块，因此首点、末尾不足 40 点的尾段以及块内越界点均不进入输出。
7. 将保留下来的行按当前顺序重新编号为 `Cycle=1...N`，最终保存 `BatteryName, Cycle, Capacity, SoH, Resistance, CCCT, CVCT` 到 `CALCE_Data.npy`。容量仍以 Ah 保存，此阶段不做额定容量归一化。

下面直接报告我们 version3 数据及 MGI-DSSM 实际使用的循环数。这里的“官方原始”是从官方 Excel 成功识别并积分得到容量的有效放电循环，不包含 charge/CV 等非放电步骤；“模型实际输入”就是 `CALCE_Data.npy` 中真正送入当前实验的记录数。

| 电池 | 实验角色 | 官方原始有效放电循环 | version3/模型实际输入循环 | 相差循环 | 相差比例 |
|---|---|---:|---:|---:|---:|
| CS2_35 | 测试 | 932 | 882 | -50 | -5.36% |
| CS2_36 | 训练/验证 | 973 | 936 | -37 | -3.80% |
| CS2_37 | 训练/验证 | 1038 | 972 | -66 | -6.36% |
| CS2_38 | 训练/验证 | 1078 | 996 | -82 | -7.61% |
| 合计 | 3 块训练池 + 1 块测试 | 4021 | 3786 | -235 | -5.84% |

因此，我们实际持有 3786 个 CALCE 循环：训练/验证电池 CS2_36/37/38 共 2904 个，留出测试电池 CS2_35 共 882 个。MGI-DSSM 使用 64 周期历史窗口，并在每块训练电池末尾保留 15% 作验证：

| 电池 | version3 循环 | 前 85% 拟合统计量的循环 | 训练窗口 | 验证窗口 |
|---|---:|---:|---:|---:|
| CS2_36 | 936 | 795 | 731 | 141 |
| CS2_37 | 972 | 826 | 762 | 146 |
| CS2_38 | 996 | 846 | 782 | 150 |
| 合计 | 2904 | 2467 | 2275 | 437 |

CS2_35 的 882 个循环全部保留为测试轨迹，不参与参数拟合或归一化统计；按当前 SP=200/400 的评估设置分别形成 683/483 个测试窗口。

version3 的 `CALCE_Data.npy` 与最后使用的 `data/processed-version2.0/CALCE data/CALCE_Data.npy` 字节完全一致，SHA-256 为 `6FDD78C1F0E0B2094841D3887331AD514045B18AA9E55D82C4535DC884656AC4`；归档过程没有再次筛选、插值或平滑。

### 2.2 TJU：与 PatchFormer 相同的预处理

官方输入来自 TJU Dataset 3。PatchFormer 只选择 NCM/NCA 电池的 `CY25-05_1-#1.csv`、`#2.csv`、`#3.csv` 三个 1C 文件，分别映射为 CY25_1、CY25_2、CY25_3；`CY25-05_2-*` 和 `CY25-05_4-*` 不进入当前实验。处理代码对应 `Compare-Models/PatchFormer/TJUDataPreProcess.py`。

1. 读取每个 CSV；在原数据倒数第 1 列之前插入零基的 `cycle index=0...N-1`，用于保留被筛选前的官方行位置。
2. 将正负无穷替换为 NaN，并删除任何字段含 NaN 的整行。
3. 对当前 DataFrame 的每个数值列分别计算全序列均值与样本标准差。某行只要在任意一列满足 `x < μ-3σ` 或 `x > μ+3σ`，就加入删除集合；最后一次性删除所有列的异常行。这里包括插入的 `cycle index`、16 个循环级统计量和 `capacity`。
4. 删除后不插值、不回填；按保留顺序重新编号 `Cycle=1...N`。原始 `cycle index` 继续保留，因此仍可追溯到官方 CSV 的行号。
5. 将小写 `capacity` 改名为 `Capacity`，加入 `BatteryName`，保存为 `Dataset_3_NCM_NCA_battery_1C.npy`。
6. 保留的 16 个非容量特征为：电压 mean/std/kurtosis/skewness、CC Q、CC charge time、voltage slope/entropy、电流 mean/std/kurtosis/skewness、CV Q、CV charge time、current slope/entropy。容量仍以 Ah 保存。

实际筛选数量如下：

| 电池 | 官方 CSV 行数 | NaN/Inf 删除 | 任一列全局 3σ 删除 | 最终循环数 |
|---|---:|---:|---:|---:|
| CY25_1 | 902 | 0 | 16 | 886 |
| CY25_2 | 918 | 0 | 14 | 904 |
| CY25_3 | 954 | 0 | 17 | 937 |

为适配 MGI-DSSM，只额外生成 `TJU_Data_version2_model_adapter.npy`：把键名 CY25_1/2/3 改为 CY25-1/2/3，并补齐统一的 `BatteryName`、`Cycle` 字段；不改变任何容量值。原文件、version3 文件以及适配文件中的容量逐元素完全一致。version3 主文件 SHA-256 为 `0704EF1B4AB712A9461EFFED349F2B442C5F720E370FE5F52CFEE7CA35FD8B1E`。

### 2.3 NASA：PatchFormer 提取方式 + 7 点最小修复

官方输入是 B0005、B0006、B0007、B0018 四个 MATLAB 文件。基础提取与 PatchFormer 的 `NASADataPreProcess.py` 相同：

1. 读取顶层 `cycle` 结构，忽略 charge 和 impedance，只保留 `type=discharge` 的记录。
2. 容量直接采用官方放电记录中的 `data.Capacity`（Ah），不重新对电流积分。
3. 按放电记录出现顺序编号 `Cycle=1...N`，得到 B0005/B0006/B0007 各 168 个循环、B0018 132 个循环，共 636 个循环。

相对于 PatchFormer 的原始提取结果，我们只对 7 个预先固定的孤立上冲点作一次最小修复：用该点前、后两个原始容量的算术平均值替换当前值，公式为 `Q_clean(i)=[Q_raw(i-1)+Q_raw(i+1)]/2`。不删除循环、不更改循环号、不处理端点，也不对其他点作平滑或异常检测。

| 电池 | 周期（1-based） | 原始容量/Ah | 修复后容量/Ah |
|---|---:|---:|---:|
| B0005 | 31 | 1.8518025517 | 1.8173904429 |
| B0005 | 90 | 1.6058188991 | 1.5406675693 |
| B0005 | 151 | 1.3601216767 | 1.3317015545 |
| B0006 | 90 | 1.5935866593 | 1.4940322154 |
| B0007 | 90 | 1.6888211162 | 1.6083223051 |
| B0007 | 151 | 1.4672063485 | 1.4468607842 |
| B0018 | 121 | 1.4268427818 | 1.3763395124 |

修复比例为 `7/636=1.100629%`。处理后增加 `Capacity_SOH=Capacity/2.0`，输出 `NASA_Data_minimal_interpolated.npy`；其余字段和 629 个容量点保持原样。该文件 SHA-256 为 `F97C9C3940835259615024E00E8F7FDEF7447DB541F8E9898EED3A7BF4C0D6C1`。完整逐点审计位于 `data/processed/NASA data/nasa_minimal_interpolation_audit.csv`，复现脚本为 `tools/prepare_nasa_minimal_interpolation.py`。

### 2.4 离线处理之后的统一训练处理

上述 `.npy` 中 `Capacity` 始终保存为 Ah，离线文件不预先写入某个测试折的 Min–Max 值。训练时再执行以下统一且防泄漏的步骤：

- 留一电池测试：NASA=B0005、CALCE=CS2_35、TJU=CY25-1；测试电池不参与训练、验证、归一化或模型选择。
- 其余电池分别按时间顺序取前 80% 训练、后 20% 验证；历史窗口为 NASA 16、CALCE/TJU 64，执行真实历史驱动的下一周期容量预测。
- 额定容量分别为 2.0、1.1、2.5 Ah；SOH/target 在模型需要时由 `Capacity/rated_capacity` 计算。
- 模型需要 Min–Max 或 z-score 时，统计量只由训练电池的训练段拟合，再原样应用于验证和测试；不会使用测试电池范围。
- 早/晚评估起点分别为 NASA 50/90、CALCE 200/400、TJU 200/400；正式随机种子为 7、17、27、37、47、57、67、77、87、97。
- IC2ML 仅补充模型原生输入：NASA 从 3.9–4.0 V 充电段提取 10 点增量容量，CALCE 从 3.6–3.7 V 提取 10 点增量容量，TJU 使用上述 16 个循环级统计量；这些输入不会替换或改写统一容量标签。

## 3. 七模型输入与保留结构

| 模型 | version3 输入 | 保留的核心思想 |
|---|---|---|
| MGI-DSSM | 容量窗口、放电曲线及物理状态缓存 | 宏观观测引导的微观状态反演、物理状态监督与容量联合学习 |
| PatchFormer | 单变量容量历史 | patch 表征、DPAN/FAN 和投影头 |
| RUL-Mamba | NASA/CALCE 容量；TJU 17 个循环级变量 | Mamba 编码器及 Mamba-GRN 解码器 |
| BATTER-MoE | NASA/CALCE 容量；TJU 容量与 16 个统计量 | 多尺度 patch、注意力和稀疏共享 MoE |
| IC2ML | NASA/CALCE 充电增量曲线；TJU 16 个统计量 | 循环内/循环间表征、Inception 与多任务结构 |
| Autoformer | 容量及因果循环时间标记 | 渐进分解、FFT Auto-Correlation 和时间延迟聚合 |
| iTransformer | 容量、差分、3/7 周期后向趋势 | variate token、inverted embedding 和变量间自注意力 |

## 4. 我们模型（MGI-DSSM）的三数据集配置

### 通用训练参数

| 参数名 | 含义 | NASA | CALCE | TJU |
|---|---|---:|---:|---:|
| `head` | 模型训练头 | mgi-physics | mgi-physics | mgi-physics |
| `dataset` | 数据集标识 | nasa | calce | tju |
| `data_dir` | 统一数据根目录 | data/version3 | data/version3 | data/version3 |
| `output_dir` | 单次运行输出目录 | 按 seed 生成 | 按 seed 生成 | 按 seed 生成 |
| `seed` | 随机种子 | 十种固定值 | 十种固定值 | 十种固定值 |
| `test_names` | 留出测试电池 | B0005 | CS2_35 | CY25-1 |
| `seq_len` | 历史窗口长度 | 16 | 64 | 64 |
| `start_points` | 早/晚评估起点 | 50/90 | 200/400 | 200/400 |
| `rated_capacity` | 额定容量/Ah | 2.0 | 1.1 | 2.5 |
| `epochs` | 最大训练轮数 | 80 | 60 | 60 |
| `batch_size` | 批大小 | 63 | 128 | 128 |
| `hidden_dim` | 隐藏维数 | 48 | 32 | 48 |
| `lr` | Adam 学习率 | 5e-4 | 5e-4 | 5e-4 |
| `dropout` | dropout 比例 | 0.0 | 0.0 | 0.0 |
| `physics_num_layers` | 物理反演网络层数 | 2 | 2 | 1 |
| `physics_validation_fraction` | 验证比例 | 0.15 | 0.15 | 0.15 |
| `physics_early_stopping_patience` | 早停耐心值 | 16 | 12 | 12 |

### 损失与监督参数

| 参数名 | 含义 | NASA | CALCE | TJU |
|---|---|---:|---:|---:|
| `physics_state_loss_weight` | 物理状态监督权重 | 0.4 | 0.4 | 0.4 |
| `physics_state_supervision` | 状态监督来源 | curve | curve | curve |
| `physics_curve_loss_weight` | 放电曲线重建权重 | 0.0275 | 0.02 | 0.02 |
| `physics_weak_state_loss_weight` | 弱状态约束权重 | 0.005 | 0.01 | 0.02 |
| `physics_capacity_loss_weight` | 容量预测权重 | 0.22 | 1.0 | 1.0 |
| `physics_direction_loss_weight` | 退化方向约束权重 | 0.0 | 0.05 | 0.0 |
| `physics_late_life_weight` | 后期寿命样本附加权重 | 0.2 | 0.5 | 0.5 |
| `physics_voltage_error_scale` | 电压重建误差尺度 | 0.045 | 0.05 | 0.05 |
| `physics_regeneration_loss_weight` | 容量回升样本加权系数 | 0.05 | 0.0 | 0.0 |

### 优化器、缩放与趋势参数

| 参数名 | 含义 | NASA | CALCE | TJU |
|---|---|---:|---:|---:|
| `physics_weight_decay` | Adam 权重衰减 | 1e-4 | 1e-4 | 1e-4 |
| `physics_adam_beta1` | Adam β1 | 0.9 | 0.9 | 0.9 |
| `physics_adam_beta2` | Adam β2 | 0.9995 | 0.999 | 0.999 |
| `physics_adam_eps` | Adam ε | 1e-5 | 1e-8 | 1e-8 |
| `physics_lr_scheduler` | 学习率调度器 | none | none | none |
| `physics_scheduler_min_lr_ratio` | 最小学习率比例 | 0.05 | 0.05 | 0.05 |
| `physics_scheduler_patience` | plateau 耐心值 | 6 | 6 | 6 |
| `physics_capacity_huber_beta` | 容量 Huber loss 转折点 | 0.017 | 0.01 | 0.01 |
| `physics_grad_clip_norm` | 梯度裁剪范数 | 1.0 | 1.0 | 1.0 |
| `physics_state_scaling` | 状态特征缩放 | protocol | protocol | protocol |
| `physics_capacity_target_scaling` | 容量目标缩放 | protocol | protocol | protocol |
| `physics_thermo_step_scale` | 热力学状态步长尺度 | 0.03 | 0.02 | 0.02 |
| `physics_kinetic_step_scale` | 动力学状态步长尺度 | 0.05 | 0.03 | 0.03 |
| `physics_trend_short_window` | 短趋势窗口 | 2 | 8 | 8 |
| `physics_trend_long_window` | 长趋势窗口 | 4 | 32 | 32 |

### 物理与离散化参数

| 参数名 | 含义 | NASA | CALCE | TJU |
|---|---|---:|---:|---:|
| `physics_cutoff_voltage` | 放电截止电压/V | 2.7 | 2.7 | 2.5 |
| `physics_discharge_current` | 标称放电电流/A | 2.0 | 1.1 | 2.5 |
| `physics_tau_p_seconds` | 极化时间常数/s | 120 | 120 | 120 |
| `physics_q_grid_max_ah` | 容量积分网格上限/Ah | 2.4 | 1.5 | 3.0 |
| `physics_q_grid_points` | 容量网格点数 | 1200 | 400 | 600 |
| `physics_max_self_reconstruction_mae` | 物理自重建 MAE 上限 | 0.004 | 0.004 | 0.008 |
| `physics_preprocessing_protocol` | 预处理协议 | batter_moe | legacy | batter_moe |
| `physics_evaluation_protocol` | 评估协议 | patchformer（运行时覆盖） | patchformer | patchformer（运行时覆盖） |
| `physics_ocp_profile` | OCP 化学体系 | 默认 | 默认 | nmc_graphite_siox |

### 数据文件及数据集专属参数

| 数据集 | `physics_summary_filename` | `physics_cache_name` | 专属参数 |
|---|---|---|---|
| NASA | `NASA_Data_minimal_interpolated.npy` | `physics_curve_cache_nasa_tuned_best.npz` | `physics_capacity_huber_beta=0.017`；Adam β1/β2/ε = 0.9/0.9995/1e-5；`physics_thermo_step_scale=0.03`；`physics_kinetic_step_scale=0.05`；趋势窗口 2/4 |
| CALCE | `CALCE_Data.npy` | `physics_curve_cache_calce_version2_best_profile_v1.npz` | SOH 阈值偏置校准 0.00012、校准带宽 0.015、symmetric；EOL phase alignment=none、clip=2 |
| TJU | `TJU_Data_version2_model_adapter.npy` | `physics_curve_cache_tju_version2_best_profile_v1.npz` | `physics_ocp_profile=nmc_graphite_siox` |

预处理与 EOL 专属参数：NASA、CALCE、TJU 的 `physics_outlier_sigma_window` 均为 0；`physics_outlier_preserve_endpoints` 分别为 true、false、true。CALCE 使用 `physics_threshold_bias_calibration_soh=0.00012`、`physics_threshold_bias_band_soh=0.015`、`physics_threshold_bias_mode=symmetric`、`physics_eol_event_phase_alignment=none` 和 `physics_eol_event_phase_clip_cycles=2`；其余数据集使用默认 EOL 阈值策略。

配置源文件位于 `configs/final_nasa_progressive_best.json`、`configs/final_calce_version2_optimized.json` 和 `configs/final_tju_batter_moe_preprocessed.json`；正式调度器会覆盖 `data_dir`、测试电池、随机种子、输出目录、summary/cache 名称及统一评估协议。

## 5. 训练命令

### 七模型 × 三数据集 × 十随机种子

当前环境为 CPU 版 PyTorch：

```powershell
python run_seven_models_version3_10seeds.py --device cpu --output-root outputs/seven_models_version3_10seeds
```

若 `python -c "import torch; print(torch.cuda.is_available())"` 返回 `True`，可以使用：

```powershell
python run_seven_models_version3_10seeds.py --device cuda --output-root outputs/seven_models_version3_10seeds
```

### 按数据集或模型运行

```powershell
# 七模型只跑 CALCE
python run_seven_models_version3_10seeds.py --datasets calce --device cpu --output-root outputs/seven_models_version3_10seeds

# 只跑新加入的两个模型
python run_seven_models_version3_10seeds.py --models autoformer itransformer --datasets nasa calce tju --device cpu --output-root outputs/seven_models_version3_10seeds

# 只跑我们的模型
python run_seven_models_version3_10seeds.py --models our_model --datasets nasa calce tju --device cpu --output-root outputs/seven_models_version3_10seeds
```

正式入口要求恰好十个互不重复的随机种子。每个 `results.json` 只有在状态、模型配置和 version3 数据源均匹配时才会断点跳过。

## 6. 输出

单次结果位于：

```text
outputs/seven_models_version3_10seeds/<model>/<dataset>/<battery>/seed_<seed>/
```

总目录生成 `all_results.csv/json`、`mean_std_over_10seeds.csv/json`、`best_run_by_metric.csv/json`、`completion_report.json` 和 `_logs/`。
=======
# CL-MGI-DSSM
MGI-DSSM的源码及对比模型源码

阅读README.md查看如何训练，并阅读终端日志查看结果文档
>>>>>>> 4cf4562ce0d8ae0179ea094fae2aa505e0e438ee
