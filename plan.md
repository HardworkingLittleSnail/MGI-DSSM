# BATTER-MoE实验章节逐节梳理与NASA实验配置

文献：*BATTER-MoE: A Sparse Mixture-of-Experts Model for Accurate and Efficient Remaining Useful Life Prediction of Lithium-Ion Batteries*，IEEE Transactions on Transportation Electrification，DOI: 10.1109/TTE.2026.3697742。

本文实验相关内容由两部分组成：第IV节“Experimental Setup”规定数据、任务和训练协议；第V节“Results and Discussion”通过六个小节完成性能、消融、参数、效率、鲁棒性和迁移验证。

## 一、IV. Experimental Setup

### IV-A. Datasets and Preprocessing

#### 这个小节做了什么

介绍实验使用的数据集、各数据集承担的验证任务，以及所有数据采用的预处理方法。其目的是建立一个同时覆盖实验室电池、工业大容量电池和多变量BMS观测的评价环境。

#### 具体怎么做

1. 使用三个公开数据集：
   - NASA：B0005、B0006、B0007和B0018，2 Ah LCO电池；
   - GOTION：Cell01、Cell02和Cell03，27 Ah LFP商用方形电池；
   - TJU：CY25-1、CY25-2和CY25-3，包含NCM/NCA电池。

2. 设置两种输入任务：
   - NASA和GOTION只输入历史容量序列，执行单变量容量预测；
   - TJU从每个循环中提取17个健康指标，执行多变量容量预测。

3. TJU的17个指标包括：
   - 6个电压统计量：均值、标准差、峰度、偏度、斜率和熵；
   - 6个电流统计量：均值、标准差、峰度、偏度、斜率和熵；
   - 4个充电过程指标：恒流充电容量、恒流充电时间、恒压充电容量和恒压充电时间；
   - 当前循环容量。

4. 所有数据采用统一预处理：
   - 通过3σ准则识别并删除孤立异常点；
   - 对被删除点进行线性插值；
   - 容量使用`C/C0`归一化；
   - TJU多变量指标使用Min–Max归一化；
   - 归一化统计量只由训练集计算，再应用到验证集和测试集。

5. 论文没有专门识别或校正静置引起的容量恢复，只删除孤立异常点，连续的局部容量波动仍被保留。

#### 对应证据

- 原文p.5，Section IV-A；
- Table I：三个数据集的额定容量、电流和截止电压；
- Fig. 2：预测流程和三个数据集的退化轨迹。

### IV-B. Problem Formulation and Evaluation Protocol

#### 这个小节做了什么

定义模型到底输入什么、预测什么、如何从容量预测得到RUL，以及训练电池和测试电池怎样划分。

#### 具体怎么做

1. 采用滑动窗口单步预测：
   - 在循环`k`，输入截至`k`的长度为`L_in`的历史窗口；
   - 预测下一循环容量`C_(k+1)`。

2. 不同数据集的窗口输入不同：
   - NASA和GOTION输入`{C_(k-L_in+1), ..., C_k}`；
   - TJU输入17维循环级特征序列`{x_(k-L_in+1), ..., x_k}`。

3. 测试时每一次单步预测均使用真实历史窗口，而不是将上一时刻预测值递归输入。因此，该实验不是自由滚动的多步预测。

4. 从连续单步预测形成的容量序列中，找到第一次低于EOL阈值`τC0`的循环，并据此计算预测RUL和RE。

5. 采用held-out-cell协议：
   - 测试电池完全不参与训练、验证和归一化统计量估计；
   - 其余电池内部按80%/20%划分训练集和验证集。

6. 具体划分为：

| 数据集 | 训练/验证电池       | 测试电池 | EOL阈值 | Early/Late SP |
| ------ | ------------------- | -------- | ------: | ------------: |
| NASA   | B0006、B0007、B0018 | B0005    |     70% |         50/90 |
| GOTION | Cell02、Cell03      | Cell01   |     80% |       450/750 |
| TJU    | CY25-2、CY25-3      | CY25-1   |     70% |       200/400 |

#### 对应证据

- 原文pp.5–6，Section IV-B；
- Eq. 11：SOH和RUL定义；
- Table II：held-out-cell划分、阈值和SP。

### IV-C. Evaluation Metrics

#### 这个小节做了什么

规定容量轨迹和RUL预测分别使用什么指标评价。

#### 具体怎么做

- 使用MAE评价平均绝对容量误差；
- 使用RMSE增强对较大误差的敏感性；
- 使用R²评价预测轨迹对真实退化变化的拟合程度；
- 使用RE评价预测RUL与真实RUL的相对偏差。

#### 对应证据

- 原文p.6，Section IV-C；
- Eqs. 12–15：MAE、RMSE、R²和RE的数学定义。

### IV-D. Baseline Models

#### 这个小节做了什么

建立主实验的对比模型集合，使对比覆盖卷积网络、Transformer、状态空间模型以及电池专用模型。

#### 具体怎么做

主对比实验使用5个基线：

| 模型         | 方法类别            | 在实验中的作用                      |
| ------------ | ------------------- | ----------------------------------- |
| ModernTCN    | CNN/TCN             | 比较局部和层次化时序建模能力        |
| Autoformer   | Transformer         | 比较基于自相关的长依赖建模能力      |
| iTransformer | Transformer         | 比较变量token化和跨变量依赖建模能力 |
| PatchFormer  | 电池专用Transformer | 比较面向电池退化的patch建模方法     |
| RUL-Mamba    | 状态空间模型        | 比较近年的电池RUL状态空间模型       |

NASA B0005专项文献对比还加入PSO multi-kernel RVM、TCN-LSTM、TCN-GRU-DNN和AttMoE，但这些结果来自已发表文献，并非Table IV统一框架下的主要基线。

#### 对应证据

- 原文p.6，Section IV-D；
- Table IV：5个统一主基线；
- Table V：NASA B0005已发表方法专项对比。

### IV-E. Training and Hyperparameters

#### 这个小节做了什么

说明模型如何训练、超参数如何选择，以及最终在三个数据集上使用什么配置。

#### 具体怎么做

1. 只使用训练集和验证集进行有界超参数搜索，不使用held-out测试电池选择参数。
2. 搜索范围覆盖：
   - 多尺度patch组合；
   - 模型维度和层数；
   - expert数量和Top-k；
   - 路由均衡损失权重；
   - dropout、学习率和batch size。
3. 最终配置同时依据验证性能、训练稳定性和计算效率确定。
4. 三个数据集统一使用Adam和`1×10⁻³`学习率，batch size均为128，early stopping监控MAE，patience为10。
5. NASA最终设置为：窗口16、patch集合`{2,4,8}`、`d_model=64`、1层编码器、`d_ff=128`、4个experts、Top-k=2、dropout=0.05。
6. 实验环境为RTX 3060、Intel Core i5-13490F和PyTorch 2.1.0。

#### 对应证据

- 原文p.6，Section IV-E；
- Table III：三个数据集的最终架构和训练参数。

## 二、V. Results and Discussion

论文在本节开头声明：所有结果均来自10个不同随机种子的独立运行，并报告其统计结果。

### V-A. Comparative Performance Analysis

#### 这个小节做了什么

验证BATTER-MoE在不同数据规模、材料体系、输入维度和预测起点下，是否比现有模型具有更高的预测精度和更稳定的结果。

#### 具体怎么做

该小节包含四项实验或分析。

##### 实验1：三个数据集的Early/Late主对比

- 在NASA、GOTION和TJU上分别设置Early SP和Late SP；
- BATTER-MoE与ModernTCN、Autoformer、iTransformer、PatchFormer和RUL-Mamba比较；
- 报告MAE、RMSE、R²和RE；
- 计算BATTER-MoE相对第二名的提升比例；
- 所有数值为10个随机种子独立运行的统计结果。

对应Table IV。

##### 实验2：预测轨迹和重复运行稳定性

- 绘制三个数据集在Early/Late SP下的容量退化预测曲线；
- 同时展示MAE/RMSE对比；
- 使用误差条表示10次独立运行的标准差；
- 观察模型预测曲线与真实退化曲线的接近程度及运行稳定性。

对应Fig. 4。

##### 实验3：TJU特征重要性

- 在TJU的SP200和SP400上分析输入特征梯度；
- 对每次运行的评估样本计算梯度特征重要性；
- 在batch间聚合后，于每次运行内部归一化；
- 展示5次重复运行R1–R5，观察模型是否持续关注相近的关键特征。

对应Fig. 3。

##### 实验4：NASA B0005专项文献对比

- 选取已在B0005上报告结果的PSO multi-kernel RVM、TCN-LSTM、TCN-GRU-DNN、AttMoE和RUL-Mamba；
- 列出各方法年份、SP、MAE、RMSE和RE；
- 重点比较相同SP50下的方法；
- BATTER-MoE报告MAE=0.0045、RMSE=0.0088、RE=0。

对应Table V。

### V-B. Synergistic Effects of Core Components

#### 这个小节做了什么

通过大型消融实验判断每个核心模块是否有效，并验证多个模块是否具有互补作用，而不是只靠某一个模块获得性能提升。

#### 具体怎么做

1. 在三个数据集的Early设置上进行：NASA SP50、GOTION SP450和TJU SP200。
2. 对每个变体报告MAE、RMSE和RE。
3. 括号中报告完整模型相对该变体的误差下降比例。
4. R²因接近0.99且区分度较低，没有放入消融表。
5. 具体变体包括：
   - w/o CS-SE；
   - Classic SE替换CS-SE；
   - w/o CT-Reweighting；
   - 同时去除CS-SE和CT-Reweighting；
   - w/o MoE；
   - w/o Patch；
   - Mean Pooling；
   - Index Pooling；
   - Sequential-index RoPE；
   - w/o RoPE；
   - Last-token Pooling；
   - Full Model。
6. 通过`w/o CS-SE + CT-Reweighting`与分别去除单个模块的结果，验证跨尺度通道重标定与跨时间重加权的互补性。
7. 通过Classic SE和CS-SE的结构与结果对比，验证显式跨尺度交互是否优于单尺度通道注意力。
8. 通过不同pooling与RoPE变体，验证最新有效时间戳和真实时间位置信息的作用。

#### 对应证据

- 原文pp.7–8，Section V-B；
- Table VI：完整消融结果；
- Fig. 5：Classic SE与CS-SE结构对比。

### V-C. Optimization of MoE Key Parameters

#### 这个小节做了什么

研究MoE中Top-k和expert数量对精度、稳定性和参数量的影响，并确定主实验采用`k=2`的依据。

#### 具体怎么做

1. 分别设置4个和8个总experts。
2. 改变Top-k，图中测试`k=1、2、3、4、6、8`。
3. 在NASA上统计预测MAE及总参数量。
4. 在GOTION上统计预测RMSE及总参数量。
5. 同时观察平均误差、重复运行方差和参数规模。
6. 结果表明`k=1`或`k=2`误差及方差较小；`k>2`后性能下降。
7. 由于激活expert内部维度按`d_ff/k`设置，增大`k`反而会缩小每个expert并降低总参数量。
8. 综合精度和参数效率，主实验最终采用`k=2`。

#### 对应证据

- 原文p.9，Section V-C；
- Fig. 6：Top-k对NASA MAE、GOTION RMSE和参数量的影响。

### V-D. Parameter Efficiency Under Resource-Constrained Settings

#### 这个小节做了什么

验证BATTER-MoE不仅精度高，而且训练、推理和部署开销较低；同时检查模型在严格参数预算下是否仍能维持性能。

#### 具体怎么做

该小节包含三项实验。

##### 实验1：不同方法训练与推理时间

- 在NASA、GOTION和TJU上比较全部主基线；
- 记录10次运行下的平均停止epoch；
- 记录总训练时间和推理时间；
- 各方法使用相同early-stopping协议；
- SP分别固定为NASA 50、GOTION 450和TJU 200。

对应Table VII。

##### 实验2：完整模型部署指标

- 对三个数据集的最终BATTER-MoE配置计算参数量和FLOPs；
- 在batch size=1下测量GPU推理延迟和CPU推理延迟；
- 测量峰值运行内存；
- 用于评价实际部署占用，而不只是参数量。

对应Table VIII。

##### 实验3：受限参数预算

- 通过缩小编码层数`L`、隐藏维度`d_model`和前馈维度`d_ff`构造不同大小的模型；
- NASA设置1.34K、10.16K和94.5K三个规模；
- GOTION设置5.77K、36.78K和1.1M三个规模；
- 比较各规模的MAE和RMSE；
- 观察模型压缩后精度是否平滑退化。

对应Table IX。

### V-E. Robustness to Noisy Training Inputs

#### 这个小节做了什么

验证训练数据受到传感噪声或异常污染时，BATTER-MoE是否比传统机器学习和其他深度学习模型更加稳定。

#### 具体怎么做

1. 只污染NASA训练集，不污染测试集。
2. 人工注入混合噪声：30% burst noise和70% Gaussian noise。
3. 设置三种corruption ratio：0、0.1和0.2。
4. 对比SVR、GPR、PatchFormer、RUL-Mamba和BATTER-MoE。
5. 在每个噪声比例下报告MAE、RMSE和RE。
6. 比较不同方法随噪声比例增加时的性能退化速度。

#### 对应证据

- 原文p.10，Section V-E；
- Table X：NASA噪声鲁棒性结果。

### V-F. Transfer Learning with Limited Target Data

#### 这个小节做了什么

验证当目标域GOTION只有少量训练数据时，NASA预训练是否能够提升预测精度、数据效率和训练稳定性。

#### 具体怎么做

1. 首先在NASA上预训练BATTER-MoE。
2. 将NASA预训练权重迁移到GOTION。
3. 在GOTION上进行全参数微调，不冻结任何网络参数。
4. 设置GOTION训练数据比例为0.1、0.2、0.3和0.5。
5. 设置两个对照：
   - Scratch：使用相同GOTION配置从头训练；
   - Transfer：从NASA预训练权重开始全参数微调。
6. 所有设置固定SP=450。
7. 对每个数据比例报告MAE和RMSE的均值±标准差。
8. 比较不同数据量下迁移学习带来的误差下降和方差变化。

#### 对应证据

- 原文p.10，Section V-F；
- Table XI：不同目标域数据比例下Scratch与Transfer结果。

## 三、BATTER-MoE实验章节的完整证据顺序

| 顺序 | 小节 | 核心问题                 | 实验手段                                  |
| ---: | ---- | ------------------------ | ----------------------------------------- |
|    1 | V-A  | 模型是否比现有方法准确   | 三数据集、Early/Late、多基线、10次重复    |
|    2 | V-B  | 性能提升来自哪些模块     | 三数据集大型模块消融和替换实验            |
|    3 | V-C  | MoE关键参数为何这样选择  | Top-k和expert数量的精度—参数量分析        |
|    4 | V-D  | 模型是否高效且可部署     | 时间、参数量、FLOPs、延迟、内存和压缩模型 |
|    5 | V-E  | 数据含噪时是否稳定       | NASA训练集混合噪声注入                    |
|    6 | V-F  | 少量目标域数据下能否适应 | NASA预训练到GOTION全参数迁移              |

## 四、我们的NASA实验应写入论文的精选配置

下面只保留其他论文通常报告且影响复现的设置，不照搬完整JSON。

### 4.1 数据与任务协议

| 项目          |            NASA设置 |
| ------------- | ------------------: |
| 测试电池      |               B0005 |
| 训练/验证电池 | B0006、B0007、B0018 |
| 验证比例      |                 15% |
| 预测起点      |               SP=50 |
| 历史窗口      |            16个循环 |
| 预测步长      |             1个循环 |
| EOL阈值       |       额定容量的70% |
| 重复实验      |    10个独立随机种子 |

### 4.2 模型与训练超参数

| 类别       | 参数                      |                    设置 |
| ---------- | ------------------------- | ----------------------: |
| 状态转移器 | GRU hidden dimension      |                      48 |
| 状态转移器 | GRU layers                |                       2 |
| 状态转移器 | Dropout                   |                     0.0 |
| 优化器     | Optimizer                 |                   AdamW |
| 优化器     | Learning rate             |                `5×10⁻⁴` |
| 优化器     | Batch size                |                      63 |
| 优化器     | Maximum epochs            |                      80 |
| 优化器     | Early-stopping patience   |                      16 |
| 优化器     | Weight decay              |                `1×10⁻⁴` |
| 优化器     | `(β₁, β₂, ε)`             | `(0.9, 0.9995, 1×10⁻⁵)` |
| 损失       | Capacity loss weight      |                    0.22 |
| 损失       | Voltage-curve loss weight |                  0.0275 |
| 损失       | Weak-state loss weight    |                     0.0 |
| 损失       | Regeneration-aware weight |                    0.05 |
| 损失       | Late-life weight          |                    0.20 |
| 损失       | Smooth-L1/Huber `β`       |                0.017 Ah |

### 4.3 物理解码实现细节

| 参数                            |              设置 |
| ------------------------------- | ----------------: |
| 极化时间常数`τ_p`               |             120 s |
| 容量求解区间                    |          0–2.4 Ah |
| 容量网格点数                    |              1200 |
| 电压误差尺度                    |           0.045 V |
| 放电电流                        |               2 A |
| B0005/B0006/B0007/B0018截止电压 | 2.7/2.5/2.2/2.5 V |

### 4.4 预处理需要交代的内容

- 使用恒流放电阶段的电压—容量曲线；
- 将电流积分获得的容量轴对齐至数据集提供的循环容量；
- 采用局部3σ异常检测和线性插值；
- 当前最优配置的局部窗口长度为13个循环；
- 只删除孤立异常点，保留连续容量恢复波动；
- 隐状态标准化统计量只由训练电池计算。

### 4.5 不写入论文主文的配置

- 输出目录、缓存名和脚本名；
- 日志与模型保存频率；
- 内部闭合检查阈值；
- `seed=87`这一单次最优种子；
- 调参过程中尝试但最终未采用的候选参数。

当前`seed=87`最优单次结果为MAE=0.0046353736 Ah、RMSE=0.0077219708 Ah、R²=0.9966367483、RE=0。论文主表应报告10次独立运行的均值和标准差，单次最优值只能作为补充结果。

