# 本模型保留最优配置

本目录固定保存截至 2026-08-10 已确认的三个数据集最优配置，供后续复现实验使用。目录内文件是来源配置的独立快照，后续调参不应直接覆盖。

| 数据集 | 固定配置 | 来源配置 | 评测电池 | 预测设置 | 预测起点 |
|---|---|---|---|---|---|
| NASA | `nasa_best.json` | `configs/final_nasa_progressive_best.json` | B0005 | 16 步预测 1 步 | 50 / 90 |
| CALCE | `calce_best.json` | `configs/final_calce_version2_optimized.json` | CS2_35 | 64 步预测 1 步 | 200 / 400 |
| TJU | `tju_best.json` | `configs/final_tju_batter_moe_preprocessed.json` | CY25-1（当前配置仍列出同组其余电池） | 64 步预测 1 步 | 200 / 400 |

注意：三个配置保留各自已经确认的数据目录和预处理口径，并不代表三者的数据预处理参数相同。
