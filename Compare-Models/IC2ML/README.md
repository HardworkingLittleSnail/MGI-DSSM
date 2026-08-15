# IC2ML: Unified battery state-of-health, degradation trajectory and remaining useful life prediction via intra-nd inter-cycle enhanced machine learning 
Strategic management of lithium-ion batteries (LIBs) depends on evaluating current health status and predicting future degradation paths. Yet despite extensive research on core management tasks like 
state of health (SOH) estimation, degradation trajectory prediction, and remaining useful life (RUL) prediction, these tasks remain isolated without leveraging their inherent connections. This work 
proposes an unified framework that enables joint battery SOH, degradation trajectory and RUL prediction via an intra-cycle and inter-cycle enhanced machine learning (IC2ML). The IC2ML uses 1-D time-serials 
voltage data to implement SOH prediction, where the inter-cycle embeddings are further self-attentioned for degradation trajectory prediction. The RUL is derived from degradation trajectory prediction based 
on anticipated SOH levels, enabled by cross attention between output embeddings and input inter-intra cycle embeddings. The results demonstrate that using only 0.1V sampling interval data that can be extracted 
on-site, the average average root mean square error for SOH, degradation trajectory, and RUL prediction is 1.85%, 2.36% and 23.90 cycles, respectively, validated experimentally on 121 batteries spanning 
10 operation conditions. Sensitivity analysis shows that IC2ML can be adapted to scenarios where a few historical data is accessible. Broadly, this work highlights the significant poteintial of strategical 
battery management algorithm co-design using intra-cycle and inter-cycle battery degradation information for various management tasks.

## Highlights
- IC2ML, a unified framework jointly predicting SOH, degradation trajectory, and RUL, is proposed. 
- Health indicator are extracted from both 1-D voltage time series and 2-D images of voltage-capacity data.
- Spatiotemporal interaction among SOH, degradation trajectory and RUL is implemented through attention-based methods.
- The generalizability of IC2ML is validated with batteries of 3 materials and 10 operating conditions.
- IC2ML can adapt to limited data and extend to 100-cycle trajectory prediction with 1.77% RMSE.

# 1. Setup
## 1.1 Enviroments
* Python (Jupyter notebook) 
## 1.2 Python requirements
* python=3.11.5
* numpy=1.26.4
* torch=2.4.1
* keras=2.15.0
* matplotlib=3.9.2
* scipy=1.13.1
* scikit-learn=1.3.1
* pandas=2.2.2

# 2. Datasets
The raw data can be accessed via the following link:
* [Dataset](https://doi.org/10.5281/zenodo.6379165)

# 3. Demo
We provide a detailed demo of our code running .
1. Run the `run.py` file to train our model. The program will generate a folder named `checkpoint` and save the results in it.
2. You can change `setattr(args,'dataset')` to select the NCA, NCM, NCANCM datasets. It will generate a folder in the `checkpoint` to save the results of the corresponding datasets.

**Note: The results presented in this paper were not obtained through specific hyperparameter optimization. You may experiment with alternative hyperparameters to achieve similar or potentially improved outcomes.
Due to the inherent stochasticity of neural networks, the acquired expert weights will not remain identical across different runs. However, it is evident that significant differences exist among distinct aging stages.**
## Acknowledgement
This repo is constructed based on the following repos:
- https://github.com/thuml/Time-Series-Library
- Thanks to the following code for the assistance it provided in this paper.
- https://github.com/terencetaothucb/Early-Battery-Degradation-Prediction-via-Chemical-Process-Inference
- https://zenodo.org/records/15350607
- https://github.com/wang-fujin/PINN4SOH

## MGI-DSSM-aligned RUL benchmark

The common comparison task is one-step capacity prediction followed by the
MGI-DSSM threshold-RUL calculation. IC2ML keeps its native ten historical
cycles and ten charge-capacity-increment samples per cycle. Evaluation uses a
leave-one-battery-out split. The final 15% of each training battery's **raw
cycle sequence** is validation data. Checkpoints are selected by validation
capacity MAE, which is the common benchmark target; IC2ML's direct RUL head is
also trained and reported separately.

Capacity is below EOL only after two consecutive points at or below 70% rated
capacity. Capacity MAE/RMSE are in Ah; threshold-RUL AE and direct-RUL errors
are in cycles. Training uses Adam, learning rate 1e-4, batch size 128, at most
500 epochs, patience 100, and the paper loss
`MSE(SOH) + MSE(trajectory) + 0.5*MSE(direct RUL)`. Capacity increments, SOH,
and trajectory labels are divided only by rated capacity; there is no second
min-max normalization. Results use seeds
`7,17,27,37,47,57,67,77,87,97`. “Best” is selected independently per metric;
“mean” is the arithmetic ten-seed mean.

### Implementation alignment

- The released `10 -> 128 -> 256` intra-cycle embedding, two-head self
  attention, two-stage `1 -> 64 -> 128` Inception encoder, and MLP prediction
  heads are retained.
- The paper's equation (7) requires interaction between inter-cycle and 2-D
  features. The released global-pooling implementation reduces attention to
  one key and a constant softmax weight of one. This benchmark retains the 2-D
  Inception grid as tokens before cross-attention, so the stated multimodal
  interaction is actually computed.
- NASA and CALCE both use 3.9-4.0 V **charging** segments. CALCE cycles are
  order-preserving aligned to the official capacity summary. Missing segments
  are filled only from earlier cycles of the same battery; leading missing
  segments use zeros, never future data.
- The paper's 50-step trajectory head is configured to one step only because
  the shared MGI-DSSM comparison target is next-cycle capacity. The remaining
  architecture and three-task supervision are unchanged.

### NASA results (10 -> 1 capacity prediction)

NASA rated capacity is 2.0 Ah. Evaluation starts at cycles 50, 70, and 90;
each run-level value below is the mean over those three start points.

| Battery | Runs | MAE best / mean (Ah) | RMSE best / mean (Ah) | R2 best / mean | Threshold-RUL AE best / mean | Threshold-RUL RE best / mean |
|---|---:|---:|---:|---:|---:|---:|
| B0005 | 10 | 0.012967 / 0.018904 | 0.018922 / 0.025634 | 0.9602 / 0.9290 | 1.00 / 3.10 | 0.0205 / 0.0635 |
| B0006 | 10 | 0.084444 / 0.103246 | 0.100224 / 0.117027 | 0.2095 / -0.0772 | 29.00 / 30.00 | 0.8270 / 0.8488 |
| B0007 | 10 | 0.015490 / 0.046666 | 0.023922 / 0.053641 | 0.9094 / 0.4051 | 4.00 / 15.30 | 0.0416 / 0.1589 |
| B0018 | 10 | 0.027866 / 0.031961 | 0.033153 / 0.037509 | 0.3497 / 0.0978 | 9.00 / 24.50 | 0.5139 / 0.7775 |

### CALCE results (10 -> 1 capacity prediction)

CALCE rated capacity is 1.1 Ah. Evaluation starts at cycle 200 for CS2_35 and
CS2_36, and cycle 300 for CS2_37 and CS2_38.

| Battery | Runs | MAE best / mean (Ah) | RMSE best / mean (Ah) | R2 best / mean | Threshold-RUL AE best / mean | Threshold-RUL RE best / mean |
|---|---:|---:|---:|---:|---:|---:|
| CS2_35 | 10 | 0.024016 / 0.043908 | 0.031389 / 0.051250 | 0.9774 / 0.9325 | 1.00 / 26.20 | 0.0023 / 0.0595 |
| CS2_36 | 10 | 0.026970 / 0.037128 | 0.040749 / 0.049561 | 0.9748 / 0.9615 | 1.00 / 40.90 | 0.0022 / 0.0919 |
| CS2_37 | 10 | 0.015380 / 0.036631 | 0.021112 / 0.043591 | 0.9895 / 0.9501 | 12.00 / 81.60 | 0.0288 / 0.1962 |
| CS2_38 | 10 | 0.019242 / 0.033590 | 0.025247 / 0.039370 | 0.9843 / 0.9547 | 2.00 / 15.40 | 0.0044 / 0.0337 |

### Direct IC2ML RUL-head results

| Dataset | Battery | Direct RUL MAE best / mean (cycles) | Direct RUL RMSE best / mean (cycles) |
|---|---|---:|---:|
| NASA | B0005 | 6.39 / 13.84 | 8.27 / 16.45 |
| NASA | B0006 | 6.40 / 11.86 | 9.84 / 14.56 |
| NASA | B0007 | 15.30 / 25.85 | 17.62 / 29.99 |
| NASA | B0018 | 7.60 / 17.65 | 9.83 / 18.30 |
| CALCE | CS2_35 | 31.58 / 69.10 | 41.64 / 96.83 |
| CALCE | CS2_36 | 36.33 / 62.65 | 53.46 / 99.92 |
| CALCE | CS2_37 | 15.58 / 57.26 | 24.59 / 86.63 |
| CALCE | CS2_38 | 17.65 / 41.44 | 28.56 / 59.55 |

Per-seed checkpoints, `results.json`, and cycle-level `predictions.csv` files
are stored under `benchmark_results/<dataset>/<battery>/seed_<seed>/`.
