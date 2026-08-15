# RUL-Mamba paper/code consistency audit (NASA first)

Reference: `RUL-Mamba.pdf`, Journal of Energy Storage 120 (2025) 116376.

## Paper-defined NASA reproduction contract

- Input/output (Fig. 1 and Sec. 2.1): a univariate historical capacity sequence
  is embedded from one variable to `D` features and mapped to the next capacity.
- Encoder (Fig. 1 and Sec. 2): variable embedding -> FAN -> Mamba; the final
  encoder state is the decoder context.
- FAN (Eqs. 2-4): global pooling -> FC -> ReLU -> FC -> sigmoid -> feature-wise
  scaling.
- Decoder (Fig. 1 and Sec. 2.3): `N` Mamba-GRN blocks followed by a linear
  projection. GRN uses ELU, dropout, GLU, a residual connection and LayerNorm.
- Mamba (Fig. 1(d), Sec. 2.4): input projection, depth-wise convolution, SiLU,
  selective SSM, multiplicative SiLU gate and output projection.
- Loss (Eq. 12): SMAPE; validation loss selects the trained model.
- Preprocessing (Sec. 2.6): linear interpolation for missing values, 2-sigma
  outlier removal and Min-Max normalization.
- Split (Table 3): B0006/B0007/B0018 are the training dataset and B0005 is the
  held-out test battery; the first/last 80%/20% of the training dataset are used
  for fitting/validation.
- NASA settings (Tables 2, 4 and 5): 2 Ah rated capacity, 1.4 Ah EOL, prediction
  starting points 50/70/90, `D=48`, one decoder layer, learning rate 0.0022 and
  dropout 0.0615.
- Training (Sec. 3.3): Adam, TPE with 200 trials, early stopping and 10 repeated
  train/test experiments.
- Evaluation (Eqs. 13-17): MAE, RMSE, R2, AE and RE. The first B0005 cycle at or
  below 1.4 Ah is cycle 125, so TRUL is 75/55/35 for SP 50/70/90.

## Material inconsistencies found and corrected

1. `Configs/NASA/Univariable/RULMamba.yaml` used dropout 0.1 instead of the
   NASA optimum 0.0615, and inherited learning rate 0.001 instead of 0.0022.
2. PyTorch Forecasting defaults to the Ranger optimizer, while the paper states
   Adam. The RUL-Mamba build now explicitly requests Adam.
3. The old data filter added the observed B0005 prefix to the supervised
   training dataset. It now enforces Table 3 exactly.
4. Every battery previously shared one group id, allowing windows to cross a
   battery boundary. Windows are now grouped by `BatteryName`.
5. PyTorch Forecasting previously applied an undocumented StandardScaler after
   the paper's Min-Max input normalization. The capacity scaler is now disabled.
6. Missing-value interpolation and 2-sigma filtering stated in Sec. 2.6 were
   absent. They are now implemented per battery. The supplied NASA capacity
   series contain neither missing values nor 2-sigma outliers, so this does not
   alter the four supplied curves.
7. The decoder loop repeatedly called every layer with `x_dec=None`; layers did
   not feed their output to the following layer. Decoder outputs are now passed
   through the stack. NASA uses one layer, but the defect affected Oxford/TJU.
8. Encoder feature selection relied on “all continuous columns except the last”.
   It now selects exactly `enc_in` declared inputs, preventing target leakage if
   dataset-generated columns change.
9. EOL/RUL code used an invalid chained inequality and returned zero if a
   prediction never crossed EOL. It now uses the first `capacity <= threshold`
   offset and reproduces the paper's B0005 TRUL values.
10. `loadMat()` parsed only POSIX paths and failed on Windows. It now uses
    `Path.stem`.
11. The NASA files in this checkout were Git LFS pointer text. The four exact
    objects were restored (their SHA-256 values match the LFS OIDs), and
    `NASA.npy` was regenerated with the exact expected LFS SHA-256
    `0c7f9b93ec2961891569315230d9fbc71fecc1fa259b974ffb07346fa1a31718`.
12. The standalone optimizer used a 30-step window, one TPE trial and non-paper
    base dimensions. NASA defaults/search ranges now use the repository's
    16-step window, 200 trials, Adam and the Table 4 fixed settings for its base
    run. Unused hidden-layer trial parameters were removed from RUL-Mamba TPE.

## Details not disclosed by the paper

The PDF does not provide the historical window length, batch size, maximum epoch
count, early-stopping patience/minimum delta, gradient clipping, random seed,
Mamba expansion factor/state dimension/convolution width, FAN reduction ratio,
or exact target-normalizer implementation. For NASA, the author repository's
values are retained: sequence length 16, batch size 16, maximum 200 epochs,
patience 20, gradient clipping 0.2, seed 2025, expansion 2, convolution width 4,
FAN ratio 2 and `EncoderNormalizer` for the target.

The paper presents one-step forecasting `x[t-l+1:t] -> y[t+1]` but does not state
whether an entire post-SP curve is generated recursively. The author pipeline
performs rolling one-step evaluation: every predicted cycle receives its actual
preceding capacity window. This behavior is retained because it matches the
published architecture and is not contradicted by an explicit rollout rule, but
it must not be interpreted as an open-loop forecast of the whole remaining life
from only the SP data.

Because these items are absent from the article, no implementation can be proven
“completely identical” from the PDF alone. They are recorded here instead of
being silently presented as paper-specified choices.

## Verification completed on 2026-08-03

- Restored data hashes match all five NASA Git LFS OIDs.
- Static compilation passed for every Python file under `Models` and `Scripts`.
- A two-layer decoder forward/backward test produced finite `[2, 1, 1]` output
  and gradients in both decoder layers.
- The instantiated NASA model has 106,657 trainable parameters, consumes one
  capacity input, outputs one next-cycle capacity, and resolves to Adam at
  learning rate 0.0022 with SMAPE.
- One complete GPU repeat was run at all paper starting points with maximum 200
  epochs and early stopping. This is an execution verification, not the paper's
  reported 10-run average.

| SP | stopped epoch | MAE | RMSE | R2 | TRUL | PRUL | AE | RE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 25 | 0.0101 | 0.0147 | 0.9881 | 75 | 77 | 2 | 0.0267 |
| 70 | 40 | 0.0099 | 0.0154 | 0.9759 | 55 | 56 | 1 | 0.0182 |
| 90 | 39 | 0.0103 | 0.0163 | 0.9542 | 35 | 37 | 2 | 0.0571 |

## Commands

Regenerate NASA cache:

```powershell
python Scripts/Data_Process/NASA_Data_Process.py
```

Run the paper configuration (10 repeats, SP 50/70/90):

```powershell
python Scripts/NASA_Univariable_RUL_Prediction/Train_NASA_Univariable.py `
  --config Configs/NASA/Univariable/Base.yaml `
  --model RULMamba `
  --model-config Configs/NASA/Univariable/RULMamba.yaml `
  --test-name B0005
```
