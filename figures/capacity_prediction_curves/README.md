# Capacity prediction figures

## Figure contract

- Core conclusion: MGI-DSSM follows the measured capacity trajectory and accurately identifies the EOL region across the NASA, CALCE and TJU test batteries.
- Archetype: asymmetric quantitative comparison, with the full trajectory as the primary panel and the EOL enlargement as validation.
- Test unit: one held-out battery per dataset (NASA B0005, CALCE CS2_35 and TJU CY25-1).
- EOL definition: first cycle at or below 70% of rated capacity.
- Output size: approximately 183 mm wide and 88 mm high.
- Primary editable formats: SVG and PDF. Submission raster: 600-dpi LZW-compressed RGB TIFF. PNG is provided for review.

## Paper-consistent method names

MGI-DSSM; MSTEA-Net; IC2ML; BATTER-MoE; RUL-Mamba; PatchFormer; SG-DiTs; Autoformer; iTransformer.

## Proposed legends

### NASA

**Fig. X | Capacity prediction on the NASA dataset.** **a,** Measured capacity trajectory of the held-out B0005 battery from cycle 16. The grey region denotes the observed history, and MGI-DSSM predictions begin at cycle 50. The green curve reports the cycle-wise absolute capacity error on the right axis. The blue dashed line denotes the EOL threshold of 1.40 Ah, and the shaded rectangle identifies the enlarged region. **b,** Comparison of measured and predicted capacity trajectories around EOL for MGI-DSSM and eight reference methods. Black dotted and red dash-dotted vertical lines denote the measured and MGI-DSSM-predicted EOL cycles, respectively. Source data are provided in `source_data_NASA.csv`.

### CALCE

**Fig. X | Capacity prediction on the CALCE dataset.** **a,** Measured capacity trajectory of the held-out CS2_35 battery from cycle 64. The grey region denotes the observed history, and MGI-DSSM predictions begin at cycle 200. The green curve reports the cycle-wise absolute capacity error on the right axis. The blue dashed line denotes the EOL threshold of 0.77 Ah, and the shaded rectangle identifies the enlarged region. **b,** Comparison of measured and predicted capacity trajectories around EOL for MGI-DSSM and eight reference methods. Black dotted and red dash-dotted vertical lines denote the measured and MGI-DSSM-predicted EOL cycles, respectively. Source data are provided in `source_data_CALCE.csv`.

### TJU

**Fig. X | Capacity prediction on the TJU dataset.** **a,** Measured capacity trajectory of the held-out CY25-1 battery from cycle 64. The grey region denotes the observed history, and MGI-DSSM predictions begin at cycle 200. The green curve reports the cycle-wise absolute capacity error on the right axis. The blue dashed line denotes the EOL threshold of 1.75 Ah, and the shaded rectangle identifies the enlarged region. **b,** Comparison of measured and predicted capacity trajectories around EOL for MGI-DSSM and eight reference methods. Black dotted and red dash-dotted vertical lines denote the measured and MGI-DSSM-predicted EOL cycles, respectively. Source data are provided in `source_data_TJU.csv`.

## Data and visual QA notes

- All measured-capacity observations from the declared window start to the last available cycle are retained.
- All nine prediction curves start at the declared prediction cycle; no prediction is drawn in the observed-history region.
- No smoothing, interpolation, downsampling or aesthetic row exclusion is applied during plotting.
- The measured trajectory is taken from the common versioned dataset rather than duplicated model-specific truth columns. This avoids propagating two inconsistent NASA truth entries found in one model export.
- The provided prediction tables contain one representative run per model and dataset. Accordingly, these figures show trajectories rather than across-seed confidence intervals.
- SVG text remains editable; PDF fonts are embedded; TIFF files are exported at 600 dpi.
