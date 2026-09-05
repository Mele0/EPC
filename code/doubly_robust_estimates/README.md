# Doubly robust estimates

The manuscript's primary aggregate result is a cross-fitted doubly robust estimator of
how many register entries would fall below a policy threshold under pre-revision carbon
accounting, and how much floor area those entries represent. This directory holds that
estimator's output: the Stage-1 register-conditional estimates, the Stage-2
external-composition sensitivities, the transport and double-robustness diagnostics, and
the full-pipeline bootstrap. These are the values reported in the manuscript.

`code/src/epc_artefact/dr_estimator.py` implements the estimator from Supplementary 1.21 and
runs as part of the pipeline. It recomputes the Stage-1 quantities from the register and
writes a direct comparison against the values held here:

| Quantity | Reported | Recomputed |
| --- | --- | --- |
| Eligible frame, EPC-B | 118,865 | 118,865 |
| Calibration sample, EPC-B | 1,110 | 1,110 |
| Plug-in term, Model C | 94,770 / 64.85 Mm2 | 94,770 / 64.85 Mm2 |
| Primary estimate, Model C | 94,911 / 57.24 Mm2 | 94,690 / 57.18 Mm2 |
| Primary count, F/G | 36,847 | 36,831 |

The frame, the calibration sample and the plug-in term recompute exactly. The remaining
difference sits entirely in the residual correction, which at the EPC-B threshold is
about 0.15 per cent of the plug-in term: a weighted mean residual of roughly 0.001 across
1,110 binary outcomes. That term is sensitive to the cross-fitting partition, so it is
reported alongside the estimate rather than treated as a fixed quantity.

## Contents

| File | Reported in |
| --- | --- |
| `bridge_stage1_frozen.csv` | Table 2 Panel B; Supplementary Table S31 |
| `bridge_bootstrap_draws.csv` | Supplementary Table S37 (1,000 replicates) |
| `bridge_dr_verification.csv` | Double-robustness numerical checks, Section 1.21 |
| `bridge_dr_leakage_audit.csv`, `bridge_dr_paired.csv`, `bridge_dr_oos_validation.csv` | Supplementary Table S30 |
| `bridge_stage2_ndneed.csv`, `bridge_stage2_ndneed_ladder.csv`, `bridge_stage2_ndneed_dropped.csv` | Supplementary Table S29 |
| `bridge_transport_support.csv`, `bridge_transport_trimming.csv` | Supplementary Table S32 |
| `bridge_reduced_rank_calibration.csv` | Section 1.21, reduced-rank calibration alternative |
| `bridge_fuel_area_dr.csv`, `bridge_sector_area_dr.csv`, `bridge_sectorcost_dr.csv` | Supplementary Table S36; Figure 3 panels c and d |

Seven of these are read at run time, by `figures.fig_panel2_thresholds_capex` when
drawing Figure 3 and by `paper_tables.table2_thresholds` when assembling Table 2.
