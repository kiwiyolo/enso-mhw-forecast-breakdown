# Supplementary Figure: robustness of the Figure 3 statistical driver framework

## Caption

**Observational diagnostics and sensitivity tests for the statistical driver-regime framework used in Figure 3.** **a**, Fraction of local monthly SST-anomaly variance linearly associated with the contemporaneous CPC ERSSTv5 Nino3.4 index during 1985-2020, expressed as squared Pearson correlation. The red outline marks the baseline broad tropical-Pacific source box (10S-10N, 120E-80W). **b**, Squared lag-1 correlation of monthly SST anomalies after linearly removing the contemporaneous Nino3.4 component. This quantity is interpreted as an ENSO-removed SST-persistence proxy, not as a resolved mixed-layer heat-budget term. **c**, Temporally held-out validation of that proxy: residual persistence is estimated during 1985-2002 and related to observed MHW continuation during 2003-2020. Points show eight equal-area bins and shading gives 95% confidence intervals from resampling 10-degree latitude by 20-degree longitude spatial blocks. The annotation reports the Spearman association across block means. **d**, Sensitivity of 2023-24 MHW-intensity-weighted global association shares to four explicit source-box definitions. Changing the box redistributes the ENSO-associated component between direct and remote labels while leaving local persistence and residual shares unchanged. **e**, Area-weighted agreement of alternative categorical regime maps with the baseline rule requiring a maximum share of at least 0.50 and a lead over the runner-up of at least 0.15. The red outline identifies the baseline rule; cell labels are percentages.

## Main results

- ENSO-associated variance and ENSO-removed persistence have distinct spatial structures, supporting their use as separate statistical predictors rather than interchangeable fields.
- Across 189 spatial blocks, 1985-2002 residual SST persistence is positively associated with 2003-2020 observed MHW continuation (Spearman rho=0.342, P=1.503e-06). The non-overlapping periods reduce same-sample circularity and support the interpretation of lag-1 residual SST correlation as an MHW-memory proxy, but not as direct identification of a local physical process.
- Under the baseline source box, the global 2023-24 intensity-weighted shares are direct ENSO 0.067, remote ENSO 0.047, local persistence 0.582, and residual 0.304.
- Alternative source boxes retain at least 99.9% area-weighted agreement with the baseline categorical regime map. The direct-versus-remote split is boundary-dependent and should not be interpreted as an independently observed physical partition.
- Across the 25 tested categorical thresholds, agreement with the baseline regime map ranges from 72.3% to 100%, with median agreement 88.1%. Complete regime fractions for every threshold pair are stored in `Figure3_driver_threshold_sensitivity.csv`.

## Calculation

For each ocean grid cell, the ENSO association score is

    S_ENSO = corr[SST'(t), Nino3.4(t)]^2.

The contemporaneous linear Nino3.4 component is removed grid-cell by grid-cell:

    SST'_res(t) = SST'(t) - beta Nino3.4(t),

after which the persistence score is

    S_local = corr[SST'_res(t), SST'_res(t-1)]^2.

For a specified source box, `S_ENSO` is labelled direct inside the box and remote outside it. The remaining score is `clip(1 - S_ENSO - S_local, 0.05, 1)`, after which all four scores are normalized to unit sum. A categorical regime is assigned only when the largest of direct, remote and local shares meets both the maximum-share and runner-up-margin requirements; otherwise it is labelled mixed.

Observed monthly MHW occurrence is defined when ERSST exceeds the local calendar-month 90th percentile for 1985-2014. For the held-out validation, the persistence predictor is estimated over 1985-2002 and MHW continuation is `P(MHW_t = 1 | MHW_t-1 = 1)` over 2003-2020. At least six predecessor MHW months are required at a grid cell.

## Interpretation boundary

This SI figure supports the internal statistical behaviour of the framework; it does not convert the framework into causal attribution. In simple one-predictor regression, squared Pearson correlation equals explained variance, but serial correlation and omitted climate modes limit causal interpretation. Linear removal of contemporaneous Nino3.4 does not fully remove lagged, nonlinear or seasonally varying ENSO teleconnections. Likewise, residual lag-1 persistence can reflect mixed-layer heat capacity, re-emergence, advection, other climate modes or unresolved forcing. Accordingly, the preferred terminology is `ENSO-associated source-region variance`, `remote ENSO association`, and `ENSO-removed local SST persistence`. Identifying physical basin-local drivers requires mixed-layer heat-budget terms or controlled forecast perturbation experiments.

## References

- Holbrook, N. J. et al. A global assessment of marine heatwaves and their drivers. *Nature Communications* 10, 2624 (2019). https://doi.org/10.1038/s41467-019-10206-z
- Jacox, M. G. et al. Global seasonal forecasts of marine heatwaves. *Nature* 604, 486-490 (2022). https://doi.org/10.1038/s41586-022-04573-9
- Compo, G. P. & Sardeshmukh, P. D. Removing ENSO-related variations from the climate record. *Journal of Climate* 23, 1957-1978 (2010). https://doi.org/10.1175/2009JCLI2735.1
- Gunnarson, J. L. et al. Removing ENSO's influence from global SST variability, with insights into the record-setting marine heat waves of 2023-24. *Bulletin of the American Meteorological Society* 106 (2025). https://doi.org/10.1175/BAMS-D-24-0023.1
- Shi, H. et al. Global decline in ocean memory over the 21st century. *Science Advances* 8, eabm3468 (2022). https://doi.org/10.1126/sciadv.abm3468
- Hobday, A. J. et al. A hierarchical approach to defining marine heatwaves. *Progress in Oceanography* 141, 227-238 (2016). https://doi.org/10.1016/j.pocean.2015.12.014

## Reproducibility

- Calculation and plotting script: `/data3/luoq/p03_Ocean_SST_ensemble/kw_99_paper/Codes/k18_plot_figure3_driver_method_sensitivity.py`
- SST: `/data3/luoq/p03_Ocean_SST_ensemble/kw_99_paper/Data/Nature_real_rebuild/raw/noaa_ersstv5_sst_monthly.nc`
- Nino3.4: `/data3/luoq/p03_Ocean_SST_ensemble/kw_99_paper/Data/Nature_real_rebuild/raw/cpc_ersst5_nino_monthly_1991_2020_base.txt`
- Gridded diagnostics: `Figure3_driver_method_diagnostics.nc`
- Binned persistence diagnostic: `Figure3_driver_persistence_validation.csv`
- Source-box sensitivity: `Figure3_driver_source_box_sensitivity.csv`
- Regime-threshold sensitivity: `Figure3_driver_threshold_sensitivity.csv`
- Machine-readable summary: `Figure3_driver_method_sensitivity.json`
