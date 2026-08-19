# Figure 4 | Multiple pathway failures underpinned the 2023-2024 MHW forecast-skill breakdown

## Main message

The 2023-2024 forecast failure did not begin in one place. Part of the useful signal was lost in the ENSO source signal, more was lost during teleconnection transmission towards distant oceans, and a further part was lost through basin-local processes. Figure 4 follows that chain from the initial error to the final loss of marine-heatwave forecast skill.

## Caption

**Failures at several stages contributed to the 2023-2024 marine-heatwave forecast breakdown.** **a**, The global loss in forecast skill identified in Figure 1 (`0.109` AUC) is divided among six ocean basins. Longer bars indicate places that contributed more to the global loss. Colours identify ENSO source-signal error, teleconnection transmission error, basin-local process error and the unresolved remainder, using the terminology of Figures 2 and 3. **b**, The same calculation was repeated using different earlier-event references and forecast-lead selections. Diamonds show the primary result, points show 14 alternatives and vertical lines show their full range. **c**, The `72` independently held-out basin-event-lead predictions are ordered by their pathway-inferred skill loss and divided into three equal groups (`n=24` each). The single sector summarizes how closely the three inferred group means reconstruct their actual counterparts (`96%`).

## Questions answered

The figure follows one quantitative sequence:

1. **Where was skill lost?** Panel a shows which oceans contributed most and at which stage of the forecast chain errors appeared.
2. **Does that result depend on one analysis choice?** Panel b repeats the calculation after changing the earlier-event reference and the forecast months included.
3. **Do these errors help anticipate losses not used to fit the model?** Panel c tests that question on withheld samples.

## Visual design

The main figure uses a restrained Nature-style information hierarchy: no in-figure headline, short panel titles, a shared mechanism palette, direct labelling only where it carries the central result, and detailed statistical or methodological annotation in this companion document. The visual benchmark was the main-figure treatment in England et al., *Nature* (2025), https://www.nature.com/articles/s41586-025-08903-5, and Peng et al., *Nature Geoscience* (2025), https://www.nature.com/articles/s41561-025-01700-9. This is a design reference only; all plotted values are generated from the data and methods documented here.


## Real-data result

Panel a retains the original numerical result exactly. ENSO source-signal error accounts for `30.5%` of the global loss, teleconnection transmission error for `21.9%`, and basin-local process error for `23.3%`; the remaining `24.2%` is unresolved. Across the 14 alternatives in panel b, the three diagnosed shares range from `26.2-34.5%`, `15.3-30.0%`, and `17.8-29.6%`. Thus all three stages remain relevant when the analysis choices change, although their exact ranking does not.

Across withheld samples, the three measured error routes reproduced `55.2%` of the variation in forecast-skill loss (`R2=0.552`), with a typical error of `0.083` AUC. After grouping the held-out predictions into low, medium and high inferred-loss thirds, the mean absolute difference between inferred and actual group means was only `0.015` AUC. This is an out-of-sample representativeness check: the observations were not used to fit their corresponding model, and neither the individual predictions nor the grouped means were adjusted to force closure of the global loss.

Panel c gives the following direct comparison:

| Inferred-loss level | Inferred mean | Actual mean | Absolute difference | Samples |
|---|---:|---:|---:|---:|
| Low | -0.067 | -0.073 | 0.006 | 24 |
| Medium | 0.027 | 0.016 | 0.010 | 24 |
| High | 0.186 | 0.157 | 0.029 | 24 |

The sample-level out-of-sample statistics (`R2=0.552`, RMSE `0.083` AUC), the group-mean absolute difference (`0.015` AUC) and all three group-specific comparisons are documented here rather than annotated in the panel.

The single percentage displayed in panel c is the grouped reconstruction agreement:

```text
agreement = 100 x [1 - sum((actual_g - inferred_g)^2)
                     / sum((actual_g - mean(actual))^2)]
          = 96.4%
```

This compact score compares the inferred and actual means jointly across the low, medium and high groups and penalizes disagreement in magnitude. The plotted sector is clipped to the display range 0-100%, although the present value lies within that range. It is not a confidence level, classification accuracy or replacement for the sample-level cross-validated `R2`; it is a descriptive calibration summary based on three aggregated group means.

## Data and cross-validation

- Independent validation unit: `basin x event x lead` (`72` rows from `3` events, `6` basins and lead months 1, 3, 6 and 9).
- ENSO source-signal diagnostic: tropical-Pacific forecast-observation source-pattern fidelity inherited from Figure 2.
- Teleconnection transmission diagnostic: NMME-ERA5 Z200 anomaly pattern fidelity in the PNA, PSA, Indian and Atlantic bridge domains.
- Basin-local process signal in panels a-b: the original archived quantity `1 - precipitation-pattern fidelity`, multiplied by the Figure 3 basin-local association share. It is retained to reproduce the previously approved panel-a values. It is a broad basin-local response proxy, not a direct local-ocean forecast-error measurement.
- Basin-local diagnostic in panel c: forecast-observation error in the basin SST threshold-excess pattern from Figure 3. This stricter quantity is used only for independent reconstruction and does not alter panel a.
- Target: basin MHW AUC at the same event and lead.
- Reference model: fixed basin and lead indicators only. These represent the typical difficulty of each region and forecast range without using pathway information.
- Cross-validation: each event is held out in full, so no lead or basin from that event enters model fitting. This prevents event leakage.
- Panel-c aggregation: all 72 out-of-sample predictions are ranked by inferred loss and divided into three equal-sized groups. Grouping is applied only after cross-validation. The lower and upper tertile boundaries are approximately `-0.015` and `0.084` AUC. The sector summarizes agreement across the three group means; sample-level `R2` and RMSE remain the formal performance statistics.
- Sensitivity analysis in panel b: the primary definition uses the equal 1997/98 and 2015/16 reference and leads 1, 3, 6 and 9. Alternatives use either reference event separately and the complete or leave-one-lead-out sets, yielding 15 definitions including the primary. These are structured sensitivity cases, not confidence intervals or independent replicates. Full values are stored in `Figure4_allocation_sensitivity.csv`.

## Panel-a accounting

Panel a remains an **association-based diagnostic accounting**, not an independent causal decomposition. The Figure 1 relationship gives expected global AUC `0.705`; observed 2023/24 AUC is `0.596`, producing the plotted gap of `0.109`. Raw positive comparable-event minus 2023/24 basin deficits sum to `0.349`; they are normalized to shares and multiplied by `0.109`. Therefore the basin bar lengths are allocated shares of the global gap, not the raw basin AUC differences. The scaling factor is `0.3124`.

The six allocated basin bars sum to `0.109`, and the four segments of every bar close to its allocated total within numerical precision. The original descriptive path-model R2 (`0.758`) defines the accounted-for fraction. Within each basin, that amount is divided in proportion to the archived source, Z200 transmission and regional-process indices; the remainder is labelled not accounted for. These are the same numbers used in the earlier Figure 4. They identify co-occurring diagnostic pathways and must not be described as counterfactual causal effects.

No confidence interval is drawn on the panel-a basin totals because the available accounting table contains one event-composite deficit per basin rather than independently resampled monthly basin-deficit fields. Panel b provides a definition-sensitivity analysis instead of a sampling-confidence interval. Independent out-of-sample evidence is confined to panel c.

## Critical interpretation boundary

Only three events have complete common pathway fields. Panel b tests dependence on the two available historical references and four archived leads, but cannot replace a larger event sample. ENSO source-signal and teleconnection transmission errors are physically and statistically related. The basin-local process signal retained in panels a-b includes precipitation-pattern fidelity and is therefore a broad regional-response proxy, not a pure ocean-process attribution. The Atlantic bridge is shared by the three Atlantic basin targets. The title uses **underpinned** to denote convergent diagnostic and held-out predictive evidence, not formal causal mediation. Dedicated perturbation hindcasts and longer process archives would be required for causal attribution.

The primary earlier-event reference is the equal mean of 1997/98 and 2015/16. Reproducible outputs are `Figure4_skill_loss_attribution.csv`, `Figure4_allocation_sensitivity.csv`, `Figure4_cross_validated_models.csv`, `Figure4_error_reduction.csv`, `Figure4_out_of_sample_reconstruction.csv`, `Figure4_out_of_sample_calibration.csv` and `Figure4_attribution_summary.json`. Source definitions and checksums are recorded in `/data3/luoq/p03_Ocean_SST_ensemble/kw_99_paper/Data/Nature_real_rebuild/derived/calculation_provenance.json`.
