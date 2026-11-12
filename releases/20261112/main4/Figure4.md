# Figure 4 | Regional MHW forecast shortfalls and their historical SST associations

## Caption

**a**, Local MHW forecast skill relative to the global AUC expected from historical El Niño intensity. Colours show the uniform expected global AUC (0.705) minus local 2023–2024 AUC: red indicates skill below that benchmark, blue above it. Local AUC is calculated using the unchanged fixed 11° × 11° same-basin ocean-cell neighbourhoods for April 2023–September 2024 and equally averaged over leads 1–9. Grey ocean areas remain undefined. The native 1° display centres, valid-cell support and colour range (−1 to +1 AUC) are retained; no interpolation or smoothing is added. The effective diagnostic support is 11°, not independent 1° resolution. Dashed boundaries distinguish TP, NP, SP, TA, NA, SA and IO, using the same 23.5° tropical boundaries and major-basin definitions as Fig.3. Inset bars and their year-block bootstrap intervals are unchanged historical regional SST association shares for ENSO, other climate modes, local processes and an unresolved fraction; they are not shares of the current forecast shortfall.

**b**, Regional accounting of lower and higher forecast skill relative to the global benchmark in a. Positive mapped differences (Lower forecast skill, +) and negative mapped differences (Higher forecast skill, −) are area-weighted and summed separately within each region, after averaging across leads. All percentages use the net mapped shortfall (total positive shortfall minus the higher-skill magnitude) as their denominator. Positive shares sum to +120.8% and negative shares to −20.8%, giving a signed total of 100%. Both signs remain separately visible rather than being cancelled within regions. Solid-filled ribbons indicate lower skill; paler ribbons with dashed centre lines indicate higher skill. Ribbon widths use the same magnitude scale for both signs; the fixed-width dashed lines are visibility aids, not quantitative widths. Paired labels give positive / negative percentages at every region and association node. Other seas retains valid cells outside the seven named basins. Each signed amount is distributed across the unchanged historical SST association weights; local processes retain the earlier subdivision into prior ocean thermal state, surface heating and horizontal-advection proxies. The ribbons describe association-weighted accounting, not independently estimated causal effects of process-specific forecast errors. Unresolved components use grey stippling, distinct from purple Other climate modes.

## Methods and exact values

### Display hierarchy (v27; unchanged from v26)

Panel b follows **Regions → Positive / Negative → Drivers → Local components**.
The inserted sign nodes collect positive and negative amounts separately before
redistributing their totals to the unchanged driver groups. Region-to-sign
ribbons are neutral grey; downstream colours identify the driver categories.
Dashed locators continue to identify higher-skill (negative) flows. The aggregated
sign nodes simplify the diagram but do not depict individual region-to-driver
paths; their full numerical allocation remains in the unchanged source table.
The heading “Drivers” follows the requested reference nomenclature and retains
the association-based, non-causal interpretation described above. The combined
category and Lower/Higher forecast-skill legend is restored beside the colourbar,
in the reference two-column, three-row arrangement. No historical activity map,
old percentages or old denominator has been restored.

Let E = 0.7050185254528283 and D(g) = E − mean_lead AUC(g). The historical nine-event regression excludes 2023–2024. For valid mapped cells, let a(g) = cos(latitude(g)) / sum_valid cos(latitude). Define:

- L(region) = sum_region a(g) max[D(g), 0].
- H(region) = sum_region a(g) max[−D(g), 0].
- Net denominator N = sum_all_regions [L(region) − H(region)] = 0.10651437039268086 AUC.
- Regional below-benchmark share = 100 × L(region) / N.
- Regional higher-skill signed percentage = −100 × H(region) / N.
- For component k, positive ribbon width is proportional to L(region) × historical_association_weight(region,k); negative ribbon width uses H(region) × that same weight. Both use one width scale.

The area-weighted positive local shortfall is **0.1286593 AUC**, the separately retained above-benchmark component is **0.0221449 AUC**, and their net difference is **0.1065144 AUC**. These are summaries of local neighbourhood AUC, **not** the global AUC estimator or its unchanged **0.1090497** gap in Fig.1. They must not be rescaled to force equality with that gap.

The signed totals are +120.790563% and −20.790563%, summing to 100%. For example, TP is +16.208278% / −10.024685%, with a net contribution of 6.183594% to the mapped net shortfall. Complete displayed regional percentages are in render_validation.json under signed_regional_accounting.

All driver and local-component percentages are divided by the same N; their signed totals also close to their corresponding parent nodes. These are descriptive allocation shares, not evidence that process errors caused those fractions of the collapse. All inherited input tables and statistics.json are retained unchanged for provenance: percentage fields explicitly based on the positive total in those legacy inputs are not the v27 plotted percentages. The active denominator is recorded in render_validation.json and contract.json.

All 27,508 previously defined mapped cells are retained. Of these, 159 lie outside the seven named major-basin masks and remain in Other seas; 1,297 originally undefined cells stay missing. The common-valid area coverage is 96.0576%. Boundaries are evaluated directly on each native grid using the same geographic rule; no label-only reassignment or interpolation of regional statistics is used.

## Uncertainty and limitations

The unchanged panel-a association intervals use the existing 400 year-block bootstrap replicates and 432 monthly observations (January 1985–December 2020). The map and new regional shares are descriptive point estimates; lead-wise sensitivity is exported separately in source_data/regional_lead_sensitivity.csv. Overlapping neighbourhoods are not independent spatial samples. No new significance claim or confidence interval is attached to the Sankey.

The expected value is a **global reference**, not a fitted local ENSO-conditioned expectation. Its source regression uses July–June monthly global spatial AUC, whereas the mapped local field uses 18-month pooled neighbourhood AUC. Thus a positive difference means lower skill than the global benchmark, not necessarily a decline from that grid cell's own historical expected skill. The uniform benchmark contributes no additional spatial structure beyond the inverse ordering of local AUC.

The archived forecast-selection profile and its previously documented initialization-coordinate limitations are inherited unchanged at the user's direction. This revision provides a map-consistent descriptive figure, not a resolution of that source-data issue or a claim of submission-level causal attribution. The historical activity map and all original figures remain archived, not overwritten. Manuscript and SI text have not been silently modified; any old Fig.4 global-gap allocation percentages must be updated before use with this figure.

## Sources and reproduction

Source checksums are in input_manifest.json. The signed field is in source_data/Figure4a_local_shortfall.nc; region totals and coverage are in regional_map_shortfall.csv; the underlying regional component allocations are in association_weighted_local_shortfall.csv. This v27 revision changes only the displayed percentage denominator, using unchanged v26 input amounts. Run plot_figure.py and package.py in the ocean_sst environment. The entire panel a is pixel-identical to v26; Fig.1–3 PDFs are byte-identical. Layout, palette, legend and flow-width ratios are unchanged.
