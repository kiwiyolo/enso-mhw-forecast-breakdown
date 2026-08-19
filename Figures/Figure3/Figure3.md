# Figure 3

## Caption

**Remote MHW forecast skill reflected competition between El Nino teleconnections and basin-local drivers.** **a**, Dominant driver regime from historical ERSST SST-Nino3.4 association, lag-1 persistence after removing Nino3.4, and their mixed regime, displayed over the 60S-60N analysis domain. Contours show observed 2023/24 monthly MHW threshold exceedance at 0.25, 0.50 and 0.90 degrees C, with increasing line width; stippling marks basins with mean AUC loss above 0.05. The complete map legend is arranged vertically to the right. **b**, MHW-intensity-weighted continuous association shares by basin. **c**, NMME-ERA5 Z200 pattern agreement for four bridge regions, standardized within region and comparing the comparable-event mean with 2023/24. **d**, Annual MHW activity in the `6` basins classified as basin-local dominated. Thin lines show individual basins, the blue line is their equal-basin mean, and the dashed line is its linear trend.

## Information moved from the figure

Exact teleconnection z-scores, all continuous association shares and the complete annual activity series remain in `Figure3_teleconnection_efficiency.csv`, `Figure3_driver_contributions.csv` and `Figure3_local_mhw_activity.csv`. Panel b reuses the Figure 3 palette in fixed stack order: Direct ENSO, Remote ENSO, Basin-local and Residual; its redundant legend is omitted from the figure. The residual share is not the same quantity as the categorical Mixed class in panel a. Mixed is assigned after applying the 50% dominance and 15-percentage-point separation thresholds, whereas the residual share is the normalized continuous remainder `max(0.05, 1 - ENSO score - persistence score)`. The Theil-Sen trend, Kendall statistic and P value are reported below and in `Figure3_local_mhw_activity_statistics.json`.

## Visual design

The main figure uses a restrained Nature-style information hierarchy: no in-figure headline, short panel titles, a shared mechanism palette, direct labelling only where it carries the central result, and detailed statistical or methodological annotation in this companion document. The visual benchmark was the main-figure treatment in England et al., *Nature* (2025), https://www.nature.com/articles/s41586-025-08903-5, and Peng et al., *Nature Geoscience* (2025), https://www.nature.com/articles/s41561-025-01700-9. This is a design reference only; all plotted values are generated from the data and methods documented here.


## Real-data result

The global 2023-24 mean MHW threshold exceedance was `0.239` degrees C. The equal-basin MHW activity index increased by a Theil-Sen estimate of `10.44` degree-C days per decade over `34` years (Kendall `tau=0.622`, `P=2.29e-07`). This documents increasingly active aggregate MHW conditions across basin-local-dominated oceans.

## Interpretation boundary

Driver regimes and panel-b shares are association-based and should not be described as causal fractions. Panel a and all reported basin and global calculations use the same 60S-60N domain. Panel b is currently weighted by positive 2023/24 MHW threshold exceedance at each grid cell; it is an intensity-weighted grid-cell mean rather than a strict cosine-latitude area-weighted attribution. The annual activity proxy is the calendar-year sum of monthly, area-weighted positive SST exceedance above the fixed 1985-2014 calendar-month 90th percentile, multiplied by days per month. It combines intensity and persistence at monthly resolution; it is evidence of an aggregate trend, not proof that local processes caused the trend or that every basin was simultaneously active. Stippling inherits basin-scale skill estimates and is not a grid-cell significance test. The comparable-event reference is the equal mean of 1997/98 and 2015/16; exact events are documented here rather than labelled separately in the panel. All source definitions are in `/data3/luoq/p03_Ocean_SST_ensemble/kw_99_paper/Data/Nature_real_rebuild/derived/calculation_provenance.json`.
