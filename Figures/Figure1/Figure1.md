# Figure 1

## Caption

**Breakdown of the historical relationship between El Nino amplitude and global marine-heatwave forecast skill in 2023-24.** **a**, Event-mean global MHW AUC against the peak centred 3-month CPC ERSSTv5 Nino3.4 anomaly for 9 historical events. The line and shading are the historical ordinary-least-squares fit and its 95% external prediction interval; 2023-24 was excluded from that fit. **b**, Monthly NMME ensemble-mean MHW AUC after target-aligned averaging over lead months 1-9; shading marks 1997/98, 2015/16 and 2023/24, and the strip shows CPC Nino3.4. **c**, Pearson correlations between absolute Nino3.4 intensity and monthly AUC, shown separately for El Nino and La Nina months. Whiskers are 95% calendar-year block-bootstrap intervals; labels give the ordinary Pearson correlation and its two-sided significance.

## Information moved from the figure

The main graphic deliberately labels only the two highlighted historical events and 2023/24. Names and exact coordinates for all historical events are retained in `Figure1_source_events.csv`. The external predictive P value, sample size, exact expected and observed AUC values, bootstrap intervals and El Nino-La Nina contrast are reported below and in `Figure1_statistics.json` and `Figure1_phase_slopes.csv`; they are omitted from the plot to keep the visual hierarchy centred on the historical fit and the 2023/24 residual.

## Visual design

The main figure uses a restrained Nature-style information hierarchy: no in-figure headline, short panel titles, a shared mechanism palette, direct labelling only where it carries the central result, and detailed statistical or methodological annotation in this companion document. The visual benchmark was the main-figure treatment in England et al., *Nature* (2025), https://www.nature.com/articles/s41586-025-08903-5, and Peng et al., *Nature Geoscience* (2025), https://www.nature.com/articles/s41561-025-01700-9. This is a design reference only; all plotted values are generated from the data and methods documented here.


## Real-data result

The historical event relationship was `r=0.940` (`P=0.0002`). For the 2023-24 amplitude, the historical relationship predicted AUC `0.705`, whereas the observed event-window forecast skill was `0.596` (residual `-0.109`; external predictive `P=0.010`). Monthly skill rose much more consistently with intensity during El Nino (`r=0.601`, `P=4.66e-11`) than during La Nina (`r=0.153`, `P=0.0863`). Panel c presents the two relationships directly and does not plot a derived difference bar.

The queried 2002/03 value was independently recomputed from the four raw NMME SST archives. All 108 target-month/lead values matched the stored table exactly (maximum absolute difference `0.0e+00`), giving event-mean AUC `0.608530`. No manual correction was applied.

## Methods and sources

- NMME MHW forecasts: four-model ensemble mean, target-aligned lead months 1-9.
- Verification: ERSST MHW occurrence between 60S and 60N; monthly 90th-percentile threshold using 1985-2014.
- Nino3.4: newly downloaded NOAA CPC ERSSTv5 index.
- Comparable-event reference used in subsequent figures: equal mean of 1997/98 and 2015/16.
- Derived tables: `Figure1_source_events.csv`, `Figure1_phase_slopes.csv`, `Figure1_2002_03_AUC_audit.csv`, `Figure1_statistics.json`.
- Download provenance: `/data3/luoq/p03_Ocean_SST_ensemble/kw_99_paper/Data/Nature_real_rebuild/raw/download_manifest.json`.
