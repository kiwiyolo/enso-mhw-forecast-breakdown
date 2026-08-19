# Supplementary Figure: SEDI sensitivity analysis for Figure 1a

## Caption

**The direction of the breakdown identified by AUC in Figure 1a is consistent under a threshold-dependent rare-event metric.** Event-mean global symmetric extremal dependence index (SEDI) is plotted against the peak centred 3-month CPC ERSSTv5 Nino3.4 anomaly for 9 historical El Nino events. The line and shading show the historical ordinary-least-squares fit and its 95% external prediction interval; 2023-24 was excluded from the fit. Black points identify the two strongest historical events used as visual anchors, and the red point and arrow show the observed 2023-24 SEDI and its departure from the historical expectation. Forecasts are the four-model NMME ensemble mean, target aligned and averaged over lead months 1-9.

## Real-data result

The historical event relationship was r=0.892 (P=0.0012). For the 2023-24 amplitude, the historical relationship predicted SEDI 0.347, whereas the observed event-window forecast skill was 0.193 (residual -0.154; external predictive P=0.151). The negative residual is directionally consistent with Figure 1a, but it does not independently pass a two-sided 0.05 external predictive test; it should therefore be interpreted as sensitivity evidence rather than a second significant detection.

## Metric and calculation

SEDI is calculated from the cosine-latitude-area-weighted hit rate H and false-alarm rate F:

    SEDI = [ln(F) - ln(H) - ln(1-F) + ln(1-H)]
           / [ln(F) + ln(H) + ln(1-F) + ln(1-H)]

Rates are clipped to [1e-6, 1-1e-6] before taking logarithms. SEDI ranges from -1 to 1; higher values indicate better discrimination of rare MHW occurrence at the selected binary forecast threshold, while zero denotes no extremal association. Unlike AUC, SEDI is threshold-dependent.

## Methods and sources

- Forecast: four-model NMME ensemble-mean SST, target-aligned lead months 1-9.
- Observed event: ERSST SST exceeds the target-calendar-month 90th-percentile MHW threshold.
- Forecast event: NMME forecast SST exceeds the same threshold.
- MHW climatology: 1985-2014, calculated separately for each calendar month.
- Domain weighting: cosine-latitude weighting over valid ocean cells from 60S to 60N.
- Event windows, Nino3.4 data, historical highlights, colours and regression treatment are identical to main Figure 1a.
- Source metric table: /data3/luoq/p03_Ocean_SST_ensemble/kw_99_paper/output/Figure1/candidate09_metrics/Figure1_candidate09_metrics_by_init_lead.csv.
- Updated 2002/03 source-field metrics: /data3/luoq/p03_Ocean_SST_ensemble/kw_99_paper/Data/Nature_real_rebuild/derived/figure1_2002_03_current_source_metrics.csv; these 108 target-lead values replace the corresponding cached SEDI entries.
- Companion outputs: Figure1_SEDI_source_events.csv and Figure1_SEDI_statistics.json. The monthly and phase-slope tables are retained as audit products but are not displayed in this single-panel SI figure.

## Interpretation boundary

SEDI diagnoses binary rare-event association and complements the threshold-independent ranking information in AUC. Differences between the SEDI and AUC panels are expected because SEDI depends on the physical MHW threshold and the resulting hit and false-alarm rates.
