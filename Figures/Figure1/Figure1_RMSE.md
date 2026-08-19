# Supplementary Figure: MHW-intensity robustness of Figure 1a

## Caption

**The 2023-24 departure in Figure 1a is also evident in forecast errors for MHW intensity.** Event-mean global MHW intensity root-mean-square error (RMSE) is plotted against the peak centred 3-month CPC ERSSTv5 Nino3.4 anomaly for 9 historical El Nino events. The line and shading show the historical ordinary-least-squares fit and its 95% external prediction interval; 2023-24 was excluded from the fit. Black points identify the two strongest historical events used as visual anchors, and the red point and arrow show the observed 2023-24 intensity RMSE and its departure from the historical expectation. Forecasts are the four-model NMME ensemble mean, target aligned and aggregated over lead months 1-9. Lower RMSE indicates better intensity prediction.

## Real-data result

The historical event relationship was r=0.793 (P=0.0108). For the 2023-24 amplitude, the historical relationship predicted an intensity RMSE of 0.390 degrees C, whereas the observed event-window RMSE was 0.591 degrees C (residual +0.201 degrees C; external predictive P=0.010). The positive residual means that intensity errors were larger than expected, independently supporting the occurrence-skill breakdown diagnosed by AUC in Figure 1a.

## Metric and calculation

For grid cell i and target month t, the observed and forecast MHW intensities are threshold exceedances:

    I_obs(i,t) = max[SST_obs(i,t) - Q90(i,m(t)), 0]
    I_fcst(i,t,l) = max[SST_fcst(i,t,l) - Q90(i,m(t)), 0]

The verification mask contains valid ocean cells between 60S and 60N where the observation identifies a MHW, I_obs > 0. For each initialization and lead, the area-weighted intensity RMSE is:

    RMSE(t,l) = sqrt[ sum_i w_i (I_fcst - I_obs)^2 / sum_i w_i ]

where w_i is proportional to cos(latitude). Target-month RMSE across lead months 1-9 is sqrt(mean_l[RMSE(t,l)^2]); event RMSE across the 12 event months is sqrt(mean_t[RMSE(t)^2]). This preserves the root-mean-square definition at every aggregation level.

## Methods and sources

- Forecast: four-model NMME ensemble-mean SST, target-aligned lead months 1-9.
- Observation: ERSST SST and observation-defined MHW cells.
- MHW threshold: target-calendar-month 90th percentile for 1985-2014.
- Domain weighting: cosine-latitude weighting over valid ocean cells from 60S to 60N.
- Event windows, Nino3.4 data, historical highlights, colours and regression treatment are identical to main Figure 1a.
- Source metric table: /data3/luoq/p03_Ocean_SST_ensemble/kw_99_paper/output/Figure1/candidate09_metrics/Figure1_candidate09_metrics_by_init_lead.csv.
- Updated 2002/03 source-field metrics: /data3/luoq/p03_Ocean_SST_ensemble/kw_99_paper/Data/Nature_real_rebuild/derived/figure1_2002_03_current_source_metrics.csv; these 108 target-lead values replace the corresponding cached intensity-RMSE entries.
- Companion outputs: Figure1_RMSE_source_events.csv and Figure1_RMSE_statistics.json. The monthly and phase-slope tables are retained as audit products but are not displayed in this single-panel SI figure.

## Interpretation boundary

This metric is conditional on observed MHW occurrence and evaluates the magnitude of threshold-excess temperature error at observed-MHW grid cells. Forecast values below the threshold contribute zero predicted intensity and are therefore penalized as missed intensity. False alarms outside observation-defined MHW cells are not included; occurrence discrimination should be assessed with AUC or SEDI. RMSE is unbounded above and lower values indicate better forecasts.
