# Statistics and reproducibility

## Supplementary Fig. 1

a, Event-mean MHW-intensity root-mean-square error (RMSE) versus peak Niño3.4 anomaly. The regression and grey 95% prediction interval use nine earlier El Niño events; the red star marks 2023–2024, excluded from fitting. b, Monthly global intensity RMSE (404 months); shading marks the three strong events. c, Intensity RMSE in 2023–2024 minus the reference from 1997–1998 and 2015–2016, using 18-month windows and 11° × 11° same-basin neighbourhoods. Red indicates larger errors and blue smaller errors; grey ocean cells lack a valid comparison. The inset summarizes all seven analysis regions, retaining all nine leads and their range. TP, tropical Pacific; NP, North Pacific; SP, South Pacific; TA, tropical Atlantic; NA, North Atlantic; SA, South Atlantic; IO, Indian Ocean. Intensity is positive SST excess above the local 1985–2014 calendar-month 90th percentile. Verification includes only observed-MHW samples: smaller RMSE indicates better intensity prediction but does not assess false alarms outside observed MHWs.

Spatial windows run from April of the event year to September of the next year. The reference RMSE is the root of the equal-event mean squared error; regional summaries also aggregate squared errors before taking roots. Lead ranges are not confidence intervals. Panels a,b retain monthly spatial aggregation. The 1° display grid does not imply independent 1° information. Coverage and pooling sensitivity are shown in Supplementary Figs. 16–17.

Inputs: source_data/OLD/Supplementary_Figure1_monthly_t1_t9.csv, source_data/OLD/Supplementary_Figure1_source_events.csv, source_data/SPATIAL10/canonical_relative_changes.nc, source_data/SPATIAL10/regional_changes.csv, source_data/SPATIAL10/regional_changes_by_lead.csv

## Supplementary Fig. 2

a,b, Global MHW area under the ROC curve (AUC) against absolute Niño3.4 anomaly during El Niño and La Niña months, respectively. The thresholds are +0.5 °C and −0.5 °C; neutral months are outside this comparison. Lines are ordinary-least-squares fits and r is the monthly Pearson correlation. Both panels use the same AUC axis. The 95% correlation intervals are percentile intervals from 2,000 whole-calendar-year resamples; the same sampled years are used for both phases when estimating the slope difference. Months within years and successive ENSO months are dependent; they are not independent events. Year-block resampling addresses within-year dependence but may not capture the full duration of multiyear ENSO episodes. The 95% year-block intervals for the correlations are 0.31 to 0.78 for El Niño and −0.13 to 0.39 for La Niña. The El Niño minus La Niña slope difference is 0.062 AUC °C⁻¹ (95% year-block interval -0.011 to 0.128); the interval includes zero.

All phase-eligible months are retained; no independent-month P values or multiple-comparison significance symbols are used.

Inputs: source_data/F1/Figure1_monthly_t1_t9.csv

## Supplementary Fig. 3

a, Mean AUC versus midpoint year for the nine earlier El Niño events. b, Niño3.4 amplitude and AUC after removing their respective linear relationships with event year. c, The Niño3.4 coefficient in AUC = intercept + amplitude coefficient × Niño3.4 + year coefficient × event year, refitted after omitting each named event. Each point uses eight events; the red dashed line is the coefficient from all nine events. The 2023–2024 event is excluded throughout. These nine omission estimates describe sensitivity to event selection; their spread is not a confidence interval and they are not nine independent experiments. Adjustment for event year assesses linear temporal confounding only.

Nine historical events; deterministic ordinary-least-squares fits. Event amplitudes and AUCs are identical to main Fig. 1a.

Inputs: source_data/F1/Figure1_source_events.csv

## Supplementary Fig. 4

a, AUC expected from earlier events (dark line) and realized in 2023–2024 (red line), evaluated separately at each lead from 1 to 9 months. Connecting vertical segments show the expected-minus-realized difference. At each lead, AUC is averaged over July–June and regressed on El Niño amplitude using the nine earlier events; 2023–2024 is excluded. b, AUC averaged over leads 1–9 for each verifying month in the three strong events. The light-grey, dark-grey and red lines denote 1997–1998, 2015–2016 and 2023–2024. All 1,080 event–month–lead records are used: ten events × twelve months × nine leads. Leads and months are repeated evaluations, not independent events.

No lead-specific significance test is claimed. These panels retain the forecast extraction and monthly spatial-AUC definition of main Fig. 1a,b.

Inputs: source_data/F1/Figure1_source_events.csv, source_data/F4/sample_AUC_profiles.csv

## Supplementary Fig. 5

a,c,e, Cosine-latitude-weighted spatial correlations between forecast and observed DJF anomalies for SST, zonal wind stress and convection, represented by precipitation. All variables use NASA GEOS-S2S, the same four ensemble members (1–4), the same target months and leads, and the same tropical Pacific ocean mask (20°S–20°N, 140°E–100°W; 4,833 grid cells). The two historical strong El Niño events are shown separately. b,d,f, Relative correlation decrease: 100 × (r_reference − r_2023–2024)/r_reference, using each historical event and their equal-event mean. Positive values indicate deterioration; negative values indicate improvement. All right-hand panels share the same vertical scale. Lead 9 is unavailable and remains blank/NA; no nearest-lead substitution is used.

Forecast initialization equals the verifying month minus the labelled lead; the internal S coordinate must match that initialization and L must equal lead + 0.5. Monthly anomalies are calculated against 1992–2020, with separate lead-dependent model and observed climatologies using identical available calendar months. DJF is averaged before spatial correlation. The same ocean mask is used for all 72 valid event–variable–lead correlations. Eight leads within each of three events are repeated evaluations, not independent events; reference curves are sensitivity comparisons, not confidence intervals. This is a controlled single-model comparison, not a multimodel attribution of MHW forecast-skill loss.

Inputs: matched_s5/source_data/correlations.csv, matched_s5/source_data/correlation_changes.csv, matched_s5/statistics.json

## Supplementary Fig. 6

Rows show December–February (DJF) of 1997–1998, 2015–2016 and 2023–2024; columns show ERA5 200-hPa geopotential-height anomalies, multimodel forecast anomalies and forecast minus observed anomalies. Forecasts first average members within each model, then equally weight the three available models and the eight common future leads. Observed and forecast fields share the ±160 m colour scale; differences share ±100 m, with extensions marking values beyond the limits. Red denotes positive height anomalies or positive forecast error, blue negative values. Anomalies use calendar-month means from 1992–2020; February 2018 is omitted consistently from all model and observation climatologies because the corresponding GEOS-S2S field is missing. No event DJF month is excluded. Lead 9 is absent from the fixed three-model cohort. Maps retain their native 1° resolution without added smoothing.

Models are COLA–RSMAS–CCSM4, NASA–GEOS-S2S and GFDL–SPEAR. These are seasonal composites, not isolated causal ENSO responses. Signed averaging can cancel opposing errors; Fig. 8 also evaluates RMSE.

Inputs: source_data/F3/Figure3_Z200_fields.nc

## Supplementary Fig. 7

a–c, Forecast–observation spatial correlations of DJF 200-hPa height anomalies in 1997–1998, 2015–2016 and 2023–2024. d, Correlation in 2023–2024 minus the equal-event mean of the two earlier events. Panels a–c share the −1 to 1 scale; in d, red denotes lower and blue higher correlation. Grey cells labelled NA denote unavailable lead 9, not zero skill; 48 region–lead combinations are valid per event. NP, North Pacific; SP, South Pacific; TA, tropical Atlantic; NA, North Atlantic; SA, South Atlantic; IO, Indian Ocean. Here NA on a row label is North Atlantic, whereas NA inside a cell means not available. Pacific and Atlantic masks use ±23.5° and ±60° boundaries as in main Figs. 3–4; tropical Pacific is the source and is omitted from this six-recipient comparison. The full boundaries are in the coverage table.

These six ocean-mask diagnostics are not four idealized PNA/PSA/Indian/Atlantic waveguide boxes. No significance is inferred from the number of leads or grid cells.

Inputs: source_data/F3/SI_regional_skill_change.csv

## Supplementary Fig. 8

a, Global spatial correlation of forecast and observed DJF 200-hPa height anomalies. b, Corresponding area-weighted RMSE. Each row is one model or the equal-weight multimodel mean. Colours denote 1997–1998 (light grey), 2015–2016 (dark grey) and 2023–2024 (red). Large points show the mean over eight leads; small points retain all lead values; whiskers span their minimum and maximum. The multimodel mean is formed from fields before verification. The domain is all longitudes, 60°S–60°N, including land, weighted by cosine latitude. NCEP–CFSv2 is absent because 1997–1998 200-hPa data were unavailable in the common source set used for main Fig. 3; it is assessed separately in Fig. 14.

The plot includes 72 single-model and 24 multimodel event–lead evaluations from three events. Lead ranges are not confidence intervals. RMSE squares spatial errors before averaging and is therefore not reduced by cancellation of positive and negative errors.

Inputs: source_data/F3/Figure3_Z200_fields.nc

## Supplementary Fig. 9

a–c, Tropical Pacific precipitation forecast–observation correlation versus remote 200-hPa height forecast–observation correlation, shown separately for the three events. Symbols identify the six recipient regions defined in Fig. 7. Each panel contains all 48 region–lead pairs; r summarizes their within-event association. d, The pooled association for all events and after omitting each event. The pooled relationship compares both differences between events and differences within events, whereas a–c assess the latter alone. Each tropical lead value is repeated for six recipients. Thus 144 plotted pairs represent three event clusters, not 144 independent events; the weak within-event associations cannot establish a stable predictive relationship or one-way transmission of forecast error.

All values, including negative circulation correlations, are retained. Event omission is a sensitivity analysis, not out-of-event prediction validation. No independent-sample P value is reported.

Inputs: source_data/F3/Figure3b_all_samples.csv, source_data/F3/SI_event_omission.csv

## Supplementary Fig. 10

a, Global anomaly correlations of the 2023–2024 forecast (red) and the forecast composite of the two earlier strong events (dark grey) against each of six other El Niño winters, identified by start year. These are events in the main Fig. 1 list within the historical-reference window 1992–1993 to 2021–2022, excluding 1997–1998 and 2015–2016; 1991–1992 is outside that window. b, Forecast-to-observed spatial standard-deviation ratio of height anomalies. c, Skill against a zero-anomaly forecast, defined as 1 − MSE_forecast/MSE_zero-anomaly. Values above zero indicate lower error than climatology. Panels b,c use 30°–60°N and 40°–60°S along the sector from 120°E eastward to 20°E. The dashed line in b marks equal forecast and observed amplitude.

Individual-event correlations are not equivalent to correlation with the averaged observed field. A smaller forecast anomaly amplitude does not by itself establish better or worse forecast skill. No confidence intervals are estimated for these deterministic comparisons.

Inputs: source_data/F3/SI_climatology_distance_and_verification.csv, source_data/F3/SI_global_individual_el_nino_correlations.csv

## Supplementary Fig. 11

a,c, Observed (solid) and forecast (dashed) DJF 200-hPa height anomalies along 45°N and 50°S. Dark grey denotes the equal-event composite of 1997–1998 and 2015–2016; red denotes 2023–2024. b,d, Forecast minus observed height along the same sections. All 360 longitude nodes per section are retained, starting at 120°E and continuing eastward around the globe. The two latitudes sample the northern and southern midlatitude circulation to illustrate differences in the longitude and strength of anomaly centres; they do not identify fixed Rossby-wave trajectories or the seasonally most active waveguide. Positive errors indicate heights predicted too high and negative errors heights predicted too low, relative to the observed field, irrespective of the sign of the observed anomaly.

Forecasts average the eight common leads. Anomalies and source fields are identical to Fig. 6. The focus is longitudinal error structure, not a new estimate of global MHW skill.

Inputs: source_data/F3/Figure3cd_transects.csv

## Supplementary Fig. 12

a, Allocation of historical regional SST variance among ENSO, other climate modes, prior ocean thermal state, surface heating, horizontal advection and an unresolved fraction. The denominator is total SST variance. b, The three local components divided by their combined share, so the denominator is the local-group allocation, not total variance. Prior thermal state comprises SST and mixed-layer temperature one month earlier. Other modes comprise IOD, PDO, NAO and SAM indices. Both panels use 432 monthly records per region during 1985–2020, with the ±23.5° masks of main Fig. 4. Seasonal means and a linear trend are removed before group-respecting variance allocation. Shared explanatory information is averaged over predictor-group orders, with nested ordering within the local group.

These are statistical SST associations, not heat-budget fractions or measured contributions of process-specific forecast errors in 2023–2024. Their connection to main Fig. 4b is the explicitly defined weighting rule in Supplementary Methods.

Inputs: source_data/F4/SI_observed_local_process_hierarchy.csv

## Supplementary Fig. 13

This analysis uses contemporaneous observed climate indices, surface heating and horizontal advection, together with prior SST and mixed-layer temperature; it is diagnostic reconstruction, not an operational forecast. a, Later-period regional SST reconstruction using climate indices alone or all measured predictors. b, Skill decrease after removing each predictor group from the full model; negative values indicate improved reconstruction after removal. Three forward test blocks are 2003–2008, 2009–2014 and 2015–2020. Each model is trained only on earlier months beginning in 1985; seasonal terms, linear trend, scaling and coefficients are fitted within training data. Reconstruction skill is 1 − (model SSE / baseline SSE), where SSE is the sum of squared errors and the baseline is the training-fitted seasonal cycle plus linear trend. Squared errors are summed across test blocks before taking the ratio.

All seven regions and seven predictor configurations are retained. The three successive test blocks are not independent ENSO events. This score is not MHW AUC and does not validate the causal interpretation of the Sankey diagram.

Inputs: source_data/F4/Figure4_observed_association_samples.csv

## Supplementary Fig. 14

a, CFSv2 circulation RMSE versus net surface heat-flux RMSE, each divided by its own 1997–1998/2015–2016 mean at the same region and lead. Colours and symbols distinguish the three events. The dashed line is the pooled fit. b, Percentage RMSE increases in 2023–2024 relative to those earlier-event means, averaged over leads 1, 3 and 6 months. Red denotes greater error and blue smaller error. c, Correlations between the two normalized errors after omitting each event; the red dashed line denotes the pooled estimate. All masks now use the ±23.5° divisions of main Figs. 3–4. The 54 remote pairs comprise three events × six recipient regions × three leads; nine additional tropical-Pacific pairs enter a.

This is a separately downloaded and processed CFSv2-only check, not an independent model: CFSv2 also contributes to the main SST ensemble. Circulation is departure from contemporaneous zonal mean, verified against ERA5; net heat flux is verified against ORAS5 and is positive into the ocean. Four members and three DJF months are equally weighted. The circulation metric differs from the anomaly correlation in main Fig. 3; no causal or independent-event significance is inferred.

Inputs: source_data/BRIDGE/Figure4b_paired_error_fields.nc

## Supplementary Fig. 15

a, Area-weighted regional contributions to the net difference between the globally expected AUC (0.7050185) and the local neighbourhood AUC in 2023–2024, matching main Fig. 4. Red bars sum cells below that reference; blue hatched bars subtract cells above it; black diamonds show the net contribution. b, The same signed amounts allocated using historical SST-association weights. The three local components are shown separately and their parent group is not added again. Both panels divide by the same net map difference, 0.1065144 AUC. Positive shares sum to 120.79% and negative shares to −20.79%, giving 100% net. Regions use ±23.5° boundaries; Other seas comprises evaluated cells outside the seven masks.

The map uses April 2023–September 2024, nine leads and 11° same-basin neighbourhoods. Its area-weighted local-AUC difference is not the monthly spatial-AUC deficit in main Fig. 1 (0.1090497). Component weights describe historical SST associations, not causal contributions of process forecast errors. Unresolved is the unallocated association weight. Supplementary Methods gives the signed-sum identity and distinguishes the two AUC metrics.

Inputs: source_data/CURRENT4/association_weighted_local_shortfall.csv, source_data/CURRENT4/regional_map_shortfall.csv

## Supplementary Fig. 16

a, Mean 1997–1998/2015–2016 neighbourhood AUC minus 2023–2024 AUC. c, 2023–2024 conditional intensity RMSE minus the two-event reference. Bars show region summaries; small points and whiskers retain all nine lead values and their minimum–maximum range. Positive differences indicate poorer forecasts in 2023–2024; negative differences indicate improvement. b,d, Cosine-latitude-weighted area coverage where all three events and all nine leads are evaluable for the respective metric. Denominators are the full original analysis-region ocean areas, not the already valid cells. TP, tropical Pacific; NP, North Pacific; SP, South Pacific; TA, tropical Atlantic; NA, North Atlantic; SA, South Atlantic; IO, Indian Ocean. These ±20° sensitivity-analysis boundaries differ from the ±23.5° masks in main Figs. 3–4 and Supplementary Fig. 15.

All spatial comparisons use April–September of the following year and 11° × 11° same-basin ocean neighbourhoods. AUC requires both observed classes; missing comparisons stay undefined. Neighbourhoods overlap and months/leads are dependent. Regional AUC differences are not additive contributions to the global monthly spatial-AUC gap. Lead ranges are descriptive, not confidence intervals.

Inputs: source_data/SPATIAL10/regional_changes.csv, source_data/SPATIAL10/regional_changes_by_lead.csv

## Supplementary Fig. 17

Four combinations compare the original July–June window (12 months) and its symmetric three-month extension (April–September of the following year; 18 months), evaluated at individual grid cells or within fixed 11° × 11° neighbourhoods. a,c, Regional AUC decreases and conditional intensity RMSE increases. b,d, Their common valid ocean-area coverage. Red diamonds identify the 18-month, 11° configuration used in Supplementary Figs. 1c and 16. All three events use the same window, lead range and neighbourhood definition. The 12-month, single-cell calculation provides the unpooled reference. All seven fixed regions are retained; no region is removed because its sign differs. Individual-cell estimates use each grid cell separately; pooled estimates combine samples within an 11° × 11° same-basin neighbourhood.

Each point is a region-level summary, not an independent replicate or significance estimate. The longer window changes the seasonal/event-life-cycle support. The larger neighbourhood changes spatial support; higher coverage is not evidence of higher independent spatial resolution or a more accurate forecast. This sensitivity analysis retains the main archived input profile and its documented source-selection limitations.

Inputs: source_data/SPATIAL10/coverage_comparison.csv, source_data/SPATIAL10/regional_sensitivity.csv