"""Reader captions for all seventeen figures; scientific limits remain explicit."""
from copy import deepcopy
from captions import CAPTIONS as OLD_CAPTIONS, METHODS as OLD_METHODS, COVERAGE as OLD_COVERAGE
CAPTIONS=deepcopy(OLD_CAPTIONS)
METHODS=deepcopy(OLD_METHODS)
COVERAGE=deepcopy(OLD_COVERAGE)

TITLES={
1:'MHW-intensity forecast errors in time and space',
2:'ENSO-phase associations with MHW forecast skill',
3:'Sensitivity of the ENSO–skill relationship to event selection',
4:'MHW forecast skill across lead months and event development',
5:'Tropical forecast correlations and historical-event comparisons',
6:'Observed and forecast upper-level circulation in three El Niño events',
7:'Regional circulation forecast skill across lead months',
8:'Circulation forecast correlations and RMSE across models',
9:'Within-event and between-event tropical–remote error relationships',
10:'Circulation pattern similarity, amplitude and error relative to climatology',
11:'Longitudinal differences in observed and forecast height anomalies',
12:'Historical SST variance associated with climate modes and local processes',
13:'Regional SST reconstruction in later, withheld periods',
14:'CFSv2-only circulation and surface heat-flux error comparison',
15:'Regional and component accounting of lower and higher forecast skill',
16:'Regional skill changes and valid ocean-area coverage',
17:'Sensitivity to verification duration and spatial pooling'}
for n,title in TITLES.items():
    CAPTIONS[n]['title']=title

CAPTIONS[1]['caption']=(
    'a, Event-mean MHW-intensity root-mean-square error (RMSE) versus peak Niño3.4 anomaly. '
    'The regression and grey 95% prediction interval use nine earlier El Niño events; the red star marks '
    '2023–2024, excluded from fitting. b, Monthly global intensity RMSE (404 months); shading marks '
    'the three strong events. c, Intensity RMSE in 2023–2024 minus the reference from 1997–1998 and '
    '2015–2016, using 18-month windows and 11° × 11° same-basin neighbourhoods. Red indicates larger '
    'errors and blue smaller errors; grey ocean cells lack a valid comparison. The inset summarizes '
    'all seven analysis regions, retaining all nine leads and their range. TP, tropical Pacific; '
    'NP, North Pacific; SP, South Pacific; TA, tropical Atlantic; NA, North Atlantic; SA, South Atlantic; '
    'IO, Indian Ocean. Intensity is positive SST '
    'excess above the local 1985–2014 calendar-month 90th percentile. Verification includes only '
    'observed-MHW samples: smaller RMSE indicates better intensity prediction but does not assess '
    'false alarms outside observed MHWs.')
CAPTIONS[1]['statistics']=(
    'Spatial windows run from April of the event year to September of the next year. The reference '
    'RMSE is the root of the equal-event mean squared error; regional summaries also aggregate squared '
    'errors before taking roots. Lead ranges are not confidence intervals. Panels a,b retain monthly '
    'spatial aggregation. The 1° display grid does not imply independent 1° information. Coverage and '
    'pooling sensitivity are shown in Supplementary Figs. 16–17.')
CAPTIONS[5]['caption']=(
    'a,c,e, Forecast–observation spatial correlations for SST, zonal wind stress and precipitation '
    '(a proxy for convection). Darker blue indicates stronger agreement. Grey cells labelled NA lack '
    'a verified exact lead: wind-stress label 9 is excluded because the source used nearest-lead '
    'substitution. b,d,f, Relative correlation decreases in 2023–2024: '
    '100 × (r_reference − r_2023–2024)/r_reference. Light grey, dark grey and red use 1997–1998, '
    '2015–2016 and their equal-event mean as the reference, respectively. Positive values indicate '
    'lower correlation. References with nonpositive correlation are undefined. The displayed labels '
    'follow the source table; variable-specific windows and model availability are detailed in '
    'Supplementary Methods and Table 1.')
CAPTIONS[5]['statistics']=(
    'Three events; 78 displayed coefficients from 81 archived records after excluding three '
    'unverified wind-stress values. Repeated leads are not independent events. Reference curves are '
    'not confidence bounds. Percentages use different denominators and cannot be added as '
    'contributions to MHW skill loss; the variables are not a matched-cohort coupling experiment.')
CAPTIONS[15]=dict(title=TITLES[15],role='Exact signed accounting of the current main Figure 4 map',
caption=(
    'a, Area-weighted regional contributions to the net difference between the globally expected AUC '
    '(0.7050185) and the local neighbourhood AUC in 2023–2024, matching main Fig. 4. Red bars sum '
    'cells below that reference; blue hatched bars subtract cells above it; black diamonds show '
    'the net contribution. b, The same signed amounts allocated using historical SST-association '
    'weights. The three local components are shown separately and their parent group is not added '
    'again. Both panels divide by the same net map difference, 0.1065144 AUC. Positive shares sum '
    'to 120.79% and negative shares to −20.79%, giving 100% net. Regions use ±23.5° boundaries; '
    'Other seas comprises evaluated cells outside the seven masks.'),
statistics=(
    'The map uses April 2023–September 2024, nine leads and 11° same-basin neighbourhoods. '
    'Its area-weighted local-AUC difference is not the monthly spatial-AUC deficit in main Fig. 1 '
    '(0.1090497). Component weights describe historical SST associations, not causal contributions '
    'of process forecast errors. Unresolved is the unallocated association weight. Supplementary '
    'Methods gives the signed-sum identity and distinguishes the two AUC metrics.'))
CAPTIONS[16]['role']='All-region spatial skill changes and their valid-area support'
CAPTIONS[16]['caption']=CAPTIONS[16]['caption'].replace(
    'Canonical-minus-2023–2024 neighbourhood AUC','Mean 1997–1998/2015–2016 neighbourhood AUC minus 2023–2024 AUC').replace(
    'Target-minus-canonical conditional intensity RMSE','2023–2024 conditional intensity RMSE minus the two-event reference').replace(
    'The retained ±20° boundaries match main Fig. 1c, not the ±23.5° process-analysis masks.',
    'These ±20° sensitivity-analysis boundaries differ from the ±23.5° masks in main Figs. 3–4 and Supplementary Fig. 15.')
CAPTIONS[17]['caption']=CAPTIONS[17]['caption'].replace('Four predeclared combinations','Four combinations').replace(
    'Red diamonds identify the new main-Fig. 1c and Supplementary-Fig. 1c configuration.',
    'Red diamonds identify the 18-month, 11° configuration used in Supplementary Figs. 1c and 16.').replace(
    'The original 12-month, single-cell results are independently replayed before pooling.',
    'The 12-month, single-cell calculation provides the unpooled reference.')

for i,(head,body) in enumerate(METHODS):
    if head=='Forecast extraction shared with the main SST figures':
        body=body.replace('Figures 1–4, 15 and 16','Figures 1–4 and 15–17').replace(
            'This source limitation must be resolved together with the main-text SST analysis before submission claims of robust magnitude are made.',
            'The magnitude of the SST-derived skill differences is therefore conditional on this extraction rule.')
    if head=='Scope of the supplementary analyses':
        body=body.replace('global AUC difference','regional map difference').replace('global AUC accounting','signed map accounting')
    if head=='AUC and conditional intensity errors':
        body=body.replace('Main Fig. 1c instead pools','The spatial sensitivity analyses in Supplementary Figs. 1c and 16–17 pool')
    if head=='Region definitions':
        body=body.replace('The earlier main Fig. 1c spatial skill analysis and its Supplementary Figs. 1c and 16',
                          'The spatial sensitivity analyses in Supplementary Figs. 1c and 16–17')
    if head=='Tropical correlation comparisons':
        body=body.replace('Figure 5 reproduces the 81 coefficients underlying main Fig. 2.',
            'Figure 5 starts from the 81 coefficients underlying main Fig. 2 and displays 78 after excluding three unverified wind-stress values.').replace(
            'Wind stress at nominal label 9 uses a nearest available lead where an exact lead is absent.',
            'Wind stress at nominal label 9 used a nearest available lead where an exact lead was absent; all three such values are now marked unavailable, not interpreted as exact lead-9 forecasts.')
    if head=='Exact relationship between Supplementary Fig. 15 and main Fig. 4':
        body=(
            'Let d(g) = 0.7050185255 − AUC(g), where AUC(g) is the nine-lead mean of the local '
            '18-month, 11°-neighbourhood AUC, and let a(g) be cosine-latitude area weights normalized '
            'over all evaluable map cells. For region b, L(b) = sum a(g) max[d(g),0] and '
            'H(b) = sum a(g) max[−d(g),0]. The common denominator is '
            'N = sum_b [L(b) − H(b)] = 0.1065143704. The positive and negative percentages are '
            '100 L(b)/N and −100 H(b)/N. They sum to +120.790563% and −20.790563%, respectively, '
            'and hence 100% net. Historical SST-association weights w(b,k) sum to one within '
            'each region; component amounts are sum_b L(b)w(b,k) and −sum_b H(b)w(b,k). '
            'Local subcomponents replace, rather than duplicate, their parent allocation. '
            'The global reference is not an estimate of each grid cell\'s expected AUC. The map '
            'aggregate differs from main Fig. 1\'s globally pooled monthly spatial-AUC deficit '
            '(0.1090497372); these are distinct estimands, with different ranking samples and '
            'aggregation order. Exact closure verifies accounting, not causal attribution. '
            'Source tables provide regional and component crosswalks to main Fig. 4.')
    METHODS[i]=(head,body)
COVERAGE[2]=('S5','SST, eastward stress, precipitation correlations','Main-Fig. 2 coefficient archive; ERSST/ERA5',
    'SST: DJF; atmosphere: two-year extraction, 23 observed months',
    '20°S–20°N; 140°E–100°W','3 events; 78 of 81 coefficients displayed',
    'Wind label 9 excluded: exact lead unverified; full per-value model counts unavailable; other extraction limitations in Methods')
COVERAGE[7]=('S15','Local AUC differences and historical SST-association weights',
    'Same spatial AUC map and association weights as main Fig. 4',
    'April 2023–September 2024; weights:1985–2020',
    'Seven ±23.5° regions + Other seas; 11° same-basin pools',
    '27,508 valid map centres; nine leads; weights:432 months/region',
    'Undefined AUC excluded; net map denominator 0.1065143704; distinct from Fig. 1 global AUC gap')

# Final language and matched-data update. Numerical definitions remain explicit.
from pathlib import Path
import json
_root=Path(__file__).resolve().parent
CAPTIONS[2]['caption']+=' The 95% year-block intervals for the correlations are 0.31 to 0.78 for El Niño and −0.13 to 0.39 for La Niña.'
CAPTIONS[2]['caption']=CAPTIONS[2]['caption'].replace('The displayed 95% correlation intervals','The 95% correlation intervals')
CAPTIONS[9]['title']='Links between tropical and remote forecast errors depend on the El Niño event'
CAPTIONS[13]['title']='Testing regional SST estimates in years excluded from fitting'
CAPTIONS[14]['caption']=CAPTIONS[14]['caption'].replace('pairs enter b','pairs enter a')
CAPTIONS[17]['caption']+=' Individual-cell estimates use each grid cell separately; pooled estimates combine samples within an 11° × 11° same-basin neighbourhood.'
_matched=_root/'matched_s5/caption.json'
if _matched.exists():
    CAPTIONS[5]=json.loads(_matched.read_text())
    for i,(head,body) in enumerate(METHODS):
        if head=='Tropical correlation comparisons':
            METHODS[i]=(head,CAPTIONS[5]['caption']+' '+CAPTIONS[5]['statistics']+' Across eight leads, mean correlations change from 0.933 to 0.668 for SST, from 0.764 to 0.273 for zonal wind stress and from 0.810 to 0.612 for convection. Thus this matched comparison supports particularly strong wind-stress deterioration, but does not show greater deterioration in both atmospheric variables than in SST. The earlier mixed-protocol figure is retained in the preceding release, not used for this strict comparison.')
    COVERAGE[2]=('S5','SST, zonal wind stress, convection correlations','NASA GEOS-S2S members 1–4; ERSST and ERA5',
        'DJF 1997–98, 2015–16, 2023–24; climatology 1992–2020',
        '20°S–20°N, 140°E–100°W; 4,833 common ocean cells',
        '3 events × 3 variables × 8 leads = 72 correlations',
        'Exact starts and leads 1–8; lead 9 unavailable for all variables')
