"""v19: replace only the first c,d reference with other paper-listed El Ninos.

Event selection precedes calculation and is independent of correlation values.
Reuse the audited v18 monthly/seasonal archive; do not change its baseline.
"""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
import xarray as xr
from calculate_anomaly_correlations import weighted_r

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'source_data'
EVENT_SOURCE = Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures/Design_20261027_v5/source_data/Figure1_source_events.csv')
ANCHORS = ['forecast_2023', 'forecast_canonical']
TARGETS = ['other_el_nino', 'observed_canonical', 'observed_2023']

def main():
    events = pd.read_csv(DATA / 'Figure1_source_events.csv')
    assert hashlib.sha256((DATA/'Figure1_source_events.csv').read_bytes()).digest() == hashlib.sha256(EVENT_SOURCE.read_bytes()).digest()
    history = xr.load_dataset(DATA/'Figure3cd_other_historical_fields.nc')
    candidates = list(map(int, history.start_year.values))
    assert candidates == list(range(1992, 2022))
    event_years = events.Event.str.split('/').str[0].astype(int).tolist()
    years = [y for y in event_years if y in candidates and y not in [1997, 2015, 2023]]
    assert years == [1994, 2002, 2004, 2006, 2009, 2018]
    ledger = []
    for y in sorted(set(candidates + event_years)):
        if y in [1997, 2015]:
            reason = 'excluded: canonical event'
        elif y == 2023:
            reason = 'excluded: focal event'
        elif y not in candidates:
            reason = 'excluded: outside unchanged v18 historical window'
        elif y not in event_years:
            reason = 'excluded: not in paper Figure1 El Nino event list'
        else:
            reason = 'included: other paper-listed El Nino within historical window'
        ledger.append(dict(start_year=y, winter=f'{y}/{str(y+1)[-2:]}',
            included=y in years, paper_listed_el_nino=y in event_years,
            within_v18_window=y in candidates, reason=reason))
    seasonal = history.seasonal_anomaly.sel(start_year=years)
    reference = seasonal.mean('start_year', skipna=False)
    selected = xr.Dataset({'seasonal_anomaly': seasonal, 'other_el_nino': reference})
    selected.attrs.update(definition='Equal observed DJF anomaly composite of six other paper-listed El Nino events',
        event_source=str(EVENT_SOURCE), selection='Paper Figure1 list intersected with v18 1992/93-2021/22 window; exclude canonical and focal events',
        baseline=history.attrs['baseline'], aggregation='Equal DJF months then equal events, before spatial correlation',
        no_recentring='Fixed v12 monthly climatology; do not refit on selected subset')
    selected.to_netcdf(DATA/'Figure3cd_other_el_nino_fields.nc',
        encoding={k: {'zlib': True, 'complevel': 3} for k in selected.data_vars})
    original = xr.load_dataset(DATA/'Figure3cd_anomaly_fields.nc')
    fields = original.drop_vars('climatology')
    fields['other_el_nino'] = reference
    fields.attrs.update(active_reference='Other El Nino composite; not climatology',
        n_other_el_nino_events=len(years), version='20261104_v19')
    fields.to_netcdf(DATA/'Figure3cd_el_nino_comparison_fields.nc',
        encoding={k: {'zlib': True, 'complevel': 3} for k in fields.data_vars})
    masks = xr.load_dataset(DATA/'Figure3cd_band_masks.nc').mask.astype(bool)
    latgrid = np.broadcast_to(fields.lat.values[:,None], reference.shape)
    rows, amplitudes, individual = [], [], []
    for band in ['NH', 'SH']:
        mask = masks.sel(band=band).values
        w = np.cos(np.deg2rad(latgrid[mask]))
        for anchor, target in itertools.product(ANCHORS, TARGETS):
            r = weighted_r(fields[anchor].values[mask], fields[target].values[mask], w)
            rows.append(dict(band=band, anchor=anchor, target=target, r=r,
                n_grid_cells=int(mask.sum()), reference_winters=6 if target=='other_el_nino' else (2 if target=='observed_canonical' else 1),
                status='defined', field_type='DJF_seasonal_mean_Z200_anomaly'))
        x = reference.values[mask]
        mean = np.average(x, weights=w)
        amplitudes.append(dict(band=band, mean_m=mean, rms_m=np.sqrt(np.average(x*x, weights=w)),
            spatial_std_m=np.sqrt(np.average((x-mean)**2, weights=w))))
        for anchor, y in itertools.product(ANCHORS, years):
            individual.append(dict(band=band, anchor=anchor, start_year=y,
                r=weighted_r(fields[anchor].values[mask], seasonal.sel(start_year=y).values[mask], w),
                role='individual-event sensitivity, not the composite-first main statistic'))
    table = pd.DataFrame(rows)
    assert len(table)==12 and np.isfinite(table.r).all() and table.r.between(-1,1).all()
    old = pd.read_csv(DATA/'Figure3cd_historical_anomaly_correlations.csv')
    paired = table.merge(old, on=['band','anchor','target'], suffixes=('_new','_old'))
    assert len(paired)==8 and np.allclose(paired.r_new, paired.r_old, rtol=0, atol=1e-12)
    table.to_csv(DATA/'Figure3cd_el_nino_anomaly_correlations.csv', index=False)
    pd.DataFrame(ledger).to_csv(DATA/'other_el_nino_event_selection.csv', index=False)
    pd.DataFrame(amplitudes).to_csv(DATA/'Figure3cd_el_nino_reference_amplitude.csv', index=False)
    pd.DataFrame(individual).to_csv(DATA/'SI_individual_other_el_nino_correlations.csv', index=False)
    report = dict(version='20261104_v19', parent='Figure3_other_historical_anomalies_20261104_v18',
        request='Only first c,d comparison changes from all other historical winters to other historical El Nino events',
        event_source=str(EVENT_SOURCE), event_source_sha256=hashlib.sha256(EVENT_SOURCE.read_bytes()).hexdigest(),
        start_years=years, n_events=len(years), inherited_history_window='1992/93-2021/22',
        outside_window='1991/92 is paper-listed but outside unchanged v18 historical window',
        classification='Use existing paper event list, not a newly inferred event classification or outcome-directed subset',
        baseline='Unchanged 1992-2020 monthly ERA5 climatology, Feb2018 excluded from baseline',
        aggregation='DJF anomaly mean per event, equal six-event composite, then cosine-latitude-weighted centred spatial Pearson r',
        scope='NH30-60N and SH40-60S, 120E eastward to20E; land and ocean, no smoothing',
        uncertainty='Deterministic composite correlations; no confidence interval or grid-cell significance',
        reference_amplitude=amplitudes, values=rows,
        interpretation='Cross-reference pattern resemblance, not amplitude, physical distance, causal attribution or proof of regression towards climatology')
    (ROOT/'el_nino_anomaly_statistics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(table.to_string(index=False))

if __name__ == '__main__':
    main()
