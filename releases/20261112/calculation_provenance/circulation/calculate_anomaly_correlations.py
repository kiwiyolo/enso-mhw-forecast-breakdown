"""v12: v10 DJF spatial correlations after removing the matched climatology.

No result-directed selection or added smoothing. Climatology has zero anomaly
variance, so its correlations are undefined (NaN), never zero or one.
"""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'source_data'
ANCHORS = ['forecast_2023', 'forecast_canonical']
TARGETS = ['climatology', 'observed_canonical', 'observed_2023']
SOURCE = {'observed_canonical': 'canonical_anomaly',
          'forecast_canonical': 'canonical_forecast_anomaly',
          'observed_2023': 'observed_anomaly', 'forecast_2023': 'forecast_anomaly'}
BANDS = {'NH': (30, 60), 'SH': (-60, -40)}

def weighted_r(x, y, w):
    x, y, w = (np.asarray(a, dtype=float) for a in (x, y, w))
    if not (x.shape == y.shape == w.shape):
        raise ValueError('Shape mismatch')
    if x.size < 2 or not all(np.isfinite(a).all() for a in (x, y, w)):
        raise ValueError('Incomplete or insufficient support; no silent dropping')
    if np.any(w <= 0):
        raise ValueError('Weights must be positive')
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    w = w / w.sum()
    a, b = x - np.sum(w*x), y - np.sum(w*y)
    return float(np.sum(w*a*b) / np.sqrt(np.sum(w*a*a)*np.sum(w*b*b)))

def main():
    g = xr.load_dataset(DATA / 'Figure3_geostrophic_and_height_fields.nc')
    masks = xr.load_dataset(DATA / 'Figure3cd_band_masks.nc').mask.astype(bool)
    fields = xr.Dataset({name: g[source] for name, source in SOURCE.items()})
    fields['climatology'] = g.climatology - g.climatology
    fields.attrs.update(units='m', baseline=g.attrs['climatology'],
        statistic='Cosine-latitude weighted centred Pearson spatial correlation of DJF seasonal-mean anomalies',
        aggregation='Same v10 DJF mean, equal models and leads 1-8; canonical equal two-event composite',
        undefined='Climatology minus itself is exactly zero; no anomaly correlation exists',
        source='Figure3_geostrophic_and_height_fields.nc; preserved v10 anomaly fields, no smoothing')
    fields.to_netcdf(DATA / 'Figure3cd_anomaly_fields.nc',
        encoding={v: {'zlib': True, 'complevel': 3} for v in fields.data_vars})
    lon, lat = np.meshgrid(g.lon, g.lat)
    weights = np.cos(np.deg2rad(lat))
    rows, amplitudes, verification = [], [], []
    for band, (lo, hi) in BANDS.items():
        mask = masks.sel(band=band).values
        expected = ((lon >= 120) | (lon <= 20)) & (lat >= lo) & (lat <= hi)
        assert np.array_equal(mask, expected)
        w = weights[mask]; w = w / w.sum()
        values = {name: da.values[mask] for name, da in fields.data_vars.items()}
        assert all(np.isfinite(v).all() for v in values.values())
        for anchor, target in itertools.product(ANCHORS, TARGETS):
            r = weighted_r(values[anchor], values[target], w)
            rows.append(dict(band=band, latitude_min=lo, latitude_max=hi,
                longitude_start=120, longitude_end_unwrapped=380,
                anchor=anchor, target=target, r=r, n_grid_cells=int(mask.sum()),
                status='undefined_zero_climatology_anomaly' if target == 'climatology' else 'defined',
                field_type='DJF_seasonal_mean_Z200_anomaly', weight='cos(latitude)',
                temporal_aggregation='DJF mean before spatial correlation'))
        for name, x in values.items():
            mean = float(np.sum(w*x))
            amplitudes.append(dict(band=band, state=name, n_grid_cells=int(mask.sum()),
                mean_anomaly_m=mean, rms_anomaly_m=float(np.sqrt(np.sum(w*x*x))),
                spatial_std_anomaly_m=float(np.sqrt(np.sum(w*(x-mean)**2)))))
        for event in ['2023', 'canonical']:
            f, o = values['forecast_'+event], values['observed_'+event]
            frms, orms = np.sqrt(np.sum(w*f*f)), np.sqrt(np.sum(w*o*o))
            fstd = np.sqrt(np.sum(w*(f-np.sum(w*f))**2))
            ostd = np.sqrt(np.sum(w*(o-np.sum(w*o))**2))
            mse = np.sum(w*(f-o)**2)
            verification.append(dict(band=band, event=event,
                anomaly_r=weighted_r(f, o, w), forecast_rms_anomaly_m=float(frms),
                observed_rms_anomaly_m=float(orms), forecast_observed_rms_ratio=float(frms/orms),
                forecast_observed_spatial_std_ratio=float(fstd/ostd),
                forecast_observed_rmse_m=float(np.sqrt(mse)),
                climatology_observed_rmse_m=float(orms),
                mse_skill_vs_climatology=float(1-mse/(orms**2)),
                note='Descriptive seasonal composite; not event-held-out validation or a proof of model drift'))
    table = pd.DataFrame(rows)
    table.to_csv(DATA/'Figure3cd_anomaly_correlations.csv', index=False, na_rep='')
    pd.DataFrame(amplitudes).to_csv(DATA/'SI_anomaly_amplitude.csv', index=False)
    pd.DataFrame(verification).to_csv(DATA/'SI_climatology_distance_and_verification.csv', index=False)
    clean = [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
              for k, v in row.items()} for row in rows]
    metadata = dict(version='20261103_v12', parent='20261103_v10',
        metric='Centred cosine-latitude weighted Pearson correlation between DJF Z200 anomaly maps',
        time='DJF seasonal average before correlation, as explicitly requested v10; not v11 joint month-grid statistic',
        baselines='Observed monthly ERA5 climatology and model/lead/month-specific forecast climatologies, 1992-2020; February 2018 excluded',
        adjusted_height_identity='Subtract common ERA5 DJF climatology from each v10 adjusted full-height field to recover archived anomalies',
        anchors=ANCHORS, targets=TARGETS, bands=BANDS, longitude_unwrapped=[120, 380],
        climatology_handling='Exactly zero anomaly; four undefined pairs retained as N/A, never plotted as zero bars',
        defined_pairs=8, undefined_pairs=4, grid_support={'NH': 8091, 'SH': 5481},
        evidence='Descriptive comparison of seasonal composite patterns; no independent-grid p-values or confidence intervals',
        interpretation='Anomaly correlation measures spatial agreement, not amplitude or a tendency toward climatology. Auxiliary RMS and RMSE quantify distances without redefining correlation.',
        original_other_panels='a,b identical to user-specified v10', values=clean)
    (ROOT/'anomaly_statistics.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    print(table[['band', 'anchor', 'target', 'r', 'status']].to_string(index=False))
    print(pd.DataFrame(verification).to_string(index=False))

if __name__ == '__main__':
    main()
