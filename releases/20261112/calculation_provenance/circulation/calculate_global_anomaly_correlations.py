"""v20: area-weighted spatial correlations across the full available global belt.

All v19 anomaly fields and event definitions are retained. Merge c,d by
recomputing over every native grid cell, never by averaging hemispheric r.
"""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
import xarray as xr
from calculate_anomaly_correlations import weighted_r

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data'
PARENT=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures/Figure3_other_el_nino_anomalies_20261104_v19')
ANCHORS=['forecast_2023','forecast_canonical']
TARGETS=['other_el_nino','observed_canonical','observed_2023']

def main():
    source=DATA/'Figure3cd_el_nino_comparison_fields.nc'
    assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256((PARENT/'source_data'/source.name).read_bytes()).digest()
    fields=xr.load_dataset(source)
    assert fields.lat.values.tolist()==list(range(-60,61))
    assert fields.lon.values.tolist()==list(range(360))
    assert all(np.isfinite(fields[k]).all() for k in ANCHORS+TARGETS)
    lat=np.broadcast_to(fields.lat.values[:,None],(121,360))
    weights=np.cos(np.deg2rad(lat))
    mask=np.ones((121,360),dtype=np.int8)
    domain=xr.Dataset({'mask':(('lat','lon'),mask),'area_weight':(('lat','lon'),weights)},
        coords={'lat':fields.lat,'lon':fields.lon})
    domain.attrs.update(definition='All available native grid nodes: 60S-60N, all longitudes, land and ocean',
        polar_coverage='No data outside60S-60N in inherited fields; no interpolation or extrapolation',
        no_cyclic_duplicate='Longitude0-359 exactly once; no360 node')
    domain.to_netcdf(DATA/'Figure3c_global_domain.nc')
    rows=[];amplitude=[]
    for anchor,target in itertools.product(ANCHORS,TARGETS):
        r=weighted_r(fields[anchor].values.ravel(),fields[target].values.ravel(),weights.ravel())
        rows.append(dict(domain='Global_60S_60N',anchor=anchor,target=target,r=r,n_grid_cells=43560,
            target_events=6 if target=='other_el_nino' else (2 if target=='observed_canonical' else 1),
            field_type='DJF_mean_Z200_anomaly',weight='cos(latitude)',status='defined'))
    for k in dict.fromkeys(ANCHORS+TARGETS):
        x=fields[k].values;mu=np.average(x,weights=weights)
        amplitude.append(dict(state=k,mean_m=mu,rms_m=np.sqrt(np.average(x*x,weights=weights)),
            spatial_std_m=np.sqrt(np.average((x-mu)**2,weights=weights))))
    table=pd.DataFrame(rows)
    assert len(table)==6 and table.r.between(-1,1).all()
    table.to_csv(DATA/'Figure3c_global_anomaly_correlations.csv',index=False)
    pd.DataFrame(amplitude).to_csv(DATA/'Figure3c_global_anomaly_amplitudes.csv',index=False)
    seasons=xr.load_dataset(DATA/'Figure3cd_other_el_nino_fields.nc').seasonal_anomaly
    sensitivity=[]
    for anchor,year in itertools.product(ANCHORS,seasons.start_year.values):
        sensitivity.append(dict(anchor=anchor,start_year=int(year),
            r=weighted_r(fields[anchor].values.ravel(),seasons.sel(start_year=year).values.ravel(),weights.ravel()),
            role='individual-event sensitivity only; main reference is composite-first'))
    pd.DataFrame(sensitivity).to_csv(DATA/'SI_global_individual_el_nino_correlations.csv',index=False)
    report=dict(version='20261104_v20',parent=str(PARENT),
        request='Merge hemispheric c,d into one all-longitude global-belt anomaly-correlation panel c',
        latitude=[-60,60],longitude_native=[0,359],n_grid_cells=43560,
        coverage='All longitudes over60S-60N; not pole-to-pole coverage',
        field='Unchanged v19 DJF Z200 anomalies, not full heights',
        metric='Centred cosine-latitude-weighted spatial Pearson correlation over the whole domain',
        order='Fixed monthly climatology -> equal DJF months -> equal models/leads/events -> whole-domain spatial correlation',
        other_el_nino_start_years=[1994,2002,2004,2006,2009,2018],
        canonical_events=['1997/98','2015/16'],focal_event='2023/24',
        uncertainty='Deterministic composite-pattern correlations; no inferential intervals or grid-cell p values',
        preservation='a,b and all inherited input arrays unchanged; c,d source tables archived',
        limits='Not an average of regional r; includes tropical and extratropical nodes. Pattern correlation does not measure amplitude, RMSE, climatological proximity or causal transmission.',
        results=rows,amplitudes=amplitude)
    (ROOT/'global_anomaly_statistics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(table.to_string(index=False))

if __name__=='__main__':main()
