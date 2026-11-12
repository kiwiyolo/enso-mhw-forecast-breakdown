"""Descriptive area-weighted spatial correlations of the five v8 Z200 states."""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data'
STATES={
    'climatology':'climatology',
    'observed_canonical':'canonical_height',
    'forecast_canonical':'canonical_forecast_height_climatology_adjusted',
    'observed_2023':'observed_height',
    'forecast_2023':'forecast_height_climatology_adjusted',
}
ANCHORS=['forecast_2023','forecast_canonical']
TARGETS=['climatology','observed_canonical','observed_2023']
BANDS={'NH':(30,60),'SH':(-60,-40)}

def spatial_r(x,y,w):
    x,y,w=np.asarray(x,float),np.asarray(y,float),np.asarray(w,float)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(w).all()
    w=w/w.sum();a=x-np.sum(w*x);b=y-np.sum(w*y)
    denominator=np.sqrt(np.sum(w*a*a)*np.sum(w*b*b))
    if denominator<=0:return np.nan
    return float(np.sum(w*a*b)/denominator)

def main():
    d=xr.load_dataset(DATA/'Figure3_geostrophic_and_height_fields.nc')
    lat,lon=np.meshgrid(d.lat,d.lon,indexing='ij')
    sector=(lon>=120)|(lon<=20)
    weights=np.cos(np.deg2rad(lat))
    masks=[];rows=[];all_pairs=[]
    for band,(south,north) in BANDS.items():
        mask=sector & (lat>=south)&(lat<=north)
        masks.append(mask)
        values={k:d[v].values[mask] for k,v in STATES.items()}
        common=np.logical_and.reduce([np.isfinite(x) for x in values.values()])
        assert common.all(),'Incomplete common spatial support; do not drop cells silently'
        for anchor,target in itertools.combinations(STATES,2):
            all_pairs.append(dict(band=band,state_1=anchor,state_2=target,
                r=spatial_r(values[anchor],values[target],weights[mask]),n_grid_cells=int(mask.sum())))
        for anchor,target in itertools.product(ANCHORS,TARGETS):
            rows.append(dict(band=band,latitude_min=south,latitude_max=north,
                longitude_start=120,longitude_end_unwrapped=380,anchor=anchor,target=target,
                r=spatial_r(values[anchor],values[target],weights[mask]),n_grid_cells=int(mask.sum()),
                field_type='complete_climatology_adjusted_Z200',weight='cos(latitude)'))
    result=pd.DataFrame(rows)
    result.to_csv(DATA/'Figure3cd_band_correlations.csv',index=False)
    pd.DataFrame(all_pairs).to_csv(DATA/'SI_five_state_pairwise_correlations.csv',index=False)
    xr.Dataset({'mask':(('band','lat','lon'),np.array(masks,dtype='int8'))},
        coords={'band':list(BANDS),'lat':d.lat,'lon':d.lon},
        attrs={'domain':'Land and ocean; native 1-degree grid centres; inclusive latitude bounds; 120E eastward to 20E',
               'meaning':'2D latitude-band spatial correlations; not a zonal mean or a single-latitude section'}).to_netcdf(DATA/'Figure3cd_band_masks.nc')
    metadata={'definition':'Cosine-latitude weighted Pearson correlation over all grid cells within each two-dimensional domain',
        'anchors':ANCHORS,'targets':TARGETS,'states':STATES,'bands':BANDS,'longitude_unwrapped':[120,380],
        'aggregation':'Correlate the composed full-height fields; not the mean of correlations over leads, months or events',
        'baseline':'Inherited 1992-2020 calendar-month climatology; February 2018 excluded; equal DJF month weights',
        'forecast':'Equal-model, equal-lead (1-8) anomaly plus ERA5 climatology; canonical equal two-event composite',
        'independent_replicates':'No independent replicates assigned to deterministic bars; spatial grid cells are not independent events',
        'uncertainty':'No confidence intervals or significance tests; bars are descriptive point estimates',
        'interpretation':'Full-height pattern similarity can be dominated by shared climatological meridional gradients; not anomaly forecast skill',
        'original_map_dotted_lines':'45N/50S retained for exact v8 map preservation; contextual central latitudes only, not the full bar domains',
        'all_values':rows}
    (ROOT/'band_statistics.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(result[['band','anchor','target','r','n_grid_cells']].to_string(index=False))

if __name__=='__main__':main()
