"""Replace zero-anomaly climatology with a user-confirmed historical anomaly composite."""
from pathlib import Path
import hashlib
import itertools
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr
from calculate_anomaly_correlations import weighted_r

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data'
RAW=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/teleconnection_data/observations/ERA5')
sys.path.insert(0,str(ROOT/'dependencies'))
import k02_calculate_teleconnection as reader
START_YEARS=list(range(1992,2022))
EXCLUDED=[1997,2015]
HISTORY=[y for y in START_YEARS if y not in EXCLUDED]
ANCHORS=['forecast_2023','forecast_canonical']
TARGETS=['other_historical','observed_canonical','observed_2023']

def main():
    old=xr.load_dataset(DATA/'Figure3cd_anomaly_fields.nc')
    era=xr.load_dataset(ROOT/'cache/z200_ERA5_DJF.nc')
    lat,lon=old.lat.values,old.lon.values
    arrays=[];dates=[];audit=[]
    # Preserve original baseline and retain every eligible complete pre-2023 DJF.
    for year in range(1992,2023):
        p=RAW/f'era5_pressure_levels_{year}.nc'
        with xr.open_dataset(p) as ds:
            sub=ds.sel(valid_time=ds.valid_time.dt.month.isin([12,1,2]),
                       pressure_level=200.,latitude=lat,longitude=lon,drop=True)
            a=reader.preprocess_field_dataarray(sub,'z200',200.).transpose('valid_time','lat','lon').load()
            months=pd.to_datetime(a.valid_time.values).to_period('M').astype(str).tolist()
            assert months==[f'{year}-01',f'{year}-02',f'{year}-12'],months
            assert a.attrs['units']=='m' and np.isfinite(a.values).all(),str(p)
            arrays.extend(a.values);dates.extend(months)
            audit.append(dict(year=year,path=str(p.resolve()),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
                selected_height_sha256=hashlib.sha256(a.values.tobytes()).hexdigest(),
                months=';'.join(months),units='m'))
        print('Read',year,flush=True)
    heights=xr.DataArray(np.stack(arrays),dims=['target','lat','lon'],
                        coords={'target':dates,'lat':lat,'lon':lon})
    heights.to_dataset(name='height').to_netcdf(DATA/'Figure3cd_historical_monthly_Z200.nc',
        encoding={'height':{'zlib':True,'complevel':3}})
    seasons=[];selection=[]
    for year in START_YEARS:
        targets=[f'{year}-12',f'{year+1}-01',f'{year+1}-02']
        # Float32 monthly arithmetic matches original ERA5 preprocessing; average in float64.
        monthly=np.stack([(heights.sel(target=t).values-era.climatology.sel(month=m).values)
                          for t,m in zip(targets,[12,1,2])])
        assert np.isfinite(monthly).all()
        seasons.append(monthly.astype(float).mean(axis=0))
        selection.append(dict(start_year=year,winter=f'{year}/{str(year+1)[-2:]}',
            targets=';'.join(targets),included=year not in EXCLUDED,
            reason='excluded canonical event' if year in EXCLUDED else 'eligible complete pre-2023 winter'))
    seasonal=xr.DataArray(np.stack(seasons),dims=['start_year','lat','lon'],
        coords={'start_year':START_YEARS,'lat':lat,'lon':lon})
    reference=seasonal.sel(start_year=HISTORY).mean('start_year',skipna=False)
    out=xr.Dataset({'seasonal_anomaly':seasonal,'other_historical':reference})
    out.attrs.update(definition='Equal average of 28 DJF observed anomaly fields before correlation',
        first_winter='1992/93',last_winter='2021/22',excluded_canonical='1997/98;2015/16',
        baseline=era.attrs['baseline'],cutoff='Every verifying month strictly before 2023-01',
        not_neutral='All other winters, including other ENSO states; not a neutral-ENSO composite',
        temporal_weight='Equal three DJF months, then equal winters',
        february_2018='Excluded from climatology as in v12; valid ERA5 verification retained in winter 2017/18')
    out.to_netcdf(DATA/'Figure3cd_other_historical_fields.nc',
        encoding={k:{'zlib':True,'complevel':3} for k in out.data_vars})
    fields=old.drop_vars('climatology')
    fields['other_historical']=reference
    fields.to_netcdf(DATA/'Figure3cd_historical_comparison_fields.nc',
        encoding={k:{'zlib':True,'complevel':3} for k in fields.data_vars})
    masks=xr.load_dataset(DATA/'Figure3cd_band_masks.nc').mask.astype(bool)
    latgrid=np.broadcast_to(lat[:,None],reference.shape)
    rows=[];amplitudes=[];sensitivity=[]
    for band in ['NH','SH']:
        mask=masks.sel(band=band).values;w=np.cos(np.deg2rad(latgrid[mask]))
        for anchor,target in itertools.product(ANCHORS,TARGETS):
            r=weighted_r(fields[anchor].values[mask],fields[target].values[mask],w)
            rows.append(dict(band=band,anchor=anchor,target=target,r=r,n_grid_cells=int(mask.sum()),
                reference_winters=28 if target=='other_historical' else (2 if target=='observed_canonical' else 1),
                status='defined',field_type='DJF_seasonal_mean_Z200_anomaly'))
        x=reference.values[mask];mu=np.average(x,weights=w)
        amplitudes.append(dict(band=band,mean_m=mu,rms_m=np.sqrt(np.average(x*x,weights=w)),
            spatial_std_m=np.sqrt(np.average((x-mu)**2,weights=w))))
        for anchor in ANCHORS:
            f=fields[anchor].values[mask]
            for y in HISTORY:
                rv=weighted_r(f,seasonal.sel(start_year=y).values[mask],w)
                sensitivity.append(dict(band=band,anchor=anchor,start_year=y,r=rv,role='supplementary individual winter comparison, not main statistic'))
    result=pd.DataFrame(rows)
    assert len(result)==12 and np.isfinite(result.r).all()
    old_table=pd.read_csv(DATA/'Figure3cd_anomaly_correlations.csv').query("target != 'climatology'")
    merged=result.merge(old_table,on=['band','anchor','target'],suffixes=('_new','_old'))
    assert len(merged)==8 and np.max(np.abs(merged.r_new-merged.r_old))<1e-12
    result.to_csv(DATA/'Figure3cd_historical_anomaly_correlations.csv',index=False)
    pd.DataFrame(amplitudes).to_csv(DATA/'Figure3cd_historical_reference_amplitude.csv',index=False)
    pd.DataFrame(audit).to_csv(DATA/'historical_ERA5_source_audit.csv',index=False)
    pd.DataFrame(selection).to_csv(DATA/'historical_winter_selection.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(DATA/'SI_individual_historical_winter_correlations.csv',index=False)
    report={'version':'20261104_v18','parent':'Figure3_anomaly_correlations_20261103_v12',
        'user_confirmed_order':'Composite other historical DJF anomalies first, then spatial correlation',
        'historical_start_years':HISTORY,'n_historical_winters':len(HISTORY),'exclusions':EXCLUDED,
        'start_year_basis':'1992 study baseline start; not all earlier available ERA5 years',
        'cutoff':'All DJF months before 2023-01; latest 2021/22',
        'reference_baseline':'Fixed v12 1992-2020 calendar-month means, Feb 2018 excluded',
        'spatial_domain':'30-60N and 40-60S, 120E eastward to20E, land and ocean',
        'aggregation':'Monthly anomalies -> equal DJF mean -> equal 28-winter mean -> cosine-latitude weighted spatial Pearson r',
        'uncertainty':'Deterministic composite correlations; no confidence interval or causal test',
        'reference_amplitude':amplitudes,'all_values':rows,
        'limitations':'Other historical is not climatology or a neutral-ENSO reference. It mixes regimes and overlaps the anomaly baseline; exclusion of canonical winters can influence the sign. Correlation alone does not measure amplitude or physical distance.'}
    (ROOT/'historical_anomaly_statistics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.to_string(index=False),flush=True)
if __name__=='__main__':main()
