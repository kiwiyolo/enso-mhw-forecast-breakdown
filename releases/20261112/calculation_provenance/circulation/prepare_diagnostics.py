"""Z200-derived geostrophic vectors and prespecified continuous transects."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'source_data'
G, OMEGA, RADIUS = 9.80665, 7.2921159e-5, 6371000.

def geostrophic(height, lat, lon):
    """Height [m] on a regular global lon grid -> geostrophic u/v [m s-1]."""
    z = np.asarray(height, dtype=float)
    phi, lam = np.deg2rad(lat), np.deg2rad(lon)
    assert np.allclose(np.diff(lam), np.diff(lam)[0])
    assert np.isclose(len(lam) * np.diff(lam)[0], 2 * np.pi)
    dzdl = (np.roll(z, -1, axis=-1) - np.roll(z, 1, axis=-1)) / (2 * np.diff(lam)[0])
    dzdp = np.gradient(z, phi, axis=-2, edge_order=2)
    f = 2 * OMEGA * np.sin(phi)
    f = np.where(np.abs(lat) >= 23.5, f, np.nan)
    u = -G * dzdp / (RADIUS * f[:, None])
    v = G * dzdl / (RADIUS * f[:, None] * np.cos(phi)[:, None])
    return u, v

def main():
    # Convert before differencing/addition: float32 complete heights (~12000 m)
    # otherwise lose millimetre precision when recovering small anomalies.
    d = xr.load_dataset(DATA / 'Figure3_Z200_fields.nc').astype('float64')
    era = xr.load_dataset(ROOT / 'cache/z200_ERA5_DJF.nc').astype('float64')
    clim = era.climatology.mean('month', skipna=False)
    canonical, obs = d.canonical, d.observed.sel(event='2023/24', drop=True)
    fc = d.forecast.sel(event='2023/24', drop=True).mean('lead', skipna=False)
    canonical_fc = d.forecast.sel(event=['1997/98','2015/16']).mean('lead',skipna=False).mean('event',skipna=False)
    err = fc - obs
    fields = xr.Dataset({'climatology':clim, 'canonical_anomaly':canonical,
        'observed_anomaly':obs, 'forecast_anomaly':fc, 'forecast_error':err,
        'canonical_height':clim+canonical, 'observed_height':clim+obs,
        'forecast_height_climatology_adjusted':clim+fc,
        'canonical_forecast_anomaly':canonical_fc,
        'canonical_forecast_height_climatology_adjusted':clim+canonical_fc,
        'canonical_forecast_error':canonical_fc-canonical})
    fields.canonical_forecast_height_climatology_adjusted.attrs.update(
        definition='ERA5 DJF climatology plus equal-event (1997/98,2015/16), equal-lead (1-8), equal-model forecast anomaly',
        units='m',comparison='Compare with canonical_height; not a forecast of the climatology')
    for key, z in [('canonical', canonical), ('observed',obs), ('forecast',fc), ('error',err)]:
        u,v = geostrophic(z.values, d.lat.values, d.lon.values)
        for component, values in [('u',u),('v',v)]:
            fields[f'{key}_{component}g'] = (('lat','lon'),values)
            fields[f'{key}_{component}g'].attrs.update(units='m s-1',
                definition='Z200-derived geostrophic approximation; equatorial band masked')
    fields.attrs.update(climatology='1992-2020 calendar-month means, equal DJF weighting; Feb 2018 excluded',
        wind='Spherical finite-difference geostrophic wind, not actual total winds or wave-activity flux',
        forecast_height='ERA5 climatology plus equal-model/equal-lead forecast anomaly; not raw model absolute Z200',
        smoothing='none',forecast_leads='1-8, exact available common leads',latitude_boundary=23.5)
    fields.to_netcdf(DATA/'Figure3_geostrophic_and_height_fields.nc')
    # Full longitude files are retained; main figure uses the fixed Pacific-Americas-Atlantic sector.
    rows=[]
    for latitude in [45,-50]:
        for lon in range(360):
            s=fields.sel(lat=latitude,lon=lon)
            rows.append(dict(latitude=latitude,longitude=lon,unwrapped_longitude=lon if lon>=120 else lon+360,
                climatology=float(s.climatology),canonical=float(s.canonical_height),
                observed=float(s.observed_height),forecast=float(s.forecast_height_climatology_adjusted),
                canonical_anomaly=float(s.canonical_anomaly),observed_anomaly=float(s.observed_anomaly),
                forecast_anomaly=float(s.forecast_anomaly),forecast_error=float(s.forecast_error),
                forecast_canonical=float(s.canonical_forecast_height_climatology_adjusted),
                forecast_canonical_anomaly=float(s.canonical_forecast_anomaly),
                forecast_canonical_error=float(s.canonical_forecast_error)))
    profiles=pd.DataFrame(rows).sort_values(['latitude','unwrapped_longitude'])
    profiles.to_csv(DATA/'Figure3cd_transects.csv',index=False)
    main_profiles=profiles[profiles.unwrapped_longitude<=380]
    summary=[]
    for lat,p in main_profiles.groupby('latitude'):
        summary.append(dict(latitude=int(lat),longitude_start=120,longitude_end=380,
            rms_height_error_m=float(np.sqrt(np.mean(p.forecast_error**2))),
            mean_height_error_m=float(p.forecast_error.mean()),
            forecast_observed_anomaly_r=float(p.forecast_anomaly.corr(p.observed_anomaly)),
            observed_canonical_anomaly_r=float(p.observed_anomaly.corr(p.canonical_anomaly)),
            canonical_forecast_observed_anomaly_r=float(p.forecast_canonical_anomaly.corr(p.canonical_anomaly)),
            canonical_rms_height_error_m=float(np.sqrt(np.mean(p.forecast_canonical_error**2))),
            canonical_mean_height_error_m=float(p.forecast_canonical_error.mean()),
            largest_positive_error_m=float(p.forecast_error.max()),
            positive_error_longitude=float(p.loc[p.forecast_error.idxmax(),'longitude']),
            largest_negative_error_m=float(p.forecast_error.min()),
            negative_error_longitude=float(p.loc[p.forecast_error.idxmin(),'longitude'])))
    pd.DataFrame(summary).to_csv(DATA/'Figure3cd_transect_statistics.csv',index=False)
    # Area-based concentration is evaluated in prespecified belts, not along a zero-area line.
    lat,lon=np.meshgrid(d.lat.values,d.lon.values,indexing='ij')
    w=np.cos(np.deg2rad(lat))
    sector=(lon>=120)|(lon<=20)
    mse_map=err.values**2
    mse_leads=(d.forecast_error.sel(event='2023/24')**2).mean('lead').values
    results=[]
    for halfwidth in [3,5,7]:
        for hemisphere,centres in [('NH',[45]),('SH',[-50]),('both',[45,-50])]:
            belt=sector & np.logical_or.reduce([np.abs(lat-c)<=halfwidth for c in centres])
            for label,ref in [('60S-60N',np.ones_like(belt)),('extratropical',np.abs(lat)>=23.5)]:
                area=float(w[belt].sum()/w[ref].sum())
                for metric,sq in [('squared_lead_mean_error',mse_map),('mean_lead_squared_error',mse_leads)]:
                    share=float((w[belt]*sq[belt]).sum()/(w[ref]*sq[ref]).sum())
                    results.append(dict(halfwidth_deg=halfwidth,hemisphere=hemisphere,reference=label,
                        metric=metric,area_fraction=area,error_fraction=share,concentration=share/area,
                        rms_inside_m=float(np.sqrt(np.average(sq[belt],weights=w[belt]))),
                        rms_outside_m=float(np.sqrt(np.average(sq[ref & ~belt],weights=w[ref & ~belt])))))
    pd.DataFrame(results).to_csv(DATA/'SI_prespecified_error_concentration.csv',index=False)
    checks={}
    lat=d.lat.values;lon=d.lon.values
    zero=geostrophic(np.zeros_like(err.values),lat,lon)
    checks['constant_height_zero_wind']=bool(all(np.nanmax(np.abs(a))==0 for a in zero))
    checks['equator_masked']=bool(np.isnan(fields.observed_ug.sel(lat=slice(-23,23))).all())
    checks['wind_linearity']=bool(np.allclose(fields.error_ug,fields.forecast_ug-fields.observed_ug,equal_nan=True,atol=1e-10)
        and np.allclose(fields.error_vg,fields.forecast_vg-fields.observed_vg,equal_nan=True,atol=1e-10))
    phi,lam=np.meshgrid(np.deg2rad(lat),np.deg2rad(lon),indexing='ij')
    analytic=100*np.cos(phi)*np.cos(lam)
    u,v=geostrophic(analytic,lat,lon)
    f=2*OMEGA*np.sin(phi)
    with np.errstate(divide='ignore',invalid='ignore'):
        ua=G*100*np.sin(phi)*np.cos(lam)/(f*RADIUS)
        va=-G*100*np.sin(lam)/(f*RADIUS)
    valid=(np.abs(lat)>=24)&(np.abs(lat)<60)
    checks['spherical_derivative_analytic_max_error_ms']=float(max(np.nanmax(np.abs(u[valid]-ua[valid])),np.nanmax(np.abs(v[valid]-va[valid]))))
    checks['analytic_derivatives_pass']=checks['spherical_derivative_analytic_max_error_ms']<.001
    checks['full_longitudes_per_section']=len(profiles)==720
    checks['native_1_degree_no_smoothing']=bool(np.all(np.diff(d.lat)==1) and np.all(np.diff(d.lon)==1))
    checks['common_climatology_recovery']=bool(np.allclose(fields.forecast_height_climatology_adjusted-fields.observed_height,err))
    (ROOT/'diagnostic_QA.json').write_text(json.dumps(checks,indent=2)+'\n')
    assert all(v for k,v in checks.items() if k!='spherical_derivative_analytic_max_error_ms')
    print(json.dumps({'checks':checks,'transects':summary,'primary_concentration':[r for r in results if r['halfwidth_deg']==5 and r['hemisphere']=='both']},indent=2))

if __name__=='__main__':main()
