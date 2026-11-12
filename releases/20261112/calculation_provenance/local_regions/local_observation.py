"""Seven-region ORAS5 mixed-layer diagnostics; unresolved terms stay residual."""
from pathlib import Path
import numpy as np,pandas as pd,xarray as xr,json
from concurrent.futures import ProcessPoolExecutor
from regions import masks,ORDER
from recompute_circulation import ROOT,CONFIG
DATA_ROOT=Path(CONFIG['roots']['data'])
ORAS=DATA_ROOT/'processed/ocean/ecmwf/oras5/state'
OUT=ROOT/'source_data';CACHE=ROOT/'cache/oras_regions';CACHE.mkdir(exist_ok=True)
VARS={'votemper':'ocean-variables','vozocrte':'ocean-variables','vomecrtn':'ocean-variables','somxl030':'ocean-variables','sohefldo':'surface-variables'}
LAT=np.arange(-59.5,60.,1);LON=np.arange(360.)
MASK=masks(LAT,LON).values
WEIGHTS=MASK*np.cos(np.deg2rad(LAT))[None,:,None]
def regmean(a):
    ok=np.isfinite(a)
    return np.sum(np.where(ok,a,0)[None]*WEIGHTS,axis=(1,2))/np.sum(ok[None]*WEIGHTS,axis=(1,2))
def one_month(month):
    out=CACHE/f'{month}.npz'
    if out.exists():return str(out)
    fields={};sources=[]
    for var,sub in VARS.items():
        paths=sorted((ORAS/sub/var).glob(f'*_{month}_*.nc'))
        if len(paths)!=1:raise ValueError((var,month,len(paths)))
        sources.append(str(paths[0]))
        with xr.open_dataset(paths[0]) as d:
            da=d[var].isel(time=0).sel(lat=slice(-60,60))
            # All depth levels are read, so deep seasonal mixed layers are not clipped.
            fields[var]=da.load()
    T=fields['votemper'];z=T.depth.values;h=fields['somxl030'].values
    interfaces=np.r_[0,(z[:-1]+z[1:])/2,z[-1]+(z[-1]-z[-2])/2]
    dz=np.maximum(0,np.minimum(h[None],interfaces[1:,None,None])-interfaces[:-1,None,None])
    valid=np.isfinite(T.values)&np.isfinite(fields['vozocrte'].values)&np.isfinite(fields['vomecrtn'].values)
    dz=np.where(valid,dz,0);thick=dz.sum(axis=0)
    def avg(v):return np.divide(np.nansum(v*dz,axis=0),thick,out=np.full_like(thick,np.nan),where=(thick>0)&(h>0))
    tm=avg(T.values);u=avg(fields['vozocrte'].values);v=avg(fields['vomecrtn'].values)
    # Spherical-grid centred gradients; coast-adjacent missing gradients excluded.
    dy=6371000*np.deg2rad(1.);dx=dy*np.cos(np.deg2rad(LAT))[:,None]
    gradx=(np.roll(tm,-1,axis=1)-np.roll(tm,1,axis=1))/(2*dx)
    grady=np.gradient(tm,dy,axis=0)
    adv=-u*gradx-v*grady
    q=fields['sohefldo'].values
    heating=np.divide(q,1025*3990*h,out=np.full_like(q,np.nan),where=h>0)
    # Both terms use the same cells. No artificial minimum MLD truncation.
    common=np.isfinite(adv)&np.isfinite(heating)&np.isfinite(tm)
    result={'sst':regmean(T.isel(depth=0).values),'mlt':regmean(np.where(common,tm,np.nan)),
        'qnet':regmean(q),'mld':regmean(h),'surface_heating':regmean(np.where(common,heating,np.nan))*86400*30,
        'horizontal_advection':regmean(np.where(common,adv,np.nan))*86400*30,
        'valid_area_fraction':np.sum(common[None]*WEIGHTS,axis=(1,2))/WEIGHTS.sum(axis=(1,2))}
    np.savez_compressed(out,**result,sources=np.array(sources))
    return str(out)
def main():
    months=pd.period_range('1984-12','2024-12',freq='M')
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,p in enumerate(ex.map(one_month,months.strftime('%Y%m'))):
            if i%36==0:print('ORAS5',months[i],flush=True)
    arrays={};manifest=[]
    for m in months:
        with np.load(CACHE/f'{m.strftime("%Y%m")}.npz') as d:
            for k in d.files:
                if k!='sources':arrays.setdefault(k,[]).append(d[k])
            manifest.extend(d['sources'].tolist())
    ds=xr.Dataset({k:(('time','region'),np.stack(v)) for k,v in arrays.items()},coords={'time':months.to_timestamp(),'region':ORDER})
    secs=months.to_timestamp().view('int64')/1e9
    ds['temperature_tendency']=(('time','region'),np.gradient(ds.mlt.values,secs,axis=0)*86400*30)
    ds['residual']=ds.temperature_tendency-ds.surface_heating-ds.horizontal_advection
    for v in ['surface_heating','horizontal_advection','temperature_tendency','residual']:ds[v].attrs['units']='degree_C per 30 days'
    ds.attrs.update(source='ORAS5 monthly reanalysis; pre-existing 1-degree remapped products',
       interpretation='reanalysis process estimates, not forecasts; monthly mean advection omits submonthly covariance',
       residual='entrainment, vertical processes, unrepresented advection, remapping and data-assimilation increments; not named mixing',
       mld='native ORAS5 density threshold 0.03; no lower clipping',heat_flux='net downward positive; rho=1025 kg/m3; cp=3990 J/kg/K')
    ds.to_netcdf(OUT/'Figure4_ORAS5_regional_monthly.nc')
    (OUT/'ORAS5_source_paths.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DONE ORAS5',len(months),flush=True)
if __name__=='__main__':main()
