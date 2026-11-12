"""Observed regional SST associations, separated from forecast-loss attribution."""
from pathlib import Path
import math,json,itertools
import numpy as np,pandas as pd,xarray as xr
from regions import masks,ORDER
from recompute_circulation import ROOT,PAPER,CONFIG
OUT=ROOT/'source_data';DR=Path(CONFIG['roots']['data'])
GROUPS=[[0],[1,2,3,4],[5,6,7,8]]
def r2fit(x,y):
    a=np.column_stack([np.ones(len(y)),x]);coef=np.linalg.lstsq(a,y,rcond=None)[0]
    return float(1-np.sum((y-a@coef)**2)/np.sum((y-y.mean())**2)),coef
def attribution(x,y):
    scores={0:0.}
    for bits in range(1,8):
        ids=[c for g in range(3) if bits&(1<<g) for c in GROUPS[g]]
        scores[bits]=r2fit(x[:,ids],y)[0]
    phi=[]
    for g in range(3):
        value=0
        for s in range(8):
            if s&(1<<g):continue
            k=s.bit_count();w=math.factorial(k)*math.factorial(2-k)/6
            value+=w*(scores[s|(1<<g)]-scores[s])
        phi.append(value)
    return np.r_[phi,1-scores[7]]
def main():
    src=DR/'processed/ocean/noaa/ersst/legacy-benchmarks/ersst_observation.nc'
    sst=xr.open_dataset(src).sst.sel(time=slice('1984-12','2024-12'),lat=slice(-60,60)).load()
    mask=masks(sst.lat.values,sst.lon.values);w=mask*np.cos(np.deg2rad(sst.lat))
    regional=(sst*w).sum(('lat','lon'))/w.where(sst.notnull()).sum(('lat','lon'))
    regional=regional.transpose('time','region')
    clim=regional.sel(time=slice('1985','2014')).groupby('time.month').mean('time')
    anomaly=regional.groupby('time.month')-clim
    q90=regional.sel(time=slice('1985','2014')).groupby('time.month').quantile(.9,dim='time')
    state=xr.Dataset({'sst':regional,'sst_anomaly':anomaly,'monthly_q90':q90})
    state.to_netcdf(OUT/'Figure4_ERSST_regional_state.nc')
    obs=xr.load_dataset(OUT/'Figure4_ORAS5_regional_monthly.nc')
    index=pd.read_csv(PAPER/'Variants/Four_regimes_20260921_components/Data/derived/climate_indices_1985_2020.csv',parse_dates=['time']).set_index('time')
    rows=[];validation=[];sample=[]
    for region in ORDER:
        od=obs.sel(region=region).to_dataframe()[['mlt','surface_heating','horizontal_advection']]
        tab=index.join(pd.DataFrame({'sst':regional.sel(region=region).to_pandas()}))
        tab=tab.join(od).assign(sst_memory=regional.sel(region=region).to_pandas().shift(1),ml_memory=od.mlt.shift(1)).dropna()
        cols=['ENSO','IOD','PDO','NAO','SAM','sst_memory','ml_memory','surface_heating','horizontal_advection']
        # Trend and scaling nuisance parameters are fitted in each training fold.
        def transform(train,test):
            def residual(v,ixtrain,ixtest):
                t=np.arange(len(tab),dtype=float)
                nuisance_month=np.eye(12)[tab.index.month.values-1][:,1:]
                nuis=np.column_stack([np.ones(len(t)),t,nuisance_month])
                beta=np.linalg.lstsq(nuis[ixtrain],v[ixtrain],rcond=None)[0]
                a=v[ixtrain]-nuis[ixtrain]@beta;b=v[ixtest]-nuis[ixtest]@beta
                scale=np.std(a,axis=0);scale=np.where(scale>0,scale,1)
                return a/scale,b/scale
            x0,x1=residual(tab[cols].values,train,test);y0,y1=residual(tab.sst.values[:,None],train,test)
            return x0,x1,y0.ravel(),y1.ravel()
        ix=np.arange(len(tab));x,_,y,_=transform(ix,ix);shares=attribution(x,y)
        assert np.isclose(shares.sum(),1,atol=1e-10) and shares.min()>-1e-9
        rng=np.random.default_rng(20261029);boot=[];years=tab.index.year.unique()
        for k in range(400):
            yi=rng.choice(years,len(years),replace=True);bi=np.concatenate([np.flatnonzero(tab.index.year==yr) for yr in yi])
            boot.append(attribution(x[bi],y[bi]))
        lower,upper=np.quantile(boot,[.025,.975],axis=0)
        for j,name in enumerate(['ENSO','Other climate modes','Local processes','Unresolved']):
            rows.append(dict(Region=region,Group=name,Share=shares[j],Low=lower[j],High=upper[j],N=len(tab)))
        errors={k:[] for k in ['Full','Without ENSO','Without other','Without local','Null']}
        for yr in years:
            train=np.flatnonzero(tab.index.year!=yr);test=np.flatnonzero(tab.index.year==yr)
            x0,x1,y0,y1=transform(train,test)
            for k,omit in [('Full',None),('Without ENSO',0),('Without other',1),('Without local',2)]:
                ids=[c for g in range(3) if g!=omit for c in GROUPS[g]]
                coef=r2fit(x0[:,ids],y0)[1];pred=np.column_stack([np.ones(len(test)),x1[:,ids]])@coef
                errors[k].extend((y1-pred)**2)
            errors['Null'].extend(y1**2)
        den=np.mean(errors['Null'])
        for k in errors:validation.append(dict(Region=region,Model=k,Heldout_year_R2=1-np.mean(errors[k])/den))
        t=tab.reset_index();t['Region']=region;sample.append(t)
    pd.DataFrame(rows).to_csv(OUT/'Figure4_observed_group_associations.csv',index=False)
    pd.DataFrame(validation).to_csv(OUT/'SI_Figure4_association_year_holdout.csv',index=False)
    pd.concat(sample).to_csv(OUT/'Figure4_observed_association_samples.csv',index=False)
    print(pd.DataFrame(rows).pivot(index='Region',columns='Group',values='Share').round(3).to_string(),flush=True)
if __name__=='__main__':main()
