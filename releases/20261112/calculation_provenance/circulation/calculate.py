"""DJF Fig3 from immutable forecast files; exact starts, leads and member IDs."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mhw-fig3-patterns-mpl')
import argparse
import hashlib
import itertools
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
PAPER = Path('/data3/luoq/FEIS/research/papers/mhw-predictability')
RAW = PAPER / 'teleconnection_data'
OLD = PAPER / 'Figures/Design_20261029_v7'
sys.path.insert(0, str(ROOT/'dependencies'))
import k02_calculate_teleconnection as reader
CONTRACT = json.loads((ROOT/'contract.json').read_text())
EVENTS = list(CONTRACT['event_windows'])
MODELS = list(CONTRACT['models'])
LEADS = np.arange(1, 9)
MONTHS = [12, 1, 2]
LAT = np.arange(-60., 61.)
LON = np.arange(360.)
DATA = ROOT/'source_data'
CACHE = ROOT/'cache'
for directory in [DATA, CACHE, ROOT/'figures']:
    directory.mkdir(exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def write_nc(ds, path):
    ds.to_netcdf(path, encoding={v: {'zlib': True, 'complevel': 3} for v in ds.data_vars})

def targets():
    base = [pd.Period(year=y, month=m, freq='M') for y in range(1992,2021) for m in MONTHS]
    base = [t for t in base if str(t) not in CONTRACT.get('baseline_excluded_months',[])]
    event = [[p for p in pd.period_range(*CONTRACT['event_windows'][e],freq='M')] for e in EVENTS]
    return base, event

def forecast(field, model):
    base, event = targets()
    all_targets = sorted(set(base).union(*map(set,event)))
    requests = {}
    for t in all_targets:
        for li, lead in enumerate(LEADS):
            requests.setdefault(t-int(lead), []).append((li,t))
    members = np.arange(1, CONTRACT['models'][model][1]+1)
    plim=CONTRACT['precipitation_latitude_limit_deg']
    lat = LAT if field=='z200' else np.arange(-float(plim),float(plim)+1)
    shape = (8, 3, len(lat), 360)
    # All selected member values must be valid, so member-mean and common
    # monthly climatology commute exactly. No changing-member ensemble mean.
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros((8,3),dtype=np.int32)
    ev = np.full((3,8,3,len(lat),360),np.nan,dtype=np.float64)
    audit=[]; invalid_baseline=[]
    for n,(init,requests_at_init) in enumerate(sorted(requests.items())):
        path = reader.model_file_path(RAW,model,init,field)
        if not path.exists(): raise FileNotFoundError(path)
        with xr.open_dataset(path,decode_times=False) as ds:
            expected = (init.year-1960)*12+init.month-1
            sh = np.flatnonzero(np.isclose(ds.S.values,expected,atol=1e-6,rtol=0))
            if len(sh)!=1: raise ValueError(f'Exact start unavailable: {path} {expected}')
            lead_indices=[]
            for li,t in requests_at_init:
                hit=np.flatnonzero(np.isclose(ds.L.values,float(LEADS[li])+.5,atol=1e-6,rtol=0))
                if len(hit)!=1: raise ValueError(f'Exact future lead absent: {path} {LEADS[li]}')
                lead_indices.append(int(hit[0]))
            subset=ds.isel(S=int(sh[0]),L=lead_indices).sel(M=members)
            # Slice before unit conversion/preprocessing to avoid loading unused starts.
            yi=np.flatnonzero((subset.Y.values>=lat.min())&(subset.Y.values<=lat.max()))
            subset=subset.isel(Y=yi)
            a=reader.preprocess_field_dataarray(subset,field,200.)
            a=reader.normalize_member_dimension(a).transpose('L','member','lat','lon')
            units=str(a.attrs.get('units',''))
            if field=='precipitation' and units!='mm/day': raise ValueError((model,units))
            if field=='z200' and units!='m': raise ValueError((model,units))
            assert np.array_equal(a.lat,lat) and np.array_equal(a.lon,LON)
            values=a.load().values.astype(np.float64)
            mean=values.mean(axis=1)
            digest=hashlib.sha256(values.tobytes()).hexdigest()
            for k,(li,t) in enumerate(requests_at_init):
                if not np.isfinite(values[k]).all() or np.max(np.abs(values[k]))>1e8:
                    if t in base:
                        invalid_baseline.append({'year':t.year,'target':str(t),'lead':int(LEADS[li]),'path':str(path),'nonfinite_fraction':float((~np.isfinite(values[k])).mean())})
                        continue
                    raise ValueError(f'Incomplete event data: {path} target={t}')
                mi=MONTHS.index(t.month)
                if t in base:
                    sums[li,mi]+=mean[k];counts[li,mi]+=1
                for ei,ets in enumerate(event):
                    if t in ets:ev[ei,li,mi]=mean[k]
                audit.append(dict(field=field,model=model,path=str(path.resolve()),file_bytes=path.stat().st_size,
                    file_mtime_ns=path.stat().st_mtime_ns,initialization=str(init),target=str(t),lead=int(LEADS[li]),
                    exact_S=float(ds.S.values[sh[0]]),exact_L=float(ds.L.values[lead_indices[k]]),
                    member_first=1,member_last=len(members),selected_array_sha256=digest))
        if n%65==0: print(field,model,f'{n+1}/{len(requests)}',str(init),flush=True)
    if invalid_baseline:
        (DATA/f'invalid_baseline_{field}_{model}.json').write_text(json.dumps(invalid_baseline,indent=2)+'\n')
        raise ValueError(f'Baseline requires common exclusion audit: {invalid_baseline}')
    expected_counts=np.array([sum(t.month==m for t in base) for m in MONTHS])
    assert (counts==expected_counts[None]).all(),counts
    assert np.isfinite(ev).all()
    climatology=sums/counts[:,:,None,None]
    anom=ev-climatology[None]
    out=xr.Dataset({'forecast':(('event','lead','lat','lon'),anom.mean(axis=2).astype('float32')),
        'monthly_anomaly':(('event','lead','month','lat','lon'),anom.astype('float32')),
        'climatology':(('lead','month','lat','lon'),climatology.astype('float32')),
        'baseline_count':(('lead','month'),counts)},
        coords={'event':EVENTS,'lead':LEADS,'month':MONTHS,'lat':lat,'lon':LON},
        attrs={'model':model,'field':field,'units':units,'baseline':CONTRACT['baseline'],
            'window':'DJF; equal month weights','members':','.join(map(str,members)),
            'lead':'target calendar month minus initialization month; exact L=lead+0.5'})
    write_nc(out,CACHE/f'{field}_{model}_DJF.nc')
    pd.DataFrame(audit).to_csv(DATA/f'coordinate_audit_{field}_{model}.csv',index=False)
    print('DONE',field,model,flush=True)

def observations(only_field=None):
    base,event=targets()
    years=sorted({t.year for t in base}.union(*[{t.year for t in ets} for ets in event]))
    audit=[]
    for field in ([only_field] if only_field else ['z200','precipitation']):
        plim=CONTRACT['precipitation_latitude_limit_deg']
        lat=LAT if field=='z200' else np.arange(-float(plim),float(plim)+1)
        frames=[];times=[]
        for year in years:
            prefix='pressure' if field=='z200' else 'single'
            path=RAW/'observations/ERA5'/f'era5_{prefix}_levels_{year}.nc'
            with xr.open_dataset(path) as ds:
                subset=ds.sel(valid_time=ds.valid_time.dt.month.isin(MONTHS))
                if field=='z200': subset=subset.sel(pressure_level=200.,drop=True)
                # ERA5 0.25-degree grid includes exact nodes of the 1-degree analysis grid.
                subset=subset.sel(latitude=lat,longitude=LON)
                a=reader.preprocess_field_dataarray(subset,field,200.)
                a=a.transpose('valid_time','lat','lon').load()
                if field=='precipitation': a=a*1000. # ERA5 monthly mean daily accumulation, m -> mm/day
                frames.append(a.values)
                times.extend(pd.to_datetime(a.valid_time.values).to_period('M').astype(str).tolist())
                audit.append(dict(field=field,path=str(path.resolve()),file_bytes=path.stat().st_size,
                    file_mtime_ns=path.stat().st_mtime_ns,selected_array_sha256=hashlib.sha256(a.values.tobytes()).hexdigest()))
        series=xr.DataArray(np.concatenate(frames),dims=['target','lat','lon'],coords={'target':times,'lat':lat,'lon':LON})
        clim=np.stack([series.sel(target=[str(t) for t in base if t.month==m]).mean('target').values for m in MONTHS])
        ev=np.stack([np.stack([series.sel(target=str(next(t for t in ets if t.month==m))).values for m in MONTHS])-clim for ets in event])
        assert np.isfinite(ev).all()
        out=xr.Dataset({'observed':(('event','lat','lon'),ev.mean(axis=1)),
            'monthly_anomaly':(('event','month','lat','lon'),ev),
            'climatology':(('month','lat','lon'),clim)},coords={'event':EVENTS,'month':MONTHS,'lat':lat,'lon':LON},
            attrs={'baseline':CONTRACT['baseline'],'window':'DJF','units':'m' if field=='z200' else 'mm/day','interpolation':'exact 1-degree grid-node sampling'})
        write_nc(out,CACHE/f'{field}_ERA5_DJF.nc')
        print('OBS DONE',field,flush=True)
    pd.DataFrame(audit).to_csv(DATA/'ERA5_source_audit.csv',index=False)

def corr(a,b,lat,mask=None):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    w=np.broadcast_to(np.cos(np.deg2rad(np.asarray(lat)))[:,None],a.shape)
    keep=np.isfinite(a)&np.isfinite(b)
    if mask is not None: keep &= np.asarray(mask,dtype=bool)
    assert keep.sum()>3
    a,b,w=a[keep],b[keep],w[keep];w=w/w.sum()
    a=a-np.sum(w*a);b=b-np.sum(w*b)
    return float(np.sum(w*a*b)/np.sqrt(np.sum(w*a*a)*np.sum(w*b*b)))

def domain_mask(region):
    # Preserve major-basin identity; revised half-open +/-23.5-degree bands.
    with xr.open_dataset(DATA/'seven_ocean_regions.nc') as ds:
        return ds['mask'].sel(region=region).values.astype(bool)

def assemble():
    z=xr.concat([xr.load_dataset(CACHE/f'z200_{m}_DJF.nc').forecast for m in MODELS],dim=xr.IndexVariable('model',MODELS))
    p=xr.concat([xr.load_dataset(CACHE/f'precipitation_{m}_DJF.nc').forecast for m in MODELS],dim=xr.IndexVariable('model',MODELS))
    zo=xr.load_dataset(CACHE/'z200_ERA5_DJF.nc').observed
    po=xr.load_dataset(CACHE/'precipitation_ERA5_DJF.nc').observed
    zf=z.mean('model',skipna=False);pf=p.mean('model',skipna=False)
    masks=xr.load_dataset(DATA/'seven_ocean_regions.nc')
    assert np.asarray(masks['mask'].sum('region')).max()==1
    assert np.array_equal(masks.lat,LAT) and np.array_equal(masks.lon,LON)
    # Re-serialize the data unchanged; verify logical identity in validate.py.
    masks.to_netcdf(DATA/'seven_ocean_regions.nc')
    region_order=['NP','SP','TA','NA','SA','IO']
    atm=masks['mask'].sel(region=region_order).rename('atmospheric_mask')
    atm.attrs['diagnostic_support']='Fig4 major-basin identity with revised +/-23.5 degree bands; excludes land'
    tp=masks['mask'].sel(region='TP',lat=p.lat)
    maps=xr.Dataset({'observed':zo,'forecast':zf,'forecast_error':zf-zo,
        'canonical':zo.sel(event=EVENTS[:2]).mean('event'),
        'model_forecast':z,'atmospheric_mask':atm},attrs={'window':'DJF','baseline':CONTRACT['baseline'],
            'error_map_leads':'equally averaged t+1 through t+8','units':'m',
            'canonical':'equal two-event observed composite, not a causal ENSO isolation',
            'regional_support':'Fig4 major-basin identity; revised +/-23.5 degree bands; land excluded'})
    write_nc(maps,DATA/'Figure3_Z200_fields.nc')
    write_nc(xr.Dataset({'forecast':pf,'observed':po,'model_forecast':p,'source_mask':tp}),DATA/'Figure3_precipitation_fields.nc')
    rows=[]
    for e in EVENTS:
        for lead in LEADS:
            rp=corr(pf.sel(event=e,lead=lead),po.sel(event=e),p.lat,tp)
            for b in region_order:
                rz=corr(zf.sel(event=e,lead=lead),zo.sel(event=e),LAT,atm.sel(region=b))
                rows.append(dict(Event=e,Lead=int(lead),Region=b,Rain_r=rp,Z200_r=rz,Rain_error=1-rp,Z200_error=1-rz))
    df=pd.DataFrame(rows)
    assert len(df)==144
    df.to_csv(DATA/'Figure3b_all_samples.csv',index=False)
    skill=df.pivot(index=['Region','Lead'],columns='Event',values='Z200_r')
    skill['Delta_r']=skill['2023/24']-(skill['1997/98']+skill['2015/16'])/2
    full=pd.MultiIndex.from_product([region_order,range(1,10)],names=['Region','Lead'])
    skill.reindex(full).reset_index().to_csv(DATA/'SI_regional_skill_change.csv',index=False)
    # Exact cluster bootstrap: 3^3 equiprobable ordered resamples of whole events.
    x=df.Rain_error.to_numpy();y=df.Z200_error.to_numpy();xx=np.linspace(x.min(),x.max(),200)
    fit=np.polyfit(x,y,1);r=float(np.corrcoef(x,y)[0,1])
    bands=[];boot=[]
    for chosen in itertools.product(EVENTS,repeat=3):
        d=pd.concat([df[df.Event==e] for e in chosen])
        f=np.polyfit(d.Rain_error,d.Z200_error,1)
        bands.append(np.polyval(f,xx));boot.append(dict(events='|'.join(chosen),slope=float(f[0]),intercept=float(f[1])))
    bounds=np.percentile(bands,[2.5,97.5],axis=0)
    pd.DataFrame({'x':xx,'fit':np.polyval(fit,xx),'lower':bounds[0],'upper':bounds[1]}).to_csv(DATA/'Figure3b_event_bootstrap_envelope.csv',index=False)
    pd.DataFrame(boot).to_csv(DATA/'SI_event_bootstrap_fits.csv',index=False)
    omission=[]
    for e in EVENTS:
        d=df[df.Event!=e];f=np.polyfit(d.Rain_error,d.Z200_error,1)
        omission.append(dict(omitted_event=e,slope=float(f[0]),intercept=float(f[1]),r=float(np.corrcoef(d.Rain_error,d.Z200_error)[0,1])))
    pd.DataFrame(omission).to_csv(DATA/'SI_event_omission.csv',index=False)
    blocks=[df[df.Event==e].sort_values(['Region','Lead']) for e in EVENTS]
    perm=[]
    xp=np.concatenate([b.Rain_error for b in blocks])
    for order in itertools.permutations(range(3)):
        yp=np.concatenate([blocks[i].Z200_error for i in order]);perm.append(float(np.corrcoef(xp,yp)[0,1]))
    pvalue=float(np.mean(np.abs(perm)>=abs(r)-1e-12))
    by_event=df.groupby('Event')[['Rain_error','Z200_error']].mean()
    diagnostics=[]
    canonical=maps.canonical.values
    for e in EVENTS:
        obs=zo.sel(event=e).values
        err=(zf.sel(event=e)-zo.sel(event=e)).mean('lead').values
        for b in region_order:
            m=domain_mask(b)
            w=np.broadcast_to(np.cos(np.deg2rad(LAT))[:,None],canonical.shape)[m];w=w/w.sum()
            c=canonical[m];o=obs[m];v=err[m]
            c=c-np.sum(w*c);o=o-np.sum(w*o);v=v-np.sum(w*v)
            diagnostics.append(dict(Event=e,Region=b,observed_vs_canonical_r=corr(obs,canonical,LAT,m),
                observed_projection=float(np.sum(w*o*c)/np.sum(w*c*c)),error_vs_canonical_r=corr(err,canonical,LAT,m),
                error_projection=float(np.sum(w*v*c)/np.sum(w*c*c))))
    pd.DataFrame(diagnostics).to_csv(DATA/'SI_canonical_pattern_correspondence.csv',index=False)
    stats={'sample_rows':len(df),'independent_events':3,'common_leads':LEADS.tolist(),'missing_leads':[9],
        'pooled_r_descriptive':r,'pooled_slope':float(fit[0]),'event_permutation_p_two_sided':pvalue,
        'permutation_values':perm,'bootstrap_samples_exact':27,'bootstrap_limitation':'only three event clusters; descriptive envelope, not a reliable population CI',
        'event_omission':omission,'mean_errors_by_event':by_event.to_dict(orient='index'),
        'negative_cells':int((skill.Delta_r<0).sum()),'positive_cells':int((skill.Delta_r>0).sum()),
        'valid_heatmap_cells':len(skill),'missing_heatmap_cells':6,
        'delta_min':float(skill.Delta_r.min()),'delta_max':float(skill.Delta_r.max()),
        'source_domain':'Pacific ocean mask between -23.5 and +23.5 latitude; not the old +/-20 TP mask',
        'recipient_domain':'Fig4 major-basin identity repartitioned at +/-23.5 degrees; land excluded',
        'model_counts':{m:CONTRACT['models'][m][1] for m in MODELS}}
    (ROOT/'statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['forecast','observations','assemble']);p.add_argument('--field');p.add_argument('--model');a=p.parse_args()
    if a.action=='forecast': forecast(a.field,a.model)
    elif a.action=='observations':observations(a.field)
    else:assemble()
