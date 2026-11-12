"""Fixed-support neighbourhood ROC AUC. Original sources and headline untouched."""
from pathlib import Path
import importlib.util, json, hashlib, warnings, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
OLD=WORK/'Figure1_skill_decrease_20261111_v9'
CACHE=WORK/'figure4_auc_attribution_20261104_v9'
DATA=ROOT/'source_data'
spec=importlib.util.spec_from_file_location('legacy',CACHE/'calculate_auc_partition.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
YEARS=[1997,2015,2023];EVENTS=['1997/98','2015/16','2023/24'];REG=['TP','NP','SP','TA','NA','SA','IO']
tests=[];sources=[]
def record(p):
    sources.append(dict(path=str(p),sha256=legacy.digest(p)));return p
def check(name,ok,detail=None):
    tests.append(dict(name=name,passed=bool(ok),detail=detail));assert ok,(name,detail)
def mean(v,axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);return np.nanmean(v,axis=axis)

def prepare():
    """Replay archived core exactly; read only extra target months from same reader."""
    dom=np.load(record(CACHE/'source_data/evaluation_domain.npz'));inside=dom['domain']
    ref=np.load(record(legacy.REF));months=list(ref['months'].astype(str));obsall=ref['observations'][:,inside];qall=ref['threshold'][:,inside]
    masks=np.load(record(legacy.MASK));lookup={(str(i),str(t),int(l)):v[inside] for i,t,l,v in zip(masks['initialization'],masks['target_month'],masks['lead_month'],masks['sample_masks'])}
    ns=legacy.load_reader()
    for year in YEARS:
        output=ROOT/'cache'/f'{year}_extended.npz'
        if output.exists():
            print('Reuse',output.name,flush=True);continue
        targets=[str(t) for t in pd.period_range(f'{year}-04',periods=18,freq='M')]
        meta=pd.read_csv(record(CACHE/f'legacy_audit_cache/{year}_samples.csv'))
        core=np.load(record(CACHE/f'legacy_audit_cache/{year}_fields.npz'))['forecast']
        pred=np.full((9,18,int(inside.sum())),np.nan,dtype=np.float32);counts=np.zeros((9,18),int)
        for i,row in enumerate(meta.itertuples()):
            li=int(row.Lead_Month)-1;ti=targets.index(row.Target_Month)
            m=lookup.get((row.Initialization,row.Target_Month,li+1),np.ones(inside.sum(),bool))
            pred[li,ti]=np.where(m,core[i],np.nan);counts[li,ti]=row.model_count
        needed={}
        for li in range(9):
            for ti in [0,1,2,15,16,17]:
                init=str(pd.Period(targets[ti])-li-1)
                needed.setdefault(init,[]).append((li,ti))
        audit=[]
        for init,pairs in sorted(needed.items()):
            fields={}
            for model in legacy.MODELS:
                p=WORK/'data'/model/f'sst_{init.replace("-", "")}.nc'
                row=dict(model=model,initialization=init,path=str(p),exists=p.exists())
                if p.exists():
                    record(p)
                    with xr.open_dataset(p,decode_times=False) as d:
                        row.update(S=np.atleast_1d(d.S.values).tolist(),L=d.L.values.tolist())
                        check(f'lead coordinate {model} {init}',np.allclose(d.L.values,np.arange(len(d.L))+.5))
                    fields[model]=legacy.legacy_means(p,9,ns)
                audit.append(row)
            for li,ti in pairs:
                arr=[f[li+1] for f in fields.values() if li+1 in f]
                if not arr:continue
                p=mean(np.stack(arr),0).astype('float32')[inside]
                m=lookup.get((init,targets[ti],li+1),np.ones(inside.sum(),bool))
                pred[li,ti]=np.where(m,p,np.nan);counts[li,ti]=len(arr)
            print('Extend',year,init,'models',len(fields),flush=True)
        obs=obsall[[months.index(t) for t in targets]];q=qall[[int(t[5:])-1 for t in targets]]
        np.savez_compressed(output,pred=pred,obs=obs,threshold=q,targets=targets,model_count=counts)
        (ROOT/'cache'/f'{year}_source_audit.json').write_text(json.dumps(audit,indent=2))
        check(f'all 162 event-lead-month samples {year}',np.all(counts>0))
    # Cache-source manifest is separate from final derived-source manifest.
    cache_manifest=ROOT/'cache_manifest.json'
    previous=json.loads(cache_manifest.read_text()) if cache_manifest.exists() else []
    cache_manifest.write_text(json.dumps(list({x['path']:x for x in previous+sources}.values()),indent=2))

def neighbourhood(dom,eligible,size=11,same_basin=True):
    lat=dom['lat'];lon=dom['lon'];ii=np.rint(lat+89.5).astype(int);jj=np.rint(lon-.5).astype(int)
    grid=np.full((180,360),-1,int);grid[ii,jj]=np.arange(len(ii))
    broad=np.where(dom['region20']<3,0,np.where(dom['region20']<6,1,np.where(dom['region20']==6,2,3)))
    offsets=np.arange(-(size//2),size//2+1)
    di,dj=np.meshgrid(offsets,offsets,indexing='ij')
    i=ii[:,None]+di.ravel();j=(jj[:,None]+dj.ravel())%360
    ids=grid[np.clip(i,0,179),j];safe=np.maximum(ids,0)
    keep=(ids>=0)&(i>=0)&(i<180)&eligible[safe]
    if same_basin:keep &= broad[safe]==broad[:,None]
    ids[~keep]=-1
    check('11-degree coordinate span' if size==11 else 'native neighbourhood',ids.shape[1]==size*size)
    check('longitude wrap',np.all(np.minimum(abs(lon[np.maximum(ids,0)]-lon[:,None]),360-abs(lon[np.maximum(ids,0)]-lon[:,None]))[ids>=0]<=size//2+1e-6))
    return ids

def auc_rows(score,truth,weight):
    """Weighted Mann-Whitney statistic, exact half weight for tied pairs."""
    order=np.argsort(score,axis=1,kind='stable')
    s=np.take_along_axis(score,order,1);y=np.take_along_axis(truth,order,1);w=np.take_along_axis(weight,order,1)
    p=w*y;n=w*(1-y);c=np.cumsum(n,axis=1);before=c-n
    # Most physical SST scores have no ties; retain exact tie handling regardless.
    tied=np.any(s[:,1:]==s[:,:-1],axis=1)
    numerator=np.sum(p*before,axis=1)
    for k in np.flatnonzero(tied):
        starts=np.r_[0,np.flatnonzero(s[k,1:]!=s[k,:-1])+1]
        pn=np.add.reduceat(p[k],starts);nn=np.add.reduceat(n[k],starts)
        numerator[k]=np.sum(pn*(np.cumsum(nn)-.5*nn))
    den=p.sum(1)*n.sum(1)
    return np.divide(numerator,den,out=np.full(len(score),np.nan),where=den>0)

def pooled(pred,obs,q,ids,weights):
    n=len(ids);t=len(obs);a=np.full((9,n),np.nan);m=np.full((9,n),np.nan)
    positives=np.zeros(n,int);negatives=np.zeros(n,int)
    for start in range(0,n,256):
        end=min(n,start+256);ix=ids[start:end];good=ix>=0;ix=np.maximum(ix,0)
        y=(obs[:,ix]>q[:,ix]).transpose(1,0,2).reshape(end-start,-1)
        w=np.broadcast_to(np.where(good,weights[ix],0)[:,None,:],(end-start,t,ids.shape[1])).reshape(end-start,-1)
        positives[start:end]=np.sum(y&(w>0),axis=1);negatives[start:end]=np.sum((~y)&(w>0),axis=1)
        o=(obs[:,ix]-q[:,ix]).transpose(1,0,2).reshape(end-start,-1)
        for lead in range(9):
            s=(pred[lead][:,ix]-q[:,ix]).transpose(1,0,2).reshape(end-start,-1)
            # Zero-weight padding must be finite for cumsum/sort and MSE arithmetic.
            s=np.where(w>0,s,0);o_safe=np.where(w>0,o,0)
            a[lead,start:end]=auc_rows(s,y,w)
            wm=w*y;den=wm.sum(1)
            se=(np.maximum(s,0)-np.maximum(o_safe,0))**2
            m[lead,start:end]=np.divide(np.sum(wm*se,1),den,out=np.full(end-start,np.nan),where=den>0)
    return a,m,positives,negatives

def main():
    prepare()
    dom=np.load(record(CACHE/'source_data/evaluation_domain.npz'));n=len(dom['lat']);w=dom['weights'].astype(float)
    arrays=[np.load(record(ROOT/'cache'/f'{y}_extended.npz')) for y in YEARS]
    records=[];variants={};stats=[]
    old=np.load(record(OLD/'source_data/event_lead_fields.npz'))
    for months,sl in [(12,slice(3,15)),(18,slice(None))]:
        ps=[z['pred'][:,sl] for z in arrays];os=[z['obs'][sl] for z in arrays];qs=[z['threshold'][sl] for z in arrays]
        eligible=np.logical_and.reduce([np.isfinite(p).all((0,1))&np.isfinite(o).all(0)&np.isfinite(q).all(0) for p,o,q in zip(ps,os,qs)])
        for size in [1,11]:
            name=f'{months}months_{size}deg';out=DATA/f'{name}.npz'
            ids=neighbourhood(dom,eligible,size)
            if out.exists():z=np.load(out);a=z['auc'];m=z['mse'];pos=z['positive'];neg=z['negative']
            else:
                with ThreadPoolExecutor(max_workers=3) as pool:
                    jobs=[pool.submit(pooled,p,o,q,ids,w) for p,o,q in zip(ps,os,qs)]
                    results=[j.result() for j in jobs]
                a,m,pos,neg=[np.stack([v[k] for v in results]) for k in range(4)]
                np.savez_compressed(out,auc=a,mse=m,positive=pos,negative=neg,eligible=eligible,neighbour_count=(ids>=0).sum(1))
            ac=np.isfinite(a).all((0,1));rc=np.isfinite(m).all((0,1))
            variants[name]=(a,m,ac,rc)
            if months==12 and size==1:
                check('original native AUC replay',np.allclose(a,old['auc'],equal_nan=True,atol=1e-12))
                check('original intensity MSE replay',np.allclose(m,old['conditional_intensity_mse'],equal_nan=True,atol=2e-7))
            # Independent sklearn verification on deterministic real and tied cases.
            for e in range(3):
                valid=np.flatnonzero(ac)
                for cell in valid[np.linspace(0,len(valid)-1,6,dtype=int)]:
                    ni=ids[cell];ni=ni[ni>=0]
                    for li in [0,8]:
                        yy=(os[e][:,ni]>qs[e][:,ni]).ravel();ss=(ps[e][li][:,ni]-qs[e][:,ni]).ravel();ww=np.broadcast_to(w[ni],(months,len(ni))).ravel()
                        truth=roc_auc_score(yy,ss,sample_weight=ww)
                        check(f'sklearn {name} {e} {li} {cell}',abs(a[e,li,cell]-truth)<1e-11)
            for metric,mask in [('AUC',ac),('RMSE',rc)]:
                stats.append(dict(variant=name,metric=metric,cells=int(mask.sum()),total_cells=n,cell_coverage=float(mask.mean()),area_coverage=float(w[mask].sum()/w.sum()),eligible_cells=int(eligible.sum())))
            print('Completed',name,'AUC cell/area',ac.mean(),w[ac].sum()/w.sum(),flush=True)
            for ri,r in enumerate(REG):
                full=dom['region20']==ri
                for metric,mask in [('AUC',ac),('RMSE',rc)]:
                    use=full&mask;ww=w[use];ww=ww/ww.sum()
                    aa=np.sum(a[:,:,use]*ww,axis=2) if metric=='AUC' else np.sum(m[:,:,use]*ww,axis=2)
                    target=float(aa[2].mean()) if metric=='AUC' else float(np.sqrt(aa[2].mean()))
                    historical=float(.5*(aa[0].mean()+aa[1].mean())) if metric=='AUC' else float(np.sqrt(.5*(aa[0].mean()+aa[1].mean())))
                    diff=.5*(aa[0]+aa[1])-aa[2] if metric=='AUC' else np.sqrt(aa[2])-np.sqrt(.5*(aa[0]+aa[1]))
                    records.append(dict(variant=name,metric=metric,region=r,canonical=historical,target=target,change=historical-target if metric=='AUC' else target-historical,lead_min=float(diff.min()),lead_max=float(diff.max()),coverage=float(w[use].sum()/w[full].sum()),cells=int(use.sum())))
    stats=pd.DataFrame(stats);stats.to_csv(DATA/'coverage_comparison.csv',index=False)
    tab=pd.DataFrame(records);tab.to_csv(DATA/'regional_sensitivity.csv',index=False)
    a,m,ac,rc=variants['18months_11deg'];tab[tab.variant=='18months_11deg'].drop(columns='variant').to_csv(DATA/'regional_changes.csv',index=False)
    ad=mean(.5*(a[0]+a[1])-a[2],0);ad[~ac]=np.nan
    rd=np.sqrt(mean(m[2],0))-np.sqrt(.5*(mean(m[0],0)+mean(m[1],0)));rd[~rc]=np.nan
    leads=[]
    for ri,r in enumerate(REG):
        for metric,mask in [('AUC',ac),('RMSE',rc)]:
            use=mask&(dom['region20']==ri);ww=w[use]/w[use].sum()
            v=np.sum((a if metric=='AUC' else m)[:,:,use]*ww,axis=2)
            d=.5*(v[0]+v[1])-v[2] if metric=='AUC' else np.sqrt(v[2])-np.sqrt(.5*(v[0]+v[1]))
            if metric=='RMSE':v=np.sqrt(v)
            for li in range(9):leads.append(dict(metric=metric,region=r,lead=li+1,change=d[li],event1997=v[0,li],event2015=v[1,li],event2023=v[2,li]))
    pd.DataFrame(leads).to_csv(DATA/'regional_changes_by_lead.csv',index=False)
    ds=xr.Dataset({k:('cell',v) for k,v in dict(auc_decrease=ad,rmse_increase=rd,auc_common=ac,rmse_common=rc,region=dom['region20'],area_weight=w).items()},coords=dict(cell=np.arange(n),lat=('cell',dom['lat']),lon=('cell',dom['lon'])))
    ds.attrs.update(window='April-September following year (18 months)',support='11x11 one-degree ocean-cell neighbourhood within same broad basin',profile='Archived; original source-coordinate limitations unresolved',no_interpolation='true')
    ds.to_netcdf(DATA/'canonical_relative_changes.nc',encoding={k:dict(zlib=True,complevel=4) for k in ds.data_vars})
    np.savez_compressed(DATA/'event_lead_fields.npz',auc=a,conditional_intensity_mse=m,events=EVENTS,auc_common=ac,rmse_common=rc)
    check('undefined cells retained',np.isnan(ad[~ac]).all() and np.isnan(rd[~rc]).all())
    check('both sign directions retained',np.nanmin(ad)<0<np.nanmax(ad))
    check('valid ROC bounds',np.nanmin(a)>=-1e-10 and np.nanmax(a)<=1+1e-10)
    check('tied scores AUC one-half',np.allclose(auc_rows(np.ones((1,4)),np.array([[0,1,0,1]]),np.array([[1,2,3,4]])),.5))
    (ROOT/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    (ROOT/'calculation_validation.json').write_text(json.dumps(dict(checks=tests,passed=len(tests)),indent=2))
    print(stats.to_string(index=False),flush=True)
if __name__=='__main__':main()
