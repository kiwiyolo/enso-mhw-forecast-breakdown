"""Reaggregate the retained Figure4 accounting under Fig3 latitude boundaries.

This is association-weighted bookkeeping, not causal process attribution.
Raw forecast selection, global AUC, OLS baseline and estimator are unchanged.
"""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np,pandas as pd,xarray as xr
from regions import masks,ORDER

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data'
PAPER=Path('/data3/luoq/FEIS/research/papers/mhw-predictability')
BASE=PAPER/'Figures/Figure4_clear_skill_legend_20261110_v22'
FIG3=PAPER/'Figures/Figure1-3_Nature_layout_20261110_v7/source_data/Fig3/seven_ocean_regions.nc'
AUDIT=Path('/data3/luoq/FEIS-archive/legacy/p03-ocean-sst-ensemble/figure4_auc_attribution_20261104_v9')
REGIONS=ORDER+['Other seas']
PARTS=['ENSO','Other','Memory','Atmosphere','Ocean','Unallocated']
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def record(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')

def main():
    checks={};manifest=[]
    d=xr.load_dataset(FIG3);m=masks(d.lat.values,d.lon.values)
    assert np.array_equal(m.values,d['mask'].values)
    d.attrs['figure4_sync']='Figure4 v23 uses the same major-basin masks and +/-23.5-degree boundaries.'
    d.to_netcdf(DATA/'seven_ocean_regions.nc')
    checks['fig3_mask_identical']=True
    checks['disjoint_regions']=bool(m.sum('region').max()<=1)
    assert digest(DATA/'Figure4a_activity_valid_extent.nc')==digest(BASE/'source_data/Figure4a_activity_valid_extent.nc')
    checks['activity_map_byte_identical']=True
    old=module(ROOT/'reference_code/prepare_v8.py','owen_original')
    raw=pd.read_csv(DATA/'Figure4_observed_association_samples.csv',keep_default_na=False)
    assoc=pd.read_csv(DATA/'Figure4_observed_group_associations.csv',keep_default_na=False)
    oldassoc=pd.read_csv(BASE/'source_data/Figure4_observed_group_associations.csv',keep_default_na=False)
    cols=['ENSO','IOD','PDO','NAO','SAM','sst_memory','ml_memory','surface_heating','horizontal_advection']
    weights=np.zeros((8,6));nested=[]
    names=['ENSO','Other climate modes','Local ocean memory proxy','Local atmospheric heating','Local horizontal advection','Unresolved']
    for b,r in enumerate(ORDER):
        z=raw[raw.Region==r];dates=pd.to_datetime(z.iloc[:,0]);n=len(z)
        assert n==432
        nuis=np.column_stack([np.ones(n),np.arange(n),np.eye(12)[dates.dt.month.values-1][:,1:]])
        v=z[cols+['sst']].to_numpy(float);v-=nuis@np.linalg.lstsq(nuis,v,rcond=None)[0];v/=v.std(0)
        full,local,unres=old.owen_shares(v[:,:9],v[:,9]);weights[b]=np.r_[full[:2],local,unres]
        ref=assoc[assoc.Region==r].set_index('Group').loc[['ENSO','Other climate modes','Local processes','Unresolved']].Share.to_numpy()
        assert np.allclose(np.r_[full,unres],ref,atol=1e-12)
        assert np.all(weights[b]>=-1e-12) and abs(weights[b].sum()-1)<1e-12
        for k,name in enumerate(names):nested.append(dict(Region=r,Group=name,Share=weights[b,k]))
    weights[-1,-1]=1.
    pd.DataFrame(nested).to_csv(DATA/'SI_observed_local_process_hierarchy.csv',index=False)
    comparison=oldassoc.merge(assoc,on=['Region','Group'],suffixes=('_20','_235'))
    comparison['Share_change_pp']=100*(comparison.Share_235-comparison.Share_20)
    comparison.to_csv(DATA/'boundary_association_comparison.csv',index=False)
    checks['IO_association_max_change']=float(comparison[comparison.Region=='IO'].Share_change_pp.abs().max())
    assert checks['IO_association_max_change']<1e-10

    audit=module(AUDIT/'calculate_auc_partition.py','exact_pair_audit')
    g=masks(audit.LAT,audit.LON).values
    groups=np.where(g.any(0),g.argmax(0),7).astype('int8')
    assert np.array_equal(groups,audit.region_ids(23.5))
    checks['cached_auc_region_definition_matches']=True
    obs,threshold,domain,weight,sample_masks=audit.reference_data()
    et=pd.read_csv(DATA/'event_amplitudes_and_strict_AUC.csv')
    h=np.load(BASE/'source_data/event_regional_AUC.npz')['ols_weights']
    assert np.allclose(h,np.array([1,et.Peak_Nino34.iloc[-1]])@np.linalg.pinv(np.column_stack([np.ones(9),et.Peak_Nino34.iloc[:9]])),atol=1e-14)
    records=[];arrays={};sample_rows=[];max_auc=0.;max_recompute=0.
    for profile in ['Archived','Strict']:
        events=[]
        for event in et.Event:
            year=event[:4];folder='cache' if profile=='Strict' or year=='2002' else 'legacy_audit_cache'
            p=AUDIT/folder/(year+'_fields.npz');t=pd.read_csv(AUDIT/folder/(year+'_samples.csv'))
            with np.load(p) as z:
                e=z['endpoint235'].copy()
                # Independently recompute first/middle/last samples for every event,
                # rather than accepting cached regional allocations by label.
                f=z['forecast']
                for j in [0,54,107]:
                    row=t.iloc[j];target=row.Target_Month;month=int(target[-2:]);q=threshold[month-1];o=obs[target]
                    y=np.where(np.isfinite(o)&np.isfinite(q),o>q,np.nan).astype('float32')
                    w=weight.copy();w[~sample_masks.get((row.Initialization,target,int(row.Lead_Month)),domain)]=0
                    _,actual,_=audit.pair_partition(y[domain],f[j]-q[domain],w[domain],groups[domain])
                    max_recompute=max(max_recompute,float(np.max(np.abs(actual-e[j]))))
            assert e.shape==(108,8)
            max_auc=max(max_auc,float(np.max(np.abs(e.sum(1)-t.AUC.to_numpy()))))
            assert np.allclose(e.sum(1),t.AUC,atol=1e-12)
            events.append(e.mean(0));sample_rows.append(t.assign(Profile=profile))
            manifest.append(dict(source=str(p),sha256=digest(p),dataset='endpoint235',profile=profile))
        a=np.array(events);arrays[profile.lower()]=a
        expected=h@a[:9];observed=a[-1];loss=expected-observed
        for b,r in enumerate(REGIONS):records.append(dict(Profile=profile,Region=r,Expected=expected[b],Observed=observed[b],Deficit=loss[b]))
    assert max_recompute<1e-12
    checks.update(max_global_auc_reproduction_error=max_auc,max_independent_regional_recalculation_error=max_recompute,recomputed_samples=60)
    region=pd.DataFrame(records);region.to_csv(DATA/'regional_AUC_accounting.csv',index=False)
    np.savez_compressed(DATA/'event_regional_AUC.npz',**arrays,ols_weights=h)
    pd.concat(sample_rows,ignore_index=True).to_csv(DATA/'sample_AUC_profiles.csv',index=False)
    flow=[]
    for profile in ['Archived','Strict']:
        loss=region[region.Profile==profile].Deficit.to_numpy()
        for b,r in enumerate(REGIONS):
            for k,g in enumerate(PARTS):flow.append(dict(Profile=profile,Region=r,Component=g,Association_weight=weights[b,k],Regional_deficit=loss[b],Allocated_deficit=loss[b]*weights[b,k]))
    pd.DataFrame(flow).to_csv(DATA/'association_weighted_AUC_allocation.csv',index=False)
    oldreg=pd.read_csv(BASE/'source_data/regional_AUC_accounting.csv',keep_default_na=False)
    oldreg.merge(region,on=['Profile','Region'],suffixes=('_20','_235')).to_csv(DATA/'boundary_auc_comparison.csv',index=False)
    z=region[region.Profile=='Archived'];loss=z.Deficit.to_numpy();a=loss[:,None]*weights
    stats=json.loads((BASE/'statistics.json').read_text())
    assert abs(loss.sum()-stats['deficit'])<1e-12
    stats.update(primary_boundary=23.5,deficit=float(loss.sum()),expected=float(z.Expected.sum()),observed=float(z.Observed.sum()),
        gross_loss=float(np.maximum(loss,0).sum()),offset=float(-np.minimum(loss,0).sum()),
        parent_totals=dict(zip(['ENSO','Other','Local process','Unallocated'],[a[:,0].sum(),a[:,1].sum(),a[:,2:5].sum(),a[:,5].sum()])),
        local_children=dict(zip(['Local ocean memory','Local atmospheric drivers','Local oceanic drivers'],a[:,2:5].sum(0).tolist())),
        exact_parent_agreement=float(np.max(np.abs(weights.sum(1)-1))),closure_error=float(abs(a.sum()-stats['deficit'])),
        strict_input_sensitivity=float(region[region.Profile=='Strict'].Deficit.sum()),
        revision='Latitude boundary alignment with Fig3; all regional associations and archived AUC accounting reaggregated',
        fig3_reference=str(FIG3),previous_version=str(BASE))
    assert stats['closure_error']<1e-12
    record(ROOT/'statistics.json',stats)
    contract=json.loads((ROOT/'contract.json').read_text())
    contract.update(latitude_boundary_degrees=23.5,latitude_scope='Pacific and Atlantic T/N/S; Indian Ocean unchanged',
        fig3_reference=str(FIG3),revision_authority='Align Figure4 T/N/S boundaries with Figure3; retain all other definitions and display settings')
    record(ROOT/'contract.json',contract)
    checks.update(global_deficit_preserved=True,regression_weights_preserved=True,new_boundary=23.5,old_boundary=20,
        association_samples_per_region=432,figure4a_activity_unchanged=True,old_figures_overwritten=False)
    record(ROOT/'provenance/numerical_validation.json',checks)
    for p in [FIG3,BASE/'statistics.json',BASE/'source_data/event_regional_AUC.npz',ROOT/'cache/woa98_basin.nc']:
        manifest.append(dict(source=str(p),sha256=digest(p)))
    record(ROOT/'input_manifest.json',manifest)
    print(json.dumps(stats,indent=2));print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
