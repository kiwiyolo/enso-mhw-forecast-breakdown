"""Author-requested archived-input REVIEW figure, not validated attribution.

First allocate archived AUC exactly to pair endpoints. Then apply a declared,
fixed regional association-weighting convention. The second step is bookkeeping,
not an estimate of physical process effects or a test of local dominance.
"""
from pathlib import Path
import hashlib,importlib.util,itertools,json,shutil
import numpy as np,pandas as pd,xarray as xr

ROOT=Path(__file__).resolve().parent
WORK=Path('/data3/luoq/FEIS-archive/legacy/p03-ocean-sst-ensemble')
PAPER=Path('/data3/luoq/FEIS/research/papers/mhw-predictability')
BASE=PAPER/'Figures/Figure4_region_layout_20261104_v8'
AUDIT=WORK/'figure4_auc_attribution_20261104_v9'
DATA=ROOT/'source_data';RC=ROOT/'reference_code'
ORDER=['TP','NP','SP','TA','NA','SA','IO','Other seas']
PARTS=['ENSO','Other','Memory','Atmosphere','Ocean','Unallocated']
LEAF_TO_PARENT=np.array([0,1,2,2,2,3])

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def loadmodule(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
    DATA.mkdir(exist_ok=True);RC.mkdir(exist_ok=True);manifest=[]
    copies={BASE/'source_data'/n:DATA/n for n in ['Figure4a_activity_valid_extent.nc','seven_ocean_regions.nc','Figure4_observed_group_associations.csv','Figure4_observed_association_samples.csv','SI_observed_local_process_hierarchy.csv']}
    copies.update({BASE/'coverage.json':ROOT/'coverage.json',BASE/'figure_style.py':ROOT/'figure_style.py',BASE/'prepare_revision.py':RC/'prepare_v8.py',
                  AUDIT/'source_audit.json':ROOT/'source_audit.json',AUDIT/'SOURCE_AUDIT.md':ROOT/'SOURCE_AUDIT.md',
                  AUDIT/'source_data/AUC_deficit_reconciliation.csv':DATA/'AUC_deficit_reconciliation.csv',
                  AUDIT/'source_data/wrong_initialization_sources.csv':DATA/'wrong_initialization_sources.csv',
                  AUDIT/'source_data/event_AUC_reproduced.csv':DATA/'event_amplitudes_and_strict_AUC.csv',
                  PAPER/'Figures/Design_20261029_v7/source_data/Figure1_statistics.json':DATA/'Figure1_archived_statistics.json'})
    for s,d in copies.items():
        shutil.copy2(s,d);manifest.append(dict(source=str(s),copy=str(d.relative_to(ROOT)),sha256=digest(s)))
    old=loadmodule(RC/'prepare_v8.py','owen_definition')
    raw=pd.read_csv(DATA/'Figure4_observed_association_samples.csv',keep_default_na=False)
    archived=pd.read_csv(DATA/'SI_observed_local_process_hierarchy.csv',keep_default_na=False)
    names=['ENSO','Other climate modes','Local ocean memory proxy','Local atmospheric heating','Local horizontal advection','Unresolved']
    cols=['ENSO','IOD','PDO','NAO','SAM','sst_memory','ml_memory','surface_heating','horizontal_advection']
    weights=np.zeros((8,6));errors=[]
    for b,r in enumerate(ORDER[:7]):
        z=raw[raw.Region==r];dates=pd.to_datetime(z.iloc[:,0]);n=len(z)
        nuis=np.column_stack([np.ones(n),np.arange(n),np.eye(12)[dates.dt.month.values-1][:,1:]])
        v=z[cols+['sst']].to_numpy(float);v-=nuis@np.linalg.lstsq(nuis,v,rcond=None)[0];v/=v.std(0)
        full,local,unres=old.owen_shares(v[:,:9],v[:,9]);q=np.r_[full[:2],local,unres]
        ref=archived[archived.Region==r].set_index('Group').loc[names].Share.to_numpy()
        errors.append(float(np.max(abs(q-ref))));assert np.allclose(q,ref,atol=1e-12)
        weights[b]=q
        assert len(z)==432 and q.min()>-1e-12 and abs(q.sum()-1)<1e-12
    weights[-1,-1]=1. # No association fit for enclosed/unassigned seas.
    et=pd.read_csv(DATA/'event_amplitudes_and_strict_AUC.csv')
    h=np.array([1,et.Peak_Nino34.iloc[-1]])@np.linalg.pinv(np.column_stack([np.ones(9),et.Peak_Nino34.iloc[:9]]))
    rows=[];event_arrays={};all_samples=[]
    for profile in ['Archived','Strict']:
        events=[]
        for event in et.Event:
            year=event[:4];folder='cache' if profile=='Strict' or year=='2002' else 'legacy_audit_cache'
            p=AUDIT/folder/(year+'_fields.npz');t=pd.read_csv(AUDIT/folder/(year+'_samples.csv'))
            with np.load(p) as z:pair=z['pair_auc'].copy()
            A=pair.mean(0);events.append(.5*(A.sum(0)+A.sum(1)))
            assert len(t)==108 and np.allclose(pair.sum((1,2)),t.AUC,atol=1e-12)
            all_samples.append(t.assign(Profile=profile))
            manifest.append(dict(source=str(p),sha256=digest(p),role=profile+' regional pair data'))
            dest=RC/(profile+'_'+year+'_sources.json');s=AUDIT/folder/(year+'_sources.json');shutil.copy2(s,dest)
            manifest.append(dict(source=str(s),copy=str(dest.relative_to(ROOT)),sha256=digest(s)))
        A=np.array(events);event_arrays[profile]=A
        expected=h@A[:9];observed=A[-1];loss=expected-observed
        for b,r in enumerate(ORDER):rows.append(dict(Profile=profile,Region=r,Expected=expected[b],Observed=observed[b],Deficit=loss[b]))
    region=pd.DataFrame(rows);region.to_csv(DATA/'regional_AUC_accounting.csv',index=False)
    pd.concat(all_samples,ignore_index=True).to_csv(DATA/'sample_AUC_profiles.csv',index=False)
    np.savez_compressed(DATA/'event_regional_AUC.npz',archived=event_arrays['Archived'],strict=event_arrays['Strict'],ols_weights=h)
    expected_deficit=json.loads((DATA/'Figure1_archived_statistics.json').read_text())['absolute_residual']
    loss=region[region.Profile=='Archived'].Deficit.to_numpy()
    assert abs(loss.sum()-expected_deficit)<1e-12
    records=[]
    for profile in ['Archived','Strict']:
        L=region[region.Profile==profile].Deficit.to_numpy()
        for b,r in enumerate(ORDER):
            for k,g in enumerate(PARTS):records.append(dict(Profile=profile,Region=r,Component=g,Association_weight=weights[b,k],Regional_deficit=L[b],Allocated_deficit=L[b]*weights[b,k]))
    flow=pd.DataFrame(records);flow.to_csv(DATA/'association_weighted_AUC_allocation.csv',index=False)
    primary=loss[:,None]*weights;groups=np.zeros((8,4))
    for k,g in enumerate(LEAF_TO_PARENT):groups[:,g]+=primary[:,k]
    for p in ['Archived','Strict']:
        z=flow[flow.Profile==p];assert abs(z.Allocated_deficit.sum()-region[region.Profile==p].Deficit.sum())<1e-12
    assert np.allclose(primary[:,2:5].sum(1),groups[:,2])
    stats=dict(status='REVIEW ONLY: archived input validation unresolved; not submission-cleared',
       endpoint='Archived Figure1 expected-minus-observed global AUC; retained at author request',
       method='Exact AUC regional pair accounting, followed by explicit historical association-weighted diagnostic bookkeeping',
       limitation='Regional component amounts are allocation by fixed SST association weights, not measured process-specific forecast skill losses. Large local allocations are not independent evidence of local causation.',
       deficit=float(loss.sum()),expected=float(region[region.Profile=='Archived'].Expected.sum()),observed=float(region[region.Profile=='Archived'].Observed.sum()),
       gross_loss=float(np.maximum(loss,0).sum()),offset=float(-np.minimum(loss,0).sum()),
       parent_totals=dict(zip(['ENSO','Other','Local process','Unallocated'],groups.sum(0).tolist())),
       local_children=dict(zip(['Local ocean memory','Local atmospheric drivers','Local oceanic drivers'],primary[:,2:5].sum(0).tolist())),
       primary_boundary=20,association_samples_per_region=432,association_fit_period='1985-2020',
       exact_parent_agreement=max(errors),closure_error=float(abs(primary.sum()-expected_deficit)),
       strict_input_sensitivity=float(region[region.Profile=='Strict'].Deficit.sum()),
       no_independent_process_attribution=True,raw_data_modified=False,main_documents_modified=False)
    record(ROOT/'statistics.json',stats);record(ROOT/'input_manifest.json',manifest)
    record(ROOT/'contract.json',dict(status='exploratory_review',archetype='asymmetric_mixed',
       authority='User requests original Figure1 endpoint and full Figure4 with nested Local processes',
       panels=dict(a='Preserved monthly MHW activity and seven regional historical SST association bars',b='Association-weighted archived-AUC accounting; Local split using nested Owen shares'),
       prohibits=['Calling association weights causal AUC contributions','Claiming local forecast-error dominance from these weights','Removing negative regional offsets','Re-labelling advection as all ocean dynamics','Suppressing input-provenance discrepancy'],
       requested_endpoint=0.109,calculated_archived_endpoint=stats['deficit'],protected_originals=True))
    record(ROOT/'decision_log.json',[
      dict(status='active',decision='Complete layout and retain original Figure1 reporting basis as requested'),
      dict(status='exploratory',decision='Use nested historical SST associations for a declared weighting convention; this extends previous categorical-regime stratification and must not be confused with it'),
      dict(status='needs_verification',decision='Input audit unresolved; identical September and December 2023 raw GFDL SST and S=731 conflict with their intended start months'),
      dict(status='active',decision='Do not overwrite original manuscript, figures, or strict-input v9 result')])
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
