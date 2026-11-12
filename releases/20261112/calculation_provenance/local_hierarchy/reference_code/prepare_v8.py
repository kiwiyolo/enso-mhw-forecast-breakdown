"""Preserve inputs, audit coverage and decompose observed associations only.

No CFSv2 value is rescaled into the multimodel Figure 1 deficit.
"""
from pathlib import Path
import hashlib, itertools, json, shutil
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent
PAPER = Path('/data3/luoq/FEIS/research/papers/mhw-predictability')
BASE = PAPER/'Figures/Design_20261029_v7'
REF = PAPER/'Figures/Design_20261027_v5'
OUT = ROOT/'source_data'
ORDER = ['TP','NP','SP','TA','NA','SA','IO']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def r2(x, y, ids):
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.var(y)<=1e-15:
        raise ValueError('Finite predictors and a nonconstant target are required')
    a = np.column_stack([np.ones(len(y)), x[:, ids]])
    residual = y-a@np.linalg.lstsq(a,y,rcond=None)[0]
    return float(1-residual@residual/np.sum((y-y.mean())**2))

def owen_shares(x,y):
    # Outer coalitions: ENSO, other modes, local. Within local: memory,
    # atmospheric heating, ocean advection. Whole-group order is averaged.
    groups = [[0],[1,2,3,4],[5,6,7,8]]
    children = [[5,6],[7],[8]]
    cache = {():0.}
    def score(ids):
        key=tuple(sorted(ids))
        if key not in cache: cache[key]=r2(x,y,list(key))
        return cache[key]
    full=np.zeros(3);local=np.zeros(3)
    for sequence in itertools.permutations(range(3)):
        before=[]
        for g in sequence:
            full[g] += (score(before+groups[g])-score(before))/6
            if g==2:
                for inner in itertools.permutations(range(3)):
                    subset=before.copy()
                    for j in inner:
                        local[j]+=(score(subset+children[j])-score(subset))/36
                        subset+=children[j]
            before+=groups[g]
    assert np.isclose(sum(local),full[2],atol=1e-10)
    return full,local,1-score(list(range(9)))

def main():
    OUT.mkdir(exist_ok=True)
    files=['Figure1_statistics.json','Figure4_observed_group_associations.csv',
           'Figure4_observed_association_samples.csv','Figure4_CFSv2_regional_process_series.csv',
           'Figure4_ERSST_regional_state.nc','Figure4_CFSv2_Sankey_flows.csv',
           'Figure4_CFSv2_event_diagnostics.csv','SI_CFSv2_validation_summary.csv',
           'seven_ocean_regions.nc','SI_Figure4_association_year_holdout.csv']
    lineage=[]
    for name in files:
        src=BASE/'source_data'/name
        shutil.copy2(src,OUT/name)
        lineage.append(dict(source=str(src.resolve()),copy='source_data/'+name,sha256=sha(src)))
    src=REF/'source_data/Figure3_activity_display_grid.nc'
    shutil.copy2(src,OUT/'activity_display_grid.nc')
    lineage.append(dict(source=str(src),copy='source_data/activity_display_grid.nc',sha256=sha(src)))
    (ROOT/'input_manifest.json').write_text(json.dumps(lineage,indent=2)+'\n')
    d=xr.load_dataset(OUT/'activity_display_grid.nc').sortby('lat')
    a=d.activity_native_cell_equivalent
    finite=np.isfinite(a)
    validlat=a.lat.where(finite.any('lon'),drop=True)
    cropped=d.sel(lat=slice(float(validlat.min()),float(validlat.max())))
    # Preserve zeros; discard no valid cell and never fill missing ocean cells.
    assert int(np.isfinite(cropped.activity_native_cell_equivalent).sum())==int(finite.sum())
    cropped.to_netcdf(OUT/'Figure4a_activity_valid_extent.nc')
    coverage=dict(lat_min=float(validlat.min()),lat_max=float(validlat.max()),
                  finite_cells=int(finite.sum()),finite_cells_dropped=0,
                  zero_cells_retained=int((a==0).sum()),
                  total_cells=int(a.size),source_grid_degrees=1,
                  interpolation_or_smoothing_added=False,
                  inherited_display_grid=True)
    (ROOT/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    samples=pd.read_csv(OUT/'Figure4_observed_association_samples.csv',keep_default_na=False)
    shares=pd.read_csv(OUT/'Figure4_observed_group_associations.csv',keep_default_na=False)
    rows=[]
    cols=['ENSO','IOD','PDO','NAO','SAM','sst_memory','ml_memory','surface_heating','horizontal_advection']
    for region in ORDER:
        z=samples[samples.Region==region].copy()
        dates=pd.to_datetime(z.iloc[:,0])
        nuisance=np.column_stack([np.ones(len(z)),np.arange(len(z)),np.eye(12)[dates.dt.month.values-1][:,1:]])
        values=z[cols+['sst']].to_numpy(float)
        residual=values-nuisance@np.linalg.lstsq(nuisance,values,rcond=None)[0]
        residual/=residual.std(axis=0)
        full,local,unres=owen_shares(residual[:,:9],residual[:,9])
        old=shares[shares.Region==region].set_index('Group')
        for j,g in enumerate(['ENSO','Other climate modes','Local processes']):
            assert np.isclose(full[j],old.loc[g,'Share'],atol=1e-9),(region,g)
        for group,value in zip(['ENSO','Other climate modes','Local ocean memory proxy',
                                'Local atmospheric heating','Local horizontal advection','Unresolved'],
                               [full[0],full[1],*local,unres]):
            rows.append(dict(Region=region,Group=group,Share=float(value),N=len(z),
                             start=str(dates.min().date()),end=str(dates.max().date())))
    pd.DataFrame(rows).to_csv(OUT/'SI_observed_local_process_hierarchy.csv',index=False)
    assert np.allclose(pd.DataFrame(rows).groupby('Region').Share.sum(),1)
    flows=pd.read_csv(OUT/'Figure4_CFSv2_Sankey_flows.csv',keep_default_na=False)
    terms=['Tropical source','Circulation','Local ocean state','Local atmospheric forcing','Local ocean advection','Unresolved']
    closure=float(np.max(abs(flows[terms].sum(axis=1)-flows['Observed loss'])))
    assert closure<1e-10
    stats=json.loads((OUT/'Figure1_statistics.json').read_text())
    perf=pd.read_csv(OUT/'SI_CFSv2_validation_summary.csv').set_index('Model')
    audit=dict(figure1_deficit=stats['absolute_residual'],cfsv2_accounting_closure=closure,
               full_holdout_RMSE=float(perf.loc['Full','RMSE']),
               baseline_holdout_RMSE=float(perf.loc['Without All process diagnostics','RMSE']),
               cfsv2_validation_passed=bool(perf.loc['Full','RMSE']<perf.loc['Without All process diagnostics','RMSE']),
               figure1_score='ensemble-mean SST minus monthly MHW threshold',
               cfsv2_score='fraction of four forecast members exceeding monthly MHW threshold',
               same_forecast_system=False,same_baseline=False,same_AUC_score=False,
               forecast_ocean_memory_measured=False,
               complete_multimodel_process_inputs_available=False,
               fig1_process_sankey_authorized_for_submission=False)
    (ROOT/'attribution_gate.json').write_text(json.dumps(audit,indent=2)+'\n')
    protected=[BASE/'figures/Figure4.pdf',BASE/'figures/Figure4.png',REF/'figures/Figures_2-3.pdf',BASE/'source_data/Figure1_statistics.json']
    (ROOT/'original_hashes.json').write_text(json.dumps({str(p):sha(p) for p in protected},indent=2)+'\n')
    print(json.dumps({'coverage':coverage,'attribution_gate':audit},indent=2))

if __name__=='__main__': main()
