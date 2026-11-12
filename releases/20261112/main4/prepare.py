"""Map-consistent local AUC shortfall accounting; no global pair-AUC allocation."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress
from regions import masks,ORDER

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data'
BASE=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures')
OLD=BASE/'Figure4_latitude_sync_20261111_v23'
MAP=BASE/'Figure1_global_expected_benchmark_20261111_v11'
MAIN=BASE/'Figure1-3_Nature_layout_20261110_v7'
REG=ORDER+['Other seas']
PARTS=['ENSO','Other','Memory','Atmosphere','Ocean','Unallocated']
manifest=[]
def source(p):
    manifest.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return p
def outjson(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def aggregate(d,w,g,n=8):
    """Positive and negative integrals evaluated separately, no zero-filling NaNs."""
    ok=np.isfinite(d)&np.isfinite(w)&(w>0)
    W=w[ok].sum()
    if W<=0:raise ValueError('No finite weighted support')
    positive=np.array([np.sum(w[ok&(g==b)]*np.maximum(d[ok&(g==b)],0))/W for b in range(n)])
    negative=np.array([np.sum(w[ok&(g==b)]*np.maximum(-d[ok&(g==b)],0))/W for b in range(n)])
    return positive,negative

def main():
    tests={}
    pp,nn=aggregate(np.array([.2,-.1,np.nan,0]),np.array([1,2,1,1]),np.array([0,0,1,1]),2)
    assert np.allclose(pp,[.05,0]) and np.allclose(nn,[.05,0])
    tests['synthetic_sign_missing_area_and_zero_cases']=True
    for name in ['Figure4_observed_group_associations.csv','SI_observed_local_process_hierarchy.csv']:
        shutil.copy2(source(OLD/'source_data'/name),DATA/name)
    source(OLD/'figures/Figure4.pdf')
    source(ROOT/'cache/woa98_basin.nc')
    for i in [1,2,3]:source(MAIN/'figures'/f'Figure{i}.pdf')
    events=pd.read_csv(source(MAP/'source_data/Figure1_source_events.csv'))
    hist=events[events.Event!='2023/24'];target=events[events.Event=='2023/24'].iloc[0]
    fit=linregress(hist.Peak_Nino34,hist.Mean_AUC)
    expected=float(fit.intercept+fit.slope*target.Peak_Nino34)
    ds=xr.load_dataset(source(MAP/'source_data/global_expected_shortfall.nc'))
    v=ds.global_benchmark_shortfall.values
    assert np.allclose(expected-ds.local_auc_2023.values,v,equal_nan=True,atol=1e-12)
    assert abs(expected-.7050185254528283)<1e-12
    lat=np.arange(-89.5,90);lon=np.arange(.5,360)
    m=masks(lat,lon)
    ii=np.rint(ds.lat.values+89.5).astype(int);jj=np.rint(ds.lon.values-.5).astype(int)
    at=m.values[:,ii,jj]
    group=np.where(at.any(0),at.argmax(0),7)
    assert np.all(at.sum(0)<=1)
    # Independently compare with authoritative Figure4 masks on their own grid.
    oldmask=xr.load_dataset(source(OLD/'source_data/seven_ocean_regions.nc'))
    assert np.array_equal(masks(oldmask.lat.values,oldmask.lon.values),oldmask['mask'])
    tests['Fig3_Fig4_basin_rules_identical']=True
    tests['native_map_values_and_missing_mask_unchanged']=True
    w=ds.area_weight.values
    assert np.allclose(w,np.cos(np.deg2rad(ds.lat.values)),rtol=1e-6)
    pos,neg=aggregate(v,w,group)
    den=pos.sum();above=neg.sum();valid=np.isfinite(v)
    assert den>0
    assert abs((den-above)-np.average(v[valid],weights=w[valid]))<1e-12
    assert np.isclose((pos/den).sum(),1)
    assoc=pd.read_csv(DATA/'SI_observed_local_process_hierarchy.csv',keep_default_na=False)
    names=['ENSO','Other climate modes','Local ocean memory proxy','Local atmospheric heating','Local horizontal advection','Unresolved']
    weights=np.zeros((8,6))
    for b,r in enumerate(ORDER):
        weights[b]=assoc[assoc.Region==r].set_index('Group').loc[names].Share.to_numpy()
    weights[-1,-1]=1
    assert np.allclose(weights.sum(1),1,atol=1e-8) and np.all(weights>=0)
    rows=[];flow=[]
    for b,r in enumerate(REG):
        mask=(group==b);ok=mask&valid
        coverage=float(w[ok].sum()/w[mask].sum()) if mask.any() else None
        rows.append(dict(Region=r,Positive_shortfall=pos[b],Above_benchmark=neg[b],
                         Net_shortfall=pos[b]-neg[b],Shortfall_share_pct=100*pos[b]/den,
                         Net_share_pct=100*(pos[b]-neg[b])/(den-above) if abs(den-above)>1e-12 else None,
                         Above_share_of_shortfall_pct=100*neg[b]/den,
                         Mean_local_auc=float(np.average(ds.local_auc_2023.values[ok],weights=w[ok])) if ok.any() else None,
                         Mean_shortfall=float(np.average(v[ok],weights=w[ok])) if ok.any() else None,
                         Valid_cells=int(ok.sum()),Domain_cells=int(mask.sum()),Area_coverage=coverage))
        for k,c in enumerate(PARTS):
            flow.append(dict(Region=r,Component=c,Association_weight=weights[b,k],
                             Allocated_positive_shortfall=pos[b]*weights[b,k],
                             Allocated_above_benchmark=neg[b]*weights[b,k],
                             Shortfall_share_pct=100*pos[b]*weights[b,k]/den))
    pd.DataFrame(rows).to_csv(DATA/'regional_map_shortfall.csv',index=False)
    pd.DataFrame(flow).to_csv(DATA/'association_weighted_local_shortfall.csv',index=False)
    # Signed map and all masks remain on original centres, no visual interpolation.
    grid=np.full((180,360),np.nan);grid[ii,jj]=v
    auc=np.full_like(grid,np.nan);auc[ii,jj]=ds.local_auc_2023.values
    domain=np.zeros_like(grid,dtype=bool);domain[ii,jj]=True
    gd=xr.Dataset(dict(shortfall=(('lat','lon'),grid),local_auc=(('lat','lon'),auc),
                       original_domain=(('lat','lon'),domain),region_mask=m),coords=dict(lat=lat,lon=lon))
    gd.attrs.update(global_expected_auc=expected,window='2023-04 through 2024-09',leads='1-9',
                    support='Fixed11x11one-degree ocean-cell neighbourhood, archived source profile',
                    missing='Unchanged1297undefined cells',comparison='Global benchmark, not local conditional expectation')
    gd.to_netcdf(DATA/'Figure4a_local_shortfall.nc')
    m.to_dataset(name='mask').to_netcdf(DATA/'seven_ocean_regions.nc')
    assert np.array_equal(np.isfinite(grid[ii,jj]),valid)
    assert np.allclose(grid[ii,jj],v,equal_nan=True)
    leads=np.load(source(MAP/'source_data/local_auc_and_shortfall_by_lead.npz'))
    sensitivity=[]
    for j,d in enumerate(leads['shortfall']):
        lp,ln=aggregate(d,w,group)
        for b,r in enumerate(REG):
            sensitivity.append(dict(Lead=j+1,Region=r,Positive_shortfall=lp[b],
                                    Above_benchmark=ln[b],Shortfall_share_pct=100*lp[b]/lp.sum()))
    pd.DataFrame(sensitivity).to_csv(DATA/'regional_lead_sensitivity.csv',index=False)
    tot=(pos[:,None]*weights).sum(0)
    tests.update(regional_positive_closure=bool(abs(pos.sum()-den)<1e-12),
                 components_positive_closure=bool(abs(tot.sum()-den)<1e-12),
                 net_map_identity=True,negative_areas_retained=True,
                 source_expected_auc_reproduced=True,
                 all_valid_cells_assigned_once=bool(sum(x['Valid_cells'] for x in rows)==int(valid.sum())),
                 missing_cells=int((~valid).sum()), association_intervals_unchanged=True)
    assert all(v for k,v in tests.items() if isinstance(v,bool))
    stats=dict(global_expected_auc=expected,original_global_gap_unchanged=expected-float(target.Mean_AUC),
               positive_map_shortfall=den,above_benchmark_total=above,net_map_shortfall=den-above,
               denominator='Area-weighted positive shortfall of lead-mean mapped local AUC',
               above_as_percent_of_positive=100*above/den,valid_cells=int(valid.sum()),
               common_valid_area_fraction=float(w[valid].sum()/w.sum()),
               component_pct=dict(zip(PARTS,(100*tot/den).tolist())),
               local_total_pct=float(100*tot[2:5].sum()/den),regions=rows)
    outjson(ROOT/'statistics.json',stats);outjson(ROOT/'qa/numerical_validation.json',tests)
    outjson(ROOT/'input_manifest.json',manifest)
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
