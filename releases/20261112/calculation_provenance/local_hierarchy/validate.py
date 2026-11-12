"""Numerical/visual checks are distinct from unresolved source validity."""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np,pandas as pd,xarray as xr,fitz

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'source_data'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    checks=[]
    def check(name,condition,detail=None):
        checks.append(dict(name=name,passed=bool(condition),detail=detail))
        assert condition,(name,detail)
    s=json.loads((ROOT/'statistics.json').read_text())
    reg=pd.read_csv(DATA/'regional_AUC_accounting.csv',keep_default_na=False)
    t=pd.read_csv(DATA/'association_weighted_AUC_allocation.csv',keep_default_na=False)
    old=json.loads((DATA/'Figure1_archived_statistics.json').read_text())
    for profile in ['Archived','Strict']:
        a=t[t.Profile==profile];r=reg[reg.Profile==profile].set_index('Region')
        check(profile+'_eight_regions',len(r)==8)
        check(profile+'_six_components',a.groupby('Region').size().eq(6).all())
        check(profile+'_fixed_weights_close',np.allclose(a.groupby('Region').Association_weight.sum(),1,atol=1e-12))
        check(profile+'_regional_allocation_closes',np.allclose(a.groupby('Region').Allocated_deficit.sum().sort_index(),r.Deficit.sort_index(),atol=1e-12))
        check(profile+'_defined_weighting',np.allclose(a.Allocated_deficit,a.Association_weight*a.Regional_deficit,atol=1e-12))
        check(profile+'_negative_offsets_kept',(a.Allocated_deficit<0).any())
    a=t[t.Profile=='Archived']
    check('archived_endpoint_not_rescaled',abs(a.Allocated_deficit.sum()-old['absolute_residual'])<1e-12)
    check('local_children_equal_parent',abs(sum(s['local_children'].values())-s['parent_totals']['Local process'])<1e-12)
    check('all_parents_close',abs(sum(s['parent_totals'].values())-s['deficit'])<1e-12)
    check('gross_minus_offsets_close',abs(s['gross_loss']-s['offset']-s['deficit'])<1e-12)
    check('positive_ribbon_geometry_not_positive_loss_claim',s['offset']>0)
    aa=xr.load_dataset(DATA/'Figure4a_activity_valid_extent.nc').activity_native_cell_equivalent
    cov=json.loads((ROOT/'coverage.json').read_text())
    check('activity_finite_cells',int(np.isfinite(aa).sum())==cov['finite_cells'])
    check('activity_zeros_retained',int((aa==0).sum())==cov['zero_cells_retained'])
    check('activity_valid_latitude_extent',float(aa.lat.min())==cov['lat_min'] and float(aa.lat.max())==cov['lat_max'])
    md=xr.load_dataset(DATA/'seven_ocean_regions.nc');m=next(iter(md.data_vars.values()))
    check('seven_nonoverlapping_basins',m.sizes['region']==7 and m.sum('region').max().item()==1)
    assoc=pd.read_csv(DATA/'Figure4_observed_group_associations.csv',keep_default_na=False)
    check('panel_a_associations_close',np.allclose(assoc.groupby('Region').Share.sum(),1))
    check('panel_a_n_months',assoc.N.eq(432).all())
    # Frozen nested allocation algorithm: efficiency and within-parent closure.
    spec=importlib.util.spec_from_file_location('nested',ROOT/'reference_code/prepare_v8.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rng=np.random.default_rng(20261104);x=rng.normal(size=(240,9))
    for k in [0,2,5,7,8]:
        y=x[:,k]+.2*rng.normal(size=240);full,local,u=mod.owen_shares(x,y)
        check('nested_total_'+str(k),abs(sum(full)+u-1)<1e-12)
        check('nested_local_'+str(k),abs(sum(local)-full[2])<1e-12)
    graphics=json.loads((ROOT/'render_validation.json').read_text())['Figure4']
    check('no_text_outside_canvas',not graphics['outside_canvas_text'])
    check('font_floor',graphics['minimum_rendered_text_pt']>=6)
    with fitz.open(ROOT/'figures/Figure4.pdf') as d:
        p=d[0];text=p.get_text();check('single_page',len(d)==1)
        check('183mm_width',abs(p.rect.width/72*25.4-183)<.01)
        check('height_under_247mm',p.rect.height/72*25.4<=247)
        check('vector_paths_retained',len(p.get_drawings())>100)
        for label in ['Local ocean memory','Local atmospheric drivers','Local oceanic drivers','0.109','Review copy']:
            check('label_'+label,label in text)
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    for e in manifest:
        if 'copy' in e:check('copied_input_'+e['copy'],sha(ROOT/e['copy'])==e['sha256'])
    root=Path('/data3/luoq/FEIS-archive/legacy/p03-ocean-sst-ensemble/data/GFDL-SPEAR')
    audit=dict(status='not repeated: original raw files unavailable')
    if (root/'sst_202309.nc').exists():
        with xr.open_dataset(root/'sst_202309.nc',decode_times=False) as d1,xr.open_dataset(root/'sst_202312.nc',decode_times=False) as d2:
            audit=dict(status='repeated live',S_202309=d1.S.values.tolist(),S_202312=d2.S.values.tolist(),
                       intended_S_202309=764,intended_S_202312=767,
                       arrays_identical=bool(np.array_equal(d1.sst.values,d2.sst.values,equal_nan=True)),
                       source_202309=str(root/'sst_202309.nc'),source_202312=str(root/'sst_202312.nc'),
                       hash_202309=sha(root/'sst_202309.nc'),hash_202312=sha(root/'sst_202312.nc'))
        check('provenance_problem_remains_visible',audit['arrays_identical'] and audit['S_202309']==[731.0])
    (ROOT/'coordinate_recheck.json').write_text(json.dumps(audit,indent=2)+'\n')
    result=dict(technical_checks_passed=sum(c['passed'] for c in checks),checks=checks,
                scientific_status='REVIEW ONLY; archived input validation unresolved',
                causal_attribution_validated=False,process_forecast_error_contributions_estimated=False,
                weighting_convention='Historical regional SST R2 weights times regional AUC accounting amounts',
                submission_ready=False)
    (ROOT/'QA.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
