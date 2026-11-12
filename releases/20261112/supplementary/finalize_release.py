"""Additional field-level arithmetic, preservation, and delivery checks."""
from pathlib import Path
import json, hashlib, shutil, zipfile
import numpy as np
import pandas as pd
import xarray as xr
import fitz

ROOT=Path(__file__).resolve().parent
BASE=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
tests=[]
def check(n,v):
    tests.append(dict(test=n,passed=bool(v)))
    assert v,n

src=BASE/'Figure4_net_normalized_20261111_v27/source_data/Figure4a_local_shortfall.nc'
dst=ROOT/'source_data/CURRENT4'/src.name
if not dst.exists():shutil.copy2(src,dst)
check('Frozen map unchanged',sha(dst)==sha(src))
manifest=json.loads((ROOT/'source_manifest.json').read_text())
manifest.append(dict(source=str(src),frozen=str(dst.relative_to(ROOT)),sha256=sha(dst)))
ds=xr.load_dataset(dst)
d=ds.shortfall.values
valid=np.isfinite(d)
weight_src=BASE/'Figure1_global_expected_benchmark_20261111_v11/source_data/global_expected_shortfall.nc'
weight_dst=ROOT/'source_data/CURRENT4'/weight_src.name
if not weight_dst.exists():shutil.copy2(weight_src,weight_dst)
check('Frozen original weights unchanged',sha(weight_dst)==sha(weight_src))
manifest.append(dict(source=str(weight_src),frozen=str(weight_dst.relative_to(ROOT)),sha256=sha(weight_dst)))
save(ROOT/'source_manifest.json',list({m['frozen']:m for m in manifest}.values()))
wd=xr.load_dataset(weight_dst)
w=np.zeros_like(d)
ii=np.rint(wd.lat.values+89.5).astype(int);jj=np.rint(wd.lon.values-.5).astype(int)
w[ii,jj]=wd.area_weight.values
check('Stored weights equal cosine latitude within source precision',np.allclose(w[ii,jj],np.cos(np.deg2rad(wd.lat.values)),rtol=1e-6))
w[~valid]=0;w/=w.sum()
check('Map equals expected minus local AUC',np.allclose(d,ds.attrs['global_expected_auc']-ds.local_auc.values,equal_nan=True))
reg=pd.read_csv(ROOT/'derived_data/S15_main_Fig4_region_crosswalk.csv',keep_default_na=False)
masks=ds.region_mask.values.astype(bool)
check('Region masks mutually exclusive',masks.sum(axis=0).max()<=1)
rows=[]
for _,q in reg.iterrows():
    mask=(~masks.any(axis=0)) if q.Region=='Other seas' else ds.region_mask.sel(region=q.Region).values.astype(bool)
    positive=np.nansum(w*mask*np.maximum(d,0))
    negative=np.nansum(w*mask*np.maximum(-d,0))
    check('Field-to-table positive '+q.Region,np.isclose(positive,q.Positive_shortfall,atol=1e-12,rtol=0))
    check('Field-to-table negative '+q.Region,np.isclose(negative,q.Above_benchmark,atol=1e-12,rtol=0))
    rows.append(dict(Region=q.Region,positive_recomputed=positive,negative_recomputed=negative))
pd.DataFrame(rows).to_csv(ROOT/'derived_data/S15_field_level_recalculation.csv',index=False)
check('27508 finite map centres',valid.sum()==27508)

# Verify the preceding package against its own recorded, pre-revision hashes.
old=ROOT.parent/'Supplementary_Figures_20261111_v4'
old_manifest=json.loads((ROOT/'provenance/release_manifest_v4.json').read_text())
for m in old_manifest:
    p=old/m['path']
    check('Previous release preserved '+m['path'],p.exists() and sha(p)==m['sha256'])

# A layout revision must not silently alter any previously calculated results.
for p in sorted((old/'derived_data').glob('*.csv')):
    q=ROOT/'derived_data'/p.name
    check('Numerical table unchanged '+p.name,q.exists() and sha(q)==sha(p))
s1=pd.read_csv(ROOT/'derived_data/S01c_all_seven_regions.csv',keep_default_na=False)
lead=pd.read_csv(ROOT/'derived_data/S01c_all_seven_regions_by_lead.csv',keep_default_na=False)
check('S1c complete seven regions',s1.region.tolist()==['TP','NP','SP','TA','NA','SA','IO'])
check('S1c complete 63 lead points',len(lead)==63 and (lead.groupby('region').size()==9).all())
records=json.loads((ROOT/'figure_registry.json').read_text())
oldrecords=json.loads((old/'figure_registry.json').read_text())
geometry=[]
for r,o in zip(records,oldrecords):
    check('Reduced artwork height '+r['id'],r['height_mm']<o['height_mm'])
    geometry.append(dict(Figure=r['number'],Old_height_mm=o['height_mm'],New_height_mm=r['height_mm'],Reduction_percent=100*(1-r['height_mm']/o['height_mm'])))
pd.DataFrame(geometry).to_csv(ROOT/'qa/layout_comparison.csv',index=False)
with fitz.open(ROOT/'Supplementary_Figures_with_captions.pdf') as newdoc,fitz.open(old/'Supplementary_Figures_with_captions.pdf') as olddoc:
    check('All caption pages compacted',all(n.rect.height<o.rect.height for n,o in zip(newdoc,olddoc)))
    page_reduction=100*(1-sum(p.rect.height for p in newdoc)/sum(p.rect.height for p in olddoc))

with zipfile.ZipFile(ROOT/'Individual_SI_PDFs.zip') as z:
    check('34 individual PDFs',len(z.namelist())==34)
    check('Individual ZIP CRC',z.testzip() is None)
    for n in z.namelist():
        check('ZIP current '+n,hashlib.sha256(z.read(n)).hexdigest()==sha(ROOT/n))
with fitz.open(ROOT/'Supplementary_Information.pdf') as doc:
    text=' '.join(p.get_text() for p in doc)
    for term in ['Review package','Status:','source alignment unresolved','local SST information','TBD','TODO']:
        check('Reader-facing cleanup '+term,term not in text)

save(ROOT/'qa/final_field_and_preservation_checks.json',tests)
q=json.loads((ROOT/'qa/release_validation.json').read_text())
report=(
    '# Compact SI release QA\n\n'
    f'- {len(q["tests"])} numerical/geometry/PDF checks passed; {len(tests)} additional map, preservation and delivery checks passed.\n'
    '- 17 separate vector PDFs, 17 captioned PDFs, 22-page integrated SI.\n'
    '- Artwork width 180 mm; ordinary type at least 7.5 pt; captions 9.5 pt; fonts embedded.\n'
    f'- Caption booklet total page area reduced by {page_reduction:.1f}% at unchanged page width and font sizes.\n'
    '- S1c shows seven regions and all 63 lead points; original numerical tables unchanged.\n'
    '- Visual inspection: all 17 figures in contact sheets; full-size caption pages S1, S5, S7, S13 and S15; remaining pages checked for rendered text bounds.\n'
    '- S15 independently recalculated from the original Fig.4 map, including Other seas and separate positive/negative sums.\n'
    '- All previous-release manifest files remain unchanged. All 81 S5 raw rows preserved; 3 unverified wind lead-9 values displayed as NA.\n'
    '- Raw-data limits remain: SST initialization selection and tropical atmospheric window/cohort alignment. Scientific submission clearance is NOT asserted.\n'
    '- Main manuscript text/citation order was not changed. Current official journal guidance could not be verified online.\n')
(ROOT/'qa/QA_REPORT.md').write_text(report)
sk=json.loads((ROOT/'.paper-workflow/submission-checklist.yml').read_text())
sk.update(visual_inspection_passed=True,placeholder_scan_passed=True,scientific_submission_ready=False,
          blockers=['SST initialization coordinate selection','Tropical atmospheric common windows and model cohorts','Main-text citation integration'])
save(ROOT/'.paper-workflow/submission-checklist.yml',sk)
exclude={'release_manifest.json','__pycache__'}
entries=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(ROOT.rglob('*'))
         if p.is_file() and not any(x in exclude for x in p.relative_to(ROOT).parts)]
save(ROOT/'release_manifest.json',entries)
print(json.dumps(dict(extra_checks=len(tests),files_in_release=len(entries),scientific_submission_ready=False),indent=2))
