"""Validate numerical lineage, figure geometry and same-page PDF delivery."""
from pathlib import Path
import json, hashlib, shutil, csv
import fitz, numpy as np, pandas as pd, xarray as xr

ROOT=Path(__file__).resolve().parent
BASE=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures')
WF=ROOT/'.paper-workflow'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for z in iter(lambda:f.read(2**20),b''):h.update(z)
    return h.hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
tests=[]
def check(name,condition,detail=''):
    tests.append(dict(test=name,passed=bool(condition),detail=detail))
    if not condition:raise AssertionError((name,detail))

records=json.loads((ROOT/'figure_registry.json').read_text())
manifest=json.loads((ROOT/'source_manifest.json').read_text())
# The original metric definition confirms the 1985–2014 threshold for S1a,b.
src=BASE/'Revised_20260921/Supplementary/Supplementary_Figure1.md'
dst=ROOT/'source_data/OLD/Supplementary_Figure1.md'
if not dst.exists():shutil.copy2(src,dst)
manifest.append(dict(source=str(src),frozen=str(dst.relative_to(ROOT)),sha256=sha(dst)))
manifest=list({m['frozen']:m for m in manifest}.values())
dump(ROOT/'source_manifest.json',manifest)
for m in manifest:
    check('input integrity: '+m['frozen'],sha(ROOT/m['frozen'])==m['sha256']==sha(m['source']))
check('17 sequential figures',[r['number'] for r in records]==list(range(1,18)))
for r in records:
    check(f'S{r["number"]} text inside artwork',not r['outside_text'])
    check(f'S{r["number"]} font floor',r['min_font_pt']>=7.5)

build_checks=json.loads((ROOT/'qa/numerical_checks.json').read_text())
check('all renderer numerical checks passed',all(r['passed'] for r in build_checks),len(build_checks))
p=pd.read_csv(ROOT/'source_data/F2/pattern_correlations.csv')
check('S5 includes all 81 values',len(p)==81 and p.Pattern_correlation.notna().all())
shown=pd.read_csv(ROOT/'derived_data/S05_displayed_correlations.csv')
check('S5 excludes unverified wind label 9',shown.Pattern_correlation.notna().sum()==78 and shown.loc[(shown.Process=='Zonal wind stress')&(shown.Lead==9),'Pattern_correlation'].isna().all())
z=pd.read_csv(ROOT/'source_data/F3/SI_regional_skill_change.csv',keep_default_na=False,na_values=[''])
cols=['1997/98','2015/16','2023/24']
check('S7 correct event-difference definition',np.allclose(z.Delta_r,z[cols[2]]-(z[cols[0]]+z[cols[1]])/2,equal_nan=True))
check('S7 has 48 valid rows and 6 missing lead-9 rows',z.Delta_r.notna().sum()==48 and z[z.Lead==9].Delta_r.isna().all())
q=pd.read_csv(ROOT/'derived_data/S08_model_skill.csv')
check('S8 96 model-event-lead evaluations',len(q)==96 and (q.groupby(['event','model']).size()==8).all())
q=pd.read_csv(ROOT/'derived_data/S14_paired_errors.csv',keep_default_na=False)
check('S14 63 region-event-lead evaluations',len(q)==63)
ds=xr.load_dataset(ROOT/'derived_data/S14_paired_error_fields_235.nc')
check('S14 regional masks do not overlap',int(ds.region_mask.sum('region').max())<=1)
for region in ['TP','TA']:
    latmask=ds.region_mask.sel(region=region).any('lon')
    used=ds.lat.where(latmask,drop=True).values
    check('S14 tropical boundary '+region,used.min()>=-23.5 and used.max()<23.5)
for k in ['Z200','Qnet']:
    check('S14 ratio '+k,np.allclose(q[k+'_relative'],q[k+'_RMSE']/q[k+'_reference']))
reg=pd.read_csv(ROOT/'derived_data/S15_main_Fig4_region_crosswalk.csv',keep_default_na=False)
comp=pd.read_csv(ROOT/'derived_data/S15_signed_components.csv')
check('S15 signed regional closure',np.isclose(reg.Net_shortfall.sum(),.10651437039268086,rtol=0,atol=1e-12))
check('S15 signed component closure',np.isclose(comp.net.sum(),reg.Net_shortfall.sum(),rtol=0,atol=1e-12))
check('S15 percentage denominator',np.isclose(reg.Net_pct.sum(),100,atol=1e-10))
check('S15 positive and negative shares',np.allclose([reg.Positive_pct.sum(),reg.Negative_pct.sum()],[120.79056301494,-20.79056301494]))
check('S15 component signed percentages',np.allclose([comp.Positive_pct.sum(),comp.Negative_pct.sum()],[reg.Positive_pct.sum(),reg.Negative_pct.sum()]))
s2=pd.read_csv(ROOT/'derived_data/S02_slope_difference.csv').iloc[0]
check('S2 reports interval including zero',s2.low<0<s2.high)
check('S2 caption contains calculated interval','−0.011' in records[1]['caption'] or '-0.011' in records[1]['caption'])

# Same-page caption placement, embedded fonts and final-size text bounds.
pdf_details=[]
for path in sorted((ROOT/'figures').glob('*.pdf')):
    with fitz.open(path) as doc:
        p=doc[0];check(path.name+' width',abs(p.rect.width*25.4/72-180)<.01)
        spans=[s for b in p.get_text('dict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans'] if s['text'].strip()]
        fonts={f[0]:f for f in p.get_fonts(full=True)}
        embedded=all(bool(doc.extract_font(x)[3]) for x in fonts)
        check(path.name+' embedded fonts',embedded)
        check(path.name+' PDF text in page',all(fitz.Rect(s['bbox']).x0>=-1 and fitz.Rect(s['bbox']).y0>=-1 and fitz.Rect(s['bbox']).x1<=p.rect.width+1 and fitz.Rect(s['bbox']).y1<=p.rect.height+1 for s in spans))
        # Mathematical scripts can be smaller; standard typography is also checked in the renderer.
        minfont=min(s['size'] for s in spans)
        check(path.name+' exported font floor',minfont>=7.45)
        dpi=[]
        for im in p.get_image_info():
            rect=fitz.Rect(im['bbox']);dpi.append(min(im['width']/rect.width*72,im['height']/rect.height*72))
        # Matrix heatmaps deliberately embed one un-interpolated pixel per cell;
        # their numerical grid must not be smoothed merely to increase DPI.
        if dpi and path.stem in ['Supplementary_Fig_01','Supplementary_Fig_06']:
            check(path.name+' map raster export resolution',min(dpi)>290)
        pdf_details.append(dict(file=path.name,pages=len(doc),width_mm=p.rect.width*25.4/72,minimum_text_pt=minfont,fonts_embedded=embedded,raster_min_dpi=min(dpi) if dpi else None))
with fitz.open(ROOT/'Supplementary_Figures_with_captions.pdf') as doc:
    check('caption booklet 17 pages',len(doc)==17)
    for i,p in enumerate(doc):
        text=p.get_text();check('caption same page S'+str(i+1),'Supplementary Fig. '+str(i+1)+' |' in text)
        check('no audit-status stamp S'+str(i+1),'Status:' not in text and 'Review package' not in text)
        check('caption last sentence rendered S'+str(i+1),records[i]['statistics'].split()[-1] in text)
        if i in [0,4,5,6,12,13,14,15]:p.get_pixmap(dpi=120).save(ROOT/'qa'/f'caption_page_{i+1:02d}.png')
with fitz.open(ROOT/'Supplementary_Information.pdf') as doc:
    info_pages=len(doc)
    alltext=' '.join(p.get_text() for p in doc)
    check('no obsolete main Fig1c references','main Fig. 1c' not in alltext and 'Main Fig. 1c' not in alltext)
    check('combined PDF bookmarks',len(doc.get_toc())>=26)
    for i,p in enumerate(doc):
        footer=p.get_textbox(fitz.Rect(280,p.rect.height-32,320,p.rect.height-7)).strip()
        check('continuous combined page '+str(i+1),footer==str(i+1),footer)
    doc[0].get_pixmap(dpi=120).save(ROOT/'qa/methods_page_01.png')
with fitz.open(ROOT/'Supplementary_Methods_and_coverage.pdf') as doc:
    methods_pages=len(doc)
    for i in [len(doc)-2,len(doc)-1]:doc[i].get_pixmap(dpi=120).save(ROOT/'qa'/f'coverage_{i+1:02d}.png')

dump(ROOT/'qa/release_validation.json',dict(tests=tests,pdf_details=pdf_details,figure_count=17,full_pdf_pages=info_pages,methods_pdf_pages=methods_pages,
    technical_status='passed',scientific_status='limited by documented SST and tropical-input issues; not submission certification'))

# Populate the skill's project records, not the user's main-manuscript state.
p=json.loads((WF/'project.yml').read_text())
p.update(research_question='How well do supplementary analyses support the figure-led MHW forecast diagnosis?',
    primary_claim='Seventeen source-traceable SI figures, including fixed-neighbourhood and extended-window spatial skill sensitivity.',
    current_phase='supplementary_figure_delivery',official_guidance_checked='Attempted 2026-11-11; official web access failed; no current-guidance certification',
    official_guidance_urls=['https://www.nature.com/ngeo/submission-guidelines/aip-and-formatting'],
    authoritative_results='figure_registry.json; source_manifest.json; derived_data/',target_manuscript='Not edited in this scoped figure revision',random_seed=20261111,
    notes='Earlier main and supplementary figures preserved. Scientific source limitations are explicit. Current manuscript cross-references require author-stage integration.')
dump(WF/'project.yml',p)
coverage=pd.read_csv(ROOT/'data_coverage.csv',keep_default_na=False)
dump(WF/'data-contract.yml',dict(schema_version=1,analysis=dict(sample_unit='Event; months/leads/regions are dependent repetitions',evaluation_cohort='Per-figure coverage table',selection_cohort='Retained main-figure event sets',temporal_weighting='Defined per figure in Supplementary Methods',spatial_weighting='Cosine latitude and native region masks',aggregation_order='See Supplementary_Methods.md',uncertainty_method='S2 common year-block bootstrap; S1 regression prediction interval; lead ranges elsewhere',missing_data_policy='NA; no zero fill or invented leads',random_seed=20261111),datasets=coverage.to_dict('records'),checksum_manifest='source_manifest.json',raw_immutable=True))
dump(WF/'figure-contracts.yml',dict(schema_version=1,figures=[dict(id=r['id'],status='rendered_and_checked',question=r['role'],answer=r['title'],source_products=r['sources'],calculation_script='revise_si.py; base_si.py',plotting_script='revise_si.py',outputs=['figures/'+r['id']+'.pdf'],uncertainty=r['statistics'],interpretation_boundary='Supplementary_Methods.md',validation_report='qa/release_validation.json') for r in records]))
df=pd.DataFrame([dict(decision_id='D'+str(i+1).zfill(3),date='2026-11-11',topic=t,decision=d,status='active',rationale='Current user request and attached review',evidence=e,affected_artifacts=a,replaces='none') for i,(t,d,e,a) in enumerate([
    ('scope','Preserve main figures and old SI; redraw all 17 SI figures; synchronize S15 with current Fig4','source_data/CURRENT4/regional_map_shortfall.csv','S1-S17'),
    ('regions','S12-S15 follow current Fig4 +/-23.5; S1c/S16 sensitivity masks retain +/-20','source_manifest.json','S1 S7 S12-S16'),
    ('science','No unsupported causal attribution or independent-model label','Supplementary_Methods.md','S9 S12-S15'),
    ('statistics','S2 joint year-block intervals; event and lead ranges distinguished','derived_data/S02_slope_difference.csv','S2'),
    ('limits','Retain substantive source limitations; do not silently clear P0','REVIEWER_RESPONSE.md','Methods and release')])])
df.to_csv(WF/'decision-log.csv',index=False)
pd.DataFrame([dict(claim_id='S'+str(r['number']),claim_text=r['title'],role='supporting',evidence_class='diagnostic or descriptive',status='qualified',authoritative_artifact='figures/'+r['id']+'.pdf',uncertainty_test=r['statistics'],alternative_explanation='See Methods and reviewer response',permitted_wording=r['caption'],figure_or_section='Supplementary Fig. '+str(r['number'])) for r in records]).to_csv(WF/'claim-evidence.csv',index=False)
pd.DataFrame([dict(number_id='N'+str(i+1).zfill(3),display_value=v,units=u,source_path=s,selector=c,precision='full numeric',used_in=used,tolerance=1e-12) for i,(v,u,s,c,used) in enumerate([
    (.10651437039268086,'AUC','derived_data/S15_main_Fig4_region_crosswalk.csv','sum Net_shortfall','S15'),
    (float(s2.estimate),'AUC per degree C','derived_data/S02_slope_difference.csv','estimate','S2'),
    (float(s2.low),'AUC per degree C','derived_data/S02_slope_difference.csv','low','S2'),
    (float(s2.high),'AUC per degree C','derived_data/S02_slope_difference.csv','high','S2')])]).to_csv(WF/'number-registry.csv',index=False)
pd.DataFrame([dict(supplementary_figure=r['number'],main_figure=r['main_figure'],purpose=r['role'],suggested_section=('Results 1 / Methods: MHW verification' if r['number']<=4 or r['number']>=16 else 'Results 2 / Methods: atmospheric verification' if r['number']<=11 else 'Results 3 / Methods: regional diagnostics'),manuscript_status='Proposed citation location; Word file not modified or audited') for r in records]).to_csv(ROOT/'manuscript_reference_crosswalk.csv',index=False)
sk=json.loads((WF/'submission-checklist.yml').read_text());sk.update(figures_complete=True,captions_complete=True,number_registry_passed=True,document_integrity_passed=True,release_validation_passed=True,notes=['Technical SI delivery only; unresolved source-selection issues remain','Main manuscript not edited; current cross-reference sequence not certified','Official journal page could not be checked live'])
dump(WF/'submission-checklist.yml',sk)
pdfs=sorted(ROOT.glob('*.pdf'))+sorted((ROOT/'figures').glob('*.pdf'))+sorted((ROOT/'individual_with_captions').glob('*.pdf'))
pd.DataFrame([dict(artifact_id='A'+str(i+1).zfill(3),type='PDF',path=str(f.relative_to(ROOT)),status='validated_figure_delivery',source_artifacts='source_manifest.json',generator='revise_si.py; package_si.py',validation='qa/release_validation.json',sha256=sha(f),updated_at='2026-11-11') for i,f in enumerate(pdfs)]).to_csv(WF/'artifact-registry.csv',index=False)
print(json.dumps(dict(passed_checks=len(tests),figures=len(records),full_pdf_pages=info_pages,methods_pages=methods_pages),indent=2))
