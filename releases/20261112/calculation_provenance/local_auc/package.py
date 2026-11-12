"""Independent numeric checks, immutable lineage and captioned delivery."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import pandas as pd
import xarray as xr
import fitz
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';DATA=ROOT/'source_data';QA=ROOT/'qa'
SI=ROOT.parent/'Supplementary_Figures_20261111_v3'
OLD=ROOT.parent/'Figure1_skill_decrease_20261111_v9'
tests=[]
def check(name,ok,detail=None):
    tests.append(dict(test=name,passed=bool(ok),detail=detail));assert ok,(name,detail)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()
def captioned(title,pdf,paras,output):
    d=fitz.open();p=d.new_page(width=595.28,height=841.89)
    for n,f in [('Body','LiberationSans-Regular.ttf'),('Bold','LiberationSans-Bold.ttf')]:p.insert_font(fontname=n,fontfile=str(ROOT/'fonts'/f))
    assert p.insert_textbox(fitz.Rect(42,28,553,70),title,fontname='Bold',fontsize=11)>=0
    with fitz.open(pdf) as art:
        w,h=art[0].rect.width,art[0].rect.height
        rect=fitz.Rect((595.28-w)/2,76,(595.28+w)/2,76+h)
        p.show_pdf_page(rect,art,0)
    y=rect.y1+12
    for para in paras:
        box=fitz.Rect(42,y,553,802);space=p.insert_textbox(box,para,fontname='Body',fontsize=9,lineheight=1.18)
        if space<0:
            p=d.new_page(width=595.28,height=841.89)
            p.insert_font(fontname='Body',fontfile=str(ROOT/'fonts/LiberationSans-Regular.ttf'))
            box=fitz.Rect(42,40,553,802);space=p.insert_textbox(box,para,fontname='Body',fontsize=9,lineheight=1.18)
        check('caption paragraph fits',space>=0);y=box.y1-space+7
    d.save(output,garbage=4,deflate=True);d.close()

def main():
    qa=json.loads((ROOT/'calculation_validation.json').read_text());check('all calculation checks',all(x['passed'] for x in qa['checks']))
    sources=json.loads((ROOT/'source_manifest.json').read_text())+json.loads((ROOT/'cache_manifest.json').read_text())
    for source in {x['path']:x for x in sources}.values():check('immutable '+source['path'],sha(source['path'])==source['sha256'])
    for r in json.loads((QA/'render_validation.json').read_text()):
        check(r['figure']+' width',abs(r['width_mm']-180)<.01)
        if 'outside_text' in r:check(r['figure']+' no clipping',not r['outside_text'])
        if 'top_crop_max_channel_difference' in r:check('Fig1ab unchanged',r['top_crop_max_channel_difference']<=8,r['top_crop_max_channel_difference'])
    z=np.load(DATA/'event_lead_fields.npz');ds=xr.load_dataset(DATA/'canonical_relative_changes.nc');a=z['auc'];m=z['conditional_intensity_mse']
    ac=np.isfinite(a).all((0,1));rc=np.isfinite(m).all((0,1))
    check('common AUC mask',np.array_equal(ac,ds.auc_common.values))
    check('common RMSE mask',np.array_equal(rc,ds.rmse_common.values))
    check('map AUC values',np.allclose(ds.auc_decrease.values[ac],(.5*(a[0]+a[1])-a[2]).mean(0)[ac]))
    check('map RMSE values',np.allclose(ds.rmse_increase.values[rc],(np.sqrt(m[2].mean(0))-np.sqrt(.5*(m[0].mean(0)+m[1].mean(0))))[rc]))
    check('no invented values',np.isnan(ds.auc_decrease.values[~ac]).all() and np.isnan(ds.rmse_increase.values[~rc]).all())
    tab=pd.read_csv(DATA/'regional_changes.csv',keep_default_na=False);lead=pd.read_csv(DATA/'regional_changes_by_lead.csv',keep_default_na=False)
    reg=['TP','NP','SP','TA','NA','SA','IO'];w=ds.area_weight.values
    for ri,r in enumerate(reg):
        for metric,mask in [('AUC',ac),('RMSE',rc)]:
            full=ds.region.values==ri;use=full&mask;ww=w[use]/w[use].sum()
            vals=np.array([[np.dot((a if metric=='AUC' else m)[e,l,use],ww) for l in range(9)] for e in range(3)])
            dif=.5*(vals[0]+vals[1])-vals[2] if metric=='AUC' else np.sqrt(vals[2])-np.sqrt(.5*(vals[0]+vals[1]))
            value=dif.mean() if metric=='AUC' else np.sqrt(vals[2].mean())-np.sqrt(.5*(vals[0].mean()+vals[1].mean()))
            row=tab[(tab.metric==metric)&(tab.region==r)].iloc[0]
            check(metric+' regional value '+r,abs(value-row.change)<1e-12)
            check(metric+' coverage '+r,abs(row.coverage-w[use].sum()/w[full].sum())<1e-12)
            check(metric+' nine leads '+r,np.allclose(dif,lead[(lead.metric==metric)&(lead.region==r)].sort_values('lead').change))
    check('selected regions not reselected',json.loads((ROOT/'selected_regions.json').read_text())['regions']==json.loads((OLD/'selected_regions.json').read_text())['regions'])
    cov=pd.read_csv(DATA/'coverage_comparison.csv');v=cov[cov.metric=='AUC'].set_index('variant');new=v.loc['18months_11deg'];old=v.loc['12months_1deg']
    paras=[
        'a, Historical relationship between El Niño amplitude and global MHW forecast skill (AUC). Error bars, regression and 95% prediction interval are unchanged; 2023–2024 is excluded from fitting. b, Unchanged monthly global forecast skill and Niño3.4, with three strong El Niño events highlighted.',
        'c, Canonical-minus-2023–2024 AUC: the equal mean of 1997–1998 and 2015–2016 minus 2023–2024. Red denotes decreased skill, blue improved skill. For each event and lead, raw cell–month forecast scores and observed MHW occurrence are pooled within a fixed 11° × 11° same-basin ocean neighbourhood, using cosine-latitude weights. The window runs from April of the event year through September of the next year (18 months), adding three months at each end of July–June. All events use the same neighbour support and leads 1–9. Comparisons require both observed classes in every event and all leads; undefined ocean locations remain grey.',
        'The inset belongs to c. Bars summarize the same four basins selected in the previous figure (TA, NP, SA, NA), without new selection. Points show the nine lead-specific differences; whiskers span their minimum–maximum range, not a confidence interval. Basin summaries are cosine-latitude averages on common valid centres. Dashed boundaries retain the original ±20° analysis divisions. All seven regions and duration/neighbourhood sensitivity are shown in Supplementary Figs. 16–17; the matching intensity-RMSE map is in Supplementary Fig. 1c.',
        f'Common valid AUC coverage is {new.area_coverage:.1%} of the analysis ocean area ({new.cell_coverage:.1%} of grid centres), compared with {old.area_coverage:.1%} under the original single-cell, 12-month method. The 1° output grid and colour range are unchanged; values are not interpolated or smoothed after computation. The effective diagnostic support is 11°, so adjacent values are not independent 1° estimates. Extending the window changes event-life-cycle support. Panel c is a pooled space–time discrimination diagnostic and is not the monthly spatial AUC in a,b; regional differences are not additive components of their global gap.',
        'The author-requested archived SST extraction is retained, including its documented initialization-coordinate limitations. This spatial-support revision does not resolve those source-level limitations or alter the headline AUC difference.'
    ]
    (ROOT/'captions').mkdir(exist_ok=True)
    (ROOT/'captions/Figure1.md').write_text('# Figure 1 | MHW forecast skill fell below historical expectations\n\n'+'\n\n'.join(paras)+'\n')
    captioned('Figure 1 | MHW forecast skill fell below historical expectations',OUT/'Figure1.pdf',paras,ROOT/'Figure1_with_caption.pdf')
    review=fitz.open()
    for p in [ROOT/'Figure1_with_caption.pdf',SI/'individual_with_captions/Supplementary_Fig_01.pdf',SI/'individual_with_captions/Supplementary_Fig_16.pdf',SI/'individual_with_captions/Supplementary_Fig_17.pdf']:
        with fitz.open(p) as d:review.insert_pdf(d)
    review.save(ROOT/'Figure1_and_RMSE_review.pdf',garbage=4,deflate=True);review.close()
    check('SI updated source identity',sha(SI/'source_data/SPATIAL10/canonical_relative_changes.nc')==sha(DATA/'canonical_relative_changes.nc'))
    for name in ['Figure1','Figure1c']:
        with fitz.open(OUT/f'{name}.pdf') as d:
            p=d[0];spans=[s for b in p.get_text('dict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans']]
            check(name+' embedded fonts',all(bool(d.extract_font(f[0])[3]) for f in p.get_fonts(full=True)))
            check(name+' PDF text inside',all(fitz.Rect(s['bbox']).x0>=-1 and fitz.Rect(s['bbox']).y0>=-1 and fitz.Rect(s['bbox']).x1<=p.rect.width+1 and fitz.Rect(s['bbox']).y1<=p.rect.height+1 for s in spans))
    (QA/'release_validation.json').write_text(json.dumps(dict(checks=tests,passed=len(tests),coverage=cov.to_dict('records')),indent=2))
    (ROOT/'README.md').write_text('# Figure 1 spatial-support revision\n\nFig1a,b, the global AUC deficit, and earlier releases are unchanged. Fig1c uses 18-month, same-basin 11° neighbourhood ROC AUC; no interpolation or post-calculation smoothing.\n\n- figures/Figure1.pdf and PNG: updated complete figure.\n- Figure1_with_caption.pdf: full scientific caption.\n- Figure1_and_RMSE_review.pdf: main figure plus S1, S16 and S17.\n- source_data/: gridded metrics, regional values and all four sensitivity variants.\n- contract.json: definitions fixed before computation.\n- calculation_validation.json and qa/: numerical and layout checks.\n- cache_manifest.json: added archived source months with hashes.\n\nReproduce with ocean_sst Python: calculate.py, render.py; then the sibling Supplementary_Figures_20261111_v3/revise_si.py --only 1 16 17, followed by package.py.\n\nOriginal archived-source initialization limitations persist; this release addresses sampling coverage, not source-coordinate validation. Current journal-guidance fetch was attempted but access failed; no new formatting certification is claimed.\n')
    wf=ROOT/'.paper-workflow';wf.mkdir(exist_ok=True)
    (wf/'project.yml').write_text(json.dumps(dict(title='Unexpectedly low marine heatwave forecast skill during the 2023–2024 El Niño',journal='Nature Geoscience',current_phase='Figure1 spatial sampling revision',raw_data_immutable=True,contract='../contract.json',validation='../qa/release_validation.json'),indent=2))
    pd.DataFrame([dict(date='2026-11-11',status='active',decision=k,source='contract.json') for k in ['Preserve Fig1ab and headline','18-month symmetric extension','11-degree same-basin sample pooling','Preserve undefined cells','Freeze highlighted basins','Add four-way SI sensitivity']]).to_csv(wf/'decision-log.csv',index=False)
    (ROOT/'release_manifest.json').write_text(json.dumps([dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='release_manifest.json' and '__pycache__' not in p.parts],indent=2))
    print('Validated',len(tests),'checks',flush=True)
if __name__=='__main__':main()
