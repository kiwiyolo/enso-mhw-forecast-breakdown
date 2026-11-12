"""Integrate the matched S5, package the final SI and check preserved numerics."""
from pathlib import Path
import json,hashlib,shutil
import fitz,numpy as np,pandas as pd
import package_si as p
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'Supplementary_Figures_20261111_v5'
MATCHED=ROOT.parent/'Supplementary_Fig5_matched_DJF_20261111_v1'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
records=json.loads((ROOT/'figure_registry.json').read_text())
r=next(r for r in records if r['number']==5)
for ext in ['pdf','png','svg']:
    shutil.copy2(ROOT/f'matched_s5/figures/Supplementary_Fig_05.{ext}',ROOT/f'figures/Supplementary_Fig_05.{ext}')
r.update(json.loads((ROOT/'matched_s5/caption.json').read_text()))
r.update(sources=['matched_s5/source_data/correlations.csv','matched_s5/source_data/correlation_changes.csv','matched_s5/statistics.json'],
         min_font_pt=6.5,width_mm=180,height_mm=140,outside_text=[],
         calculation='Frozen exact-start, common-model DJF recalculation; identical to matched-DJF v1',
         status='Matched single-model comparison; not multimodel attribution')
p.package(records)
checks=[]
def check(name,ok,details=None):
    checks.append(dict(check=name,passed=bool(ok),details=details))
    if not ok:raise AssertionError((name,details))
for f in sorted((OLD/'derived_data').glob('*.csv')):
    g=ROOT/'derived_data'/f.name
    check('unchanged derived table '+f.name,sha(f)==sha(g))
for name in ['correlations.csv','correlation_changes.csv']:
    check('matched S5 data '+name,sha(ROOT/'matched_s5/source_data'/name)==sha(MATCHED/'source_data'/name))
for ext in ['pdf','png','svg']:
    check('S7 unchanged '+ext,sha(ROOT/f'figures/Supplementary_Fig_07.{ext}')==sha(OLD/f'figures/Supplementary_Fig_07.{ext}'))
bad=['All seven regions','95% interval:','Year-adjusted AUC','Omit one event','Event midpoint year','AUC per °C','Within-event evolution','Verifying month','Height RMSE','Pooled association','Sensitivity to event omission','Later-period SST reconstruction','Predictor-removal test','Association-weighted allocation','AUC comparison coverage','Common ocean area','RMSE comparison coverage','months · grid cell','months · 11° neighbourhood']
centering=[]
for r in records:
    n=r['number'];f=ROOT/f'figures/{r["id"]}.pdf'
    check(f'S{n} no clipped text',not r['outside_text'],r['outside_text'])
    with fitz.open(f) as d:
        check(f'S{n} artwork width 180 mm',abs(d[0].rect.width-180/25.4*72)<.02)
        text=d[0].get_text()
        if n!=7:check(f'S{n} requested old phrases removed',not [s for s in bad if s in text],[s for s in bad if s in text])
        pix=d[0].get_pixmap(dpi=200,alpha=False)
        pix.save(ROOT/f'figures/{r["id"]}_preview.png')
        a=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)
        ys,xs=np.where(a[:,:,:3].min(axis=2)<225)
        imbalance=float((int(xs.min())+int(xs.max())-(pix.width-1))/2*72/200)
        centering.append(dict(figure=n,ink_centre_offset_pt=imbalance))
        if n!=7:check(f'S{n} horizontally centred within 2 mm',abs(imbalance)<72/25.4*2,imbalance)
with fitz.open(ROOT/'Supplementary_Figures_with_captions.pdf') as d:
    check('17 captioned figures',len(d)==17)
    for i,page in enumerate(d):
        check(f'S{i+1} single A4 page',abs(page.rect.height-841.89)<.02)
        check(f'S{i+1} caption present',f'Supplementary Fig. {i+1} |' in page.get_text())
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for span in line['spans']:
                    rect=fitz.Rect(span['bbox'])
                    check(f'S{i+1} text inside page: {span["text"][:20]}',rect.x0>=-.5 and rect.x1<=596 and rect.y0>=0 and rect.y1<=842)
with fitz.open(ROOT/'Supplementary_Information.pdf') as d:pages=len(d)
pd.DataFrame(centering).to_csv(ROOT/'qa/centering.csv',index=False)
(ROOT/'qa/final_validation.json').write_text(json.dumps(dict(checks=checks,passed=all(c['passed'] for c in checks),combined_pages=pages,scientific_submission_clearance=False),indent=2))
changes={1:'Remove seven-region title; compact rows; retain every region and lead.',2:'Move correlation intervals to caption.',3:'Use explicit event-year, time-trend and omitted-event labels.',4:'Use Month being forecast / Monthly forecast skill.',5:'Use matched DJF recalculation; compact columns and legend.',6:'Compact nine-map grid; add latitude/longitude labels.',7:'Original artwork byte-for-byte unchanged.',8:'Centre complete artwork; name 200-hPa geopotential-height RMSE.',9:'Use event-specific convection/circulation wording and concise event-dependence title.',10:'Complete shared event legend.',11:'Move legend closer to panels.',12:'Move legend closer to panels.',13:'Centre complete artwork; describe out-of-fit SST estimates and removal tests plainly.',14:'Panel b lower left; top and bottom rows have equal outer axis widths.',15:'Narrow column gap; single-row legend; explicit historical-SST relationship allocation.',16:'Use Ocean area included and valid AUC/intensity-RMSE labels.',17:'Explicit time windows and individual versus pooled cells; compact legend.'}
(ROOT/'LAYOUT_REVISION.md').write_text('# Final SI polishing\n\n'+'\n'.join(f'- S{n}: {v}' for n,v in changes.items())+'\n\nAll panel letters follow row/column anchors; complete artwork is centred including labels and legends. S7 is exempt from repositioning to preserve the original. Maps, numerical scales, masks, resolution, smoothing and statistical definitions are unchanged. S5 uses the already recalculated matched-DJF release.\n\nOfficial NCC guidance: https://www.nature.com/nclimate/submission-guidelines/aip-and-formatting — SI is published as supplied; figures are sequentially numbered, each with its legend on one PDF page, and combined with supplementary text/tables. A4 pages are our consistent layout choice, not a stated NCC requirement.\n')
(ROOT/'README.md').write_text('# Final SI layout release v6\n\n17 individually exported figures, each with its complete caption on one A4 page. Earlier releases and all main figures remain unchanged.\n\n- Supplementary_Information.pdf: supplementary methods, data-coverage table and 17 figures.\n- Supplementary_Figures_with_captions.pdf: 17 captioned figures.\n- Supplementary_Figures_only.pdf: full-size vector artwork.\n- Individual_SI_PDFs.zip: individual artwork and captioned PDFs.\n- SI_overview.pdf: contact sheets.\n- qa/final_validation.json: numerical-preservation, centring and page checks.\n- matched_s5/: exact-start, same-model DJF S5 and data.\n- LAYOUT_REVISION.md: item-by-item edits and official formatting reference.\n\nTypography: 180-mm artwork, sans serif, generally 6–7 pt labels and 9-pt panel letters; S7 retains the previous artwork without alteration.\n\nReproduction: run revise_si.py --only 1 2 3 4 6 8 9 10 11 12 13 14 15 16 17, matched_s5/render.py, then complete_release.py using the ocean_sst Python environment.\n\nFormatting checks do not constitute scientific submission clearance. Remaining archived-SST input and manuscript-integration limitations are recorded in SUBMISSION_BLOCKERS.md. S5 now has matched model, season, leads and spatial support; its single-model scope remains explicit.\n')
manifest={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and f.suffix in ['.pdf','.csv','.py','.json'] and 'provenance' not in f.parts and '__pycache__' not in f.parts and f.name!='release_manifest.json'}
(ROOT/'release_manifest.json').write_text(json.dumps(dict(release=ROOT.name,files=manifest,checks_passed=len(checks)),indent=2))
print('FINAL:',len(records),'figures;',pages,'pages;',len(checks),'checks passed')
