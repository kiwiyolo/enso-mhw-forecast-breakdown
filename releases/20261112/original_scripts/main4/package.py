"""Preserve original vector panels; build review PDFs and check their provenance."""
from pathlib import Path
import hashlib,json,shutil,csv
import fitz
import numpy as np
import pandas as pd
import xarray as xr

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures'
BASE=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures')
MAIN=BASE/'Figure1-3_Nature_layout_20261110_v7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')

def main():
    checks={};placements=[]
    for i in [1,2,3]:
        source=MAIN/'figures'/f'Figure{i}.pdf'
        shutil.copy2(source,OUT/source.name)
        checks[f'Figure{i}_original_vector_hash_identical']=sha(source)==sha(OUT/source.name)
        with fitz.open(source) as d:
            d[0].get_pixmap(dpi=300,alpha=False).save(OUT/f'Figure{i}.png')
        svg=source.with_suffix('.svg')
        if svg.exists():shutil.copy2(svg,OUT/svg.name)
    # Verify removing c truly restores a,b, not a different earlier version.
    with fitz.open(OUT/'Figure1.pdf') as a,fitz.open(BASE/'Figure1_neighbourhood_skill_20261111_v10/figures/Figure1.pdf') as b:
        aa=a[0].get_pixmap(dpi=150,alpha=False)
        bb=b[0].get_pixmap(dpi=150,alpha=False,clip=a[0].rect)
        diff=int(np.max(np.abs(np.frombuffer(aa.samples,np.uint8).astype(int)-np.frombuffer(bb.samples,np.uint8).astype(int))))
        checks['Figure1ab_original_crop_max_pixel_difference']=diff
        # PDF embedding introduces a subpixel translation (~3e-5 pt) in v10.
        # Pixel differences up to one 8-bit level are rasterizer rounding only;
        # additionally require the archived pre-embedding vector hash and text.
        assert diff<=1
        assert a[0].get_text()==b[0].get_text(clip=a[0].rect)
        original=BASE/'Figure1_neighbourhood_skill_20261111_v10/provenance/original_Figure1.pdf'
        assert sha(original)==sha(OUT/'Figure1.pdf')
        checks['Figure1ab_exact_preembedding_vector_hash_identical']=True
    paths=[OUT/f'Figure{i}.pdf' for i in range(1,5)]
    book=fitz.open()
    for p in paths:
        with fitz.open(p) as d:book.insert_pdf(d)
    book.save(OUT/'Figures1-4.pdf',garbage=4,deflate=True);book.close()
    # Actual publication sizes are retained, rather than fitting to A4 and shrinking labels.
    docs=[fitz.open(p) for p in paths]
    mm=72/25.4;margin=5*mm;gap=7*mm;header=7*mm
    col=[max(docs[0][0].rect.width,docs[2][0].rect.width),max(docs[1][0].rect.width,docs[3][0].rect.width)]
    heights=[max(docs[0][0].rect.height,docs[1][0].rect.height),max(docs[2][0].rect.height,docs[3][0].rect.height)]
    comp=fitz.open();page=comp.new_page(width=2*margin+sum(col)+gap,height=2*margin+sum(heights)+gap+2*header)
    for i,d in enumerate(docs):
        row,c=divmod(i,2);x=margin+(col[0]+gap)*c;y=margin+header+(heights[0]+gap+header)*row
        rect=fitz.Rect(x,y,x+d[0].rect.width,y+d[0].rect.height)
        page.insert_text((x,y-2*mm),f'Figure {i+1}',fontname='hebo',fontsize=10)
        page.show_pdf_page(rect,d,0)
        placements.append(dict(figure=i+1,source=str(paths[i]),sha256=sha(paths[i]),rectangle=list(rect),scale=1))
    comp.save(OUT/'Figures1-4_2x2.pdf',garbage=4,deflate=True)
    page.get_pixmap(dpi=120,alpha=False).save(OUT/'Figures1-4_2x2_preview.png')
    comp.close()
    for d in docs:d.close()
    record(ROOT/'composition_manifest.json',placements)
    # PDF-level text geometry and consistency checks supplement the visual review.
    for p in paths:
        with fitz.open(p) as d:
            spans=[s for b in d[0].get_text('dict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans']]
            bad=[s['text'] for s in spans if not (fitz.Rect(s['bbox']) & d[0].rect).get_area()]
            assert len(d)==1 and not bad
            checks[p.stem+'_one_page_and_visible_text']=True
    rendered=json.loads((ROOT/'render_validation.json').read_text())
    assert not rendered['Figure4']['outside_canvas_text']
    checks['Figure4_no_outside_canvas_text']=True
    checks['Figure4_min_text_pt']=rendered['Figure4']['minimum_rendered_text_pt']
    numbers=pd.read_csv(ROOT/'source_data/regional_map_shortfall.csv',keep_default_na=False)
    assert abs(numbers.Shortfall_share_pct.sum()-100)<1e-9
    flow=pd.read_csv(ROOT/'source_data/association_weighted_local_shortfall.csv',keep_default_na=False)
    assert np.allclose(flow.groupby('Region').Allocated_positive_shortfall.sum().sort_index(),numbers.set_index('Region').Positive_shortfall.sort_index())
    checks['regions_and_ribbons_close_to_map_positive_shortfall']=True
    assert np.allclose(flow.groupby('Region').Allocated_above_benchmark.sum().sort_index(),numbers.set_index('Region').Above_benchmark.sort_index(),rtol=1e-10,atol=1e-12)
    checks['regions_and_ribbons_close_to_map_negative_shortfall']=True
    old=ROOT.parent/'Figure4_sign_hierarchy_20261111_v26'
    checks['all_source_tables_and_map_unchanged']=all(sha(p)==sha(old/'source_data'/p.name) for p in (ROOT/'source_data').iterdir() if p.is_file())
    assert checks['all_source_tables_and_map_unchanged']
    with fitz.open(OUT/'Figure4.pdf') as newdoc,fitz.open(old/'figures/Figure4.pdf') as olddoc:
        # All of panel a including its legend is unchanged in this revision.
        clip=fitz.Rect(0,0,newdoc[0].rect.width,94*72/25.4)
        na=np.frombuffer(newdoc[0].get_pixmap(dpi=200,clip=clip,alpha=False).samples,np.uint8).astype(int)
        oa=np.frombuffer(olddoc[0].get_pixmap(dpi=200,clip=clip,alpha=False).samples,np.uint8).astype(int)
        checks['Figure4a_map_and_insets_max_pixel_difference']=int(np.abs(na-oa).max())
        assert checks['Figure4a_map_and_insets_max_pixel_difference']==0
    signed=rendered['signed_regional_accounting']
    assert signed['hierarchy']==['Regions','Positive / Negative','Drivers','Local components']
    assert signed['all_sign_nodes_conserve_magnitude']
    checks['explicit_sign_stage_conserved']=True
    den=numbers.Net_shortfall.sum()
    assert abs(signed['lower_total_pct']-100*numbers.Positive_shortfall.sum()/den)<1e-9
    assert abs(signed['higher_total_pct']+100*numbers.Above_benchmark.sum()/den)<1e-9
    assert abs(signed['lower_total_pct']+signed['higher_total_pct']-100)<1e-9
    checks['positive_and_negative_common_denominator_verified']=True
    for r,row in numbers.set_index('Region').iterrows():
        assert abs(signed['regions'][r]['lower_pct']-100*row.Positive_shortfall/den)<1e-9
        assert abs(signed['regions'][r]['higher_pct']+100*row.Above_benchmark/den)<1e-9
    checks['signed_percentages_sum_to_100']=True
    checks['display_denominator_auc']=float(den)
    checks['positive_percent']=signed['lower_total_pct']
    checks['negative_percent']=signed['higher_total_pct']
    svg=(OUT/'Figure4.svg').read_text()
    checks['negative_ribbons_and_dashed_locators_present']='_higher' in svg and 'higher_skill_locator' in svg
    assert checks['negative_ribbons_and_dashed_locators_present']
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    assert all(sha(Path(z['path']))==z['sha256'] for z in manifest)
    checks['all_input_hashes_unchanged']=True
    checks['review_status']='Rendered and numerically verified; inherited source limitations and global-vs-local benchmark boundary remain'
    record(ROOT/'qa/release_validation.json',checks)
    rows=[]
    for p in paths+[OUT/'Figures1-4.pdf',OUT/'Figures1-4_2x2.pdf']:
        rows.append(dict(artifact_id=p.stem,type='figure_pdf',path=str(p.relative_to(ROOT)),status='active',sha256=sha(p)))
    with (ROOT/'.paper-workflow/artifact-registry.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
