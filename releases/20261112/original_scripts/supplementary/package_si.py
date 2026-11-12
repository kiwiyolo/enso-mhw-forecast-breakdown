"""Vector figure/caption pages, reader-facing Methods and reproducibility package."""
from pathlib import Path
import json,hashlib,zipfile
import fitz,pandas as pd,numpy as np
from submission_captions import METHODS,COVERAGE,CAPTIONS
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'figures';FONT=ROOT/'fonts'

def font_page(doc,height=841.89):
    p=doc.new_page(width=595.28,height=height)
    p.insert_font(fontname='Body',fontfile=str(FONT/'LiberationSans-Regular.ttf'))
    p.insert_font(fontname='Bold',fontfile=str(FONT/'LiberationSans-Bold.ttf'))
    return p
def text_height(text,width,size=10.5,bold=False):
    f=fitz.Font(fontfile=str(FONT/('LiberationSans-Bold.ttf' if bold else 'LiberationSans-Regular.ttf')))
    lines=0
    for para in text.split('\n'):
        cur='';lines+=1
        for word in para.split():
            if f.text_length((cur+' '+word).strip(),fontsize=size)>width:lines+=1;cur=word
            else:cur=(cur+' '+word).strip()
    return lines*size*1.28+size
def number_pages(doc):
    for p in doc:
        p.insert_font(fontname='Footer',fontfile=str(FONT/'LiberationSans-Regular.ttf'))
        p.insert_text((292,p.rect.height-19),str(p.number+1),fontsize=9,fontname='Footer')
def caption_pages(records):
    doc=fitz.open();toc=[];(ROOT/'individual_with_captions').mkdir(exist_ok=True)
    for r in records:
        single=fitz.open();header=f'Supplementary Fig. {r["number"]} | {r["title"]}'
        body=r['caption']+'\n\n'+r['statistics']
        head_h=text_height(header,511,12,True)
        art_y=22+head_h+7
        with fitz.open(OUT/(r['id']+'.pdf')) as art:
            art_h=art[0].rect.height
        body_y=art_y+art_h+7
        body_h=text_height(body,511,9.5)
        page_h=841.89
        assert body_y+body_h+34<=page_h,(r['number'],body_y+body_h+34)
        p=font_page(single,height=page_h)
        assert p.insert_textbox(fitz.Rect(42,22,553,22+head_h+5),header,fontsize=12,fontname='Bold',lineheight=1.15)>=0
        with fitz.open(OUT/(r['id']+'.pdf')) as art:
            # Exact 180-mm artwork width: no silent shrinking of figure typography.
            w=art[0].rect.width;h=art[0].rect.height;rect=fitz.Rect((595.28-w)/2,art_y,(595.28+w)/2,art_y+h)
            # S7 artwork is immutable; centre its visible content by page placement only.
            if r['number']==7:rect=rect+(-7.02,0,-7.02,0)
            p.show_pdf_page(rect,art,0)
        spare=p.insert_textbox(fitz.Rect(42,body_y,553,page_h-30),body,fontname='Body',fontsize=9.5,lineheight=1.2)
        if spare<0:raise RuntimeError(f'Caption overflow S{r["number"]}: {spare}; reduce graphic whitespace, not font size')
        single.set_metadata(dict(title=header,author='Qun Luo'))
        single.save(ROOT/'individual_with_captions'/(r['id']+'.pdf'),garbage=4,deflate=True)
        doc.insert_pdf(single);toc.append([1,header,len(doc)]);single.close()
    number_pages(doc);doc.set_toc(toc);doc.set_metadata(dict(title='Supplementary figures: marine heatwave forecast skill',author='Qun Luo'))
    doc.save(ROOT/'Supplementary_Figures_with_captions.pdf',garbage=4,deflate=True);doc.close()

def methods_pdf(records):
    doc=fitz.open();p=font_page(doc);y=35;toc=[]
    title='Unexpectedly low marine heatwave forecast skill during the 2023–2024 El Niño'
    h=text_height(title,511,14,True);assert p.insert_textbox(fitz.Rect(42,y,553,y+h+5),title,fontname='Bold',fontsize=14,lineheight=1.2)>=0;y+=h+18
    p.insert_text((42,y),'Supplementary Methods and data coverage',fontname='Bold',fontsize=12);y+=30
    for head,txt in METHODS:
        hh=text_height(head,511,11,True);th=text_height(txt,511,10.5)
        if y+hh+th+12>795:p=font_page(doc);y=38
        toc.append([1,head,len(doc)])
        assert p.insert_textbox(fitz.Rect(42,y,553,y+hh+4),head,fontname='Bold',fontsize=11,lineheight=1.2)>=0;y+=hh+2
        assert p.insert_textbox(fitz.Rect(42,y,553,y+th+6),txt,fontname='Body',fontsize=10.5,lineheight=1.2)>=0;y+=th+13
    for start in [0,4]:
        p=font_page(doc);p.insert_text((42,42),'Supplementary Table 1 | Data coverage'+(' (continued)' if start else ''),fontname='Bold',fontsize=12)
        toc.append([1,'Data coverage'+(' continued' if start else ''),len(doc)])
        xs=[42,96,330,553];y=67
        for j,s in enumerate(['Figures','Variables, sources and periods','Spatial support, samples and leads']):
            p.draw_rect(fitz.Rect(xs[j],y,xs[j+1],y+32),fill=(.93,.95,.97),color=None)
            p.insert_textbox(fitz.Rect(xs[j]+5,y+5,xs[j+1]-5,y+32),s,fontname='Bold',fontsize=9.5)
        y+=35
        for row in COVERAGE[start:start+4]:
            a,v,src,period,region,n,leads=row
            texts=[a,v+'\n'+src+'\n'+period,region+'\n'+n+'\n'+leads]
            heights=[text_height(t,xs[j+1]-xs[j]-10,9.5) for j,t in enumerate(texts)];h=max(heights)+12
            assert y+h<800,(a,y+h)
            for j,t in enumerate(texts):
                spare=p.insert_textbox(fitz.Rect(xs[j]+5,y+5,xs[j+1]-5,y+h),t,fontname='Body',fontsize=9.5,lineheight=1.16)
                assert spare>=0,(a,j,spare)
            p.draw_line((42,y+h),(553,y+h),color=(.8,.8,.8),width=.4);y+=h+3
    number_pages(doc);doc.set_toc(toc);doc.save(ROOT/'Supplementary_Methods_and_coverage.pdf',garbage=4,deflate=True);doc.close()
    pd.DataFrame(COVERAGE,columns=['Figures','Variables','Data_sources','Period','Domain','Samples','Leads_and_missingness']).to_csv(ROOT/'data_coverage.csv',index=False)
    (ROOT/'Supplementary_Methods.md').write_text('\n\n'.join('# '+h+'\n\n'+t for h,t in METHODS)+'\n')

def package(records):
    # Caption revisions can be packaged without rerunning unchanged diagnostics.
    for r in records:
        c=CAPTIONS[r['number']]
        for k in ['title','role','caption','statistics']:r[k]=c[k]
        if r['number']==2 and 'slope difference is' not in r['caption']:
            q=pd.read_csv(ROOT/'derived_data/S02_slope_difference.csv').iloc[0]
            r['caption']+=f' The El Niño minus La Niña slope difference is {q.estimate:.3f} AUC °C⁻¹ (95% year-block interval {q.low:.3f} to {q.high:.3f}); '+('the interval includes zero.' if q.low<=0<=q.high else 'the interval excludes zero under this resampling scheme.')
        (ROOT/'captions'/f'{r["id"]}.md').write_text(f'# Supplementary Fig. {r["number"]} | {r["title"]}\n\n{r["caption"]}\n\n{r["statistics"]}\n')
    (ROOT/'figure_registry.json').write_text(json.dumps(records,indent=2,ensure_ascii=False)+'\n')
    caption_pages(records);methods_pdf(records)
    doc=fitz.open()
    for r in records:
        with fitz.open(OUT/(r['id']+'.pdf')) as q:doc.insert_pdf(q)
    doc.set_toc([[1,f'Supplementary Fig. {r["number"]} | {r["title"]}',i+1] for i,r in enumerate(records)])
    doc.save(ROOT/'Supplementary_Figures_only.pdf',garbage=4,deflate=True);doc.close()
    doc=fitz.open();toc=[]
    for n in ['Supplementary_Methods_and_coverage.pdf','Supplementary_Figures_with_captions.pdf']:
        with fitz.open(ROOT/n) as q:
            offset=len(doc);doc.insert_pdf(q);toc.extend([[lev,t,pg+offset] for lev,t,pg in q.get_toc()])
    for p in doc:
        p.add_redact_annot(fitz.Rect(280,p.rect.height-32,320,p.rect.height-7),fill=(1,1,1))
        p.apply_redactions(images=0,graphics=0)
    number_pages(doc)
    doc.set_toc(toc);doc.save(ROOT/'Supplementary_Information.pdf',garbage=4,deflate=True);doc.close()
    overview=fitz.open()
    for start in range(0,len(records),4):
        p=overview.new_page(width=842,height=842 if len(records)-start>2 else 421);p.insert_font(fontname='Bold',fontfile=str(FONT/'LiberationSans-Bold.ttf'))
        for j,r in enumerate(records[start:start+4]):
            x=20+(j%2)*421;y=24+(j//2)*421
            assert p.insert_textbox(fitz.Rect(x,y-10,x+380,y+32),'S'+str(r['number'])+' | '+r['title'],fontname='Bold',fontsize=10)>=0
            with fitz.open(OUT/(r['id']+'.pdf')) as art:p.show_pdf_page(fitz.Rect(x,y+35,x+380,y+385),art,0)
    overview.save(ROOT/'SI_overview.pdf',deflate=True)
    for i,p in enumerate(overview):p.get_pixmap(dpi=100).save(ROOT/'qa'/f'overview_{i+1}.png')
    overview.close()
    # Exact numerical crosswalk to the current main Figure4, not the superseded six-basin figure.
    reg=pd.read_csv(ROOT/'derived_data/S15_main_Fig4_region_crosswalk.csv',keep_default_na=False)
    comp=pd.read_csv(ROOT/'derived_data/S15_main_Fig4_component_crosswalk.csv',keep_default_na=False)
    assert np.isclose(reg.Net_shortfall.sum(),.10651437039268086,atol=1e-12)
    assert abs(comp.net.sum()-reg.Net_shortfall.sum())<1e-12
    rows=[dict(figure=r['number'],title=r['title'],role=r['role'],main_link=r['main_figure'],minimum_font=r['min_font_pt'],clipped=r['outside_text']) for r in records]
    pd.DataFrame(rows).to_csv(ROOT/'figure_roles_and_QA.csv',index=False)
    (ROOT/'STATISTICS_AND_REPRODUCIBILITY.md').write_text('# Statistics and reproducibility\n\n'+ '\n\n'.join(f'## Supplementary Fig. {r["number"]}\n\n{r["caption"]}\n\n{r["statistics"]}\n\nInputs: '+', '.join(r['sources']) for r in records))
    (ROOT/'README.md').write_text('# Revised supplementary figures\n\n16 figure PDFs, with separate full-size vector artwork and complete same-page captions. Earlier versions and main figures are unchanged.\n\n- Supplementary_Information.pdf: Methods, coverage table and 16 captioned figures.\n- Supplementary_Figures_with_captions.pdf: 16 figures, one figure and caption per page.\n- Supplementary_Figures_only.pdf: artwork only.\n- individual_with_captions/: individual captioned PDFs.\n- figures/: individual PDF/SVG/300-dpi PNG.\n- source_data/, derived_data/, source_manifest.json: data and lineage.\n- qa/ and REVIEWER_RESPONSE.md: review disposition and checks.\n\nReproduce: OPENBLAS_NUM_THREADS=1 /home/luoq/miniconda3/envs/ocean_sst/bin/python revise_si.py\n\nScientific status: readability and regional consistency have been revised; shared SST extraction and tropical common-cohort limitations remain substantive, and are stated in Supplementary Methods. This package does not declare those input issues resolved.\n')
    with zipfile.ZipFile(ROOT/'Individual_SI_PDFs.zip','w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['figures','individual_with_captions']:
            for f in sorted((ROOT/folder).glob('*.pdf')):z.write(f,str(f.relative_to(ROOT)))
    print('PACKAGED',len(records),'supplementary figures',flush=True)
    (ROOT/'README.md').write_text(
        '# Supplementary spatial-support revision\n\n'
        '17 individually exported figures and an integrated PDF. S1c and S16 are synchronized with '
        'Figure1_neighbourhood_skill_20261111_v10; new S17 compares two time windows and two neighbourhood sizes. '
        'S1a,b and all other supplementary figure artwork are unchanged from v2.\n\n'
        '- Supplementary_Information.pdf: Methods, data coverage and 17 captioned figures.\n'
        '- Supplementary_Figures_with_captions.pdf: 17 figures with complete same-page captions.\n'
        '- Supplementary_Figures_only.pdf: artwork only.\n'
        '- Individual_SI_PDFs.zip: artwork and captioned individual PDFs.\n'
        '- source_data/SPATIAL10/: frozen current spatial metrics.\n'
        '- derived_data/S17_*.csv: duration and neighbourhood sensitivity.\n\n'
        'Reproduce: run the sibling Figure1_neighbourhood_skill_20261111_v10/calculate.py, '
        'then revise_si.py --only 1 16 17 and validate_release.py with the ocean_sst Python environment.\n\n'
        'The new spatial maps use 18 months and same-basin 11-degree neighbourhoods on a 1-degree display grid. '
        'No interpolation fills undefined cells. Overlapping pools do not provide independent one-degree resolution. '
        'The original archived SST initialization limitations and tropical-input limitations remain stated in Methods.\n'
    )
    (ROOT/'README.md').write_text(
        '# Supplementary figure revision v5: compact layout\n\n'
        'All 17 figures regenerated with reader-facing captions and supplementary methods. '
        'Earlier releases and main Figures 1–4 are unchanged. S1c now displays all seven regions '
        'and all 63 lead values; all previously computed numerical tables are unchanged.\n\n'
        '- Supplementary_Information.pdf: 5 methods/coverage pages and 17 captioned figures.\n'
        '- Supplementary_Figures_with_captions.pdf: 17 full-width figures and captions; page heights follow content, avoiding unused A4 foot space.\n'
        '- Supplementary_Figures_only.pdf: vector artwork only.\n'
        '- Individual_SI_PDFs.zip: individual artwork and captioned PDFs.\n'
        '- SI_overview.pdf: compact contact sheets.\n'
        '- source_manifest.json, source_data/, derived_data/: input hashes and numerical products.\n'
        '- LAYOUT_REVISION.md: current changes; qa/layout_comparison.csv: dimensions before and after.\n'
        '- REVIEWER_RESPONSE.md: item-by-item response; SUBMISSION_BLOCKERS.md: remaining scientific checks.\n\n'
        'S15 now matches Figure4_net_normalized_20261111_v27: +120.790563% and −20.790563% '
        'of the signed net map difference. S5 masks three unverified wind-stress lead-9 values, '
        'while retaining all 81 original records in source_data.\n\n'
        'Reproduce: OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mhw_si_v5 '
        '/home/luoq/miniconda3/envs/ocean_sst/bin/python revise_si.py; then run validate_release.py and finalize_release.py.\n\n'
        'Technical checks are not scientific submission clearance. SST initialization-selection '
        'and tropical-input cohort issues remain explicitly documented; no unsupported causal '
        'interpretation or independent-event claim is introduced.\n')
