"""Independent read-only data parity and exported-figure checks."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import xarray as xr
import fitz

ROOT=Path(__file__).resolve().parent
checks=[]
def check(name,condition,detail=None):
    checks.append({'check':name,'pass':bool(condition),'detail':detail})
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'source_manifest.json').read_text())
for row in manifest:
    check('unchanged source: '+row['frozen'],sha(Path(row['source']))==row['sha256']==sha(ROOT/row['frozen']))
qa=json.loads((ROOT/'render_QA.json').read_text())
for name,q in qa.items():
    check(name+' no out-of-canvas text',not q['outside_text'])
    check(name+' minimum font >=6pt',q['min_font_pt']>=6)
    regular=[v['font_pt'] for v in q['texts'] if v['text'] not in ['a','b','c']]
    check(name+' regular font <=7pt',max(regular)<=7.01)
    doc=fitz.open(ROOT/'figures'/f'{name}.pdf'); page=doc[0]
    check(name+' width 180mm',abs(page.rect.width*25.4/72-180)<.01)
    check(name+' text editable',len(page.get_text())>100)
    fonts=page.get_fonts(full=True)
    check(name+' embedded TTF',all(doc.extract_font(f[0])[3] for f in fonts),[f[3] for f in fonts])
    text=page.get_text().lower()
    check(name+' obsolete labels absent',not any(t in text for t in ['pattern fidelity','comparable events','precipitation']))
check('Figure1 all 10 event bars',qa['Figure1']['events']==10 and qa['Figure1']['all_event_whiskers'])
check('Figure1 no arrow',qa['Figure1']['arrow_removed'])
check('Figure1 exact time alignment',qa['Figure1']['shared_x_exact'])
check('Figure1 original gap',np.isclose(qa['Figure1']['residual'],-0.10904973718077582,rtol=0,atol=1e-14))
with fitz.open(ROOT/'figures/Figure1.pdf') as doc:
    text=doc[0].get_text()
    check('Figure1 Expected and delta labels absent','Expected' not in text and 'Δ' not in text)
    check('Figure1 requested legend present','Canonical strong El Nino' in text)
    check('Figure1 observed value annotation absent','Observed' not in text and '0.596' not in text)
check('Figure1 narrower a and wider b',qa['Figure1']['main_panel_width_ratio']==[60,94])
check('Figure1 expected marker and comparison line removed',qa['Figure1']['expected_marker_removed'] and qa['Figure1']['comparison_line_removed'])
check('Figure1 top and bottom alignment',qa['Figure1']['main_and_strip_bottom_mm']==[75,75] and qa['Figure1']['main_and_series_top_mm']==[9,9])
with fitz.open(ROOT/'figures/Figure2.pdf') as doc:
    text=doc[0].get_text()
    check('Figure2 both canonical labels present',text.count('Canonical strong El Nino')==2 and 'Earlier events' not in text)
    check('Figure2 concrete colourbar label present','Forecast − observed (% of observed anomaly magnitude)' in text and 'Signal-relative bias' not in text)
    spans=[s for b in doc[0].get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
    tops=[s['bbox'][1] for s in spans if s['text'] in ['Spatial correlation','2023–2024']]
    check('Figure2 aligned column title baselines',len(tops)==2 and abs(tops[0]-tops[1])<0.2,tops)
with fitz.open(ROOT/'figures/Figure3.pdf') as doc:
    text=doc[0].get_text()
    check('Figure3 requested descriptions removed',all(s not in text for s in ['Contours:','Dotted lines:','Tropical convection error']))
    check('Figure3 domain annotation removed','All longitudes' not in text)
    check('Figure3 restored correlation axes',all(s in text for s in ['Tropical convection correlation (r)','Z200 forecast correlation (r)']))
caption=(ROOT/'Captions.md').read_text()
check('New label defined in caption','Forecast − observed (% of observed anomaly magnitude)' in caption and 'spatial RMS of the observed anomaly' in caption)
previous=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures/Figure1-3_Nature_layout_20261108_v6')
old_report=json.loads((previous/'render_QA.json').read_text())
for key in old_report['Figure2']:
    if key not in ['texts','axis_labels']:
        check('Figure2 v5 preserved '+key,qa['Figure2'][key]==old_report['Figure2'][key])
for number in [1]:
    for ext in ['pdf','svg','png']:
        name=f'Figure{number}.{ext}'
        check('v5 unchanged export '+name,sha(ROOT/'figures'/name)==sha(previous/'figures'/name))
check('Figure3 removed explanations retained in caption',all(s in caption for s in ['horizontal axis is tropical convection correlation','45°N','50°S','40 m','geostrophic wind errors','DJF','60°S–60°N','all longitudes']))
parent=json.loads((ROOT/'provenance/parent_v1_render_QA.json').read_text())
last=json.loads((ROOT/'provenance/parent_v4_render_QA.json').read_text())
for name,q in qa.items():
    check(name+' compact label and tick pads',q['axis_label_pad_pt']==1 and q['tick_label_pad_pt']==1)
    check(name+' axis labels clear of ticks',all(0.75<=v['tick_gap_pt']<=1.25 for v in q['axis_labels']),q['axis_labels'])
    check(name+' axis labels not inside adjacent panels',all(not v['overlaps_other_axes'] for v in q['axis_labels']))
    check(name+' width retained; height compacted or retained',q['width_mm']==last[name]['width_mm'] and q['height_mm']<=last[name]['height_mm'])
    for key,value in last[name].items():
        if key not in ['texts','outside_text','height_mm','axes_positions','axis_labels','main_panel_width_ratio','right_column_edges_mm']:
            check(name+' v4 retained: '+key,q.get(key)==value)
check('Figure1 panel gap reduced 18 to 10mm',qa['Figure1']['panel_gap_mm']==10)
check('Figure1 strip gap reduced 3 to 1.5mm',qa['Figure1']['series_strip_gap_mm']==1.5)
check('Figure2 column gap reduced 20 to 11.5mm',qa['Figure2']['map_right_column_gap_mm']==11.5)
check('Figure2 map row and column gaps compacted',qa['Figure2']['map_row_gap_mm']==2 and qa['Figure2']['map_column_gap_mm']==1.5)
check('Figure2 b-c gap reduced 13 to 10mm',qa['Figure2']['b_c_frame_gap_mm']==10)
p=qa['Figure2']['axes_positions']
check('Figure2 right axes edges aligned',all(np.allclose([a[0],a[0]+a[2]],[128/180,178/180]) for a in p[7:]))
check('Figure2 map colourbar edges aligned',np.isclose(p[6][0],p[0][0]) and np.isclose(p[6][0]+p[6][2],p[1][0]+p[1][2]))
with fitz.open(ROOT/'figures/Figure3.pdf') as doc:
    text=doc[0].get_text()
    check('Figure3 explicit physical quantity','200-hPa geopotential height' in text and 'Forecast − observed (m)' in text and 'Z200 error (m)' not in text)
for key,value in old_report['Figure3'].items():
    if key not in ['texts','axis_labels']:
        check('Figure3 retained '+key,qa['Figure3'][key]==value)
for name in qa:
    check(name+' reduced canvas height',qa[name]['height_mm']<parent[name]['height_mm'])
for key in ['displayed_curves','bars','stipple_counts','map_limits']:
    check('Figure2 v1 parity: '+key,qa['Figure2'][key]==parent['Figure2'][key])
for key in ['bar_values','map_limits','vector_locations','vector_scale','contour_levels','scatter_r']:
    check('Figure3 v1 parity: '+key,qa['Figure3'][key]==parent['Figure3'][key])

path=ROOT/'source_data/Fig2'
old=pd.read_csv(path/'plotted_b_curves.csv')
new=pd.DataFrame(qa['Figure2']['displayed_curves'],columns=['Process','Lead','historical_mean','historical_min','historical_max','target'])
check('Figure2 all 27 displayed lead summaries match',np.allclose(old.iloc[:,2:],new.iloc[:,2:],rtol=0,atol=1e-14))
input=pd.read_csv(path/'pattern_correlations.csv')
sensitivity=pd.read_csv(path/'historical_reference_sensitivity.csv')
for j,process in enumerate(['SST','Zonal wind stress','Convection proxy']):
    means=input[input.Process==process].groupby('Event').Pattern_correlation.mean()
    h=means[['1997/98','2015/16']].mean();v=100*(h-means['2023/24'])/h
    refs=100*(means[['1997/98','2015/16']]-means['2023/24'])/means[['1997/98','2015/16']]
    actual=qa['Figure2']['bars'][j]
    check('Figure2 '+process+' bar and interval recomputed',np.allclose([v,refs.min(),refs.max()],
        [actual['value'],actual['low'],actual['high']],rtol=0,atol=1e-11))
check('Figure2 81 coefficients / 6 reference points',qa['Figure2']['curve_coefficients']==81 and qa['Figure2']['reference_points']==6)
check('Figure2 native map grid unchanged',qa['Figure2']['map_shape']=={'lat':51,'lon':121})
check('Figure2 colour range unchanged',qa['Figure2']['map_limits']==[-200,200])
ag=xr.load_dataset(path/'Figure3_model_sign_agreement.nc')
for name,n in qa['Figure2']['stipple_counts'].items():
    check('Figure2 '+name+' stippling retained',int(ag[name+'_sign_agreement_mask'].astype(bool).isel(lat=slice(None,None,3),lon=slice(None,None,3)).sum())==n)

path=ROOT/'source_data/Fig3'
samples=pd.read_csv(path/'Figure3b_all_samples.csv',keep_default_na=False)
displayed=pd.DataFrame(qa['Figure3']['plotted_correlations'])
check('Figure3b all sample identifiers unchanged',displayed[['Event','Lead','Region']].equals(samples[['Event','Lead','Region']]))
check('Figure3b actual r values drawn',np.allclose(displayed[['Rain_r','Z200_r']],samples[['Rain_r','Z200_r']],rtol=0,atol=1e-14))
check('Figure3b r equals one minus former error',np.allclose(displayed[['Rain_r','Z200_r']],1-samples[['Rain_error','Z200_error']],rtol=0,atol=1e-14))
check('Figure3b negative correlations retained',qa['Figure3']['negative_z200_samples']==int((samples.Z200_r<0).sum()))
check('Figure3b pooled correlation invariant',abs(qa['Figure3']['transformed_r']-qa['Figure3']['scatter_r'])<1e-12)
envelope=pd.read_csv(path/'Figure3b_event_bootstrap_envelope.csv')
expected=pd.DataFrame({'x':1-envelope.x,'fit':1-envelope['fit'],'lower':1-envelope.upper,'upper':1-envelope.lower}).sort_values('x')
check('Figure3b regression envelope transformed exactly',np.allclose(pd.DataFrame(qa['Figure3']['correlation_envelope'])[expected.columns],expected,rtol=0,atol=1e-14))
check('Figure3b all values and envelope within axis limits',qa['Figure3']['scatter_x_limits'][0]<samples.Rain_r.min() and qa['Figure3']['scatter_x_limits'][1]>samples.Rain_r.max() and qa['Figure3']['scatter_y_limits'][0]<samples.Z200_r.min() and qa['Figure3']['scatter_y_limits'][1]>max(samples.Z200_r.max(),expected.upper.max()))
fields=xr.load_dataset(path/'Figure3cd_el_nino_comparison_fields.nc')
table=pd.read_csv(path/'Figure3c_global_anomaly_correlations.csv')
weight=np.broadcast_to(np.cos(np.deg2rad(fields.lat.values))[:,None],(121,360))
for row in table.itertuples():
    a=fields[row.anchor].values;b=fields[row.target].values
    a=a-np.average(a,weights=weight);b=b-np.average(b,weights=weight)
    corr=np.sum(weight*a*b)/np.sqrt(np.sum(weight*a*a)*np.sum(weight*b*b))
    check('Figure3c recomputed '+row.anchor+' / '+row.target,abs(corr-row.r)<1e-12,float(corr))
check('Figure3 all 144 points / 6 bars',qa['Figure3']['scatter_points']==144 and len(qa['Figure3']['bar_values'])==6)
check('Figure3 legends inside axes',qa['Figure3']['internal_legends_inside_axes'])
check('Figure3 no scatter points covered by legends',qa['Figure3']['scatter_points_under_legends']==0)
check('Figure3 native values and colour limits',qa['Figure3']['source_grid_unchanged'] and qa['Figure3']['map_limits']==[160,160,100])
check('Figure3 vector sampling unchanged',qa['Figure3']['vector_samples_per_map']==120 and qa['Figure3']['vector_scale']==75)
check('Figure3 bars start at zero',qa['Figure3']['bar_zero_baseline'] and qa['Figure3']['bar_ylim'][0]==0)
check('Figure3 same projection as Fig4',qa['Figure3']['projection'].startswith('Robinson, central_longitude=-10'))
combined=fitz.open(ROOT/'figures/Figures1-3.pdf')
check('combined PDF 3 pages',len(combined)==3)
result={'status':'PASS presentation and parity checks' if all(c['pass'] for c in checks) else 'FAIL',
        'n_checks':len(checks),'failed':[c for c in checks if not c['pass']], 'checks':checks,
        'scientific_status':'Inherited limitations retained; not a new scientific or submission-readiness approval.',
        'font_exception':'Liberation Sans substituted transparently; Arial/Helvetica unavailable.'}
(ROOT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
if result['failed']:raise SystemExit(1)
