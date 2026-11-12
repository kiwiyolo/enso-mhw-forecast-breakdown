"""Independent numerical, source-artwork and final-page validation for v16."""
from pathlib import Path
import csv, math, json, hashlib
import numpy as np
import pandas as pd
import fitz

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with (DATA/'pattern_correlations.csv').open() as stream:
    raw=list(csv.DictReader(stream))
lookup={(r['Process'],r['Event'],int(r['Lead'])):float(r['Pattern_correlation']) for r in raw}
assert len(raw)==len(lookup)==81
processes=['SST','Zonal wind stress','Convection proxy']
events=['1997/98','2015/16','2023/24']
means={p:{e:math.fsum(lookup[p,e,l] for l in range(1,10))/9 for e in events} for p in processes}
def loss(h,t):
    return 100*(h-t)/h if h>0 else float('nan')
expected={p:loss((means[p][events[0]]+means[p][events[1]])/2,means[p][events[2]]) for p in processes}
summary=pd.read_csv(DATA/'relative_loss_summary.csv')
for r in summary.itertuples():
    assert math.isclose(r.Relative_decline_percent,expected[r.Process],abs_tol=1e-12)
    assert math.isclose(r.Relative_to_SST,expected[r.Process]/expected['SST'],abs_tol=1e-12)
curves=pd.read_csv(DATA/'plotted_b_curves.csv')
assert len(curves)==27 and not curves.duplicated(['Process','Lead']).any()
for r in curves.itertuples():
    historical=[lookup[r.Process,e,r.Lead] for e in events[:2]]
    assert abs(r.historical_mean-sum(historical)/2)<1e-14
    assert abs(r.historical_min-min(historical))<1e-14
    assert abs(r.historical_max-max(historical))<1e-14
    assert abs(r.target-lookup[r.Process,events[2],r.Lead])<1e-14
bars=json.loads((DATA/'plotted_bars.json').read_text())
points=pd.read_csv(DATA/'plotted_reference_points.csv')
assert len(bars)==3 and len(points)==6
assert not points.duplicated(['Process','Historical_event']).any()
for p in processes:
    row=next(r for r in bars if r['Process']==p)
    values=[loss(means[p][e],means[p][events[2]]) for e in events[:2]]
    assert abs(row['height']-expected[p])<1e-12
    assert abs(row['error_low']-min(values))<1e-12
    assert abs(row['error_high']-max(values))<1e-12
    assert 0<row['error_low']<=row['height']<=row['error_high']<65
for r in points.itertuples():
    assert abs(r.y-loss(means[r.Process][r.Historical_event],means[r.Process][events[2]]))<1e-12

lineage=json.loads((DATA/'input_lineage.json').read_text())
assert all(sha(Path(r['path']))==r['sha256'] for r in lineage)
render=json.loads((ROOT/'render_QA.json').read_text())
base=fitz.open(render['baseline']);out=fitz.open(ROOT/'figures/Figure2.pdf');page=out[0]
pt=72/25.4
assert len(out)==1
assert abs(page.rect.width/pt-183)<.01 and abs(page.rect.height/pt-107)<.01
region=fitz.Rect(0,0,123.8*pt,99.8*pt)
def pixels(page):
    p=page.get_pixmap(dpi=300,clip=region,alpha=False)
    return np.frombuffer(p.samples,np.uint8).reshape(p.height,p.width,3)
before=pixels(base[0]);after=pixels(page)
delta=np.abs(before.astype(int)-after.astype(int))
changed=int(np.any(delta,axis=-1).sum())
assert changed/before.shape[0]/before.shape[1]<1e-4 and delta.max()<=8
def images(p):
    return sorted((i['digest'].hex(),i['width'],i['height'],tuple(i['bbox']))
        for i in p.get_image_info(hashes=True) if fitz.Rect(i['bbox']).intersects(region))
old=images(base[0]);new=images(page)
assert len(old)==len(new)>0
for a,b in zip(old,new):
    assert a[:3]==b[:3]
    assert np.allclose(a[3],b[3],atol=1e-4,rtol=0)
spans=[s for block in page.get_text('dict')['blocks'] if 'lines' in block
       for line in block['lines'] for s in line['spans'] if s['text'].strip()]
assert min(s['size'] for s in spans)>=5
assert all(fitz.Rect(s['bbox']) in page.rect for s in spans)
text=page.get_text()
for r in bars:assert r['label'] in text
assert all(label in text for label in ['a','b','c','Relative fidelity loss','Correlation decline (%)'])
assert 'Greater relative loss of atmospheric pattern fidelity' not in text
# Numerical parity against the previous release, independent of file formatting.
previous=Path(render['baseline']).parent.parent
for name in ['pattern_correlations.csv','relative_loss_summary.csv','relative_loss_by_lead.csv','historical_reference_sensitivity.csv']:
    pd.testing.assert_frame_equal(pd.read_csv(DATA/name),pd.read_csv(previous/'source_data'/name),check_exact=True)
report=dict(status='PASS for scoped presentation revision; legacy comparison limits retained',
    independently_recomputed=True,coefficient_count=81,plotted_b_curve_rows=27,
    bars=3,reference_sensitivity_points=6,error_bars='Two historical-reference extrema, not confidence intervals',
    previous_values_exactly_preserved=True,a_source_image_bytes_identical=True,a_physical_scale_unchanged=True,
    a_antialias_changed_pixels=changed,a_max_channel_difference=int(delta.max()),
    source_hashes_unchanged=True,minimum_font_pt=min(s['size'] for s in spans),all_text_inside_page=True,
    page_dimensions_mm=[183,107],no_significance_test=True,common_window_comparability_not_established=True,
    displayed_values={r['Process']:r['height'] for r in bars})
(ROOT/'validation_QA.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
