"""v21: retain numerical checks and verify layout-only content invariance."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import xarray as xr
import fitz
from PIL import Image
from calculate_anomaly_correlations import weighted_r

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data'
PARENT=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures/Figure3_global_el_nino_anomaly_correlations_20261104_v20')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks={}
manifest=json.loads((PARENT/'manifest.json').read_text())
checks['parent_all_hashes_preserved']=all(sha(PARENT/e['path'])==e['sha256'] for e in manifest)
checks['parent_v20_checks_passed']=json.loads((PARENT/'QA.json').read_text())['pass']
# The v19 source data/cache/dependencies must remain byte-identical.
for e in manifest:
    path=e['path']
    if path.startswith(('source_data/','cache/','dependencies/')) or path in ['statistics.json','el_nino_anomaly_statistics.json','global_anomaly_statistics.json','baseline_snapshot.json']:
        checks['input_preserved:'+path]=sha(ROOT/path)==e['sha256']
checks['earlier_releases_preserved']=all(sha(Path(x['path']))==x['sha256'] for x in json.loads((ROOT/'baseline_snapshot.json').read_text()))
f=xr.load_dataset(DATA/'Figure3cd_el_nino_comparison_fields.nc')
domain=xr.load_dataset(DATA/'Figure3c_global_domain.nc')
table=pd.read_csv(DATA/'Figure3c_global_anomaly_correlations.csv')
stats=json.loads((ROOT/'global_anomaly_statistics.json').read_text())
checks['native_latitudes']=f.lat.values.tolist()==list(range(-60,61))
checks['all_360_unique_longitudes']=f.lon.values.tolist()==list(range(360))
checks['all_nodes_included']=domain.mask.shape==(121,360) and bool((domain.mask==1).all())
checks['same_coordinates']=np.array_equal(domain.lat,f.lat) and np.array_equal(domain.lon,f.lon)
checks['six_unique_pairs']=len(table)==6 and not table.duplicated(['anchor','target']).any()
checks['correct_anchors']=set(table.anchor)=={'forecast_2023','forecast_canonical'}
checks['correct_targets']=set(table.target)=={'other_el_nino','observed_canonical','observed_2023'}
checks['whole_domain_43560_cells']=table.n_grid_cells.eq(43560).all()
checks['not_time_space_or_full_height']=all(v.dims==('lat','lon') for v in f.data_vars.values()) and table.field_type.eq('DJF_mean_Z200_anomaly').all()
checks['no_missing_nodes']=all(bool(np.isfinite(v).all()) for v in f.data_vars.values())
checks['finite_bounded_results']=np.isfinite(table.r).all() and table.r.between(-1,1).all()
weights=np.cos(np.deg2rad(f.lat.values))[:,None]*np.ones((1,360))
checks['cos_latitude_weights']=np.array_equal(weights,domain.area_weight.values)
w=weights.ravel()/weights.sum()

# Independent reconstruction from preserved height arrays and monthly observations.
g=xr.load_dataset(DATA/'Figure3_geostrophic_and_height_fields.nc')
mapping={'forecast_2023':'forecast_height_climatology_adjusted',
         'forecast_canonical':'canonical_forecast_height_climatology_adjusted',
         'observed_2023':'observed_height','observed_canonical':'canonical_height'}
reconstructed={}
for name,key in mapping.items():
    reconstructed[name]=(g[key]-g.climatology).values
    checks['anomaly_identity:'+name]=np.allclose(reconstructed[name],f[name],rtol=0,atol=1e-10)
months=xr.load_dataset(DATA/'Figure3cd_historical_monthly_Z200.nc').height
clim=xr.load_dataset(ROOT/'cache/z200_ERA5_DJF.nc').climatology
seasons=[]
selected=[1994,2002,2004,2006,2009,2018]
ledger=pd.read_csv(DATA/'other_el_nino_event_selection.csv')
checks['six_el_nino_events_unchanged']=ledger.loc[ledger.included,'start_year'].tolist()==selected
paper=pd.read_csv(DATA/'Figure1_source_events.csv')
years=paper.Event.str.split('/').str[0].astype(int).tolist()
checks['selection_matches_paper']=[y for y in years if 1992<=y<=2021 and y not in [1997,2015]]==selected
for year in selected:
    s=sum((months.sel(target=f'{yy}-{mm:02d}').values-clim.sel(month=mm).values).astype(float)
          for yy,mm in [(year,12),(year+1,1),(year+1,2)])/3
    seasons.append(s)
reconstructed['other_el_nino']=np.stack(seasons).mean(axis=0)
checks['reference_composite_before_correlation']=np.allclose(reconstructed['other_el_nino'],f.other_el_nino,rtol=0,atol=1e-12)
checks['no_polar_extrapolation']=stats['latitude']==[-60,60] and 'not pole-to-pole' in stats['coverage']
deviations=[]
for row in table.itertuples():
    x=reconstructed[row.anchor].ravel();y=reconstructed[row.target].ravel()
    a=np.sqrt(w)*(x-np.dot(w,x));b=np.sqrt(w)*(y-np.dot(w,y))
    rr=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    deviations.append(abs(rr-row.r))
    checks[f'independent_correlation:{row.anchor}:{row.target}']=abs(rr-row.r)<1e-12
    direct=weighted_r(f[row.anchor].values.ravel(),f[row.target].values.ravel(),weights.ravel())
    checks[f'CSV_matches_active_metric:{row.anchor}:{row.target}']=abs(direct-row.r)<1e-12
    report_row=next(r for r in stats['results'] if r['anchor']==row.anchor and r['target']==row.target)
    checks[f'JSON_CSV_agreement:{row.anchor}:{row.target}']=abs(report_row['r']-row.r)<1e-12

amps=pd.read_csv(DATA/'Figure3c_global_anomaly_amplitudes.csv')
for row in amps.itertuples():
    x=f[row.state].values.ravel();mean=np.dot(w,x)
    checks['amplitude:'+row.state]=np.allclose([row.mean_m,row.rms_m,row.spatial_std_m],
        [mean,np.sqrt(np.dot(w,x*x)),np.sqrt(np.dot(w,(x-mean)**2))],rtol=0,atol=1e-10)
individual=pd.read_csv(DATA/'SI_global_individual_el_nino_correlations.csv')
checks['all_12_sensitivity_pairs']=len(individual)==12 and not individual.duplicated(['anchor','start_year']).any()
for row in individual.itertuples():
    v=seasons[selected.index(row.start_year)]
    rr=weighted_r(f[row.anchor].values.ravel(),v.ravel(),weights.ravel())
    checks[f'individual_event:{row.anchor}:{row.start_year}']=abs(rr-row.r)<1e-12
x=np.array([-2.,-1.,0.,1.,2.]);wt=np.array([1.,2.,3.,2.,1.])
checks['metric_self']=abs(weighted_r(x,x,wt)-1)<1e-14
checks['metric_inverse']=abs(weighted_r(x,-x,wt)+1)<1e-14
checks['metric_affine_invariance']=abs(weighted_r(x,3*x+7,wt)-1)<1e-14
checks['constant_correlation_undefined']=np.isnan(weighted_r(x,np.zeros_like(x),wt))
for name,yy in [('missing',np.full_like(x,np.nan)),('single_sample',np.array([0.]))]:
    try:
        weighted_r(x,yy,wt)
        checks['reject_'+name]=False
    except ValueError:checks['reject_'+name]=True

r=json.loads((ROOT/'render_QA.json').read_text())
checks['single_c_no_d']=r['correlation_panels']==1 and not any(t['text']=='d' for t in r['all_text_bounds'])
checks['six_rendered_bars']=r['bars_per_panel']==6 and r['pair_positions_per_panel']==6
checks['plotted_global_extent']=r['bar_latitude_bands']==[[-60,60]] and r['bar_longitude_extent']==[0,360]
checks['anomaly_label']=r['bar_field']=='DJF_mean_Z200_anomaly'
checks['correct_display_targets']=r['bar_targets']==['other_el_nino','observed_canonical','observed_2023']
checks['geographic_scope_visible']=any(t['text']=='60°S–60°N · All longitudes' for t in r['all_text_bounds'])
checks['values_match_direct_labels']=all(any(t['text']==f'{row.r:.3f}' for t in r['all_text_bounds']) for row in table.itertuples())
checks['axis_contains_values']=table.r.between(*r['bar_y_limits']).all()
checks['zero_baseline_no_truncation']=r['bar_zero_baseline'] and r['bar_y_limits']==[0,1.05]
checks['labels_inside_axes']=r['bar_value_labels_inside_axes']
checks['no_text_off_canvas']=not r['outside_canvas_text']
checks['font_floor']=r['minimum_text_pt']>=5
bounds=r['map_axes_bounds']
checks['maps_aligned']=np.ptp([b[0] for b in bounds])<1e-12 and np.ptp([b[0]+b[2] for b in bounds])<1e-12
im=np.asarray(Image.open(ROOT/'figures/Figure3.png'))
before=np.asarray(Image.open(PARENT/'figures/Figure3.png'))
h,width=im.shape[:2]
checks['width_preserved_height_reduced']=im.shape[1]==before.shape[1] and im.shape[0]<before.shape[0]
checks['png_600dpi']=abs(width-183/25.4*600)<=1 and abs(h-140/25.4*600)<=1
pdf=fitz.open(ROOT/'figures/Figure3.pdf')
checks['single_page']=len(pdf)==1
checks['183x140mm']=abs(pdf[0].rect.width*25.4/72-183)<.02 and abs(pdf[0].rect.height*25.4/72-140)<.02
checks['vector_drawing']=len(pdf[0].get_drawings())>100
checks['fonts_embedded']=all(pdf.extract_font(font[0])[3] for font in pdf.get_page_fonts(0))
checks['svg_exists']=(ROOT/'figures/Figure3.svg').stat().st_size>10000
pdf[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(ROOT/'figures/Figure3_pdf_preview.png')

# Cosmetic-only invariance: no displayed content or scientific transform changed.
from collections import Counter
import ast
old_render=json.loads((PARENT/'render_QA.json').read_text())
checks['all_displayed_text_preserved']=Counter(t['text'] for t in r['all_text_bounds'])==Counter(t['text'] for t in old_render['all_text_bounds'])
for key in ['map_limits','map_interpolation','scatter_points','wind_scale_ms_per_inch',
            'wind_arrow_reference_ms','arrows_per_map','geostrophic_sample','map_reference_latitudes',
            'bar_latitude_bands','bar_longitude_extent','bar_field','bar_anchors','bar_targets',
            'pair_positions_per_panel','bars_per_panel','bar_y_limits']:
    checks['render_contract_unchanged:'+key]=r[key]==old_render[key]
for path in PARENT.glob('calculate*.py'):
    checks['calculation_script_unchanged:'+path.name]=sha(ROOT/path.name)==sha(path)
# AST equality for physical field geometry/derived-wind plotting routines.
def functions(path):
    return {n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
old_funcs=functions(PARENT/'plot_figure3.py');new_funcs=functions(ROOT/'plot_figure3.py')
for name in ['basin_lines','domain_lines','add_wind']:
    checks['scientific_geometry_unchanged:'+name]=ast.dump(old_funcs[name])==ast.dump(new_funcs[name])
checks['map_group_wider']=all(b[2]>o[2] for b,o in zip(r['map_axes_bounds'],old_render['map_axes_bounds']))
new_gaps=[(r['map_axes_bounds'][i][1]-(r['map_axes_bounds'][i+1][1]+r['map_axes_bounds'][i+1][3]))*140 for i in [0,1]]
old_gaps=[(old_render['map_axes_bounds'][i][1]-(old_render['map_axes_bounds'][i+1][1]+old_render['map_axes_bounds'][i+1][3]))*160 for i in [0,1]]
checks['map_row_gaps_smaller']=all(0<n<o for n,o in zip(new_gaps,old_gaps))
checks['right_panel_edges_aligned']=np.allclose(np.array(r['scatter_axes_bounds'])[[0,2]],np.array(r['bar_axes_bounds'])[[0,2]],rtol=0,atol=1e-12)
checks['all_fonts_at_least6pt']=r['minimum_text_pt']>=6
checks['canvas_height_reduced12_5percent']=r['height_mm']==140 and r['width_mm']==183
# Detect substantial text bounding-box overlaps, keeping all annotations.
overlaps=[]
texts=r['all_text_bounds']
for i,a in enumerate(texts):
    xa,ya,xb,yb=a['bounds_px']
    for b in texts[i+1:]:
        xc,yc,xd,yd=b['bounds_px']
        ix=min(xb,xd)-max(xa,xc);iy=min(yb,yd)-max(ya,yc)
        if ix>1 and iy>1:
            overlaps.append({'first':a['text'],'second':b['text'],'intersection_px':[ix,iy]})
checks['no_substantial_text_overlap']=not overlaps
(ROOT/'layout_QA.json').write_text(json.dumps({'text_overlap_candidates':overlaps,'map_gaps_mm_new':new_gaps,
    'map_gaps_mm_old':old_gaps,'content_text_identical':checks['all_displayed_text_preserved'],
    'canvas_reduction_percent':12.5,'pixel_identity_not_applicable':'Layout dimensions intentionally changed; source/metric/text/geometry invariance checked instead'},indent=2)+'\n')

checks={k:bool(v) for k,v in checks.items()}
report={'pass':all(checks.values()),'checks':checks,'maximum_independent_r_difference':max(deviations),
        'scope':'All longitudes,60S-60N; same v19 DJF anomalies and event composites; not pole-to-pole',
        'visual_review':'VISUAL_QA.md'}
(ROOT/'QA.json').write_text(json.dumps(report,indent=2)+'\n')
out=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size}
     for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['manifest.json','publication_QA.json'] and '__pycache__' not in p.parts]
(ROOT/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'pass':report['pass'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],
                  'max_r_difference':max(deviations)},indent=2))
assert report['pass'],[k for k,v in checks.items() if not v]
