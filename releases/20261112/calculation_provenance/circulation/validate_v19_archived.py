"""Independent numerical, boundary, provenance and exported-artifact checks."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import xarray as xr
import fitz
from PIL import Image

ROOT=Path(__file__).resolve().parent
BASE=Path('/data3/luoq/FEIS-archive/legacy/p03-ocean-sst-ensemble/figure3_fig4_basin_masks_20261031_v3')
DATA=ROOT/'source_data'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def corr(a,b,mask,lat):
    valid=mask & np.isfinite(a) & np.isfinite(b)
    w=np.broadcast_to(np.cos(np.deg2rad(lat))[:,None],a.shape)[valid].astype(float)
    x=np.asarray(a,dtype=float)[valid];y=np.asarray(b,dtype=float)[valid]
    x=x-np.sum(w*x)/w.sum();y=y-np.sum(w*y)/w.sum()
    return np.sum(w*x*y)/np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))

checks={}
PARENT=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures/Figure3_other_historical_anomalies_20261104_v18')
parent_manifest=json.loads((PARENT/'manifest.json').read_text())
checks['v18_all_original_files_unchanged']=all(sha(PARENT/e['path'])==e['sha256'] for e in parent_manifest)
for unchanged in ['statistics.json','source_data/Figure3_Z200_fields.nc','source_data/Figure3b_all_samples.csv','source_data/seven_ocean_regions.nc']:
    checks['unchanged_v18:'+unchanged]=sha(ROOT/unchanged)==sha(PARENT/unchanged)
V11=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures/Figure3_spatiotemporal_correlations_20261103_v11')
checks['v11_preserved_not_used_as_baseline']=all(sha(V11/e['path'])==e['sha256'] for e in json.loads((V11/'manifest.json').read_text()))
checks['old_versions_unchanged']=all(sha(Path(x['path']))==x['sha256'] for x in json.loads((ROOT/'baseline_snapshot.json').read_text()))
published_parent=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures/Figure3_fig4_basin_masks_20261031_v3/figures/Figure3.pdf')
checks['requested_published_v3_matches_preserved_parent']=sha(published_parent)==sha(BASE/'figures/Figure3.pdf')
d=xr.load_dataset(DATA/'Figure3_Z200_fields.nc')
old=xr.load_dataset(BASE/'source_data/Figure3_Z200_fields.nc')
rain=xr.load_dataset(DATA/'Figure3_precipitation_fields.nc')
g=xr.load_dataset(DATA/'Figure3_geostrophic_and_height_fields.nc')
masks=xr.load_dataset(DATA/'seven_ocean_regions.nc').mask
om=xr.load_dataset(BASE/'source_data/seven_ocean_regions.nc').mask
for v in ['observed','forecast','forecast_error','canonical','model_forecast']:
    checks['original_Z200_values_preserved:'+v]=np.array_equal(d[v],old[v])
checks['mask_regions_nonoverlap']=bool((masks.astype(int).sum('region')<=1).all())
checks['ocean_union_unchanged']=bool((masks.any('region')==om.any('region')).all())
checks['Indian_Ocean_unchanged']=bool((masks.sel(region='IO')==om.sel(region='IO')).all())
for group in [['TP','NP','SP'],['TA','NA','SA']]:
    checks['basin_union_unchanged:'+group[0]]=bool((masks.sel(region=group).any('region')==om.sel(region=group).any('region')).all())
for region in ['TP','TA']:
    m=masks.sel(region=region)
    checks['tropical_rows_exact:'+region]=m.lat[m.any('lon')].values.tolist()==list(range(-23,24))
for region in ['NP','NA']:
    m=masks.sel(region=region)
    checks['northern_rows_exact:'+region]=float(m.lat[m.any('lon')].min())==24
for region in ['SP','SA']:
    m=masks.sel(region=region)
    checks['southern_rows_exact:'+region]=float(m.lat[m.any('lon')].max())==-24
checks['recipient_masks_exact']=bool((d.atmospheric_mask==masks.sel(region=d.region)).all())
checks['rain_TP_mask_exact']=bool((rain.source_mask==masks.sel(region='TP').sel(lat=rain.lat,lon=rain.lon)).all())
checks['rain_new_band_47_rows']=rain.sizes['lat']==47 and rain.lat.values.tolist()==list(range(-23,24))
models=['COLA-RSMAS-CCSM4','NASA-GEOSS2S','GFDL-SPEAR','ERA5']
for model in models:
    oldp=xr.load_dataset(BASE/f'cache/precipitation_{model}_DJF.nc')
    newp=xr.load_dataset(ROOT/f'cache/precipitation_{model}_DJF.nc').sel(lat=oldp.lat)
    key='observed' if model=='ERA5' else 'forecast'
    checks['precipitation_original_band_exact:'+model]=np.array_equal(newp[key],oldp[key])
    if model!='ERA5':
        counts=xr.load_dataset(ROOT/f'cache/precipitation_{model}_DJF.nc').baseline_count
        checks['baseline_counts_29_29_28:'+model]=bool((counts.values==np.array([29,29,28])[None,:]).all())
        audit=pd.read_csv(DATA/f'coordinate_audit_precipitation_{model}.csv')
        checks['exact_future_leads:'+model]=np.allclose(audit.exact_L,audit.lead+.5,rtol=0,atol=0)
        checks['fixed_members:'+model]=audit.member_first.eq(1).all() and audit.member_last.nunique()==1
checks['forecast_height_error_identity']=np.allclose(g.forecast_height_climatology_adjusted-g.observed_height,g.forecast_error,rtol=0,atol=1e-10)
checks['climatology_equal_month_weight']=np.array_equal(g.climatology,xr.load_dataset(ROOT/'cache/z200_ERA5_DJF.nc').climatology.astype(float).mean('month'))
df=pd.read_csv(DATA/'Figure3b_all_samples.csv',keep_default_na=False)
rows=[]
for event in d.event.values:
    for lead in d.lead.values:
        rr=corr(rain.forecast.sel(event=event,lead=lead).values,rain.observed.sel(event=event).values,rain.source_mask.values,rain.lat.values)
        for region in d.region.values:
            zr=corr(d.forecast.sel(event=event,lead=lead).values,d.observed.sel(event=event).values,masks.sel(region=region).values,d.lat.values)
            rows.append({'Event':str(event),'Lead':int(lead),'Region':str(region),'Rain_r':rr,'Z200_r':zr})
joined=df.merge(pd.DataFrame(rows),on=['Event','Lead','Region'],suffixes=('_saved','_check'),validate='one_to_one')
max_rain=float(abs(joined.Rain_r_saved-joined.Rain_r_check).max())
max_z=float(abs(joined.Z200_r_saved-joined.Z200_r_check).max())
checks['all_144_rows']=len(df)==len(joined)==144
checks['independent_rain_correlation_recheck']=max_rain<1e-6
checks['independent_height_correlation_recheck']=max_z<1e-6
checks['three_events_not_144_independent']=df.Event.nunique()==3
checks['leads_exact_1_8']=df.Lead.unique().tolist()==list(range(1,9))
stats=json.loads((ROOT/'statistics.json').read_text())
checks['pooled_r_recomputed']=abs(np.corrcoef(df.Rain_error,df.Z200_error)[0,1]-stats['pooled_r_descriptive'])<1e-12
checks['pooled_r_changed_after_new_masks']=abs(stats['pooled_r_descriptive']-json.loads((BASE/'statistics.json').read_text())['pooled_r_descriptive'])>.001
change=pd.read_csv(DATA/'SI_regional_skill_change.csv',keep_default_na=False).replace('',np.nan)
checks['lead9_left_missing']=change.loc[change.Lead==9,'Delta_r'].isna().all()
checks['old_heatmap_not_generated']=not (DATA/'Figure3c_skill_change.csv').exists()
profiles=pd.read_csv(DATA/'Figure3cd_transects.csv')
parent_profiles=pd.read_csv(PARENT/'source_data/Figure3cd_transects.csv')
checks['northern_five_curves_and_errors_preserved']=np.allclose(profiles.loc[profiles.latitude==45,parent_profiles.columns],parent_profiles.loc[parent_profiles.latitude==45],rtol=0,atol=1e-10)
checks['southern_section_replaced_with_50S']=set(profiles.latitude)=={45,-50}
parent_geo=xr.load_dataset(PARENT/'source_data/Figure3_geostrophic_and_height_fields.nc')
checks['full_height_and_wind_fields_unchanged']=all(np.array_equal(g[v],parent_geo[v],equal_nan=True) for v in parent_geo.data_vars)
expected_cf=d.forecast.sel(event=['1997/98','2015/16']).astype(float).mean('lead',skipna=False).mean('event',skipna=False)
checks['canonical_forecast_equal_events_and_leads']=np.allclose(g.canonical_forecast_anomaly,expected_cf,rtol=0,atol=1e-12)
checks['canonical_forecast_common_climatology']=np.allclose(g.canonical_forecast_height_climatology_adjusted,g.climatology+expected_cf,rtol=0,atol=1e-10)
cf_values=[float(g.canonical_forecast_height_climatology_adjusted.sel(lat=r.latitude,lon=r.longitude)) for r in profiles.itertuples()]
checks['all_720_new_canonical_forecast_values']=len(cf_values)==720 and np.allclose(cf_values,profiles.forecast_canonical,rtol=0,atol=1e-10)
checks['canonical_error_definition']=np.allclose(profiles.forecast_canonical_error,profiles.forecast_canonical-profiles.canonical,rtol=0,atol=1e-10)
for series,field in [('climatology','climatology'),('canonical','canonical_height'),('observed','observed_height'),('forecast','forecast_height_climatology_adjusted')]:
    vals=[float(g[field].sel(lat=r.latitude,lon=r.longitude)) for r in profiles.itertuples()]
    checks['all_720_section_values:'+series]=len(vals)==720 and np.allclose(vals,profiles[series],rtol=0,atol=1e-10)
checks['continuous_longitude_261_nodes_per_section']=all(len(p)==261 and np.array_equal(np.diff(p.unwrapped_longitude),np.ones(260)) for _,p in profiles[profiles.unwrapped_longitude<=380].groupby('latitude'))
diagnostics=json.loads((ROOT/'diagnostic_QA.json').read_text())
for k,v in diagnostics.items():
    if isinstance(v,bool):checks['geostrophic:'+k]=v
concentration=pd.read_csv(DATA/'SI_prespecified_error_concentration.csv')
checks['concentration_defined_by_area_weighted_share']=np.allclose(concentration.concentration,concentration.error_fraction/concentration.area_fraction)
checks['concentration_fractions_bounded']=concentration.error_fraction.between(0,1).all() and concentration.area_fraction.between(0,1).all()
r=json.loads((ROOT/'render_QA.json').read_text())
checks['no_text_outside']=not r['outside_canvas_text']
checks['font_floor']=r['minimum_text_pt']>=5
checks['heatmap_removed_from_layout']=r['old_heatmap_removed']
checks['requested_latitude_bands']=r['bar_latitude_bands']==[[30,60],[-60,-40]]
checks['reference_lines_unchanged']=r['map_reference_latitudes']==[45,-50]
checks['six_defined_pairs']=r['transects_replaced'] and r['bars_per_panel']==6 and r['pair_positions_per_panel']==6 and r['undefined_pairs_per_panel']==0 and r['bar_zero_baseline']
checks['wind_uniform_scale']=r['wind_scale_ms_per_inch']==75
table=pd.read_csv(DATA/'Figure3cd_el_nino_anomaly_correlations.csv')
elnino=xr.load_dataset(DATA/'Figure3cd_other_el_nino_fields.nc')
history=xr.load_dataset(DATA/'Figure3cd_other_historical_fields.nc')
comparisons=xr.load_dataset(DATA/'Figure3cd_el_nino_comparison_fields.nc')
anomaly=xr.load_dataset(DATA/'Figure3cd_anomaly_fields.nc')
band_masks=xr.load_dataset(DATA/'Figure3cd_band_masks.nc').mask.astype(bool)
fields={'climatology':'climatology','observed_canonical':'canonical_height','forecast_canonical':'canonical_forecast_height_climatology_adjusted','observed_2023':'observed_height','forecast_2023':'forecast_height_climatology_adjusted'}
for name,field in fields.items():
    checks['anomaly_equals_full_minus_climatology:'+name]=np.allclose(anomaly[name],g[field]-g.climatology,rtol=0,atol=1e-10)
checks['climatology_anomaly_exact_zero']=np.array_equal(anomaly.climatology.values,np.zeros_like(anomaly.climatology.values))
checks['seasonal_not_spatiotemporal']=all(da.dims==('lat','lon') for da in anomaly.data_vars.values())
lon,lat=np.meshgrid(g.lon,g.lat)
for name,lo,hi,nrows in [('NH',30,60,31),('SH',-60,-40,21)]:
    expected=((lon>=120)|(lon<=20))&(lat>=lo)&(lat<=hi)
    checks['band_mask:'+name]=np.array_equal(expected,band_masks.sel(band=name))
    checks['complete_grid_support:'+name]=int(expected.sum())==nrows*261
    rows=table[table.band==name]
    checks['six_distinct_pairs:'+name]=len(rows)==6 and not rows.duplicated(['anchor','target']).any()
    checks['all_comparison_states:'+name]=set(rows.anchor)=={'forecast_2023','forecast_canonical'} and set(rows.target)=={'other_el_nino','observed_canonical','observed_2023'}
    for row in rows.itertuples():
        # Independent route: subtract v10 complete height's own common baseline.
        x=(g[fields[row.anchor]]-g.climatology).values[expected]
        y=elnino.other_el_nino.values[expected] if row.target=='other_el_nino' else (g[fields[row.target]]-g.climatology).values[expected]
        w=np.cos(np.deg2rad(lat[expected]));w=w/w.sum()
        # Independent weighted-vector covariance implementation.
        a=(x-np.dot(w,x))*np.sqrt(w);bvec=(y-np.dot(w,y))*np.sqrt(w)
        if row.target=='climatology':
            checks[f'undefined_value:{name}:{row.anchor}']=np.isnan(row.r) and np.linalg.norm(bvec)==0 and row.status=='undefined_zero_climatology_anomaly'
        else:
            verified=np.dot(a,bvec)/(np.linalg.norm(a)*np.linalg.norm(bvec))
            checks[f'bar_value:{name}:{row.anchor}:{row.target}']=abs(row.r-verified)<1e-12 and row.status=='defined'
defined=table[table.status=='defined']
checks['all_twelve_defined']=len(defined)==12 and table.r.isna().sum()==0
checks['all_defined_values_bounded']=defined.r.between(-1,1).all()
checks['axis_contains_all_values']=defined.r.between(*r['bar_y_limits']).all()
checks['anomaly_field_explicit']=r['bar_field']=='DJF_mean_Z200_anomaly'
checks['no_climatology_placeholders']=r['undefined_rendering']=='none; all six comparisons defined' and not any(t['text'] in ['N/A','Zero anomaly','Climatology'] for t in r['all_text_bounds'])
from calculate_anomaly_correlations import weighted_r
t=np.array([-2.,-1.,0.,1.,2.]);wtest=np.array([1.,2.,3.,2.,1.])
checks['metric_perfect']=abs(weighted_r(t,t,wtest)-1)<1e-14
checks['metric_reversed']=abs(weighted_r(t,-t,wtest)+1)<1e-14
checks['metric_zero_field_undefined']=np.isnan(weighted_r(t,np.zeros_like(t),wtest))
checks['metric_constant_field_undefined']=np.isnan(weighted_r(t,np.full_like(t,5.),wtest))
checks['metric_shift_invariant']=abs(weighted_r(t,t+7,wtest)-1)<1e-14
checks['metric_amplitude_invariant']=abs(weighted_r(t,t*.01,wtest)-1)<1e-14
for label,xx,yy,ww in [('missing',t,np.full_like(t,np.nan),wtest),('one_sample',t[:1],t[:1],wtest[:1]),('invalid_weight',t,t,-wtest)]:
    try:
        weighted_r(xx,yy,ww)
        checks['metric_rejects_'+label]=False
    except ValueError:
        checks['metric_rejects_'+label]=True
amp=pd.read_csv(DATA/'SI_anomaly_amplitude.csv')
ver=pd.read_csv(DATA/'SI_climatology_distance_and_verification.csv')
for row in amp.itertuples():
    mask=band_masks.sel(band=row.band).values
    x=anomaly[row.state].values[mask];w=np.cos(np.deg2rad(lat[mask]));w=w/w.sum()
    mean=float(np.dot(w,x));rms=float(np.sqrt(np.dot(w,x*x)));std=float(np.sqrt(np.dot(w,(x-mean)**2)))
    checks[f'amplitude:{row.band}:{row.state}']=np.allclose([row.mean_anomaly_m,row.rms_anomaly_m,row.spatial_std_anomaly_m],[mean,rms,std],rtol=0,atol=1e-10)
for row in ver.itertuples():
    mask=band_masks.sel(band=row.band).values;w=np.cos(np.deg2rad(lat[mask]));w=w/w.sum()
    f=anomaly['forecast_'+row.event].values[mask];o=anomaly['observed_'+row.event].values[mask]
    mse=np.dot(w,(f-o)**2);oms=np.dot(w,o*o)
    checks[f'distance:{row.band}:{row.event}']=np.allclose([row.forecast_observed_rmse_m,row.climatology_observed_rmse_m,row.mse_skill_vs_climatology],[np.sqrt(mse),np.sqrt(oms),1-mse/oms],rtol=0,atol=1e-10)
# Independent historical-composite reconstruction and source lineage.
monthly=xr.load_dataset(DATA/'Figure3cd_historical_monthly_Z200.nc').height
clim=xr.load_dataset(ROOT/'cache/z200_ERA5_DJF.nc').climatology
selection=pd.read_csv(DATA/'historical_winter_selection.csv')
included=[y for y in range(1992,2022) if y not in [1997,2015]]
checks['exact_28_other_winters']=selection.loc[selection.included,'start_year'].tolist()==included and len(included)==28
checks['two_canonical_exclusions']=selection.loc[~selection.included,'start_year'].tolist()==[1997,2015]
checks['all_dates_before_2023']=all(t<'2023-01' for ts in selection.targets for t in ts.split(';'))
checks['31_years_93_months']=monthly.sizes['target']==93 and np.isfinite(monthly).all()
recomputed=[]
for year in range(1992,2022):
    # Explicitly sum three anomalies, then divide by three.
    acc=np.zeros_like(history.other_historical.values,dtype=float)
    for yy,mm in [(year,12),(year+1,1),(year+1,2)]:
        acc+=(monthly.sel(target=f'{yy}-{mm:02d}').values-clim.sel(month=mm).values).astype(float)
    acc=acc/3
    checks['DJF_recompute:'+str(year)]=np.allclose(acc,history.seasonal_anomaly.sel(start_year=year),rtol=0,atol=1e-12)
    if year in included:recomputed.append(acc)
verified=np.sum(np.stack(recomputed),axis=0)/28
checks['composite_before_correlation']=np.allclose(verified,history.other_historical,rtol=0,atol=1e-12)
checks['comparison_reference_matches']=np.array_equal(comparisons.other_el_nino,elnino.other_el_nino)
for name in ['forecast_2023','forecast_canonical','observed_2023','observed_canonical']:
    checks['old_comparison_preserved:'+name]=np.array_equal(comparisons[name],anomaly[name])
oldtable=pd.read_csv(PARENT/'source_data/Figure3cd_anomaly_correlations.csv')
pair=table.merge(oldtable[oldtable.target!='climatology'],on=['band','anchor','target'],suffixes=('_new','_old'))
checks['all_eight_existing_correlations_unchanged']=len(pair)==8 and np.allclose(pair.r_new,pair.r_old,rtol=0,atol=1e-12)
canon_check=history.seasonal_anomaly.sel(start_year=[1997,2015]).mean('start_year')
checks['canonical_raw_reconstruction_agrees']=np.allclose(canon_check,anomaly.observed_canonical,rtol=0,atol=1e-4)
for month in [12,1,2]:
    dates=[f'{year}-{month:02d}' for year in range(1992,2021) if not(year==2018 and month==2)]
    cc=monthly.sel(target=dates).mean('target').values
    checks['fixed_baseline_reconstructed:'+str(month)]=np.allclose(cc,clim.sel(month=month),rtol=0,atol=1e-3)
# Raw archive check uses direct geopotential conversion, independent of reader helper.
audit=pd.read_csv(DATA/'historical_ERA5_source_audit.csv')
for row in audit.itertuples():
    path=Path(row.path)
    with xr.open_dataset(path) as raw:
        zz=raw.z.sel(valid_time=raw.valid_time.dt.month.isin([1,2,12]),pressure_level=200.,
                     latitude=monthly.lat.values,longitude=monthly.lon.values).values/9.80665
        observed=monthly.sel(target=row.months.split(';')).values
        checks['raw_ERA5_exact:'+str(row.year)]=np.array_equal(zz,observed)
        checks['selected_array_hash:'+str(row.year)]=hashlib.sha256(zz.tobytes()).hexdigest()==row.selected_height_sha256
    checks['raw_file_metadata:'+str(row.year)]=path.stat().st_size==row.bytes and path.stat().st_mtime_ns==row.mtime_ns
amplitudes=pd.read_csv(DATA/'Figure3cd_historical_reference_amplitude.csv')
for row in amplitudes.itertuples():
    m=band_masks.sel(band=row.band).values;x=verified[m];w=np.cos(np.deg2rad(lat[m]))
    mu=np.average(x,weights=w);std=np.sqrt(np.average((x-mu)**2,weights=w))
    checks['historical_reference_nonconstant:'+row.band]=std>1e-5 and abs(std-row.spatial_std_m)<1e-10
checks['symmetric_axis_no_negative_clipping']=r['bar_y_limits']==[-1.2,1.2]
checks['bar_value_labels_clear_of_axes']=r['bar_value_labels_inside_axes']
# Active v19 reference: explicit paper-list selection, independently reconstructed.
event_source=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures/Design_20261027_v5/source_data/Figure1_source_events.csv')
checks['event_list_matches_paper']=sha(DATA/'Figure1_source_events.csv')==sha(event_source)
paper=pd.read_csv(event_source)
paper_years=paper.Event.str.split('/').str[0].astype(int).tolist()
chosen=[y for y in paper_years if 1992<=y<=2021 and y not in [1997,2015]]
ledger=pd.read_csv(DATA/'other_el_nino_event_selection.csv')
checks['exact_six_other_el_nino_events']=chosen==[1994,2002,2004,2006,2009,2018]
checks['selection_ledger_exact']=ledger.loc[ledger.included,'start_year'].tolist()==chosen
checks['ledger_covers_all_candidates_and_paper_events']=ledger.start_year.tolist()==sorted(set(range(1992,2022))|set(paper_years))
checks['1991_exclusion_disclosed']=ledger.loc[ledger.start_year==1991,'reason'].iloc[0]=='excluded: outside unchanged v18 historical window'
checks['canonical_and_focal_excluded']=not ledger.loc[ledger.start_year.isin([1997,2015,2023]),'included'].any()
direct=[]
for year in chosen:
    arr=sum((monthly.sel(target=f'{yy}-{mm:02d}').values-clim.sel(month=mm).values).astype(float)
            for yy,mm in [(year,12),(year+1,1),(year+1,2)])/3
    direct.append(arr)
    checks['active_season_exact:'+str(year)]=np.allclose(arr,elnino.seasonal_anomaly.sel(start_year=year),rtol=0,atol=1e-12)
ref=np.stack(direct).sum(axis=0)/len(chosen)
checks['active_composite_before_correlation']=np.allclose(ref,elnino.other_el_nino,rtol=0,atol=1e-12)
checks['no_season_dropped']=elnino.start_year.values.tolist()==chosen and bool(np.isfinite(elnino.seasonal_anomaly).all())
checks['unchanged_monthly_archive']=sha(DATA/'Figure3cd_historical_monthly_Z200.nc')==sha(PARENT/'source_data/Figure3cd_historical_monthly_Z200.nc')
checks['unchanged_seasonal_archive']=sha(DATA/'Figure3cd_other_historical_fields.nc')==sha(PARENT/'source_data/Figure3cd_other_historical_fields.nc')
checks['unchanged_anomaly_baseline']=sha(ROOT/'cache/z200_ERA5_DJF.nc')==sha(PARENT/'cache/z200_ERA5_DJF.nc')
checks['new_label_only']=r['bar_targets']==['other_el_nino','observed_canonical','observed_2023']
parent_table=pd.read_csv(PARENT/'source_data/Figure3cd_historical_anomaly_correlations.csv')
same=table.merge(parent_table,on=['band','anchor','target'],suffixes=('_new','_old'))
checks['v18_eight_unchanged_values']=len(same)==8 and np.allclose(same.r_new,same.r_old,rtol=0,atol=1e-12)
active_amp=pd.read_csv(DATA/'Figure3cd_el_nino_reference_amplitude.csv')
for row in active_amp.itertuples():
    mask=band_masks.sel(band=row.band).values
    x=ref[mask];w=np.cos(np.deg2rad(lat[mask]));mu=np.average(x,weights=w)
    sd=np.sqrt(np.average((x-mu)**2,weights=w))
    checks['active_reference_nonconstant:'+row.band]=sd>1e-5 and abs(sd-row.spatial_std_m)<1e-10
b=r['map_axes_bounds']
checks['map_alignment']=np.ptp([x[0] for x in b])<1e-12 and np.ptp([x[0]+x[2] for x in b])<1e-12
p=fitz.open(ROOT/'figures/Figure3.pdf')
checks['one_page']=len(p)==1
checks['physical_183x160mm']=abs(p[0].rect.width*25.4/72-183)<.02 and abs(p[0].rect.height*25.4/72-160)<.02
checks['vector_drawing']=len(p[0].get_drawings())>100
checks['fonts_embedded']=all(p.extract_font(f[0])[3] for f in p.get_page_fonts(0))
im=Image.open(ROOT/'figures/Figure3.png')
original_image=Image.open(PARENT/'figures/Figure3.png')
new_pixels=np.asarray(im);old_pixels=np.asarray(original_image)
height,width=new_pixels.shape[:2]
checks['panel_a_pixel_identical']=np.array_equal(new_pixels[:,:int(.70*width)],old_pixels[:,:int(.70*width)])
checks['panel_b_pixel_identical']=np.array_equal(new_pixels[:int(.37*height),int(.705*width):],old_pixels[:int(.37*height),int(.705*width):])
checks['png_600dpi']=abs(im.size[0]-183/25.4*600)<=1 and abs(im.size[1]-160/25.4*600)<=1
checks['svg_exists']=(ROOT/'figures/Figure3.svg').stat().st_size>10000
p[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(ROOT/'figures/Figure3_pdf_preview.png')
checks={k:bool(v) for k,v in checks.items()}
report={'pass':all(checks.values()),'checks':checks,'correlation_max_absolute_recheck_difference':{'rain':max_rain,'height':max_z},
        'limits':'Other El Nino is a six-event observed DJF anomaly composite selected from the paper event list within the unchanged v18 period, not climatology. Composite correlation is not amplitude, physical distance, causal evidence or same-event forecast verification.',
        'visual_review':'VISUAL_QA.md'}
(ROOT/'QA.json').write_text(json.dumps(report,indent=2)+'\n')
manifest=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size}
          for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name not in ['manifest.json','publication_QA.json'] and '__pycache__' not in p.parts]
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'pass':report['pass'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],
                  'correlation_recheck':report['correlation_max_absolute_recheck_difference']},indent=2))
assert report['pass'],[k for k,v in checks.items() if not v]
