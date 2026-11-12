# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# All panels: retained project build_si.py / render.py -> param inherit.
# Heatmaps: heatmap/plot_comparison.py -> param inherit (cell labels, missing fill).
# Trends: LineTrend/plot_trend.py -> param inherit (direct comparisons, thin curves).
# Generic assets use different estimands, per-column normalization or publication
# counts. Retain project data and figure types; do not execute their mock data.
# The three verbatim style baselines execute in base_si.py before panel code.
import base_si as b
from base_si import *
from matplotlib import patches
from submission_captions import CAPTIONS, METHODS, COVERAGE
import importlib.util,zipfile

WORK=Path('/data3/luoq/FEIS-archive/legacy/p03-ocean-sst-ensemble')
SPATIAL=WORK/'Figure1_neighbourhood_skill_20261111_v10'
BRIDGE=BASE/'Figure4_clear_skill_legend_20261110_v22'
mpl.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
    'axes.labelpad':4,'xtick.major.pad':3,'ytick.major.pad':3})
SEQ=LinearSegmentedColormap.from_list('skill_blue',['#F7FBFF','#9ECAE1','#2166AC','#08306B'])
NAME={'TP':'Tropical Pacific','NP':'North Pacific','SP':'South Pacific','TA':'Tropical Atlantic',
      'NA':'North Atlantic','SA':'South Atlantic','IO':'Indian Ocean'}
original_finish=b.finish

def freeze(src,group):
    if group=='SPATIAL':group='SPATIAL10'
    dst=SRC/group/Path(src).name;dst.parent.mkdir(exist_ok=True)
    if not dst.exists():shutil.copy2(src,dst)
    check('source hash '+str(dst.relative_to(ROOT)),sha(src)==sha(dst))
    MANIFEST.append(dict(source=str(src),frozen=str(dst.relative_to(ROOT)),sha256=sha(dst)))
    CURRENT.append(str(dst.relative_to(ROOT)));return dst

def newfig(rows=1,cols=2,h=140,**kw):
    height=150 if rows==2 else 128
    f,axs=plt.subplots(rows,cols,figsize=(180/25.4,height/25.4),squeeze=False,**kw)
    f.subplots_adjust(left=.10,right=.97,bottom=.14,top=.91,wspace=.42,hspace=.55)
    return f,axs
b.newfig=newfig
def label(ax,c):
    ax.text(-.08,1.06,c,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
b.label=label

def heat(ax,a,x,y,vmin=-1,vmax=1,cmap=DIV,annot=True):
    a=np.asarray(a,float);cmap=cmap.copy();cmap.set_bad('#DEDEDE')
    im=ax.imshow(np.ma.masked_invalid(a),vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto',interpolation='none')
    ax.set_xticks(range(len(x)),x);ax.set_yticks(range(len(y)),y);ax.tick_params(length=0)
    for sp in ax.spines.values():sp.set_visible(False)
    for i,j in np.ndindex(a.shape):
        if not np.isfinite(a[i,j]):ax.text(j,i,'NA',ha='center',va='center',fontsize=7.5,color='#555555')
        elif annot:
            rgb=cmap((a[i,j]-vmin)/(vmax-vmin))[:3];lum=np.dot(rgb,[.299,.587,.114])
            ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=7.5,color='white' if lum<.5 else DARK)
    return im
b.heat=heat

def fig01():
    d=csv('OLD','Supplementary_Figure1_source_events.csv');m=csv('OLD','Supplementary_Figure1_monthly_t1_t9.csv')
    h=d[d.Event!='2023/24'];x=h.Peak_Nino34.to_numpy();y=h.Mean_RMSE.to_numpy();fit=stats.linregress(x,y)
    xx=np.linspace(.55,2.75,200);yy=fit.intercept+fit.slope*xx
    se=np.sqrt(np.sum((y-fit.intercept-fit.slope*x)**2)/(len(x)-2))
    pi=stats.t.ppf(.975,len(x)-2)*se*np.sqrt(1+1/len(x)+(xx-x.mean())**2/np.sum((x-x.mean())**2))
    f=plt.figure(figsize=(180/25.4,136/25.4))
    ax=f.add_axes([.095,.64,.30,.30]);label(ax,'a');ax.fill_between(xx,yy-pi,yy+pi,color='#ECECEC');ax.plot(xx,yy,c=DARK,lw=.9)
    for _,z in d.iterrows():ax.scatter(z.Peak_Nino34,z.Mean_RMSE,s=35 if z.Event=='2023/24' else 18,c=RED if z.Event=='2023/24' else DARK,marker='*' if z.Event=='2023/24' else 'o',zorder=3)
    ax.set(xlabel='Peak Niño3.4 anomaly (°C)',ylabel='MHW-intensity RMSE (°C)',title='Event-mean error')
    ax=f.add_axes([.51,.64,.47,.30]);label(ax,'b');ax.plot(pd.to_datetime(m.time),m.RMSE,color=DARK,lw=.7)
    for ev,c in zip(EVENTS,COLORS):
        q=d[d.Event==ev].iloc[0];ax.axvspan(pd.Timestamp(q.Start_Month),pd.Timestamp(q.End_Month),color=c,alpha=.13)
    ax.set(xlabel='Year',ylabel='MHW-intensity RMSE (°C)',title='Monthly error')
    ds=xr.load_dataset(freeze(SPATIAL/'source_data/canonical_relative_changes.nc','SPATIAL'))
    summary=pd.read_csv(freeze(SPATIAL/'source_data/regional_changes.csv','SPATIAL'),keep_default_na=False)
    lead=pd.read_csv(freeze(SPATIAL/'source_data/regional_changes_by_lead.csv','SPATIAL'),keep_default_na=False)
    grid=np.full((180,360),np.nan);ii=np.rint(ds.lat.values+89.5).astype(int);jj=np.rint(ds.lon.values-.5).astype(int)
    grid[ii,jj]=ds.rmse_increase.values;v,lon=add_cyclic_point(grid,coord=np.arange(.5,360.))
    ax=f.add_axes([.025,.17,.67,.35],projection=PROJ);f.text(.025,.52,'c',fontweight='bold',fontsize=10)
    im=ax.pcolormesh(lon,np.arange(-89.5,90),v,transform=PC,cmap=DIV,vmin=-2,vmax=2,rasterized=True,shading='auto',antialiased=False)
    b.mapframe(ax);ax.set_facecolor('#DADADA')
    import cartopy.feature as cf
    ax.add_feature(cf.LAND.with_scale('110m'),facecolor='#F0F0F0',edgecolor='none',zorder=3)
    selected=REG
    for r in selected:
        mask=np.zeros_like(grid);mask[ii,jj]=ds.region.values==REG.index(r);vv,ll=add_cyclic_point(mask,coord=np.arange(.5,360.))
        ax.contour(ll,np.arange(-89.5,90),vv,levels=[.5],colors=DARK,linewidths=.5,linestyles='--',transform=PC,zorder=4)
    for r,x,y in [('TP',205,0),('NP',205,38),('SP',205,-36),('NA',320,38),('TA',330,0),('SA',335,-36),('IO',75,-25)]:
        ax.text(x,y,r,transform=PC,ha='center',va='center',fontsize=7.5,fontweight='bold',zorder=5,
                bbox=dict(facecolor='white',alpha=.72,edgecolor='none',pad=.5))
    ca=f.add_axes([.10,.113,.52,.016]);cb=f.colorbar(im,cax=ca,orientation='horizontal',extend='both');cb.set_label('MHW-intensity RMSE increase (°C)')
    ax=f.add_axes([.77,.17,.21,.35]);change_bars(ax,summary,lead,'RMSE',selected)
    ax.set_title('All seven regions',fontsize=8);ax.set_xlabel('RMSE increase (°C)')
    allreg=summary[summary.metric=='RMSE'].set_index('region').loc[REG].reset_index()
    table('S01c_all_seven_regions',allreg)
    allleads=lead[(lead.metric=='RMSE')&lead.region.isin(REG)]
    table('S01c_all_seven_regions_by_lead',allleads)
    check('S1c seven regions and 63 lead values',len(allreg)==7 and len(allleads)==63 and (allleads.groupby('region').size()==9).all())
    table('S01_fit',pd.DataFrame([dict(r=fit.rvalue,slope=fit.slope,expected=fit.intercept+fit.slope*d.iloc[-1].Peak_Nino34,observed=d.iloc[-1].Mean_RMSE)]))
    finish(f,1,'', '', 'Fig. 1','')

def fig02():
    d=csv('F1','Figure1_monthly_t1_t9.csv');d['year']=pd.to_datetime(d.time).dt.year;d['amp']=d.Nino34.abs()
    d['phase']=np.where(d.Nino34>=.5,'El Niño',np.where(d.Nino34<=-.5,'La Niña','Neutral'))
    years=d.year.unique();byyear=[np.flatnonzero(d.year==y) for y in years];rng=np.random.default_rng(20261111)
    boots=[]
    for k in range(2000):
        ids=np.concatenate([byyear[i] for i in rng.integers(len(years),size=len(years))]);z=d.iloc[ids];row=[]
        for ph in ['El Niño','La Niña']:
            a=z[z.phase==ph];fit=stats.linregress(a.amp,a.AUC);row.extend([fit.slope,fit.rvalue])
        boots.append(row)
    boot=np.array(boots);f,a=newfig();rows=[]
    for k,(ph,color) in enumerate([('El Niño',RED),('La Niña',BLUE)]):
        z=d[d.phase==ph];fit=stats.linregress(z.amp,z.AUC);lo,hi=np.quantile(boot[:,2*k+1],[.025,.975])
        ax=a[0,k];label(ax,chr(97+k));ax.scatter(z.amp,z.AUC,s=15,c=color,alpha=.5,edgecolors='white',linewidths=.2)
        xx=np.linspace(z.amp.min(),z.amp.max(),100);ax.plot(xx,fit.intercept+fit.slope*xx,c=color,lw=1)
        ax.set(xlabel='Absolute Niño3.4 anomaly (°C)',ylabel='Global MHW forecast skill (AUC)',ylim=(.3,.95),title=ph)
        ax.text(.04,.97,f'r = {fit.rvalue:.2f}\n95% interval: {lo:.2f} to {hi:.2f}',transform=ax.transAxes,va='top',fontsize=8)
        sl,sh=np.quantile(boot[:,2*k],[.025,.975]);rows.append(dict(phase=ph,n_months=len(z),n_years=z.year.nunique(),r=fit.rvalue,r_low=lo,r_high=hi,slope=fit.slope,low=sl,high=sh))
    delta=boot[:,0]-boot[:,2];dl,dh=np.quantile(delta,[.025,.975]);val=rows[0]['slope']-rows[1]['slope']
    table('S02_phase_statistics',pd.DataFrame(rows));table('S02_slope_difference',pd.DataFrame([dict(estimate=val,low=dl,high=dh,bootstrap_repetitions=2000)]));table('S02_monthly_phase_membership',d)
    CAPTIONS[2]['caption']+=f' The El Niño minus La Niña slope difference is {val:.3f} AUC °C⁻¹ (95% year-block interval {dl:.3f} to {dh:.3f}); '+('the interval includes zero.' if dl<=0<=dh else 'the interval excludes zero under this resampling scheme.')
    finish(f,2,'','','Fig. 1','')

def fig05():
    d=csv('F2','pattern_correlations.csv');r=csv('F2','relative_loss_by_lead.csv');f,a=plt.subplots(3,2,figsize=(180/25.4,168/25.4),gridspec_kw={'width_ratios':[1.15,1]})
    d=d.copy()
    unavailable=(d.Process=='Zonal wind stress')&(d.Lead==9)
    d['Missing_reason']=''
    d.loc[unavailable,'Missing_reason']='Exact lead not verified; nearest-lead substitution excluded'
    d.loc[unavailable,'Pattern_correlation']=np.nan
    table('S05_displayed_correlations',d)
    f.subplots_adjust(left=.115,right=.90,bottom=.15,top=.93,wspace=.6,hspace=.82)
    for i,(p,name) in enumerate(zip(['SST','Zonal wind stress','Convection proxy'],['SST','Zonal wind stress','Precipitation (convective proxy)'])):
        z=d[d.Process==p].pivot(index='Event',columns='Lead',values='Pattern_correlation').reindex(EVENTS)
        ax=a[i,0];im=heat(ax,z.values,z.columns,['1997–98','2015–16','2023–24'],0,1,cmap=SEQ,annot=False);label(ax,chr(97+2*i));ax.set(title=name,xlabel='Lead label')
        ax=a[i,1];label(ax,chr(98+2*i))
        for ev,c in zip(EVENTS[:2],COLORS[:2]):
            h=d[(d.Process==p)&(d.Event==ev)].sort_values('Lead');t=d[(d.Process==p)&(d.Event==EVENTS[2])].sort_values('Lead');v=100*(h.Pattern_correlation.to_numpy()-t.Pattern_correlation.to_numpy())/h.Pattern_correlation.to_numpy()
            ax.plot(h.Lead,v,'o-',c=c,ms=3,lw=.8,label='Reference: '+ev)
        ref=z.loc[EVENTS[:2]].mean(axis=0,skipna=False);target=z.loc[EVENTS[2]]
        relative=100*(ref-target)/ref.where(ref>0)
        ax.plot(z.columns,relative,c=RED,lw=1.2,label='Two-event mean');zero(ax)
        ax.set(xlabel='Lead label',ylabel='Correlation decrease (%)',xticks=[1,3,5,7,9])
        if i==0:handles,labels=ax.get_legend_handles_labels()
    f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.015),ncol=3,fontsize=7.5)
    ca=f.add_axes([.955,.15,.013,.78]);cb=f.colorbar(im,cax=ca);cb.ax.yaxis.set_ticks_position('left');cb.set_label('Spatial correlation, r',labelpad=3)
    check('S05 all source rows retained; 78 verified/displayed values',len(d)==81 and d.Pattern_correlation.notna().sum()==78)
    finish(f,5,'','','Fig. 2','')

def fig15():
    main=BASE/'Figure4_net_normalized_20261111_v27'/'source_data'
    reg=pd.read_csv(freeze(main/'regional_map_shortfall.csv','CURRENT4'),keep_default_na=False)
    flow=pd.read_csv(freeze(main/'association_weighted_local_shortfall.csv','CURRENT4'),keep_default_na=False)
    total=reg.Net_shortfall.sum()
    reg['Positive_pct']=100*reg.Positive_shortfall/total
    reg['Negative_pct']=-100*reg.Above_benchmark/total
    reg['Net_pct']=100*reg.Net_shortfall/total
    table('S15_main_Fig4_region_crosswalk',reg)
    comp=flow.groupby('Component')[['Allocated_positive_shortfall','Allocated_above_benchmark']].sum()
    comp['net']=comp.Allocated_positive_shortfall-comp.Allocated_above_benchmark
    comp['Positive_pct']=100*comp.Allocated_positive_shortfall/total
    comp['Negative_pct']=-100*comp.Allocated_above_benchmark/total
    comp['Net_pct']=100*comp.net/total
    comp=comp.reindex(['ENSO','Other','Memory','Atmosphere','Ocean','Unallocated'])
    table('S15_signed_components',comp.reset_index())
    table('S15_main_Fig4_component_crosswalk',comp.reset_index())
    check('S15 map net denominator',np.isclose(total,.10651437039268086,rtol=0,atol=1e-12))
    check('S15 regional signed percentages close',np.isclose((reg.Positive_pct+reg.Negative_pct).sum(),100))
    check('S15 component positive closure',np.isclose(comp.Allocated_positive_shortfall.sum(),reg.Positive_shortfall.sum()))
    check('S15 component negative closure',np.isclose(comp.Allocated_above_benchmark.sum(),reg.Above_benchmark.sum()))
    f,axes=plt.subplots(1,2,figsize=(180/25.4,137/25.4),gridspec_kw={'width_ratios':[1,1.16]})
    f.subplots_adjust(left=.115,right=.98,bottom=.20,top=.91,wspace=.72)
    items=[(reg,reg.Region.tolist()),(comp,['ENSO','Other climate\nmodes','Prior ocean\nthermal state','Surface\nheating','Horizontal\nadvection','Unresolved'])]
    for i,(ax,(q,names)) in enumerate(zip(axes,items)):
        yy=np.arange(len(q));ax.barh(yy,q.Positive_pct,height=.56,color=RED,alpha=.85,label='Lower forecast skill')
        ax.barh(yy,q.Negative_pct,height=.56,color='#2166AC',hatch='///',edgecolor='#FFFFFF',lw=.4,label='Higher forecast skill')
        ax.scatter(q.Net_pct,yy,c=DARK,s=15,marker='D',zorder=5,label='Net difference')
        ax.axvline(0,c=DARK,lw=.7);ax.set_yticks(yy,names);ax.invert_yaxis()
        ax.set_xlabel('Share of net map difference (%)');label(ax,chr(97+i))
        ax.set_title(['Regional contributions','Association-weighted allocation'][i],pad=10)
    handles,labels=axes[0].get_legend_handles_labels()
    f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.015),ncol=2,fontsize=7.5)
    finish(f,15,'','','Fig. 4','')

def fig06():
    d=nc('F3','Figure3_Z200_fields.nc');f=plt.figure(figsize=(180/25.4,134/25.4));ims=[]
    for i,ev in enumerate(EVENTS):
        for j,typ in enumerate(['observed','forecast','forecast_error']):
            ax=f.add_axes([.045+j*.315,.70-i*.265,.295,.215],projection=PROJ);field=d[typ].sel(event=ev)
            if 'lead' in field.dims:field=field.mean('lead')
            ims.append(b.mapfield(ax,field,100 if j==2 else 160));label(ax,chr(97+3*i+j))
            if i==0:ax.set_title(['Observed','Forecast','Forecast − observed'][j],fontsize=8.5,pad=12)
            if j==0:f.text(.004,.81-i*.265,ev.replace('/','–'),rotation=90,va='center',fontsize=8.5)
    for x,w,im,lab in [(.075,.55,ims[0],'200-hPa height anomaly (m)'),(.71,.25,ims[2],'Forecast − observed (m)')]:
        ca=f.add_axes([x,.083,w,.015]);cb=f.colorbar(im,cax=ca,orientation='horizontal',extend='both');cb.set_label(lab)
    finish(f,6,'','','Fig. 3','')

def fig07():
    d=csv('F3','SI_regional_skill_change.csv');f,a=plt.subplots(2,2,figsize=(180/25.4,168/25.4));f.subplots_adjust(left=.07,right=.87,bottom=.12,top=.94,wspace=.25,hspace=.42)
    ims=[]
    for i,col in enumerate(EVENTS+['Delta_r']):
        z=d.pivot(index='Region',columns='Lead',values=col).reindex(REM)
        limit=1 if i<3 else np.ceil(np.nanmax(np.abs(z.values))*10)/10
        im=heat(a.flat[i],z.values,z.columns,REM,-limit,limit,cmap=DIV.reversed(),annot=True);ims.append(im)
        label(a.flat[i],chr(97+i));a.flat[i].set(xlabel='Lead month',title=col if i<3 else '2023–24 minus earlier events')
    ca=f.add_axes([.91,.56,.018,.36]);cb=f.colorbar(ims[0],cax=ca);cb.set_label('Spatial correlation, r')
    ca=f.add_axes([.91,.12,.018,.36]);cb=f.colorbar(ims[-1],cax=ca);cb.set_label('Change in correlation, Δr')
    finish(f,7,'','','Fig. 3','')

def model_plot():
    q=pd.read_csv(DER/'S08_model_skill.csv');models=['COLA-RSMAS-CCSM4','NASA-GEOSS2S','GFDL-SPEAR','Multimodel mean']
    f,a=plt.subplots(1,2,figsize=(180/25.4,140/25.4));f.subplots_adjust(left=.25,right=.97,bottom=.15,top=.90,wspace=.33)
    for j,metric in enumerate(['r','RMSE']):
        ax=a[j];label(ax,chr(97+j))
        for k,(ev,c) in enumerate(zip(EVENTS,COLORS)):
            z=q[q.event==ev];center=z.groupby('model')[metric].mean().reindex(models);lo=z.groupby('model')[metric].min().reindex(models);hi=z.groupby('model')[metric].max().reindex(models)
            yy=np.arange(4)+(k-1)*.22;ax.errorbar(center,yy,xerr=[center-lo,hi-center],fmt='o',ms=4,c=c,lw=.8,capsize=2,label=ev)
            for n,m in enumerate(models):ax.scatter(z[z.model==m][metric],yy[n]+np.linspace(-.04,.04,8),s=8,c=c,alpha=.4)
        labels=['COLA–RSMAS–CCSM4','NASA–GEOS-S2S','GFDL–SPEAR','Multimodel mean']
        ax.set_yticks(range(4),labels if j==0 else ['']*4);ax.invert_yaxis();ax.set_xlabel('Spatial correlation, r' if j==0 else 'Height RMSE (m)');ax.set_title('Spatial agreement' if j==0 else 'Error magnitude')
    f.legend(*a[0].get_legend_handles_labels(),loc='lower center',ncol=3,bbox_to_anchor=(.58,.015));return f

def fig10():
    d=csv('F3','SI_global_individual_el_nino_correlations.csv');amp=csv('F3','SI_climatology_distance_and_verification.csv')
    f=plt.figure(figsize=(180/25.4,158/25.4));ax=f.add_axes([.11,.60,.85,.31]);label(ax,'a')
    for anchor,c,lab in [('forecast_2023',RED,'2023–24 forecast'),('forecast_canonical',DARK,'Earlier strong-event forecast')]:
        z=d[d.anchor==anchor].sort_values('start_year');ax.plot(range(6),z.r,'o-',c=c,ms=4,lw=.9,label=lab)
    ax.set_xticks(range(6),z.start_year);ax.set(ylabel='Spatial correlation, r',xlabel='Other El Niño start year',ylim=(-.1,1));ax.legend(loc='lower left',fontsize=7.5)
    for i,(metric,ylabel) in enumerate([('forecast_observed_spatial_std_ratio','Forecast / observed anomaly SD'),('mse_skill_vs_climatology','Skill relative to climatology')]):
        ax=f.add_axes([.12+i*.48,.13,.35,.31]);label(ax,chr(98+i))
        for j,(ev,c) in enumerate([('canonical',DARK),('2023',RED)]):
            z=amp[amp.event.astype(str)==ev].set_index('band').loc[['NH','SH']];ax.bar(np.arange(2)+(j-.5)*.27,z[metric],width=.25,color=c)
        ax.set_xticks([0,1],['30–60°N','40–60°S']);ax.set_ylabel(ylabel);zero(ax)
        if i==0:ax.axhline(1,c=GRAY,ls='--',lw=.65);ax.set_ylim(0,1.15)
    finish(f,10,'','','Fig. 3','')

def reconstruction_plot():
    v=pd.read_csv(DER/'S13_pooled_skill.csv',keep_default_na=False).set_index('Region').loc[REG]
    f=plt.figure(figsize=(180/25.4,163/25.4));a=f.add_axes([.29,.57,.56,.34]);c=f.add_axes([.29,.14,.56,.32]);label(a,'a');label(c,'b')
    vv=v[['Climate modes only','All information']].T;limit=max(1,np.abs(vv).to_numpy().max());im=heat(a,vv,REG,['Climate modes','All measured predictors'],-limit,limit,cmap=DIV.reversed());ca=f.add_axes([.9,.57,.015,.34]);cb=f.colorbar(im,cax=ca);cb.set_label('SST reconstruction skill')
    rem=['No ENSO','No other modes','No prior thermal state','No surface heating','No horizontal advection'];dv=np.stack([v['All information']-v[r] for r in rem],axis=0);limit=max(.05,np.max(np.abs(dv)))
    im=heat(c,dv,REG,['ENSO','Other modes','Prior thermal state','Surface heating','Horizontal advection'],-limit,limit,cmap=DIV.reversed());ca=f.add_axes([.9,.14,.015,.32]);cb=f.colorbar(im,cax=ca);cb.set_label('Skill lost when group is omitted')
    a.set_title('Later-period SST reconstruction',pad=12);c.set_title('Predictor-removal test',pad=12);return f

def recalc_bridge():
    path=freeze(BRIDGE/'source_data/Figure4b_paired_error_fields.nc','BRIDGE');ds=xr.load_dataset(path)
    spec=importlib.util.spec_from_file_location('new_regions',BASE/'Figure4_latitude_sync_20261111_v23/regions.py');reg=importlib.util.module_from_spec(spec);spec.loader.exec_module(reg)
    masks=reg.masks(ds.lat.values,ds.lon.values);ds['region_mask']=masks.astype('int8')
    ds.attrs['region_boundary']='Pacific/Atlantic +/-23.5 degrees; Fig3/Fig4-aligned; native grid-centre membership'
    ds.to_netcdf(DER/'S14_paired_error_fields_235.nc',encoding={v:{'zlib':True,'complevel':4} for v in ds.data_vars})
    rows=[]
    for e,l,r in itertools.product(ds.event.values,ds.lead.values,REG):
        z=ds.sel(event=e,lead=l);ma=ds.region_mask.sel(region=r).values.astype(bool)&ds.common_support.values.astype(bool);w=ma*np.cos(np.deg2rad(ds.lat.values))[:,None];row=dict(Event=int(e),Lead=int(l),Region=r,N_cells=int(ma.sum()))
        for name,var in [('Z200','z200_eddy_error'),('Qnet','qnet_error')]:
            v=z[var].values;check('S14 finite '+str(e)+str(l)+r+name,np.isfinite(v[:,ma]).all());row[name+'_RMSE']=np.sqrt(np.sum(np.where(ma,v**2,0)*w)/(w.sum()*3))
        rows.append(row)
    d=pd.DataFrame(rows);ref=d[d.Event!=2023].groupby(['Region','Lead'])[['Z200_RMSE','Qnet_RMSE']].mean().rename(columns=lambda c:c.replace('_RMSE','_reference'));d=d.join(ref,on=['Region','Lead'])
    for k in ['Z200','Qnet']:d[k+'_relative']=d[k+'_RMSE']/d[k+'_reference']
    table('S14_paired_errors',d);c=(d[d.Event==2023].groupby('Region')[['Z200_relative','Qnet_relative']].mean().reindex(REG)-1)*100;table('S14_regional_change',c.reset_index())
    return d,c

def fig14():
    d,c=recalc_bridge();r=d[d.Region!='TP'];f=plt.figure(figsize=(180/25.4,158/25.4));a=f.add_axes([.11,.55,.37,.36]);h=f.add_axes([.62,.55,.27,.36]);o=f.add_axes([.29,.13,.59,.25]);label(a,'a');label(h,'b');label(o,'c')
    for ev,col,m in zip([1997,2015,2023],COLORS,['o','s','^']):
        z=r[r.Event==ev];a.scatter(z.Z200_relative,z.Qnet_relative,s=24,color=col,marker=m,edgecolors='white',linewidths=.3,alpha=.8,label=str(ev)+'–'+str(ev+1)[-2:])
    fit=stats.linregress(r.Z200_relative,r.Qnet_relative);xx=np.linspace(r.Z200_relative.min(),r.Z200_relative.max(),100);a.plot(xx,fit.intercept+fit.slope*xx,c=DARK,ls='--',lw=.8)
    a.set(xlabel='Circulation RMSE / earlier-event mean',ylabel='Heat-flux RMSE / earlier-event mean');a.legend(loc='upper left')
    limit=np.ceil(abs(c.values).max()/10)*10;im=heat(h,c.values,['Circulation','Heat flux'],REG,-limit,limit,cmap=DIV,annot=False);h.set_title('RMSE increase (%)',pad=12);ca=f.add_axes([.93,.55,.015,.36]);f.colorbar(im,cax=ca)
    vals=[]
    for ev in [1997,2015,2023]:
        z=r[r.Event!=ev];vals.append(corr(z.Z200_relative,z.Qnet_relative))
    o.scatter(vals,range(3),c=DARK,s=24);o.set_yticks(range(3),['Omit 1997–98','Omit 2015–16','Omit 2023–24']);o.invert_yaxis();o.axvline(fit.rvalue,c=RED,ls='--',lw=.8,label='All events');o.set(xlabel='Correlation between the two RMSE ratios, r',xlim=(-1,1));o.legend(loc='lower left')
    table('S14_event_omission',pd.DataFrame(dict(event=[1997,2015,2023],r=vals)))
    finish(f,14,'','','Figs. 3–4','')

def change_bars(ax,summary,lead,metric,regions):
    q=summary[summary.metric==metric].set_index('region').loc[regions]
    ax.barh(range(len(q)),q.change,color=[RED if x>=0 else BLUE for x in q.change],height=.55,alpha=.85)
    for j,r in enumerate(regions):
        row=q.loc[r];ax.plot([row.lead_min,row.lead_max],[j,j],c=DARK,lw=.8)
        z=lead[(lead.metric==metric)&(lead.region==r)].sort_values('lead');ax.scatter(z.change,j+np.linspace(-.13,.13,9),s=10,facecolor='white',edgecolor=DARK,lw=.4,zorder=4)
    ax.set_yticks(range(len(q)),regions);ax.set_ylim(len(q)-.5,-.5);ax.axvline(0,c=DARK,lw=.6,ls='--');ax.set_xlabel('AUC decrease' if metric=='AUC' else 'RMSE increase (°C)')

def fig16():
    d=pd.read_csv(freeze(SPATIAL/'source_data/regional_changes.csv','SPATIAL'),keep_default_na=False);l=pd.read_csv(freeze(SPATIAL/'source_data/regional_changes_by_lead.csv','SPATIAL'),keep_default_na=False)
    f,a=plt.subplots(2,2,figsize=(180/25.4,160/25.4));f.subplots_adjust(left=.08,right=.97,bottom=.12,top=.93,wspace=.32,hspace=.45)
    for row,metric in enumerate(['AUC','RMSE']):
        change_bars(a[row,0],d,l,metric,REG);label(a[row,0],chr(97+row*2));ax=a[row,1];label(ax,chr(98+row*2));q=d[d.metric==metric].set_index('region').loc[REG]
        ax.barh(range(7),q.coverage*100,color='#999999',height=.55);ax.set(xlim=(0,110),yticks=range(7),yticklabels=REG,xlabel='Common ocean area (%)');ax.invert_yaxis()
        for j,v in enumerate(q.coverage):ax.text(v*100+2,j,f'{v*100:.1f}',va='center',fontsize=8)
        ax.set_title(metric+' comparison coverage')
    finish(f,16,'','','Fig. 1','')

def fig17():
    d=pd.read_csv(freeze(SPATIAL/'source_data/regional_sensitivity.csv','SPATIAL'),keep_default_na=False)
    coverage=pd.read_csv(freeze(SPATIAL/'source_data/coverage_comparison.csv','SPATIAL'))
    table('S17_global_coverage',coverage)
    table('S17_regional_sensitivity',d)
    f,a=plt.subplots(2,2,figsize=(180/25.4,156/25.4))
    f.subplots_adjust(left=.09,right=.97,bottom=.12,top=.83,wspace=.36,hspace=.43)
    variants=['12months_1deg','12months_11deg','18months_1deg','18months_11deg']
    colors=['#B4B4B4','#2166AC','#303030',RED]
    names=['12 months · grid cell','12 months · 11° neighbourhood','18 months · grid cell','18 months · 11° neighbourhood']
    for row,metric in enumerate(['AUC','RMSE']):
        for k,(variant,c,name) in enumerate(zip(variants,colors,names)):
            q=d[(d.variant==variant)&(d.metric==metric)].set_index('region').loc[REG]
            for col,values in enumerate([q.change,q.coverage*100]):
                ax=a[row,col];yy=np.arange(7)+(k-1.5)*.13
                ax.scatter(values,yy,s=18,marker=['o','s','^','D'][k],facecolor=c,edgecolor='white',linewidth=.3,label=name,zorder=3)
                ax.set_yticks(range(7),REG);ax.set_ylim(6.55,-.55)
        for col in range(2):label(a[row,col],chr(97+2*row+col))
        a[row,0].axvline(0,color=DARK,lw=.7,ls='--')
        a[row,0].set_xlabel('AUC decrease' if metric=='AUC' else 'RMSE increase (°C)')
        a[row,1].set(xlim=(0,105),xticks=[0,25,50,75,100],xlabel='Common ocean area (%)')
    handles,labels=a[0,0].get_legend_handles_labels()
    f.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.99),ncol=2,fontsize=7.5,columnspacing=1,handletextpad=.4)
    finish(f,17,'','','Fig. 1','')

def finish(fig,num,title,caption,main,stats_text,status='',calc=''):
    if num==8:plt.close(fig);fig=model_plot()
    if num==13:plt.close(fig);fig=reconstruction_plot()
    if num==3:
        d=pd.read_csv(DER/'S03_partial_residuals.csv');coef=np.linalg.lstsq(np.c_[np.ones(len(d)),d.Peak_Nino34,d.year-d.year.mean()],d.Mean_AUC,rcond=None)[0][1]
        ax=fig.axes[2];ax.axvline(coef,c=RED,lw=1,ls='--',label='All 9 events');ax.legend(loc='upper left',fontsize=7.5)
    if num==4:
        q=pd.read_csv(DER/'S04_lead_benchmarks.csv');fig.axes[0].vlines(q.lead,q.observed,q.expected,color='#AAAAAA',lw=.75);fig.axes[0].set_title('AUC at each lead')
    if num==9:
        fig.subplots_adjust(left=.13,right=.98,bottom=.13,top=.92,wspace=.62,hspace=.53)
        for ax in fig.axes[:3]:ax.set_xlabel('Tropical precipitation correlation, r');ax.set_ylabel('Remote height correlation, r')
        fig.axes[3].set_title('Sensitivity to event omission')
    if num==11:
        fig.subplots_adjust(left=.10,right=.98,bottom=.19,top=.92,wspace=.35,hspace=.46)
        for t in fig.legends:t.set_bbox_to_anchor((.5,.025));t.set_ncols(2)
    if num==12:
        fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.91,wspace=.3);fig.axes[0].set_title('Total SST-variance allocation');fig.axes[1].set_title('Within the local group')
    from compact_layout import apply_layout
    apply_layout(fig,num)
    # Readable final-size type. Deliberately do not modify data or axes limits.
    for t in fig.findobj(mpl.text.Text):
        if t.get_fontsize()<7.5:t.set_fontsize(7.5)
        replacements={'Historical time dependence':'AUC versus event year','After removing event year':'After adjusting for year','Omit one historical event':'Omit one event (n = 8)',
            'Local information hierarchy':'Within the local group','Historical SST-variance share (%)':'Share of total SST variance (%)','Unassigned':'Unresolved',
            'Observed canonical':'Observed 1997–98/2015–16','Forecast canonical':'Forecast 1997–98/2015–16',
            'Spatial agreement':'Spatial correlation','Error magnitude':'Height RMSE',
            'Historical expectation':'Expected from earlier events','Prior thermal state':'Prior ocean thermal state'}
        if t.get_text() in replacements:t.set_text(replacements[t.get_text()])
    from final_polish import polish
    polish(fig,num)
    c=CAPTIONS[num];original_finish(fig,num,c['title'],c['caption'],main,c.get('statistics',''),'documented',calc)
    rec=RECORDS[-1];rec['role']=c['role'];rec['status']='scientific limits in Supplementary Methods'
    (ROOT/'captions'/f'{rec["id"]}.md').write_text(f'# Supplementary Fig. {num} | {c["title"]}\n\n{c["caption"]}\n\n{c.get("statistics", "")}\n')
b.finish=finish

def main():
    funcs=[fig01,fig02,b.fig03,b.fig04,fig05,fig06,fig07,b.fig08,b.fig09,fig10,b.fig11,b.fig12,b.fig13,fig14,fig15,fig16,fig17]
    parser=argparse.ArgumentParser();parser.add_argument('--only',type=int,nargs='*');args=parser.parse_args()
    old=json.loads((ROOT/'figure_registry.json').read_text()) if (ROOT/'figure_registry.json').exists() else []
    for i,fn in enumerate(funcs,1):
        if args.only and i not in args.only:continue
        print('RENDER',i,flush=True);fn()
    if args.only:
        ids={r['number'] for r in RECORDS};RECORDS.extend(r for r in old if r['number'] not in ids)
        if (ROOT/'source_manifest.json').exists():MANIFEST.extend(json.loads((ROOT/'source_manifest.json').read_text()))
    RECORDS.sort(key=lambda r:r['number'])
    dump(ROOT/'figure_registry.json',RECORDS);dump(ROOT/'source_manifest.json',list({m['frozen']:m for m in MANIFEST}.values()));dump(ROOT/'qa/numerical_checks.json',CHECKS)
    from package_si import package
    package(RECORDS)
if __name__=='__main__':main()
