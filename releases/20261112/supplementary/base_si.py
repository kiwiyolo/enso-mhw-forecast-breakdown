# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# S01-S05, S08-S15: project Figure1-3 render.py and LineTrend/plot_trend.py -> param inherit
# S06-S07, A02: project Figure3 map/heatmap implementation -> param inherit
# S11: project archived transect source tables -> param inherit
# The generic publication-count template is semantically incompatible with these metrics.
# User-requested Fig1-4 style and scientific definitions override generic template layouts.
# No template/mock observations are used. All geographic data remain on their native grids.
# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})
# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"
# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

from pathlib import Path
import argparse, hashlib, json, shutil, itertools, textwrap, sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.path import Path as MPath
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import fitz

ROOT=Path(__file__).resolve().parent
BASE=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures')
MAIN=BASE/'Figure1-3_Nature_layout_20261110_v7'
FOUR=BASE/'Figure4_latitude_sync_20261111_v23'
OLD=BASE/'Revised_20260921/Supplementary'
SRC=ROOT/'source_data'; DER=ROOT/'derived_data'; OUT=ROOT/'figures'
for p in [SRC,DER,OUT,ROOT/'captions',ROOT/'qa',ROOT/'provenance',ROOT/'.paper-workflow']:
    p.mkdir(parents=True,exist_ok=True)
RED='#D94B3A'; DARK='#303030'; GRAY='#AAAAAA'; BLUE='#2166AC'; PURPLE='#8064A2'
COLORS=[GRAY,DARK,RED]; EVENTS=['1997/98','2015/16','2023/24']
REG=['TP','NP','SP','TA','NA','SA','IO']; REM=REG[1:]
PC=ccrs.PlateCarree(); PROJ=ccrs.Robinson(central_longitude=-10)
DIV=LinearSegmentedColormap.from_list('main_diverging',
    ['#053061','#2166AC','#4393C3','#92C5DE','#D1E5F0','#F7F7F7',
     '#FDDBC7','#F4A582','#D6604D','#B2182B','#67001F'],N=256)
FONTDIR=ROOT/'fonts';FONTDIR.mkdir(exist_ok=True)
for p in (MAIN/'fonts').glob('*'):
    if p.is_file() and not (FONTDIR/p.name).exists():shutil.copy2(p,FONTDIR/p.name)
for p in FONTDIR.glob('*.ttf'): font_manager.fontManager.addfont(str(p))
mpl.rcParams.update({'font.sans-serif':['Liberation Sans'],'font.size':6.5,
    'axes.labelsize':6.5,'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6,
    'legend.fontsize':6,'axes.labelpad':2,'xtick.major.pad':2,'ytick.major.pad':2,
    'xtick.major.size':2.5,'ytick.major.size':2.5,'savefig.bbox':None,'pdf.compression':9})
MANIFEST=[]; RECORDS=[]; CHECKS=[]; CURRENT=[]
def dump(p,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(group,name):
    d=SRC/group/name
    if not d.is_file():raise FileNotFoundError('Missing bundled source: '+str(d))
    MANIFEST.append({'source':'bundled', 'frozen':str(d.relative_to(ROOT)), 'sha256':sha(d)})
    CURRENT.append(str(d.relative_to(ROOT)))
    return d
def csv(group,name): return pd.read_csv(source(group,name),keep_default_na=False,na_values=[''])
def nc(group,name): return xr.load_dataset(source(group,name))
def table(name,df): df.to_csv(DER/(name+'.csv'),index=False)
def check(name,condition,detail=''):
    CHECKS.append({'test':name,'passed':bool(condition),'detail':str(detail)})
    if not condition: raise AssertionError(name+': '+str(detail))
def newfig(rows=1,cols=2,h=85,**kwargs):
    f,axs=plt.subplots(rows,cols,figsize=(180/25.4,h/25.4),squeeze=False,**kwargs)
    f.subplots_adjust(left=.09,right=.97,bottom=.17,top=.90,wspace=.34,hspace=.48)
    return f,axs
def label(ax,c): ax.text(-.105,1.045,c,transform=ax.transAxes,fontweight='bold',fontsize=8,va='bottom')
def zero(ax): ax.axhline(0,color='#999999',lw=.6,ls='--',zorder=0)
def legend(ax,**kw): return ax.legend(frameon=False,**kw)
def heat(ax,a,x,y,vmin=-1,vmax=1,cmap=DIV,annot=True):
    a=np.asarray(a,float); im=ax.imshow(np.ma.masked_invalid(a),vmin=vmin,vmax=vmax,cmap=cmap,aspect='auto',interpolation='none')
    ax.set_xticks(range(len(x)),x);ax.set_yticks(range(len(y)),y)
    ax.tick_params(length=0);ax.set_facecolor('#E4E4E4')
    for sp in ax.spines.values():sp.set_visible(False)
    if annot:
        for i,j in np.ndindex(a.shape):
            if np.isfinite(a[i,j]):
                ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=5.5,
                    color='white' if abs(a[i,j])>.65*max(abs(vmin),abs(vmax)) else DARK)
    return im
def cbar(fig,im,axs,title):
    cb=fig.colorbar(im,ax=list(np.ravel(axs)),fraction=.025,pad=.025,shrink=.9)
    cb.set_label(title,labelpad=3);return cb
def corr(x,y,w=None):
    x=np.asarray(x).ravel(); y=np.asarray(y).ravel()
    w=np.ones_like(x) if w is None else np.asarray(w).ravel()
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
    x=x[ok];y=y[ok];w=w[ok];w=w/w.sum();x=x-np.sum(w*x);y=y-np.sum(w*y)
    den=np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
    return float(np.sum(w*x*y)/den) if den>0 else np.nan
def finish(fig,num,title,caption,main,stats_text,status='descriptive',calc=''):
    key=f'Supplementary_Fig_{num:02d}' if isinstance(num,int) else f'Audit_Fig_{num}'
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); outside=[];fonts=[]
    ignored=set()
    for ax in fig.axes:
        if not ax.axison:
            ignored.update(id(t) for t in ax.xaxis.get_majorticklabels()+ax.yaxis.get_majorticklabels())
        for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            if not axis.get_visible():
                ignored.update(id(t) for t in axis.get_majorticklabels())
            for loc,t in zip(axis.get_majorticklocs(),axis.get_majorticklabels()):
                if not min(lim)<=loc<=max(lim): ignored.add(id(t))
    for t in fig.findobj(mpl.text.Text):
        if id(t) in ignored or not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.width==0 or b.height==0 or not np.isfinite(b.extents).all():continue
        fonts.append(t.get_fontsize())
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:outside.append(t.get_text())
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{key}.{ext}',dpi=300,bbox_inches=None,facecolor='white')
    with fitz.open(OUT/f'{key}.pdf') as d:d[0].get_pixmap(dpi=150,alpha=False).save(OUT/f'{key}_preview.png')
    record={'id':key,'number':num,'title':title,'caption':caption,'main_figure':main,
        'statistics':stats_text,'status':status,'calculation':calc,
        'sources':sorted(set(CURRENT)),'outside_text':outside,'min_font_pt':min(fonts),
        'width_mm':180,'height_mm':float(fig.get_size_inches()[1]*25.4)}
    RECORDS.append(record)
    (ROOT/'captions'/f'{key}.md').write_text(f'# {key.replace("_"," ")} | {title}\n\n{caption}\n\n## Statistics and interpretation\n\n{stats_text}\n\nStatus: {status}.\n\nCalculation: {calc or "Replot frozen source tables; no change to estimands."}\n\nMain-text link: {main}.\n\n## Sources\n\n'+''.join(f'- `{s}`\n' for s in record['sources']))
    print(key,'clipping:',outside,flush=True);CURRENT.clear();plt.close(fig)

def fig01():
    d=csv('OLD','Supplementary_Figure1_source_events.csv'); m=csv('OLD','Supplementary_Figure1_monthly_t1_t9.csv')
    h=d[d.Event!='2023/24'];t=d[d.Event=='2023/24'].iloc[0];x=h.Peak_Nino34.to_numpy();y=h.Mean_RMSE.to_numpy()
    fit=stats.linregress(x,y);xx=np.linspace(0.55,2.75,200);yy=fit.intercept+fit.slope*xx
    se=np.sqrt(np.sum((y-fit.intercept-fit.slope*x)**2)/(len(x)-2));ci=stats.t.ppf(.975,len(x)-2)*se*np.sqrt(1+1/len(x)+(xx-x.mean())**2/np.sum((x-x.mean())**2))
    f,a=newfig(h=80);ax=a[0,0];label(ax,'a');ax.fill_between(xx,yy-ci,yy+ci,color='#EEEEEE');ax.plot(xx,yy,color=DARK,lw=1)
    for _,q in d.iterrows(): ax.scatter(q.Peak_Nino34,q.Mean_RMSE,s=23 if q.Event=='2023/24' else 15,c=RED if q.Event=='2023/24' else DARK,marker='*' if q.Event=='2023/24' else 'o',zorder=3)
    ax.set(xlabel='Peak Niño3.4 anomaly (°C)',ylabel='MHW-intensity RMSE (°C)',title='Historical intensity benchmark')
    ax=a[0,1];label(ax,'b');ax.plot(pd.to_datetime(m.time),m.RMSE,color=DARK,lw=.65)
    for ev,c in zip(EVENTS,COLORS):
        q=d[d.Event==ev].iloc[0];ax.axvspan(pd.Timestamp(q.Start_Month),pd.Timestamp(q.End_Month),color=c,alpha=.12)
    ax.set(xlabel='Year',ylabel='MHW-intensity RMSE (°C)',title='Monthly record')
    table('S01_fit',pd.DataFrame([{'r':fit.rvalue,'slope':fit.slope,'expected':fit.intercept+fit.slope*t.Peak_Nino34,'observed':t.Mean_RMSE}]))
    finish(f,1,'An alternative error metric for MHW intensity',
        'a, Archived event-mean MHW-intensity RMSE against El Niño amplitude; the target event is excluded from the nine-event regression. Grey shading is a 95% external prediction interval. b, All 404 archived monthly values; shading marks the three strong events. Lower RMSE is better.',
        'Fig. 1','RMSE is evaluated only on observed-MHW cells; false alarms outside those cells are not included. Lead and month aggregation uses root-mean-square, not an arithmetic average of RMSE. This inherited metric has not been revalidated against corrected initialization sources; it is not an independent input audit.',
        'archived metric; source alignment unresolved')

def fig02():
    d=csv('F1','Figure1_monthly_t1_t9.csv');d['year']=pd.to_datetime(d.time).dt.year;d['amp']=d.Nino34.abs()
    d['phase']=np.where(d.Nino34>=.5,'El Niño',np.where(d.Nino34<=-.5,'La Niña','Neutral'))
    rng=np.random.default_rng(20261111);rows=[];f,a=newfig(h=80)
    for k,(phase,color) in enumerate([('El Niño',RED),('La Niña',BLUE)]):
        z=d[d.phase==phase];ax=a[0,k];label(ax,chr(97+k));ax.scatter(z.amp,z.AUC,c=color,s=8,alpha=.45,linewidths=0)
        fit=stats.linregress(z.amp,z.AUC);xx=np.linspace(z.amp.min(),z.amp.max(),100);ax.plot(xx,fit.intercept+fit.slope*xx,c=color,lw=1)
        years=d.year.unique();boots=[]
        for b in range(2000):
            zz=pd.concat([d[d.year==yr] for yr in rng.choice(years,len(years),replace=True)])
            zz=zz[zz.phase==phase];boots.append(stats.linregress(zz.amp,zz.AUC).slope)
        lo,hi=np.quantile(boots,[.025,.975]);rows.append(dict(phase=phase,n_months=len(z),n_years=z.year.nunique(),r=fit.rvalue,slope=fit.slope,low=lo,high=hi))
        ax.set(xlabel='Absolute Niño3.4 anomaly (°C)',ylabel='Global MHW forecast skill (AUC)',title=phase,ylim=(.3,.95))
        ax.text(.04,.95,f'r = {fit.rvalue:.2f}\nn = {len(z)} months',transform=ax.transAxes,va='top')
    table('S02_phase_statistics',pd.DataFrame(rows));table('S02_monthly_phase_membership',d)
    finish(f,2,'ENSO-phase dependence of the historical skill association',
        'a–b, All available El Niño and La Niña months from the archived monthly series, classified by Niño3.4 ≥0.5 °C and ≤−0.5 °C. Lines are descriptive ordinary-least-squares fits. Neutral months remain in the source table but are outside this two-phase comparison.',
        'Fig. 1','Months are serially dependent. Supplementary source tables provide slope intervals from 2,000 whole-calendar-year resamples (seed 20261111). No independent-month significance or phase-difference significance is claimed.',
        'archived AUC; phase comparison', 'New phase selection and calendar-year-block sensitivity from frozen monthly values.')

def fig03():
    d=csv('F1','Figure1_source_events.csv');h=d[d.Event!='2023/24'].copy();h['year']=pd.to_datetime(h.Start_Month).dt.year+.5
    X=np.c_[np.ones(len(h)),h.year-h.year.mean()];x=h.Peak_Nino34.to_numpy();y=h.Mean_AUC.to_numpy()
    xr_=x-X@np.linalg.lstsq(X,x,rcond=None)[0];yr_=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
    f,a=newfig(1,3,h=79);f.subplots_adjust(left=.08,wspace=.48,right=.98)
    a[0,0].scatter(h.year,y,c=DARK,s=17);a[0,0].set(xlabel='Event midpoint year',ylabel='MHW forecast skill (AUC)',title='Historical time dependence')
    fit=stats.linregress(xr_,yr_);a[0,1].scatter(xr_,yr_,c=DARK,s=17);xx=np.linspace(xr_.min(),xr_.max(),80);a[0,1].plot(xx,fit.intercept+fit.slope*xx,c=DARK,lw=.8)
    a[0,1].set(xlabel='Year-adjusted Niño3.4 (°C)',ylabel='Year-adjusted AUC',title='After removing event year')
    rows=[]
    for ev in h.Event:
        q=h[h.Event!=ev];xx=np.c_[np.ones(len(q)),q.Peak_Nino34,q.year-q.year.mean()];coef=np.linalg.lstsq(xx,q.Mean_AUC,rcond=None)[0]
        rows.append({'omitted':ev,'amplitude_coefficient':coef[1],'year_coefficient':coef[2]})
    rr=pd.DataFrame(rows);ax=a[0,2];ax.scatter(rr.amplitude_coefficient,range(9),c=DARK,s=15);ax.set_yticks(range(9),rr.omitted);ax.invert_yaxis();ax.axvline(0,c=GRAY,ls='--',lw=.6);ax.set(xlabel='AUC per °C',title='Omit one historical event')
    for i,ax in enumerate(a.flat):label(ax,chr(97+i))
    table('S03_omit_event',rr);table('S03_partial_residuals',h.assign(amplitude_residual=xr_,AUC_residual=yr_))
    finish(f,3,'Sensitivity of the historical benchmark to event year and event selection',
        'a, Historical event skill versus event midpoint year. b, Residual amplitude–skill association after removing a linear dependence on event year from both quantities. c, Amplitude coefficients from two-predictor regressions after omitting each historical event. The 2023–2024 event is excluded throughout.',
        'Fig. 1','Nine historical events; omission fits contain eight events. The nine estimates are sensitivity cases, not independent replicates or confidence intervals. Calendar-year adjustment tests one confounder only.',
        'archived AUC; recomputed sensitivity')

def fig04():
    d=csv('F4','sample_AUC_profiles.csv');d=d[d.Profile=='Archived'];e=csv('F1','Figure1_source_events.csv')
    rows=[]
    for lead in range(1,10):
        v=d[d.Lead_Month==lead].groupby('Event').AUC.mean().reindex(e.Event);h=e.Event!='2023/24';fit=stats.linregress(e.loc[h,'Peak_Nino34'],v.to_numpy()[h])
        expected=fit.intercept+fit.slope*e.loc[~h,'Peak_Nino34'].iloc[0];rows.append({'lead':lead,'expected':expected,'observed':v.iloc[-1],'deficit':expected-v.iloc[-1]})
    q=pd.DataFrame(rows);f,a=newfig(h=83);ax=a[0,0];label(ax,'a');ax.plot(q.lead,q.expected,'o-',c=DARK,ms=3,lw=1,label='Historical expectation');ax.plot(q.lead,q.observed,'o-',c=RED,ms=3,lw=1,label='2023–2024');legend(ax,loc='lower left');ax.set(xlabel='Lead month',ylabel='MHW forecast skill (AUC)',title='Lead-specific historical benchmarks',ylim=(.4,.9))
    ax=a[0,1];label(ax,'b')
    for ev,c in zip(EVENTS,COLORS):
        z=d[d.Event==ev].groupby('Target_Month').AUC.mean().sort_index();check('S04 '+ev+' 12 months',len(z)==12);ax.plot(range(12),z.values,'o-',c=c,ms=2.8,lw=.9,label=ev)
    ax.set_xticks(range(0,12,2),['Jul','Sep','Nov','Jan','Mar','May']);ax.set(ylabel='MHW forecast skill (AUC)',xlabel='Verifying month',title='Within-event evolution',ylim=(.3,.95));legend(ax,loc='lower left')
    table('S04_lead_benchmarks',q)
    finish(f,4,'Lead-time and within-event structure of the skill shortfall',
        'a, Historical expected and realized 2023–2024 AUC, recalculated separately for each future lead. Each historical fit uses the same nine earlier events. b, Monthly AUC averaged over leads 1–9 during July–June for the three strong events.',
        'Fig. 1','All 1,080 archived event–month–lead values are retained. Each point in a averages 12 verifying months; leads are repeated diagnostics, not independent events. No new significance is inferred.',
        'archived AUC; lead-resolved sensitivity')

def fig05():
    d=csv('F2','pattern_correlations.csv');s=csv('F2','historical_reference_sensitivity.csv');r=csv('F2','relative_loss_by_lead.csv')
    f,a=newfig(2,3,h=130);f.subplots_adjust(left=.08,right=.92,bottom=.13,top=.92,wspace=.45,hspace=.65)
    pro=['SST','Zonal wind stress','Convection proxy'];names=['SST','Wind stress','Convection']
    for j,(p,n) in enumerate(zip(pro,names)):
        z=d[d.Process==p].pivot(index='Event',columns='Lead',values='Pattern_correlation').reindex(EVENTS)
        im=heat(a[0,j],z.values,list(z.columns),['1997–98','2015–16','2023–24'],0,1,annot=False);a[0,j].set(xlabel='Lead month',title=n);label(a[0,j],chr(97+j))
        for ev,c in zip(EVENTS[:2],COLORS[:2]):
            h=d[(d.Process==p)&(d.Event==ev)].sort_values('Lead');t=d[(d.Process==p)&(d.Event==EVENTS[2])].sort_values('Lead');v=100*(h.Pattern_correlation.to_numpy()-t.Pattern_correlation.to_numpy())/h.Pattern_correlation.to_numpy()
            a[1,j].plot(h.Lead,v,'o-',c=c,lw=.9,ms=2.5,label=ev)
        mean=r[r.Process==p];a[1,j].plot(mean.Lead,mean.Relative_decline_percent,c=RED,lw=1.1,label='Two-event mean');zero(a[1,j]);a[1,j].set(xlabel='Lead month',ylabel='Correlation decline (%)');label(a[1,j],chr(100+j))
    cbar(f,im,a[0,:],'Spatial correlation, r');legend(a[1,0],loc='upper left')
    check('S05 all 81 field correlations',len(d)==81)
    finish(f,5,'Event-specific tropical forecast correlations and reference sensitivity',
        'a–c, All three events and nine archived leads for SST, wind stress and convection. d–f, Relative correlation declines using each historical event separately and their equal-event mean. Positive values indicate lower correlation in 2023–2024. Convection uses precipitation as a proxy.',
        'Fig. 2','81 correlations; two historical reference cases, not uncertainty replicates. Variable-specific windows and model availability are inherited. Relative declines have separate denominators and are not additive loss contributions. No nearest-lead filling or cross-variable common-cohort claim is introduced.',
        'archived variable-specific cohorts; descriptive')

def mapframe(ax):
    left,right=-190+1e-5,170-1e-5
    xx=np.r_[np.linspace(left,right,721),np.full(241,right),np.linspace(right,left,721),np.full(241,left)]
    yy=np.r_[np.full(721,-60),np.linspace(-60,60,241),np.full(721,60),np.linspace(60,-60,241)]
    xy=PROJ.transform_points(PC,xx,yy)[:,:2];ax.set_boundary(MPath(xy),transform=ax.transData)
    ax.set_xlim(xy[:,0].min(),xy[:,0].max());ax.set_ylim(xy[:,1].min(),xy[:,1].max());ax.coastlines(resolution='110m',lw=.3,color='#666666');ax.spines['geo'].set_linewidth(.4)
def mapfield(ax,field,limit):
    a,x=add_cyclic_point(field.values,coord=field.lon.values)
    im=ax.pcolormesh(x,field.lat,a,transform=PC,cmap=DIV,vmin=-limit,vmax=limit,shading='auto',rasterized=True,antialiased=False);mapframe(ax);return im

def fig06():
    d=nc('F3','Figure3_Z200_fields.nc');f=plt.figure(figsize=(180/25.4,108/25.4));axs=[];ims=[]
    for i,ev in enumerate(EVENTS):
        for j,typ in enumerate(['observed','forecast','forecast_error']):
            ax=f.add_axes([.04+j*.315,.70-i*.275,.285,.235],projection=PROJ);field=d[typ].sel(event=ev)
            if 'lead' in field.dims:field=field.mean('lead')
            im=mapfield(ax,field,100 if j==2 else 160);ax.set_title((['Observed','Forecast','Forecast − observed'][j]+' · '+ev),fontsize=6.2,pad=3);label(ax,chr(97+i*3+j));axs.append(ax);ims.append(im)
    for x,w,im,lab in [(.10,.49,ims[0],'200-hPa geopotential height anomaly (m)'),(.72,.23,ims[2],'Height error (m)')]:
        cax=f.add_axes([x,.075,w,.017]);cb=f.colorbar(im,cax=cax,orientation='horizontal',extend='both');cb.set_label(lab)
    check('S06 actual common leads',list(d.lead.values)==list(range(1,9)))
    finish(f,6,'Observed and predicted upper-level circulation in all three strong events',
        'Rows show 1997–1998, 2015–2016 and 2023–2024; columns show observed DJF anomalies, forecast anomalies and their signed difference. Forecasts are equal-lead means across the eight available common leads. Observed and forecast maps share ±160 m; errors share ±100 m. Colourbar extensions identify saturation outside these unchanged main-figure ranges.',
        'Fig. 3','Native 1° fields, no extra smoothing. Anomalies use the archived 1992–2020 calendar-month baseline, with February 2018 excluded uniformly. Signed mean errors can cancel; non-cancelling RMSE is supplied in S08. Maps are seasonal composites, not causal ENSO isolation.',
        'exact-coordinate atmospheric products')

def fig07():
    d=csv('F3','SI_regional_skill_change.csv');f,a=newfig(2,2,h=139);f.subplots_adjust(left=.09,right=.88,bottom=.12,top=.92,wspace=.25,hspace=.45)
    for i,col in enumerate(EVENTS+['Delta_r']):
        ax=a.flat[i];z=d.pivot(index='Region',columns='Lead',values=col).reindex(REM);im=heat(ax,z.to_numpy(),z.columns,REM,-1,1,cmap=DIV.reversed() if col=='Delta_r' else DIV,annot=True);label(ax,chr(97+i));ax.set(xlabel='Lead month',title=col if i<3 else '2023–24 minus historical mean')
        cb=f.colorbar(im,ax=ax,fraction=.035,pad=.03);cb.set_label('Δr' if i==3 else 'r')
    check('S07 explicit missing lead 9',d.loc[d.Lead==9,EVENTS].isna().all().all())
    finish(f,7,'Regional and lead-dependent circulation forecast skill',
        'a–c, Forecast–observation height-anomaly correlations for each event, six recipient regions and all nine nominal leads. d, Target-event correlation minus the arithmetic mean of the two historical correlations. In d, red denotes decline and blue improvement. Grey cells are unavailable lead 9, not zero skill.',
        'Fig. 3','48 valid region–lead combinations per event; six missing combinations remain explicit. The regional support follows Fig. 3 (±23.5° split). No significance is inferred from repeated leads or grid-cell count.',
        'exact-coordinate products; explicit missing support')

def fig08():
    d=nc('F3','Figure3_Z200_fields.nc');w=np.broadcast_to(np.cos(np.deg2rad(d.lat.values))[:,None],(d.sizes['lat'],d.sizes['lon']));rows=[]
    for model in list(d.model.values)+['Multimodel mean']:
        for ev in EVENTS:
            for lead in d.lead.values:
                pred=d.forecast.sel(event=ev,lead=lead).values if model=='Multimodel mean' else d.model_forecast.sel(model=model,event=ev,lead=lead).values
                obs=d.observed.sel(event=ev).values;mask=np.isfinite(pred)&np.isfinite(obs);weights=w[mask];err=(pred-obs)[mask]
                rows.append(dict(model=model,event=ev,lead=lead,r=corr(pred,obs,w),bias=np.average(err,weights=weights),RMSE=np.sqrt(np.average(err**2,weights=weights))))
    q=pd.DataFrame(rows);table('S08_model_skill',q);f,a=newfig(1,2,h=87);models=list(d.model.values)+['Multimodel mean'];names=['CCSM4','GEOS-S2S','SPEAR','Multimodel']
    for j,metric in enumerate(['r','RMSE']):
        ax=a[0,j];label(ax,chr(97+j))
        for k,(ev,c) in enumerate(zip(EVENTS,COLORS)):
            z=q[q.event==ev];center=z.groupby('model')[metric].mean().reindex(models).values;low=z.groupby('model')[metric].min().reindex(models).values;high=z.groupby('model')[metric].max().reindex(models).values
            xx=np.arange(4)+(k-1)*.22;ax.errorbar(xx,center,yerr=[center-low,high-center],fmt='o',ms=3,c=c,lw=.8,capsize=2,label=ev)
            for n,m in enumerate(models):ax.scatter(np.full(8,xx[n])+np.linspace(-.035,.035,8),z[z.model==m][metric],s=4,c=c,alpha=.4)
        ax.set_xticks(range(4),names,rotation=20);ax.set_ylabel('Spatial correlation, r' if j==0 else 'Height-anomaly RMSE (m)');ax.set_title('Pattern agreement' if j==0 else 'Non-cancelling error magnitude')
    legend(a[0,0],loc='lower left');check('S08 RMSE >= abs bias',np.all(q.RMSE>=abs(q.bias)-1e-9))
    finish(f,8,'Model dependence and magnitude of circulation forecast errors',
        'a, Global anomaly spatial correlations by model and event. b, Area-weighted RMSE of seasonal anomaly fields. Large points are the eight-lead mean; small points show all eight leads and whiskers their full range. Multimodel forecasts are verified after averaging model fields.',
        'Fig. 3','72 single-model and 24 multimodel event–lead evaluations, all longitudes, 60°S–60°N, including land. RMSE squares gridpoint errors before spatial averaging. Lead ranges are not confidence intervals; model dependence is descriptive.',
        'new metrics from frozen forecast and observed fields')

def fig09():
    d=csv('F3','Figure3b_all_samples.csv');om=csv('F3','SI_event_omission.csv');f,a=newfig(2,2,h=133);rows=[]
    markers=['o','s','^','v','D','P']
    for i,(ev,c) in enumerate(zip(EVENTS,COLORS)):
        ax=a.flat[i];z=d[d.Event==ev];label(ax,chr(97+i))
        for reg,m in zip(REM,markers):
            q=z[z.Region==reg];ax.scatter(q.Rain_r,q.Z200_r,c=c,marker=m,s=14,alpha=.7,label=reg)
        rr=corr(z.Rain_r,z.Z200_r);rows.append(dict(event=ev,n=len(z),r=rr));ax.set(xlabel='Tropical convection correlation, r',ylabel='Remote circulation correlation, r',title=ev,ylim=(-.6,1.05),xlim=(.55,.95));ax.text(.04,.94,f'r = {rr:.2f}',transform=ax.transAxes,va='top')
    legend(a[0,0],loc='lower left',ncol=3,columnspacing=.7,handletextpad=.3)
    ax=a[1,1];label(ax,'d');vals=[corr(d.Rain_r,d.Z200_r)]+om.r.tolist();ax.scatter(vals,range(4),c=[RED,DARK,DARK,DARK],s=20);ax.set_yticks(range(4),['All three events']+['Omit '+v for v in om.omitted_event]);ax.invert_yaxis();ax.set(xlabel='Pooled association, r',title='Dependence on event selection',xlim=(-.1,1));ax.axvline(0,c=GRAY,lw=.6,ls='--')
    table('S09_within_event_associations',pd.DataFrame(rows));check('S09 all 144 points',len(d)==144)
    finish(f,9,'Event dependence of the tropical–remote forecast association',
        'a–c, All event-specific convection–circulation correlation pairs, retaining every region and lead. Symbols distinguish the six recipient regions. d, Pooled association with all events and after omitting each event.',
        'Fig. 3','48 repeated samples per event; only three independent event clusters. Coefficients are descriptive. Event omission is sensitivity, not cross-validation of a predictive model or an uncertainty interval. All negative remote correlations are retained.',
        'sample-dependence sensitivity')

def fig10():
    d=csv('F3','SI_global_individual_el_nino_correlations.csv');amp=csv('F3','SI_climatology_distance_and_verification.csv');f,a=newfig(1,3,h=86);f.subplots_adjust(left=.075,right=.98,wspace=.48,bottom=.2)
    ax=a[0,0];label(ax,'a')
    for anchor,c in [('forecast_2023',RED),('forecast_canonical',DARK)]:
        z=d[d.anchor==anchor].sort_values('start_year');ax.plot(range(6),z.r,'o-',c=c,ms=3,lw=.8,label='2023–24' if anchor=='forecast_2023' else 'Canonical')
    ax.set_xticks(range(6),[str(x) for x in z.start_year],rotation=45);ax.set(ylabel='Spatial correlation, r',xlabel='Other El Niño start year',title='Individual reference events',ylim=(-.1,1));legend(ax,loc='lower left')
    for i,(metric,title,ylabel) in enumerate([('forecast_observed_spatial_std_ratio','Anomaly spatial amplitude','Forecast / observed SD'),('mse_skill_vs_climatology','Skill relative to climatology','1 − MSE / MSE(climatology)')],1):
        ax=a[0,i];label(ax,chr(97+i))
        for j,(ev,c) in enumerate([('canonical',DARK),('2023',RED)]):
            z=amp[amp.event.astype(str)==ev].set_index('band').loc[['NH','SH']];ax.bar(np.arange(2)+(j-.5)*.25,z[metric],width=.24,color=c,label=ev)
        ax.set_xticks([0,1],['Northern','Southern']);ax.set(xlabel='Midlatitudes',ylabel=ylabel,title=title);zero(ax)
        if i==1:ax.axhline(1,c=GRAY,lw=.6,ls='--');ax.set_ylim(0,1.12)
    finish(f,10,'Historical-reference sensitivity and anomaly amplitude',
        'a, Global correlations against each of the six other historical El Niño winters, rather than their composite. b, Forecast-to-observed spatial standard-deviation ratios of height anomalies. c, Mean-square-error skill relative to a zero-anomaly climatological forecast. Northern and southern bands are 30–60°N and 40–60°S, along the archived 120°E–20°E eastward sector.',
        'Fig. 3','Individual-event correlations do not average to the composite-first main statistic. Positive values in c mean better than climatology: a muted anomaly is not evidence of climatological forecast performance. Four deterministic band–event comparisons; no confidence interval is inferred.',
        'reference and amplitude sensitivity')

def fig11():
    d=csv('F3','Figure3cd_transects.csv');f,a=newfig(2,2,h=130);f.subplots_adjust(left=.09,right=.98,bottom=.14,top=.92,wspace=.3,hspace=.48)
    for row,lat in enumerate([45,-50]):
        z=d[d.latitude==lat].sort_values('unwrapped_longitude');ax=a[row,0];label(ax,chr(97+row*2))
        for col,c,ls,lab in [('canonical_anomaly',DARK,'-','Observed canonical'),('forecast_canonical_anomaly',DARK,'--','Forecast canonical'),('observed_anomaly',RED,'-','Observed 2023–24'),('forecast_anomaly',RED,'--','Forecast 2023–24')]:ax.plot(z.unwrapped_longitude,z[col],c=c,ls=ls,lw=.85,label=lab)
        ax.set(title=f'{abs(lat)}°'+('N' if lat>0 else 'S')+' anomalies',ylabel='Height anomaly (m)');zero(ax)
        ax=a[row,1];label(ax,chr(98+row*2));ax.plot(z.unwrapped_longitude,z.forecast_canonical_error,c=DARK,lw=.85,label='Canonical');ax.plot(z.unwrapped_longitude,z.forecast_error,c=RED,lw=.85,label='2023–24');zero(ax);ax.set(title='Signed forecast error',ylabel='Forecast − observed (m)')
    for ax in a.flat:ax.set_xticks([120,180,240,300,360,420],['120°E','180°','120°W','60°W','0°','60°E']);ax.set_xlim(120,479)
    f.legend(handles=[Line2D([],[],color=c,ls=ls,lw=.9,label=lab) for c,ls,lab in [(DARK,'-','Observed canonical'),(DARK,'--','Forecast canonical'),(RED,'-','Observed 2023–24'),(RED,'--','Forecast 2023–24')]],loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,fontsize=5.7)
    finish(f,11,'Longitude-dependent anomaly and forecast-error structure',
        'a,c, Observed and forecast height anomalies along 45°N and 50°S, at all longitudes starting at 120°E and continuing eastward around the globe. b,d, Signed forecast-minus-observed errors along the same sections. Canonical fields equally average the two historical strong events; the forecast uses the eight available leads.',
        'Fig. 3','All 720 archived section nodes are retained. Fixed latitude sections illustrate longitudinal differences, not diagnosed wave propagation paths or seasonally fixed Rossby waveguides. Negative errors can represent underestimated positive anomalies rather than observed trough deepening.',
        'descriptive transects')

def fig12():
    d=csv('F4','SI_observed_local_process_hierarchy.csv');parts=['ENSO','Other climate modes','Local ocean memory proxy','Local atmospheric heating','Local horizontal advection','Unresolved']
    names=['ENSO','Other modes','Prior thermal state','Surface heating','Horizontal advection','Unresolved'];colors=[RED,PURPLE,BLUE,'#4393C3','#92C5DE','#D9D9D9']
    vals=d.pivot(index='Region',columns='Group',values='Share').loc[REG,parts]*100
    check('S12 shares close to 100',np.allclose(vals.sum(1),100,atol=1e-7))
    f,a=newfig(h=89);ax=a[0,0];label(ax,'a');bottom=np.zeros(7)
    for i,p in enumerate(parts):ax.barh(range(7),vals[p],left=bottom,color=colors[i],height=.64,label=names[i],hatch='..' if i==5 else None,edgecolor='#777777' if i==5 else 'none',lw=.25);bottom+=vals[p].values
    ax.set_yticks(range(7),REG);ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Historical SST-variance share (%)',title='All association groups')
    ax=a[0,1];label(ax,'b');local=vals.iloc[:,2:5];den=local.sum(1);v=local.div(den,axis=0)*100;bottom=np.zeros(7)
    for i,p in enumerate(local.columns):ax.barh(range(7),v[p],left=bottom,color=colors[2+i],height=.64);bottom+=v[p].values
    ax.set_yticks(range(7),REG);ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Share within the local group (%)',title='Local information hierarchy')
    f.legend(handles=[mpl.patches.Patch(color=c,label=n) for c,n in zip(colors,names)],loc='lower center',ncol=3,bbox_to_anchor=(.5,.005),fontsize=6);f.subplots_adjust(bottom=.23)
    table('S12_local_normalized',v.reset_index())
    finish(f,12,'Regional composition of local SST information',
        'a, Historical SST-variance shares of the five measured predictor groups and the fit residual. b, The three local subgroups expressed as fractions of their parent local share. Prior thermal state comprises one-month-lag SST and mixed-layer temperature.',
        'Fig. 4','432 monthly observations per region, 1985–2020. Nested group-respecting R² allocation; panel b uses a different denominator explicitly shown on its axis. These are in-sample association weights, not heat-budget fractions, independent physical causes or process-forecast-error contributions.',
        'historical association, non-causal')

def fig13():
    d=csv('F4','Figure4_observed_association_samples.csv');features=['ENSO','IOD','PDO','NAO','SAM','sst_memory','ml_memory','surface_heating','horizontal_advection']
    configs={'Climate modes only':features[:5],'All information':features,
        'No prior thermal state':features[:5]+features[7:], 'No surface heating':[x for x in features if x!='surface_heating'],
        'No horizontal advection':[x for x in features if x!='horizontal_advection'],
        'No ENSO':features[1:],'No other modes':[features[0]]+features[5:]}
    rows=[];predictions=[]
    for reg in REG:
        z=d[d.Region==reg].copy();date=pd.to_datetime(z.time);years=date.dt.year.to_numpy();month=date.dt.month.to_numpy();nuis=np.c_[np.ones(len(z)),(years-1985+(month-1)/12)/10,np.eye(12)[month-1][:,1:]];y=z.sst.to_numpy(float)
        for start,end in [(2003,2008),(2009,2014),(2015,2020)]:
            tr=years<start;te=(years>=start)&(years<=end);base=nuis[te]@np.linalg.lstsq(nuis[tr],y[tr],rcond=None)[0];den=np.sum((y[te]-base)**2)
            for name,cols in configs.items():
                x=z[cols].to_numpy(float);mu=x[tr].mean(0);sd=x[tr].std(0);check(f'S13 finite training SD {reg} {start} {name}',np.all(sd>0));X=np.c_[nuis,(x-mu)/sd];coef=np.linalg.lstsq(X[tr],y[tr],rcond=None)[0];pred=X[te]@coef;sse=np.sum((y[te]-pred)**2);skill=1-sse/den
                rows.append(dict(Region=reg,model=name,test_start=start,test_end=end,n_train=int(tr.sum()),n_test=int(te.sum()),SSE=sse,baseline_SSE=den,skill=skill))
                predictions.extend(dict(Region=reg,model=name,time=str(t),observed=yy,predicted=pp,baseline=bb) for t,yy,pp,bb in zip(date[te],y[te],pred,base))
    q=pd.DataFrame(rows);table('S13_temporal_validation',q);table('S13_heldout_predictions',pd.DataFrame(predictions))
    # Pool held-out squared errors; never average ratios with unequal denominators.
    agg=q.groupby(['Region','model'])[['SSE','baseline_SSE']].sum();agg['skill']=1-agg.SSE/agg.baseline_SSE;v=agg.skill.unstack().reindex(REG)
    table('S13_pooled_skill',v.reset_index())
    f,a=newfig(h=104);f.subplots_adjust(left=.085,right=.90,bottom=.28,top=.92,wspace=.4)
    vv=v[['Climate modes only','All information']];limit=max(1,abs(vv.to_numpy()).max());im=heat(a[0,0],vv.values,['Climate\nmodes','All\ninformation'],REG,-limit,limit,annot=True);label(a[0,0],'a');a[0,0].set_title('Held-out SST reconstruction');cb=f.colorbar(im,ax=a[0,0],fraction=.04,pad=.03);cb.set_label('Skill versus seasonal trend')
    removals=['No ENSO','No other modes','No prior thermal state','No surface heating','No horizontal advection'];dv=np.stack([v['All information']-v[r] for r in removals],axis=1);limit=max(.05,np.max(abs(dv)));im=heat(a[0,1],dv,['ENSO','Other modes','Prior state','Heating','Advection'],REG,-limit,limit,annot=True);label(a[0,1],'b');a[0,1].set_title('Information-removal sensitivity');a[0,1].tick_params(axis='x',labelrotation=65);cb=f.colorbar(im,ax=a[0,1],fraction=.04,pad=.03);cb.set_label('Full minus reduced skill')
    finish(f,13,'Temporal generalization and information-removal tests',
        'a, Out-of-period regional SST reconstruction using climate-mode indices alone or all measured information. b, Change in held-out skill after removing each predictor group; positive values indicate that removal worsens reconstruction. Three forward validation blocks are 2003–2008, 2009–2014 and 2015–2020, each trained on all available earlier months starting in 1985.',
        'Fig. 4','New supplementary diagnostic. Seasonal terms, linear trend, scaling and regression coefficients are fitted only on the training period. Skill = 1 − held-out SSE / SSE of the training-fitted seasonal-and-trend baseline; pooled scores sum SSE before taking the ratio. Test predictors include contemporaneous observed indices/heating/advection, so this is held-out diagnostic reconstruction, NOT an operational forecast or validation of the AUC Sankey. Negative values are retained. No model selection or regularization tuning.',
        'new forward-held-out diagnostic; not AUC attribution', 'OLS with month indicators and monthly linear trend; seven model configurations; all seven regions and three test blocks.')

def fig14():
    d=csv('F4','Figure4b_paired_errors.csv');c=csv('F4','Figure4b_regional_changes.csv');om=csv('F4','SI_bridge_leave_event_out.csv');s=csv('F4','SI_bridge_correlation_sensitivity.csv')
    f,a=newfig(1,3,h=92);f.subplots_adjust(left=.08,right=.97,wspace=.62,bottom=.19)
    ax=a[0,0];label(ax,'a');r=d[d.Region!='TP']
    for ev,col in zip([1997,2015,2023],COLORS):
        z=r[r.Event==ev];ax.scatter(z.Z200_relative,z.Qnet_relative,s=14,color=col,alpha=.7,label=str(ev))
    fit=stats.linregress(r.Z200_relative,r.Qnet_relative);xx=np.linspace(r.Z200_relative.min(),r.Z200_relative.max(),100);ax.plot(xx,fit.intercept+fit.slope*xx,c=DARK,ls='--',lw=.8);ax.set(xlabel='Relative circulation RMSE',ylabel='Relative heat-flux RMSE',title='CFSv2 independent check');legend(ax,loc='upper left')
    ax=a[0,1];label(ax,'b');z=c.set_index('Region').loc[REG];lim=np.ceil(abs(z.to_numpy()).max()/10)*10;im=heat(ax,z.values,['Circulation','Heat flux'],REG,-lim,lim,annot=False);ax.tick_params(axis='x',labelrotation=40);ax.set_title('2023–24 error changes');cb=f.colorbar(im,ax=ax,fraction=.06,pad=.04);cb.set_label('RMSE change (%)')
    ax=a[0,2];label(ax,'c');ax.scatter(om.r,range(3),c=DARK,s=20);ax.set_yticks(range(3),['Omit '+str(x) for x in om.Omitted_event]);ax.invert_yaxis();ax.set(xlabel='Association, r',title='Event omission',xlim=(0,1));ax.axvline(fit.rvalue,c=RED,lw=.8,ls='--')
    check('S14 54 remote observations',len(r)==54)
    finish(f,14,'Independent CFSv2 circulation and surface heat-flux error check',
        'a, Paired circulation and net surface heat-flux RMSEs, each divided by the mean of the two historical events at the same region and lead. b, Target-event percentage RMSE changes in all seven regions. c, Remote-error association after omitting each event; the dashed line is the all-event estimate.',
        'Figs. 3–4','CFSv2-only, DJF, leads 1/3/6; 54 remote combinations from three events and six regions, plus nine TP combinations retained in b and data. Circulation uses departures from contemporaneous zonal mean against ERA5; net flux is verified against ORAS5, positive into the ocean. RMSE squares errors before averaging. This differs from Fig. 3 in model, circulation metric and ±20° region split. No independent-sample P value, causal claim or multimodel equivalence.',
        'independent single-model supporting diagnostic')

def fig15():
    d=csv('F4','association_weighted_AUC_allocation.csv');d=d[d.Profile=='Archived'];r=csv('F4','regional_AUC_accounting.csv');r=r[r.Profile=='Archived'].set_index('Region');order=REG+['Other seas'];r=r.loc[order]
    f,a=newfig(h=105);f.subplots_adjust(left=.12,right=.98,bottom=.19,wspace=.5)
    ax=a[0,0];label(ax,'a');ax.barh(range(8),r.Deficit,color=[BLUE if v<0 else RED for v in r.Deficit],height=.64);ax.set_yticks(range(8),order);ax.invert_yaxis();ax.axvline(0,c=DARK,lw=.6);ax.set(xlabel='Contribution to global AUC deficit',title='Signed regional accounting')
    ax=a[0,1];label(ax,'b');parts=['ENSO','Other','Memory','Atmosphere','Ocean','Unallocated'];names=['ENSO','Other modes','Prior thermal state','Surface heating','Advection','Unassigned'];colors=[RED,PURPLE,BLUE,'#4393C3','#92C5DE','#AAAAAA'];pos=[];neg=[]
    for p in parts:
        z=d[d.Component==p].Allocated_deficit;pos.append(z.clip(lower=0).sum());neg.append(z.clip(upper=0).sum())
    ax.barh(range(6),pos,color=colors,height=.6);ax.barh(range(6),neg,color=colors,height=.6,hatch='///',edgecolor=DARK,lw=.25);ax.set_yticks(range(6),names);ax.invert_yaxis();ax.axvline(0,c=DARK,lw=.6);ax.set(xlabel='Allocated AUC amount',title='Reductions and counterbalances')
    check('S15 exact allocation closure',abs(d.Allocated_deficit.sum()-r.Deficit.sum())<1e-10)
    table('S15_signed_components',pd.DataFrame({'component':parts,'reduction':pos,'counterbalance':neg,'net':np.array(pos)+np.array(neg)}))
    finish(f,15,'Signed accounting and closure of the retained skill deficit',
        'a, Each region’s signed contribution to the retained global AUC deficit, including the Other seas remainder. b, Positive and negative amounts for every leaf component of the accounting. Hatched negative bars counterbalance positive reductions; no absolute-value normalization is applied.',
        'Fig. 4','Regional amounts allocate positive–negative ranking pairs, not independent basin AUCs. Component amounts multiply regional deficit by fixed historical SST-association weights. Exact closure is algebraic, not evidence of predictive accuracy or causality. Local children are shown without adding the parent again. The unassigned amount is not a fitted unexplained forecast-error residual.',
        'archived AUC; association-weighted accounting only')

def audit01():
    d=csv('F4','AUC_deficit_reconciliation.csv');e=csv('F1','Figure1_source_events.csv');s=csv('F4','event_amplitudes_and_strict_AUC.csv')
    f,a=newfig(h=86);ax=a[0,0];label(ax,'a');x=np.arange(3);ax.bar(x-.15,d.expected,width=.28,color=DARK,label='Expected');ax.bar(x+.15,d.observed,width=.28,color=RED,label='Realized');ax.set_xticks(x,['Retained\narchive','Matched starts\nold filter','Matched starts\ncurrent filter']);ax.set(ylim=(0,.85),ylabel='AUC',title='Input-selection sensitivity');legend(ax,loc='upper left')
    ax=a[0,1];label(ax,'b');z=e[['Event','Mean_AUC']].merge(s[['Event','Mean_AUC']],on='Event',suffixes=('_archive','_matched'));ax.plot(range(10),z.Mean_AUC_archive,'o-',c=GRAY,ms=3,label='Retained archive');ax.plot(range(10),z.Mean_AUC_matched,'o-',c=BLUE,ms=3,label='Matched starts');ax.set_xticks(range(10),[v[:4] for v in z.Event],rotation=55);ax.set(ylabel='Event-mean AUC',xlabel='El Niño start year',title='All ten events');legend(ax,loc='lower left')
    finish(f,'A1','Input-alignment discrepancy retained for author review',
        'a, Three previously computed input-selection profiles. b, Event means under the retained and initialization-matched inputs. Main Figs. 1 and 4 are not changed. This audit is separate from the numbered SI figures and must not be interpreted as a new adopted analysis.',
        'Figs. 1 and 4','The retained deficit is 0.10905; matched-source/current-filter deficit is 0.03158. Source-coordinate disagreements remain unresolved. Omitting this audit from the manuscript does not resolve the scientific issue. These profiles differ only according to the documented input audit, not result-directed event selection.',
        'author-review audit; unresolved input provenance')

def audit02():
    d3=nc('F3','seven_ocean_regions.nc')['mask'];d4=nc('F4','seven_ocean_regions.nc')['mask'];check('A02 disjoint Fig3 masks',int(d3.sum('region').max())<=1);check('A02 disjoint Fig4 masks',int(d4.sum('region').max())<=1)
    f=plt.figure(figsize=(180/25.4,105/25.4));cols=['#D94B3A','#2166AC','#4393C3','#8064A2','#B59ACB','#D2C2E2','#E7A659'];cmap=ListedColormap(cols);rows=[]
    for k,(m,title) in enumerate([(d3,'Fig. 3: ±23.5° split'),(d4,'Fig. 4: ±20° split')]):
        ax=f.add_axes([.055+k*.48,.43,.44,.49],projection=PROJ);arr=np.where(m.sum('region')>0,m.argmax('region')+1,np.nan);v,l=add_cyclic_point(arr,coord=m.lon.values);ax.pcolormesh(l,m.lat,v,transform=PC,cmap=cmap,vmin=.5,vmax=7.5,rasterized=True);mapframe(ax);ax.set_title(title);label(ax,chr(97+k))
    for reg in REG:
        x=d3.sel(region=reg).values;y=d4.sel(region=reg).values;rows.append(dict(Region=reg,Fig3_cells=int(x.sum()),Fig4_cells=int(y.sum()),changed_cells=int((x!=y).sum())))
    q=pd.DataFrame(rows);table('A02_mask_comparison',q);ax=f.add_axes([.12,.14,.78,.25]);label(ax,'c');ax.bar(range(7),q.changed_cells,color=cols);ax.set_xticks(range(7),REG);ax.set_ylabel('Changed 1° cells');ax.set_title('Membership differences, not overlapping regions')
    finish(f,'A2','Region definitions and cross-figure support differences',
        'a–b, Native 1° masks archived with the latest Fig. 3 and Fig. 4. c, Number of cells whose membership changes for each region. Each set is internally disjoint; the two sets are not identical.',
        'Figs. 3–4','The changed-cell counts are not area-weighted fractions. This comparison preserves both current main figures; recalculating all Fig. 4 quantities at ±23.5° remains a separate unresolved consistency task.',
        'author-review domain audit')

FUNCTIONS=[fig01,fig02,fig03,fig04,fig05,fig06,fig07,fig08,fig09,fig10,fig11,fig12,fig13,fig14,fig15,audit01,audit02]
def booklet(records,path,include_index=True):
    doc=fitz.open();toc=[]
    def page():
        p=doc.new_page(width=595.28,height=841.89)
        p.insert_font(fontname='Body',fontfile=str(FONTDIR/'LiberationSans-Regular.ttf'))
        p.insert_font(fontname='BodyBold',fontfile=str(FONTDIR/'LiberationSans-Bold.ttf'))
        return p
    if include_index:
        p=page()
        p.insert_text((42,48),'Supplementary figures | MHW forecast skill',fontsize=16,fontname='BodyBold')
        p.insert_text((42,70),'Review package | 11 November 2026',fontsize=10,fontname='Body')
        y=102
        for rec in records:
            display=('S'+str(rec['number'])) if isinstance(rec['number'],int) else rec['number']
            p.insert_text((42,y),display+'  '+rec['title'],fontsize=8.5,fontname='Body');y+=25
        note='Main Figures 1-4 are unchanged. This package separates archived-input sensitivities, atmospheric diagnostics, historical SST associations and independent CFSv2 checks. Visual completeness does not certify submission readiness. Captions define dependencies, missing support and interpretation limits.'
        p.insert_textbox(fitz.Rect(42,y+15,552,780),note,fontsize=9,lineheight=1.4,fontname='Body')
    for rec in records:
        p=page();toc.append([1,rec['id']+' | '+rec['title'],len(doc)])
        header=('Supplementary Fig. '+str(rec['number']) if isinstance(rec['number'],int) else 'Audit Fig. '+rec['number'])+' | '+rec['title']
        p.insert_textbox(fitz.Rect(42,30,553,82),header,fontsize=11,fontname='BodyBold')
        with fitz.open(OUT/(rec['id']+'.pdf')) as art:
            h=art[0].rect.height;w=art[0].rect.width;newh=min(h,475);neww=w*newh/h
            rect=fitz.Rect((595.28-neww)/2,90,(595.28+neww)/2,90+newh);p.show_pdf_page(rect,art,0)
        y=rect.y1+18
        txt=rec['caption']+'\n\n'+rec['statistics']+'\n\nStatus: '+rec['status']+'.'
        spare=p.insert_textbox(fitz.Rect(42,y,553,797),txt,fontsize=8.5,lineheight=1.25,fontname='Body')
        if spare<0:
            # Preserve all text on a continuation page, rather than shrinking below readable size.
            p2=page();p2.insert_text((42,42),header[:95]+' (caption)',fontsize=10,fontname='BodyBold')
            check(rec['id']+' caption fits',p2.insert_textbox(fitz.Rect(42,70,553,790),txt,fontsize=9,lineheight=1.3,fontname='Body')>=0)
        p.insert_text((290,824),str(p.number+1),fontsize=8,fontname='Body')
    doc.set_toc(toc);doc.set_metadata({'title':'Supplementary evidence for MHW forecast skill','author':'Qun Luo','subject':'Source-traceable review package; main figures unchanged'})
    doc.save(path,garbage=4,deflate=True);doc.close()

def package():
    records=sorted(RECORDS,key=lambda r:(not isinstance(r['number'],int),str(r['number']).zfill(2)))
    science=[r for r in records if isinstance(r['number'],int)];audit=[r for r in records if not isinstance(r['number'],int)]
    for arr,name in [(science,'Supplementary_Figures_with_captions.pdf'),(audit,'Input_and_domain_audit.pdf'),(records,'All_figures_review.pdf')]:booklet(arr,ROOT/name)
    doc=fitz.open()
    for r in science:
        with fitz.open(OUT/(r['id']+'.pdf')) as d:doc.insert_pdf(d)
    doc.save(ROOT/'Supplementary_Figures_only.pdf',garbage=4,deflate=True);doc.close()
    # Compact vector overview, three pages of five distinct figures; all full-size versions retained.
    overview=fitz.open()
    for start in range(0,len(science),5):
        p=overview.new_page(width=842,height=1191)
        p.insert_font(fontname='BodyBold',fontfile=str(FONTDIR/'LiberationSans-Bold.ttf'))
        for j,r in enumerate(science[start:start+5]):
            top=30+j*227;p.insert_text((25,top),f'S{r["number"]}: '+r['title'],fontsize=10,fontname='BodyBold')
            with fitz.open(OUT/(r['id']+'.pdf')) as d:p.show_pdf_page(fitz.Rect(25,top+10,817,top+210),d,0)
    overview.save(ROOT/'SI_overview.pdf',deflate=True)
    for i,p in enumerate(overview):p.get_pixmap(dpi=90,alpha=False).save(ROOT/'qa'/f'overview_{i+1}.png')
    overview.close()
    for source_name in ['METHODS.md','SOURCE_AUDIT.md','BRIDGE_METHODS.md','bridge_contract.json','bridge_statistics.json']:
        s=FOUR/source_name;d=ROOT/'provenance'/source_name
        if source_name=='SOURCE_AUDIT.md' and not s.exists():
            s=ROOT.parent/'figure4_auc_attribution_20261104_v9/SOURCE_AUDIT.md'
        if not d.exists():shutil.copy2(s,d)
        MANIFEST.append({'source':str(s),'frozen':str(d.relative_to(ROOT)),'sha256':sha(d)})
    unique={m['frozen']:m for m in MANIFEST};dump(ROOT/'source_manifest.json',list(unique.values()));dump(ROOT/'figure_registry.json',records)
    dump(ROOT/'qa'/'numerical_checks.json',CHECKS)
    dump(ROOT/'qa'/'render_checks.json',[{k:r[k] for k in ['id','outside_text','min_font_pt','width_mm','height_mm']} for r in records])
    coverage=[]
    for r in records:coverage.append({'figure':r['id'],'main_figure':r['main_figure'],'question':r['title'],'status':r['status']})
    pd.DataFrame(coverage).to_csv(ROOT/'claim_evidence_matrix.csv',index=False)
    (ROOT/'STATISTICS_AND_REPRODUCIBILITY.md').write_text('# Statistics and reproducibility\n\nSeed for new bootstrap: 20261111. All figure-source paths and SHA-256 hashes are in source_manifest.json. NA means North Atlantic in region columns; it is never parsed as missing. All archived negative or null results are retained. No significance symbols were added.\n\n'+ '\n\n'.join(f'## {r["id"]}\n\n{r["statistics"]}\n\n{r["calculation"]}' for r in records))
    gaps=[
      'Fig. 1/4 initialization-source disagreement: preserved as Audit A1, not silently resolved or ignored.',
      'Fig. 3 uses ±23.5° and Fig. 4 ±20°: Audit A2 documents the difference; main figures remain unchanged.',
      'Fig. 2 common-model, common-season source-field rerun is not available in the retained table; S05 is reference sensitivity, not that test.',
      'Archived intensity RMSE requires a source-aligned rebuild before it can serve as independent corroboration.',
      'No matched process-error-to-AUC model exists for the current SST-weighted Sankey. Old pathway-model CV and 96.4% grouped agreement are not reused.',
      'Complete local heat-budget forecast validation, especially vertical mixing and entrainment, is not supplied by the CFSv2 flux check.',
      'Forward-held-out SST reconstruction is new diagnostic evidence, not operational MHW forecast skill or causal attribution.',
      'The attached October manuscript and current chat Results differ from September project metadata. SI numbering is newly proposed; DOCX cross-references have not been edited.'
    ]
    (ROOT/'REMAINING_EVIDENCE_GAPS.md').write_text('# Evidence still needed before submission\n\n'+'\n'.join('- '+g for g in gaps)+'\n')
    (ROOT/'README.md').write_text('# Supplementary figures — 11 November 2026\n\n15 separately rendered SI figures, plus two separate input/domain audit figures. Main Fig.1–4 and all manuscripts are unchanged.\n\n## Files\n\n- `Supplementary_Figures_with_captions.pdf`: numbered SI with captions.\n- `Supplementary_Figures_only.pdf`: the 15 figures without captions.\n- `All_figures_review.pdf`: all SI plus the two audits.\n- `Input_and_domain_audit.pdf`: audit-only report.\n- `SI_overview.pdf`: three-page overview.\n- `figures/`: individual vector PDF, editable SVG, 300 dpi PNG and previews.\n- `captions/`, `derived_data/`, `source_data/`, `source_manifest.json`: traceable scientific content.\n\nThe package is complete as a review rendering of available evidence, not as proof that all submission-critical scientific gaps are resolved. See REMAINING_EVIDENCE_GAPS.md.\n\n## Reproduce\n\n`OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mhw_si_20261111 /home/luoq/miniconda3/envs/ocean_sst/bin/python build_si.py`\n\nStyle: 180 mm width, Liberation Sans, 6–7 pt body labels, main-figure grey/vermilion event encoding and identical map colour limits; no new smoothing.\n')
    dump(ROOT/'.paper-workflow'/'project.json',{'title':'Unexpectedly low marine heatwave forecast skill during the 2023–2024 El Niño','mode':'supplementary figures','date':'2026-11-11','baseline_main_figures':[str(MAIN),str(FOUR)],'manuscript_context':'current chat Results plus MS_MHW_forecast_20261019.docx','protected_main_figures':True,'status':'review package; evidence gaps explicit'})
    pd.DataFrame([{'status':'active','decision':'Preserve current main figures and all archived inputs; build separate numbered SI.'},{'status':'active','decision':'Use current main Fig1–4 aesthetics, real frozen arrays, no mock values.'},{'status':'exploratory','decision':'New forward-held-out regional SST diagnostic; do not label as AUC attribution.'},{'status':'needs_verification','decision':'Input coordinate discrepancy, unequal masks, common-cohort Fig2 comparison and process-error-to-AUC validation remain open.'}]).to_csv(ROOT/'.paper-workflow'/'decision-log.csv',index=False)
    check('15 SI and 2 audit figures',len(science)==15 and len(audit)==2)
    check('all sources unchanged',all(sha(ROOT/m['frozen'])==m['sha256'] for m in unique.values()))
    dump(ROOT/'qa'/'numerical_checks.json',CHECKS)
    print('PACKAGED',len(records),'figures; numerical checks',len(CHECKS),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--only',type=int,nargs='*');args=parser.parse_args()
    chosen=set(args.only or range(1,18))
    old=[]
    if (ROOT/'figure_registry.json').exists():old=json.loads((ROOT/'figure_registry.json').read_text())
    for i,fn in enumerate(FUNCTIONS,1):
        if i in chosen:fn()
    if args.only:
        ids={r['id'] for r in RECORDS};RECORDS.extend(r for r in old if r['id'] not in ids)
        if (ROOT/'source_manifest.json').exists():MANIFEST.extend(json.loads((ROOT/'source_manifest.json').read_text()))
    package()
