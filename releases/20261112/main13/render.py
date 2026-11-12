# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# Fig1a, Fig2a, Fig3a,b -> existing project scientific plotting scripts -> param inherit
# Fig1b, Fig2b -> LineTrend/plot_trend.py -> param inherit (line styling only)
# Fig2c, Fig3c -> GroupedBarChart/plot_GroupedBarChartv1.py -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
# User-directed cosmetic redesign overrides template layout, fonts and colours.
# No fabricated template values, transformations or statistical tests are imported.

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
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import xarray as xr
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MPath
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import LineString
from shapely.ops import unary_union, linemerge
import contourpy
import fitz

ROOT = Path(__file__).resolve().parent
BASE = Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures')
SOURCE1 = BASE/'Design_20261029_v7'
SOURCE2 = BASE/'Figure2_relative_atmospheric_loss_20261029_v16'
SOURCE3 = BASE/'Figure3_compact_layout_20261104_v21'
SOURCE2MAP = BASE/'Revised_20260921/Figure3'
OUT = ROOT/'figures'
DATA = ROOT/'source_data'
for directory in [OUT, DATA, ROOT/'provenance', ROOT/'fonts']:
    directory.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
LINEAGE = []

def freeze(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(source) == sha(target), f'Frozen source changed: {target}'
    else:
        shutil.copy2(source, target)
    LINEAGE.append({'source':str(source), 'frozen':str(target.relative_to(ROOT)), 'sha256':sha(target)})
    return target

# Portable release: all calculation products are frozen locally.
for font in sorted((ROOT/'fonts').glob('*.ttf')):
    font_manager.fontManager.addfont(str(font))
FONT = 'Liberation Sans'
mpl.rcParams.update({
    'font.family':'sans-serif', 'font.sans-serif':[FONT], 'mathtext.fontset':'dejavusans',
    'font.size':6.5, 'axes.labelsize':6.5, 'axes.titlesize':7,
    'xtick.labelsize':6, 'ytick.labelsize':6, 'legend.fontsize':6,
    'axes.labelpad':2.8, 'xtick.major.pad':2, 'ytick.major.pad':2,
    'xtick.major.size':2, 'ytick.major.size':2, 'axes.linewidth':.55,
    'savefig.bbox':None, 'pdf.compression':9,
})
RED='#D94B3A'; DARK='#303030'; MID='#777777'; BLUE='#2166AC'
PC=ccrs.PlateCarree()
PROJ=ccrs.Robinson(central_longitude=-10)
DIV=LinearSegmentedColormap.from_list('preserved_Fig3_rdbu',
    ['#053061','#2166AC','#4393C3','#92C5DE','#D1E5F0','#F7F7F7',
     '#FDDBC7','#F4A582','#D6604D','#B2182B','#67001F'],N=256)
REPORT={}

def axmm(fig, x, top, w, h, **kwargs):
    W,H=fig.get_size_inches()*25.4
    return fig.add_axes([x/W, 1-(top+h)/H, w/W, h/H], **kwargs)

def txt(fig, x, top, s, **kwargs):
    W,H=fig.get_size_inches()*25.4
    return fig.text(x/W, 1-top/H, s, va='top', **kwargs)

def letter(fig, x, y, s):
    return txt(fig,x,y,s,fontsize=8,fontweight='bold')

def panel_letter(fig, ax, s):
    """Place every panel letter 3 mm left and 4.5 mm above its actual frame."""
    W,H=fig.get_size_inches()*25.4
    box=ax.get_position()
    label=letter(fig,box.x0*W-3,(1-box.y1)*H-4.5,s)
    label.set_gid('panel_'+s)
    return label

def export(fig, name):
    # Compact, consistent label ownership: each label follows its own tick envelope.
    # Keep every panel frame, font size, scale and plotted datum unchanged.
    for axes in fig.axes:
        axes.tick_params(axis='both',which='major',pad=1)
        axes.xaxis.labelpad=1
        axes.yaxis.labelpad=1
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    axis_labels=[]
    for i,axes in enumerate(fig.axes):
        for direction,axis,limits in [('x',axes.xaxis,axes.get_xlim()),('y',axes.yaxis,axes.get_ylim())]:
            label=axis.label
            if not label.get_visible() or not label.get_text():continue
            lb=label.get_window_extent(renderer)
            ticks=[t.get_window_extent(renderer) for loc,t in zip(axis.get_majorticklocs(),axis.get_majorticklabels())
                   if min(limits)<=loc<=max(limits) and t.get_visible() and t.get_text()]
            if direction=='x':
                edge=min([axes.bbox.y0]+[t.y0 for t in ticks]); gap=edge-lb.y1
            elif axis.get_label_position()=='right':
                edge=max([axes.bbox.x1]+[t.x1 for t in ticks]); gap=lb.x0-edge
            else:
                edge=min([axes.bbox.x0]+[t.x0 for t in ticks]); gap=edge-lb.x1
            other_axes=[j for j,other in enumerate(fig.axes) if other is not axes and lb.overlaps(other.bbox)]
            axis_labels.append({'axes_index':i,'direction':direction,'text':label.get_text(),
                'tick_gap_pt':float(gap*72/fig.dpi),'overlaps_other_axes':other_axes})
    outside=[]; records=[]; ignored=set()
    for axes in fig.axes:
        for axis,limits in [(axes.xaxis,axes.get_xlim()),(axes.yaxis,axes.get_ylim())]:
            for loc,label in zip(axis.get_majorticklocs(),axis.get_majorticklabels()):
                if not min(limits)<=loc<=max(limits): ignored.add(id(label))
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text() or id(t) in ignored:continue
        box=t.get_window_extent(renderer)
        if not np.isfinite(box.extents).all() or box.width==0 or box.height==0:continue
        records.append({'text':t.get_text(),'font_pt':t.get_fontsize(),'bounds_px':box.extents.tolist()})
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:outside.append(t.get_text())
    REPORT[name].update(width_mm=180, height_mm=float(fig.get_size_inches()[1]*25.4), font=FONT,
                        min_font_pt=min(t['font_pt'] for t in records), outside_text=outside, texts=records,
                        axis_labels=axis_labels,axis_label_pad_pt=1,tick_label_pad_pt=1,
                        axes_positions=[list(a.get_position().bounds) for a in fig.axes])
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=600,bbox_inches=None,facecolor='white')
    with fitz.open(OUT/f'{name}.pdf') as doc:
        doc[0].get_pixmap(dpi=180,alpha=False).save(OUT/f'{name}_preview.png')
    print(name, 'outside:', outside, flush=True)
    plt.close(fig)

def figure1():
    path=DATA/'Fig1'
    event=pd.read_csv(path/'Figure1_source_events.csv').merge(
        pd.read_csv(path/'Figure1_event_uncertainty.csv')[['Event','CI_low','CI_high']],on='Event',validate='one_to_one')
    hist=event[event.Event!='2023/24']; target=event[event.Event=='2023/24'].iloc[0]
    s=json.loads((path/'Figure1_statistics.json').read_text())
    band=pd.read_csv(path/'Figure1a_regression_prediction_band.csv')
    fig=plt.figure(figsize=(180/25.4,88/25.4))
    a=axmm(fig,14,9,60,66)
    panel_letter(fig,a,'a')
    a.fill_between(band.Peak_Nino34,band.PI_low,band.PI_high,color='#EAEAEA',zorder=0)
    a.plot(band.Peak_Nino34,band.Fit,color=DARK,lw=.8,zorder=1)
    for row in hist.itertuples():
        strong=row.Event in ['1997/98','2015/16']
        a.errorbar(row.Peak_Nino34,row.Mean_AUC,
            yerr=[[row.Mean_AUC-row.CI_low],[row.CI_high-row.Mean_AUC]],fmt='o',
            mfc=DARK if strong else 'white',mec=DARK if strong else MID,
            ecolor=DARK if strong else '#A5A5A5',markersize=3.7,elinewidth=.7,capsize=2,zorder=3)
    xt=target.Peak_Nino34
    a.errorbar(xt,target.Mean_AUC,yerr=[[target.Mean_AUC-target.CI_low],[target.CI_high-target.Mean_AUC]],
        color=RED,fmt='none',elinewidth=.85,capsize=2,zorder=4)
    a.scatter(xt,s['observed'],s=45,marker='*',color=RED,edgecolor='white',linewidth=.35,zorder=7)
    # Expected marker, comparison line and gap text removed by explicit request.
    # The observed event's original error bar is not a comparison line and stays.
    handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=MID,ms=3.7,label='Earlier El Niño'),
             Line2D([],[],marker='o',ls='',color=DARK,ms=3.7,label='Canonical strong El Nino'),
             Line2D([],[],marker='*',ls='',color=RED,ms=5.5,label='2023–24')]
    a.legend(handles=handles,loc='upper left',handlelength=1,handletextpad=.4,labelspacing=.5,borderpad=.35)
    a.set(xlim=(.25,2.95),ylim=(.40,.95),xlabel='Peak Niño3.4 anomaly (°C)',ylabel='Global MHW forecast skill (AUC)')
    a.set_xticks([.5,1,1.5,2,2.5]); a.set_yticks(np.arange(.4,1,.1))
    b=axmm(fig,84,9,94,57.5)
    panel_letter(fig,b,'b')
    m=pd.read_csv(path/'Figure1_monthly_t1_t9.csv',parse_dates=['time']).set_index('time')
    m=m.reindex(pd.date_range(m.index.min(),m.index.max(),freq='MS'))
    b.plot(m.index,m.AUC,color=DARK,lw=.75)
    for start,end,text,col in [('1997-07','1998-07','1997–98','#DDDDDD'),('2015-07','2016-07','2015–16','#DDDDDD'),('2023-07','2024-07','2023–24','#F4C7C1')]:
        b.axvspan(pd.Timestamp(start),pd.Timestamp(end),facecolor=col,alpha=.5,lw=0)
        xpos=pd.Timestamp(start)+(pd.Timestamp(end)-pd.Timestamp(start))/2
        b.text(xpos,1.035,text,transform=b.get_xaxis_transform(),ha='right' if text=='2023–24' else 'center',va='bottom',fontsize=6,color=RED if text=='2023–24' else DARK)
    b.set(ylim=(min(.4,np.floor(m.AUC.min()*20)/20),.95),ylabel='Forecast skill (AUC)')
    b.tick_params(axis='x',labelbottom=False); b.set_yticks(np.arange(.4,1,.1))
    strip=axmm(fig,84,68,94,7,sharex=b)
    dates=mdates.date2num(m.index.to_pydatetime());end=mdates.date2num((m.index[-1]+pd.DateOffset(months=1)).to_pydatetime())
    strip.pcolormesh(np.r_[dates,end],[0,1],np.ma.masked_invalid(m.Nino34.to_numpy()[None,:]),
        cmap='RdBu_r',vmin=-2.5,vmax=2.5,rasterized=True)
    strip.set_yticks([]);strip.spines['left'].set_visible(False)
    strip.annotate('Niño3.4',xy=(0,.5),xycoords='axes fraction',xytext=(-2.83,0),
        textcoords='offset points',ha='right',va='center',fontsize=6)
    strip.set_xticks([pd.Timestamp(f'{y}-01-01') for y in [1995,2005,2015,2024]])
    strip.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));strip.set_xlabel('Year');strip.set_xlim(dates[0],end)
    REPORT['Figure1']={'events':len(event),'all_event_whiskers':True,'fit_band_rows':len(band),
        'arrow_removed':True,'monthly_rows':len(m),'expected':s['expected'],'observed':s['observed'],
        'residual':s['residual'],'Nino34_limits':[-2.5,2.5],
        'main_panel_width_ratio':[60,94], 'shared_x_exact':b.get_xlim()==strip.get_xlim(),
        'panel_gap_mm':10,'series_strip_gap_mm':1.5,
        'observed_annotation_removed':True,
        'expected_marker_removed':True,'gap_annotation_removed':True,'comparison_line_removed':True,
        'main_and_strip_bottom_mm':[75,75],'main_and_series_top_mm':[9,9]}
    export(fig,'Figure1')

def figure2():
    path=DATA/'Fig2'
    ds=xr.load_dataset(path/'Figure3_source_process_maps_t1_t9.nc')
    agree=xr.load_dataset(path/'Figure3_model_sign_agreement.nc')
    correlations=pd.read_csv(path/'pattern_correlations.csv')
    summary=pd.read_csv(path/'relative_loss_summary.csv')
    sensitivity=pd.read_csv(path/'historical_reference_sensitivity.csv')
    fig=plt.figure(figsize=(180/25.4,98/25.4))
    counts={}; mapaxes=[]
    for row,(stem,display) in enumerate([('sst','SST'),('stress','Wind stress'),('precipitation','Convection')]):
        for col,group in enumerate(['comparable','target']):
            ax=axmm(fig,11+col*53.5,9+row*(52*50/120+2),52,52*50/120,projection=ccrs.PlateCarree(central_longitude=180))
            mapaxes.append(ax)
            name=f'{stem}_{group}_relative_error'
            field=ds[name]
            mesh=ax.pcolormesh(field.lon,field.lat,field,cmap='RdBu_r',vmin=-200,vmax=200,
                shading='auto',transform=PC,rasterized=True)
            ax.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#F2F2F2',edgecolor='#777777',linewidth=.25)
            ax.coastlines(resolution='110m',lw=.25,color='#666666')
            ax.set_extent((140,260,-25,25),crs=PC);ax.spines['geo'].set_linewidth(.45)
            mask=agree[name+'_sign_agreement_mask'].astype(bool).isel(lat=slice(None,None,3),lon=slice(None,None,3))
            lon,lat=np.meshgrid(mask.lon,mask.lat);selected=mask.values
            ax.scatter(lon[selected],lat[selected],s=.65,marker='.',color=DARK,alpha=.60,
                       linewidth=0,transform=PC,zorder=5)
            counts[name]=int(selected.sum())
            longitudes=(140,180,-140) if col==0 else (180,-140,-100)
            grid=ax.gridlines(draw_labels=True,xlocs=longitudes,ylocs=(-20,-10,0,10,20),
                linewidth=.2,color='#A8A8A8',linestyle=':')
            grid.top_labels=False;grid.right_labels=False;grid.bottom_labels=row==2;grid.left_labels=col==0
            grid.xlabel_style={'size':6};grid.ylabel_style={'size':6}
            grid.xpadding=1;grid.ypadding=1
            if row==0:ax.set_title('Canonical strong El Nino' if col==0 else '2023–2024',fontsize=6.5,pad=4)
            if row==0 and col==0:panel_letter(fig,ax,'a')
            if col==0:ax.text(.018,.96,display,transform=ax.transAxes,va='top',fontsize=6.3,color=BLUE,
                fontweight='bold',bbox={'facecolor':'white','edgecolor':'none','alpha':.80,'pad':.65},zorder=6)
    cax=axmm(fig,11,83,105.5,2)
    cb=fig.colorbar(mesh,cax=cax,orientation='horizontal',ticks=[-200,-100,0,100,200])
    cb.ax.tick_params(labelsize=6,length=2,pad=1.3);cb.outline.set_linewidth(.4)
    cb.set_label('Forecast − observed (% of observed anomaly magnitude)',fontsize=6.3,labelpad=2)
    processes=['SST','Zonal wind stress','Convection proxy']
    curves=[]
    for j,process in enumerate(processes):
        a=axmm(fig,128,9+16*j,50,14)
        if j==0:
            panel_letter(fig,a,'b')
            a.set_title('Spatial correlation',fontsize=6.5,pad=4)
        t=correlations[correlations.Process.eq(process)].pivot(index='Lead',columns='Event',values='Pattern_correlation').sort_index()
        h=t[['1997/98','2015/16']]
        a.fill_between(t.index,h.min(axis=1),h.max(axis=1),color='#D8D8D8',alpha=.6,lw=0)
        l1,=a.plot(t.index,h.mean(axis=1),'o-',color=DARK,ms=2,lw=.75,label='Canonical strong El Nino')
        l2,=a.plot(t.index,t['2023/24'],'o-',color=RED,ms=2,lw=.8,label='2023–24')
        a.set(xlim=(.65,9.35),ylim=(-.05,1.02));a.set_xticks([1,3,5,7,9]);a.set_yticks([0,.5,1])
        a.tick_params(labelbottom=j==2,length=2,pad=1.4)
        a.set_ylabel(['SST','Wind stress','Convection'][j],fontsize=6.2,labelpad=3)
        if j==0:a.legend(handles=[l1,l2],loc='lower left',fontsize=6,handlelength=1.2,
            handletextpad=.35,borderaxespad=.12,labelspacing=.2)
        if j==2:a.set_xlabel('Lead month',labelpad=2)
        for lead in t.index:curves.append([process,int(lead),float(h.mean(axis=1)[lead]),float(h.min(axis=1)[lead]),float(h.max(axis=1)[lead]),float(t.loc[lead,'2023/24'])])
    ax=axmm(fig,128,65,50,26)
    panel_letter(fig,ax,'c')
    ax.set(xlim=(-.55,2.55),ylim=(0,65),ylabel='Correlation decline (%)')
    ax.set_xticks([0,1,2],['SST','Wind','Convection']);ax.set_yticks([0,20,40,60])
    ref=float(summary.loc[summary.Process.eq('SST'),'Relative_decline_percent'].iloc[0])
    ax.axhline(ref,color='#999999',ls=(0,(3,3)),lw=.55,zorder=0)
    values=[]
    for x,process in enumerate(processes):
        row=summary[summary.Process.eq(process)].iloc[0];g=sensitivity[sensitivity.Process.eq(process)]
        value=float(row.Relative_decline_percent);lo=float(g.Relative_decline_percent.min());hi=float(g.Relative_decline_percent.max())
        ax.bar(x,value,width=.38,color=MID if x==0 else RED,alpha=.85,zorder=1)
        ax.errorbar(x,value,yerr=[[value-lo],[hi-value]],fmt='none',ecolor=DARK,elinewidth=.75,capsize=2,zorder=3)
        for r,dx in zip(g.itertuples(),[-.12,.12]):
            ax.scatter(x+dx,r.Relative_decline_percent,s=8,facecolors='white',edgecolors=DARK,linewidths=.4,zorder=4)
        ax.text(x,hi+3.5,f'{value:.1f}%',fontsize=6.2,ha='center',va='bottom')
        values.append({'value':value,'low':lo,'high':hi})
    REPORT['Figure2']={'map_shape':dict(ds.sizes),'map_limits':[-200,200],'smoothing':'none',
        'stipple_counts':counts,'stipple_subsample':'every third original lat/lon node',
        'curve_coefficients':len(correlations),'displayed_curves':curves,'bars':values,'reference_points':len(sensitivity),
        'whiskers':'two historical-reference extremes, not CI','right_column_edges_mm':[128,178],
        'map_right_column_gap_mm':11.5,'map_row_gap_mm':2,'map_column_gap_mm':1.5,
        'curve_row_gap_mm':2,'b_c_frame_gap_mm':10}
    export(fig,'Figure2')

def basin_lines(masks):
    lines=[]
    for code in masks.region.values:
        values,lons=add_cyclic_point(masks.sel(region=code).values.astype(float),coord=masks.lon.values)
        cg=contourpy.contour_generator(x=lons,y=masks.lat.values,z=values,name='serial',line_type='Separate')
        lines.extend(LineString(points) for points in cg.lines(.5) if len(points)>1)
    return linemerge(unary_union(lines))

def map_boundary(ax):
    left,right=-190+1e-5,170-1e-5
    xx=np.r_[np.linspace(left,right,721),np.full(241,right),np.linspace(right,left,721),np.full(241,left)]
    yy=np.r_[np.full(721,-60),np.linspace(-60,60,241),np.full(721,60),np.linspace(60,-60,241)]
    xy=PROJ.transform_points(PC,xx,yy)[:,:2]
    ax.set_boundary(MPath(xy),transform=ax.transData)
    ax.set_xlim(xy[:,0].min(),xy[:,0].max());ax.set_ylim(xy[:,1].min(),xy[:,1].max())

def figure3():
    path=DATA/'Fig3'
    d=xr.load_dataset(path/'Figure3_Z200_fields.nc');geo=xr.load_dataset(path/'Figure3_geostrophic_and_height_fields.nc')
    masks=xr.load_dataset(path/'seven_ocean_regions.nc')['mask'];edges=list(basin_lines(masks).geoms)
    df=pd.read_csv(path/'Figure3b_all_samples.csv',keep_default_na=False)
    envelope=pd.read_csv(path/'Figure3b_event_bootstrap_envelope.csv')
    bars=pd.read_csv(path/'Figure3c_global_anomaly_correlations.csv')
    stats=json.loads((ROOT/'provenance/original_Figure3_statistics.json').read_text())
    fig=plt.figure(figsize=(180/25.4,133/25.4))
    titles=['Observed canonical','Observed 2023–2024','Forecast − observed 2023–2024']
    fields=[d.canonical,d.observed.sel(event='2023/24'),d.forecast_error.sel(event='2023/24').mean('lead')]
    limits=[160,160,100]; maps=[]
    for i,(field,title,prefix,limit) in enumerate(zip(fields,titles,['canonical','observed','error'],limits)):
        ax=axmm(fig,7,8+i*41,99,36.5,projection=PROJ)
        vals,lons=add_cyclic_point(field.values,coord=d.lon.values)
        im=ax.pcolormesh(lons,d.lat,vals,transform=PC,cmap=DIV,vmin=-limit,vmax=limit,
            rasterized=True,shading='auto',antialiased=False)
        map_boundary(ax);ax.coastlines(resolution='110m',lw=.3,color='#545454',zorder=4)
        if i==0:panel_letter(fig,ax,'a')
        ax.spines['geo'].set_linewidth(.45)
        ax.set_title(title,loc='left',fontsize=6.5,pad=3)
        ax.add_geometries(edges,crs=PC,facecolor='none',edgecolor='#555555',linewidth=.3,
            linestyle=(0,(3,2.4)),zorder=5)
        grid=ax.gridlines(draw_labels={'left':'y','bottom':'x'},xlocs=[-120,-60,0,60,120],ylocs=[-60,-30,0,30,60],
            linewidth=0,rotate_labels=False,x_inline=False,y_inline=False)
        grid.xlabel_style={'size':6};grid.ylabel_style={'size':6};grid.xpadding=1;grid.ypadding=1
        if i<2:grid.bottom_labels=False
        if i==0:
            pos={'TP':(215,0),'NP':(205,40),'SP':(210,-38),'TA':(330,0),'NA':(320,40),'SA':(330,-38),'IO':(78,-27)}
            for name,(lon,lat) in pos.items():
                ax.text(lon,lat,name,transform=PC,ha='center',va='center',fontsize=6.5,fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=1.5,foreground='white')],zorder=8)
        if i==2:
            z,lon=add_cyclic_point(d.canonical.values,coord=d.lon.values)
            levels=np.array([-120,-80,-40,40,80,120])
            ax.contour(lon,d.lat,z,levels=levels,colors='#252525',linewidths=.38,
                linestyles=['--' if v<0 else '-' for v in levels],transform=PC,zorder=6)
        sample=geo.sel(lat=[-54,-42,-30,30,42,54],lon=np.arange(9,360,18))
        x,y=np.meshgrid(sample.lon,sample.lat)
        q=ax.quiver(x,y,sample[prefix+'_ug'].values,sample[prefix+'_vg'].values,
            transform=PC,color='#202020',alpha=.8,pivot='middle',scale_units='inches',scale=75,
            width=.00155,headwidth=3.7,headlength=4.3,headaxislength=3.8,minlength=.4,zorder=7)
        for lat in [45,-50]:ax.plot(np.arange(120,381),np.full(261,lat),transform=PC,
            color='#333333',lw=.55,ls=(0,(1.2,2.2)),zorder=9)
        maps.append((ax,im))
    for idx,top,height,ticks,label in [(0,8,77.5,[-160,-80,0,80,160],'200-hPa geopotential height anomaly (m)'),(2,90,36.5,[-100,-50,0,50,100],'200-hPa geopotential height\nForecast − observed (m)')]:
        cax=axmm(fig,109,top,1.6,height)
        cb=fig.colorbar(maps[idx][1],cax=cax,ticks=ticks)
        cb.set_label(label,fontsize=6.3,labelpad=3);cb.ax.tick_params(labelsize=6,length=2,pad=1.5);cb.outline.set_linewidth(.4)
    maps[2][0].quiverkey(q,X=.79,Y=1.055,U=10,label='10 m s$^{-1}$',labelpos='E',coordinates='axes',fontproperties={'size':6})
    # Restore the horizontal title and use actual field correlations on both axes.
    # Original error-space inputs are preserved; transformation is exact, not a refit.
    assert np.allclose(df.Rain_r,1-df.Rain_error,rtol=0,atol=1e-14)
    assert np.allclose(df.Z200_r,1-df.Z200_error,rtol=0,atol=1e-14)
    corr_envelope=pd.DataFrame({'x':1-envelope.x,'fit':1-envelope['fit'],
                               'lower':1-envelope.upper,'upper':1-envelope.lower}).sort_values('x')
    ax=axmm(fig,134,8,44,52)
    panel_letter(fig,ax,'b')
    ax.fill_between(corr_envelope.x,corr_envelope.lower,corr_envelope.upper,color='#888888',alpha=.17,lw=0,zorder=1)
    ax.plot(corr_envelope.x,corr_envelope['fit'],color=DARK,lw=.8,ls=(0,(4,2)),zorder=2)
    colors=['#B2B2B2',DARK,RED];markers=['o','s','^','v','D','P'];regions=['NP','SP','TA','NA','SA','IO']
    for event,col in zip(['1997/98','2015/16','2023/24'],colors):
        for region,marker in zip(regions,markers):
            rows=df[(df.Event==event)&(df.Region==region)]
            ax.scatter(rows.Rain_r,rows.Z200_r,s=12,color=col,marker=marker,alpha=.78,edgecolors='white',linewidths=.2,zorder=3)
    # Preserve all negative correlations and the full transformed regression envelope.
    # Extra empty lower margin accommodates legends without covering any samples.
    ax.set(xlim=(.55,.95),ylim=(-.65,1.10),
           xlabel='Tropical convection correlation (r)',ylabel='Z200 forecast correlation (r)')
    ax.set_xticks([.6,.7,.8,.9]);ax.set_yticks([-.5,0,.5,1])
    eventhandles=[Line2D([],[],marker='o',ls='',color=c,ms=3,label=e) for c,e in zip(colors,['1997–98','2015–16','2023–24'])]
    legend=ax.legend(handles=eventhandles,loc='lower left',fontsize=6,handlelength=.8,handletextpad=.3,labelspacing=.25,borderpad=.2)
    ax.add_artist(legend)
    regionhandles=[Line2D([],[],marker=m,ls='',color='#555555',ms=3,label=b) for b,m in zip(regions,markers)]
    ax.legend(handles=regionhandles,loc='lower right',ncol=2,fontsize=6,handlelength=.6,handletextpad=.3,
        columnspacing=.55,labelspacing=.3,borderpad=.2)
    ax.text(.04,.96,f'r = {stats["pooled_r_descriptive"]:.2f}',transform=ax.transAxes,fontsize=6.5,va='top')
    c=axmm(fig,134,72,44,54.5)
    panel_letter(fig,c,'c')
    targets=['other_el_nino','observed_canonical','observed_2023']
    for j,(anchor,color) in enumerate(zip(['forecast_2023','forecast_canonical'],[RED,DARK])):
        rows=bars[bars.anchor==anchor].set_index('target').loc[targets]
        x=np.arange(3)+(j-.5)*.36
        c.bar(x,rows.r,width=.32,color=color,edgecolor='white',linewidth=.3,zorder=3)
        for xx,value in zip(x,rows.r):
            c.text(xx+(-.05 if j==0 else .05),value+.035,f'{value:.3f}',ha='center',va='bottom',fontsize=6,color=color)
    c.set(xlim=(-.55,2.55),ylim=(0,1.30),ylabel='Anomaly spatial correlation, r')
    c.set_yticks([0,.25,.5,.75,1]);c.set_xticks(range(3),['Other\nEl Niño','Observed\ncanonical','Observed\n2023–24'])
    c.legend(handles=[Patch(facecolor=RED,label='Forecast 2023–24'),Patch(facecolor=DARK,label='Forecast canonical')],
        loc='upper left',fontsize=6,handlelength=1.1,labelspacing=.45,borderpad=.3)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    legend_boxes=[legend.get_window_extent(renderer),ax.get_legend().get_window_extent(renderer)]
    xy=ax.transData.transform(df[['Rain_r','Z200_r']].values)
    overlaps=sum(any(box.padded(2).contains(*p) for box in legend_boxes) for p in xy)
    inside=all(ax.bbox.contains(box.x0,box.y0) and ax.bbox.contains(box.x1,box.y1) for box in legend_boxes)
    cbbox=c.get_legend().get_window_extent(renderer)
    clegend_inside=c.bbox.contains(cbbox.x0,cbbox.y0) and c.bbox.contains(cbbox.x1,cbbox.y1)
    REPORT['Figure3']={'scatter_points':len(df),'bar_values':bars.to_dict('records'),
        'map_limits':limits,'projection':'Robinson, central_longitude=-10 (same as Figure4a)',
        'source_grid_unchanged':True,'map_smoothing':'none; native 1-degree pcolormesh',
        'vector_samples_per_map':120,'vector_locations':{'lat':[-54,-42,-30,30,42,54],'lon':list(range(9,360,18))},
        'vector_scale':75,'contour_levels':[-120,-80,-40,40,80,120],'reference_latitudes':[45,-50],
        'retained_mask_boundary':23.5,'Fig4_mask_boundary_unchanged':20,'bar_zero_baseline':True,
        'bar_ylim':[0,1.30],'bar_upper_padding_reason':'Place requested internal legend above data; ticks stop at r=1.',
        'right_column_edges_mm':[134,178],'scatter_r':stats['pooled_r_descriptive'],
        'scatter_x_limits':list(ax.get_xlim()),'scatter_y_limits':list(ax.get_ylim()),
        'scatter_columns':['Rain_r','Z200_r'],'scatter_axis_metric':'forecast-observation spatial correlation r',
        'negative_z200_samples':int((df.Z200_r<0).sum()),
        'plotted_correlations':df[['Event','Lead','Region','Rain_r','Z200_r']].to_dict('records'),
        'correlation_envelope':corr_envelope.to_dict('records'),
        'transformed_r':float(np.corrcoef(df.Rain_r,df.Z200_r)[0,1]),
        'internal_legends_inside_axes':bool(inside and clegend_inside),
        'scatter_points_under_legends':int(overlaps)}
    export(fig,'Figure3')

if __name__=='__main__':
    figure1()
    figure2()
    figure3()
    (ROOT/'render_QA.json').write_text(json.dumps(REPORT,indent=2)+'\n')
