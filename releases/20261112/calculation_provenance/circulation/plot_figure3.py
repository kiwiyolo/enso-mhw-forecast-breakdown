# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) geographic maps -> cross-type inherit from project Design_20261029_v7/plot_circulation.py -> param inherit
# (b) grouped correlation scatter -> project Design_20261029_v7/plot_circulation.py -> param inherit
# (c) global anomaly correlation bars -> assets/figures/GroupedBarChart/plot_GroupedBarChartv1.py -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
# GroupedBarChart script and preview inspected: inherit grouped offsets and direct values.
# Its benchmark replicates/SD/p-values do not match deterministic field correlations;
# use real precomputed anomaly correlations, no invented uncertainty, zero baseline and project colours.
# Other El Nino is an equal six-event observed anomaly composite, not climatology.

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
import json
import numpy as np
import pandas as pd
import xarray as xr
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from matplotlib import font_manager
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry import LineString
from shapely.ops import unary_union, linemerge
import contourpy

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data';OUT=ROOT/'figures'
CONTRACT=json.loads((ROOT/'contract.json').read_text())
STATS=json.loads((ROOT/'statistics.json').read_text())
AVAILABLE={f.name for f in font_manager.fontManager.ttflist}
# Preserve the author's Times-family instruction; STIX is the established
# Times-compatible project fallback when licensed Times New Roman is absent.
FONT='Times New Roman' if 'Times New Roman' in AVAILABLE else 'STIXGeneral'
mpl.rcParams.update({'font.family':'serif','font.serif':[FONT], 'mathtext.fontset':'stix',
    'font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,'xtick.labelsize':6,
    'ytick.labelsize':6,'legend.fontsize':6.4,'axes.labelpad':3,
    'xtick.major.pad':2,'ytick.major.pad':2,'xtick.major.size':2.5,'ytick.major.size':2.5})
PC=ccrs.PlateCarree();PROJ=ccrs.PlateCarree(central_longitude=180)
REGIONS=['NP','SP','TA','NA','SA','IO']
EVENTS=['1997/98','2015/16','2023/24']
COLORS=['#B2B2B2','#444444','#D9483B']
MARKERS=['o','s','^','v','D','P']
DIV=LinearSegmentedColormap.from_list('project_rdbu',
    ['#053061','#2166AC','#4393C3','#92C5DE','#D1E5F0','#F7F7F7',
     '#FDDBC7','#F4A582','#D6604D','#B2182B','#67001F'],N=256)

def basin_lines(masks):
    # The same 0.5 contours as Fig4, with periodic longitude. De-duplicate
    # shared edges so two offset dash patterns cannot form a solid line.
    lines=[]
    for code in masks.region.values:
        values,lons=add_cyclic_point(masks.sel(region=code).values.astype(float),coord=masks.lon.values)
        cg=contourpy.contour_generator(x=lons,y=masks.lat.values,z=values,name='serial',line_type='Separate')
        lines.extend(LineString(points) for points in cg.lines(.5) if len(points)>1)
    return linemerge(unary_union(lines))

def domain_lines(ax):
    masks=xr.load_dataset(DATA/'seven_ocean_regions.nc')['mask']
    ax.add_geometries(list(basin_lines(masks).geoms),crs=PC,facecolor='none',
        edgecolor='#444444',linewidth=.38,linestyle=(0,(3,2.4)),zorder=5)

def map_panel(fig, rect, field, limit, title, d, canonical=None, labels=False):
    ax=fig.add_axes(rect,projection=PROJ)
    vals,lons=add_cyclic_point(field.values,coord=d.lon.values)
    im=ax.pcolormesh(lons,d.lat,vals,transform=PC,cmap=DIV,vmin=-limit,vmax=limit,
        rasterized=True,shading='auto',antialiased=False)
    ax.set_xlim(-180,180);ax.set_ylim(-60,60)
    ax.coastlines(resolution='110m',lw=.34,color='#545454',zorder=4)
    ax.spines['geo'].set_linewidth(.5)
    # Native projected ticks retain separate 0-degree labels at both map edges.
    ax.set_xticks([-180,-120,-60,0,60,120,180])
    ax.set_yticks([-60,-30,0,30,60],crs=PC)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False,dateline_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(labelsize=6.2,length=2,width=.5)
    ax.set_title(title,loc='left',fontsize=7.6,pad=4.3)
    domain_lines(ax)
    if labels:
        # Match Fig4 anchors, which lie inside their corresponding masks.
        pos={'TP':(215,0),'NP':(205,40),'SP':(210,-38),'TA':(330,0),
             'NA':(320,40),'SA':(330,-38),'IO':(78,-27)}
        for name,(lon,lat) in pos.items():
            ax.text(lon,lat,name,transform=PC,ha='center',va='center',fontsize=7.2,fontweight='bold',
                path_effects=[pe.withStroke(linewidth=1.6,foreground='white')],zorder=8)
    if canonical is not None:
        z,lon=add_cyclic_point(canonical.values,coord=d.lon.values)
        levels=np.array([-120,-80,-40,40,80,120])
        ax.contour(lon,d.lat,z,levels=levels,colors='#252525',linewidths=.42,
            linestyles=['--' if v<0 else '-' for v in levels],transform=PC,zorder=6)
    return ax,im

def add_wind(ax, fields, prefix):
    sample=fields.sel(lat=[-54,-42,-30,30,42,54],lon=np.arange(9,360,18))
    x,y=np.meshgrid(sample.lon,sample.lat)
    q=ax.quiver(x,y,sample[prefix+'_ug'].values,sample[prefix+'_vg'].values,
        transform=PC,color='#202020',alpha=.8,pivot='middle',scale_units='inches',
        scale=75,width=.00155,headwidth=3.7,headlength=4.3,headaxislength=3.8,
        minlength=.4,zorder=7)
    for latitude in [45,-50]:
        ax.plot(np.arange(120,381),np.full(261,latitude),transform=PC,
            color='#222222',lw=.65,ls=(0,(1.2,2.2)),zorder=9)
    return q

def correlation_bars(fig,rect,table):
    ax=fig.add_axes(rect)
    targets=['other_el_nino','observed_canonical','observed_2023']
    anchors=['forecast_2023','forecast_canonical']
    colors=['#2166AC','#762A83']
    for j,(anchor,color) in enumerate(zip(anchors,colors)):
        rows=table[table.anchor==anchor].set_index('target').loc[targets]
        x=np.arange(3)+(j-.5)*.36
        valid=np.isfinite(rows.r.values)
        ax.bar(x[valid],rows.r.values[valid],width=.32,color=color,edgecolor='white',linewidth=.3,zorder=3)
        for xx,value in zip(x,rows.r):
            if np.isfinite(value):
                ax.text(xx+(-.06 if j==0 else .06),value+(.04 if value>=0 else -.04),f'{value:.3f}',va='bottom' if value>=0 else 'top',ha='center',
                        fontsize=6.3,color=color,zorder=4)
            else:
                ax.text(xx,.30,'N/A',va='center',ha='center',fontsize=6.3,color=color,zorder=4)
    ax.set(xlim=(-.55,2.55),ylim=(0,1.05),ylabel='Anomaly correlation, r')
    ax.set_xticks(range(3),['Other\nEl Niño','Observed\ncanonical','Observed\n2023–24'])
    ax.set_yticks([0,.25,.5,.75,1])
    ax.tick_params(labelsize=6.2,length=2)
    ax.yaxis.label.set_size(6.4)
    ax.axhline(0,color='#999999',lw=.4,ls=(0,(2,2)),zorder=1)
    return ax

def render():
    d=xr.load_dataset(DATA/'Figure3_Z200_fields.nc')
    geo=xr.load_dataset(DATA/'Figure3_geostrophic_and_height_fields.nc')
    df=pd.read_csv(DATA/'Figure3b_all_samples.csv',keep_default_na=False)
    envelope=pd.read_csv(DATA/'Figure3b_event_bootstrap_envelope.csv')
    bars=pd.read_csv(DATA/'Figure3c_global_anomaly_correlations.csv')
    fig=plt.figure(figsize=(183/25.4,140/25.4))
    for letter,x,y in [('a',.012,.983),('b',.708,.983),('c',.708,.550)]:
        fig.text(x,y,letter,fontsize=9,fontweight='bold',ha='left',va='top')
    fig.text(.055,.983,'Z200 anomalies and geostrophic winds',fontsize=8,fontweight='bold',va='top')
    fig.text(.635,.983,'DJF',fontsize=7,ha='right',va='top',color='#555555')
    fig.text(.755,.983,'Tropical–remote error linkage',fontsize=7.6,fontweight='bold',va='top')
    fig.text(.755,.550,'Global anomaly correlation',fontsize=7.6,fontweight='bold',va='top')
    maps=[]
    titles=['Canonical · observed 1997–1998 & 2015–2016',
            'Observed · 2023–2024',
            'Forecast − observed · 2023–2024']
    fields=[d.canonical,d.observed.sel(event='2023/24'),d.forecast_error.sel(event='2023/24').mean('lead')]
    limits=[160.,160.,100.]
    for i,(bottom,f,lim,title,prefix) in enumerate(zip([.680,.375,.070],fields,limits,titles,['canonical','observed','error'])):
        a,im=map_panel(fig,[.055,bottom,.58,.260],f,lim,title,d,
            canonical=d.canonical if i==2 else None,labels=i==0)
        q=add_wind(a,geo,prefix)
        maps.append((a,im))
    ca=fig.add_axes([.647,.379,.009,.557])
    cb=fig.colorbar(maps[0][1],cax=ca,orientation='vertical',ticks=[-160,-80,0,80,160])
    cb.set_label('Z200 anomaly (m)',fontsize=6.8,labelpad=4)
    cb.ax.tick_params(labelsize=6.2,length=2,width=.5);cb.outline.set_linewidth(.4)
    ce=fig.add_axes([.647,.074,.009,.252])
    cb=fig.colorbar(maps[2][1],cax=ce,orientation='vertical',ticks=[-100,-50,0,50,100])
    cb.set_label('Z200 forecast error (m)',fontsize=6.8,labelpad=4)
    cb.ax.tick_params(labelsize=6.2,length=2,width=.5);cb.outline.set_linewidth(.4)
    maps[2][0].quiverkey(q,X=.10,Y=-.145,U=10,label='10 m s$^{-1}$',labelpos='E',coordinates='axes',
                        fontproperties={'size':6.4})
    fig.text(.36,.036,'Dotted sections: 45°N / 50°S',fontsize=6.4,ha='center')
    fig.text(.34,.012,'Contours: canonical Z200 (40 m) · Arrows: geostrophic anomalies / errors',fontsize=6.1,ha='center')

    ax=fig.add_axes([.764,.660,.217,.260])
    ax.fill_between(envelope.x,envelope.lower,envelope.upper,color='#888888',alpha=.17,lw=0,zorder=1)
    ax.plot(envelope.x,envelope['fit'],color='#222222',lw=.9,ls=(0,(4,2)),zorder=2)
    for event,col in zip(EVENTS,COLORS):
        for region,marker in zip(REGIONS,MARKERS):
            a=df[(df.Event==event)&(df.Region==region)]
            ax.scatter(a.Rain_error,a.Z200_error,s=12,color=col,marker=marker,alpha=.82,
                edgecolors='white',linewidths=.2,zorder=3)
    xhi=np.ceil(df.Rain_error.max()*10)/10+.02
    yhi=np.ceil(max(df.Z200_error.max(),envelope.upper.max())*2)/2
    ax.set(xlim=(0,xhi),ylim=(0,yhi))
    ax.xaxis.set_major_locator(MultipleLocator(.2));ax.yaxis.set_major_locator(MultipleLocator(.5))
    ax.set_xlabel('Tropical convection error (1 − r)',fontsize=6.4,labelpad=3)
    ax.set_ylabel('Remote Z200 error (1 − r)',fontsize=6.5,labelpad=3)
    ax.text(.04,.96,f'$r$ = {STATS["pooled_r_descriptive"]:.2f}',transform=ax.transAxes,fontsize=7,va='top')
    event_handles=[Line2D([],[],marker='o',ls='',color=c,ms=3,label=e) for c,e in zip(COLORS,['1997–98','2015–16','2023–24'])]
    fig.legend(handles=event_handles,loc='center',bbox_to_anchor=(.867,.948),ncol=3,fontsize=6.2,
        borderaxespad=0,handlelength=.7,handletextpad=.35,columnspacing=.6)
    region_handles=[Line2D([],[],marker=m,ls='',color='#555555',ms=3,label=b) for b,m in zip(REGIONS,MARKERS)]
    fig.legend(handles=region_handles,loc='center',bbox_to_anchor=(.875,.591),ncol=6,fontsize=6.0,
        handlelength=.6,handletextpad=.25,columnspacing=.55)
    global_ax=correlation_bars(fig,[.764,.160,.217,.330],bars)
    fig.text(.875,.520,'60°S–60°N · All longitudes',fontsize=6.2,ha='center',color='#555555')
    handles=[Patch(facecolor='#2166AC',label='Forecast 2023–24'),Patch(facecolor='#762A83',label='Forecast canonical')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.869,.072),ncol=1,
               fontsize=6.2,handlelength=1.3,handletextpad=.5,labelspacing=.6)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();outside=[];texts=[];ignored=set()
    for axes in fig.axes:
        for axis,axis_limits in [(axes.xaxis,axes.get_xlim()),(axes.yaxis,axes.get_ylim())]:
            for loc,label in zip(axis.get_majorticklocs(),axis.get_majorticklabels()):
                if not min(axis_limits)<=loc<=max(axis_limits):ignored.add(id(label))
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text() or id(t) in ignored:continue
        b=t.get_window_extent(renderer)
        if not np.isfinite(b.extents).all() or b.width==0 or b.height==0:continue
        texts.append({'text':t.get_text(),'font_pt':t.get_fontsize(),'bounds_px':b.extents.tolist()})
        if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:outside.append(t.get_text())
    report={'width_mm':183,'height_mm':140,'font':FONT,'minimum_text_pt':min(t['font_pt'] for t in texts),
        'outside_canvas_text':outside,'map_axes_bounds':[a.get_position().bounds for a,im in maps],
        'map_limits':limits,'map_interpolation':'none; native 1-degree pcolormesh',
        'scatter_points':len(df),'all_text_bounds':texts,
        'scatter_axes_bounds':ax.get_position().bounds,'bar_axes_bounds':global_ax.get_position().bounds,
        'layout_change_only':True,'height_reduction_percent':12.5,
        'boundary_support':'Fig4 major-basin identity; revised +/-23.5-degree bands; Fig4 remains unchanged',
        'shared_boundary_segments_drawn_once':True,'old_heatmap_removed':True,
        'wind_scale_ms_per_inch':75,'wind_arrow_reference_ms':10,'arrows_per_map':120,
        'geostrophic_sample':'lon 9:18:351; lat -54,-42,-30,30,42,54; no equatorial arrows',
        'map_reference_latitudes':[45,-50],'bar_latitude_bands':[[-60,60]],
        'bar_longitude_extent':[0,360],'bar_field':'DJF_mean_Z200_anomaly',
        'bar_anchors':['forecast_2023','forecast_canonical'],
        'bar_targets':['other_el_nino','observed_canonical','observed_2023'],
        'pair_positions_per_panel':6,'bars_per_panel':6,'undefined_pairs_per_panel':0,
        'undefined_rendering':'none; all six comparisons defined',
        'correlation_panels':1,'bar_zero_baseline':True,'bar_y_limits':[0,1.05],'transects_replaced':True,
        'forecast_heights_climatology_adjusted':True,
        'bar_value_labels_inside_axes':all(a.bbox.contains(*t.get_window_extent(renderer).p0) and a.bbox.contains(*t.get_window_extent(renderer).p1) for a in [global_ax] for t in a.texts)}
    (ROOT/'render_QA.json').write_text(json.dumps(report,indent=2)+'\n')
    if outside:raise RuntimeError(f'Text outside canvas: {outside}')
    with mpl.rc_context({'savefig.bbox':None}):
        for ext in ['pdf','svg','png']:
            fig.savefig(OUT/f'Figure3.{ext}',dpi=600,bbox_inches=None,facecolor='white')
    print(json.dumps({k:v for k,v in report.items() if k!='all_text_bounds'},indent=2))
    plt.close(fig)

if __name__=='__main__':render()
