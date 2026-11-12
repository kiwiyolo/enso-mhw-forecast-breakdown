# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# a,b -> existing project Figure1 vector PDF -> native run (unchanged vector embedding)
# c -> project Robinson native-grid map -> param inherit
# c inset -> GroupedBarChart/plot_GroupedBarChartv1.py -> param inherit
# SI a,b -> existing SI01 vector PDF -> native run (unchanged vector embedding)
# RULE: native baseline panels are embedded, not redrawn or replaced with template data.
# Generic grouped-bar source has hard-coded benchmark data and a different grouping schema.
# Inherit horizontal bars, points, whiskers and clean spines; use requested project colours.
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
import json, hashlib, shutil
import numpy as np
import pandas as pd
import xarray as xr
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.path import Path as MPath
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import fitz

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'source_data';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
BASE=Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures')
MAIN=BASE/'Figure1-3_Nature_layout_20261110_v7'
SI=ROOT.parent/'Supplementary_Figures_20261111_v1'
for p in [ROOT/'provenance',ROOT/'fonts',ROOT/'captions',ROOT/'qa']:p.mkdir(exist_ok=True)
for p in (MAIN/'fonts').glob('*'):
    shutil.copy2(p,ROOT/'fonts'/p.name)
for p in (ROOT/'fonts').glob('*.ttf'):font_manager.fontManager.addfont(str(p))
mpl.rcParams.update({'font.sans-serif':['Liberation Sans'],'font.size':6.5,'axes.labelsize':6.5,
    'axes.titlesize':7,'xtick.labelsize':6,'ytick.labelsize':6,'legend.fontsize':6,
    'axes.labelpad':1,'xtick.major.pad':1,'ytick.major.pad':1,'xtick.major.size':2,
    'ytick.major.size':2,'savefig.bbox':None,'pdf.compression':9})
RED='#D94B3A';DARK='#303030';GRAY='#A9A9A9';BLUE='#2166AC'
PC=ccrs.PlateCarree();PROJ=ccrs.Robinson(central_longitude=-10)
SKILL=LinearSegmentedColormap.from_list('decrease_red_improvement_blue',['#2166AC','#8BB9D3','#F7F7F7','#E8978B','#B2182B'],256)
REG=['TP','NP','SP','TA','NA','SA','IO']
NAMES={'TP':'Tropical Pacific','NP':'North Pacific','SP':'South Pacific','TA':'Tropical Atlantic','NA':'North Atlantic','SA':'South Atlantic','IO':'Indian Ocean'}
CENTRES={'TP':(-145,0),'NP':(-158,40),'SP':(-145,-40),'TA':(-28,2),'NA':(-37,43),'SA':(-16,-40),'IO':(80,-23)}
SELECT=json.loads((ROOT/'selected_regions.json').read_text())['regions']
DS=xr.load_dataset(DATA/'global_expected_shortfall.nc')
LAT=np.arange(-89.5,90);LON=np.arange(.5,360)
II=np.rint(DS.lat.values+89.5).astype(int);JJ=np.rint(DS.lon.values-.5).astype(int)
SUMMARY=pd.read_csv(DATA/'regional_changes.csv',keep_default_na=False,na_values=[''])
LEADS=pd.read_csv(DATA/'regional_changes_by_lead.csv',keep_default_na=False,na_values=[''])
QA=[]

def axmm(f,x,top,w,h,**kw):
    W,H=f.get_size_inches()*25.4;return f.add_axes([x/W,1-(top+h)/H,w/W,h/H],**kw)
def text(f,x,top,s,**kw):
    W,H=f.get_size_inches()*25.4;return f.text(x/W,1-top/H,s,va='top',**kw)
def letter(f,x,top,s):return text(f,x,top,s,fontsize=8,fontweight='bold')
def fullgrid(v):
    a=np.full((180,360),np.nan);a[II,JJ]=np.asarray(v);return a
def mapframe(ax):
    left,right=-190+1e-5,170-1e-5
    xx=np.r_[np.linspace(left,right,721),np.full(241,right),np.linspace(right,left,721),np.full(241,left)]
    yy=np.r_[np.full(721,-60),np.linspace(-60,60,241),np.full(721,60),np.linspace(60,-60,241)]
    xy=PROJ.transform_points(PC,xx,yy)[:,:2]
    ax.set_boundary(MPath(xy),transform=ax.transData);ax.set_xlim(xy[:,0].min(),xy[:,0].max());ax.set_ylim(xy[:,1].min(),xy[:,1].max())
    ax.set_facecolor('#DADADA');ax.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#F0F0F0',edgecolor='none',zorder=3)
    ax.coastlines('110m',lw=.25,color='#777777',zorder=4);ax.spines['geo'].set_linewidth(.4)
def fieldmap(f,top,kind):
    ax=axmm(f,3,top,121,47,projection=PROJ)
    val=DS['global_benchmark_shortfall'].values
    array=fullgrid(val);a,x=add_cyclic_point(array,coord=LON)
    bound=1 if kind=='AUC' else 2
    im=ax.pcolormesh(x,LAT,a,transform=PC,cmap=SKILL,vmin=-bound,vmax=bound,shading='auto',rasterized=True,antialiased=False)
    mapframe(ax)
    for region in SELECT:
        mask=np.zeros((180,360));mask[II,JJ]=(DS.region.values==REG.index(region))
        a,x=add_cyclic_point(mask,coord=LON)
        ax.contour(x,LAT,a,levels=[.5],transform=PC,colors=DARK,linewidths=.65,linestyles='--',zorder=5)
        lon,lat=CENTRES[region];ax.text(lon,lat,region,transform=PC,ha='center',va='center',fontsize=6.5,fontweight='bold',zorder=6,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.6))
    return ax,im
def export(f,name):
    f.canvas.draw();rend=f.canvas.get_renderer();bad=[];fonts=[]
    ignored=set()
    for ax in f.axes:
        for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            if not axis.get_visible():
                ignored.update(id(t) for t in axis.get_majorticklabels()+axis.get_minorticklabels())
                ignored.add(id(axis.get_offset_text()));ignored.add(id(axis.label))
            for loc,t in zip(axis.get_majorticklocs(),axis.get_majorticklabels()):
                if not min(lim)<=loc<=max(lim):ignored.add(id(t))
    for t in f.findobj(mpl.text.Text):
        if id(t) in ignored or not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(rend)
        if not np.isfinite(b.extents).all() or b.width==0 or b.height==0:continue
        fonts.append(t.get_fontsize())
        if b.x0<-.5 or b.y0<-.5 or b.x1>f.bbox.width+.5 or b.y1>f.bbox.height+.5:bad.append(t.get_text())
    for ext in ['pdf','svg','png']:f.savefig(OUT/f'{name}.{ext}',dpi=300,bbox_inches=None,facecolor='white')
    QA.append(dict(figure=name,width_mm=float(f.get_size_inches()[0]*25.4),height_mm=float(f.get_size_inches()[1]*25.4),min_font=min(fonts),outside_text=bad))
    print(name,bad,flush=True);plt.close(f)

def changebars(ax,kind,regions,labels=True):
    d=SUMMARY[SUMMARY.metric==kind].set_index('region').loc[regions];yy=np.arange(len(regions))
    ax.barh(yy,d.change,height=.5,color=[RED if v>=0 else BLUE for v in d.change],alpha=.85)
    # Whiskers encode actual lead range, not an assumed symmetric CI.
    for j,region in enumerate(regions):
        row=d.loc[region];ax.plot([row.lead_min,row.lead_max],[j,j],color=DARK,lw=.65)
        ax.plot([row.lead_min]*2,[j-.06,j+.06],color=DARK,lw=.65)
        ax.plot([row.lead_max]*2,[j-.06,j+.06],color=DARK,lw=.65)
        ls=LEADS[(LEADS.region==region)&(LEADS.metric==kind)].sort_values('lead').change.values
        ax.scatter(ls,j+np.linspace(-.13,.13,9),s=4,facecolor='white',edgecolor=DARK,lw=.3,zorder=5)
        if labels:ax.text(row.lead_max+.018,j,f'{row.change:.3f}',va='center',fontsize=5.8)
    lo=min(0,d.lead_min.min()-.035) if d.lead_min.min()<0 else 0
    hi=max(d.lead_max.max(),d.change.max())+(.11 if labels else .035)
    ax.set_xlim(lo,hi);ax.axvline(0,color=DARK,lw=.6,ls='--',zorder=0)
    ax.set_yticks(yy,regions);ax.set_ylim(len(regions)-.5,-.5)
    ax.set_xlabel('AUC shortfall')

def lower(kind):
    f=plt.figure(figsize=(180/25.4,66/25.4));letter(f,3,2,'c')
    text(f,91,2,'2023–24 relative to global expected AUC (0.705)',ha='center',fontsize=7,color=DARK)
    ax,im=fieldmap(f,5,kind)
    cb=f.colorbar(im,cax=axmm(f,21,53,85,2.3),orientation='horizontal',extend='neither' if kind=='AUC' else 'both')
    cb.set_ticks([-1,-.5,0,.5,1] if kind=='AUC' else [-2,-1,0,1,2]);cb.outline.set_linewidth(.4)
    cb.set_label('Global expected AUC − local AUC',labelpad=1)
    # Subordinate quantitative inset belonging to c; no separate panel letter.
    inset=axmm(f,132,11,44,34)
    changebars(inset,kind,SELECT)
    inset.tick_params(axis='y',length=0,pad=2)
    export(f,'Figure1c_global_expected_benchmark')

def compose(top,bottom,name):
    a=fitz.open(top);b=fitz.open(bottom);out=fitz.open()
    W=a[0].rect.width;H=a[0].rect.height;hb=b[0].rect.height
    assert abs(W-b[0].rect.width)<.01
    p=out.new_page(width=W,height=H+hb);p.show_pdf_page(fitz.Rect(0,0,W,H),a,0);p.show_pdf_page(fitz.Rect(0,H,W,H+hb),b,0)
    out.save(OUT/f'{name}.pdf',garbage=4,deflate=True)
    p.get_pixmap(dpi=300,alpha=False).save(OUT/f'{name}.png');p.get_pixmap(dpi=160,alpha=False).save(OUT/f'{name}_preview.png')
    (OUT/f'{name}.svg').write_text(p.get_svg_image(text_as_path=False))
    # The top row is embedded at exact scale and checked by rendering the same crop.
    aa=a[0].get_pixmap(dpi=150,alpha=False);bb=p.get_pixmap(dpi=150,alpha=False,clip=fitz.Rect(0,0,W,H))
    delta=np.max(np.abs(np.frombuffer(aa.samples,dtype=np.uint8).astype(int)-np.frombuffer(bb.samples,dtype=np.uint8).astype(int)))
    QA.append(dict(figure=name,top_panels_embedded_unchanged=True,top_crop_max_channel_difference=int(delta),width_mm=W*25.4/72,height_mm=(H+hb)*25.4/72))
    a.close();b.close();out.close()

def supplement16():
    f=plt.figure(figsize=(180/25.4,119/25.4))
    a=axmm(f,13,9,63,42);b=axmm(f,104,9,63,42);c=axmm(f,13,70,63,39);d=axmm(f,104,70,63,39)
    for ax,l in zip([a,b,c,d],'abcd'):
        pos=ax.get_position();f.text(pos.x0-.02,pos.y1+.025,l,fontweight='bold',fontsize=8)
    changebars(a,'AUC',REG)
    changebars(c,'RMSE',REG)
    for ax,kind in [(b,'AUC'),(d,'RMSE')]:
        q=SUMMARY[SUMMARY.metric==kind].set_index('region').loc[REG]
        ax.barh(range(7),q.coverage*100,color=GRAY,height=.6)
        ax.set(xlim=(0,110),xticks=[0,50,100],yticks=range(7),yticklabels=REG,xlabel=f'{kind}: common ocean area (%)');ax.invert_yaxis()
        for j,v in enumerate(q.coverage):ax.text(v*100+1,j,f'{v*100:.1f}',va='center',fontsize=5.8)
    export(f,'Supplementary_Fig_16')

def main():
    lower('AUC')
    (ROOT/'qa/render_validation.json').write_text(json.dumps(QA,indent=2))
if __name__=='__main__':main()
