# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# a,c,e: existing SI v5 heatmap visual system -> param inherit
# b,d,f: existing SI v5 / LineTrend/plot_trend.py -> param inherit
# Generic templates have incompatible publication-count estimands; no mock data used.
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
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})
def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import fitz
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent
for directory in ['figures','qa']:(ROOT/directory).mkdir(exist_ok=True)
from matplotlib import font_manager
for font in (OLD/'fonts').glob('*.ttf'):font_manager.fontManager.addfont(str(font))
mpl.rcParams['font.sans-serif']=['Liberation Sans']
mpl.rcParams['savefig.bbox']=None  # fixed 180-mm canvas; geometry checked below
sys.path.insert(0,str(OLD))
import package_si as pack
EVENTS=['1997/98','2015/16','2023/24']
variables=['sst','surface_stress_x','precipitation']
names=['SST','Zonal wind stress','Convection']
d=pd.read_csv(ROOT/'source_data/correlations.csv')
c=pd.read_csv(ROOT/'source_data/correlation_changes.csv')
mpl.rcParams.update({'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'axes.labelpad':3})
f=plt.figure(figsize=(180/25.4,140/25.4))
f.text(.51,.978,'Matched DJF comparison · NASA GEOS-S2S',ha='center',fontsize=9)
cm=LinearSegmentedColormap.from_list('shared_correlation',DIVERGING).copy();cm.set_bad('#DDDDDD')
ylow=min(-5,float(c.relative_drop_percent.min())-5);yhigh=max(100,float(c.relative_drop_percent.max())+5)
for i,(v,n) in enumerate(zip(variables,names)):
    y=.73-.265*i
    ax=f.add_axes([.13,y,.39,.18]);ax.text(-.16,1.05,chr(97+2*i),transform=ax.transAxes,fontweight='bold',fontsize=10)
    z=d[d.variable==v].pivot(index='event',columns='lead',values='r').reindex(EVENTS)
    im=ax.imshow(z.values,aspect='auto',vmin=-1,vmax=1,cmap=cm,interpolation='none')
    ax.set_yticks(range(3),['1997–98','2015–16','2023–24']);ax.set_xticks(range(9),range(1,10));ax.tick_params(length=0,pad=3)
    ax.set_title(n,pad=5);ax.set_xlabel('Lead month')
    for sp in ax.spines.values():sp.set_visible(False)
    for a,b in np.ndindex(z.shape):
        val=z.values[a,b]
        ax.text(b,a,'NA' if not np.isfinite(val) else f'{val:.2f}',ha='center',va='center',fontsize=6.5,color='white' if np.isfinite(val) and abs(val)>.65 else BLACK)
    ax=f.add_axes([.605,y,.365,.18]);ax.text(-.145,1.05,chr(98+2*i),transform=ax.transAxes,fontweight='bold',fontsize=9)
    for ref,col,mark in [('1997/98','#AAAAAA','o'),('2015/16','#333333','s'),('Two-event mean','#D94B3D','D')]:
        q=c[(c.variable==v)&(c.reference==ref)]
        ax.plot(q.lead,q.relative_drop_percent,color=col,marker=mark,ms=2.8,lw=1 if ref=='Two-event mean' else .8,label=ref.replace('/','–'))
    ax.axhline(0,color='#999999',ls='--',lw=.6);ax.set(xlim=(.7,9.3),ylim=(ylow,yhigh),xticks=[1,3,5,7,9],xlabel='Lead month',ylabel='Correlation decrease (%)')
    if i==0:handles,labels=ax.get_legend_handles_labels()
ca=f.add_axes([.13,.093,.39,.013]);cb=f.colorbar(im,cax=ca,orientation='horizontal',ticks=[-1,-.5,0,.5,1]);cb.set_label('Spatial correlation, r',labelpad=2)
f.legend(handles,labels,loc='lower center',bbox_to_anchor=(.78,.033),ncol=1,title='Historical reference',title_fontsize=7,handlelength=1.8,labelspacing=.15,fontsize=7)
for t in f.findobj(mpl.text.Text):
    if len(t.get_text())==1 and t.get_text().islower():t.set_fontsize(9)
    elif t.get_fontsize()>7:t.set_fontsize(7)
from final_polish import polish
polish(f,5)
f.canvas.draw();renderer=f.canvas.get_renderer();outside=[]
for t in f.findobj(mpl.text.Text):
    if t.get_visible() and t.get_text():
        box=t.get_window_extent(renderer)
        if box.x0<0 or box.y0<0 or box.x1>f.bbox.width or box.y1>f.bbox.height:outside.append(t.get_text())
assert not outside,outside
for ext in ['pdf','png','svg']:f.savefig(ROOT/f'figures/Supplementary_Fig_05.{ext}',dpi=300,bbox_inches=None)
plt.close(f)
meta=json.loads((ROOT/'statistics.json').read_text())
caption=(
 'a,c,e, Cosine-latitude-weighted spatial correlations between forecast and observed DJF anomalies '
 'for SST, zonal wind stress and convection, represented by precipitation. All variables use NASA GEOS-S2S, '
 'the same four ensemble members (1–4), the same target months and leads, and the same tropical Pacific '
 f'ocean mask (20°S–20°N, 140°E–100°W; {meta["n_common_cells"]:,} grid cells). '
 'The two historical strong El Niño events are shown separately. b,d,f, Relative correlation decrease: '
 '100 × (r_reference − r_2023–2024)/r_reference, using each historical event and their equal-event mean. '
 'Positive values indicate deterioration; negative values indicate improvement. All right-hand panels share '
 'the same vertical scale. Lead 9 is unavailable and remains blank/NA; no nearest-lead substitution is used.'
)
stats=(
 'Forecast initialization equals the verifying month minus the labelled lead; the internal S coordinate '
 'must match that initialization and L must equal lead + 0.5. Monthly anomalies are calculated against '
 '1992–2020, with separate lead-dependent model and observed climatologies using identical available '
 'calendar months. DJF is averaged before spatial correlation. The same ocean mask is used for all '
 '72 valid event–variable–lead correlations. Eight leads within each of three events are repeated evaluations, '
 'not independent events; reference curves are sensitivity comparisons, not confidence intervals. '
 'This is a controlled single-model comparison, not a multimodel attribution of MHW forecast-skill loss.'
)
if meta['excluded_months']:stats+=' Common baseline exclusions: '+', '.join(meta['excluded_months'])+'.'
title='Tropical forecast skill under matched model, season and lead conditions'
(ROOT/'caption.json').write_text(json.dumps(dict(title=title,role='Matched-model DJF comparison of ocean and atmospheric forecast correlations',caption=caption,statistics=stats),indent=2))
(ROOT/'caption.md').write_text('# Supplementary Fig. 5 | '+title+'\n\n'+caption+'\n\n'+stats+'\n')
pack.ROOT=ROOT;pack.OUT=ROOT/'figures';pack.FONT=OLD/'fonts'
pack.caption_pages([dict(number=5,id='Supplementary_Fig_05',title=title,caption=caption,statistics=stats)])
doc=fitz.open(ROOT/'individual_with_captions/Supplementary_Fig_05.pdf');doc[0].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(ROOT/'qa/caption_preview.png');doc.close()
(ROOT/'qa/render.json').write_text(json.dumps(dict(outside_text=outside,width_mm=180,height_mm=140,minimum_text_pt=6.5,shared_correlation_limits=[-1,1],shared_change_limits=[ylow,yhigh]),indent=2))
