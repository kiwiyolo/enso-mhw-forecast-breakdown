# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) exact v15 PDF crop -> preserved vector/raster artwork without regeneration
# (b) existing LineTrend style -> parameter inheritance, all 81 coefficients
# (c) GroupedBarChart/plot_GroupedBarChartv1.py -> cross-type parameter inheritance
# Bars + historical-reference range; no imported P values, fabricated data or random jitter.
# User's existing serif font/event palette overrides generic style recommendations.
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mhw-fig2-relative-mpl')
# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.use('Agg')
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
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import fitz,yaml
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
roots=yaml.safe_load(Path('/data3/luoq/FEIS/configs/paths.yaml').read_text())['roots']
previous=Path(roots['source'])/'research/papers/mhw-predictability/Figures/Figure2_relative_atmospheric_loss_20261029_v15'
source=previous/'figures/Figure2.pdf'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lineage=json.loads((DATA/'input_lineage.json').read_text())
assert all(sha(Path(r['path']))==r['sha256'] for r in lineage)
summary=pd.read_csv(DATA/'relative_loss_summary.csv')
leads=pd.read_csv(DATA/'relative_loss_by_lead.csv')
correlations=pd.read_csv(DATA/'pattern_correlations.csv')
sensitivity=pd.read_csv(DATA/'historical_reference_sensitivity.csv')
assert len(summary)==3 and len(leads)==27 and len(correlations)==81 and len(sensitivity)==6
mpl.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],
    'mathtext.fontset':'stix','font.size':7,'axes.labelsize':7.5,'axes.titlesize':7.8,
    'xtick.labelsize':6.8,'ytick.labelsize':7.3,'legend.fontsize':6.5,'savefig.bbox':None})
W,H=183.,107.
fig=plt.figure(figsize=(W/25.4,H/25.4))
fig.patch.set_alpha(0)
RED='#E54737';DARK='#242424'
def axes(left,top,width,height):
    return fig.add_axes([left/W,1-(top+height)/H,width/W,height/H])
def text_mm(x,y,s,**kw):
    return fig.text(x/W,1-y/H,s,va='top',**kw)
text_mm(126,2,'b',fontsize=8.5,fontweight='bold')
text_mm(158.5,2,'Pattern fidelity',fontsize=7.8,ha='center')
processes=['SST','Zonal wind stress','Convection proxy']
curve_rows=[]
for j,process in enumerate(processes):
    a=axes(137,10+16.5*j,43,13.5)
    t=correlations[correlations.Process.eq(process)].pivot(index='Lead',columns='Event',values='Pattern_correlation').sort_index()
    historical=t[['1997/98','2015/16']]
    mean=historical.mean(axis=1);low=historical.min(axis=1);high=historical.max(axis=1)
    band=a.fill_between(t.index,low,high,color='#D8D8D8',alpha=.6,lw=0,label='Historical range')
    hist,=a.plot(t.index,mean,'o-',color=DARK,ms=2.5,lw=.85,label='Historical mean')
    target,=a.plot(t.index,t['2023/24'],'o-',color=RED,ms=2.5,lw=.85,label='2023–2024')
    a.set_xlim(.65,9.35);a.set_ylim(-.05,1.02)
    a.set_xticks([1,3,5,7,9]);a.set_yticks([0,.5,1])
    a.tick_params(labelsize=6.2,length=2,pad=1.3,labelbottom=j==2)
    if j==0:
        a.legend(handles=[hist,target],loc='lower left',bbox_to_anchor=(.005,.015),
                 fontsize=5.6,handlelength=1.5,handletextpad=.35,borderaxespad=.08,
                 labelspacing=.65,frameon=False)
    if j==2:a.set_xlabel('Lead month',fontsize=6.8,labelpad=4.5)
    if j==0:
        a.text(.97,.10,'SST',transform=a.transAxes,fontsize=6.1,ha='right',va='bottom')
    else:
        a.text(.035,.10,['','Wind stress','Precipitation'][j],transform=a.transAxes,
               fontsize=6.1,va='bottom')
    for lead in t.index:
        curve_rows.append(dict(Process=process,Lead=int(lead),historical_mean=float(mean[lead]),
                               historical_min=float(low[lead]),historical_max=float(high[lead]),target=float(t.loc[lead,'2023/24'])))
text_mm(126,66,'c',fontsize=8.5,fontweight='bold')
text_mm(158.5,66,'Relative fidelity loss',fontsize=7.8,ha='center')
ax=axes(137,74,43,26)
ax.set_xlim(-.55,2.55);ax.set_ylim(0,65)
ax.set_xticks([0,1,2],['SST','Wind','Precip.'])
ax.set_yticks([0,20,40,60]);ax.tick_params(labelsize=6.2,length=2,pad=1.7)
ax.set_ylabel('Correlation decline (%)',fontsize=6.8,labelpad=2)
ref=float(summary.loc[summary.Process.eq('SST'),'Relative_decline_percent'].iloc[0])
ax.axhline(ref,color='#999999',ls=(0,(3,3)),lw=.55,zorder=0)
bars=[];reference_points=[]
for x,process in enumerate(processes):
    row=summary[summary.Process.eq(process)].iloc[0]
    g=sensitivity[sensitivity.Process.eq(process)]
    color='#8A8A8A' if process=='SST' else RED
    value=float(row.Relative_decline_percent)
    lo=float(g.Relative_decline_percent.min());hi=float(g.Relative_decline_percent.max())
    assert lo<=value<=hi
    ax.bar(x,value,width=.48,color=color,alpha=.78,edgecolor='none',zorder=1)
    ax.errorbar(x,value,yerr=[[value-lo],[hi-value]],fmt='none',ecolor=DARK,
                elinewidth=.8,capsize=2,capthick=.8,zorder=3)
    # The two actual historical-reference values, not pseudoreplicate leads.
    for r,dx in zip(g.itertuples(),[-.12,.12]):
        ax.scatter([x+dx],[r.Relative_decline_percent],s=7,facecolors='white',
                   edgecolors=DARK,linewidths=.4,zorder=4)
        reference_points.append(dict(Process=process,Historical_event=r.Historical_event,
                                     x=x+dx,y=float(r.Relative_decline_percent)))
    label=f'{value:.1f}%'
    ax.text(x,hi+3.5,label,color=DARK,fontsize=6.5,va='bottom',ha='center')
    bars.append(dict(Process=process,x=x,height=value,error_low=lo,error_high=hi,
                     relative_to_SST=float(row.Relative_to_SST),label=label))
fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
for t in fig.findobj(mpl.text.Text):
    if not t.get_visible() or not t.get_text():continue
    b=t.get_window_extent(renderer)
    if not np.isfinite([b.x0,b.x1,b.y0,b.y1]).all() or b.width==0:continue
    if b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height:outside.append(t.get_text())
assert not outside,outside
fig.savefig(OUT/'Panels_bc.pdf',bbox_inches=None)
plt.close(fig)

base=fitz.open(source);pt=72/25.4
upper=fitz.open();upper.insert_pdf(base)
upper[0].add_redact_annot(fitz.Rect(124*pt,0,183*pt,155*pt),fill=False)
upper[0].add_redact_annot(fitz.Rect(0,100*pt,124*pt,155*pt),fill=False)
upper[0].apply_redactions(images=0,graphics=0,text=0)
upper[0].set_cropbox(fitz.Rect(0,0,124*pt,100*pt))
c=fitz.open(OUT/'Panels_bc.pdf')
result=fitz.open();page=result.new_page(width=W*pt,height=H*pt)
page.show_pdf_page(fitz.Rect(0,0,124*pt,100*pt),upper,0)
page.show_pdf_page(page.rect,c,0)
result.set_metadata({'title':'Figure 2 | Atmospheric pattern-fidelity losses exceed the SST loss',
    'subject':'Historical-normalized correlation decline; descriptive comparison, not physical error or causal attribution'})
result.save(OUT/'Figure2.pdf',garbage=4,deflate=True)
page.get_pixmap(dpi=450,alpha=False).save(OUT/'Figure2.png')
page.get_pixmap(dpi=180,alpha=False).save(OUT/'Figure2_preview.png')
pd.DataFrame(curve_rows).to_csv(DATA/'plotted_b_curves.csv',index=False)
pd.DataFrame(reference_points).to_csv(DATA/'plotted_reference_points.csv',index=False)
(DATA/'plotted_bars.json').write_text(json.dumps(bars,indent=2)+'\n')
region=fitz.Rect(0,0,123.8*pt,99.8*pt)
oldpix=base[0].get_pixmap(dpi=300,clip=region,alpha=False)
newpix=page.get_pixmap(dpi=300,clip=region,alpha=False)
oldarray=np.frombuffer(oldpix.samples,np.uint8).reshape(oldpix.height,oldpix.width,3)
newarray=np.frombuffer(newpix.samples,np.uint8).reshape(newpix.height,newpix.width,3)
delta=np.abs(oldarray.astype(int)-newarray.astype(int))
changed=int(np.any(delta,axis=-1).sum())
# Repositioned PDF forms introduce sub-pixel antialiasing at a few coastline edges.
# Preserve embedded source image bytes and physical map dimensions, never resample.
assert changed/(oldpix.height*oldpix.width)<1e-4 and delta.max()<=8,(changed,delta.max())
def image_evidence(p):
    return sorted((i['digest'].hex(),i['width'],i['height']) for i in p.get_image_info(hashes=True)
                  if fitz.Rect(i['bbox']).intersects(fitz.Rect(0,0,124*pt,100*pt)))
assert image_evidence(base[0])==image_evidence(page),'Embedded map pixels changed'
assert all(sha(Path(r['path']))==r['sha256'] for r in lineage)
qa=dict(status='PASS render checks',baseline=str(source),baseline_sha256=sha(source),
    a_source_image_bytes_identical=True,a_physical_scale_unchanged=True,
    a_antialias_changed_pixels_300dpi=changed,a_max_channel_difference=int(delta.max()),
    b_curve_rows=27,summary_bars=3,historical_reference_points=6,
    excluded_leads=0,width_mm=W,height_mm=H,dpi=450,outside_text=outside,
    c_type='Vertical bars with asymmetric historical-reference range and two reference points',
    reference_line=ref,point_display='Two reference-event values with deterministic horizontal offsets',
    no_significance_test=True,all_sources_unchanged=True)
(ROOT/'render_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
print(json.dumps(qa,indent=2))
