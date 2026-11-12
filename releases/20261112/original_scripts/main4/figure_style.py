# Shared, preserved production style for the corrected real-data alternative.
# Figure 1 values are preserved; plotting logic is in plot_production.py.
# Author's layout and key-only annotation override generic template composition.
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/mhw-redesign-mpl')
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
import json
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'source_data'
OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
available={f.name for f in font_manager.fontManager.ttflist}
FONT=next((f for f in ['Arial','Liberation Sans','DejaVu Sans'] if f in available),'DejaVu Sans')
mpl.rcParams.update({'font.sans-serif':[FONT],'mathtext.fontset':'dejavusans','axes.labelpad':3,
                     'xtick.major.pad':2,'ytick.major.pad':2,'figure.facecolor':'white',
                     'hatch.linewidth':.35})
RED='#C53935';BLUE='#2469A6';GREEN='#648A5C';ORANGE='#E65B3B'
DIV=LinearSegmentedColormap.from_list('ocean_atmosphere',DIVERGING,N=256)
HEAT=LinearSegmentedColormap.from_list('mhw_intensity',['#FFF7CB','#FDD77B','#FAA84F','#ED6835','#D12D29','#8E1023'],N=256)
PC=ccrs.PlateCarree()
REPORT={}
PDF_BOOK=None

def label(fig,letter,x,y):
    fig.text(x,y,letter,ha='left',va='top',fontsize=8,fontweight='bold')

def title(fig,text):
    # Full figure titles belong in captions; preserve only panel labels on art.
    return None

def save(fig,name):
    fig.canvas.draw()
    # Check every non-empty visible text artist against the exact export canvas.
    renderer=fig.canvas.get_renderer();outside=[]
    ignored=set()
    for ax in fig.axes:
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            lo,hi=sorted(limits)
            for minor in [False,True]:
                locs=axis.get_minorticklocs() if minor else axis.get_majorticklocs()
                labels=axis.get_minorticklabels() if minor else axis.get_majorticklabels()
                for loc,t in zip(locs,labels):
                    if not ax.axison or not axis.get_visible() or not lo-1e-10<=loc<=hi+1e-10:ignored.add(id(t))
    for t in fig.findobj(mpl.text.Text):
        if not t.get_visible() or not t.get_text() or id(t) in ignored:continue
        b=t.get_window_extent(renderer)
        if not np.isfinite([b.x0,b.y0,b.x1,b.y1]).all():continue
        if b.width==0 or b.height==0:continue
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:
            # Matplotlib creates unused tick labels beyond limits; they are not rendered.
            if t.axes is not None and t in t.axes.get_xticklabels()+t.axes.get_yticklabels():continue
            outside.append(t.get_text())
    fonts=[t.get_fontsize() for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
    REPORT[name]={'width_mm':round(fig.get_figwidth()*25.4,2),'height_mm':round(fig.get_figheight()*25.4,2),
                  'font':FONT,'minimum_rendered_text_pt':min(fonts),'outside_canvas_text':outside,'axes':len(fig.axes)}
    # Override the inherited tight-bbox default: all three physical widths must
    # remain exactly 183 mm rather than varying with the visible artist bounds.
    with mpl.rc_context({'savefig.bbox':None}):
        for ext in ['pdf','svg','png']:
            fig.savefig(OUT/f'{name}.{ext}',dpi=450,bbox_inches=None,facecolor='white')
        if PDF_BOOK is not None:PDF_BOOK.savefig(fig,bbox_inches=None,dpi=450)
    plt.close(fig)


