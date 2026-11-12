# Academic Figure Skill Asset Confirmation
# All panels inherit the existing source-verified SI renderers; cosmetic changes only.
# Baseline typography, palette and export are injected by base_si.py.
"""Author-requested final wording, geometry and panel-label pass."""
from pathlib import Path
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def polish(fig,n):
    a=fig.axes
    def place(i,r):a[i].set_position(r)
    if n==1:
        fig.set_size_inches(180/25.4,128/25.4)
        place(0,[.095,.64,.30,.30]);place(1,[.51,.64,.47,.30])
        place(2,[.025,.175,.67,.38]);place(3,[.10,.105,.52,.016]);place(4,[.77,.175,.21,.38])
        a[4].set_title('')
    if n==2:
        for ax in a:
            for t in ax.texts:
                if '95% interval:' in t.get_text():t.set_text(t.get_text().split('\n')[0])
    if n==3:
        a[0].set(xlabel='El Niño event year',title='Historical forecast skill')
        a[1].set(xlabel='Niño3.4 anomaly after\nremoving its time trend (°C)',ylabel='AUC after removing\nits time trend',title='Removing the time trend')
        a[2].set(xlabel='AUC change per 1°C\nstronger El Niño',title='Exclude each event in turn')
        a[2].set_ylabel('Excluded event',labelpad=2)
        place(0,[.10,.22,.235,.66]);place(1,[.44,.22,.235,.66]);place(2,[.81,.22,.17,.66])
        handles,labels=a[2].get_legend_handles_labels()
        a[2].legend(handles,labels,loc='lower left',bbox_to_anchor=(0,-.025),fontsize=7,frameon=False)
    if n==4:a[1].set(xlabel='Month being forecast',title='Monthly forecast skill')
    if n==6:
        fig.set_size_inches(180/25.4,99/25.4)
        for row,y in enumerate([.715,.455,.195]):
            for col in range(3):
                ax=a[3*row+col];ax.set_position([.075+col*.303,y,.287,.21])
                gl=ax.gridlines(draw_labels=True,xlocs=[-120,0,120],ylocs=[-30,0,30],linewidth=0)
                gl.top_labels=False;gl.right_labels=False;gl.left_labels=col==0;gl.bottom_labels=row==2
                gl.geo_labels=False;gl.xlabel_style={'size':6};gl.ylabel_style={'size':6}
                gl.xpadding=1;gl.ypadding=1;gl.rotate_labels=False
        place(9,[.09,.087,.55,.018]);place(10,[.73,.087,.22,.018])
        for i,t in enumerate(fig.texts):
            if len(t.get_text())>1:t.set_position((.005,[.82,.56,.30][i]));t.set_fontsize(7)
    if n==8:
        a[0].set_title('Forecast–observation correlation');a[1].set_title('Upper-level circulation error')
        a[1].set_xlabel('200-hPa geopotential-height\nforecast RMSE (m)')
        place(0,[.25,.23,.31,.65]);place(1,[.65,.23,.31,.65])
        for leg in fig.legends:leg.set_bbox_to_anchor((.5,.06))
    if n==9:
        for ax in a[:3]:ax.set_xlabel('Tropical convection correlation, r');ax.set_ylabel('Remote circulation correlation, r')
        a[3].set_title('Dependence on the event sample');a[3].set_xlabel('Tropical–remote correlation, r')
        a[3].set_yticklabels(['All three events','Without 1997–98','Without 2015–16','Without 2023–24'])
    if n==10:
        for ax in a:
            if ax.get_legend():ax.get_legend().remove()
        fig.legend(handles=[Line2D([],[],color='#303030',marker='o',label='1997–98 / 2015–16 forecast'),Line2D([],[],color='#D94B3A',marker='o',label='2023–24 forecast')],loc='lower center',bbox_to_anchor=(.53,.08),ncol=2,fontsize=7)
        place(1,[.105,.20,.37,.27]);place(2,[.61,.20,.37,.27])
    if n==11:
        for leg in fig.legends:leg.set_bbox_to_anchor((.5,.10))
    if n==12:
        for leg in fig.legends:leg.set_bbox_to_anchor((.5,.065))
    if n==13:
        a[0].set_title('SST estimates in years not used for fitting',pad=9)
        a[1].set_title('How much does each variable improve SST estimates?',pad=9)
        a[3].set_ylabel('Decrease in skill after\nexcluding each variable',labelpad=3)
    if n==14:
        # Scatter spans full first row; regional heatmap is lower left, sensitivity lower right.
        place(0,[.14,.61,.81,.28]);place(1,[.14,.16,.29,.30]);place(2,[.70,.16,.25,.30]);place(3,[.46,.16,.015,.30])
        a[0].set_xlabel('Circulation error / historical-event mean')
        a[0].set_ylabel('Surface heat-flux error /\nhistorical-event mean')
        a[2].set_xlabel('Circulation–heat-flux\nerror correlation, r')
        a[2].set_yticklabels(['Without 1997–98','Without 2015–16','Without 2023–24'])
    if n==15:
        place(0,[.105,.22,.345,.66]);place(1,[.625,.22,.345,.66])
        a[1].set_title('Division using historical SST relationships')
        for ax in a:ax.set_xlabel('Share of net skill difference (%)')
        for leg in list(fig.legends):
            handles=leg.legend_handles;labels=[t.get_text() for t in leg.get_texts()];leg.remove()
            fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,.075),fontsize=7,frameon=False)
    if n==16:
        a[1].set_title('Ocean area with valid AUC');a[3].set_title('Ocean area with valid intensity RMSE')
        for ax in [a[1],a[3]]:ax.set_xlabel('Ocean area included (%)')
    if n==17:
        names=['12-month window, individual cells','12-month window, pooled nearby cells','18-month window, individual cells','18-month window, pooled nearby cells']
        for leg in fig.legends:
            for t,name in zip(leg.get_texts(),names):t.set_text(name)
            leg.set_bbox_to_anchor((.52,.96))
        for ax in [a[1],a[3]]:ax.set_xlabel('Ocean area included (%)')
    if n==7:return
    # One panel-label coordinate system. Align labels by row and column, not string width.
    panel_axes=[]
    for ax in a:
        for t in list(ax.texts):
            if len(t.get_text())==1 and t.get_text().islower():panel_axes.append((ax,t.get_text()));t.remove()
    for t in list(fig.texts):
        if len(t.get_text())==1 and t.get_text().islower():
            if n==1:panel_axes.append((a[2],t.get_text()))
            t.remove()
    fig.canvas.draw()
    for ax,letter in panel_axes:
        pos=ax.get_position();x=max(.008,pos.x0-.032)
        if n==1 and letter=='c':x=a[0].get_position().x0-.032
        fig.text(x,pos.y1+.027,letter,fontsize=9,fontweight='bold',va='bottom')
    for t in fig.findobj(mpl.text.Text):
        if len(t.get_text())==1 and t.get_text().islower() and t.get_fontweight()=='bold':continue
        if t.get_fontsize()>7:t.set_fontsize(7)
    fig.canvas.draw()
    # Centre the full visible artwork, including labels, rather than only axes.
    bounds=fig.get_tightbbox(fig.canvas.get_renderer())
    shift=(fig.get_figwidth()/2-(bounds.x0+bounds.x1)/2)/fig.get_figwidth()
    for ax in a:
        p=ax.get_position(original=True);ax.set_position([p.x0+shift,p.y0,p.width,p.height])
    for t in fig.texts:
        x,y=t.get_position();t.set_position((x+shift,y))
    for leg in fig.legends:
        p=leg.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
        leg.set_bbox_to_anchor((p.x0+shift,p.y0,p.width,p.height),transform=fig.transFigure)
    fig.canvas.draw()
    # Store source figure geometry and text for auditable alignment checks.
    import json
    root=Path(__file__).resolve().parent
    (root/'qa'/f'S{n:02d}_layout.json').write_text(json.dumps({'panels':[{ 'label':s,'axes_bounds':list(ax.get_position().bounds)} for ax,s in panel_axes], 'changes':'labels/layout only; data and limits unchanged'},indent=2))
