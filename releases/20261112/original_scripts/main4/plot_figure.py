# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) map -> Figure1_global_expected_benchmark_v11 field + Figure4_v23 layout -> param inherit
# (a) bars -> GroupedBarChart/plot_GroupedBarChartv1.py -> param inherit
# (main b) Sankey -> SankeyDiagram/plot_SankeyDiagram.py -> param inherit
# Assets and companion PNGs inspected; existing production project copied first.
# Continuous area-weighted allocations are incompatible with the asset's simulated
# count table. Inherit thin nodes / low-alpha curves / direct labels only.
# All three verbatim baselines execute in the frozen figure_style.py first.
from figure_style import *
from matplotlib.patches import Rectangle,PathPatch
from matplotlib.path import Path as MPath
import figure_style as style
import fitz

# Match the verified Figure1-3_Nature_layout_20261107_v2 font package.
# User-directed cosmetics override generic palette/layout recommendations.
for font_file in sorted((ROOT/'fonts').glob('*.ttf')):
    font_manager.fontManager.addfont(str(font_file))
FONT='Liberation Sans'
style.FONT=FONT
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':[FONT],
                    'mathtext.fontset':'dejavusans','hatch.linewidth':.28,
                    'font.size':6.5,'axes.linewidth':.5,
                    'xtick.major.width':.5,'ytick.major.width':.5})
REG=['TP','NP','SP','TA','NA','SA','IO','Other seas']
NAMES=['Tropical Pacific','North Pacific','South Pacific','Tropical Atlantic','North Atlantic','South Atlantic','Indian Ocean']
COL=['#D94B3A','#8064A2','#2166AC','#4393C3','#92C5DE','#D9D9D9']
LOCAL='#2166AC'
PARENT_COL=[COL[0],COL[1],LOCAL,COL[5]]
SHORTFALL_CMAP=LinearSegmentedColormap.from_list('global_reference_shortfall',
    ['#2166AC','#8BB9D3','#F7F7F7','#E8978B','#B2182B'],N=256)
FLOW_ALPHA=.46
PARTS=['ENSO','Other','Memory','Atmosphere','Ocean','Unallocated']
PARENT=np.array([0,1,2,2,2,3])

def map_boundary(ax,proj,lo,hi):
    left,right=-190+1e-5,170-1e-5
    xx=np.r_[np.linspace(left,right,721),np.full(241,right),np.linspace(right,left,721),np.full(241,left)]
    yy=np.r_[np.full(721,lo),np.linspace(lo,hi,241),np.full(721,hi),np.linspace(hi,lo,241)]
    xy=proj.transform_points(PC,xx,yy)[:,:2]
    ax.set_boundary(MPath(xy),transform=ax.transData)
    ax.set_xlim(xy[:,0].min(),xy[:,0].max());ax.set_ylim(xy[:,1].min(),xy[:,1].max())

def panel_a(fig):
    # Physical millimetres, measured from the top of the final 180 × 178 mm page.
    # Inset backplates overlap only the map fringes; bars remain fully opaque.
    def box(x,y,w,h):return [x/180,1-(y+h)/178,w/180,h/178]
    assoc=pd.read_csv(DATA/'Figure4_observed_group_associations.csv',keep_default_na=False)
    d=xr.load_dataset(DATA/'Figure4a_local_shortfall.nc')
    label(fig,'a',.020,1-2/178)
    barcols=[LOCAL,COL[0],COL[1],COL[5]]
    groupnames=['Local processes','ENSO','Other climate modes','Unresolved']
    bars={};anchors={'NP':(210,38),'TP':(225,0),'SP':(205,-38),'TA':(330,0),'NA':(330,40),'SA':(330,-38),'IO':(80,-28)}
    arrangement=[('NP',22,7,38),('TP',72,7,38),('NA',122,7,38),
                 ('SP',22,65,31),('TA',60,65,31),('SA',98,65,31),('IO',136,65,31)]
    for i,(r,x,y,w) in enumerate(arrangement):
        rect=box(x,y,w,15)
        ax=fig.add_axes(rect,zorder=10);bars[r]=ax
        ax.patch.set_facecolor('white');ax.patch.set_alpha(.62)
        z=assoc[assoc.Region==r].set_index('Group').loc[groupnames];v=z.Share.to_numpy()*100
        # Fixed physical bar width across the unequal-width inset panels.
        bs=ax.bar(range(4),v,width=.50*31.5/w,color=barcols,zorder=3)
        bs[-1].set_hatch('..');bs[-1].set_edgecolor('#777777');bs[-1].set_linewidth(.35)
        ax.errorbar(range(4),v,yerr=np.vstack([np.maximum(v-z.Low.to_numpy()*100,0),np.maximum(z.High.to_numpy()*100-v,0)]),fmt='none',ecolor='#333333',elinewidth=.45,capsize=1.3,zorder=4)
        ax.set(xlim=(-.6,3.6),ylim=(0,100));ax.set_xticks([]);ax.set_yticks([0,50,100]);ax.tick_params(length=1.8,pad=1,labelsize=6.3)
        if i not in [0,3]:ax.tick_params(labelleft=False)
        else:ax.set_ylabel('SST anomaly\ncontribution (%)',fontsize=7,linespacing=1.05)
        ax.set_title(NAMES[REG.index(r)],fontsize=7,fontweight='bold',pad=4)
        # Numbers are in source tables; bars and intervals carry the comparison.
    proj=ccrs.Robinson(central_longitude=-10)
    rect=box(23,19,138,51.5)
    ax=fig.add_axes(rect,projection=proj,zorder=1)
    arr,lon=add_cyclic_point(d.shortfall.values,coord=d.lon.values)
    im=ax.pcolormesh(lon,d.lat,arr,transform=PC,cmap=SHORTFALL_CMAP,vmin=-1,vmax=1,rasterized=True,shading='auto',antialiased=False)
    map_boundary(ax,proj,-60,60)
    ax.set_facecolor('#DADADA')
    ax.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#EBECEB',edgecolor='none',zorder=3)
    ax.coastlines(resolution='110m',lw=.25,color='#777777');ax.spines['geo'].set_linewidth(.4)
    md=xr.load_dataset(DATA/'seven_ocean_regions.nc')
    mask=md['ocean_mask'] if 'ocean_mask' in md else next(iter(md.data_vars.values()))
    for r in REG[:7]:
        m=mask.sel(region=r);ax.contour(m.lon,m.lat,m.astype(float),levels=[.5],colors=['#777777'],linewidths=.30,linestyles='dashed',transform=PC,zorder=4)
    offsets={'NP':(0,-9),'TP':(20,-9),'SP':(14,12),'TA':(21,-9),'NA':(9,-9),'SA':(0,12),'IO':(-15,8)}
    for r,(lo,la) in anchors.items():
        ax.scatter(lo,la,transform=PC,s=11,color='#555555',edgecolor='white',lw=.45,zorder=7)
        dx,dy=offsets[r];ax.text(lo+dx,la+dy,r,transform=PC,ha='center',va='center',fontsize=7,fontweight='bold',zorder=8,path_effects=[pe.withStroke(linewidth=1.15,foreground='white')])
        pos=(.5,0) if r in ['NP','TP','NA'] else (.5,1)
        if r=='TP':pos=(.12,0)
        con=ConnectionPatch(xyA=pos,coordsA=bars[r].transAxes,xyB=proj.transform_point(lo,la,PC),coordsB=ax.transData,color='#777777',lw=.42,shrinkB=2.5,zorder=6)
        con.set_path_effects([pe.withStroke(linewidth=1.0,foreground='white')]);fig.add_artist(con)
    rect=box(22,85,78,1.5)
    ca=fig.add_axes(rect);cb=fig.colorbar(im,cax=ca,orientation='horizontal',extend='neither',ticks=[-1,-.5,0,.5,1])
    cb.ax.tick_params(labelsize=6.3,length=2,pad=2);cb.set_label('Global expected AUC − local AUC',fontsize=6.5,labelpad=2)
    cb.outline.set_linewidth(.4)
    # Category and sign are independent encodings; sign is relative to the
    # historical expectation, not warming/cooling or a process causal effect.
    handles=[Patch(facecolor=c,label=l,edgecolor='#777777' if j==3 else 'none',linewidth=.35,hatch='..' if j==3 else None) for j,(c,l) in enumerate(zip(barcols,groupnames))]
    from matplotlib.lines import Line2D
    lower=Patch(facecolor='#AAAAAA',label='Lower forecast skill')
    higher=Line2D([],[],color='#333333',lw=.8,ls=(0,(2.5,1.5)),label='Higher forecast skill')
    fig.legend(handles=[handles[0],handles[2],lower,handles[1],handles[3],higher],loc='center',bbox_to_anchor=(.770,1-89/178),ncol=2,fontsize=6.5,handlelength=1.5,handletextpad=.4,columnspacing=.85,labelspacing=.35)
    style.REPORT['layout_revision']={'canvas_mm':[180,178],
        'inset_backplate_alpha':.62,'insets':7,'map_requested_mm':[138,51.5],
        'map_resolution_smoothing_colour_range':'inherited exactly from global-reference Fig1c map; limits [-1,1]',
        'inset_overlap':'map fringes only; bars and error bars remain opaque'}

def ribbon(ax,x0,x1,y0,y1,width,color,negative=False):
    dx=(x1-x0)*.38
    v=[(x0,y0),(x0+dx,y0),(x1-dx,y1),(x1,y1),(x1,y1+width),(x1-dx,y1+width),(x0+dx,y0+width),(x0,y0+width),(x0,y0)]
    codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
    unresolved=color==COL[5]
    band=PathPatch(MPath(v,codes),facecolor=mpl.colors.to_rgba(color,.18 if negative else FLOW_ALPHA),
        edgecolor='#777777' if unresolved else color if negative else 'none',
        lw=.35 if negative else .20 if unresolved else 0,
        hatch='..' if unresolved else None,zorder=2 if negative else 1)
    band.set_gid(f'flow_{COL.index(color) if color in COL else "neutral"}_{"higher" if negative else "lower"}')
    ax.add_patch(band)
    if negative:
        # This fixed-width locator is deliberately separate from quantitative
        # ribbon thickness. No small negative allocation is inflated or dropped.
        t=np.linspace(0,1,121)
        xx=(1-t)**3*x0+3*(1-t)**2*t*(x0+dx)+3*(1-t)*t*t*(x1-dx)+t**3*x1
        yy=(1-t)**3*(y0+width/2)+3*(1-t)**2*t*(y0+width/2)+3*(1-t)*t*t*(y1+width/2)+t**3*(y1+width/2)
        line,=ax.plot(xx,yy,color='#303030',lw=.65,ls=(0,(2.5,1.5)),zorder=2.2)
        line.set_gid('higher_skill_locator')
        line.set_path_effects([pe.Stroke(linewidth=1.15,foreground='white'),pe.Normal()])

def positions(totals,scale,gap,center=.50):
    top=center+(sum(totals)*scale+gap*(len(totals)-1))/2;res=[]
    for t in totals:
        h=t*scale;res.append((top-h,h));top-=h+gap
    return res

def panel_b(fig):
    d=pd.read_csv(DATA/'Figure4b_paired_errors.csv',keep_default_na=False)
    d=d[d.Region!='TP']
    stat=json.loads((ROOT/'bridge_statistics.json').read_text())
    label(fig,'b',.020,.535)
    fig.text(.06,.535,'Circulation and surface heat-flux forecast errors',fontsize=8.5,fontweight='bold',va='top')
    fig.text(.965,.535,'CFSv2 independent check · DJF',ha='right',va='top',fontsize=7,color='#555555')
    ax=fig.add_axes([.095,.365,.405,.130])
    x='Z200_relative';y='Qnet_relative'
    styles=[(1997,'#BBBBBB','o','1997–98'),(2015,'#555555','s','2015–16'),(2023,'#D94B3A','^','2023–24')]
    for event,color,marker,name in styles:
        a=d[d.Event==event]
        ax.scatter(a[x],a[y],s=20,alpha=.7 if event==2023 else .5,color=color,marker=marker,linewidths=.2,edgecolors='white',label=name,zorder=3)
    lo=min(.55,float(d[x].min())-.05,float(d[y].min())-.05)
    hi=max(1.45,float(d[x].max())+.05,float(d[y].max())+.05)
    ax.set(xlim=(lo,hi),ylim=(lo,hi))
    ax.axvline(1,color='#AAAAAA',lw=.5,ls='--');ax.axhline(1,color='#AAAAAA',lw=.5,ls='--')
    coef=np.polyfit(d[x],d[y],1);xx=np.array([d[x].min(),d[x].max()])
    ax.plot(xx,np.polyval(coef,xx),color='#555555',lw=.7,zorder=2)
    ax.set_xlabel('Circulation error / canonical',fontsize=7,labelpad=3)
    ax.set_ylabel('Surface heat-flux error\n/ canonical',fontsize=7,labelpad=3)
    ax.set_xticks([.6,.8,1,1.2,1.4]);ax.set_yticks([.6,.8,1,1.2,1.4]);ax.tick_params(labelsize=6.5,length=2)
    ax.text(.04,.95,f'Descriptive r = {stat["descriptive_pearson_r"]:.2f}',transform=ax.transAxes,fontsize=7,va='top')
    ax.legend(loc='lower left',bbox_to_anchor=(-.02,1.015),ncol=3,fontsize=6.5,handletextpad=.25,columnspacing=.85,borderaxespad=0)
    hm=fig.add_axes([.665,.365,.215,.130])
    t=pd.read_csv(DATA/'Figure4b_regional_changes.csv',keep_default_na=False).set_index('Region').loc[REG[:7]]
    val=t[['Z200_error_change_pct','Qnet_error_change_pct']].values
    lim=np.ceil(np.max(abs(val))/10)*10
    cm=LinearSegmentedColormap.from_list('bridge_error_change',['#2166AC','#F7F7F7','#B2182B'])
    im=hm.imshow(val,cmap=cm,vmin=-lim,vmax=lim,aspect='auto')
    hm.set_yticks(range(7),REG[:7],fontsize=7);hm.set_xticks([0,1],['Circulation','Heat flux'],fontsize=7)
    hm.tick_params(length=0,pad=3)
    for spine in hm.spines.values():spine.set_visible(False)
    for (i,j),v in np.ndenumerate(val):
        hm.text(j,i,f'{v:+.1f}',ha='center',va='center',fontsize=6.5,color='white' if abs(v)>.65*lim else '#222222')
    hm.set_title('2023–24 error change (%)',fontsize=7.5,pad=5)
    ca=fig.add_axes([.902,.365,.013,.130]);cb=fig.colorbar(im,cax=ca,ticks=[-lim,0,lim])
    cb.ax.tick_params(labelsize=6.5,length=2);cb.outline.set_linewidth(.4)
    fig.text(.665,.340,'− smaller error     + larger error',fontsize=6.5,color='#555555')

def panel_c(fig,base=.130,height=.325):
    t=pd.read_csv(DATA/'association_weighted_local_shortfall.csv',keep_default_na=False)
    tab=t.pivot(index='Region',columns='Component',values='Allocated_positive_shortfall').loc[REG,PARTS].to_numpy()
    higher=t.pivot(index='Region',columns='Component',values='Allocated_above_benchmark').loc[REG,PARTS].to_numpy()
    mag=tab+higher;parent=np.zeros((8,4));parent_higher=np.zeros((8,4))
    for k,g in enumerate(PARENT):
        parent[:,g]+=mag[:,k]
        parent_higher[:,g]+=higher[:,k]
    s=json.loads((ROOT/'statistics.json').read_text())
    pct=100/s['positive_map_shortfall']
    label(fig,'b',.020,1-96/178)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],color='#555555',lw=3,alpha=.65,label='Lower forecast skill (+)'),
                        Line2D([],[],color='#333333',lw=.8,ls=(0,(2.5,1.5)),label='Higher forecast skill (−)')],
               loc='center',bbox_to_anchor=(.54,1-98/178),ncol=2,fontsize=6.5,frameon=False,
               handlelength=2.2,columnspacing=2.3)
    ax=fig.add_axes([.02,base,.96,height]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    scale=.63/mag.sum();rp=positions(mag.sum(1),scale,.036);gp=positions(parent.sum(0),scale,.070)
    # Children are separated for readability, without inflating their heights.
    cp=positions(mag[:,2:5].sum(0),scale,.120,center=.47)
    x0,x1,x2=.24,.52,.84;nw=.010
    ax.text(x0,.988,'Regions',ha='center',fontsize=7)
    ax.text(x1,.988,'Association groups',ha='center',fontsize=7)
    ax.text(x2,.988,'Local components',ha='center',fontsize=7)
    # Separate signed subsegments within each node; a common width scale retains
    # both signs without cancellation or inflation of small negative amounts.
    for negative,values in [(False,tab),(True,higher)]:
        rc=[p[0]+(0 if negative else higher[b].sum()*scale) for b,p in enumerate(rp)]
        gc=[p[0]+(0 if negative else parent_higher[:,g].sum()*scale) for g,p in enumerate(gp)]
        cc=[p[0]+(0 if negative else higher[:,j+2].sum()*scale) for j,p in enumerate(cp)]
        for b,r in enumerate(REG):
            for k in reversed(range(len(PARENT))):
                g=PARENT[k];v=values[b,k]
                if v==0:continue
                h=v*scale
                ribbon(ax,x0+nw,x1,rc[b],gc[g],h,COL[k],negative)
                if g==2:
                    ribbon(ax,x1+nw,x2,gc[g],cc[k-2],h,COL[k],negative);cc[k-2]+=h
                rc[b]+=h;gc[g]+=h
    def paired(lo,hi):
        # Do not round the small, nonzero Other-seas component to zero.
        neg=f'{hi*pct:.2f}' if 0<hi*pct<.05 else f'{hi*pct:.1f}'
        return f'+{lo*pct:.1f}% / −{neg}%'
    for b,r in enumerate(REG):
        y,h=rp[b];ax.add_patch(Rectangle((x0,y),nw,h,color='#666666',lw=0))
        ax.text(x0-.012,y+h/2,f'{r}  {paired(tab[b].sum(),higher[b].sum())}',ha='right',va='center',fontsize=6.5)
    parents=['ENSO','Other climate modes','Local processes','Unresolved']
    for j,g in enumerate(parents):
        y,h=gp[j];ax.add_patch(Rectangle((x1,y),nw,h,facecolor=PARENT_COL[j],
            edgecolor='#777777' if j==3 else 'none',hatch='..' if j==3 else None,lw=.35 if j==3 else 0,zorder=3))
        val=tab[:,PARENT==j].sum()
        ax.text(x1+.018,y+h/2,f'{g}\n{paired(val,higher[:,PARENT==j].sum())}',ha='left',va='center',fontsize=7,linespacing=1.1,bbox=dict(facecolor='white',edgecolor='none',alpha=.91,pad=.6))
    labels=['Local ocean memory','Local atmospheric drivers','Local oceanic drivers']
    for j,g in enumerate(labels):
        y,h=cp[j];ax.add_patch(Rectangle((x2,y),nw,h,color=COL[j+2],lw=0))
        ax.text(x2+.018,y+h/2,g+'\n'+paired(tab[:,j+2].sum(),higher[:,j+2].sum()),ha='left',va='center',fontsize=6.6,linespacing=1.15)
    style.REPORT['signed_regional_accounting']={'denominator':'sum of area-weighted positive mapped shortfall',
        'lower_total_pct':float(tab.sum()*pct),'higher_total_pct':float(-higher.sum()*pct),
        'width_scale':'same for both signs; negative centerlines are locators only',
        'regions':{r:{'lower_pct':float(tab[i].sum()*pct),'higher_pct':float(-higher[i].sum()*pct)} for i,r in enumerate(REG)}}

def panel_signed_hierarchy(fig,base=2/178,height=77/178):
    """Conserved four-stage display; sign totals are explicit intermediate nodes."""
    t=pd.read_csv(DATA/'association_weighted_local_shortfall.csv',keep_default_na=False)
    vals=[t.pivot(index='Region',columns='Component',values=col).loc[REG,PARTS].to_numpy()
          for col in ['Allocated_positive_shortfall','Allocated_above_benchmark']]
    lo,hi=vals;mag=lo+hi
    den=json.loads((ROOT/'statistics.json').read_text())['net_map_shortfall'];pct=100/den
    assert den>0 and np.isclose((lo.sum()-hi.sum())/den,1)
    parent=np.array([[v[:,PARENT==g].sum() for g in range(4)] for v in vals])
    child=np.array([v[:,2:5].sum(0) for v in vals])
    totals=np.array([v.sum() for v in vals])
    assert np.allclose(parent.sum(1),totals)
    assert np.allclose(child.sum(1),parent[:,2])
    label(fig,'b',.020,1-96/178)
    ax=fig.add_axes([.02,base,.96,height]);ax.set(xlim=(0,1.3),ylim=(0,1));ax.axis('off')
    scale=.63/mag.sum();nw=.012
    rp=positions(mag.sum(1),scale,.036)
    sp=positions(totals,scale,.15)
    gp=positions(parent.sum(0),scale,.070)
    cp=positions(child.sum(0),scale,.120,center=.47)
    xr,xs,xg,xc=.20,.46,.79,1.15
    for x,heading in [(xr,'Regions'),(xs,'Sign'),(xg,'Drivers'),(xc,'Local components')]:
        ax.text(x,.988,heading,ha='center',fontsize=7)
    # Region -> sign: two flows per region, without process subdivisions.
    # Driver identities are intentionally aggregated at the explicit sign node.
    for sign,v in enumerate(vals):
        dest=sp[sign][0]
        for b in reversed(range(len(REG))):
            h=v[b].sum()*scale
            y=rp[b][0]+(hi[b].sum()*scale if sign==0 else 0)
            if h:ribbon(ax,xr+nw,xs,y,dest,h,'#9C9C9C',negative=bool(sign))
            dest+=h
        assert np.isclose(dest,sp[sign][0]+sp[sign][1])
    # Sign -> drivers and local -> components; each sign closes independently.
    for sign in range(2):
        cursor=sp[sign][0]
        for g in reversed(range(4)):
            y=gp[g][0]+(parent[1,g]*scale if sign==0 else 0)
            h=parent[sign,g]*scale
            if h:ribbon(ax,xs+nw,xg,cursor,y,h,PARENT_COL[g],negative=bool(sign))
            cursor+=h
        cursor=gp[2][0]+(parent[1,2]*scale if sign==0 else 0)
        for k in reversed(range(3)):
            y=cp[k][0]+(child[1,k]*scale if sign==0 else 0)
            h=child[sign,k]*scale
            if h:ribbon(ax,xg+nw,xc,cursor,y,h,COL[k+2],negative=bool(sign))
            cursor+=h
    def paired(l,h):
        n=f'{h*pct:.2f}' if 0<h*pct<.05 else f'{h*pct:.1f}'
        return f'+{l*pct:.1f}% / −{n}%'
    box=dict(facecolor='white',edgecolor='none',alpha=.92,pad=.6)
    for b,r in enumerate(REG):
        y,h=rp[b];ax.add_patch(Rectangle((xr,y),nw,h,color='#666666',lw=0))
        ax.text(xr-.014,y+h/2,r+'\n'+paired(lo[b].sum(),hi[b].sum()),ha='right',va='center',fontsize=6.5,linespacing=1.08)
    for j,name in enumerate(['Positive','Negative']):
        y,h=sp[j];ax.add_patch(Rectangle((xs,y),nw,h,facecolor='#777777' if j==0 else '#DADADA',edgecolor='#555555',lw=.4))
        text=f'{name}\n{"+" if j==0 else "−"}{totals[j]*pct:.1f}%'
        ax.text(xs+.018,y+h/2,text,ha='left',va='center',fontsize=6.7,linespacing=1.15,bbox=box)
    for j,name in enumerate(['ENSO','Other climate modes','Local processes','Unresolved']):
        y,h=gp[j];ax.add_patch(Rectangle((xg,y),nw,h,facecolor=PARENT_COL[j],edgecolor='#777777' if j==3 else 'none',hatch='..' if j==3 else None,lw=.35 if j==3 else 0))
        ax.text(xg+.018,y+h/2,name+'\n'+paired(parent[0,j],parent[1,j]),ha='left',va='center',fontsize=6.5,linespacing=1.1,bbox=box)
    for j,name in enumerate(['Local ocean memory','Local atmospheric drivers','Local oceanic drivers']):
        y,h=cp[j];ax.add_patch(Rectangle((xc,y),nw,h,color=COL[j+2],lw=0))
        ax.text(xc+.018,y+h/2,name+'\n'+paired(child[0,j],child[1,j]),ha='left',va='center',fontsize=6.5,linespacing=1.1)
    style.REPORT['signed_regional_accounting']={
        'denominator':'net area-weighted mapped shortfall (positive minus higher-skill magnitude)',
        'denominator_auc':float(den),'net_total_pct':float((totals[0]-totals[1])*pct),
        'lower_total_pct':float(totals[0]*pct),
        'higher_total_pct':float(-totals[1]*pct),'hierarchy':['Regions','Positive / Negative','Drivers','Local components'],
        'all_sign_nodes_conserve_magnitude':True,
        'regions':{r:{'lower_pct':float(lo[i].sum()*pct),'higher_pct':float(-hi[i].sum()*pct)} for i,r in enumerate(REG)}}

def above_benchmark(fig):
    tab=pd.read_csv(DATA/'regional_map_shortfall.csv',keep_default_na=False).set_index('Region').loc[REG]
    val=tab.Above_share_of_shortfall_pct.to_numpy()
    ax=fig.add_axes([18/180,6/178,149/180,8/178])
    ax.bar(np.arange(len(REG)),val,color='#2166AC',alpha=.85,width=.45)
    ax.set_xticks(np.arange(len(REG)),REG,fontsize=6)
    upper=max(val.max()*1.30,1)
    ax.set_ylim(0,upper);ax.set_xlim(-.6,len(REG)-.4)
    ax.set_yticks([0],['0']);ax.tick_params(length=1.5,labelsize=6,pad=1)
    ax.spines[['top','right','left']].set_visible(False)
    for i,v in enumerate(val):
        shown='<0.1' if 0<v<.05 else f'{v:.1f}'
        ax.text(i,v+.04*upper,shown,ha='center',va='bottom',fontsize=6)
    ax.set_title('Above global expectation (% of total shortfall)',fontsize=6.5,loc='left',pad=7)
    style.REPORT['above_benchmark_total_pct']=float(val.sum())

def compact_layout(fig):
    # Match the full visible widths, including axis and node labels.
    from matplotlib.transforms import Bbox
    sankey=fig.axes[-1]
    for _ in range(12):
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        target=Bbox.from_extents(8/180*fig.bbox.width,0,174/180*fig.bbox.width,1)
        # An invisible axes rectangle must not determine visible alignment.
        current=Bbox.union([artist.get_window_extent(renderer)
                            for artist in [*sankey.texts,*sankey.patches]
                            if artist.get_visible()])
        x,y,w,h=sankey.get_position().bounds
        ratio=target.width/current.width
        sankey.set_position([(target.x0+(x*fig.bbox.width-current.x0)*ratio)/fig.bbox.width,y,w*ratio,h])
    fig.canvas.draw()
    map_ax=fig.axes[7]
    map_box=map_ax.get_position()
    style.REPORT['layout_revision']['map_actual_mm']=[map_box.width*180,map_box.height*178]
    style.REPORT['layout_revision']['inset_map_overlap_mm2']=[
        max(0,min(a.get_position().x1,map_box.x1)-max(a.get_position().x0,map_box.x0))*180*
        max(0,min(a.get_position().y1,map_box.y1)-max(a.get_position().y0,map_box.y0))*178
        for a in fig.axes[:7]]

def main():
    fig=plt.figure(figsize=(180/25.4,178/25.4));panel_a(fig);panel_signed_hierarchy(fig);compact_layout(fig)
    save(fig,'Figure4')
    # Only the requested Figure 4 is exported; do not inherit stale SI QA claims.
    (ROOT/'render_validation.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    with fitz.open(OUT/'Figure4.pdf') as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(OUT/'Figure4_preview.png')
    print(json.dumps(REPORT,indent=2))

if __name__=='__main__':main()
