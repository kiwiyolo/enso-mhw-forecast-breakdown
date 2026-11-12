# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) map + seven bars -> preserved Figure4_region_layout_v8 -> param inherit
# (b) Sankey -> SankeyDiagram/plot_SankeyDiagram.py -> param inherit
# Continuous signed allocations are incompatible with the asset's simulated
# count table. Inherit thin nodes / low-alpha curves / direct labels only.
# All three verbatim baselines execute in the frozen figure_style.py first.
from figure_style import *
from matplotlib.patches import Rectangle,PathPatch
from matplotlib.path import Path as MPath
import figure_style as style
import fitz

FONT='Times New Roman' if 'Times New Roman' in available else 'STIXGeneral'
style.FONT=FONT
mpl.rcParams.update({'font.family':FONT,'font.serif':[FONT],'mathtext.fontset':'stix','hatch.linewidth':.22})
REG=['TP','NP','SP','TA','NA','SA','IO','Other seas']
NAMES=['Tropical Pacific','North Pacific','South Pacific','Tropical Atlantic','North Atlantic','South Atlantic','Indian Ocean']
COL=['#D94B3A','#3976B9','#376C4C','#82A36F','#B7C8A1','#AAAAAA']
PARENT_COL=[COL[0],COL[1],'#648A5C',COL[5]]
PARTS=['ENSO','Other','Memory','Atmosphere','Ocean','Unallocated']
PARENT=np.array([0,1,2,2,2,3])

def map_boundary(ax,proj,lo,hi):
    left,right=-190+1e-5,170-1e-5
    xx=np.r_[np.linspace(left,right,721),np.full(241,right),np.linspace(right,left,721),np.full(241,left)]
    yy=np.r_[np.full(721,lo),np.linspace(lo,hi,241),np.full(721,hi),np.linspace(hi,lo,241)]
    xy=proj.transform_points(PC,xx,yy)[:,:2]
    ax.set_boundary(MPath(xy),transform=ax.transData)
    ax.set_xlim(xy[:,0].min(),xy[:,0].max());ax.set_ylim(xy[:,1].min(),xy[:,1].max())

def panel_a(fig,base=.425,height=.565):
    def box(x,y,w,h):return [x,base+height*y,w,height*h]
    assoc=pd.read_csv(DATA/'Figure4_observed_group_associations.csv',keep_default_na=False)
    d=xr.load_dataset(DATA/'Figure4a_activity_valid_extent.nc');cov=json.loads((ROOT/'coverage.json').read_text())
    label(fig,'a',.020,base+height*.995)
    fig.text(.06,base+height*.985,'Regional MHW activity and historical SST associations',fontsize=9,fontweight='bold',va='top')
    barcols=['#648A5C',COL[0],COL[1],COL[5]]
    groupnames=['Local processes','ENSO','Other climate modes','Unresolved']
    bars={};anchors={'NP':(210,38),'TP':(225,0),'SP':(205,-38),'TA':(330,0),'NA':(330,40),'SA':(330,-38),'IO':(80,-28)}
    arrangement=[('NP',.080,.790,.235),('TP',.390,.790,.235),('NA',.700,.790,.235),('SP',.080,.135,.175),('TA',.310,.135,.175),('SA',.540,.135,.175),('IO',.770,.135,.175)]
    for i,(r,x,y,w) in enumerate(arrangement):
        ax=fig.add_axes(box(x,y,w,.120));bars[r]=ax
        z=assoc[assoc.Region==r].set_index('Group').loc[groupnames];v=z.Share.to_numpy()*100
        bs=ax.bar(range(4),v,width=.52,color=barcols,zorder=3)
        bs[-1].set_hatch('///');bs[-1].set_edgecolor('#777777');bs[-1].set_linewidth(0)
        ax.errorbar(range(4),v,yerr=np.vstack([np.maximum(v-z.Low.to_numpy()*100,0),np.maximum(z.High.to_numpy()*100-v,0)]),fmt='none',ecolor='#333333',elinewidth=.45,capsize=1.3,zorder=4)
        ax.set(xlim=(-.6,3.6),ylim=(0,100));ax.set_xticks(range(4),['Local','ENSO','Other','Unres.']);ax.set_yticks([0,50,100]);ax.tick_params(length=1.8,pad=1,labelsize=6.3)
        if i not in [0,3]:ax.tick_params(labelleft=False)
        else:ax.set_ylabel('SST association (%)',fontsize=7)
        ax.set_title(NAMES[REG.index(r)],fontsize=8,fontweight='bold',pad=4)
        # Numbers are in source tables; bars and intervals carry the comparison.
    proj=ccrs.Robinson(central_longitude=-10)
    ax=fig.add_axes(box(.045,.315,.920,.405),projection=proj)
    arr,lon=add_cyclic_point(d.activity_native_cell_equivalent.values/1e6,coord=d.lon.values)
    im=ax.pcolormesh(lon,d.lat,arr,transform=PC,cmap=HEAT,vmin=0,vmax=40,rasterized=True,shading='auto')
    map_boundary(ax,proj,cov['lat_min'],cov['lat_max'])
    ax.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#DEDEDE',edgecolor='none',zorder=3)
    ax.coastlines(resolution='110m',lw=.35,color='#666666');ax.spines['geo'].set_linewidth(.5)
    md=xr.load_dataset(DATA/'seven_ocean_regions.nc')
    mask=md['ocean_mask'] if 'ocean_mask' in md else next(iter(md.data_vars.values()))
    for r in REG[:7]:
        m=mask.sel(region=r);ax.contour(m.lon,m.lat,m.astype(float),levels=[.5],colors=['#555555'],linewidths=.45,linestyles='dashed',transform=PC,zorder=4)
    offsets={'NP':(0,-9),'TP':(20,-9),'SP':(14,-5),'TA':(21,-9),'NA':(0,12),'SA':(12,-8),'IO':(-15,-7)}
    for r,(lo,la) in anchors.items():
        ax.scatter(lo,la,transform=PC,s=15,color=BLACK,edgecolor='white',lw=.6,zorder=7)
        dx,dy=offsets[r];ax.text(lo+dx,la+dy,r,transform=PC,ha='center',va='center',fontsize=8,fontweight='bold',zorder=8,path_effects=[pe.withStroke(linewidth=2,foreground='white')])
        pos=(.5,-.44) if r in ['NP','TP','NA'] else (.5,1.30)
        con=ConnectionPatch(xyA=pos,coordsA=bars[r].transAxes,xyB=proj.transform_point(lo,la,PC),coordsB=ax.transData,color='#444444',lw=.6,shrinkB=3,zorder=6)
        con.set_path_effects([pe.withStroke(linewidth=1.8,foreground='white')]);fig.add_artist(con)
    ca=fig.add_axes(box(.080,.058,.46,.011));cb=fig.colorbar(im,cax=ca,orientation='horizontal',extend='max',ticks=[0,10,20,30,40])
    cb.ax.tick_params(labelsize=6.5,length=2,pad=2);cb.set_label(r'Monthly MHW activity ($10^{6}$ °C days km$^{2}$)',fontsize=7,labelpad=2)
    fig.legend(handles=[Patch(facecolor=c,label=l,edgecolor='#777777' if j==3 else 'none',hatch='///' if j==3 else None) for j,(c,l) in enumerate(zip(barcols,['Local processes','ENSO','Other modes','Unresolved']))],loc='center',bbox_to_anchor=(.76,base+height*.057),ncol=2,fontsize=7,handlelength=1.2,handletextpad=.4,columnspacing=1.1)

def ribbon(ax,x0,x1,y0,y1,width,color,negative=False):
    dx=(x1-x0)*.43
    v=[(x0,y0),(x0+dx,y0),(x1-dx,y1),(x1,y1),(x1,y1+width),(x1-dx,y1+width),(x0+dx,y0+width),(x0,y0+width),(x0,y0)]
    codes=[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(v,codes),facecolor=mpl.colors.to_rgba(color,.12 if negative else .25),edgecolor=color if negative else 'none',lw=.2 if negative else 0,hatch='///' if negative else None))

def positions(totals,scale,gap,center=.50):
    top=center+(sum(totals)*scale+gap*(len(totals)-1))/2;res=[]
    for t in totals:
        h=t*scale;res.append((top-h,h));top-=h+gap
    return res

def panel_b(fig,base=.055,height=.325):
    t=pd.read_csv(DATA/'association_weighted_AUC_allocation.csv',keep_default_na=False)
    tab=t[t.Profile=='Archived'].pivot(index='Region',columns='Component',values='Allocated_deficit').loc[REG,PARTS].to_numpy()
    mag=abs(tab);parent=np.zeros((8,4))
    for k,g in enumerate(PARENT):parent[:,g]+=mag[:,k]
    s=json.loads((ROOT/'statistics.json').read_text())
    pct=100/s['deficit']
    label(fig,'b',.020,.418)
    fig.text(.06,.418,'Association-weighted accounting of the archived AUC deficit',fontsize=8.5,fontweight='bold',va='top')
    fig.text(.06,.397,f'Net 100% ({s["deficit"]:.3f} AUC)  =  gross loss {pct*s["gross_loss"]:.1f}%  −  offset {pct*s["offset"]:.1f}%',fontsize=7,color='#444444')
    ax=fig.add_axes([.02,base,.96,height]);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    scale=.63/mag.sum();rp=positions(mag.sum(1),scale,.036);gp=positions(parent.sum(0),scale,.070)
    # Children are separated for readability, without inflating their heights.
    cp=positions(mag[:,2:5].sum(0),scale,.120,center=.47)
    x0,x1,x2=.135,.445,.775;nw=.010
    ax.text(x0,.988,'Region',ha='center',fontsize=7.5)
    ax.text(x1,.988,'Association group',ha='center',fontsize=7.5)
    ax.text(x2,.988,'Local components',ha='center',fontsize=7.5)
    rc=[p[0] for p in rp];gc=[p[0] for p in gp];cc=[p[0] for p in cp]
    for b,r in enumerate(REG):
        for k,g in enumerate(PARENT):
            v=tab[b,k]
            if v==0:continue
            h=abs(v)*scale;negative=v<0
            ribbon(ax,x0+nw,x1,rc[b],gc[g],h,COL[k],negative)
            if g==2:
                ribbon(ax,x1+nw,x2,gc[g],cc[k-2],h,COL[k],negative);cc[k-2]+=h
            rc[b]+=h;gc[g]+=h
    for b,r in enumerate(REG):
        y,h=rp[b];ax.add_patch(Rectangle((x0,y),nw,h,color='#666666',lw=0))
        ax.text(x0-.012,y+h/2,f'{r}\n{pct*tab[b].sum():+.1f}%',ha='right',va='center',fontsize=6.5,linespacing=1.08)
    parents=['ENSO','Other climate modes','Local processes','Unallocated']
    for j,g in enumerate(parents):
        y,h=gp[j];ax.add_patch(Rectangle((x1,y),nw,h,color=PARENT_COL[j],lw=0))
        val=tab[:,PARENT==j].sum()
        ax.text(x1+.018,y+h/2,f'{g}\n{pct*val:.1f}%',ha='left',va='center',fontsize=7,linespacing=1.1,bbox=dict(facecolor='white',edgecolor='none',alpha=.91,pad=.6))
    labels=['Local ocean memory\n(lagged thermal state)','Local atmospheric drivers\n(surface heating)','Local oceanic drivers\n(horizontal advection)']
    for j,g in enumerate(labels):
        y,h=cp[j];ax.add_patch(Rectangle((x2,y),nw,h,color=COL[j+2],lw=0))
        ax.text(x2+.018,y+h/2,g+f'\n{pct*tab[:,j+2].sum():.1f}%',ha='left',va='center',fontsize=6.6,linespacing=1.15)
    fig.legend(handles=[Patch(facecolor='#CCCCCC',label='Loss'),Patch(facecolor='white',edgecolor='#666666',hatch='///',label='Offset')],loc='center left',bbox_to_anchor=(.04,.032),ncol=2,fontsize=7,handlelength=1.2,columnspacing=1.2)
    fig.text(.97,.032,'Association weights do not identify causes',fontsize=6.8,ha='right',va='center',color='#444444')

def main():
    fig=plt.figure(figsize=(183/25.4,240/25.4));panel_a(fig);panel_b(fig)
    fig.text(.5,.010,'Review copy · archived-input validation unresolved',ha='center',fontsize=7,color='#8B3C2D')
    save(fig,'Figure4')
    (ROOT/'render_validation.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    with fitz.open(OUT/'Figure4.pdf') as doc:
        doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(OUT/'Figure4_preview.png')
    print(json.dumps(REPORT,indent=2))

if __name__=='__main__':main()
