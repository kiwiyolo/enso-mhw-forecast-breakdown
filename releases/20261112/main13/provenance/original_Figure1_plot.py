# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# Fig1a scatter -> existing project production -> param inherit (reference colours)
# Fig1b / Fig2b lines -> LineTrend/plot_trend.py -> param inherit (no cumulative-count transform)
# Fig2a / Fig3 maps -> existing geographic production + supplied reference -> param inherit
# Fig2c bars -> GroupedBarChart/plot_GroupedBarChartv1.py -> param inherit (paired-event ranges, not replicates)
# Fig3 components -> BarComposition/plot_brute_force.py -> param inherit (hatch convention only)
# Companion images inspected; source structures differ, so original approved geographic
# layouts are retained. Only actual archived/recomputed data enter the plots.
from figure_style import *
import fitz

# Explicit author-reference overrides; the shared skill baseline remains in figure_style.py.
REF_RED='#D94B3A';REF_BLUE='#3976B9';REF_BLACK='#222222'
REF_DIV=mpl.colormaps['RdBu_r']

def figure1():
    event=pd.read_csv(DATA/'Figure1_source_events.csv')
    ci=pd.read_csv(DATA/'Figure1_event_uncertainty.csv')
    event=event.merge(ci[['Event','CI_low','CI_high']],on='Event',validate='one_to_one')
    hist=event[event.Event!='2023/24'];target=event[event.Event=='2023/24'].iloc[0]
    s=json.loads((DATA/'Figure1_statistics.json').read_text())
    band=pd.read_csv(DATA/'Figure1a_regression_prediction_band.csv')
    fig=plt.figure(figsize=(183/25.4,116/25.4))
    title(fig,'Fig. 1 | Low MHW forecast skill during the 2023–2024 El Niño')
    a=fig.add_axes([.085,.16,.405,.69]);label(fig,'a',.024,.895)
    a.fill_between(band.Peak_Nino34,band.PI_low,band.PI_high,color='#EAEAEA',zorder=0)
    a.plot(band.Peak_Nino34,band.Fit,color=REF_BLACK,lw=.9)
    for row in hist.itertuples():
        strong=row.Event in ['1997/98','2015/16']
        a.errorbar(row.Peak_Nino34,row.Mean_AUC,yerr=[[row.Mean_AUC-row.CI_low],[row.CI_high-row.Mean_AUC]],
                   fmt='o',mfc=REF_BLACK if strong else '#D5D5D5',mec=REF_BLACK if strong else '#555555',
                   ecolor=REF_BLACK if strong else '#BFBFBF',markersize=4.2,elinewidth=.85,capsize=2,zorder=3)
    xt=target.Peak_Nino34
    a.errorbar(xt,target.Mean_AUC,yerr=[[target.Mean_AUC-target.CI_low],[target.CI_high-target.Mean_AUC]],
               color=REF_RED,fmt='none',elinewidth=.9,capsize=2,zorder=4)
    a.scatter(xt,s['expected'],s=22,color='#555555',zorder=5)
    a.scatter(xt,s['observed'],s=95,marker='*',color=REF_RED,edgecolor='white',linewidth=.5,zorder=7)
    a.annotate('',xy=(xt,s['observed']+.013),xytext=(xt,s['expected']-.009),
               arrowprops={'arrowstyle':'->','lw':.9,'color':REF_RED})
    a.text(xt+.1,s['expected']-.008,f"Expected {s['expected']:.3f}",fontsize=7.2,va='top')
    a.text(xt+.11,(s['expected']+s['observed'])/2+.004,f"Δ = {s['residual']:.3f}",fontsize=7.2,color=REF_RED)
    a.text(xt+.1,s['observed']-.023,f"Observed {s['observed']:.3f}",color=REF_RED,fontsize=7.2)
    # Exact regression statistics are retained in the caption/source JSON.
    a.set(xlim=(.25,2.95),ylim=(.40,.95),xlabel='Peak Niño3.4 anomaly (°C)',ylabel='Global MHW forecast skill (AUC)')
    a.set_xticks([.5,1,1.5,2,2.5]);a.set_yticks(np.arange(.4,1,.1))
    b=fig.add_axes([.61,.30,.365,.55]);label(fig,'b',.545,.895)
    m=pd.read_csv(DATA/'Figure1_monthly_t1_t9.csv',parse_dates=['time']).set_index('time')
    m=m.reindex(pd.date_range(m.index.min(),m.index.max(),freq='MS'))
    b.plot(m.index,m.AUC,color=REF_BLACK,lw=.8)
    for start,end,text,col in [('1997-07','1998-07','1997–98','#DDDDDD'),('2015-07','2016-07','2015–16','#DDDDDD'),('2023-07','2024-07','2023–24','#F4C7C1')]:
        b.axvspan(pd.Timestamp(start),pd.Timestamp(end),facecolor=col,alpha=.5,lw=0)
        xpos=pd.Timestamp(start)+(pd.Timestamp(end)-pd.Timestamp(start))/2
        b.text(xpos,1.02,text,transform=b.get_xaxis_transform(),ha='center',va='bottom',fontsize=6.6,color=REF_RED if text=='2023–24' else REF_BLACK)
    b.set(ylim=(min(.4,np.floor(m.AUC.min()*20)/20),.95),ylabel='Forecast skill (AUC)')
    b.tick_params(axis='x',labelbottom=False);b.set_yticks(np.arange(.4,1,.1))
    strip=fig.add_axes([.61,.17,.365,.065],sharex=b)
    dates=mdates.date2num(m.index.to_pydatetime());end=mdates.date2num((m.index[-1]+pd.DateOffset(months=1)).to_pydatetime())
    strip.pcolormesh(np.r_[dates,end],[0,1],np.ma.masked_invalid(m.Nino34.to_numpy()[None,:]),cmap=REF_DIV,vmin=-2.5,vmax=2.5,rasterized=True)
    strip.set_yticks([]);strip.spines['left'].set_visible(False)
    strip.text(-.03,.5,'Niño3.4',transform=strip.transAxes,ha='right',va='center',fontsize=7)
    strip.set_xticks([pd.Timestamp(f'{year}-01-01') for year in [1995,2005,2015,2024]])
    strip.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));strip.set_xlabel('Year');strip.set_xlim(dates[0],end)
    REPORT['Figure1_scope']={'independent_temperature_colourbar_removed':True,'Nino34_time_strip_retained':True,'source_numbers_unchanged':True}
    save(fig,'Figure1')

def figure2():
    fig=plt.figure(figsize=(183/25.4,137/25.4))
    title(fig,'Fig. 2 | Degraded forecasts of tropical ocean–atmosphere processes')
    label(fig,'a',.024,.931);label(fig,'b',.712,.931);label(fig,'c',.024,.31)
    ds=xr.load_dataset(DATA/'Figure2a_relative_forecast_error.nc')
    curves=pd.read_csv(DATA/'Figure2_selected_process_skill.csv')
    mapaxes=[];lineaxes=[]
    for row,(var,name) in enumerate([('sst','SST'),('stress','Zonal wind stress'),('precipitation','Precipitation')]):
        field=ds.relative_error.sel(variable=var).mean('lead');y=.744-row*.169
        for col in range(2):
            ax=fig.add_axes([.083+col*.308,y,.294,.164],projection=ccrs.PlateCarree(central_longitude=200));mapaxes.append(ax)
            val=field.sel(event=['1997/98','2015/16']).mean('event') if col==0 else field.sel(event='2023/24')
            im=ax.pcolormesh(ds.lon,ds.lat,val,transform=PC,cmap=REF_DIV,vmin=-200,vmax=200,shading='auto',rasterized=True)
            ax.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#F1F1F1',edgecolor='none',zorder=3)
            ax.coastlines(resolution='110m',lw=.35,color='#666666');ax.set_extent([140,260,-25,25],crs=PC)
            ax.set_yticks([-20,0,20],crs=PC);ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f',degree_symbol='°'))
            ax.set_xticks([140,180,220,260],crs=PC);ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f',degree_symbol='°'))
            ax.tick_params(labelsize=6.2,length=2,pad=2)
            if col>0:ax.set_yticklabels([])
            if row<2:ax.tick_params(labelbottom=False)
            else:
                ax.get_xticklabels()[0].set_ha('left');ax.get_xticklabels()[-1].set_ha('right')
            if col==0:ax.text(.018,.94,name,transform=ax.transAxes,va='top',fontsize=6.8,fontweight='bold',color=REF_BLUE,
                bbox={'facecolor':'white','edgecolor':'none','pad':.8,'alpha':.88},zorder=6)
            if row==0:ax.set_title('Comparable events' if col==0 else '2023–2024',fontsize=7.8,pad=7)
        bounds=mapaxes[-1].get_position()
        ax=fig.add_axes([.752,bounds.y0,.223,bounds.height]);lineaxes.append(ax)
        z=curves[curves.Variable==var];hist=z[z.Event!='2023/24'].pivot(index='Lead',columns='Event',values='Pattern_r')
        tar=z[z.Event=='2023/24'].set_index('Lead').Pattern_r.sort_index()
        ax.fill_between(hist.index,hist.min(axis=1),hist.max(axis=1),color='#E6E6E6',lw=0,label='Comparable range')
        ax.plot(hist.index,hist.mean(axis=1),color=REF_BLACK,marker='o',ms=2.5,lw=.85,label='Comparable mean')
        ax.plot(tar.index,tar,color=REF_RED,marker='o',ms=2.5,lw=.95,label='2023–2024')
        ax.set(xlim=(.7,9.3),ylim=(-.05,1.02));ax.set_yticks([0,.5,1]);ax.tick_params(labelsize=6.3,length=2,pad=2)
        ax.set_xticks([1,3,5,7,9]);ax.axhline(0,color='#BBBBBB',lw=.45)
        if row<2:ax.tick_params(labelbottom=False)
        else:ax.set_xlabel('Lead month',fontsize=7)
        if row==0:
            ax.set_title('Pattern fidelity',fontsize=7.8,pad=7)
            ax.legend(loc='lower left',bbox_to_anchor=(.015,.02),fontsize=5.8,labelspacing=.15,handlelength=1.4,handletextpad=.35,borderaxespad=0)
    fig.text(.707,.68,'Pattern correlation',rotation=90,ha='center',va='center',fontsize=7)
    cax=fig.add_axes([.083,.355,.602,.012]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',extend='both',ticks=[-200,-100,0,100,200])
    cb.ax.tick_params(labelsize=6.5,length=2,pad=2);cb.outline.set_linewidth(.4)
    cb.set_label('Relative forecast error (% of observed pattern RMS)',fontsize=7,labelpad=3)
    d=pd.read_csv(DATA/'Figure2c_canonical_error_audit.csv');ax=fig.add_axes([.083,.105,.892,.19]);x=np.arange(6);v=d.Relative_error_change_percent.values
    spread=max(1,float(d.High.max()-min(0,d.Low.min())));pad=.045*spread
    ax.axhline(0,color='#888888',lw=.65,zorder=0)
    ax.bar(x,v,width=.46,color=[REF_RED if value>=0 else REF_BLUE for value in v],alpha=.92,zorder=2)
    ax.errorbar(x,v,yerr=[v-d.Low.values,d.High.values-v],fmt='o',color=REF_BLACK,ecolor=REF_BLACK,ms=3.7,capsize=2.2,elinewidth=.8,zorder=4)
    for j,r in d.iterrows():
        if not np.isfinite(r.Relative_error_change_percent):
            ax.text(j,pad,'N/A',ha='center',va='bottom',fontsize=7.5,color='#666666')
            continue
        ax.scatter([j-.06,j+.06],[r.Reference_1997,r.Reference_2015],s=10,facecolors='white',edgecolors='#555555',linewidths=.6,zorder=5)
        val=r.Relative_error_change_percent
        ax.text(j,r.High+pad if val>=0 else r.Low-pad,f'{val:+.0f}%',ha='center',va='bottom' if val>=0 else 'top',fontsize=7.5)
    lower=0 if d.Low.min()>=0 else d.Low.min()-6*pad
    ax.set(xlim=(-.65,5.65),ylim=(lower,max(0,d.High.max())+4*pad),ylabel='Error change (%)')
    ax.yaxis.label.set_fontsize(8);ax.yaxis.set_label_coords(-.061,.5)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5));ax.tick_params(labelsize=7)
    ax.set_xticks(x,['MHW skill\n(Niño3.4)','Niño3.4\namplitude','Niño3.4\ntiming','SST\npattern','Wind\npattern','Precipitation\npattern'])
    # Event windows and lead availability are documented in Figure2.md.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();yb=ax.yaxis.label.get_window_extent(renderer)
    ytitle_tick_overlap=any(yb.overlaps(t.get_window_extent(renderer)) for t in ax.get_yticklabels())
    assert not ytitle_tick_overlap, 'Figure2c y-title overlaps tick labels'
    REPORT['Figure2_layout']={'map_axes':[list(a.get_position().bounds) for a in mapaxes],
          'line_axes':[list(a.get_position().bounds) for a in lineaxes],
          'matching_row_y':all(abs(mapaxes[2*i].get_position().y0-lineaxes[i].get_position().y0)<1e-4 for i in range(3)),
          'right_edge':lineaxes[-1].get_position().x1,'bottom_right_edge':ax.get_position().x1,
          'canonical_two_events_only':True,'artificial_stippling':False,'ytitle_tick_overlap':bool(ytitle_tick_overlap)}
    save(fig,'Figure2')

BASINS={'North Pacific':(0,60,120,280),'South Pacific':(-60,0,120,290),'Indian Ocean':(-60,30,20,120),
        'North Atlantic':(20,60,280,360),'Tropical Atlantic':(-20,20,280,360),'South Atlantic':(-60,0,290,360)}

def figure3():
    fig=plt.figure(figsize=(183/25.4,158/25.4));title(fig,'Fig. 3 | Local memory leads the resolved regional associations')
    d=pd.read_csv(DATA/'Figure3_continuous_components.csv').set_index('Basin')
    top=['North Pacific','Tropical Atlantic','North Atlantic'];bot=['South Pacific','South Atlantic','Indian Ocean']
    keys=['local','enso','other','unresolved'];colors=[GREEN,ORANGE,BLUE,'#AAAAAA'];axes={}
    for row,names in enumerate([top,bot]):
        for col,basin in enumerate(names):
            x=.085+col*.313;y=.775 if row==0 else .125;ax=fig.add_axes([x,y,.245,.13]);axes[basin]=ax
            val=d.loc[basin,keys].values.astype(float)*100;bars=ax.bar(range(4),val,width=.53,color=colors,alpha=.94)
            bars[-1].set_hatch('///');bars[-1].set_edgecolor('#777777');bars[-1].set_linewidth(0)
            for j,v in enumerate(val):ax.text(j,v+2,f'{v:.1f}',ha='center',va='bottom',fontsize=6.5)
            ax.set(ylim=(0,65),xlim=(-.6,3.6));ax.set_xticks(range(4),['Local','ENSO','Other','Unres.']);ax.set_yticks([0,20,40,60]);ax.tick_params(labelsize=6.5)
            ax.set_title(basin,fontsize=8.2,fontweight='bold',pad=9)
            if col==0:ax.set_ylabel('Association\nshare (%)',fontsize=7)
            else:ax.tick_params(labelleft=False)
            label(fig,chr(97+row*3+col),x-.045,y+.155)
    proj=ccrs.Robinson(central_longitude=-10);mapax=fig.add_axes([.06,.29,.92,.435],projection=proj)
    ds=xr.load_dataset(DATA/'Figure3_activity_display_grid.nc')
    arr,lon=add_cyclic_point(ds.activity_native_cell_equivalent.values/1e6,coord=ds.lon.values)
    im=mapax.pcolormesh(lon,ds.lat,arr,transform=PC,cmap=HEAT,vmin=0,vmax=40,rasterized=True,shading='auto')
    mapax.set_global();mapax.add_feature(cfeature.LAND.with_scale('110m'),facecolor='#D6D6D6',edgecolor='none',zorder=3)
    mapax.coastlines(resolution='110m',lw=.35,color='#555555');mapax.spines['geo'].set_linewidth(.6)
    segments={}
    for s,n,w,e in BASINS.values():
        for lat in [s,n]:segments.setdefault(('h',lat),[]).append((w,e))
        for lon0 in [w,e]:segments.setdefault(('v',lon0),[]).append((s,n))
    for (direction,fixed),intervals in segments.items():
        bounds=sorted(set(v for pair in intervals for v in pair))
        for start,end in zip(bounds[:-1],bounds[1:]):
            if not any(a<=(start+end)/2<=b for a,b in intervals):continue
            coords=np.linspace(start,end,max(3,int(end-start)+1));xx,yy=(coords,np.full_like(coords,fixed)) if direction=='h' else (np.full_like(coords,fixed),coords)
            mapax.plot(xx,yy,transform=PC,lw=.7,color='#343434',ls=(0,(4,3)),zorder=4)
    anchors={'North Pacific':(210,35),'Tropical Atlantic':(320,6),'North Atlantic':(335,40),'South Pacific':(205,-30),'South Atlantic':(330,-35),'Indian Ocean':(80,-30)}
    offsets={'North Pacific':(5,-10),'Tropical Atlantic':(14,-8),'North Atlantic':(24,-3),'South Pacific':(15,-4),'South Atlantic':(-16,-7),'Indian Ocean':(-15,-6)}
    for basin,(lo,la) in anchors.items():
        mapax.scatter(lo,la,transform=PC,color=BLACK,edgecolor='white',s=20,lw=.7,zorder=7);dx,dy=offsets[basin]
        mapax.text(lo+dx,la+dy,basin.replace(' ','\n',1),transform=PC,ha='center',va='center',fontsize=7,fontweight='bold',zorder=8,path_effects=[pe.withStroke(linewidth=2,foreground='white')])
        ax=axes[basin];xy=proj.transform_point(lo,la,PC)
        con=ConnectionPatch(xyA=(.5,-.22 if basin in top else 1.29),coordsA=ax.transAxes,xyB=xy,coordsB=mapax.transData,color='#444444',lw=.8,zorder=6,shrinkB=4)
        con.set_path_effects([pe.withStroke(linewidth=2,foreground='white')]);fig.add_artist(con)
    cax=fig.add_axes([.085,.052,.46,.012]);cb=fig.colorbar(im,cax=cax,orientation='horizontal',extend='max',ticks=[0,10,20,30,40])
    cb.ax.tick_params(labelsize=6.2,length=2);cb.set_label('Monthly MHW activity (10⁶ °C days km²)',fontsize=7,labelpad=2);cb.outline.set_linewidth(.4)
    handles=[Patch(color=GREEN,label='Local memory'),Patch(color=ORANGE,label='ENSO'),Patch(color=BLUE,label='Other modes'),Patch(facecolor='#AAAAAA',edgecolor='#777777',hatch='///',label='Unresolved')]
    fig.legend(handles=handles,loc='center',bbox_to_anchor=(.78,.053),ncol=2,fontsize=6.3,handlelength=1.1,handletextpad=.4,columnspacing=1.3)
    REPORT['Figure3_scope']={'background':'monthly area-integrated MHW activity','association_bar_values_unchanged':True,
                            'daily_event_metric':False,'map_limit_million':40}
    save(fig,'Figure3')

if __name__=='__main__':
    figure1()
    figure2()
    (ROOT/'render_validation.json').write_text(json.dumps(REPORT,indent=2)+'\n')
    print(json.dumps(REPORT,indent=2))
