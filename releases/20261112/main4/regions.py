"""Disjoint ocean domains from NOAA WOA98 surface basin identity."""
from pathlib import Path
import numpy as np
import xarray as xr
ROOT=Path(__file__).resolve().parent
ORDER=['TP','NP','SP','TA','NA','SA','IO']
NAMES=['Tropical Pacific','North Pacific','South Pacific','Tropical Atlantic','North Atlantic','South Atlantic','Indian Ocean']
# Atmospheric circulation domains include land and are not ocean-statistic masks.
ATM={'NP':(20,60,120,260),'SP':(-60,-20,150,290),
     'TA':(-20,20,300,360),'NA':(20,60,280,360),
     'SA':(-60,-20,300,380),'IO':(-60,20,40,120)}
def masks(lat,lon):
    raw=xr.open_dataset(ROOT/'cache/woa98_basin.nc',decode_times=False).basin.isel(time=0,level=0,drop=True).sortby('lat')
    raw=xr.concat([raw.assign_coords(lon=raw.lon-360),raw,raw.assign_coords(lon=raw.lon+360)],dim='lon')
    b=raw.sel(lat=xr.DataArray(np.asarray(lat),dims='lat'),lon=xr.DataArray(np.asarray(lon)%360,dims='lon'),method='nearest').values
    yy,xx=np.meshgrid(lat,np.asarray(lon)%360,indexing='ij')
    # Southern Ocean partitions follow the NOAA WOA major-basin convention.
    pac=(b==2)|(b==12)|((b==10)&(xx>=147)&(xx<297))
    atl=(b==1)|((b==10)&((xx>=297)|(xx<20)))
    ind=(b==3)|(b==56)|((b==10)&(xx>=20)&(xx<147))
    tropical=(yy>=-23.5)&(yy<23.5);north=(yy>=23.5)&(yy<=60);south=(yy>=-60)&(yy<-23.5)
    out=np.stack([pac&tropical,pac&north,pac&south,atl&tropical,atl&north,atl&south,ind&(yy>=-60)&(yy<=60)])
    assert out.sum(axis=0).max()<=1
    return xr.DataArray(out,dims=('region','lat','lon'),coords={'region':ORDER,'lat':lat,'lon':lon},
        attrs={'source':'NOAA WOA98 surface basin mask','source_url':'https://psl.noaa.gov/thredds/fileServer/Datasets/nodc.woa98/masks/basin.nc',
        'latitude_edges':'-23.5 belongs tropical; +23.5 belongs north; +/-60 included; grid-centre membership',
        'marginal_seas':'Japan Sea included Pacific; Bay of Bengal included Indian; separately coded enclosed seas excluded',
        'southern_ocean':'Atlantic 297E-20E; Indian 20E-147E; Pacific 147E-297E'})

def weighted_r(a,b,lat,mask=None):
    a=np.asarray(a);b=np.asarray(b);w=np.broadcast_to(np.cos(np.deg2rad(np.asarray(lat)))[:,None],a.shape)
    valid=np.isfinite(a)&np.isfinite(b)
    if mask is not None:valid &= mask
    if valid.sum()<4:return np.nan
    av=a[valid];bv=b[valid];wv=w[valid];wv=wv/wv.sum()
    av=av-np.sum(av*wv);bv=bv-np.sum(bv*wv)
    return float(np.sum(wv*av*bv)/np.sqrt(np.sum(wv*av*av)*np.sum(wv*bv*bv)))

def atm_mask(region,lat,lon):
    s,n,w,e=ATM[region];yy,xx=np.meshgrid(lat,lon,indexing='ij')
    xx=np.where(xx<w,xx+360,xx)
    return (yy>=s)&(yy<=n)&(xx>=w)&(xx<=e)
