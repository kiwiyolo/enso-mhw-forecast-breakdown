from pathlib import Path
import hashlib
import json
import numpy as np
import xarray as xr
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'figure3_fig4_basin_masks_20261031_v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

snapshot=ROOT/'baseline_snapshot.json'
if not snapshot.exists():
    targets=[BASE,ROOT.parent/'figure3_circulation_evidence_20261102_v5']
    files=[p for directory in targets for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    snapshot.write_text(json.dumps([{'path':str(p),'sha256':sha(p)} for p in files],indent=2)+'\n')
old=xr.load_dataset(BASE/'source_data/seven_ocean_regions.nc').mask
pac=old.sel(region=['TP','NP','SP']).any('region')
atl=old.sel(region=['TA','NA','SA']).any('region')
lat=old.lat
bands={'T':(lat>=-23.5)&(lat<23.5),'N':lat>=23.5,'S':lat<-23.5}
arrays=[pac&bands['T'],pac&bands['N'],pac&bands['S'],atl&bands['T'],atl&bands['N'],atl&bands['S'],old.sel(region='IO',drop=True)]
new=xr.concat(arrays,dim=xr.IndexVariable('region',['TP','NP','SP','TA','NA','SA','IO'])).transpose('region','lat','lon')
assert bool((new.sum('region')<=1).all())
assert bool((new.any('region')==old.any('region')).all())
assert bool((new.sel(region='IO')==old.sel(region='IO')).all())
ds=xr.Dataset({'mask':new},attrs={'tropical_boundary_deg':23.5,'major_basin_provenance':str(BASE/'source_data/seven_ocean_regions.nc'),
     'grid_rule':'Grid-centre membership; tropical rows -23 through +23; no fractional cells at +/-23.5',
     'figure4_sync':'Major-basin identity retained; existing Fig4 latitude bands not yet updated.'})
ds.to_netcdf(ROOT/'source_data/seven_ocean_regions.nc')
pd.DataFrame([{'Region':str(r),'Old_cells':int(old.sel(region=r).sum()),'New_cells':int(new.sel(region=r).sum()),
               'Changed_cells':int((new.sel(region=r)!=old.sel(region=r)).sum())} for r in old.region.values]).to_csv(ROOT/'source_data/region_boundary_change.csv',index=False)
print(ds.attrs)
print(pd.read_csv(ROOT/'source_data/region_boundary_change.csv',keep_default_na=False).to_string(index=False))
