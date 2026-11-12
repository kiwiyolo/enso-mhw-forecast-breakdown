"""A uniform global benchmark applied to preserved local ROC AUC, not a local expectation."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / 'Figure1_neighbourhood_skill_20261111_v10'
BASE = Path('/data3/luoq/FEIS/research/papers/mhw-predictability/Figures')
DATA = ROOT / 'source_data'
REG = ['TP', 'NP', 'SP', 'TA', 'NA', 'SA', 'IO']
sources = [PREV/'source_data/event_lead_fields.npz',
           PREV/'source_data/canonical_relative_changes.nc',
           BASE/'Figure1-3_Nature_layout_20261110_v7/source_data/Fig1/Figure1_source_events.csv',
           BASE/'Design_20261029_v7/source_data/Figure1_statistics.json',
           PREV/'contract.json', PREV/'selected_regions.json']
manifest = [dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]
shutil.copy2(sources[2], DATA/'Figure1_source_events.csv')
events = pd.read_csv(sources[2])
target = events.loc[events.Event == '2023/24'].iloc[0]
hist = events[events.Event != '2023/24']
assert len(hist) == 9
fit = linregress(hist.Peak_Nino34, hist.Mean_AUC)
expected = float(fit.intercept + fit.slope * target.Peak_Nino34)
archived = json.loads(sources[3].read_text())
assert abs(expected-archived['expected']) < 1e-12
ds = xr.load_dataset(sources[1])
z = np.load(sources[0])
assert list(z['events']) == ['1997/98', '2015/16', '2023/24']
keep = ds.auc_common.values.astype(bool)
assert np.array_equal(keep, z['auc_common'])
assert np.isfinite(z['auc'][:,:,keep]).all()
local = np.where(keep[None,:], z['auc'][2], np.nan)
shortfall = expected-local
mean_local = local.mean(axis=0)
mean_shortfall = shortfall.mean(axis=0)
assert np.array_equal(np.isfinite(mean_shortfall), np.isfinite(ds.auc_decrease.values))
assert np.allclose(mean_shortfall[keep], expected-mean_local[keep], atol=1e-12)
# Independently verify what changed relative to the original canonical reference.
canonical = z['auc'][:2].mean(axis=(0,1))
assert np.allclose(mean_shortfall[keep]-ds.auc_decrease.values[keep],
                   expected-canonical[keep], atol=1e-12)
result = ds[['auc_common', 'region', 'area_weight']].copy()
result['local_auc_2023'] = ('cell', mean_local)
result['global_benchmark_shortfall'] = ('cell', mean_shortfall)
result['global_benchmark_shortfall'].attrs.update(
    definition='Historical global expected AUC minus local 2023/24 AUC',
    interpretation='Uniform global reference, not a locally fitted expected skill or causal loss')
result.attrs.update(global_expected_auc=expected, local_window='2023-04 to 2024-09',
                    global_fit_window='July-June for each event', comparison_status='Exploratory; different estimands')
result.to_netcdf(DATA/'global_expected_shortfall.nc')
np.savez_compressed(DATA/'local_auc_and_shortfall_by_lead.npz',
                    local_auc=local, shortfall=shortfall, lead=np.arange(1,10), common=keep)
rows, lead_rows = [], []
for i, region in enumerate(REG):
    whole = ds.region.values == i
    m = whole & keep
    w = ds.area_weight.values[m]
    lv = np.average(shortfall[:,m], axis=1, weights=w)
    rows.append(dict(metric='AUC', region=region, global_benchmark=expected,
                     target=float(np.average(mean_local[m], weights=w)), change=float(lv.mean()),
                     lead_min=float(lv.min()), lead_max=float(lv.max()),
                     coverage=float(w.sum()/ds.area_weight.values[whole].sum()), cells=int(m.sum())))
    lead_rows.extend(dict(metric='AUC', region=region, lead=j+1, change=float(v)) for j,v in enumerate(lv))
pd.DataFrame(rows).to_csv(DATA/'regional_changes.csv', index=False)
pd.DataFrame(lead_rows).to_csv(DATA/'regional_changes_by_lead.csv', index=False)
stats = dict(expected=expected, target_peak_nino34=float(target.Peak_Nino34),
             historical_events=len(hist), slope=float(fit.slope), intercept=float(fit.intercept),
             historical_r=float(fit.rvalue), historical_p=float(fit.pvalue),
             observed_global_auc=float(target.Mean_AUC),
             unchanged_global_gap=expected-float(target.Mean_AUC),
             common_cells=int(keep.sum()), domain_cells=int(keep.size),
             common_area_fraction=float(ds.area_weight.values[keep].sum()/ds.area_weight.values.sum()),
             positive_shortfall_cells=int((mean_shortfall>0).sum()),
             negative_shortfall_cells=int((mean_shortfall<0).sum()),
             min_shortfall=float(np.nanmin(mean_shortfall)), max_shortfall=float(np.nanmax(mean_shortfall)),
             regional_results=rows)
(ROOT/'statistics.json').write_text(json.dumps(stats, indent=2)+'\n')
(ROOT/'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
contract = json.loads((PREV/'contract.json').read_text())
contract.update(status='New exploratory figure; old Fig1-4 untouched',
                goal='Apply the Fig1a historical global expectation as a uniform benchmark for the Fig1c local field',
                difference='Global expected AUC minus 2023/24 local AUC, equally averaged across leads 1-9',
                reference_value=expected,
                interpretation_limits=['Not a locally calibrated expected AUC',
                  'Global regression uses July-June monthly spatial AUC; local field uses April-September pooled local AUC',
                  'Not an additive decomposition of global AUC or its gap',
                  'Uniform reference adds no spatial information beyond reversing local AUC ordering',
                  'Fig4a activity spans May2023-Aug2024, differs from this field',
                  'Fig1c basin splits remain 20 degrees; Fig4 uses 23.5 degrees',
                  'No spatial correlation with Fig4a estimated here'])
(ROOT/'contract.json').write_text(json.dumps(contract, indent=2)+'\n')
(ROOT/'qa/calculation_validation.json').write_text(json.dumps(dict(
    expected_independently_reproduced=True, original_mask_unchanged=True,
    original_local_auc_unchanged=True, subtraction_identity_verified=True,
    difference_from_previous_reference_verified=True, original_regions_unchanged=True,
    positive_and_negative_values_retained=True, confidence_interval_not_claimed=True), indent=2)+'\n')
print(json.dumps(stats, indent=2))
